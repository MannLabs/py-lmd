from __future__ import annotations

import csv
import multiprocessing as mp
import os
import platform
import sys
import warnings
from functools import partial, reduce
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
from scipy.spatial import cKDTree
from tqdm.auto import tqdm

from lmd.path import calc_len, tsp_greedy_solve, tsp_hilbert_solve
from lmd.segmentation import _create_coord_index_sparse, get_coordinate_form

from ._geom import Collection
from ._utils import _create_poly, _execute_indexed_parallel, transform_to_map


class SegmentationLoader:
    """Select single cells from a segmentation and generate cutting data

        Args:
            config (dict): Dict containing configuration parameters. See Note for further explanation.

            processes (int): Number of processes used for parallel processing of cell sets. Total processes can be calculated as `processes * threads`.

            threads (int): Number of threads used for parallel processing of shapes within a cell set. Total processes can be calculated as `processes * threads`.

            cell_sets (list(dict)): List of dictionaries containing the sets of cells which should be sorted into a single well.

            calibration_points (np.array): Array of size '(3,2)' containing the calibration marker coordinates in the '(row, column)' format.

            coords_lookup (None, dict): precalculated lookup table for coordinates of individual cell ids. If not provided will be calculated.

            classes (np.array): Array of classes found in the provided segmentation mask. If not provided will be calculated based on the assumption that cell_ids are assigned in ascending order.

        Example:

            .. code-block:: python

                import numpy as np
                from PIL import Image
                from lmd.lib import SegmentationLoader

    Example:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from lmd.lib import SegmentationLoader
            from lmd._utils import _download_segmentation_example_file

            # use example image provided within py-lmd
            example_image_path = _download_segmentation_example_file()
            im = Image.open(example_image_path)
            segmentation = np.array(im).astype(np.uint32)

            all_classes = np.unique(segmentation)

            cell_sets = [{"classes": all_classes, "well": "A1"}]

            calibration_points = np.array([[0, 0], [0, 1000], [1000, 1000]])

            loader_config = {"orientation_transform": np.array([[0, -1], [1, 0]])}

            sl = SegmentationLoader(config=loader_config)
            shape_collection = sl(segmentation, cell_sets, calibration_points)

            shape_collection.plot(fig_size=(10, 10))

        .. image:: images/segmentation1.png

    Note:

        Basic explanation of the parameters in the config dict:

        .. code-block:: yaml

            # dilation of the cutting mask in pixel before intersecting shapes in a selection group are merged
            shape_dilation: 0

            # erosion of the cutting mask in pixel before intersecting shapes in a selection group are merged
            shape_erosion: 0

            # Cutting masks are transformed by binary dilation and erosion
            binary_smoothing: 3

            # number of datapoints which are averaged for smoothing
            # the resoltion of datapoints is twice as high as the resolution of pixel
            convolution_smoothing: 15

            # strength of coordinate reduction through the Ramer-Douglas-Peucker algorithm 0 is small 1 is very high
            rdp_epsilon: 0.1

            # Optimization of the cutting path inbetween shapes
            # optimized paths improve the cutting time and the microscopes focus
            # valid options are ["none", "hilbert", "greedy"]
            path_optimization: "hilbert"

            # Parameter required for hilbert curve based path optimization.
            # Defines the order of the hilbert curve used, which needs to be tuned with the total cutting area.
            # For areas of 1 x 1 mm we recommend at least p = 4,  for whole slides we recommend p = 7.
            hilbert_p: 7

            # Parameter required for greedy path optimization.
            # Instead of a global distance matrix, the k nearest neighbours are approximated.
            # The optimization problem is then greedily solved for the known set of nearest neighbours until the first set of neighbours is exhausted.
            # Established edges are then removed and the nearest neighbour approximation is recursively repeated.
            greedy_k: 20

            # Overlapping shapes are merged based on a nearest neighbour heuristic.
            # All selected shapes closer than distance_heuristic pixel are checked for overlap.
            distance_heuristic: 300


    """

    # define all valid path optimization methods used with the "path_optimization" argument in the configuration
    VALID_PATH_OPTIMIZERS = ["none", "hilbert", "greedy"]
    DEFAULT_SEGMENTATION_DTYPE = np.uint64

    def __init__(self, config=None, verbose=False, processes=1):
        if config is None:
            config = {}
        self.config = config
        self.verbose = verbose
        self._get_context()  # setup context for multiprocessing function calls to work with different operating systems

        self.register_parameter("shape_dilation", 0)
        self.register_parameter("shape_erosion", 0)
        self.register_parameter("binary_smoothing", 3)
        self.register_parameter("convolution_smoothing", 15)
        self.register_parameter("rdp_epsilon", 0.1)
        self.register_parameter("path_optimization", "hilbert")
        self.register_parameter("greedy_k", 0)
        self.register_parameter("hilbert_p", 7)
        self.register_parameter("xml_decimal_transform", 100)
        self.register_parameter("distance_heuristic", 300)
        self.register_parameter("join_intersecting", True)
        self.register_parameter("orientation_transform", np.eye(2))
        self.register_parameter("threads", 10)

        self.coords_lookup = None
        self.processes = processes

        self._configure_path_optimizer()

    def _configure_path_optimizer(self):
        # configure path optimizer
        if "path_optimization" in self.config:
            optimization_method = self.config["path_optimization"]
        else:
            optimization_method = "none"

            # check if the optimizer is a valid option
        if optimization_method in self.VALID_PATH_OPTIMIZERS:
            pathoptimizer = optimization_method
        else:
            self.log("Path optimizer is no valid option, no optimization will be used.")
            pathoptimizer = "none"

        self.log(f"Path optimizer used for XML generation: {optimization_method}")
        self.optimization_method = pathoptimizer

    def _get_context(self):
        if platform.system() == "Windows":
            self.context = "spawn"
        elif platform.system() == "Darwin":
            self.context = "spawn"
        elif platform.system() == "Linux":
            self.context = "fork"

    def __call__(
        self,
        input_segmentation: np.ndarray | None,
        cell_sets,
        calibration_points,
        coords_lookup=None,
        classes=np.array([], dtype=np.uint64),
    ):
        if input_segmentation is None:
            assert coords_lookup is not None, "If no input segmentation is provided, a coords_lookup must be provided."

        self.calibration_points = calibration_points
        sets = []

        # iterate over all defined sets, perform sanity checks and load external data
        for i, cell_set in enumerate(cell_sets):
            self.check_cell_set_sanity(cell_set)
            cell_set["classes_loaded"] = self.load_classes(cell_set)
            sets.append(cell_set)
            self.log(f"cell set {i} passed sanity check")

        if len(sets) < self.processes:
            self.processes = len(sets)  # reduce number of processes if there are less cell sets than processes

        self.input_segmentation = input_segmentation

        if coords_lookup is None:
            self.log("Calculating coordinate locations of all cells.")
            # deprecated infavour of more computationally efficient solution
            # self.coords_lookup = _create_coord_index(self.input_segmentation, classes = classes)
            # self.coords_lookup = {k: np.array(v) for k, v in self.coords_lookup.items()}
            self.coords_lookup = _create_coord_index_sparse(self.input_segmentation)
        else:
            self.log("Loading coordinates from external source")
            self.coords_lookup = coords_lookup

        # try multithreading
        if self.processes > 1:
            self.multi_threading = True
            self.log("Processing cell sets in parallel")
            args = []
            for i, cell_set in enumerate(cell_sets):
                args.append((i, cell_set))

            collections = _execute_indexed_parallel(
                self.generate_cutting_data,
                args=args,
                tqdm_args={
                    "file": sys.stdout,
                    "disable": not self.verbose,
                    "desc": "collecting cell sets",
                },
                n_threads=self.processes,
            )
        else:
            print("Processing cell sets in serial")
            self.multi_threading = False
            collections = []
            for i, cell_set in enumerate(cell_sets):
                collections.append(self.generate_cutting_data(i, cell_set))

        return reduce(lambda a, b: a.join(b), collections)

    def generate_cutting_data(self, i: int, cell_set: dict) -> Collection:
        if 0 in cell_set["classes_loaded"]:
            cell_set["classes_loaded"] = cell_set["classes_loaded"][cell_set["classes_loaded"] != 0]
            warnings.warn("Class 0 is not a valid class and was removed from the cell set", stacklevel=2)

        self.log("Convert label format into coordinate format")
        center, length, coords = get_coordinate_form(cell_set["classes_loaded"], self.coords_lookup)

        self.log("Conversion finished, performing sanity check.")

        # Sanity check 1
        if len(center) == len(cell_set["classes_loaded"]):
            pass
        else:
            self.log(
                "Check failed, returned lengths do not match cell set.\n Some classes were not found in the segmentation and were therefore removed.\n Please make sure all classes specified are present in your segmentation."
            )
            elements_removed = len(cell_set["classes_loaded"]) - len(center)
            self.log(f"{elements_removed} classes were not found and therefore removed.")

        # Sanity check 2: for the returned coordinates
        if len(center) == len(length):
            pass
        else:
            self.log(
                "Check failed, returned lengths do not match. Please check if all classes specified are present in your segmentation"
            )

        # Sanity check 3
        zero_elements = 0
        for el in coords:
            if len(el) == 0:
                zero_elements += 1

        if zero_elements <= 2:  # allow at most for 2 zero elements (x = 0 and y = 0)
            pass
        else:
            self.log(
                "Check failed, returned coordinates contain empty elements. Please check if all classes specified are present in your segmentation"
            )

        if self.config["join_intersecting"]:
            center, length, coords = self.merge_dilated_shapes(
                center, length, coords, dilation=self.config["shape_dilation"], erosion=self.config["shape_erosion"]
            )

        # Calculate dilation and erosion based on if merging was activated
        dilation = (
            self.config["binary_smoothing"]
            if self.config["join_intersecting"]
            else self.config["binary_smoothing"] + self.config["shape_dilation"]
        )
        erosion = (
            self.config["binary_smoothing"]
            if self.config["join_intersecting"]
            else self.config["binary_smoothing"] + self.config["shape_erosion"]
        )

        if self.config["threads"] == 1:
            shapes = []
            for coord in tqdm(coords, desc="creating shapes"):
                shapes.append(transform_to_map(coord, dilation=dilation, erosion=erosion, coord_format=False))
        else:
            with mp.get_context(self.context).Pool(processes=self.config["threads"]) as pool:
                shapes = list(
                    tqdm(
                        pool.imap(
                            partial(
                                transform_to_map,
                                erosion=erosion,
                                dilation=dilation,
                                coord_format=False,
                            ),
                            coords,
                        ),
                        total=len(center),
                        disable=not self.verbose,
                        desc="creating shapes",
                    )
                )

        if self.config["threads"] == 1:
            polygons = []
            for shape in tqdm(shapes, desc="calculating polygons"):
                polygons.append(
                    _create_poly(
                        shape,
                        smoothing_filter_size=self.config["convolution_smoothing"],
                        rdp_epsilon=self.config["rdp_epsilon"],
                    )
                )
        else:
            with mp.get_context(self.context).Pool(processes=self.config["threads"]) as pool:
                polygons = list(
                    tqdm(
                        pool.imap(
                            partial(
                                _create_poly,
                                smoothing_filter_size=self.config["convolution_smoothing"],
                                rdp_epsilon=self.config["rdp_epsilon"],
                            ),
                            shapes,
                        ),
                        total=len(center),
                        disable=not self.verbose,
                        desc="calculating polygons",
                    )
                )

        # perform path optimization to minimize the total distance that the LMD travels during cutting (this improves cutting speed and focus)
        center = np.array(center)
        unoptimized_length = calc_len(center)
        self.log(f"Current path length: {unoptimized_length:,.2f} units")

        if self.optimization_method != "none":
            if self.optimization_method == "greedy":
                optimized_idx = tsp_greedy_solve(center, k=self.config["greedy_k"])

            elif self.optimization_method == "hilbert":
                optimized_idx = tsp_hilbert_solve(center, p=self.config["hilbert_p"])

            # update order of centers
            center = center[optimized_idx]
            self.indexes = optimized_idx

            # calculate optimized path length and optimization factor
            optimized_length = calc_len(center)
            self.log(f"Optimized path length: {optimized_length:,.2f} units")

            # TODO: Remove unused variable optimization_factor
            optimization_factor = unoptimized_length / optimized_length
            self.log(f"Optimization factor: {optimization_factor:,.1f}x")
        else:
            self.log("No path optimization used")
            optimization_factor = 1
            optimized_idx = list(range(len(center)))
        # order list of shapes by the optimized index array
        polygons = [x for _, x in sorted(zip(optimized_idx, polygons))]

        # Plot coordinates if in debug mode
        if self.verbose:
            fig, axs = plt.subplots(1, 1, figsize=(10, 10))

            if "background_image" in self.config:
                axs.imshow(self.config["background_image"])

            axs.scatter(center[:, 1], center[:, 0], s=1)

            for shape in polygons:
                axs.plot(shape[:, 1], shape[:, 0], color="red", linewidth=1)

            axs.scatter(
                self.calibration_points[:, 1],
                self.calibration_points[:, 0],
                color="blue",
            )
            axs.plot(center[:, 1], center[:, 0], color="grey")
            axs.invert_yaxis()
            axs.set_aspect("equal", adjustable="box")
            axs.axis("off")
            axs.set_title("Final cutting path")
            fig.tight_layout()

            if self.multi_threading:
                self.log("Plotting shapes in debug mode is not supported in multi-threading mode.")
                self.log("Saving plots to disk instead.")
                fig.savefig(f"debug_plot_{i}.png")
                plt.close(fig)
            else:
                plt.show(fig)

        # Generate array of marker cross positions
        ds = Collection(calibration_points=self.calibration_points, scale=self.config["xml_decimal_transform"])
        ds.orientation_transform = self.config["orientation_transform"]

        for shape in polygons:
            # Check if well key is set in cell set definition
            if "well" in cell_set:
                ds.new_shape(shape, well=cell_set["well"])
            else:
                ds.new_shape(shape)
        return ds

    def merge_dilated_shapes(self, input_center, input_length, input_coords, dilation=0, erosion=0):
        print("Intersecting Shapes will be merged into a single shape.")

        # initialize all shapes and create dilated coordinates
        # coordinates are created as complex numbers to facilitate comparison with np.isin
        dilated_coords = []

        if self.config["threads"] == 1:
            for coord in tqdm(input_coords, desc="dilating shapes"):
                dilated_coords.append(transform_to_map(coord, dilation=dilation))

        else:
            with mp.get_context(self.context).Pool(processes=self.config["threads"]) as pool:
                dilated_coords = list(
                    tqdm(
                        pool.imap(partial(transform_to_map, dilation=dilation), input_coords),
                        total=len(input_center),
                        desc="dilating shapes",
                    )
                )

        dilated_coords = [np.apply_along_axis(lambda args: [complex(*args)], 1, d).flatten() for d in dilated_coords]

        # A sparse distance matrix is calculated for all cells which are closer than distance_heuristic
        center_arr = np.array(input_center)
        center_tree = cKDTree(center_arr)

        sparse_distance = center_tree.sparse_distance_matrix(center_tree, self.config["distance_heuristic"])
        sparse_distance = scipy.sparse.tril(sparse_distance)

        # sparse intersection matrix is calculated based on the sparse distance matrix
        intersect_data = []
        for col, row in zip(sparse_distance.col, sparse_distance.row):
            # diagonal entries are known to intersect
            if col == row:
                intersect_data.append(1)
            else:
                # np.isin is used with the two complex coordinate arrays
                do_intersect = np.isin(dilated_coords[col], dilated_coords[row]).any()
                # intersect_data uses the same columns and rows as sparse_distance
                # if two shapes intersect, an edge is created otherwise, no edge is created.
                # zero entries will be dropped later
                intersect_data.append(1 if do_intersect else 0)

        # create sparse intersection matrix and drop zero elements
        sparse_intersect = scipy.sparse.coo_matrix((intersect_data, (sparse_distance.row, sparse_distance.col)))
        sparse_intersect.eliminate_zeros()

        # create networkx graph from sparse intersection matrix
        g = nx.from_scipy_sparse_array(sparse_intersect)

        # find unconnected subgraphs
        # to_merge contains a list of lists with indexes pointing to shapes to be merged
        to_merge = [list(g.subgraph(c).nodes()) for c in nx.connected_components(g)]

        output_center = []
        output_length = []
        output_coords = []

        # merge coords
        for new_shape in to_merge:
            coords = []

            for idx in new_shape:
                coords.append(dilated_coords[idx])

            coords_complex = np.concatenate(coords)
            coords_complex = np.unique(coords_complex)
            coords_2d = np.array([coords_complex.real, coords_complex.imag], dtype=int).T

            # calculate properties length and center from coords
            new_center = np.mean(coords_2d, axis=0)
            new_len = len(coords_2d)

            # append values to output lists
            output_center.append(new_center)
            output_length.append(new_len)
            output_coords.append(coords_2d)

        print(
            len(to_merge) - len(output_center),
            "shapes that were intersecting were found and merged.",
        )
        return output_center, output_length, output_coords

    def check_cell_set_sanity(self, cell_set):
        """Check if cell_set dictionary contains the right keys"""

        if "classes" in cell_set:
            if not isinstance(cell_set["classes"], (list, str, np.ndarray)):
                self.log("No list of classes specified for cell set")
                raise TypeError("No list of classes specified for cell set")
        else:
            self.log("No classes specified for cell set")
            raise KeyError("No classes specified for cell set")

        if "well" in cell_set:
            if not isinstance(cell_set["well"], str):
                self.log("No well of type str specified for cell set")
                raise TypeError("No well of type str specified for cell set")

    def load_classes(self, cell_set):
        """Identify cell class definition and load classes

        Identify if cell classes are provided as list of integers or as path pointing to a csv file.
        Depending on the type of the cell set, the classes are loaded and returned for selection.
        """
        if isinstance(cell_set["classes"], list):
            return cell_set["classes"]

        if isinstance(cell_set["classes"], np.ndarray):
            if np.issubdtype(cell_set["classes"].dtype.type, np.integer):
                return cell_set["classes"]

        if isinstance(cell_set["classes"], str):
            # If the path is relative, it is interpreted relative to the project directory
            if os.path.isabs(cell_set["classes"]):
                path = cell_set["classes"]
            else:
                path = os.path.join(Path(self.directory).parents[0], cell_set["classes"])

            # TODO: Close file again https://github.com/MannLabs/py-lmd/pull/61#discussion_r2692370486
            if os.path.isfile(path):
                try:
                    cr = csv.reader(open(path))
                    filtered_classes = np.array([int(el[0]) for el in list(cr)], dtype="int64")
                    self.log(f"Loaded {len(filtered_classes)} classes from csv")
                    return filtered_classes
                except (ValueError, TypeError) as e:
                    self.log(f"CSV file could not be converted to list of integers: {path}")
                    raise ValueError() from e
            else:
                self.log("Path containing classes could not be read: {path}")
                raise ValueError()
        else:
            self.log(
                "classes argument for a cell set needs to be a list of integer ids or a path pointing to a csv of integer ids."
            )
            raise TypeError(
                "classes argument for a cell set needs to be a list of integer ids or a path pointing to a csv of integer ids."
            )

    def log(self, msg):
        if self.verbose:
            print(msg)

    def register_parameter(self, key, value):
        if isinstance(key, str):
            config_handle = self.config

        elif isinstance(key, list):
            raise NotImplementedError("registration of parameters is not yet supported for nested parameters")

        else:
            raise TypeError("Key must be of string or a list of strings")

        if key not in config_handle:
            self.log(f"No configuration for {key} found, parameter will be set to {value}")
            config_handle[key] = value
