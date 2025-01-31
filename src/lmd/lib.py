from __future__ import annotations

import csv
import gc
import multiprocessing as mp
import os
import platform
import re
import sys

# import warnings
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial, reduce

import numpy as np
import matplotlib.pyplot as plt
from lxml import etree as ET
from matplotlib import image
from skimage import data, color
import matplotlib.ticker as ticker
from svgelements import SVG
from lmd.segmentation import get_coordinate_form, tsp_greedy_solve, tsp_hilbert_solve, calc_len, _create_coord_index,_create_coord_index_sparse, _filter_coord_index
from tqdm import tqdm
# import warnings
import warnings
from rdp import rdp

from skimage.morphology import dilation as binary_dilation
from skimage.morphology import binary_erosion, disk
from skimage.segmentation import find_boundaries

from pathlib import Path
from typing import Callable, Optional, Union, Iterable

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import shapely
from lmd.segmentation import (
    _create_coord_index,
    _filter_coord_index,
    calc_len,
    get_coordinate_form,
    tsp_greedy_solve,
    tsp_hilbert_solve,
)
from lxml import etree as ET
from matplotlib import image
from scipy import ndimage
from scipy.signal import convolve2d
from scipy.spatial import cKDTree
from skimage import color, data
from skimage.morphology import binary_erosion, disk
from skimage.morphology import dilation as binary_dilation
from skimage.segmentation import find_boundaries
from svgelements import SVG
from tqdm import tqdm
from tqdm.auto import tqdm


def _execute_indexed_parallel(
    func: Callable, *, args: list, tqdm_args: dict = None, n_threads: int = 10
) -> list:
    """parallelization of function call with indexed arguments using ThreadPoolExecutor. Returns a list of results in the order of the input arguments.

    Args:
        func (Callable): _description_
        args (list): _description_
        tqdm_args (dict, optional): _description_. Defaults to None.
        n_threads (int, optional): _description_. Defaults to 10.

    Returns:
        list: containing the results of the function calls in the same order as the input arguments
    """
    if tqdm_args is None:
        tqdm_args = {"total": len(args)}
    elif "total" not in tqdm_args:
        tqdm_args["total"] = len(args)

    results = [None for _ in range(len(args))]
    with ProcessPoolExecutor(n_threads) as executor:
        with tqdm(**tqdm_args) as pbar:
            futures = {executor.submit(func, *arg): i for i, arg in enumerate(args)}
            for future in as_completed(futures):
                index = futures[future]
                results[index] = future.result()
                pbar.update(1)

    return results


class Collection:
    """Class which is used for creating shape collections for the Leica LMD6 & 7. Contains a coordinate system defined by calibration points and a collection of various shapes.

    Args:
        calibration_points: Calibration coordinates in the form of :math:`(3, 2)`.
        orientation_transform: defines transformations performed on the provided coordinate system prior to export as XML. Defaults to the identity matrix.

    Attributes:
        shapes (List[Shape]): Contains all shapes which are part of the collection.

        calibration_points (Optional[np.ndarray]): Calibration coordinates in the form of :math:`(3, 2)`.

        orientation_transform (np.ndarray): defines transformations performed on the provided coordinate system prior to export as XML. This orientation_transform is always applied to shapes when there is no individual orientation_transform provided.
    """

    def __init__(
        self,
        calibration_points: Optional[np.ndarray] = None,
        orientation_transform: Optional[np.ndarray] = None,
    ):
        self.shapes: list[Shape] = []

        self.calibration_points: Optional[np.ndarray] = calibration_points

        if orientation_transform is None:
            orientation_transform = np.eye(2)  # assign default value

        self.orientation_transform: np.ndarray = orientation_transform

        self.scale = 100

        self.global_coordinates = 1

    def stats(self):
        """Print statistics about the Collection in the form of:

        .. code-block::

            ===== Collection Stats =====
            Number of shapes: 208
            Number of vertices: 126,812
            ============================
            Mean vertices: 609.67
            Min vertices: 220.00
            5% percentile vertices: 380.20
            Median vertices: 594.00
            95% percentile vertices: 893.20
            Max vertices: 1,300.00

        """
        lengths = np.array([len(shape.points) for shape in self.shapes])

        num_shapes = len(self.shapes)
        num_vertices = np.sum(lengths)

        median_dp = np.median(lengths).astype(float)
        mean_dp = np.mean(lengths).astype(float)
        max_dp = np.max(lengths).astype(float)
        min_dp = np.min(lengths).astype(float)
        percentile_5 = np.percentile(lengths, 5).astype(float)
        percentile_95 = np.percentile(lengths, 95).astype(float)

        print('===== Collection Stats =====')
        print(f'Number of shapes: {num_shapes:,}')
        print(f'Number of vertices: {num_vertices:,}')
        print('============================')
        print(f'Mean vertices: {mean_dp:,.0f}')
        print(f'Min vertices: {min_dp:,.0f}')
        print(f'5% percentile vertices: {percentile_5:,.0f}')
        print(f'Median vertices: {median_dp:,.0f}')
        print(f'95% percentile vertices: {percentile_95:,.0f}')
        print(f'Max vertices: {max_dp:,.0f}')
        
    def plot(self, calibration: bool = True, 
             mode: str = "line", 
             fig_size: tuple = (5,5),
             apply_orientation_transform: bool = True,
             apply_scale: bool = False, 
             save_name: Optional[str] = None, 
             return_fig: bool = False,
             **kwargs):
        
        """This function can be used to plot all shapes of the corresponding shape collection.

        Args:
            calibration: Controls wether the calibration points should be plotted as crosshairs. Deactivating the crosshairs will result in the size of the canvas adapting to the shapes. Can be especially usefull for small shapes or debugging.

            fig_size: Defaults to :math:`(10, 10)` Controls the size of the matplotlib figure. See `matplotlib documentation <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.figure.html#matplotlib-pyplot-figure>`_ for more information.

            apply_orientation_transform: Define wether the orientation transform should be applied before plotting.

            save_name (Optional[str], default: None): Specify a filename  for saving the generated figure. By default `None` is provided which will not save a figure.
        """

        modes = ["line", "dots"]

        # Check if Collection scale should be applied or not
        if apply_scale:
            scale = self.scale
        else:
            scale = 1
            
        if mode not in modes:
            raise ValueError("Mode not known. Please use on of the following plotting modes: line, dots")
        
        #close current figures
        plt.clf()
        plt.cla()
        plt.close("all")

        fig, ax = plt.subplots(figsize=fig_size, **kwargs)

        # Plot calibration points
        if calibration and self.calibration_points is not None:
            # Apply orientation transform as default behavior
            if apply_orientation_transform:
                calibration = (
                    self.calibration_points @ self.orientation_transform * scale
                )
            else:
                calibration = self.calibration_points * scale

            plt.scatter(calibration[:, 0], calibration[:, 1], marker="x")

        for shape in self.shapes:
            # Apply orientation transform as default behavior
            if apply_orientation_transform:
                # Use local transform if defined, else use Collection transform
                if shape.orientation_transform is not None:
                    points = shape.points @ shape.orientation_transform * scale
                else:
                    points = shape.points @ self.orientation_transform * scale
            else:
                points = shape.points * scale

            if mode == "line":
                ax.plot(points[:, 0], points[:, 1])

            elif mode == "dots":
                ax.scatter(points[:, 0], points[:, 1], s=10)

        ax.grid(True)
        ax.ticklabel_format(useOffset=False)
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_aspect('equal', adjustable='box')
        
        fig.tight_layout()
        
        if save_name is not None:
            plt.savefig(save_name)
        
        if return_fig:
            return fig
    
        plt.show()

    def add_shape(self, shape: Shape):
        """Add a new shape to the collection.

        Args:
            shape: Shape which should be added.
        """

        if isinstance(shape, Shape):
            self.shapes.append(shape)
        else:
            TypeError("Provided shape is not of type Shape")

    def new_shape(
        self, points: np.ndarray, well: Optional[str] = None, name: Optional[str] = None
    ):
        """Directly create a new Shape in the current collection.

        Args:
            points: Array or list of lists in the shape of `(N,2)`. Contains the points of the polygon forming a shape.

            well: Well in which to sort the shape after cutting. For example A1, A2 or B3.

            name: Name of the shape.
        """

        to_add = Shape(
            points,
            well=well,
            name=name,
            orientation_transform=self.orientation_transform,
        )
        self.add_shape(to_add)

    def join(self, collection: Collection, update_orientation_transform: bool = True):
        """Join the collection with the shapes of a different collection. The calibration markers of the current collection are kept. Please keep in mind that coordinate systems and calibration points must be compatible for correct joining of collections.

        Args:
            collection: Collection which should be joined with the current collection object.
            orientation_transform: If set to True, the orientation transform of the joined collection will be updated to the current collection. If set to False, the orientation transform of the joined collection will not be updated.

        Returns:
            returns self
        """
        if not np.all(self.orientation_transform == collection.orientation_transform):
            if update_orientation_transform:
                shapes = collection.shapes
                for shape in shapes:
                    shape.orientation_transform = self.orientation_transform
                else:
                    Warning(
                        "Orientation transform of the joined collection is not equal to the current collection, but update_orientation_transform is set to False. Shapes will be merged without updating the orientation transform."
                    )
        self.shapes += collection.shapes

        return self

    def to_geopandas(self, *attrs: str) -> gpd.GeoDataFrame:
        """Return geopandas dataframe of collection

        Args:
            *attrs (str): Optional attributes of the shapes in the collection to be added as metadata columns

        Returns:
            geopandas.GeoDataFrame: Representation of all shapes and optional metadata

        Example:
        .. code-block:: python
            # Generate collection
            collection = pylmd.Collection()
            shape = pylmd.Shape(np.array([[ 0,  0], [ 0, -1], [ 1,  0], [ 0,  0]]), well="A1", name="Shape_1", orientation_transform=None)
            collection.add_shape(shape)

            # Get geopandas object
            collection.to_geopandas()
            >       geometry
                0   POLYGON ((0 0, 0 -1, 1 0, 0 0))

            collection.to_geopandas("well", "name")
            >   well     name                         geometry
                0   A1  Shape_1  POLYGON ((0 0, 0 -1, 1 0, 0 0))
        """
        metadata = (
            pd.DataFrame(
                [
                    [shape.__getattribute__(att) for att in attrs]
                    for shape in self.shapes
                ],
                columns=attrs,
            )
            if (attrs is not None)
            else None
        )
        geometry = [shape.to_shapely() for shape in self.shapes]

        return gpd.GeoDataFrame(data=metadata, geometry=geometry)
    
    # load xml from file
    def load(self, file_location: str):
        """Can be used to load a shape file from XML. Both, XMLs generated with py-lmd and the Leica software can be used.
        Args:
            file_location: File path pointing to the XML file.

        """

        tree = ET.parse(file_location)
        root = tree.getroot()

        cal_point_len = 0

        # count calibration points
        for child in root:
            if "CalibrationPoint" in child.tag:
                cal_point_len += 1

        self.calibration_points = np.ones((cal_point_len // 2, 2), dtype=int)

        for child in root:
            if child.tag == "GlobalCoordinates":
                self.global_coordinates = int(child.text)

            # Load calibration points
            elif "CalibrationPoint" in child.tag:
                axes = child.tag[0]
                axes_id = 0 if axes == "X" else 1
                shape_id = int(child.tag[-1]) - 1
                value = int(child.text)

                self.calibration_points[shape_id, axes_id] = value

            # Load shapes
            elif "Shape_" in child.tag:
                new_shape = Shape()
                new_shape.from_xml(child)
                self.shapes.append(new_shape)
    
    def load_geopandas(
            self, 
            gdf: gpd.GeoDataFrame, 
            geometry_column: str = "geometry",
            name_column: Optional[str] = None,
            well_column: Optional[str] = None,
            calibration_points: Optional[np.ndarray] = None, 
            global_coordinates: Optional[int] = None,
        ) -> None:
        """Create collection from a geopandas dataframe
        
        Args:
            gdf (geopandas.GeoDataFrame): Collection of shapes and optional metadata
            geometry_column (str, default: geometry): Name of column storing Shapes as `shapely.Polygon`, defaults to geometry
            well_column (str, optional): Column storing of well id as additional metadata
            calibration_points (np.ndarray, optional): Calibration points of collection 
            global_coordinates (int, optional): Number of global coordinates

        Example:

        ..  code-block:: python

            from lmd.lib import Collection
            import geopandas as gpd
            import shapely

            gdf = gpd.GeoDataFrame(
                data={"well": ["A1"], "name": ["test"]},
                geometry=[shapely.Polygon([[0, 0], [0, 1], [1, 0], [0, 0]])]
            )

            # Create collection
            c = Collection()

            # Export well metadata
            c.load_geopandas(gdf, well_column="well")
            assert c.to_geopandas("well").equals(gdf)

            # Do not export well metadata
            c.load_geopandas(gdf)
            assert c.to_geopandas().equals(gdf.drop(columns="well"))
        """
        # Update attributes
        if calibration_points is not None:
            self.calibration_points = calibration_points
        if global_coordinates is not None:
            self.global_coordinates = global_coordinates

        self.shapes = [
            Shape(
                points=np.array(row[geometry_column].exterior.coords), 
                name=row[name_column] if name_column is not None else None,
                well=row[well_column] if well_column is not None else None,
            )
            for _, row in gdf.iterrows()
        ]

    # save xml to file
    def save(self, file_location: str, encoding: str = "utf-8"):
        """Can be used to save the shape collection as XML file.

        file_location: File path pointing to the XML file.
        """

        root = ET.Element("ImageData")

        # write global coordinates
        global_coordinates = ET.SubElement(root, "GlobalCoordinates")
        global_coordinates.text = "1"

        # transform calibration points
        transformed_calibration_points = (
            self.calibration_points @ self.orientation_transform * self.scale
        )

        # write calibration points
        for i, point in enumerate(transformed_calibration_points):
            print(point)

            id = i + 1
            x = ET.SubElement(root, "X_CalibrationPoint_{}".format(id))
            x.text = "{}".format(np.floor(point[0]).astype(int))

            y = ET.SubElement(root, "Y_CalibrationPoint_{}".format(id))
            y.text = "{}".format(np.floor(point[1]).astype(int))

        # write shape length
        shape_count = ET.SubElement(root, "ShapeCount")
        shape_count.text = "{}".format(len(self.shapes))

        # write shapes
        for i, shape in enumerate(self.shapes):
            id = i + 1

            # apply Collection orientation_transform and scale
            root.append(shape.to_xml(id, self.orientation_transform, self.scale))

        # write root
        tree = ET.ElementTree(element=root)
        tree.write(
            file_location, encoding="utf-8", xml_declaration=True, pretty_print=True
        )

    def svg_to_lmd(
        self,
        file_location,
        offset=[0, 0],
        divisor=3,
        multiplier=60,
        rotation_matrix=np.eye(2),
        orientation_transform=None,
    ):
        """Can be used to save the shape collection as XML file.

        Args:
            file_location: File path pointing to the SVG file.

            orientation_transform: Will superseed the global transform of the Collection.

            rotation_matrix:

        """

        orientation_transform = (
            self.orientation_transform
            if orientation_transform is None
            else orientation_transform
        )

        svg = SVG.parse(file_location)
        paths = list(svg.elements())

        poly_list = []
        for path in svg:
            pl = []
            n_points = int(path.length() // divisor)
            linspace = np.linspace(0, 1, n_points)

            for index in linspace:
                poly = np.array(path.point(index))
                pl.append([poly[0], -poly[1]])

            arr = np.array(pl) @ rotation_matrix * multiplier + offset

            to_add = Shape(points=arr, orientation_transform=orientation_transform)
            self.add_shape(to_add)


class Shape:
    """Class for creating a single shape object."""

    def __init__(
        self,
        points: np.ndarray = np.empty((1, 2)),
        well: Optional[str] = None,
        name: Optional[str] = None,
        orientation_transform=None,
    ):
        """Class for creating a single shape.

        Args:
            points: Array or list of lists in the shape of `(N,2)`. Contains the points of the polygon forming a shape.

            well: Well in which to sort the shape after cutting. For example A1, A2 or B3.

            name: Name of the shape.
        """

        # Orientation transform of shapes
        self.orientation_transform: Optional[np.ndarray] = orientation_transform

        # Allthoug a numpy array is recommended, list of lists is accepted
        points = np.array(points)

        # Assert correct dimensions
        point_shapes = points.shape
        if (len(point_shapes) != 2) or (point_shapes[1] != 2) or (point_shapes[0] == 0):
            raise ValueError("please provide a numpy array of shape (N, 2)")

        self.points: np.ndarray = points

        self.name: Optional[str] = name
        self.well: Optional[str] = well

    def from_xml(self, root):
        """Load a shape from an XML shape node. Used internally for reading LMD generated XML files.
        
        Args:
            root: XML input node.
        """
        self.name = root.tag

        # get number of points
        point_count = int(root.find("PointCount").text)   
        points = np.empty((point_count, 2), dtype=int)

        # compile regex 
        xpattern = re.compile("X_(\d+)")
        ypattern = re.compile("Y_(\d+)")
        
        # parse all points
        for child in root:

            xmatch = re.findall(xpattern, child.tag)
            ymatch = re.findall(ypattern, child.tag)
            
            if xmatch:
                point_id = int(xmatch[0]) - 1
                points[point_id, 0] = int(child.text)            
            elif ymatch:
                point_id = int(ymatch[0]) - 1
                points[point_id, 1] = int(child.text)  
            elif child.tag == "CapID":
                self.well = str(child.text)

        self.points = np.array(points)

    def to_xml(self, id: int, orientation_transform: np.ndarray, scale: int):
        """Generate XML shape node needed internally for export.

        Args:
            id: Sequential identifier of the shape as used in the LMD XML format.

            orientation_transform (np.array): Pass orientation_transform which is used if no local orientation transform is set.

            scale (int): Scalling factor used to enable higher decimal precision.

        Note:
            If the Shape has a custom orientation_transform defined, the custom orientation_transform is applied at this point. If not, the oritenation_transform passed by the parent Collection is used. This highlights an important difference between the Shape and Collection class. The Collection will always has an orientation transform defined and will use `np.eye(2)` by default. The Shape object can have a orientation_transform but can also be set to `None` to use the Collection value.

        """

        # Apply orientation transform. If the Shape has a custom orientation_transform defined, the custom orientation_transform is applied at this point. If not, the oritenation_transform passed by the parent Collection is used. This highlights an important difference between the Shape and Collection class. The Collection will always has an orientation transform defined and will use `np.eye(2)` by default. The Shape object can have a orientation_transform but can also be set to `None` to use the Collection value.

        if self.orientation_transform is not None:
            transformed_points = self.points @ self.orientation_transform * scale
        else:
            transformed_points = self.points @ orientation_transform * scale

        shape = ET.Element("Shape_{}".format(id))

        point_count = ET.SubElement(shape, "PointCount")
        point_count.text = "{}".format(len(transformed_points))

        if self.well is not None:
            cap_id = ET.SubElement(shape, "CapID")
            cap_id.text = self.well

        # write points
        for i, point in enumerate(transformed_points):
            id = i + 1
            x = ET.SubElement(shape, "X_{}".format(id))
            x.text = "{}".format(np.floor(point[0]).astype(int))

            y = ET.SubElement(shape, "Y_{}".format(id))
            y.text = "{}".format(np.floor(point[1]).astype(int))

        return shape

    def to_shapely(self):
        return shapely.Polygon(self.points)


class SegmentationLoader:
    """Select single cells from a segmentation and generate cutting data
        
        Args:
            config (dict): Dict containing configuration parameters. See Note for further explanation.
            processes (int): Number of processes used for parallel processing of cell sets. Total processes can be calculated as `processes * threads`.
            threads (int): Number of threads used for parallel processing of shapes within a cell set. Total processes can be calculated as `processes * threads`.
            
            cell_sets (list(dict)): List of dictionaries containing the sets of cells which should be sorted into a single well.
            
            calibration_marker (np.array): Array of size '(3,2)' containing the calibration marker coordinates in the '(row, column)' format.    

            coords_lookup (None, dict): precalculated lookup table for coordinates of individual cell ids. If not provided will be calculated.
            
            classes (np.array): Array of classes found in the provided segmentation mask. If not provided will be calculated based on the assumption that cell_ids are assigned in ascending order.
                    
        Example:
                    
            .. code-block:: python
            
                import numpy as np
                from PIL import Image
                from lmd.lib import SegmentationLoader

    Args:
        config (dict): Dict containing configuration parameters. See Note for further explanation.

        cell_sets (list(dict)): List of dictionaries containing the sets of cells which should be sorted into a single well.

        calibration_marker (np.array): Array of size '(3,2)' containing the calibration marker coordinates in the '(row, column)' format.

    Example:

        .. code-block:: python

            import numpy as np
            from PIL import Image
            from lmd.lib import SegmentationLoader

            im = Image.open('segmentation_cytosol.tiff')
            segmentation = np.array(im).astype(np.uint32)

            all_classes = np.unique(segmentation)

            cell_sets = [{"classes": all_classes, "well": "A1"}]

            calibration_points = np.array([[0,0],[0,1000],[1000,1000]])

            loader_config = {
                'orientation_transform': np.array([[0, -1],[1, 0]])
            }

            sl = SegmentationLoader(config = loader_config)
            shape_collection = sl(segmentation,
                                cell_sets,
                                calibration_points)

            shape_collection.plot(fig_size = (10, 10))

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

            # Paramter required for hilbert curve based path optimization.
            # Defines the order of the hilbert curve used, which needs to be tuned with the total cutting area.
            # For areas of 1 x 1 mm we recommend at least p = 4,  for whole slides we recommend p = 7.
            hilbert_p: 7

            # Parameter required for greedy path optimization.
            # Instead of a global distance matrix, the k nearest neighbours are approximated.
            # The optimization problem is then greedily solved for the known set of nearest neighbours until the first set of neighbours is exhausted.
            # Established edges are then removed and the nearest neighbour approximation is recursivly repeated.
            greedy_k: 20

            # Overlapping shapes are merged based on a nearest neighbour heuristic.
            # All selected shapes closer than distance_heuristic pixel are checked for overlap.
            distance_heuristic: 300


    """

    # define all valid path optimization methods used with the "path_optimization" argument in the configuration
    VALID_PATH_OPTIMIZERS = ["none", "hilbert", "greedy"]
    DEFAULT_SEGMENTATION_DTYPE = np.uint64
    
    def __init__(self, config = {}, verbose = False, processes = 1):
        self.config = config
        self.verbose = verbose
        self._get_context()  # setup context for multiprocessing function calls to work with different operating systems

        self.register_parameter('shape_dilation', 0)
        self.register_parameter('shape_erosion', 0)
        self.register_parameter('binary_smoothing', 3)
        self.register_parameter('convolution_smoothing', 15)
        self.register_parameter('rdp_epsilon', 0.1)
        self.register_parameter('path_optimization', 'hilbert')
        self.register_parameter('greedy_k', 0)
        self.register_parameter('hilbert_p', 7)
        self.register_parameter('xml_decimal_transform', 100)
        self.register_parameter('distance_heuristic', 300)
        self.register_parameter('join_intersecting', True)
        self.register_parameter('orientation_transform', np.eye(2))
        self.register_parameter('threads', 10)

        self.coords_lookup = None
        self.processes = processes

        self._configure_path_optimizer()

    def _configure_path_optimizer(self):
        #configure path optimizer
        if 'path_optimization' in self.config:
            optimization_method = self.config['path_optimization']
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
    
    def __call__(self, input_segmentation: np.ndarray | None, cell_sets, calibration_points, coords_lookup = None, classes = np.array([], dtype=np.uint64)):
        
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
            self.processes = len(sets) #reduce number of processes if there are less cell sets than processes
        
        self.input_segmentation = input_segmentation

        if coords_lookup is None:
            self.log("Calculating coordinate locations of all cells.")
            #deprecated infavour of more computationally efficient solution
            #self.coords_lookup = _create_coord_index(self.input_segmentation, classes = classes)
            #self.coords_lookup = {k: np.array(v) for k, v in self.coords_lookup.items()}
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
                tqdm_args=dict(
                    file=sys.stdout,
                    disable=not self.verbose,
                    desc="collecting cell sets",
                ),
                n_threads=self.processes,
            )
        else:
            print("Processing cell sets in serial")
            self.multi_threading = False
            collections = []
            for i, cell_set in enumerate(cell_sets):
                collections.append(self.generate_cutting_data(i, cell_set))

        return reduce(lambda a, b: a.join(b), collections)

    def generate_cutting_data(self, i, cell_set):
        if 0 in cell_set["classes_loaded"]:
            cell_set["classes_loaded"] = cell_set["classes_loaded"][
                cell_set["classes_loaded"] != 0
            ]
            warnings.warn(
                "Class 0 is not a valid class and was removed from the cell set"
            )

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
            self.log(
                f"{elements_removed} classes were not found and therefore removed."
            )

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
                
        if zero_elements <= 2: #allow at most for 2 zero elements (x = 0 and y = 0)
            pass
        else:
            self.log(
                "Check failed, returned coordinates contain empty elements. Please check if all classes specified are present in your segmentation"
            )

        if self.config['join_intersecting']:
            center, length, coords = self.merge_dilated_shapes(center, length, coords, 
                                                               dilation = self.config['shape_dilation'],
                                                               erosion = self.config['shape_erosion'])

        # Calculate dilation and erosion based on if merging was activated
        dilation = self.config['binary_smoothing'] if self.config['join_intersecting'] else self.config['binary_smoothing'] + self.config['shape_dilation']
        erosion = self.config['binary_smoothing'] if self.config['join_intersecting'] else self.config['binary_smoothing'] + self.config['shape_erosion']
        
        if self.config["threads"] == 1:  
            shapes = []
            for coord in tqdm(coords, desc = "creating shapes"):
                shapes.append(transform_to_map(coord, dilation = dilation, erosion = erosion, coord_format = False))
        else:
            with mp.get_context(self.context).Pool(processes=self.config['threads']) as pool:           
                shapes = list(tqdm(pool.imap(partial(transform_to_map, 
                                                    erosion = erosion,
                                                    dilation = dilation,
                                                    coord_format = False),
                                                    coords), total=len(center), 
                                                    disable = not self.verbose, 
                                                    desc = "creating shapes"))
                
            
        if self.config["threads"] == 1:  
            polygons = []
            for shape in tqdm(shapes, desc = "calculating polygons"):
                polygons.append(_create_poly(shape, 
                                          smoothing_filter_size = self.config['convolution_smoothing'],
                                          rdp_epsilon = self.config['rdp_epsilon']))
        else:
            with mp.get_context(self.context).Pool(processes=self.config['threads']) as pool:      
                polygons = list(tqdm(pool.imap(partial(_create_poly, 
                                                    smoothing_filter_size = self.config['convolution_smoothing'],
                                                    rdp_epsilon = self.config['rdp_epsilon']
                                                    ),
                                                    shapes), total=len(center), 
                                                    disable = not self.verbose, 
                                                    desc = "calculating polygons" ))
        
        #perform path optimization to minimize the total distance that the LMD travels during cutting (this improves cutting speed and focus)
        center = np.array(center)
        unoptimized_length = calc_len(center)
        self.log(f"Current path length: {unoptimized_length:,.2f} units")

        if self.optimization_method != "none":
            if self.optimization_method  == "greedy":
                optimized_idx = tsp_greedy_solve(center, k=self.config['greedy_k'])
        
            elif self.optimization_method  == "hilbert":
                optimized_idx = tsp_hilbert_solve(center, p=self.config['hilbert_p'])
            
            #update order of centers
            center = center[optimized_idx]
            self.indexes = optimized_idx

            # calculate optimized path length and optimization factor
            optimized_length = calc_len(center)
            self.log(f"Optimized path length: {optimized_length:,.2f} units")

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

            if 'background_image' in self.config:
                axs.imshow(self.config['background_image'])

            axs.scatter(center[:,1], center[:,0], s=1)

            for shape in polygons:
                axs.plot(shape[:,1], shape[:,0], color="red",linewidth=1)
    

            axs.scatter(self.calibration_points[:,1], self.calibration_points[:,0], color="blue")
            axs.plot(center[:,1],center[:,0], color="grey")
            axs.invert_yaxis()
            axs.set_aspect('equal', adjustable='box')
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
        ds = Collection(calibration_points=self.calibration_points)
        ds.orientation_transform = self.config["orientation_transform"]

        for shape in polygons:
            # Check if well key is set in cell set definition
            if "well" in cell_set:
                ds.new_shape(shape, well=cell_set["well"])
            else:
                ds.new_shape(shape)
        return ds

    def merge_dilated_shapes(self,
                        input_center, 
                        input_length, 
                        input_coords, 
                        dilation = 0,
                        erosion = 0):
        print("Intersecting Shapes will be merged into a single shape.")
        
        # initialize all shapes and create dilated coordinates
        # coordinates are created as complex numbers to facilitate comparison with np.isin
        dilated_coords = []

        if self.config["threads"] == 1:
            for coord in tqdm(input_coords, desc = "dilating shapes"):
                dilated_coords.append(transform_to_map(coord, dilation = dilation))
        
        else:
            with mp.get_context(self.context).Pool(processes=self.config['threads']) as pool:           
                dilated_coords = list(tqdm(pool.imap(partial(transform_to_map, 
                                                    dilation = dilation),
                                                    input_coords), total=len(input_center), 
                                                    desc = "dilating shapes"))
            
        dilated_coords = [np.apply_along_axis(lambda args: [complex(*args)], 1, d).flatten() for d in dilated_coords]

        # A sparse distance matrix is calculated for all cells which are closer than distance_heuristic
        center_arr = np.array(input_center)
        center_tree = cKDTree(center_arr)

        sparse_distance = center_tree.sparse_distance_matrix(
            center_tree, self.config["distance_heuristic"]
        )
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
        sparse_intersect = scipy.sparse.coo_matrix(
            (intersect_data, (sparse_distance.row, sparse_distance.col))
        )
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
            coords_2d = np.array(
                [coords_complex.real, coords_complex.imag], dtype=int
            ).T

            # calculate properties length and center from coords
            new_center = np.mean(coords_2d, axis=0)
            new_len = len(coords_2d)

            # append values to output lists
            output_center.append(new_center)
            output_length.append(new_len)
            output_coords.append(coords_2d)

        print(len(to_merge) - len(output_center), "shapes that were intersecting were found and merged.")
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
                path = os.path.join(
                    Path(self.directory).parents[0], cell_set["classes"]
                )

            if os.path.isfile(path):
                try:
                    cr = csv.reader(open(path, "r"))
                    filtered_classes = np.array(
                        [int(el[0]) for el in list(cr)], dtype="int64"
                    )
                    self.log("Loaded {} classes from csv".format(len(filtered_classes)))
                    return filtered_classes
                except:
                    self.log(
                        "CSV file could not be converted to list of integers: {path}"
                    )
                    raise ValueError()
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
                raise NotImplementedError('registration of parameters is not yet supported for nested parameters')
                
            else:
                raise TypeError('Key musst be of string or a list of strings')
            
            if key not in config_handle:
                self.log(f'No configuration for {key} found, parameter will be set to {value}')
                config_handle[key] = value
            
            
def transform_to_map(coords, 
                    dilation = 0, 
                    erosion = 0, 
                    coord_format = True,
                    debug = False):
    # safety boundary which extands the generated map size
    safety_offset = 3
    dilation_offset = int(dilation)

    coords = np.array(coords).astype(int)

    # top left offset used for creating the offset map
    if coords.size == 0:
        raise ValueError("coords array is empty; cannot compute minimum.")
    offset = np.min(coords, axis=0) - safety_offset - dilation_offset
    mat = np.array([offset, [0, 0]])
    offset = np.max(mat, axis=0)

    offset_coords = coords - offset
    offset_coords = offset_coords.astype(np.uint)

    offset_map_size = (
        np.max(offset_coords, axis=0) + 2 * safety_offset + dilation_offset
    )
    offset_map_size = offset_map_size.astype(np.uint)

    offset_map = np.zeros(offset_map_size, dtype=np.ubyte)

    y = tuple(offset_coords.T[0])
    x = tuple(offset_coords.T[1])

    offset_map[(y, x)] = 1

    if debug:
        plt.imshow(offset_map)
        plt.show()

    offset_map = binary_dilation(offset_map, footprint=disk(dilation))
    offset_map = binary_erosion(offset_map, footprint=disk(erosion))
    offset_map = ndimage.binary_fill_holes(offset_map).astype(int)

    if debug:
        plt.imshow(offset_map)
        plt.show()

    # coord_format will return a sparse format of [[2, 1],[1, 2],[0, 2]]
    # otherwise will return a dense matrix and the offset [[0, 0, 1],[0, 0, 1],[0, 1, 0]]

    if coord_format:
        idx_local = np.argwhere(offset_map == 1)
        idx_global = idx_local + offset
        return idx_global
    else:
        return (offset_map, offset)

def _create_poly(in_tuple, 
                smoothing_filter_size: int = 12,
                rdp_epsilon: float = 0, 
                debug: bool = False):

    """Converts a list of pixels into a polygon.
    Args
        smoothing_filter_size (int, default = 12): The smoothing filter is the circular convolution with a vector of length smoothing_filter_size and all elements 1 / smoothing_filter_size.
        
        rdp_epsilon (float, default = 0 ): When compression is wanted, this specifies the epsilon value for the Ramer-Douglas-Peucker algorithm. Higher values will result in more compression.

        dilation (int, default = 0): Binary dilation used before polygon creation for increasing the mask size. This Dilation ignores potential neighbours. Neighbour aware dilation of segmentation mask needs to be defined during segmentation.
    """
    (offset_map, offset) = in_tuple

    # find polygon bounds from mask
    bounds = find_boundaries(offset_map, connectivity=1, mode="subpixel", background=0)
    
    edges = np.array(np.where(bounds == 1))/2
    edges = edges.T
    edges = _sort_edges(edges)

    # smoothing resulting shape
    smk = np.ones((smoothing_filter_size,1))/smoothing_filter_size
    edges = convolve2d(edges, smk,mode="full",boundary="wrap")
    
    # compression of the resulting polygon   
    poly = rdp(edges, epsilon = rdp_epsilon) # Ramer-Douglas-Peucker algorithm for polygon simplification

    # debuging
    """
    print(self.poly.shape)
    fig = plt.figure(frameon=False)
    fig.set_size_inches(10,10)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(bounds)
    ax.plot(edges[:,1]*2,edges[:,0]*2)
    ax.plot(self.poly[:,1]*2,self.poly[:,0]*2)
    """

    return poly + offset


def _sort_edges(edges):
    """Sorts the vertices of the polygon.
    Greedy sorting is performed, might have difficulties with complex shapes.

    """

    it = len(edges)
    new = []
    new.append(edges[0])

    edges = np.delete(edges, 0, 0)

    for i in range(1, it):
        old = np.array(new[i - 1])

        dist = np.linalg.norm(edges - old, axis=1)

        min_index = np.argmin(dist)
        new.append(edges[min_index])
        edges = np.delete(edges, min_index, 0)

    return np.array(new)
