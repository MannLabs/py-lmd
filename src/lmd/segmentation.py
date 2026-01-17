import gc
import warnings

import numba as nb
import numpy as np
from numba import njit, prange, types
from scipy.sparse import coo_array

# =============================================================================
# Deprecation Aliases - Path Optimization Functions
# =============================================================================
# These functions have been moved to lmd.path module.
# Imports from lmd.segmentation are deprecated and will be removed in v3.0.0
from lmd.path import (
    _get_closest as __get_closest,
)
from lmd.path import (
    _get_nodes as __get_nodes,
)
from lmd.path import (
    _tsp_greedy_solve as __tsp_greedy_solve,
)
from lmd.path import (
    assign_vertices as _assign_vertices,
)
from lmd.path import (
    calc_len as _calc_len,
)
from lmd.path import (
    tsp_greedy_solve as _tsp_greedy_solve,
)
from lmd.path import (
    tsp_hilbert_solve as _tsp_hilbert_solve,
)

# TODO: Rename index_list to index_dict to correctly represent type
# TODO: Rename index_list to index_dict to correctly represent type


# TODO: Add parameter documentation, return information
@njit
def _numba_accelerator_coord_calculation(
    _ids: np.ndarray, inverse_indices: np.ndarray, sparse_coords_0: np.ndarray, sparse_coords_1: np.ndarray
) -> dict:
    """Accelerate the transformation of a sparse array to a coordinate list using numba"""

    # initialize datastructure for storing results: dict[cell_id] = np.array([0, 0, 0, ...])
    # final structure will be a dictionary with class_id as key and an array of coordinates as value
    # the array for the coordinates is initialized with zero values and its size dynamically increased during processing
    # in a second array the next unfilled coordinate position is stored
    index_list = {}
    stop_list = {}
    initial_size = 32  # type: int # initial size of the place holder array for storing coordinate information

    for i in _ids:
        index_list[i] = np.zeros((initial_size, 2), dtype=np.uint64)
        stop_list[i] = 0

    for idx, _ix in enumerate(inverse_indices):
        _id = _ids[_ix]  # get current cell id
        current_size = index_list[_id].shape[0]
        # ensure array for storing coords is large enough to store the new coordinates
        if stop_list[_id] >= current_size:
            new_size = current_size * 2
            new_array = np.zeros((new_size, 2), dtype=np.uint64)
            new_array[: stop_list[_id]] = index_list[_id]
            index_list[_id] = new_array

        index_list[_id][stop_list[_id]][0] = sparse_coords_0[idx]
        index_list[_id][stop_list[_id]][1] = sparse_coords_1[idx]
        stop_list[_id] += 1

    # resize index list
    for _id in _ids:
        index_list[_id] = index_list[_id][: stop_list[_id]]

    return index_list


def _create_coord_index_sparse(mask: np.ndarray) -> dict:
    """Create a coordinate index from a segmentation mask.
    In the coordinate index each key is a unique class id and the value is a list of coordinates for this class.

    Args:
        mask (np.ndarray): A 2D segmentation mask

    Returns:
        dict: A dictionary containing the class ids as keys
              and a list of coordinates as values
    """
    # convert to a sparse array (this will be faster than iterating over the dense array because segmentation masks are usually sparse)
    sparse_mask = coo_array(mask)

    cell_ids = sparse_mask.data
    sparse_coords_0 = sparse_mask.coords[0]
    sparse_coords_1 = sparse_mask.coords[1]

    del sparse_mask  # this is no longer needed for calculations and should free up memory when explicitly deleted
    gc.collect()

    _ids, inverse_indices = np.unique(cell_ids, return_inverse=True)

    coords_lookup = _numba_accelerator_coord_calculation(_ids, inverse_indices, sparse_coords_0, sparse_coords_1)
    coords_lookup = dict(coords_lookup)
    return coords_lookup


# TODO: add type hints. See GitHub comment by @sophiamaedler: https://github.com/MannLabs/py-lmd/pull/54#discussion_r2689850963
# TODO: Remove unused argument `background`. See GitHub comment by @sophiamaedler: https://github.com/MannLabs/py-lmd/pull/54#discussion_r2689831082
# TODO: Clarify whether it is desired that non-existent classes (e.g. 1 in case classes=[0, 2]) are still returned
@njit
def _create_coord_index(mask, background=0, classes=np.array([], dtype=np.uint64), dtype=np.uint64):
    if len(classes) == 0:
        num_classes = np.max(mask) + 1
        classes = np.arange(num_classes, dtype=dtype)
    else:
        num_classes = len(classes)

    # each class will have a list of coordinates
    # each coordinate is a 2D array with row and column
    # initial size is 32, whenever it exceeds the size, it will be doubled
    initial_size = 32

    # Use dictionaries for faster access and creation, as only requested class_ids are generated
    index_list = {}
    stop_list = {i: 0 for i in classes}

    for i in classes:
        index_list[i] = np.zeros((initial_size, 2), dtype=dtype)

    # Create index list
    for col in range(mask.shape[1]):
        for row in range(mask.shape[0]):
            class_id = mask[row, col]
            if class_id in classes:
                if stop_list[class_id] >= index_list[class_id].shape[0]:
                    new_size = index_list[class_id].shape[0] * 2
                    new_array = np.zeros((new_size, 2), dtype=dtype)
                    new_array[: stop_list[class_id]] = index_list[class_id]
                    index_list[class_id] = new_array
                index_list[class_id][stop_list[class_id]][0] = row
                index_list[class_id][stop_list[class_id]][1] = col
                stop_list[class_id] += 1

    # Resize index list to remove empty elements
    for class_id in classes:
        index_list[class_id] = index_list[class_id][: stop_list[class_id]]

    return index_list


# TODO: add type hints
def _filter_coord_index(index_list, classes, background=0):
    filtered_index_list = []
    for _idx, class_id in enumerate(classes):
        if class_id != background:
            _coords = index_list[class_id]
            if len(_coords) > 0:
                filtered_index_list.append(index_list[class_id])
            else:
                Warning(f"Coordinate list for {class_id} is empty and was dropped.")
    return filtered_index_list


# TODO: Make private
def get_coordinate_form(classes, coords_lookup, debug=False):
    # return with empty lists if no classes are provided
    if len(classes) == 0:
        return [], [], []

    coords_filtered = _filter_coord_index(coords_lookup, classes)

    center = [np.mean(el, axis=0) for el in coords_filtered]

    # TODO: Remove print statement
    if debug:
        print("start length calculation")

    length = [len(el) for el in coords_filtered]

    return center, length, coords_filtered


def calc_len(data):
    """Deprecated: Use lmd.path.calc_len instead."""
    warnings.warn(
        "calc_len has been moved to lmd.path. "
        "Import from lmd.segmentation is deprecated and will be removed in v3.0.0. "
        "Please use: from lmd.path import calc_len",
        DeprecationWarning,
        stacklevel=2,
    )
    return _calc_len(data)


def tsp_hilbert_solve(data, p=3):
    """Deprecated: Use lmd.path.tsp_hilbert_solve instead."""
    warnings.warn(
        "tsp_hilbert_solve has been moved to lmd.path. "
        "Import from lmd.segmentation is deprecated and will be removed in v3.0.0. "
        "Please use: from lmd.path import tsp_hilbert_solve",
        DeprecationWarning,
        stacklevel=2,
    )
    return _tsp_hilbert_solve(data, p=p)


def tsp_greedy_solve(node_list, k=100, return_sorted=False):
    """Deprecated: Use lmd.path.tsp_greedy_solve instead."""
    warnings.warn(
        "tsp_greedy_solve has been moved to lmd.path. "
        "Import from lmd.segmentation is deprecated and will be removed in v3.0.0. "
        "Please use: from lmd.path import tsp_greedy_solve",
        DeprecationWarning,
        stacklevel=2,
    )
    return _tsp_greedy_solve(node_list, k=k, return_sorted=return_sorted)


def assign_vertices(hilbert_points, data_rounded):
    """Deprecated: Use lmd.path.assign_vertices instead."""
    warnings.warn(
        "assign_vertices has been moved to lmd.path. "
        "Import from lmd.segmentation is deprecated and will be removed in v3.0.0. "
        "Please use: from lmd.path import assign_vertices",
        DeprecationWarning,
        stacklevel=2,
    )
    return _assign_vertices(hilbert_points, data_rounded)


def _get_closest(used, choices, world_size):
    """Deprecated: Use lmd.path._get_closest instead."""
    warnings.warn(
        "_get_closest has been moved to lmd.path. "
        "Import from lmd.segmentation is deprecated and will be removed in v3.0.0. "
        "Please use: from lmd.path import _get_closest",
        DeprecationWarning,
        stacklevel=2,
    )
    return __get_closest(used, choices, world_size)


def _get_nodes(data, sorted_data):
    """Deprecated: Use lmd.path._get_nodes instead."""
    warnings.warn(
        "_get_nodes has been moved to lmd.path. "
        "Import from lmd.segmentation is deprecated and will be removed in v3.0.0. "
        "Please use: from lmd.path import _get_nodes",
        DeprecationWarning,
        stacklevel=2,
    )
    return __get_nodes(data, sorted_data)


def _tps_greedy_solve(data, k=100):
    """Deprecated: Use lmd.path._tsp_greedy_solve instead."""
    warnings.warn(
        "_tps_greedy_solve has been moved to lmd.path and renamed to _tsp_greedy_solve. "
        "Import from lmd.segmentation is deprecated and will be removed in v3.0.0. "
        "Please use: from lmd.path import _tsp_greedy_solve",
        DeprecationWarning,
        stacklevel=2,
    )
    return __tsp_greedy_solve(data, k=k)
