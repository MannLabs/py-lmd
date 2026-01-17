from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from rdp import rdp
from scipy import ndimage
from scipy.signal import convolve2d
from skimage.morphology import binary_erosion, disk
from skimage.morphology import dilation as binary_dilation
from skimage.segmentation import find_boundaries
from tqdm.auto import tqdm


# TODO: Rename tqdm_args to tqdm_kwargs
def _execute_indexed_parallel(func: Callable, *, args: list, tqdm_args: dict = None, n_threads: int = 10) -> list:
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


# TODO: Remove debug argument [Breaking]
# TODO: Add type hints
# TODO: Add docstring to public method
def transform_to_map(coords, dilation=0, erosion=0, coord_format=True, debug=False):
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

    offset_map_size = np.max(offset_coords, axis=0) + 2 * safety_offset + dilation_offset
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


# TODO: Remove debugging logic
def _create_poly(
    in_tuple,
    smoothing_filter_size: int = 12,
    rdp_epsilon: float = 0,
    debug: bool = False,
):
    """Converts a list of pixels into a polygon.
    Args
        smoothing_filter_size (int, default = 12): The smoothing filter is the circular convolution with a vector of length smoothing_filter_size and all elements 1 / smoothing_filter_size.

        rdp_epsilon (float, default = 0 ): When compression is wanted, this specifies the epsilon value for the Ramer-Douglas-Peucker algorithm. Higher values will result in more compression.

        dilation (int, default = 0): Binary dilation used before polygon creation for increasing the mask size. This Dilation ignores potential neighbours. Neighbour aware dilation of segmentation mask needs to be defined during segmentation.
    """
    (offset_map, offset) = in_tuple

    # find polygon bounds from mask
    bounds = find_boundaries(offset_map, connectivity=1, mode="subpixel", background=0)

    edges = np.array(np.where(bounds == 1)) / 2
    edges = edges.T
    edges = _sort_edges(edges)

    # smoothing resulting shape
    smk = np.ones((smoothing_filter_size, 1)) / smoothing_filter_size
    edges = convolve2d(edges, smk, mode="full", boundary="wrap")

    # compression of the resulting polygon
    poly = rdp(edges, epsilon=rdp_epsilon)  # Ramer-Douglas-Peucker algorithm for polygon simplification

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
