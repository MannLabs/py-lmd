"""Path optimization utilities for LMD cutting path generation.

This module provides algorithms for solving the Traveling Salesman Problem (TSP)
to optimize the order of cutting shapes, minimizing total travel distance for
laser microdissection operations.
"""

from typing import Any, TypeVar, Union

import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
from numba import njit

try:
    import umap

    UMAP_INSTALLED = True
except ImportError:
    UMAP_INSTALLED = False

T = TypeVar("T")


def calc_len(data: np.ndarray) -> float:
    """Calculate the length of a path based on a list of coordinates

    Args:
        data: Array of shape `(N, 2)` containing a list of coordinates

    Returns:
        The total length of the path.
    """

    index = np.arange(len(data)).astype(int)

    not_shifted = data[index[:-1]]
    shifted = data[index[1:]]

    diff = not_shifted - shifted
    sq = np.square(diff)
    dist = np.sum(np.sqrt(np.sum(sq, axis=1)))

    return dist


@njit()
def assign_vertices(hilbert_points: np.ndarray, data_rounded: np.ndarray) -> np.ndarray:
    data_rounded = data_rounded.astype(np.int64)
    hilbert_points = hilbert_points.astype(np.int64)

    output_order = np.zeros(len(data_rounded)).astype(np.int64)
    current_index = 0

    for hilbert_point in hilbert_points:
        for i, data_point in enumerate(data_rounded):
            if np.array_equal(hilbert_point, data_point):
                output_order[current_index] = i
                current_index += 1

    return output_order


def tsp_hilbert_solve(data: np.ndarray, p: int = 3) -> np.ndarray:
    """Approximate a short traversal path between centroids with a Hilbert Curve

    A Hilbert curve provides a space-filling mapping from 2D coordinates to a
    one-dimensional order that tends to preserve spatial locality. Here, this
    property is used as a heuristic to obtain an ordering of point centroids that
    typically yields a shorter traversal path than a random ordering, but does
    not guarantee an optimal Traveling Salesperson solution.

    Args:
        data: 2D Array of shape `(N, 2)` containing a list of coordinates
        p: Iterations to use in constructing the Hilbert curve.

    Returns:
        Ordered indices of data of the shape `(N,)` according to their position along the Hilbert curve.
    """
    n = 2
    max_n = 2 ** (p * n)
    hilbert_curve = HilbertCurve(p, n)
    distances = list(range(max_n))
    hilbert_points = hilbert_curve.points_from_distances(distances)
    hilbert_points = np.array(hilbert_points)

    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)

    hilbert_min = np.min(hilbert_points, axis=0)
    hilbert_max = np.max(hilbert_points, axis=0)

    data_scaled = data - data_min
    data_scaled = data_scaled / (data_max - data_min) * (hilbert_max - hilbert_min)

    data_rounded = np.round(data_scaled).astype(int)

    order = assign_vertices(hilbert_points, data_rounded)

    return order


# TODO: Remove unused argument `world_size`
def _get_closest(used: list[T], choices: Union[list[T], np.ndarray], world_size: Any) -> Union[T, None]:
    """Greedily select the first unvisited element in a list of k-nearest neighbors

    Args:
        used: List of elements that have been used already
        choices: List of all available choices (nearest neighbors)
        world_size: Unused argument

    Returns:
        First element in `choices` that is not in `used` or
        None if all nearest neighbors have been visited.
    """
    for element in choices:
        if element not in used:
            # knn matrix contains -1 if the number of elements is smaller than k
            if element == -1:
                return None
            else:
                return element

    # all choices have been taken, return closest free index due to local optimality
    return None


def _tsp_greedy_solve(data: np.ndarray, k: int = 100) -> np.ndarray:
    """Approximate a short traversal path between centroids with a greedy nearest neighbors search

    Args:
        data: Array of shape `(N, 2)` containing a list of coordinates
        k: K-Nearest neighbors selection

    Returns:
        Array of shape `(N, 2)` containing the input coordinates ordered
        along the greedy TSP path.
    """
    if not UMAP_INSTALLED:
        raise ImportError(
            "umap-learn is required for this function. Install it with: pip install py-lmd[umap]"
        ) from None
    samples = len(data)

    print(f"{samples} nodes left")
    # recursive abort
    if samples == 1:
        return data

    try:
        import umap
    except ImportError:
        raise ImportError(
            "umap-learn is required for this function. " "Install it with: pip install py-lmd[umap]"
        ) from None

    knn_index, knn_dist, _ = umap.umap_.nearest_neighbors(
        data, n_neighbors=k, metric="euclidean", metric_kwds={}, angular=True, random_state=np.random.RandomState(42)
    )

    knn_index = knn_index[:, 1:]
    knn_dist = knn_dist[:, 1:]

    # follow greedy knn as long as a nearest neighbour is found in the current tree
    nodes = []
    current_node = 0
    while current_node is not None:
        nodes.append(current_node)
        # print(current_node, knn_index[current_node], next_node)
        next_node = _get_closest(nodes, knn_index[current_node], samples)

        current_node = next_node

    # as soon as no nearest neigbour can be found, create a new list of all elements still remeining
    # nodes: [0, 2, 5], nodes_left: [1, 3, 4, 6, 7, 8, 9]
    # add the last node assigned as starting point to the new list
    # nodes: [0, 2], nodes_left: [5, 1, 3, 4, 6, 7, 8, 9]
    nodes_left = list(set(range(samples)) - set(nodes))

    # add last node from nodes to nodes_left
    nodes_left = [nodes.pop(-1)] + nodes_left

    node_data_left = data[nodes_left]

    # join lists
    return np.concatenate([data[nodes], _tsp_greedy_solve(node_data_left, k=k)])


@njit()
def _get_nodes(data: np.ndarray, sorted_data: np.ndarray) -> list[int]:
    """Find indices that map original coordinates to their sorted positions.

    Given an unsorted array and its sorted version, returns the indices
    such that `data[result]` produces `sorted_data`.

    Args:
        data: Array of shape `(N, 2)` containing original coordinates.
        sorted_data: Array of shape `(N, 2)` containing the same coordinates
            in sorted order.

    Returns:
        List of indices representing the order to traverse `data` to match
        `sorted_data`.
    """
    indexed_data = list(enumerate(data))

    nodes = []
    for element in sorted_data:
        for j, tup in enumerate(indexed_data):
            i, el = tup

            if np.array_equal(el, element):
                nodes.append(i)
                indexed_data.pop(j)
    return nodes


def tsp_greedy_solve(node_list: np.ndarray, k: int = 100, return_sorted: bool = False) -> Union[np.ndarray, list[int]]:
    """Find an approximation of the shortest path through a list of coordinates

    Args:
        node_list: Array of shape `(N, 2)` containing a list of coordinates
        k: Number of Nearest Neighbours calculated for each Node.
        return_sorted: If set to False a list of indices is returned. If set to True the sorted coordinates are returned.

    Returns:
        An array
            - `return_sorted=True`: Array of sorted nodes.
            - `return_sorted=False`: Ordered indices of nodes
    """
    sorted_nodes = _tsp_greedy_solve(node_list, k=k)

    if return_sorted:
        return sorted_nodes

    else:
        nodes_order = _get_nodes(node_list, sorted_nodes)
        return nodes_order
