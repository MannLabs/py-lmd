"""Tests for lmd.path module - path optimization functions."""

from __future__ import annotations

import numpy as np
import pytest

from lmd.path import (
    _get_closest,
    _get_nodes,
    assign_vertices,
    calc_len,
    tsp_hilbert_solve,
)

ATOL = 1e-10


@pytest.mark.parametrize(
    ("data", "p"),
    [
        # Two points - should return valid ordering
        (np.array([[0.0, 0.0], [1.0, 1.0]]), 3),
        # Four corner points
        (np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]), 3),
        # Linear points
        (np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]), 3),
        # More points with different p value
        (np.array([[0.0, 0.0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.75], [1.0, 1.0]]), 4),
    ],
    ids=("two_points", "four_corners", "linear_points", "five_points_p4"),
)
def test_tsp_hilbert_solve(data: np.ndarray, p: int) -> None:
    """Test `tsp_hilbert_solve` returns valid ordering"""
    # Act
    result = tsp_hilbert_solve(data=data, p=p)

    # Assert - result should be a permutation of indices
    assert len(result) == len(data)
    assert set(result) == set(range(len(data)))


@pytest.mark.parametrize(
    ("used", "choices", "world_size", "expected_result"),
    [
        # First element not in used is returned
        ([], [0, 1, 2], 10, 0),
        # First element is used, return second
        ([0], [0, 1, 2], 10, 1),
        # First two elements used, return third
        ([0, 1], [0, 1, 2], 10, 2),
        # All elements are used, return None
        ([0, 1, 2], [0, 1, 2], 10, None),
        # First available element is -1, return None
        ([0], [0, -1, 2], 10, None),
        # -1 at start and not in used returns None
        ([], [-1, 1, 2], 10, None),
        # Skip used elements to find first available
        ([0, 2], [0, 2, 1, 3], 10, 1),
        # Empty choices returns None
        ([0], [], 10, None),
    ],
    ids=(
        "first_available",
        "skip_one_used",
        "skip_two_used",
        "all_used",
        "minus_one_element",
        "minus_one_first",
        "non_sequential_choices",
        "empty_choices",
    ),
)
def test__get_closest(used: list[int], choices: list[int], world_size: int, expected_result: int | None) -> None:
    """Test `_get_closest` for finding first unused element from choices"""
    # Act
    result = _get_closest(used=used, choices=choices, world_size=world_size)

    # Assert
    assert result == expected_result


# TODO: Rename test when fixing function name
# Note: _tps_greedy_solve and tsp_greedy_solve require umap dependency and are tested in integration tests


@pytest.mark.parametrize(
    ("data", "sorted_data", "expected_nodes"),
    [
        # Single element
        (np.array([[0, 0]]), np.array([[0, 0]]), [0]),
        # Already sorted - indices are 0, 1, 2
        (
            np.array([[0, 0], [1, 1], [2, 2]]),
            np.array([[0, 0], [1, 1], [2, 2]]),
            [0, 1, 2],
        ),
        # Reversed order - indices map sorted back to original positions
        (
            np.array([[2, 2], [1, 1], [0, 0]]),
            np.array([[0, 0], [1, 1], [2, 2]]),
            [2, 1, 0],
        ),
        # Arbitrary reordering
        (
            np.array([[1, 0], [0, 1], [2, 2], [0, 0]]),
            np.array([[0, 0], [1, 0], [0, 1], [2, 2]]),
            [3, 0, 1, 2],
        ),
    ],
    ids=("single_element", "already_sorted", "reversed", "arbitrary_reorder"),
)
def test__get_nodes(data: np.ndarray, sorted_data: np.ndarray, expected_nodes: list[int]) -> None:
    """Test `_get_nodes` for calculating index array mapping sorted to original data"""
    result = _get_nodes(data=data, sorted_data=sorted_data)

    assert list(result) == expected_nodes


@pytest.mark.parametrize(
    ("hilbert_points", "data_rounded", "expected_order"),
    [
        # Single point
        (np.array([[0, 0]]), np.array([[0, 0]]), np.array([0])),
        # Two points in hilbert order
        (np.array([[0, 0], [1, 0]]), np.array([[0, 0], [1, 0]]), np.array([0, 1])),
        # Two points in reverse hilbert order - order reflects hilbert traversal
        (np.array([[0, 0], [1, 0]]), np.array([[1, 0], [0, 0]]), np.array([1, 0])),
        # Three points with reordering
        (
            np.array([[0, 0], [0, 1], [1, 1]]),
            np.array([[1, 1], [0, 0], [0, 1]]),
            np.array([1, 2, 0]),
        ),
        # Points not all on hilbert curve - only matching points are ordered
        (
            np.array([[0, 0], [1, 0], [2, 0]]),
            np.array([[0, 0], [2, 0]]),
            np.array([0, 1]),
        ),
    ],
    ids=(
        "single_point",
        "two_points_in_order",
        "two_points_reversed",
        "three_points_reorder",
        "partial_match",
    ),
)
def test_assign_vertices(hilbert_points: np.ndarray, data_rounded: np.ndarray, expected_order: np.ndarray) -> None:
    """Test `assign_vertices` for ordering data points along hilbert curve"""
    result = assign_vertices(hilbert_points=hilbert_points, data_rounded=data_rounded)

    assert np.array_equal(result[: len(expected_order)], expected_order)


@pytest.mark.parametrize(
    ("array", "expected_result"),
    [
        (np.array([[0], [1], [2]]), 2),
        (np.array([[0, 0], [1, 0], [2, 0]]), 2),
        (np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]), 4),
    ],
    ids=("1d_like", "linear", "square_circular_path"),
)
def test_calc_len(array: np.ndarray, expected_result: float) -> None:
    """Test `calc_len` for the computation of a path length"""
    result = calc_len(array)

    assert result == expected_result
