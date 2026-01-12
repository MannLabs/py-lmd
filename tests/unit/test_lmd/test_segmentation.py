import numpy as np
import pytest

from lmd.segmentation import (
    _create_coord_index,
    _create_coord_index_sparse,
    _filter_coord_index,
    _get_closest,
    _get_nodes,
    _numba_accelerator_coord_calculation,
    assign_vertices,
    calc_len,
    get_coordinate_form,
    tsp_hilbert_solve,
)

ATOL = 1e-10


@pytest.mark.parametrize(
    ("_ids", "inverse_indices", "sparse_coords_0", "sparse_coords_1", "expected_result"),
    [
        (np.array([1]), np.array([0]), np.array([1]), np.array([1]), {1: np.array([[1, 1]])}),
        (
            np.array([1, 2]),
            np.array([0, 1, 0]),
            np.array([1, 2, 3]),
            np.array([1, 2, 3]),
            {1: np.array([[1, 1], [3, 3]]), 2: np.array([[2, 2]])},
        ),
        (
            np.array([1], dtype=np.uint64),
            np.zeros(shape=(33,), dtype=np.uint64),
            np.arange(33, dtype=np.uint64),
            np.arange(33, dtype=np.uint64),
            {1: np.stack([np.arange(33), np.arange(33)], axis=-1)},
        ),
    ],
    ids=("minimal", "non_consecutive_indices", "extend_array"),
)
def test__numba_accelerator_coord_calculation(
    _ids: np.ndarray,
    inverse_indices: np.ndarray,
    sparse_coords_0: np.ndarray,
    sparse_coords_1: np.ndarray,
    expected_result: dict[int, np.ndarray],
) -> None:
    """Test `_numba_accelerator_coord_calculation`

    Parameters
    ----------
    _ids
        cell_ids / Unique Cell IDs
    inverse_indices
        cell_ids[idx] represents the cell ID
    sparse_coords_0
        x0 coordinate of cell at the same index in inverse_indices
    sparse_coords_1
        x1 coordinate of cell at the same index in inverse_indices
    expected_result
    """
    assert len(_ids) == len(np.unique(inverse_indices))
    assert len(inverse_indices) == len(sparse_coords_0)
    assert len(sparse_coords_0) == len(sparse_coords_1)

    result = _numba_accelerator_coord_calculation(
        _ids=_ids, inverse_indices=inverse_indices, sparse_coords_0=sparse_coords_0, sparse_coords_1=sparse_coords_1
    )

    # Returns numba.typed.typeddict.Dict
    result = dict(result)

    assert result.keys() == expected_result.keys()
    assert all(
        np.allclose(result_values, expected_result_values, atol=ATOL)
        for result_values, expected_result_values in zip(result.values(), expected_result.values())
    ), [
        np.array_equal(result_values, expected_result_values)
        for result_values, expected_result_values in zip(result.values(), expected_result.values())
    ]


@pytest.mark.parametrize(
    ("mask", "expected_result"),
    [
        (np.array([[1]], dtype=np.uint64), {1: np.array([[0, 0]], dtype=np.uint64)}),
        (np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.uint64), {1: np.array([[0, 2]], dtype=np.uint64)}),
        (np.array([[0, 0, 2], [0, 0, 0], [0, 0, 0]], dtype=np.uint64), {2: np.array([[0, 2]], dtype=np.uint64)}),
        (
            np.array([[1, 0, 2], [0, 0, 0], [0, 0, 0]], dtype=np.uint64),
            {1: np.array([[0, 0]], dtype=np.uint64), 2: np.array([[0, 2]], dtype=np.uint64)},
        ),
    ],
    ids=("minimal", "larger", "non_consecutive_ids", "multiple_ids"),
)
def test__create_coord_index_sparse(mask: np.ndarray, expected_result: dict[str, np.ndarray]) -> None:
    result = _create_coord_index_sparse(mask=mask)

    assert result.keys() == expected_result.keys()
    assert all(
        np.allclose(result_values, expected_result_values, atol=ATOL)
        for result_values, expected_result_values in zip(result.values(), expected_result.values())
    ), [
        np.allclose(result_values, expected_result_values)
        for result_values, expected_result_values in zip(result.values(), expected_result.values())
    ]


# TODO: Remove unused background argument
@pytest.mark.parametrize("background", [0])
@pytest.mark.parametrize(
    ("mask", "expected_result"),
    [
        (np.array([[0]], dtype=np.uint64), {0: np.array([[0, 0]], dtype=np.uint64)}),
        (
            np.array([[0, 1], [0, 0]], dtype=np.uint64),
            {0: np.array([[0, 0], [1, 0], [1, 1]]), 1: np.array([[0, 1]], dtype=np.uint64)},
        ),
        # TODO: This fails ("non_consecutive_ids")
        # (
        #     np.array([[0, 2], [0, 0]], dtype=np.uint64),
        #     {0: np.array([[0, 0], [1, 0], [1, 1]]), 2: np.array([[0, 1]], dtype=np.uint64)},
        # ),
        (
            np.array([[1, 0, 2]], dtype=np.uint64),
            {0: np.array([[0, 1]]), 1: np.array([[0, 0]], dtype=np.uint64), 2: np.array([[0, 2]], dtype=np.uint64)},
        ),
        (
            np.zeros(shape=(33, 1), dtype=np.uint64),
            {0: np.stack([np.arange(33), np.zeros(33)], axis=-1)},
        ),
    ],
    ids=("minimal", "larger", "multiple_ids", "extend_array"),
)
def test__create_coord_index(mask: np.ndarray, expected_result: dict[int, np.ndarray], background: int) -> None:
    result = _create_coord_index(mask=mask, background=background)
    # Returns numba.typed.typeddict.Dict
    result = dict(result)

    assert result.keys() == expected_result.keys(), result.keys()
    assert all(
        np.allclose(result_values, expected_result_values, atol=ATOL)
        for result_values, expected_result_values in zip(result.values(), expected_result.values())
    ), [
        np.allclose(result_values, expected_result_values)
        for result_values, expected_result_values in zip(result.values(), expected_result.values())
    ]


# TODO: Remove unused argument
@pytest.mark.parametrize("background", [0])
@pytest.mark.parametrize(
    ("mask", "classes", "expected_result"),
    [
        (np.array([[0]], dtype=np.uint64), np.array([0]), {0: np.array([[0, 0]], dtype=np.uint64)}),
        (
            np.ones(shape=(33, 1), dtype=np.uint64),
            np.array([1]),
            {1: np.stack([np.arange(33), np.zeros(33)], axis=-1)},
        ),
    ],
    ids=("minimal", "extend_array"),
)
def test__create_coord_index__classes(
    mask: np.ndarray, background: int, classes: np.ndarray, expected_result: dict[int, np.ndarray]
) -> None:
    result = _create_coord_index(mask=mask, background=background, classes=classes)
    # Returns numba.typed.typeddict.Dict
    result = dict(result)

    assert result.keys() == expected_result.keys(), result.keys()
    assert all(
        np.allclose(result_values, expected_result_values, atol=ATOL)
        for result_values, expected_result_values in zip(result.values(), expected_result.values())
    ), [
        np.allclose(result_values, expected_result_values)
        for result_values, expected_result_values in zip(result.values(), expected_result.values())
    ]


@pytest.mark.parametrize(
    ("index_list", "classes", "background", "expected_result"),
    [
        ({0: np.array([[0, 0]])}, np.array([0]), 0, []),
        ({1: np.array([[0, 0]])}, np.array([0]), 0, np.array([[0, 0]])),
        ({1: np.array([[0, 0]]), 2: np.array([[]])}, np.array([1, 2]), 0, np.array([[0, 0]])),
    ],
    ids=("only_background", "single_shape", "contains_empty"),
)
def test__filter_coord_index(
    index_list: dict[int, np.ndarray], classes: np.ndarray, background: int, expected_result: list[np.ndarray]
) -> None:
    result = _filter_coord_index(index_list=index_list, classes=classes, background=background)
    assert all(np.allclose(res, expected_res) for res, expected_res in zip(result, expected_result))


@pytest.mark.parametrize(
    ("classes", "coords_lookup", "expected_center", "expected_length", "expected_coords"),
    [
        # Empty classes returns empty lists
        (np.array([]), {1: np.array([[0, 0]])}, [], [], []),
        # Single class with single coordinate
        (
            np.array([1]),
            {1: np.array([[2, 4]])},
            [np.array([2.0, 4.0])],
            [1],
            [np.array([[2, 4]])],
        ),
        # Single class with multiple coordinates - center is mean
        (
            np.array([1]),
            {1: np.array([[0, 0], [2, 2]])},
            [np.array([1.0, 1.0])],
            [2],
            [np.array([[0, 0], [2, 2]])],
        ),
        # Multiple classes
        (
            np.array([1, 2]),
            {1: np.array([[0, 0], [2, 0]]), 2: np.array([[4, 4]])},
            [np.array([1.0, 0.0]), np.array([4.0, 4.0])],
            [2, 1],
            [np.array([[0, 0], [2, 0]]), np.array([[4, 4]])],
        ),
    ],
    ids=("empty_classes", "single_class_single_coord", "single_class_multiple_coords", "multiple_classes"),
)
def test_get_coordinate_form(
    classes: np.ndarray,
    coords_lookup: dict[int, np.ndarray],
    expected_center: list[np.ndarray],
    expected_length: list[int],
    expected_coords: list[np.ndarray],
) -> None:
    """Test `get_coordinate_form` for returning center, length, and filtered coordinates"""
    # Act
    center, length, coords_filtered = get_coordinate_form(classes, coords_lookup)

    # Assert
    assert len(center) == len(expected_center)
    assert length == expected_length
    assert len(coords_filtered) == len(expected_coords)

    for c, ec in zip(center, expected_center):
        assert np.allclose(c, ec, atol=ATOL)

    for cf, ecf in zip(coords_filtered, expected_coords):
        assert np.allclose(cf, ecf, atol=ATOL)


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
