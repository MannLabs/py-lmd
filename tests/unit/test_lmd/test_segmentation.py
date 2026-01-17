import warnings

import numpy as np
import pytest

from lmd.segmentation import (
    _create_coord_index,
    _create_coord_index_sparse,
    _filter_coord_index,
    _numba_accelerator_coord_calculation,
    get_coordinate_form,
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
        (np.array([[0]], dtype=np.uint64), np.array([0], dtype=np.uint64), {0: np.array([[0, 0]], dtype=np.uint64)}),
        (
            np.ones(shape=(33, 1), dtype=np.uint64),
            np.array([1], dtype=np.uint64),
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


# =============================================================================
# Deprecation Warning Tests
# =============================================================================
# These tests verify that importing path functions from lmd.segmentation
# raises deprecation warnings. The actual function tests are in test_path.py.


class TestDeprecationWarnings:
    """Test that deprecated imports from lmd.segmentation raise DeprecationWarning."""

    def test_calc_len_deprecation_warning(self) -> None:
        """Test that calc_len raises DeprecationWarning when imported from segmentation."""
        from lmd.segmentation import calc_len

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calc_len(np.array([[0, 0], [1, 1]]))

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "calc_len has been moved to lmd.path" in str(w[0].message)

    def test_tsp_hilbert_solve_deprecation_warning(self) -> None:
        """Test that tsp_hilbert_solve raises DeprecationWarning when imported from segmentation."""
        from lmd.segmentation import tsp_hilbert_solve

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tsp_hilbert_solve(np.array([[0.0, 0.0], [1.0, 1.0]]), p=3)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "tsp_hilbert_solve has been moved to lmd.path" in str(w[0].message)

    def test_tsp_greedy_solve_deprecation_warning(self) -> None:
        """Test that tsp_greedy_solve raises DeprecationWarning when imported from segmentation."""
        from lmd.segmentation import tsp_greedy_solve

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Note: tsp_greedy_solve requires umap, so we just check the warning is raised
            # before the actual call would happen
            tsp_greedy_solve(np.array([[0.0, 0.0], [1.0, 1.0]]))

            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "tsp_greedy_solve has been moved to lmd.path" in str(w[0].message)

    def test_assign_vertices_deprecation_warning(self) -> None:
        """Test that assign_vertices raises DeprecationWarning when imported from segmentation."""
        from lmd.segmentation import assign_vertices

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assign_vertices(np.array([[0, 0]]), np.array([[0, 0]]))

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "assign_vertices has been moved to lmd.path" in str(w[0].message)

    def test__get_closest_deprecation_warning(self) -> None:
        """Test that _get_closest raises DeprecationWarning when imported from segmentation."""
        from lmd.segmentation import _get_closest

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _get_closest([], [0, 1, 2], 10)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "_get_closest has been moved to lmd.path" in str(w[0].message)

    def test__get_nodes_deprecation_warning(self) -> None:
        """Test that _get_nodes raises DeprecationWarning when imported from segmentation."""
        from lmd.segmentation import _get_nodes

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _get_nodes(np.array([[0, 0]]), np.array([[0, 0]]))

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "_get_nodes has been moved to lmd.path" in str(w[0].message)

    def test__tps_greedy_solve_deprecation_warning(self) -> None:
        """Test that _tps_greedy_solve raises DeprecationWarning when imported from segmentation."""
        from lmd.segmentation import _tps_greedy_solve

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            _tps_greedy_solve(np.array([[0.0, 0.0], [1.0, 1.0]]))

            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "_tps_greedy_solve has been moved to lmd.path" in str(w[0].message)
