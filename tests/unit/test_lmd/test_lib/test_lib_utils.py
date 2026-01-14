import operator
from typing import Optional

import numpy as np
import pytest

from lmd.lib import _create_poly, _execute_indexed_parallel, _sort_edges, transform_to_map


@pytest.mark.parametrize("tqdm_args", [None, {}])
def test__execute_indexed_parallel(tqdm_args: Optional[dict]) -> None:
    """Test parallelized execution"""
    values = range(10)
    result = _execute_indexed_parallel(operator.mul, args=[[i, 2] for i in values], tqdm_args=tqdm_args, n_threads=2)

    assert result == [2 * value for value in values]


@pytest.fixture
def square_mask() -> np.ndarray:
    """Create a simple 10x10 square mask"""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    return mask


def test__create_poly_returns_polygon(square_mask: np.ndarray) -> None:
    """Test that _create_poly returns a polygon from a binary mask"""
    offset = np.array([0, 0])
    in_tuple = (square_mask, offset)

    result = _create_poly(in_tuple, smoothing_filter_size=3, rdp_epsilon=0)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape[1] == 2  # Each point has x, y coordinates
    assert len(result) >= 4  # At least 4 points for a quadrilateral


def test__create_poly_applies_offset(square_mask: np.ndarray) -> None:
    """Test that _create_poly correctly applies the offset"""
    offset = np.array([100, 200])
    in_tuple = (square_mask, offset)

    result = _create_poly(in_tuple, smoothing_filter_size=3, rdp_epsilon=0)

    assert np.all(result[:, 0] >= offset[0])
    assert np.all(result[:, 1] >= offset[1])


def test__create_poly_rdp_epsilon_reduces_points(square_mask: np.ndarray) -> None:
    """Test that higher rdp_epsilon results in fewer polygon points"""
    offset = np.array([0, 0])
    in_tuple = (square_mask, offset)

    result_no_compression = _create_poly(in_tuple, smoothing_filter_size=3, rdp_epsilon=0)
    result_with_compression = _create_poly(in_tuple, smoothing_filter_size=3, rdp_epsilon=1.0)

    assert len(result_with_compression) <= len(result_no_compression)


@pytest.mark.parametrize(
    ("edges", "expected_result"),
    [
        # Already sorted - nearest neighbor from each point is the next one
        (
            np.array([[0, 0], [1, 0], [2, 0]]),
            np.array([[0, 0], [1, 0], [2, 0]]),
        ),
        # Reversed order - greedy sort starts from first element
        (
            np.array([[0, 0], [2, 0], [1, 0]]),
            np.array([[0, 0], [1, 0], [2, 0]]),
        ),
        # Square vertices in scrambled order
        (
            np.array([[0, 0], [1, 1], [1, 0], [0, 1]]),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
        ),
        # Two points - minimal case
        (
            np.array([[0, 0], [1, 1]]),
            np.array([[0, 0], [1, 1]]),
        ),
    ],
    ids=("already_sorted", "reversed_middle", "square_scrambled", "minimal"),
)
def test__sort_edges(edges: np.ndarray, expected_result: np.ndarray) -> None:
    """Test `_sort_edges` for greedy nearest-neighbor sorting of polygon vertices"""
    result = _sort_edges(edges)

    assert np.array_equal(result, expected_result)


class TestTransformToMap:
    @pytest.mark.parametrize(
        ("coords",),
        [(np.array([[0, 0]]),), (np.array([[2, 1]]),), (np.array([[0, 0], [2, 1]]),)],
        ids=("simple", "assymmetric", "multiple_coords"),
    )
    def test_transform_to_map__no_changes__coord_format(self, coords: np.ndarray) -> None:
        """Test that `coord_format=True` just returns the coords if no dilation/erosion is applied"""
        # Essentially identity transform
        result = transform_to_map(coords=coords, dilation=0, erosion=0, coord_format=True, debug=False)

        assert np.array_equal(result, coords)

    @pytest.mark.parametrize(
        ("coords", "expected_shape", "expected_offsets", "sparse_coords_result_mask"),
        [
            (np.array([[0, 0]]), (6, 6), np.array([0, 0]), np.array([[0, 0]])),  # minimal size with buffer
            (np.array([[2, 1]]), (8, 7), np.array([0, 0]), np.array([[2, 1]])),
            (np.array([[0, 0], [2, 1]]), (8, 7), np.array([0, 0]), np.array([[0, 0], [2, 1]])),
            (np.array([[100, 100]]), (9, 9), np.array([97, 97]), np.array([[3, 3]])),  # boundary left/right
            (
                np.array([[0, 0], [100, 100]]),
                (106, 106),
                np.array([0, 0]),
                np.array([[0, 0], [100, 100]]),
            ),  # boundary left/right, no offset
        ],
        ids=("simple", "assymmetric", "multiple_coords", "with_offsets", "large_without_offsets"),
    )
    def test_transform_to_map__no_changes__mask_format(
        self,
        coords: np.ndarray,
        expected_shape: np.ndarray,
        expected_offsets: tuple[int, int],
        sparse_coords_result_mask: np.ndarray,
    ) -> None:
        """Test that `coord_format=False` returns a binary mask with the indicated coords if no dilation/erosion is applied"""
        # Build expected results mask from sparse coords (N, 2)
        expected_mask = np.zeros(shape=expected_shape)
        for dim0_index, dim1_index in sparse_coords_result_mask:
            expected_mask[dim0_index, dim1_index] = 1

        binary_mask, offset = transform_to_map(coords=coords, dilation=0, erosion=0, coord_format=False, debug=False)

        assert binary_mask.shape == expected_shape
        assert np.allclose(offset, expected_offsets)
        assert np.allclose(binary_mask, expected_mask)
