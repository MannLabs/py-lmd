from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from lmd import tools
from lmd.lib import Collection

ATOL = 1e-10


@pytest.mark.parametrize(
    ("angle_rad", "expected_matrix"),
    [
        (0, np.array([[1, 0], [0, 1]])),
        (np.pi / 4, np.array([[1 / np.sqrt(2), -1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2)]])),
        (np.pi / 2, np.array([[0, -1], [1, 0]])),
        (2 * np.pi, np.array([[1, 0], [0, 1]])),
        (-1 * np.pi / 2, np.array([[0, 1], [-1, 0]])),
    ],
    ids=("identity", "45_degrees", "90_degrees", "identity_360_degrees", "-90_degrees"),
)
def test__get_rotation_matrix(angle_rad: float, expected_matrix: np.ndarray) -> None:
    """Test that function returns a correct affine transformation matrix, given an eangle in rad"""
    result = tools._get_rotation_matrix(angle=angle_rad)

    assert np.allclose(result, expected_matrix, atol=ATOL)


class TestGlyphPath:
    @pytest.mark.parametrize(("glyph_string",), argvalues=list("013456789abcdefghiABCDEFGHI"))
    def test_glyph_path(self, glyph_string: str) -> None:
        """Test that implemented glyphs exist (only glyphs A-I implemented)"""
        result = tools.glyph_path(glyph=glyph_string)

        assert Path(result).exists()

    @pytest.mark.parametrize(("glyph_string",), argvalues=[("X",)])
    def test_glyph_path__raises(self, glyph_string: str) -> None:
        """Test that implemented glyphs exist (only glyphs A-I implemented)"""
        with pytest.raises(NotImplementedError, match="This has not been implemented yet."):
            _ = tools.glyph_path(glyph=glyph_string)


class TestGlyph:
    @pytest.mark.parametrize(("glyph_string",), argvalues=list("013456789abcdefghiABCDEFGHI"))
    def test_glyph(self, glyph_string: str) -> None:
        """Test that collection is created from a glyph"""

        result = tools.glyph(glyph=glyph_string)

        assert isinstance(result, Collection)
        assert len(result.shapes) >= 1
        assert result.calibration_points is None


class TestText:
    @pytest.mark.parametrize(("text_string",), argvalues=[("a",), ("abc",)])
    def test_text(self, text_string: str) -> None:
        """Test that function returns a collection"""
        result = tools.text(text=text_string)

        assert isinstance(result, Collection)
        # Text consists of multiple glyphs. Each glyph consists of at least one shape
        assert len(result.shapes) >= len(text_string)
        assert result.calibration_points is None


class TestRectangle:
    @pytest.mark.parametrize(
        ("width", "height", "expected_points"),
        [
            (1, 1, np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])),
            (2, 1, np.array([[-1.0, -0.5], [-1.0, 0.5], [1.0, 0.5], [1.0, -0.5], [-1.0, -0.5]])),
        ],
        ids=("unit_square", "rectangle"),
    )
    def test_rectangle(self, width: float, height: float, expected_points: np.ndarray) -> None:
        shape = tools.rectangle(width=width, height=height, offset=0, rotation=0, rotation_offset=0)

        assert np.array_equal(shape.points, expected_points)

    @pytest.mark.parametrize(
        ("offset", "expected_points"),
        [
            ((0, 0), np.array([[-1.0, -0.5], [-1.0, 0.5], [1.0, 0.5], [1.0, -0.5], [-1.0, -0.5]])),
            ((1, 1), np.array([[-1.0, -0.5], [-1.0, 0.5], [1.0, 0.5], [1.0, -0.5], [-1.0, -0.5]]) + np.array([1, 1])),
            ((2, 1), np.array([[-1.0, -0.5], [-1.0, 0.5], [1.0, 0.5], [1.0, -0.5], [-1.0, -0.5]]) + np.array([2, 1])),
            ((-1, 0), np.array([[-1.0, -0.5], [-1.0, 0.5], [1.0, 0.5], [1.0, -0.5], [-1.0, -0.5]]) + np.array([-1, 0])),
        ],
        ids=("zero_offset", "diagonal_offset", "unequal_offset", "negative_offset"),
    )
    def test_rectangle__offset(self, offset: tuple[int, int], expected_points: np.ndarray) -> None:
        """Test that offset works as expected"""
        shape = tools.rectangle(width=2.0, height=1.0, offset=offset, rotation=0, rotation_offset=0)

        assert np.array_equal(shape.points, expected_points)

    @pytest.mark.parametrize(
        ("rotation", "expected_points"),
        [
            (0, np.array([[-1.0, -0.5], [-1.0, 0.5], [1.0, 0.5], [1.0, -0.5], [-1.0, -0.5]])),
            (
                np.pi / 4,
                np.array(
                    [
                        [-1.06066017, 0.35355339],
                        [-0.35355339, 1.06066017],
                        [1.06066017, -0.35355339],
                        [0.35355339, -1.06066017],
                        [-1.06066017, 0.35355339],
                    ]
                ),
            ),
            (np.pi / 2, np.array([[-0.5, 1.0], [0.5, 1.0], [0.5, -1.0], [-0.5, -1.0], [-0.5, 1.0]])),
            (np.pi, np.array([[1.0, 0.5], [1.0, -0.5], [-1.0, -0.5], [-1.0, 0.5], [1.0, 0.5]])),
        ],
        ids=(
            "no_rotation",
            "45_degrees",
            "90_degrees",
            "180_degrees",
        ),
    )
    def test_rectangle__rotation(self, rotation: float, expected_points: np.ndarray) -> None:
        shape = tools.rectangle(width=2.0, height=1.0, offset=(0, 0), rotation=rotation, rotation_offset=0)

        assert np.allclose(shape.points, expected_points, atol=ATOL)

    # TODO: Understand non-zero offsets geometrically
    @pytest.mark.parametrize(
        ("rotation_offset", "expected_points"),
        [
            ((0, 0), np.array([[-1.0, 1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])),
        ],
        ids=("zero_offset",),
    )
    def test_rectangle__rotation_offset(self, rotation_offset: tuple[int, int], expected_points: np.ndarray) -> None:
        """Test that rotation offset is correctly performed."""
        shape = tools.rectangle(
            width=2.0, height=2.0, offset=(0, 0), rotation=np.pi / 2, rotation_offset=rotation_offset
        )

        assert np.allclose(shape.points, expected_points, atol=ATOL)


class TestEllipse:
    @pytest.mark.parametrize(
        ("major_axis", "minor_axis"),
        [(1, 1), (2, 1), (100, 1)],
        ids=("circle", "ellipse", "wide_ellipse"),
    )
    def test_ellipse(self, major_axis: float, minor_axis: float) -> None:
        shape = tools.ellipse(major_axis=major_axis, minor_axis=minor_axis, offset=0, rotation=0, polygon_resolution=1)

        # Validate that points follow standard ellipsis equation: x^2/a^2 + y^2/b^2 = 1
        curve = shape.points[:, 0] ** 2 / major_axis**2 + shape.points[:, 1] ** 2 / minor_axis**2
        expected_result = np.ones_like(curve)

        assert np.allclose(curve, expected_result, rtol=0.01)

    @pytest.mark.parametrize(
        ("offset",),
        [((0, 0),), ((1, 1),), ((2, 1),), ((-1, 0),)],
        ids=("zero_offset", "diagonal_offset", "unequal_offset", "negative_offset"),
    )
    def test_ellipse__offset(self, offset: tuple[int, int]) -> None:
        major_axis = 2
        minor_axis = 1
        shape = tools.ellipse(
            major_axis=major_axis, minor_axis=minor_axis, offset=offset, rotation=0, polygon_resolution=1
        )
        centered_points = shape.points - np.array(offset)

        # Validate that points follow standard ellipsis equation: x^2/a^2 + y^2/b^2 = 1
        curve = centered_points[:, 0] ** 2 / major_axis**2 + centered_points[:, 1] ** 2 / minor_axis**2
        expected_result = np.ones_like(curve)

        assert np.allclose(curve, expected_result, rtol=0.01)

    # TODO: Add case with negative values (currently passes)
    @pytest.mark.parametrize("incorrect_resolution", [0.0])
    def test_ellipse__raises(self, incorrect_resolution: float) -> None:
        with pytest.raises(ValueError, match="Polygon resolution has to be larger than 0"):
            _ = tools.ellipse(
                major_axis=2, minor_axis=1, offset=(0, 0), rotation=0, polygon_resolution=incorrect_resolution
            )


class TestMakeCross:
    def test_makeCross(self) -> None:
        """Test that collection"""
        cross_1 = tools.makeCross([20, 20], [50, 50, 50, 50], 1, 10)
        assert isinstance(cross_1, Collection)
        assert len(cross_1.shapes) == 5  # 4 arms + centroid
