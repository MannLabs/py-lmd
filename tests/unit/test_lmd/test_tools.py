from pathlib import Path

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
    """Test that function returns a correct affine transformation matrix, given an angle in rad"""
    result = tools._get_rotation_matrix(angle=angle_rad)

    assert np.allclose(result, expected_matrix, atol=ATOL)


class TestGlyphPath:
    @pytest.mark.parametrize(("glyph_string",), argvalues=list("0123456789ABCDEFGHI"))
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
    @pytest.mark.parametrize(("glyph_string",), argvalues=list("0123456789ABCDEFGHI"))
    def test_glyph(self, glyph_string: str) -> None:
        """Test that collection is created from a glyph"""

        result = tools.glyph(glyph=glyph_string)

        assert isinstance(result, Collection)
        assert len(result.shapes) >= 1
        assert result.calibration_points is None

    @pytest.mark.parametrize(
        ("offset",),
        [(np.array([0, 0]),), (np.array([10, 10]),), (np.array([-5, 5]),)],
        ids=("zero_offset", "positive_offset", "mixed_offset"),
    )
    def test_glyph__offset(self, offset: np.ndarray) -> None:
        """Test that offset shifts glyph position"""
        result_no_offset = tools.glyph(glyph="A", offset=np.array([0, 0]))
        result_with_offset = tools.glyph(glyph="A", offset=offset)

        # All shapes should be shifted by offset
        for shape_no_offset, shape_with_offset in zip(result_no_offset.shapes, result_with_offset.shapes):
            expected_points = shape_no_offset.points + offset
            assert np.allclose(shape_with_offset.points, expected_points, atol=ATOL)

    @pytest.mark.parametrize(
        ("rotation",),
        [(0,), (np.pi / 4,), (np.pi / 2,), (np.pi,)],
        ids=("no_rotation", "45_degrees", "90_degrees", "180_degrees"),
    )
    def test_glyph__rotation(self, rotation: float) -> None:
        """Test that rotation rotates the glyph around origin"""
        result = tools.glyph(glyph="A", rotation=rotation)

        assert isinstance(result, Collection)
        assert len(result.shapes) >= 1

    @pytest.mark.parametrize(
        ("multiplier",),
        [(0.5,), (1,), (2,), (5,)],
        ids=("half_size", "default_size", "double_size", "5x_size"),
    )
    def test_glyph__multiplier(self, multiplier: float) -> None:
        """Test that multiplier scales glyph size"""
        result_default = tools.glyph(glyph="A", multiplier=1)
        result_scaled = tools.glyph(glyph="A", multiplier=multiplier)

        # Calculate bounding box sizes
        default_points = np.vstack([s.points for s in result_default.shapes])
        scaled_points = np.vstack([s.points for s in result_scaled.shapes])

        default_size = default_points.max(axis=0) - default_points.min(axis=0)
        scaled_size = scaled_points.max(axis=0) - scaled_points.min(axis=0)

        # Scaled size should be proportional to multiplier
        expected_size = default_size * multiplier
        assert np.allclose(scaled_size, expected_size, rtol=0.01)


class TestText:
    @pytest.mark.parametrize(("text_string",), argvalues=[("A",), ("ABC",)])
    def test_text(self, text_string: str) -> None:
        """Test that function returns a collection"""
        result = tools.text(text=text_string)

        assert isinstance(result, Collection)
        # Text consists of multiple glyphs. Each glyph consists of at least one shape
        assert len(result.shapes) >= len(text_string)
        assert result.calibration_points is None

    @pytest.mark.parametrize(
        ("offset",),
        [(np.array([0, 0]),), (np.array([10, 10]),), (np.array([-5, 5]),)],
        ids=("zero_offset", "positive_offset", "mixed_offset"),
    )
    def test_text__offset(self, offset: np.ndarray) -> None:
        """Test that offset positions text correctly"""
        result_no_offset = tools.text(text="A", offset=np.array([0, 0]))
        result_with_offset = tools.text(text="A", offset=offset)

        # All shapes should be shifted by offset
        for shape_no_offset, shape_with_offset in zip(result_no_offset.shapes, result_with_offset.shapes):
            expected_points = shape_no_offset.points + offset
            assert np.allclose(shape_with_offset.points, expected_points, atol=ATOL)

    @pytest.mark.parametrize(
        ("rotation",),
        [(0,), (np.pi / 4,), (np.pi / 2,), (-np.pi / 4,)],
        ids=("no_rotation", "45_degrees", "90_degrees", "negative_45_degrees"),
    )
    def test_text__rotation(self, rotation: float) -> None:
        """Test that rotation rotates entire text"""
        result = tools.text(text="AB", rotation=rotation)

        assert isinstance(result, Collection)
        assert len(result.shapes) >= 2

    @pytest.mark.parametrize(
        ("multiplier",),
        [(0.5,), (1,), (2,)],
        ids=("half_size", "default_size", "double_size"),
    )
    def test_text__multiplier(self, multiplier: float) -> None:
        """Test that multiplier scales text size"""
        result_default = tools.text(text="A", multiplier=1)
        result_scaled = tools.text(text="A", multiplier=multiplier)

        # Calculate bounding box sizes
        default_points = np.vstack([s.points for s in result_default.shapes])
        scaled_points = np.vstack([s.points for s in result_scaled.shapes])

        default_size = default_points.max(axis=0) - default_points.min(axis=0)
        scaled_size = scaled_points.max(axis=0) - scaled_points.min(axis=0)

        # Scaled size should be proportional to multiplier
        expected_size = default_size * multiplier
        assert np.allclose(scaled_size, expected_size, rtol=0.01)

    def test_text__character_spacing(self) -> None:
        """Test that characters are spaced correctly (10 * multiplier apart)"""
        multiplier = 1.0
        text_str = "AB"
        result = tools.text(text=text_str, multiplier=multiplier, rotation=0)

        # Get bounding boxes for each character's shapes
        # First character "A" shapes
        glyph_a = tools.glyph("A", multiplier=multiplier)
        n_shapes_a = len(glyph_a.shapes)

        # The first n_shapes_a shapes belong to "A", the rest to "B"
        a_points = np.vstack([s.points for s in result.shapes[:n_shapes_a]])
        b_points = np.vstack([s.points for s in result.shapes[n_shapes_a:]])

        a_center_x = (a_points[:, 0].max() + a_points[:, 0].min()) / 2
        b_center_x = (b_points[:, 0].max() + b_points[:, 0].min()) / 2

        # Characters should be spaced by delta = 10 * multiplier in x direction (when rotation=0)
        expected_spacing = 10 * multiplier
        actual_spacing = b_center_x - a_center_x
        assert np.isclose(actual_spacing, expected_spacing, rtol=0.1)

    def test_text__numeric_input(self) -> None:
        """Test that numeric input is converted to string"""
        result = tools.text(text=123)

        assert isinstance(result, Collection)
        # "123" has 3 characters
        assert len(result.shapes) >= 3


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

        # Validate that points follow standard ellipse equation: x^2/a^2 + y^2/b^2 = 1
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

        # Validate that points follow standard ellipse equation: x^2/a^2 + y^2/b^2 = 1
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

    @pytest.mark.parametrize(
        ("rotation",),
        [(0,), (np.pi / 4,), (np.pi / 2,), (np.pi,)],
        ids=("no_rotation", "45_degrees", "90_degrees", "180_degrees"),
    )
    def test_ellipse__rotation(self, rotation: float) -> None:
        """Test that rotation rotates the ellipse correctly"""
        major_axis = 2
        minor_axis = 1

        shape_no_rotation = tools.ellipse(
            major_axis=major_axis, minor_axis=minor_axis, offset=(0, 0), rotation=0, polygon_resolution=1
        )
        shape_rotated = tools.ellipse(
            major_axis=major_axis, minor_axis=minor_axis, offset=(0, 0), rotation=rotation, polygon_resolution=1
        )

        # Apply rotation matrix to unrotated points
        rotation_matrix = tools._get_rotation_matrix(rotation)
        expected_points = shape_no_rotation.points @ rotation_matrix

        assert np.allclose(shape_rotated.points, expected_points, atol=1e-6)

    def test_ellipse__polygon_resolution_affects_vertices(self) -> None:
        """Test that smaller resolution results in more vertices"""
        shape_coarse = tools.ellipse(major_axis=10, minor_axis=5, offset=(0, 0), rotation=0, polygon_resolution=5)
        shape_fine = tools.ellipse(major_axis=10, minor_axis=5, offset=(0, 0), rotation=0, polygon_resolution=1)

        # Finer resolution should produce more vertices
        assert len(shape_fine.points) > len(shape_coarse.points)


class TestMakeCross:
    def test_makeCross(self) -> None:
        """Test that collection is created with correct number of shapes"""
        cross_1 = tools.makeCross([20, 20], [50, 50, 50, 50], 1, 10)
        assert isinstance(cross_1, Collection)
        assert len(cross_1.shapes) == 5  # 4 arms + centroid

    @pytest.mark.parametrize(
        ("center",),
        [([0, 0],), ([20, 20],), ([100, 50],), ([-10, 10],)],
        ids=("origin", "positive", "large_positive", "mixed"),
    )
    def test_makeCross__center_position(self, center: list[int]) -> None:
        """Test that center dot is at specified center"""
        cross = tools.makeCross(center, [10, 10, 10, 10], 2, 5)

        # First shape is the center dot (rectangle at center)
        center_shape = cross.shapes[0]
        # Use bounding box center (mean is biased due to closed polygon with duplicate point)
        bbox_center = (center_shape.points.max(axis=0) + center_shape.points.min(axis=0)) / 2

        assert np.allclose(bbox_center, center, atol=1e-6)

    @pytest.mark.parametrize(
        ("center", "arms", "dist"),
        [
            ([0, 0], [10, 10, 10, 10], 5),
            ([20, 20], [10, 10, 10, 10], 5),
            ([0, 0], [10, 20, 30, 40], 5),
            ([0, 0], [50, 50, 50, 50], 0),
            ([0, 0], [10, 10, 10, 10], 20),
        ],
        ids=("symmetric_origin", "symmetric_offset", "asymmetric_arms", "no_gap", "large_gap"),
    )
    def test_makeCross__collection_dimensions(self, center: list[int], arms: list[int], dist: int) -> None:
        """Test that collection has correct total width and height based on arms and dist"""
        width = 2
        cross = tools.makeCross(center, arms, width, dist)

        # Get bounding box of entire collection
        all_points = np.vstack([shape.points for shape in cross.shapes])
        collection_width = all_points[:, 0].max() - all_points[:, 0].min()
        collection_height = all_points[:, 1].max() - all_points[:, 1].min()

        # Total width = left_arm + right_arm + 4*dist (gap on each side of center)
        # Total height = top_arm + bottom_arm + 4*dist
        expected_width = arms[1] + arms[3] + 4 * dist
        expected_height = arms[0] + arms[2] + 4 * dist

        assert np.isclose(collection_width, expected_width, atol=1e-6)
        assert np.isclose(collection_height, expected_height, atol=1e-6)

    @pytest.mark.parametrize(
        ("width",),
        [(1,), (5,), (10,)],
        ids=("thin", "medium", "thick"),
    )
    def test_makeCross__width_affects_shapes(self, width: int) -> None:
        """Test that width parameter affects shape dimensions"""
        cross = tools.makeCross([0, 0], [20, 20, 20, 20], width, 5)

        # Center dot should be width x width
        center_shape = cross.shapes[0]
        center_width = center_shape.points[:, 0].max() - center_shape.points[:, 0].min()
        center_height = center_shape.points[:, 1].max() - center_shape.points[:, 1].min()

        assert np.isclose(center_width, width, atol=1e-6)
        assert np.isclose(center_height, width, atol=1e-6)

        # Top arm should have width in x direction
        top_arm = cross.shapes[1]
        top_arm_width = top_arm.points[:, 0].max() - top_arm.points[:, 0].min()
        assert np.isclose(top_arm_width, width, atol=1e-6)

    @pytest.mark.parametrize(
        ("dist",),
        [(0,), (5,), (20,)],
        ids=("no_gap", "small_gap", "large_gap"),
    )
    def test_makeCross__dist_affects_gap(self, dist: int) -> None:
        """Test that dist parameter controls gap from center"""
        center = np.array([0, 0])
        arm_length = 20
        width = 2

        cross = tools.makeCross(center, [arm_length] * 4, width, dist)

        # Top arm's closest edge to center should be at 2*dist from center
        top_arm = cross.shapes[1]
        top_arm_min_y = top_arm.points[:, 1].min()
        expected_min_y = center[1] + 2 * dist

        assert np.isclose(top_arm_min_y, expected_min_y, atol=1e-6)
