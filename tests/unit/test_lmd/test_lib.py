import operator
import os
from typing import Optional

import geopandas as gpd
import numpy as np
import pytest
import shapely
from lxml import etree as ET

from lmd.lib import Collection, Shape, _create_poly, _execute_indexed_parallel, _sort_edges


@pytest.fixture
def shape_xml():
    """Shape XML"""
    # Define shape in xml
    return """
    <Shape_1>
        <PointCount>3</PointCount>
        <CapID>A1</CapID>
        <TEST>this is a test</TEST>
        <test2>1</test2>
        <test3>3.1415</test3>
        <X_1>0</X_1>
        <Y_1>-0</Y_1>
        <X_2>0</X_2>
        <Y_2>-1</Y_2>
        <X_3>1</X_3>
        <Y_3>-0</Y_3>
    </Shape_1>
    """.strip()


@pytest.fixture
def incorrect_shape_xml():
    """Shape XML"""
    # Define shape in xml
    return """
    <Shape_1>
        <PointCount>2</PointCount>
        <CapID>A1</CapID>
        <TEST>this is a test</TEST>
        <test2>1</test2>
        <test3>3.1415</test3>
        <X_1>0</X_1>
        <Y_1>-0</Y_1>
        <X_2>0</X_2>
        <Y_2>-1</Y_2>
    </Shape_1>
    """.strip()


@pytest.fixture
def collection_xml(tmpdir, shape_xml):
    """Shape XML"""
    # Define shape in xml
    collection = f"""
    <?xml version='1.0' encoding='UTF-8'?>
    <ImageData>
        <GlobalCoordinates>1</GlobalCoordinates>
        <X_CalibrationPoint_1>0</X_CalibrationPoint_1>
        <Y_CalibrationPoint_1>0</Y_CalibrationPoint_1>
        <X_CalibrationPoint_2>0</X_CalibrationPoint_2>
        <Y_CalibrationPoint_2>10000</Y_CalibrationPoint_2>
        <X_CalibrationPoint_3>5000</X_CalibrationPoint_3>
        <Y_CalibrationPoint_3>5000</Y_CalibrationPoint_3>
        <ShapeCount>1</ShapeCount>
        {shape_xml}
    </ImageData>
    """.strip()

    tmpfile = os.path.join(tmpdir, "test.xml")
    with open(tmpfile, "w") as f:
        f.write(collection)
    yield tmpfile
    os.remove(tmpfile)


@pytest.fixture
def incorrect_collection_xml(tmpdir, incorrect_shape_xml):
    """Shape XML"""
    # Define shape in xml
    collection = f"""
    <?xml version='1.0' encoding='UTF-8'?>
    <ImageData>
        <GlobalCoordinates>1</GlobalCoordinates>
        <X_CalibrationPoint_1>0</X_CalibrationPoint_1>
        <Y_CalibrationPoint_1>0</Y_CalibrationPoint_1>
        <X_CalibrationPoint_2>0</X_CalibrationPoint_2>
        <Y_CalibrationPoint_2>10000</Y_CalibrationPoint_2>
        <X_CalibrationPoint_3>5000</X_CalibrationPoint_3>
        <Y_CalibrationPoint_3>5000</Y_CalibrationPoint_3>
        <ShapeCount>1</ShapeCount>
        {incorrect_shape_xml}
    </ImageData>
    """.strip()

    tmpfile = os.path.join(tmpdir, "test.xml")

    with open(tmpfile, "w") as f:
        f.write(collection)
    yield tmpfile
    os.remove(tmpfile)


@pytest.mark.parametrize("tqdm_args", [None, {}])
def test__execute_indexed_parallel(tqdm_args: Optional[dict]) -> None:
    """Test parallelized execution"""
    values = range(10)
    result = _execute_indexed_parallel(operator.mul, args=[[i, 2] for i in values], tqdm_args=tqdm_args, n_threads=2)

    assert result == [2 * value for value in values]


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


def test_collection() -> None:
    calibration = np.array([[0, 0], [0, 100], [50, 50]])
    Collection(calibration_points=calibration)


def test_collection_load(collection_xml) -> None:
    """Test collection loading from xml"""
    collection = Collection()
    collection.load(collection_xml)


def test_collection_invalid_shapes_raise(incorrect_collection_xml) -> None:
    """Test collection loading from xml with incorrect shapes"""
    collection = Collection()
    with pytest.raises(ValueError):
        collection.load(incorrect_collection_xml, raise_shape_errors=True)


def test_collection_invalid_shapes_warn(incorrect_collection_xml) -> None:
    """Test collection loading from xml with incorrect shapes"""
    collection = Collection()
    with pytest.warns():
        collection.load(incorrect_collection_xml, raise_shape_errors=False)


def test_shape():
    rectangle_coordinates = np.array([[10, 10], [40, 10], [40, 40], [10, 40], [10, 10]])
    Shape(rectangle_coordinates)


@pytest.mark.parametrize(
    ("invalid_shape", "error_message"),
    [
        # 2 points
        (
            np.array([[0, 0], [0, 1]]),
            "Valid shape must contain at least 3 points, but only contains 2",
        ),
        # Additional dimension
        (
            np.zeros(shape=(2, 2, 2)),
            "Shape dimensionality is not valid",
        ),
        # 3d points
        (
            np.zeros(shape=(5, 3)),
            "Shape dimensionality is not valid",
        ),
        # 3d points and too few points - covered by dimensionality validation
        (
            np.zeros(shape=(2, 3)),
            "Shape dimensionality is not valid",
        ),
    ],
)
def test_shape_invalid_shapes(invalid_shape, error_message):
    with pytest.raises(ValueError, match=error_message):
        Shape(points=invalid_shape)


def test_shape_from_xml(shape_xml):
    """Read a minimal xml representation of a cell shape and associated metadata"""
    # Load xml with Shape
    shape_xml = ET.fromstring(bytes(shape_xml, encoding="utf-8"))
    shape = Shape.from_xml(shape_xml)
    assert (shape.points == np.array([[0, 0], [0, -1], [1, 0]])).all()
    assert shape.well == "A1"
    assert shape.custom_attributes["TEST"] == "this is a test"
    assert shape.custom_attributes["test2"] == "1"
    assert shape.custom_attributes["test3"] == "3.1415"


def test_plotting():
    calibration = np.array([[0, 0], [0, 100], [50, 50]])
    my_first_collection = Collection(calibration_points=calibration)

    # create a custom rectangle

    rectangle_coordinates = np.array([[10, 10], [40, 10], [40, 40], [10, 40], [10, 10]])
    rectangle = Shape(rectangle_coordinates)

    my_first_collection.add_shape(rectangle)

    my_first_collection.plot(calibration=True)

    triangle_coordinates = np.array([[10, 70], [40, 70], [40, 100], [10, 70]])
    my_first_collection.new_shape(triangle_coordinates)

    my_first_collection.plot(calibration=True)


@pytest.fixture
def geopandas_collection():
    """Geopandas shape collection with both controlled (name, well) and custom metadata"""
    return gpd.GeoDataFrame(
        data={"well": ["A1"], "name": "my_shape", "string_attribute": "a"},
        geometry=[shapely.Polygon([[0, 0], [0, 1], [1, 0], [0, 0]])],
    )


@pytest.mark.parametrize(
    ("well_column", "name_column", "custom_attributes"),
    [
        ("well", None, None),
        (None, "well", None),
        (None, None, "string_attribute"),
        ("well", "name", None),
        ("well", "name", "string_attribute"),
    ],
)
def test_collection_load_geopandas(
    geopandas_collection: gpd.GeoDataFrame,
    well_column: str,
    name_column: str,
    custom_attributes: list[str],
) -> None:
    # Export well metadata
    c = Collection(calibration_points=np.array([[-1, -1], [1, 1], [0, 1]]))
    calibration_points_old = c.calibration_points
    c.load_geopandas(
        geopandas_collection,
        well_column=well_column,
        name_column=name_column,
        custom_attribute_columns=custom_attributes,
    )

    all_columns = [col for col in (well_column, custom_attributes) if col is not None]

    assert c.to_geopandas(*all_columns).equals(geopandas_collection[[*all_columns, "geometry"]])  # type: ignore  # mixed-type unpacking; safe by inspection
    assert (c.calibration_points == calibration_points_old).all()

    # Overwrite calibration points
    c = Collection(calibration_points=np.array([[-1, -1], [1, 1], [0, 1]]))
    calibration_points_new = np.array([[0, 0], [100, 0], [0, 100]])

    c.load_geopandas(
        geopandas_collection,
        calibration_points=calibration_points_new,
        well_column=well_column,
        name_column=name_column,
        custom_attribute_columns=custom_attributes,
    )
    assert c.to_geopandas(*all_columns).equals(geopandas_collection[[*all_columns, "geometry"]])  # type: ignore  # mixed-type unpacking; safe by inspection
    assert (c.calibration_points == calibration_points_new).all()

    # Do not export well metadata
    c = Collection(calibration_points=np.array([[-1, -1], [1, 1], [0, 1]]))
    c.load_geopandas(geopandas_collection)
    assert c.to_geopandas().equals(geopandas_collection[["geometry"]])


def test_collection_save():
    calibration = np.array([[0, 0], [0, 100], [50, 50]])
    my_first_collection = Collection(calibration_points=calibration)

    # create a custom rectangle

    rectangle_coordinates = np.array([[10, 10], [40, 10], [40, 40], [10, 40], [10, 10]])
    rectangle = Shape(rectangle_coordinates)

    my_first_collection.add_shape(rectangle)

    my_first_collection.save("first_collection.xml")


@pytest.fixture
def collection1() -> Collection:
    """Collection with identity orientation transform and one shape"""
    calibration = np.array([[0, 0], [0, 100], [50, 50]])
    collection = Collection(calibration_points=calibration, orientation_transform=np.eye(2))
    collection.add_shape(Shape(np.array([[0, 0], [1, 0], [1, 1], [0, 0]])))
    return collection


@pytest.fixture
def collection2_same_orientation_transform() -> Collection:
    """Collection with same orientation transform as collection1"""
    calibration = np.array([[0, 0], [0, 100], [50, 50]])
    collection = Collection(calibration_points=calibration, orientation_transform=np.eye(2))
    collection.add_shape(Shape(np.array([[2, 2], [3, 2], [3, 3], [2, 2]])))
    return collection


@pytest.fixture
def collection2_different_orientation_transform() -> Collection:
    """Collection with different orientation transform (90 degree rotation)"""
    calibration = np.array([[0, 0], [0, 100], [50, 50]])
    orientation_transform = np.array([[0, -1], [1, 0]])
    collection = Collection(calibration_points=calibration, orientation_transform=orientation_transform)
    collection.add_shape(Shape(np.array([[2, 2], [3, 2], [3, 3], [2, 2]])))
    return collection


def test_collection_join_same_orientation_transform(
    collection1: Collection, collection2_same_orientation_transform: Collection
) -> None:
    """Test joining collections with the same orientation_transform"""
    original_shape = collection1.shapes[0].points.copy()
    joined_shape = collection2_same_orientation_transform.shapes[0].points.copy()

    collection1.join(collection2_same_orientation_transform)

    assert len(collection1.shapes) == 2
    assert np.array_equal(collection1.shapes[0].points, original_shape)
    assert np.array_equal(collection1.shapes[1].points, joined_shape)


def test_collection_join_different_orientation_transform_update(
    collection1: Collection, collection2_different_orientation_transform: Collection
) -> None:
    """Test joining collections with different orientation_transforms and update_orientation_transform=True"""
    original_shape = collection1.shapes[0].points.copy()
    joined_shape = collection2_different_orientation_transform.shapes[0].points.copy()

    collection1.join(collection2_different_orientation_transform, update_orientation_transform=True)

    assert len(collection1.shapes) == 2
    assert np.array_equal(collection1.shapes[0].points, original_shape)
    assert np.array_equal(collection1.shapes[1].points, joined_shape)
    assert np.array_equal(collection1.shapes[1].orientation_transform, collection1.orientation_transform)


def test_collection_join_different_orientation_transform_no_update_warns(
    collection1: Collection, collection2_different_orientation_transform: Collection
) -> None:
    """Test joining collections with different orientation_transforms and update_orientation_transform=False warns"""
    with pytest.warns(UserWarning):
        collection1.join(collection2_different_orientation_transform, update_orientation_transform=False)

    assert len(collection1.shapes) == 2


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
