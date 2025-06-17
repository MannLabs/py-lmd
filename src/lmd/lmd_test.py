import numpy as np
from lmd.lib import Collection, Shape
from lmd import tools
from PIL import Image
from lmd.lib import SegmentationLoader
import pathlib
import os
import geopandas as gpd
import shapely
from lxml import etree as ET
import pytest


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


def test_collection() -> None:
    calibration = np.array([[0, 0], [0, 100], [50, 50]])
    my_first_collection = Collection(calibration_points=calibration)


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
    rectangle = Shape(rectangle_coordinates)


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

    assert c.to_geopandas(*all_columns).equals(
        geopandas_collection[[*all_columns, "geometry"]]
    )
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
    assert c.to_geopandas(*all_columns).equals(
        geopandas_collection[[*all_columns, "geometry"]]
    )
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


def test_tools_square():
    calibration = np.array([[0, 0], [0, 100], [50, 50]])
    my_first_collection = Collection(calibration_points=calibration)

    my_square = tools.rectangle(10, 10, offset=(10, 10))
    my_first_collection.add_shape(my_square)

    my_square = tools.rectangle(10, 10, offset=(30, 30))
    my_first_collection.add_shape(my_square)


def test_glyphs():
    calibration = np.array([[0, 0], [0, 100], [50, 50]])
    my_first_collection = Collection(calibration_points=calibration)

    digit_1 = tools.glyph(1)
    my_first_collection.join(digit_1)

    digit_2 = tools.glyph(2, offset=(0, 80), multiplier=5)
    my_first_collection.join(digit_2)

    glyph_A = tools.glyph("A", offset=(0, 80), rotation=-np.pi / 4)
    my_first_collection.join(glyph_A)


def test_text():
    calibration = np.array([[0, 0], [0, 100], [100, 50]])
    my_first_collection = Collection(calibration_points=calibration)

    identifier_1 = tools.text("0123_A1", offset=np.array([0, 40]), rotation=-np.pi / 4)
    my_first_collection.join(identifier_1)

    identifier_2 = tools.text("0456_B2", offset=np.array([30, 40]), rotation=-np.pi / 4)
    my_first_collection.join(identifier_2)

    identifier_3 = tools.text(
        "0123456789-_ABCDEFGHI", offset=np.array([60, 40]), rotation=-np.pi / 4
    )
    my_first_collection.join(identifier_3)


def test_segmentation_loader():
    package_base_path = pathlib.Path(__file__).parent.parent.parent.resolve().absolute()
    test_segmentation_path = os.path.join(
        package_base_path,
        "docs/pages/notebooks/Image_Segmentation/segmentation_cytosol.tiff",
    )

    im = Image.open(test_segmentation_path)
    segmentation = np.array(im).astype(np.uint32)

    all_classes = np.unique(segmentation)

    cell_sets = [{"classes": all_classes, "well": "A1"}]

    calibration_points = np.array([[0, 0], [0, 1000], [1000, 1000]])

    loader_config = {"orientation_transform": np.array([[0, -1], [1, 0]])}

    sl = SegmentationLoader(config=loader_config)
    shape_collection = sl(segmentation, cell_sets, calibration_points)
