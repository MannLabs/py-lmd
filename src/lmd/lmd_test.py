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

def test_collection():
    calibration = np.array([[0, 0], [0, 100], [50, 50]])
    my_first_collection = Collection(calibration_points = calibration)
    
def test_shape():
    rectangle_coordinates = np.array([[10,10], [40,10], [40,40], [10,40], [10,10]])
    rectangle = Shape(rectangle_coordinates)

def test_shape_from_xml():
    # Define shape in xml
    shape_xml = """
    <Shape_1>
        <PointCount>3</PointCount>
        <CapID>A1</CapID>
        <TEST>this is a test</TEST>
        <X_1>0</X_1>
        <Y_1>-0</Y_1>
        <X_2>0</X_2>
        <Y_2>-1</Y_2>
        <X_3>1</X_3>
        <Y_3>-0</Y_3>
    </Shape_1>
    """.strip()

    # Parse xml
    shape_xml = ET.fromstring(bytes(shape_xml, encoding="utf-8"))

    # Load xml with Shape
    shape = Shape()
    shape.from_xml(shape_xml)
    assert (shape.points == np.array([[ 0,  0], [ 0, -1], [ 1,  0]])).all()
    assert shape.well == "A1"

def test_plotting():
    calibration = np.array([[0, 0], [0, 100], [50, 50]])
    my_first_collection = Collection(calibration_points = calibration)

    # create a custom rectangle

    rectangle_coordinates = np.array([[10,10], [40,10], [40,40], [10,40], [10,10]])
    rectangle = Shape(rectangle_coordinates)

    my_first_collection.add_shape(rectangle)

    my_first_collection.plot(calibration = True)

    triangle_coordinates = np.array([[10,70], [40,70], [40,100], [10,70]])
    my_first_collection.new_shape(triangle_coordinates)

    my_first_collection.plot(calibration = True)

def test_collection_load_geopandas():
    gdf = gpd.GeoDataFrame(
        data={"well": ["A1"], "name": "my_shape"},
        geometry=[shapely.Polygon([[0, 0], [0, 1], [1, 0], [0, 0]])]
    )


    # Export well metadata
    c = Collection(calibration_points=np.array([[-1, -1], [1, 1], [0, 1]]))
    calibration_points_old = c.calibration_points
    c.load_geopandas(gdf, well_column="well", name_column="name")
    assert c.to_geopandas("well", "name").equals(gdf)
    assert (c.calibration_points == calibration_points_old).all()

    # Overwrite calibration points 
    c = Collection(calibration_points=np.array([[-1, -1], [1, 1], [0, 1]]))
    calibration_points_new = np.array([[0, 0], [100, 0], [0, 100]])
    c.load_geopandas(gdf, well_column="well", name_column="name", calibration_points=calibration_points_new)
    assert c.to_geopandas("well", "name").equals(gdf)
    assert (c.calibration_points == calibration_points_new).all()

    # Do not export well metadata
    c = Collection(calibration_points=np.array([[-1, -1], [1, 1], [0, 1]]))
    c.load_geopandas(gdf)
    assert c.to_geopandas().equals(gdf.drop(columns=["well", "name"]))

def test_collection_save():
    calibration = np.array([[0, 0], [0, 100], [50, 50]])
    my_first_collection = Collection(calibration_points = calibration)

    # create a custom rectangle

    rectangle_coordinates = np.array([[10,10], [40,10], [40,40], [10,40], [10,10]])
    rectangle = Shape(rectangle_coordinates)

    my_first_collection.add_shape(rectangle)

    my_first_collection.save("first_collection.xml")
    
def test_tools_square():
    calibration = np.array([[0, 0], [0, 100], [50, 50]])
    my_first_collection = Collection(calibration_points = calibration)

    my_square = tools.rectangle(10, 10, offset=(10,10))
    my_first_collection.add_shape(my_square)

    my_square = tools.rectangle(10, 10, offset=(30,30))
    my_first_collection.add_shape(my_square)
                                  
def test_glyphs():
    calibration = np.array([[0, 0], [0, 100], [50, 50]])
    my_first_collection = Collection(calibration_points = calibration)

    digit_1 = tools.glyph(1)
    my_first_collection.join(digit_1)

    digit_2 = tools.glyph(2, offset = (0,80), multiplier = 5)
    my_first_collection.join(digit_2)

    glyph_A = tools.glyph('A', offset=(0,80), rotation =-np.pi/4)
    my_first_collection.join(glyph_A)
                                  
def test_text():
    calibration = np.array([[0, 0], [0, 100], [100, 50]])
    my_first_collection = Collection(calibration_points = calibration)

    identifier_1 = tools.text('0123_A1', offset=np.array([0, 40]), rotation = -np.pi/4)
    my_first_collection.join(identifier_1)

    identifier_2 = tools.text('0456_B2', offset=np.array([30, 40]), rotation = -np.pi/4)
    my_first_collection.join(identifier_2)

    identifier_3 = tools.text('0123456789-_ABCDEFGHI', offset=np.array([60, 40]), rotation = -np.pi/4)
    my_first_collection.join(identifier_3)                 

def test_segmentation_loader():

    package_base_path = pathlib.Path(__file__).parent.parent.parent.resolve().absolute()
    test_segmentation_path = os.path.join(package_base_path, 'docs/pages/notebooks/Image_Segmentation/segmentation_cytosol.tiff')

    im = Image.open(test_segmentation_path)
    segmentation = np.array(im).astype(np.uint32)

    all_classes = np.unique(segmentation)

    cell_sets = [{"classes": all_classes, "well": "A1"}]

    calibration_points = np.array([[0,0],[0,1000],[1000,1000]])

    loader_config = {
        'orientation_transform': np.array([[0, -1],[1, 0]])
    }

    sl = SegmentationLoader(config = loader_config)
    shape_collection = sl(segmentation, 
                        cell_sets, 
                        calibration_points)