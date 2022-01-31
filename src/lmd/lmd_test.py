import numpy as np
from lmd.lib import Collection, Shape
from lmd import tools


def test_collection():
    calibration = np.array([[0, 0], [0, 100], [50, 50]])
    my_first_collection = Collection(calibration_points = calibration)
    
def test_shape():
    rectangle_coordinates = np.array([[10,10], [40,10], [40,40], [10,40], [10,10]])
    rectangle = Shape(rectangle_coordinates)
    
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

    my_square = tools.square(10, 10, offset=(10,10))
    my_first_collection.add_shape(my_square)

    my_square = tools.square(10, 10, offset=(30,30))
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
                    