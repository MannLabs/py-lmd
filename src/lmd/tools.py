from lmd.lib import Collection, Shape
from pathlib import Path
import numpy as np
import os
import pkgutil

def get_rotation_matrix(angle: float):
    """Returns a rotation matrix for clockwise rotation.
    
        Args: 
            angle (float): Rotation in radian.      
        Returns:
            np.ndarray: Matrix in the shape of (2, 2).
    """
    
    return np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])

def glyph_path(glyph):
    """Returns the path for a glyph of interest. Raises a NotImplementedError if an unknown glyph is requested.
    
        Args: 
            glyph (str): Single glyph as string.
        
        Returns:
            str: Path for the glyph.
    """
    
    file_path = Path(os.path.realpath(__file__))
    file_path = file_path.parents[1]
    
    svg_path = os.path.join(file_path, f"glyphs/{glyph}.svg")
    
    # Check if glyph exists, raise NotImplementedError if not
    if os.path.isfile(svg_path):
        return svg_path
    else:
        raise NotImplementedError(f'You tried to load the glyph {glyph}. This has not been implemented yet.')
    

def glyph(glyph, 
          offset=np.array([0,0]), 
          rotation = 0, 
          divisor=10, 
          multiplier=1, **kwargs):
    """Get an uncalibrated lmd.lib.Collection for a glyph of interest.
    
        Args: 
            glyph (str): Single glyph as string.
            
            divisor (int): Parameter which determines the resolution when creating a polygon from a SVG. A larger divisor will lead to fewer datapoints for the glyph. Default value: 10
            
            offset (np.ndarray): Location of the glyph based on the top left corner. Default value: np.array((0, 0))
            
            multiplier (float): Scaling parameter for defining the size of the glyph. The default height of a glyph is 10 units. Default value: 1
        
        Returns:
            lmd.lib.Collection: Uncalibrated Collection which contains the Shapes for the glyph.
    """
        
    svg_path = glyph_path(glyph)
    
    local_multiplier = 0.01
    multiplier = multiplier * local_multiplier
    
    shapefile = Collection()
    shapefile.svg_to_lmd(svg_path, 
                         offset=offset, 
                         rotation_matrix = get_rotation_matrix(rotation), 
                         multiplier=multiplier, 
                         **kwargs)
    
    return shapefile

def text(text, 
           offset=np.array([0,0]), 
           divisor=1, 
           multiplier=1,
           rotation = 0, 
           **kwargs):
    """Get an uncalibrated lmd.lib.Collection for a text.
    
        Args: 
            text (str): Text as string.
            
            divisor (int): Parameter which determines the resolution when creating a polygon from a SVG. A larger divisor will lead to fewer datapoints for the glyph. Default value: 10
            
            offset (np.ndarray): Location of the text based on the top left corner. Default value: np.array((0, 0))
            
            multiplier (float): Scaling parameter for defining the size of the text. The default height of a glyph is 10 units. Default value: 1
        
        Returns:
            lmd.lib.Collection: Uncalibrated Collection which contains the Shapes for the text.
            
        Example:
        
            .. code-block:: python

                import numpy as np
                from lmd.lib import Collection, Shape
                from lmd import tools

                calibration = np.array([[0, 0], [0, 100], [100, 50]])
                my_first_collection = Collection(calibration_points = calibration)

                identifier_1 = tools.text('0456_B2', offset=np.array([30, 40]), rotation = -np.pi/4)
                my_first_collection.join(identifier_1)
                my_first_collection.plot(calibration = True)
                
            .. image:: images/fig10.png
               :scale: 100%
    """
    
    # Convert text to str and assert 
    text = str(text)
    
    
    # delta offset for every digit
    delta = np.array([10, 0]) @ get_rotation_matrix(rotation) * multiplier
    
    heap = Collection()
    
    # enumerate all glyphs and append shapes
    for i, current_glyph in enumerate(text):
        
        current = glyph(current_glyph, 
                        offset = offset+i*delta, 
                        multiplier = multiplier,  
                        rotation = rotation,
                        **kwargs)
        heap.join(current)
        
    return heap

def rectangle(width, height, 
           offset = (0, 0), 
           angle = 0, 
           rotation_offset = (0, 0)):
    
    """Get a lmd.lib.Shape for rectangle of choosen dimensions.
    
        Args: 
            width (float): Width of the rectangle.
            
            offset (np.ndarray): Location of the rectangle based on the center. Default value: np.array((0, 0))
            
            angle (float): Rotation in radian.
            
            rotation_offset (np.ndarray): Location of the center of rotation relative to the center of the rectangle. Default value: np.array((0, 0))
        
        Returns:
            lmd.lib.Shape: Shape which contains the rectangle.
            
        Example:
    """
    
    offset = np.array(offset)
    rotation_offset = np.array(rotation_offset)
    rotation_mat = get_rotation_matrix(angle)
    
    points = np.array([[-height/2,-width/2],
                     [-height/2,width/2],
                     [height/2,width/2],
                     [height/2,-width/2],
                     [-height/2,-width/2]])
    
    points = (points + rotation_offset) @ rotation_mat - rotation_offset + offset
    return Shape(points)