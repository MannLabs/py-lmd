from lmd.lib import Collection, Shape
from pathlib import Path
import numpy as np
import os

def get_rotation_matrix(degrees: float):
    """Returns a rotation matrix for counterclock wise rotation.
        Args: 
            degrees: Rotation in degrees.            
    """
    
    return np.array([[np.cos(degrees), -np.sin(degrees)],[np.sin(degrees), np.cos(degrees)]])

def glyph_path(glyph):
    """Returns a rotation matrix for counterclock wise rotation.
        Args: 
            degrees: Rotation in degrees.            
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

def square(width, height, 
           offset = (0, 0), 
           rotation = 0, 
           rotation_offset = (0, 0)):
    
    offset = np.array(offset)
    rotation_offset = np.array(rotation_offset)
    rotation_mat = get_rotation_matrix(rotation)
    
    points = np.array([[-height/2,-width/2],
                     [-height/2,width/2],
                     [height/2,width/2],
                     [height/2,-width/2],
                     [-height/2,-width/2]])
    
    points = (points + rotation_offset) @ rotation_mat - rotation_offset + offset
    return Shape(points)