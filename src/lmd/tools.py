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
    
    if os.path.isfile(svg_path):
        return svg_path
    else:
        raise NotImplementedError()
    

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
    
    
    # delta offset for every digit
    delta = np.array([10, 0]) @ get_rotation_matrix(rotation) * multiplier
    
    heap = Collection()
    
    for i, current_digit in enumerate(str(number)):
        current_digit = int(current_digit)
        
        print(current_digit,offset+i*delta,multiplier,get_rotation_matrix(rotation))
        
        current = digit(current_digit, 
                        offset = offset+i*delta, 
                        multiplier = multiplier,  
                        rotation = rotation,
                        **kwargs)
        heap.join(current)
        
    return heap

def _letter(letter, offset=np.array([0,0]), divisor=1, multiplier=1, **kwargs):
    
    letter = str(letter).upper()
    
    file_path = Path(os.path.realpath(__file__))
    file_path = file_path.parents[1]
    
    svg_path = os.path.join(file_path, f"letters/Letters_{letter}.svg")
    
    local_offset = np.array([100,-20])*multiplier
    offset = offset + local_offset
    
    local_divisor = 50
    divisor = divisor * local_divisor
    
    local_multiplier = 0.1
    multiplier = multiplier * local_multiplier
    
    shapefile = LMD_object()
    shapefile.svg_to_lmd(svg_path, offset=offset, multiplier=multiplier, **kwargs)
    
    return shapefile


    
def letter(letter, offset=np.array([0,0]), divisor=1, multiplier=1, **kwargs):
    
    delta = np.array([0, 80]) * multiplier
    
    heap = LMD_object()
    
    for i, current_letter in enumerate(str(letter)):
        current_letter = str(current_letter)
        
        current = _letter(current_letter, offset=offset+i*delta, multiplier=multiplier,  **kwargs)
        heap.join(current)
        
    return heap  
        
