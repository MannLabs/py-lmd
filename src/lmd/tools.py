from lmd.lmd import LMD_object, LMD_shape
from pathlib import Path
import numpy as np
import os

def digit( number, offset=np.array([0,0]), divisor=1, multiplier=1, **kwargs):
    number = int(number)
    if not 0 <= number <= 9:
        raise ValueError('The number musst be a digit between 0 and 9')
        

    file_path = Path(os.path.realpath(__file__))
    file_path = file_path.parents[1]
    
    svg_path = os.path.join(file_path, f"numbers/numbers_{number}.svg")
    
    local_offset = np.array([100,-20])*multiplier
    offset = offset + local_offset
    
    local_divisor = 50
    divisor = divisor * local_divisor
    
    local_multiplier = 0.1
    multiplier = multiplier * local_multiplier
    
    
    shapefile = LMD_object()
    shapefile.svg_to_lmd(svg_path, offset=offset, multiplier=multiplier, **kwargs)
    
    return shapefile

def number( number, offset=np.array([0,0]), divisor=1, multiplier=1, **kwargs):
    delta = np.array([0, 80]) * multiplier
    
    heap = LMD_object()
    
    for i, current_digit in enumerate(str(number)):
        current_digit = int(current_digit)
        
        current = digit(current_digit, offset=offset+i*delta, multiplier=multiplier,  **kwargs)
        heap.join(current)
        
    return heap
        