import numpy as np
import gdspy


def compass(dimensions = [4,2], layer = 0, dtype = 0):
    """ Creates a rectangular contact pad with centered ports on edges of the
    rectangle (north, south, east, and west)
    
    """
    
    d = Device(name = 'pad_compass')
    d.add_polygon(rectangle_centered(dimensions, center = [0,0]))
    
    dx = dimensions[0]
    dy = dimensions[1]
    d.add_port(name = 'N', midpoint = [0, dy/2],  width = dx, orientation = 90)
    d.add_port(name = 'S', midpoint = [0, -dy/2], width = dx, orientation = -90)
    d.add_port(name = 'E', midpoint = [dx/2, 0],  width = dy, orientation = 0)
    d.add_port(name = 'W', midpoint = [-dx/2, 0], width = dy, orientation = 180)
    
    return d
    
    
    
#==============================================================================
# Example code
#==============================================================================

cp = compass(dimensions = [4,2])
quickplot(cp)