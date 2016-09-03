import numpy as np
import gdspy


def rectangle(point1 = (2,2), point2 = (0,0), layer = 0, datatype = 0):
    """ Creates a rectangle polygon by specifying opposing corners ``point1`` and ``point2`` """
    points = np.array([[point1[0], point1[1]], [point1[0], point2[1]], [point2[0], point2[1]], [point2[0], point1[1]]])
    return gdspy.Polygon(points, layer = layer, datatype = datatype)
    

def rectangle_centered(size = (2,2), center = (0,0), layer = 0, datatype = 0):
    """ Creates a rectangle polygon of size ``size`` and centered at ``center`` """
    point1 = np.array(center) - np.array(size)/2
    point2 = np.array(center) + np.array(size)/2
    return rectangle(point1, point2, layer = layer, datatype = datatype)

# TODO: Write ring definition
def ring(width, layer = 0, datatype = 0):
    arc()
    
# TODO: Write ring definition    
def arc(radius, width, initial_angle = 0, final_angle = 90, angle_resolution = 5):
    pass
    
#==============================================================================
# Example code
#==============================================================================
    
#r = rectangle_centered([5,50], layer = 5, datatype = 15)
#quickplot(r)