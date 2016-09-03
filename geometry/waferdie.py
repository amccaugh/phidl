import numpy as np
import gdspy
from numpy import sqrt, pi


def basic_die(size = (10000, 10000), street_width = 100, street_length = 1000,
              die_text = None, text_size = 100, text_location = 'SW', layer = 99, datatype = 0,
              draw_bbox = True, bbox_layer = 99, bbox_datatype = 99):
    
    #==========================================================================
    #  Create the basic geometry
    #==========================================================================
d = Device(name = 'die')
xpts = np.array([size[0]/2, size[0]/2, size[0]/2-street_width, size[0]/2-street_width, size[0]/2-street_length, size[0]/2-street_length])
ypts = np.array([size[1]/2, size[1]/2-street_length, size[0]/2-street_length, size[0]/2-street_width, size[0]/2-street_width, size[1]/2])
d.add_polygon([xpts,ypts], layer = layer, datatype = datatype)
d.add_polygon([-xpts,ypts], layer = layer, datatype = datatype)
d.add_polygon([xpts,-ypts], layer = layer, datatype = datatype)
d.add_polygon([-xpts,-ypts], layer = layer, datatype = datatype)

if draw_bbox is True:
    s = np.array(size)/2
    d.add_polygon([[s[0],s[1]], [s[0],-s[1]],[-s[0],-s[1]],[-s[0],s[1]]], layer = bbox_layer, datatype = bbox_datatype)

# FIXME START HERE
d.add_device(text())
quickplot(d)
    #==========================================================================
    #  Create a blank device, add the geometry, and define the ports
    #==========================================================================
    d = Device(name = 'ytron')
    d.add_polygon([xpts,ypts], layer = layer, datatype = datatype)
    d.add_port(name = 'left', midpoint = [-(xc+arm_x+width_left/2), yc+arm_y],  width = width_left, orientation = 90)
    d.add_port(name = 'right', midpoint = [xc+arm_x+width_right/2, yc+arm_y],  width = width_right, orientation = 90)
    d.add_port(name = 'source', midpoint = [0+(width_right-width_left)/2, -source_length+yc],  width = width_left + width_right + 2*xc, orientation = -90)
    
    #==========================================================================
    #  Record any parameters you may want to access later
    #==========================================================================
    d.parameters['radius_curvature'] = rho_intersection
    d.parameters['left_arm_width'] = width_left
    d.parameters['right_arm_width'] = width_right

    return d
    
    
    
#==============================================================================
# Example code
#==============================================================================

#y = ytron_round(rho_intersection = 5, theta_intersection = 5, theta_resolution = 10, arm_length = 500, \
#                source_length = 500, width_right = 200, width_left = 200, layer = 0, datatype = 0)
#quickplot(y)
