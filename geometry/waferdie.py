import numpy as np
import gdspy
from numpy import sqrt, pi


def basic_die(size = (10000, 10000),
              street_width = 100,
              street_length = 1000,
              die_name = 'chip99',
              text_size = 100,
              text_location = 'SW', 
              layer = 0, 
              datatype = 0,
              draw_bbox = True, 
              bbox_layer = 99, 
              bbox_datatype = 99):
    
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
    
    
    if text_location.upper() == 'SW':
        justify = 'left'
        text_position = (-size[0]/2 + street_width*2, -size[1]/2 + street_width*2)
    d.add_device(text(text = die_name, size = text_size, position=text_position, justify = justify, layer=layer, datatype=datatype))
    #quickplot(d)
    
    return d
    
    
    
#==============================================================================
# Example code
#==============================================================================

d = basic_die(size = (10000, 10000), street_width = 100, street_length = 1000, 
              die_name = 'chip99', text_size = 300, text_location = 'SW',  layer = 0,  
              datatype = 0, draw_bbox = True,  bbox_layer = 99,  bbox_datatype = 99)
quickplot(d)
