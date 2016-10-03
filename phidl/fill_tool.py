from __future__ import division
import gdspy
import numpy as np
from phidl import Device
import phidl.geometry as pg

from skimage import draw, morphology
#.draw import polygon_perimeter, polygon, line_aa, line, ellipse
#from skimage.morphology import binary_dilation
import itertools

from matplotlib import pyplot as plt

def rasterize_polygons(polygons, bounds = [[-100, -100], [100, 100]], dx = 1, dy = 1):
    
    # Prepare polygon array by shifting all points into the first quadrant and 
    # separating points into x and y lists
    xpts = []
    ypts = []
    for p in polygons:
        p_array = np.asarray(p)
        x = p_array[:,0]
        y = p_array[:,1]
        xpts.append((x-bounds[0][0])/dx-0.5)
        ypts.append((y-bounds[0][1])/dy-0.5)

    # Initialize the raster matrix we'll be writing to
    xsize = int(np.ceil((bounds[1][0]-bounds[0][0]))/dx)
    ysize = int(np.ceil((bounds[1][1]-bounds[0][1]))/dy)
    raster = np.zeros((ysize, xsize), dtype=np.bool)
    
    # TODO: Replace polygon_perimeter with the supercover version
    for n in range(len(xpts)):
        rr, cc = draw.polygon(ypts[n], xpts[n], shape=raster.shape)
        rrp, ccp = draw.polygon_perimeter(ypts[n], xpts[n], shape=raster.shape, clip=False)
        raster[rr, cc] = 1
        raster[rrp, ccp] = 1
        
    return raster
    
def _raster_index_to_coords(i, j, bounds = [[-100, -100], [100, 100]], dx = 1, dy = 1):
    x = (j+0.5)*dx + bounds[0][0]
    y = (i+0.5)*dy + bounds[0][1]
    return x,y


def expand_raster(raster, distance = (4,2)):
    if distance[0] <= 0.5 and distance[1] <= 0.5: return raster
        
    num_pixels = map(int, np.ceil(distance))
    neighborhood = np.zeros((num_pixels[1]*2+1, num_pixels[0]*2+1), dtype=np.bool)
    rr, cc = draw.ellipse(r = num_pixels[1], c = num_pixels[0], yradius = distance[1]+0.5, xradius = distance[0]+0.5)
    neighborhood[rr, cc] = 1
    
    return morphology.binary_dilation(image = raster, selem=neighborhood)

    
#def expand_polygons(polygons, expand_distance = 10, precision = 0.001, join_first = False):
#    """ Takes a list of polygons and if join_first is True, does a boolean AND
#    of all of them then does an offset/expansion.  If join_first is false, 
#    expands each polygon individually (faster)"""
#    return gdspy.clipper.offset(polygons, expand_distance, 'miter', 2, 1/precision, join_first)
            
            
def fill_rectangular(size = (20,20), layers = (0,1,3), densities = (0.5, 0.25, 0.7), datatype = 77):
    d = Device(name = 'fill_cell')
    for layer, density in zip(layers, densities):
        rectangle_size = np.array(size)*np.sqrt(density)
        point1 = -np.array(rectangle_size)/2
        point2 = np.array(rectangle_size)/2
        d.add(gdspy.Rectangle(point1, point2, layer = layer, datatype = datatype))
    return d
            
def dummy_fill_rectangular(d, fill_size = (40,10), exclude_layers = None, fill_layers = (0,1,3), fill_densities = (0.5, 0.25, 0.7), margin = 100, bbox = None):
    fill_cell = fill_rectangular(size = fill_size, layers = fill_layers, densities = fill_densities, datatype = 77)
    fill_pattern = Device(name = 'fill_pattern')
    
    if exclude_layers is None:
        poly = d.get_polygons(by_spec=False, depth=None)
    elif np.array(exclude_layers).ndim == 1: # Then exclude_layers is a list of just layers e.g. [0,2,3]
        poly = d.get_polygons(by_spec=True, depth=None)
        poly = {key:poly[key] for key in poly if key[0] in exclude_layers} # Filter the dict
        poly = itertools.chain.from_iterable(poly.values()) # Concatenate dict values to long list
    elif np.array(exclude_layers).ndim == 2: # Then exclude_layers is a list of layers + datatypes e.g. [(0,1),(0,2),(1,0)]
        poly = d.get_polygons(by_spec=True, depth=None)
        poly = {key:poly[key] for key in poly if key in exclude_layers}
        poly = itertools.chain.from_iterable(poly.values())
        
    if bbox is None:  bbox = d.bbox

    raster = rasterize_polygons(polygons = poly, bounds = bbox, dx = fill_size[0], dy = fill_size[1])
    raster = expand_raster(raster, distance = margin/np.array(fill_size))
    
    for i in range(np.size(raster,0)):
        sub_rasters = [list(g) for k, g in itertools.groupby(raster[i])]
        j = 0
        for s in sub_rasters:
            if s[0] == 0:
                x,y = _raster_index_to_coords(i, j, bbox, fill_size[0], fill_size[1])
                fill_pattern.add(gdspy.CellArray(ref_cell = fill_cell, columns = len(s), rows = 1, spacing = fill_size, origin = (x, y)))
            j += len(s)
    
    return fill_pattern
