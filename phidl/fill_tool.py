# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 18:40:54 2016
 
@author: amcc
"""

from __future__ import division
import gdspy
import numpy as np
from matplotlib import pyplot as plt
 

from skimage.draw import polygon_perimeter, polygon, line_aa, line
from itertools import groupby

def rasterize_polygons(polygons, bounds = [[-100, -100], [100, 100]], dx = 1, dy = 1):
    
    # Prepare polygon array by shifting all points into the first quadrant and 
    # separating points into x and y lists
    xpts = []
    ypts = []
    for p in polygons:
        x = np.array([v[0] for v in p])
        y = np.array([v[1] for v in p])
        xpts.append(x)
        ypts.append(y)
    
    xpts = [(x-bounds[0][0])/dx for x in xpts]
    ypts = [(y-bounds[0][1])/dy for y in ypts]
    
    # Initialize the raster matrix we'll be writing to
    xsize = int(np.ceil((bounds[1][0]-bounds[0][0]))/dx)
    ysize = int(np.ceil((bounds[1][1]-bounds[0][1]))/dy)
    raster = np.zeros((ysize, xsize), dtype=np.bool)
    
    # TODO: Replace this with the scan-line polygon filling algorithm
    # which is more robust against overlap when the exclude margins are small
    # TODO: Replace polygon_perimeter with a supercover version of itself
    # as shown in http://playtechs.blogspot.ca/2007/03/raytracing-on-grid.html
    for x,y in zip(xpts, ypts):
        rr, cc = polygon(y, x, shape=raster.shape)
        rrp, ccp = polygon_perimeter(y, x, shape=raster.shape, clip=False)
        raster[rr, cc] = 1
        raster[rrp, ccp] = 1
        
    return raster
    
def _raster_index_to_coords(i, j, bounds = [[-100, -100], [100, 100]], dx = 1, dy = 1):
    x = j*dx + bounds[0][0]
    y = i*dy + bounds[0][1]
    return x,y
    
    
def expand_polygons(polygons, expand_distance = 10, precision = 0.001, join_first = False):
    """ Takes a list of polygons and if join_first is True, does a boolean AND
    of all of them then does an offset/expansion.  If join_first is false, 
    expands each polygon individually (faster)"""
    return gdspy.clipper.offset(polygons, expand_distance, 'miter', 2, 1/precision, join_first)
            
            
def dummy_fill(cell, exclude_layers, fill_cell, margin = None, bounds =  [[-100, -100], [100, 100]], dx = 10, dy = 10):
    cell.get_polygons(by_spec=True, depth=None)
    if margin == None: margin = max([dx,dy])/2


# Test fill tool with large file
gdspy.Cell.cell_dict.clear()
filename = "C:/Users/anm16/Documents/Python Scripts/SNSPD Integrator.gds"
gds_file = gdspy.GdsImport(filename)
top_cell_name = 'toplevel'
gds_file.extract(top_cell_name)
top_cell = gdspy.Cell.cell_dict[top_cell_name]
top_cell_polygons = top_cell.get_polygons(by_spec=True, depth=None)
# %timeit check_box_for_polygons(top_cell_polygons, x = 0, y = 0, dx = 0.1, dy = 0.1)
# %timeit check_box_for_polygons_rtree(top_cell_polygons, idx, x = 0, y = 0, dx = 0.1, dy = 0.1)
# bool_list  = raster_box_rtree(top_cell_polygons, idx, bbox = [[-4000,-4000],[4000,4000]], dx = 40, dy = 40)
# %timeit raster_box_rtree(top_cell_polygons, idx, bbox = [[-4000,-4000],[4000,4000]], dx = 400, dy = 400)


bounds = [[-6500, -6500], [6500, 6500]]
dx = 5
dy = 5

#, precision = 0.001, join_first = False)
raster = rasterize_polygons(polygons = expand_polygons(top_cell_polygons, expand_distance = 15), bounds = bounds, dx = dx, dy = dy)
plt.imshow(raster,  interpolation='nearest')


fill_cell = gdspy.Cell('fillcell')
fill_cell.add(gdspy.Polygon([(-dx/2,-dy/2), (-dx/2,dy/2), (dx/2,dy/2),(dx/2,-dy/2)], layer = 98))
fill_cell.add(gdspy.Polygon([(-dx/4,-dy/4), (-dx/4,dy/4), (dx/4,dy/4),(dx/4,-dy/4)], layer = 2))


for i in range(np.size(raster,0)):
    sub_rasters = [list(g) for k, g in groupby(raster[i])]
    j = 0
    for s in sub_rasters:
        if s[0] == 0:
            x,y = _raster_index_to_coords(i, j, bounds, dx, dy)
            top_cell.add(gdspy.CellArray(ref_cell = fill_cell, columns = len(s), rows = 1, spacing = (dx, dy), origin = (x, y)))
        j += len(s)

gdspy.gds_print('mytestgdsout.gds', unit=1.0e-6, precision=1.0e-9)





