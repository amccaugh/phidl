# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 18:40:54 2016
 
@author: amcc
"""
 
 from __future__ import division
import gdspy
import numpy as np
 
 
# gdspy.Cell.cell_dict.clear()
# poly_cell = gdspy.Cell('POLYGONS')
# x = [0,0,1.1,2]
# y = [0,1,1,0]
# xy = zip(x,y)
# poly1 = gdspy.Polygon(xy, layer = 1)
# #poly1.fillet([0.2, 0.05, 0.8, 0.0])
# poly_cell.add(poly1)


#  >>> from skimage.draw import polygon
# >>> img = np.zeros((10, 10), dtype=np.uint8)
# >>> x = np.array([1, 7, 4, 1])
# >>> y = np.array([1, 2, 8, 1])
# >>> rr, cc = polygon(y, x)
# >>> img[rr, cc] = 1
# >>> img
# array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
#        [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
#        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
#        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
#        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
 
 
 
 
# x = [0, 0, 0.01, 0.01]
# y = [0, 0.01, 0.01, 0]
# xy = zip(x,y)
# fill_bbox = gdspy.Polygon(xy, layer = 0)
# polygons = poly_cell.get_polygons(by_spec=False)
# poly_out = gdspy.fast_boolean(operandA = fill_bbox, operandB = polygons, operation = 'and', precision=0.001, max_points=199, layer=0, datatype=0)

# poly_cell.add(poly_out)

# gdspy.LayoutViewer()
 

# top_cell = poly_cell
# top_cell_bbox = top_cell.get_bounding_box()
# top_cell_polygons = top_cell.get_polygons(by_spec=False, depth=None)

# x = 10
# y = 10
# dx = 0.1
# dy = 0.1
# xpts = [x, x, x+dx, x+dx]
# ypts = [y, y+dy, y+dy, y]
# checking_polygon = gdspy.Polygon(zip(xpts,ypts))


# top_cell_polygons = top_cell.get_polygons(by_spec=False, depth=None)

# box_overlap_poly = gdspy.fast_boolean(operandA = checking_polygon, operandB = top_cell_polygons, \
                # operation = 'and', precision=0.001, max_points=199, layer=0, datatype=0)
 
 
#def check_box_for_polygons(polygons, x = 0, y = 0, dx = 0.1, dy = 0.1, layers = None):
#    """ Checks to see if any of the polygons in polygons exist in the box-shaped region
#    from (x,y) to (x+dx, y+dy).  Returns ``True`` if any of the polgyons overlap that region """
#    xpts = [x, x, x+dx, x+dx]
#    ypts = [y, y+dy, y+dy, y]
#    checking_polygon = gdspy.Polygon(zip(xpts,ypts))
#
#    overlapping_polygons = gdspy.fast_boolean(operandA = checking_polygon, operandB = polygons, \
#                    operation = 'and', precision=0.001, max_points=199, layer=0, datatype=0)
#    return len(overlapping_polygons.polygons) > 0
#    
#
#def check_box_for_polygons_rtree(polygons, rtree_idx, x = 0, y = 0, dx = 0.1, dy = 0.1, layers = None):
#    """ Checks to see if any of the polygons in polygons exist in the box-shaped region
#    from (x,y) to (x+dx, y+dy).  Returns ``True`` if any of the polgyons overlap that region """
#    xpts = [x, x, x+dx, x+dx]
#    ypts = [y, y+dy, y+dy, y]
#    checking_polygon = gdspy.Polygon(zip(xpts,ypts))
#    intersected_polygons_index = list(rtree_idx.intersection((x, y, x+dx, y+dy)))
#    intersected_polygons = [top_cell_polygons[i] for i in intersected_polygons_index]
#
#    overlapping_polygons = gdspy.fast_boolean(operandA = checking_polygon, operandB = intersected_polygons, \
#                    operation = 'and', precision=0.001, max_points=199, layer=0, datatype=0)
#    return len(overlapping_polygons.polygons) > 0


#def raster_pixel(polygons, x = 0, y = 0, dx = 0.1, dy = 0.1, layers = None):
#    """ Checks to see if any of the polygons in polygons exist in the box-shaped region
#    from (x,y) to (x+dx, y+dy).  Returns ``True`` if any of the polgyons overlap that region """
#    xpts = [x, x, x+dx, x+dx]
#    ypts = [y, y+dy, y+dy, y]
#    checking_polygon = gdspy.Polygon(zip(xpts,ypts))
#    
#    
#    xmin = np.array([np.min(p[:,0]) for p in polygons])
#    xmax = np.array([np.max(p[:,0]) for p in polygons])
#    ymin = np.array([np.min(p[:,1]) for p in polygons])
#    ymax = np.array([np.max(p[:,1]) for p in polygons])
#
#    intersected_polygons_bool = np.less(xmin, x+dx) & np.greater(xmax, x) & np.less(ymin, y+dy) & np.greater(ymax, y)
#    intersected_polygons = [top_cell_polygons[i] for i in intersected_polygons_bool if i is True]
#
#    # TODO Replace fast_boolean with clipper.clip directly (avoid recreating polygon array each time)
#    overlapping_polygons = gdspy.fast_boolean(operandA = checking_polygon, operandB = intersected_polygons, \
#                    operation = 'and', precision=0.001, max_points=199, layer=0, datatype=0)
#    return len(overlapping_polygons.polygons) > 0


#def raster_pixel(polygons, bounding_boxes, x = 0, y = 0, dx = 0.1, dy = 0.1, layers = None):
#    """ Checks to see if any of the polygons in polygons exist in the box-shaped region
#    from (x,y) to (x+dx, y+dy).  Returns ``True`` if any of the polygons overlap that region """
#    xpts = [x, x, x+dx, x+dx]
#    ypts = [y, y+dy, y+dy, y]
##    checking_polygon = gdspy.Polygon(zip(xpts,ypts))
#    checking_polygon = zip(xpts, ypts)
#    
#    
#    xmin = bounding_boxes[0]
#    xmax = bounding_boxes[1]
#    ymin = bounding_boxes[2]
#    ymax = bounding_boxes[3]
#
#    intersected_polygons_bool = np.less(xmin, x+dx) & np.greater(xmax, x) & np.less(ymin, y+dy) & np.greater(ymax, y)
#    intersected_polygons = [top_cell_polygons[i].tolist() for i in intersected_polygons_bool if i is True]
#
#    # TODO Replace fast_boolean with clipper.clip directly (avoid recreating polygon array each time)
##    overlapping_polygons = gdspy.fast_boolean(operandA = checking_polygon, operandB = intersected_polygons, \
##                    operation = 'and', precision=0.001, max_points=1990, layer=0, datatype=0)
#    overlapping_polygons = gdspy.clipper.clip(intersected_polygons, [checking_polygon], 'and', 1/0.001)
#    return len(overlapping_polygons) > 0 # True if pixel contains polygons


# def raster_box(polygons, bbox = [[0,0],[2,2]], dx = 0.1, dy = 0.1):
#     xpts = np.arange(bbox[0][0], bbox[1][0], dx)
#     ypts = np.arange(bbox[0][1], bbox[1][1], dy)
    
#     bool_list = np.zeros([len(xpts), len(ypts)])
#     for ii, x in enumerate(xpts):
#         for jj, y in enumerate(ypts):
#             bool_list[ii][jj] = check_box_for_polygons(polygons, x, y, dx, dy)
#     return bool_list
    
# def raster_box_rtree(polygons, rtree_idx, bbox = [[0,0],[2,2]], dx = 0.1, dy = 0.1):
#     xpts = np.arange(bbox[0][0], bbox[1][0], dx)
#     ypts = np.arange(bbox[0][1], bbox[1][1], dy)
    
#     bool_list = np.zeros([len(xpts), len(ypts)])
#     for ii, x in enumerate(xpts):
#         for jj, y in enumerate(ypts):
#             bool_list[ii][jj] = check_box_for_polygons_rtree(polygons, rtree_idx, x, y, dx, dy)
#     return bool_list
    
    
#def rasterize_polygons(polygons, point1 = [0,0], point2 = [1,1], dx = 0.1, dy = 0.1):
#    """
#    Parameters
#    ----------
#    point1 : array-like[2]
#        Coordinates of a corner of the rectangle.
#    point2 : array-like[2]
#        Coordinates of the corner of the rectangle opposite to ``point1``.
#    """
#    xpts = np.arange(point1[0], point2[0], dx)
#    ypts = np.arange(point1[1], point2[1], dy)
#    
#    pixels = np.zeros([len(xpts), len(ypts)])
#    coords = np.zeros([len(xpts), len(ypts)]).tolist()
#    for ii, x in enumerate(xpts):
#        for jj, y in enumerate(ypts):
#            pixels[ii][jj] = raster_pixel(polygons, x, y, dx, dy)
#            coords[ii][jj] = (x,y)
#    return pixels, coords



#def rasterize_polygons(polygons, point1 = [0,0], point2 = [1,1], dx = 0.1, dy = 0.1):
#    """
#    Parameters
#    ----------
#    point1 : array-like[2]
#        Coordinates of a corner of the rectangle.
#    point2 : array-like[2]
#        Coordinates of the corner of the rectangle opposite to ``point1``.
#    """
#    xpts = np.arange(point1[0], point2[0], dx)
#    ypts = np.arange(point1[1], point2[1], dy)
#    
#    xmin = np.array([np.min(p[:,0]) for p in polygons])
#    xmax = np.array([np.max(p[:,0]) for p in polygons])
#    ymin = np.array([np.min(p[:,1]) for p in polygons])
#    ymax = np.array([np.max(p[:,1]) for p in polygons])
#    bounding_boxes = [xmin, xmax, ymin, ymax]
#    
#    pixels = np.zeros([len(xpts), len(ypts)])
#    coords = np.zeros([len(xpts), len(ypts)]).tolist()
#    for ii, x in enumerate(xpts):
#        for jj, y in enumerate(ypts):
#            pixels[ii][jj] = raster_pixel(polygons, bounding_boxes, x, y, dx, dy)
#            coords[ii][jj] = (x,y)
#    return pixels, coords

# # Test with  a single polygon
# top_cell = poly_cell
# top_cell_polygons = top_cell.get_polygons(by_spec=False, depth=None)
# check_box_for_polygons(top_cell_polygons, x = 0, y = 0, dx = 0.1, dy = 0.1)
# bool_list  = raster_box(top_cell_polygons)
# plt.imshow(bool_list)




# for cell_name in gds_file.cell_dict:
#     gds_file.extract(cell_name)
# gdspy.LayoutViewer()
# gdspy.Cell.cell_dict


# def raster_pixel(pixel_size, pixel_coords):
#     x = [0, 0, pixel_size, pixel_size]
#     y = [0,pixel_size,pixel_size,0]
#     xy = zip(x,y)
#     pixel_poly = gdspy.Polygon(xy, layer = 1)
     
#     poly_out = gdspy.fast_boolean(operandA = fill_bbox, operandB = polygons, operation = 'and', precision=0.001, max_points=199, layer=0, datatype=0)
#     if len(poly_out.polygons) == 0: return False:
#     else: return True
    

     
#import PIL
#from PIL import Image, ImageDraw
# 
#im = Image.new(mode = '1', size = (800, 600)) # mode=1 is black & white
#draw = ImageDraw.Draw(im)
#for p in top_cell_polygons:
#    draw.polygon(p, fill=1, outline=1)
#im.save('hellokitty1234.png', "PNG")

# xmin = np.array([np.min(p[:,0]) for p in top_cell_polygons])
# xmax = np.array([np.max(p[:,0]) for p in top_cell_polygons])
# ymin = np.array([np.min(p[:,1]) for p in top_cell_polygons])
# ymax = np.array([np.max(p[:,1]) for p in top_cell_polygons])

# x = -4555
# y = -3999
# dx = 1
# dy= 1
# np.greater(xmin, x+dx) | np.less(xmax, x+dx) | np.greater(ymin, y+dy) | np.less(ymax, y)
# np.less(xmin, x+dx) & np.greater(xmax, x) & np.less(ymin, y+dy) & np.greater(ymax, y)





# # Pseudocode for manually sorting polygons and filtering them before intersection
#     # Get bounding boxes for all polygons in a big list
#     # Created sorted arrays of the bounding boxes by xmin, xmax, ymin, ymax
#     # Prune the list by bounding boxes which have coordinates in the X section you're checking
#     # Prune the pruned list by the same method for y
#     # Perform intersection with remaining polygons

# # Pseudocode for using rTree
# # For p in polygons
# idx = rtree.index.Index()
# for n,p in enumerate(top_cell_polygons):
#     # Get bounding box
#     left, bottom, right, top = (min(p[:,0]), min(p[:,1]), max(p[:,0]), max(p[:,1]))
#     # Insert bounding boxes into rtree index
#     idx.insert(n, (left, bottom, right, top))
# intersected_polygons = top_cell_polygons(idx.intersection((-4500, -4500, 200, 200)))
# # Query which polygons are in the checking box
# # Clip against those polygons (%timeit)
# 

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
            
            
def fill_tool(exclude_area, exclude_layers, fill_cell, margin = None, bounds =  [[-100, -100], [100, 100]], dx = 10, dy = 10):
    if margin == None: margin = max([dx,dy])/2


# Test fill tool with large file
gdspy.Cell.cell_dict.clear()
filename = "C:/Users/anm16/Adam's Work/NIST Work/160718 Fill tool for gdspy python/sE04a.gds"
gds_file = gdspy.GdsImport(filename)
top_cell_name = 'nSS'
gds_file.extract(top_cell_name)
top_cell = gdspy.Cell.cell_dict['nSS']
top_cell_polygons = top_cell.get_polygons(by_spec=False, depth=None)
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





