# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
import itertools
from numpy import sqrt, pi, cos, sin, log, exp, sinh
# from scipy.interpolate import interp1d

import gdspy
from phidl.device_layout import Device, Port, Polygon, CellArray
from phidl.device_layout import _parse_layer, DeviceReference
import copy as python_copy
from collections import OrderedDict
import pickle
import json
import os
import warnings
from functools import update_wrapper



##### Categories:
# Polygons / shapes
# Boolean functions
# Lithography test structures
# Utility functions (copying, importing, extracting)
# Pads
# Taper
# Text
# Wafer / Die
# Waveguide
# Fill tool
# Photonics
# Optimal (current-crowding) curves
# Superconducting devices


#==============================================================================
#
# Polygons / Shapes
#
#==============================================================================


def rectangle(size = (4,2), layer = 0):
    """Generate rectangle geometry.

    Parameters
    ----------
    size : tuple
        Width and height of rectangle.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    A Device with a single rectangle in it
    """

    D = Device(name = 'rectangle')
    points = [[size[0], size[1]], [size[0], 0], [0, 0], [0, size[1]]]
    D.add_polygon(points, layer = layer)
    return D


def bbox(bbox = [(-1,-1),(3,4)], layer = 0):
    """ Creates a bounding box rectangle from coordinates, to allow
    creation of a rectangle bounding box directly form another shape.

    Parameters
    ----------
    bbox : list of tuples
        Coordinates of the box [(x1,y1),(x2,y2)].
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    A Device with a single rectangle in it

    Examples
    --------
    >>> D = pg.bbox(anothershape.bbox)
    """

    D = Device(name = 'bbox')
    (a,b),(c,d)  = bbox
    points = ((a,b), (c,b), (c,d), (a,d))
    D.add_polygon(points, layer = layer)
    return D


def cross(length = 10, width = 3, layer = 0):
    """Generates a right-angle cross (+ shape, symmetric) from two
    rectangles of specified length and width.

    Parameters
    ----------
    length : float
        Length of the cross from one end to the other.
    width : float
        Width of the arms of the cross.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    A Device with a cross in it
    """

    D = Device(name = 'cross')
    R = rectangle(size = (width, length), layer = layer)
    r1 = D.add_ref(R).rotate(90)
    r2 = D.add_ref(R)
    r1.center = (0,0)
    r2.center = (0,0)
    return D


def ellipse(radii = (10,5), angle_resolution = 2.5, layer = 0):
    """Generate an ellipse geometry.

    Parameters
    ----------
    radii : tuple
        Semimajor (x) and semiminor (y) axis lengths of the ellipse.
    angle_resolution : float
        Resolution of the curve of the ring (# of degrees per point).
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    A Device with an ellipse polygon in it
    """

    D = Device(name = 'ellipse')
    a = radii[0]
    b = radii[1]
    t = np.linspace(0, 360, int(np.ceil(360/angle_resolution) + 1))*pi/180
    r = a*b/(sqrt((b*cos(t))**2 + (a*sin(t))**2))
    xpts = r*cos(t)
    ypts = r*sin(t)
    D.add_polygon(points = (xpts,ypts), layer = layer)
    return D


def circle(radius = 10, angle_resolution = 2.5, layer = 0):
    """Generate a circle geometry.

    Parameters
    ----------
    radius : float
        Radius of the circle.
    angle_resolution : float
        Resolution of the curve of the ring (# of degrees per point).
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    A Device with an circle polygon in it

    """

    D = Device(name = 'circle')
    t = np.linspace(0, 360, int(np.ceil(360/angle_resolution) + 1))*pi/180
    xpts = (radius*cos(t)).tolist()
    ypts = (radius*sin(t)).tolist()
    D.add_polygon(points = (xpts,ypts), layer = layer)
    return D


def ring(radius = 10, width = 0.5, angle_resolution = 2.5, layer = 0):
    """Generate a ring geometry.

    Parameters
    ----------
    radius : float
        Middle radius of the ring.
    width : float
        Width of the ring.
    angle_resolution : float
        Resolution of the curve of the ring (# of degrees per point).
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    A Device with an ring polygon in it

    Notes
    -----
    The ring is formed by taking the radius out to the specified value, and then constructing the thickness by dividing the width in half and adding that value to either side of the radius.

    The angle_resolution alters the precision of the curve of the ring. Larger values yield lower resolution.
    """

    D = Device(name = 'ring')
    inner_radius = radius - width/2
    outer_radius = radius + width/2
    n = int(np.round(360/angle_resolution))
    t = np.linspace(0, 360, n+1)*pi/180
    inner_points_x = (inner_radius*cos(t)).tolist()
    inner_points_y = (inner_radius*sin(t)).tolist()
    outer_points_x = (outer_radius*cos(t)).tolist()
    outer_points_y = (outer_radius*sin(t)).tolist()
    xpts = inner_points_x + outer_points_x[::-1]
    ypts = inner_points_y + outer_points_y[::-1]
    D.add_polygon(points = (xpts,ypts), layer = layer)
    return D


def arc(radius = 10, width = 0.5, theta = 45, start_angle = 0, angle_resolution = 2.5, layer = 0):
    """ Creates an arc of arclength ``theta`` starting at angle ``start_angle``

    Parameters
    ----------
    radius : float
        Radius of the arc centerline.
    width : float
        Width of the arc.
    theta : float
        Total angle coverage of the arc.
    start_angle : float
        Starting angle.
    angle_resolution : float
        Resolution of the curve of the arc.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    A Device with an arc polygon in it, and two ports (`1` and `2`) on either end


    Notes
    -----
    Theta = 0 is located along the positive x-axis relative to the centre of the arc.
    Ports are added to each end of the arc to facilitate connecting those ends to other geometries.
    """

    inner_radius = radius-width/2
    outer_radius = radius+width/2
    angle1 = (start_angle)*pi/180
    angle2 = (start_angle + theta)*pi/180
    t = np.linspace(angle1, angle2, int(np.ceil(abs(theta)/angle_resolution)))
    inner_points_x = (inner_radius*cos(t)).tolist()
    inner_points_y = (inner_radius*sin(t)).tolist()
    outer_points_x = (outer_radius*cos(t)).tolist()
    outer_points_y = (outer_radius*sin(t)).tolist()
    xpts = inner_points_x + outer_points_x[::-1]
    ypts = inner_points_y + outer_points_y[::-1]

    D = Device('arc')
    D.add_polygon(points = (xpts,ypts), layer = layer)
    D.add_port(name = 1, midpoint = (radius*cos(angle1), radius*sin(angle1)),  width = width, orientation = start_angle - 90 + 180*(theta<0))
    D.add_port(name = 2, midpoint = (radius*cos(angle2), radius*sin(angle2)),  width = width, orientation = start_angle + theta + 90 - 180*(theta<0))
    D.info['length'] = (abs(theta)*pi/180)*radius
    return D


def turn(port, radius = 10, angle = 270, angle_resolution = 2.5, layer = 0):
    """ Starting form a port, create a arc which connects to the port
    """

    D = arc(radius = radius, width = port.width, theta = angle, start_angle = 0,
            angle_resolution = angle_resolution, layer = layer)
    D.rotate(angle =  180 + port.orientation - D.ports[1].orientation, center = D.ports[1].midpoint)
    D.move(origin = D.ports[1], destination = port)
    return D


def straight(size = (4,2), layer = 0):
    """Generates a rectangular wire geometry with ports on the length edges.

    Parameters
    ----------
    size : tuple
        The length and width of the rectangle.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    A Device with an rectangle polygon in it, and two ports (`1` and `2`) on either end

    Notes
    -----
    Ports are included on both sides of the length edge (i.e. size[0]) of the geometry.
    """

    D = Device(name = 'wire')
    points = [[size[0], size[1]], [size[0], 0], [0, 0], [0, size[1]]]
    D.add_polygon(points, layer = layer)
    D.add_port(name = 1, midpoint = (size[0]/2, size[1]),  width = size[0], orientation = 90)
    D.add_port(name = 2, midpoint = (size[0]/2, 0),  width = size[0], orientation = -90)
    return D


def L(width = 1, size = (10,20) , layer = 0):
    """Generates an "L" geometry with ports on both ends.

    Parameters
    ----------
    width : float
        Width of the L.
    size : tuple
        Lengths of the base and height of the L, respectively.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    A Device with an L-shaped polygon in it, and two ports (`1` and `2`) on
    either end of the L
    """

    D = Device(name = 'L')
    w = width/2
    s1, s2 = size
    points = [(-w,-w), (s1,-w), (s1,w), (w,w), (w,s2), (-w,s2), (-w,-w)]
    D.add_polygon(points, layer = layer)
    D.add_port(name = 1, midpoint = (0,s2),  width = width, orientation = 90)
    D.add_port(name = 2, midpoint = (s1, 0),  width = width, orientation = 0)
    return D


def C(width = 1, size = (10,20) , layer = 0):
    """Generates a "C" geometry with ports on both ends.

    Parameters
    ----------
    width : float
        Width of the C.
    size : tuple
        Lengths of the base + top edges and the height of the C, respectively.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    A Device with an [-bracket-shaped polygon in it, and two ports (`1` and `2`) on
    either end of the [ shape
    """

    D = Device(name = 'C')
    w = width/2
    s1, s2 = size
    points = [(-w,-w), (s1,-w), (s1,w), (w,w), (w,s2-w), (s1,s2-w), (s1,s2+w), (-w, s2+w), (-w,-w)]
    D.add_polygon(points, layer = layer)
    D.add_port(name = 1, midpoint = (s1,s2),  width = width, orientation = 0)
    D.add_port(name = 2, midpoint = (s1, 0),  width = width, orientation = 0)
    return D



#==============================================================================
#
# Boolean functions
#
#==============================================================================

def offset(elements, distance = 0.1, join_first = True, precision = 1e-6, layer = 0):
    if type(elements) is not list: elements = [elements]
    polygons_to_offset = []
    for e in elements:
        if isinstance(e, (Device, DeviceReference)): polygons_to_offset += e.get_polygons(by_spec = False)
        elif isinstance(e, (Polygon, gdspy.Polygon)): polygons_to_offset.append(e)
    polygons_to_offset = _merge_floating_point_errors(polygons_to_offset, tol = 1e-10)
    gds_layer, gds_datatype = _parse_layer(layer)
    p = gdspy.offset(polygons_to_offset, distance = distance, join='miter', tolerance=2,
                     precision=precision, join_first=join_first,
                     max_points=4000, layer=gds_layer, datatype = gds_datatype)
    D = Device('offset')
    D.add_polygon(p, layer=layer)
    return D


def inset(elements, distance = 0.1, join_first = True, precision = 1e-6, layer = 0):
    print('[PHIDL] pg.inset() is deprecated, please use pg.offset()')
    return offset(elements = elements, distance = -distance, join_first = join_first,
                 precision = precision, layer = layer)


def invert(elements, border = 10, precision = 1e-6, layer = 0):
    """ Creates an inverted version of the input shapes with an additional
    border around the edges """
    D = Device()
    if type(elements) is not list: elements = [elements]
    for e in elements:
        if isinstance(e, Device): D.add_ref(e)
        else: D.add(e)
    gds_layer, gds_datatype = _parse_layer(layer)

    # Build the rectangle around the device D
    R = rectangle(size = (D.xsize + 2*border, D.ysize + 2*border))
    R.center = D.center

    operand1 = R.get_polygons()
    operand2 = D.get_polygons()
    p = gdspy.fast_boolean(operand1, operand2, operation = 'not', precision=precision,
                 max_points=4000, layer=gds_layer, datatype=gds_datatype)

    D = Device('invert')
    D.add_polygon(p, layer=layer)
    return D


def boolean(A, B, operation, precision = 1e-6, layer = 0):
    """
    Performs boolean operations between 2 Device/DeviceReference objects,
    or lists of Devices/DeviceReferences.

    ``operation`` should be {not, and, or, xor, 'A-B', 'B-A', 'A+B'}.  Note
    that 'A+B' is equivalent to 'or', 'A-B' is equivalent to 'not', and
    'B-A' is equivalent to 'not' with the operands switched
    """
    A_polys = []
    B_polys = []
    if type(A) is not list: A = [A]
    if type(B) is not list: B = [B]
    for e in A:
        if isinstance(e, Device): A_polys += e.get_polygons()
        elif isinstance(e, DeviceReference): A_polys += e.get_polygons()
    for e in B:
        if isinstance(e, Device): B_polys += e.get_polygons()
        elif isinstance(e, DeviceReference): B_polys += e.get_polygons()

    gds_layer, gds_datatype = _parse_layer(layer)

    operation = operation.lower().replace(' ','')
    if operation == 'a-b':
        operation = 'not'
    elif operation == 'b-a':
        operation = 'not'
        A_polys, B_polys = B_polys, A_polys
    elif operation == 'a+b':
        operation = 'or'
    elif operation not in ['not', 'and', 'or', 'xor', 'a-b', 'b-a', 'a+b']:
        raise ValueError("[PHIDL] phidl.geometry.boolean() `operation` parameter not recognized, must be one of the following:  'not', 'and', 'or', 'xor', 'A-B', 'B-A', 'A+B'")


    p = gdspy.fast_boolean(operand1 = A_polys, operand2 = B_polys, operation = operation, precision=precision,
                 max_points=4000, layer=gds_layer, datatype=gds_datatype)

    D = Device('boolean')
    if p is not None: D.add_polygon(p, layer = layer)
    return D


def outline(elements, distance = 1, precision = 1e-6, layer = 0):
    """ Creates an outline around all the polygons passed in the `elements`
    argument.  `elements` may be a Device, Polygon, or list of Devices
    """
    D = Device('outline')
    if type(elements) is not list: elements = [elements]
    for e in elements:
        if isinstance(e, Device): D.add_ref(e)
        else: D.add(e)
    gds_layer, gds_datatype = _parse_layer(layer)

    D_bloated = offset(D, distance = distance, join_first = True, precision = precision, layer = layer)
    Outline = boolean(A = D_bloated, B = D, operation = 'A-B', precision = precision, layer = layer)
    return Outline


def xor_diff(A,B, precision = 1e-6):
    """ Given two Devices A and B, performs the layer-by-layer XOR
    difference between A and B, and returns polygons representing
    the differences between A and B.
    """
    D = Device()
    A_polys = A.get_polygons(by_spec = True)
    B_polys = B.get_polygons(by_spec = True)
    A_layers = A_polys.keys()
    B_layers = B_polys.keys()
    all_layers = set()
    all_layers.update(A_layers )
    all_layers.update(B_layers)
    for layer in all_layers:
        if (layer in A_layers) and (layer in B_layers):
            p = gdspy.fast_boolean(operand1 = A_polys[layer], operand2 = B_polys[layer],
                                   operation = 'xor', precision=precision,
                                   max_points=4000, layer=layer[0], datatype=layer[1])
        elif (layer in A_layers):
            p = A_polys[layer]
        elif (layer in B_layers):
            p = B_polys[layer]
        if p is not None:
            D.add_polygon(p, layer = layer)
    return D


def union(D, by_layer = False, precision=1e-6, layer = 0):
    U = Device()

    if by_layer == True:
        all_polygons = D.get_polygons(by_spec = True)
        for layer, polygons in all_polygons.items():
            unioned_polygons = _union_polygons(polygons, precision = precision)
            U.add_polygon(unioned_polygons, layer = layer)
    else:
        all_polygons = D.get_polygons(by_spec = False)
        unioned_polygons = _union_polygons(all_polygons, precision = precision)
        U.add_polygon(unioned_polygons, layer = layer)
    return U

def _union_polygons(polygons, precision=1e-6):
    polygons = _merge_floating_point_errors(polygons, tol = 1e-10)
    unioned = gdspy.fast_boolean(polygons, [], operation = 'or',
                                 precision=precision, max_points=4000)
    return unioned


def _merge_floating_point_errors(polygons, tol = 1e-10):
    stacked_polygons = np.vstack(polygons)
    x = stacked_polygons[:,0]
    y = stacked_polygons[:,1]
    polygon_indices = np.cumsum([len(p) for p in polygons])

    xfixed = _merge_nearby_floating_points(x, tol = tol)
    yfixed = _merge_nearby_floating_points(y, tol = tol)
    stacked_polygons_fixed = np.vstack([xfixed, yfixed]).T
    polygons_fixed = np.vsplit(stacked_polygons_fixed, polygon_indices)
    return polygons_fixed


def _merge_nearby_floating_points(x, tol = 1e-10):
    """ Takes an array `x` and merges any values within the tolerance `tol`
    So if given
    >>> x = [-2, -1, 0, 1.00001, 1.0002, 1.0003, 4, 5, 5.003, 6, 7, 8]
    >>> _merge_nearby_floating_points(x, tol = 1e-3)
    will then return:
    >>> [-2, -1, 0, 1.00001, 1.0001, 1.0001, 4, 5, 5.003, 6, 7, 8] """
    xargsort = np.argsort(x)
    xargunsort = np.argsort(xargsort)
    xsort = x[xargsort]
    xsortthreshold = (np.diff(xsort) < tol)
    xsortthresholdind = np.argwhere(xsortthreshold)

    # Merge nearby floating point values
    for xi in xsortthresholdind:
         xsort[xi+1] = xsort[xi]
    return xsort[xargunsort]


#==============================================================================
#
# Lithography test structures
#
#==============================================================================



def litho_steps(
        line_widths = [1,2,4,8,16],
        line_spacing = 10,
        height = 100,
        layer = 0
        ):
    """ Produces a positive + negative tone linewidth test, used for
    lithography resolution test patterning """
    D = Device('litho_steps')

    height = height / 2
    T1 = text(text = '%s' % str(line_widths[-1]),
        size = height, justify = 'center', layer = layer)
    t1 = D.add_ref(T1).rotate(90).movex(-height/10)
    R1 = rectangle(size = (line_spacing, height), layer = layer)
    r1 = D.add_ref(R1).movey(-height)
    count = 0
    for i in reversed(line_widths):
        count += line_spacing + i
        R2 = rectangle(size = (i, height), layer = layer)
        r1 = D.add_ref(R1).movex(count).movey(-height)
        r2 = D.add_ref(R2).movex(count - i)

    return(D)


def litho_star(
        num_lines = 20,
        line_width = 2,
        diameter = 200,
        layer = 0
        ):
    """ Creates a circular-star shape from lines, used as a lithographic
    resolution test pattern """
    D = Device('litho_star')

    degree = 180 / num_lines
    R1 = rectangle(size = (line_width, diameter), layer = layer)
    for i in range(num_lines):
        r1 = D.add_ref(R1).rotate(degree * i)
        r1.center = (0,0)

    return(D)


def litho_calipers(
        notch_size = [2,5],
        notch_spacing = 2,
        num_notches = 11,
        offset_per_notch = 0.1,
        row_spacing = 0,
        layer1 = 1,
        layer2 = 2):
    """ Creates a vernier caliper structure for lithography alignment
    tests.  Vernier structure is made horizontally. """

    D = Device('litho_calipers')
    num_notches_total = num_notches*2+1
    centre_notch = num_notches
    R1 = rectangle(size = (notch_size), layer = layer1)
    R2 = rectangle(size = (notch_size), layer = layer2)
    for i in range(num_notches_total):
        if i == centre_notch:
            r1 = D.add_ref(R1).movex(i * (notch_size[0] + notch_spacing)).movey(notch_size[1])
            r2 = D.add_ref(R2).movex(i * (notch_size[0] + notch_spacing) + offset_per_notch * (centre_notch - i)).movey(-2 * notch_size[1] - row_spacing)
        r1 = D.add_ref(R1).movex(i * (notch_size[0] + notch_spacing))
        r2 = D.add_ref(R2).movex(i * (notch_size[0] + notch_spacing) + offset_per_notch * (centre_notch - i)).movey(-notch_size[1] - row_spacing)

    return(D)



#==============================================================================
#
# Utility functions
#
#==============================================================================



def extract(D, layers = [0,1]):
    D_extracted = Device('extract')
    if type(layers) not in (list, tuple):
        raise ValueError('[PHIDL] pg.extract() Argument `layers` needs to be passed a list or tuple')
    poly_dict = D.get_polygons(by_spec = True)
    parsed_layer_list = [_parse_layer(layer) for layer in layers]
    for layer, polys in poly_dict.items():
        if _parse_layer(layer) in parsed_layer_list:
            D_extracted.add_polygon(polys, layer = layer)
    return D_extracted


def copy(D):
    D_copy = Device(name = D._internal_name)
    D_copy.info = python_copy.deepcopy(D.info)
    for ref in D.references:
        new_ref = DeviceReference(device = ref.parent,
                                origin = ref.origin,
                                rotation = ref.rotation,
                                magnification = ref.magnification,
                                x_reflection = ref.x_reflection)
        D_copy.add(new_ref)
        for alias_name, alias_ref in D.aliases.items():
            if alias_ref == ref: D_copy.aliases[alias_name] = new_ref

    for port in D.ports.values():      D_copy.add_port(port = port)
    for poly in D.polygons:   D_copy.add_polygon(poly)
    for label in D.labels:    D_copy.label(text = label.text,
                                           position = label.position,
                                           layer = (label.layer, label.texttype))
    return D_copy


def deepcopy(D):
    D_copy = python_copy.deepcopy(D)
    D_copy.uid = Device._next_uid
    Device._next_uid += 1
    D_copy._internal_name = D._internal_name
    D_copy.name = '%s%06d' % (D_copy._internal_name[:20], D_copy.uid) # Write name e.g. 'Unnamed000005'

    return D_copy


def copy_layer(D, layer = 1, new_layer = 2):
    D_copied_layer = extract(D, layers = [layer])
    D_copied_layer.flatten(single_layer = new_layer)
    return D_copied_layer


def import_gds(filename, cellname = None, flatten = False):
    gdsii_lib = gdspy.GdsLibrary()
    gdsii_lib.read_gds(filename)
    top_level_cells = gdsii_lib.top_level()
    if cellname is not None:
        if cellname not in gdsii_lib.cell_dict:
            raise ValueError('[PHIDL] import_gds() The requested cell (named %s) \
                        is not present in file %s' % (cellname,filename))
        topcell = gdsii_lib.cell_dict[cellname]
    elif cellname is None and len(top_level_cells) == 1:
        topcell = top_level_cells[0]
    elif cellname is None and len(top_level_cells) > 1:
        raise ValueError('[PHIDL] import_gds() There are multiple top-level cells, \
                        you must specify `cellname` to select of one of them')

    if flatten == False:
        D_list = []
        c2dmap = {}
        for cell in gdsii_lib.cell_dict.values():
            D = Device(name = cell.name)
            D.polygons = cell.polygons
            D.references = cell.references
            D.name = cell.name
            D.labels = cell.labels
            c2dmap.update({cell:D})
            D_list += [D]

        for D in D_list:
            # First convert each reference so it points to the right Device
            converted_references = []
            for e in D.references:
                ref_device = c2dmap[e.ref_cell]
                if isinstance(e, gdspy.CellReference):
                    dr = DeviceReference(
                        device = ref_device,
                        origin = e.origin,
                        rotation = e.rotation,
                        magnification = e.magnification,
                        x_reflection = e.x_reflection,
                        )
                    converted_references.append(dr)
                elif isinstance(e, gdspy.CellArray):
                    dr = CellArray(
                        device = ref_device,
                        columns = e.columns, 
                        rows = e.rows, 
                        spacing = e.spacing, 
                        origin = e.origin,
                        rotation = e.rotation,
                        magnification = e.magnification,
                        x_reflection = e.x_reflection,
                        )
                    converted_references.append(dr)
            D.references = converted_references
            # Next convert each Polygon
            temp_polygons = list(D.polygons)
            D.polygons = []
            for p in temp_polygons:
                D.add_polygon(p)

        topdevice = c2dmap[topcell]
        return topdevice

    elif flatten == True:
        D = Device('import_gds')
        polygons = topcell.get_polygons(by_spec = True)

        for layer_in_gds, polys in polygons.items():
            D.add_polygon(polys, layer = layer_in_gds)
        return D


def import_oas(filename, cellname = None, flatten = False):
    if filename.lower().endswith('.gds'):
        # you are looking for import_gds
        retval = import_gds(filename, cellname = cellname, flatten = flatten)
        return retval
    try:
        import klayout.db as pya
    except ImportError as err:
        err.args = ('[PHIDL] klayout package needed to import OASIS. pip install klayout\n' + err.args[0], ) + err.args[1:]
        raise
    if not filename.lower().endswith('.oas'): filename += '.oas'
    fileroot = os.path.splitext(filename)[0]
    tempfilename = fileroot + '-tmp.gds'

    layout = pya.Layout()
    layout.read(filename)
    # We want to end up with one Device. If the imported layout has multiple top cells,
    # a new toplevel is created, and they go into the second level
    if len(layout.top_cells()) > 1:
        topcell = layout.create_cell('toplevel')
        rot_DTrans = pya.DTrans.R0
        origin = pya.DPoint(0, 0)
        for childcell in layout.top_cells():
            if childcell == topcell: continue
            topcell.insert(pya.DCellInstArray(childcell.cell_index(), pya.DTrans(rot_DTrans, origin)))
    else:
        topcell = layout.top_cell()
    topcell.write(tempfilename)

    retval = import_gds(tempfilename, cellname = cellname, flatten = flatten)
    os.remove(tempfilename)
    return retval


def _translate_cell(c):
    D = Device(name = c.name)
    for e in c.elements:
        if isinstance(e, gdspy.PolygonSet):
            for n, points in enumerate(e.polygons):
                polygon_layer = _parse_layer((e.layers[n], e.datatypes[n]))
                D.add_polygon(points = points, layer = polygon_layer)
        elif isinstance(e, gdspy.CellReference):
            dr = DeviceReference(device = _translate_cell(e.ref_cell),
                            origin = e.origin,
                            rotation = e.rotation, magnification = None,
                            x_reflection = e.x_reflection)
            D.elements.append(dr)
    D.labels = c.labels
    return D


def preview_layerset(ls, size = 100):
    """ Generates a preview Device with representations of all the layers,
    used for previewing LayerSet color schemes in quickplot or saved .gds
    files """
    D = Device()
    scale = size/100
    num_layers = len(ls._layers)
    matrix_size = int(np.ceil(np.sqrt(num_layers)))
    sorted_layers = sorted(ls._layers.values(), key = lambda x: (x.gds_layer, x.gds_datatype))
    for n, layer in enumerate(sorted_layers):
        R = rectangle(size = (100*scale, 100*scale), layer = layer)
        T = text(
                text = '%s\n%s / %s' % (layer.name, layer.gds_layer, layer.gds_datatype),
                size = 20*scale,
                justify = 'center',
                layer = layer)

        T.move((50*scale,-20*scale))
        xloc = n % matrix_size
        yloc = int(n // matrix_size)
        D.add_ref(R).movex(200 * xloc *scale).movey(-200 * yloc*scale)
        D.add_ref(T).movex(200 * xloc *scale).movey(-200 * yloc*scale)
    return D

class device_lru_cache:
    def __init__(self, fn):
        self.maxsize = 32
        self.fn = fn
        self.memo = OrderedDict()
        update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        pickle_str = pickle.dumps(args, 1) + pickle.dumps(kwargs, 1)
        if pickle_str not in self.memo.keys():
            new_cache_item = self.fn(*args, **kwargs)
            if not isinstance(new_cache_item, Device):
                raise ValueError('[PHIDL] @device_lru_cache can only be used on functions which return a Device')
            if len(self.memo) > self.maxsize:
                self.memo.popitem(last = False) # Remove oldest item from cache
            # Add a deepcopy of new item to cache so that if we change the
            # returned device, our stored cache item is not changed
            self.memo[pickle_str] = python_copy.deepcopy(new_cache_item)
            return new_cache_item
        else: # if found in cache
            # Pop cache item out and put it back on the top of the cache
            cached_output = self.memo.pop(pickle_str)
            self.memo[pickle_str] = cached_output
            # Then return a copy of the cached Device
            return deepcopy(cached_output)


def _convert_port_to_geometry(port, layer = 0):
    ''' Converts a Port to a label and a triangle Device that are then added to the parent.
        The Port must start with a parent.
    '''
    if port.parent is None:
        raise ValueError('Port {}: Port needs a parent in which to draw'.format(port.name))

    # A visual marker
    triangle_points = [[0, 0]] * 3
    triangle_points[0] = port.endpoints[0]
    triangle_points[1] = port.endpoints[1]
    triangle_points[2] = (port.midpoint + (port.normal - port.midpoint) * port.width / 10)[1]
    port.parent.add_polygon(triangle_points, layer)

    # Label carrying actual information that will be recovered
    label_contents = (str(port.name),
                      # port.midpoint,  # rather than put this in the text, use the label position
                      float(np.round(port.width, decimals=3)),  # this can have rounding errors that are less than a nanometer
                      float(port.orientation),
                      # port.parent,  # this is definitely not serializable
                      # port.info,  # would like to include, but it might go longer than 1024 characters
                      # port.uid,  # not including because it is part of the build process, not the port state
                     )
    label_text = json.dumps(label_contents)
    port.parent.label(text = label_text, position = port.midpoint + _calculate_label_offset(port),
                      magnification = .04 * port.width, rotation = (90 + port.orientation) % 360,
                      layer = layer)


def _calculate_label_offset(port):
    ''' Used to put the label in a pretty position.
        It is added when drawing and substracted when extracting.
    '''
    offset_position = np.array((-np.cos(np.pi / 180 * port.orientation),
                                -np.sin(np.pi / 180 * port.orientation)))
    offset_position *= port.width * .05
    return offset_position


def _convert_geometry_to_port(label, layer = 0):
    ''' Converts a label into a Port in the parent Device.
        The label contains name, width, orientation.
        Does not remove that label from the parent.
        Returns the new port.
    '''
    name, width, orientation = json.loads(label.text)
    new_port = Port(name=name, width=width, orientation=orientation)
    new_port.midpoint = label.position - _calculate_label_offset(new_port)
    return new_port


def ports_to_geometry(device, layer = 0):
    ''' Converts Port objects over the whole Device hierarchy to geometry and labels.
        layer: the special port record layer
        Does not change the device used as argument. Returns a new one lacking all Ports.
    '''
    temp_device = deepcopy(device)
    all_cells = list(temp_device.get_dependencies(recursive=True))
    all_cells.append(temp_device)
    for subcell in all_cells:
        for port in subcell.ports.values():
            _convert_port_to_geometry(port, layer=layer)
            subcell.remove(port)
    return temp_device


def geometry_to_ports(device, layer = 0):
    ''' Converts geometry representing ports over the whole Device hierarchy into Port objects.
        layer: the special port record layer
        Does not mutate the device in the argument. Returns a new one lacking all port geometry (incl. labels)
    '''
    temp_device = deepcopy(device)
    all_cells = list(temp_device.get_dependencies(recursive=True))
    all_cells.append(temp_device)
    for subcell in all_cells: # Walk through cells
        for lab in subcell.labels:
            if lab.layer == layer:
                the_port = _convert_geometry_to_port(lab)
                subcell.add_port(name=the_port.name, port=the_port)
    temp_device.remove_layers(layers=[layer], include_labels=True)
    return temp_device


#==============================================================================
#
# Connectors
#
#==============================================================================

def connector(midpoint = (0,0), width = 1, orientation = 0):
    """ Creates a Device which has back-to-back ports """
    D = Device(name = 'connector')
    D.add_port(name = 1, midpoint = [midpoint[0], midpoint[1]],  width = width, orientation = orientation)
    D.add_port(name = 2, midpoint = [midpoint[0], midpoint[1]],  width = width, orientation = orientation-180)
    return D



#==============================================================================
#
# Contact pads
#
#==============================================================================


def compass(size = (4,2), layer = 0):
    """ Creates a rectangular contact pad with centered ports on edges of the
    rectangle (north, south, east, and west)
    """

    D = Device(name = 'compass')
    r = D.add_ref( rectangle(size, layer = layer) )
    r.center = (0,0)

    dx = size[0]
    dy = size[1]
    D.add_port(name = 'N', midpoint = [0, dy/2],  width = dx, orientation = 90)
    D.add_port(name = 'S', midpoint = [0, -dy/2], width = dx, orientation = -90)
    D.add_port(name = 'E', midpoint = [dx/2, 0],  width = dy, orientation = 0)
    D.add_port(name = 'W', midpoint = [-dx/2, 0], width = dy, orientation = 180)

    return D


def compass_multi(size = (4,2), ports = {'N':3,'S':4}, layer = 0):
    """ Creates a rectangular contact pad with multiple ports along the edges
    rectangle (north, south, east, and west).
    """

    D = Device(name = 'compass_multi')
    r = D.add_ref( rectangle(size, layer = layer) )
    r.center = (0,0)

    dx = size[0]/2
    dy = size[1]/2

    if 'N' in ports:
        num_ports = ports['N']
        m = dx-dx/num_ports
        p_list = np.linspace(-m, m, num_ports)
        [D.add_port(name = ('N%s' % (n+1)), midpoint = [p, dy],  width = dx/num_ports*2, orientation = 90) for n,p in enumerate(p_list)]
    if 'S' in ports:
        num_ports = ports['S']
        m = dx-dx/num_ports
        p_list = np.linspace(-m, m, num_ports)
        [D.add_port(name = ('S%s' % (n+1)), midpoint = [p, -dy],  width = dx/num_ports*2, orientation = -90) for n,p in enumerate(p_list)]
    if 'E' in ports:
        num_ports = ports['E']
        m = dy-dy/num_ports
        p_list = np.linspace(-m, m, num_ports)
        [D.add_port(name = ('E%s' % (n+1)), midpoint = [dx, p],  width = dy/num_ports*2, orientation = 0) for n,p in enumerate(p_list)]
    if 'W' in ports:
        num_ports = ports['W']
        m = dy-dy/num_ports
        p_list = np.linspace(-m, m, num_ports)
        [D.add_port(name = ('W%s' % (n+1)), midpoint = [-dx, p],  width = dy/num_ports*2, orientation = 180) for n,p in enumerate(p_list)]

    return D



# TODO: Fix the fillet here, right now only goes halfway down
def flagpole(size = (4,2), stub_size = (2,1), shape = 'p', taper_type = 'straight', layer = 0):
    f = np.array(size)
    p = np.array(stub_size)
    shape = shape.lower()

    assert shape in 'pqbd', '[DEVICE]  flagpole() shape must be p, q, b, or D'
    assert taper_type in ['straight','fillet'], '[DEVICE]  flagpole() taper_type must "straight" or "fillet" or None'

    if shape ==   'p':
        orientation = -90
    elif shape == 'q':
        f[0], p[0] = -size[0], -stub_size[0]
        orientation = -90
    elif shape == 'b':
        f[1], p[1] = -size[1], -stub_size[1]
        orientation = 90
    elif shape == 'D':
        f[1], p[1] = -size[1], -stub_size[1]
        f[0], p[0] = -size[0], -stub_size[0]
        orientation = 90
    xpts = [0, 0, f[0], f[0], p[0], p[0], 0]
    ypts = [0, f[1], f[1], 0, 0, -p[1], -p[1]]

    D = Device(name = 'flagpole')
    pad_poly = D.add_polygon([xpts,ypts], layer = layer)
    if taper_type == 'fillet':
        taper_amount = min([abs(f[0]-p[0]), abs(p[1])])
        pad_poly.fillet([0,0,0,0,taper_amount,0,0])
    elif taper_type == 'straight':
        D.add_polygon([xpts[3:6],ypts[3:6]], layer = layer)

    D.add_port(name = 1, midpoint = [p[0]/2, -p[1]],  width = abs(p[0]), orientation = orientation)
    D.add_port(name = 2, midpoint = [f[0]/2, f[1]],  width = abs(f[0]), orientation = orientation-180)
    return D


def tee(size = (4,2), stub_size = (2,1), taper_type = None, layer = 0):
    f = np.array(size)
    p = np.array(stub_size)

    xpts = np.array([f[0], f[0], p[0], p[0], -p[0], -p[0], -f[0], -f[0]])/2
    ypts = [f[1], 0, 0, -p[1], -p[1], 0, 0, f[1]]

    D = Device(name = 'tee')
    pad_poly = D.add_polygon([xpts,ypts], layer = layer)
    if taper_type == 'fillet':
        taper_amount = min([abs(f[0]-p[0]), abs(p[1])])
        pad_poly.fillet([0,0,taper_amount,0,0,taper_amount,0,0])
    elif taper_type == 'straight':
        taper_poly1 = D.add_polygon([xpts[1:4],ypts[1:4]], layer = layer)
        taper_poly2 = D.add_polygon([xpts[4:7],ypts[4:7]], layer = layer)

    D.add_port(name = 1, midpoint = [f[0]/2, f[1]/2],  width = f[1], orientation = 0)
    D.add_port(name = 2, midpoint = [-f[0]/2, f[1]/2],  width = f[1], orientation = 180)
    D.add_port(name = 3, midpoint = [0, -p[1]],  width = p[0], orientation = -90)
    return D


#==============================================================================
# Example code
#==============================================================================

#cp = compass(size = [4,2])
#quickplot(cp)


#cpm = compass_multi(size = [40,20], ports = {'N':3,'S':4, 'E':1, 'W':8}, layer = 0)
#quickplot(cpm)


#cpm = compass_multi(size = [40,20], ports = {'N':3,'S':4, 'E':1, 'W':8}, layer = 0)
#inset_polygon = offset(cpm, distance = -2, layer = 1)
#cpm.add_polygon(inset_polygon)
#quickplot(cpm)

#fp = flagpole(size = [4,2], stub_size = [2,1], shape = 'p', taper_type = 'straight', layer = 0)
#quickplot(fp)


#tp = tee(size = [4,2], stub_size = [2,1], taper_type = 'fillet', layer = 0)
#quickplot(tp)







#==============================================================================
#
# Tapers
#
#==============================================================================

# TODO change this so "width1" and "width2" arguments can accept Port directly
def taper(length = 10, width1 = 5, width2 = None, port = None, layer = 0):
    if type(port) is Port and width1 is None: width1 = port.width
    if width2 is None: width2 = width1
    xpts = [0, length, length, 0]
    ypts = [width1/2, width2/2, -width2/2, -width1/2]

    D = Device('taper')
    D.add_polygon([xpts,ypts], layer = layer)
    D.add_port(name = 1, midpoint = [0, 0], width = width1, orientation = 180)
    D.add_port(name = 2, midpoint = [length, 0], width = width2, orientation = 0)
    if type(port) is Port:
        D.rotate(angle = port.orientation, center = [0,0])
        D.move(origin = [0,0], destination = port.midpoint)
    return D


def ramp(length = 10, width1 = 5, width2 = 8, layer = 0):
    if width2 is None: width2 = width1
    xpts = [0, length, length, 0]
    ypts = [width1, width2, 0, 0]
    D = Device('ramp')
    D.add_polygon([xpts,ypts], layer = layer)
    D.add_port(name = 1, midpoint = [0, width1/2], width = width1, orientation = 180)
    D.add_port(name = 2, midpoint = [length, width2/2], width = width2, orientation = 0)
    return D


# Equations taken from
# Hammerstad, E., & Jensen, O. (1980). Accurate Models for Microstrip
# Computer-Aided Design.  http://doi.org/10.1109/MWSYM.1980.1124303
def _microstrip_Z(wire_width, dielectric_thickness, eps_r):
    # Note these equations can be further corrected for thick films (Hammersted Eqs 6-9)
    # and also for frequency since microstrips are dispersive  (Hammersted Eqs 10-12)

    u = wire_width/dielectric_thickness
    eta = 376.73 # Vacuum impedance

    a = 1 + log((u**4 + (u/52)**2)/(u**4 + 0.432))/49 + log(1 + (u/18.1)**3)/18.7;
    b = 0.564*((eps_r-0.9)/(eps_r+3))**0.053;
    F = 6 + (2*pi-6)*exp(-(30.666/u)**0.7528);
    eps_eff = 0.5*(eps_r+1) + 0.5*(eps_r-1)*(1 + 10/u)**(-a*b);
    Z = eta/(2*pi) * log(F/u + sqrt(1+(2/u)**2)) /sqrt(eps_eff);
    return Z,eps_eff


def _microstrip_LC_per_meter(wire_width, dielectric_thickness, eps_r):
    # Use the fact that v = 1/sqrt(L_m*C_m) = 1/sqrt(eps*mu) and
    # Z = sqrt(L_m/C_m)   [Where L_m is inductance per meter]

    Z, eps_eff =  _microstrip_Z(wire_width, dielectric_thickness, eps_r)
    eps0 =  8.854e-12
    mu0 = 4*pi*1e-7

    eps = eps_eff*eps0
    mu = mu0
    L_m = sqrt(eps*mu)*Z
    C_m = sqrt(eps*mu)/Z
    return L_m, C_m


def _microstrip_Z_with_Lk(wire_width, dielectric_thickness, eps_r, Lk_per_sq):
    # Add a kinetic inductance and recalculate the impedance, be careful
    # to input Lk as a per-meter inductance

    L_m, C_m = _microstrip_LC_per_meter(wire_width, dielectric_thickness, eps_r)
    Lk_m = Lk_per_sq*(1.0/wire_width)
    Z = sqrt((L_m+Lk_m)/C_m)
    return Z

def _microstrip_v_with_Lk(wire_width, dielectric_thickness, eps_r, Lk_per_sq):
    L_m, C_m = _microstrip_LC_per_meter(wire_width, dielectric_thickness, eps_r)
    Lk_m = Lk_per_sq*(1.0/wire_width)
    v = 1/sqrt((L_m+Lk_m)*C_m)
    return v

def _find_microstrip_wire_width(Z_target, dielectric_thickness, eps_r, Lk_per_sq):

    def error_fun(wire_width):
        Z_guessed = _microstrip_Z_with_Lk(wire_width, dielectric_thickness, eps_r, Lk_per_sq)
        return (Z_guessed-Z_target)**2 # The error

    x0 = dielectric_thickness
    try:
        from scipy.optimize import fmin
    except:
        raise ImportError(""" [PHIDL] To run the microsctrip functions you need scipy,
          please install it with `pip install scipy` """)
    w = fmin(error_fun, x0, args=(), disp=False)
    return w[0]

def _G_integrand(xip, B):
    try:
        from scipy.special import iv as besseli
    except:
        """ [PHIDL] To run this function you need scipy, please install it with
        pip install scipy """
    return besseli(0, B*sqrt(1-xip**2))


def _G(xi, B):
    try:
        from scipy.optimize import integrate
    except:
        raise ImportError(""" [PHIDL] To run the microsctrip functions you need scipy,
          please install it with `pip install scipy` """)
    return B/sinh(B)*integrate.quad(_G_integrand, 0, xi, args = (B))[0]

@device_lru_cache
def hecken_taper(length = 200, B = 4.0091, dielectric_thickness = 0.25, eps_r = 2,
                 Lk_per_sq = 250e-12, Z1 = None, Z2 = None, width1 = None, width2 = None,
                 num_pts = 100, layer = 0):
    if width1 is not None:  Z1 = _microstrip_Z_with_Lk(width1*1e-6, dielectric_thickness*1e-6, eps_r, Lk_per_sq)
    if width2 is not None:  Z2 = _microstrip_Z_with_Lk(width2*1e-6, dielectric_thickness*1e-6, eps_r, Lk_per_sq)
    xi_list = np.linspace(-1,1, num_pts) # Normalized length of the wire [-1 to +1]
    Z = [np.exp( 0.5*log(Z1*Z2) + 0.5*log(Z2/Z1)*_G(xi, B) ) for xi in xi_list]
    widths = np.array([_find_microstrip_wire_width(z, dielectric_thickness*1e-6, eps_r, Lk_per_sq)*1e6 for z in Z])
    x = ((xi_list/2)*length)

    # Compensate for varying speed of light in the microstrip by shortening
    # and lengthening sections according to the speed of light in that section
    v = np.array([_microstrip_v_with_Lk(w*1e-6, dielectric_thickness*1e-6, eps_r, Lk_per_sq) for w in widths])
    dx = np.diff(x)
    dx_compensated = dx/v[:-1]
    x_compensated = np.cumsum(dx_compensated)
    x = np.hstack([0,x_compensated])/max(x_compensated)*length

    # Create blank device and add taper polygon
    D = Device('hecken')
    xpts = np.concatenate([x, x[::-1]])
    ypts = np.concatenate([widths/2, -widths[::-1]/2])
    D.add_polygon((xpts,ypts), layer = layer)
    D.add_port(name = 1, midpoint = (0,0), width = widths[0], orientation = 180)
    D.add_port(name = 2, midpoint = (length,0), width = widths[-1], orientation = 0)

    # Add meta information about the taper
    D.info['num_squares'] = np.sum(np.diff(x)/widths[:-1])
    D.info['width1'] = widths[0]
    D.info['width2'] = widths[-1]
    D.info['Z1'] = Z[0]
    D.info['Z2'] = Z[-1]
    # Note there are two values for v/c (and f_cutoff) because the speed of
    # light is different at the beginning and end of the taper
    D.info['w'] = widths
    D.info['x'] = x
    D.info['Z'] = Z
    D.info['v/c'] = v/3e8
    BetaLmin = np.sqrt(B**2 + 6.523)
    D.info['f_cutoff'] = BetaLmin*D.info['v/c'][0]*3e8/(2*pi*length*1e-6)
    D.info['length'] = length

    return D


@device_lru_cache
def meander_taper(x_taper, w_taper, meander_length = 1000, spacing_factor = 3,
                  min_spacing = 0.5, layer = 0):

    def taper_width(x):
        return np.interp(x, x_taper, w_taper)


    def taper_section(x_start, x_end, num_pts = 30, layer = 0):
        D = Device('tapersec')
        length =  x_end - x_start
        x = np.linspace(0, length, num_pts)
        widths = np.linspace(taper_width(x_start), taper_width(x_end), num_pts)
        xpts = np.concatenate([x, x[::-1]])
        ypts = np.concatenate([widths/2, -widths[::-1]/2])
        D.add_polygon((xpts,ypts), layer = layer)
        D.add_port(name = 1, midpoint = (0,0), width = widths[0], orientation = 180)
        D.add_port(name = 2, midpoint = (length,0), width = widths[-1], orientation = 0)
        return D

    def arc_tapered(radius = 10, width1 = 1, width2 = 2, theta = 45, angle_resolution = 2.5, layer = 0):
        D = Device('arctaper')
        path1 = gdspy.Path(width = width1, initial_point = (0, 0))
        path1.turn(radius = radius, angle = theta*np.pi/180, number_of_points=int(abs(2*theta/angle_resolution)), final_width = width2)
        [D.add_polygon(p, layer = layer) for p in path1.polygons]
        D.add_port(name = 1, midpoint = (0, 0), width = width1, orientation = 180)
        D.add_port(name = 2, midpoint = (path1.x, path1.y), width = width2, orientation = path1.direction*180/np.pi)
        return D

    D = Device('meander-taper')
    xpos1 = min(x_taper)
    xpos2 = min(x_taper) + meander_length
    t = D.add_ref( taper_section(x_start = xpos1, x_end = xpos2, num_pts = 50, layer = layer) )
    D.add_port(t.ports[1])
    dir_toggle = -1
    while xpos2 < max(x_taper):
        arc_width1 = taper_width(xpos2)
        arc_radius = max(spacing_factor*arc_width1, min_spacing)
        arc_length = np.pi*arc_radius
        arc_width2 = taper_width(xpos2 + arc_length)
        A = arc_tapered(radius = arc_radius, width1 = arc_width1,
                        width2 = arc_width2, theta = 180*dir_toggle, layer = layer)
        a = D.add_ref(A)
        a.connect(port = 1, destination = t.ports[2])
        dir_toggle = -dir_toggle
        xpos1 = xpos2 + arc_length
        xpos2 = xpos1 + meander_length
        t = D.add_ref( taper_section(x_start = xpos1, x_end = xpos2, num_pts = 30, layer = layer) )
        t.connect(port = 1, destination = a.ports[2])
    D.add_port(t.ports[2])

    return D

#==============================================================================
# Example code
#==============================================================================

#D = racetrack_gradual(width, R = 5, N=3)
#quickplot(D)

# D = hecken_taper(length = 200, B = 4.0091, dielectric_thickness = 0.25, eps_r = 2,
#                  Lk_per_sq = 250e-12, Z1 = 50, width2 = 0.3,
#                  num_pts = 100, layer = 0)
# quickplot(D)

#t = np.linspace(0,1)
#x,y = _racetrack_gradual_parametric(t, R = 5, N = 3)
#plt.plot(x,y)
#plt.axis('equal')





#==============================================================================
#
# Text
#
#==============================================================================

# The DEPLOF font is made by David Elata, MEMS Lab, Technion, Haifa, Israel and
# and is used with permission.  The raw polygon entries are sourced from
# Ulf Griesmann's modified DEPLOF implementation used in the MATLAB gdsii
# toolbox https://github.com/ulfgri/gdsii-toolbox/ and are used with permission


_glyph = {}
_width = {}
_indent = {}

_glyph[33] = [[[100,-100],[100,100],[300,100],[300,-100],[100,-100]], [[100,250],[100,1100],[ 300,1100],[300,250],[100,250]]]
_glyph[34] = [[[300,800],[300,1200],[500,1200],[500,1000],[300,800]], [[600,800],[600,1200],[ 800,1200],[800,1000],[600,800]]]
_glyph[35] = [[[150,0],[170,200],[50,200],[50,400],[190,400],[210,600],[100,600],[100,800],[230,800],[ 250,1000],[450,1000],[390,400],[530,400],[510,200],[370,200],[350,0],[150,0]],  [[550,0],[610,600],[470,600],[490,800],[630,800],[650,1000],[850,1000],[830,800],[950,800],[ 950,600],[810,600],[790,400],[900,400],[900,200],[770,200],[750,0],[550,0]]]
_glyph[36] = [[[400,1000],[400,1200],[600,1200],[600,1000],[800,1000],[900,900],[900,800],[300,800],[ 300,600],[400,600],[400,700],[600,700],[600,600],[800,600],[900,500],[900,100],[ 800,0],[600,0],[600,-200],[400,-200],[400,0],[200,0],[100,100],[100,200],[700,200],[ 700,400],[600,400],[600,300],[400,300],[400,400],[200,400],[100,500],[100,900],[ 200,1000],[400,1000]]]
_glyph[37] = [[[100,100],[800,1000],[900,900],[200,0],[100,100]],  [[100,900],[400,900],[400,650],[350,600],[200,600],[300,700],[300,800],[200,800],[ 200,700],[100,600],[100,900]], [[650,400],[800,400],[700,300],[700,200],[ 800,200],[800,300],[900,400],[900,100],[600,100],[600,350],[650,400]]]
_glyph[38] = [[[700,0],[100,600],[100,800],[200,900],[400,900],[500,800],[500,800],[500,600],[ 450,550],[350,650],[400,700],[300,800],[200,700],[600,300],[700,400],[800,300],[ 700,200],[900,0],[700,0]],  [[550,50],[500,0],[100,0],[0,100],[0,300],[100,400],[150,450],[250,350],[100,200],[ 100,150],[150,100],[400,100],[450,150],[550,50]]]
_glyph[39] = [[[300,800],[300,1200],[500,1200],[500,1000],[300,800]]]
_glyph[40] = [[[100,500],[125,700],[175,900],[250,1100],[450,1100],[375,900],[325,700],[300,500],[ 325,300],[375,100],[450,-100],[250,-100],[175,100],[125,300],[100,500]]]
_glyph[41] = [[[100,1100],[300,1100],[375,900],[425,700],[450,500],[425,300],[375,100],[ 300,-100],[100,-100],[175,100],[225,300],[250,500],[225,700],[175,900],[100,1100]]]
_glyph[42] = [[[450,750],[450,1000],[550,1000],[550,750],[800,750],[800,650],[550,650],[550,400],[ 450,400],[450,650],[200,650],[200,750],[450,750]],  [[350,850],[250,850],[200,900],[200,1000],[300,1000],[350,950],[350,850]],  [[650,850],[650,950],[700,1000],[800,1000],[800,900],[750,850],[650,850]],  [[650,550],[750,550],[800,500],[800,400],[700,400],[650,450],[650,550]],  [[350,550],[350,450],[300,400],[200,400],[200,500],[250,550],[350,550]]]
# +
_glyph[43] = [[[400,600],[400,900],[600,900],[600,600],[900,600],[900,400],[600,400],[600,100],[ 400,100],[400,400],[100,400],[100,600],[400,600]]]
_glyph[44] = [[[300,200],[300,0],[100,-200],[100,200],[300,200]]]
_glyph[45] = [[[900,550],[900,350],[100,350],[100,550],[900,550]]]
_glyph[46] = [[[300,200],[300,0],[100,0],[100,200],[300,200]]]
_glyph[47] = [[[500,1200],[300,-200],[100,-200],[300,1200],[500,1200]]]
# 0
_glyph[48] = [[[400,800],[300,700],[300,300],[400,200],[350,0],[300,0],[100,200],[100,800],[ 300,1000],[530,1000],[480,800],[400,800]],  [[600,200],[700,300],[700,700],[600,800],[650,1000],[700,1000],[900,800],[ 900,200],[700,0],[470,0],[520,200],[600,200]]]
_glyph[49] = [[[200,600],[100,600],[100,800],[300,1000],[400,1000],[400,200],[500,200],[500,0],[ 100,0],[100,200],[200,200],[200,600],[200,600]]]
_glyph[50] = [[[100,900],[200,1000],[700,1000],[800,900],[800,600],[400,200],[800,200],[800,0],[ 100,0],[100,200],[600,700],[600,800],[300,800],[300,700],[100,700],[100,900]]]
_glyph[51] = [[[600,1000],[800,800],[800,600],[700,500],[800,400],[800,200],[600,0],[300,0],[ 100,200],[100,300],[300,300],[400,200],[500,200],[600,300],[500,400],[500,600],[ 600,700],[500,800],[400,800],[300,700],[100,700],[100,800],[300,1000],[600,1000]]]
_glyph[52] = [[[800,500],[800,300],[700,300],[700,0],[500,0],[500,300],[100,300],[100,1000],[300,1000],[ 300,500],[500,500],[500,1000],[700,1000],[700,500],[800,500]]]
_glyph[53] = [[[800,800],[300,800],[300,600],[700,600],[800,500],[800,100],[700,0],[200,0],[100,100],[ 100,300],[300,300],[300,200],[600,200],[600,400],[200,400],[100,500],[100,1000],[ 800,1000],[800,800]]]
_glyph[54] = [[[800,700],[600,700],[600,800],[300,800],[300,600],[700,600],[800,500],[800,100],[ 700,0],[500,0],[500,200],[600,200],[600,400],[300,400],[300,200],[400,200],[400,0],[ 200,0],[100,100],[100,900],[200,1000],[700,1000],[800,900],[800,700]]]
_glyph[55] = [[[560,800],[100,800],[100,1000],[800,1000],[600,0],[400,0],[480,400],[520,600],[560,800]]]
_glyph[56] = [[[400,800],[300,800],[300,600],[600,600],[600,800],[500,800],[500,1000],[ 700,1000],[800,900],[800,600],[700,500],[800,400],[800,100],[700,0],[ 500,0],[500,200],[600,200],[600,400],[300,400],[300,200],[400,200],[400,0],[ 200,0],[100,100],[100,400],[200,500],[100,600],[100,900],[200,1000],[400,1000],[400,800]]]
# 9
_glyph[57] = [[[100,300],[300,300],[300,200],[600,200],[600,400],[200,400],[100,500],[100,900],[ 200,1000],[400,1000],[400,800],[300,800],[300,600],[600,600],[600,800],[ 500,800],[500,1000],[700,1000],[800,900],[800,100],[700,0],[200,0],[100,100],[100,300]]]
_glyph[58] = [[[300,200],[300,0],[100,0],[100,200],[300,200]], [[300,600],[300,400],[100,400],[100,600],[300,600]]]
_glyph[59] = [[[300,200],[300,0],[100,-200],[100,200],[300,200]], [[300,600],[300,400],[100,400],[100,600],[300,600]]]
_glyph[60] = [[[700,900],[700,700],[400,500],[700,300],[700,100],[100,500],[700,900]]]
_glyph[61] = [[[100,400],[900,400],[900,200],[100,200],[100,400]], [[100,800],[900,800],[900,600],[100,600],[100,800]]]
_glyph[62] = [[[700,500],[100,100],[100,300],[400,500],[100,700],[100,900],[700,500]]]
_glyph[63] = [[[100,1000],[200,1100],[800,1100],[900,1000],[900,500],[800,400],[600,400],[ 600,200],[400,200],[400,500],[500,600],[700,600],[700,900],[300,900],[300,800],[100,800],[100,1000]],  [[600,50],[600,-150],[400,-150],[400,50],[600,50]]]
_glyph[64] = [[[900,200],[900,100],[800,0],[300,0],[100,200],[100,800],[300,1000],[700,1000],[ 900,800],[900,500],[800,400],[450,400],[400,450],[400,600],[450,700],[600,700],[ 550,600],[550,500],[700,500],[700,700],[600,800],[400,800],[300,700],[300,300],[ 400,200],[900,200]]]
# A
_glyph[65] = [[[100,800],[300,1000],[601,1000],[800,800],[800,0],[601,0],[601,200],[500,200],[ 500,400],[601,400],[601,700],[500,800],[400,800],[300,700],[300,400],[400,400],[ 400,200],[300,200],[300,0],[99,0],[100,800]]]
_glyph[66] = [[[600,1000],[800,800],[800,600],[700,500],[800,400],[800,200],[600,0],[100,0],[100,400],[ 300,400],[300,200],[500,200],[600,300],[500,400],[420,400],[420,600],[500,600],[ 600,700],[500,800],[300,800],[300,600],[100,600],[100,1000],[600,1000]]]
_glyph[67] = [[[300,0],[100,200],[100,800],[300,1000],[600,1000],[800,800],[800,600],[600,600],[ 600,700],[500,800],[400,800],[300,700],[300,300],[400,200],[500,200],[600,300],[ 600,400],[800,400],[800,200],[600,0],[300,0]]]
_glyph[68] = [[[100,0],[100,400],[300,400],[300,200],[500,200],[600,300],[600,700],[500,800],[ 300,800],[300,600],[100,600],[100,1000],[600,1000],[800,800],[800,200],[600,0],[100,0]]]
_glyph[69] = [[[700,1000],[700,800],[300,800],[300,600],[500,600],[500,400],[300,400],[300,200],[ 700,200],[700,0],[100,0],[100,1000],[700,1000]]]
_glyph[70] = [[[100,0],[100,1000],[700,1000],[700,800],[300,800],[300,600],[500,600],[500,400],[300,400],[300,0],[100,0]]]
_glyph[71] = [[[300,0],[100,200],[100,800],[300,1000],[600,1000],[800,800],[800,700],[600,700],[500,800],[ 400,800],[300,700],[300,300],[400,200],[600,200],[600,300],[500,300],[500,500],[800,500],[ 800,100],[700,0],[300,0]]]
_glyph[72] = [[[100,1000],[300,1000],[300,600],[600,600],[600,1000],[800,1000],[800,0],[600,0],[600,400],[ 300,400],[300,0],[100,0],[100,1000]]]
_glyph[73] = [[[100,0],[100,200],[300,200],[300,800],[100,800],[100,1000],[700,1000],[700,800],[500,800],[ 500,200],[700,200],[700,0],[100,0]]]
_glyph[74] = [[[300,200],[500,200],[500,1000],[700,1000],[700,100],[600,0],[200,0],[100,100],[100,300],[ 300,300],[300,200]]]
_glyph[75] = [[[100,1000],[300,1000],[300,600],[600,1000],[800,1000],[800,900],[500,500],[800,100],[ 800,0],[600,0],[300,400],[300,0],[100,0],[100,1000]]]
_glyph[76] = [[[100,1000],[300,1000],[300,200],[800,200],[800,0],[100,0],[100,1000]]]
_glyph[77] = [[[100,1000],[300,1000],[500,700],[700,1000],[900,1000],[900,0],[700,0],[700,600],[500,300],[ 300,600],[300,0],[100,0],[100,1000]]]
_glyph[78] = [[[100,1000],[300,1000],[700,400],[700,1000],[900,1000],[900,0],[700,0],[300,600],[300,0],[100,0],[100,1000]]]
_glyph[79] = [[[100,800],[300,1000],[430,1000],[430,800],[400,800],[300,700],[300,300],[400,200],[600,200],[ 700,300],[700,700],[600,800],[570,800],[570,1000],[700,1000],[900,800],[900,200],[700,0],[ 300,0],[100,200],[100,800]]]
_glyph[80] = [[[100,1000],[700,1000],[900,800],[900,600],[700,400],[500,400],[500,600],[600,600],[700,700],[ 600,800],[300,800],[300,0],[100,0],[100,1000]]]
_glyph[81] = [[[100,800],[300,1000],[700,1000],[900,800],[900,200],[800,100],[900,0],[600,0],[600,400],[ 700,400],[700,700],[600,800],[400,800],[300,700],[300,300],[400,200],[400,0],[300,0],[ 100,200],[100,800]]]
_glyph[82] = [[[100,1000],[700,1000],[900,800],[900,600],[700,400],[900,200],[900,0],[700,0],[700,100],[ 500,300],[500,600],[600,600],[700,700],[600,800],[300,800],[300,0],[100,0],[100,1000]]]
_glyph[83] = [[[900,800],[300,800],[300,600],[800,600],[900,500],[900,100],[800,0],[200,0],[100,100],[ 100,200],[700,200],[700,400],[200,400],[100,500],[100,900],[200,1000],[800,1000],[900,900],[900,800]]]
_glyph[84] = [[[900,1000],[900,800],[600,800],[600,0],[400,0],[400,800],[100,800],[100,1000],[900,1000]]]
_glyph[85] = [[[300,1000],[300,300],[400,200],[500,200],[600,300],[600,1000],[800,1000],[800,200],[ 600,0],[300,0],[100,200],[100,1000],[300,1000]]]
_glyph[86] = [[[300,1000],[500,400],[700,1000],[900,1000],[600,0],[400,0],[100,1000],[300,1000]]]
_glyph[87] = [[[100,1000],[300,1000],[300,400],[500,700],[700,400],[700,1000],[900,1000],[900,0],[ 700,0],[500,300],[300,0],[100,0],[100,1000]]]
_glyph[88] = [[[367,500],[100,900],[100,1000],[300,1000],[500,700],[700,1000],[900,1000],[ 900,900],[633,500],[900,100],[900,0],[700,0],[500,300],[300,0],[100,0],[100,100],[367,500]]]
_glyph[89] = [[[600,450],[600,0],[400,0],[400,450],[100,900],[100,1000],[300,1000],[500,700],[ 700,1000],[900,1000],[900,900],[600,450]]]
# Z
_glyph[90] = [[[100,1000],[900,1000],[900,700],[300,200],[900,200],[900,0],[100,0],[100,300],[ 700,800],[100,800],[100,1000]]]
_glyph[91] = [[[400,1200],[400,1000],[300,1000],[300,0],[400,0],[400,-200],[100,-200],[100,1200],[400,1200]]]
_glyph[92] = [[[300,1200],[500,-200],[300,-200],[100,1200],[300,1200]]]
_glyph[93] = [[[400,1200],[400,-200],[100,-200],[100,0],[200,0],[200,1000],[100,1000],[100,1200],[400,1200]]]
_glyph[94] = [[[0,500],[400,900],[800,500],[600,500],[400,700],[200,500],[0,500]]]
_glyph[95] = [[[100,200],[900,200],[900,0],[100,0],[100,200]]]
_glyph[96] = [[[300,1000],[300,1200],[500,1200],[500,800],[300,1000]]]
# a
_glyph[97] = [[[800,0],[300,0],[100,200],[100,500],[334,700],[600,700],[600,775],[800,775],[800,400],[ 600,400],[600,500],[400,500],[300,400],[300,300],[400,200],[600,200],[600,300],[800,300],[800,0]]]
_glyph[98] = [[[100,300],[300,300],[300,200],[500,200],[600,300],[600,400],[500,500],[300,500],[300,400],[ 100,400],[100,1000],[300,1000],[300,700],[600,700],[800,500],[800,200],[600,0],[100,0],[100,300]]]
_glyph[99] = [[[800,200],[600,0],[300,0],[100,200],[100,500],[300,700],[600,700],[800,500],[800,400],[ 600,400],[500,500],[400,500],[300,400],[300,300],[400,200],[500,200],[600,300],[800,300],[800,200]]]
_glyph[100] = [[[800,0],[300,0],[100,200],[100,500],[300,700],[600,700],[600,1000],[800,1000],[800,400],[ 600,400],[600,500],[400,500],[300,400],[300,300],[400,200],[600,200],[600,300],[800,300],[800,0]]]
_glyph[101] = [[[200,0],[100,100],[100,700],[200,800],[700,800],[800,700],[800,400],[700,300],[440,300],[ 440,500],[600,500],[600,600],[300,600],[300,200],[800,200],[800,100],[700,0],[200,0]]]
_glyph[102] = [[[600,800],[300,800],[300,600],[500,600],[500,400],[300,400],[300,0],[100,0],[100,900],[ 200,1000],[600,1000],[600,800]]]
_glyph[103] = [[[800,400],[600,400],[600,500],[400,500],[300,400],[300,300],[400,200],[600,200],[ 600,300],[800,300],[800,-200],[700,-300],[300,-300],[200,-200],[100,-100],[ 600,-100],[600,0],[334,0],[100,200],[100,500],[300,700],[800,700],[800,400]]]
_glyph[104] = [[[600,0],[600,400],[500,500],[400,500],[300,400],[300,0],[100,0],[100,1100],[300,1100],[ 300,600],[400,700],[600,700],[800,500],[800,0],[600,0]]]
_glyph[105] = [[[100,0],[100,600],[300,600],[300,0],[100,0]],  [[300,1000],[300,800],[100,800],[100,1000],[300,1000]]]
_glyph[106] = [[[100,-100],[100,0],[300,0],[300,600],[500,600],[500,-100],[400,-200],[200,-200],[100,-100]],  [[500,1000],[500,800],[300,800],[300,1000],[500,1000]]]
_glyph[107] = [[[300,500],[600,700],[800,700],[800,600],[500,400],[800,100],[800,0],[600,0],[ 300,300],[300,0],[100,0],[100,1100],[300,1100],[300,500]]]
_glyph[108] = [[[500,0],[200,0],[100,100],[100,1000],[300,1000],[300,200],[500,200],[500,0]]]
_glyph[109] = [[[500,400],[400,500],[300,400],[300,0],[100,0],[100,700],[300,700],[300,600],[ 400,700],[500,700],[600,600],[700,700],[900,700],[1100,500],[1100,0],[900,0],[ 900,400],[800,500],[700,400],[700,0],[500,0],[500,400]]]
_glyph[110] = [[[600,0],[600,400],[500,500],[400,500],[300,400],[300,0],[100,0],[100,700],[300,700],[ 300,600],[400,700],[600,700],[800,500],[800,0],[600,0]]]
_glyph[111] = [[[600,700],[800,500],[800,200],[600,0],[300,0],[100,200],[100,500],[300,700],[ 400,700],[400,500],[300,400],[300,300],[400,200],[500,200],[600,300],[600,400],[ 500,500],[500,700],[600,700]]]
_glyph[112] = [[[100,700],[600,700],[800,500],[800,200],[600,0],[300,0],[300,-300],[100,-300],[ 100,300],[300,300],[300,200],[500,200],[600,300],[600,400],[500,500],[300,500],[ 300,400],[100,400],[100,700]]]
_glyph[113] = [[[800,400],[600,400],[600,500],[400,500],[300,400],[300,300],[400,200],[600,200],[ 600,300],[800,300],[800,-300],[600,-300],[600,0],[300,0],[100,200],[100,500],[ 300,700],[800,700],[800,400]]]
_glyph[114] = [[[600,400],[600,500],[400,500],[300,400],[300,0],[100,0],[100,700],[300,700],[ 300,600],[400,700],[700,700],[800,600],[800,400],[600,400]]]
_glyph[115] = [[[200,0],[100,100],[100,200],[600,200],[600,300],[200,300],[100,400],[100,700],[ 200,800],[700,800],[800,700],[800,600],[300,600],[300,500],[700,500],[800,400],[ 800,100],[700,0],[200,0]]]
_glyph[116] = [[[600,0],[400,0],[300,100],[300,600],[100,600],[100,800],[300,800],[300,1000],[ 500,1000],[500,800],[700,800],[700,600],[500,600],[500,200],[600,200],[600,0]]]
_glyph[117] = [[[300,700],[300,300],[400,200],[500,200],[600,300],[600,700],[800,700],[800,0],[ 600,0],[600,100],[500,0],[300,0],[100,200],[100,700],[300,700]]]
_glyph[118] = [[[300,0],[100,700],[300,700],[400,350],[500,700],[700,700],[500,0],[300,0]]]
_glyph[119] = [[[600,350],[500,0],[300,0],[100,700],[300,700],[400,350],[500,700],[700,700],[ 800,350],[900,700],[1100,700],[900,0],[700,0],[600,350]]]
_glyph[120] = [[[308,350],[100,600],[100,700],[300,700],[450,520],[600,700],[800,700],[ 800,600],[592,350],[800,100],[800,0],[600,0],[450,180],[300,0],[100,0],[100,100],[308,350]]]
_glyph[121] = [[[214,-300],[300,0],[100,700],[300,700],[400,350],[500,700],[700,700],[500,0],[414,-300],[214,-300]]]
#z
_glyph[122] = [[[100,500],[100,700],[700,700],[700,500],[400,200],[700,200],[700,0],[100,0],[ 100,200],[400,500],[100,500]]]
_glyph[123] = [[[100,500],[200,600],[200,1000],[400,1200],[500,1200],[500,1000],[400,1000],[ 400,600],[300,500],[400,400],[400,0],[500,0],[500,-200],[400,-200],[200,0],[200,400],[100,500]]]
_glyph[124] = [[[100,-100],[100,1100],[300,1100],[300,-100],[100,-100]]]
_glyph[125] = [[[500,500],[400,600],[400,1000],[200,1200],[100,1200],[100,1000],[200,1000],[ 200,600],[300,500],[200,400],[200,0],[100,0],[100,-200],[200,-200],[400,0],[400,400],[500,500]]]
_glyph[126] = [[[100,700],[250,800],[350,800],[650,600],[750,600],[900,700],[ 900,500],[ 750,400],[650,400],[350,600],[250,600],[100,500],[100,700]]]
_glyph[181] = [[[300,700],[300,300],[400,200],[500,200],[600,300],[600,700],[800,700],[800,0],[600,0],[ 600,100],[500,0],[400,0],[300,100],[300,-300],[100,-300],[100,700],[300,700]]]


# _glyph _widths and _indents
_width[33] = 400;  _indent[33] = 100 # !
_width[34] = 500;  _indent[34] = 200 # "
_width[35] = 800;  _indent[35] = 100 # #
_width[36] = 800;  _indent[36] = 100 # $
_width[37] = 800;  _indent[37] = 100 # #
_width[38] = 900;  _indent[38] = 000 # &
_width[39] = 200;  _indent[39] = 300 # '
_width[40] = 450;  _indent[40] = 100 # (
_width[41] = 450;  _indent[41] = 100 # )
_width[42] = 600;  _indent[42] = 200 # *
_width[43] = 800;  _indent[43] = 100
_width[44] = 200;  _indent[44] = 100
_width[45] = 800;  _indent[45] = 100
_width[46] = 200;  _indent[46] = 100
_width[47] = 400;  _indent[47] = 100
_width[48] = 800;  _indent[48] = 100
_width[49] = 400;  _indent[49] = 100
_width[50] = 700;  _indent[50] = 100
_width[51] = 700;  _indent[51] = 100
_width[52] = 700;  _indent[52] = 100
_width[53] = 700;  _indent[53] = 100
_width[54] = 700;  _indent[54] = 100
_width[55] = 700;  _indent[55] = 100
_width[56] = 700;  _indent[56] = 100
_width[57] = 700;  _indent[57] = 100
_width[58] = 200;  _indent[58] = 100
_width[59] = 200;  _indent[59] = 100
_width[60] = 600;  _indent[60] = 100
_width[61] = 800;  _indent[61] = 100
_width[62] = 600;  _indent[62] = 100
_width[63] = 800;  _indent[63] = 100
_width[64] = 800;  _indent[64] = 100
_width[65] = 700;  _indent[65] = 100
_width[66] = 700;  _indent[66] = 100
_width[67] = 700;  _indent[67] = 100
_width[68] = 700;  _indent[68] = 100
_width[69] = 600;  _indent[69] = 100
_width[70] = 600;  _indent[70] = 100
_width[71] = 700;  _indent[71] = 100
_width[72] = 700;  _indent[72] = 100
_width[73] = 600;  _indent[73] = 100
_width[74] = 600;  _indent[74] = 100
_width[75] = 700;  _indent[75] = 100
_width[76] = 700;  _indent[76] = 100
_width[77] = 800;  _indent[77] = 100
_width[78] = 800;  _indent[78] = 100
_width[79] = 800;  _indent[79] = 100
_width[80] = 800;  _indent[80] = 100
_width[81] = 800;  _indent[81] = 100
_width[82] = 800;  _indent[82] = 100
_width[83] = 800;  _indent[83] = 100
_width[84] = 800;  _indent[84] = 100
_width[85] = 700;  _indent[85] = 100
_width[86] = 800;  _indent[86] = 100
_width[87] = 800;  _indent[87] = 100
_width[88] = 800;  _indent[88] = 100
_width[89] = 800;  _indent[89] = 100
_width[90] = 800;  _indent[90] = 100
_width[91] = 300;  _indent[91] = 100
_width[92] = 400;  _indent[92] = 100
_width[93] = 300;  _indent[93] = 100
_width[94] = 800;  _indent[94] = 000
_width[95] = 800;  _indent[95] = 100
_width[96] = 200;  _indent[96] = 300
_width[97] = 700;  _indent[97] = 100
_width[98] = 700;  _indent[98] = 100
_width[99] = 700;  _indent[99] = 100
_width[100] = 700;  _indent[100] = 100
_width[101] = 700;  _indent[101] = 100
_width[102] = 500;  _indent[102] = 100
_width[103] = 700;  _indent[103] = 100
_width[104] = 700;  _indent[104] = 100
_width[105] = 200;  _indent[105] = 100
_width[106] = 400;  _indent[106] = 100
_width[107] = 700;  _indent[107] = 100
_width[108] = 400;  _indent[108] = 100
_width[109] = 1000;  _indent[109] = 100
_width[110] = 700;  _indent[110] = 100
_width[111] = 700;  _indent[111] = 100
_width[112] = 700;  _indent[112] = 100
_width[113] = 700;  _indent[113] = 100
_width[114] = 700;  _indent[114] = 100
_width[115] = 700;  _indent[115] = 100
_width[116] = 600;  _indent[116] = 100
_width[117] = 700;  _indent[117] = 100
_width[118] = 600;  _indent[118] = 100
_width[119] = 1000;  _indent[119] = 100
_width[120] = 700;  _indent[120] = 100
_width[121] = 600;  _indent[121] = 100
_width[122] = 600;  _indent[122] = 100
_width[123] = 500;  _indent[123] = 100  # {
_width[124] = 400;  _indent[124] = 100  # |
_width[125] = 500;  _indent[125] = 100  # }
_width[126] = 800;  _indent[126] = 100  # ~
_width[181] = 700;  _indent[181] = 100  # Greek mu

def text(text = 'abcd', size = 10, justify = 'left', layer = 0):
    scaling = size/1000
    position = (0,0)
    xoffset = 0
    yoffset = 0
    t = Device('text')
    for line in text.split('\n'):
        l = Device(name = 'textline')
        for c in line:
            ascii_val = ord(c)
            if c == ' ':
                xoffset += 500*scaling
            elif (33 <= ascii_val <= 126) or (ascii_val == 181):
                for poly in _glyph[ascii_val]:
                    xpts = np.array(poly)[:,0]*scaling
                    ypts = np.array(poly)[:,1]*scaling
                    l.add_polygon([xpts + xoffset,ypts + yoffset], layer=layer)
                xoffset += (_width[ascii_val] + _indent[ascii_val])*scaling
            else:
                valid_chars = '!"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~µ'
                warnings.warn('[PHIDL] text(): Warning, some characters ignored, no geometry for character "%s" with ascii value %s. ' \
                'Valid characters: %s'  % (chr(ascii_val), ascii_val,valid_chars))
        t.add_ref(l)
        yoffset -= 1500*scaling
        xoffset = position[0]
    justify = justify.lower()
    for l in t.references:
        if justify == 'left':   pass
        if justify == 'right':  l.xmax = position[0]
        if justify == 'center': l.move(origin = l.center, destination = position, axis = 'x')
    t.flatten()
    return t


#==============================================================================
# Example code
#==============================================================================

#D = text('the quick brown\n fox jumped over\nthe lazy dog', justify = 'center', size = 800)
#quickplot(D)




#==============================================================================
#
# Wafer and die
#
#==============================================================================



def basic_die(
              size = (10000, 10000),
              street_width = 100,
              street_length = 1000,
              die_name = 'chip99',
              text_size = 100,
              text_location = 'SW',
              layer = 0,
              draw_bbox = True,
              bbox_layer = 99,
              ):

    #==========================================================================
    #  Create the basic geometry
    #==========================================================================
    D = Device(name = 'die')
    xpts = np.array([size[0]/2, size[0]/2, size[0]/2-street_width, size[0]/2-street_width, size[0]/2-street_length, size[0]/2-street_length])
    ypts = np.array([size[1]/2, size[1]/2-street_length, size[0]/2-street_length, size[0]/2-street_width, size[0]/2-street_width, size[1]/2])
    D.add_polygon([xpts,ypts], layer = layer)
    D.add_polygon([-xpts,ypts], layer = layer)
    D.add_polygon([xpts,-ypts], layer = layer)
    D.add_polygon([-xpts,-ypts], layer = layer)

    if draw_bbox is True:
        s = np.array(size)/2
        D.add_polygon([[s[0],s[1]], [s[0],-s[1]],[-s[0],-s[1]],[-s[0],s[1]]], layer = bbox_layer)
    D.center = (0,0)
    t = D.add_ref(text(text = die_name, size = text_size, layer=layer))

    d = street_width + 20
    sx, sy = size[0]/2, size[1]/2
    if type(text_location) is str:
        if text_location.upper() == 'NW':
            t.xmin, t.ymax = [-sx + d, sy - d]
        elif text_location.upper() == 'N':
            t.x, t.ymax = [0, sy - d]
        elif text_location.upper() == 'NE':
            t.xmax, t.ymax = [sx - d, sy - d]
        if text_location.upper() == 'SW':
            t.xmin, t.ymin = [-sx + d, -sy + d]
        elif text_location.upper() == 'S':
            t.x, t.ymin = [0, -sy + d]
        elif text_location.upper() == 'SE':
            t.xmax, t.ymin = [sx - d, -sy + d]
    else:
        t.x, t.y = text_location

    return D



#==============================================================================
# Example code
#==============================================================================

# D = basic_die(size = (10000, 10000), street_width = 100, street_length = 1000,
#               die_name = 'chip99', text_size = 300, text_location = 'SW',  layer = 0,
#               draw_bbox = True,  bbox_layer = 99)
# quickplot(D)





#==============================================================================
#
# Waveguide curves
#
#==============================================================================


def racetrack_gradual(width = 0.3, R = 5, N = 3, layer = 0):
    curve_fun = lambda t: _racetrack_gradual_parametric(t, R = 5, N = 3)
    route_path = gdspy.Path(width = width, initial_point = [0,0])
    route_path.parametric(curve_fun, number_of_evaluations=99,\
            max_points=4000,  final_distance=None, layer=layer)
    D = Device('racetrack')
    D.add(route_path)
    return D


def _racetrack_gradual_parametric(t, R, N):
    """ Takes in a parametric value ``t`` on (0,1), returns the x,y coordinates
    of a racetrack bent according to 20090810_EOS4_modulator_designs_excerptForJasonGradualBends.ppt """
    x0 = R/2**(1/N)
    Rmin = 2**(0.5-1/N)/(N-1)*R
    R0 = R-(x0-Rmin/sqrt(2))
    t = np.array(t)
    x,y = np.zeros(t.shape), np.zeros(t.shape)

    # Doing the math
    x = cos(t*pi/2)*R0 # t (0-1) while x (0 to R0)
    ii =  (Rmin/sqrt(2) < x) & (x <= R0)
    jj =  (0 < x) & (x <= Rmin/sqrt(2))
    y[ii] = (R**N - (x[ii]+(x0-Rmin/sqrt(2)))**N)**(1/N)
    y[jj] = (x0-Rmin/sqrt(2))+sqrt(Rmin**2-x[jj]**2)
    return x,y



#==============================================================================
# Example code
#==============================================================================

# D = racetrack_gradual(width = 0.3, R = 5, N = 3)
# quickplot(D)



# t = np.linspace(0,1)
# x,y = _racetrack_gradual_parametric(t, R = 5, N = 3)
# plt.plot(x,y)
# plt.axis('equal')







#==============================================================================
#
# Fill
#
#==============================================================================


def _rasterize_polygons(polygons, bounds = [[-100, -100], [100, 100]], dx = 1, dy = 1):
    try:
        from skimage import draw
    except:
        raise ImportError("""
            The fill function requires the module "scikit-image"
            to operate.  Please retry after installing scikit-image:

            $ pip install --upgrade scikit-image """)


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


def _expand_raster(raster, distance = (4,2)):
    try:
        from skimage import draw, morphology
    except:
        raise ImportError("""
            The fill function requires the module "scikit-image"
            to operate.  Please retry after installing scikit-image:

            $ pip install --upgrade scikit-image """)
    if distance[0] <= 0.5 and distance[1] <= 0.5: return raster

    num_pixels = np.array(np.ceil(distance), dtype = int)
    neighborhood = np.zeros((num_pixels[1]*2+1, num_pixels[0]*2+1), dtype=np.bool)
    rr, cc = draw.ellipse(num_pixels[1], num_pixels[0], distance[1]+0.5, distance[0]+0.5)
    neighborhood[rr, cc] = 1

    return morphology.binary_dilation(image = raster, selem=neighborhood)



def _fill_cell_rectangle(size = (20,20), layers = (0,1,3),
                         densities = (0.5, 0.25, 0.7), inverted = (False, False, False)):
    D = Device('fillcell')
    for layer, density, inv in zip(layers, densities, inverted):
        rectangle_size = np.array(size)*np.sqrt(density)
#        r = D.add_ref(rectangle(size = rectangle_size, layer = layer))
        R = rectangle(size = rectangle_size, layer = layer)
        R.center = (0,0)
        if inv is True:
            A = rectangle(size = size)
            A.center = (0,0)
            A = A.get_polygons()
            B = R.get_polygons()
            p = gdspy.fast_boolean(A, B, operation = 'not')
            D.add_polygon(p, layer = layer)
        else:
            D.add_ref(R)
    return D

def _loop_over(var):
    # Checks if a variable is in the form of an iterable (list/tuple)
    # and if not, returns it as a list.  Useful for allowing argument
    # inputs to be either lists (e.g. [1,3,4]) or single-valued (e.g. 3)
    if hasattr(var,"__iter__"):
        return var
    else:
        return [var]

def fill_rectangle(D, fill_size = (40,10), avoid_layers = 'all', include_layers = None,
                    margin = 100, fill_layers = (0,1,3),
                   fill_densities = (0.5, 0.25, 0.7), fill_inverted = None, bbox = None):

    # Create the fill cell.  If fill_inverted is not specified, assume all False
    fill_layers = _loop_over(fill_layers)
    fill_densities = _loop_over(fill_densities)
    if fill_inverted is None: fill_inverted = [False]*len(fill_layers)
    fill_inverted = _loop_over(fill_inverted)
    if len(fill_layers) != len(fill_densities):
        raise ValueError("[PHIDL] phidl.geometry.fill_rectangle() `fill_layers` and" +
        " `fill_densities` parameters must be lists of the same length")
    if len(fill_layers) != len(fill_inverted):
        raise ValueError("[PHIDL] phidl.geometry.fill_rectangle() `fill_layers` and" +
        " `fill_inverted` parameters must be lists of the same length")

    fill_cell = _fill_cell_rectangle(size = fill_size, layers = fill_layers,
                                     densities = fill_densities, inverted = fill_inverted)
    F = Device(name = 'fill_pattern')

    if avoid_layers == 'all':
        exclude_polys = D.get_polygons(by_spec=False, depth=None)
    else:
        avoid_layers = [_parse_layer(l) for l in _loop_over(avoid_layers)]
        exclude_polys = D.get_polygons(by_spec=True, depth=None)
        exclude_polys = {key:exclude_polys[key] for key in exclude_polys if key in avoid_layers}
        exclude_polys = itertools.chain.from_iterable(exclude_polys.values())

    if include_layers is None:
        include_polys = []
    else:
        include_layers = [_parse_layer(l) for l in _loop_over(include_layers)]
        include_polys = D.get_polygons(by_spec=True, depth=None)
        include_polys = {key:include_polys[key] for key in include_polys if key in include_layers}
        include_polys = itertools.chain.from_iterable(include_polys.values())



    if bbox is None:  bbox = D.bbox

    raster = _rasterize_polygons(polygons = exclude_polys, bounds = bbox, dx = fill_size[0], dy = fill_size[1])
    raster = raster & ~_rasterize_polygons(polygons = include_polys, bounds = bbox, dx = fill_size[0], dy = fill_size[1])
    raster = _expand_raster(raster, distance = margin/np.array(fill_size))

    for i in range(np.size(raster,0)):
        sub_rasters = [list(g) for k, g in itertools.groupby(raster[i])]
        j = 0
        for s in sub_rasters:
            if s[0] == 0:
                x,y = _raster_index_to_coords(i, j, bbox, fill_size[0], fill_size[1])
                F.add(gdspy.CellArray(ref_cell = fill_cell, columns = len(s), rows = 1, spacing = fill_size, origin = (x, y)))
            j += len(s)

    return F





#==============================================================================
#
# Photonics
#
#==============================================================================


def polygon_ports(xpts=[-1,-1, 0, 0],
            ypts = [0, 1, 1, 0],
            layer = 0):
    # returns a polygon with ports on all edges
    P = Device('polygon')
    P.add_polygon([xpts, ypts], layer = layer)
    n = len(xpts)
    xpts.append(xpts[0])
    ypts.append(ypts[0])
    #determine if clockwise or counterclockwise
    cc = 0
    for i in range(0,n):
        cc += ((xpts[i+1]-xpts[i])*(ypts[i+1]+ypts[i]))

    for i in range(0,n):
        midpoint_n = [(xpts[i+1]+xpts[i])/2, (ypts[i+1]+ypts[i])/2]
        orientation_n = np.arctan2(np.sign(cc)*(xpts[i+1]-xpts[i]),np.sign(cc)*(ypts[i]-ypts[i+1]))*180/np.pi
        width_n = np.abs(np.sqrt((xpts[i+1]-xpts[i])**2+(ypts[i+1]-ypts[i])**2))
        P.add_port(name = str(i+1), midpoint = midpoint_n, width = width_n, orientation = orientation_n)

    return P

#==============================================================================
# Example code
#==============================================================================

# P = polygon(xpts=[-1,-3, 0, 0], ypts = [0, 1, 2, 0], layer = 3)
# quickplot(P)


@device_lru_cache
def grating(num_periods = 20, period = 0.75, fill_factor = 0.5, width_grating = 5, length_taper = 10, width = 0.4, partial_etch = False):
    #returns a fiber grating
    G = Device('grating')

# make the deep etched grating
    if partial_etch is False:
        # make the grating teeth
        for i in range(num_periods):
            cgrating = G.add_ref(compass(size=[period*fill_factor,width_grating], layer = 0))
            cgrating.x+=i*period

        # make the taper
        tgrating = G.add_ref(taper(length = length_taper, width1 = width_grating, width2 = width, port = None, layer = 0))
        tgrating.xmin = cgrating.xmax
        # define the port of the grating
        p = G.add_port(port = tgrating.ports[2], name = 1)
# make a partially etched grating
    if partial_etch is True:
        # hard coded overlap
            partetch_overhang = 5
            # make the etched areas (opposite to teeth)
            for i in range(num_periods):
                cgrating = G.add_ref(compass(size=[period*(1-fill_factor),width_grating+partetch_overhang*2]), layer = 1)
                cgrating.x+=i*period
                        # define the port of the grating
            p = G.add_port(port = cgrating.ports['E'], name = 1)
            p.midpoint=p.midpoint+np.array([(1-fill_factor)*period,0])

        #draw the deep etched square around the grating
            deepbox = G.add_ref(compass(size=[num_periods*period, width_grating]), layer=0)
    return G

#==============================================================================
# Example code
#==============================================================================

# G = grating(num_periods = 20, period = 0.75, fill_factor = 0.5, width_grating = 20, length_taper = 10, width = 0.4, partial_etch = False)
# quickplot(G)




#==============================================================================
#
# Test Structures
#
#==============================================================================


# Via Route ----------------------------------------
def _via_iterable(via_spacing, wire_width, wiring1_layer, wiring2_layer, via_layer, via_width):
    VI = Device('test_via_iter')
    wire1 = VI.add_ref(compass(size=(via_spacing, wire_width), layer=wiring1_layer))
    wire2 = VI.add_ref(compass(size=(via_spacing, wire_width), layer=wiring2_layer))
    via1 = VI.add_ref(compass(size=(via_width, via_width), layer=via_layer))
    via2 = VI.add_ref(compass(size=(via_width, via_width), layer=via_layer))
    wire1.connect(port='E', destination = wire2.ports['W'], overlap=wire_width)
    via1.connect(port='W', destination = wire1.ports['E'], overlap = (wire_width + via_width)/2)
    via2.connect(port='W', destination = wire2.ports['E'], overlap = (wire_width + via_width)/2)
    VI.add_port(name='W', port = wire1.ports['W'])
    VI.add_port(name='E', port = wire2.ports['E'])
    VI.add_port(name='S', midpoint = [(1*wire_width)+ wire_width/2,-wire_width/2], width = wire_width, orientation = -90)
    VI.add_port(name='N', midpoint = [(1*wire_width)+ wire_width/2,wire_width/2], width = wire_width, orientation = 90)

    return VI

def test_via(num_vias = 100, wire_width = 10, via_width = 15, via_spacing = 40, pad_size = (300,300), min_pad_spacing = 0,
                             pad_layer = 0, wiring1_layer = 1, wiring2_layer = 2, via_layer = 3):
    """
    Usage:
        Call via_route_test_structure() by indicating the number of vias you want drawn. You can also change the other parameters however
        if you do not specifiy a value for a parameter it will just use the default value
        Ex::

            via_route_test_structure(num_vias=54)

        - or -::

            via_route_test_structure(num_vias=12, pad_size=(100,100),wire_width=8)

        total requested vias (num_vias) -> this needs to be even
        pad size (pad_size) -> given in a pair (width, height)
        wire_width -> how wide each wire should be
        pad_layer -> GDS layer number of the pads
        wiring1_layer -> GDS layer number of the top wiring
        wiring2_layer -> GDS layer number of the bottom wiring
        via_layer -> GDS layer number of the vias
        ex: via_route(54, min_pad_spacing=300)
    """

    VR = Device('test_via')
    pad1 = VR.add_ref(rectangle(size=pad_size, layer=pad_layer))
    pad1_overlay = VR.add_ref(rectangle(size=pad_size, layer=wiring1_layer))
    pad2 = VR.add_ref(rectangle(size=pad_size, layer=pad_layer))
    pad2_overlay = VR.add_ref(rectangle(size=pad_size, layer=wiring1_layer))
    nub = VR.add_ref(compass(size=(3*wire_width,wire_width),layer=pad_layer))
    nub_overlay = VR.add_ref(compass(size=(3*wire_width,wire_width),layer=wiring1_layer))
    head = VR.add_ref(compass(size=(wire_width,wire_width),layer=pad_layer))
    head_overlay = VR.add_ref(compass(size=(wire_width,wire_width),layer=wiring1_layer))
    nub.ymax = pad1.ymax-5
    nub.xmin = pad1.xmax
    nub_overlay.ymax = pad1.ymax-5
    nub_overlay.xmin = pad1.xmax
    head.connect(port = "W", destination = nub.ports["E"])
    head_overlay.connect(port = "W", destination = nub_overlay.ports["E"])
    pad1_overlay.xmin = pad1.xmin
    pad1_overlay.ymin = pad1.ymin

    old_port = head.ports['S']
    count = 0
    width_via_iter = 2*via_spacing - 2*wire_width

    pad2.xmin = pad1.xmax + min_pad_spacing
    up = False
    down = True
    edge = True
    current_width = 3*wire_width + wire_width #width of nub and 1 overlap
    obj_old = head
    obj = head
    via_iterable = _via_iterable(via_spacing, wire_width, wiring1_layer, wiring2_layer, via_layer, via_width)
    while( (count+2) <= num_vias):
        obj = VR.add_ref(via_iterable)
        obj.connect(port = 'W', destination = old_port, overlap = wire_width)
        old_port = obj.ports['E']
        edge = False
        if(obj.ymax > pad1.ymax):
            obj.connect(port = 'W', destination = obj_old.ports['S'], overlap = wire_width)
            old_port = obj.ports['S']
            current_width += width_via_iter
            down = True
            up = False
            edge = True

        elif(obj.ymin < pad1.ymin):
            obj.connect(port = 'W', destination = obj_old.ports['N'], overlap = wire_width)
            old_port = obj.ports['N']
            current_width += width_via_iter
            up = True
            down = False
            edge = True
        count = count + 2
        obj_old = obj

    if(current_width < min_pad_spacing and (min_pad_spacing - current_width) > 3*wire_width):
        tail = VR.add_ref(compass(size=(min_pad_spacing-current_width+wire_width,wire_width),layer=wiring1_layer))
        tail_overlay = VR.add_ref(compass(size=(min_pad_spacing-current_width+wire_width,wire_width),layer=pad_layer))
    else:
        tail = VR.add_ref(compass(size=(3*wire_width,wire_width),layer=wiring1_layer))
        tail_overlay = VR.add_ref(compass(size=(3*wire_width,wire_width),layer=wiring1_layer))

    if(up == True and edge != True):
        tail.connect(port = 'W', destination = obj.ports['S'], overlap = wire_width)
        tail_overlay.connect(port = 'W', destination = obj.ports['S'], overlap = wire_width)
    elif(down == True and edge != True):
        tail.connect(port = 'W', destination = obj.ports['N'], overlap = wire_width)
        tail_overlay.connect(port = 'W', destination = obj.ports['N'], overlap = wire_width)
    else:
        tail.connect(port = 'W', destination = obj.ports['E'], overlap = wire_width)
        tail_overlay.connect(port = 'W', destination = obj.ports['E'], overlap = wire_width)


    pad2.xmin = tail.xmax
    pad2_overlay.xmin = pad2.xmin
    pad2_overlay.ymin = pad2.ymin

    return VR





def test_comb(pad_size = (200,200), wire_width = 1, wire_gap = 3,
              comb_layer = 0, overlap_zigzag_layer = 1,
              comb_pad_layer = None, comb_gnd_layer = None, overlap_pad_layer = None):
    """
    Usage:

    Call comb_insulation_test_structure() with any of the
    parameters shown below which you'd like to change. You
    only need to supply the parameters which you intend on
    changing You can alternatively call it with no parameters
    and it will take all the default alues shown below.
    Ex::

        comb_insulation_test_structure(pad_size=(175,175), wire_width=2, wire_gap=5)

    - or -::

        comb_insulation_test_structure()
    """
    CI = Device("test_comb")

    if comb_pad_layer is None:  comb_pad_layer = comb_layer
    if comb_gnd_layer is None:  comb_gnd_layer = comb_layer
    if overlap_pad_layer is None:  overlap_pad_layer = overlap_zigzag_layer
    wire_spacing = wire_width + wire_gap*2



    #%% pad overlays
    overlay_padb = CI.add_ref(rectangle(size=(pad_size[0]*9/10,pad_size[1]*9/10), layer=overlap_pad_layer))
    overlay_padl = CI.add_ref(rectangle(size=(pad_size[0]*9/10,pad_size[1]*9/10), layer=comb_pad_layer ) )
    overlay_padt = CI.add_ref(rectangle(size=(pad_size[0]*9/10,pad_size[1]*9/10), layer=comb_pad_layer ) )
    overlay_padr = CI.add_ref(rectangle(size=(pad_size[0]*9/10,pad_size[1]*9/10), layer=comb_gnd_layer))

    overlay_padl.xmin = 0
    overlay_padl.ymin = 0
    overlay_padb.ymax = 0
    overlay_padb.xmin = overlay_padl.xmax + pad_size[1]/5
    overlay_padr.ymin = overlay_padl.ymin
    overlay_padr.xmin = overlay_padb.xmax + pad_size[1]/5
    overlay_padt.xmin = overlay_padl.xmax + pad_size[1]/5
    overlay_padt.ymin = overlay_padl.ymax

    #%% pads
    padl = CI.add_ref(rectangle(size=pad_size, layer=comb_layer))
    padt = CI.add_ref(rectangle(size=pad_size, layer=comb_layer))
    padr = CI.add_ref(rectangle(size=pad_size, layer=comb_layer))
    padb = CI.add_ref(rectangle(size=pad_size, layer=overlap_zigzag_layer))
    padl_nub = CI.add_ref(rectangle(size=(pad_size[0]/4,pad_size[1]/2), layer=comb_layer))
    padr_nub = CI.add_ref(rectangle(size=(pad_size[0]/4,pad_size[1]/2), layer=comb_layer))

    padl.xmin = overlay_padl.xmin
    padl.center = [padl.center[0],overlay_padl.center[1]]
    padt.ymax = overlay_padt.ymax
    padt.center = [overlay_padt.center[0],padt.center[1]]
    padr.xmax = overlay_padr.xmax
    padr.center = [padr.center[0],overlay_padr.center[1]]
    padb.ymin = overlay_padb.ymin
    padb.center = [overlay_padb.center[0],padb.center[1]]
    padl_nub.xmin = padl.xmax
    padl_nub.center = [padl_nub.center[0],padl.center[1]]
    padr_nub.xmax = padr.xmin
    padr_nub.center = [padr_nub.center[0],padr.center[1]]

    #%% connected zig

    head = CI.add_ref(compass(size=(pad_size[0]/12, wire_width), layer=comb_layer))
    head.xmin = padl_nub.xmax
    head.ymax = padl_nub.ymax
    connector = CI.add_ref(compass(size=(wire_width, wire_width), layer=comb_layer))
    connector.connect(port = 'W', destination=head.ports['E'])
    old_port = connector.ports['S']
    top = True
    obj = connector
    while(obj.xmax + pad_size[0]/12 < padr_nub.xmin):
        #long zig zag rectangle
        obj = CI.add_ref(compass(size=(pad_size[1]/2 - 2*wire_width, wire_width), layer=comb_layer))
        obj.connect(port = 'W', destination=old_port)
        old_port = obj.ports['E']
        if(top):
            #zig zag edge rectangle
            obj = CI.add_ref(compass(size=(wire_width, wire_width), layer=comb_layer))
            obj.connect(port = 'N', destination=old_port)
            top = False
        else:
            #zig zag edge rectangle
            obj = CI.add_ref(compass(size=(wire_width, wire_width), layer=comb_layer))
            obj.connect(port = 'S', destination=old_port)
            top = True
            # comb rectange
            comb = CI.add_ref(rectangle(size=((padt.ymin-head.ymax)+pad_size[1]/2  - (wire_spacing + wire_width)/2, wire_width), layer=comb_layer))
            comb.rotate(90)
            comb.ymax = padt.ymin
            comb.xmax = obj.xmax - (wire_spacing+wire_width)/2
        old_port = obj.ports['E']
        obj = CI.add_ref(compass(size=(wire_spacing, wire_width), layer=comb_layer))
        obj.connect(port = 'W', destination=old_port)
        old_port = obj.ports['E']
        obj = CI.add_ref(compass(size=(wire_width, wire_width), layer=comb_layer))
        obj.connect(port = 'W', destination=old_port)
        if(top):
            old_port = obj.ports['S']
        else:
            old_port = obj.ports['N']
    old_port = obj.ports['E']
    if(padr_nub.xmin-obj.xmax > 0):
        tail = CI.add_ref(compass(size=(padr_nub.xmin-obj.xmax, wire_width), layer=comb_layer))
    else:
        tail = CI.add_ref(compass(size=(wire_width, wire_width), layer=comb_layer))
    tail.connect(port = 'W', destination=old_port)

    #%% disconnected zig

    dhead = CI.add_ref(compass(size=(padr_nub.ymin -padb.ymax - wire_width, wire_width), layer=overlap_zigzag_layer))
    dhead.rotate(90)
    dhead.ymin = padb.ymax
    dhead.xmax = tail.xmin-(wire_spacing+wire_width)/2
    connector = CI.add_ref(compass(size=(wire_width, wire_width), layer=overlap_zigzag_layer))
    connector.connect(port = 'S', destination=dhead.ports['E'])
    old_port = connector.ports['N']
    right = True
    obj = connector
    while(obj.ymax + wire_spacing + wire_width < head.ymax):
        obj = CI.add_ref(compass(size=(wire_spacing, wire_width), layer=overlap_zigzag_layer))
        obj.connect(port = 'W', destination=old_port)
        old_port = obj.ports['E']
        if(right):
            obj = CI.add_ref(compass(size=(wire_width, wire_width), layer=overlap_zigzag_layer))
            obj.connect(port = 'W', destination=old_port)
            right = False
        else:
            obj = CI.add_ref(compass(size=(wire_width, wire_width), layer=overlap_zigzag_layer))
            obj.connect(port = 'E', destination=old_port)
            right = True
        old_port = obj.ports['N']
        obj = CI.add_ref(compass(size=(dhead.xmin-(head.xmax+head.xmin+wire_width)/2, wire_width), layer=overlap_zigzag_layer))
        obj.connect(port = 'E', destination=old_port)
        old_port = obj.ports['W']
        obj = CI.add_ref(compass(size=(wire_width, wire_width), layer=overlap_zigzag_layer))
        obj.connect(port = 'S', destination=old_port)
        if(right):
            old_port = obj.ports['W']
        else:
            old_port = obj.ports['E']

    return CI




#This is a helper function to make the Ic step wire structure
def _test_ic_wire_step(thick_width = 10, thin_width = 1, wire_layer = 2):
    WS4 = Device('test_ic_step')
    wire_stepa = WS4.add_ref(optimal_step(thick_width/2, thin_width/2, layer=wire_layer))
    wire_stepb = WS4.add_ref(optimal_step(thin_width/2, thick_width/2, layer=wire_layer))
    wire_stepc = WS4.add_ref(optimal_step(thick_width/2, thin_width/2, layer=wire_layer))
    wire_stepd = WS4.add_ref(optimal_step(thin_width/2, thick_width/2, layer=wire_layer))
    wire_stepb.rotate(180)
    wire_stepb.xmin = wire_stepa.xmin
    wire_stepc.rotate(180)
    wire_stepc.xmin = wire_stepa.xmax
    wire_stepd.xmin = wire_stepc.xmin
    return WS4


def test_ic(wire_widths = [0.25, 0.5,1,2,4], wire_widths_wide = [0.75, 1.5, 3, 4, 6], pad_size = (200,200), pad_gap=75,
            wire_layer = 0, pad_layer = 1, gnd_layer = None):
    """
    Usage:

    Call ic_test_structure() with either a list of widths for the thickest part of each wire to test and a list for the
    thinnest parts of each wire. Alternatively, specify a list of widths for the thinnest part of each wire and ignore the
    wire_widths parameter. Instead you should specify the width_growth_factor which indicates by what factor the thick
    part of the wire will be larger than the thin part.
    Ex::

        ic_test_structure(wire_widths = [5,10,10,10,10], thin_width=[0.5,1,2,3,4])

    - or -::

        ic_test_structure(width_growth_factor = 5, thin_width=[0.5,1,2,3,4])
    """
    ICS = Device('test_ic')

    if gnd_layer is None: gnd_layer = pad_layer
    translation = 0
    padb = ICS.add_ref(rectangle(size=(np.size(wire_widths) * (pad_size[0]*6/5), pad_size[1]), layer=wire_layer))
    padb_overlay = ICS.add_ref(rectangle(size=((np.size(wire_widths) * (pad_size[0]*6/5))*9/10, pad_size[1]*9/10), layer=gnd_layer))
    padb_overlay.center = padb.center
    padb_overlay.ymin = padb.ymin
    for i, x in enumerate(wire_widths_wide):
        padt = ICS.add_ref(rectangle(pad_size, wire_layer))
        padt.xmin = padb.xmin + translation
        padt.ymin = padb.ymax + pad_gap
        padt_overlay = ICS.add_ref(rectangle(size=(pad_size[0]*9/10, pad_size[1]*9/10), layer=pad_layer))
        padt_overlay.center = padt.center
        padt_overlay.ymax = padt.ymax
        difference = padt.ymin-padb.ymax
        wire_step = ICS.add_ref(_test_ic_wire_step(wire_widths_wide[i], wire_widths[i], wire_layer=wire_layer))
        wire_step.rotate(90)
        wire_step.center = (padt.center[0], padb.ymax + difference/2)
        translation = translation + pad_size[0]*12/10
        conn_wire_top = ICS.add_ref(rectangle(size=(wire_widths_wide[i], padt.ymin-wire_step.ymax), layer=wire_layer))
        conn_wire_bottom = ICS.add_ref(rectangle(size=(wire_widths_wide[i], wire_step.ymin-padb.ymax), layer=wire_layer))
        conn_wire_top.ymax = padt.ymin
        conn_wire_top.xmin = wire_step.xmin
        conn_wire_bottom.ymin = padb.ymax
        conn_wire_bottom.xmin = wire_step.xmin
    return ICS

def test_res(pad_size = [50,50],
                     num_squares = 1000,
                     width = 1,
                     res_layer = 0,
                     pad_layer = None,
                     gnd_layer = None):

    """ Creates an efficient resonator structure for a wafer layout.

    Keyword arguments:
    pad_size    -- Size of the two matched impedance pads (microns)
    num_squares -- Number of squares comprising the resonator wire
    width       -- The width of the squares (microns)
    """

    x = pad_size[0]
    z = pad_size[1]

    # Checking validity of input
    if x <= 0 or z <= 0:
        raise ValueError('Pad must have positive, real dimensions')
    elif width > z:
        raise ValueError('Width of cell cannot be greater than height of pad')
    elif num_squares <= 0:
        raise ValueError('Number of squares must be a positive real number')
    elif width <= 0:
        raise ValueError('Width of cell must be a positive real number')

    # Performing preliminary calculations
    num_rows = int(np.floor(z / (2 * width)))
    if num_rows % 2 == 0:
        num_rows -= 1
    num_columns = num_rows - 1
    squares_in_row = (num_squares - num_columns - 2) / num_rows

    # Compensating for weird edge cases
    if squares_in_row < 1:
        num_rows = round(num_rows / 2) - 2
        squares_in_row = 1
    if width * 2 > z:
        num_rows = 1
        squares_in_row = num_squares - 2

    length_row = squares_in_row * width

    # Creating row/column corner combination structure
    T = Device()
    Row = rectangle(size = (length_row, width), layer = res_layer)
    Col = rectangle(size = (width, width), layer = res_layer)

    row = T.add_ref(Row)
    col = T.add_ref(Col)
    col.move([length_row - width, -width])

    # Creating entire waveguide net
    N = Device('net')
    n = 1
    for i in range(num_rows):
        if i != num_rows - 1:
            d = N.add_ref(T)
        else:
            d = N.add_ref(Row)
        if n % 2 == 0:
            d.reflect(p1 = (d.x, d.ymax), p2 = (d.x, d.ymin))
        d.movey(-(n - 1) * T.ysize)
        n += 1
    d = N.add_ref(Col).movex(-width)
    d = N.add_ref(Col).move([length_row, -(n - 2) * T.ysize])

    # Creating pads
    P = Device('pads')
    Pad1 = rectangle(size = (x,z), layer = pad_layer)
    Pad2 = rectangle(size = (x + 5, z), layer = pad_layer)
    Gnd1 = offset(Pad1, distance = -5, layer = gnd_layer)
    Gnd2 = offset(Pad2, distance = -5, layer = gnd_layer)
    pad1 = P.add_ref(Pad1).movex(-x - width)
    pad2 = P.add_ref(Pad1).movex(length_row + width)
    gnd1 = P.add_ref(Gnd1).center = pad1.center
    gnd2 = P.add_ref(Gnd2)
    nets = P.add_ref(N).y = pad1.y
    gnd2.center = pad2.center
    gnd2.movex(2.5)

    return P


#==============================================================================
#
# Optimal current-crowding superconducting structures
#
#==============================================================================

@device_lru_cache
def optimal_hairpin(width = 0.2, pitch = 0.6, length = 10,
    turn_ratio = 4, num_pts = 50, layer = 0):

    #==========================================================================
    #  Create the basic geometry
    #==========================================================================
    a = (pitch + width)/2
    y = -(pitch - width)/2
    x = -pitch
    dl = width/(num_pts*2)
    n = 0

    # Get points of ideal curve from conformal mapping
    # TODO This is an inefficient way of finding points that you need
    xpts = [x]; ypts = [y]
    while (y < 0) & (n<1e6):
        s = x + 1j*y
        w = np.sqrt(1 - np.exp(pi*s/a))
        wx = np.real(w); wy = np.imag(w)
        wx = wx/np.sqrt(wx**2+wy**2); wy = wy/np.sqrt(wx**2+wy**2)
        x = x + wx*dl; y = y + wy*dl
        xpts.append(x); ypts.append(y)
        n = n+1
    ypts[-1] = 0 # Set last point be on the x=0 axis for sake of cleanliness
    ds_factor = int(len(xpts)/num_pts) # Downsample the total number of points
    xpts = xpts[::-ds_factor]; xpts = xpts[::-1]    # This looks confusing, but it's just flipping the arrays around
    ypts = ypts[::-ds_factor]; ypts = ypts[::-1]    # so the last point is guaranteed to be included when downsampled

    # Add points for the rest of meander
    xpts.append(xpts[-1] + turn_ratio*width); ypts.append(0)
    xpts.append(xpts[-1]); ypts.append(-a)
    xpts.append(xpts[0]); ypts.append(-a)
    xpts.append(max(xpts)-length); ypts.append(-a)
    xpts.append(xpts[-1]); ypts.append(-a + width)
    xpts.append(xpts[0]); ypts.append(ypts[0])

    xpts = np.array(xpts)
    ypts = np.array(ypts)

    #==========================================================================
    #  Create a blank device, add the geometry, and define the ports
    #==========================================================================
    D = Device(name = 'hairpin')
    D.add_polygon([xpts,ypts], layer = layer)
    D.add_polygon([xpts,-ypts], layer = layer)

    xports = min(xpts)
    yports = -a + width/2
    D.add_port(name = 1, midpoint = [xports,-yports], width = width, orientation = 180)
    D.add_port(name = 2, midpoint = [xports,yports], width = width, orientation = 180)

    return D


# TODO Include parameter which specifies "half" (one edge flat) vs "full" (both edges curved)
@device_lru_cache
def optimal_step(start_width = 10, end_width = 22, num_pts = 50, width_tol = 1e-3,
                 anticrowding_factor = 1.2, symmetric = False, layer = 0):

    #==========================================================================
    #  Create the basic geometry
    #==========================================================================
    def step_points(eta, W, a):
        # Returns points from a unit semicircle in the w (= u + iv) plane to
        # the optimal curve in the zeta (= x + iy) plane which transitions
        # a wire from a width of 'W' to a width of 'a'
        # eta takes value 0 to pi

        W = np.complex(W)
        a = np.complex(a)

        gamma = (a*a + W*W)/(a*a - W*W)

        w = np.exp(1j*eta)

        zeta = 4*1j/pi*(W*np.arctan(np.sqrt((w-gamma)/(gamma+1))) \
                           + a*np.arctan(np.sqrt((gamma-1)/(w-gamma))))

        x = np.real(zeta)
        y = np.imag(zeta)
        return x,y


    def invert_step_point(x_desired = -10, y_desired = None, W = 1, a = 2):
        # Finds the eta associated with the value x_desired along the optimal curve
        def fh(eta):
            guessed_x, guessed_y = step_points(eta, W = W, a = a)
            if y_desired is None:   return (guessed_x-x_desired)**2 # The error
            else:                   return (guessed_y-y_desired)**2

        try:
            from scipy.optimize import fminbound
        except:
            raise ImportError(""" [PHIDL] To run the optimal-curve geometry functions you need scipy,
              please install it with `pip install scipy` """)
        found_eta = fminbound(fh, x1 = 0, x2 = pi, args=())
        return step_points(found_eta, W = W, a = a)

    if start_width > end_width:
        reverse = True
        start_width, end_width = end_width, start_width
    else:
        reverse = False

    if start_width == end_width: # Just return a square
        ypts = [0, start_width, start_width,           0]
        xpts = [0,           0, start_width, start_width]
    else:
        xmin,ymin = invert_step_point(y_desired = start_width*(1+width_tol), W = start_width, a = end_width)
        xmax,ymax = invert_step_point(y_desired = end_width*(1-width_tol), W = start_width, a = end_width)

        xpts = np.linspace(xmin, xmax, num_pts).tolist()
        ypts = []
        for x in xpts:
            x,y = invert_step_point(x_desired = x, W = start_width, a = end_width)
            ypts.append(y)

        ypts[-1] = end_width
        ypts[0] =  start_width
        if symmetric == False:
            xpts.append(xpts[-1])
            ypts.append(0)
            xpts.append(xpts[0])
            ypts.append(0)
        else:
            xpts += [x for x in xpts[::-1]]
            ypts += [-y for y in ypts[::-1]]
            xpts = [x/2 for x in xpts]
            ypts = [y/2 for y in ypts]

        # anticrowding_factor stretches the wire out; a stretched wire is a gentler
        # transition, so there's less chance of current crowding if the fabrication
        # isn't perfect but as a result, the wire isn't as short as it could be
        xpts = (np.array(xpts)*anticrowding_factor).tolist()

        if reverse is True:
            xpts = (-np.array(xpts)).tolist()
            start_width, end_width = end_width, start_width

    #==========================================================================
    #  Create a blank device, add the geometry, and define the ports
    #==========================================================================
    D = Device(name = 'step')
    D.add_polygon([xpts,ypts], layer = layer)

    if symmetric == False:
        D.add_port(name = 1, midpoint = [min(xpts),start_width/2], width = start_width, orientation = 180)
        D.add_port(name = 2, midpoint = [max(xpts),end_width/2], width = end_width, orientation = 0)
    if symmetric == True:
        D.add_port(name = 1, midpoint = [min(xpts),0], width = start_width, orientation = 180)
        D.add_port(name = 2, midpoint = [max(xpts),0], width = end_width, orientation = 0)

    return D


def optimal_90deg(width = 100.0, num_pts = 15, length_adjust = 1, layer = 0):
    D = Device()

    # Get points of ideal curve
    a = 2*width
    v = np.logspace(-length_adjust,length_adjust,num_pts)
    xi = a/2.0*((1+2/np.pi*np.arcsinh(1/v)) + 1j*(1+2/np.pi*np.arcsinh(v)))
    xpts = list(np.real(xi)); ypts = list(np.imag(xi))

    # Add points for the rest of curve
    d = 2*xpts[0] # Farthest point out * 2, rounded to nearest 100
    xpts.append(width); ypts.append(d)
    xpts.append(0); ypts.append(d)
    xpts.append(0); ypts.append(0)
    xpts.append(d); ypts.append(0)
    xpts.append(d); ypts.append(width)
    xpts.append(xpts[0]); ypts.append(ypts[0])

    D.add_polygon([xpts, ypts], layer = layer)

    D.add_port(name = 1, midpoint = [a/4,d], width = a/2, orientation = 90)
    D.add_port(name = 2, midpoint = [d,a/4], width = a/2, orientation = 0)
    return D



#==============================================================================
# Example code
#==============================================================================

#hairpin = optimal_hairpin(width = 1, pitch = 3, length = 30, num_pts = 20)
#quickplot(hairpin)


#step = optimal_step(start_width = 5, end_width = 1, num_pts = 80, width_tol = 1e-3)
#quickplot(step)


#turn = optimal_90deg(width = 90, length_adjust = 1)
#quickplot(turn)





#==============================================================================
#
# Superconducting devices
#
#==============================================================================


@device_lru_cache
def snspd(wire_width = 0.2, wire_pitch = 0.6, size = (10,8),
        num_squares = None, turn_ratio = 4,
        terminals_same_side = False, layer = 0):
    # Convenience tests to auto-shape the size based
    # on the number of squares
    if num_squares is not None and ((size is None) or ((size[0] is None) and (size[1]) is None)):
        xy = np.sqrt(num_squares*wire_pitch*wire_width)
        size = [xy,xy]
        num_squares = None
    if ([size[0], size[1], num_squares].count(None) != 1):
        raise ValueError('[PHIDL] snspd() requires that exactly ONE value of' +
                         ' the arguments ``num_squares`` and ``size`` be None'+
                         ' to prevent overconstraining, for example:\n' +
                         '>>> snspd(size = (3, None), num_squares = 2000)')
    if size[0] is None:
        ysize = size[1]
        xsize = num_squares*wire_pitch*wire_width/ysize
    elif size[1] is None:
        xsize = size[0]
        ysize = num_squares*wire_pitch*wire_width/xsize
    else:
        xsize = size[0]
        ysize = size[1]

    num_meanders = int(np.ceil(ysize/wire_pitch))

    D = Device(name = 'snspd')
    hairpin = optimal_hairpin(width = wire_width, pitch = wire_pitch,
        turn_ratio = turn_ratio, length = xsize/2, num_pts = 20, layer = layer)


    if (terminals_same_side is False) and ((num_meanders % 2) == 0):
        num_meanders += 1
    elif (terminals_same_side is True) and ((num_meanders % 2) == 1):
        num_meanders += 1

    start_nw = D.add_ref(compass(size = [xsize/2 ,wire_width], layer = layer))

    hp_prev = D.add_ref(hairpin)
    hp_prev.connect(1, start_nw.ports['E'])
    alternate = True
    for n in range(2,num_meanders):
        hp = D.add_ref(hairpin)
        if alternate:
            hp.connect(2, hp_prev.ports[2])
            last_port = hp.ports[1]
        else:
            hp.connect(1, hp_prev.ports[1])
            last_port = hp.ports[2]
        hp_prev = hp
        alternate = not alternate

    finish_se = D.add_ref(compass(size = [xsize/2 ,wire_width], layer = layer))
    finish_se.connect('E', last_port)

    D.add_port(port = start_nw.ports['W'], name = 1)
    D.add_port(port = finish_se.ports['W'], name = 2)

    D.info['num_squares'] = num_meanders*(xsize/wire_width)
    D.info['area'] = xsize*ysize
    D.info['size'] = (xsize, ysize)

    return D


def snspd_expanded(wire_width = 0.2, wire_pitch = 0.6, size = (10,8),
           num_squares = None, connector_width = 1, turn_ratio = 4,
           terminals_same_side = False, layer = 0):
    """ Creates an optimally-rounded SNSPD with wires coming out of it that expand"""
    D = Device('snspd_expanded')
    S = snspd(wire_width = wire_width, wire_pitch = wire_pitch,
                        size = size, num_squares = num_squares, turn_ratio = turn_ratio,
                        terminals_same_side = terminals_same_side, layer = layer)
    s = D.add_ref(S)
    step_device = optimal_step(start_width = wire_width, end_width = connector_width,
                            num_pts = 100, anticrowding_factor = 2, width_tol = 1e-3,
                            layer = layer)
    step1 = D.add_ref(step_device)
    step2 = D.add_ref(step_device)
    step1.connect(port = 1, destination = s.ports[1])
    step2.connect(port = 1, destination = s.ports[2])
    D.add_port(name = 1, port = step1.ports[2])
    D.add_port(name = 2, port = step2.ports[2])

    D.info = S.info
    S.info = {}

    return D


#==============================================================================
# Example code
#==============================================================================

#s = snspd(wire_width = 0.2, wire_pitch = 0.6, size = [10,3], terminals_same_side = True)
#quickplot(s)

#s = snspd(wire_width = 0.2, wire_pitch = 0.6, size = [10, None],
#          num_squares = 1000, terminals_same_side = True)
#quickplot(s)

def ytron_round(rho = 1, arm_lengths = (500,300),  source_length = 500,
                arm_widths = (200, 200), theta = 2.5, theta_resolution = 10,
                layer = 0):

    #==========================================================================
    #  Create the basic geometry
    #==========================================================================
    theta = theta*pi/180
    theta_resolution = theta_resolution*pi/180
    thetalist = np.linspace(-(pi-theta),-theta, int((pi-2*theta)/theta_resolution) + 2)
    semicircle_x = rho*cos(thetalist)
    semicircle_y = rho*sin(thetalist)+rho

    # Rest of yTron
    xc = rho*cos(theta)
    yc = rho*sin(theta)
    arm_x_left  = arm_lengths[0]*sin(theta)
    arm_y_left  = arm_lengths[0]*cos(theta)
    arm_x_right = arm_lengths[1]*sin(theta)
    arm_y_right = arm_lengths[1]*cos(theta)

    # Write out x and y coords for yTron polygon
    xpts = semicircle_x.tolist() + [xc+arm_x_right, xc+arm_x_right+arm_widths[1], xc+arm_widths[1], \
           xc+arm_widths[1], 0, -(xc+arm_widths[0]), -(xc+arm_widths[0]), -(xc+arm_x_left+arm_widths[0]), -(xc+arm_x_left)]
    ypts = semicircle_y.tolist() + [yc+arm_y_right,      yc+arm_y_right,      yc,   yc-source_length, yc-source_length,  \
            yc-source_length,        yc,        yc+arm_y_left,    yc+arm_y_left]

    #==========================================================================
    #  Create a blank device, add the geometry, and define the ports
    #==========================================================================
    D = Device(name = 'ytron')
    D.add_polygon([xpts,ypts], layer = layer)
    D.add_port(name = 'left', midpoint = [-(xc+arm_x_left+arm_widths[0]/2), yc+arm_y_left],  width = arm_widths[0], orientation = 90)
    D.add_port(name = 'right', midpoint = [xc+arm_x_right+arm_widths[1]/2, yc+arm_y_right],  width = arm_widths[1], orientation = 90)
    D.add_port(name = 'source', midpoint = [0+(arm_widths[1]-arm_widths[0])/2, -source_length+yc],  width = arm_widths[0] + arm_widths[1] + 2*xc, orientation = -90)

    #==========================================================================
    #  Record any parameters you may want to access later
    #==========================================================================
    D.info['rho'] = rho
    D.info['left_width'] =   arm_widths[0]
    D.info['right_width'] =  arm_widths[1]
    D.info['source_width'] = arm_widths[0] + arm_widths[1] + 2*xc

    return D


#==============================================================================
# Example code
#==============================================================================

#y = ytron_round(rho = 1, arm_lengths = (500,300),  source_length = 500,
                # arm_widths = (200, 200), theta = 2.5, theta_resolution = 10,
                # layer = 0)
#quickplot(y)
