# # -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import copy as python_copy
import itertools
import json
import os.path
import pickle
import warnings
from collections import OrderedDict
from functools import update_wrapper

import gdspy
import numpy as np
from gdspy import clipper
from numpy import cos, exp, log, pi, sin, sinh, sqrt

from phidl.constants import _glyph, _indent, _width
from phidl.device_layout import (
    CellArray,
    Device,
    DeviceReference,
    Group,
    Polygon,
    Port,
    _parse_layer,
    make_device,
)

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
# Packer tool / Fill tool
# Photonics
# Optimal (current-crowding) curves
# Superconducting devices

# ==============================================================================
#
# Polygons / Shapes
#
# ==============================================================================


def rectangle(size=(4, 2), layer=0):
    """Generates a rectangle geometry.

    Parameters
    ----------
    size : tuple of int or float
        Width and height of rectangle.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing a single rectangle polygon.
    """
    D = Device(name="rectangle")
    points = [[size[0], size[1]], [size[0], 0], [0, 0], [0, size[1]]]
    D.add_polygon(points, layer=layer)
    return D


def bbox(bbox=[(-1, -1), (3, 4)], layer=0):
    """Creates a bounding box rectangle from coordinates, to allow
    creation of a rectangle bounding box directly form another shape.

    Parameters
    ----------
    bbox : list of tuples of int or float
        Coordinates of the box [(x1, y1), (x2, y2)].
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing a single rectangle polygon.

    Examples
    --------
    >>> D = pg.bbox(anothershape.bbox)
    """
    D = Device(name="bbox")
    (a, b), (c, d) = bbox
    points = ((a, b), (c, b), (c, d), (a, d))
    D.add_polygon(points, layer=layer)
    return D


def cross(length=10, width=3, layer=0):
    """Generates a right-angle cross (+ shape, symmetric) from two
    rectangles of specified length and width.

    Parameters
    ----------
    length : int or float
        Length of the cross from one end to the other.
    width : int or float
        Width of the arms of the cross.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D: Device
        A Device containing a cross polygon.
    """
    D = Device(name="cross")
    R = rectangle(size=(width, length), layer=layer)
    r1 = D.add_ref(R).rotate(90)
    r2 = D.add_ref(R)
    r1.center = (0, 0)
    r2.center = (0, 0)
    return D


def ellipse(radii=(10, 5), angle_resolution=2.5, layer=0):
    """Generates an ellipse geometry.

    Parameters
    ----------
    radii : tuple of int or float
        Semimajor (x) and semiminor (y) axis lengths of the ellipse.
    angle_resolution : int or float
        Resolution of the curve of the ring (# of degrees per point).
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing an ellipse polygon.
    """
    D = Device(name="ellipse")
    a = radii[0]
    b = radii[1]
    t = np.linspace(0, 360, int(np.ceil(360 / angle_resolution) + 1)) * pi / 180
    r = a * b / (sqrt((b * cos(t)) ** 2 + (a * sin(t)) ** 2))
    xpts = r * cos(t)
    ypts = r * sin(t)
    D.add_polygon(points=(xpts, ypts), layer=layer)
    return D


def circle(radius=10, angle_resolution=2.5, layer=0):
    """Generates a circle geometry.

    Parameters
    ----------
    radius : int or float
        Radius of the circle.
    angle_resolution : int or float
        Resolution of the curve of the ring (# of degrees per point).
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing a circle polygon.
    """
    D = Device(name="circle")
    t = np.linspace(0, 360, int(np.ceil(360 / angle_resolution) + 1)) * pi / 180
    xpts = (radius * cos(t)).tolist()
    ypts = (radius * sin(t)).tolist()
    D.add_polygon(points=(xpts, ypts), layer=layer)
    return D


def ring(radius=10, width=0.5, angle_resolution=2.5, layer=0):
    """Generates a ring geometry.

    Parameters
    ----------
    radius : int or float
        Radius of the ring centerline
    width : int or float
        Width of the ring.
    angle_resolution : int or float
        Resolution of the curve of the ring (# of degrees per point).
    layer :
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing a ring polygon.

    Notes
    -----
    The ring is formed by taking the radius out to the specified value, and
    then constructing the thickness by dividing the width in half and adding
    that value to either side of the radius.

    The angle_resolution alters the precision of the curve of the ring. Larger
    values yield lower resolution.
    """
    D = Device(name="ring")
    inner_radius = radius - width / 2
    outer_radius = radius + width / 2
    n = int(np.round(360 / angle_resolution))
    t = np.linspace(0, 360, n + 1) * pi / 180
    inner_points_x = (inner_radius * cos(t)).tolist()
    inner_points_y = (inner_radius * sin(t)).tolist()
    outer_points_x = (outer_radius * cos(t)).tolist()
    outer_points_y = (outer_radius * sin(t)).tolist()
    xpts = inner_points_x + outer_points_x[::-1]
    ypts = inner_points_y + outer_points_y[::-1]
    D.add_polygon(points=(xpts, ypts), layer=layer)
    return D


def arc(radius=10, width=0.5, theta=45, start_angle=0, angle_resolution=2.5, layer=0):
    """Creates an arc of arclength ``theta`` starting at angle
    ``start_angle``.

    Parameters
    ----------
    radius : int or float
        Radius of the arc centerline.
    width : int or float
        Width of the arc.
    theta : int or float
        Total angle coverage of the arc.
    start_angle : int or float
        Starting angle.
    angle_resolution : int or float
        Resolution of the curve of the arc.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing an arc polygon and two ports (`1` and `2`) on
        either end.

    Notes
    -----
    Theta = 0 is located along the positive x-axis relative to the center of
    the arc.

    Ports are added to each end of the arc to facilitate connecting those ends
    to other geometries.
    """
    inner_radius = radius - width / 2
    outer_radius = radius + width / 2
    angle1 = (start_angle) * pi / 180
    angle2 = (start_angle + theta) * pi / 180
    t = np.linspace(angle1, angle2, int(np.ceil(abs(theta) / angle_resolution)))
    inner_points_x = (inner_radius * cos(t)).tolist()
    inner_points_y = (inner_radius * sin(t)).tolist()
    outer_points_x = (outer_radius * cos(t)).tolist()
    outer_points_y = (outer_radius * sin(t)).tolist()
    xpts = inner_points_x + outer_points_x[::-1]
    ypts = inner_points_y + outer_points_y[::-1]

    D = Device("arc")
    D.add_polygon(points=(xpts, ypts), layer=layer)
    D.add_port(
        name=1,
        midpoint=(radius * cos(angle1), radius * sin(angle1)),
        width=width,
        orientation=start_angle - 90 + 180 * (theta < 0),
    )
    D.add_port(
        name=2,
        midpoint=(radius * cos(angle2), radius * sin(angle2)),
        width=width,
        orientation=start_angle + theta + 90 - 180 * (theta < 0),
    )
    D.info["length"] = (abs(theta) * pi / 180) * radius
    return D


def turn(port, radius=10, angle=270, angle_resolution=2.5, layer=0):
    """Starting from a port, creates an arc which connects to the specified
    port on one end.

    Parameters
    ----------
    port : Port
        Port to anchor arc to.
    radius : int or float
        Radius of the arc centerline.
    angle : int or float
        Total angle coverage of the arc.
    angle_resolution : int or float
        Resolution of the curve of the arc.
    layer : int, array-like[2], or set
        Specific layer(s) to put the polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing an arc polygon and two ports (`1` and `2`) on
        either end.

    Notes
    -----
    Angle = 0 is located along the positive x-axis relative to the center of
    the arc.
    Ports are added to each end of the arc to facilitate connecting those ends
    to other geometries.
    Port `2` is aligned to connect to the specified port.
    """
    D = arc(
        radius=radius,
        width=port.width,
        theta=angle,
        start_angle=0,
        angle_resolution=angle_resolution,
        layer=layer,
    )
    D.rotate(
        angle=180 + port.orientation - D.ports[1].orientation,
        center=D.ports[1].midpoint,
    )
    D.move(origin=D.ports[1], destination=port)
    return D


def straight(size=(4, 2), layer=0):
    """Generates a rectangular wire geometry with ports on the length edges.

    Parameters
    ----------
    size : tuple of int or float
        The length and width of the rectangle.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing a rectangle polygon and two ports (`1` and `2`) on
        either end.

    Notes
    -----
    Ports are included on both sides of the length edge (i.e. size[0]) of the
    geometry.
    """
    D = Device(name="wire")
    points = [[size[0], size[1]], [size[0], 0], [0, 0], [0, size[1]]]
    D.add_polygon(points, layer=layer)
    D.add_port(name=1, midpoint=(size[0] / 2, size[1]), width=size[0], orientation=90)
    D.add_port(name=2, midpoint=(size[0] / 2, 0), width=size[0], orientation=-90)
    return D


def L(width=1, size=(10, 20), layer=0):
    """Generates an 'L' geometry with ports on both ends.

    Parameters
    ----------
    width : int or float
        Thickness of the line forming the L.
    size : tuple of int or float
        Lengths of the base and height of the L, respectively.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing an L-shaped polygon and two ports (`1` and `2`) on
        either end of the L.
    """
    D = Device(name="L")
    w = width / 2
    s1, s2 = size
    points = [(-w, -w), (s1, -w), (s1, w), (w, w), (w, s2), (-w, s2), (-w, -w)]
    D.add_polygon(points, layer=layer)
    D.add_port(name=1, midpoint=(0, s2), width=width, orientation=90)
    D.add_port(name=2, midpoint=(s1, 0), width=width, orientation=0)
    return D


def C(width=1, size=(10, 20), layer=0):
    """Generates a 'C' geometry with ports on both ends.

    Parameters
    ----------
    width : int or float
        Thickness of the line forming the C.
    size : tuple of int or float
        Lengths of the base + top edges and the height of the C, respectively.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing a [-bracket-shaped polygon and two ports (`1` and
        `2`) on either end of the [ shape.
    """
    D = Device(name="C")
    w = width / 2
    s1, s2 = size
    points = [
        (-w, -w),
        (s1, -w),
        (s1, w),
        (w, w),
        (w, s2 - w),
        (s1, s2 - w),
        (s1, s2 + w),
        (-w, s2 + w),
        (-w, -w),
    ]
    D.add_polygon(points, layer=layer)
    D.add_port(name=1, midpoint=(s1, s2), width=width, orientation=0)
    D.add_port(name=2, midpoint=(s1, 0), width=width, orientation=0)
    return D


# ==============================================================================
#
# Boolean functions
#
# ==============================================================================


def offset(
    elements,
    distance=0.1,
    join_first=True,
    precision=1e-4,
    num_divisions=[1, 1],
    join="miter",
    tolerance=2,
    max_points=4000,
    layer=0,
):
    """Shrinks or expands a polygon or set of polygons.

    Parameters
    ----------
    elements : Device(/Reference), list of Device(/Reference), or Polygon
        Polygons to offset or Device containing polygons to offset.
    distance : int or float
        Distance to offset polygons. Positive values expand, negative shrink.
    precision : float
        Desired precision for rounding vertex coordinates.
    num_divisions : array-like[2] of int
        The number of divisions with which the geometry is divided into
        multiple rectangular regions. This allows for each region to be
        processed sequentially, which is more computationally efficient.
    join : {'miter', 'bevel', 'round'}
        Type of join used to create the offset polygon.
    tolerance : int or float
        For miter joints, this number must be at least 2 and it represents the
        maximal distance in multiples of offset between new vertices and their
        original position before beveling to avoid spikes at acute joints. For
        round joints, it indicates the curvature resolution in number of
        points per full circle.
    max_points : int
        The maximum number of vertices within the resulting polygon.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing a polygon(s) with the specified offset applied.
    """
    if type(elements) is not list:
        elements = [elements]
    polygons_to_offset = []
    for e in elements:
        if isinstance(e, (Device, DeviceReference)):
            polygons_to_offset += e.get_polygons(by_spec=False)
        elif isinstance(e, (Polygon, gdspy.Polygon)):
            polygons_to_offset.append(e)
    if len(polygons_to_offset) == 0:
        return Device("offset")

    polygons_to_offset = _merge_floating_point_errors(
        polygons_to_offset, tol=precision / 1000
    )
    gds_layer, gds_datatype = _parse_layer(layer)
    if all(np.array(num_divisions) == np.array([1, 1])):
        p = gdspy.offset(
            polygons_to_offset,
            distance=distance,
            join=join,
            tolerance=tolerance,
            precision=precision,
            join_first=join_first,
            max_points=max_points,
            layer=gds_layer,
            datatype=gds_datatype,
        )
    else:
        p = _offset_polygons_parallel(
            polygons_to_offset,
            distance=distance,
            num_divisions=num_divisions,
            join_first=join_first,
            precision=precision,
            join=join,
            tolerance=tolerance,
        )

    D = Device("offset")
    polygons = D.add_polygon(p, layer=layer)
    [
        polygon.fracture(max_points=max_points, precision=precision)
        for polygon in polygons
    ]
    return D


def boolean(  # noqa: C901
    A, B, operation, precision=1e-4, num_divisions=[1, 1], max_points=4000, layer=0
):
    """Performs boolean operations between 2 Device/DeviceReference objects
    or lists of Devices/DeviceReferences.

    Parameters
    ----------
    A : Device(/Reference) or list of Device(/Reference) or Polygon
        Input Devices.
    B : Device(/Reference) or list of Device(/Reference) or Polygon
        Input Devices.
    operation : {'not', 'and', 'or', 'xor', 'A-B', 'B-A', 'A+B'}
        Boolean operation to perform.
    precision : float
        Desired precision for rounding vertex coordinates.
    num_divisions : array-like[2] of int
        The number of divisions with which the geometry is divided into
        multiple rectangular regions. This allows for each region to be
        processed sequentially, which is more computationally efficient.
    max_points : int
        The maximum number of vertices within the resulting polygon.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D: Device
        A Device containing a polygon(s) with the boolean operations between
        the 2 input Devices performed.

    Notes
    -----
    'A+B' is equivalent to 'or'.
    'A-B' is equivalent to 'not'.
    'B-A' is equivalent to 'not' with the operands switched.
    """
    D = Device("boolean")

    A_polys = []
    B_polys = []
    if type(A) is not list:
        A = [A]
    if type(B) is not list:
        B = [B]
    for e in A:
        if isinstance(e, Device):
            A_polys += e.get_polygons()
        elif isinstance(e, DeviceReference):
            A_polys += e.get_polygons()
    for e in B:
        if isinstance(e, Device):
            B_polys += e.get_polygons()
        elif isinstance(e, DeviceReference):
            B_polys += e.get_polygons()

    gds_layer, gds_datatype = _parse_layer(layer)

    operation = operation.lower().replace(" ", "")
    if operation == "a-b":
        operation = "not"
    elif operation == "b-a":
        operation = "not"
        A_polys, B_polys = B_polys, A_polys
    elif operation == "a+b":
        operation = "or"
    elif operation not in ["not", "and", "or", "xor", "a-b", "b-a", "a+b"]:
        raise ValueError(
            "[PHIDL] phidl.geometry.boolean() `operation` "
            "parameter not recognized, must be one of the "
            "following:  'not', 'and', 'or', 'xor', 'A-B', "
            "'B-A', 'A+B'"
        )

    # Check for trivial solutions
    if (len(A_polys) == 0) or (len(B_polys) == 0):
        if operation == "not":
            if len(A_polys) == 0:
                p = None
            elif len(B_polys) == 0:
                p = A_polys
        elif operation == "and":
            p = None
        elif (operation == "or") or (operation == "xor"):
            if (len(A_polys) == 0) and (len(B_polys) == 0):
                p = None
            elif len(A_polys) == 0:
                p = B_polys
            elif len(B_polys) == 0:
                p = A_polys
    else:
        # If no trivial solutions, run boolean operation either in parallel or
        # straight
        if all(np.array(num_divisions) == np.array([1, 1])):
            p = gdspy.boolean(
                operand1=A_polys,
                operand2=B_polys,
                operation=operation,
                precision=precision,
                max_points=max_points,
                layer=gds_layer,
                datatype=gds_datatype,
            )
        else:
            p = _boolean_polygons_parallel(
                polygons_A=A_polys,
                polygons_B=B_polys,
                num_divisions=num_divisions,
                operation=operation,
                precision=precision,
            )

    if p is not None:
        polygons = D.add_polygon(p, layer=layer)
        [
            polygon.fracture(max_points=max_points, precision=precision)
            for polygon in polygons
        ]
    return D


def outline(
    elements,
    distance=1,
    precision=1e-4,
    num_divisions=[1, 1],
    join="miter",
    tolerance=2,
    join_first=True,
    max_points=4000,
    open_ports=False,
    layer=0,
):
    """Creates an outline around all the polygons passed in the `elements`
    argument. `elements` may be a Device, Polygon, or list of Devices.

    Parameters
    ----------
    elements : Device(/Reference), list of Device(/Reference), or Polygon
        Polygons to outline or Device containing polygons to outline.
    distance : int or float
        Distance to offset polygons. Positive values expand, negative shrink.
    precision : float
        Desired precision for rounding vertex coordinates.
    num_divisions : array-like[2] of int
        The number of divisions with which the geometry is divided into
        multiple rectangular regions. This allows for each region to be
        processed sequentially, which is more computationally efficient.
    join : {'miter', 'bevel', 'round'}
        Type of join used to create the offset polygon.
    tolerance : int or float
        For miter joints, this number must be at least 2 and it represents the
        maximal distance in multiples of offset between new vertices and their
        original position before beveling to avoid spikes at acute joints. For
        round joints, it indicates the curvature resolution in number of
        points per full circle.
    join_first : bool
        Join all paths before offsetting to avoid unnecessary joins in
        adjacent polygon sides.
    max_points : int
        The maximum number of vertices within the resulting polygon.
    open_ports : bool or float
        If not False, holes will be cut in the outline such that the Ports are
        not covered. If True, the holes will have the same width as the Ports.
        If a float, the holes will be be widened by that value (useful for fully
        clearing the outline around the Ports for positive-tone processes
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.)

    Returns
    -------
    D : Device
        A Device containing the outlined polygon(s).
    """
    D = Device("outline")
    if type(elements) is not list:
        elements = [elements]
    port_list = []
    for e in elements:
        if isinstance(e, Device):
            D.add_ref(e)
            port_list += list(e.ports.values())
        else:
            D.add(e)
    gds_layer, gds_datatype = _parse_layer(layer)

    D_bloated = offset(
        D,
        distance=distance,
        join_first=join_first,
        num_divisions=num_divisions,
        precision=precision,
        max_points=max_points,
        join=join,
        tolerance=tolerance,
        layer=layer,
    )

    Trim = Device()
    if open_ports is not False:
        if open_ports is True:
            trim_width = 0
        else:
            trim_width = open_ports * 2
        for port in port_list:
            trim = compass(size=(distance + 6 * precision, port.width + trim_width))
            trim_ref = Trim << trim
            trim_ref.connect("E", port, overlap=2 * precision)

    Outline = boolean(
        A=D_bloated,
        B=[D, Trim],
        operation="A-B",
        num_divisions=num_divisions,
        max_points=max_points,
        precision=precision,
        layer=layer,
    )
    if open_ports is not False and len(elements) == 1:
        for port in port_list:
            Outline.add_port(port=port)
    return Outline


def inset(elements, distance=0.1, join_first=True, precision=1e-4, layer=0):
    raise ValueError("[PHIDL] pg.inset() is deprecated, " "please use pg.offset()")


def invert(
    elements, border=10, precision=1e-4, num_divisions=[1, 1], max_points=4000, layer=0
):
    """Creates an inverted version of the input shapes with an additional
    border around the edges.

    Parameters
    ----------
    elements : Device(/Reference), list of Device(/Reference), or Polygon
        A Device containing the polygons to invert.
    border : int or float
        Size of the border around the inverted shape (border value is the
        distance from the edges of the boundary box defining the inverted
        shape to the border, and is applied to all 4 sides of the shape).
    precision : float
        Desired precision for rounding vertex coordinates.
    num_divisions : array-like[2] of int
        The number of divisions with which the geometry is divided into
        multiple rectangular regions. This allows for each region to be
        processed sequentially, which is more computationally efficient.
    max_points : int
        The maximum number of vertices within the resulting polygon.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing the inverted version of the input shape(s) and the
        corresponding border(s).
    """
    Temp = Device()
    if type(elements) is not list:
        elements = [elements]
    for e in elements:
        if isinstance(e, Device):
            Temp.add_ref(e)
        else:
            Temp.add(e)
    gds_layer, gds_datatype = _parse_layer(layer)

    # Build the rectangle around the device D
    R = rectangle(size=(Temp.xsize + 2 * border, Temp.ysize + 2 * border))
    R.center = Temp.center

    D = boolean(
        A=R,
        B=Temp,
        operation="A-B",
        precision=precision,
        num_divisions=num_divisions,
        max_points=max_points,
        layer=layer,
    )
    return D


def xor_diff(A, B, precision=1e-4):
    """Given two Devices A and B, performs the layer-by-layer XOR
    difference between A and B and returns polygons representing the
    differences between A and B.

    Parameters
    ----------
    A : Device(/Reference) or list of Device(/Reference)
        A Device containing a polygon(s).
    B : Device(/Reference) or list of Device(/Reference)
        A Device containing a polygon(s).
    precision : float
        Desired precision for rounding vertex coordinates.

    Returns
    ------
    D: Device
        A Device containing a polygon(s) defined by the XOR difference result
        between A and B.
    """

    D = Device("xor_diff")
    A_polys = A.get_polygons(by_spec=True)
    B_polys = B.get_polygons(by_spec=True)
    A_layers = A_polys.keys()
    B_layers = B_polys.keys()
    all_layers = set()
    all_layers.update(A_layers)
    all_layers.update(B_layers)
    for layer in all_layers:
        if (layer in A_layers) and (layer in B_layers):
            p = gdspy.boolean(
                operand1=A_polys[layer],
                operand2=B_polys[layer],
                operation="xor",
                precision=precision,
                max_points=4000,
                layer=layer[0],
                datatype=layer[1],
            )
        elif layer in A_layers:
            p = A_polys[layer]
        elif layer in B_layers:
            p = B_polys[layer]
        if p is not None:
            D.add_polygon(p, layer=layer)
    return D


def union(D, by_layer=False, precision=1e-4, join_first=True, max_points=4000, layer=0):
    """Performs the union of all polygons within a Device.

    Parameters
    ----------
    D : Device(/Reference) or list of Device(/Reference)
        A Device containing polygons to perform a union on.
    by_Layer : bool
        If true, performs the union operation layer-wise so each layer can be
        individually combined.
    precision : float
        Desired precision for rounding vertex coordinates.
    join_first : bool
        Join all paths before offsetting to avoid unnecessary joins in
        adjacent polygon sides.
    max_points : int
        The maximum number of vertices within the resulting polygon.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    U : Device
        A Device containing the union of the polygons within the input Device.
    """
    U = Device("union")

    if by_layer:
        all_polygons = D.get_polygons(by_spec=True)
        for layer, polygons in all_polygons.items():
            unioned_polygons = _union_polygons(
                polygons, precision=precision, max_points=max_points
            )
            U.add_polygon(unioned_polygons, layer=layer)
    else:
        all_polygons = D.get_polygons(by_spec=False)
        unioned_polygons = _union_polygons(
            all_polygons, precision=precision, max_points=max_points
        )
        U.add_polygon(unioned_polygons, layer=layer)
    return U


def _union_polygons(polygons, precision=1e-4, max_points=4000):
    """Performs the union of all polygons within a PolygonSet or list of
    polygons.

    Parameters
    ----------
    polygons : PolygonSet or list of polygons
        A set containing the input polygons.
    precision : float
        Desired precision for rounding vertex coordinates.
    max_points : int
        The maximum number of vertices within the resulting polygon.

    Returns
    -------
    unioned : polygon
        The result of the union of all the polygons within the input
        PolygonSet.
    """
    polygons = _merge_floating_point_errors(polygons, tol=precision / 1000)
    unioned = gdspy.boolean(
        polygons, [], operation="or", precision=precision, max_points=max_points
    )
    return unioned


def _merge_floating_point_errors(polygons, tol=1e-10):
    """Fixes floating point errors in the input polygon(s) by merging values
    within the tolerance `tol`. See _merge_nearby_floating_points for
    specifics.

    Parameters
    ----------
    polygons : PolygonSet or list of polygons
        Set of polygons with floating point errors.
    tol : float
        Tolerance within which points will be merged.

    Returns
    -------
    polygons_fixed : PolygonSet
        Set of corrected polygons.
    """
    stacked_polygons = np.vstack(polygons)
    x = stacked_polygons[:, 0]
    y = stacked_polygons[:, 1]
    polygon_indices = np.cumsum([len(p) for p in polygons])

    xfixed = _merge_nearby_floating_points(x, tol=tol)
    yfixed = _merge_nearby_floating_points(y, tol=tol)
    stacked_polygons_fixed = np.vstack([xfixed, yfixed]).T
    polygons_fixed = np.vsplit(stacked_polygons_fixed, polygon_indices[:-1])
    return polygons_fixed


def _merge_nearby_floating_points(x, tol=1e-10):
    """Takes an array `x` and merges any values within the tolerance `tol`.

    Parameters
    ----------
    x : list of int or float
        Array of values with floating point errors.
    tol : float
        Tolerance within which points will be merged.

    Returns
    -------
    xsort : list of int or float
        Corrected and sorted array.

    Examples
    --------
    If given:
    >>> x = [-2, -1, 0, 1.0001, 1.0002, 1.0003, 4, 5, 5.003, 6, 7, 8]
    >>> _merge_nearby_floating_points(x, tol = 1e-3)
    Will return:
    >>> [-2, -1, 0, 1.0001, 1.0001, 1.0001, 4, 5, 5.003, 6, 7, 8].
    """
    xargsort = np.argsort(x)
    xargunsort = np.argsort(xargsort)
    xsort = x[xargsort]
    xsortthreshold = np.diff(xsort) < tol
    xsortthresholdind = np.argwhere(xsortthreshold)

    # Merge nearby floating point values
    for xi in xsortthresholdind:
        xsort[xi + 1] = xsort[xi]
    return xsort[xargunsort]


def _crop_region(polygons, left, bottom, right, top, precision):
    """Given a rectangular boundary defined by left/bottom/right/top, this
    takes a list of polygons and cuts them at the boundary, discarding parts
    of the polygons outside the rectangle.

    Parameters
    ----------
    polygons : PolygonSet or list of polygons
        Set or list of polygons to be cropped.
    left : int or float
        The x-coordinate of the lefthand boundary.
    bottom : int or float
        The y-coordinate of the bottom boundary.
    right : int or float
        The x-coordinate of the righthand boundary.
    top : int or float
        The y-coordinate of the top boundary.
    precision : float
        Desired precision for rounding vertex coordinates.

    Returns
    -------
    cropped_polygons : PolygonSet or list of polygons
        Set or list of polygons that are cropped according to the specified
        boundary.
    """
    cropped_polygons = []
    for p in polygons:
        clipped_polys = clipper._chop(p, [top, bottom], 1, 1 / precision)
        # polygon, [cuts], axis, scale
        for cp in clipped_polys[1]:
            result = clipper._chop(cp, [left, right], 0, 1 / precision)
            cropped_polygons += list(result[1])
    return cropped_polygons


def _crop_edge_polygons(all_polygons, bboxes, left, bottom, right, top, precision):
    """Parses out which polygons are along the edge of the rectangle and need
    to be cropped and which are deep inside the rectangle region and can be
    left alone, then crops only those polygons along the edge.

    Parameters
    ----------
    all_polygons : PolygonSet or list of polygons
        Set or list of polygons to be cropped.
    bboxes : list
        List of all polygon bboxes in all_polygons.
    left : int or float
        The x-coordinate of the lefthand boundary.
    bottom : int or float
        The y-coordinate of the bottom boundary.
    right : int or float
        The x-coordinate of the righthand boundary.
    top : int or float
        The y-coordinate of the top boundary.
    precision : float
        Desired precision for rounding vertex coordinates.

    Returns
    -------
    polygons_to_process : PolygonSet or list of polygons
        Set or list of polygons with crop applied to edge polygons.
    """
    polygons_in_rect_i = _find_bboxes_in_rect(bboxes, left, bottom, right, top)
    polygons_edge_i = _find_bboxes_on_rect_edge(bboxes, left, bottom, right, top)
    polygons_in_rect_no_edge_i = polygons_in_rect_i & (~polygons_edge_i)

    # Crop polygons along the edge and recombine them with polygons inside the
    # rectangle
    polygons_edge = all_polygons[polygons_edge_i]
    polygons_in_rect_no_edge = all_polygons[polygons_in_rect_no_edge_i].tolist()
    polygons_edge_cropped = _crop_region(
        polygons_edge, left, bottom, right, top, precision=precision
    )
    polygons_to_process = polygons_in_rect_no_edge + polygons_edge_cropped

    return polygons_to_process


def _find_bboxes_in_rect(bboxes, left, bottom, right, top):
    """Given a list of polygon bounding boxes and a rectangle defined by
    left/bottom/right/top, this function returns those polygons which overlap
    the rectangle.

    Parameters
    ----------
    bboxes : list
        List of all polygon bboxes.
    left : int or float
        The x-coordinate of the lefthand boundary.
    bottom : int or float
        The y-coordinate of the bottom boundary.
    right : int or float
        The x-coordinate of the righthand boundary.
    top : int or float
        The y-coordinate of the top boundary.

    Returns
    -------
    result : list
        List of all polygon bboxes that overlap with the defined rectangle.
    """
    result = (
        (bboxes[:, 0] <= right)
        & (bboxes[:, 2] >= left)
        & (bboxes[:, 1] <= top)
        & (bboxes[:, 3] >= bottom)
    )
    return result


# _find_bboxes_on_rect_edge
def _find_bboxes_on_rect_edge(bboxes, left, bottom, right, top):
    """Given a list of polygon bounding boxes and a rectangular boundary
    defined by left/bottom/right/top, this function returns those polygons
    which intersect the rectangular boundary.

    Parameters
    ----------
    bboxes : list
        List of all polygon bboxes.
    left : int or float
        The x-coordinate of the lefthand boundary.
    bottom : int or float
        The y-coordinate of the bottom boundary.
    right : int or float
        The x-coordinate of the righthand boundary.
    top : int or float
        The y-coordinate of the top boundary.

    Returns
    -------
    result : list
        List of all polygon bboxes that intersect the defined rectangular
        boundary.
    """
    bboxes_left = _find_bboxes_in_rect(bboxes, left, bottom, left, top)
    bboxes_right = _find_bboxes_in_rect(bboxes, right, bottom, right, top)
    bboxes_top = _find_bboxes_in_rect(bboxes, left, top, right, top)
    bboxes_bottom = _find_bboxes_in_rect(bboxes, left, bottom, right, bottom)
    result = bboxes_left | bboxes_right | bboxes_top | bboxes_bottom
    return result


def _offset_region(
    all_polygons,
    bboxes,
    left,
    bottom,
    right,
    top,
    distance=5,
    join_first=True,
    precision=1e-4,
    join="miter",
    tolerance=2,
):
    """Taking a region of e.g. size (x, y) which needs to be offset by
    distance d, this function crops out a region (x+2*d, y+2*d) large, offsets
    that region, then crops it back to size (x, y) to create a valid result.

    Parameters
    ----------
    all_polygons : PolygonSet or list of polygons
        Set or list of polygons to be cropped and offset.
    bboxes : list
        List of all polygon bboxes in all_polygons.
    left : int or float
        The x-coordinate of the lefthand boundary.
    bottom : int or float
        The y-coordinate of the bottom boundary.
    right : int or float
        The x-coordinate of the righthand boundary.
    top : int or float
        The y-coordinate of the top boundary.
    distance : int or float
        Distance to offset polygons. Positive values expand, negative shrink.
    join_first : bool
        Join all paths before offsetting to avoid unnecessary joins in
        adjacent polygon sides.
    precision : float
        Desired precision for rounding vertex coordinates.
    join : {'miter', 'bevel', 'round'}
        Type of join used to create the offset polygon.
    tolerance : int or float
        For miter joints, this number must be at least 2 and it represents the
        maximal distance in multiples of offset between new vertices and their
        original position before beveling to avoid spikes at acute joints. For
        round joints, it indicates the curvature resolution in number of
        points per full circle.

    Returns
    -------
    polygons_offset_cropped :
        The resulting input polygons that are cropped to be between the
        coordinates (left, bottom, right, top)

    """

    # Mark out a region slightly larger than the final desired region
    d = distance * 1.01

    polygons_to_offset = _crop_edge_polygons(
        all_polygons,
        bboxes,
        left - d,
        bottom - d,
        right + d,
        top + d,
        precision=precision,
    )

    # Offset the resulting cropped polygons and recrop to final desired size
    polygons_offset = clipper.offset(
        polygons_to_offset, distance, join, tolerance, 1 / precision, int(join_first)
    )
    polygons_offset_cropped = _crop_region(
        polygons_offset, left, bottom, right, top, precision=precision
    )

    return polygons_offset_cropped


def _polygons_to_bboxes(polygons):
    """Generates the bboxes of all input polygons.

    Parameters
    ----------
    polygons : PolygonSet or list of polygons
        Set or list of polygons to generate bboxes of.

    Returns
    -------
    bboxes : list
        List of all polygon bboxes in polygons.
    """
    #    Build bounding boxes
    bboxes = np.empty([len(polygons), 4])
    for n, p in enumerate(polygons):
        left, bottom = np.min(p, axis=0)
        right, top = np.max(p, axis=0)
        bboxes[n] = [left, bottom, right, top]
    return bboxes


def _offset_polygons_parallel(
    polygons,
    distance=5,
    num_divisions=[10, 10],
    join_first=True,
    precision=1e-4,
    join="miter",
    tolerance=2,
):
    """Performs the offset function on a list of subsections of the original
    geometry

    Parameters
    ----------
    polygons : PolygonSet or list of polygons

    distance : int or float
        Distance to offset polygons. Positive values expand, negative shrink.
    num_divisions : array-like[2] of int
        The number of divisions with which the geometry is divided into
        multiple rectangular regions. This allows for each region to be
        processed sequentially, which is more computationally efficient.
    join_first : bool
        Join all paths before offsetting to avoid unnecessary joins in
        adjacent polygon sides.
    precision : float
        Desired precision for rounding vertex coordinates.
    join : {'miter', 'bevel', 'round'}
        Type of join used to create the offset polygon.
    tolerance : int or float
        For miter joints, this number must be at least 2 and it represents the
        maximal distance in multiples of offset between new vertices and their
        original position before beveling to avoid spikes at acute joints. For
        round joints, it indicates the curvature resolution in number of
        points per full circle.

    Returns
    -------
    offset_polygons :

    """
    # Build bounding boxes
    polygons = np.asarray(polygons)
    bboxes = _polygons_to_bboxes(polygons)

    xmin, ymin = np.min(bboxes[:, 0:2], axis=0) - distance
    xmax, ymax = np.max(bboxes[:, 2:4], axis=0) + distance

    xsize = xmax - xmin
    ysize = ymax - ymin
    xdelta = xsize / num_divisions[0]
    ydelta = ysize / num_divisions[1]
    xcorners = xmin + np.arange(num_divisions[0]) * xdelta
    ycorners = ymin + np.arange(num_divisions[1]) * ydelta

    offset_polygons = []
    for n, xc in enumerate(xcorners):
        for m, yc in enumerate(ycorners):
            left = xc
            right = xc + xdelta
            bottom = yc
            top = yc + ydelta
            _offset_region_polygons = _offset_region(
                polygons,
                bboxes,
                left,
                bottom,
                right,
                top,
                distance=distance,
                join_first=join_first,
                precision=precision,
                join=join,
                tolerance=tolerance,
            )
            offset_polygons += _offset_region_polygons

    return offset_polygons


def _boolean_region(
    all_polygons_A,
    all_polygons_B,
    bboxes_A,
    bboxes_B,
    left,
    bottom,
    right,
    top,
    operation="and",
    precision=1e-4,
):
    """Taking a region of e.g. size (x, y) which needs to be booleaned,
    this function crops out a region (x, y) large from each set of polygons
    (A and B), booleans that cropped region and returns the result.

    Parameters
    ----------
    all_polygons_A : PolygonSet or list of polygons
        Set or list of polygons to be booleaned.
    all_polygons_B : PolygonSet or list of polygons
        Set or list of polygons to be booleaned.
    bboxes_A : list
        List of all polygon bboxes in all_polygons_A
    bboxes_B : list
        List of all polygon bboxes in all_polygons_B
    left : int or float
        The x-coordinate of the lefthand boundary.
    bottom : int or float
        The y-coordinate of the bottom boundary.
    right : int or float
        The x-coordinate of the righthand boundary.
    top : int or float
        The y-coordinate of the top boundary.
    operation : {'not', 'and', 'or', 'xor', 'A-B', 'B-A', 'A+B'}
        Boolean operation to perform.
    precision : float
        Desired precision for rounding vertex coordinates.

    Returns
    -------
    polygons_boolean : PolygonSet or list of polygons
        Set or list of polygons with boolean operation applied.
    """

    polygons_to_boolean_A = _crop_edge_polygons(
        all_polygons_A, bboxes_A, left, bottom, right, top, precision
    )
    polygons_to_boolean_B = _crop_edge_polygons(
        all_polygons_B, bboxes_B, left, bottom, right, top, precision
    )
    polygons_boolean = clipper.clip(
        polygons_to_boolean_A, polygons_to_boolean_B, operation, 1 / precision
    )
    return polygons_boolean


def _boolean_polygons_parallel(
    polygons_A, polygons_B, num_divisions=[10, 10], operation="and", precision=1e-4
):
    """Performs the boolean function on a list of subsections of the original
    geometry

    Parameters
    ----------
    polygons_A : PolygonSet or list of polygons
        Set or list of polygons to be booleaned.
    polygons_B : PolygonSet or list of polygons
        Set or list of polygons to be booleaned.
    num_divisions : array-like[2] of int
        The number of divisions with which the geometry is divided into
        multiple rectangular regions. This allows for each region to be
        processed sequentially, which is more computationally efficient.
    operation : {'not', 'and', 'or', 'xor', 'A-B', 'B-A', 'A+B'}
        Boolean operation to perform.
    precision : float
        Desired precision for rounding vertex coordinates.

    Returns
    -------
    boolean_polygons : list of polygons
        All the booleaned polygons from each of the subsections

    """
    # Build bounding boxes
    polygons_A = np.asarray(polygons_A)
    polygons_B = np.asarray(polygons_B)
    bboxes_A = _polygons_to_bboxes(polygons_A)
    bboxes_B = _polygons_to_bboxes(polygons_B)

    xmin, ymin = np.min(
        [np.min(bboxes_A[:, 0:2], axis=0), np.min(bboxes_B[:, 0:2], axis=0)], axis=0
    )
    xmax, ymax = np.max(
        [np.max(bboxes_A[:, 2:4], axis=0), np.max(bboxes_B[:, 2:4], axis=0)], axis=0
    )

    xsize = xmax - xmin
    ysize = ymax - ymin
    xdelta = xsize / num_divisions[0]
    ydelta = ysize / num_divisions[1]
    xcorners = xmin + np.arange(num_divisions[0]) * xdelta
    ycorners = ymin + np.arange(num_divisions[1]) * ydelta

    boolean_polygons = []
    for n, xc in enumerate(xcorners):
        for m, yc in enumerate(ycorners):
            left = xc
            right = xc + xdelta
            bottom = yc
            top = yc + ydelta
            _boolean_region_polygons = _boolean_region(
                polygons_A,
                polygons_B,
                bboxes_A,
                bboxes_B,
                left,
                bottom,
                right,
                top,
                operation=operation,
                precision=precision,
            )
            boolean_polygons += _boolean_region_polygons

    return boolean_polygons


# ==============================================================================
#
# Lithography test structures
#
# ==============================================================================


def litho_steps(line_widths=[1, 2, 4, 8, 16], line_spacing=10, height=100, layer=0):
    """Produces a positive + negative tone linewidth test, used for
    lithography resolution test patterning.

    Parameters
    ----------
    line_widths : array-like[N] of int or float
        Widths of the steps (positive side).
    line_spacing : int or float
        Space between each step (negative side).
    height : int or float
        Height of the steps.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing the lithographic linewidth resolution test
        geometry.
    """
    D = Device("litho_steps")

    height = height / 2
    T1 = text(
        text="%s" % str(line_widths[-1]), size=height, justify="center", layer=layer
    )
    _ = D.add_ref(T1).rotate(90).movex(-height / 10)
    R1 = rectangle(size=(line_spacing, height), layer=layer)
    D.add_ref(R1).movey(-height)
    count = 0
    for i in reversed(line_widths):
        count += line_spacing + i
        R2 = rectangle(size=(i, height), layer=layer)
        D.add_ref(R1).movex(count).movey(-height)
        D.add_ref(R2).movex(count - i)

    return D


def litho_star(num_lines=20, line_width=2, diameter=200, layer=0):
    """Creates a circular-star shape from lines, used as a lithographic
    resolution test pattern.

    Parameters
    ----------
    num_lines : int
        Number of lines in the circular-star shape.
    line_width : int or float
        Thickness of star spike lines.
    diameter : int or float
        Diameter of the circular-star shape (total length of each star spike).
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing a line-based circular-star shape.
    """
    D = Device("litho_star")

    degree = 180 / num_lines
    R1 = rectangle(size=(line_width, diameter), layer=layer)
    for i in range(num_lines):
        r1 = D.add_ref(R1).rotate(degree * i)
        r1.center = (0, 0)

    return D


def litho_calipers(
    notch_size=[2, 5],
    notch_spacing=2,
    num_notches=11,
    offset_per_notch=0.1,
    row_spacing=0,
    layer1=1,
    layer2=2,
):
    """Creates a vernier caliper structure for lithography alignment
    tests. Vernier structure is made horizontally.

    Parameters
    ----------
    notch_size : array-like[2] of int or flaot
        x, y size of the notches.
    notch_spacing : int or float
        Spacing between notches on the control structure.
    num_notches : int
        Number of notches on one side of the structure (total number of
        notches is 2*num_notches + 1).
    offset_per_notch : int or float
        The amount of horizontal offset to apply to the notch spacing per
        notch on the non-control structure.
    row_spacing : int or float
        The amount of vertical space between the control and non-control
        structures.
    layer1 : int, array-like[2], or set
        Specific layer(s) to put the control geometry on.
    layer2 : int, array-like[2], or set
        Specific layer(s) to put the non-control geometry on.

    Returns
    -------
    Device
        A Device containing the caliper structures.
    """
    D = Device("litho_calipers")
    num_notches_total = num_notches * 2 + 1
    centre_notch = num_notches
    R1 = rectangle(size=(notch_size), layer=layer1)
    R2 = rectangle(size=(notch_size), layer=layer2)
    for i in range(num_notches_total):
        if i == centre_notch:
            (
                D.add_ref(R1)
                .movex(i * (notch_size[0] + notch_spacing))
                .movey(notch_size[1])
            )
            (
                D.add_ref(R2)
                .movex(
                    i * (notch_size[0] + notch_spacing)
                    + offset_per_notch * (centre_notch - i)
                )
                .movey(-2 * notch_size[1] - row_spacing)
            )
        D.add_ref(R1).movex(i * (notch_size[0] + notch_spacing))
        (
            D.add_ref(R2)
            .movex(
                i * (notch_size[0] + notch_spacing)
                + offset_per_notch * (centre_notch - i)
            )
            .movey(-notch_size[1] - row_spacing)
        )

    return D


def litho_ruler(
    height=2,
    width=0.5,
    spacing=1.2,
    scale=[3, 1, 1, 1, 1, 2, 1, 1, 1, 1],
    num_marks=21,
    layer=0,
):
    """Creates a ruler structure for lithographic measurement with marks of
    varying scales to allow for easy reading by eye.

    Parameters
    ----------
    height : float
        Height of the ruling marks.
    width : float
        Width of the ruling marks.
    spacing : float
        Center-to-center spacing of the ruling marks
    scale : array-like
        Height scale pattern of marks
    num_marks : int
        Total number of marks to generate
    num_marks : int
        Total number of marks to generate
    layer : int, array-like[2], or set
        Specific layer(s) to put the ruler geometry on.

    Returns
    -------
    Device
        A Device containing the ruler structure
    """

    D = Device("litho_ruler")
    for n in range(num_marks):
        h = height * scale[n % len(scale)]
        D << rectangle(size=(width, h), layer=layer)

    D.distribute(direction="x", spacing=spacing, separation=False, edge="x")
    D.align(alignment="ymin")
    D.flatten()
    return D


# ==============================================================================
#
# Utility functions
#
# ==============================================================================


def extract(D, layers=[0, 1]):
    """Extracts polygons from a given Device.

    Parameters
    ----------
    D : Device
        Device to extract polygons from.
    layers : array-like[2] or set
        Specific layer(s) to extract polygon geometry from.

    Returns
    -------
    Device
        A Device containing the extracted polygons.
    """
    D_extracted = Device("extract")
    if type(layers) not in (list, tuple):
        raise ValueError(
            "[PHIDL] pg.extract() Argument `layers` needs to be "
            "passed a list or tuple"
        )
    poly_dict = D.get_polygons(by_spec=True)
    parsed_layer_list = [_parse_layer(layer) for layer in layers]
    for layer, polys in poly_dict.items():
        if _parse_layer(layer) in parsed_layer_list:
            D_extracted.add_polygon(polys, layer=layer)
    return D_extracted


def copy(D):
    """Copies a Device.

    Parameters
    ----------
    D : Device
        Device to be copied.

    Returns
    -------
    Device
        Copied Device.
    """
    D_copy = Device(name=D.name)
    D_copy.info = python_copy.deepcopy(D.info)
    for ref in D.references:
        new_ref = DeviceReference(
            device=ref.parent,
            origin=ref.origin,
            rotation=ref.rotation,
            magnification=ref.magnification,
            x_reflection=ref.x_reflection,
        )
        new_ref.owner = D_copy
        D_copy.add(new_ref)
        for alias_name, alias_ref in D.aliases.items():
            if alias_ref == ref:
                D_copy.aliases[alias_name] = new_ref

    for port in D.ports.values():
        D_copy.add_port(port=port)
    for poly in D.polygons:
        D_copy.add_polygon(poly)
    for label in D.labels:
        D_copy.add_label(
            text=label.text,
            position=label.position,
            layer=(label.layer, label.texttype),
        )
    return D_copy


def deepcopy(D):
    """Deep copies a Device.

    Parameters
    ----------
    D : Device
        Device to be deep copied.

    Returns
    -------
    Device
        Deep copied Device.
    """
    D_copy = python_copy.deepcopy(D)
    D_copy.uid = Device._next_uid
    Device._next_uid += 1
    D_copy.name = D.name
    # Make sure _bb_valid is set to false for these new objects so new
    # bounding boxes are created in the cache
    for D in D_copy.get_dependencies(True):
        D._bb_valid = False
    D_copy._bb_valid = False

    return D_copy


def copy_layer(D, layer=1, new_layer=2):
    """Copies a layer within a Device to another layer in the same Device.

    Parameters
    ----------
    D : Device
        Device containing layer to be copied.
    layer : int, array-like[2], or set
        Specific layer(s) to copy.
    new_layer : int, array-like[2], or set
        Specific layer(s) to put copied layer on.

    Returns
    -------
    Device
        A Device containing the original and copied layers.
    """
    D_copied_layer = extract(D, layers=[layer])
    D_copied_layer.flatten(single_layer=new_layer)
    return D_copied_layer


def import_gds(filename, cellname=None, flatten=False):
    """Imports a GDS file and returns a Device with all the corresponding
    geometry

    Parameters
    ----------
    filename : str
        Path or name of file to be imported
    cellname : str or None
        Name of the cell that will be returned as a Device.  If None,
        will automatically select the topmost cell
    flatten : bool
        Whether to flatten the imported geometry, removing all cell heirarchy

    Returns
    -------
    Device
        A PHIDL Device with all the geometry/labels/etc imported from the GDS file

    """
    gdsii_lib = gdspy.GdsLibrary()
    gdsii_lib.read_gds(filename)
    top_level_cells = gdsii_lib.top_level()
    if cellname is not None:
        if cellname not in gdsii_lib.cells:
            raise ValueError(
                "[PHIDL] import_gds() The requested cell "
                "(named %s) is not present in file %s" % (cellname, filename)
            )
        topcell = gdsii_lib.cells[cellname]
    elif cellname is None and len(top_level_cells) == 1:
        topcell = top_level_cells[0]
    elif cellname is None and len(top_level_cells) > 1:
        raise ValueError(
            "[PHIDL] import_gds() There are multiple top-level "
            "cells, you must specify `cellname` to select of "
            "one of them"
        )

    if not flatten:
        D_list = []
        c2dmap = {}
        for cell in gdsii_lib.cells.values():
            D = Device(name=cell.name)
            D.polygons = cell.polygons
            D.references = cell.references
            D.name = cell.name
            for label in cell.labels:
                rotation = label.rotation
                if rotation is None:
                    rotation = 0
                l = D.add_label(
                    text=label.text,
                    position=np.asfarray(label.position),
                    magnification=label.magnification,
                    rotation=rotation * 180 / np.pi,
                    layer=(label.layer, label.texttype),
                )
                l.anchor = label.anchor
            c2dmap.update({cell: D})
            D_list += [D]

        for D in D_list:
            # First convert each reference so it points to the right Device
            converted_references = []
            for e in D.references:
                ref_device = c2dmap[e.ref_cell]
                if isinstance(e, gdspy.CellReference):
                    dr = DeviceReference(
                        device=ref_device,
                        origin=e.origin,
                        rotation=e.rotation,
                        magnification=e.magnification,
                        x_reflection=e.x_reflection,
                    )
                    dr.owner = D
                    converted_references.append(dr)
                elif isinstance(e, gdspy.CellArray):
                    dr = CellArray(
                        device=ref_device,
                        columns=e.columns,
                        rows=e.rows,
                        spacing=e.spacing,
                        origin=e.origin,
                        rotation=e.rotation,
                        magnification=e.magnification,
                        x_reflection=e.x_reflection,
                    )
                    dr.owner = D
                    converted_references.append(dr)
            D.references = converted_references
            # Next convert each Polygon
            temp_polygons = list(D.polygons)
            D.polygons = []
            for p in temp_polygons:
                D.add_polygon(p)

        topdevice = c2dmap[topcell]
        return topdevice

    elif flatten:
        D = Device("import_gds")
        polygons = topcell.get_polygons(by_spec=True)

        for layer_in_gds, polys in polygons.items():
            D.add_polygon(polys, layer=layer_in_gds)
        return D


def _translate_cell(c):
    D = Device(name=c.name)
    for e in c.elements:
        if isinstance(e, gdspy.PolygonSet):
            for n, points in enumerate(e.polygons):
                polygon_layer = _parse_layer((e.layers[n], e.datatypes[n]))
                D.add_polygon(points=points, layer=polygon_layer)
        elif isinstance(e, gdspy.CellReference):
            dr = DeviceReference(
                device=_translate_cell(e.ref_cell),
                origin=e.origin,
                rotation=e.rotation,
                magnification=None,
                x_reflection=e.x_reflection,
            )
            D.elements.append(dr)
    D.labels = c.labels
    return D


def preview_layerset(ls, size=100, spacing=100):
    """Generates a preview Device with representations of all the layers,
    used for previewing LayerSet color schemes in quickplot or saved .gds
    files.

    Parameters
    ----------
    ls : LayerSet
        Set of layers to preview color schemes.
    size : int or float
        Resizing factor for the preview Device.
    spacing : int or float
        The space between each layer representation.

    Returns
    -------
    D : Device
        A Device containing a representation of all the layers in the input
        LayerSet.
    """

    D = Device("layerset")
    scale = size / 100
    num_layers = len(ls._layers)
    matrix_size = int(np.ceil(np.sqrt(num_layers)))
    sorted_layers = sorted(
        ls._layers.values(), key=lambda x: (x.gds_layer, x.gds_datatype)
    )
    for n, layer in enumerate(sorted_layers):
        R = rectangle(size=(100 * scale, 100 * scale), layer=layer)
        T = text(
            text="{}\n{} / {}".format(layer.name, layer.gds_layer, layer.gds_datatype),
            size=20 * scale,
            justify="center",
            layer=layer,
        )

        T.move((50 * scale, -20 * scale))
        xloc = n % matrix_size
        yloc = int(n // matrix_size)
        D.add_ref(R).movex((100 + spacing) * xloc * scale).movey(
            -(100 + spacing) * yloc * scale
        )
        D.add_ref(T).movex((100 + spacing) * xloc * scale).movey(
            -(100 + spacing) * yloc * scale
        )
    return D


class device_lru_cache:
    """Least-recently-used (LRU) cache for Devices"""

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
                raise ValueError(
                    "[PHIDL] @device_lru_cache can only be "
                    "used on functions which return a Device"
                )
            if len(self.memo) > self.maxsize:
                self.memo.popitem(last=False)  # Remove oldest item from cache
            # Add a deepcopy of new item to cache so that if we change the
            # returned device, our stored cache item is not changed
            self.memo[pickle_str] = python_copy.deepcopy(new_cache_item)
            return new_cache_item
        else:  # if found in cache
            # Pop cache item out and put it back on the top of the cache
            cached_output = self.memo.pop(pickle_str)
            self.memo[pickle_str] = cached_output
            # Then return a copy of the cached Device
            return deepcopy(cached_output)


def _convert_port_to_geometry(port, layer=0):
    """Converts a Port to a label and a triangle Device that are then added
    to the parent.

    Parameters
    ----------
    port : Port
        Port to be converted.
    layer : int, array-like[2], or set
        Specific layer(s) to put label and geometry on.

    Notes
    -----
    The Port must start with a parent.
    """
    if port.parent is None:
        raise ValueError(
            "Port {}: Port needs a parent in which to draw".format(port.name)
        )
    if isinstance(port.parent, DeviceReference):
        device = port.parent.parent
    else:
        device = port.parent

    # A visual marker
    triangle_points = [[0, 0]] * 3
    triangle_points[0] = port.endpoints[0]
    triangle_points[1] = port.endpoints[1]
    triangle_points[2] = (
        port.midpoint + (port.normal - port.midpoint) * port.width / 10
    )[1]
    device.add_polygon(triangle_points, layer)

    # Label carrying actual information that will be recovered
    label_contents = (
        str(port.name),
        # port.midpoint,
        # rather than put this in the text, use the label
        # position
        # this can have rounding errors that are less than a
        # nanometer
        float(np.round(port.width, decimals=3)),
        float(port.orientation),
        # device, # this is definitely not serializable
        # port.info, # would like to include, but it might go
        # longer than 1024 characters
        # port.uid, # not including because it is part of the
        # build process, not the port state
    )
    label_text = json.dumps(label_contents)
    device.add_label(
        text=label_text,
        position=port.midpoint + _calculate_label_offset(port),
        magnification=0.04 * port.width,
        rotation=(90 + port.orientation) % 360,
        layer=layer,
    )


def _calculate_label_offset(port):
    """Used to put the label in a pretty position. It is added when drawing
    and substracted when extracting.

    Parameters
    ----------
    port : Port
        Port-converted label to move.

    Returns
    -------
    offset_position : array-like
        Coordinates of new port position.
    """
    offset_position = np.array(
        (-cos(pi / 180 * port.orientation), -sin(pi / 180 * port.orientation))
    )
    offset_position *= port.width * 0.05
    return offset_position


def _convert_geometry_to_port(label, layer=0):
    """Converts a label into a Port in the parent Device. The label contains
    name, width, orientation. Does not remove that label from the parent.
    Returns the new port.

    Parameters
    ----------
    label : Label
        Label to be converted into a Port.
    layer : int, array-like[2], or set
        Specific layer(s) to put Port on.

    Returns
    -------
    new_port : Port
        Port-converted label.
    """
    name, width, orientation = json.loads(label.text)
    new_port = Port(name=name, width=width, orientation=orientation)
    new_port.midpoint = label.position - _calculate_label_offset(new_port)
    return new_port


def ports_to_geometry(device, layer=0):
    """Converts Port objects over the whole Device hierarchy to geometry and
    labels.

    Parameters
    ----------
    device : Device
        Device containing Port objects to convert.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on (the special port record
        layer).

    Returns
    -------
    temp_device : Device
        A Device with all Ports converted to geometry/labels and removed.
    """
    temp_device = deepcopy(device)
    all_cells = list(temp_device.get_dependencies(recursive=True))
    all_cells.append(temp_device)
    for subcell in all_cells:
        for port in subcell.ports.values():
            _convert_port_to_geometry(port, layer=layer)
            subcell.remove(port)
    return temp_device


def geometry_to_ports(device, layer=0):
    """Converts geometry representing ports over the whole Device hierarchy
    into Port objects.

    Parameters
    ----------
    device : Device
        Device containing geometry representing ports to convert.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on (the special port record
        layer)

    Returns
    -------
    temp_device : Device
        A Device with all geometry representing ports converted to Ports and
        removed.

    Notes
    -----
    Does not mutate the device in the argument. Returns a new one lacking all
    port geometry (incl. labels)
    """
    temp_device = deepcopy(device)
    all_cells = list(temp_device.get_dependencies(recursive=True))
    all_cells.append(temp_device)
    for subcell in all_cells:  # Walk through cells
        for lab in subcell.labels:
            if lab.layer == layer:
                the_port = _convert_geometry_to_port(lab)
                subcell.add_port(name=the_port.name, port=the_port)
    temp_device.remove_layers(layers=[layer], include_labels=True)
    return temp_device


# ==============================================================================
#
# Connectors
#
# ==============================================================================


def connector(midpoint=(0, 0), width=1, orientation=0):
    """Creates a Device which has back-to-back ports.

    Parameters
    ----------
    midpoint : array-like
        Coordinates of Device midpoint.
    width : int or float
        Width of connector on non-port axis.
    orientation : int or float
        Orientation of the ports.

    Returns
    -------
    D : Device
        A Device containing a back-to-back port geometry.
    """
    D = Device(name="connector")
    D.add_port(
        name=1,
        midpoint=[midpoint[0], midpoint[1]],
        width=width,
        orientation=orientation,
    )
    D.add_port(
        name=2,
        midpoint=[midpoint[0], midpoint[1]],
        width=width,
        orientation=orientation - 180,
    )
    return D


# ==============================================================================
#
# Contact pads
#
# ==============================================================================


def compass(size=(4, 2), layer=0):
    """Creates a rectangular contact pad with centered ports on edges of the
    rectangle (north, south, east, and west).

    Parameters
    ----------
    size : array_like[2]
        Dimensions of the rectangular contact pad.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing a rectangular contact pad with centered ports.
    """
    D = Device(name="compass")
    r = D.add_ref(rectangle(size, layer=layer))
    r.center = (0, 0)

    dx = size[0]
    dy = size[1]
    D.add_port(name="N", midpoint=[0, dy / 2], width=dx, orientation=90)
    D.add_port(name="S", midpoint=[0, -dy / 2], width=dx, orientation=-90)
    D.add_port(name="E", midpoint=[dx / 2, 0], width=dy, orientation=0)
    D.add_port(name="W", midpoint=[-dx / 2, 0], width=dy, orientation=180)

    return D


def compass_multi(size=(4, 2), ports={"N": 3, "S": 4}, layer=0):
    """Creates a rectangular contact pad with multiple ports along the edges
    rectangle (north, south, east, and west).

    Parameters
    ----------
    size : array_like
        Dimensions of the rectangular contact pad.
    ports : dict
        Number of ports on each edge of the rectangle.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing a rectangular contact pad with multiple ports.
    """
    D = Device(name="compass_multi")
    r = D.add_ref(rectangle(size, layer=layer))
    r.center = (0, 0)

    dx = size[0] / 2
    dy = size[1] / 2

    if "N" in ports:
        num_ports = ports["N"]
        m = dx - dx / num_ports
        p_list = np.linspace(-m, m, num_ports)
        [
            D.add_port(
                name=("N%s" % (n + 1)),
                midpoint=[p, dy],
                width=dx / num_ports * 2,
                orientation=90,
            )
            for n, p in enumerate(p_list)
        ]
    if "S" in ports:
        num_ports = ports["S"]
        m = dx - dx / num_ports
        p_list = np.linspace(-m, m, num_ports)
        [
            D.add_port(
                name=("S%s" % (n + 1)),
                midpoint=[p, -dy],
                width=dx / num_ports * 2,
                orientation=-90,
            )
            for n, p in enumerate(p_list)
        ]
    if "E" in ports:
        num_ports = ports["E"]
        m = dy - dy / num_ports
        p_list = np.linspace(-m, m, num_ports)
        [
            D.add_port(
                name=("E%s" % (n + 1)),
                midpoint=[dx, p],
                width=dy / num_ports * 2,
                orientation=0,
            )
            for n, p in enumerate(p_list)
        ]
    if "W" in ports:
        num_ports = ports["W"]
        m = dy - dy / num_ports
        p_list = np.linspace(-m, m, num_ports)
        [
            D.add_port(
                name=("W%s" % (n + 1)),
                midpoint=[-dx, p],
                width=dy / num_ports * 2,
                orientation=180,
            )
            for n, p in enumerate(p_list)
        ]

    return D


# TODO: Fix the fillet here, right now only goes halfway down
def flagpole(size=(4, 2), stub_size=(2, 1), shape="p", taper_type="straight", layer=0):
    """Creates a flagpole geometry of one of four configurations, all
    involving a vertical central column and a outward-pointing flag.

    Parameters
    ----------
    size : array-like
        (width, height) of the flag.
    stub_size : array-like
        (width, height) of the pole stub.
    shape : {'p', 'q', 'b', 'd'}
        Configuration of the flagpole, where the curved portion of the
        letters represents the flag and the straight portion the pole.
    taper_type : {'straight', 'fillet', None}
        Type of taper between the bottom corner of the stub on the side of
        the flag and the corner of the flag closest to the stub.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing a flagpole geometry.
    """
    f = np.array(size)
    p = np.array(stub_size)
    shape = shape.lower()

    assert shape in "pqbd", "[DEVICE]  flagpole() shape must be p, q, b, or d"
    assert taper_type in ["straight", "fillet", None], (
        '[DEVICE]  flagpole() taper_type must "straight" ' ' or "fillet" or None'
    )

    if shape == "p":
        orientation = -90
    elif shape == "q":
        f[0], p[0] = -size[0], -stub_size[0]
        orientation = -90
    elif shape == "b":
        f[1], p[1] = -size[1], -stub_size[1]
        orientation = 90
    elif shape == "d":
        f[1], p[1] = -size[1], -stub_size[1]
        f[0], p[0] = -size[0], -stub_size[0]
        orientation = 90
    xpts = [0, 0, f[0], f[0], p[0], p[0], 0]
    ypts = [0, f[1], f[1], 0, 0, -p[1], -p[1]]

    D = Device(name="flagpole")
    pad_poly = D.add_polygon([xpts, ypts], layer=layer)
    if taper_type == "fillet":
        taper_amount = min([abs(f[0] - p[0]), abs(p[1])])
        pad_poly.fillet([0, 0, 0, 0, taper_amount, 0, 0])
    elif taper_type == "straight":
        D.add_polygon([xpts[3:6], ypts[3:6]], layer=layer)

    D.add_port(
        name=1, midpoint=[p[0] / 2, -p[1]], width=abs(p[0]), orientation=orientation
    )
    D.add_port(
        name=2,
        midpoint=[f[0] / 2, f[1]],
        width=abs(f[0]),
        orientation=orientation - 180,
    )
    return D


def tee(size=(4, 2), stub_size=(2, 1), taper_type=None, layer=0):
    """Creates a T-shaped geometry.

    Parameters
    ----------
    size : array-like
        (width, height) of the horizontal top part of the T shape.
    stub_size : array-like
        (width, height) of the vertical stub part of the T shape.
    taper_type : {'straight', 'fillet', None}
        If specified, the type of taper between the bottom corners of the stub
        and the bottom corners of the T shape.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing a T-shaped geometry.
    """
    f = np.array(size)
    p = np.array(stub_size)

    xpts = np.array([f[0], f[0], p[0], p[0], -p[0], -p[0], -f[0], -f[0]]) / 2
    ypts = [f[1], 0, 0, -p[1], -p[1], 0, 0, f[1]]

    D = Device(name="tee")
    pad_poly = D.add_polygon([xpts, ypts], layer=layer)
    if taper_type == "fillet":
        taper_amount = min([abs(f[0] - p[0]), abs(p[1])])
        pad_poly.fillet([0, 0, taper_amount, 0, 0, taper_amount, 0, 0])
    elif taper_type == "straight":
        D.add_polygon([xpts[1:4], ypts[1:4]], layer=layer)  # taper_poly1
        D.add_polygon([xpts[4:7], ypts[4:7]], layer=layer)  # taper_poly2

    D.add_port(name=1, midpoint=[f[0] / 2, f[1] / 2], width=f[1], orientation=0)
    D.add_port(name=2, midpoint=[-f[0] / 2, f[1] / 2], width=f[1], orientation=180)
    D.add_port(name=3, midpoint=[0, -p[1]], width=p[0], orientation=-90)
    return D


# ==============================================================================
# Example code
# ==============================================================================

# cp = compass(size = [4,2])
# quickplot(cp)


# cpm = compass_multi(size = [40,20], ports = {'N':3,'S':4, 'E':1, 'W':8}, layer = 0)
# quickplot(cpm)


# cpm = compass_multi(size = [40,20], ports = {'N':3,'S':4, 'E':1, 'W':8}, layer = 0)
# inset_polygon = offset(cpm, distance = -2, layer = 1)
# cpm.add_polygon(inset_polygon)
# quickplot(cpm)

# fp = flagpole(size = [4,2], stub_size = [2,1], shape = 'p', taper_type = 'straight', layer = 0)
# quickplot(fp)


# tp = tee(size = [4,2], stub_size = [2,1], taper_type = 'fillet', layer = 0)
# quickplot(tp)


# ==============================================================================
#
# Tapers
#
# ==============================================================================


def taper(length=10, width1=5, width2=None, port=None, layer=0):
    """Creates a tapered trapezoid/rectangle geometry.

    Parameters
    ----------
    length : int or float
        Length of the shape.
    width1 : int, float, or None
        Width of end 1 of the taper section (width is equal to the port width
        if Port is not None and width1 is None).
    width2 : int, float, or None
        Width of end 2 of the taper section (width is equal to the port width
        if Port is not None and width2 is None).
    port : Port or None
        Port with which to match the width of the taper ends.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D: Device
        A Device containing a taper geometry.
    """
    if type(port) is Port and width1 is None:
        width1 = port.width
    if width2 is None:
        width2 = width1
    xpts = [0, length, length, 0]
    ypts = [width1 / 2, width2 / 2, -width2 / 2, -width1 / 2]

    D = Device("taper")
    D.add_polygon([xpts, ypts], layer=layer)
    D.add_port(name=1, midpoint=[0, 0], width=width1, orientation=180)
    D.add_port(name=2, midpoint=[length, 0], width=width2, orientation=0)
    if type(port) is Port:
        D.rotate(angle=port.orientation, center=[0, 0])
        D.move(origin=[0, 0], destination=port.midpoint)
    return D


def ramp(length=10, width1=5, width2=8, layer=0):
    """Creates a ramp geometry.

    Parameters
    ----------
    length : int or float
        Length of the ramp section.
    width1 : int or float
        Width of the start of the ramp section.
    width2 : int, float, or None
        Width of the end of the ramp section (if width2 is None, width2
        becomes the same as width1).
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing a ramp geometry.
    """
    if width2 is None:
        width2 = width1
    xpts = [0, length, length, 0]
    ypts = [width1, width2, 0, 0]
    D = Device("ramp")
    D.add_polygon([xpts, ypts], layer=layer)
    D.add_port(name=1, midpoint=[0, width1 / 2], width=width1, orientation=180)
    D.add_port(name=2, midpoint=[length, width2 / 2], width=width2, orientation=0)
    return D


def _microstrip_Z(wire_width, dielectric_thickness, eps_r):
    """Calculates the impedance of a microstrip given the wire width and
    dielectric thickness and constant.

    Parameters
    ----------
    wire_width : int or float
        Width of the conducting strip.
    dielectric_thickness : int or float
        Thickness of the substrate.
    eps_r : int or float
        Dielectric constant of the substrate.

    Returns
    -------
    Z : float
        Impedance of the microstrip.
    eps_eff : float
        Effective dielectric constant of the microstrip.

    Notes
    -----
    Equations taken from [1]_.

    These equations can be further corrected for thick films (Hammerstad Eqs.
    6-9) and also for frequency since microstrips are dispersive (Hammerstad
    Eqs. 10-12)

    References
    ----------
    .. [1] Hammerstad, E., & Jensen, O. (1980). Accurate Models for Microstrip
       Computer-Aided Design.  http://doi.org/10.1109/MWSYM.1980.1124303
    """
    u = wire_width / dielectric_thickness
    eta = 376.73  # Vacuum impedance

    a = (
        1
        + log((u**4 + (u / 52) ** 2) / (u**4 + 0.432)) / 49
        + log(1 + (u / 18.1) ** 3) / 18.7
    )
    b = 0.564 * ((eps_r - 0.9) / (eps_r + 3)) ** 0.053
    F = 6 + (2 * pi - 6) * exp(-((30.666 / u) ** 0.7528))
    eps_eff = 0.5 * (eps_r + 1) + 0.5 * (eps_r - 1) * (1 + 10 / u) ** (-a * b)
    Z = eta / (2 * pi) * log(F / u + sqrt(1 + (2 / u) ** 2)) / sqrt(eps_eff)
    return Z, eps_eff


def _microstrip_LC_per_meter(wire_width, dielectric_thickness, eps_r):
    """Calculates the inductance and capacitance per meter of a microstrip
    given wire width and dielectric thickness and constant.

    Parameters
    ----------
    wire_width : int or float
        Width of the conducting strip.
    dielectric_thickness : int or float
        Thickness of the substrate.
    eps_r : int or float
        Dielectric constant of the substrate.

    Returns
    -------
    L_m : float
        Inductance per meter of the microstrip.
    C_m : float
        Capacitance per meter of the microstrip.

    Notes
    -----
    Equations taken from:
        Hammerstad, E., & Jensen, O. (1980). Accurate Models for Microstrip
        Computer-Aided Design.  http://doi.org/10.1109/MWSYM.1980.1124303

    These equations can be further corrected for thick films (Hammerstad Eqs
    6-9) and also for frequency since microstrips are dispersive (Hammerstad
    Eqs 10-12)
    """
    # Use the fact that v = 1/sqrt(L_m*C_m) = 1/sqrt(eps*mu) and
    # Z = sqrt(L_m/C_m)   [Where L_m is inductance per meter]
    Z, eps_eff = _microstrip_Z(wire_width, dielectric_thickness, eps_r)
    eps0 = 8.854e-12
    mu0 = 4 * pi * 1e-7

    eps = eps_eff * eps0
    mu = mu0
    L_m = sqrt(eps * mu) * Z
    C_m = sqrt(eps * mu) / Z
    return L_m, C_m


def _microstrip_Z_with_Lk(wire_width, dielectric_thickness, eps_r, Lk_per_sq):
    """Calculates the impedance of a microstrip given wire width, dielectric
    thickness and constant, and kinetic inductance per square.

    Parameters
    ----------
    wire_width : int or float
        Width of the conducting strip.
    dielectric_thickness : int or float
        Thickness of the substrate.
    eps_r : int or float
        Dielectric constant of the substrate.
    Lk_per_sq : int or float
        Kinetic inductance per square of the microstrip.

    Returns
    -------
    Z : float
        Impedance of the microstrip.

    Notes
    -----
    Equations taken from:
        Hammerstad, E., & Jensen, O. (1980). Accurate Models for Microstrip
        Computer-Aided Design.  http://doi.org/10.1109/MWSYM.1980.1124303

    These equations can be further corrected for thick films (Hammerstad Eqs
    6-9) and also for frequency since microstrips are dispersive (Hammerstad
    Eqs 10-12)
    """
    # Add a kinetic inductance and recalculate the impedance, be careful
    # to input Lk as a per-meter inductance
    L_m, C_m = _microstrip_LC_per_meter(wire_width, dielectric_thickness, eps_r)
    Lk_m = Lk_per_sq * (1.0 / wire_width)
    Z = sqrt((L_m + Lk_m) / C_m)
    return Z


def _microstrip_v_with_Lk(wire_width, dielectric_thickness, eps_r, Lk_per_sq):
    """Calculates the propagation velocity in a microstrip

    Parameters
    ----------
    wire_width : int or float
        Width of the conducting strip.
    dielectric_thickness : int or float
        Thickness of the substrate.
    eps_r : int or float
        Dielectric constant of the substrate.
    Lk_per_sq : int or float
        Kinetic inductance per square of the microstrip.

    Returns
    -------
    v : float
        Propagation velocity in the microstrip

    Notes
    -----
    Equations taken from:
        Hammerstad, E., & Jensen, O. (1980). Accurate Models for Microstrip
        Computer-Aided Design.  http://doi.org/10.1109/MWSYM.1980.1124303

    These equations can be further corrected for thick films (Hammerstad Eqs
    6-9) and also for frequency since microstrips are dispersive (Hammerstad
    Eqs 10-12)
    """
    L_m, C_m = _microstrip_LC_per_meter(wire_width, dielectric_thickness, eps_r)
    Lk_m = Lk_per_sq * (1.0 / wire_width)
    v = 1 / sqrt((L_m + Lk_m) * C_m)
    return v


def _find_microstrip_wire_width(Z_target, dielectric_thickness, eps_r, Lk_per_sq):
    """Calculates the wire width of a microstrip given a target impedance,
    dielectric thickness and constant, and kinetic inductance per square.

    Parameters
    ----------
    Z_target : int or float
        Target impedance of the microstrip.
    dielectric_thickness : int or float
        Thickness of the substrate.
    eps_r : int or float
        Dielectric constant of the substrate.
    Lk_per_sq : int or float
        Kinetic inductance per square of the microstrip.

    Returns
    -------
    w : float
        Wire width of the microstrip.

    Notes
    -----
    Equations taken from:
        Hammerstad, E., & Jensen, O. (1980). Accurate Models for Microstrip
        Computer-Aided Design.  http://doi.org/10.1109/MWSYM.1980.1124303

    These equations can be further corrected for thick films (Hammerstad Eqs
    6-9) and also for frequency since microstrips are dispersive (Hammerstad
    Eqs 10-12)
    """

    def error_fun(wire_width):
        Z_guessed = _microstrip_Z_with_Lk(
            wire_width, dielectric_thickness, eps_r, Lk_per_sq
        )
        return (Z_guessed - Z_target) ** 2  # The error

    x0 = dielectric_thickness
    try:
        from scipy.optimize import fmin
    except Exception:
        raise ImportError(
            " [PHIDL] To run the microstrip functions you "
            "need scipy, please install it with "
            "`pip install scipy`"
        )
    w = fmin(error_fun, x0, args=(), disp=False)
    return w[0]


def _G_integrand(xip, B):
    """Special function for microstrip calculations"""
    try:
        from scipy.special import iv as besseli
    except Exception:
        """[PHIDL] To run this function you need scipy, please install it with
        pip install scipy"""
    return besseli(0, B * sqrt(1 - xip**2))


def _G(xi, B):
    """Special function for microstrip calculations"""
    try:
        import scipy.integrate
    except Exception:
        raise ImportError(
            " [PHIDL] To run the microstrip functions you "
            "need scipy, please install it with "
            "`pip install scipy`"
        )
    return B / sinh(B) * scipy.integrate.quad(_G_integrand, 0, xi, args=(B))[0]


@device_lru_cache
def hecken_taper(
    length=200,
    B=4.0091,
    dielectric_thickness=0.25,
    eps_r=2,
    Lk_per_sq=250e-12,
    Z1=None,
    Z2=None,
    width1=None,
    width2=None,
    num_pts=100,
    layer=0,
):
    """Creates a Hecken-tapered microstrip.

    Parameters
    ----------
    length : int or float
        Length of the microstrip.
    B : int or float
        Controls the intensity of the taper.
    dielectric_thickness : int or float
        Thickness of the substrate.
    eps_r : int or float
        Dielectric constant of the substrate.
    Lk_per_sq : float
        Kinetic inductance per square of the microstrip.
    Z1 : int, float, or None
        Impedance of the left side region of the microstrip.
    Z2 : int, float, or None
        Impedance of the right side region of the microstrip.
    width1 : int, float, or None
        Width of the left side of the microstrip.
    width2 : int, float, or None
        Width of the right side of the microstrip.
    num_pts : int
        Number of points comprising the curve of the entire microstrip.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing a Hecken-tapered microstrip.
    """
    if width1 is not None:
        Z1 = _microstrip_Z_with_Lk(
            width1 * 1e-6, dielectric_thickness * 1e-6, eps_r, Lk_per_sq
        )
    if width2 is not None:
        Z2 = _microstrip_Z_with_Lk(
            width2 * 1e-6, dielectric_thickness * 1e-6, eps_r, Lk_per_sq
        )
    # Normalized length of the wire [-1 to +1]
    xi_list = np.linspace(-1, 1, num_pts)
    Z = [np.exp(0.5 * log(Z1 * Z2) + 0.5 * log(Z2 / Z1) * _G(xi, B)) for xi in xi_list]
    widths = np.array(
        [
            _find_microstrip_wire_width(
                z, dielectric_thickness * 1e-6, eps_r, Lk_per_sq
            )
            * 1e6
            for z in Z
        ]
    )
    x = (xi_list / 2) * length

    # Compensate for varying speed of light in the microstrip by shortening
    # and lengthening sections according to the speed of light in that section
    v = np.array(
        [
            _microstrip_v_with_Lk(
                w * 1e-6, dielectric_thickness * 1e-6, eps_r, Lk_per_sq
            )
            for w in widths
        ]
    )
    dx = np.diff(x)
    dx_compensated = dx * v[:-1]
    x_compensated = np.cumsum(dx_compensated)
    x = np.hstack([0, x_compensated]) / max(x_compensated) * length

    # Create blank device and add taper polygon
    D = Device("hecken")
    xpts = np.concatenate([x, x[::-1]])
    ypts = np.concatenate([widths / 2, -widths[::-1] / 2])
    D.add_polygon((xpts, ypts), layer=layer)
    D.add_port(name=1, midpoint=(0, 0), width=widths[0], orientation=180)
    D.add_port(name=2, midpoint=(length, 0), width=widths[-1], orientation=0)

    # Add meta information about the taper
    D.info["num_squares"] = np.sum(np.diff(x) / widths[:-1])
    D.info["width1"] = widths[0]
    D.info["width2"] = widths[-1]
    D.info["Z1"] = Z[0]
    D.info["Z2"] = Z[-1]
    # Note there are two values for v/c (and f_cutoff) because the speed of
    # light is different at the beginning and end of the taper
    D.info["w"] = widths
    D.info["x"] = x
    D.info["Z"] = Z
    D.info["v/c"] = v / 3e8
    D.info["time_length"] = np.sum(
        np.diff(D.info["x"] * 1e-6) / (D.info["v/c"][:-1] * 3e8)
    )
    D.info["f_cutoff"] = 1 / (2 * D.info["time_length"])
    D.info["length"] = length

    return D


@device_lru_cache
def meander_taper(
    x_taper, w_taper, meander_length=1000, spacing_factor=3, min_spacing=0.5, layer=0
):
    """Takes in an array of x-positions and a array of widths (corresponding to
    each x-position) and creates a meander.  Typically used for creating
    meandered tapers

    Parameters
    ----------
    x_taper : array-like[N]
        The x-coordinates of the data points, must be increasing.
    w_taper : array-like[N]
        The y-coordinates of the data points, same length as ``x_taper``.
    meander_length : int or float
        Length of each section of the meander
    spacing_factor : int or float
        Multiplicative spacing factor between adjacent meanders
    min_spacing : int or float
        Minimum spacing between adjacent meanders

    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device

    """

    def taper_width(x):
        return np.interp(x, x_taper, w_taper)

    def taper_section(x_start, x_end, num_pts=30, layer=0):
        D = Device("tapersec")
        length = x_end - x_start
        x = np.linspace(0, length, num_pts)
        widths = np.linspace(taper_width(x_start), taper_width(x_end), num_pts)
        xpts = np.concatenate([x, x[::-1]])
        ypts = np.concatenate([widths / 2, -widths[::-1] / 2])
        D.add_polygon((xpts, ypts), layer=layer)
        D.add_port(name=1, midpoint=(0, 0), width=widths[0], orientation=180)
        D.add_port(name=2, midpoint=(length, 0), width=widths[-1], orientation=0)
        return D

    def arc_tapered(
        radius=10, width1=1, width2=2, theta=45, angle_resolution=2.5, layer=0
    ):
        D = Device("arctaper")
        path1 = gdspy.Path(width=width1, initial_point=(0, 0))
        path1.turn(
            radius=radius,
            angle=theta * pi / 180,
            number_of_points=int(abs(2 * theta / angle_resolution)),
            final_width=width2,
        )
        [D.add_polygon(p, layer=layer) for p in path1.polygons]
        D.add_port(name=1, midpoint=(0, 0), width=width1, orientation=180)
        D.add_port(
            name=2,
            midpoint=(path1.x, path1.y),
            width=width2,
            orientation=path1.direction * 180 / pi,
        )
        return D

    D = Device("meander-taper")
    xpos1 = min(x_taper)
    xpos2 = min(x_taper) + meander_length
    t = D.add_ref(taper_section(x_start=xpos1, x_end=xpos2, num_pts=50, layer=layer))
    D.add_port(t.ports[1])
    dir_toggle = -1
    while xpos2 < max(x_taper):
        arc_width1 = taper_width(xpos2)
        arc_radius = max(spacing_factor * arc_width1, min_spacing)
        arc_length = pi * arc_radius
        arc_width2 = taper_width(xpos2 + arc_length)
        A = arc_tapered(
            radius=arc_radius,
            width1=arc_width1,
            width2=arc_width2,
            theta=180 * dir_toggle,
            layer=layer,
        )
        a = D.add_ref(A)
        a.connect(port=1, destination=t.ports[2])
        dir_toggle = -dir_toggle
        xpos1 = xpos2 + arc_length
        xpos2 = xpos1 + meander_length
        t = D.add_ref(
            taper_section(x_start=xpos1, x_end=xpos2, num_pts=30, layer=layer)
        )
        t.connect(port=1, destination=a.ports[2])
    D.add_port(t.ports[2])

    return D


# ==============================================================================
# Example code
# ==============================================================================

# D = racetrack_gradual(width, R = 5, N=3)
# quickplot(D)

# D = hecken_taper(length = 200, B = 4.0091, dielectric_thickness = 0.25, eps_r = 2,
#                  Lk_per_sq = 250e-12, Z1 = 50, width2 = 0.3,
#                  num_pts = 100, layer = 0)
# quickplot(D)

# t = np.linspace(0,1)
# x,y = _racetrack_gradual_parametric(t, R = 5, N = 3)
# plt.plot(x,y)
# plt.axis('equal')


# ==============================================================================
#
# Text
#
# ==============================================================================


def text(text="abcd", size=10, justify="left", layer=0, font="DEPLOF"):
    """Creates geometries of text

    Parameters
    ----------
    text : str
        Text string to be written.
    size : int or float
        Size of the text
    justify : {'left', 'right', 'center'}
        Justification of the text.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.
    font: str
        Font face to use. Default DEPLOF does not require additional libraries, otherwise
        freetype will be used to load fonts. Font can be given either by name (e.g. "Times New Roman"),
        or by file path. OTF or TTF fonts are supported.

    Returns
    -------
    t : Device
        A Device containing the text geometry.
    """
    t = Device("text")
    xoffset = 0
    yoffset = 0

    face = font
    if face == "DEPLOF":
        scaling = size / 1000

        for line in text.split("\n"):
            l = Device(name="textline")
            for c in line:
                ascii_val = ord(c)
                if c == " ":
                    xoffset += 500 * scaling
                elif (33 <= ascii_val <= 126) or (ascii_val == 181):
                    for poly in _glyph[ascii_val]:
                        xpts = np.array(poly)[:, 0] * scaling
                        ypts = np.array(poly)[:, 1] * scaling
                        l.add_polygon([xpts + xoffset, ypts + yoffset], layer=layer)
                    xoffset += (_width[ascii_val] + _indent[ascii_val]) * scaling
                else:
                    valid_chars = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~µ"
                    warnings.warn(
                        '[PHIDL] text(): Warning, some characters ignored, no geometry for character "%s" with ascii value %s. '
                        "Valid characters: %s"
                        % (chr(ascii_val), ascii_val, valid_chars)
                    )
            t.add_ref(l)
            yoffset -= 1500 * scaling
            xoffset = 0
    else:
        from .font import _get_font_by_file, _get_font_by_name, _get_glyph

        # Load the font
        # If we've passed a valid file, try to load that, otherwise search system fonts
        font = None
        if (face.endswith(".otf") or face.endswith(".ttf")) and os.path.exists(face):
            font = _get_font_by_file(face)
        else:
            try:
                font = _get_font_by_name(face)
            except ValueError:
                pass
        if font is None:
            raise ValueError(
                (
                    '[PHIDL] Failed to find font: "%s". '
                    + "Try specifying the exact (full) path to the .ttf or .otf file. "
                    + "Otherwise, it might be resolved by rebuilding the matplotlib font cache"
                )
                % (face)
            )

        # Render each character
        for line in text.split("\n"):
            l = Device("textline")
            xoffset = 0
            for letter in line:
                letter_dev = Device("letter")
                letter_template, advance_x = _get_glyph(font, letter)
                for poly in letter_template.polygons:
                    letter_dev.add_polygon(poly.polygons, layer=layer)
                ref = l.add_ref(letter_dev)
                ref.move(destination=(xoffset, 0))
                ref.magnification = size
                xoffset += size * advance_x

            ref = t.add_ref(l)
            ref.move(destination=(0, yoffset))
            yoffset -= size

    justify = justify.lower()
    for l in t.references:
        if justify == "left":
            pass
        if justify == "right":
            l.xmax = 0
        if justify == "center":
            l.move(origin=l.center, destination=(0, 0), axis="x")

    t.flatten()
    return t


# ==============================================================================
# Example code
# ==============================================================================

# D = text('the quick brown\n fox jumped over\nthe lazy dog', justify = 'center', size = 800)
# quickplot(D)


# ==============================================================================
#
# Wafer and die
#
# ==============================================================================


def basic_die(
    size=(10000, 10000),
    street_width=100,
    street_length=1000,
    die_name="chip99",
    text_size=100,
    text_location="SW",
    layer=0,
    draw_bbox=True,
    bbox_layer=99,
):
    """Creates a basic chip/die template, with 4 right angle corners marking
    the boundary of the chip/die and a label with the name of the die.

    Parameters
    ----------
    size : array_like[2] of int or float
        x, y dimensions of the die.
    street_width : int or float
        Width of the corner marks for die-sawing.
    street_length : int or float
        Length of the corner marks for die-sawing.
    die_name : str
        Label text.
    text_size : int or float
        Label text size.
    text_location : {'NW', 'N', 'NE', 'SW', 'S', 'SE'}
        Label text compass location.
    layer : int
        Specific layer(s) to put polygon geometry on.
    draw_bbox : bool
        If true, drawns a bbox around the chip die geometry.
    bbox_layer : int
        Layer on which bbox is placed if ``draw_bbox`` is true.

    Returns
    -------
    D : Device
        A Device containing a basic die geometry.
    """
    D = Device(name="die")
    sx, sy = size[0] / 2, size[1] / 2
    xpts = np.array(
        [
            sx,
            sx,
            sx - street_width,
            sx - street_width,
            sx - street_length,
            sx - street_length,
        ]
    )
    ypts = np.array(
        [
            sy,
            sy - street_length,
            sy - street_length,
            sy - street_width,
            sy - street_width,
            sy,
        ]
    )
    D.add_polygon([xpts, ypts], layer=layer)
    D.add_polygon([-xpts, ypts], layer=layer)
    D.add_polygon([xpts, -ypts], layer=layer)
    D.add_polygon([-xpts, -ypts], layer=layer)

    if draw_bbox is True:
        D.add_polygon([[sx, sy], [sx, -sy], [-sx, -sy], [-sx, sy]], layer=bbox_layer)
    D.center = (0, 0)
    t = D.add_ref(text(text=die_name, size=text_size, layer=layer))

    d = street_width + 20
    if type(text_location) is str:
        text_location = text_location.upper()
        if text_location == "NW":
            t.xmin, t.ymax = [-sx + d, sy - d]
        elif text_location == "N":
            t.x, t.ymax = [0, sy - d]
        elif text_location == "NE":
            t.xmax, t.ymax = [sx - d, sy - d]
        if text_location == "SW":
            t.xmin, t.ymin = [-sx + d, -sy + d]
        elif text_location == "S":
            t.x, t.ymin = [0, -sy + d]
        elif text_location == "SE":
            t.xmax, t.ymin = [sx - d, -sy + d]
    else:
        t.x, t.y = text_location

    return D


# ==============================================================================
# Example code
# ==============================================================================

# D = basic_die(size = (10000, 10000), street_width = 100, street_length = 1000,
#               die_name = 'chip99', text_size = 300, text_location = 'SW',  layer = 0,
#               draw_bbox = True,  bbox_layer = 99)
# quickplot(D)


# ==============================================================================
#
# Waveguide curves
#
# ==============================================================================


def racetrack_gradual(width=0.3, R=5, N=3, layer=0):
    """Creates a gradual racetrack bent geometry.

    Parameters
    ----------
    width : int or float
        Width of the track.
    R : int or float
        Radius of the track at its most curved point.
    N : int or float
        Radius of the track at its least curved point.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing a gradual racetrack bent geometry.
    """
    curve_fun = lambda t: _racetrack_gradual_parametric(t, R=5, N=3)
    route_path = gdspy.Path(width=width, initial_point=[0, 0])
    route_path.parametric(
        curve_fun,
        number_of_evaluations=99,
        max_points=4000,
        final_distance=None,
        layer=layer,
    )
    D = Device("racetrack")
    D.add(route_path)
    return D


def _racetrack_gradual_parametric(t, R, N):
    """Takes in a parametric value ``t`` on (0,1), returns the x,y coordinates
    of a racetrack bent according to
    20090810_EOS4_modulator_designs_excerptForJasonGradualBends.ppt

    Parameters
    ----------
    t : array-like[N]
        Parametric values for the racetrack.
    R : int or float
        Radius of the track at its most curved point.
    N : int or float
        Radius of the track at its least curved point.

    Returns
    -------
    x : array-like[N]
        x-coordinates of the racetrack curve.
    y : array-like[N]
        y-coordinates of the racetrack curve.
    """
    x0 = R / 2 ** (1 / N)
    Rmin = 2 ** (0.5 - 1 / N) / (N - 1) * R
    R0 = R - (x0 - Rmin / sqrt(2))
    t = np.array(t)
    x, y = np.zeros(t.shape), np.zeros(t.shape)

    # Doing the math
    x = cos(t * pi / 2) * R0  # t (0-1) while x (0 to R0)
    ii = (Rmin / sqrt(2) < x) & (x <= R0)
    jj = (0 < x) & (x <= Rmin / sqrt(2))
    y[ii] = (R**N - (x[ii] + (x0 - Rmin / sqrt(2))) ** N) ** (1 / N)
    y[jj] = (x0 - Rmin / sqrt(2)) + sqrt(Rmin**2 - x[jj] ** 2)
    return x, y


# ==============================================================================
# Example code
# ==============================================================================

# D = racetrack_gradual(width = 0.3, R = 5, N = 3)
# quickplot(D)


# t = np.linspace(0,1)
# x,y = _racetrack_gradual_parametric(t, R = 5, N = 3)
# plt.plot(x,y)
# plt.axis('equal')


# ==============================================================================
#
# Arange, packing and fill
#
# ==============================================================================


def grid(
    device_list,
    spacing=(5, 10),
    separation=True,
    shape=None,
    align_x="x",
    align_y="y",
    edge_x="x",
    edge_y="ymax",
):
    """Places the devices in the `device_list` (1D or 2D) on a grid.

    Parameters
    ----------
    device_list : array-like[N] of Device
        Devices to be placed onto a grid.
    spacing : int, float, or array-like[2] of int or float
        Spacing between adjacent elements on the grid, can be a tuple for
        different distances in width and height (x,y).
    separation : bool
        If True, guarantees elements are speparated with a fixed spacing
        between; if  False, elements are spaced evenly along a grid.
    shape : array-like[2]
        x, y shape of the grid (see np.reshape). If no shape is given and the
        list is 1D, the output is as if np.reshape were run with (1, -1).
    align_x : {'x', 'xmin', 'xmax'}
        Which edge to perform the x (column) alignment along
    align_y : {'y', 'ymin', 'ymax'}
        Which edge to perform the y (row) alignment along
    edge_x : {'x', 'xmin', 'xmax'}
        Which edge to perform the x (column) distribution along (unused if
        separation == True)
    edge_y : {'y', 'ymin', 'ymax'}
        Which edge to perform the y (row) distribution along (unused if
        separation == True)

    Returns
    -------
    device_matrix : Device
        A Device containing all the Devices in `device_list` in a grid.
    """

    # Change (y,x) shape to (x,y) shape
    shape = shape[::-1]
    device_array = np.asarray(device_list)
    # Check arguments
    if device_array.ndim not in (1, 2):
        raise ValueError("[PHIDL] grid() The device_list needs to be 1D or 2D.")
    if shape is not None and len(shape) != 2:
        raise ValueError(
            "[PHIDL] grid() shape argument must be None or"
            + " have a length of 2, for example shape=(4,6)"
        )

    # Check that shape is valid and reshape array if needed
    if (shape is None) and (device_array.ndim == 2):  # Already in desired shape
        shape = device_array.shape
    elif (shape is None) and (device_array.ndim == 1):
        shape = (device_array.size, -1)
    elif 0 < shape[0] * shape[1] < device_array.size:
        raise ValueError(
            "[PHIDL] grid() The shape is too small for all the items in device_list"
        )
    else:
        if np.min(shape) == -1:
            max_shape = np.max(shape)
            min_devices = int(np.ceil(device_array.size / max_shape) * max_shape)
            extra_devices = min_devices - device_array.size
        else:
            extra_devices = shape[0] * shape[1] - device_array.size
        if extra_devices != 0:
            device_array = np.append(
                device_array,
                [
                    None,
                ]
                * extra_devices,
            )
    device_array = np.reshape(device_array, shape)

    # Create a blank Device and reference all the Devices in it
    D = Device("grid")
    ref_array = np.empty(device_array.shape, dtype=np.object)
    dummy = Device()
    for idx, d in np.ndenumerate(device_array):
        if d is not None:
            ref_array[idx] = D << d
        else:
            ref_array[idx] = D << dummy  # Create dummy devices

    rows = [Group(ref_array[n, :]) for n in range(ref_array.shape[0])]
    cols = [Group(ref_array[:, n]) for n in range(ref_array.shape[1])]

    # Align rows and columns independently
    for n, r in enumerate(rows):
        r.align(alignment=align_y)
    for n, c in enumerate(cols):
        c.align(alignment=align_x)

    # Distribute rows and columns
    Group(cols).distribute(
        direction="x", spacing=spacing[0], separation=separation, edge=edge_x
    )
    Group(rows[::-1]).distribute(
        direction="y", spacing=spacing[1], separation=separation, edge=edge_y
    )

    # return device_matrix
    return D


def _parameter_combinations(parameters_dict):
    """Creates parameter combinations from a dict filled with list values, e.g.
    parameters_dict = {
        'width' : [0.1,0.2],
        'length' : [4,5,6],
        'height' : [22]
        }
    Will produce a list of dictionaries, each of which can be used as kwargs input:
        [{'height': 22, 'length': 4, 'width': 0.1},
         {'height': 22, 'length': 5, 'width': 0.1},
         {'height': 22, 'length': 6, 'width': 0.1},
         {'height': 22, 'length': 4, 'width': 0.2},
         {'height': 22, 'length': 5, 'width': 0.2},
         {'height': 22, 'length': 6, 'width': 0.2}]"""
    value_combinations = list(itertools.product(*parameters_dict.values()))
    keys = list(parameters_dict.keys())
    return [
        {keys[n]: values[n] for n in range(len(keys))} for values in value_combinations
    ]


def _gen_param_variations(
    function, param_variations, param_defaults={}, param_override={}, label_layer=255
):
    """Takes e.g.

    param_variations = {
            'channel_width' : [1,2,3]
            'gate_width' : [0.1,0.2,0.4],
            }
       or equivalently
    param_variations = dict(
            channel_width = [1,2,3],
            gate_width = [0.1,0.2,0.4],
            )
    """
    parameter_list = _parameter_combinations(param_variations)

    D_list = []
    for params in parameter_list:
        new_params = dict()
        new_params.update(params)
        new_params.update(param_override)
        D_new = make_device(function, config=param_defaults, **new_params)
        label_text = ""
        for name, value in params.items():
            label_text += ("{}={}".format(name, value)) + "\n"
        if label_layer is not None:
            D_new.add_label(text=label_text, position=D_new.center, layer=label_layer)

        D_list.append(D_new)
    return D_list


def gridsweep(
    function,
    param_x={"width": [1, 5, 6, 7]},
    param_y={"length": [1.1, 2, 70]},
    param_defaults={},
    param_override={},
    spacing=(50, 100),
    separation=True,
    align_x="x",
    align_y="y",
    edge_x="x",
    edge_y="ymin",
    label_layer=255,
):
    """Creates a parameter sweep of devices and places them on a grid with
    labels for each device. For instance, given a function defined like
    `myshape(width, height, offset, layer)` and the following parameters:
        param_x = {'width' :  [4, 5, 6]  },
        param_y = {'height' : [7, 9],
                   'layer' :  [1, 2, 3]  },
        param_defaults = {'scale' : 1}

    gridsweep() produce a grid of devices with the following layout/parameters:
        (w=4, h=7, s=1, l=1)    (w=5, h=7, s=1, l=1)    (w=6, h=7, s=1, l=1)
        (w=4, h=7, s=1, l=2)    (w=5, h=7, s=1, l=2)    (w=6, h=7, s=1, l=2)
        (w=4, h=7, s=1, l=3)    (w=5, h=7, s=1, l=3)    (w=6, h=7, s=1, l=3)
        (w=4, h=9, s=1, l=1)    (w=5, h=9, s=1, l=1)    (w=6, h=9, s=1, l=1)
        (w=4, h=9, s=1, l=2)    (w=5, h=9, s=1, l=2)    (w=6, h=9, s=1, l=2)
        (w=4, h=9, s=1, l=3)    (w=5, h=9, s=1, l=3)    (w=6, h=9, s=1, l=3)


    Parameters
    ----------
    function : function that produces a Device
        The function which will be used to create the individual devices in the
        grid.  Must only return a single Device (e.g. any of the functions in
        pg.geometry)
    param_x : dict
        A dictionary of one or more parameters to sweep in the x-direction
    param_y : dict
        A dictionary of one or more parameters to sweep in the y-direction
    param_defaults : dict
        Default parameters to pass to every device in the grid
    param_override : dict
        Parameters that will override `param_defaults`, equivalent to changing
        param_defaults (useful )
    spacing : int, float, or array-like[2] of int or float
        Spacing between adjacent elements on the grid, can be a tuple for
        different distances in width and height (x,y).
    separation : bool
        If True, guarantees elements are separated with a fixed spacing
        between; if False, elements are spaced evenly along a grid.
    shape : array-like[2]
        x, y shape of the grid (see np.reshape). If no shape is given and the
        list is 1D, the output is as if np.reshape were run with (1, -1).
    align_x : {'x', 'xmin', 'xmax'}
        Which edge to perform the x (column) alignment along
    align_y : {'y', 'ymin', 'ymax'}
        Which edge to perform the y (row) alignment along
    edge_x : {'x', 'xmin', 'xmax'}
        Which edge to perform the x (column) distribution along (unused if
        separation == True)
    edge_y : {'y', 'ymin', 'ymax'}
        Which edge to perform the y (row) distribution along (unused if
        separation == True)
    label_layer : None or layer
        If not None, will place a label that describes the parameters on the device

    Returns
    -------
    device_matrix : Device
        A Device containing all the Devices in `device_list` in a grid.
    """

    param_variations = OrderedDict()
    param_variations.update(param_y)
    param_variations.update(param_x)

    D_list = _gen_param_variations(
        function=function,
        param_variations=param_variations,
        param_defaults=param_defaults,
        param_override=param_override,
        label_layer=label_layer,
    )

    num_x_parameters = len(_parameter_combinations(param_x))
    num_y_parameters = len(_parameter_combinations(param_y))
    D = grid(
        D_list,
        spacing=spacing,
        separation=separation,
        shape=(num_x_parameters, num_y_parameters),
        align_x=align_x,
        align_y=align_y,
        edge_x=edge_x,
        edge_y=edge_y,
    )

    label_text = {}
    label_text.update(param_defaults)
    label_text.update(param_override)
    if label_layer is not None:
        D.add_label(text=str(label_text), position=(D.xmin, D.ymin), layer=label_layer)

    return D


def _pack_single_bin(
    rect_dict, aspect_ratio, max_size, sort_by_area, density, precision, verbose
):
    """Takes a `rect_dict` argument of the form {id:(w, h)} and tries to
    pack it into a bin as small as possible with aspect ratio `aspect_ratio`.
    Will iteratively grow the bin size until everything fits or the bin size
    reaches `max_size`. Returns a dictionary of the packed rectanglesn in the
    form {id:(x, y, w, h)}, and a dictionary of remaining unpacked rects.
    """
    try:
        import rectpack
    except Exception:
        raise ImportError(
            "[PHIDL] The packer() function requires the "
            'module "rectpack" to operate.  Please retry '
            "after installing rectpack:\n\n"
            "$ pip install rectpack"
        )

    # Compute total area and use it for an initial estimate of the bin size
    total_area = 0
    for r in rect_dict.values():
        total_area += r[0] * r[1]
    # Normalize
    aspect_ratio = np.asarray(aspect_ratio) / np.linalg.norm(aspect_ratio)

    # Setup variables
    box_size = np.asarray(aspect_ratio * np.sqrt(total_area), dtype=np.float64)
    box_size = np.clip(box_size, None, max_size)
    if sort_by_area:
        rp_sort = rectpack.SORT_AREA
    else:
        rp_sort = rectpack.SORT_NONE

    # Repeatedly run the rectangle-packing algorithm with increasingly larger
    # areas until everything fits or we've reached the maximum size
    while True:
        # Create the packer object
        rect_packer = rectpack.newPacker(
            mode=rectpack.PackingMode.Offline,
            pack_algo=rectpack.MaxRectsBlsf,
            sort_algo=rp_sort,
            bin_algo=rectpack.PackingBin.BBF,
            rotation=False,
        )

        # Add each rectangle to the packer, create a single bin, and pack
        for rid, r in rect_dict.items():
            rect_packer.add_rect(width=r[0], height=r[1], rid=rid)
        rect_packer.add_bin(width=box_size[0], height=box_size[1])
        rect_packer.pack()

        # Adjust the box size for next time
        box_size *= density  # Increase area to try to fit
        box_size = np.clip(box_size, None, max_size)
        if verbose:
            print(
                "Trying to pack in bin size "
                "(%0.2f, %0.2f)" % tuple(box_size * precision)
            )

        # Quit the loop if we've packed all the rectangles
        # or reached the max size
        if len(rect_packer.rect_list()) == len(rect_dict):
            if verbose:
                print("Success!")
            break
        elif all(box_size >= max_size):
            if verbose:
                print("Reached max_size, creating " "an additional bin")
            break

    # Separate packed from unpacked rectangles, make dicts of form
    # {id:(x,y,w,h)}
    packed_rect_dict = {r[-1]: r[:-1] for r in rect_packer[0].rect_list()}
    unpacked_rect_dict = {}
    for k, v in rect_dict.items():
        if k not in packed_rect_dict:
            unpacked_rect_dict[k] = v

    return (packed_rect_dict, unpacked_rect_dict)


def packer(
    D_list,
    spacing=10,
    aspect_ratio=(1, 1),
    max_size=(None, None),
    sort_by_area=True,
    density=1.1,
    precision=1e-2,
    verbose=False,
):
    """Packs geometries together into rectangular bins.

    Parameters
    ----------
    D_list : array-like of Devices
        Input Devices to be packed.
    spacing : int or float
        The minimum distance between adjacent shapes.
    aspect_ratio : array-like
        The (width, height) ratio of the rectangular bin.
    max_size : array-like
        Limits the size into which the shapes will be packed.
    sort_by_area : bool
        If true, pre-sorts the shapes by area.
    density : int or float
        Density of packing. Values closer to 1 pack tighter but require more
        computation.
    precision : float
        Desired precision for rounding vertex coordinates.
    verbose : bool
        Whether to display results of packing attempts


    Returns
    -------
    D_packed_list : Device or list of Devices
        A Device or list of Devices containing all the packed rectangular
        bins generated.

    Notes
    -----
    If a max-size is specified, the function will create as many bins as
    necessary to pack all the geometries and then return a list of the
    filled-bin Devices.
    """
    if density < 1.01:
        raise ValueError(
            "[PHIDL] packer() was given a `density` argument "
            "that is too small.  The density argument must be "
            ">= 1.01"
        )

    # Santize max_size variable
    max_size = [np.inf if v is None else v for v in max_size]
    max_size = np.asarray(max_size, dtype=np.float64)  # In case it's integers
    max_size = max_size / precision

    # Convert Devices to rectangles
    rect_dict = {}
    for n, D in enumerate(D_list):
        w, h = (D.size + spacing) / precision
        w, h = int(w), int(h)
        if (w > max_size[0]) or (h > max_size[1]):
            raise ValueError(
                "[PHIDL] packer() failed because one of "
                "the objects in `D_list` is has an x or "
                "y dimension larger than `max_size` and "
                "so cannot be packed"
            )
        rect_dict[n] = (w, h)

    packed_list = []
    while len(rect_dict) > 0:
        (packed_rect_dict, rect_dict) = _pack_single_bin(
            rect_dict,
            aspect_ratio=aspect_ratio,
            max_size=max_size,
            sort_by_area=sort_by_area,
            density=density,
            precision=precision,
            verbose=verbose,
        )
        packed_list.append(packed_rect_dict)

    D_packed_list = []
    for rect_dict in packed_list:
        D_packed = Device()
        for n, rect in rect_dict.items():
            x, y, w, h = rect
            xcenter = x + w / 2 + spacing / 2
            ycenter = y + h / 2 + spacing / 2
            d = D_packed.add_ref(D_list[n])
            d.center = (xcenter * precision, ycenter * precision)
        D_packed_list.append(D_packed)

    return D_packed_list


def _rasterize_polygons(polygons, bounds=[[-100, -100], [100, 100]], dx=1, dy=1):
    """Converts polygons to a black/white (1/0) matrix"""
    try:
        from skimage import draw
    except Exception:
        raise ImportError(
            "The fill function requires the module "
            '"scikit-image" to operate.  Please retry '
            "after installing scikit-image:\n\n"
            "$ pip install --upgrade scikit-image"
        )

    # Prepare polygon array by shifting all points into the first quadrant and
    # separating points into x and y lists
    xpts = []
    ypts = []
    for p in polygons:
        p_array = np.asarray(p)
        x = p_array[:, 0]
        y = p_array[:, 1]
        xpts.append((x - bounds[0][0]) / dx - 0.5)
        ypts.append((y - bounds[0][1]) / dy - 0.5)

    # Initialize the raster matrix we'll be writing to
    xsize = int(np.ceil(bounds[1][0] - bounds[0][0]) / dx)
    ysize = int(np.ceil(bounds[1][1] - bounds[0][1]) / dy)
    raster = np.zeros((ysize, xsize), dtype=np.bool)

    # TODO: Replace polygon_perimeter with the supercover version
    for n in range(len(xpts)):
        rr, cc = draw.polygon(ypts[n], xpts[n], shape=raster.shape)
        rrp, ccp = draw.polygon_perimeter(
            ypts[n], xpts[n], shape=raster.shape, clip=False
        )
        raster[rr, cc] = 1
        raster[rrp, ccp] = 1

    return raster


def _raster_index_to_coords(i, j, bounds=[[-100, -100], [100, 100]], dx=1, dy=1):
    """Converts (i,j) index of raster matrix to real coordinates"""
    x = (j + 0.5) * dx + bounds[0][0]
    y = (i + 0.5) * dy + bounds[0][1]
    return x, y


def _expand_raster(raster, distance=(4, 2)):
    """Expands all black (1) pixels in the raster"""
    try:
        from skimage import draw, morphology
    except Exception:
        raise ImportError(
            "The fill function requires the module "
            '"scikit-image" to operate.  Please retry '
            "after installing scikit-image:\n\n"
            "$ pip install --upgrade scikit-image"
        )
    if distance[0] <= 0.5 and distance[1] <= 0.5:
        return raster

    num_pixels = np.array(np.ceil(distance), dtype=int)
    neighborhood = np.zeros(
        (num_pixels[1] * 2 + 1, num_pixels[0] * 2 + 1), dtype=np.bool
    )
    rr, cc = draw.ellipse(
        num_pixels[1], num_pixels[0], distance[1] + 0.5, distance[0] + 0.5
    )
    neighborhood[rr, cc] = 1

    return morphology.binary_dilation(image=raster, selem=neighborhood)


def _fill_cell_rectangle(
    size=(20, 20),
    layers=(0, 1, 3),
    densities=(0.5, 0.25, 0.7),
    inverted=(False, False, False),
):
    """Creates a single Device on multiple layers to be used as fill

    Parameters
    ----------
    size : array-like of int or float
        x, y dimensions of the fill area for all layers.
    layers : int, array-like[2], or set
        Specific layer(s) to put fill cell rectangle geometry on.
    densities : array-like of int or float
        Fill densities for each layer specified in ``layers``. Must be the same
        size as ``layers``.
    inverted : array-like or bool
        If true, inverts the fill area for corresponding layer. Must be the
        same size as ``layers``.

    Returns
    -------
    D : Device
        A Device containing filled cell rectangles.
    """
    D = Device("fillcell")
    for layer, density, inv in zip(layers, densities, inverted):
        rectangle_size = np.array(size) * sqrt(density)
        # r = D.add_ref(rectangle(size = rectangle_size, layer = layer))
        R = rectangle(size=rectangle_size, layer=layer)
        R.center = (0, 0)
        if inv is True:
            A = rectangle(size=size)
            A.center = (0, 0)
            A = A.get_polygons()
            B = R.get_polygons()
            p = gdspy.boolean(A, B, operation="not")
            D.add_polygon(p, layer=layer)
        else:
            D.add_ref(R)
    return D


def _loop_over(var):
    """Checks if a variable is in the form of an iterable (list/tuple)
    and if not, returns it as a list.  Useful for allowing argument
    inputs to be either lists (e.g. [1, 3, 4]) or single-valued (e.g. 3).

    Parameters
    ----------
    var : int or float or list
        Variable to check for iterability.

    Returns
    -------
    var : list
        Variable converted to list if single-valued input.
    """

    if hasattr(var, "__iter__"):
        return var
    else:
        return [var]


def fill_rectangle(
    D,
    fill_size=(40, 10),
    avoid_layers="all",
    include_layers=None,
    margin=100,
    fill_layers=(0, 1, 3),
    fill_densities=(0.5, 0.25, 0.7),
    fill_inverted=None,
    bbox=None,
):
    """Creates a rectangular fill pattern and fills all empty areas
    in the input device D

    Parameters
    ----------
    D : Device
        Device to be filled
    fill_size : array-like[2]
        Rectangular size of the fill element
    avoid_layers : 'all' or list of layers
        Layers to be avoided (not filled) in D
    include_layers :
        Layers to be included (filled) in D, supercedes avoid_layers
    margin : int or float
        Margin spacing around avoided areas -- fill will not come within
        `margin` of the geometry in D
    fill_layers : list of layers
        Defines the fill pattern layers
    fill_densities : float between 0 and 1
        Defines the fill pattern density (1.0 == fully filled)
    fill_inverted : bool
        Inverts the fill pattern
    bbox : array-like[2][2]
        Limit the fill pattern to the area defined by this bounding box

    Returns
    -------
    F : Device

    """
    # Create the fill cell.
    # If fill_inverted is not specified, assume all False
    fill_layers = _loop_over(fill_layers)
    fill_densities = _loop_over(fill_densities)
    if fill_inverted is None:
        fill_inverted = [False] * len(fill_layers)
    fill_inverted = _loop_over(fill_inverted)
    if len(fill_layers) != len(fill_densities):
        raise ValueError(
            "[PHIDL] phidl.geometry.fill_rectangle() "
            "`fill_layers` and `fill_densities` parameters "
            "must be lists of the same length"
        )
    if len(fill_layers) != len(fill_inverted):
        raise ValueError(
            "[PHIDL] phidl.geometry.fill_rectangle() "
            "`fill_layers` and `fill_inverted` parameters must "
            "be lists of the same length"
        )

    fill_cell = _fill_cell_rectangle(
        size=fill_size,
        layers=fill_layers,
        densities=fill_densities,
        inverted=fill_inverted,
    )
    F = Device(name="fill_pattern")

    if avoid_layers == "all":
        exclude_polys = D.get_polygons(by_spec=False, depth=None)
    else:
        avoid_layers = [_parse_layer(l) for l in _loop_over(avoid_layers)]
        exclude_polys = D.get_polygons(by_spec=True, depth=None)
        exclude_polys = {
            key: exclude_polys[key] for key in exclude_polys if key in avoid_layers
        }
        exclude_polys = itertools.chain.from_iterable(exclude_polys.values())

    if include_layers is None:
        include_polys = []
    else:
        include_layers = [_parse_layer(l) for l in _loop_over(include_layers)]
        include_polys = D.get_polygons(by_spec=True, depth=None)
        include_polys = {
            key: include_polys[key] for key in include_polys if key in include_layers
        }
        include_polys = itertools.chain.from_iterable(include_polys.values())

    if bbox is None:
        bbox = D.bbox

    raster = _rasterize_polygons(
        polygons=exclude_polys, bounds=bbox, dx=fill_size[0], dy=fill_size[1]
    )
    raster = raster & ~_rasterize_polygons(
        polygons=include_polys, bounds=bbox, dx=fill_size[0], dy=fill_size[1]
    )
    raster = _expand_raster(raster, distance=margin / np.array(fill_size))

    for i in range(np.size(raster, 0)):
        sub_rasters = [list(g) for k, g in itertools.groupby(raster[i])]
        j = 0
        for s in sub_rasters:
            if s[0] == 0:
                x, y = _raster_index_to_coords(i, j, bbox, fill_size[0], fill_size[1])
                # F.add(gdspy.CellArray(ref_cell = fill_cell,
                #                       columns = len(s), rows = 1,
                #                       spacing = fill_size, ))
                a = F.add_array(fill_cell, columns=len(s), rows=1, spacing=fill_size)
                a.move((x, y))
            j += len(s)

    return F


# ==============================================================================
#
# Photonics
#
# ==============================================================================


def polygon_ports(xpts=[-1, -1, 0, 0], ypts=[0, 1, 1, 0], layer=0):
    """Creates a polygon with ports on all edges.

    Parameters
    ----------
    xpts : array-like
        x-coordinate values of the polygon vertices.
    ypts : array-like
        y-coordinate values of the polygon vertices.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    P : Device
        A Device containing a polygon with ports on all edges.
    """
    # returns a polygon with ports on all edges
    P = Device("polygon")
    P.add_polygon([xpts, ypts], layer=layer)
    n = len(xpts)
    xpts.append(xpts[0])
    ypts.append(ypts[0])
    # determine if clockwise or counterclockwise
    cc = 0
    for i in range(0, n):
        cc += (xpts[i + 1] - xpts[i]) * (ypts[i + 1] + ypts[i])

    for i in range(0, n):
        midpoint_n = [(xpts[i + 1] + xpts[i]) / 2, (ypts[i + 1] + ypts[i]) / 2]
        orientation_n = (
            np.arctan2(
                np.sign(cc) * (xpts[i + 1] - xpts[i]),
                np.sign(cc) * (ypts[i] - ypts[i + 1]),
            )
            * 180
            / pi
        )
        width_n = np.abs(
            sqrt((xpts[i + 1] - xpts[i]) ** 2 + (ypts[i + 1] - ypts[i]) ** 2)
        )
        P.add_port(
            name=str(i + 1),
            midpoint=midpoint_n,
            width=width_n,
            orientation=orientation_n,
        )

    return P


# ==============================================================================
# Example code
# ==============================================================================

# P = polygon(xpts=[-1,-3, 0, 0], ypts = [0, 1, 2, 0], layer = 3)
# quickplot(P)


@device_lru_cache
def grating(
    num_periods=20,
    period=0.75,
    fill_factor=0.5,
    width_grating=5,
    length_taper=10,
    width=0.4,
    partial_etch=False,
):
    """Simple grating structure for photonics

    Parameters
    ----------
    num_periods : int
        Number of gratings.
    period : int or float
        Distance between gratings.
    fill_factor : int or float
        Thickness of the gratings.
    width_grating : int or float
        Width of the gratings.
    length_taper : int or float
        Length of the taper section.
    width : int or float
        Width of the end of the taper section.
    partial_etch : bool
        If True, makes an untapered, partially-etched grating.

    Returns
    -------
    G : Device
        A Device containing a fiber grating geometry.
    """
    # returns a fiber grating
    G = Device("grating")

    # make the deep etched grating
    if partial_etch is False:
        # make the grating teeth
        for i in range(num_periods):
            cgrating = G.add_ref(
                compass(size=[period * fill_factor, width_grating], layer=0)
            )
            cgrating.x += i * period

        # make the taper
        tgrating = G.add_ref(
            taper(
                length=length_taper,
                width1=width_grating,
                width2=width,
                port=None,
                layer=0,
            )
        )
        tgrating.xmin = cgrating.xmax
        # define the port of the grating
        p = G.add_port(port=tgrating.ports[2], name=1)
    # make a partially etched grating
    if partial_etch is True:
        # hard coded overlap
        partetch_overhang = 5
        # make the etched areas (opposite to teeth)
        for i in range(num_periods):
            cgrating = G.add_ref(
                compass(
                    size=[
                        period * (1 - fill_factor),
                        width_grating + partetch_overhang * 2,
                    ],
                    layer=1,
                )
            )
            cgrating.x += i * period
        # define the port of the grating
        p = G.add_port(port=cgrating.ports["E"], name=1)
        p.midpoint = p.midpoint + np.array([(1 - fill_factor) * period, 0])

        # draw the deep etched square around the grating
        G.add_ref(  # deepbox
            compass(size=[num_periods * period, width_grating], layer=0)
        )
    return G


# ==============================================================================
# Example code
# ==============================================================================

# G = grating(num_periods = 20, period = 0.75, fill_factor = 0.5,
#             width_grating = 20, length_taper = 10, width = 0.4,
#             partial_etch = False)
# quickplot(G)


# ==============================================================================
#
# Test Structures
#
# ==============================================================================

# Via Route ----------------------------------------
def _via_iterable(
    via_spacing, wire_width, wiring1_layer, wiring2_layer, via_layer, via_width
):
    """Helper function for test_via

    Parameters
    ----------
    via_spacing : int or float
        Distance between vias.
    wire_width : int or float
        The width of the wires.
    wiring1_layer : int
        Specific layer to put the top wiring on.
    wiring2_layer : int
        Specific layer to put the bottom wiring on.
    via_layer : int
        Specific layer to put the vias on.
    via_width : int or float
        Diameter of the vias.

    Returns
    -------
    VI : Device

    """
    VI = Device("test_via_iter")
    wire1 = VI.add_ref(compass(size=(via_spacing, wire_width), layer=wiring1_layer))
    wire2 = VI.add_ref(compass(size=(via_spacing, wire_width), layer=wiring2_layer))
    via1 = VI.add_ref(compass(size=(via_width, via_width), layer=via_layer))
    via2 = VI.add_ref(compass(size=(via_width, via_width), layer=via_layer))
    wire1.connect(port="E", destination=wire2.ports["W"], overlap=wire_width)
    via1.connect(
        port="W", destination=wire1.ports["E"], overlap=(wire_width + via_width) / 2
    )
    via2.connect(
        port="W", destination=wire2.ports["E"], overlap=(wire_width + via_width) / 2
    )
    VI.add_port(name="W", port=wire1.ports["W"])
    VI.add_port(name="E", port=wire2.ports["E"])
    VI.add_port(
        name="S",
        midpoint=[(1 * wire_width) + wire_width / 2, -wire_width / 2],
        width=wire_width,
        orientation=-90,
    )
    VI.add_port(
        name="N",
        midpoint=[(1 * wire_width) + wire_width / 2, wire_width / 2],
        width=wire_width,
        orientation=90,
    )

    return VI


def test_via(
    num_vias=100,
    wire_width=10,
    via_width=15,
    via_spacing=40,
    pad_size=(300, 300),
    min_pad_spacing=0,
    pad_layer=0,
    wiring1_layer=1,
    wiring2_layer=2,
    via_layer=3,
):
    """Via chain test structure

    Parameters
    ----------
    num_vias : int
        The total number of requested vias (must be an even number).
    wire_width : int or float
        The width of the wires.
    via_width : int or float
        Diameter of the vias.
    via_spacing : int or float
        Distance between vias.
    pad_size : array-like[2]
        (width, height) of the pads.
    min_pad_spacing : int or float
        Defines the minimum distance between the two pads.
    pad_layer : int
        Specific layer to put the pads on.
    wiring1_layer : int
        Specific layer to put the top wiring on.
    wiring2_layer : int
        Specific layer to put the bottom wiring on.
    via_layer : int
        Specific layer to put the vias on.

    Returns
    -------
    VR : Device
        A Device containing the test via structures.

    Usage
    -----
    Call via_route_test_structure() by indicating the number of vias you want
    drawn. You can also change the other parameters however if you do not
    specifiy a value for a parameter it will just use the default value
    Ex::

        via_route_test_structure(num_vias=54)

    - or -::

        via_route_test_structure(num_vias=12, pad_size=(100,100),wire_width=8)

    ex: via_route(54, min_pad_spacing=300)
    """
    VR = Device("test_via")
    pad1 = VR.add_ref(rectangle(size=pad_size, layer=pad_layer))
    pad1_overlay = VR.add_ref(rectangle(size=pad_size, layer=wiring1_layer))
    pad2 = VR.add_ref(rectangle(size=pad_size, layer=pad_layer))
    pad2_overlay = VR.add_ref(rectangle(size=pad_size, layer=wiring1_layer))
    nub = VR.add_ref(compass(size=(3 * wire_width, wire_width), layer=pad_layer))
    nub_overlay = VR.add_ref(
        compass(size=(3 * wire_width, wire_width), layer=wiring1_layer)
    )
    head = VR.add_ref(compass(size=(wire_width, wire_width), layer=pad_layer))
    head_overlay = VR.add_ref(
        compass(size=(wire_width, wire_width), layer=wiring1_layer)
    )
    nub.ymax = pad1.ymax - 5
    nub.xmin = pad1.xmax
    nub_overlay.ymax = pad1.ymax - 5
    nub_overlay.xmin = pad1.xmax
    head.connect(port="W", destination=nub.ports["E"])
    head_overlay.connect(port="W", destination=nub_overlay.ports["E"])
    pad1_overlay.xmin = pad1.xmin
    pad1_overlay.ymin = pad1.ymin

    old_port = head.ports["S"]
    count = 0
    width_via_iter = 2 * via_spacing - 2 * wire_width

    pad2.xmin = pad1.xmax + min_pad_spacing
    up = False
    down = True
    edge = True
    current_width = 3 * wire_width + wire_width  # width of nub and 1 overlap
    obj_old = head
    obj = head
    via_iterable = _via_iterable(
        via_spacing, wire_width, wiring1_layer, wiring2_layer, via_layer, via_width
    )
    while (count + 2) <= num_vias:
        obj = VR.add_ref(via_iterable)
        obj.connect(port="W", destination=old_port, overlap=wire_width)
        old_port = obj.ports["E"]
        edge = False
        if obj.ymax > pad1.ymax:
            obj.connect(port="W", destination=obj_old.ports["S"], overlap=wire_width)
            old_port = obj.ports["S"]
            current_width += width_via_iter
            down = True
            up = False
            edge = True

        elif obj.ymin < pad1.ymin:
            obj.connect(port="W", destination=obj_old.ports["N"], overlap=wire_width)
            old_port = obj.ports["N"]
            current_width += width_via_iter
            up = True
            down = False
            edge = True
        count = count + 2
        obj_old = obj

    if (
        current_width < min_pad_spacing
        and (min_pad_spacing - current_width) > 3 * wire_width
    ):
        tail = VR.add_ref(
            compass(
                size=(min_pad_spacing - current_width + wire_width, wire_width),
                layer=wiring1_layer,
            )
        )
        tail_overlay = VR.add_ref(
            compass(
                size=(min_pad_spacing - current_width + wire_width, wire_width),
                layer=pad_layer,
            )
        )
    else:
        tail = VR.add_ref(
            compass(size=(3 * wire_width, wire_width), layer=wiring1_layer)
        )
        tail_overlay = VR.add_ref(
            compass(size=(3 * wire_width, wire_width), layer=wiring1_layer)
        )

    if up and not edge:
        tail.connect(port="W", destination=obj.ports["S"], overlap=wire_width)
        tail_overlay.connect(port="W", destination=obj.ports["S"], overlap=wire_width)
    elif down and not edge:
        tail.connect(port="W", destination=obj.ports["N"], overlap=wire_width)
        tail_overlay.connect(port="W", destination=obj.ports["N"], overlap=wire_width)
    else:
        tail.connect(port="W", destination=obj.ports["E"], overlap=wire_width)
        tail_overlay.connect(port="W", destination=obj.ports["E"], overlap=wire_width)

    pad2.xmin = tail.xmax
    pad2_overlay.xmin = pad2.xmin
    pad2_overlay.ymin = pad2.ymin

    return VR


def test_comb(
    pad_size=(200, 200),
    wire_width=1,
    wire_gap=3,
    comb_layer=0,
    overlap_zigzag_layer=1,
    comb_pad_layer=2,
    comb_gnd_layer=3,
    overlap_pad_layer=4,
):
    """Overlap comb test structure for checking whether two layers
    are electrically isolated

    Parameters
    ----------
    pad_size : array-like
        x and y dimensions of the comb pads.
    wire_width : int or float
        Width of the test comb teeth.
    wire_gap : int or float
        Size of the gap between comb teeth.
    comb_layer : int, array-like[2], or set
        Specific layer(s) to put comb geometry on.
    overlap_zigzag_layer : int, array-like[2], or set
        Specific layer(s) to put overlap zigzag geometry on.
    comb_pad_layer : int, array-like[2], or set
        Specific layer(s) to put comb pads on.
    comb_gnd_layer : int, array-like[2], or set
        Specific layer(s) to put the comb ground on.
    overlap_pad_layer : int, array-like[2], or set
        Specific layer(s) to put overlap pads on.

    Returns
    -------
    CI : Device
        A Device containing a test comb geometry.

    Notes
    -----
    Call comb_insulation_test_structure() with any of the parameters shown
    below which you'd like to change. You only need to supply the parameters
    which you intend on changing You can alternatively call it with no
    parameters and it will take all the default values shown below.

    Ex::
        comb_insulation_test_structure(pad_size=(175,175), wire_width=2, wire_gap=5)
    - or -::
        comb_insulation_test_structure()
    """
    CI = Device("test_comb")

    # if comb_pad_layer is None: comb_pad_layer = comb_layer
    # if comb_gnd_layer is None: comb_gnd_layer = comb_layer
    # if overlap_pad_layer is None: overlap_pad_layer = overlap_zigzag_layer
    wire_spacing = wire_width + wire_gap * 2

    # pad overlays
    overlay_padb = CI.add_ref(
        rectangle(
            size=(pad_size[0] * 9 / 10, pad_size[1] * 9 / 10), layer=overlap_pad_layer
        )
    )
    overlay_padl = CI.add_ref(
        rectangle(
            size=(pad_size[0] * 9 / 10, pad_size[1] * 9 / 10), layer=comb_pad_layer
        )
    )
    overlay_padt = CI.add_ref(
        rectangle(
            size=(pad_size[0] * 9 / 10, pad_size[1] * 9 / 10), layer=comb_pad_layer
        )
    )
    overlay_padr = CI.add_ref(
        rectangle(
            size=(pad_size[0] * 9 / 10, pad_size[1] * 9 / 10), layer=comb_gnd_layer
        )
    )
    overlay_padl.xmin = 0
    overlay_padl.ymin = 0
    overlay_padb.ymax = 0
    overlay_padb.xmin = overlay_padl.xmax + pad_size[1] / 5
    overlay_padr.ymin = overlay_padl.ymin
    overlay_padr.xmin = overlay_padb.xmax + pad_size[1] / 5
    overlay_padt.xmin = overlay_padl.xmax + pad_size[1] / 5
    overlay_padt.ymin = overlay_padl.ymax

    # pads
    padl = CI.add_ref(rectangle(size=pad_size, layer=comb_layer))
    padt = CI.add_ref(rectangle(size=pad_size, layer=comb_layer))
    padr = CI.add_ref(rectangle(size=pad_size, layer=comb_layer))
    padb = CI.add_ref(rectangle(size=pad_size, layer=overlap_zigzag_layer))
    padl_nub = CI.add_ref(
        rectangle(size=(pad_size[0] / 4, pad_size[1] / 2), layer=comb_layer)
    )
    padr_nub = CI.add_ref(
        rectangle(size=(pad_size[0] / 4, pad_size[1] / 2), layer=comb_layer)
    )
    padl.xmin = overlay_padl.xmin
    padl.center = [padl.center[0], overlay_padl.center[1]]
    padt.ymax = overlay_padt.ymax
    padt.center = [overlay_padt.center[0], padt.center[1]]
    padr.xmax = overlay_padr.xmax
    padr.center = [padr.center[0], overlay_padr.center[1]]
    padb.ymin = overlay_padb.ymin
    padb.center = [overlay_padb.center[0], padb.center[1]]
    padl_nub.xmin = padl.xmax
    padl_nub.center = [padl_nub.center[0], padl.center[1]]
    padr_nub.xmax = padr.xmin
    padr_nub.center = [padr_nub.center[0], padr.center[1]]

    # connected zig
    head = CI.add_ref(compass(size=(pad_size[0] / 12, wire_width), layer=comb_layer))
    head.xmin = padl_nub.xmax
    head.ymax = padl_nub.ymax
    connector = CI.add_ref(compass(size=(wire_width, wire_width), layer=comb_layer))
    connector.connect(port="W", destination=head.ports["E"])
    old_port = connector.ports["S"]
    top = True
    obj = connector
    while obj.xmax + pad_size[0] / 12 < padr_nub.xmin:
        # long zig zag rectangle
        obj = CI.add_ref(
            compass(
                size=(pad_size[1] / 2 - 2 * wire_width, wire_width), layer=comb_layer
            )
        )
        obj.connect(port="W", destination=old_port)
        old_port = obj.ports["E"]
        if top:
            # zig zag edge rectangle
            obj = CI.add_ref(compass(size=(wire_width, wire_width), layer=comb_layer))
            obj.connect(port="N", destination=old_port)
            top = False
        else:
            # zig zag edge rectangle
            obj = CI.add_ref(compass(size=(wire_width, wire_width), layer=comb_layer))
            obj.connect(port="S", destination=old_port)
            top = True
            # comb rectange
            comb = CI.add_ref(
                rectangle(
                    size=(
                        (padt.ymin - head.ymax)
                        + pad_size[1] / 2
                        - (wire_spacing + wire_width) / 2,
                        wire_width,
                    ),
                    layer=comb_layer,
                )
            )
            comb.rotate(90)
            comb.ymax = padt.ymin
            comb.xmax = obj.xmax - (wire_spacing + wire_width) / 2
        old_port = obj.ports["E"]
        obj = CI.add_ref(compass(size=(wire_spacing, wire_width), layer=comb_layer))
        obj.connect(port="W", destination=old_port)
        old_port = obj.ports["E"]
        obj = CI.add_ref(compass(size=(wire_width, wire_width), layer=comb_layer))
        obj.connect(port="W", destination=old_port)
        if top:
            old_port = obj.ports["S"]
        else:
            old_port = obj.ports["N"]
    old_port = obj.ports["E"]
    if padr_nub.xmin - obj.xmax > 0:
        tail = CI.add_ref(
            compass(size=(padr_nub.xmin - obj.xmax, wire_width), layer=comb_layer)
        )
    else:
        tail = CI.add_ref(compass(size=(wire_width, wire_width), layer=comb_layer))
    tail.connect(port="W", destination=old_port)

    # disconnected zig
    dhead = CI.add_ref(
        compass(
            size=(padr_nub.ymin - padb.ymax - wire_width, wire_width),
            layer=overlap_zigzag_layer,
        )
    )
    dhead.rotate(90)
    dhead.ymin = padb.ymax
    dhead.xmax = tail.xmin - (wire_spacing + wire_width) / 2
    connector = CI.add_ref(
        compass(size=(wire_width, wire_width), layer=overlap_zigzag_layer)
    )
    connector.connect(port="S", destination=dhead.ports["E"])
    old_port = connector.ports["N"]
    right = True
    obj = connector
    while obj.ymax + wire_spacing + wire_width < head.ymax:
        obj = CI.add_ref(
            compass(size=(wire_spacing, wire_width), layer=overlap_zigzag_layer)
        )
        obj.connect(port="W", destination=old_port)
        old_port = obj.ports["E"]
        if right:
            obj = CI.add_ref(
                compass(size=(wire_width, wire_width), layer=overlap_zigzag_layer)
            )
            obj.connect(port="W", destination=old_port)
            right = False
        else:
            obj = CI.add_ref(
                compass(size=(wire_width, wire_width), layer=overlap_zigzag_layer)
            )
            obj.connect(port="E", destination=old_port)
            right = True
        old_port = obj.ports["N"]
        obj = CI.add_ref(
            compass(
                size=(
                    dhead.xmin - (head.xmax + head.xmin + wire_width) / 2,
                    wire_width,
                ),
                layer=overlap_zigzag_layer,
            )
        )
        obj.connect(port="E", destination=old_port)
        old_port = obj.ports["W"]
        obj = CI.add_ref(
            compass(size=(wire_width, wire_width), layer=overlap_zigzag_layer)
        )
        obj.connect(port="S", destination=old_port)
        if right:
            old_port = obj.ports["W"]
        else:
            old_port = obj.ports["E"]

    return CI


def _test_ic_wire_step(thick_width=10, thin_width=1, wire_layer=2):
    """Helper function used to make the IC step wire structure."""
    WS4 = Device("test_ic_step")
    wire_stepa = WS4.add_ref(
        optimal_step(thick_width / 2, thin_width / 2, layer=wire_layer)
    )
    wire_stepb = WS4.add_ref(
        optimal_step(thin_width / 2, thick_width / 2, layer=wire_layer)
    )
    wire_stepc = WS4.add_ref(
        optimal_step(thick_width / 2, thin_width / 2, layer=wire_layer)
    )
    wire_stepd = WS4.add_ref(
        optimal_step(thin_width / 2, thick_width / 2, layer=wire_layer)
    )
    wire_stepb.rotate(180)
    wire_stepb.xmin = wire_stepa.xmin
    wire_stepc.rotate(180)
    wire_stepc.xmin = wire_stepa.xmax
    wire_stepd.xmin = wire_stepc.xmin
    return WS4


def test_ic(
    wire_widths=[0.25, 0.5, 1, 2, 4],
    wire_widths_wide=[0.75, 1.5, 3, 4, 6],
    pad_size=(200, 200),
    pad_gap=75,
    wire_layer=0,
    pad_layer=1,
    gnd_layer=1,
):
    """Critical current test structure for superconducting wires.

    Parameters
    ----------
    wire_widths : array-like[N]
        The widths of the thinnest parts of the test wires.
    wire_widths_wide : array-like[N]
        The widths of the thickest parts of the test wires.
    pad_size : array-like[2] of int or float
        (width, height) of the pads.
    pad_gap : int or float
        Distance between the pads and the ground plane.
    wire_layer : int
        Specific layer(s) to put the wires on.
    pad_layer : int
        Specific layer(s) to put the pads on.
    gnd_layer : int or None
        Specific layer(s) to put the ground plane on.

    Returns
    -------
    Device
        A Device containing the critical-current test structure.
    """
    ICS = Device("test_ic")

    # if gnd_layer is None: gnd_layer = pad_layer
    translation = 0
    padb = ICS.add_ref(
        rectangle(
            size=(np.size(wire_widths) * (pad_size[0] * 6 / 5), pad_size[1]),
            layer=wire_layer,
        )
    )

    padb_overlay = ICS.add_ref(
        rectangle(
            size=(
                (np.size(wire_widths) * (pad_size[0] * 6 / 5)) * 9 / 10,
                pad_size[1] * 9 / 10,
            ),
            layer=gnd_layer,
        )
    )
    padb_overlay.center = padb.center
    padb_overlay.ymin = padb.ymin
    for i, x in enumerate(wire_widths_wide):
        padt = ICS.add_ref(rectangle(pad_size, wire_layer))
        padt.xmin = padb.xmin + translation
        padt.ymin = padb.ymax + pad_gap
        padt_overlay = ICS.add_ref(
            rectangle(
                size=(pad_size[0] * 9 / 10, pad_size[1] * 9 / 10), layer=pad_layer
            )
        )
        padt_overlay.center = padt.center
        padt_overlay.ymax = padt.ymax
        difference = padt.ymin - padb.ymax
        wire_step = ICS.add_ref(
            _test_ic_wire_step(
                wire_widths_wide[i], wire_widths[i], wire_layer=wire_layer
            )
        )
        wire_step.rotate(90)
        wire_step.center = (padt.center[0], padb.ymax + difference / 2)
        translation = translation + pad_size[0] * 12 / 10
        conn_wire_top = ICS.add_ref(
            rectangle(
                size=(wire_widths_wide[i], padt.ymin - wire_step.ymax), layer=wire_layer
            )
        )
        conn_wire_bottom = ICS.add_ref(
            rectangle(
                size=(wire_widths_wide[i], wire_step.ymin - padb.ymax), layer=wire_layer
            )
        )
        conn_wire_top.ymax = padt.ymin
        conn_wire_top.xmin = wire_step.xmin
        conn_wire_bottom.ymin = padb.ymax
        conn_wire_bottom.xmin = wire_step.xmin

    return ICS


def test_res(
    pad_size=[50, 50], num_squares=1000, width=1, res_layer=0, pad_layer=1, gnd_layer=2
):
    """Creates an efficient resonator structure for a wafer layout.

    Parameters
    ----------
    pad_size : array-like[2] of int or float
        (width, height) of the two matched impedance pads in microns.
    num_squares : int or float
        Number of squares comprising the resonator wire.
    width : int or float
        The width of the squares in microns.
    res_layer : int
        Specific layer(s) to put the resonator structure on.
    pad_layer : int or None
        Specific layer(s) to put the pads on.
    gnd_layer :  int or None
        Specific layer(s) to put the ground plane on.

    Returns
    -------
    P : Device
        A Device containing an efficient resonator structure.
    """
    x = pad_size[0]
    z = pad_size[1]

    # Checking validity of input
    if x <= 0 or z <= 0:
        raise ValueError("Pad must have positive, real dimensions")
    elif width > z:
        raise ValueError("Width of cell cannot be greater than height of pad")
    elif num_squares <= 0:
        raise ValueError("Number of squares must be a positive real number")
    elif width <= 0:
        raise ValueError("Width of cell must be a positive real number")

    # Performing preliminary calculations
    num_rows = int(np.floor(z / (2 * width)))
    if num_rows % 2 == 0:
        num_rows -= 1
    num_columns = num_rows - 1
    squares_in_row = (num_squares - num_columns - 2) / num_rows

    # Compensating for weird edge cases
    if squares_in_row < 1:
        num_rows = int(round(num_rows / 2) - 2)
        squares_in_row = 1
    if width * 2 > z:
        num_rows = 1
        squares_in_row = num_squares - 2

    length_row = squares_in_row * width

    # Creating row/column corner combination structure
    T = Device()
    Row = rectangle(size=(length_row, width), layer=res_layer)
    Col = rectangle(size=(width, width), layer=res_layer)

    T.add_ref(Row)
    col = T.add_ref(Col)
    col.move([length_row - width, -width])

    # Creating entire waveguide net
    N = Device("net")
    n = 1
    for i in range(num_rows):
        if i != num_rows - 1:
            d = N.add_ref(T)
        else:
            d = N.add_ref(Row)
        if n % 2 == 0:
            d.mirror(p1=(d.x, d.ymax), p2=(d.x, d.ymin))
        d.movey(-(n - 1) * T.ysize)
        n += 1
    d = N.add_ref(Col).movex(-width)
    d = N.add_ref(Col).move([length_row, -(n - 2) * T.ysize])

    # Creating pads
    P = Device("test_res")
    Pad1 = rectangle(size=(x, z), layer=pad_layer)
    Pad2 = rectangle(size=(x + 5, z), layer=pad_layer)
    Gnd1 = offset(Pad1, distance=-5, layer=gnd_layer)
    Gnd2 = offset(Pad2, distance=-5, layer=gnd_layer)
    pad1 = P.add_ref(Pad1).movex(-x - width)
    pad2 = P.add_ref(Pad1).movex(length_row + width)
    P.add_ref(Gnd1).center = pad1.center  # gnd1
    gnd2 = P.add_ref(Gnd2)
    P.add_ref(N).y = pad1.y  # nets
    gnd2.center = pad2.center
    gnd2.movex(2.5)

    return P


# ==============================================================================
#
# Optimal current-crowding superconducting structures
#
# ==============================================================================


@device_lru_cache
def optimal_hairpin(width=0.2, pitch=0.6, length=10, turn_ratio=4, num_pts=50, layer=0):
    """Creates an optimally-rounded hairpin geometry, with a 180 degree turn
    on the right end of the polygon connected to two prongs extending towards
    ports on the left end.

    Parameters
    ----------
    width : int or float
        Width of the hairpin leads.
    pitch : int or float
        Distance between the two hairpin leads. Must be greater than width.
    length : int or float
        Length of the hairpin from the connectors to the opposite end of the
        curve.
    turn_ratio : int or float
        Specifies how much of the hairpin is dedicated to the 180 degree turn.
        A turn_ratio of 10 will result in 20% of the hairpin being comprised of the turn.
    num_pts : int
        Number of points constituting the 180 degree turn.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing an optimally-rounded hairpin geometry.

    Notes
    -----
    Hairpin pitch must be greater than width.

    Optimal structure from https://doi.org/10.1103/PhysRevB.84.174510
    Clem, J., & Berggren, K. (2011). Geometry-dependent critical currents in
    superconducting nanocircuits. Physical Review B, 84(17), 1–27.
    """
    # ==========================================================================
    #  Create the basic geometry
    # ==========================================================================
    a = (pitch + width) / 2
    y = -(pitch - width) / 2
    x = -pitch
    dl = width / (num_pts * 2)
    n = 0

    # Get points of ideal curve from conformal mapping
    # TODO This is an inefficient way of finding points that you need
    xpts = [x]
    ypts = [y]
    while (y < 0) & (n < 1e6):
        s = x + 1j * y
        w = sqrt(1 - np.exp(pi * s / a))
        wx = np.real(w)
        wy = np.imag(w)
        wx = wx / sqrt(wx**2 + wy**2)
        wy = wy / sqrt(wx**2 + wy**2)
        x = x + wx * dl
        y = y + wy * dl
        xpts.append(x)
        ypts.append(y)
        n = n + 1
    ypts[-1] = 0  # Set last point be on the x=0 axis for sake of cleanliness
    ds_factor = int(len(xpts) / num_pts)  # Downsample the total number of points
    xpts = xpts[::-ds_factor]
    xpts = xpts[::-1]  # This looks confusing, but it's just flipping the arrays around
    ypts = ypts[::-ds_factor]
    ypts = ypts[::-1]  # so the last point is guaranteed to be included when downsampled

    # Add points for the rest of meander
    xpts.append(xpts[-1] + turn_ratio * width)
    ypts.append(0)
    xpts.append(xpts[-1])
    ypts.append(-a)
    xpts.append(xpts[0])
    ypts.append(-a)
    xpts.append(max(xpts) - length)
    ypts.append(-a)
    xpts.append(xpts[-1])
    ypts.append(-a + width)
    xpts.append(xpts[0])
    ypts.append(ypts[0])

    xpts = np.array(xpts)
    ypts = np.array(ypts)

    # ==========================================================================
    #  Create a blank device, add the geometry, and define the ports
    # ==========================================================================
    D = Device(name="hairpin")
    D.add_polygon([xpts, ypts], layer=layer)
    D.add_polygon([xpts, -ypts], layer=layer)

    xports = min(xpts)
    yports = -a + width / 2
    D.add_port(name=1, midpoint=[xports, -yports], width=width, orientation=180)
    D.add_port(name=2, midpoint=[xports, yports], width=width, orientation=180)

    return D


@device_lru_cache
def optimal_step(
    start_width=10,
    end_width=22,
    num_pts=50,
    width_tol=1e-3,
    anticrowding_factor=1.2,
    symmetric=False,
    layer=0,
):
    """Creates an optimally-rounded step geometry.

    Parameters
    ----------
    start_width : int or float
        Width of the connector on the left end of the step.
    end_width : int or float
        Width of the connector on the right end of the step.
    num_pts : int
        The number of points comprising the entire step geometry.
    width_tol : float
        Point at which to terminate the calculation of the optimal step
    anticrowding_factor : int or float
        Factor to reduce current crowding by elongating
        the structure and reducing the curvature
    symmetric : bool
        If True, adds a mirrored copy of the step across the x-axis to the
        geometry and adjusts the width of the ports.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing an optimally-rounded step.

    Notes
    -----
    Optimal structure from https://doi.org/10.1103/PhysRevB.84.174510
    Clem, J., & Berggren, K. (2011). Geometry-dependent critical currents in
    superconducting nanocircuits. Physical Review B, 84(17), 1–27.
    """
    # ==========================================================================
    #  Create the basic geometry
    # ==========================================================================
    def step_points(eta, W, a):
        # Returns points from a unit semicircle in the w (= u + iv) plane to
        # the optimal curve in the zeta (= x + iy) plane which transitions
        # a wire from a width of 'W' to a width of 'a'
        # eta takes value 0 to pi

        W = np.complex(W)
        a = np.complex(a)

        gamma = (a * a + W * W) / (a * a - W * W)

        w = np.exp(1j * eta)

        zeta = (
            4
            * 1j
            / pi
            * (
                W * np.arctan(sqrt((w - gamma) / (gamma + 1)))
                + a * np.arctan(sqrt((gamma - 1) / (w - gamma)))
            )
        )

        x = np.real(zeta)
        y = np.imag(zeta)
        return x, y

    def invert_step_point(x_desired=-10, y_desired=None, W=1, a=2):
        # Finds the eta associated with the value x_desired along the
        # optimal curve
        def fh(eta):
            guessed_x, guessed_y = step_points(eta, W=W, a=a)
            if y_desired is None:
                return (guessed_x - x_desired) ** 2  # The error
            else:
                return (guessed_y - y_desired) ** 2

        try:
            from scipy.optimize import fminbound
        except Exception:
            raise ImportError(
                " [PHIDL] To run the optimal-curve geometry "
                "functions you need scipy, please install "
                "it with `pip install scipy`"
            )
        found_eta = fminbound(fh, x1=0, x2=pi, args=())
        return step_points(found_eta, W=W, a=a)

    if start_width > end_width:
        reverse = True
        start_width, end_width = end_width, start_width
    else:
        reverse = False

    D = Device(name="step")
    if start_width == end_width:  # Just return a square
        if symmetric:
            ypts = [
                -start_width / 2,
                start_width / 2,
                start_width / 2,
                -start_width / 2,
            ]
            xpts = [0, 0, start_width, start_width]
        if not symmetric:
            ypts = [0, start_width, start_width, 0]
            xpts = [0, 0, start_width, start_width]
        D.info["num_squares"] = 1
    else:
        xmin, ymin = invert_step_point(
            y_desired=start_width * (1 + width_tol), W=start_width, a=end_width
        )
        xmax, ymax = invert_step_point(
            y_desired=end_width * (1 - width_tol), W=start_width, a=end_width
        )

        xpts = np.linspace(xmin, xmax, num_pts).tolist()
        ypts = []
        for x in xpts:
            x, y = invert_step_point(x_desired=x, W=start_width, a=end_width)
            ypts.append(y)

        ypts[-1] = end_width
        ypts[0] = start_width
        y_num_sq = np.array(ypts)
        x_num_sq = np.array(xpts)

        if not symmetric:
            xpts.append(xpts[-1])
            ypts.append(0)
            xpts.append(xpts[0])
            ypts.append(0)
        else:
            xpts += [x for x in xpts[::-1]]
            ypts += [-y for y in ypts[::-1]]
            xpts = [x / 2 for x in xpts]
            ypts = [y / 2 for y in ypts]

        # anticrowding_factor stretches the wire out; a stretched wire is a
        # gentler transition, so there's less chance of current crowding if
        # the fabrication isn't perfect but as a result, the wire isn't as
        # short as it could be
        xpts = (np.array(xpts) * anticrowding_factor).tolist()

        if reverse is True:
            xpts = (-np.array(xpts)).tolist()
            start_width, end_width = end_width, start_width

        D.info["num_squares"] = np.sum(
            np.diff(x_num_sq) / ((y_num_sq[:-1] + y_num_sq[1:]) / 2)
        )

    # ==========================================================================
    #  Create a blank device, add the geometry, and define the ports
    # ==========================================================================
    D.add_polygon([xpts, ypts], layer=layer)

    if not symmetric:
        D.add_port(
            name=1,
            midpoint=[min(xpts), start_width / 2],
            width=start_width,
            orientation=180,
        )
        D.add_port(
            name=2, midpoint=[max(xpts), end_width / 2], width=end_width, orientation=0
        )
    if symmetric:
        D.add_port(name=1, midpoint=[min(xpts), 0], width=start_width, orientation=180)
        D.add_port(name=2, midpoint=[max(xpts), 0], width=end_width, orientation=0)

    return D


def optimal_90deg(width=100.0, num_pts=15, length_adjust=1, layer=0):
    """Creates an optimally-rounded 90 degree bend that is sharp on the outer
    corner.

    Parameters
    ----------
    width : int or float
        Width of the ports on either side of the bend.
    num_pts : int
        The number of points comprising the curved section of the bend.
    length_adjust : int or float
        Adjusts the length of the non-curved portion of the bend.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing an optimally-rounded 90 degree bend.

    Notes
    -----
    Optimal structure from https://doi.org/10.1103/PhysRevB.84.174510
    Clem, J., & Berggren, K. (2011). Geometry-dependent critical currents in
    superconducting nanocircuits. Physical Review B, 84(17), 1–27.
    """
    D = Device("90deg")

    # Get points of ideal curve
    a = 2 * width
    v = np.logspace(-length_adjust, length_adjust, num_pts)
    xi = (
        a / 2.0 * ((1 + 2 / pi * np.arcsinh(1 / v)) + 1j * (1 + 2 / pi * np.arcsinh(v)))
    )
    xpts = list(np.real(xi))
    ypts = list(np.imag(xi))

    # Add points for the rest of curve
    d = 2 * xpts[0]  # Farthest point out * 2, rounded to nearest 100
    xpts.append(width)
    ypts.append(d)
    xpts.append(0)
    ypts.append(d)
    xpts.append(0)
    ypts.append(0)
    xpts.append(d)
    ypts.append(0)
    xpts.append(d)
    ypts.append(width)
    xpts.append(xpts[0])
    ypts.append(ypts[0])

    D.add_polygon([xpts, ypts], layer=layer)

    D.add_port(name=1, midpoint=[a / 4, d], width=a / 2, orientation=90)
    D.add_port(name=2, midpoint=[d, a / 4], width=a / 2, orientation=0)
    return D


# ==============================================================================
# Example code
# ==============================================================================

# hairpin = optimal_hairpin(width = 1, pitch = 3, length = 30, num_pts = 20)
# quickplot(hairpin)


# step = optimal_step(start_width = 5, end_width = 1, num_pts = 80, width_tol = 1e-3)
# quickplot(step)


# turn = optimal_90deg(width = 90, length_adjust = 1)
# quickplot(turn)


# ==============================================================================
#
# Superconducting devices
#
# ==============================================================================


@device_lru_cache
def snspd(
    wire_width=0.2,
    wire_pitch=0.6,
    size=(10, 8),
    num_squares=None,
    turn_ratio=4,
    terminals_same_side=False,
    layer=0,
):
    """Creates an optimally-rounded SNSPD.

    Parameters
    ----------
    width : int or float
        Width of the wire.
    pitch : int or float
        Distance between two adjacent wires. Must be greater than `width`.
    size : None or array-like[2] of int or float
        (width, height) of the rectangle formed by the outer boundary of the
        SNSPD. Must be none if `num_squares` is specified.
    num_squares : int or None
        Total number of squares inside the SNSPD length. Must be none if
        `size` is specified.
    turn_ratio : int or float
        Specifies how much of the SNSPD width is dedicated to the 180 degree
        turn. A `turn_ratio` of 10 will result in 20% of the width being
        comprised of the turn.
    terminals_same_side : bool
        If True, both ports will be located on the same side of the SNSPD.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing an optimally-rounded SNSPD.
    """
    # Convenience tests to auto-shape the size based
    # on the number of squares
    if num_squares is not None and (
        (size is None) or ((size[0] is None) and (size[1]) is None)
    ):
        xy = np.sqrt(num_squares * wire_pitch * wire_width)
        size = [xy, xy]
        num_squares = None
    if [size[0], size[1], num_squares].count(None) != 1:
        raise ValueError(
            "[PHIDL] snspd() requires that exactly ONE value of "
            "the arguments ``num_squares`` and ``size`` be None "
            "to prevent overconstraining, for example:\n"
            ">>> snspd(size = (3, None), num_squares = 2000)"
        )
    if size[0] is None:
        ysize = size[1]
        xsize = num_squares * wire_pitch * wire_width / ysize
    elif size[1] is None:
        xsize = size[0]
        ysize = num_squares * wire_pitch * wire_width / xsize
    else:
        xsize = size[0]
        ysize = size[1]

    num_meanders = int(np.ceil(ysize / wire_pitch))

    D = Device(name="snspd")
    hairpin = optimal_hairpin(
        width=wire_width,
        pitch=wire_pitch,
        turn_ratio=turn_ratio,
        length=xsize / 2,
        num_pts=20,
        layer=layer,
    )

    if (terminals_same_side is False) and ((num_meanders % 2) == 0):
        num_meanders += 1
    elif (terminals_same_side is True) and ((num_meanders % 2) == 1):
        num_meanders += 1

    start_nw = D.add_ref(compass(size=[xsize / 2, wire_width], layer=layer))

    hp_prev = D.add_ref(hairpin)
    hp_prev.connect(1, start_nw.ports["E"])
    alternate = True
    for n in range(2, num_meanders):
        hp = D.add_ref(hairpin)
        if alternate:
            hp.connect(2, hp_prev.ports[2])
            last_port = hp.ports[1]
        else:
            hp.connect(1, hp_prev.ports[1])
            last_port = hp.ports[2]
        hp_prev = hp
        alternate = not alternate

    finish_se = D.add_ref(compass(size=[xsize / 2, wire_width], layer=layer))
    finish_se.connect("E", last_port)

    D.add_port(port=start_nw.ports["W"], name=1)
    D.add_port(port=finish_se.ports["W"], name=2)

    D.info["num_squares"] = num_meanders * (xsize / wire_width)
    D.info["area"] = xsize * ysize
    D.info["size"] = (xsize, ysize)

    return D


def snspd_expanded(
    wire_width=0.2,
    wire_pitch=0.6,
    size=(10, 8),
    num_squares=None,
    connector_width=1,
    connector_symmetric=False,
    turn_ratio=4,
    terminals_same_side=False,
    layer=0,
):
    """Creates an optimally-rounded SNSPD with wires coming out of it that
    expand.

    Parameters
    ----------
    width : int or float
        Width of the wire.
    pitch : int or float
        Distance between two adjacent wires. Must be greater than `width`.
    size : None or array-like[2] of int or float
        (width, height) of the rectangle formed by the outer boundary of the
        SNSPD. Must be none if `num_squares` is specified.
    num_squares : int or None
        Total number of squares inside the SNSPD length. Must be none if
        `size` is specified.
    connector_width : int or float
        Width of the connectors.
    connector_symmetric : bool
        If True, mirrors the connectors across their flat edge and adds them
        to the connector geometry.
    turn_ratio : int or float
        Specifies how much of the SNSPD width is dedicated to the 180 degree
        turn. A `turn_ratio` of 10 will result in 20% of the width being
        comprised of the turn.
    terminals_same_side : bool
        If True, both ports will be located on the same side of the SNSPD.
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing an optimally-rounded SNSPD with expanding input/
        output wires.
    """
    D = Device("snspd_expanded")
    S = snspd(
        wire_width=wire_width,
        wire_pitch=wire_pitch,
        size=size,
        num_squares=num_squares,
        turn_ratio=turn_ratio,
        terminals_same_side=terminals_same_side,
        layer=layer,
    )
    s = D.add_ref(S)
    step_device = optimal_step(
        start_width=wire_width,
        end_width=connector_width,
        num_pts=100,
        anticrowding_factor=2,
        width_tol=1e-3,
        symmetric=connector_symmetric,
        layer=layer,
    )
    step1 = D.add_ref(step_device)
    step2 = D.add_ref(step_device)
    step1.connect(port=1, destination=s.ports[1])
    step2.connect(port=1, destination=s.ports[2])
    D.add_port(name=1, port=step1.ports[2])
    D.add_port(name=2, port=step2.ports[2])

    D.info = S.info
    S.info = {}

    return D


# ==============================================================================
# Example code
# ==============================================================================

# s = snspd(wire_width = 0.2, wire_pitch = 0.6, size = [10,3], terminals_same_side = True)
# quickplot(s)

# s = snspd(wire_width = 0.2, wire_pitch = 0.6, size = [10, None],
#          num_squares = 1000, terminals_same_side = True)
# quickplot(s)


def ytron_round(
    rho=1,
    arm_lengths=(500, 300),
    source_length=500,
    arm_widths=(200, 200),
    theta=2.5,
    theta_resolution=10,
    layer=0,
):
    """Ytron structure for superconducting nanowires

    McCaughan, A. N., Abebe, N. S., Zhao, Q.-Y. & Berggren, K. K.
    Using Geometry To Sense Current. Nano Lett. 16, 7626–7631 (2016).
    http://dx.doi.org/10.1021/acs.nanolett.6b03593

    Parameters
    ----------
    rho : int or float
        Radius of curvature of ytron intersection point
    arm_lengths : array-like[2] of int or float
        Lengths of the left and right arms of the yTron, respectively.
    source_length : int or float
        Length of the source of the yTron.
    arm_widths : array-like[2] of int or float
        Widths of the left and right arms of the yTron, respectively.
    theta : int or float
        Angle between the two yTron arms.
    theta_resolution : int or float
        Angle resolution for curvature of ytron intersection point
    layer : int, array-like[2], or set
        Specific layer(s) to put polygon geometry on.

    Returns
    -------
    D : Device
        A Device containing a yTron geometry.
    """
    # ==========================================================================
    #  Create the basic geometry
    # ==========================================================================
    theta = theta * pi / 180
    theta_resolution = theta_resolution * pi / 180
    thetalist = np.linspace(
        -(pi - theta), -theta, int((pi - 2 * theta) / theta_resolution) + 2
    )
    semicircle_x = rho * cos(thetalist)
    semicircle_y = rho * sin(thetalist) + rho

    # Rest of yTron
    xc = rho * cos(theta)
    yc = rho * sin(theta)
    arm_x_left = arm_lengths[0] * sin(theta)
    arm_y_left = arm_lengths[0] * cos(theta)
    arm_x_right = arm_lengths[1] * sin(theta)
    arm_y_right = arm_lengths[1] * cos(theta)

    # Write out x and y coords for yTron polygon
    xpts = semicircle_x.tolist() + [
        xc + arm_x_right,
        xc + arm_x_right + arm_widths[1],
        xc + arm_widths[1],
        xc + arm_widths[1],
        0,
        -(xc + arm_widths[0]),
        -(xc + arm_widths[0]),
        -(xc + arm_x_left + arm_widths[0]),
        -(xc + arm_x_left),
    ]
    ypts = semicircle_y.tolist() + [
        yc + arm_y_right,
        yc + arm_y_right,
        yc,
        yc - source_length,
        yc - source_length,
        yc - source_length,
        yc,
        yc + arm_y_left,
        yc + arm_y_left,
    ]

    # ==========================================================================
    #  Create a blank device, add the geometry, and define the ports
    # ==========================================================================
    D = Device(name="ytron")
    D.add_polygon([xpts, ypts], layer=layer)
    D.add_port(
        name="left",
        midpoint=[-(xc + arm_x_left + arm_widths[0] / 2), yc + arm_y_left],
        width=arm_widths[0],
        orientation=90,
    )
    D.add_port(
        name="right",
        midpoint=[xc + arm_x_right + arm_widths[1] / 2, yc + arm_y_right],
        width=arm_widths[1],
        orientation=90,
    )
    D.add_port(
        name="source",
        midpoint=[0 + (arm_widths[1] - arm_widths[0]) / 2, -source_length + yc],
        width=arm_widths[0] + arm_widths[1] + 2 * xc,
        orientation=-90,
    )

    # ==========================================================================
    #  Record any parameters you may want to access later
    # ==========================================================================
    D.info["rho"] = rho
    D.info["left_width"] = arm_widths[0]
    D.info["right_width"] = arm_widths[1]
    D.info["source_width"] = arm_widths[0] + arm_widths[1] + 2 * xc

    return D


# ==============================================================================
# Example code
# ==============================================================================

# y = ytron_round(rho = 1, arm_lengths = (500,300),  source_length = 500,
# arm_widths = (200, 200), theta = 2.5, theta_resolution = 10,
# layer = 0)
# quickplot(y)


# %%
