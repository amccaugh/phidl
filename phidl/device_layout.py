# -*- coding: utf-8 -*-
# ==============================================================================
# Major TODO
# ==============================================================================
# Add D.add_gdsii_path() to allow creation of GDSII paths
# Add D.write_gds(max_points_per_polygon)
# Remove Device.add()
# Show labels in quickplot

# ==============================================================================
# Minor TODO
# ==============================================================================
# Add Group.get_polygons()
# Allow Boolean to use Groups
# Add pp.delay_sine(distance = 10, length = 20, num_periods = 2)
# Allow connect(overlap) to be a tuple (0, 0.7)
# Possibly replace gdspy bezier (font rendering) with
#   https://stackoverflow.com/a/12644499
# ==============================================================================
# Documentation TODO
# ==============================================================================
# Tutorials
# - Using Aliases
# - "Using info / metadata" tutorial with .info explanation and tutorial of get_info
# - Advanced and Misc (simplify)

# Examples
# - An electrical device with contact pads
# - An optoelectronic device
#     - Waveguide + LED
#     - route_manhattan


# ==============================================================================
# Imports
# ==============================================================================

from __future__ import division  # Otherwise integer division e.g.  20 / 7 = 2
from __future__ import print_function  # Use print('hello') instead of print 'hello'
from __future__ import absolute_import

import hashlib
import warnings
from copy import deepcopy as _deepcopy

import gdspy

# Remove this once gdspy fully deprecates current_library
import gdspy.library
import numpy as np
from numpy import cos, mod, pi, sin, sqrt
from numpy.linalg import norm

from phidl.constants import _CSS3_NAMES_TO_HEX

gdspy.library.use_current_library = False

__version__ = "1.6.0"


# ==============================================================================
# Useful transformation functions
# ==============================================================================


def _rotate_points(points, angle=45, center=(0, 0)):
    """Rotates points around a centerpoint defined by ``center``.  ``points``
    may be input as either single points [1,2] or array-like[N][2], and will
    return in kind.

    Parameters
    ----------
    points : array-like[N][2]
        Coordinates of the element to be rotated.
    angle : int or float
        Angle to rotate the points.
    center : array-like[2]
        Centerpoint of rotation.

    Returns
    -------
    A new set of points that are rotated around ``center``.
    """
    if angle == 0:
        return points
    angle = angle * pi / 180
    ca = cos(angle)
    sa = sin(angle)
    sa = np.array((-sa, sa))
    c0 = np.array(center)
    if np.asarray(points).ndim == 2:
        return (points - c0) * ca + (points - c0)[:, ::-1] * sa + c0
    if np.asarray(points).ndim == 1:
        return (points - c0) * ca + (points - c0)[::-1] * sa + c0


def _reflect_points(points, p1=(0, 0), p2=(1, 0)):
    """Reflects points across the line formed by p1 and p2.  ``points`` may be
    input as either single points [1,2] or array-like[N][2], and will return in kind.

    Parameters
    ----------
    points : array-like[N][2]
        Coordinates of the element to be reflected.
    p1 : array-like[2]
        Coordinates of the start of the reflecting line.
    p2 : array-like[2]
        Coordinates of the end of the reflecting line.

    Returns
    -------
    A new set of points that are reflected across ``p1`` and ``p2``.
    """
    # From http://math.stackexchange.com/questions/11515/point-reflection-across-a-line
    points = np.array(points)
    p1 = np.array(p1)
    p2 = np.array(p2)
    if np.asarray(points).ndim == 1:
        return (
            2 * (p1 + (p2 - p1) * np.dot((p2 - p1), (points - p1)) / norm(p2 - p1) ** 2)
            - points
        )
    if np.asarray(points).ndim == 2:
        return np.array(
            [
                2 * (p1 + (p2 - p1) * np.dot((p2 - p1), (p - p1)) / norm(p2 - p1) ** 2)
                - p
                for p in points
            ]
        )


def _is_iterable(items):
    """Checks if the passed variable is iterable.

    Parameters
    ----------
    items : any
        Item to check for iterability.
    """
    return isinstance(items, (list, tuple, set, np.ndarray))


def _parse_coordinate(c):
    """Translates various inputs (lists, tuples, Ports) to an (x,y) coordinate.

    Parameters
    ----------
    c : array-like[N] or Port
        Input to translate into a coordinate.

    Returns
    -------
    c : array-like[2]
        Parsed coordinate.
    """
    if isinstance(c, Port):
        return c.midpoint
    elif np.array(c).size == 2:
        return c
    else:
        raise ValueError(
            "[PHIDL] Could not parse coordinate, input should be array-like (e.g. [1.5,2.3] or a Port"
        )


def _parse_move(origin, destination, axis):
    """Translates various input coordinates to changes in position in the x-
    and y-directions.

    Parameters
    ----------
    origin : array-like[2] of int or float, Port, or key
        Origin point of the move.
    destination : array-like[2] of int or float, Port, key, or None
        Destination point of the move.
    axis : {'x', 'y'}
        Direction of move.

    Returns
    -------
    dx : int or float
        Change in position in the x-direction.
    dy : int or float
        Change in position in the y-direction.
    """
    # If only one set of coordinates is defined, make sure it's used to move things
    if destination is None:
        destination = origin
        origin = [0, 0]

    d = _parse_coordinate(destination)
    o = _parse_coordinate(origin)
    if axis == "x":
        d = (d[0], o[1])
    if axis == "y":
        d = (o[0], d[1])
    dx, dy = np.array(d) - o

    return dx, dy


def _distribute(elements, direction="x", spacing=100, separation=True, edge=None):
    """Takes a list of elements and distributes them either equally along a
    grid or with a fixed spacing between them.

    Parameters
    ----------
    elements : array-like of PHIDL objects
        Elements to distribute.
    direction : {'x', 'y'}
        Direction of distribution; either a line in the x-direction or
        y-direction.
    spacing : int or float
        Distance between elements.
    separation : bool
        If True, guarantees elements are speparated with a fixed spacing between; if False, elements are spaced evenly along a grid.
    edge : {'x', 'xmin', 'xmax', 'y', 'ymin', 'ymax'}
        Which edge to perform the distribution along (unused if
        separation == True)

    Returns
    -------
    elements : Device, DeviceReference, Port, Polygon, CellArray, Label, or Group
        Distributed elements.
    """
    if len(elements) == 0:
        return elements
    if direction not in ({"x", "y"}):
        raise ValueError(
            "[PHIDL] distribute(): 'direction' argument must be either 'x' or'y'"
        )
    if (
        (direction == "x")
        and (edge not in ({"x", "xmin", "xmax"}))
        and (not separation)
    ):
        raise ValueError(
            "[PHIDL] distribute(): When `separation` == False and direction == 'x',"
            + " the `edge` argument must be one of {'x', 'xmin', 'xmax'}"
        )
    if (
        (direction == "y")
        and (edge not in ({"y", "ymin", "ymax"}))
        and (not separation)
    ):
        raise ValueError(
            "[PHIDL] distribute(): When `separation` == False and direction == 'y',"
            + " the `edge` argument must be one of {'y', 'ymin', 'ymax'}"
        )

    if direction == "y":
        sizes = [e.ysize for e in elements]
    if direction == "x":
        sizes = [e.xsize for e in elements]

    spacing = np.array([spacing] * len(elements))

    if separation:  # Then `edge` doesn't apply
        if direction == "x":
            edge = "xmin"
        if direction == "y":
            edge = "ymin"
    else:
        sizes = np.zeros(len(spacing))

    # Calculate new positions and move each element
    start = elements[0].__getattribute__(edge)
    positions = np.cumsum(np.concatenate(([start], (spacing + sizes))))
    for n, e in enumerate(elements):
        e.__setattr__(edge, positions[n])
    return elements


def _align(elements, alignment="ymax"):
    """Aligns lists of PHIDL elements

    Parameters
    ----------
    elements : array-like of PHIDL objects
        Elements to align.
    alignment : {'x', 'y', 'xmin', 'xmax', 'ymin', 'ymax'}
        Which edge to align along (e.g. 'ymax' will align move the elements such
        that all of their topmost points are aligned)


    Returns
    -------
    elements : array-like of PHIDL objects
        Aligned elements.
    """
    if len(elements) == 0:
        return elements
    if alignment not in (["x", "y", "xmin", "xmax", "ymin", "ymax"]):
        raise ValueError(
            "[PHIDL] 'alignment' argument must be one of 'x','y','xmin', 'xmax', 'ymin','ymax'"
        )
    value = Group(elements).__getattribute__(alignment)
    for e in elements:
        e.__setattr__(alignment, value)
    return elements


def _line_distances(points, start, end):
    if np.all(start == end):
        return np.linalg.norm(points - start, axis=1)

    vec = end - start
    cross = np.cross(vec, start - points)
    return np.divide(abs(cross), np.linalg.norm(vec))


def _simplify(points, tolerance=0):
    """Ramer–Douglas–Peucker algorithm for line simplification. Takes an
    array of points of shape (N,2) and removes excess points in the line. The
    remaining points form a identical line to within `tolerance` from the
    original"""
    # From https://github.com/fhirschmann/rdp/issues/7
    # originally written by Kirill Konevets https://github.com/kkonevets

    M = np.asarray(points)
    start, end = M[0], M[-1]
    dists = _line_distances(M, start, end)

    index = np.argmax(dists)
    dmax = dists[index]

    if dmax > tolerance:
        result1 = _simplify(M[: index + 1], tolerance)
        result2 = _simplify(M[index:], tolerance)

        result = np.vstack((result1[:-1], result2))
    else:
        result = np.array([start, end])

    return result


def reset():
    """Resets the built-in Layer dictionary (controls the coloring in
    quickplot() ), and sets the Device universal ID (uid) to zero."""
    Layer.layer_dict = {}
    Device._next_uid = 0


class LayerSet(object):
    """Set of layer objects."""

    def __init__(self):
        """Initialises an empty LayerSet."""
        self._layers = {}

    def add_layer(
        self,
        name="unnamed",
        gds_layer=0,
        gds_datatype=0,
        description=None,
        color=None,
        inverted=False,
        alpha=0.6,
        dither=None,
    ):
        """Adds a layer to an existing LayerSet object.

        Parameters
        ----------
        name : str
            Name of the Layer.
        gds_layer : int
            GDSII Layer number.
        gds_datatype : int
            GDSII datatype.
        description : str
            Layer description.
        color : str
            Hex code of color for the Layer.
        inverted : bool
            If true, inverts the Layer.
        alpha : int or float
            Alpha parameter (opacity) for the Layer, value must be between 0.0
            and 1.0.
        dither : str
            KLayout dither style (only used in phidl.utilities.write_lyp() )

        """
        new_layer = Layer(
            gds_layer=gds_layer,
            gds_datatype=gds_datatype,
            name=name,
            description=description,
            inverted=inverted,
            color=color,
            alpha=alpha,
            dither=dither,
        )
        if name in self._layers:
            raise ValueError(
                "[PHIDL] LayerSet: Tried to add layer named "
                '"%s"' % (name) + ", but a layer with that "
                "name already exists in this LayerSet"
            )
        else:
            self._layers[name] = new_layer

    def __getitem__(self, val):
        """If you have a LayerSet `ls`, allows access to the layer names like
        ls['gold2'].

        Parameters
        ----------
        val : str
            Layer name to access within the LayerSet.

        Returns
        -------
        self._layers[val] : Layer
            Accessed Layer in the LayerSet.
        """
        try:
            return self._layers[val]
        except Exception:
            raise ValueError(
                "[PHIDL] LayerSet: Tried to access layer "
                'named "%s"' % (val) + " which does not exist"
            )

    def __repr__(self):
        """Prints the number of Layers in the LayerSet object."""
        return "LayerSet (%s layers total)" % (len(self._layers))


class Layer(object):
    """Layer object.

    Parameters
    ----------
    gds_layer : int
        GDSII Layer number.
    gds_datatype : int
        GDSII datatype.
    name : str
        Name of the Layer.
    color : str
        Hex code of color for the Layer.
    alpha : int or float
        Alpha parameter (opacity) for the Layer.
    dither : str
        KLayout dither parameter (texture) for the Layer
        (only used in phidl.utilities.write_lyp)
    """

    layer_dict = {}

    def __init__(
        self,
        gds_layer=0,
        gds_datatype=0,
        name="unnamed",
        description=None,
        inverted=False,
        color=None,
        alpha=0.6,
        dither=None,
    ):
        if isinstance(gds_layer, Layer):
            l = gds_layer  # We were actually passed Layer(mylayer), make a copy
            gds_datatype = l.gds_datatype
            name = l.name
            description = l.description
            alpha = l.alpha
            dither = l.dither
            inverted = l.inverted
            gds_layer = l.gds_layer

        self.gds_layer = gds_layer
        self.gds_datatype = gds_datatype
        self.name = name
        self.description = description
        self.inverted = inverted
        self.alpha = alpha
        self.dither = dither

        try:
            if color is None:  # not specified
                self.color = None
            elif np.size(color) == 3:  # in format (0.5, 0.5, 0.5)
                color = np.array(color)
                if np.any(color > 1) or np.any(color < 0):
                    raise ValueError
                color = np.array(np.round(color * 255), dtype=int)
                self.color = "#{:02x}{:02x}{:02x}".format(*color)
            elif color[0] == "#":  # in format #1d2e3f
                if len(color) != 7:
                    raise ValueError
                int(color[1:], 16)  # Will throw error if not hex format
                self.color = color
            else:  # in named format 'gold'
                self.color = _CSS3_NAMES_TO_HEX[color.lower()]
        except Exception:
            raise ValueError(
                "[PHIDL] Layer() color must be specified as a "
                + "0-1 RGB triplet, (e.g. [0.5, 0.1, 0.9]), an HTML hex color string "
                + "(e.g. '#a31df4'), or a CSS3 color name (e.g. 'gold' or "
                + "see http://www.w3schools.com/colors/colors_names.asp )"
            )

        Layer.layer_dict[(gds_layer, gds_datatype)] = self

    def __repr__(self):
        """Prints a description of the Layer object, including the name, GDS
        layer, GDS datatype, description, and color of the Layer."""
        return (
            "Layer (name %s, GDS layer %s, GDS datatype %s, description %s, color %s)"
            % (
                self.name,
                self.gds_layer,
                self.gds_datatype,
                self.description,
                self.color,
            )
        )


def _parse_layer(layer):
    """Check if the variable layer is a Layer object, a 2-element list like
    [0, 1] representing layer = 0 and datatype = 1, or just a layer number.

    Parameters
    ----------
    layer : int, array-like[2], or set
        Variable to check.

    Returns
    -------
    (gds_layer, gds_datatype) : array-like[2]
        The layer number and datatype of the input.
    """
    if isinstance(layer, Layer):
        gds_layer, gds_datatype = layer.gds_layer, layer.gds_datatype
    elif np.shape(layer) == (2,):  # In form [3,0]
        gds_layer, gds_datatype = layer[0], layer[1]
    elif np.shape(layer) == (1,):  # In form [3]
        gds_layer, gds_datatype = layer[0], 0
    elif layer is None:
        gds_layer, gds_datatype = 0, 0
    elif isinstance(layer, (int, float)):
        gds_layer, gds_datatype = layer, 0
    else:
        raise ValueError(
            """[PHIDL] _parse_layer() was passed something
            that could not be interpreted as a layer: layer = %s"""
            % layer
        )
    return (gds_layer, gds_datatype)


class _GeometryHelper(object):
    """This is a helper class. It can be added to any other class which has
    the functions move() and the property ``bbox`` (as in self.bbox). It uses
    that function+property to enable you to do things like check what the
    center of the bounding box is (self.center), and also to do things like
    move the bounding box such that its maximum x value is 5.2
    (self.xmax = 5.2).
    """

    @property
    def center(self):
        """Returns the center of the bounding box."""
        return np.sum(self.bbox, 0) / 2

    @center.setter
    def center(self, destination):
        """Sets the center of the bounding box.

        Parameters
        ----------
        destination : array-like[2]
            Coordinates of the new bounding box center.
        """
        self.move(destination=destination, origin=self.center)

    @property
    def x(self):
        """Returns the x-coordinate of the center of the bounding box."""
        return np.sum(self.bbox, 0)[0] / 2

    @x.setter
    def x(self, destination):
        """Sets the x-coordinate of the center of the bounding box.

        Parameters
        ----------
        destination : int or float
            x-coordinate of the bbox center.
        """
        destination = (destination, self.center[1])
        self.move(destination=destination, origin=self.center, axis="x")

    @property
    def y(self):
        """Returns the y-coordinate of the center of the bounding box."""
        return np.sum(self.bbox, 0)[1] / 2

    @y.setter
    def y(self, destination):
        """Sets the y-coordinate of the center of the bounding box.

        Parameters
        ----------
        destination : int or float
            y-coordinate of the bbox center.
        """
        destination = (self.center[0], destination)
        self.move(destination=destination, origin=self.center, axis="y")

    @property
    def xmax(self):
        """Returns the maximum x-value of the bounding box."""
        return self.bbox[1][0]

    @xmax.setter
    def xmax(self, destination):
        """Sets the x-coordinate of the maximum edge of the bounding box.

        Parameters
        ----------
        destination : int or float
            x-coordinate of the maximum edge of the bbox.
        """
        self.move(destination=(destination, 0), origin=self.bbox[1], axis="x")

    @property
    def ymax(self):
        """Returns the maximum y-value of the bounding box."""
        return self.bbox[1][1]

    @ymax.setter
    def ymax(self, destination):
        """Sets the y-coordinate of the maximum edge of the bounding box.

        Parameters
        ----------
        destination : int or float
            y-coordinate of the maximum edge of the bbox.
        """
        self.move(destination=(0, destination), origin=self.bbox[1], axis="y")

    @property
    def xmin(self):
        """Returns the minimum x-value of the bounding box."""
        return self.bbox[0][0]

    @xmin.setter
    def xmin(self, destination):
        """Sets the x-coordinate of the minimum edge of the bounding box.

        Parameters
        ----------
        destination : int or float
            x-coordinate of the minimum edge of the bbox.
        """
        self.move(destination=(destination, 0), origin=self.bbox[0], axis="x")

    @property
    def ymin(self):
        """Returns the minimum y-value of the bounding box."""
        return self.bbox[0][1]

    @ymin.setter
    def ymin(self, destination):
        """Sets the y-coordinate of the minimum edge of the bounding box.

        Parameters
        ----------
        destination : int or float
            y-coordinate of the minimum edge of the bbox.
        """
        self.move(destination=(0, destination), origin=self.bbox[0], axis="y")

    @property
    def size(self):
        """Returns the (x, y) size of the bounding box."""
        bbox = self.bbox
        return bbox[1] - bbox[0]

    @property
    def xsize(self):
        """Returns the horizontal size of the bounding box."""
        bbox = self.bbox
        return bbox[1][0] - bbox[0][0]

    @property
    def ysize(self):
        """Returns the vertical size of the bounding box."""
        bbox = self.bbox
        return bbox[1][1] - bbox[0][1]

    def movex(self, origin=0, destination=None):
        """Moves an object by a specified x-distance.

        Parameters
        ----------
        origin : array-like[2], Port, or key
            Origin point of the move.
        destination : array-like[2], Port, key, or None
            Destination point of the move.
        """
        if destination is None:
            destination = origin
            origin = 0
        self.move(origin=(origin, 0), destination=(destination, 0))
        return self

    def movey(self, origin=0, destination=None):
        """Moves an object by a specified y-distance.

        Parameters
        ----------
        origin : array-like[2], Port, or key
            Origin point of the move.
        destination : array-like[2], Port, or key
            Destination point of the move.
        """
        if destination is None:
            destination = origin
            origin = 0
        self.move(origin=(0, origin), destination=(0, destination))
        return self

    def __add__(self, element):
        """Adds an element to a Group.

        Parameters
        ----------
        element : Device, DeviceReference, Port, Polygon, CellArray, Label, or Group
            Element to add.
        """
        if isinstance(self, Group):
            G = Group()
            G.add(self.elements)
            G.add(element)
        else:
            G = Group([self, element])
        return G


class Port(object):
    """Port object that can be used to easily snap together other geometric objects

    Parameters
    ----------
    name : str
        Name of the Port object.
    midpoint : array-like[2] of int or float
        Midpoint of the Port location.
    width : int or float
        Width of the Port.
    orientation : int or float
        Orientation (rotation) of the Port.
    parent :
    """

    _next_uid = 0

    def __init__(self, name=None, midpoint=(0, 0), width=1, orientation=0, parent=None):
        self.name = name
        self.midpoint = np.array(midpoint, dtype="float64")
        self.width = width
        self.orientation = mod(orientation, 360)
        self.parent = parent
        self.info = {}
        self.uid = Port._next_uid
        if self.width < 0:
            raise ValueError("[PHIDL] Port creation " "error: width must be >=0")
        Port._next_uid += 1

    def __repr__(self):
        """Prints a description of the Port object, including the name,
        midpoint, width, and orientation of the Port."""
        return "Port (name {}, midpoint {}, width {}, orientation {})".format(
            self.name,
            self.midpoint,
            self.width,
            self.orientation,
        )

    @property
    def endpoints(self):
        """Returns the endpoints of the Port."""
        dxdy = np.array(
            [
                self.width / 2 * cos((self.orientation - 90) * pi / 180),
                self.width / 2 * sin((self.orientation - 90) * pi / 180),
            ]
        )
        left_point = self.midpoint - dxdy
        right_point = self.midpoint + dxdy
        return np.array([left_point, right_point])

    @endpoints.setter
    def endpoints(self, points):
        """Sets the endpoints of a Port.

        Parameters
        ----------
        points : array-like[2] of int or float
            Endpoints to assign to the Port.
        """
        p1, p2 = np.array(points[0]), np.array(points[1])
        self.midpoint = (p1 + p2) / 2
        dx, dy = p2 - p1
        self.orientation = np.arctan2(dx, -dy) * 180 / pi
        self.width = sqrt(dx**2 + dy**2)

    @property
    def normal(self):
        """Returns a vector normal to the Port

        Returns
        -------
        array-like[2]
            Vector normal to the Port
        """
        dx = cos((self.orientation) * pi / 180)
        dy = sin((self.orientation) * pi / 180)
        return np.array([self.midpoint, self.midpoint + np.array([dx, dy])])

    @property
    def x(self):
        """Returns the x-coordinate of the Port midpoint."""
        return self.midpoint[0]

    @property
    def y(self):
        """Returns the y-coordinate of the Port midpoint."""
        return self.midpoint[1]

    @property
    def center(self):
        """Returns the midpoint of the Port."""
        return self.midpoint

    def _copy(self, new_uid=True):
        """Copies a Port.

        Returns
        -------
        Port
            Copied Port.

        Notes
        -----
        Use this function instead of copy() (which will not create a new numpy
        array for self.midpoint) or deepcopy() (which will also deepcopy the
        self.parent DeviceReference recursively, causing performance issues).
        """
        new_port = Port(
            name=self.name,
            midpoint=self.midpoint,
            width=self.width,
            orientation=self.orientation,
            parent=self.parent,
        )
        new_port.info = _deepcopy(self.info)
        if not new_uid:
            new_port.uid = self.uid
            Port._next_uid -= 1
        return new_port

    def rotate(self, angle=45, center=None):
        """Rotates a Port around the specified center point,
        if no centerpoint specified will rotate around (0,0).

        Parameters
        ----------
        angle : int or float
            Angle to rotate the Port in degrees.
            center : array-like[2] or None
            Midpoint of the Port.
        """
        self.orientation = mod(self.orientation + angle, 360)
        if center is None:
            center = self.midpoint
        self.midpoint = _rotate_points(self.midpoint, angle=angle, center=center)
        return self


class Polygon(gdspy.Polygon, _GeometryHelper):
    """Polygonal geometric object.

    Parameters
    ----------
    points : array-like[N][2]
        Coordinates of the vertices of the Polygon.
    gds_layer : int
        GDSII layer of the Polygon.
    gds_datatype : int
        GDSII datatype of the Polygon.
    parent :

    """

    def __init__(self, points, gds_layer, gds_datatype, parent):
        self.parent = parent
        super(Polygon, self).__init__(
            points=points, layer=gds_layer, datatype=gds_datatype
        )

    @property
    def bbox(self):
        """Returns the bounding box of the Polygon."""
        return self.get_bounding_box()

    def rotate(self, angle=45, center=(0, 0)):
        """Rotates a Polygon by the specified angle.

        Parameters
        ----------
        angle : int or float
            Angle to rotate the Polygon in degrees.
        center : array-like[2] or None
            Midpoint of the Polygon.
        """
        super(Polygon, self).rotate(angle=angle * pi / 180, center=center)
        if self.parent is not None:
            self.parent._bb_valid = False
        return self

    def move(self, origin=(0, 0), destination=None, axis=None):
        """Moves elements of the Device from the origin point to the
        destination. Both origin and destination can be 1x2 array-like, Port,
        or a key corresponding to one of the Ports in this device.

        Parameters
        ----------
        origin : array-like[2], Port, or key
            Origin point of the move.
        destination : array-like[2], Port, or key
            Destination point of the move.
        axis : {'x', 'y'}
            Direction of move.

        """
        dx, dy = _parse_move(origin, destination, axis)

        super(Polygon, self).translate(dx, dy)
        if self.parent is not None:
            self.parent._bb_valid = False
        return self

    def mirror(self, p1=(0, 1), p2=(0, 0)):
        """Mirrors a Polygon across the line formed between the two
        specified points. ``points`` may be input as either single points
        [1,2] or array-like[N][2], and will return in kind.

        Parameters
        ----------
        p1 : array-like[N][2]
            First point of the line.
        p2 : array-like[N][2]
            Second point of the line.
        """
        for n, points in enumerate(self.polygons):
            self.polygons[n] = _reflect_points(points, p1, p2)
        if self.parent is not None:
            self.parent._bb_valid = False
        return self

    def reflect(self, p1=(0, 1), p2=(0, 0)):
        """
        .. deprecated:: 1.3.0
            `reflect` will be removed in May 2021, please replace with
            `mirror`.
        """
        warnings.warn(
            "[PHIDL] Warning: reflect() will be deprecated in May 2021, please replace with mirror()"
        )
        return self.mirror(p1, p2)

    def simplify(self, tolerance=1e-3):
        """Removes points from the polygon but does not change the polygon
        shape by more than `tolerance` from the original. Uses the
        Ramer-Douglas-Peucker algorithm.

        Parameters
        ----------
        tolerance : float
            Tolerance value for the simplification algorithm.  All points that
            can be removed without changing the resulting polygon by more than
            the value listed here will be removed. Also known as `epsilon` here
            https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm
        """

        for n, points in enumerate(self.polygons):
            self.polygons[n] = _simplify(points, tolerance=tolerance)
        if self.parent is not None:
            self.parent._bb_valid = False
        return self


def make_device(fun, config=None, **kwargs):
    """Makes a Device from a function.

    Parameters
    ----------
    fun : str
        Name of the function to make the Device with.
    config : dict or None
        A dictionary containing arguments for the given function.

    Returns
    -------
    D : Device
        A Device constructed from the specified function.
    """
    config_dict = {}
    if type(config) is dict:
        config_dict = dict(config)
    elif config is None:
        pass
    else:
        raise TypeError(
            """[PHIDL] When creating Device() from a function, the
        second argument should be a ``config`` argument which is a
        dictionary containing arguments for the function.
        e.g. make_device(ellipse, config = ellipse_args_dict) """
        )
    config_dict.update(**kwargs)
    D = fun(**config_dict)
    if not isinstance(D, Device):
        raise ValueError(
            """[PHIDL] Device() was passed a function, but that
        function does not produce a Device."""
        )
    return D


class Device(gdspy.Cell, _GeometryHelper):
    """The basic object that holds polygons, labels, and ports in PHIDL"""

    _next_uid = 0

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            if callable(args[0]):
                raise ValueError(
                    "[PHIDL] You can no longer create geometry "
                    "by calling Device(device_making_function), please use "
                    "make_device(device_making_function) instead"
                )

        # Allow name to be set like Device('arc') or Device(name = 'arc')
        if "name" in kwargs:
            name = kwargs["name"]
        elif (len(args) == 1) and (len(kwargs) == 0):
            name = args[0]
        else:
            name = "Unnamed"

        # Make a new blank device
        self.ports = {}
        self.info = {}
        self.aliases = {}
        # self.a = self.aliases
        # self.p = self.ports
        self.uid = Device._next_uid
        super(Device, self).__init__(name=name)
        Device._next_uid += 1

    def __getitem__(self, key):
        """If you have a Device D, allows access to aliases you made like
        D['arc2'].

        Parameters
        ----------
        key : str
            Element name to access within the Device.

        Returns
        -------
        self._layers[val] : Layer
            Accessed element in the Device.
        """
        try:
            return self.aliases[key]
        except Exception:
            raise ValueError(
                '[PHIDL] Tried to access alias "%s" in Device '
                '"%s", which does not exist' % (key, self.name)
            )

    def __repr__(self):
        """Prints a description of the Device, including the name, uid,
        ports, aliases, polygons, and references.
        """
        return (
            'Device (name "%s" (uid %s), ports %s, aliases %s, %s '
            "polygons, %s references)"
            % (
                self.name,
                self.uid,
                list(self.ports.keys()),
                list(self.aliases.keys()),
                len(self.polygons),
                len(self.references),
            )
        )

    def __str__(self):
        """Prints a description of the Device, including the name, uid,
        ports, aliases, polygons, and references."""
        return self.__repr__()

    def __lshift__(self, element):
        """Convenience operators equivalent to add_ref()

        Parameters
        ----------
        elements : Device
            Element to reference
        """
        return self.add_ref(element)

    def __setitem__(self, key, element):
        """Allow adding polygons and cell references like D['arc3'] = pg.arc()

        Parameters
        ----------
        key :
            Alias name
        element :
            Object that will be accessible by alias name


        Returns
        -------

        """
        if isinstance(element, (DeviceReference, Polygon, CellArray)):
            self.aliases[key] = element
        else:
            raise ValueError(
                '[PHIDL] Tried to assign alias "%s" in '
                'Device "%s", but failed because the item was '
                "not a DeviceReference" % (key, self.name)
            )

    @property
    def layers(self):
        """Returns a set of the Layers in the Device."""
        return self.get_layers()

    # @property
    # def references(self):
    #     return [e for e in self.elements if isinstance(e, DeviceReference)]

    # @property
    # def polygons(self):
    #     return [e for e in self.elements if isinstance(e, gdspy.PolygonSet)]

    @property
    def bbox(self):
        """Returns the bounding box of the Device."""
        bbox = self.get_bounding_box()
        if bbox is None:
            bbox = ((0, 0), (0, 0))
        return np.array(bbox)

    def add_ref(self, device, alias=None):
        """Takes a Device and adds it as a DeviceReference to the current
        Device.

        Parameters
        ----------
        device : Device
            Device to be added as a DeviceReference.
        alias : str
            Alias of the Device.

        Returns
        -------
        d : DeviceReference
            A DeviceReference that is added to the current Device.
        """
        if _is_iterable(device):
            return [self.add_ref(E) for E in device]
        if not isinstance(device, Device):
            raise TypeError(
                """[PHIDL] add_ref() was passed something that
            was not a Device object. """
            )
        d = DeviceReference(device)  # Create a DeviceReference (CellReference)
        d.owner = self
        self.add(d)  # Add DeviceReference (CellReference) to Device (Cell)

        if alias is not None:
            self.aliases[alias] = d
        return d  # Return the DeviceReference (CellReference)

    def add_polygon(self, points, layer=np.nan):
        """Adds a Polygon to the Device.

        Parameters
        ----------
        points : array-like[N][2]
            Coordinates of the vertices of the Polygon.
        layer : int, array-like[2], or set
            Specific layer(s) to put polygon geometry on.
        """
        if layer is None:
            return None

        # Check if input a list of polygons by seeing if it's 3 levels deep
        try:
            points[0][0][0]  # Try to access first x point
            return [self.add_polygon(p, layer) for p in points]
        except Exception:
            pass  # Verified points is not a list of polygons, continue on

        if isinstance(points, gdspy.PolygonSet):
            if layer is np.nan:
                layers = zip(points.layers, points.datatypes)
            else:
                layers = [layer] * len(points.polygons)
            return [
                self.add_polygon(p, layer) for p, layer in zip(points.polygons, layers)
            ]

        if layer is np.nan:
            layer = 0

        # Check if layer is actually a list of Layer objects
        try:
            if isinstance(layer, LayerSet):
                return [self.add_polygon(points, l) for l in layer._layers.values()]
            elif isinstance(layer, set):
                return [self.add_polygon(points, l) for l in layer]
            elif all([isinstance(l, (Layer)) for l in layer]):
                return [self.add_polygon(points, l) for l in layer]
            elif len(layer) > 2:  # Someone wrote e.g. layer = [1,4,5]
                raise ValueError(
                    """ [PHIDL] If specifying multiple layers
                you must use set notation, e.g. {1,5,8} """
                )
        except Exception:
            pass

        # If in the form [[1,3,5],[2,4,6]]
        if len(points[0]) > 2:
            # Convert to form [[1,2],[3,4],[5,6]]
            points = np.column_stack(points)

        gds_layer, gds_datatype = _parse_layer(layer)
        polygon = Polygon(
            points=points, gds_layer=gds_layer, gds_datatype=gds_datatype, parent=self
        )
        self.add(polygon)
        return polygon

    def add_array(self, device, columns=2, rows=2, spacing=(100, 100), alias=None):
        """Creates a CellArray reference to a Device.

        Parameters
        ----------
        device : Device
            The referenced Device.
        columns : int
            Number of columns in the array.
        rows : int
            Number of rows in the array.
        spacing : array-like[2] of int or float
            Distances between adjacent columns and adjacent rows.
        alias : str or None
            Alias of the referenced Device.

        Returns
        -------
        a : CellArray
            A CellArray containing references to the input Device.
        """
        if not isinstance(device, Device):
            raise TypeError(
                """[PHIDL] add_array() was passed something that
            was not a Device object. """
            )
        a = CellArray(
            device=device,
            columns=int(round(columns)),
            rows=int(round(rows)),
            spacing=spacing,
        )
        a.owner = self
        self.add(a)  # Add DeviceReference (CellReference) to Device (Cell)
        if alias is not None:
            self.aliases[alias] = a
        return a  # Return the CellArray

    def add_port(self, name=None, midpoint=(0, 0), width=1, orientation=45, port=None):
        """Adds a Port to the Device.

        Parameters
        ----------
        name : str
            Name of the Port object.
        midpoint : array-like[2] of int or float
            Midpoint of the Port location.
        width : int or float
            Width of the Port.
        orientation : int or float
            Orientation (rotation) of the Port.
        port : Port or None
            A Port if the added Port is a copy of an existing Port.

        Notes
        -----
        Can be called to copy an existing port like
        add_port(port = existing_port) or to create a new port
        add_port(myname, mymidpoint, mywidth, myorientation).
        Can also be called to copy an existing port with a new name like
        add_port(port = existing_port, name = new_name)
        """
        if port is not None:
            if not isinstance(port, Port):
                raise ValueError(
                    "[PHIDL] add_port() error: Argument `port` must be a Port for copying"
                )
            p = port._copy(new_uid=True)
            p.parent = self
        elif isinstance(name, Port):
            p = name._copy(new_uid=True)
            p.parent = self
            name = p.name
        else:
            p = Port(
                name=name,
                midpoint=midpoint,
                width=width,
                orientation=orientation,
                parent=self,
            )
        if name is not None:
            p.name = name
        if p.name in self.ports:
            raise ValueError(
                '[DEVICE] add_port() error: Port name "%s" already exists in this Device (name "%s", uid %s)'
                % (p.name, self.name, self.uid)
            )
        self.ports[p.name] = p
        return p

    def add_label(
        self,
        text="hello",
        position=(0, 0),
        magnification=None,
        rotation=None,
        anchor="o",
        layer=255,
    ):
        """Adds a Label to the Device.

        Parameters
        ----------
        text : str
            Label text.
        position : array-like[2]
            x-, y-coordinates of the Label location.
        magnification : int, float, or None
            Magnification factor for the Label text.
        rotation : int, float, or None
            Angle rotation of the Label text.
        anchor : {'n', 'e', 's', 'w', 'o', 'ne', 'nw', ...}
            Position of the anchor relative to the text.
        layer : int, array-like[2], or set
            Specific layer(s) to put Label on.
        """
        if layer is None:
            return None
        if len(text) >= 1023:
            raise ValueError("[DEVICE] label() error: Text too long (limit 1024 chars)")
        gds_layer, gds_datatype = _parse_layer(layer)

        if type(text) is not str:
            text = str(text)
        l = Label(
            text=text,
            position=position,
            anchor=anchor,
            magnification=magnification,
            rotation=rotation,
            layer=gds_layer,
            texttype=gds_datatype,
        )
        self.add(l)
        return l

    def label(self, *args, **kwargs):
        """
        .. deprecated:: 1.3.0
            `label` will be removed, please replace with
            `add_label`.
        """
        warnings.warn(
            "[PHIDL] WARNING: label() will be deprecated, please replace with add_label()"
        )
        return self.add_label(*args, **kwargs)

    def write_gds(
        self,
        filename,
        unit=1e-6,
        precision=1e-9,
        auto_rename=True,
        max_cellname_length=28,
        cellname="toplevel",
    ):
        """Writes a Device to a GDS file.

        Parameters
        ----------
        filename : str or file
            The GDS file to write to.
        unit : int or float
            Unit size for the objects in the library (in `meters`).
        precision : float
            Precision for the dimensions of the objects in the library (in
            `meters`).
        auto_rename : bool
            If True, fixes any duplicate cell names.
        max_cellname_length : int or None
            If given, and if `auto_rename` is True, enforces a limit on the
            length of the fixed duplicate cellnames.
        cellname : str
            Name of the top-level cell in the saved GDS

        Returns
        -------

        """
        # If string, try to append ".gds" to the end, otherwise leave alone
        try:
            if filename[-4:] != ".gds":
                filename += ".gds"
        except Exception:
            pass
        referenced_cells = list(self.get_dependencies(recursive=True))
        all_cells = [self] + referenced_cells

        # Autofix names so there are no duplicates
        if auto_rename:
            all_cells_sorted = sorted(all_cells, key=lambda x: x.uid)
            all_cells_original_names = [c.name for c in all_cells_sorted]
            used_names = {cellname}
            n = 1
            for c in all_cells_sorted:
                if max_cellname_length is not None:
                    new_name = c.name[:max_cellname_length]
                else:
                    new_name = c.name
                temp_name = new_name
                while temp_name in used_names:
                    n += 1
                    temp_name = new_name + ("%0.3i" % n)
                new_name = temp_name
                used_names.add(new_name)
                c.name = new_name
            self.name = cellname

        # Write the gds
        lib = gdspy.GdsLibrary(unit=unit, precision=precision)
        lib.write_gds(filename, cells=all_cells)
        # Return cells to their original names if they were auto-renamed
        if auto_rename:
            for n, c in enumerate(all_cells_sorted):
                c.name = all_cells_original_names[n]
        return filename

    def remap_layers(self, layermap={}, include_labels=True):
        """Moves all polygons in the Device from one layer to another
        according to the layermap argument.

        Parameters
        ----------
        layermap : dict
            Dictionary of values in format {layer_from : layer_to}
        include_labels : bool
            Selects whether to move Labels along with polygons
        """
        layermap = {_parse_layer(k): _parse_layer(v) for k, v in layermap.items()}

        all_D = list(self.get_dependencies(True))
        all_D += [self]
        for D in all_D:
            for p in D.polygons:
                for n, layer in enumerate(p.layers):
                    original_layer = (p.layers[n], p.datatypes[n])
                    original_layer = _parse_layer(original_layer)
                    if original_layer in layermap.keys():
                        new_layer = layermap[original_layer]
                        p.layers[n] = new_layer[0]
                        p.datatypes[n] = new_layer[1]
            if include_labels:
                for l in D.labels:
                    original_layer = (l.layer, l.texttype)
                    original_layer = _parse_layer(original_layer)
                    if original_layer in layermap.keys():
                        new_layer = layermap[original_layer]
                        l.layer = new_layer[0]
                        l.texttype = new_layer[1]
        return self

    def remove_layers(self, layers=(), include_labels=True, invert_selection=False):
        """Removes layers from a Device.

        Parameters
        ----------
        layers : int, array-like[2], or set
            Specific layer(s) to remove.
        include_labels : bool
            If True, keeps the labels corresponding to the input layers.
        invert_selection : bool
            If True, removes all layers except those specified.
        """
        layers = [_parse_layer(l) for l in layers]
        all_D = list(self.get_dependencies(True))
        all_D += [self]
        for D in all_D:
            for polygonset in D.polygons:
                polygon_layers = zip(polygonset.layers, polygonset.datatypes)
                polygons_to_keep = [(pl in layers) for pl in polygon_layers]
                if not invert_selection:
                    polygons_to_keep = [(not p) for p in polygons_to_keep]
                polygonset.polygons = [
                    p for p, keep in zip(polygonset.polygons, polygons_to_keep) if keep
                ]
                polygonset.layers = [
                    p for p, keep in zip(polygonset.layers, polygons_to_keep) if keep
                ]
                polygonset.datatypes = [
                    p for p, keep in zip(polygonset.datatypes, polygons_to_keep) if keep
                ]

            if include_labels:
                new_labels = []
                for l in D.labels:
                    original_layer = (l.layer, l.texttype)
                    original_layer = _parse_layer(original_layer)
                    if invert_selection:
                        keep_layer = original_layer in layers
                    else:
                        keep_layer = original_layer not in layers
                    if keep_layer:
                        new_labels += [l]
                D.labels = new_labels
        return self

    def distribute(
        self, elements="all", direction="x", spacing=100, separation=True, edge="center"
    ):
        """Distributes the specified elements in the Device.

        Parameters
        ----------
        elements : array-like of PHIDL objects or 'all'
            Elements to distribute.
        direction : {'x', 'y'}
            Direction of distribution; either a line in the x-direction or
            y-direction.
        spacing : int or float
            Distance between elements.
        separation : bool
            If True, guarantees elements are speparated with a fixed spacing
            between; if  False, elements are spaced evenly along a grid.
        edge : {'x', 'xmin', 'xmax', 'y', 'ymin', 'ymax'}
            Which edge to perform the distribution along (unused if
            separation == True)

        """
        if elements == "all":
            elements = self.polygons + self.references
        _distribute(
            elements=elements,
            direction=direction,
            spacing=spacing,
            separation=separation,
            edge=edge,
        )
        return self

    def align(self, elements="all", alignment="ymax"):
        """Align elements in the Device

        Parameters
        ----------
        elements : array-like of PHIDL objects, or 'all'
            Elements in the Device to align.
        alignment : {'x', 'y', 'xmin', 'xmax', 'ymin', 'ymax'}
            Which edge to align along (e.g. 'ymax' will move the elements such
            that all of their topmost points are aligned)

        """
        if elements == "all":
            elements = self.polygons + self.references
        _align(elements, alignment=alignment)
        return self

    def flatten(self, single_layer=None):
        """Flattens the heirarchy of the Device such that there are no longer
        any references to other Devices.  All polygons and labels from
        underlying references are copied and placed in the top-level Device.
        If single_layer is specified, all polygons are moved to that layer.

        Parameters
        ----------
        single_layer : None, int, tuple of int, or set of int
            If not None, all polygons are moved to the specified
        """
        if single_layer is None:
            super(Device, self).flatten(
                single_layer=None, single_datatype=None, single_texttype=None
            )
        else:
            gds_layer, gds_datatype = _parse_layer(single_layer)
            super(Device, self).flatten(
                single_layer=gds_layer,
                single_datatype=gds_datatype,
                single_texttype=gds_datatype,
            )

        temp_polygons = list(self.polygons)
        self.references = []
        self.polygons = []
        [self.add_polygon(poly) for poly in temp_polygons]
        return self

    def absorb(self, reference):
        """Flattens and absorbs polygons from an underlying DeviceReference
        into the Device, destroying the reference in the process but keeping
        the polygon geometry.

        Parameters
        ----------
        reference : DeviceReference
            DeviceReference to be absorbed into the Device.
        """
        if reference not in self.references:
            raise ValueError(
                """[PHIDL] Device.absorb() failed -
                the reference it was asked to absorb does not
                exist in this Device. """
            )
        ref_polygons = reference.get_polygons(by_spec=True)
        for (layer, polys) in ref_polygons.items():
            [self.add_polygon(points=p, layer=layer) for p in polys]
        self.remove(reference)
        return self

    def get_ports(self, depth=None):
        """Returns copies of all the ports of the Device, rotated and
        translated so that they're in their top-level position. The Ports
        returned are copies of the originals, but each copy has the same
        ``uid`` as the original so that they can be traced back to the
        original if needed.

        Parameters
        ----------
        depth : int or None
            If not None, defines from how many reference levels to
            retrieve Ports from.

        Returns
        -------
        port_list : list of Port
            List of all Ports in the Device.
        """
        port_list = [p._copy(new_uid=False) for p in self.ports.values()]

        if depth is None or depth > 0:
            for r in self.references:
                if depth is None:
                    new_depth = None
                else:
                    new_depth = depth - 1
                ref_ports = r.parent.get_ports(depth=new_depth)

                # Transform ports that came from a reference
                ref_ports_transformed = []
                for rp in ref_ports:
                    new_port = rp._copy(new_uid=False)
                    new_midpoint, new_orientation = r._transform_port(
                        rp.midpoint,
                        rp.orientation,
                        r.origin,
                        r.rotation,
                        r.x_reflection,
                    )
                    new_port.midpoint = new_midpoint
                    new_port.new_orientation = new_orientation
                    ref_ports_transformed.append(new_port)
                port_list += ref_ports_transformed

        return port_list

    def get_info(self):
        """Gathers the .info dictionaries from every sub-Device and returns
        them in a list.

        Parameters
        ----------
        depth : int or None
            If not None, defines from how many reference levels to
            retrieve Ports from.

        Returns
        -------
        list of dictionaries
            List of the ".info" property dictionaries from all sub-Devices
        """
        D_list = self.get_dependencies(recursive=True)
        info_list = []
        for D in D_list:
            info_list.append(D.info.copy())
        return info_list

    def remove(self, items):
        """Removes items from a Device, which can include Ports, PolygonSets,
        CellReferences, and Labels.

        Parameters
        ----------
        items : array-like[N]
            Items to be removed from the Device.
        """
        if not _is_iterable(items):
            items = [items]
        for item in items:
            if isinstance(item, Port):
                try:
                    self.ports = {k: v for k, v in self.ports.items() if v != item}
                except Exception:
                    raise ValueError(
                        """[PHIDL] Device.remove() cannot find the
                                     Port
                                     it was asked to remove in the Device:
                                     "%s"."""
                        % (item)
                    )
            else:
                try:
                    if isinstance(item, gdspy.PolygonSet):
                        self.polygons.remove(item)
                    if isinstance(item, gdspy.CellReference):
                        self.references.remove(item)
                    if isinstance(item, gdspy.Label):
                        self.labels.remove(item)
                    self.aliases = {k: v for k, v in self.aliases.items() if v != item}
                except Exception:
                    raise ValueError(
                        """[PHIDL] Device.remove() cannot find the
                                     item
                                     it was asked to remove in the Device:
                                     "%s"."""
                        % (item)
                    )

        self._bb_valid = False
        return self

    def rotate(self, angle=45, center=(0, 0)):
        """Rotates all Polygons in the Device around the specified
        center point.

        Parameters
        ----------
        angle : int or float
            Angle to rotate the Device in degrees.
        center : array-like[2] or None
            Midpoint of the Device.
        """
        if angle == 0:
            return self
        for e in self.polygons:
            e.rotate(angle=angle, center=center)
        for e in self.references:
            e.rotate(angle, center)
        for e in self.labels:
            e.rotate(angle, center)
        for p in self.ports.values():
            p.midpoint = _rotate_points(p.midpoint, angle, center)
            p.orientation = mod(p.orientation + angle, 360)
        self._bb_valid = False
        return self

    def move(self, origin=(0, 0), destination=None, axis=None):
        """Moves elements of the Device from the origin point to the
        destination. Both origin and destination can be 1x2 array-like, Port,
        or a key corresponding to one of the Ports in this Device.

        Parameters
        ----------
        origin : array-like[2], Port, or key
            Origin point of the move.
        destination : array-like[2], Port, or key
            Destination point of the move.
        axis : {'x', 'y'}
            Direction of the move.
        """
        dx, dy = _parse_move(origin, destination, axis)

        # Move geometries
        for e in self.polygons:
            e.translate(dx, dy)
        for e in self.references:
            e.move((dx, dy))
        for e in self.labels:
            e.move((dx, dy))
        for p in self.ports.values():
            p.midpoint = np.array(p.midpoint) + np.array((dx, dy))

        self._bb_valid = False
        return self

    def mirror(self, p1=(0, 1), p2=(0, 0)):
        """Mirrors a Device across the line formed between the two
        specified points. ``points`` may be input as either single points
        [1,2] or array-like[N][2], and will return in kind.

        Parameters
        ----------
        p1 : array-like[N][2]
            First point of the line.
        p2 : array-like[N][2]
            Second point of the line.
        """
        for e in self.polygons + self.references + self.labels:
            e.mirror(p1, p2)
        for p in self.ports.values():
            p.midpoint = _reflect_points(p.midpoint, p1, p2)
            phi = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / pi
            p.orientation = 2 * phi - p.orientation
        self._bb_valid = False
        return self

    def hash_geometry(self, precision=1e-4):
        """Computes an SHA1 hash of the geometry in the Device. For each layer,
        each polygon is individually hashed and then the polygon hashes are
        sorted, to ensure the hash stays constant regardless of the ordering
        the polygons.  Similarly, the layers are sorted by (layer, datatype)

        Parameters
        ----------
        precision : float
            Roudning precision for the the objects in the Device.  For instance,
            a precision of 1e-2 will round a point at (0.124, 1.748) to (0.12, 1.75)

        Returns
        -------
        str
            Hash result in the form of an SHA1 hex digest string

        Notes
        -----
        Algorithm:

        .. code-block:: python

            hash(
                hash(First layer information: [layer1, datatype1]),
                hash(Polygon 1 on layer 1 points: [(x1,y1),(x2,y2),(x3,y3)] ),
                hash(Polygon 2 on layer 1 points: [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] ),
                hash(Polygon 3 on layer 1 points: [(x1,y1),(x2,y2),(x3,y3)] ),
                hash(Second layer information: [layer2, datatype2]),
                hash(Polygon 1 on layer 2 points: [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] ),
                hash(Polygon 2 on layer 2 points: [(x1,y1),(x2,y2),(x3,y3)] ),
            )
        """
        polygons_by_spec = self.get_polygons(by_spec=True)
        layers = np.array(list(polygons_by_spec.keys()))
        sorted_layers = layers[np.lexsort((layers[:, 0], layers[:, 1]))]

        # A random offset which fixes common rounding errors intrinsic
        # to floating point math. Example: with a precision of 0.1, the
        # floating points 7.049999 and 7.050001 round to different values
        # (7.0 and 7.1), but offset values (7.220485 and 7.220487) don't
        magic_offset = 0.17048614

        final_hash = hashlib.sha1()
        for layer in sorted_layers:
            layer_hash = hashlib.sha1(layer.astype(np.int64)).digest()
            polygons = polygons_by_spec[tuple(layer)]
            polygons = [
                np.ascontiguousarray((p / precision) + magic_offset, dtype=np.int64)
                for p in polygons
            ]
            polygon_hashes = np.sort([hashlib.sha1(p).digest() for p in polygons])
            final_hash.update(layer_hash)
            for ph in polygon_hashes:
                final_hash.update(ph)

        return final_hash.hexdigest()


class DeviceReference(gdspy.CellReference, _GeometryHelper):
    """Simple reference to an existing Device.

    Parameters
    ----------
    device : Device
        The referenced Device.
    origin : array-like[2] of int or float
        Position where the Device is inserted.
    rotation : int or float
        Angle of rotation of the reference (in `degrees`)
    magnification : int or float
        Magnification factor for the reference.
    x_reflection : bool
        If True, the reference is reflected parallel to the x-direction before
        being rotated.
    """

    def __init__(
        self, device, origin=(0, 0), rotation=0, magnification=None, x_reflection=False
    ):
        super(DeviceReference, self).__init__(
            ref_cell=device,
            origin=origin,
            rotation=rotation,
            magnification=magnification,
            x_reflection=x_reflection,
            ignore_missing=False,
        )
        self.parent = device
        self.owner = None
        # The ports of a DeviceReference have their own unique id (uid),
        # since two DeviceReferences of the same parent Device can be
        # in different locations and thus do not represent the same port
        self._local_ports = {
            name: port._copy(new_uid=True) for name, port in device.ports.items()
        }

    def __repr__(self):
        """Prints a description of the DeviceReference, including parent
        Device, ports, origin, rotation, and x_reflection.
        """
        return (
            'DeviceReference (parent Device "%s", ports %s, origin %s, rotation %s, x_reflection %s)'
            % (
                self.parent.name,
                list(self.ports.keys()),
                self.origin,
                self.rotation,
                self.x_reflection,
            )
        )

    def __str__(self):
        """Prints a description of the DeviceReference, including parent
        Device, ports, origin, rotation, and x_reflection.
        """
        return self.__repr__()

    def __getitem__(self, val):
        """This allows you to access an alias from the reference's parent and
        receive a copy of the reference which is correctly rotated and
        translated.

        Parameters
        ----------
        val : str
            Alias from the reference's parent to be accessed.

        Returns
        -------
        new_reference : DeviceReference
            DeviceReference for the copied parent reference.
        """
        try:
            alias_device = self.parent[val]
        except KeyError:
            raise ValueError(
                '[PHIDL] Tried to access alias "%s" from parent '
                'Device "%s", which does not exist' % (val, self.parent.name)
            )
        new_reference = DeviceReference(
            alias_device.parent,
            origin=alias_device.origin,
            rotation=alias_device.rotation,
            magnification=alias_device.magnification,
            x_reflection=alias_device.x_reflection,
        )

        if self.x_reflection:
            new_reference.mirror((1, 0))
        if self.rotation is not None:
            new_reference.rotate(self.rotation)
        if self.origin is not None:
            new_reference.move(self.origin)

        return new_reference

    @property
    def ports(self):
        """This property allows you to access myref.ports, and receive a copy
        of the ports dict which is correctly rotated and translated."""
        for name, port in self.parent.ports.items():
            port = self.parent.ports[name]
            new_midpoint, new_orientation = self._transform_port(
                port.midpoint,
                port.orientation,
                self.origin,
                self.rotation,
                self.x_reflection,
            )
            if name not in self._local_ports:
                self._local_ports[name] = port._copy(new_uid=True)
            self._local_ports[name].midpoint = new_midpoint
            self._local_ports[name].orientation = mod(new_orientation, 360)
            self._local_ports[name].parent = self
        # Remove any ports that no longer exist in the reference's parent
        parent_names = self.parent.ports.keys()
        local_names = list(self._local_ports.keys())
        for name in local_names:
            if name not in parent_names:
                self._local_ports.pop(name)
        return self._local_ports

    @property
    def info(self):
        """Returns information about the properties of the reference's
        parent.
        """
        return self.parent.info

    @property
    def bbox(self):
        """Returns the bounding box of the DeviceReference."""
        bbox = self.get_bounding_box()
        if bbox is None:
            bbox = ((0, 0), (0, 0))
        return np.array(bbox)

    def _transform_port(
        self, point, orientation, origin=(0, 0), rotation=None, x_reflection=False
    ):
        """Applies various transformations to a Port.

        Parameters
        ----------
        point : array-like[N][2]
            Coordinates of the Port.
        orientation : int, float, or None
            Orientation of the Port
        origin : array-like[2] or None
            If given, shifts the transformed points to the specified origin.
        rotation : int, float, or None
            Angle of rotation to apply
        x_reflection : bool
            If True, reflects the Port across the x-axis before applying
            rotation.

        Returns
        -------
        new_point : array-like[N][2]
            Coordinates of the transformed Port.
        new_orientation : int, float, or None

        """
        # Apply GDS-type transformations to a port (x_ref)
        new_point = np.array(point)
        new_orientation = orientation

        if x_reflection:
            new_point[1] = -new_point[1]
            new_orientation = -orientation
        if rotation is not None:
            new_point = _rotate_points(new_point, angle=rotation, center=[0, 0])
            new_orientation += rotation
        if origin is not None:
            new_point = new_point + np.array(origin)
        new_orientation = mod(new_orientation, 360)

        return new_point, new_orientation

    def move(self, origin=(0, 0), destination=None, axis=None):
        """Moves the DeviceReference from the origin point to the
        destination. Both origin and destination can be 1x2 array-like,
        Port, or a key corresponding to one of the Ports in this
        DeviceReference.

        Parameters
        ----------
        origin : array-like[2], Port, or key
            Origin point of the move.
        destination : array-like[2], Port, or key
            Destination point of the move.
        axis : {'x', 'y'}
            Direction of move.
        """
        dx, dy = _parse_move(origin, destination, axis)
        self.origin = np.array(self.origin) + np.array((dx, dy))

        if self.owner is not None:
            self.owner._bb_valid = False
        return self

    def rotate(self, angle=45, center=(0, 0)):
        """Rotates all Polygons in the DeviceReference around the specified
        centerpoint.

        Parameters
        ----------
        angle : int or float
            Angle to rotate the DeviceReference in degrees.
        center : array-like[2] or None
            Midpoint of the DeviceReference.
        """
        if angle == 0:
            return self
        if type(center) is Port:
            center = center.midpoint
        self.rotation += angle
        self.origin = _rotate_points(self.origin, angle, center)

        if self.owner is not None:
            self.owner._bb_valid = False
        return self

    def mirror(self, p1=(0, 1), p2=(0, 0)):
        """Mirrors a DeviceReference across the line formed between the two
        specified points. ``points`` may be input as either single points
        [1,2] or array-like[N][2], and will return in kind.

        Parameters
        ----------
        p1 : array-like[N][2]
            First point of the line.
        p2 : array-like[N][2]
            Second point of the line.
        """
        if type(p1) is Port:
            p1 = p1.midpoint
        if type(p2) is Port:
            p2 = p2.midpoint
        p1 = np.array(p1)
        p2 = np.array(p2)
        # Translate so reflection axis passes through origin
        self.origin = self.origin - p1

        # Rotate so reflection axis aligns with x-axis
        angle = np.arctan2((p2[1] - p1[1]), (p2[0] - p1[0])) * 180 / pi
        self.origin = _rotate_points(self.origin, angle=-angle, center=[0, 0])
        self.rotation -= angle

        # Reflect across x-axis
        self.x_reflection = not self.x_reflection
        self.origin[1] = -self.origin[1]
        self.rotation = -self.rotation

        # Un-rotate and un-translate
        self.origin = _rotate_points(self.origin, angle=angle, center=[0, 0])
        self.rotation += angle
        self.origin = self.origin + p1

        if self.owner is not None:
            self.owner._bb_valid = False
        return self

    def reflect(self, p1=(0, 1), p2=(0, 0)):
        """
        .. deprecated:: 1.3.0
            `reflect` will be removed in May 2021, please replace with
            `mirror`.
        """
        warnings.warn(
            "[PHIDL] Warning: reflect() will be deprecated in "
            "May 2021, please replace with mirror()"
        )
        return self.mirror(p1, p2)

    def connect(self, port, destination, overlap=0):
        """Moves and rotates this object such that the the Port specified by
        `port` is connected (aligned and adjacent) with the Port specified by
        `destination`

        Parameters
        ----------
        port : str or Port
        destination : array-like[2]
        overlap : int or float

        """
        # ``port`` can either be a string with the name or an actual Port
        if port in self.ports:  # Then ``port`` is a key for the ports dict
            p = self.ports[port]
        elif type(port) is Port:
            p = port
        else:
            raise ValueError(
                "[PHIDL] connect() did not receive a Port or valid port name"
                + " - received (%s), ports available are (%s)"
                % (port, tuple(self.ports.keys()))
            )
        self.rotate(
            angle=180 + destination.orientation - p.orientation, center=p.midpoint
        )
        self.move(origin=p, destination=destination)
        self.move(
            -overlap
            * np.array(
                [
                    cos(destination.orientation * pi / 180),
                    sin(destination.orientation * pi / 180),
                ]
            )
        )
        return self


class CellArray(gdspy.CellArray, _GeometryHelper):
    """Multiple references to an existing cell in an array format.

    Parameters
    ----------
    device : Device
        The referenced Device.
    columns : int
        Number of columns in the array.
    rows : int
        Number of rows in the array.
    spacing : array-like[2] of int or float
        Distances between adjacent columns and adjacent rows.
    origin : array-like[2] of int or float
        Position where the cell is inserted.
    rotation : int or float
        Angle of rotation of the reference (in `degrees`).
    magnification : int or float
        Magnification factor for the reference.
    x_reflection : bool
        If True, the reference is reflected parallel to the x direction
        before being rotated.
    """

    def __init__(
        self,
        device,
        columns,
        rows,
        spacing,
        origin=(0, 0),
        rotation=0,
        magnification=None,
        x_reflection=False,
    ):
        super(CellArray, self).__init__(
            columns=columns,
            rows=rows,
            spacing=spacing,
            ref_cell=device,
            origin=origin,
            rotation=rotation,
            magnification=magnification,
            x_reflection=x_reflection,
            ignore_missing=False,
        )
        self.parent = device
        self.owner = None

    @property
    def bbox(self):
        """Returns the bounding box of the CellArray."""
        bbox = self.get_bounding_box()
        if bbox is None:
            bbox = ((0, 0), (0, 0))
        return np.array(bbox)

    def move(self, origin=(0, 0), destination=None, axis=None):
        """Moves the CellArray from the origin point to the destination. Both
        origin and destination can be 1x2 array-like, Port, or a key
        corresponding to one of the Ports in this CellArray.

        Parameters
        ----------
        origin : array-like[2], Port, or key
            Origin point of the move.
        destination : array-like[2], Port, or key
            Destination point of the move.
        axis : {'x', 'y'}
            Direction of the move.
        """
        dx, dy = _parse_move(origin, destination, axis)
        self.origin = np.array(self.origin) + np.array((dx, dy))

        if self.owner is not None:
            self.owner._bb_valid = False
        return self

    def rotate(self, angle=45, center=(0, 0)):
        """Rotates all elements in the CellArray around the specified
        centerpoint.

        Parameters
        ----------
        angle : int or float
            Angle to rotate the CellArray in degrees.
        center : array-like[2], Port, or None
            Midpoint of the CellArray.
        """
        if angle == 0:
            return self
        if type(center) is Port:
            center = center.midpoint
        self.rotation += angle
        self.origin = _rotate_points(self.origin, angle, center)
        if self.owner is not None:
            self.owner._bb_valid = False
        return self

    def mirror(self, p1=(0, 1), p2=(0, 0)):
        """Mirrors a CellArray across the line formed between the two
        specified points.

        Parameters
        ----------
        p1 : array-like[N][2]
            First point of the line.
        p2 : array-like[N][2]
            Second point of the line.
        """
        if type(p1) is Port:
            p1 = p1.midpoint
        if type(p2) is Port:
            p2 = p2.midpoint
        p1 = np.array(p1)
        p2 = np.array(p2)
        # Translate so reflection axis passes through origin
        self.origin = self.origin - p1

        # Rotate so reflection axis aligns with x-axis
        angle = np.arctan2((p2[1] - p1[1]), (p2[0] - p1[0])) * 180 / pi
        self.origin = _rotate_points(self.origin, angle=-angle, center=[0, 0])
        self.rotation -= angle

        # Reflect across x-axis
        self.x_reflection = not self.x_reflection
        self.origin[1] = -self.origin[1]
        self.rotation = -self.rotation

        # Un-rotate and un-translate
        self.origin = _rotate_points(self.origin, angle=angle, center=[0, 0])
        self.rotation += angle
        self.origin = self.origin + p1

        if self.owner is not None:
            self.owner._bb_valid = False
        return self

    def reflect(self, p1=(0, 1), p2=(0, 0)):
        """
        .. deprecated:: 1.3.0
            `reflect` will be removed in May 2021, please replace with
            `mirror`.
        """
        warnings.warn(
            "[PHIDL] Warning: reflect() will be deprecated in "
            "May 2021, please replace with mirror()"
        )
        return self.mirror(p1, p2)


class Label(gdspy.Label, _GeometryHelper):
    """Text that can be used to label parts of the geometry or display
    messages. The text does not create additional geometry, it’s meant for
    display and labeling purposes only.
    """

    def __init__(self, *args, **kwargs):
        super(Label, self).__init__(*args, **kwargs)
        self.position = np.array(self.position, dtype="float64")

    @property
    def bbox(self):
        """Returns the bounding box of the Label."""
        return np.array(
            [[self.position[0], self.position[1]], [self.position[0], self.position[1]]]
        )

    def rotate(self, angle=45, center=(0, 0)):
        """Rotates Label around the specified centerpoint.

        Parameters
        ----------
        angle : int or float
            Angle to rotate the Label in degrees.
        center : array-like[2] or None
            Midpoint of the Label.
        """
        self.position = _rotate_points(self.position, angle=angle, center=center)
        return self

    def move(self, origin=(0, 0), destination=None, axis=None):
        """Moves the Label from the origin point to the destination. Both
        origin and destination can be 1x2 array-like, Port, or a key
        corresponding to one of the Ports in this Label.

        Parameters
        ----------
        origin : array-like[2], Port, or key
            Origin point of the move.
        destination : array-like[2], Port, or key
            Destination point of the move.
        axis : {'x', 'y'}
            Direction of the move.
        """
        dx, dy = _parse_move(origin, destination, axis)
        self.position += np.asarray((dx, dy))
        return self

    def mirror(self, p1=(0, 1), p2=(0, 0)):
        """Mirrors a Label across the line formed between the two
        specified points. ``points`` may be input as either single points
        [1,2] or array-like[N][2], and will return in kind.

        Parameters
        ----------
        p1 : array-like[N][2]
            First point of the line.
        p2 : array-like[N][2]
            Second point of the line.
        """
        self.position = _reflect_points(self.position, p1, p2)
        return self

    def reflect(self, p1=(0, 1), p2=(0, 0)):
        """
        .. deprecated:: 1.3.0
            `reflect` will be removed in May 2021, please replace with
            `mirror`.
        """
        warnings.warn(
            "[PHIDL] Warning: reflect() will be deprecated in "
            "May 2021, please replace with mirror()"
        )
        return self.mirror(p1, p2)


class Group(_GeometryHelper):
    """Groups objects together so they can be manipulated as though
    they were a single object (move/rotate/mirror)."""

    def __init__(self, *args):
        self.elements = []
        self.add(args)

    def __repr__(self):
        """Prints the number of elements in the Group."""
        return "Group (%s elements total)" % (len(self.elements))

    def __len__(self):
        """Returns the number of elements in the Group."""
        return len(self.elements)

    def __iadd__(self, element):
        """Adds an element to the Group.

        Parameters
        ----------
        element : Device, DeviceReference, Port, Polygon, CellArray, Label, or Group
            Element to be added.
        """
        return self.add(element)

    @property
    def bbox(self):
        """Returns the bounding boxes of the Group."""
        if len(self.elements) == 0:
            raise ValueError("[PHIDL] Group is empty, no bbox is available")
        bboxes = np.empty([len(self.elements), 4])
        for n, e in enumerate(self.elements):
            bboxes[n] = e.bbox.flatten()

        bbox = (
            (bboxes[:, 0].min(), bboxes[:, 1].min()),
            (bboxes[:, 2].max(), bboxes[:, 3].max()),
        )
        return np.array(bbox)

    def add(self, element):
        """Adds an element to the Group.

        Parameters
        ----------
        element : Device, DeviceReference, Port, Polygon, CellArray, Label, or Group
            Element to add.
        """
        if _is_iterable(element):
            [self.add(e) for e in element]
        elif element is None:
            return self
        elif isinstance(element, PHIDL_ELEMENTS):
            self.elements.append(element)
        else:
            raise ValueError(
                "[PHIDL] add() Could not add element to Group, the only "
                "allowed element types are "
                "(Device, DeviceReference, Polygon, CellArray, Label, Group)"
            )
        # Remove non-unique entries
        used = set()
        self.elements = [
            x for x in self.elements if x not in used and (used.add(x) or True)
        ]
        return self

    def rotate(self, angle=45, center=(0, 0)):
        """Rotates all elements in a Group around the specified centerpoint.

        Parameters
        ----------
        angle : int or float
            Angle to rotate the Group in degrees.
        center : array-like[2] or None
            Midpoint of the Group.
        """
        for e in self.elements:
            e.rotate(angle=angle, center=center)
        return self

    def move(self, origin=(0, 0), destination=None, axis=None):
        """Moves the Group from the origin point to the destination. Both
        origin and destination can be 1x2 array-like, Port, or a key
        corresponding to one of the Ports in this Group.

        Parameters
        ----------
        origin : array-like[2], Port, or key
            Origin point of the move.
        destination : array-like[2], Port, or key
            Destination point of the move.
        axis : {'x', 'y'}
            Direction of the move.
        """
        for e in self.elements:
            e.move(origin=origin, destination=destination, axis=axis)
        return self

    def mirror(self, p1=(0, 1), p2=(0, 0)):
        """Mirrors a Group across the line formed between the two
        specified points. ``points`` may be input as either single points
        [1,2] or array-like[N][2], and will return in kind.

        Parameters
        ----------
        p1 : array-like[N][2]
            First point of the line.
        p2 : array-like[N][2]
            Second point of the line.
        """
        for e in self.elements:
            e.mirror(p1=p1, p2=p2)
        return self

    def distribute(self, direction="x", spacing=100, separation=True, edge="center"):
        """Distributes the elements in the Group.

        Parameters
        ----------
        direction : {'x', 'y'}
            Direction of distribution; either a line in the x-direction or
            y-direction.
        spacing : int or float
            Distance between elements.
        separation : bool
            If True, guarantees elements are speparated with a fixed spacing
            between; if False, elements are spaced evenly along a grid.
        edge : {'x', 'xmin', 'xmax', 'y', 'ymin', 'ymax'}
            Which edge to perform the distribution along (unused if
            separation == True)
        """
        _distribute(
            elements=self.elements,
            direction=direction,
            spacing=spacing,
            separation=separation,
            edge=edge,
        )
        return self

    def align(self, alignment="ymax"):
        """Aligns the elements in the Group.

        Parameters
        ----------
        alignment : {'x', 'y', 'xmin', 'xmax', 'ymin', 'ymax'}
            Which edge to align along (e.g. 'ymax' will align move the elements
            such that all of their topmost points are aligned)
        """
        _align(elements=self.elements, alignment=alignment)
        return self


def _linear_transition(y1, y2):
    dx = y2 - y1
    return lambda t: y1 + t * dx


PHIDL_ELEMENTS = (Device, DeviceReference, Polygon, CellArray, Label, Group)


class Path(_GeometryHelper):
    """The Path object for making smooth Paths.  To be used in combination
    with a CrossSection to create a Device.

    Parameters
    ----------
    path : array-like[N][2], Path, or list of Paths
        Points or Paths to append() initially
    """

    def __init__(self, path=None):
        self.points = np.array([[0, 0]], dtype=np.float64)
        self.start_angle = 0
        self.end_angle = 0
        self.info = {}
        if path is not None:
            # If array[N][2]
            if (
                (np.ndim(path) == 2)
                and np.issubdtype(np.array(path).dtype, np.number)
                and (np.shape(path)[1] == 2)
            ):
                self.points = np.array(path, dtype=np.float64)
                nx1, ny1 = self.points[1] - self.points[0]
                self.start_angle = np.arctan2(ny1, nx1) / np.pi * 180
                nx2, ny2 = self.points[-1] - self.points[-2]
                self.end_angle = np.arctan2(ny2, nx2) / np.pi * 180
            elif isinstance(path, Path):
                self.points = np.array(path.points, dtype=np.float64)
                self.start_angle = path.start_angle
                self.end_angle = path.end_angle
                self.info = {}
            elif np.size(path) > 1:
                self.append(path)
            else:
                raise ValueError(
                    '[PHIDL] Path() the "path" argument must be either '
                    + "blank, a Path object, an array-like[N][2] list of points, or a list of these"
                )

    def __len__(self):
        return len(self.points)

    @property
    def bbox(self):
        """Returns the bounding box of the Path."""
        bbox = [
            (np.min(self.points[:, 0]), np.min(self.points[:, 1])),
            (np.max(self.points[:, 0]), np.max(self.points[:, 1])),
        ]
        return np.array(bbox)

    def append(self, path):
        """Attaches the input path to the end of this object.  The input path
        will be automatically rotated and translated such that it continues
        smoothly from the previous segment.

        Parameters
        ----------
        path : Path, array-like[N][2], or list of Paths
            The input path that will be appended
        """
        # If appending another Path, load relevant variables
        if isinstance(path, Path):
            start_angle = path.start_angle
            end_angle = path.end_angle
            points = path.points
        # If array[N][2]
        elif (
            (np.ndim(path) == 2)
            and np.issubdtype(np.array(path).dtype, np.number)
            and (np.shape(path)[1] == 2)
        ):
            points = np.asfarray(path)
            nx1, ny1 = points[1] - points[0]
            start_angle = np.arctan2(ny1, nx1) / np.pi * 180
            nx2, ny2 = points[-1] - points[-2]
            end_angle = np.arctan2(ny2, nx2) / np.pi * 180
        # If list of Paths or arrays
        elif isinstance(path, (list, tuple)):
            for p in path:
                self.append(p)
            return self
        else:
            raise ValueError(
                '[PHIDL] Path.append() the "path" argument must be either '
                + "a Path object, an array-like[N][2] list of points, or a list of these"
            )

        # Connect beginning of new points with old points
        points = _rotate_points(points, angle=self.end_angle - start_angle)
        points += self.points[-1, :] - points[0, :]

        # Update end angle
        self.end_angle = mod(end_angle + self.end_angle - start_angle, 360)

        # Concatenate old points + new points
        self.points = np.vstack([self.points, points[1:]])

        return self

    def extrude(self, width, layer=np.nan, simplify=None):
        """Combines the 1D Path with a 1D cross-section to form 2D polygons.

        Parameters
        ----------
        width : int, float, array-like[2], or CrossSection
            If set to a single number (e.g. `width=1.7`): makes a constant-width extrusion
            If set to a 2-element array (e.g. `width=[1.8,2.5]`): makes an extrusion
                whose width varies linearly from width[0] to width[1]
            If set to a CrossSection: uses the CrossSection parameters for extrusion
        layer : int, tuple of int, or set of int
            The layer to put the extruded polygons on. `layer=0` is used by default.
        simplify : float
            Tolerance value for the simplification algorithm.  All points that
            can be removed without changing the resulting polygon by more than
            the value listed here will be removed. Also known as `epsilon` here
            https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm

        Returns
        -------
        Device
            A Device with polygons added that correspond to the extrusion of the
            Path
        """

        if isinstance(width, CrossSection) and (layer is not np.nan):
            raise ValueError(
                """[PHIDL] extrude(): when using a CrossSection as the
                `width` argument cannot also define the `layer` argument"""
            )
        if not isinstance(width, CrossSection) and (layer is np.nan):
            layer = 0
        if isinstance(width, CrossSection):
            X = width
        elif np.size(width) == 1:
            X = CrossSection()
            X.add(width=width, layer=layer)
        elif np.size(width) == 2:
            X = CrossSection()
            X.add(width=_linear_transition(width[0], width[1]), layer=layer)
        else:
            raise ValueError(
                """[PHIDL] extrude(): width argument must be one of
                int, float, array-like[2], or CrossSection"""
            )

        D = Device("extrude")
        for section in X.sections:
            width = section["width"]
            offset = section["offset"]
            layer = section["layer"]
            ports = section["ports"]

            if callable(offset):
                P_offset = self.copy()
                P_offset.offset(offset)
                points = P_offset.points
                start_angle = P_offset.start_angle
                end_angle = P_offset.end_angle
                offset = 0
            else:
                points = self.points
                start_angle = self.start_angle
                end_angle = self.end_angle

            if callable(width):
                # Compute lengths
                dx = np.diff(self.points[:, 0])
                dy = np.diff(self.points[:, 1])
                lengths = np.cumsum(np.sqrt((dx) ** 2 + (dy) ** 2))
                lengths = np.concatenate([[0], lengths])
                width = width(lengths / lengths[-1])
            else:
                pass

            points1 = self._centerpoint_offset_curve(
                points,
                offset_distance=offset + width / 2,
                start_angle=start_angle,
                end_angle=end_angle,
            )
            points2 = self._centerpoint_offset_curve(
                points,
                offset_distance=offset - width / 2,
                start_angle=start_angle,
                end_angle=end_angle,
            )

            # Simplify lines using the Ramer–Douglas–Peucker algorithm
            if isinstance(simplify, bool):
                raise ValueError(
                    "[PHIDL] the simplify argument must be a number (e.g. 1e-3) or None"
                )
            if simplify is not None:
                points1 = _simplify(points1, tolerance=simplify)
                points2 = _simplify(points2, tolerance=simplify)

            # Join points together
            points = np.concatenate([points1, points2[::-1, :]])

            # Combine the offset-lines into a polygon and union if join_after == True
            # if join_after == True: # Use clipper to perform a union operation
            #     points = np.array(clipper.offset([points], 0, 'miter', 2, int(1/simplify), 0)[0])

            D.add_polygon(points, layer=layer)

            # Add ports if they were specified
            if ports[0] is not None:
                new_port = D.add_port(name=ports[0])
                new_port.endpoints = (points1[0], points2[0])
            if ports[1] is not None:
                new_port = D.add_port(name=ports[1])
                new_port.endpoints = (points2[-1], points1[-1])

        return D

    def offset(self, offset=0):
        """Offsets the Path so that it follows the Path centerline plus
        an offset.  The offset can either be a fixed value, or a function
        of the form my_offset(t) where t goes from 0->1

        Parameters
        ----------
        offset : int or float, callable
            Magnitude of the offset
        """
        if offset == 0:
            points = self.points
            start_angle = self.start_angle
            end_angle = self.end_angle
        elif callable(offset):
            # Compute lengths
            dx = np.diff(self.points[:, 0])
            dy = np.diff(self.points[:, 1])
            lengths = np.cumsum(np.sqrt((dx) ** 2 + (dy) ** 2))
            lengths = np.concatenate([[0], lengths])
            # Create list of offset points and perform offset
            points = self._centerpoint_offset_curve(
                self.points,
                offset_distance=offset(lengths / lengths[-1]),
                start_angle=self.start_angle,
                end_angle=self.end_angle,
            )
            # Numerically compute start and end angles
            tol = 1e-6
            ds = tol / lengths[-1]
            ny1 = offset(ds) - offset(0)
            start_angle = np.arctan2(-ny1, tol) / np.pi * 180 + self.start_angle
            start_angle = np.round(start_angle, decimals=6)
            ny2 = offset(1) - offset(1 - ds)
            end_angle = np.arctan2(-ny2, tol) / np.pi * 180 + self.end_angle
            end_angle = np.round(end_angle, decimals=6)
        else:  # Offset is just a number
            points = self._centerpoint_offset_curve(
                self.points,
                offset_distance=offset,
                start_angle=self.start_angle,
                end_angle=self.end_angle,
            )
            start_angle = self.start_angle
            end_angle = self.end_angle

        self.points = points
        self.start_angle = start_angle
        self.end_angle = end_angle
        return self

    def copy(self):
        """Creates a copy of the Path.

        Returns
        -------
        Path
            A copy of the Path

        """
        P = Path()
        P.info = self.info.copy()
        P.points = np.array(self.points)
        P.start_angle = self.start_angle
        P.end_angle = self.end_angle
        return P

    def move(self, origin=(0, 0), destination=None, axis=None):
        """Moves the Path from the origin point to the
        destination. Both origin and destination can be 1x2 array-like
        or a Port.

        Parameters
        ----------
        origin : array-like[2], Port
            Origin point of the move.
        destination : array-like[2], Port
            Destination point of the move.
        axis : {'x', 'y'}
            Direction of move.

        """
        dx, dy = _parse_move(origin, destination, axis)
        self.points += np.array([dx, dy])

        return self

    def rotate(self, angle=45, center=(0, 0)):
        """Rotates all Polygons in the Device around the specified
        center point. If no center point specified will rotate around (0,0).

        Parameters
        ----------
        angle : int or float
            Angle to rotate the Device in degrees.
        center : array-like[2] or None
            Midpoint of the Device.
        """
        if angle == 0:
            return self
        self.points = _rotate_points(self.points, angle, center)
        if self.start_angle is not None:
            self.start_angle = mod(self.start_angle + angle, 360)
        if self.end_angle is not None:
            self.end_angle = mod(self.end_angle + angle, 360)
        return self

    def mirror(self, p1=(0, 1), p2=(0, 0)):
        """Mirrors the Path across the line formed between the two
        specified points. ``points`` may be input as either single points
        [1,2] or array-like[N][2], and will return in kind.

        Parameters
        ----------
        p1 : array-like[N][2]
            First point of the line.
        p2 : array-like[N][2]
            Second point of the line.
        """
        self.points = _reflect_points(self.points, p1, p2)
        angle = np.arctan2((p2[1] - p1[1]), (p2[0] - p1[0])) * 180 / pi
        if self.start_angle is not None:
            self.start_angle = mod(2 * angle - self.start_angle, 360)
        if self.end_angle is not None:
            self.end_angle = mod(2 * angle - self.end_angle, 360)
        return self

    def _centerpoint_offset_curve(
        self, points, offset_distance, start_angle, end_angle
    ):
        """Creates a offset curve (but does not account for cusps etc)
        by computing the centerpoint offset of the supplied x and y points"""
        new_points = np.array(points, dtype=np.float64)
        dx = np.diff(points[:, 0])
        dy = np.diff(points[:, 1])
        theta = np.arctan2(dy, dx)
        theta = np.concatenate([theta[0:1], theta, theta[-1:]])
        theta_mid = (np.pi + theta[1:] + theta[:-1]) / 2  # Mean angle between segments
        dtheta_int = np.pi + theta[:-1] - theta[1:]  # Internal angle between segments
        offset_distance = offset_distance / np.sin(dtheta_int / 2)
        new_points[:, 0] -= offset_distance * np.cos(theta_mid)
        new_points[:, 1] -= offset_distance * np.sin(theta_mid)
        if start_angle is not None:
            new_points[0, :] = points[0, :] + (
                np.sin(start_angle * np.pi / 180) * offset_distance[0],
                -np.cos(start_angle * np.pi / 180) * offset_distance[0],
            )
        if end_angle is not None:
            new_points[-1, :] = points[-1, :] + (
                np.sin(end_angle * np.pi / 180) * offset_distance[-1],
                -np.cos(end_angle * np.pi / 180) * offset_distance[-1],
            )
        return new_points

    def _parametric_offset_curve(self, points, offset_distance, start_angle, end_angle):
        """Creates a parametric offset (does not account for cusps etc)
        by using gradient of the supplied x and y points"""
        x = points[:, 0]
        y = points[:, 1]
        dxdt = np.gradient(x)
        dydt = np.gradient(y)
        if start_angle is not None:
            dxdt[0] = np.cos(start_angle * np.pi / 180)
            dydt[0] = np.sin(start_angle * np.pi / 180)
        if end_angle is not None:
            dxdt[-1] = np.cos(end_angle * np.pi / 180)
            dydt[-1] = np.sin(end_angle * np.pi / 180)
        x_offset = x + offset_distance * dydt / np.sqrt(dxdt**2 + dydt**2)
        y_offset = y - offset_distance * dxdt / np.sqrt(dydt**2 + dxdt**2)
        return np.array([x_offset, y_offset]).T

    def length(self):
        """Computes the cumulative length (arc length) of the path.

        Returns
        -------
        float
            The length of the Path
        """
        x = self.points[:, 0]
        y = self.points[:, 1]
        dx = np.diff(x)
        dy = np.diff(y)
        return np.sum(np.sqrt((dx) ** 2 + (dy) ** 2))

    def curvature(self):
        """Calculates the curvature of the Path. Note this curvature is
        numerically computed so areas where the curvature jumps instantaneously
        (such as between an arc and a straight segment) will be slightly
        interpolated, and sudden changes in point density along the curve can
        cause discontinuities.

        Returns
        -------
        s : array-like[N]
            The arc-length of the Path
        K : array-like[N]
            The curvature of the Path
        """
        x = self.points[:, 0]
        y = self.points[:, 1]
        dx = np.diff(x)
        dy = np.diff(y)
        ds = np.sqrt((dx) ** 2 + (dy) ** 2)
        s = np.cumsum(ds)
        theta = np.arctan2(dy, dx)

        # Fix discontinuities arising from np.arctan2
        dtheta = np.diff(theta)
        dtheta[np.where(dtheta > np.pi)] += -2 * np.pi
        dtheta[np.where(dtheta < -np.pi)] += 2 * np.pi
        theta = np.concatenate([[0], np.cumsum(dtheta)]) + theta[0]

        K = np.gradient(theta, s, edge_order=2)
        return s, K

    def hash_geometry(self, precision=1e-4):
        """Computes an SHA1 hash of the points in the Path and the start_angle
        and end_angle

        Parameters
        ----------
        precision : float
            Roudning precision for the the objects in the Device.  For instance,
            a precision of 1e-2 will round a point at (0.124, 1.748) to (0.12, 1.75)

        Returns
        -------
        str
            Hash result in the form of an SHA1 hex digest string

        Notes
        -----
        Algorithm:

        .. code-block:: python

            hash(
                hash(First layer information: [layer1, datatype1]),
                hash(Polygon 1 on layer 1 points: [(x1,y1),(x2,y2),(x3,y3)] ),
                hash(Polygon 2 on layer 1 points: [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] ),
                hash(Polygon 3 on layer 1 points: [(x1,y1),(x2,y2),(x3,y3)] ),
                hash(Second layer information: [layer2, datatype2]),
                hash(Polygon 1 on layer 2 points: [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] ),
                hash(Polygon 2 on layer 2 points: [(x1,y1),(x2,y2),(x3,y3)] ),
            )
        """
        # A random offset which fixes common rounding errors intrinsic
        # to floating point math. Example: with a precision of 0.1, the
        # floating points 7.049999 and 7.050001 round to different values
        # (7.0 and 7.1), but offset values (7.220485 and 7.220487) don't
        magic_offset = 0.17048614

        final_hash = hashlib.sha1()
        p = np.ascontiguousarray(
            (self.points / precision) + magic_offset, dtype=np.int64
        )
        final_hash.update(p)
        p = np.ascontiguousarray((self.start_angle, self.end_angle), dtype=np.float64)
        final_hash.update(p)
        return final_hash.hexdigest()


class CrossSection(object):
    """The CrossSection object for extruding a Path.  To be used in
    combination with a Path to create a Device.

    Parameters
    ----------
    path : array-like[N][2], Path, or list of Paths
        Points or Paths to append() initially
    """

    def __init__(self):
        self.sections = []
        self.ports = set()
        self.aliases = {}
        self.info = {}

    def add(self, width=1, offset=0, layer=0, ports=(None, None), name=None):
        """Adds a cross-sectional element to the CrossSection.  If ports are
        specified, when creating a Device with the extrude() command there be
        have Ports at the ends.

        Parameters
        ----------
        width : float
            Width of the segment
        offset : float
            Offset of the segment (positive values = right hand side)
        layer : int, tuple of int, or set of int
            The polygon layer to put the segment on
        ports : array-like[2] of str, int, or None
            If not None, specifies the names for the ports at the ends of the
            cross-sectional element
        name : str, int, or None
            Name of the cross-sectional element for later access
        """
        if isinstance(width, (float, int)) and (width <= 0):
            raise ValueError("[PHIDL] CrossSection.add(): widths must be >0")
        if len(ports) != 2:
            raise ValueError("[PHIDL] CrossSection.add(): must receive 2 port names")
        for p in ports:
            if p in self.ports:
                raise ValueError(
                    '[PHIDL] CrossSection.add(): a port named "%s" already '
                    "exists in this CrossSection, please rename port" % p
                )
        if name in self.aliases:
            raise ValueError(
                '[PHIDL] CrossSection.add(): an element named "%s" already '
                "exists in this CrossSection, please change the name" % name
            )

        new_segment = dict(
            width=width,
            offset=offset,
            layer=layer,
            ports=ports,
        )

        if name is not None:
            self.aliases[name] = new_segment
        self.sections.append(new_segment)
        [self.ports.add(p) for p in ports if p is not None]

        return self

    def extrude(self, path, simplify=None):
        """Combines the 1D CrossSection with a 1D Path to form 2D polygons.

        Parameters
        ----------
        path : Path
            The Path for the CrossSection to follow
        simplify : float
            Tolerance value for the simplification algorithm.  All points that
            can be removed without changing the resulting polygon by more than
            the value listed here will be removed. Also known as `epsilon` here
            https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm

        Returns
        -------
        Device
            A Device with polygons added that correspond to the extrusion of the
            Path with the CrossSection
        """

        D = path.extrude(width=self, simplify=simplify)
        return D

    def copy(self):
        """Creates a copy of the CrosSection.

        Returns
        -------
        CrossSection
            A copy of the CrossSection

        """
        X = CrossSection()
        X.info = self.info.copy()
        X.sections = list(self.sections)
        X.ports = set(self.ports)
        X.aliases = dict(self.aliases)
        return X

    def __getitem__(self, key):
        """Allows access to cross-sectional elements by name like X['etch2'].

        Parameters
        ----------
        key : str
            Element name to access within the CrossSection.
        """
        try:
            return self.aliases[key]
        except KeyError:
            raise ValueError(
                '[PHIDL] Tried to access name "%s" in CrossSection '
                "which does not exist" % (key)
            )
