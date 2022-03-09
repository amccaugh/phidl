from __future__ import absolute_import, division, print_function

import warnings

import gdspy
import numpy as np
from numpy import cos, mod, pi, sin
from numpy.linalg import norm

import phidl.path as pp
from phidl import Path
from phidl.device_layout import CrossSection, Device, Port
from phidl.geometry import turn


def _get_rotated_basis(theta):
    """Returns basis vectors rotated CCW by theta (in degrees)"""
    theta = np.radians(theta)
    e1 = np.array([np.cos(theta), np.sin(theta)])
    e2 = np.array([-1 * np.sin(theta), np.cos(theta)])
    return e1, e2


def _arc(radius=10, width=0.5, theta=45, start_angle=0, angle_resolution=2.5, layer=0):
    """Creates an arc of arclength ``theta`` starting at angle ``start_angle``"""
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


def _gradual_bend(
    radius=20,
    width=1.0,
    angular_coverage=15,
    num_steps=10,
    angle_resolution=0.1,
    start_angle=0,
    direction="ccw",
    layer=0,
):

    """
    creates a 90-degree bent waveguide
    the bending radius is gradually increased until it reaches the minimum
    value of the radius at the "angular coverage" angle.
    it essentially creates a smooth transition to a bent waveguide mode.
    user can control number of steps provided.
    direction determined by start angle and cw or ccw switch
    ############
    with the default 10 "num_steps" and 15 degree coverage, effective radius is about 1.5*radius.
    """
    angular_coverage = np.deg2rad(angular_coverage)
    D = Device()

    # determines the increment in radius through its inverse from 0 to 1/r
    inc_rad = (radius**-1) / (num_steps)
    angle_step = angular_coverage / num_steps

    # construct a series of sub-arcs with equal angles but gradually decreasing bend radius
    arcs = []
    for i in range(num_steps):
        A = _arc(
            radius=1 / ((i + 1) * inc_rad),
            width=width,
            theta=np.rad2deg(angle_step),
            start_angle=i * np.rad2deg(angle_step),
            angle_resolution=angle_resolution,
            layer=layer,
        )
        a = D.add_ref(A)
        arcs.append(a)
        if i > 0:
            a.connect(port=1, destination=prevPort)  # noqa: F821
        prevPort = a.ports[2]
    D.add_port(name=1, port=arcs[0].ports[1])

    # now connect a regular bend for the normal curved portion
    B = _arc(
        radius=radius,
        width=width,
        theta=45 - np.rad2deg(angular_coverage),
        start_angle=angular_coverage,
        angle_resolution=angle_resolution,
        layer=layer,
    )
    b = D.add_ref(B)
    b.connect(port=1, destination=prevPort)
    prevPort = b.ports[2]
    D.add_port(name=2, port=prevPort)

    # now create the overall structure
    Total = Device()

    # clone the half-curve into two objects and connect for a 90 deg bend.
    D1 = Total.add_ref(D)
    D2 = Total.add_ref(D)
    D2.mirror(p1=[0, 0], p2=[1, 1])
    D2.connect(port=2, destination=D1.ports[2])
    Total.xmin = 0
    Total.ymin = 0

    # orient to default settings...
    Total.mirror(p1=[0, 0], p2=[1, 1])
    Total.mirror(p1=[0, 0], p2=[1, 0])

    # orient to user-provided settings
    if direction == "cw":
        Total.mirror(p1=[0, 0], p2=[1, 0])
    Total.rotate(angle=start_angle, center=Total.center)
    Total.center = [0, 0]
    Total.add_port(name=1, port=D1.ports[1])
    Total.add_port(name=2, port=D2.ports[1])

    return Total


def route_basic(
    port1,
    port2,
    path_type="sine",
    width_type="straight",
    width1=None,
    width2=None,
    num_path_pts=99,
    layer=0,
):
    """
    .. deprecated:: 1.6.0
        `route_basic` will be removed in August 2022, please replace with
        `route_quad()` or `route_smooth()`.
    """
    warnings.warn(
        """[PHIDL] Warning: route_basic() will be deprecated
    in August 2022, please replace with `route_quad()` or `route_smooth()`"""
    )
    # Assuming they're both Ports for now
    point_a = np.array(port1.midpoint)
    if width1 is None:
        width1 = port1.width
    point_b = np.array(port2.midpoint)
    if width2 is None:
        width2 = port2.width
    if round(abs(mod(port1.orientation - port2.orientation, 360)), 3) != 180:
        raise ValueError(
            "[DEVICE] route() error: Ports do not face each other (orientations must be 180 apart)"
        )
    orientation = port1.orientation

    separation = point_b - point_a  # Vector drawn from A to B
    distance = norm(separation)  # Magnitude of vector from A to B
    rotation = (
        np.arctan2(separation[1], separation[0]) * 180 / pi
    )  # Rotation of vector from A to B
    angle = (
        rotation - orientation
    )  # If looking out along the normal of ``a``, the angle you would have to look to see ``b``
    forward_distance = distance * cos(angle * pi / 180)
    lateral_distance = distance * sin(angle * pi / 180)

    # Create a path assuming starting at the origin and setting orientation = 0
    # use the "connect" function later to move the path to the correct location
    xf = forward_distance
    yf = lateral_distance
    if path_type == "straight":
        curve_fun = lambda t: [xf * t, yf * t]
        curve_deriv_fun = lambda t: [xf + t * 0, t * 0]
    if path_type == "sine":
        curve_fun = lambda t: [xf * t, yf * (1 - cos(t * pi)) / 2]
        curve_deriv_fun = lambda t: [xf + t * 0, yf * (sin(t * pi) * pi) / 2]
    # if path_type == 'semicircle':
    #    def semicircle(t):
    #        t = np.array(t)
    #        x,y = np.zeros(t.shape), np.zeros(t.shape)
    #        ii = (0 <= t) & (t <= 0.5)
    #        jj = (0.5 < t) & (t <= 1)
    #        x[ii] = (cos(-pi/2 + t[ii]*pi/2))*xf
    #        y[ii] = (sin(-pi/2 + t[ii]*pi/2)+1)*yf*2
    #        x[jj] = (cos(pi*3/2 - t[jj]*pi)+2)*xf/2
    #        y[jj] = (sin(pi*3/2 - t[jj]*pi)+1)*yf/2
    #        return x,y
    #    curve_fun = semicircle
    #    curve_deriv_fun = None
    if width_type == "straight":
        width_fun = lambda t: (width2 - width1) * t + width1
    if width_type == "sine":
        width_fun = lambda t: (width2 - width1) * (1 - cos(t * pi)) / 2 + width1

    route_path = gdspy.Path(width=width1, initial_point=(0, 0))
    route_path.parametric(
        curve_fun,
        curve_deriv_fun,
        number_of_evaluations=num_path_pts,
        max_points=199,
        final_width=width_fun,
        final_distance=None,
    )
    route_path_polygons = route_path.polygons

    # Make the route path into a Device with ports, and use "connect" to move it
    # into the proper location
    D = Device()
    D.add_polygon(route_path_polygons, layer=layer)
    p1 = D.add_port(name=1, midpoint=(0, 0), width=width1, orientation=180)
    p2 = D.add_port(  # noqa: F841
        name=2,
        midpoint=[forward_distance, lateral_distance],
        width=width2,
        orientation=0,
    )
    D.info["length"] = route_path.length

    D.rotate(angle=180 + port1.orientation - p1.orientation, center=p1.midpoint)
    D.move(origin=p1, destination=port1)
    return D


def route_quad(port1, port2, width1=None, width2=None, layer=0):
    """Routes a basic quadrilateral polygon directly between two ports.

    Parameters
    ----------
    port1, port2 : Port objects
        Ports to route between.
    width1, width2 : int, float or None
        Width of quadrilateral at ports. If None, uses port widths.
    layer : int or array-like[2]
        Layer to put the route on.

    Returns
    ---------
    D : Device
        A Device containing the route and two ports (`1` and `2`) on either end.
    """

    def get_port_edges(port, width):
        _, e1 = _get_rotated_basis(port.orientation)
        pt1 = port.midpoint + e1 * width / 2
        pt2 = port.midpoint - e1 * width / 2
        return pt1, pt2

    if width1 is None:
        width1 = port1.width
    if width2 is None:
        width2 = port2.width
    vertices = np.array(get_port_edges(port1, width1) + get_port_edges(port2, width2))
    center = np.mean(vertices, axis=0)
    displacements = vertices - center
    # sort vertices by angle from center of quadrilateral to make convex polygon
    angles = np.array([np.arctan2(disp[0], disp[1]) for disp in displacements])
    vertices = [vert for _, vert in sorted(zip(angles, vertices), key=lambda x: x[0])]
    D = Device("route_quad")
    D.add_polygon(points=vertices, layer=layer)
    D.add_port(
        name=1,
        midpoint=port1.midpoint,
        orientation=port1.orientation + 180,
        width=width1,
    )
    D.add_port(
        name=2,
        midpoint=port2.midpoint,
        orientation=port2.orientation + 180,
        width=width2,
    )
    return D


def route_smooth(
    port1,
    port2,
    radius=5,
    width=None,
    path_type="manhattan",
    manual_path=None,
    smooth_options={"corner_fun": pp.euler, "use_eff": True},
    layer=np.nan,
    **kwargs
):

    """Convenience function that routes a path between ports using pp.smooth(),
    then immediately extrudes the path to create polygons. Has several waypoint
    path type options.  Equivalent to e.g.
        >>> pts = pr.path_manhattan(port1, port2, radius)
        >>> P = pp.smooth(pts, radius)
        >>> D = P.extrude(width)

    Parameters
    ----------
    port1, port2 : Port objects
        Ports to route between.
    radius : int or float
        Bend radius passed to pp.smooth
    width : None, int, float, array-like[2], or CrossSection
        If None, the route linearly tapers between the widths the ports
        If set to a single number (e.g. `width=1.7`): makes a fixed-width route
        If set to a 2-element array (e.g. `width=[1.8,2.5]`): makes a route
            whose width varies linearly from width[0] to width[1]
        If set to a CrossSection: uses the CrossSection parameters for the route
    path_type : {'manhattan', 'L', 'U', 'J', 'C', 'V', 'Z', 'straight', 'manual'}
        Method of waypoint path creation. Should be one of
            - 'manhattan' - automatic manhattan routing
                    (see path_manhattan() ).
            - 'L' - L-shaped path for orthogonal ports that can be directly
                    connected (see path_L() ).
            - 'U' - U-shaped path for parrallel or facing ports
                    (see path_U() ).
            - 'J' - J-shaped path for orthogonal ports that cannot be
                    directly connected (see path_J() ).
            - 'C' - C-shaped path for ports that face away from each
                    other (see path_C() ).
            - 'Z' - Z-shaped path with three segments for ports at any
                    angles (see path_Z() ).
            - 'V' - V-shaped path with two segments for ports at any
                    angles (see path_V() ).
            - 'straight' - straight path for ports that face each other
                    see path_straight() ).
            - 'manual' - use an explicit waypoint path provided
                    in manual_path.
    manual_path : array-like[N][2] or Path
        Waypoint path for creating a manual route
    smooth_options: dict
        Keyword arguments passed to pp.smooth
    layer : int or array-like[2]
        Layer to put route on. `layer=0` is used by default.
    **kwargs :
        Keyword arguments passed to the waypoint path function.

    Returns
    ----------
    D : Device
        A Device containing the route and two ports (`1` and `2`) on either end.
    """
    if path_type == "straight":
        P = path_straight(port1, port2)
    elif path_type == "manual":
        if not isinstance(manual_path, Path):
            P = Path(manual_path)
        else:
            P = manual_path
    elif path_type == "L":
        P = path_L(port1, port2)
    elif path_type == "U":
        P = path_U(port1, port2, **kwargs)
    elif path_type == "J":
        P = path_J(port1, port2, **kwargs)
    elif path_type == "C":
        P = path_C(port1, port2, **kwargs)
    elif path_type == "manhattan":
        if smooth_options["corner_fun"] == pp.euler:
            use_eff = smooth_options.get("use_eff")
            if use_eff is None or not use_eff:
                raise ValueError(
                    """[PHIDL] route_smooth(): when using manhattan path type with euler
                    bends, smooth_options['use_eff'] must be True."""
                )
        P = path_manhattan(port1, port2, radius=radius)
    elif path_type == "Z":
        P = path_Z(port1, port2, **kwargs)
    elif path_type == "V":
        P = path_V(port1, port2)
    else:
        raise ValueError(
            """[PHIDL] route_smooth() received an invalid path_type.  Must be one of
        {'manhattan', 'L', 'U', 'J', 'C', 'V', 'Z', 'straight', 'manual'}"""
        )

    P = pp.smooth(points=P, radius=radius, **smooth_options)
    if width is None:
        X1 = CrossSection().add(width=port1.width, ports=(1, 2), layer=layer, name="a")
        X2 = CrossSection().add(width=port2.width, ports=(1, 2), layer=layer, name="a")
        cross_section = pp.transition(
            cross_section1=X1, cross_section2=X2, width_type="linear"
        )
        D = P.extrude(width=cross_section)
    else:
        D = P.extrude(width=width, layer=layer)
        if not isinstance(width, CrossSection):
            newport1 = D.add_port(port=port1, name=1).rotate(180)
            newport2 = D.add_port(port=port2, name=2).rotate(180)
            if np.size(width) == 1:
                newport1.width = width
                newport2.width = width
            if np.size(width) == 2:
                newport1.width = width[0]
                newport2.width = width[1]
    return D


def route_sharp(
    port1,
    port2,
    width=None,
    path_type="manhattan",
    manual_path=None,
    layer=np.nan,
    **kwargs
):

    """Convenience function that routes a path between ports and immediately
    extrudes the path to create polygons. Has several waypoint path type
    options.  Equivalent to e.g.
        >>> P = pr.path_manhattan(port1, port2, radius)
        >>> D = P.extrude(width)

    Parameters
    ----------
    port1, port2 : Port objects
        Ports to route between.
    width : None, int, float, array-like[2], or CrossSection
        If None, the route linearly tapers between the widths the ports
        If set to a single number (e.g. `width=1.7`): makes a fixed-width route
        If set to a 2-element array (e.g. `width=[1.8,2.5]`): makes a route
            whose width varies linearly from width[0] to width[1]
        If set to a CrossSection: uses the CrossSection parameters for the route
    path_type : {'manhattan', 'L', 'U', 'J', 'C', 'V', 'Z', 'straight', 'manual'}
        Method of waypoint path creation. Should be one of
            - 'manhattan' - automatic manhattan routing
                    (see path_manhattan() ).
            - 'L' - L-shaped path for orthogonal ports that can be directly
                    connected (see path_L() ).
            - 'U' - U-shaped path for parrallel or facing ports
                    (see path_U() ).
            - 'J' - J-shaped path for orthogonal ports that cannot be
                    directly connected (see path_J() ).
            - 'C' - C-shaped path for ports that face away from each
                    other (see path_C() ).
            - 'Z' - Z-shaped path with three segments for ports at any
                    angles (see path_Z() ).
            - 'V' - V-shaped path with two segments for ports at any
                    angles (see path_V() ).
            - 'straight' - straight path for ports that face each other
                    see path_straight() ).
            - 'manual' - use an explicit waypoint path provided
                    in manual_path.
    manual_path : array-like[N][2] or Path
        Waypoint path for creating a manual route
    layer : int or array-like[2]
        Layer to put route on. `layer=0` is used by default.
    **kwargs :
        Keyword arguments passed to the waypoint path function.

    Returns
    ----------
    D : Device
        A Device containing the route and two ports (`1` and `2`) on either end.
    """
    if path_type == "straight":
        P = path_straight(port1, port2)
    elif path_type == "manual":
        if not isinstance(manual_path, Path):
            P = Path(manual_path)
        else:
            P = manual_path
    elif path_type == "L":
        P = path_L(port1, port2)
    elif path_type == "U":
        P = path_U(port1, port2, **kwargs)
    elif path_type == "J":
        P = path_J(port1, port2, **kwargs)
    elif path_type == "C":
        P = path_C(port1, port2, **kwargs)
    elif path_type == "manhattan":
        radius = max(port1.width, port2.width)
        P = path_manhattan(port1, port2, radius=radius)
    elif path_type == "Z":
        P = path_Z(port1, port2, **kwargs)
    elif path_type == "V":
        P = path_V(port1, port2)
    else:
        raise ValueError(
            """[PHIDL] route_sharp() received an invalid path_type.  Must be one of
        {'manhattan', 'L', 'U', 'J', 'C', 'V', 'Z', 'straight', 'manual'}"""
        )
    if width is None:
        X1 = CrossSection().add(width=port1.width, ports=(1, 2), layer=layer, name="a")
        X2 = CrossSection().add(width=port2.width, ports=(1, 2), layer=layer, name="a")
        cross_section = pp.transition(
            cross_section1=X1, cross_section2=X2, width_type="linear"
        )
        D = P.extrude(width=cross_section)
    else:
        D = P.extrude(width=width, layer=layer)
        if not isinstance(width, CrossSection):
            newport1 = D.add_port(port=port1, name=1).rotate(180)
            newport2 = D.add_port(port=port2, name=2).rotate(180)
            if np.size(width) == 1:
                newport1.width = width
                newport2.width = width
            if np.size(width) == 2:
                newport1.width = width[0]
                newport2.width = width[1]
    return D


def path_straight(port1, port2):
    """Return waypoint path between port1 and port2 in a straight line.
    Useful when ports point directly at each other.

    Parameters
    ----------
    port1, port2 : Port objects
        Ports to route between.

    Returns
    ----------
    points : array[2][2]
        Waypoints for the route path to follow.
    """
    delta_orientation = np.round(
        np.abs(np.mod(port1.orientation - port2.orientation, 360)), 3
    )
    e1, e2 = _get_rotated_basis(port1.orientation)
    displacement = port2.midpoint - port1.midpoint
    xrel = np.round(
        np.dot(displacement, e1), 3
    )  # relative position of port 2, forward/backward
    yrel = np.round(
        np.dot(displacement, e2), 3
    )  # relative position of port 2, left/right
    if (delta_orientation not in (0, 180, 360)) or (yrel != 0) or (xrel <= 0):
        raise ValueError(
            "[PHIDL] path_straight(): ports must point directly at each other."
        )
    return Path(np.array([port1.midpoint, port2.midpoint]))


def path_L(port1, port2):
    """Return waypoint path between port1 and port2 in an L shape. Useful
    when orthogonal ports can be directly connected with one turn.

    Parameters
    ----------
    port1, port2 : Port objects
        Ports to route between.

    Returns
    ----------
    points : Path
        Waypoints for the route path to follow.
    """
    delta_orientation = np.round(
        np.abs(np.mod(port1.orientation - port2.orientation, 360)), 3
    )
    if delta_orientation not in (90, 270):
        raise ValueError("[PHIDL] path_L(): ports must be orthogonal.")
    e1, e2 = _get_rotated_basis(port1.orientation)
    # assemble waypoints
    pt1 = port1.midpoint
    pt3 = port2.midpoint
    delta_vec = pt3 - pt1
    pt2 = pt1 + np.dot(delta_vec, e1) * e1
    return Path(np.array([pt1, pt2, pt3]))


def path_U(port1, port2, length1=200):
    """Return waypoint path between port1 and port2 in a U shape. Useful
    when ports face the same direction or toward each other.

    Parameters
    ----------
    port1, port2 : Port objects
        Ports to route between.
    length1 : int or float
        Length of segment exiting port1. Should be larger than bend radius.

    Returns
    ----------
    points : Path
        Waypoints for the route path to follow.
    """
    delta_orientation = np.round(
        np.abs(np.mod(port1.orientation - port2.orientation, 360)), 3
    )
    if delta_orientation not in (0, 180, 360):
        raise ValueError("[PHIDL] path_U(): ports must be parrallel.")
    theta = np.radians(port1.orientation)
    e1 = np.array([np.cos(theta), np.sin(theta)])
    e2 = np.array([-1 * np.sin(theta), np.cos(theta)])
    # assemble waypoints
    pt1 = port1.midpoint
    pt4 = port2.midpoint
    pt2 = pt1 + length1 * e1  # outward by length1 distance
    delta_vec = pt4 - pt2
    pt3 = pt2 + np.dot(delta_vec, e2) * e2
    return Path(np.array([pt1, pt2, pt3, pt4]))


def path_J(port1, port2, length1=200, length2=200):
    """Return waypoint path between port1 and port2 in a J shape. Useful
    when orthogonal ports cannot be connected directly with an L shape.

    Parameters
    ----------
    port1, port2 : Port objects
        Ports to route between.
    length1 : int or float
        Length of segment exiting port1. Should be larger than bend radius.
    length2 : int or float
        Length of segment exiting port2. Should be larger than bend radius.

    Returns
    ----------
    points : Path
        Waypoints for the route path to follow.
    """
    delta_orientation = np.round(
        np.abs(np.mod(port1.orientation - port2.orientation, 360)), 3
    )
    if delta_orientation not in (90, 270):
        raise ValueError("[PHIDL] path_J(): ports must be orthogonal.")
    e1, _ = _get_rotated_basis(port1.orientation)
    e2, _ = _get_rotated_basis(port2.orientation)
    # assemble waypoints
    pt1 = port1.midpoint
    pt2 = pt1 + length1 * e1  # outward from port1 by length1
    pt5 = port2.midpoint
    pt4 = pt5 + length2 * e2  # outward from port2 by length2
    delta_vec = pt4 - pt2
    pt3 = pt2 + np.dot(delta_vec, e2) * e2  # move orthogonally in e2 direction
    return Path(np.array([pt1, pt2, pt3, pt4, pt5]))


def path_C(port1, port2, length1=100, left1=100, length2=100):
    """Return waypoint path between port1 and port2 in a C shape. Useful
    when ports are parrallel and face away from each other.

    Parameters
    ----------
    port1, port2 : Port objects
        Ports to route between.
    length1 : int or float
        Length of route segment coming out of port1. Should be at larger
        than bend radius.
    left1 : int or float
        Length of route segment that turns left (or right if negative)
        from port1. Should be larger than twice the bend radius.
    length2 : int or float
        Length of route segment coming out of port2. Should be larger
        than bend radius.

    Returns
    ----------
    points : Path
        Waypoints for the route path to follow.
    """
    delta_orientation = np.round(
        np.abs(np.mod(port1.orientation - port2.orientation, 360)), 3
    )
    if delta_orientation not in (0, 180, 360):
        raise ValueError("[PHIDL] path_C(): ports must be parrallel.")
    e1, e_left = _get_rotated_basis(port1.orientation)
    e2, _ = _get_rotated_basis(port2.orientation)
    # assemble route points
    pt1 = port1.midpoint
    pt2 = pt1 + length1 * e1  # outward from port1 by length1
    pt3 = pt2 + left1 * e_left  # leftward by left1
    pt6 = port2.midpoint
    pt5 = pt6 + length2 * e2  # outward from port2 by length2
    delta_vec = pt5 - pt3
    pt4 = pt3 + np.dot(delta_vec, e1) * e1  # move orthogonally in e1 direction
    return Path(np.array([pt1, pt2, pt3, pt4, pt5, pt6]))


def path_manhattan(port1, port2, radius):
    """Return waypoint path between port1 and port2 using manhattan routing.
    Routing is performed using straight, L, U, J, or C  waypoint path
    as needed. Ports must face orthogonal or parallel directions.

    Parameters
    ----------
    port1, port2 : Port objects
        Ports to route between.
    radius : float or int
        Bend radius for 90 degree bend.

    Returns
    ----------
    points : Path
        Waypoints for the route path to follow.
    """
    radius = radius + 0.1  # ensure space for bend radius
    e1, e2 = _get_rotated_basis(port1.orientation)
    displacement = port2.midpoint - port1.midpoint
    xrel = np.round(
        np.dot(displacement, e1), 3
    )  # port2 position, forward(+)/backward(-) from port 1
    yrel = np.round(
        np.dot(displacement, e2), 3
    )  # port2 position, left(+)/right(-) from port1
    orel = np.round(
        np.abs(np.mod(port2.orientation - port1.orientation, 360)), 3
    )  # relative orientation
    if orel not in (0, 90, 180, 270, 360):
        raise ValueError(
            "[PHIDL] path_manhattan(): ports must face parrallel or orthogonal directions."
        )
    if orel in (90, 270):
        # Orthogonal case
        if (
            (orel == 90 and yrel < -1 * radius) or (orel == 270 and yrel > radius)
        ) and xrel > radius:
            pts = path_L(port1, port2)
        else:
            # Adjust length1 and length2 to ensure intermediate segments fit bend radius
            direction = -1 if (orel == 270) else 1
            length2 = (
                2 * radius - direction * yrel
                if (np.abs(radius + direction * yrel) < 2 * radius)
                else radius
            )
            length1 = (
                2 * radius + xrel if (np.abs(radius - xrel) < 2 * radius) else radius
            )
            pts = path_J(port1, port2, length1=length1, length2=length2)
    else:
        # Parrallel case
        if orel == 180 and yrel == 0 and xrel > 0:
            pts = path_straight(port1, port2)
        elif (orel == 180 and xrel <= 2 * radius) or (np.abs(yrel) < 2 * radius):
            # Adjust length1 and left1 to ensure intermediate segments fit bend radius
            left1 = (
                np.abs(yrel) + 2 * radius if (np.abs(yrel) < 4 * radius) else 2 * radius
            )
            y_direction = -1 if (yrel < 0) else 1
            left1 = y_direction * left1
            length2 = radius
            x_direction = -1 if (orel == 180) else 1
            segmentx_length = np.abs(xrel + x_direction * length2 - radius)
            if segmentx_length < 2 * radius:
                length1 = xrel + x_direction * length2 + 2 * radius
            else:
                length1 = radius
            pts = path_C(port1, port2, length1=length1, length2=length2, left1=left1)
        else:
            # Adjust length1 to ensure segment comes out of port2
            length1 = radius + xrel if (orel == 0 and xrel > 0) else radius
            pts = path_U(port1, port2, length1=length1)
    return pts


def path_Z(port1, port2, length1=100, length2=100):
    """Return waypoint path between port1 and port2 in a Z shape. Ports can have any relative
    orientation.

    Parameters
    ----------
    port1, port2 : Port objects
        Ports to route between.
    length1 : int or float
        Length of route segment coming out of port1.
    length2 : int or float
        Length of route segment coming out of port2.

    Returns
    ----------
    points : Path
        Waypoints for the route path to follow.
    """
    # get basis vectors in port directions
    e1, _ = _get_rotated_basis(port1.orientation)
    e2, _ = _get_rotated_basis(port2.orientation)
    # assemble route  points
    pt1 = port1.midpoint
    pt2 = pt1 + length1 * e1  # outward from port1 by length1
    pt4 = port2.midpoint
    pt3 = pt4 + length2 * e2  # outward from port2 by length2
    return Path(np.array([pt1, pt2, pt3, pt4]))


def path_V(port1, port2):
    """Return waypoint path between port1 and port2 in a V shape. Useful when
    ports point to a single connecting point

    Parameters
    ----------
    port1, port2 : Port objects
        Ports to route between.

    Returns
    ----------
    points : Path
        Waypoints for the route path to follow.
    """
    # get basis vectors in port directions
    e1, _ = _get_rotated_basis(port1.orientation)
    e2, _ = _get_rotated_basis(port2.orientation)
    # assemble route  points
    pt1 = port1.midpoint
    pt3 = port2.midpoint
    # solve for intersection
    E = np.column_stack((e1, -1 * e2))
    pt2 = np.matmul(np.linalg.inv(E), pt3 - pt1)[0] * e1 + pt1
    return Path(np.array([pt1, pt2, pt3]))


def path_xy(port1, port2, directions="xxyx"):
    """Creates a Path that travels only in x and y directions (manhattan) from
    one point (or Port) to another. The `directions` string determines the order
    of the x/y steps. Example: `directions = 'xyx'` will travel
        1/2 the distance in x from p1 to p2
        The whole distance in y from p1 to p2
        1/2 of the distance in x from p1 to p2

    Parameters
    ----------
    p1, p2 : array-like[2] points or Port objects
        Points to route between.
    directions : string of {'x','y'} characters
        Directions the Path will be routed along

    Returns
    ----------
    Path
        Waypoints for the route path to follow.
    """
    if isinstance(port1, Port):
        p1 = port1.midpoint
    elif np.size(port1) == 2:
        p1 = port1
    if isinstance(port2, Port):
        p2 = port2.midpoint
    elif np.size(port2) == 2:
        p2 = port2

    directions = directions.lower()
    num_x = sum(xy == "x" for xy in directions)
    num_y = sum(xy == "y" for xy in directions)
    distance = np.asarray(p2) - p1

    points = [p1]
    for xy in directions:
        if xy == "x":
            travel = np.array([distance[0] / num_x, 0])
            new_point = points[-1] + travel
        elif xy == "y":
            travel = np.array([0, distance[1] / num_y])
            new_point = points[-1] + travel
        else:
            raise ValueError(
                '[PHIDL] path_xy() directions argument must be string with only "x" or "y" characters'
            )
        if (
            np.abs(np.sum(travel)) > 1e-6
        ):  # Only add point if traveling some significant distance
            points.append(new_point)

    return Path(np.array(points))


def route_xy(port1, port2, directions="xxyx", width=None, layer=np.nan):
    """Routes a path in x and y directions (manhattan) from one point (or Port)
    to another. The `directions` string determines the order of the x/y steps.
    Example: `directions = 'xyx'` will travel
        1/2 the distance in x from p1 to p2
        The whole distance in y from p1 to p2
        1/2 of the distance in x from p1 to p2

    Parameters
    ----------
    port1, port2 : Ports pr array-like[2] points or Port objects
        Points to route between.
    directions : string of {'x','y'} characters
        Directions the Path will be routed along

    Returns
    ----------
    D : Device
        A Device containing the route and two ports (`1` and `2`) on either end.
    """

    P = path_xy(port1, port2, directions=directions)
    if width is None:
        X1 = CrossSection().add(width=port1.width, ports=(1, 2), layer=layer, name="a")
        X2 = CrossSection().add(width=port2.width, ports=(1, 2), layer=layer, name="a")
        cross_section = pp.transition(
            cross_section1=X1, cross_section2=X2, width_type="linear"
        )
        D = P.extrude(width=cross_section)
    else:
        D = P.extrude(width=width, layer=layer)
        if not isinstance(width, CrossSection):
            newport1 = D.add_port(port=port1, name=1).rotate(180)
            newport2 = D.add_port(port=port2, name=2).rotate(180)
            if np.size(width) == 1:
                newport1.width = width
                newport2.width = width
            if np.size(width) == 2:
                newport1.width = width[0]
                newport2.width = width[1]
    # D = P.extrude(width, layer = layer)
    return D


# ################
# gradual_bend() - variable radius-of-curvature bends for low-loss routing
# note - these are not bezier spline curves, but are instead constructed by
# subdividing the coverage angle into equal segments and implementing a gradual
# decrease in bending radius until the minimum is reached. """

# ################
# route_manhattan() - routing between any two ports rotated in 90 degree increments
# note - ports must be located diagonally from each other and oriented along
# cardinal directions.  Routing can be done with circular or gradual bends.
# Waveguide width is set by the width parameter of the first port.

# ################
# route_manhattan_auto() - handy routine which performs route_manhattan() on a
# vector of ports provided to it, allowing easy connecting of many objects.


class RoutingError(ValueError):
    pass


def route_manhattan(  # noqa: C901
    port1, port2, bendType="circular", layer=0, radius=20
):
    """
    .. deprecated:: 1.6.0
        `route_manhattan` will be removed in August 2022, please replace with
        `route_smooth()`.
    """
    warnings.warn(
        """[PHIDL] Warning: route_manhattan() will be deprecated
    in August 2022, please replace with `route_smooth()`"""
    )
    # route along cardinal directions between any two ports placed diagonally
    # from each other

    valid_bend_types = ["circular", "gradual"]

    if bendType not in valid_bend_types:
        raise ValueError("bendType{}= not in {}".format(bendType, valid_bend_types))

    if bendType == "gradual":
        b = _gradual_bend(radius=radius)
        radius_eff = b.xsize
    else:
        radius_eff = radius

    if (
        abs(port1.midpoint[0] - port2.midpoint[0]) < 2 * radius_eff
        or abs(port1.midpoint[1] - port2.midpoint[1]) < 2 * radius_eff
    ):
        raise RoutingError(
            "bend does not fit (radius = %s) you need radius <" % radius_eff,
            min(
                [
                    abs(port1.midpoint[0] - port2.midpoint[0]) / 2,
                    abs(port1.midpoint[1] - port2.midpoint[1]) / 2,
                ]
            ),
        )

    Total = Device()
    width = port1.width
    # first map into uniform plane with normal x,y coords
    # allows each situation to be put into uniform cases of quadrants for routing.
    # this is because bends change direction and positioning.
    if port1.orientation == 0:
        p2 = [port2.midpoint[0], port2.midpoint[1]]
        p1 = [port1.midpoint[0], port1.midpoint[1]]
    if port1.orientation == 90:
        p2 = [port2.midpoint[1], -port2.midpoint[0]]
        p1 = [port1.midpoint[1], -port1.midpoint[0]]
    if port1.orientation == 180:
        p2 = [-port2.midpoint[0], -port2.midpoint[1]]
        p1 = [-port1.midpoint[0], -port1.midpoint[1]]
    if port1.orientation == 270:
        p2 = [-port2.midpoint[1], port2.midpoint[0]]
        p1 = [-port1.midpoint[1], port1.midpoint[0]]

    Total.add_port(name=1, port=port1)
    Total.add_port(name=2, port=port2)

    if p2[1] == p1[1] or p2[0] == p1[0]:
        raise ValueError("Error - ports must be at different x AND y values.")

    # if it is parallel or anti-parallel, route with 180 option
    if (
        np.round(np.abs(np.mod(port1.orientation - port2.orientation, 360)), 3) == 180
    ) or (np.round(np.abs(np.mod(port1.orientation - port2.orientation, 360)), 3) == 0):
        R1 = _route_manhattan180(
            port1=port1, port2=port2, bendType=bendType, layer=layer, radius=radius
        )
        r1 = Total.add_ref(R1)

    else:
        # first quadrant case
        if (p2[1] > p1[1]) & (p2[0] > p1[0]):
            # simple 90 degree single-bend case
            if (
                port2.orientation == port1.orientation - 90
                or port2.orientation == port1.orientation + 270
            ):
                R1 = _route_manhattan90(
                    port1=port1,
                    port2=port2,
                    bendType=bendType,
                    layer=layer,
                    radius=radius,
                )
                r1 = Total.add_ref(R1)
            elif (
                port2.orientation == port1.orientation + 90
                or port2.orientation == port1.orientation - 270
            ):
                if bendType == "circular":
                    B1 = _arc(
                        radius=radius,
                        width=width,
                        layer=layer,
                        angle_resolution=1,
                        start_angle=port1.orientation,
                        theta=90,
                    )
                    radiusEff = radius
                if bendType == "gradual":
                    B1 = _gradual_bend(
                        radius=radius,
                        width=width,
                        layer=layer,
                        start_angle=port1.orientation,
                        direction="ccw",
                    )
                    radiusEff = B1.xsize - width / 2
                b1 = Total.add_ref(B1)
                b1.connect(port=1, destination=port1)

                R1 = _route_manhattan180(
                    port1=b1.ports[2],
                    port2=port2,
                    bendType=bendType,
                    layer=layer,
                    radius=radius,
                )
                r1 = Total.add_ref(R1)
        # second quadrant case
        if (p2[1] > p1[1]) & (p2[0] < p1[0]):
            if (
                np.abs(port1.orientation - port2.orientation) == 90
                or np.abs(port1.orientation - port2.orientation) == 270
            ):
                if bendType == "circular":
                    B1 = _arc(
                        radius=radius,
                        width=width,
                        layer=layer,
                        angle_resolution=1,
                        start_angle=port1.orientation,
                        theta=90,
                    )
                    radiusEff = radius
                if bendType == "gradual":
                    B1 = _gradual_bend(
                        radius=radius,
                        width=width,
                        layer=layer,
                        start_angle=port1.orientation,
                        direction="ccw",
                    )
                    radiusEff = B1.xsize - width / 2
                b1 = Total.add_ref(B1)
                b1.connect(port=1, destination=port1)
                R1 = _route_manhattan180(
                    port1=b1.ports[2],
                    port2=port2,
                    bendType=bendType,
                    layer=layer,
                    radius=radius,
                )
                r1 = Total.add_ref(R1)
        # third quadrant case
        if (p2[1] < p1[1]) & (p2[0] < p1[0]):
            if (
                np.abs(port1.orientation - port2.orientation) == 90
                or np.abs(port1.orientation - port2.orientation) == 270
            ):
                if bendType == "circular":
                    B1 = _arc(
                        radius=radius,
                        width=width,
                        layer=layer,
                        angle_resolution=1,
                        start_angle=port1.orientation,
                        theta=-90,
                    )
                    radiusEff = radius
                if bendType == "gradual":
                    B1 = _gradual_bend(
                        radius=radius,
                        width=width,
                        layer=layer,
                        start_angle=port1.orientation,
                        direction="cw",
                    )
                    radiusEff = B1.xsize - width / 2
                b1 = Total.add_ref(B1)
                b1.connect(port=1, destination=port1)
                R1 = _route_manhattan180(
                    port1=b1.ports[2],
                    port2=port2,
                    bendType=bendType,
                    layer=layer,
                    radius=radius,
                )
                r1 = Total.add_ref(R1)
        # fourth quadrant case
        if (p2[1] < p1[1]) & (p2[0] > p1[0]):
            # simple 90 degree single-bend case
            if (
                port2.orientation == port1.orientation + 90
                or port2.orientation == port1.orientation - 270
            ):
                R1 = _route_manhattan90(
                    port1=port1,
                    port2=port2,
                    bendType=bendType,
                    layer=layer,
                    radius=radius,
                )
                r1 = Total.add_ref(R1)  # noqa: F841
            elif (
                port2.orientation == port1.orientation - 90
                or port2.orientation == port1.orientation + 270
            ):
                if bendType == "circular":
                    B1 = _arc(
                        radius=radius,
                        width=width,
                        layer=layer,
                        angle_resolution=1,
                        start_angle=port1.orientation,
                        theta=-90,
                    )
                    radiusEff = radius
                if bendType == "gradual":
                    B1 = _gradual_bend(
                        radius=radius,
                        width=width,
                        layer=layer,
                        start_angle=port1.orientation,
                        direction="cw",
                    )
                    radiusEff = B1.xsize - width / 2  # noqa: F841
                b1 = Total.add_ref(B1)
                b1.connect(port=1, destination=port1)
                R1 = _route_manhattan180(
                    port1=b1.ports[2],
                    port2=port2,
                    bendType=bendType,
                    layer=layer,
                    radius=radius,
                )
                Total.add_ref(R1)
    return Total


def _route_manhattan180(  # noqa: C901
    port1, port2, bendType="circular", layer=0, radius=20
):
    # this is a subroutine of route_manhattan() and should not be used by itself.
    Total = Device()
    width = port1.width
    # first map into uniform plane with normal x,y coords
    # allows each situation to be put into uniform cases of quadrants for routing.
    # this is because bends change direction and positioning.
    if port1.orientation == 0:
        p2 = [port2.midpoint[0], port2.midpoint[1]]
        p1 = [port1.midpoint[0], port1.midpoint[1]]
    if port1.orientation == 90:
        p2 = [port2.midpoint[1], -port2.midpoint[0]]
        p1 = [port1.midpoint[1], -port1.midpoint[0]]
    if port1.orientation == 180:
        p2 = [-port2.midpoint[0], -port2.midpoint[1]]
        p1 = [-port1.midpoint[0], -port1.midpoint[1]]
    if port1.orientation == 270:
        p2 = [-port2.midpoint[1], port2.midpoint[0]]
        p1 = [-port1.midpoint[1], port1.midpoint[0]]

    # create placeholder ports based on the imaginary coordinates we created
    Total.add_port(name="t1", midpoint=[0, 0], orientation=0, width=width)
    if port1.orientation != port2.orientation:
        Total.add_port(
            name="t2", midpoint=list(np.subtract(p2, p1)), orientation=180, width=width
        )
    else:
        Total.add_port(
            name="t2", midpoint=list(np.subtract(p2, p1)), orientation=0, width=width
        )

    if port1.orientation == port2.orientation:
        # first quadrant target
        if (p2[1] > p1[1]) & (p2[0] > p1[0]):
            if bendType == "circular":
                B1 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=0,
                    theta=90,
                )
                B2 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=90,
                    theta=90,
                )
                radiusEff = radius
            if bendType == "gradual":
                B1 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=0,
                    direction="ccw",
                )
                B2 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=90,
                    direction="ccw",
                )
                radiusEff = B1.xsize - width / 2
            b1 = Total.add_ref(B1)
            b2 = Total.add_ref(B2)

            b1.connect(port=b1.ports[1], destination=Total.ports["t1"])
            b1.move([p2[0] - p1[0], 0])
            b2.connect(port=b2.ports[1], destination=b1.ports[2])
            b2.move([0, p2[1] - p1[1] - radiusEff * 2])
            R1 = route_basic(port1=Total.ports["t1"], port2=b1.ports[1], layer=layer)
            r1 = Total.add_ref(R1)
            R2 = route_basic(port1=b1.ports[2], port2=b2.ports[1], layer=layer)
            r2 = Total.add_ref(R2)
            Total.add_port(name=1, port=r1.ports[1])
            Total.add_port(name=2, port=b2.ports[2])
        # second quadrant target
        if (p2[1] > p1[1]) & (p2[0] < p1[0]):
            if bendType == "circular":
                B1 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=0,
                    theta=90,
                )
                B2 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=90,
                    theta=90,
                )
                radiusEff = radius
            if bendType == "gradual":
                B1 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=0,
                    direction="ccw",
                )
                B2 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=90,
                    direction="ccw",
                )
                radiusEff = B1.xsize - width / 2
            b1 = Total.add_ref(B1)
            b2 = Total.add_ref(B2)
            b1.connect(port=b1.ports[1], destination=Total.ports["t1"])

            b2.connect(port=b2.ports[1], destination=b1.ports[2])
            b2.move([0, p2[1] - p1[1] - radiusEff * 2])
            R1 = route_basic(port1=b1.ports[2], port2=b2.ports[1], layer=layer)
            r1 = Total.add_ref(R1)
            R2 = route_basic(port1=b2.ports[2], port2=Total.ports["t2"], layer=layer)
            r2 = Total.add_ref(R2)
            Total.add_port(name=1, port=b1.ports[1])
            Total.add_port(name=2, port=r2.ports[2])
        # third quadrant target
        if (p2[1] < p1[1]) & (p2[0] < p1[0]):
            if bendType == "circular":
                B1 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=0,
                    theta=-90,
                )
                B2 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=-90,
                    theta=-90,
                )
                radiusEff = radius
            if bendType == "gradual":
                B1 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=0,
                    direction="cw",
                )
                B2 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=-90,
                    direction="cw",
                )
                radiusEff = B1.xsize - width / 2
            b1 = Total.add_ref(B1)
            b2 = Total.add_ref(B2)
            b1.connect(port=b1.ports[1], destination=Total.ports["t1"])

            b2.connect(port=b2.ports[1], destination=b1.ports[2])
            b2.move([0, p2[1] - p1[1] + radiusEff * 2])
            R1 = route_basic(port1=b1.ports[2], port2=b2.ports[1], layer=layer)
            r1 = Total.add_ref(R1)
            R2 = route_basic(port1=b2.ports[2], port2=Total.ports["t2"], layer=layer)
            r2 = Total.add_ref(R2)
            Total.add_port(name=1, port=b1.ports[1])
            Total.add_port(name=2, port=r2.ports[2])
        # fourth quadrant target
        if (p2[1] < p1[1]) & (p2[0] > p1[0]):
            if bendType == "circular":
                B1 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=0,
                    theta=-90,
                )
                B2 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=-90,
                    theta=-90,
                )
                radiusEff = radius
            if bendType == "gradual":
                B1 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=0,
                    direction="cw",
                )
                B2 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=-90,
                    direction="cw",
                )
                radiusEff = B1.xsize - width / 2
            b1 = Total.add_ref(B1)
            b2 = Total.add_ref(B2)

            b1.connect(port=b1.ports[1], destination=Total.ports["t1"])
            b1.move([p2[0] - p1[0], 0])
            b2.connect(port=b2.ports[1], destination=b1.ports[2])
            b2.move([0, p2[1] - p1[1] + radiusEff * 2])
            R1 = route_basic(port1=Total.ports["t1"], port2=b1.ports[1], layer=layer)
            r1 = Total.add_ref(R1)
            R2 = route_basic(port1=b1.ports[2], port2=b2.ports[1], layer=layer)
            r2 = Total.add_ref(R2)
            Total.add_port(name=1, port=r1.ports[1])
            Total.add_port(name=2, port=b2.ports[2])

    # other port orientations are not supported:
    elif np.round(np.abs(np.mod(port1.orientation - port2.orientation, 360)), 3) != 180:
        raise ValueError(
            "[DEVICE] route() error: Ports do not face each other (orientations must be 180 apart)"
        )
    # otherwise, they are 180 degrees apart:
    else:
        # first quadrant target
        if (p2[1] > p1[1]) & (p2[0] > p1[0]):
            if bendType == "circular":
                B1 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=0,
                    theta=90,
                )
                B2 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=90,
                    theta=-90,
                )
                radiusEff = radius
            if bendType == "gradual":
                B1 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=0,
                    direction="ccw",
                )
                B2 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=90,
                    direction="cw",
                )
                radiusEff = B1.xsize - width / 2
            b1 = Total.add_ref(B1)
            b2 = Total.add_ref(B2)

            b1.connect(port=b1.ports[1], destination=Total.ports["t1"])
            b1.move([p2[0] - p1[0] - radiusEff * 2, 0])
            b2.connect(port=b2.ports[1], destination=b1.ports[2])
            b2.move([0, p2[1] - p1[1] - radiusEff * 2])
            R1 = route_basic(port1=Total.ports["t1"], port2=b1.ports[1], layer=layer)
            r1 = Total.add_ref(R1)
            R2 = route_basic(port1=b1.ports[2], port2=b2.ports[1], layer=layer)
            r2 = Total.add_ref(R2)
            Total.add_port(name=1, port=r1.ports[1])
            Total.add_port(name=2, port=b2.ports[2])
        # second quadrant target
        if (p2[1] > p1[1]) & (p2[0] < p1[0]):
            if bendType == "circular":
                B1 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=0,
                    theta=90,
                )
                B2 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=90,
                    theta=90,
                )
                B3 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=180,
                    theta=-90,
                )
                B4 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=90,
                    theta=-90,
                )
                radiusEff = radius
            if bendType == "gradual":
                B1 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=0,
                    direction="ccw",
                )
                B2 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=90,
                    direction="ccw",
                )
                B3 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=180,
                    direction="cw",
                )
                B4 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=90,
                    direction="cw",
                )
                radiusEff = B1.xsize - width / 2
            b1 = Total.add_ref(B1)
            b2 = Total.add_ref(B2)
            b3 = Total.add_ref(B3)
            b4 = Total.add_ref(B4)

            b1.connect(port=b1.ports[1], destination=Total.ports["t1"])

            b2.connect(port=b2.ports[1], destination=b1.ports[2])
            b2.move([0, p2[1] - p1[1] - radiusEff * 4])
            R1 = route_basic(port1=b1.ports[2], port2=b2.ports[1], layer=layer)
            r1 = Total.add_ref(R1)
            b3.connect(port=b3.ports[1], destination=b2.ports[2])
            b3.move([p2[0] - p1[0], 0])
            R2 = route_basic(port1=b2.ports[2], port2=b3.ports[1], layer=layer)
            r2 = Total.add_ref(R2)

            b4.connect(port=b4.ports[1], destination=b3.ports[2])

            Total.add_port(name=1, port=r1.ports[1])
            Total.add_port(name=2, port=b4.ports[2])
        # third quadrant target
        if (p2[1] < p1[1]) & (p2[0] < p1[0]):
            if bendType == "circular":
                B1 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=0,
                    theta=-90,
                )
                B2 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=-90,
                    theta=-90,
                )
                B3 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=-180,
                    theta=90,
                )
                B4 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=-90,
                    theta=90,
                )
                radiusEff = radius
            if bendType == "gradual":
                B1 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=0,
                    direction="cw",
                )
                B2 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=-90,
                    direction="cw",
                )
                B3 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=-180,
                    direction="ccw",
                )
                B4 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=-90,
                    direction="ccw",
                )
                radiusEff = B1.xsize - width / 2
            b1 = Total.add_ref(B1)
            b2 = Total.add_ref(B2)
            b3 = Total.add_ref(B3)
            b4 = Total.add_ref(B4)

            b1.connect(port=b1.ports[1], destination=Total.ports["t1"])

            b2.connect(port=b2.ports[1], destination=b1.ports[2])
            b2.move([0, p2[1] - p1[1] + radiusEff * 4])
            R1 = route_basic(port1=b1.ports[2], port2=b2.ports[1], layer=layer)
            r1 = Total.add_ref(R1)
            b3.connect(port=b3.ports[1], destination=b2.ports[2])
            b3.move([p2[0] - p1[0], 0])
            R2 = route_basic(port1=b2.ports[2], port2=b3.ports[1], layer=layer)
            r2 = Total.add_ref(R2)

            b4.connect(port=b4.ports[1], destination=b3.ports[2])

            Total.add_port(name=1, port=r1.ports[1])
            Total.add_port(name=2, port=b4.ports[2])
        # fourth quadrant target
        if (p2[1] < p1[1]) & (p2[0] > p1[0]):
            if bendType == "circular":
                B1 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=0,
                    theta=-90,
                )
                B2 = _arc(
                    radius=radius,
                    width=width,
                    layer=layer,
                    angle_resolution=1,
                    start_angle=-90,
                    theta=90,
                )
                radiusEff = radius
            if bendType == "gradual":
                B1 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=0,
                    direction="cw",
                )
                B2 = _gradual_bend(
                    radius=radius,
                    width=width,
                    layer=layer,
                    start_angle=-90,
                    direction="ccw",
                )
                radiusEff = B1.xsize - width / 2
            b1 = Total.add_ref(B1)
            b2 = Total.add_ref(B2)

            b1.connect(port=b1.ports[1], destination=Total.ports["t1"])
            b1.move([p2[0] - p1[0] - radiusEff * 2, 0])
            b2.connect(port=b2.ports[1], destination=b1.ports[2])
            b2.move([0, p2[1] - p1[1] + radiusEff * 2])
            R1 = route_basic(port1=Total.ports["t1"], port2=b1.ports[1], layer=layer)
            r1 = Total.add_ref(R1)
            R2 = route_basic(port1=b1.ports[2], port2=b2.ports[1], layer=layer)
            r2 = Total.add_ref(R2)
            Total.add_port(name=1, port=r1.ports[1])
            Total.add_port(name=2, port=b2.ports[2])

    Total.rotate(angle=port1.orientation, center=p1)
    Total.move(origin=Total.ports["t1"], destination=port1)
    return Total


def _route_manhattan90(port1, port2, bendType="circular", layer=0, radius=20):
    # this is a subroutine of route_manhattan() and should not be used by itself.
    Total = Device()
    width = port1.width
    # first map into uniform plane with normal x,y coords
    # allows each situation to be put into uniform cases of quadrants for routing.
    # this is because bends change direction and positioning.
    if port1.orientation == 0:
        p2 = [port2.midpoint[0], port2.midpoint[1]]
        p1 = [port1.midpoint[0], port1.midpoint[1]]
    if port1.orientation == 90:
        p2 = [port2.midpoint[1], -port2.midpoint[0]]
        p1 = [port1.midpoint[1], -port1.midpoint[0]]
    if port1.orientation == 180:
        p2 = [-port2.midpoint[0], -port2.midpoint[1]]
        p1 = [-port1.midpoint[0], -port1.midpoint[1]]
    if port1.orientation == 270:
        p2 = [-port2.midpoint[1], port2.midpoint[0]]
        p1 = [-port1.midpoint[1], port1.midpoint[0]]

    # create placeholder ports based on the imaginary coordinates we created
    Total.add_port(name="t1", midpoint=[0, 0], orientation=0, width=width)

    # CHECK THIS

    # first quadrant target, route upward
    if (p2[1] > p1[1]) & (p2[0] > p1[0]):
        Total.add_port(
            name="t2", midpoint=list(np.subtract(p2, p1)), orientation=-90, width=width
        )
        if bendType == "circular":
            B1 = _arc(
                radius=radius,
                width=width,
                layer=layer,
                angle_resolution=1,
                start_angle=0,
                theta=90,
            )
            radiusEff = radius
        if bendType == "gradual":
            B1 = _gradual_bend(
                radius=radius, width=width, layer=layer, start_angle=0, direction="ccw"
            )
            radiusEff = B1.xsize - width / 2
        b1 = Total.add_ref(B1)
        b1.connect(port=b1.ports[1], destination=Total.ports["t1"])
        b1.move([p2[0] - p1[0] - radiusEff, 0])

        R1 = route_basic(port1=Total.ports["t1"], port2=b1.ports[1], layer=layer)
        R2 = route_basic(port1=b1.ports[2], port2=Total.ports["t2"], layer=layer)
        r1 = Total.add_ref(R1)
        r2 = Total.add_ref(R2)
        Total.add_port(name=1, port=r1.ports[1])
        Total.add_port(name=2, port=r2.ports[2])

    # fourth quadrant target, route downward
    if (p2[1] < p1[1]) & (p2[0] > p1[0]):
        Total.add_port(
            name="t2", midpoint=list(np.subtract(p2, p1)), orientation=90, width=width
        )
        if bendType == "circular":
            B1 = _arc(
                radius=radius,
                width=width,
                layer=layer,
                angle_resolution=1,
                start_angle=0,
                theta=-90,
            )
            radiusEff = radius
        if bendType == "gradual":
            B1 = _gradual_bend(
                radius=radius, width=width, layer=layer, start_angle=0, direction="cw"
            )
            radiusEff = B1.xsize - width / 2
        b1 = Total.add_ref(B1)
        b1.connect(port=b1.ports[1], destination=Total.ports["t1"])
        b1.move([p2[0] - p1[0] - radiusEff, 0])
        R1 = route_basic(port1=Total.ports["t1"], port2=b1.ports[1], layer=layer)
        R2 = route_basic(port1=b1.ports[2], port2=Total.ports["t2"], layer=layer)
        r1 = Total.add_ref(R1)
        r2 = Total.add_ref(R2)
        Total.add_port(name=1, port=r1.ports[1])
        Total.add_port(name=2, port=r2.ports[2])
    Total.rotate(angle=port1.orientation, center=p1)
    Total.move(origin=Total.ports["t1"], destination=port1)

    return Total


def route_manhattan_auto(ports, bendType="circular", layer=0, radius=20):
    """routes a one-dimensional array of ports using manhattan algorithm
    and give it a series of ports to route to in a continuous list.
    accepts same parameters as ordinary route_manhattan to determine bending"""
    Total = Device()
    for x in range(int(np.floor(len(ports) / 2)) + 1):
        R = route_manhattan(
            port1=ports[x],
            port2=ports[x + 1],
            bendType=bendType,
            layer=layer,
            radius=radius,
        )
        Total.add_ref(R)

    return Total


def route_turn_manhattan(port1, port2, layer=0, radius=20):
    """
    Mahattan routing between two ports. If directions are not cardinal, adds a
    turn to make cardinal and then routes.

    Parameters
    ----------
    port1, port2: Port objects
        Ports to route to and from
    layer: int (default: 0)
        Layer to use for the routes
    radius: float (default: 20)
        Curve radius for bends

    Returns
    ----------
    Device object

    Notes
    ----------
    If direction is not cardinal, will route to nearest cardinal, then call
    route_manhattan.
    """

    D = Device()
    new_ports = []
    for port in (port1, port2):
        if port.orientation % 90 == 0:
            new_ports.append(port)
        else:
            turn_angle = _get_turn_angle(
                port.orientation, _to_cardinal(port.orientation)
            )
            turn_route = turn(port, radius=radius, angle=turn_angle, layer=layer)
            D.add_ref(turn_route)
            new_ports.append(turn_route.ports[2])

    # Manhattan on new ports
    route = route_manhattan(
        new_ports[0], new_ports[1], bendType="circular", layer=layer, radius=radius
    )

    D.add_ref(route)

    return D


def _to_cardinal(angle):
    """
    Determines which cardinal direction is closest to input angle

    Parameters
    ----------
    angle : float

    Returns
    -------
    angle : [-180, -90, 0, 90]
        Which cardinal direction is closest to the input angle
    """

    angle = _map_to_pm180(angle)

    cardinals = np.array([-180, -90, 0, 90])

    arg = np.argmin(np.abs(angle - cardinals))

    return cardinals[arg]


def _map_to_pm180(angle):
    """converts an angle to an angle between -180 (inclusive) to +180 (exclusive)"""
    return np.mod(angle + 180, 360) - 180


def _get_turn_angle(start_angle, target_angle):
    """
    Difference in angle in the range -180 to +180 (where negative is counter clockwise)

    Parameters
    ----------
    start_angle, target_angle : float

    Returns
    -------
    float
        difference in angle.
    """
    return _map_to_pm180(target_angle - start_angle)


# ==============================================================================
# Test code for route_manhattan
# ==============================================================================

# D=Device()
# A=pg.compass()
# A.add_port(name=1,port=A.ports['N'])
# A.add_port(name=2,port=A.ports['E'])
# A.add_port(name=3,port=A.ports['S'])
# A.add_port(name=4,port=A.ports['W'])
# points=[]
# points.append((300,300))
# points.append((-300,300))
# points.append((-300,-300))
# points.append((300,-300))

# xoff=0
# yoff=0
# for x in range(4):
#     for y in range(4):
#         for z in range(4):
#             a = D.add_ref(A)
#             b = D.add_ref(A)
#             a.center=(xoff,yoff)
#             b.center=(xoff+(points[y])[0],yoff+(points[y])[1])
#             C = route_manhattan(bendType='gradual',port1=a.ports[z+1],port2=b.ports[x+1])
#             c=D.add_ref(C)
#             yoff+=600
#         yoff+=600
#     xoff+=600
#     yoff=0

# quickplot(D)
