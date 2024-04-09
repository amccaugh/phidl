import numpy as np

import phidl.path as pp
from phidl import Path
from phidl.device_layout import CrossSection, Device, Port


def _get_rotated_basis(theta):
    """Returns basis vectors rotated CCW by theta (in degrees)"""
    theta = np.radians(theta)
    e1 = np.array([np.cos(theta), np.sin(theta)])
    e2 = np.array([-1 * np.sin(theta), np.cos(theta)])
    return e1, e2


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
    **kwargs,
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
    **kwargs,
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
