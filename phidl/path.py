from __future__ import absolute_import, division, print_function

import numpy as np

from phidl.device_layout import CrossSection, Path, _rotate_points


def arc(radius=10, angle=90, num_pts=720):
    """Create a circular arc Path

    Parameters
    ----------
    radius : int or float
        Radius of arc
    angle : int or float
        Total angle of arc
    num_pts : int
        Number of points used per 360 degrees

    Returns
    -------
    Path
        A Path object with the specified arc
    """
    num_pts = abs(int(num_pts * angle / 360))
    t = np.linspace(-90 * np.pi / 180, (angle - 90) * np.pi / 180, num_pts)
    x = radius * np.cos(t)
    y = radius * (np.sin(t) + 1)
    points = np.array((x, y)).T * np.sign(angle)

    P = Path()
    # Manually add points & adjust start and end angles
    P.points = points
    P.start_angle = 0
    P.end_angle = angle
    return P


def straight(length=5, num_pts=100):
    """Creates a straight Path

    Parameters
    ----------
    length : int or float
        Total length of straight path
    num_pts : int
        Number of points along Path

    Returns
    -------
    Path
        A Path object with the specified straight section
    """
    x = np.linspace(0, length, num_pts)
    y = x * 0
    points = np.array((x, y)).T

    P = Path()
    P.append(points)
    return P


#### Euler bends


def _cumtrapz(x):
    """Numpy-based implementation of the cumulative trapezoidal integration
    function usually found in scipy (scipy.integrate.cumtrapz)"""
    return np.cumsum((x[1:] + x[:-1]) / 2)


def _fresnel(R0, s, num_pts, n_iter=8):
    """Fresnel integral using a series expansion"""
    t = np.linspace(0, s / (np.sqrt(2) * R0), num_pts)
    x = np.zeros(num_pts)
    y = np.zeros(num_pts)

    for n in range(0, n_iter):
        x += (-1) ** n * t ** (4 * n + 1) / (np.math.factorial(2 * n) * (4 * n + 1))
        y += (-1) ** n * t ** (4 * n + 3) / (np.math.factorial(2 * n + 1) * (4 * n + 3))

    return np.array([np.sqrt(2) * R0 * x, np.sqrt(2) * R0 * y])


def euler(radius=3, angle=90, p=1.0, use_eff=False, num_pts=720):
    """Create an Euler bend (also known as "racetrack" or "clothoid" curves)
    that adiabatically transitions from straight to curved.  By default,
    `radius` corresponds to the minimum radius of curvature of the bend.
    However, if `use_eff` is set to True, `radius` corresponds to the effective
    radius of curvature (making the curve a drop-in replacement for an arc). If
    p < 1.0, will create a "partial euler" curve as described in Vogelbacher et.
    al. https://dx.doi.org/10.1364/oe.27.031394

    Parameters
    ----------
    radius : int or float
        Minimum radius of curvature
    angle : int or float
        Total angle of curve
    p : float
        Proportion of curve that is an Euler curve
    use_eff : bool
        If False: `radius` corresponds to minimum radius of curvature of the bend
        If True: The curve will be scaled such that the endpoints match an arc
        with parameters `radius` and `angle`
    num_pts : int
        Number of points used per 360 degrees

    Returns
    -------
    Path
        A Path object with the specified Euler curve
    """
    if (p < 0) or (p > 1):
        raise ValueError("[PHIDL] euler() requires argument `p` be between 0 and 1")
    if p == 0:
        P = arc(radius=radius, angle=angle, num_pts=num_pts)
        P.info["Reff"] = radius
        P.info["Rmin"] = radius
        return P

    if angle < 0:
        mirror = True
        angle = np.abs(angle)
    else:
        mirror = False

    R0 = 1
    alpha = np.radians(angle)
    Rp = R0 / (np.sqrt(p * alpha))
    sp = R0 * np.sqrt(p * alpha)
    s0 = 2 * sp + Rp * alpha * (1 - p)
    num_pts = abs(int(num_pts * angle / 360))
    num_pts_euler = int(np.round(sp / (s0 / 2) * num_pts))
    num_pts_arc = num_pts - num_pts_euler

    xbend1, ybend1 = _fresnel(R0, sp, num_pts_euler)
    xp, yp = xbend1[-1], ybend1[-1]

    dx = xp - Rp * np.sin(p * alpha / 2)
    dy = yp - Rp * (1 - np.cos(p * alpha / 2))

    s = np.linspace(sp, s0 / 2, num_pts_arc)
    xbend2 = Rp * np.sin((s - sp) / Rp + p * alpha / 2) + dx
    ybend2 = Rp * (1 - np.cos((s - sp) / Rp + p * alpha / 2)) + dy

    x = np.concatenate([xbend1, xbend2[1:]])
    y = np.concatenate([ybend1, ybend2[1:]])
    points1 = np.array([x, y]).T
    points2 = np.flipud(np.array([x, -y]).T)

    points2 = _rotate_points(points2, angle - 180)
    points2 += -points2[0, :] + points1[-1, :]

    points = np.concatenate([points1[:-1], points2])

    # Find y-axis intersection point to compute Reff
    start_angle = 180 * (angle < 0)
    end_angle = start_angle + angle
    dy = np.tan(np.radians(end_angle - 90)) * points[-1][0]
    Reff = points[-1][1] - dy
    Rmin = Rp

    # Fix degenerate condition at angle == 180
    if np.abs(180 - angle) < 1e-3:
        Reff = points[-1][1] / 2

    # Scale curve to either match Reff or Rmin
    if use_eff:
        scale = radius / Reff
    else:
        scale = radius / Rmin
    points *= scale

    P = Path()
    # Manually add points & adjust start and end angles
    P.points = points
    P.start_angle = start_angle
    P.end_angle = end_angle
    P.info["Reff"] = Reff * scale
    P.info["Rmin"] = Rmin * scale
    if mirror:
        P.mirror((1, 0))
    return P


def spiral(num_turns=5, gap=1, inner_gap=2, num_pts=10000):
    """Creates a spiral geometry consisting of two oddly-symmetric
    semi-circular arcs in the centre and two Archimedean (involute) spiral arms
    extending outward from the ends of both arcs.

    Parameters
    ----------
    num_turns : int or float
        The number of turns in the spiral. Must be greater than 1. A full
        spiral rotation counts as 1 turn, and the center arcs will together
        always be 0.5 turn.
    gap : int or float
        The distance between any point on one arm of the spiral and a point
        with the same angular coordinate on an adjacent arm.
    inner_gap : int or float
        The inner size of the spiral, equal to twice the chord length of the
        centre arcs.
    num_pts: int
        The number of points in the entire spiral. The actual number of points
        will be slightly different than the specified value, as they are
        dynamically allocated using the path lengths of the spiral.

    Returns
    -------
    Path
        A Path object forming a spiral

    Notes
    -----
    ``num_turns`` usage (x is any whole number):
        - ``num_turns = x.0``: Output arm will be extended 0.5 turn to be on
        the same side as the input.
        - ``num_turns < x.5``: Input arm will be extended by the fractional
        amount.
        - ``num_turns = x.5``: Both arms will be the same length and the input
        and output will be on opposite sides.
        - ``num_turns > x.5``: Output arm will be extended by the fractional
        amount.
    """
    # Establishing number of turns in each arm
    if num_turns <= 1:
        raise ValueError("num_turns must be greater than 1")
    diff = num_turns - np.floor(num_turns)
    if diff < 0.5:
        num_turns1 = np.floor(num_turns) - 1 + 2 * diff
    else:
        num_turns1 = np.floor(num_turns)
    if diff > 0.5:
        num_turns2 = np.floor(num_turns) - 1 + 2 * diff
    else:
        num_turns2 = np.floor(num_turns)

    # Establishing relevant angles and spiral/centre arc parameters
    a1 = np.pi / 2
    a2 = np.array([np.pi * num_turns1 + a1, np.pi * num_turns2 + a1])
    a = inner_gap / 2 - gap / 2
    b = gap / np.pi
    Rc = inner_gap * np.sqrt(1 + (b / (a + b * a1)) ** 2) / 4
    theta = np.degrees(2 * np.arcsin(inner_gap / 4 / Rc))

    # Establishing number of points in each arm
    s_centre = Rc * np.radians(theta)
    s_spiral = ((a + a2 * b) ** 2 + b**2) ** (3 / 2) / (3 * (a * b + (a2 * b**2)))
    z = num_pts / (s_spiral[0] + s_spiral[1] + 2 * s_centre)
    num_pts0 = int(z * s_centre)
    num_pts1 = int(z * s_spiral[0])
    num_pts2 = int(z * s_spiral[1]) - num_pts1

    # Forming both spiral arms
    arm1 = np.linspace(a1, a2[0], num_pts1)
    arm2 = np.linspace(a2[0], a2[1], num_pts2)[1:]
    a_spiral = np.array([arm1, np.concatenate([arm1, arm2])])
    r_spiral = a + b * a_spiral
    x_spiral = np.array([np.zeros(num_pts1), np.zeros(len(a_spiral[1]))])
    y_spiral = np.array([np.zeros(num_pts1), np.zeros(len(a_spiral[1]))])
    for i in range(2):
        x_spiral[i] = r_spiral[i] * np.cos(a_spiral[i])
        y_spiral[i] = r_spiral[i] * np.sin(a_spiral[i])

    # Forming centre arcs
    pts = _rotate_points(arc(Rc, theta, 360 * num_pts0 / theta).points, -theta / 2 + 90)
    x_centre = pts[:, 0] + x_spiral[0][0] - pts[:, 0][-1]
    y_centre = pts[:, 1] + y_spiral[0][0] - pts[:, 1][-1]
    x_centre = np.concatenate([-np.flip(x_centre), x_centre[1:]])
    y_centre = np.concatenate([-np.flip(y_centre), y_centre[1:]])

    # Combining into final spiral
    x = np.concatenate([-np.flip(x_spiral[1]), x_centre[1:-1], x_spiral[0]])
    y = np.concatenate([-np.flip(y_spiral[1]), y_centre[1:-1], y_spiral[0]])
    points = np.array((x, y)).T

    P = Path()
    # Manually add points & adjust start and end angles
    P.points = points
    nx1, ny1 = points[1] - points[0]
    P.start_angle = np.arctan2(ny1, nx1) / np.pi * 180
    nx2, ny2 = points[-1] - points[-2]
    P.end_angle = np.arctan2(ny2, nx2) / np.pi * 180

    return P


def _compute_segments(points):
    points = np.asfarray(points)
    normals = np.diff(points, axis=0)
    normals = (normals.T / np.linalg.norm(normals, axis=1)).T
    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])
    ds = np.sqrt(dx**2 + dy**2)
    theta = np.degrees(np.arctan2(dy, dx))
    dtheta = np.diff(theta)
    dtheta = dtheta - 360 * np.floor((dtheta + 180) / 360)
    return points, normals, ds, theta, dtheta


def smooth(
    points=[
        (20, 0),
        (40, 0),
        (80, 40),
        (80, 10),
        (100, 10),
    ],
    radius=4,
    corner_fun=euler,
    **kwargs
):
    """Create a smooth path from a series of waypoints. Corners will be rounded
    using `corner_fun` and any additional key word arguments (for example,
    `use_eff = True` when `corner_fun = pp.euler`)

    Parameters
    ----------
    points : array-like[N][2] or Path
        List of waypoints for the path to follow
    radius : int or float
        Radius of curvature, this argument will be passed to `corner_fun`
    corner_fun : function
        The function that controls how the corners are rounded. Typically either
        `arc()` or `euler()`
    **kwargs : dict
        Extra keyword arguments that will be passed to `corner_fun`

    Returns
    -------
    Path
        A Path object with the specified smoothed path.
    """

    if isinstance(points, Path):
        points = points.points

    points, normals, ds, theta, dtheta = _compute_segments(points)
    colinear_elements = np.concatenate([[False], np.abs(dtheta) < 1e-6, [False]])
    if np.any(colinear_elements):
        new_points = points[~colinear_elements, :]
        points, normals, ds, theta, dtheta = _compute_segments(new_points)

    if np.any(np.abs(np.abs(dtheta) - 180) < 1e-6):
        raise ValueError(
            "[PHIDL] smooth() received points which double-back on themselves"
            + "--turns cannot be computed when going forwards then exactly backwards."
        )

    # FIXME add caching
    # Create arcs
    paths = []
    radii = []
    for dt in dtheta:
        P = corner_fun(radius=radius, angle=dt, **kwargs)
        chord = np.linalg.norm(P.points[-1, :] - P.points[0, :])
        r = (chord / 2) / np.sin(np.radians(dt / 2))
        r = np.abs(r)
        radii.append(r)
        paths.append(P)

    d = np.abs(np.array(radii) / np.tan(np.radians(180 - dtheta) / 2))
    encroachment = np.concatenate([[0], d]) + np.concatenate([d, [0]])
    if np.any(encroachment > ds):
        raise ValueError(
            "[PHIDL] smooth(): Not enough distance between points to to fit curves.  Try reducing the radius or spacing the points out farther"
        )
    p1 = points[1:-1, :] - normals[:-1, :] * d[:, np.newaxis]

    # Move arcs into position
    new_points = []
    new_points.append([points[0, :]])
    for n, dt in enumerate(dtheta):
        P = paths[n]
        P.rotate(theta[n] - 0)
        P.move(p1[n])
        new_points.append(P.points)
    new_points.append([points[-1, :]])
    new_points = np.concatenate(new_points)

    P = Path()
    P.rotate(theta[0])
    P.append(new_points)
    P.move(points[0, :])

    return P


def _sinusoidal_transition(y1, y2):
    dx = y2 - y1
    return lambda t: y1 + (1 - np.cos(np.pi * t)) / 2 * dx


def _linear_transition(y1, y2):
    dx = y2 - y1
    return lambda t: y1 + t * dx


def transition(cross_section1, cross_section2, width_type="sine"):
    """Creates a CrossSection that smoothly transitions between two input
    CrossSections. Only cross-sectional elements that have the `name` (as in
    X.add(..., name = 'wg') ) parameter specified in both input CrosSections
    will be created. Port names will be cloned from the input CrossSections in
    reverse.

    Parameters
    ----------
    cross_section1 : CrossSection
        First input CrossSection
    cross_section2 : CrossSection
        Second input CrossSection
    width_type : {'sine', 'linear'}
        Sets the type of width transition used if any widths are different
        between the two input CrossSections.

    Returns
    -------
    CrossSection
        A smoothly-transitioning CrossSection
    """

    X1 = cross_section1
    X2 = cross_section2
    Xtrans = CrossSection()

    if not X1.aliases or not X2.aliases:
        raise ValueError(
            """[PHIDL] transition() found no named sections in one
        or both inputs (cross_section1/cross_section2)."""
        )

    for alias in X1.aliases.keys():
        if alias in X2.aliases:

            offset1 = X1[alias]["offset"]
            offset2 = X2[alias]["offset"]
            width1 = X1[alias]["width"]
            width2 = X2[alias]["width"]

            if callable(offset1):
                offset1 = offset1(1)
            if callable(offset2):
                offset2 = offset2(0)
            if callable(width1):
                width1 = width1(1)
            if callable(width2):
                width2 = width2(0)

            offset_fun = _sinusoidal_transition(offset1, offset2)

            if width_type == "sine":
                width_fun = _sinusoidal_transition(width1, width2)
            elif width_type == "linear":
                width_fun = _linear_transition(width1, width2)
            else:
                raise ValueError(
                    "[PHIDL] transition() width_type "
                    + "argument must be one of {'sine','linear'}"
                )

            Xtrans.add(
                width=width_fun,
                offset=offset_fun,
                layer=X1[alias]["layer"],
                ports=(X2[alias]["ports"][0], X1[alias]["ports"][1]),
                name=alias,
            )

    return Xtrans
