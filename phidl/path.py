# TODO
# phidl: finish variable offset/width functiaonlity
# phidl: change group tutorial to rotate by 90 degrees



# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
import warnings
from phidl.device_layout import Path, _rotate_points


def arc(radius = 10, angle = 90, num_pts = 720):
    """ Create a circular arc Path

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
    num_pts = abs(int(num_pts*angle/360))
    t = np.linspace(-90*np.pi/180, (angle-90)*np.pi/180, num_pts)
    x = radius*np.cos(t)
    y = radius*(np.sin(t)+1)
    points = np.array((x,y)).T

    P = Path()
    # Manually add points & adjust start and end angles
    P.points = points
    P.start_angle = 180*(angle<0)
    P.end_angle = P.start_angle + angle
    return P


def straight(length = 5, num_pts = 100):
    """ Creates a straight Path

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
    y = x*0
    points = np.array((x,y)).T
    
    P = Path()
    P.append(points)
    return P


#### Euler bends

def _cumtrapz(x):
    """ Numpy-based implementation of the cumulative trapezoidal integration 
    function usually found in scipy (scipy.integrate.cumtrapz) """
    return np.cumsum((x[1:] + x[:-1])/2)


def _fresnel(R0, s, num_pts, n_iter=8):
    """ Fresnel integral using a series expansion """
    t = np.linspace(0,s/(np.sqrt(2)*R0), num_pts)
    x = np.zeros(num_pts)
    y = np.zeros(num_pts)
    
    for n in range(0,n_iter):
      x += (-1)**n * t**(4*n+1)/(np.math.factorial(2*n) * (4*n+1))
      y += (-1)**n * t**(4*n+3)/(np.math.factorial(2*n+1) * (4*n+3))
    
    x = np.concatenate([[0], x])
    y = np.concatenate([[0], y])
    return np.array([np.sqrt(2)*R0*x, np.sqrt(2)*R0*y])


def euler(radius = 3, angle = 90, p = 1.0, use_eff = False, num_pts = 720):
    """ Create an Euler bend (also known as "racetrack" or "clothoid" curve)
    with smoothly varying curvature.  By default, `radius` corresponds to the
    minimum radius of curvature of the bend. However, if `use_eff` is set to
    True, `radius` corresponds to the effective radius of curvature (making the
    curve a drop-in replacement for an arc). If p < 1.0, will create a "partial
    euler" curve as described in Vogelbacher et. al.
    https://dx.doi.org/10.1364/oe.27.031394

    Parameters
    ----------
    angle : int or float
        Total angle of curve
    radius : int or float
        Minimum radius of curvature
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
        raise ValueError('[PHIDL] euler() requires argument `p` be between 0 and 1')
    if p == 0:
        P = arc(radius = radius, angle = angle, num_pts = num_pts)
        P.info['Reff'] = radius
        P.info['Rmin'] = radius
        return P

    if angle < 0:
        mirror = True
        angle = np.abs(angle)
    else:
        mirror = False
        
    R0 = 1
    alpha = np.radians(angle)
    Rp = R0 / (np.sqrt(p*alpha))
    sp = R0 * np.sqrt(p*alpha)
    s0 = 2*sp + Rp*alpha*(1-p)
    num_pts = abs(int(num_pts*angle/360))
    num_pts_euler = int(np.round(sp/(s0/2)*num_pts))
    num_pts_arc = num_pts - num_pts_euler

    xbend1, ybend1 = _fresnel(R0, sp, num_pts_euler)
    xp, yp = xbend1[-1], ybend1[-1]

    dx = xp - Rp*np.sin(p*alpha/2)
    dy = yp - Rp*(1-np.cos(p*alpha/2))

    s = np.linspace(sp, s0/2, num_pts_arc)
    xbend2 = Rp*np.sin((s-sp)/Rp + p*alpha/2) + dx
    ybend2 = Rp*(1 - np.cos((s-sp)/Rp + p*alpha/2)) + dy

    x = np.concatenate([xbend1, xbend2[1:]])
    y = np.concatenate([ybend1, ybend2[1:]])
    points1 = np.array([x,y]).T
    points2 = np.flipud(np.array([x,-y]).T)

    points2 = _rotate_points(points2, angle-180)
    points2 += -points2[0,:] + points1[-1,:]

    points = np.concatenate([points1[:-1],points2])

    # Find y-axis intersection point to compute Reff
    start_angle = 180*(angle<0)
    end_angle = start_angle + angle
    dy = np.tan(np.radians(end_angle-90)) * points[-1][0]
    Reff = points[-1][1] - dy
    Rmin = Rp

    # Fix degenerate condition at angle == 180
    if np.abs(180-angle) < 1e-3:
        Reff = points[-1][1]/2

    # Scale curve to either match Reff or Rmin
    if use_eff == True:
        scale = radius/Reff
    else:
        scale = radius/Rmin
    points *= scale

    P = Path()
    # Manually add points & adjust start and end angles
    P.points = points
    P.start_angle = start_angle
    P.end_angle = end_angle
    P.info['Reff'] = Reff*scale
    P.info['Rmin'] = Rmin*scale
    if mirror == True:
        P.mirror((1,0))
    return P


# def spiral(num_turns = 3.25, gap = 1, inner_gap = 9, num_pts = 720):
#     # FIXME: Every 0.25 num_turns = 0.125 actual turns
#     num_pts = abs(int(num_pts*num_turns*360))
#     num_turns1 = np.floor(num_turns)
#     if (num_turns % 2) == 0:
#         num_turns1 -= 1
#     a1 = np.pi*num_turns1 + np.pi/2
#     a2 = np.pi*num_turns + np.pi/2
#     a = np.array([np.linspace(0, a1, num_pts),
#                     np.concatenate([np.linspace(0, a1, num_pts),
#                                     np.arange(a1,a2, a1/(num_pts-1))[1:]])])
#     i1 = np.argmax(a[0] > np.pi/2)
#     i2 = [len(x) for x in a]
#     r = np.array([np.ones(i2[0]), np.ones(i2[1])])
#     for i in range(2):
#         r[i][:i1] = inner_gap/2 * np.sin(a[0][:i1])
#         r[i][i1:i2[0]] = inner_gap/2 + (a[0][i1:i2[0]] - np.pi/2)/np.pi*gap
#     if i2[0] == 0 or i2[1] != 0:
#         r[1][i2[0]:] = inner_gap/2 + (a[1][i2[0]:] - np.pi/2)/np.pi*gap
#     else: pass
#     a, r = np.concatenate([[np.flip(a[1]), -np.flip(r[1])], [a[0], r[0]]],
#                             axis = 1)
#     x = r * np.cos(a); y = r * np.sin(a)
#     points = np.array((x,y)).T

#     P = Path()
#     # Manually add points & adjust start and end angles
#     P.points = points[::-1]
#     P.start_angle = 180
#     P.end_angle = np.mod(360*num_turns, 360)
#     return P