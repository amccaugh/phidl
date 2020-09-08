# TODO
# phidl: finish variable offset/width functiaonlity
# phidl: change group tutorial to rotate by 90 degrees



# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
import warnings
from phidl.device_layout import Path


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


def euler(Rmin = 3, angle = 90, p = 1.0, num_pts = 720):
    """ Create an Euler curve (also known as "racetrack" or "clothoid" curve)
    with smoothly varying curvature and with a miminum radius of curvature equal
    to Rmin.  If p < 1.0, will create a "partial euler" curve as described in
    Vogelbacher et. al. https://dx.doi.org/10.1364/oe.27.031394

    Parameters
    ----------
    angle : int or float
        Total angle of curve
    Rmin : int or float
        Minimum radius of curvature
    p : float
        Proportion of curve that is an Euler curve
    num_pts : int
        Number of points used per 360 degrees

    Returns
    -------
    Path
        A Path object with the specified Euler curve
    """
    if (p < 0) or (p > 1):
        raise ValueError('[PHIDL] euler() requires argument `p` be between 0 and 1')

    num_pts = abs(int(num_pts*angle/360))
    num_pts_euler = int(np.round(num_pts/2*p))

    K1 = np.linspace(0, 1/Rmin, num_pts_euler)
    K2 = np.ones(num_pts - 2*num_pts_euler)/Rmin
    K3 = np.linspace(1/Rmin, 0, num_pts_euler)
    K = np.concatenate([K1,K2,K3])
    # Integrating to find x and y
    ds = np.radians(angle)/np.trapz(K)
    phi = _cumtrapz(K*ds)
    x, y = np.concatenate([np.array([[0],[0]]), 
                        np.cumsum([ds*np.cos(phi),
                        ds*np.sin(phi)], axis = 1)],
                        axis = 1)
    points = np.array((x,y)).T

    P = Path()
    P.append(points)
    return P


def spiral(num_turns = 3.25, gap = 1, inner_gap = 9, num_pts = 720):
    # FIXME: Every 0.25 num_turns = 0.125 actual turns
    num_pts = abs(int(num_pts*num_turns*360))
    num_turns1 = np.floor(num_turns)
    if (num_turns % 2) == 0:
        num_turns1 -= 1
    a1 = np.pi*num_turns1 + np.pi/2
    a2 = np.pi*num_turns + np.pi/2
    a = np.array([np.linspace(0, a1, num_pts),
                    np.concatenate([np.linspace(0, a1, num_pts),
                                    np.arange(a1,a2, a1/(num_pts-1))[1:]])])
    i1 = np.argmax(a[0] > np.pi/2)
    i2 = [len(x) for x in a]
    r = np.array([np.ones(i2[0]), np.ones(i2[1])])
    for i in range(2):
        r[i][:i1] = inner_gap/2 * np.sin(a[0][:i1])
        r[i][i1:i2[0]] = inner_gap/2 + (a[0][i1:i2[0]] - np.pi/2)/np.pi*gap
    if i2[0] == 0 or i2[1] != 0:
        r[1][i2[0]:] = inner_gap/2 + (a[1][i2[0]:] - np.pi/2)/np.pi*gap
    else: pass
    a, r = np.concatenate([[np.flip(a[1]), -np.flip(r[1])], [a[0], r[0]]],
                            axis = 1)
    x = r * np.cos(a); y = r * np.sin(a)
    points = np.array((x,y)).T

    P = Path()
    # Manually add points & adjust start and end angles
    P.points = points[::-1]
    P.start_angle = 180
    P.end_angle = np.mod(360*num_turns, 360)
    return P