# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
import warnings
from phidl.device_layout import Path


def arc(radius = 10, angle = 90, num_pts = 720):
    """ Produces an arc of points with `num_pts` per 360 degrees """
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
    """ Produces an straight section """
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

def partial_euler(angle = 90, Rmin = 3, p = 0.2, num_pts = 720):
    num_pts = abs(int(num_pts*angle/360))
    # Overhead calculations
    angle = np.radians(angle)
    sp = np.sqrt(p*angle)      # Clothoid-to-normal transition point s value
    s0 = angle*Rmin + sp    # Total path length derived from curvature integral = angle
    eta = 1 / (2*sp*Rmin) # Scaling factor to enforce Rmin
    # Constructing s and K arrays
    s = np.linspace(0, s0, num_pts)
    K = np.zeros(num_pts)
    if p == 0: K += 1/Rmin
    else:
        i1 = np.argmax(s > sp)
        i2 = np.argmax(s >= s0 - sp)
        K = eta * np.concatenate([np.multiply(np.ones(i1), 2*s[:i1]),
                                np.multiply(np.ones(i2-i1), 2*sp),
                                np.multiply(np.ones(num_pts-i2), 
                                            2*(s0 - s[i2:num_pts]))])
    # Integrating to find x and y
    ds = s[1] - s[0]
    phi = _cumtrapz(K*ds)
    x, y = np.concatenate([np.array([[0],[0]]), 
                           np.cumsum([ds*np.cos(phi),
                           ds*np.sin(phi)], axis = 1)],
                           axis = 1)
    points = np.array((x,y)).T

    P = Path()
    P.append(points)
    return P

