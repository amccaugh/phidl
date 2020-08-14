# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
import warnings


def arc(radius = 10, angle = 90, num_pts = 720):
    """ Produces an arc of points with `num_pts` per 360 degrees.  An extra point is
    tacked on each end to ensure that the numerical gradient is accurate """
    t = np.linspace(0, angle*np.pi/180, abs(int(num_pts*angle/360))-2)
    x = radius*np.cos(t)
    y = radius*np.sin(t)
    points = np.array((x,y)).T

    start_angle = 90*np.sign(angle)
    end_angle = start_angle + angle
    
    return points, start_angle, end_angle


def straight(length = 5, num_pts = 100):
    """ Produces an straight section """
    x = np.linspace(0, length, num_pts)
    y = x*0
    return np.array((x,y)).T


#### Euler bends

def _cumtrapz(x):
    """ Numpy-based implementation of the cumulative trapezoidal integration 
    function usually found in scipy (scipy.integrate.cumtrapz) """
    return np.cumsum((x[1:] + x[:-1])/2)

def partial_euler(Rmin = 3, a = 90, p = 0.2, num_pts = 4000):
    # Overhead calculations
    a = np.radians(a)
    sp = np.sqrt(p*a)      # Clothoid-to-normal transition point s value
    s0 = a*Rmin + sp    # Total path length derived from curvature integral = a
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
    phi = cumtrapz(K*ds)
    x, y = np.concatenate([np.array([[0],[0]]), 
                           np.cumsum([ds*cos(phi), ds*sin(phi)], axis = 1)],
                          axis = 1)
    return x, y

