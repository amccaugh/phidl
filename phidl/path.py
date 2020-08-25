# TODO
# phidl: finish variable offset/width functiaonlity
# phidl: change group tutorial to rotate by 90 degrees



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