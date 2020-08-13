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

# def fresnel(R0, s, num_pts = 1000):
#     t = np.linspace(0,s, num_pts)
#     dt = t[1]-t[0]
#     dxdt = np.cos(t**2/(2*R0**2))
#     dydt = np.sin(t**2/(2*R0**2))
#     x = _cumtrapz(dxdt)*dt
#     y = _cumtrapz(dydt)*dt
#     x = np.concatenate([[0], x])
#     y = np.concatenate([[0], y])
#     return x,y

def partial_euler(Reff = 3, a = 90, p = 0.2, num_pts = 1000, *args, **kwargs):
    """ Taken from "Analysis of silicon nitride partial Euler waveguide bends"
    by Vogelbacher et. al. https://doi.org/10.1364/OE.27.031394 """
    η = kwargs.get('η', 1)
    j = kwargs.get('j', 0)
    if a <= 0 or a > 263: 
        raise ValueError("'a' must be a float such that 0 < a ≤ 263.")
    # Overhead calculations
    a     = np.radians(a)
    asp   = p * a / 2
    Rp    = 1 / 2 / np.sqrt(asp) * (j * Reff * η + 1 - j)
    sp    = np.sqrt(2 * asp)     * (j * Reff * η + 1 - j)
    s0    = 2 * sp + Rp * a * (1 - p)
    scale = a / (2 * sp * (s0 - sp))
    if p == 0: s0 = a * (j * Reff * η + 1 - j); scale = a / s0
    # Constructing s and K arrays
    s = np.linspace(0, s0, num_pts)
    K = np.zeros(num_pts)
    if p == 0: K += 1
    else:
        i1 = np.argmax(s > sp)
        i2 = np.argmax(s >= s0 - sp)
        K  = np.concatenate([np.multiply(np.ones(i1), 2 * s[:i1]),
                             np.multiply(np.ones(i2 - i1),2 * sp),
                             np.multiply(np.ones(num_pts - i2), 
                                                 2 * (s0 - s[i2:num_pts]))])
    K *= scale * ((1 - j) / Reff + j)
    s *= Reff  * (1 - j) + j
    # Integrating to find x and y
    ds = s[1] - s[0]
    φ = _cumtrapz(K * ds)
    x = np.cumsum(ds * np.cos(φ))
    y = np.cumsum(ds * np.sin(φ))
    x = np.concatenate([[0], x])
    y = np.concatenate([[0], y])
    # Calculating η rescaling factor
    middle = int((num_pts - 1) / 2)
    η      = Reff / (y[middle] + x[middle] / np.tan(a / 2))
    if j == 1: return x, y
    else: return partial_euler(Reff, np.degrees(a), p, num_pts, η = η, j = 1)

