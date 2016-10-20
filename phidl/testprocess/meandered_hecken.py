# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:33:25 2016

@author: anm16
"""

from __future__ import division, print_function, absolute_import
from phidl import Device, quickplot, inset
import phidl.geometry as pg
import numpy as np
import gdspy



def meander_taper(x_taper, w_taper, meander_length = 1000, spacing_factor = 3, min_spacing = 0.5):
#        
#    w_taper = H.meta['w'] # Widths of taper along x axis
#    x_taper = H.meta['x'] # X points along x axis
    
    
    def taper_width(x):
        return np.interp(x, x_taper, w_taper)
        
        
    def taper_section(x_start, x_end, num_pts = 30):
        D = Device()
        length =  x_end - x_start
        x = np.linspace(0, length, num_pts)
        widths = np.linspace(taper_width(x_start), taper_width(x_end), num_pts)
        xpts = np.concatenate([x, x[::-1]])
        ypts = np.concatenate([widths/2, -widths[::-1]/2])
        D.add_polygon((xpts,ypts), layer = 0, datatype = 0)
        D.add_port(name = 1, midpoint = (0,0), width = widths[0], orientation = 180)
        D.add_port(name = 2, midpoint = (length,0), width = widths[-1], orientation = 0)
        return D
        
    def arc_tapered(radius = 10, width1 = 1, width2 = 2, theta = 45, angle_resolution = 2.5, layer = 0, datatype = 0):
        D = Device()
        path1 = gdspy.Path(width = width1, initial_point = (0, 0))
        path1.turn(radius = radius, angle = theta*np.pi/180, number_of_points=int(abs(2*theta/angle_resolution)), final_width = width2)
        [D.add_polygon(p, layer = layer, datatype = datatype) for p in path1.polygons]
        D.add_port(name = 1, midpoint = (0, 0), width = width1, orientation = 180)
        D.add_port(name = 2, midpoint = (path1.x, path1.y), width = width2, orientation = path1.direction*180/np.pi)
        return D
    
        
    D = Device('meander-taper')
    xpos1 = min(x_taper)
    xpos2 = min(x_taper) + meander_length
    t = D.add_device( taper_section(x_start = xpos1, x_end = xpos2, num_pts = 50) )
    D.add_port(t.ports[1])
    dir_toggle = -1
    while xpos2 < max(x_taper):
        arc_width1 = taper_width(xpos2)
        arc_radius = max(spacing_factor*arc_width1, min_spacing)
        arc_length = np.pi*arc_radius
        arc_width2 = taper_width(xpos2 + arc_length)
        a = D.add_device(  arc_tapered(radius = arc_radius, width1 = arc_width1, width2 = arc_width2, theta = 180*dir_toggle) )
        a.connect(port = 1, destination = t.ports[2])
        dir_toggle = -dir_toggle
        xpos1 = xpos2 + arc_length
        xpos2 = xpos1 + meander_length
        t = D.add_device( taper_section(x_start = xpos1, x_end = xpos2, num_pts = 30) )
        t.connect(port = 1, destination = a.ports[2])
    D.add_port(t.ports[2])
        
    return D

H = pg.hecken_taper(length = 10000, B = 4.0091, dielectric_thickness = 0.25, eps_r = 2,
                  Lk_per_sq = 250e-12, Z1 = 50, width2 = 0.10,
                  num_pts = 199, layer = 0, datatype = 0)

M = meander_taper(x_taper = H.meta['x'], w_taper = H.meta['w'], meander_length = 1000, spacing_factor = 3, min_spacing = 2)