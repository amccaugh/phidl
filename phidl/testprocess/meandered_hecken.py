# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:33:25 2016

@author: anm16
"""

from __future__ import division, print_function, absolute_import
from phidl import Device, quickplot
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
        D.add_polygon((xpts,ypts), layer = 0)
        D.add_port(name = 1, midpoint = (0,0), width = widths[0], orientation = 180)
        D.add_port(name = 2, midpoint = (length,0), width = widths[-1], orientation = 0)
        return D
        
    def arc_tapered(radius = 10, width1 = 1, width2 = 2, theta = 45, angle_resolution = 2.5, layer = 0):
        D = Device()
        path1 = gdspy.Path(width = width1, initial_point = (0, 0))
        path1.turn(radius = radius, angle = theta*np.pi/180, number_of_points=int(abs(2*theta/angle_resolution)), final_width = width2)
        [D.add_polygon(p, layer = layer) for p in path1.polygons]
        D.add_port(name = 1, midpoint = (0, 0), width = width1, orientation = 180)
        D.add_port(name = 2, midpoint = (path1.x, path1.y), width = width2, orientation = path1.direction*180/np.pi)
        return D
    
        
    D = Device('meander-taper')
    xpos1 = min(x_taper)
    xpos2 = min(x_taper) + meander_length
    t = D.add_ref( taper_section(x_start = xpos1, x_end = xpos2, num_pts = 50) )
    D.add_port(t.ports[1])
    dir_toggle = -1
    while xpos2 < max(x_taper):
        arc_width1 = taper_width(xpos2)
        arc_radius = max(spacing_factor*arc_width1, min_spacing)
        arc_length = np.pi*arc_radius
        arc_width2 = taper_width(xpos2 + arc_length)
        a = D.add_ref(  arc_tapered(radius = arc_radius, width1 = arc_width1, width2 = arc_width2, theta = 180*dir_toggle) )
        a.connect(port = 1, destination = t.ports[2])
        dir_toggle = -dir_toggle
        xpos1 = xpos2 + arc_length
        xpos2 = xpos1 + meander_length
        t = D.add_ref( taper_section(x_start = xpos1, x_end = xpos2, num_pts = 30) )
        t.connect(port = 1, destination = a.ports[2])
    D.add_port(t.ports[2])
        
    return D

mils2um = 25.4
in2um = 25.4e3
H = pg.hecken_taper(length = 870*in2um, B = 4.0091, dielectric_thickness = 1e3, eps_r = 4.6,
                 Lk_per_sq = 0, Z1 = 50, Z2 = 100,
                 num_pts = 99, layer = 0)


M = meander_taper(x_taper = H.meta['x'], w_taper = H.meta['w'],
                  meander_length = 18.5*in2um, spacing_factor = 3, min_spacing = 0.5)


print('Trace width start: %0.1f' % (H.meta['w'][0]/mils2um))
print('Trace width end:   %0.1f' % (H.meta['w'][-1]/mils2um))
print('X size: %0.3f inches' % (M.xsize/in2um))
print('Y size: %0.3f inches' % (M.ysize/in2um))
print('Cutoff frequency = %0.1f MHz' % (H.meta['f_cutoff']/1e6))

M.write_gds('Optimus01 Meandered Hecken Taper PCB.gds')

quickplot(M)

