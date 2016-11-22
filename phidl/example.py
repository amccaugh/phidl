# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:28:09 2016

@author: smb2
"""

from __future__ import division # Makes it so 1/4 = 0.25 instead of zero


from phidl import Device, quickplot
import numpy as np
import phidl.geometry as pg
import basicPhotonics as bP

#
##
interaction_length = 10
gap1 = 1
gap2 = 0.1
port1 = 0.4
port2 = 0.5
port3 = 0.4
port4 = 0.4
hSines = 3
rMin = 10
lSines = np.sqrt((np.pi**2)*hSines*rMin/2)

#
D = Device("adiabatic beam splitter")
#
# start with the actual beamsplitter part
xpts_upper = [0, 0, interaction_length, interaction_length]
ypts_upper = [0, port1, port3, 0]

wg_upper = D.add_ref(bP.polygon(xcoords = xpts_upper, ycoords = ypts_upper, layer = 0))
#
xpts_lower = [0, 0, interaction_length, interaction_length]
ypts_lower = [-gap1, -gap1-port2, -gap2-port4, -gap2]
wg_lower = D.add_ref(bP.polygon(xcoords = xpts_lower, ycoords = ypts_lower, layer = 0))

#locate the straight sections after the sine bends
P = Device('ports')
P.add_port(name = 'port 1', midpoint = [wg_upper.xmin-lsines, wg_upper.center[1]+hsines], width = port1, orientation = 0)
P.add_port(name = 'port 2', midpoint = [wg_lower.xmin-lsines, wg_lower.center[1]-hsines], width = port1, orientation = 0)
P.add_port(name = 'port 3', midpoint = [wg_upper.xmax+lsines, wg_upper.center[1]+hsines], width = port1, orientation = 180)
P.add_port(name = 'port 4', midpoint = [wg_lower.xmax+lsines, wg_lower.center[1]-hsines], width = port1, orientation = 180)
route1 = D.add_ref(pg.route(port1 = P.ports['port 1'], port2 = wg_upper.ports['0'], path_type = 'sine'))
route2 = D.add_ref(pg.route(port1 = P.ports['port 2'], port2 = wg_lower.ports['0'], path_type = 'sine'))
route3 = D.add_ref(pg.route(port1 = P.ports['port 3'], port2 = wg_upper.ports['2'], path_type = 'sine'))
route4 = D.add_ref(pg.route(port1 = P.ports['port 4'], port2 = wg_lower.ports['2'], path_type = 'sine'))

D.add_port(port = route1.ports[1], name = 'port 1')
D.add_port(port = route2.ports[1], name = 'port 2')
D.add_port(port = route3.ports[1], name = 'port 3')
D.add_port(port = route4.ports[1], name = 'port 4')


quickplot(D)

#wg = D.add_ref(pg.compass(size=[lWg,wWg],layer=wgLayer))
#wRegion = D.add_ref(pg.compass(size=[lWg,wE],layer=wLayer))
#pRegion = D.add_ref(pg.compass(size=[lWg,wDope],layer=pLayer))
#nRegion = D.add_ref(pg.compass(size=[lWg,wDope],layer=nLayer))
#taper = D.add_ref(pg.taper(length = lTaper, width1 = wWg, width2 = wTaper))
#
#taper.xmin = wg.xmax
#wg.connect(port = 'W', destination = taper.ports[1])
#wg.center = wRegion.center
#pRegion.ymin = wRegion.ymax + wDopeOffset
#pRegion.center[0] = wRegion.center[0]
#nRegion.ymax = wRegion.ymin - wDopeOffset
#nRegion.center[0] = wRegion.center[0]
#D.add_port(port = taper.ports[2], name = 'LED')
#quickplot(D)

# returns a polygon with ports on all edges
#P = Device('polygon')
#xcoords =xpts_lower
#ycoords = ypts_lower
#P.add_polygon([xcoords, ycoords], layer = layer)
#n = len(xcoords)
#xcoords.append(xcoords[0])
#ycoords.append(ycoords[0]) 
# #determine if clockwise or counterclockwise
#cc = 0     
#for i in range(0,n):
#    cc += ((xcoords[i+1]-xcoords[i])*(ycoords[i+1]+ycoords[i]))
#        
#for i in range(0,n):
#    midpoint_n = [(xcoords[i+1]+xcoords[i])/2, (ycoords[i+1]+ycoords[i])/2]
#    test1 = xcoords[i+1]-xcoords[i]
#    test2 = ycoords[i+1]-ycoords[i]
#    
#    orientation_n = np.arctan2(np.sign(cc)*(xcoords[i+1]-xcoords[i]),np.sign(cc)*(ycoords[i]-ycoords[i+1]))*180/np.pi
#        
#    width_n = np.sqrt((xcoords[i+1]-xcoords[i])**2+(ycoords[i+1]-ycoords[i])**2)    
#    
#    P.add_port(name = str(i), midpoint = midpoint_n, width = width_n, orientation = orientation_n)
#    
#quickplot(P)
#
#g = D.add_ref(G)
#g2 = D.add_ref(G)
#
#g.connect(port = 1, destination = wg.ports['W'])
#g.xmax = wg.xmin
#g2.connect(port = 1, destination = wg.ports['E'])
#g2.xmin = wg.xmax
#
#D.add_ref(pg.route(port1 = g.ports[1], port2 = wg.ports['W']))
#D.add_ref(pg.route(port1 = g2.ports[1], port2 = wg.ports['E']))
#
#R = Device();
#
#for i, g in enumerate(gaps):
#    r = R.add_ref(pg.ring(radius = radius + dR*i, width = wRing, angle_resolution = 1, layer = 0)) 
#    r.move([period*i,0])
#    r.ymax = wg.ymin - g
#    rwg = R.add_ref(pg.compass(size=[10, wWg]))
#    rwg.move([period*i, 0])
#    rwg.ymax = r.ymin-g
#    rarc = R.add_ref(pg.arc(radius = 10, width = wWg, theta = 180, start_angle = 90, angle_resolution = 2.5, layer = 0))
#    rarc.xmax = rwg.xmin
#    rarc.ymax = rwg.ymax
#    rtap = R.add_ref(pg.taper(length = 10, width1 = wWg, width2 = 0.1, port = None, layer = 0))
#    rtap.xmin = rwg.xmax
#    rtap.ymax = rwg.ymax
#    
#    rwg2 = R.add_ref(pg.compass(size=[10, wWg]))
#    rwg2.xmin = rarc.xmax
#    rwg2.ymin = rarc.ymin
#    
#    g3 = R.add_ref(G)
#    g3.xmin = rwg2.xmax
#    g3.connect(port = 1, destination = rwg2.ports['E'])
#    
#
#
#R.x = D.x
#D.add_ref(R)
#
#quickplot(D)