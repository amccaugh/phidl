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


d = Device('critically coupled rings')
lr = d.add_ref(bP.ccRings())

#quickplot(d)

radius = 10
gaps = [0.1, 0.2, 0.3]
wRing = 0.5 
wWg = 0.4
dR = 0.15 
period = 30

nrings = len(gaps)
lWg = (nrings + 1)*period

D = Device()
wg = D.add_ref(pg.compass(size=[lWg, wWg]))

G = Device()

# parameters for the gratings
nperiods = 20
gperiod = 0.75
ff = 0.5
wGrating = 20
lTaper = 20

for i in range(nperiods):
    cgrating = G.add_ref(pg.compass(size=[gperiod*ff,wGrating]))
    cgrating.x+=i*gperiod

p = G.add_port(port = cgrating.ports['E'], name = 1)
p.midpoint=p.midpoint+np.array([(1-ff)*gperiod,0])

g = D.add_ref(G)
g2 = D.add_ref(G)
g.connect(port = 1, destination = wg.ports['W'])
g.move([-1*lTaper, 0])
g2.connect(port = 1, destination = wg.ports['E'])
g2.move([lTaper, 0])

D.add_ref(pg.route(port1 = g.ports[1], port2 = wg.ports['W']))
D.add_ref(pg.route(port1 = g2.ports[1], port2 = wg.ports['E']))

R = Device();

for i, g in enumerate(gaps):
    r = R.add_ref(pg.ring(radius = radius + dR*i, width = wRing, angle_resolution = 1, layer = 0)) 
    r.move([period*i,0])
    r.ymax = wg.ymin - g
    rwg = R.add_ref(pg.compass(size=[10, 1]))
    rwg.move([period*i, 0])
    rwg.ymax = r.ymin-g
    rarc = R.add_ref(pg.arc(radius = 10, width = 0.5, theta = 180, start_angle = 90, angle_resolution = 2.5, layer = 0))
    rarc.xmax = r.center[0]
    

    
R.x = D.x
D.add_ref(R)

quickplot(D)