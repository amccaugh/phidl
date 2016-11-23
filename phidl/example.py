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

interaction_length = 10
hSines = 5
rMin = 10
lSines = np.sqrt(hSines*np.pi**2*rMin/2)
gap = 0.5
wWg= 0.4

D = Device()

WG = pg.compass(size=[interaction_length, wWg])
wg1 = D.add_ref(WG)
wg2 = D.add_ref(WG)

wg1.center = wg2.center
wg1.ymin = wg1.ymax + gap

P = Device()
port1 = P.add_port(name = 1, midpoint = wg1.ports['W'].midpoint+[-1*lSines, hSines], width = wWg, orientation = 0)
port2 = P.add_port(name = 2, midpoint = wg2.ports['W'].midpoint+[-1*lSines, 0], width = wWg, orientation = 0)
port3 = P.add_port(name = 3, midpoint = wg1.ports['E'].midpoint+[lSines,hSines], width = wWg, orientation = 180)
port4 = P.add_port(name = 4, midpoint = wg2.ports['E'].midpoint+[lSines, 0], width = wWg, orientation = 180)

route1 = D.add_ref(pg.route(port1 = wg1.ports['W'], port2 = port1))
route2 = D.add_ref(pg.route(port1 = wg2.ports['W'], port2 = port2))
route3 = D.add_ref(pg.route(port1 = wg1.ports['E'], port2 = port3))
route4 = D.add_ref(pg.route(port1 = wg2.ports['E'], port2 = port4))
D.add_ref(P)

D.add_port(port = route1.ports[2], name = 1)
D.add_port(port = route2.ports[2], name = 2)
D.add_port(port = route3.ports[2], name = 3)
D.add_port(port = route4.ports[2], name = 4)
quickplot(D)