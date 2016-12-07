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
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d



r = np.linspace(0.01,2,50)
n = len(r)
length = np.empty(n)
F = np.empty(n)
for i in range(0, n):
    lSines = 100
    hSines = lSines*r[i]
    D = Device()
    BS = Device()
    B = BS.add_ref(bP.adiabaticBeamsplitter(interaction_length = 100, hSines = 20))
    bS1 = D.add_ref(B)
    bS2 = D.add_ref(B)
    bS1.reflect(p1 = bS1.ports['port 3'], p2 = bS1.ports['port 4'])
    P = Device()
    P.add_port(name = 1, midpoint = bS1.ports['port 3'].midpoint+[lSines/2,hSines], width = port3, orientation = 180)
    P.add_port(name = 2, midpoint = bS1.ports['port 3'].midpoint+[lSines/2,hSines], width = port3, orientation = 0)
    D.add_ref(P)
    bS1.movex(lSines)
    route1 = D.add_ref(pg.route(port1 = P.ports[1], port2 = bS2.ports['port 3'], path_type = 'sine'))
    route2 = D.add_ref(pg.route(port1 = P.ports[2], port2 = bS1.ports['port 3'], path_type = 'sine'))
    route3 = D.add_ref(pg.route(port1 = bS1.ports['port 4'], port2 = bS2.ports['port 4']))
    length[i] = route1.meta['length']*2
    F[i] = 2*np.pi*length[i]/lSines
    
#quickplot(D)
MZI_data = np.empty([n, 2])
MZI_data[:,0] = r
MZI_data[:,1] = F
np.save('MZI_factors', MZI_data)
f = interp1d(r,F,kind='cubic')

xnew = np.linspace(0.01, 2, 500)
plt.plot(r, F, 'o', xnew, f(xnew))
plt.ylabel('Adjustment factor')
plt.xlabel('hSines/lSines')
