from __future__ import division # Makes it so 1/4 = 0.25 instead of zero


from phidl import Device, quickplot
import numpy as np
import phidl.geometry as pg

# parameters for rings
radius = 10
gaps = [0.1, 0.2, 0.3]
wRing = 0.5
wWg = 0.4
dR = 0.015
period = radius*3

nrings = len(gaps)
lWg = (nrings + 1)*period

D = Device()
wg = D.add_device(pg.compass(size=[lWg, wWg]))
R = Device();

for i, g in enumerate(gaps):
    r = R.add_device(pg.ring(radius = radius + dR*i, width = wRing, angle_resolution = 1, layer = 0, datatype = 0)) 
    r.move([period*i,0])
    r.ymax = wg.ymin - g

R.x = D.x
D.add_device(R)

# parameters for the gratings
nperiods = 20
period = 0.75
ff = 0.5
wGrating = 20
lTaper = 20

G = Device()

for i in range(nperiods):
    cgrating = G.add_device(pg.compass(size=[period*ff,wGrating]))
    cgrating.x+=i*period

p = G.add_port(port = cgrating.ports['E'], name = 1)
p.midpoint=p.midpoint+np.array([(1-ff)*period,0])

g = D.add_device(G)
g2 = D.add_device(G)
g.connect(port = 1, destination = wg.ports['W'])
g.move([-1*lTaper, 0])
g2.connect(port = 1, destination = wg.ports['E'])
g2.move([lTaper, 0])


D.route(port1 = g.ports[1], port2 = wg.ports['W'])
D.route(port1 = g2.ports[1], port2 = wg.ports['E'])

quickplot(D)