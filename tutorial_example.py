# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 16:33:13 2016

@author: anm16
"""

# %% Connecting together several waveguides

d = Device('MultiWaveguide')
wg1 = d.add_device(waveguide(width=10, height = 1))
wg2 = d.add_device(waveguide(width=12, height = 2))
wg3 = d.add_device(waveguide(width=14, height = 3))

quickplot(d)

wg2.move(origin = 'wgport1', destination = wg1.ports['wgport2'])
wg3.move(origin = 'wgport1', destination = wg2.ports['wgport2'])

quickplot(d)

wg1.rotate(angle = 45, center = wg1.ports['wgport2'].midpoint)
wg3.rotate(angle = 45, center = wg3.ports['wgport1'].midpoint)

quickplot(d)

wg3.reflect(p1 = wg3.ports['wgport1'].midpoint, p2 = wg3.ports['wgport1'].midpoint + np.array([1,0]))

quickplot(d); plt.plot([2,10],[-8,15])

[sd.reflect(p1 = [2,-8], p2 = [10,15]) for sd in [wg1, wg2, wg3]]

quickplot(d); plt.plot([2,10],[-8,15])

d.copy_port(name = 1, port = wg1.ports['wgport1'])
d.copy_port(name = 2, port = wg3.ports['wgport2'])

quickplot(d)

dsquared = Device('MultiMultiWaveguide')
mwg1 = dsquared.add_device(d)
mwg2 = dsquared.add_device(d)
mwg2.move(origin = 1, destination = mwg1.ports[2])

quickplot(dsquared)

mwg1.connect(port = 1, destination = mwg2.ports[2], translate = True, rotate = True)

quickplot(dsquared)