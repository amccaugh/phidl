# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 16:33:13 2016

@author: anm16
"""
# %% Connecting together several waveguides

def waveguide(width = 10, height = 1, name = 'waveguide'):
    wg = Device(name)
    wg.add_polygon([(0, 0), (width, 0), (width, height), (0, height)])
    wg.add_port(name = 'wgport1', midpoint = [0,height/2], width = height, orientation = 180)
    wg.add_port(name = 'wgport2', midpoint = [width,height/2], width = height, orientation = 0)
    return wg



d = Device('MultiWaveguide')
wg1 = d.add_device(waveguide(width=10, height = 1))
wg2 = d.add_device(waveguide(width=12, height = 2))
wg3 = d.add_device(waveguide(width=14, height = 3))

quickplot(d)

poly1 = d.add_polygon([[8,6,7,9],[6,8,9,5]]) # Add as xpts list, ypts list
poly2 = d.add_polygon([(0, 0), (1, 1), (1, 3), (-3, 3)]) # Add as list of points
poly2.translate(5,4)

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

quickplot(d, new_window = True); plt.plot([2,10],[-8,15])

d.copy_port(name = 1, port = wg1.ports['wgport1'])
d.copy_port(name = 2, port = wg3.ports['wgport2'])

quickplot(d)

dsquared = Device('MultiMultiWaveguide')
mwg1 = dsquared.add_device(d)
mwg2 = dsquared.add_device(d)
mwg2.move(destination = [4,4])

quickplot(dsquared)

mwg1.connect(port = 1, destination = mwg2.ports[2], translate = True, rotate = True)

quickplot(dsquared)

mwg2.translate(dx = 40, dy = 25)

quickplot(dsquared)

dsquared.route(port_a = mwg1.ports[1], port_b = mwg2.ports[2], path_type = 'sine', width_type = 'straight')

quickplot(dsquared)

dsquared.write_gds('MultiMultiWaveguideTutorial.gds')

# %% Testing routing functions

d = Device()

cp1 = d.add_device(compass(dimensions = [2,2])).move(destination = [-10,-50])
cp2 = d.add_device(compass(dimensions = [2,2])).move(destination = [-5,-50])
cp3 = d.add_device(compass(dimensions = [2,2])).move(destination = [0,-50])
cp4 = d.add_device(compass(dimensions = [2,2])).move(destination = [5,-50])
cp5 = d.add_device(compass(dimensions = [2,2])).move(destination = [10,-50])


cpm = d.add_device(compass_multi(dimensions = [40,20], ports = {'S':5, 'N':1})).move(destination = [0,15])


quickplot(d)


r1 = d.route(port_a = cpm.ports['S1'], port_b = cp1.ports['N'], path_type = 'sine', width_type = 'straight')
r2 = d.route(port_a = cpm.ports['S2'], port_b = cp2.ports['N'], path_type = 'sine', width_type = 'straight')
r3 = d.route(port_a = cpm.ports['S3'], port_b = cp3.ports['N'], path_type = 'sine', width_type = 'straight')
r4 = d.route(port_a = cpm.ports['S4'], port_b = cp4.ports['N'], path_type = 'sine', width_type = 'straight')
r5 = d.route(port_a = cpm.ports['S5'], port_b = cp5.ports['N'], path_type = 'sine', width_type = 'straight')

quickplot(d)
