from __future__ import division, print_function, absolute_import
from phidl import Device, quickplot, inset
import phidl.geometry as pg
import numpy as np


    
#==============================================================================
# SNSPD Imager with rounded bends
#==============================================================================
size = [5000, 5000]
nanowire_width = 0.5
nanowire_spacing = 5
pad_size = [200,200]
taper_length = 2500
dielectric_thickness = 0.25
dielectric_eps_r = 3.8
Lk_per_sq = 250e-12
Zout = 50
label = 'A1'

# Create blank device
D = Device('SNSPD-imager-rounded')

#==============================================================================
# Setup variables
#==============================================================================
num_meanders = np.ceil( size[1]/(nanowire_width + nanowire_spacing) )

#==============================================================================
# Create and place components
#==============================================================================

Straight = pg.compass(size = [size[0], nanowire_width], layer = 0, datatype = 0)
Turn = pg.arc(radius = nanowire_spacing/2, width = nanowire_width, theta = 180, angle_resolution = 5, layer = 0, datatype = 0)
Taper = pg.hecken_taper(length = taper_length, dielectric_thickness = dielectric_thickness, eps_r = dielectric_eps_r,
                 Lk_per_sq = Lk_per_sq, Z1 = Zout, width2 = nanowire_width,
                 num_pts = 99, layer = 0, datatype = 0)
Pad = pg.compass(size = pad_size, layer = 0, datatype = 0)


s = D.add_device(Straight)
taper_west = D.add_device(Taper).connect(2, s.ports['W'])
for n in range(int(num_meanders/2)):
    t = D.add_device(Turn).connect(1, s.ports['E'])
    s = D.add_device(Straight).connect('E', t.ports[2])
    t = D.add_device(Turn).connect(2, s.ports['W'])
    s = D.add_device(Straight).connect('W', t.ports[1])
taper_east = D.add_device(Taper).connect(2, s.ports['E'])
pad_east = D.add_device(Pad).connect('E', taper_east.ports[1])
pad_west = D.add_device(Pad).connect('W', taper_west.ports[1])


#==============================================================================
# Add gold pads which are slightly inset
#==============================================================================

D.add_polygon( inset(pad_east, distance = 2, layer=1, datatype=0) )
D.add_polygon( inset(pad_west, distance = 2, layer=1, datatype=0) )


#==============================================================================
# Include meta information about device
#==============================================================================

D.meta['length'] = (Turn.meta['length'] + size[0])*num_meanders*2
D.meta['num_squares'] = D.meta['length']/nanowire_width
D.meta['Lk_total'] = D.meta['num_squares']*Lk_per_sq
D.meta['Z_nanowire'] = pg._microstrip_Z_with_Lk(nanowire_width*1e-6, dielectric_thickness*1e-6, dielectric_eps_r, Lk_per_sq)
D.meta['v/c'] = pg._microstrip_v_with_Lk(nanowire_width*1e-6, dielectric_thickness*1e-6, dielectric_eps_r, Lk_per_sq)/3e8
D.meta['delay'] = D.meta['length']*1e-6/(D.meta['v/c']*3e8)
# Calculate reduction of Ic due to non-optimal rounded bends (Clem 2011, Eqn (18))
a = nanowire_spacing/2
b = a + nanowire_width
D.meta['Ic_reduction'] = 1 - a*np.log(b/a)/(b-a)


#==============================================================================
# Label device
#==============================================================================
l1 = D.add_device( pg.text(label, size = 100, layer = 1) )
l2 = D.add_device( pg.text(label, size = 100, layer = 1) )
l1.ymin = pad_west.ymax + 100
l1.xmin = pad_west.xmin 
l2.ymax = pad_east.ymin - 100
l2.xmax = pad_east.xmax

D.write_gds('IMG001 SNSPD Imager.gds')
#quickplot(D)

