from __future__ import division, print_function, absolute_import
from phidl import Device, quickplot, inset
import phidl.geometry as pg
import numpy as np

    
#==============================================================================
# SNSPD Imager with rounded bends
#==============================================================================
def snspd_imager_rounded(
                size = [500, 500],
                nanowire_width = 0.5,
                nanowire_spacing = 5,
                pad_size = [300,500],
                taper_length = 2500,
                dielectric_thickness = 0.25,
                dielectric_eps_r = 3.8,
                Lk_per_sq = 250e-12,
                Zout = 50,
                label = 'A1',
                ):

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
    taper_width = Taper.ports[1].width
    Pad = pg.flagpole(size = pad_size[::-1], stub_size = (taper_width, taper_width), shape = 'p', taper_type = 'fillet', layer = 0, datatype = 0)
    
    
    s = D.add_device(Straight)
    taper_west = D.add_device(Taper).connect(2, s.ports['W'])
    for n in range(int(num_meanders/2)):
        t = D.add_device(Turn).connect(1, s.ports['E'])
        s = D.add_device(Straight).connect('E', t.ports[2])
        t = D.add_device(Turn).connect(2, s.ports['W'])
        s = D.add_device(Straight).connect('W', t.ports[1])
    taper_east = D.add_device(Taper).connect(2, s.ports['E'])
    ext_east = D.add_device(pg.taper(length = taper_width, port = taper_east.ports[1], layer = 0, datatype = 0))
    ext_west = D.add_device(pg.taper(length = taper_width, port = taper_west.ports[1], layer = 0, datatype = 0))
    pad_east = D.add_device(Pad).connect(1, ext_east.ports[2])
    pad_west = D.add_device(Pad).connect(1, ext_west.ports[2])
    
    
    #==============================================================================
    # Add gold pads which are slightly inset
    #==============================================================================
    
    D.add_polygon( inset([pad_east, ext_east], distance = 2, layer=1, datatype=0) )
    D.add_polygon( inset([pad_west, ext_west], distance = 2, layer=1, datatype=0) )
    
    #==============================================================================
    # Remove gold underneath pads to avoid bonding issues
    #==============================================================================
    
    D.add_polygon( inset([pad_east], distance = -5, layer=3, datatype=0) )
    D.add_polygon( inset([pad_west], distance = -5, layer=3, datatype=0) )
    
    
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
    D.meta['f_cutoff1_MHz'] = Taper.meta['f_cutoff1']/1e6
    D.meta['f_cutoff2_MHz'] = Taper.meta['f_cutoff2']/1e6
    
    
    #==============================================================================
    # Label device
    #==============================================================================
    l1 = D.add_device( pg.text(label, size = 200, layer = 1) )
    l2 = D.add_device( pg.text(label, size = 200, layer = 1) )
    l1.ymin = pad_west.ymax + 100
    l1.xmin = pad_west.xmin 
    l2.ymax = pad_east.ymin - 100
    l2.xmax = pad_east.xmax
    
    return D

    
    
#==============================================================================
# Device parameters
#==============================================================================

# Material parameters
dielectric_thickness = 0.94
dielectric_eps_r = 3.8
Lk_per_sq = 244e-12

# Die geometry
die_width = 10000
edge_gold_width = 800
pad_width = 300
imager_width = 2000
taper_length = (die_width - 2*edge_gold_width - 2*pad_width - imager_width - 500)/2


#==============================================================================
# Die 1: Varying meander size
#==============================================================================
D = Device()
die_name = 'SE004'

y = 0
imager_heights = 2.0**np.array(range(0,6))*(5.5)
for n, h in enumerate(imager_heights):
    S = snspd_imager_rounded(label = ('A%s' % (n+1)), size = [imager_width, h], nanowire_width = 0.5,
                             nanowire_spacing = 5, taper_length = taper_length)
    l = S.add_label(str(S.meta).replace(",", "\n"))
    l.center = S.center
    s = D.add_device(S)
    s.ymax = y
    y = s.ymin
D.center = [0,0]

# Create die, label, and streets
die = D.add_device( pg.basic_die(size = (10000, 10000), street_width = 100, street_length = 1000, 
              die_name = die_name, text_size = 250, text_location = 'S',  layer = 0,  
              datatype = 0, draw_bbox = False) )


# Create edge gold sidebars
R = Device()
R.add_polygon( pg.rectangle([edge_gold_width, 10000-200], layer = 2))
r1, r2 = D.add_device(R), D.add_device(R)
r1.xmin = die.xmin + 100
r1.ymin = die.ymin + 100
r2.xmax = die.xmax - 100
r2.ymin = die.ymin + 100

D.write_gds('%s SNSPD Imager.gds' % die_name)


#==============================================================================
# Die 2: Varying meander size
#==============================================================================
D = Device()
die_name = 'SE005'

y = 0
imager_heights = 2.0**np.array(range(0,6))*(5.5)
for n, h in enumerate(imager_heights):
    S = snspd_imager_rounded(label = ('A%s' % (n+1)), size = [imager_width, h], nanowire_width = 0.5,
                             nanowire_spacing = 5, taper_length = taper_length)
    l = S.add_label(str(S.meta).replace(",", "\n"))
    l.center = S.center
    s = D.add_device(S)
    s.ymax = y
    y = s.ymin
D.center = [0,0]

# Create die, label, and streets
die = D.add_device( pg.basic_die(size = (10000, 10000), street_width = 100, street_length = 1000, 
              die_name = die_name, text_size = 250, text_location = 'S',  layer = 0,  
              datatype = 0, draw_bbox = False) )


# Create edge gold sidebars
R = Device()
R.add_polygon( pg.rectangle([edge_gold_width, 10000-200], layer = 2))
r1, r2 = D.add_device(R), D.add_device(R)
r1.xmin = die.xmin + 100
r1.ymin = die.ymin + 100
r2.xmax = die.xmax - 100
r2.ymin = die.ymin + 100

D.write_gds('%s SNSPD Imager.gds' % die_name)

