from __future__ import division, print_function, absolute_import
from phidl import Device, quickplot, Layer
import phidl.geometry as pg
import numpy as np

#    
#def meander_taper(x_taper, w_taper, meander_length = 1000, spacing_factor = 3, min_spacing = 0.5):
##        
#    def taper_width(x):
#        return np.interp(x, x_taper, w_taper)
#        
#        
#    def taper_section(x_start, x_end, num_pts = 30):
#        D = Device()
#        length =  x_end - x_start
#        x = np.linspace(0, length, num_pts)
#        widths = np.linspace(taper_width(x_start), taper_width(x_end), num_pts)
#        xpts = np.concatenate([x, x[::-1]])
#        ypts = np.concatenate([widths/2, -widths[::-1]/2])
#        D.add_polygon((xpts,ypts), layer = 0, datatype = 0)
#        D.add_port(name = 1, midpoint = (0,0), width = widths[0], orientation = 180)
#        D.add_port(name = 2, midpoint = (length,0), width = widths[-1], orientation = 0)
#        return D
#        
#    def arc_tapered(radius = 10, width1 = 1, width2 = 2, theta = 45, angle_resolution = 2.5, layer = 0):
#        D = Device()
#        path1 = gdspy.Path(width = width1, initial_point = (0, 0))
#        path1.turn(radius = radius, angle = theta*np.pi/180, number_of_points=int(abs(2*theta/angle_resolution)), final_width = width2)
#        [D.add_polygon(p, layer = layer) for p in path1.polygons]
#        D.add_port(name = 1, midpoint = (0, 0), width = width1, orientation = 180)
#        D.add_port(name = 2, midpoint = (path1.x, path1.y), width = width2, orientation = path1.direction*180/np.pi)
#        return D
#    
#        
#    D = Device('meander-taper')
#    xpos1 = min(x_taper)
#    xpos2 = min(x_taper) + meander_length
#    t = D.add_ref( taper_section(x_start = xpos1, x_end = xpos2, num_pts = 30) )
#    D.add_port(t.ports[1])
#    dir_toggle = -1
#    while xpos2 < max(x_taper):
#        arc_width1 = taper_width(xpos2)
#        arc_radius = max(spacing_factor*arc_width1, min_spacing)
#        arc_length = np.pi*arc_radius
#        arc_width2 = taper_width(xpos2 + arc_length)
#        a = D.add_ref(  arc_tapered(radius = arc_radius, width1 = arc_width1, width2 = arc_width2, theta = 180*dir_toggle) )
#        a.connect(port = 1, destination = t.ports[2])
#        dir_toggle = -dir_toggle
#        xpos1 = xpos2 + arc_length
#        xpos2 = xpos1 + meander_length
#        t = D.add_ref( taper_section(x_start = xpos1, x_end = xpos2, num_pts = 30) )
#        t.connect(port = 1, destination = a.ports[2])
#    D.add_port(t.ports[2])
#        
#    return D

    
#==============================================================================
# Example code
#==============================================================================
#H = hecken_taper(length = 10000, B = 4.0091, dielectric_thickness = 0.25, eps_r = 2,
#                  Lk_per_sq = 250e-12, Z1 = 50, width2 = 0.5,
#                  num_pts = 99, layer = 0, datatype = 0)
#
#M = meander_taper(x_taper = H.meta['x'], w_taper = H.meta['w'], meander_length = 1000, spacing_factor = 3, min_spacing = 2)
#


#==============================================================================
# SNSPD Imager with rounded bends
#==============================================================================
def snspd_imager_rounded(
                size = [500, 500],
                nanowire_width = 0.5,
                nanowire_spacing = 5,
                pad_size = [300,500],
                taper_width = 1000,
                taper_length = 11000,
                taper_separation = 200,
                dielectric_thickness = 0.25,
                dielectric_eps_r = 3.8,
                Lk_per_sq = 250e-12,
                Zout = 50,
                label = 'A1',
                L_gndplane = 0,
                L_wsi = 1,
                L_goldpads = 2,
                L_gndetch = 3,
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
    
    Straight = pg.compass(size = [size[0], nanowire_width], layer = L_wsi)
    Turn = pg.arc(radius = nanowire_spacing/2, width = nanowire_width, theta = 180,
                  angle_resolution = 5, layer = L_wsi)
    Taper = pg.hecken_taper(length = taper_length, dielectric_thickness = dielectric_thickness,
                            eps_r = dielectric_eps_r, Lk_per_sq = Lk_per_sq,
                            Z1 = Zout, width2 = nanowire_width, num_pts = 99,
                            layer = L_wsi)
    Meander_taper = pg.meander_taper(x_taper = Taper.meta['x'], w_taper = Taper.meta['w'],
                                 meander_length = taper_width, spacing_factor = 3,
                                 min_spacing = 5, layer = L_wsi)
    mtwidth= Meander_taper.ports[1].width
    
    Separator = pg.taper(length = taper_separation, width1 = nanowire_width, width2 = None, layer = L_wsi)
    Pad_Extension = pg.taper(length = 150, width1 = mtwidth, width2 = None, layer = L_wsi)
    
    Pad = pg.flagpole(size = pad_size[::-1], stub_size = (mtwidth, mtwidth), shape = 'p', taper_type = 'fillet', layer = L_wsi)
#    def taper(length = 10, width1 = 5, width2 = 8, port = None, layer = 0)
    
    S = Device('SNSPI')
    s = S.add_ref(Straight)
    S.add_port(s.ports['W'])
    for n in range(int(num_meanders/2)):
        t = S.add_ref(Turn).connect(1, s.ports['E'])
        s = S.add_ref(Straight).connect('E', t.ports[2])
        t = S.add_ref(Turn).connect(2, s.ports['W'])
        s = S.add_ref(Straight).connect('W', t.ports[1])
    S.add_port(s.ports['E'])
    
    snspi = D.add_ref(S)
    separator_east = D.add_ref(Separator).connect(1, snspi.ports['E'])
    separator_west = D.add_ref(Separator).connect(1, snspi.ports['W'])
    taper_east = D.add_ref(Meander_taper).connect(2, separator_east.ports[2])
    taper_west = D.add_ref(Meander_taper).reflect(p1 = (0,1)).reflect(p1 = (1,0)).connect(2, separator_west.ports[2])
    ext_east = D.add_ref(Pad_Extension).connect(1, taper_east.ports[1])
    ext_west = D.add_ref(Pad_Extension).connect(1, taper_west.ports[1])
    pad_east = D.add_ref(Pad).reflect(p1 = (1,0)).connect(1, ext_east.ports[2])
    pad_west = D.add_ref(Pad).reflect(p1 = (1,0)).connect(1, ext_west.ports[2])
    
    
    #==============================================================================
    # Add gold pads which are slightly inset
    #==============================================================================
    
    D.add_ref( pg.inset([pad_east, ext_east], distance = 2,  layer=L_goldpads) )
    D.add_ref( pg.inset([pad_west, ext_west], distance = 2, layer=L_goldpads) )
    
    #==============================================================================
    # Remove gold underneath pads to avoid bonding issues
    #==============================================================================
    
    D.add_ref( pg.inset([pad_east], distance = -50, layer=L_gndplane) )
    D.add_ref( pg.inset([pad_west], distance = -50, layer=L_gndplane) )
    
    
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
    D.meta['f_cutoff_MHz'] = Taper.meta['f_cutoff']/1e6
    
    
    #==============================================================================
    # Label device
    #==============================================================================
    l1 = D.add_ref( pg.text(label, size = 200, layer = L_goldpads) )
    l2 = D.add_ref( pg.text(label, size = 200, layer = L_goldpads) )
    l1.ymin = pad_west.ymax + 100
    l1.xmin = pad_west.xmin 
    l2.ymax = pad_east.ymin - 100
    l2.xmax = pad_east.xmax
    
    return D

    
    
#==============================================================================
# Device parameters
#==============================================================================

# Material parameters
dielectric_thickness = 0.100
dielectric_eps_r = 3.8
Lk_per_sq = 244e-12

# Die geometry
die_width = 10000
edge_gold_width = 800
pad_width = 300
imager_width = 2000
#taper_length = (die_width - 2*edge_gold_width - 2*pad_width - imager_width - 500)/2
taper_width = 2000
taper_length = taper_width*11

#==============================================================================
# Layer parameters
#==============================================================================


layers = {
        'gndplane'  : Layer(gds_layer = 0, gds_datatype = 0, description = 'Ground plane', color = 'DarkGoldenRod', inverted = True),
        'wsi'  : Layer(gds_layer = 1, gds_datatype = 0, description = 'WSi etch', color = 'lightgreen', inverted = True),
        'goldpads' : Layer(gds_layer = 2, gds_datatype = 0, description = 'Gold pads', color = 'gold'),
        'gndetch' : Layer(gds_layer = 3, gds_datatype = 0, description = 'Etch through to ground plane', color = 'blue'),
         }

#==============================================================================
# Base configurations
#==============================================================================

snspi_config = {
        'size' : [500, 500],
        'nanowire_width' : 0.5,
        'nanowire_spacing' : 5,
        'pad_size' : [300,500],
        'taper_width' : 1000,
        'taper_length' : 21800,
        'taper_separation' : 200,
        'dielectric_thickness' : 0.100,
        'dielectric_eps_r' : 3.8,
        'Lk_per_sq' : 250e-12,
        'Zout' : 50,
        'label' : 'A1',
        'L_gndplane' : layers['gndplane'],
        'L_wsi' : layers['wsi'],
        'L_goldpads' : layers['goldpads'],
        'L_gndetch' : layers['gndetch'],
        }



#==============================================================================
# Die 1: Varying SNSPI length - nanowire width 300 nm
#==============================================================================
D = Device()
die_name = 'SE006'

y = 0
imager_heights = 2.0**np.array(range(0,4))*(5.5)
for n, h in enumerate(imager_heights):
    S = Device(snspd_imager_rounded, config = snspi_config, label = ('A%s' % (n+1)),
               nanowire_width = 0.3, size = [2000, h])
    l = S.annotate(str(S.meta).replace(",", "\n"))
    l.center = S.center
    s = D.add_ref(S)
    s.ymax = y
    y = s.ymin
D.center = [0,0]

# Create die, label, and streets
die = D.add_ref( pg.basic_die(size = (10000, 10000), street_width = 100, street_length = 1000, 
              die_name = die_name, text_size = 250, text_location = 'S',  layer = layers['goldpads'],  
              draw_bbox = False) )


# Create edge gold sidebars
R = pg.rectangle((9300, 9300))
Rinv = pg.invert(R, border = 300, layer = layers['gndetch'])
Rinv.center = D.center
D.add_ref(Rinv)

D.write_gds('%s SNSPD Imager.gds' % die_name)



#==============================================================================
# Die 2: Varying SNSPI length - nanowire width 500 nm
#==============================================================================
D = Device()
die_name = 'SE007'

y = 0
imager_heights = 2.0**np.array(range(0,4))*(5.5)
for n, h in enumerate(imager_heights):
    S = Device(snspd_imager_rounded, config = snspi_config, label = ('A%s' % (n+1)),
               nanowire_width = 0.5, size = [2000, h])
    l = S.annotate(str(S.meta).replace(",", "\n"))
    l.center = S.center
    s = D.add_ref(S)
    s.ymax = y
    y = s.ymin
D.center = [0,0]

# Create die, label, and streets
die = D.add_ref( pg.basic_die(size = (10000, 10000), street_width = 100, street_length = 1000, 
              die_name = die_name, text_size = 250, text_location = 'S',  layer = layers['goldpads'],  
              draw_bbox = False) )


# Create edge gold sidebars
R = pg.rectangle((9300, 9300))
Rinv = pg.invert(R, border = 300, layer = layers['gndetch'])
Rinv.center = D.center
D.add_ref(Rinv)

D.write_gds('%s SNSPD Imager.gds' % die_name)



#==============================================================================
# Die 3: Large device - 300 nm width
#==============================================================================
D = Device()
die_name = 'SE008'

S = Device(snspd_imager_rounded, config = snspi_config, label = ('A1'),
           nanowire_width = 0.3, size = [2500, 2500])

l = S.annotate(str(S.meta).replace(",", "\n"))
l.center = S.center
s = D.add_ref(S)
D.center = [0,0]

# Create die, label, and streets
die = D.add_ref( pg.basic_die(size = (10000, 10000), street_width = 100, street_length = 1000, 
              die_name = die_name, text_size = 250, text_location = 'S',  layer = layers['goldpads'],  
              draw_bbox = False) )


# Create edge gold sidebars
R = pg.rectangle((9300, 9300))
Rinv = pg.invert(R, border = 300, layer = layers['gndetch'])
Rinv.center = D.center
D.add_ref(Rinv)

D.write_gds('%s SNSPD Imager.gds' % die_name)

#==============================================================================
# Die 4: Large device - 500 nm width
#==============================================================================
D = Device()
die_name = 'SE009'

S = Device(snspd_imager_rounded, config = snspi_config, label = ('A1'),
           nanowire_width = 0.5, size = [2500, 2500])

l = S.annotate(str(S.meta).replace(",", "\n"))
l.center = S.center
s = D.add_ref(S)
D.center = [0,0]

# Create die, label, and streets
die = D.add_ref( pg.basic_die(size = (10000, 10000), street_width = 100, street_length = 1000, 
              die_name = die_name, text_size = 250, text_location = 'S',  layer = layers['goldpads'],  
              draw_bbox = False) )


# Create edge gold sidebars
R = pg.rectangle((9300, 9300))
Rinv = pg.invert(R, border = 300, layer = layers['gndetch'])
Rinv.center = D.center
D.add_ref(Rinv)

D.write_gds('%s SNSPD Imager.gds' % die_name)

