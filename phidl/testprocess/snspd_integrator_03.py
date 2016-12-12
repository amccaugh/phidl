from __future__ import division, print_function, absolute_import
from phidl import Device, Layer, quickplot
import phidl.geometry as pg
import numpy as np


    
#==============================================================================
# SNSPD yTron-based integrator
#==============================================================================
def snspd_integrator(
                    nanowire_width = 1.0,
                    fill_factor = 1/3,
                    contact_pad_width = 5,
                    snspd_size = [250,250],
                    pad_size = [250,250],
                    pad_pole_size = [50, 50],
                    pad_spacing = 50,
                    inset_distance = 0.5,
                    num_devices = 2,
                    label = 'A1',
                    rho_intersection = 0.3,
                    theta_intersection = 2.5,
                    arm_widths = (12,2),
                    pad_layer = 1,
                    gndpad_layer = 1,
                    sc_layer = 0,
                    ):

    # Create blank device
    D = Device('SNSPD-yTron-integator')

    source_length = (arm_widths[0] + arm_widths[1])
    arm_lengths = np.array([12,2])*2

    #==============================================================================
    # Create and place components
    #==============================================================================
    T = pg.tee(size = pad_size, stub_size = pad_pole_size, taper_type = 'fillet', layer = sc_layer)
    S = pg.snspd_expanded(wire_width = nanowire_width, wire_pitch = nanowire_width*3, size = snspd_size, connector_width = contact_pad_width,
                                             terminals_same_side = False, layer = sc_layer).rotate(90)
    
    cpm = D.add_ref(pg.compass_multi(size = [100,25], ports = {'N':num_devices,'S':1}, layer = sc_layer))
    cpm.center = [0,-200]
    f = D.add_ref(pg.flagpole(size = [100,25], stub_size = [arm_widths[1],arm_widths[1]], shape = 'p', taper_type = 'fillet', layer = sc_layer))
    fy = D.add_ref(pg.flagpole(size = pad_size, stub_size = [arm_widths[0],arm_widths[0]], shape = 'q', taper_type = 'fillet', layer = sc_layer))
    
    y =  D.add_ref(pg.ytron_round(rho_intersection, theta_intersection, arm_lengths, source_length,
                      arm_widths, theta_resolution = 10, layer = sc_layer))
    pad_array = D.add_array(T, start = [0, 1000], spacing = [T.xsize+pad_spacing, 0], num_devices = num_devices)
    snspd_array = D.add_array(S, start = [0,500], spacing = [S.xsize+20, 0], num_devices = num_devices)
               
    f.connect(port = 2, destination = cpm.ports['S1'])
    y.connect(port = 'right', destination = f.ports[1])
    fy.connect(port = 1, destination = y.ports['left']).move([-200*np.sin(theta_intersection/180*np.pi), 200])
    
    gnd = D.add_ref(pg.compass(size = pad_size, layer = sc_layer))
    gnd.move(origin = gnd.center, destination = y.ports['source'])
    gnd.move([0,-500])
    
    
    #==============================================================================
    # Route components
    #==============================================================================
    pad_routes = []
    connector_routes = []
    for n in range(num_devices):
        s = snspd_array[n]; p = pad_array[n]
        rc = D.add_ref( pg.route(port1 = s.ports[1], port2 = cpm.ports['N%s' % (n+1)], path_type = 'sine', width_type = 'sine', width1 = None, width2 = contact_pad_width*4, num_path_pts = 99, layer = sc_layer) )
        rp = D.add_ref( pg.route(port1 = p.ports['S'], port2 = s.ports[2], path_type = 'sine', width_type = 'sine', width1 = None, width2 = None, num_path_pts = 99, layer = sc_layer) )
        pad_routes.append(rp)
        connector_routes.append(rc)
    r_ytron_gnd = D.add_ref( pg.route(y.ports['source'], gnd.ports['N'], path_type = 'sine', width_type = 'sine') )
    r_ytron_pad = D.add_ref( pg.route(y.ports['left'], fy.ports[1], path_type = 'straight', width_type = 'sine') )
    
    
    #==============================================================================
    # Add gold pads which are slightly pg.inset
    #==============================================================================
    D.add_ref( pg.inset([r_ytron_gnd, gnd], distance = inset_distance, join_first = True, layer=pad_layer) )
    D.add_ref( pg.inset(pad_array + pad_routes, distance = inset_distance, join_first = True, layer=pad_layer) )
    D.add_ref( pg.inset([f,cpm] + connector_routes, distance = inset_distance, join_first = True, layer=pad_layer) )
    D.add_ref( pg.inset([fy, r_ytron_pad], distance = inset_distance, join_first = True, layer=pad_layer) )
    
    
    #==============================================================================
    # Short everything to ground
    #==============================================================================
    
    gndshort = D.add_ref(pg.compass(size = pad_size, layer = gndpad_layer))
    gndshort.connect(port = 'N', destination = gnd.ports['S'])
    gndshort.y += inset_distance
    
    
    #==============================================================================
    # Label device
    #==============================================================================
    D.add_ref( pg.text(label, justify = 'right', size = 200, layer = pad_layer) ).center = [-500,-500]



    return D

    

#==============================================================================
# To change in this version
#==============================================================================

die_name = 'SE010'

layers = {
        'goldpads' : Layer(gds_layer = 1, gds_datatype = 0, description = 'Gold pads', color = 'gold'),
        'goldgnd'  : Layer(gds_layer = 1, gds_datatype = 1, description = 'Gold ground', color = 'DarkGoldenRod'),
        'wsi'  : Layer(gds_layer = 0, gds_datatype = 0, description = 'WSi', color = 'lightgreen'),
         }

D = Device()

#==============================================================================
# Write default settings
#==============================================================================

default_config = {
        'nanowire_width' : 1.0,
        'fill_factor' : 1/3,
        'contact_pad_width' : 5,
        'snspd_size' : [200,200],
        'pad_size' : [250,250],
        'pad_pole_size' : [50, 50],
        'pad_spacing' : 50,
        'inset_distance' : 0.5,
        'num_devices' : 3,
        'label' : 'Z9',
        'rho_intersection' : 0.3,
        'theta_intersection' : 2.5,
        'arm_widths' : (12,2),
        'pad_layer' : layers['goldpads'],
        'gndpad_layer' : layers['goldgnd'],
        'sc_layer' : layers['wsi'],
        }

#==============================================================================
# Row A: Varying right arm width
#==============================================================================
x,y = (0,0)
parameters = [1.5,2,2,2.5,3]
for n, p in enumerate(parameters):
    S = Device(snspd_integrator, config = default_config, label = 'A'+str(n+1),
               arm_widths = [12,p])
    s = D.add_ref( S )
    s.xmin = x
    s.ymax = y
    x = s.xmax + 300
    D.annotate(('yTron right arm width = %s um' % p), s.center)
    

#==============================================================================
# Row B: Varying SNSPD width
#==============================================================================
x,y = 0, D.ymin-400
parameters = [1,1.25,1.5,1.75,2]
for n, p in enumerate(parameters):
    S = Device(snspd_integrator, config = default_config, label = 'B'+str(n+1),
               nanowire_width = p, snspd_size = np.array([200,200])*p, num_devices = 2)
    s = D.add_ref( S )
    s.xmin = x
    s.ymax = y
    x = s.xmax + 300
    D.annotate(('SNSPD nanowire width = %s um' % p), s.center)
    
#==============================================================================
# Row C: Varying left arm width
#==============================================================================
x,y = 0, D.ymin-400
parameters = [9,12,15,18,21]
for n, p in enumerate(parameters):
    S = Device(snspd_integrator, config = default_config, label = 'C'+str(n+1),
               arm_widths = [p,2])
    s = D.add_ref( S )
    s.move([x, y])
    s.xmin = x
    s.ymax = y
    x = s.xmax + 300
    D.annotate(('yTron left arm width = %s um' % p), s.center)
    

    
D.center = (0,0)
die = D.add_ref( pg.basic_die(size = (10000, 10000), street_width = 100, street_length = 1000, 
              die_name = die_name, text_size = 300, text_location = 'SW', layer = layers['goldpads'],  
              draw_bbox = False,  bbox_layer = 99) )
              
D.add_ref( pg.fill_rectangle(D, fill_size = [50,50], exclude_layers = [layers['wsi'], layers['goldpads']],
                                 margin = 100, fill_layers = [0,1], fill_densities = [0.2, 0.2], 
                                 fill_inverted = [False, True], bbox = None) )

D.write_gds('%s SNSPD Integrator.gds' % die_name, precision = 1e-9)

#quickplot(D)
