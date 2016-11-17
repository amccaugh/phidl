from __future__ import division, print_function, absolute_import
from phidl import Device, quickplot, inset
import phidl.geometry as pg
import numpy as np


    
#==============================================================================
# SNSPD yTron-based integrator
#==============================================================================
def snspd_integrator(
                    nanowire_width = 0.5,
                    fill_factor = 1/3,
                    contact_pad_width = 5,
                    snspd_size = [20,20],
                    pad_flag_size = [250,250],
                    pad_pole_size = [50, 50],
                    pad_spacing = 50,
                    connector_size = [400,150],
                    inset_distance = 1,
                    num_devices = 5,
                    label = 'A1',
                    rho_intersection = 1,
                    theta_intersection = 5,
                    arm_length = 50,
                    source_length = 50,
                    width_right = 20,
                    width_left = 20,
                    ):

    # Create blank device
    D = Device('SNSPD-yTron-integator')


    #==============================================================================
    # Create and place components
    #==============================================================================
    cpm = D.add_device(pg.compass_multi, size = connector_size, center = [0,-200], ports = {'N':num_devices,'S':1}, layer = 0, datatype = 0)
    f = D.add_device(pg.flagpole(size = connector_size, stub_size = [width_right,width_right], shape = 'p', taper_type = 'fillet', layer = 0, datatype = 0))
    fy = D.add_device(pg.flagpole(size = [connector_size[0],connector_size[1]*2], stub_size = [width_left,width_left], shape = 'q', taper_type = 'fillet', layer = 0, datatype = 0))
    
    t = pg.tee(size = pad_flag_size, stub_size = pad_pole_size, taper_type = 'fillet', layer = 0, datatype = 0)
    pad_array = D.add_array(t, start = [-(t.xsize+pad_spacing)*num_devices/2, 1000], spacing = [t.xsize+pad_spacing, 0], num_devices = num_devices)
    s = pg.snspd_expanded(wire_width = nanowire_width, wire_pitch = nanowire_width*3, size = snspd_size, connector_width = contact_pad_width,
                                             terminals_same_side = False, layer = 0, datatype = 0).rotate(90)
    snspd_array = D.add_array(s, start = [-100,300], spacing = [s.xsize+20, 0], num_devices = num_devices)
               
    y =  D.add_device(pg.ytron_round(rho_intersection, theta_intersection, arm_length, source_length,
                      width_right, width_left, theta_resolution = 10, layer = 0, datatype = 0))
    f.connect(port = 2, destination = cpm.ports['S1'])
    y.connect(port = 'right', destination = f.ports[1])
    fy.connect(port = 1, destination = y.ports['left']).move([-200*np.sin(5/180*np.pi), 200])
    
    gnd = D.add_device(pg.compass(size = pad_flag_size, layer = 0, datatype = 0))
    gnd.move(origin = gnd.center, destination = y.ports['source'])
    gnd.move([0,-500])
    
    
    #==============================================================================
    # Route components
    #==============================================================================
    pad_routes = []
    connector_routes = []
    for n in range(num_devices):
        s = snspd_array[n]; p = pad_array[n]
        rc = D.route(port1 = s.ports[1], port2 = cpm.ports['N%s' % (n+1)], path_type = 'sine', width_type = 'sine', width1 = None, width2 = contact_pad_width*4, num_path_pts = 99, layer = 0, datatype = 0)
        rp = D.route(port1 = p.ports['S'], port2 = s.ports[2], path_type = 'sine', width_type = 'sine', width1 = None, width2 = None, num_path_pts = 99, layer = 0, datatype = 0)
        pad_routes.append(rp)
        connector_routes.append(rc)
    r_ytron_gnd = D.route(y.ports['source'], gnd.ports['N'], path_type = 'sine', width_type = 'sine')
    r_ytron_pad = D.route(y.ports['left'], fy.ports[1], path_type = 'straight', width_type = 'sine')
    
    
    #==============================================================================
    # Add gold pads which are slightly inset
    #==============================================================================
    D.add_polygon( inset([r_ytron_gnd, gnd], distance = inset_distance, join_first = True, layer=1, datatype=0) )
    D.add_polygon( inset([r_ytron_gnd, gnd], distance = inset_distance, join_first = True, layer=1, datatype=0) )
    D.add_polygon( inset(pad_array + pad_routes, distance = inset_distance, join_first = True, layer=1, datatype=0) )
    D.add_polygon( inset([f,cpm] + connector_routes, distance = inset_distance, join_first = True, layer=1, datatype=0) )
    D.add_polygon( inset([fy, r_ytron_pad], distance = inset_distance, join_first = True, layer=1, datatype=0) )
    
    
    #==============================================================================
    # Label device
    #==============================================================================
    D.add_device( pg.text(label, justify = 'right', size = 200, layer = 1) ).center = [-500,-500]

    return D




#==============================================================================
# Row A: Varying sharpness of yTron intersection (rho_intersection)
#==============================================================================
d = Device()
rho = [0.25,0.5,1,1.5,2]
for n, r in enumerate(rho):
    s = d.add_device( snspd_integrator(label = 'A'+str(n+1), width_right = 20, width_left = 20, rho_intersection = r, num_devices = 5) )
    s.move([(s.xsize + 300)*n, 0])
    d.add_label(('Varying yTron rho\n rho = %s\n20um arms' % rho[n]), s.center)
    
    
#==============================================================================
# Row B: Varying sharpness of yTron intersection (rho_intersection) (different arm width)
#==============================================================================
rho = [0.25,0.5,1,1.5,2]
for n, r in enumerate(rho):
    s = d.add_device( snspd_integrator(label = 'B'+str(n+1), width_right = 10, width_left = 20, rho_intersection = rho[n], num_devices = 5) )
    s.move([(s.xsize + 300)*n, -(s.ysize+200)])
    d.add_label(('Varying yTron rho\n rho = %s\n20 & 10um arms' % rho[n]), s.center)
    
    
#==============================================================================
# Row C: Varying sharpness of yTron intersection (rho_intersection) (different arm width)
#==============================================================================
rho = [0.25,0.5,1,1.5,2]
for n, r in enumerate(rho):
    s = d.add_device( snspd_integrator(label = 'C'+str(n+1), width_right = 5, width_left = 20, rho_intersection = rho[n], num_devices = 5) )
    s.move([(s.xsize + 300)*n, -2*(s.ysize+200)])
    d.add_label(('Varying yTron rho\n rho = %s\n20 & 5um arms' % rho[n]), s.center)

d.center = (0,0)
die = d.add_device( pg.basic_die(size = (10000, 10000), street_width = 100, street_length = 1000, 
              die_name = 'SCE001', text_size = 300, text_location = 'SW',  layer = 0,  
              datatype = 0, draw_bbox = False,  bbox_layer = 99,  bbox_datatype = 99) )
              
fill = dummy_fill_rectangular(d, fill_size = (50,50), exclude_layers = None, fill_layers = (0,1), fill_densities = (0.2, 0.2), margin = 100, bbox = None)
d.add_device( fill )

d.write_gds('SCE001 SNSPD Integrator.gds', precision = 1e-9)

