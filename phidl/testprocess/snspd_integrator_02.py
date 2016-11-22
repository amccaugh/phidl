from __future__ import division, print_function, absolute_import
from phidl import Device, quickplot
import phidl.geometry as pg
import numpy as np


    
#==============================================================================
# SNSPD yTron-based integrator
#==============================================================================
def snspd_integrator(
                    nanowire_width = 0.5,
                    fill_factor = 1/3,
                    contact_pad_width = 5,
                    snspd_size = [100,100],
                    pad_flag_size = [250,250],
                    pad_pole_size = [50, 50],
                    pad_spacing = 50,
                    connector_size = [400,150],
                    inset_distance = 0.5,
                    num_devices = 5,
                    label = 'A1',
                    rho_intersection = 0.5,
                    theta_intersection = 2.5,
                    width_right = 20,
                    width_left = 20,
                    ):

    # Create blank device
    D = Device('SNSPD-yTron-integator')

    source_length = (width_right + width_left)
    arm_length = max([width_right, width_left])*2

    #==============================================================================
    # Create and place components
    #==============================================================================
    cpm = D.add_ref(pg.compass_multi, size = connector_size, ports = {'N':num_devices,'S':1}, layer = 0)
    cpm.center = [0,-200]
    f = D.add_ref(pg.flagpole(size = connector_size, stub_size = [width_right,width_right], shape = 'p', taper_type = 'fillet', layer = 0))
    fy = D.add_ref(pg.flagpole(size = [connector_size[0],connector_size[1]*2], stub_size = [width_left,width_left], shape = 'q', taper_type = 'fillet', layer = 0))
    
    t = pg.tee(size = pad_flag_size, stub_size = pad_pole_size, taper_type = 'fillet', layer = 0)
    pad_array = D.add_array(t, start = [-(t.xsize+pad_spacing)*num_devices/2, 1000], spacing = [t.xsize+pad_spacing, 0], num_devices = num_devices)
    s = pg.snspd_expanded(wire_width = nanowire_width, wire_pitch = nanowire_width*3, size = snspd_size, connector_width = contact_pad_width,
                                             terminals_same_side = False, layer = 0).rotate(90)
    snspd_array = D.add_array(s, start = [-100,300], spacing = [s.xsize+20, 0], num_devices = num_devices)
               
    y =  D.add_ref(pg.ytron_round(rho_intersection, theta_intersection, arm_length, source_length,
                      width_right, width_left, theta_resolution = 10, layer = 0))
    f.connect(port = 2, destination = cpm.ports['S1'])
    y.connect(port = 'right', destination = f.ports[1])
    fy.connect(port = 1, destination = y.ports['left']).move([-200*np.sin(theta_intersection/180*np.pi), 200])
    
    gnd = D.add_ref(pg.compass(size = pad_flag_size, layer = 0))
    gnd.move(origin = gnd.center, destination = y.ports['source'])
    gnd.move([0,-500])
    
    
    #==============================================================================
    # Route components
    #==============================================================================
    pad_routes = []
    connector_routes = []
    for n in range(num_devices):
        s = snspd_array[n]; p = pad_array[n]
        rc = D.add_ref( pg.route(port1 = s.ports[1], port2 = cpm.ports['N%s' % (n+1)], path_type = 'sine', width_type = 'sine', width1 = None, width2 = contact_pad_width*4, num_path_pts = 99, layer = 0) )
        rp = D.add_ref( pg.route(port1 = p.ports['S'], port2 = s.ports[2], path_type = 'sine', width_type = 'sine', width1 = None, width2 = None, num_path_pts = 99, layer = 0) )
        pad_routes.append(rp)
        connector_routes.append(rc)
    r_ytron_gnd = D.add_ref( pg.route(y.ports['source'], gnd.ports['N'], path_type = 'sine', width_type = 'sine') )
    r_ytron_pad = D.add_ref( pg.route(y.ports['left'], fy.ports[1], path_type = 'straight', width_type = 'sine') )
    
    
    #==============================================================================
    # Add gold pads which are slightly pg.inset
    #==============================================================================
    D.add_ref( pg.inset([r_ytron_gnd, gnd], distance = inset_distance, join_first = True, layer=1) )
    D.add_ref( pg.inset([r_ytron_gnd, gnd], distance = inset_distance, join_first = True, layer=1) )
    D.add_ref( pg.inset(pad_array + pad_routes, distance = inset_distance, join_first = True, layer=1) )
    D.add_ref( pg.inset([f,cpm] + connector_routes, distance = inset_distance, join_first = True, layer=1) )
    D.add_ref( pg.inset([fy, r_ytron_pad], distance = inset_distance, join_first = True, layer=1) )
    
    
    #==============================================================================
    # Label device
    #==============================================================================
    D.add_ref( pg.text(label, justify = 'right', size = 200, layer = 1) ).center = [-500,-500]



    return D

    

#==============================================================================
# To change in this version
#==============================================================================
# Wire-bondable resistor shunt for ytron integrator arm?
# Inductor on yTron readout arm

die_name = 'SE005test'
D = Device()

#==============================================================================
# Row A: Varying rho with right arm (SNSPD-integrator)
#==============================================================================
rho = [0.2,0.3,0.4,0.5,0.75]
for n, r in enumerate(rho):
    s = D.add_ref( snspd_integrator(label = 'A'+str(n+1), width_right = 5, width_left = 5, rho_intersection = r, num_devices = 5) )
    s.move([(s.xsize + 300)*n, 0])
    D.annotate(('Varying yTron rho\n rho = %s\n5um arms' % rho[n]), s.center)
    
    
#==============================================================================
# Row B: Varying right arm (SNSPD-integrator) widths with left arm 3 um
#==============================================================================
right_arm = [2,2.5,3,3.5,4]
for n, r in enumerate(right_arm):
    s = D.add_ref( snspd_integrator(label = 'B'+str(n+1), width_right = r, width_left = 3, rho_intersection = 0.3, num_devices = 5) )
    s.move([(s.xsize + 300)*n, -(s.ysize+200)])
    D.annotate(('Varying yTron right arm width\n Right arm = %sum\nLeft arm = 3um' % r), s.center)
    
    
#==============================================================================
# Row C: Varying left arm (readout arm) widths with right arm 3 um
#==============================================================================
left_arm = [3,6,9,12,15]
for n, l in enumerate(left_arm):
    s = D.add_ref( snspd_integrator(label = 'C'+str(n+1), width_right = 3, width_left = l, rho_intersection = 0.3, num_devices = 5) )
    s.move([(s.xsize + 300)*n, -2*(s.ysize+200)])
    D.annotate(('Varying yTron right arm width\nRight arm = 3um\n Left arm = %sum' % l), s.center)
    

    
D.center = (0,0)
die = D.add_ref( pg.basic_die(size = (10000, 10000), street_width = 100, street_length = 1000, 
              die_name = die_name, text_size = 300, text_location = 'SW',  layer = 0,  
              draw_bbox = False,  bbox_layer = 99) )
              
#D.add_ref( pg.fill_rectangle(D, fill_size = (50,50), exclude_layers = None,
#                                fill_layers = (0,1), fill_densities = (0.2, 0.2), margin = 100, bbox = None) )

D.write_gds('%s SNSPD Integrator.gds' % die_name, precision = 1e-9)

