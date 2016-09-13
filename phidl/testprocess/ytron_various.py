from __future__ import division, print_function, absolute_import
from phidl import Device, quickplot, inset
import phidl.geometry as pg


def ytron_with_pads(
        pad_flag_size = [300,300],
        width_right = 20,
        width_left = 20,
        rho_intersection = 1,
        theta_intersection = 5,
        arm_length = 50,
        source_length = 50,
        inset_distance = 1,
        label = 'A1',
        ):
        
    #==============================================================================
    # yTron with pads
    #==============================================================================
    
    # Create blank device
    D = Device('yTron-various')
    
    
    #==============================================================================
    # Create components
    #==============================================================================
    f_right = D.add_device(pg.flagpole(flag_size = pad_flag_size, pole_size = [width_right,width_right], shape = 'p', taper_type = 'fillet', layer = 0, datatype = 0))
    f_left = D.add_device(pg.flagpole(flag_size = pad_flag_size, pole_size = [width_left,width_left], shape = 'q', taper_type = 'fillet', layer = 0, datatype = 0))
    gnd = D.add_device( pg.tee(top_size = pad_flag_size, leg_size = [width_right+width_left, width_right+width_left], taper_type = 'fillet', layer = 0, datatype = 0) ).rotate(180)
    y =  D.add_device(pg.ytron_round(rho_intersection, theta_intersection, arm_length, source_length,
                      width_right, width_left, theta_resolution = 10, layer = 0, datatype = 0))
    
    
    #==============================================================================
    # Connect/place components
    #==============================================================================
    
    D.connect(port = f_right.ports[1], destination = y.ports['right']).move([100*np.sin(theta_intersection/180*np.pi), 100])
    D.connect(port = f_left.ports[1], destination = y.ports['left']).move([-100*np.sin(theta_intersection/180*np.pi), 100])
    D.connect(port = gnd.ports['S'], destination = y.ports['source'])
    
    
    
    
    #==============================================================================
    # Route components
    #==============================================================================
    r_ytron_right = D.route(y.ports['right'], f_right.ports[1], path_type = 'straight', width_type = 'sine')
    r_ytron_left = D.route(y.ports['left'], f_left.ports[1], path_type = 'straight', width_type = 'sine')
    
    
    #==============================================================================
    # Create and place label
    #==============================================================================
    
    l = D.add_device( pg.text(text = label, justify = 'right', size = 100, layer = 1) )
    (l.xmax, l.ymin) = (gnd.xmin - 50, gnd.ymin)
    
    
    #==============================================================================
    # Add gold pads which are slightly inset
    #==============================================================================
    D.add_polygon( inset(gnd, distance = inset_distance, join_first = True, layer=1, datatype=0) )
    D.add_polygon( inset([r_ytron_right, f_right], distance = inset_distance, join_first = True, layer=1, datatype=0) )
    D.add_polygon( inset([r_ytron_left, f_left], distance = inset_distance, join_first = True, layer=1, datatype=0) )
    
    
    return D




#==============================================================================
# Die creation script
#==============================================================================
d = Device()

#==============================================================================
# Row A: Varying sharpness of yTron intersection (rho_intersection)
#==============================================================================
rho = [0.25,0.5,1,1.5,2]

for n, r in enumerate(rho):
    s = d.add_device( ytron_with_pads(label = 'A'+str(n+1), width_right = 20, width_left = 20, rho_intersection = r) )
    s.move([(s.width + 300)*n, 0])
    d.label(('Varying yTron rho\n rho = %s\n20um arms' % rho[n]), s.center)
    
quickplot(d)
#==============================================================================
# Row B: Varying sharpness of yTron intersection (rho_intersection) (different arm width)
#==============================================================================
rho = [0.25,0.5,1,1.5,2]
for n, r in enumerate(rho):
    s = d.add_device( snspd_integrator(label = 'B'+str(n+1), width_right = 10, width_left = 20, rho_intersection = rho[n], num_devices = 5) )
    s.move([(s.width + 300)*n, -(s.height+200)])
    d.label(('Varying yTron rho\n rho = %s\n20 & 10um arms' % rho[n]), s.center)
    
    
#==============================================================================
# Row C: Varying sharpness of yTron intersection (rho_intersection) (different arm width)
#==============================================================================
rho = [0.25,0.5,1,1.5,2]
for n, r in enumerate(rho):
    s = d.add_device( snspd_integrator(label = 'C'+str(n+1), width_right = 5, width_left = 20, rho_intersection = rho[n], num_devices = 5) )
    s.move([(s.width + 300)*n, -2*(s.height+200)])
    d.label(('Varying yTron rho\n rho = %s\n20 & 5um arms' % rho[n]), s.center)

final_center = d.center
die = d.add_device( pg.basic_die(size = (10000, 10000), street_width = 100, street_length = 1000, 
              die_name = 'SCE001', text_size = 300, text_location = 'SW',  layer = 0,  
              datatype = 0, draw_bbox = True,  bbox_layer = 99,  bbox_datatype = 99) )
die.center = final_center
              
#quickplot(d)
d.write_gds('SNSPD Integrator.gds')

