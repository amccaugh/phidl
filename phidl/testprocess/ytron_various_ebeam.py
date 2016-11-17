from __future__ import division, print_function, absolute_import
import numpy as np
from phidl import Device, quickplot, inset
import phidl.geometry as pg


def ytron_ebeam_with_pads(
        pad_size = [300,300],
        connector_width = 4,
        width_right = 0.3,
        width_left = 0.3,
        rho_intersection = 0.025,
        theta_intersection = 5,
        arm_length = 10,
        source_length = 10,
        inset_distance = 1,
        label = 'A1',
        ):
        
    #==============================================================================
    # yTron with pads
    #==============================================================================
    
    # Create blank device
    D = Device('yTron-ebeam')
    
    
    #==============================================================================
    # Create components
    #==============================================================================
    f_right = D.add_device(pg.flagpole(size = pad_size, stub_size = [connector_width,connector_width], shape = 'p', taper_type = 'fillet', layer = 1, datatype = 0))
    f_left =  D.add_device(pg.flagpole(size = pad_size, stub_size = [connector_width,connector_width], shape = 'q', taper_type = 'fillet', layer = 1, datatype = 0))
    y =  D.add_device(pg.ytron_round(rho_intersection, theta_intersection, arm_length, source_length,
                      width_right, width_left, theta_resolution = 10, layer = 0, datatype = 0))
    gnd = D.add_device( pg.tee(size = pad_size, stub_size = [connector_width, connector_width], taper_type = 'fillet', layer = 1, datatype = 0) ).rotate(180)
    
    
    #==============================================================================
    # Connect/place components
    #==============================================================================
    
    f_right.connect(port = 1, destination = y.ports['right']).move([50*np.sin(theta_intersection/180*np.pi), 50])
    f_left.connect(port = 1, destination = y.ports['left']).move([-50*np.sin(theta_intersection/180*np.pi), 50])
    gnd.connect(port = 'S', destination = y.ports['source']).move([0,-20])
    
    
    
    
    #==============================================================================
    # Route components
    #==============================================================================
    r_ytron_right = D.route(y.ports['right'], f_right.ports[1], path_type = 'straight', width_type = 'sine', layer = 0)
    r_ytron_left =  D.route(y.ports['left'], f_left.ports[1], path_type = 'straight', width_type = 'sine', layer = 0)
    r_ytron_gnd =   D.route(y.ports['source'], gnd.ports['S'], path_type = 'straight', width_type = 'sine', layer = 0)
    
    
    #==============================================================================
    # Create and place label
    #==============================================================================
    
    l = D.add_device( pg.text(text = label, justify = 'right', size = 150, layer = 1) )
    (l.xmax, l.ymin) = (gnd.xmin - 50, gnd.ymin)
    
    
    #==============================================================================
    # Add gold pads which are slightly inset
    #==============================================================================
    D.add_polygon( inset([gnd,r_ytron_gnd], distance = inset_distance, join_first = True, layer=1, datatype=0) )
    D.add_polygon( inset([r_ytron_right, f_right], distance = inset_distance, join_first = True, layer=1, datatype=0) )
    D.add_polygon( inset([r_ytron_left, f_left], distance = inset_distance, join_first = True, layer=1, datatype=0) )
    
    
    return D




#==============================================================================
# Die creation script
#==============================================================================
D = Device()
die_name = 'SE003'

#==============================================================================
# Row A: Varying sharpness of yTron intersection (rho_intersection)
#==============================================================================
rho = [0.005,0.010,0.020,0.040,0.060,0.080,0.120,0.160]

y = 0
x = 0
for n, r in enumerate(rho):
    s = D.add_device( ytron_ebeam_with_pads(label = 'A'+str(n+1), width_right = 0.3, width_left = 0.3, rho_intersection = r) )
    s.move([(s.xsize + 300)*n, 0])
    D.add_label(('Varying sharpness\n rho = %s\n300nm arms' % rho[n]), s.center)



#==============================================================================
# Row B: Varying sharpness of yTron intersection (rho_intersection) with wider wires
#==============================================================================
rho = [0.005,0.010,0.020,0.040,0.060,0.080,0.120,0.160]

y = D.ymin - 500
x = 0
for n, r in enumerate(rho):
    s = D.add_device( ytron_ebeam_with_pads(label = 'B'+str(n+1), width_right = 0.6, width_left = 0.6, rho_intersection = r) )
    s.ymax = y
    s.xmin = x
    x = s.xmax + 200
    D.add_label(('Varying sharpness\n rho = %s\n600nm arms' % rho[n]), s.center)



#==============================================================================
# Row C: Varying arm widths
#==============================================================================.
rho = 0.040
width_left = 1
width_right = 2**(np.array(range(-4,3))+1.0)*width_left

y = D.ymin - 500
x = 0
for n, w in enumerate(width_right):
    s = D.add_device( ytron_ebeam_with_pads(label = 'C'+str(n+1), width_right = w, width_left = width_left, rho_intersection = rho, arm_length = 30, source_length = 30) )
    s.ymax = y
    s.xmin = x
    x = s.xmax + 200
    D.add_label(('Varying arm width ratio\nRight arm width = %sum\nLeft arm width = 1um' % width_right[n]), s.center)

    
D2 = Device()
d1 = D2.add_device(D)
d2 = D2.add_device(D)
d2.ymax = d1.ymin - 1000


D2.center = [0,0]
die = D2.add_device( pg.basic_die(size = (10000, 10000), street_width = 100, street_length = 1000, 
              die_name = die_name, text_size = 300, text_location = 'SW',  layer = 1,  
              datatype = 0, draw_bbox = False) )
              

              
#quickplot(D)

fill = dummy_fill_rectangular(D2, fill_size = (50,50), exclude_layers = None, fill_layers = (0,1), fill_densities = (0.2, 0.2), margin = 100, bbox = None)
D2.add_device( fill )
D2.write_gds('%s yTron ebeam variations.gds' % die_name)
