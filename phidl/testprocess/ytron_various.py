from __future__ import division, print_function, absolute_import
import numpy as np
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
    f_right = D.add_device(pg.flagpole(size = pad_flag_size, stub_size = [width_right,width_right], shape = 'p', taper_type = 'fillet', layer = 0, datatype = 0))
    f_left = D.add_device(pg.flagpole(size = pad_flag_size, stub_size = [width_left,width_left], shape = 'q', taper_type = 'fillet', layer = 0, datatype = 0))
    y =  D.add_device(pg.ytron_round(rho_intersection, theta_intersection, arm_length, source_length,
                      width_right, width_left, theta_resolution = 10, layer = 0, datatype = 0))
    gnd = D.add_device( pg.tee(size = pad_flag_size, stub_size = [y.ports['source'].width, y.ports['source'].width], taper_type = 'fillet', layer = 0, datatype = 0) ).rotate(180)
    
    
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
    
    l = D.add_device( pg.text(text = label, justify = 'right', size = 150, layer = 1) )
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
rho = [0.25,0.5,1,1.5,2,4,8,16]

for n, r in enumerate(rho):
    s = d.add_device( ytron_with_pads(label = 'A'+str(n+1), width_right = 20, width_left = 20, rho_intersection = r) )
    s.move([(s.xsize + 300)*n, 0])
    d.annotate(('Varying sharpness\n rho = %s\n20um arms' % rho[n]), s.center)
    
    
#==============================================================================
# Row B: Varying arm widths
#==============================================================================
rho = 1
width_left = 20
width_right = 2**(np.array(range(-5,3))+1.0)*width_left

y = d.ymin - 500
x = 0
for n, w in enumerate(width_right):
    s = d.add_device( ytron_with_pads(label = 'B'+str(n+1), width_right = w, width_left = 20, rho_intersection = rho) )
    s.ymax = y
    s.xmin = x
    x = s.xmax + 200
    d.annotate(('Varying arm width ratio\nLeft arm width = %sum\nLeft arm width = 20um' % width_right[n]), s.center)
    
#==============================================================================
# Row C: Varying source length
#==============================================================================
rho = 1
width_left = 20
width_right = 20
source_length = np.arange(10,100,10)

y = d.ymin - 500
x = 0
for n, p in enumerate(source_length):
    s = d.add_device( ytron_with_pads(label = 'C'+str(n+1), width_right = 20, width_left = 20, rho_intersection = rho, source_length = p) )
    s.ymax = y
    s.xmin = x
    x = s.xmax + 200
    d.annotate(('Varying source length\nLength = %sum' % p), s.center)
    
#==============================================================================
# Row D: Varying arm length
#==============================================================================
rho = 1
width_left = 20
width_right = 20
source_length = 50
arm_length = np.arange(5,130,15)

y = d.ymin - 500
x = 0
for n, p in enumerate(arm_length):
    s = d.add_device( ytron_with_pads(label = 'D'+str(n+1), width_right = 20, width_left = 20, rho_intersection = rho, arm_length = p) )
    s.ymax = y
    s.xmin = x
    x = s.xmax + 200
    d.annotate(('Varying arm length\nLength = %sum' % p), s.center)
    

d.center = [0,0]
die = d.add_device( pg.basic_die(size = (10000, 10000), street_width = 100, street_length = 1000, 
              die_name = 'SCE002', text_size = 300, text_location = 'SW',  layer = 0,  
              datatype = 0, draw_bbox = False,  bbox_layer = 99,  bbox_datatype = 99) )
              
#quickplot(d)

fill = dummy_fill_rectangular(d, fill_size = (50,50), exclude_layers = None, fill_layers = (0,1), fill_densities = (0.2, 0.2), margin = 100, bbox = None)
d.add_device( fill )
d.write_gds('SCE002 yTron variations.gds')
