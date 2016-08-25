import numpy as np
import gdspy


def snspd(wire_width = 0.2, wire_pitch = 0.6, dimensions = [3,3], num_pts = 20, \
          terminals_same_side = False, layer = 0, datatype = 0):
    xsize = dimensions[0]
    ysize = dimensions[1]
    num_meanders = int(ysize/wire_pitch)
    if terminals_same_side: num_meanders += mod(num_meanders,2) # Make number of meanders even
    
    d = Device(name = 'snspd')
    hairpin = optimal_hairpin(width = wire_width, pitch = wire_pitch, length = xsize/2, num_pts = num_pts)
    
    hp2 = d.add_device(hairpin)
    top_port = hp2.ports[1]
    while num_meanders > 1:
        # Repeatedly add two new subdevices
        hp1 = d.add_device(hairpin)
        hp1.rotate(180)
        hp1.connect(2, hp2.ports[2])
        hp2 = d.add_device(hairpin)
        hp2.connect(1, hp1.ports[1])
        
        num_meanders -= 2
        
    bottom_port = hp2.ports[2]
    
    if terminals_same_side is False:
        hp1 = d.add_device(hairpin)
        hp1.rotate(180)
        hp1.connect(2, hp2.ports[2])
        bottom_port = hp1.ports[1]
    
    
    c_nw = d.add_device(compass(dimensions = [xsize/2 ,wire_width]))
    c_se = d.add_device(compass(dimensions = [xsize/2 ,wire_width]))
    c_nw.connect('E', top_port)
    c_se.connect('E', bottom_port)
    
    d.copy_port(c_nw.ports['W'], name = 1)
    d.copy_port(c_se.ports['W'], name = 2)
    
    return d

    
    
    
#==============================================================================
# Example code
#==============================================================================
    
#s = snspd(wire_width = 0.2, wire_pitch = 0.6, dimensions = [10,3], num_pts = 20, terminals_same_side = True)
#quickplot(s)


#step = optimal_step(start_width = 10, end_width = 1, num_pts = 50, width_tol = 1e-3)
#quickplot(step)