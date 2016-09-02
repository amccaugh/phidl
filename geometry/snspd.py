import numpy as np
import gdspy


def snspd(wire_width = 0.2, wire_pitch = 0.6, size = (3,3),
          terminals_same_side = False, layer = 0, datatype = 0):
    xsize = size[0]
    ysize = size[1]
    num_meanders = int(ysize/wire_pitch)
    if terminals_same_side: num_meanders += mod(num_meanders,2) # Make number of meanders even
    
    d = Device(name = 'snspd')
    hairpin = optimal_hairpin(width = wire_width, pitch = wire_pitch, length = xsize/2, num_pts = 20)
    
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
    
    
    c_nw = d.add_device(compass(size = [xsize/2 ,wire_width]))
    c_se = d.add_device(compass(size = [xsize/2 ,wire_width]))
    c_nw.connect('E', top_port)
    c_se.connect('E', bottom_port)
    
    d.add_port(port = c_nw.ports['W'], name = 1)
    d.add_port(port = c_se.ports['W'], name = 2)
    
    return d

    
def snspd_expanded(wire_width = 0.2, wire_pitch = 0.6, size = (3,3), connector_width = 1,
           num_pts = 20, terminals_same_side = False, layer = 0, datatype = 0):
    """ Creates an optimally-rounded SNSPD with wires coming out of it that expand"""
    d = Device('snspd_expanded')
    s = d.add_device(snspd(wire_width = wire_width, wire_pitch = wire_pitch, size = size,
                     terminals_same_side = terminals_same_side, layer = layer, datatype = datatype))
    step_device = optimal_step(start_width = wire_width, end_width = connector_width, num_pts = 50, anticrowding_factor = 2, width_tol = 1e-3)
    step1 = d.add_device(step_device)
    step2 = d.add_device(step_device)
    d.connect(step1.ports[1], destination = s.ports[1])
    d.connect(step2.ports[1], destination = s.ports[2])
    d.add_port(name = 1, port = step1.ports[2])
    d.add_port(name = 2, port = step2.ports[2])
    
    return d
    
    

def snspd_squares(wire_width = 0.2, wire_pitch = 0.6, size = (3,3), num_pts = 20,
                  terminals_same_side = False, layer = 0, datatype = 0):
    pass
    
#==============================================================================
# Example code
#==============================================================================
    
#s = snspd(wire_width = 0.2, wire_pitch = 0.6, size = [10,3], num_pts = 20, terminals_same_side = True)
#quickplot(s)


#step = optimal_step(start_width = 10, end_width = 1, num_pts = 50, width_tol = 1e-3)
#quickplot(step)