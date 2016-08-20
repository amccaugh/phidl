import numpy as np
import gdspy


def compass(dimensions = [4,2], layer = 0, dtype = 0):
    """ Creates a rectangular contact pad with centered ports on edges of the
    rectangle (north, south, east, and west)
    """
    
    d = Device(name = 'pad_compass')
    d.add_polygon(rectangle_centered(dimensions, center = [0,0]))
    
    dx = dimensions[0]
    dy = dimensions[1]
    d.add_port(name = 'N', midpoint = [0, dy/2],  width = dx, orientation = 90)
    d.add_port(name = 'S', midpoint = [0, -dy/2], width = dx, orientation = -90)
    d.add_port(name = 'E', midpoint = [dx/2, 0],  width = dy, orientation = 0)
    d.add_port(name = 'W', midpoint = [-dx/2, 0], width = dy, orientation = 180)
    
    return d
    
def compass_multi(dimensions = [4,2], ports = {'N':3,'S':4}, layer = 0, dtype = 0):
    """ Creates a rectangular contact pad with multiple ports along the edges
    rectangle (north, south, east, and west).
    """
    
    d = Device(name = 'pad_compass_multi')
    d.add_polygon(rectangle_centered(dimensions, center = [0,0]))
    
    dx = dimensions[0]/2
    dy = dimensions[1]/2
    
    if ports.has_key('N'):
        num_ports = ports['N']
        m = dx-dx/num_ports
        p_list = np.linspace(-m, m, num_ports)
        [d.add_port(name = ('N%s' % n), midpoint = [p, dy],  width = dx/num_ports*2, orientation = 90) for n,p in enumerate(p_list)]
    if ports.has_key('S'):
        num_ports = ports['S']
        m = dx-dx/num_ports
        p_list = np.linspace(-m, m, num_ports)
        [d.add_port(name = ('S%s' % n), midpoint = [p, -dy],  width = dx/num_ports*2, orientation = -90) for n,p in enumerate(p_list)]
    if ports.has_key('E'):
        num_ports = ports['E']
        m = dy-dy/num_ports
        p_list = np.linspace(-m, m, num_ports)
        [d.add_port(name = ('E%s' % n), midpoint = [dx, p],  width = dy/num_ports*2, orientation = 0) for n,p in enumerate(p_list)]
    if ports.has_key('W'):
        num_ports = ports['W']
        m = dy-dy/num_ports
        p_list = np.linspace(-m, m, num_ports)
        [d.add_port(name = ('W%s' % n), midpoint = [-dx, p],  width = dy/num_ports*2, orientation = 180) for n,p in enumerate(p_list)]
    
    return d

#==============================================================================
# Example code
#==============================================================================

#cp = compass(dimensions = [4,2])
#quickplot(cp)


#cpm = compass_multi(dimensions = [40,20], ports = {'N':3,'S':4, 'E':1, 'W':8}, layer = 0, dtype = 0)
#quickplot(cpm)


#cpm = compass_multi(dimensions = [40,20], ports = {'N':3,'S':4, 'E':1, 'W':8}, layer = 0, dtype = 0)
#inset_polygon = inset(cpm, distance = 2, layer = 1, datatype = 1)
#cpm.add(inset_polygon)
#quickplot(cpm)