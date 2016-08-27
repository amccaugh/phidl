import numpy as np
import gdspy

def compass(size = [4,2], center = [0,0], layer = 0, datatype = 0):
    """ Creates a rectangular contact pad with centered ports on edges of the
    rectangle (north, south, east, and west)
    """
    
    d = Device(name = 'contact_compass')
    d.add_polygon(rectangle_centered(size, center), layer = layer, datatype = datatype)
    
    dx = size[0]
    dy = size[1]
    d.add_port(name = 'N', midpoint = [0, dy/2],  width = dx, orientation = 90)
    d.add_port(name = 'S', midpoint = [0, -dy/2], width = dx, orientation = -90)
    d.add_port(name = 'E', midpoint = [dx/2, 0],  width = dy, orientation = 0)
    d.add_port(name = 'W', midpoint = [-dx/2, 0], width = dy, orientation = 180)
    
    return d
    
    
# TODO fix centering of this
def compass_multi(size = [4,2], ports = {'N':3,'S':4}, center = [0,0], layer = 0, datatype = 0):
    """ Creates a rectangular contact pad with multiple ports along the edges
    rectangle (north, south, east, and west).
    """
    
    d = Device(name = 'contact_compass_multi')
    d.add_polygon(rectangle_centered(size, center = [0,0]))
    
    dx = size[0]/2
    dy = size[1]/2
    
    if ports.has_key('N'):
        num_ports = ports['N']
        m = dx-dx/num_ports
        p_list = np.linspace(-m, m, num_ports)
        [d.add_port(name = ('N%s' % (n+1)), midpoint = [p, dy],  width = dx/num_ports*2, orientation = 90) for n,p in enumerate(p_list)]
    if ports.has_key('S'):
        num_ports = ports['S']
        m = dx-dx/num_ports
        p_list = np.linspace(-m, m, num_ports)
        [d.add_port(name = ('S%s' % (n+1)), midpoint = [p, -dy],  width = dx/num_ports*2, orientation = -90) for n,p in enumerate(p_list)]
    if ports.has_key('E'):
        num_ports = ports['E']
        m = dy-dy/num_ports
        p_list = np.linspace(-m, m, num_ports)
        [d.add_port(name = ('E%s' % (n+1)), midpoint = [dx, p],  width = dy/num_ports*2, orientation = 0) for n,p in enumerate(p_list)]
    if ports.has_key('W'):
        num_ports = ports['W']
        m = dy-dy/num_ports
        p_list = np.linspace(-m, m, num_ports)
        [d.add_port(name = ('W%s' % (n+1)), midpoint = [-dx, p],  width = dy/num_ports*2, orientation = 180) for n,p in enumerate(p_list)]
    
    d.move(origin = [0,0], destination = center)
    return d
    
    

# TODO: Fix the fillet here, right now only goes halfway down
def flagpole(flag_size = [4,2], pole_size = [2,1], shape = 'p', taper_type = 'straight', layer = 0, datatype = 0):
    f = deepcopy(flag_size)
    p = deepcopy(pole_size)
    if shape ==   'p':
        orientation = -90
    elif shape == 'q':
        f[0], p[0] = -flag_size[0], -pole_size[0]
        orientation = -90
    elif shape == 'b':
        f[1], p[1] = -flag_size[1], -pole_size[1]
        orientation = 90
    elif shape == 'd':
        f[1], p[1] = -flag_size[1], -pole_size[1]
        f[0], p[0] = -flag_size[0], -pole_size[0]
        orientation = 90
    xpts = [0, 0, f[0], f[0], p[0], p[0], 0]
    ypts = [0, f[1], f[1], 0, 0, -p[1], -p[1]]
    
    d = Device(name = 'tapered')
    pad_poly = d.add_polygon([xpts,ypts], layer = layer, datatype = datatype)
    if taper_type == 'fillet':
        taper_amount = min([abs(f[0]-p[0]), abs(p[1])])
        pad_poly.fillet([0,0,0,0,taper_amount,0,0])
    if taper_type == 'straight':
        taper_poly = d.add_polygon([xpts[3:6],ypts[3:6]], layer = layer, datatype = datatype)
    
    d.add_port(name = 1, midpoint = [p[0]/2, -p[1]],  width = p[0], orientation = orientation)
    return d


#==============================================================================
# Example code
#==============================================================================

#cp = compass(size = [4,2])
#quickplot(cp)


#cpm = compass_multi(size = [40,20], ports = {'N':3,'S':4, 'E':1, 'W':8}, layer = 0, datatype = 0)
#quickplot(cpm)


#cpm = compass_multi(size = [40,20], ports = {'N':3,'S':4, 'E':1, 'W':8}, layer = 0, datatype = 0)
#inset_polygon = inset(cpm, distance = 2, layer = 1, datatype = 1)
#cpm.add(inset_polygon)
#quickplot(cpm)

#fp = flagpole(flag_size = [4,2], pole_size = [2,1], shape = 'p', taper_type = 'straight', layer = 0, datatype = 0)
#quickplot(tp)