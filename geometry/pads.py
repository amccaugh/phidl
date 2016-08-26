import numpy as np
import gdspy

def compass(dimensions = [4,2], center = [0,0], layer = 0, datatype = 0):
    """ Creates a rectangular contact pad with centered ports on edges of the
    rectangle (north, south, east, and west)
    """
    
    d = Device(name = 'contact_compass')
    d.add_polygon(rectangle_centered(dimensions, center), layer = layer, datatype = datatype)
    
    dx = dimensions[0]
    dy = dimensions[1]
    d.add_port(name = 'N', midpoint = [0, dy/2],  width = dx, orientation = 90)
    d.add_port(name = 'S', midpoint = [0, -dy/2], width = dx, orientation = -90)
    d.add_port(name = 'E', midpoint = [dx/2, 0],  width = dy, orientation = 0)
    d.add_port(name = 'W', midpoint = [-dx/2, 0], width = dy, orientation = 180)
    
    return d
    
    
# TODO fix centering of this
def compass_multi(dimensions = [4,2], ports = {'N':3,'S':4}, center = [0,0], layer = 0, datatype = 0):
    """ Creates a rectangular contact pad with multiple ports along the edges
    rectangle (north, south, east, and west).
    """
    
    d = Device(name = 'contact_compass_multi')
    d.add_polygon(rectangle_centered(dimensions, center = [0,0]))
    
    dx = dimensions[0]/2
    dy = dimensions[1]/2
    
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
    
    return d


def flagpole(dimensions = [4,2], connection_dimensions = [2,1], shape = 'p', taper_type = 'straight', layer = 0, datatype = 0):
    if shape ==   'p':
        orientation = -90
    elif shape == 'q':
        dimensions[0], connection_dimensions[0] = -dimensions[0], -connection_dimensions[0]
        orientation = -90
    elif shape == 'b':
        dimensions[1], connection_dimensions[1] = -dimensions[1], -connection_dimensions[1]
        orientation = 90
    elif shape == 'd':
        dimensions[1], connection_dimensions[1] = -dimensions[1], -connection_dimensions[1]
        dimensions[0], connection_dimensions[0] = -dimensions[0], -connection_dimensions[0]
        orientation = 90
    xpts = [0, 0, dimensions[0], dimensions[0], connection_dimensions[0], connection_dimensions[0], 0]
    ypts = [0, dimensions[1], dimensions[1], 0, 0, -connection_dimensions[1], -connection_dimensions[1]]
    
    d = Device(name = 'tapered')
    pad_poly = d.add_polygon([xpts,ypts], layer = layer, datatype = datatype)
    if taper_type == 'fillet':
        taper_amount = min([abs(dimensions[0]-connection_dimensions[0]), abs(connection_dimensions[1])])
        pad_poly.fillet([0,0,0,0,taper_amount,0,0])
    if taper_type == 'straight':
        taper_poly = d.add_polygon([xpts[3:6],ypts[3:6]], layer = layer, datatype = datatype)
    
    d.add_port(name = 1, midpoint = [connection_dimensions[0]/2, -connection_dimensions[1]],  width = connection_dimensions[0], orientation = orientation)
    return d


#==============================================================================
# Example code
#==============================================================================

#cp = compass(dimensions = [4,2])
#quickplot(cp)


#cpm = compass_multi(dimensions = [40,20], ports = {'N':3,'S':4, 'E':1, 'W':8}, layer = 0, datatype = 0)
#quickplot(cpm)


#cpm = compass_multi(dimensions = [40,20], ports = {'N':3,'S':4, 'E':1, 'W':8}, layer = 0, datatype = 0)
#inset_polygon = inset(cpm, distance = 2, layer = 1, datatype = 1)
#cpm.add(inset_polygon)
#quickplot(cpm)

#fp = flagpole(dimensions = [4,2], connection_dimensions = [2,1], shape = 'p', taper_type = 'straight', layer = 0, datatype = 0)
#quickplot(tp)