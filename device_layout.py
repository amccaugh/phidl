# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 17:47:14 2016

@author: anm16
"""


from __future__ import division # Otherwise integer division e.g.  20 / 7 = 2
from __future__ import print_function # Use print('hello') instead of print 'hello'
from __future__ import absolute_import

import numpy as np
import gdspy
from copy import deepcopy

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.collections import PatchCollection



def rotate_points(points, angle = 45, center = [0,0]):
    angle = angle*np.pi/180
    ca = np.cos(angle)
    sa = np.sin(angle)
    sa = np.array((-sa, sa))
    c0 = np.array(center)
    if np.array(points).ndim == 2: 
        return (points - c0) * ca + (points - c0)[:,::-1] * sa + c0
    if np.array(points).ndim == 1: 
        return (points - c0) * ca + (points - c0)[::-1] * sa + c0
    

def reflect_points(points, p1, p2):
    """ Reflects points across the line formed by p1 and p2.  ``points`` may be
    input as either single points [1,2] or array-like[N][2], and will return in kind
    """
    # From http://math.stackexchange.com/questions/11515/point-reflection-across-a-line
    points = np.array(points); p1 = np.array(p1); p2 = np.array(p2);
    if np.array(points).ndim == 1: 
        p = points
        return 2*(p1 + (p2-p1)*np.dot((p2-p1),(p-p1))/np.linalg.norm(p2-p1)**2) - p
    if np.array(points).ndim == 2: 
        return np.array([2*(p1 + (p2-p1)*np.dot((p2-p1),(p-p1))/np.linalg.norm(p2-p1)**2) - p for p in points])

class Port(object):
    # TODO: Make so normal and bounds are properties which can be set and
    # which will set their midpoint and orientation and width accordingly
    def __init__(self, midpoint = [0,0], width = 1, orientation = 90, parent = None):
        self.midpoint = midpoint
        self.width = width
        self.orientation = orientation
        self.parent = parent
        
    def __repr__(self):
        return ('Port (midpoint %s, width %s, orientation %s)' % (self.midpoint, self.width, self.orientation))
        
    def bounds(self):
        dx = self.width/2*np.cos((self.orientation - 90)*np.pi/180)
        dy = self.width/2*np.sin((self.orientation - 90)*np.pi/180)
        left_point = self.midpoint - np.array([dx,dy])
        right_point = self.midpoint + np.array([dx,dy])
        return np.array([left_point, right_point])
        
    def normal(self):
        dx = np.cos((self.orientation)*np.pi/180)
        dy = np.sin((self.orientation)*np.pi/180)
        return np.array([self.midpoint, self.midpoint + np.array([dx,dy])])
        
        
        
class Device(gdspy.Cell):
    id = 0
    
    def __init__(self, name = 'Unnamed', exclude_from_global=True):
        self.ports = {}
        self.parameters = {}
        self.subdevices = []
        Device.id += 1
        name = '%s%06d' % (name, Device.id) # Write name e.g. 'Unnamed000005'
        super(Device, self).__init__(name, exclude_from_global)


    @property
    def layers(self):
        return self.get_layers()
        
    def add_device(self, device):
        subdevice = SubDevice(device)   # Create a SubDevice (CellReference)
        self.add(subdevice)             # Add SubDevice (CellReference) to Device (Cell)
        self.subdevices.append(subdevice) # Add to the list of subdevices (for convenience)
        return subdevice                # Return the SubDevice (CellReference)

    # QUESTION: Could make this add_element but that implies we're stuck to GDS elements
    # TODO: Allow input of either [xpts, ypts], or [[1,2],[0,2],[2,3]]
    def add_polygon(self, polygon, layer = 0, datatype = 0):
        if type(polygon) is gdspy.Polygon:
            pass
        elif type(polygon) is gdspy.PolygonSet:
            pass
        elif len(polygon[0]) == 2: # Then it must be of the form [[1,2],[3,4],[5,6]]
            polygon = gdspy.Polygon(polygon, layer, datatype)
        elif len(polygon[0]) > 2: # Then it must be of the form [[1,3,5],[2,4,6]]
            polygon = gdspy.Polygon(xy2p(polygon), layer, datatype)
        self.add(polygon)
        return polygon
        
    # QUESTION: ``name`` implies string -- should this be id?
    def add_port(self, name, midpoint = [0,0], width = 1, orientation = 90):
        if self.ports.has_key(name):
            raise ValueError('[DEVICE] add_port() error: Port name already exists in this device') 
        p = Port(midpoint, width, orientation, parent = self)
        self.ports[name] = p
        return p
        
    def copy_port(self, port, name = None):
        if name is None: raise ValueError('[DEVICE] copy_port() error: Port must be assigned name') 
        self.add_port(name, port.midpoint, port.width, port.orientation)
        
    def remove_port(self, name):
        self.ports.pop(name, None)
        
    def bbox(self, boundary = None):
        box = self.get_bounding_box() # Returns like [(-1,-2), (4,5)]
        if type(boundary) is str:
            boundary = boundary.upper() # Make uppercase
            if boundary == 'NE':    return np.array(box[1])
            if boundary == 'SE':    return np.array([box[1][0], box[0][1]])
            if boundary == 'SW':    return np.array(box[0])
            if boundary == 'NW':    return np.array([box[0][0], box[1][1]])
            if boundary == 'N':     return box[1][1]
            if boundary == 'S':     return box[0][1]
            if boundary == 'E':     return box[1][0]
            if boundary == 'W':     return box[0][0]
        else: return box
    
    def write_gds(self, filename, unit = 1e-6, precision = 1e-9):
        gdspy.gds_print(filename, cells=[self], name='library', unit=unit, precision=precision)
        
            
    def route(self, port_a, port_b, path_type = 'sine', width_type = 'straight', width_a = None, width_b = None, layer = 0, datatype = 0):
        # TODO: Check for mismatched orientations and give warning
        # TODO: Change port_a and port_b names        
        # Assuming they're both Ports for now
        point_a = np.array(port_a.midpoint)
        if width_a is None:  width_a = port_a.width
        point_b = np.array(port_b.midpoint)
        if width_b is None:  width_b = port_b.width
        orientation = port_a.orientation
        
        separation = point_b - point_a  # Vector drawn from A to B
        distance = np.linalg.norm(separation) # Magnitude of vector from A to B
        rotation = np.arctan(separation[1]/separation[0])*180/np.pi # Rotation of vector from A to B
        angle = rotation - orientation   # If looking out along the normal of ``a``, the angle you would have to look to see ``b``
        forward_distance = distance*np.cos(angle*np.pi/180)
        lateral_distance = distance*np.sin(angle*np.pi/180)
        
        # Create a path assuming starting at the origin and setting orientation = 0
        # use the "connect" function later to move the path to the correct location
        xf = forward_distance
        yf = lateral_distance
        if path_type == 'straight':
            curve_fun = lambda t: [xf*t, yf*t]
            curve_deriv_fun = lambda t: [xf + t*0, yf + t*0]
        if path_type == 'sine':
            curve_fun = lambda t: [xf*t, yf*(1-np.cos(t*np.pi))/2]
            curve_deriv_fun = lambda t: [xf  + t*0, yf*(np.sin(t*np.pi)*np.pi)/2]
        #if path_type == 'semicircle':
        #    def semicircle(t):
        #        t = np.array(t)
        #        x,y = np.zeros(t.shape), np.zeros(t.shape)
        #        ii = (0 <= t) & (t <= 0.5)
        #        jj = (0.5 < t) & (t <= 1)
        #        x[ii] = (np.cos(-np.pi/2 + t[ii]*np.pi/2))*xf
        #        y[ii] = (np.sin(-np.pi/2 + t[ii]*np.pi/2)+1)*yf*2
        #        x[jj] = (np.cos(np.pi*3/2 - t[jj]*np.pi)+2)*xf/2
        #        y[jj] = (np.sin(np.pi*3/2 - t[jj]*np.pi)+1)*yf/2
        #        return x,y
        #    curve_fun = semicircle
        #    curve_deriv_fun = None
        if width_type == 'straight':
            width_fun = lambda t: (width_b - width_a)*t + width_a
        if width_type == 'sine':
            print('hello')
            width_fun = lambda t: (width_b - width_a)*(1-np.cos(t*np.pi))/2 + width_a
        
        route_path = gdspy.Path(width = width_a, initial_point = [0,0])
        route_path.parametric(curve_fun, curve_deriv_fun, number_of_evaluations=99,\
                max_points=199, final_width=width_fun, final_distance=None, layer=layer, datatype=datatype)
        
        # Make the route path into a Device with ports, and use "connect" to move it
        # into the proper location
        d = Device()
        d.add(route_path)
        d.add_port(name = 1, midpoint = [0,0], width = width_a, orientation = 180)
        d.add_port(name = 2, midpoint = [forward_distance,lateral_distance], width = width_b, orientation = 0)
        r = self.add_device(d)
        r.connect(1, port_a)
        return r


    
    
    
class SubDevice(gdspy.CellReference):
    def __init__(self, device, origin=(0, 0), rotation=0, magnification=None, x_reflection=False):
        super(SubDevice, self).__init__(device, origin, rotation, magnification, x_reflection)
        self.parent_ports = device.ports
        self._local_ports = deepcopy(device.ports)
    
    @property
    def ports(self):
        """ This property allows you to access mysubdevice.ports, and receive a copy
        of the ports dict which is correctly rotated and translated"""
        for key in self.parent_ports.keys():
            port = self.parent_ports[key] 
            new_midpoint, new_orientation = self._transform_port(port.midpoint, \
                port.orientation, self.origin, self.rotation, self.x_reflection)
            self._local_ports[key].midpoint = new_midpoint
            self._local_ports[key].orientation = new_orientation
            self._local_ports[key].parent = self
        return self._local_ports


    def _transform_port(self, point, orientation, origin=[0, 0], rotation=None, x_reflection=False):
        # Apply GDS-type transformations (x_ref)
        new_point = np.array(point)
        new_orientation = orientation
        
        if x_reflection:
            new_point[1] = -new_point[1]
            new_orientation = -orientation
        if rotation is not None:
            new_point = rotate_points(new_point, angle = rotation, center = [0, 0])
            new_orientation += rotation
        if origin is not None:
            new_point = new_point + np.array(origin)
            
        return new_point, new_orientation
        
        
        
    def bbox(self, boundary = None):
        """ Returns the bounding box in the format of the southwest and northeast
        corners [(-1,-2), (4,5)].  ``boundary`` can be specified to be edges
        or vertices of the bounding box.  For instance specifying east 'E'
        returns the maximum +x coordinate, while 'NE' returns the max [+x,+y] """
        box = self.get_bounding_box() # Returns like [(-1,-2), (4,5)]
        if type(boundary) is str:
            boundary = boundary.upper() # Make uppercase
            if boundary == 'NE':    return np.array(box[1])
            if boundary == 'SE':    return np.array([box[1][0], box[0][1]])
            if boundary == 'SW':    return np.array(box[0])
            if boundary == 'NW':    return np.array([box[0][0], box[1][1]])
            if boundary == 'N':     return box[1][1]
            if boundary == 'S':     return box[0][1]
            if boundary == 'E':     return box[1][0]
            if boundary == 'W':     return box[0][0]
        else: return box
                
                

    def translate(self, dx = 0, dy = 0):
        self.origin = np.array(self.origin) + np.array([dx,dy])
        
        
    def move(self, origin = [0,0], destination = [0,0]):
        """ Moves the SubDevice from the origin point to the destination.  Both
         origin and destination can be 1x2 array-like, Port, or a key
         corresponding to one of the Ports in this subdevice """
        if type(origin) is Port:            o = origin.midpoint
        elif self.ports.has_key(origin):    o = self.ports[origin].midpoint
        elif np.array(origin).size == 2:    o = origin
        else: raise ValueError('[SubDevice.move()] ``origin`` not array-like, a port, or port name')
            
        if type(destination) is Port:           d = destination.midpoint
        elif self.ports.has_key(destination):   d = self.ports[destination].midpoint
        elif np.array(origin).size == 2:        d = destination
        else: raise ValueError('[SubDevice.move()] ``destination`` not array-like, a port, or port name')
            
        self.origin = np.array(self.origin) + np.array(d) - np.array(o)
        
        
    def rotate(self, angle = 45, center = [0,0]):
        self.rotation += angle
        self.origin = rotate_points(self.origin, angle, center)
        
        
    def reflect(self, p1, p2):
        p1 = np.array(p1);  p2 = np.array(p2)
        # Translate so reflection axis passes through origin
        self.origin = self.origin - p1
        
        # Rotate so reflection axis aligns with x-axis
        angle = np.arctan((p2[1]-p1[1])/(p2[0]-p1[0]))*180/np.pi
        self.origin = rotate_points(self.origin, angle = -angle, center = [0,0])
        self.rotation -= angle
        
        # Reflect across x-axis
        self.x_reflection = not self.x_reflection
        self.origin[1] = -self.origin[1]
        self.rotation = -self.rotation
        
        # Un-rotate and un-translate
        self.origin = rotate_points(self.origin, angle = angle, center = [0,0])
        self.rotation += angle
        self.origin = self.origin + p1
        
        
    def connect(self, port, destination, translate = True, rotate = True, offset = 0):
        # ``port`` can either be a string with the name or an actual Port
        if self.ports.has_key(port):
            p = self.ports[port]
        elif type(port) is Port:
            p = port
        
        if rotate is True:
            self.rotate(angle =  180 + destination.orientation - p.orientation, center = p.midpoint)
        if translate is True:
            self.move(origin = p, destination = destination)
    



def p2xy(points):
    """ Takes in a list of [x,y] pairs and converts them to lists of x points 
    and y points.  So p2xy([[1,5],[2,6],[3,7]]) returns [[1,2,3],[5,6,7]]
    """
    p = np.array(points)
    x = p[:,0]
    y = p[:,1]
    return np.array([x,y])
    
def xy2p(*args):
    """ Takes in lists of x points and y points, e.g. [1,2,3],[5,6,7] and
    converts it to the point format e.g. [[1,5],[2,6],[3,7]].  Can either be
    called as xy2p(xpts, ypts) or as xy2p(xy) where xy = [xpts, ypts]
    """
    if len(args) == 1:      x,y = args[0][0], args[0][1] 
    elif len(args) == 2:    x,y = args[0],    args[1]
    return np.array(zip(*[x,y]))
    
    
def quickplot(items, overlay_ports = True, label_ports = True, new_window = True):
    """ Takes a list of devices/subdevices/polygons or single one of those, and
    plots them.  Also has the option to overlay their ports """
    if new_window: fig, ax = plt.subplots(1)
    else:
        ax = plt.gca()  # Get current figure
        ax.cla()        # Clears the axes of all previous polygons
    
    # Iterate through each each Device/Subdevice/Polygon
    patches = []
    if type(items) is not list:  items = [items]
    for item in items:
        if type(item) is Device or type(item) is SubDevice:
            polygons = item.get_polygons(by_spec=False, depth=None)
            for p in polygons:
                patches.append(PolygonPatch(p, closed=True, alpha = 0.4))
            for name, port in item.ports.items():
                _draw_port(port, arrow_scale = 2, shape = 'full', color = 'k')
                plt.text(port.midpoint[0], port.midpoint[1], name)
        if type(item) is Device:
            for sd in item.subdevices:
                for name, port in sd.ports.items():
                    _draw_port(port, arrow_scale = 1, shape = 'right', color = 'r')
                    plt.text(port.midpoint[0], port.midpoint[1], name)
        if type(item) is gdspy.Polygon:
            p = item.points
            patches.append(PolygonPatch(p, closed=True, alpha = 0.4))
    pc = PatchCollection(patches, alpha=0.4)
    colors = 100*np.random.rand(len(patches))
    pc.set_array(np.array(colors))
    ax.add_collection(pc)
    plt.axis('equal')
    ax.grid(True, which='both', alpha = 0.4)
    ax.axhline(y=0, color='k', alpha = 0.2, linewidth = 1)
    ax.axvline(x=0, color='k', alpha = 0.2, linewidth = 1)
    plt.draw()


def _draw_port(port, arrow_scale = 1, **kwargs):
    x = port.midpoint[0]
    y = port.midpoint[1]
    nv = port.normal()
    n = (nv[1]-nv[0])*arrow_scale
    dx, dy = n[0], n[1]
    xbound, ybound = p2xy(port.bounds())
    #plt.plot(x, y, 'rp', markersize = 12) # Draw port midpoint
    plt.plot(xbound, ybound, 'r', linewidth = 3) # Draw port edge
    plt.arrow(x, y, dx, dy,length_includes_head=True, width = 0.1*arrow_scale, head_width=0.3*arrow_scale, **kwargs)


#my_wg = Device('Waveguide')
#my_wg.add_port(name = 'term1', midpoint = [1,1], width = 1, orientation = 90)
#my_snspd = Device('SNSPD')
#my_snspd.add_port(name = 'term1', midpoint = [1,1], width = 1, orientation = 90)
#my_snspd.add_port(name = 'term2', midpoint = [1,1], width = 1, orientation = 90)
#
#integrated_device = Device('Integrated', exclude_from_global = False)
#wg_ref = DeviceReference(my_wg, (25, 0), rotation=180)
#snspd_ref = DeviceReference(my_snspd, (2, 1), rotation=45)
#p = snspd_ref.get_port('term1')
#print p.orientation
#print p.midpoint
#
#integrated_device.add(wg_ref)       
#integrated_device.add(snspd_ref)


#%% Pre-create some devices
#def SNSPD(name = 'snspd', config = 'snspd.yaml'):
#    snspd = Device(name)
#    snspd.add_polygon(gdspy.Polygon([(0, 0), (2, 2), (2, 6), (-6, 6)]))
#    snspd.add_port(name = 'term1', midpoint = [0,0], width = 1, orientation = 90)
#    snspd.add_port(name = 'term2', midpoint = [5,2], width = 1, orientation = -90)
#    snspd.remove_port('term2')
#    snspd.add_port(name = 'term2', midpoint = [5,3], width = 1, orientation = -90)
#    return snspd
#    
#def waveguide(name = 'waveguide', width = 10, height = 1):
#    wg = Device(name)
#    wg.add_polygon(gdspy.Polygon([(0, 0), (width, 0), (width, height), (0, height)]))
#    wg.add_port(name = 'wgport1', midpoint = [0,height/2], width = height, orientation = 180)
#    wg.add_port(name = 'wgport2', midpoint = [width,height/2], width = height, orientation = 0)
#    return wg


##%% How we might want to be coding
#
#
#
## Construct a new device 'd'
#d = Device('Integrated')
#
## Create an SNSPD and waveguide and add references to them into 'd'
#snspd = d.add_device(SNSPD(name = 'my_snspd', config = 'snspd22.yaml'))
#wg1 = d.add_device(waveguide(name = 'important_wg', width=10, height = 1))
#
## Create another waveguide separately, then add it to new device 'd'
#temp = waveguide(width=7, height = 1) # This creates a Device and calls it temp
#wg2 = d.add_device(temp) # This replaces wg2 with its DeviceReference
#wg2.translate(dx = 0.5, dy = 1.7)
#
## Manipulate the subdevice references
#snspd.translate(dx = 4, dy = 6) # Move by dx = 4, dy = 6
#snspd.move(origin = [4,6], destination = [5,9]) # can either move from point to point
#snspd.rotate(angle = 15)
#wg1.translate(dx = 1, dy = 2) # Calculates dx, dy automatically
#
## To implement: Translate using Ports or their names
#snspd.move(origin = [5,6], destination = 'term2') # Takes port and sends to destination
#wg1.move(origin = 'wgport1', destination = snspd.ports['term2']) # Takes port and sends to destination
#
## Add some new geometry
#poly1 = d.add_polygon(gdspy.Polygon([(0, 0), (2, 2), (2, 6), (-6, 6), (-6, -6), (-4, -4), (-4, 4), (0, 4)]))
#poly2 = gdspy.Polygon([(2.0, 2), (2, 6), (-6, 6)])
#poly1.fillet(0.5) # Can fillet it after adding or before
#poly2.translate(dx = 0.4, dy = 0.6)
#d.add_polygon(poly1)
#d.add_polygon(poly2)
#
## Add new ports to 'd' in a few different ways
#d.add_port(name = 'integratedport1', midpoint = [3,4]) # Use the Device.add_port function
#d.add_port(snspd.get_port('term2'))                    # Copy any existing port
#
## Connect device together
#snspd.connect_port(port = 'term1', destination = wg1.get_port('wgport1'), rotate = False, translate = True)
#snspd.connect_port(port = 'term2', destination = [1,5], orientation = 45) # Can specify either name of port or object
#
## How do you move a port?  Does the port know what it's parent is?
#    # If the port knows it's parent
#        # You can write a general thing to mv
#    # If not
#        # The DeviceReference you want to move must precede
#
#        
##d.connect_ports(snspd.get_port('term2'), wg1.get_port('wgport1')) # BAD idea: which one gets moved?
##d.ports['integratedport2'] = Port(midpoint = [0,1])    # BAD idea: Make the Port() separately and attach it to the ports{} dict
#
#
## 
#d.plot(overlay_ports = True)