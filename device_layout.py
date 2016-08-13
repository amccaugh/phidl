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
    # From http://math.stackexchange.com/questions/11515/point-reflection-across-a-line
    points = np.array(points); p1 = np.array(p1); p2 = np.array(p2);
    if np.array(points).ndim == 1: 
        p = points
        return 2*(p1 + (p2-p1)*np.dot((p2-p1),(p-p1))/np.linalg.norm(p2-p1)**2) - p
    if np.array(points).ndim == 2: 
        return np.array([2*(p1 + (p2-p1)*np.dot((p2-p1),(p-p1))/np.linalg.norm(p2-p1)**2) - p for p in points])



class Port(object):
    def __init__(self, midpoint = [0,0], width = 1, orientation = 90):
        self.midpoint = midpoint
        self.width = width
        self.orientation = orientation
        
        

# TODO: Make it so if you don't specify a name, it auto-assigns a unique one
class Device(gdspy.Cell):
    id = 0    
    
    def __init__(self, name = 'Unnamed', exclude_from_global=True):
        super(Device, self).__init__(name, exclude_from_global)
        self.ports = {}
        self.subdevices = []
        Device.id += 1

    def add_device(self, device):
        subdevice = SubDevice(device)   # Create a SubDevice (CellReference)
        self.add(subdevice)             # Add SubDevice (CellReference) to Device (Cell)
        self.subdevices.append(subdevice) # Add to the list of subdevices (for convenience)
        return subdevice                # Return the SubDevice (CellReference)

    # QUESTION: Could make this add_element but that implies we're stuck to GDS elements
    def add_polygon(self, polygon):
        self.add(polygon)
        return polygon
        
    # QUESTION: ``name`` implies string -- should this be id?
    def add_port(self, name, midpoint = [0,0], width = 1, orientation = 90):
        if self.ports.has_key(name):
            raise ValueError('[DEVICE] add_port() error: Port name already exists in this device') 
        p = Port(midpoint, width, orientation)
        self.ports[name] = p
        return p
        
    def copy_port(self, port, name = None):
        if name is None: raise ValueError('[DEVICE] copy_port() error: Port must be assigned name') 
        self.add_port(name, port.midpoint, port.width, port.orientation)
        
    def remove_port(self, name):
        self.ports.pop(name, None)
        
    def bounding_box(self, boundary = None):
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
    
        
    
    
class SubDevice(gdspy.CellReference):
    def __init__(self, device, origin=(0, 0), rotation=0, magnification=None, x_reflection=False):
        super(SubDevice, self).__init__(device, origin, rotation, magnification, x_reflection)
        self.parent_ports = device.ports
        self._local_ports = deepcopy(device.ports)
    
    @property
    def ports(self):
        """ This property allows you to call mysubdevice.ports, and receive a copy
        of the ports dict which is correctly rotated and translated"""
        for key in self.parent_ports.keys():
            port = self.parent_ports[key] 
            new_midpoint, new_orientation = self._transform_port(port.midpoint, \
                port.orientation, self.origin, self.rotation, self.x_reflection)
            self._local_ports[key].midpoint = new_midpoint
            self._local_ports[key].orientation = new_orientation
        return self._local_ports


    def _transform_port(self, point, orientation, origin=[0, 0], rotation=None, x_reflection=False):
        # Apply GDS-type transformations (x_ref)
        new_point = np.array(point)
        new_orientation = orientation
        
        if x_reflection:
            new_point[1] = -new_point[1]
            new_orientation = mod(180-orientation, 360)
        if rotation is not None:
            new_point = rotate_points(new_point, angle = rotation, center = [0, 0])
            new_orientation += rotation
        if origin is not None:
            new_point = new_point + np.array(origin)
            
        return new_point, new_orientation
        
        
        
    def bounding_box(self, boundary = None):
        """ Returns the bounding box in the format of the southwest and northeast
        corners [(-1,-2), (4,5)].  ``boundary`` can be specified to be edges
        or vertices of the bounding box.  For instance specifying east 'E'
        returns the maximum +x coordinate, while 'NE' returns the max (+x,+y) """
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
        elif np.array(origin).size == 2:     o = origin
        else: raise ValueError('[SubDevice.move()] ``origin`` not array-like, a port, or dict key')
            
        if type(destination) is Port:           d = destination.midpoint
        elif self.ports.has_key(destination):   d = self.ports[destination].midpoint
        elif np.array(origin).size == 2:         d = destination
        else: raise ValueError('[SubDevice.move()] ``destination`` not array-like, a port, or dict key')
            
        self.origin = np.array(self.origin) + np.array(d) - np.array(o)
        
        
    def rotate(self, angle = 45, center = [0,0]):
        self.rotation += angle
        self.origin = rotate_points(self.origin, angle, center)
        
        
    def reflect(self, normal = [1,0]):
        pass
        
        
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
    
    
    def route(self):
        pass



def quickplot(items, overlay_ports = True):
    """ Takes a list of devices/subdevices/polygons or single one of those, and
    plots them.  Also has the option to overlay their ports """
    fig, ax = plt.subplots()
    
    # Iterate through each each Device/Subdevice/Polygon and display it
    if type(items) is not list:  items = [items]
    for item in items:
        if type(item) is Device or type(item) is SubDevice:
            polygons = item.get_polygons(by_spec=False, depth=None)
            patches = []
            for p in polygons:
                xy = zip(*p)
                patches.append(PolygonPatch(p, closed=True, alpha = 0.4))
            for port in item.ports.values():
                plt.plot(port.midpoint[0], port.midpoint[1], 'rp', markersize = 12)
        if type(item) is Device:
            for sd in item.subdevices:
                for port in sd.ports.values():
                    plt.plot(port.midpoint[0], port.midpoint[1], 'y*', markersize = 8)
        if type(item) is gdspy.Polygon:
                p = item.points
                xy = zip(*p)
                patches.append(PolygonPatch(p, closed=True, alpha = 0.4))
    pc = PatchCollection(patches, alpha=0.4)
    colors = 100*np.random.rand(len(patches))
    pc.set_array(np.array(colors))
    ax.add_collection(pc)
    plt.axis('equal')

    plt.show()
            



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
def SNSPD(name = 'snspd', config = 'snspd.yaml'):
    snspd = Device(name)
    snspd.add_polygon(gdspy.Polygon([(0, 0), (2, 2), (2, 6), (-6, 6)]))
    snspd.add_port(name = 'term1', midpoint = [0,0], width = 1, orientation = 90)
    snspd.add_port(name = 'term2', midpoint = [5,2], width = 1, orientation = -90)
    snspd.remove_port('term2')
    snspd.add_port(name = 'term2', midpoint = [5,3], width = 1, orientation = -90)
    return snspd
    
def waveguide(name = 'waveguide', width = 10, height = 1):
    wg = Device(name)
    wg.add_polygon(gdspy.Polygon([(0, 0), (width, 0), (width, height), (0, height)]))
    wg.add_port(name = 'wgport1', midpoint = [0,height/2], width = height, orientation = 180)
    wg.add_port(name = 'wgport2', midpoint = [width,height/2], width = height, orientation = 0)
    return wg


#%% How we might want to be coding



# Construct a new device 'd'
d = Device('Integrated')

# Create an SNSPD and waveguide and add references to them into 'd'
snspd = d.add_device(SNSPD(name = 'my_snspd', config = 'snspd22.yaml'))
wg1 = d.add_device(waveguide(name = 'important_wg', width=10, height = 1))

# Create another waveguide separately, then add it to new device 'd'
temp = waveguide(width=7, height = 1) # This creates a Device and calls it temp
wg2 = d.add_device(temp) # This replaces wg2 with its DeviceReference
wg2.translate(dx = 0.5, dy = 1.7)

# Manipulate the subdevice references
snspd.translate(dx = 4, dy = 6) # Move by dx = 4, dy = 6
snspd.move(origin = [4,6], destination = [5,9]) # can either move from point to point
snspd.rotate(angle = 15)
wg1.translate(dx = 1, dy = 2) # Calculates dx, dy automatically

# To implement: Translate using Ports or their names
snspd.move(origin = [5,6], destination = 'term2') # Takes port and sends to destination
wg1.move(origin = 'wgport1', destination = snspd.ports['term2']) # Takes port and sends to destination

# Add some new geometry
poly1 = d.add_polygon(gdspy.Polygon([(0, 0), (2, 2), (2, 6), (-6, 6), (-6, -6), (-4, -4), (-4, 4), (0, 4)]))
poly2 = gdspy.Polygon([(2.0, 2), (2, 6), (-6, 6)])
poly1.fillet(0.5) # Can fillet it after adding or before
poly2.translate(dx = 0.4, dy = 0.6)
d.add_polygon(poly1)
d.add_polygon(poly2)

# Add new ports to 'd' in a few different ways
d.add_port(name = 'integratedport1', midpoint = [3,4]) # Use the Device.add_port function
d.add_port(snspd.get_port('term2'))                    # Copy any existing port

# Connect device together
snspd.connect_port(port = 'term1', destination = wg1.get_port('wgport1'), rotate = False, translate = True)
snspd.connect_port(port = 'term2', destination = [1,5], orientation = 45) # Can specify either name of port or object

# How do you move a port?  Does the port know what it's parent is?
    # If the port knows it's parent
        # You can write a general thing to mv
    # If not
        # The DeviceReference you want to move must precede

        
#d.connect_ports(snspd.get_port('term2'), wg1.get_port('wgport1')) # BAD idea: which one gets moved?
#d.ports['integratedport2'] = Port(midpoint = [0,1])    # BAD idea: Make the Port() separately and attach it to the ports{} dict


# 
d.plot(overlay_ports = True)



# %% Connecting together several waveguides

d = Device('MultiWaveguide')
wg1 = d.add_device(waveguide(width=10, height = 1))
wg2 = d.add_device(waveguide(width=12, height = 2))
wg3 = d.add_device(waveguide(width=14, height = 3))

quickplot(d)

wg2.move(origin = 'wgport1', destination = wg1.ports['wgport2'])
wg3.move(origin = 'wgport1', destination = wg2.ports['wgport2'])

quickplot(d)

wg3.rotate(angle = 45, center =  wg2.ports['wgport2'].midpoint)

quickplot(d)

wg3.connect(port = 'wgport2', destination = wg1.ports['wgport1'])

quickplot(d)

d.copy_port(name = '1', port = wg1.ports['wgport1'])
d.copy_port(name = '2', port = wg3.ports['wgport2'])

quickplot(d)


dsquared = Device('MultiMultiWaveguide')
mwg1 = dsquared.add_device(d)
mwg2 = dsquared.add_device(d)
mwg2.move(origin = '1', destination = mwg1.get_port('2'))

quickplot(dsquared)


#%%

def quickplot_polygon(p):
    plt.plot(p.points[:,0], p.points[:,1],'.-')
    plt.axis('equal')
    
    
def quickplot_ports(ports):
    [plt.plot(x,y, 'rx') for x,y in ports.values()]
    plt.axis('equal')


def sine_curve_polygon(width_begin, width_end, length, layer = 1, num_pts = 20, mirrored = False):
            
    sine_curve_x = linspace(0, length, num_pts)
    sine_curve_y = (sin(linspace(-pi/2, pi/2, num_pts)) + 1)*(width_end-width_begin)
    sine_curve_y += width_begin
    
    xpts = sine_curve_x.tolist() + [length, 0]
    ypts = sine_curve_y.tolist() + [0, 0]
    
    # Ports
    ports = {
        'begin_midpoint': [0, width_begin/2],
        'end_midpoint': [length, width_end/2],
    }
    
    
    if mirrored: ypts = -ypts
    return gdspy.Polygon(zip(xpts,ypts), layer), ports
    

def ytron_polygon(rho_c = 5, theta_c = 5.0/180*pi, theta_res = 10.0/180*pi, L_arm = 500, L_gnd = 500, W_ch = 200, W_g = 200, layer = 1):
    # theta_c = theta_c*pi/180
    # theta_res = theta_c*pi/180
    thetalist = linspace(-(pi-theta_c),-theta_c, (pi-theta_c)/theta_res)
    semicircle_x = rho_c*cos(thetalist)
    semicircle_y = rho_c*sin(thetalist)+rho_c

    # Rest of yTron
    xc = rho_c*cos(theta_c) 
    yc = rho_c*sin(theta_c) 
    L_arm_x = L_arm*sin(theta_c) 
    L_arm_y = L_arm*cos(theta_c) 

    # Write out x and y coords for yTron polygon
    xpts = semicircle_x.tolist() + [xc+L_arm_x, xc+L_arm_x+W_ch, xc+W_ch,    xc+W_ch, 0, -(xc+W_g), -(xc+W_g), -(xc+L_arm_x+W_g), -(xc+L_arm_x)] 
    ypts = semicircle_y.tolist() + [yc+L_arm_y,      yc+L_arm_y,      yc,   yc-L_gnd, yc-L_gnd,  yc-L_gnd,        yc,        yc+L_arm_y,    yc+L_arm_y] 
    
    # Ports
    ports = {
        'left_arm_midpoint': [-(xc+L_arm_x+W_g/2), yc+L_arm_y],
        'right_arm_midpoint': [xc+L_arm_x+W_ch/2, yc+L_arm_y],
        'source_midpoint': [0, -L_gnd],
    }

    return gdspy.Polygon(zip(xpts,ypts), layer), ports


def connect_ports(poly_to_move, point1, point2, rotation):
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    poly_to_move.translate(dx, dy)
    poly_to_move.rotate(angle = rotation, center = point1)
    
    






#%% Pre-create some devices
def SNSPD(name = 'snspd', config = 'snspd.yaml'):
    snspd = Device(name)
    snspd.add_polygon(gdspy.Polygon([(0, 0), (2, 2), (2, 6), (-6, 6)]))
    snspd.add_port(name = 'term1', midpoint = [0,0], width = 1, orientation = 90)
    snspd.add_port(name = 'term2', midpoint = [5,2], width = 1, orientation = -90)
    snspd.remove_port('term2')
    snspd.add_port(name = 'term2', midpoint = [5,3], width = 1, orientation = -90)
    return snspd
    
def waveguide(name = 'waveguide', width = 10, height = 1):
    wg = Device(name)
    wg.add_polygon(gdspy.Polygon([(0, 0), (width, 0), (width, height), (0, height)]))
    wg.add_port(name = 'wgport1', midpoint = [0,height/2], width = height, orientation = 180)
    wg.add_port(name = 'wgport2', midpoint = [width,height/2], width = height, orientation = 0)
    return wg


#%% How we might want to be coding



# Construct a new device 'd'
d = Device('Integrated')

# Create an SNSPD and waveguide and add references to them into 'd'
snspd = d.add_device(SNSPD(name = 'my_snspd', config = 'snspd22.yaml'))
wg1 = d.add_device(waveguide(name = 'important_wg', width=10, height = 1))

# Create another waveguide separately, then add it to new device 'd'
temp = waveguide(width=7, height = 1) # This creates a Device and calls it temp
wg2 = d.add_device(temp) # This replaces wg2 with its DeviceReference
wg2.translate(dx = 0.5, dy = 1.7)

# Manipulate the subdevice references
snspd.translate(dx = 4, dy = 6) # Move by dx = 4, dy = 6
snspd.move(origin = [4,6], destination = [5,9]) # can either move from point to point
snspd.rotate(angle = 15)
wg1.translate(dx = 1, dy = 2) # Calculates dx, dy automatically

# To implement: Translate using Ports or their names
snspd.move(origin = [5,6], destination = 'term2') # Takes port and sends to destination
wg1.move(origin = 'wgport1', destination = snspd.ports['term2']) # Takes port and sends to destination

# Add some new geometry
poly1 = d.add_polygon(gdspy.Polygon([(0, 0), (2, 2), (2, 6), (-6, 6), (-6, -6), (-4, -4), (-4, 4), (0, 4)]))
poly2 = gdspy.Polygon([(2.0, 2), (2, 6), (-6, 6)])
poly1.fillet(0.5) # Can fillet it after adding or before
poly2.translate(dx = 0.4, dy = 0.6)
d.add_polygon(poly1)
d.add_polygon(poly2)

# Add new ports to 'd' in a few different ways
d.add_port(name = 'integratedport1', midpoint = [3,4]) # Use the Device.add_port function
d.add_port(snspd.get_port('term2'))                    # Copy any existing port

# Connect device together
snspd.connect_port(port = 'term1', destination = wg1.get_port('wgport1'), rotate = False, translate = True)
snspd.connect_port(port = 'term2', destination = [1,5], orientation = 45) # Can specify either name of port or object

# How do you move a port?  Does the port know what it's parent is?
    # If the port knows it's parent
        # You can write a general thing to mv
    # If not
        # The DeviceReference you want to move must precede

        
#d.connect_ports(snspd.get_port('term2'), wg1.get_port('wgport1')) # BAD idea: which one gets moved?
#d.ports['integratedport2'] = Port(midpoint = [0,1])    # BAD idea: Make the Port() separately and attach it to the ports{} dict


# 
d.plot(overlay_ports = True)



# %% Connecting together several waveguides

d = Device('MultiWaveguide')
wg1 = d.add_device(waveguide(width=10, height = 1))
wg2 = d.add_device(waveguide(width=12, height = 2))
wg3 = d.add_device(waveguide(width=14, height = 3))

quickplot(d)

wg2.move(origin = 'wgport1', destination = wg1.ports['wgport2'])
wg3.move(origin = 'wgport1', destination = wg2.ports['wgport2'])

quickplot(d)

d.copy_port(name = '1', port = wg1.get_port('wgport1'))
d.copy_port(name = '2', port = wg3.get_port('wgport2'))

quickplot(d)


dsquared = Device('MultiMultiWaveguide')
mwg1 = dsquared.add_device(d)
mwg2 = dsquared.add_device(d)
mwg2.move(origin = '1', destination = mwg1.get_port('2'))

quickplot(dsquared)