# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 17:47:14 2016

@author: anm16
"""


from __future__ import division # Otherwise integer division e.g.  20 / 7 = 2
from __future__ import print_function # Use print('hello') instead of print 'hello'
#from __future__ import unicode_literals
import numpy as np
import gdspy
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.collections import PatchCollection


class Port(object):
    def __init__(self, midpoint = [0,0], width = 1, orientation = 90):
        self.midpoint = midpoint
        self.width = width
        self.orientation = orientation


class Device(gdspy.Cell):
    def __init__(self, name, exclude_from_global=True):
        super(Device, self).__init__(name, exclude_from_global)
        self.ports = {}

    def add_device(self, device):
        subdevice = SubDevice(device)   # Create a SubDevice (CellReference)
        self.add(subdevice)             # Add SubDevice (CellReference) to Device (Cell)
        return subdevice                # Return the SubDevice (CellReference)

    # QUESTION: Could make this add_element but that implies we're stuck to GDS elements
    def add_polygon(self, polygon):
        self.add(polygon)
        return polygon
        
    def add_port(self, name, midpoint = [0,0], width = 1, orientation = 90):
        if self.ports.has_key(name):
            raise ValueError('[DEVICE] add_port() error: Port name already exists in this device') 
        p = Port(midpoint, width, orientation)
        self.ports[name] = p
        return p
        
    def get_ports(self):
        return self.ports
        
    def remove_port(self, name):
        self.ports.pop(name, None)
        

        
        
#    def add_subdevice(self, subdevice):
#        if type(subdevice) is Device:
#            pass # Create the device reference
#        if type(subdevice) is DeviceReference:
#            pass # 
#        if type(subdevice) is list:
#            if type(device[0]) is Device:
#                pass
#            if type(device[0]) is DeviceReference:
#                pass
#    
#    def get_ports(self, depth=None):
#        pass
#    
#    
#        
#    def copy_port(self, port, new_port_name):
#        p = Port(port.midpoint, port.width, port.orientation)
    
    
        
    
    
class SubDevice(gdspy.CellReference):
    def __init__(self, device, origin=(0, 0), rotation=0, magnification=None, x_reflection=False):
        super(SubDevice, self).__init__(device, origin, rotation, magnification, x_reflection)
        self.ports = device.ports
        
    def translate(self, dx = 0, dy = 0):
        self.origin = np.array(self.origin) + np.array([dx,dy])
        
        
    def move(self, origin = [0,0], destination = [0,0]):
        """
        Moves the SubDevice from the origin point to the destination.  Both origin
        and destination can be 1x2 array-like, Port, or a string with name of a
        Pot in this subdevice
        """
        if type(origin) is Port:
            o = origin.midpoint
        elif type(origin) is str:
            port = self.get_port(origin)
            o = port.midpoint
        else:
            o = origin
            
        if type(destination) is Port:
            d = destination.midpoint
        elif type(destination) is str:
            port = self.get_port(destination)
            d = port.midpoint
        else:
            d = destination
            
        self.origin = np.array(self.origin) + np.array(d) - np.array(o)
        
        
    def rotate(self, angle = 45):
        self.rotation += angle
        
        
    def get_port(self, name):
        """ Returns a translated, reflected, and rotated version of the 
        subdevice's port which corresponds the location of the port in the 
        Device where this SubDevice lives """
        
        # First get the base location of the referenced port
        ref_port = self.ports[name]
        width = ref_port.width
        new_midpoint = np.array(ref_port.midpoint)
        new_orientation = ref_port.orientation
        
        # Then transform the referenced port location according to the
        # transformations in the SubDevice (CellReference) properties
        if self.x_reflection:
            new_midpoint[0] = -new_midpoint[0]
            new_orientation = mod(180-orientation, 360)
        if self.magnification is not None:
            pass
        if self.rotation is not None:
            ct = np.cos(self.rotation * np.pi / 180.0)
            st = np.sin(self.rotation * np.pi / 180.0)
            st = np.array([-st, st])
            new_midpoint = new_midpoint * ct + new_midpoint[::-1] * st
            new_orientation += self.rotation
        if self.origin is not None:
            new_midpoint = new_midpoint + np.array(self.origin)
        
        return Port(new_midpoint, width, new_orientation)
        
    def get_ports(self):
        return [self.get_port(name) for name in self.ports.keys()]
        
    def connect_port(self, port, destination, translate = True, rotate = True):
        # port can either be a string with the name or an actual Port
        if type(port) is str:
            p = self.get_port(name = port)
        if type(port) is Port:
            p = port
        
        if translate is True:
            self.translate(d = p.midpoint, origin = destination.midpoint)
        if rotate is True:
            self.rotate(angle =  p.orientation - destination.orientation)
        
        
        
                    
def quickplot(items, overlay_ports = True):
    fig, ax = plt.subplots()
    
    if type(items) is not list:  items = [items]
    for item in items:
        if type(item) is Device or type(item) is SubDevice:
            polygons = item.get_polygons(by_spec=False, depth=None)
            patches = []
            for p in polygons:
#                p.append(p[-1]) # Close polygon
                xy = zip(*p)
#                plt.plot(xy[0], xy[1], '.-')
                patches.append(PolygonPatch(p, closed=True, alpha = 0.4))
            for port in item.get_ports():
                pass # TODO Draw ports too
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
    
def waveguide(name = 'snspd', width = 1, height = 1):
    wg = Device(name)
    wg.add_polygon(gdspy.Polygon([(0, 0), (width, 0), (width, height), (0, height)]))
    wg.add_port(name = 'wgport1', midpoint = [3,3], width = 1, orientation = 45)
    wg.add_port(name = 'wgport2', midpoint = [2,1.5], width = 1, orientation = -60)
    return wg


#%% How we might want to be coding

# Construct a new device 'd'
d = Device('Integrated')

# Create an SNSPD and waveguide and add references to them into 'd'
snspd = d.add_device(SNSPD(name = 'my_snspd', config = 'snspd22.yaml'))
wg1 = d.add_device(waveguide(name = 'important_wg', width=5, height = 10))

# Create another waveguide separately, then add it to new device 'd'
temp = waveguide(width=7, height = 1) # Should this return a DeviceReference?
wg2 = d.add_device(temp) # This replaces wg2 with its DeviceReference
wg2.translate(dx = 0.5, dy = 1.7)

# Manipulate the subdevice references
snspd.translate(dx = 4, dy = 6) # Move by dx = 4, dy = 6
snspd.move(origin = [4,6], destination = [5,9]) # can either move from point to point
snspd.rotate(angle = 45)
wg1.translate(dx = 1, dy = 2) # Calculates dx, dy automatically

# To implement: Translate using Ports or their names
snspd.move(origin = [5,6], destination = 'term2') # Takes port and sends to destination
wg1.move(origin = 'wgport1', destination = snspd.get_port('term2')) # Takes port and sends to destination

# Add some new geometry
poly1 = d.add_polygon(gdspy.Polygon([(0, 0), (2, 2), (2, 6), (-6, 6), (-6, -6), (-4, -4), (-4, 4), (0, 4)]))
poly2 = gdspy.Polygon([(2, 2), (2, 6), (-6, 6)])
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
    
    


rho_c = 10
theta_c = 5*pi/180.0
theta_res = 10*pi/180.0
L_arm = 100
L_gnd = 100
W_ch = 100
W_g = 100

ytron_poly, ytron_ports = ytron_polygon(rho_c, theta_c, theta_res, L_arm, L_gnd, W_ch, W_g)
sine_poly, sine_ports = sine_curve_polygon(width_begin = L_arm, width_end = L_arm*2, length = L_arm*4, mirrored = False, num_pts = 20)

#sine_poly.translate(dx = left_port[0], dy = left_port[1])
connect_ports(sine_poly,   ytron_ports['left_arm_midpoint'], sine_ports['begin_midpoint'],rotation = np.pi/2)

quickplot_polygon(ytron_poly)
quickplot_polygon(sine_poly)

quickplot_ports(ytron_ports)
quickplot_ports(sine_ports)
plt.axis('square')


xpts, ypts, ports = ytron_points(rho_c, theta_c, theta_res, L_arm, L_gnd, W_ch, W_g)
plot(xpts,ypts, '.-')
axis('square')


# %% More ideal version








# %% GDSPY code



