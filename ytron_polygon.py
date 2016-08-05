# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 17:47:14 2016

@author: anm16
"""


from __future__ import division # Otherwise integer division e.g.  20 / 7 = 2
from __future__ import print_function # Use print('hello') instead of print 'hello'
import numpy as np
import gdspy


class Port(object):
    def __init__(self, midpoint = [0,0], width = 1, orientation = 90):
        self.midpoint = midpoint
        self.width = width
        self.orientation = orientation
#        
#    def rotate(self, angle, center = [0,0]):
#        ca = np.cos(angle)
#        sa = np.sin(angle)
#        sa = np.array((-sa, sa))
#        c0 = np.array(center)
#        self.midpoint = (self.midpoint - c0) * ca + (self.midpoint[::-1] - c0) * sa + c0
#        self.orientation += angle
#        
#    def translate(self, dx, dy):
#        self.midpoint = [self.midpoint[0] + dx, self.midpoint[1] + dy]
         


class Device(gdspy.Cell):
    def __init__(self, name, exclude_from_global=True):
        super(Device, self).__init__(name, exclude_from_global)
        self.ports = {}
        self.devices = {}
        
    def add_port(self, name, midpoint = [0,0], width = 1, orientation = 90):
        p = Port(midpoint, width, orientation)
        self.ports[name] = p
        
    def remove_port(self, name):
        self.ports.pop(name, None)
        
    def copy_port(self, port, new_port_name):
        p = Port(port.midpoint, port.width, port.orientation)
        
    def get_ports(self, depth=None):
        pass
    
    def plot(self, overlay_ports = True):
        pass
    
    def add_device(self, device):
        if type(device) is Device:
            pass # Create the device reference
        if type(device) is DeviceReference:
            pass # 
        if type(device) is list
            if type(device[0]) is Device:
                pass
            if type(device[0]) is DeviceReference:
                pass
    
        
    
    
class DeviceReference(gdspy.CellReference):
    def __init__(self, ref_cell, origin=(0, 0), rotation=None, magnification=None, x_reflection=False):
        super(DeviceReference, self).__init__(ref_cell, origin, rotation, magnification, x_reflection)
        self.ports = ref_cell.ports
        
    def get_port(self, name):
        # First get the base location of the referenced port
        ref_port = self.ports[name]
        width = ref_port.width
        new_midpoint = np.array(ref_port.midpoint)
        new_orientation = ref_port.orientation
        
        # Then transform the referenced port location to match the CellReference
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

    def translate(self, device, dx = None, dy = None, origin = None, destination = None):
        if dx is None and dy is None:
            dx = destination[0] - origin[0]
            dy = destination[1] - origin[1]
        self.translate(dx,dy)


my_wg = Device('Waveguide')
my_wg.add_port(name = 'term1', midpoint = [1,1], width = 1, orientation = 90)
my_snspd = Device('SNSPD')
my_snspd.add_port(name = 'term1', midpoint = [1,1], width = 1, orientation = 90)
my_snspd.add_port(name = 'term2', midpoint = [1,1], width = 1, orientation = 90)

integrated_device = Device('Integrated', exclude_from_global = False)
wg_ref = DeviceReference(my_wg, (25, 0), rotation=180)
snspd_ref = DeviceReference(my_snspd, (2, 1), rotation=45)
p = snspd_ref.get_port('term1')
print p.orientation
print p.midpoint

integrated_device.add(wg_ref)       
integrated_device.add(snspd_ref)


#%% How we might want to be coding

# Construct a new device 'd'
d = Device('Integrated')

# Create an SNSPD and waveguide and add references to them into 'd'
snspd = d.add_device(SNSPD(name = 'my_snspd', width=5, length = 10, nanowire = 15), center = (6,0), rotation = 45)
wg1 = d.add_device(waveguide(name = 'important_wg', width=5, length = 10))

# Create another waveguide separately, then add it to new device 'd'
wg = waveguide(width=5, length = 10) # Should this return a DeviceReference?
wg2 = d.add_device(wg2, center = (7,0)) # This replaces wg2 with its DeviceReference
wg2.translate(dx = 0.5, dy = 1.7)

# Manipulate the 
snspd.translate(dx = 4, dy = 6)
snspd.translate(origin = [4,6], destination = [5,9]) # can either move from point to pint
snspd.rotate(rotation = 45)
wg1.translate(origin = [1,0], destination = [5,6]) # Calculates dx, dy automatically

# Add some new geometry
p1 = d.add_polygon([(0, 0), (2, 2), (2, 6), (-6, 6), (-6, -6), (-4, -4), (-4, 4), (0, 4)])
p2 = Polygon([(2, 2), (2, 6), (-6, 6)])
p1.fillet(0.5)
p2.translate()

# Add new ports to 'd' in a few different ways
d.add_port(name = 'integratedport1', midpoint = [3,4]) # Use the Device.add_port function
d.ports['integratedport2'] = Port(midpoint = [0,1])    # Make the Port() separately and attach it to the ports{} dict
d.add_port(snspd.ports['term2'])                      # Copy any existing port

# Connect ports together
snspd.move_port(source_port = 'term1', destination_port = wg1.ports[''])
snspd.move_port(port = 'term2', midpoint = [1,5]) # Can specify either name of port or object
d.connect_ports(snspd, wg1)



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



