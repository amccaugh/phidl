from __future__ import division, print_function, absolute_import
import numpy as np
import itertools
from numpy import sqrt, pi, cos, sin, log, exp, sinh, mod
from numpy.linalg import norm
from scipy.special import iv as besseli
from scipy.optimize import fmin, fminbound
from scipy import integrate

import gdspy
from phidl import Device, Port
from phidl.device_layout import _parse_layer

from skimage import draw, morphology


##### Categories:
# Polygons / shapes
# Optimal (current-crowding) curves
# Pads
# SNSPD
# Taper
# Text
# Wafer / Die
# Waveguide
# yTron
# Fill tool
# Routing
# Boolean functions



#==============================================================================
#
# Connectors
#
#==============================================================================

def connector(midpoint = (0,0), width = 1, orientation = 0):
    """ Creates a Device which has back-to-back ports """
    D = Device(name = 'connector')
    D.add_port(name = 1, midpoint = [midpoint[0]/2, midpoint[1]/2],  width = width, orientation = orientation)
    D.add_port(name = 2, midpoint = [midpoint[0]/2, midpoint[1]/2], width = width, orientation = orientation-180)
    return D



#==============================================================================
#
# Optimal current-crowding superconducting structures
#
#==============================================================================


def optimal_hairpin(width = 0.2, pitch = 0.6, length = 10, num_pts = 50, layer = 0):

    #==========================================================================
    #  Create the basic geometry
    #==========================================================================
    a = (pitch + width)/2
    y = -(pitch - width)/2
    x = -pitch
    dl = width/(num_pts*2)
    n = 0

    # Get points of ideal curve from conformal mapping
    # TODO This is an inefficient way of finding points that you need
    xpts = [x]; ypts = [y]
    while (y < 0) & (n<1e6):
        s = x + 1j*y
        w = np.sqrt(1 - np.exp(pi*s/a))
        wx = np.real(w); wy = np.imag(w)
        wx = wx/np.sqrt(wx**2+wy**2); wy = wy/np.sqrt(wx**2+wy**2)
        x = x + wx*dl; y = y + wy*dl
        xpts.append(x); ypts.append(y)
        n = n+1
    ypts[-1] = 0 # Set last point be on the x=0 axis for sake of cleanliness
    ds_factor = int(len(xpts)/num_pts) # Downsample the total number of points
    xpts = xpts[::-ds_factor]; xpts = xpts[::-1]    # This looks confusing, but it's just flipping the arrays around
    ypts = ypts[::-ds_factor]; ypts = ypts[::-1]    # so the last point is guaranteed to be included when downsampled

    # Add points for the rest of meander
    xpts.append(xpts[-1] + 4*width); ypts.append(0)
    xpts.append(xpts[-1]); ypts.append(-a)
    xpts.append(xpts[0]); ypts.append(-a)
    xpts.append(max(xpts)-length); ypts.append(-a)
    xpts.append(xpts[-1]); ypts.append(-a + width)
    xpts.append(xpts[0]); ypts.append(ypts[0])
    
    xpts = np.array(xpts)
    ypts = np.array(ypts)

    #==========================================================================
    #  Create a blank device, add the geometry, and define the ports
    #==========================================================================
    D = Device(name = 'hairpin')
    D.add_polygon([xpts,ypts])
    D.add_polygon([xpts,-ypts])
    
    xports = min(xpts)
    yports = -a + width/2
    D.add_port(name = 1, midpoint = [xports,-yports], width = width, orientation = 180)
    D.add_port(name = 2, midpoint = [xports,yports], width = width, orientation = 180)
    
    return D
    
    

    
# TODO Include parameter which specifies "half" (one edge flat) vs "full" (both edges curved)
# TODO Include parameter to make curve sub-optimal (gentler than optimal)
def optimal_step(start_width = 10, end_width = 22, num_pts = 50, width_tol = 1e-3, anticrowding_factor = 1.2):

    #==========================================================================
    #  Create the basic geometry
    #==========================================================================
    def step_points(eta, W, a):
        # Returns points from a unit semicircle in the w (= u + iv) plane to 
        # the optimal curve in the zeta (= x + iy) plane which transitions
        # a wire from a width of 'W' to a width of 'a'
        # eta takes value 0 to pi

        W = np.complex(W)
        a = np.complex(a)

        gamma = (a*a + W*W)/(a*a - W*W)

        w = np.exp(1j*eta)

        zeta = 4*1j/pi*(W*np.arctan(np.sqrt((w-gamma)/(gamma+1))) \
                           + a*np.arctan(np.sqrt((gamma-1)/(w-gamma))))

        x = np.real(zeta)
        y = np.imag(zeta)
        return x,y


    def invert_step_point(x_desired = -10, y_desired = None, W = 1, a = 2):
        # Finds the eta associated with the value x_desired along the optimal curve
        def fh(eta):
            guessed_x, guessed_y = step_points(eta, W = W, a = a)
            if y_desired is None:   return (guessed_x-x_desired)**2 # The error
            else:                   return (guessed_y-y_desired)**2

        found_eta = fminbound(fh, x1 = 0, x2 = pi, args=())
        return step_points(found_eta, W = W, a = a)

    if start_width > end_width:
        reverse = True
        start_width, end_width = end_width, start_width
    else:
        reverse = False
        
    xmin,ymin = invert_step_point(y_desired = start_width*(1+width_tol), W = start_width, a = end_width)
    xmax,ymax = invert_step_point(y_desired = end_width*(1-width_tol), W = start_width, a = end_width)
    
    xpts = np.linspace(xmin, xmax, num_pts).tolist()
    ypts = []
    for x in xpts:
        x,y = invert_step_point(x_desired = x, W = start_width, a = end_width)
        ypts.append(y)
    
    ypts[-1] = end_width
    ypts[0] =  start_width
    xpts.append(xpts[-1])
    ypts.append(0)
    xpts.append(xpts[0])
    ypts.append(0)
    
    # anticrowding_factor stretches the wire out; a stretched wire is a gentler
    # transition, so there's less chance of current crowding if the fabrication 
    # isn't perfect but as a result, the wire isn't as short as it could be
    xpts = (np.array(xpts)*anticrowding_factor).tolist()

    if reverse is True:
        xpts = (-np.array(xpts)).tolist()
        start_width, end_width = end_width, start_width

    #==========================================================================
    #  Create a blank device, add the geometry, and define the ports
    #==========================================================================
    D = Device(name = 'step')
    D.add_polygon([xpts,ypts])
    
    D.add_port(name = 1, midpoint = [min(xpts),start_width/2], width = start_width, orientation = 180)
    D.add_port(name = 2, midpoint = [max(xpts),end_width/2], width = end_width, orientation = 0)
    
    return D
    
    
    
    
    
#==============================================================================
# Example code
#==============================================================================
    
#hairpin = optimal_hairpin(width = 1, pitch = 3, length = 30, num_pts = 20)
#quickplot(hairpin)


#step = optimal_step(start_width = 5, end_width = 1, num_pts = 80, width_tol = 1e-3)
#quickplot(step)





#==============================================================================
#
# Contact pads
#
#==============================================================================


def compass(size = (4,2), layer = 0):
    """ Creates a rectangular contact pad with centered ports on edges of the
    rectangle (north, south, east, and west)
    """
    
    D = Device(name = 'compass')
    r = D.add_ref( rectangle(size, layer = layer) )
    r.center = (0,0)
    
    dx = size[0]
    dy = size[1]
    D.add_port(name = 'N', midpoint = [0, dy/2],  width = dx, orientation = 90)
    D.add_port(name = 'S', midpoint = [0, -dy/2], width = dx, orientation = -90)
    D.add_port(name = 'E', midpoint = [dx/2, 0],  width = dy, orientation = 0)
    D.add_port(name = 'W', midpoint = [-dx/2, 0], width = dy, orientation = 180)
    
    return D
    
    
def compass_multi(size = (4,2), ports = {'N':3,'S':4}, layer = 0):
    """ Creates a rectangular contact pad with multiple ports along the edges
    rectangle (north, south, east, and west).
    """
    
    D = Device(name = 'compass_multi')
    r = D.add_ref( rectangle(size, layer = layer) )
    r.center = (0,0)
    
    dx = size[0]/2
    dy = size[1]/2
    
    if 'N' in ports:
        num_ports = ports['N']
        m = dx-dx/num_ports
        p_list = np.linspace(-m, m, num_ports)
        [D.add_port(name = ('N%s' % (n+1)), midpoint = [p, dy],  width = dx/num_ports*2, orientation = 90) for n,p in enumerate(p_list)]
    if 'S' in ports:
        num_ports = ports['S']
        m = dx-dx/num_ports
        p_list = np.linspace(-m, m, num_ports)
        [D.add_port(name = ('S%s' % (n+1)), midpoint = [p, -dy],  width = dx/num_ports*2, orientation = -90) for n,p in enumerate(p_list)]
    if 'E' in ports:
        num_ports = ports['E']
        m = dy-dy/num_ports
        p_list = np.linspace(-m, m, num_ports)
        [D.add_port(name = ('E%s' % (n+1)), midpoint = [dx, p],  width = dy/num_ports*2, orientation = 0) for n,p in enumerate(p_list)]
    if 'W' in ports:
        num_ports = ports['W']
        m = dy-dy/num_ports
        p_list = np.linspace(-m, m, num_ports)
        [D.add_port(name = ('W%s' % (n+1)), midpoint = [-dx, p],  width = dy/num_ports*2, orientation = 180) for n,p in enumerate(p_list)]
    
    return D
    
    

# TODO: Fix the fillet here, right now only goes halfway down
def flagpole(size = (4,2), stub_size = (2,1), shape = 'p', taper_type = 'straight', layer = 0):
    f = np.array(size)
    p = np.array(stub_size)
    shape = shape.lower()

    assert shape in 'pqbd', '[DEVICE]  flagpole() shape must be p, q, b, or D'
    assert taper_type in ['straight','fillet'], '[DEVICE]  flagpole() taper_type must "straight" or "fillet" or None'
    
    if shape ==   'p':
        orientation = -90
    elif shape == 'q':
        f[0], p[0] = -size[0], -stub_size[0]
        orientation = -90
    elif shape == 'b':
        f[1], p[1] = -size[1], -stub_size[1]
        orientation = 90
    elif shape == 'D':
        f[1], p[1] = -size[1], -stub_size[1]
        f[0], p[0] = -size[0], -stub_size[0]
        orientation = 90
    xpts = [0, 0, f[0], f[0], p[0], p[0], 0]
    ypts = [0, f[1], f[1], 0, 0, -p[1], -p[1]]
    
    D = Device(name = 'flagpole')
    pad_poly = D.add_polygon([xpts,ypts], layer = layer)
    if taper_type == 'fillet':
        taper_amount = min([abs(f[0]-p[0]), abs(p[1])])
        pad_poly.fillet([0,0,0,0,taper_amount,0,0])
    elif taper_type == 'straight':
        D.add_polygon([xpts[3:6],ypts[3:6]], layer = layer)
            
    D.add_port(name = 1, midpoint = [p[0]/2, -p[1]],  width = abs(p[0]), orientation = orientation)
    D.add_port(name = 2, midpoint = [f[0]/2, f[1]],  width = abs(f[0]), orientation = orientation-180)
    return D


def tee(size = (4,2), stub_size = (2,1), taper_type = 'straight', layer = 0):
    f = np.array(size)
    p = np.array(stub_size)
    
    xpts = np.array([f[0], f[0], p[0], p[0], -p[0], -p[0], -f[0], -f[0]])/2
    ypts = [f[1], 0, 0, -p[1], -p[1], 0, 0, f[1]]
    
    D = Device(name = 'tee')
    pad_poly = D.add_polygon([xpts,ypts], layer = layer)
    if taper_type == 'fillet':
        taper_amount = min([abs(f[0]-p[0]), abs(p[1])])
        pad_poly.fillet([0,0,taper_amount,0,0,taper_amount,0,0])
    if taper_type == 'straight':
        taper_poly1 = D.add_polygon([xpts[1:4],ypts[1:4]], layer = layer)
        taper_poly2 = D.add_polygon([xpts[4:7],ypts[4:7]], layer = layer)
        
    D.add_port(name = 'N', midpoint = [0, f[1]],  width = f[0], orientation = 90)
    D.add_port(name = 'S', midpoint = [0, -p[1]],  width = p[0], orientation = -90)
    D.add_port(name = 'E', midpoint = [f[0]/2, f[1]/2],  width = f[1], orientation = 0)
    D.add_port(name = 'W', midpoint = [-f[0]/2, f[1]/2],  width = f[1], orientation = 180)
    return D
    

#==============================================================================
# Example code
#==============================================================================

#cp = compass(size = [4,2])
#quickplot(cp)


#cpm = compass_multi(size = [40,20], ports = {'N':3,'S':4, 'E':1, 'W':8}, layer = 0)
#quickplot(cpm)


#cpm = compass_multi(size = [40,20], ports = {'N':3,'S':4, 'E':1, 'W':8}, layer = 0)
#inset_polygon = inset(cpm, distance = 2, layer = 1)
#cpm.add(inset_polygon)
#quickplot(cpm)

#fp = flagpole(size = [4,2], stub_size = [2,1], shape = 'p', taper_type = 'straight', layer = 0)
#quickplot(fp)


#tp = tee(size = [4,2], stub_size = [2,1], taper_type = 'fillet', layer = 0)
#quickplot(tp)



#==============================================================================
#
# Polygons / Shapes
#
#==============================================================================



def rectangle(size = (4,2), layer = 0):
    D = Device(name = 'rectangle')
    points = [[size[0], size[1]], [size[0], 0], [0, 0], [0, size[1]]]
    D.add_polygon(points, layer = layer)
    return D


def ring(radius = 10, width = 0.5, angle_resolution = 2.5, layer = 0):
    D = Device(name = 'ring')
    inner_radius = radius - width/2
    outer_radius = radius + width/2
    t = np.linspace(0, 360, np.ceil(360/angle_resolution))*pi/180
    inner_points_x = (inner_radius*cos(t)).tolist()
    inner_points_y = (inner_radius*sin(t)).tolist()
    outer_points_x = (outer_radius*cos(t)).tolist()
    outer_points_y = (outer_radius*sin(t)).tolist()
    xpts = inner_points_x + outer_points_x[::-1]
    ypts = inner_points_y + outer_points_y[::-1]
    D.add_polygon(points = (xpts,ypts), layer = layer)
    return D
    
    
def arc(radius = 10, width = 0.5, theta = 45, start_angle = 0, angle_resolution = 2.5, layer = 0):
    """ Creates an arc of arclength ``theta`` starting at angle ``start_angle`` """
    inner_radius = radius-width/2
    outer_radius = radius+width/2
    angle1 = (start_angle)*pi/180
    angle2 = (start_angle + theta)*pi/180
    t = np.linspace(angle1, angle2, np.ceil(abs(theta)/angle_resolution))
    inner_points_x = (inner_radius*cos(t)).tolist()
    inner_points_y = (inner_radius*sin(t)).tolist()
    outer_points_x = (outer_radius*cos(t)).tolist()
    outer_points_y = (outer_radius*sin(t)).tolist()
    xpts = inner_points_x + outer_points_x[::-1]
    ypts = inner_points_y + outer_points_y[::-1]
    
    D = Device('arc')
    D.add_polygon(points = (xpts,ypts), layer = layer)
    D.add_port(name = 1, midpoint = (radius*cos(angle1), radius*sin(angle1)),  width = width, orientation = start_angle - 90 + 180*(theta<0))
    D.add_port(name = 2, midpoint = (radius*cos(angle2), radius*sin(angle2)),  width = width, orientation = start_angle + theta + 90 - 180*(theta<0))
    D.meta['length'] = (abs(theta)*pi/180)*radius
    return D


def turn(port, radius = 10, angle = 270, angle_resolution = 2.5, layer = 0):
    """ Starting form a port, create a arc which connects to the port """
    D = arc(radius = radius, width = port.width, theta = angle, start_angle = 0, 
            angle_resolution = angle_resolution, layer = layer)
    D.rotate(angle =  180 + port.orientation - D.ports[1].orientation, center = D.ports[1].midpoint)
    D.move(origin = D.ports[1], destination = port)
    return D


#==============================================================================
# Example code
#==============================================================================
    
#R = rectangle(size = (4,2), layer = 0)
#quickplot(R)



#==============================================================================
#
# SNSPD
#
#==============================================================================




def snspd(wire_width = 0.2, wire_pitch = 0.6, size = (3,3),
          terminals_same_side = False, layer = 0):
    xsize = size[0]
    ysize = size[1]
    num_meanders = int(ysize/wire_pitch)
    if terminals_same_side: num_meanders += np.mod(num_meanders,2) # Make number of meanders even
    
    D = Device(name = 'snspd')
    hairpin = optimal_hairpin(width = wire_width, pitch = wire_pitch, length = xsize/2, num_pts = 20)
    
    hp2 = D.add_ref(hairpin)
    top_port = hp2.ports[1]
    while num_meanders > 1:
        # Repeatedly add two new device references
        hp1 = D.add_ref(hairpin)
        hp1.rotate(180)
        hp1.connect(2, hp2.ports[2])
        hp2 = D.add_ref(hairpin)
        hp2.connect(1, hp1.ports[1])
        
        num_meanders -= 2
        
    bottom_port = hp2.ports[2]
    
    if terminals_same_side is False:
        hp1 = D.add_ref(hairpin)
        hp1.rotate(180)
        hp1.connect(2, hp2.ports[2])
        bottom_port = hp1.ports[1]
    
    
    c_nw = D.add_ref(compass(size = [xsize/2 ,wire_width]))
    c_se = D.add_ref(compass(size = [xsize/2 ,wire_width]))
    c_nw.connect('E', top_port)
    c_se.connect('E', bottom_port)
    
    D.add_port(port = c_nw.ports['W'], name = 1)
    D.add_port(port = c_se.ports['W'], name = 2)
    
    D.meta['num_squares'] = (int(ysize/wire_pitch)*xsize/2)/wire_width
    
    return D

    
def snspd_expanded(wire_width = 0.2, wire_pitch = 0.6, size = (3,3), connector_width = 1,
           num_pts = 20, terminals_same_side = False, layer = 0):
    """ Creates an optimally-rounded SNSPD with wires coming out of it that expand"""
    D = Device('snspd_expanded')
    s = D.add_ref(snspd(wire_width = wire_width, wire_pitch = wire_pitch, size = size,
                     terminals_same_side = terminals_same_side, layer = layer))
    step_device = optimal_step(start_width = wire_width, end_width = connector_width, num_pts = 100, anticrowding_factor = 2, width_tol = 1e-3)
    step1 = D.add_ref(step_device)
    step2 = D.add_ref(step_device)
    step1.connect(port = 1, destination = s.ports[1])
    step2.connect(port = 1, destination = s.ports[2])
    D.add_port(name = 1, port = step1.ports[2])
    D.add_port(name = 2, port = step2.ports[2])
    
    D.meta = s.meta
    
    return D
    
    

def snspd_squares(wire_width = 0.2, wire_pitch = 0.6, size = (3,3), num_pts = 20,
                  terminals_same_side = False, layer = 0):
    pass
    
#==============================================================================
# Example code
#==============================================================================
    
#s = snspd(wire_width = 0.2, wire_pitch = 0.6, size = [10,3], num_pts = 20, terminals_same_side = True)
#quickplot(s)


#step = optimal_step(start_width = 10, end_width = 1, num_pts = 50, width_tol = 1e-3)
#quickplot(step)




#==============================================================================
#
# Tapers
#
#==============================================================================


def taper(length = 10, width1 = 5, width2 = 8, port = None, layer = 0):
    if type(port) is Port and width1 is None: width1 = port.width
    if width2 is None: width2 = width1
    xpts = [0, length, length, 0]
    ypts = [width1/2, width2/2, -width2/2, -width1/2]
    
    D = Device('taper')
    D.add_polygon([xpts,ypts], layer = layer)
    D.add_port(name = 1, midpoint = [0, 0], width = width1, orientation = 180)
    D.add_port(name = 2, midpoint = [length, 0], width = width2, orientation = 0)
    if type(port) is Port: 
        D.rotate(angle = port.orientation, center = [0,0])
        D.move(origin = [0,0], destination = port.midpoint)
    return D
    
    
def ramp(length = 10, width1 = 5, width2 = 8, layer = 0):
    if width2 is None: width2 = width1
    xpts = [0, length, length, 0]
    ypts = [width1, width2, 0, 0]
    D = Device('ramp')
    D.add_polygon([xpts,ypts], layer = layer)
    D.add_port(name = 1, midpoint = [0, width1/2], width = width1, orientation = 180)
    D.add_port(name = 2, midpoint = [length, width2/2], width = width2, orientation = 0)
    return D
    

# Equations taken from
# Hammerstad, E., & Jensen, O. (1980). Accurate Models for Microstrip
# Computer-Aided Design.  http://doi.org/10.1109/MWSYM.1980.1124303
def _microstrip_Z(wire_width, dielectric_thickness, eps_r):
    # Note these equations can be further corrected for thick films (Hammersted Eqs 6-9)
    # and also for frequency since microstrips are dispersive  (Hammersted Eqs 10-12)

    u = wire_width/dielectric_thickness
    eta = 376.73 # Vacuum impedance
    
    a = 1 + log((u**4 + (u/52)**2)/(u**4 + 0.432))/49 + log(1 + (u/18.1)**3)/18.7;
    b = 0.564*((eps_r-0.9)/(eps_r+3))**0.053;
    F = 6 + (2*pi-6)*exp(-(30.666/u)**0.7528);
    eps_eff = 0.5*(eps_r+1) + 0.5*(eps_r-1)*(1 + 10/u)**(-a*b);
    Z = eta/(2*pi) * log(F/u + sqrt(1+(2/u)**2)) /sqrt(eps_eff);
    return Z,eps_eff


def _microstrip_LC_per_meter(wire_width, dielectric_thickness, eps_r):
    # Use the fact that v = 1/sqrt(L_m*C_m) = 1/sqrt(eps*mu) and
    # Z = sqrt(L_m/C_m)   [Where L_m is inductance per meter]

    Z, eps_eff =  _microstrip_Z(wire_width, dielectric_thickness, eps_r)
    eps0 =  8.854e-12
    mu0 = 4*pi*1e-7
    
    eps = eps_eff*eps0
    mu = mu0
    L_m = sqrt(eps*mu)*Z
    C_m = sqrt(eps*mu)/Z
    return L_m, C_m


def _microstrip_Z_with_Lk(wire_width, dielectric_thickness, eps_r, Lk_per_sq):
    # Add a kinetic inductance and recalculate the impedance, be careful
    # to input Lk as a per-meter inductance

    L_m, C_m = _microstrip_LC_per_meter(wire_width, dielectric_thickness, eps_r)
    Lk_m = Lk_per_sq*(1.0/wire_width)
    Z = sqrt((L_m+Lk_m)/C_m)
    return Z
    
def _microstrip_v_with_Lk(wire_width, dielectric_thickness, eps_r, Lk_per_sq):
    L_m, C_m = _microstrip_LC_per_meter(wire_width, dielectric_thickness, eps_r)
    Lk_m = Lk_per_sq*(1.0/wire_width)
    v = 1/sqrt((L_m+Lk_m)*C_m)
    return v
    
def _find_microstrip_wire_width(Z_target, dielectric_thickness, eps_r, Lk_per_sq):
    
    def error_fun(wire_width):
        Z_guessed = _microstrip_Z_with_Lk(wire_width, dielectric_thickness, eps_r, Lk_per_sq)
        return (Z_guessed-Z_target)**2 # The error
    
    x0 = dielectric_thickness
    w = fmin(error_fun, x0, args=(), disp=False)
    return w[0]

def _G_integrand(xip, B):
    return besseli(0, B*sqrt(1-xip**2))


def _G(xi, B):
    return B/sinh(B)*integrate.quad(_G_integrand, 0, xi, args = (B))[0]


def hecken_taper(length = 200, B = 4.0091, dielectric_thickness = 0.25, eps_r = 2,
                 Lk_per_sq = 250e-12, Z1 = None, Z2 = None, width1 = None, width2 = None,
                 num_pts = 100, layer = 0):
    if width1 is not None:  Z1 = _microstrip_Z_with_Lk(width1*1e-6, dielectric_thickness*1e-6, eps_r, Lk_per_sq)
    if width2 is not None:  Z2 = _microstrip_Z_with_Lk(width2*1e-6, dielectric_thickness*1e-6, eps_r, Lk_per_sq)
    xi_list = np.linspace(-1,1, num_pts) # Normalized length of the wire [-1 to +1]
    Z = [np.exp( 0.5*log(Z1*Z2) + 0.5*log(Z2/Z1)*_G(xi, B) ) for xi in xi_list]
    widths = np.array([_find_microstrip_wire_width(z, dielectric_thickness*1e-6, eps_r, Lk_per_sq)*1e6 for z in Z])
    x = ((xi_list/2)*length)
    
    # TODO: Compensate for varying speed of light in the microstrip
    # by shortening and lengthening sections according to the speed of light
    # in that section
    v = np.array([_microstrip_v_with_Lk(w*1e-6, dielectric_thickness*1e-6, eps_r, Lk_per_sq) for w in widths])
    
    # Create blank device and add taper polygon
    D = Device()
    xpts = np.concatenate([x, x[::-1]])
    ypts = np.concatenate([widths/2, -widths[::-1]/2])
    D.add_polygon((xpts,ypts), layer = layer)
    D.add_port(name = 1, midpoint = (-length/2,0), width = widths[0], orientation = 180)
    D.add_port(name = 2, midpoint = (length/2,0), width = widths[-1], orientation = 0)
    
    # Add meta information about the taper
    dx = x[1]-x[0]
    D.meta['num_squares'] = np.sum(dx/widths)
    D.meta['width1'] = widths[0]
    D.meta['width2'] = widths[-1]
    D.meta['Z1'] = Z[0]
    D.meta['Z2'] = Z[-1]
    # Note there are two values for v/c (and f_cutoff) because the speed of
    # light is different at the beginning and end of the taper
    D.meta['w'] = widths
    D.meta['x'] = x
    D.meta['Z'] = Z
    D.meta['v/c'] = v/3e8
    BetaLmin = np.sqrt(B**2 + 6.523)
    D.meta['f_cutoff'] = BetaLmin*D.meta['v/c'][0]*3e8/(2*pi*length*1e-6)
    
    return D



def meander_taper(x_taper, w_taper, meander_length = 1000, spacing_factor = 3, min_spacing = 0.5):
    
    def taper_width(x):
        return np.interp(x, x_taper, w_taper)
        
        
    def taper_section(x_start, x_end, num_pts = 30):
        D = Device()
        length =  x_end - x_start
        x = np.linspace(0, length, num_pts)
        widths = np.linspace(taper_width(x_start), taper_width(x_end), num_pts)
        xpts = np.concatenate([x, x[::-1]])
        ypts = np.concatenate([widths/2, -widths[::-1]/2])
        D.add_polygon((xpts,ypts), layer = 0)
        D.add_port(name = 1, midpoint = (0,0), width = widths[0], orientation = 180)
        D.add_port(name = 2, midpoint = (length,0), width = widths[-1], orientation = 0)
        return D
        
    def arc_tapered(radius = 10, width1 = 1, width2 = 2, theta = 45, angle_resolution = 2.5, layer = 0):
        D = Device()
        path1 = gdspy.Path(width = width1, initial_point = (0, 0))
        path1.turn(radius = radius, angle = theta*pi/180, number_of_points=int(abs(2*theta/angle_resolution)), final_width = width2)
        [D.add_polygon(p.points, layer = layer) for p in path1.polygons]
        D.add_port(name = 1, midpoint = (0, 0), width = width1, orientation = 180)
        D.add_port(name = 2, midpoint = (path1.x, path1.y), width = width2, orientation = path1.direction*180/pi)
        return D
        
    D = Device('meander-taper')
    xpos1 = min(x_taper)
    xpos2 = min(x_taper) + meander_length
    t = D.add_ref( taper_section(x_start = xpos1, x_end = xpos2, num_pts = 50) )
    D.add_port(t.ports[1])
    dir_toggle = -1
    while xpos2 < max(x_taper):
        arc_width1 = taper_width(xpos2)
        arc_radius = max(spacing_factor*arc_width1, min_spacing)
        arc_length = pi*arc_radius
        arc_width2 = taper_width(xpos2 + arc_length)
        a = D.add_ref(  arc_tapered(radius = arc_radius, width1 = arc_width1, width2 = arc_width2, theta = 180*dir_toggle) )
        a.connect(port = 1, destination = t.ports[2])
        dir_toggle = -dir_toggle
        xpos1 = xpos2 + arc_length
        xpos2 = xpos1 + meander_length
        t = D.add_ref( taper_section(x_start = xpos1, x_end = xpos2, num_pts = 30) )
        t.connect(port = 1, destination = a.ports[2])
    D.add_port(t.ports[2])
        
    return D
    
#==============================================================================
# Example code
#==============================================================================

#D = racetrack_gradual(width, R = 5, N=3)
#quickplot(D)

# D = hecken_taper(length = 200, B = 4.0091, dielectric_thickness = 0.25, eps_r = 2,
#                  Lk_per_sq = 250e-12, Z1 = 50, width2 = 0.3,
#                  num_pts = 100, layer = 0)
# quickplot(D)

#t = np.linspace(0,1)
#x,y = _racetrack_gradual_parametric(t, R = 5, N = 3)
#plt.plot(x,y)
#plt.axis('equal')





#==============================================================================
#
# Text
#
#==============================================================================

# The DEPLOF font is made by David Elata, MEMS Lab, Technion, Haifa, Israel and
# and is used with permission.  The raw polygon entries are sourced from
# Ulf Griesmann's modified DEPLOF implementation used in the MATLAB gdsii 
# toolbox https://github.com/ulfgri/gdsii-toolbox/ and are used with permission


glyph = {}
width = {}
indent = {}

glyph[33] = [[[100,-100],[100,100],[300,100],[300,-100],[100,-100]], [[100,250],[100,1100],[ 300,1100],[300,250],[100,250]]]
glyph[34] = [[[300,800],[300,1200],[500,1200],[500,1000],[300,800]], [[600,800],[600,1200],[ 800,1200],[800,1000],[600,800]]]
glyph[35] = [[[150,0],[170,200],[50,200],[50,400],[190,400],[210,600],[100,600],[100,800],[230,800],[ 250,1000],[450,1000],[390,400],[530,400],[510,200],[370,200],[350,0],[150,0]],  [[550,0],[610,600],[470,600],[490,800],[630,800],[650,1000],[850,1000],[830,800],[950,800],[ 950,600],[810,600],[790,400],[900,400],[900,200],[770,200],[750,0],[550,0]]]
glyph[36] = [[[400,1000],[400,1200],[600,1200],[600,1000],[800,1000],[900,900],[900,800],[300,800],[ 300,600],[400,600],[400,700],[600,700],[600,600],[800,600],[900,500],[900,100],[ 800,0],[600,0],[600,-200],[400,-200],[400,0],[200,0],[100,100],[100,200],[700,200],[ 700,400],[600,400],[600,300],[400,300],[400,400],[200,400],[100,500],[100,900],[ 200,1000],[400,1000]]] 
glyph[37] = [[[100,100],[800,1000],[900,900],[200,0],[100,100]],  [[100,900],[400,900],[400,650],[350,600],[200,600],[300,700],[300,800],[200,800],[ 200,700],[100,600],[100,900]], [[650,400],[800,400],[700,300],[700,200],[ 800,200],[800,300],[900,400],[900,100],[600,100],[600,350],[650,400]]]
glyph[38] = [[[700,0],[100,600],[100,800],[200,900],[400,900],[500,800],[500,800],[500,600],[ 450,550],[350,650],[400,700],[300,800],[200,700],[600,300],[700,400],[800,300],[ 700,200],[900,0],[700,0]],  [[550,50],[500,0],[100,0],[0,100],[0,300],[100,400],[150,450],[250,350],[100,200],[ 100,150],[150,100],[400,100],[450,150],[550,50]]] 
glyph[39] = [[[300,800],[300,1200],[500,1200],[500,1000],[300,800]]]
glyph[40] = [[[100,500],[125,700],[175,900],[250,1100],[450,1100],[375,900],[325,700],[300,500],[ 325,300],[375,100],[450,-100],[250,-100],[175,100],[125,300],[100,500]]]
glyph[41] = [[[100,1100],[300,1100],[375,900],[425,700],[450,500],[425,300],[375,100],[ 300,-100],[100,-100],[175,100],[225,300],[250,500],[225,700],[175,900],[100,1100]]]
glyph[42] = [[[450,750],[450,1000],[550,1000],[550,750],[800,750],[800,650],[550,650],[550,400],[ 450,400],[450,650],[200,650],[200,750],[450,750]],  [[350,850],[250,850],[200,900],[200,1000],[300,1000],[350,950],[350,850]],  [[650,850],[650,950],[700,1000],[800,1000],[800,900],[750,850],[650,850]],  [[650,550],[750,550],[800,500],[800,400],[700,400],[650,450],[650,550]],  [[350,550],[350,450],[300,400],[200,400],[200,500],[250,550],[350,550]]]
# +
glyph[43] = [[[400,600],[400,900],[600,900],[600,600],[900,600],[900,400],[600,400],[600,100],[ 400,100],[400,400],[100,400],[100,600],[400,600]]]
glyph[44] = [[[300,200],[300,0],[100,-200],[100,200],[300,200]]]
glyph[45] = [[[900,550],[900,350],[100,350],[100,550],[900,550]]]
glyph[46] = [[[300,200],[300,0],[100,0],[100,200],[300,200]]]
glyph[47] = [[[500,1200],[300,-200],[100,-200],[300,1200],[500,1200]]]
# 0
glyph[48] = [[[400,800],[300,700],[300,300],[400,200],[350,0],[300,0],[100,200],[100,800],[ 300,1000],[530,1000],[480,800],[400,800]],  [[600,200],[700,300],[700,700],[600,800],[650,1000],[700,1000],[900,800],[ 900,200],[700,0],[470,0],[520,200],[600,200]]]
glyph[49] = [[[200,600],[100,600],[100,800],[300,1000],[400,1000],[400,200],[500,200],[500,0],[ 100,0],[100,200],[200,200],[200,600],[200,600]]]
glyph[50] = [[[100,900],[200,1000],[700,1000],[800,900],[800,600],[400,200],[800,200],[800,0],[ 100,0],[100,200],[600,700],[600,800],[300,800],[300,700],[100,700],[100,900]]]
glyph[51] = [[[600,1000],[800,800],[800,600],[700,500],[800,400],[800,200],[600,0],[300,0],[ 100,200],[100,300],[300,300],[400,200],[500,200],[600,300],[500,400],[500,600],[ 600,700],[500,800],[400,800],[300,700],[100,700],[100,800],[300,1000],[600,1000]]]
glyph[52] = [[[800,500],[800,300],[700,300],[700,0],[500,0],[500,300],[100,300],[100,1000],[300,1000],[ 300,500],[500,500],[500,1000],[700,1000],[700,500],[800,500]]]
glyph[53] = [[[800,800],[300,800],[300,600],[700,600],[800,500],[800,100],[700,0],[200,0],[100,100],[ 100,300],[300,300],[300,200],[600,200],[600,400],[200,400],[100,500],[100,1000],[ 800,1000],[800,800]]]
glyph[54] = [[[800,700],[600,700],[600,800],[300,800],[300,600],[700,600],[800,500],[800,100],[ 700,0],[500,0],[500,200],[600,200],[600,400],[300,400],[300,200],[400,200],[400,0],[ 200,0],[100,100],[100,900],[200,1000],[700,1000],[800,900],[800,700]]]
glyph[55] = [[[560,800],[100,800],[100,1000],[800,1000],[600,0],[400,0],[480,400],[520,600],[560,800]]]
glyph[56] = [[[400,800],[300,800],[300,600],[600,600],[600,800],[500,800],[500,1000],[ 700,1000],[800,900],[800,600],[700,500],[800,400],[800,100],[700,0],[ 500,0],[500,200],[600,200],[600,400],[300,400],[300,200],[400,200],[400,0],[ 200,0],[100,100],[100,400],[200,500],[100,600],[100,900],[200,1000],[400,1000],[400,800]]]
# 9
glyph[57] = [[[100,300],[300,300],[300,200],[600,200],[600,400],[200,400],[100,500],[100,900],[ 200,1000],[400,1000],[400,800],[300,800],[300,600],[600,600],[600,800],[ 500,800],[500,1000],[700,1000],[800,900],[800,100],[700,0],[200,0],[100,100],[100,300]]]
glyph[58] = [[[300,200],[300,0],[100,0],[100,200],[300,200]], [[300,600],[300,400],[100,400],[100,600],[300,600]]]
glyph[59] = [[[300,200],[300,0],[100,-200],[100,200],[300,200]], [[300,600],[300,400],[100,400],[100,600],[300,600]]]
glyph[60] = [[[700,900],[700,700],[400,500],[700,300],[700,100],[100,500],[700,900]]]
glyph[61] = [[[100,400],[900,400],[900,200],[100,200],[100,400]], [[100,800],[900,800],[900,600],[100,600],[100,800]]]
glyph[62] = [[[700,500],[100,100],[100,300],[400,500],[100,700],[100,900],[700,500]]]
glyph[63] = [[[100,1000],[200,1100],[800,1100],[900,1000],[900,500],[800,400],[600,400],[ 600,200],[400,200],[400,500],[500,600],[700,600],[700,900],[300,900],[300,800],[100,800],[100,1000]],  [[600,50],[600,-150],[400,-150],[400,50],[600,50]]]
glyph[64] = [[[900,200],[900,100],[800,0],[300,0],[100,200],[100,800],[300,1000],[700,1000],[ 900,800],[900,500],[800,400],[450,400],[400,450],[400,600],[450,700],[600,700],[ 550,600],[550,500],[700,500],[700,700],[600,800],[400,800],[300,700],[300,300],[ 400,200],[900,200]]]
# A
glyph[65] = [[[100,800],[300,1000],[601,1000],[800,800],[800,0],[601,0],[601,200],[500,200],[ 500,400],[601,400],[601,700],[500,800],[400,800],[300,700],[300,400],[400,400],[ 400,200],[300,200],[300,0],[99,0],[100,800]]]
glyph[66] = [[[600,1000],[800,800],[800,600],[700,500],[800,400],[800,200],[600,0],[100,0],[100,400],[ 300,400],[300,200],[500,200],[600,300],[500,400],[420,400],[420,600],[500,600],[ 600,700],[500,800],[300,800],[300,600],[100,600],[100,1000],[600,1000]]]
glyph[67] = [[[300,0],[100,200],[100,800],[300,1000],[600,1000],[800,800],[800,600],[600,600],[ 600,700],[500,800],[400,800],[300,700],[300,300],[400,200],[500,200],[600,300],[ 600,400],[800,400],[800,200],[600,0],[300,0]]]
glyph[68] = [[[100,0],[100,400],[300,400],[300,200],[500,200],[600,300],[600,700],[500,800],[ 300,800],[300,600],[100,600],[100,1000],[600,1000],[800,800],[800,200],[600,0],[100,0]]]
glyph[69] = [[[700,1000],[700,800],[300,800],[300,600],[500,600],[500,400],[300,400],[300,200],[ 700,200],[700,0],[100,0],[100,1000],[700,1000]]]
glyph[70] = [[[100,0],[100,1000],[700,1000],[700,800],[300,800],[300,600],[500,600],[500,400],[300,400],[300,0],[100,0]]]
glyph[71] = [[[300,0],[100,200],[100,800],[300,1000],[600,1000],[800,800],[800,700],[600,700],[500,800],[ 400,800],[300,700],[300,300],[400,200],[600,200],[600,300],[500,300],[500,500],[800,500],[ 800,100],[700,0],[300,0]]]
glyph[72] = [[[100,1000],[300,1000],[300,600],[600,600],[600,1000],[800,1000],[800,0],[600,0],[600,400],[ 300,400],[300,0],[100,0],[100,1000]]]
glyph[73] = [[[100,0],[100,200],[300,200],[300,800],[100,800],[100,1000],[700,1000],[700,800],[500,800],[ 500,200],[700,200],[700,0],[100,0]]]
glyph[74] = [[[300,200],[500,200],[500,1000],[700,1000],[700,100],[600,0],[200,0],[100,100],[100,300],[ 300,300],[300,200]]]
glyph[75] = [[[100,1000],[300,1000],[300,600],[600,1000],[800,1000],[800,900],[500,500],[800,100],[ 800,0],[600,0],[300,400],[300,0],[100,0],[100,1000]]]
glyph[76] = [[[100,1000],[300,1000],[300,200],[800,200],[800,0],[100,0],[100,1000]]]
glyph[77] = [[[100,1000],[300,1000],[500,700],[700,1000],[900,1000],[900,0],[700,0],[700,600],[500,300],[ 300,600],[300,0],[100,0],[100,1000]]]
glyph[78] = [[[100,1000],[300,1000],[700,400],[700,1000],[900,1000],[900,0],[700,0],[300,600],[300,0],[100,0],[100,1000]]]
glyph[79] = [[[100,800],[300,1000],[430,1000],[430,800],[400,800],[300,700],[300,300],[400,200],[600,200],[ 700,300],[700,700],[600,800],[570,800],[570,1000],[700,1000],[900,800],[900,200],[700,0],[ 300,0],[100,200],[100,800]]]
glyph[80] = [[[100,1000],[700,1000],[900,800],[900,600],[700,400],[500,400],[500,600],[600,600],[700,700],[ 600,800],[300,800],[300,0],[100,0],[100,1000]]]
glyph[81] = [[[100,800],[300,1000],[700,1000],[900,800],[900,200],[800,100],[900,0],[600,0],[600,400],[ 700,400],[700,700],[600,800],[400,800],[300,700],[300,300],[400,200],[400,0],[300,0],[ 100,200],[100,800]]]
glyph[82] = [[[100,1000],[700,1000],[900,800],[900,600],[700,400],[900,200],[900,0],[700,0],[700,100],[ 500,300],[500,600],[600,600],[700,700],[600,800],[300,800],[300,0],[100,0],[100,1000]]]
glyph[83] = [[[900,800],[300,800],[300,600],[800,600],[900,500],[900,100],[800,0],[200,0],[100,100],[ 100,200],[700,200],[700,400],[200,400],[100,500],[100,900],[200,1000],[800,1000],[900,900],[900,800]]]
glyph[84] = [[[900,1000],[900,800],[600,800],[600,0],[400,0],[400,800],[100,800],[100,1000],[900,1000]]]
glyph[85] = [[[300,1000],[300,300],[400,200],[500,200],[600,300],[600,1000],[800,1000],[800,200],[ 600,0],[300,0],[100,200],[100,1000],[300,1000]]]
glyph[86] = [[[300,1000],[500,400],[700,1000],[900,1000],[600,0],[400,0],[100,1000],[300,1000]]]
glyph[87] = [[[100,1000],[300,1000],[300,400],[500,700],[700,400],[700,1000],[900,1000],[900,0],[ 700,0],[500,300],[300,0],[100,0],[100,1000]]]
glyph[88] = [[[367,500],[100,900],[100,1000],[300,1000],[500,700],[700,1000],[900,1000],[ 900,900],[633,500],[900,100],[900,0],[700,0],[500,300],[300,0],[100,0],[100,100],[367,500]]]
glyph[89] = [[[600,450],[600,0],[400,0],[400,450],[100,900],[100,1000],[300,1000],[500,700],[ 700,1000],[900,1000],[900,900],[600,450]]]
# Z
glyph[90] = [[[100,1000],[900,1000],[900,700],[300,200],[900,200],[900,0],[100,0],[100,300],[ 700,800],[100,800],[100,1000]]]
glyph[91] = [[[400,1200],[400,1000],[300,1000],[300,0],[400,0],[400,-200],[100,-200],[100,1200],[400,1200]]]
glyph[92] = [[[300,1200],[500,-200],[300,-200],[100,1200],[300,1200]]]
glyph[93] = [[[400,1200],[400,-200],[100,-200],[100,0],[200,0],[200,1000],[100,1000],[100,1200],[400,1200]]]
glyph[94] = [[[0,500],[400,900],[800,500],[600,500],[400,700],[200,500],[0,500]]]
glyph[95] = [[[100,200],[900,200],[900,0],[100,0],[100,200]]]
glyph[96] = [[[300,1000],[300,1200],[500,1200],[500,800],[300,1000]]]
# a
glyph[97] = [[[800,0],[300,0],[100,200],[100,500],[334,700],[600,700],[600,775],[800,775],[800,400],[ 600,400],[600,500],[400,500],[300,400],[300,300],[400,200],[600,200],[600,300],[800,300],[800,0]]]
glyph[98] = [[[100,300],[300,300],[300,200],[500,200],[600,300],[600,400],[500,500],[300,500],[300,400],[ 100,400],[100,1000],[300,1000],[300,700],[600,700],[800,500],[800,200],[600,0],[100,0],[100,300]]]
glyph[99] = [[[800,200],[600,0],[300,0],[100,200],[100,500],[300,700],[600,700],[800,500],[800,400],[ 600,400],[500,500],[400,500],[300,400],[300,300],[400,200],[500,200],[600,300],[800,300],[800,200]]]
glyph[100] = [[[800,0],[300,0],[100,200],[100,500],[300,700],[600,700],[600,1000],[800,1000],[800,400],[ 600,400],[600,500],[400,500],[300,400],[300,300],[400,200],[600,200],[600,300],[800,300],[800,0]]]
glyph[101] = [[[200,0],[100,100],[100,700],[200,800],[700,800],[800,700],[800,400],[700,300],[440,300],[ 440,500],[600,500],[600,600],[300,600],[300,200],[800,200],[800,100],[700,0],[200,0]]]
glyph[102] = [[[600,800],[300,800],[300,600],[500,600],[500,400],[300,400],[300,0],[100,0],[100,900],[ 200,1000],[600,1000],[600,800]]]
glyph[103] = [[[800,400],[600,400],[600,500],[400,500],[300,400],[300,300],[400,200],[600,200],[ 600,300],[800,300],[800,-200],[700,-300],[300,-300],[200,-200],[100,-100],[ 600,-100],[600,0],[334,0],[100,200],[100,500],[300,700],[800,700],[800,400]]]
glyph[104] = [[[600,0],[600,400],[500,500],[400,500],[300,400],[300,0],[100,0],[100,1100],[300,1100],[ 300,600],[400,700],[600,700],[800,500],[800,0],[600,0]]]
glyph[105] = [[[100,0],[100,600],[300,600],[300,0],[100,0]],  [[300,1000],[300,800],[100,800],[100,1000],[300,1000]]]
glyph[106] = [[[100,-100],[100,0],[300,0],[300,600],[500,600],[500,-100],[400,-200],[200,-200],[100,-100]],  [[500,1000],[500,800],[300,800],[300,1000],[500,1000]]]
glyph[107] = [[[300,500],[600,700],[800,700],[800,600],[500,400],[800,100],[800,0],[600,0],[ 300,300],[300,0],[100,0],[100,1100],[300,1100],[300,500]]]
glyph[108] = [[[500,0],[200,0],[100,100],[100,1000],[300,1000],[300,200],[500,200],[500,0]]]
glyph[109] = [[[500,400],[400,500],[300,400],[300,0],[100,0],[100,700],[300,700],[300,600],[ 400,700],[500,700],[600,600],[700,700],[900,700],[1100,500],[1100,0],[900,0],[ 900,400],[800,500],[700,400],[700,0],[500,0],[500,400]]]
glyph[110] = [[[600,0],[600,400],[500,500],[400,500],[300,400],[300,0],[100,0],[100,700],[300,700],[ 300,600],[400,700],[600,700],[800,500],[800,0],[600,0]]]
glyph[111] = [[[600,700],[800,500],[800,200],[600,0],[300,0],[100,200],[100,500],[300,700],[ 400,700],[400,500],[300,400],[300,300],[400,200],[500,200],[600,300],[600,400],[ 500,500],[500,700],[600,700]]]
glyph[112] = [[[100,700],[600,700],[800,500],[800,200],[600,0],[300,0],[300,-300],[100,-300],[ 100,300],[300,300],[300,200],[500,200],[600,300],[600,400],[500,500],[300,500],[ 300,400],[100,400],[100,700]]]
glyph[113] = [[[800,400],[600,400],[600,500],[400,500],[300,400],[300,300],[400,200],[600,200],[ 600,300],[800,300],[800,-300],[600,-300],[600,0],[300,0],[100,200],[100,500],[ 300,700],[800,700],[800,400]]]
glyph[114] = [[[600,400],[600,500],[400,500],[300,400],[300,0],[100,0],[100,700],[300,700],[ 300,600],[400,700],[700,700],[800,600],[800,400],[600,400]]]
glyph[115] = [[[200,0],[100,100],[100,200],[600,200],[600,300],[200,300],[100,400],[100,700],[ 200,800],[700,800],[800,700],[800,600],[300,600],[300,500],[700,500],[800,400],[ 800,100],[700,0],[200,0]]]
glyph[116] = [[[600,0],[400,0],[300,100],[300,600],[100,600],[100,800],[300,800],[300,1000],[ 500,1000],[500,800],[700,800],[700,600],[500,600],[500,200],[600,200],[600,0]]]
glyph[117] = [[[300,700],[300,300],[400,200],[500,200],[600,300],[600,700],[800,700],[800,0],[ 600,0],[600,100],[500,0],[300,0],[100,200],[100,700],[300,700]]]
glyph[118] = [[[300,0],[100,700],[300,700],[400,350],[500,700],[700,700],[500,0],[300,0]]]
glyph[119] = [[[600,350],[500,0],[300,0],[100,700],[300,700],[400,350],[500,700],[700,700],[ 800,350],[900,700],[1100,700],[900,0],[700,0],[600,350]]]
glyph[120] = [[[308,350],[100,600],[100,700],[300,700],[450,520],[600,700],[800,700],[ 800,600],[592,350],[800,100],[800,0],[600,0],[450,180],[300,0],[100,0],[100,100],[308,350]]]
glyph[121] = [[[214,-300],[300,0],[100,700],[300,700],[400,350],[500,700],[700,700],[500,0],[414,-300],[214,-300]]]
#z
glyph[122] = [[[100,500],[100,700],[700,700],[700,500],[400,200],[700,200],[700,0],[100,0],[ 100,200],[400,500],[100,500]]]
glyph[123] = [[[100,500],[200,600],[200,1000],[400,1200],[500,1200],[500,1000],[400,1000],[ 400,600],[300,500],[400,400],[400,0],[500,0],[500,-200],[400,-200],[200,0],[200,400],[100,500]]]
glyph[124] = [[[100,-100],[100,1100],[300,1100],[300,-100],[100,-100]]]
glyph[125] = [[[500,500],[400,600],[400,1000],[200,1200],[100,1200],[100,1000],[200,1000],[ 200,600],[300,500],[200,400],[200,0],[100,0],[100,-200],[200,-200],[400,0],[400,400],[500,500]]]
glyph[126] = [[[100,700],[250,800],[350,800],[650,600],[750,600],[900,700],[ 900,500],[ 750,400],[650,400],[350,600],[250,600],[100,500],[100,700]]]
glyph[230] = [[[300,700],[300,300],[400,200],[500,200],[600,300],[600,700],[800,700],[800,0],[600,0],[ 600,100],[500,0],[400,0],[300,100],[300,-300],[100,-300],[100,700],[300,700]]]


# glyph widths and indents
width[33] = 400;  indent[33] = 100 # !
width[34] = 500;  indent[34] = 200 # "
width[35] = 800;  indent[35] = 100 # #
width[36] = 800;  indent[36] = 100 # $
width[37] = 800;  indent[37] = 100 # #
width[38] = 900;  indent[38] = 000 # &
width[39] = 200;  indent[39] = 300 # '
width[40] = 450;  indent[40] = 100 # (
width[41] = 450;  indent[41] = 100 # )
width[42] = 600;  indent[42] = 200 # *
width[43] = 800;  indent[43] = 100
width[44] = 200;  indent[44] = 100
width[45] = 800;  indent[45] = 100
width[46] = 200;  indent[46] = 100
width[47] = 400;  indent[47] = 100
width[48] = 800;  indent[48] = 100
width[49] = 400;  indent[49] = 100
width[50] = 700;  indent[50] = 100
width[51] = 700;  indent[51] = 100
width[52] = 700;  indent[52] = 100
width[53] = 700;  indent[53] = 100
width[54] = 700;  indent[54] = 100
width[55] = 700;  indent[55] = 100
width[56] = 700;  indent[56] = 100
width[57] = 700;  indent[57] = 100
width[58] = 200;  indent[58] = 100
width[59] = 200;  indent[59] = 100
width[60] = 600;  indent[60] = 100
width[61] = 800;  indent[61] = 100
width[62] = 600;  indent[62] = 100
width[63] = 800;  indent[63] = 100
width[64] = 800;  indent[64] = 100
width[65] = 700;  indent[65] = 100
width[66] = 700;  indent[66] = 100
width[67] = 700;  indent[67] = 100
width[68] = 700;  indent[68] = 100
width[69] = 600;  indent[69] = 100
width[70] = 600;  indent[70] = 100
width[71] = 700;  indent[71] = 100
width[72] = 700;  indent[72] = 100
width[73] = 600;  indent[73] = 100
width[74] = 600;  indent[74] = 100
width[75] = 700;  indent[75] = 100
width[76] = 700;  indent[76] = 100
width[77] = 800;  indent[77] = 100
width[78] = 800;  indent[78] = 100
width[79] = 800;  indent[79] = 100
width[80] = 800;  indent[80] = 100
width[81] = 800;  indent[81] = 100
width[82] = 800;  indent[82] = 100
width[83] = 800;  indent[83] = 100
width[84] = 800;  indent[84] = 100
width[85] = 700;  indent[85] = 100
width[86] = 800;  indent[86] = 100
width[87] = 800;  indent[87] = 100
width[88] = 800;  indent[88] = 100
width[89] = 800;  indent[89] = 100
width[90] = 800;  indent[90] = 100
width[91] = 300;  indent[91] = 100
width[92] = 400;  indent[92] = 100
width[93] = 300;  indent[93] = 100
width[94] = 800;  indent[94] = 000
width[95] = 800;  indent[95] = 100
width[96] = 200;  indent[96] = 300
width[97] = 700;  indent[97] = 100
width[98] = 700;  indent[98] = 100
width[99] = 700;  indent[99] = 100
width[100] = 700;  indent[100] = 100
width[101] = 700;  indent[101] = 100
width[102] = 500;  indent[102] = 100
width[103] = 700;  indent[103] = 100
width[104] = 700;  indent[104] = 100
width[105] = 200;  indent[105] = 100
width[106] = 400;  indent[106] = 100
width[107] = 700;  indent[107] = 100
width[108] = 400;  indent[108] = 100
width[109] = 1000;  indent[109] = 100
width[110] = 700;  indent[110] = 100
width[111] = 700;  indent[111] = 100
width[112] = 700;  indent[112] = 100
width[113] = 700;  indent[113] = 100
width[114] = 700;  indent[114] = 100
width[115] = 700;  indent[115] = 100
width[116] = 600;  indent[116] = 100
width[117] = 700;  indent[117] = 100
width[118] = 600;  indent[118] = 100
width[119] = 1000;  indent[119] = 100
width[120] = 700;  indent[120] = 100
width[121] = 600;  indent[121] = 100
width[122] = 600;  indent[122] = 100
width[123] = 500;  indent[123] = 100  # {
width[124] = 400;  indent[124] = 100  # |
width[125] = 500;  indent[125] = 100  # }
width[126] = 800;  indent[126] = 100  # ~
width[230] = 700;  indent[230] = 100  # Greek mu

def text(text = 'abcd', size = 10, position=(0, 0), justify = 'left', layer = 0):
    scaling = size/800
    xoffset = position[0]
    yoffset = position[1]
    t = Device()
    for line in text.split('\n'):
        l = Device()
        for c in line:
            ascii_val = ord(c)
            if c == ' ':
                xoffset += 500*scaling
            elif 33 <= ascii_val <= 126:
                for poly in glyph[ascii_val]:
                    xpts = np.array(poly)[:,0]*scaling
                    ypts = np.array(poly)[:,1]*scaling
                    l.add_polygon([xpts + xoffset,ypts + yoffset], layer=layer)
                xoffset += (width[ascii_val] + indent[ascii_val])*scaling
            else: ValueError('[PHIDL] text(): No glyph for character with ascii value %s' % ascii_val)
        t.add_ref(l)
        yoffset -= 1500*scaling
        xoffset = position[0]
    justify = justify.lower()
    for l in t.references:
        if justify == 'left':   pass
        if justify == 'right':  l.xmax = position[0]# l.move(origin = l.bounds('NE'), destination = position, axis = 'x')
        if justify == 'center': l.move(origin = l.center, destination = position, axis = 'x')
    return t
    
    
#==============================================================================
# Example code    
#==============================================================================
    
#D = text('the quick brown\n fox jumped over\nthe lazy dog', justify = 'center', size = 800)
#quickplot(D)




#==============================================================================
#
# Wafer and die
#
#==============================================================================



def basic_die(
              size = (10000, 10000),
              street_width = 100,
              street_length = 1000,
              die_name = 'chip99',
              text_size = 100,
              text_location = 'SW', 
              layer = 0, 
              draw_bbox = True, 
              bbox_layer = 99,
              ):
    
    #==========================================================================
    #  Create the basic geometry
    #==========================================================================
    D = Device(name = 'die')
    xpts = np.array([size[0]/2, size[0]/2, size[0]/2-street_width, size[0]/2-street_width, size[0]/2-street_length, size[0]/2-street_length])
    ypts = np.array([size[1]/2, size[1]/2-street_length, size[0]/2-street_length, size[0]/2-street_width, size[0]/2-street_width, size[1]/2])
    D.add_polygon([xpts,ypts], layer = layer)
    D.add_polygon([-xpts,ypts], layer = layer)
    D.add_polygon([xpts,-ypts], layer = layer)
    D.add_polygon([-xpts,-ypts], layer = layer)
    
    if draw_bbox is True:
        s = np.array(size)/2
        D.add_polygon([[s[0],s[1]], [s[0],-s[1]],[-s[0],-s[1]],[-s[0],s[1]]], layer = bbox_layer)
    
    if type(text_location) is str:
        if text_location.upper() == 'SW':
            justify = 'left'
            text_position = (-size[0]/2 + street_width*2, -size[1]/2 + street_width*2)
        elif text_location.upper() == 'S':
            justify = 'center'
            text_position = (0, -size[1]/2 + street_width*2)
        elif text_location.upper() == 'SE':
            justify = 'right'
            text_position = (size[0]/2 - street_width*2, -size[1]/2 + street_width*2)
    else:
        text_position = text_location
        justify = 'center'
    D.add_ref(text(text = die_name, size = text_size, position=text_position, justify = justify, layer=layer))
    
    return D
    
    
    
#==============================================================================
# Example code
#==============================================================================

# D = basic_die(size = (10000, 10000), street_width = 100, street_length = 1000, 
#               die_name = 'chip99', text_size = 300, text_location = 'SW',  layer = 0,  
#               draw_bbox = True,  bbox_layer = 99)
# quickplot(D)





#==============================================================================
#
# Waveguide curves
#
#==============================================================================


def racetrack_gradual(width = 0.3, R = 5, N = 3, layer = 0):
    curve_fun = lambda t: _racetrack_gradual_parametric(t, R = 5, N = 3)
    route_path = gdspy.Path(width = width, initial_point = [0,0])
    route_path.parametric(curve_fun, number_of_evaluations=99,\
            max_points=199,  final_distance=None, layer=layer)
    D = Device()
    D.add(route_path)
    return D
    

def _racetrack_gradual_parametric(t, R, N):
    """ Takes in a parametric value ``t`` on (0,1), returns the x,y coordinates
    of a racetrack bent according to 20090810_EOS4_modulator_designs_excerptForJasonGradualBends.ppt """
    x0 = R/2**(1/N)
    Rmin = 2**(0.5-1/N)/(N-1)*R
    R0 = R-(x0-Rmin/sqrt(2))
    t = np.array(t)
    x,y = np.zeros(t.shape), np.zeros(t.shape)
    
    # Doing the math
    x = cos(t*pi/2)*R0 # t (0-1) while x (0 to R0)
    ii =  (Rmin/sqrt(2) < x) & (x <= R0)
    jj =  (0 < x) & (x <= Rmin/sqrt(2))
    y[ii] = (R**N - (x[ii]+(x0-Rmin/sqrt(2)))**N)**(1/N)
    y[jj] = (x0-Rmin/sqrt(2))+sqrt(Rmin**2-x[jj]**2)
    return x,y
    
    
    
#==============================================================================
# Example code
#==============================================================================

# D = racetrack_gradual(width = 0.3, R = 5, N = 3)
# quickplot(D)



# t = np.linspace(0,1)
# x,y = _racetrack_gradual_parametric(t, R = 5, N = 3)
# plt.plot(x,y)
# plt.axis('equal')




#==============================================================================
#
# yTron
#
#==============================================================================


def ytron_round(rho_intersection = 1, theta_intersection = 5, arm_length = 500, source_length = 500,
                  width_right = 200, width_left = 200, theta_resolution = 10, layer = 0):
    
    #==========================================================================
    #  Create the basic geometry
    #==========================================================================
    theta = theta_intersection*pi/180
    theta_resolution = theta_resolution*pi/180
    thetalist = np.linspace(-(pi-theta),-theta, int((pi-2*theta)/theta_resolution) + 2)
    semicircle_x = rho_intersection*cos(thetalist)
    semicircle_y = rho_intersection*sin(thetalist)+rho_intersection

    # Rest of yTron
    xc = rho_intersection*cos(theta) 
    yc = rho_intersection*sin(theta) 
    arm_x = arm_length*sin(theta) 
    arm_y = arm_length*cos(theta) 

    # Write out x and y coords for yTron polygon
    xpts = semicircle_x.tolist() + [xc+arm_x, xc+arm_x+width_right, xc+width_right, \
           xc+width_right, 0, -(xc+width_left), -(xc+width_left), -(xc+arm_x+width_left), -(xc+arm_x)] 
    ypts = semicircle_y.tolist() + [yc+arm_y,      yc+arm_y,      yc,   yc-source_length, yc-source_length,  \
            yc-source_length,        yc,        yc+arm_y,    yc+arm_y] 
    
    #==========================================================================
    #  Create a blank device, add the geometry, and define the ports
    #==========================================================================
    D = Device(name = 'ytron')
    D.add_polygon([xpts,ypts], layer = layer)
    D.add_port(name = 'left', midpoint = [-(xc+arm_x+width_left/2), yc+arm_y],  width = width_left, orientation = 90)
    D.add_port(name = 'right', midpoint = [xc+arm_x+width_right/2, yc+arm_y],  width = width_right, orientation = 90)
    D.add_port(name = 'source', midpoint = [0+(width_right-width_left)/2, -source_length+yc],  width = width_left + width_right + 2*xc, orientation = -90)
    
    #==========================================================================
    #  Record any parameters you may want to access later
    #==========================================================================
    D.meta['rho'] = rho_intersection
    D.meta['left_width'] = width_left
    D.meta['right_width'] = width_right
    D.meta['source_width'] = width_left + width_right + 2*xc

    return D
    
    
    
#==============================================================================
# Example code
#==============================================================================

#y = ytron_round(rho_intersection = 5, theta_intersection = 5, theta_resolution = 10, arm_length = 500, \
#                source_length = 500, width_right = 200, width_left = 200, layer = 0)
#quickplot(y)





#==============================================================================
#
# Fill
#
#==============================================================================


def _rasterize_polygons(polygons, bounds = [[-100, -100], [100, 100]], dx = 1, dy = 1):
    
    # Prepare polygon array by shifting all points into the first quadrant and 
    # separating points into x and y lists
    xpts = []
    ypts = []
    for p in polygons:
        p_array = np.asarray(p)
        x = p_array[:,0]
        y = p_array[:,1]
        xpts.append((x-bounds[0][0])/dx-0.5)
        ypts.append((y-bounds[0][1])/dy-0.5)

    # Initialize the raster matrix we'll be writing to
    xsize = int(np.ceil((bounds[1][0]-bounds[0][0]))/dx)
    ysize = int(np.ceil((bounds[1][1]-bounds[0][1]))/dy)
    raster = np.zeros((ysize, xsize), dtype=np.bool)
    
    # TODO: Replace polygon_perimeter with the supercover version
    for n in range(len(xpts)):
        rr, cc = draw.polygon(ypts[n], xpts[n], shape=raster.shape)
        rrp, ccp = draw.polygon_perimeter(ypts[n], xpts[n], shape=raster.shape, clip=False)
        raster[rr, cc] = 1
        raster[rrp, ccp] = 1
        
    return raster
    
def _raster_index_to_coords(i, j, bounds = [[-100, -100], [100, 100]], dx = 1, dy = 1):
    x = (j+0.5)*dx + bounds[0][0]
    y = (i+0.5)*dy + bounds[0][1]
    return x,y


def _expand_raster(raster, distance = (4,2)):
    if distance[0] <= 0.5 and distance[1] <= 0.5: return raster
        
    num_pixels = map(int, np.ceil(distance))
    neighborhood = np.zeros((num_pixels[1]*2+1, num_pixels[0]*2+1), dtype=np.bool)
    rr, cc = draw.ellipse(r = num_pixels[1], c = num_pixels[0], yradius = distance[1]+0.5, xradius = distance[0]+0.5)
    neighborhood[rr, cc] = 1
    
    return morphology.binary_dilation(image = raster, selem=neighborhood)

    
            
def _fill_cell_rectangle(size = (20,20), layers = (0,1,3), densities = (0.5, 0.25, 0.7)):
    D = Device()
    for layer, density in zip(layers, densities):
        rectangle_size = np.array(size)*np.sqrt(density)
        r = D.add_ref(rectangle(size = rectangle_size, layer = layer))
        r.center = (0,0)
    return D

    
def _fill_cell_device(size = None):
    D = Device(name = 'fill_cell')
    return D

    
def fill_rectangle(D, fill_size = (40,10), exclude_layers = None, fill_layers = (0,1,3), fill_densities = (0.5, 0.25, 0.7), margin = 100, bbox = None):
    fill_cell = _fill_cell_rectangle(size = fill_size, layers = fill_layers, densities = fill_densities)
    F = Device(name = 'fill_pattern')
    
    if exclude_layers is None:
        poly = D.get_polygons(by_spec=False, depth=None)
    else:
        exclude_layers = [_parse_layer(l) for l in exclude_layers]
        poly = D.get_polygons(by_spec=True, depth=None)
        poly = {key:poly[key] for key in poly if key in exclude_layers}
        poly = itertools.chain.from_iterable(poly.values())
#    elif np.array(exclude_layers).ndim == 1: # Then exclude_layers is a list of just layers e.g. [0,2,3]
#        poly = D.get_polygons(by_spec=True, depth=None)
#        poly = {key:poly[key] for key in poly if key[0] in exclude_layers} # Filter the dict
#        poly = itertools.chain.from_iterable(poly.values()) # Concatenate dict values to long list
#    elif np.array(exclude_layers).ndim == 2: # Then exclude_layers is a list of layers + datatypes e.g. [(0,1),(0,2),(1,0)]
#        poly = D.get_polygons(by_spec=True, depth=None)
#        poly = {key:poly[key] for key in poly if key in exclude_layers}
#        poly = itertools.chain.from_iterable(poly.values())
        
    if bbox is None:  bbox = D.bbox

    raster = _rasterize_polygons(polygons = poly, bounds = bbox, dx = fill_size[0], dy = fill_size[1])
    raster = _expand_raster(raster, distance = margin/np.array(fill_size))
    
    for i in range(np.size(raster,0)):
        sub_rasters = [list(g) for k, g in itertools.groupby(raster[i])]
        j = 0
        for s in sub_rasters:
            if s[0] == 0:
                x,y = _raster_index_to_coords(i, j, bbox, fill_size[0], fill_size[1])
                F.add(gdspy.CellArray(ref_cell = fill_cell, columns = len(s), rows = 1, spacing = fill_size, origin = (x, y)))
            j += len(s)
    
    return F


#==============================================================================
#
# Routing
#
#==============================================================================
        
def route(port1, port2, path_type = 'sine', width_type = 'straight', width1 = None, width2 = None, num_path_pts = 99, layer = 0):
    # Assuming they're both Ports for now
    point_a = np.array(port1.midpoint)
    if width1 is None:  width1 = port1.width
    point_b = np.array(port2.midpoint)
    if width2 is None:  width2 = port2.width
    if round(abs(mod(port1.orientation - port2.orientation,360)),3) != 180:
        raise ValueError('[DEVICE] route() error: Ports do not face each other (orientations must be 180 apart)') 
    orientation = port1.orientation
    
    separation = point_b - point_a  # Vector drawn from A to B
    distance = norm(separation) # Magnitude of vector from A to B
    rotation = np.arctan2(separation[1],separation[0])*180/pi # Rotation of vector from A to B
    angle = rotation - orientation   # If looking out along the normal of ``a``, the angle you would have to look to see ``b``
    forward_distance = distance*cos(angle*pi/180)
    lateral_distance = distance*sin(angle*pi/180)
    
    # Create a path assuming starting at the origin and setting orientation = 0
    # use the "connect" function later to move the path to the correct location
    xf = forward_distance
    yf = lateral_distance
    if path_type == 'straight':
        curve_fun = lambda t: [xf*t, yf*t]
        curve_deriv_fun = lambda t: [xf + t*0, t*0]
    if path_type == 'sine':
        curve_fun = lambda t: [xf*t, yf*(1-cos(t*pi))/2]
        curve_deriv_fun = lambda t: [xf  + t*0, yf*(sin(t*pi)*pi)/2]
    #if path_type == 'semicircle':
    #    def semicircle(t):
    #        t = np.array(t)
    #        x,y = np.zeros(t.shape), np.zeros(t.shape)
    #        ii = (0 <= t) & (t <= 0.5)
    #        jj = (0.5 < t) & (t <= 1)
    #        x[ii] = (cos(-pi/2 + t[ii]*pi/2))*xf
    #        y[ii] = (sin(-pi/2 + t[ii]*pi/2)+1)*yf*2
    #        x[jj] = (cos(pi*3/2 - t[jj]*pi)+2)*xf/2
    #        y[jj] = (sin(pi*3/2 - t[jj]*pi)+1)*yf/2
    #        return x,y
    #    curve_fun = semicircle
    #    curve_deriv_fun = None
    if width_type == 'straight':
        width_fun = lambda t: (width2 - width1)*t + width1
    if width_type == 'sine':
        width_fun = lambda t: (width2 - width1)*(1-cos(t*pi))/2 + width1
    
    route_path = gdspy.Path(width = width1, initial_point = (0,0))
    gds_layer, gds_datatype = _parse_layer(layer)
    route_path.parametric(curve_fun, curve_deriv_fun, number_of_evaluations=num_path_pts,
            max_points=199, final_width=width_fun, final_distance=None,
            layer=gds_layer, datatype = gds_datatype)
    
    # Make the route path into a Device with ports, and use "connect" to move it
    # into the proper location
    D = Device()
    D.add(route_path)
    p1 = D.add_port(name = 1, midpoint = (0,0), width = width1, orientation = 180)
    p2 = D.add_port(name = 2, midpoint = [forward_distance,lateral_distance], width = width2, orientation = 0)
    D.meta['length'] = route_path.length

    D.rotate(angle =  180 + port1.orientation - p1.orientation, center = p1.midpoint)
    D.move(origin = p1, destination = port1)
    return D


#==============================================================================
#
# Boolean functions
#
#==============================================================================

def inset(elements, distance = 0.1, join_first = True, precision = 0.001, layer = 0):

    if type(elements) is not list: elements = [elements]
    new_elements = []
    for e in elements:
        if isinstance(e, Device): new_elements += e.get_polygons()
        else: new_elements.append(e)
        
    gds_layer, gds_datatype = _parse_layer(layer)
    # This pre-joining (by expanding by precision) is makes this take twice as
    # long but is necessary because of floating point errors which otherwise
    # separate polygons which are nominally joined
    joined = gdspy.offset(new_elements, precision, join='miter', tolerance=2,
                          precision=precision, join_first=join_first,
                          max_points=199, layer=gds_layer, datatype = gds_datatype)
    p = gdspy.offset(joined, -distance, join='miter', tolerance=2,
                     precision=precision, join_first=join_first,
                     max_points=199, layer=gds_layer, datatype = gds_datatype)
    D = Device()
    D.add_polygon(p, layer=layer)
    return D
    