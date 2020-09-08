from __future__ import division, print_function, absolute_import
import numpy as np
from numpy import sqrt, pi, cos, sin, log, exp, sinh, mod
from numpy.linalg import norm
from phidl.device_layout import Device
from phidl.device_layout import _parse_layer
from phidl.geometry import turn
import gdspy


def _arc(radius = 10, width = 0.5, theta = 45, start_angle = 0, angle_resolution = 2.5, layer = 0):
    """ Creates an arc of arclength ``theta`` starting at angle ``start_angle`` """
    inner_radius = radius-width/2
    outer_radius = radius+width/2
    angle1 = (start_angle)*pi/180
    angle2 = (start_angle + theta)*pi/180
    t = np.linspace(angle1, angle2, int(np.ceil(abs(theta)/angle_resolution)))
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
    D.info['length'] = (abs(theta)*pi/180)*radius
    return D



def _gradual_bend(
    radius = 20,
    width = 1.0,
    angular_coverage=15,
    num_steps=10,
    angle_resolution=0.1,
    start_angle=0,
    direction='ccw',
    layer=0,
    ):

    """
    creates a 90-degree bent waveguide
    the bending radius is gradually increased until it reaches the minimum
    value of the radius at the "angular coverage" angle.
    it essentially creates a smooth transition to a bent waveguide mode.
    user can control number of steps provided.
    direction determined by start angle and cw or ccw switch
    ############
    with the default 10 "num_steps" and 15 degree coverage, effective radius is about 1.5*radius.
    """
    angular_coverage=np.deg2rad(angular_coverage)
    D = Device()

    #determines the increment in radius through its inverse from 0 to 1/r
    inc_rad =(radius**-1)/(num_steps)
    angle_step = angular_coverage/num_steps

    #construct a series of sub-arcs with equal angles but gradually decreasing bend radius
    arcs = []
    for x in range(num_steps):
        A = _arc(radius=1/((x+1)*inc_rad),width=width,theta=np.rad2deg(angle_step),start_angle=x*np.rad2deg(angle_step),angle_resolution=angle_resolution,layer=layer)
        a = D.add_ref(A)
        arcs.append(a)
        if x>0:
            a.connect(port=1,destination=prevPort)
        prevPort=a.ports[2]
    D.add_port(name=1,port=arcs[0].ports[1])

    #now connect a regular bend for the normal curved portion
    B = _arc(radius=radius,width=width,theta=45-np.rad2deg(angular_coverage),start_angle=angular_coverage,angle_resolution=angle_resolution,layer=layer)
    b = D.add_ref(B)
    b.connect(port=1,destination=prevPort)
    prevPort=b.ports[2]
    D.add_port(name=2,port=prevPort)

    #now create the overall structure
    Total = Device()

    #clone the half-curve into two objects and connect for a 90 deg bend.
    D1 = Total.add_ref(D)
    D2 = Total.add_ref(D)
    D2.mirror(p1=[0,0],p2=[1,1])
    D2.connect(port=2,destination=D1.ports[2])
    Total.xmin=0
    Total.ymin=0

    #orient to default settings...
    Total.mirror(p1=[0,0],p2=[1,1])
    Total.mirror(p1=[0,0],p2=[1,0])

    #orient to user-provided settings
    if direction == 'cw':
        Total.mirror(p1=[0,0],p2=[1,0])
    Total.rotate(angle=start_angle,center=Total.center)
    Total.center=[0,0]
    Total.add_port(name=1,port=D1.ports[1])
    Total.add_port(name=2,port=D2.ports[1])

    return Total


def route_basic(port1, port2, path_type = 'sine', width_type = 'straight', width1 = None, width2 = None, num_path_pts = 99, layer = 0):

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
    route_path.parametric(curve_fun, curve_deriv_fun, number_of_evaluations=num_path_pts,
            max_points=199, final_width=width_fun, final_distance=None)
    route_path_polygons = route_path.polygons

    # Make the route path into a Device with ports, and use "connect" to move it
    # into the proper location
    D = Device()
    D.add_polygon(route_path_polygons, layer = layer)
    p1 = D.add_port(name = 1, midpoint = (0,0), width = width1, orientation = 180)
    p2 = D.add_port(name = 2, midpoint = [forward_distance,lateral_distance], width = width2, orientation = 0)
    D.info['length'] = route_path.length

    D.rotate(angle =  180 + port1.orientation - p1.orientation, center = p1.midpoint)
    D.move(origin = p1, destination = port1)
    return D


# ################
# gradual_bend() - variable radius-of-curvature bends for low-loss routing
# note - these are not bezier spline curves, but are instead constructed by
# subdividing the coverage angle into equal segments and implementing a gradual
# decrease in bending radius until the minimum is reached. """

# ################
# route_manhattan() - routing between any two ports rotated in 90 degree increments
# note - ports must be located diagonally from each other and oriented along
# cardinal directions.  Routing can be done with circular or gradual bends.
# Waveguide width is set by the width parameter of the first port.

# ################
# route_manhattan_auto() - handy routine which performs route_manhattan() on a
# vector of ports provided to it, allowing easy connecting of many objects.




def route_manhattan(
    port1,
    port2,
    bendType='circular',
    layer=0,
    radius=20
    ):
    #route along cardinal directions between any two ports placed diagonally
    #from each other

    Total = Device()
    width=port1.width
    #first map into uniform plane with normal x,y coords
    #allows each situation to be put into uniform cases of quadrants for routing.
    #this is because bends change direction and positioning.
    if port1.orientation==0:
        p2=[port2.midpoint[0],port2.midpoint[1]]
        p1=[port1.midpoint[0],port1.midpoint[1]]
    if port1.orientation==90:
        p2=[port2.midpoint[1],-port2.midpoint[0]]
        p1=[port1.midpoint[1],-port1.midpoint[0]]
    if port1.orientation==180:
        p2=[-port2.midpoint[0],-port2.midpoint[1]]
        p1=[-port1.midpoint[0],-port1.midpoint[1]]
    if port1.orientation==270:
        p2=[-port2.midpoint[1],port2.midpoint[0]]
        p1=[-port1.midpoint[1],port1.midpoint[0]]

    Total.add_port(name=1,port=port1)
    Total.add_port(name=2,port=port2)

    if p2[1] == p1[1] or p2[0] == p1[0]:
        raise ValueError('Error - ports must be at different x AND y values.')

    #if it is parallel or anti-parallel, route with 180 option
    if (np.round(np.abs(np.mod(port1.orientation - port2.orientation,360)),3) == 180) or (np.round(np.abs(np.mod(port1.orientation - port2.orientation,360)),3) == 0):
        R1 = route_manhattan180(port1=port1,port2=port2,bendType=bendType,layer=layer,radius=radius)
        r1 = Total.add_ref(R1)

    else:
        #first quadrant case
        if (p2[1] > p1[1]) & (p2[0] > p1[0]):
            #simple 90 degree single-bend case
            if port2.orientation == port1.orientation-90 or port2.orientation == port1.orientation+270:
                R1 = route_manhattan90(port1=port1,port2=port2,bendType=bendType,layer=layer,radius=radius)
                r1 = Total.add_ref(R1)
            elif port2.orientation == port1.orientation+90 or port2.orientation == port1.orientation-270:
                if bendType == 'circular':
                    B1=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=port1.orientation,theta=90)
                    radiusEff=radius
                if bendType == 'gradual':
                    B1=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=port1.orientation,direction='ccw')
                    radiusEff=B1.xsize-width/2
                b1=Total.add_ref(B1)
                b1.connect(port=1,destination=port1)

                R1 = route_manhattan180(port1=b1.ports[2],port2=port2,bendType=bendType,layer=layer,radius=radius)
                r1 = Total.add_ref(R1)
        #second quadrant case
        if (p2[1] > p1[1]) & (p2[0] < p1[0]):
            if np.abs(port1.orientation-port2.orientation) == 90 or np.abs(port1.orientation-port2.orientation) == 270:
                if bendType == 'circular':
                    B1=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=port1.orientation,theta=90)
                    radiusEff=radius
                if bendType == 'gradual':
                    B1=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=port1.orientation,direction='ccw')
                    radiusEff=B1.xsize-width/2
                b1=Total.add_ref(B1)
                b1.connect(port=1,destination=port1)
                R1 = route_manhattan180(port1=b1.ports[2],port2=port2,bendType=bendType,layer=layer,radius=radius)
                r1 = Total.add_ref(R1)
        #third quadrant case
        if (p2[1] < p1[1]) & (p2[0] < p1[0]):
            if np.abs(port1.orientation-port2.orientation) == 90 or np.abs(port1.orientation-port2.orientation) == 270:
                if bendType == 'circular':
                    B1=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=port1.orientation,theta=-90)
                    radiusEff=radius
                if bendType == 'gradual':
                    B1=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=port1.orientation,direction='cw')
                    radiusEff=B1.xsize-width/2
                b1=Total.add_ref(B1)
                b1.connect(port=1,destination=port1)
                R1 = route_manhattan180(port1=b1.ports[2],port2=port2,bendType=bendType,layer=layer,radius=radius)
                r1 = Total.add_ref(R1)
        #fourth quadrant case
        if (p2[1] < p1[1]) & (p2[0] > p1[0]):
            #simple 90 degree single-bend case
            if port2.orientation == port1.orientation+90 or port2.orientation == port1.orientation-270:
                R1 = route_manhattan90(port1=port1,port2=port2,bendType=bendType,layer=layer,radius=radius)
                r1 = Total.add_ref(R1)
            elif port2.orientation == port1.orientation-90 or port2.orientation == port1.orientation+270:
                if bendType == 'circular':
                    B1=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=port1.orientation,theta=-90)
                    radiusEff=radius
                if bendType == 'gradual':
                    B1=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=port1.orientation,direction='cw')
                    radiusEff=B1.xsize-width/2
                b1=Total.add_ref(B1)
                b1.connect(port=1,destination=port1)
                R1 = route_manhattan180(port1=b1.ports[2],port2=port2,bendType=bendType,layer=layer,radius=radius)
                r1 = Total.add_ref(R1)
    return Total

def route_manhattan180(
    port1,
    port2,
    bendType='circular',
    layer=0,
    radius=20
    ):
    #this is a subroutine of route_manhattan() and should not be used by itself.
    Total = Device()
    width=port1.width
    #first map into uniform plane with normal x,y coords
    #allows each situation to be put into uniform cases of quadrants for routing.
    #this is because bends change direction and positioning.
    if port1.orientation==0:
        p2=[port2.midpoint[0],port2.midpoint[1]]
        p1=[port1.midpoint[0],port1.midpoint[1]]
    if port1.orientation==90:
        p2=[port2.midpoint[1],-port2.midpoint[0]]
        p1=[port1.midpoint[1],-port1.midpoint[0]]
    if port1.orientation==180:
        p2=[-port2.midpoint[0],-port2.midpoint[1]]
        p1=[-port1.midpoint[0],-port1.midpoint[1]]
    if port1.orientation==270:
        p2=[-port2.midpoint[1],port2.midpoint[0]]
        p1=[-port1.midpoint[1],port1.midpoint[0]]

    #create placeholder ports based on the imaginary coordinates we created
    Total.add_port(name='t1',midpoint=[0,0],orientation=0,width=width)
    if(port1.orientation!=port2.orientation):
        Total.add_port(name='t2',midpoint=list(np.subtract(p2,p1)),orientation=180,width=width)
    else:
        Total.add_port(name='t2',midpoint=list(np.subtract(p2,p1)),orientation=0,width=width)

    if port1.orientation==port2.orientation:
        #first quadrant target
        if (p2[1] > p1[1]) & (p2[0] > p1[0]):
            if bendType == 'circular':
                B1=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=0,theta=90)
                B2=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=90,theta=90)
                radiusEff=radius
            if bendType == 'gradual':
                B1=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=0,direction='ccw')
                B2=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=90,direction='ccw')
                radiusEff=B1.xsize-width/2
            b1=Total.add_ref(B1)
            b2=Total.add_ref(B2)


            b1.connect(port=b1.ports[1],destination=Total.ports['t1'])
            b1.move([p2[0]-p1[0],0])
            b2.connect(port=b2.ports[1],destination=b1.ports[2])
            b2.move([0,p2[1]-p1[1]-radiusEff*2])
            R1=route_basic(port1=Total.ports['t1'],port2=b1.ports[1],layer=layer)
            r1=Total.add_ref(R1)
            R2=route_basic(port1=b1.ports[2],port2=b2.ports[1],layer=layer)
            r2=Total.add_ref(R2)
            Total.add_port(name=1,port=r1.ports[1])
            Total.add_port(name=2,port=b2.ports[2])
        #second quadrant target
        if (p2[1] > p1[1]) & (p2[0] < p1[0]):
            if bendType == 'circular':
                B1=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=0,theta=90)
                B2=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=90,theta=90)
                radiusEff=radius
            if bendType == 'gradual':
                B1=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=0,direction='ccw')
                B2=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=90,direction='ccw')
                radiusEff=B1.xsize-width/2
            b1=Total.add_ref(B1)
            b2=Total.add_ref(B2)
            b1.connect(port=b1.ports[1],destination=Total.ports['t1'])

            b2.connect(port=b2.ports[1],destination=b1.ports[2])
            b2.move([0,p2[1]-p1[1]-radiusEff*2])
            R1=route_basic(port1=b1.ports[2],port2=b2.ports[1],layer=layer)
            r1=Total.add_ref(R1)
            R2=route_basic(port1=b2.ports[2],port2=Total.ports['t2'],layer=layer)
            r2=Total.add_ref(R2)
            Total.add_port(name=1,port=b1.ports[1])
            Total.add_port(name=2,port=r2.ports[2])
        #third quadrant target
        if (p2[1] < p1[1]) & (p2[0] < p1[0]):
            if bendType == 'circular':
                B1=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=0,theta=-90)
                B2=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=-90,theta=-90)
                radiusEff=radius
            if bendType == 'gradual':
                B1=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=0,direction='cw')
                B2=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=-90,direction='cw')
                radiusEff=B1.xsize-width/2
            b1=Total.add_ref(B1)
            b2=Total.add_ref(B2)
            b1.connect(port=b1.ports[1],destination=Total.ports['t1'])

            b2.connect(port=b2.ports[1],destination=b1.ports[2])
            b2.move([0,p2[1]-p1[1]+radiusEff*2])
            R1=route_basic(port1=b1.ports[2],port2=b2.ports[1],layer=layer)
            r1=Total.add_ref(R1)
            R2=route_basic(port1=b2.ports[2],port2=Total.ports['t2'],layer=layer)
            r2=Total.add_ref(R2)
            Total.add_port(name=1,port=b1.ports[1])
            Total.add_port(name=2,port=r2.ports[2])
        #fourth quadrant target
        if (p2[1] < p1[1]) & (p2[0] > p1[0]):
            if bendType == 'circular':
                B1=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=0,theta=-90)
                B2=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=-90,theta=-90)
                radiusEff=radius
            if bendType == 'gradual':
                B1=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=0,direction='cw')
                B2=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=-90,direction='cw')
                radiusEff=B1.xsize-width/2
            b1=Total.add_ref(B1)
            b2=Total.add_ref(B2)


            b1.connect(port=b1.ports[1],destination=Total.ports['t1'])
            b1.move([p2[0]-p1[0],0])
            b2.connect(port=b2.ports[1],destination=b1.ports[2])
            b2.move([0,p2[1]-p1[1]+radiusEff*2])
            R1=route_basic(port1=Total.ports['t1'],port2=b1.ports[1],layer=layer)
            r1=Total.add_ref(R1)
            R2=route_basic(port1=b1.ports[2],port2=b2.ports[1],layer=layer)
            r2=Total.add_ref(R2)
            Total.add_port(name=1,port=r1.ports[1])
            Total.add_port(name=2,port=b2.ports[2])

    #other port orientations are not supported:
    elif np.round(np.abs(np.mod(port1.orientation - port2.orientation,360)),3) != 180:
        raise ValueError('[DEVICE] route() error: Ports do not face each other (orientations must be 180 apart)')
    #otherwise, they are 180 degrees apart:
    else:
        #first quadrant target
        if (p2[1] > p1[1]) & (p2[0] > p1[0]):
            if bendType == 'circular':
                B1=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=0,theta=90)
                B2=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=90,theta=-90)
                radiusEff=radius
            if bendType == 'gradual':
                B1=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=0,direction='ccw')
                B2=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=90,direction='cw')
                radiusEff=B1.xsize-width/2
            b1=Total.add_ref(B1)
            b2=Total.add_ref(B2)


            b1.connect(port=b1.ports[1],destination=Total.ports['t1'])
            b1.move([p2[0]-p1[0]-radiusEff*2,0])
            b2.connect(port=b2.ports[1],destination=b1.ports[2])
            b2.move([0,p2[1]-p1[1]-radiusEff*2])
            R1=route_basic(port1=Total.ports['t1'],port2=b1.ports[1],layer=layer)
            r1=Total.add_ref(R1)
            R2=route_basic(port1=b1.ports[2],port2=b2.ports[1],layer=layer)
            r2=Total.add_ref(R2)
            Total.add_port(name=1,port=r1.ports[1])
            Total.add_port(name=2,port=b2.ports[2])
        #second quadrant target
        if (p2[1] > p1[1]) & (p2[0] < p1[0]):
            if bendType == 'circular':
                B1=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=0,theta=90)
                B2=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=90,theta=90)
                B3=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=180,theta=-90)
                B4=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=90,theta=-90)
                radiusEff=radius
            if bendType == 'gradual':
                B1=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=0,direction='ccw')
                B2=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=90,direction='ccw')
                B3=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=180,direction='cw')
                B4=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=90,direction='cw')
                radiusEff=B1.xsize-width/2
            b1=Total.add_ref(B1)
            b2=Total.add_ref(B2)
            b3=Total.add_ref(B3)
            b4=Total.add_ref(B4)


            b1.connect(port=b1.ports[1],destination=Total.ports['t1'])

            b2.connect(port=b2.ports[1],destination=b1.ports[2])
            b2.move([0,p2[1]-p1[1]-radiusEff*4])
            R1=route_basic(port1=b1.ports[2],port2=b2.ports[1],layer=layer)
            r1=Total.add_ref(R1)
            b3.connect(port=b3.ports[1],destination=b2.ports[2])
            b3.move([p2[0]-p1[0],0])
            R2=route_basic(port1=b2.ports[2],port2=b3.ports[1],layer=layer)
            r2=Total.add_ref(R2)

            b4.connect(port=b4.ports[1],destination=b3.ports[2])

            Total.add_port(name=1,port=r1.ports[1])
            Total.add_port(name=2,port=b4.ports[2])
         #third quadrant target
        if (p2[1] < p1[1]) & (p2[0] < p1[0]):
            if bendType == 'circular':
                B1=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=0,theta=-90)
                B2=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=-90,theta=-90)
                B3=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=-180,theta=90)
                B4=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=-90,theta=90)
                radiusEff=radius
            if bendType == 'gradual':
                B1=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=0,direction='cw')
                B2=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=-90,direction='cw')
                B3=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=-180,direction='ccw')
                B4=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=-90,direction='ccw')
                radiusEff=B1.xsize-width/2
            b1=Total.add_ref(B1)
            b2=Total.add_ref(B2)
            b3=Total.add_ref(B3)
            b4=Total.add_ref(B4)


            b1.connect(port=b1.ports[1],destination=Total.ports['t1'])

            b2.connect(port=b2.ports[1],destination=b1.ports[2])
            b2.move([0,p2[1]-p1[1]+radiusEff*4])
            R1=route_basic(port1=b1.ports[2],port2=b2.ports[1],layer=layer)
            r1=Total.add_ref(R1)
            b3.connect(port=b3.ports[1],destination=b2.ports[2])
            b3.move([p2[0]-p1[0],0])
            R2=route_basic(port1=b2.ports[2],port2=b3.ports[1],layer=layer)
            r2=Total.add_ref(R2)

            b4.connect(port=b4.ports[1],destination=b3.ports[2])

            Total.add_port(name=1,port=r1.ports[1])
            Total.add_port(name=2,port=b4.ports[2])
        #fourth quadrant target
        if (p2[1] < p1[1]) & (p2[0] > p1[0]):
            if bendType == 'circular':
                B1=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=0,theta=-90)
                B2=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=-90,theta=90)
                radiusEff=radius
            if bendType == 'gradual':
                B1=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=0,direction='cw')
                B2=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=-90,direction='ccw')
                radiusEff=B1.xsize-width/2
            b1=Total.add_ref(B1)
            b2=Total.add_ref(B2)


            b1.connect(port=b1.ports[1],destination=Total.ports['t1'])
            b1.move([p2[0]-p1[0]-radiusEff*2,0])
            b2.connect(port=b2.ports[1],destination=b1.ports[2])
            b2.move([0,p2[1]-p1[1]+radiusEff*2])
            R1=route_basic(port1=Total.ports['t1'],port2=b1.ports[1],layer=layer)
            r1=Total.add_ref(R1)
            R2=route_basic(port1=b1.ports[2],port2=b2.ports[1],layer=layer)
            r2=Total.add_ref(R2)
            Total.add_port(name=1,port=r1.ports[1])
            Total.add_port(name=2,port=b2.ports[2])

    Total.rotate(angle =  port1.orientation, center = p1)
    Total.move(origin = Total.ports['t1'], destination = port1)
    return Total

def route_manhattan90(
    port1,
    port2,
    bendType='circular',
    layer=0,
    radius=20
    ):
    #this is a subroutine of route_manhattan() and should not be used by itself.
    Total = Device()
    width=port1.width
    #first map into uniform plane with normal x,y coords
    #allows each situation to be put into uniform cases of quadrants for routing.
    #this is because bends change direction and positioning.
    if port1.orientation==0:
        p2=[port2.midpoint[0],port2.midpoint[1]]
        p1=[port1.midpoint[0],port1.midpoint[1]]
    if port1.orientation==90:
        p2=[port2.midpoint[1],-port2.midpoint[0]]
        p1=[port1.midpoint[1],-port1.midpoint[0]]
    if port1.orientation==180:
        p2=[-port2.midpoint[0],-port2.midpoint[1]]
        p1=[-port1.midpoint[0],-port1.midpoint[1]]
    if port1.orientation==270:
        p2=[-port2.midpoint[1],port2.midpoint[0]]
        p1=[-port1.midpoint[1],port1.midpoint[0]]

    #create placeholder ports based on the imaginary coordinates we created
    Total.add_port(name='t1',midpoint=[0,0],orientation=0,width=width)

    #CHECK THIS


    #first quadrant target, route upward
    if (p2[1] > p1[1]) & (p2[0] > p1[0]):
        Total.add_port(name='t2',midpoint=list(np.subtract(p2,p1)),orientation=-90,width=width)
        if bendType == 'circular':
            B1=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=0,theta=90)
            radiusEff=radius
        if bendType == 'gradual':
            B1=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=0,direction='ccw')
            radiusEff=B1.xsize-width/2
        b1=Total.add_ref(B1)
        b1.connect(port=b1.ports[1],destination=Total.ports['t1'])
        b1.move([p2[0]-p1[0]-radiusEff,0])

        R1 = route_basic(port1=Total.ports['t1'],port2=b1.ports[1],layer=layer)
        R2 = route_basic(port1=b1.ports[2],port2=Total.ports['t2'],layer=layer)
        r1 = Total.add_ref(R1)
        r2 = Total.add_ref(R2)
        Total.add_port(name=1,port=r1.ports[1])
        Total.add_port(name=2,port=r2.ports[2])

    #fourth quadrant target, route downward
    if (p2[1] < p1[1]) & (p2[0] > p1[0]):
        Total.add_port(name='t2',midpoint=list(np.subtract(p2,p1)),orientation=90,width=width)
        if bendType == 'circular':
            B1=_arc(radius=radius,width=width,layer=layer,angle_resolution=1,start_angle=0,theta=-90)
            radiusEff=radius
        if bendType == 'gradual':
            B1=_gradual_bend(radius=radius,width=width,layer=layer,start_angle=0,direction='cw')
            radiusEff=B1.xsize-width/2
        b1=Total.add_ref(B1)
        b1.connect(port=b1.ports[1],destination=Total.ports['t1'])
        b1.move([p2[0]-p1[0]-radiusEff,0])
        R1 = route_basic(port1=Total.ports['t1'],port2=b1.ports[1],layer=layer)
        R2 = route_basic(port1=b1.ports[2],port2=Total.ports['t2'],layer=layer)
        r1 = Total.add_ref(R1)
        r2 = Total.add_ref(R2)
        Total.add_port(name=1,port=r1.ports[1])
        Total.add_port(name=2,port=r2.ports[2])
    Total.rotate(angle =  port1.orientation, center = p1)
    Total.move(origin = Total.ports['t1'], destination = port1)

    return Total


def route_manhattan_auto(
    ports,
    bendType='circular',
    layer=0,
    radius=20
    ):
    """ routes a one-dimensional array of ports using manhattan algorithm
    and give it a series of ports to route to in a continuous list.
    accepts same parameters as ordinary route_manhattan to determine bending """
    Total=Device()
    for x in range(int(np.floor(len(ports)/2))+1):
        R = route_manhattan(port1=ports[x],port2=ports[x+1],bendType=bendType,layer=layer,radius=radius)
        r = Total.add_ref(R)

    return Total

def route_turn_manhattan(
    port1,
    port2,
    layer=0,
    radius=20
    ):
    """
    Mahattan routing between two ports. If directions are not cardinal, adds a
    turn to make cardinal and then routes.
    
    Parameters
    ----------
    port1, port2: Port objects
        Ports to route to and from
    layer: int (default: 0)
        Layer to use for the routes
    radius: float (default: 20)
        Curve radius for bends
        
    Returns
    ----------
    Device object
    
    Notes
    ----------
    If direction is not cardinal, will route to nearest cardinal, then call
    route_manhattan.
    """
    
    D = Device()    
    new_ports = []
    for port in (port1, port2):
        if port.orientation % 90 == 0:
            new_ports.append(port)
        else:
            turn_angle = get_turn_angle(port.orientation, to_cardinal(port.orientation))
            turn_route = turn(port, radius=radius, angle=turn_angle, layer=layer)
            D.add_ref(turn_route)
            new_ports.append(turn_route.ports[2])
    
    #Manhattan on new ports
    route = route_manhattan(new_ports[0], new_ports[1], bendType='circular', layer=layer,
                       radius=radius)
    
    D.add_ref(route)
    
    return D
    

def to_cardinal(angle):
    """
    Determines which cardinal direction is closest to input angle

    Parameters
    ----------
    angle : float

    Returns
    -------
    angle : [-180, -90, 0, 90]
        Which cardinal direction is closest to the input angle
    """

    angle = map_to_pm180(angle)

    cardinals = np.array([-180, -90, 0, 90])

    arg = np.argmin(np.abs(angle - cardinals))

    return cardinals[arg]


def map_to_pm180(angle):
    """converts an angle to an angle between -180 (inclusive) to +180 (exclusive)"""
    return np.mod(angle + 180,360)-180

def get_turn_angle(start_angle, target_angle):
    """
    Difference in angle in the range -180 to +180 (where negative is counter clockwise)

    Parameters
    ----------
    start_angle, target_angle : float

    Returns
    -------
    float
        difference in angle.
    """
    return map_to_pm180(target_angle - start_angle)


#==============================================================================
# Test code for route_manhattan
#==============================================================================

# D=Device()
# A=pg.compass()
# A.add_port(name=1,port=A.ports['N'])
# A.add_port(name=2,port=A.ports['E'])
# A.add_port(name=3,port=A.ports['S'])
# A.add_port(name=4,port=A.ports['W'])
# points=[]
# points.append((300,300))
# points.append((-300,300))
# points.append((-300,-300))
# points.append((300,-300))

# xoff=0
# yoff=0
# for x in range(4):
#     for y in range(4):
#         for z in range(4):
#             a = D.add_ref(A)
#             b = D.add_ref(A)
#             a.center=(xoff,yoff)
#             b.center=(xoff+(points[y])[0],yoff+(points[y])[1])
#             C = route_manhattan(bendType='gradual',port1=a.ports[z+1],port2=b.ports[x+1])
#             c=D.add_ref(C)
#             yoff+=600
#         yoff+=600
#     xoff+=600
#     yoff=0

# quickplot(D)



def point_path(points = [(0,0), (4,0), (4,8)], width = 1, layer = 0):
    points = np.asarray(points)
    dxdy = points[1:] - points[:-1]
    angles = (np.arctan2(dxdy[:,1], dxdy[:,0])).tolist()
    angles = np.array([angles[0]] + angles + [angles[-1]])
    diff_angles = (angles[1:] - angles[:-1])
    mean_angles = (angles[1:] + angles[:-1])/2
    dx = width/2*np.cos((mean_angles - pi/2))/np.cos((diff_angles/2))
    dy = width/2*np.sin((mean_angles - pi/2))/np.cos((diff_angles/2))
    left_points =  points.T - np.array([dx,dy])
    right_points = points.T + np.array([dx,dy])
    all_points = np.concatenate([left_points.T, right_points.T[::-1]])

    D = Device()
    D.add_polygon(all_points, layer = layer)
    D.add_port(name = 1, midpoint = points[0],  width = width, orientation = angles[0]*180/pi+180)
    D.add_port(name = 2, midpoint = points[-1], width = width, orientation = angles[-1]*180/pi)
    return D

# quickplot(point_path())