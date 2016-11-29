# -*- coding: utf-8 -*-
"""
advancedRouting
PHIDL module
created by JTC 2016

contains several modules and subroutines useful for photonic waveguide routing

################
gradualBend() - variable radius-of-curvature bends for low-loss routing
note - these are not bezier spline curves, but are instead constructed by
subdividing the coverage angle into equal segments and implementing a gradual
decrease in bending radius until the minimum is reached.

################
routeManhattan() - routing between any two ports rotated in 90 degree increments
note - ports must be located diagonally from each other and oriented along 
cardinal directions.  Routing can be done with circular or gradual bends.
Waveguide width is set by the width parameter of the first port.

################
routeManhattanAuto() - handy routine which performs routeManhattan() on a
vector of ports provided to it, allowing easy connecting of many objects.

"""
from __future__ import division
from phidl import Device, quickplot
import numpy as np
import phidl.geometry as pg
from numpy import sqrt, pi, cos, sin, log, exp, sinh, mod


def gradualBend(
    radius = 20,
    wWg = 1.0,
    angularCoverage=15,
    nsteps=10,
    layer=0,
    angle_resolution=0.1,
    start_angle=0,
    direction='ccw'
    ):
    
    """
    creates a 90-degree bent waveguide
    the bending radius is gradually increased until it reaches the minimum
    value of the radius at the "angular coverage" angle.
    it essentially creates a smooth transition to a bent waveguide mode.
    user can control number of steps provided.
    direction determined by start angle and cw or ccw switch
    ############
    with the default 10 "nsteps" and 15 degree coverage, effective radius is about 1.5*radius.
    """
    angularCoverage=np.deg2rad(angularCoverage)
    D = Device()
    
    #determines the increment in radius through its inverse from 0 to 1/r
    inc_rad =(radius**-1)/(nsteps)
    angle_step = angularCoverage/nsteps
    
    #construct a series of sub-arcs with equal angles but gradually decreasing bend radius
    arcs = []
    for x in xrange(nsteps):
        A = pg.arc(radius=1/((x+1)*inc_rad),width=wWg,theta=np.rad2deg(angle_step),start_angle=x*np.rad2deg(angle_step),angle_resolution=angle_resolution,layer=layer)
        a = D.add_ref(A)
        arcs.append(a)
        if x>0:        
            a.connect(port=1,destination=prevPort)
        prevPort=a.ports[2]
    D.add_port(name=1,port=arcs[0].ports[1])
    
    #now connect a regular bend for the normal curved portion
    B = pg.arc(radius=radius,width=wWg,theta=45-np.rad2deg(angularCoverage),start_angle=angularCoverage,angle_resolution=angle_resolution,layer=layer)
    b = D.add_ref(B)
    b.connect(port=1,destination=prevPort)
    prevPort=b.ports[2]
    D.add_port(name=2,port=prevPort)
    
    #now create the overall structure
    total = Device()
    
    #clone the half-curve into two objects and connect for a 90 deg bend.
    D1 = total.add_ref(D)
    D2 = total.add_ref(D)
    D2.reflect(p1=[0,0],p2=[1,1])
    D2.connect(port=2,destination=D1.ports[2])
    total.xmin=0
    total.ymin=0
    
    #orient to default settings...
    total.reflect(p1=[0,0],p2=[1,1])
    total.reflect(p1=[0,0],p2=[1,0])

    #orient to user-provided settings
    if direction == 'cw':
        total.reflect(p1=[0,0],p2=[1,0])
    total.rotate(angle=start_angle,center=total.center)
    total.center=[0,0]
    total.add_port(name=1,port=D1.ports[1])
    total.add_port(name=2,port=D2.ports[1]) 
    return total
    
def routeManhattan(
    port1,
    port2,
    bendType='circular',
    layer=0,
    radius=20
    ):
    #route along cardinal directions between any two ports placed diagonally
    #from each other
    
    total = Device()
    wWg=port1.width
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

    total.add_port(name=1,port=port1)
    total.add_port(name=2,port=port2)   

    if p2[1] == p1[1] or p2[0] == p1[0]:
        raise ValueError('Error - ports must be at different x AND y values.')
    
    #if it is parallel or anti-parallel, route with 180 option
    if (np.round(np.abs(np.mod(port1.orientation - port2.orientation,360)),3) == 180) or (np.round(np.abs(np.mod(port1.orientation - port2.orientation,360)),3) == 0):
        R1 = routeManhattan180(port1=port1,port2=port2,bendType=bendType,layer=layer,radius=radius)
        r1 = total.add(R1)

    else:
        #first quadrant case
        if (p2[1] > p1[1]) & (p2[0] > p1[0]):
            #simple 90 degree single-bend case
            if port2.orientation == port1.orientation-90 or port2.orientation == port1.orientation+270:
                R1 = routeManhattan90(port1=port1,port2=port2,bendType=bendType,layer=layer,radius=radius)
                r1 = total.add(R1)
            elif port2.orientation == port1.orientation+90 or port2.orientation == port1.orientation-270: 
                if bendType == 'circular':
                    B1=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=port1.orientation,theta=90)
                    radiusEff=radius
                if bendType == 'gradual':
                    B1=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=port1.orientation,direction='ccw')
                    radiusEff=B1.xsize-wWg/2
                b1=total.add_ref(B1)
                b1.connect(port=1,destination=port1)

                R1 = routeManhattan180(port1=b1.ports[2],port2=port2,bendType=bendType,layer=layer,radius=radius)
                r1 = total.add(R1)  
        #second quadrant case
        if (p2[1] > p1[1]) & (p2[0] < p1[0]):
            if np.abs(port1.orientation-port2.orientation) == 90 or np.abs(port1.orientation-port2.orientation) == 270: 
                if bendType == 'circular':
                    B1=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=port1.orientation,theta=90)
                    radiusEff=radius
                if bendType == 'gradual':
                    B1=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=port1.orientation,direction='ccw')
                    radiusEff=B1.xsize-wWg/2
                b1=total.add_ref(B1)
                b1.connect(port=1,destination=port1)
                R1 = routeManhattan180(port1=b1.ports[2],port2=port2,bendType=bendType,layer=layer,radius=radius)
                r1 = total.add(R1)   
        #third quadrant case
        if (p2[1] < p1[1]) & (p2[0] < p1[0]):
            if np.abs(port1.orientation-port2.orientation) == 90 or np.abs(port1.orientation-port2.orientation) == 270: 
                if bendType == 'circular':
                    B1=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=port1.orientation,theta=-90)
                    radiusEff=radius
                if bendType == 'gradual':
                    B1=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=port1.orientation,direction='cw')
                    radiusEff=B1.xsize-wWg/2
                b1=total.add_ref(B1)
                b1.connect(port=1,destination=port1)
                R1 = routeManhattan180(port1=b1.ports[2],port2=port2,bendType=bendType,layer=layer,radius=radius)
                r1 = total.add(R1)   
        #fourth quadrant case
        if (p2[1] < p1[1]) & (p2[0] > p1[0]):
            #simple 90 degree single-bend case
            if port2.orientation == port1.orientation+90 or port2.orientation == port1.orientation-270:
                R1 = routeManhattan90(port1=port1,port2=port2,bendType=bendType,layer=layer,radius=radius)
                r1 = total.add(R1)
            elif port2.orientation == port1.orientation-90 or port2.orientation == port1.orientation+270: 
                if bendType == 'circular':
                    B1=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=port1.orientation,theta=-90)
                    radiusEff=radius
                if bendType == 'gradual':
                    B1=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=port1.orientation,direction='cw')
                    radiusEff=B1.xsize-wWg/2
                b1=total.add_ref(B1)
                b1.connect(port=1,destination=port1)
                R1 = routeManhattan180(port1=b1.ports[2],port2=port2,bendType=bendType,layer=layer,radius=radius)
                r1 = total.add(R1) 
    return total

def routeManhattan180(
    port1,
    port2,
    bendType='circular',
    layer=0,
    radius=20
    ):
    #this is a subroutine of routeManhattan() and should not be used by itself.
    total = Device()
    wWg=port1.width
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
    total.add_port(name='t1',midpoint=[0,0],orientation=0,width=wWg)
    if(port1.orientation!=port2.orientation):
        total.add_port(name='t2',midpoint=list(np.subtract(p2,p1)),orientation=180,width=wWg)
    else:
        total.add_port(name='t2',midpoint=list(np.subtract(p2,p1)),orientation=0,width=wWg)

    if port1.orientation==port2.orientation:
        #first quadrant target
        if (p2[1] > p1[1]) & (p2[0] > p1[0]):
            if bendType == 'circular':
                B1=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=0,theta=90)
                B2=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=90,theta=90)
                radiusEff=radius
            if bendType == 'gradual':
                B1=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=0,direction='ccw')
                B2=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=90,direction='ccw')
                radiusEff=B1.xsize-wWg/2
            b1=total.add_ref(B1)
            b2=total.add_ref(B2)
            
            
            b1.connect(port=b1.ports[1],destination=total.ports['t1'])
            b1.move([p2[0]-p1[0],0])
            b2.connect(port=b2.ports[1],destination=b1.ports[2])
            b2.move([0,p2[1]-p1[1]-radiusEff*2])
            R1=pg.route(port1=total.ports['t1'],port2=b1.ports[1],layer=layer)
            r1=total.add_ref(R1)
            R2=pg.route(port1=b1.ports[2],port2=b2.ports[1],layer=layer)
            r2=total.add_ref(R2)
            total.add_port(name=1,port=r1.ports[1])
            total.add_port(name=2,port=b2.ports[2])
        #second quadrant target
        if (p2[1] > p1[1]) & (p2[0] < p1[0]):
            if bendType == 'circular':
                B1=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=0,theta=90)
                B2=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=90,theta=90)
                radiusEff=radius
            if bendType == 'gradual':
                B1=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=0,direction='ccw')
                B2=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=90,direction='ccw')
                radiusEff=B1.xsize-wWg/2
            b1=total.add_ref(B1)
            b2=total.add_ref(B2)
            b1.connect(port=b1.ports[1],destination=total.ports['t1'])

            b2.connect(port=b2.ports[1],destination=b1.ports[2])
            b2.move([0,p2[1]-p1[1]-radiusEff*2])
            R1=pg.route(port1=b1.ports[2],port2=b2.ports[1],layer=layer)
            r1=total.add_ref(R1)
            R2=pg.route(port1=b2.ports[2],port2=total.ports['t2'],layer=layer)
            r2=total.add_ref(R2)
            total.add_port(name=1,port=b1.ports[1])
            total.add_port(name=2,port=r2.ports[2])
        #third quadrant target
        if (p2[1] < p1[1]) & (p2[0] < p1[0]):
            if bendType == 'circular':
                B1=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=0,theta=-90)
                B2=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=-90,theta=-90)
                radiusEff=radius
            if bendType == 'gradual':
                B1=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=0,direction='cw')
                B2=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=-90,direction='cw')
                radiusEff=B1.xsize-wWg/2
            b1=total.add_ref(B1)
            b2=total.add_ref(B2)
            b1.connect(port=b1.ports[1],destination=total.ports['t1'])

            b2.connect(port=b2.ports[1],destination=b1.ports[2])
            b2.move([0,p2[1]-p1[1]+radiusEff*2])
            R1=pg.route(port1=b1.ports[2],port2=b2.ports[1],layer=layer)
            r1=total.add_ref(R1)
            R2=pg.route(port1=b2.ports[2],port2=total.ports['t2'],layer=layer)
            r2=total.add_ref(R2)
            total.add_port(name=1,port=b1.ports[1])
            total.add_port(name=2,port=r2.ports[2])
        #fourth quadrant target
        if (p2[1] < p1[1]) & (p2[0] > p1[0]):
            if bendType == 'circular':
                B1=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=0,theta=-90)
                B2=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=-90,theta=-90)
                radiusEff=radius
            if bendType == 'gradual':
                B1=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=0,direction='cw')
                B2=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=-90,direction='cw')
                radiusEff=B1.xsize-wWg/2
            b1=total.add_ref(B1)
            b2=total.add_ref(B2)
            
            
            b1.connect(port=b1.ports[1],destination=total.ports['t1'])
            b1.move([p2[0]-p1[0],0])
            b2.connect(port=b2.ports[1],destination=b1.ports[2])
            b2.move([0,p2[1]-p1[1]+radiusEff*2])
            R1=pg.route(port1=total.ports['t1'],port2=b1.ports[1],layer=layer)
            r1=total.add_ref(R1)
            R2=pg.route(port1=b1.ports[2],port2=b2.ports[1],layer=layer)
            r2=total.add_ref(R2)
            total.add_port(name=1,port=r1.ports[1])
            total.add_port(name=2,port=b2.ports[2])

    #other port orientations are not supported:
    elif np.round(np.abs(np.mod(port1.orientation - port2.orientation,360)),3) != 180:
        raise ValueError('[DEVICE] route() error: Ports do not face each other (orientations must be 180 apart)')    
    #otherwise, they are 180 degrees apart:
    else:
        #first quadrant target
        if (p2[1] > p1[1]) & (p2[0] > p1[0]):
            if bendType == 'circular':
                B1=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=0,theta=90)
                B2=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=90,theta=-90)
                radiusEff=radius
            if bendType == 'gradual':
                B1=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=0,direction='ccw')
                B2=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=90,direction='cw')
                radiusEff=B1.xsize-wWg/2
            b1=total.add_ref(B1)
            b2=total.add_ref(B2)
            
            
            b1.connect(port=b1.ports[1],destination=total.ports['t1'])
            b1.move([p2[0]-p1[0]-radiusEff*2,0])
            b2.connect(port=b2.ports[1],destination=b1.ports[2])
            b2.move([0,p2[1]-p1[1]-radiusEff*2])
            R1=pg.route(port1=total.ports['t1'],port2=b1.ports[1],layer=layer)
            r1=total.add_ref(R1)
            R2=pg.route(port1=b1.ports[2],port2=b2.ports[1],layer=layer)
            r2=total.add_ref(R2)
            total.add_port(name=1,port=r1.ports[1])
            total.add_port(name=2,port=b2.ports[2])  
        #second quadrant target
        if (p2[1] > p1[1]) & (p2[0] < p1[0]):
            if bendType == 'circular':
                B1=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=0,theta=90)
                B2=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=90,theta=90)
                B3=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=180,theta=-90)
                B4=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=90,theta=-90)
                radiusEff=radius
            if bendType == 'gradual':
                B1=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=0,direction='ccw')
                B2=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=90,direction='ccw')
                B3=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=180,direction='cw')
                B4=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=90,direction='cw')
                radiusEff=B1.xsize-wWg/2
            b1=total.add_ref(B1)
            b2=total.add_ref(B2)
            b3=total.add_ref(B3)
            b4=total.add_ref(B4)
            
            
            b1.connect(port=b1.ports[1],destination=total.ports['t1'])

            b2.connect(port=b2.ports[1],destination=b1.ports[2])
            b2.move([0,p2[1]-p1[1]-radiusEff*4])
            R1=pg.route(port1=b1.ports[2],port2=b2.ports[1],layer=layer)
            r1=total.add_ref(R1)
            b3.connect(port=b3.ports[1],destination=b2.ports[2])
            b3.move([p2[0]-p1[0],0])
            R2=pg.route(port1=b2.ports[2],port2=b3.ports[1],layer=layer)
            r2=total.add_ref(R2)            
            
            b4.connect(port=b4.ports[1],destination=b3.ports[2])
            
            total.add_port(name=1,port=r1.ports[1])
            total.add_port(name=2,port=b4.ports[2])
         #third quadrant target
        if (p2[1] < p1[1]) & (p2[0] < p1[0]):
            if bendType == 'circular':
                B1=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=0,theta=-90)
                B2=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=-90,theta=-90)
                B3=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=-180,theta=90)
                B4=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=-90,theta=90)
                radiusEff=radius
            if bendType == 'gradual':
                B1=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=0,direction='cw')
                B2=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=-90,direction='cw')
                B3=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=-180,direction='ccw')
                B4=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=-90,direction='ccw')
                radiusEff=B1.xsize-wWg/2
            b1=total.add_ref(B1)
            b2=total.add_ref(B2)
            b3=total.add_ref(B3)
            b4=total.add_ref(B4)
            
            
            b1.connect(port=b1.ports[1],destination=total.ports['t1'])

            b2.connect(port=b2.ports[1],destination=b1.ports[2])
            b2.move([0,p2[1]-p1[1]+radiusEff*4])
            R1=pg.route(port1=b1.ports[2],port2=b2.ports[1],layer=layer)
            r1=total.add_ref(R1)
            b3.connect(port=b3.ports[1],destination=b2.ports[2])
            b3.move([p2[0]-p1[0],0])
            R2=pg.route(port1=b2.ports[2],port2=b3.ports[1],layer=layer)
            r2=total.add_ref(R2)            
            
            b4.connect(port=b4.ports[1],destination=b3.ports[2])
            
            total.add_port(name=1,port=r1.ports[1])
            total.add_port(name=2,port=b4.ports[2])
        #fourth quadrant target
        if (p2[1] < p1[1]) & (p2[0] > p1[0]):
            if bendType == 'circular':
                B1=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=0,theta=-90)
                B2=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=-90,theta=90)
                radiusEff=radius
            if bendType == 'gradual':
                B1=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=0,direction='cw')
                B2=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=-90,direction='ccw')
                radiusEff=B1.xsize-wWg/2
            b1=total.add_ref(B1)
            b2=total.add_ref(B2)
            
            
            b1.connect(port=b1.ports[1],destination=total.ports['t1'])
            b1.move([p2[0]-p1[0]-radiusEff*2,0])
            b2.connect(port=b2.ports[1],destination=b1.ports[2])
            b2.move([0,p2[1]-p1[1]+radiusEff*2])
            R1=pg.route(port1=total.ports['t1'],port2=b1.ports[1],layer=layer)
            r1=total.add_ref(R1)
            R2=pg.route(port1=b1.ports[2],port2=b2.ports[1],layer=layer)
            r2=total.add_ref(R2)
            total.add_port(name=1,port=r1.ports[1])
            total.add_port(name=2,port=b2.ports[2])            
                        
    total.rotate(angle =  port1.orientation, center = p1)
    total.move(origin = total.ports['t1'], destination = port1)
    return total

def routeManhattan90(
    port1,
    port2,
    bendType='circular',
    layer=0,
    radius=20
    ):
    #this is a subroutine of routeManhattan() and should not be used by itself.
    total = Device()
    wWg=port1.width
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
    total.add_port(name='t1',midpoint=[0,0],orientation=0,width=wWg)

    #CHECK THIS


    #first quadrant target, route upward
    if (p2[1] > p1[1]) & (p2[0] > p1[0]):
        total.add_port(name='t2',midpoint=list(np.subtract(p2,p1)),orientation=-90,width=wWg)
        if bendType == 'circular':
            B1=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=0,theta=90)
            radiusEff=radius
        if bendType == 'gradual':
            B1=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=0,direction='ccw')
            radiusEff=B1.xsize-wWg/2
        b1=total.add_ref(B1)
        b1.connect(port=b1.ports[1],destination=total.ports['t1'])
        b1.move([p2[0]-p1[0]-radiusEff,0])

        R1 = pg.route(port1=total.ports['t1'],port2=b1.ports[1],layer=layer)
        R2 = pg.route(port1=b1.ports[2],port2=total.ports['t2'],layer=layer)
        r1 = total.add_ref(R1)
        r2 = total.add_ref(R2)
        total.add_port(name=1,port=r1.ports[1])
        total.add_port(name=2,port=r2.ports[2])  
    
    #fourth quadrant target, route downward
    if (p2[1] < p1[1]) & (p2[0] > p1[0]):
        total.add_port(name='t2',midpoint=list(np.subtract(p2,p1)),orientation=90,width=wWg)
        if bendType == 'circular':
            B1=pg.arc(radius=radius,width=wWg,layer=layer,angle_resolution=1,start_angle=0,theta=-90)
            radiusEff=radius
        if bendType == 'gradual':
            B1=gradualBend(radius=radius,wWg=wWg,layer=layer,start_angle=0,direction='cw')
            radiusEff=B1.xsize-wWg/2
        b1=total.add_ref(B1)
        b1.connect(port=b1.ports[1],destination=total.ports['t1'])
        b1.move([p2[0]-p1[0]-radiusEff,0])
        R1 = pg.route(port1=total.ports['t1'],port2=b1.ports[1],layer=layer)
        R2 = pg.route(port1=b1.ports[2],port2=total.ports['t2'],layer=layer)
        r1 = total.add_ref(R1)
        r2 = total.add_ref(R2)
        total.add_port(name=1,port=r1.ports[1])
        total.add_port(name=2,port=r2.ports[2])  
    total.rotate(angle =  port1.orientation, center = p1)
    total.move(origin = total.ports['t1'], destination = port1)    
    
    return total


def routeManhattanAuto(
    ports,    
    bendType='circular',
    layer=0,
    radius=20    
    ):
    #routes a one-dimensional array of ports using manhattan algorithm
    #give it a series of ports to route to in a continuous list.
    #accepts same parameters as ordinary routeManhattan to determine bending
    total=Device()
    for x in xrange(int(np.floor(len(ports)/2))+1):
        R = routeManhattan(port1=ports[x],port2=ports[x+1],bendType=bendType,layer=layer,radius=radius)
        r = total.add_ref(R)

    return total
    

"""
The code below is a test-case for routeManhattan().

D=Device()
A=pg.compass()
A.add_port(name=1,port=A.ports['N'])
A.add_port(name=2,port=A.ports['E'])
A.add_port(name=3,port=A.ports['S'])
A.add_port(name=4,port=A.ports['W'])
points=[]
points.append((300,300))
points.append((-300,300))
points.append((-300,-300))
points.append((300,-300))

xoff=0
yoff=0
for x in xrange(4):
    for y in xrange(4):
        for z in xrange(4):
            a = D.add_ref(A)
            b = D.add_ref(A)
            a.center=(xoff,yoff)
            b.center=(xoff+(points[y])[0],yoff+(points[y])[1])
            C = routeManhattan(bendType='gradual',port1=a.ports[z+1],port2=b.ports[x+1])
            c=D.add_ref(C)
            yoff+=600
        yoff+=600
    xoff+=600
    yoff=0

quickplot(D)
"""