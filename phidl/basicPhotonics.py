from __future__ import division # Makes it so 1/4 = 0.25 instead of zero


from phidl import Device, quickplot
import numpy as np
import phidl.geometry as pg
import gdspy

## parameters for rings
#radius = 10
#gaps = [0.1, 0.2, 0.3]
#wRing = 0.5
#wWg = 0.4
#dR = 0.015
#period = radius*3

def polygon(xcoords = [0, 0, 10, 10], ycoords = [0, 10, 10, 0], layer = 0):
    # returns a polygon with ports on all edges
    P = Device('polygon')
    P.add_polygon([xcoords, ycoords], layer = layer)
    n = len(xcoords)
    xcoords.append(xcoords[0])
    ycoords.append(ycoords[0]) 
 #determine if clockwise or counterclockwise
    cc = 0     
    for i in range(0,n):
        cc += ((xcoords[i+1]-xcoords[i])*(ycoords[i+1]+ycoords[i]))
            
    for i in range(0,n):
        midpoint_n = [(xcoords[i+1]+xcoords[i])/2, (ycoords[i+1]+ycoords[i])/2]
        test1 = xcoords[i+1]-xcoords[i]
        test2 = ycoords[i+1]-ycoords[i]
        
        orientation_n = np.arctan2(np.sign(cc)*(xcoords[i+1]-xcoords[i]),np.sign(cc)*(ycoords[i]-ycoords[i+1]))*180/np.pi
            
        width_n = np.sqrt((xcoords[i+1]-xcoords[i])**2+(ycoords[i+1]-ycoords[i])**2)    
        
        P.add_port(name = str(i), midpoint = midpoint_n, width = width_n, orientation = orientation_n)
    
    return P
    
def grating(nperiods = 20, period = 0.75, ff = 0.5, wGrating = 20, lTaper = 10, wWg = 0.4, partialetch = 0):
    #returns a fiber grating
    G = Device('grating')

# make the deep etched grating
    if partialetch is 0:
        # make the grating teeth
        for i in range(nperiods):
            cgrating = G.add_ref(pg.compass(size=[period*ff,wGrating]), layer = 0)
            cgrating.x+=i*period
            
        # make the taper
        tgrating = G.add_ref(pg.taper(length = lTaper, width1 = wGrating, width2 = wWg, port = None, layer = 0))
        tgrating.xmin = cgrating.xmax
        # define the port of the grating
        p = G.add_port(port = tgrating.ports[2], name = 1)
# make a partially etched grating
    if partialetch is 1:
        # hard coded overlap
            partetch_overhang = 5
            # make the etched areas (opposite to teeth)
            for i in range(nperiods):
                cgrating = G.add_ref(pg.compass(size=[period*(1-ff),wGrating+partetch_overhang*2]), layer = 1)
                cgrating.x+=i*period
                        # define the port of the grating
            p = G.add_port(port = cgrating.ports['E'], name = 1)
            p.midpoint=p.midpoint+np.array([(1-ff)*period,0])
                
        #draw the deep etched square around the grating
            deepbox = G.add_ref(pg.compass(size=[nperiods*period, wGrating]), layer=0)    
    return G

def ccRings(radius = 10, gaps = [0.1, 0.2, 0.3], wRing = 0.5, wWg = 0.4, dR = 0.15, period = 30, GratingDevice = gdspy.Cell):
# number of rings defined by the length of the gaps vector    
    nrings = len(gaps)
    lWg = (nrings + 1)*period
    
    D = Device('critically coupled rings')
# make the main bus
    wg = D.add_ref(pg.compass(size=[lWg, wWg]))
    R = Device();
 # make the rings with different gaps and radii   
    for i, g in enumerate(gaps):
        r = R.add_ref(pg.ring(radius = radius + dR*i, width = wRing, angle_resolution = 1, layer = 0)) 
        r.move([period*i,0])
        r.ymax = wg.ymin - g
# put the rings in the main device    
    R.x = D.x
    D.add_ref(R)
# add the gratings    
    g = D.add_ref(GratingDevice)
    g2 = D.add_ref(GratingDevice)
# define port connections   
    g.connect(port = 1, destination = wg.ports['W'])
    g.xmax = wg.xmin
    g2.connect(port = 1, destination = wg.ports['E'])
    g2.xin = wg.xmax
    
# route between connected ports with tapers  
    D.add_ref(pg.route(port1 = g.ports[1], port2 = wg.ports['W']))
    D.add_ref(pg.route(port1 = g2.ports[1], port2 = wg.ports['E']))
    
    return D
    
def lossRings(radius = 10, gaps = [0.1, 0.2, 0.3], wRing = 0.5, wWg = 0.4, dR = 0.15, period = 30, GratingDevice = gdspy.Cell):
    
    D = Device("loss rings")
    nrings = len(gaps)
    lWg = (nrings + 1)*period
    
    wg = D.add_ref(pg.compass(size=[lWg, wWg]))
    
    g = D.add_ref(GratingDevice)
    g2 = D.add_ref(GratingDevice)
    
    g.connect(port = 1, destination = wg.ports['W'])
    g.xmax = wg.xmin
    g2.connect(port = 1, destination = wg.ports['E'])
    g2.xmin = wg.xmax
    
    D.add_ref(pg.route(port1 = g.ports[1], port2 = wg.ports['W']))
    D.add_ref(pg.route(port1 = g2.ports[1], port2 = wg.ports['E']))
    
    R = Device();
    
    for i, g in enumerate(gaps):
        r = R.add_ref(pg.ring(radius = radius + dR*i, width = wRing, angle_resolution = 1, layer = 0)) 
        r.move([period*i,0])
        r.ymax = wg.ymin - g
        rwg = R.add_ref(pg.compass(size=[10, wWg]))
        rwg.move([period*i, 0])
        rwg.ymax = r.ymin-g
        rarc = R.add_ref(pg.arc(radius = 10, width = wWg, theta = 180, start_angle = 90, angle_resolution = 2.5, layer = 0))
        rarc.xmax = rwg.xmin
        rarc.ymax = rwg.ymax
        rtap = R.add_ref(pg.taper(length = 10, width1 = wWg, width2 = 0.1, port = None, layer = 0))
        rtap.xmin = rwg.xmax
        rtap.ymax = rwg.ymax
        
        rwg2 = R.add_ref(pg.compass(size=[10, wWg]))
        rwg2.xmin = rarc.xmax
        rwg2.ymin = rarc.ymin
        
        g3 = R.add_ref(GratingDevice)
        g3.xmin = rwg2.xmax
        g3.connect(port = 1, destination = rwg2.ports['E'])
          
    R.x = D.x
    D.add_ref(R)
    
    return D
    
def adiabaticBeamsplitter(interaction_length = 10,gap1 = 1, gap2 = 0.1, 
                          port1 = 0.4, port2 = 0.5, port3 = 0.4, port4 = 0.4,
                          length_straight_biggapside = 5,hsines = 3, lsines = 6,
                          length_straight_aftersinebend = 2, wWg = 0.7,
                          taper_length = 5):
    #
    D = Device("adiabatic beam splitter")
    #
    # start with the actual beamsplitter part
    xpts_upper = [0, 0, interaction_length, interaction_length]
    ypts_upper = [0, port1, port3, 0]
    
    wg_upper = D.add_ref(bP.polygon(xcoords = xpts_upper, ycoords = ypts_upper, layer = 0))
    #
    xpts_lower = [0, 0, interaction_length, interaction_length]
    ypts_lower = [-gap1, -gap1-port2, -gap2-port4, -gap2]
    wg_lower = D.add_ref(bP.polygon(xcoords = xpts_lower, ycoords = ypts_lower, layer = 0))
    
    # add a straight section on the left
    wg_straight_biggapside1 = D.add_ref(pg.compass(size=[length_straight_biggapside, port1]))
    wg_straight_biggapside2 = D.add_ref(pg.compass(size=[length_straight_biggapside, port2]))
    
    # connect to the beamsplitter
    wg_straight_biggapside1.connect(port = 'E', destination = wg_upper.ports['0'])
    wg_straight_biggapside2.connect(port = 'E', destination = wg_lower.ports['0'])
    
    #draw straight sections that will come after the sinebends
    wg_straight_aftersinebend = D.add_ref(pg.compass(size=[length_straight_aftersinebend, port1]))
    wg_straight_aftersinebend2 = D.add_ref(pg.compass(size=[length_straight_aftersinebend, port2]))
    wg_straight_aftersinebend3 = D.add_ref(pg.compass(size=[length_straight_aftersinebend, port3]))
    wg_straight_aftersinebend4 = D.add_ref(pg.compass(size=[length_straight_aftersinebend, port4]))
    
    #locate the straight sections after the sine bends
    wg_straight_aftersinebend.center = [wg_straight_biggapside1.xmin-lsines, wg_straight_biggapside1.center[1]+hsines]
    wg_straight_aftersinebend2.center = [wg_straight_biggapside2.xmin-lsines, wg_straight_biggapside2.center[1]-hsines]
    wg_straight_aftersinebend3.center = [wg_upper.xmax+lsines, wg_upper.center[1]+hsines]
    wg_straight_aftersinebend4.center = [wg_lower.xmax+lsines, wg_lower.center[1]-hsines]
    
    #route with sine bends
    D.add_ref(pg.route(port1 = wg_straight_aftersinebend.ports['E'], port2 = wg_straight_biggapside1.ports['W'], path_type = 'sine'))
    D.add_ref(pg.route(port1 = wg_straight_aftersinebend2.ports['E'], port2 = wg_straight_biggapside2.ports['W'], path_type = 'sine'))
    D.add_ref(pg.route(port1 = wg_straight_aftersinebend3.ports['W'], port2 = wg_upper.ports['2'], path_type = 'sine'))
    D.add_ref(pg.route(port1 = wg_straight_aftersinebend4.ports['W'], port2 = wg_lower.ports['2'], path_type = 'sine'))
    
    #add tapers at either end
    T = pg.taper(length = taper_length, width1 = port1, width2 = wWg, layer = 0)
    t1 = D.add_ref(T)
    t2 = D.add_ref(T)
    t3 = D.add_ref(T)
    t4 = D.add_ref(T)
    
    t1.connect(port = 1, destination = wg_straight_aftersinebend.ports['W'])
    t2.connect(port = 1, destination = wg_straight_aftersinebend2.ports['W'])
    t3.connect(port = 1, destination = wg_straight_aftersinebend3.ports['E'])
    t4.connect(port = 1, destination = wg_straight_aftersinebend4.ports['E'])
    
    D.add_port(port = t1.ports[2], name = 'port 1')
    D.add_port(port = t2.ports[2], name = 'port 2')
    D.add_port(port = t3.ports[2], name = 'port 3')
    D.add_port(port = t4.ports[2], name = 'port 4')
    
    return D

def beamTap():
    pass

def MZI():
    pass

def wgSNSPD():
    pass

def LED():
    pass

def hidra():
    pass

def hexapod():
    pass