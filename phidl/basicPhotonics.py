from __future__ import division # Makes it so 1/4 = 0.25 instead of zero


from phidl import Device, quickplot
import numpy as np
import phidl.geometry as pg
import gdspy
from scipy.interpolate import interp1d

## parameters for rings
#radius = 10
#gaps = [0.1, 0.2, 0.3]
#wRing = 0.5
#wWg = 0.4
#dR = 0.015
#period = radius*3

def polygon(xcoords=[-1,-1, 0, 0],
            ycoords = [0, 1, 1, 0],
            layer = 0):
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
        orientation_n = np.arctan2(np.sign(cc)*(xcoords[i+1]-xcoords[i]),np.sign(cc)*(ycoords[i]-ycoords[i+1]))*180/np.pi           
        width_n = np.abs(np.sqrt((xcoords[i+1]-xcoords[i])**2+(ycoords[i+1]-ycoords[i])**2))    
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
    
def pad(width = 100,height = 300,poOffset = 20,padLayer = 2,poLayer = 3):
    D = Device()
    pad = D.add_ref(pg.compass(size = [width, height], layer = padLayer))
    pad_opening = D.add_ref(pg.compass(size = [width-2*poOffset, height-2*poOffset], layer = poLayer))
    D.add_port(port=pad.ports['S'], name = 1)
    return D
    
def dblpad(gap = 10, PadDevice = gdspy.Cell):
    D = Device()
    Pad = pad()
    pad1 = D.add_ref(PadDevice)
    pad2 = D.add_ref(PadDevice)
    pad1.center = pad2.center
    pad1.xmax = pad2.xmin-gap
    D.add_port(port=pad1.ports[1], name = 1)
    D.add_port(port=pad2.ports[1], name = 2)
    return D
    
def ccRings(radius = 10, gaps = [0.1, 0.2, 0.3], wRing = 0.5, wWg = 0.4, dR = 0.15, period = 30, GratingDevice = gdspy.Cell, wgLayer=0):
# number of rings defined by the length of the gaps vector    
    nrings = len(gaps)
    lWg = (nrings + 1)*period
    
    D = Device('critically coupled rings')
# make the main bus
    wg = D.add_ref(pg.compass(size=[lWg, wWg]))
    R = Device();
 # make the rings with different gaps and radii   
    for i, g in enumerate(gaps):
        r = R.add_ref(pg.ring(radius = radius + dR*i, width = wRing, angle_resolution = 1, wgLayer = 0)) 
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
    
def lossRings(radius = 10, gaps = [0.1, 0.2, 0.3], wRing = 0.5, wWg = 0.4, dR = 0.15, period = 30, GratingDevice = gdspy.Cell, wgLayer = 0):
    
    D = Device("loss rings")
    nrings = len(gaps)
    lWg = (nrings + 1)*period
    
    wg = D.add_ref(pg.compass(size=[lWg, wWg], layer = wgLayer))
    
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
        r = R.add_ref(pg.ring(radius = radius + dR*i, width = wRing, angle_resolution = 1, layer = wgLayer)) 
        r.move([period*i,0])
        r.ymax = wg.ymin - g
        rwg = R.add_ref(pg.compass(size=[10, wWg], layer = wgLayer))
        rwg.move([period*i, 0])
        rwg.ymax = r.ymin-g
        rarc = R.add_ref(pg.arc(radius = 10, width = wWg, theta = 180, start_angle = 90, angle_resolution = 2.5, layer = wgLayer))
        rarc.xmax = rwg.xmin
        rarc.ymax = rwg.ymax
        rtap = R.add_ref(pg.taper(length = 10, width1 = wWg, width2 = 0.1, port = None, layer = wgLayer))
        rtap.xmin = rwg.xmax
        rtap.ymax = rwg.ymax
        
        rwg2 = R.add_ref(pg.compass(size=[10, wWg],layer = wgLayer))
        rwg2.xmin = rarc.xmax
        rwg2.ymin = rarc.ymin
        
        g3 = R.add_ref(GratingDevice)
        g3.xmin = rwg2.xmax
        g3.connect(port = 1, destination = rwg2.ports['E'])
          
    R.x = D.x
    D.add_ref(R)
    
    return D
    
def adiabaticBeamsplitter(interaction_length = 10, gap1 = 1, gap2 = 0.1, 
                          port1 = 0.4, port2 = 0.5, port3 = 0.4, port4 = 0.4,
                          hSines = 3, rMin = 10, **kwargs):

    lSines = np.sqrt((np.pi**2)*hSines*rMin/2)
    #
    D = Device("adiabatic beam splitter")
    #
    # start with the actual beamsplitter part
    xpts_upper = [0, 0, interaction_length, interaction_length]
    ypts_upper = [0, port1, port3, 0]
    
    wg_upper = D.add_ref(polygon(xcoords = xpts_upper, ycoords = ypts_upper, layer = 0))
    #
    xpts_lower = [0, 0, interaction_length, interaction_length]
    ypts_lower = [-gap1, -gap1-port2, -gap2-port4, -gap2]
    wg_lower = D.add_ref(polygon(xcoords = xpts_lower, ycoords = ypts_lower, layer = 0))
    
    #locate the straight sections after the sine bends
    P = Device('ports')
    P.add_port(name = 'port 1', midpoint = [wg_upper.xmin-lSines, wg_upper.center[1]+hSines], width = port1, orientation = 0)
    P.add_port(name = 'port 2', midpoint = [wg_lower.xmin-lSines, wg_lower.center[1]-hSines], width = port1, orientation = 0)
    P.add_port(name = 'port 3', midpoint = [wg_upper.xmax+lSines, wg_upper.center[1]+hSines], width = port1, orientation = 180)
    P.add_port(name = 'port 4', midpoint = [wg_lower.xmax+lSines, wg_lower.center[1]-hSines], width = port1, orientation = 180)
    route1 = D.add_ref(pg.route(port1 = P.ports['port 1'], port2 = wg_upper.ports['0'], path_type = 'sine'))
    route2 = D.add_ref(pg.route(port1 = P.ports['port 2'], port2 = wg_lower.ports['0'], path_type = 'sine'))
    route3 = D.add_ref(pg.route(port1 = P.ports['port 3'], port2 = wg_upper.ports['2'], path_type = 'sine'))
    route4 = D.add_ref(pg.route(port1 = P.ports['port 4'], port2 = wg_lower.ports['2'], path_type = 'sine'))
        
    n = len(kwargs)
    dest = {0: route1.ports[1], 1: route2.ports[1], 2: route3.ports[1], 3: route4.ports[1]}
    
    for i in range(0, n):
        dP = D.add_ref(kwargs[kwargs.keys()[i]])
        dP.connect(port = dP.ports[1], destination = dest[i])
        D.add_ref(dP)
    for i in range(n, 4):
        D.add_port(port = dest[i], name = i+1)
    
    return D

def beamTap(interaction_length = 10, hSines = 5, rMin = 10, gap = 0.5, wWg= 0.4, **kwargs): 
                                   
    lSines = np.sqrt(hSines*np.pi**2*rMin/2)    
    D = Device()
    
    WG = pg.compass(size=[interaction_length, wWg])
    wg1 = D.add_ref(WG)
    wg2 = D.add_ref(WG)
    
    wg1.center = wg2.center
    wg1.ymin = wg1.ymax + gap
    
    P = Device()
    port1 = P.add_port(name = 1, midpoint = wg1.ports['W'].midpoint+[-1*lSines, hSines], width = wWg, orientation = 0)
    port2 = P.add_port(name = 2, midpoint = wg2.ports['W'].midpoint+[-1*lSines, 0], width = wWg, orientation = 0)
    port3 = P.add_port(name = 3, midpoint = wg1.ports['E'].midpoint+[lSines,hSines], width = wWg, orientation = 180)
    port4 = P.add_port(name = 4, midpoint = wg2.ports['E'].midpoint+[lSines, 0], width = wWg, orientation = 180)
    
    route1 = D.add_ref(pg.route(port1 = port1, port2 = wg1.ports['W']))
    route2 = D.add_ref(pg.route(port1 = port2, port2 = wg2.ports['W']))
    route3 = D.add_ref(pg.route(port1 = port3, port2 = wg1.ports['E']))
    route4 = D.add_ref(pg.route(port1 = port4, port2 = wg2.ports['E']))
    D.add_ref(P)
    
    n = len(kwargs)
    dest = {0: route1.ports[1], 1: route2.ports[1], 2: route3.ports[1], 3: route4.ports[1]}
    
    for i in range(0, n):
        dP = D.add_ref(kwargs[kwargs.keys()[i]])
        dP.connect(port = dP.ports[1], destination = dest[i])
        D.add_ref(dP)
    for i in range(n, 4):
        D.add_port(port = dest[i], name = i+1)

    return D 

def MZI(FSR = 0.05, ng = 4, rMin = 10, wavelength = 1.55, BS = gdspy.Cell, **kwargs):
    # MZI. input the FSR and ng and it calculates the largest hSines for a given minimum radius of curvature.
# optionally you can input devices for the four ports of the MZI. If you leave them empty it will just put ports on.


    # for a given ratio of hSines and lSines we can analytically estimate the FSR based 
    #on the length of a sine from 2*pi to 0 = 7.64. For some reason here we have a scale factor to 6.66
    dL = wavelength**2/ng/FSR
    # ok, but we still have to make our best initial guess at lSines
    lSines_approx = dL/(9/2/np.pi-1)
    # based on that guess + the rMin we calculate the ratio of hSines/lSines
    r = 2*lSines_approx/np.pi**2/rMin
    # now using that ratio we actually calculate the factor F, which is an elliptic integral
    F = get_const_MZI(r)
    # now we can recalculate lSines and hSines
    lSines = dL/(F/(2*np.pi)-1)
    hSines = r*lSines
    
    # build the device
    D = Device()
    bS1 = D.add_ref(BS)
    bS2 = D.add_ref(BS)
    bS1.reflect(p1 = bS1.ports[3], p2 = bS1.ports[4])
    P = Device()
    P.add_port(name = 1, midpoint = bS1.ports[3].midpoint+[lSines/2,hSines], width = bS1.ports[3].width, orientation = 180)
    P.add_port(name = 2, midpoint = bS1.ports[3].midpoint+[lSines/2,hSines], width = bS1.ports[3].width, orientation = 0)
    D.add_ref(P)
    bS1.movex(lSines)
    route1 = D.add_ref(pg.route(port1 = P.ports[1], port2 = bS2.ports[3], path_type = 'sine'))
    route2 = D.add_ref(pg.route(port1 = P.ports[2], port2 = bS1.ports[3], path_type = 'sine'))
    route3 = D.add_ref(pg.route(port1 = bS1.ports[4], port2 = bS2.ports[4]))
    calc_length = route1.meta['length']+route2.meta['length']-route3.meta['length']
    calc_fsr = wavelength**2/ng/calc_length
    D.meta['Calculated FSR'] = calc_fsr
    # now we put either devices or ports on the 4 outputs of the MZI
    n = len(kwargs)
    dest = {0: bS2.ports[1], 1: bS2.ports[2], 2: bS1.ports[1], 3: bS1.ports[2]}
    
    for i in range(0, n):
        dP = D.add_ref(kwargs[kwargs.keys()[i]])
        dP.connect(port = dP.ports[1], destination = dest[i])
        D.add_ref(dP)
    for i in range(n, 4):
        D.add_port(port = dest[i], name = i+1)

    return D

def wgSNSPD(meanderWireWidth = 0.4, meanderPitch = 0.8, meanderLength = 50, 
            nsquares = 5000, wgNwWidth = 0.1, wgNwPitch = 0.3, wgNwLength = 100, 
            padDistance = 500, landingPadOffset = 10, 
            nwLayer = 6, wgLayer = 1, metalLayer = 2, DblPadDevice = gdspy.Cell):
    
    
    D = Device()
    nsquares_per_turn = (meanderLength/2-(meanderWireWidth+meanderPitch))/meanderWireWidth + 1
    meanderOffset = 4*meanderPitch
    # the number of turns that we will actually make. Must be an odd number if the leads need to come out the same side
    nturns = np.floor(nsquares/nsquares_per_turn)-1
    
    meanderWidth = nturns*meanderPitch
    SNSPD = pg.snspd(wire_width = meanderWireWidth, wire_pitch = meanderPitch, size = (meanderLength,meanderWidth),
              terminals_same_side = True, layer = nwLayer)
    SNSPD.add_port(name = 3, midpoint = [SNSPD.xmin+meanderWireWidth/2, SNSPD.ymin], width = meanderWireWidth, orientation = -90)
    SNSPD.add_port(name = 4, midpoint = [SNSPD.xmin+meanderWireWidth/2, SNSPD.ymax], width = meanderWireWidth, orientation = 90)
    meander = D.add_ref(SNSPD)
    wgNw = D.add_ref(pg.optimal_hairpin(width = wgNwWidth, pitch = wgNwPitch, length = wgNwLength, layer = nwLayer))
    wgNw.reflect(p1 = wgNw.ports[1].midpoint, p2 = wgNw.ports[2].midpoint)
    wgNw.xmax = meander.xmin - meanderWireWidth
    wgNw.ymax = meander.ymin-meanderOffset
    
    # connector between the hairpin and the meander
    Hairpin2meander = pg.rectangle(size = [meanderWireWidth, meanderOffset-meanderWireWidth+wgNwWidth], layer = nwLayer)
    Hairpin2meander.add_port(name = 1, midpoint = [meanderWireWidth/2, Hairpin2meander.ymin], width = meanderWireWidth, orientation = -90)
    Hairpin2meander.add_port(name = 2, midpoint = [meanderWireWidth, Hairpin2meander.ymax-wgNwWidth/2], width = wgNwWidth, orientation = 0)
    hairpin2meander = D.add_ref(Hairpin2meander)
    hairpin2meander.connect(port = hairpin2meander.ports[1], destination = SNSPD.ports[3])
    wgNw.connect(port = wgNw.ports[1], destination = hairpin2meander.ports[2])
    
    # vertical fill rectangles
    meander_size = [meander.bbox[1][0]-meander.bbox[0][1], meander.bbox[0][1]-meander.bbox[1][1]]
    R1 = pg.rectangle(size = [meanderWireWidth, np.abs(meander_size[1])+ 6*meanderPitch], layer = nwLayer)
    
    for i in range(0, 3):
        fill = D.add_ref(R1)
        fill2 = D.add_ref(R1)
        fill.xmin = meander.xmin - (i+1)*meanderPitch
        fill.y = meander.y
        fill2.xmax = meander.xmax + (i+1)*meanderPitch
        fill2.y = meander.y
        
    # horizontal fill rectangles
    meander_size = [meander.xmax-meander.xmin, meander.bbox[0][1]-meander.bbox[1][1]]
    R2 = pg.rectangle(size = [np.abs(meander_size[0]) - meanderPitch, meanderWireWidth], layer = nwLayer)
    
    for i in range(0, 3):
        fill = D.add_ref(R2)
        fill2 = D.add_ref(R2)
        fill.ymin = meander.ymin - (i+1)*meanderPitch
        fill.x = meander.x +meanderPitch-meanderWireWidth
        fill2.ymax = meander.ymax + (i+1)*meanderPitch
        fill2.x = meander.x+meanderPitch-meanderWireWidth
    
    # connectors between nw and pad and meander and pad
    
    R3 = pg.rectangle(size = [meanderWireWidth, meanderPitch*4-meanderWireWidth + 5])
    R4 = pg.rectangle(size = [meanderWireWidth, 5])
    R3.add_port(name = 1, midpoint = [meanderWireWidth/2, 0], width = meanderWireWidth, orientation = -90)
    R3.add_port(name = 2, midpoint = [meanderWireWidth, R3.ymax-5/2], width = 5, orientation = 0)
    R4.add_port(name = 1, midpoint = [0, R4.ymax - wgNwWidth/2], width = wgNwWidth, orientation = 180)  
    R4.add_port(name = 2, midpoint = [meanderWireWidth, 5/2], width = 5, orientation = 0)
    hairpin2pads = D.add_ref(R4)
    meander2pads = D.add_ref(R3)
    hairpin2pads.connect(port = 1, destination = wgNw.ports[2])
    meander2pads.connect(port = 1, destination = meander.ports[4])
    
    # nw layer pads
    
    R4 = pg.compass(size = [np.abs(meander_size[0])/2, 5], layer = nwLayer)
    nwPad1 = D.add_ref(R4)
    nwPad2= D.add_ref(R4)
    nwPad1.connect(port = 'W', destination = meander2pads.ports[2])
    nwPad2.connect(port = 'W', destination = hairpin2pads.ports[2])
    
    # metal layer pads
    M1 = pg.compass(size = [np.abs(meander_size[0])/2, 5], layer = metalLayer)
    metalWire1 = D.add_ref(M1)
    metalWire2= D.add_ref(M1)
    metalWire1.center = nwPad1.center
    metalWire2.center = nwPad2.center
    
    pads = D.add_ref(DblPadDevice)
    pads.rotate(angle = -90)
    pads.xmin = meander.xmax + padDistance
    pads.y = meander.y
    Route1 = pg.route(port1 = pads.ports[1], port2 = metalWire1.ports['E'], path_type = 'straight', layer = metalLayer)
    Route2 = pg.route(port1 = pads.ports[2], port2 = metalWire2.ports['E'], path_type = 'straight', layer = metalLayer)
    D.add_ref(Route1)
    D.add_ref(Route2)
    
    # wg layer
    
    # wg layer wg
    
    wg = D.add_ref(pg.compass(size = [wgNwLength + wgNwPitch, wgNwPitch*2], layer = wgLayer)) 
    wg.xmax = wgNw.xmax
    wg.y = wgNw.y
    D.add_port(name = 1, port = wg.ports['W'])
    
    # wg layer landing pad
    # padside
    size = [pads.xmax - pads.xmin + landingPadOffset, pads.ymax - pads.ymin + landingPadOffset] 
    landingPad1 = D.add_ref(pg.compass(size = size, layer = wgLayer))   
    landingPad1.center = pads.center
    
    # nw side
    size = [nwPad1.xmax - hairpin2pads.xmin + 1, nwPad1.ymax - nwPad2.ymin + 1]
    landingPad2 = D.add_ref(pg.compass(size = size, layer = wgLayer))   
    landingPad2.y = meander.y
    landingPad2.xmin = nwPad1.xmin - 1
    Route3 = pg.route(port1 = landingPad1.ports['W'], port2 = landingPad2.ports['E'], layer = wgLayer)
    D.add_ref(Route3)
    return D

def LED(wWg=1, lWg=10, wDopeOffset=0.2, wDope=5, wE=1, wTaper = 0.4, lTaper = 10, 
        metalInset = 0.2, padDeviceDistance = [50,0], padWireWidth = 0.5,
        wgLayer = 0, pLayer = 1, nLayer = 2, wLayer = 3, padtaperLayer = 4, DblPadDevice = gdspy.Cell):

    D = Device("LED")
    
    wg = D.add_ref(pg.compass(size=[lWg,wWg],layer=wgLayer))
    wRegion = D.add_ref(pg.compass(size=[lWg,wE],layer=wLayer))
    pRegion = D.add_ref(pg.compass(size=[lWg,wDope],layer=pLayer))
    nRegion = D.add_ref(pg.compass(size=[lWg,wDope],layer=nLayer))
    taper = D.add_ref(pg.taper(length = lTaper, width1 = wWg, width2 = wTaper))
    PW = pg.compass(size = [lWg, padWireWidth], layer = padtaperLayer)
    padWire1 = D.add_ref(PW)
    padWire2 = D.add_ref(PW)
    pads = D.add_ref(DblPadDevice)
    pads.rotate(angle = 90)
    
    taper.xmin = wg.xmax
    wg.connect(port = 'W', destination = taper.ports[1])
    wg.center = wRegion.center
    pRegion.ymin = wRegion.ymax + wDopeOffset
    pRegion.center[0] = wRegion.center[0]
    nRegion.ymax = wRegion.ymin - wDopeOffset
    nRegion.center[0] = wRegion.center[0]
    
    padWire1.ymax = pRegion.ymax - metalInset
    padWire2.ymax = nRegion.ymin + metalInset
    pads.center = wg.center
    pads.xmax = wg.xmin - padDeviceDistance[0]
    pads.movey = padDeviceDistance[1]
    D.add_ref(pg.route(port1 = padWire1.ports['W'], port2 =pads.ports[2], layer = padtaperLayer,path_type = 'straight'))
    D.add_ref(pg.route(port1 = padWire2.ports['W'], port2 =pads.ports[1], layer = padtaperLayer,path_type = 'straight'))
    
    D.add_port(port = taper.ports[2], name = 'LED')
    return D
    
def hidra():
    pass

def hexapod():
    pass

def get_const_MZI(r = 0.5):    
    MZI_factors = [1.00000000e-02,   6.28473520e+00],
    [5.06122449e-02,   6.32270847e+00],
    [9.12244898e-02,   6.41026850e+00],
    [1.31836735e-01,   6.54452249e+00],
    [1.72448980e-01,   6.72150108e+00],
    [2.13061224e-01,   6.93669682e+00],
    [2.53673469e-01,   7.18552226e+00],
    [2.94285714e-01,   7.46362417e+00],
    [3.34897959e-01,   7.76705683e+00],
    [3.75510204e-01,   8.09234892e+00],
    [4.16122449e-01,   8.43650338e+00],
    [4.56734694e-01,   8.79696150e+00],
    [4.97346939e-01,   9.17155183e+00],
    [5.37959184e-01,   9.55843601e+00],
    [5.78571429e-01,   9.95605762e+00],
    [6.19183673e-01,   1.03630968e+01],
    [6.59795918e-01,   1.07784312e+01],
    [7.00408163e-01,   1.12011031e+01],
    [7.41020408e-01,   1.16302926e+01],
    [7.81632653e-01,   1.20652941e+01],
    [8.22244898e-01,   1.25054988e+01],
    [8.62857143e-01,   1.29503786e+01],
    [9.03469388e-01,   1.33994738e+01],
    [9.44081633e-01,   1.38523824e+01],
    [9.84693878e-01,   1.43087520e+01],
    [1.02530612e+00,   1.47682721e+01],
    [1.06591837e+00,   1.52306680e+01],
    [1.10653061e+00,   1.56956963e+01],
    [1.14714286e+00,   1.61631404e+01],
    [1.18775510e+00,   1.66328068e+01],
    [1.22836735e+00,   1.71045224e+01],
    [1.26897959e+00,   1.75781315e+01],
    [1.30959184e+00,   1.80534942e+01],
    [1.35020408e+00,   1.85304839e+01],
    [1.39081633e+00,   1.90089862e+01],
    [1.43142857e+00,   1.94888972e+01],
    [1.47204082e+00,   1.99701224e+01],
    [1.51265306e+00,   2.04525757e+01],
    [1.55326531e+00,   2.09361784e+01],
    [1.59387755e+00,   2.14208586e+01],
    [1.63448980e+00,   2.19065500e+01],
    [1.67510204e+00,   2.23931922e+01],
    [1.71571429e+00,   2.28807293e+01],
    [1.75632653e+00,   2.33691096e+01],
    [1.79693878e+00,   2.38582857e+01],
    [1.83755102e+00,   2.43482136e+01],
    [1.87816327e+00,   2.48388525e+01],
    [1.91877551e+00,   2.53301646e+01],
    [1.95938776e+00,   2.58221146e+01],
    [2.00000000e+00,   2.63146699e+01]
    f = interp1d(MZI_factors[:,0],MZI_factors[:,1],kind='cubic')
    F = f(r)
    return F
    