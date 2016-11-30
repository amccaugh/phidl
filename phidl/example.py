# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 16:28:09 2016

@author: smb2
"""

from __future__ import division # Makes it so 1/4 = 0.25 instead of zero


from phidl import Device, quickplot
import numpy as np
import phidl.geometry as pg
import basicPhotonics as bP

P = bP.pad()
DP = bP.dblpad(PadDevice = P)
D = bP.wgSNSPD(DblPadDevice = DP)
quickplot(D)
#meanderWireWidth = 0.4
#meanderPitch = 0.8
#meanderLength = 50
#nwLayer = 6
#wgLayer = 1
#nsquares = 5000
#wgNwWidth = 0.1
#wgNwPitch = 0.3
#wgNwLength = 100 
#padDistance = 500
#meanderOffset = 4*meanderPitch
#padLayer = 2
#landingPadOffset = 10
#
#D = Device()
#nsquares_per_turn = (meanderLength/2-(meanderWireWidth+meanderPitch))/meanderWireWidth + 1
## the number of turns that we will actually make. Must be an odd number if the leads need to come out the same side
#nturns = np.floor(nsquares/nsquares_per_turn)-1
#
#meanderWidth = nturns*meanderPitch
#SNSPD = pg.snspd(wire_width = meanderWireWidth, wire_pitch = meanderPitch, size = (meanderLength,meanderWidth),
#          terminals_same_side = True, layer = nwLayer)
#SNSPD.add_port(name = 3, midpoint = [SNSPD.xmin+meanderWireWidth/2, SNSPD.ymin], width = meanderWireWidth, orientation = -90)
#SNSPD.add_port(name = 4, midpoint = [SNSPD.xmin+meanderWireWidth/2, SNSPD.ymax], width = meanderWireWidth, orientation = 90)
#meander = D.add_ref(SNSPD)
#wgNw = D.add_ref(pg.optimal_hairpin(width = wgNwWidth, pitch = wgNwPitch, length = wgNwLength, layer = nwLayer))
#wgNw.reflect(p1 = wgNw.ports[1].midpoint, p2 = wgNw.ports[2].midpoint)
#wgNw.xmax = meander.xmin - meanderWireWidth
#wgNw.ymax = meander.ymin-meanderOffset
#
## connector between the hairpin and the meander
#Hairpin2meander = pg.rectangle(size = [meanderWireWidth, meanderOffset-meanderWireWidth+wgNwWidth], layer = nwLayer)
#Hairpin2meander.add_port(name = 1, midpoint = [meanderWireWidth/2, Hairpin2meander.ymin], width = meanderWireWidth, orientation = -90)
#Hairpin2meander.add_port(name = 2, midpoint = [meanderWireWidth, Hairpin2meander.ymax-wgNwWidth/2], width = wgNwWidth, orientation = 0)
#hairpin2meander = D.add_ref(Hairpin2meander)
#hairpin2meander.connect(port = hairpin2meander.ports[1], destination = SNSPD.ports[3])
#wgNw.connect(port = wgNw.ports[1], destination = hairpin2meander.ports[2])
#
## vertical fill rectangles
#meander_size = [meander.bbox[1][0]-meander.bbox[0][1], meander.bbox[0][1]-meander.bbox[1][1]]
#R1 = pg.rectangle(size = [meanderWireWidth, np.abs(meander_size[1])+ 6*meanderPitch], layer = nwLayer)
#
#for i in range(0, 3):
#    fill = D.add_ref(R1)
#    fill2 = D.add_ref(R1)
#    fill.xmin = meander.xmin - (i+1)*meanderPitch
#    fill.y = meander.y
#    fill2.xmax = meander.xmax + (i+1)*meanderPitch
#    fill2.y = meander.y
#    
## horizontal fill rectangles
#meander_size = [meander.xmax-meander.xmin, meander.bbox[0][1]-meander.bbox[1][1]]
#R2 = pg.rectangle(size = [np.abs(meander_size[0]) - meanderPitch, meanderWireWidth], layer = nwLayer)
#
#for i in range(0, 3):
#    fill = D.add_ref(R2)
#    fill2 = D.add_ref(R2)
#    fill.ymin = meander.ymin - (i+1)*meanderPitch
#    fill.x = meander.x +meanderPitch-meanderWireWidth
#    fill2.ymax = meander.ymax + (i+1)*meanderPitch
#    fill2.x = meander.x+meanderPitch-meanderWireWidth
#
## connectors between nw and pad and meander and pad
#
#R3 = pg.rectangle(size = [meanderWireWidth, meanderPitch*4-meanderWireWidth + 5])
#R4 = pg.rectangle(size = [meanderWireWidth, 5])
#R3.add_port(name = 1, midpoint = [meanderWireWidth/2, 0], width = meanderWireWidth, orientation = -90)
#R3.add_port(name = 2, midpoint = [meanderWireWidth, R3.ymax-5/2], width = 5, orientation = 0)
#R4.add_port(name = 1, midpoint = [0, R4.ymax - wgNwWidth/2], width = wgNwWidth, orientation = 180)  
#R4.add_port(name = 2, midpoint = [meanderWireWidth, 5/2], width = 5, orientation = 0)
#hairpin2pads = D.add_ref(R4)
#meander2pads = D.add_ref(R3)
#hairpin2pads.connect(port = 1, destination = wgNw.ports[2])
#meander2pads.connect(port = 1, destination = meander.ports[4])
#
## nw layer pads
#
#R4 = pg.compass(size = [np.abs(meander_size[0])/2, 5], layer = nwLayer)
#nwPad1 = D.add_ref(R4)
#nwPad2= D.add_ref(R4)
#nwPad1.connect(port = 'W', destination = meander2pads.ports[2])
#nwPad2.connect(port = 'W', destination = hairpin2pads.ports[2])
#
## metal layer pads
#M1 = pg.compass(size = [np.abs(meander_size[0])/2, 5], layer = padLayer)
#metalWire1 = D.add_ref(M1)
#metalWire2= D.add_ref(M1)
#metalWire1.center = nwPad1.center
#metalWire2.center = nwPad2.center
#
#P = bP.pad(width = 100,height = 200,poOffset = 2,padLayer = padLayer,poLayer = 3)
#DP = bP.dblpad(gap = 10, PadDevice = P)
#pads = D.add_ref(DP)
#pads.rotate(angle = -90)
#pads.xmin = meander.xmax + padDistance
#pads.y = meander.y
#Route1 = pg.route(port1 = pads.ports[1], port2 = metalWire1.ports['E'], path_type = 'straight', layer = padLayer)
#Route2 = pg.route(port1 = pads.ports[2], port2 = metalWire2.ports['E'], path_type = 'straight', layer = padLayer)
#D.add_ref(Route1)
#D.add_ref(Route2)
#
## wg layer
#
## wg layer wg
#
#wg = D.add_ref(pg.compass(size = [wgNwLength + wgNwPitch, wgNwPitch*2], layer = wgLayer)) 
#wg.xmax = wgNw.xmax
#wg.y = wgNw.y
#D.add_port(name = 1, port = wg.ports['W'])
#
## wg layer landing pad
## padside
#size = [pads.xmax - pads.xmin + landingPadOffset, pads.ymax - pads.ymin + landingPadOffset] 
#landingPad1 = D.add_ref(pg.compass(size = size, layer = wgLayer))   
#landingPad1.center = pads.center
#
## nw side
#size = [nwPad1.xmax - hairpin2pads.xmin + 1, nwPad1.ymax - nwPad2.ymin + 1]
#landingPad2 = D.add_ref(pg.compass(size = size, layer = wgLayer))   
#landingPad2.y = meander.y
#landingPad2.xmin = nwPad1.xmin - 1
#Route3 = pg.route(port1 = landingPad1.ports['W'], port2 = landingPad2.ports['E'], layer = wgLayer)
#D.add_ref(Route3)
#quickplot(D)