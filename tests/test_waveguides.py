import os
import numpy as np

# Functions under test
from phidl.device_layout import Layer
from phidl.waveguides import WG_XS, WG_XS_Component
import phidl.waveguides
phidl.waveguides.minimum_bent_edge_length = 0.1

# Testing frameworks
# Note that lytest stores GDS's corresponding to the correct layout for examination and XOR
import pytest
import lytest
from lytest import contained_phidlDevice, difftest_it


# Default WG_XS
xs = WG_XS(components=[WG_XS_Component(width=1, offset=0, layer=Layer(name='wg_deep', gds_layer=22)),
                       WG_XS_Component(width=3, offset=0, layer=Layer(name='wg_shallow', gds_layer=21)),
                       WG_XS_Component(width=1, offset=1, layer=Layer(name='offset_part', gds_layer=1))],
           radius=5)


@contained_phidlDevice
def Eulers(TOP):
    for theta in [20, 45, 90]:
        for rr in [2, 4, 6, 10, 20]:
            eb = TOP << xs.cell_bend(theta, radius=rr, euler=True)
            eb.xmin += 5*rr
            eb.y += theta
            cb = TOP << xs.cell_bend(theta, radius=eb.info['Reff'], euler=False)
            cb.xmin += 5*rr
            cb.y += theta + 10

def test_Eulers(): difftest_it(Eulers)()


@contained_phidlDevice
def Sbend1(TOP):
    with pytest.raises(ValueError):
        # too tight
        TOP << xs.cell_s_bend_by_offset(15, 70, radius=20)
    TOP << xs.cell_s_bend_by_offset(60, 70, radius=20).movex(30)
    TOP << xs.cell_s_bend_by_offset(50, 3, radius=20).movex(60)

def test_Sbend1(): difftest_it(Sbend1)()


@contained_phidlDevice
def Sbend2(TOP):
    TOP << xs.cell_s_bend(max_slope=30)
    TOP << xs.cell_s_bend(max_slope=-2).move((20, 0))

def test_Sbend2(): difftest_it(Sbend2)()


@contained_phidlDevice
def Manhattan_Routing(TOP):
    wgsrc = TOP << xs.cell_straight(10)
    for direction in [0, 90, -90, 180]:
        wgdest = TOP << xs.cell_straight(10)
        if direction == 0:
            wgdest.reflect((1,0))
        wgdest.move((40, 30))
        wgdest.rotate(direction)
        for euler in [False, True]:
            TOP << xs.route_ports_manhattan(wgsrc.ports['wg_out_1'], wgdest.ports['wg_out_1'], euler=euler)

def test_Manhattan_Routing(): difftest_it(Manhattan_Routing)()


@contained_phidlDevice
def General_Routing(TOP):
    for euler in [True, False]:
        wg = TOP << xs.route_points([[0,0], [20, 20], [80, 20], [10, 0], [0,-50], [-20, -20], [0,0]], euler=euler)
    wg.movey(70)

def test_General_Routing(): difftest_it(General_Routing)()

