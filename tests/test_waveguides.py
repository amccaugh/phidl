import os
import numpy as np

# Functions under test
from phidl.device_layout import Layer
from phidl.waveguides import WG_XS, WG_XS_Component, concatenate_waveguides
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
                       WG_XS_Component(width=1, offset=1, layer=Layer(name='offset_left', gds_layer=1))],
           radius=5)


def test_equality():
    xs2 = xs.copy()
    assert xs == xs2
    # unsorted components
    xs2.components.append(xs2.components[0])
    xs2.components.pop(0)
    assert xs == xs2
    # not equal
    xs2.radius = 6
    assert xs != xs2
    xs2.radius = 5
    assert xs == xs2
    xs2.components[1].offset = -1
    assert xs != xs2


def test_component_setting():
    xs1 = xs.copy()
    xs2 = xs.copy()
    xs1.components[0].offset = 2
    xs2.components[0].max = 2.5
    xs2.components[0].min = 1.5
    assert xs1 == xs2
    # Doing it this direction is not ideal, but the width will be corrected to positive
    xs2.components[0].min = 2.5
    xs2.components[0].max = 1.5
    assert xs1 == xs2
    # inner and outer
    xs1.components[2].offset = 2
    xs2.components[2].outer = 2.5
    xs2.components[2].inner = 1.5
    assert xs1 == xs2


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

# def test_Sbend1(): difftest_it(Sbend1)()


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


@contained_phidlDevice
def Sections(TOP):
    ''' More complex multi-section waveguide with tapers '''
    xs_middle =  WG_XS(components=[WG_XS_Component(width=2, offset=0, layer=Layer(name='wg_deep', gds_layer=22)),
                       WG_XS_Component(width=6, offset=0, layer=Layer(name='wg_shallow', gds_layer=21)),
                       WG_XS_Component(width=.5, offset=2, layer=Layer(name='offset_left', gds_layer=1)),
                       WG_XS_Component(width=.5, offset=-2, layer=Layer(name='offset_right', gds_layer=2))])
    for side in [-1, 1]:
        # A useful strategy for inner/outer initialization
        contact = WG_XS_Component(width=1, offset=10*side, layer=Layer(name='contact', gds_layer=16))
        contact.inner = xs_middle.get_by_layer('wg_deep')[0].max
        contact.outer = xs_middle.get_by_layer('wg_shallow')[0].max + .5
        xs_middle.components.append(contact)

    sections = [# entry taper from nothing (i.e. a terminator)
                WG_XS().cell_taper(xs, 5, keep_layernames=['wg_deep', 'wg_shallow']),
                # straight entry
                xs.cell_straight(5),
                # taper to complicated middle
                xs.cell_taper(xs_middle, 5),
                # complicated middle
                xs_middle.cell_straight(5),
                # taper back to simple (with taper options)
                xs_middle.cell_taper(xs, 5,
                                     route_basic_options=dict(width_type='straight', path_type='straight')),
                # straight exit
                xs.cell_straight(5),
                # terminator that does not have wg_shallow
                xs.cell_taper(WG_XS(), 5, keep_layernames=['wg_deep'])]
    TOP << concatenate_waveguides(sections)

def test_Sections(): difftest_it(Sections)()
