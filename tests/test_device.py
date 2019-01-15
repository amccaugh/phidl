import pytest

from phidl import Device#, Layer, LayerSet, make_device, Port
# import phidl.geometry as pg
# import phidl.routing as pr
# import phidl.utilities as pu

def test_add_polygon1():
    D = Device()
    D.add_polygon( [(8,6,7,9), (6,8,9,5)] )
    h = D.hash_geometry(precision = 1e-4)
    assert(h == 'c0629d2a7c557f72fad131ae8260df22c1df2d56')

def test_add_polygon2():
    D = Device()
    D.add_polygon( [(8,6), (6,8), (7,9), (9,5)] )
    h = D.hash_geometry(precision = 1e-4)
    assert(h == 'c0629d2a7c557f72fad131ae8260df22c1df2d56')

# Test polygon manipulation
def test_move():
    D = Device()
    p = D.add_polygon( [(8,6,7,9), (6,8,9,5)] )
    p.move([1.7,0.8])
    h = D.hash_geometry(precision = 1e-4)
    assert(h == '57a86bce5f60f7bc78c7c30473a544b736d2afb3')
    p.movex(13.9)
    h = D.hash_geometry(precision = 1e-4)
    assert(h == '8fe6706e05ebe1512ee2efe2582546b949fbc48f')
    p.movey(19.2)
    h = D.hash_geometry(precision = 1e-4)
    assert(h == '7df43241eca2dd11f267c25876e650eadaca7d9f')

def test_rotate():
    D = Device()
    p = D.add_polygon( [(8,6,7,9), (6,8,9,5)] )
    p.rotate(37.5)
    h = D.hash_geometry(precision = 1e-4)
    assert(h == '2e4815072eabe053c3029d9e29a5b3ed59fe9bb7')

def test_reflect():
    D = Device()
    p = D.add_polygon( [(8,6,7,9), (6,8,9,5)] )
    p.reflect(p1 = (1.7,2.5), p2 = (4.5, 9.1))
    h = D.hash_geometry(precision = 1e-4)
    assert(h == 'bc6ae5308c2240e425cd503e0cdda30007bbfc4d')