import pytest

from phidl import Device, Layer, LayerSet, make_device, Port
import phidl.geometry as pg
import phidl.routing as pr
import phidl.utilities as pu

def test_rectangle():
    D = pg.rectangle(size = (4,2), layer = 0)
    h = D.hash_geometry(precision = 1e-4)
    assert(h == '699e68d5cd3ddad4174b33662ff0bba12c431fed')

def test_bbox():
    D = pg.bbox(bbox = [(-1,-1),(3,4)], layer = 0)
    h = D.hash_geometry(precision = 1e-4)
    assert(h == '4e43d392bd0b44caa9c61c57bd85b813039a85bb')

def test_cross():
    D = pg.cross(length = 10, width = 3, layer = 0)
    h = D.hash_geometry(precision = 1e-4)
    assert(h == 'da7ef74460ca0c784851ea8364c5ab2489e80268')

def test_ellipse():
    D = pg.ellipse(radii = (10,5), angle_resolution = 2.5, layer = 0)
    h = D.hash_geometry(precision = 1e-4)
    assert(h == '3e16b501e62fe3c66188cbfc1a36fdd735dbfd02')

def test_circle():
    D = pg.circle(radius = 10, angle_resolution = 2.5, layer = 0)
    h = D.hash_geometry(precision = 1e-4)
    assert(h == '54bca9b60a8d1ed5e697e2a79d6127a0a0613c78')
    
def test_ring():
    D = pg.ring(radius = 10, width = 0.5, angle_resolution = 2.5, layer = 0)
    h = D.hash_geometry(precision = 1e-4)
    assert(h == '48dda67dd508b1f285b2b59969b2cab9abbb7c8c')
    
def test_arc():
    D = pg.arc(radius = 10, width = 0.5, theta = 45, start_angle = 0, angle_resolution = 2.5, layer = 0)
    h = D.hash_geometry(precision = 1e-4)
    assert(h == 'a948ec57400107337e52dba647783f97764d3335')
    
def test_turn():
    port = Port(name = 1, midpoint = (7.5,6.7), orientation = 47)
    D = pg.turn(port, radius = 10, angle = 270, angle_resolution = 2.5, layer = 0)
    h = D.hash_geometry(precision = 1e-4)
    assert(h == 'b360957ce57ff8dced09d1fff00fb073f3319f38')
    
def test_straight():
    D = pg.straight(size = (4,2), layer = 0)
    h = D.hash_geometry(precision = 1e-4)
    assert(h == '699e68d5cd3ddad4174b33662ff0bba12c431fed')
    
def test_L():
    D = pg.L(width = 1, size = (10,20) , layer = 0)
    h = D.hash_geometry(precision = 1e-4)
    assert(h == '076724801440378da6109679f749ce1fd7cc87a5')
    
def test_C():
    D = pg.C(width = 1, size = (10,20) , layer = 0)
    h = D.hash_geometry(precision = 1e-4)
    assert(h == '42bb6456054b431a2cbd9fa1ff7e009f5acd2a5b')
    
def test_offset():
    A = pg.cross(length = 10, width = 3, layer = 0)
    B = pg.ellipse(radii = (10,5), angle_resolution = 2.5, layer = 1)
    D = pg.offset([A,B], distance = 0.1, join_first = True, precision = 0.001, max_points = 4000, layer = 2)
    h = D.hash_geometry(precision = 1e-4)
    assert(h == '0ee12c1b504b0ea6784f8a1a7aaa2d03fd51a1c2')
