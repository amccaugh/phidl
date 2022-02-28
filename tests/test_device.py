import numpy as np

import phidl.geometry as pg
from phidl import Device, Group  # , Layer, LayerSet, make_device, Port

# import phidl.routing as pr
# import phidl.utilities as pu


def test_add_polygon1():
    D = Device()
    D.add_polygon([(8, 6, 7, 9), (6, 8, 9, 5)])
    h = D.hash_geometry(precision=1e-4)
    assert h == "c0629d2a7c557f72fad131ae8260df22c1df2d56"


def test_add_polygon2():
    D = Device()
    D.add_polygon([(8, 6), (6, 8), (7, 9), (9, 5)])
    h = D.hash_geometry(precision=1e-4)
    assert h == "c0629d2a7c557f72fad131ae8260df22c1df2d56"


def test_add_polygon3():
    D = Device()
    D.add_polygon([(8, 6), (6, 8), (7, 9), (9, 5)], layer=7)
    D.add_polygon([(8, 0), (6, 8), (7, 9), (9, 5)], layer=(8, 0))
    D.add_polygon([(8, 1), (6, 8), (7, 9), (9, 5)], layer=(9, 1))
    h = D.hash_geometry(precision=1e-4)
    assert h == "96abc3c9e30f3bbb32c5a39aeea2ba0fa3b13ebe"


def test_bbox():
    D = Device()
    D.add_polygon([(0, 0), (10, 0), (10, 10), (0, 10)], layer=2)
    assert D._bb_valid is False
    # Calculating the bbox should change _bb_valid to True once it's cached
    assert D.bbox.tolist() == [[0, 0], [10, 10]]
    assert D._bb_valid is True

    E = Device()
    e1 = E.add_ref(D)
    e2 = E.add_ref(D)
    e2.movex(30)
    assert E._bb_valid is False
    assert E.bbox.tolist() == [[0, 0], [40, 10]]
    assert E._bb_valid is True

    D.add_polygon([(0, 0), (100, 0), (100, 100), (0, 100)], layer=2)
    D.add_polygon([(0, 0), (100, 0), (100, 100), (0, 100)], layer=2)
    assert E.bbox.tolist() == [[0, 0], [130, 100]]
    assert e1.bbox.tolist() == [[0, 0], [100, 100]]
    assert e2.bbox.tolist() == [[30, 0], [130, 100]]


def test_add_array():
    D = Device()
    E = Device()
    E.add_polygon([[30, 20], [30, 0], [0, 0], [0, 20]], layer=7)
    A = D.add_array(E, columns=7, rows=5, spacing=(31, 21))
    assert A.bbox.tolist() == [[0.0, 0.0], [216.0, 104.0]]
    A.rotate(10)
    A.move((15, 1.5))
    A.mirror((0, 1))
    A.get_polygons()
    h = D.hash_geometry(precision=1e-4)
    assert h == "418b7503baff80fbe93031d45d87557c277f07b4"

    F = Device()
    f1 = F << D
    f2 = F << D
    f1.movex(300)
    f2.rotate(45)
    h = F.hash_geometry(precision=1e-4)
    assert h == "fd7c2b4adb811342b836d9fca13992eff951630d"


# Test polygon manipulation
def test_move():
    # Test polygon move
    D = Device()
    p = D.add_polygon([(8, 6, 7, 9), (6, 8, 9, 5)])
    p.move([1.7, 0.8])
    h = D.hash_geometry(precision=1e-4)
    assert h == "57a86bce5f60f7bc78c7c30473a544b736d2afb3"
    p.movex(13.9)
    h = D.hash_geometry(precision=1e-4)
    assert h == "8fe6706e05ebe1512ee2efe2582546b949fbc48f"
    p.movey(19.2)
    h = D.hash_geometry(precision=1e-4)
    assert h == "7df43241eca2dd11f267c25876e650eadaca7d9f"
    # Test Device move
    D = Device()
    D.add_polygon([(8, 6, 7, 9), (6, 8, 9, 5)])
    D.add_polygon([(8, 6, 7, 9, 7, 0), (6, 8, 9, 5, 7, 0)])
    D.move([1.7, 0.8])
    h = D.hash_geometry(precision=1e-4)
    assert h == "c863156dd00a590dc02823e1791554d4142b1ea9"
    # Test label move
    D = Device()
    D.add_polygon([(8, 8, 8, 8), (6, 6, 6, 6)])
    label = D.add_label("testing", position=D.center)
    print(all(label.center == D.center))
    D.rotate(45)
    print(np.allclose(label.center, D.center))
    D.move([70000.5, 30000.5])
    print(np.allclose(label.center, D.center))
    D.rotate(75)
    print(np.allclose(label.center, D.center))
    D.mirror([7, 5])
    print(np.allclose(label.center, D.center))


def test_rotate():
    # Test polygon rotation
    D = Device()
    p = D.add_polygon([(8, 6, 7, 9), (6, 8, 9, 5)])
    p.rotate(37.5)
    h = D.hash_geometry(precision=1e-4)
    assert h == "2e4815072eabe053c3029d9e29a5b3ed59fe9bb7"
    # Test Device rotation
    D = Device()
    p = D.add_polygon([(8, 6, 7, 9), (6, 8, 9, 5)])
    D.rotate(37.5)
    h = D.hash_geometry(precision=1e-4)
    assert h == "2e4815072eabe053c3029d9e29a5b3ed59fe9bb7"


def test_reflect():
    # Test polygon reflection
    D = Device()
    p = D.add_polygon([(8, 6, 7, 9), (6, 8, 9, 5)])
    p.mirror(p1=(1.7, 2.5), p2=(4.5, 9.1))
    h = D.hash_geometry(precision=1e-4)
    assert h == "bc6ae5308c2240e425cd503e0cdda30007bbfc4d"
    # Test Device reflection
    D = Device()
    p = D.add_polygon([(8, 6, 7, 9), (6, 8, 9, 5)])
    D.mirror(p1=(1.7, 2.5), p2=(4.5, 9.1))
    h = D.hash_geometry(precision=1e-4)
    assert h == "bc6ae5308c2240e425cd503e0cdda30007bbfc4d"


def test_port_add():
    D = Device()
    D.add_port(name="test123", midpoint=(5.7, 9.2), orientation=37)
    D.add_port(name="test456", midpoint=(1.5, 6.7), orientation=99)
    assert len(D.ports) == 2
    assert np.allclose(D.ports["test123"].midpoint, (5.7, 9.2))
    assert np.allclose(D.ports["test456"].midpoint, (1.5, 6.7))
    assert D.ports["test123"].orientation == 37
    assert D.ports["test456"].orientation == 99


def test_port_reference_manipulate():
    D = Device()
    D.add_port(name="test123", midpoint=(5.7, 9.2), orientation=37)
    d = D.add_ref(D).move([1, 1]).rotate(45)
    assert np.allclose(
        d.ports["test123"].midpoint, (-2.474873734152916, 11.950104602052654)
    )
    assert d.ports["test123"].orientation == 37 + 45


def test_port_remove():
    D = Device()
    D.add_port(name="test123", midpoint=(5.7, 9.2), orientation=37)
    D.add_port(name="test456", midpoint=(1.5, 6.7), orientation=99)
    E = Device()
    d = E << D
    D.remove(D.ports["test123"])
    assert len(D.ports) == 1
    assert len(d.ports) == 1
    assert D.ports["test456"]
    assert d.ports["test456"]


def test_flatten():
    D = Device()
    E1 = Device()
    E2 = Device()
    E1.add_polygon([(8, 6, 7, 9, 7, 0), (6, 8, 9, 5, 7, 0)], layer=8)
    E2.add_polygon([(18, 16, 17, 19, 17, 10), (16, 18, 19, 15, 17, 10)], layer=9)
    D << E1
    D << E2
    h = D.hash_geometry(precision=1e-4)
    assert h == "8a057feca51d8097f2a915eda558fe2a9b88fb13"
    D.flatten()
    h = D.hash_geometry(precision=1e-4)
    assert h == "8a057feca51d8097f2a915eda558fe2a9b88fb13"
    D.flatten(single_layer=(5, 5))
    h = D.hash_geometry(precision=1e-4)
    assert h == "cfc1ba30384f5f1f7d888f47f16d1f310f95b464"


def test_remove_layers():
    D = Device()
    D.add_polygon([(8, 6, 7, 9, 7), (6, 8, 9, 5, 7)], layer=13)
    D.add_polygon([(18, 16, 17, 19, 17), (16, 18, 19, 15, 17)], layer=14)
    xpts = list(range(1000))
    ypts = [x % 73 for x in xpts]
    p = D.add_polygon([xpts, ypts], layer=15)
    p.fracture(max_points=13, precision=1e-4)
    # Switch part of the polygons to layer (14,0)
    p.layers[13:17] = [14] * 4
    # Switch part of the polygons to layer (14,1)
    p.layers[23:27] = [14] * 4
    p.datatypes[23:27] = [1] * 4
    h = D.hash_geometry(precision=1e-4)
    assert h == "7a7aa6a22b3d0b852a0e465398018dd19a1be305"
    D.remove_layers(layers=[13, (14, 0)])
    h = D.hash_geometry(precision=1e-4)
    assert h == "bb81ec3b3a6be2372a7ffc32f57121a9f1a97b34"


def test_group():
    # Test all types
    D = Device()
    E1 = pg.ellipse(radii=(10, 5), angle_resolution=2.5, layer=0)
    E2 = pg.rectangle(size=(4, 2), layer=0).movex(15)
    e1 = D << E1
    e2 = D << E2
    e3 = D << E2
    e4 = D.add_label("hello", position=(1.5, -1.5))
    e5 = pg.snspd()
    e6 = D.add_polygon([(8, 6, 7, 9, 7, 0), (6, 8, 9, 5, 7, 0)])
    e7 = D.add_array(pg.cross())
    e2verify = D << E2

    # Test creation and addition
    G = Group()
    G.add(e1)
    G.add(e2)
    G.add([e3, e4, e5])
    G += (e6, e7)
    assert np.allclose(G.bbox.flatten(), np.array([-10.0, -8.5, 105.0, 105.0]))

    # Test movement
    G.move((2, 7))
    e2verify.move((2, 7))
    assert np.allclose(G.bbox.flatten(), np.array([-8.0, -1.5, 107.0, 112.0]))
    assert all(e2.center == e2verify.center)
    assert e2.rotation == e2verify.rotation

    # Test rotation
    G.rotate(90, center=(5, 5))
    e2verify.rotate(90, center=(5, 5))
    assert np.allclose(G.bbox.flatten(), np.array([-102.0, -8.0, 11.5, 107.0]))
    assert all(e2.center == e2verify.center)
    assert e2.rotation == e2verify.rotation

    # Test mirroring
    G.mirror(p1=(1, 1), p2=(-1, 1))
    e2verify.mirror(p1=(1, 1), p2=(-1, 1))
    assert np.allclose(G.bbox.flatten(), np.array([-102.0, -105.0, 11.5, 10.0]))
    assert all(e2.center == e2verify.center)
    assert e2.rotation == e2verify.rotation
    h = D.hash_geometry(precision=1e-4)
    assert h == "3964acb3971771c6e70ceb587c2ae8b37f2ed112"


def test_distribute():
    D = Device()
    # Create different-sized rectangles and add them to D
    [
        D.add_ref(pg.rectangle(size=[n * 15 + 20, n * 15 + 20]).move((n, n * 4)))
        for n in [0, 2, 3, 1, 2]
    ]
    # Distribute all the rectangles in D along the x-direction with a separation of 5
    D.distribute(
        elements="all",  # either 'all' or a list of objects
        direction="x",  # 'x' or 'y'
        spacing=5,
        separation=True,
    )

    h = D.hash_geometry(precision=1e-4)
    assert h == "1aa688d7dfb59e94d28dd0d9b8f324ff30281d70"

    D = Device()
    [
        D.add_ref(pg.rectangle(size=[n * 15 + 20, n * 15 + 20]).move((n, n * 4)))
        for n in [0, 2, 3, 1, 2]
    ]
    D.distribute(
        elements="all", direction="x", spacing=100, separation=False, edge="xmin"
    )
    h = D.hash_geometry(precision=1e-4)
    assert h == "18be0ef1db78095233d2f3ae5f065d9f453a6c07"


def test_align():
    D = Device()
    # Create different-sized rectangles and add them to D then distribute them
    [
        D.add_ref(pg.rectangle(size=[n * 15 + 20, n * 15 + 20]).move((n, n * 4)))
        for n in [0, 2, 3, 1, 2]
    ]
    D.distribute(elements="all", direction="x", spacing=5, separation=True)
    # Align top edges
    D.align(elements="all", alignment="ymax")
    h = D.hash_geometry(precision=1e-4)
    assert h == "38025959a80e46e47eabcf3f096c6273427dabc3"

    D = Device()
    # Create different-sized rectangles and add them to D then distribute them
    [
        D.add_ref(pg.rectangle(size=[n * 15 + 20, n * 15 + 20]).move((n, n * 4)))
        for n in [0, 2, 3, 1, 2]
    ]
    D.distribute(elements="all", direction="x", spacing=5, separation=True)
    # Align top edges
    D.align(elements="all", alignment="y")
    h = D.hash_geometry(precision=1e-4)
    assert h == "ed32ee1ce1f3da8f6216020877d6c1b64097c600"


def test_polygon_simplify():
    D = Device()
    t = np.linspace(0, np.pi, 1000)
    x = np.cos(t)
    y = np.sin(t)
    poly = D.add_polygon([x, y])
    h = D.hash_geometry(precision=1e-4)
    assert h == "0c3b1465c8b6ffd911c41b02114b9a06f606ad91"
    # qp(D)
    poly.simplify(tolerance=1e-1)
    h = D.hash_geometry(precision=1e-4)
    assert h == "7d9ebcb231fb0107cbbf618353adeb583782ca11"
    # qp(D)
