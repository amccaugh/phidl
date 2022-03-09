# -*- coding: utf-8 -*-
import sys

import numpy as np
import pytest

import phidl.geometry as pg
from phidl import Device, Port


def test_rectangle():
    D = pg.rectangle(size=(4, 2), layer=0)
    h = D.hash_geometry(precision=1e-4)
    assert h == "729832d93c8f0c9100ac8fe665894920bc47654a"


def test_bbox():
    D = pg.bbox(bbox=[(-1, -1), (3, 4)], layer=0)
    h = D.hash_geometry(precision=1e-4)
    assert h == "e04a2dc0457f4ae300e9d1235fde335362c8b454"


def test_cross():
    D = pg.cross(length=10, width=3, layer=0)
    h = D.hash_geometry(precision=1e-4)
    assert h == "18470cb24843265fe42a283f1cde91cbc8a2abe2"


def test_ellipse():
    D = pg.ellipse(radii=(10, 5), angle_resolution=2.5, layer=0)
    h = D.hash_geometry(precision=1e-4)
    assert h == "f46840e01a8b8d292a23d4c651a064d65c563575"


def test_circle():
    D = pg.circle(radius=10, angle_resolution=2.5, layer=0)
    h = D.hash_geometry(precision=1e-4)
    assert h == "15a748245d7bdfd4987f081e1494b41096e9c95a"


def test_ring():
    D = pg.ring(radius=10, width=0.5, angle_resolution=2.5, layer=0)
    h = D.hash_geometry(precision=1e-4)
    assert h == "66b57b1544c9c48d9babc107dc93e7b0bf378499"


def test_arc():
    D = pg.arc(
        radius=10, width=0.5, theta=45, start_angle=0, angle_resolution=2.5, layer=0
    )
    h = D.hash_geometry(precision=1e-4)
    assert h == "a273d16c6c7daf4f80fcdfdd82514be2484d6349"


def test_turn():
    port = Port(name=1, midpoint=(7.5, 6.7), orientation=47)
    D = pg.turn(port, radius=10, angle=270, angle_resolution=2.5, layer=0)
    h = D.hash_geometry(precision=1e-4)
    assert h == "77dc856a16ed48184b743fe801ce70bedfa82a83"


def test_straight():
    D = pg.straight(size=(4, 2), layer=0)
    h = D.hash_geometry(precision=1e-4)
    assert h == "729832d93c8f0c9100ac8fe665894920bc47654a"


def test_L():
    D = pg.L(width=1, size=(10, 20), layer=0)
    h = D.hash_geometry(precision=1e-4)
    assert h == "fc8e0a442ddc6f348876c14da19fcdaf0024705a"


def test_C():
    D = pg.C(width=1, size=(10, 20), layer=0)
    h = D.hash_geometry(precision=1e-4)
    assert h == "c021f4b8dc8bcdad6ccc48bd3eb00d2895f7ac98"


def test_offset():
    A = pg.cross(length=10, width=3, layer=0)
    B = pg.ellipse(radii=(10, 5), angle_resolution=2.5, layer=1)
    D = pg.offset([A, B], distance=0.1, join_first=True, precision=0.001, layer=2)
    h = D.hash_geometry(precision=1e-4)
    assert h == "dea81b4adf9f163577cb4c750342f5f50d4fbb6d"


def test_invert():
    A = pg.cross(length=10, width=3, layer=0)
    B = pg.ellipse(radii=(10, 5), angle_resolution=2.5, layer=1)
    D = pg.invert([A, B], border=4, precision=1e-6, layer=2)
    h = D.hash_geometry(precision=1e-4)
    assert h == "eed5a4cb31da61a495c9ff4c5dc4d06fe28707aa"


def test_boolean():
    A = pg.cross(length=10, width=3, layer=0)
    B = pg.ellipse(radii=(10, 5), angle_resolution=2.5, layer=1)
    D = pg.boolean(A=A, B=B, operation="and", precision=1e-6, layer=2)
    h = D.hash_geometry(precision=1e-4)
    assert h == "fcf1d0809488be01480027a5914dfb399faf088c"


def test_outline():
    A = pg.cross(length=10, width=3, layer=0)
    B = pg.ellipse(radii=(10, 5), angle_resolution=2.5, layer=1)
    D = pg.outline([A, B], distance=1, precision=1e-6, layer=2)
    h = D.hash_geometry(precision=1e-4)
    assert h == "503522b071080be6c98017cdc616752c1a3d75ce"


def test_port_geometry():
    # Conversion between object and geometric representation of ports
    def geom_equal(A, B):
        h1 = A.hash_geometry(precision=1e-4)
        h2 = B.hash_geometry(precision=1e-4)
        return h1 == h2

    init_D = pg.compass(layer=1)
    geom_D = pg.ports_to_geometry(init_D, layer=2)
    end_D = pg.geometry_to_ports(geom_D, layer=2)
    assert geom_equal(init_D, end_D)

    assert len(geom_D.ports) == 0
    geom_D.remove_layers([2], include_labels=True)
    assert geom_equal(init_D, geom_D)

    for pnam, port in init_D.ports.items():
        assert np.all(end_D.ports[pnam].midpoint == port.midpoint)


def test_text():
    valid_chars = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~µ"  # noqa: E501
    D = pg.text(text=valid_chars, size=77, justify="left", layer=0)
    h = D.hash_geometry(precision=1e-4)
    assert h == "6a5eef6483c46cdec54c9c284a5e5002b934a06d"
    D = pg.text(text=valid_chars, size=77, justify="right", layer=0)
    h = D.hash_geometry(precision=1e-4)
    assert h == "e7a48db1e61e006e4869853fcf6ef53ae0ac508b"
    D = pg.text(text=valid_chars, size=77, justify="center", layer=0)
    h = D.hash_geometry(precision=1e-4)
    assert h == "6e9d0977b510a079daf15b22387d32222934ba75"


def test_truetype():
    _ = pytest.importorskip(
        "phidl.font", reason="Testing of ttf/otf fonts requires the freetype package."
    )
    from os import path

    cd = path.abspath(path.dirname(__file__))
    font_file = path.join(cd, "SourceCodePro-Regular.ttf")
    test_text = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~µ"  # noqa: E501

    D = pg.text(text=test_text, font=font_file, size=77, justify="center", layer=0)
    h = D.hash_geometry(precision=1e-4)
    assert h == "45044fed1303f1b847ecc2522fa02bff36b98e3b"

    # Test multilayer
    layers = {(1, 0), (2, 0), (3, 5), 4}
    D = pg.text(text=test_text, font=font_file, size=77, justify="center", layer=layers)
    h = D.hash_geometry(precision=1e-4)
    assert h == "4ce33f201596995f57432fc30e54b7d78430b8a5"


@pytest.mark.skipif(sys.version_info < (3, 0), reason="unicode test requires python3")
def test_unicode():
    _ = pytest.importorskip(
        "phidl.font", reason="Testing of ttf/otf fonts requires the freetype package."
    )
    from os import path

    cd = path.abspath(path.dirname(__file__))
    font_file = path.join(cd, "SourceCodePro-Regular.ttf")
    test_unicode = "Árvíztűrő tükörfúrógép"

    D = pg.text(text=test_unicode, font=font_file, size=77, justify="center", layer=0)
    h = D.hash_geometry(precision=1e-4)
    assert h == "fc7cac5c10bb38785db8908658afcc82552e916f"


def test_copy_deepcopy():
    D = Device()
    A = pg.ellipse(radii=(10, 5), angle_resolution=2.5, layer=1)
    B = pg.ellipse(radii=(10, 5), angle_resolution=2.5, layer=1)
    a = D << A  # noqa: F841
    b1 = D << B  # noqa: F841
    b2 = D << B  # noqa: F841

    Dcopy = pg.copy(D)
    Ddeepcopy = pg.deepcopy(D)
    h = D.hash_geometry(precision=1e-4)
    assert h == "0313cd7e58aa265b44dd1ea10265d1088a2f1c6d"
    h = Dcopy.hash_geometry(precision=1e-4)
    assert h == "0313cd7e58aa265b44dd1ea10265d1088a2f1c6d"
    h = Ddeepcopy.hash_geometry(precision=1e-4)
    assert h == "0313cd7e58aa265b44dd1ea10265d1088a2f1c6d"

    D << pg.ellipse(radii=(12, 5), angle_resolution=2.5, layer=2)
    h = D.hash_geometry(precision=1e-4)
    assert h == "856cedcbbb53312ff839b9fe016996357e658d33"
    h = Dcopy.hash_geometry(precision=1e-4)
    assert h == "0313cd7e58aa265b44dd1ea10265d1088a2f1c6d"
    h = Ddeepcopy.hash_geometry(precision=1e-4)
    assert h == "0313cd7e58aa265b44dd1ea10265d1088a2f1c6d"

    A.add_ref(pg.ellipse(radii=(12, 5), angle_resolution=2.5, layer=2))
    B.add_polygon([[3, 4, 5], [6.7, 8.9, 10.15]], layer=0)
    h = D.hash_geometry(precision=1e-4)
    assert h == "c007b674e8053c11c877860f0552fff18676b68e"
    h = Dcopy.hash_geometry(precision=1e-4)
    assert h == "2590bd786348ab684616eecdfdbcc9735b156e18"
    h = Ddeepcopy.hash_geometry(precision=1e-4)
    assert h == "0313cd7e58aa265b44dd1ea10265d1088a2f1c6d"


def test_write_and_import_gds():
    D = Device()
    D.add_ref(pg.rectangle(size=[1.5, 2.7], layer=[3, 2]))
    D.add_ref(pg.rectangle(size=[0.8, 2.5], layer=[9, 7]))
    D.add_array(
        pg.rectangle(size=[1, 2], layer=[4, 66]), rows=3, columns=2, spacing=[14, 7.5]
    )
    D.add_array(
        pg.rectangle(size=[1.5, 2.5], layer=[4, 67]),
        rows=1,
        columns=2,
        spacing=[14, 7.5],
    )
    D.add_polygon([[3, 4, 5], [6.7, 8.9, 10.15]], layer=[7, 8])
    D.add_polygon([[3, 4, 5], [1.7, 8.9, 10.15]], layer=[7, 9])
    precision = 1e-4
    unit = 1e-6
    h1 = D.hash_geometry(precision=precision)
    D.write_gds("temp.gds", precision=unit * precision, unit=1e-6)
    Dimport = pg.import_gds("temp.gds", flatten=False)
    h2 = Dimport.hash_geometry(precision=precision)
    assert h1 == h2


def test_packer():
    np.random.seed(5)
    D_list = [
        pg.ellipse(radii=np.random.rand(2) * n + 2).move(np.random.rand(2) * 100 + 2)
        for n in range(50)
    ]
    D_list += [
        pg.rectangle(size=np.random.rand(2) * n + 2).move(np.random.rand(2) * 1000 + 2)
        for n in range(50)
    ]

    D_packed_list = pg.packer(
        D_list,  # Must be a list or tuple of Devices
        spacing=1.25,  # Minimum distance between adjacent shapes
        aspect_ratio=(1, 2),  # Shape of the box
        max_size=(None, None),  # Limits the size into which the shapes will be packed
        density=1.5,
        sort_by_area=True,  # Pre-sorts the shapes by area
        verbose=False,
    )
    # The function will return a list of packed Devices.  If not all the Devices
    # in D_list can fit in the area `max_size`, it will fill up the first box to
    # capacity then create another, repeating until all the shapes are packed
    # into boxes of max_size.  (`max_size` can be (None, None))
    # of `max_size` as is necessary
    D = D_packed_list[0]
    h = D.hash_geometry(precision=1e-4)
    assert h == "d90e43693a5840bdc21eae85f56fdaa57fdb88b2"
