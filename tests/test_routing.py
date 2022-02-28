import numpy as np

import phidl.path as pp
import phidl.routing as pr
from phidl import Device

# import phidl.utilities as pu


def test_route_quad():
    D = Device()
    port1 = D.add_port(name=1, midpoint=(30, 30), width=20, orientation=270)
    port2 = D.add_port(name=2, midpoint=(0, 0), width=20, orientation=60)
    R = pr.route_quad(
        port1, port2, width1=None, width2=None
    )  # width = None means use Port width
    h = R.hash_geometry(precision=1e-4)
    assert h == "c42b130a3822aaefd368464d9e6558c750928b52"


def test_route_smooth1():
    D = Device()
    port1 = D.add_port(name="smooth1", midpoint=(40, 0), width=5, orientation=180)
    port2 = D.add_port(name="smooth2", midpoint=(0, -40), width=5, orientation=270)
    R = pr.route_smooth(
        port1,
        port2,
        radius=5,
        width=None,
        path_type="manhattan",
        manual_path=None,
        smooth_options={"corner_fun": pp.euler, "use_eff": True},
        layer=np.nan,
    )
    h = R.hash_geometry(precision=1e-4)
    assert h == "f7865e2b7f865a214b7847fdaa704d729842fde6"


def test_route_smooth2():
    D = Device()
    port1 = D.add_port(name="smooth1", midpoint=(40, 0), width=5, orientation=180)
    port2 = D.add_port(name="smooth2", midpoint=(0, -40), width=5, orientation=270)
    R = pr.route_smooth(
        port1,
        port2,
        radius=1.7,
        width=2.5,
        path_type="manhattan",
        manual_path=None,
        smooth_options={"corner_fun": pp.arc},
        layer=3,
    )
    h = R.hash_geometry(precision=1e-4)
    assert h == "e47d0573b1c1adba357d7ea8f523f84425e474e4"


def test_route_straight():
    # straight path
    D = Device()
    port1 = D.add_port(name="S1", midpoint=(-50, 0), width=4, orientation=90)
    port2 = D.add_port(name="S2", midpoint=(-50, 50), width=4, orientation=270)
    D = pr.route_smooth(port1, port2, path_type="straight")
    h = D.hash_geometry(precision=1e-4)
    assert h == "051c27aed3cbf01700aadd83cf2dc7304d0236cf"


def test_route_L():
    D = Device()  # L path
    port1 = D.add_port(name="L1", midpoint=(30, 0), width=4, orientation=180)
    port2 = D.add_port(name="L2", midpoint=(0, 50), width=4, orientation=270)
    D = pr.route_smooth(port1, port2, path_type="L")
    h = D.hash_geometry(precision=1e-4)
    assert h == "f3b8ee6096184a6cae2f6dab7cda038c7f80ba4b"


def test_route_U1():
    D = Device()
    # U path
    port1 = D.add_port(name="U1", midpoint=(50, 50), width=2, orientation=270)
    port2 = D.add_port(name="U2", midpoint=(80, 50), width=4, orientation=270)
    D = pr.route_smooth(port1, port2, radius=10, path_type="U", length1=50)
    h = D.hash_geometry(precision=1e-4)
    assert h == "abfd2a4c8d3d1d388675dc613efacf369bbd7e4a"


def test_route_U2():
    D = Device()
    # U path
    port1 = D.add_port(name="U3", midpoint=(50, 80), width=4, orientation=10)
    port2 = D.add_port(name="U4", midpoint=(80, 130), width=4, orientation=190)
    D = pr.route_smooth(port1, port2, path_type="U", length1=20)
    h = D.hash_geometry(precision=1e-4)
    assert h == "21f43cfcbede169789b4808771286e368bb363ca"


def test_route_J1():
    D = Device()
    # J path
    port1 = D.add_port(name="J1", midpoint=(100, 25), width=4, orientation=270)
    port2 = D.add_port(name="J2", midpoint=(130, 50), width=4, orientation=180)
    D = pr.route_smooth(port1, port2, path_type="J", length1=25, length2=10)
    h = D.hash_geometry(precision=1e-4)
    assert h == "1b17edad12788318cc113a8e1e76893ae301e6ca"


def test_route_J2():
    D = Device()
    port1 = D.add_port(name="J3", midpoint=(115, 105), width=5, orientation=270)
    port2 = D.add_port(name="J4", midpoint=(131, 130), width=5, orientation=180)
    D = pr.route_smooth(port1, port2, path_type="J", length1=25, length2=30)
    h = D.hash_geometry(precision=1e-4)
    assert h == "6872cc085a7f8829a8962cfe6b3ed53e253ac075"


def test_route_C1():
    D = Device()
    # C path
    port1 = D.add_port(name="C1", midpoint=(180, 35), width=4, orientation=90)
    port2 = D.add_port(name="C2", midpoint=(178, 15), width=4, orientation=270)
    D = pr.route_smooth(port1, port2, path_type="C", length1=15, left1=30, length2=15)
    h = D.hash_geometry(precision=1e-4)
    assert h == "d9f1f76304ce01a775fe92b3ef662f26373fc2a6"


def test_route_C2():
    D = Device()
    port1 = D.add_port(name="C3", midpoint=(150, 105), width=4, orientation=90)
    port2 = D.add_port(name="C4", midpoint=(180, 105), width=4, orientation=270)
    D = pr.route_smooth(port1, port2, path_type="C", length1=25, left1=-15, length2=25)
    h = D.hash_geometry(precision=1e-4)
    assert h == "b894692aa396ccda7f28fdcb3727d0e814e97935"


def test_route_C3():
    D = Device()
    port1 = D.add_port(name="C5", midpoint=(150, 170), width=4, orientation=0)
    port2 = D.add_port(name="C6", midpoint=(175, 170), width=4, orientation=0)
    D = pr.route_smooth(
        port1, port2, path_type="C", length1=10, left1=10, length2=10, radius=4
    )
    h = D.hash_geometry(precision=1e-4)
    assert h == "098cfe5cac505408cd4c78a1a239595d801904f2"


def test_route_V():
    D = Device()
    port1 = D.add_port(name="V1", midpoint=(200, 50), width=5, orientation=284)
    port2 = D.add_port(name="V2", midpoint=(230, 50), width=5, orientation=270 - 14)
    D = pr.route_smooth(port1, port2, path_type="V")
    h = D.hash_geometry(precision=1e-4)
    assert h == "b8a5cb77ad0e35e367530168f3aa9c72b3538117"


def test_route_Z():
    D = Device()
    port1 = D.add_port(name="Z1", midpoint=(280, 0), width=4, orientation=190)
    port2 = D.add_port(name="Z2", midpoint=(250, 50), width=3, orientation=-10)
    D = pr.route_smooth(port1, port2, path_type="Z", length1=30, length2=40)
    h = D.hash_geometry(precision=1e-4)
    assert h == "1332b33414a4362b9e83d11649b8ecaffe8604b0"
