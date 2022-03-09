import numpy as np

import phidl.path as pp
from phidl import CrossSection, Path

# import phidl.utilities as pu


def test_path_extrude_width1_constant():
    P = pp.arc(radius=10, angle=90, num_pts=720)
    D = P.extrude(width=1, layer=np.nan, simplify=None)
    h = D.hash_geometry(precision=1e-4)
    assert h == "048ae3e85ff393362d2283c0046db1f0a53b5fe7"


def test_path_extrude_layer():
    P = pp.arc(radius=10, angle=90, num_pts=720)
    D = P.extrude(width=1, layer=3, simplify=None)
    h = D.hash_geometry(precision=1e-4)
    assert h == "3ec778d98ce1760c4ba067bd1bdc0baf80dcbaf7"


def test_path_extrude_simplify():
    P = pp.arc(radius=10, angle=90, num_pts=300)
    D = P.extrude(width=1, layer=3, simplify=1e-1)
    h = D.hash_geometry(precision=1e-4)
    assert h == "92a41fb5afa37cde3b06b521ff14d8445d962069"


def test_path_extrude_width2_linearly_varying():
    P = pp.arc(radius=10, angle=90, num_pts=720)
    D = P.extrude(width=[2.5, 3], layer=np.nan, simplify=None)
    h = D.hash_geometry(precision=1e-4)
    assert h == "2768e7f7ce7503a512089ddb80cd52ed0b43fd8d"


def test_path_extrude_width3_crossection():
    # Create a blank CrossSection
    X = CrossSection()
    # Add a a few "sections" to the cross-section
    X.add(width=1, offset=0, layer=0, ports=("in", "out"))
    X.add(width=3, offset=2, layer=2)
    X.add(width=3, offset=-2, layer=2)
    P = pp.arc(radius=10, angle=90, num_pts=720)
    D = P.extrude(width=X, layer=np.nan, simplify=None)
    h = D.hash_geometry(precision=1e-4)
    assert h == "e17c4ed7a7467412728dc893610006fa2d9982b0"


def test_path_extrude_CrossSection():
    # Create a blank CrossSection
    X = CrossSection()
    # Add a a few "sections" to the cross-section
    X.add(width=1, offset=0, layer=0, ports=("in", "out"))
    X.add(width=3, offset=2, layer=2)
    X.add(width=3, offset=-2, layer=2)
    P = pp.arc(radius=10, angle=90, num_pts=720)
    D = X.extrude(path=P, simplify=None)
    h = D.hash_geometry(precision=1e-4)
    assert h == "e17c4ed7a7467412728dc893610006fa2d9982b0"


def test_blank_path():
    P = Path()
    h = P.hash_geometry(precision=1e-4)
    assert h == "de8a847bff8c343d69b853a215e6ee775ef2ef96"


def test_path_straight():
    P = pp.straight(length=15, num_pts=100)
    h = P.hash_geometry(precision=1e-4)
    assert h == "3672036ccf29992546436e1cc5e62e667cde4af0"


def test_path_euler1():
    P = pp.euler(radius=3, angle=90, p=1.0, use_eff=False, num_pts=720)
    h = P.hash_geometry(precision=1e-4)
    assert h == "d6134ba90167caf551de6d2f8c7230f7f072c562"


def test_path_euler2():
    P = pp.euler(radius=3, angle=90, p=0.5, use_eff=False, num_pts=720)
    h = P.hash_geometry(precision=1e-4)
    assert h == "78b325576400e3f4f9a04c27243432102e84cfe7"


def test_path_euler3():
    P = pp.euler(radius=3, angle=90, p=0.5, use_eff=True, num_pts=720)
    h = P.hash_geometry(precision=1e-4)
    assert h == "0c5fc2a47b910522b4bd37176dbb096f99ea1083"


def test_path_arc():
    P = pp.arc(radius=10, angle=90, num_pts=720)
    h = P.hash_geometry(precision=1e-4)
    assert h == "53eaa037b8872155a1d1cb81e2daa02a0da2eb91"


def test_path_spiral():
    P = pp.spiral(num_turns=5, gap=1, inner_gap=2, num_pts=10000)
    h = P.hash_geometry(precision=1e-4)
    assert h == "1d46b2f8ab45f35ce781083755df41609d2ee628"


def test_path_smooth1():
    P = pp.smooth(
        points=[
            (20, 0),
            (40, 0),
            (80, 40),
            (80, 10),
            (100, 10),
        ],
        radius=2,
        corner_fun=pp.euler,
        use_eff=True,
    )
    h = P.hash_geometry(precision=1e-4)
    assert h == "3a2526edee6cf4577cf7fb700ee18a28048e4967"


def test_path_smooth2():
    P = pp.smooth(
        points=[
            (20, 0),
            (40, 0),
            (80, 40),
            (80, 10),
            (100, 10),
        ],
        radius=4,
        corner_fun=pp.arc,
    )
    h = P.hash_geometry(precision=1e-4)
    assert h == "f075520ab38933b35f3c8af110627de29e34b695"


def test_path_rotate1():
    P = Path()
    P.rotate(47)
    h = P.hash_geometry(precision=1e-4)
    assert h == "f30d4e52e113a2954e7facee8144ac8bf4ea66c9"


def test_path_rotate2():
    P = pp.arc(radius=10, angle=90, num_pts=720)
    P.rotate(47)
    h = P.hash_geometry(precision=1e-4)
    assert h == "d3bc47a30787fb6d9514fdd91f45d619c9afff02"


def test_path_move1():
    P = Path()
    P.move((16, 17.7))
    h = P.hash_geometry(precision=1e-4)
    assert h == "6be1a6e602c7ce3d5560176aa7c9ebbf1b6788d5"


def test_path_move2():
    P = pp.arc(radius=10, angle=90, num_pts=720)
    P.move((16, 17.7))
    h = P.hash_geometry(precision=1e-4)
    assert h == "200e897720aa4d4a7f54afeaa2ca1faaa0e6b3de"


def test_path_bbox():
    P = pp.arc(radius=10, angle=90, num_pts=720)
    assert np.allclose(
        P.bbox, np.array([[6.123234e-16, 0.000000e00], [1.000000e01, 1.000000e01]])
    )


def test_path_append1():
    P1 = pp.arc(radius=12, angle=47, num_pts=720)
    P2 = pp.arc(radius=12, angle=63, num_pts=720)
    P = Path()
    P.append(P1)
    P.append(P2)
    h = P.hash_geometry(precision=1e-4)
    assert h == "57a2cec018e695db668437a9b3dcdd8c595e300f"


def test_path_append2():
    P1 = pp.arc(radius=12, angle=47, num_pts=720)
    P2 = pp.arc(radius=12, angle=63, num_pts=720)
    P = Path([P1, P2])
    h = P.hash_geometry(precision=1e-4)
    assert h == "57a2cec018e695db668437a9b3dcdd8c595e300f"


def test_path_append3():
    P1 = pp.arc(radius=12, angle=47, num_pts=720)
    P2 = pp.arc(radius=12, angle=-63, num_pts=720)
    P = Path([[P1, P2], [P2, P1, P2]])
    h = P.hash_geometry(precision=1e-4)
    assert h == "85646fba990cd2e72334df5d1e31678f090f6ce2"
