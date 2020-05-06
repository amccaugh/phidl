from phidl import Device
import phidl
from matplotlib import pyplot as plt
import numpy as np
#def qp(*args):return None # Dummy quickplot


import phidl.geometry as pg
from phidl import quickplot as qp

import os


def create_image(D, filename, filepath = '_static/'):
    # if any(D.size == 0):
    #     D = pg.text('?')
    qp(D)
    fig = plt.gcf()
    # ax = plt.gca()
    scale = 0.75
    fig.set_size_inches(10*scale, 4*scale, forward=True)
    # ax.autoscale()
    # plt.draw()
    # plt.show(block = False)
    filename +=  '.png'
    filepathfull = os.path.join(os.path.curdir, filepath, filename)
    print(filepathfull)
    fig.savefig(filepathfull, dpi=int(96/scale))


# example-rectangle
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.rectangle(size = (4.5, 2), layer = 0)
qp(D) # quickplot the geometry
create_image(D, 'rectangle')


# example-bbox
import phidl.geometry as pg
from phidl import quickplot as qp

from phidl import Device
D = Device()
arc = D << pg.arc(radius = 10, width = 0.5, theta = 85, layer = 1).rotate(25)
 # Draw a rectangle around the arc we created by using the arc's bounding box
rect = D << pg.bbox(bbox = arc.bbox, layer = 0)
qp(D) # quickplot the geometry
create_image(D, 'bbox')

# example-cross
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.cross(length = 10, width = 0.5, layer = 0)
qp(D) # quickplot the geometry
create_image(D, 'cross')

# example-ellipse
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.ellipse(radii = (10,5), angle_resolution = 2.5, layer = 0)
create_image(D, 'ellipse')

# example-circle
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.circle(radius = 10, angle_resolution = 2.5, layer = 0)
qp(D) # quickplot the geometry
create_image(D, 'circle')

# example-ring
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.ring(radius = 5, width = 0.5, angle_resolution = 2.5, layer = 0)
qp(D) # quickplot the geometry
create_image(D, 'ring')

# example-arc
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.arc(radius = 2.0, width = 0.5, theta = 45, start_angle = 0,
           angle_resolution = 2.5, layer = 0)
qp(D) # quickplot the geometry
create_image(D, 'arc')

# example-straight
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.straight(size = (4,2), layer = 0)
qp(D) # quickplot the geometry
create_image(D, 'straight')

# example-L
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.L(width = 7, size = (10,20) , layer = 0)
qp(D) # quickplot the geometry
create_image(D, 'L')

# example-C
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.C(width = 7, size = (10,20) , layer = 0)
qp(D) # quickplot the geometry
create_image(D, 'C')

# example-offset
import phidl.geometry as pg
from phidl import quickplot as qp

from phidl import Device
# Create `T`, an ellipse and rectangle which will be offset (expanded / contracted)
T = Device()
e = T << pg.ellipse(radii = (10,5), layer = 1)
r = T << pg.rectangle(size = [15,5], layer = 2)
r.move([3,-2.5])

Texpanded = pg.offset(T, distance = 2, join_first = True, precision = 1e-6, 
        num_divisions = [1,1], layer = 0)
Tshrink = pg.offset(T, distance = -1.5, join_first = True, precision = 1e-6, 
        num_divisions = [1,1], layer = 0)

# Plot the original geometry, the expanded, and the shrunk versions
D = Device()
t1 = D.add_ref(T)
t2 = D.add_ref(Texpanded)
t3 = D.add_ref(Tshrink)
D.distribute([t1,t2,t3], direction = 'x', spacing = 5)
qp(D) # quickplot the geometry
create_image(D, 'offset')

# example-invert
import phidl.geometry as pg
from phidl import quickplot as qp

E = pg.ellipse(radii = (10,5))
D = pg.invert(E, border = 0.5, precision = 1e-6, layer = 0)
qp(D) # quickplot the geometry
create_image(D, 'invert')

# example-boolean
import phidl.geometry as pg
from phidl import quickplot as qp

from phidl import Device
E = pg.ellipse(radii = (10,5), layer = 1)
R = pg.rectangle(size = [15,5], layer = 2).movey(-1.5)
C = pg.boolean(A = E, B = R, operation = 'not', precision = 1e-6, num_divisions = [1,1], layer = 0)
# Other operations include 'and', 'or', 'xor', or equivalently 'A-B', 'B-A', 'A+B'

# Plot the originals and the result
D = Device()
D.add_ref(E)
D.add_ref(R)
D.add_ref(C).movex(30)
qp(D) # quickplot the geometry
create_image(D, 'boolean')

# example-outline
import phidl.geometry as pg
from phidl import quickplot as qp

from phidl import Device
# Create a blank device and add two shapes
X = Device()
X.add_ref(pg.cross(length = 25, width = 1, layer = 1))
X.add_ref(pg.ellipse(radii = [10,5], layer = 2))

O = pg.outline(X, distance = 1.5, precision = 1e-6, layer = 0)

# Plot the original geometry and the result
D = Device()
D.add_ref(X)
D.add_ref(O).movex(30)
qp(D) # quickplot the geometry
create_image(D, 'outline')

# example-xor_diff
import phidl.geometry as pg
from phidl import quickplot as qp
from phidl import Device

A = Device()
A.add_ref(pg.ellipse(radii = [10,5], layer = 1))
A.add_ref(pg.text('A')).move([3,0])
B = Device()
B.add_ref(pg.ellipse(radii = [11,4], layer = 1).movex(4))
B.add_ref(pg.text('B')).move([3.2,0])
X = pg.xor_diff(A = A, B = B, precision = 1e-6)

# Plot the original geometry and the result
# Upper left: A / Upper right: B
# Lower left: A and B / Lower right: A xor B "diff" comparison
D = Device()
D.add_ref(A).move([-15,25])
D.add_ref(B).move([15,25])
D.add_ref(A).movex(-15)
D.add_ref(B).movex(-15)
D.add_ref(X).movex(15)
qp(D) # quickplot the geometry
create_image(D, 'xor_diff')

# example-union
import phidl.geometry as pg
from phidl import quickplot as qp

D = Device()
D << pg.ellipse(layer = 0)
D << pg.ellipse(layer = 0).rotate(15*1)
D << pg.ellipse(layer = 0).rotate(15*2)
D << pg.ellipse(layer = 0).rotate(15*3)
D << pg.ellipse(layer = 1).rotate(15*4)
D << pg.ellipse(layer = 1).rotate(15*5)

# We have two options to unioning - take all polygons, regardless of 
# layer, and join them together (in this case on layer 4) like so:
D_joined = pg.union(D, by_layer = False, layer = 4)

# Or we can perform the union operate by-layer
D_joined_by_layer = pg.union(D, by_layer = True)

# Space out shapes
D.add_ref(D_joined).movex(25)
D.add_ref(D_joined_by_layer).movex(50)
qp(D) # quickplot the geometry
create_image(D, 'union')

# example-litho_steps
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.litho_steps(
        line_widths = [1,2,4,8,16],
        line_spacing = 10,
        height = 100,
        layer = 0
        )
qp(D) # quickplot the geometry
create_image(D, 'litho_steps')

# example-litho_star
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.litho_star(
        num_lines = 20,
        line_width = 0.4,
        diameter = 20,
        layer = 0
        )
qp(D) # quickplot the geometry
create_image(D, 'litho_star')

# example-litho_calipers
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.litho_calipers(
        notch_size = [1,5],
        notch_spacing = 2,
        num_notches = 7,
        offset_per_notch = 0.1,
        row_spacing = 0,
        layer1 = 1,
        layer2 = 2)
qp(D) # quickplot the geometry
create_image(D, 'litho_calipers')

# example-extract
import phidl.geometry as pg
from phidl import quickplot as qp
from phidl import Device

E = Device()
E.add_ref(pg.ellipse(layer = {0,1}))
#X = pg.ellipse(layer = {0,1})
D = pg.extract(E, layers = [0,1])
qp(D) # quickplot the geometry
create_image(D, 'extract')

# example-copy
import phidl.geometry as pg
from phidl import quickplot as qp

X = pg.ellipse()
D = pg.copy(X)
qp(D) # quickplot the geometry
create_image(D, 'copy')

# example-deepcopy
import phidl.geometry as pg
from phidl import quickplot as qp

X = pg.ellipse()
D = pg.deepcopy(X)
qp(D) # quickplot the geometry
create_image(D, 'deepcopy')

# example-copy_layer
import phidl.geometry as pg
from phidl import quickplot as qp

X = Device()
X << pg.ellipse(layer = 0)
X << pg.ellipse(layer = 1)
D = pg.copy_layer(X, layer = 1, new_layer = 2)
qp(D) # quickplot the geometry
create_image(D, 'copy_layer')

# example-import_gds
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.ellipse()
D.write_gds('myoutput.gds')
D = pg.import_gds(filename = 'myoutput.gds', cellname = None, flatten = False)
qp(D) # quickplot the geometry
create_image(D, 'import_gds')

# example-preview_layerset
import phidl.geometry as pg
from phidl import quickplot as qp

from phidl import LayerSet
lys = LayerSet()
lys.add_layer('p', color = 'lightblue', gds_layer = 1, gds_datatype = 0)
lys.add_layer('p+', color = 'blue', gds_layer = 2, gds_datatype = 0)
lys.add_layer('p++', color = 'darkblue', gds_layer = 3, gds_datatype = 0)
lys.add_layer('n', color = 'lightgreen', gds_layer = 4, gds_datatype = 0)
lys.add_layer('n+', color = 'green', gds_layer = 4, gds_datatype = 98)
lys.add_layer('n++', color = 'darkgreen', gds_layer = 5, gds_datatype = 99)
D = pg.preview_layerset(lys, size = 100, spacing = 100)
qp(D) # quickplot the geometry
create_image(D, 'preview_layerset')
phidl.reset()

# example-connector
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.connector(midpoint = (0,0), width = 1, orientation = 0)
qp(D) # quickplot the geometry
create_image(D, 'connector')

# example-compass
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.compass(size = (4,2), layer = 0)
qp(D) # quickplot the geometry
create_image(D, 'compass')

# example-compass_multi
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.compass_multi(size = (4,2), ports = {'N':3,'S':4}, layer = 0)
qp(D) # quickplot the geometry
create_image(D, 'compass_multi')

# example-flagpole
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.flagpole(size = (50,25), stub_size = (4,8), shape = 'p', taper_type = 'fillet', layer = 0)
# taper_type should be None, 'fillet', or 'straight'

qp(D) # quickplot the geometry
create_image(D, 'flagpole')

# example-tee
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.tee(size = (8,4), stub_size = (1,2), taper_type = 'fillet', layer = 0)
# taper_type should be None, 'fillet', or 'straight'

qp(D) # quickplot the geometry
create_image(D, 'tee')

# example-taper
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.taper(length = 10, width1 = 6, width2 = 4, port = None, layer = 0)
qp(D) # quickplot the geometry
create_image(D, 'taper')

# example-ramp
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.ramp(length = 10, width1 = 4, width2 = 8, layer = 0)
qp(D) # quickplot the geometry
create_image(D, 'ramp')

# example-text
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.text(text = 'Hello world!\nMultiline text\nLeft-justified', size = 10,
            justify = 'left', layer = 0)
# `justify` should be either 'left', 'center', or 'right'
qp(D) # quickplot the geometry
create_image(D, 'text')

# example-basic_die
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.basic_die(
              size = (5000, 5000),  # Size of die
              street_width = 100,   # Width of corner marks for die-sawing
              street_length = 1000, # Length of corner marks for die-sawing
              die_name = 'chip99',  # Label text
              text_size = 500,      # Label text size
              text_location = 'SW', # Label text compass location e.g. 'S', 'SE', 'SW'
              layer = 0,
              draw_bbox = False,
              bbox_layer = 99,
              )
qp(D) # quickplot the geometry
create_image(D, 'basic_die')

# example-fill_rectangle
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.fill_rectangle(D, fill_size = (40,10), avoid_layers = 'all', include_layers = None,
                    margin = 100, fill_layers = (0,1,3),
                   fill_densities = (0.5, 0.25, 0.7), fill_inverted = None, bbox = None)
qp(D) # quickplot the geometry
create_image(D, 'fill_rectangle')

# example-grating
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.grating(num_periods = 20, period = 0.75, fill_factor = 0.5, width_grating = 5, length_taper = 10, width = 0.4, partial_etch = False)
qp(D) # quickplot the geometry
create_image(D, 'grating')

# example-test_via
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.test_via(num_vias = 100, wire_width = 10, via_width = 15, via_spacing = 40, pad_size = (300,300), min_pad_spacing = 0,
                             pad_layer = 0, wiring1_layer = 1, wiring2_layer = 2, via_layer = 3)
qp(D) # quickplot the geometry
create_image(D, 'test_via')

# example-test_comb
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.test_comb(pad_size = (200,200), wire_width = 1, wire_gap = 3,
              comb_layer = 0, overlap_zigzag_layer = 1,
              comb_pad_layer = None, comb_gnd_layer = None, overlap_pad_layer = None)
qp(D) # quickplot the geometry
create_image(D, 'test_comb')

# example-test_ic
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.test_ic(wire_widths = [0.25, 0.5,1,2,4], wire_widths_wide = [0.75, 1.5, 3, 4, 6], pad_size = (200,200), pad_gap=75,
            wire_layer = 0, pad_layer = 1, gnd_layer = None)
qp(D) # quickplot the geometry
create_image(D, 'test_ic')

# example-test_res
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.test_res(pad_size = [50,50],
                     num_squares = 1000,
                     width = 1,
                     res_layer = 0,
                     pad_layer = None,
                     gnd_layer = None)
qp(D) # quickplot the geometry
create_image(D, 'test_res')

# example-optimal_hairpin
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.optimal_hairpin(width = 0.2, pitch = 0.6, length = 10,
    turn_ratio = 4, num_pts = 50, layer = 0)
qp(D) # quickplot the geometry
create_image(D, 'optimal_hairpin')

# example-optimal_step
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.optimal_step(start_width = 10, end_width = 22, num_pts = 50, width_tol = 1e-3,
                 anticrowding_factor = 1.2, symmetric = False, layer = 0)
qp(D) # quickplot the geometry
create_image(D, 'optimal_step')

# example-optimal_90deg
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.optimal_90deg(width = 100.0, num_pts = 15, length_adjust = 1, layer = 0)
qp(D) # quickplot the geometry
create_image(D, 'optimal_90deg')

# example-snspd
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.snspd(wire_width = 0.2, wire_pitch = 0.6, size = (10,8),
        num_squares = None, turn_ratio = 4,
        terminals_same_side = False, layer = 0)
qp(D) # quickplot the geometry
create_image(D, 'snspd')

# example-snspd_expanded
import phidl.geometry as pg
from phidl import quickplot as qp

D = pg.snspd_expanded(wire_width = 0.3, wire_pitch = 0.6, size = (10,8),
           num_squares = None, connector_width = 1, connector_symmetric = False,
            turn_ratio = 4, terminals_same_side = False, layer = 0)
qp(D) # quickplot the geometry
create_image(D, 'snspd_expanded')


# example-packer1
import phidl.geometry as pg
from phidl import quickplot as qp

np.random.seed(5)
D_list = [pg.ellipse(radii = np.random.rand(2)*n+2) for n in range(50)]
D_list += [pg.rectangle(size = np.random.rand(2)*n+2) for n in range(50)]

D_packed_list = pg.packer(
        D_list,                 # Must be a list or tuple of Devices
        spacing = 1.25,         # Minimum distance between adjacent shapes
        aspect_ratio = (2,1),   # (width, height) ratio of the rectangular bin
        max_size = (None,None), # Limits the size into which the shapes will be packed
        density = 1.05,          # Values closer to 1 pack tighter but require more computation
        sort_by_area = True,    # Pre-sorts the shapes by area
        verbose = False,
        )
D = D_packed_list[0] # Only one bin was created, so we plot that
qp(D) # quickplot the geometry
create_image(D, 'packer1')

# example-packer2
import phidl.geometry as pg
from phidl import quickplot as qp

np.random.seed(1)
D_list = [pg.ellipse(radii = np.random.rand(2)*n+2) for n in range(120)]
D_list += [pg.rectangle(size = np.random.rand(2)*n+2) for n in range(120)]

D_packed_list = pg.packer(
        D_list,                 # Must be a list or tuple of Devices
        spacing = 4,         # Minimum distance between adjacent shapes
        aspect_ratio = (1,1),   # Shape of the box
        max_size = (500,500),   # Limits the size into which the shapes will be packed
        density = 1.05,         # Values closer to 1 pack tighter but require more computation
        sort_by_area = True,    # Pre-sorts the shapes by area
        verbose = False,
        )

# Put all packed bins into a single device and spread them out with distribute()
F = Device()
[F.add_ref(D) for D in D_packed_list]
F.distribute(elements = 'all', direction = 'x', spacing = 100, separation = True)
qp(F)
create_image(F, 'packer2')


# example-distribute1
from phidl import Device

D = Device()
# Create differents-sized rectangles and add them to D
[D.add_ref(pg.rectangle(size = [n*15+20,n*15+20]).move((n,n*4))) for n in [0,2,3,1,2]]

qp(D) # quickplot the geometry
create_image(D, 'distribute1')


# example-distribute2
from phidl import Device

D = Device()
# Create differents-sized rectangles and add them to D
[D.add_ref(pg.rectangle(size = [n*15+20,n*15+20]).move((n,n*4))) for n in [0,2,3,1,2]]
# Distribute all the rectangles in D along the x-direction with a separation of 5
D.distribute(elements = 'all',   # either 'all' or a list of objects
             direction = 'x',    # 'x' or 'y'
             spacing = 5,
             separation = True)

qp(D) # quickplot the geometry
create_image(D, 'distribute2')

# example-distribute3
from phidl import Device

D = Device()
[D.add_ref(pg.rectangle(size = [n*15+20,n*15+20]).move((n,n*4))) for n in [0,2,3,1,2]]
D.distribute(elements = 'all', direction = 'x', spacing = 100, separation = False,
             edge = 'min') # edge must be either 'min', 'max', or 'center'

qp(D) # quickplot the geometry
create_image(D, 'distribute3')

# example-distribute4
from phidl import Device

D = Device()
[D.add_ref(pg.rectangle(size = [n*15+20,n*15+20]).move((n-10,n*4))) for n in [0,2,3,1,2]]
D.distribute(elements = 'all', direction = 'x', spacing = 100, separation = False,
             edge = 'center')  # edge must be either 'min', 'max', or 'center'

qp(D) # quickplot the geometry
create_image(D, 'distribute4')


# example-align1
from phidl import Device

D = Device()
# Create differents-sized rectangles and add them to D then distribute them
[D.add_ref(pg.rectangle(size = [n*15+20,n*15+20]).move((n,n*4))) for n in [0,2,3,1,2]]
D.distribute(elements = 'all', direction = 'x', spacing = 5, separation = True)

qp(D) # quickplot the geometry
create_image(D, 'align1')


# example-align2
from phidl import Device

D = Device()
# Create differents-sized rectangles and add them to D then distribute them
[D.add_ref(pg.rectangle(size = [n*15+20,n*15+20]).move((n,n*4))) for n in [0,2,3,1,2]]
D.distribute(elements = 'all', direction = 'x', spacing = 5, separation = True)

# Align top edges
D.align(elements = 'all', alignment = 'ymax')

qp(D) # quickplot the geometry
create_image(D, 'align2')

# example-align3
from phidl import Device

D = Device()
# Create differents-sized rectangles and add them to D then distribute them
[D.add_ref(pg.rectangle(size = [n*15+20,n*15+20]).move((n,n*4))) for n in [0,2,3,1,2]]
D.distribute(elements = 'all', direction = 'x', spacing = 5, separation = True)

# Align top edges
D.align(elements = 'all', alignment = 'y')

qp(D) # quickplot the geometry
create_image(D, 'align3')
