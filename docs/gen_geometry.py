
import phidl.utilities as pu
from phidl import Device


import phidl.geometry as pg
import os


def create_image(D, filename, size = 300, filepath = '_static/'):
    if any(D.size == 0):
        D = pg.text('?')
    scale = size/max(D.size)
    pu.write_svg(D, os.path.join(os.path.curdir, filepath, filename), scale = scale)


# example-rectangle
import phidl.geometry as pg
D = pg.rectangle(size = (4.5, 2), layer = 0)
create_image(D, 'rectangle.svg')


# example-bbox
import phidl.geometry as pg
A = pg.arc(radius = 10, width = 0.5, theta = 85, layer = 1)
D = pg.bbox(bbox = A.bbox, layer = 0)
create_image(D, 'bbox.svg')

# example-cross
import phidl.geometry as pg
D = pg.cross(length = 10, width = 0.5, layer = 0)
create_image(D, 'cross.svg')

# example-ellipse
import phidl.geometry as pg
D = pg.ellipse(radii = (10,5), angle_resolution = 2.5, layer = 0)
create_image(D, 'ellipse.svg')

# example-circle
import phidl.geometry as pg
D = pg.circle(radius = 10, angle_resolution = 2.5, layer = 0)
create_image(D, 'circle.svg')

# example-ring
import phidl.geometry as pg
D = pg.ring(radius = 10, width = 0.5, angle_resolution = 2.5, layer = 0)
create_image(D, 'ring.svg')

# example-arc
import phidl.geometry as pg
D = pg.arc(radius = 10, width = 0.5, theta = 45, start_angle = 0, angle_resolution = 2.5, layer = 0)
create_image(D, 'arc.svg')

# example-straight
import phidl.geometry as pg
D = pg.straight(size = (4,2), layer = 0)
create_image(D, 'straight.svg')

# example-L
import phidl.geometry as pg
D = pg.L(width = 1, size = (10,20) , layer = 0)
create_image(D, 'L.svg')

# example-C
import phidl.geometry as pg
D = pg.C(width = 1, size = (10,20) , layer = 0)
create_image(D, 'C.svg')

# example-offset
import phidl.geometry as pg
# Create geometry to be offset (expanded / contracted)
T = Device()
T << pg.ellipse(layer = 1)
T << pg.rectangle(layer = 2)

Texpanded = pg.offset(T, distance = 1, join_first = True, precision = 1e-6, 
        num_divisions = [1,1], layer = 0)
Tshrink = pg.offset(T, distance = 1.5, join_first = True, precision = 1e-6, 
        num_divisions = [1,1], layer = 0)

D = Device()
D.add_ref(Texpanded).movex(0)
D.add_ref(Tshrink).movex(20)
create_image(D, 'offset.svg')

# example-invert
import phidl.geometry as pg
D = pg.invert(pg.ellipse(), border = 10, precision = 1e-6, layer = 0)
create_image(D, 'invert.svg')

# example-boolean
import phidl.geometry as pg
A = pg.ellipse()
B = pg.rectangle()
D = pg.boolean(A, B, operation = 'not', precision = 1e-6, num_divisions = [1,1], layer = 0)
create_image(D, 'boolean.svg')

# example-outline
import phidl.geometry as pg
D = pg.outline(pg.ellipse(), distance = 1, precision = 1e-6, layer = 0)
create_image(D, 'outline.svg')

# example-xor_diff
import phidl.geometry as pg
D = pg.xor_diff(A,B, precision = 1e-6)
create_image(D, 'xor_diff.svg')

# example-union
import phidl.geometry as pg
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
create_image(D, 'union.svg')

# example-litho_steps
import phidl.geometry as pg
D = pg.litho_steps(
        line_widths = [1,2,4,8,16],
        line_spacing = 10,
        height = 100,
        layer = 0
        )
create_image(D, 'litho_steps.svg')

# example-litho_star
import phidl.geometry as pg
D = pg.litho_star(
        num_lines = 20,
        line_width = 2,
        diameter = 200,
        layer = 0
        )
create_image(D, 'litho_star.svg')

# example-litho_calipers
import phidl.geometry as pg
D = pg.litho_calipers(
        notch_size = [2,5],
        notch_spacing = 2,
        num_notches = 11,
        offset_per_notch = 0.1,
        row_spacing = 0,
        layer1 = 1,
        layer2 = 2)
create_image(D, 'litho_calipers.svg')

# example-extract
import phidl.geometry as pg
X = pg.ellipse(layer = {0,1})
D = pg.extract(X, layers = [0,1])
create_image(D, 'extract.svg')

# example-copy
import phidl.geometry as pg
X = pg.ellipse()
D = pg.copy(X)
create_image(D, 'copy.svg')

# example-deepcopy
import phidl.geometry as pg
X = pg.ellipse()
D = pg.deepcopy(X)
create_image(D, 'deepcopy.svg')

# example-copy_layer
import phidl.geometry as pg
X = Device()
X << pg.ellipse(layer = 0)
X << pg.ellipse(layer = 1)
D = pg.copy_layer(X, layer = 1, new_layer = 2)
create_image(D, 'copy_layer.svg')

# example-import_gds
import phidl.geometry as pg
D = pg.ellipse()
D.write_gds('myoutput.gds')
D = pg.import_gds(filename = 'myoutput.gds', cellname = None, flatten = False)
create_image(D, 'import_gds.svg')

# example-preview_layerset
import phidl.geometry as pg
from phidl import LayerSet
lys = LayerSet()
lys.add_layer('intrinsic', color = 'gray', gds_layer = 0, gds_datatype = 0)
lys.add_layer('p', color = 'lightblue', gds_layer = 1, gds_datatype = 0)
lys.add_layer('p+', color = 'blue', gds_layer = 2, gds_datatype = 0)
lys.add_layer('p++', color = 'darkblue', gds_layer = 3, gds_datatype = 0)
lys.add_layer('n', color = 'lightgreen', gds_layer = 4, gds_datatype = 0)
lys.add_layer('n+', color = 'green', gds_layer = 4, gds_datatype = 0)
lys.add_layer('n++', color = 'darkgreen', gds_layer = 5, gds_datatype = 0)
D = pg.preview_layerset(lys, size = 100, spacing = 100)
create_image(D, 'preview_layerset.svg')

# example-connector
import phidl.geometry as pg
D = pg.connector(midpoint = (0,0), width = 1, orientation = 0)
create_image(D, 'connector.svg')

# example-compass
import phidl.geometry as pg
D = pg.compass(size = (4,2), layer = 0)
create_image(D, 'compass.svg')

# example-compass_multi
import phidl.geometry as pg
D = pg.compass_multi(size = (4,2), ports = {'N':3,'S':4}, layer = 0)
create_image(D, 'compass_multi.svg')

# example-flagpole
import phidl.geometry as pg
D = pg.flagpole(size = (4,2), stub_size = (2,1), shape = 'p', taper_type = 'straight', layer = 0)
create_image(D, 'flagpole.svg')

# example-tee
import phidl.geometry as pg
D = pg.tee(size = (4,2), stub_size = (2,1), taper_type = None, layer = 0)
create_image(D, 'tee.svg')

# example-taper
import phidl.geometry as pg
D = pg.taper(length = 10, width1 = 5, width2 = None, port = None, layer = 0)
create_image(D, 'taper.svg')

# example-ramp
import phidl.geometry as pg
D = pg.ramp(length = 10, width1 = 5, width2 = 8, layer = 0)
create_image(D, 'ramp.svg')

# example-text
import phidl.geometry as pg
D = pg.text(text = 'abcd', size = 10, justify = 'left', layer = 0)
create_image(D, 'text.svg')

# example-basic_die
import phidl.geometry as pg
D = pg.basic_die(
              size = (10000, 10000),
              street_width = 100,
              street_length = 1000,
              die_name = 'chip99',
              text_size = 100,
              text_location = 'SW',
              layer = 0,
              draw_bbox = True,
              bbox_layer = 99,
              )
create_image(D, 'basic_die.svg')

# example-fill_rectangle
import phidl.geometry as pg
D = pg.fill_rectangle(D, fill_size = (40,10), avoid_layers = 'all', include_layers = None,
                    margin = 100, fill_layers = (0,1,3),
                   fill_densities = (0.5, 0.25, 0.7), fill_inverted = None, bbox = None)
create_image(D, 'fill_rectangle.svg')

# example-grating
import phidl.geometry as pg
D = pg.grating(num_periods = 20, period = 0.75, fill_factor = 0.5, width_grating = 5, length_taper = 10, width = 0.4, partial_etch = False)
create_image(D, 'grating.svg')

# example-test_via
import phidl.geometry as pg
D = pg.test_via(num_vias = 100, wire_width = 10, via_width = 15, via_spacing = 40, pad_size = (300,300), min_pad_spacing = 0,
                             pad_layer = 0, wiring1_layer = 1, wiring2_layer = 2, via_layer = 3)
create_image(D, 'test_via.svg')

# example-test_comb
import phidl.geometry as pg
D = pg.test_comb(pad_size = (200,200), wire_width = 1, wire_gap = 3,
              comb_layer = 0, overlap_zigzag_layer = 1,
              comb_pad_layer = None, comb_gnd_layer = None, overlap_pad_layer = None)
create_image(D, 'test_comb.svg')

# example-test_ic
import phidl.geometry as pg
D = pg.test_ic(wire_widths = [0.25, 0.5,1,2,4], wire_widths_wide = [0.75, 1.5, 3, 4, 6], pad_size = (200,200), pad_gap=75,
            wire_layer = 0, pad_layer = 1, gnd_layer = None)
create_image(D, 'test_ic.svg')

# example-test_res
import phidl.geometry as pg
D = pg.test_res(pad_size = [50,50],
                     num_squares = 1000,
                     width = 1,
                     res_layer = 0,
                     pad_layer = None,
                     gnd_layer = None)
create_image(D, 'test_res.svg')

# example-optimal_hairpin
import phidl.geometry as pg
D = pg.optimal_hairpin(width = 0.2, pitch = 0.6, length = 10,
    turn_ratio = 4, num_pts = 50, layer = 0)
create_image(D, 'optimal_hairpin.svg')

# example-optimal_step
import phidl.geometry as pg
D = pg.optimal_step(start_width = 10, end_width = 22, num_pts = 50, width_tol = 1e-3,
                 anticrowding_factor = 1.2, symmetric = False, layer = 0)
create_image(D, 'optimal_step.svg')

# example-optimal_90deg
import phidl.geometry as pg
D = pg.optimal_90deg(width = 100.0, num_pts = 15, length_adjust = 1, layer = 0)
create_image(D, 'optimal_90deg.svg')

# example-snspd
import phidl.geometry as pg
D = pg.snspd(wire_width = 0.2, wire_pitch = 0.6, size = (10,8),
        num_squares = None, turn_ratio = 4,
        terminals_same_side = False, layer = 0)
create_image(D, 'snspd.svg')

# example-snspd_expanded
import phidl.geometry as pg
D = pg.snspd_expanded(wire_width = 0.2, wire_pitch = 0.6, size = (10,8),
           num_squares = None, connector_width = 1, connector_symmetric = False,
            turn_ratio = 4, terminals_same_side = False, layer = 0)
create_image(D, 'snspd_expanded.svg')