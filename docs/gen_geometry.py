
import phidl.utilies as pu


from phidl import Device


import phidl.geometry as pg
import os


def create_image(D, filename, filepath = '/_static/'):
    scale = 500/max(D.size)
    pu.write_svg(os.path.join([os.path.curdir(), filepath, filename]), scale = scale)


# example-rectangle
import phidl.geometry as pg
D = pg.rectangle(size = (4.5, 2), layer = 0)
pu.write_svg('rectangle.svg')


# example-bbox
import phidl.geometry as pg
D = pg.bbox(bbox = [(-1,-1),(3,4)], layer = 0)
pu.write_svg('bbox.svg')

# example-cross
import phidl.geometry as pg
D = pg.cross(length = 10, width = 3, layer = 0)
pu.write_svg('cross.svg')

# example-ellipse
import phidl.geometry as pg
D = pg.ellipse(radii = (10,5), angle_resolution = 2.5, layer = 0)
pu.write_svg('ellipse.svg')

# example-circle
import phidl.geometry as pg
D = pg.circle(radius = 10, angle_resolution = 2.5, layer = 0)
pu.write_svg('circle.svg')

# example-ring
import phidl.geometry as pg
D = pg.ring(radius = 10, width = 0.5, angle_resolution = 2.5, layer = 0)
pu.write_svg('ring.svg')

# example-arc
import phidl.geometry as pg
D = pg.arc(radius = 10, width = 0.5, theta = 45, start_angle = 0, angle_resolution = 2.5, layer = 0)
pu.write_svg('arc.svg')

# example-straight
import phidl.geometry as pg
D = pg.straight(size = (4,2), layer = 0)
pu.write_svg('straight.svg')

# example-L
import phidl.geometry as pg
D = pg.L(width = 1, size = (10,20) , layer = 0)
pu.write_svg('L.svg')

# example-C
import phidl.geometry as pg
D = pg.C(width = 1, size = (10,20) , layer = 0)
pu.write_svg('C.svg')

# example-offset
import phidl.geometry as pg
D = pg.offset(elements, distance = 0.1, join_first = True, precision = 1e-6, 
        num_divisions = [offset1,1], layer = 0)
pu.write_svg('.svg')

# example-invert
import phidl.geometry as pg
D = pg.invert(elements, border = 10, precision = 1e-6, layer = 0)
pu.write_svg('invert.svg')

# example-boolean
import phidl.geometry as pg
D = pg.boolean(A, B, operation, precision = 1e-6, num_divisions = [1,1], layer = 0)
pu.write_svg('boolean.svg')

# example-outline
import phidl.geometry as pg
D = pg.outline(elements, distance = 1, precision = 1e-6, layer = 0)
pu.write_svg('outline.svg')

# example-xor_diff
import phidl.geometry as pg
D = pg.xor_diff(A,B, precision = 1e-6)
pu.write_svg('xor_diff.svg')

# example-union
import phidl.geometry as pg
D = pg.union(D, by_layer = False, precision=1e-6, layer = 0)
pu.write_svg('union.svg')

# example-litho_steps
import phidl.geometry as pg
D = pg.litho_steps(
        line_widths = [litho_steps1,2,4,8,16],
        line_spacing = 10,
        height = 100,
        layer = 0
        )
pu.write_svg('.svg')

# example-litho_star
import phidl.geometry as pg
D = pg.litho_star(
        num_lines = 2litho_star0,
        line_width = 2,
        diameter = 200,
        layer = 0
        )
pu.write_svg('.svg')

# example-litho_calipers
import phidl.geometry as pg
D = pg.litho_calipers(
        notch_size = [litho_calipers2,5],
        notch_spacing = 2,
        num_notches = 11,
        offset_per_notch = 0.1,
        row_spacing = 0,
        layer1 = 1,
        layer2 = 2)
pu.write_svg('.svg')

# example-extract
import phidl.geometry as pg
D = pg.extract(D, layers = [0,1])
pu.write_svg('extract.svg')

# example-copy
import phidl.geometry as pg
D = pg.copy(D)
pu.write_svg('copy.svg')

# example-deepcopy
import phidl.geometry as pg
D = pg.deepcopy(D)
pu.write_svg('deepcopy.svg')

# example-copy_layer
import phidl.geometry as pg
D = pg.copy_layer(D, layer = 1, new_layer = 2)
pu.write_svg('copy_layer.svg')

# example-import_gds
import phidl.geometry as pg
D = pg.import_gds(filename, cellname = None, flatten = False)
pu.write_svg('import_gds.svg')

# example-preview_layerset
import phidl.geometry as pg
D = pg.preview_layerset(ls, size = 100, spacing = 100)
pu.write_svg('preview_layerset.svg')

# example-connector
import phidl.geometry as pg
D = pg.connector(midpoint = (0,0), width = 1, orientation = 0)
pu.write_svg('connector.svg')

# example-compass
import phidl.geometry as pg
D = pg.compass(size = (4,2), layer = 0)
pu.write_svg('compass.svg')

# example-compass_multi
import phidl.geometry as pg
D = pg.compass_multi(size = (4,2), ports = {'N':3,'S':4}, layer = 0)
pu.write_svg('compass_multi.svg')

# example-flagpole
import phidl.geometry as pg
D = pg.flagpole(size = (4,2), stub_size = (2,1), shape = 'p', taper_type = 'straight', layer = 0)
pu.write_svg('flagpole.svg')

# example-tee
import phidl.geometry as pg
D = pg.tee(size = (4,2), stub_size = (2,1), taper_type = None, layer = 0)
pu.write_svg('tee.svg')

# example-taper
import phidl.geometry as pg
D = pg.taper(length = 10, width1 = 5, width2 = None, port = None, layer = 0)
pu.write_svg('taper.svg')

# example-ramp
import phidl.geometry as pg
D = pg.ramp(length = 10, width1 = 5, width2 = 8, layer = 0)
pu.write_svg('ramp.svg')

# example-text
import phidl.geometry as pg
D = pg.text(text = 'abcd', size = 10, justify = 'left', layer = 0)
pu.write_svg('text.svg')

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
pu.write_svg('basic_die.svg')

# example-fill_rectangle
import phidl.geometry as pg
D = pg.fill_rectangle(D, fill_size = (40,10), avoid_layers = 'all', include_layers = None,
                    margin = 100, fill_layers = (0,1,3),
                   fill_densities = (0.5, 0.25, 0.7), fill_inverted = None, bbox = None)
pu.write_svg('fill_rectangle.svg')

# example-grating
import phidl.geometry as pg
D = pg.grating(num_periods = 20, period = 0.75, fill_factor = 0.5, width_grating = 5, length_taper = 10, width = 0.4, partial_etch = False)
pu.write_svg('grating.svg')

# example-test_via
import phidl.geometry as pg
D = pg.test_via(num_vias = 100, wire_width = 10, via_width = 15, via_spacing = 40, pad_size = (300,300), min_pad_spacing = 0,
                             pad_layer = 0, wiring1_layer = 1, wiring2_layer = 2, via_layer = 3)
pu.write_svg('test_via.svg')

# example-test_comb
import phidl.geometry as pg
D = pg.test_comb(pad_size = (200,200), wire_width = 1, wire_gap = 3,
              comb_layer = 0, overlap_zigzag_layer = 1,
              comb_pad_layer = None, comb_gnd_layer = None, overlap_pad_layer = None)
pu.write_svg('test_comb.svg')

# example-test_ic
import phidl.geometry as pg
D = pg.test_ic(wire_widths = [0.25, 0.5,1,2,4], wire_widths_wide = [0.75, 1.5, 3, 4, 6], pad_size = (200,200), pad_gap=75,
            wire_layer = 0, pad_layer = 1, gnd_layer = None)
pu.write_svg('test_ic.svg')

# example-test_res
import phidl.geometry as pg
D = pg.test_res(pad_size = [50,50],
                     num_squares = 1000,
                     width = 1,
                     res_layer = 0,
                     pad_layer = None,
                     gnd_layer = None)
pu.write_svg('test_res.svg')

# example-optimal_hairpin
import phidl.geometry as pg
D = pg.optimal_hairpin(width = 0.2, pitch = 0.6, length = 10,
    turn_ratio = 4, num_pts = 50, layer = 0)
pu.write_svg('optimal_hairpin.svg')

# example-optimal_step
import phidl.geometry as pg
D = pg.optimal_step(start_width = 10, end_width = 22, num_pts = 50, width_tol = 1e-3,
                 anticrowding_factor = 1.2, symmetric = False, layer = 0)
pu.write_svg('optimal_step.svg')

# example-optimal_90deg
import phidl.geometry as pg
D = pg.optimal_90deg(width = 100.0, num_pts = 15, length_adjust = 1, layer = 0)
pu.write_svg('optimal_90deg.svg')

# example-snspd
import phidl.geometry as pg
D = pg.snspd(wire_width = 0.2, wire_pitch = 0.6, size = (10,8),
        num_squares = None, turn_ratio = 4,
        terminals_same_side = False, layer = 0)
pu.write_svg('snspd.svg')

# example-snspd_expanded
import phidl.geometry as pg
D = pg.snspd_expanded(wire_width = 0.2, wire_pitch = 0.6, size = (10,8),
           num_squares = None, connector_width = 1, connector_symmetric = False,
            turn_ratio = 4, terminals_same_side = False, layer = 0)
pu.write_svg('snspd_expanded.svg')