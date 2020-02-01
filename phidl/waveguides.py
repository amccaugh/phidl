''' Waveguide cross sections
    WG_XS are not Devices. They produce waveguides such as straight sections, bends, euler bends, S bends.
    A WG_XS contains multiple WG_XS_Components, which each have a layer, width, and offset.

    Waveguides produced by WG_XS methods are Devices that follow these conventions:
        - two ports: 'wg_in_1' and 'wg_out_1'
        - length info: WG.info['wg_length']
        - xsect info: WG.info['xsect'] (assigned to the WG_XS that produced the waveguide)
'''
import numpy as np
from copy import deepcopy

from phidl.device_layout import Device, Layer, Port
import phidl.routing as pr, phidl.geometry as pg

#: The spacing of points around bend sections. It depends on your lithography tool.
#: For EBeam it should be about 4 nm, and the GDS will be big (unavoidable)
#: For photolith it should be >50 nm, and the GDS will be small
#: For developing your layout, it can be 500nm, and layout builds will be much faster.
minimum_bent_edge_length = 0.1


class WG_XS(object):
    ''' Waveguide cross section specification.
        It generates various types of Devices, such as straight sections and S-bends.

        The resulting Devices have the convention of two ports: 'wg_in_1' and 'wg_out_1'
            as well as length information in Device.info['wg_length']
    '''
    def __init__(self, radius=5, loss=0, ngroup=4, wavelength=1.550, components=None, explicit_rib_width=None):
        ''' "radius" is the minimum and default radius, not necessarily the only radius
            "components" is a list of WG_XS_Components that each have layer, width, and offset
        '''
        self.radius = float(radius)
        self.loss = float(loss)
        self.ngroup = float(ngroup)
        self.wavelength = float(wavelength)
        self.components = components if components is not None else []
        self.explicit_rib_width = explicit_rib_width

    def copy(self):
        return deepcopy(self)

    def __repr__(self):
        core_attributes = ['radius', 'loss', 'ngroup', 'wavelength', 'components']
        attrstrs = ('{k}={my_attr}'.format(k=k, my_attr=getattr(self,k)) for k in core_attributes)
        fullstr = ',\n'.join(attrstrs)
        return '{name}(\n{fullstr}\n)'.format(name = type(self).__name__, fullstr = fullstr)

    def __eq__(self, other):
        for attr in ['radius', 'loss', 'ngroup', 'wavelength']:
            if getattr(self, attr) != getattr(other, attr):
                return False
        # components might not be in the same order, so use get by layer
        layers = set(comp.layer for comp in self.components)
        for layer in layers:
            if self.get_by_layer(layer) != other.get_by_layer(layer):
                return False
        return True


    ## xsection utilities ##

    #: determines behavior of "rib_width". You can set this at the class or instance level if needed
    rib_layer = Layer(name='wg_deep')

    def get_by_layer(self, lay):
        ''' Gets the WG_XS_Components based on the name of the Layer argument. There can be multiple components.
            Guaranteed to be sorted in terms of increasing offset and then width

            Returns:
                (list): matching components
        '''
        list_all = []
        for comp in self.components:
            try:
                lay_name = lay.name
            except:
                if isinstance(lay, str):
                    # raise ValueError('Deprecation Warning: use Layer, not string, arguments to get_by_layer.')
                    lay_name = lay
                else:
                    raise
            if comp.layer.name == lay_name:
                list_all.append(comp)
        list_all.sort(key=lambda co: (co.offset, co.width))
        return list_all

    def flipped(self):
        ''' mirrors the waveguide xsection. Useful for example if it has a PN junction '''
        new_xsect = self.copy()
        for comp in new_xsect.components:
            comp.offset *= -1
        return new_xsect

    @property
    def rib_width(self):
        ''' Width of the wg_deep layer. For convenience because this is used commonly '''
        si_comps = self.get_by_layer(self.rib_layer)
        if self.explicit_rib_width is not None:
            return self.explicit_rib_width
        if len(si_comps) > 1:
            raise ValueError('Rib width not defined for Xsections with multiple wg_deep components.')
        rib_comp = si_comps[0]
        if rib_comp.offset != 0:
            raise ValueError('WG_XS Warning: rib is off-centered, which is weird.\nIf this is right, write a try, except')
        return rib_comp.width

    @property
    def full_width(self):
        ''' Width of the full WG_XS '''
        going_min = 0
        going_max = 0
        for comp in self.components:
            going_max = max(going_max, comp.max)
            going_min = min(going_min, comp.min)
        return (going_max - going_min)


    ## Device generators ##

    def cell_straight(self, length):
        ''' A straight waveguide Device.
            length=0 will give an empty Device with ports 'wg_in_1' and 'wg_in_2' centered at [0,0]
        '''
        if length < 0:
            print('Warning: Waveguide length was negative.')
        WG = Device('WG straight')
        for comp in self.components:
            el = WG.add_ref(pg.rectangle(size=[length, comp.width], layer=comp.layer))
            el.y -= comp.width / 2
            el.y += comp.offset
        WG.add_port(name = 'wg_in_1', midpoint = [0,0], width = self.full_width, orientation = 180)
        WG.add_port(name = 'wg_out_1', midpoint = [length, 0], width = self.full_width, orientation = 0)
        WG.info['wg_length'] = length
        WG.info['xsect'] = self
        return WG.flatten()

    def cell_bend(self, theta=90, radius=None, angle_resolution=None, euler=False):
        ''' A circular arc waveguide Device.

            By default, radius is the waveguide minimum radius.
            Angle resolution is degrees. If it is None, it will be determined by minimum_bent_edge_length and radius
            The euler option creates a gradually bent waveguide. Use cell_euler_bend for more advanced options.
        '''
        if theta == 0:
            return self.cell_straight(0)
        if euler:
            return self.cell_euler_bend(theta, radius, angle_resolution=angle_resolution)
        BEND = Device('WG bend')
        if radius is None:
            radius = self.radius
        if angle_resolution is None:
            angle_resolution = minimum_bent_edge_length / radius * 180 / np.pi
        angle_resolution = min(abs(angle_resolution), abs(theta/2))
        angle_resolution = max(abs(angle_resolution), 0.001)

        if theta < 0:
            start_angle = 90
        else:
            start_angle = -90
        maxwidth = 0
        angular_geometry = dict(theta=theta, start_angle=start_angle, angle_resolution=angle_resolution)
        central_el = None
        for comp in self.components:
            this_radius = radius + comp.offset * (1 if theta < 0 else -1)
            if theta != 0:
                arc_obj = pr._arc(radius=this_radius, width=comp.width, layer=comp.layer, **angular_geometry)
                # arc_obj.fracture(max_points=400)
                el = BEND << arc_obj
                el.move(el.ports[1], (0, comp.offset))
            maxwidth = max(maxwidth, comp.width)
            if comp.offset == 0:
                central_el = el
        if central_el is None:
            print('You have a waveguide with no central component. It\'s going to cause problems with connecting bent waveguides')
            central_el = el
        if theta != 0:
            BEND.add_port(name = 'wg_in_1', midpoint = central_el.ports[1].midpoint, width = maxwidth, orientation = central_el.ports[1].orientation)
            BEND.add_port(name = 'wg_out_1', midpoint = central_el.ports[2].midpoint, width = maxwidth, orientation = central_el.ports[2].orientation)
            BEND.flatten()
        else:
            BEND.add_port(name = 'wg_in_1', midpoint = (0, 0), width = maxwidth, orientation = 180)
            BEND.add_port(name = 'wg_out_1', midpoint = (0, 0), width = maxwidth, orientation = 0)
        BEND.info['wg_length'] = 2 * np.pi * radius * theta / 360
        BEND.info['xsect'] = self
        BEND.info['Reff'] = radius
        return BEND

    def _make_half_euler(self, theta=90, radius=None, angle_resolution=None, num_steps=10, angular_coverage=None, flip_xsect=False):
        ''' Helper for making Euler curves

            radius is the minimum radius of the curve, in the central segment
            angular_coverage is the angle of section that will be greater than the minimum radius.
            angle_resolution feeds through to cell_bend
            num_steps is number of bend segments of different radii
        '''
        if angular_coverage is None:
            angular_coverage = abs(theta) / 4
        if radius is None:
            radius = self.radius
        if angle_resolution is None:
            angle_resolution = minimum_bent_edge_length / radius * 180 / np.pi

        #: the arcangle that is a solid turn of the minimum radius.
        circular_arcangle = abs(theta) / 2 - angular_coverage
        if circular_arcangle < 0:
            raise ValueError('Cannot make the gradual section bigger than the whole bend')

        #determines the increment in radius through its inverse from 0 to 1/r
        inc_rad = radius * num_steps
        angle_step = angular_coverage / num_steps

        HALF_EULER = Device('WG HalfEuler')
        xsect = self if flip_xsect else self.flipped()
        HALF_EULER.info['wg_length'] = 0
        for x in range(num_steps):
            sub_arc = HALF_EULER << xsect.cell_bend(theta=angle_step * np.sign(theta),
                                                   radius=inc_rad / (x+1),
                                                   angle_resolution=angle_resolution,
                                                   euler=False)
            HALF_EULER.info['wg_length'] += sub_arc.info['wg_length']
            if x>0:
                sub_arc.connect('wg_in_1', prev_arc.ports['wg_out_1'])
            else:
                HALF_EULER.add_port(name='wg_in_1', port = sub_arc.ports['wg_in_1'])
            prev_arc = sub_arc

        #now connect a regular bend for the normal curved portion
        sub_circle = HALF_EULER << xsect.cell_bend(theta=circular_arcangle * np.sign(theta),
                                                  radius=radius,
                                                  angle_resolution=angle_resolution)
        HALF_EULER.info['wg_length'] += sub_circle.info['wg_length']
        HALF_EULER.info['xsect'] = self
        sub_circle.connect('wg_in_1', prev_arc.ports['wg_out_1'])
        HALF_EULER.add_port(name='wg_out_1', port=sub_circle.ports['wg_out_1'])
        return HALF_EULER

    def cell_euler_bend(self, theta=90, radius=None, angle_resolution=None, num_steps=10, angular_coverage=None):
        ''' Construct a series of sub-arcs with equal angles
            but gradually decreasing bend radius.
            This eliminates straight-to-bend losses.

            radius is the minimum radius of the curve in the central segment. If unspecified, radius will the the minimum set by this WG_XS.
            angular_coverage is the angle of section that will be greater than the minimum radius.
            angle_resolution feeds through to cell_bend. If it is None, it will be determined by minimum_bent_edge_length and radius
            num_steps is number of bend segments of different radii

            The output Device has a Device.info['Reff'] item that is the radius of the equivalent circular bend
        '''
        if theta == 0:
            return self.cell_straight(0)
        EULER = Device('WG Euler')
        halfeuler_kwargs = dict(theta=theta,
                                radius=radius,
                                angle_resolution=angle_resolution,
                                num_steps=num_steps,
                                angular_coverage=angular_coverage)
        HALF = self._make_half_euler(flip_xsect=True, **halfeuler_kwargs)
        half1 = EULER << HALF
        half2 = EULER << HALF
        half2.reflect()
        half2.connect('wg_out_1', half1.ports['wg_out_1'])
        EULER = pg.union(EULER, by_layer=True)

        EULER.add_port('wg_in_1', port=half1.ports['wg_in_1'])
        EULER.add_port('wg_out_1', port=half2.ports['wg_in_1'])
        EULER.info['wg_length'] = half1.info['wg_length'] + half2.info['wg_length']
        EULER.info['xsect'] = self

        # calculate effective radius
        port_dydx = np.array(EULER.ports['wg_out_1'].midpoint) - np.array(EULER.ports['wg_in_1'].midpoint)
        port_dist = np.sqrt(np.sum(port_dydx ** 2))
        EULER.info['Reff'] = abs(port_dist / 2 / np.sin(np.radians(theta/2)))
        return EULER

    def cell_s_bend(self, max_slope=45, radius=None, angle_resolution=None, num_steps=10, angular_coverage=None):
        ''' Two euler bends of equal, opposite turn in an S configuration

            max_slope refers to the angle of the waveguide in the center, relative to the in/out angles.
            See cell_euler_bend for description of other arguments.

            No guarantees about the sizes or offsets.
            You usually don't care since you can use the connect functions
            If you would like to do S bends based on precise offset:
                - use cell_s_bend_by_offset (circular arcs). Have to accept the straight-to-bend loss
                - guess and check or do a binary search over this method until the desired offset is achieved
        '''
        SBEND = Device('WG Sbend')
        euler_kwargs = dict(theta=max_slope,
                                radius=radius,
                                angle_resolution=angle_resolution,
                                num_steps=num_steps,
                                angular_coverage=angular_coverage)
        half1 = SBEND << self.cell_euler_bend(**euler_kwargs)
        half2 = SBEND << self.cell_euler_bend(**euler_kwargs)
        half2.connect('wg_out_1', half1.ports['wg_out_1'])

        SBEND.add_port('wg_in_1', port=half1.ports['wg_in_1'])
        SBEND.add_port('wg_out_1', port=half2.ports['wg_in_1'])
        SBEND.info['xsect'] = self
        return SBEND.flatten()

    def cell_s_bend_by_offset(self, longitudinal_offset, lateral_offset, radius=None):
        ''' Two opposing circular segments that give a specified offset

            longitudinal_offset: in the direction of the input port
            lateral_offset: perpendicular to the direction of the port
        '''
        if radius is None:
            radius = self.radius
        if longitudinal_offset < 2 * radius:
            # case 1: too tight
            raise ValueError('Requested S bend is too tight. Longitudinal length is too small')
        theta = np.arccos(1 - lateral_offset/2/radius) * 180/np.pi
        if np.isnan(theta) or abs(theta) > 90:
            # case 2: lateral offset is very big, so use 5 sections with central piece
            theta = np.sign(lateral_offset) * 90
            central = self.cell_straight(lateral_offset - 2 * radius)
        elif longitudinal_offset > 2 * radius:
            # case 3: longitudinal offset to accomodate the bends, so use 4 sections
            central = self.cell_straight(0)

        bend1 = self.cell_bend(radius=radius, theta=theta, euler=False)
        bend2 = self.cell_bend(radius=radius, theta=-theta, euler=False)
        inner_S = concatenate_waveguides([bend1, central, bend2])
        port_spacing = inner_S.ports['wg_out_1'].midpoint[0] - inner_S.ports['wg_in_1'].midpoint[0]
        entry_exit = self.cell_straight(longitudinal_offset/2 - port_spacing/2)
        return concatenate_waveguides([entry_exit, bend1, central, bend2, entry_exit])

    def cell_taper(self, wg_dest=None, taper_len=10, keep_layernames=['wg_shallow'], min_tip_width=0.1,
                   route_basic_options = dict(path_type='sine', width_type='sine')):
        ''' Creates a taper between this WG_XS and another. Ports are the widths of the xsections

            Args:
                wg_dest: the other WG_XS. If None, it will taper to nothing, typically acting as a WG terminator
                taper_len: length of taper
                keep_layernames:
                    If any layer is not present in one of the two, it is ignored,
                    unless it is in keep_layernames, in which case its width is set to min_tip_width
                min_tip_width: minimum width when there is a layer not present in of the WG_XSs. Use it to respect DRC.
                route_basic_options: shape properties of the taper.
                    "width_type='straight'" will give trapezoidal instead of smooth shapes.
                    "path type" doesn't have any effect because this is straight.
        '''
        if wg_dest is None:
            wg_dest = WG_XS(components=[])
            keep_layernames = [comp.layer.name for comp in self.components]
        # Find all of the relevant layers
        all_layers = set()
        for c in self.components + wg_dest.components:
            all_layers.add(c.layer)
        TAPER = Device()

        for lay in all_layers:
            src_comps = self.get_by_layer(lay)
            dest_comps = wg_dest.get_by_layer(lay)

            # prune it if only one is present, unless it is explicitly in keep_layernames
            if len(src_comps) == 0 or len(dest_comps) == 0:
                if lay.name in keep_layernames:
                    if len(src_comps) == 0:
                        new_comp = dest_comps[0].copy()
                        new_comp.width = min_tip_width
                        src_comps.append(new_comp)
                    elif len(dest_comps) == 0:
                        new_comp = src_comps[0].copy()
                        new_comp.width = min_tip_width
                        dest_comps.append(new_comp)
                else:
                    continue

            # If there is not the same number of components for this layer, pick off the remainder
            # An alternate behavior would be setting that width to zero
            while len(src_comps) != len(dest_comps):
                if len(src_comps) < len(dest_comps):
                    dest_comps.pop(0 if dest_comps[0].offset != 0 else -1)
                else:
                    src_comps.pop(0 if src_comps[0].offset != 0 else -1)

            # We have to make temporary ports because of route_basic.
            for sco, dco in zip(src_comps, dest_comps):
                src_po = Port(midpoint=(0, sco.offset),
                              width=sco.width, orientation=0)
                dest_po = Port(midpoint=(taper_len, dco.offset),
                               width=dco.width, orientation=180)
                TAPER << pr.route_basic(src_po, dest_po, layer=lay, **route_basic_options)

        # figure out the ports
        port_extrema = np.zeros((2, 2))
        for i, wg in enumerate([self, wg_dest]):
            for co in wg.components:
                port_extrema[i, 0] = min(port_extrema[i, 0], co.min)
                port_extrema[i, 1] = max(port_extrema[i, 1], co.max)
        port_ys = np.mean(port_extrema, axis=1)
        port_widths = np.diff(port_extrema, axis=1)[:,0]
        TAPER.add_port(name = 'wg_in_1', midpoint = [0, port_ys[0]],
                   width = port_widths[0], orientation = 180)
        TAPER.add_port(name = 'wg_out_1', midpoint = [taper_len, port_ys[1]],
                   width = port_widths[1], orientation = 0)
        TAPER.info['wg_length'] = taper_len
        TAPER.info['xsect'] = [self, wg_dest]
        return TAPER.flatten()

    def route_points(self, point_list, radius=None, euler=False):
        ''' A connected sequence of sections of this waveguide.
            Points correspond to the intersections of straight segments.
            This can handle non-Manhattan points
        '''
        if radius is None:
            radius = self.radius

        # Compute angles and lengths (not accounting for arcs) of segments
        points = np.asarray(point_list)
        dxdy = np.diff(points, axis=0)
        abs_angles = (np.arctan2(dxdy[:,1], dxdy[:,0])).tolist()
        abs_angles = np.array(abs_angles + [abs_angles[-1]]) * 180 / np.pi
        turns = ((abs_angles[1:] - abs_angles[:-1]) + 180) % 360 - 180
        if any(abs(turns) > 165):
            print('Warning: very sharp turns')
        lengths = np.sqrt(np.sum(dxdy ** 2, axis=1))

        # Check for repeated points
        nz = np.nonzero(lengths)
        lengths = lengths[nz]
        turns = turns[nz]

        # Make the bend Devices
        bend_refs = []
        eff_radii = []  # keeps track of radii of Euler segments
        straight_refs = []
        for turn_angle in turns:
            this_bend = self.cell_bend(turn_angle, radius=radius, euler=euler)
            bend_refs.append(this_bend)
            eff_radii.append(this_bend.info.get('Reff', radius))

        # adjust the lengths to accomodate bend sizes
        correction_arr = eff_radii[:-1] * abs(np.tan(turns[:-1] / 2 * np.pi / 180))
        lengths[1:] -= correction_arr
        lengths[:-1] -= correction_arr
        if any(lengths < 0):
            print('Warning: During routing, length was negative ({}). Points are too close together or turns are too sharp'.format(lengths))

        # Straight section Devices
        for straight_length in lengths:
            straight_refs.append(self.cell_straight(straight_length))

        # Interleave the sections and stick together
        all_secs = sum(zip(straight_refs, bend_refs), ())
        WG = concatenate_waveguides(all_secs)
        WG.rotate(abs_angles[0])
        WG.move(point_list[0])
        WG.info['xsect'] = self
        return WG.flatten()

    def route_ports_manhattan(self, port1, port2, radius=None, euler=False):
        ''' Make a waveguide between the two ports.
            Detects and handles cases where relative port orientation is 0, 90, 180, or 270 degrees.
            For S shapes (180 degree difference), the perpendicular segment occurs closer to port1, so the order of arguments matters
        '''
        if radius is None:
            radius = self.radius
        points = manhattan_points(port1, port2, radius=radius)
        return self.route_points(points, radius=radius, euler=euler)


def manhattan_points(port1, port2, radius=5):
    ''' Returns point list corresponding to manhattan joints between two ports.
        Detects and handles cases where relative port orientation is 0, 90, 180, or 270 degrees.
        For S shapes (180 degree difference), the perpendicular segment occurs closer to port1, so the order of arguments matters

        There is no layout or shapes in this function. You use the points elsewhere.
        That means the radius argument does not matter if there is a -90 or 90 degree difference – there is only one joint
    '''
    # Vector math helper functions
    def rotate(coords, angle_degrees):
        ''' rotates coords around the origin '''
        angle_rad = np.radians(angle_degrees)
        new_x = coords[:,0] * np.cos(angle_rad) - coords[:,1] * np.sin(angle_rad)
        new_y = coords[:,0] * np.sin(angle_rad) + coords[:,1] * np.cos(angle_rad)
        return np.array([new_x, new_y]).T

    def frame(coords, port):
        ''' puts coords in the frame of reference of the provided port '''
        return rotate(coords - port.midpoint, -port.orientation)

    def unframe(coords, port):
        ''' opposite of frame '''
        return rotate(coords, port.orientation) + port.midpoint

    # start and end points in the frame of reference of port1
    point_arr = np.array([port1.midpoint, port2.midpoint])
    framed_points = frame(point_arr, port1)

    # Three cases of relative angle
    dtheta = (port2.orientation - port1.orientation) % 360
    if dtheta == 0:
        turnx = np.max(framed_points[:,0]) + radius
        framed_points = np.insert(framed_points, 1, [turnx, framed_points[0][1]], axis=0)
        framed_points = np.insert(framed_points, 2, [turnx, framed_points[-1][1]], axis=0)
    elif dtheta in [90, 270]:
        rel_point = [framed_points[1,0], 0]
        framed_points = np.insert(framed_points, 1, rel_point, axis=0)
    elif dtheta == 180:
        if framed_points[1][1] != 0:
            rel_1 = [framed_points[0,0] + radius, 0]
            rel_2 = [rel_1[0], framed_points[1][1]]
            framed_points = np.insert(framed_points, 1, rel_1, axis=0)
            framed_points = np.insert(framed_points, 2, rel_2, axis=0)
        # otherwise, it is just straight and there is no middle joint
    else:
        raise ValueError('Ports are not on a manhattan grid')
    unframed_points = unframe(framed_points, port1)
    return unframed_points.tolist()


def concatenate_waveguides(sections, promote_other_ports=True):
    ''' Place given sections into a parent device and connect wg_in_1's to wg_out_1's
        The order matters.

        Args:
            sections (list[Devices]): each having a wg_in_1 and wg_out_1 port
            promote_ports (bool): if True, all ports not starting with 'wg_' will end up in the returned Device

        Returns:
            waveguide (Device): that has an updated wg_length and two 'wg_' ports
    '''
    CAT = Device()
    CAT.info['wg_length'] = 0
    prev_sec = None
    for sec in sections:
        # construction
        this_sec = CAT << sec
        if prev_sec is None:
            try:
                CAT.add_port(port = this_sec.ports['wg_in_1'])
            except KeyError:
                # This might be a terminator
                pass
        else:
            this_sec.connect('wg_in_1', prev_sec.ports['wg_out_1'])
        prev_sec = this_sec
        CAT.info['wg_length'] += sec.info.get('wg_length', 0)
        # There might be other ports if you have an electrooptic device. Promote them.
        if promote_other_ports:
            for port in this_sec.ports.values():
                if not port.name.startswith('wg_'):
                    CAT.add_port(port=port)
    # and the last port
    try:
        CAT.add_port(port = this_sec.ports['wg_out_1'])
    except KeyError:
        # This might be a terminator
        pass
    return CAT.flatten()


class WG_XS_Component(object):
    ''' A contiguous element of a cross section that has one width, offset, and layer

        Has property access to min, max, inner, and outer, which are complements to width/offset.
        Changing min changes the width. If instead you want to move without affecting width::

            wgxcomp.offset += target_min - wgxcomp.min

        Inner and outer behavior is determined by the sign of the center offset,
        so don't change that sign, or your frame of reference will get complicated.
        Positive values correspond to away from centerline
    '''
    def __init__(self, width=1, offset=0, layer=None):
        self.width = width
        self.offset = offset
        self.layer = layer
        if self.layer is None:
            raise ValueError('When initializing a WG_XS_Component, you must specify one Layer')

    def copy(self):
        return deepcopy(self)

    def __repr__(self):
        ''' Human readable '''
        core_attributes = ['width', 'offset', 'layer']
        attrstrs = ('{k}={my_attr}'.format(k=k, my_attr=getattr(self,k)) for k in core_attributes)
        fullstr = ',\n'.join(attrstrs)
        return '{name}(\n{fullstr}\n)'.format(name = type(self).__name__, fullstr = fullstr)

    def __eq__(self, other):
        ''' Layers are considered equal if they're name, layer, and datatype are the same '''
        if self.width != other.width:
            return False
        if self.offset != other.offset:
            return False
        for layer_prop in ['name', 'gds_layer', 'gds_datatype']:
            if getattr(self.layer, layer_prop) != getattr(other.layer, layer_prop):
                return False
        return True


    ## Component utilities ##

    @property
    def min(self):
        return self.offset - self.width / 2

    @property
    def max(self):
        return self.offset + self.width / 2

    @min.setter
    def min(self, new_min):
        ''' The setter changes the width/offset but not the max '''
        new_max = self.max
        self.width = np.abs(new_max - new_min)
        self.offset = (new_max + new_min) / 2

    @max.setter
    def max(self, new_max):
        ''' The setter changes the width/offset but not the min '''
        new_min = self.min
        self.width = np.abs(new_max - new_min)
        self.offset = (new_max + new_min) / 2

    def _is_offset_positive(self):
        ''' Based on the sign of offset. Used for inner/outer '''
        if self.offset > 0:
            return True
        elif self.offset < 0:
            return False
        else:
            # raise ValueError('WG_XS_Component with offset==0 have no inner and outer')
            print('Warning inner/outer requested with offset==0')
            return False

    # Inner and outer only apply to offset components
    @property
    def inner(self):
        return self.min if self._is_offset_positive() else -self.max

    @property
    def outer(self):
        return self.max if self._is_offset_positive() else -self.min

    @inner.setter
    def inner(self, new_inner):
        if self._is_offset_positive():
            self.min = new_inner
        else:
            self.max = -new_inner

    @outer.setter
    def outer(self, new_outer):
        if self._is_offset_positive():
            self.max = new_outer
        else:
            self.min = -new_outer
