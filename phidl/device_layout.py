#==============================================================================
# Major TODO
#==============================================================================
# Replace move command with _input2coordinate

#==============================================================================
# Minor TODO
#==============================================================================
# geometry: add packer, basic_wire
# fix remove -- allow removal of labels and cellarrays
# Let both References/Arrays/Polygons to be assigned as D['waveguide'] = D << WG
# Offset improvement for >100,000 polygons:
#   Get list of polygons, np.vstack together, sort, find where x & y medians,
#   _chop all polygons along median (so half of points on one side, half on the
#   other), perform offset on each quadrant of polygons, chop again to cut off
#   incorrect offsets (at value `distance`), stitch together

#==============================================================================
# Imports
#==============================================================================

from __future__ import division # Otherwise integer division e.g.  20 / 7 = 2
from __future__ import print_function # Use print('hello') instead of print 'hello'
from __future__ import absolute_import

import gdspy
from copy import deepcopy
import numpy as np
from numpy import sqrt, mod, pi, sin, cos
from numpy.linalg import norm
import webcolors
import warnings
import hashlib


__version__ = '1.1.0'



#==============================================================================
# Useful transformation functions
#==============================================================================

def _rotate_points(points, angle = 45, center = (0,0)):
    """ Rotates points around a centerpoint defined by ``center``.  ``points`` may be
    input as either single points [1,2] or array-like[N][2], and will return in kind
    """
    if angle == 0:
         return points
    angle = angle*pi/180
    ca = cos(angle)
    sa = sin(angle)
    sa = np.array((-sa, sa))
    c0 = np.array(center)
    if np.asarray(points).ndim == 2:
        return (points - c0) * ca + (points - c0)[:,::-1] * sa + c0
    if np.asarray(points).ndim == 1:
        return (points - c0) * ca + (points - c0)[::-1] * sa + c0

def _reflect_points(points, p1 = (0,0), p2 = (1,0)):
    """ Reflects points across the line formed by p1 and p2.  ``points`` may be
    input as either single points [1,2] or array-like[N][2], and will return in kind
    """
    # From http://math.stackexchange.com/questions/11515/point-reflection-across-a-line
    points = np.array(points); p1 = np.array(p1); p2 = np.array(p2);
    if np.asarray(points).ndim == 1:
        return 2*(p1 + (p2-p1)*np.dot((p2-p1),(points-p1))/norm(p2-p1)**2) - points
    if np.asarray(points).ndim == 2:
        return np.array([2*(p1 + (p2-p1)*np.dot((p2-p1),(p-p1))/norm(p2-p1)**2) - p for p in points])

def _is_iterable(items):
    return isinstance(items, (list, tuple, set, np.ndarray))

def _parse_coordinate(c, ports = {}):
    """ Translates various inputs (lists, tuples, Ports, ) to an (x,y) coordinate """
    if isinstance(c, Port):
        return c.midpoint
    elif np.array(c).size == 2:
        return c
    elif c in ports: 
        return ports[c].midpoint
    else:
        return None




def reset():
    Layer.layer_dict = {}
    Device._next_uid = 0



class LayerSet(object):

    def __init__(self):
        self._layers = {}

    def add_layer(self, name = 'unnamed', gds_layer = 0, gds_datatype = 0,
                 description = None, color = None, inverted = False,
                  alpha = 0.6, dither = None):
        new_layer = Layer(gds_layer = gds_layer, gds_datatype = gds_datatype, name = name,
                 description = description, inverted = inverted,
                 color = color, alpha = alpha, dither = dither)
        if name in self._layers:
            raise ValueError('[PHIDL] LayerSet: Tried to add layer named "%s", but a layer'
                ' with that name already exists in this LayerSet' % (name))
        else:
            self._layers[name] = new_layer

    def __getitem__(self, val):
        """ If you have a LayerSet `ls`, allows access to the layer names like ls['gold2'] """
        try:
            return self._layers[val]
        except:
            raise ValueError('[PHIDL] LayerSet: Tried to access layer named "%s"'
                ' which does not exist' % (val))


    def __repr__(self):
        return ('LayerSet (%s layers total)' % (len(self._layers)))



class Layer(object):
    layer_dict = {}

    def __init__(self, gds_layer = 0, gds_datatype = 0, name = 'unnamed',
                 description = None, inverted = False,
                 color = None, alpha = 0.6, dither = None):
        if isinstance(gds_layer, Layer):
            l = gds_layer # We were actually passed Layer(mylayer), make a copy
            gds_datatype = l.gds_datatype
            name = l.name
            description = l.description
            alpha = l.alpha
            dither = l.dither
            inverted = l.inverted
            gds_layer = l.gds_layer


        self.gds_layer = gds_layer
        self.gds_datatype = gds_datatype
        self.name = name
        self.description = description
        self.inverted = inverted
        self.alpha = alpha
        self.dither = dither

        try:
            if color is None: # not specified
                self.color = None
            elif np.size(color) == 3: # in format (0.5, 0.5, 0.5)
                self.color = webcolors.rgb_to_hex(np.array( np.array(color)*255, dtype = int))
            elif color[0] == '#': # in format #1d2e3f
                self.color = webcolors.hex_to_rgb(color)
                self.color = webcolors.rgb_to_hex(self.color)
            else: # in named format 'gold'
                self.color = webcolors.name_to_hex(color)
        except:
            raise ValueError("""[PHIDL] Layer() color must be specified as a
            0-1 RGB triplet, (e.g. [0.5, 0.1, 0.9]), an HTML hex  color
            (e.g. #a31df4), or a CSS3 color name (e.g. 'gold' or
            see http://www.w3schools.com/colors/colors_names.asp )
            """)

        Layer.layer_dict[(gds_layer, gds_datatype)] = self

    def __repr__(self):
        return ('Layer (name %s, GDS layer %s, GDS datatype %s, description %s, color %s)' % \
                (self.name, self.gds_layer, self.gds_datatype, self.description, self.color))


def _parse_layer(layer):
    """ Check if the variable layer is a Layer object, a 2-element list like
    [0,1] representing layer=0 and datatype=1, or just a layer number """
    if isinstance(layer, Layer):
        gds_layer, gds_datatype = layer.gds_layer, layer.gds_datatype
    elif np.shape(layer) == (2,): # In form [3,0]
        gds_layer, gds_datatype = layer[0], layer[1]
    elif np.shape(layer) == (1,): # In form [3]
        gds_layer, gds_datatype = layer[0], 0
    elif layer is None:
        gds_layer, gds_datatype = 0, 0
    elif isinstance(layer, (int, float)):
        gds_layer, gds_datatype = layer, 0
    else:
        raise ValueError("""[PHIDL] _parse_layer() was passed something
            that could not be interpreted as a layer: layer = %s""" % layer)
    return (gds_layer, gds_datatype)



class _GeometryHelper(object):
    """ This is a helper class. It can be added to any other class which has
    the functions move() and the property ``bbox`` (as in self.bbox).  It uses
    that function+property to enable you to do things like check what the center
    of the bounding box is (self.center), and also to do things like move the
    bounding box such that its maximum x value is 5.2 (self.xmax = 5.2) """

    @property
    def center(self):
        return np.sum(self.bbox,0)/2

    @center.setter
    def center(self, destination):
        self.move(destination = destination, origin = self.center)

    @property
    def x(self):
        return np.sum(self.bbox,0)[0]/2

    @x.setter
    def x(self, destination):
        destination = (destination, self.center[1])
        self.move(destination = destination, origin = self.center, axis = 'x')

    @property
    def y(self):
        return np.sum(self.bbox,0)[1]/2

    @y.setter
    def y(self, destination):
        destination = ( self.center[0], destination)
        self.move(destination = destination, origin = self.center, axis = 'y')

    @property
    def xmax(self):
        return self.bbox[1][0]

    @xmax.setter
    def xmax(self, destination):
        self.move(destination = (destination, 0), origin = self.bbox[1], axis = 'x')

    @property
    def ymax(self):
        return self.bbox[1][1]

    @ymax.setter
    def ymax(self, destination):
        self.move(destination = (0, destination), origin = self.bbox[1], axis = 'y')

    @property
    def xmin(self):
        return self.bbox[0][0]

    @xmin.setter
    def xmin(self, destination):
        self.move(destination = (destination, 0), origin = self.bbox[0], axis = 'x')

    @property
    def ymin(self):
        return self.bbox[0][1]

    @ymin.setter
    def ymin(self, destination):
        self.move(destination = (0, destination), origin = self.bbox[0], axis = 'y')

    @property
    def size(self):
        bbox = self.bbox
        return bbox[1] - bbox[0]

    @property
    def xsize(self):
        bbox = self.bbox
        return bbox[1][0] - bbox[0][0]

    @property
    def ysize(self):
        bbox = self.bbox
        return bbox[1][1] - bbox[0][1]

    def movex(self, origin = 0, destination = None):
        if destination is None:
            destination = origin
            origin = 0
        self.move(origin = (origin,0), destination = (destination,0))
        return self

    def movey(self, origin = 0, destination = None):
        if destination is None:
            destination = origin
            origin = 0
        self.move(origin = (0,origin), destination = (0,destination))
        return self



class Port(object):
    _next_uid = 0

    def __init__(self, name = None, midpoint = (0,0), width = 1, orientation = 0, parent = None):
        self.name = name
        self.midpoint = np.array(midpoint, dtype = 'float64')
        self.width = width
        self.orientation = mod(orientation,360)
        self.parent = parent
        self.info = {}
        self.uid = Port._next_uid
        if self.width < 0: raise ValueError('[PHIDL] Port creation error: width must be >=0')
        Port._next_uid += 1

    def __repr__(self):
        return ('Port (name %s, midpoint %s, width %s, orientation %s)' % \
                (self.name, self.midpoint, self.width, self.orientation))

    @property
    def endpoints(self):
        dxdy = np.array([
            self.width/2*np.cos((self.orientation - 90)*pi/180),
            self.width/2*np.sin((self.orientation - 90)*pi/180)
            ])
        left_point = self.midpoint - dxdy
        right_point = self.midpoint + dxdy
        return np.array([left_point, right_point])

    @endpoints.setter
    def endpoints(self, points):
        p1, p2 = np.array(points[0]), np.array(points[1])
        self.midpoint = (p1+p2)/2
        dx, dy = p2-p1
        self.orientation = np.arctan2(dx,dy)*180/pi
        self.width = sqrt(dx**2 + dy**2)

    @property
    def normal(self):
        dx = np.cos((self.orientation)*pi/180)
        dy = np.sin((self.orientation)*pi/180)
        return np.array([self.midpoint, self.midpoint + np.array([dx,dy])])

    @property
    def x(self):
        return self.midpoint[0]

    @property
    def y(self):
        return self.midpoint[1]

    @property
    def center(self):
        return self.midpoint
    
    # Use this function instead of copy() (which will not create a new numpy array
    # for self.midpoint) or deepcopy() (which will also deepcopy the self.parent
    # DeviceReference recursively, causing performance issues)
    def _copy(self, new_uid = True):
        new_port = Port(name = self.name, midpoint = self.midpoint,
            width = self.width, orientation = self.orientation,
            parent = self.parent)
        new_port.info = deepcopy(self.info)
        if new_uid == False:
            new_port.uid = self.uid
            Port._next_uid -= 1
        return new_port

    def rotate(self, angle = 45, center = None):
        self.orientation = mod(self.orientation + angle, 360)
        if center is None:
            center = self.midpoint
        self.midpoint = _rotate_points(self.midpoint, angle = angle, center = center)
        return self


class Polygon(gdspy.Polygon, _GeometryHelper):

    def __init__(self, points, gds_layer, gds_datatype, parent):
        self.parent = parent
        super(Polygon, self).__init__(points = points, layer=gds_layer,
            datatype=gds_datatype)


    @property
    def bbox(self):
        return self.get_bounding_box()

    def rotate(self, angle = 45, center = (0,0)):
        super(Polygon, self).rotate(angle = angle*pi/180, center = center)
        if self.parent is not None:
            self.parent._bb_valid = False
        return self

    def move(self, origin = (0,0), destination = None, axis = None):
        """ Moves elements of the Device from the origin point to the destination.  Both
         origin and destination can be 1x2 array-like, Port, or a key
         corresponding to one of the Ports in this device """

        # If only one set of coordinates is defined, make sure it's used to move things
        if destination is None:
            destination = origin
            origin = [0,0]

        if isinstance(origin, Port):            o = origin.midpoint
        elif np.array(origin).size == 2:    o = origin
        elif origin in self.ports:    o = self.ports[origin].midpoint
        else: raise ValueError('[PHIDL] [DeviceReference.move()] ``origin`` not array-like, a port, or port name')

        if isinstance(destination, Port):           d = destination.midpoint
        elif np.array(destination).size == 2:        d = destination
        elif destination in self.ports:   d = self.ports[destination].midpoint
        else: raise ValueError('[PHIDL] [DeviceReference.move()] ``destination`` not array-like, a port, or port name')

        if axis == 'x': d = (d[0], o[1])
        if axis == 'y': d = (o[0], d[1])

        dx,dy = np.array(d) - o

        super(Polygon, self).translate(dx, dy)
        if self.parent is not None:
            self.parent._bb_valid = False
        return self


    def reflect(self, p1 = (0,1), p2 = (0,0)):
        for n, points in enumerate(self.polygons):
            self.polygons[n] = _reflect_points(points, p1, p2)
        if self.parent is not None:
            self.parent._bb_valid = False
        return self



def make_device(fun, config = None, **kwargs):
    config_dict = {}
    if type(config) is dict:
        config_dict = dict(config)
    elif config is None:
        pass
    else:
        raise TypeError("""[PHIDL] When creating Device() from a function, the
        second argument should be a ``config`` argument which is a
        dictionary containing arguments for the function.
        e.g. make_device(ellipse, config = my_config_dict) """)
    config_dict.update(**kwargs)
    D = fun(**config_dict)
    if not isinstance(D, Device):
        raise ValueError("""[PHIDL] Device() was passed a function, but that
        function does not produce a Device.""")
    return D



class Device(gdspy.Cell, _GeometryHelper):

    _next_uid = 0

    def __init__(self, *args, **kwargs):
        if len(args) > 0:
            if callable(args[0]):
                raise ValueError('[PHIDL] You can no longer create geometry '
                    'by calling Device(device_making_function), please use '
                    'make_device(device_making_function) instead')


        # Allow name to be set like Device('arc') or Device(name = 'arc')
        if 'name' in kwargs:                          _internal_name = kwargs['name']
        elif (len(args) == 1) and (len(kwargs) == 0): _internal_name = args[0]
        else:                                         _internal_name = 'Unnamed'

        # Make a new blank device
        self.ports = {}
        self.info = {}
        self.aliases = {}
        # self.a = self.aliases
        # self.p = self.ports
        self.uid = Device._next_uid
        self._internal_name = _internal_name
        gds_name = '%s%06d' % (self._internal_name[:20], self.uid) # Write name e.g. 'Unnamed000005'
        super(Device, self).__init__(name = gds_name, exclude_from_current=True)
        Device._next_uid += 1


    def __getitem__(self, key):
        """ If you have a Device D, allows access to aliases you made like D['arc2'] """
        try:
            return self.aliases[key]
        except:
            raise ValueError('[PHIDL] Tried to access alias "%s" in Device "%s",  '
                'which does not exist' % (key, self.name))

    def __repr__(self):
        return ('Device (name "%s" (uid %s),  ports %s, aliases %s, %s polygons, %s references)' % \
                (self._internal_name, self.uid, list(self.ports.keys()), list(self.aliases.keys()),
                len(self.polygons), len(self.references)))


    def __str__(self):
        return self.__repr__()

    def __lshift__(self, element):
        return self.add_ref(element)

    def __setitem__(self, key, element):
        """ Allow adding polygons and cell references like D['arc3'] = pg.arc() """
        if isinstance(element, DeviceReference):
            self.aliases[key] = element
        else:
            raise ValueError('[PHIDL] Tried to assign alias "%s" in Device "%s",  '
                'but failed because the item was not a DeviceReference' % (key, self.name))

    @property
    def layers(self):
        return self.get_layers()

    # @property
    # def references(self):
    #     return [e for e in self.elements if isinstance(e, DeviceReference)]

    # @property
    # def polygons(self):
    #     return [e for e in self.elements if isinstance(e, gdspy.PolygonSet)]



    @property
    def bbox(self):
        bbox = self.get_bounding_box()
        if bbox is None:  bbox = ((0,0),(0,0))
        return np.array(bbox)

    def add_ref(self, device, alias = None):
        """ Takes a Device and adds it as a DeviceReference to the current
        Device.  """
        if _is_iterable(device):
            return [self.add_ref(E) for E in device]
        if not isinstance(device, Device):
            raise TypeError("""[PHIDL] add_ref() was passed something that
            was not a Device object. """)
        d = DeviceReference(device)   # Create a DeviceReference (CellReference)
        self.add(d)             # Add DeviceReference (CellReference) to Device (Cell)

        if alias is not None:
            self.aliases[alias] = d
        return d                # Return the DeviceReference (CellReference)


    def add_polygon(self, points, layer = None):
        # Check if input a list of polygons by seeing if it's 3 levels deep
        try:
            points[0][0][0] # Try to access first x point
            return [self.add_polygon(p, layer) for p in points]
        except: pass # Verified points is not a list of polygons, continue on

        if isinstance(points, gdspy.PolygonSet):
            if layer is None:   layers = zip(points.layers, points.datatypes)
            else:   layers = [layer]*len(points.polygons)
            return [self.add_polygon(p, layer) for p, layer in zip(points.polygons, layers)]

        # Check if layer is actually a list of Layer objects
        try:
            if isinstance(layer, LayerSet):
                return [self.add_polygon(points, l) for l in layer._layers.values()]
            elif isinstance(layer, set):
                return [self.add_polygon(points, l) for l in layer]
            elif all([isinstance(l, (Layer)) for l in layer]):
                return [self.add_polygon(points, l) for l in layer]
            elif len(layer) > 2: # Someone wrote e.g. layer = [1,4,5]
                raise ValueError(""" [PHIDL] When using add_polygon() with
                    multiple layers, each element in your `layer` argument
                    list must be of type Layer(), e.g.:
                    `layer = [Layer(1,0), my_layer, Layer(4)]""")
        except: pass

        # If in the form [[1,3,5],[2,4,6]]
        if len(points[0]) > 2:
            # Convert to form [[1,2],[3,4],[5,6]]
            points = np.column_stack((points))

        gds_layer, gds_datatype = _parse_layer(layer)
        polygon = Polygon(points = points, gds_layer = gds_layer,
            gds_datatype = gds_datatype, parent = self)
        self.add(polygon)
        return polygon


    def add_array(self, device, columns = 2, rows = 2, spacing = (100,100), alias = None):
        if not isinstance(device, Device):
            raise TypeError("""[PHIDL] add_array() was passed something that
            was not a Device object. """)
        a = CellArray(device = device, columns = columns, rows = rows, spacing = spacing)
        self.add(a)             # Add DeviceReference (CellReference) to Device (Cell)
        if alias is not None:
            self.aliases[alias] = a
        return a                # Return the CellArray


    def add_port(self, name = None, midpoint = (0,0), width = 1, orientation = 45, port = None):
        """ Can be called to copy an existing port like add_port(port = existing_port) or
        to create a new port add_port(myname, mymidpoint, mywidth, myorientation).
        Can also be called to copy an existing port with a new name like add_port(port = existing_port, name = new_name)"""
        if port is not None:
            if not isinstance(port, Port):
                raise ValueError('[PHIDL] add_port() error: Argument `port` must be a Port for copying')
            p = port._copy(new_uid = True)
            p.parent = self
        elif isinstance(name, Port):
            p = name._copy(new_uid = True)
            p.parent = self
            name = p.name
        else:
            p = Port(name = name, midpoint = midpoint, width = width,
                orientation = orientation, parent = self)
        if name is not None: p.name = name
        if p.name in self.ports:
            raise ValueError('[DEVICE] add_port() error: Port name "%s" already exists in this Device (name "%s", uid %s)' % (p.name, self._internal_name, self.uid))
        self.ports[p.name] = p
        return p


    def label(self, text = 'hello', position = (0,0), magnification = None, rotation = None, layer = 255):
        if len(text) >= 1023:
            raise ValueError('[DEVICE] label() error: Text too long (limit 1024 chars)')
        gds_layer, gds_datatype = _parse_layer(layer)

        if type(text) is not str: text = str(text)
        self.add(gdspy.Label(text = text, position = position, anchor = 'o', magnification = magnification, rotation = rotation,
                                 layer = gds_layer, texttype = gds_datatype))
        return self

    def annotate(self, *args, **kwargs):
        warnings.warn('[PHIDL] WARNING: annotate() has been deprecated, please replace with label()')
        return self.label(*args, **kwargs)


    def write_gds(self, filename, unit = 1e-6, precision = 1e-9,
                  auto_rename = True, max_cellname_length = 28):
        if filename[-4:] != '.gds':  filename += '.gds'
        tempname = self.name
        referenced_cells = list(self.get_dependencies(recursive=True))
        all_cells = [self] + referenced_cells

        # Autofix names so there are no duplicates
        if auto_rename == True:
            all_cells_sorted = sorted(all_cells, key=lambda x: x.uid)
            used_names = {'toplevel':1}
            for c in all_cells_sorted:
                if max_cellname_length is not None:
                    new_name = c._internal_name[:max_cellname_length]
                else:
                    new_name = c._internal_name
                if new_name not in used_names:
                    used_names[new_name] = 1
                    c.name = new_name
                else:
                    c.name = new_name + ('%0.3i' % used_names[new_name])
                    used_names[new_name] += 1
            self.name = 'toplevel'
        gdspy.write_gds(filename, cells=all_cells, name='library',
                        unit=unit, precision=precision)
        self.name = tempname
        return filename


    def remap_layers(self, layermap = {}, include_labels = True):
        layermap = {_parse_layer(k):_parse_layer(v) for k,v in layermap.items()}

        all_D = list(self.get_dependencies(True))
        all_D += [self]
        for D in all_D:
            for p in D.polygons:
                for n, layer in enumerate(p.layers):
                    original_layer = (p.layers[n], p.datatypes[n])
                    original_layer = _parse_layer(original_layer)
                    if original_layer in layermap.keys():
                        new_layer = layermap[original_layer]
                        p.layers[n] = new_layer[0]
                        p.datatypes[n] = new_layer[1]
            if include_labels == True:
                for l in D.labels:
                    original_layer = (l.layer, l.texttype)
                    original_layer = _parse_layer(original_layer)
                    if original_layer in layermap.keys():
                        new_layer = layermap[original_layer]
                        l.layer = new_layer[0]
                        l.texttype = new_layer[1]
        return self

    def remove_layers(self, layers = (), include_labels = True, invert_selection = False):
        layers = [_parse_layer(l) for l in layers]
        all_D = list(self.get_dependencies(True))
        all_D += [self]
        for D in all_D:
            for polygonset in D.polygons:
                polygon_layers = zip(polygonset.layers, polygonset.datatypes)
                polygons_to_keep = [(pl in layers) for pl in polygon_layers]
                if invert_selection == False: polygons_to_keep = [(not p) for p in polygons_to_keep]
                polygonset.polygons =  [p for p,keep in zip(polygonset.polygons,  polygons_to_keep) if keep]
                polygonset.layers =    [p for p,keep in zip(polygonset.layers,    polygons_to_keep) if keep]
                polygonset.datatypes = [p for p,keep in zip(polygonset.datatypes, polygons_to_keep) if keep]

            if include_labels == True:
                new_labels = []
                for l in D.labels:
                    original_layer = (l.layer, l.texttype)
                    original_layer = _parse_layer(original_layer)
                    if invert_selection: keep_layer = (original_layer in layers)
                    else:                keep_layer = (original_layer not in layers)
                    if keep_layer:
                        new_labels += [l]
                D.labels = new_labels
        return self


    def distribute(self, elements = 'all', direction = 'x', spacing = 100, separation = True):
        if direction not in (['+x','-x','x','+y','-y','y']):
            raise ValueError("[PHIDL] distribute(): 'direction' argument must be one of '+x','-x','x','+y','-y','y'")

        if elements == 'all': elements = (self.polygons + self.references)

        if direction == 'x': direction = '+x'
        elif direction == 'y': direction = '+y'

        sizes = [e.size for e in elements]
        xy = elements[0].center
        for n, e in enumerate(elements[:-1]):
            e.center = xy
            if direction == '+x':  xy[0] += spacing + separation*(sizes[n] + sizes[n+1])[0]/2
            if direction == '-x':  xy[0] -= spacing + separation*(sizes[n] + sizes[n+1])[0]/2
            if direction == '+y':  xy[1] += spacing + separation*(sizes[n] + sizes[n+1])[1]/2
            if direction == '-y':  xy[1] -= spacing + separation*(sizes[n] + sizes[n+1])[1]/2
        elements[-1].center = xy
        return self


    def align(self, elements = 'all', alignment = 'ymax'):
        if elements == 'all': elements = (self.polygons + self.references)
        if alignment not in (['x','y','xmin', 'xmax', 'ymin','ymax']):
            raise ValueError("[PHIDL] align(): 'alignment' argument must be one of 'x','y','xmin', 'xmax', 'ymin','ymax'")
        if elements is None:
            elements = (self.polygons + self.references)
        value = self.__getattribute__(alignment)
        for e in elements:
            e.__setattr__(alignment, value)
        return self


    def flatten(self,  single_layer = None):
        if single_layer is None:
            super(Device, self).flatten(single_layer=None, single_datatype=None, single_texttype=None)
        else:
            gds_layer, gds_datatype = _parse_layer(single_layer)
            super(Device, self).flatten(single_layer = gds_layer, single_datatype = gds_datatype, single_texttype=gds_datatype)

        temp_polygons = list(self.polygons)
        self.references = []
        self.polygons = []
        [self.add_polygon(poly) for poly in temp_polygons]
        return self


    def absorb(self, reference):
        """ Flattens and absorbs polygons from an underlying
        DeviceReference into the Device, destroying the reference
        in the process but keeping the polygon geometry """
        if reference not in self.references:
            raise ValueError("""[PHIDL] Device.absorb() failed -
                the reference it was asked to absorb does not
                exist in this Device. """)
        ref_polygons = reference.get_polygons(by_spec = True)
        for (layer, polys) in ref_polygons.items():
            [self.add_polygon(points = p, layer = layer) for p in polys]
        self.remove(reference)
        return self


    def get_ports(self, depth = None):
        """ Returns copies of all the ports of the Device, rotated
        and translated so that they're in their top-level position.
        The Ports returned are copies of the originals, but each copy
        has the same ``uid'' as the original so that they can be
        traced back to the original if needed"""
        port_list = [p._copy(new_uid = False) for p in self.ports.values()]

        if depth is None or depth > 0:
            for r in self.references:
                if depth is None: new_depth = None
                else:             new_depth = depth - 1
                ref_ports = r.parent.get_ports(depth=new_depth)

                # Transform ports that came from a reference
                ref_ports_transformed = []
                for rp in ref_ports:
                    new_port = rp._copy(new_uid = False)
                    new_midpoint, new_orientation = r._transform_port(rp.midpoint, \
                    rp.orientation, r.origin, r.rotation, r.x_reflection)
                    new_port.midpoint = new_midpoint
                    new_port.new_orientation = new_orientation
                    ref_ports_transformed.append(new_port)
                port_list += ref_ports_transformed

        return port_list


    def remove(self, items):
        if not _is_iterable(items):  items = [items]
        for item in items:
            if isinstance(item, Port):
                try:
                    self.ports = { k:v for k, v in self.ports.items() if v != item}
                except:
                    raise ValueError("""[PHIDL] Device.remove() cannot find the Port
                                     it was asked to remove in the Device: "%s".""" % (item))
            else:
                try:
                    if isinstance(item, gdspy.PolygonSet):
                        self.polygons.remove(item)
                    if isinstance(item, gdspy.CellReference):
                        self.references.remove(item)
                    if isinstance(item, gdspy.Label):
                        self.labels.remove(item)
                    self.aliases = { k:v for k, v in self.aliases.items() if v != item}
                except:
                    raise ValueError("""[PHIDL] Device.remove() cannot find the item
                                     it was asked to remove in the Device: "%s".""" % (item))

        self._bb_valid = False
        return self


    def rotate(self, angle = 45, center = (0,0)):
        if angle == 0: return self
        for e in self.polygons:
            e.rotate(angle = angle, center = center)
        for e in self.references:
            e.rotate(angle, center)
        for p in self.ports.values():
            p.midpoint = _rotate_points(p.midpoint, angle, center)
            p.orientation = mod(p.orientation + angle, 360)
        self._bb_valid = False
        return self


    def move(self, origin = (0,0), destination = None, axis = None):
        """ Moves elements of the Device from the origin point to the destination.  Both
         origin and destination can be 1x2 array-like, Port, or a key
         corresponding to one of the Ports in this device """

        # If only one set of coordinates is defined, make sure it's used to move things
        if destination is None:
            destination = origin
            origin = [0,0]

        if isinstance(origin, Port):            o = origin.midpoint
        elif np.array(origin).size == 2:    o = origin
        elif origin in self.ports:    o = self.ports[origin].midpoint
        else: raise ValueError('[PHIDL] DeviceReference.move() ``origin`` not array-like, a port, or port name')

        if isinstance(destination, Port):           d = destination.midpoint
        elif np.array(destination).size == 2:        d = destination
        elif destination in self.ports:   d = self.ports[destination].midpoint
        else: raise ValueError('[PHIDL] DeviceReference.move() ``destination`` not array-like, a port, or port name')

        if axis == 'x': d = (d[0], o[1])
        if axis == 'y': d = (o[0], d[1])

        dx,dy = np.array(d) - o

        # Move geometries
        for e in self.polygons:
            e.translate(dx,dy)
        for e in self.references:
            e.move(destination = d, origin = o)
        for p in self.ports.values():
            p.midpoint = np.array(p.midpoint) + np.array(d) - np.array(o)

        # Move labels
        for l in self.labels:
            l.translate(dx,dy)

        self._bb_valid = False
        return self

    def reflect(self, p1 = (0,1), p2 = (0,0)):
        for e in (self.polygons+self.references):
            e.reflect(p1, p2)
        for p in self.ports.values():
            p.midpoint = _reflect_points(p.midpoint, p1, p2)
            phi = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])*180/pi
            p.orientation = 2*phi - p.orientation
        self._bb_valid = False
        return self


    def hash_geometry(self, precision = 1e-4):
        """
        Algorithm:
        hash(
            hash(First layer information: [layer1, datatype1]),
            hash(Polygon 1 on layer 1 points: [(x1,y1),(x2,y2),(x3,y3)] ),
            hash(Polygon 2 on layer 1 points: [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] ),
            hash(Polygon 3 on layer 1 points: [(x1,y1),(x2,y2),(x3,y3)] ),
            hash(Second layer information: [layer2, datatype2]),
            hash(Polygon 1 on layer 2 points: [(x1,y1),(x2,y2),(x3,y3),(x4,y4)] ),
            hash(Polygon 2 on layer 2 points: [(x1,y1),(x2,y2),(x3,y3)] ),
        )
        ...
        Note: For each layer, each polygon is individually hashed and then
              the polygon hashes are sorted, to ensure the hash stays constant
              regardless of the ordering the polygons.  Similarly, the layers
              are sorted by (layer, datatype)
        """
        polygons_by_spec = self.get_polygons(by_spec = True)
        layers = np.array(list(polygons_by_spec.keys()))
        sorted_layers = layers[np.lexsort((layers[:,0], layers[:,1]))]

        # A random offset which fixes common rounding errors intrinsic
        # to floating point math. Example: with a precision of 0.1, the
        # floating points 7.049999 and 7.050001 round to different values
        # (7.0 and 7.1), but offset values (7.220485 and 7.220487) don't
        magic_offset = .17048614

        final_hash = hashlib.sha1()
        for layer in sorted_layers:
            layer_hash = hashlib.sha1(layer.astype(np.int64)).digest()
            polygons = polygons_by_spec[tuple(layer)]
            polygons = [((p/precision) + magic_offset).astype(np.int64) for p in polygons]
            polygon_hashes = np.sort([hashlib.sha1(p).digest() for p in polygons])
            final_hash.update(layer_hash)
            for ph in polygon_hashes:
                final_hash.update(ph)

        return final_hash.hexdigest()



class DeviceReference(gdspy.CellReference, _GeometryHelper):
    def __init__(self, device, origin=(0, 0), rotation=0, magnification=None, x_reflection=False):
        super(DeviceReference, self).__init__(
                 ref_cell = device,
                 origin=origin,
                 rotation=rotation,
                 magnification=magnification,
                 x_reflection=x_reflection,
                 ignore_missing=False)
        self.parent = device
        # The ports of a DeviceReference have their own unique id (uid),
        # since two DeviceReferences of the same parent Device can be
        # in different locations and thus do not represent the same port
        self._local_ports = {name:port._copy(new_uid = True) for name, port in device.ports.items()}


    def __repr__(self):
        return ('DeviceReference (parent Device "%s", ports %s, origin %s, rotation %s, x_reflection %s)' % \
                (self.parent.name, list(self.ports.keys()), self.origin, self.rotation, self.x_reflection))


    def __str__(self):
        return self.__repr__()


    def __getitem__(self, val):
        """ This allows you to access an alias from the reference's parent, and receive
        a copy of the reference which is correctly rotated and translated"""
        try:
            alias_device = self.parent[val]
        except:
            raise ValueError('[PHIDL] Tried to access alias "%s" from parent '
                'Device "%s", which does not exist' % (val, self.parent.name))
        new_reference = DeviceReference(alias_device.parent, origin=alias_device.origin, rotation=alias_device.rotation, magnification=alias_device.magnification, x_reflection=alias_device.x_reflection)

        if self.x_reflection:
            new_reference.reflect((1,0))
        if self.rotation is not None:
            new_reference.rotate(self.rotation)
        if self.origin is not None:
            new_reference.move(self.origin)

        return new_reference


    @property
    def ports(self):
        """ This property allows you to access myref.ports, and receive a copy
        of the ports dict which is correctly rotated and translated"""
        for name, port in self.parent.ports.items():
            port = self.parent.ports[name]
            new_midpoint, new_orientation = self._transform_port(port.midpoint, \
                port.orientation, self.origin, self.rotation, self.x_reflection)
            if name not in self._local_ports:
                self._local_ports[name] = port._copy(new_uid = True)
            self._local_ports[name].midpoint = new_midpoint
            self._local_ports[name].orientation = mod(new_orientation,360)
            self._local_ports[name].parent = self
        # Remove any ports that no longer exist in the reference's parent
        parent_names = self.parent.ports.keys()
        local_names = list(self._local_ports.keys())
        for name in local_names:
            if name not in parent_names: self._local_ports.pop(name)
        return self._local_ports

    @property
    def info(self):
        return self.parent.info

    @property
    def bbox(self):
        bbox = self.get_bounding_box()
        if bbox is None:  bbox = ((0,0),(0,0))
        return np.array(bbox)



    def _transform_port(self, point, orientation, origin=(0, 0), rotation=None, x_reflection=False):
        # Apply GDS-type transformations to a port (x_ref)
        new_point = np.array(point)
        new_orientation = orientation

        if x_reflection:
            new_point[1] = -new_point[1]
            new_orientation = -orientation
        if rotation is not None:
            new_point = _rotate_points(new_point, angle = rotation, center = [0, 0])
            new_orientation += rotation
        if origin is not None:
            new_point = new_point + np.array(origin)
        new_orientation = mod(new_orientation, 360)

        return new_point, new_orientation

    def move(self, origin = (0,0), destination = None, axis = None):
        """ Moves the DeviceReference from the origin point to the destination.  Both
         origin and destination can be 1x2 array-like, Port, or a key
         corresponding to one of the Ports in this device_ref """

        # If only one set of coordinates is defined, make sure it's used to move things
        if destination is None:
            destination = origin
            origin = (0,0)

        if isinstance(origin, Port):            o = origin.midpoint
        elif np.array(origin).size == 2:    o = origin
        elif origin in self.ports:    o = self.ports[origin].midpoint
        else: raise ValueError('[DeviceReference.move()] ``origin`` not array-like, a port, or port name')

        if isinstance(destination, Port):           d = destination.midpoint
        elif np.array(destination).size == 2:   d = destination
        elif destination in self.ports:   d = self.ports[destination].midpoint
        else: raise ValueError('[DeviceReference.move()] ``destination`` not array-like, a port, or port name')

        # Lock one axis if necessary
        if axis == 'x': d = (d[0], o[1])
        if axis == 'y': d = (o[0], d[1])

        # This needs to be done in two steps otherwise floating point errors can accrue
        dxdy = np.array(d) - np.array(o)
        self.origin = np.array(self.origin) + dxdy
        return self


    def rotate(self, angle = 45, center = (0,0)):
        if angle == 0: return self
        if type(center) is Port:  center = center.midpoint
        self.rotation += angle
        self.origin = _rotate_points(self.origin, angle, center)
        return self


    def reflect(self, p1 = (0,1), p2 = (0,0)):
        if type(p1) is Port:  p1 = p1.midpoint
        if type(p2) is Port:  p2 = p2.midpoint
        p1 = np.array(p1);  p2 = np.array(p2)
        # Translate so reflection axis passes through origin
        self.origin = self.origin - p1

        # Rotate so reflection axis aligns with x-axis
        angle = np.arctan2((p2[1]-p1[1]),(p2[0]-p1[0]))*180/pi
        self.origin = _rotate_points(self.origin, angle = -angle, center = [0,0])
        self.rotation -= angle

        # Reflect across x-axis
        self.x_reflection = not self.x_reflection
        self.origin[1] = -self.origin[1]
        self.rotation = -self.rotation

        # Un-rotate and un-translate
        self.origin = _rotate_points(self.origin, angle = angle, center = [0,0])
        self.rotation += angle
        self.origin = self.origin + p1

        return self


    def connect(self, port, destination, overlap = 0):
        # ``port`` can either be a string with the name or an actual Port
        if port in self.ports: # Then ``port`` is a key for the ports dict
            p = self.ports[port]
        elif type(port) is Port:
            p = port
        else:
            raise ValueError('[PHIDL] connect() did not receive a Port or valid port name' + \
                ' - received (%s), ports available are (%s)' % (port, tuple(self.ports.keys())))
        self.rotate(angle =  180 + destination.orientation - p.orientation, center = p.midpoint)
        self.move(origin = p, destination = destination)
        self.move(-overlap*np.array([cos(destination.orientation*pi/180),
                                     sin(destination.orientation*pi/180)]))
        return self




class CellArray(gdspy.CellArray, _GeometryHelper):
    def __init__(self, device, columns, rows, spacing, origin=(0, 0),
                 rotation=0, magnification=None, x_reflection=False):
        super(CellArray, self).__init__(
            columns = columns,
            rows = rows,
            spacing = spacing,
            ref_cell = device,
            origin=origin,
            rotation=rotation,
            magnification=magnification,
            x_reflection=x_reflection,
            ignore_missing=False)
        self.parent = device

    @property
    def bbox(self):
        bbox = self.get_bounding_box()
        if bbox is None:  bbox = ((0,0),(0,0))
        return np.array(bbox)


    def move(self, origin = (0,0), destination = None, axis = None):
        """ Moves the CellArray from the origin point to the destination.  Both
         origin and destination can be 1x2 array-like, Port, or a key
         corresponding to one of the Ports in this device_ref """

        # If only one set of coordinates is defined, make sure it's used to move things
        if destination is None:
            destination = origin
            origin = (0,0)

        if isinstance(origin, Port):            o = origin.midpoint
        elif np.array(origin).size == 2:    o = origin
        elif origin in self.ports:    o = self.ports[origin].midpoint
        else: raise ValueError('[CellArray.move()] ``origin`` not array-like, a port, or port name')

        if isinstance(destination, Port):           d = destination.midpoint
        elif np.array(destination).size == 2:   d = destination
        elif destination in self.ports:   d = self.ports[destination].midpoint
        else: raise ValueError('[CellArray.move()] ``destination`` not array-like, a port, or port name')

        # Lock one axis if necessary
        if axis == 'x': d = (d[0], o[1])
        if axis == 'y': d = (o[0], d[1])

        # This needs to be done in two steps otherwise floating point errors can accrue
        dxdy = np.array(d) - np.array(o)
        self.origin = np.array(self.origin) + dxdy
        return self


    def rotate(self, angle = 45, center = (0,0)):
        if angle == 0: return self
        if type(center) is Port:  center = center.midpoint
        self.rotation += angle
        self.origin = _rotate_points(self.origin, angle, center)
        return self


    def reflect(self, p1 = (0,1), p2 = (0,0)):
        if type(p1) is Port:  p1 = p1.midpoint
        if type(p2) is Port:  p2 = p2.midpoint
        p1 = np.array(p1);  p2 = np.array(p2)
        # Translate so reflection axis passes through origin
        self.origin = self.origin - p1

        # Rotate so reflection axis aligns with x-axis
        angle = np.arctan2((p2[1]-p1[1]),(p2[0]-p1[0]))*180/pi
        self.origin = _rotate_points(self.origin, angle = -angle, center = [0,0])
        self.rotation -= angle

        # Reflect across x-axis
        self.x_reflection = not self.x_reflection
        self.origin[1] = -self.origin[1]
        self.rotation = -self.rotation

        # Un-rotate and un-translate
        self.origin = _rotate_points(self.origin, angle = angle, center = [0,0])
        self.rotation += angle
        self.origin = self.origin + p1

        return self
