#==============================================================================
# Major TODO
#==============================================================================

# TODO Add support for gdspy.CellArray

#==============================================================================
# Minor TODO
#==============================================================================
# TODO Move quickplot to phidl.quickplotter
# TODO PHIDL Make rotation and magnification _rotation and _magnification so they don't show up

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
import yaml

try:
    from matplotlib import pyplot as plt
except:
    warnings.warn("""PHIDL tried to import matplotlib but it failed. It will'
                     still work but quickplot() may not.  Try using
                     quickplot2() instead (see note in tutorial) """)

__version__ = '0.8.9'



#==============================================================================
# Useful transformation functions
#==============================================================================

def _rotate_points(points, angle = 45, center = (0,0)):
    """ Rotates points around a centerpoint defined by ``center``.  ``points`` may be
    input as either single points [1,2] or array-like[N][2], and will return in kind
    """
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
    def __init__(self, name = None, midpoint = (0,0), width = 1, orientation = 0, parent = None):
        self.name = name
        self.midpoint = np.array(midpoint, dtype = 'float64')
        self.width = width
        self.orientation = mod(orientation,360)
        self.parent = parent
        self.info = {}
        if self.width < 0: raise ValueError('[PHIDL] Port creation error: width must be >=0')
        
    def __repr__(self):
        return ('Port (name %s, midpoint %s, width %s, orientation %s)' % \
                (self.name, self.midpoint, self.width, self.orientation))
       
    @property
    def endpoints(self):
        dx = self.width/2*np.cos((self.orientation - 90)*pi/180)
        dy = self.width/2*np.sin((self.orientation - 90)*pi/180)
        left_point = self.midpoint - np.array([dx,dy])
        right_point = self.midpoint + np.array([dx,dy])
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
        
    # Use this function instead of copy() (which will not create a new numpy array
    # for self.midpoint) or deepcopy() (which will also deepcopy the self.parent
    # DeviceReference recursively, causing performance issues)
    def _copy(self):
        new_port = Port(name = self.name, midpoint = self.midpoint,
            width = self.width, orientation = self.orientation,
            parent = self.parent)
        new_port.info = deepcopy(self.info)
        return new_port


class Polygon(gdspy.Polygon, _GeometryHelper):
    
    def __init__(self, points, gds_layer, gds_datatype, parent):
        self.parent = parent
        super(Polygon, self).__init__(points = points, layer=gds_layer,
            datatype=gds_datatype, verbose=False)


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
    if type(config) is str:
        with open(config) as f:
            config_dict = yaml.load(f) # Load arguments from config file
    elif type(config) is dict:
        config_dict = dict(config)
    elif config is None:
        pass
    else:
        raise TypeError("""[PHIDL] When creating Device() from a function, the
        second argument should be a ``config`` argument which is either a
        filename or a dictionary containing arguments for the function.
        e.g. make_device(ellipse, config = 'myconfig.yaml') """)
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
        return ('Device (name "%s" (uid %s),  ports %s, aliases %s, %s elements, %s references)' % \
                (self._internal_name, self.uid, list(self.ports.keys()), list(self.aliases.keys()),
                len(self.elements), len(self.references)))


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

    @property
    def references(self):
        return [e for e in self.elements if isinstance(e, DeviceReference)]

    @property
    def polygons(self):
        return [e for e in self.elements if isinstance(e, gdspy.PolygonSet)]

    @property
    def meta(self):
        warnings.warn('[PHIDL] WARNING: .meta is being deprecated, please use .info instead')
        return self.info
        
    @property
    def bbox(self):
        bbox = self.get_bounding_box()
        if bbox is None:  bbox = ((0,0),(0,0))
        return np.array(bbox)

    def add_ref(self, D, alias = None):
        """ Takes a Device and adds it as a DeviceReference to the current
        Device.  """
        if type(D) in (list, tuple):
            return [self.add_ref(E) for E in D]
        if not isinstance(D, Device):
            raise TypeError("""[PHIDL] add_ref() was passed something that
            was not a Device object. """)
        d = DeviceReference(D)   # Create a DeviceReference (CellReference)
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
        
        
    def add_port(self, name = None, midpoint = (0,0), width = 1, orientation = 45, port = None):
        """ Can be called to copy an existing port like add_port(port = existing_port) or
        to create a new port add_port(myname, mymidpoint, mywidth, myorientation).
        Can also be called to copy an existing port with a new name like add_port(port = existing_port, name = new_name)"""
        if port is not None:
            if not isinstance(port, Port):
                raise ValueError('[PHIDL] add_port() error: Argument `port` must be a Port for copying')
            p = port._copy()
            p.parent = self
        elif isinstance(name, Port):
            p = name._copy()
            p.parent = self
            name = p.name
        else:
            p = Port(name = name, midpoint = midpoint, width = width,
                orientation = orientation, parent = self)
        if name is not None: p.name = name
        if p.name in self.ports:
            raise ValueError('[DEVICE] add_port() error: Port name already exists in this device') 
        self.ports[p.name] = p
        return p
        
        
    def add_array(self, device, start = (0,0), spacing = (10,0), num_devices = 6, config = None, **kwargs):
         # Check if ``device`` is actually a device-making function
        if callable(device):    d = make_device(fun = device, config = config, **kwargs)
        else:                   d = device
        references = []
        for n in range(num_devices):
            sd = self.add_ref(d)
            sd.move(destination = np.array(spacing)*n, origin = -np.array(start))
            references.append(sd)
        return references
        

    def label(self, text = 'hello', position = (0,0), layer = 255):
        if len(text) >= 1023:
            raise ValueError('[DEVICE] label() error: Text too long (limit 1024 chars)') 
        gds_layer, gds_datatype = _parse_layer(layer)

        if type(text) is not str: text = str(text)
        self.add(gdspy.Label(text = text, position = position, anchor = 'o',
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
            new_elements = []
            for e in D.elements:
                if isinstance(e, gdspy.PolygonSet):
                    new_polygons = []
                    new_layers = []
                    new_datatypes = []
                    for n, layer in enumerate(e.layers):
                        original_layer = (e.layers[n], e.datatypes[n])
                        original_layer = _parse_layer(original_layer)
                        if invert_selection: keep_layer = (original_layer in layers)
                        else:                keep_layer = (original_layer not in layers)
                        if keep_layer:
                            new_polygons += [e.polygons[n]]
                            new_layers += [e.layers[n]]
                            new_datatypes += [e.datatypes[n]]
                     # Don't re-add an empty polygon
                    if len(new_polygons) > 0:
                        e.polygons = new_polygons
                        e.layers = new_layers
                        e.datatypes = new_datatypes
                        new_elements.append(e)
            D.elements = new_elements

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

        


    def distribute(self, elements, direction = 'x', spacing = 100, separation = True):
        multiplier = 1
        if   direction[0] == '+':
            direction = direction[1:]
        elif direction[0] == '-':
            direction = direction[1:]
            multiplier = -1

        xy = np.array([0,0])
        for e in elements:
            e.center = xy
            if direction == 'x':
                xy = xy + (np.array([spacing, 0]) + np.array([e.xsize, 0])*(separation==True))*multiplier
            elif direction == 'y':
                xy = xy + (np.array([0, spacing]) + np.array([0, e.ysize])*(separation==True))*multiplier
            else:
                raise ValueError('[PHIDL] distribute() needs a direction of "x", "+y", "-x", etc')


    def flatten(self,  single_layer = None):
        if single_layer is None:
            super(Device, self).flatten(single_layer=None, single_datatype=None, single_texttype=None)
        else:
            gds_layer, gds_datatype = _parse_layer(single_layer)
            super(Device, self).flatten(single_layer = gds_layer, single_datatype = gds_datatype, single_texttype=gds_datatype)

        temp = self.elements
        self.elements = []
        [self.add_polygon(poly) for poly in temp]
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
        """ Returns copies of all the ports of the Device"""
        port_list = [p._copy() for p in self.ports.values()]
        
        if depth is None or depth > 0:
            for r in self.references:
                if depth is None: new_depth = None
                else:             new_depth = depth - 1
                ref_ports = r.parent.get_ports(depth=new_depth)
                
                # Transform ports that came from a reference
                ref_ports_transformed = []
                for rp in ref_ports:
                    new_port = rp._copy()
                    new_midpoint, new_orientation = r._transform_port(rp.midpoint, \
                    rp.orientation, r.origin, r.rotation, r.x_reflection)
                    new_port.midpoint = new_midpoint
                    new_port.new_orientation = new_orientation
                    ref_ports_transformed.append(new_port)
                port_list += ref_ports_transformed
            
        return port_list


    def remove(self, items):
        if type(items) not in (list, tuple):  items = [items]
        for item in items:
            try:
                self.elements.remove(item)
            except:
                raise ValueError("""[PHIDL] Device.remove() cannot find the item
                                 it was asked to remove in the Device "%s".""" % (self.name))
            if isinstance(item, DeviceReference):
                # If appears in list of aliases, remove that alias
                self.aliases = { k:v for k, v in self.aliases.items() if v != item}

        self._bb_valid = False
        return self

    
    def rotate(self, angle = 45, center = (0,0)):
        if angle == 0: return self
        for e in self.elements:
            if isinstance(e, Polygon):
                e.rotate(angle = angle, center = center)
            elif isinstance(e, DeviceReference):
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
        else: raise ValueError('[DeviceReference.move()] ``origin`` not array-like, a port, or port name')
            
        if isinstance(destination, Port):           d = destination.midpoint
        elif np.array(destination).size == 2:        d = destination
        elif destination in self.ports:   d = self.ports[destination].midpoint
        else: raise ValueError('[DeviceReference.move()] ``destination`` not array-like, a port, or port name')

        if axis == 'x': d = (d[0], o[1])
        if axis == 'y': d = (o[0], d[1])

        dx,dy = np.array(d) - o
        
        # Move geometries
        for e in self.elements:
            if isinstance(e, Polygon): 
                e.translate(dx,dy)
            if isinstance(e, DeviceReference): 
                e.move(destination = d, origin = o)
        for p in self.ports.values():
            p.midpoint = np.array(p.midpoint) + np.array(d) - np.array(o)
        
        # Move labels
        for l in self.labels:
            l.translate(dx,dy)
        
        self._bb_valid = False
        return self
            
    def reflect(self, p1 = (0,1), p2 = (0,0)):
        for e in self.elements:
            e.reflect(p1, p2)
        for p in self.ports.values():
            p.midpoint = _reflect_points(p.midpoint, p1, p2)
            phi = np.arctan2(p2[1]-p1[1], p2[0]-p1[0])*180/pi
            p.orientation = 2*phi - p.orientation
        self._bb_valid = False
        return self
    
    
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
        self._parent_ports = device.ports
        self._local_ports = {name:port._copy() for name, port in device.ports.items()}


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
        """ This property allows you to access my_device_reference.ports, and receive a copy
        of the ports dict which is correctly rotated and translated"""
        for key in self._parent_ports.keys():
            port = self._parent_ports[key] 
            new_midpoint, new_orientation = self._transform_port(port.midpoint, \
                port.orientation, self.origin, self.rotation, self.x_reflection)
            self._local_ports[key].midpoint = new_midpoint
            self._local_ports[key].orientation = mod(new_orientation,360)
            self._local_ports[key].parent = self
        return self._local_ports

    @property
    def info(self):
        return self.parent.info

    @property
    def meta(self):
        warnings.warn('[PHIDL] WARNING: .meta is being deprecated, please use .info instead')
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
        self.parent._bb_valid = False
        return self

        
    def rotate(self, angle = 45, center = (0,0)):
        if angle == 0: return self
        if type(center) is Port:  center = center.midpoint
        self.rotation += angle
        self.origin = _rotate_points(self.origin, angle, center)
        self.parent._bb_valid = False
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

        self.parent._bb_valid = False
        return self
        

    def connect(self, port, destination, overlap = 0):
        # ``port`` can either be a string with the name or an actual Port
        if port in self.ports: # Then ``port`` is a key for the ports dict
            p = self.ports[port]
        elif type(port) is Port:
            p = port
        else:
            raise ValueError('[PHIDL] connect() did not receive a Port or valid port name' + \
                ' - received (%s), ports available are (%s)' % (port, self.ports.keys()))
        self.rotate(angle =  180 + destination.orientation - p.orientation, center = p.midpoint)
        self.move(origin = p, destination = destination)
        self.move(-overlap*np.array([cos(destination.orientation*pi/180),
                                     sin(destination.orientation*pi/180)]))
        return self


#==============================================================================
# Plotting functions
#==============================================================================


def quickplot(items, show_ports = True, show_subports = True,
              label_ports = True, label_aliases = False, new_window = False):
    """ Takes a list of devices/references/polygons or single one of those, and
    plots them.  Also has the option to overlay their ports """
    if new_window: fig, ax = plt.subplots(1)
    else:
        ax = plt.gca()  # Get current figure
        ax.cla()        # Clears the axes of all previous polygons
    ax.axis('equal')
    ax.grid(True, which='both', alpha = 0.4)
    ax.axhline(y=0, color='k', alpha = 0.2, linewidth = 1)
    ax.axvline(x=0, color='k', alpha = 0.2, linewidth = 1)
    
    # Iterate through each each Device/DeviceReference/Polygon
    np.random.seed(0)
    if type(items) is not list:  items = [items]
    for item in items:
        if isinstance(item, (Device, DeviceReference, gdspy.CellArray)):
            polygons_spec = item.get_polygons(by_spec=True, depth=None)
            for key in sorted(polygons_spec):
                polygons = polygons_spec[key]
                layerprop = _get_layerprop(layer = key[0], datatype = key[1])
                _draw_polygons(polygons, ax, facecolor = layerprop['color'],
                               edgecolor = 'k', alpha = layerprop['alpha'])
                for name, port in item.ports.items():
                    if (port.width is None) or (port.width == 0):
                        _draw_port_as_point(port)
                    else:
                        _draw_port(port, arrow_scale = 2, shape = 'full', color = 'k')
                    ax.text(port.midpoint[0], port.midpoint[1], name)
            if isinstance(item, Device) and show_subports is True:
                for sd in item.references:
                    for name, port in sd.ports.items():
                        _draw_port(port, arrow_scale = 1, shape = 'right', color = 'r')
                        ax.text(port.midpoint[0], port.midpoint[1], name)
            if isinstance(item, Device) and label_aliases is True:
                for name, ref in item.aliases.items():
                    ax.text(ref.x, ref.y, str(name), style = 'italic', color = 'blue',
                             weight = 'bold', size = 'large', ha = 'center')
        elif isinstance(item, Polygon):
            polygons = item.polygons
            layerprop = _get_layerprop(item.layers[0], item.datatypes[0])
            _draw_polygons(polygons, ax, facecolor = layerprop['color'],
                           edgecolor = 'k', alpha = layerprop['alpha'])
    plt.draw()
    plt.show(block = False)
    


def _get_layerprop(layer, datatype):
    # Colors generated from here: http://phrogz.net/css/distinct-colors.html
    layer_colors = ['#3dcc5c', '#2b0fff', '#cc3d3d', '#e5dd45', '#7b3dcc',
    '#cc860c', '#73ff0f', '#2dccb4', '#ff0fa3', '#0ec2e6', '#3d87cc', '#e5520e']
                     
    l = Layer.layer_dict.get((layer, datatype))
    if l is not None:
        color = l.color
        alpha = l.alpha
    else:
        color = layer_colors[np.mod(layer, len(layer_colors))]
        alpha = 0.8
    return {'color':color, 'alpha':alpha}
    
    
def _draw_polygons(polygons, ax, **kwargs):
    """ This function uses a trick where all polygon points are concatenated, 
    separated only by NaN values.  This speeds up drawing considerably, see
    http://exnumerus.blogspot.com/2011/02/how-to-quickly-plot-polygons-in.html
    """
    nan_pt = np.array([[np.nan, np.nan]])
    polygons_with_nans = [np.concatenate((p, [p[0]], nan_pt), axis = 0) for p in polygons]
    all_polygons = np.vstack(polygons_with_nans)
    plt.fill(all_polygons[:,0], all_polygons[:,1], **kwargs)


def _draw_port(port, arrow_scale = 1, **kwargs):
    x = port.midpoint[0]
    y = port.midpoint[1]
    nv = port.normal
    n = (nv[1]-nv[0])*arrow_scale
    dx, dy = n[0], n[1]
    xbound, ybound = np.column_stack(port.endpoints)
    #plt.plot(x, y, 'rp', markersize = 12) # Draw port midpoint
    plt.plot(xbound, ybound, 'r', alpha = 0.5, linewidth = 3) # Draw port edge
    plt.arrow(x, y, dx, dy,length_includes_head=True, width = 0.1*arrow_scale,
              head_width=0.3*arrow_scale, alpha = 0.5, **kwargs)


def _draw_port_as_point(port, **kwargs):
    x = port.midpoint[0]
    y = port.midpoint[1]
    plt.plot(x, y, 'r+', alpha = 0.5, markersize = 15, markeredgewidth = 2) # Draw port edge
