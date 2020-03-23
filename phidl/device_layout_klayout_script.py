
#%%

import numpy as np
import klayout.db as kdb

# Overview given by https://www.klayout.de/doc-qt5/programming/database_api.html#h2-518

# Create layout aka library
layout = kdb.Layout()

# Create cell https://www.klayout.de/doc-qt5/code/class_Cell.html
kl_cell = layout.create_cell("toplevel")
other_cell = layout.create_cell("toplevel")



# Create polygon
coords = [[1,2],[4,5],[8,3]]
new_poly = kdb.DSimplePolygon([kdb.DPoint(x, y) for x, y in coords])

# Transform polygon
transformation = kdb.CplxTrans(
        1,  # Magnification
        37,  # Rotation
        False,# X-axis mirroring
        10.0, # X-displacement
        20.0  # Y-displacement
        )
new_poly.transform(transformation)



# Create layer https://www.klayout.de/doc-qt5/code/class_Layout.html#m_layer
new_layer = layout.layer(1,6) # new_layer represents the *index* of the relevant layer in the layout's layer list
new_layer2 = layout.layer(3,7) 
new_layer3 = layout.layer(4,8) 
layer_indices = layout.layer_indexes() # Get the layout's indices of layers
layout.layer_infos() # Get the layout's indices of layers
grab_layer = layout.layer(3,7)



# Add polygon to the cell
p1 = kl_cell.shapes(new_layer).insert(new_poly)
p2 = kl_cell.shapes(new_layer).insert(new_poly.transform(transformation))
p3 = kl_cell.shapes(new_layer2).insert(new_poly.transform(transformation))
#shapes = kl_cell.shapes(new_layer)

# Iterate through polygons in one layer of a cell
list(kl_cell.each_shape(new_layer))

# Iterate through all polygons in a cell (get_polygons()) https://www.klayout.de/doc/code/class_Cell.html#method9
all_polygons_iterator = kl_cell.begin_shapes_rec(new_layer)
while not all_polygons_iterator.at_end():
    polygon = all_polygons_iterator.shape()
    print(polygon)
    all_polygons_iterator.next()

# List all layers in use in the entire layout
layers = [(l.layer, l.datatype) for l in layout.layer_infos()]

# List all layers which contain >0 polygons in a cell
layers = []
layer_infos = layout.layer_infos()
for layer_idx in layout.layer_indices():
    kl_iterator = kl_cell.begin_shapes_rec(layer_idx)
    if not kl_iterator.at_end(): # Then there are shapes on that layer
         layers.append( (layer_infos[layer_idx].layer, layer_infos[layer_idx].datatype) )
    
layers = [(l.layer, l.datatype) for l in layout.layer_infos()]    


# Get bounding box of polygon
b = kl_cell.dbbox()
bbox = [[b.left, b.bottom],[b.right, b.top]] 


# Create new cell reference (cell instance) https://www.klayout.de/doc/code/class_CellInstArray.html#method39
ref_cell = other_cell
transformation = kdb.CplxTrans(
        1,  # Magnification
        37,  # Rotation
        False,# X-axis mirroring
        10.0, # X-displacement
        20.0  # Y-displacement
        )
x = kl_cell.insert(kdb.DCellInstArray(ref_cell.cell_index(), transformation))



#%% Functions



def rotate(polygon, angle):
    # Transform polygon
    transformation = kdb.CplxTrans(
            1,  # Magnification
            angle,  # Rotation
            False,# X-axis mirroring
            0, # X-displacement
            0,  # Y-displacement
            )
    polygon.transform(transformation)
    return polygon


#%% Notes

# Get child cells: Cell#each_child_cell. 
# Get dependent cells:  Cell#called_cells 
# Get parent cells:  Cell#caller_cells

# Get_polygons in section "Recursive full or region queries" https://www.klayout.de/doc-qt5/programming/database_api.html#h2-907
#layout = RBA::Application::instance.main_window.current_view.active_cellview.layout
## start iterating shapes from cell "TOP", layer index 0
#si = layout.begin_shapes(layout.cell_by_name("TOP"), 0)
#while !si.at_end?
#  puts si.shape.to_s + " with transformation " + si.trans.to_s
#  si.next
#end
    

#%%
from phidl.device_layout import _parse_layer
from gdspy import CellArray
import gdspy
from phidl.quickplotter import _get_layerprop, _draw_polygons, _update_bbox


def _kl_polygon_to_array(polygon):
    return [ (pt.x,pt.y) for pt in polygon.each_point() ]
    
    
    

class Polygon(_GeometryHelper):

    def __init__(self, points, device, gds_layer, gds_datatype):
        self.parent = device.kl_cell
        points = np.array(points, dtype  = np.float64)
        polygon = kdb.DSimplePolygon([kdb.DPoint(x, y) for x, y in points]) # x and y must be floats
        kl_layer = layout.layer(gds_layer, gds_datatype)
        self.kl_polygon = device.kl_cell.shapes(kl_layer).insert(polygon)
    
    def _to_array(self):
        [ (pt.x,pt.y) for pt in self.kl_polygon.each_point() ]

    @property
    def bbox(self):
        b = new_poly.bbox() # Get KLayout bounding box object
        return [[b.left, b.bottom],[b.right, b.top]] 

    def rotate(self, angle = 45, center = (0,0)):
        transformation = kdb.DCplxTrans(
                1.0,  # Magnification
                angle,  # Rotation
                False,# X-axis mirroring
                0, # X-displacement
                0,  # Y-displacement
                )
        self.kl_polygon.transform(transformation)
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
        
        transformation = kdb.CplxTrans(
                1.0,  # Magnification
                0,  # Rotation
                False,# X-axis mirroring
                dx, # X-displacement
                dy,  # Y-displacement
                )
        self.kl_polygon.transform(transformation)
        
        return self


    def reflect(self, p1 = (0,1), p2 = (0,0)):
        print('Not yet implemented')
        return self


class Device(object):

    _next_uid = 0

    def __init__(self, name = 'Unnamed'):

        # Make a new blank device
        self.ports = {}
        self.info = {}
        self.aliases = {}
        # self.a = self.aliases
        # self.p = self.ports
        self.uid = Device._next_uid
        self._internal_name = name
        self.name = name
        gds_name = '%s%06d' % (self._internal_name[:20], self.uid) # Write name e.g. 'Unnamed000005'
        self.kl_cell = layout.create_cell(gds_name)
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
        if isinstance(element, (DeviceReference,Polygon,CellArray)):
            self.aliases[key] = element
        else:
            raise ValueError('[PHIDL] Tried to assign alias "%s" in Device "%s",  '
                'but failed because the item was not a DeviceReference' % (key, self.name))

    @property
    def layers(self):# List all layers which contain >0 polygons in a cell
        layers = []
        layer_infos = layout.layer_infos()
        for layer_idx in layout.layer_indices():
            kl_iterator = kl_cell.begin_shapes_rec(layer_idx)
            if not kl_iterator.at_end(): # Then there are shapes on that layer
                 layers.append( (layer_infos[layer_idx].layer, layer_infos[layer_idx].datatype) )

    # @property
    # def references(self):
    #     return [e for e in self.elements if isinstance(e, DeviceReference)]

    # @property
    # def polygons(self):
    #     return [e for e in self.elements if isinstance(e, gdspy.PolygonSet)]



    @property
    def bbox(self):
        b = self.kl_cell.dbbox()
        bbox = ((b.left, b.bottom),(b.right, b.top))
        return bbox

    def add_ref(self, device, alias = None):
        """ Takes a Device and adds it as a DeviceReference to the current
        Device.  """
        if _is_iterable(device):
            return [self.add_ref(E) for E in device]
        if not isinstance(device, Device):
            raise TypeError("""[PHIDL] add_ref() was passed something that
            was not a Device object. """)
        d = DeviceReference(device)   # Create a DeviceReference (CellReference)
        self.kl_cell.insert(d)

        if alias is not None:
            self.aliases[alias] = d
        return d                # Return the DeviceReference (CellReference)


    def add_polygon(self, points, layer = None):
        # Check if input a list of polygons by seeing if it's 3 levels deep
        try:
            points[0][0][0] # Try to access first x point
            return [self.add_polygon(p, layer) for p in points]
        except: pass # Verified points is not a list of polygons, continue on
#
#        if isinstance(points, gdspy.PolygonSet):
#            if layer is None:   layers = zip(points.layers, points.datatypes)
#            else:   layers = [layer]*len(points.polygons)
#            return [self.add_polygon(p, layer) for p, layer in zip(points.polygons, layers)]

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
        print(points)
        gds_layer, gds_datatype = _parse_layer(layer)
        polygon = Polygon(points = points, device = self, gds_layer = gds_layer,
            gds_datatype = gds_datatype)
        return polygon

    def get_polygons(self, by_spec = True, depth = None):
        # FIXME depth not implemented
        layer_infos = layout.layer_infos()
        if by_spec: polygons = {}
        else:       polygons = []
        # Loop through each layer in the layout collecting polygons
        for layer_idx in layout.layer_indices():
            layer_polygons = []
            all_polygons_iterator = self.kl_cell.begin_shapes_rec(layer_idx)
            while not all_polygons_iterator.at_end():
                polygon = all_polygons_iterator.shape().dsimple_polygon
                layer_polygons.append( _kl_polygon_to_array(polygon) )
                all_polygons_iterator.next()
            if not by_spec:
                polygons += layer_polygons
            elif by_spec and (len(layer_polygons) > 0):
                l = layer_infos[layer_idx]
                polygons[(l.layer, l.datatype)] = layer_polygons
        return polygons
        


class DeviceReference(object):
    def __init__(self, device, origin=(0, 0), rotation=0, magnification=None, x_reflection=False):
        if magnification == None: magnification = 1
        transformation = kdb.DCplxTrans(
                magnification,  # Magnification
                rotation,  # Rotation
                x_reflection,# X-axis mirroring
                origin[0], # X-displacement
                origin[1]  # Y-displacement
                )
        self.kl_instance = kl_cell.insert(kdb.DCellInstArray(device.kl_cell.cell_index(), transformation))
        
        # The ports of a DeviceReference have their own unique id (uid),
        # since two DeviceReferences of the same parent Device can be
        # in different locations and thus do not represent the same port
        self._local_ports = {name:port._copy(new_uid = True) for name, port in device.ports.items()}


D = Device()
p = D.add_polygon([[1,2],[3,4],[6,9]], layer = (2,3))
p2 = D.add_polygon([[1,2],[3,4],[6,90]])
p3 = D.add_polygon([[1,2,3],[4,6,90]], layer = 7)
#p.rotate(90)

#D.kl_cell
print(D.get_polygons(True))





quickplot(D)