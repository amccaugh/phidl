
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
x2 = kl_cell.insert(kdb.DCellInstArray(ref_cell.cell_index(), transformation*transformation))
x3 = kl_cell.insert(kdb.DCellInstArray(ref_cell.cell_index(), transformation*transformation*transformation))

# Iterate through each instance in a cell
for kl_instance in kl_cell.each_inst():
    print(kl_instance)

# Iterate through each polygon in a cell
for kl_polygon in kl_cell.each_shape(0):
    print(kl_polygon)


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
import phidl
import phidl.geometry as pg
from phidl import Device, quickplot as qp
import klayout.db as kdb
from phidl.device_layout import DeviceReference, Polygon
from phidl.device_layout import layout
from phidl.device_layout import _parse_layer, _kl_shape_iterator, _get_kl_layer, _objects_to_kl_region

layout.clear()
phidl.reset()

D = Device()
ref1 = D << pg.snspd(layer = 1).rotate(3)
item = D << pg.snspd(layer = 2).rotate(45)
polygon = D.add_polygon([[0,0],[1,2],[5,5]])
item2 = D.add_label('test')
self= D

#kl_shapes = item.kl_cell.shapes(item.kl_layer_idx)
#kl_shapes.find(item.kl_shape)
self.kl_cell.erase(item.kl_instance)
item.kl_shape.is_valid()

qp(D)

#%%

layout.clear()

#D2 = pg.ellipse(layer = 77)

    def remap_layers(self, layermap = {}, include_labels = True):
        layermap = {_parse_layer(k):_parse_layer(v) for k,v in layermap.items()}


gds_layer, gds_datatype = _parse_layer(layer)
kl_layer_idx, kl_layer =  _get_kl_layer(gds_layer, gds_datatype)
new_gds_layer, new_gds_datatype = _parse_layer(new_layer)
new_kl_layer_idx, new_kl_layer =  _get_kl_layer(new_gds_layer, new_gds_datatype)

if include_labels == True:
    shape_type = kdb.Shapes.SPolygons | kdb.Shapes.STexts
else:
    shape_type = kdb.Shapes.SPolygons

iterator_dict = _kl_shape_iterator(D.kl_cell, shape_type = shape_type, depth = None)
iterator = iterator_dict[kl_layer_idx]
for kl_shape in iterator:
    kl_shape.layer = new_kl_layer_idx
    
    

qp(D)
#%%

return D


#%%

cell_list2 = [cell for cell in layout2.each_cell()]
cell_indices2 = {cell.name: cell.cell_index() for cell in cell_list2}

for i in cell_indices.values():
    layout.rename_cell(i, "")

qp(D3, new_window = True)
#%%
D = Device()
p = D.add_polygon([[1,2],[3,4],[6,9]], layer = (2,3))
p2 = D.add_polygon([[1,2],[3,4],[6,90]])
p3 = D.add_polygon([[1,2,3],[4,6,90]], layer = 7)
#p.rotate(90)

l  = D.add_label('test123', position = [50,0])
print(l.kl_shape)
print(l.kl_text)
l.move([700,7])
print(l.kl_shape)
print(l.kl_text)

D.movex(111)
#D.rotate(45)
print(l.kl_shape)
print(l.kl_text)

kl_shapes = []
for layer_idx in layout.layer_indices():
    kl_shapes += D.kl_cell.each_shape(layer_idx)
print(kl_shapes)
transformation = kdb.DCplxTrans(
    1,
    37,  # Rotation
    False,# X-axis mirroring
    77, # X-displacement
    77,  # Y-displacement
    )

t2 = kdb.DTrans(
    37,  # Rotation
    False,# X-axis mirroring
    77, # X-displacement
    77,  # Y-displacement
    )
x = kl_shapes[3]
[klp.transform(transformation) for klp in kl_shapes]
kl_shapes = []
for layer_idx in layout.layer_indices():
    kl_shapes += D.kl_cell.each_shape(layer_idx)
## Add a text
#kl_text = kdb.DText.new('This is a test', 1.7, 2.9).
#kl_text_shape = D.kl_cell.shapes(0).insert(kl_text)
#kl_text_shape_text = kl_text_shape.dtext


D.kl_cell.write("both.gds")

#%%




#%%
import phidl
import phidl.geometry as pg
from phidl import quickplot as qp
import klayout.db as kdb
from phidl.device_layout import Device

D = pg.rectangle()
D.add_polygon([[0,0],[1,2],[.1,0]], layer = 2)

D.center = [47,47]

qp(D)
#%%

# Get instances

# Transform each instance

# Get polygons

# Transform each polygon


print(d.kl_instance)
transformation = kdb.DCplxTrans(
    float(1),  # Magnification
    float(37),  # Rotation
    False,# X-axis mirroring
    float(0), # X-displacement
    float(0),  # Y-displacement
    )
d.kl_instance.transform(transformation)
print(d.kl_instance)