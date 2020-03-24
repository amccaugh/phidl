
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
#%% Functions



def rotate(polygon, angle):
    # Transform polygon
    transformation = kdb.DCplxTrans(
            1,  # Magnification
            37,  # Rotation
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
import phidl
import phidl.geometry as pg
from phidl import quickplot as qp
import klayout.db as kdb
from phidl.device_layout import DeviceReference
   
D = pg.rectangle(size = [10,10])
D2 = pg.rectangle(layer = 1)



D.add_polygon([[1,2],[-3,4],[6,9]], layer = (2,3))

#D.rotate(35)
D.move([20,15])
#D.rotate(-45)

d = D.add_ref(D2)

d.rotate(35)

qp(D, new_window = True)



D3 = pg.snspd()
#d = DeviceReference(D2)
qp(D3, new_window = True)
#%%
D = Device()
p = D.add_polygon([[1,2],[3,4],[6,9]], layer = (2,3))
p2 = D.add_polygon([[1,2],[3,4],[6,90]])
p3 = D.add_polygon([[1,2,3],[4,6,90]], layer = 7)
#p.rotate(90)

#D.kl_cell
print(D.get_polygons(True))

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