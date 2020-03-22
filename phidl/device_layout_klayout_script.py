
#%%

import numpy as np
import klayout.db as kdb

# Overview given by https://www.klayout.de/doc-qt5/programming/database_api.html#h2-518

# Create layout aka library
layout = kdb.Layout()

# Create cell https://www.klayout.de/doc-qt5/code/class_Cell.html
new_cell = layout.create_cell("toplevel")



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
p1 = new_cell.shapes(new_layer).insert(new_poly)
p2 = new_cell.shapes(new_layer).insert(new_poly.transform(transformation))
p3 = new_cell.shapes(new_layer2).insert(new_poly.transform(transformation))
#shapes = new_cell.shapes(new_layer)

# Iterate through polygons in one layer of a cell
list(new_cell.each_shape(new_layer))

# Iterate through all polygons in a cell (get_polygons()) https://www.klayout.de/doc/code/class_Cell.html#method9
all_polygons_iterator = new_cell.begin_shapes_rec(new_layer)
while not all_polygons_iterator.at_end():
    polygon = all_polygons_iterator.shape()
    print(polygon)
    all_polygons_iterator.next()

# List all layers in use

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