# device-layout
CAD layout and geometry creation utilities for photonic and superconducting circuits

# Changelog

## 0.6.3

### Bugfixes
 -  Under the hood
 
## 0.6.2

### New features
 - Added label_aliases=False default option to quickplot.  Do quickplot(D, label_aliases = True) to draw text with aliases on it
 -  Added Device.extract().  See the tutorial for details
 -  Device.add_polygon() can now receive lists of polygons, and will return lists in kind

### Changes
 - pg.snspd() and pg.snspd_expanded() can now be defined using any 2 constrains of: xsize, ysize, or num_squares.
 -  Nomenclature change on pg.fill_rectangle(), from exclude_layers to avoid_layers
 -  Changed pg.ytron_round() variable names, fixed layer error on cross

### Bugfixes
 -  Fixed SNSPD squares calculation and added num_squares constraints
 -  

## 0.6.1

### New features
 - Added ability to make "alias" for DeviceReference.  See the tutorial
 - Can now use Port.x and Port.y instead of calling Port.midpoint[0] and Port.midpoint[1]
 - Added Device.size property.  Returns the equivalent of [Device.xsize, Device.ysize]
 - Added include_layers to pg.fill_rectangle, allowing you to override exclude_layers
 - Added pg.circle(), pg.cross(), and pg.ellipse()

### Changes
 - None

### Bugfixes
 - Fixed route_manhattan, had issue with using Device.add() instead of Device.add_ref()
 - Fixed pg.snspd layer = 0 by default
 - Fixed Port.endpoints

## 0.6.0

### Changes
 - phidl.geometry.route() works still but is being deprecated, will now be in phid.routing.route_basic().  pg.route() will be deleted in the near future

### New features
 - Several new photonic geometries by Sonia Buckley
 - Advanced Manhattan routing written by Jeff Chiles

### Bugfixes
 - Very likely we added more bugs than we fixed in this version!

## 0.5.6
### Bugfixes
 - Fixes to phidl.geometry.hecken_taper()

### Changes
 - pg.taper() now has a default argument of width2=None

### New features
 - Fill tool (phidl.geometry.fill_rectangle) now allows you to invert the fill
- New function pg.invert() which will perform a boolean inversion on elements

## 0.5.5
### Bugfixes
 - Problem with route() and inset() caused by implementation of Layer().  You can now pass route() and inset() a Layer and it will parse it correctly

## 0.5.4
### Changes
 - A few under-the-hood optimizations
## 0.5.5
### Bugfixes
 - Problem with route() and inset() caused by implementation of Layer().  You can now pass route() and inset() a Layer and it will parse it correctly

## 0.5.4
### Changes
 - A few under-the-hood optimizations
 
### Bugfixes
 - Fixed error with quickplot where the last edge of a polygon was not rendered

## 0.5.3
### New features
 - Layers() are now implemented.  See tutorial_example.py "Using Layers" section for a demonstration
 - You can now construct a Device using a set of parameters.  See "Constructing a Device from set of parameters" in tutorial_example.py
 - Usage of the annotate() function has been added to tutorial_example.py
 - quickplot rendering speed has been sped up by a factor of 10x
 
### Changes
 - pg.rectangle() now takes "size" as a parameter rather than "point1" and "point2"
