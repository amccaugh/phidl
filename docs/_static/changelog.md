# Changelog

## 0.8.2

### New features
- Added the LayerSet class.  See the tutorial, but essentially this class makes a convenient container to stores layers
- Added `phidl.utilities.write_lyp()` (Contribution from Dylan Oh @dmwo).  Using a LayerSet, you can now create KLayout-compatible .lyp files.  This allows you to get the same coloring in the KLayout viewer as you have specified in PHIDL.
- Several new electrical test structures (Contribution from Jacob Melonis @melonisj)  Specifically: via chain tests (`pg.test_via()`), inter- and intra-layer comb insulation tests (`pg.test_comb()`), and critical current test structures (`pg.test_ic`).
- `add_ref()` can now take a list of input Devices and will return a list of the generated references, e.g. `ref_a,ref_b,ref_c = D.add_ref([A,B,C])` 

### Changes

### Bugfixes
- Fixed issue with pg.import_gds() when `layers` (remapping argument) was None.
- Bugfix in pg.copy() which produced incorrectly-copied ports



## 0.8.1

### New features
 - New function `pg.extract()` which extracts all the polygons from a set of specified layers from a Device, and creates a new Device with those polygons in them. See tutorial for details
 - New Device-copying functions `pg.copy()` and `pg.deepcopy()` which allows you to copy a Device wholesale (very useful if you want to flatten() a Device but not destroy the original).  `pg.copy` maintains the underlying connections & references to other Devices, while `pg.deepcopy` creates completely new copies of every underlying polygon and reference.  See tutorial for details
 - Introduced an LRU cache for computationally-intensive Device-making functions.  By using the `@device_lru_cache` decorator, any function which returns a Device can be memoized.  See tutorial for more details

### Changes
- Since the `extract()` process creates new geometry, `D.extract()` has been removed in favor of placing it in the geometry library `pg.extract()`
- `pg.import_gds` default argument is now `flatten = False`
- Updated tutorial text

### Bugfixes
 - Fixed port deepcopy bug, should result in large performance enhancement for Devices with lots of sub-references and ports
 - Fixed some rare errors with D.flatten()
 - Some internal changes to make working with Device.uid easier


## 0.8.0

### New features
 - `pg.import_gds()` can now import without flattening all the polygons to a single layer
 - Added `Device.flatten()` function to flatten references into raw polygons.  See tutorial for details
 - Added `Device.remove()` function to remove geometry.  See tutorial for details
 - Added more informative error messages
 - `__repr__` and `__str__` implemented for Device & DeviceReference: You can now quickly get useful information about a Device by just typing the variable into the console.  For instance entering `D` or `print(D)` into the Python console will print `Device (name "Myshape003191", ports ['R_center', 'bottom', 'input'], aliases ['hello'], 13 elements, 13 references)`

### Changes
- Using a config dictionary as a specification is no longer done with `Device(device_function, config = myconfig)`.  Now it is done with an explicit function, `make_device(device_function, config = myconfig)` (importable as `import phidl.make_device`).  See the tutorial for more info
 - `Device.meta` is now being replaced with `Device.info` for clarity of nomenclature.  `Device.meta` will still work but will issue a warning.
 - `Device.annotate()` is now being replaced with `Device.label()` to be more consistent with GDS naming conventions.  `Device.annotate()` will still work but will issue a warning.

### Bugfixes
 - Made compatible with gdspy >= 1.2
 - Specified names for phidl.geometry objects

## 0.7.1

### New features
 - Updated tutorial text

### Changes
 - Large changes to pg.import_gds().  If your GDS file only has one toplevel cell, you do not need to specify a cellname, pg.import_gds() will automatically grab that cell.  Also, it imports all layers by default now, although you can still choose which layers and even create a layer mapping based on whether you pass the `layers` argument a list or dict.  See tutorial for more information.

### Bugfixes
 - Many small ones under the hood


## 0.7.0

### New features
 - Updated tutorial text significantly
 - Added import_gds() function!  You can now import existing GDS files and use them in your geometry.  Try adding premade organization logos and hand-drawn test structures without having to code them from scratch.  See tutorial for full explanation.
 - Added ``overlap`` keyword argument to the connect() function.  Now when connecting two ports together, you can force them to overlap.  See tutorial for full explanation.
 - Added point_path() function to phidl.routing. It takes a width value and list of points and and will create path of constant width which follows the list of points.

### Changes
 - Text scaling has been changed to be more sensible; now pg.text('abc', size = 100) will produce letters 100 units tall (previously would produce text 125 units tall).

### Bugfixes
 - Many small ones under the hood

## 0.6.5

### New features
 - Added pg.boolean() to perform AND/NOT/OR/XOR operations on shapes
 - Added pg.outline() to create outlines of shapes, useful for positive-tone resist processes

### Changes
 - Switched development to Python 3.  Python 2 will still be supported going forward however
 - The function to expand/contract polygons pg.inset() has been deprecated in favor of the more aptly-named pg.offset().  pg.offset(D, distance = -1) will contract the shapes in D, pg.offset(D, distance = 1) will expand them.

### Bugfixes
 - Fixes necessary to make compatible with Python 3
 - Fixed pg.connector() midpoint argument
 - Compatibility fixes to make compatible with gdspy>=1.1.2


## 0.6.4

### New features
 - Added "quickplot2", a more robust/easier to use viewer which instead of being based on matplotlib is based Qt.
   - Zoom/scroll uses same buttons as KLayout
   - Toggle labels for ports/aliases with F1, F2, F3
   - Reset view with Escape key


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
