[![Build Status](https://travis-ci.com/amccaugh/phidl.svg?branch=master)](https://travis-ci.com/amccaugh/phidl)

# PHIDL
PHotonic and Integrated Device Layout - GDS CAD layout and geometry creation for photonic and superconducting circuits

- [Installation / requirements](#installation--requirements)
- [About PHIDL](#about-phidl)
- [Changelog](#changelog)
- [Tutorial + examples](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py#L35)

# Installation / requirements
- Install or upgrade with `pip install -U phidl`
- Python 2 >=2.6 or Python 3 >=3.5
- If you are on Windows and don't already have `gdspy` installed, you will need a C++ compiler
    - For Python 3, install the [MS C++ Build Tools for VS 2017](https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017)
    - For Python 2, install [Microsoft Visual C++ Compiler for Python 2.7](https://www.microsoft.com/en-us/download/details.aspx?id=44266)

# About PHIDL

*fiddle (verb) - /ˈfidl/ - to make minor manual movements, especially to adjust something*

PHIDL is an open-source GDS-based CAD tool for Python 2 and 3 which extends and simplifies the excellent [gdspy](https://github.com/heitzmann/gdspy). It strives to simplify GDSII geometry creation by making the design process layout-driven, rather than coordinate-driven.  The base installation includes a large library of simple shapes (e.g. rectangles, circles), photonic structures (e.g. sine curve waveguides), and superconducting nanowire shapes (e.g. single photon detectors) which are fully parameterized. It also has a built-in quick-plotting function based on Qt (or matplotlib) which allows you view the state of any GDS object, useful when scripting geometry-making functions. It also has a [__very thorough tutorial__](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py#L35) as well which will walk you through the process of getting acquainted with PHIDL.

The purpose of PHIDL is to fill a void in the GDS design space: creation of elements in a simple, layout-driven, parameterized way, without a large amount of code overhead. Many GDS tools exist, but they tend to fall in one of two categories: (1) GUI-based layout tools with ad-hoc scripting interfaces, or (2) full-featured Cadence-style layout software which requires 30 lines of boilerplate/overhead code just to draw a simple ring. 

The goal is to bring the usability of Illustrator / Inkscape drawing programs to the GDS scripting world. Like Python itself, it aims to be readable, and intuitive.  For instance, when building a geometry you don't have to worry about what the exact coordinates are anymore. If you want to separate two ellipses in the x direction by 5 units, you can do things like this:

`ellipse1.xmin = ellipse2.xmax + 5`

or if you want to move then rotate one ellipse by 45 degrees you can do

`ellipse2.move([1,7]).rotate(45)`

There's a few dozen shortcuts like this that make life easier built into PHIDL--they're simple, but they make a world of difference when you just want to e.g. space a ring resonator some distance from a waveguide without having to track each and every coordinate of the shape.

[](http://amccaugh.github.io/phidl)

![phidl example image](https://amccaugh.github.io/phidl/readme_1.png)


There's also a "port" functionality which allows you to snap together geometry like Legos without caring about where exactly the absolute coordinates of either geometry is.  For instance, connecting the above misaligned rectangles is a two-line command:

![phidl example image](https://amccaugh.github.io/phidl/readme_2.png)

It also allows you to do things like add text and create smooth or straight routing curves between "ports" of different devices, convenient for making electrical or optical connections:

![phidl example image](https://amccaugh.github.io/phidl/readme_3.png)
![phidl example image](https://amccaugh.github.io/phidl/readme_4.png)
    

Other useful functionality available are standard operations like booleans:

![phidl example image](https://amccaugh.github.io/phidl/readme_8.png)

 and less standard ones like creating outlines. A whole layout can be outlined directly in the GDS without requiring you to use Beamer (useful for positive-tone resist structures):

`pg.outline(D, distance = 0.7, layer = 4)`

![phidl example image](https://amccaugh.github.io/phidl/readme_5.png)
 
The geometry library also has useful resolution test-structures built into it, for instance

```
pg.litho_calipers(num_notches = 7, offset_per_notch = 0.1)
pg.litho_steps(line_widths = [1,2,4,8,16])
pg.litho_star(num_lines = 16, line_width = 3)
```

![phidl example image](https://amccaugh.github.io/phidl/readme_7.png)


You can also do things like create a backing fill to make sure the resist develops uniformly while still creating a solid ground plane, with user-defined margins.  Below is an image of a device which needed a ground plane.  A single-line fill function was able to fill the required area (purple), electrically connecting all of the ground structures together:

![phidl example image](https://amccaugh.github.io/phidl/readme_6.png)



# Changelog

## 1.0.2 (March 26, 2019)

### New features
- Added tutorial section for phidl.geometry library lithographic shapes (resolution tests, calipers, stars, etc)
- Added `symmetric` argument to pg.optimal_step()
- Experimental port phidl.geometry function `pg.port_to_geometry()` which converts Ports in a Device into polygon-geometry so they can be saved into the GDS file (in the style of SiEPIC). (contribution thanks to Alex Tait @atait)

### Changes
- Precision for boolean functions set to 1e-6 by default now
- `position` argument removed from pg.text()

### Bugfixes
- Fixed rare but persistent bug affecting boolean operations on polygons with sub-precision floating point errors.  Will no longer cause jagged edges when two points are misaligned by very small amounts (e.g. points differ by 4e-27 units)
- Fix for `remove_layers()` correctly preserves references now  (contribution thanks to Alex Tait @atait)
- Suppressed unecessary warnings


## 1.0.1 (Jan 21, 2019)

### New features
- `D.remove()` can now remove Ports as well as references/polygons

### Bugfixes
- Can't have a major release without at least one bug!  Fixed errors introduced by optimized-rotation algorithm.

## 1.0.0 (Jan 14, 2019)
- 1.0 release!  The core functionality of phidl has been stable for over 18 months, and all major planned features have been implemented.  Time to reflect that in the version number!

### New features
- Significant upgrades to `quickplot2`: now shows coordinates, a help message box (press ?), and a scale notation (along with several under-the-hood optimizations)
- Added `D.hash_geometry()` -- use to generate a SHA1-based hash of the polygons in a Device
- Added `phidl.utilities.load_lyp()`, which loads a KLayout layer properties (.lyp) file and converts it into a LayerSet (contribution thanks to Alex Tait @atait)

### Changes
- Optimized rotation so 90-degree rotations (contribution thanks to Alex Tait @atait)
- Function documentation for geometry module (contribution thanks to Jimmy Gammell @jgammell and Dylan Oh @dmwo)
- `pytest` implementation for internal consistency checking


## 0.9.0 (Oct 9, 2018)
### New features
- Added `pg.union()` - a handy convenience function to join (union) polygons together, either by-layer or all together (see [tutorial](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py) for full details)
- Added `phidl.utilities.write_svg()` - allows you to write your geometry directly to SVG for the sake of publishing figures easily (see [tutorial](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py) for full details)
- Added pg.xor_diff(A,B) - Produces an XOR operation between Devices A and B, which can by used to compare differences between A and B (see [tutorial](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py) for full details)
- Allow usage of a Python `set` (e.g. {3,5,6}) as an input to the `layer` argument of add_polygon (and thus all geometry-creation functions in phidl.geometry) to create the polygons on multiple layers.  (see [tutorial](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py) for full details)

### Bugfixes
- None


## 0.8.10 (August 23, 2018)
### Bugfixes
- Minor upkeep bugfixes


## 0.8.9 (July 24, 2018)
### New features
- The addition of the argument `max_cellname_length` added to `D.write_gds()`.  It is `28` by default, to guarantee maximum compatibility with GDS specifications (32 is generally the lower limit, and `write_gds()` applies a # afterwards to prevent duplicate cellnames).
- New documentation backend (contribution thanks to Alex Tait @atait)
- Added `D.remap_layers()` which allows you to to move all polygons contained on a layer within your Device to another layer. See [tutorial](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py) for details
- Added `D.remove_layers()` which lets you remove all polygon geometry (optionally including labels) from a Device on the specified layers. See [tutorial](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py) for details
 
### Bugfixes
- Further fixes to `D.write_gds()` for rare edge cases

## 0.8.8 (July 19, 2018)
### New features
- You can now add any shape to multiple layers at once by passing a whole `LayerSet` to the `layer` argument.  See [tutorial](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py) for details
- Update to `D.write_gds()` which guarantees cell names within a Device will never overlap.  If you want to disable this feature for more control over cell names, change the `auto_rename` argument to `False` ( `D.write('myfile.gds', auto_rename = False)`)

### Bugfixes
- Modifications made to work with gdspy>=1.3.1

## 0.8.7 (July 11, 2018)
### Bugfixes
- Minor bugfixes to `pg.litho_calipers()` and `pg.litho_star()`

## 0.8.6 (July 9, 2018)
### New features
- `D.absorb(my_reference)` can be used to easily absorb references into a Device; polygons will be extracted from the reference, added to the Device, and then the reference will be removed. See the [tutorial](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py) for more details
- Added lithographic-resolution test structures including stars (`pg.litho_star()`), calipers (`pg.litho_calipers()`), and variable-size negative-tone and positive-tone steps (`pg.litho_steps()`) (Contribution from Dylan Oh @dmwo).  

### Changes
- Made `write_gds()` autofix names to guarantee no duplicates cell names ever appear

### Bugfixes
- The gdspy bounding box caching has been reallowed
- Single-layer flatten fix for `D.flatten()`
- `quickplot` and `quickplot2` now fail gracefully if the user does not have matlplotlib or Qt respectively.

## 0.8.5 (June 15, 2018)
### New features
- Added `pg.optimal_90deg()`, the optimal 90-degree turn for superconducting nanowires from Clem & Berggren

### Bugfixes
- `quickplot2` visual improvement: Ports now show up on top of subports
- `quickplot2` visual improvement: Port colors now colorblind-friendly
- Fixed very rare `make_device()` error

## 0.8.4 (June 6, 2018)
### New features
- Added `<<` operator to add references.  `r = D.add_ref(Rect)` can now be (optionally) written as `r = D << Rect`.
- Added `D.get_ports()` which allows you to gather the locations and information about all ports in Device.
- A `LayerSet` can now be previewed.  Running the geometry function `pg.preview_layerset()` will generate a `Device` which shows all of the layers, useful for previewing color schemes.
- `quickplot()` now shows zero-width ports (e.g. a "pin") as a + sign.
- `quickplot()` now defaults to redrawing within a single window, rather than creating a new window for every call
- Added a `.info` dictionary to `Port`, useful for recording information about a port (e.g. `myport.info['wavelength'] = 1550`)
- Updated [tutorial](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py)

### Changes
- `pg.optimal_hairpin()`,  `pg.snspd()`,  and `pg.snspd_expanded()` now have the argument `turn_ratio` which defines how wide the turn is w.r.t. the argument `wire_width`
- The `layer` argument in `D.add_polygon()` can now accept lists of `Layer`s. Use this if you want to a single polygon shape to multiple layers.
- Rearranged an argument location: The `name` argument for the `Layer()` class is now the third argument to allow the ability to make `Layer`s like Layer(1,0)
- Removed some deprecated old geometry

### Bugfixes
- Minor bugfix to guarantee quickplot() shows up from the Python/IPython console.
- Minor bugfix in [tutorial](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py) example file

## 0.8.2 (Apr 19, 2018)

### New features
- Added the LayerSet class.  See the [tutorial](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py), but essentially this class makes a convenient container to stores layers
- Added `phidl.utilities.write_lyp()` (Contribution from Dylan Oh @dmwo).  Using a LayerSet, you can now create KLayout-compatible .lyp files.  This allows you to get the same coloring in the KLayout viewer as you have specified in PHIDL.
- Several new electrical test structures (Contribution from Jacob Melonis @melonisj)  Specifically: via chain tests (`pg.test_via()`), inter- and intra-layer comb insulation tests (`pg.test_comb()`), and critical current test structures (`pg.test_ic`).
- `add_ref()` can now take a list of input Devices and will return a list of the generated references, e.g. `ref_a,ref_b,ref_c = D.add_ref([A,B,C])` 

### Changes

### Bugfixes
- Fixed issue with pg.import_gds() when `layers` (remapping argument) was None.
- Bugfix in pg.copy() which produced incorrectly-copied ports



## 0.8.1 (Feb 7, 2018)

### New features
 - New function `pg.extract()` which extracts all the polygons from a set of specified layers from a Device, and creates a new Device with those polygons in them. See [tutorial](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py) for details
 - New Device-copying functions `pg.copy()` and `pg.deepcopy()` which allows you to copy a Device wholesale (very useful if you want to flatten() a Device but not destroy the original).  `pg.copy` maintains the underlying connections & references to other Devices, while `pg.deepcopy` creates completely new copies of every underlying polygon and reference.  See [tutorial](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py) for details
 - Introduced an LRU cache for computationally-intensive Device-making functions.  By using the `@device_lru_cache` decorator, any function which returns a Device can be memoized.  See [tutorial](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py) for more details

### Changes
- Since the `extract()` process creates new geometry, `D.extract()` has been removed in favor of placing it in the geometry library `pg.extract()`
- `pg.import_gds` default argument is now `flatten = False`
- Updated [tutorial](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py) text

### Bugfixes
 - Fixed port deepcopy bug, should result in large performance enhancement for Devices with lots of sub-references and ports
 - Fixed some rare errors with D.flatten()
 - Some internal changes to make working with Device.uid easier


## 0.8.0 (Dec 6, 2017)

### New features
 - `pg.import_gds()` can now import without flattening all the polygons to a single layer
 - Added `Device.flatten()` function to flatten references into raw polygons.  See [tutorial](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py) for details
 - Added `Device.remove()` function to remove geometry.  See [tutorial](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py) for details
 - Added more informative error messages
 - `__repr__` and `__str__` implemented for Device & DeviceReference: You can now quickly get useful information about a Device by just typing the variable into the console.  For instance entering `D` or `print(D)` into the Python console will print `Device (name "Myshape003191", ports ['R_center', 'bottom', 'input'], aliases ['hello'], 13 elements, 13 references)`

### Changes
- Using a config dictionary as a specification is no longer done with `Device(device_function, config = myconfig)`.  Now it is done with an explicit function, `make_device(device_function, config = myconfig)` (importable as `import phidl.make_device`).  See the [tutorial](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py) for more info
 - `Device.meta` is now being replaced with `Device.info` for clarity of nomenclature.  `Device.meta` will still work but will issue a warning.
 - `Device.annotate()` is now being replaced with `Device.label()` to be more consistent with GDS naming conventions.  `Device.annotate()` will still work but will issue a warning.

### Bugfixes
 - Made compatible with gdspy >= 1.2
 - Specified names for phidl.geometry objects

## 0.7.1 (August 28, 2017)

### New features
 - Updated [tutorial](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py) text

### Changes
 - Large changes to pg.import_gds().  If your GDS file only has one toplevel cell, you do not need to specify a cellname, pg.import_gds() will automatically grab that cell.  Also, it imports all layers by default now, although you can still choose which layers and even create a layer mapping based on whether you pass the `layers` argument a list or dict.  See tutorial for more information.

### Bugfixes
 - Many small ones under the hood


## 0.7.0 (May 26, 2017)

### New features
 - Updated tutorial text significantly
 - Added import_gds() function!  You can now import existing GDS files and use them in your geometry.  Try adding premade organization logos and hand-drawn test structures without having to code them from scratch.  See tutorial for full explanation.
 - Added ``overlap`` keyword argument to the connect() function.  Now when connecting two ports together, you can force them to overlap.  See tutorial for full explanation.
 - Added point_path() function to phidl.routing. It takes a width value and list of points and and will create path of constant width which follows the list of points.

### Changes
 - Text scaling has been changed to be more sensible; now pg.text('abc', size = 100) will produce letters 100 units tall (previously would produce text 125 units tall).

### Bugfixes
 - Many small ones under the hood

## 0.6.5 (Apr 3, 2017)

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


## 0.6.4 (Feb 21, 2017)

### New features
 - Added "quickplot2", a more robust/easier to use viewer which instead of being based on matplotlib is based Qt.
   - Zoom/scroll uses same buttons as KLayout
   - Toggle labels for ports/aliases with F1, F2, F3
   - Reset view with Escape key


## 0.6.3 (Jan 23, 2017)

### Bugfixes
 -  Under the hood
 

## 0.6.2 (Jan 13, 2017)

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


## 0.6.1 (Jan 9, 2017)

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

## 0.6.0 (Dec 13, 2016)

### Changes
 - phidl.geometry.route() works still but is being deprecated, will now be in phid.routing.route_basic().  pg.route() will be deleted in the near future

### New features
 - Several new photonic geometries by Sonia Buckley
 - Advanced Manhattan routing written by Jeff Chiles

### Bugfixes
 - Very likely we added more bugs than we fixed in this version!

## 0.5.6 (Dec 12, 2016)
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

## 0.5.4 (Dec 5, 2016)
### Changes
 - A few under-the-hood optimizations
 
### Bugfixes
 - Fixed error with quickplot where the last edge of a polygon was not rendered
 - Problem with route() and inset() caused by implementation of Layer().  You can now pass route() and inset() a Layer and it will parse it correctly

## 0.5.3 (Nov 22, 2016)
### New features
 - Layers() are now implemented.  See tutorial_example.py "Using Layers" section for a demonstration
 - You can now construct a Device using a set of parameters.  See "Constructing a Device from set of parameters" in tutorial_example.py
 - Usage of the annotate() function has been added to tutorial_example.py
 - quickplot rendering speed has been sped up by a factor of 10x
 
### Changes
 - pg.rectangle() now takes "size" as a parameter rather than "point1" and "point2"

## 0.4.1 (Nov 3, 2016)
- Large number of upgrades

## 0.3.0 (Sep 12, 2016)
- Initial release!
