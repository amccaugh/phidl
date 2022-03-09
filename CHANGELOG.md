# Changelog

## 1.6.1 (March 8, 2022)

### Bugfixes
- Fixed naming of some previously-unnamed geometries

## 1.6.0 (Sept 28, 2021)

### New features
- Huge new routing rewrite for `phidl.routing`, including automatic manhattan routing with custom cross-sections!  See [the routing documentation](https://phidl.readthedocs.io/en/latest/tutorials/routing.html) for details. Big thanks to Jeffrey Holzgrafe @jolzgrafe for this contribution
- `Path`s can now be used to produce sharp angles, in addition to smooth bends. See [the Path documentation](https://phidl.readthedocs.io/en/latest/tutorials/waveguides.html#Sharp/angular-paths)

### Changes
- The `Path()` constructor no longer forces the path to start at (0,0)

### Bugfixes
- Interactive zoom for `quickplot()` is now disabled by default in Jupyter-type notebooks to avoid strange scrolling behavior.  This can be manually changed using `set_quickplot_options(interactive_zoom = True)`

## 1.5.2 (May 19, 2021)
- Hotfix to guarantee Python 2 compatibility

## 1.5.1 (May 19, 2021)

### New features
- New `pg.gridsweep()` function, allowing for easy creation of parameter sweeps on a grid.  See [the documentation](https://phidl.readthedocs.io/en/latest/geometry_reference.html#Gridsweep) for details.

![phidl example image](https://phidl.readthedocs.io/en/dev/_images/geometry_reference_36_0.png)

- New [tutorial on layers](https://phidl.readthedocs.io/en/latest/tutorials/layers.html)
- Can now define layer to be `None` to prevent creation of polygons
- Added `pg.litho_ruler()` for creation of lithographic rulers/measurement

### Bugfixes
- pg.grid() bugfix: `shape` now in format (x,y), and array size correctly calculated when making `shape` e.g. (4,-1)



## 1.5.0 (April 19, 2021)

### New features
- Better interactive windows for `quickplot()`! Now you can use the mousewheel/trackpad scroll to zoom in and out, and right-click or left-click to zoom to a region.
- Added `blocking` option for `quickplot()` (thanks @giumc)

### Changes
- Quickplot options (such as displaying ports, subports, or aliases) are now set using `set_quickplot_options()`

### Bugfixes
- Fix for Path function `smooth()`, which broke when sequential waypoints were co-linear (thanks @giumc)
- Fix for non-C-continguous arrays in `hash_geometry()`  (thanks Joaquin Matres @joamatab)


### Bugfixes

## 1.4.4 (Feb 23, 2021)

Bugfix release

### Bugfixes
- Allow labels imported through `import_gds()` to be moved (thanks Joaquin Matres @joamatab)
- Fix to Path.smooth() to prevent right-angle turns from accidentally having an additional +180 degrees applied to them (thanks Jeffrey Holzgrafe @jolzgrafe)


## 1.4.3 (Dec 11, 2020)

### New features
- Added `open_ports` argument to `pg.outline`, which allows you to cut holes in the outline at Port locations on a Device. See the [outline reference here](https://phidl.readthedocs.io/en/latest/geometry_reference.html#Outline) (thanks to Owen Medeiros @omedeiro)

![phidl example image](https://amccaugh.github.io/phidl/outline_open_port.png)

- Easier-to-read quickstart tutorial
- Added `num_squares` to `info` dictionary of `pg.optimal_step` (thanks Ekkehart Schmidt)

### Bugfixes
- Fixed bug in `pp.smooth()` that forced paths to start out traveling to the right (orientation = 0 degrees) (thanks to Sebastian Pauka @spauka)


## 1.4.2 (Oct 7, 2020)

Bugfix release

### Bugfixes
- Fix for Device xmin/xmax/ymin/ymax property assignment (e.g. `D.xmin = 30`) causing incorrect movement of references and labels


## 1.4.1 (Oct 6, 2020)

### New features
- Added font support to `pg.text()` - Now you can use built-in fonts or specify a .TTF/.OTF font, including full unicode support (thanks to Sebastian Pauka @spauka).  See the [geometry reference library here](https://phidl.readthedocs.io/en/latest/geometry_reference.html#Text)
- Added new `smooth()` function that allows you to construct a smooth path by defining waypoints.  The corners are smoothed either with the circular `pp.arc()` function or the adiabatic straight-to-bend `pp.euler()` function. See the [path/waveguide tutorial here](https://phidl.readthedocs.io/en/latest/tutorials/waveguides.html#Waypoint-based-path-creation)

![phidl example image](https://amccaugh.github.io/phidl/feature-smooth.png)

- Added `route_turn_manhattan()` function for more flexible manhattan routing (thanks to @mr-roger-a)

### Changes
- Fix to start/end angles for `pp.spiral()`
- Style consistency fix for `pp.arc()` when angle negative

### Bugfixes
- Fix to casting issue when moving components with labels #78 (thanks to Joaquin Matres @joamatab)



## 1.4.0 (Sept 14, 2020)

Huge update with lots of quality-of-life improvements.

### New features
- New [path / waveguide module](https://phidl.readthedocs.io/en/latest/tutorials/waveguides.html) featuring intuitive and fast path building, sub-millisecond polygon generation, and modular cross-sections (thanks to Alex Tait @atait, Dylan Oh @dmwo, Samuel Gyger @gyger, and Florian Vogelbacher).
- Now you can easily `Group` objects for easier manipulation. See the [Group tutorial here](https://phidl.readthedocs.io/en/latest/tutorials/group.html)
- Significantly extended documentation, including new tutorials, geometry library description with images, and API / function reference.  See https://phidl.readthedocs.io/
- Docstrings added for all functions
- Addition of `pg.grid()` a grid-placement function for creating 2D arrays of devices (thanks to Samuel Gyger @gyger)

### Changes
- `filename` argument in `write_gds()` can now accept `pathlib` or file buffer   (thanks to Samuel Gyger @gyger)

### Bugfixes
- int-casting fix in `routing`  (thanks to Samuel Gyger @gyger)
- Fix for `pg.optimal_step()` if `start_width==end_width` and `symmetric==True` (thanks to Ekkehart Schmidt)
- Fix capitalization errors of color names in `Layer`  (thanks to Jeff Shainline)
- Fix to `@endpoints.setter`


## 1.3.0 (May 5, 2020)

### New features
- Now introducing the automatic `pg.packer()` geometry-packing tool:
![phidl example image](https://amccaugh.github.io/phidl/packer.png)
- New documentation for `pg.packer()`, `align()`, and `distribute()`.  See [**Geometry + function documentation**](https://phidl.readthedocs.io/)

### Changes
- Configurable toplevel `cellname` argument in `write_gds()`
- Change to the arguments available in `distribute()`.  See the [**Geometry + function documentation**](https://phidl.readthedocs.io/)
- Rename `reflect()` to `mirror()`. Note that `reflect()` will continue to work until May 2021 so as not to break any existing code

### Bugfixes
- Int-casting compatibility fix with latest numpy (thanks @gyger)
- Bugfix to `pg.basic_die()` for non-square die (thanks @jonnyfountain)
- Fixed harmless but annoying warning if PyQt was not installed
- Small under-the-hood optimizations


## 1.2.2 (January 17, 2020)

### Bugfixes
- Fixed extremely rare bug with `write_gds()` which could potentially cause cell name collisions
- `pg.boolean()` no longer errors when passed empty geometries

## 1.2.1 (January 13, 2020)

- Maintenance update to work with `gdspy` 1.5

### New features
- References, arrays and polygons can all be assigned to a Device using `D['myname'] = `

### Changes
- Default precision changed to `1e-4` on boolean functions (for 1 unit = 1 micron, this corresponds to 0.1 nanometer precision)
- Added `join`, `miter` and `max_points` arguments to `pg.offset` to match the arguments with gdspy
- The `Device.label()` function is going to be move to `Device.add_label()` - both will still work for now, but when using `label()` a warning will pop up suggesting you switch to `add_label()` since it will be removed in future versions.

### Bugfixes
- Maintenance update to work with `gdspy` 1.5 (specifically `pg.import_gds()` fixes)
- Allow DeviceReferences to be used with `pg.port_to_geometry()` (thanks Alex Tait @atait )

## 1.2.0 (December 1, 2019)

### New features
- Major optimization of `pg.boolean()`, `pg.offset()`, `pg.outline()`, and `pg.invert()`:  The `num_divisions` argument can now be used to divide up the geometry into multiple rectangular regions and process each region sequentially (which is much, much more computationally efficient).  If you have a large geometry that takes a long time to process, try using `num_divisions = [10,10]` to optimize the operation -- you may see speed improvements well over 100x for very large geometries (>1 million points).
- New geometry documentation with quick picture references and code examples! See [**Geometry + function documentation**](https://phidl.readthedocs.io/)

### Changes
- Big update to `quickplot()`, should be faster now and not have issues with overlapping polygons generating whitespace.
- Can now use `port.center`, which is identical to `port.midpoint`


### Bugfixes
- Allow labels to be correctly moved/rotated
- Fix fontsize and figure initialization of `quickplot()`
- Bugfix for 'd' shape in `pg.flagpole()`



## 1.1.0 (October 16, 2019)

### New features
- New online notebook to try out PHIDL!  Try now in an interactive online notebook: [Link](https://mybinder.org/v2/gh/amccaugh/phidl/master?filepath=phidl_tutorial_example.ipynb)
- Added full CellArray support, use the `D.add_array()` function (see the [tutorial](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py) for more details)
- Allow plotting of `DeviceReference`s directly in `quickplot`

### Changes
- Added `connector_symmetric` argument to `pg.snspd_expanded()`


### Bugfixes
- Bounding box cache speed improvement


## 1.0.3 (May 23, 2019)
- Maintenance release to work with `gdspy 1.4`
- Removal of `scipy` from strict installation requirements

### Bugfixes
- Minor fix to `distribute()`


## 1.0.2 (March 26, 2019)

### New features
- Added tutorial section for phidl.geometry library lithographic shapes (resolution tests, calipers, stars, etc)
- Added `symmetric` argument to pg.optimal_step()
- Experimental port phidl.geometry function `pg.ports_to_geometry()` and `pg.geometry_to_ports()` which converts Ports in a Device into polygon-geometry (and vice-versa) so they can be saved into the GDS file (in the style of SiEPIC). (contribution thanks to Alex Tait @atait)
- Added support for `magnification` and `rotation` of `Label`s  (contribution thanks to Alex Tait @atait)

### Changes
- Precision for boolean functions set to 1e-6 by default now
- `position` argument removed from pg.text()

### Bugfixes
- Fixed rare but persistent bug affecting boolean operations (e.g. `pg.offset()`,  `pg.outline()`, `pg.boolean()`, `pg.union()`) on polygons with sub-precision floating point errors.  Will no longer cause jagged edges when two points are misaligned by very small amounts (e.g. when points that should be equal differ by 1e-27 due to floating point imprecision)
- Fix for `pg.import_gds()` so that items can be moved/rotated correctly after importing
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
