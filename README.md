[![Build Status](https://travis-ci.com/amccaugh/phidl.svg?branch=master)](https://travis-ci.com/amccaugh/phidl)

# PHIDL
PHotonic and Integrated Device Layout - GDS CAD layout and geometry creation for photonic and superconducting circuits

- [**Installation / requirements**](#installation--requirements)
- [**Tutorial + examples**](https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py#L35) (or [Try now in an interactive notebook](https://mybinder.org/v2/gh/amccaugh/phidl/master?filepath=phidl_tutorial_example.ipynb))
- [**Geometry + function documentation**](https://phidl.readthedocs.io/)
- [About PHIDL](#about-phidl)
- [Changelog](https://github.com/amccaugh/phidl/blob/master/phidl/CHANGELOG.md) (latest update 1.3.0 (May 5, 2020))
  - Now introducing the automatic [`pg.packer()` geometry-packing tool](https://phidl.readthedocs.io/en/latest/#packer-align-distribute):

`pg.packer(D_list, spacing = 1.25, aspect_ratio = (2,1))`

![phidl example image](https://amccaugh.github.io/phidl/packer.png)


# Installation / requirements
- Install or upgrade with `pip install -U phidl`
- Python 2 >=2.6 or Python 3 >=3.5
- If you are on Windows or Mac and don't already have `gdspy` installed, you will need a C++ compiler
    - For Windows + Python 3, install the Microsoft ["Build Tools for Visual Studio"](https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017)
    - For Windows + Python 2, install [Microsoft Visual C++ Compiler for Python 2.7](https://www.microsoft.com/en-us/download/details.aspx?id=44266)
    - For Mac, install "Xcode" from the App Store, then run the command `xcode-select --install` in the terminal

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

There are also handy functions to help pack shapes into as small an area as possible:

```
pg.packer(D_list, spacing = 1.25, aspect_ratio = (2,1))
```

![phidl example image](https://amccaugh.github.io/phidl/packer.png)

You can also do things like create a backing fill to make sure the resist develops uniformly while still creating a solid ground plane, with user-defined margins.  Below is an image of a device which needed a ground plane.  A single-line fill function was able to fill the required area (purple), electrically connecting all of the ground structures together:

![phidl example image](https://amccaugh.github.io/phidl/readme_6.png)
