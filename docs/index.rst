.. phidl documentation master file, created by
   sphinx-quickstart on Sat Nov  2 18:17:54 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root ``toctree`` directive.

###############
PHIDL Geometry and Function Reference
###############

Geometry and function reference for `PHIDL <https://github.com/amccaugh/phidl>`_, an open-source Python GDS layout and CAD geometry creation tool

Note: This is still a work-in-progress. For a more complete tutorial please see  `the tutorial here <https://github.com/amccaugh/phidl/blob/master/phidl/phidl_tutorial_example.py>`_

###############
Basic shapes
###############


***********
Rectangle
***********

To create a simple rectangle, there are two functions.  The first is ``pg.rectangle()`` which creates a basic rectangle:

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-rectangle
   :end-before: create_image

.. image:: _static/rectangle.png
   :align: center

The other way to create a rectangle is to define one by its bounding box using ``pg.bbox()``.  This is useful if you want to create a rectangle which exactly surrounds a piece of existing geometry.  For example, if we have an arc geometry and we want to define a box around it, we can use ``pg.bbox()``: 

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-bbox
   :end-before: create_image

.. image:: _static/bbox.png
   :align: center


***********
Cross
***********

The ``pg.cross()`` function creates a cross structure:

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-cross
   :end-before: create_image

.. image:: _static/cross.png
   :align: center


***********
Ellipse
***********

The ``pg.ellipse()`` function creates an ellipse by defining the major and minor radii:

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-ellipse
   :end-before: create_image

.. image:: _static/ellipse.png
   :align: center


***********
Circle
***********

The ``pg.circle()`` function creates a circle:

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-circle
   :end-before: create_image

.. image:: _static/circle.png
   :align: center

***********
Ring
***********

The ``pg.ring()`` function creates a ring.  The radius refers to the center radius of the ring structure (halfway between the inner and outer radius).

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-ring
   :end-before: create_image

.. image:: _static/ring.png
   :align: center


***********
Arc
***********

The ``pg.arc()`` function creates an arc.  The radius refers to the center radius of the arc (halfway between the inner and outer radius).  The arc has two ports, ``1`` and ``2``, on either end, allowing you to easily connect it to other structures.

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-arc
   :end-before: create_image

.. image:: _static/arc.png
   :align: center


***********
Tapers
***********

The taper is defined by setting its length and its start and end length.  It has two ports, ``1`` and ``2``, on either end, allowing you to easily connect it to other structures.

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-taper
   :end-before: create_image

.. image:: _static/taper.png
   :align: center

The ramp structure is similar to the taper except it is asymmetric.  It also has two ports, ``1`` and ``2``, on either end.

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-ramp
   :end-before: create_image

.. image:: _static/ramp.png
   :align: center



***********
Common compound shapes
***********


.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-L
   :end-before: create_image

.. image:: _static/L.png
   :align: center



.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-C
   :end-before: create_image

.. image:: _static/C.png
   :align: center


###############
Text
###############

PHIDL has an implementation of the DEPLOF font with the majority of english ASCII characters represented.

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-text
   :end-before: create_image

.. image:: _static/text.png
   :align: center



###############
Packer / align / distribute
###############


***********
Packer
***********

The ``pg.packer()`` function is able to pack geometries together into rectangular bins. If a ``max_size`` is specified, the function will create as many bins as is necessary to pack all the geometries and then return a list of the filled-bin Devices.

Here we generate several random shapes then pack them together automatically. We allow the bin to be as large as needed to fit all the Devices by specifying ``max_size = (None, None)``.  By setting ``aspect_ratio = (2,1)``, we specify the rectangular bin it tries to pack them into should be twice as wide as it is tall:

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-packer1
   :end-before: create_image

.. image:: _static/packer1.png
   :align: center

Say we need to pack many shapes into multiple 500x500 unit die. If we set ``max_size = (500,500)`` the shapes will be packed into as many 500x500 unit die as required to fit them all:

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-packer2
   :end-before: create_image

.. image:: _static/packer2.png
   :align: center

Note that the packing problem is an NP-complete problem, so ``pg.packer()`` may be slow if there are more than a few hundred Devices to pack (in that case, try pre-packing a few dozen at a time then packing the resulting bins). Requires the ``rectpack`` python package.


***********
Distribute
***********

The ``distribute()`` function allows you to space out elements within a Device evenly in the x or y direction.  It is meant to duplicate the distribute functionality present in Inkscape / Adobe Illustrator:

.. image:: _static/inkscape_distribute.png
   :align: center

Say we start out with a few random-sized rectangles we want to space out:

.. image:: _static/distribute1.png
   :align: center

Oftentimes, we want to guarantee some distance between the objects.  By setting ``separation = True`` we move each object such that there is ``spacing`` distance between them:

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-distribute2
   :end-before: create_image

.. image:: _static/distribute2.png
   :align: center

Alternatively, we can spread them out on a fixed grid by setting ``separation = False``. Here we align the left edge (``edge = 'min'``) of each object along a grid spacing of 100:

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-distribute3
   :end-before: create_image

.. image:: _static/distribute3.png
   :align: center

The alignment can be done along the right edge as well by setting ``edge = 'max'``, or along the center by setting ``edge = 'center'`` like in the following:


.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-distribute4
   :end-before: create_image

.. image:: _static/distribute4.png
   :align: center




***********
Align
***********

The ``align()`` function allows you to elements within a Device horizontally or vertically.  It is meant to duplicate the alignment functionality present in Inkscape / Adobe Illustrator:

.. image:: _static/inkscape_align.png
   :align: center

Say we ``distribute()`` a few objects, but they're all misaligned:

.. image:: _static/align1.png
   :align: center

we can use the ``align()`` function to align their top edges (``alignment = 'ymax'):

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-align2
   :end-before: create_image

.. image:: _static/align2.png
   :align: center

or align their centers (``alignment = 'y'):

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-align3
   :end-before: create_image

.. image:: _static/align3.png
   :align: center

other valid alignment options include ``'xmin', 'x', 'xmax', 'ymin', 'y', and 'ymax'``

###############
Boolean / outline / offset / invert
###############

There are several common boolean-type operations available in the geometry library.  These include typicaly boolean operations (and/or/not/xor), offsetting (expanding/shrinking polygons), outlining, inverting, and 

***********
Boolean
***********

The ``pg.boolean()`` function can perform AND/OR/NOT/XOR operations, and will return a new geometry with the result of that operation.

Speedup note: The ``num_divisions`` argument can be used to divide up the geometry into multiple rectangular regions and process each region sequentially (which is more computationally efficient).  If you have a large geometry that takes a long time to process, try using ``num_divisions = [10,10]`` to optimize the operation.

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-boolean
   :end-before: create_image

.. image:: _static/boolean.png
   :align: center

***********
Offset
***********

The ``pg.offset()`` function takes the polygons of the input geometry, combines them together, and expands/contracts them.  The function returns polygons on a single layer -- it does not respect layers.

Speedup note: The ``num_divisions`` argument can be used to divide up the geometry into multiple rectangular regions and process each region sequentially (which is more computationally efficient).  If you have a large geometry that takes a long time to process, try using ``num_divisions = [10,10]`` to optimize the operation.

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-offset
   :end-before: create_image

.. image:: _static/offset.png
   :align: center


***********
Outline
***********

The ``pg.outline()`` function takes the polygons of the input geometry then performs an offset and "not" boolean operation to create an outline.  The function returns polygons on a single layer -- it does not respect layers.

Speedup note: The ``num_divisions`` argument can be used to divide up the geometry into multiple rectangular regions and process each region sequentially (which is more computationally efficient).  If you have a large geometry that takes a long time to process, try using ``num_divisions = [10,10]`` to optimize the operation.

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-outline
   :end-before: create_image

.. image:: _static/outline.png
   :align: center


***********
Invert
***********

The ``pg.invert()`` function creates an inverted version of the input geometry.  The function creates a rectangle around the geometry (with extra padding of distance ``border``), then subtract all polygons from all layers from that rectangle, resulting in an inverted version of the geometry.

Speedup note: The ``num_divisions`` argument can be used to divide up the geometry into multiple rectangular regions and process each region sequentially (which is more computationally efficient).  If you have a large geometry that takes a long time to process, try using ``num_divisions = [10,10]`` to optimize the operation.

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-invert
   :end-before: create_image

.. image:: _static/invert.png
   :align: center


***********
Union
***********

The ``pg.union()`` function is a "join" function, and is functionally identical to the "OR" operation of ``pg.boolean()``.  The one difference is it's able to perform this function layer-wise, so each layer can be individually combined.

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-union
   :end-before: create_image

.. image:: _static/union.png
   :align: center

***********
XOR / diff
***********

The ``pg.xor_diff()`` function can be used to compare two geometries and identify where they are different.  Specifically, it performs a layer-wise XOR operation.  If two geometries are identical, the result will be an empty Device.  If they are not identical, any areas not shared by the two geometries will remain.

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-xor_diff
   :end-before: create_image

.. image:: _static/xor_diff.png
   :align: center





###############
Lithography structures
###############


***********
Step-resolution
***********

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-litho_steps
   :end-before: create_image

.. image:: _static/litho_steps.png
   :align: center

***********
Lithographic star
***********


.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-litho_star
   :end-before: create_image

.. image:: _static/litho_star.png
   :align: center


***********
Calipers (inter-layer alignment)
***********


.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-litho_calipers
   :end-before: create_image

.. image:: _static/litho_calipers.png
   :align: center




###############
Importing GDS files
###############


.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-import_gds
   :end-before: create_image

.. image:: _static/import_gds.png
   :align: center



.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-preview_layerset
   :end-before: create_image

.. image:: _static/preview_layerset.png
   :align: center



###############
Useful contact pads / connectors
###############


.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-compass
   :end-before: create_image

.. image:: _static/compass.png
   :align: center



.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-compass_multi
   :end-before: create_image

.. image:: _static/compass_multi.png
   :align: center



.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-flagpole
   :end-before: create_image

.. image:: _static/flagpole.png
   :align: center



.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-straight
   :end-before: create_image

.. image:: _static/straight.png
   :align: center



.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-connector
   :end-before: create_image

.. image:: _static/connector.png
   :align: center


.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-tee
   :end-before: create_image

.. image:: _static/tee.png
   :align: center






###############
Chip / die template
###############


.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-basic_die
   :end-before: create_image

.. image:: _static/basic_die.png
   :align: center


###############
Optimal superconducting curves
###############

The following structures are meant to reduce "current crowding" in superconducting thin-film structures (such as superconducting nanowires).  They are the result of conformal mapping equations derived in  Clem, J. & Berggren, K. "Geometry-dependent critical currents in superconducting nanocircuits." Phys. Rev. B 84, 1–27 (2011). ``link to paper <http://dx.doi.org/10.1103/PhysRevB.84.174510>``_


.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-optimal_hairpin
   :end-before: create_image

.. image:: _static/optimal_hairpin.png
   :align: center



.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-optimal_step
   :end-before: create_image

.. image:: _static/optimal_step.png
   :align: center



.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-optimal_90deg
   :end-before: create_image

.. image:: _static/optimal_90deg.png
   :align: center



.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-snspd
   :end-before: create_image

.. image:: _static/snspd.png
   :align: center



.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-snspd_expanded
   :end-before: create_image

.. image:: _static/snspd_expanded.png
   :align: center



###############
Copying and extracting geometry
###############

.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-extract
   :end-before: create_image

.. image:: _static/extract.png
   :align: center



.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-copy
   :end-before: create_image

.. image:: _static/copy.png
   :align: center



.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-deepcopy
   :end-before: create_image

.. image:: _static/deepcopy.png
   :align: center



.. literalinclude:: gen_geometry.py
   :language: python
   :dedent: 0
   :start-after: example-copy_layer
   :end-before: create_image

.. image:: _static/copy_layer.png
   :align: center



.. ###############
.. To-be documented
.. ###############




.. .. literalinclude:: gen_geometry.py
..    :language: python
..    :dedent: 0
..    :start-after: example-fill_rectangle
..    :end-before: create_image

.. .. image:: _static/fill_rectangle.png
..    :align: center



.. .. literalinclude:: gen_geometry.py
..    :language: python
..    :dedent: 0
..    :start-after: example-grating
..    :end-before: create_image

.. .. image:: _static/grating.png
..    :align: center



.. .. literalinclude:: gen_geometry.py
..    :language: python
..    :dedent: 0
..    :start-after: example-test_via
..    :end-before: create_image

.. .. image:: _static/test_via.png
..    :align: center



.. .. literalinclude:: gen_geometry.py
..    :language: python
..    :dedent: 0
..    :start-after: example-test_comb
..    :end-before: create_image

.. .. image:: _static/test_comb.png
..    :align: center



.. .. literalinclude:: gen_geometry.py
..    :language: python
..    :dedent: 0
..    :start-after: example-test_ic
..    :end-before: create_image

.. .. image:: _static/test_ic.png
..    :align: center



.. .. literalinclude:: gen_geometry.py
..    :language: python
..    :dedent: 0
..    :start-after: example-test_res
..    :end-before: create_image

.. .. image:: _static/test_res.png
..    :align: center



Subsub heading
====================


Welcome to phidl's documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:``genindex``
* :ref:``modindex``
* :ref:``search``
