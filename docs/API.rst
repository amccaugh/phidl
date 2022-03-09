#############
API Reference
#############


*********
Quickplot
*********

quickplot
=========

.. autofunction:: phidl.quickplot

set_quickplot_options
=====================

.. autofunction:: phidl.set_quickplot_options


****************
Geometry Library
****************

arc
===

.. autofunction:: phidl.geometry.arc


basic_die
=========

.. autofunction:: phidl.geometry.basic_die


bbox
====

.. autofunction:: phidl.geometry.bbox


boolean
=======

.. autofunction:: phidl.geometry.boolean


C
=

.. autofunction:: phidl.geometry.C


circle
======

.. autofunction:: phidl.geometry.circle


compass
=======

.. autofunction:: phidl.geometry.compass


compass_multi
=============

.. autofunction:: phidl.geometry.compass_multi


connector
=========

.. autofunction:: phidl.geometry.connector


copy
====

.. autofunction:: phidl.geometry.copy


copy_layer
==========

.. autofunction:: phidl.geometry.copy_layer


cross
=====

.. autofunction:: phidl.geometry.cross


deepcopy
========

.. autofunction:: phidl.geometry.deepcopy


ellipse
=======

.. autofunction:: phidl.geometry.ellipse


extract
=======

.. autofunction:: phidl.geometry.extract


fill_rectangle
==============

.. autofunction:: phidl.geometry.fill_rectangle


flagpole
========

.. autofunction:: phidl.geometry.flagpole


geometry_to_ports
=================

.. autofunction:: phidl.geometry.geometry_to_ports


grating
=======

.. autofunction:: phidl.geometry.grating


grid
====

.. autofunction:: phidl.geometry.grid


gridsweep
=========

.. autofunction:: phidl.geometry.gridsweep


hecken_taper
============

.. autofunction:: phidl.geometry.hecken_taper


import_gds
==========

.. autofunction:: phidl.geometry.import_gds


inset
=====

.. autofunction:: phidl.geometry.inset


invert
======

.. autofunction:: phidl.geometry.invert


L
=

.. autofunction:: phidl.geometry.L


litho_calipers
==============

.. autofunction:: phidl.geometry.litho_calipers


litho_star
==========

.. autofunction:: phidl.geometry.litho_star


litho_steps
===========

.. autofunction:: phidl.geometry.litho_steps


litho_ruler
===========

.. autofunction:: phidl.geometry.litho_ruler



meander_taper
=============

.. autofunction:: phidl.geometry.meander_taper


offset
======

.. autofunction:: phidl.geometry.offset


optimal_90deg
=============

.. autofunction:: phidl.geometry.optimal_90deg


optimal_hairpin
===============

.. autofunction:: phidl.geometry.optimal_hairpin


optimal_step
============

.. autofunction:: phidl.geometry.optimal_step


outline
=======

.. autofunction:: phidl.geometry.outline


packer
======

.. autofunction:: phidl.geometry.packer


polygon_ports
=============

.. autofunction:: phidl.geometry.polygon_ports


ports_to_geometry
=================

.. autofunction:: phidl.geometry.ports_to_geometry


preview_layerset
================

.. autofunction:: phidl.geometry.preview_layerset


racetrack_gradual
=================

.. autofunction:: phidl.geometry.racetrack_gradual


ramp
====

.. autofunction:: phidl.geometry.ramp


rectangle
=========

.. autofunction:: phidl.geometry.rectangle


ring
====

.. autofunction:: phidl.geometry.ring


snspd
=====

.. autofunction:: phidl.geometry.snspd


snspd_expanded
==============

.. autofunction:: phidl.geometry.snspd_expanded


straight
========

.. autofunction:: phidl.geometry.straight


taper
=====

.. autofunction:: phidl.geometry.taper


tee
===

.. autofunction:: phidl.geometry.tee


test_comb
=========

.. autofunction:: phidl.geometry.test_comb


test_ic
=======

.. autofunction:: phidl.geometry.test_ic


test_res
========

.. autofunction:: phidl.geometry.test_res


test_via
========

.. autofunction:: phidl.geometry.test_via


text
====

.. autofunction:: phidl.geometry.text


turn
====

.. autofunction:: phidl.geometry.turn


union
=====

.. autofunction:: phidl.geometry.union


xor_diff
========

.. autofunction:: phidl.geometry.xor_diff


ytron_round
===========

.. autofunction:: phidl.geometry.ytron_round


device_lru_cache
================

.. autoclass:: phidl.geometry.device_lru_cache
   :members:
   :inherited-members:
   :show-inheritance:


************
Path Library
************

arc
===

.. autofunction:: phidl.path.arc


euler
=====

.. autofunction:: phidl.path.euler


smooth
======

.. autofunction:: phidl.path.smooth


straight
========

.. autofunction:: phidl.path.straight


spiral
======

.. autofunction:: phidl.path.spiral


***************
Routing Library
***************

route_sharp
===========

.. autofunction:: phidl.routing.route_sharp


route_smooth
============

.. autofunction:: phidl.routing.route_smooth

route_quad
==========

.. autofunction:: phidl.routing.route_quad

route_xy
========

.. autofunction:: phidl.routing.route_xy

path_manhattan
==============

.. autofunction:: phidl.routing.path_manhattan

path_straight
=============

.. autofunction:: phidl.routing.path_straight

path_L
======

.. autofunction:: phidl.routing.path_L

path_U
======

.. autofunction:: phidl.routing.path_U

path_J
======

.. autofunction:: phidl.routing.path_J

path_C
======

.. autofunction:: phidl.routing.path_C

path_V
======

.. autofunction:: phidl.routing.path_V

path_Z
======

.. autofunction:: phidl.routing.path_Z



**************
Layout Classes
**************

CellArray
=========

.. autoclass:: phidl.device_layout.CellArray
   :members:
   :inherited-members:
   :show-inheritance:


CrossSection
============

.. autoclass:: phidl.device_layout.CrossSection
   :members:
   :inherited-members:
   :show-inheritance:


Device
======

.. autoclass:: phidl.device_layout.Device
   :members:
   :inherited-members:
   :show-inheritance:


DeviceReference
===============

.. autoclass:: phidl.device_layout.DeviceReference
   :members:
   :inherited-members:
   :show-inheritance:


Group
=====

.. autoclass:: phidl.device_layout.Group
   :members:
   :inherited-members:
   :show-inheritance:


Label
=====

.. autoclass:: phidl.device_layout.Label
   :members:
   :inherited-members:
   :show-inheritance:


Layer
=====

.. autoclass:: phidl.device_layout.Layer
   :members:
   :inherited-members:
   :show-inheritance:


LayerSet
========

.. autoclass:: phidl.device_layout.LayerSet
   :members:
   :inherited-members:
   :show-inheritance:


Path
====

.. autoclass:: phidl.device_layout.Path
   :members:
   :inherited-members:
   :show-inheritance:


Polygon
=======

.. autoclass:: phidl.device_layout.Polygon
   :members:
   :inherited-members:
   :show-inheritance:


Port
====

.. autoclass:: phidl.device_layout.Port
   :members:
   :inherited-members:
   :show-inheritance:


make_device
===========

.. autofunction:: phidl.device_layout.make_device


reset
=====

.. autofunction:: phidl.device_layout.reset
