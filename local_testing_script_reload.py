# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%  Add the local PHIDL package/module/folder path to the beginning of the Python PATH
local_phidl_path = r'C:\Users\amcc\Documents\GitHub\phidl'
import sys
sys.path.insert(0, local_phidl_path)

#%% Import as usual

from phidl import Device, Layer, make_device as md, quickplot2 as qp
import phidl.geometry as pg, phidl.routing as pr
import numpy as np
from phidl.device_layout import DeviceReference

import phidl
print(phidl.__version__)

#%% Run this whenever you change files on the disk
import importlib
importlib.reload(phidl)
importlib.reload(phidl.device_layout)
importlib.reload(phidl.geometry)
importlib.reload(phidl.routing)
Device = phidl.device_layout.Device
DeviceReference = phidl.device_layout.DeviceReference

print(phidl.__version__)

#%%
D = Device()
D = pg.import_gds(r'C:\Users\amcc\Documents\GitHub\die-designs\gds-files\SE005 SNSPD Integrator.gds', flatten = False)
p = D.add_polygon([[100,200,300,400],[500,600,900,1000]], layer = 5)
label = D.label('hello kitty2')

