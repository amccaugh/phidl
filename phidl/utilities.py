# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 11:34:09 2018

@author: Dylan Oh
"""

#%% Creating Layer object

import webcolors
import numpy as np

class Layer(object):
    """Python class for layer dictionary creation,
    written by Adam McCaughan"""
    layer_dict = {}

    def __init__(self, name = 'goldpads', gds_layer = 0, gds_datatype = 0,
                 description = 'Gold pads liftoff', inverted = False,
                 color = None, alpha = None, dither = None):
        self.name = name
        self.gds_layer = gds_layer
        self.gds_datatype = gds_datatype
        self.description = description
        self.alpha = alpha
        self.dither = dither
        
        try:
            if color is None: # not specified
                self.color = None
            elif np.size(color) == 3: # in format (0.5, 0.5, 0.5)
                self.color = webcolors.rgb_to_hex(np.array( np.array(color)*255, dtype = int))
            elif color[0] == '#': # in format #1d2e3f
                self.color = webcolors.hex_to_rgb(color)
                self.color = webcolors.rgb_to_hex(self.color)
            else: # in named format 'gold'
                self.color = webcolors.name_to_hex(color)
        except:
            raise ValueError("""[PHIDL] Layer() color must be specified as a
            0-1 RGB triplet, (e.g. [0.5, 0.1, 0.9]), an HTML hex  color 
            (e.g. #a31df4), or a CSS3 color name (e.g. 'gold' or
            see http://www.w3schools.com/colors/colors_names.asp )
            """)
            
        Layer.layer_dict[(gds_layer, gds_datatype)] = self

    def __repr__(self):
        return ('Layer (name %s, GDS layer %s, GDS datatype %s, description %s, color %s)' % \
                (self.name, self.gds_layer, self.gds_datatype, self.description, self.color))
                         
        
#%% Just some random sample layers; change params as needed
        
layers = {
        'nb_gnd'  : Layer(gds_layer = 1, gds_datatype = 0, description = 'Ground layer', color = 'grey', dither = 'I0'),
        'wsi_snspd'  : Layer(gds_layer = 2, gds_datatype = 0, description = 'WSi', color = 'lightgreen', dither = 'I2'),
        'wsi_snspd_gnd'  : Layer(gds_layer = 2, gds_datatype = 1, description = 'WSi', color = 'lightgreen', dither = 'I5'),
        'myresistorlayer' : Layer(gds_layer = 3, gds_datatype = 0, description = 'resistor and pads', color = 'red', dither = 'I9'),
        'snspd_pads' : Layer(gds_layer = 3, gds_datatype = 1, description = 'SNSPD pads', color = 'red', dither = 'I12'),
        'wsi_mw'  : Layer(gds_layer = 4, gds_datatype = 0, description = 'WSi', color = 'lightblue', dither = 'I14'),
        'wsi_mw_pads'  : Layer(gds_layer = 4, gds_datatype = 1, description = 'WSi', color = 'blue', dither = 'I17'),
        'via'  : Layer(gds_layer = 5, gds_datatype = 0, description = 'via', color = 'red', alpha = 1, dither = 'I22'),
#        'avoid' : Layer(gds_layer = 200, gds_datatype = 0, description = 'Avoid', color = 'pink', alpha = 0.1),
         }

layers1 = {
        'nb_gnd'  : Layer(gds_layer = 1, gds_datatype = 0, description = 'Ground layer', color = '#fa0687', dither = 'I46'),
        'wsi_snspd'  : Layer(gds_layer = 2, gds_datatype = 0, description = 'WSi', color = '#7d8bb6', dither = 'I30'),
        'wsi_snspd_gnd'  : Layer(gds_layer = 2, gds_datatype = 1, description = 'WSi', color = '#edf043', dither = 'I28'),
        'myresistorlayer' : Layer(gds_layer = 3, gds_datatype = 0, description = 'resistor and pads', color = '#95e1ab', dither = 'I29'),
        'snspd_pads' : Layer(gds_layer = 3, gds_datatype = 1, description = 'SNSPD pads', color = '#3d3804', dither = 'I22'),
        'wsi_mw'  : Layer(gds_layer = 4, gds_datatype = 0, description = 'WSi', color = '#36b3d8', dither = 'I23'),
        'wsi_mw_pads'  : Layer(gds_layer = 4, gds_datatype = 1, description = 'WSi', color = '#1f5fbb', dither = 'I24'),
        'via'  : Layer(gds_layer = 5, gds_datatype = 0, description = 'via', color = '#48a0b9', alpha = 1, dither = 'I16'),
#        'avoid' : Layer(gds_layer = 200, gds_datatype = 0, description = 'Avoid', color = 'pink', alpha = 0.1),        
         }


layers2 = {
        'nb_gnd'  : Layer(gds_layer = 1, gds_datatype = 0, description = 'Ground layer', color = '#abde04', dither = 'I17'),
        'wsi_snspd'  : Layer(gds_layer = 2, gds_datatype = 0, description = 'WSi', color = '#a783dd', dither = 'I19'),
        'wsi_snspd_gnd'  : Layer(gds_layer = 2, gds_datatype = 1, description = 'WSi', color = '#5336d1', dither = 'I20'),
        'myresistorlayer' : Layer(gds_layer = 3, gds_datatype = 0, description = 'resistor and pads', color = '#1294da', dither = 'I25'),
        'snspd_pads' : Layer(gds_layer = 3, gds_datatype = 1, description = 'SNSPD pads', color = '#1f5209', dither = 'I26'),
        'wsi_mw'  : Layer(gds_layer = 4, gds_datatype = 0, description = 'WSi', color = '#3cc0f0', dither = 'I34'),
        'wsi_mw_pads'  : Layer(gds_layer = 4, gds_datatype = 1, description = 'WSi', color = '#0911ab', dither = 'I37'),
        'via'  : Layer(gds_layer = 5, gds_datatype = 0, description = 'via', color = '#1af5a6', alpha = 1, dither = 'I42'),
#        'avoid' : Layer(gds_layer = 200, gds_datatype = 0, description = 'Avoid', color = 'pink', alpha = 0.1),        
         }
        
        
#%% Writing dictionary to a .lyp file

def write_lyp(name, layers):
    # Generating list of default values for stipple
    stipple_default = ['I0','I2','I5','I9','I17','I19','I22','I33','I38']
    stipple_count = 0

    # Opening file for writing
    f = open('%s' % name,'w+')

    # Writing header string
    f.write('<?xml version="1.0" encoding="utf-8"?>\n')

    # Writing layer properties opener
    f.write('<layer-properties>\n')

    for key, value in layers.items():  
        # Extracting information from dictionary layer by layer
        gds_layer = value.gds_layer
        gds_datatype = value.gds_datatype
        color = value.color
        name = '%s/%s - ' % (str(gds_layer), str(gds_datatype)) + key

        # Setting brightness
        alpha = value.alpha
        if alpha is None:
            alpha = 0
        elif alpha is float:
            raise ValueError("""alpha must be int type""")
        else:
            pass

        # Setting stipple or 'dither'
        dither = value.dither
        if dither is None:
            dither = stipple_default[stipple_count]
            stipple_count += 1
        elif dither[0] != 'I':
            raise TypeError("""Stipple must begin with an I""")
        elif int(dither[1:len(dither)]) < 0:
            raise ValueError("""Stipple index cannot be less than 0""")
        elif int(dither[1:len(dither)]) > 46:
            raise ValueError("""Stipple index cannot be greater than 46""")
        else:
            pass

        # Writing properties header for speciic layer
        f.write(' <properties>\n')
        # Writing line to specify frame colour
        f.write('  <frame-color>%s</frame-color>\n' % color)
        # Writing line to specify fill colour
        f.write('  <fill-color>%s</fill-color>\n' % color)
        # Writing line to specify brightness
        f.write('  <frame-brightness>%s</frame-brightness>\n  <fill-brightness>%s</fill-brightness>\n' % (str(alpha), str(alpha)))
        # Writing line to specify dither pattern
        f.write('  <dither-pattern>%s</dither-pattern>\n' % dither)
        # Writing lines to specify line style
        f.write('  <line-style/>\n')
        # Writing line to specify validity
        f.write('  <valid>true</valid>\n')
        # Writing line to specify visibility
        f.write('  <visible>true</visible>\n')
        # Writing line to specify transparency
        f.write('  <transparent>false</transparent>\n')
        # Writing line to specify width
        f.write('  <width/>\n')
        # Writing line to specity markedness
        f.write('  <marked>false</marked>\n')
        # Writing line to specify xfill
        f.write('  <xfill>false</xfill>\n')
        # Writing line to specify animation
        f.write('  <animation>0</animation>\n')
        # Writing line to specify layer name
        f.write('  <name>%s</name>\n' % name)
        # Writing line to specify source
        f.write('  <source>%s/%s@1</source>\n' % (str(gds_layer), str(gds_datatype)))
        # Writing properties closer for specific layer
        f.write(' </properties>\n')

    # Writing layer properties trailer
    f.write('</layer-properties>\n')
    f.close()
