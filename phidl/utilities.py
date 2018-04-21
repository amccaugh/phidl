import operator

def write_lyp(filename, layerset):
    """ Creates a KLayout .lyp Layer Properties file from a set of 
    PHIDL layers """
    stipple_default = ['I2','I5','I9','I17','I19','I22','I33','I38']
    stipple_count = 0

    if filename[-4:] != '.lyp': filename = filename + '.lyp'

    # Opening file for writing
    with open('%s' % filename,'w+') as f:
    
        # Writing header string
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
    
        # Writing layer properties opener
        f.write('<layer-properties>\n')
        
        unsorted_layers = layerset._layers.values()
        sorted_layers = sorted(unsorted_layers, key = operator.attrgetter('gds_layer', 'gds_datatype'))

        for layer in sorted_layers:  
            # Extracting information from dictionary layer by layer
            gds_layer = layer.gds_layer
            gds_datatype = layer.gds_datatype
            color = layer.color

            name = '%s/%s - ' % (str(gds_layer), str(gds_datatype)) + layer.name
            if layer.description is not None:
                name = name + ' - ('  + layer.description + ')'

            
            # Setting stipple or 'dither'
            dither = layer.dither
            if dither is None:
                dither = stipple_default[stipple_count]
                stipple_count = (stipple_count + 1) % len(stipple_default)
            elif dither[0] != 'I':
                raise ValueError("""Stipple must begin with an I""")
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
#            # Writing line to specify brightness (value between [-255, 255])
#            f.write('  <frame-brightness>%s</frame-brightness>\n  <fill-brightness>%s</fill-brightness>\n' % (int(brightness), int(brightness)))
            frame_brightness = -25
            f.write('  <frame-brightness>%s</frame-brightness>\n' % (int(frame_brightness)))
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


def in_ipynb():
    ''' Detects if running in an ipython-notebook frontend
    '''
    try:
        ipy_str = str(type(get_ipython()))
        if 'zmqshell' in ipy_str:
            return True
        if 'terminal' in ipy_str:
            return False
    except NameError:
        return False
