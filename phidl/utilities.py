import operator
from phidl.quickplotter import _get_layerprop

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


def write_svg(D, filename):
    xsize, ysize = D.size
    dcx, dcy = D.center
    dx, dy = dcx-xsize/2, dcy-ysize/2
    group_num = 1
    if filename[-4:] != '.svg':  filename += '.svg'
    with open(filename, 'w+') as f:
        f.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
        f.write('<svg width="%0.6f" height="%0.6f">\n' % (xsize, ysize))
        
        all_polygons = D.get_polygons(by_spec = True)
        for layer, polygons in all_polygons.items():
        #    color = '#800000'
            color = _get_layerprop(layer = layer[0] , datatype = layer[1])['color']
            f.write('  <g id="layer%03i_datatype%03i">\n' % (layer[0], layer[1]))
            group_num += 1
            
            for polygon in polygons:
                poly_str = '    <path style="fill:%s"\n          d="' % color
                n = 0
                for p in polygon:
                    if n == 0: poly_str+= 'M '
                    else:      poly_str+= 'L '
                    poly_str += '%0.6f %0.6f '  % (p[0]-dx,-(p[1]-dy)+ysize)
                    n += 1
                poly_str+= 'Z"/>\n'
                f.write(poly_str)
            f.write('  </g>\n')
        
        f.write('</svg>\n')
    return filename