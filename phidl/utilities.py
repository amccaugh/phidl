import operator
from collections import defaultdict
from xml.etree import cElementTree


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


def read_lyp(filename):
    from .device_layout import LayerSet
    if filename[-4:] != '.lyp': filename = filename + '.lyp'
    with open(filename, 'r') as fx:
        lyp_list = xml_to_dict(fx.read())['layer-properties']['properties']
    lys = LayerSet()
    for entry in lyp_list:
        phidl_LayerArgs = dict()
        layerInfo = entry['source'].split('@')[0]
        phidl_LayerArgs['gds_layer'] = int(layerInfo.split('/')[0])
        phidl_LayerArgs['gds_datatype'] = int(layerInfo.split('/')[1])
        phidl_LayerArgs['color'] = entry['fill-color']
        name_components = entry['name'].split(' - ')
        if len(name_components) == 1:
            phidl_LayerArgs['name'] = name_components[0].strip()
        else:
            phidl_LayerArgs['name'] = name_components[1].strip()
        if len(name_components) == 3:
            phidl_LayerArgs['description'] = name_components[2].strip()[1:-1]
        lys.add_layer(**phidl_LayerArgs)
    return lys


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


def etree_to_dict(t):
    ''' Used recursively '''
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v) for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d


def xml_to_dict(t):
    try:
        e = cElementTree.XML(t)
    except:
        raise IOError("Error in the XML string.")
    return etree_to_dict(e)
