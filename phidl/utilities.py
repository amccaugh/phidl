import operator

from phidl.quickplotter import _get_layerprop


def write_lyp(filename, layerset):
    """Creates a KLayout .lyp Layer Properties file from a set of
    PHIDL layers"""
    stipple_default = ["I2", "I5", "I9", "I17", "I19", "I22", "I33", "I38"]
    stipple_count = 0

    if filename[-4:] != ".lyp":
        filename = filename + ".lyp"

    # Opening file for writing
    with open("%s" % filename, "w+") as f:

        # Writing header string
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')

        # Writing layer properties opener
        f.write("<layer-properties>\n")

        unsorted_layers = layerset._layers.values()
        sorted_layers = sorted(
            unsorted_layers, key=operator.attrgetter("gds_layer", "gds_datatype")
        )

        for layer in sorted_layers:
            # Extracting information from dictionary layer by layer
            gds_layer = layer.gds_layer
            gds_datatype = layer.gds_datatype
            color = layer.color

            name = "{}/{} - ".format(str(gds_layer), str(gds_datatype)) + layer.name
            if layer.description is not None:
                name = name + " - (" + layer.description + ")"

            # Setting stipple or 'dither'
            dither = layer.dither
            if dither is None:
                dither = stipple_default[stipple_count]
                stipple_count = (stipple_count + 1) % len(stipple_default)
            elif dither[0] != "I":
                raise ValueError("""Stipple must begin with an I""")
            elif int(dither[1 : len(dither)]) < 0:
                raise ValueError("""Stipple index cannot be less than 0""")
            elif int(dither[1 : len(dither)]) > 46:
                raise ValueError("""Stipple index cannot be greater than 46""")
            else:
                pass

            # Writing properties header for speciic layer
            f.write(" <properties>\n")
            # Writing line to specify frame colour
            f.write("  <frame-color>%s</frame-color>\n" % color)
            # Writing line to specify fill colour
            f.write("  <fill-color>%s</fill-color>\n" % color)
            #            # Writing line to specify brightness (value between [-255, 255])
            #            f.write('  <frame-brightness>%s</frame-brightness>\n  <fill-brightness>%s</fill-brightness>\n' % (int(brightness), int(brightness)))
            frame_brightness = -25
            f.write(
                "  <frame-brightness>%s</frame-brightness>\n" % (int(frame_brightness))
            )
            # Writing line to specify dither pattern
            f.write("  <dither-pattern>%s</dither-pattern>\n" % dither)
            # Writing lines to specify line style
            f.write("  <line-style/>\n")
            # Writing line to specify validity
            f.write("  <valid>true</valid>\n")
            # Writing line to specify visibility
            f.write("  <visible>true</visible>\n")
            # Writing line to specify transparency
            f.write("  <transparent>false</transparent>\n")
            # Writing line to specify width
            f.write("  <width/>\n")
            # Writing line to specity markedness
            f.write("  <marked>false</marked>\n")
            # Writing line to specify xfill
            f.write("  <xfill>false</xfill>\n")
            # Writing line to specify animation
            f.write("  <animation>0</animation>\n")
            # Writing line to specify layer name
            f.write("  <name>%s</name>\n" % name)
            # Writing line to specify source
            f.write(
                "  <source>{}/{}@1</source>\n".format(str(gds_layer), str(gds_datatype))
            )
            # Writing properties closer for specific layer
            f.write(" </properties>\n")

        # Writing layer properties trailer
        f.write("</layer-properties>\n")


def load_lyp(filename):
    """Creates a LayerSet object from a lyp file that is XML"""
    try:
        import xmltodict
    except Exception:
        raise ImportError(
            """This function is in development, and currently requires
            the module "xmltodict" to operate.  Please retry after installing xmltodict
            $ pip install xmltodict """
        )

    from phidl.device_layout import LayerSet

    if filename[-4:] != ".lyp":
        filename = filename + ".lyp"
    with open(filename, "r") as fx:
        lyp_dict = xmltodict.parse(fx.read(), process_namespaces=True)
    # lyp files have a top level that just has one dict: layer-properties
    # That has multiple children 'properties', each for a layer. So it gives a list
    lyp_list = lyp_dict["layer-properties"]["properties"]
    if not isinstance(lyp_list, list):
        lyp_list = [lyp_list]

    lys = LayerSet()

    def add_entry(entry, lys):
        """Entry is a dict of one element of 'properties'.
        No return value. It adds it to the lys variable directly
        """
        layerInfo = entry["source"].split("@")[0]
        phidl_LayerArgs = dict()
        phidl_LayerArgs["gds_layer"] = int(layerInfo.split("/")[0])
        phidl_LayerArgs["gds_datatype"] = int(layerInfo.split("/")[1])
        phidl_LayerArgs["color"] = entry["fill-color"]
        phidl_LayerArgs["dither"] = entry["dither-pattern"]
        # These functions are customizable. See below
        phidl_LayerArgs["name"] = name2shortName(entry["name"])
        phidl_LayerArgs["description"] = name2description(entry["name"])
        lys.add_layer(**phidl_LayerArgs)
        return lys

    for entry in lyp_list:
        try:
            group_members = entry["group-members"]
        except KeyError:  # it is a real layer
            add_entry(entry, lys)
        else:  # it is a group of other entries
            if not isinstance(group_members, list):
                group_members = [group_members]
            for member in group_members:
                add_entry(member, lys)
    return lys


def name2shortName(name_str):
    """Maps the name entry of the lyp element to a name of the phidl layer,
    i.e. the dictionary key used to access it.
    Default format of the lyp name is
        layer/datatype - phidl_key - description
        or
        phidl_key - description

    Reassign for different layer naming conventions with::

        phidl.utilities.name2shortName = someOtherFunction(string)
    """
    if name_str is None:
        raise IOError("This layer has no name")
    components = name_str.split(" - ")
    if len(components) > 1:
        short_name = components[1]
    else:
        short_name = components[0]
    return short_name


def name2description(name_str):
    """Gets the description of the layer contained in the lyp name field.
    It is not strictly necessary to have a description. If none there, it returns ''.

    Default format of the lyp name is
        layer/datatype - phidl_key - description
        or
        phidl_key - description

    Reassign for different layer naming conventions with::

        phidl.utilities.name2description = someOtherFunction(string)
    """
    if name_str is None:
        raise IOError("This layer has no name")
    components = name_str.split(" - ")
    description = ""
    if len(components) > 2:
        description = components[2][1:-1]
    return description


def write_svg(D, filename, scale=1):
    xsize, ysize = D.size
    dcx, dcy = D.center
    dx, dy = dcx - xsize / 2, dcy - ysize / 2
    group_num = 1
    if filename[-4:] != ".svg":
        filename += ".svg"
    with open(filename, "w+") as f:
        f.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n')
        f.write(
            (
                '<svg\n   width="%0.6f" \n   height="%0.6f"\n'
                '   version="1.1"\n'
                '   xmlns:svg="http://www.w3.org/2000/svg"\n'
                '   xmlns="http://www.w3.org/2000/svg">\n'
            )
            % (xsize * scale, ysize * scale)
        )

        all_polygons = D.get_polygons(by_spec=True)
        for layer, polygons in all_polygons.items():
            #    color = '#800000'
            color = _get_layerprop(layer=layer[0], datatype=layer[1])["color"]
            f.write('  <g id="layer%03i_datatype%03i">\n' % (layer[0], layer[1]))
            group_num += 1

            for polygon in polygons:
                poly_str = '    <path style="fill:%s"\n          d="' % color
                n = 0
                for p in polygon:
                    if n == 0:
                        poly_str += "M "
                    else:
                        poly_str += "L "
                    poly_str += "{:0.6f} {:0.6f} ".format(
                        (p[0] - dx) * scale,
                        (-(p[1] - dy) + ysize) * scale,
                    )
                    n += 1
                poly_str += 'Z"/>\n'
                f.write(poly_str)
            f.write("  </g>\n")

        f.write("</svg>\n")
    return filename
