"""
Support for font rendering in GDS files.
"""

from __future__ import division

import six
import numpy as np
from matplotlib import font_manager
import gdspy
from .device_layout import Device


_cached_fonts = {}

try:
    import freetype
except ImportError as imp_err:
    raise ImportError("PHIDL requires freetype to use real fonts. "
                      "Either use the default DEPLOF font or install the freetype package:"
                      "\n\n $ pip install freetype-py"
                      "\n\n (Note: Windows users may have to find and replace the 'libfreetype.dll' "
                      "file in their Python package directory /freetype/ with the correct one"
                      "from here: https://github.com/ubawurinna/freetype-windows-binaries"
                      " -- be sure to rename 'freetype.dll' to 'libfreetype.dll') ")

def _get_font_by_file(file):
    """
    Load a given font file.

    Args:
        file [str, BinaryIO]: Load a font face from a given file
    """
    # Cache opened fonts
    if file in _cached_fonts:
        return _cached_fonts[file]

    font_renderer = freetype.Face(file)
    font_renderer.set_char_size(32*64) # 32pt size
    _cached_fonts[file] = font_renderer
    return font_renderer

def _get_font_by_name(name):
    """
    Try and load a system font by name.

    Args:
        name [str]: Load a system font
    """
    try:
        font_file = font_manager.findfont(name, fallback_to_default=False)
    except:
        raise ValueError(('[PHIDL] Failed to find font: "%s". ' +
        'Try specifying the exact (full) path to the .ttf or .otf file. ' +
        'Otherwise, it might be resolved by rebuilding the matplotlib font cache') % (name))
    return _get_font_by_file(font_file)

def _get_glyph(font, letter):
    """
    Get a block reference to the given letter
    """
    if not isinstance(letter, six.string_types) and len(letter) == 1:
        raise TypeError("Letter must be a string of length 1. Got: (%s)." % (letter))

    if not isinstance(font, freetype.Face):
        raise TypeError(("font %r must be a freetype font face. "
                        "Load a font using _get_font_by_name first.") % (font))

    if getattr(font, "gds_glyphs", None) is None:
        font.gds_glyphs = {}

    if letter in font.gds_glyphs:
        return font.gds_glyphs[letter]

    # Get the font name
    font_name = font.get_sfnt_name(freetype.TT_NAME_IDS['TT_NAME_ID_PS_NAME']).string.decode()
    if not font_name:
        # If there is no postscript name, use the family name
        font_name = font.family_name.replace(" ", "_")

    block_name = "*char_%s_0x%2X" % (font_name, ord(letter))

    # Load control points from font file
    font.load_char(letter, freetype.FT_LOAD_FLAGS['FT_LOAD_NO_BITMAP'])
    glyph = font.glyph
    outline = glyph.outline
    points = np.array(outline.points, dtype=float)/font.size.ascender
    tags = outline.tags

    # Add polylines
    start, end = 0, -1
    polylines = []
    for contour in outline.contours:
        start = end + 1
        end = contour

        # Build up the letter as a curve
        cpoint = start
        curve = gdspy.Curve(*points[cpoint], tolerance=0.001)
        while cpoint <= end:
            # Figure out what sort of point we are looking at
            if tags[cpoint] & 1:
                # We are at an on-curve control point. The next point may be
                # another on-curve point, in which case we create a straight
                # line interpolation, or it may be a quadratic or cubic
                # bezier curve. But first we check if we are at the end of the array
                if cpoint == end:
                    ntag = tags[start]
                    npoint = points[start]
                else:
                    ntag = tags[cpoint+1]
                    npoint = points[cpoint+1]

                # Then add the control points
                if ntag & 1:
                    curve.L(*npoint)
                    cpoint += 1
                elif ntag & 2:
                    # We are at a cubic bezier curve point
                    if cpoint+3 <= end:
                        curve.C(*points[cpoint+1:cpoint+4].flatten())
                    elif cpoint+2 <= end:
                        plist = list(points[cpoint+1:cpoint+3].flatten())
                        plist.extend(points[start])
                        curve.C(*plist)
                    else:
                        raise ValueError("Missing bezier control points. We require at least"
                                         " two control points to get a cubic curve.")
                    cpoint += 3
                else:
                    # Otherwise we're at a quadratic bezier curve point
                    if cpoint + 2 > end:
                        cpoint_2 = start
                        end_tag = tags[start]
                    else:
                        cpoint_2 = cpoint + 2
                        end_tag = tags[cpoint_2]
                    p1 = points[cpoint+1]
                    p2 = points[cpoint_2]

                    # Check if we are at a sequential control point. In that case,
                    # p2 is actually the midpoint of p1 and p2.
                    if end_tag & 1 == 0:
                        p2 = (p1 + p2)/2

                    # Add the curve
                    curve.Q(p1[0], p1[1], p2[0], p2[1])
                    cpoint += 2
            else:
                # We are looking at a control point
                if not tags[cpoint] & 2:
                    # We are at a quadratic sequential control point.
                    # Check if we're at the end of the segment
                    if cpoint == end:
                        cpoint_1 = start
                        end_tag = tags[start]
                    else:
                        cpoint_1 = cpoint + 1
                        end_tag = tags[cpoint_1]

                    # If we are at the beginning, this is a special case,
                    # we need to reset the starting position
                    if cpoint == start:
                        p0 =points[end]
                        p1 =points[cpoint]
                        p2 =points[cpoint_1]
                        if tags[end] & 1 == 0:
                            # If the last point is also a control point, then the end is actually
                            # halfway between here and the last point
                            p0 = (p0 + p1)/2
                        # And reset the starting position of the spline
                        curve = gdspy.Curve(*p0, tolerance=0.001)
                    else:
                        # The first control point is at the midpoint of this control point and the
                        # previous control point
                        p0 = points[cpoint-1]
                        p1 = points[cpoint]
                        p2 = points[cpoint_1]
                        p0 = (p0 + p1)/2

                    # Check if we are at a sequential control point again
                    if end_tag & 1 == 0:
                        p2 = (p1 + p2)/2

                    # And add the segment
                    curve.Q(p1[0], p1[1], p2[0], p2[1])
                    cpoint += 1
                else:
                    raise ValueError("Sequential control points not valid for cubic splines.")
        polylines.append(gdspy.Polygon(curve.get_points()))

    # Construct the device
    device = Device(block_name)
    if polylines:
        letter_polyline = polylines[0]
        for polyline in polylines[1:]:
            letter_polyline = gdspy.boolean(letter_polyline, polyline, "xor")
        device.add_polygon(letter_polyline)

    # Cache the return value and return it
    font.gds_glyphs[letter] = (device, glyph.advance.x/font.size.ascender)
    return font.gds_glyphs[letter]
