# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:18:40 2017

@author: amcc
"""

# TODO:
# PHIDL Allow toggling of layers with 1-9 keypresses
# PHIDL alias font should be slightly transparent
# PHIDL Show annotations
# PHIDL add adjustable depth of aliases by doing shift-F3


from __future__ import division, print_function, absolute_import
import numpy as np
import sys

import phidl
from phidl.device_layout import Device, DeviceReference, Port, Layer

from PyQt5 import QtCore, QtGui, QtWidgets, QtOpenGL
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QApplication, QGraphicsEllipseItem, QGraphicsItem, QRubberBand, QGraphicsLineItem, QMainWindow
from PyQt5.QtCore import Qt, QPoint, QPointF, QRectF, QRect, QSize,  QCoreApplication, QLineF
from PyQt5.QtGui import QColor, QPolygonF, QPen


class ViewerWindow(QMainWindow):
    def __init__(self):
        super(ViewerWindow,self).__init__()
        self.viewer = Viewer()
        self.setCentralWidget(self.viewer)
        self.show()
    

class Viewer(QGraphicsView):
    def __init__(self):
        QGraphicsView.__init__(self)

        self.setGeometry(QRect(100, 100, 800, 600))
        self.setWindowTitle("PIHDL Graphics Window");
        
        # Create a QGraphicsScene which this view looks at
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(QRectF())
        self.setScene(self.scene)
        
        # Customize QGraphicsView
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setInteractive(False)
        self.scale(1,-1) # Flips around the Y axis
         # Use OpenGL http://ralsina.me/stories/BBS53.html
#        self.setViewport(QtOpenGL.QGLWidget())
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.pen = QPen(QtCore.Qt.black, 0)
        self.portpen = QPen(QtCore.Qt.red, 3)
        self.portpen.setCosmetic(True) # Makes constant width
        self.portfont = QtGui.QFont('Arial', pointSize = 14)
        self.portfontcolor = QtCore.Qt.red
        self.subportpen = QPen(QtCore.Qt.darkGreen, 3)
        self.subportpen.setCosmetic(True) # Makes constant width
        self.subportfont = QtGui.QFont('Arial', pointSize = 14)
        self.subportfontcolor = QtCore.Qt.darkGreen
        
        # Tracking ports

        # Various status variables
        self._mousePressed = None
        self._rb_origin = QPoint()
        self.zoom_factor_total = 1
        
        # Grid variables
        self.gridpen = QPen(QtCore.Qt.black, 0)
        self.gridpen.setStyle(QtCore.Qt.DotLine)
        self.gridpen.setDashPattern([1,4])
        self.gridpen.setColor(QtGui.QColor(0, 0, 0, 125))
#        self.gridpen = QPen(QtCore.Qt.black, 1)
#        self.gridpen.setCosmetic(True) # Makes constant width
#        self.gridlinesx = [self.scene.addLine(-10,-10,10,10, self.gridpen) for n in range(100)]
#        self.gridlinesy = [self.scene.addLine(-10,-10,10,10, self.gridpen) for n in range(100)]
        

        self.initialize()
            
    def itemsBoundingRect_nogrid(self):
        self.remove_grid()
        r = self.scene.itemsBoundingRect()
        self.create_grid()
        return r
    
    def add_polygons(self, polygons, color = '#A8F22A', alpha = 1):
        qcolor = QColor()
        qcolor.setNamedColor(color)
        qcolor.setAlphaF(alpha)
        for points in polygons:
            qpoly = QPolygonF( [QPointF(p[0], p[1]) for p in points] )
            scene_poly = self.scene.addPolygon(qpoly)
            scene_poly.setBrush(qcolor)
            scene_poly.setPen(self.pen)
        
        sr = self.itemsBoundingRect_nogrid()
        ymax = sr.top()
        xmin = sr.left()
        width = sr.width()
        height = sr.height()
        self.scene.setSceneRect(QRectF(xmin-2*width, ymax-2*height, width*5, height*5))
        
    def reset_view(self):
        self.fitInView(self.itemsBoundingRect_nogrid(), Qt.KeepAspectRatio)
        self.update_grid()
        
    def add_port(self, port, is_subport = False):
        if (port.width is None) or (port.width == 0):
            x,y = port.midpoint
            cs = 1 # cross size
            pn = QPointF(x, y+cs)
            ps = QPointF(x, y-cs)
            pe = QPointF(x+cs, y)
            pw = QPointF(x-cs, y)
            qline1 = self.scene.addLine(QLineF(pn, ps))
            qline2 = self.scene.addLine(QLineF(pw, pe))
            port_shapes = [qline1,qline2]
        else:
            point1, point2 = port.endpoints
            point1 = QPointF(point1[0], point1[1])
            point2 = QPointF(point2[0], point2[1])
            qline = self.scene.addLine(QLineF(point1, point2))
            arrow_points = np.array([[0,0],[10,0],[6,4],[6,2],[0,2]])/(40)*port.width
            arrow_qpoly = QPolygonF( [QPointF(p[0], p[1]) for p in arrow_points] )
            port_scene_poly = self.scene.addPolygon(arrow_qpoly)
            port_scene_poly.setRotation(port.orientation)
            port_scene_poly.moveBy(port.midpoint[0], port.midpoint[1])
            port_shapes = [qline,port_scene_poly]
        qtext = self.scene.addText(str(port.name), self.portfont)
        port_items = port_shapes + [qtext]
        rad = port.orientation*np.pi/180
        x,y = port.endpoints[0]*1/4 +  port.endpoints[1]*3/4 + np.array([np.cos(rad), np.sin(rad)])*port.width/8
#        x,y = port.midpoint[0], port.midpoint[1]
#        x,y  = x - qtext.boundingRect().width()/2, y - qtext.boundingRect().height()/2
        qtext.setPos(QPointF(x,y))
        qtext.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        
        if not is_subport:
            [shape.setPen(self.portpen) for shape in port_shapes]
            qtext.setDefaultTextColor(self.portfontcolor)
            self.portitems += port_items
        else:
            [shape.setPen(self.subportpen) for shape in port_shapes]
            qtext.setDefaultTextColor(self.subportfontcolor)
            self.subportitems += port_items
#        self.portlabels.append(qtext)
        
    def add_aliases(self, aliases):
        for name, ref in aliases.items():
            qtext = self.scene.addText(str(name), self.portfont)
            x,y = ref.center
            qtext.setPos(QPointF(x,y))
            qtext.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            self.aliasitems += [qtext]
            
#        x,y = port.midpoint[0], port.midpoint[1]
#        x,y  = x - qtext.boundingRect().width()/2, y - qtext.boundingRect().height()/2

    def set_port_visibility(self, visible = True):
        for item in self.portitems:
            item.setVisible(visible)
        self.ports_visible = visible

             
    def set_subport_visibility(self, visible = True):
        for item in self.subportitems:
            item.setVisible(visible)
        self.subports_visible = visible
                
    def set_alias_visibility(self, visible = True):
        for item in self.aliasitems:
            item.setVisible(visible)
        self.aliases_visible = visible
                
                
            
    def initialize(self):
        self.scene.clear()
        self.polygons = {}
        self.portitems = []
        self.subportitems = []
        self.aliasitems = []
        self.aliases_visible = True
        self.ports_visible = True
        self.subports_visible = True
        
        self.create_grid()        
        self.update_grid()
        
#==============================================================================
#   Grid creation
#==============================================================================
    def update_grid(self):
        grid_pixels = 50
        grid_snaps = [1,2,4]
        
         # Number of pixels in the viewer 
        view_width, view_height = self.rect().width(), self.rect().height()
         # Rectangle of viewport in terms of scene coordinates
        r = self.mapToScene(self.rect()).boundingRect()
        width, height = r.width(), r.height()
        xmin, ymin, xmax, ymax = r.x(), r.y(), r.x() + width, r.y() + height
                                    
        grid_size = grid_pixels*(width / view_width)
        exponent = np.floor( np.log10(grid_size) )
        digits  = round(grid_size / 10**(exponent), 2)
        digits_snapped = min(grid_snaps, key=lambda x:abs(x-digits))
        grid_size_snapped = digits_snapped * 10**(exponent)
        
        # Starting coordinates for gridlines
        x = round((xmin - 2*width )/grid_size_snapped) * grid_size_snapped
        y = round((ymin - 2*height)/grid_size_snapped) * grid_size_snapped
#        print('\n xmin = %s, xmax = %s, ymin = %s, ymax = %s' % (xmin, xmax, ymin, ymax))
#        print('Starting at x = %s' % x)
#        print('Starting at y = %s' % y)
        for gl in self.gridlinesx:
            gl.setLine(x, -1e10, x, 1e10)
            x += grid_size_snapped
        for gl in self.gridlinesy:
            gl.setLine(-1e10, y, 1e10, y)
            y += grid_size_snapped
            
    def create_grid(self):
        self.gridlinesx = [self.scene.addLine(-10,-10,10,10, self.gridpen) for n in range(200)]
        self.gridlinesy = [self.scene.addLine(-10,-10,10,10, self.gridpen) for n in range(200)]
        self.update_grid()
        
            
    def remove_grid(self):
        for gl in self.gridlinesx + self.gridlinesy:
            self.scene.removeItem(gl)
        self.gridlinesx == []
        self.gridlinesy == []
        
            
#==============================================================================
#  Mousewheel zoom, taken from http://stackoverflow.com/a/29026916
#==============================================================================
    def wheelEvent(self, event):
        # Zoom Factor
        zoom_percentage = 1.4
    
        # Set Anchors
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
    
        # Save the scene pos
        oldPos = self.mapToScene(event.pos())
    
        # Zoom
        mousewheel_rotation = event.angleDelta().y() # Typically = 120 on most mousewheels
        zoom_factor = zoom_percentage**(mousewheel_rotation/120)
        zoom_factor = np.clip(zoom_factor, 0.5, 2.0)

        
        # Check to make sure we're not overzoomed
        actual_rect = self.mapToScene(self.rect())
        bbox_size = actual_rect[0] - actual_rect[2]
        actual_width = abs(bbox_size.x())
        actual_height = abs(bbox_size.y())
        max_width = abs(self.scene.sceneRect().x()*3)
        max_height = abs(self.scene.sceneRect().y()*3)
        min_width = 1
        min_height = 1
        if ((actual_width > max_width) or (actual_height > max_height)) and (zoom_factor < 1):
            pass
        elif ((actual_width < min_width) or (actual_height < min_height)) and (zoom_factor > 1):
            pass
        else:
            self.zoom_view(zoom_factor)
    
        # Get the new position and move scene to old position
        newPos = self.mapToScene(event.pos())
        delta = newPos - oldPos
        self.translate(delta.x(), delta.y())
        
        
        self.update_grid()
        
        
    def zoom_view(self, zoom_factor):
        self.scale(zoom_factor, zoom_factor)
        self.zoom_factor_total *= zoom_factor
        
        
    def mousePressEvent(self, event):
        #==============================================================================
        #  Zoom to rectangle, from
        #  https://wiki.python.org/moin/PyQt/Selecting%20a%20region%20of%20a%20widget
        #==============================================================================
        if event.button() == Qt.RightButton:
            self._mousePressed = Qt.RightButton
            self._rb_origin = QPoint(event.pos())
            self.rubberBand.setGeometry(QRect(self._rb_origin, QSize()))
            self.rubberBand.show()
         #==============================================================================
        # Mouse panning, taken from
        # http://stackoverflow.com/a/15043279
        #==============================================================================
        elif event.button() == Qt.MidButton:
            self._mousePressed = Qt.MidButton
            self._mousePressedPos = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            self._dragPos = event.pos()


    def mouseMoveEvent(self, event):
        if not self._rb_origin.isNull() and self._mousePressed == Qt.RightButton:
            self.rubberBand.setGeometry(QRect(self._rb_origin, event.pos()).normalized())
                
        if self._mousePressed == Qt.MidButton:
            newPos = event.pos()
            diff = newPos - self._dragPos
            self._dragPos = newPos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - diff.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - diff.y())
#            event.accept()
            

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton:
            self.rubberBand.hide()
            rb_rect = QRect(self._rb_origin, event.pos())
            rb_center = rb_rect.center()
            rb_size = rb_rect.size()
            
            if abs(rb_size.width()) > 3 and abs(rb_size.height()) > 3:
                viewport_size = self.viewport().geometry().size()
                
                zoom_factor_x = abs(viewport_size.width() / rb_size.width())
                zoom_factor_y = abs(viewport_size.height() / rb_size.height())
                
                new_center = self.mapToScene(rb_center)
                
                zoom_factor = min(zoom_factor_x, zoom_factor_y)
                self.zoom_view(zoom_factor)
                self.centerOn(new_center)
                
            self.update_grid()
    
        if event.button() == Qt.MidButton:
            self.setCursor(Qt.ArrowCursor)
            self._mousePressed = None
            self.update_grid()
            
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.reset_view()
                
        if event.key() == Qt.Key_F1:
            self.set_alias_visibility(not self.aliases_visible)
                
        if event.key() == Qt.Key_F2:
            self.set_port_visibility(not self.ports_visible)
                
        if event.key() == Qt.Key_F3:
            self.set_subport_visibility(not self.subports_visible)


def quickplot2(item_list, *args, **kwargs):
    global app
    if QCoreApplication.instance() is None:
        app = QApplication(sys.argv)
    if 'viewer_window' not in globals():
        global viewer_window
        viewer_window = ViewerWindow()
    viewer = viewer_window.viewer
    viewer.initialize()
    if type(item_list) not in (list, tuple):
        item_list = [item_list]
    for element in item_list:
        if isinstance(element, (phidl.device_layout.Device, phidl.device_layout.DeviceReference)):
            polygons_spec = element.get_polygons(by_spec=True, depth=None)
            for key in sorted(polygons_spec):
                polygons = polygons_spec[key]
                layerprop = _get_layerprop(layer = key[0], datatype = key[1])
                viewer.add_polygons(polygons, color = layerprop['color'], alpha = layerprop['alpha'])
            for name, port in element.ports.items():
                viewer.add_port(port)
            if isinstance(element, phidl.device_layout.Device):
                for ref in element.references:
                    for name, port in ref.ports.items():
                        viewer.add_port(port, is_subport = True)
                viewer.add_aliases(element.aliases)
        elif isinstance(element, (phidl.device_layout.Polygon)):
                layerprop = _get_layerprop(layer = element.layer, datatype = element.datatype)
                viewer.add_polygons([element.points], color = layerprop['color'], alpha = layerprop['alpha'])
    viewer.reset_view()
    viewer_window.setVisible(True)
    viewer_window.show()
    viewer_window.raise_()


def _get_layerprop(layer, datatype):
    # Colors generated from here: http://phrogz.net/css/distinct-colors.html
    layer_colors = ['#3dcc5c', '#2b0fff', '#cc3d3d', '#e5dd45', '#7b3dcc',
    '#cc860c', '#73ff0f', '#2dccb4', '#ff0fa3', '#0ec2e6', '#3d87cc', '#e5520e']
                     
    l = Layer.layer_dict.get((layer, datatype))
    if l is not None:
        color = l.color
        alpha = l.alpha
        if color is None:
            color = layer_colors[np.mod(layer, len(layer_colors))]
    else:
        color = layer_colors[np.mod(layer, len(layer_colors))]
        alpha = 0.7
    return {'color':color, 'alpha':alpha}





# quickplot2(pg.snspd())
