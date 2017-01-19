# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:18:40 2017

@author: amcc
"""

from __future__ import division, print_function, absolute_import
import numpy as np
import sys
#try:
from PyQt5 import QtCore, QtGui, QtWidgets, QtOpenGL
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QApplication, QGraphicsEllipseItem, QRubberBand
from PyQt5.QtCore import QPoint, QPointF, QRectF, QRect, QSize, Qt, QCoreApplication, QLineF
from PyQt5.QtGui import QColor, QPolygonF, QPen
#except:
#    #from qtpy import QtWidgets, QtCore, QtGui
#    from PyQt4 import QtCore, QtGui, QtOpenGL, QGraphicsView
#    from PyQt4.QtCore import QPoint, QRect, QSize, Qt


class Viewer(QGraphicsView):
    def __init__(self):
        QGraphicsView.__init__(self)

        self.setGeometry(QRect(100, 100, 600, 250))
        self.setWindowTitle("PIHDL Graphics Windfiow");
        
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
        self.portpen = QPen(QtCore.Qt.red, 2)
#        self.portpen.setCosmetic(True) # Makes constant width

        # Various status variables
        self._mousePressed = None
        self._rb_origin = QPoint()
        
        self.zoom_factor_total = 1
        
        

        for i in range(5):
            self.item = QGraphicsEllipseItem(i*75, 10, 60, 40)
            self.scene.addItem(self.item)
            
            
    
    def add_polygons(self, polygons, color = '#A8F22A', alpha = 1):
        qcolor = QColor()
        qcolor.setNamedColor(color)
        qcolor.setAlphaF(alpha)
        for points in polygons:
            qpoly = QPolygonF()
            for p in points:
                qpoly.append(QPointF(p[0], p[1]))
            scene_poly = self.scene.addPolygon(qpoly)
            scene_poly.setBrush(qcolor)
            scene_poly.setPen(self.pen)
        sr = self.scene.itemsBoundingRect()
        ymax = sr.top()
        xmin = sr.left()
        width = sr.width()
        height = sr.height()
        self.scene.setSceneRect(QRectF(xmin-2*width, ymax-2*height, width*5, height*5))
        
    def reset_view(self):
        self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        
    def add_port(self, port):
        point1, point2 = port.endpoints
        point1 = QPointF(point1[0], point1[1])
        point2 = QPointF(point2[0], point2[1])
        self.scene.addLine(QLineF(point1, point2), self.portpen)
    
            
    def clear(self):
        self.scene.clear()
        
            
#==============================================================================
#  Mousewheel zoom, taken from http://stackoverflow.com/a/29026916
#==============================================================================
    def wheelEvent(self, event):
        # Zoom Factor
        zoomInFactor = 1.4
        zoomOutFactor = 1 / zoomInFactor
        zoom_percentage = 1.4
    
        # Set Anchors
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
    
        # Save the scene pos
        oldPos = self.mapToScene(event.pos())
    
        # Zoom
        mousewheel_rotation = event.angleDelta().y() # Typically = 15 on most mousewheels
        zoomFactor = zoom_percentage **(mousewheel_rotation/15)
#        print([mousewheel_rotation, zoomFactor])
#        if event.angleDelta().y() > 0:
#            zoomFactor = zoomInFactor
#        else:
#            zoomFactor = zoomOutFactor

        
        # Check to make sure we're not overzoomed
        actual_rect = self.mapToScene(self.rect())
        bbox_size = actual_rect[0] - actual_rect[2]
        actual_width = abs(bbox_size.x())
        actual_height = abs(bbox_size.y())
        max_width = abs(self.scene.sceneRect().x()*3)
        max_height = abs(self.scene.sceneRect().y()*3)
        min_width = 10
        min_height = 10
        if ((actual_width > max_width) or (actual_height > max_height)) and (zoomFactor < 1):
            pass
        elif ((actual_width < min_width) or (actual_height < min_height)) and (zoomFactor > 1):
            pass
        else:
            self.zoom_view(np.clip(zoomFactor, 0.5, 2.0))
#            zoom_factor = min(max_width/actual_width, max_height/actual_height)
#            print(zoom_factor)
#            view.scale(zoom_factor, zoom_factor)
        
    
        # Get the new position
        newPos = self.mapToScene(event.pos())
    
        # Move scene to old position
        delta = newPos - oldPos
        self.translate(delta.x(), delta.y())
        
        # Make sure Z isn't too much or too little
        
        
        
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
    
        if event.button() == Qt.MidButton:
            self.setCursor(Qt.ArrowCursor)
            self._mousePressed = None
            

#if QCoreApplication.instance() is None:
#    app = QApplication(sys.argv) 
#viewer = Viewer()

def quickplot2(item):
    viewer.clear()
    if isinstance(item, (phidl.device_layout.Device, phidl.device_layout.DeviceReference)):
        polygons_spec = item.get_polygons(by_spec=True, depth=None)
        for key in sorted(polygons_spec):
            polygons = polygons_spec[key]
            layerprop = _get_layerprop(layer = key[0], datatype = key[1])
            viewer.add_polygons(polygons, color = layerprop['color'], alpha = layerprop['alpha'])



def _get_layerprop(layer, datatype):
    # Colors generated from here: http://phrogz.net/css/distinct-colors.html
    layer_colors = ['#3dcc5c', '#2b0fff', '#cc3d3d', '#e5dd45', '#7b3dcc',
    '#cc860c', '#73ff0f', '#2dccb4', '#ff0fa3', '#0ec2e6', '#3d87cc', '#e5520e']
                     
    l = Layer.layer_dict.get((layer, datatype))
    if l is not None:
        color = l.color
        alpha = l.alpha
    else:
        color = layer_colors[np.mod(layer, len(layer_colors))]
        alpha = 0.8
    return {'color':color, 'alpha':alpha}


    
    
polygon1 = np.array([[1,2],[3,6],[1,5]])*20
polygon2 = np.array([[1,1],[3,8],[3,5]])*20 + 80
polygon3 = np.array([[2,1],[3,9],[7,5]])*15 + 80
my_polygons = [polygon1, polygon2]
my_polygons2 = [polygon1, polygon2, polygon3]

if QCoreApplication.instance() is None:
    app = QApplication(sys.argv) 
view = Viewer()
view.show()
view.add_polygons(my_polygons2)
view.reset_view()
#view.add_polygons(device_polygons, alpha = 0.5)
#p = view.add_polygons([polygon3], color = 'red')
#view.add_port(Port(width = 100))


#QRectF scenerect = QRectF(mapToScene(0,0), mapToScene(width(), height()));