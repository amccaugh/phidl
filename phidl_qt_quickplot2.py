# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:18:40 2017

@author: amcc
"""

import numpy as np

polygon1 = np.array([[1,2],[3,6],[1,5]])*20
polygon2 = np.array([[1,1],[3,8],[3,5]])*20 + 80
polygon3 = np.array([[2,1],[3,9],[7,5]])*15 + 80
my_polygons = [polygon1, polygon2]
my_polygons2 = [polygon1, polygon2, polygon3]


import sys
try:
    #from qtpy import QtWidgets, QtCore, QtGui
    from PyQt4 import QtCore, QtGui, QtOpenGL, QGraphicsView
    from PyQt4.QtCore import QPoint, QRect, QSize, Qt
except:
    from PyQt5 import QtCore, QtGui, QtWidgets, QtOpenGL
    from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QApplication, QGraphicsEllipseItem, QRubberBand
    from PyQt5.QtCore import QPoint, QPointF, QRectF, QSize, Qt, QCoreApplication
    from PyQt5.QtGui import QColor, QPolygonF, QPen


class MyView(QGraphicsView):
    def __init__(self):
        QGraphicsView.__init__(self)

        self.setGeometry(QRect(100, 100, 600, 250))
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
        self.pen = QPen(QtCore.Qt.black)
        self.pen.setCosmetic(True)

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
        self.scene.setSceneRect(QRectF(xmin-width, ymax-height, width*5, height*5))
        
    
            
    def clear(self):
        self.scene.clear()
        
            
#==============================================================================
#  Mousewheel zoom, taken from http://stackoverflow.com/a/29026916
#==============================================================================
    def wheelEvent(self, event):
        # Zoom Factor
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor
    
        # Set Anchors
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
    
        # Save the scene pos
        oldPos = self.mapToScene(event.pos())
    
        # Zoom
        if event.angleDelta().y() > 0:
            zoomFactor = zoomInFactor
        else:
            zoomFactor = zoomOutFactor
        self.scale(zoomFactor, zoomFactor)
    
        # Get the new position
        newPos = self.mapToScene(event.pos())
    
        # Move scene to old position
        delta = newPos - oldPos
        self.translate(delta.x(), delta.y())
        
        
#==============================================================================
#  Zoom to rectangle, from
#  https://wiki.python.org/moin/PyQt/Selecting%20a%20region%20of%20a%20widget
#==============================================================================
    def mousePressEvent(self, event):
        # Create the rubberband object (for zoom to rectangle)
        self._rb_origin = QPoint()
        
        if event.button() == Qt.LeftButton:
            self._rb_origin = QPoint(event.pos())
            self.rubberBand.setGeometry(QRect(self._rb_origin, QSize()))
#            self.rubberBand.show()
            
    def mouseMoveEvent(self, event):
        if not self._rb_origin.isNull():
            self.rubberBand.setGeometry(QRect(self._rb_origin, event.pos()).normalized())
            if abs(self.rubberBand.width()) > 3 or abs(self.rubberBand.height()) > 3:
                self.rubberBand.show()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.rubberBand.hide()
            rb_rect = QRect(self._rb_origin, event.pos())
#            rb_corner1 = self._rb_origin # Each of these is a QPoint
#            rb_corner2 = event.pos()
#            rb_center = (rb_corner1 + rb_corner2)/2
            rb_center = rb_rect.center()
            rb_size = rb_rect.size()
            
            if abs(rb_size.width()) > 2 and abs(rb_size.height()) > 2:
                viewport_size = self.viewport().geometry().size()
                
                zoom_factor_x = abs(viewport_size.width() / rb_size.width())
                zoom_factor_y = abs(viewport_size.height() / rb_size.height())
                
    #            viewport_center = self.viewport().geometry().center()
                
    #            old_center = self.mapToScene(viewport_center)
                new_center = self.mapToScene(rb_center)
    #            delta =  old_center - new_center
    #            self.translate(delta.x(), delta.y())
                
                zoom_factor = min(zoom_factor_x, zoom_factor_y)
                self.scale(zoom_factor, zoom_factor)
                self.centerOn(new_center)
                
#                print('\n###\n')
#                print('Rubberband corner1 = %s' % rb_corner1)
#                print('Rubberband corner2 = %s' % rb_corner2)
#                print('Rubberband center  = %s' % rb_center)
##                print('Viewport center    = %s' % viewport_center)
#                print('Viewport center mapped   = %s' % old_center)
#                print('Rubberband center mapped = %s' % new_center)
    #            print('Total translation = %s' % delta)
            
            
#            print('curen)
#        zoom_factor = 
#        newPos = self.mapToScene(rb_center)
#        self.scale(zoomFactor, zoomFactor)
            
            
            

if QCoreApplication.instance() is None:
    app = QApplication(sys.argv) 
view = MyView()
view.show()
#p = view.add_polygons([polygon3], color = 'red')
view.add_polygons(device_polygons, alpha = 0.5)



#view.setDragMode(view.ScrollHandDrag)
#view.setInteractive(False)

#sys.exit(app.exec_())



#void Widget::mousePressEvent(QMouseEvent *event)
#{
#    origin = event->pos();
#    if (!rubberBand)
#        rubberBand = new QRubberBand(QRubberBand::Rectangle, this);
#    rubberBand->setGeometry(QRect(origin, QSize()));
#    rubberBand->show();
#}
#
#void Widget::mouseMoveEvent(QMouseEvent *event)
#{
#    rubberBand->setGeometry(QRect(origin, event->pos()).normalized());
#}
#
#void Widget::mouseReleaseEvent(QMouseEvent *event)
#{
#    rubberBand->hide();
#    // determine selection, for example using QRect::intersects()
#    // and QRect::contains().
#}
