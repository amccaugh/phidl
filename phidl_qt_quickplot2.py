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
from qtpy import QtWidgets, QtCore, QtGui

class MyView(QtWidgets.QGraphicsView):
    def __init__(self):
        QtWidgets.QGraphicsView.__init__(self)

        self.setGeometry(QtCore.QRect(100, 100, 600, 250))
        self.setWindowTitle("PIHDL Graphics Window");
        
        # Create a QGraphicsScene which this view looks at
        self.scene = QtWidgets.QGraphicsScene(self)
        self.scene.setSceneRect(QtCore.QRectF())
        self.setScene(self.scene)

        for i in range(5):
            self.item = QtWidgets.QGraphicsEllipseItem(i*75, 10, 60, 40)
            self.scene.addItem(self.item)
            
    def add_polygon(self, points, color = '#A8F22A', alpha = 1):
        qpoly = QtGui.QPolygonF()
        for p in points:
            qpoly.append(QtCore.QPointF(p[0], p[1]))
        poly = self.scene.addPolygon(qpoly)
        
        if color is not None:
            qcolor = QtGui.QColor()
            qcolor.setNamedColor(color)
            qcolor.setAlphaF(alpha)
            poly.setBrush(qcolor)
        return poly
        
    
    def add_polygons(self, polygons, color = '#A8F22A', alpha = 1):
        for p in polygons:
            self.add_polygon(p, color = '#A8F22A', alpha = 1)
            
            
    # Mousewheel zoom, taken from http://stackoverflow.com/a/29026916
    def wheelEvent(self, event):
        # Zoom Factor
        zoomInFactor = 1.25
        zoomOutFactor = 1 / zoomInFactor
    
        # Set Anchors
        self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)
    
        # Save the scene pos
        oldPos = self.mapToScene(event.pos())
    
        # Zoom
        if event.delta() > 0:
            zoomFactor = zoomInFactor
        else:
            zoomFactor = zoomOutFactor
        self.scale(zoomFactor, zoomFactor)
    
        # Get the new position
        newPos = self.mapToScene(event.pos())
    
        # Move scene to old position
        delta = newPos - oldPos
        self.translate(delta.x(), delta.y())
        
        

if QtCore.QCoreApplication.instance() is None:
    app = QtWidgets.QApplication(sys.argv) 
view = MyView()
view.show()
p = view.add_polygon(polygon3, color = 'red')
view.add_polygons(device_polygons)
sys.exit(app.exec_())
