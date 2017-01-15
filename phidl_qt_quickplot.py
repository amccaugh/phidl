# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 14:56:44 2017

@author: amcc
"""
import numpy as np

polygon1 = np.array([[1,2],[3,6],[1,5]])
polygon2 = np.array([[1,1],[3,8],[3,5]]) + 8
my_polygons = [polygon1, polygon2]

import sys, math
from qtpy import QtCore, QtGui, QtWidgets

class MyWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.pen = QtGui.QPen(QtGui.QColor(0,0,0))                      # set lineColor
        self.pen.setWidth(3)                                            # set lineWidth
        self.brush = QtGui.QBrush(QtGui.QColor(255,255,255,255))        # set fillColor  
        self.polygon = self.createPoly(8,150,0)                         # polygon with n points, radius, angle of the first point
        self.update_polygons(my_polygons)

        
    def createPoly(self, n, r, s):
        polygon = QtGui.QPolygonF() 
        w = 360/n                                                       # angle per step
        for i in range(n):                                              # add the points of polygon
            t = w*i + s
            x = r*math.cos(math.radians(t))
            y = r*math.sin(math.radians(t))
            polygon.append(QtCore.QPointF(self.width()/2 +x, self.height()/2 + y))  

        return polygon

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setPen(self.pen)
        painter.setBrush(self.brush)  
#        painter.drawPolygon(self.polygon)
        for poly in self.polygons:
            painter.drawPolygon(poly)

        
    def create_polygon(self, points):
        new_poly = QtGui.QPolygonF()
        for p in points:
            new_poly.append(QtCore.QPointF(p[0], p[1]))
        return new_poly
        
        
    def update_polygons(self, polygons):
        self.polygons = [self.create_polygon(poly) for poly in polygons]
            
        
app = QtWidgets.QApplication(sys.argv) 

widget = MyWidget()
widget.show()

sys.exit(app.exec_())