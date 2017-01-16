# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 14:56:44 2017

@author: amcc
"""
import numpy as np

polygon1 = np.array([[1,2],[3,6],[1,5]])*20
polygon2 = np.array([[1,1],[3,8],[3,5]])*20 + 80
polygon3 = np.array([[2,1],[3,9],[7,5]])*15 + 80
my_polygons = [polygon1, polygon2]
my_polygons2 = [polygon1, polygon2, polygon3]



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
                         
                         
            
if QtCore.QCoreApplication.instance() is None:
    app = QtWidgets.QApplication(sys.argv) 

widget = MyWidget()
widget.show()

#sys.exit(app.exec_())


def add_polygon_to_scene(points, scene):
    qpoly = QtGui.QPolygonF()
    for p in points:
        qpoly.append(QtCore.QPointF(p[0], p[1]))
    QtWidgets.QGraphicsPolygonItem(qpoly, scene)
    
    
    
scene = QtWidgets.QGraphicsScene()
scene.addText("Hello, world!")
add_polygon_to_scene(points = polygon1, scene)
view = QtGQtWidgetsui.QGraphicsView(scene)
view.show()




#==============================================================================
# MyViewer
#==============================================================================


class MyViewer(QtGui.QGraphicsView):
    def __init__(self, parent):
        super(MyViewer, self).__init__(parent)

#    def wheelEvent(self, event):
#        """
#        Zoom in or out of the view.
#        """
#        zoomInFactor = 1.25
#        zoomOutFactor = 1 / zoomInFactor
#    
#        # Save the scene pos
#        oldPos = self.mapToScene(event.pos())
#    
#        # Zoom
#        if event.angleDelta().y() > 0:
#            zoomFactor = zoomInFactor
#        else:
#            zoomFactor = zoomOutFactor
#        self.scale(zoomFactor, zoomFactor)
#    
#        # Get the new position
#        newPos = self.mapToScene(event.pos())
#    
#        # Move scene to old position
#        delta = newPos - oldPos
#        self.translate(delta.x(), delta.y())