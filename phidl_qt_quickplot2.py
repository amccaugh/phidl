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

        self.scene = QtWidgets.QGraphicsScene(self)
        self.scene.setSceneRect(QtCore.QRectF())

        self.setScene(self.scene)

        for i in range(5):
            self.item = QtWidgets.QGraphicsEllipseItem(i*75, 10, 60, 40)
            self.scene.addItem(self.item)
            
    def add_polygon(self, points):
        qpoly = QtGui.QPolygonF()
        for p in points:
            qpoly.append(QtCore.QPointF(p[0], p[1]))
        self.scene.addPolygon(qpoly)
        


if QtCore.QCoreApplication.instance() is None:
    app = QtWidgets.QApplication(sys.argv) 
view = MyView()
view.show()
view.add_polygon(polygon1)
sys.exit(app.exec_())