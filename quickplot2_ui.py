# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\anm16\amcc\Code\device-layout\quickplot2.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1101, 756)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.myGraphicsView = MyViewer(self.centralwidget)
        self.myGraphicsView.setGeometry(QtCore.QRect(10, 10, 881, 671))
        self.myGraphicsView.setObjectName("myGraphicsView")
        self.checkbox_aliases = QtWidgets.QCheckBox(self.centralwidget)
        self.checkbox_aliases.setGeometry(QtCore.QRect(910, 30, 131, 17))
        self.checkbox_aliases.setObjectName("checkbox_aliases")
        self.checkbox_ports = QtWidgets.QCheckBox(self.centralwidget)
        self.checkbox_ports.setGeometry(QtCore.QRect(910, 50, 131, 17))
        self.checkbox_ports.setObjectName("checkbox_ports")
        self.checkbox_subports = QtWidgets.QCheckBox(self.centralwidget)
        self.checkbox_subports.setGeometry(QtCore.QRect(910, 70, 131, 17))
        self.checkbox_subports.setObjectName("checkbox_subports")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1101, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.checkbox_aliases.setText(_translate("MainWindow", "Display aliases (F1)"))
        self.checkbox_ports.setText(_translate("MainWindow", "Display ports (F2)"))
        self.checkbox_subports.setText(_translate("MainWindow", "Display subports (F3)"))

from MyCustomClasses import MyViewer
