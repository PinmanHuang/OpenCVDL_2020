# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\hw1_05.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(274, 364)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox_7 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_7.setGeometry(QtCore.QRect(20, 10, 231, 311))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.groupBox_7.setFont(font)
        self.groupBox_7.setObjectName("groupBox_7")
        self.pushButton_10 = QtWidgets.QPushButton(self.groupBox_7)
        self.pushButton_10.setGeometry(QtCore.QRect(30, 20, 161, 41))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_11 = QtWidgets.QPushButton(self.groupBox_7)
        self.pushButton_11.setGeometry(QtCore.QRect(30, 70, 161, 41))
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_12 = QtWidgets.QPushButton(self.groupBox_7)
        self.pushButton_12.setGeometry(QtCore.QRect(30, 120, 161, 41))
        self.pushButton_12.setObjectName("pushButton_12")
        self.pushButton_13 = QtWidgets.QPushButton(self.groupBox_7)
        self.pushButton_13.setGeometry(QtCore.QRect(30, 170, 161, 41))
        self.pushButton_13.setObjectName("pushButton_13")
        self.pushButton_14 = QtWidgets.QPushButton(self.groupBox_7)
        self.pushButton_14.setGeometry(QtCore.QRect(30, 250, 161, 41))
        self.pushButton_14.setObjectName("pushButton_14")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_7)
        self.lineEdit.setGeometry(QtCore.QRect(30, 220, 161, 20))
        self.lineEdit.setObjectName("lineEdit")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_7.setTitle(_translate("MainWindow", "5. Training Cifar10 Classifier"))
        self.pushButton_10.setText(_translate("MainWindow", "5.1 Show Train Images"))
        self.pushButton_11.setText(_translate("MainWindow", "5.2 Show Hyperparameters"))
        self.pushButton_12.setText(_translate("MainWindow", "5.3 Show Model Strucuture"))
        self.pushButton_13.setText(_translate("MainWindow", "5.4 Show Accuracy"))
        self.pushButton_14.setText(_translate("MainWindow", "5.5 Test"))
        self.lineEdit.setText(_translate("MainWindow", "(0~9999)"))
