#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:16:23 2019

@author: matthewconlin
"""


from PyQt5.QtWidgets import *
import PyQt5.QtGui as gui
import PyQt5.QtCore as qtCore
import RECALL
import sys
import os



class CameraLocationWindow(QMainWindow):
   def __init__(self):
        super().__init__()    
        
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()  
        
        self.resize(250,150)
        self.center()
        self.title = 'RECALL Test'

        self.opt = QComboBox(self)
        self.opt.addItem('Miami')
        self.opt.addItem('Bradenton')
        
        self.setCentralWidget(self.opt)
        self.show()
        
   def center(self):
       qr = self.frameGeometry()
       cp = QDesktopWidget().availableGeometry().center()
       qr.moveCenter(cp)
       self.move(qr.topLeft())


class ChooseCameraWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()  
            
        self.resize(250,150)
        self.center()
        self.title = 'RECALL Test'
        
        
        w = QWidget()
        vBig = QVBoxLayout(w)
        vInner = QVBoxLayout()
        
        self.t = QLabel('Choose camera type:')
        self.WebCatOpt = QRadioButton('Select WebCAT camera from list')
        self.OtherOpt = QRadioButton('Input location of other camera')
        
        vInner.addWidget(self.t)
        vInner.addWidget(self.WebCatOpt)
        vInner.addWidget(self.OtherOpt)
        vBig.addLayout(vInner)
        
        self.WebCatOpt.clicked.connect(self.WebCAT_select)  
        w.show()
        sys.exit(app.exec_())


    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    def WebCAT_select(self):
        self.w = CameraLocationWindow()  
        self.hide()
        self.w.show()
 

test = ChooseCameraWindow()





import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, 
                             QToolTip, QMessageBox, QLabel)

class Window2(QMainWindow):                           # <===
    def __init__(self):
        super(Window2,self).__init__()
        self.setWindowTitle("Window22222")
        
        w = QWidget()
        vBig = QVBoxLayout(w)
        vInner = QVBoxLayout()
        
        self.t = QLabel('Choose camera type:')
        self.WebCatOpt = QRadioButton('Select WebCAT camera from list')
        self.OtherOpt = QRadioButton('Input location of other camera')
        
        vInner.addWidget(self.t)
        vInner.addWidget(self.WebCatOpt)
        vInner.addWidget(self.OtherOpt)
        vBig.addLayout(vInner)
        
        self.show()



class Window(QMainWindow):
    def __init__(self):
        super(Window,self).__init__()

        self.title = "First Window"
        self.top = 100
        self.left = 100
        self.width = 680
        self.height = 500

        self.pushButton = QPushButton("Start", self)
        self.pushButton.move(275, 200)
        self.pushButton.setToolTip("<h3>Start the Session</h3>")

        self.pushButton.clicked.connect(self.window2)              # <===

        self.main_window()

    def main_window(self):
        self.label = QLabel("Manager", self)
        self.label.move(285, 175)
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.show()

    def window2(self):                                             # <===
        self.w = Window2()
        self.w.show()


if __name__ == "__main__":
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
    window = Window()
    sys.exit(app.exec())