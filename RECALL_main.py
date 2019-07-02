#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:16:23 2019

@author: matthewconlin
"""


from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap,QIcon
import PyQt5.QtGui as gui
import PyQt5.QtCore as qtCore
import RECALL
import sys
import os
import pickle
import glob


wd = '/Users/matthewconlin/Documents/Research/WebCAT/'
                       
class ShowImageWindow(QWidget):
   
   def __init__(self):
        super().__init__()    
        
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()             
        self.initUI()
        
   def initUI(self):
       # Get all the frames #
       frames = glob.glob('frame'+'*')
       frame = frames[1]
       
       label = QLabel()
       pixmap = QPixmap(frame)
       label.setPixmap(pixmap)

       #label.resize(pixmap.width(),pixmap.height())
       txt = QLabel('Is this image clear enough to allow feature identification? Pressing "Yes" will launch the lidar data download process.')
       noBut = QPushButton('No')
       yesBut = QPushButton('Yes')
       
       grd = QGridLayout()
       grd.addWidget(label,0,0,4,4)
       grd.addWidget(txt,5,0,1,4)
       grd.addWidget(noBut,6,0,1,1)
       grd.addWidget(yesBut,6,1,1,1)
       
       self.setLayout(grd)
       self.setGeometry(400,100,10,10)
       self.setWindowTitle('RECALL')
       self.show()
        

class OtherCameraLocationInputWindow(QWidget):
   def __init__(self):
        super().__init__()    
        
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()             
        self.initUI()
        
   def initUI(self):
       lblDir1 = QLabel('Input the name of this camera:')
       self.bxName = QLineEdit()
       lblDir = QLabel('Input the location (lat/lon) of the camera below:')
       lblLat = QLabel('Camera Latitude (decimal degrees):')
       lblLon = QLabel('Camera Longitude (decimal degrees):')
       self.bxLat = QLineEdit()
       self.bxLon = QLineEdit()
       lblPth = QLabel('Input the path to the folder containing the imagery (in .mp4 format):')
       self.bxPth = QLineEdit()
       backBut = QPushButton('< Back')
       contBut = QPushButton('Continue >')
       
       backBut.clicked.connect(self.GoBack)
       contBut.clicked.connect(self.getInputs)
       
       grd = QGridLayout()
       grd.addWidget(lblDir1,0,0,1,3)
       grd.addWidget(self.bxName,0,3,1,3)
       grd.addWidget(lblDir,1,0,1,6)
       grd.addWidget(lblLat,2,1,1,3)
       grd.addWidget(self.bxLat,2,4,1,2)
       grd.addWidget(lblLon,3,1,1,3)
       grd.addWidget(self.bxLon,3,4,1,2)
       grd.addWidget(lblPth,4,0,1,6)
       grd.addWidget(self.bxPth,5,0,1,4)
       grd.addWidget(backBut,6,0,1,2)
       grd.addWidget(contBut,6,4,1,2)
       
       self.setLayout(grd)
       
       self.setGeometry(400,100,200,250)
       self.setWindowTitle('RECALL')
       self.show()
       
   def GoBack(self):
       self.close()
       self.backToOne = ChooseCameraWindow()    
       
   def getInputs(self):
       cameraName = self.bxName.text()
       cameraLocation = [float(self.bxLat.text()),float(self.bxLon.text())]
       pthToImagery = self.bxPth.text()
       # Save the camera name and location #
       with open(wd+'CameraLocation.pkl','wb') as f:
           pickle.dump(cameraLocation,f)
       with open(wd+'CameraName.pkl','wb') as f:
           pickle.dump(cameraName,f)
       with open(wd+'ImageryPath.pkl','wb') as f:
           pickle.dump(pthToImagery,f)       


class WebCATLocationWindow(QWidget):
   
   def __init__(self):
        super().__init__()    
        
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()             
        self.initUI()
        
   def initUI(self):
                    
       txt = QLabel('Select WebCAT camera:')
       opt = QComboBox()
       opt.addItem('--')
       opt.addItem('Buxton Coastal Hazard')
       opt.addItem('Cherry Grove Pier (south)')
       opt.addItem('Folly Beach Pier (north)')
       opt.addItem('Folly Beach Pier (south)')
       opt.addItem('St. Augustine Pier')
       opt.addItem('Twin Piers/Bradenton')
       opt.addItem('Miami 40th Street')
       opt.setCurrentIndex(0)
       backBut = QPushButton('< Back')
       contBut = QPushButton('Continue >')
    
       
       opt.activated.connect(self.getSelected)
       backBut.clicked.connect(self.GoBack)
       contBut.clicked.connect(self.DownloadVidAndExtractStills)
       
       grid = QGridLayout()
       
       grid.addWidget(txt,0,1,1,4)
       grid.addWidget(opt,1,1,1,4)
       grid.addWidget(backBut,8,1,1,2)
       grid.addWidget(contBut,8,3,1,2)

       
       self.setLayout(grid)
    
       self.setGeometry(400,100,300,100)
       self.setWindowTitle('RECALL')
       self.show()

   def getSelected(self,item):
          
       WebCATdict = {'Placeholder':[0,0],
                    'buxtoncoastalcam':[35.267777,-75.518448],
                    'cherrypiersouthcam':[ 33.829960, -78.633320],
                    'follypiernorthcam':[32.654731,-79.939322],
                    'follypiersouthcam':[32.654645,-79.939597],
                    'staugustinecam':[29.856559,-81.265545],
                    'twinpierscam':[27.466685,-82.699540],
                    'miami40thcam':[ 25.812227, -80.122400]}
      
       # Get location of selected camera #
       cams = ['Placeholder','buxtoncoastalcam','cherrypiersouthcam','follypiernorthcam','follypiersouthcam','staugustinecam','twinpierscam','miami40thcam']
       cameraLocation = WebCATdict[cams[item]]
       cameraName = cams[item]
       # Save the WebCAT camera location and name #
       with open(wd+'CameraLocation.pkl','wb') as f:
           pickle.dump(cameraLocation,f)
       with open(wd+'CameraName.pkl','wb') as f:
           pickle.dump(cameraName,f)
   
   def GoBack(self):
       self.close()
       self.backToOne = ChooseCameraWindow()
       
   def DownloadVidAndExtractStills(self):
       # Download the video #
       f = open(wd+'CameraName.pkl','rb')
       camToInput = pickle.load(f)
       vidFile = RECALL.GetVideo(camToInput)
       
       # Get the path to the video file #
       fullVidPth = wd + vidFile   
        
       # Decimate the video to 20 still-images #
       RECALL.DecimateVideo(fullVidPth)
       self.close()
       self.imWindow = ShowImageWindow()
       self.imWindow.show()

       
#       # Deal with Buxton name change #
#       fname = glob.glob(pickle.load(f)+'*')[0]
#       fs = os.path.getsize(wd+fname)
#       if fs<1000:
#           vidPth = RECALL.GetVideo('buxtonnorthcam')
       
       # Make sure we are still in the same directory as the video # 
       #os.chdir(vidPth.rsplit('/',1)[0]) # Go to the directory defined by the path prior to the final backslash in the vidFile string #
    

class ChooseCameraWindow(QWidget):
    def __init__(self):
        super().__init__()
        
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()  
          
        t = QLabel('Choose camera type:')
        WebCatOpt = QRadioButton('Select WebCAT camera from list')
        OtherOpt = QRadioButton('Input location of other camera')    
         
        WebCatOpt.clicked.connect(self.WebCAT_select)
        OtherOpt.clicked.connect(self.Other_select)
            
        vBox = QVBoxLayout()
        
        vBox.addWidget(t)
        vBox.addWidget(WebCatOpt)
        vBox.addWidget(OtherOpt)
        
        self.setLayout(vBox)
            
        self.setGeometry(400,100,300,100)
        self.setWindowTitle('RECALL')
        self.show()
        

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    def WebCAT_select(self):
        self.close()
        self.ww = WebCATLocationWindow()  
        self.ww.show()
    def Other_select(self):
        self.close()
        self.www = OtherCameraLocationInputWindow()
        self.www.show()
 

class WelcomeWindow(QWidget):
    
   def __init__(self):
        super().__init__()    
        
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()             

              
        txt = QLabel('Welcome to the Remote Coastal Camera Calibration Tool (RECALL)!')
        txt2 = QLabel('Developed in partnership with the Southeastern Coastal Ocean Observing Regional Association (SECOORA), '
                      +'the United States Geological Survey (USGS), and the National Oceanic and Atmospheric administration (NOAA), this tool allows you to calibrate any coastal camera of  '
                      +'known location with accessible video footage. For documentation on the methods employed by the tool, please refer to the GitHub readme (link here). If you have an '
                      +'issue, please post it on the GitHib issues page.')      
        txt2.setWordWrap(True)
        txt3 = QLabel('Press Continue to start calibrating a camera!')
        contBut = QPushButton('Continue >')
       
        contBut.clicked.connect(self.StartTool)
       
        grd = QGridLayout()
        
        grd.addWidget(txt,0,0,1,4)
        grd.addWidget(txt2,1,0,4,4)
        grd.addWidget(txt3,6,0,1,4)
        grd.addWidget(contBut,7,3,1,2)
       
        self.setLayout(grd)
        
        self.setGeometry(400,100,500,250)
        self.setWindowTitle('RECALL')
        self.show()
         
   def StartTool(self):
        self.close()
        self.tool = ChooseCameraWindow()
        self.tool.show()



test = WelcomeWindow()











