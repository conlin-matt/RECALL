#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:16:23 2019

@author: matthewconlin
"""


from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap,QIcon,QMouseEvent,QFont
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QAbstractTableModel
import PyQt5.QtGui as gui
import PyQt5.QtCore as qtCore
import RECALL
import sys
import os
import pickle
import glob
import requests
import pandas as pd
import pptk
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

wd = '/Users/matthewconlin/Documents/Research/WebCAT/'


#=============================================================================#
# Calibration module #
#=============================================================================#
        
class calibrate_GetHorizonWindow(QWidget):
   def __init__(self):
        super().__init__()    
        
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()             
                 
        # Left menu box setup #
        bf = QFont()
        bf.setBold(True)
        leftBar1 = QLabel('• Welcome!')
        leftBar2 = QLabel('• Get imagery')
        leftBar3 = QLabel('• Get lidar data')
        leftBar4 = QLabel('• Pick GCPs')
        leftBar5 = QLabel('• Calibrate')
        leftBar5.setFont(bf)

        leftGroupBox = QGroupBox('Contents:')
        vBox = QVBoxLayout()
        vBox.addWidget(leftBar1)
        vBox.addWidget(leftBar2)
        vBox.addWidget(leftBar3)
        vBox.addWidget(leftBar4)
        vBox.addWidget(leftBar5)
        vBox.addStretch(1)
        leftGroupBox.setLayout(vBox)
        ########################  

        # Right contents box setup #
        plt.ioff()
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        
        frames = glob.glob('frame'+'*')
        frame = frames[1]
        
        img = mpimg.imread(wd+'/'+frame)
        imgplot = plt.imshow(img)
        
        self.canvas.draw()
        
        self.introLab = QLabel('Welcome to the Calibration module! In just a few more steps, you will obtain calibration parameters for this camera. First, click on two points on the horizon. Make sure to click the more-left point first.')
        self.introLab.setWordWrap(True)

        self.rightGroupBox = QGroupBox()
        self.grd = QGridLayout()
        self.grd.addWidget(self.introLab,0,0,1,4)
        self.grd.addWidget(self.canvas,2,0,4,4)
        self.rightGroupBox.setLayout(self.grd)
        ###############################
        
        # Full widget layout setup #
        fullLayout = QGridLayout()
        fullLayout.addWidget(leftGroupBox,0,0,2,2)
        fullLayout.addWidget(self.rightGroupBox,0,3,2,4)
        self.setLayout(fullLayout)

        self.setGeometry(400,100,1000,500)
        self.setWindowTitle('RECALL')
        self.show()
        ############################ 
        

class calibrate_CalibrateThread1(QThread):   
        
    finishSignal1 = pyqtSignal('PyQt_PyObject')
    finishSignal2 = pyqtSignal('PyQt_PyObject')
    finishSignal3 = pyqtSignal('PyQt_PyObject')

    def __init__(self,GCPs_im,GCPs_lidar,horizonPts,cameraElev,cameraDir):
        super().__init__()
        self.GCPs_im = GCPs_im
        self.GCPs_lidar = GCPs_lidar
        self.horizonPts = horizonPts
        self.cameraElev = cameraElev
        self.cameraDir = cameraDir
        
    def run(self):
        
        print('Thread Started')
        
        t,k,R = RECALL.calibrate_GetInitialEstimate(self.GCPs_im,self.GCPs_lidar,self.horizonPts,self.cameraElev,self.cameraDir)                   
        self.finishSignal1.emit(1)    
        Kopt,Ropt,topt,k1,k2 = RECALL.calibrate_OptimizeEstimate(t,k,R,self.GCPs_im,self.GCPs_lidar)
        self.finishSignal2.emit(1)    
        resid = RECALL.calibrate_CheckCalibrationAccuracy(Kopt,Ropt,topt,k1,k2,self.GCPs_im,self.GCPs_lidar)
        self.finishSignal3.emit(1)    
           
        print('Thread Done')


class calibrate_FinalInputs(QWidget):
   def __init__(self):
        super().__init__()    
        
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()             
                 
        # Left menu box setup #
        bf = QFont()
        bf.setBold(True)
        leftBar1 = QLabel('• Welcome!')
        leftBar2 = QLabel('• Get imagery')
        leftBar3 = QLabel('• Get lidar data')
        leftBar4 = QLabel('• Pick GCPs')
        leftBar5 = QLabel('• Calibrate')
        leftBar5.setFont(bf)

        leftGroupBox = QGroupBox('Contents:')
        vBox = QVBoxLayout()
        vBox.addWidget(leftBar1)
        vBox.addWidget(leftBar2)
        vBox.addWidget(leftBar3)
        vBox.addWidget(leftBar4)
        vBox.addWidget(leftBar5)
        vBox.addStretch(1)
        leftGroupBox.setLayout(vBox)
        ########################  

        # Right contents box setup #
        self.lab = QLabel('Just a couple more things:')
        self.lab1 = QLabel('Input estimate for camera elevation (in meters):')
        self.elevBx = QLineEdit()
        self.lab2 = QLabel('In which direction does the camera look?')
        self.cb = QComboBox()
        self.cb.addItem('--')
        self.cb.addItem('Between North and East')
        self.cb.addItem('Between East and South')
        self.cb.addItem('Between North and West')
        self.cb.addItem('Between West and South')
        self.calibBut = QPushButton('CALIBRATE')
        
        self.rightGroupBox = QGroupBox()
        self.grd = QGridLayout()
        self.grd.addWidget(self.lab,0,0,1,5)
        self.grd.addWidget(self.lab1,1,0,1,3)
        self.grd.addWidget(self.elevBx,1,3,1,2)
        self.grd.addWidget(self.lab2,2,0,1,3)
        self.grd.addWidget(self.cb,2,3,1,2)
        self.grd.addWidget(self.calibBut,3,1,1,2)
        self.grd.setAlignment(Qt.AlignCenter)
        self.rightGroupBox.setLayout(self.grd)
        ###############################
        
        # Full widget layout setup #
        fullLayout = QGridLayout()
        fullLayout.addWidget(leftGroupBox,0,0,2,2)
        fullLayout.addWidget(self.rightGroupBox,0,3,2,4)
        self.setLayout(fullLayout)

        self.setGeometry(400,100,1000,500)
        self.setWindowTitle('RECALL')
        self.show()
        ############################ 
        
        # Connect widgets with signals #
        self.cb.activated.connect(self.getInputs)
        self.calibBut.clicked.connect(self.calibrate)
        ################################
        
   def getInputs(self,item):
        dirCodeList = [0,1,2,3,4]
        self.cameraDir = dirCodeList[item]
        
        self.cameraElev = float(self.elevBx.text())
        
        f1 = open(wd+'GCPs_im.pkl','rb') 
        f2 = open(wd+'GCPs_lidar.pkl','rb') 
        f3 = open(wd+'horizonPts.pkl','rb') 
        GCPs_im = pickle.load(f1)
        GCPs_lidar = pickle.load(f2)
        horizonPts = pickle.load(f3)

        
        # Instantiate worker thread now that we have all the inputs #
        self.worker = calibrate_CalibrateThread1(GCPs_im,GCPs_lidar,horizonPts,self.cameraElev,self.cameraDir)
        #############################################################
        
   def calibrate(self):
        self.lab.setParent(None)
        self.lab1.setParent(None)
        self.lab2.setParent(None)
        self.elevBx.setParent(None)
        self.cb.setParent(None)
        self.calibBut.setParent(None)
        
        self.firstThreadLab = QLabel('Getting initial estimates:')
        self.grd.addWidget(self.firstThreadLab,0,0,1,3)
        
        self.worker.start()
        self.worker.finishSignal1.connect(self.on_closeSignal1)  
        self.worker.finishSignal2.connect(self.on_closeSignal2)  
        self.worker.finishSignal3.connect(self.on_closeSignal3)  
   
   def on_closeSignal1(self):
        self.firstThreadDoneLab = QLabel('Done.')
        self.secondThreadLab = QLabel('Optimizing estimates:')
        self.grd.addWidget(self.firstThreadDoneLab,0,3,1,2)
        self.grd.addWidget(self.secondThreadLab,1,0,1,3)
        
   def on_closeSignal2(self):
        self.secondThreadDoneLab = QLabel('Done.')
        self.thirdThreadLab = QLabel('Computing residuals:')
        self.grd.addWidget(self.secondThreadDoneLab,1,3,1,2)
        self.grd.addWidget(self.thirdThreadLab,2,0,1,3)
        
   def on_closeSignal3(self):
       self.thirdThreadDoneLab = QLabel('Done.')
       self.compLab = QLabel('Calibration complete!')
       self.resBut = QPushButton('Results >')
       self.grd.addWidget(self.thirdThreadDoneLab,2,3,1,2)
       self.grd.addWidget(self.compLab,3,0,1,3)
       self.grd.addWidget(self.resBut,3,3,1,2)
       
       self.resBut.clicked.connect(self.on_resClick)
       
   def on_resClick(self):
        self.close
        self.finalWindow = ShowCalibResultsWindow()
        self.finalWindow.show()
        


class calibrate_GetHorizonWindow(QWidget):
   def __init__(self):
        super().__init__()    
        
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()             
                 
        # Left menu box setup #
        bf = QFont()
        bf.setBold(True)
        leftBar1 = QLabel('• Welcome!')
        leftBar2 = QLabel('• Get imagery')
        leftBar3 = QLabel('• Get lidar data')
        leftBar4 = QLabel('• Pick GCPs')
        leftBar5 = QLabel('• Calibrate')
        leftBar5.setFont(bf)

        leftGroupBox = QGroupBox('Contents:')
        vBox = QVBoxLayout()
        vBox.addWidget(leftBar1)
        vBox.addWidget(leftBar2)
        vBox.addWidget(leftBar3)
        vBox.addWidget(leftBar4)
        vBox.addWidget(leftBar5)
        vBox.addStretch(1)
        leftGroupBox.setLayout(vBox)
        ########################  

        # Right contents box setup #
        plt.ioff()
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        
        frames = glob.glob('frame'+'*')
        frame = frames[1]
        
        img = mpimg.imread(wd+'/'+frame)
        imgplot = plt.imshow(img)
        
        self.canvas.draw()
        
        self.introLab = QLabel('Welcome to the Calibration module! In just a few more steps, you will obtain calibration parameters for this camera. First, click on two points on the horizon. Make sure to click the more-left point first.')
        self.introLab.setWordWrap(True)

        self.rightGroupBox = QGroupBox()
        self.grd = QGridLayout()
        self.grd.addWidget(self.introLab,0,0,1,4)
        self.grd.addWidget(self.canvas,2,0,4,4)
        self.rightGroupBox.setLayout(self.grd)
        ###############################
        
        # Full widget layout setup #
        fullLayout = QGridLayout()
        fullLayout.addWidget(leftGroupBox,0,0,2,2)
        fullLayout.addWidget(self.rightGroupBox,0,3,2,4)
        self.setLayout(fullLayout)

        self.setGeometry(400,100,1000,500)
        self.setWindowTitle('RECALL')
        self.show()
        ############################ 
        
        # Get the horizon points #
        self.pt = plt.ginput(2,show_clicks=True)   
        ##########################
        
        self.afterClick()
    
   def afterClick(self):
        with open(wd+'horizonPts.pkl','wb') as f:
            pickle.dump(self.pt,f)
            
        self.close()
        self.moreInputs = calibrate_FinalInputs()
        self.moreInputs.show()
        


#=============================================================================#
# Interactive GCP picking module #
#=============================================================================#

class PickGCPsWindow(QWidget):
   def __init__(self):
        super().__init__()    
        
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()             
          
        # Define variables which will hold the picked GCPs #    
        self.GCPs_im = np.empty([0,2])
        self.GCPs_lidar = np.empty([0,3])
        ####################################################
        
        # Left menu box setup #
        bf = QFont()
        bf.setBold(True)
        leftBar1 = QLabel('• Welcome!')
        leftBar2 = QLabel('• Get imagery')
        leftBar3 = QLabel('• Get lidar data')
        leftBar4 = QLabel('• Pick GCPs')
        leftBar4.setFont(bf)
        leftBar5 = QLabel('• Calibrate')
        
        leftGroupBox = QGroupBox('Contents:')
        vBox = QVBoxLayout()
        vBox.addWidget(leftBar1)
        vBox.addWidget(leftBar2)
        vBox.addWidget(leftBar3)
        vBox.addWidget(leftBar4)
        vBox.addWidget(leftBar5)
        vBox.addStretch(1)
        leftGroupBox.setLayout(vBox)
        ########################  
        
        # Right contents box setup #
        plt.ioff()
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        
        frames = glob.glob('frame'+'*')
        frame = frames[1]
        
        img = mpimg.imread(wd+'/'+frame)
        imgplot = plt.imshow(img)
        
        self.canvas.draw()
        
        self.introLab = QLabel('Welcome to the GCP picking module! Here, you will be guided through the process of co-locating points in the image and the lidar observations. You will need to identify the correspondence of at least 6 unique points for the calibration to work.')
        self.introLab.setWordWrap(True)
        self.goLab = QLabel('Ready to co-locate a point?:')
        self.goBut = QPushButton('Go')
        
        self.rightGroupBox = QGroupBox()
        self.grd = QGridLayout()
        self.grd.addWidget(self.goBut,7,3,1,1)
        self.grd.addWidget(self.goLab,7,0,1,1)
        self.grd.addWidget(self.introLab,0,0,1,4)
        self.grd.addWidget(self.canvas,2,0,4,4)
        self.rightGroupBox.setLayout(self.grd)
        ###############################

        # Connect widgets with signals #
        self.goBut.clicked.connect(self.getPoints1)
        ################################
        
        # Full widget layout setup #
        fullLayout = QGridLayout()
        fullLayout.addWidget(leftGroupBox,0,0,2,2)
        fullLayout.addWidget(self.rightGroupBox,0,3,2,4)
        self.setLayout(fullLayout)

        self.setGeometry(400,100,1000,500)
        self.setWindowTitle('RECALL')
        self.show()
        ############################


   def getPoints1(self):
       print('In Function')

       self.goBut.setParent(None)
       self.goLab.setParent(None)
       self.introLab.setParent(None)
       
#       self.setWindowTitle(str(len(self.GCPs_lidar))+'/6 GCPs identified')
             
       self.dirLab = QLabel('Click on the point in the image:')      
       self.grd.addWidget(self.dirLab,0,0,1,2)
       
       self.pt = plt.ginput(show_clicks=True)   
       print(self.pt)
       
       self.afterClick()
       
   def afterClick(self):
       print('In Function')
       
       self.grd.removeWidget(self.dirLab)
       self.dirLab.deleteLater()
       self.dirLab = None
       
       self.savedLab = QLabel('Image coordinate of point saved!')
       self.dirLab2 = QLabel('Now, identify the point in the lidar point cloud (click Help for directions). Then, click Continue (to pick more) or Stop (to finish picking).')
       self.dirLab2.setWordWrap(True)
       self.contBut = QPushButton('Continue')
       self.stopBut = QPushButton('Stop')
       self.helpBut = QPushButton('Help')
       
       self.helpBut.clicked.connect(self.onHelpClick)
       self.contBut.clicked.connect(self.onContClick)
       self.stopBut.clicked.connect(self.onStopClick)
              
       self.grd.addWidget(self.savedLab,0,0,1,4)
       self.grd.addWidget(self.dirLab2,1,0,1,4)
       self.grd.addWidget(self.stopBut,7,2,1,1)
       self.grd.addWidget(self.contBut,7,3,1,1)
       self.grd.addWidget(self.helpBut,7,0,1,1)
       
#       if len(self.GCPs_lidar)<5:
#           self.stopBut.setEnabled(False)
#       else:
#           self.stopBut.setEnabled(True)
       
       f = open(wd+'lidarPC.pkl','rb')
       self.pc = pickle.load(f)
       self.v = pptk.viewer(self.pc,self.pc.iloc[:,2])
       self.v.set(point_size=0.1,theta=-25,phi=0,lookat=[0,0,20],color_map_scale=[-1,10],r=0)
              
   def onHelpClick(self):
       msg = QMessageBox(self)
       msg.setIcon(msg.Question)
       msg.setText('The lidar point cloud has been opened in a seperate window. The viewer can be navigated by clicking and dragging (to rotate view) as well as zooming in/out. Try to rotate/zoom the view until it looks as similar to the image as you can. To select a point, first right click anywhere in the viewer. Then, hold Control (Windows) or Command (Mac) and left click on the point to select it. Then return to this program to continue.')
       msg.setStandardButtons(msg.Ok)
       msg.show()
       
   def onContClick(self):    
       
       self.GCPs_im = np.append(self.GCPs_im,self.pt,axis = 0)
       print(self.GCPs_im)

       p = self.v.get('selected')
       self.GCPs_lidar = np.vstack((self.GCPs_lidar,self.pc.iloc[p,:]))
       print(self.GCPs_lidar)

#       self.setWindowTitle(str(len(self.GCPs_lidar))+'/6 GCPs identified')

       self.savedLab.setParent(None)
       self.dirLab2.setParent(None)
       self.contBut.setParent(None)
       self.stopBut.setParent(None)
       self.helpBut.setParent(None)

       
       self.savedLab2 = QLabel('Real world coordinates of point saved!')
       self.dirLab3 = QLabel('Select another point in the image:')
       
       self.grd.addWidget(self.savedLab2,0,0,1,2)
       self.grd.addWidget(self.dirLab3,1,0,1,2)

       self.pt = plt.ginput(show_clicks=True)   
       print(self.pt)
       
       self.afterClick2()

   def afterClick2(self):
       
       self.savedLab2.setParent(None)
       self.dirLab3.setParent(None)
       
       self.savedLab = QLabel('Image coordinate of point saved!')
       self.dirLab2 = QLabel('Now, identify the point in the lidar point cloud (click Help for directions). Then, click Continue (to pick more) or Stop (to finish picking).')
       self.dirLab2.setWordWrap(True)
       self.contBut = QPushButton('Continue')
       self.stopBut = QPushButton('Stop')
       self.helpBut = QPushButton('Help')
       
       self.helpBut.clicked.connect(self.onHelpClick)
       self.contBut.clicked.connect(self.onContClick)
       self.stopBut.clicked.connect(self.onStopClick)
       
       self.grd.addWidget(self.savedLab,0,0,1,4)
       self.grd.addWidget(self.dirLab2,1,0,1,4)
       self.grd.addWidget(self.stopBut,7,2,1,1)
       self.grd.addWidget(self.contBut,7,3,1,1)
       self.grd.addWidget(self.helpBut,7,0,1,1)
       
#       if len(self.GCPs_lidar)<5:
#           self.stopBut.setEnabled(False)
#       else:
#           self.stopBut.setEnabled(True)
               
   def onStopClick(self):
       
       self.GCPs_im = np.append(self.GCPs_im,self.pt,axis = 0)
       print(self.GCPs_im)

       p = self.v.get('selected')
       self.GCPs_lidar = np.vstack((self.GCPs_lidar,self.pc.iloc[p,:]))
       print(self.GCPs_lidar)

#       self.setWindowTitle(str(len(self.GCPs_lidar))+'/6 GCPs identified')


       self.savedLab.setParent(None)
       self.dirLab2.setParent(None)
       self.contBut.setParent(None)
       self.stopBut.setParent(None)
       self.helpBut.setParent(None)
       
    
       self.ax.plot(self.GCPs_im[:,0],self.GCPs_im[:,1],'ro')

       
       self.lab = QLabel('Your GCPs are shown on the image below. Are you happy with them? Press Continue to perform the calibration using these GCPs or select Retry to pick again.')
       self.lab.setWordWrap(True)
       self.contBut = QPushButton('Continue')
       self.retryBut = QPushButton('Retry')
       
       self.contBut.clicked.connect(self.GotoCalibration)
       self.retryBut.clicked.connect(self.Retry)
       
       self.grd.addWidget(self.lab,0,0,1,4)
       self.grd.addWidget(self.retryBut,7,0,1,1)
       self.grd.addWidget(self.contBut,7,1,1,1)
       

   def Retry(self):
       self.v.close()
       self.close()
       self.a = PickGCPsWindow()
       self.a.show()
       
   def GotoCalibration(self):
       
       with open(wd+'GCPs_im.pkl','wb') as f:
            pickle.dump(self.GCPs_im,f)
       with open(wd+'GCPs_lidar.pkl','wb') as f:
            pickle.dump(self.GCPs_lidar,f)
            
       self.close()
       self.calibrateWindow = calibrate_GetHorizonWindow()
       self.calibrateWindow.show()
    
#=============================================================================#      
#=============================================================================#
       
       



#=============================================================================#
# Lidar dataset search, selection, and download module #
#=============================================================================#
           
class getLidar_FormatChosenSetThread(QThread):   
        
    finishSignal = pyqtSignal('PyQt_PyObject')

    def __init__(self,cameraLoc_lat,cameraLoc_lon):
        super().__init__()
        self.cameraLoc_lat = cameraLoc_lat
        self.cameraLoc_lon = cameraLoc_lon
        
    def run(self):
        
        f = open(wd+'lidarDat.pkl','rb')
        lidarDat = pickle.load(f)

        pc = RECALL.getLidar_CreatePC(lidarDat,self.cameraLoc_lat,self.cameraLoc_lon)
          
        with open(wd+'lidarPC.pkl','wb') as f:
            pickle.dump(pc,f)
            
        self.finishSignal.emit(1)    
        
        print('Thread Done')
        

class getLidar_DownloadChosenSetThread(QThread):

    threadSignal = pyqtSignal('PyQt_PyObject')
    finishSignal = pyqtSignal('PyQt_PyObject')

    def __init__(self,cameraLoc_lat,cameraLoc_lon):
        super().__init__()
        self.cameraLoc_lat = cameraLoc_lat 
        self.cameraLoc_lon = cameraLoc_lon
        
    def run(self):
        print('Thread Started')
        
        f = open(wd+'tilesKeep.pkl','rb')
        tilesKeep = pickle.load(f)
        
        f = open(wd+'chosenLidarID.pkl','rb')
        IDToDownload = pickle.load(f)
        
        i = 0
        lidarDat = np.empty([0,3])
        for thisFile in tilesKeep:
            
            i = i+1
            perDone = i/len(tilesKeep)
            self.threadSignal.emit(perDone)
            
            lidarXYZsmall = RECALL.getLidar_Download(thisFile,IDToDownload,self.cameraLoc_lat,self.cameraLoc_lon,wd)
            
            lidarDat = np.append(lidarDat,lidarXYZsmall,axis=0)

            
        with open(wd+'lidarDat.pkl','wb') as f:
            pickle.dump(lidarDat,f)
            
        self.finishSignal.emit(1)   
        
        print('Thread Done')



class getLidar_PrepChosenSetThread(QThread):

    threadSignal = pyqtSignal('PyQt_PyObject')
    finishSignal = pyqtSignal('PyQt_PyObject')

    def __init__(self,cameraLoc_lat,cameraLoc_lon):
        super().__init__()
        self.cameraLoc_lat = cameraLoc_lat 
        self.cameraLoc_lon = cameraLoc_lon

    def run(self):
        
        print('Thread Started')
                
        f = open(wd+'chosenLidarID.pkl','rb')
        IDToDownload = pickle.load(f)
        sf = RECALL.getLidar_GetShapefile(wd,IDToDownload)
        
        tilesKeep = list()
        i = 0
        for shapeNum in range(0,len(sf)):
            
            i = i+1
            perDone = i/len(sf)
            self.threadSignal.emit(perDone)
            
            out = RECALL.getLidar_SearchTiles(sf,shapeNum,self.cameraLoc_lat,self.cameraLoc_lon)
            if out:
                tilesKeep.append(out)
        

        with open(wd+'tilesKeep.pkl','wb') as f:
            pickle.dump(tilesKeep,f)
            
        self.finishSignal.emit(1)
        
        print('Thread Done')
    
 


class getLidar_ChooseLidarSetWindow(QWidget):
    def __init__(self, data, rows, columns):
        QWidget.__init__(self)
        
        # Left menu box setup #
        bf = QFont()
        bf.setBold(True)
        leftBar1 = QLabel('• Welcome!')
        leftBar2 = QLabel('• Get imagery')
        leftBar3 = QLabel('• Get lidar data')
        leftBar3.setFont(bf)
        leftBar4 = QLabel('• Pick GCPs')
        leftBar5 = QLabel('• Calibrate')
        
        leftGroupBox = QGroupBox('Contents:')
        vBox = QVBoxLayout()
        vBox.addWidget(leftBar1)
        vBox.addWidget(leftBar2)
        vBox.addWidget(leftBar3)
        vBox.addWidget(leftBar4)
        vBox.addWidget(leftBar5)
        vBox.addStretch(1)
        leftGroupBox.setLayout(vBox)
        ########################    
        
        # Right content box setup #
        self.table = QTableWidget(rows, columns, self)
        tblHeaders = ['Choose Dataset','Year Collected','Dataset Name']
        for self.column in range(0,columns):
            for self.row in range(0,rows):
                item = QTableWidgetItem(str(data.iloc[self.row][self.column]))
                if self.column == 0:
                    item.setFlags(Qt.ItemIsUserCheckable |
                                     Qt.ItemIsEnabled)
                    item.setCheckState(Qt.Unchecked)
                self.table.setItem(self.row, self.column, item)
                
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(0,QHeaderView.Stretch)
            header.setSectionResizeMode(1,QHeaderView.ResizeToContents)
            header.setSectionResizeMode(2,QHeaderView.ResizeToContents)
            
        self.table.setHorizontalHeaderLabels(["Dataset ID","Year Collected","Dataset Name"])
        self.table.setSelectionBehavior(QTableView.SelectRows)
        
        self.dir = QLabel('Select the dataset you want to use by checking its box:')
        self.contBut = QPushButton('Continue >')
        self.backBut = QPushButton('< Back')
        
        rightGroupBox = QGroupBox()
        self.layout = QGridLayout(self)
        self.layout.addWidget(self.dir,0,0,1,1)
        self.layout.addWidget(self.table,1,0,4,4)
        self.layout.addWidget(self.contBut,6,3,1,1)
        self.layout.addWidget(self.backBut,6,0,1,1)
        self.layout.setAlignment(Qt.AlignCenter)
        rightGroupBox.setLayout(self.layout)
        ##############################
        
        # Connect widgets to signals #
        self.table.itemClicked.connect(self.dataChoice)
        self.contBut.clicked.connect(self.downloadCorrectData)
        self.backBut.clicked.connect(self.GoBack)
        ##############################
        
         # Full widget layout setup #
        fullLayout = QGridLayout()
        fullLayout.addWidget(leftGroupBox,0,0,2,2)
        fullLayout.addWidget(rightGroupBox,0,2,2,6)
        self.setLayout(fullLayout)

        self.setGeometry(400,100,200,800)
        self.setWindowTitle('RECALL')
        self.show()
        ############################
        
        # Instantiate worker threads #
        f = open(wd+'CameraLocation.pkl','rb')
        cameraLocation = pickle.load(f)
        
        self.worker = getLidar_PrepChosenSetThread(cameraLocation[0],cameraLocation[1])
        self.worker.threadSignal.connect(self.on_threadSignal)
        
        self.worker2 = getLidar_DownloadChosenSetThread(cameraLocation[0],cameraLocation[1])
        self.worker2.threadSignal.connect(self.on_threadSignal2)
        
        self.worker3 = getLidar_FormatChosenSetThread(cameraLocation[0],cameraLocation[1])
        ##############################
        
    def dataChoice(self,item):
        print(str(item.text())) 
        
        num = int(item.text())
        with open(wd+'chosenLidarID.pkl','wb') as f:
            pickle.dump(num,f)
    
    def downloadCorrectData(self):   
        lab1 = QLabel('Sorting tiles:')
        self.pb1 = QProgressBar()
        
        self.layout.removeWidget(self.contBut)
        self.contBut.deleteLater()
        self.contBut = None
        self.layout.removeWidget(self.backBut)
        self.backBut.deleteLater()
        self.backBut = None
        
        self.layout.addWidget(lab1,6,0,1,2)
        self.layout.addWidget(self.pb1,6,2,1,2)
 
        self.worker.start()
        self.worker.finishSignal.connect(self.on_closeSignal)                

    def on_threadSignal(self,perDone):
        self.pb1.setValue(perDone*100)
        
    def on_closeSignal(self):
        lab2 = QLabel('Downloading lidar data near camera:')
        self.pb2 = QProgressBar()
        
        self.layout.addWidget(lab2,7,0,1,2)
        self.layout.addWidget(self.pb2,7,2,1,2)
        
        self.worker2.start()
        self.worker2.finishSignal.connect(self.on_closeSignal2)
        
    def on_threadSignal2(self,perDone):
        self.pb2.setValue(perDone*100)
        
    def on_closeSignal2(self):
        lab3 = QLabel('Creating data point cloud...')
        
        self.layout.addWidget(lab3,8,0,1,2)
        
        self.worker3.start()
        self.worker3.finishSignal.connect(self.on_closeSignal3)
        
    def on_closeSignal3(self):
        labDone = QLabel('Done')
        self.layout.addWidget(labDone,8,2,1,2)
        
        self.label = QLabel('Lidar downloaded! Press continue to pick GCPs:')
        contBut2 = QPushButton('Continue >')
        backBut2 = QPushButton('< Back')

        self.layout.addWidget(self.label,9,0,1,2)
        self.layout.addWidget(contBut2,10,3,1,1)
        self.layout.addWidget(backBut2,10,0,1,1)
        
        contBut2.clicked.connect(self.moveToNext)
        backBut2.clicked.connect(self.GoBack)
        
    def moveToNext(self):
        self.close()
        self.nextWindow = PickGCPsWindow()
        self.nextWindow().show()
        
        
    def GoBack(self):
        self.close()
        self.backToOne = ShowImageWindow()    


class getLidar_SearchThread(QThread):

    threadSignal = pyqtSignal('PyQt_PyObject')
    finishSignal = pyqtSignal('PyQt_PyObject')

    def __init__(self,cameraLoc_lat,cameraLoc_lon):
        super().__init__()
        self.cameraLoc_lat = cameraLoc_lat 
        self.cameraLoc_lon = cameraLoc_lon

    def run(self):
        
        print('Thread Started')
        
        IDs = RECALL.getLidar_GetIDs()
        
        appropID = list() # Initiate list of IDs which contain the camera location #
        i = 0
        for ID in IDs:          
            
            i = i+1
            perDone = i/len(IDs)
            self.threadSignal.emit(perDone)  
            
            tiles = RECALL.getLidar_TryID(ID,self.cameraLoc_lat,self.cameraLoc_lon)
            
            if tiles:
                if len(tiles)>0:       
                    appropID.append(ID)
        
        matchingTable = RECALL.getLidar_GetMatchingNames(appropID)
          
        print('Thread Done')   

        with open(wd+'lidarTable.pkl','wb') as f:
            pickle.dump(matchingTable,f)

        self.finishSignal.emit(1)
    

class getLidar_StartSearchWindow(QWidget):    
     def __init__(self):
        super().__init__()    
        
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()             
        self.initUI()
        
       
     def initUI(self):
              
       # Left menu box setup #
       bf = QFont()
       bf.setBold(True)
       leftBar1 = QLabel('• Welcome!')
       leftBar2 = QLabel('• Get imagery')
       leftBar3 = QLabel('• Get lidar data')
       leftBar3.setFont(bf)
       leftBar4 = QLabel('• Pick GCPs')
       leftBar5 = QLabel('• Calibrate')
       
       leftGroupBox = QGroupBox('Contents:')
       vBox = QVBoxLayout()
       vBox.addWidget(leftBar1)
       vBox.addWidget(leftBar2)
       vBox.addWidget(leftBar3)
       vBox.addWidget(leftBar4)
       vBox.addWidget(leftBar5)
       vBox.addStretch(1)
       leftGroupBox.setLayout(vBox)
       ########################    
       
       # Right contents box setup #
       self.pb = QProgressBar()
       but = QPushButton('Start')
       info = QLabel('Press start to find lidar datsets for this camera:')
       self.val = QLabel('0%')
        
       rightGroupBox = QGroupBox()
       self.grd = QGridLayout()
       self.grd.addWidget(info,0,0,1,6)
       self.grd.addWidget(but,1,1,1,4)
       self.grd.addWidget(self.val,2,0,1,1)
       self.grd.addWidget(self.pb,2,1,1,5)
       self.grd.setAlignment(Qt.AlignCenter)
       rightGroupBox.setLayout(self.grd)
       ##############################
       
       # Assign signals to widgets #
       but.clicked.connect(self.startWorker)
       #############################
      
       # Full widget layout setup #
       fullLayout = QHBoxLayout()
       fullLayout.addWidget(leftGroupBox)
       fullLayout.addWidget(rightGroupBox)
       self.setLayout(fullLayout)

       self.setGeometry(400,100,200,300)
       self.setWindowTitle('RECALL')
       self.show()
       ############################
       
       # Instantiate worker threads #
       f = open(wd+'CameraLocation.pkl','rb')
       cameraLocation = pickle.load(f)
       
       self.worker = getLidar_SearchThread(cameraLocation[0],cameraLocation[1])
       self.worker.threadSignal.connect(self.on_threadSignal)
       self.worker.finishSignal.connect(self.on_closeSignal)
       ##############################
       
     def startWorker(self):
         '''
         Function to start getLidar_SearchThread
         '''
         self.worker.start()                     

     def on_threadSignal(self,perDone):
         '''
         Update progress bar value each time getLidar_SearchThread sends a signal (which is every time it finishes looking at a particular dataset)
         '''
         self.pb.setValue(perDone*100)
         self.val.setText(str(round(perDone*100))+'%')
        
     def on_closeSignal(self):
         '''
         When sorting of lidar datasets is completed, show that it is done and allow the user to click Continue to choose the dataset they want to use.
         '''
         doneInfo = QLabel('Lidar datasets found! Press continue to select the dataset you want to use for remote GCP extraction:')
         doneInfo.setWordWrap(True)
         contBut = QPushButton('Continue >')
         backBut = QPushButton('< Back')
         
         contBut.clicked.connect(self.GoToChooseLidarSet)
         backBut.clicked.connect(self.GoBack)
         
         self.grd.addWidget(doneInfo,4,0,1,6)
         self.grd.addWidget(contBut,5,4,1,2)
         self.grd.addWidget(backBut,5,0,1,2)
         
     def GoToChooseLidarSet(self):
         '''
         When Continue is pushed, open the table of lidar datasets.
         '''
         self.close()
         
         f = open(wd+'lidarTable.pkl','rb')
         lidarTable = pickle.load(f)
         
         self.lw = getLidar_ChooseLidarSetWindow(lidarTable,lidarTable.shape[0],lidarTable.shape[1])
         self.lw.resize(900,350)
         self.lw.show()
         
        
     def GoBack(self):
         '''
         Go back to camera choice window on Back click.
         '''
         self.close()
         self.backToOne = ChooseCameraWindow()    

#=============================================================================#
#=============================================================================#




#=============================================================================#
# Get image module #
#=============================================================================#

class ShowImageWindow(QWidget):
    '''
    Window showing an example image from the downloaded video. Window allows the user to determine weather or not the downloaded video can be used for GCP extraction.
    If yes, the user can click Continue to move on the the lidar acquisition module. If no, the UI will return the user to camera selection window after they click No.
    Note: should make this so, if no, the uer can manually enter a date and time for video download?
    '''
   
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
       ######################
       
       # Make a pixmap out of the image #
       label = QLabel()
       pixmap = QPixmap(frame)
       label.setPixmap(pixmap)
       ##################################
       
       # Define widgets #
       label.resize(pixmap.width(),pixmap.height())
       txt = QLabel('Is this image clear enough to allow feature identification? Pressing "Yes" will launch the lidar data download process.')
       noBut = QPushButton('No')
       yesBut = QPushButton('Yes')
       ##################
       
       # Link widgets with signals #
       yesBut.clicked.connect(self.getLidar_SearchThread)
       #############################
       
       # Set the layout of the window #
       grd = QGridLayout()
       grd.addWidget(label,0,0,4,4)
       grd.addWidget(txt,5,0,1,4)
       grd.addWidget(noBut,6,0,1,1)
       grd.addWidget(yesBut,6,1,1,1)
       
       self.setLayout(grd)
       self.setGeometry(400,100,10,10)
       self.setWindowTitle('RECALL')
       self.show()
       ################################
       
    def getLidar_SearchThread(self):
       '''
       Takes user to the lidar acquisition module on Yes click
       '''
       self.close()
       self.lidar = getLidar_StartSearchWindow()
       self.getLidar_StartSearchWindow.show()
        


class OtherCameraLocationInputWindow(QWidget):
    '''
    Window allowing the user to input necessary info on any (non WebCAT) surfcam, such as location and name. Still working on exactly what info is needed,
    will use an example (Dania Beach?) to help better determine.
    '''
    def __init__(self):
        super().__init__()    
        
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()             
        self.initUI()
        
    def initUI(self):
       
       # Left menu box setup #
       bf = QFont()
       bf.setBold(True)
       leftBar1 = QLabel('• Welcome!')
       leftBar2 = QLabel('• Get imagery')
       leftBar2.setFont(bf)
       leftBar3 = QLabel('• Get lidar data')
       leftBar4 = QLabel('• Pick GCPs')
       leftBar5 = QLabel('• Calibrate')
       
       leftGroupBox = QGroupBox('Contents:')
       vBox = QVBoxLayout()
       vBox.addWidget(leftBar1)
       vBox.addWidget(leftBar2)
       vBox.addWidget(leftBar3)
       vBox.addWidget(leftBar4)
       vBox.addWidget(leftBar5)
       vBox.addStretch(1)
       leftGroupBox.setLayout(vBox)
       ########################    
       
       # Right contents box setup #
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
       
       rightGroupBox = QGroupBox()
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
       grd.setAlignment(Qt.AlignCenter)
       rightGroupBox.setLayout(grd)
       ##############################
       
       # Assign signals to widgets #
       backBut.clicked.connect(self.GoBack)
       contBut.clicked.connect(self.getInputs)
       #############################

       
       # Full widget layout setup #
       fullLayout = QHBoxLayout()
       fullLayout.addWidget(leftGroupBox)
       fullLayout.addWidget(rightGroupBox)
       self.setLayout(fullLayout)

       self.setGeometry(400,100,200,300)
       self.setWindowTitle('RECALL')
       self.show()
       ############################
       
    def GoBack(self):
       '''
       Go back to camera choice window on Back click
       '''
       self.close()
       self.backToOne = ChooseCameraWindow()    
       
    def getInputs(self):
       '''
       Get user-input information on Continue click
       '''
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



class DownloadVidThread(QThread):
    '''
    Worker thread to perform WebCAT video download from online. Uses the GetVideo functon from RECALL.
    '''
    threadSignal = pyqtSignal('PyQt_PyObject')
    finishSignal = pyqtSignal('PyQt_PyObject')

    def __init__(self):
        super().__init__()
        
    def run(self):
        
       print('Thread Started')
       
       f = open(wd+'CameraName.pkl','rb')      
       camToInput = pickle.load(f)
     
       vidFile = RECALL.getImagery_GetVideo(camToInput)
       
       # Deal with Buxton camera name change #
       fs = os.path.getsize(wd+vidFile) # Get size of video file #  
       if camToInput == 'buxtoncoastalcam' and fs<1000:
           vidFile = RECALL.getImagery_GetVideo('buxtonnorthcam')
       #######################################
       
       with open(wd+'vidFile.pkl','wb') as f:
           pickle.dump(vidFile,f)
           
       self.finishSignal.emit(1)   
        
       print('Thread Done')

class DecimateVidThread(QThread):
    ''' 
    Worker thread to decimate obtained video to 20 still-images. Uses the DecimateVideo function from RECALL.
    '''
    finishSignal = pyqtSignal('PyQt_PyObject')

    def __init__(self):
        super().__init__()
        
    def run(self):
        
       print('Thread Started')
       
       f = open(wd+'vidFile.pkl','rb')      
       vidFile = pickle.load(f)
       
       # Decimate the video to 20 still-images #       
       fullVidPth = wd + vidFile           
       RECALL.getImagery_DecimateVideo(fullVidPth)
           
       self.finishSignal.emit(1)   
        
       print('Thread Done')


class WebCATLocationWindow(QWidget):
    '''
    Window allowing the user to choose desired WebCAT camera from dropdown menu.
    '''
   
    def __init__(self):
        super().__init__()    
        
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()             
        self.initUI()
        
    def initUI(self):
       
       # Left menu box setup #
       bf = QFont()
       bf.setBold(True)
       leftBar1 = QLabel('• Welcome!')
       leftBar2 = QLabel('• Get imagery')
       leftBar2.setFont(bf)
       leftBar3 = QLabel('• Get lidar data')
       leftBar4 = QLabel('• Pick GCPs')
       leftBar5 = QLabel('• Calibrate')
        
       leftGroupBox = QGroupBox('Contents:')
       vBox = QVBoxLayout()
       vBox.addWidget(leftBar1)
       vBox.addWidget(leftBar2)
       vBox.addWidget(leftBar3)
       vBox.addWidget(leftBar4)
       vBox.addWidget(leftBar5)
       vBox.addStretch(1)
       leftGroupBox.setLayout(vBox)
       ########################    
       
       # Right contents box setup #
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
       
       self.rightGroupBox = QGroupBox()
       self.grd = QGridLayout()
       self.grd.addWidget(txt,0,0,1,2)
       self.grd.addWidget(opt,1,0,1,2)
       self.grd.addWidget(backBut,2,0,1,1)
       self.grd.addWidget(contBut,2,1,1,1)
       self.grd.setAlignment(Qt.AlignCenter)
       self.rightGroupBox.setLayout(self.grd)
       ############################
       
       # Connect widgets with signals #
       opt.activated.connect(self.getSelected)
       backBut.clicked.connect(self.GoBack)
       contBut.clicked.connect(self.DownloadVidAndExtractStills)
       ################################

       # Full widget layout setup #
       fullLayout = QHBoxLayout()
       fullLayout.addWidget(leftGroupBox)
       fullLayout.addWidget(self.rightGroupBox)
       self.setLayout(fullLayout)

       self.setGeometry(400,100,200,300)
       self.setWindowTitle('RECALL')
       self.show()
       ############################
        
       # Instantiate worker threads #
       self.worker = DownloadVidThread()
       self.worker2 = DecimateVidThread()
       ##############################

    def getSelected(self,item):
       '''
       Function gets saved location of WebCAT camera when it is selected from the combobox
       '''
          
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
       '''
       Function goes back to previous window when Back button is clicked
       '''
       self.close()
       self.backToOne = ChooseCameraWindow()
          
    def DownloadVidAndExtractStills(self):
       '''
       Sets a label that video is downloading when Continue is clicked, and starts a worker thread to download the video
       '''
       lab1 = QLabel('Downloading Video...')
       self.grd.addWidget(lab1,3,0,1,1)
       
       self.worker.start()
       self.worker.finishSignal.connect(self.on_closeSignal)

    def on_closeSignal(self):
       '''
       When download video thread is done, function shows a done label and starts the video decimation worker thread
       '''
       labDone = QLabel('Done.')
       self.grd.addWidget(labDone,3,1,1,1)
       
       lab2 = QLabel('Decimating video to images...')
       self.grd.addWidget(lab2,4,0,1,1)
       
       self.worker2.start()
       self.worker2.finishSignal.connect(self.on_closeSignal2)       
    
    def on_closeSignal2(self):
       '''
       When video decimation thread is complete, function shows a Done label and moves to the next window (ShowImageWindow)
       '''
       self.close()
       self.imWindow = ShowImageWindow()
       self.imWindow.show()
       


class ChooseCameraWindow(QWidget):
    '''
    Window allowing the user to choose weather they want to calibrate a WebCAT surfcam or some other surfcam. Next steps will depend on this choice.
    '''
    def __init__(self):
        super().__init__()
        
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()  
        
        # Left menu box setup #
        bf = QFont()
        bf.setBold(True)
        leftBar1 = QLabel('• Welcome!')
        leftBar2 = QLabel('• Get imagery')
        leftBar2.setFont(bf)
        leftBar3 = QLabel('• Get lidar data')
        leftBar4 = QLabel('• Pick GCPs')
        leftBar5 = QLabel('• Calibrate')
        
        leftGroupBox = QGroupBox('Contents:')
        vBox = QVBoxLayout()
        vBox.addWidget(leftBar1)
        vBox.addWidget(leftBar2)
        vBox.addWidget(leftBar3)
        vBox.addWidget(leftBar4)
        vBox.addWidget(leftBar5)
        vBox.addStretch(1)
        leftGroupBox.setLayout(vBox)
        ########################    
        
        # Right contents box setup #
        t = QLabel('Choose camera type:')
        WebCatOpt = QRadioButton('Select WebCAT camera from list')
        OtherOpt = QRadioButton('Input location of other camera')    
        
        rightGroupBox = QGroupBox()
        vBox2 = QVBoxLayout()
        vBox2.addWidget(t)
        vBox2.addWidget(WebCatOpt)
        vBox2.addWidget(OtherOpt)
        vBox2.setAlignment(Qt.AlignCenter)
        rightGroupBox.setLayout(vBox2)
        ############################
         
        # Connect widgets with signals #
        WebCatOpt.clicked.connect(self.WebCAT_select)
        OtherOpt.clicked.connect(self.Other_select)
        ################################
        
        # Full widget layout setup #
        fullLayout = QHBoxLayout()
        fullLayout.addWidget(leftGroupBox)
        fullLayout.addWidget(rightGroupBox)
        self.setLayout(fullLayout)

        self.setGeometry(400,100,200,300)
        self.setWindowTitle('RECALL')
        self.show()
        ###############################
     
        
    def WebCAT_select(self):
        '''
        If WebCAT camera selected, this funciton will open new window (WebCATLocationWindow) to choose the WebCAT camera
        '''
        self.close()
        self.ww = ww = WebCATLocationWindow()  
        self.ww.show()
    def Other_select(self):
        '''
        If other-type selected, this function will open a new window (OtherCameraLocationInputWindow) to allow input of camera details
        '''
        self.close()
        self.www = OtherCameraLocationInputWindow()
        self.www.show()
 

class WelcomeWindow(QWidget):
    ''' 
    Welcome window with textual information about the tool and a Start button.
    '''
    
    def __init__(self):
        super().__init__()    
        
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()    
            
            
        # Left menu box setup #
        bf = QFont()
        bf.setBold(True)
        leftBar1 = QLabel('• Welcome!')
        leftBar1.setFont(bf)
        leftBar2 = QLabel('• Get imagery')
        leftBar3 = QLabel('• Get lidar data')
        leftBar4 = QLabel('• Pick GCPs')
        leftBar5 = QLabel('• Calibrate')
        
        leftGroupBox = QGroupBox('Contents:')
        vBox = QVBoxLayout()
        vBox.addWidget(leftBar1)
        vBox.addWidget(leftBar2)
        vBox.addWidget(leftBar3)
        vBox.addWidget(leftBar4)
        vBox.addWidget(leftBar5)
        vBox.addStretch(1)
        leftGroupBox.setLayout(vBox)
        ########################    
            

        # Right contents box setup #      
        txt = QLabel('Welcome to the SurFcamera Remote CAlibration Tool (SurfR-CaT)!')
        txt2 = QLabel('Developed in partnership with the Southeastern Coastal Ocean Observing Regional Association (SECOORA), '
                      +'the United States Geological Survey (USGS), and the National Oceanic and Atmospheric administration (NOAA), this tool allows you to calibrate any coastal camera of  '
                      +'known location with accessible video footage. For documentation on the methods employed by the tool, please refer to the GitHub readme (link here). If you have an '
                      +'issue, please post it on the GitHib issues page.')      
        txt2.setWordWrap(True)
        txt3 = QLabel('Press Continue to start calibrating a camera!')
        contBut = QPushButton('Continue >')
        
        rightGroupBox = QGroupBox()
        hBox1 = QHBoxLayout()
        hBox1.addWidget(txt3)
        hBox1.addWidget(contBut)
        vBox = QVBoxLayout()
        vBox.addWidget(txt)
        vBox.addWidget(txt2)
        vBox.addLayout(hBox1)
        #vBox.setAlignment(Qt.AlignCenter)
        rightGroupBox.setLayout(vBox)
        ############################
        
        # Connect widgets with signals #
        contBut.clicked.connect(self.StartTool)
        ################################
        
        # Full widget layout setup #
        fullLayout = QHBoxLayout()
        fullLayout.addWidget(leftGroupBox)
        fullLayout.addWidget(rightGroupBox)
        self.setLayout(fullLayout)

        self.setGeometry(400,100,200,300)
        self.setWindowTitle('SurFR-CaT')
        self.show()
        ###############################
         
    def StartTool(self):
       '''
       Moves to the first window of the tool when Start is selected
       '''
       self.close()
       self.tool = ChooseCameraWindow()
       self.tool.show()



test = WelcomeWindow()









