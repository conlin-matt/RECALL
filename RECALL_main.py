#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:16:23 2019

@author: matthewconlin
"""


from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap,QIcon
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


wd = '/Users/matthewconlin/Documents/Research/WebCAT/'
                       
    
class ChooseLidarWindow(QAbstractTableModel):
    def __init__(self, data, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=None):
        return len(self._data.values)

    def columnCount(self, parent=None):
        return self._data.columns.size

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.values[index.row()][index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role ==Qt.DisplayRole:
            return self._data.columns[col]
        return None



class StartLidarDownload(QThread):

    threadSignal = pyqtSignal('PyQt_PyObject')
    finishSignal = pyqtSignal('PyQt_PyObject')

    def __init__(self,cameraLoc_lat,cameraLoc_lon):
        super().__init__()
        self.cameraLoc_lat = cameraLoc_lat 
        self.cameraLoc_lon = cameraLoc_lon

    def run(self):
#        print('Starting Thread')
#        for ii in range(1,self.i,1):
#            perDone = ii/self.i            
#            self.threadSignal.emit(perDone)
#        print('Thread Done')
        
        
        print('Thread Started')
        import ftplib
        import re
        # First, pull the numeric IDs from all datasets which exist #
        ftp = ftplib.FTP('ftp.coast.noaa.gov',timeout=1000000)
        ftp.login('anonymous','anonymous')
        ftp.cwd('/pub/DigitalCoast/lidar2_z/geoid12b/data/')
        IDs = ftp.nlst()
        # Get rid of spurious IDs which have letters
        IDsGood = list()
        for tryThisString in IDs:
            testResult = re.search('[a-zA-Z]',tryThisString) # Use regular expressions to see if any letters exist in the string #
            if testResult:
                pass
            else:
                IDsGood.append(tryThisString)
        IDs = IDsGood
    
    
        # Loop through all datasets to see which capture where the camera can see. Store the datasets that do. #
        appropID = list() # Initiate list of IDs which contain the camera location #
        i = 0
        for ID in IDs:
            
            i = i+1
            perDone = i/len(IDs)
            self.threadSignal.emit(perDone)
            
            # Get the bounds of all of the regions in the current set #
            ftp.cwd('/pub/DigitalCoast/lidar2_z/geoid12b/data/'+str(ID))  
            
            # Find the minmax csv file which shows the min and max extents of each tile within the current dataset #
            files = ftp.nlst()
            fileWant = str([s for s in files if "minmax" in s])
            
            if len(fileWant)>2:
                # Get the file name and save it. We need to get rid of the ' or " in the name. Sometimes this means we need to get rid of the first 2 characters, sometimes the first 3 #
                if len(fileWant.split()) == 2:
                    fileWant = '['+fileWant.split()[1]
                fileWant = fileWant[2:len(fileWant)-2]
                # Save the file locally #
                gfile = open('minmax.csv','wb') # Create the local file #
                ftp.retrbinary('RETR '+fileWant,gfile.write) # Copy the contents of the file on FTP into the local file #
                gfile.close() # Close the remote file #
            
            
                # See if the location of the camera is contained within any of the tiles in this dataset. If it is, save the ID #
                tiles = list()
                with open('minmax.csv') as infile:
                    next(infile)
                    for line in infile:
                        if float(line.split()[1][0:7]) <= self.cameraLoc_lon <= float(line.split()[2][0:7]) and float(line.split()[3][0:7])<= self.cameraLoc_lat <= float(line.split()[4][0:7]):
                            tiles.append(line)
        
                if len(tiles)>0:       
                    appropID.append(ID)
        
        
        # Get the data tabel on NOAAs website #
        url = 'https://coast.noaa.gov/htdata/lidar1_z/'
        html = requests.get(url).content
        df_list = pd.read_html(html)
        dataTable = df_list[-1]
        # Make a list of all IDs and names #   
        IDlist = dataTable.loc[:,'ID #']
        nameList = dataTable.loc[:,'Dataset Name']    
        # Find the indicies in the data table that match the appropriate IDs # 
        appropIDNums = list(map(int,appropID))  
        matchingTableRows = [i for i, x in enumerate(IDlist) for j,y in enumerate(appropIDNums) if x==y] # Get indicies of matching IDs in the dataTable
        # Create a new data frame with data for the appropriate IDs #
        matchingTable = pd.DataFrame(columns=['ID','Year Collected','Name'])
        matchingTable.loc[:,'ID'] = IDlist[matchingTableRows]
        matchingTable.loc[:,'Year Collected'] = dataTable.loc[:,'Year'][matchingTableRows]
        matchingTable.loc[:,'Name'] = nameList[matchingTableRows]
          
        print('Thread Done')   

        with open(wd+'lidarTable.pkl','wb') as f:
            pickle.dump(matchingTable,f)

        self.finishSignal.emit(1)
    

class GetLidarWindow(QWidget):    
     def __init__(self):
        super().__init__()    
        
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()             
        self.initUI()
        
       
     def initUI(self):
       self.pb = QProgressBar()
       but = QPushButton('Start')
       info = QLabel('Press start to find lidar datsets for this camera:')
       self.val = QLabel('0%')

       but.clicked.connect(self.startWorker)
            
       self.grd = QGridLayout()
       self.grd.addWidget(info,0,0,1,6)
       self.grd.addWidget(but,1,1,1,4)
       self.grd.addWidget(self.val,2,0,1,1)
       self.grd.addWidget(self.pb,2,1,1,5)

       self.setLayout(self.grd)      
       self.setGeometry(400,100,200,100)
       self.setWindowTitle('RECALL')
       self.show()
       
       self.worker = StartLidarDownload(29.856559,-81.265545)
       self.worker.threadSignal.connect(self.on_threadSignal)
       self.worker.finishSignal.connect(self.on_closeSignal)

       
     def startWorker(self):
        self.worker.start()                     

     def on_threadSignal(self,perDone):
        self.pb.setValue(perDone*100)
        self.val.setText(str(round(perDone*100))+'%')
        
     def on_closeSignal(self):
         doneInfo = QLabel('Lidar datasets found! Press continue to select the dataset you want to use for remote GCP extraction:')
         doneInfo.setWordWrap(True)
         contBut = QPushButton('Continue >')
         backBut = QPushButton('< Back')
         
         contBut.clicked.connect(self.GoToChooseLidarSet)
         backBut.clicked.connect(self.GoBack)
         
         self.grd.addWidget(doneInfo,4,1,1,6)
         self.grd.addWidget(contBut,5,4,1,2)
         self.grd.addWidget(backBut,5,0,1,2)
         self.setGeometry(400,100,200,250)
         self.show()
         
     def GoToChooseLidarSet(self):
         self.close()
         
         f = open(wd+'lidarTable.pkl','rb')
         lidarTable = pickle.load(f)
         
         self.view = QTableView()
         header = self.view.horizontalHeader()
         header.setSectionResizeMode(QHeaderView.ResizeToContents)
         self.view.resizeColumnsToContents()

         
         self.model = ChooseLidarWindow(lidarTable)
         self.view.setModel(self.model)
         self.view.show()

        
     def GoBack(self):
         self.close()
         self.backToOne = ChooseCameraWindow()    

         


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

       label.resize(pixmap.width(),pixmap.height())
       txt = QLabel('Is this image clear enough to allow feature identification? Pressing "Yes" will launch the lidar data download process.')
       noBut = QPushButton('No')
       yesBut = QPushButton('Yes')
       
       yesBut.clicked.connect(self.StartLidarDownload)
       
       grd = QGridLayout()
       grd.addWidget(label,0,0,4,4)
       grd.addWidget(txt,5,0,1,4)
       grd.addWidget(noBut,6,0,1,1)
       grd.addWidget(yesBut,6,1,1,1)
       
       self.setLayout(grd)
       self.setGeometry(400,100,10,10)
       self.setWindowTitle('RECALL')
       self.show()
       
   def StartLidarDownload(self):
       self.close()
       self.lidar = GetLidarWindow()
       self.GetLidarWindow.show()
        

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











