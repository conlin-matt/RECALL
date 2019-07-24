#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 10:16:23 2019

@author: matthewconlin
"""


from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap,QIcon,QMouseEvent
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

wd = '/Users/matthewconlin/Documents/Research/WebCAT/'
                       


       class PickGCPsWindow(QWidget):
   def __init__(self):
        super().__init__()    
        
        if not QApplication.instance():
            app = QApplication(sys.argv)
        else:
            app = QApplication.instance()             
        #self.initUI()
        
        self.figure = plt.figure()
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
        
        self.goBut.clicked.connect(self.getPoints1)
        
        self.grd = QGridLayout()
        self.grd.addWidget(self.introLab,0,0,2,4)
        self.grd.addWidget(self.canvas,2,0,4,4)
        self.grd.addWidget(self.goLab,7,0,1,1)
        self.grd.addWidget(self.goBut,7,3,1,1)
        
        self.setLayout(self.grd)
        
        self.setWindowTitle('RECALL')
        self.show()

        
   def getPoints1(self):
       print('In Function')

       while self.grd.count() > 0:
           item = self.grd.takeAt(0)
           if not item:
               continue

           w = item.widget()
           if w:
               w.deleteLater()

#       self.grd.removeWidget(self.introLab)
#       self.introLab.deleteLater()
#       self.introLab = None
#       self.grd.removeWidget(self.goLab)
#       self.goLab.deleteLater()
#       self.goLab = None
#       self.grd.removeWidget(self.goBut)
#       self.goBut.deleteLater()
#       self.goBut = None

       
       self.dirLab = QLabel('Click on the point in the image:')
       self.grd.addWidget(self.dirLab,0,0,1,2)
       
       pt = plt.ginput(show_clicks=True)   
       print(pt)
       
       self.afterClick()
       
   def afterClick(self):
       print('In Function')
       self.grd.removeWidget(self.dirLab)
       self.dirLab.deleteLater()
       self.dirLab = None
       
       savedLab = QLabel('Image coordinate of point saved!')
       dirLab2 = QLabel('Now, identify the point lidar point cloud (click Help for directions). When done, return here and Continue (to pick more) or Stop (to finish picking).')
       contBut = QPushButton('Continue')
       stopBut = QPushButton('Stop')
       helpBut = QPushButton('Help')
       
       self.grd.addWidget(savedLab,0,0,1,2)
       self.grd.addWidget(dirLab2,1,0,1,2)
       self.grd.addWidget(stopBut,7,2,1,1)
       self.grd.addWidget(contBut,7,3,1,1)
       self.grd.addWidget(helpBut,7,0,1,1)


        
        
        

class CreateLidarPC(QThread):   
    
    import pandas as pd 
    
#    threadSignal = pyqtSignal('PyQt_PyObject')
    finishSignal = pyqtSignal('PyQt_PyObject')

    def __init__(self,cameraLoc_lat,cameraLoc_lon):
        super().__init__()
        self.cameraLoc_lat = cameraLoc_lat
        self.cameraLoc_lon = cameraLoc_lon
        
    def run(self):
        
        f = open(wd+'lidarDat.pkl','rb')
        lidarDat = pickle.load(f)
       
        # Turn the numpy array into a Pandas data frame #
        pc = pd.DataFrame({'x':lidarDat[:,0],'y':lidarDat[:,1],'z':lidarDat[:,2]})
        
        
        # Convert eveything to UTM and translate to camera at (0,0) #
        #pipInstall('utm')
        import utm
        import numpy
        utmCoordsX = list()
        utmCoordsY = list()
        for ix,iy in zip(pc['x'],pc['y']):
            utmCoords1 = utm.from_latlon(iy,ix)
            utmCoordsX.append( utmCoords1[0] )
            utmCoordsY.append( utmCoords1[1] )
        utmCoords = numpy.array([utmCoordsX,utmCoordsY])
            
        utmCam = utm.from_latlon(self.cameraLoc_lat,self.cameraLoc_lon)
            
        # Translate to camera position #
        utmCoords[0,:] = utmCoords[0,:]-utmCam[0]
        utmCoords[1,:] = utmCoords[1,:]-utmCam[1]
        
            
        # Put these new coordinates into the point cloud %
        pc['x'] = numpy.transpose(utmCoords[0,:])
        pc['y'] = numpy.transpose(utmCoords[1,:])

            
        with open(wd+'lidarPC.pkl','wb') as f:
            pickle.dump(pc,f)
            
        self.finishSignal.emit(1)    
        
        print('Thread Done')
        

class DownloadLidar(QThread):

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
        import numpy
        import math
        import json    
        import pdal
        
        f = open(wd+'tilesKeep.pkl','rb')
        tilesKeep = pickle.load(f)
        
        f = open(wd+'chosenLidarID.pkl','rb')
        IDToDownload = pickle.load(f)
    
        
        ftp = ftplib.FTP('ftp.coast.noaa.gov',timeout=1000000)
        ftp.login('anonymous','anonymous')
        ftp.cwd('/pub/DigitalCoast/lidar2_z/geoid12b/data/'+str(IDToDownload))
        i = 0
        lidarDat = numpy.empty([0,3])
        for thisFile in tilesKeep:
            
            i = i+1
            perDone = i/len(tilesKeep)
            self.threadSignal.emit(perDone)
           
            # Save the laz file locally - would prefer not to do this, but can't seem to get the pipeline to download directly from the ftp??? #
            gfile = open('lazfile.laz','wb') # Create the local file #
            ftp.retrbinary('RETR '+thisFile,gfile.write) # Copy the contents of the file on FTP into the local file #
            gfile.close() # Close the remote file #
                
            # Construct the json PDAL pipeline to read the file and take only points within +-.5 degree x and y of the camera. Read the data in as an array #
            fullFileName = wd+'lazfile.laz'
            pipeline=(json.dumps([{'type':'readers.las','filename':fullFileName},{'type':'filters.range','limits':'X['+str(self.cameraLoc_lon-.5)+':'+str(self.cameraLoc_lon+.5)+'],Y['+str(self.cameraLoc_lat-.5)+':'+str(self.cameraLoc_lat+.5)+']'}],sort_keys=False,indent=4))
                
            # Go through the pdal steps to use the pipeline
            r = pdal.Pipeline(pipeline)  
            r.validate()  
            r.execute()
                
            # Get the arrays of data and format them so we can use them #
            datArrays = r.arrays
            datArrays = datArrays[int(0)] # All of the fields are now accessable with the appropriate index #
              # allDatArrays.append(datArrays) 
            
            # Extract x,y,z values #
            lidarX = datArrays['X']
            lidarY = datArrays['Y']
            lidarZ = datArrays['Z']
        
            # Only take points within 500 m of the camera #
            R = 6373000 # ~radius of Earth in m #
            dist = list()
            for px,py in zip(lidarX,lidarY):
                dlon = math.radians(abs(px)) - math.radians(abs(self.cameraLoc_lon))
                dlat = math.radians(abs(py)) - math.radians(abs(self.cameraLoc_lat))
                a = math.sin(dlat/2)**2 + math.cos(math.radians(abs(py))) * math.cos(math.radians(abs(self.cameraLoc_lat))) * math.sin(dlon/2)**2
                c = 2*math.atan2(math.sqrt(a),math.sqrt(1-a))
                dist.append(R*c)
           
            lidarXsmall = list()
            lidarYsmall = list()
            lidarZsmall = list()    
            for xi,yi,zi,di in zip(lidarX,lidarY,lidarZ,dist):
                if di<300:
                    lidarXsmall.append(xi)
                    lidarYsmall.append(yi)
                    lidarZsmall.append(zi)
            lidarXYZsmall = numpy.vstack((lidarXsmall,lidarYsmall,lidarZsmall))
            lidarXYZsmall = numpy.transpose(lidarXYZsmall)
            
            lidarDat = numpy.append(lidarDat,lidarXYZsmall,axis=0)

            
        with open(wd+'lidarDat.pkl','wb') as f:
            pickle.dump(lidarDat,f)
            
        self.finishSignal.emit(1)   
        
        print('Thread Done')



class GetCorrectLidarSetThread(QThread):

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
        #pipInstall('pyshp')
        import shapefile
        #pipInstall('utm')
        import utm
        import numpy
        import math
        
        f = open(wd+'chosenLidarID.pkl','rb')
        IDToDownload = pickle.load(f)
    
        
        # Establish the location of the camera in UTM coordinates #
        cameraLoc_UTMx = utm.from_latlon(self.cameraLoc_lat,self.cameraLoc_lon)[0]
        cameraLoc_UTMy = utm.from_latlon(self.cameraLoc_lat,self.cameraLoc_lon)[1]
    
        
        ftp = ftplib.FTP('ftp.coast.noaa.gov',timeout=1000000)
        ftp.login('anonymous','anonymous')
        ftp.cwd('/pub/DigitalCoast/lidar2_z/geoid12b/data/'+str(IDToDownload))
        files = ftp.nlst()
        
        
        # Now that we have the dataset, we need to search the dataset for tiles which are near the camera. Otherwise,
        # we will be loading a whole lot of useless data into memory, which takes forever. #
        
        # Load the datset shapefile and dbf file from the ftp. These describe the tiles #
        shpFile = str([s for s in files if "shp" in s])
        shpFile = shpFile[2:len(shpFile)-2]
        dbfFile = str([s for s in files if "dbf" in s])
        dbfFile = dbfFile[2:len(dbfFile)-2]
        
        # Write them locally so we can work with them #
        gfile = open('shapefileCreate.shp','wb') # Create the local file #
        ftp.retrbinary('RETR '+shpFile,gfile.write)
        
        gfile = open('shapefileCreate.dbf','wb') # Create the local file #
        ftp.retrbinary('RETR '+dbfFile,gfile.write)
        
        # Load them into an object using the PyShp library #
        sf = shapefile.Reader(wd+"shapefileCreate.shp")
        
        # Loop through all of the tiles to find the ones close to the camera #
        tilesKeep = list()
        i = 0
        for shapeNum in range(0,len(sf)):
            
            i = i+1
            perDone = i/len(sf)
            self.threadSignal.emit(perDone)

            
            bx = sf.shape(shapeNum).bbox # Get the bounding box #
            # Get the bounding box verticies in utm. bl = bottom-left, etc. #
            bx_bl = utm.from_latlon(bx[1],bx[0]) 
            bx_br = utm.from_latlon(bx[1],bx[2]) 
            bx_tr = utm.from_latlon(bx[3],bx[2]) 
            bx_tl = utm.from_latlon(bx[3],bx[0]) 
            # Min distance between camera loc and horizontal lines connecting tile verticies #
            line_minXbb = numpy.array([numpy.linspace(bx_bl[0],bx_br[0],num=1000),numpy.linspace(bx_bl[1],bx_br[1],num=1000)])
            line_maxXbb = numpy.array([numpy.linspace(bx_tl[0],bx_tr[0],num=1000),numpy.linspace(bx_tl[1],bx_tr[1],num=1000)])
            dist1 = list()
            dist2 = list()
            for ixMin,iyMin,ixMax,iyMax in zip(line_minXbb[0,:],line_minXbb[1,:],line_maxXbb[0,:],line_maxXbb[1,:]):
                dist1.append(math.sqrt((ixMin-cameraLoc_UTMx)**2 + (iyMin-cameraLoc_UTMy)**2))
                dist2.append(math.sqrt((ixMax-cameraLoc_UTMx)**2 + (iyMax-cameraLoc_UTMy)**2))
            # Keep the tile if min distance to either of lines meets criterion #
            try:
                rec = sf.record(shapeNum)
                if min(dist1)<300 or min(dist2)<300:
                    tilesKeep.append(rec['Name'])
            except:
                pass
            
        with open(wd+'tilesKeep.pkl','wb') as f:
            pickle.dump(tilesKeep,f)
            
        self.finishSignal.emit(1)
        print('Thread Done')
    
 


class ChooseLidarWindow(QWidget):
    def __init__(self, data, rows, columns):
        QWidget.__init__(self)
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

        self.table.itemClicked.connect(self.dataChoice)

        self.dir = Qlabel('Select the dataset you want to use by checking its box:')
        self.contBut = QPushButton('Continue >')
        self.backBut = QPushButton('< Back')
        
        self.contBut.clicked.connect(self.downloadCorrectData)
        self.backBut.clicked.connect(self.GoBack)
        
        
        self.layout = QGridLayout(self)
        self.layout.addWidget(self.dir,0,0,4,4)
        self.layout.addWidget(self.table,1,0,4,4)
        self.layout.addWidget(self.contBut,6,3,1,1)
        self.layout.addWidget(self.backBut,6,0,1,1)
        
        f = open(wd+'CameraLocation.pkl','rb')
        cameraLocation = pickle.load(f)
        
        self.worker = GetCorrectLidarSetThread(cameraLocation[0],cameraLocation[1])
        self.worker.threadSignal.connect(self.on_threadSignal)
        
        self.worker2 = DownloadLidar(cameraLocation[0],cameraLocation[1])
        self.worker2.threadSignal.connect(self.on_threadSignal2)
        
        self.worker3 = CreateLidarPC(cameraLocation[0],cameraLocation[1])
        
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
        for ID in IDs[1:30]:
            
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
       
       f = open(wd+'CameraLocation.pkl','rb')
       cameraLocation = pickle.load(f)
       
       self.worker = StartLidarDownload(cameraLocation[0],cameraLocation[1])
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
         
         self.grd.addWidget(doneInfo,4,0,1,6)
         self.grd.addWidget(contBut,5,4,1,2)
         self.grd.addWidget(backBut,5,0,1,2)
         self.setGeometry(400,100,200,250)
         self.show()
         
     def GoToChooseLidarSet(self):
         self.close()
         
         f = open(wd+'lidarTable.pkl','rb')
         lidarTable = pickle.load(f)
         
         self.chooseLidarWindow = ChooseLidarWindow(lidarTable,lidarTable.shape[0],lidarTable.shape[1])
         self.chooseLidarWindow.resize(900,350)
         self.chooseLidarWindow.show()
         
        
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



class DownloadVidThread(QThread):
    threadSignal = pyqtSignal('PyQt_PyObject')
    finishSignal = pyqtSignal('PyQt_PyObject')

    def __init__(self):
        super().__init__()
        
    def run(self):
        
       print('Thread Started')
       
       f = open(wd+'CameraName.pkl','rb')      
       camToInput = pickle.load(f)
       
       vidFile = RECALL.GetVideo(camToInput)
       
       with open(wd+'vidFile.pkl','wb') as f:
           pickle.dump(vidFile,f)
           
       self.finishSignal.emit(1)   
        
       print('Thread Done')

class DecimateVidThread(QThread):
    finishSignal = pyqtSignal('PyQt_PyObject')

    def __init__(self):
        super().__init__()
        
    def run(self):
        
       print('Thread Started')
       
       f = open(wd+'vidFile.pkl','rb')      
       vidFile = pickle.load(f)
       
       # Decimate the video to 20 still-images #       
       fullVidPth = wd + vidFile           
       RECALL.DecimateVideo(fullVidPth)
           
       self.finishSignal.emit(1)   
        
       print('Thread Done')


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
       
       self.grid = QGridLayout()
       
       self.grid.addWidget(txt,0,1,1,4)
       self.grid.addWidget(opt,1,1,1,4)
       self.grid.addWidget(backBut,2,1,1,2)
       self.grid.addWidget(contBut,2,3,1,2)

       
       self.setLayout(self.grid)
    
       self.setGeometry(400,100,300,100)
       self.setWindowTitle('RECALL')
       self.show()
       
       self.worker = DownloadVidThread()

       self.worker2 = DecimateVidThread()

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
       
       lab1 = QLabel('Downloading Video...')
       self.grid.addWidget(lab1,3,1,1,2)
       
       self.worker.start()
       self.worker.finishSignal.connect(self.on_closeSignal)

   def on_closeSignal(self):
       
       labDone = QLabel('Done.')
       self.grid.addWidget(labDone,3,3,1,1)
       
       lab2 = QLabel('Decimating video to images...')
       self.grid.addWidget(lab2,4,1,1,2)
       
       self.worker2.start()
       self.worker2.finishSignal.connect(self.on_closeSignal2)       
    
   def on_closeSignal2(self):
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











