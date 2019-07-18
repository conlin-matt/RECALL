#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:36:31 2019

@author: matthewconlin
"""

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
        for ID in IDs[0:5]:
            
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
        print('Thread Done')           
#        return appropID
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
       self.val = QLabel()

       but.clicked.connect(self.startWorker)
            
       grd = QGridLayout()
       grd.addWidget(self.val,0,0,1,1)
       grd.addWidget(self.pb,1,0,1,1)
       grd.addWidget(but,2,0,1,1)

       self.setLayout(grd)      
       self.setGeometry(400,100,200,250)
       self.setWindowTitle('RECALL')
       self.show()
       
       self.worker = StartLidarDownload(29.856559,-81.265545)
       self.worker.threadSignal.connect(self.on_threadSignal)
       self.worker.finishSignal.connect(self.close)

       
     def startWorker(self):
        self.worker.start()                     

     def on_threadSignal(self,perDone):
        self.pb.setValue(perDone*100)
        self.val.setText(str(round(perDone*100))+'%')



test = GetLidarWindow()    
       
   