#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:27:15 2019

@author: matthewconlin
"""

# Import packages and allow for downloading of packages via Pip #
import subprocess
import sys
def pipInstall(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

### Import packages ###
import requests
import os
import cv2 
    


#============================================================================#
# Get WebCAT video #
#============================================================================#

def GetVideo(camToInput,year=2018,month=6,day=3,hour=1000):
    
    """
    Function to download a video clip from a specified WebCAT camera to local directory. The desired year, month, day, and time can be given, 
    however for those not given the function will use default values. If you don't like the video from the default date, an examination of WebCAT
    clips on the website can help determine the desired date/time to use. 
    
    """

    # Add zeros to day and month values if needed #
    if month<10:
        month = '0'+str(month)
    else:
        month = str(month)
    
    if day<10:
        day = '0'+str(day)
    else:
        day = str(day)
       
    # Get the video # 
    url = 'http://webcat-video.axds.co/{}/raw/{}/{}_{}/{}_{}_{}/{}.{}-{}-{}_{}.mp4'.format(camToInput,year,year,month,year,month,day,camToInput,year,month,day,hour)   
    
    # Read and load the video file from that URL using requests library
    filename = url.split('/')[-1] # Get the filename as everything after the last backslash #
    r = requests.get(url,stream = True) # Create the Response Object, which contains all of the information about the file and file location %
    with open(filename,'wb') as f: # This loop does the downloading 
        for chunk in r.iter_content(chunk_size = 1024*1024):
            if chunk:
                f.write(chunk)
    
    ## The specified video is now saved to the directory ##
    
    # Get the video file name #
    vidFile = camToInput+'.'+str(year)+'-'+month+'-'+day+'_'+str(hour)+'.mp4'
       
    return vidFile


#=============================================================================#
# Decimate the video to some still images #
#=============================================================================#

def DecimateVideo(vidPth):
    
    """
    Function to decimate WebCAT video clip downloaded with RECALL_GetVideo into still images so that a still image may be pulled
    for remote-GCP extraction. Function pulls 20 equally-spaced frames from the 10 minute video.
    
    """
    
    # Import packages #
#    pipInstall('opencv-python')
#    import cv2
    
    vid = cv2.VideoCapture(vidPth)

    # Find the number of frames in the video #
    vidLen = int(vid.get(7))
   
    # Pull 20 frames evenly distributed through the 10 minute video and save them to the video directory #
    for count in range(0,vidLen,int(vidLen/20)):
        vid.set(1,count) #Set the property that we will pull the frame numbered by count #  
        test,image = vid.read()
        cv2.imwrite('frame'+str(count)+'.png', image)


#============================================================================#
# Find lidar datasets that cover the camera's location #
#============================================================================#
def FindLidarObs(cameraLoc_lat,cameraLoc_lon):
    """
    Function which will search for existing lidar datsets which cover the FOV of the camera at location
    given by cameraLat and cameraLon. The function will give the user all datasets, and will ask the user
    to choose one to donwload.
    """
        
    
    
    # Download specified file from FTP to local file #
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
    for ID in IDs:
        
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
                    if float(line.split()[1][0:7]) <= cameraLoc_lon <= float(line.split()[2][0:7]) and float(line.split()[3][0:7])<= cameraLoc_lat <= float(line.split()[4][0:7]):
                        tiles.append(line)
    
            if len(tiles)>0:       
                appropID.append(ID)
                
    return appropID



#=============================================================================#
# Allow user to select the dataset that they want to use #
#=============================================================================#

def SelectDesiredLidarSet(appropID):
    
    #pipInstall('pandas')
    import pandas as pd
    import requests
    import re
    
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
    # Display the data frame and wait for input of dataset to use #
    print(matchingTable)
    matchingTable_IDUse = input('Select the dataset which you would like to use. Specify as the ID in the left-most column above: ')
    
    curDir = os.getcwd()
    curDir = curDir+'/'
    
    # Get the chosen ID #
    IDToDownload = matchingTable.loc[int(matchingTable_IDUse),'ID']
    
    return IDToDownload,curDir
    
#============================================================================#
# Download the lidar dataset #
#============================================================================#
    
def DownloadLidarSet(IDToDownload,curDir,cameraLoc_lat,cameraLoc_lon):
    
    import ftplib
    #pipInstall('pyshp')
    import shapefile
    #pipInstall('utm')
    import utm
    import numpy
    import math
    import json    
    import pdal

    
    # Establish the location of the camera in UTM coordinates #
    cameraLoc_UTMx = utm.from_latlon(cameraLoc_lat,cameraLoc_lon)[0]
    cameraLoc_UTMy = utm.from_latlon(cameraLoc_lat,cameraLoc_lon)[1]

    
    ftp = ftplib.FTP('ftp.coast.noaa.gov',timeout=1000000)
    ftp.login('anonymous','anonymous')
    ftp.cwd('/pub/DigitalCoast/lidar2_z/geoid12b/data/'+str(IDToDownload))
    files = ftp.nlst()
    
    
    # Now that we have the dataset, we need to search the dataset for tiles which are near the camera. Otherwise,
    # we will be loading a whole lot of useless data into memory, which takes forever. #
    
    # Load the datset shapefile and dbf file from the ftp. These describe the tiles #
    print('Getting shapefile')
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
    sf = shapefile.Reader(curDir+"shapefileCreate.shp")
    
    # Loop through all of the tiles to find the ones close to the camera #
    print('Determining tiles to keep')
    tilesKeep = list()
    for shapeNum in range(0,len(sf)):

        
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
    
    

    # Finally, extract data from the identified laz files using PDAL and save to variable #
    print('Downloading tiles')
    ftp = ftplib.FTP('ftp.coast.noaa.gov',timeout=1000000)
    ftp.login('anonymous','anonymous')
    ftp.cwd('/pub/DigitalCoast/lidar2_z/geoid12b/data/'+str(IDToDownload))
    i = 0
    lidarDat = numpy.empty([0,3])
    for thisFile in tilesKeep:
        i = i+1
        
        print('Working on tile ' + str(i)+' of '+str(len(tilesKeep))+'. Data variable up to '+str(len(lidarDat))+ ' points.')
        # Save the laz file locally - would prefer not to do this, but can't seem to get the pipeline to download directly from the ftp??? #
        gfile = open('lazfile.laz','wb') # Create the local file #
        ftp.retrbinary('RETR '+thisFile,gfile.write) # Copy the contents of the file on FTP into the local file #
        gfile.close() # Close the remote file #
            
        # Construct the json PDAL pipeline to read the file and take only points within +-.5 degree x and y of the camera. Read the data in as an array #
        fullFileName = curDir+'lazfile.laz'
        pipeline=(json.dumps([{'type':'readers.las','filename':fullFileName},{'type':'filters.range','limits':'X['+str(cameraLoc_lon-.5)+':'+str(cameraLoc_lon+.5)+'],Y['+str(cameraLoc_lat-.5)+':'+str(cameraLoc_lat+.5)+']'}],sort_keys=False,indent=4))
            
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
            dlon = math.radians(abs(px)) - math.radians(abs(cameraLoc_lon))
            dlat = math.radians(abs(py)) - math.radians(abs(cameraLoc_lat))
            a = math.sin(dlat/2)**2 + math.cos(math.radians(abs(py))) * math.cos(math.radians(abs(cameraLoc_lat))) * math.sin(dlon/2)**2
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
    
        del lidarX
        del lidarY
        del lidarZ
        del lidarXsmall
        del lidarYsmall
        del lidarZsmall
        del lidarXYZsmall
        del datArrays
        del dist    
        
    return lidarDat        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        