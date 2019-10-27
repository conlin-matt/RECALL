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
def getImagery_GetVideo(camToInput,year=2018,month=11,day=3,hour=1000):
    
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
 

#=============================================================================#
# Check if the camera is a PTZ camera #
#=============================================================================#
def getImagery_CheckPTZ(vidPth,numErosionIter):
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    import cv2 
    import os
    import pandas as pd

    # Get the video capture #
    vid = cv2.VideoCapture(vidPth)
    
    # Find the number of frames in the video #
    vidLen = int(vid.get(7))

    # Calc horizon angle of each frame #
    psis = np.array([])
    frameNum = np.array([])
    for count in range(0,vidLen,int(vidLen/1000)):
        vid.set(1,count) #Set the property that we will pull the frame numbered by count #  
        test,image = vid.read()
        
        # Erode the image (morphological erosion) #
        kernel = np.ones((5,5),np.uint8)
        imeroded = cv2.erode(image,kernel,iterations = numErosionIter)

        # Find edges using Canny Edge Detector #
        edges = cv2.Canny(imeroded,50,100)
    
        # Find longest straight edge with Hough Transform #
        lines = cv2.HoughLines(edges,1,np.pi/180,200)
        if lines is not None:
            for rho,theta in lines[0]: # The first element is the longest edge #
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
            
            # Calc horizon angle (psi) #
            psi = math.atan2(y1-y2,x2-x1)
        
            # Save horizon angle and the number of the frame #
            psis = np.append(psis,psi)
            frameNum = np.append(frameNum,count)
        
    
    # Round angles to remove small fluctuations, and take abs #
    psis = np.round(abs(psis),3)

    # Find the frames where calculated angle changes #
    dif = np.diff(psis)
    changes = np.array(np.where(dif!=0))

    # Calculate the length of each angle segment between when the angle changes.
    if np.size(changes)>0:
        segLens = np.array([])
        vals = np.array([])
        for i in range(0,len(changes[0,:])+1):
            if i == len(changes[0,:]):
                segLen = len(psis)-(changes[0,i-1]+1)
                val = psis[len(psis)-1]    
            elif changes[0,i] == changes[0,0]:
                segLen = changes[0,i]
                val = psis[0]
            else:
                segLen = changes[0,i]-changes[0,i-1]
                val = psis[changes[0,i-1]+1]
                
            segLens = np.append(segLens,segLen)
            vals = np.append(vals,val)
            
        # Keep only chunks that are continuous over a threshold #    
        IDs_good = np.array(np.where(segLens>=10)) # Using 10 seems to work, but this could be changed #
        valsKeep = vals[IDs_good]
    
        # Find the unique views #
        viewAngles = np.unique(valsKeep)

        # Find and extract the frames contained within each view #
        frameVec = np.array(range(0,vidLen,int(vidLen/1000)))
        angles = []
        frames = []
        for i in viewAngles:
            iFrames = frameVec[np.array(np.where(psis == i))]
            angles.append(i)
            frames.append([iFrames])
        
        viewDict = {'View Angles':angles,'Frames':frames}
        viewDF = pd.DataFrame(viewDict)
        
        return viewDF,frameVec
    
    else:
   
        # Find the unique views #
        viewAngles = np.unique(psis)
    
        # Find and extract the frames contained within each view #
        frameVec = np.array(range(0,vidLen,int(vidLen/1000)))
        angles = []
        frames = []
        for i in viewAngles:
            iFrames = np.array(np.where(psis == i))
            angles.append(i)
            frames.append([iFrames])
            
        viewDict = {'View Angles':angles,'Frames':frames}
        viewDF = pd.DataFrame(viewDict)
        
        return viewDF,frameVec

    

def getImagery_SeperateViewsAndGetFrames(vidPth,viewDF):
    import pandas as pd
    
    numViews = len(viewDF)
    vid = cv2.VideoCapture(vidPth)
    
    frameDF = pd.DataFrame(columns=['View','Image'])
    for i in range(0,numViews):
        frameTake = viewDF['Frames'][i][0][0][1]
        
        vid.set(1,frameTake) # Set the property that we will pull the desired frame #  
        test,image = vid.read()
        
        frameDF = frameDF.append({'View':i,'Image':image},ignore_index=True)
        
    return frameDF


#=============================================================================#
# Decimate the video to some still images #
#=============================================================================#
def getImagery_ExtractStills(vidPth,viewDF,frameVec):
    
    """
    Function to decimate WebCAT video clip downloaded with RECALL_GetVideo into still images so that a still image may be pulled
    for remote-GCP extraction. Function pulls 20 equally-spaced frames from the 10 minute video.
    
    """
    
    # Import packages #
#    pipInstall('opencv-python')
#    import cv2
    
    vid = cv2.VideoCapture(vidPth)

    # Find the number of frames in the video #
    frames
   
    # Pull 20 frames evenly distributed through the 10 minute video and save them to the video directory #
    for count in range(0,vidLen,int(vidLen/20)):
        vid.set(1,count) #Set the property that we will pull the frame numbered by count #  
        test,image = vid.read()
        cv2.imwrite('frame'+str(count)+'.png', image)
#=============================================================================#




#=============================================================================#
# Search for and identify lidar datasets that cover camera location #
#=============================================================================#
def getLidar_GetIDs():
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
    return IDs



def getLidar_TryID(ID,cameraLoc_lat,cameraLoc_lon):
    import ftplib
    ftp = ftplib.FTP('ftp.coast.noaa.gov',timeout=1000000)
    ftp.login('anonymous','anonymous')
    ftp.cwd('/pub/DigitalCoast/lidar2_z/geoid12b/data/'+str(ID))  
        
    # Find the minmax csv file which shows the min and max extents of each tile within the current dataset #
    files = ftp.nlst()
    fileWant = str([s for s in files if "minmax" in s])
    if fileWant:
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
    
            return tiles


def getLidar_GetMatchingNames(appropID):
    import pandas as pd
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
    return matchingTable

#=============================================================================#


#=============================================================================#
# Prepare and download the chosen dataset
#=============================================================================#
def getLidar_GetShapefile(wd,IDToDownload):
    import ftplib
    #pipInstall('pyshp')
    import shapefile


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
    
    return sf



def getLidar_SearchTiles(sf,shapeNum,cameraLoc_lat,cameraLoc_lon):
    #pipInstall('utm')
    import utm
    import math
    import numpy
    
    # Establish the location of the camera in UTM coordinates #
    cameraLoc_UTMx = utm.from_latlon(cameraLoc_lat,cameraLoc_lon)[0]
    cameraLoc_UTMy = utm.from_latlon(cameraLoc_lat,cameraLoc_lon)[1]
    
    # See if the tile is near the camera #
    bx = sf.shape(shapeNum).bbox # Get the bounding box #
    # Get the bounding box verticies in utm. bl = bottom-left, etc. #
    bx_bl = utm.from_latlon(bx[1],bx[0]) 
    bx_br = utm.from_latlon(bx[1],bx[2]) 
    bx_tr = utm.from_latlon(bx[3],bx[2]) 
    bx_tl = utm.from_latlon(bx[3],bx[0]) 
    # Min distance between camera loc and horizontal lines connecting tile verticies #
    line_minXbb = numpy.array([numpy.linspace(bx_bl[0],bx_br[0],num=1000),numpy.linspace(bx_bl[1],bx_br[1],num=1000)])
    line_maxXbb = numpy.array([numpy.linspace(bx_tl[0],bx_tr[0],num=1000),numpy.linspace(bx_tl[1],bx_tr[1],num=1000)])
    
    # Distance from camera to midpoint of tile #
    meanX = numpy.mean(numpy.array([line_minXbb[0,:],line_maxXbb[0,:]]))
    meanY = numpy.mean(numpy.array([line_minXbb[1,:],line_maxXbb[1,:]]))
    dist = math.sqrt((meanX-cameraLoc_UTMx)**2 + (meanY-cameraLoc_UTMy)**2)
    
    # Distance from camera to edges of tile #
    dist1 = list()
    dist2 = list()
    for ixMin,iyMin,ixMax,iyMax in zip(line_minXbb[0,:],line_minXbb[1,:],line_maxXbb[0,:],line_maxXbb[1,:]):
        dist1.append(math.sqrt((ixMin-cameraLoc_UTMx)**2 + (iyMin-cameraLoc_UTMy)**2))
        dist2.append(math.sqrt((ixMax-cameraLoc_UTMx)**2 + (iyMax-cameraLoc_UTMy)**2))
    
    # If either distance is <350 m, keep the tile. This ensures close tiles are kept and the tile containing the camera is kept. #
    try:
        rec = sf.record(shapeNum)
        if min(dist1)<600 or min(dist2)<600:
            return rec['Name']
    except:
        pass
    
    


def getLidar_Download(thisFile,IDToDownload,cameraLoc_lat,cameraLoc_lon,wd):
    import ftplib
    import numpy
    import math
    import json    
    import pdal
          
    ftp = ftplib.FTP('ftp.coast.noaa.gov',timeout=1000000)
    ftp.login('anonymous','anonymous')
    ftp.cwd('/pub/DigitalCoast/lidar2_z/geoid12b/data/'+str(IDToDownload))
           
    # Save the laz file locally - would prefer not to do this, but can't seem to get the pipeline to download directly from the ftp??? #
    gfile = open('lazfile.laz','wb') # Create the local file #
    ftp.retrbinary('RETR '+thisFile,gfile.write) # Copy the contents of the file on FTP into the local file #
    gfile.close() # Close the remote file #
        
    # Construct the json PDAL pipeline to read the file and take only points within +-.5 degree x and y of the camera. Read the data in as an array #
    fullFileName = wd+'lazfile.laz'
    pipeline=(json.dumps([{'type':'readers.las','filename':fullFileName},{'type':'filters.range','limits':'X['+str(cameraLoc_lon-.5)+':'+str(cameraLoc_lon+.5)+'],Y['+str(cameraLoc_lat-.5)+':'+str(cameraLoc_lat+.5)+']'}],sort_keys=False,indent=4))
        
    # Go through the pdal steps to use the pipeline
    r = pdal.Pipeline(pipeline)  
    r.validate()  
    r.execute()
        
    # Get the arrays of data and format them so we can use them #
    datArrays = r.arrays
    datArrays = datArrays[int(0)] # All of the fields are now accessable with the appropriate index #
    
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
    
    return lidarXYZsmall



def getLidar_CreatePC(lidarDat,cameraLoc_lat,cameraLoc_lon):   
    #pipInstall('utm')
    import utm
    import numpy 
    import pandas as pd
    
    pc = pd.DataFrame({'x':lidarDat[:,0],'y':lidarDat[:,1],'z':lidarDat[:,2]})

    # Convert eveything to UTM and translate to camera at (0,0) #
    utmCoordsX = list()
    utmCoordsY = list()
    for ix,iy in zip(pc['x'],pc['y']):
        utmCoords1 = utm.from_latlon(iy,ix)
        utmCoordsX.append( utmCoords1[0] )
        utmCoordsY.append( utmCoords1[1] )
    utmCoords = numpy.array([utmCoordsX,utmCoordsY])
        
    utmCam = utm.from_latlon(cameraLoc_lat,cameraLoc_lon)
        
    # Translate to camera position #
    utmCoords[0,:] = utmCoords[0,:]-utmCam[0]
    utmCoords[1,:] = utmCoords[1,:]-utmCam[1]
   
    # Put these new coordinates into the point cloud %
    pc['x'] = numpy.transpose(utmCoords[0,:])
    pc['y'] = numpy.transpose(utmCoords[1,:])
    
    return pc

#=============================================================================#


#=============================================================================#
# Perform Calibration #
#=============================================================================#
def calibrate_getInitialApprox_IOPs(img):
    '''
    Get initial approximatation for camera IOPs (focal length (f) and principal points (x0,y0)) using the geometry of the image.
    - Estimate focal length by assuming a horizontal field of view of 60 degrees (typical of webcams), and using simple geometry with this and the width of the image.
    - Estimate principal points as the center of the image.
    '''
    
    import math
    w = len(img[1,:,:])
    a = math.radians(60)
    f = (w/2)*(1/math.tan(a/2))
    
    x0 = len(img[1,:,1])/2
    y0 = len(img[:,1,1])/2
    
    return f,x0,y0

def calibrate_getInitialApprox_op(horizonPts,f,ZL):
    '''
    Get initial approximation for camera look-angles (omega, phi, kappa) using the horizon, following Sanchez-Garcia et al. (2017).
    '''
    
    import math
    xa = horizonPts[0][0]
    xb = horizonPts[1][0]
    ya = horizonPts[0][1]
    yb = horizonPts[1][1]
    psi = math.atan2(ya-yb,xb-xa)
    
    d = ((ya*xb)-(yb*xa))/math.sqrt( (xb-xa)**2+(yb-ya)**2 )
    C = math.atan2(d,f)
    D = math.sqrt( (ZL+6371000)**2 - (6371000**2) )
    beta = math.acos( (ZL+ (.42*(D**2)/6371000))/D )
    xi = beta-C
    
    phi = -math.asin( math.sin(xi)*math.sin(psi) )
    omega = math.acos( math.cos(xi)/math.sqrt((math.cos(psi)**2) + (math.cos(xi)**2 * math.sin(psi)**2)) )
  
    return omega,phi,xi,psi


def calibrate_PerformSpaceResection_CPro(initApproxVec,gcps_lidar,gcps_im,x0,y0,xi,psi):
    import math
    import numpy as np
    from scipy.optimize import least_squares
    
    omega1 = initApproxVec
    
    
    def model(unknownsVec):
        omega = unknownsVec[0]
        phi = unknownsVec[1]
        kappa = unknownsVec[2]
        XL = unknownsVec[3]
        YL = unknownsVec[4]
        ZL = unknownsVec[5]
        f = unknownsVec[6]
        
        m11 = math.cos(phi)*math.cos(kappa)
        m12 = (math.sin(omega)*math.sin(phi)*math.cos(kappa)) + (math.cos(omega)*math.sin(kappa))
        m13 = (-math.cos(omega)*math.sin(phi)*math.cos(kappa)) + (math.sin(omega)*math.sin(kappa))
        m21 = -math.cos(phi)*math.sin(kappa)
        m22 = (-math.sin(omega)*math.sin(phi)*math.sin(kappa)) + (math.cos(omega)*math.cos(kappa))
        m23 = (math.cos(omega)*math.sin(phi)*math.sin(kappa)) + (math.sin(omega)*math.cos(kappa))
        m31 = math.sin(phi)
        m32 = -math.sin(omega)*math.cos(phi)
        m33 = math.cos(omega)*math.cos(phi)
        
    #    P = np.vstack([[1,0,0,0],[0,1,0,0],[0,0,-1/f,0]])
    #    R = np.vstack([[m11,m12,m13,0],[m21,m22,m23,0],[m31,m32,m33,0],[0,0,0,1]])
    #    T = np.vstack([[1,0,0,-XL],[0,1,0,YL],[0,0,1,-ZL],[0,0,0,1]])
    #    XYZ = np.vstack([np.transpose(gcps_lidar),np.ones(len(gcps_lidar))])
    #    
    #    xyw = ((P@R)@T)@XYZ
    #    
    #    xy = np.divide(xyw,xyw[2,:])[0:2,:]
    #    uv = np.subtract(np.vstack([x01,y01]),xy)
    #    
    #    H1 = math.acos(math.cos(phi)*math.cos(omega))-xi
    #    H2 = math.atan2(-math.sin(phi),math.cos(phi)*math.sin(omega))-psi
    #    uv = np.hstack([uv,np.hstack([np.vstack([H1,H1]),np.vstack([H2,H2])])])
        
        u = np.empty([0,1])
        v = np.empty([0,1])
        for i in range(0,len(gcps_lidar)):
            XA = gcps_lidar[i,0]
            YA = gcps_lidar[i,1]
            ZA = gcps_lidar[i,2]
            
            dx = XA-XL
            dy = YA-YL
            dz = ZA-ZL
            
            u1 = x01-(f*(((m11*dx)+(m12*dy)+(m13*dz))/((m31*dx)+(m32*dy)+(m33*dz))))
            v1 = y01-(f*(((m21*dx)+(m22*dy)+(m23*dz))/((m31*dx)+(m32*dy)+(m33*dz))))
            
            u = np.vstack([u,u1])
            v = np.vstack([v,v1])
        
        uv = np.hstack([u,v])
       
        
        return uv
        
    def calcResid(unknownsVec,observations):
        uv = model(unknownsVec)
        
    #    observations = np.transpose(observations)
    #    observations = np.hstack([observations,np.vstack([0,0]),np.vstack([0,0])])
    
        resid_uv = np.subtract(observations,uv)
        resid = np.reshape(resid_uv,[np.size(resid_uv)])
        resid1d = np.sqrt(np.add(resid_uv[0,:]**2,resid_uv[1,:]**2))
        
        
        return resid
        
    initApprox = np.hstack([omega1,phi1,kappa1,XL1,YL1,ZL1,f1])
    boundsVec = ((-math.pi*2,-math.pi*2,-math.pi*2,XL1-50,YL1-50,0,f1-500),(math.pi*2,math.pi*2,math.pi*2,XL1+50,YL1+50,math.inf,f1+500))
    
    results = least_squares(calcResid,initApprox,bounds=boundsVec,jac='3-point',method='dogbox',max_nfev=5000,x_scale='jac',loss='cauchy',tr_solver='exact',args=[gcps_im])
    finalVals = results['x']
    finalResid = results['fun']
    cpe = math.sqrt(finalVals[3]**2 + finalVals[4]**2 + (45-finalVals[5])**2)
    
    return results

def calibrate_CalcReprojPos(gcps_lidar,calibResults,x0,y0):
    import math
    import numpy as np
    
    # Get the final parameters and the calculated control point residuals #
    finalVals = calibResults['x']
    finalResid = calibResults['fun']    

    omega = finalVals[0]
    phi = finalVals[1]
    kappa = finalVals[2]
    XL = finalVals[3]
    YL = finalVals[4]
    ZL = finalVals[5]
    f = finalVals[6]
    
    
    # Calculate the projected position of each GCP based on final vals #
    m11 = math.cos(phi)*math.cos(kappa)
    m12 = (math.sin(omega)*math.sin(phi)*math.cos(kappa)) + (math.cos(omega)*math.sin(kappa))
    m13 = (-math.cos(omega)*math.sin(phi)*math.cos(kappa)) + (math.sin(omega)*math.sin(kappa))
    m21 = -math.cos(phi)*math.sin(kappa)
    m22 = (-math.sin(omega)*math.sin(phi)*math.sin(kappa)) + (math.cos(omega)*math.cos(kappa))
    m23 = (math.cos(omega)*math.sin(phi)*math.sin(kappa)) + (math.sin(omega)*math.cos(kappa))
    m31 = math.sin(phi)
    m32 = -math.sin(omega)*math.cos(phi)
    m33 = math.cos(omega)*math.cos(phi)
    
    u = np.empty([0,1])
    v = np.empty([0,1])
    for i in range(0,len(gcps_lidar)):
        XA = gcps_lidar[i,0]
        YA = gcps_lidar[i,1]
        ZA = gcps_lidar[i,2]
        
        deltaX = XA-XL
        deltaY = YA-YL
        deltaZ = ZA-ZL
        
        q = (m31*deltaX)+(m32*deltaY)+(m33*deltaZ)
        r = (m11*deltaX)+(m12*deltaY)+(m13*deltaZ)
        s = (m21*deltaX)+(m22*deltaY)+(m23*deltaZ)
            
            u1 = x0-(f*(r/q))
            v1 = y0-(f*(s/q))
        
        u1 = x0 - (f*(((m11*(XA-XL)) + (m12*(YA-YL)) + (m13*(ZA-ZL))) / ((m31*(XA-XL)) + (m32*(YA-YL)) + (m33*(ZA-ZL)))))
        v1 = y0 - (f*(((m21*(XA-XL)) + (m22*(YA-YL)) + (m23*(ZA-ZL))) / ((m31*(XA-XL)) + (m32*(YA-YL)) + (m33*(ZA-ZL)))))
        
        u = np.vstack([u,u1])
        v = np.vstack([v,v1])
        
    return u,v








