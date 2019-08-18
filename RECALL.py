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
def getImagery_CheckPTZ(vidPth):
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
        imeroded = cv2.erode(image,kernel,iterations = 2)

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

def calibrate_GetInitialEstimate(GCPs_im,GCPs_lidar,horizonPts,cameraElev,cameraDir):
    import numpy as np
    import math
    from scipy.optimize import least_squares
    import os
    import glob
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg


    ### Get initial parameter estimation using Direct Linear Transform, as implemented in the online lecture ###

    # Create the M matrix and fill it with GCP info #
    M = np.zeros([(3*len(GCPs_im)),(13+((len(GCPs_im)-1)*4))])

    startRow = -3
    for i in range(0,6):
        startRow = startRow+3
        
        # Lidar GCP coordinates #
        Xx = GCPs_lidar[i,0]
        Xy = GCPs_lidar[i,1]
        Xz = GCPs_lidar[i,2]
        # Image GCP coordinates #
        x = GCPs_im[i,0]
        y = GCPs_im[i,1]
        camVec = [-x,-y,-1]
        
        # Put the coordinates in the correct places in the M matrix #
        for subi in range(0,3):
            zeroSpace = subi*4
            
            M[startRow+subi,0+zeroSpace] = Xx
            M[startRow+subi,1+zeroSpace] = Xy
            M[startRow+subi,2+zeroSpace] = Xz
            M[startRow+subi,3+zeroSpace] = 1
            
            M[startRow+subi,12+(i*4)] = camVec[subi]

    # Make the norm of the M matrix = 1 by dividing it by its current norm #
    M = M/np.linalg.norm(M)        
    
    # Create the MTM matrix #
    MTM = np.transpose(M) @ M
    
    # SVD on MTM and take the eigenvector with the smallest eigenvalue as the solution for v #
    u,s,v = np.linalg.svd(MTM)
    vStar = v[:,len(v)-1]
    
    # Create P as the first 12 elements of the eigenvector, in a 3x4 shape (?????? is this correct ????????) #
    P = np.vstack((np.transpose(vStar[0:4]),np.transpose(vStar[4:8]),np.transpose(vStar[8:12])))



    ### Factor P into K by using RQ factorization ###
    
    # Get A and a. I think this is correct????? #
    A = P[:,0:3]
    a = P[:,3]
    
    A1 = A[0,:]
    A2 = A[1,:]
    A3 = A[2,:]
    
    # Solve the third row #
    f = np.linalg.norm(A3)
    R3 = (1/f)*A3
    # Solve the second row #
    e = np.dot(A2,R3)/np.linalg.norm(R3)
    rhs = A2-(e*R3)
    d = np.linalg.norm(rhs)
    R2 = (1/d)*(rhs)
    # Solve the first row # ## Not sure if this is correct... ##
    b = np.dot(A1,R2)/np.linalg.norm(R2)
    c = np.dot(A1,R3)/np.linalg.norm(R3)
    
    lhs = A1-(b*R2)-(c*R3)
    a = np.linalg.norm(lhs)
    R1 = 1/a*(lhs)

    # Create the intrinsic (k), rotation (R) and, and translation (t) matricies. Need to normalize k by element (3,3) at the end. DLT
    # has given us initial estimations of all the unknowns. #
    k = np.vstack((np.array([a,b,c]),np.array([0,d,e]),np.array([0,0,f])))
    k = k/k[2,2]
    R = np.vstack([R1,R2,R3])
    t = np.dot(np.linalg.inv(k),P)[:,3]


    # Define t better by asking for user input for elevation estimate #
    t = [0,0,float(cameraElev)]

    # Make R better using the horizon #
    xa = horizonPts[0][0]
    ya = horizonPts[0][1]
    xb = horizonPts[1][0]
    yb = horizonPts[1][1]

    # Solve for the two angles (psi and xi) using the horizon #
    psi = math.atan2(ya-yb,xb-xa)
    dhorizon = ((ya-xb)-(yb-xa))/math.sqrt((xb-xa)**2+(yb-ya)**2)
    Cc = np.arctan(dhorizon/k[1,1])
    Rt = 6371000
    D = math.sqrt((t[2]+Rt)**2-Rt**2)
    beta = np.arccos((t[2]+(.42*(D**2/Rt)))/D)
    xi = beta-Cc

# Now compute the three camera orientation angles from the two horizon angles #
    phi = -np.arcsin(math.sin(xi)*math.sin(psi))
    omega = np.arccos(math.cos(xi)/(math.sqrt(math.cos(psi)+(math.cos(xi)**2*math.sin(psi)))))
    kk = cameraDir
    if kk == 1:
        kappa = -math.pi/4
    elif kk == 2:
        kappa = -math.pi*3/4
    elif kk == 3:
        kappa = math.pi/4
    else:
        kappa = math.pi*3/4
    
    # Put the angles into the Holland convention -- not sure which correspond to which. #
    tau = omega
    sigma = phi
    phi = kappa


    # Re-dedine rotation matrix using these angles and equations in Holland et al. (1997) #
    r11 = (math.cos(phi)*math.cos(sigma))+(math.sin(phi)*math.cos(tau)*math.sin(sigma))
    r12 = (-math.sin(phi)*math.cos(sigma))+(math.cos(phi)*math.cos(tau)+math.sin(sigma))
    r13 = math.sin(tau)*math.sin(sigma)
    r21 = (-math.cos(phi)*math.sin(sigma))+(math.sin(phi)*math.cos(tau)*math.cos(sigma))
    r22 = (math.sin(phi)*math.sin(sigma))+(math.cos(phi)*math.cos(tau)*math.cos(sigma))
    r23 = math.sin(tau)*math.cos(sigma)
    r31 = math.sin(phi)*math.sin(tau)
    r32 = math.cos(phi)*math.sin(tau)
    r33 = -math.cos(tau)
    R = np.vstack([[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]])
    
    return t,k,R



def calcResid_withDistortion(toOptVec,GCPs_lidar,GCPs_im):
    import numpy as np
    resid = 0
    for i in range(0,len(GCPs_im)): 
        ki = np.vstack([[toOptVec[0],toOptVec[1],toOptVec[2]],[toOptVec[3],toOptVec[4],toOptVec[5]],[toOptVec[6],toOptVec[7],toOptVec[8]]])
        Ri = np.vstack([toOptVec[9:12],toOptVec[12:15],toOptVec[15:18]])
        ti = toOptVec[18:21]
        Xw = np.append(np.array(GCPs_lidar[i,:]),1)
        Xc = GCPs_im[i,:]
        
        # Project world to image with current guess of calibration params #
        t = [toOptVec[18],toOptVec[19],toOptVec[20]]
        Pi = np.dot(ki,np.c_[Ri,t])
        uv = np.dot(Pi,Xw)
        u = uv[0]/uv[2]
        v = uv[1]/uv[2]
        
        # Rigid transformation from world coords to camera coords #
        camCoords = np.dot(Ri,Xw[0:3])+ti
        # Perspective transformation from 3d camera coords to ideal image coords #
        x = toOptVec[4]*(camCoords[0]/camCoords[2])
        y = toOptVec[4]*(camCoords[1]/camCoords[2])
        # Distort the projected image coords #
        dx = .2
        dy = .2
        u = x/dx
        v = y/dy
        
        uHat = u+((u-toOptVec[2])*((toOptVec[21]*(x**2+y**2))+(toOptVec[22]*(x**2+y**2)**2)))
        vHat = v+((v-toOptVec[5])*((toOptVec[21]*(x**2+y**2))+(toOptVec[22]*(x**2+y**2)**2)))
        
        # Calculate the residual between projected/distorted image coords and measured image coords #
        residV = np.array([Xc[0]-uHat,Xc[1]-vHat]) 
        residi = np.linalg.norm(residV)**2
        
        resid = resid+residi
        
    residToReturn = np.tile(resid,[23])
    
    return residToReturn


def calibrate_OptimizeEstimate(t,k,R,GCPs_im,GCPs_lidar):
    import numpy as np
    from scipy.optimize import least_squares

    toOptVec = np.array([k[0,0],k[0,1],k[0,2],k[1,0],k[1,1],k[1,2],k[2,0],k[2,1],k[2,2],R[0,0],R[0,1],R[0,2],R[1,0],R[1,1],R[1,2],R[2,0],R[2,1],R[2,2],t[0],t[1],t[2],0,0]) # Need to put everything into a vector for the function to work #

    # Optimize the parameters using the Levenberg-Marquardt algorithm #
    out = least_squares(calcResid_withDistortion,toOptVec,args=(GCPs_lidar,GCPs_im),method='lm',max_nfev=100000,xtol=None)
    optVec = out['x']

    # Build optimized k, R, and t and distortion coeffs #
    Kopt = np.vstack([[optVec[0],optVec[1],optVec[2]],[optVec[3],optVec[4],optVec[5]],[optVec[6],optVec[7],optVec[8]]])
    Ropt = np.vstack([[optVec[9],optVec[10],optVec[11]],[optVec[12],optVec[13],optVec[14]],[optVec[15],optVec[16],optVec[17]]])
    topt = np.array([optVec[18],optVec[19],optVec[20]])
    k1 = optVec[21]
    k2 = optVec[22]
    
    return Kopt,Ropt,topt,k1,k2



def calibrate_GetPointProjection(Kopt,Ropt,topt,k1,k2,Xw,Xc):
    import numpy as np
 
    # Project world to image #
    Ptest = np.dot(Kopt,np.c_[Ropt,topt])
    uv = np.dot(Ptest,Xw)
    uProj = uv[0]/uv[2]
    vProj = uv[1]/uv[2]
        
    # Distort #
    # Rigid transformation from world coords to camera coords #
    camCoords = np.dot(Ropt,Xw[0:3])+topt
    # Perspective transformation from 3d camera coords to ideal image coords #
    x = Kopt[1,1]*(camCoords[0]/camCoords[2])
    y = Kopt[1,1]*(camCoords[1]/camCoords[2])
    dx = .2
    dy = .2
    uProj = x/dx
    vProj = y/dy
    uProjD = uProj+((uProj-Kopt[0,2])*((k1*(x**2+y**2))+(k2*(x**2+y**2)**2)))
    vProjD = vProj+((vProj-Kopt[1,2])*((k1*(x**2+y**2))+(k2*(x**2+y**2)**2)))
        
    return uProjD,vProjD





