#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:22:40 2019

@author: matthewconlin
"""


# Need to figure out a way to install pdal from within this script. Need a conda equivalent of the pipInstall function below,
# and need to be able to specify the channel. Also need to download gdal seperately (I think). 


# Create function to install package using pip #
def pipInstall(package):
    import subprocess
    import sys
    subprocess.call([sys.executable, "-m", "pip", "install", package])
    

# Download specified file from FTP to local file #
import ftplib
import requests
import re
import pdal
import os
import json
import numpy
import math

#pipInstall('pyshp')
import shapefile

#pipInstall('pandas')
import pandas as pd

pipInstall('utm')
import utm


# Function to draw progress bar while code is working #
def drawProgressBar(percent, barLen = 20):
    import sys
    sys.stdout.write("\r")
    progress = ""
    for i in range(barLen):
        if i < int(barLen * percent):
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[ %s ] %.2f%%" % (progress, percent * 100))
    sys.stdout.flush()


# Inputs #
cameraLoc_lat = 25.810
cameraLoc_lon = -80.122
cameraLoc_UTMx = utm.from_latlon(cameraLoc_lat,cameraLoc_lon)[0]
cameraLoc_UTMy = utm.from_latlon(cameraLoc_lat,cameraLoc_lon)[1]


# First, get all of the IDs which exist #
ftp = ftplib.FTP('ftp.coast.noaa.gov','anonymous','conlinm@ufl.edu')
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

# Loop through each dataset to see if it captures where the camera can see. Store the datasets which do. #
appropID = list() # Initiate list of IDs which contain the camera location #
i = 0
for ID in IDs:
    i = i+1
    IDuse = int(ID)
    drawProgressBar(i/len(IDs),20)
    
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


########### Select the desired dataset by linking IDs to names ############
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
# Go back to the FTP #
ftp = ftplib.FTP('ftp.coast.noaa.gov','anonymous','conlinm@ufl.edu')
ftp.cwd('/pub/DigitalCoast/lidar2_z/geoid12b/data/'+str(IDToDownload))
files = ftp.nlst()



################ Get the shapefile of all the tiles so we can get rid of any tiles not near the camera #########################
# Load the tile shapefile and dbf file from the ftp #
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

# Loop through all of the tiles to find the one containing the camera #
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
        if min(dist1)<500 or min(dist2)<500:
            tilesKeep.append(rec['Name'])
    except:
        pass

    



########### Extract data from the appropriate laz file(s)  #####################
allDatArrays = list()
lidarDat = numpy.empty([0,3])
i = 0
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
#allDatArrays = allDatArrays[int(0)]
    lidarX = datArrays['X']
    lidarY = datArrays['Y']
    lidarZ = datArrays['Z']

#numpy.savetxt(curDir+'lidarXYZfile.txt',lidarXYZ)

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
        if di<500:
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
        
        
        
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(lidarX,lidarY,s=1,c=lidarZ,cmap='terrain')
plt.plot(cameraLoc_lon,cameraLoc_lat,'.',c='k')
plt.colorbar()
plt.axis('equal')

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(lidarX,lidarY,lidarZ,c=lidarZ)


