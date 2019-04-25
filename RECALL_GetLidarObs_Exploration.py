#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:22:40 2019

@author: matthewconlin
"""

# Download specified file from FTP to local file #
import ftplib
import sys
import requests
import pandas as pd
import re


# Function to draw progress bar while code is working #
def drawProgressBar(percent, barLen = 20):
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
ftp = ftplib.FTP('ftp.coast.noaa.gov','anonymous','conlinm@ufl.edu')


# First, get all of the IDs which exist #
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


########### Download the desired dataset #####################

import pdal
import os

curDir = os.getcwd()
curDir = curDir+'/'

# Get the chosen ID #
IDToDownload = matchingTable.loc[int(matchingTable_IDUse),'ID']
# Go back to the FTP and download the chosen ID #
ftp = ftplib.FTP('ftp.coast.noaa.gov','anonymous','conlinm@ufl.edu')
ftp.cwd('/pub/DigitalCoast/lidar2_z/geoid12b/data/'+str(IDToDownload))
files = ftp.nlst()

# Extract data from each laz file and append it all together #
allDatArrays = list()
for thisFile in files:
    fileExt = thisFile.split('.')[1] # Get the file extension so that we can skip over any files that aren't .laz files #
    if fileExt == 'laz':
        fullFileName = 'ftp.coast.noaa.gov/pub/DigitalCoast/lidar2_z/geoid12b/data/'+str(IDToDownload)+ '/'+ files[thisFile]
        # Save the laz file locally - would prefer not to do this, but can't seem to get the pipeline to download directly from the ftp??? #
        gfile = open('lazfile.laz','wb') # Create the local file #
        ftp.retrbinary('RETR '+files[thisFile],gfile.write) # Copy the contents of the file on FTP into the local file #
        gfile.close() # Close the remote file #
        
        # Construct the json PDAL pipeline to read the file and read it in as an array #
        pipeline="""{
        "pipeline": [
                {
                        "type": "readers.las",
                        "filename": "/Users/matthewconlin/Documents/Research/WebCAT/lazfile.laz"
                        }
                ]
        }"""

    r = pdal.Pipeline(pipeline)  
    r.validate()  
    r.execute()
    datArrays = r.arrays
    datArrays = datArrays[int(0)] # All of the fields are now accessable with the appropriate index #
    allDatArrays.append(datArrays)
 
    
# Extract x,y,z values #
allDatArrays = allDatArrays[int(0)]
lidarX = datArrays['X']
lidarY = datArrays['Y']
lidarZ = datArrays['Z']

import matplotlib
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D

def scatter3d(x,y,z, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, c=scalarMap.to_rgba(cs))
    scalarMap.set_array(cs)
    fig.colorbar(scalarMap)
    plt.show()

scatter3d(lidarX,lidarY,lidarZ,lidarZ)



