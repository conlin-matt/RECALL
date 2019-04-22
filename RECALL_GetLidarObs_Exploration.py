#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:22:40 2019

@author: matthewconlin
"""

# Download specified file from FTP to local file #
import ftplib
import sys


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


# Loop through each dataset to see if it captures where the camera can see #
appropID = list()
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
    fileWant = fileWant[2:len(fileWant)-2]
    # If there is more than one csv file, take only the first #
    if len(fileWant.split()) == 2:
        fileWant = fileWant.split()[1]
    
    # Save the minmax file locally #
    gfile = open('minmax.csv','wb') # Create the local file #
    ftp.retrbinary('RETR '+fileWant,gfile.write) # Copy the contents of the file on FTP into the local file #
    gfile.close()
    
    # See if the location of the camera is contained within any of the tiles in this dataset #
    with open('minmax.csv') as infile:
        next(infile)
        for line in infile:
            if float(line.split()[1][0:7]) <= cameraLoc_lon <= float(line.split()[2][0:7]) and float(line.split()[3][0:7])<= cameraLoc_lat <= float(line.split()[4][0:7]):
                appropID.append(ID)
        

# Display the names of the sets that contain the data and choose one # 












ftp.close() 






      
    
    
    

