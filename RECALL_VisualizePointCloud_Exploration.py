#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:30:45 2019

@author: matthewconlin
"""




def pipInstall(package):
    import subprocess
    import sys
    subprocess.call([sys.executable, "-m", "pip", "install", package])
    
def condaInstall(package,channel=None):
    import conda.cli
    if channel:
        conda.cli.main('conda','install','-c',channel,package)
    else:
        conda.cli.main('conda','install','y',package)
 
    
condaInstall(package = 'eigen',channel = 'conda-forge') # Can't get this to work, need to figure it out #
pipInstall('tbb')
pipInstall('pptk')

import os
import pptk
import numpy as np
import pandas as pd
import math
import glob
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Turn the numpy array into a Pandas data frame #
pc = pd.DataFrame({'x':lidarDat[:,0],'y':lidarDat[:,1],'z':lidarDat[:,2]})



# Transform lidar data to local coordinates using the Argus method. Do this by manually dertmining the orientation angle. 
# If this whole process works, we will need an automated way to estimate orientation e.g. using the horizon #

# Define the angle of the beach #
theta = 130 # Obtained by giput in Matlab imshow, then doing what Nathaniel did inupdateDVTsite.m (lines 74-75)

curd = os.getcwd() + '/TempForVideo'
os.chdir(curd)
imgs = glob.glob('*.png')
img = cv2.imread(curd+'/'+imgs[1])
fig = cv2.imshow('image',img)

img = mpimg.imread(curd+'/'+imgs[1])
imgplot = plt.imshow(img)
pts = plt.ginput(n=1,show_clicks=True,mouse_add=1,mouse_stop=2)

fig = plt.figure(1)
plt.plot([2,2],[3,4])
pts = plt.ginput(n=1,show_clicks=True,mouse_add=1,mouse_stop=2)
plt.close(fig)

# Convert eveything to UTM #
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
    
cameraLoc_lat = 25.810
cameraLoc_lon = -80.122
utmCam = utm.from_latlon(cameraLoc_lat,cameraLoc_lon)
    
# Translate to camera position #
utmCoords[0,:] = utmCoords[0,:]-utmCam[0]
utmCoords[1,:] = utmCoords[1,:]-utmCam[1]
    
# Rotate to camera angle #
R = numpy.array([[math.cos(math.radians(theta)),-math.sin(math.radians(theta))],[math.sin(math.radians(theta)),math.cos(math.radians(theta))]])
utmCoordsr = numpy.matmul(R,utmCoords)
    
# Put these new coordinates into the point cloud %
pc['x'] = numpy.transpose(utmCoordsr[0,:])
pc['y'] = numpy.transpose(utmCoordsr[1,:])
        
    
    


    
# Visualize using pptk #
v = pptk.viewer(pc,pc.iloc[:,2])
v.set(point_size=0.1,theta=-25,phi=0,lookat=[0,0,13],color_map_scale=[-1,10])
