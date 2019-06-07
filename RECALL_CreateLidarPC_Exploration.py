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

#pipInstall('matplotlib==2.1.0')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Turn the numpy array into a Pandas data frame #
pc = pd.DataFrame({'x':lidarDat[:,0],'y':lidarDat[:,1],'z':lidarDat[:,2]})



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
    
cameraLoc_lat = 32.654731
cameraLoc_lon = -79.939322
utmCam = utm.from_latlon(cameraLoc_lat,cameraLoc_lon)
    
# Translate to camera position #
utmCoords[0,:] = utmCoords[0,:]-utmCam[0]
utmCoords[1,:] = utmCoords[1,:]-utmCam[1]

    
# Put these new coordinates into the point cloud %
pc['x'] = numpy.transpose(utmCoords[0,:])
pc['y'] = numpy.transpose(utmCoords[1,:])
        
    





