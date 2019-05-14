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

import pptk
import numpy as np
import pandas as pd
import math

# Read in the lidar file as a dataframe using Pandas #
fname = '/Users/matthewconlin/Documents/Research/WebCAT/lidarXYZsmallfile.txt'
col_names = ['x','y','z']
col_dtype = {'x':np.float32,'y':np.float32,'z':np.float32}
pc = pd.read_csv(fname,names=col_names,dtype=col_dtype,delim_whitespace=True)



# Convert x and y in degrees to distance from camera in m. This is tecnically creating a local coordinate system with the camera centered 
# at the origin, but could maybe do this better (e.g. by converting to utm and then using transformToDVT site trabslation and rotation method )
distX = list()
distY = list()
for px,py in zip(pc['x'],pc['y']):
    
    # Convert difference in latitude to difference in north/south distance
    dlatm = (py-cameraLoc_lat)*(4008000/360)
    
    # Convert difference in longitude into east-west distance. This is a bit trickier, since distance between meridians
    # changes with changing latitude. We will use the latitude of the camera to approx. the calculation. We will
    # therefore be using a Equirectangular projection to do this, i.e. assuming the Earth is flat and meridians
    # are not curved over these relatively short distances.
    dlonm = (px-cameraLoc_lon)*40075160*math.cos(cameraLoc_lat)/360
    
    # Append them to lists #
    distX.append(dlonm)
    distY.append(dlatm)
    
pc['x'] = distX   
pc['y'] = distY   




# Transform lidar data to local coordinates using the Argus method. Do this by manually dertmining the orientation angle. 
# If this whole process works, we will need an automated way to estimate orientation e.g. using the horizon #
    theta = 70 # Obtained by giput in Matlab imshow, then doing what Nathaniel did inupdateDVTsite.m (lines 74-75)
    # Convert eveything to UTM #
    pipInstall('utm')
    import utm
    utmCoords = utm.from_latlon(20,-80)






     
# Visualize using pptk #
v = pptk.viewer(pc,pc.iloc[:,2])
v.set(point_size=0.005)
v.set(lookat=[0,0,20])
