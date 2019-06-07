#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:30:58 2019

@author: matthewconlin
"""

def RECALL_PickGCPs(pc,imDir):

    import os
    import pptk
    import numpy as np
    import glob
    
    #pipInstall('matplotlib==2.1.0')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    
    
    # Plot the image to examine #
    os.chdir(imDir)
    imgs = glob.glob('*.png')
    img = mpimg.imread(imDir+'/'+imgs[1])
    imgplot = plt.imshow(img)
    
    # Plot the lidar data #
    v = pptk.viewer(pc,pc.iloc[:,2])
    v.set(point_size=0.1,theta=-25,phi=0,lookat=[0,0,13],color_map_scale=[-1,10],r=0)
    
    
    
    # Loop through and get the GCPs #
    GCPs_lidar = np.empty([0,3])
    GCPs_im = np.empty([0,2])
    val=1
    while val!=0:
        
        
        # Pick point in the lidar cloud #
        startLidar=int(input('Right click, and then click on a point in the lidar data. Press 1 when done. '))
        if startLidar==1:
            p = v.get('selected')
            GCPs_lidar = np.vstack((GCPs_lidar,pc.iloc[p,:]))
        else:
            pass    
        
        
        # Pick the corresponding point in the image #
        startIm=int(input('Press 1 when you are ready to pick the corresponding point in the image. '))
        plt.close()
        if startIm==1:
            os.chdir('/Users/matthewconlin/Documents/Research/WebCAT')
            curd = os.getcwd()+'/TempForVideo'
            os.chdir(curd)
            imgs = glob.glob('*.png')
            img = mpimg.imread(curd+'/'+imgs[1])
            imgplot = plt.imshow(img)
            
            pt = plt.ginput(show_clicks=True)
            GCPs_im = np.append(GCPs_im,pt,axis = 0)
            plt.close()
        else:
            pass
        
        val=int(input('Do you want to continue picking GCPs? Press 0 to stop, and other key to continue.'))
        
    return GCPs_lidar,GCPs_im
    
        
GCPs_lidar,GCPs_im = RECALL_PickGCPs(pc,'/Users/matthewconlin/Documents/Research/WebCAT/TempForVideo')   
    
        
    
    
    
    
    
    
    
    
    
    
    
    
