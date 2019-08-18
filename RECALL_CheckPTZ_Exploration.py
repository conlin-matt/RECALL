#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:18:52 2019

@author: matthewconlin
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import RECALL
import cv2 
import os
import pandas as pd
from scipy import signal,stats

vidPth = RECALL.getImagery_GetVideo('miami40thcam')
vidPth = '/Users/matthewconlin/Documents/Research/WebCAT/'+vidPth



# Make sure we are still in the same directory as the video # 
os.chdir(vidPth.rsplit('/',1)[0]) # Go to the directory defined by the path prior to the final backslash in the vidFile string #
    
# Get the video capture #
vid = cv2.VideoCapture(vidPth)
    
# Find the number of frames in the video #
vidLen = int(vid.get(7))
   
# Get horizon angle of a bunch of different frames using edge detection. We will take different views as those which have 
# substantially different horizon angles #
psis = np.array([])
frameNum = np.array([])
for count in range(0,vidLen,int(vidLen/1000)):
    vid.set(1,count) #Set the property that we will pull the frame numbered by count #  
    test,image = vid.read()
        
    # Erode the image #
    kernel = np.ones((5,5),np.uint8)
    imeroded = cv2.erode(image,kernel,iterations = 2)
    
    
#    # H-LSC method #
#    horizPts = np.array([])
#    for colnum in range(0,len(image[1,:,:])):
#        colR = imeroded[:,colnum,0]
#        colG = imeroded[:,colnum,1]
#        colB = imeroded[:,colnum,2]
#        changeVec = np.sqrt(np.diff(colR)**2 + np.diff(colG)**2 + np.diff(colB)**2)
#        maxRow = np.where(changeVec == max(changeVec))
#        maxRow = int(maxRow[0][0])
#        horizPts = np.append(horizPts,maxRow)
#        
#    horizPts = signal.medfilt(horizPts)
#    
#    slope,intercept,r,p,se = stats.linregress(range(0,len(horizPts)),horizPts)
#    horizLine = (slope*(range(0,len(horizPts))))+intercept
#        
#    psi = math.atan2(horizLine[len(horizLine)-1]-horizLine[0],len(horizLine))
#    
#    psis = np.append(psis,psi)
#    frameNum = np.append(frameNum,count)
#    
#    plt.figure(1)
#    plt.subplot(111),plt.imshow(image)
#    plt.plot(range(0,len(horizLine)),horizLine)
#    plt.show()
    
    
#    # H-MED method #
#    horizPts = np.array([])
#    for colnum in range(0,len(image[1,:,:])):
#        colR = imeroded[:,colnum,0]
#        colG = imeroded[:,colnum,1]
#        colB = imeroded[:,colnum,2]
#        medDifsCol = np.array([])
#        pixNum = np.array([])
#        for pix in range(4,len(colR)-6):
#            upMean = np.mean(np.array([np.median(colR[pix-4:pix+1]),np.median(colG[pix-4:pix+1]),np.median(colB[pix-4:pix+1])]))
#            downMean = np.mean(np.array([np.median(colR[pix+1:pix+6]),np.median(colG[pix+1:pix+6]),np.median(colB[pix+1:pix+6])]))
#            medDif = abs(upMean-downMean)
#            medDifsCol = np.append(medDifsCol,medDif)
#            pixNum = np.append(pixNum,pix)
#            
#        iHoriz = np.where(medDifsCol == max(medDifsCol))
#        horizPt = int(pixNum[iHoriz[0][0]])
#        horizPts = np.append(horizPts,horizPt)

    

        
    

    # Find edges using Canny Edge Detector #
    edges = cv2.Canny(imeroded,50,100)
    
#    # Do a Hough transform manually just for fun- this section is meant to be run independently #
#    thetaSpace = np.linspace(0,180,180)
#    rhoSpace = np.linspace(-1200,1200,3200)
#    accumulator = np.zeros([len(rhoSpace),len(thetaSpace)])
#    ones = np.where((edges == 255))
#    for row,col in zip(ones[0],ones[1]):
#        theta_line = thetaSpace
#        rho_line = (row*np.cos(np.radians(theta_line)))+(col*np.sin(np.radians(theta_line)))
#        
#        
#        for t,r in zip(theta_line,rho_line):
#            place_theta = (np.digitize(t,thetaSpace))-1
#            place_rho = (np.digitize(r,rhoSpace))-1
#            accumulator[place_rho,place_theta] = accumulator[place_rho,place_theta]+1
#            
#    fig = plt.figure()
#    plt.pcolormesh(thetaSpace,rhoSpace,accumulator,cmap='Greys_r',vmin=0,vmax=100,shading='gouraud')
#    plt.colorbar()
#    plt.show()
#    fig.savefig('HoughTransform.png')
#    #############################################################################################

    
    
        
#    plt.figure(1)
#    plt.subplot(121),plt.imshow(imeroded)
#    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#    plt.show()
                
    # Determine longest straight edge as horizon using Hough Transform #
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    if lines is not None:
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

    
#        cv2.line(imeroded,(x1,y1),(x2,y2),(255,0,0),2)
#        cv2.imshow('image',imeroded)
            
        # Calc horizon angle (psi) #
        psi = math.atan2(y1-y2,x2-x1)
        
        psis = np.append(psis,psi)
        frameNum = np.append(frameNum,count)
        
    
    
                
            



      
############# Determine how many views there are ##################  

# Round angles to remove small fluctuations #
psis = np.round(abs(psis),3)

# Continuity criterion: Often, it seems that bad horizon picks happen pne-a couple times in a row. So, let's only count the angle as a view if it is found at least x times in a row #
# Count length of chunks of value #
dif = np.diff(psis)
changes = np.array(np.where(dif!=0))


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
    IDs_good = np.array(np.where(segLens>=10))
    valsKeep = vals[IDs_good]
    
    # Find the unique views #
    viewAngles = np.unique(valsKeep)
    numViews = len(viewAngles)
    print('Found '+str(int(numViews))+' view angle(s)')
    
    # Find and extract the frames contained within each view #
    angles = []
    frames = []
    for i in viewAngles:
        iFrames = np.array(np.where(psis == i))
        angles.append(i)
        frames.append([iFrames])
        
    viewDict = {'View Angles':angles,'Frames':frames}
    viewDF = pd.DataFrame(viewDict)
        
    

    ################# Plot each view ############################
    frameVec = np.array(range(0,vidLen,int(vidLen/1000)))
    
    subPlotNum = int('1'+str(numViews))
    iii = 0
    plt.figure(1)
    for i in range(0,len(viewDF)):
        iii = i+1
        
        frames = np.array(viewDF['Frames'][i])
        exFrame = np.array(frames)[0,0,1]
        
        vid.set(1,frameVec[exFrame]) #Set the property that we will pull the frame numbered by count #  
        test,image = vid.read()
            
        sub = int(str(subPlotNum)+str(iii))
        plt.subplot(sub)
        plt.imshow(image)
        plt.title('View ' + str(iii) + ' of ' + str(numViews))
        plt.xticks([])
        plt.yticks([])





else:
    
   # Find the unique views #
    viewAngles = np.unique(psis)
    numViews = len(viewAngles)
    print('Found '+str(int(numViews))+' view angle(s)')
    
    # Find and extract the frames contained within each view #
    angles = []
    frames = []
    for i in viewAngles:
        iFrames = np.array(np.where(psis == i))
        angles.append(i)
        frames.append([iFrames])
        
    viewDict = {'View Angles':angles,'Frames':frames}
    viewDF = pd.DataFrame(viewDict)

    ################# Plot each view ############################
    subPlotNum = int('1'+str(numViews))
    iii = 0
    plt.figure(1)
    for i in range(0,len(viewDF)):
        iii = i+1
        
        frames = np.array(viewDF['Frames'][i])
        exFrame = np.array(frames)[0,0,1]
        
        vid.set(1,exFrame) #Set the property that we will pull the frame numbered by count #  
        test,image = vid.read()
            
        sub = int(str(subPlotNum)+str(iii))
        plt.subplot(sub)
        plt.imshow(image)
        plt.title('View ' + str(iii) + ' of ' + str(numViews))
        plt.xticks([])
        plt.yticks([])












