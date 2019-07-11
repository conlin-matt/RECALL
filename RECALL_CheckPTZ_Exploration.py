#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 14:18:52 2019

@author: matthewconlin
"""

import numpy as np
import math
import matplotlib.pyplot as plt


def RECALL_GetVideo(cam,year=2018,month=6,day=3,hour=1000):

    ### Import packages ###
    import requests
    import os
    
    
    ### Detect current directory and make a temp directory to put the video in ###
    curDir = os.getcwd() # Detect current directory
    os.mkdir(curDir+'/TempForVideo') # Make a new temporary directory for the video #
    os.chdir(curDir+'/TempForVideo') # Go into the new directory #
    
    
    ### Format inputs as needed ###
    # Remove spaces in cam name of they exist #
    if ' ' in cam:
        camToInput1 = cam.replace(" ","")
    else:
        camToInput1 = cam
        
    # Remove capital letters in cam name if they exist # 
    if any(x.isupper() for x in camToInput1):
        camToInput = str.lower(camToInput1)
    else:
        camToInput = camToInput1;
    
    # Add zeros to day and month values if needed #
    if month<10:
        month = '0'+str(month)
    else:
        month = str(month)
    
    
    if day<10:
        day = '0'+str(day)
    else:
        day = str(day)
        
        
    ### Load the data ###  
    # Get the desired URL #
    url = 'http://webcat-video.axds.co/{}cam/raw/{}/{}_{}/{}_{}_{}/{}cam.{}-{}-{}_{}.mp4'.format(camToInput,year,year,month,year,month,day,camToInput,year,month,day,hour)   
    
    # Read and load the video file from that URL using requests library
    filename = url.split('/')[-1] # Get the filename as everything after the last backslash #
    r = requests.get(url,stream = True) # Create the Response Object, which contains all of the information about the file and file location %
    with open(filename,'wb') as f: # This loop does the downloading 
        for chunk in r.iter_content(chunk_size = 1024*1024):
            if chunk:
                f.write(chunk)
    
    ## The specified video is now saved to the directory ##
    
    # Get the path to the video #
    vidFile1 = str(os.listdir(os.getcwd()))
    vidFile = vidFile1[2:len(vidFile1)-2]
    fullVidPth = os.getcwd() + '/' +  vidFile
    
    return fullVidPth
            
  



vidPth = RECALL_GetVideo('miami40th') 



def RECALL_DecimateVideo(vidPth):
    
    # Import packages #
    #pipInstall('opencv-python')
    import cv2 
    
    import os
    
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
    for count in range(0,vidLen,int(vidLen/100)):
        vid.set(1,count) #Set the property that we will pull the frame numbered by count #  
        test,image = vid.read()
        
        # Find edges using Canny Edge Detector #
        edges = cv2.Canny(image,50,100)
        
#        plt.figure(1)
#        plt.subplot(121),plt.imshow(image)
#        plt.plot(1,1100,'r*')
#        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#        plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#        plt.show()
                
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
    
    #        cv2.line(image,(x1,y1),(x2,y2),(255,0,0),2)
    #        cv2.imshow('image',image)
            
            # Calc horizon angle (psi) #
            psi = math.atan2(y1-y2,x2-x1)
            dhorizon = ((y1-x2)-(y2-x1))/math.sqrt((x2-x1)**2+(y2-y1)**2)   
            psis = np.append(psis,psi)
            frameNum = np.append(frameNum,count)
                
            
#        plt.close()
#        cv2.destroyAllWindows()
      
    # Determine how many views there are #    
    psis = np.round(abs(psis),3)
    views = np.unique(psis)
    numViews = len(views)
    
    # Get example frames from each view #
    exFrames = list()
    for v in views:
        boolA = [psis==v]
        print(len(frameNum[boolA]))
        exFrame = frameNum[boolA][0]
        exFrames.append(exFrame)
        
    # Pull and display the views #
    subPlotNum = int('1'+str(numViews))
    i = 0
    plt.figure(1)
    for f in exFrames:
        i = i+1
        vid.set(1,f) #Set the property that we will pull the frame numbered by count #  
        test,image = vid.read()
        
        sub = int(str(subPlotNum)+str(i))
        plt.subplot(sub)
        plt.imshow(image)
        plt.title('View ' + str(i) + ' of ' + str(numViews))
        plt.xticks([])
        plt.yticks([])
        
        
        
        
    


















    
        
        from scipy import signal 
        

        image_YCrCb = cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
        xHorizon = np.array([])
        yHorizon = np.array([])
        for iCol in range(0,len(image_YCrCb[0,:])):
            col_Y = image_YCrCb[:,iCol,0]
            col_Cr = image_YCrCb[:,iCol,1]
            col_Cb = image_YCrCb[:,iCol,2]
            
            col_Cb_peaks = signal.find_peaks(col_Cb)
            col_Y_peaks = signal.find_peaks(-col_Y)
            
            iHorizons = [a for a in col_Cb_peaks[0] if a in col_Y_peaks[0]]
            if iHorizons:
                iHorizon = iHorizons[0]
            
                xHorizon = np.append(xHorizon,iCol)
                yHorizon = np.append(yHorizon,iHorizon)
            else:
                xHorizon = np.append(xHorizon,0)
                yHorizon = np.append(yHorizon,0)
#            plt.figure(1)
#            plt.plot(col_Y,'r')
#            plt.plot(col_Cb,'b')
#            plt.plot(col_Cb_peaks[0],col_Cb[col_Cb_peaks[0]],'g*')
#            plt.plot(col_Y_peaks[0],col_Y[col_Y_peaks[0]],'g*')
#            plt.plot(iHorizon,col_Cb[iHorizon],'k*')
#            plt.plot(iHorizon,col_Y[iHorizon],'k*')            
#            plt.show()
#            
#            plt.pause(2)  
#            plt.close()
        
        
        plt.figure(1)
        plt.imshow(image)
        plt.plot(xHorizon,yHorizon,'r.')
        
#        
#        
#        
#        cv2.imwrite('frame'+str(count)+'.png', image)





