#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:08:03 2019

@author: Matthew P. Conlin


Function to decimate WebCAT video clip downloaded with TReC_GetVideo into still images so that a still image may be pulled
for remote-GCP extraction. Function pulls 20 equally-spaced frames from the 10 minute video.
    Inputs:
        vidFile: The path to the saved video. This is the output of TReC_GetVideo 

"""

def TreC_DecimateVideo(vidPth):
    
    # Import packages #
    import cv2
    import os
    
    # Make sure we are still in the same directory as the video # 
    os.chdir(vidPth.rsplit('/',1)[0]) # Go to the directory defined by the path prior to the final backslash in the vidFile string #
    
    # Get the video capture #
    vid = cv2.VideoCapture(vidPth)
    
    # Find the number of frames in the video #
    vidLen = int(vid.get(7))
   
    # Pull 20 frames evenly distributed through the 10 minute video and save them to the video directory #
    for count in range(0,vidLen,int(vidLen/20)):
        vid.set(1,count) #Set the property that we will pull the frame numbered by count #  
        test,image = vid.read()
        cv2.imwrite('frame'+str(count)+'.png', image)



