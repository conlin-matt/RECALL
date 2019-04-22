#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:57:21 2019

@author: matthewconlin
"""

import cv2
import os


vid = cv2.VideoCapture(vidPth)

# Play the video, just for fun 
while vid.isOpened():
    test,frame = vid.read() # Pull frame from video #
    
    cv2.imshow('frame',frame)
    cv2.waitKey(25)
vid.release()
cv2.destroyAllWindows()


# Extract and save frames from the video
os.chdir(vidPth.rsplit('/',1)[0]) # Get in same directory as the video #

# Pull 20 frames evenly distributed through the 10 minute video and save the to the video directory #
vidLen = int(vid.get(7))
for count in range(0,vidLen,int(vidLen/20)):
    vid.set(1,count) #Set the property that we will pull the frame numbered by count #  
    test,image = vid.read()
    cv2.imwrite('frame'+str(count)+'.png', image)
    

