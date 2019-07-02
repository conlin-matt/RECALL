#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:27:15 2019

@author: matthewconlin
"""

# Import packages and allow for downloading of packages via Pip #
import subprocess
import sys
def pipInstall(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])

### Import packages ###
import requests
import os
import cv2 
    




def GetVideo(camToInput,year=2018,month=6,day=3,hour=1000):
    
    """
    Function to download a video clip from a specified WebCAT camera to local directory. The desired year, month, day, and time can be given, 
    however for those not given the function will use default values. If you don't like the video from the default date, an examination of WebCAT
    clips on the website can help determine the desired date/time to use. 
    
    """

    # Add zeros to day and month values if needed #
    if month<10:
        month = '0'+str(month)
    else:
        month = str(month)
    
    if day<10:
        day = '0'+str(day)
    else:
        day = str(day)
       
    # Get the video # 
    url = 'http://webcat-video.axds.co/{}/raw/{}/{}_{}/{}_{}_{}/{}.{}-{}-{}_{}.mp4'.format(camToInput,year,year,month,year,month,day,camToInput,year,month,day,hour)   
    
    # Read and load the video file from that URL using requests library
    filename = url.split('/')[-1] # Get the filename as everything after the last backslash #
    r = requests.get(url,stream = True) # Create the Response Object, which contains all of the information about the file and file location %
    with open(filename,'wb') as f: # This loop does the downloading 
        for chunk in r.iter_content(chunk_size = 1024*1024):
            if chunk:
                f.write(chunk)
    
    ## The specified video is now saved to the directory ##
    
    # Get the video file name #
    vidFile = camToInput+'.'+str(year)+'-'+month+'-'+day+'_'+str(hour)+'.mp4'
       
    return vidFile


#=============================================================================#


def DecimateVideo(vidPth):
    
    """
    Function to decimate WebCAT video clip downloaded with RECALL_GetVideo into still images so that a still image may be pulled
    for remote-GCP extraction. Function pulls 20 equally-spaced frames from the 10 minute video.
    
    """
    
    # Import packages #
#    pipInstall('opencv-python')
#    import cv2
    
    vid = cv2.VideoCapture(vidPth)

    # Find the number of frames in the video #
    vidLen = int(vid.get(7))
   
    # Pull 20 frames evenly distributed through the 10 minute video and save them to the video directory #
    for count in range(0,vidLen,int(vidLen/20)):
        vid.set(1,count) #Set the property that we will pull the frame numbered by count #  
        test,image = vid.read()
        cv2.imwrite('frame'+str(count)+'.png', image)


