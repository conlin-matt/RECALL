#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 16:26:05 2019

@author: Matthew P. Conlin


Function to download a video clip from a specified WebCAT camera to local directory. The desired year, month, day, and time can be given, 
however if none are given function will use default values. If you don't like the video from the default date, an examination of WebCAT
clips on the website can help determine the desired date/time to use. 


"""

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
            
  



vidPth = RECALL_GetVideo('follypiernorth')       
        
        
        
        
        
        
        
        
        
        