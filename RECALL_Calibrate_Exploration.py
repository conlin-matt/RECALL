#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:57:43 2019

@author: matthewconlin
"""

import numpy as np
import math
from scipy.optimize import least_squares
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


### Get initial parameter estimation using Direct Linear Transform, as implemented in the online lecture ###

# Create the M matrix and fill it with GCP info #
M = np.zeros([(3*len(GCPs_im)),(13+((len(GCPs_im)-1)*4))])

startRow = -3
for i in range(0,6):
    startRow = startRow+3
    
    # Lidar GCP coordinates #
    Xx = GCPs_lidar[i,0]
    Xy = GCPs_lidar[i,1]
    Xz = GCPs_lidar[i,2]
    # Image GCP coordinates #
    x = GCPs_im[i,0]
    y = GCPs_im[i,1]
    camVec = [-x,-y,-1]
    
    # Put the coordinates in the correct places in the M matrix #
    for subi in range(0,3):
        zeroSpace = subi*4
        
        M[startRow+subi,0+zeroSpace] = Xx
        M[startRow+subi,1+zeroSpace] = Xy
        M[startRow+subi,2+zeroSpace] = Xz
        M[startRow+subi,3+zeroSpace] = 1
        
        M[startRow+subi,12+(i*4)] = camVec[subi]

# Make the norm of the M matrix = 1 by dividing it by its current norm #
M = M/np.linalg.norm(M)        

# Create the MTM matrix #
MTM = np.transpose(M) @ M

# SVD on MTM and take the eigenvector with the smallest eigenvalue as the solution for v #
u,s,v = np.linalg.svd(MTM)
vStar = v[:,len(v)-1]

# Create P as the first 12 elements of the eigenvector, in a 3x4 shape (?????? is this correct ????????) #
P = np.vstack((np.transpose(vStar[0:4]),np.transpose(vStar[4:8]),np.transpose(vStar[8:12])))



# Factor P into K by using RQ factorization #

# Get A and a. I think this is correct????? #
A = P[:,0:3]
a = P[:,3]

A1 = A[0,:]
A2 = A[1,:]
A3 = A[2,:]

# Solve the third row #
f = np.linalg.norm(A3)
R3 = (1/f)*A3
# Solve the second row #
e = np.dot(A2,R3)/np.linalg.norm(R3)
rhs = A2-(e*R3)
d = np.linalg.norm(rhs)
R2 = (1/d)*(rhs)
# Solve the first row # ## Not sure if this is correct... ##
b = np.dot(A1,R2)/np.linalg.norm(R2)
c = np.dot(A1,R3)/np.linalg.norm(R3)

lhs = A1-(b*R2)-(c*R3)
a = np.linalg.norm(lhs)
R1 = 1/a*(lhs)

# Create the intrinsic (k), rotation (R) and, and translation (t) matricies. Need to normalize k by element (3,3) at the end. DLT
# has given us initial estimations of all the unknowns. #
k = np.vstack((np.array([a,b,c]),np.array([0,d,e]),np.array([0,0,f])))
k = k/k[2,2]
R = np.vstack([R1,R2,R3])
t = np.dot(np.linalg.inv(k),P)[:,3]

#
## Define R more precisly by using angles generated from horizon pick #
#os.chdir('/Users/matthewconlin/Documents/Research/WebCAT')
#curd = os.getcwd()+'/TempForVideo'
#os.chdir(curd)
#imgs = glob.glob('*.png')
#img = mpimg.imread(curd+'/'+imgs[1])
#imgplot = plt.imshow(img)  
#plt.title('Click two points on the horizon; first on the left side of the image, then the right')     
#horizonPts = plt.ginput(n=2,show_clicks=True)
#
#xa = horizonPts[0][0]
#ya = horizonPts[0][1]
#xb = horizonPts[1][0]
#yb = horizonPts[1][1]
#
## Solve for the two angles (psi and xi) using the horizon #
#psi = math.atan2(ya-yb,xb-xa)
#dhorizon = ((ya-xb)-(yb-xa))/math.sqrt((xb-xa)**2+(yb-ya)**2)
#Cc = math.atan2(dhorizon,focal)
#Rt = 6371000
#D = math.sqrt((t[2]+Rt)**2-Rt**2)
#beta = math.acos((t[2]+(.42*(D**2/Rt)))/D)
#xi = beta-Cc
#
## Now compute the three camera orientation angles from the two horizon angles #
#phi = -math.asin(math.sin(xi)*math.sin(psi))
#omega = math.acos(math.cos(xi)/(math.sqrt(math.cos(psi)+(math.cos(xi)**2*math.sin(psi)))))
#kk = int(input('In what direction does the camera look? Press 1 for north-east, 2 for east-south, 3 for north-west, 4 for west-south'))
#if kk == 1:
#    kappa = -math.pi/4
#elif kk == 2:
#    kappa = -math.pi*3/4
#elif kk == 3:
#    kappa = math.pi/4
#else:
#    kappa = math.pi*3/4
#    
## Define a new R using the euqations in Holland et al. 1997.
#r11 =(math.cos(omega)*math.cos(kappa))+(math.sin(omega)*math.cos(phi)*math.sin(kappa))  
#r12 = (-math.sin(omega)*math.cos(kappa))+(math.cos(omega)*math.cos(phi)*math.sin(kappa))
#r13 = math.sin(phi)*math.sin(kappa)
#r21 = (-math.cos(omega)*math.sin(kappa))+(math.sin(omega)*math.cos(phi)*math.cos(kappa))
#r22 = (math.sin(omega)*math.sin(kappa))+(math.cos(omega)*math.cos(phi)*math.cos(kappa))
#r23 = math.sin(phi)*math.cos(kappa)
#r31 = math.sin(omega)*math.sin(phi)
#r32 = math.cos(omega)*math.sin(phi)
#r33 = -math.cos(phi)
#Rh = np.vstack([[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]])





### Optimize parameter estimation using nonlinear fit with DLT output as initial guess for parameters ###

# Define residual calculation function to use in optimization of P #
toOptVec = np.array([k[0,0],k[0,1],k[0,2],k[1,0],k[1,1],k[1,2],k[2,0],k[2,1],k[2,2],R[0,0],R[0,1],R[0,2],R[1,0],R[1,1],R[1,2],R[2,0],R[2,1],R[2,2],t[0],t[1],t[2],0,0]) # Need to put everything into a vector for the function to work #
def calcResid_withDistortion(toOptVec,GCPs_lidar,GCPs_im):
    resid = 0
    for i in range(0,len(GCPs_im)): 
        ki = np.vstack([[toOptVec[0],toOptVec[1],toOptVec[2]],[toOptVec[3],toOptVec[4],toOptVec[5]],[toOptVec[6],toOptVec[7],toOptVec[8]]])
        Ri = np.vstack([toOptVec[9:12],toOptVec[12:15],toOptVec[15:18]])
        ti = toOptVec[18:21]
        Xw = np.append(np.array(GCPs_lidar[i,:]),1)
        Xc = GCPs_im[i,:]
        
        # Project world to image with current guess of calibration params #
        Pi = np.dot(ki,np.c_[Ri,t])
        uv = np.dot(Pi,Xw)
        u = uv[0]/uv[2]
        v = uv[1]/uv[2]
        
        # Rigid transformation from world coords to camera coords #
        camCoords = np.dot(Ri,Xw[0:3])+ti
        # Perspective transformation from 3d camera coords to ideal image coords #
        x = toOptVec[4]*(camCoords[0]/camCoords[2])
        y = toOptVec[4]*(camCoords[1]/camCoords[2])
        # Distort the projected image coords #
        uHat = u+((u-toOptVec[2])*((toOptVec[21]*(x**2+y**2))+(toOptVec[22]*(x**2+y**2)**2)))
        vHat = v+((v-toOptVec[5])*((toOptVec[21]*(x**2+y**2))+(toOptVec[22]*(x**2+y**2)**2)))
        
        # Calculate the residual between projected/distorted image coords and measured image coords #
        residV = np.array([Xc[0]-uHat,Xc[1]-vHat]) 
        residi = np.linalg.norm(residV)**2
        
        resid = resid+residi
        
    residToReturn = np.tile(resid,[23])
    
    return residToReturn

# Optimize the parameters using the Levenberg-Marquardt algorithm #
out = least_squares(calcResid_withDistortion,toOptVec,args=(GCPs_lidar,GCPs_im),method='lm')
optVec = out['x']

# Build optimized K, R, and t and distortion coeffs #
Kopt = np.vstack([[optVec[0],optVec[1],optVec[2]],[optVec[3],optVec[4],optVec[5]],[optVec[6],optVec[7],optVec[8]]])
Ropt = np.vstack([[optVec[9],optVec[10],optVec[11]],[optVec[12],optVec[13],optVec[14]],[optVec[15],optVec[16],optVec[17]]])
topt = np.array([optVec[18],optVec[19],optVec[20]])
k1 = optVec[21]
k2 = optVec[22]

# Test the optimized calibration by computing the residual (reprojection error) for each GCP #
resid = np.empty([0])
for i in range(0,len(GCPs_im)):
        Xw = np.append(np.array(GCPs_lidar[i,:]),1)
        Xc = GCPs_im[i,:]

        # Project world to image #
        Ptest = np.dot(Kopt,np.c_[Ropt,topt])
        uv = np.dot(Ptest,Xw)
        uProj = uv[0]/uv[2]
        vProj = uv[1]/uv[2]
        
        # Distort #
        # Rigid transformation from world coords to camera coords #
        camCoords = np.dot(Ropt,Xw[0:3])+topt
        # Perspective transformation from 3d camera coords to ideal image coords #
        x = Kopt[1,1]*(camCoords[0]/camCoords[2])
        y = Kopt[1,1]*(camCoords[1]/camCoords[2])
        uProjD = uProj+((uProj-Kopt[0,2])*((k1*(x**2+y**2))+(k2*(x**2+y**2)**2)))
        vProjD = vProj+((vProj-Kopt[1,2])*((k1*(x**2+y**2))+(k2*(x**2+y**2)**2)))
        
        # Compute residual #
        residV = np.array([Xc[0]-uProjD,Xc[1]-vProjD]) 
        resid = np.append(resid,np.linalg.norm(residV))








