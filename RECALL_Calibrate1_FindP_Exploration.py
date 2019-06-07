#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:55:07 2019

@author: matthewconlin
"""

import numpy as np
import math
from scipy.optimize import least_squares

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
        

# Create the MTM matrix #
MTM = np.dot(np.transpose(M),M)

# SVD on MTM and take the eigenvector with the smallest eigenvalue as the solution for v #
u,s,v = np.linalg.svd(MTM)
vStar = v[:,len(v)-1]

# Create P as the first 12 elements of the eigenvector, in a 3x4 shape (?????? is this correct ????????) #
P = numpy.vstack((numpy.transpose(vStar[0:4]),numpy.transpose(vStar[4:8]),numpy.transpose(vStar[8:12])))





# Factor P into K by using RQ factorization #

# Get A and a. I think this is correct?????
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
# Solve the first row # ## DON'T THINK THIS IS CORRECT ##
b = np.dot(A1,R2)/np.linalg.norm(R2)
c = np.dot(A1,R3)/np.linalg.norm(R3)

lhs = A1-(b*R2)-(c*R3)
a = np.linalg.norm(lhs)
R1 = 1/a*(lhs)

# Create the intrinsic (k), rotation (R) and, and location (t) matricies. Need to normalize k by element (3,3) at the end. DLT
# has given us initial estimations of all the unknowns. #
k = np.vstack((np.array([a,b,c]),np.array([0,d,e]),np.array([0,0,f])))
R = np.vstack([R1,R2,R3])
t = np.dot(np.linalg.inv(k),P)[:,3]

k = k/k[2,2]

# Pull the individual parameters #
focal = k[1,1]
aspect = k[0,0]/focal
skew = k[0,1]/focal
xo = k[0,2]
yo = k[1,2]




# Get better initial angle orientation estimates using the horizon; following methods of C-Pro tool #

# Display the image and pcik two points on the horizon #
os.chdir('/Users/matthewconlin/Documents/Research/WebCAT')
curd = os.getcwd()+'/TempForVideo'
os.chdir(curd)
imgs = glob.glob('*.png')
img = mpimg.imread(curd+'/'+imgs[1])
imgplot = plt.imshow(img)  
plt.title('Click two points on the horizon; first on the left side of the image, then the right')     
horizonPts = plt.ginput(n=2,show_clicks=True)

xa = horizonPts[0][0]
ya = horizonPts[0][1]
xb = horizonPts[1][0]
yb = horizonPts[1][1]

# Solve for the two angles (psi and xi) using the horizon #
psi = math.atan2(ya-yb,xb-xa)
dhorizon = ((ya-xb)-(yb-xa))/math.sqrt((xb-xa)**2+(yb-ya)**2)
Cc = math.atan2(dhorizon,focal)
Rt = 6371000
D = math.sqrt((t[2]+Rt)**2-Rt**2)
beta = math.acos((t[2]+(.42*(D**2/Rt)))/D)
xi = beta-Cc

# Now compute the three camera orientation angles from the two horizon angles #
phi = -math.asin(math.sin(xi)*math.sin(psi))
omega = math.acos(math.cos(xi)/(math.sqrt(math.cos(psi)+(math.cos(xi)**2*math.sin(psi)))))
k = int(input('In what direction does the camera look? Press 1 for north-east, 2 for east-south, 3 for north-west, 4 for west-south'))
if k == 1:
    kappa = -math.pi/4
elif k == 2:
    kappa = -math.pi*3/4
elif k == 3:
    kappa = math.pi/4
else:
    kappa = math.pi*3/4


       
# Optimize P using nonlinear fit method #
Pvec = np.array([P[0,0],P[0,1],P[0,2],P[0,3],P[1,0],P[1,1],P[1,2],P[1,3],P[2,0],P[2,1],P[2,2],P[2,3]])
def computeResid(Pvec,GCP_world,GCP_im):
    resid = list()
    Puse = np.reshape(Pvec,[3,4])
    for i in range(0,len(GCP_world)):
        X = np.array(GCP_world[i,:])
        X = np.append(X,1)
        x = GCP_im[i,:]
        x = np.append(x,1)

        projection = np.dot(Puse,X)
        residThisPt = (abs(x[0]-projection[0])+abs(projection[1]-x[1]))**2
        
        resid.append(residThisPt)

    return resid

testresid = computeResid(Pvec,GCPs_lidar,GCPs_im)

x,cost,fun = least_squares(computeResid,Pvec,args=(GCPs_lidar,GCPs_im),method='lm')








