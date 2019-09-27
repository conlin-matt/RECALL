#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:29:17 2019

@author: matthewconlin


Get IOPs and EOPs of a camera and then recitify its imagery using the CPro alorithm. Steps:
    1. Use SurfRCaT to get control points from an image
    2. Solve for DLT parameters using DLT method outlined in Sanchez Garcia paper
    3. Recover estimates of IOPs and EOPs from DLT params using method sent by ESG in word doc
        Problems: not sure which Ls each a1,b1,etc correcpond to. Not sure what h,p,or d parameters in formulas are.
    3.5. Use horizon line to get better estimates of omega, phi, and kappa using method outlined in CPro paper.
    4. Do nonlinear least squares to iteratively solve for corrections to each paramater. Augment the collinearity equations with the horizon constraint equations outlined in the paper. 
        Problems: not sure if implementing the equations correctly, not sure if calculating angles the correct way 

 Note: It was sort of converging when I did the order of Ls as a1,a2,a3,b1,b2,b3,c1,c2,c3,d1,d2 and got rid of the weight matrix. I had to help it with a few values also though.


"""

wd = '/Users/matthewconlin/Documents/Research/WebCAT/'

# Import packages #
import pickle
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import glob

os.chdir(wd)




# Load in the image and control points #
f = open(wd+'GCPs_im.pkl','rb')
f2 =  open(wd+'GCPs_lidar.pkl','rb')     
gcps_im = pickle.load(f)
gcps_lidar = pickle.load(f2)


####### Direct Linear Transformation to get initial estimates of IOPs and EOPs from control points ######

# Set up the matricies #
A = np.empty([0,11])
for i in range(0,len(gcps_im)):
    v1 = np.array([ gcps_lidar[i,0],gcps_lidar[i,1],gcps_lidar[i,2],1,0,0,0,0,-gcps_im[i,0]*gcps_lidar[i,0],-gcps_im[i,0]*gcps_lidar[i,1],-gcps_im[i,0]*gcps_lidar[i,2] ])
    v2 = np.array([ 0,0,0,0,gcps_lidar[i,0],gcps_lidar[i,1],gcps_lidar[i,2],1,-gcps_im[i,1]*gcps_lidar[i,0],-gcps_im[i,1]*gcps_lidar[i,1],-gcps_im[i,1]*gcps_lidar[i,2] ])
    vv = np.vstack([v1,v2])
    A = np.vstack([A,vv])
    
O = np.empty([0,1])
for i in range(0,len(gcps_im)):
    p1 = gcps_im[i,0]
    p2 = gcps_im[i,1]
    vv = np.vstack([p1,p2])
    O = np.vstack([O,vv])
    
# Solve for 11 DLT parameters through least-squares (use pseudo-inverse) #
L = np.linalg.inv(np.transpose(A) @ A) @ (np.transpose(A) @ O)


# Recover initial estimates of IOPs and EOPs from DLT parameters (L) #
a1 = float(L[0]);b1 = float(L[1]);c1 = float(L[2]);d1 = float(L[3]);a2 = float(L[4]);b2 = float(L[5]);c2 = float(L[6]);d2 = float(L[7]);a3 = float(L[8]);b3=float(L[9]);c3 = float(L[10])

x0 = ( np.array([a1,b1,c1]) @ np.array([a3,b3,c3]) ) / ( np.array([a3,b3,c3]) @ np.array([a3,b3,c3]) )
y0 = ( np.array([a2,b2,c2]) @ np.array([a3,b3,c3]) ) / ( np.array([a3,b3,c3]) @ np.array([a3,b3,c3]) )
f = math.sqrt( (( np.array([a1,b1,c1]) @ np.array([a1,b1,c1]) ) / ( np.array([a3,b3,c3]) @ np.array([a3,b3,c3]) )) - x0**2 ) 
XL,YL,ZL = np.linalg.inv(np.transpose(np.vstack([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]]))) @ np.array([-d1,-d2,-1])
if ZL<0:
    ZL = 2
else:
    pass

p = -1/f
h = ZL/f
R = np.transpose( ((1/(p*h*f)) * np.vstack([[h,0,-h*x0],[-d1,1,(x0*d1)-y0],[0,0,-h*f]])) @ np.transpose(np.vstack([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])) )
phi = math.asin(R[2,0])
omega = math.atan2(-R[2,1],R[2,2])
kappa = math.atan2(-R[1,0],R[0,0])





################ Get initial estimates of omega,phi,kappa from horizon line instead ########################

# User input of 2 horizon points #
figure = plt.figure()
ax = figure.add_subplot(111)
        
frames = glob.glob('frame'+'*')
frame = frames[1]
        
img = mpimg.imread(wd+'/'+frame)
imgplot = plt.imshow(img)
pt = plt.ginput(n=2,show_clicks=True)   


# Calculate the intermediate angles from the points #
xa = pt[0][0]
xb = pt[1][0]
ya = pt[0][1]
yb = pt[1][1]
psi = math.atan2(ya-yb,xb-xa)

d = ((ya*xb)-(yb*xa))/math.sqrt( (xb-xa)**2+(yb-ya)**2 )
C = math.atan2(d,f)
D = math.sqrt( (ZL+6371000)**2 - 6371000**2 )
beta = math.acos( (ZL+ (.42*(D**2)/6371000))/D )
xi = beta-C

# Calculate the actual angles from the intermediate angles #
phi = -math.asin( math.sin(xi)*math.sin(psi) )
omega = math.acos( math.cos(xi)/math.sqrt(math.cos(psi)**2 + (math.cos(xi)**2 * math.sin(psi)**2)) )
kappa = -math.pi/4




# Now the real fun begins. Use the solved for EOPs and IOPs as initial values in a non-linear least squared adjustment to solve for refined values for each parameter. Page 615 in Photogrammetry textbook has the elements of the Jacobian matrix computed.  #


# Step 0: calculate the elements of the M matrix as shown in the book since the R matrix defined above is not the same #
m11 = math.cos(phi)*math.cos(kappa)
m12 = (math.sin(omega)*math.sin(phi)*math.cos(kappa)) + (math.cos(omega)*math.sin(kappa))
m13 = (-math.cos(omega)*math.sin(phi)*math.cos(kappa)) + (math.sin(omega)*math.sin(kappa))
m21 = -math.cos(phi)*math.sin(kappa)
m22 = (-math.sin(omega)*math.sin(phi)*math.sin(kappa)) + (math.cos(omega)*math.cos(kappa))
m23 = (math.cos(omega)*math.sin(phi)*math.sin(kappa)) + (math.sin(omega)*math.cos(kappa))
m31 = math.sin(phi)
m32 = -math.sin(omega)*math.cos(phi)
m33 = math.cos(omega)*math.cos(phi)


iteration = 0
allvals = np.empty([0,9]) # Matrix to store the values (rows) for each parameter (columns) as the least squares solution iterates #
changes = np.array([10,10,10,10,10,10,10,10,10])
while np.max(np.abs(changes))>.00001:
    
    iteration = iteration+1
    if iteration>200:
        print('Error: The soultion is likely diverging')
        break
    else:
        pass
    
    if iteration == 1:
        vals = np.array([omega,phi,kappa,XL,YL,ZL,x0,y0,f])
        allvals = np.vstack([allvals,vals])
    else:
        pass
    
    # Step 1: Form the B (Jacobian) and e (observation) matricies 
    B = np.empty([0,9])
    epsilon = np.empty([0,1])
    for i in range(0,len(gcps_lidar)):
        XA = gcps_lidar[i,0]
        YA = gcps_lidar[i,1]
        ZA = gcps_lidar[i,2]
        xa = gcps_im[i,0]
        ya = gcps_im[i,1]
        
        # Deltas #
        deltaX = XA-XL
        deltaY = YA-YL
        deltaZ = ZA-ZL
        
        # Numerator and denominator of collinearity conditions #
        q = (m31*deltaX)+(m32*deltaY)+(m33*deltaZ)
        r = (m11*deltaX)+(m12*deltaY)+(m13*deltaZ)
        s = (m21*deltaX)+(m22*deltaY)+(m23*deltaZ)
        
        # Now all the b's of the B (Jacobian) matrix #
        b11 = (f/q**2) * ( (r*((-m33*deltaY)+(m32*deltaZ))) - (q*((-m13*deltaY)+(m12*deltaZ))) )
        b12 = (f/q**2) * ( (r*( (math.cos(phi)*deltaX) + (math.sin(omega)*math.sin(phi)*deltaY) - (math.cos(omega)*math.sin(phi)*deltaZ) )) - (q*( (-math.sin(phi)*math.cos(kappa)*deltaX) + (math.sin(omega)*math.cos(phi)*math.cos(kappa)*deltaY) - (math.cos(omega)*math.cos(phi)*math.cos(kappa)*deltaZ) ))       )
        b13 = (-f/q) * ((m21*deltaX)+(m22*deltaY)+(m23*deltaZ))
        b14 = (f/q**2) * ((r*m31) - (q*m11))
        b15 = (f/q**2) * ((r*m32) - (q*m12))
        b16 = (f/q**2) * ((r*m33) - (q*m13))
        b17 = 1
        b18 = 0
        b19 = -r/q
    
        b21 = (f/q**2) * ( (s*((-m33*deltaY)+(m32*deltaZ))) - (q*((-m23*deltaY)+(m22*deltaZ))) )
        b22 = (f/q**2) * ( (s*( (math.cos(phi)*deltaX) + (math.sin(omega)*math.sin(phi)*deltaY) - (math.cos(omega)*math.sin(phi)*deltaZ) )) - (q*( (-math.sin(phi)*math.sin(kappa)*deltaX) - (math.sin(omega)*math.cos(phi)*math.sin(kappa)*deltaY) + (math.cos(omega)*math.cos(phi)*math.sin(kappa)*deltaZ) ))       )
        b23 = (f/q) * ((m11*deltaX)+(m12*deltaY)+(m13*deltaZ))
        b24 = (f/q**2) * ((s*m31) - (q*m21))
        b25 = (f/q**2) * ((s*m32) - (q*m22))
        b26 = (f/q**2) * ((s*m33) - (q*m23))
        b27 = 0
        b28 = 1
        b29 = -s/q
        
        B1 = np.vstack([[b11,b12,b13,b14,b15,b16,b17,b18,b19],[b21,b22,b23,b24,b25,b26,b27,b28,b29]])
        B = np.vstack([B,B1])
    
        # Now make epsilon #
        e1 = xa- (x0 + (f*r/q))
        e2 = ya- (y0 + (f*s/q))
        
        epsilon1 = np.vstack([[e1],[e2]])
        epsilon = np.vstack([epsilon,epsilon1])
        
    # Step 2: Add the horizon constraint equations to B and epsilon. This adds two more equations to the system #
    H11 = ((-1/math.sqrt( 1-(math.cos(phi)*math.cos(omega))**2 )) * (-math.cos(phi)*math.sin(omega))) + math.acos(math.cos(phi)*math.cos(omega))-xi
    H12 = ((-1/math.sqrt( 1-(math.cos(phi)*math.cos(omega))**2 )) * (-math.cos(omega)*math.sin(phi))) + math.acos(math.cos(phi)*math.cos(omega))-xi
    H21 = ((1/( 1+ (-math.sin(phi)/(math.cos(phi)*math.sin(omega)))**2 )) * ((math.sin(phi)*math.cos(phi)*math.cos(omega))/((math.cos(phi)*math.sin(omega))**2))) + math.atan2(-math.sin(phi),math.cos(phi)*math.sin(omega))-psi
    H22 = ((1/( 1+ (-math.sin(phi)/(math.cos(phi)*math.sin(omega)))**2 )) * (((-math.cos(phi)*math.sin(omega)*math.cos(phi))  - (math.sin(phi)*math.sin(phi)*math.sin(omega)))/((math.cos(phi)*math.sin(omega))**2))) + math.atan2(-math.sin(phi),math.cos(phi)*math.sin(omega))-psi
        
    H = np.vstack([[H11,H12,0,0,0,0,0,0,0],[H21,H22,0,0,0,0,0,0,0]])
        
    B = np.vstack([B,H])
    epsilon = np.vstack([epsilon,np.vstack([-math.acos(math.cos(phi)*math.cos(omega))-xi,math.atan2(-math.sin(phi),(math.cos(phi)*math.sin(omega)))-psi])])
        
    # Step 2.5: Create the weight matrix where weights are assigned to each parameter. Give the horizon equations a weight of 10e12 following the pape, while others have a weight of 1 #
    W = np.zeros([14,14])
    np.fill_diagonal(W,1)
    W[12,12] = 10**12
    W[13,13] = 10**12
    
    # Step 3: Solve for corrections to each parameter using the weighted normal equation #
    Delta = np.linalg.inv(np.transpose(B) @ (W @ B)) @ (np.transpose(B) @ (W @ epsilon))
        
    # Step 4: Apply the corrections to the parameters #
    omega = float(omega+Delta[0])
    phi = float(phi+Delta[1])
    kappa = float(kappa+Delta[2])
    XL = float(XL+Delta[3])
    YL = float(YL+Delta[4])
    ZL = float(ZL+Delta[5])
    x0 = float(x0+Delta[6])
    y0 = float(y0+Delta[7])
    f = float(f+Delta[8])
    
    # Step 5: Add the new values to the values matrix, and calculate the change in each parameter #    
    allvals = np.vstack([allvals,[omega,phi,kappa,XL,YL,ZL,x0,y0,f]])
    changes = allvals[iteration,:]-allvals[iteration-1,:]
        
        

plt.figure()
plt.plot(allvals[:,8])







