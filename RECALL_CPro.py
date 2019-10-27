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

- Finally got convergence at Miami using an undistorted image, freeing all 9 parameters, and using relatively small weights (100 or less) for horizon eqs. Will test with distorted image.

- 10/21: Standard space resection works at Miami and Folly north. Added horizon at Folly North, and works if f is not included (though So is a bit higher than without horizon). However,
         getting impossible values for some parameters (e.g. negetive ZL). Blows up immedietly if f is included. 
         
         Using an undistorted Miami image didn't help- same result. Also, freeing f without x0 and y0 also works, but error is larger and some values still impossible 
         

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
from scipy.optimize import least_squares
import cv2
from scipy.interpolate import interp2d,griddata,RectBivariateSpline

os.chdir(wd)


# Load in control points #
f = open(wd+'GCPs_im.pkl','rb')
f2 =  open(wd+'GCPs_lidar.pkl','rb')     
gcps_im = pickle.load(f)
gcps_lidar = pickle.load(f2)



#=============================================================================#
# Direct Linear Transformation to get initial estimates of IOPs and EOPs #
#=============================================================================#

# Set up the matricies #
#A = np.empty([0,11])
#for i in range(0,len(gcps_im)):
#    v1 = np.array([ gcps_lidar[i,0],gcps_lidar[i,1],gcps_lidar[i,2],1,0,0,0,0,-gcps_im[i,0]*gcps_lidar[i,0],-gcps_im[i,0]*gcps_lidar[i,1],-gcps_im[i,0]*gcps_lidar[i,2] ])
#    v2 = np.array([ 0,0,0,0,gcps_lidar[i,0],gcps_lidar[i,1],gcps_lidar[i,2],1,-gcps_im[i,1]*gcps_lidar[i,0],-gcps_im[i,1]*gcps_lidar[i,1],-gcps_im[i,1]*gcps_lidar[i,2] ])
#    vv = np.vstack([v1,v2])
#    A = np.vstack([A,vv])
#    
#O = np.empty([0,1])
#for i in range(0,len(gcps_im)):
#    p1 = gcps_im[i,0]
#    p2 = gcps_im[i,1]
#    vv = np.vstack([p1,p2])
#    O = np.vstack([O,vv])
#    
## Solve for 11 DLT parameters through least-squares (use pseudo-inverse) #
#L = np.linalg.inv(np.transpose(A) @ A) @ (np.transpose(A) @ O)
#
#
## Recover initial estimates of IOPs and EOPs from DLT parameters (L) #
#a1 = float(L[0]);b1 = float(L[1]);c1 = float(L[2]);d1 = float(L[3]);a2 = float(L[4]);b2 = float(L[5]);c2 = float(L[6]);d2 = float(L[7]);a3 = float(L[8]);b3 = float(L[9]);c3 = float(L[10])
#
#x0 = ( np.array([a1,b1,c1]) @ np.array([a3,b3,c3]) ) / ( np.array([a3,b3,c3]) @ np.array([a3,b3,c3]) )
#y0 = ( np.array([a2,b2,c2]) @ np.array([a3,b3,c3]) ) / ( np.array([a3,b3,c3]) @ np.array([a3,b3,c3]) )
#f = math.sqrt( (( np.array([a1,b1,c1]) @ np.array([a1,b1,c1]) ) / ( np.array([a3,b3,c3]) @ np.array([a3,b3,c3]) )) - x0**2 ) 
#XL,YL,ZL = np.linalg.inv(np.transpose(np.vstack([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]]))) @ np.array([-d1,-d2,-1])




#=============================================================================#
# Direct initial approximations #
#=============================================================================#

# Initial approx of XL,YL as origin and guessed elevation for ZL #
XL1 = 5
YL1 = 3
ZL1 = 16


# Initial approx for f based on geometry #
frames = glob.glob('frame'+'*')
frame = frames[0]  
img = mpimg.imread(wd+'/'+frame)
w = len(img[1,:,:])
a = math.radians(60)
f1 = (w/2)*(1/math.tan(a/2))


# Use two horizon points with C-Pro method to get initial approx for omega, phi, and kappa #
figure = plt.figure()
ax = figure.add_subplot(111)        
img = mpimg.imread(wd+'/'+frame)
imgplot = plt.imshow(img)
pt = plt.ginput(n=2,show_clicks=True)  

xa = pt[0][0]
xb = pt[1][0]
ya = pt[0][1]
yb = pt[1][1]
psi = math.atan2(ya-yb,xb-xa)

d = ((ya*xb)-(yb*xa))/math.sqrt( (xb-xa)**2+(yb-ya)**2 )
C = math.atan2(d,f1)
D = math.sqrt( (ZL1+6371000)**2 - (6371000**2) )
beta = math.acos( (ZL1+ (.42*(D**2)/6371000))/D )
xi = beta-C

phi1 = -math.asin( math.sin(xi)*math.sin(psi) )
omega1 = math.acos( math.cos(xi)/math.sqrt((math.cos(psi)**2) + (math.cos(xi)**2 * math.sin(psi)**2)) )
kappa1 = -math.pi/4


# Use center of image as initial approx for principal points #
x01 = len(img[1,:,1])/2
y01 = len(img[:,1,1])/2
 


# Least-squares with manual LM algorithm #
#while np.max(np.abs(changes))>.00001:
#    
#    iteration = iteration+1
#    if iteration>1200:
#        print('Error: The soultion is likely diverging')
#        break
#    else:
#        pass
#    
#    if iteration == 1:
#        vals = np.array([omega,phi,kappa,XL,YL,ZL,f])
#        allvals = np.vstack([allvals,vals])
#    else:
#        pass
#    
#    
#    # Step 0: calculate the elements of the M matrix as shown in the book #
#    m11 = math.cos(phi)*math.cos(kappa)
#    m12 = (math.sin(omega)*math.sin(phi)*math.cos(kappa)) + (math.cos(omega)*math.sin(kappa))
#    m13 = (-math.cos(omega)*math.sin(phi)*math.cos(kappa)) + (math.sin(omega)*math.sin(kappa))
#    m21 = -math.cos(phi)*math.sin(kappa)
#    m22 = (-math.sin(omega)*math.sin(phi)*math.sin(kappa)) + (math.cos(omega)*math.cos(kappa))
#    m23 = (math.cos(omega)*math.sin(phi)*math.sin(kappa)) + (math.sin(omega)*math.cos(kappa))
#    m31 = math.sin(phi)
#    m32 = -math.sin(omega)*math.cos(phi)
#    m33 = math.cos(omega)*math.cos(phi)
#    
#    # Form gradient vector and residual vector for each point #
#    gv = list()
#    R = list()
#    H = list()
#    for i in range(0,len(gcps_lidar)):
#        XA = gcps_lidar[i,0]
#        YA = gcps_lidar[i,1]
#        ZA = gcps_lidar[i,2]
#        xa = gcps_im[i,0]
#        ya = gcps_im[i,1]
#        
#        # Deltas #
#        deltaX = XA-XL
#        deltaY = YA-YL
#        deltaZ = ZA-ZL
#        
#        # Numerators and denominator of collinearity conditions #
#        q = (m31*deltaX)+(m32*deltaY)+(m33*deltaZ)
#        r = (m11*deltaX)+(m12*deltaY)+(m13*deltaZ)
#        s = (m21*deltaX)+(m22*deltaY)+(m23*deltaZ)
#        
#        # Now all the b's of the B (Jacobian) matrix #
#        b11 = (f/q**2) * ( (r*((-m33*deltaY)+(m32*deltaZ))) - (q*((-m13*deltaY)+(m12*deltaZ))) )
#        b12 = (f/q**2) * ( (r*( (math.cos(phi)*deltaX) + (math.sin(omega)*math.sin(phi)*deltaY) - (math.cos(omega)*math.sin(phi)*deltaZ) )) - (q*( (-math.sin(phi)*math.cos(kappa)*deltaX) + (math.sin(omega)*math.cos(phi)*math.cos(kappa)*deltaY) - (math.cos(omega)*math.cos(phi)*math.cos(kappa)*deltaZ) ))       )
#        b13 = (-f/q) * ((m21*deltaX)+(m22*deltaY)+(m23*deltaZ))
#        b14 = (f/q**2) * ((r*m31) - (q*m11))
#        b15 = (f/q**2) * ((r*m32) - (q*m12))
#        b16 = (f/q**2) * ((r*m33) - (q*m13))
#        b17 = 1
#        b18 = 0
#        b19 = -r/q
#    
#        b21 = (f/q**2) * ( (s*((-m33*deltaY)+(m32*deltaZ))) - (q*((-m23*deltaY)+(m22*deltaZ))) )
#        b22 = (f/q**2) * ( (s*( (math.cos(phi)*deltaX) + (math.sin(omega)*math.sin(phi)*deltaY) - (math.cos(omega)*math.sin(phi)*deltaZ) )) - (q*( (-math.sin(phi)*math.sin(kappa)*deltaX) - (math.sin(omega)*math.cos(phi)*math.sin(kappa)*deltaY) + (math.cos(omega)*math.cos(phi)*math.sin(kappa)*deltaZ) ))       )
#        b23 = (f/q) * ((m11*deltaX)+(m12*deltaY)+(m13*deltaZ))
#        b24 = (f/q**2) * ((s*m31) - (q*m21))
#        b25 = (f/q**2) * ((s*m32) - (q*m22))
#        b26 = (f/q**2) * ((s*m33) - (q*m23))
#        b27 = 0
#        b28 = 1
#        b29 = -s/q
#        
#        # Calc F and G #
#        Fi = x0-(f*(r/q))
#        Gi= y0-(f*(s/q))
#        
#        # Create this point's gradient vector #
#        gvi = np.vstack([b11,b21,b12,b22,b13,b23,b14,b24,b15,b25,b16,b26,b19,b29])
#        gv.append([gvi])
#        
#        # Create this point's Hessian matrix #
#        Hi = np.empty([6,6])
#        for row in range(1,7):
#            for col in range(1,7):
#                first = np.hstack([eval('b1'+str(col)),eval('b2'+str(col))])
#                second = np.vstack([eval('b1'+str(row)),eval('b2'+str(row))])
#                Hi[row-1,col-1] = float(first@second)
#        H.append(Hi)
#        
#        # Create this point's residual vector #
#        Ri = np.vstack([Fi-xa,Gi-ya])
#        R.append(Ri)
#
                
        
    

        
        





# Non-linear least squares using Python optimization function #

def model(unknownsVec):
    omega = unknownsVec[0]
    phi = unknownsVec[1]
    kappa = unknownsVec[2]
    XL = unknownsVec[3]
    YL = unknownsVec[4]
    ZL = unknownsVec[5]
    f = unknownsVec[6]
    
    m11 = math.cos(phi)*math.cos(kappa)
    m12 = (math.sin(omega)*math.sin(phi)*math.cos(kappa)) + (math.cos(omega)*math.sin(kappa))
    m13 = (-math.cos(omega)*math.sin(phi)*math.cos(kappa)) + (math.sin(omega)*math.sin(kappa))
    m21 = -math.cos(phi)*math.sin(kappa)
    m22 = (-math.sin(omega)*math.sin(phi)*math.sin(kappa)) + (math.cos(omega)*math.cos(kappa))
    m23 = (math.cos(omega)*math.sin(phi)*math.sin(kappa)) + (math.sin(omega)*math.cos(kappa))
    m31 = math.sin(phi)
    m32 = -math.sin(omega)*math.cos(phi)
    m33 = math.cos(omega)*math.cos(phi)
    
#    P = np.vstack([[1,0,0,0],[0,1,0,0],[0,0,-1/f,0]])
#    R = np.vstack([[m11,m12,m13,0],[m21,m22,m23,0],[m31,m32,m33,0],[0,0,0,1]])
#    T = np.vstack([[1,0,0,-XL],[0,1,0,YL],[0,0,1,-ZL],[0,0,0,1]])
#    XYZ = np.vstack([np.transpose(gcps_lidar),np.ones(len(gcps_lidar))])
#    
#    xyw = ((P@R)@T)@XYZ
#    
#    xy = np.divide(xyw,xyw[2,:])[0:2,:]
#    uv = np.subtract(np.vstack([x01,y01]),xy)
#    
#    H1 = math.acos(math.cos(phi)*math.cos(omega))-xi
#    H2 = math.atan2(-math.sin(phi),math.cos(phi)*math.sin(omega))-psi
#    uv = np.hstack([uv,np.hstack([np.vstack([H1,H1]),np.vstack([H2,H2])])])
    
    u = np.empty([0,1])
    v = np.empty([0,1])
    for i in range(0,len(gcps_lidar)):
        XA = gcps_lidar[i,0]
        YA = gcps_lidar[i,1]
        ZA = gcps_lidar[i,2]
        
        dx = XA-XL
        dy = YA-YL
        dz = ZA-ZL
        
        u1 = x01-(f*(((m11*dx)+(m12*dy)+(m13*dz))/((m31*dx)+(m32*dy)+(m33*dz))))
        v1 = y01-(f*(((m21*dx)+(m22*dy)+(m23*dz))/((m31*dx)+(m32*dy)+(m33*dz))))
        
        u = np.vstack([u,u1])
        v = np.vstack([v,v1])
    
    uv = np.hstack([u,v])
   
    
    return uv
    
def calcResid(unknownsVec,observations):
    uv = model(unknownsVec)
    
#    observations = np.transpose(observations)
#    observations = np.hstack([observations,np.vstack([0,0]),np.vstack([0,0])])

    resid_uv = np.subtract(observations,uv)
    resid = np.reshape(resid_uv,[np.size(resid_uv)])
    resid1d = np.sqrt(np.add(resid_uv[0,:]**2,resid_uv[1,:]**2))
    
    
    return resid
    
initApprox = np.hstack([omega1,phi1,kappa1,XL1,YL1,ZL1,f1])
boundsVec = ((-math.pi*2,-math.pi*2,-math.pi*2,XL1-50,YL1-50,0,f1-500),(math.pi*2,math.pi*2,math.pi*2,XL1+50,YL1+50,math.inf,f1+500))

results = least_squares(calcResid,initApprox,bounds=boundsVec,jac='3-point',method='dogbox',max_nfev=5000,x_scale='jac',loss='cauchy',tr_solver='exact',args=[gcps_im])
finalVals = results['x']
CPE = math.sqrt(finalVals[3]**2 + finalVals[4]**2 + (15-finalVals[5])**2)
rmsResid = np.sqrt(np.sum(results['fun']**2)/len(results['fun']))



# Compute the re-proj image coords to check #
omega = finalVals[0]
phi = finalVals[1]
kappa = finalVals[2]
XL = finalVals[3]
YL = finalVals[4]
ZL = finalVals[5]
f = finalVals[6]

m11 = math.cos(phi)*math.cos(kappa)
m12 = (math.sin(omega)*math.sin(phi)*math.cos(kappa)) + (math.cos(omega)*math.sin(kappa))
m13 = (-math.cos(omega)*math.sin(phi)*math.cos(kappa)) + (math.sin(omega)*math.sin(kappa))
m21 = -math.cos(phi)*math.sin(kappa)
m22 = (-math.sin(omega)*math.sin(phi)*math.sin(kappa)) + (math.cos(omega)*math.cos(kappa))
m23 = (math.cos(omega)*math.sin(phi)*math.sin(kappa)) + (math.sin(omega)*math.cos(kappa))
m31 = math.sin(phi)
m32 = -math.sin(omega)*math.cos(phi)
m33 = math.cos(omega)*math.cos(phi)

P = np.vstack([[1,0,0,0],[0,1,0,0],[0,0,-1/f,0]])
R = np.vstack([[m11,m12,m13,0],[m21,m22,m23,0],[m31,m32,m33,0],[0,0,0,1]])
T = np.vstack([[1,0,0,-XL],[0,1,0,YL],[0,0,1,-ZL],[0,0,0,1]])
XYZ = np.vstack([np.transpose(gcps_lidar),np.ones(len(gcps_lidar))])

xyw = ((P@R)@T)@XYZ

xy = np.divide(xyw,xyw[2,:])[0:2,:]
uv = np.subtract(np.vstack([x0,y0]),xy)



##=============================================================================#
## Manual non-linear least squares to solve for each EOP and IOP #
##=============================================================================#
iteration = 0
allvals = np.empty([0,7]) # Matrix to store the values (rows) for each parameter (columns) as the least squares solution iterates #
changes = np.array([10,10,10,10,10,10,10])
So2 = np.empty([0,1])
while np.max(np.abs(changes))>.00001:
    
    iteration = iteration+1
    if iteration>1200:
        print('Error: The soultion is likely diverging')
        break
    else:
        pass
    
    if iteration == 1:
        vals = np.array([omega,phi,kappa,XL,YL,ZL,f])
        allvals = np.vstack([allvals,vals])
    else:
        pass
    
    
    # Step 0: calculate the elements of the M matrix as shown in the book #
    m11 = math.cos(phi)*math.cos(kappa)
    m12 = (math.sin(omega)*math.sin(phi)*math.cos(kappa)) + (math.cos(omega)*math.sin(kappa))
    m13 = (-math.cos(omega)*math.sin(phi)*math.cos(kappa)) + (math.sin(omega)*math.sin(kappa))
    m21 = -math.cos(phi)*math.sin(kappa)
    m22 = (-math.sin(omega)*math.sin(phi)*math.sin(kappa)) + (math.cos(omega)*math.cos(kappa))
    m23 = (math.cos(omega)*math.sin(phi)*math.sin(kappa)) + (math.sin(omega)*math.cos(kappa))
    m31 = math.sin(phi)
    m32 = -math.sin(omega)*math.cos(phi)
    m33 = math.cos(omega)*math.cos(phi)
    
    # Step 1: Form the B (Jacobian) and e (observation) matricies 
    B = np.empty([0,7])
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
        
        # Numerators and denominator of collinearity conditions #
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
        
        B1 = np.vstack([[b11,b12,b13,-b14,-b15,-b16,b19],[b21,b22,b23,-b24,-b25,-b26,b29]])
        B = np.vstack([B,B1])
    
        # Now make epsilon #
        e1 = xa- (x0 - (f*r/q))
        e2 = ya- (y0 - (f*s/q))
        
        epsilon1 = np.vstack([[e1],[e2]])
        epsilon = np.vstack([epsilon,epsilon1])
        
    # Step 2: Add the horizon constraint equations to B and epsilon. This adds two more equations to the system #
    H11 = (math.sin(omega)*math.cos(phi))/math.sqrt(1-(math.cos(omega)**2 * math.cos(phi)**2))
    H12 = (math.cos(omega)*math.sin(phi))/math.sqrt(1-(math.cos(omega)**2 * math.cos(phi)**2))
    H21 = (math.sin(omega)*(1/math.tan(omega))*math.tan(phi))/(math.tan(phi)**2 + math.sin(omega)**2)
    H22 = -(math.sin(omega)*((1/math.cos(phi))**2))/(math.tan(phi)**2 + math.sin(omega)**2)
       
    H = np.vstack([np.hstack([H11,H12,np.zeros(len(vals)-2)]) ,np.hstack([H21,H22,np.zeros(len(vals)-2)]) ])  
    B = np.vstack([B,H])
    
    He1 = -(math.acos(math.cos(phi)*math.cos(omega))-xi)
    He2 = -(math.atan2(-math.sin(phi),math.cos(phi)*math.sin(omega))-psi)
    epsilon = np.vstack([epsilon,np.vstack([He1,He2])])
        
    # Step 2.5: Create the weight matrix where weights are assigned to each parameter. Give the horizon equations a weight of 10e12 following the pape, while others have a weight of 1 #
    W = np.zeros([len(B[:,1]),len(B[:,1])])
    np.fill_diagonal(W,1)
    W[len(B[:,1])-2,len(B[:,1])-2] = 1e6
    W[len(B[:,1])-1,len(B[:,1])-1] = 1e6
    
    # Step 3: Solve for corrections to each parameter using the weighted normal equation #
    Delta = np.linalg.inv(np.transpose(B) @ (W @ B)) @ (np.transpose(B) @ (W @ epsilon))
    
    v = (B@Delta)-epsilon
    
    # Step 4: Apply the corrections to the parameters #
    omega = float(omega+Delta[0])
    phi = float(phi+Delta[1])
    kappa = float(kappa+Delta[2])
    XL = float(XL+Delta[3])
    YL = float(YL+Delta[4])
    ZL = float(ZL+Delta[5])
#    x0 = float(x0+Delta[6])
#    y0 = float(y0+Delta[7])
    f = float(f+Delta[6])
    
    # Step 5: Add the new values to the values matrix, and calculate the change in each parameter #    
    allvals = np.vstack([allvals,[omega,phi,kappa,XL,YL,ZL,f]])
    changes = allvals[iteration,:]-allvals[iteration-1,:]
    
    So = math.sqrt((np.transpose(v) @ v)/(len(B)-len(Delta)))
    So2 = np.vstack([So2,So])
        
        




#==============================================================================#
# Rectification. Set up spatial grid and inverse map each xyz point in the grid to a UV to get its color (billinear interpolation)
#==============================================================================#
# Extract all the final parameters and re-calculate M #
omega = finalVals[0]
phi = finalVals[1]
kappa = finalVals[2]
XL = finalVals[3]
YL = finalVals[4]
ZL = finalVals[5]
f = finalVals[6]

m11 = math.cos(phi)*math.cos(kappa)
m12 = (math.sin(omega)*math.sin(phi)*math.cos(kappa)) + (math.cos(omega)*math.sin(kappa))
m13 = (-math.cos(omega)*math.sin(phi)*math.cos(kappa)) + (math.sin(omega)*math.sin(kappa))
m21 = -math.cos(phi)*math.sin(kappa)
m22 = (-math.sin(omega)*math.sin(phi)*math.sin(kappa)) + (math.cos(omega)*math.cos(kappa))
m23 = (math.cos(omega)*math.sin(phi)*math.sin(kappa)) + (math.sin(omega)*math.cos(kappa))
m31 = math.sin(phi)
m32 = -math.sin(omega)*math.cos(phi)
m33 = math.cos(omega)*math.cos(phi)


# Set up the world grid #
z = 0
dx = 5
dy = 5
xg = np.arange(-1000,2000,dx)
yg = np.arange(100,2000,dy)
xgrd,ygrd = np.meshgrid(xg,yg)
zgrd = np.zeros([len(xgrd[:,1]),len(xgrd[1,:])])+z

#P = np.vstack([[1,0,0,0],[0,1,0,0],[0,0,-1/f,0]])
#R = np.vstack([[m11,m12,m13,0],[m21,m22,m23,0],[m31,m32,m33,0],[0,0,0,1]])
#T = np.vstack([[1,0,0,-XL],[0,1,0,-YL],[0,0,1,-ZL],[0,0,0,1]])
#XYZ = np.vstack([np.reshape(xgrd,[1,np.size(xgrd)]),np.reshape(ygrd,[1,np.size(ygrd)]),np.ones([1,np.size(xgrd)])+z,np.ones([1,np.size(xgrd)])])
#
#UV1 = ((P@R)@T)@XYZ
#UV = np.divide(UV1,UV1[2,:])
#
#u = np.reshape(np.reshape(xgrd,[1,np.size(xgrd)]),[len(xgrd[:,1]),len(xgrd[1,:])])
#v = np.reshape(UV[1,:],[len(xgrd[:,1]),len(xgrd[1,:])])

u = x0 - (f*(((m11*(xgrd-XL)) + (m12*(ygrd-YL)) + (m13*(zgrd-ZL))) / ((m31*(xgrd-XL)) + (m32*(ygrd-YL)) + (m33*(zgrd-ZL)))))
v = y0 - (f*(((m21*(xgrd-XL)) + (m22*(ygrd-YL)) + (m23*(zgrd-ZL))) / ((m31*(xgrd-XL)) + (m32*(ygrd-YL)) + (m33*(zgrd-ZL)))))




uIM = np.arange(len(img[1,:,1]))
vIM = np.arange(len(img[:,1,1]))

uIM = np.reshape(uIM,[1,len(uIM)])
vIM = np.reshape(vIM,[len(vIM),1])

uIMg,vIMg = np.meshgrid(uIM,vIM)

uIMgr = np.reshape(uIMg,[np.size(uIMg)])
vIMgr = np.reshape(vIMg,[np.size(vIMg)])

rr = np.reshape(img[:,:,0],[np.size(vIMg)])
gr = np.reshape(img[:,:,1],[np.size(vIMg)])
br = np.reshape(img[:,:,2],[np.size(vIMg)])

col_r = griddata((uIMgr,vIMgr),rr,(u,v))
col_g = griddata((uIMgr,vIMgr),gr,(u,v))
col_b = griddata((uIMgr,vIMgr),br,(u,v))
                 











u = np.reshape(np.reshape(xgrd,[1,np.size(xgrd)]),[len(xgrd[:,1]),len(xgrd[1,:])])
v = np.reshape(UV[1,:],[len(xgrd[:,1]),len(xgrd[1,:])])

u = x0-u
v = y0-v


# Calc UV with equations instead #
#u = x0 - (f*(((m11*(xgrd-XL)) + (m12*(ygrd-YL)) + (m13*(zgrd-ZL))) / ((m31*(xgrd-XL)) + (m32*(ygrd-YL)) + (m33*(zgrd-ZL)))))
#v = y0 - (f*(((m21*(xgrd-XL)) + (m22*(ygrd-YL)) + (m23*(zgrd-ZL))) / ((m31*(xgrd-XL)) + (m32*(ygrd-YL)) + (m33*(zgrd-ZL)))))


# Find color of each UV by finding color at that location in the image # 
uIM = np.arange(len(img[1,:,1]))
vIM = np.arange(len(img[:,1,1]))

iR = interp2d(np.reshape(uIM,[1,len(uIM)]),np.reshape(vIM,[len(vIM),1]),img[:,:,0],fill_value=0) # Create interpolation object of image locations #
iG = interp2d(np.reshape(uIM,[1,len(uIM)]),np.reshape(vIM,[len(vIM),1]),img[:,:,1],fill_value=0)
iB = interp2d(np.reshape(uIM,[1,len(uIM)]),np.reshape(vIM,[len(vIM),1]),img[:,:,2],fill_value=0)


col_r = np.zeros([len(u[:,1]),len(u[1,:])])
col_g = np.zeros([len(u[:,1]),len(u[1,:])])
col_b = np.zeros([len(u[:,1]),len(u[1,:])])
for row in range(0,len(u[:,1])):
    for col in range(0,len(u[1,:])):
        col_r1 = float(iR(u[row,col],v[row,col])) # Interpolate each calculated UV to image to get color #
        col_g1 = float(iG(u[row,col],v[row,col]))
        col_b1 = float(iB(u[row,col],v[row,col]))
        
        col_r[row,col] = col_r1
        col_g[row,col] = col_g1
        col_b[row,col] = col_b1


# Flip the final matricies up-down #
col_r = np.flipud(col_r)
col_g = np.flipud(col_g)
col_b = np.flipud(col_b)


# Create and plot the rectified image #
im_rectif = np.stack([col_r,col_g,col_b],axis=2)

fig = plt.figure()
plt.imshow(im_rectif,extent=[(-.5*dx)+min(xg),max(xg)+(.5*dx),min(yg)-(.5*dy),max(yg)+(.5*dy)],interpolation='bilinear')
plt.axis('equal')
fig.show()
        
        
        
        
        
        



















