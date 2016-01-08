import numpy as np
import cv2
import math
import sys
import random
rng = random.Random()
rng.seed(3)

def calib(world,image): # main calibration method
    
    A = np.zeros((2*world.shape[0],12)) # creating A matrix from input world and image points
    for i in range(A.shape[0]/2):
        A[i*2,0:3] = world[i,:3]
        A[i*2,3] = 1
        A[i*2,8:11] = -image[i,0]*world[i,:3]
        A[i*2,11] = -image[i,0]
        A[(i*2)+1,4:7] = world[i,:3]
        A[(i*2)+1,7] = 1
        A[(i*2)+1,8:11] = -image[i,1]*world[i,:3]
        A[(i*2)+1,11] = -image[i,1]

    u, s, V = np.linalg.svd(A, full_matrices=True) # using svd to get weighted M
    wM = V[11,:].reshape(3,4)
    a1 = np.transpose(wM[0,:3]) # getting a1, a2, a3, and b from weighted M
    a2 = np.transpose(wM[1,:3])
    a3 = np.transpose(wM[2,:3])
    b = wM[:,3]

    mp = 1/np.linalg.norm(a3) # finding intrinsic parameters
    u0 = math.pow(mp,2)*np.dot(a1,a3)
    v0 = math.pow(mp,2)*np.dot(a2,a3)
    av = np.sqrt((math.pow(mp,2)*np.dot(a2,a2))-(v0*v0))
    s = math.pow(mp,4)/av*np.dot(np.cross(a1,a3),np.cross(a2,a3))
    au = np.sqrt((mp*mp*np.dot(a1,a1))-s*s-u0*u0)

    ks = np.zeros((3,3)) # creating a matrix out of intrinsic parameters
    ks[0][0] = au
    ks[0][1] = s
    ks[1][1],ks[2][2] = av, 1
    ks[0][2], ks[1][2] = u0, v0

    ep = np.sign(b[2]) # finding extrinsic parameters
    kinv = np.linalg.inv(ks)
    ts = ep*mp*(np.dot(kinv,b))
    r3 = ep*mp*a3
    r1 = math.pow(mp,2)/av*np.cross(a2,a3)
    r2 = np.cross(r3, r1)

    finS = np.zeros((3,4)) # creating projection matrix
    finS[:,:3] = ks
    finRTS = np.zeros((4,4))
    finRTS[0,:3], finRTS[1,:3], finRTS[2,:3], finRTS[3,3] = r1, r2, r3, 1
    finRTS[0][3], finRTS[1][3], finRTS[2][3] = ts[0], ts[1], ts[2]
    M = np.dot(finS,finRTS) # final projection matrix

    return ks, ts, r1, r2, r3, M

def calib_verify(M): # main calibration verification method
    global world 
    global image
    global n_flag
    global Rval

    wp = np.ones((4,1)) # creating matrices for storing temporary results
    cimg = np.ones((world.shape[0],3))
    err = np.ones((world.shape[0],1))
    
    for r in range(world.shape[0]):
        wp[0][0],wp[1][0],wp[2][0] = world[r][0], world[r][1], world[r][2] # getting homogenous world point

        cimg[r,:] = np.transpose(np.dot(M,wp)) # computing image points using projection matrix

        cimg[r][0] = cimg[r][0]/cimg[r][2] # homogenizing image points
        cimg[r][1] = cimg[r][1]/cimg[r][2]

        errval = ((np.sqrt(math.pow((image[r][0]-cimg[r][0]),2)+math.pow((image[r][1]-cimg[r][1]),2)))) # calculating error values for each point
        if (n_flag == 1) and (errval < Rval[2]):
            err[r] = errval 
        else:
            err[r] = errval

    return np.sum(err)/len(err), len(err) # return overall error

def print_val(Kmat, Tmat, Rmat1, Rmat2, Rmat3,Err): # method to print all values
        print '\nIntrinsic parameters are as follows.\nu0 = %f, v0 = %f, alphaU = %f, alphaV = %f, s = %f' \
        % (Kmat[0][2],Kmat[1][2], Kmat[0][0], Kmat[1][1], round(Kmat[0][1]))
        print 'Extrinsic parameters are as follows.\nT* matrix is as below.'
        print Tmat
        print 'R* matrix is as below.'
        print Rmat1, Rmat2, Rmat3
        print '\nMean error between the known and computed image points = %f\n' % Err
