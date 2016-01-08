import numpy as np
import cv2
import sys

def corner_det(in_img):
	objpts = np.zeros((6*7,3), np.float32) # making object points
	objpts[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

	objectpoints, imagepoints = [], [] # containers for world points and image points
	gray = cv2.cvtColor(in_img,cv2.COLOR_BGR2GRAY)

	retval, corners = cv2.findChessboardCorners(gray, (7,6),None) # finding the chess board corners

	if retval == True: # storing and printing world and image points
	    objectpoints.append(objpts)
            print objectpoints
            wf=open('oworld.txt','w')
            for item in objectpoints:
		wf.write("%s\n" % item)
	    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # condition for stopping
	    corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),crit)
	    imagepoints.append(corners2)
            print imagepoints
            imf=open('oimage.txt','w')
            for item in imagepoints:
		imf.write("%s\n" % item)
	    
	    in_img = cv2.drawChessboardCorners(in_img, (7,6), corners2,retval) # showing the corners
	    cv2.imshow('BCCD v1.0',in_img)
	    cv2.waitKey()

	cv2.destroyAllWindows()

#Program execution starts here

print ("**Starting Basic Chess Corner Detector v1.0**");

arg_list = []

for arg in sys.argv:
    arg_list.append(arg)
if len(arg_list) == 2:
    try:
        image = cv2.imread(arg_list[1])
        corner_det(image)

    except:
        print "EXITED with ERROR: Unable to read/process file from the path!"

else:
    print "EXITED with ERROR: Incorrect argument count."
