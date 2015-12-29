import cv2
import numpy as np
import sys
import copy


##Function to display the current image
def draw_img():
    global curr_img
    cv2.imshow('ELE v1.0 :: Press ESC to EXIT, h for HELP',curr_img)


##Function to listen to mouse events
def gather_pts(event, x, y, flags, param):
    global first_img
    global op_flag
    global l_pts_list
    global r_pts_list
    global sel_point
    global trial
    ##On the event of button click, below routine records points
    if event == cv2.EVENT_LBUTTONDBLCLK:
        ##If current operation flag is p, all click points are gathered in a list
	if op_flag == "p":
 	    if trial == 0:
	    	l_pts_list.append(float(x))
	    	l_pts_list.append(float(y))
		trial = 1
                print "Double click detected. X value: %f, Y value: %f" % (x,y)
	    else:
		r_pts_list.append(float(x-first_img.shape[1]))
	    	r_pts_list.append(float(y))
		trial = 0
                print "Double click detected. X value: %f, Y value: %f" % (x-first_img.shape[1],y)
        ##If current operation flag is P, only one click point is recorded
	if op_flag == "P":
                if x<=first_img.shape[1]:
		    sel_point[:2] = [float(x),float(y)]
                    sel_point[3] = 'l'
                    print "Double click detected. X value: %f, Y value: %f" % (x,y)
                else:
		    sel_point[:2] = [float(x-first_img.shape[1]),float(y)]
                    sel_point[3] = 'r'
                    print "Double click detected. X value: %f, Y value: %f" % (x-first_img.shape[1],y)


