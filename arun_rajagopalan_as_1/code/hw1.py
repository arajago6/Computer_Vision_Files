import cv2
import numpy as np
import sys
import copy

#Function to plot gradient vectors
def plot_gradvec(xderv,yderv):
    global curr_img
    global pi_pos
    
    crow, ccol = curr_img.shape[0], curr_img.shape[1]
    length = pi_pos/2
    outimg = np.zeros((crow,ccol),np.uint8)
    outimg = copy.copy(curr_img)
    for row in range(crow):	
	crrow = row*pi_pos
	if crrow >= crow:
	    break
   	for col in range(ccol):
	    crcol = col*pi_pos
	    if crcol >= ccol:
		break
	    cv2.circle(outimg,(crcol,crrow), 1, (255,255,255))
	    #Length of plotline
	    vlength = length*curr_img[crrow][crcol]*0.055
	    #Angle of plotline
	    angle = np.arctan2(yderv[crrow][crcol],xderv[crrow][crcol])
	    #Plotline endpoint co-ordinates
	    p2x = int(crrow + vlength * np.cos (angle+np.pi/180))
	    p2y = int(crcol + vlength * np.sin (angle+np.pi/180))
	    #Draw plotline
	    cv2.line(outimg,(crcol,crrow),(p2y,p2x),(255,255,255),1)
    return outimg

#Function to calculate derivatives along axis 
def der_conv(axis):
    global curr_img

    sft = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])   #Sobel x filter
    if axis == 'Y':
	sft = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]	#Sobel y filter
    crow, ccol = curr_img.shape[0], curr_img.shape[1]
    outimg = np.zeros((crow,ccol),np.float)
    for i in range(1,crow-1):
    	for j in range (1,ccol-1):
	    outimg[i][j] = ((curr_img[i-1:i+2,j-1:j+2]*sft).sum())
    return outimg

#Function to apply averaging filter
def avg_filter():
    global curr_img
    global b_pos
    
    filt = np.ones((b_pos,b_pos),np.uint8)
    crow, ccol = curr_img.shape[0], curr_img.shape[1]
    outimg = np.zeros((crow,ccol),np.uint8)
    stin = b_pos-((b_pos+1)/2)
    for i in range(stin,crow-stin):
    	for j in range (stin,ccol-stin):
	    outimg[i][j] = ((curr_img[i-stin:i+(b_pos-stin),j-stin:j+(b_pos-stin)]*filt).sum())/(b_pos*b_pos)
    return outimg

#Function that listens to gaussian blur ('s') tracker
def gblur_img (x):
    global gb_pos

    gb_pos = x
    if gb_pos == 0:
	gb_pos = 1
    elif gb_pos % 2 == 0:
        gb_pos = gb_pos-1

#Function that listens to averaging ('S') tracker
def blur_img (x):
    global b_pos

    b_pos = x
    if b_pos == 0:
	b_pos = 1
    elif b_pos % 2 == 0:
        b_pos = b_pos-1

#Function that listens to rotation and pixel interval ('r', 'p') tracker
def rot_pixint_val (x):
    global r_pos
    global pi_pos
    global op_flag

    if(op_flag == 'r'):
        r_pos = x
    else:
	pi_pos = x

#Function to juggle colors
def juggle_color():
    global curr_img
    
    c_atmpt1 = c_atmpt + 1
    if c_atmpt1 == 3:
        c_atmpt1 = 0
    curr_img[:,:,c_atmpt] = 0
    curr_img[:,:,c_atmpt1] = 0

#Function that shows current image
def draw_img():
    global curr_img
    cv2.imshow('BIP v1.0 :: Press ESC to EXIT, h for HELP',curr_img)

#Function that does much of the main processing
def process_img():
    global curr_img
    global gb_pos
    global b_pos
    global pi_pos

    if op_flag == 'g':
	curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        draw_img()

    if op_flag == 'G':
        crow, ccol = curr_img.shape[0], curr_img.shape[1]
	outimg = np.zeros((crow,ccol),np.uint8)
	for row in range(crow):
   	    for col in range(ccol):
      		outimg[row][col] = 0.299*curr_img[row][col][2]+0.587*curr_img[row][col][1]+0.114*curr_img[row][col][0]
        curr_img = outimg        
	draw_img()

    elif op_flag == 'i':
	draw_img()

    elif op_flag == 'c':
	juggle_color()
	draw_img()

    elif op_flag == 's':
	curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
	curr_img = cv2.GaussianBlur(curr_img,(gb_pos,gb_pos),0)
	draw_img()

    elif op_flag == 'S':
	curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
	curr_img = avg_filter()
	curr_img = cv2.convertScaleAbs(curr_img)
	draw_img()

    elif op_flag == 'r':
	curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
	row, col = curr_img.shape[0], curr_img.shape[1]
	rmat = cv2.getRotationMatrix2D((col/2,row/2),r_pos,1)
	curr_img = cv2.warpAffine(curr_img,rmat,(col,row))
	draw_img()

    elif op_flag == 'x':
	curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
	curr_img = cv2.Sobel(curr_img,cv2.CV_64F,1,0,ksize = 3)
	curr_img = cv2.convertScaleAbs(curr_img,255.0)
	draw_img()

    elif op_flag == 'X':
	curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
	curr_img = der_conv('X')
        curr_img = cv2.convertScaleAbs(curr_img,255.0)
	draw_img()

    elif op_flag == 'y':
	curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
	curr_img = cv2.Sobel(curr_img,cv2.CV_64F,0,1,ksize = 3)
        curr_img = cv2.convertScaleAbs(curr_img)
	draw_img()

    elif op_flag == 'Y':
	curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
	curr_img = der_conv('Y')
        curr_img = cv2.convertScaleAbs(curr_img)
	draw_img()

    elif op_flag == 'm':
	curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
	xd_img = cv2.Sobel(curr_img,cv2.CV_64F,0,1,ksize = 3)
	yd_img = cv2.Sobel(curr_img,cv2.CV_64F,1,0,ksize = 3)
	curr_img = np.sqrt((xd_img*xd_img)+(yd_img*yd_img))
        curr_img = cv2.convertScaleAbs(curr_img)
	draw_img()

    elif op_flag == 'M':
	curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
	xd_img = der_conv('X')
	yd_img = der_conv('Y')
	curr_img = np.sqrt((xd_img*xd_img)+(yd_img*yd_img))
        curr_img = cv2.convertScaleAbs(curr_img)
	draw_img()

    elif op_flag == 'p':
	curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
	#Can also use own function der_conv('X') below, but OpenCV function is faster, so that it would be easy to see vectors in video.
	xd_img = cv2.Sobel(curr_img,cv2.CV_64F,0,1,ksize = 3) 
	#Can also use own function der_conv('Y') below.
	yd_img = cv2.Sobel(curr_img,cv2.CV_64F,1,0,ksize = 3)
	curr_img = np.sqrt((xd_img*xd_img)+(yd_img*yd_img))
        curr_img = cv2.convertScaleAbs(curr_img)
	curr_img = plot_gradvec(xd_img,yd_img) 
	draw_img()

    elif op_flag == 'h':
	curr_img = cv2.blur(curr_img,(75,75),0)
	cv2.putText(curr_img, "Basic Image Processor v1.0", (5, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (50, 255, 50),\
            thickness=2)
	text = "This program does image manipulation using OpenCV.\nReads image from file if path is given in command line\nor else reads from camera and processes continuously.\nKeys on keyboard that work with this app follow.\n\n'i'-Reload original image. 'w'-Save current image.\n'g'-Grayscale using OpenCV. 'G'-Grayscale on own.\n'c'-Juggle color channel. 's'-Smooth using OpenCV.\n'S'-Smooth on own. 'x'-x derivative using OpenCV.\n'X'-x derivative on own. 'y'-y derivative using OpenCV.\n'Y'-y derivative on own. 'm'- |gradient| using OpenCV.\n'M'- |gradient| on own. 'p'-Discrete gradient plot.\n'r'-Grayscale and rotate. 'h'-Display help\n\nTrackbars can be used to control manipulations\nin options 's','S','p' and 'r'.\n\nPress ESC to EXIT and 'i' to startover."
	t0, dt = 40, 10
	for i, line in enumerate(text.split('\n')):
    	    t = t0 + dt
    	    cv2.putText(curr_img, line, (5, t), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness=1)
	    dt += 15
	draw_img()

    else:
	draw_img()

#Function that listens to keyboard and maintains the global op_flag
def kb_listen():

    global op_flag
    global c_atmpt 
    global b_flag

    n_op_flag = cv2.waitKey(10) & 0xFF
    
    if n_op_flag == ord('w'):
	print "--'w' pressed. Saving the current image."
	cv2.imwrite("out.jpg",curr_img)

    elif n_op_flag == ord('g'):
	print "--'g' pressed. Applying grayscale conversion using OpenCV function."
	op_flag = 'g'

    elif n_op_flag == ord('G'):
	print "--'G' pressed. Applying grayscale conversion using own function."
	op_flag = 'G'

    elif n_op_flag == ord('i'):
	print "--'i' pressed. Loading the original, unprocessed image."
	op_flag = 'i'

    elif n_op_flag == ord('c'):
	print "--'c' pressed. Color channel of the image will be juggled."
	op_flag = 'c'
	c_atmpt += 1
	if c_atmpt == 3:
	    c_atmpt = 0

    elif n_op_flag == ord('s'):
	print "--'s' pressed. Smoothing as per trackbar position using OpenCV function."
	op_flag = 's'
	cv2.createTrackbar('BLUR RADIUS\nPress s to enable!','BIP v1.0 :: Press ESC to EXIT, h for HELP',0,25,gblur_img)

    elif n_op_flag == ord('S'):
	print "--'S' pressed. Smoothing as per trackbar position using own function."
	cv2.createTrackbar('BLUR RADIUS\nPress S to enable!\nWait after moving slider!','BIP v1.0 :: Press ESC to EXIT, h for HELP',0,25,blur_img)
	op_flag = 'S'
	

    elif n_op_flag == ord('r'):
	print "--'r' pressed. Rotating the image as per trackbar position."
	op_flag = 'r'
	cv2.createTrackbar('ROTATION ANGLE\nPress r to enable!','BIP v1.0 :: Press ESC to EXIT, h for HELP',0,360,rot_pixint_val)

    elif n_op_flag == ord('x'):
	print "--'x' pressed. Displaying x-derivative of the image using OpenCV function."
	op_flag = 'x'

    elif n_op_flag == ord('X'):
	print "--'X' pressed. Displaying x-derivative of the image using own function."
	op_flag = 'X'

    elif n_op_flag == ord('y'):
	print "--'y' pressed. Displaying y-derivative of the image using OpenCV function."
	op_flag = 'y'

    elif n_op_flag == ord('Y'):
	print "--'Y' pressed. Displaying y-derivative of the image using own function."
	op_flag = 'Y'

    elif n_op_flag == ord('m'):
	print "--'m' pressed. Displaying magnitude of gradient using OpenCV function."
	op_flag = 'm'

    elif n_op_flag == ord('M'):
	print "--'M' pressed. Displaying magnitude of gradient using own function."
	op_flag = 'M'

    elif n_op_flag == ord('p'):
	print "--'p' pressed. Plotting gradient vectors spaced as per trackbar position."
	op_flag = 'p'
	cv2.createTrackbar('PIXEL INTERVAL\nPress p to enable!','BIP v1.0 :: Press ESC to EXIT, h for HELP',2,25,rot_pixint_val)

    elif n_op_flag == ord('h'):
	print "--'h' pressed. Displaying help in the main window."
	op_flag = 'h'

    elif n_op_flag == 27:
	print "INTENDED EXIT: ESC pressed. Program will terminate now!"
	op_flag = 'e'
	return

    else: 
	return
    
#Program execution starts here

print ("**Starting Basic Image Processor v1.0**");
arg_list = []
op_flag = ''
c_atmpt = -1
gb_pos = 1
b_pos = 1
r_pos = 0
pi_pos = 2

for arg in sys.argv:
    arg_list.append(arg)
if len(arg_list) == 1:
    try:
	print "Capturing and displaying image from camera..."
    	cam_input = cv2.VideoCapture(0)
    	cv2.namedWindow('BIP v1.0 :: Press ESC to EXIT, h for HELP', cv2.WINDOW_AUTOSIZE)

   	while(True):

            retvalue, orig_img = cam_input.read()
            curr_img = orig_img
    
            process_img()
            kb_listen()

            if (op_flag == 'e'):
	    	break

    	cv2.destroyAllWindows()

    except:
        print "EXITED with ERROR: Unable to read/process file from the camera!"

    else:
	cam_input.release()

elif len(arg_list) == 2:
    try:
	print "Reading and displaying image from the given path..."
	cv2.namedWindow('BIP v1.0 :: Press ESC to EXIT, h for HELP', cv2.WINDOW_AUTOSIZE) 
	while(True):
	    
	    orig_img = cv2.imread(arg_list[1],1)
	    curr_img = orig_img

	    process_img()
            kb_listen()

	    if (op_flag == 'e'):
	        break

    	cv2.destroyAllWindows()

    except:
        print "EXITED with ERROR: Unable to read/process file from the path!"

else:
    print "EXITED with ERROR: Maximum number of arguments allowed is 2!"

