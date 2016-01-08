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


##Function to normalize the image points
def norm_pts(in_array):
    ##Calculates mean and variance of points, subtracts mean and divides by variance for
    ##each point. Returns array of normalized points and the transformation matrix mp
    mean_x = float(np.mean(in_array[:,0]))
    #print "x mean is %f" % mean_x
    mean_y = float(np.mean(in_array[:,1]))
    #print "y mean is %f" % mean_y
    var_x = float(np.var(in_array[:,0]))
    #print "x variance is %f" % var_x
    var_y = float(np.var(in_array[:,1]))
    #print "y variance is %f" % var_y
    in_array[:,0] = abs(in_array[:,0]-mean_x)/var_x
    in_array[:,1] = abs(in_array[:,1]-mean_y)/var_y

    v_mat = np.zeros((3,3),np.float64)
    m_mat = np.zeros((3,3),np.float64)
    v_mat[0][0],v_mat[1][1], v_mat[2][2] = 1/var_x, 1/var_y, 1
    m_mat[0][0], m_mat[1][1], m_mat[2][2] = 1,1,1
    m_mat[0][2], m_mat[1][2] = -mean_x, -mean_y
    mp_mat = np.dot(v_mat,m_mat)
    #print mp_mat
    return in_array,mp_mat


##Function to compute and display the fundamental matrix
def fund_matrix(l_array,r_array):    
    global g_fund_mat
    print "The selected points in the left image are as below."
    print l_array	
    print "The selected points in the right image are as below."
    print r_array
    if l_array.shape[0] < 8 or r_array.shape[0] < 8:
	print "ERROR: Incorrect number of points.\nA minimum of 8 point pairs are needed for fundamental matrix calculation!"
        exit()
    nl_array,mp_prime_mat = norm_pts(l_array)
    nr_array,mp_mat = norm_pts(r_array)
    A_mat = np.ones((nl_array.shape[0],9),np.float64)
    ##From normalized left and right image points, A matrix is calculated
    for i in range(nl_array.shape[0]):
	A_mat[i][0] = nr_array[i][0]*nl_array[i][0]
	A_mat[i][1] = nr_array[i][0]*nl_array[i][1]
	A_mat[i][2] = nr_array[i][0]
	A_mat[i][3] = nr_array[i][1]*nl_array[i][0]
	A_mat[i][4] = nr_array[i][1]*nl_array[i][1]
	A_mat[i][5] = nr_array[i][1]
	A_mat[i][6] = nl_array[i][0]
	A_mat[i][7] = nl_array[i][1]
    #print A_mat

    ##After SVD of A matrix, last column of v is taken as initial fundamental matrix 
    u, d, vT = np.linalg.svd(A_mat, full_matrices=True)
    #print vT
    F = vT[8,:].reshape(3,3)
    #print F

    ##After SVD of F matrix, smallest value of d1 is made zero 
    u1, d1, vT1 = np.linalg.svd(F, full_matrices=True)
    #print d1
    s_idx = np.where(d1==d1.min())
    d1[s_idx] = 0
    #print d1

    ##F matrix is reconstructed to get F_prime which is always a rank 2 matrix
    D1 = np.zeros((u1.shape[0], vT1.shape[0]), dtype=float)
    D1[:d1.shape[0], :d1.shape[0]] = np.diag(d1)
    F_prime = np.dot(u1,np.dot(D1,vT1))
    #print F_prime

    ##From F_prime and transformation matrices of left and right images, fundamental matrix is calculated
    g_fund_mat = np.dot(np.transpose(mp_mat),np.dot(F_prime,mp_prime_mat))
    print "The estimated fundamental matrix for the given points is as below."
    print g_fund_mat
    return


#Function to calculate and display the epipoles of each image
def calc_epipoles(in_fund_mat):
    global g_l_epole
    global g_r_epole

    ##After SVD of incoming fundamental matrix, last column of v is taken as left epipole and is homogenized 
    u, d, vT = np.linalg.svd(in_fund_mat, full_matrices=True) # using svd to get weighted M
    #print vT, u
    l_epole = vT[2,:].reshape(3,1)
    l_epole[0][0], l_epole[1][0] = l_epole[0][0]/l_epole[2][0], l_epole[1][0]/l_epole[2][0]

    ##After SVD of incoming fundamental matrix, last column of u is taken as right epipole and is homogenized 
    r_epole = u[:,2].reshape(3,1)
    r_epole[0][0], r_epole[1][0] = r_epole[0][0]/r_epole[2][0], r_epole[1][0]/r_epole[2][0]
    g_l_epole, g_r_epole = l_epole[:2,:], r_epole[:2,:]
    print "The epipole of the left image is as below."
    print g_l_epole
    print "The epipole of the right image is as below."
    print g_r_epole
    return


##Function to calculate and draw the epilines corresponding to the selected point
def draw_epilines():
    global sel_point
    global g_coeffs
    global g_l_epole
    global g_r_epole
    global curr_img
    if sel_point[3] == 'l':
        ##Using co-efficients of right epipolar line, draw line in right image
    	angle = abs(np.arctan2(g_coeffs[1,0],g_coeffs[0,0]))
    	p2x =  round((sel_point[0]+(curr_img.shape[1]/2)) + 100 * np.cos(angle * 3.14 / 180.0));
    	p2y =  round(sel_point[1] + 100 * np.sin(angle * 3.14 / 180.0));
    	cv2.line(curr_img,(int(curr_img.shape[1]/2),int(sel_point[1])),(int(p2x),int(p2y)),(255,255,255),5)
    else:
        ##Using co-efficients of left epipolar line, draw line in left image
    	angle = abs(np.arctan2(g_coeffs[1,0],g_coeffs[0,0]))
    	p2x =  round((sel_point[0]) + 100 * np.cos(angle * 3.14 / 180.0));
    	p2y =  round(sel_point[1] + 100 * np.sin(angle * 3.14 / 180.0));
    	cv2.line(curr_img,(int(curr_img.shape[1]/2),int(sel_point[1])),(int(p2x),int(p2y)),(255,255,255),5)
    return


#Perform image processing based on the current operation flag
def process_img():
    
    global curr_img
    global sel_point
    global g_fund_mat

    if op_flag == 'i':
	#Loads unprocessed original image
	draw_img()

    elif op_flag == 'p':
        #Enables the mouse callback function so that all click points are captured in an array
	cv2.putText(curr_img,"Your double clicks are being recorded!", (5, 25), cv2.FONT_HERSHEY_PLAIN,1,(255, 255, 255),thickness=2)
	cv2.putText(curr_img,"FIRST click on a point in LEFT image, then click on corresponding pt in right image.",(5, 45), cv2.FONT_HERSHEY_PLAIN,1,(255, 255, 255),thickness=2)
	cv2.putText(curr_img,"Press X to exit and export clicked points. Choose atleast 8 pairs.",(5, 65),cv2.FONT_HERSHEY_PLAIN, 1,(255, 255, 255),thickness=2)
	draw_img()

    elif op_flag == 'P':
        #Enables the mouse callback function so that a single click points is captured for drawing epipolar line
	cv2.putText(curr_img,"Click ONCE anywhere on the images!", (5, 25), cv2.FONT_HERSHEY_PLAIN,1,(255, 255, 255),thickness=2)
        cv2.putText(curr_img,"After clicking, press X to exit and export clicked point.", (5, 45), cv2.FONT_HERSHEY_PLAIN,1,(255, 255, 255),thickness=2)
	draw_img()

    elif op_flag == 'L':
        ##Calls the function that is responsible for drawing epipolar line for user selected point
        draw_epilines()
        draw_img()

    elif op_flag == 'h':
	#Displays help text
	curr_img = cv2.blur(curr_img,(150,150),0)
	cv2.putText(curr_img, "Epipolar Line Estimator v1.0", (5, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0),\
            thickness=2)
	text = "This program calculates fundamental matrix and displays epipolar lines.\nReads images from file paths that are given in command line.\nKeys on keyboard that work with this app follow.\n\n'i'-Reload original image.\n'w'-Save current image.\n'p'-Gather multiple points using mouse to use for fundamental matrix calculation.\n'X'-Exit point gathering session and export clicked point(s).\n'F'-Calculate and display fundamental matrix using the user specified points.\n'E'-Calculate and display epipoles for left and right images.\n'P'-Gather a single point in left or right image for drawing epipolar line.\n'L'-Draw epipolar line corresponding to the selected single point.\n'h'-Display help.\n\nSuggested Flow: 'p'->Click on points(more than 8)->'X'->'F'->'E'->'P'->Click on single point->'X'->'L'->'i'\n\nPress ESC to EXIT and 'i' to startover."
	t0, dt = 40, 10
	for i, line in enumerate(text.split('\n')):
    	    t = t0 + dt
    	    cv2.putText(curr_img, line, (5, t), cv2.FONT_HERSHEY_PLAIN, 1.0, (255), thickness=1)
	    dt += 15
	draw_img()

    else:
	draw_img()


#Function that listens to keyboard and maintains the global op_flag
def kb_listen():

    global op_flag
    global l_pts_list
    global r_pts_list
    global l_array
    global r_array
    global g_fund_mat
    global sel_point
    global g_coeffs

    n_op_flag = cv2.waitKey(10) & 0xFF
    
    if n_op_flag == ord('w'):
	print "--'w' pressed. Saving the current image."
	cv2.imwrite("out.jpg",curr_img)

    elif n_op_flag == ord('i'):
	print "--'i' pressed. Loading the original, unprocessed image."
	op_flag = 'i'

    elif n_op_flag == ord('X'):
	print "--'X' pressed. Exiting point gathering session and exporting clicked point(s)."
	l_array = np.reshape(np.asarray(l_pts_list),(-1,2))
	r_array = np.reshape(np.asarray(r_pts_list),(-1,2))
        sel_point = sel_point
	op_flag = 'X'

    elif n_op_flag == ord('F'):
	print "--'F' pressed. Calculating fundamental matrix using the user specified points."
	op_flag = 'F'
	fund_matrix(l_array,r_array)

    elif n_op_flag == ord('E'):
	print "--'E' pressed. Calculating epipoles for left and right images."
	op_flag = 'E'
	calc_epipoles(copy.copy(g_fund_mat))

    elif n_op_flag == ord('L'):
	print "--'L' pressed. Epipolar line corresponding to the selected point will be drawn."
        if sel_point[3] == 'l':
	    print "The selected point in the left image is as below."
            print np.asarray(sel_point[:2])
	    repl_coeff = np.dot(g_fund_mat,(np.asarray(sel_point[:3])).reshape(3,1))
            print "The co-efficient of right epipolar line for the selected left point is as below."
            print repl_coeff
            g_coeffs = repl_coeff
        else:
	    print "The selected point in the right image is as below."
            print np.asarray(sel_point[:2])
	    lepl_coeff = np.dot(np.transpose(g_fund_mat),(np.asarray(sel_point[:3])).reshape(3,1))
            print "The co-efficient of left epipolar line for the selected right point is as below."
            print lepl_coeff
            g_coeffs = lepl_coeff
	op_flag = 'L'

    elif n_op_flag == ord('p'):
	print "--'p' pressed. Entering point gathering session (multiple points)."
	op_flag = 'p'
    
    elif n_op_flag == ord('P'):
	print "--'P' pressed. Entering point gathering session (single point)."
	op_flag = 'P'

    elif n_op_flag == ord('h'):
	print "--'h' pressed. Displaying help."
	op_flag = 'h'

    elif n_op_flag == 27:
	print "INTENDED EXIT: ESC pressed. Program will terminate now!"
	op_flag = 'e'
	return

    else: 
	return


#Program execution starts here
print ("**Starting Epipolar Line Estimator v1.0**");
arg_list = []
l_pts_list, r_pts_list = [], []
sel_point = [1,1,1,'']
g_l_epole, g_r_epole = np.zeros((2,2),np.float64),np.zeros((2,2),np.float64)
op_flag = ''
trial = 0
cv2.namedWindow('ELE v1.0 :: Press ESC to EXIT, h for HELP', cv2.WINDOW_AUTOSIZE) 
cv2.setMouseCallback('ELE v1.0 :: Press ESC to EXIT, h for HELP', gather_pts)
l_array = np.ones((8,2),np.float64)
r_array = np.ones((8,2),np.float64)
g_fund_mat = np.ones((3,3),np.float64)
g_coeffs = np.ones((3,1),np.float64)

#Gets all arguments, if filenames are specified, reads those particular images for processing
#Also handles exceptions in filenames and number of arguments
for arg in sys.argv:
    arg_list.append(arg)
if len(arg_list) == 3:

    try:
	print "Reading and displaying image from the given path..."
        while(True):
            first_img = cv2.imread(arg_list[1],1)
            first_gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
            second_img = cv2.imread(arg_list[2],1)
            second_gray = cv2.cvtColor(second_img, cv2.COLOR_BGR2GRAY)
            orig_img = np.concatenate((first_gray, second_gray), axis=1)
            curr_img = orig_img

            process_img()
            kb_listen()
            if (op_flag == 'e'):
	        break

    	cv2.destroyAllWindows()

    except:
        print "EXITED with ERROR: Unable to read/process file from the path!"

else:
    print "EXITED with ERROR: Incorrect number of arguments. Allowed is 2 (filenames)!"
