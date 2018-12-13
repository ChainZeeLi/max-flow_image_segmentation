# USAGE
# python click_and_crop.py --image jurassic_park_kitchen.jpg

# import the necessary packages
import argparse
import cv2
import math
import sklearn.mixture
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False

		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		#cv2.imshow("image", image)

# construct the argument parser and parse the arguments

def collect_img_data(is_forebground):
	output=0
	if is_forebground:
		output=1
	if len(refPt) == 2:
		roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	#cv2.imshow("ROI", roi)
	#cv2.waitKey(0)
		img = cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY)
		print img
		data_file = open("train_data.txt","a") 
		for i in range(len(img)):
			for j in range(len(img[0])):
				data_file.write(str(img[i,j])+' '+str(output)+"\n")
			

	# close all open windows
	cv2.destroyAllWindows()


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# keep looping until the 'q' key is pressed
foreground_mode=True


while True:
	# display the image and wait for a keypress
	if foreground_mode:
		cv2.imshow("Select Foreground", image)
		cv2.setMouseCallback("Select Foreground", click_and_crop)
	else:
		cv2.imshow("Select Background", image)
		cv2.setMouseCallback("Select Background", click_and_crop)

	
	key = cv2.waitKey(1) & 0xFF

	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()

	# if the 'c' key is pressed, break from the loop
	elif key == ord("f"):
		#break
		collect_img_data(True)
		image = clone.copy()
		foreground_mode=False
		#cv2.setMouseCallback("image", click_and_crop)

	elif key == ord("b"):
		#key = cv2.waitKey(1) & 0xFF
		collect_img_data(False)
		image = clone.copy()
		break

	elif key == ord("q"):
		break



