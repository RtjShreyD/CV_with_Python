import argparse
import imutils
import numpy as np
import cv2
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

# load the image and resize it to a smaller factor so that
# the shapes can be approximated better
img = cv2.imread(args["image"])
resized = imutils.resize(img, width=300)
ratio = img.shape[0] / float(resized.shape[0])
 
# convert the resized image to grayscale, blur it slightly,
# and threshold it
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 215, 255, cv2.THRESH_BINARY)[1]
thresh = abs(255-thresh)
cv2.imshow("Thresholded image",thresh)
cv2.waitKey()
 
# find contours in the thresholded image and initialize the
# shape detector
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#cnts= sorted(cnts, key = lambda x:cv2.contourArea(x), reverse = True)[:1]
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
print(cnts)

pt = (180, 3 * img.shape[0] // 4)
for cnt in cnts:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    # print len(cnt)
    print(len(approx))
    if len(approx) ==6 :
        print("Cube")
        cv2.drawContours(img,[cnt],-1,(255,0,0),3)
        cv2.putText(img,'Cube', pt ,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, [0,255, 255], 2)
    elif len(approx) == 7:
        print("Cube")
        cv2.drawContours(img,[cnt],-1,(255,0,0),3)
        cv2.putText(img,'Cube', pt ,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, [0, 255, 255], 2)
    elif len(approx) == 8:
        print("Cylinder")
        cv2.drawContours(img,[cnt],-1,(255,0,0),3)
        cv2.putText(img,'Cylinder', pt ,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, [0, 255, 255], 2)
    elif len(approx) > 10:
        print("Sphere")
        cv2.drawContours(img,[cnt],-1,(255,0,0),3)
        cv2.putText(img,'Sphere', pt ,cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, [255, 0, 0], 2)

cv2.namedWindow("Shape", cv2.WINDOW_NORMAL)
cv2.imshow('Shape',img)

corners    = cv2.goodFeaturesToTrack(thresh,6,0.06,25)
corners    = np.float32(corners)
for    item in    corners:
    x,y    = item[0]
    cv2.circle(img,(x,y),10,255,-1)
cv2.namedWindow("Corners", cv2.WINDOW_NORMAL)
cv2.imshow("Corners",img)

cv2.waitKey()

