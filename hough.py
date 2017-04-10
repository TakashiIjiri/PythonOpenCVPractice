# coding: utf-8

import cv2
import numpy as np

img_gray  =  cv2.imread  ("imgs/book4.jpg",0) 
img_canny =  cv2.Canny   (img_gray, 75,150,apertureSize = 3)
img_vis   =  cv2.merge   ((img_gray,img_gray,img_gray))



ACCURACY_RHO   = 1
ACCURACY_THETA = np.pi/180.0
THRESH_VOTE    = 120
hough_lines = cv2.HoughLines(img_canny, ACCURACY_RHO, ACCURACY_THETA, THRESH_VOTE)

#ëÂÇ´Ç»â~åüèo
CANNY_PARAM    = 150
THRESH_VOTE    = 500
hough_circles1 = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=4, minDist=5, 
                                param1=CANNY_PARAM, param2=THRESH_VOTE, minRadius=70, maxRadius=200 )

#è¨Ç≥Ç»â~åüèo
CANNY_PARAM    = 150
THRESH_VOTE    = 140
hough_circles2 = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, dp=4, minDist=5, 
                                param1=CANNY_PARAM, param2=THRESH_VOTE, minRadius=20, maxRadius=30 )


for line in hough_lines : 
    for rho, theta in  line : 
        cosT = np.cos(theta)
        sinT = np.sin(theta)
        x1 = int( rho*cosT + 1000*(-sinT))
        y1 = int( rho*sinT + 1000*( cosT))
        x2 = int( rho*cosT - 1000*(-sinT))
        y2 = int( rho*sinT - 1000*( cosT))
        cv2.line(img_vis, (x1,y1), (x2,y2), (0,0,255),1)

if( hough_circles1 is not None ) : 
    for c in hough_circles1 : 
        for x,y,r in c:
            cv2.circle(img_vis, (x,y),r, (0,255,0), 2)

if( hough_circles2 is not None ) : 
    for c in hough_circles2 : 
        for x,y,r in c:
            cv2.circle(img_vis, (x,y),r, (255,0,0), 2)

cv2.imshow("c", img_canny)
cv2.imshow("a", img_vis )


cv2.waitKey(0)
