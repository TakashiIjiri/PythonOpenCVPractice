# coding: utf-8

import cv2
import numpy as np



img_orig  =  cv2.imread  ("imgs/book1.jpg") 
img_gray  =  cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
img_canny =  cv2.Canny   (img_gray, 50,150,apertureSize = 3)
img_vis   =  cv2.merge   ((img_gray,img_gray,img_gray))


ACCURACY_RHO   = 1
ACCURACY_THETA = np.pi/180.0
THRESH_VOTE    = 150

hough_lines = cv2.HoughLines(img_canny, ACCURACY_RHO, ACCURACY_THETA, THRESH_VOTE)

for line in hough_lines : 
    for r,t in  line : 
        #r = x cos t + y sin t
        a = np.cos(t)
        b = np.sin(t)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*( a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*( a))

        cv2.line(img_vis, (x1,y1), (x2,y2), (0,0,255),1)




cv2.imshow("c", img_canny)
cv2.imshow("a", img_vis )


cv2.waitKey(0)
