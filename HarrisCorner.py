# coding: UTF-8

# 参考URL
# http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html

import cv2
import numpy as np

def visKeyPoints( baseImg, cornerImg, THRESH) :
    vis = baseImg
    for y in range(cornerImg.shape[0]) :
        for x in range(cornerImg.shape[1]) :
            if( cornerImg[y,x] > THRESH ) :
                cv2.circle(vis, (x,y), 3, (0,255,0), 1)
    return vis


img      = cv2.imread('imgs/boxBallHeart.jpg')
img_gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

#cornerHarris( block size, sobel kernel size, k-value)
img_corners = cv2.cornerHarris(np.float32(img_gray), 10, 7, 0.04)
thresh = img_corners.max() * 0.05

#単純に閾値以上の画素を塗る実装
#img[ img_corners > thresh ] = [0,255,0]

#丸を描く実装
img_vis = cv2.merge((img_gray,img_gray,img_gray))
img_vis = visKeyPoints(img_vis, img_corners, thresh)

cv2.imshow('cornerHarris',img_vis  )
cv2.waitKey(0)
