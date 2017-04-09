# coding: utf-8

import cv2
import scipy as sp
import numpy as np
import random




def visKeyPoints(img1, points1) :
    tmp = np.zeros(img1.shape)
    tmp = img1
    vis = cv2.merge((tmp, tmp, tmp))

    for p in points1:
        if( p.size < 5 ) : continue
        #temp = (p.pt, p.size, p.angle, p.response, p.octave, p.class_id)
        color = (random.uniform(1,255), random.uniform(1,255), random.uniform(1,255))
        cv2.circle(vis, (int(p.pt[0]),int(p.pt[1])), int(p.size), color, 2)
    return vis



def visMatching(img1, img2, points1, points2) :
    H1, W1 = img1.shape[:2]
    H2, W2 = img2.shape[:2]

    print(img1.shape)
    print(img2.shape)

    gray = sp.zeros( (max(H1, H2), (W1 + W2 + 30)), sp.uint8)
    gray[ :H1,   :W1] = img1
    gray[ :H2, W1+ 30:] = img2
    visImg = cv2.merge((gray, gray, gray))

    for i in range(len(points1)):
        if( i % 2 != 0) : continue
        color = (random.uniform(1,255), random.uniform(1,255), random.uniform(1,255))
        p1 = ( int(points1[i][0]     ), int(points1[i][1] ) )
        p2 = ( int(points2[i][0] + W1+30), int(points2[i][1] ) )
        cv2.line(visImg, p1, p2, color)

    return visImg





img1 =  cv2.imread("imgs/book1.jpg", 0) 
img2 =  cv2.imread("imgs/book3.jpg", 0) 

#compute SIFT
sift = cv2.xfeatures2d.SIFT_create()
key1, des1 = sift.detectAndCompute (img1, None      )
key2, des2 = sift.detectAndCompute (img2, None      )

#matching and triming
matches       = cv2.BFMatcher( cv2.NORM_L2 ).match(des1, des2)
threshold     = np.array([m.distance for m in matches]).mean() * 1.2
matches_trim  = [m for m in matches if m.distance < threshold]


point1  = np.array( [ np.array( key1[m.queryIdx].pt) for m in matches_trim  ])
point2  = np.array( [ np.array( key2[m.trainIdx].pt) for m in matches_trim  ])
imgMatch= visMatching(img1, img2, point1, point2)

imgVis1 = visKeyPoints( img1, key1 )
imgVis2 = visKeyPoints( img2, key2 )


# wait key input
cv2.imshow("key points1",imgVis1  )
cv2.imshow("key points2",imgVis2  )
cv2.imshow("key match"  ,imgMatch )
cv2.waitKey(0)
