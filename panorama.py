# coding: UTF-8
#複数枚の画像を読み込みパノラマ合成を行う


# 参考にしたweb page
# http://qiita.com/bohemian916/items/4d3cf6506ec7d8f628f3
# http://authorunknown408.blog.fc2.com/blog-entry-38.html

#findHomography
#http://opencv.jp/opencv-2svn/cpp/camera_calibration_and_3d_reconstruction.html#cv-findhomography

import cv2
import scipy as sp
import numpy as np
import random
  

VIS_FOR_DEBUG = True

def visMatching(img1, img2, points1, points2) :
    H1, W1 = img1.shape[:2]
    H2, W2 = img2.shape[:2]

    print(img1.shape)
    print(img2.shape)

    gray = sp.zeros( (max(H1, H2), (W1 + W2)), sp.uint8)
    gray[ :H1,   :W1] = img1
    gray[ :H2, W1:  ] = img2
    visImg = cv2.merge((gray, gray, gray))

    for i in range(len(points1)):

        if( i % 2 != 0) : 
            continue

        color = (random.uniform(1,255), random.uniform(1,255), random.uniform(1,255))
        p1 = ( int(points1[i][0]     ), int(points1[i][1] ) )
        p2 = ( int(points2[i][0] + W1), int(points2[i][1] ) )
        cv2.line(visImg, p1, p2, color)

    return visImg
 

# Mat2 : Homography mat for warping img2 
def margeImages( img1, img2, Mat2) : 
     
    TransMat = np.identity(3)
    if( Mat2[0,2] < 0 ): TransMat[0,2] = -Mat2[0,2]
    if( Mat2[1,2] < 0 ): TransMat[1,2] = -Mat2[1,2]

    Mat2 = TransMat.dot( Mat2 ) 

    H1, W1 = img1.shape[:2]
    H2, W2 = img2.shape[:2]
    W = max( W1,W2 ) + int( max(Mat2[0,2],TransMat[0,2]) )
    H = max( H1,H2 ) + int( max(Mat2[1,2],TransMat[1,2]) )

    warp1 = cv2.warpPerspective(img1,TransMat,(W,H))
    warp2 = cv2.warpPerspective(img2,Mat2    ,(W,H))
    output= np.array(warp1)
 
    # warp1 <-- warp2
    for i in range(W):
        for j in range(H):
            if  ( warp1[j,i,0] and warp2[j,i,0] ) : output[j,i] = (warp1[j,i]/ 2 + warp2[j,i]/ 2) 
            elif( warp1[j,i,0]                  ) : output[j,i] =  warp1[j,i]
            elif( warp2[j,i,0]                  ) : output[j,i] =  warp2[j,i]
    
    return warp1, warp2, output


def calc_macheing_mat(grayImg1, grayImg2) : 

    #key point detection keyPt:点配列, des:特徴ベクトル配列
    detector = cv2.AKAZE_create()
    keyPt1, des1 = detector.detectAndCompute(grayImg1, None)
    keyPt2, des2 = detector.detectAndCompute(grayImg2, None)

    #matching key points --> trimming
    matches       = cv2.BFMatcher( cv2.NORM_HAMMING ).match(des1, des2)
    threshold     = np.array([m.distance for m in matches]).mean() * 0.8
    matches_trim  = [m for m in matches if m.distance < threshold]

    #compute 変換行列
    point1 = np.array( [ np.array(keyPt1[m.queryIdx].pt) for m in matches_trim  ])
    point2 = np.array( [ np.array(keyPt2[m.trainIdx].pt) for m in matches_trim  ])
    
    if( VIS_FOR_DEBUG ) :
        visImg = visMatching(grayImg1, grayImg2, point1, point2)
        cv2.imshow("matching", visImg )

    H, Hstatus = cv2.findHomography(point2,point1,cv2.RANSAC)
    return H


def panorama(img1, img2) : 
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    Mat = calc_macheing_mat( gray1, gray2)
    warp1, warp2, res = margeImages(img1, img2, Mat)


    if( VIS_FOR_DEBUG ) :
        cv2.imshow("1", warp1) 
        cv2.imshow("2", warp2) 
        cv2.imshow("3", res  ) 
        cv2.waitKey(0)

    return res



#main
I1 = cv2.imread("imgs/pano1.JPG")
I2 = cv2.imread("imgs/pano2.JPG")
I3 = cv2.imread("imgs/pano3.JPG")
I4 = cv2.imread("imgs/pano4.JPG")

res = panorama( I1, I2)
res = panorama(res, I3)
res = panorama(res, I4)
cv2.imwrite("panoramaRes.png", res)
cv2.imshow("result", res) 
cv2.waitKey(0)




