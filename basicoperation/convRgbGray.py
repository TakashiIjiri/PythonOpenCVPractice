# coding: utf-8
# カラー画像/Grayscale/各チャンネル変換
import time
import numpy as np
import cv2

img     = cv2.imread("imgs/lenaColCd.png")
#RGB --> grayscale
imgGray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

#RGB --> each channel
t1 = time.clock()
imgR = img[:,:,2]
imgG = img[:,:,1]
imgB = img[:,:,0]

#下でもOK
t2 = time.clock()
img_rgb = cv2.split(img)
imgR = img_rgb[2]
imgG = img_rgb[1]
imgB = img_rgb[0]

#下でもよいが遅い
t3 = time.clock()
imgR = np.zeros( (img.shape[0], img.shape[1]), np.uint8 )
imgG = np.zeros( (img.shape[0], img.shape[1]), np.uint8 )
imgB = np.zeros( (img.shape[0], img.shape[1]), np.uint8 )
for v in range(img.shape[0]) :
    for u in range(img.shape[1]) :
        imgR[v,u] = img[v,u,2]
        imgG[v,u] = img[v,u,1]
        imgB[v,u] = img[v,u,0]

t4 = time.clock()
print( t2-t1, t3-t2, t4-t3)

# each chaannel --> RGB
imgRGB = cv2.merge((imgB, imgG, imgR))

#表示
cv2.imshow('org', img )
cv2.imshow('gry', imgGray )
cv2.imshow('chR', imgR)
cv2.imshow('chG', imgG)
cv2.imshow('chB', imgB)
cv2.imshow('RGB', imgRGB)
cv2.waitKey(0)
