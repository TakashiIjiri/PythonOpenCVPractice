# coding: utf-8

import numpy as np
import cv2


# 画像をnp.arrayとして読み込む
img = cv2.imread('lenaColCd.png')

# np.uing8型のnp.arrayを作成する
imgR = np.zeros( (img.shape[0],img.shape[1]), np.uint8 )
imgG = np.zeros( (img.shape[0],img.shape[1]), np.uint8 )
imgB = np.zeros( (img.shape[0],img.shape[1]), np.uint8 )

#RGB画像を抽出
for v in range(img.shape[0]) :
    for u in range(img.shape[1]) :
        imgR[v,u] = img[v,u,2]
        imgG[v,u] = img[v,u,1]
        imgB[v,u] = img[v,u,0]

#RGB各画像を書き出す
cv2.imwrite("R.png",imgR)
cv2.imwrite("G.png",imgG)
cv2.imwrite("B.png",imgB)

#RGB各画像を表示する 
cv2.imshow('original', img )
cv2.imshow('R', imgR)
cv2.imshow('G', imgG)
cv2.imshow('B', imgB)

# wait key input
cv2.waitKey(0)