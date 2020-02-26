# coding: utf-8
# カラー画像/Grayscale/各チャンネル変換
import time
import numpy as np
import cv2

#load image
img = cv2.imread("../imgs/lenaColCd.png")
print("height : " , img.shape[0])
print("width  : " , img.shape[1])
print("channel: " , img.shape[2])

#RGB --> grayscale
img_gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

#rgb channels
img_r = img[:,:,2]
img_g = img[:,:,1]
img_b = img[:,:,0]

#下でもOK
#img_rgb = cv2.split(img)
#img_r = img_rgb[2]
#img_g = img_rgb[1]
#img_b = img_rgb[0]

#show images
cv2.imshow('org', img )
cv2.imshow('gry', img_gray )
cv2.imshow('chR', img_r)
cv2.imshow('chG', img_g)
cv2.imshow('chB', img_b)

cv2.waitKey(0)
