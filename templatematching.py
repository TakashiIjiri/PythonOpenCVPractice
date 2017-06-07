# coding: utf-8
# 画像にテンプレートマッチングを掛ける

import numpy as np
import cv2


def normalize255( img ) :
    maxV = 0
    for y,x in np.ndindex(img.shape[0]) : maxV = max(maxV,img[y][x])


def diffSSD(trgt, temp) :
    H,   W = trgt.shape[0], trgt.shape[1]
    tH, tW = temp.shape[0], temp.shape[1]
    res = np.zeros( (H,W) )
    maxV = 0
    for y,x in np.ndindex((H-tH, W-tW)) :
        diff = np.int32(trgt[y:y+tH, x:x+tW]) - np.int32(temp)
        d    = np.sum(diff**2)
        maxV = max( maxV, d)
        res[y + int(tH/2),x + int(tW/2)] = d
    return np.uint8( res * (255.0 / maxV) )

def diffSAD(trgt, temp) :
    H,   W = trgt.shape[0], trgt.shape[1]
    tH, tW = temp.shape[0], temp.shape[1]
    res = np.zeros( (H,W) )
    maxV = 0
    for y,x in np.ndindex((H-tH, W-tW)) :
        diff = np.abs( np.int32(trgt[y:y+tH, x:x+tW]) - np.int32(temp))
        d    = np.sum( diff )
        maxV = max( maxV, d)
        res[y + int(tH/2),x + int(tW/2)] = d
    return np.uint8( res * (255.0 / maxV) )

def diffNCC(trgt, temp) :
    H,   W = trgt.shape[0], trgt.shape[1]
    tH, tW = temp.shape[0], temp.shape[1]
    res = np.zeros( (H,W) )
    Tsum = np.linalg.norm(temp)
    for y,x in np.ndindex((H-tH, W-tW)) :
        I    = np.int32(trgt[y:y+tH, x:x+tW])
        Isum = np.linalg.norm(I)
        res[y + int(tH/2),x + int(tW/2)] = np.sum( I * np.int32(temp) ) / Tsum / Isum
    return np.uint8( res * 255.0 )

# 画像をnp.arrayとして読み込む
trgt = cv2.imread("imgs/templatematching.jpg")
temp = cv2.imread("imgs/template.jpg")
#trgt = cv2.imread("imgs/templatematching1.jpg")
#temp = cv2.imread("imgs/template1.jpg")

trgtGry = cv2.cvtColor( trgt, cv2.COLOR_BGR2GRAY )
tempGry = cv2.cvtColor( temp, cv2.COLOR_BGR2GRAY )
res1 = diffSSD(trgtGry, tempGry)
res2 = diffSAD(trgtGry, tempGry)
res3 = diffNCC(trgtGry, tempGry)
cv2.imshow("trgt", trgt)
cv2.imshow("temp", temp)
cv2.imshow("diffSSD  ", res1)
cv2.imshow("diffSAD  ", res2)
cv2.imshow("diffNCC  ", res3)
cv2.waitKey(0)
