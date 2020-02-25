# coding: utf-8
# 画像に 3x3フィルタを掛ける（自作関数を利用)
#自作convolution (OpenCVの関数を利用しないので遅い)

import numpy as np
import cv2
import itertools


def MyConvolve(srcImg, filter) :
    H = srcImg.shape[0]
    W = srcImg.shape[1]
    R = int(filter.shape[0] / 2)
    trgtImg = np.zeros(srcImg.shape)

    for v, u in itertools.product(range(R,H-R),range(R,W-R)):
        pix = 0.

        for vv, uu in itertools.product(range(-R,R+1),range(-R,R+1)) :
            pix += filter[R + vv][R + uu] * srcImg[v+vv][u+uu]
        trgtImg[v][u] = min(255,max(0,abs(pix)))

    return np.uint8(trgtImg)

#グレースケール画像の読み込み
img = cv2.imread  ("../imgs/lena.png")
img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

filter_smooth = np.array( [[ 1,  1,  1 ],[ 1.,1.,1.],[ 1.,1.,1.]])/9.0
filter_sobelV = np.array( [[-1.,-2.,-1.],[ 0.,0.,0.],[ 1.,2.,1.]])
filter_sobelH = np.array( [[-1., 0., 1.],[-2.,0.,2.],[-1.,0.,1.]])

img_smooth = MyConvolve(img, filter_smooth )
img_sobelV = MyConvolve(img, filter_sobelV )
img_sobelH = MyConvolve(img, filter_sobelH )

cv2.imshow("original  ", img        )
cv2.imshow("img_smooth", img_smooth )
cv2.imshow("img_sovelV", img_sobelV )
cv2.imshow("img_sovelH", img_sobelH )
cv2.waitKey(0)
