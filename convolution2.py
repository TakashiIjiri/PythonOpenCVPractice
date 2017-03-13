# coding: utf-8
# 画像に 3x3フィルタを掛ける（自作関数を利用)


import numpy as np
import cv2
import itertools


#自作convolution (OpenCVの関数を利用しない)
def myConvolve(srcImg, filter) : 
    H = srcImg.shape[0]
    W = srcImg.shape[1]
    R = int(filter.shape[0] / 2)
    trgtImg = np.zeros(srcImg.shape)
        
    for v, u in itertools.product(range(1,H-1),range(1,W-1)):
        pix = 0.
        for vv, uu in itertools.product(range(-R,R+1),range(-R,R+1)) : 
            pix += filter[R + vv][R + uu] * srcImg[v+vv][u+uu]
        trgtImg[v][u] = min(255,max(0,abs(pix)))
       
    return np.uint8(trgtImg)

#参考URL
#http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_filtering/py_filtering.html

#画像をnp.arrayとして読み込む
img = cv2.imread("imgs/lenaColCd.png")
img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

filter_smooth = np.array( [[ 1,  1,  1 ],[ 1.,1.,1.],[ 1.,1.,1.]])/9.0
filter_sobelV = np.array( [[-1.,-2.,-1.],[ 0.,0.,0.],[ 1.,2.,1.]]) 
filter_sobelH = np.array( [[-1., 0., 1.],[-2.,0.,2.],[-1.,0.,1.]]) 

img_smooth = myConvolve(img, filter_smooth )
img_sobelV = myConvolve(img, filter_sobelV )
img_sobelH = myConvolve(img, filter_sobelH )

cv2.imshow("original  ", img        )
cv2.imshow("img_smooth", img_smooth )
cv2.imshow("img_sovelV", img_sobelV )
cv2.imshow("img_sovelH", img_sobelH )

# wait key input
cv2.waitKey(0)