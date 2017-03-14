# coding: utf-8
# 画像に 3x3フィルタを掛ける（cv2.filter2D関数を利用）

import numpy as np
import cv2

#参考URL
#http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_filtering/py_filtering.html

# 画像をnp.arrayとして読み込む
img = cv2.imread("imgs/lenaColCd.png")
img = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

filter_smooth = np.array( [[ 1, 1, 1],[ 1,1,1],[ 1,1,1]])/9.0
filter_sobelV = np.array( [[-1,-2,-1],[ 0,0,0],[ 1,2,1]])
filter_sobelH = np.array( [[-1, 0, 1],[-2,0,2],[-1,0,1]])

img_smooth = cv2.filter2D( img, -1, filter_smooth)
img_sobelV = cv2.filter2D( img, -1, filter_sobelV)
img_sobelH = cv2.filter2D( img, -1, filter_sobelH)

cv2.imshow("original  ", img        )
cv2.imshow("img_smooth", img_smooth )
cv2.imshow("img_sovelV", img_sobelV )
cv2.imshow("img_sovelH", img_sobelH )

# wait key input
cv2.waitKey(0)
