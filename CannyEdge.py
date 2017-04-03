# coding: UTF-8

# 参考URL
# http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html

import cv2
import numpy as np

#グレースケールで画像を読み込み canny edge detectorにかける
img       = cv2.imread('imgs/boxBallHeart.jpg', 0)
img_edges = cv2.Canny(img,100,200)

cv2.imshow('original  ',img)
cv2.imshow('canny edge',img_edges)
if cv2.waitKey(0) :
    cv2.destroyAllWindows()
