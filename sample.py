# coding: UTF-8

import numpy as np
import cv2

a = np.array([1,1,1,1,])
print(a)
print(a.shape)

#画像読み込み
img = cv2.imread('imgs/aaa.png') 
print(img.shape)

##np.arrayを画像として保存
cv2.imwrite("aaa.png", img)

##windowを生成して画像を表示
cv2.imshow("AAAA", img )

cv2.waitKey(0) #キーボード入力待ち