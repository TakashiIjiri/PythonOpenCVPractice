# coding: UTF-8

import numpy as np
import cv2

#画像読み込み
img = cv2.imread("sample.png") 
print(img.shape)


#np.arrayを画像として保存
cv2.imwrite("aaa.png", img)

#windowを生成して画像を表示
cv2.imshow("AAAA", img )

cv2.waitKey(0) #キーボード入力待ち