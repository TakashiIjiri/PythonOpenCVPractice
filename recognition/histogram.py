# coding: UTF-8
# 画像のヒストグラムをしてプロット

import numpy as np
import pylab as plt
import cv2

#画像読み込み & グレースケール化
#img_gry = cv2.imread("imgs/sample.png", 0) #←変換結果が汚い
img     = cv2.imread("../imgs/sample.png")
img_gry = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )


#histogram生成
hist = np.zeros(256)
for y in range(img_gry.shape[0]):
    for x in range(img_gry.shape[1]):
        hist[ img_gry[y,x] ] += 1


#windowを生成して画像を表示
cv2.imshow("Image", img_gry )

#histをmatplotlibで表示
plt.plot(hist)
plt.xlim([0,256])
plt.show()
