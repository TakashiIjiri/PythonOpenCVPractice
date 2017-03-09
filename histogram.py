# coding: UTF-8
import numpy as np
import pylab as plt
import cv2

#画像読み込み
img  = cv2.imread("sample.png") 

#グレースケール化
img_gry = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

#histogram生成 
#cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) 
hist = cv2.calcHist( [img_gry], [0], None, [256],[0,256])

#windowを生成して画像を表示
cv2.imshow("Image", img )

#histをmatplotlibで表示
plt.plot(hist)
plt.xlim([0,256])
plt.show()


#cv2.waitKey(0) #キーボード入力待ち