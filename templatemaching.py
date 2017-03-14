# coding: utf-8
# 画像にテンプレートマッチングを掛ける

import numpy as np
import cv2


def normalize255( img ) :
    maxV = 0
    for y,x in np.ndindex(img.shape[0]) : maxV = max(maxV,img[y][x])
     

def calcDefSqImg(trgt, temp) : 
    res = np.zeros( (trgt.shape[0],trgt.shape[1]) )

    H, W   = trgt.shape[0], trgt.shape[1]
    tH, tW = temp.shape[0], temp.shape[1]
    maxV = 0
    for y,x in np.ndindex((H-tH, W-tW)) : 
        diff = np.int32(trgt[y:y+tH, x:x+tW,:]) - np.int32(temp)
        d    = np.linalg.norm(diff)
        maxV = max( maxV, d)
        res[y + int(tH/2),x + int(tW/2)] = d
        #cv2.imshow("aa", diff)
        #cv2.waitKey(0)        
        
        if( d < 100 ) :
            print(x,y)

    print( maxV )
    return np.uint8( res * (255.0 / maxV) )
        #下の実装はおそすぎる
        #d=0
        #for yy,xx in np.ndindex((tH, tW) ) :
        #    d += np.linalg.norm((trgt[y+yy,x+xx,:] - temp[yy,xx,:]))
        #res[y + int(tH/2),x + int(tW/2)] = d
        



# 画像をnp.arrayとして読み込む
trgt = cv2.imread("imgs/templatematching.jpg")
temp = cv2.imread("imgs/template.jpg")


res1 = calcDefSqImg(trgt,temp)
cv2.imshow("trgt", trgt)
cv2.imshow("temp", temp)
cv2.imshow("calcDefSqImg  ", res1)

# wait key input
cv2.waitKey(0)