# coding: utf-8
# 画像にテンプレートマッチングを掛ける

import numpy as np
import cv2

THRESH_POSI =  10
THRESH_NEGA = -20
k = 2
s = 2.0
N = 5




def visKeyPoints(img, peakPosi, peakNega) :
    vis = cv2.merge((img, img, img))
    for p in peakPosi:
        cv2.circle(vis, (int(p[2]), int(p[1])),p[0]*10, (255,0,0), 1)
    for p in peakNega:
        cv2.circle(vis, (int(p[2]), int(p[1])),p[0]*10, (0,255,0), 1)
    return vis

def isPeakPosi( imgs, t,y,x ) :
    c = imgs[t][y,x]
    if( c <= THRESH_POSI ) : return False
    tf = True
    for tt in range(0,1) :
        for xx in range(-1,2) :
            for yy in range(-1,2) :
                if( tt ==0 and xx == 0 and yy==0 ) : continue
                if( c <= imgs[t+tt][y+yy,x+xx] ) :
                    tf = False
    return tf

def isPeakNega( imgs, t,y,x ) :
    c = imgs[t][y,x]
    if( c >= THRESH_NEGA ) : return False
    tf = True
    for tt in range(0,1) :
        for xx in range(-1,2) :
            for yy in range(-1,2) :
                if( tt ==0 and xx == 0 and yy==0 ) : continue
                if( c >= imgs[t+tt][y+yy,x+xx] ) :
                    tf = False
    return tf





img_orig = cv2.imread("imgs/pano1.jpg", 0)
imgs     = []
imgs_DoG = []

#compute DoG
for i in range(0,N) :
    imgs.append( cv2.GaussianBlur(img_orig, (0,0), s * (k**i) ) )
    if( i > 0) :
        imgs_DoG.append( np.int32(imgs[i]) - np.int32(imgs[i-1] ) )

#search peak
peakPtPosi = []
peakPtNega = []
for t in range(1,N-2) :
    for y in range(1,imgs[0].shape[0]-1) :
        for x in range(1,imgs[0].shape[1]-1) :
            if( isPeakPosi(imgs_DoG,t,y,x ) ) :
                peakPtPosi.append( (t,y,x))
            if( isPeakNega(imgs_DoG,t,y,x ) ) :
                peakPtNega.append( (t,y,x))

for t in range(N) :
    fname = "smooth" + str(t) + ".png"
    cv2.imwrite(fname, imgs[t])

for t in range(N-1) :
    fname = "Dog" + str(t) + ".png"
    cv2.imwrite(fname, imgs_DoG[t] * 3 / 2 + 128)

vis= visKeyPoints(img_orig, peakPtPosi, peakPtNega)
cv2.imshow("vis", vis)


# wait key input
cv2.waitKey(0)
