# coding: utf-8
# 画像にテンプレートマッチングを掛ける

import numpy as np
import cv2

def visKeyPoints(img, peakPosi, peakNega) :
    vis = cv2.merge((img, img, img))
    for p in peakPosi:
        cv2.circle(vis, (int(p[1]),int(p[2])),p[0]*10, (255,0,0), 1)
    for p in peakNega:
        cv2.circle(vis, (int( p[1]),int(p[2])),p[0]*10, (0,255,0), 1)
    return vis


def isPeakPosi( imgs, t,y,x ) :
    c = imgs[t][y,x] 
    if( c <= 20 ) :
        return False

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
    if( c >= -20 ) :
        return False
 
    tf = True
 
    for tt in range(0,1) : 
        for xx in range(-1,2) : 
            for yy in range(-1,2) : 
                if( tt ==0 and xx == 0 and yy==0 ) : continue
                if( c >= imgs[t+tt][y+yy,x+xx] ) :                
                    tf = False
    return tf



# 画像をnp.arrayとして読み込む
k = 2
s = 1.0
N = 6


imgs     = []
imgs_DoG = []

imgs.append( cv2.imread("imgs/pano1.jpg", 0) )

#compute DoG
for i in range(1,N) : 
    imgs.append( cv2.GaussianBlur(imgs[0], (0,0), s * (k**i) ) )
    imgs_DoG.append( np.int32(imgs[i]) - np.int32(imgs[i-1] ) )


peakPtPosi = []
peakPtNega = []
for y in range(1,imgs[0].shape[0]-1) : 
    for x in range(1,imgs[0].shape[1]-1) :
        if( isPeakPosi(imgs_DoG,1,y,x ) ) : 
            peakPtPosi.append( (1,y,x)) 
        if( isPeakNega(imgs_DoG,1,y,x ) ) : 
            peakPtNega.append( (1,y,x)) 

vis= visKeyPoints( np.uint8(imgs_DoG[1] / 2 + 128), peakPtPosi, peakPtNega)
cv2.imshow("vis", vis)
cv2.waitKey(0)



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

#cv2.imshow("lv0", imgs[0])
#cv2.imshow("lv1", imgs[1])
#cv2.imshow("lv2", imgs[2])
#cv2.imshow("lv3", imgs[3])
#cv2.imshow("lv4", imgs[4])

print( type( imgs_DoG[0] ), type( imgs_DoG[0][0,0] ) ) 
cv2.imshow("dlv0", np.uint8(imgs_DoG[0]/2+128))
cv2.imshow("dlv1", np.uint8(imgs_DoG[1]/2+128))
cv2.imshow("dlv2", np.uint8(imgs_DoG[2]/2+128))
cv2.imshow("dlv3", np.uint8(imgs_DoG[3]/2+128))


vis= visKeyPoints(imgs[0], peakPtPosi, peakPtNega)
cv2.imshow("vis", vis)


# wait key input
cv2.waitKey(0)
