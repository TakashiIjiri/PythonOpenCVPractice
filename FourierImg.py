# coding: UTF-8
import cv2
import numpy as np
import pylab as plt
import itertools


SCALE = 3

img       = cv2.cvtColor( cv2.imread("lena.png"), cv2.COLOR_BGR2GRAY )
img_dft   = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
img_dft   = np.fft.fftshift(img_dft)

#gen masks
H  = img.shape[0]
W  = img.shape[1]
cH = int(H/2) 
cW = int(W/2) 

lowPass_mask  = np.zeros( (H,W,2) ) 
highPass_mask = np.zeros( (H,W,2) ) 
bandPass_mask = np.zeros( (H,W,2) ) 
LOW  = 20
HIGH = 40

for y, x in itertools.product(range(H),range(W)):
    d = np.sqrt( (y-cH)*(y-cH) + (x-cW)*(x-cW) )
    
    lowPass_mask [y,x,:] = 1.0 if d < LOW              else 0.
    highPass_mask[y,x,:] = 1.0 if HIGH < d             else 0.
    bandPass_mask[y,x,:] = 1.0 if LOW < d and d < HIGH else 0.

img_dft_power = np.zeros((H,W))
for y, x in itertools.product(range(H),range(W)):
    img_dft_power[y,x] = np.linalg.norm(img_dft[y,x,:])

#直流成分は1に
lowPass_mask [cH,cW,:]=1.0
highPass_mask[cH,cW,:]=1.0
bandPass_mask[cH,cW,:]=1.0

#mult mask
img_dft_low  = lowPass_mask  * img_dft 
img_dft_high = highPass_mask * img_dft 
img_dft_band = bandPass_mask * img_dft 

#inverse dft
img_dft_low_idft   = cv2.idft(np.fft.ifftshift(img_dft_low ), flags=cv2.DFT_SCALE )
img_dft_high_idft  = cv2.idft(np.fft.ifftshift(img_dft_high), flags=cv2.DFT_SCALE )
img_dft_band_idft  = cv2.idft(np.fft.ifftshift(img_dft_band), flags=cv2.DFT_SCALE )

cv2.imshow("img", img)
cv2.imshow("imgDft", img_dft_power*0.00005)
cv2.imshow("low pass" , np.uint8(img_dft_low_idft [:,:,0]))
cv2.imshow("high pass", np.uint8(img_dft_high_idft[:,:,0]))
cv2.imshow("band pass", np.uint8(img_dft_band_idft[:,:,0]))

cv2.imshow("lowPass_mask" , np.uint8(255.0*lowPass_mask[:,:,0]))
cv2.imshow("highPass_mask", np.uint8(255.0*highPass_mask[:,:,0]))
cv2.imshow("bandPass_mask", np.uint8(255.0*bandPass_mask[:,:,0]))



#cv2.imwrite("0.png", np.uint8(img_dft_low_idft [:,:,0]))
#cv2.imwrite("1.png", img_dft_power*0.01)
#cv2.imwrite("2.png", np.uint8(img_dft_high_idft[:,:,0]))
#cv2.imwrite("3.png", np.uint8(img_dft_band_idft[:,:,0]))

#cv2.imwrite("4.png" , np.uint8(255.0*lowPass_mask[:,:,0]))
#cv2.imwrite("5.png", np.uint8(255.0*highPass_mask[:,:,0]))
#cv2.imwrite("6.png", np.uint8(255.0*bandPass_mask[:,:,0]))

cv2.waitKey(0)

drawing   = False


