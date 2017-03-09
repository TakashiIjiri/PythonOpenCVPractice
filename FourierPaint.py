# coding: UTF-8
import cv2
import numpy as np
import pylab as plt



def update_mask(x,y) : 
    global img_dft,img_idft,img_mask,img_viz, img_wave, drawing
    img_mask[y-5:y+5,x-5:x+5,:] = 1.
    img_viz  = img_dft * img_mask 
    img_idft = cv2.idft(np.fft.ifftshift(img_viz), flags=cv2.DFT_SCALE )

    tmp_mask  = np.zeros( img_dft.shape ) 
    tmp_mask[y,x,:] = 1
    img_wave= cv2.idft(np.fft.ifftshift(tmp_mask))
    img_wave = (img_wave - img_wave.min()) / (img_wave.max()-img_wave.min()) * 255.0



# mouse callback function
def mouse_listener(event,x,y,flags,param):
    global drawing, SCALE
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        update_mask(int(x/SCALE),int(y/SCALE))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            update_mask(int(x/SCALE),int(y/SCALE))


SCALE = 3

img       = cv2.cvtColor( cv2.imread("lenaColCdsml.png"), cv2.COLOR_BGR2GRAY )
img_dft   = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
img_dft   = np.fft.fftshift(img_dft)

img_mask  = np.zeros( img_dft.shape ) 
img_mask[int(img_mask.shape[0]/2),int(img_mask.shape[1]/2),:] = 1. #直流成分

img_viz   = img_mask * img_dft
img_idft  = cv2.idft(np.fft.ifftshift(img_viz), flags=cv2.DFT_SCALE )
img_wave  = img_idft


drawing   = False






cv2.namedWindow('mask')
cv2.setMouseCallback('mask',mouse_listener)

while(1):
    img_viz3  = cv2.resize(img_viz , None, fx = SCALE, fy = SCALE)
    img_idft3 = cv2.resize(img_idft, None, fx = SCALE, fy = SCALE)
    img_wave3 = cv2.resize(img_wave, None, fx = SCALE, fy = SCALE)

    cv2.imshow('mask',0.0001 * img_viz3 [:,:,0])
    cv2.imshow('idft',np.uint8(img_idft3[:,:,0]))
    cv2.imshow('wave',np.uint8(img_wave3[:,:,0]))

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()