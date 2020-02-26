# -*- coding: utf-8 -*-
import numpy as np
import sys
import cv2
import math


# 2D gaussian funciton  f(x,y) = 1/(2 pi s^2) exp(- (x^2+y^2) / (2*s^2) )
# its fourier transform F(u,v) = exp(- (x^2+y^2) * s^2 / 2 )

# W,H : resolution of (x,y) and (u,v)
# sigma : standard deviation
def gaussian_kernel ( H, W,dtype = np.float32, sigma = 3.0) :
    coef = 1.0 / (2.0 * math.pi * sigma * sigma)
    c1   = 1.0 / (2.0 * sigma * sigma)
    c2   = sigma * sigma / 2

    kernel   = np.zeros((H,W), dtype )
    kernel_f = np.zeros((H,W), dtype )
    for y in range ( H ) :
        for x in range ( W ) :
            px = x if( x <  W / 2) else (W-x)
            py = y if( y <  H / 2) else (H-y)
            pu = px * 2 * math.pi / W
            pv = py * 2 * math.pi / H
            kernel  [y,x] = coef * math.exp( - c1 * (px*px + py*py) )
            kernel_f[y,x] =        math.exp( - c2 * (pu*pu + pv*pv) )
    return kernel, kernel_f


# w(y,x) = 1/(2*W)  (線分上), 0 (線分上)　: 線分は長さ -W ~ W, 傾きtheta
# W(v,u) =
def line_kernel( H, W, dtype = np.float32, width = 10, theta = 0.0) :

    kernel   = np.zeros((H,W), dtype )
    kernel_f = np.zeros((H,W), dtype )

    c = math.cos(theta)
    s = math.sin(theta)

    for y in range ( H ) :
        for x in range ( W ) :
            px = x if( x <  W / 2) else -(W-x)
            py = y if( y <  H / 2) else -(H-y)
            pu = px * 2 * math.pi / W
            pv = py * 2 * math.pi / H

            #本来は直線を書くべきだけど少し雑な実装を
            if( abs( px*c + py*s) <= width + 0.1 and abs(px*s - py*c) < 0.5 )  :
                kernel  [y,x] = 1 / (2 * width)

            a = (width) * ( pu * c + pv * s )
            if (    abs(a) > 0.00001) :
                kernel_f[y,x] = math.sin( a ) / a
            else :
                kernel_f[y,x] = 1

    return kernel, kernel_f







def MyFourierTransform2d(img) :
    #出力画像を準備(グレースケール，float型)
    H = img.shape[0]
    W = img.shape[1]
    Rvu = np.zeros_like( img )
    Ivu = np.zeros_like( img )

    #Fourier transform フィルタ処理 (for文は遅いけど今回は気にしない)
    for v in range( H ) :
        for u in range( W ) :
            for y in range( H ) :
                for x in range( W ) :
                    c = math.cos( - 2 * math.pi * ( u*x/W +  v*y/H) )
                    s = math.sin( - 2 * math.pi * ( u*x/W +  v*y/H) )
                    Rvu[v,u] += c * img[y,x]
                    Ivu[v,u] += s * img[y,x]
    return Rvu, Ivu




if __name__ == '__main__':
    gauss, gauss_f = gaussian_kernel( 50, 50, np.float64, 0.5)
    gauss_fft   = np.fft.fft2(gauss)
    #gauss_myDftR, gauss_myDftI = MyFourierTransform2d(gauss)

    cv2.imwrite("kernel.png"     , np.uint8( gauss    * 1024 ))
    cv2.imwrite("kernel_f.png"   , np.uint8( gauss_f  * 300 ))
    cv2.imwrite("kernel_fft.png" , np.uint8( np.real(gauss_fft) * 300 ))
    #cv2.imwrite("kernel_myR.png" , np.uint8( gauss_myDftR * 1024 ) )
    #cv2.imwrite("kernel_myI.png" , np.uint8( gauss_myDftI * 1024 ) )


    kernel1D, kernel1D_f = line_kernel( 51, 51, np.float64, 5, math.pi * 0.1)
    kernel1D_fft   = np.fft.fft2(kernel1D)
    cv2.imwrite("lineKernel.png"     , np.uint8( kernel1D    * 20 * 200 ))
    cv2.imwrite("lineKernel_f.png"   , np.uint8( kernel1D_f  * 100 + 128))
    cv2.imwrite("lineKernel_fft.png", np.uint8( np.real(kernel1D_fft)  * 100 + 128))
