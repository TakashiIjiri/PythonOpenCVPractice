# -*- coding: utf-8 -*-
import numpy as np
import sys
import cv2
import math

import KernelGenerator as kernel

#自作すると遅いので　np.fft に頼る
def export_complex_img( fname, _img) :
    img = np.copy(_img)
    img_r = np.real( img )
    img_i = np.imag( img )
    img_a = np.abs ( img )

    #可視化時はマイナス項無視
    img_a *= 255 / max( 0.1, min( 100000, np.max(img_r)) )
    img_r *= 255 / max( 0.1, min( 100000, np.max(img_r)) )
    img_i *= 255 / max( 0.1, min( 100000, np.max(img_i)) )
    cv2.imwrite(fname + "_real.png" , np.uint8( img_r ))
    cv2.imwrite(fname + "_imag.png" , np.uint8( img_i ))
    cv2.imwrite(fname + "_abso.png" , np.uint8( img_a ))



def deconvolution( img_f, kernel_f, threshold = 0.01 ) :
    H = img_f.shape[0]
    W = img_f.shape[1]
    img_fft1 = np.copy(img_f)
    img_fft2 = np.copy(img_f)
    img_fft3 = np.copy(img_f)
    img_fft4 = np.copy(img_f)

    for v in range ( H ) :
        for u in range ( W ) :
            #simple algorithm
            simple_v = kernel_f[v,u]
            if( abs(simple_v) < threshold) :
                if( simple_v < 0 ) : simple_v = -threshold
                else: simple_v = threshold
            img_fft1[v,u] /= simple_v

            #Weiner filter
            val = kernel_f[v,u]
            img_fft2[v,u] *= val / (val * val + 0.00001 )
            img_fft3[v,u] *= val / (val * val + 0.00100 )
            img_fft4[v,u] *= val / (val * val + 0.10000 )

    cv2.imwrite("tmp.png" , np.uint8( 0.01 * np.abs( np.fft.fftshift(img_fft1)) ))

    deconv1 = np.real(np.fft.ifft2( img_fft1 ))
    deconv2 = np.real(np.fft.ifft2( img_fft2 ))
    deconv3 = np.real(np.fft.ifft2( img_fft3 ))
    deconv4 = np.real(np.fft.ifft2( img_fft4 ))
    return deconv1, deconv2, deconv3, deconv4



#2種類(1+3パターン)のgaussian kernel deconvolutionを適用し、2枚の画像を返す
def deconvolution_gauss( img, sigma ) :
    H = img.shape[0]
    W = img.shape[1]
    img_fft = np.fft.fft2(img)
    gauss, gauss_f = kernel.gaussian_kernel( H,W, np.float64, sigma)
    return deconvolution(img_fft, gauss_f, 0.1)

#4種類のgaussian kernel deconvolutionを適用し、2枚の画像を返す
def deconvolution_linePSF( img, width, theta ) :
    H = img.shape[0]
    W = img.shape[1]
    img_fft = np.fft.fft2(img)
    linekernel, linekernel_f = kernel.line_kernel( H,W, np.float64, width, theta)
    #return deconvolution(img_fft, np.fft.fft2(linekernel))
    return deconvolution(img_fft, linekernel_f, 0.1)



if __name__ == '__main__':
    #1. load input image
    fname_in  = sys.argv[1]
    img = cv2.cvtColor( cv2.imread( fname_in ), cv2.COLOR_RGB2GRAY)
    img = np.float64( img )
    H, W = img.shape[0], img.shape[1]

    sigma = 6.0
    gauss, gauss_f = kernel.gaussian_kernel(H,W, np.float64, sigma)

    width = 20.0
    theta = math.pi * 0.8
    lineKernel, lineKernel_f = kernel.line_kernel(H,W, np.float64, width, theta)

    img_noise = 0.1 * np.random.rand(img.shape[0], img.shape[1] )

    #lineKernel_f = np.fft.fft2(lineKernel)
    #compute blurred image
    img_fft        = np.fft.fft2 ( img   )
    img_gaussBlurr = np.fft.ifft2( gauss_f      * img_fft) + img_noise
    img_lineBlurr  = np.fft.ifft2( lineKernel_f * img_fft) + img_noise

    debug_export = 1
    if( debug_export ) :
        cv2.imwrite("img.png"            , np.uint8( img ) )
        cv2.imwrite("img_noise.png"      , np.uint8( 10*img_noise ) )
        cv2.imwrite("img_gaussBlurr.png" , np.uint8( np.real(img_gaussBlurr) ))
        cv2.imwrite("img_lineBlurr.png"  , np.uint8( np.real(img_lineBlurr ) ))

        lineblurr_fft =  np.fft.fft2(img_lineBlurr)
        cv2.imwrite("lineblurr_fft.png" , np.uint8( 0.01 * np.abs( np.fft.fftshift(lineblurr_fft)) ))

        cv2.imwrite("kernelL.png"        , np.uint8( width * 2 * 255 * np.fft.fftshift( lineKernel )) )
        cv2.imwrite("kernelL_fft.png"    , np.uint8( 120 * np.fft.fftshift( lineKernel_f ) + 127) )
        cv2.imwrite("kernelG.png"        , np.uint8(50000 * np.fft.fftshift( gauss)      ) )
        cv2.imwrite("kernelG_fft.png"    , np.uint8( 255 * np.abs( np.fft.fftshift(gauss_f  )) ))

        """
        cv2.imwrite("img_fft.png"        , np.uint8( np.real( np.fft.fftshift( img_fft) ) / W / H ) )

        blurr_fft =  np.fft.fft2(img_gaussBlurr)
        cv2.imwrite("blurr_fft.png" , np.uint8( 0.01 * np.abs( np.fft.fftshift(blurr_fft)) ))
        cv2.imwrite("img_fft.png"   , np.uint8( 0.01 * np.abs( np.fft.fftshift(img_fft  )) ))
        """

    decImg1, decImg2, decImg3, decImg4 = deconvolution_gauss(img_gaussBlurr, sigma)
    decImg1[ decImg1 > 255] = 255
    decImg2[ decImg2 > 255] = 255
    cv2.imwrite("decImg1.png",  np.uint8( decImg1 ) )
    cv2.imwrite("decImg2.png",  np.uint8( decImg2 ) )
    cv2.imwrite("decImg3.png",  np.uint8( decImg3 ) )
    cv2.imwrite("decImg4.png",  np.uint8( decImg4 ) )

    decImg1, decImg2, decImg3, decImg4 = deconvolution_linePSF(img_lineBlurr, width, theta)
    decImg1[ decImg1 > 255] = 255
    decImg2[ decImg2 > 255] = 255
    cv2.imwrite("decImgLine1.png",  np.uint8( decImg1 ) )
    cv2.imwrite("decImgLine2.png",  np.uint8( decImg2 ) )
    cv2.imwrite("decImgLine3.png",  np.uint8( decImg3 ) )
    cv2.imwrite("decImgLine4.png",  np.uint8( decImg4 ) )
