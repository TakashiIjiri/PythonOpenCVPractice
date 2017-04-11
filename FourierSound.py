#coding:utf-8
#片岡秀公AT立命館大が書いたコードを井尻が修正したもの
import os
import os.path
import csv
import wave as wave_loader
import numpy as np
import pylab as plt


# load wave and extract right ch
def wave_load( f_path ):
    wf         =   wave_loader.open( f_path, "r")
    ch_N       =   wf.getnchannels()
    sample_N   =   wf.getnframes()
    framerate  =   wf.getframerate()
    data       =   np.frombuffer( wf.readframes(sample_N), dtype="int16")
    wf.close()

    #extract right ch
    data       =   data[ 0::ch_N ]

    print(f_path)
    print("\n\n\nLOAD WAVE")
    print("ch Num   ", ch_N)
    print("data size", len(data) )
    print("framerate", framerate )
    print("LOAD WAVE DONE\n\n\n")
    return data, framerate


def genTimeArrayForWave( data, framerate):
    times = np.arange(0, float(len(data)) / framerate, 1.0 / framerate)
    if len(times) < len(data) : data  = data [:len(times)]
    if len(times) > len(data) : times = times[:len(data) ]
    return times


if __name__ == "__main__":
    #音データの読み込み
    wave,framerate = wave_load( "imgs/apple-004.wav" )
    #hammingWindow = np.hamming(len(wave))
    #wave = wave * hammingWindow

    #グラフの横軸作成
    x_times    = genTimeArrayForWave( wave, framerate)
    x_waveCoef = np.arange(0,len(wave),1)

    # FFT
    wave_fft = np.fft.fft(wave, len(wave))
    wave_fft = np.fft.fftshift( wave_fft )

    #lowpass mask to FFT data
    wave_fft_mask = np.zeros(wave_fft.shape,dtype=complex)
    K = int(len(wave_fft)/2 )
    for i in range(K-500, K + 500) :
        wave_fft_mask[i] = wave_fft[i]

    #plot 1)original, 2)FFT, 3)masked FFT 4) IFFT of (3)
    plt.figure(1)
    plt.plot(x_times, wave    )
    plt.savefig("1.png")

    plt.figure(2)
    plt.plot(x_waveCoef, wave_fft )
    plt.savefig("2.png")

    plt.figure(3)
    plt.plot(x_waveCoef, wave_fft_mask  )
    plt.savefig("3.png")

    plt.figure(4)
    plt.plot(x_waveCoef, np.fft.ifft( np.fft.fftshift( wave_fft_mask )  ) )
    plt.savefig("4.png")

    plt.show()
