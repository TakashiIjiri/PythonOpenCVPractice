#coding:utf-8
import os
import os.path
import csv
import wave as wave_loader
import numpy as np
import pylab as plt



N = 500

sinA = [0]*6
sinA[0] = [0.81 * np.sin(0 * 2 * np.pi * i / N ) for i in range(N) ]
sinA[1] = [0.81 * np.sin(1 * 2 * np.pi * i / N ) for i in range(N) ]
sinA[2] = [0.81 * np.sin(2 * 2 * np.pi * i / N ) for i in range(N) ]
sinA[3] = [0.81 * np.sin(3 * 2 * np.pi * i / N ) for i in range(N) ]
sinA[4] = [0.81 * np.sin(10 * 2 * np.pi * i / N ) for i in range(N) ]
sinA[5] = [0.81 * np.sin(15   * 2 * np.pi * i / N ) for i in range(N) ]

xAxis  = [i for i in range(N)]

for i in range(7) :
    plt.figure(i)
    plt.plot(xAxis, sinA[i] )
    plt.savefig(str(i) + ".png")
