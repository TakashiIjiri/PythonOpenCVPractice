#coding:utf-8
import os
import os.path
import csv
import wave as wave_loader
import numpy as np
import pylab as plt




def weight_func(a,x) :
    x = abs(x) 
    if(   x <  1 ) : return (a+2)*x*x*x + -(a+3)*x*x + 1
    elif( x <  2 ) : return a*x*x*x - 5*a*x*x + 8*a*x - 4*a
    return 0



N = 500

xAxis  = [-2.0 + 4.0 * i / N for i in range(N)]
chart  = [weight_func(-0, xAxis[i]) for i in range(N)]

plt.figure(0)
plt.plot(xAxis, chart )
plt.savefig("weightFunc00.png")




#sinA = [0]*6
#sinA[0] = [0.81 * np.sin(0 * 2 * np.pi * i / N ) for i in range(N) ]
#sinA[1] = [0.81 * np.sin(1 * 2 * np.pi * i / N ) for i in range(N) ]
#sinA[2] = [0.81 * np.sin(2 * 2 * np.pi * i / N ) for i in range(N) ]
#sinA[3] = [0.81 * np.sin(3 * 2 * np.pi * i / N ) for i in range(N) ]
#sinA[4] = [0.81 * np.sin(10 * 2 * np.pi * i / N ) for i in range(N) ]
#sinA[5] = [0.81 * np.sin(15   * 2 * np.pi * i / N ) for i in range(N) ]

#xAxis  = [i for i in range(N)]

#for i in range(7) :
#    plt.figure(i)
#    plt.plot(xAxis, sinA[i] )
#    plt.savefig(str(i) + ".png")
