#coding:utf-8
import os
import os.path
import csv
import wave as wave_loader
import numpy as np
import pylab as plt


T = 2.0
N = 300

#simple sin and cos
xAxis     = [-8.0 +  16.0 * i / N for i in range(N)]
plt.clf()
plt.ylim([-1.5,1.5])
plt.plot( xAxis , [np.sin( xAxis[i]) for i in range(N)], "g" )
plt.savefig("sin.png")
plt.clf()
plt.ylim([-1.5,1.5])
plt.plot( xAxis , [np.cos( xAxis[i]) for i in range(N)], "r"  )
plt.savefig("cos.png")



xAxis     = [-T/2.0 +  T * i / N for i in range(N)]
sin_funcs = []
cos_funcs = []


for K in range(6) :
    sin_funcs.append( np.array( [np.sin( xAxis[i] * K * 2 * np.pi / T ) for i in range(N)]) )
    cos_funcs.append( np.array( [np.cos( xAxis[i] * K * 2 * np.pi / T ) for i in range(N)]) )

    plt.clf()
    plt.ylim([-1.5,1.5])
    plt.plot(xAxis, sin_funcs[K], "g" )
    plt.savefig("sin" + str(K) + ".png")

    plt.clf()
    plt.ylim([-1.5,1.5])
    plt.plot(xAxis, cos_funcs[K], "r" )
    plt.savefig("cos" + str(K) + ".png")

print( type(cos_funcs[0]) )
cos_funcs[0] = 0.3*cos_funcs[0]
cos_funcs[1] = 0.2*cos_funcs[1]
cos_funcs[2] = 0.7*cos_funcs[2]
cos_funcs[3] = 0.1*cos_funcs[3]
cos_funcs[4] = 0.1*cos_funcs[4]
cos_funcs[5] = 0.1*cos_funcs[5]
sin_funcs[0] = 0.0*sin_funcs[0]
sin_funcs[1] = 0.3*sin_funcs[1]
sin_funcs[2] = 0.1*sin_funcs[2]
sin_funcs[3] = 0.3*sin_funcs[3]
sin_funcs[4] = 0.1*sin_funcs[4]
sin_funcs[5] = 0.2*sin_funcs[5]

plt.clf()
plt.ylim([-1.5,1.5])
plt.plot(xAxis, cos_funcs[0], "r" )
plt.plot(xAxis, cos_funcs[1], "r" )
plt.plot(xAxis, cos_funcs[2], "r" )
plt.plot(xAxis, cos_funcs[3], "r" )
plt.plot(xAxis, cos_funcs[4], "r" )
plt.plot(xAxis, cos_funcs[5], "r" )
plt.plot(xAxis, sin_funcs[0], "g" )
plt.plot(xAxis, sin_funcs[1], "g" )
plt.plot(xAxis, sin_funcs[2], "g" )
plt.plot(xAxis, sin_funcs[3], "g" )
plt.plot(xAxis, sin_funcs[4], "g" )
plt.plot(xAxis, sin_funcs[5], "g" )
plt.savefig("Waves.png")

funcSum = np.zeros(cos_funcs[0].shape)
for K in range(6) :
    funcSum = funcSum + cos_funcs[K] +  sin_funcs[K]


plt.clf()
plt.ylim([-2.0,2.0])
plt.plot(xAxis, funcSum, "b" )
plt.savefig("func.png")
