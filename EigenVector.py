# coding: UTF-8
# 固有値の意味理解のため，線形変換結果を可視化する

# 参考にしたweb page
# http://qiita.com/kenmatsu4/items/2a8573e3c878fc2da306

import matplotlib.pyplot as plt 
import numpy as np
from matplotlib import animation as ani

N = 30
R = 2.0

colPalet = []
for i in range(N) :
    colPalet.append([0.1 * (i%10), 0.3 * (i/10), 0])

if(0) : 
    #通常行列0
    A = [[ 4, 2], [ 1, 3] ]
    x1 = 2
    v1 = [-1,1]
    x2 = 5
    v2 = [ 2,1]
elif(0) : 
    #通常行列1
    A = [[ 1, 0], [ 1, 3] ]
    x1 = 1
    v1 = [2,-1]
    x2 = 3
    v2 = [ 0,1]
elif(0) : 
    #通常行列2
    A = [[ 1, 3], [ 2, -1] ]
    x1 = -np.sqrt(7)
    v1 = [0.5*(-np.sqrt(7) + 1), 1] 
    x2 =  np.sqrt(7)
    v2 = [0.5*( np.sqrt(7) + 1), 1] 
elif(0) : 
    #通常行列3
    A = [[ 2, 1], [ 1.5, 2] ]
    x1 = (-np.sqrt(6) + 4.0 ) / 2.0
    v1 = [-np.sqrt(6)/3, 1] 
    x2 = ( np.sqrt(6) + 4.0 ) / 2.0
    v2 = [ np.sqrt(6)/3, 1] 
elif(1) : 
    #対称行列1
    A = [[ 1.0, 2.0], [ 2.0, 1.0] ]
    x1 = -1
    v1 = [-1, 1]
    x2 = 3
    v2 = [ 1,1]
else:
    #対称行列1
    A = [[ 1.0, -2.0], [ -2.0, 1.0] ]
    x1 = -1
    v1 = [ 1, 1]
    x2 = 3
    v2 = [-1, 1]


fig, (figL, figR) = plt.subplots(ncols=2, figsize=(2*5,5), sharex=True )
figL.set_xlim(-10, 10)
figL.set_ylim(-10, 10)
figR.set_xlim(-10, 10)
figR.set_ylim(-10, 10)
figL.plot([-10,10],[  0, 0],"k", linewidth=1)
figL.plot([  0, 0],[-10,10],"k", linewidth=1)
figR.plot([-10,10],[  0, 0],"k", linewidth=1)
figR.plot([  0, 0],[-10,10],"k", linewidth=1)
figL.plot([0,R*x1*v1[0]],[0,R*x1*v1[1]],"k", linewidth=3, color="r")
figL.plot([0,R*x2*v2[0]],[0,R*x2*v2[1]],"k", linewidth=3, color="b")
figR.plot([0,R*x1*v1[0]],[0,R*x1*v1[1]],"k", linewidth=3, color="r")
figR.plot([0,R*x2*v2[0]],[0,R*x2*v2[1]],"k", linewidth=3, color="b")

for i in range(N) : 
    angle = i * 2.0 * np.pi / N
    a = [R * np.cos(angle), R * np.sin(angle)]
    b = np.dot(A , a)
    c = colPalet[i%len(colPalet)]
    figL.scatter(a[0], a[1], color=c)
    figR.scatter(b[0], b[1], color=c)
    figR.plot([a[0],b[0]],[a[1],b[1]],"k" , linewidth=1)


plt.show()

