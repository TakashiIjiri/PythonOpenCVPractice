# coding: UTF-8

# 参考にしたweb page
# http://qiita.com/kenmatsu4/items/2a8573e3c878fc2da306

import matplotlib.pyplot as plt 
import numpy as np
from matplotlib import animation as ani


#target matrix 
A = [[ 2.0, 1.0],
     [-0.5,-1.5] ]

# 
A = [[ 1.0, 2.0],
     [ 2.0, 1.0] ]



N = 5

xmin = -15
xmax =  15
ymin = -15
ymax =  15

plt.figure( figsize=(N,N) )

for i in range( -N , N ):
    for j in range( -N, N ):
        a = np.array( [float(j), float(i)] )

        idx = (i*N + j)
        offset = .2  
        plt.text(a[0]-offset, a[1]-offset, "%d"%idx, color="blue")

        plt.plot([xmin,xmax],[0,0],"k", linewidth=1)
        plt.plot([0,0],[ymin,ymax],"k", linewidth=1)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

plt.figure( figsize=(N,N) )

        
for i in range( -N , N ):
    for j in range( -N, N ):
        a = np.array( [float(j), float(i)] )
        b = np.dot(A , a)

        idx = (i*N + j)
        offset = .2  
        plt.text(b[0]-offset, b[1]-offset, "%d"%idx, color="red")

        plt.plot([xmin,xmax],[0,0],"k", linewidth=1)
        plt.plot([0,0],[ymin,ymax],"k", linewidth=1)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)



plt.show()

