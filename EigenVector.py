# coding: UTF-8
# 固有値の意味理解のため，線形変換結果を可視化する

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

N = 4
xmin = -12
xmax =  12
ymin = -12
ymax =  12

fig, (figL, figR) = plt.subplots(ncols=2, figsize=(2*N,N), sharex=True )
figL.set_xlim(xmin, xmax)
figL.set_ylim(ymin, ymax)
figR.set_xlim(xmin, xmax)
figR.set_ylim(ymin, ymax)
figL.plot([xmin,xmax],[0,0],"k", linewidth=1)
figL.plot([0,0],[ymin,ymax],"k", linewidth=1)
figR.plot([xmin,xmax],[0,0],"k", linewidth=1)
figR.plot([0,0],[ymin,ymax],"k", linewidth=1)

for i in range( -N+1 , N ):
    for j in range( -N+1, N ):
        a = np.array( [float(j), float(i)] )
        b = np.dot(A , a)

        idx = (i*N + j)
        offset = .2  
        figL.text(a[0]-offset, a[1]-offset, "%d"%idx, color="blue")
        figR.text(b[0]-offset, b[1]-offset, "%d"%idx, color="red" )

plt.show()

