# coding: UTF-8

import numpy as np
import cv2

from   mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

N = 40

#X, Y, Z = axes3d.get_test_data(0.05)
X = np.zeros((N,N))
Y = np.zeros((N,N))
Z = np.zeros((N,N))

scale = 0.25
s = 1.
c = 1.0 / ( 2 * np.pi * s*s )

for y in range(N) :
    for x in range(N) :
        xp = (x - N/2)* scale
        yp = (y - N/2)* scale
        X[y,x] = xp
        Y[y,x] = yp
        #simple gaussian
        #Z[y,x] = c * np.exp( -(xp*xp + yp*yp) / (2*s*s) )

        Z[y,x] = - 2 * xp * xp -  yp * yp

        #sinc function
        #sincx = 1
        #sincy = 1
        #if( xp != 0 ) : sincx = np.sin( np.pi * xp  ) / (np.pi * xp)
        #if( yp != 0 ) : sincy = np.sin( np.pi * yp  ) / (np.pi * yp)
        #Z[y,x] = c * sincx * sincy

maxV = np.max(Z)
img = Z / maxV
cv2.imwrite( "output.bmp", np.uint8(254 * img) )

# Plot a basic wireframe.
ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)



plt.show()
