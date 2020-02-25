# coding: UTF-8
import numpy as np
import cv2
from   mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

N = 40

#x, y, z = axes3d.get_test_data(0.05)
points = np.zeros((N,N,3))

scale = 0.25
s = 1.
c = 1.0 / ( 2 * np.pi * s*s )

for y in range(N) :
    for x in range(N) :
        xp = (x - N/2)* scale
        yp = (y - N/2)* scale
        points[y,x,0] = xp
        points[y,x,1] = yp

        #simple gaussian
        points[y,x,2] = c * np.exp( -(xp*xp + yp*yp) / (2*s*s) )

        #quad func
        #points[y,x,2] = - 2 * xp * xp -  yp * yp

        #sinc function
        #sincx = 1
        #sincy = 1
        #if( xp != 0 ) : sincx = np.sin( np.pi * xp  ) / (np.pi * xp)
        #if( yp != 0 ) : sincy = np.sin( np.pi * yp  ) / (np.pi * yp)
        #points[y,x,2] = c * sincx * sincy

maxV = np.max(points[:,:,2])
img = points[:,:,2] / maxV
#cv2.imwrite( "chart3D.bmp", np.uint8(255 * img) )

# Plot a basic wireframe.
ax.plot_wireframe(points[:,:,0], points[:,:,1], points[:,:,2], rstride=1, cstride=1)
plt.show()
