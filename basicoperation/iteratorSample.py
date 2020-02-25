# coding: UTF-8

import numpy as np
import time
import itertools

N = 1000
d = 0
img = np.ones((N,N))

t1 = time.clock() #----------------
for y in range(N) :
    for x in range(N) :
        d += img[y,x]

t2 = time.clock() #----------------
for i in img  :
    for j in i :
        d += j

t3 = time.clock() #----------------
for y,x in np.ndindex(img.shape)  :
    d += img[y,x]

t4 = time.clock() #----------------
for y, x in itertools.product(range(img.shape[0]),range(img.shape[1])):
    d += img[y,x]

t5 = time.clock()
d+= np.sum( np.sum(img) )
t6 = time.clock()

print(d)
print( t2-t1, t3-t2, t4-t3, t5-t4, t6-t5)
