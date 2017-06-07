#coding:utf-8

import pylab as plt
import numpy as np
import random

mean = [2,2]
cov  = [[3,0],[0,3]]
X, Y = np.random.multivariate_normal(mean, cov, 500).T

plt.plot(X, Y, 'ro')




mean = [10,2]
cov  = [[0.3,0],[0,0.3]]
X, Y = np.random.multivariate_normal(mean, cov, 500).T

plt.plot(X, Y, 'bo')


plt.show()
