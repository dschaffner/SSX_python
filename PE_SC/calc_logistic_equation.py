#calc_logistic_function

import numpy as np
import matplotlib.pylab as plt

initialx = 0.8
r = 3.999
xarr = np.zeros([10001])
xarr[0]=initialx
for xn in np.arange(10000):
    xarr[xn+1]=r*xarr[xn]*(1-xarr[xn])
    
plt.figure(1)
plt.plot(xarr)

initialx=0.1
a=0.2
xarr = np.zeros(10001)
xarr[0]=initialx
for xn in np.arange(10000):
    xarr[xn+1]=a*np.sin(np.pi*xarr[xn])
    
#plt.figure(2)
plt.plot(xarr)