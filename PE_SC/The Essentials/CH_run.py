# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 13:57:52 2016

@author: dschaffner
"""
import numpy as np
import chaotichenonsave_davidmod as henon
import calc_PE_SC_test as pe
import matplotlib.pylab as plt

#Make Chaotic henon map timeseries
arr = henon.henonsave(1000)
arr = arr[0,:]#extrat data from function return
plt.figure(1)
plt.plot(arr)
plt.title('Henon Map')



from random import gauss
from random import seed
from pandas import Series
from pandas.tools.plotting import autocorrelation_plot
# seed random number generator
seed(1)
# create white noise series
series = [gauss(0.0, 1.0) for i in range(1000)]
series = Series(series)

arr = series

#CH Plane analysis
#Run array through code at different timestep integers using n=5
h=np.zeros([10])
c=np.zeros([10])

#t=1
s,se,A,count = pe.PE(arr,5,delay=1)
h[0],c[0] =pe.CH(arr,5,delay=1)

#t=2 (skips every other data point in arr)
#h[1],c[1]=pe.CH(arr,5,delay=2)

#t=3 (skips every three data point in arr)
#h[2],c[2]=pe.CH(arr,5,delay=3)

#t=4
#h[3],c[3]=pe.CH(arr,5,delay=4)

#t=5
#h[4],c[4]=pe.CH(arr,5,delay=5)

#t=6
#h[5],c[5]=pe.CH(arr,5,delay=6)

#t=7
#h[6],c[6]=pe.CH(arr,5,delay=7)

#t=8
#h[7],c[7]=pe.CH(arr,5,delay=8)

#t=9
#h[8],c[8]=pe.CH(arr,5,delay=9)

#t=10
#h[9],c[9]=pe.CH(arr,5,delay=10)

plt.figure(2)
plt.plot(h)
plt.plot(c)
plt.title('PE (upper line) and SC as a function of delay')