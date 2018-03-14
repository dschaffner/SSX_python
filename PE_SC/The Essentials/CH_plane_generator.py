# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 13:57:52 2016

@author: dschaffner
"""
import numpy as np
import chaotichenonsave_davidmod as henon
import PE_CH_davidmod as pe
import matplotlib.pylab as plt
import Cmaxmin as cpl

#Make Chaotic henon map timeseries
arr = henon.henonsave(1000)
arr = arr[0,:]#extrat data from function return
#plt.figure(1)
#plt.plot(arr)
#plt.title('Henon Map')


#CH Plane analysis
#Run array through code at different timestep integers using n=5
h=np.zeros([10])
c=np.zeros([10])

#embedding dimension
ndim=5

#t=1
h[0],c[0]=pe.CH(arr,ndim,delay=1)

#t=2 (skips every other data point in arr)
h[1],c[1]=pe.CH(arr,ndim,delay=2)

#t=3 (skips every three data point in arr)
h[2],c[2]=pe.CH(arr,ndim,delay=3)

#t=4
h[3],c[3]=pe.CH(arr,ndim,delay=4)

#t=5
h[4],c[4]=pe.CH(arr,ndim,delay=5)

#t=6
h[5],c[5]=pe.CH(arr,ndim,delay=6)

#t=7
h[6],c[6]=pe.CH(arr,ndim,delay=7)

#t=8
h[7],c[7]=pe.CH(arr,ndim,delay=8)

#t=9
h[8],c[8]=pe.CH(arr,ndim,delay=9)

#t=10
h[9],c[9]=pe.CH(arr,ndim,delay=10)

#Plot C vs H as a CHplane
Cminx, Cminy, Cmaxx, Cmaxy = cpl.Cmaxmin(200,ndim)

fig1=plt.figure(2)
plt.plot(Cminx,Cminy,'k-',Cmaxx,Cmaxy,'k-')
plt.xlabel("Entropy", fontsize=15)
plt.ylabel("Jensen-Shannon Complexity", fontsize=15)
plt.axis([0,1.0,0,0.45])
plt.xticks(np.arange(0,1.1,0.1))
plt.yticks(np.arange(0,0.45,0.05))

plt.plot(h,c,'-o')