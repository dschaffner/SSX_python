# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:00:45 2017

@author: dschaffner
"""
import numpy as np
import matplotlib.pylab as plt
from loadnpyfile import loadnpyfile
from calc_PE_SC import PE, CH, PE_dist, PE_calc_only
import Cmaxmin as cpl
from collections import Counter
from math import factorial
import os

x=np.arange(100000)
y=np.sin(0.003*x)
#y=np.random.normal(0,1,100000)
import chaotichenonsave_davidmod as henon
arr = henon.henonsave(1000)
#y = arr[0,:]#extrat data from function return

x=np.arange(10000)*0.003
y1=np.sin(1*x)
y2=np.sin(2*x)
y3=np.sin(3*x)
y10=np.sin(10*x)
y20=np.sin(20*x)

y = 10*y1+2*y2+4*y10
y = 0.1*y1+10*y3+10*y10+20*y20
y=y1+y2+y3

datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\Codie Thesis Data\\'
fileheader = 'qp_(m=4)_(th=30.0)_(t=1.0)_(CR=6.0)_(eps=0.4)_(x0=2.350639412)_(y0=6.62220828293)_(vx0=-243.996156434)_(vy0=40.276745914)_data'
npy='.npy'

datafile = loadnpyfile(datadir+fileheader+npy)
y=datafile[:,0]

num_delays = 249
PEs = np.zeros([num_delays+1])
SCs = np.zeros([num_delays+1])

delay = 1
embed_delay = 5
nfac = factorial(embed_delay)

for loop_delay in np.arange(1,num_delays+1):
    
    PEs[loop_delay],SCs[loop_delay] = CH(y,5,delay=loop_delay)
    print 'On Delay ',loop_delay


#datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\'    
#filename='PE_SC_sinewave_'+str(num_delays)+'_delays.npz'
#np.savez(datadir+filename,PEs=PEs,SCs=SCs)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)

fig=plt.figure(num=2,figsize=(3.5,1.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.2  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.2  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

ax1=plt.subplot(1,1,1)
plt.plot(x/10.,y,color='black')

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel(r't[s]',fontsize=9)
#ax1.set_xticklabels([])
#plt.ylabel('Permutation Entropy',fontsize=9)
plt.xlim(1,10)
#plt.ylim(0,0.5)
#plt.grid(b=True,which='both',axis='x',linestyle='dotted',linewidth=0.05)
#plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)

maxx,maxy,minx,miny=Cmaxmin(1000,5)
plt.figure(10)
plt.plot(maxx,maxy)
plt.plot(minx,miny)
plt.plot(PEs[1:],SCs[1:])

plt.figure(20)
plt.plot(x,y)

savefilename='C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\DSCOVR Data\\henontimeseries.png'
#savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefilename,dpi=300,facecolor='w',edgecolor='k')

