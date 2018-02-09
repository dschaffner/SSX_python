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

x=np.arange(1000)
y=np.sin(0.003*x)
y=np.random.normal(0,1,100000)
import chaotichenonsave_davidmod as henon
arr = henon.henonsave(1000)
y = arr[0,:]#extrat data from function return

num_delays = 1#249
PEs = np.zeros([num_delays+1])
SCs = np.zeros([num_delays+1])

delay = 1
embed_delay = 5
nfac = factorial(embed_delay)

for loop_delay in np.arange(1,num_delays+1):
    
    PEs[loop_delay],SCs[loop_delay] = CH(y,5,delay=loop_delay)
    print 'On Delay ',loop_delay


datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\'    
filename='PE_SC_sinewave_'+str(num_delays)+'_delays.npz'
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


savefilename='C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\DSCOVR Data\\henontimeseries.png'
#savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefilename,dpi=300,facecolor='w',edgecolor='k')

