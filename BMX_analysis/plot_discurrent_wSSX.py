# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:03:09 2018

@author: dschaffner
"""

import load_picoscope_bmx_102918 as ldbmx
import numpy as np
import matplotlib.pylab as plt
import os

date='102918'
#for shot in np.arange(10,30):
shot = 13
time_ms,time_s,timeB_s,Brdot7,Brdot9,Btdot7,Btdot9,Bzdot7,Bzdot9,Br7,Br9,Bt7,Bt9,Bz7,Bz9,disI_2p2,light=ldbmx.load_picoscope(shot)
shot = 20
time_ms,time_s,timeB_s,Brdot7,Brdot9,Btdot7,Btdot9,Bzdot7,Bzdot9,Br7,Br9,Bt7,Bt9,Bz7,Bz9,disI_1p8,light=ldbmx.load_picoscope(shot)


import ssxreadin as sdr
data=sdr.scope_data('100313',41)
#ncolors=5#34
#colors = np.zeros([ncolors,4])
#for i in np.arange(ncolors):
#    c = cm.spectral(i/float(ncolors),1)
#    colors[i,:]=c
#points = ['o','v','s','p','*','h','^','D','+','>','H','d','x','<']
        
plt.rc('axes',linewidth=2.0)
plt.rc('xtick.major',width=2.0)
plt.rc('ytick.major',width=2.0)
plt.rc('xtick.minor',width=2.0)
plt.rc('ytick.minor',width=2.0)
plt.rc('lines',markersize=8,markeredgewidth=0.0,linewidth=1.0)

fig=plt.figure(num=1,figsize=(10,6),dpi=600,facecolor='w',edgecolor='k')
left  = 0.07  # the left side of the subplots of the figure
right = 0.97    # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.15   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

ax1=plt.subplot(1,1,1)
plt.plot(time_ms,disI_2p2/1000.0,color='red',label='2.2kV')
plt.plot(time_ms,disI_1p8/1000.0,color='orange',label='1.8kV')
plt.xticks(fontsize=12)
plt.xlabel(r't [$\mu$s]',fontsize=16)
plt.xlim(0,198)
plt.yticks(fontsize=12)
plt.ylabel('Discharge Current kA',fontsize=16)
plt.ylim(0,80)
#plt.vlines(0,0,80,color='gray',linestyle='dotted',linewidth=3.5)

leg=plt.legend(loc='upper right',fontsize=12,frameon=False,handlelength=5)

savefilename='Discharge_current_compare_forAPSDPP18.png'
#savefilename='Discharge_current_'+date+'_shot'+str(shot)+'_forAPSDPP18.png'
savefile = os.path.normpath(savefilename)
#plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
#plt.clf()
#plt.close()