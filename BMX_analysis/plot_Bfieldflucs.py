# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:03:09 2018

@author: dschaffner
"""

import load_picoscope_bmx_102918 as ldbmx
import numpy as np
import matplotlib.pylab as plt
import os
import scipy.signal as sps

date='102918'
#for shot in np.arange(10,30):
shot = 13
time_ms,time_s,timeB_s,Brdot7a,Brdot9a,Btdot7a,Btdot9a,Bzdot7a,Bzdot9a,Br7a,Br9a,Bt7a,Bt9a,Bz7a,Bz9a,disI_2p2,light=ldbmx.load_picoscope(shot)
Br7a_dtr = sps.detrend(Br7a)
Br9a_dtr = sps.detrend(Br9a)
Bt7a_dtr = sps.detrend(Bt7a)
Bt9a_dtr = sps.detrend(Bt9a)
Bz7a_dtr = sps.detrend(Bz7a)
Bz9a_dtr = sps.detrend(Bz9a)
B7tota = np.sqrt(Br7a_dtr**2+Bt7a_dtr**2+Bz7a_dtr**2)
B9tota = np.sqrt(Br9a_dtr**2+Bt9a_dtr**2+Bz9a_dtr**2)
shot = 20
time_ms,time_s,timeB_s,Brdot7b,Brdot9b,Btdot7b,Btdot9b,Bzdot7b,Bzdot9b,Br7b,Br9b,Bt7b,Bt9b,Bz7b,Bz9b,disI_1p8,light=ldbmx.load_picoscope(shot)
Br7b_dtr = sps.detrend(Br7b)
Br9b_dtr = sps.detrend(Br9b)
Bt7b_dtr = sps.detrend(Bt7b)
Bt9b_dtr = sps.detrend(Bt9b)
Bz7b_dtr = sps.detrend(Bz7b)
Bz9b_dtr = sps.detrend(Bz9b)
B7totb = np.sqrt(Br7b_dtr**2+Bt7b_dtr**2+Bz7b_dtr**2)
B9totb = np.sqrt(Br9b_dtr**2+Bt9b_dtr**2+Bz9b_dtr**2)


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

fig=plt.figure(num=1,figsize=(8,10),dpi=600,facecolor='w',edgecolor='k')
left  = 0.12  # the left side of the subplots of the figure
right = 0.97    # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.25   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

ax1=plt.subplot(2,1,1)
plt.plot(timeB_s*1e6,B7tota,color='red',label='z=26.0cm')
plt.plot(timeB_s*1e6,B9tota,color='blue',label='z=28.5cm')
plt.xticks(fontsize=12)
plt.xlabel(r't [$\mu$s]',fontsize=16)
plt.xlim(0,198)
yticks = [0,250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000]
plt.yticks(np.array(yticks),yticks,fontsize=12)
plt.ylabel('|B| [G]',fontsize=16)
plt.ylim(0,3000)
plt.title('10292018 Shot 13 2.2kV',fontsize=12)
plt.hlines(933,0,198,color='red',linestyle='solid',linewidth=0.5)
plt.hlines(726,0,198,color='blue',linestyle='solid',linewidth=0.5)
leg=plt.legend(loc='upper right',fontsize=12,frameon=False,handlelength=5)


ax1=plt.subplot(2,1,2)
plt.plot(timeB_s*1e6,B7totb,color='red',label='z=26.0cm')
plt.plot(timeB_s*1e6,B9totb,color='blue',label='z=28.5cm')
plt.xticks(fontsize=12)
plt.xlabel(r't [$\mu$s]',fontsize=16)
plt.xlim(0,198)
plt.yticks(np.array(yticks),yticks,fontsize=12)
plt.ylabel('|B| [G]',fontsize=16)
plt.ylim(0,3000)
plt.title('10292018 Shot 20 1.8kV',fontsize=12)
plt.hlines(774,0,198,color='red',linestyle='solid',linewidth=0.5)
plt.hlines(698,0,198,color='blue',linestyle='solid',linewidth=0.5)
leg=plt.legend(loc='upper right',fontsize=12,frameon=False,handlelength=5)



savefilename='Btot_compare_forAPSDPP18.png'
#savefilename='Discharge_current_'+date+'_shot'+str(shot)+'_forAPSDPP18.png'
savefile = os.path.normpath(savefilename)
#plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()
