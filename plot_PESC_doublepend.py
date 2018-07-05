# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:00:45 2017

@author: dschaffner
"""
import numpy as np
import matplotlib.pylab as plt
from loadnpzfile import loadnpzfile
from calc_PE_SC import PE, CH, PE_dist, PE_calc_only
from collections import Counter
from math import factorial
import os

#calc_PESC_solarwind_chen.py
datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\DoublePendulum\\'
fileheader='DoubPen_L1-1_L2-1_m1-1_m2-1_9p8_scan_theta1_ic'
fileheader='DoubPen_L1-1_L2-1_m1-1_m2-1_9p8_chaos'
fileheader = 'PE_SC_DPDoubPen_LsMsEq1_9p8_ICP1_embeddelay5_41_delays'
npz='.npz'
datafile = loadnpzfile(datadir+fileheader+npz)
#initial_theta1s=datafile['initial_theta1s']
#theta1s=datafile['theta1s']
#num_thetas = len(initial_theta1s)

PEsx1_P=datafile['PEsx1']
SCsx1_P=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_9p8_ICQ1_embeddelay5_41_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_Q=datafile['PEsx1']
SCsx1_Q=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_9p8_ICC1_embeddelay5_41_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_C=datafile['PEsx1']
SCsx1_C=datafile['SCsx1']

delayindex=datafile['delays']

tmax, dt = 100, 0.001
t = np.arange(0, tmax+dt, dt)

import matplotlib.cm as cm
colors = np.zeros([7,4])
for i in np.arange(7):
    c = cm.spectral(i/7.,1)
    colors[i,:]=c
points = ['o','v','s','p','*','h','^','D','+','>','H','d','x','<']
        
plt.rc('axes',linewidth=0.75)
plt.rc('xtick.major',width=0.75)
plt.rc('ytick.major',width=0.75)
plt.rc('xtick.minor',width=0.75)
plt.rc('ytick.minor',width=0.75)
plt.rc('lines',markersize=2,markeredgewidth=0.0)

plt.rc('lines',markersize=1.5,markeredgewidth=0.0)
fig=plt.figure(num=1,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.2  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.97      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
plt.semilogx(delayindex,SCsx1_P,color=colors[1,:],label='Periodic')
plt.semilogx(delayindex,SCsx1_Q,color=colors[3,:],label='Quasi-Periodic')
plt.semilogx(delayindex,SCsx1_C,color=colors[6,:],label='Chaotic')


#plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Delay Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Statistical Complexity',fontsize=9)
#plt.xlim(1,250)
plt.ylim(0,0.5)
plt.legend(loc='upper left',fontsize=5,frameon=False,handlelength=5)

savefilename='SC_DPx1_ICP1_ICQ1_ICC1.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')




fig=plt.figure(num=2,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.2  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.97      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
plt.semilogx(delayindex,PEsx1_P,color=colors[1,:],label='Periodic')
plt.semilogx(delayindex,PEsx1_Q,color=colors[3,:],label='Quasi-Periodic')
plt.semilogx(delayindex,PEsx1_C,color=colors[6,:],label='Chaotic')


#plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Delay Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Permutation Entropy',fontsize=9)
#plt.xlim(1,250)
plt.ylim(0,1,0)
plt.legend(loc='upper left',fontsize=5,frameon=False,handlelength=5)

savefilename='PE_DPx1_ICP1_ICQ1_ICC1.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')
