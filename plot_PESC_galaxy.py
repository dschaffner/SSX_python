# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:00:45 2017

@author: dschaffner
"""
import numpy as np
import matplotlib.pylab as plt
from loadnpzfile import loadnpzfile
from calc_PE_SC import PE, CH, PE_dist, PE_calc_only
import Cmaxmin as cpl
from collections import Counter
from math import factorial
import matplotlib.cm as cm
import os

#calc_PESC_fluid.py

datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\'
fileheader = 'PE_SC_New_DavidData_Class_1_249_delays_longrec'
npy='.npz'

delayindex = np.arange(1,250)

datafile = loadnpzfile(datadir+fileheader+npy)
PEs1 = datafile['PEs']
SCs1 = datafile['SCs']

fileheader = 'PE_SC_New_DavidData_Class_2_249_delays_longrec'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs2 = datafile['PEs']
SCs2 = datafile['SCs']

fileheader = 'PE_SC_New_DavidData_Class_3_249_delays_longrec'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs3 = datafile['PEs']
SCs3 = datafile['SCs']

fileheader = 'PE_SC_sinewave_249_delays'
datafile = loadnpzfile(datadir+fileheader+npy)
PEsin = datafile['PEs']
SCsin = datafile['SCs']

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
left  = 0.15  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.2  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
plt.plot(delayindex,SCs1[1:],color=colors[1,:],label='Trapped Orbits')
plt.plot(delayindex,SCs2[1:],color=colors[3,:],label='Never Trapped Orbits')
plt.plot(delayindex,SCs3[1:],color=colors[6,:],label='ULR Orbits')
plt.plot(delayindex,SCsin[1:],color='black',label='Sine Wave')

plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Delay Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Complexity',fontsize=9)
plt.xlim(1,250)
plt.ylim(0,0.5)
plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)


savefilename='SC_fulldat.png'
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
plt.plot(delayindex,PEs1[1:],color=colors[1,:],label='Trapped Orbits')
plt.plot(delayindex,PEs2[1:],color=colors[3,:],label='Never Trapped Orbits')
plt.plot(delayindex,PEs3[1:],color=colors[6,:],label='ULR Orbits')
plt.plot(delayindex,PEsin[1:],color='black',label='Sine Wave')

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Delay Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Permutation Entropy',fontsize=9)
plt.xlim(1,250)
plt.ylim(0,1.0)
plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)

savefilename='PE_fulldat.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')



ndim=5
#Plot C vs H as a CHplane
Cminx, Cminy, Cmaxx, Cmaxy = cpl.Cmaxmin(200,ndim)
plt.rc('lines',markersize=5,markeredgewidth=0.0)

fig=plt.figure(num=3,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.25  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.97      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


plt.plot(Cminx,Cminy,'k-',Cmaxx,Cmaxy,'k-')
plt.plot(PEs1[81],SCs1[81],color=colors[1,:],marker='o',label='Trapped Orbits, Delay 81')
plt.plot(PEs2[81],SCs2[81],color=colors[3,:],marker='o',label='Never Trapped Orbits, Delay 81')
plt.plot(PEs3[81],SCs3[81],color=colors[6,:],marker='o',label='ULR Orbits, Delay 81')
plt.plot(PEsin[81],SCsin[81],color='black',marker='o',label='Sine Wave, Delay 81')
plt.xlabel("Permutation Entropy", fontsize=9)
plt.ylabel("Complexity", fontsize=9)
plt.axis([0,1.0,0,0.45])
plt.xticks(np.arange(0,1.1,0.1),fontsize=9)
plt.yticks(np.arange(0,0.45,0.05),fontsize=9)
plt.legend(loc='lower center',fontsize=5,frameon=False,handlelength=0)

savefilename='CH_fulldat.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')