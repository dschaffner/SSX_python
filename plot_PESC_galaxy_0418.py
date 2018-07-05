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

datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\Data040318\\'
fileheader = 'Data_0418_type1_249_delays'
npy='.npz'

delayindex = np.arange(1,250)

datafile = loadnpzfile(datadir+fileheader+npy)
PEs1 = datafile['PEs']
SCs1 = datafile['SCs']

fileheader = 'Data_0418_type2_249_delays'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs2 = datafile['PEs']
SCs2 = datafile['SCs']

fileheader = 'Data_0418_type31_249_delays'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs31 = datafile['PEs']
SCs31 = datafile['SCs']

fileheader = 'Data_0418_type32_249_delays'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32 = datafile['PEs']
SCs32 = datafile['SCs']

fileheader = 'Data_0418_type4_249_delays'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs4 = datafile['PEs']
SCs4 = datafile['SCs']

fileheader = 'Data_sine100period_ranphasestart_249_delays'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs100 = datafile['PEs']
SCs100 = datafile['SCs']

fileheader = 'Data_sine200period_ranphasestart_249_delays'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs200 = datafile['PEs']
SCs200 = datafile['SCs']

fileheader = 'Data_sine300period_ranphasestart_249_delays'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs300 = datafile['PEs']
SCs300 = datafile['SCs']

fileheader = 'Data_sine400period_ranphasestart_249_delays'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs400 = datafile['PEs']
SCs400 = datafile['SCs']

fileheader = 'Data_sine500period_ranphasestart_249_delays'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs500 = datafile['PEs']
SCs500 = datafile['SCs']

fileheader = 'Data_sine600period_ranphasestart_249_delays'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs600 = datafile['PEs']
SCs600 = datafile['SCs']

fileheader = 'Data_sine700period_ranphasestart_249_delays'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs700 = datafile['PEs']
SCs700 = datafile['SCs']

#fileheader = 'PE_SC_sinewave_249_delays'
#datafile = loadnpzfile(datadir+fileheader+npy)
#PEsin = datafile['PEs']
#SCsin = datafile['SCs']

colors = np.zeros([8,4])
for i in np.arange(8):
    c = cm.spectral(i/8.,1)
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
plt.plot(delayindex,SCs1[1:],color=colors[1,:],label='Type 1')
plt.plot(delayindex,SCs2[1:],color=colors[2,:],label='Type 2')
plt.plot(delayindex,SCs31[1:],color=colors[3,:],label='Type 3-1')
plt.plot(delayindex,SCs32[1:],color=colors[4,:],label='Type 3-2')
plt.plot(delayindex,SCs4[1:],color=colors[5,:],label='Type 4')
#plt.plot(delayindex,SCsin[1:],color='black',label='Sine Wave')

plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Delay Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Complexity',fontsize=9)
plt.xlim(1,250)
plt.ylim(0,0.5)
plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)


savefilename='SC_0418_1000timesteps.png'
savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')


fig=plt.figure(num=2,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.2  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.97      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
plt.plot(delayindex,PEs1[1:],color=colors[1,:],label='Type 1')
plt.plot(delayindex,PEs2[1:],color=colors[2,:],label='Type 2')
plt.plot(delayindex,PEs31[1:],color=colors[3,:],label='Type 3-1')
plt.plot(delayindex,PEs32[1:],color=colors[4,:],label='Type 3-2')
plt.plot(delayindex,PEs4[1:],color=colors[5,:],label='Type 4')
#plt.plot(delayindex,PEsin[1:],color='black',label='Sine Wave')

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Delay Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Permutation Entropy',fontsize=9)
plt.xlim(1,250)
plt.ylim(0,1.0)
plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)

savefilename='PE_0418_1000timesteps.png'
savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')



ndim=5
#Plot C vs H as a CHplane
Cminx, Cminy, Cmaxx, Cmaxy = cpl.Cmaxmin(200,ndim)
plt.rc('lines',markersize=5,markeredgewidth=0.0)

fig=plt.figure(num=3,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.25  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.90      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

timestep = 40

plt.plot(Cminx,Cminy,'k-',Cmaxx,Cmaxy,'k-')
plt.plot(PEs1[timestep],SCs1[timestep],color=colors[1,:],marker='o',label='Type 1')
plt.plot(PEs2[timestep],SCs2[timestep],color=colors[2,:],marker='o',label='Type 2')
plt.plot(PEs31[timestep],SCs31[timestep],color=colors[3,:],marker='o',label='Type 3-1')
plt.plot(PEs32[timestep],SCs32[timestep],color=colors[4,:],marker='o',label='Type 3-2')
plt.plot(PEs4[timestep],SCs4[timestep],color=colors[5,:],marker='o',label='Type 4')
#plt.plot(PEsin[81],SCsin[81],color='black',marker='o',label='Sine Wave, Delay 81')
plt.xlabel("Permutation Entropy", fontsize=9)
plt.ylabel("Complexity", fontsize=9)
plt.title('Timestep '+str(timestep))
plt.axis([0,1.0,0,0.45])
plt.xticks(np.arange(0,1.1,0.1),fontsize=9)
plt.yticks(np.arange(0,0.45,0.05),fontsize=9)
plt.legend(loc='lower center',fontsize=5,frameon=False,handlelength=0)

savefilename='CH_0418_1000timesteps_timestep'+str(timestep)+'.png'
savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')




fig=plt.figure(num=4,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.2  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.97      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
plt.plot(delayindex,SCs100[1:],color=colors[1,:],label='Period 100 Steps')
plt.plot(delayindex,SCs200[1:],color=colors[2,:],label='Period 200 Steps')
plt.plot(delayindex,SCs300[1:],color=colors[3,:],label='Period 300 Steps')
plt.plot(delayindex,SCs400[1:],color=colors[4,:],label='Period 400 Steps')
plt.plot(delayindex,SCs500[1:],color=colors[5,:],label='Period 500 Steps')
plt.plot(delayindex,SCs600[1:],color=colors[6,:],label='Period 600 Steps')
plt.plot(delayindex,SCs700[1:],color=colors[7,:],label='Period 700 Steps')
#plt.plot(delayindex,PEsin[1:],color='black',label='Sine Wave')

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Delay Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Complexity',fontsize=9)
plt.xlim(1,250)
#plt.ylim(0,1.0)
plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)

savefilename='SC_sines_with_difperiods_1000timesteps.png'
savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')
