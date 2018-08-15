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
delayindex=datafile['delays']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_9p8_ICQ1_embeddelay5_41_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_Q=datafile['PEsx1']
SCsx1_Q=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_9p8_ICC1_embeddelay5_41_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_C=datafile['PEsx1']
SCsx1_C=datafile['SCsx1']

fileheader = 'DoubPen_LsMsEq1_9p8_ICC1'
datafile = loadnpzfile(datadir+fileheader+npz)
x1_9p8=datafile['x1']


fileheader = 'PE_SC_DPDoubPen_LsMsEq1_4p9_ICC1_embeddelay5_41_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_C_4p9=datafile['PEsx1']
SCsx1_C_4p9=datafile['SCsx1']

fileheader = 'DoubPen_LsMsEq1_4p9_ICC1'
datafile = loadnpzfile(datadir+fileheader+npz)
x1_4p9=datafile['x1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_20p0_ICC1_embeddelay5_41_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_C_20p0=datafile['PEsx1']
SCsx1_C_20p0=datafile['SCsx1']

fileheader = 'DoubPen_LsMsEq1_20p0_ICC1'
datafile = loadnpzfile(datadir+fileheader+npz)
x1_20p0=datafile['x1']

gravscan_PEs=np.zeros([20,999])
gravscan_SCs=np.zeros([20,999])

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav1_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
gravscan_PEs[0,:]=datafile['PEsx1']
gravscan_SCs[0,:]=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav2_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
gravscan_PEs[1,:]=datafile['PEsx1']
gravscan_SCs[1,:]=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav3_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
gravscan_PEs[2,:]=datafile['PEsx1']
gravscan_SCs[2,:]=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav4_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
gravscan_PEs[3,:]=datafile['PEsx1']
gravscan_SCs[3,:]=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav5_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
gravscan_PEs[4,:]=datafile['PEsx1']
gravscan_SCs[4,:]=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav6_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
gravscan_PEs[5,:]=datafile['PEsx1']
gravscan_SCs[5,:]=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav7_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
gravscan_PEs[6,:]=datafile['PEsx1']
gravscan_SCs[6,:]=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav8_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
gravscan_PEs[7,:]=datafile['PEsx1']
gravscan_SCs[7,:]=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav9_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
gravscan_PEs[8,:]=datafile['PEsx1']
gravscan_SCs[8,:]=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav10_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
gravscan_PEs[9,:]=datafile['PEsx1']
gravscan_SCs[9,:]=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav11_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
gravscan_PEs[10,:]=datafile['PEsx1']
gravscan_SCs[10,:]=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav12_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
gravscan_PEs[11,:]=datafile['PEsx1']
gravscan_SCs[11,:]=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav13_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
gravscan_PEs[12,:]=datafile['PEsx1']
gravscan_SCs[12,:]=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav14_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
gravscan_PEs[13,:]=datafile['PEsx1']
gravscan_SCs[13,:]=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav15_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
gravscan_PEs[14,:]=datafile['PEsx1']
gravscan_SCs[14,:]=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav16_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
gravscan_PEs[15,:]=datafile['PEsx1']
gravscan_SCs[15,:]=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav17_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
gravscan_PEs[16,:]=datafile['PEsx1']
gravscan_SCs[16,:]=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav18_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
gravscan_PEs[17,:]=datafile['PEsx1']
gravscan_SCs[17,:]=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav19_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
gravscan_PEs[18,:]=datafile['PEsx1']
gravscan_SCs[18,:]=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav20_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
gravscan_PEs[19,:]=datafile['PEsx1']
gravscan_SCs[19,:]=datafile['SCsx1']
delayindex=datafile['delays']
#import spectrum_wwind as spec

#freq, freq2, comp, pwr9p8, mag, phase2, cos_phase, dt = spec.spectrum_wwind(x1_9p8,np.arange(1,100001),window='hanning')
#freq, freq2, comp, pwr4p9, mag, phase2, cos_phase, dt = spec.spectrum_wwind(x1_4p9,np.arange(1,100001),window='hanning')
#freq, freq2, comp, pwr20p0, mag, phase2, cos_phase, dt = spec.spectrum_wwind(x1_20p0,np.arange(1,100001),window='hanning')




tmax, dt = 100, 0.001
t = np.arange(0, tmax+dt, dt)

import matplotlib.cm as cm
colors = np.zeros([20,4])
for i in np.arange(20):
    c = cm.spectral(i/20.,1)
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
plt.semilogx(delayindex,gravscan_SCs[0,:],color=colors[0,:],label='g=1')
plt.semilogx(delayindex,gravscan_SCs[1,:],color=colors[1,:],label='g=2')
plt.semilogx(delayindex,gravscan_SCs[2,:],color=colors[2,:],label='g=3')
plt.semilogx(delayindex,gravscan_SCs[3,:],color=colors[3,:],label='g=4')
plt.semilogx(delayindex,gravscan_SCs[4,:],color=colors[4,:],label='g=5')
plt.semilogx(delayindex,gravscan_SCs[5,:],color=colors[5,:],label='g=6')
plt.semilogx(delayindex,gravscan_SCs[6,:],color=colors[6,:],label='g=7')
plt.semilogx(delayindex,gravscan_SCs[7,:],color=colors[7,:],label='g=8')
plt.semilogx(delayindex,gravscan_SCs[8,:],color=colors[8,:],label='g=9')
plt.semilogx(delayindex,gravscan_SCs[9,:],color=colors[9,:],label='g=10')
plt.semilogx(delayindex,gravscan_SCs[10,:],color=colors[10,:],label='g=11')
plt.semilogx(delayindex,gravscan_SCs[11,:],color=colors[11,:],label='g=12')
plt.semilogx(delayindex,gravscan_SCs[12,:],color=colors[12,:],label='g=13')
plt.semilogx(delayindex,gravscan_SCs[13,:],color=colors[13,:],label='g=14')
plt.semilogx(delayindex,gravscan_SCs[14,:],color=colors[14,:],label='g=15')
plt.semilogx(delayindex,gravscan_SCs[15,:],color=colors[15,:],label='g=16')
plt.semilogx(delayindex,gravscan_SCs[16,:],color=colors[16,:],label='g=17')
plt.semilogx(delayindex,gravscan_SCs[17,:],color=colors[17,:],label='g=18')
plt.semilogx(delayindex,gravscan_SCs[18,:],color=colors[18,:],label='g=19')
plt.semilogx(delayindex,gravscan_SCs[19,:],color=colors[19,:],label='g=20')


#plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Delay Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Statistical Complexity',fontsize=9)
#plt.xlim(1,250)
plt.ylim(0,0.5)
plt.legend(loc='upper left',fontsize=5,frameon=False,handlelength=5)

savefilename='SC_DPx1_ICC1_g1to20scan.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')



fig=plt.figure(num=11,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.2  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.97      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
ax1=plt.subplot(1,1,1)

gs = np.arange(1,21,1)
plt.contourf(gs,np.log10(delayindex),(np.transpose(gravscan_SCs)),256,cmap='jet',vmin=0.1,vmax=0.35)
plt.ylim(0,np.log10(300))
plt.yticks(fontsize=9)
plt.ylabel('Embedding Delay',fontsize=9)
plt.xlim(1,20)
plt.xticks(gs,list(gs),fontsize=5)
plt.xlabel(r'Gravity $[m/s^{2}]$')




"""


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
"""