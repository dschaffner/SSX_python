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

datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Corrsin Wind Tunnel Data\\NPZ_files\\m20_5mm\\streamwise\\'
fileheader = 'PE_SC_m20_5mm_embed5'
npy='.npz'

delta_t = 2.5e-5
delayindex = np.arange(1,248)*delta_t


datafile = loadnpzfile(datadir+fileheader+npy)
PEs1 = datafile['PEs']
SCs1 = datafile['SCs']
PEs1 = np.mean(PEs1,axis=1)
SCs1 = np.mean(SCs1,axis=1)

datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Corrsin Wind Tunnel Data\\NPZ_files\\m50_5mm\\streamwise\\'
fileheader = 'PE_SC_m50_5mm_embed6'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs2 = datafile['PEs']
SCs2 = datafile['SCs']
PEs2 = np.mean(PEs2,axis=1)
SCs2 = np.mean(SCs2,axis=1)

taus = datafile['taus']

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
bottom = 0.2  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(2,1,2)
plt.semilogx(taus,SCs1,color=colors[1,:],label='20m 5mm')
#plt.semilogx(taus,SCs2,color=colors[3,:],label='50m 5mm')

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel(r'$\tau$[s]',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Complexity',fontsize=9)
#plt.xlim(1,250)
#plt.ylim(0,0.5)
plt.grid(b=True,which='both',axis='x',linestyle='dotted',linewidth=0.05)
plt.legend(loc='upper right',fontsize=5,frameon=False,handlelength=5)


#savefilename='SC_fluid_streamwise_embed5.png'
#savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')


#fig=plt.figure(num=2,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
#left  = 0.2  # the left side of the subplots of the figure
#right = 0.94    # the right side of the subplots of the figure
#bottom = 0.2  # the bottom of the subplots of the figure
#top = 0.97      # the top of the subplots of the figure
#wspace = 0.2   # the amount of width reserved for blank space between subplots
#hspace = 0.1   # the amount of height reserved for white space between subplots
#plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(2,1,1)
plt.semilogx(taus,PEs1,color=colors[1,:],label='20m 5mm')
#plt.semilogx(taus,PEs2,color=colors[3,:],label='50m 5mm')

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
#plt.xlabel(r'$\tau$[s]',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Perm. Entropy',fontsize=9)
ax1.set_xticklabels([])
#plt.xlim(1,250)
#plt.ylim(0,1.0)
plt.grid(b=True,which='both',axis='x',linestyle='dotted',linewidth=0.05)
plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)

savefilename='PEandSC_fluid_streamwise_embed5.png.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')

"""

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
"""