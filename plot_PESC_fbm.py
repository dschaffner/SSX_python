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
datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\fbm\\'
fileheader = 'PE_SC_fbm_H0.5_N20000_1_embeddelay5_999_delays'
fileheader = 'PE_SC_fbm_H0.9_N20000_4_embeddelay5_999_delays'
fileheader = 'PE_SC_fbm_H0.1_N20000_3_embeddelay5_999_delays'
npz='.npz'
datafile = loadnpzfile(datadir+fileheader+npz)
#initial_theta1s=datafile['initial_theta1s']
#theta1s=datafile['theta1s']
#num_thetas = len(initial_theta1s)

PEs=datafile['PEs']
SCs=datafile['SCs']
delayindex=datafile['delays']

"""



tmax, dt = 100, 0.001
t = np.arange(0, tmax+dt, dt)

import matplotlib.cm as cm
colors = np.zeros([7,4])
for i in np.arange(7):
    c = cm.spectral(i/7.,1)
    colors[i,:]=c
points = ['o','v','s','p','*','h','^','D','+','>','H','d','x','<']
        
plt.rc('axes',linewidth=2.0)
plt.rc('xtick.major',width=2.0)
plt.rc('ytick.major',width=2.0)
plt.rc('xtick.minor',width=2.0)
plt.rc('ytick.minor',width=2.0)
plt.rc('lines',markersize=8,markeredgewidth=0.0,linewidth=2.0)

fig=plt.figure(num=1,figsize=(7,9),dpi=600,facecolor='w',edgecolor='k')
left  = 0.16  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.08  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.0   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

timeindex=delayindex*dt
ax1=plt.subplot(2,1,1)
ax1.tick_params(axis='x',direction='inout',top=True)
ax1.tick_params(axis='y',direction='inout',top=True)

plt.semilogx(timeindex,SCsx1_P,color=colors[1,:],marker=points[0],markevery=(5),label='Periodic')
plt.semilogx(timeindex,SCsx1_Q,color=colors[3,:],marker=points[1],markevery=(5),label='Quasi-Periodic')
plt.semilogx(timeindex,SCsx1_C,color=colors[6,:],marker=points[2],markevery=(5),label='Chaotic')


#plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.yticks(fontsize=10)
plt.xlabel(r'$\tau_s [s]$',fontsize=15)
ax1.set_xticklabels([])
plt.ylabel(r'$C$',fontsize=15)
#plt.xlim(1,250)
plt.ylim(0.0,0.45)
#plt.legend(loc='best',fontsize=8,frameon=False,handlelength=5)
plt.text(0.04,0.95,'(a)',fontsize=16,horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes)


ax1=plt.subplot(2,1,2)
plt.semilogx(timeindex,PEsx1_P,color=colors[1,:],marker=points[0],markevery=(5),label='Periodic')
plt.semilogx(timeindex,PEsx1_Q,color=colors[3,:],marker=points[1],markevery=(5),label='Quasi-Periodic')
plt.semilogx(timeindex,PEsx1_C,color=colors[6,:],marker=points[2],markevery=(5),label='Chaotic')

plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
plt.xlabel(r'$\tau_s [s]$',fontsize=15)
#ax1.set_xticklabels([])
plt.ylabel(r'$H$',fontsize=15)
#plt.xlim(1,250)
plt.ylim(0,0.985)
plt.legend(loc='center left',fontsize=9,frameon=False,handlelength=5)
plt.text(0.04,0.95,'(b)',fontsize=16,horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes)


savefilename='SC_and_PE_DPx1_ICP1_ICQ1_ICC1.eps'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()
"""