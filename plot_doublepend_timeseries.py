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
fileheader = 'DoubPen_LsMsEq1_9p8_ICP1'
npz='.npz'
datafile = loadnpzfile(datadir+fileheader+npz)
#initial_theta1s=datafile['initial_theta1s']
#theta1s=datafile['theta1s']
#num_thetas = len(initial_theta1s)

x1=datafile['x1']
x2=datafile['x2']
x3=datafile['x3']
x4=datafile['x4']
y1=datafile['y1']
y2=datafile['y2']
y3=datafile['y3']
y4=datafile['y4']
embeddelay = 5
nfac = factorial(embeddelay)

tmax, dt = 100, 0.001
t = np.arange(0, tmax+dt, dt)

###Storage Arrays###
#delta_t = 1.0
#delays = np.arange(1,101) #248 elements
#taus = delays*delta_t
#freq = 1.0/taus
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
fig=plt.figure(num=1,figsize=(6,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.2  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
plt.plot(t,x1,color=colors[1,:],label='Periodic')

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Time [s]',fontsize=9)
plt.ylabel('Mass 1 x-Position')
plt.xlim(0,7.0)
plt.title('Periodic - Low Energy')
#plt.ylim(0,0.4)
#plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)


savefilename='Timeseries_ICP1.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')



fig=plt.figure(num=12,figsize=(6,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.2  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
lyax=np.log(x3-x1)
plt.plot(t,lyax,color=colors[1,:])

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Time [s]',fontsize=9)
plt.ylabel('Log (Distance)')
plt.xlim(0,7.0)
plt.title('Periodic - Low Energy')
#plt.ylim(0,0.4)
#plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)

idx = np.isfinite(t[0:7000]) & np.isfinite(lyax[0:7000])#clean up NaNs in lyax array
z=np.polyfit(t[idx],lyax[idx],1)
plt.plot(t[idx],z[1]+t[idx]*z[0],color='red',label='lyapunov exponent = '+str(round(z[0],4)))
plt.legend(loc='upper left',fontsize=5,frameon=False,handlelength=5)



savefilename='LogDistance_Lyapanov_ICP1.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')









fileheader = 'DoubPen_LsMsEq1_9p8_ICQ1'
npz='.npz'
datafile = loadnpzfile(datadir+fileheader+npz)
#initial_theta1s=datafile['initial_theta1s']
#theta1s=datafile['theta1s']
#num_thetas = len(initial_theta1s)

x1=datafile['x1']
x2=datafile['x2']
x3=datafile['x3']
x4=datafile['x4']
y1=datafile['y1']
y2=datafile['y2']
y3=datafile['y3']
y4=datafile['y4']
embeddelay = 5
nfac = factorial(embeddelay)

tmax, dt = 100, 0.001
t = np.arange(0, tmax+dt, dt)

###Storage Arrays###
#delta_t = 1.0
#delays = np.arange(1,101) #248 elements
#taus = delays*delta_t
#freq = 1.0/taus
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
fig=plt.figure(num=2,figsize=(6,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.2  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
plt.plot(t,x1,color=colors[1,:])

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Time [s]',fontsize=9)
plt.ylabel('Mass 1 x-Position')
plt.xlim(0,50)
plt.title('Quasi-Periodic - Medium Energy')
#plt.ylim(0,0.4)
#plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)


savefilename='Timeseries_ICQ1.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')




fig=plt.figure(num=22,figsize=(6,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.2  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
lyax=np.log(x3-x1)
plt.plot(t,lyax,color=colors[1,:])

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Time [s]',fontsize=9)
plt.ylabel('Log (Distance)')
plt.xlim(0,50)
plt.title('Quasi-Periodic - Medium Energy')
#plt.ylim(0,0.4)
#plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)

idx = np.isfinite(t[0:50000]) & np.isfinite(lyax[0:50000])#clean up NaNs in lyax array
z=np.polyfit(t[idx],lyax[idx],1)
plt.plot(t[idx],z[1]+t[idx]*z[0],color='red',label='lyapunov exponent = '+str(round(z[0],4)))
plt.legend(loc='upper left',fontsize=5,frameon=False,handlelength=5)


savefilename='LogDistance_Lyapanov_ICQ1.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')








fileheader = 'DoubPen_LsMsEq1_9p8_ICC1'
npz='.npz'
datafile = loadnpzfile(datadir+fileheader+npz)
#initial_theta1s=datafile['initial_theta1s']
#theta1s=datafile['theta1s']
#num_thetas = len(initial_theta1s)

x1=datafile['x1']
x2=datafile['x2']
x3=datafile['x3']
x4=datafile['x4']
y1=datafile['y1']
y2=datafile['y2']
y3=datafile['y3']
y4=datafile['y4']
embeddelay = 5
nfac = factorial(embeddelay)

tmax, dt = 100, 0.001
t = np.arange(0, tmax+dt, dt)

###Storage Arrays###
#delta_t = 1.0
#delays = np.arange(1,101) #248 elements
#taus = delays*delta_t
#freq = 1.0/taus
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
fig=plt.figure(num=3,figsize=(6,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.2  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
plt.plot(t,x1,color=colors[1,:])

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Time [s]',fontsize=9)
plt.ylabel('Mass 1 x-Position')
plt.xlim(0,50)
plt.title('Chaotic - High Energy')
#plt.ylim(0,0.4)
#plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)


savefilename='Timeseries_ICC1.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')




fig=plt.figure(num=32,figsize=(6,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.2  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
lyax=np.log(x3-x1)
plt.plot(t,lyax,color=colors[1,:])

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Time [s]',fontsize=9)
plt.ylabel('Log (Distance)')
plt.xlim(0,50)
plt.title('Chaotic - High Energy')
#plt.ylim(0,0.4)
#plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)

idx = np.isfinite(t[0:20000]) & np.isfinite(lyax[0:20000])#clean up NaNs in lyax array
z=np.polyfit(t[idx],lyax[idx],1)
plt.plot(t[idx],z[1]+t[idx]*z[0],color='red',label='lyapunov exponent = '+str(round(z[0],4)))
plt.legend(loc='upper left',fontsize=5,frameon=False,handlelength=5)



savefilename='LogDistance_Lyapanov_ICC1.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')
