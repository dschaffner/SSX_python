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
import get_corr as gc

#calc_PESC_solarwind_chen.py
datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\DoublePendulum\\'
datadir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Galatic Dynamics Data\\DoublePendulum\\'
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

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC0_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang0=datafile['PEsx1']
SCsx1_icang0=datafile['SCsx1']

"""
gs = np.arange(1,21,1)
for gravity in gs:
    
    fileheader = 'DoubPen_LsMsEq1_grav'+str(gravity)+'_ICC1'
    datafile = loadnpzfile(datadir+fileheader+npz)
    x1=datafile['x1']
    x2=datafile['x2']
    
    
    plt.rc('lines',markersize=1.5,markeredgewidth=0.0)
    fig=plt.figure(num=gravity,figsize=(6,3.5),dpi=300,facecolor='w',edgecolor='k')
    left  = 0.2  # the left side of the subplots of the figure
    right = 0.94    # the right side of the subplots of the figure
    bottom = 0.17  # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.2   # the amount of width reserved for blank space between subplots
    hspace = 0.1   # the amount of height reserved for white space between subplots
    plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    
    
    ax1=plt.subplot(1,1,1)
    plt.plot(t,x1,color='blue',label='Gravity = '+str(gravity)+r'$m/s^2$')
    
    #plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)
    
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.xlabel('Time [s]',fontsize=9)
    plt.ylabel('Mass 1 x-Position')
    plt.xlim(0,50.0)
    plt.title('Gravity Scan Chaotic - '+str(gravity)+r'$m/s^2$')
    #plt.ylim(0,0.4)
    #plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)
    
    
    savefilename='Gravity_ScanTimeseries_ICC1_gravity'+str(gravity)+'.png'
    savefile = os.path.normpath(datadir+savefilename)
    plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')
    plt.clf()
"""
###Storage Arrays###
#delta_t = 1.0
#delays = np.arange(1,101) #248 elements
#taus = delays*delta_t
#freq = 1.0/taus
import matplotlib.cm as cm
#colors = np.zeros([7,4])
#for i in np.arange(7):
#    c = cm.spectral(i/7.,1)
#    colors[i,:]=c
points = ['o','v','s','p','*','h','^','D','+','>','H','d','x','<']
        
plt.rc('axes',linewidth=2.0)
plt.rc('xtick.major',width=2.0)
plt.rc('ytick.major',width=2.0)
plt.rc('xtick.minor',width=2.0)
plt.rc('ytick.minor',width=2.0)
plt.rc('lines',markersize=8,markeredgewidth=0.0,linewidth=2.0)
#plt.rcParams['ps.fonttype'] = 42
#plt.rcParams['pdf.fonttype'] = 42
fig=plt.figure(num=1,figsize=(9,7),dpi=300,facecolor='w',edgecolor='k')
left  = 0.1  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.08  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.3   # the amount of width reserved for blank space between subplots
hspace = 0.0   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(3,2,1)
ax1.tick_params(axis='x',direction='inout',top=True)
ax1.tick_params(axis='y',direction='inout',top=True)

fileheader = 'DoubPen_LsMsEq1_9p8_ICP1'
npz='.npz'
datafile = loadnpzfile(datadir+fileheader+npz)
x1=datafile['x1']
x3=datafile['x3']
plt.plot(t,x1,color='orange',linestyle='dotted',linewidth=0.75)
plt.plot(t,x3,color='blue',linewidth=2.0)
ax1.set_xticklabels([])
#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time [s]',fontsize=15)
plt.ylabel('Mass 1 x(t)',fontsize=15)
plt.xlim(0,30.0)
plt.ylim(-1.1,1.1)
#plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)


#savefilename='Timeseries_ICP1.png'
#savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')


ax1=plt.subplot(3,2,2)
lyax=np.log(np.abs(x3-x1)/1e-9)
plt.plot(t,lyax,color='red')#colors[1,:])

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time [s]',fontsize=15)
plt.ylabel(r'Log $\frac{|\Delta x|}{\Delta x_{0}}$',fontsize=12)
plt.xlim(0,100.0)
ax1.set_xticklabels([])
#plt.ylim(0,0.4)
#plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)

fitlength=100000
fittime=fitlength*dt
#idx = np.isfinite(t[0:fitlength]) & np.isfinite(lyax[0:fitlength])#clean up NaNs in lyax array
z=np.polyfit(t,lyax,1)
plt.plot(t,z[1]+t*z[0],color='red',label='lyapunov exponent = '+str(round(z[0],4)/fittime))
plt.legend(loc='upper right',fontsize=10,frameon=False,handlelength=5)
plt.ylim(-9,22)
ax1.set_xticklabels([])

ax1=plt.subplot(3,2,3)
fileheader = 'DoubPen_LsMsEq1_9p8_ICQ1'
npz='.npz'
datafile = loadnpzfile(datadir+fileheader+npz)
x1=datafile['x1']
x3=datafile['x3']
plt.plot(t,x1,color='orange',linestyle='dotted',linewidth=0.75)
plt.plot(t,x3,color='blue',linewidth=2.0)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time [s]',fontsize=15)
plt.ylabel('Mass 1 x(t)',fontsize=15)
plt.xlim(0,30.0)
plt.ylim(-1.1,1.1)
ax1.set_xticklabels([])

ax1=plt.subplot(3,2,4)
lyax=np.log(np.abs(x3-x1)/1e-9)
plt.plot(t,lyax,color='red')#colors[1,:])

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time [s]',fontsize=15)
plt.ylabel(r'Log $\frac{|\Delta x|}{\Delta x_{0}}$',fontsize=12)
plt.xlim(0,100.0)
ax1.set_xticklabels([])
fitlength=100000
fittime=fitlength*dt
#idx = np.isfinite(t[0:fitlength]) & np.isfinite(lyax[0:fitlength])#clean up NaNs in lyax array
z=np.polyfit(t,lyax,1)
plt.plot(t,z[1]+t*z[0],color='red',label='lyapunov exponent = '+str(round(z[0],4)/fittime))
plt.legend(loc='lower right',fontsize=10,frameon=False,handlelength=5)
plt.ylim(-9,22)

ax1=plt.subplot(3,2,5)
fileheader = 'DoubPen_LsMsEq1_9p8_ICC1'
npz='.npz'
datafile = loadnpzfile(datadir+fileheader+npz)
x1=datafile['x1']
x3=datafile['x3']
plt.plot(t,x1,color='orange',linestyle='solid',linewidth=0.75)
plt.plot(t,x3,color='blue',linewidth=2.0)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time [s]',fontsize=15)
plt.ylabel('Mass 1 x(t)',fontsize=15)
plt.xlim(0,30.0)
plt.ylim(-1.1,1.1)

ax1=plt.subplot(3,2,6)
lyax=np.log(np.abs(x3-x1)/1e-9)
plt.plot(t,lyax,color='red')#colors[1,:])

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Time [s]',fontsize=15)
plt.ylabel(r'Log $\frac{|\Delta x|}{\Delta x_{0}}$',fontsize=12)
plt.xlim(0,100.0)

fitlength=20000
fittime=fitlength*dt
#idx = np.isfinite(t[0:fitlength]) & np.isfinite(lyax[0:fitlength])#clean up NaNs in lyax array
z=np.polyfit(t[0:fitlength],lyax[0:fitlength],1)
plt.plot(t[0:fitlength],z[1]+t[0:fitlength]*z[0],color='red',label='lyapunov exponent = '+str(round(z[0],4)/fittime))
plt.legend(loc='lower right',fontsize=10,frameon=False,handlelength=5)
plt.ylim(-9,22)

savefilename='Timeseries_withLypExp_forpaper_3.eps'
savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
#plt.clf()
#plt.close()

"""
fileheader = 'DoubPen_LsMsEq1_9p8_ICP1'
fileheader = 'DoubPen_LsMsEq1_9p8_ICQ1'
#fileheader = 'DoubPen_LsMsEq1_9p8_ICC1'
#fileheader = 'DoubPen_LsEq0p1_MsEq1_grav9p8_ICC1'
#fileheader = 'DoubPen_LsEq0p5_MsEq1_grav9p8_ICC1'
npz='.npz'
datafile = loadnpzfile(datadir+fileheader+npz)
x1=datafile['x1']
x3=datafile['x3']

tau,corr = gc.get_corr(t,x1,x1,normalized=False)
plt.plot(tau,corr)
#Periodic T=2.63s
#Quasi-Periodic Dominant Period: T=2.99s
#Chaos First Period: T=1.73s
"""
"""






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
"""