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

classtype = 'Class_1'

datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\'
fileheader = 'PE_SC_New_DavidData_'+classtype+'_249_delays_500_timesteps'
npy='.npz'

delayindex = np.arange(1,250)

datafile = loadnpzfile(datadir+fileheader+npy)
PEs500 = datafile['PEs']
SCs500 = datafile['SCs']

fileheader = 'PE_SC_New_DavidData_'+classtype+'_249_delays_1000_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1000 = datafile['PEs']
SCs1000 = datafile['SCs']

fileheader = 'PE_SC_New_DavidData_'+classtype+'_249_delays_2000_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs2000 = datafile['PEs']
SCs2000 = datafile['SCs']

fileheader = 'PE_SC_New_DavidData_'+classtype+'_249_delays_750_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs750 = datafile['PEs']
SCs750 = datafile['SCs']

fileheader = 'PE_SC_New_DavidData_'+classtype+'_249_delays_850_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs850 = datafile['PEs']
SCs850 = datafile['SCs']

fileheader = 'PE_SC_New_DavidData_'+classtype+'_249_delays_950_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs950 = datafile['PEs']
SCs950 = datafile['SCs']

fileheader = 'PE_SC_New_DavidData_'+classtype+'_249_delays_1150_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1150 = datafile['PEs']
SCs1150 = datafile['SCs']

fileheader = 'PE_SC_New_DavidData_'+classtype+'_249_delays_1250_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1250 = datafile['PEs']
SCs1250 = datafile['SCs']

fileheader = 'PE_SC_New_DavidData_'+classtype+'_249_delays_1500_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1500 = datafile['PEs']
SCs1500 = datafile['SCs']

fileheader = 'PE_SC_New_DavidData_'+classtype+'_249_delays_2500_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs2500 = datafile['PEs']
SCs2500 = datafile['SCs']

fileheader = 'PE_SC_New_DavidData_'+classtype+'_249_delays_5000_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs5000 = datafile['PEs']
SCs5000 = datafile['SCs']

fileheader = 'PE_SC_New_DavidData_'+classtype+'_249_delays_10000_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs10000 = datafile['PEs']
SCs10000 = datafile['SCs']


fileheader = 'PE_SC_DavidData_Class_3_249_delays_500_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs500old = datafile['PEs']
SCs500old = datafile['SCs']

fileheader = 'PE_SC_DavidData_Class_3_249_delays_550_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs550old = datafile['PEs']
SCs550old = datafile['SCs']

fileheader = 'PE_SC_DavidData_Class_3_249_delays_600_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs600old = datafile['PEs']
SCs600old = datafile['SCs']

fileheader = 'PE_SC_DavidData_Class_3_249_delays_650_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs650old = datafile['PEs']
SCs650old = datafile['SCs']

fileheader = 'PE_SC_DavidData_Class_3_249_delays_700_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs700old = datafile['PEs']
SCs700old = datafile['SCs']

fileheader = 'PE_SC_DavidData_Class_3_249_delays_750_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs750old = datafile['PEs']
SCs750old = datafile['SCs']

fileheader = 'PE_SC_DavidData_Class_3_249_delays_800_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs800old = datafile['PEs']
SCs800old = datafile['SCs']

fileheader = 'PE_SC_DavidData_Class_3_249_delays_850_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs850old = datafile['PEs']
SCs850old = datafile['SCs']

fileheader = 'PE_SC_DavidData_Class_3_249_delays_900_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs900old = datafile['PEs']
SCs900old = datafile['SCs']

fileheader = 'PE_SC_DavidData_Class_3_249_delays_950_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs950old = datafile['PEs']
SCs950old = datafile['SCs']

colors = np.zeros([9,4])
for i in np.arange(9):
    c = cm.spectral(i/9.,1)
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
left  = 0.16  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.2  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
plt.plot(delayindex[0:123],SCs500[1:124],color=colors[1,:],label='500 timesteps')
plt.plot(delayindex[0:186],SCs750[1:187],color=colors[2,:],label='750 timesteps')
plt.plot(delayindex,SCs1000[1:],color=colors[3,:],label='1000 timesteps')
plt.plot(delayindex,SCs1500[1:],color=colors[4,:],label='1500 timesteps')
plt.plot(delayindex,SCs2000[1:],color=colors[5,:],label='2000 timesteps')
plt.plot(delayindex,SCs2500[1:],color=colors[6,:],label='2500 timesteps')
plt.plot(delayindex,SCs5000[1:],color=colors[7,:],label='5000 timesteps')
plt.plot(delayindex,SCs10000[1:],color=colors[8,:],label='10000 timesteps')

plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.yticks(np.array([0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5]),[0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5],fontsize=9)
plt.xlabel('Delay Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Statistical Complexity',fontsize=9)
plt.xlim(1,250)
plt.ylim(0,0.4)
plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)


savefilename='SC_recordlengthvariation_'+classtype+'.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')

fig=plt.figure(num=11,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.16  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.2  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
plt.plot(delayindex[0:123],SCs500[1:124],color=colors[1,:],label='500 timesteps')
plt.plot(delayindex[0:186],SCs750[1:187],color=colors[2,:],label='750 timesteps')
plt.plot(delayindex[0:209],SCs850[1:210],color=colors[3,:],label='850 timesteps')
plt.plot(delayindex[0:209],SCs950[1:210],color=colors[4,:],label='950 timesteps')
plt.plot(delayindex,SCs1000[1:],color=colors[5,:],label='1000 timesteps')
plt.plot(delayindex,SCs1150[1:],color=colors[6,:],label='1150 timesteps')
plt.plot(delayindex,SCs1250[1:],color=colors[7,:],label='1250 timesteps')

plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.yticks(np.array([0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5,0.6,0.7,1.0]),[0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5,0.6,0.7,1.0],fontsize=9)
plt.xlabel('Delay Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Statistical Complexity',fontsize=9)
plt.xlim(1,250)
plt.ylim(0.2,0.4)
plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)


savefilename='SC_recordlengthvariation_hiressteps_'+classtype+'.png'
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
plt.plot(delayindex[0:123],PEs500[1:124],color=colors[1,:],label='500 timesteps')
plt.plot(delayindex[0:186],PEs750[1:187],color=colors[2,:],label='750 timesteps')
plt.plot(delayindex,PEs1000[1:],color=colors[3,:],label='1000 timesteps')
plt.plot(delayindex,PEs1500[1:],color=colors[4,:],label='1500 timesteps')
plt.plot(delayindex,PEs2000[1:],color=colors[5,:],label='2000 timesteps')
plt.plot(delayindex,PEs2500[1:],color=colors[6,:],label='2500 timesteps')
plt.plot(delayindex,PEs5000[1:],color=colors[7,:],label='5000 timesteps')
plt.plot(delayindex,PEs10000[1:],color=colors[8,:],label='10000 timesteps')


#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Delay Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Permutation Entropy',fontsize=9)
plt.xlim(1,250)
plt.ylim(0,1.0)
plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)

savefilename='PE_recordlengthvariation_'+classtype+'.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')



ndim=5
#Plot C vs H as a CHplane
Cminx, Cminy, Cmaxx, Cmaxy = cpl.Cmaxmin(200,ndim)
plt.rc('lines',markersize=3,markeredgewidth=0.0)

fig=plt.figure(num=3,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.25  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.97      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


plt.plot(Cminx,Cminy,'k-',Cmaxx,Cmaxy,'k-')
plt.plot(PEs10000[1:],SCs10000[1:],color=colors[8,:],marker='o',label='10000 timesteps')
plt.plot(PEs5000[1:],SCs5000[1:],color=colors[7,:],marker='o',label='5000 timesteps')
plt.plot(PEs2500[1:],SCs2500[1:],color=colors[6,:],marker='o',label='2500 timesteps')
plt.plot(PEs2000[1:],SCs2000[1:],color=colors[5,:],marker='o',label='2000 timesteps')
plt.plot(PEs1500[1:],SCs1500[1:],color=colors[4,:],marker='o',label='1500 timesteps')
plt.plot(PEs1000[1:],SCs1000[1:],color=colors[3,:],marker='o',label='1000 timesteps')
plt.plot(PEs750[1:187],SCs750[1:187],color=colors[2,:],marker='o',label='750 timesteps')
plt.plot(PEs500[1:124],SCs500[1:124],color=colors[1,:],marker='o',label='500 timesteps')

plt.xlabel("Permutation Entropy", fontsize=9)
plt.ylabel("Statistical Complexity", fontsize=9)
plt.axis([0,1.0,0,0.45])
plt.xticks(np.arange(0,1.1,0.1),fontsize=9)
plt.yticks(np.arange(0,0.45,0.05),fontsize=9)
plt.legend(loc='lower center',fontsize=5,frameon=False,handlelength=0)

savefilename='CH_recordlengthvariation_'+classtype+'.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')


colors = np.zeros([10,4])
for i in np.arange(10):
    c = cm.spectral(i/10.,1)
    colors[i,:]=c
points = ['o','v','s','p','*','h','^','D','+','>','H','d','x','<']

fig=plt.figure(num=4,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.25  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
plt.plot(delayindex[0:123],SCs500old[1:124],color=colors[0,:],label='500 timesteps')
plt.plot(delayindex[0:136],SCs550old[1:137],color=colors[1,:],label='550 timesteps')
plt.plot(delayindex[0:149],SCs600old[1:150],color=colors[2,:],label='600 timesteps')
plt.plot(delayindex[0:161],SCs650old[1:162],color=colors[3,:],label='650 timesteps')
plt.plot(delayindex[0:174],SCs700old[1:175],color=colors[4,:],label='700 timesteps')
plt.plot(delayindex[0:186],SCs750old[1:187],color=colors[5,:],label='750 timesteps')
plt.plot(delayindex[0:199],SCs800old[1:200],color=colors[6,:],label='800 timesteps')
plt.plot(delayindex[0:211],SCs850old[1:212],color=colors[7,:],label='850 timesteps')
plt.plot(delayindex[0:219],SCs900old[1:220],color=colors[8,:],label='900 timesteps')
plt.plot(delayindex[0:229],SCs950old[1:230],color=colors[9,:],label='950 timesteps')

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Delay Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Statistical Complexity',fontsize=9)
plt.xlim(1,250)
plt.ylim(0,0.4)
plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)


savefilename='SC_recordlengthvariation_Class_3_old.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')

fig=plt.figure(num=46,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.25  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
plt.plot(delayindex[0:123],SCs500old[1:124],color=colors[0,:],label='500 timesteps')
#plt.plot(delayindex[0:136],SCs550old[1:137],color=colors[1,:],label='550 timesteps')
#plt.plot(delayindex[0:149],SCs600old[1:150],color=colors[2,:],label='600 timesteps')
#plt.plot(delayindex[0:161],SCs650old[1:162],color=colors[3,:],label='650 timesteps')
plt.plot(delayindex[0:174],SCs700old[1:175],color=colors[5,:],label='700 timesteps')
#plt.plot(delayindex[0:186],SCs750old[1:187],color=colors[5,:],label='750 timesteps')
#plt.plot(delayindex[0:199],SCs800old[1:200],color=colors[6,:],label='800 timesteps')
#plt.plot(delayindex[0:211],SCs850old[1:212],color=colors[7,:],label='850 timesteps')
plt.plot(delayindex[0:219],SCs900old[1:220],color=colors[9,:],label='900 timesteps')
#plt.plot(delayindex[0:229],SCs950old[1:230],color=colors[9,:],label='950 timesteps')

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Delay Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Complexity',fontsize=9)
plt.xlim(1,80)
plt.ylim(0.2,0.4)
plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)


savefilename='SC_recordlengthvariation_Class_3_old_zoom.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')

fig=plt.figure(num=44,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.25  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
plt.plot(delayindex[0:123],SCs500old[1:124],color=colors[0,:],label='500-Class 3')
plt.plot(delayindex[0:123],SCs500[1:124],color=colors[0,:],linestyle='dashed',label='500-Class 1')
#plt.plot(delayindex[0:136],SCs550old[1:137],color=colors[1,:],label='550 timesteps')
#plt.plot(delayindex[0:149],SCs600old[1:150],color=colors[2,:],label='600 timesteps')
#plt.plot(delayindex[0:161],SCs650old[1:162],color=colors[3,:],label='650 timesteps')
#plt.plot(delayindex[0:174],SCs700old[1:175],color=colors[4,:],label='700-Class 3')
#plt.plot(delayindex[0:174],SCs700[1:175],color=colors[4,:],linestyle='dashed',label='700-Class 1')
plt.plot(delayindex[0:186],SCs750old[1:187],color=colors[5,:],label='750-Class 3')
plt.plot(delayindex[0:186],SCs750[1:187],color=colors[5,:],linestyle='dashed',label='750-Class 1')
#plt.plot(delayindex[0:199],SCs800old[1:200],color=colors[6,:],label='800 timesteps')
#plt.plot(delayindex[0:211],SCs850old[1:212],color=colors[7,:],label='850 timesteps')
#plt.plot(delayindex[0:224],SCs900old[1:225],color=colors[8,:],label='900 timesteps')
plt.plot(delayindex[0:230],SCs950old[1:231],color=colors[9,:],label='950-Class 3')
plt.plot(delayindex[0:230],SCs950[1:231],color=colors[9,:],linestyle='dashed',label='950-Class 1')

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Delay Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Statistical Complexity',fontsize=9)
plt.xlim(1,250)
plt.ylim(0,0.4)
plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)


savefilename='SC_recordlengthvariation_Class_3vsClass_1.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')