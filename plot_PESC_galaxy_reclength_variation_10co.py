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

datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\GalpyData_July2018\\'
fileheader = 'PE_SC_IDdatabase_Type_32_data_2000_499_delays_5000orbits_2000_timesteps'
npy='.npz'

delayindex = np.arange(1,251)#250
timestep_arr = [500,550,600,650,700,750,800,850,900,950,1000]#,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500,1550,1600,1650,1700,1750,1800,1850,1900,1950,2000]
timeindex = (delayindex*1e5)/(1e6)

PEs_32 = np.zeros([11,250])
SCs_32 = np.zeros([11,250])
for file in np.arange(len(timestep_arr)):
    #fileheader = 'PE_SC_IDdatabase_Type_32_10co_data_8000_499_delays_329orbits_'+str(timestep_arr[file])+'_timesteps'
    fileheader = 'PE_SC_IDdatabase_Type_32_8co_data_8000_249_delays_2120orbits_'+str(timestep_arr[file])+'_timesteps'
    datafile = loadnpzfile(datadir+fileheader+npy)
    PEs_32[file,:]=datafile['PEs']
    SCs_32[file,:]=datafile['SCs']

#PEs_31 = np.zeros([31,500])
#SCs_31 = np.zeros([31,500])
#for file in np.arange(len(timestep_arr)):
#    fileheader = 'PE_SC_IDdatabase_Type_31_data_2000_499_delays_4212orbits_'+str(timestep_arr[file])+'_timesteps'
#    datafile = loadnpzfile(datadir+fileheader+npy)
#    PEs_31[file,:]=datafile['PEs']
#    SCs_31[file,:]=datafile['SCs']


"""
fileheader = 'PE_SC_IDdatabase_Type_2_data_249_delays_3000_orbits_galpy0718'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1000 = datafile['PEs']
SCs1000 = datafile['SCs']

#fileheader = 'PE_SC_New_DavidData_'+classtype+'_249_delays_2000_timesteps'
#datafile = loadnpzfile(datadir+fileheader+npy)
#PEs2000 = datafile['PEs']
#SCs2000 = datafile['SCs']

fileheader = 'PE_SC_IDdatabase_Type_2_data_249_delays_3000orbits_550_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs550 = datafile['PEs']
SCs550 = datafile['SCs']

fileheader = 'PE_SC_IDdatabase_Type_2_data_249_delays_3000orbits_600_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs600 = datafile['PEs']
SCs600 = datafile['SCs']

fileheader = 'PE_SC_IDdatabase_Type_2_data_249_delays_3000orbits_650_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs650 = datafile['PEs']
SCs650 = datafile['SCs']

fileheader = 'PE_SC_IDdatabase_Type_2_data_249_delays_3000orbits_700_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs700 = datafile['PEs']
SCs700 = datafile['SCs']

fileheader = 'PE_SC_IDdatabase_Type_2_data_249_delays_3000orbits_750_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs750 = datafile['PEs']
SCs750 = datafile['SCs']

fileheader = 'PE_SC_IDdatabase_Type_2_data_249_delays_3000orbits_800_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs800 = datafile['PEs']
SCs800 = datafile['SCs']

fileheader = 'PE_SC_IDdatabase_Type_2_data_249_delays_3000orbits_850_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs850 = datafile['PEs']
SCs850 = datafile['SCs']

fileheader = 'PE_SC_IDdatabase_Type_2_data_249_delays_3000orbits_900_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs900 = datafile['PEs']
SCs900 = datafile['SCs']

fileheader = 'PE_SC_IDdatabase_Type_2_data_249_delays_3000orbits_950_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs950 = datafile['PEs']
SCs950 = datafile['SCs']
"""

"""
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
"""

colors = np.zeros([12,4])
for i in np.arange(12):
    c = cm.spectral(i/12.,1)
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
plt.plot(timeindex,SCs_32[0,:],color=colors[0,:],label='500 timesteps')
plt.plot(timeindex,SCs_32[1,:],color=colors[1,:],label='550 timesteps')
plt.plot(timeindex,SCs_32[2,:],color=colors[2,:],label='600 timesteps')
plt.plot(timeindex,SCs_32[3,:],color=colors[3,:],label='650 timesteps')
plt.plot(timeindex,SCs_32[4,:],color=colors[4,:],label='700 timesteps')
plt.plot(timeindex,SCs_32[5,:],color=colors[5,:],label='750 timesteps')
plt.plot(timeindex,SCs_32[6,:],color=colors[6,:],label='800 timesteps')
plt.plot(timeindex,SCs_32[7,:],color=colors[7,:],label='850 timesteps')
plt.plot(timeindex,SCs_32[8,:],color=colors[8,:],label='900 timesteps')
plt.plot(timeindex,SCs_32[9,:],color=colors[9,:],label='950 timesteps')
plt.plot(timeindex,SCs_32[10,:],color=colors[10,:],label='1000 timesteps')

#plt.plot(timeindex,SCs_32[11,:],color=colors[11,:],label='1050 timesteps')
#plt.plot(timeindex,SCs_32[12,:],color=colors[12,:],label='1100 timesteps')
#plt.plot(timeindex,SCs_32[13,:],color=colors[13,:],label='1150 timesteps')
#plt.plot(timeindex,SCs_32[14,:],color=colors[14,:],label='1200 timesteps')
#plt.plot(timeindex,SCs_32[15,:],color=colors[15,:],label='1250 timesteps')

#plt.plot(timeindex,SCs_32[16,:],color=colors[16,:],label='1300 timesteps')
#plt.plot(timeindex,SCs_32[17,:],color=colors[17,:],label='1350 timesteps')
#plt.plot(timeindex,SCs_32[18,:],color=colors[18,:],label='1400 timesteps')
#plt.plot(timeindex,SCs_32[19,:],color=colors[19,:],label='1450 timesteps')
#plt.plot(timeindex,SCs_32[20,:],color=colors[20,:],label='1500 timesteps')
#plt.plot(timeindex,SCs_32[21,:],color=colors[21,:],label='1550 timesteps')
#plt.plot(timeindex,SCs_32[22,:],color=colors[22,:],label='1600 timesteps')


#plt.plot(timeindex,SCs_32[23,:],color=colors[23,:],label='1650 timesteps')
#plt.plot(timeindex,SCs_32[24,:],color=colors[24,:],label='1700 timesteps')
#plt.plot(timeindex,SCs_32[25,:],color=colors[25,:],label='1750 timesteps')
#plt.plot(timeindex,SCs_32[26,:],color=colors[26,:],label='1800 timesteps')

#plt.plot(timeindex,SCs_32[27,:],color=colors[27,:],label='1850 timesteps')
#plt.plot(timeindex,SCs_32[28,:],color=colors[28,:],label='1900 timesteps')
#plt.plot(timeindex,SCs_32[29,:],color=colors[29,:],label='1950 timesteps')
#plt.plot(timeindex,SCs_32[30,:],color=colors[30,:],label='2000 timesteps')



#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

#plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.xticks(fontsize=9)
plt.yticks(np.array([0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5]),[0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5],fontsize=9)
plt.xlabel('Delay Time [Myr]',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Statistical Complexity',fontsize=9)
#plt.xlim(1,499)
#plt.ylim(0,0.4)
plt.legend(loc='lower right',fontsize=3,frameon=False,handlelength=5)

savefilename='SC_recordlengthvariation_type32_8co_500to1000.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')

"""
fig=plt.figure(num=2,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.16  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.2  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
plt.plot(timeindex,SCs_31[0,:],color=colors[0,:],label='500 timesteps')
plt.plot(timeindex,SCs_31[1,:],color=colors[1,:],label='550 timesteps')
plt.plot(timeindex,SCs_31[2,:],color=colors[2,:],label='600 timesteps')
plt.plot(timeindex,SCs_31[3,:],color=colors[3,:],label='650 timesteps')
plt.plot(timeindex,SCs_31[4,:],color=colors[4,:],label='700 timesteps')
plt.plot(timeindex,SCs_31[5,:],color=colors[5,:],label='750 timesteps')
plt.plot(timeindex,SCs_31[6,:],color=colors[6,:],label='800 timesteps')
plt.plot(timeindex,SCs_31[7,:],color=colors[7,:],label='850 timesteps')
plt.plot(timeindex,SCs_31[8,:],color=colors[8,:],label='900 timesteps')
plt.plot(timeindex,SCs_31[9,:],color=colors[9,:],label='950 timesteps')
plt.plot(timeindex,SCs_31[10,:],color=colors[10,:],label='1000 timesteps')
plt.plot(timeindex,SCs_31[11,:],color=colors[11,:],label='1050 timesteps')
plt.plot(timeindex,SCs_31[12,:],color=colors[12,:],label='1100 timesteps')
plt.plot(timeindex,SCs_31[13,:],color=colors[13,:],label='1150 timesteps')
plt.plot(timeindex,SCs_31[14,:],color=colors[14,:],label='1200 timesteps')
plt.plot(timeindex,SCs_31[15,:],color=colors[15,:],label='1250 timesteps')
plt.plot(timeindex,SCs_31[16,:],color=colors[16,:],label='1300 timesteps')
plt.plot(timeindex,SCs_31[17,:],color=colors[17,:],label='1350 timesteps')
plt.plot(timeindex,SCs_31[18,:],color=colors[18,:],label='1400 timesteps')
plt.plot(timeindex,SCs_31[19,:],color=colors[19,:],label='1450 timesteps')
plt.plot(timeindex,SCs_31[20,:],color=colors[20,:],label='1500 timesteps')
plt.plot(timeindex,SCs_31[21,:],color=colors[21,:],label='1550 timesteps')
plt.plot(timeindex,SCs_31[22,:],color=colors[22,:],label='1600 timesteps')
plt.plot(timeindex,SCs_31[23,:],color=colors[23,:],label='1650 timesteps')
plt.plot(timeindex,SCs_31[24,:],color=colors[24,:],label='1700 timesteps')
plt.plot(timeindex,SCs_31[25,:],color=colors[25,:],label='1750 timesteps')
plt.plot(timeindex,SCs_31[26,:],color=colors[26,:],label='1800 timesteps')
plt.plot(timeindex,SCs_31[27,:],color=colors[27,:],label='1850 timesteps')
plt.plot(timeindex,SCs_31[28,:],color=colors[28,:],label='1900 timesteps')
plt.plot(timeindex,SCs_31[29,:],color=colors[29,:],label='1950 timesteps')
plt.plot(timeindex,SCs_31[30,:],color=colors[30,:],label='2000 timesteps')


#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

#plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.xticks(fontsize=9)
plt.yticks(np.array([0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5]),[0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5],fontsize=9)
plt.xlabel('Delay Time [Myr]',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Statistical Complexity',fontsize=9)
#plt.xlim(1,499)
#plt.ylim(0,0.4)
plt.legend(loc='lower right',fontsize=3,frameon=False,handlelength=5)

savefilename='SC_recordlengthvariation_type31_500to2000.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')
"""
"""
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
"""
"""
colors = np.zeros([11,4])
for i in np.arange(11):
    c = cm.spectral(i/11.,1)
    colors[i,:]=c
points = ['o','v','s','p','*','h','^','D','+','>','H','d','x','<']

plt.rc('axes',linewidth=0.75)
plt.rc('xtick.major',width=0.75)
plt.rc('ytick.major',width=0.75)
plt.rc('xtick.minor',width=0.75)
plt.rc('ytick.minor',width=0.75)
plt.rc('lines',markersize=2,markeredgewidth=0.0)

fig=plt.figure(num=4,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.25  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.9      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
plt.plot(delayindex[0:123],SCs500[1:124],color=colors[0,:],label='500 timesteps')
plt.plot(delayindex[0:136],SCs550[1:137],color=colors[1,:],label='550 timesteps')
plt.plot(delayindex[0:149],SCs600[1:150],color=colors[2,:],label='600 timesteps')
plt.plot(delayindex[0:161],SCs650[1:162],color=colors[3,:],label='650 timesteps')
plt.plot(delayindex[0:174],SCs700[1:175],color=colors[4,:],label='700 timesteps')
plt.plot(delayindex[0:186],SCs750[1:187],color=colors[5,:],label='750 timesteps')
plt.plot(delayindex[0:199],SCs800[1:200],color=colors[6,:],label='800 timesteps')
plt.plot(delayindex[0:211],SCs850[1:212],color=colors[7,:],label='850 timesteps')
plt.plot(delayindex[0:219],SCs900[1:220],color=colors[8,:],label='900 timesteps')
plt.plot(delayindex[0:229],SCs950[1:230],color=colors[9,:],label='950 timesteps')
plt.plot(delayindex[0:229],SCs1000[1:230],color=colors[10,:],label='1000 timesteps')
#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Delay Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Statistical Complexity',fontsize=9)
plt.xlim(1,250)
plt.ylim(0,0.4)
plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)


savefilename='SC_recordlengthvariation_Type2_galpy0718.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')
"""
"""
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
"""