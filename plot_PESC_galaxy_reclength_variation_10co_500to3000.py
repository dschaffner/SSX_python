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
npy='.npz'

delayindex = np.arange(1,501)#250
timestep_arr = [500,550,600,650,700,750,800,850,900,950,1000,1250,1500,2000,4000,6000,8000]
timeindex = (delayindex*1e5)/(1e6)
num_timesteps = 17

PEs_32 = np.zeros([num_timesteps,500])
SCs_32 = np.zeros([num_timesteps,500])
for file in np.arange(len(timestep_arr)):
    fileheader = 'PE_SC_IDdatabase_Type_32_10co_data_8000_499_delays_329orbits_'+str(timestep_arr[file])+'_timesteps'
    datafile = loadnpzfile(datadir+fileheader+npy)
    PEs_32[file,:]=datafile['PEs']
    SCs_32[file,:]=datafile['SCs']    

    
#PEs_31 = np.zeros([34,500])
#SCs_31 = np.zeros([34,500])
#for file in np.arange(len(timestep_arr)):
#    fileheader = 'PE_SC_IDdatabase_Type_31_10co_data_3000_499_delays_253orbits_'+str(timestep_arr[file])+'_timesteps'
#    datafile = loadnpzfile(datadir+fileheader+npy)
#    PEs_31[file,:]=datafile['PEs']
#    SCs_31[file,:]=datafile['SCs']



ncolors=num_timesteps
colors = np.zeros([ncolors,4])
for i in np.arange(ncolors):
    c = cm.spectral(i/float(ncolors),1)
    colors[i,:]=c
points = ['o','v','s','p','*','h','^','D','+','>','H','d','x','<']
        
plt.rc('axes',linewidth=0.75)
plt.rc('xtick.major',width=0.75)
plt.rc('ytick.major',width=0.75)
plt.rc('xtick.minor',width=0.75)
plt.rc('ytick.minor',width=0.75)
plt.rc('lines',markersize=2,markeredgewidth=0.0)


plt.rc('lines',markersize=1.5,markeredgewidth=0.0)
fig=plt.figure(num=1,figsize=(6,6),dpi=300,facecolor='w',edgecolor='k')
left  = 0.16  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.07  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

SCs_32_500_endarray=np.where(SCs_32[0,1:]==0)[0][0]
SCs_32_550_endarray=np.where(SCs_32[1,1:]==0)[0][0]
SCs_32_600_endarray=np.where(SCs_32[2,1:]==0)[0][0]
SCs_32_650_endarray=np.where(SCs_32[3,1:]==0)[0][0]
SCs_32_700_endarray=np.where(SCs_32[4,1:]==0)[0][0]
SCs_32_750_endarray=np.where(SCs_32[5,1:]==0)[0][0]
SCs_32_800_endarray=np.where(SCs_32[6,1:]==0)[0][0]
SCs_32_850_endarray=np.where(SCs_32[7,1:]==0)[0][0]
SCs_32_900_endarray=np.where(SCs_32[8,1:]==0)[0][0]
SCs_32_950_endarray=np.where(SCs_32[9,1:]==0)[0][0]
SCs_32_1000_endarray=np.where(SCs_32[10,1:]==0)[0][0]
#SCs_32_1050_endarray=np.where(SCs_32[11,1:]==0)[0][0]
#SCs_32_1100_endarray=np.where(SCs_32[12,1:]==0)[0][0]
#SCs_32_1150_endarray=np.where(SCs_32[13,1:]==0)[0][0]
#SCs_32_1200_endarray=np.where(SCs_32[14,1:]==0)[0][0]
SCs_32_1250_endarray=np.where(SCs_32[11,1:]==0)[0][0]
#SCs_32_1300_endarray=np.where(SCs_32[16,1:]==0)[0][0]
#SCs_32_1350_endarray=np.where(SCs_32[17,1:]==0)[0][0]
#SCs_32_1400_endarray=np.where(SCs_32[18,1:]==0)[0][0]
#SCs_32_1450_endarray=np.where(SCs_32[19,1:]==0)[0][0]
SCs_32_1500_endarray=np.where(SCs_32[12,1:]==0)[0][0]
#SCs_32_1550_endarray=np.where(SCs_32[21,1:]==0)[0][0]
#SCs_32_1600_endarray=np.where(SCs_32[22,1:]==0)[0][0]
#SCs_32_1650_endarray=np.where(SCs_32[23,1:]==0)[0][0]
#SCs_32_1700_endarray=np.where(SCs_32[24,1:]==0)[0][0]
#SCs_32_1750_endarray=np.where(SCs_32[25,1:]==0)[0][0]
#SCs_32_1800_endarray=np.where(SCs_32[26,1:]==0)[0][0]
#SCs_32_1850_endarray=np.where(SCs_32[27,1:]==0)[0][0]
#SCs_32_1900_endarray=np.where(SCs_32[28,1:]==0)[0][0]
#SCs_32_1950_endarray=np.where(SCs_32[29,1:]==0)[0][0]
SCs_32_2000_endarray=500
#SCs_32_2200_endarray=500
#SCs_32_2500_endarray=500
SCs_32_4000_endarray=500
SCs_32_6000_endarray=500
SCs_32_8000_endarray=500

ax1=plt.subplot(1,1,1)
plt.plot(timeindex[:SCs_32_500_endarray-1],SCs_32[0,1:SCs_32_500_endarray],color=colors[0,:],label='500 timesteps')
plt.plot(timeindex[:SCs_32_550_endarray-1],SCs_32[1,1:SCs_32_550_endarray],color=colors[1,:],label='550 timesteps')
plt.plot(timeindex[:SCs_32_600_endarray-1],SCs_32[2,1:SCs_32_600_endarray],color=colors[2,:],label='600 timesteps')
plt.plot(timeindex[:SCs_32_650_endarray-1],SCs_32[3,1:SCs_32_650_endarray],color=colors[3,:],label='650 timesteps')
plt.plot(timeindex[:SCs_32_700_endarray-1],SCs_32[4,1:SCs_32_700_endarray],color=colors[4,:],label='700 timesteps')
plt.plot(timeindex[:SCs_32_750_endarray-1],SCs_32[5,1:SCs_32_750_endarray],color=colors[5,:],label='750 timesteps')
plt.plot(timeindex[:SCs_32_800_endarray-1],SCs_32[6,1:SCs_32_800_endarray],color=colors[6,:],label='800 timesteps')
plt.plot(timeindex[:SCs_32_850_endarray-1],SCs_32[7,1:SCs_32_850_endarray],color=colors[7,:],label='850 timesteps')
plt.plot(timeindex[:SCs_32_900_endarray-1],SCs_32[8,1:SCs_32_900_endarray],color=colors[8,:],label='900 timesteps')
plt.plot(timeindex[:SCs_32_950_endarray-1],SCs_32[9,1:SCs_32_950_endarray],color=colors[9,:],label='950 timesteps')
plt.plot(timeindex[:SCs_32_1000_endarray-1],SCs_32[10,1:SCs_32_1000_endarray],color=colors[10,:],label='1000 timesteps')

#plt.plot(timeindex[:SCs_32_1050_endarray-1],SCs_32[11,1:SCs_32_1050_endarray],color=colors[11,:],label='1050 timesteps')
#plt.plot(timeindex[:SCs_32_1100_endarray-1],SCs_32[12,1:SCs_32_1100_endarray],color=colors[12,:],label='1100 timesteps')
#plt.plot(timeindex[:SCs_32_1150_endarray-1],SCs_32[13,1:SCs_32_1150_endarray],color=colors[13,:],label='1150 timesteps')
#plt.plot(timeindex[:SCs_32_1200_endarray-1],SCs_32[14,1:SCs_32_1200_endarray],color=colors[14,:],label='1200 timesteps')
plt.plot(timeindex[:SCs_32_1250_endarray-1],SCs_32[11,1:SCs_32_1250_endarray],color=colors[11,:],label='1250 timesteps')

#plt.plot(timeindex[:SCs_32_1300_endarray-1],SCs_32[16,1:SCs_32_1300_endarray],color=colors[16,:],label='1300 timesteps')
#plt.plot(timeindex[:SCs_32_1350_endarray-1],SCs_32[17,1:SCs_32_1350_endarray],color=colors[17,:],label='1350 timesteps')
#plt.plot(timeindex[:SCs_32_1400_endarray-1],SCs_32[18,1:SCs_32_1400_endarray],color=colors[18,:],label='1400 timesteps')
#plt.plot(timeindex[:SCs_32_1450_endarray-1],SCs_32[19,1:SCs_32_1450_endarray],color=colors[19,:],label='1450 timesteps')
plt.plot(timeindex[:SCs_32_1500_endarray-1],SCs_32[12,1:SCs_32_1500_endarray],color=colors[12,:],label='1500 timesteps')
#plt.plot(timeindex[:SCs_32_1550_endarray-1],SCs_32[21,1:SCs_32_1550_endarray],color=colors[21,:],label='1550 timesteps')
#plt.plot(timeindex[:SCs_32_1600_endarray-1],SCs_32[22,1:SCs_32_1600_endarray],color=colors[22,:],label='1600 timesteps')


#plt.plot(timeindex[:SCs_32_1650_endarray-1],SCs_32[23,1:SCs_32_1650_endarray],color=colors[23,:],label='1650 timesteps')
#plt.plot(timeindex[:SCs_32_1700_endarray-1],SCs_32[24,1:SCs_32_1700_endarray],color=colors[24,:],label='1700 timesteps')
#plt.plot(timeindex[:SCs_32_1750_endarray-1],SCs_32[25,1:SCs_32_1750_endarray],color=colors[25,:],label='1750 timesteps')
#plt.plot(timeindex[:SCs_32_1800_endarray-1],SCs_32[26,1:SCs_32_1800_endarray],color=colors[26,:],label='1800 timesteps')

#plt.plot(timeindex[:SCs_32_1850_endarray-1],SCs_32[27,1:SCs_32_1850_endarray],color=colors[27,:],label='1850 timesteps')
#plt.plot(timeindex[:SCs_32_1900_endarray-1],SCs_32[28,1:SCs_32_1900_endarray],color=colors[28,:],label='1900 timesteps')
#plt.plot(timeindex[:SCs_32_1950_endarray-1],SCs_32[29,1:SCs_32_1950_endarray],color=colors[29,:],label='1950 timesteps')
plt.plot(timeindex[:SCs_32_2000_endarray-1],SCs_32[13,1:SCs_32_2000_endarray],color=colors[13,:],label='2000 timesteps')

#plt.plot(timeindex[:SCs_32_2200_endarray-1],SCs_32[31,1:SCs_32_2200_endarray],color=colors[31,:],label='2200 timesteps')
#plt.plot(timeindex[:SCs_32_2500_endarray-1],SCs_32[32,1:SCs_32_2500_endarray],color=colors[32,:],label='2500 timesteps')
#plt.plot(timeindex[:SCs_32_3000_endarray-1],SCs_32[33,1:SCs_32_3000_endarray],color=colors[33,:],label='3000 timesteps')
plt.plot(timeindex[:SCs_32_4000_endarray-1],SCs_32[14,1:SCs_32_4000_endarray],color=colors[14,:],label='4000 timesteps')
plt.plot(timeindex[:SCs_32_6000_endarray-1],SCs_32[15,1:SCs_32_6000_endarray],color=colors[15,:],label='6000 timesteps')
plt.plot(timeindex[:SCs_32_8000_endarray-1],SCs_32[16,1:SCs_32_8000_endarray],color=colors[16,:],label='8000 timesteps')

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

#plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)

delayarray = np.array([0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360])
timearray = (delayarray*1e5)/(1e6)
timelist = list(timearray.astype(int))
plt.xticks(timearray,timelist,fontsize=8)
plt.xlabel('Delay Time [Myr]',fontsize=11)

#ax1.set_xticklabels([])
plt.yticks(np.array([0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5]),[0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5],fontsize=9)
plt.ylabel('Statistical Complexity',fontsize=9)
plt.xlim(0,36)
#plt.ylim(0.1,0.4)
leg=plt.legend(loc='lower right',fontsize=3,frameon=False,handlelength=5)
leg.set_title('Analysis Length',prop={'size':5})

savefilename='SC_recordlengthvariation_type32_8000steps_10co_500to8000.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')

"""
fig=plt.figure(num=2,figsize=(6,6),dpi=300,facecolor='w',edgecolor='k')
left  = 0.16  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.07  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

SCs_31_500_endarray=np.where(SCs_31[0,1:]==0)[0][0]
SCs_31_550_endarray=np.where(SCs_31[1,1:]==0)[0][0]
SCs_31_600_endarray=np.where(SCs_31[2,1:]==0)[0][0]
SCs_31_650_endarray=np.where(SCs_31[3,1:]==0)[0][0]
SCs_31_700_endarray=np.where(SCs_31[4,1:]==0)[0][0]
SCs_31_750_endarray=np.where(SCs_31[5,1:]==0)[0][0]
SCs_31_800_endarray=np.where(SCs_31[6,1:]==0)[0][0]
SCs_31_850_endarray=np.where(SCs_31[7,1:]==0)[0][0]
SCs_31_900_endarray=np.where(SCs_31[8,1:]==0)[0][0]
SCs_31_950_endarray=np.where(SCs_31[9,1:]==0)[0][0]
SCs_31_1000_endarray=np.where(SCs_31[10,1:]==0)[0][0]
SCs_31_1050_endarray=np.where(SCs_31[11,1:]==0)[0][0]
SCs_31_1100_endarray=np.where(SCs_31[12,1:]==0)[0][0]
SCs_31_1150_endarray=np.where(SCs_31[13,1:]==0)[0][0]
SCs_31_1200_endarray=np.where(SCs_31[14,1:]==0)[0][0]
SCs_31_1250_endarray=np.where(SCs_31[15,1:]==0)[0][0]
SCs_31_1300_endarray=np.where(SCs_31[16,1:]==0)[0][0]
SCs_31_1350_endarray=np.where(SCs_31[17,1:]==0)[0][0]
SCs_31_1400_endarray=np.where(SCs_31[18,1:]==0)[0][0]
SCs_31_1450_endarray=np.where(SCs_31[19,1:]==0)[0][0]
SCs_31_1500_endarray=np.where(SCs_31[20,1:]==0)[0][0]
SCs_31_1550_endarray=np.where(SCs_31[21,1:]==0)[0][0]
SCs_31_1600_endarray=np.where(SCs_31[22,1:]==0)[0][0]
SCs_31_1650_endarray=np.where(SCs_31[23,1:]==0)[0][0]
SCs_31_1700_endarray=np.where(SCs_31[24,1:]==0)[0][0]
SCs_31_1750_endarray=np.where(SCs_31[26,1:]==0)[0][0]
SCs_31_1800_endarray=np.where(SCs_31[27,1:]==0)[0][0]
SCs_31_1850_endarray=np.where(SCs_31[28,1:]==0)[0][0]
SCs_31_1900_endarray=np.where(SCs_31[29,1:]==0)[0][0]
SCs_31_1950_endarray=500
SCs_31_2000_endarray=500
SCs_31_2200_endarray=500
SCs_31_2500_endarray=500
SCs_31_3000_endarray=500

ax1=plt.subplot(1,1,1)
plt.plot(timeindex[:SCs_31_500_endarray-1],SCs_31[0,1:SCs_31_500_endarray],color=colors[0,:],label='500 timesteps')
plt.plot(timeindex[:SCs_31_550_endarray-1],SCs_31[1,1:SCs_31_550_endarray],color=colors[1,:],label='550 timesteps')
plt.plot(timeindex[:SCs_31_600_endarray-1],SCs_31[2,1:SCs_31_600_endarray],color=colors[2,:],label='600 timesteps')
plt.plot(timeindex[:SCs_31_650_endarray-1],SCs_31[3,1:SCs_31_650_endarray],color=colors[3,:],label='650 timesteps')
plt.plot(timeindex[:SCs_31_700_endarray-1],SCs_31[4,1:SCs_31_700_endarray],color=colors[4,:],label='700 timesteps')
plt.plot(timeindex[:SCs_31_750_endarray-1],SCs_31[5,1:SCs_31_750_endarray],color=colors[5,:],label='750 timesteps')
plt.plot(timeindex[:SCs_31_800_endarray-1],SCs_31[6,1:SCs_31_800_endarray],color=colors[6,:],label='800 timesteps')
plt.plot(timeindex[:SCs_31_850_endarray-1],SCs_31[7,1:SCs_31_850_endarray],color=colors[7,:],label='850 timesteps')
plt.plot(timeindex[:SCs_31_900_endarray-1],SCs_31[8,1:SCs_31_900_endarray],color=colors[8,:],label='900 timesteps')
plt.plot(timeindex[:SCs_31_950_endarray-1],SCs_31[9,1:SCs_31_950_endarray],color=colors[9,:],label='950 timesteps')
plt.plot(timeindex[:SCs_31_1000_endarray-1],SCs_31[10,1:SCs_31_1000_endarray],color=colors[10,:],label='1000 timesteps')

plt.plot(timeindex[:SCs_31_1050_endarray-1],SCs_31[11,1:SCs_31_1050_endarray],color=colors[11,:],label='1050 timesteps')
plt.plot(timeindex[:SCs_31_1100_endarray-1],SCs_31[12,1:SCs_31_1100_endarray],color=colors[12,:],label='1100 timesteps')
plt.plot(timeindex[:SCs_31_1150_endarray-1],SCs_31[13,1:SCs_31_1150_endarray],color=colors[13,:],label='1150 timesteps')
plt.plot(timeindex[:SCs_31_1200_endarray-1],SCs_31[14,1:SCs_31_1200_endarray],color=colors[14,:],label='1200 timesteps')
plt.plot(timeindex[:SCs_31_1250_endarray-1],SCs_31[15,1:SCs_31_1250_endarray],color=colors[15,:],label='1250 timesteps')

plt.plot(timeindex[:SCs_31_1300_endarray-1],SCs_31[16,1:SCs_31_1300_endarray],color=colors[16,:],label='1300 timesteps')
plt.plot(timeindex[:SCs_31_1350_endarray-1],SCs_31[17,1:SCs_31_1350_endarray],color=colors[17,:],label='1350 timesteps')
plt.plot(timeindex[:SCs_31_1400_endarray-1],SCs_31[18,1:SCs_31_1400_endarray],color=colors[18,:],label='1400 timesteps')
plt.plot(timeindex[:SCs_31_1450_endarray-1],SCs_31[19,1:SCs_31_1450_endarray],color=colors[19,:],label='1450 timesteps')
plt.plot(timeindex[:SCs_31_1500_endarray-1],SCs_31[20,1:SCs_31_1500_endarray],color=colors[20,:],label='1500 timesteps')
plt.plot(timeindex[:SCs_31_1550_endarray-1],SCs_31[21,1:SCs_31_1550_endarray],color=colors[21,:],label='1550 timesteps')
plt.plot(timeindex[:SCs_31_1600_endarray-1],SCs_31[22,1:SCs_31_1600_endarray],color=colors[22,:],label='1600 timesteps')


plt.plot(timeindex[:SCs_31_1650_endarray-1],SCs_31[23,1:SCs_31_1650_endarray],color=colors[23,:],label='1650 timesteps')
plt.plot(timeindex[:SCs_31_1700_endarray-1],SCs_31[24,1:SCs_31_1700_endarray],color=colors[24,:],label='1700 timesteps')
plt.plot(timeindex[:SCs_31_1750_endarray-1],SCs_31[25,1:SCs_31_1750_endarray],color=colors[25,:],label='1750 timesteps')
plt.plot(timeindex[:SCs_31_1800_endarray-1],SCs_31[26,1:SCs_31_1800_endarray],color=colors[26,:],label='1800 timesteps')

plt.plot(timeindex[:SCs_31_1850_endarray-1],SCs_31[27,1:SCs_31_1850_endarray],color=colors[27,:],label='1850 timesteps')
plt.plot(timeindex[:SCs_31_1900_endarray-1],SCs_31[28,1:SCs_31_1900_endarray],color=colors[28,:],label='1900 timesteps')
plt.plot(timeindex[:SCs_31_1950_endarray-1],SCs_31[29,1:SCs_31_1950_endarray],color=colors[29,:],label='1950 timesteps')
plt.plot(timeindex[:SCs_31_2000_endarray-1],SCs_31[30,1:SCs_31_2000_endarray],color=colors[30,:],label='2000 timesteps')

plt.plot(timeindex[:SCs_31_2200_endarray-1],SCs_31[31,1:SCs_31_2200_endarray],color=colors[31,:],label='2200 timesteps')
plt.plot(timeindex[:SCs_31_2500_endarray-1],SCs_31[32,1:SCs_31_2500_endarray],color=colors[32,:],label='2500 timesteps')
plt.plot(timeindex[:SCs_31_3000_endarray-1],SCs_31[33,1:SCs_31_3000_endarray],color=colors[33,:],label='3000 timesteps')



#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

delayarray = np.array([0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360])
timearray = (delayarray*1e5)/(1e6)
timelist = list(timearray.astype(int))
plt.xticks(timearray,timelist,fontsize=8)
plt.xlabel('Delay Time [Myr]',fontsize=11)

plt.yticks(np.array([0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5]),[0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5],fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Statistical Complexity',fontsize=9)
plt.xlim(0,36)
#plt.ylim(0.1,0.4)
leg=plt.legend(loc='lower right',fontsize=3,frameon=False,handlelength=5)
leg.set_title('Analysis Length',prop={'size':5})

savefilename='SC_recordlengthvariation_type31_3000steps_10co_500to3000.png'
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