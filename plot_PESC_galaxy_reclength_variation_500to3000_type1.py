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
fileheader = 'PE_SC_IDdatabase_Type_1_data_2000_499_delays_5000orbits_2000_timesteps'
npy='.npz'

delayindex = np.arange(1,501)#250
timestep_arr = [500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150,1200,1250,1300,1350,1400,1450,1500,1550,1600,1650,1700,1750,1800,1850,1900,1950,2000,2200,2500,3000]
timeindex = (delayindex*1e5)/(1e6)

PEs_1 = np.zeros([34,500])
SCs_1 = np.zeros([34,500])
for file in np.arange(len(timestep_arr)):
    fileheader = 'PE_SC_IDdatabase_Type_1_data_3000_499_delays_3227orbits_'+str(timestep_arr[file])+'_timesteps'
    datafile = loadnpzfile(datadir+fileheader+npy)
    PEs_1[file,:]=datafile['PEs']
    SCs_1[file,:]=datafile['SCs']    



ncolors=34
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

SCs_1_500_endarray=np.where(SCs_1[0,1:]==0)[0][0]
SCs_1_550_endarray=np.where(SCs_1[1,1:]==0)[0][0]
SCs_1_600_endarray=np.where(SCs_1[2,1:]==0)[0][0]
SCs_1_650_endarray=np.where(SCs_1[3,1:]==0)[0][0]
SCs_1_700_endarray=np.where(SCs_1[4,1:]==0)[0][0]
SCs_1_750_endarray=np.where(SCs_1[5,1:]==0)[0][0]
SCs_1_800_endarray=np.where(SCs_1[6,1:]==0)[0][0]
SCs_1_850_endarray=np.where(SCs_1[7,1:]==0)[0][0]
SCs_1_900_endarray=np.where(SCs_1[8,1:]==0)[0][0]
SCs_1_950_endarray=np.where(SCs_1[9,1:]==0)[0][0]
SCs_1_1000_endarray=np.where(SCs_1[10,1:]==0)[0][0]
SCs_1_1050_endarray=np.where(SCs_1[11,1:]==0)[0][0]
SCs_1_1100_endarray=np.where(SCs_1[12,1:]==0)[0][0]
SCs_1_1150_endarray=np.where(SCs_1[13,1:]==0)[0][0]
SCs_1_1200_endarray=np.where(SCs_1[14,1:]==0)[0][0]
SCs_1_1250_endarray=np.where(SCs_1[15,1:]==0)[0][0]
SCs_1_1300_endarray=np.where(SCs_1[16,1:]==0)[0][0]
SCs_1_1350_endarray=np.where(SCs_1[17,1:]==0)[0][0]
SCs_1_1400_endarray=np.where(SCs_1[18,1:]==0)[0][0]
SCs_1_1450_endarray=np.where(SCs_1[19,1:]==0)[0][0]
SCs_1_1500_endarray=np.where(SCs_1[20,1:]==0)[0][0]
SCs_1_1550_endarray=np.where(SCs_1[21,1:]==0)[0][0]
SCs_1_1600_endarray=np.where(SCs_1[22,1:]==0)[0][0]
SCs_1_1650_endarray=np.where(SCs_1[23,1:]==0)[0][0]
SCs_1_1700_endarray=np.where(SCs_1[24,1:]==0)[0][0]
SCs_1_1750_endarray=np.where(SCs_1[25,1:]==0)[0][0]
SCs_1_1800_endarray=np.where(SCs_1[26,1:]==0)[0][0]
SCs_1_1850_endarray=np.where(SCs_1[27,1:]==0)[0][0]
SCs_1_1900_endarray=np.where(SCs_1[28,1:]==0)[0][0]
SCs_1_1950_endarray=np.where(SCs_1[29,1:]==0)[0][0]
SCs_1_2000_endarray=500
SCs_1_2200_endarray=500
SCs_1_2500_endarray=500
SCs_1_3000_endarray=500

ax1=plt.subplot(1,1,1)
plt.plot(timeindex[:SCs_1_500_endarray-1],SCs_1[0,1:SCs_1_500_endarray],color=colors[0,:],label='500 timesteps')
plt.plot(timeindex[:SCs_1_550_endarray-1],SCs_1[1,1:SCs_1_550_endarray],color=colors[1,:],label='550 timesteps')
plt.plot(timeindex[:SCs_1_600_endarray-1],SCs_1[2,1:SCs_1_600_endarray],color=colors[2,:],label='600 timesteps')
plt.plot(timeindex[:SCs_1_650_endarray-1],SCs_1[3,1:SCs_1_650_endarray],color=colors[3,:],label='650 timesteps')
plt.plot(timeindex[:SCs_1_700_endarray-1],SCs_1[4,1:SCs_1_700_endarray],color=colors[4,:],label='700 timesteps')
plt.plot(timeindex[:SCs_1_750_endarray-1],SCs_1[5,1:SCs_1_750_endarray],color=colors[5,:],label='750 timesteps')
plt.plot(timeindex[:SCs_1_800_endarray-1],SCs_1[6,1:SCs_1_800_endarray],color=colors[6,:],label='800 timesteps')
plt.plot(timeindex[:SCs_1_850_endarray-1],SCs_1[7,1:SCs_1_850_endarray],color=colors[7,:],label='850 timesteps')
plt.plot(timeindex[:SCs_1_900_endarray-1],SCs_1[8,1:SCs_1_900_endarray],color=colors[8,:],label='900 timesteps')
plt.plot(timeindex[:SCs_1_950_endarray-1],SCs_1[9,1:SCs_1_950_endarray],color=colors[9,:],label='950 timesteps')
plt.plot(timeindex[:SCs_1_1000_endarray-1],SCs_1[10,1:SCs_1_1000_endarray],color=colors[10,:],label='1000 timesteps')

plt.plot(timeindex[:SCs_1_1050_endarray-1],SCs_1[11,1:SCs_1_1050_endarray],color=colors[11,:],label='1050 timesteps')
plt.plot(timeindex[:SCs_1_1100_endarray-1],SCs_1[12,1:SCs_1_1100_endarray],color=colors[12,:],label='1100 timesteps')
plt.plot(timeindex[:SCs_1_1150_endarray-1],SCs_1[13,1:SCs_1_1150_endarray],color=colors[13,:],label='1150 timesteps')
plt.plot(timeindex[:SCs_1_1200_endarray-1],SCs_1[14,1:SCs_1_1200_endarray],color=colors[14,:],label='1200 timesteps')
plt.plot(timeindex[:SCs_1_1250_endarray-1],SCs_1[15,1:SCs_1_1250_endarray],color=colors[15,:],label='1250 timesteps')

plt.plot(timeindex[:SCs_1_1300_endarray-1],SCs_1[16,1:SCs_1_1300_endarray],color=colors[16,:],label='1300 timesteps')
plt.plot(timeindex[:SCs_1_1350_endarray-1],SCs_1[17,1:SCs_1_1350_endarray],color=colors[17,:],label='1350 timesteps')
plt.plot(timeindex[:SCs_1_1400_endarray-1],SCs_1[18,1:SCs_1_1400_endarray],color=colors[18,:],label='1400 timesteps')
plt.plot(timeindex[:SCs_1_1450_endarray-1],SCs_1[19,1:SCs_1_1450_endarray],color=colors[19,:],label='1450 timesteps')
plt.plot(timeindex[:SCs_1_1500_endarray-1],SCs_1[20,1:SCs_1_1500_endarray],color=colors[20,:],label='1500 timesteps')
plt.plot(timeindex[:SCs_1_1550_endarray-1],SCs_1[21,1:SCs_1_1550_endarray],color=colors[21,:],label='1550 timesteps')
plt.plot(timeindex[:SCs_1_1600_endarray-1],SCs_1[22,1:SCs_1_1600_endarray],color=colors[22,:],label='1600 timesteps')


plt.plot(timeindex[:SCs_1_1650_endarray-1],SCs_1[23,1:SCs_1_1650_endarray],color=colors[23,:],label='1650 timesteps')
plt.plot(timeindex[:SCs_1_1700_endarray-1],SCs_1[24,1:SCs_1_1700_endarray],color=colors[24,:],label='1700 timesteps')
plt.plot(timeindex[:SCs_1_1750_endarray-1],SCs_1[25,1:SCs_1_1750_endarray],color=colors[25,:],label='1750 timesteps')
plt.plot(timeindex[:SCs_1_1800_endarray-1],SCs_1[26,1:SCs_1_1800_endarray],color=colors[26,:],label='1800 timesteps')

plt.plot(timeindex[:SCs_1_1850_endarray-1],SCs_1[27,1:SCs_1_1850_endarray],color=colors[27,:],label='1850 timesteps')
plt.plot(timeindex[:SCs_1_1900_endarray-1],SCs_1[28,1:SCs_1_1900_endarray],color=colors[28,:],label='1900 timesteps')
plt.plot(timeindex[:SCs_1_1950_endarray-1],SCs_1[29,1:SCs_1_1950_endarray],color=colors[29,:],label='1950 timesteps')
plt.plot(timeindex[:SCs_1_2000_endarray-1],SCs_1[30,1:SCs_1_2000_endarray],color=colors[30,:],label='2000 timesteps')

plt.plot(timeindex[:SCs_1_2200_endarray-1],SCs_1[31,1:SCs_1_2200_endarray],color=colors[31,:],label='2200 timesteps')
plt.plot(timeindex[:SCs_1_2500_endarray-1],SCs_1[32,1:SCs_1_2500_endarray],color=colors[32,:],label='2500 timesteps')
plt.plot(timeindex[:SCs_1_3000_endarray-1],SCs_1[33,1:SCs_1_3000_endarray],color=colors[33,:],label='3000 timesteps')

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
plt.ylim(0.1,0.4)
leg=plt.legend(loc='lower right',fontsize=3,frameon=False,handlelength=5)
leg.set_title('Analysis Length',prop={'size':5})

savefilename='SC_recordlengthvariation_type1_3000steps_6co_500to3000.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')
