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

datadir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Galatic Dynamics Data\\GalpyData_July2018\\resorted_data\\'#8CR_timescan\\'#\\type32\\'

npy='.npz'

fileheader = 'PE_SC_Type_32i_Rg_499_delays_1orbits_of1_total200000_timesteps_resorted_et_HiResShot0'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32_0 = datafile['PEs']
SCs32_0= datafile['SCs']

fileheader = 'PE_SC_Type_32i_Rg_499_delays_1orbits_of1_total200000_timesteps_resorted_et_HiResShot1'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32_1 = datafile['PEs']
SCs32_1= datafile['SCs']

fileheader = 'PE_SC_Type_1_Rg_499_delays_1orbits_of1_total200000_timesteps_resorted_et_HiResShot0'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1_0 = datafile['PEs']
SCs1_0= datafile['SCs']

fileheader = 'PE_SC_Type_1_Rg_499_delays_1orbits_of1_total200000_timesteps_resorted_et_HiResShot1'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1_1 = datafile['PEs']
SCs1_1= datafile['SCs']





M4dyn=1117
M5dyn=1396
M6dyn=1676
M7dyn=1955
M8dyn=2234
M9dyn=2513
M10dyn=2793

delayindex = np.arange(1,49901)
timeindex=(delayindex*1e5)/(1e6*100.0)
patterntime=timeindex*4

timeindex_normM4=patterntime/(M4dyn/10)
timeindex_normM5=patterntime/(M5dyn/10)
timeindex_normM6=patterntime/(M6dyn/10)
timeindex_normM7=patterntime/(M7dyn/10)
timeindex_normM8=patterntime/(M8dyn/10)
timeindex_normM9=patterntime/(M9dyn/10)
timeindex_normM10=patterntime/(M10dyn/10)

orbitdur1=np.round(1300/M4dyn,1)
orbitdur2=np.round(1600/M5dyn,1)
orbitdur3=np.round(2000/M6dyn,1)
orbitdur4=np.round(2550/M7dyn,1)
orbitdur5=np.round(3500/M8dyn,1) 
orbitdur6=np.round(4000/M9dyn,1) 
orbitdur7=np.round(4000/M10dyn,1) 

numcolors=7
colors=np.zeros([numcolors,4])
for i in np.arange(numcolors):
    c = plt.cm.plasma(i/(float(numcolors)),1)
    colors[i,:]=c

points = ['o','v','s','p','*','h','^','D','+','>','H','d','x','<']
        
plt.rc('axes',linewidth=2.0)
plt.rc('xtick.major',width=2.0)
plt.rc('ytick.major',width=2.0)
plt.rc('xtick.minor',width=2.0)
plt.rc('ytick.minor',width=2.0)
plt.rc('lines',markersize=8,markeredgewidth=0.0,linewidth=3.0)

#plt.rcParams['ps.fonttype'] = 42
#plt.rcParams['pdf.fonttype'] = 42

fig=plt.figure(num=1,figsize=(4,5),dpi=200,facecolor='w',edgecolor='k')
left  = 0.16  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.2  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.0   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)



ax1=plt.subplot(1,1,1)

plt.plot(timeindex_normM6,SCs32_0[1:],color=colors[0,:],label='Shot0 Type32')
plt.plot(timeindex_normM6,SCs32_1[1:],color=colors[0,:],label='Shot1 Type32')
plt.plot(timeindex_normM6,SCs1_0[1:],color=colors[3,:],label='Shot0 Type 1')
plt.plot(timeindex_normM6,SCs1_1[1:],color=colors[3,:],label='Shot1 Type 1')

plt.xticks(fontsize=12)
plt.yticks(np.array([0.10,0.20,0.30,0.40]),[0.10,0.20,0.30,0.40],fontsize=12)
plt.xlabel(r'$t_{pat}/T_{Dyn}^{M-}$',fontsize=18)
#plt.xlabel('Delay Steps',fontsize=9)
plt.ylabel(r'$C$',fontsize=12)
#plt.xlim(0,1.0)
#plt.ylim(0.11,0.4)
#plt.legend(loc='lower right',fontsize=12,frameon=False,handlelength=5,numpoints=3)
#plt.text(0.07,0.92,'(a)',horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes,fontsize=16)



leg=plt.legend(loc='lower left',fontsize=8,ncol=1,frameon=False,handlelength=5,numpoints=5)
#leg.set_title('CR: (Orbit Duration=$T_{D}$)',prop={'size':10})
#plt.text(0.07,0.92,'(b)',horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes,fontsize=16)

savefilename='SC_and_PE_HiRes_singleshot.png'
#savefilename='SC_and_PE_CRscan_peakcomplexitysignature_et_dyntimenorm_ApJver_newcolor.eps'
#savefilename='PE_galpy0718_1000timesteps_all_orbits.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()

