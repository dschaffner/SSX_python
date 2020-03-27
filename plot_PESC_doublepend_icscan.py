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
datadir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Galatic Dynamics Data\\DoublePendulum\\'
fileheader = 'PE_SC_DPDoubPen_LsMsEq1_grav1_ICC1_embeddelay5_999_delays'
npz='.npz'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_L1=datafile['PEsx1']
SCsx1_L1=datafile['SCsx1']
delayindex=datafile['delays']

fileheader = 'PE_SC_DPDoubPen_LsEq0p1_MsEq1_grav9p8_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_Lp1=datafile['PEsx1']
SCsx1_Lp1=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq0p5_MsEq1_grav9p8_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_Lp5=datafile['PEsx1']
SCsx1_Lp5=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq10_MsEq1_grav9p8_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_L10=datafile['PEsx1']
SCsx1_L10=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq100_MsEq1_grav9p8_ICC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_L100=datafile['PEsx1']
SCsx1_L100=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icscanIC0_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_ic0=datafile['PEsx1']
SCsx1_ic0=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icscanIC1_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_ic1=datafile['PEsx1']
SCsx1_ic1=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icscanIC2_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_ic2=datafile['PEsx1']
SCsx1_ic2=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icscanIC3_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_ic3=datafile['PEsx1']
SCsx1_ic3=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icscanIC4_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_ic4=datafile['PEsx1']
SCsx1_ic4=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icscanIC5_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_ic5=datafile['PEsx1']
SCsx1_ic5=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icscanIC8_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_ic8=datafile['PEsx1']
SCsx1_ic8=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icscanIC9_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_ic9=datafile['PEsx1']
SCsx1_ic9=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icscanIC10_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_ic10=datafile['PEsx1']
SCsx1_ic10=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icscanIC11_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_ic11=datafile['PEsx1']
SCsx1_ic11=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icscanIC12_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_ic12=datafile['PEsx1']
SCsx1_ic12=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icscanIC13_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_ic13=datafile['PEsx1']
SCsx1_ic13=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icscanIC16_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_ic16=datafile['PEsx1']
SCsx1_ic16=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icscanIC17_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_ic17=datafile['PEsx1']
SCsx1_ic17=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icscanIC18_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_ic18=datafile['PEsx1']
SCsx1_ic18=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icscanIC19_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_ic19=datafile['PEsx1']
SCsx1_ic19=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icscanIC20_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_ic20=datafile['PEsx1']
SCsx1_ic20=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icscanIC21_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_ic21=datafile['PEsx1']
SCsx1_ic21=datafile['SCsx1']

fileheader = 'PE_SC_DPDoubPen_LsEq1_MsEq1_g9p81_tstep001_icanglescan_IC0_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx1_icang0=datafile['PEsx1']
SCsx1_icang0=datafile['SCsx1']

tmax, dt = 100, 0.001
t = np.arange(0, tmax+dt, dt)
timeindex = delayindex*0.001
"""
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
plt.semilogx(delayindex,SCsx1_Lp1,color='black',label='L=0.1')
plt.semilogx(delayindex,SCsx1_Lp5,color='blue',label='L=0.5')
plt.semilogx(delayindex,SCsx1_L1,color='green',label='L=1.0')
plt.semilogx(delayindex,SCsx1_L10,color='red',label='L=10.0')
plt.semilogx(delayindex,SCsx1_L100,color='orange',label='L=100.0')




#plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Delay Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Statistical Complexity',fontsize=9)
#plt.xlim(1,250)
plt.ylim(0,0.5)
plt.legend(loc='upper left',fontsize=5,frameon=False,handlelength=5)

savefilename='SC_DPx1_ICC1_Lscan_0p1to100.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')
"""