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
datadir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Galatic Dynamics Data\\DoublePendulum\\nPend\\'
fileheader = 'PE_SC_npend2mass_135deg_0velIC_embeddelay5_999_delays'
npz='.npz'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx_2M=datafile['PExs']
SCsx_2M=datafile['SCxs']
delayindex=datafile['delays']

fileheader = 'PE_SC_npend3mass_135deg_0velIC_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx_3M=datafile['PExs']
SCsx_3M=datafile['SCxs']

fileheader = 'PE_SC_npend4mass_135deg_0velIC_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx_4M=datafile['PExs']
SCsx_4M=datafile['SCxs']

fileheader = 'PE_SC_npend5mass_135deg_0velIC_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx_5M=datafile['PExs']
SCsx_5M=datafile['SCxs']

fileheader = 'PE_SC_npend6mass_135deg_0velIC_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx_6M=datafile['PExs']
SCsx_6M=datafile['SCxs']

fileheader = 'PE_SC_npend7mass_135deg_0velIC_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx_7M=datafile['PExs']
SCsx_7M=datafile['SCxs']

fileheader = 'PE_SC_npend8mass_135deg_0velIC_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx_8M=datafile['PExs']
SCsx_8M=datafile['SCxs']

fileheader = 'PE_SC_npend9mass_135deg_0velIC_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx_9M=datafile['PExs']
SCsx_9M=datafile['SCxs']

fileheader = 'PE_SC_npend10mass_135deg_0velIC_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx_10M=datafile['PExs']
SCsx_10M=datafile['SCxs']

fileheader = 'PE_SC_npend20mass_135deg_0velIC_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEsx_20M=datafile['PExs']
SCsx_20M=datafile['SCxs']

fileheader = 'PE_SC_interpolated_noise_100kInto10k_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEs_noise=datafile['PEs']
SCs_noise=datafile['SCs']

fileheader = 'PE_SC_interpolated_noise_100kInto5k_embeddelay5_999_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEs_noise2=datafile['PEs']
SCs_noise2=datafile['SCs']

fileheader='nPen_2masses_LsEq1_MsEq1_g9p81_1000sec_tstep001_135degIC_0velIC'
datafile = loadnpzfile(datadir+fileheader+npz)
x2M=datafile['x']
fileheader='nPen_3masses_LsEq1_MsEq1_g9p81_1000sec_tstep001_135degIC_0velIC'
datafile = loadnpzfile(datadir+fileheader+npz)
x3M=datafile['x']
fileheader='nPen_4masses_LsEq1_MsEq1_g9p81_1000sec_tstep001_135degIC_0velIC'
datafile = loadnpzfile(datadir+fileheader+npz)
x4M=datafile['x']
fileheader='nPen_5masses_LsEq1_MsEq1_g9p81_1000sec_tstep001_135degIC_0velIC'
datafile = loadnpzfile(datadir+fileheader+npz)
x5M=datafile['x']
fileheader='nPen_6masses_LsEq1_MsEq1_g9p81_1000sec_tstep001_135degIC_0velIC'
datafile = loadnpzfile(datadir+fileheader+npz)
x6M=datafile['x']
fileheader='nPen_7masses_LsEq1_MsEq1_g9p81_1000sec_tstep001_135degIC_0velIC'
datafile = loadnpzfile(datadir+fileheader+npz)
x7M=datafile['x']
fileheader='nPen_8masses_LsEq1_MsEq1_g9p81_1000sec_tstep001_135degIC_0velIC'
datafile = loadnpzfile(datadir+fileheader+npz)
x8M=datafile['x']
fileheader='nPen_9masses_LsEq1_MsEq1_g9p81_1000sec_tstep001_135degIC_0velIC'
datafile = loadnpzfile(datadir+fileheader+npz)
x9M=datafile['x']
fileheader='nPen_10masses_LsEq1_MsEq1_g9p81_1000sec_tstep001_135degIC_0velIC'
datafile = loadnpzfile(datadir+fileheader+npz)
x10M=datafile['x']
fileheader='nPen_20masses_LsEq1_MsEq1_g9p81_1000sec_tstep001_135degIC_0velIC'
datafile = loadnpzfile(datadir+fileheader+npz)
x20M=datafile['x']

noise_array=np.random.uniform(-1,1,size=100000)
delay_array = np.arange(1,1000)
num_delays = len(delay_array)+1
PEs_noise100k = np.zeros(num_delays)
SCs_noise100k = np.zeros(num_delays)
for loop_delay in delay_array:
    print ('On Delay ', loop_delay)
    PEs_noise100k[loop_delay],SCs_noise100k[loop_delay] = CH(noise_array,5,delay=loop_delay)

tmax, dt = 100, 0.001
t = np.arange(0, tmax+dt, dt)
timeindex = delayindex*0.001

plt.plot(SCsx_20M[1:,0])
plt.plot(SCsx_10M[1:,0])
plt.plot(SCsx_5M[1:,0])
plt.plot(SCsx_2M[1:,0])
plt.plot(SCs_noise100k[1:])

plt.figure(2)
plt.plot(PEsx_20M[1:,0])
plt.plot(PEsx_10M[1:,0])
plt.plot(PEsx_5M[1:,0])
plt.plot(PEsx_2M[1:,0])
plt.plot(PEs_noise100k[1:])

plt.figure(3)
plt.loglog(SCsx_2M[1:,0])
plt.loglog(SCsx_5M[1:,0])
plt.loglog(SCsx_10M[1:,0])
plt.loglog(SCsx_20M[1:,0])
plt.loglog(SCs_noise100k[1:])

"""
import matplotlib.cm as cm
colors = np.zeros([20,4])
for i in np.arange(20):
    c = cm.spectral(i/20.,1)
    colors[i,:]=c
points = ['o','v','s','p','*','h','^','D','+','>','H','d','x','<']
        
plt.rc('axes',linewidth=2.0)
plt.rc('xtick.major',width=2.0)
plt.rc('ytick.major',width=2.0)
plt.rc('xtick.minor',width=2.0)
plt.rc('ytick.minor',width=2.0)
plt.rc('lines',markersize=8,markeredgewidth=0.0,linewidth=2.0)
#plt.rcParams['ps.fonttype'] = 3
fig=plt.figure(num=1,figsize=(7,6),dpi=600,facecolor='w',edgecolor='k')
left  = 0.15  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.1   # the amount of width reserved for blank space between subplots
hspace = 0.17   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

timeindex=delayindex*dt
ax1=plt.subplot(1,1,1)
plt.semilogx(timeindex,SCsx1_Lp1,color='black',marker=points[0],markevery=(0.2),label='L=0.1')
plt.semilogx(timeindex,SCsx1_Lp5,color='blue',marker=points[1],markevery=(0.2),label='L=0.5')
plt.semilogx(timeindex,SCsx1_L1,color='green',marker=points[2],markevery=(0.2),label='L=1.0')
plt.semilogx(timeindex,SCsx1_L10,color='red',marker=points[3],markevery=(0.2),label='L=10.0')
plt.semilogx(timeindex,SCsx1_L100,color='orange',marker=points[4],markevery=(0.2),label='L=100.0')




#plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)
plt.yticks(fontsize=16)
plt.xticks(fontsize=15)
plt.xlabel(r'$\tau_s [s]$',fontsize=15)
#ax1.set_xticklabels([])
plt.ylabel(r'$C$',fontsize=15)
#plt.xlim(1,250)
plt.ylim(0,0.35)
plt.legend(loc='lower left',fontsize=13,frameon=False,handlelength=5)

savefilename='SC_DPx1_ICC1_Lscan_0p1to100_3.eps'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()
"""