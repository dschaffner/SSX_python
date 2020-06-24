# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:00:45 2017

@author: dschaffner
"""
import numpy as np
import matplotlib.pylab as plt
from loadnpzfile import loadnpzfile
from calc_PE_SC import PE, CH, PE_dist, PE_calc_only
#import Cmaxmin as cpl
from collections import Counter
from math import factorial
import matplotlib.cm as cm
import os

#datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\GalpyData_July2018\\'
datadir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Galatic Dynamics Data\\Examples\\'
savedir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Galatic Dynamics Data\\Examples\\'

npz='.npz'

delayindex = np.arange(1,501)#250
timestep_arr = [2050]
print(len(timestep_arr))
timeindex = (delayindex*1e5)/(1e6)

fileheader = 'PESC_noise100k_embed5_1000_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEs_noise = datafile['PEs']
SCs_noise = datafile['SCs']
taus_noise = datafile['taus']
timeseries_noise = datafile['timeseries']

fileheader = 'PESC_henon100k_embed5_1000_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEs_henon = datafile['PEs']
SCs_henon = datafile['SCs']
taus_henon = datafile['taus']
timeseries_henon = datafile['timeseries']

fileheader = 'PESC_sine1x_embed5_1000_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEs_sine1x = datafile['PEs']
SCs_sine1x = datafile['SCs']
taus_sine1x = datafile['taus']
timeseries_sine1x = datafile['timeseries']

fileheader = 'PESC_sine20x_embed5_1000_delays'
datafile = loadnpzfile(datadir+fileheader+npz)
PEs_sine20x = datafile['PEs']
SCs_sine20x = datafile['SCs']
taus_sine20x = datafile['taus']
timeseries_sine20x = datafile['timeseries']

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




points = ['o','v','s','p','*','h','^','D','+','>','H','d','x','<']
        
plt.rc('axes',linewidth=1.0)
plt.rc('xtick.major',width=1.0)
plt.rc('ytick.major',width=1.0)
plt.rc('xtick.minor',width=1.0)
plt.rc('ytick.minor',width=1.0)
plt.rc('lines',markersize=8,markeredgewidth=0.0,linewidth=1.0)

#plt.rcParams['ps.fonttype'] = 42
#plt.rcParams['pdf.fonttype'] = 42

fig=plt.figure(num=1,figsize=(4,6),dpi=300,facecolor='w',edgecolor='k',tight_layout=True)
plt.clf()
left  = 0.1  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.12  # the bottom of the subplots of the figure
top = 0.98      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

import matplotlib.gridspec as gridspec
gs=gridspec.GridSpec(2,3)

ax1=fig.add_subplot(gs[0,:])
plt.plot(x2M[:,1],color='blue',label='m=2')
plt.plot(x5M[:,1]+4,color='purple',label='m=5')
plt.plot(x10M[:,1]+8,color='red',label='m=10')
plt.plot(x20M[:,1]+12,color='orange',label='m=20')
plt.plot(timeseries_noise+16,color='gray',label='white noise')
plt.xlabel('Time',fontsize=8)
plt.ylabel('Amplitude',fontsize=8)
ax1.set_xticklabels([])
ax1.set_xticks([])
ax1.set_yticklabels([])
ax1.set_yticks([])
plt.xlim([1000,3000])
plt.ylim([-2,20])
plt.title('(a) Timeseries',fontsize=5)
plt.legend(loc='upper center',fontsize=4,ncol=2,frameon=False,handlelength=5)

ax2=fig.add_subplot(gs[1,0])
plt.plot(SCs_noise[1:],color='blue',label='White Noise')
plt.plot(SCs_henon[1:],color='purple',label='Henon Map')
plt.plot(SCs_sine1x[1:],color='red',label='sin(x)')
plt.plot(SCs_sine20x[1:],color='orange',label='sin(20x')
plt.xlim(0,25)
plt.xlabel(r'$\tau$',fontsize=6)
plt.xticks([0,5,10,15,20,25],[0,5,10,15,20,25],fontsize=5)
plt.ylabel('C',fontsize=6)
plt.yticks(fontsize=5)
plt.title('(b) Complexity vs Delay',fontsize=5)

ax3=fig.add_subplot(gs[1,1])
plt.loglog(SCs_noise,color='blue',label='White Noise')
plt.loglog(SCs_henon,color='purple',label='Henon Map')
plt.loglog(SCs_sine20x,color='orange',label='sin(20x')
plt.loglog(SCs_sine1x,color='red',label='sin(x)')
plt.xlabel(r'$\tau$',fontsize=6)
plt.xticks([1e0,1e1,1e2,1e3],[1e0,1e1,1e2,1e3],fontsize=5)
plt.xlim(1e0,1e3)
plt.yticks([0.0001,0.001,0.01,0.1,0.4],[0.0001,0.001,0.01,0.1,0.4],fontsize=5)
plt.ylim(1e-4,0.5)
plt.ylabel('C',fontsize=6)
plt.title('(c) Log-Log Complexity vs Delay',fontsize=5)

ax4=fig.add_subplot(gs[1,2])
plt.loglog(PEs_noise,color='blue',label='White Noise')
plt.loglog(PEs_henon,color='purple',label='Henon Map')
plt.loglog(PEs_sine20x,color='orange',label='sin(20x')
plt.loglog(PEs_sine1x,color='red',label='sin(x)')
plt.xlabel(r'$\tau$',fontsize=6)
plt.xticks([1e0,1e1,1e2,1e3],[1e0,1e1,1e2,1e3],fontsize=5)
plt.xlim(1e0,1e3)
plt.yticks([0.1,0.4,1.0],[0.1,0.4,1.0],fontsize=5)
#plt.ylim(0.1,1.0)
plt.ylabel('H',fontsize=6)
plt.title('(d) Log-Log Norm. PE vs Delay',fontsize=5)


"""
plt.plot(timeindex_normM6,SCs1[1:500],color='blue',marker=points[0],markevery=(20,100),label='Type 1')
#plt.plot(timeindex,SCs1_resort[1:],color='blue',linestyle='dashed')
#plt.plot(timeindex,SCsT25[1:500],color=colors[1,:],label='Type 2 [Beyond CR]')
#plt.plot(timeindex,SCs31[1:],color='green',label='Type 3-1')
plt.plot(timeindex_normM6,SCs32i[1:],color='red',marker=points[1],markevery=(20,100),label=r'Type 3$\rightarrow$2i')
#plt.plot(timeindex,SCs32_resort[1:],color='red',linestyle='dashed')
#plt.plot(timeindex,SCs31[1:],color='green')
#plt.plot(timeindex,SCs31_resort[1:],color='green',linestyle='dashed')
#plt.plot(timeindex,SCs32_4CRresort[1:],color='purple')
#plt.plot(timeindex,SCs4[1:],color='purple',label='Type 4')
#plt.plot(delayindex,SCsin[1:],color='black',label='Sine Wave')

#plt.vlines(85,0,1,color='red',linestyle='dotted',linewidth=0.5)

#plt.xticks(np.array([1,20,40,60,80,100,120,140,160,180,200,220,240]),[1,20,40,60,80,100,120,140,160,180,200,220,240],fontsize=9)

plt.xticks(fontsize=15)
plt.yticks(np.array([0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40]),[0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40],fontsize=12)
plt.xlabel(r'$\tau_s/\tau_{dyn}$',fontsize=18)
#plt.xlabel('Delay Steps',fontsize=9)
ax1.set_xticklabels([])
plt.ylabel(r'$C$',fontsize=18)
plt.xlim(0,0.25)
plt.ylim(0.11,0.4)
plt.legend(loc='lower right',fontsize=12,frameon=False,handlelength=5,numpoints=2)
plt.text(0.04,0.95,'(a)',fontsize=16,horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes)

#savefilename='SC_galpy0718_1000timesteps_3000_orbits.png'
#savefilename='SC_galpy0718_1000timesteps_all_orbits.png'
#savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')


#fig=plt.figure(num=2,figsize=(5,3.5),dpi=300,facecolor='w',edgecolor='k')
#plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(2,1,2)
plt.plot(timeindex_normM6,PEs1[1:500],color='blue',marker=points[0],markevery=(20,100),label='Type 1')
#plt.plot(timeindex,PEsT25[1:500],color=colors[1,:],label='Type 2 [Beyond CR]')
#plt.plot(timeindex,PEs31[1:],color='green',label='Type 3-1')
plt.plot(timeindex_normM6,PEs32i[1:],color='red',marker=points[1],markevery=(20,100),label=r'Type 3$\rightarrow$2i')
#plt.plot(timeindex,PEs4[1:],color='purple',label='Type 4')
#plt.plot(delayindex,PEsin[1:],color='black',label='Sine Wave')

plt.xticks(fontsize=15)
plt.xlabel(r'$\tau_s/\tau_{dyn}$',fontsize=18)
#ax1.set_xticklabels([])
plt.yticks(fontsize=15)
plt.ylabel(r'$H$',fontsize=15)
plt.xlim(0,0.25)
plt.ylim(0,0.95)
plt.legend(loc='lower right',fontsize=15,frameon=False,handlelength=5)
plt.text(0.04,0.95,'(b)',fontsize=16,horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes)

#savefilename='SC_and_PE_CR6_2000timesteps_Type1vsType32i_normdyntime_ApJver.png'
savefilename='SC_and_PE_CR6_2000timesteps_Type1vsType32i_normdyntime_ApJver.eps'
#savefilename='PE_galpy0718_1000timesteps_all_orbits.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()
"""
"""
ndim=5
#Plot C vs H as a CHplane
Cminx, Cminy, Cmaxx, Cmaxy = cpl.Cmaxmin(1000,ndim)
plt.rc('lines',markersize=5,markeredgewidth=0.0)

fig=plt.figure(num=33,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.25  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.97      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

timestep = 86

plt.plot(Cminx,Cminy,'k-',Cmaxx,Cmaxy,'k-')
plt.plot(PEs1[timestep],SCs1[timestep],color=colors[0,:],marker='o',label='Type 1')
plt.plot(PEsT25[timestep],SCsT25[timestep],color=colors[1,:],marker='o',label='Type 2')
plt.plot(PEs31[timestep],SCs31[timestep],color='green',marker='o',label='Type 3-1')
plt.plot(PEs32[timestep],SCs32[timestep],color='red',marker='o',label='Type 3-2')
plt.plot(PEs4[timestep],SCs4[timestep],color='purple',marker='o',label='Type 4')
#plt.plot(PEsin[81],SCsin[81],color='black',marker='o',label='Sine Wave, Delay 81')

timestep0 = 1
plt.plot(PEs1[timestep0],SCs1[timestep0],color=colors[0,:],marker='v')
plt.plot(PEsT25[timestep0],SCsT25[timestep0],color=colors[1,:],marker='v')
plt.plot(PEs31[timestep0],SCs31[timestep0],color='green',marker='v')
plt.plot(PEs32[timestep0],SCs32[timestep0],color='red',marker='v')
plt.plot(PEs4[timestep0],SCs4[timestep0],color='purple',marker='v')

plt.xlabel("Permutation Entropy", fontsize=9)
plt.ylabel("Statistical Complexity", fontsize=9)
#plt.title('Delay Timescale '+str(timeindex[timestep])+' Myr',fontsize=9)
plt.axis([0,1.0,0,0.45])
plt.xticks(np.arange(0,1.1,0.1),fontsize=9)
plt.yticks(np.arange(0,0.45,0.05),fontsize=9)
leg=plt.legend(loc='lower center',fontsize=5,frameon=False,handlelength=0)
leg.set_title(r'At Delay $\tau=$'+str(timeindex[timestep])+ 'Myr',prop={'size':4})

savefilename='CH_galpy0718_2000plustimesteps_timestep'+str(timestep)+'_3000plusorbits.eps'
#savefilename='CH_galpy0718_1000timesteps_timestep'+str(timestep)+'_all_orbits.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
"""
"""
fig=plt.figure(num=4,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.25  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.90      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


plt.plot(Cminx,Cminy,'k-',Cmaxx,Cmaxy,'k-')
plt.plot(PEs1,SCs1,color=colors[0,:],label='Type 1')
plt.plot(PEs2,SCs2,color=colors[1,:],label='Type 2')
plt.plot(PEs31,SCs31,color='green',label='Type 3-1')
plt.plot(PEs32,SCs32,color='red',label='Type 3-2')
plt.plot(PEs4,SCs4,color='purple',label='Type 4')
#plt.plot(PEsin[81],SCsin[81],color='black',marker='o',label='Sine Wave, Delay 81')
plt.xlabel("Permutation Entropy", fontsize=9)
plt.ylabel("Complexity", fontsize=9)
plt.axis([0,1.0,0,0.45])
plt.xticks(np.arange(0,1.1,0.1),fontsize=9)
plt.yticks(np.arange(0,0.45,0.05),fontsize=9)
plt.legend(loc='lower center',fontsize=5,frameon=False,handlelength=0)

savefilename='CH_galpy0718_1000timesteps_timestep_3000_orbits.png'
savefilename='CH_galpy0718_1000timesteps_timestep_all_orbits.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')
"""



"""
colors = np.zeros([5,4])
for i in np.arange(5):
    c = cm.spectral(i/5.,1)
    colors[i,:]=c
points = ['o','v','s','p','*','h','^','D','+','>','H','d','x','<']
        
plt.rc('axes',linewidth=0.75)
plt.rc('xtick.major',width=0.75)
plt.rc('ytick.major',width=0.75)
plt.rc('xtick.minor',width=0.75)
plt.rc('ytick.minor',width=0.75)
plt.rc('lines',markersize=2,markeredgewidth=0.0)

plt.rc('lines',markersize=1.5,markeredgewidth=0.0)
fig=plt.figure(num=1,figsize=(5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.15  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.2  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
plt.plot(timeindex,SCs32[1:],color=colors[0,:],label='Type 32 first 3000')
plt.plot(timeindex,SCs32_2[1:],color=colors[1,:],label='Type 32 second 3000')
plt.plot(timeindex,SCs32_3[1:],color='green',label='Type 32 third 3000')
#plt.plot(delayindex,SCsin[1:],color='black',label='Sine Wave')

plt.vlines(9.19,0,1,color=colors[0,:],linestyle='dotted',linewidth=0.5)
plt.vlines(9.09,0,1,color=colors[1,:],linestyle='dotted',linewidth=0.5)
plt.vlines(9.4,0,1,color='green',linestyle='dotted',linewidth=0.5)
delayarray = np.array([1,20,40,60,80,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,110,120,140,160,180,200,220,240])
timearray = (delayarray*1e5)/(1e6)
timelist = list(timearray)
plt.xticks(timearray,timelist,fontsize=6)
plt.yticks(fontsize=9)
plt.xlabel('Myr',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Complexity',fontsize=9)
plt.xlim(8.5,10.5)
plt.ylim(0.34,0.36)
plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)

savefilename='SC_galpy0718_1000timesteps_3000_orbits_sepranges.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')
"""

"""
colors = np.zeros([5,4])
for i in np.arange(5):
    c = cm.spectral(i/5.,1)
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
left  = 0.15  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.2  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

delayindex = np.arange(1,750)
timeindex=(delayindex*1e5)/(1e6)

ax1=plt.subplot(1,1,1)
plt.plot(timeindex,SCsT21[1:],color='black',label='Type 2 (NT) [0,3.5)')
plt.plot(timeindex,SCsT22[1:],color='blue',label='Type 2 (NT) [3.5,4)')
plt.plot(timeindex,SCsT23[1:],color='green',label='Type 2 (NT) [4,4.5)')
plt.plot(timeindex,SCsT24[1:],color='red',label='Type 2 (NT) [4.5,5.5)')
plt.plot(timeindex,SCsT25[1:],color='purple',label='Type 2 (NT) [7,9]')

#delayarray = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250])
delayarray = np.array([0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500,520,540,560,580,600,620,640,660,680,700,720,740])

timearray = (delayarray*1e5)/(1e6)
timelist = list(timearray.astype(int))
plt.xticks(timearray,timelist,fontsize=8)
plt.yticks(np.array([0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40]),[0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40],fontsize=9)
plt.xlabel('Delay Time [Myr]',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Statistical Complexity',fontsize=9)
#plt.xlim(8.5,10.5)
#plt.ylim(0.34,0.36)
plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)

savefilename='SC_galpy0718_4000timesteps_Type2_diffIC.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')
"""