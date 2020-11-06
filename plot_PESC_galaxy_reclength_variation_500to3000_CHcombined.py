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
import matplotlib.gridspec as gridspec
import os

m6_dyntime_index=1676
dyntime=m6_dyntime_index/10

datadir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Galatic Dynamics Data\\GalpyData_July2018\\resorted_data\\M6_3352_et_timescan\\'#\\type32\\'
npy='.npz'

delayindex = np.arange(1,501)#250
timestep_arr = [500,750,1000,1300,1350,1400,1450,1500,1550,1600,1650,1676,1700,1750,1800,1850,1900,1950,2000,2050,2100,2150,2200,2500,3352]#,2350,2400,2450,2500,2750,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000]
print(len(timestep_arr))
timeindex = (delayindex*1e5)/(1e6)
normtimeindex=timeindex/dyntime

#CR4 -> 32
#CR6 -> 42
#CR7 -> 42
#CR8 -> 53
PEs_32 = np.zeros([int(len(timestep_arr)),500])
SCs_32 = np.zeros([int(len(timestep_arr)),500])
for file in np.arange(len(timestep_arr)):
    fileheader = 'PE_SC_Type_32i_Rg_499_delays_3145orbits_of3899_total'+str(timestep_arr[file])+'_timesteps_resorted_et'\
    #fileheader = 'PE_SC_Type_32o_Rg_499_delays_2021orbits_of2458_total'+str(timestep_arr[file])+'_timesteps_resorted_et'
    datafile = loadnpzfile(datadir+fileheader+npy)
    PEs_32[file,:]=datafile['PEs']
    SCs_32[file,:]=datafile['SCs']




orbitdur1=np.round(750/1676,1)
orbitdur2=np.round(1000/1676,1)
orbitdur3=np.round(1676/1676,1)
orbitdur4=np.round(2000/1676,1)
orbitdur5=np.round(3352/1676,1)    

numcolors=5
#colors = np.zeros([int(len(timestep_arr))+1,4])
#for i in np.arange(int(len(timestep_arr))+1):
#    c = plt.cm.Plasma(i/(int(len(timestep_arr))+1),1)
#    colors[i,:]=c

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
plt.rc('lines',markersize=2,markeredgewidth=0.0,linewidth=2.5)
xaxis_label_fontsize=15
xaxis_ticks_size=12
yaxis_label_fontsize=18
yaxis_ticks_size=12
title_fontsize=15
legend_fontsize=12
legend_title_fontsize=9
#plt.rcParams['ps.fonttype'] = 42
#plt.rcParams['pdf.fonttype'] = 42

#figsize=(width,height)
fig=plt.figure(num=1,figsize=(7,5),dpi=600,facecolor='w',edgecolor='k')
left  = 0.16  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.15  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.0   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

SCs_32_500_endarray=np.where(SCs_32[0,1:]==0)[0][0]
SCs_32_750_endarray=np.where(SCs_32[1,1:]==0)[0][0]
SCs_32_1000_endarray=np.where(SCs_32[2,1:]==0)[0][0]
SCs_32_1300_endarray=np.where(SCs_32[3,1:]==0)[0][0]
SCs_32_1350_endarray=np.where(SCs_32[4,1:]==0)[0][0]
SCs_32_1400_endarray=np.where(SCs_32[5,1:]==0)[0][0]
SCs_32_1450_endarray=np.where(SCs_32[6,1:]==0)[0][0]
SCs_32_1500_endarray=np.where(SCs_32[7,1:]==0)[0][0]
SCs_32_1550_endarray=np.where(SCs_32[8,1:]==0)[0][0]
SCs_32_1600_endarray=np.where(SCs_32[9,1:]==0)[0][0]
SCs_32_1650_endarray=np.where(SCs_32[10,1:]==0)[0][0]
SCs_32_1676_endarray=np.where(SCs_32[11,1:]==0)[0][0]
SCs_32_1700_endarray=np.where(SCs_32[12,1:]==0)[0][0]
SCs_32_1750_endarray=np.where(SCs_32[13,1:]==0)[0][0]
SCs_32_1800_endarray=np.where(SCs_32[14,1:]==0)[0][0]
SCs_32_1850_endarray=np.where(SCs_32[15,1:]==0)[0][0]
SCs_32_1900_endarray=np.where(SCs_32[16,1:]==0)[0][0]
SCs_32_1950_endarray=np.where(SCs_32[17,1:]==0)[0][0]
SCs_32_2000_endarray=500#np.where(SCs_32[18,1:]==0)[0][0]
SCs_32_2050_endarray=500#np.where(SCs_32[18,1:]==0)[0][0]
SCs_32_2100_endarray=500#np.where(SCs_32[19,1:]==0)[0][0]
SCs_32_2150_endarray=500#np.where(SCs_32[20,1:]==0)[0][0]
SCs_32_2200_endarray=500#np.where(SCs_32[21,1:]==0)[0][0]
SCs_32_2500_endarray=500#np.where(SCs_32[22,1:]==0)[0][0]
SCs_32_3352_endarray=500#np.where(SCs_32[23,1:]==0)[0][0]

ax1=plt.subplot(1,1,1)
plt.plot(normtimeindex[:SCs_32_750_endarray-1],SCs_32[1,1:SCs_32_750_endarray],color=colors[0,:],linestyle='solid',markevery=(20,100),label=str(orbitdur1)+r' $\tau_{dyn}^{M6}$')
plt.plot(normtimeindex[:SCs_32_1000_endarray-1],SCs_32[2,1:SCs_32_1000_endarray],color=colors[1,:],linestyle='solid',label=str(orbitdur2)+r' $\tau_{dyn}^{M6}$')
plt.plot(normtimeindex[:SCs_32_1676_endarray-1],SCs_32[13,1:SCs_32_1676_endarray],color=colors[2,:],linestyle='solid',label=str(orbitdur3)+r' $\tau_{dyn}^{M6}$')
plt.plot(normtimeindex[:SCs_32_2000_endarray-1],SCs_32[18,1:SCs_32_2000_endarray],color=colors[3,:],linestyle='solid',label=str(orbitdur4)+r' $\tau_{dyn}^{M6}$')
plt.plot(normtimeindex[:SCs_32_3352_endarray-1],SCs_32[23,1:SCs_32_3352_endarray],color=colors[4,:],linestyle='solid',label=str(orbitdur5)+r' $\tau_{dyn}^{M6}$')

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

#plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)

delayarray = np.array([0,40,80,120,160,200,240,280,320,360,400,440,480,520])
timearray = (delayarray*1e5)/(1e6)
normtimearray=timearray/dyntime
timelist = list(timearray.astype(int))
#plt.xticks(timearray,timelist,fontsize=12)
plt.xticks(normtimearray,fontsize=xaxis_ticks_size)
plt.xlabel(r'$\tau_s/T_{dyn}$',fontsize=xaxis_label_fontsize)
#ax1.set_xticklabels([])
#plt.title(r'Type 3$\rightarrow$2i - M6',fontsize=title_fontsize)
#plt.yticks(np.array([0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5]),[0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5],fontsize=yaxis_ticks_size)
plt.ylabel(r'$C$ (solid), $H$ (dashed)',fontsize=yaxis_label_fontsize)
#plt.xlim(0,0.25)
#plt.ylim(0.11,0.4)
leg=plt.legend(loc='lower right',fontsize=9,frameon=False,handlelength=5,numpoints=20,ncol=2)
leg.set_title('Orbit Duration',prop={'size':9})
#plt.text(0.07,0.92,'(a)',horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes,fontsize=16)
#plt.vlines(9.3/dyntime,0,1,color='red',linestyle='dotted',linewidth=0.5)

#plt.plot(timeindex[:SCs_32_500_endarray-1],PEs_32[0,1:SCs_32_500_endarray],color=colors[0,:],label='50 Myr')
plt.plot(normtimeindex[:SCs_32_750_endarray-1],PEs_32[1,1:SCs_32_750_endarray],color=colors[0,:],linestyle='dashed',label=str(orbitdur1)+r'$\tau_s/T^{M6}_{dyn}$')
plt.plot(normtimeindex[:SCs_32_1000_endarray-1],PEs_32[2,1:SCs_32_1000_endarray],color=colors[1,:],linestyle='dashed',label=str(orbitdur2)+r'$\tau_s/T^{M6}_{dyn}$')
plt.plot(normtimeindex[:SCs_32_1676_endarray-1],PEs_32[13,1:SCs_32_1676_endarray],color=colors[2,:],linestyle='dashed',label=str(orbitdur3)+r'$\tau_s/T^{M6}_{dyn}$')
plt.plot(normtimeindex[:SCs_32_2000_endarray-1],PEs_32[18,1:SCs_32_2000_endarray],color=colors[3,:],linestyle='dashed',label=str(orbitdur4)+r'$\tau_s/T^{M6}_{dyn}$')
plt.plot(normtimeindex[:SCs_32_3352_endarray-1],PEs_32[23,1:SCs_32_3352_endarray],color=colors[4,:],linestyle='dashed',label=str(orbitdur5)+r'$\tau_s/T^{M6}_{dyn}$')

#plt.xticks(timearray,timelist,fontsize=12)
#plt.xticks(normtimearray,list(np.round(normtimearray,2)),fontsize=12)
#plt.xticks(normtimearray,list(np.round(normtimearray,2)),fontsize=12)
plt.xticks(np.array([0.0,0.05,0.1,0.15,0.2,0.25,0.3]),[0.0,0.05,0.10,0.15,0.20,0.25,0.30],fontsize=xaxis_ticks_size)
#plt.xlabel(r'$\tau_s$ [Myr]',fontsize=15)
plt.xlabel(r'$\tau_s/T^{M6}_{dyn}$',fontsize=xaxis_label_fontsize)

plt.yticks(fontsize=yaxis_ticks_size)
#plt.ylabel(r'$H$',fontsize=yaxis_label_fontsize)
plt.xlim(0,0.25)
plt.ylim(0.0,1.0)
#leg=plt.legend(loc='lower right',fontsize=legend_fontsize,frameon=False,handlelength=5,numpoints=20)
#leg.set_title('Orbit Duration',prop={'size':legend_title_fontsize})
#plt.text(0.07,0.92,'(b)',horizontalalignment='center',verticalalignment='center',transform=ax2.transAxes,fontsize=16)
#plt.vlines(9.3/dyntime,0,1,color='red',linestyle='dotted',linewidth=0.5)


savefilename='SC_and_PE_recordlengthvariation_M6_type32i_normdyntime_CHcombined.png'
#savefilename='SC_and_PE_recordlengthvariation_M6_type32i_normdyntime_ApJver_newcolor.eps'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()



























timestep_arr = [750,1000,1676,2000,3352]#,2350,2400,2450,2500,2750,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000]

PEs_1 = np.zeros([int(len(timestep_arr)),500])
SCs_1 = np.zeros([int(len(timestep_arr)),500])
for file in np.arange(len(timestep_arr)):
    fileheader = 'PE_SC_Type_1_Rg_499_delays_3910orbits_of3910_total'+str(timestep_arr[file])+'_timesteps_resorted_et'
    #fileheader = 'PE_SC_Type_32o_Rg_499_delays_2021orbits_of2458_total'+str(timestep_arr[file])+'_timesteps_resorted_et'
    datafile = loadnpzfile(datadir+fileheader+npy)
    PEs_1[file,:]=datafile['PEs']
    SCs_1[file,:]=datafile['SCs']


fig=plt.figure(num=2,figsize=(7,9),dpi=600,facecolor='w',edgecolor='k')
left  = 0.16  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.08  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.0   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

SCs_1_750_endarray=np.where(SCs_32[0,1:]==0)[0][0]
SCs_1_1000_endarray=np.where(SCs_32[1,1:]==0)[0][0]
SCs_1_1676_endarray=np.where(SCs_32[2,1:]==0)[0][0]
SCs_1_2000_endarray=500#np.where(SCs_32[18,1:]==0)[0][0]
SCs_1_3352_endarray=500#np.where(SCs_32[23,1:]==0)[0][0]

ax1=plt.subplot(2,1,1)
plt.plot(normtimeindex[:SCs_1_750_endarray-1],SCs_1[0,1:SCs_1_750_endarray],color=colors[0,:],linestyle='solid',markevery=(20,100),label=str(orbitdur1)+r' $\tau_{dyn}^{M6}$')
plt.plot(normtimeindex[:SCs_1_1000_endarray-1],SCs_1[1,1:SCs_1_1000_endarray],color=colors[1,:],linestyle='solid',label=str(orbitdur2)+r' $\tau_{dyn}^{M6}$')
plt.plot(normtimeindex[:SCs_1_1676_endarray-1],SCs_1[2,1:SCs_1_1676_endarray],color=colors[2,:],linestyle='solid',label=str(orbitdur3)+r' $\tau_{dyn}^{M6}$')
plt.plot(normtimeindex[:SCs_1_2000_endarray-1],SCs_1[3,1:SCs_1_2000_endarray],color=colors[3,:],linestyle='solid',label=str(orbitdur4)+r' $\tau_{dyn}^{M6}$')
plt.plot(normtimeindex[:SCs_1_3352_endarray-1],SCs_1[4,1:SCs_1_3352_endarray],color=colors[4,:],linestyle='solid',label=str(orbitdur5)+r' $\tau_{dyn}^{M6}$')

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

#plt.xticks(np.array([1,40,80,120,160,200,240]),[1,40,80,120,160,200,240],fontsize=9)

delayarray = np.array([0,40,80,120,160,200,240,280,320,360,400,440,480,520])
timearray = (delayarray*1e5)/(1e6)
normtimearray=timearray/dyntime
timelist = list(timearray.astype(int))
#plt.xticks(timearray,timelist,fontsize=12)
plt.xticks(normtimearray,fontsize=xaxis_ticks_size)
plt.xlabel(r'$\tau_s/T^{M6}_{dyn}$',fontsize=xaxis_label_fontsize)
ax1.set_xticklabels([])
plt.title('Type 1 - M6',fontsize=title_fontsize)
plt.yticks(np.array([0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5]),[0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.5],fontsize=yaxis_ticks_size)
plt.ylabel(r'$C$',fontsize=yaxis_label_fontsize)
plt.xlim(0,0.25)
plt.ylim(0.11,0.4)
leg=plt.legend(loc='lower right',fontsize=legend_fontsize,frameon=False,handlelength=5,numpoints=20)
leg.set_title('Orbit Duration',prop={'size':legend_title_fontsize})
plt.text(0.07,0.92,'(a)',horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes,fontsize=16)
plt.vlines(9.3/dyntime,0,1,color='red',linestyle='dotted',linewidth=0.5)



ax2=plt.subplot(2,1,2)
#plt.plot(timeindex[:SCs_32_500_endarray-1],PEs_32[0,1:SCs_32_500_endarray],color=colors[0,:],label='50 Myr')
plt.plot(normtimeindex[:SCs_1_750_endarray-1],PEs_1[0,1:SCs_1_750_endarray],color=colors[0,:],linestyle='solid',label=str(orbitdur1)+r' $\tau_{dyn}^{M6}$')
plt.plot(normtimeindex[:SCs_1_1000_endarray-1],PEs_1[1,1:SCs_1_1000_endarray],color=colors[1,:],linestyle='solid',label=str(orbitdur2)+r' $\tau_{dyn}^{M6}$')
plt.plot(normtimeindex[:SCs_1_1676_endarray-1],PEs_1[2,1:SCs_1_1676_endarray],color=colors[2,:],linestyle='solid',label=str(orbitdur3)+r' $\tau_{dyn}^{M6}$')
plt.plot(normtimeindex[:SCs_1_2000_endarray-1],PEs_1[3,1:SCs_1_2000_endarray],color=colors[3,:],linestyle='solid',label=str(orbitdur4)+r' $\tau_{dyn}^{M6}$')
plt.plot(normtimeindex[:SCs_1_3352_endarray-1],PEs_1[4,1:SCs_1_3352_endarray],color=colors[4,:],linestyle='solid',label=str(orbitdur5)+r' $\tau_{dyn}^{M6}$')

#plt.xticks(timearray,timelist,fontsize=12)
#plt.xticks(normtimearray,list(np.round(normtimearray,2)),fontsize=12)
#plt.xticks(normtimearray,list(np.round(normtimearray,2)),fontsize=12)
plt.xticks(np.array([0.0,0.05,0.1,0.15,0.2,0.25,0.3]),[0.0,0.05,0.10,0.15,0.20,0.25,0.30],fontsize=xaxis_ticks_size)
#plt.xlabel(r'$\tau_s$ [Myr]',fontsize=15)
plt.xlabel(r'$\tau_s/T^{M6}_{dyn}$',fontsize=xaxis_label_fontsize)

plt.yticks(fontsize=yaxis_ticks_size)
plt.ylabel(r'$H$',fontsize=yaxis_label_fontsize)
plt.xlim(0,0.25)
plt.ylim(0.0,0.95)
leg=plt.legend(loc='lower right',fontsize=legend_fontsize,frameon=False,handlelength=5,numpoints=20)
leg.set_title('Orbit Duration',prop={'size':legend_title_fontsize})
plt.text(0.07,0.92,'(b)',horizontalalignment='center',verticalalignment='center',transform=ax2.transAxes,fontsize=16)
plt.vlines(9.3/dyntime,0,1,color='red',linestyle='dotted',linewidth=0.5)


savefilename='SC_and_PE_recordlengthvariation_M6_type1_normdyntime_ApJver_newcolor.png'
#savefilename='SC_and_PE_recordlengthvariation_M6_type1_normdyntime_ApJver_newcolor.eps'
savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()









#CH plane plot with color gradient

fig=plt.figure(num=3,figsize=(9,6),dpi=600,facecolor='w',edgecolor='k')
left  = 0.16  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.15  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.0   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
spec2 = gridspec.GridSpec(ncols=8, nrows=1, figure=fig)

ax1 = fig.add_subplot(spec2[:, 0:7])
points = ['o','v','s','p','*','h','^','D','+','>','H','d','x','<']
import Cmaxmin as cpl
ndim=5
#Plot C vs H as a CHplane
Cminx, Cminy, Cmaxx, Cmaxy = cpl.Cmaxmin(1000,ndim)
plt.rc('lines',markersize=2,markeredgewidth=0.0)
plt.plot(Cminx,Cminy,'k-',Cmaxx,Cmaxy,'k-')


plt.rc('lines',markersize=2,markeredgewidth=0.0)
numpoints=len(SCs_32[13,1:418])
plt.scatter(PEs_32[13,1:418],SCs_32[13,1:418],marker='o',color=cm.plasma(np.arange(numpoints)),label=str(orbitdur1)+r'$\tau_s/T^{M6}_{dyn}$')
numpoints=len(SCs_1[2,1:418])
plt.scatter(PEs_1[2,1:418],SCs_1[2,1:418],marker='d',color=cm.plasma(np.arange(numpoints)),label=str(orbitdur1)+r'$\tau_s/T^{M6}_{dyn}$')

plt.xticks(fontsize=xaxis_ticks_size)
plt.xlabel(r'$H$',fontsize=xaxis_label_fontsize)
plt.yticks(fontsize=yaxis_ticks_size)
plt.ylabel(r'$C$',fontsize=yaxis_label_fontsize)

ax2=fig.add_subplot(spec2[:,7])
import matplotlib as mpl
cmap=mpl.cm.plasma
norm=mpl.colors.Normalize(vmin=0.0,vmax=0.25)
cb1=mpl.colorbar.ColorbarBase(ax2,cmap=cmap,norm=norm,orientation='vertical')
cb1.set_label(r'$\tau_s/T^{M6}_{dyn}$',fontsize=15)

savefilename='CH_trajectory_comparison_test.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()