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

fileheader = '4CR_timescan\\PE_SC_Type_32_4CR_3000_Rg_499_delays_7324orbits_1400_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEsCR4 = datafile['PEs']
SCsCR4 = datafile['SCs']
SCsCR4_endarray=np.where(SCsCR4[1:]==0)[0][0]

fileheader = '6CR_timescan\\type32\\PE_SC_Type_32_6CR_3000_Rg_499_delays_4318orbits_2000_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEsCR6 = datafile['PEs']
SCsCR6 = datafile['SCs']
SCsCR6_endarray=500

fileheader = '7CR_timescan\\PE_SC_Type_32_7CR_3000_Rg_499_delays_6058orbits_2750_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEsCR7 = datafile['PEs']
SCsCR7 = datafile['SCs']
SCsCR7_endarray=500

fileheader = '8CR_timescan\\PE_SC_Type_32_8CR_4000_Rg_499_delays_2870orbits_4000_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEsCR8 = datafile['PEs']
SCsCR8 = datafile['SCs']
SCsCR8_endarray=500

datadir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Galatic Dynamics Data\\GalpyData_July2018\\'
fileheader = 'PE_SC_IDdatabase_Type_32_10co_data_8000_499_delays_329orbits_8000_timesteps'
datafile = loadnpzfile(datadir+fileheader+npy)
PEsCR10 = datafile['PEs']
SCsCR10 = datafile['SCs']
SCsCR10_endarray=500



#colors = np.zeros([5,4])
#for i in np.arange(5):
#    c = cm.spectral(i/5.,1)
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

fig=plt.figure(num=1,figsize=(7,9),dpi=600,facecolor='w',edgecolor='k')
left  = 0.16  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.08  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.0   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

delayindex = np.arange(1,500)
timeindex=(delayindex*1e5)/(1e6)

ax1=plt.subplot(2,1,1)
plt.plot(timeindex[:SCsCR4_endarray-1],SCsCR4[1:SCsCR4_endarray],marker=points[1],markevery=50,linestyle='dashed',color='blue',label='M4 (140 Myr)')
plt.plot(timeindex,SCsCR6[1:],marker=points[1],markevery=50,color='red',linestyle='solid',label='M6 (200 Myr)')
plt.plot(timeindex,SCsCR7[1:],marker=points[1],markevery=50,color='green',linestyle='dotted',label='M7 (240 Myr)')
plt.plot(timeindex,SCsCR8[1:],marker=points[1],markevery=50,color='orange',linestyle='-.',label='M8 (350 Myr)')
plt.rc('lines',markersize=2,markeredgewidth=0.0,linewidth=2.0)
plt.plot(timeindex,SCsCR10[1:],marker=points[0],markevery=3,color='purple',linestyle='None',label='M10 (800 Myr)')

#plt.vlines(85,0,1,color='red',linestyle='dotted',linewidth=0.5)
delayarray = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250])
delayarray = np.array([0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500])
timearray = (delayarray*1e5)/(1e6)
timelist = list(timearray.astype(int))
#plt.xticks(np.array([1,20,40,60,80,100,120,140,160,180,200,220,240]),[1,20,40,60,80,100,120,140,160,180,200,220,240],fontsize=9)

plt.xticks(timearray,timelist,fontsize=12)
plt.yticks(np.array([0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40]),[0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40],fontsize=12)
plt.xlabel(r'$\tau_s$ [Myr]',fontsize=15)
#plt.xlabel('Delay Steps',fontsize=9)
ax1.set_xticklabels([])
plt.ylabel(r'$C$',fontsize=15)
plt.xlim(0,40)
plt.ylim(0.11,0.4)
#plt.legend(loc='lower right',fontsize=12,frameon=False,handlelength=5,numpoints=3)
plt.text(0.07,0.92,'(a)',horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes,fontsize=16)

#savefilename='SC_galpy0718_1000timesteps_3000_orbits.png'
#savefilename='SC_galpy0718_1000timesteps_all_orbits.png'
#savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')


#fig=plt.figure(num=2,figsize=(5,3.5),dpi=300,facecolor='w',edgecolor='k')
#plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

plt.rc('lines',markersize=8,markeredgewidth=0.0,linewidth=2.0)
ax1=plt.subplot(2,1,2)
plt.plot(timeindex[:SCsCR4_endarray-1],PEsCR4[1:SCsCR4_endarray],marker=points[1],markevery=50,linestyle='dashed',color='blue',label='M4 (140 Myr)')
plt.plot(timeindex,PEsCR6[1:],marker=points[1],markevery=50,color='red',linestyle='solid',label='M6 (200 Myr)')
plt.plot(timeindex,PEsCR7[1:],marker=points[1],markevery=50,color='green',linestyle='dotted',label='M7 (240 Myr)')
plt.plot(timeindex,PEsCR8[1:],marker=points[1],markevery=50,color='orange',linestyle='-.',label='M8 (350 Myr)')
plt.rc('lines',markersize=2,markeredgewidth=0.0,linewidth=2.0)
plt.plot(timeindex,PEsCR10[1:],marker=points[1],markevery=3,color='purple',linestyle='None',label='M10 (800 Myr)')

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)
delayarray = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250])
delayarray = np.array([0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500])
timearray = (delayarray*1e5)/(1e6)
timelist = list(timearray.astype(int))
plt.xticks(timearray,timelist,fontsize=12)
plt.xlabel(r'$\tau_s$ [Myr]',fontsize=15)
#ax1.set_xticklabels([])
plt.yticks(fontsize=12)
plt.ylabel(r'$H$',fontsize=15)
plt.xlim(0,40)
plt.ylim(0,0.95)
plt.legend(loc='lower right',fontsize=12,frameon=False,handlelength=5,numpoints=5)
plt.text(0.07,0.92,'(b)',horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes,fontsize=16)

savefilename='SC_and_PE_CRscan_peakcomplexitysignature_resorted.png'
#savefilename='PE_galpy0718_1000timesteps_all_orbits.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()


"""
Maxes and Mins
CR4 Max: 0.3311   Min: 0.1884   Range: 0.1427   MaxLya: 0.4660   MinLya: 0.0008   RangeLya: 1.53e-5
CR6 Max: 0.3461   Min: 0.1582   Range: 0.1879   MaxLya: 0.0929   MinLya: 0.0027   RangeLya: 0.0003
CR7 Max: 0.3430   Min: 0.2242   Range: 0.1188   MaxLya: 0.1296   MinLya: 0.0002   RangeLya: 6.4e-6
CR8 Max: 0.3311   Min: 0.3067   Range: 0.0244   MaxLya: 0.4660   MinLya: 6.09e-6  RangeLya: 5.8e-9
"""

CR4max = 5.6
CR6max = 8.6
CR7max = 11.6
CR8max = 15.5

#resorted values
CR4max = 5.9
CR6max = 9.7
CR7max = 12.7
CR8max = 17.0


CRs = np.array([4,5,6,7,8])
CRmaxes = np.array([CR4max,-1,CR6max,CR7max,CR8max])
plt.rc('lines',markersize=6)
fig=plt.figure(num=33,figsize=(7,11),dpi=600,facecolor='w',edgecolor='k')
left  = 0.16  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.05  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.0   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(3,1,1)
plt.plot(CRs,CRmaxes,'o',color='blue')

#linear regression
from scipy import stats
a=np.array([4,5,7,8])
b=np.array([CR4max,CR6max,CR7max,CR8max])
slope, intercept, r_value, p_value, std_err = stats.linregress(a, b)
plt.plot(a,(a*slope)+intercept,color='red',label=r'Linear Fit, $R^{2}=$'+str(np.round(r_value**2,3)))

#quad fit
qd=np.polyfit(a,b,2)
qd_line = np.poly1d(qd)
#plt.plot(CRs,qd_line(CRs),color='blue',label='Quadratic Fit')

plt.yticks(fontsize=12)
plt.xticks(np.array([4,5,6,7,8]),[4,5,6,7,8],fontsize=12)
#plt.xlabel('Radius of Co-Rotation',fontsize=15)
ax1.set_xticklabels([])
#plt.setp(ax1.get_xticklabels(), rotation='vertical', fontsize=14)
ax1.tick_params(axis='x',direction='in',top=True)
plt.ylabel(r'$\tau_s^{max}$ [Myr]',fontsize=15)
plt.ylim(1,17.5)
plt.text(0.07,0.92,'(a)',horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes,fontsize=16)
plt.legend(loc='lower right',fontsize=12,frameon=False,handlelength=5)


#savefilename='ComplexityPeak_vs_CR.png'
#savefilename='PE_galpy0718_1000timesteps_all_orbits.png'
#savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')

OD4min = 140
OD6min = 200
OD7min = 240
OD8min = 350

#resorted
OD4min = 125
OD6min = 200
OD7min = 275
OD8min = 400

CRs = np.array([4,5,6,7,8])
ODmins = np.array([OD4min,-1,OD6min,OD7min,OD8min])
plt.rc('lines',markersize=6)
ax2=plt.subplot(3,1,2)
plt.plot(CRs,ODmins,'o',color='blue')

#linear regression
a=np.array([4,5,7,8])
c=np.array([OD4min,OD6min,OD7min,OD8min])
slope, intercept, r_value, p_value, std_err = stats.linregress(a, c)
plt.plot(a,(a*slope)+intercept,color='red',label=r'Linear Fit, $R^{2}=$'+str(np.round(r_value**2,3)))

#quad fit
qd=np.polyfit(a,c,2)
qd_line = np.poly1d(qd)
#plt.plot(CRs,qd_line(CRs),color='blue',label='Quadratic Fit')

plt.yticks(fontsize=12)
plt.xticks(np.array([4,5,6,7,8]),[4,5,6,7,8],fontsize=12)
#plt.xlabel('Radius of Co-Rotation',fontsize=15)
ax2.set_xticklabels([])
ax2.tick_params(axis='x',direction='in',top=True)
plt.ylabel(r'$T_{D}$ [Myr]',fontsize=15)
plt.ylim(10,440)
plt.text(0.07,0.92,'(b)',horizontalalignment='center',verticalalignment='center',transform=ax2.transAxes,fontsize=16)
plt.legend(loc='lower right',fontsize=12,frameon=False,handlelength=5)
#savefilename='OrbitDuration_vs_CR.png'
#savefilename='PE_galpy0718_1000timesteps_all_orbits.png'
#savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')

ax3=plt.subplot(3,1,3)

plt.plot(CRs,ODmins/CRmaxes,'o',color='blue')
plt.plot(CRs,np.repeat(21.75,5),color='red',label=r'Mean=22.88 $\pm \sigma=$1.54')
#mean = 22.88 std=1.54
ax3.fill_between(CRs, 21.75-1.09,21.75+1.09, facecolor='lightpink', transform=ax3.transData)
plt.yticks(fontsize=12)
plt.xticks(np.array([4,5,6,7,8]),[4,5,6,7,8],fontsize=12)
ax3.tick_params(axis='x',direction='inout',top=True)
plt.xlabel(r'$R_{CR}$ [kpc]',fontsize=15)
plt.ylabel(r'$T_{D}/\tau_s^{max}$',fontsize=15)
plt.ylim(10,29)
plt.text(0.07,0.92,'(c)',horizontalalignment='center',verticalalignment='center',transform=ax3.transAxes,fontsize=16)
plt.legend(loc='lower right',fontsize=12,frameon=False,handlelength=5)

savefilename='Peaks_OD_ratio_vs_CR_resorted.png'
#savefilename='PE_galpy0718_1000timesteps_all_orbits.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()











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

