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

fileheader = 'M4_2234_et_timescan\\PE_SC_Type_32i_Rg_499_delays_1464orbits_of1727_total1300_timesteps_resorted_et'
datafile = loadnpzfile(datadir+fileheader+npy)
PEsCR4 = datafile['PEs']
SCsCR4 = datafile['SCs']
SCsCR4_endarray=np.where(SCsCR4[1:]==0)[0][0]

fileheader = 'M5_2792_et_timescan\\PE_SC_Type_32i_Rg_499_delays_1114orbits_of1331_total1600_timesteps_resorted_et'
datafile = loadnpzfile(datadir+fileheader+npy)
PEsCR5 = datafile['PEs']
SCsCR5 = datafile['SCs']
SCsCR5_endarray=500

fileheader = 'M6_3352_et_timescan\\PE_SC_Type_32i_Rg_499_delays_3145orbits_of3899_total2000_timesteps_resorted_et'
datafile = loadnpzfile(datadir+fileheader+npy)
PEsCR6 = datafile['PEs']
SCsCR6 = datafile['SCs']
SCsCR6_endarray=500

fileheader = 'M7_3910_et_timescan\\PE_SC_Type_32i_Rg_499_delays_1277orbits_of1619_total2550_timesteps_resorted_et'
datafile = loadnpzfile(datadir+fileheader+npy)
PEsCR7 = datafile['PEs']
SCsCR7 = datafile['SCs']
SCsCR7_endarray=500

fileheader = 'M8_4468_et_timescan\\PE_SC_Type_32i_Rg_499_delays_1521orbits_of1589_total3500_timesteps_resorted_et'
datafile = loadnpzfile(datadir+fileheader+npy)
PEsCR8 = datafile['PEs']
SCsCR8 = datafile['SCs']
SCsCR8_endarray=500

fileheader = 'M9_5026_et_timescan\\PE_SC_Type_32i_Rg_499_delays_556orbits_of677_total4000_timesteps_resorted_et'
datafile = loadnpzfile(datadir+fileheader+npy)
PEsCR9 = datafile['PEs']
SCsCR9 = datafile['SCs']
SCsCR9_endarray=500

fileheader = 'M10_5586_et_timescan\\PE_SC_Type_32i_Rg_499_delays_224orbits_of264_total4000_timesteps_resorted_et'
datafile = loadnpzfile(datadir+fileheader+npy)
PEsCR10 = datafile['PEs']
SCsCR10 = datafile['SCs']
SCsCR10_endarray=500


M4dyn=1117
M5dyn=1396
M6dyn=1676
M7dyn=1955
M8dyn=2234
M9dyn=2513
M10dyn=2793

delayindex = np.arange(1,500)
timeindex=(delayindex*1e5)/(1e6)
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

fig=plt.figure(num=1,figsize=(7,9),dpi=600,facecolor='w',edgecolor='k')
left  = 0.16  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.08  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.0   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)



ax1=plt.subplot(2,1,1)
plt.plot(timeindex_normM4[:SCsCR4_endarray-1],SCsCR4[1:SCsCR4_endarray],color=colors[0,:],label='M4e:'+str(orbitdur1)+r' $T_{dyn}^{M4}$')
plt.plot(timeindex_normM5,SCsCR5[1:],color=colors[1,:],label='M5e:'+str(orbitdur2)+r' $T_{dyn}^{M5}$')
plt.plot(timeindex_normM6,SCsCR6[1:],color=colors[2,:],label='M6e:'+str(orbitdur3)+r' $T_{dyn}^{M6}$')
plt.plot(timeindex_normM7,SCsCR7[1:],color=colors[3,:],label='M7e:'+str(orbitdur4)+r' $T_{dyn}^{M7}$')
plt.plot(timeindex_normM8,SCsCR8[1:],color=colors[4,:],label='M8e:'+str(orbitdur5)+r' $T_{dyn}^{M8}$')
plt.plot(timeindex_normM9,SCsCR9[1:],color=colors[5,:],label='M9e:'+str(orbitdur6)+r' $T_{dyn}^{M9}$')
plt.plot(timeindex_normM10,SCsCR10[1:],color=colors[6,:],label='M10e:'+str(orbitdur7)+r' $T_{dyn}^{M10}$')
#plt.rc('lines',markersize=2,markeredgewidth=0.0,linewidth=2.0)
#plt.plot(timeindex,SCsCR10[1:],marker=points[0],markevery=3,color='purple',linestyle='None',label='M10 (800 Myr)')

#plt.vlines(85,0,1,color='red',linestyle='dotted',linewidth=0.5)
#delayarray = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250])
#delayarray = np.array([0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500])
#timearray = (delayarray*1e5)/(1e6)/100.0
#timelist = list(timearray.astype(int))
#plt.xticks(np.array([1,20,40,60,80,100,120,140,160,180,200,220,240]),[1,20,40,60,80,100,120,140,160,180,200,220,240],fontsize=9)

plt.xticks(fontsize=18)
plt.yticks(np.array([0.10,0.20,0.30,0.40]),[0.10,0.20,0.30,0.40],fontsize=18)
plt.xlabel(r'$t_{pat}/T_{Dyn}^{M-}$',fontsize=18)
#plt.xlabel('Delay Steps',fontsize=9)
ax1.set_xticklabels([])
plt.ylabel(r'$C$',fontsize=20)
plt.xlim(0,1.0)
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
plt.plot(timeindex_normM4[:SCsCR4_endarray-1],PEsCR4[1:SCsCR4_endarray],color=colors[0,:],label='M4e')
plt.plot(timeindex_normM5,PEsCR5[1:],color=colors[1,:],label='M5e')
plt.plot(timeindex_normM6,PEsCR6[1:],color=colors[2,:],label='M6e')
plt.plot(timeindex_normM7,PEsCR7[1:],color=colors[3,:],label='M7e')
plt.plot(timeindex_normM8,PEsCR8[1:],color=colors[4,:],label='M8e')
plt.plot(timeindex_normM9,PEsCR9[1:],color=colors[5,:],label='M9e')
plt.plot(timeindex_normM10,PEsCR10[1:],color=colors[6,:],label='M10e')

#plt.rc('lines',markersize=2,markeredgewidth=0.0,linewidth=2.0)
#plt.plot(timeindex,PEsCR10[1:],marker=points[1],markevery=3,color='purple',linestyle='None',label='M10 (800 Myr)')

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)
#delayarray = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250])
#delayarray = np.array([0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500])
#timearray = (delayarray*1e5)/(1e6)/100.0
#timelist = list(timearray.astype(int))
plt.xticks(fontsize=18)
plt.xlabel(r'$t_{pat}/T_{dyn}$',fontsize=20)
#ax1.set_xticklabels([])
plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],[0.0,0.2,0.4,0.6,0.8,1.0],fontsize=18)
plt.ylabel(r'$H$',fontsize=20)
plt.xlim(0,1.0)
plt.ylim(-0.1,1.1)

leg=plt.legend(loc='lower right',fontsize=12,ncol=2,frameon=False,handlelength=5,numpoints=5)
leg.set_title('CR: (Orbit Duration=$T_{D}$)',prop={'size':10})
plt.text(0.07,0.92,'(b)',horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes,fontsize=16)

savefilename='SC_and_PE_CRscan_peakcomplexitysignature_et_dyntimenorm_ApJver_newcolor_wtpat.eps'
#savefilename='SC_and_PE_CRscan_peakcomplexitysignature_et_dyntimenorm_ApJver_newcolor.eps'
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
"""
M4-32i 1300
maxSC=0.3342 at index 54 or 5.4Myr
Dyntime=1117
Td/Dyn=1.164
Tsmax/Dyn=0.048

M5-32i 1600
maxSC=0.3418 at index 72 or 7.2Myr
Dyntime=1396
Td/Dyn=1.146
Tsmax/Dyn=0.052

M6-32i 2000, maxSC=0.3464 at index 93 or 9.3Myr
Dyntime=1676
Td/Dyn=1.223
Tsmax/Dyn=0.055

M7-32i 2550, maxSC=0.3420 at index 137 or 13.7Myr
Dyntime=1955
Td/Dyn=1.304
Tsmax/Dyn=0.070

M8-32i 3500, maxSC=0.3390 at index 203 or 20.3Myr
Dyntime=2234
Td/Dyn=1.663
Tsmax/Dyn=0.091

M9-32i
Dyntime=2513

M10-32i
Dyntime=2793

"""
#Pattern Max is tau max * 4
CR4max=54*4
CR5max=72*4
CR6max=93*4
CR7max=137*4
CR8max=203*4

CR4dyn=1117
CR5dyn=1396
CR6dyn=1676
CR7dyn=1955
CR8dyn=2234

CR4orbitmin=1300
CR5orbitmin=1600
CR6orbitmin=2000
CR7orbitmin=2550
CR8orbitmin=3500

CR4meantrap_period=2.31#Already normalized to CR4dyn (from calc_autocorr_galaxy_abscorr_halfperonly.py)
CR5meantrap_period=2.4
CR6meantrap_period=2.61
CR7meantrap_period=2.81
CR8meantrap_period=3.14
trap_periods=np.array([CR4meantrap_period,CR5meantrap_period,CR6meantrap_period,CR7meantrap_period,CR8meantrap_period])


CRs = np.array([4,5,6,7,8])
CRmaxes = np.array([CR4max,CR5max,CR6max,CR7max,CR8max])
CRmaxesnorm = np.array([CR4max/CR4dyn,CR5max/CR5dyn,CR6max/CR6dyn,CR7max/CR7dyn,CR8max/CR8dyn])
plt.rc('lines',markersize=10)
fig=plt.figure(num=33,figsize=(7,11),dpi=600,facecolor='w',edgecolor='k')
left  = 0.16  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.0   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(3,1,1)
plt.plot(CRs,CRmaxesnorm,'o',color='blue')

#linear regression
from scipy import stats
a=CRs
b=CRmaxesnorm
slope, intercept, r_value, p_value, std_err = stats.linregress(a, b)
plt.plot(a,(a*slope)+intercept,color='red',label=r'Linear Fit, $R^{2}=$'+str(np.round(r_value**2,3)))

#quad fit
qd=np.polyfit(a,b,2)
qd_line = np.poly1d(qd)
#plt.plot(CRs,qd_line(CRs),color='blue',label='Quadratic Fit')

plt.yticks(fontsize=15)
plt.xticks(np.array([4,5,6,7,8]),[4,5,6,7,8],fontsize=15)
#plt.xlabel('Radius of Co-Rotation',fontsize=15)
ax1.set_xticklabels([])
#plt.setp(ax1.get_xticklabels(), rotation='vertical', fontsize=14)
ax1.tick_params(axis='x',direction='in',top=True)
plt.ylabel(r'$t_{pat}^{max}/T_{dyn}$',fontsize=20)
#plt.ylim(0,0.1)
plt.text(0.07,0.92,'(a)',horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes,fontsize=16)
plt.legend(loc='upper center',fontsize=8,frameon=False,handlelength=5)


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

ODmins = np.array([CR4orbitmin,CR5orbitmin,CR6orbitmin,CR7orbitmin,CR8orbitmin])
ODminsnorm = np.array([CR4orbitmin/CR4dyn,CR5orbitmin/CR5dyn,CR6orbitmin/CR6dyn,CR7orbitmin/CR7dyn,CR8orbitmin/CR8dyn])
plt.rc('lines',markersize=10)
ax2=plt.subplot(3,1,2)
plt.plot(CRs,ODminsnorm,'o',color='blue',label=r'$T_{D}/T_{dyn}$')
plt.plot(CRs,trap_periods/2.0,'^',color='red',label=r'$\frac{1}{2}T_{trap}/T_{dyn}$')

#linear regression
a=CRs
c=ODminsnorm
slope, intercept, r_value, p_value, std_err = stats.linregress(a, c)
plt.plot(a,(a*slope)+intercept,color='red',label=r'Linear Fit, $R^{2}=$'+str(np.round(r_value**2,3)))

#quad fit
qd=np.polyfit(a,c,2)
qd_line = np.poly1d(qd)
#plt.plot(CRs,qd_line(CRs),color='blue',label='Quadratic Fit')

plt.yticks(fontsize=15)
plt.xticks(np.array([4,5,6,7,8]),[4,5,6,7,8],fontsize=15)
#plt.xlabel('Radius of Co-Rotation',fontsize=15)
ax2.set_xticklabels([])
ax2.tick_params(axis='x',direction='in',top=True)
plt.ylabel(r'Frac. of $T_{dyn}$',fontsize=20)
#plt.ylim(0,2.2)
plt.text(0.07,0.92,'(b)',horizontalalignment='center',verticalalignment='center',transform=ax2.transAxes,fontsize=16)
plt.legend(loc='upper center',fontsize=8,frameon=False,handlelength=5)
#savefilename='OrbitDuration_vs_CR.png'
#savefilename='PE_galpy0718_1000timesteps_all_orbits.png'
#savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')

ax3=plt.subplot(3,1,3)

#plt.plot(CRs,CRmaxesnorm/ODminsnorm,'o',color='blue')
#plt.plot(CRs,np.repeat(0.049,5),color='red',label=r'Mean=0.049 $\pm \sigma=$0.006')
#mean = 0.049 std=0.006
#ax3.fill_between(CRs, 0.049-0.006,0.049+0.006, facecolor='lightpink', transform=ax3.transData)
#plt.yticks(fontsize=15)
#plt.xticks(np.array([4,5,6,7,8]),[4,5,6,7,8],fontsize=18)
#ax3.tick_params(axis='x',direction='inout',top=True)
#plt.xlabel(r'$R_{CR}$ [kpc]',fontsize=20)
#plt.ylabel(r'$\tau_s^{max}/T_{D}$',fontsize=15)
#plt.ylim(0,0.11)

#inverted
plt.plot(CRs,ODmins/CRmaxes,'o',color='blue')
plt.plot(CRs,np.repeat(5.18,5),color='red',label=r'Mean=5.18 $\pm \sigma=$0.62')
#mean = 20.7 std=2.5
ax3.fill_between(CRs, 5.18-0.62,5.2+0.62, facecolor='lightpink', transform=ax3.transData)
plt.yticks(fontsize=15)
plt.xticks(np.array([4,5,6,7,8]),[4,5,6,7,8],fontsize=18)
ax3.tick_params(axis='x',direction='inout',top=True)
plt.xlabel(r'$R_{CR}$ [kpc]',fontsize=20)
plt.ylabel(r'$T_{D}/t_{pat}^{max}$',fontsize=20)
plt.ylim(0,10)

plt.text(0.07,0.92,'(c)',horizontalalignment='center',verticalalignment='center',transform=ax3.transAxes,fontsize=16)
plt.legend(loc='lower right',fontsize=8,frameon=False,handlelength=5)

savefilename='Peaks_OD_ratio_vs_CR_resorted_ApJver_wtpat.eps'
#savefilename='Peaks_OD_ratio_vs_CR_resorted_ApJver.eps'
#savefilename='PE_galpy0718_1000timesteps_all_orbits.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()
