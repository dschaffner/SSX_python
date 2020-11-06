# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:00:45 2017

@author: dschaffner
"""
import numpy as np
import matplotlib.pylab as plt
from loadnpyfile import loadnpyfile
from calc_PE_SC import PE, CH, PE_dist, PE_calc_only
#import Cmaxmin as cpl
from collections import Counter
from math import factorial
import matplotlib.cm as cm
import os

#calc_PESC_fluid.py

#datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\GalpyData_July2018\\'
#datadir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Galatic Dynamics Data\\GalpyData_July2018\\resorted_data\\CR6_3t_Rg_Full\\'
#datadir = 'C:\\Users\\dschaffner\\Dropbox\\PESC_Chaos\\CR6_3t\\'
datadir = 'C:\\Users\\dschaffner\\Dropbox\\PESC_Chaos\\Sorted_CR6_et\\'
npy='.npy'
#fileheader = 'PE_SC_IDdatabase_Type_1_data_249_delays_3000_orbits_galpy0718'
#fileheader = 'PE_SC_IDdatabase_Type_1_data_249_delays_galpy0718'
#fileheader = 'CR6_3t_Rg_Full'
fileheader = 'CR6_1t_Rg_Full'
datafile = loadnpyfile(datadir+fileheader+npy)
#PEs = datafile['PEs']
#SCs = datafile['SCs']

startindex=1700
endindex=startindex+1700

plt.figure(1)
plt.clf()
hist0=np.histogram(datafile[:,0],bins=80,range=(2,9))
hist1=np.histogram(datafile[:,startindex],bins=80,range=(2,9))
hist2=np.histogram(datafile[:,endindex],bins=80,range=(2,9))
x=hist1[1][1:]
y=hist2[0]-hist1[0]
plt.plot(x,y,'black')



plt.fill_between(x, y, 0, where=y >= 0, facecolor='green', interpolate=True)
plt.fill_between(x, y, 0, where=y<0,facecolor='red',interpolate=True)
plt.xlabel('R [kpc]')
plt.ylabel(r'$\Delta$ Count')
plt.title(r'$\Delta$ Count by Radius $T_{dyn}=1$ to $T_{dyn}=2$')
plt.ylim(-1000,1000)



#Resonance Lines
#plt.vlines(5,-1000,1000)#CR
#plt.vlines(4.12,-1000,1000,linestyle='dotted')#ULR
#plt.vlines(5.88,-1000,1000,linestyle='dotted')#ULR
#plt.vlines(3.23,-1000,1000,linestyle='dashed')#LR
#plt.vlines(6.77,-1000,1000,linestyle='dashed')#LR
#plt.vlines(6,-1000,1000,color='blue')
plt.vlines(4.94,-1000,1000,color='blue',linestyle='dotted')#ULR
plt.vlines(7.06,-1000,1000,color='blue',linestyle='dotted')#ULR
plt.vlines(3.88,-1000,1000,color='blue',linestyle='dashed')#LR
plt.vlines(8.12,-1000,1000,color='blue',linestyle='dashed')#LR
#plt.vlines(7,-1000,1000,color='purple')
#plt.vlines(5.76,-1000,1000,color='purple',linestyle='dotted')#ULR
#plt.vlines(8.24,-1000,1000,color='purple',linestyle='dotted')#ULR
#plt.vlines(4.53,-1000,1000,color='purple',linestyle='dashed')#LR
#plt.vlines(9.47,-1000,1000,color='purple',linestyle='dashed')#LR

ax2=plt.subplot(1,1,1)
bins2=(np.arange(1000)*0.005)+3.5
r5range=(np.arange(1000)*0.005)+3.5
r=np.where(r5range<4.16)
r2=np.where(r5range>5.98)
r5range[r]=0.0
r5range[r2]=0.0
plt.plot(bins2,r5range,linewidth=0.0)
ax2.fill_between(bins2, 0, 40, ec='none',where=r5range > 4.16 ,
   color='black', alpha=0.15, transform=ax2.get_xaxis_transform())

r6range=(np.arange(1000)*0.005)+3.5
r=np.where(r6range<5.04)
r2=np.where(r6range>7.1)
r6range[r]=0.0
r6range[r2]=0.0
ax2.fill_between(bins2, 0, 40, ec='none',where=r6range > 5.04 ,
   color='blue', alpha=0.15, transform=ax2.get_xaxis_transform())

r7range=(np.arange(1000)*0.005)+3.5
r=np.where(r7range<5.96)
r2=np.where(r7range>8.18)
r7range[r]=0.0
r7range[r2]=0.0
ax2.fill_between(bins2, 0, 40, ec='none',where=r7range > 5.96 ,
   color='purple', alpha=0.15, transform=ax2.get_xaxis_transform())

savefilename='CR6_3t_Rg_Full_DeltaCount_tdyn1totdyn2_wranges.png'
#savefilename='CR6_1t_Rg_Full_DeltaCount_tdyn1totdyn2_wranges.png'
savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefile,dpi=200,facecolor='w',edgecolor='k')


histT0=np.histogram(datafile[:,0],bins=80,range=(2,9))
histT1=np.histogram(datafile[:,1700],bins=80,range=(2,9))
histT2=np.histogram(datafile[:,3400],bins=80,range=(2,9))
histT3=np.histogram(datafile[:,5100],bins=80,range=(2,9))
histT4=np.histogram(datafile[:,6799],bins=80,range=(2,9))

plt.rc('axes',linewidth=1.0)
plt.rc('xtick.major',width=1.0)
plt.rc('ytick.major',width=1.0)
plt.rc('xtick.minor',width=1.0)
plt.rc('ytick.minor',width=1.0)
plt.rc('lines',markersize=8,markeredgewidth=0.0,linewidth=1.0)
#plt.rcParams['ps.fonttype'] = 42
#plt.rcParams['pdf.fonttype'] = 42
points = ['o','v','s','p','*','h','^','D','+','>','H','d','x','<']

     

fig=plt.figure(num=44,figsize=(2,7),dpi=300,facecolor='w',edgecolor='k')
plt.clf()
left  = 0.3  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.15  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.05   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

ax1=plt.subplot(5,1,1)
plt.bar(histT0[1][1:],histT0[0],width=0.09)
ax1.set_xticklabels([])
plt.ylim(0,2500)
plt.yticks(np.array([0,1500]),fontsize=7)
ax1=plt.subplot(5,1,2)
plt.bar(histT1[1][1:],histT1[0],width=0.09)
ax1.set_xticklabels([])
plt.ylim(0,2500)
plt.yticks(np.array([0,1500]),fontsize=7)
ax1=plt.subplot(5,1,3)
plt.bar(histT2[1][1:],histT2[0],width=0.09)
ax1.set_xticklabels([])
plt.ylabel('Num. Orbits @ R',fontsize=10)
plt.ylim(0,2500)
plt.yticks(np.array([0,1500]),fontsize=7)
ax1=plt.subplot(5,1,4)
plt.bar(histT3[1][1:],histT3[0],width=0.09)
ax1.set_xticklabels([])
plt.ylim(0,2500)
plt.yticks(np.array([0,1500]),fontsize=7)
ax1=plt.subplot(5,1,5)
plt.bar(histT4[1][1:],histT4[0],width=0.09)
plt.ylim(0,2500)
plt.yticks(np.array([0,1500]),fontsize=7)
plt.xlabel('R [kpc]',fontsize=10)
plt.xticks(np.array([2,3,4,5,6,7,8,9]),fontsize=7)


#normalized to initial count
plt.figure(2)
plt.clf()
n=hist0[0][:]#27:62]#[8:74]
xn=hist1[1][1:]#[27:62]#[8:74]
yn=hist2[0][:]-hist1[0][:]#[27:62]-hist1[0][27:62]
plt.plot(xn,yn/n,'black')
plt.fill_between(xn, yn/n, 0, where=(yn/n)>= 0, facecolor='green', interpolate=True)
plt.fill_between(xn, yn/n, 0, where=(yn/n)<0,facecolor='red',interpolate=True)
plt.xlabel('R [kpc]')
plt.ylabel(r'$\Delta$ Count/$n_{0}$')
plt.title(r'$\Delta$ Count by Radius $T_{dyn}=1$ to $T_{dyn}=2$')
#plt.ylim(-1000,1000)


#compute slope of initial distribution to use as normalization
plt.figure(3)
plt.clf()
x=np.arange(80)
z=np.polyfit(x[27:62],n[27:62],1)
zn=z[1]+x*z[0]
plt.plot(xn,yn/zn,'black')
plt.fill_between(xn, yn/zn, 0, where=(yn/zn)>= 0, facecolor='green', interpolate=True)
plt.fill_between(xn, yn/zn, 0, where=(yn/zn)<0,facecolor='red',interpolate=True)
plt.xlabel('R [kpc]')
plt.ylabel(r'$\Delta$ Count/$n_{0}$')
plt.title(r'$\Delta$ Count by Radius $T_{dyn}=1$ to $T_{dyn}=2$')

plt.figure(4)
plt.clf()
plt.plot(x,n)
plt.plot(x,zn)



#Contour Plot
from loadnpzfile import loadnpzfile
datadir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Galatic Dynamics Data\\GalpyData_July2018\\resorted_data\\CR6_3t_Rg_Full\\'
npy='.npz'
fileheader = 'PE_SC_1t_binnedBy0p1_3p5to8p5_50bins_start1700_length3400_499delays_499_delays'
#fileheader = 'PE_SC_1t_binnedBy0p1_3p5to8p5_50bins_start0_end1700_length1700_499delays_499_delays'
#fileheader = 'PE_SC_1t_binnedBy0p1_3p5to8p5_50bins_start1700_length3400_499delays_499_delays_binnedbyfinalr'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs = datafile['PEs']
SCs = datafile['SCs']

delayindex = np.arange(1,501)
timeindex=(delayindex*1e5)/(1e6)





#Complexity curves from contour plot
fig=plt.figure(num=20,figsize=(3,6),dpi=200,facecolor='w',edgecolor='k')
left  = 0.2  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.15  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.0   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

delayindex = np.arange(1,501)
timeindex=(delayindex*1e5)/(1e6)
binlabels=['3.5-4.0kpc','4.0-4.5kpc','4.5-5.0kpc','5.0-5.5kpc','5.5-6.0kpc','6.0-6.5kpc','6.5-7.0kpc','7.0-7.5kpc','7.5-8.0kpc','8.0-8.5kpc']
binlabels=['3.5','4.0','4.5','5.0','5.5','6.0','6.5','7.0','7.5','8.0']





#Map bins to radii
"""
Both simulations have one spiral that has a spiral with R_CR = 6 kpc.  So,
                R_CR =   6 = bins[25]
                R_ULR = 4.94 ~bins[15] and  7.06~bins[35]
                R_LR =    3.88 ~bins[4]    and  8.12~bins[46]
3t has two more patterns where,
                R_CR =   5
                R_ULR = 4.12      and        5.88
                R_LR =    3.23     and        6.77
                R_CR =   7
                R_ULR = 5.76      and        8.24
                R_LR =    4.53     and        9.47
"""


ax1=plt.subplot(1,1,1)
plt.plot(timeindex[1:450],SCs[1:450,4],color='blue',marker=points[0],markevery=(20,100),label='r=3.88')#binlabels[0])
plt.plot(timeindex[1:450],SCs[1:450,15],color='red',linestyle='solid',label='r=4.94')#binlabels[1])
plt.plot(timeindex[1:450],SCs[1:450,25],color='green',linestyle='solid',label='r=6.0')#binlabels[2])
plt.plot(timeindex[1:450],SCs[1:450,35],color='orange',linestyle='solid',label='r=7.06')#binlabels[3])
plt.plot(timeindex[1:450],SCs[1:450,40],color='darkorange',linestyle='solid',label='r=7.5')
plt.plot(timeindex[1:450],SCs[1:450,46],color='purple',linestyle='solid',label='r=8.12')#binlabels[4])

#plt.xticks(np.array([1,20,40,60,80,100,120,140,160,180,200,220,240]),[1,20,40,60,80,100,120,140,160,180,200,220,240],fontsize=9)

#plt.xticks(timearray,timelist,fontsize=12)
#plt.yticks(np.array([0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40]),[0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40],fontsize=12)
plt.xlabel(r'$\tau_s$ [Myr]',fontsize=15)
#plt.xlabel('Delay Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel(r'$C$',fontsize=15)
plt.xlim(0,40)
plt.ylim(0.11,0.4)
plt.legend(loc='lower right',fontsize=12,frameon=False,handlelength=5,numpoints=2)
plt.text(0.04,0.95,'(a)',fontsize=16,horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes)

#savefilename='SC_galpy0718_1000timesteps_3000_orbits.png'
#savefilename='SC_galpy0718_1000timesteps_all_orbits.png'
#savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')

#contour plot
fig=plt.figure(num=33,figsize=(4,7),dpi=300,facecolor='w',edgecolor='k')
left  = 0.2  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.2  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax=plt.subplot(2,1,1)
bins=(np.arange(50)*0.1)+3.5
x,y=np.meshgrid(bins,timeindex[1:400])
cp = ax.contourf(x,y,SCs[1:400,:],levels=50,cmap=cm.plasma,vmin=0.25)
#fig.colorbar(cp)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.ylabel(r'$C$',fontsize=8)

plt.vlines(6,1,40,color='blue')
plt.vlines(4.94,1,40,color='blue',linestyle='dotted')#ULR
plt.vlines(7.06,1,40,color='blue',linestyle='dotted')#ULR
plt.vlines(3.88,1,40,color='blue',linestyle='dashed')#LR
plt.vlines(8.12,1,40,color='blue',linestyle='dashed')#LR



#normalized to initial count
ax=plt.subplot(2,1,2)
n=hist0[0][:]#27:62]#[8:74]
xn=hist1[1][1:]#[27:62]#[8:74]
yn=hist2[0][:]-hist1[0][:]#[27:62]-hist1[0][27:62]
plt.plot(xn,yn/n,'black')
plt.fill_between(xn, yn/n, 0, where=(yn/n)>= 0, facecolor='green', interpolate=True)
plt.fill_between(xn, yn/n, 0, where=(yn/n)<0,facecolor='red',interpolate=True)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel('R [kpc]',fontsize=8)
plt.ylabel(r'$\Delta$ Count/$n_{0}$',fontsize=8)
#plt.title(r'$\Delta$ Count by Radius $T_{dyn}=1$ to $T_{dyn}=2$')
plt.ylim(-1,1)
plt.xlim(3.5,8.4)


plt.vlines(6,-1,1,color='blue')
plt.vlines(4.94,-1,1,color='blue',linestyle='dotted')#ULR
plt.vlines(7.06,-1,1,color='blue',linestyle='dotted')#ULR
plt.vlines(3.88,-1,1,color='blue',linestyle='dashed')#LR
plt.vlines(8.12,-1,1,color='blue',linestyle='dashed')#LR




fig=plt.figure(num=50,figsize=(7,5),dpi=600,facecolor='w',edgecolor='k')
left  = 0.2  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.2  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.2   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax=plt.subplot(1,1,1)
bins=(np.arange(50)*0.1)+3.5
x,y=np.meshgrid(bins,timeindex[1:400])
cp = ax.contourf(x,y,SCs[1:400,:],levels=50,cmap=cm.plasma,vmin=0.25)
#fig.colorbar(cp)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylabel(r'$C$',fontsize=15)
plt.xlabel('R [kpc]',fontsize=15)

plt.vlines(6,1,40,color='blue')
plt.vlines(4.94,1,40,color='white',linestyle='dotted',linewidth=4)#ULR
plt.vlines(7.06,1,40,color='white',linestyle='dotted',linewidth=4)#ULR
plt.vlines(3.88,1,40,color='white',linestyle='dashed',linewidth=4)#LR
plt.vlines(8.12,1,40,color='white',linestyle='dashed',linewidth=4)#LR

plt.xlim(3.48,6.82)



savefilename='1t_complexitycurves_Td1_to_Td2.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()