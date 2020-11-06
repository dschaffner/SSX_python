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

#calc_PESC_fluid.py

#datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\GalpyData_July2018\\'
datadir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Galatic Dynamics Data\\GalpyData_July2018\\resorted_data\\CR6_3t_Rg_Full\\'
npy='.npz'
#fileheader = 'PE_SC_IDdatabase_Type_1_data_249_delays_3000_orbits_galpy0718'
#fileheader = 'PE_SC_IDdatabase_Type_1_data_249_delays_galpy0718'

"""
fileheader='radiusAttimestep0_1t'
datafile=loadnpzfile(datadir+fileheader+npy)
radii1=datafile['radius']
plt.figure(1)
plt.hist(radii1,bins=50,range=(3.5,8.5))
plt.title('Radius Dist at 0ts')
plt.ylim(0,1500)

fileheader='radiusAttimestep500_1t'
datafile=loadnpzfile(datadir+fileheader+npy)
radii2=datafile['radius']
plt.figure(2)
plt.hist(radii2,bins=50,range=(3.5,8.5))
plt.title('Radius Dist at 500ts')
plt.ylim(0,1500)

fileheader='radiusAttimestep1000_1t'
datafile=loadnpzfile(datadir+fileheader+npy)
radii3=datafile['radius']
plt.figure(3)
plt.hist(radii3,bins=50,range=(3.5,8.5))
plt.title('Radius Dist at 1000ts')
plt.ylim(0,1500)

fileheader='radiusAttimestep1200_1t'
datafile=loadnpzfile(datadir+fileheader+npy)
radii4=datafile['radius']
plt.figure(4)
plt.hist(radii4,bins=50,range=(3.5,8.5))
plt.title('Radius Dist at 1200ts')
plt.ylim(0,1500)

fileheader='radiusAttimestep1500_1t'
datafile=loadnpzfile(datadir+fileheader+npy)
radii5=datafile['radius']
plt.figure(5)
plt.hist(radii5,bins=50,range=(3.5,8.5))
plt.title('Radius Dist at 1500ts')
plt.ylim(0,1500)

fileheader='radiusAttimestep1700_1t'
datafile=loadnpzfile(datadir+fileheader+npy)
radii6=datafile['radius']
plt.figure(6)
plt.hist(radii6,bins=50,range=(3.5,8.5))
plt.title('Radius Dist at 1700ts')
plt.ylim(0,1500)

fileheader='radiusAttimestep2000_1t'
datafile=loadnpzfile(datadir+fileheader+npy)
radii7=datafile['radius']
plt.figure(7)
plt.hist(radii7,bins=50,range=(3.5,8.5))
plt.title('Radius Dist at 2000ts')
plt.ylim(0,1500)

fileheader='radiusAttimestep2500_1t'
datafile=loadnpzfile(datadir+fileheader+npy)
radii8=datafile['radius']
plt.figure(8)
plt.hist(radii8,bins=50,range=(3.5,8.5))
plt.title('Radius Dist at 2500ts')
plt.ylim(0,1500)

fileheader='radiusAttimestep3000_1t'
datafile=loadnpzfile(datadir+fileheader+npy)
radii9=datafile['radius']
plt.figure(9)
plt.hist(radii9,bins=50,range=(3.5,8.5))
plt.title('Radius Dist at 3000ts')
plt.ylim(0,1500)

fileheader='radiusAttimestep3400_1t'
datafile=loadnpzfile(datadir+fileheader+npy)
radii10=datafile['radius']
plt.figure(10)
plt.hist(radii10,bins=50,range=(3.5,8.5))
plt.title('Radius Dist at 3400ts')
plt.ylim(0,1500)

fileheader='radiusAttimestep4000_1t'
datafile=loadnpzfile(datadir+fileheader+npy)
radii11=datafile['radius']
plt.figure(11)
plt.hist(radii11,bins=50,range=(3.5,8.5))
plt.title('Radius Dist at 4000ts')
plt.ylim(0,1500)

fileheader='radiusAttimestep5000_1t'
datafile=loadnpzfile(datadir+fileheader+npy)
radii12=datafile['radius']
plt.figure(12)
plt.hist(radii12,bins=50,range=(3.5,8.5))
plt.title('Radius Dist at 5000ts')
plt.ylim(0,1500)

fileheader='radiusAttimestep6000_1t'
datafile=loadnpzfile(datadir+fileheader+npy)
radii13=datafile['radius']
plt.figure(13)
plt.hist(radii13,bins=50,range=(3.5,8.5))
plt.title('Radius Dist at 6000ts')
plt.ylim(0,1500)
"""
"""
fig=plt.figure(num=14,figsize=(15,1.5),dpi=200,facecolor='w',edgecolor='k')
left  = 0.05  # the left side of the subplots of the figure
right = 0.99    # the right side of the subplots of the figure
bottom = 0.25  # the bottom of the subplots of the figure
top = 0.85      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.0   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
ax1=plt.subplot(1,13,1)
plt.hist(radii1,bins=50,range=(3.5,8.5))
plt.xticks([4,5,6,7,8],fontsize=5)
plt.xlabel('Radius [kpc]',fontsize=5)
plt.yticks(fontsize=5)
plt.ylabel('Count')
plt.ylim(0,1500)
plt.title('0 ts',fontsize=10)

ax2=plt.subplot(1,13,2)
plt.hist(radii2,bins=50,range=(3.5,8.5))
plt.xticks([4,5,6,7,8],fontsize=5)
plt.xlabel('Radius [kpc]',fontsize=5)
plt.yticks(fontsize=5)
ax2.set_yticklabels([])
plt.ylim(0,1500)
plt.title('500 ts',fontsize=10)

ax2=plt.subplot(1,13,3)
plt.hist(radii3,bins=50,range=(3.5,8.5))
plt.xticks([4,5,6,7,8],fontsize=5)
plt.xlabel('Radius [kpc]',fontsize=5)
plt.yticks(fontsize=5)
ax2.set_yticklabels([])
plt.ylim(0,1500)
plt.title('1000 ts',fontsize=10)

ax2=plt.subplot(1,13,4)
plt.hist(radii4,bins=50,range=(3.5,8.5))
plt.xticks([4,5,6,7,8],fontsize=5)
plt.xlabel('Radius [kpc]',fontsize=5)
plt.yticks(fontsize=5)
ax2.set_yticklabels([])
plt.ylim(0,1500)
plt.title('1200 ts',fontsize=10)

ax2=plt.subplot(1,13,5)
plt.hist(radii5,bins=50,range=(3.5,8.5))
plt.xticks([4,5,6,7,8],fontsize=5)
plt.xlabel('Radius [kpc]',fontsize=5)
plt.yticks(fontsize=5)
ax2.set_yticklabels([])
plt.ylim(0,1500)
plt.title('1500 ts',fontsize=10)

ax3=plt.subplot(1,13,6)
plt.hist(radii6,bins=50,range=(3.5,8.5))
plt.xticks([4,5,6,7,8],fontsize=5)
plt.xlabel('Radius [kpc]',fontsize=5)
plt.yticks(fontsize=5)
ax3.set_yticklabels([])
plt.ylim(0,1500)
plt.title('1700 ts',fontsize=10)
ax3.spines['bottom'].set_color('red')
ax3.spines['top'].set_color('red') 
ax3.spines['right'].set_color('red')
ax3.spines['left'].set_color('red')

ax2=plt.subplot(1,13,7)
plt.hist(radii7,bins=50,range=(3.5,8.5))
plt.xticks([4,5,6,7,8],fontsize=5)
plt.xlabel('Radius [kpc]',fontsize=5)
plt.yticks(fontsize=5)
ax2.set_yticklabels([])
plt.ylim(0,1500)
plt.title('2000 ts',fontsize=10)

ax2=plt.subplot(1,13,8)
plt.hist(radii8,bins=50,range=(3.5,8.5))
plt.xticks([4,5,6,7,8],fontsize=5)
plt.xlabel('Radius [kpc]',fontsize=5)
plt.yticks(fontsize=5)
ax2.set_yticklabels([])
plt.ylim(0,1500)
plt.title('2500 ts',fontsize=10)

ax2=plt.subplot(1,13,9)
plt.hist(radii9,bins=50,range=(3.5,8.5))
plt.xticks([4,5,6,7,8],fontsize=5)
plt.xlabel('Radius [kpc]',fontsize=5)
plt.yticks(fontsize=5)
ax2.set_yticklabels([])
plt.ylim(0,1500)
plt.title('3000 ts',fontsize=10)

ax3=plt.subplot(1,13,10)
plt.hist(radii10,bins=50,range=(3.5,8.5))
plt.xticks([4,5,6,7,8],fontsize=5)
plt.xlabel('Radius [kpc]',fontsize=5)
plt.yticks(fontsize=5)
ax3.set_yticklabels([])
plt.ylim(0,1500)
plt.title('3400 ts',fontsize=10)
ax3.spines['bottom'].set_color('red')
ax3.spines['top'].set_color('red') 
ax3.spines['right'].set_color('red')
ax3.spines['left'].set_color('red')

ax2=plt.subplot(1,13,11)
plt.hist(radii11,bins=50,range=(3.5,8.5))
plt.xticks([4,5,6,7,8],fontsize=5)
plt.xlabel('Radius [kpc]',fontsize=5)
plt.yticks(fontsize=5)
ax2.set_yticklabels([])
plt.ylim(0,1500)
plt.title('4000 ts',fontsize=10)

ax2=plt.subplot(1,13,12)
plt.hist(radii12,bins=50,range=(3.5,8.5))
plt.xticks([4,5,6,7,8],fontsize=5)
plt.xlabel('Radius [kpc]',fontsize=5)
plt.yticks(fontsize=5)
ax2.set_yticklabels([])
plt.ylim(0,1500)
plt.title('5000 ts',fontsize=10)

ax2=plt.subplot(1,13,13)
plt.hist(radii13,bins=50,range=(3.5,8.5))
plt.xticks([4,5,6,7,8],fontsize=5)
plt.xlabel('Radius [kpc]',fontsize=5)
plt.yticks(fontsize=5)
ax2.set_yticklabels([])
plt.ylim(0,1500)
plt.title('6000 ts',fontsize=10)

savefilename='CR6_1t_Rg_Full_OrbitDistbyTime.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=200,facecolor='w',edgecolor='k')

"""




fileheader = 'PE_SC_allorbits_start1700_length3400_499delays_499_delays'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs_all = datafile['PEs']
SCs_all = datafile['SCs']

fileheader = 'PE_SC_btwn2and4_start1700_length3400_499delays_499_delays'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs_2t4 = datafile['PEs']
SCs_2t4 = datafile['SCs']

fileheader = 'PE_SC_btwn4and6_start1700_length3400_499delays_499_delays'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs_4t6 = datafile['PEs']
SCs_4t6 = datafile['SCs']

fileheader = 'PE_SC_btwn6and8_start1700_length3400_499delays_499_delays'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs_6t8 = datafile['PEs']
SCs_6t8 = datafile['SCs']

fileheader = 'PE_SC_btwn5and7_start1700_length3400_499delays_499_delays'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs_5t7 = datafile['PEs']
SCs_5t7 = datafile['SCs']

#fileheader = 'PE_SC_binnedBy0p2_3p5to8p3_24bins_start1700_length3400_499delays_499_delays'
#datafile = loadnpzfile(datadir+fileheader+npy)
#PEs = datafile['PEs']
#SCs = datafile['SCs']

#fileheader = 'PE_SC_binnedBy0p5_3p5to8p5_10bins_start1700_length3400_499delays_499_delays_2000orblimit'
#datafile = loadnpzfile(datadir+fileheader+npy)
#PEs = datafile['PEs']
#SCs = datafile['SCs']

fileheader = 'PE_SC_binnedBy0p1_4to8_40bins_start1700_length3400_499delays_499_delays'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs = datafile['PEs']
SCs = datafile['SCs']

fileheader = 'PE_SC_1t_binnedBy0p1_3p5to8p5_50bins_start1700_length3400_499delays_499_delays'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs = datafile['PEs']
SCs = datafile['SCs']

delayindex = np.arange(1,501)
timeindex=(delayindex*1e5)/(1e6)

#plt.figure(2)
#plt.plot(timeindex[1:]/167,SCs_all[1:],color='black')
#plt.plot(timeindex[1:]/167,SCs_2t4[1:],color='blue')
#plt.plot(timeindex[1:]/167,SCs_4t6[1:],color='green')
#plt.plot(timeindex[1:]/167,SCs_6t8[1:],color='red')

"""
#colors = np.zeros([5,4])
#for i in np.arange(5):
#    c = cm.spectral(i/5.,1)
#    colors[i,:]=c
"""
points = ['o','v','s','p','*','h','^','D','+','>','H','d','x','<']

        
plt.rc('axes',linewidth=2.0)
plt.rc('xtick.major',width=2.0)
plt.rc('ytick.major',width=2.0)
plt.rc('xtick.minor',width=2.0)
plt.rc('ytick.minor',width=2.0)
plt.rc('lines',markersize=8,markeredgewidth=0.0,linewidth=3.0)

numcolors=5
colors=np.zeros([numcolors,4])
for i in np.arange(numcolors):
    c = plt.cm.plasma(i/(float(numcolors)),1)
    colors[i,:]=c

#plt.rcParams['ps.fonttype'] = 42
#plt.rcParams['pdf.fonttype'] = 42

fig=plt.figure(num=20,figsize=(9,6),dpi=600,facecolor='w',edgecolor='k')
left  = 0.16  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.15  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.0   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

delayindex = np.arange(1,501)
timeindex=(delayindex*1e5)/(1e6)
m6_dyntime_index=1676
dyntime=m6_dyntime_index/10
normtimeindex=timeindex/dyntime
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
plt.plot(normtimeindex[1:450],SCs[1:450,4],color=colors[0,:],linestyle='solid',label='3.88 kpc (LR)')#binlabels[0])
plt.plot(normtimeindex[1:450],SCs[1:450,15],color=colors[1,:],linestyle='solid',label='4.94 kpc (ULR)')#binlabels[1])
plt.plot(normtimeindex[1:450],SCs[1:450,20],color=colors[2,:],linestyle='solid',label='5.50 kpc')#binlabels[2])
plt.plot(normtimeindex[1:450],SCs[1:450,25],color=colors[3,:],linestyle='solid',label='6.00 kpc (CR)')#binlabels[3])
plt.plot(normtimeindex[1:450],SCs[1:450,30],color=colors[4,:],linestyle='solid',label='6.5 kpc')
#plt.plot(timeindex[1:450],SCs[1:450,46],color='purple',linestyle='solid',label='r=8.12')#binlabels[4])
#plt.plot(timeindex[1:450],SCs[1:450,25],color='black',linestyle='solid',label=binlabels[5])
#plt.plot(timeindex[1:450],SCs[1:450,30],color='saddlebrown',linestyle='solid',label=binlabels[6])
#plt.plot(timeindex[1:450],SCs[1:450,35],color='teal',linestyle='solid',label=binlabels[7])
#plt.plot(timeindex[1:450],SCs[1:450,40],color='yellowgreen',linestyle='solid',label=binlabels[8])
#plt.plot(timeindex[1:450],SCs[1:450,45],color='magenta',linestyle='solid',label=binlabels[9])
#plt.plot(timeindex,SCsT25[1:500],color=colors[1,:],label='Type 2 [Beyond CR]')
#plt.plot(timeindex,SCs31[1:],color='green',label='Type 3-1')
#plt.plot(timeindex,SCs32[1:],color='red',marker=points[1],markevery=(20,100),label='Type 3-2')
#plt.plot(timeindex,SCs32_resort[1:],color='red',linestyle='dashed')
#plt.plot(timeindex,SCs31[1:],color='green')
#plt.plot(timeindex,SCs31_resort[1:],color='green',linestyle='dashed')
#plt.plot(timeindex,SCs32_4CRresort[1:],color='purple')
#plt.plot(timeindex,SCs4[1:],color='purple',label='Type 4')
#plt.plot(delayindex,SCsin[1:],color='black',label='Sine Wave')

#plt.vlines(85,0,1,color='red',linestyle='dotted',linewidth=0.5)
#delayarray = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250])
#delayarray = np.array([0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500])
#timearray = (delayarray*1e5)/(1e6)
#timelist = list(timearray.astype(int))
#plt.xticks(np.array([1,20,40,60,80,100,120,140,160,180,200,220,240]),[1,20,40,60,80,100,120,140,160,180,200,220,240],fontsize=9)

#plt.xticks(timearray,timelist,fontsize=12)
#plt.yticks(np.array([0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40]),[0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40],fontsize=12)
plt.xlabel(r'$\tau_{s}/T^{M6e}_{dyn}$',fontsize=20)
plt.xticks(fontsize=18)
#plt.xlabel('Delay Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel(r'$C$',fontsize=20)
plt.yticks(fontsize=18)
plt.xlim(0.001,0.23)
plt.ylim(0.11,0.4)
plt.legend(loc='lower right',fontsize=14,frameon=False,handlelength=5,numpoints=2)
#plt.text(0.04,0.95,'(a)',fontsize=16,horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes)

#savefilename='SC_galpy0718_1000timesteps_3000_orbits.png'
#savefilename='SC_galpy0718_1000timesteps_all_orbits.png'
#savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')

# X, Y = np.meshgrid(xlist, ylist)
# Z = np.sqrt(X**2 + Y**2)
# fig,ax=plt.subplots(1,1)
# cp = ax.contourf(X, Y, Z)
# fig.colorbar(cp) # Add a colorbar to a plot
# ax.set_title('Filled Contours Plot')
# #ax.set_xlabel('x (cm)')
# ax.set_ylabel('y (cm)')
# plt.show()
savefilename='CR6_1t_Rg_Full_harmonicsSlices_1Td_to_2Td.eps'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()




#fig=plt.figure(num=2,figsize=(5,3.5),dpi=300,facecolor='w',edgecolor='k')
#plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
fig=plt.figure(num=33,figsize=(9,6),dpi=600,facecolor='w',edgecolor='k')
left  = 0.16  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.2  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.0   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
ax=plt.subplot(1,1,1)
bins=(np.arange(50)*0.1)+3.5
x,y=np.meshgrid(bins,normtimeindex[1:400])
cp = ax.contourf(x,y,SCs[1:400,:],levels=100,cmap=cm.plasma,vmin=0.25,vmax=0.4)
cbar=fig.colorbar(cp)
cbar.set_label('C',fontsize=20)#, rotation=270)

plt.vlines(6,0,40,color='white',linewidth=4.0)
plt.vlines(4.94,0,40,color='antiquewhite',linestyle='dotted',linewidth=4.0)#ULR
plt.vlines(7.06,0,40,color='antiquewhite',linestyle='dotted',linewidth=4.0)#ULR
plt.vlines(3.88,0,40,color='antiquewhite',linestyle='dashed',linewidth=4.0)#LR
plt.vlines(8.12,0,40,color='antiquewhite',linestyle='dashed',linewidth=4.0)#LR


plt.xlabel(r'$R_{L}(t=1T_{dyn}^{M6e})$ [kpc]',fontsize=20)
plt.ylabel(r'$\tau_{s}/T^{M6}_{dyn}$',fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.xlim(3.5,6.82)
plt.ylim(0.001,0.23)

#ax=fig.gca(projection='3d')
#ax=fig.gca()
#bins=(np.arange(40)*0.1)+4.0

#surf=ax.plot_surface(x,y,SCs[1:400,:],cmap=cm.coolwarm,vmin=0.15,linewidth=1,antialiased=False)
#ax.set_xlabel('radius [kpc]')
#ax.set_ylabel('time [Myr]')
#ax.set_zlabel('C')

#ax.view_init(elev=52.475996508582966,azim=-142.58241758241738)
#savefilename='CR6_1t_Rg_Full_0p1Myrbins_view1.png'
#savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')


#ax.view_init(elev=80,azim=-90)
savefilename='CR6_1t_Rg_Full_ComplexityContour_1Td_to_2Td.eps'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()
"""
ax1=plt.subplot(2,1,2)
plt.plot(timeindex,PEs1[1:500],color='blue',marker=points[0],markevery=(20,100),label='Type 1')
#plt.plot(timeindex,PEsT25[1:500],color=colors[1,:],label='Type 2 [Beyond CR]')
#plt.plot(timeindex,PEs31[1:],color='green',label='Type 3-1')
plt.plot(timeindex,PEs32[1:],color='red',marker=points[1],markevery=(20,100),label='Type 3-2')
#plt.plot(timeindex,PEs4[1:],color='purple',label='Type 4')
#plt.plot(delayindex,PEsin[1:],color='black',label='Sine Wave')

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
plt.legend(loc='lower right',fontsize=12,frameon=False,handlelength=5)
plt.text(0.04,0.95,'(b)',fontsize=16,horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes)
"""
"""
savefilename='SC_and_PE_CR6_2000timesteps_Type1vsType32_3_wresort.png'
#savefilename='PE_galpy0718_1000timesteps_all_orbits.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
#plt.clf()
#plt.close()
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