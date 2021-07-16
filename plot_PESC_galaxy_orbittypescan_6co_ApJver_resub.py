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

#datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\GalpyData_July2018\\'
datadir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Galatic Dynamics Data\\GalpyData_July2018\\resorted_data\\M6_3352_et_timescan\\'
npy='.npz'

delayindex = np.arange(1,501)#250
timestep_arr = [2050]
print(len(timestep_arr))
timeindex = (delayindex*1e5)/(1e6)

fileheader = 'PE_SC_Type_1_Rg_499_delays_3910orbits_of3910_total2000_timesteps_resorted_et'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1 = datafile['PEs']
SCs1 = datafile['SCs']
SCs1_endarray=500

fileheader = 'PE_SC_Type_2o_Rg_499_delays_5684orbits_of5684_total2000_timesteps_resorted_et'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs2o = datafile['PEs']
SCs2o = datafile['SCs']
SCs2o_endarray=500

fileheader = 'PE_SC_Type_2i_Rg_499_delays_23495orbits_of23495_total2000_timesteps_resorted_et'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs2i = datafile['PEs']
SCs2i = datafile['SCs']
SCs2i_endarray=500

fileheader = 'PE_SC_Type_31_Rg_499_delays_3279orbits_of4547_total2000_timesteps_resorted_et'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs31 = datafile['PEs']
SCs31 = datafile['SCs']
SCs31_endarray=500

fileheader = 'PE_SC_Type_32i_Rg_499_delays_3145orbits_of3899_total2000_timesteps_resorted_et'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32i = datafile['PEs']
SCs32i = datafile['SCs']
SCs32i_endarray=500

fileheader = 'PE_SC_Type_32o_Rg_499_delays_2021orbits_of2458_total2000_timesteps_resorted_et'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32o = datafile['PEs']
SCs32o = datafile['SCs']
SCs32o_endarray=500
    
M6dyn=1676
delayindex = np.arange(1,500)
timeindex=(delayindex*1e5)/(1e6)
timeindex_normM6=timeindex/(M6dyn/10)

numcolors=5
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
plt.rc('lines',markersize=12,markeredgewidth=0.0,linewidth=2.0)

#plt.rcParams['ps.fonttype'] = 42
#plt.rcParams['pdf.fonttype'] = 42

fig=plt.figure(num=1,figsize=(7,6),dpi=600,facecolor='w',edgecolor='k')
left  = 0.15  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.1   # the amount of width reserved for blank space between subplots
hspace = 0.17   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
plt.plot(timeindex_normM6,SCs1[1:500],color='blue',marker=points[0],markevery=(20,100),label='M6-Type 1')
plt.plot(timeindex_normM6,SCs2i[1:500],color='purple',marker=points[4],markevery=(20,100),label='M6-Type 2 [Inside CR]')
plt.plot(timeindex_normM6,SCs2o[1:500],color='teal',marker=points[7],markevery=(20,100),label='M6-Type 2 [Outside CR]')
plt.plot(timeindex_normM6,SCs31[1:],color='orange',marker=points[2],markevery=(20,100),label='M6-Type 3-1')
plt.plot(timeindex_normM6,SCs32i[1:],color='red',marker=points[1],markevery=(20,100),label='M6-Type 3-2i')
plt.plot(timeindex_normM6,SCs32o[1:],color='green',marker=points[3],markevery=(20,100),label='M6-Type 3-2o')
#plt.plot(timeindex,SCs4[1:],color='teal',marker=points[5],markevery=(20,100),label='M6-Type 4')
#plt.plot(delayindex,SCsin[1:],color='black',label='Sine Wave')

#plt.vlines(85,0,1,color='red',linestyle='dotted',linewidth=0.5)
plt.xticks(fontsize=12)
plt.yticks(np.array([0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40]),[0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40],fontsize=12)
plt.xlabel(r'$\tau_s/\tau_{dyn}^{M6}$',fontsize=15)
#plt.xlabel('Delay Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel(r'$C$',fontsize=15)
plt.xlim(0,0.25)
#plt.ylim(0,0.5)
plt.legend(loc='lower left',fontsize=12,frameon=False,handlelength=5)
#plt.text(0.07,0.92,'(a)',horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes,fontsize=12)


#savefilename='SC_CR6_typescan_2000_et_normdyn_ApJver.png'
savefilename='SC_CR6_typescan_2000_et_normdyn_ApJver.eps'
#savefilename='PE_galpy0718_1000timesteps_all_orbits.png'
savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()
#savefilename='SC_galpy0718_1000timesteps_3000_orbits.png'
#savefilename='SC_galpy0718_1000timesteps_all_orbits.png'
#savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')


#fig=plt.figure(num=2,figsize=(5,3.5),dpi=300,facecolor='w',edgecolor='k')
#plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

"""
ax1=plt.subplot(2,1,2)
plt.plot(timeindex,PEs1[1:500],color=colors[0,:],label='Type 1')
plt.plot(timeindex,PEsT25[1:500],color=colors[1,:],label='Type 2 [Beyond CR]')
plt.plot(timeindex,PEs31[1:],color='green',label='Type 3-1')
plt.plot(timeindex,PEs32[1:],color='red',label='Type 3-2')
plt.plot(timeindex,PEs4[1:],color='purple',label='Type 4')
#plt.plot(delayindex,PEsin[1:],color='black',label='Sine Wave')

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)
delayarray = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250])
delayarray = np.array([0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500])
timearray = (delayarray*1e5)/(1e6)
timelist = list(timearray.astype(int))
plt.xticks(timearray,timelist,fontsize=8)
plt.xlabel('Delay Time [Myr]',fontsize=11)
#ax1.set_xticklabels([])
plt.yticks(fontsize=9)
plt.ylabel('Permutation Entropy',fontsize=9)
#plt.xlim(1,250)
plt.ylim(0,1.0)
plt.legend(loc='lower right',fontsize=6,frameon=False,handlelength=5)
plt.text(0.07,0.92,'(b)',horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes)

savefilename='SC_and_PE_galpy0718_CR6_2000plustimesteps_3000plusorbits.eps'
#savefilename='PE_galpy0718_1000timesteps_all_orbits.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')

"""


plt.rc('text.latex', preamble=r'\usepackage{color}')
import matplotlib.pyplot as plt

#plt.figure()
#plt.ylabel(r'\textcolor{red}{Today} '+
#           r'\textcolor{green}{is} '+
#           r'\textcolor{blue}{cloudy.}')



numcolors=2
colors=np.zeros([numcolors,4])
for i in np.arange(numcolors):
    c = plt.cm.plasma(i/(float(numcolors)),1)
    colors[i,:]=c

ndim=5
#Plot C vs H as a CHplane
Cminx, Cminy, Cmaxx, Cmaxy = cpl.Cmaxmin(1000,ndim)
plt.rc('lines',markersize=10,markeredgewidth=0.0)

fig=plt.figure(num=33,figsize=(7,6),dpi=600,facecolor='w',edgecolor='k')
left  = 0.15  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.1   # the amount of width reserved for blank space between subplots
hspace = 0.17   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

ax=plt.subplot(1,1,1)

plt.plot(Cminx,Cminy,'k-',Cmaxx,Cmaxy,'k-')

#timestep error
#C_std_RO[1]=0.001
#H_std_RO[1]=0.002
#C_std_CR[1]=0.002
#H_std_CR[1]=0.003

#C_std_R0[93]=0.008
#H_std_RO[93]=0.017
#C_std_CR[93]=0.018
#H_std_CR[93]=0.03

#C_std_RO[400]=0.013
#H_std_RO[400]=0.03
#C_std_CR[400]=0.032
#H_std_CR[400]=0.06

#timestep10 = 1
#plt.errorbar(PEs32i[timestep10],SCs32i[timestep10],xerr=0.002,yerr=0.001,elinewidth=0.5,color='blue',marker='s')#,label=r'$\tau_{s}/T_{dyn}^{M6}=$'+str(np.round(timestep0/M6dyn,3)))

#plt.errorbar(PEs1[timestep10],SCs1[timestep10],xerr=0.003,yerr=0.002,elinewidth=0.5,color='red',marker='s')#,label=r'$\tau_{s}/T_{dyn}^{M6}=$'+str(np.round(timestep0/M6dyn,3)))
#plt.plot(PEsT21[timestep0],SCsT21[timestep0],color='purple',marker='H')
#plt.plot(PEsT25[timestep0],SCsT25[timestep0],color='teal',marker='H')
#plt.plot(PEs31[timestep0],SCs31[timestep0],color='orange',marker='H')


timestep93 = 93
plt.plot(PEs1[0:93],SCs1[0:93],color=colors[0,:],linestyle='solid',linewidth=2.0,label='Type CR')
plt.plot(PEs32i[0:93],SCs32i[0:93],color=colors[1,:],linestyle='solid',linewidth=2.0,label='Type RO')
plt.errorbar(PEs1[timestep93],SCs1[timestep93],xerr=0.03,yerr=0.018,elinewidth=0.5,color=colors[0,:],linestyle='solid',marker=points[0])#,label=r'$\tau_{s}/T_{dyn}^{M6}=$'+str(np.round(timestep93/M6dyn,3)))

plt.plot(PEs1[93:-1],SCs1[93:-1],linestyle='dashed',linewidth=0.75,color=colors[0,:])
plt.plot(PEs32i[93:-1],SCs32i[93:-1],linestyle='dashed',linewidth=0.75,color=colors[1,:])


#plt.plot(PEsT21[timestep],SCsT21[timestep],color='purple',marker=points[4],label='Type 2 [Inside CR]')
#plt.plot(PEsT25[timestep],SCsT25[timestep],color='teal',marker=points[7],label='Type 2 [Outside CR]')
#plt.plot(PEs31[timestep],SCs31[timestep],color='orange',marker=points[2],label='Type 3-1')
plt.errorbar(PEs32i[timestep93],SCs32i[timestep93],xerr=0.017,yerr=0.008,elinewidth=0.5,color=colors[1,:],marker=points[0])#,label=r'$\tau_{s}/T_{dyn}^{M6}=$'+str(np.round(timestep93/M6dyn,3)))
#plt.plot(PEs4[timestep],SCs4[timestep],color='purple',marker='o',label='Type 4')
#plt.plot(PEsin[81],SCsin[81],color='black',marker='o',label='Sine Wave, Delay 81')

#timestep400 = 400
#plt.errorbar(PEs1[timestep400],SCs1[timestep400],xerr=0.06,yerr=0.032,elinewidth=0.5,color='blue',marker=points[1],label=r'$t_{pat}/T_{dyn}^{M6e}=$'+str(np.round(timestep400/M6dyn,3)))
#plt.errorbar(PEs32i[timestep400],SCs32i[timestep400],xerr=0.03,yerr=0.013,elinewidth=0.5,color='red',marker=points[1])#,label=r'$\tau_{s}/T_{dyn}^{M6}=$'+str(np.round(timestep400/M6dyn,3)))

plt.xlabel(r'$H$', fontsize=20)
plt.ylabel(r'$C$', fontsize=20)
#plt.title('Delay Timescale '+str(timeindex[timestep])+' Myr',fontsize=9)
plt.axis([0,1.0,0,0.45])
plt.xticks(np.arange(0,1.1,0.1),fontsize=18)
plt.yticks(np.arange(0,0.45,0.05),fontsize=18)
plt.legend(loc='lower center',fontsize=12,numpoints=3,frameon=False,handlelength=5)

#leg._legend_box.align = "center"
#leg.set_title('Type 1 (Blue), Type 3-2 (Red)',prop={'size':9})

#leg.legendHandles[0].set_markerfacecolor('green')
#leg.legendHandles[1].set_markerfacecolor('green')
#leg.legendHandles[2].set_markerfacecolor('green')

#from matplotlib.lines import Line2D
#legend_elements = [Line2D([0], [0], linestyle=None,markerfacecolor='black', markersize=10,marker='s',label=r'  $t_{pat}/T_{dyn}^{M6e}=$'+str(np.round(4*timestep10/M6dyn,4))),
#                   Line2D([0], [0], linestyle=None,markerfacecolor='black', markersize=10,marker=points[0],label=r'  $t_{pat}/T_{dyn}^{M6e}=$'+str(np.round(4*timestep93/M6dyn,3))),
#                   Line2D([0], [0], linestyle=None,markerfacecolor='black', markersize=10,marker=points[1],label=r'  $t_{pat}/T_{dyn}^{M6e}=$'+str(np.round(4*timestep400/M6dyn,3)))]

# Create the figure
#fig, ax = plt.subplots()
#leg=plt.legend(handles=legend_elements, loc='lower center',handlelength=0,frameon=False,fontsize=12)
#leg.set_title(r'Type CR (blue), Type RO (red)',prop={'size':12})

plt.vlines(0.5512,0.21,0.417,color='gray',linewidth=1.5,linestyle='dashed')


#leg.set_title(r'For $\tau_{s}/T_{dyn}^{M6}=$'+str(np.round(timestep/M6dyn,3)),prop={'size':12})
#savefilename='CH_M6_timestep'+str(timestep)+'_ApJver.png'
savefilename='CH_M6_3timesteps_start_peak_end_ApJver_newcolor_wtracks_wtpat_werrbarr.eps'
#savefilename='CH_M6_timestep'+str(timestep)+'_ApJver_newcolor.eps'
#savefilename='CH_galpy0718_1000timesteps_timestep'+str(timestep)+'_all_orbits.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()

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