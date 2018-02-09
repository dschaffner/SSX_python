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
import spectrum_wwind as spec

#calc_PESC_DSCOVR.py
days=['052417',
      '052517',
      '052617',
      '052717',
      '052817',
      '052917',
      '060217',
      '060317',
      '060517',
      '061017',
      '061717',
      '061917',
      '062017',
      '062917',
      '063017',
      '071217',
      '071317',
      '071417',
      '071517',
      '072717',
      '072917',
      '080317']
ndays = len(days)
datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\DSCOVR Data\\NPZ_files\\'
magheader='mag_gse_1sec_'
velheader='proton_speed_3sec_'
npz='.npz'

#datatype = 'proton_vz_gse'
datatype = 'bz_gse'
if datatype == 'bx': bcomp=0
if datatype == 'by': bcomp=1
if datatype == 'bz': bcomp=2  
if datatype == 'bt': bcomp=3  
if datatype == 'vx': vcomp=1
if datatype == 'bx_gse' or datatype == 'by_gse' or datatype == 'bz_gse' or datatype == 'bt': 
    fileheader=magheader
    timelabel = '1s'
if datatype == 'proton_vx_gse' or datatype == 'proton_vy_gse' or datatype == 'proton_vz_gse': 
    fileheader=velheader
    timelabel = '3s'

print fileheader, timelabel

embeddelay = 5
nfac = factorial(embeddelay)

###Storage Arrays###
delta_t = 1.0
delays = np.arange(1,101) #248 elements
taus = delays*delta_t
freq = 1.0/taus
num_delays = 310
PEs = np.zeros([num_delays+1])
SCs = np.zeros([num_delays+1])


ave_spec = np.zeros(43201)#b
#ave_spec = np.zeros(2161)#vel

for day in np.arange(ndays):#(1,120):
    print '###### On Day '+days[day]+' #####'
    datafile = loadnpzfile(datadir+fileheader+days[day]+npz)
    b = datafile[datatype]
    time = np.arange(b.shape[0])
    freq,freq2,comp,pwr,mag,phase,cos_phase,dt = spec.spectrum_wwind(b,time,window='hanning')
    ave_spec = ave_spec+pwr
    

#savedir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\DSCOVR Data\\NPZ_files\\'
#filename='PE_SC_DSCOVR_'+datatype+'_'+timelabel+'_embeddelay'+str(embeddelay)+'_over'+str(ndays)+'_days.npz'
#np.savez(savedir+filename,PEs=PEs,SCs=SCs)



fig=plt.figure(num=2,figsize=(3.5,2.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.2  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.2  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

ax1=plt.subplot(1,1,1)
plt.plot(time,b,color='black')

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel(r't[s]',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Velocity',fontsize=9)
#plt.ylabel('Magnetic Field',fontsize=9)
plt.xlim(0,2000)
plt.ylim(-100,50)
#plt.ylim(-10,10)
#plt.grid(b=True,which='both',axis='x',linestyle='dotted',linewidth=0.05)
#plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)


savefilename='C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\DSCOVR Data\\DSCOVRtimeseries_vel.png'
#savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefilename,dpi=300,facecolor='w',edgecolor='k')


fig=plt.figure(num=3,figsize=(3.5,2.0),dpi=300,facecolor='w',edgecolor='k')
left  = 0.2  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.2  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

ax1=plt.subplot(1,1,1)
plt.loglog(1.0/freq,ave_spec,color='black')
#plt.loglog(freq,5e11*freq**(-5/3),color='red')

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
#plt.xlabel(r'f[Hz]',fontsize=9)
plt.xlabel(r'tau[s]',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Power',fontsize=9)
#plt.ylim(1e11,1e16)
plt.xlim(1e0,3e2)
#plt.grid(b=True,which='both',axis='x',linestyle='dotted',linewidth=0.05)
#plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)
plt.grid(b=True,which='both',axis='x',linestyle='dotted',linewidth=0.05)

savefilename='C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\DSCOVR Data\\DSCOVRpectra_magwtau.png'
#savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefilename,dpi=300,facecolor='w',edgecolor='k')