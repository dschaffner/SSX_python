# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:00:45 2017

@author: dschaffner
"""
import numpy as np
import matplotlib.pylab as plt
from loadnpzfile import loadnpzfile
from calc_PE_SC import PE, CH
import spectrum_wwind as spec

#calc_PESC_fluid.py

datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Corrsin Wind Tunnel Data\\NPZ_files\\m20_5mm\\streamwise\\'
fileheader = 'm20_5mm_p2shot'
npz='.npz'

###Storage Arrays###
delta_t = 1.0/(40000.0)
delays = np.arange(2,250) #248 elements
taus = delays*delta_t
freq = 1.0/taus
PEs = np.zeros([248,121])
SCs = np.zeros([248,121])


ave_spec = np.zeros(150001)

for shot in np.arange(1,121):#(1,120):
    print '###### On Shot '+str(shot)+' #####'
    datafile = loadnpzfile(datadir+fileheader+str(shot)+npz)
    data = datafile['shot']
    print data.shape
    time = np.arange(data.shape[0])*delta_t
    freq,freq2,comp,pwr,mag,phase,cos_phase,dt = spec.spectrum_wwind(data,time,window='hanning')
    ave_spec = ave_spec+pwr
    

#filename='PE_SC_m20_5mm_embed5_p2.npz'
#np.savez(datadir+filename,PEs=PEs,SCs=SCs,delta_t=delta_t,taus=taus,delays=delays,freq=freq)

fig=plt.figure(num=2,figsize=(3.5,2.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.2  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.2  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

ax1=plt.subplot(1,1,1)
plt.plot(time,data,color='black')

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel(r't[s]',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Velocity [m/s]',fontsize=9)
#plt.xlim(1,10)
#plt.ylim(0,0.5)
#plt.grid(b=True,which='both',axis='x',linestyle='dotted',linewidth=0.05)
#plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)


savefilename='C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\DSCOVR Data\\fluidtimeseries.png'
#savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefilename,dpi=300,facecolor='w',edgecolor='k')


fig=plt.figure(num=3,figsize=(3.5,2.25),dpi=300,facecolor='w',edgecolor='k')
left  = 0.2  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.2  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

ax1=plt.subplot(1,1,1)
plt.loglog(1.0/freq,ave_spec,color='black')

#plt.vlines(81,0,1,color='red',linestyle='dotted',linewidth=0.5)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel(r'f[Hz]',fontsize=9)
plt.xlabel(r'tau[s]',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Power',fontsize=9)
plt.ylim(1e-5,1e2)
plt.xlim(5e-5,1e-2)
#plt.grid(b=True,which='both',axis='x',linestyle='dotted',linewidth=0.05)
#plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)
plt.grid(b=True,which='both',axis='x',linestyle='dotted',linewidth=0.05)

savefilename='C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\DSCOVR Data\\fluidspectra_wtau.png'
#savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefilename,dpi=300,facecolor='w',edgecolor='k')