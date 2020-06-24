# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:00:45 2017

@author: dschaffner
"""
import numpy as np
import matplotlib.pylab as plt
from loadnpzfile import loadnpzfile
from calc_PE_SC import PE, CH

#calc_PESC_fluid.py

datadir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Corrsin Wind Tunnel Data\\NPZ_files\\m20_5mm\\streamwise\\'
fileheader = 'm20_5mm_p2shot'
npz='.npz'

shot=1
datafile = loadnpzfile(datadir+fileheader+str(shot)+npz)
data = datafile['shot'][:20000]
#timesteps = np.linspace(1,data.shape[0],data.shape[0])
timesteps = np.arange(1,data.shape[0]+1,1)
#newtimesteps = np.linspace(1,data.shape[0],data.shape[0]*200)
newtimesteps = np.arange(1,data.shape[0]+1,1/5.0)
data_interp = np.interp(newtimesteps,timesteps,data)
#xaxis_full = np.linspace(1,1000,100000)
#noise_array=np.random.uniform(1,1000,size=5000)
#noise_interp=np.interp(xaxis_full,xaxis,noise_array)        
 

###Storage Arrays###
delta_t = 1.0/(40000.0)
delays = np.arange(1,500) #248 elements
taus = delays*delta_t
freq = 1.0/taus
PEs = np.zeros([500,2])#,121])
SCs = np.zeros([500,2])#,121])


for shot in np.arange(1,2):#(1,120):
    print ('###### On Shot '+str(shot)+' #####')
    datafile = loadnpzfile(datadir+fileheader+str(shot)+npz)
    data = datafile['shot'][:20000]
    timesteps = np.arange(1,data.shape[0]+1,1)
    newtimesteps = np.arange(1,data.shape[0]+1,1/5.0)
    data_interp = np.interp(newtimesteps,timesteps,data)
    for d in delays:
        print ('On Delay ', d)
        PEs[d,shot],SCs[d,shot] = CH(data_interp,5,delay=d)

filename='PE_SC_m20_5mm_embed5_p2_first20kPts_interpTimes5_500taus.npz'
np.savez(datadir+filename,PEs=PEs,SCs=SCs,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
