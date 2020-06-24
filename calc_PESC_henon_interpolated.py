# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:00:45 2017

@author: dschaffner
"""
import numpy as np
import matplotlib.pylab as plt
from loadnpzfile import loadnpzfile
from calc_PE_SC import PE, CH

def henonsave(N):

    import numpy as np
    import scipy as sp
    import matplotlib.pylab as plt
    X = np.zeros((2,N))
    X[0,0] = 1.
    X[1,0] = 1.
    a = 1.4
    b = 0.3
    for i in range(1,N):
        X[0,i] = 1. - a * X[0,i-1] ** 2. + X[1,i-1]
        X[1,i] = b * X[0,i-1]
    return X

#calc_PESC_fluid.py
datadir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Corrsin Wind Tunnel Data\\NPZ_files\\m20_5mm\\streamwise\\'

data = henonsave(5000)
data=data[0,:]
#timesteps = np.linspace(1,data.shape[0],data.shape[0])
timesteps = np.arange(1,data.shape[0]+1,1)
#newtimesteps = np.linspace(1,data.shape[0],(data.shape[0]*100))
newtimesteps = np.arange(1,data.shape[0]+1,1/10.0)
data_interp = np.interp(newtimesteps,timesteps,data)
#xaxis_full = np.linspace(1,1000,100000)
#noise_array=np.random.uniform(1,1000,size=5000)
#noise_interp=np.interp(xaxis_full,xaxis,noise_array)        





###Storage Arrays###
delta_t = 1.0
delays = np.arange(1,500) #248 elements
taus = delays*delta_t
freq = 1.0/taus
PEs = np.zeros([500])#,121])
SCs = np.zeros([500])#,121])
PEs_interp = np.zeros([500])#,121])
SCs_interp = np.zeros([500])#,121])

for d in delays:
    print ('On Delay ', d)
    PEs[d],SCs[d] = CH(data,5,delay=d)
    PEs_interp[d],SCs_interp[d] = CH(data_interp,5,delay=d)

filename='Henonmap_5kpts_interpTimes10_500taus.npz'
np.savez(datadir+filename,PEs_interp=PEs_interp,SCs_interp=SCs_interp,PEs=PEs,SCs=SCs,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
