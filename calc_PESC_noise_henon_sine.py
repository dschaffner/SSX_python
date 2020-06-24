# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:00:45 2017

@author: dschaffner
"""
import numpy as np
import matplotlib.pylab as plt
import scipy as sp
from loadnpzfile import loadnpzfile
from calc_PE_SC import PE, CH, PE_dist, PE_calc_only
from collections import Counter
from math import factorial

#calc_PESC_solarwind_chen.py
datadir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Galatic Dynamics Data\\Examples\\'


#NOISE
embeddelay = 5
nfac = factorial(embeddelay)
    
#NOISE
noise_array=np.random.uniform(-1,1,size=100000)
delay_array = np.arange(1,1000)
num_delays = len(delay_array)+1
PEs_noise100k = np.zeros(num_delays)
SCs_noise100k = np.zeros(num_delays)
for loop_delay in delay_array:
    print ('On Noise Delay ', loop_delay)
    PEs_noise100k[loop_delay],SCs_noise100k[loop_delay] = CH(noise_array,5,delay=loop_delay)
filename='PESC_noise100k_embed5_'+str(num_delays)+'_delays.npz'
np.savez(datadir+filename,timeseries=noise_array,taus=delay_array,PEs=PEs_noise100k,SCs=SCs_noise100k)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)


#HENON
def henonsave(N):
    X = np.zeros((2,N))
    X[0,0] = 1.
    X[1,0] = 1.
    a = 1.4
    b = 0.3
    for i in range(1,N):
        X[0,i] = 1. - a * X[0,i-1] ** 2. + X[1,i-1]
        X[1,i] = b * X[0,i-1]
    return X

henon_array=henonsave(100000)
henon_array=henon_array[0,:]
PEs_henon100k = np.zeros([num_delays])
SCs_henon100k = np.zeros([num_delays])
for loop_delay in delay_array:
    print ('On Henon Delay ', loop_delay)
    PEs_henon100k[loop_delay],SCs_henon100k[loop_delay] = CH(henon_array,5,delay=loop_delay)
filename='PESC_henon100k_embed5_'+str(num_delays)+'_delays.npz'
np.savez(datadir+filename,timeseries=henon_array,taus=delay_array,PEs=PEs_henon100k,SCs=SCs_henon100k)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)



#SINE
x=np.arange(100000)*0.003
y1=np.sin(1*x)
y20=np.sin(20*x)
PEs_sine1x = np.zeros([num_delays])
SCs_sine1x = np.zeros([num_delays])
PEs_sine20x = np.zeros([num_delays])
SCs_sine20x = np.zeros([num_delays])
for loop_delay in delay_array:
    print ('On Sine Delay ', loop_delay)
    PEs_sine1x[loop_delay],SCs_sine1x[loop_delay] = CH(y1,5,delay=loop_delay)
    PEs_sine20x[loop_delay],SCs_sine20x[loop_delay] = CH(y20,5,delay=loop_delay)
filename='PESC_sine1x_embed5_'+str(num_delays)+'_delays.npz'
np.savez(datadir+filename,timeseries=y1,taus=delay_array,PEs=PEs_sine1x,SCs=SCs_sine1x)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
filename='PESC_sine20x_embed5_'+str(num_delays)+'_delays.npz'
np.savez(datadir+filename,timeseries=y20,taus=delay_array,PEs=PEs_sine20x,SCs=SCs_sine20x)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
