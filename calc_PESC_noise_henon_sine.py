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

embeddelay = 5
nfac = factorial(embeddelay)

#NOISE
noise_array=np.random.uniform(-1,1,size=100000)
delay_array = np.arange(1,1000)
num_delays = len(delay_array)+1

"""
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

"""

#SINE
x=np.arange(100000)*0.003
y1=np.sin(1*x)
y20=np.sin(20*x)
y50=np.sin(50*x)
y100=np.sin(100*x)
y200=np.sin(200*x)
y500=np.sin(500*x)
y1000=np.sin(1000*x)


ysum1=0.333*y1+0.333*y20+0.333*y50#+0.25*y100
#plt.plot(ysum1)

PEs_sine1x = np.zeros([num_delays])
SCs_sine1x = np.zeros([num_delays])
PEs_sine20x = np.zeros([num_delays])
SCs_sine20x = np.zeros([num_delays])
PEs_sine50x = np.zeros([num_delays])
SCs_sine50x = np.zeros([num_delays])
PEs_sine100x = np.zeros([num_delays])
SCs_sine100x = np.zeros([num_delays])
PEs_sine200x = np.zeros([num_delays])
SCs_sine200x = np.zeros([num_delays])
PEs_sine500x = np.zeros([num_delays])
SCs_sine500x = np.zeros([num_delays])
PEs_sine1000x = np.zeros([num_delays])
SCs_sine1000x = np.zeros([num_delays])
PEs_sine_sum1 = np.zeros([num_delays])
SCs_sine_sum1 = np.zeros([num_delays])
for loop_delay in delay_array:
    print ('On Sine Delay ', loop_delay)
    #PEs_sine1x[loop_delay],SCs_sine1x[loop_delay] = CH(y1,5,delay=loop_delay)
    #PEs_sine20x[loop_delay],SCs_sine20x[loop_delay] = CH(y20,5,delay=loop_delay)
    #PEs_sine50x[loop_delay],SCs_sine50x[loop_delay] = CH(y50,5,delay=loop_delay)
    #PEs_sine100x[loop_delay],SCs_sine100x[loop_delay] = CH(y100,5,delay=loop_delay)    
    #PEs_sine200x[loop_delay],SCs_sine200x[loop_delay] = CH(y200,5,delay=loop_delay)
    #PEs_sine500x[loop_delay],SCs_sine500x[loop_delay] = CH(y500,5,delay=loop_delay)
    #PEs_sine1000x[loop_delay],SCs_sine1000x[loop_delay] = CH(y1000,5,delay=loop_delay)
    PEs_sine_sum1[loop_delay],SCs_sine_sum1[loop_delay]=CH(ysum1,5,delay=loop_delay)
filename='PESC_sine1x_embed5_'+str(num_delays)+'_delays.npz'
#np.savez(datadir+filename,timeseries=y1,taus=delay_array,PEs=PEs_sine1x,SCs=SCs_sine1x)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
filename='PESC_sine20x_embed5_'+str(num_delays)+'_delays.npz'
#np.savez(datadir+filename,timeseries=y20,taus=delay_array,PEs=PEs_sine20x,SCs=SCs_sine20x)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
filename='PESC_sine50x_embed5_'+str(num_delays)+'_delays.npz'
#np.savez(datadir+filename,timeseries=y50,taus=delay_array,PEs=PEs_sine50x,SCs=SCs_sine50x)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
filename='PESC_sine100x_embed5_'+str(num_delays)+'_delays.npz'
#np.savez(datadir+filename,timeseries=y100,taus=delay_array,PEs=PEs_sine100x,SCs=SCs_sine100x)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
filename='PESC_sine200x_embed5_'+str(num_delays)+'_delays.npz'
#np.savez(datadir+filename,timeseries=y200,taus=delay_array,PEs=PEs_sine200x,SCs=SCs_sine200x)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
filename='PESC_sine500x_embed5_'+str(num_delays)+'_delays.npz'
#np.savez(datadir+filename,timeseries=y500,taus=delay_array,PEs=PEs_sine500x,SCs=SCs_sine500x)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
filename='PESC_sine1000x_embed5_'+str(num_delays)+'_delays.npz'
#np.savez(datadir+filename,timeseries=y1000,taus=delay_array,PEs=PEs_sine1000x,SCs=SCs_sine1000x)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
filename='PESC_sine_sum1_embed5_'+str(num_delays)+'_delays.npz'
np.savez(datadir+filename,timeseries=ysum1,taus=delay_array,PEs=PEs_sine_sum1,SCs=SCs_sine_sum1)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)



"""
#Triangle Waveform
import scipy.signal as sig
x=np.arange(100000)*0.95
triangle=sig.sawtooth(x,0.5)
plt.plot(x,triangle,'o')

PEs_triangle = np.zeros([num_delays])
SCs_triangle = np.zeros([num_delays])

for loop_delay in delay_array:
    print ('On Sine Delay ', loop_delay)
    PEs_triangle[loop_delay],SCs_triangle[loop_delay] = CH(triangle,5,delay=loop_delay)

filename='PESC_triangle_embed5_'+str(num_delays)+'_delays.npz'
np.savez(datadir+filename,timeseries=triangle,taus=delay_array,PEs=PEs_triangle,SCs=SCs_triangle)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
"""


"""
#fbm 0.5
data=loadnpzfile(datadir+'fbm_H0.05_N10000_1.npz')
y=data['x']
PEs_fbm_p5 = np.zeros([num_delays])
SCs_fbm_p5 = np.zeros([num_delays])

for loop_delay in delay_array:
    print ('On Sine Delay ', loop_delay)
    PEs_fbm_p5[loop_delay],SCs_fbm_p5[loop_delay] = CH(y,5,delay=loop_delay)

filename='PESC_fbm_p5_embed5_'+str(num_delays)+'_delays.npz'
np.savez(datadir+filename,timeseries=y,taus=delay_array,PEs=PEs_fbm_p5,SCs=SCs_fbm_p5)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
"""

