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

#calc_PESC_solarwind_chen.py
datadir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Galatic Dynamics Data\\DoublePendulum\\nPend\\'



embeddelay = 5
nfac = factorial(embeddelay)
    
    ###Storage Arrays###
    #delta_t = 1.0
    #delays = np.arange(1,101) #248 elements
    #taus = delays*delta_t
    #freq = 1.0/taus
#xaxis = np.linspace(1,10,10000)
#xaxis_full = np.linspace(1,10,100000)
#noise_array=np.random.uniform(1,10,size=10000)
#noise_interp=np.interp(xaxis_full,xaxis,noise_array)    
xaxis = np.linspace(1,1000,5000)
xaxis_full = np.linspace(1,1000,1000000)
noise_array=np.random.uniform(1,1000,size=5000)
noise_interp=np.interp(xaxis_full,xaxis,noise_array)        
    
    #delay_array = np.array([1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000,50000])
delay_array = np.arange(1,1000)
num_delays = len(delay_array)

PEs = np.zeros([num_delays])
SCs = np.zeros([num_delays])
PEs_interp = np.zeros([num_delays])
SCs_interp = np.zeros([num_delays])

"""  
for loop_delay in np.arange(len(delay_array)):
    #for loop_delay in np.arange(150,151):
    if (loop_delay%100)==0: print ('On Delay ',delay_array[loop_delay])
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    arr,nperms = PE_dist(noise_interp,embeddelay,delay=delay_array[loop_delay])
    permstore_counter = permstore_counter+arr
    tot_perms = tot_perms+nperms
    PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
    C =  -2.*((PE_tot_Se - 0.5*PE_tot - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot/np.log2(nfac)))
    PEs[loop_delay]=PE_tot/np.log2(nfac)
    SCs[loop_delay]=C
print( 'x1 completed')
"""

for loop_delay in delay_array:
    print ('On Delay ', loop_delay)
    PEs[loop_delay],SCs[loop_delay] = CH(noise_array,5,delay=loop_delay)
    PEs_interp[loop_delay],SCs_interp[loop_delay] = CH(noise_interp,5,delay=loop_delay)

filename='PE_SC_interpolated_noise_100MInto5k_embeddelay'+str(embeddelay)+'_'+str(num_delays)+'_delays.npz'
np.savez(datadir+filename,PEs=PEs,SCs=SCs,
         PEs_interp=PEs_interp,SCs_interp=SCs_interp,
                              #PEsx2=PEsx2,SCsx2=SCsx2,
                              #PEsx3=PEsx3,SCsx3=SCsx3, 
                              #PEsx4=PEsx4,SCsx4=SCsx4, 
                              #PEsx5=PEsx5,SCsx5=SCsx5, 
                              #PEsy2=PEsy2,SCsy2=SCsy2, 
                              #PEsy3=PEsy3,SCsy3=SCsy3, 
                              #PEsy4=PEsy4,SCsy4=SCsy4, 
                              delays=delay_array)
