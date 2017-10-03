# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:00:45 2017

@author: dschaffner
"""
import numpy as np
import matplotlib.pylab as plt
from loadnpyfile import loadnpyfile
from calc_PE_SC import PE, CH, PE_dist, PE_calc_only
import Cmaxmin as cpl
from collections import Counter
from math import factorial

x=np.arange(100000)
y=np.sin(0.003*x)

num_delays = 249
PEs = np.zeros([num_delays+1])
SCs = np.zeros([num_delays+1])

delay = 1
embed_delay = 5
nfac = factorial(embed_delay)

for loop_delay in np.arange(1,num_delays+1):
    
    PEs[loop_delay],SCs[loop_delay] = CH(y,5,delay=loop_delay)
    print 'On Delay ',loop_delay


datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\'    
filename='PE_SC_sinewave_'+str(num_delays)+'_delays.npz'
np.savez(datadir+filename,PEs=PEs,SCs=SCs)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
