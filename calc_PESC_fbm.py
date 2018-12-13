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

tags = [1]

h='0.1'
N='20000'
npz='.npz'

delay_array = np.arange(1,1000)
num_delays = len(delay_array)
PEs = np.zeros([num_delays])
SCs = np.zeros([num_delays])

embeddelay = 5
nfac = factorial(embeddelay)

for loop_delay in np.arange(len(delay_array)):
#for loop_delay in np.arange(150,151):
    if (loop_delay%100)==0: print 'On Delay ',delay_array[loop_delay]
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    
    for tag in tags:
        print 'On Tag ', tag
        fileheader = 'fbm_H'+h+'_N'+N+'_'+str(tag)
        datafile = loadnpzfile(fileheader+npz)
        x=datafile['x']
        arr,nperms = PE_dist(x,embeddelay,delay=delay_array[loop_delay])
        permstore_counter = permstore_counter+arr
        tot_perms = tot_perms+nperms
    PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
    C =  -2.*((PE_tot_Se - 0.5*PE_tot - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot/np.log2(nfac)))
    PEs[loop_delay]=PE_tot/np.log2(nfac)
    SCs[loop_delay]=C

filename='PE_SC_fbm'+fileheader+'_embeddelay'+str(embeddelay)+'_'+str(num_delays)+'_delays.npz'
np.savez(filename,PEs=PEs,SCs=SCs,
                              #PEsx2=PEsx2,SCsx2=SCsx2,
                              #PEsx3=PEsx3,SCsx3=SCsx3, 
                              #PEsx4=PEsx4,SCsx4=SCsx4, 
                              #PEsy1=PEsy1,SCsy1=SCsy1, 
                              #PEsy2=PEsy2,SCsy2=SCsy2, 
                              #PEsy3=PEsy3,SCsy3=SCsy3, 
                              #PEsy4=PEsy4,SCsy4=SCsy4, 
                              delays=delay_array)
