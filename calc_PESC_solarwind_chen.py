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
datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Solar Wind Data\\NPZ_files\\'
fileheader = 'Cluster20030212'
npz='.npz'
datafile = loadnpzfile(datadir+fileheader+npz)

embeddelay = 5
nfac = factorial(embeddelay)

###Storage Arrays###
#delta_t = 1.0
#delays = np.arange(1,101) #248 elements
#taus = delays*delta_t
#freq = 1.0/taus



delay_array = np.array([1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000])
num_delays = len(delay_array)

PEsbx = np.zeros([num_delays])
SCsbx = np.zeros([num_delays])
PEsby = np.zeros([num_delays])
SCsby = np.zeros([num_delays])
PEsbz = np.zeros([num_delays])
SCsbz = np.zeros([num_delays])

for loop_delay in np.arange(len(delay_array)):
    print 'On Delay for Bx',delay_array[loop_delay]
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    arr,nperms = PE_dist(datafile['bx'],embeddelay,delay=delay_array[loop_delay])
    permstore_counter = permstore_counter+arr
    tot_perms = tot_perms+nperms
    PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
    C =  -2.*((PE_tot_Se - 0.5*PE_tot - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot/np.log2(nfac)))
    PEsbx[loop_delay]=PE_tot/np.log2(nfac)
    SCsbx[loop_delay]=C
    
for loop_delay in np.arange(len(delay_array)):
    print 'On Delay for By',delay_array[loop_delay]
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    arr,nperms = PE_dist(datafile['by'],embeddelay,delay=delay_array[loop_delay])
    permstore_counter = permstore_counter+arr
    tot_perms = tot_perms+nperms
    PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
    C =  -2.*((PE_tot_Se - 0.5*PE_tot - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot/np.log2(nfac)))
    PEsby[loop_delay]=PE_tot/np.log2(nfac)
    SCsby[loop_delay]=C
    
for loop_delay in np.arange(len(delay_array)):
    print 'On Delay for Bz',delay_array[loop_delay]
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    arr,nperms = PE_dist(datafile['bz'],embeddelay,delay=delay_array[loop_delay])
    permstore_counter = permstore_counter+arr
    tot_perms = tot_perms+nperms
    PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
    C =  -2.*((PE_tot_Se - 0.5*PE_tot - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot/np.log2(nfac)))
    PEsbz[loop_delay]=PE_tot/np.log2(nfac)
    SCsbz[loop_delay]=C

#savedir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\DSCOVR Data\\NPZ_files\\'
filename='PE_SC_ClusterChen_embeddelay'+str(embeddelay)+'_'+str(num_delays)+'_delays.npz'
np.savez(datadir+filename,PEsbx=PEsbx,SCsbx=SCsbx,PEsby=PEsby,SCsby=SCsby,PEsbz=PEsbz,SCsbz=SCsbz,delays=delay_array)
