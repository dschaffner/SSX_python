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

datatype = 'proton_vz_gse'
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


for loop_delay in np.arange(1,num_delays+1):
    if np.mod(loop_delay,10)==0: print 'On Delay',loop_delay
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    for day in np.arange(ndays):#(1,120):
        print '###### On Day '+days[day]+' #####'
        datafile = loadnpzfile(datadir+fileheader+days[day]+npz)
        b = datafile[datatype]
        
        arr,nperms = PE_dist(b,embeddelay,delay=loop_delay)
        permstore_counter = permstore_counter+arr
        tot_perms = tot_perms+nperms
    PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
    C =  -2.*((PE_tot_Se - 0.5*PE_tot - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot/np.log2(nfac)))
    PEs[loop_delay]=PE_tot/np.log2(nfac)
    SCs[loop_delay]=C
    

savedir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\DSCOVR Data\\NPZ_files\\'
filename='PE_SC_DSCOVR_'+datatype+'_'+timelabel+'_embeddelay'+str(embeddelay)+'_over'+str(ndays)+'_days.npz'
np.savez(savedir+filename,PEs=PEs,SCs=SCs)