# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:00:45 2017

@author: dschaffner
"""
import numpy as np
import matplotlib.pylab as plt
from loadnpzfile import loadnpzfile
from calc_PE_SC import PE, CH

#calc_PESC_DSCOVR.py
day='052817'
datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\DSCOVR Data\\NPZ_files\\'
fileheader='mag_gse_1sec_'
npz='.npz'

###Storage Arrays###
delta_t = 1.0
delays = np.arange(1,101) #248 elements
taus = delays*delta_t
freq = 1.0/taus
bt_PEs = np.zeros([100,1])
bt_SCs = np.zeros([100,1])
bx_PEs = np.zeros([100,1])
bx_SCs = np.zeros([100,1])
by_PEs = np.zeros([100,1])
by_SCs = np.zeros([100,1])
bz_PEs = np.zeros([100,1])
bz_SCs = np.zeros([100,1])

for loop_delay in np.arange(1,num_delays+1):
    if np.mod(loop_delay,10)==0: print 'On Delay',loop_delay
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    for day in np.arange(0,1):#(1,120):
        print '###### On Day '+day+' #####'
        datafile = loadnpzfile(datadir+fileheader+day+npz)
    bt = datafile['bt']
    bx = datafile['bx_gse']
    by = datafile['by_gse']
    bz = datafile['bz_gse']
    for d in delays:
        print 'On Delay ', d
        bt_PEs[d-delays[0],shot],bt_SCs[d-delays[0],shot] = CH(bt,6,delay=d)
        bx_PEs[d-delays[0],shot],bx_SCs[d-delays[0],shot] = CH(bx,6,delay=d)
        by_PEs[d-delays[0],shot],by_SCs[d-delays[0],shot] = CH(by,6,delay=d)
        bz_PEs[d-delays[0],shot],bz_SCs[d-delays[0],shot] = CH(bz,6,delay=d)

savedir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\DSCOVR Data\\NPZ_files\\'
filename='PE_SC_DSCOVR_bxbybzbt_1s_embed6_'+day+'.npz'
np.savez(savedir+filename,
         bt_PEs=bt_PEs,bt_SCs=bt_SCs,
         bx_PEs=bx_PEs,bx_SCs=bx_SCs,
         by_PEs=by_PEs,by_SCs=by_SCs,
         bz_PEs=bz_PEs,bz_SCs=bz_SCs,delta_t=delta_t,taus=taus,delays=delays,freq=freq)