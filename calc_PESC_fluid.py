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

datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Corrsin Wind Tunnel Data\\NPZ_files\\m20_5mm\\streamwise\\'
fileheader = 'm20_5mm_p2shot'
npz='.npz'

###Storage Arrays###
delta_t = 1.0/(40000.0)
delays = np.arange(2,250) #248 elements
taus = delays*delta_t
freq = 1.0/taus
PEs = np.zeros([248,121])
SCs = np.zeros([248,121])


for shot in np.arange(1,121):#(1,120):
    print '###### On Shot '+str(shot)+' #####'
    datafile = loadnpzfile(datadir+fileheader+str(shot)+npz)
    data = datafile['shot']
    for d in delays:
        print 'On Delay ', d
        PEs[d-delays[0],shot],SCs[d-delays[0],shot] = CH(data,5,delay=d)

filename='PE_SC_m20_5mm_embed5_p2.npz'
np.savez(datadir+filename,PEs=PEs,SCs=SCs,delta_t=delta_t,taus=taus,delays=delays,freq=freq)