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
fileheader = 'nPen_20masses_LsEq1_MsEq1_g9p81_1000sec_tstep001_135degIC_0velIC'
shortheader = '20mass_135deg_0velIC'

npz='.npz'
datafile = loadnpzfile(datadir+fileheader+npz)
x=datafile['x']
y=datafile['y']
t=datafile['t']
n=x.shape[1]-1

embeddelay = 5
nfac = factorial(embeddelay)
delay_array = np.arange(1,1000)
num_delays = len(delay_array)

PExs = np.zeros([num_delays,n])
SCxs = np.zeros([num_delays,n])
PEys = np.zeros([num_delays,n])
SCys = np.zeros([num_delays,n])

#PEsy1 = np.zeros([num_delays])
#SCsy1 = np.zeros([num_delays])
#PEsy2 = np.zeros([num_delays])
#SCsy2 = np.zeros([num_delays])
#PEsy3 = np.zeros([num_delays])
#SCsy3 = np.zeros([num_delays])
#PEsy4 = np.zeros([num_delays])
#SCsy4 = np.zeros([num_delays])
#PEsy5 = np.zeros([num_delays])
#SCsy5 = np.zeros([num_delays])


for loop_delay in np.arange(len(delay_array)):
    #for loop_delay in np.arange(150,151):
    if (loop_delay%100)==0: print ('On Delay ',delay_array[loop_delay])
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    arr,nperms = PE_dist(x,embeddelay,delay=delay_array[loop_delay])
    permstore_counter = permstore_counter+arr
    tot_perms = tot_perms+nperms
    PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
    C =  -2.*((PE_tot_Se - 0.5*PE_tot - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot/np.log2(nfac)))
    PExs[loop_delay]=PE_tot/np.log2(nfac)
    SCxs[loop_delay]=C

#np.savez(datadir+filename,perm=permstore_counter)
filename='PE_SC_npend'+shortheader+'_embeddelay'+str(embeddelay)+'_'+str(num_delays)+'_delays.npz'
np.savez(datadir+filename,PExs=PExs,SCxs=SCxs,PEys=PEys,SCys=SCys,delays=delay_array)