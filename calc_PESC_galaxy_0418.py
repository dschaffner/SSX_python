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

#calc_PESC_fluid.py

datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\Data040318\\'
#fileheader = 'New_DavidData_Class_2'
#fileheader = 'IDdatabase_Type_1_data'
#fileheader = 'IDdatabase_Type_2_data'
#fileheader = 'IDdatabase_Type_31_data'
#fileheader = 'IDdatabase_Type_32_data'
#fileheader = 'IDdatabase_Type_4_data'
#fileheader= 'Sine_700period_randomphase'
#fileheader = 'Sine_300period_androot2lessperiod_randomphase'
fileheader = 'Sine_2000period_wp5timesroot2-180percent_randomphase'
fileheader = 'Sine_1500period_wtimesroot2-200percent_randomphase_10k'
#fileheader = 'qp_(m=4)_(th=30.0)_(t=1.0)_(CR=6.0)_(eps=0.4)_(x0=2.350639412)_(y0=6.62220828293)_(vx0=-243.996156434)_(vy0=40.276745914)_prop'
fileheader = 'Sine_500period_randomphase_shortrec'
npy='.npy'

#atafile = loadnpyfile(datadir+fileheader+npy)

#import glob
#data=glob.glob(datadir+'*data.npy')
#prop=glob.glob(datadir+'*prop.npy')

datafile = loadnpyfile(datadir+fileheader+npy)
print datafile.shape
num_orbits = datafile.shape[0]

#
#propfile = loadnpyfile(prop[1000])
#print propfile.shape

#num_orbits = int(len(data))

###Storage Arrays###
#delta_t = 1.0/(40000.0)
#delays = np.arange(2,250) #248 elements
#taus = delays*delta_t
#freq = 1.0/taus
num_delays = 249
PEs = np.zeros([num_delays+1])
SCs = np.zeros([num_delays+1])

delay = 1
embed_delay = 5
nfac = factorial(embed_delay)

for loop_delay in np.arange(1,num_delays+1):
    print 'On Delay ',loop_delay
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    for shot in np.arange(num_orbits):#(1,120):
        #datafile = loadnpyfile(datadir+fileheader+npy)
        arr, nperms = PE_dist(datafile[shot,:],5,delay=loop_delay)
        permstore_counter = permstore_counter+arr
        tot_perms = tot_perms+nperms
    PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
    C =  -2.*((PE_tot_Se - 0.5*PE_tot - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot/np.log2(nfac)))
    PEs[loop_delay]=PE_tot/np.log2(nfac)
    SCs[loop_delay]=C
#PE_arr1 = PE_dist(datafile[0,1:],5,delay=10)
#PE_arr2 = PE_dist(datafile[20,1:],5,delay=10)
#PE_arr3 = PE_dist(datafile[40,1:],5,delay=10)

#perms = 960
#tot_permutations=perms*3.0
#PE1 = PE_calc_only(PE_arr1,perms)/np.log2(120)
#PE2 = PE_calc_only(PE_arr2,perms)/np.log2(120)
#PE3 = PE_calc_only(PE_arr3,perms)/np.log2(120)
#PE_tot = PE_calc_only(PE_arr1+PE_arr2+PE_arr3,tot_permutations)/np.log2(120)

#for shot in np.arange(2):#(num_orbits):#(1,120):
#    print '###### On Shot '+str(shot)+' #####'
#    PE_arr1 = PE_dist(datafile[shot,1:],5,delay=1)
    #PEs[shot],SCs[shot]=CH(datafile[shot,1:],5,delay=1)

#filename='PE_SC_'+fileheader+'_'+str(num_delays)+'_delays_longrec.npz'
#filename='Data_0418_type4_'+str(num_delays)+'_delays.npz'
#filename='Data_twosins300and300divroot2_ranphasestart_'+str(num_delays)+'_delays.npz'
filename='Data_sine500period_ranphasestart_1k_'+str(num_delays)+'_delays.npz'
np.savez(datadir+filename,PEs=PEs,SCs=SCs)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
