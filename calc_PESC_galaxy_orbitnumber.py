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

datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\GalpyData_July2018\\'
datadir = 'C:\\Users\\dschaffner\\Dropbox\\PESC_Chaos\\Sorted_CR8\\'
#fileheader = 'New_DavidData_Class_2'
#fileheader = 'IDdatabase_Type_1_data_4000' #3227 orbits
#fileheader = 'IDdatabase_Type_2_data' #25387 orbits
#fileheader = 'IDdatabase_Type_31_data' #5770 orbits
#fileheader = 'IDdatabase_Type_32_data'#9798 orbits
fileheader = 'IDdatabase_Type_4_data_4000' #5818 orbits
fileheader = 'Type_1_Rg'#2797
fileheader = 'Type_2_Rg'#14671
fileheader = 'Type_31_Rg'#1675
fileheader = 'Type_32_Rg'#3585
fileheader = 'Type_4_Rg'#2271
#fileheader= 'Sine_700period_randomphase'
#fileheader = 'Sine_300period_androot2lessperiod_randomphase'
#fileheader = 'Sine_2000period_wp5timesroot2-180percent_randomphase'
#fileheader = 'Sine_1500period_wtimesroot2-200percent_randomphase_10k'
#fileheader = 'qp_(m=4)_(th=30.0)_(t=1.0)_(CR=6.0)_(eps=0.4)_(x0=2.350639412)_(y0=6.62220828293)_(vx0=-243.996156434)_(vy0=40.276745914)_prop'
#fileheader = 'Sine_500period_randomphase_shortrec'
npy='.npy'

#atafile = loadnpyfile(datadir+fileheader+npy)

#import glob
#data=glob.glob(datadir+'*data.npy')
#prop=glob.glob(datadir+'*prop.npy')

datafile = loadnpyfile(datadir+fileheader+npy)

#3x = np.arange(100000)*0.01
#datafile = np.zeros([1,100000])
#datafile[0,:] = np.sin(0.002*x)
#fidcut = datafile[:,78534:78547]
#datafile = np.zeros([1,fidcut.shape[1]])
#datafile = fidcut

#datafile[0,:] = -2*x

print(datafile.shape)
num_orbits = datafile.shape[0]
#num_orbits = 1
start_orbit = 0
#
#propfile = loadnpyfile(prop[1000])
#print propfile.shape

#num_orbits = int(len(data))
"""
###Storage Arrays###
#delta_t = 1.0/(40000.0)
#delays = np.arange(2,250) #248 elements
#taus = delays*delta_t
#freq = 1.0/taus
num_delays = 499
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
    for shot in np.arange(start_orbit,start_orbit+num_orbits):#(1,120):
        if (shot%1000)==0: print 'On Orbit: ',shot
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

filename='PE_SC_'+fileheader+'_'+str(num_delays)+'_delays_'+str(num_orbits)+'_orbits_galpy0718_range3.npz'
#filename='Data_0418_type4_'+str(num_delays)+'_delays.npz'
#filename='Data_twosins300and300divroot2_ranphasestart_'+str(num_delays)+'_delays.npz'
#filename='Data_sine500period_ranphasestart_1k_'+str(num_delays)+'_delays.npz'
#np.savez(datadir+filename,PEs=PEs,SCs=SCs)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
"""