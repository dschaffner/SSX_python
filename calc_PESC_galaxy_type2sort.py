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
#fileheader = 'New_DavidData_Class_2'
#fileheader = 'IDdatabase_Type_1_data' #3227 orbits
fileheader = 'IDdatabase_Type_2_data' #25387 orbits
#fileheader = 'IDdatabase_Type_31_data' #5770 orbits
#fileheader = 'IDdatabase_Type_32_data'#9798 orbits
#fileheader = 'IDdatabase_Type_4_data' #5818 orbits
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
print datafile.shape
num_orbits = datafile.shape[0]
#start_orbit = 0
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
PEs1 = np.zeros([num_delays+1])
SCs1 = np.zeros([num_delays+1])
PEs2 = np.zeros([num_delays+1])
SCs2 = np.zeros([num_delays+1])
PEs3 = np.zeros([num_delays+1])
SCs3 = np.zeros([num_delays+1])
PEs4 = np.zeros([num_delays+1])
SCs4 = np.zeros([num_delays+1])
PEs5 = np.zeros([num_delays+1])
SCs5 = np.zeros([num_delays+1])


delay = 1
embed_delay = 5
nfac = factorial(embed_delay)


for loop_delay in np.arange(1,num_delays+1):
    print 'On Delay ',loop_delay
    #Initial Radius - [0,3.5)
    permstore_counter1 = []
    permstore_counter1 = Counter(permstore_counter1)
    tot_perms1 = 0
    totshots1 = 0
    
    #Initial Radius - [3.5,4)
    permstore_counter2 = []
    permstore_counter2 = Counter(permstore_counter2)
    tot_perms2 = 0
    totshots2 = 0
    
    #Initial Radius - [4,4.5)
    permstore_counter3 = []
    permstore_counter3 = Counter(permstore_counter3)
    tot_perms3 = 0
    totshots3 = 0
    
    #Initial Radius - [4.5,5.5)
    permstore_counter4 = []
    permstore_counter4 = Counter(permstore_counter4)
    tot_perms4 = 0
    totshots4 = 0
    
    #Initial Radius - [7,9]
    permstore_counter5 = []
    permstore_counter5 = Counter(permstore_counter5)
    tot_perms5 = 0
    totshots5 = 0
    
    for shot in np.arange(num_orbits):#(1,120):
        if (shot%1000)==0: print 'On Orbit: ',shot
        #datafile = loadnpyfile(datadir+fileheader+npy)
        arr, nperms = PE_dist(datafile[shot,:],5,delay=loop_delay)
        
        if datafile[shot,0]>=0.0 and datafile[shot,0]<3.5:
            permstore_counter1 = permstore_counter1+arr
            tot_perms1 = tot_perms1+nperms
            totshots1+=1
    
        if datafile[shot,0]>=3.5 and datafile[shot,0]<4.0:
            permstore_counter2 = permstore_counter2+arr
            tot_perms2 = tot_perms2+nperms
            totshots2+=1
            
        if datafile[shot,0]>=4.0 and datafile[shot,0]<4.5:
            permstore_counter3 = permstore_counter3+arr
            tot_perms3 = tot_perms3+nperms
            totshots3+=1
            
        if datafile[shot,0]>=4.5 and datafile[shot,0]<5.5:
            permstore_counter4 = permstore_counter4+arr
            tot_perms4 = tot_perms4+nperms
            totshots4+=1
            
        if datafile[shot,0]>=7.0 and datafile[shot,0]<9.0:
            permstore_counter5 = permstore_counter5+arr
            tot_perms5 = tot_perms5+nperms
            totshots5+=1
            
    PE_tot1,PE_tot_Se1 = PE_calc_only(permstore_counter1,tot_perms1)
    PE_tot2,PE_tot_Se2 = PE_calc_only(permstore_counter2,tot_perms2)
    PE_tot3,PE_tot_Se3 = PE_calc_only(permstore_counter3,tot_perms3)
    PE_tot4,PE_tot_Se4 = PE_calc_only(permstore_counter4,tot_perms4)
    PE_tot5,PE_tot_Se5 = PE_calc_only(permstore_counter5,tot_perms5)
    
    C1 =  -2.*((PE_tot_Se1 - 0.5*PE_tot1 - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot1/np.log2(nfac)))
    PEs1[loop_delay]=PE_tot1/np.log2(nfac)
    SCs1[loop_delay]=C1
    
    C2 =  -2.*((PE_tot_Se2 - 0.5*PE_tot2 - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot2/np.log2(nfac)))
    PEs2[loop_delay]=PE_tot2/np.log2(nfac)
    SCs2[loop_delay]=C2
    
    C3 =  -2.*((PE_tot_Se3 - 0.5*PE_tot3 - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot3/np.log2(nfac)))
    PEs3[loop_delay]=PE_tot3/np.log2(nfac)
    SCs3[loop_delay]=C3
    
    C4 =  -2.*((PE_tot_Se4 - 0.5*PE_tot4 - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot4/np.log2(nfac)))
    PEs4[loop_delay]=PE_tot4/np.log2(nfac)
    SCs4[loop_delay]=C4
    
    C5 =  -2.*((PE_tot_Se5 - 0.5*PE_tot5 - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot5/np.log2(nfac)))
    PEs5[loop_delay]=PE_tot5/np.log2(nfac)
    SCs5[loop_delay]=C5
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

filename='PE_SC_'+fileheader+'_'+str(num_delays)+'_delays_'+str(num_orbits)+'_orbits_galpy0718_type2icsort.npz'
#filename='Data_0418_type4_'+str(num_delays)+'_delays.npz'
#filename='Data_twosins300and300divroot2_ranphasestart_'+str(num_delays)+'_delays.npz'
#filename='Data_sine500period_ranphasestart_1k_'+str(num_delays)+'_delays.npz'
np.savez(datadir+filename,PEs1=PEs1,SCs1=SCs1,
                          PEs2=PEs2,SCs2=SCs2,
                          PEs3=PEs3,SCs3=SCs3,
                          PEs4=PEs4,SCs4=SCs4,
                          PEs5=PEs5,SCs5=SCs5,
                          totshots1=totshots1,totshots2=totshots2,totshots3=totshots3,
                          totshots4=totshots4,totshots5=totshots5)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
