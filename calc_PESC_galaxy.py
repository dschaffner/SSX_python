# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:00:45 2017

@author: dschaffner
"""
import numpy as np
import matplotlib.pylab as plt
from loadnpyfile import loadnpyfile
from calc_PE_SC import PE, CH, PE_dist, PE_calc_only
#import Cmaxmin as cpl
from collections import Counter
from math import factorial
import time

#calc_PESC_fluid.py

datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\GalpyData_July2018\\'

datadir = 'C:\\Users\\dschaffner\\Dropbox\\PESC_Chaos\\Sorted_CR6\\'

#fileheader = 'IDdatabase_Type_1_7co_data_3000' #3227 orbits
#fileheader = 'IDdatabase_Type_2_7co_data_3000' #25387 orbits
#fileheader = 'IDdatabase_Type_31_7co_data_3000' #5770 orbits
#fileheader = 'IDdatabase_Type_32_7co_data_3000'#9798 orbits
#fileheader = 'IDdatabase_Type_4_7co_data_3000' #5818 orbits

#length=1400
#fileheader = 'Type_1_4CR_3000_Rg'#2797
#fileheader = 'Type_2_4CR_3000_Rg'#14671
#fileheader = 'Type_31_4CR_3000_Rg'#1675
#fileheader = 'Type_32_4CR_3000_Rg'#7376
#fileheader = 'Type_4_4CR_3000_Rg'#2271

#length=2000
#fileheader = 'Type_1_6CR_3000_Rg'#2718
#fileheader = 'Type_2_6CR_3000_Rg'#30134
fileheader = 'Type_31_6CR_3000_Rg'#4513
#fileheader = 'Type_32_6CR_3000_Rg'#8834
#fileheader = 'Type_4_6CR_3000_Rg'#3801

#length=2400
#fileheader = 'Type_1_7CR_3000_Rg'#2797
#fileheader = 'Type_2_7CR_3000_Rg'#14671
#fileheader = 'Type_31_7CR_3000_Rg'#1675
#fileheader = 'Type_32_7CR_3000_Rg'#6474
#fileheader = 'Type_4_7CR_3000_Rg'#2271

#Lenght=3500
#fileheader = 'Type_1_8CR_4000_Rg'#2797
#fileheader = 'Type_2_8CR_4000_Rg'#14671
#fileheader = 'Type_31_8CR_4000_Rg'#1675
#fileheader = 'Type_32_8CR_4000_Rg'#3585
#fileheader = 'Type_4_8CR_4000_Rg'#2271
#
#fileheader = 'Type_1_10CR_3000_Rg'#2797
#fileheader = 'Type_2_10CR_3000_Rg'#14671
#fileheader = 'Type_31_10CR_3000_Rg'#1675
#fileheader = 'Type_32_10CR_3000_Rg'#3585
#fileheader = 'Type_4_10CR_3000_Rg'#2271



npy='.npy'




datafile = loadnpyfile(datadir+fileheader+npy)
"""
import glob
data=glob.glob(datadir+'*data.npy')
prop=glob.glob(datadir+'*prop.npy')

datafile = loadnpyfile(data[1000])
print datafile.shape

propfile = loadnpyfile(prop[1000])
print propfile.shape
"""
num_orbits = int(datafile.shape[0])
print( num_orbits )
#record length
length=2000#int(datafile.shape[1])
print(length)
#num_orbits = 3000

###Storage Arrays###
#delta_t = 1.0/(40000.0)
#delays = np.arange(2,250) #248 elements
#taus = delays*delta_t
#freq = 1.0/taus
num_delays = 499#749#249
PEs = np.zeros([num_delays+1])
SCs = np.zeros([num_delays+1])

delay = 1
embed_delay = 5
nfac = factorial(embed_delay)

start_time = time.time()
for loop_delay in np.arange(1,num_delays+1):
    print('On Delay ',loop_delay)
    print("--- %s minutes ---" % np.round((time.time() - start_time)/60.0,4))
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    num_orbits_computed = 0
    num_orbits_skipped = 0
    for shot in np.arange(num_orbits):#(1,120):
        if (shot%1000)==0: print ('On Orbit: ',shot)
        if np.min(datafile[shot,1:length])<0.1: 
            num_orbits_skipped+=1
            continue
        
        arr, nperms = PE_dist(datafile[shot,1:length],5,delay=loop_delay)
        permstore_counter = permstore_counter+arr
        tot_perms = tot_perms+nperms
        num_orbits_computed+=1
        if num_orbits_computed == num_orbits: break
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

filename='PE_SC_'+fileheader+'_'+str(num_delays)+'_delays_'+str(num_orbits_computed)+'orbits_of'+str(num_orbits)+'_total'+str(length)+'_timesteps_resorted.npz'
np.savez(datadir+filename,PEs=PEs,SCs=SCs,num_orbits=num_orbits_computed,num_skipped=num_orbits_skipped)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
