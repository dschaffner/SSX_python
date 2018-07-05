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
datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\DoublePendulum\\'
fileheader='DoubPen_L1-1_L2-1_m1-1_m2-1_9p8_scan_theta1_ic'
fileheader='DoubPen_L1-1_L2-1_m1-1_m2-1_9p8_chaos'
fileheader = 'DoubPen_LsMsEq1_9p8_ICQ4'
npz='.npz'
datafile = loadnpzfile(datadir+fileheader+npz)
#initial_theta1s=datafile['initial_theta1s']
#theta1s=datafile['theta1s']
#num_thetas = len(initial_theta1s)

x1=datafile['x1']
x2=datafile['x2']
x3=datafile['x3']
x4=datafile['x4']
y1=datafile['y1']
y2=datafile['y2']
y3=datafile['y3']
y4=datafile['y4']
embeddelay = 5
nfac = factorial(embeddelay)

###Storage Arrays###
#delta_t = 1.0
#delays = np.arange(1,101) #248 elements
#taus = delays*delta_t
#freq = 1.0/taus



delay_array = np.array([1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000,50000])
num_delays = len(delay_array)
"""
PEs = np.zeros([num_thetas,num_delays])
SCs = np.zeros([num_thetas,num_delays])

for theta in np.arange(num_thetas):
    print 'On Theta ',initial_theta1s[theta]
    for loop_delay in np.arange(len(delay_array)):
        print 'On Delay ',delay_array[loop_delay]
        permstore_counter = []
        permstore_counter = Counter(permstore_counter)
        tot_perms = 0
        arr,nperms = PE_dist(theta1s[theta,:],embeddelay,delay=delay_array[loop_delay])
        permstore_counter = permstore_counter+arr
        tot_perms = tot_perms+nperms
        PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
        C =  -2.*((PE_tot_Se - 0.5*PE_tot - 0.5*np.log2(nfac))
                    /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                    + np.log2(nfac))*(PE_tot/np.log2(nfac)))
        PEs[theta,loop_delay]=PE_tot/np.log2(nfac)
        SCs[theta,loop_delay]=C
"""


PEsx1 = np.zeros([num_delays])
SCsx1 = np.zeros([num_delays])
PEsx2 = np.zeros([num_delays])
SCsx2 = np.zeros([num_delays])
PEsx3 = np.zeros([num_delays])
SCsx3 = np.zeros([num_delays])
PEsx4 = np.zeros([num_delays])
SCsx4 = np.zeros([num_delays])

PEsy1 = np.zeros([num_delays])
SCsy1 = np.zeros([num_delays])
PEsy2 = np.zeros([num_delays])
SCsy2 = np.zeros([num_delays])
PEsy3 = np.zeros([num_delays])
SCsy3 = np.zeros([num_delays])
PEsy4 = np.zeros([num_delays])
SCsy4 = np.zeros([num_delays])

for loop_delay in np.arange(len(delay_array)):
    print 'On Delay ',delay_array[loop_delay]
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    arr,nperms = PE_dist(x1,embeddelay,delay=delay_array[loop_delay])
    permstore_counter = permstore_counter+arr
    tot_perms = tot_perms+nperms
    PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
    C =  -2.*((PE_tot_Se - 0.5*PE_tot - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot/np.log2(nfac)))
    PEsx1[loop_delay]=PE_tot/np.log2(nfac)
    SCsx1[loop_delay]=C
    
for loop_delay in np.arange(len(delay_array)):
    print 'On Delay ',delay_array[loop_delay]
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    arr,nperms = PE_dist(x2,embeddelay,delay=delay_array[loop_delay])
    permstore_counter = permstore_counter+arr
    tot_perms = tot_perms+nperms
    PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
    C =  -2.*((PE_tot_Se - 0.5*PE_tot - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot/np.log2(nfac)))
    PEsx2[loop_delay]=PE_tot/np.log2(nfac)
    SCsx2[loop_delay]=C
    
for loop_delay in np.arange(len(delay_array)):
    print 'On Delay ',delay_array[loop_delay]
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    arr,nperms = PE_dist(x3,embeddelay,delay=delay_array[loop_delay])
    permstore_counter = permstore_counter+arr
    tot_perms = tot_perms+nperms
    PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
    C =  -2.*((PE_tot_Se - 0.5*PE_tot - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot/np.log2(nfac)))
    PEsx3[loop_delay]=PE_tot/np.log2(nfac)
    SCsx3[loop_delay]=C
    
for loop_delay in np.arange(len(delay_array)):
    print 'On Delay ',delay_array[loop_delay]
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    arr,nperms = PE_dist(x4,embeddelay,delay=delay_array[loop_delay])
    permstore_counter = permstore_counter+arr
    tot_perms = tot_perms+nperms
    PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
    C =  -2.*((PE_tot_Se - 0.5*PE_tot - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot/np.log2(nfac)))
    PEsx4[loop_delay]=PE_tot/np.log2(nfac)
    SCsx4[loop_delay]=C

for loop_delay in np.arange(len(delay_array)):
    print 'On Delay ',delay_array[loop_delay]
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    arr,nperms = PE_dist(y1,embeddelay,delay=delay_array[loop_delay])
    permstore_counter = permstore_counter+arr
    tot_perms = tot_perms+nperms
    PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
    C =  -2.*((PE_tot_Se - 0.5*PE_tot - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot/np.log2(nfac)))
    PEsy1[loop_delay]=PE_tot/np.log2(nfac)
    SCsy1[loop_delay]=C
    
for loop_delay in np.arange(len(delay_array)):
    print 'On Delay ',delay_array[loop_delay]
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    arr,nperms = PE_dist(y2,embeddelay,delay=delay_array[loop_delay])
    permstore_counter = permstore_counter+arr
    tot_perms = tot_perms+nperms
    PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
    C =  -2.*((PE_tot_Se - 0.5*PE_tot - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot/np.log2(nfac)))
    PEsy2[loop_delay]=PE_tot/np.log2(nfac)
    SCsy2[loop_delay]=C
    
for loop_delay in np.arange(len(delay_array)):
    print 'On Delay ',delay_array[loop_delay]
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    arr,nperms = PE_dist(y3,embeddelay,delay=delay_array[loop_delay])
    permstore_counter = permstore_counter+arr
    tot_perms = tot_perms+nperms
    PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
    C =  -2.*((PE_tot_Se - 0.5*PE_tot - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot/np.log2(nfac)))
    PEsy3[loop_delay]=PE_tot/np.log2(nfac)
    SCsy3[loop_delay]=C
    
for loop_delay in np.arange(len(delay_array)):
    print 'On Delay ',delay_array[loop_delay]
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    arr,nperms = PE_dist(y4,embeddelay,delay=delay_array[loop_delay])
    permstore_counter = permstore_counter+arr
    tot_perms = tot_perms+nperms
    PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
    C =  -2.*((PE_tot_Se - 0.5*PE_tot - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot/np.log2(nfac)))
    PEsy4[loop_delay]=PE_tot/np.log2(nfac)
    SCsy4[loop_delay]=C    

savedir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\DSCOVR Data\\NPZ_files\\'
filename='PE_SC_DP'+fileheader+'_embeddelay'+str(embeddelay)+'_'+str(num_delays)+'_delays.npz'
np.savez(datadir+filename,PEsx1=PEsx1,SCsx1=SCsx1,
                          PEsx2=PEsx2,SCsx2=SCsx2,
                          PEsx3=PEsx3,SCsx3=SCsx3, 
                          PEsx4=PEsx4,SCsx4=SCsx4, 
                          PEsy1=PEsy1,SCsy1=SCsy1, 
                          PEsy2=PEsy2,SCsy2=SCsy2, 
                          PEsy3=PEsy3,SCsy3=SCsy3, 
                          PEsy4=PEsy4,SCsy4=SCsy4, 
                          delays=delay_array)
#"""