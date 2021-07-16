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
from Lorenz_model_test import lorenz, lorenz_mod
from calcPESCcurves import calcPESCcurves


#calc_PESC_solarwind_chen.py
datadir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Lorenz\\'

s1=10
r1=28
b1=2.667

dt = 0.01
num_steps = 10000

# Need one more for the initial values
xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)

# Set initial values
xs[0], ys[0], zs[0] = (0., 1., 1.05)

# Step through "time", calculating the partial derivatives at the current point
# and using them to estimate the next point
for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz_mod(xs[i], ys[i], zs[i])
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)
    
    
    #delay_array = np.array([1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000,50000])
delay_array = np.arange(1,1000)
num_delays = len(delay_array)
embeddelay=5
nfac = factorial(embeddelay)
    
PEsx = np.zeros([num_delays])
SCsx = np.zeros([num_delays])
PEsy = np.zeros([num_delays])
SCsy = np.zeros([num_delays])
PEsz = np.zeros([num_delays])
SCsz = np.zeros([num_delays])
    
for loop_delay in np.arange(len(delay_array)):
    #for loop_delay in np.arange(150,151):
    if (loop_delay%100)==0: print( 'On Delay ',delay_array[loop_delay])
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    arr,nperms = PE_dist(xs,embeddelay,delay=delay_array[loop_delay])
    permstore_counter = permstore_counter+arr
    tot_perms = tot_perms+nperms
    PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
    C =  -2.*((PE_tot_Se - 0.5*PE_tot - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot/np.log2(nfac)))
    PEsx[loop_delay]=PE_tot/np.log2(nfac)
    SCsx[loop_delay]=C
print ('x completed')

PEsx2,SCsx2=calcPESCcurves(xs,n=embeddelay,max_delay=1000)


for loop_delay in np.arange(len(delay_array)):
    #for loop_delay in np.arange(150,151):
    if (loop_delay%100)==0: print( 'On Delay ',delay_array[loop_delay])
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    arr,nperms = PE_dist(ys,embeddelay,delay=delay_array[loop_delay])
    permstore_counter = permstore_counter+arr
    tot_perms = tot_perms+nperms
    PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
    C =  -2.*((PE_tot_Se - 0.5*PE_tot - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot/np.log2(nfac)))
    PEsy[loop_delay]=PE_tot/np.log2(nfac)
    SCsy[loop_delay]=C
print ('y completed')

for loop_delay in np.arange(len(delay_array)):
    #for loop_delay in np.arange(150,151):
    if (loop_delay%100)==0: print( 'On Delay ',delay_array[loop_delay])
    permstore_counter = []
    permstore_counter = Counter(permstore_counter)
    tot_perms = 0
    arr,nperms = PE_dist(zs,embeddelay,delay=delay_array[loop_delay])
    permstore_counter = permstore_counter+arr
    tot_perms = tot_perms+nperms
    PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
    C =  -2.*((PE_tot_Se - 0.5*PE_tot - 0.5*np.log2(nfac))
                /((1 + 1./nfac)*np.log2(nfac+1) - 2*np.log2(2*nfac) 
                + np.log2(nfac))*(PE_tot/np.log2(nfac)))
    PEsz[loop_delay]=PE_tot/np.log2(nfac)
    SCsz[loop_delay]=C
print ('y completed')

#filename='PE_SC_DP'+fileheader+'_embeddelay'+str(embeddelay)+'_'+str(num_delays)+'_delays.npz'
#np.savez(datadir+filename,PEsx1=PEsx1,SCsx1=SCsx1,
                              #PEsx2=PEsx2,SCsx2=SCsx2,
                              #PEsx3=PEsx3,SCsx3=SCsx3, 
                              #PEsx4=PEsx4,SCsx4=SCsx4, 
                              #PEsy1=PEsy1,SCsy1=SCsy1, 
                              #PEsy2=PEsy2,SCsy2=SCsy2, 
                              #PEsy3=PEsy3,SCsy3=SCsy3, 
                              #PEsy4=PEsy4,SCsy4=SCsy4, 
#                              delays=delay_array)
