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
import compute_wavelet as cw
import spectrum_wwind as sww
#calc_PESC_fluid.py

datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\'
#fileheader = 'New_DavidData_Class_3'
fileheader = 'DavidData_Class_3'
npy='.npy'

datafile = loadnpyfile(datadir+fileheader+npy)
num_orbits = int(datafile.shape[0])

#fullwavelet,summedwavlet,wvfreq,FFTcomp,fftfreq = cw.compute_wavelet(datafile[0,1:],time*1e6,Bfield=False)
#freq,freq2,comp,pwr,mag,phase,cos_phase,dt = sww.spectrum_wwind(datafile[0,1:],time,window='hanning')


x=np.arange(10000)
y=np.sin(0.003*x)

wv_n = 1792#2816
fq_n = 501#5001
time = np.arange(1000)
gal_wav = np.zeros(wv_n)
gal_fft = np.zeros(fq_n)

for shot in np.arange(num_orbits):
    print 'On Shot ',shot
    fullwavelet,summedwavlet,wvfreq,FFTcomp,fftfreq = cw.compute_wavelet(datafile[shot,1:],time*1e6,Bfield=False)
    freq,freq2,comp,pwr,mag,phase,cos_phase,dt = sww.spectrum_wwind(datafile[shot,1:],time,window='hanning')
    gal_wav = gal_wav+summedwavlet
    gal_fft = gal_fft+pwr
#filename='Spectra_'+fileheader+'_longrec.npz'
filename='Spectra_'+fileheader+'_shortrec.npz'
np.savez(datadir+filename,fftfreq=freq,wvfreq=wvfreq,gal_wav=gal_wav,gal_fft=gal_fft)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)



ylogarr = np.log(gal_fft[2:8])#[50:500])
xarr = freq[2:8]#[50:500]
A1 = np.array([xarr,np.ones(len(xarr))])
w1 = np.linalg.lstsq(A1.T,ylogarr)
slope = w1[0][0]

plt.plot(xarr,ylogarr)
plt.plot(xarr,w1[0][1]+(xarr*slope))
