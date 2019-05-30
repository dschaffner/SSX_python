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
import get_corr as gc
#import compute_wavelet as cw
#import spectrum_wwind as sww
#calc_PESC_fluid.py
from scipy.signal import find_peaks
import os

datadir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Galatic Dynamics Data\\GalpyData_July2018\\resorted_data\\'
#fileheader = 'New_DavidData_Class_3'
#fileheader = 'M4_2234_et_timescan\\Type_1_Rg_Full'
#fileheader = 'M5_2792_et_timescan\\Type_1_Rg_Full'
#fileheader = 'M6_3352_et_timescan\\Type_1_Rg_Full'
#fileheader = 'M7_3910_et_timescan\\Type_1_Rg_Full'
#fileheader = 'M8_4468_et_timescan\\Type_1_Rg_Full'
#fileheader = 'M9_5026_et_timescan\\Type_1_Rg_Full'
fileheader = 'M10_5586_et_timescan\\Type_1_Rg_Full'

#fileheader = 'Type_32i_Rg'
npy='.npy'

#CR4dyn=1117
#CR5dyn=1396
#CR6dyn=1676
#CR7dyn=1955
#CR8dyn=2234
#CR9dyn=2513
#CR10dyn=2793

galaxyname=r'M10'
dyntime=2793

datafile = loadnpyfile(datadir+fileheader+npy)
num_orbits = int(datafile.shape[0])
timelength = int(datafile.shape[1])
print('Num orbits=',num_orbits)
print('Time length=',timelength)


peak_seps=[]
numpeaks=np.zeros([num_orbits])
for orbit in np.arange(num_orbits):
    print('On Orbit ',orbit)
    tau,corr = gc.get_corr(np.arange(timelength),datafile[orbit,:],datafile[orbit,:],normalized=False)
    #plt.plot(tau,corr)
    #plt.figure(2)
    #plt.plot(datafile[1,:])
    #tau_deriv=np.gradient(corr)
    #plt.figure(3)
    #plt.plot(tau,tau_deriv)
    peaks=find_peaks(np.abs(corr))
    numpeaks[orbit]=len(peaks[0])
    if len(peaks[0])==3:
        peak_seps.append(2*(peaks[0][2]-peaks[0][1]))
    if len(peaks[0])==5:
        peak_seps.append(2*(peaks[0][3]-peaks[0][2]))
    if len(peaks[0])==7:
        peak_seps.append(2*(peaks[0][4]-peaks[0][3]))
peak_sep_arr=np.array(peak_seps)
peak_sep_arr_norm=peak_sep_arr/dyntime
binarr=np.arange(0,4.2,0.2)
mean_sep = np.mean(peak_sep_arr_norm)

plt.rc('axes',linewidth=2.0)
plt.rc('xtick.major',width=2.0)
plt.rc('ytick.major',width=2.0)
plt.rc('xtick.minor',width=2.0)
plt.rc('ytick.minor',width=2.0)
plt.rc('lines',markersize=8,markeredgewidth=0.0,linewidth=2.0)

#plt.rcParams['ps.fonttype'] = 42
#plt.rcParams['pdf.fonttype'] = 42

fig=plt.figure(num=1,figsize=(9,7),dpi=600,facecolor='w',edgecolor='k')
left  = 0.10  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.0   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)



ax1=plt.subplot(1,1,1)
hist=plt.hist(peak_sep_arr_norm,bins=binarr,weights=np.ones(len(peak_sep_arr_norm)) / len(peak_sep_arr_norm),density=False)#note weights and density arguement make total of y values 1, regardless of bin values
plt.xticks(binarr,fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0,0.25)
plt.xlabel(r'Period/$\tau_{dyn}^{'+galaxyname+'}$',fontsize=15)
plt.ylabel('Fraction of Orbits',fontsize=15)
plt.title(galaxyname+' - '+str(len(peak_seps))+' counted of '+str(num_orbits)+' total orbits')
plt.vlines(mean_sep,0,1,color='red',label='Mean='+str(np.round(mean_sep,2)))
leg=plt.legend(loc='upper left',fontsize=12,frameon=False)

savefilename=galaxyname+'_Type1_radialoscillationperiod_histogram_abscorr_halfperonly.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()
#fullwavelet,summedwavlet,wvfreq,FFTcomp,fftfreq = cw.compute_wavelet(datafile[0,1:],time*1e6,Bfield=False)
#freq,freq2,comp,pwr,mag,phase,cos_phase,dt = sww.spectrum_wwind(datafile[0,1:],time,window='hanning')

"""
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
"""