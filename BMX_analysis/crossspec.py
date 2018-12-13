#crossspec.py

import numpy as np
import spectrum_wwind as spec

"""
Crosspec:
Function for computing the cross spectrum between two time series
Crossphase:
Function for producing the phase angle between each Fourier Mode between the two arrays
crosspec_corr:
function that uses the crosspectrum to compute the cross-correlation value spectrally over 
a range of frequencies. For the full frequency range, this value should be the same as computing
<x(0)y(0)> for x(t) and y(t) where <> means the average the elements of the array.
crossspec_corr_norm:
a normalized version of crossspec_corr. The result of crossspec_corr is divided by the
square root of the product or the auto-spectrum correlations (i.e. <x(0)x(0)> and <y(0)y(0)>)
"""

def tindex_min(timearr,timevalue):
    minval = np.min(np.abs((timearr)-timevalue))
    tind = np.where(np.abs((timearr)-(timevalue)) == minval)
    tind = tind[0][0]
    return tind

def crossphase(arr1,arr2,time):
    f1,f0,comp1,pwr1,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(arr1,time,window='None')
    f1,f0,comp2,pwr2,mag2,phase2,cos_phase2,interval=spec.spectrum_wwind(arr2,time,window='None')
    n=len(arr1)
    factor = 2.0/(n*interval)
    cross_spec=np.conj(comp1)*comp2*factor
    crossphase = np.angle(cross_spec[0:n/2+1])
    return f1,crossphase
    
def crossspec(arr1,arr2,time):
    f1,f0,comp1,pwr1,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(arr1,time,window='None')
    f1,f0,comp2,pwr2,mag2,phase2,cos_phase2,interval=spec.spectrum_wwind(arr2,time,window='None')
    n=len(arr1)
    factor = 2.0/(n*interval)
    cross_spec=np.conj(comp1)*comp2*factor
    return f1,cross_spec
    
def crossspec_corr(arr1,arr2,time,freq1,freq2):#input frequencies in Hz
    f1,f0,comp1,pwr1,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(arr1,time,window='None')
    f1,f0,comp2,pwr2,mag2,phase2,cos_phase2,interval=spec.spectrum_wwind(arr2,time,window='None')
    n=len(arr1)
    factor = 2.0/(n*interval)
    cross_spec=np.conj(comp1)*comp2*factor
    crossphase = np.angle(cross_spec[0:n/2+1])
    cross_coh = cross_coh = ((np.abs(cross_spec[0:n/2+1]))**2)/(factor*factor*pwr1*pwr2)
    
    findex1 = tindex_min(f1,freq1)
    findex2 = tindex_min(f1,freq2)
    
    spec_corrtot=0.0
    for ff in np.arange(findex1,findex2):
        spec_corrtot = spec_corrtot + (np.sqrt(cross_coh[ff])*
                                       np.cos(crossphase[ff])*
                                       np.sqrt(factor*pwr1[ff])*
                                       np.sqrt(factor*pwr2[ff])*
                                       f1[1])
    
    return spec_corrtot
    
def crossspec_corr_norm(arr1,arr2,time,freq1,freq2):#input frequencies in Hz
    spec_corrtot = crossspec_corr(arr1,arr2,time,freq1,freq2)
    auto_corrtot1 = crossspec_corr(arr1,arr1,time,freq1,freq2)
    auto_corrtot2 = crossspec_corr(arr2,arr2,time,freq1,freq2)
    
    norm_corr = spec_corrtot/(np.sqrt(auto_corrtot1*auto_corrtot2))
    return norm_corr