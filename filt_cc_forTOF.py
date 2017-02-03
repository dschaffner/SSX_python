#cross correlation test

import get_corr as gc
import numpy as np
import scipy.signal as sig
import matplotlib.pylab as plt
import process_mjmag_data as mj
import ssxuserfunctions as ssxuf

day='123016'
shot=50

sep_dist = 7e-5#in km (7cm)between each loop
    
### LOAD SHOT ####
print 'Loading Shot ',shot
time,bdot,timeb,b,bmod,fulldata = mj.process_mjmag_data(day+str(shot))
tindex1=ssxuf.tindex_min(time,20.0)
tindex2=ssxuf.tindex_min(time,40.0)
time = time[tindex1:tindex2]
b = b[:,:,tindex1:tindex2]
bx25 = b[0,24,:]
bx24 = b[0,23,:]
bx23 = b[0,22,:]
bx22 = b[0,21,:]
bx21 = b[0,20,:]


#bdoty = bdot[1,:,:]
#bdotz = bdot[2,:,:]

### NORMALIZE SIGNAL to MAX ###

### GENERTATE FILTERS ###
#N  = 2    # Filter order
#Wn = 0.009 # Cutoff frequency
#B, A = sig.butter(N, Wn,btype='highpass', output='ba')    
low_cutoff = 1.0 #in MHZ
high_cutoff = 6.0 #in MHZ
b_high, a_high = sig.butter(4,high_cutoff/50.0, 'high')#, analog=True)  #HIGH PASS Filter
b_low, a_low = sig.butter(4,low_cutoff/50.0, 'low')#, analog=True)      #LOW PASS Filter 
b_band, a_band = sig.butter(6,[low_cutoff/50.0,high_cutoff/50.0], 'band')#, analog=True) #BAND PASS Filter    

### APPLY FILTERS ###

#BX#
bx25_hp = sig.filtfilt(b_high,a_high, bx25)
bx25_lp = sig.filtfilt(b_low,a_low, bx25)


bx24_hp = sig.filtfilt(b_high,a_high, bx24)
bx24_lp = sig.filtfilt(b_low,a_low, bx24)

bx23_hp = sig.filtfilt(b_high,a_high, bx23)
bx23_lp = sig.filtfilt(b_low,a_low, bx23)

bx22_hp = sig.filtfilt(b_high,a_high, bx22)
bx22_lp = sig.filtfilt(b_low,a_low, bx22)

bx21_hp = sig.filtfilt(b_high,a_high, bx21)
bx21_lp = sig.filtfilt(b_low,a_low, bx21)

### COMPUTE correlation function ###
#bdotx25_roll = np.roll(bdotx25_lp,-100)

#BX#
tau0,corr_bx_unfilt = gc.get_corr(time,bx25,bx24,normalized=True) #unfiltered correlation
tau2,corr_bx_lp=gc.get_corr(time,bx25_lp,bx24_lp,normalized=True) #low passed correlation
#Note on interpreting lead and lag--> If array1 leads array2 (that is, features in array1 appear before that of array2),
#then, this will result in a NEGATIVE tau in the cross-correlation function


### PLOT SHOT DATA ###

plt.figure(1)
plt.clf()
#plt.plot(time,bx25)
plt.plot(time,bx25_lp,label='Pos 25')
#plt.plot(time,-bx24)
plt.plot(time,bx24_lp,label='Pos 24')
#plt.plot(time,bx23_lp,label='Pos 23')
#plt.plot(time,bx22_lp,label='Pos 22')
#plt.plot(time,bx21_lp,label='Pos 21')
plt.legend()

### PLOT Correlation Function ###
plt.figure(5)
plt.clf()
plt.plot(tau0,corr_bx_unfilt)
plt.plot(tau2,corr_bx_lp)
plt.title('Correlation Functions')