# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:00:45 2017

@author: dschaffner
"""
import numpy as np
import matplotlib.pylab as plt
import scipy as sp
from loadnpzfile import loadnpzfile
from calc_PE_SC import PE, CH, PE_dist, PE_calc_only
from collections import Counter
from math import factorial

#calc_PESC_solarwind_chen.py
datadir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Galatic Dynamics Data\\Examples\\'

embeddelay = 5
nfac = factorial(embeddelay)

#NOISE
noise_array=np.random.uniform(-1,1,size=100000)
delay_array = np.arange(1,1000)
num_delays = len(delay_array)+1

"""
PEs_noise100k = np.zeros(num_delays)
SCs_noise100k = np.zeros(num_delays)
for loop_delay in delay_array:
    print ('On Noise Delay ', loop_delay)
    PEs_noise100k[loop_delay],SCs_noise100k[loop_delay] = CH(noise_array,5,delay=loop_delay)
filename='PESC_noise100k_embed5_'+str(num_delays)+'_delays.npz'
np.savez(datadir+filename,timeseries=noise_array,taus=delay_array,PEs=PEs_noise100k,SCs=SCs_noise100k)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)


#HENON
def henonsave(N):
    X = np.zeros((2,N))
    X[0,0] = 1.
    X[1,0] = 1.
    a = 1.4
    b = 0.3
    for i in range(1,N):
        X[0,i] = 1. - a * X[0,i-1] ** 2. + X[1,i-1]
        X[1,i] = b * X[0,i-1]
    return X

henon_array=henonsave(100000)
henon_array=henon_array[0,:]
PEs_henon100k = np.zeros([num_delays])
SCs_henon100k = np.zeros([num_delays])
for loop_delay in delay_array:
    print ('On Henon Delay ', loop_delay)
    PEs_henon100k[loop_delay],SCs_henon100k[loop_delay] = CH(henon_array,5,delay=loop_delay)
filename='PESC_henon100k_embed5_'+str(num_delays)+'_delays.npz'
np.savez(datadir+filename,timeseries=henon_array,taus=delay_array,PEs=PEs_henon100k,SCs=SCs_henon100k)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)

"""

#SINE
x=np.arange(100000)*0.003
y1=np.sin(1*x)
y2=np.sin(2*x)
y3=np.sin(3*x)
y4=np.sin(4*x)
y5=np.sin(5*x)
y20=np.sin(20*x)
y50=np.sin(50*x)
y100=np.sin(100*x)
y200=np.sin(200*x)
y500=np.sin(500*x)
y1000=np.sin(1000*x)


ysum1=0.333*y1+0.333*y20+0.333*y50#+0.25*y100

#SINE plus noise
y20noise1to1=noise_array+y20
y20noise1p5to1=(noise_array/1.5)+y20
y20noise2to1=(noise_array/2.0)+y20
y20noise2p5to1=(noise_array/2.5)+y20
y20noise3to1=(noise_array/3.0)+y20
y20noise3p5to1=(noise_array/3.5)+y20
y20noise4to1=(noise_array/4.0)+y20
y20noise4p5to1=(noise_array/4.5)+y20
y20noise5to1=(noise_array/5.0)+y20
y20noise5p5to1=(noise_array/5.5)+y20
y20noise6to1=(noise_array/6.0)+y20
y20noise6p5to1=(noise_array/6.5)+y20
y20noise7to1=(noise_array/7.0)+y20
y20noise7p5to1=(noise_array/7.5)+y20
y20noise8to1=(noise_array/8.0)+y20
y20noise8p5to1=(noise_array/8.5)+y20
y20noise9to1=(noise_array/9.0)+y20
y20noise9p5to1=(noise_array/9.5)+y20
y20noise10to1=(noise_array/10.0)+y20
y20noise11to1=(noise_array/11.0)+y20
y20noise12to1=(noise_array/12.0)+y20
y20noise13to1=(noise_array/13.0)+y20
y20noise14to1=(noise_array/14.0)+y20
y20noise15to1=(noise_array/15.0)+y20
y20noise16to1=(noise_array/16.0)+y20
y20noise17to1=(noise_array/15.0)+y20
y20noise18to1=(noise_array/15.0)+y20
y20noise19to1=(noise_array/15.0)+y20
y20noise20to1=(noise_array/20.0)+y20
y20noise50to1=(noise_array/50.0)+y20
y20noise100to1=(noise_array/100.0)+y20
y20noise200to1=(noise_array/200.0)+y20
y20noise500to1=(noise_array/500.0)+y20
#plt.plot(ysum1)
"""
PEs_sine1x = np.zeros([num_delays])
SCs_sine1x = np.zeros([num_delays])
PEs_sine20x = np.zeros([num_delays])
SCs_sine20x = np.zeros([num_delays])
PEs_sine50x = np.zeros([num_delays])
SCs_sine50x = np.zeros([num_delays])
PEs_sine100x = np.zeros([num_delays])
SCs_sine100x = np.zeros([num_delays])
PEs_sine200x = np.zeros([num_delays])
SCs_sine200x = np.zeros([num_delays])
PEs_sine500x = np.zeros([num_delays])
SCs_sine500x = np.zeros([num_delays])
PEs_sine1000x = np.zeros([num_delays])
SCs_sine1000x = np.zeros([num_delays])
PEs_sine_sum1 = np.zeros([num_delays])
SCs_sine_sum1 = np.zeros([num_delays])
for loop_delay in delay_array:
    print ('On Sine Delay ', loop_delay)
    #PEs_sine1x[loop_delay],SCs_sine1x[loop_delay] = CH(y1,5,delay=loop_delay)
    #PEs_sine20x[loop_delay],SCs_sine20x[loop_delay] = CH(y20,5,delay=loop_delay)
    #PEs_sine50x[loop_delay],SCs_sine50x[loop_delay] = CH(y50,5,delay=loop_delay)
    #PEs_sine100x[loop_delay],SCs_sine100x[loop_delay] = CH(y100,5,delay=loop_delay)    
    #PEs_sine200x[loop_delay],SCs_sine200x[loop_delay] = CH(y200,5,delay=loop_delay)
    #PEs_sine500x[loop_delay],SCs_sine500x[loop_delay] = CH(y500,5,delay=loop_delay)
    #PEs_sine1000x[loop_delay],SCs_sine1000x[loop_delay] = CH(y1000,5,delay=loop_delay)
    PEs_sine_sum1[loop_delay],SCs_sine_sum1[loop_delay]=CH(ysum1,5,delay=loop_delay)
filename='PESC_sine1x_embed5_'+str(num_delays)+'_delays.npz'
#np.savez(datadir+filename,timeseries=y1,taus=delay_array,PEs=PEs_sine1x,SCs=SCs_sine1x)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
filename='PESC_sine20x_embed5_'+str(num_delays)+'_delays.npz'
#np.savez(datadir+filename,timeseries=y20,taus=delay_array,PEs=PEs_sine20x,SCs=SCs_sine20x)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
filename='PESC_sine50x_embed5_'+str(num_delays)+'_delays.npz'
#np.savez(datadir+filename,timeseries=y50,taus=delay_array,PEs=PEs_sine50x,SCs=SCs_sine50x)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
filename='PESC_sine100x_embed5_'+str(num_delays)+'_delays.npz'
#np.savez(datadir+filename,timeseries=y100,taus=delay_array,PEs=PEs_sine100x,SCs=SCs_sine100x)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
filename='PESC_sine200x_embed5_'+str(num_delays)+'_delays.npz'
#np.savez(datadir+filename,timeseries=y200,taus=delay_array,PEs=PEs_sine200x,SCs=SCs_sine200x)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
filename='PESC_sine500x_embed5_'+str(num_delays)+'_delays.npz'
#np.savez(datadir+filename,timeseries=y500,taus=delay_array,PEs=PEs_sine500x,SCs=SCs_sine500x)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
filename='PESC_sine1000x_embed5_'+str(num_delays)+'_delays.npz'
#np.savez(datadir+filename,timeseries=y1000,taus=delay_array,PEs=PEs_sine1000x,SCs=SCs_sine1000x)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
filename='PESC_sine_sum1_embed5_'+str(num_delays)+'_delays.npz'
#np.savez(datadir+filename,timeseries=ysum1,taus=delay_array,PEs=PEs_sine_sum1,SCs=SCs_sine_sum1)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
"""



#Triangle Waveform
import scipy.signal as sig
x=np.arange(100000)*0.003
#x=np.linspace(0,1,100000)
triangle=sig.sawtooth(100*x,0.5)

#plt.plot(x,triangle,'o')
#plt.plot(x,y20,'s')

PEs_triangle = np.zeros([num_delays])
SCs_triangle = np.zeros([num_delays])

#for loop_delay in delay_array:
#    print ('On Sine Delay ', loop_delay)
#    PEs_triangle[loop_delay],SCs_triangle[loop_delay] = CH(triangle,5,delay=loop_delay)

#filename='PESC_triangle_embed5_'+str(num_delays)+'_delays_rescaled.npz'
#np.savez(datadir+filename,timeseries=triangle,taus=delay_array,PEs=PEs_triangle,SCs=SCs_triangle)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)


#Calculate Distribution Function of Patterns for Sine Waves
#for loop_delay in delay_array:
permstore_counter = []   
permstore_counter = Counter(permstore_counter)
tot_perms = 0
arr,nperms=PE_dist(y20noise100to1,5,delay=1)
permstore_counter = permstore_counter+arr
tot_perms = tot_perms+nperms
PE_tot,PE_tot_Se = PE_calc_only(permstore_counter,tot_perms)
print(arr)
print('H=',PE_tot/np.log2(120))
print('Num of Permutations =',len(arr))
print('Hmax=',np.log2(1/(len(arr)))/np.log2(120))

numnoiselevels=1000
CHvalues=np.zeros([numnoiselevels,2])
for n in np.arange(1,numnoiselevels):
    CHvalues[n,:]=CH(y20+noise_array/(float(n)/2),5,delay=1)
    
filename='PESC_y20plusNoise_curves.npz'
np.savez(datadir+filename,CHvalues=CHvalues)

#CHvals1 = CH(y20,5,delay=1)
#CHvals2 = CH(y20noise500to1,5,delay=1)
#CHvals3 = CH(y20noise200to1,5,delay=1)
# CHvals4 = CH(y20noise100to1,5,delay=1)
# CHvals5 = CH(y20noise50to1,5,delay=1)

# CHvals6 = CH(y20noise20to1,5,delay=1)
# CHvals7 = CH(y20noise19to1,5,delay=1)
# CHvals7 = CH(y20noise18to1,5,delay=1)
# CHvals7 = CH(y20noise17to1,5,delay=1)
# CHvals7 = CH(y20noise16to1,5,delay=1)
# CHvals7 = CH(y20noise15to1,5,delay=1)
# CHvals8 = CH(y20noise14to1,5,delay=1)
# CHvals9 = CH(y20noise13to1,5,delay=1)
# CHvals10 = CH(y20noise12to1,5,delay=1)
# CHvals11 = CH(y20noise11to1,5,delay=1)
# CHvals12 = CH(y20noise10to1,5,delay=1)
# CHvals13 = CH(y20noise9p5to1,5,delay=1)
# CHvals14 = CH(y20noise9to1,5,delay=1)
# CHvals15 = CH(y20noise8p5to1,5,delay=1)
# CHvals16 = CH(y20noise8to1,5,delay=1)
# CHvals17 = CH(y20noise7p5to1,5,delay=1)
# CHvals18 = CH(y20noise7to1,5,delay=1)
# CHvals19 = CH(y20noise6p5to1,5,delay=1)
# CHvals20 = CH(y20noise6to1,5,delay=1)
# CHvals21 = CH(y20noise5p5to1,5,delay=1)
# CHvals22 = CH(y20noise5to1,5,delay=1)
# CHvals23 = CH(y20noise4p5to1,5,delay=1)
# CHvals24 = CH(y20noise4to1,5,delay=1)
# CHvals25 = CH(y20noise3p5to1,5,delay=1)
# CHvals26 = CH(y20noise3to1,5,delay=1)
# CHvals27 = CH(y20noise2p5to1,5,delay=1)
# CHvals28 = CH(y20noise2to1,5,delay=1)
# CHvals29 = CH(y20noise1p5to1,5,delay=1)
# CHvals30 = CH(y20noise1to1,5,delay=1)






# print('CH of y50:',CHvals1)
# print('CH of y50+500to1 noise:',CHvals7)
# print('CH of y50+200to1 noise:',CHvals6)
# print('CH of y50+100to1 noise:',CHvals5)
# print('CH of y50+50to1 noise:',CHvals4)
# print('CH of y50+20to1 noise:',CHvals3)
# print('CH of y50+15to1 noise:',CHvals2)


# x=np.array([CHvals1[0],CHvals2[0],CHvals3[0],CHvals4[0],CHvals5[0],CHvals6[0],CHvals7[0],
#             CHvals8[0],CHvals9[0],CHvals10[0],CHvals11[0],CHvals12[0],CHvals13[0],CHvals14[0],
#             CHvals15[0],CHvals16[0],CHvals17[0],CHvals18[0],CHvals19[0],CHvals20[0],
#             CHvals21[0],CHvals22[0],CHvals23[0],CHvals24[0],CHvals25[0],CHvals26[0],
#             CHvals27[0],CHvals28[0],CHvals29[0],CHvals30[0]])
# y=np.array([CHvals1[1],CHvals2[1],CHvals3[1],CHvals4[1],CHvals5[1],CHvals6[1],CHvals7[1],
#             CHvals8[1],CHvals9[1],CHvals10[1],CHvals11[1],CHvals12[1],CHvals13[1],CHvals14[1],
#             CHvals15[1],CHvals16[1],CHvals17[1],CHvals18[1],CHvals19[1],CHvals20[1],
#             CHvals21[1],CHvals22[1],CHvals23[1],CHvals24[1],CHvals25[1],CHvals26[1],
#             CHvals27[1],CHvals28[1],CHvals29[1],CHvals30[1]])
# print(x)
# print(y)





#CHy20_and_y50_half=CH(y20py50,5,delay=1)
"""
#Calculate embedding delay scan of 50/50 y20 and y50
delay_array = np.arange(1,100)#0)
num_delays = len(delay_array)+1
y20py50 = (0.01*y20)+(0.99*y50)
PEs_sum20p50 = np.zeros([num_delays])
SCs_sum20p50 = np.zeros([num_delays])
for loop_delay in delay_array:
    print ('On Sine Sum Delay ', loop_delay)
    PEs_sum20p50[loop_delay],SCs_sum20p50[loop_delay] = CH(y20py50,5,delay=loop_delay)
filename='PESC_sum20p50_01-99_embed5_'+str(num_delays)+'_delays.npz'
np.savez(datadir+filename,timeseries=y20py50,taus=delay_array,PEs=PEs_sum20p50,SCs=SCs_sum20p50)
"""




"""
#fbm 0.5
data=loadnpzfile(datadir+'fbm_H0.05_N10000_1.npz')
y=data['x']
PEs_fbm_p5 = np.zeros([num_delays])
SCs_fbm_p5 = np.zeros([num_delays])

for loop_delay in delay_array:
    print ('On Sine Delay ', loop_delay)
    PEs_fbm_p5[loop_delay],SCs_fbm_p5[loop_delay] = CH(y,5,delay=loop_delay)

filename='PESC_fbm_p5_embed5_'+str(num_delays)+'_delays.npz'
np.savez(datadir+filename,timeseries=y,taus=delay_array,PEs=PEs_fbm_p5,SCs=SCs_fbm_p5)#,delta_t=delta_t,taus=taus,delays=delays,freq=freq)
"""

