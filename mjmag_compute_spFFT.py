#plot_mjmag.py

import process_mjmag_data as mj
import numpy as np
import matplotlib.pylab as plt
import spectrum_wwind as sp

day='123016'
shot=70

#load mj probe data [xyz,position,timestep]
#time = time index [8192]
#bdot = calibrated dB/dt data [3,25,8192]
#timeb = time index for integrated data [8191]
#b = integrated magnetic field data [3,25,8192]
#bmod = modulus of b for doublets and triplets
time,bdot,timeb,b,bmod,fulldata = mj.process_mjmag_data(day+str(shot))
pos = np.arange(19)*0.015#in meters

fft_tot = np.zeros([10])
count=0

plt.figure(1)
plt.clf()
plt.plot(pos,b[0,:19,3000])
plt.xlabel('Pos [m]')
plt.ylabel('B [G]')


for t in np.arange(len(timeb)):
    if np.mod(t,1000) == 0:
        print 'On step, ',t
    f,f0,comp1,pwrbx,mag1,phase1,cos_phase1,interval=sp.spectrum_wwind(b[0,:19,t],pos,window='hanning')
    fft_tot=fft_tot+pwrbx
    count+=1

plt.figure(2)
plt.clf()
plt.loglog(f,fft_tot,'o-')

fft_tot = np.zeros([10])
count=0

for t in np.arange(0000,2000):
    if np.mod(t,1000) == 0:
        print 'On step, ',t
    f,f0,comp1,pwrbx,mag1,phase1,cos_phase1,interval=sp.spectrum_wwind(b[0,:19,t],pos,window='hanning')
    fft_tot=fft_tot+pwrbx
    count+=1

plt.figure(3)
plt.clf()
plt.loglog(f,fft_tot,'o-')

fft_tot = np.zeros([10])
count=0

for t in np.arange(3000,5000):
    if np.mod(t,1000) == 0:
        print 'On step, ',t
    f,f0,comp1,pwrbx,mag1,phase1,cos_phase1,interval=sp.spectrum_wwind(b[0,:19,t],pos,window='hanning')
    fft_tot=fft_tot+pwrbx
    count+=1

plt.figure(4)
plt.clf()
plt.loglog(f,fft_tot,'o-')

fft_tot = np.zeros([10])
count=0

for t in np.arange(6000,7000):
    if np.mod(t,1000) == 0:
        print 'On step, ',t
    f,f0,comp1,pwrbx,mag1,phase1,cos_phase1,interval=sp.spectrum_wwind(b[0,:19,t],pos,window='hanning')
    fft_tot=fft_tot+pwrbx
    count+=1

plt.figure(5)
plt.clf()
plt.loglog(f,fft_tot,'o-')
#plt.xlabel('Time [us]')
#plt.ylabel('B [G]')

"""
plt.figure(2)
plt.clf()
pos = np.arange(19)*1.5#in cm
plt.plot(pos,b[0,0:19,2300],'o',linestyle='dotted')
plt.xlabel('Position [cnm]')
plt.ylabel('B [G]')

plt.figure(3)
plt.clf()
pos = np.arange(19)*1.5#in cm
plt.plot(pos,b[0,0:19,4000],'o',linestyle='dotted')
plt.xlabel('Position [cnm]')
plt.ylabel('B [G]')
"""