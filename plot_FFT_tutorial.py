#plot_FFT_tutorial.py

import numpy as np
import matplotlib.pylab as plt
import spectrum_wwind as spec

#Here is some sample sine wave data

x=np.arange(10000)*0.01

y1=np.sin(1*x)
y2=np.sin(2*x)
y3=np.sin(3*x)
y10=np.sin(10*x)
y20=np.sin(20*x)

#figure 1 is a plot of some of these sine waves
plt.figure(1)
plt.plot(x,y1)
plt.plot(x,y3)
plt.plot(x,y10)
plt.xlim(0,15)
plt.ylim(-2,2)

#We can make a signal out of different combinations of these sine waves with different amplitudes

sig1 = 10*y1+2*y2+4*y10
#This signal has lots of power in low frequencies, and less power in low frequencies

#figure 2 is a plot of this new combined signal
plt.figure(2)
plt.plot(x,sig1)
plt.xlim(0,20)
plt.ylim(-20,20)

sig2 = 0.1*y1+10*y3+10*y10+20*y20
#This signal has little power at low frequencies but lots of power at high frequencies

#figure 3 is a plot of this new combined signal
plt.figure(3)
plt.plot(x,sig2)
plt.xlim(0,20)
plt.ylim(-50,50)


######## Now we'll apply an FFT to the signals
# We import the python file called spectrum_wwind.py which has the function spectrum_wwind
# The inputs to this function are the array of interest, the x-array (time or position, say), and a window-->we won't worry about window now. It is an optional input.
# There are 8 outputs of this function. We typically only need two of them, but we'll need to give a variable for all of them

#Here's the spectrum for a single sine wave
freq, freq2, comp, pwr10, mag, phase2, cos_phase, dt = spec.spectrum_wwind(y10,x)
freq, freq2, comp, pwr1, mag, phase2, cos_phase, dt = spec.spectrum_wwind(y1,x)

#We want to plot the log of power versus the log of freqency
plt.figure(4)
plt.loglog(freq,pwr10)
#Note that there is a clear peak in the signal. This says that all the power in this signal is at a single frequency. 
#Since sine assumes things in radians, the frequency associated with 10*x is 10/2pi = 1.59. Thus the peak looks to be about 1.59 in log scale (between 1e0 and 2e0)

#Here's the spectrum for sig1 and sig2

freqA, freq2A, compA, pwrA, magA, phase2A, cos_phaseA, dtA = spec.spectrum_wwind(sig1,x)
freqB, freq2B, compB, pwrB, magB, phase2B, cos_phaseB, dtB = spec.spectrum_wwind(sig2,x)

#We'll now plot the two spectra on the same plot
plt.figure(5)
plt.loglog(freqA,pwrA,label='sig1')
plt.loglog(freqB,pwrB,label='sig2')
plt.loglog(freq,pwr1,label='sine(1x)')
plt.legend(loc='upper left')

#how compare the locations and heights of the different signals to the equations that define sig1 and sig2. Do they make sense? 
#Do the positions of the peaks match the frequencies you'd expect?
