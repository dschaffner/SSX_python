import numpy as np
import matplotlib.pylab as plt
from loadnpyfile import loadnpyfile
from calc_PE_SC import PE, CH, PE_dist, PE_calc_only
import Cmaxmin as cpl
from collections import Counter
from math import factorial
import os


orbits = np.zeros([1400,1001])
x=np.arange(1001)
period=500.0

y1=np.sin((2*np.pi/(period)*x)+np.random.randint(1,1000+1))
y2=0.1*np.sin((2*np.pi/(period/(np.sqrt(2)))*x)+np.random.randint(1,1000+1))
#y2=np.sin(2*x)
#y3=np.sin(3*x)
#y10=np.sin(10*x)
#y20=np.sin(20*x)


for orbit in np.arange(1400):
    orbits[orbit,:]=5.8+(np.sin((2*np.pi/(period)*x)+np.random.uniform(0,1000)))#+(0.1*np.sin((2*np.pi/(period/(np.sqrt(2)))*x)+np.random.uniform(0,1000)))
    #orbits[orbit,:]=(np.sin((2*np.pi/(period)*x)+np.random.uniform(0,1000)))+(2.0*np.sin((2*np.pi/(1.0*period/(np.sqrt(2)))*x)+np.random.uniform(0,1000)))
    
datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\Data040318\\'
#filename = 'Sine_1500period_wtimesroot2-200percent_randomphase_10k.npy'
filename = 'Sine_500period_randomphase_shortrec.npy'
np.save(datadir+filename,orbits)

#plt.plot(x,y1+y2)
#plt.plot(x,y2)

