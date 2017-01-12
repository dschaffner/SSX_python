#plot_mjmag.py

import process_mjmag_data as mj
import numpy as np
import matplotlib.pylab as plt

day='123016'
shot=70

#load mj probe data [xyz,position,timestep]
#time = time index [8192]
#bdot = calibrated dB/dt data [3,25,8192]
#timeb = time index for integrated data [8191]
#b = integrated magnetic field data [3,25,8192]
#bmod = modulus of b for doublets and triplets
time,bdot,timeb,b,bmod,fulldata = mj.process_mjmag_data(day+str(shot))

plt.figure(1)
plt.clf()
plt.plot(timeb,b[0,0,:])
plt.xlabel('Time [us]')
plt.ylabel('B [G]')

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