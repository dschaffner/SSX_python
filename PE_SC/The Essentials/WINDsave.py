import numpy as np
import matplotlib.pylab as plt

data1=np.loadtxt('WI_H0_MFI_Fast.txt', skiprows=100, usecols=(2,3,4))
for i in range(3):
	address = '/Users/peterweck/Documents/extimeseries/WI_H0_MFI_'+str(i)+'.npz'
	np.savez_compressed(address, x=data1[:,i])
data2=np.loadtxt('WI_H0_MFI_CME.txt', skiprows=100, usecols=(2,3,4))
for i in range(3):
	address = '/Users/peterweck/Documents/extimeseries/WI_H0_MFI_CME_'+str(i)+'.npz'
	np.savez_compressed(address, x=data2[:,i])
data3=np.loadtxt('WI_H0_MFI_Slow.txt', skiprows=100, usecols=(2,3,4))
for i in range(3):
	address = '/Users/peterweck/Documents/extimeseries/WI_H0_MFI_Slow_'+str(i)+'.npz'
	np.savez_compressed(address, x=data3[:,i])