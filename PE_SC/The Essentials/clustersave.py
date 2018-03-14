# Saves Cluster satellite time series, for all four spacecraft and all three orthogonal magnetic field components,
# in GSE coordinates (x along earth sun, z vertically out of earth-sun plane, etc.). 2nd function plots one of the time series.

def save():
	import numpy as np
	import scipy.io as sp

	mat = sp.loadmat('STAFFFGMCombined20030212GSE.mat')

	for i in range(4):
		for j in range(1,5):
			X = mat['data'][0,i][:,j]
			address = '/Users/Peter J Weck/Documents/Thesis/extimeseries/cluster'+str(i+1)+'_B'+str(j)+'.npz'
			np.savez_compressed(address, x=X)

def plot(craft=2,coord=1):

	import numpy as np
	import matplotlib.pylab as plt
	import math
	n=5
	data=np.load('/Users/Peter J Weck/Documents/Thesis/extimeseries/cluster'+str(craft)+'_B'+str(coord)+'.npz')['x']
	t=np.arange(0,500*0.04,0.04)
	fig1=plt.figure(1)
	plt.plot(t,data[0:500], linestyle='-', marker='None',color = '0.45',zorder=1)
	plt.xlabel("Time (s)", fontsize=15)
	plt.ylabel("Magnetic Field", fontsize=15)
	plt.title('B_'+str(coord))
#	plt.xticks(np.arange(0,501*0.04,1))
	savefile='cluster'+str(craft)+'_B'+str(coord)+'_timeseries'
	plt.savefig(str(savefile)+'.png')
