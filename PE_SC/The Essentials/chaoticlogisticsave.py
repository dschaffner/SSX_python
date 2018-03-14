#logisticsave.py

def logisave(N,r=4.):

	import numpy as np
	import scipy as sp
	import matplotlib.pylab as plt
	
	X = np.zeros([N])
	X[0] = 0.1

	for i in range(1,N):
		X[i] = r * X[i-1] * (1 - X[i-1])
	address = '/Users/peterweck/Documents/extimeseries/logis'+str(N)+'.npz'
	np.savez_compressed(address, x=X)
