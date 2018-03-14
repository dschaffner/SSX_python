#tentsave.py
#Saves length N tent series

def tentsave(N):

	import numpy as np
	import scipy as sp
	import matplotlib.pylab as plt
	
	w=0.1847
	X = np.zeros([N])
	X[0] = 0.1

	for i in range(1,N):
		if X[i-1] < w:
			X[i] = X[i-1]/w
		else:
			X[i] = (1 - X[i-1])/(1 - w)
	address = '/Users/peterweck/Documents/extimeseries/tent'+str(N)+'.npz'
	np.savez_compressed(address, x=X)
