#henonsave.py

def henonsave(N):

    import numpy as np
    import scipy as sp
    import matplotlib.pylab as plt
    X = np.zeros((2,N))
    X[0,0] = 1.
    X[1,0] = 1.
    a = 1.4
    b = 0.3
    for i in range(1,N):
        X[0,i] = 1. - a * X[0,i-1] ** 2. + X[1,i-1]
        X[1,i] = b * X[0,i-1]
    return X
	#address = '/Users/peterweck/Documents/extimeseries/henon'+str(N)+'.npz'
	#np.savez_compressed(address, x=X[0,:])
