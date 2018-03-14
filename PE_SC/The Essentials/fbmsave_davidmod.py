#fbmsave.py

def fbm(N,H,tag=0):		#Saves an fBm time series for a particular length N and Hurst exponent H. The tag distinguishes time series saved this way from those of 				# saved in a scan of the Hurst exponent as in the next function.
    import numpy as np
    C=np.zeros([N,N])
    for i in np.arange(N):
        if i%100==0:
            print 'On I',i
        for j in np.arange(N):
            C[i,j]=0.5*(i**(2*H)+j**(2*H)-abs(i-j)**(2*H))
    eigs,Q = np.linalg.eig(C)
    D=np.diag(np.sqrt(eigs))
    M=Q.dot(D).dot(np.linalg.inv(Q))
    mu, sigma = 0, 1 
    v = np.random.normal(mu, sigma, N)
    w=M.dot(v)
    address = 'C:\\Users\\dschaffner\\Documents\\GitHub\\SSX_python\\PE_SC\\The Essentials\\fbm_data\\fbm_H'+str(H)+'_N'+str(N)+'_'+str(tag)+'.npz'
    np.savez_compressed(address, x=w)
    return w

def saveall(N,tag=1):		#Saves fBm time series for a range of Hurst exponents, 0.05 to 0.95 in steps of 0.05.
	import numpy as np
	import scipy as sp
	import math
	
	for tag in np.arange(1,4):
		mu, sigma = 0, 1 
		v = np.random.normal(mu, sigma, N)
		print tag
		for H in np.arange(0.05,1,0.05):
			print H
			C=np.zeros([N,N])
			for i in np.arange(N):
				for j in np.arange(N):
					C[i,j]=0.5*(i**(2*H)+j**(2*H)-abs(i-j)**(2*H))
			eigs,Q = np.linalg.eig(C)
			D=np.diag(np.sqrt(eigs))
			M=Q.dot(D).dot(np.linalg.inv(Q))
			w=M.dot(v)
			address = '/Users/Peter J Weck/Documents/Thesis/extimeseries/fbm_H'+str(H)+'_N'+str(N)+'_'+str(tag)+'.npz'
			np.savez_compressed(address, x=w)

def plot(N=5000,tag=1):  	#Plots fBm time series for 3 Hurst exponents

	import numpy as np
	import matplotlib.pylab as plt
	import math

	N=5000
	tag=1
	n=5
	fbm1=np.load('/Users/Peter J Weck/Documents/Thesis/extimeseries/fbm_H0.1_N'+str(N)+'_'+str(tag)+'.npz')['x']
	fbm2=np.load('/Users/Peter J Weck/Documents/Thesis/extimeseries/fbm_H0.5_N'+str(N)+'_'+str(tag)+'.npz')['x']
	fbm3=np.load('/Users/Peter J Weck/Documents/Thesis/extimeseries/fbm_H0.9_N'+str(N)+'_'+str(tag)+'.npz')['x']

	fig1=plt.figure(1)
	plt.plot(fbm1[0:500], linestyle='-', marker='None',color = '0.45',zorder=1)
	plt.xlabel("Time Steps", fontsize=15)
	plt.title('H = 0.1')
	plt.xticks(np.arange(0,600,100))
	savefile='fbm_H0.1_timeseries'
	plt.savefig(str(savefile)+'.png')

	fig2=plt.figure(2)
	plt.plot(fbm2[0:500], linestyle='-', marker='None',color = '0.45',zorder=1)
	plt.xlabel("Time Steps", fontsize=15)
	plt.title('H = 0.5')
	plt.xticks(np.arange(0,600,100))
	savefile='fbm_H0.5_timeseries'
	plt.savefig(str(savefile)+'.png')

	fig3=plt.figure(3)
	plt.plot(fbm3[0:500], linestyle='-', marker='None',color = '0.45',zorder=1)
	plt.xlabel("Time Steps", fontsize=15)
	plt.title('H = 0.9')
	plt.xticks(np.arange(0,600,100))
	savefile='fbm_H0.9_timeseries'
	plt.savefig(str(savefile)+'.png')	
