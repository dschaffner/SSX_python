#Function to calculate Cmin, Cmax with steps of size nsteps. Second function plots a blank CH plane.
def Cmaxmin(nsteps,n):	
	import numpy as np
	import matplotlib.pylab as plt
	import math
	
	N = math.factorial(n)
	Cmaxx = np.zeros((N-1)*nsteps)
	Cmaxy = np.zeros((N-1)*nsteps)
	Cminx = np.zeros(nsteps)
	Cminy = np.zeros(nsteps)
	
	for i in range(nsteps):
		pk = 1./N + i*(1.-(1./N))/nsteps
		pj = (1. - pk)/(N - 1.)
		S = -pk * np.log2(pk) - (N - 1.) * pj * np.log2(pj)
		qk = pk/2. + 1./(2.*N)
		qj = pj/2. + 1./(2.*N)
		Scom = -qk * np.log2(qk) - (N - 1.) * qj * np.log2(qj)
		Cminx[i] = S / np.log2(N)
		Cminy[i] = -2. * (S/np.log2(N)) * (Scom - 0.5*S - 0.5*np.log2(N)) \
		/((1 + 1./N)*np.log2(N+1) - 2*np.log2(2*N) + np.log2(N))	
		
	for i in range(1,N):
		for l in range(nsteps):
			pk = l*(1./(N-i+1.))/nsteps
			pj = (1. - pk)/(N - i)
			if pk ==0.:
				S = -(N - i) * pj * np.log2(pj)
			else:
				S = -pk * np.log2(pk) - (N - i) * pj * np.log2(pj)
			qk = pk/2. + 1./(2.*N)
			qj = pj/2. + 1./(2.*N)
			Scom = -qk * np.log2(qk) - (N - i) * qj * np.log2(qj) - \
			(i-1)*(1./(2.*N))*np.log2(1./(2.*N))
			Cmaxx[(i-1.)*nsteps+l] = S / np.log2(N)
			Cmaxy[(i-1.)*nsteps+l] = -2.*(S/np.log2(N))*(Scom - 0.5*S - 0.5*np.log2(N)) \
			/((1. + 1./N)*np.log2(N+1.) - 2.*np.log2(2.*N) + np.log2(N))
			
	return Cminx, Cminy, Cmaxx, Cmaxy

def blank():				# Creates a blank CH plane
	import numpy as np
	import matplotlib.pylab as plt
	import math
	from Cmaxmin import Cmaxmin
	
	n=5
	Cminx, Cminy, Cmaxx, Cmaxy = Cmaxmin(100,n)

	fig1=plt.figure(1)
	plt.plot(Cminx,Cminy,'k-',Cmaxx,Cmaxy,'k-')
	plt.xlabel("Entropy", fontsize=15)
	plt.ylabel("Jensen-Shannon Complexity", fontsize=15)
	plt.axis([0,1.0,0,0.45])
	plt.xticks(np.arange(0,1.1,0.1))
	plt.yticks(np.arange(0,0.45,0.05))
	savefile='CHn5_blank'
	plt.savefig(str(savefile)+'.png')
