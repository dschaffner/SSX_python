import loadnpzfile as ld
import calc_PE_SC_qlog as ql
import matplotlib.pylab as plt

datafile=ld.loadnpzfile('C:\\Users\\dschaffner\\Google Drive\\code\\The Essentials\\fbm_data\\fbm_H0.9_N10000_0.npz')
x=datafile['x']
count,Ptot=ql.PE_dist(x,n=4,delay=1)
s,se,su = ql.q_PE_calc(count,Ptot,n=4,qnum=100000)

plt.plot(s[1:]/su[1:])