import loadnpzfile as ld
import calc_PE_SC_qlog_v2 as ql
import matplotlib.pylab as plt
import numpy as np

datafile=ld.loadnpzfile('C:\\Users\\dschaffner\\Google Drive\\code\\The Essentials\\fbm_data\\fbm_H0.45_N10000_0.npz')
x=datafile['x']
count,Ptot=ql.PE_dist(x,d=5,delay=1)
#s,se,su = ql.q_PE_calc(count,Ptot,n=4,qnum=100000)
s,su,du,dstar = ql.q_PESC_calc(count,Ptot,d=5,qnum=500000)
H=s[1:]/su[1:]
C=(H*du[1:])/dstar[1:]
plt.plot(H,C)

filename='C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Corrsin Wind Tunnel Data\\NPZ_files\\fbm0.45_PESCqlog_d5_t1_qnum500000_0to50.npz'
np.savez(filename,s=s,su=su,du=du,dstar=dstar)