import loadnpzfile as ld
import calc_PE_SC_qlog_v2 as ql
import matplotlib.pylab as plt
import numpy as np
from collections import Counter

datadir = 'C:\\Users\\dschaffner.BRYNMAWR\\OneDrive - brynmawr.edu\\Solar Wind Data\\NPZ_files\\'
fileheader = 'Wind20050101'
npz='.npz'

###Storage Arrays###
#delta_t = 1.0/(40000.0)
#delays = np.arange(2,250) #248 elements
#taus = delays*delta_t
#freq = 1.0/taus

permstore_counter = []
permstore_counter = Counter(permstore_counter)
tot_perms = 0

datafile = ld.loadnpzfile(datadir+fileheader+npz)
bx = datafile['bx']
count,Ptot=ql.PE_dist(bx,d=5,delay=1)
permstore_counter = permstore_counter+count
tot_perms = tot_perms+Ptot
s,su,du,dstar = ql.q_PESC_calc(permstore_counter,tot_perms,d=5,qnum=500000)
H=s[1:]/su[1:]
C=(H*du[1:])/dstar[1:]
plt.plot(H,C)
filename='C:\\Users\\dschaffner.BRYNMAWR\\OneDrive - brynmawr.edu\\Solar Wind Data\\NPZ_files\\windbx_PESCqlog_d5_t1_qnum500000_0to50.npz'
np.savez(filename,s=s,su=su,du=du,dstar=dstar)

permstore_counter = []
permstore_counter = Counter(permstore_counter)
tot_perms = 0
by = datafile['by']
count,Ptot=ql.PE_dist(by,d=5,delay=1)
permstore_counter = permstore_counter+count
tot_perms = tot_perms+Ptot
s,su,du,dstar = ql.q_PESC_calc(permstore_counter,tot_perms,d=5,qnum=500000)
H=s[1:]/su[1:]
C=(H*du[1:])/dstar[1:]
plt.plot(H,C)
filename='C:\\Users\\dschaffner.BRYNMAWR\\OneDrive - brynmawr.edu\\Solar Wind Data\\NPZ_files\\windby_PESCqlog_d5_t1_qnum500000_0to50.npz'
np.savez(filename,s=s,su=su,du=du,dstar=dstar)

permstore_counter = []
permstore_counter = Counter(permstore_counter)
tot_perms = 0
bz = datafile['bz']
count,Ptot=ql.PE_dist(bz,d=5,delay=1)
permstore_counter = permstore_counter+count
tot_perms = tot_perms+Ptot
s,su,du,dstar = ql.q_PESC_calc(permstore_counter,tot_perms,d=5,qnum=500000)
H=s[1:]/su[1:]
C=(H*du[1:])/dstar[1:]
plt.plot(H,C)
filename='C:\\Users\\dschaffner.BRYNMAWR\\OneDrive - brynmawr.edu\\Solar Wind Data\\NPZ_files\\windbz_PESCqlog_d5_t1_qnum500000_0to50.npz'
np.savez(filename,s=s,su=su,du=du,dstar=dstar)