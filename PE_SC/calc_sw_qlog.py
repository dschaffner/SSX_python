import loadnpzfile as ld
import calc_PE_SC_qlog_v2 as ql
import matplotlib.pylab as plt
import numpy as np
from collections import Counter

datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Corrsin Wind Tunnel Data\\NPZ_files\\m20_5mm\\streamwise\\'
fileheader = 'm20_5mm_p2shot'
npz='.npz'

###Storage Arrays###
delta_t = 1.0/(40000.0)
delays = np.arange(2,250) #248 elements
taus = delays*delta_t
freq = 1.0/taus

permstore_counter = []
permstore_counter = Counter(permstore_counter)
tot_perms = 0

for shot in np.arange(1,121):#(1,120):
    print '###### On Shot '+str(shot)+' #####'
    datafile = ld.loadnpzfile(datadir+fileheader+str(shot)+npz)
    data = datafile['shot']
    count,Ptot=ql.PE_dist(data,d=5,delay=10)
    permstore_counter = permstore_counter+count
    tot_perms = tot_perms+Ptot
s,su,du,dstar = ql.q_PESC_calc(permstore_counter,tot_perms,d=5,qnum=500000)
H=s[1:]/su[1:]
C=(H*du[1:])/dstar[1:]
plt.plot(H,C)
filename='C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Corrsin Wind Tunnel Data\\NPZ_files\\m20_5mm_p2_stream_121shots_PESCqlog_d5_t10_qnum500000_0to50.npz'
np.savez(filename,s=s,su=su,du=du,dstar=dstar)