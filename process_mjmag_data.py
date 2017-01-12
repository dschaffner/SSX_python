#process_mjmag_data.py

import dtacqmag as hdr
import numpy as np
import scipy.integrate as sp
import matplotlib.pylab as plt

def process_mjmag_data(shot):
    data = hdr.getMJMagData(shot)

    #recast array into [3,25]
    Bdot25 = np.zeros([3,25,8192])
    #1
    Bdot25[0,0,:]=data.Bdot[0,0,:]
    Bdot25[1,0,:]=data.Bdot[1,0,:]
    #2
    Bdot25[0,1,:]=data.Bdot[0,1,:]
    Bdot25[1,1,:]=data.Bdot[1,1,:]
    #3
    Bdot25[0,2,:]=data.Bdot[0,2,:]
    Bdot25[1,2,:]=data.Bdot[1,2,:]
    #4
    Bdot25[0,3,:]=data.Bdot[0,3,:]
    Bdot25[1,3,:]=data.Bdot[1,3,:]
    #5
    Bdot25[0,4,:]=data.Bdot[0,4,:]
    Bdot25[1,4,:]=data.Bdot[1,4,:]
    #6
    Bdot25[0,5,:]=data.Bdot[0,5,:]
    Bdot25[1,5,:]=data.Bdot[1,5,:]
    #7
    Bdot25[0,6,:]=data.Bdot[0,6,:]
    Bdot25[1,6,:]=data.Bdot[1,6,:]
    #8
    Bdot25[0,7,:]=data.Bdot[0,7,:]
    Bdot25[1,7,:]=data.Bdot[1,7,:]
    #9
    Bdot25[0,8,:]=data.Bdot[0,8,:]
    Bdot25[1,8,:]=data.Bdot[1,8,:]
    #10
    Bdot25[0,9,:]=data.Bdot[0,9,:]
    Bdot25[1,9,:]=data.Bdot[1,9,:]
    #11
    Bdot25[0,10,:]=data.Bdot[0,10,:]
    Bdot25[1,10,:]=data.Bdot[1,10,:]
    #12
    Bdot25[0,11,:]=data.Bdot[0,11,:]
    Bdot25[1,11,:]=data.Bdot[1,11,:]
    #13
    Bdot25[0,12,:]=data.Bdot[0,12,:]
    Bdot25[1,12,:]=data.Bdot[1,12,:]
    Bdot25[2,12,:]=data.Bdot[2,2,:]
    #14
    Bdot25[0,13,:]=data.Bdot[0,13,:]
    Bdot25[1,13,:]=data.Bdot[1,13,:]
    Bdot25[2,13,:]=data.Bdot[2,1,:]
    #15
    Bdot25[0,14,:]=data.Bdot[0,14,:]
    Bdot25[1,14,:]=data.Bdot[1,14,:]
    Bdot25[2,14,:]=data.Bdot[2,0,:]
    #16
    Bdot25[0,15,:]=data.Bdot[0,15,:]
    Bdot25[1,15,:]=data.Bdot[1,15,:]
    #17
    Bdot25[0,16,:]=data.Bdot[2,3,:]
    Bdot25[1,16,:]=data.Bdot[2,12,:]
    #18
    Bdot25[0,17,:]=data.Bdot[2,4,:]
    Bdot25[1,17,:]=data.Bdot[2,13,:]
    #19
    Bdot25[0,18,:]=data.Bdot[2,5,:]
    Bdot25[1,18,:]=data.Bdot[2,14,:]
    #20
    Bdot25[0,19,:]=data.Bdot[2,6,:]
    Bdot25[1,19,:]=data.Bdot[2,15,:]
    #21
    Bdot25[0,20,:]=data.Bdot[2,7,:]
    #22
    Bdot25[0,21,:]=data.Bdot[2,8,:]
    #23
    Bdot25[0,22,:]=data.Bdot[2,9,:]
    #24
    Bdot25[0,23,:]=data.Bdot[2,10,:]
    #25
    Bdot25[0,24,:]=data.Bdot[2,11,:]
    
    #reintegrate with later start times
    B25=np.zeros([3,25,7391])
    timeB=data.time[801:]
    for i in np.arange(3):
        for j in np.arange(25):
                #bzero = Bdot25[i,j,:]-np.mean(Bdot25[i,j,800:8000])
                bzero = plt.detrend_linear(Bdot25[i,j,800:])
                bint = sp.cumtrapz(bzero,data.time[800:])
                B25[i,j,:]=bint
                
    #compute Bmod
    Bmod25 = np.zeros([25,7391])
    for j in np.arange(25):
        Bmod25[j,:] = np.sqrt(B25[0,j,:]**2+B25[1,j,:]**2+B25[2,j,:]**2)
    return data.time,Bdot25,timeB,B25,Bmod25,data