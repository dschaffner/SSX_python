# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:03:09 2018

@author: dschaffner
"""

import load_picoscope_bmx_102918 as ldbmx
import numpy as np
import spectrum_wwind as spec
import ssx_functions as ssxf
import matplotlib.pylab as plt
import MLE as mle
import get_corr as gc

port_sep = 0.0254#m
delaytimes_2p2 = np.zeros([10])
delaytimes_1p8 = np.zeros([10])


for shot in np.arange(10,30):
    time_ms,time_s,timeB_s,Brdot7,Brdot9,Btdot7,Btdot9,Bzdot7,Bzdot9,Br7,Br9,Bt7,Bt9,Bz7,Bz9,disI,light=ldbmx.load_picoscope(shot)

    timeB_ms=timeB_s*1e6
    tindex1 = ssxf.tindex_min(timeB_ms,20.0)
    tindex2 = ssxf.tindex_min(timeB_ms,198.0)
    tau,corr = gc.get_corr(timeB_s[tindex1:tindex2],Br7[tindex1:tindex2],Br9[tindex1:tindex2],normalized=False)

    if shot==16: plt.plot(tau,corr)
    print np.argmax(11000+corr[11000:11250])
    delay=tau[11000+np.argmax(corr[11000:11250])]
    print delay,' sec'
    if shot<20: delaytimes_2p2[shot-10]=delay
    if shot>=20: delaytimes_1p8[shot-20]=delay
    
vels_2p2 = (port_sep/delaytimes_2p2)/1000.0#km/s
vels_1p8 = (port_sep/delaytimes_1p8)/1000.0#km/s
"""
date='20181022'

shot_range=[5,19]
shot_range=[20,26]
time_range=[38,198]#fsize=10001
#time_range=[50,100]#fsize=3126
fsize=10001
avespecbx=np.zeros([fsize])
avespecby=np.zeros([fsize])
avespecbz=np.zeros([fsize])

for shot in np.arange(shot_range[0],shot_range[1]+1):
    time_ms,time_s,timeB_s,Bxdot,Bydot,Bzdot,Bx,By,Bz,Bxfilt,Byfilt,Bzfilt,Btot=ldbmx.load_picoscope(date,shot)
    #time_s=time*1e-6
    tindex1 = ssxf.tindex_min(time_ms,time_range[0])
    tindex2 = ssxf.tindex_min(time_ms,time_range[1])
    f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Bxdot[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
    avespecbx=avespecbx+(pwrbxdot/(f*f))
    f,f0,comp1,pwrbydot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Bydot[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
    avespecby=avespecby+(pwrbydot/(f*f))
    f,f0,comp1,pwrbzdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Bzdot[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
    avespecbz=avespecbz+(pwrbzdot/(f*f))

plt.figure(1)    
dlog=np.log10(avespecbx[4:20])
flog=np.log10(f[4:20])
A1=np.array([flog,np.ones(len(flog))])
w1=np.linalg.lstsq(A1.T,dlog)[0]
slope=np.round(w1[0],3)
func = f**(slope)
scale_data=avespecbx[20]
scale_fit=func[20]
ratio=scale_data/scale_fit
plt.loglog(f,avespecbx,'o')
plt.loglog(f[4:20],avespecbx[4:20])
plt.loglog(f,ratio*func)
print 'Bx slope is',slope

plt.figure(2)    
dlog=np.log10(avespecby[20:400])
flog=np.log10(f[20:400])
A1=np.array([flog,np.ones(len(flog))])
w1=np.linalg.lstsq(A1.T,dlog)[0]
slope=np.round(w1[0],3)
func = f**(slope)
scale_data=avespecby[100]
scale_fit=func[100]
ratio=scale_data/scale_fit
plt.loglog(f,avespecby,'o')
plt.loglog(f[20:400],avespecby[20:400])
plt.loglog(f,ratio*func)
print 'By slope is',slope

plt.figure(3)    
dlog=np.log10(avespecbz[4:20])
flog=np.log10(f[4:20])
A1=np.array([flog,np.ones(len(flog))])
w1=np.linalg.lstsq(A1.T,dlog)[0]
slope=np.round(w1[0],3)
func = f**(slope)
scale_data=avespecbz[20]
scale_fit=func[20]
ratio=scale_data/scale_fit
plt.loglog(f,avespecbz,'o')
plt.loglog(f[4:20],avespecbz[4:20])
plt.loglog(f,ratio*func)
print 'Bz slope is',slope
"""