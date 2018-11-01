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

date='20181022'

shot_range=[5,19]#102218
shot_range=[20,26]#102218
shot_range=[10,19]#102918
shot_range=[20,29]#102918
time_range=[38,198]#fsize=10001
time_range=[20,198]#fsize=11126
#time_range=[50,100]#fsize=3126
fsize=11126
avespecbr7=np.zeros([fsize])
avespecbr9=np.zeros([fsize])
avespecbt7=np.zeros([fsize])
avespecbt9=np.zeros([fsize])
avespecbz7=np.zeros([fsize])
avespecbz9=np.zeros([fsize])

for shot in np.arange(shot_range[0],shot_range[1]+1):
    time_ms,time_s,timeB_s,Brdot7,Brdot9,Btdot7,Btdot9,Bzdot7,Bzdot9,Br7,Br9,Bt7,Bt9,Bz7,Bz9,disI,light=ldbmx.load_picoscope(shot)
    #time_s=time*1e-6
    tindex1 = ssxf.tindex_min(time_ms,time_range[0])
    tindex2 = ssxf.tindex_min(time_ms,time_range[1])
    f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Brdot7[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
    avespecbr7=avespecbr7+(pwrbxdot/(f*f))
    f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Brdot9[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
    avespecbr9=avespecbr9+(pwrbxdot/(f*f))
    f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Btdot7[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
    avespecbt7=avespecbt7+(pwrbxdot/(f*f))
    f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Btdot9[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
    avespecbt9=avespecbt9+(pwrbxdot/(f*f))
    f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Bzdot7[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
    avespecbz7=avespecbz7+(pwrbxdot/(f*f))
    f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Bzdot9[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
    avespecbz9=avespecbz9+(pwrbxdot/(f*f))

tot_spec7=avespecbr7+avespecbt7+avespecbz7
tot_spec9=avespecbr9+avespecbt9+avespecbz9
plt.figure(1)    
dlog=np.log10(tot_spec7[20:400])
flog=np.log10(f[20:400])
A1=np.array([flog,np.ones(len(flog))])
w1=np.linalg.lstsq(A1.T,dlog)[0]
slope=np.round(w1[0],3)
func = f**(slope)
scale_data=tot_spec7[20]
scale_fit=func[20]
ratio=scale_data/scale_fit
plt.loglog(f,tot_spec7,'o')
plt.loglog(f[20:400],tot_spec7[20:400])
plt.loglog(f,ratio*func)
print 'Tot B7 slope is',slope
"""
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