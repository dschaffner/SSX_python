# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:03:09 2018

@author: dschaffner
"""

import load_picoscope_bmx_102918 as ldbmx
import numpy as np
import ssx_functions as ssxf
import matplotlib.pylab as plt
import get_corr as gc
import scipy.signal as sps
import os
import spectrum_wwind as spec

port_sep = 0.0254#m
delaytimes_2p2 = np.zeros([10])
delaytimes_1p8 = np.zeros([10])


for shot in np.arange(10,30):
    time_ms,time_s,timeB_s,Brdot7,Brdot9,Btdot7,Btdot9,Bzdot7,Bzdot9,Br7,Br9,Bt7,Bt9,Bz7,Bz9,disI,light=ldbmx.load_picoscope(shot)
    Br7_dtr = sps.detrend(Br7)
    Br9_dtr = sps.detrend(Br9)
    Bt7_dtr = sps.detrend(Bt7)
    Bt9_dtr = sps.detrend(Bt9)
    Bz7_dtr = sps.detrend(Bz7)
    Bz9_dtr = sps.detrend(Bz9)
    B7tot = np.sqrt(Br7_dtr**2+Bt7_dtr**2+Bz7_dtr**2)
    B9tot = np.sqrt(Br9_dtr**2+Bt9_dtr**2+Bz9_dtr**2)
    
    timeB_ms=timeB_s*1e6
    tindex1 = ssxf.tindex_min(timeB_ms,20.0)
    tindex2 = ssxf.tindex_min(timeB_ms,198.0)
    tau,corr = gc.get_corr(timeB_s[tindex1:tindex2],B7tot[tindex1:tindex2],B9tot[tindex1:tindex2],normalized=False)

    #if shot==16: plt.plot(tau,corr)
    #print np.argmax(11000+corr[11000:11250])
    delay=tau[11000+np.argmax(corr[11000:11250])]
    #print delay,' sec'
    if shot<20: delaytimes_2p2[shot-10]=delay
    if shot>=20: delaytimes_1p8[shot-20]=delay
    
vels_2p2 = (port_sep/delaytimes_2p2)/1000.0#km/s
vels_1p8 = (port_sep/delaytimes_1p8)/1000.0#km/s
vels_tot = np.concatenate((vels_1p8,vels_2p2),axis=None)



plt.rc('axes',linewidth=2.0)
plt.rc('xtick.major',width=2.0)
plt.rc('ytick.major',width=2.0)
plt.rc('xtick.minor',width=2.0)
plt.rc('ytick.minor',width=2.0)
plt.rc('lines',markersize=8,markeredgewidth=0.0,linewidth=1.0)

fig=plt.figure(num=1,figsize=(10,8),dpi=600,facecolor='w',edgecolor='k')
left  = 0.12  # the left side of the subplots of the figure
right = 0.97    # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.25   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

ax1=plt.subplot(1,1,1)
plt.hist(abs(vels_tot), bins=10)  # arguments are passed to np.histogram
plt.xticks(fontsize=12)
plt.xlabel(r'Bulk Vel. [km/s]',fontsize=16)
#plt.xlim(0,198)
plt.yticks(np.array([0,1,2,3,4,5]),[0,1,2,3,4,5],fontsize=12)
plt.ylabel('Count',fontsize=16)
plt.vlines(66.26,0,5,color='red',linestyle='dashed')
plt.xlim(50,82)
plt.ylim(0,5)
plt.text(0.07,0.92,r'Mean: 66$\pm$8km/s',transform=ax1.transAxes,fontsize=16)
#plt.ylim(0,3000)
#leg=plt.legend(loc='upper right',fontsize=12,frameon=False,handlelength=5)


savefilename='Vdist_fromBtotall_forAPSDPP18.png'
#savefilename='Discharge_current_'+date+'_shot'+str(shot)+'_forAPSDPP18.png'
savefile = os.path.normpath(savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()
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