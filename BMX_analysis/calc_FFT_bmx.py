# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 10:03:09 2018

@author: dschaffner
"""

import load_picoscope_bmx_04232019 as ldbmx
import numpy as np
import spectrum_wwind as spec
import ssx_functions as ssxf
import matplotlib.pylab as plt
#import MLE as mle
#import os

date='20190423'

shot_range=[5,19]#102218
shot_range=[20,26]#102218
shot_range=[10,19]#102918
shot_range=[1,17]#04232019

time_range=[38,198]#fsize=10001
time_range=[20,198]#fsize=11126
time_range=[50,150]
#time_range=[50,100]#fsize=3126
fsize=6251
avespecb1=np.zeros([fsize])
avespecb2=np.zeros([fsize])
avespecb3=np.zeros([fsize])
avespecb4=np.zeros([fsize])

#avespecbr7=np.zeros([fsize])
#avespecbr9=np.zeros([fsize])
#avespecbt7=np.zeros([fsize])
#avespecbt9=np.zeros([fsize])
#avespecbz7=np.zeros([fsize])
#avespecbz9=np.zeros([fsize])

for shot in np.arange(shot_range[0],shot_range[1]+1):
    time_ms,time_s,timeB_s,timeB_ms,Bdot1,Bdot2,Bdot3,Bdot4,B1,B2,B3,B4,B1filt,B2filt,B3filt,B4filt=ldbmx.load_picoscope(shot,scopenum=3)
    #time_s=time*1e-6
    tindex1 = ssxf.tindex_min(time_ms,time_range[0])
    tindex2 = ssxf.tindex_min(time_ms,time_range[1])
    f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Bdot1[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
    avespecb1=avespecb1+(pwrbxdot/(f*f))
    f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Bdot2[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
    avespecb2=avespecb2+(pwrbxdot/(f*f))
    f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Bdot3[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
    avespecb3=avespecb3+(pwrbxdot/(f*f))
    f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Bdot4[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
    avespecb4=avespecb4+(pwrbxdot/(f*f))
    
    
    #f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Brdot9[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
    #avespecbr9=avespecbr9+(pwrbxdot/(f*f))
    #f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Btdot7[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
    #avespecbt7=avespecbt7+(pwrbxdot/(f*f))
    #f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Btdot9[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
    #avespecbt9=avespecbt9+(pwrbxdot/(f*f))
    #f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Bzdot7[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
    #avespecbz7=avespecbz7+(pwrbxdot/(f*f))
    #f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Bzdot9[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
    #avespecbz9=avespecbz9+(pwrbxdot/(f*f))

tot1=avespecb1+avespecb2
tot2=avespecb3+avespecb4
#tot_spec7_1p8=avespecbr7+avespecbt7+avespecbz7
#tot_spec9_1p8=avespecbr9+avespecbt9+avespecbz9

#shot_range=[20,29]#102918
#avespecbr7=np.zeros([fsize])
#avespecbr9=np.zeros([fsize])
#avespecbt7=np.zeros([fsize])
#avespecbt9=np.zeros([fsize])
#avespecbz7=np.zeros([fsize])
#avespecbz9=np.zeros([fsize])
#for shot in np.arange(shot_range[0],shot_range[1]+1):
#    time_ms,time_s,timeB_s,Brdot7,Brdot9,Btdot7,Btdot9,Bzdot7,Bzdot9,Br7,Br9,Bt7,Bt9,Bz7,Bz9,disI,light=ldbmx.load_picoscope(shot)
#    #time_s=time*1e-6
#    tindex1 = ssxf.tindex_min(time_ms,time_range[0])
#    tindex2 = ssxf.tindex_min(time_ms,time_range[1])
#    f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Brdot7[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
#    avespecbr7=avespecbr7+(pwrbxdot/(f*f))
#    f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Brdot9[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
#    avespecbr9=avespecbr9+(pwrbxdot/(f*f))
#    f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Btdot7[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
#    avespecbt7=avespecbt7+(pwrbxdot/(f*f))
#    f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Btdot9[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
#    avespecbt9=avespecbt9+(pwrbxdot/(f*f))
#    f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Bzdot7[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
#    avespecbz7=avespecbz7+(pwrbxdot/(f*f))
#    f,f0,comp1,pwrbxdot,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Bzdot9[tindex1:tindex2],time_s[tindex1:tindex2],window='hanning')
#    avespecbz9=avespecbz9+(pwrbxdot/(f*f))
#    
#tot_spec7_2p2=avespecbr7+avespecbt7+avespecbz7
#tot_spec9_2p2=avespecbr9+avespecbt9+avespecbz9

plt.rc('axes',linewidth=2.0)
plt.rc('xtick.major',width=2.0)
plt.rc('ytick.major',width=2.0)
plt.rc('xtick.minor',width=2.0)
plt.rc('ytick.minor',width=2.0)
plt.rc('lines',markersize=2.0,markeredgewidth=0.0,linewidth=1.0)

fig=plt.figure(num=1,figsize=(4,3),dpi=300,facecolor='w',edgecolor='k')
left  = 0.15  # the left side of the subplots of the figure
right = 0.97    # the right side of the subplots of the figure
bottom = 0.1  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.25   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

ax1=plt.subplot(1,1,1)

dlog=np.log10(tot2[10:200])
flog=np.log10(f[10:200])
A1=np.array([flog,np.ones(len(flog))])
w1=np.linalg.lstsq(A1.T,dlog)[0]
slope=np.round(w1[0],3)
func = f**(slope)
scale_data=avespecb1[20]
scale_fit=func[20]
ratio=scale_data/scale_fit

plt.loglog(f,tot2,color='blue',label=r'$\langle|\widetilde{B}|\rangle$ 1.8kV Spec.Index='+str(np.round(slope,1)))
#plt.loglog(f[20:400],tot_spec7[20:400],)
plt.loglog(f[10:200],3*ratio*func[10:200],color='blue',linestyle='dashed')
print ('slope is',slope)
"""
dlog=np.log10(tot_spec7_2p2[20:400])
flog=np.log10(f[20:400])
A1=np.array([flog,np.ones(len(flog))])
w1=np.linalg.lstsq(A1.T,dlog)[0]
slope=np.round(w1[0],3)
func = f**(slope)
scale_data=tot_spec7_2p2[20]
scale_fit=func[20]
ratio=scale_data/scale_fit

plt.loglog(f,tot_spec7_2p2,'o',color='red',label=r'$\langle|\widetilde{B}|\rangle$ 2.2kV Spec.Index='+str(np.round(slope,1)))
#plt.loglog(f[20:400],tot_spec7[20:400],)
plt.loglog(f[20:400],2e-1*ratio*func[20:400],color='red',linestyle='dashed')
plt.loglog(f[10:600],1e-2*ratio*f[10:600]**(-5./3),color='orange',linestyle='solid',label='Kolmogorov -5/3 (-1.67)')
print 'Tot B7_1p8 slope is',slope

plt.xticks(fontsize=16)
plt.xlabel(r'$f$ [Hz]',fontsize=18)
plt.yticks(fontsize=16)
plt.ylabel('Power (arb)',fontsize=18)

plt.xlim(1e4,1e7)
plt.ylim(1e-27,1e-19)
leg=plt.legend(loc='lower left',fontsize=14,frameon=False,handlelength=5,numpoints=5)

savefilename='Btot_spectra_compare_forAPSDPP18.png'
#savefilename='Discharge_current_'+date+'_shot'+str(shot)+'_forAPSDPP18.png'
savefile = os.path.normpath(savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()
"""
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