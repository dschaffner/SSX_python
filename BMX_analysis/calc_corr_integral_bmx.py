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
import scipy.fftpack as fftp
import os
import spectrum_wwind as spec
import crossspec as cp

port_sep = 0.0254#m
delaytimes_2p2 = np.zeros([10])
delaytimes_1p8 = np.zeros([10])

shot = 15

crossphase_ave = np.zeros([11126])

nshots=0
for shot in np.arange(10,30):
    nshots+=1
    time_ms,time_s,timeB_s,Brdot7,Brdot9,Btdot7,Btdot9,Bzdot7,Bzdot9,Br7,Br9,Bt7,Bt9,Bz7,Bz9,disI,light=ldbmx.load_picoscope(shot)
    Br7_dtr = sps.detrend(Br7)
    Br9_dtr = sps.detrend(Br9)
    #ratioBr9_dtr = -Br9_dtr
    Bt7_dtr = sps.detrend(Bt7)
    Bt9_dtr = sps.detrend(Bt9)
    Bz7_dtr = sps.detrend(Bz7)
    Bz9_dtr = sps.detrend(Bz9)
    B7tot = np.sqrt(Br7_dtr**2+Bt7_dtr**2+Bz7_dtr**2)
    B9tot = np.sqrt(Br9_dtr**2+Bt9_dtr**2+Bz9_dtr**2)
    
    nper=22250#16384
    timeB_ms=timeB_s*1e6
    tindex1 = ssxf.tindex_min(timeB_ms,20.0)#20.0
    tindex2 = ssxf.tindex_min(timeB_ms,198.0)#198.0
    tindex2 = tindex1+nper
    n=timeB_s[tindex1:tindex2].shape[0]
    

    
    rando=1000*np.random.rand(n)
    
    tau,corr = gc.get_corr(timeB_s[tindex1:tindex2],Br7_dtr[tindex1:tindex2],Br9_dtr[tindex1:tindex2],normalized=False)
    #tau,corr = gc.get_corr(np.arange(22250),Br7_dtr[tindex1:tindex2],Br7_dtr[tindex1:tindex2],normalized=False)
    f,f0,compr7,pwrbr7,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Br7_dtr[tindex1:tindex2],timeB_s[tindex1:tindex2],window='None')
    f,f0,compr9,pwrbr9,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(Br9_dtr[tindex1:tindex2],timeB_s[tindex1:tindex2],window='None')
    #f,f0,compran,pwrran,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(rando,timeB_s[tindex1:tindex2],window='None')
    interval = timeB_s[1]-timeB_s[0]
    
    factor = 2.0/(n*interval)    
    cross_spec = np.conj(compr7)*compr9*factor

    crossf, cross_csd = sps.csd(Br7_dtr[tindex1:tindex2],Br9_dtr[tindex1:tindex2],fs=(1.0/interval),nperseg=nper)
    #crossf2, cross_csd2 = sps.csd(Br7_dtr[tindex1:tindex2],Br9_dtr[tindex1:tindex2],fs=(1.0/interval),nperseg=11125)
    
    crossf_auto, autobr7_csd = sps.csd(Br7_dtr[tindex1:tindex2],Br7_dtr[tindex1:tindex2],fs=(1.0/interval),nperseg=nper)#scaling='density')
    crossf_auto, autobr9_csd = sps.csd(Br9_dtr[tindex1:tindex2],Br9_dtr[tindex1:tindex2],fs=(1.0/interval),nperseg=nper)#scaling='spectrum')
    autobr7 = np.conj(compr7)*compr7*factor
    autobr7_re = np.real(autobr7)
    cross_phase = np.angle(cross_spec[0:nper/2+1])
    cross_phase_csd = np.angle(cross_csd)
    cross_coh = ((np.abs(cross_spec[0:nper/2+1]))**2)/(factor*factor*pwrbr7*pwrbr9)
    crossf, coh_csd = sps.coherence(Br7_dtr[tindex1:tindex2],Br9_dtr[tindex1:tindex2],fs=(1.0/interval),nperseg=nper)#,nperseg=11126)
    spec_corrtot = 0
    for ff in np.arange(len(crossf)):
        spec_corrtot=spec_corrtot+(np.sqrt(coh_csd[ff])
                                    *np.cos(cross_phase_csd[ff])
                                    *np.sqrt(autobr7_csd[ff])
                                    *np.sqrt(autobr9_csd[ff])
                                    *crossf[1])
    corr_csd=sps.correlate(Br7_dtr[tindex1:tindex2],Br9_dtr[tindex1:tindex2],'full',)
    
    spec_corrtot2 = 0
    for ff in np.arange(0,len(f)):
        spec_corrtot2=spec_corrtot2+(np.sqrt(cross_coh[ff])
                                     *np.cos(cross_phase[ff])
                                     *np.sqrt(factor*pwrbr7[ff])
                                     *np.sqrt(factor*pwrbr9[ff])
                                     *f[1])
    spec_corrtot_bw = 0
    #for ff in np.arange(100,400):
    for ff in np.arange(0,100):
        spec_corrtot_bw=spec_corrtot_bw+(np.sqrt(cross_coh[ff])
                                     *np.cos(cross_phase[ff])
                                     *np.sqrt(factor*pwrbr7[ff])
                                     *np.sqrt(factor*pwrbr9[ff])
                                     *f[1])
        
    dircorr = np.sum(Br7_dtr[tindex1:tindex2]*Br9_dtr[tindex1:tindex2])/n
    dircorr1 = np.sum(Br7_dtr[tindex1:tindex2]*Br7_dtr[tindex1:tindex2])/n
    dircorr2 = np.sum(Br9_dtr[tindex1:tindex2]*Br9_dtr[tindex1:tindex2])/n
    ratio = dircorr/np.max(corr)
    #print ratio
    ratio2 = dircorr/spec_corrtot2
    #print ratio2
    #print dircorr
    print np.max(corr)/n
    #print np.max(corr_csd)/n
    print np.abs(spec_corrtot)
    print '   '
    print dircorr
    print spec_corrtot2
    #print spec_corrtot_bw

    
    cpf,cross_phase_func = cp.crossphase(Br7_dtr[tindex1:tindex2],Br9_dtr[tindex1:tindex2],timeB_s[tindex1:tindex2])
    #spec_corrtot_func = cp.crossspec_corr(Br7_dtr[tindex1:tindex2],Br9_dtr[tindex1:tindex2],timeB_s[tindex1:tindex2],0,62.5e6)
    #print spec_corrtot_func
    crossphase_ave = crossphase_ave+cross_phase_func
    
    
    spec_corrtot_norm = cp.crossspec_corr_norm(Bt7_dtr[tindex1:tindex2],Bz7_dtr[tindex1:tindex2],timeB_s[tindex1:tindex2],0,62.5e6)
    spec_corrtot_norm_bw = cp.crossspec_corr_norm(Bt7_dtr[tindex1:tindex2],Bz7_dtr[tindex1:tindex2],timeB_s[tindex1:tindex2],6e5,2e6)
    
    print ''
    norm_corr = dircorr/np.sqrt(dircorr1*dircorr2)
    print norm_corr
    print spec_corrtot_norm
    print spec_corrtot_norm_bw
    
crossphase_ave = crossphase_ave/float(nshots)
    #x=np.arange(1000)*0.01
#sin1 = np.sin(2*np.pi*x)
#sin2 = np.sin((2*np.pi*x)+np.pi)
#sin3 = np.sin((2*np.pi*x)+2)
#sin4 = sin1+sin3
#sin5 = sin3*sin4
#sin6 = np.sin((10*np.pi*x)+20)
#sin7 = sin1+sin6
#tau,corr_s = gc.get_corr(x,sin5,sin7,normalized=False)
#dir_corr_s = np.sum(sin5*sin7)
"""
fs = 10e3
n = 1e5
amp = 20
freq = 1234.0
noise_power = 0.001 * fs / 2
time = np.arange(n) / fs
b, a = sps.butter(2, 0.25, 'low')
x = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
y = sps.lfilter(b, a, x)
x += amp*np.sin(2*np.pi*freq*time)
y += np.random.normal(scale=0.1*np.sqrt(noise_power), size=time.shape)

nper=50000
f3, Pxy = sps.csd(x,y,fs,nperseg=nper)
f3, Px = sps.csd(x,x,fs,nperseg=nper)
f3, Py = sps.csd(y,y,fs,nperseg=nper)
f2, Cxy = sps.coherence(x, y,fs,nperseg=nper)
fx,f0,compx,pwrx,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(x,time,window='None')
fy,f0,compy,pwry,mag1,phase1,cos_phase1,interval=spec.spectrum_wwind(y,time,window='None')
factor = 2.0/(n*interval)    
cross_spec = np.conj(compx)*compy*factor
auto_x = np.conj(compx)*compx*factor
auto_y = np.conj(compy)*compy*factor
cross_phase = np.angle(cross_spec[0:50001])
cross_phase_csd = np.angle(Pxy)
cross_coh = ((np.abs(cross_spec[0:50001]))**2)/(np.abs(auto_x[:50001])*np.abs(auto_y[:50001]))
cxytest=np.abs(Pxy)**2/(Px*Py)

cwrx2=fftp.fft(x)
cwry2=fftp.fft(y)
ff=fftp.fftfreq(int(n),interval)
cross2 = np.conj(cwrx2)*cwry2
"""
"""

plt.figure(10)
plt.semilogy(f2, Cxy)
plt.xlabel('frequency [Hz]')
plt.ylabel('Coherence')

"""
"""

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