#load picoscope

import numpy as np
import scipy.integrate as sp
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pylab as plt
import ssx_functions as ssxf

#data = '061615'
#shot = 1
#time_range = [20.0,80.0] #in us

def load_picoscope(shot_number,maxrange=5,time_range=[-2.0,198.0],location='',plot=False):
    
    probe_dia = 0.003175#m (1/8'' probe)
    probe_dia = 0.00158755#m (1/16'' probe)
    hole_sep = 0.001016#m (1/16''probe)
    r_probe_area = np.pi*(probe_dia/2)**2
    tz_probe_area = probe_dia*hole_sep
    startintg_index=1800
    meancutoff = 400
    #load file
    location = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\BM2X\\Data Storage\\102918\\'
    filename = '20181029_bdr_bdt_shot ('
    print location+filename+str(shot_number)+').txt'
    data = np.loadtxt(location+filename+str(shot_number)+').txt',skiprows=2,unpack=True)

    data=data[:,startintg_index:]
    
    time_ms = data[0,:]
    time_s = time_ms*1e-6
    timeB_s = time_s[1:]
        
    Brdot7 = data[1,:]-np.mean(data[1,0:meancutoff])
    neginfs = np.isneginf(Brdot7)
    Brdot7[np.where(neginfs)] = -maxrange
    posinfs = np.isinf(Brdot7)
    Brdot7[np.where(posinfs)] = maxrange
    
    Brdot9 = data[2,:]-np.mean(data[2,0:meancutoff])
    neginfs = np.isneginf(Brdot9)
    Brdot9[np.where(neginfs)] = -maxrange
    posinfs = np.isinf(Brdot9)
    Brdot9[np.where(posinfs)] = maxrange

    Btdot7 = data[3,:]-np.mean(data[3,0:meancutoff])
    neginfs = np.isneginf(Btdot7)
    Btdot7[np.where(neginfs)] = -maxrange
    posinfs = np.isinf(Btdot7)
    Btdot7[np.where(posinfs)] = maxrange
           
    Btdot9 = data[4,:]-np.mean(data[4,0:meancutoff])
    neginfs = np.isneginf(Btdot9)
    Btdot9[np.where(neginfs)] = -maxrange
    posinfs = np.isinf(Btdot9)
    Btdot9[np.where(posinfs)] = maxrange
    
    filename = '20181029_bdz_disI_light_shot ('
    data = np.loadtxt(location+filename+str(shot_number)+').txt',skiprows=2,unpack=True)

    data=data[:,startintg_index:]
    
    time_ms = data[0,:]
    time_s = time_ms*1e-6
    timeB_s = time_s[1:]
    
    Bzdot7 = data[1,:]-np.mean(data[1,0:meancutoff])
    neginfs = np.isneginf(Bzdot7)
    Bzdot7[np.where(neginfs)] = -maxrange
    posinfs = np.isinf(Bzdot7)
    Bzdot7[np.where(posinfs)] = maxrange
           
    Bzdot9 = data[2,:]-np.mean(data[2,0:meancutoff])
    neginfs = np.isneginf(Bzdot9)
    Bzdot9[np.where(neginfs)] = -maxrange
    posinfs = np.isinf(Bzdot9)
    Bzdot9[np.where(posinfs)] = maxrange
           
    disI = data[3,:]*10000.0*2#Amps
    light = data[4,:]
    
    Br7 = sp.cumtrapz(Brdot7/r_probe_area,time_s)*1e4#Gauss
    Br9 = sp.cumtrapz(Brdot9/r_probe_area,time_s)*1e4#Gauss
    Bt7 = 3.162*sp.cumtrapz(Btdot7/tz_probe_area,time_s)*1e4#Gauss
    Bt9 = 3.162*sp.cumtrapz(Btdot9/tz_probe_area,time_s)*1e4#Gauss
    Bz7 = sp.cumtrapz(Bzdot7/tz_probe_area,time_s)*1e4#Gauss
    Bz9 = sp.cumtrapz(Bzdot9/tz_probe_area,time_s)*1e4#Gauss
    #filtering
    #def butter_highpass(cutoff, fs, order=5):
    #    nyq = 0.5 * fs
    #    normal_cutoff = cutoff / nyq
    #    b, a = signal.butter(order, normal_cutoff, btype='highpass', analog=False)
    #    return b, a

    #def butter_highpass_filter(data, cutoff, fs, order=5):
    #    b, a = butter_highpass(cutoff, fs, order=order)
    #    y = signal.filtfilt(b, a, data)
    #    return y

    #fps = 30
    #sine_fq = 10 #Hz
    #duration = 10 #seconds
    #sine_5Hz = sine_generator(fps,sine_fq,duration)
    #sine_fq = 1 #Hz
    #duration = 10 #seconds
    #sine_1Hz = sine_generator(fps,sine_fq,duration)

    #sine = sine_5Hz + sine_1Hz

    #filtered_sine = butter_highpass_filter(sine.data,10,fps)
          
    
    #Integration and Calibration    
    #Bx =sp.cumtrapz(Bxdot/probe_area,time_s)
    #Bx = 3.162*Bx/1.192485591065652224e-03
        
    #By =sp.cumtrapz(Bydot/probe_area,time_s)
    #By = 3.162*By/1.784763055992550198e-03
        
    #Bz =sp.cumtrapz(Bzdot/probe_area,time_s)
    #Bz = 3.162*Bz/1.297485014039849059e-03
    #meanBx = np.mean(Bx)
    
    #Bxfilt = butter_highpass_filter(Bx,1e4,125e6,order=3)  
    #Byfilt = butter_highpass_filter(By,1e4,125e6,order=3)  
    #Bzfilt = butter_highpass_filter(Bz,1e4,125e6,order=3)  
    #Btot = np.sqrt(Bxfilt**2+Byfilt**2+Bzfilt**2)
    #Btotave=Btotave+Btot
    
    #if plot:
    #    plt.figure(1)
    #    plt.plot(time,data[1,:])
    #    plt.figure(2)
    #    plt.plot(time[1:],Btot)
        
    return time_ms,time_s,timeB_s,Brdot7,Brdot9,Btdot7,Btdot9,Bzdot7,Bzdot9,Br7,Br9,Bt7,Bt9,Bz7,Bz9,disI,light