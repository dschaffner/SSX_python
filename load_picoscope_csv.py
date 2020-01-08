#load picoscope

import numpy as np
import scipy.integrate as sp
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pylab as plt


#data = '061615'
#shot = 1
#time_range = [20.0,80.0] #in us

def load_picoscope_csv(shot_number,date='',location='',plot=False,maxrange=5):
    
    
    probe_dia = 0.003175#m (1/8'' probe)
    #probe_dia = 0.00158755#m (1/16'' probe)
    hole_sep = 0.001016#m (1/16''probe)
    r_probe_area = np.pi*(probe_dia/2)**2
    tz_probe_area = probe_dia*hole_sep
    startintg_index=0#3000
    meancutoff = 300
    #load file
    #location = 'C:\\Users\\dschaffner\\Dropbox\\Data\\BMPL\\BMX\\2019\\Correlation Campaign\\Encoding Converted for PC\\062820
    #19\\'
    filename = date+'-00'
    print(location+filename+str(shot_number)+'.csv')
    data = np.genfromtxt(location+filename+str(shot_number)+'.csv',delimiter=',',skip_header=2)

    #return data
    dataraw=data
    Bdotraw1=dataraw[:,1]
    Bdotraw2=dataraw[:,2]
    Bdotraw3=dataraw[:,3]
    isatraw=dataraw[:,4]
    data=data[:,startintg_index:]
    
    time_ms = data[:,0]
    time_s = time_ms*1e-6
    timeB_s = time_s[1:]
    timeB_ms = time_ms[1:]
    timeraw = dataraw[:,0]
    
   

    Bdot1 = Bdotraw1-np.mean(Bdotraw1[0:meancutoff])
    neginfs = np.isneginf(Bdot1)
    Bdot1[np.where(neginfs)] = -maxrange
    posinfs = np.isinf(Bdot1)
    Bdot1[np.where(posinfs)] = maxrange
    
    Bdot2 = Bdotraw2-np.mean(Bdotraw2[0:meancutoff])
    neginfs = np.isneginf(Bdot2)
    Bdot2[np.where(neginfs)] = -maxrange
    posinfs = np.isinf(Bdot2)
    Bdot2[np.where(posinfs)] = maxrange
    
    Bdot3 = Bdotraw3-np.mean(Bdotraw3[0:meancutoff])
    neginfs = np.isneginf(Bdot3)
    Bdot3[np.where(neginfs)] = -maxrange
    posinfs = np.isinf(Bdot3)
    Bdot3[np.where(posinfs)] = maxrange
    
    isat = isatraw-np.mean(isatraw[0:meancutoff])
    neginfs = np.isneginf(isat)
    isat[np.where(neginfs)] = -maxrange
    posinfs = np.isinf(isat)
    isat[np.where(posinfs)] = maxrange
    
    B1 = sp.cumtrapz(Bdot1/r_probe_area,time_s)*1e4#Gauss
    B2 = sp.cumtrapz(Bdot2/r_probe_area,time_s)*1e4#Gauss
    B3 = sp.cumtrapz(Bdot3/r_probe_area,time_s)*1e4#Gauss

    #Bt7 = 3.162*sp.cumtrapz(Btdot7/tz_probe_area,time_s)*1e4#Gauss
    #Bt9 = 3.162*sp.cumtrapz(Btdot9/tz_probe_area,time_s)*1e4#Gauss
    #Bz7 = sp.cumtrapz(Bzdot7/tz_probe_area,time_s)*1e4#Gauss
    #Bz9 = sp.cumtrapz(Bzdot9/tz_probe_area,time_s)*1e4#Gauss
    #filtering

    return time_ms,time_s,timeB_s,timeB_ms,timeraw,Bdotraw1,Bdotraw2,Bdotraw3,isatraw,Bdot1,Bdot2,Bdot3,isat,B1,B2,B3
"""
    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='highpass', analog=False)
        return b, a

    def butter_highpass_filter(data, cutoff, fs, order=5):
        b, a = butter_highpass(cutoff, fs, order=order)
        y = signal.filtfilt(b, a, data)
        return y

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
    
    B1filt = butter_highpass_filter(B1,5e4,125e6,order=3)  
    B2filt = butter_highpass_filter(B2,5e4,125e6,order=3)  
    B3filt = butter_highpass_filter(B3,5e4,125e6,order=3) 
    B4filt = butter_highpass_filter(B4,5e4,125e6,order=3)  
    #Btot = np.sqrt(Bxfilt**2+Byfilt**2+Bzfilt**2)
    #Btotave=Btotave+Btot
    
    #if plot:
    #    plt.figure(1)
    #    plt.plot(time,data[1,:])
    #    plt.figure(2)
    #    plt.plot(time[1:],Btot)
        
    """