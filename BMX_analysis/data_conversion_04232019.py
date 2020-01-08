#data_conversion_04232019.py

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:34:02 2019

@author: dschaffner
"""

import load_picoscope_bmx_04232019 as load
import numpy as np

#time_ms,time_s,timeB_s,timeB_ms,Bdot1,Bdot2,Bdot3,Bdot4,B1,B2,B3,B4,B1filt,B2filt,B3filt,B4filt
data = load.load_picoscope(1)

shotnum = 17
time_ms,time_s,timeB_s,timeB_ms,timeraw=data[0],data[1],data[2],data[3],data[20]
datalength=data[0].shape[0]
datalengthraw=data[20].shape[0]

Bdot1raw=np.zeros([shotnum,datalengthraw])
Bdot2raw=np.zeros([shotnum,datalengthraw])
Bdot3raw=np.zeros([shotnum,datalengthraw])
Bdot4raw=np.zeros([shotnum,datalengthraw])
Bdot1=np.zeros([shotnum,datalength])
Bdot2=np.zeros([shotnum,datalength])
Bdot3=np.zeros([shotnum,datalength])
Bdot4=np.zeros([shotnum,datalength])
B1=np.zeros([shotnum,datalength-1])
B2=np.zeros([shotnum,datalength-1])
B3=np.zeros([shotnum,datalength-1])
B4=np.zeros([shotnum,datalength-1])

#Picoscope1
for shot in np.arange(1,shotnum+1):
    data=load.load_picoscope(shot,scopenum=1)
    Bdot1raw[shot-1,:]=data[16]
    Bdot2raw[shot-1,:]=data[17]
    Bdot3raw[shot-1,:]=data[18]
    Bdot4raw[shot-1,:]=data[19]
    Bdot1[shot-1,:]=data[4]
    Bdot2[shot-1,:]=data[5]
    Bdot3[shot-1,:]=data[6]
    Bdot4[shot-1,:]=data[7]
    B1[shot-1,:]=data[8]
    B2[shot-1,:]=data[9]
    B3[shot-1,:]=data[10]
    B4[shot-1,:]=data[11]

proberaw=np.zeros(1,dtype=[('theta',np.float64,(shotnum,datalengthraw)),('z',np.float64,(shotnum,datalengthraw))])
proberaw['theta'][0]=Bdot1raw
proberaw['z'][0]=Bdot2raw
probebdot=np.zeros(1,dtype=[('theta',np.float64,(shotnum,datalengthraw)),('z',np.float64,(shotnum,datalengthraw))])
probebdot['theta'][0]=Bdot1
probebdot['z'][0]=Bdot2
probeB=np.zeros(1,dtype=[('theta',np.float64,(shotnum,datalength-1)),('z',np.float64,(shotnum,datalength-1))])
probeB['theta'][0]=B1
probeB['z'][0]=B2

pos1={'raw':proberaw,'bdot':probebdot,'b':probeB}

proberaw['theta'][0]=Bdot3raw
proberaw['z'][0]=Bdot4raw
probebdot['theta'][0]=Bdot3
probebdot['z'][0]=Bdot4
probeB['theta'][0]=B3
probeB['z'][0]=B4
pos3={'raw':proberaw,'bdot':probebdot,'b':probeB}

#Picoscope2
for shot in np.arange(1,shotnum+1):
    data=load.load_picoscope(shot,scopenum=2)
    Bdot1raw[shot-1,:]=data[16]
    Bdot2raw[shot-1,:]=data[17]
    Bdot3raw[shot-1,:]=data[18]
    Bdot4raw[shot-1,:]=data[19]
    Bdot1[shot-1,:]=data[4]
    Bdot2[shot-1,:]=data[5]
    Bdot3[shot-1,:]=data[6]
    Bdot4[shot-1,:]=data[7]
    B1[shot-1,:]=data[8]
    B2[shot-1,:]=data[9]
    B3[shot-1,:]=data[10]
    B4[shot-1,:]=data[11]
    
proberaw['theta'][0]=Bdot1raw
proberaw['z'][0]=Bdot2raw
probebdot['theta'][0]=Bdot1
probebdot['z'][0]=Bdot2
probeB['theta'][0]=B1
probeB['z'][0]=B2
pos5={'raw':proberaw,'bdot':probebdot,'b':probeB}

proberaw['theta'][0]=Bdot3raw
proberaw['z'][0]=Bdot4raw
probebdot['theta'][0]=Bdot3
probebdot['z'][0]=Bdot4
probeB['theta'][0]=B3
probeB['z'][0]=B4
pos7={'raw':proberaw,'bdot':probebdot,'b':probeB}

#Picoscope3
for shot in np.arange(1,shotnum+1):
    data=load.load_picoscope(shot,scopenum=3)
    Bdot1raw[shot-1,:]=data[16]
    Bdot2raw[shot-1,:]=data[17]
    Bdot3raw[shot-1,:]=data[18]
    Bdot4raw[shot-1,:]=data[19]
    Bdot1[shot-1,:]=data[4]
    Bdot2[shot-1,:]=data[5]
    Bdot3[shot-1,:]=data[6]
    Bdot4[shot-1,:]=data[7]
    B1[shot-1,:]=data[8]
    B2[shot-1,:]=data[9]
    B3[shot-1,:]=data[10]
    B4[shot-1,:]=data[11]
    
proberaw['theta'][0]=Bdot1raw
proberaw['z'][0]=Bdot2raw
probebdot['theta'][0]=Bdot1
probebdot['z'][0]=Bdot2
probeB['theta'][0]=B1
probeB['z'][0]=B2
pos9={'raw':proberaw,'bdot':probebdot,'b':probeB}

proberaw['theta'][0]=Bdot3raw
proberaw['z'][0]=Bdot4raw
probebdot['theta'][0]=Bdot3
probebdot['z'][0]=Bdot4
probeB['theta'][0]=B3
probeB['z'][0]=B4
pos11={'raw':proberaw,'bdot':probebdot,'b':probeB}

#Picoscope4
for shot in np.arange(1,shotnum+1):
    data=load.load_picoscope(shot,scopenum=4)
    Bdot1raw[shot-1,:]=data[16]
    Bdot2raw[shot-1,:]=data[17]
    Bdot3raw[shot-1,:]=data[18]
    Bdot4raw[shot-1,:]=data[19]
    Bdot1[shot-1,:]=data[4]
    Bdot2[shot-1,:]=data[5]
    Bdot3[shot-1,:]=data[6]
    Bdot4[shot-1,:]=data[7]
    B1[shot-1,:]=data[8]
    B2[shot-1,:]=data[9]
    B3[shot-1,:]=data[10]
    B4[shot-1,:]=data[11]
    
proberaw['theta'][0]=Bdot1raw
proberaw['z'][0]=Bdot2raw
probebdot['theta'][0]=Bdot1
probebdot['z'][0]=Bdot2
probeB['theta'][0]=B1
probeB['z'][0]=B2
pos13={'raw':proberaw,'bdot':probebdot,'b':probeB}



filepath='C:\\Users\\dschaffner\\Dropbox\\Data\\BMPL\\BMX\\2019\\Correlation Campaign\\Encoding Converted for PC\\04232019\\processed\\'
filename='2kV_oddpos1to13_2ms_stuffdelay_17shots_04232019'
np.savez(filepath+filename,pos1=pos1,pos3=pos3,pos5=pos5,pos7=pos7,pos9=pos9,pos11=pos11,pos13=pos13,allow_pickle=True)

import h5py

f = h5py.File(filename+'.hdf5', 'a')
pos1=f.create_dataset("pos1",)
grp = f.create_group("pos1")




#probe1raw=np.dtype([('theta',np.float64,(shotnum,datalength)),('z',np.float64,(shotnum,datalength))])

#p1rawz=np.dtype([('direction','U10'),('theta',np.float64,np.zeros([datalength]))])
#p1rawt=np.dtype([('direction','U10'),('z',np.float64,np.zeros([datalength]))])

#p1raw=np.zeros(2,dtype_)
#probe1raw=np.zeros(datalength,dtype={'channels':{B}})

#probe1raw=np.array([('theta',Bdot1raw),('z',Bdot2raw)],dtype=[('name','U10'),('data',np.float64)])
#probe1bdot=np.array([('theta',Bdot1),('z',Bdot2)],dtype=[('name','U10'),('data',np.float64)])
#probe1B=np.array([('theta',B1),('z',B2)],dtype=[('name','U10'),('data',np.float64)])
#probe1=np.array([('raw',probe1raw),('bdot',probe1bdot),('B',probe1B)])