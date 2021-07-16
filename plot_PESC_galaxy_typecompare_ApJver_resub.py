# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:00:45 2017

@author: dschaffner
"""
import numpy as np
import matplotlib.pylab as plt
from loadnpzfile import loadnpzfile
from calc_PE_SC import PE, CH, PE_dist, PE_calc_only
import Cmaxmin as cpl
from collections import Counter
from math import factorial
import matplotlib.cm as cm
import os

#datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\GalpyData_July2018\\'
datadir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Galatic Dynamics Data\\GalpyData_July2018\\resorted_data\\M6_3352_et_timescan\\'
npy='.npz'

delayindex = np.arange(1,501)#250
timestep_arr = [2050]
print(len(timestep_arr))
timeindex = (delayindex*1e5)/(1e6)

fileheader = 'PE_SC_Type_1_Rg_499_delays_3910orbits_of3910_total2000_timesteps_resorted_et'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1 = datafile['PEs']
SCs1 = datafile['SCs']
SCs1_endarray=500

fileheader = 'PE_SC_Type_2o_Rg_499_delays_5684orbits_of5684_total2000_timesteps_resorted_et'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs2o = datafile['PEs']
SCs2o = datafile['SCs']
SCs2o_endarray=500

fileheader = 'PE_SC_Type_2i_Rg_499_delays_23495orbits_of23495_total2000_timesteps_resorted_et'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs2i = datafile['PEs']
SCs2i = datafile['SCs']
SCs2i_endarray=500

fileheader = 'PE_SC_Type_31_Rg_499_delays_3279orbits_of4547_total2000_timesteps_resorted_et'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs31 = datafile['PEs']
SCs31 = datafile['SCs']
SCs31_endarray=500

fileheader = 'PE_SC_Type_32i_Rg_499_delays_3145orbits_of3899_total2000_timesteps_resorted_et'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32i = datafile['PEs']
SCs32i = datafile['SCs']
SCs32i_endarray=500

fileheader = 'PE_SC_Type_32o_Rg_499_delays_2021orbits_of2458_total2000_timesteps_resorted_et'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32o = datafile['PEs']
SCs32o = datafile['SCs']
SCs32o_endarray=500

#put in grouped orbits for variance calcuaton here
#Type 32i (RO)
fileheader = 'PE_SC_Type_32i_Rg_499_delays_170orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits1'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32ig1 = datafile['PEs']
SCs32ig1 = datafile['SCs']
SCs32i_endarrayg1=500
fileheader = 'PE_SC_Type_32i_Rg_499_delays_151orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits2'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32ig2 = datafile['PEs']
SCs32ig2 = datafile['SCs']
SCs32i_endarrayg2=500
fileheader = 'PE_SC_Type_32i_Rg_499_delays_149orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits3'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32ig3 = datafile['PEs']
SCs32ig3 = datafile['SCs']
SCs32i_endarrayg3=500
fileheader = 'PE_SC_Type_32i_Rg_499_delays_169orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits4'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32ig4 = datafile['PEs']
SCs32ig4 = datafile['SCs']
SCs32i_endarrayg4=500
fileheader = 'PE_SC_Type_32i_Rg_499_delays_176orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits5'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32ig5 = datafile['PEs']
SCs32ig5 = datafile['SCs']
SCs32i_endarrayg5=500
fileheader = 'PE_SC_Type_32i_Rg_499_delays_191orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits6'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32ig6 = datafile['PEs']
SCs32ig6 = datafile['SCs']
SCs32i_endarrayg6=500
fileheader = 'PE_SC_Type_32i_Rg_499_delays_181orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits7'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32ig7 = datafile['PEs']
SCs32ig7 = datafile['SCs']
SCs32i_endarrayg7=500
fileheader = 'PE_SC_Type_32i_Rg_499_delays_151orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits8'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32ig8 = datafile['PEs']
SCs32ig8 = datafile['SCs']
SCs32i_endarrayg8=500
fileheader = 'PE_SC_Type_32i_Rg_499_delays_127orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits9'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32ig9 = datafile['PEs']
SCs32ig9 = datafile['SCs']
SCs32i_endarrayg9=500
fileheader = 'PE_SC_Type_32i_Rg_499_delays_156orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits10'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32ig10 = datafile['PEs']
SCs32ig10 = datafile['SCs']
SCs32i_endarrayg10=500
fileheader = 'PE_SC_Type_32i_Rg_499_delays_163orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits11'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32ig11 = datafile['PEs']
SCs32ig11 = datafile['SCs']
SCs32i_endarrayg11=500
fileheader = 'PE_SC_Type_32i_Rg_499_delays_149orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits12'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32ig12 = datafile['PEs']
SCs32ig12 = datafile['SCs']
SCs32i_endarrayg12=500
fileheader = 'PE_SC_Type_32i_Rg_499_delays_151orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits13'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32ig13 = datafile['PEs']
SCs32ig13 = datafile['SCs']
SCs32i_endarrayg13=500
fileheader = 'PE_SC_Type_32i_Rg_499_delays_170orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits14'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32ig14 = datafile['PEs']
SCs32ig14 = datafile['SCs']
SCs32i_endarrayg14=500
fileheader = 'PE_SC_Type_32i_Rg_499_delays_176orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits15'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32ig15 = datafile['PEs']
SCs32ig15 = datafile['SCs']
SCs32i_endarrayg15=500
fileheader = 'PE_SC_Type_32i_Rg_499_delays_186orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits16'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32ig16 = datafile['PEs']
SCs32ig16 = datafile['SCs']
SCs32i_endarrayg16=500
fileheader = 'PE_SC_Type_32i_Rg_499_delays_186orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits17'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32ig17 = datafile['PEs']
SCs32ig17 = datafile['SCs']
SCs32i_endarrayg17=500
fileheader = 'PE_SC_Type_32i_Rg_499_delays_152orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits18'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs32ig18 = datafile['PEs']
SCs32ig18 = datafile['SCs']
SCs32i_endarrayg18=500

#Type 1 (CR)
fileheader = 'PE_SC_Type_1_Rg_499_delays_200orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits1'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1g1 = datafile['PEs']
SCs1g1 = datafile['SCs']
SCs1_endarrayg1=500
fileheader = 'PE_SC_Type_1_Rg_499_delays_200orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits2'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1g2 = datafile['PEs']
SCs1g2 = datafile['SCs']
SCs1_endarrayg2=500
fileheader = 'PE_SC_Type_1_Rg_499_delays_200orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits3'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1g3 = datafile['PEs']
SCs1g3 = datafile['SCs']
SCs1_endarrayg3=500
fileheader = 'PE_SC_Type_1_Rg_499_delays_200orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits4'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1g4 = datafile['PEs']
SCs1g4 = datafile['SCs']
SCs1_endarrayg4=500
fileheader = 'PE_SC_Type_1_Rg_499_delays_200orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits5'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1g5 = datafile['PEs']
SCs1g5 = datafile['SCs']
SCs1_endarrayg5=500
fileheader = 'PE_SC_Type_1_Rg_499_delays_200orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits6'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1g6 = datafile['PEs']
SCs1g6 = datafile['SCs']
SCs1_endarrayg6=500
fileheader = 'PE_SC_Type_1_Rg_499_delays_200orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits7'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1g7 = datafile['PEs']
SCs1g7 = datafile['SCs']
SCs1_endarrayg1=500
fileheader = 'PE_SC_Type_1_Rg_499_delays_200orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits8'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1g8 = datafile['PEs']
SCs1g8 = datafile['SCs']
SCs1_endarrayg8=500
fileheader = 'PE_SC_Type_1_Rg_499_delays_200orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits9'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1g9 = datafile['PEs']
SCs1g9 = datafile['SCs']
SCs1_endarrayg9=500
fileheader = 'PE_SC_Type_1_Rg_499_delays_200orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits10'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1g10 = datafile['PEs']
SCs1g10 = datafile['SCs']
SCs1_endarrayg10=500
fileheader = 'PE_SC_Type_1_Rg_499_delays_200orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits11'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1g11 = datafile['PEs']
SCs1g11 = datafile['SCs']
SCs1_endarrayg11=500
fileheader = 'PE_SC_Type_1_Rg_499_delays_200orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits12'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1g12 = datafile['PEs']
SCs1g12 = datafile['SCs']
SCs1_endarrayg12=500
fileheader = 'PE_SC_Type_1_Rg_499_delays_200orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits13'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1g13 = datafile['PEs']
SCs1g13 = datafile['SCs']
SCs1_endarrayg13=500
fileheader = 'PE_SC_Type_1_Rg_499_delays_200orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits14'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1g14 = datafile['PEs']
SCs1g14 = datafile['SCs']
SCs1_endarrayg14=500
fileheader = 'PE_SC_Type_1_Rg_499_delays_200orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits15'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1g15 = datafile['PEs']
SCs1g15 = datafile['SCs']
SCs1_endarrayg15=500
fileheader = 'PE_SC_Type_1_Rg_499_delays_200orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits16'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1g16 = datafile['PEs']
SCs1g16 = datafile['SCs']
SCs1_endarrayg16=500
fileheader = 'PE_SC_Type_1_Rg_499_delays_200orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits17'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1g17 = datafile['PEs']
SCs1g17 = datafile['SCs']
SCs1_endarrayg17=500
fileheader = 'PE_SC_Type_1_Rg_499_delays_200orbits_of200_total2000_timesteps_resorted_et_grouped_about200_orbits18'
datafile = loadnpzfile(datadir+fileheader+npy)
PEs1g18 = datafile['PEs']
SCs1g18 = datafile['SCs']
SCs1_endarrayg18=500



#Compute variance of different curves
C_std_RO = np.zeros([500])
H_std_RO = np.zeros([500])
C_std_CR = np.zeros([500])
H_std_CR = np.zeros([500])

numcurves=18
for d in np.arange(1,500):
    c_array = np.zeros([numcurves])
    h_array = np.zeros([numcurves])
    c_array[0]=SCs32ig1[d]
    c_array[1]=SCs32ig2[d]
    c_array[2]=SCs32ig3[d]
    c_array[3]=SCs32ig4[d]
    c_array[4]=SCs32ig5[d]
    c_array[5]=SCs32ig6[d]
    c_array[6]=SCs32ig7[d]
    c_array[7]=SCs32ig8[d]
    c_array[8]=SCs32ig9[d]
    c_array[9]=SCs32ig10[d]
    c_array[10]=SCs32ig11[d]
    c_array[11]=SCs32ig12[d]
    c_array[12]=SCs32ig13[d]
    c_array[13]=SCs32ig14[d]
    c_array[14]=SCs32ig15[d]
    c_array[15]=SCs32ig16[d]
    c_array[16]=SCs32ig17[d]
    c_array[17]=SCs32ig18[d]
    
    h_array[0]=PEs32ig1[d]
    h_array[1]=PEs32ig2[d]
    h_array[2]=PEs32ig3[d]
    h_array[3]=PEs32ig4[d]
    h_array[4]=PEs32ig5[d]
    h_array[5]=PEs32ig6[d]
    h_array[6]=PEs32ig7[d]
    h_array[7]=PEs32ig8[d]
    h_array[8]=PEs32ig9[d]
    h_array[9]=PEs32ig10[d]
    h_array[10]=PEs32ig11[d]
    h_array[11]=PEs32ig12[d]
    h_array[12]=PEs32ig13[d]
    h_array[13]=PEs32ig14[d]
    h_array[14]=PEs32ig15[d]
    h_array[15]=PEs32ig16[d]
    h_array[16]=PEs32ig17[d]
    h_array[17]=PEs32ig18[d]
    
    C_std_RO[d]=np.std(c_array)
    H_std_RO[d]=np.std(h_array)


numcurves=15 #1g6, 1g9, and 1g17 skipped
for d in np.arange(1,500):
    c_array = np.zeros([numcurves])
    h_array = np.zeros([numcurves])
    c_array[0]=SCs1g1[d]
    c_array[1]=SCs1g2[d]
    c_array[2]=SCs1g3[d]
    c_array[3]=SCs1g4[d]
    c_array[4]=SCs1g5[d]
    c_array[5]=SCs1g7[d]
    c_array[6]=SCs1g8[d]
    c_array[7]=SCs1g10[d]
    c_array[8]=SCs1g11[d]
    c_array[9]=SCs1g12[d]
    c_array[10]=SCs1g13[d]
    c_array[11]=SCs1g14[d]
    c_array[12]=SCs1g15[d]
    c_array[13]=SCs1g17[d]
    c_array[14]=SCs1g18[d]
    
    h_array[0]=PEs1g1[d]
    h_array[1]=PEs1g2[d]
    h_array[2]=PEs1g3[d]
    h_array[3]=PEs1g4[d]
    h_array[4]=PEs1g5[d]
    h_array[5]=PEs1g7[d]
    h_array[6]=PEs1g8[d]
    h_array[7]=PEs1g10[d]
    h_array[8]=PEs1g11[d]
    h_array[9]=PEs1g12[d]
    h_array[10]=PEs1g13[d]
    h_array[11]=PEs1g14[d]
    h_array[12]=PEs1g15[d]
    h_array[13]=PEs1g17[d]
    h_array[14]=PEs1g18[d]

    C_std_CR[d]=np.std(c_array)
    H_std_CR[d]=np.std(h_array)


    
M6dyn=1676
delayindex = np.arange(1,500)
timeindex=(delayindex*1e5)/(1e6)
patterntime=timeindex*4
timeindex_normM6=timeindex/(M6dyn/10)
patterntime_normM6=patterntime/(M6dyn/10)

numcolors=2
colors=np.zeros([numcolors,4])
for i in np.arange(numcolors):
    c = plt.cm.plasma(i/(float(numcolors)),1)
    colors[i,:]=c
points = ['o','v','s','p','*','h','^','D','+','>','H','d','x','<']
        
plt.rc('axes',linewidth=2.0)
plt.rc('xtick.major',width=2.0)
plt.rc('ytick.major',width=2.0)
plt.rc('xtick.minor',width=2.0)
plt.rc('ytick.minor',width=2.0)
plt.rc('lines',markersize=8,markeredgewidth=0.0,linewidth=3.0)

#plt.rcParams['ps.fonttype'] = 42
#plt.rcParams['pdf.fonttype'] = 42

fig=plt.figure(num=1,figsize=(7,9),dpi=600,facecolor='w',edgecolor='k')
left  = 0.16  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.08  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.0   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(2,1,1)
plt.plot(patterntime_normM6,SCs1[1:500],color=colors[0,:],label='Type CR')
plt.plot(patterntime_normM6,SCs1g1[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs1g2[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs1g3[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs1g4[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs1g5[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
#plt.plot(patterntime_normM6,SCs1g6[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs1g7[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs1g8[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
#plt.plot(patterntime_normM6,SCs1g9[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs1g10[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs1g11[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs1g12[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs1g13[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs1g14[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs1g15[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
#plt.plot(patterntime_normM6,SCs1g16[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs1g17[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs1g18[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
#plt.plot(patterntime_normM6,SCs1[1:500]+C_std_CR[1:],color=colors[0,:],linestyle='dashed',linewidth=1.0)
#plt.plot(patterntime_normM6,SCs1[1:500]-C_std_CR[1:],color=colors[0,:],linestyle='dashed',linewidth=1.0)
plt.fill_between(patterntime_normM6,SCs1[1:500]-C_std_CR[1:],SCs1[1:500]+C_std_CR[1:],color=colors[0,:],alpha=0.2)
#plt.plot(timeindex,SCs1_resort[1:],color='blue',linestyle='dashed')
#plt.plot(timeindex,SCsT25[1:500],color=colors[1,:],label='Type 2 [Beyond CR]')
#plt.plot(timeindex,SCs31[1:],color='green',label='Type 3-1')
plt.plot(patterntime_normM6,SCs32i[1:],color=colors[1,:],label=r'Type RO')
#plt.plot(timeindex,SCs32_resort[1:],color='red',linestyle='dashed')
#plt.plot(timeindex,SCs31[1:],color='green')
#plt.plot(timeindex,SCs31_resort[1:],color='green',linestyle='dashed')
#plt.plot(timeindex,SCs32_4CRresort[1:],color='purple')
#plt.plot(timeindex,SCs4[1:],color='purple',label='Type 4')
#plt.plot(delayindex,SCsin[1:],color='black',label='Sine Wave')
plt.plot(patterntime_normM6,SCs32ig1[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs32ig2[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs32ig3[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs32ig4[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs32ig5[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs32ig6[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs32ig7[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs32ig8[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs32ig9[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs32ig10[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs32ig11[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs32ig12[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs32ig13[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs32ig14[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs32ig15[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs32ig16[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs32ig17[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,SCs32ig18[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)

#plt.plot(patterntime_normM6,SCs32i[1:]+C_std_RO[1:],color=colors[1,:],linestyle='dashed',linewidth=1.0)
#plt.plot(patterntime_normM6,SCs32i[1:]-C_std_RO[1:],color=colors[1,:],linestyle='dashed',linewidth=1.0)
plt.fill_between(patterntime_normM6,SCs32i[1:500]-C_std_RO[1:],SCs32i[1:500]+C_std_RO[1:],color=colors[1,:],alpha=0.2)

plt.vlines(0.222,0,1,color='gray',linewidth=1.5,linestyle='dashed')

#plt.vlines(85,0,1,color='red',linestyle='dotted',linewidth=0.5)

#plt.xticks(np.array([1,20,40,60,80,100,120,140,160,180,200,220,240]),[1,20,40,60,80,100,120,140,160,180,200,220,240],fontsize=9)

plt.xticks(fontsize=18)
plt.yticks(np.array([0.10,0.20,0.30,0.40]),[0.10,0.20,0.30,0.40],fontsize=18)
plt.xlabel(r'$\tau_s/T_{dyn}$',fontsize=18)
#plt.xlabel('Delay Steps',fontsize=9)
ax1.set_xticklabels([])
plt.ylabel(r'$C$',fontsize=20)
plt.xlim(0,1.0)
plt.ylim(0.11,0.4)
plt.legend(loc='lower right',fontsize=14,frameon=False,handlelength=5,numpoints=2)
plt.text(0.04,0.95,'(a)',fontsize=16,horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes)

#savefilename='SC_galpy0718_1000timesteps_3000_orbits.png'
#savefilename='SC_galpy0718_1000timesteps_all_orbits.png'
#savefile = os.path.normpath(datadir+savefilename)
#plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')


#fig=plt.figure(num=2,figsize=(5,3.5),dpi=300,facecolor='w',edgecolor='k')
#plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(2,1,2)
plt.plot(patterntime_normM6,PEs1[1:500],color=colors[0,:],label='Type CR')
plt.plot(patterntime_normM6,PEs1g1[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs1g2[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs1g3[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs1g4[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs1g5[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
#plt.plot(patterntime_normM6,PEs1g6[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs1g7[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs1g8[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
#plt.plot(patterntime_normM6,PEs1g9[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs1g10[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs1g11[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs1g12[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs1g13[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs1g14[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs1g15[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
#plt.plot(patterntime_normM6,PEs1g16[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs1g17[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs1g18[1:500],color=colors[0,:],linestyle='dotted',linewidth=0.25)
#plt.plot(patterntime_normM6,PEs1[1:500]+H_std_CR[1:],color=colors[1,:],linestyle='dashed',linewidth=1.0)
#plt.plot(patterntime_normM6,PEs1[1:500]-H_std_CR[1:],color=colors[1,:],linestyle='dashed',linewidth=1.0)
plt.fill_between(patterntime_normM6,PEs1[1:500]-H_std_CR[1:],PEs1[1:500]+H_std_CR[1:],color=colors[0,:],alpha=0.2)
#plt.plot(timeindex,PEsT25[1:500],color=colors[1,:],label='Type 2 [Beyond CR]')
#plt.plot(timeindex,PEs31[1:],color='green',label='Type 3-1')
plt.plot(patterntime_normM6,PEs32i[1:],color=colors[1,:],label=r'Type RO')
#plt.plot(timeindex,PEs4[1:],color='purple',label='Type 4')
#plt.plot(delayindex,PEsin[1:],color='black',label='Sine Wave')
plt.plot(patterntime_normM6,PEs32ig1[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs32ig2[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs32ig3[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs32ig4[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs32ig5[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs32ig6[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs32ig7[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs32ig8[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs32ig9[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs32ig10[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs32ig11[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs32ig12[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs32ig13[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs32ig14[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs32ig15[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs32ig16[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs32ig17[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
plt.plot(patterntime_normM6,PEs32ig18[1:],color=colors[1,:],linestyle='dotted',linewidth=0.25)
#plt.plot(patterntime_normM6,PEs32i[1:]+H_std_RO[1:],color=colors[1,:],linestyle='dashed',linewidth=1.0)
#plt.plot(patterntime_normM6,PEs32i[1:]-H_std_RO[1:],color=colors[1,:],linestyle='dashed',linewidth=1.0)
plt.fill_between(patterntime_normM6,PEs32i[1:500]-H_std_RO[1:],PEs32i[1:500]+H_std_RO[1:],color=colors[1,:],alpha=0.2)

plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],[0.0,0.2,0.4,0.6,0.8,1.0],fontsize=18)
plt.xticks(fontsize=18)
plt.xlabel(r'$t_{pat}/T_{dyn}$',fontsize=20)
#ax1.set_xticklabels([])
plt.yticks(fontsize=18)
plt.ylabel(r'$H$',fontsize=20)
plt.xlim(0,1.0)
plt.ylim(0,1.1)
plt.legend(loc='lower right',fontsize=14,frameon=False,handlelength=5)
plt.text(0.04,0.95,'(b)',fontsize=16,horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes)

plt.vlines(0.222,0.0,1.1,color='gray',linewidth=1.5,linestyle='dashed')
plt.hlines(0.55124,0,1,color='gray',linewidth=1.5,linestyle='dashed')


savefilename='SC_and_PE_CR6_2000timesteps_Type1vsType32i_normdyntime_ApJver_newcolor_wtpat_withvariance.pdf'
#savefilename='SC_and_PE_CR6_2000timesteps_Type1vsType32i_normdyntime_ApJver_newcolor_wtpat.eps'
#savefilename='PE_galpy0718_1e000timesteps_all_orbits.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()

"""
ndim=5
#Plot C vs H as a CHplane
Cminx, Cminy, Cmaxx, Cmaxy = cpl.Cmaxmin(1000,ndim)
plt.rc('lines',markersize=5,markeredgewidth=0.0)

fig=plt.figure(num=33,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.25  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.97      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

timestep = 86

plt.plot(Cminx,Cminy,'k-',Cmaxx,Cmaxy,'k-')
plt.plot(PEs1[timestep],SCs1[timestep],color=colors[0,:],marker='o',label='Type 1')
plt.plot(PEsT25[timestep],SCsT25[timestep],color=colors[1,:],marker='o',label='Type 2')
plt.plot(PEs31[timestep],SCs31[timestep],color='green',marker='o',label='Type 3-1')
plt.plot(PEs32[timestep],SCs32[timestep],color='red',marker='o',label='Type 3-2')
plt.plot(PEs4[timestep],SCs4[timestep],color='purple',marker='o',label='Type 4')
#plt.plot(PEsin[81],SCsin[81],color='black',marker='o',label='Sine Wave, Delay 81')

timestep0 = 1
plt.plot(PEs1[timestep0],SCs1[timestep0],color=colors[0,:],marker='v')
plt.plot(PEsT25[timestep0],SCsT25[timestep0],color=colors[1,:],marker='v')
plt.plot(PEs31[timestep0],SCs31[timestep0],color='green',marker='v')
plt.plot(PEs32[timestep0],SCs32[timestep0],color='red',marker='v')
plt.plot(PEs4[timestep0],SCs4[timestep0],color='purple',marker='v')

plt.xlabel("Permutation Entropy", fontsize=9)
plt.ylabel("Statistical Complexity", fontsize=9)
#plt.title('Delay Timescale '+str(timeindex[timestep])+' Myr',fontsize=9)
plt.axis([0,1.0,0,0.45])
plt.xticks(np.arange(0,1.1,0.1),fontsize=9)
plt.yticks(np.arange(0,0.45,0.05),fontsize=9)
leg=plt.legend(loc='lower center',fontsize=5,frameon=False,handlelength=0)
leg.set_title(r'At Delay $\tau=$'+str(timeindex[timestep])+ 'Myr',prop={'size':4})

savefilename='CH_galpy0718_2000plustimesteps_timestep'+str(timestep)+'_3000plusorbits.eps'
#savefilename='CH_galpy0718_1000timesteps_timestep'+str(timestep)+'_all_orbits.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
"""
"""
fig=plt.figure(num=4,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.25  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.90      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


plt.plot(Cminx,Cminy,'k-',Cmaxx,Cmaxy,'k-')
plt.plot(PEs1,SCs1,color=colors[0,:],label='Type 1')
plt.plot(PEs2,SCs2,color=colors[1,:],label='Type 2')
plt.plot(PEs31,SCs31,color='green',label='Type 3-1')
plt.plot(PEs32,SCs32,color='red',label='Type 3-2')
plt.plot(PEs4,SCs4,color='purple',label='Type 4')
#plt.plot(PEsin[81],SCsin[81],color='black',marker='o',label='Sine Wave, Delay 81')
plt.xlabel("Permutation Entropy", fontsize=9)
plt.ylabel("Complexity", fontsize=9)
plt.axis([0,1.0,0,0.45])
plt.xticks(np.arange(0,1.1,0.1),fontsize=9)
plt.yticks(np.arange(0,0.45,0.05),fontsize=9)
plt.legend(loc='lower center',fontsize=5,frameon=False,handlelength=0)

savefilename='CH_galpy0718_1000timesteps_timestep_3000_orbits.png'
savefilename='CH_galpy0718_1000timesteps_timestep_all_orbits.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')
"""



"""
colors = np.zeros([5,4])
for i in np.arange(5):
    c = cm.spectral(i/5.,1)
    colors[i,:]=c
points = ['o','v','s','p','*','h','^','D','+','>','H','d','x','<']
        
plt.rc('axes',linewidth=0.75)
plt.rc('xtick.major',width=0.75)
plt.rc('ytick.major',width=0.75)
plt.rc('xtick.minor',width=0.75)
plt.rc('ytick.minor',width=0.75)
plt.rc('lines',markersize=2,markeredgewidth=0.0)

plt.rc('lines',markersize=1.5,markeredgewidth=0.0)
fig=plt.figure(num=1,figsize=(5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.15  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.2  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)


ax1=plt.subplot(1,1,1)
plt.plot(timeindex,SCs32[1:],color=colors[0,:],label='Type 32 first 3000')
plt.plot(timeindex,SCs32_2[1:],color=colors[1,:],label='Type 32 second 3000')
plt.plot(timeindex,SCs32_3[1:],color='green',label='Type 32 third 3000')
#plt.plot(delayindex,SCsin[1:],color='black',label='Sine Wave')

plt.vlines(9.19,0,1,color=colors[0,:],linestyle='dotted',linewidth=0.5)
plt.vlines(9.09,0,1,color=colors[1,:],linestyle='dotted',linewidth=0.5)
plt.vlines(9.4,0,1,color='green',linestyle='dotted',linewidth=0.5)
delayarray = np.array([1,20,40,60,80,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,110,120,140,160,180,200,220,240])
timearray = (delayarray*1e5)/(1e6)
timelist = list(timearray)
plt.xticks(timearray,timelist,fontsize=6)
plt.yticks(fontsize=9)
plt.xlabel('Myr',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Complexity',fontsize=9)
plt.xlim(8.5,10.5)
plt.ylim(0.34,0.36)
plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)

savefilename='SC_galpy0718_1000timesteps_3000_orbits_sepranges.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')
"""

"""
colors = np.zeros([5,4])
for i in np.arange(5):
    c = cm.spectral(i/5.,1)
    colors[i,:]=c
points = ['o','v','s','p','*','h','^','D','+','>','H','d','x','<']
        
plt.rc('axes',linewidth=0.75)
plt.rc('xtick.major',width=0.75)
plt.rc('ytick.major',width=0.75)
plt.rc('xtick.minor',width=0.75)
plt.rc('ytick.minor',width=0.75)
plt.rc('lines',markersize=2,markeredgewidth=0.0)

plt.rc('lines',markersize=1.5,markeredgewidth=0.0)
fig=plt.figure(num=3,figsize=(6,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.15  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.2  # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

delayindex = np.arange(1,750)
timeindex=(delayindex*1e5)/(1e6)

ax1=plt.subplot(1,1,1)
plt.plot(timeindex,SCsT21[1:],color='black',label='Type 2 (NT) [0,3.5)')
plt.plot(timeindex,SCsT22[1:],color='blue',label='Type 2 (NT) [3.5,4)')
plt.plot(timeindex,SCsT23[1:],color='green',label='Type 2 (NT) [4,4.5)')
plt.plot(timeindex,SCsT24[1:],color='red',label='Type 2 (NT) [4.5,5.5)')
plt.plot(timeindex,SCsT25[1:],color='purple',label='Type 2 (NT) [7,9]')

#delayarray = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250])
delayarray = np.array([0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500,520,540,560,580,600,620,640,660,680,700,720,740])

timearray = (delayarray*1e5)/(1e6)
timelist = list(timearray.astype(int))
plt.xticks(timearray,timelist,fontsize=8)
plt.yticks(np.array([0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40]),[0.0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40],fontsize=9)
plt.xlabel('Delay Time [Myr]',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Statistical Complexity',fontsize=9)
#plt.xlim(8.5,10.5)
#plt.ylim(0.34,0.36)
plt.legend(loc='lower right',fontsize=5,frameon=False,handlelength=5)

savefilename='SC_galpy0718_4000timesteps_Type2_diffIC.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')
"""