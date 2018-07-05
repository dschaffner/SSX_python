# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:00:45 2017

@author: dschaffner
"""
import numpy as np
import matplotlib.pylab as plt
from loadnpyfile import loadnpyfile
from calc_PE_SC import PE, CH, PE_dist, PE_calc_only
import Cmaxmin as cpl
from collections import Counter
from math import factorial
import matplotlib.cm as cm
import os

#calc_PESC_fluid.py

datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\Data040318\\'

npy='.npy'
fileheader = 'IDdatabase_Type_1_data'
type1short = loadnpyfile(datadir+fileheader+npy)
fileheader = 'IDdatabase_Type_2_data'
type2short = loadnpyfile(datadir+fileheader+npy)
fileheader = 'IDdatabase_Type_31_data'
type31short = loadnpyfile(datadir+fileheader+npy)
fileheader = 'IDdatabase_Type_32_data'
type32short = loadnpyfile(datadir+fileheader+npy)
fileheader = 'IDdatabase_Type_4_data'
type4short = loadnpyfile(datadir+fileheader+npy)




colors = np.zeros([7,4])
for i in np.arange(7):
    c = cm.spectral(i/7.,1)
    colors[i,:]=c
points = ['o','v','s','p','*','h','^','D','+','>','H','d','x','<']
        
plt.rc('axes',linewidth=0.75)
plt.rc('xtick.major',width=0.75)
plt.rc('ytick.major',width=0.75)
plt.rc('xtick.minor',width=0.75)
plt.rc('ytick.minor',width=0.75)
plt.rc('lines',markersize=2,markeredgewidth=0.0)
plt.rc('lines',markersize=1.5,markeredgewidth=0.0)

fig=plt.figure(num=1,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.15  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.15  # the bottom of the subplots of the figure
top = 0.92      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

ax1=plt.subplot(1,1,1)
plt.title('Type 1 Orbits - Short Dataset',fontsize=6)
plt.plot(type1short[1,1:],color=colors[0,:],label='Orbit 1')
plt.plot(type1short[100,1:],color=colors[1,:],label='Orbit 100')
plt.plot(type1short[200,1:],color=colors[2,:],label='Orbit 200')
plt.plot(type1short[300,1:],color=colors[3,:],label='Orbit 300')
plt.plot(type1short[500,1:],color=colors[4,:],label='Orbit 500')
plt.plot(type1short[700,1:],color=colors[5,:],label='Orbit 700')
#plt.plot(type1short[900,1:],color=colors[6,:],label='Orbit 900')

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Time Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Orbital Radius (arb)',fontsize=9)
#plt.xlim(1,250)
#plt.ylim(0,0.5)
plt.legend(loc='lower right',fontsize=2,frameon=False,handlelength=2)

savefilename='type1Short.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')




fig=plt.figure(num=2,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.15  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.15  # the bottom of the subplots of the figure
top = 0.92      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

ax1=plt.subplot(1,1,1)
plt.title('Type 2 Orbits - Short Dataset',fontsize=6)
plt.plot(type2short[1,1:],color=colors[0,:],label='Orbit 1')
plt.plot(type2short[100,1:],color=colors[1,:],label='Orbit 100')
plt.plot(type2short[200,1:],color=colors[2,:],label='Orbit 200')
plt.plot(type2short[300,1:],color=colors[3,:],label='Orbit 300')
plt.plot(type2short[500,1:],color=colors[4,:],label='Orbit 500')
plt.plot(type2short[700,1:],color=colors[5,:],label='Orbit 700')
#plt.plot(type2short[900,1:],color=colors[6,:],label='Orbit 900')

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Time Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Orbital Radius (arb)',fontsize=9)
#plt.xlim(1,250)
#plt.ylim(0,0.5)
plt.legend(loc='lower right',fontsize=2,frameon=False,handlelength=2)

savefilename='type2Short.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')





fig=plt.figure(num=3,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.15  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.15  # the bottom of the subplots of the figure
top = 0.92      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

ax1=plt.subplot(1,1,1)
plt.title('Type 3-1 Orbits - Short Dataset',fontsize=6)
plt.plot(type31short[1,1:],color=colors[0,:],label='Orbit 1')
plt.plot(type31short[100,1:],color=colors[1,:],label='Orbit 100')
plt.plot(type31short[200,1:],color=colors[2,:],label='Orbit 200')
plt.plot(type31short[300,1:],color=colors[3,:],label='Orbit 300')
plt.plot(type31short[500,1:],color=colors[4,:],label='Orbit 500')
plt.plot(type31short[700,1:],color=colors[5,:],label='Orbit 700')
#plt.plot(type31short[900,1:],color=colors[6,:],label='Orbit 900')

#lorenztian
#arbtime = np.arange(len(type3short[500,1:]))
#lortau = 143.0
#lorentz = (lortau**2/(((arbtime-640)**2)+(lortau**2)))+3.3264
#plt.plot(arbtime,lorentz,color='gray')

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Time Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Orbital Radius (arb)',fontsize=9)
#plt.xlim(1,250)
#plt.ylim(0,0.5)
plt.legend(loc='lower right',fontsize=2,frameon=False,handlelength=2)

savefilename='type31Short.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')



fig=plt.figure(num=4,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.15  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.15  # the bottom of the subplots of the figure
top = 0.92      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

ax1=plt.subplot(1,1,1)
plt.title('Type 3-2 Orbits - Short Dataset',fontsize=6)
plt.plot(type32short[1,1:],color=colors[0,:],label='Orbit 1')
plt.plot(type32short[100,1:],color=colors[1,:],label='Orbit 100')
plt.plot(type32short[200,1:],color=colors[2,:],label='Orbit 200')
plt.plot(type32short[300,1:],color=colors[3,:],label='Orbit 300')
plt.plot(type32short[500,1:],color=colors[4,:],label='Orbit 500')
plt.plot(type32short[700,1:],color=colors[5,:],label='Orbit 700')
#plt.plot(type32short[900,1:],color=colors[6,:],label='Orbit 900')

#lorenztian
#arbtime = np.arange(len(type3short[500,1:]))
#lortau = 143.0
#lorentz = (lortau**2/(((arbtime-640)**2)+(lortau**2)))+3.3264
#plt.plot(arbtime,lorentz,color='gray')

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Time Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Orbital Radius (arb)',fontsize=9)
#plt.xlim(1,250)
#plt.ylim(0,0.5)
plt.legend(loc='lower right',fontsize=2,frameon=False,handlelength=2)

savefilename='type32Short.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')


fig=plt.figure(num=5,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.15  # the left side of the subplots of the figure
right = 0.94    # the right side of the subplots of the figure
bottom = 0.15  # the bottom of the subplots of the figure
top = 0.92      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

ax1=plt.subplot(1,1,1)
plt.title('Type 4 Orbits - Short Dataset',fontsize=6)
plt.plot(type4short[1,1:],color=colors[0,:],label='Orbit 1')
plt.plot(type4short[100,1:],color=colors[1,:],label='Orbit 100')
plt.plot(type4short[200,1:],color=colors[2,:],label='Orbit 200')
plt.plot(type4short[300,1:],color=colors[3,:],label='Orbit 300')
plt.plot(type4short[500,1:],color=colors[4,:],label='Orbit 500')
plt.plot(type4short[700,1:],color=colors[5,:],label='Orbit 700')
#plt.plot(type4short[900,1:],color=colors[6,:],label='Orbit 900')

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlabel('Time Steps',fontsize=9)
#ax1.set_xticklabels([])
plt.ylabel('Orbital Radius (arb)',fontsize=9)
#plt.xlim(1,250)
#plt.ylim(0,0.5)
plt.legend(loc='lower right',fontsize=2,frameon=False,handlelength=2)

savefilename='type4Short.png'
savefile = os.path.normpath(datadir+savefilename)
plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')