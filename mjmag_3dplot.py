#mjmag_3dplot.py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

import process_mjmag_data as pmd
#shot='012417r03'
shot='123016r92'
time,Bdot,timeB,B,Bmod,bdat=pmd.process_mjmag_data(shot)


#for t in np.arange(7391):
fig = plt.figure()
for t in np.arange(0,5010,20):

    ax = fig.add_subplot(111, projection='3d')
    plt.quiver(0,0,0,B[0,0,t]/300,B[2,0,t]/300,B[1,0,t]/300,length=Bmod[0,t]/100,pivot='tail',arrow_length_ratio=0.1)
    plt.quiver(0,1,0,B[0,1,t]/300,B[2,1,t]/300,B[1,1,t]/300,length=Bmod[1,t]/100,pivot='tail',arrow_length_ratio=0.1)
    plt.quiver(0,2,0,B[0,2,t]/300,B[2,2,t]/300,B[1,2,t]/300,length=Bmod[2,t]/100,pivot='tail',arrow_length_ratio=0.1)
    plt.quiver(0,3,0,B[0,3,t]/300,B[2,3,t]/300,B[1,3,t]/300,length=Bmod[3,t]/100,pivot='tail',arrow_length_ratio=0.1)
    plt.quiver(0,4,0,B[0,4,t]/300,B[2,4,t]/300,B[1,4,t]/300,length=Bmod[4,t]/100,pivot='tail',arrow_length_ratio=0.1)
    plt.quiver(0,5,0,B[0,5,t]/300,B[2,5,t]/300,B[1,5,t]/300,length=Bmod[5,t]/100,pivot='tail',arrow_length_ratio=0.1)
    plt.quiver(0,6,0,B[0,6,t]/300,B[2,6,t]/300,B[1,6,t]/300,length=Bmod[6,t]/100,pivot='tail',arrow_length_ratio=0.1)
    plt.quiver(0,7,0,B[0,7,t]/300,B[2,7,t]/300,B[1,7,t]/300,length=Bmod[7,t]/100,pivot='tail',arrow_length_ratio=0.1)
    plt.quiver(0,8,0,B[0,8,t]/300,B[2,8,t]/300,B[1,8,t]/300,length=Bmod[8,t]/100,pivot='tail',arrow_length_ratio=0.1)
    plt.quiver(0,9,0,B[0,9,t]/300,B[2,9,t]/300,B[1,9,t]/300,length=Bmod[9,t]/100,pivot='tail',arrow_length_ratio=0.1)
    plt.quiver(0,10,0,B[0,10,t]/300,B[2,10,t]/300,B[1,10,t]/300,length=Bmod[10,t]/100,pivot='tail',arrow_length_ratio=0.1)
    plt.quiver(0,11,0,B[0,11,t]/300,B[2,11,t]/300,B[1,11,t]/300,length=Bmod[11,t]/100,pivot='tail',arrow_length_ratio=0.1)
    plt.quiver(0,12,0,B[0,12,t]/300,B[2,12,t]/300,B[1,12,t]/300,length=Bmod[12,t]/100,pivot='tail',arrow_length_ratio=0.1)
    plt.quiver(0,13,0,B[0,13,t]/300,B[2,13,t]/300,B[1,13,t]/300,length=Bmod[13,t]/100,pivot='tail',arrow_length_ratio=0.1)
    plt.quiver(0,14,0,B[0,14,t]/300,B[2,14,t]/300,B[1,14,t]/300,length=Bmod[14,t]/100,pivot='tail',arrow_length_ratio=0.1)
    plt.quiver(0,15,0,B[0,15,t]/300,B[2,15,t]/300,B[1,15,t]/300,length=Bmod[15,t]/100,pivot='tail',arrow_length_ratio=0.1)
    plt.quiver(0,16,0,B[0,16,t]/300,B[2,16,t]/300,B[1,16,t]/300,length=Bmod[16,t]/100,pivot='tail',arrow_length_ratio=0.1)
    plt.quiver(0,17,0,B[0,17,t]/300,B[2,17,t]/300,B[1,17,t]/300,length=Bmod[17,t]/100,pivot='tail',arrow_length_ratio=0.1)
    plt.quiver(0,18,0,B[0,18,t]/300,B[2,18,t]/300,B[1,18,t]/300,length=Bmod[18,t]/100,pivot='tail',arrow_length_ratio=0.1)
    plt.quiver(0,19,0,B[0,19,t]/300,B[2,19,t]/300,B[1,19,t]/300,length=Bmod[19,t]/100,pivot='tail',arrow_length_ratio=0.1)
    #plt.quiver(0,20,0,B[0,20,t]/3000,B[2,20,t]/3000,B[1,20,t]/3000,pivot='tail',arrow_length_ratio=0.1)
    #plt.quiver(0,21,0,B[0,21,t]/3000,B[2,21,t]/3000,B[1,21,t]/3000,pivot='tail',arrow_length_ratio=0.1)
    #plt.quiver(0,22,0,B[0,22,t]/3000,B[2,22,t]/3000,B[1,22,t]/3000,pivot='tail',arrow_length_ratio=0.1)
    #plt.quiver(0,23,0,B[0,23,t]/3000,B[2,23,t]/3000,B[1,23,t]/3000,pivot='tail',arrow_length_ratio=0.1)
    #plt.quiver(0,24,0,B[0,24,t]/3000,B[2,24,t]/3000,B[1,24,t]/3000,pivot='tail',arrow_length_ratio=0.1)
    ax.set_ylim([-2,22])
    ax.set_xlim([-1,1])
    ax.set_zlim([-1,1])
    ax.view_init(elev=24, azim=9)    
    ax.dist=10

    timelabel=round(timeB[t],2)
    ax.text2D(0.17,0.92,'Time: '+str(timelabel)+'us',horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,fontsize=12)
    #saving the plot
    #for the paper draft, its best to use png. When we actually submit a paper
    #we'll need to save the plot as a .eps file instead.
    process_dir = 'C:\\Users\\dschaffner\\Documents\\ssxpython\\plots\\Bquiver\\012417r03\\'
    filename = 'Bquiver_'+shot+'_timestep_'+str(t)+'_large.png'
    savefile = os.path.normpath(process_dir+filename)
    plt.savefig(savefile,dpi=200,facecolor='w',edgecolor='k')
    plt.clf()