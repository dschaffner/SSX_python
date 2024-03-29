import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

import process_mjmag_data as pmd
#shot='012417r03'
day='123016'
day='010517'
day='010617'
day='051917'
#day='012417'
day='062321'
start_shot = 2
end_shot = 2
nshots = (end_shot-start_shot)+1
maxBs = np.zeros([25,nshots])

for shot in np.arange(start_shot,end_shot+1):
    if shot == 57: continue
    print( 'Processing Shot ',shot)
    time,Bdot,timeB,B,Bmod,bdat=pmd.process_mjmag_data(day+'r'+str(shot))
    for pos in np.arange(25):
        max_xyz = np.zeros([3])    
        for d in np.arange(3):
            max_xyz[d]=np.max(np.abs(B[d,pos,:]))
        maxBs[pos,shot-start_shot]=np.max(max_xyz)
mean_maxBs = np.mean(maxBs,axis=1)
median_maxBs = np.median(maxBs,axis=1)
axial_pos = np.array([-3,-1.5,0,1.5,3,4.5,6,7.5,9,10.5,12,13.5,15,16.5,18,19.5,21,22.5,24,25.5,32.6,39.7,46.8,53.9,61])

plt.rc('axes',linewidth=0.75)#axis border widths (I tend to like bolder than the default)
plt.rc('xtick.major',width=0.75)#tick widths (I like them the same width as the border)
plt.rc('ytick.major',width=0.75)
plt.rc('xtick.minor',width=0.75)
plt.rc('ytick.minor',width=0.75)
plt.rc('lines',markersize=4,markeredgewidth=0.0)#size of markers,no outline

left  = 0.1  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.15   # the bottom of the subplots of the figure
top = 0.90      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots


fig=plt.figure(num=1,figsize=[8,2.5],dpi=300,facecolor='w',edgecolor='k')
#apply settings for margin listed above to the figure
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
#call subplot object
ax=plt.subplot(1,1,1)#(num rows, num columns, subplot position)

plt.plot(axial_pos,mean_maxBs,marker='o',color='black',linewidth=1.5,label=r'Mean Max $B_{i}$')
plt.plot(axial_pos,median_maxBs,color='blue',linewidth=1.5,label=r'Median Max $B_{i}$')

titletext=day+' Shot '+str(start_shot)+' to '+str(end_shot)
plt.title(titletext,fontsize=9)
#set labels, labels sizes, ticks, ticks sizes
plt.xlabel('Position from end of FC [cm]',fontsize=9)
plt.ylabel('B [G]',fontsize=9)

major_ticks = np.arange(-10, 75, 5)                                              
minor_ticks = np.arange(-10, 71, 1)                                               

ax.set_xticks(major_ticks)                                                       
ax.set_xticks(minor_ticks, minor=True)                                           
#ax.set_yticks(major_ticks)                                                       
#ax.set_yticks(minor_ticks, minor=True) 

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.xlim(-5,65)
plt.ylim(0,3000)

plt.vlines(0,ax.get_ylim()[0],ax.get_ylim()[1],color='gray',linestyle='dotted',linewidth=0.5)
plt.text(0.1,0.92,'End of FC',horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,fontsize=3)
plt.vlines(29.5,ax.get_ylim()[0],ax.get_ylim()[1],color='gray',linestyle='dotted',linewidth=0.5)
plt.text(0.52,0.92,'Start of FC',horizontalalignment='center',verticalalignment='center',transform=ax.transAxes,fontsize=3)

leg=plt.legend(loc='lower right',fontsize=5,frameon=False)

savefile='AveMaxB_PerPos_'+day+'Shots'+str(start_shot)+'to'+str(end_shot)+'.png'
#plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')