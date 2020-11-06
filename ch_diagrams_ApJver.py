#permutation entropy diagram

import numpy as np
import matplotlib.pylab as plt
import os

process_dir = 'C:\\Users\\dschaffner\\Dropbox\\From OneDrive\\Galatic Dynamics Data\\CH Diagrams for Paper\\'



fig=plt.figure(num=1,figsize=(6,4),dpi=600,facecolor='w',edgecolor='k')#figsize(width,height)
left  = 0.05  # the left side of the subplots of the figure
right = 0.95   # the right side of the subplots of the figure
bottom = 0.07   # the bottom of the subplots of the figure
top = 0.95      # the top of the subplots of the figure
wspace = 0.05   # the amount of width reserved for blank space between subplots
hspace = 0.15   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

plt.rc('axes',linewidth=0.75)
plt.rc('xtick.major',width=0.75,size=0.0)
plt.rc('ytick.major',width=0.75)
plt.rc('xtick.minor',width=0.75)
plt.rc('ytick.minor',width=0.75)
plt.rc('lines',markersize=6.5,markeredgewidth=0.0)

plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42

numcolors=10
colors=np.zeros([numcolors,4])
for i in np.arange(numcolors):
    c = plt.cm.plasma(i/(float(numcolors)),1)
    colors[i,:]=c

import matplotlib.gridspec as gridspec
gs=gridspec.GridSpec(2,3)

ax1=fig.add_subplot(gs[1,0],facecolor='lavender')
x=np.array([1,2,3,4,5])
y=np.array([1,2,3,4,5])
plt.plot(x,y,'o-',color='blue')
#plt.axis('equal')
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.axes.get_xaxis().set_ticks([])
ax1.axes.get_yaxis().set_ticks([])
plt.xlim(-0.5,6.5)
plt.ylim(-2,6)
#plt.xticks(x,['1','2','3','4','5'],fontsize=5)
#plt.xlabel('Sequence Position',fontsize=5)
plt.text(1-0.1,-1,r'$1$',fontsize=8,horizontalalignment='left')
plt.text(2-0.1,-1,r'$2$',fontsize=8,horizontalalignment='left')
plt.text(3-0.1,-1,r'$3$',fontsize=8,horizontalalignment='left')
plt.text(4-0.1,-1,r'$4$',fontsize=8,horizontalalignment='left')
plt.text(5-0.1,-1,r'$5$',fontsize=8,horizontalalignment='left')
plt.text(1.5,-1,'|',fontsize=8)
plt.text(2.5,-1,'|',fontsize=8)
plt.text(3.5,-1,'|',fontsize=8)
plt.text(4.5,-1,'|',fontsize=8)
#plt.text(3,-0.5,'Ordinal Pattern:\n Sequence Postion Ordered\n From Lowest Value(Bottom) to Highest Value(Top)',fontsize=4,horizontalalignment='center')
plt.text(3,-0.2,'Ordinal Pattern:',fontsize=4,horizontalalignment='center')
plt.text(0.25,5.25,'(b)',horizontalalignment='center',verticalalignment='center',fontsize=10)


ax2=fig.add_subplot(gs[1,1],facecolor='lavender')
x=np.array([1,2,3,4,5])
y=np.array([4.1,3.5,1,1.5,0.4])
plt.plot(x,y,'o-',color='blue')
#plt.axis('equal')
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.axes.get_xaxis().set_ticks([])
ax2.axes.get_yaxis().set_ticks([])
plt.xlim(-0.5,6.5)
plt.ylim(-2,6)
#plt.xticks(x,['1','2','3','4','5'],fontsize=5)
#plt.xlabel('Sequence Position',fontsize=5)
plt.text(1-0.1,-1,r'$5$',fontsize=8,horizontalalignment='left')
plt.text(2-0.1,-1,r'$3$',fontsize=8,horizontalalignment='left')
plt.text(3-0.1,-1,r'$4$',fontsize=8,horizontalalignment='left')
plt.text(4-0.1,-1,r'$2$',fontsize=8,horizontalalignment='left')
plt.text(5-0.1,-1,r'$1$',fontsize=8,horizontalalignment='left')
plt.text(1.5,-1,'|',fontsize=8)
plt.text(2.5,-1,'|',fontsize=8)
plt.text(3.5,-1,'|',fontsize=8)
plt.text(4.5,-1,'|',fontsize=8)
#plt.text(3,-0.5,'Ordinal Pattern:\n Sequence Postion Ordered\n From Lowest Value(Bottom) to Highest Value(Top)',fontsize=4,horizontalalignment='center')
plt.text(3,-0.2,'Ordinal Pattern:',fontsize=4,horizontalalignment='center')
plt.text(0.25,5.25,'(c)',horizontalalignment='center',verticalalignment='center',fontsize=10)


ax3=fig.add_subplot(gs[1,2],facecolor='lavender')
x=np.array([1,2,3,4,5])
y=np.array([2,3,2.5,1,5])
plt.plot(x,y,'o-',color='blue')
#plt.axis('equal')
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.axes.get_xaxis().set_ticks([])
ax3.axes.get_yaxis().set_ticks([])
plt.xlim(-0.5,6.5)
plt.ylim(-2,6)
#plt.xticks(x,['0','1','2','3','4'],fontsize=5)
#plt.xlabel('Sequence Position',fontsize=5)
plt.text(1-0.1,-1,r'$4$',fontsize=8,horizontalalignment='left')
plt.text(2-0.1,-1,r'$1$',fontsize=8,horizontalalignment='left')
plt.text(3-0.1,-1,r'$3$',fontsize=8,horizontalalignment='left')
plt.text(4-0.1,-1,r'$2$',fontsize=8,horizontalalignment='left')
plt.text(5-0.1,-1,r'$5$',fontsize=8,horizontalalignment='left')
plt.text(1.5,-1,'|',fontsize=8)
plt.text(2.5,-1,'|',fontsize=8)
plt.text(3.5,-1,'|',fontsize=8)
plt.text(4.5,-1,'|',fontsize=8)
#plt.text(3,-0.5,'Ordinal Pattern:\n Sequence Postion Ordered\n From Lowest Value(Bottom) to Highest Value(Top)',fontsize=4,horizontalalignment='center')
plt.text(3,-0.2,'Ordinal Pattern:',fontsize=4,horizontalalignment='center')
plt.text(0.25,5.25,'(d)',horizontalalignment='center',verticalalignment='center',fontsize=10)


#filename = 'permutations_diagram_tau1.png'
#savefile = os.path.normpath(process_dir+filename)
#plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')


#fig=plt.figure(num=2,figsize=(6,2),dpi=300,facecolor='w',edgecolor='k')#figsize(width,height)
#left  = 0.05  # the left side of the subplots of the figure
#right = 0.95   # the right side of the subplots of the figure
#bottom = 0.15   # the bottom of the subplots of the figure
#top = 0.95      # the top of the subplots of the figure
#wspace = 0.05   # the amount of width reserved for blank space between subplots
#hspace = 0.15   # the amount of height reserved for white space between subplots
#plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

ax4=fig.add_subplot(gs[0,:],facecolor='white')
x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
y=np.array([3,1,2,3,4,5,2.8,4.1,3.5,1,1.5,0.4,1,2,3,2.5,1,5,1])
plt.plot(x,y,'o-',color='blue')
#plt.axis('equal')
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.axes.get_xaxis().set_ticks([])
ax4.axes.get_yaxis().set_ticks([])

#import matplotlib.transforms as mtransforms
#trans = mtransforms.blended_transform_factory(ax4.transData, ax4.transAxes)
ax4.axvspan(2, 6,facecolor='lavender')#, alpha=0.1)#, transform=trans)
ax4.axvspan(8, 12,facecolor='lavender')#, alpha=0.1)#, transform=trans)
ax4.axvspan(14, 18,facecolor='lavender')#, alpha=0.1)#, transform=trans)
plt.text(0.75,4.75,'(a)',horizontalalignment='center',verticalalignment='center',fontsize=10)


#filename = 'sampledata.png'
filename = 'sampledata_withhighlight_1tau.png'
savefile = os.path.normpath(process_dir+filename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()



fig=plt.figure(num=2,figsize=(6,4),dpi=600,facecolor='w',edgecolor='k')#figsize(width,height)
left  = 0.05  # the left side of the subplots of the figure
right = 0.95   # the right side of the subplots of the figure
bottom = 0.07   # the bottom of the subplots of the figure
top = 0.95      # the top of the subplots of the figure
wspace = 0.05   # the amount of width reserved for blank space between subplots
hspace = 0.15   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

gs=gridspec.GridSpec(2,3)

ax1=fig.add_subplot(gs[1,0],facecolor='lightcyan')
x=np.array([1,2,3,4,5])
y=np.array([1,4,4.1,1.5,2])
plt.plot(x,y,'o-',color='blue')
#plt.axis('equal')
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.axes.get_xaxis().set_ticks([])
ax1.axes.get_yaxis().set_ticks([])
plt.xlim(-0.5,6.5)
plt.ylim(-2,6)
#plt.xticks(x,['1','2','3','4','5'],fontsize=5)
#plt.xlabel('Sequence Position',fontsize=5)
plt.text(1-0.1,-1,r'$1$',fontsize=8,horizontalalignment='left')
plt.text(2-0.1,-1,r'$4$',fontsize=8,horizontalalignment='left')
plt.text(3-0.1,-1,r'$5$',fontsize=8,horizontalalignment='left')
plt.text(4-0.1,-1,r'$2$',fontsize=8,horizontalalignment='left')
plt.text(5-0.1,-1,r'$3$',fontsize=8,horizontalalignment='left')
plt.text(1.5,-1,'|',fontsize=8)
plt.text(2.5,-1,'|',fontsize=8)
plt.text(3.5,-1,'|',fontsize=8)
plt.text(4.5,-1,'|',fontsize=8)
#plt.text(3,-0.5,'Ordinal Pattern:\n Sequence Postion Ordered\n From Lowest Value(Bottom) to Highest Value(Top)',fontsize=4,horizontalalignment='center')
plt.text(3,-0.2,'Ordinal Pattern:',fontsize=4,horizontalalignment='center')
plt.text(0.25,5.25,'(b)',horizontalalignment='center',verticalalignment='center',fontsize=10)


ax2=fig.add_subplot(gs[1,1],facecolor='mistyrose')
x=np.array([1,2,3,4,5])
y=np.array([2,5,3.5,0.4,3])
plt.plot(x,y,'o-',color='blue')
#plt.axis('equal')
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.axes.get_xaxis().set_ticks([])
ax2.axes.get_yaxis().set_ticks([])
plt.xlim(-0.5,6.5)
plt.ylim(-2,6)
#plt.xticks(x,['1','2','3','4','5'],fontsize=5)
#plt.xlabel('Sequence Position',fontsize=5)
plt.text(1-0.1,-1,r'$4$',fontsize=8,horizontalalignment='left')
plt.text(2-0.1,-1,r'$1$',fontsize=8,horizontalalignment='left')
plt.text(3-0.1,-1,r'$5$',fontsize=8,horizontalalignment='left')
plt.text(4-0.1,-1,r'$3$',fontsize=8,horizontalalignment='left')
plt.text(5-0.1,-1,r'$2$',fontsize=8,horizontalalignment='left')
plt.text(1.5,-1,'|',fontsize=8)
plt.text(2.5,-1,'|',fontsize=8)
plt.text(3.5,-1,'|',fontsize=8)
plt.text(4.5,-1,'|',fontsize=8)
#plt.text(3,-0.5,'Ordinal Pattern:\n Sequence Postion Ordered\n From Lowest Value(Bottom) to Highest Value(Top)',fontsize=4,horizontalalignment='center')
plt.text(3,-0.2,'Ordinal Pattern:',fontsize=4,horizontalalignment='center')
plt.text(0.25,5.25,'(c)',horizontalalignment='center',verticalalignment='center',fontsize=10)


ax3=fig.add_subplot(gs[1,2],facecolor='navajowhite')
x=np.array([1,2,3,4,5])
y=np.array([3,2.8,1.0,1.0,2.5])
plt.plot(x,y,'o-',color='blue')
#plt.axis('equal')
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.axes.get_xaxis().set_ticks([])
ax3.axes.get_yaxis().set_ticks([])
plt.xlim(-0.5,6.5)
plt.ylim(-2,6)
#plt.xticks(x,['0','1','2','3','4'],fontsize=5)
#plt.xlabel('Sequence Position',fontsize=5)
plt.text(1-0.1,-1,r'$4$',fontsize=8,horizontalalignment='left')
plt.text(2-0.1,-1,r'$3$',fontsize=8,horizontalalignment='left')
plt.text(3-0.1,-1,r'$5$',fontsize=8,horizontalalignment='left')
plt.text(4-0.1,-1,r'$2$',fontsize=8,horizontalalignment='left')
plt.text(5-0.1,-1,r'$1$',fontsize=8,horizontalalignment='left')
plt.text(1.5,-1,'|',fontsize=8)
plt.text(2.5,-1,'|',fontsize=8)
plt.text(3.5,-1,'|',fontsize=8)
plt.text(4.5,-1,'|',fontsize=8)
#plt.text(3,-0.5,'Ordinal Pattern:\n Sequence Postion Ordered\n From Lowest Value(Bottom) to Highest Value(Top)',fontsize=4,horizontalalignment='center')
plt.text(3,-0.2,'Ordinal Pattern:',fontsize=4,horizontalalignment='center')
plt.text(0.25,5.25,'(d)',horizontalalignment='center',verticalalignment='center',fontsize=10)

ax4=fig.add_subplot(gs[0,:],facecolor='white')
x=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
y=np.array([3,1,2,3,4,5,2.8,4.1,3.5,1,1.5,0.4,1,2,3,2.5,1,5,1])
plt.plot(x,y,'o-',color='blue')
#plt.axis('equal')
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.axes.get_xaxis().set_ticks([])
ax4.axes.get_yaxis().set_ticks([])

#import matplotlib.transforms as mtransforms
#trans = mtransforms.blended_transform_factory(ax4.transData, ax4.transAxes)
ax4.axvspan(2-0.1, 2+0.1,facecolor='lightcyan')#, alpha=0.1)#, transform=trans)
ax4.axvspan(5-0.1, 5+0.1,facecolor='lightcyan')#, alpha=0.1)#, transform=trans)
ax4.axvspan(8-0.1, 8+0.1,facecolor='lightcyan')#, alpha=0.1)#, transform=trans)
ax4.axvspan(11-0.1, 11+0.1,facecolor='lightcyan')#, alpha=0.1)
ax4.axvspan(14-0.1, 14+0.1,facecolor='lightcyan')#, alpha=0.1)

ax4.axvspan(3-0.1, 3+0.1,facecolor='mistyrose')#, alpha=0.1)#, transform=trans)
ax4.axvspan(6-0.1, 6+0.1,facecolor='mistyrose')#, alpha=0.1)#, transform=trans)
ax4.axvspan(9-0.1, 9+0.1,facecolor='mistyrose')#, alpha=0.1)#, transform=trans)
ax4.axvspan(12-0.1, 12+0.1,facecolor='mistyrose')#, alpha=0.1)
ax4.axvspan(15-0.1, 15+0.1,facecolor='mistyrose')#, alpha=0.1)

ax4.axvspan(4-0.1, 4+0.1,facecolor='navajowhite')#, alpha=0.1)#, transform=trans)
ax4.axvspan(7-0.1, 7+0.1,facecolor='navajowhite')#, alpha=0.1)#, transform=trans)
ax4.axvspan(10-0.1, 10+0.1,facecolor='navajowhite')#, alpha=0.1)#, transform=trans)
ax4.axvspan(13-0.1, 13+0.1,facecolor='navajowhite')#, alpha=0.1)
ax4.axvspan(16-0.1, 16+0.1,facecolor='navajowhite')#, alpha=0.1)


plt.text(0.75,4.75,'(a)',horizontalalignment='center',verticalalignment='center',fontsize=10)


#filename = 'sampledata.png'
filename = 'sampledata_withhighlight_3tau.eps'
savefile = os.path.normpath(process_dir+filename)
#plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()












plt.rc('axes',linewidth=1.5)
plt.rc('xtick.major',width=1.5,size=0.0)
plt.rc('ytick.major',width=1.5)
plt.rc('xtick.minor',width=1.5)
plt.rc('ytick.minor',width=1.5)
plt.rc('lines',markersize=6.5,markeredgewidth=0.0)


fig=plt.figure(num=9,figsize=(9,7),dpi=600,facecolor='w',edgecolor='k')#figsize(width,height)
left  = 0.1  # the left side of the subplots of the figure
right = 0.95   # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.95      # the top of the subplots of the figure
wspace = 0.05   # the amount of width reserved for blank space between subplots
hspace = 0.25   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

gs=gridspec.GridSpec(3,3)
ax1=fig.add_subplot(gs[0,:],facecolor='white')


x=np.array([1,2,3,4,5,6,7,8,9,10,11])
y=np.array([95,5,0,0,0,0,0,0,0,0,0])

plt.bar(x,y,width=1.0,edgecolor='black',facecolor=colors[6,:])
plt.xticks(x,['12345','21345','31245','41235','12543','54321','41253','13254','51324','41253','...'],fontsize=11)
#plt.xlabel('Ordinal Pattern (5!=120 total bins)',fontsize=8)
plt.ylabel('Count',fontsize=10)
plt.ylim(0,100)
plt.text(0.97,0.93,'(a) - Ramp',horizontalalignment='right',verticalalignment='center',transform=ax1.transAxes,fontsize=10)

#filename = 'permutations_histogram_complex.png'
#savefile = os.path.normpath(process_dir+filename)
#plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')

ax2=fig.add_subplot(gs[1,:],facecolor='white')


x=np.array([1,2,3,4,5,6,7,8,9,10,11])
y=np.array([50,52,51,51,51,49,53,51,51,50,51])
plt.bar(x,y,width=1.0,edgecolor='black',facecolor=colors[6,:])
plt.xticks(x,['12345','21345','31245','41235','12543','54321','41253','13254','51324','41253','...'],fontsize=11)
#plt.xlabel('Ordinal Pattern (5!=120 total bins)',fontsize=8)
plt.ylabel('Count',fontsize=10)
plt.ylim(0,100)
plt.text(0.97,0.93,'(b) - Noise',horizontalalignment='right',verticalalignment='center',transform=ax2.transAxes,fontsize=10)


#filename = 'permutations_histogram_ramp.png'
#savefile = os.path.normpath(process_dir+filename)
#plt.savefig(savefile,dpi=300,facecolor='w',edgecolor='k')

ax3=fig.add_subplot(gs[2,:],facecolor='white')

x=np.array([1,2,3,4,5,6,7,8,9,10,11])
y=np.array([8,50,23,21,54,53,10,2,1,43,3])
plt.bar(x,y,width=1.0,edgecolor='black',facecolor=colors[6,:])
plt.xticks(x,['12345','21345','31245','41235','12543','54321','41253','13254','51324','41253','...'],fontsize=11)

plt.xlabel('Ordinal Pattern (5!=120 total bins)',fontsize=15)
plt.ylabel('Count',fontsize=12)
plt.ylim(0,100)
plt.text(0.97,0.93,'(c) - Complex',horizontalalignment='right',verticalalignment='center',transform=ax3.transAxes,fontsize=10)


#filename = 'permutations_histograms_complex_ramp_noise.png'
filename = 'permutations_histograms_complex_ramp_noise.eps'
savefile = os.path.normpath(process_dir+filename)
plt.savefig(savefile,dpi=600,facecolor='w',edgecolor='k')
plt.clf()
plt.close()