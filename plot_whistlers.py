#plot_whistlers

import numpy as np
import matplotlib.pylab as plt

c=299799000.0
B=400.0
n=2e15

wce = B*1.76e7
wpe = np.sqrt(n)*5.64e4
wci = B*9.58e3
wpi = np.sqrt(n)*1.32e3
va = (B*2.18e11)/(np.sqrt(n)*100.0)

w = np.arange(0,5000000,1.0)
k1=w**2/c**2
k2=((wpe/w)**2)/(1-(wce/w))
k3=((wpi/w)**2)/(1+(wci/w))
k=np.sqrt(k1*(1-k2-k3))

kva = w/va

fig=plt.figure(num=1,figsize=(3.5,3.5),dpi=300,facecolor='w',edgecolor='k')
left  = 0.15  # the left side of the subplots of the figure
right = 0.95    # the right side of the subplots of the figure
bottom = 0.23   # the bottom of the subplots of the figure
top = 0.96      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.1   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
ax = plt.subplot(1,1,1)

plt.plot(k,w/1000000,label='Whistler')
plt.plot(kva,w/1000000,label='Alfven')
plt.xlim(0,100)
plt.ylim(0,4)
plt.hlines(wci/1000000,0,100,color='green',linestyles='dotted')
plt.hlines(300000*2*np.pi/1000000,0,100,color='grey',linestyles='dashed',linewidth=0.5)
plt.xlabel(r'k [rad/m]',fontsize=10)
plt.ylabel(r'$\omega \times 10^{6}$ [Rad/Sec]',fontsize=10)
plt.text(10,3.65,r'$\omega_{ci}$',fontsize=6,color='green')
plt.text(10,2,r'$f\approx 300kHz$',fontsize=5)
plt.text(0,-1,'B='+str(B)+'G',fontsize=8)
leg=plt.legend(loc='lower right',fontsize=5,frameon=False)

plt.figure(2)
plt.semilogy(k,w/1000000,label='Whistler')
plt.semilogy(kva,w/1000000,label='Alfven')
plt.xlim(0,100)
plt.ylim(5e-1,4e0)