#plot_FFT.py

import numpy as np
import matplotlib.pylab as plt

#Here is some sample sine wave data

x=np.arange(10000)*0.01

y1=np.sin(1*x)
y2=np.sin(2*x)
y3=np.sin(3*x)
y10=np.sin(10*x)
y20=np.sin(20*x)

plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y10)