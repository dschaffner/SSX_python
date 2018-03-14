#fbmsave_loop

from fbmsave_davidmod import fbm
import numpy as np

for h in np.arange(0.05,1,0.05,dtype='float'):
    for tag in np.arange(1,10):
        x=fbm(10000,np.round(h,3),tag=tag)
        x=0.0