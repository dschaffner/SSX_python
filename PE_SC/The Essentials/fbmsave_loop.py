#fbmsave_loop

from fbmsave_davidmod import fbm
import numpy as np

for h in np.arange(0.05,1,0.05):
    for tag in np.arange(1,10):
        fbm(100,h,tag=tag)