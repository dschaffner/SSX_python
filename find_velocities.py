# -*- coding: utf-8 -*-
"""
Created on Fri Feb 03 01:18:32 2017

@author: dschaffner
"""

import get_corr as gc
import numpy as np
import scipy.signal as sig
import matplotlib.pylab as plt
import process_mjmag_data as mj
import ssxuserfunctions as ssxuf
import find_TOFvelocity as tof

day='123016'
shot=51

velocity = tof.find_TOF(day,shot,20.0,40.0)