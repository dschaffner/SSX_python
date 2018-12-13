# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 21:00:45 2017

@author: dschaffner
"""
import numpy as np
import matplotlib.pylab as plt
from loadnpzfile import loadnpzfile
from calc_PE_SC import PE, CH, PE_dist, PE_calc_only
from collections import Counter
from math import factorial
import os
import get_corr as gc

#calc_PESC_solarwind_chen.py
datadir = 'C:\\Users\\dschaffner\\OneDrive - brynmawr.edu\\Galatic Dynamics Data\\DoublePendulum\\Plots for Kate\\'
#datadir = 'C:\\Users\\dschaffner\\testfiles\\'
for files in os.walk(datadir, topdown=False):
      for names in files:
          print names
          
print names[0]
print names[1]
print names[2]