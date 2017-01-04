#!/usr/bin/env python
"""SSX python defaults"""

# 2/9/09 4:08 PM by Tim Gray
# 897978A7-5ABF-466A-AEA0-5F4BF12949DD

# $Id$
# $HeadURL$

__author__ = "Tim Gray"
__version__ = "1.0"

import os
import matplotlib as mpl
from pylab import figure

# location of Python directory and hosts
#location = os.uname()[1]
#if location == 'ion.physics.swarthmore.edu':
#	base = '/ssx/'
#elif location == 'd197.scdc1.swarthmore.edu':
#	base = os.path.expanduser('~/Documents/SSX')
#else:
#	base = os.path.expanduser('~/Documents')

#if os.getlogin() != 'tgray':
#	base = os.path.expanduser('~/Documents')
#else:
#	base = os.path.expanduser('~/Documents/SSX')

base = 'C:\Users\dschaffner\Google Drive\Data Deposit'

# figure sizes - should work in an module that import ssx_py_utils as ssxutil

pubsize1= {'figsize': (3.25,2.47), 'frameon': False, 'dpi': 300}
pubsize2= {'figsize': (3.25,2.0), 'frameon': False, 'dpi': 300}
pubsize3= {'figsize': (3.25,1.2), 'frameon': False, 'dpi': 300}
docsize={'figsize': (5,2.47), 'frameon': False, 'dpi': 300}
specsize={'figsize': (10,4), 'frameon': True, 'dpi': 72}
displaysize = {'figsize': (5.33,4), 'frameon': True, 'dpi': 96}

f = displaysize

oldSettings = {}

params = ['font.serif', 'font.sans-serif', 'legend.handletextpad', 'legend.borderaxespad', 'legend.labelspacing', 'legend.borderpad', 'font.size', 'axes.titlesize', 'axes.labelsize', 'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize']

newvalues = [['Garamond Premier Pro', 'Bitstream Vera Serif'], ['Cronos Pro', 'Arev Sans', 'Bitstream Vera Sans'], 0.5, 1.0, 0.2, 0.5, 9, 12, 10, 9, 9, 9]

oldvalues = [['Bitstream Vera Serif'], ['Arev Sans', 'Bitstream Vera Sans'], 0.02, 0.02, 0.010, 0.2, 8, 10, 9, 8, 8, 8]

pubvalues = [['Bitstream Vera Serif'], ['Arev Sans', 'Bitstream Vera Sans'], 0.02, 0.02, 0.010, 0.2, 6, 8, 7, 6, 6, 6]


timParams = dict(zip(params, newvalues))
oldParams = dict(zip(params, oldvalues))
pubParams = dict(zip(params, pubvalues))

def getSettings():
	settings = []
	for i in params:
		settings.append(mpl.rcParams[i])
	return dict(zip(params,settings))
	
def setTimFonts():
	for i in params:
		oldSettings[i] = mpl.rcParams[i]
		mpl.rcParams[i] = timParams[i]

def setPubFonts():
	for i in params:
		oldSettings[i] = mpl.rcParams[i]
		mpl.rcParams[i] = pubParams[i]
	

def restoreOldFonts():
	for i in params:
		mpl.rcParams[i] = oldSettings[i]
	
def setOldFonts():
	for i in params:
		oldSettings[i] = mpl.rcParams[i]
		mpl.rcParams[i] = oldParams[i]	
		
def pubfig(fignum, short = False, *arg):
	setPubFonts()
	if short:
		fig = figure(fignum, *arg, **pubsize3)
	else:
		fig = figure(fignum, *arg, **pubsize2)
	fig.subplotpars.left = .17
	fig.subplotpars.right = .9
	fig.clear()
	return fig	
