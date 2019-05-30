#!/usr/bin/env python
"""Core SSX functions.

Contains functions that return data for the specified diagnostic and plotting
routines.  This is the top level python file for ssx data analysis.  Import
this when running from the interactive prompt or when running a script."""

__author__ = "Tim Gray"
__version__ = "1.7.1"

# a note about version numbers
# ----------------------------
# For the most part, I try to implement the version number of this file
# (ssx.py) and ssx_data_read.py everytime there is a new version.  I also
# try to increment the version number of any other file that was editing DURING
# this version's commits.  Everything else I leave alone.  Technically, I
# sometimes leave ssx_data_read alone too if I never actually changed anything
# in that file, but that rarely happens.  Also, it's such a crucial file that
# it doesn't hurt to have the most recent number there as well.
#
# Once I am 'happy' with a version, I will tag that version with git using the
# command `git tag -am "version.number" version.number.  So files might have a
# version number newer than the most recent git tag.  Once I feel enough
# changes have occurred and that the version is relatively stable, I tag that
# commit and start working on the next version.
#
# Sometimes I forget to increment numbers while I'm working and have to do a
# mini commit of just version numbers.  See version 1.7.1 for an example.

#
#
# 1.0 - original code
# 1.1 - 10/29/08 - Changed some things for the new data storage on ion
# 1.1.1 - 11/7/08 - Fixed up the interferometer routine and changed defaults
# for the offset sight line and for the new VI setup
# 1.2 - 2/9/09 - tried to clean up some of rampant importing
# 1.5 - 2011-05-27 - Move to OS X 10.6, Python 2.7.1, Numpy 2.dev, Matplotlib
# 1.0.1, and scipy 0.10.
# 1.6 - 2012-07-12 - many updates:
#       - calibration for hires probe
#       - FINALLY added in proper gain calculations for IDS
#       - cleaned up docstrings in hires and ids
#       - added a bunch of methods to hires
# 1.7 - 2012-07-12 - started right after 1.6 finished
#       - moved many methods to private (start with _) to cleanup namespace in
#         interactive mode
# 1.7.1 - 2013-01-28 - final version from Tim
#       - fixed a couple of bugs and minor things that snuck in.  Mostly the
#         longprobe bug that I added in sometime during the 1.7 commits.

import ssxdefaults as ssxdef

import ssxreadin as sdr

import sys
import os
import getopt
import datetime as dt

import numpy as np
import scipy as sp

from numpy import array, arange, zeros, ma

from pylab import figure, close, subplot, axes, show, ion, ioff, ylim, xlim
from pylab import ylabel, xlabel, title

# my stuff
import ssxmathfuncs as mf
import ssxutilities as ssxutil
#import magnetics as mag
#import longprobe as lp
#import hiresmag as hr
#import ids
# import rga
# import oceano as oo


#import __builtin__ as builtin
#set = builtin.set

def switch_sdr(v="new"):
    """switches to the old sdr version.

    The old sdr is useful for reading old data - pre 2007.  This is data from
    the Mac G4 data acquisition computer."""
    if v == "new":
        import ssx_data_read as sdr
    else:
        import ssx_data_read_old as sdr


def tripleProbe(run, scope = '1', smoothing = 100, mult=None):
    """Reads data from triple probe and analysizes it.

    This hasn't been updated in a long time so use at your discretion.  It
    probably eneds to be tweaked."""
    data = sdr.scope_data(run, scope)

    names = ('vd2', 'vd3', 'i1', 'vfloat')
    channels = (1, 2, 3, 4)
    if not mult:
        mult = (-50, -50, 10, 200)
    units = ('V', 'V', 'A', 'V')

    data.setScopeProperties(names, channels, mult, units)


    data.temperature = data.vd2/np.log(2)
    data.temperature2 = data.vd2 / (np.log(2) - np.log( 1 + 11.3205 * (data.vd2
        / data.vd3)**3.53217))

    data.temperatureSmooth = mf.smooth(data.vd2, smoothing) / np.log(2)
    data.temperature2Smooth = mf.smooth(data.vd2, smoothing) / (np.log(2) -
        np.log( 1 + 11.3205 * (mf.smooth(data.vd2, smoothing) /
        mf.smooth(data.vd3, smoothing))**3.53217))


    mi = 1.6726e-27
    elecCharge = 1.6022e-19
    tconv = 1.1604e4
    area = 3.6e-6
    boltz = 1.3807e-23

    data.isat = data.i1 * (np.exp(-(data.vd2/data.temperature)))/(1 -
        np.exp(-(data.vd2/data.temperature)))

    data.isatSmooth =  mf.smooth(data.i1, smoothing) * (np.exp( -
        (mf.smooth(data.vd2, smoothing) / data.temperature2Smooth))) / (1 -
        np.exp( -(mf.smooth(data.vd2, smoothing) / data.temperature2Smooth)))

    data.density = (data.isat * np.exp(.5) ) / (elecCharge * area) * sqrt( 
        mi/(data.temperature * tconv * boltz)) / 1e6

    data.density2 = (data.isatSmooth * np.exp(.5) ) / (elecCharge * area) * sqrt( 
        mi/(data.temperature2Smooth * tconv * boltz)) / 1e6


    return data

def gunPlots(shot, scope = '3', writeFiles = False, ext = 'png'):
    """Plots gun data from scope 3."""
    data = sdr.scope_data(shot, scope)
    fig = figure(12, **ssxdef.f)
    fig.clear()
    a = axes()
    a.plot(data.time,data.ch2 * 170,label='East')
    a.plot(data.time,data.ch4 * 190,label='West')
    xlabel('Time (us)')
    ylabel('Gun Current (kA)')
    title(data.shotname)
    a.legend()

    fig2 = figure(13, **ssxdef.f)
    fig2.clear()
    a = axes()
    a.plot(data.time,data.ch1,label='East')
    a.plot(data.time,data.ch3,label='West')
    xlabel('Time (us)')
    ylabel('Gun voltage ()')
    title(data.shotname)
    a.legend()

    if writeFiles:
        fName = ssxutil.ssxPath('guncurrent.' + ext, 'output', data.runYear +
            '/' + data.runDate + '/' + data.shotname, mkdir = True)
        fig.savefig(fName)
        fName2 = ssxutil.ssxPath('gunvoltage.' + ext, 'output', data.runYear +
            '/' + data.runDate + '/' + data.shotname)
        fig2.savefig(fName2)
    else:
        show()

    return data
def sxr(run, scope = '2', showPlot = False):
    """Plots the SXR data."""

    data = sdr.scope_data(run, scope)
    data.ylabel = 'mA'
    names = data.header[1:]
    channels = (1,2,3,4)
    mult = (1/50. * 1000,) * 4
    units = (data.ylabel,) * 4

    data.setScopeProperties(names, channels, mult, units)
    
    return data

def interferometer(run, calib, scope = '2', diam = 15.5575, showPlot = False,
    writeFiles = False):
    """Interferometer analysis routine.

    Reads data from the scope data files and runs the analysis.  Inputting a
    proper diameter is necessary.

    Diameter is path length of interferometer in cm.  In oblate flux
    conservers it should be 50 cm.  For neck part of oblate conservers, it
    should be 15.5575 cm.
    
    calib should be a tuple/list of the maximum and minimums observed on the
    scope for each channel at the beginning of the run day."""

    data = sdr.scope_data(run, scope)
    names = ['signal1', 'signal2']
    channels = (1, 2)
    mult = (1, 1)
    units = ('arb', 'arb')

    data.setScopeProperties(names, channels, mult)
    data.calib = dict(zip(names, calib))

    dv1 = data.signal1 - np.mean(data.signal1[0:10])
    dv2 = data.signal2 - np.mean(data.signal2[0:10])
    
    arg = 1-0.5*( (dv1/data.calib[names[0]])**2 +
        (dv2/data.calib[names[1]])**2 )
    #remove values outside of arccos domain
    spikes = np.where(arg < -1.0)
    arg[spikes]=-1.0
    
    dphi = np.arccos(arg)
    data.dphi = dphi

    # .155575 m is the ID of the small part of the flux conserver
    # mks
    #density = (dphi * 4 * pi * (3e8)**2 * 8.854e-12 * 9.109e-31) /
    #((1.602e-19)**2 * 632.8e-9 * .155575)

    #cgs
    density = (dphi * (3e10)**2 * 9.109e-28) / ((4.8032e-10)**2 * 632.8e-7 *
        diam)

    data.pathlength = diam
    data.density = density

    if showPlot or writeFiles:
        fig = figure(13, **ssxdef.f)
        fig.clear()
        a = axes()
        a.plot(data.time, data.density, 'k-')
        xlabel('Time (us)')
        ylabel('Density (#/cm$^3$)')
        title(data.shotname)
        if writeFiles:
            # make output directory
            fName = ssxutil.ssxPath('interferometer.png', 'output',
                data.runYear + '/' + data.runDate + '/' + run)
            dir, trash = os.path.split(fName)
            os.spawnlp(os.P_WAIT, 'mkdir', 'mkdir', '-p', dir)
            fig.savefig(fName)
        else:
            show()

    return data

def gunandint(c, run):
    """Simple routine to plot guns and intereferometer.

    This might be out of date.  It was just a convenience function."""
    interferometer(run, c, writeFiles = 1)
    gunPlots(run, writeFiles = 1)

def vuv(run, scope = '1', amp = 100e-6, channel = 4):
    """VUV data.  

    amp is in amps/volt, so 100 ua/v is 100e-6."""

    data = sdr.scope_data(run, scope)
    names = ['vuv']
    channels = [channel]

    mult = [amp]
    units = ['A']

    data.setScopeProperties(names, channels, mult, units)

    return data

def vuvLookupCurve(ratio, showPlot = False):
    """Takes ratio of 97/155 lines and returns T_e at 3 densities.
    1 density for now (5e14).

    Added in the factor of 5 senstivity from MacPherson.

    Data from below:
    -----
    97over155

    columns:
    1: T (ev)
    2: 97/155 ratio for ne=1e14
    3: 97/155 ratio for ne=5e14
    4: 97/155 ratio for ne=2e15

    5	63.8344	69.25164	9.979064
    10	0.1552918	0.09093285	0.05106842
    15	0.03828019	0.02603559	0.01611151
    20	0.01610708	0.01156093	0.00755031
    25	0.008135443	0.00604283	0.00407848
    30	0.004953774	0.00376594	0.002603922
    35	0.003384	0.002619757	0.001845273
    40	0.002494533	0.001958196	0.001399227
    45	0.001939878	0.001539707	0.00111413
    50	0.001568844	0.00125666	9.193383E-4
    55	0.001307071	0.001054908	7.789212E-4
    60	0.001114454	9.052276E-4	6.737394E-4
    65	9.631904E-4	7.868332E-4	5.8992E-4
    70	8.449692E-4	6.937241E-4	5.231481E-4
    75	7.504907E-4	6.188965E-4	4.691748E-4
    80	6.73543E-4	5.576425E-4	4.248821E-4
    85	6.098714E-4	5.067226E-4	3.877307E-4
    90	5.564431E-4	4.638157E-4	3.564039E-4
    95	5.110662E-4	4.272359E-4	3.293811E-4
    100	4.721179E-4	3.957284E-4	3.062352E-4
    #
    """
    data = array([[  5.00000000e+00,   1.00000000e+01,   1.50000000e+01,
          2.00000000e+01,   2.50000000e+01,   3.00000000e+01,
          3.50000000e+01,   4.00000000e+01,   4.50000000e+01,
          5.00000000e+01,   5.50000000e+01,   6.00000000e+01,
          6.50000000e+01,   7.00000000e+01,   7.50000000e+01,
          8.00000000e+01,   8.50000000e+01,   9.00000000e+01,
          9.50000000e+01,   1.00000000e+02],
       [  6.38344000e+01,   1.55291800e-01,   3.82801900e-02,
          1.61070800e-02,   8.13544300e-03,   4.95377400e-03,
          3.38400000e-03,   2.49453300e-03,   1.93987800e-03,
          1.56884400e-03,   1.30707100e-03,   1.11445400e-03,
          9.63190400e-04,   8.44969200e-04,   7.50490700e-04,
          6.73543000e-04,   6.09871400e-04,   5.56443100e-04,
          5.11066200e-04,   4.72117900e-04],
       [  6.92516400e+01,   9.09328500e-02,   2.60355900e-02,
          1.15609300e-02,   6.04283000e-03,   3.76594000e-03,
          2.61975700e-03,   1.95819600e-03,   1.53970700e-03,
          1.25666000e-03,   1.05490800e-03,   9.05227600e-04,
          7.86833200e-04,   6.93724100e-04,   6.18896500e-04,
          5.57642500e-04,   5.06722600e-04,   4.63815700e-04,
          4.27235900e-04,   3.95728400e-04],
       [  9.97906400e+00,   5.10684200e-02,   1.61115100e-02,
          7.55031000e-03,   4.07848000e-03,   2.60392200e-03,
          1.84527300e-03,   1.39922700e-03,   1.11413000e-03,
          9.19338300e-04,   7.78921200e-04,   6.73739400e-04,
          5.89920000e-04,   5.23148100e-04,   4.69174800e-04,
          4.24882100e-04,   3.87730700e-04,   3.56403900e-04,
          3.29381100e-04,   3.06235200e-04]])

    te = arange(5,100,.001)
    logratio = arange(-8.13, 0, .1)
    rat = np.exp(logratio)

    x = data[0]
    xr = x[::-1]
    y = data[1]
    z = data[2]
    w = data[3]

    data = [y,z,w]

# 	logdata = [np.log(dat) for dat in data]
    datar = [dat[::-1] for dat in data]
    splr = [sp.interpolate.splrep(np.log(dat), xr, k=3) for dat in datar]
    splines = [sp.interpolate.splev(logratio, spl) for spl in splr]
# 	splines = [np.exp(logspline) for logspline in logsplines]
# 	
    y4, z4, w4 = splines

    if showPlot:
        polygon = mlab.poly_between(x,y,w)

        ioff()
        f = figure(1, **ssxdef.f)
        f.clear()
        a = axes()
        a.semilogy(y4, rat, 'k-')
        a.semilogy(z4, rat, 'r-')
        a.semilogy(w4, rat, 'b-')

        a.semilogy(x, y, 'ks', hold = 1, label='n = 1 x 10^14')	# n = 1e14
        a.semilogy(x, z, 'ro', label='n = 5 x 10^14')			# n = 5e14
        a.semilogy(x, w, 'b^', label='n = 2 x 10^15')			# n = 2e15
        xlabel('T (eV)')
        ylabel('Line Ratio (97.7/155)')
        ylim(.0001,1)
        xlim(0,100)
        a.legend(numpoints = 1)
        f.savefig('lineratios1.pdf')

        f = figure(2, **ssxdef.f)
        f.clear()
        a = axes()
        a.semilogy(z4, rat, 'k-')
        a.fill(polygon[0], polygon[1], alpha=.3)
        a.semilogy(x, z, 'ko')
        title('Line ratios for n = 5 x 10^14')
        xlabel('T (eV)')
        ylabel('Line Ratio (97.7/155)')
        ylim(.0001,1)
        xlim(0,100)
        a.text(65,.5,'upper bound is 1x10^14')
        a.text(65,.37,'lower bound is 2x10^15')
        f.savefig('lineratios2.pdf')
        show()
        ion()

    splz = splr[1]
    # factor of 5 from sensitivity from macpherson
    # less sensitive to the 97.7 line, so actual amount of light is 5 times as
    # much at this wavelength
    temp = sp.interpolate.splev(np.log(ratio*5), splz)

    temp = ma.masked_less(temp, sp.interpolate.splev(np.log(1), splz))
    temp = ma.masked_invalid(temp)

    return temp

def expDecay(x, p):
    """exponentional function for fitting."""
    scale, offset, t, z = p
    y = scale * np.exp(t/x**z) - offset
    return y

def fitExpDecay(x, y, scale, offset):
    """Fitting routine for exp decays."""
    def resid(p, x, y):
        return y - expDecay(x, p)

    z = y
    y = np.log10(z)
    t = 1
    p0 = (scale, offset, t, z)
    plsqFull = sp.optimize.leastsq(resid, p0, args = (x, y), full_output=True)
    p = plsqFull[0]
    plsq = plsqFull[1]
    chi2 = sum((resid(p, x, y))**2)
    fit = expDecay(x, p)
    fit = 10**fit

    semilogy(x,z,'ko',hold=0)
    semilogy(x,fit)
    return fit, p, plsq, chi2

def vuvCalibrationCurve(wavelength, showFit = False):
    """VUV calibration curve."""
    # summer 2006 data
    actualWL = array([63., 97.7, 121.6, 123.9, 155.0, 229.7])
    monoSetting = array([61.6, 95.6, 119., 121.4, 151.4, 224.])

    if wavelength in actualWL:
        i = actualWL.searchsorted(wavelength)
        setting = monoSetting[i]
    else:
        a,b = polyfit(actualWL, monoSetting, 1)
        setting = a * wavelength + b

    if showFit:
        wl = [40,250]
        calib = polyval([a,b], wl)

        fig = figure(7, figsize =ssxdef.f)
        a = axes()
        a.plot(wl, calib, 'b-', hold = 0)
        a.plot(actualWL, monoSetting, 'ks')
        title('VUV monochromator calibration curve - summer 2006')
        xlim(40,250)
        ylim(40,250)
        xlabel('Actual wavelength (nm)')
        ylabel('VUV setting (nm)')
        a.text(55,217,'y = %.4f * x + %.4f' % (a, b))
        fig.savefig('vuv-settings.pdf')

    print ("VUV setting: %.1f" % (setting))
    return setting
