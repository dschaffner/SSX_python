#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""IDS for SSX functions."""

# $Id$
# $HeadURL$

# 9/2/08 1:50 PM by Tim Gray
# F9745440-10ED-40B1-BF59-B93B55FD5A8B

__author__ = "Tim Gray"
__version__ = "1.7.1"

# 1.0
# 1.0.1 added code to allow processing of many runs at once from the command
# line
# 1.0.2 - 11/3/08 - Adding some code to better handle He shots with larger
# signal to noise ratio.
# 2.0b - 5/6/09 - fully working 2 gaussian fit + broke ids data getting,
# analysis, and plotting up like in magnetics.py.  Will be refactoring the code
# to use masked arrays though, so this isn't 2.0 final.
# 1.5 - 2011-06-29 - corresponds to v1.5 of ssx.py.  Am refactoring ids.py to
# use the new dtac sdr stuff.
# 1.6 - 2011-12-14 - corresponds to v1.6 of ssx.py.  No real changes to speak
# of here.
# 1.7 - 2012-07-12 - updated for private methods

import ssxdefaults as ssxdef
import ssxreadin as sdr
import ssxutilities as ssxutil
import glob, optparse
import os
import sys
import getopt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import ssxmathfuncs as mf
import scipy as sp
import scipy.optimize as optimize

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rc('axes', grid=False)

from pylab import *
import datetime as dt

import __builtin__ as builtin
set = builtin.set

import time

def setIDS(wavelength=229.68):
    """Determines settings for the IDS system. 

    Lambda is the desired wavelength in nm.  Normal value for CIII is 229.687.

    CIII:	229.68 nm, order=25, mi = 12
    He:		468.57 nm, order=12, mi = 4
    Hbeta:	486.14 nm, order=12 (from rydberg calculation, correcting for CM
            effect and refractive index of air)


    Output is three settings of the sin bar in the spectrometer - the best
    choice of order N and settings for N-1 and N+1.  The dispersion and grating
    angle are also calculated as the setting on the wavelength counter.  Not
    sure what the corrected setting is."""

    # nm to A
    wavelength = wavelength * 10
    # angle of incidence at zero order
    gamma = 4.98 * pi / 180

    # R2 grating tan(thg0)=2.000, blaze angle
    thg0 = 63.43 * pi / 180.
    # [nm] distance between grooves for 316 grooves/mm
    d = 1.0 / .0000316
    # blaze wavelength * N
    Nlam0 = 2. * d * sin(thg0) *cos(gamma)
    # output optics magnification (absolute value)
    mag = 3.7

    # fit parameters accounting for calibration of mechanical wavelength
    # counter
    a = -15.000 / 1.e4
    b =  8.355 / 1.e4


    # we calc order of the wavelength at a setting closest to the blaze angle
    N0 = round(Nlam0 / wavelength)
    Ns = [N0, N0-1, N0+2]
    allines = []

    for N in Ns:
        # calc grating angle 
        thg = arcsin(N * wavelength/ 2. / d / cos(gamma))
        # calc the dispersion
        dispersion = (1. / 1330.) * d / N * cos(thg+gamma)/mag	# [nm/mm]
        # calc the setting on the mech counter
        # d in this equation is 600 G/mm, or 1/.0000600 (A/groove)
        # thg - pi/6, which is the 30 degree correction due to the grating
        # mounting
        # wavelength setting for mechanical counter
        lmech = 2. * (1.E7/600.) * sin(thg - pi / 6.) * cos(gamma)

        # adjustement - not used
        lfit = wavelength * (1. + (a+b * (N * wavelength/d)))
        lset=2. * (1.0/.0000600) * cos(gamma) * sin(arcsin(N * lfit / 2. / d /
            cos(gamma)) - pi/6.)

        # calc other lines at this angle
        n = arange(1,32)
        allines.append((3*d*cos(gamma)) * sin(thg)/n / 10)

        print 'diffraction order: %i' % (N)
        print 'grating angle [deg]: %.4f' % (thg * 180. / pi )
        print 'dispersion [nm/mm]: %.7f' % (dispersion)
        print 'mech. counter wavelength [A]: %.1f' % (lmech)
        print 'corrected setting [A]: %.1f' % (lset)
        print '---'

    print "Other lines at this setting"
    print allines[0]

def setIDS2(m, wavelength):
    """Determines settings for the IDS system. 

    Lambda is the desired wavelength in nm and m is the desired order."""

    # nm to A
    wavelength = wavelength * 10
    # angle of incidence at zero order
    gamma = 4.98 * pi / 180	
    # R2 grating tan(thg0)=2.000, blaze angle
    thg0 = 63.43 * pi / 180.
    # [nm] distance between grooves for 316 grooves/mm
    d = 1.0 / .0000316
    # blaze wavelength * N
    Nlam0 = 2. * d * sin(thg0) *cos(gamma)
    # output optics magnification (absolute value)
    mag = 3.7

    # fit parameters accounting for calibration of mechanical wavelength
    # counter
    a = -15.000 / 1.e4
    b =  8.355 / 1.e4

    # we calc order of the wavelength at a setting closest to the blaze angle
    N = m
    # calc grating angle 
    thg = arcsin(N * wavelength/ 2. / d / cos(gamma))
    # calc the dispersion
    dispersion = (1. / 1330.) * d / N * cos(thg+gamma)/mag	# [nm/mm]
    # calc the setting on the mech counter
    # d in this equation is 600 G/mm, or 1/.0000600 (A/groove)
    # thg - pi/6, which is the 30 degree correction due to the grating mounting
    #
    # wavelength setting for mechanical counter
    lmech = 2. * (1.E7/600.) * sin(thg - pi / 6.) * cos(gamma)

    print 'diffraction order: %i' % (N)
    print 'grating angle [deg]: %.4f' % (thg * 180. / pi )
    print 'mech. counter wavelength [A]: %.1f' % (lmech)
    print '---'

def peakDetect(y):
    """Counts number of peaks"""
    ymax = max(y)
    q = where(y > 0.1 * ymax)
    q = q[0]
    if len(q) == 0:
        p = 0
        return p
    n = len(q)
    s = arange(len(y))
    q1 = q[0]+1
    qn = q[-1]
    for i in xrange(q1, qn+1):
        if (y[i] < y[i-1]):
            s[i] = -1
        if (y[i] > y[i-1]):
            s[i] = 1
    p = 0
    if (s[q1] == -1):
        p = 1
    for i in xrange(q1, qn+1):
        if (s[i] < 0) and (s[i-1] == 1):
            p = p+1
    if (s[qn] == 1):
        p = p+1

    return p


# Not used any more.  I can remove it.
# def bin(data, bintime = 1):
#     """Bin data"""
# 
#     t = data.ustime.reshape(400/bintime, bintime)
#     t = t.mean(1)
#     a = data.ussignal.reshape(16,400/bintime, bintime)
#     a = a.mean(2)
# 
#     return a, t

class ids(sdr.dtac_diag):
    """IDS data.

    There are some presets for arguments that will not show up in iPython if
    you look.  So they are detailed in full here:

        line - the line you are looking at.  The default is 'CIII', but other
          options are 'He' and 'Hbeta'.  Automatically sets wavelength, mi, and
          order.
        voltage - voltage of PMT power supply
        wavelength - wavelength of line
        mi - mass of impurity ion
        order - order of spectrometer
        fullgain - gain of PMT - shouldn't need to set
        bintime -  number of us you want to sum over.  Default is 1 (no summing).

    CIII:	229.687 nm, order=25, mi = 12
    He:		468.57 nm, order=12, mi = 4
    Hbeta:	486.14 nm, order=12 (from rydberg calculation, correcting for CM
            effect and refractive index of air)"""

    def __init__(self, shotname, **kw):

        self.dtac = True
        # set defaults
        self.line = None
        self.wavelength = 229.687
        self.mi = 12
        self.order = 25
        self.bintime = 1
        self.voltage = 800
        self.fullgain = 5.46e6
        
        # read in settings - these overwrite the above defaults.
        for k in kw.keys():
            setattr(self, k, kw[k])

        self.gain = gainCalc(self.voltage, self.fullgain)
        
        if self.line == "CIII":
            self.wavelength = 229.687
            self.mi = 12
            self.order = 25
        elif self.line == "He":
            self.wavelength = 468.57
            self.order = 12
            self.mi = 4
        elif self.line == "Hbeta":
            self.wavelength = 486.14
            self.order = 12
            self.mi = 4
        
        # in a more recent file, Chris had a valude of simply (2.e3) for this
        # number.  I'm not sure where it comes from or why he used a different
        # number.  I have not been able to track this down at all. For what
        # it's worth, the below number (1.e5 * 1.5e-2) = 1.5e3.
        # self.photons = (1.e5 * 1.5e-2) # converts to #ph/s

        # From Chris's code - need to divide signal through by this to get
        # micro amps - should give peak levels of around 500-1500 micro amps
        self.ampgain =  3.3e3 * 1e-6
        # the following are taken from Chris Cothran's version of the analysis.
        # The anode_calib numbers are from Hamamatsu's 'Anode Uniformity' chart
        # for 800 V operation.  The amp_calib number are from who knows where.
        # Only the middle 12 channels were set, so 1-10 and 23-32 were not.  I
        # just used the average values of the middle 12 (1.343) for the ones
        # that were not calibrated.
        self.anode_calib = array([
            0.833, 0.961, 0.932, 0.979, 0.943, 1.000, 0.909, 0.937,
            0.915, 0.944, 0.951, 0.921, 0.957, 0.933, 0.938, 0.901,
            0.951, 0.931, 0.977, 0.960, 0.899, 0.889, 0.927, 0.911,
            0.875, 0.848, 0.889, 0.860, 0.872, 0.829, 0.826, 0.827 ])
        self.amp_calib = array([
            1.343, 1.343, 1.343, 1.343, 1.343, 1.343, 1.343, 1.343,
            1.343, 1.343, 1.327, 1.364, 1.351, 1.384, 1.312, 1.305,
            1.302, 1.314, 1.377, 1.370, 1.361, 1.354, 1.343, 1.343,
            1.343, 1.343, 1.343, 1.343, 1.343, 1.343, 1.343, 1.343 ])

        # we only want/need the middle 16 channels for now.  Since this will
        # cost a lot of money to get more than 16 channels, I don't feel guilty
        # about hardcoding this in
        self.anode_calib = self.anode_calib[arange(16) + 8]
        self.amp_calib = self.amp_calib[arange(16) + 8]

        # stolen from sdr.dtac_diag.__init__
        self.shotname = shotname
        self.runSplit()
        self.probe = 'ids'
        self.filestrings = ['ids',]
        self.diagname = 'ids'
        self.settings = kw
        try:
            self.delays = sdr.srs_delays(shotname)
        except:
            self.delays = None
            self.delay_channel = None
            self.delay = [None, 0]
        # TODO put in exception handling here
        if not self.delays.type:
            self.delays = None
            self.delay_channel = None
            self.delay = ['T', 0.1]
        
        if self.filestrings:
            self._getData()
            self._makeFullData()
            self._processData()
            self._preProcess()
        

    def __str__(self):
        return "SSX ids: %s" % (self.shotname)

    def __repr__(self):
        if not self.filestrings:
            fs = None
        else:
            fs = self.filestrings
        return ("ids('%s', probe = '%s', filestrings = %s)" %
            (self.shotname, self.probe, fs))

    def _processData(self):
        """Preprocess the data - not the analysis.

        Preprocess data by reshaping it, applying calibrations, and binning it."""
        # decimate data to 1 us
        self.data = self.fullData.copy()

        # we need to take the mean over at least 1 us (10 samples when running
        # at 10 MHz) because that is the bandwidth of the system - mostly in
        # the cable capacitance, etc.
        d = self.data.copy() 
        d = d.reshape(d.shape[0], d.shape[1]/10, 10)
        dstd = d.std(2)
        d = d.mean(2)
        self.usstd = dstd
        self.usdata = d
        # self.photonbitnoise = self.bitnoise * self.photons
        self.ustime = self.time[::10]


        # apply calib factors - these are NOT applied to odata or fullData.
        self._applyCalib()
        # bin that shit into large bins, if we choose to.  Default is to not do
        # shit.
        self._binData()
        # automatically calculates deviations on the data
        self._calcDeviations()
        pass

    def _applyCalib(self):
        """Apply calibration factors to data.

        Accounts for channel variation on the PMT array and for the
        transimpedance amps.  The channel variation is from Hamamatsu's test
        sheet, while the gain factors on the amps come from an unknown source
        (via Chris Cothran's code)."""

        # this converts volts (per second) measured to anode current, in
        # micro Amps.  This is a channel by channel conversion, accounting for
        # the anode variation in the PMT tube as well as the overall
        # transimpedance amplifier gain and variance (amp_calib)
        self.voltsTouAmps = 1. / (self.anode_calib * self.amp_calib *
            self.ampgain)
        # this converts anode current in microamps PER microsecond to photons.
        # Takes the anode current (in uA), turns it into amps (1e-6), looks at
        # how much charge in 1 us (1e-6), then divides through by the PMT gain
        # to arrive at cathode current (self.gain), then divides through by
        # electron charge.  We are assuming (valid I think) that each electron
        # from cathode corresponds to one detected photon.  We do *not* need to
        # worry about quantum efficiency, since we aren't worried about photons
        # that we didn't detect.
        self.uAmpsToPhotons = 1e-6 * 1e-6 / (self.gain * 1.6022e-19)
        
        # Now apply these factors to the data and save the anode current (not
        # sure we will ever use this) and save the photons back to usdata
        self.data_anode = self.usdata * expand_dims(self.voltsTouAmps, 1)
        self.usdata = self.data_anode * expand_dims(self.uAmpsToPhotons, 1)

        # also apply these factors to the bitnoise calculated so we have it in
        # 'photons'
        self.photonbitnoise = (self.bitnoise * self.voltsTouAmps *
            self.uAmpsToPhotons)

    def _binData(self):
        """Bins the data.
        
        Default is by 1 us.  400 must be divisible by bintime.  Not sure if
        this works that well.  It shouldn't be needed for CIII."""

        self.odata = self.fullData.copy()
        self.otime = self.time.copy()
        bintime = self.bintime

        self.time = self.ustime.reshape(self.ustime.shape[0]/bintime, bintime)
        self.time = self.time.mean(1)

        d = self.usdata.copy()
        d = d.reshape(d.shape[0], d.shape[1]/bintime, bintime)
        d = d.sum(2)
        self.data = d.copy()

    def _calcDeviations(self):
        """Calculates deviations (sigma) on the data.

        it consists of three parts, all added in quadrature
        1 - sqrt(signal) from poisson statistics
        2 - bit noise, in photon counts - not sure if this should be squared,
          etc.
        3 - actual noise"""

        # calculate the deviations (sigma) on the data
        # it consists of three parts, all added in quadrature
        # 1 - sqrt(signal) from poisson statistics
        # 2 - bit noise, in photon counts - not sure if this should be squared,
        # etc.
        # 3 - actual noise from unnaccounted source - 8 photons seems to work
        # well signal is already 'squared' so no need to square it std of
        # signal at long times (zero signal) corresponds to about 1 bit
        # fluctuations - 2 photons.  
        # 2012-02-02 - changed it to 4 photons

        error2 = abs(self.data).T + self.photonbitnoise**2 + 4**2
        self.err = sqrt(error2.T)

    def _IDSfitGaussian(self, i, mu = None, sigma = None):
        """Gaussian fit for a line.

        Helper function.  Shouldn't need to call it directly."""
        out = {}
        x = self.xcoord
        q = self.channels
        kTinst = self.kTinst
        mi = self.mi

        # estimate mu, scale, and sigma

        if not mu:	
            mu = (x[len(x)/2] + x[len(x)/2 + 1]) / 2
        scale = max(self.data[q,i])
        # scale = 10
        if not sigma:
            # meanx = mean(x)
            # sigma = sqrt( sum((x - meanx)**2)/(len(x)-1))
            sigma = 20
        p0 = (mu, sigma, scale)
        # print i, p0

        pfit, cov, chi2, out['success'] = mf.curveFit(mf.pGaussian, x[q],
            self.data[q,i], p0, sigma = self.err[q,i])

        # fitData, pfit, cov, chi2, out['success'] = mf.fitGaussian(x[q],
        #     self.signal[q,i])

        k = ['mu', 'sigma', 'scale']
        kErr = ['muErr', 'sigmaErr', 'scaleErr']

        if out['success'] == 1 and (type(cov) == type(empty(0))):
            v = pfit
            # calculate the error from the covariance matrix - error for each
            # param is the sqrt of the diagonals
            vErr = []
            for i in xrange(3):
                vErr.append(sqrt( cov[i][i] ))

            # generate the fitted data and plot
            out['xfit'] = arange(x[q[0]], x[q[-1]])
            out['xfit'] = arange(x[q[0]], x[q[-1]])
            out['yfit'] = mf.pGaussian(out['xfit'], *pfit)
        else:
            v = [0] * 3
            vErr = v

        d = dict(zip(k,v))
        dErr = dict(zip(kErr,vErr))
        out.update(d)
        out.update(dErr)

        out['errConst'] = chi2

        # calculate dv and kT + errors from the fitting output.
        # dv is merely the center of the gaussian, while kT is related to the
        # width.  
        # subtract the inst temp off the value for kT

        out['dvFit1'] = out['mu']
        out['dvErr1'] = out['muErr']
        out['kTFit1'] = (out['sigma']/3e5)**2 * (mi * .94e9) - kTinst
        out['kTErr1'] = abs(2 * out['sigmaErr'] * out['sigma'] * ((1/3e5)**2 *
            (mi * .94e9)))

        return out

    def _IDSfitDoubleGaussian(self, i, mu = None):
        """Double Gaussian fit for a line.

        Helper function.  Shouldn't need to call it directly."""
        out = {}
        x = self.xcoord
        q = self.channels
        kTinst = self.kTinst
        mi = self.mi

        # esitmate mu, scale, and sigma
        if not mu:
            mu = (x[len(x)/2] + x[len(x)/2 + 1]) / 2
        mu1 = mu + 10
        mu2 = mu - 10
    	# scale1 = max(self.signal[q,i])
    	# scale2 = scale1
        scale1, scale2 = 10, 10
    	# meanx = mean(x)
    	# sigma1 = sqrt( sum((x - meanx)**2)/(len(x)-1))
    	# sigma2 = sigma1 

        sigma1, sigma2 = 10, 10
        p0 = (mu1, sigma1, scale1, mu2, sigma2, scale2)

        pfit, cov, chi2, out['success'] = mf.curveFit(mf.pDoubleGaussian, x[q],
            self.data[q,i], p0, sigma = self.err[q,i])

        # fitData, pfit, cov, chi2, out['success'] = mf.fitDoubleGaussian(x[q],
        #     self.signal[q,i], mu = mu, positive = True)

        k = ['mu1', 'sigma1', 'scale1', 'mu2', 'sigma2', 'scale2']
        kErr = ['mu1Err', 'sigma1Err', 'scale1Err', 'mu2Err', 'sigma2Err',
            'scale2Err']

        if out['success'] == 1 and type(cov) == type(empty(0)):
            v = pfit
            # calculate the error from the covariance matrix - error for each
            # param is the sqrt of the diagonals
            vErr = []
            for i in xrange(6):
                vErr.append(sqrt( cov[i][i] ))

            # generate the fitted data and plot
            out['xfit'] = arange(x[q[0]], x[q[-1]])
            out['yfit'] = mf.pDoubleGaussian(out['xfit'], *pfit)
            out['yfit1'] = mf.pGaussian(out['xfit'], *pfit[:3])
            out['yfit2'] = mf.pGaussian(out['xfit'], *pfit[3:])
            out['area1'] = sp.trapz(out['yfit1']) / sp.trapz(out['yfit'])
            out['area2'] = sp.trapz(out['yfit2']) / sp.trapz(out['yfit'])
        else:
            v = [0]*6
            vErr = v
            out['area1'], out['area2'] = 1,1

        d = dict(zip(k,v))
        dErr = dict(zip(kErr,vErr))
        out.update(d)
        out.update(dErr)

        out['errConst'] = chi2

        out['dvFit1'] = out['mu1']
        out['dvErr1'] = out['mu1Err']
        out['kTFit1'] = (out['sigma1']/3e5)**2 * (mi * .94e9) - kTinst
        out['kTErr1'] = abs(2 * out['sigma1Err'] * out['sigma1'] * ((1/3e5)**2
            * (mi * .94e9)))

        out['dvFit2'] = out['mu2']
        out['dvErr2'] = out['mu2Err']
        out['kTFit2'] = (out['sigma2']/3e5)**2 * (mi * .94e9) - kTinst
        out['kTErr2'] = abs(2 * out['sigma2Err'] * out['sigma2'] * ((1/3e5)**2
            * (mi * .94e9)))

        return out

    def plotRawIDS(self):
        """Plots all raw IDS channels at all times."""
        fig = figure(1, **ssxdef.f)
        fig.clear()
        numChannels = self.data.shape[0]
        for i in xrange(numChannels):
            plot(self.time, self.data_anode[i])
        ylabel('anode current (uA)')
        xlabel('time (us)')

    def im(self, timerange = [20, 80], plotLog = False, interp = 'nearest',
        saveFig = False, fig = 22, ext = 'png', **kw):
        """Plots all IDS channels asfig a 2D array image.

        Data does NOT need to be processed by processIDS() before being plotted
        by this method."""
        t = self.time
        p = self.data
        if plotLog:
            p = log10(p)
        t0 = timerange[0] + 20
        t1 = timerange[1] + 20
        
        exts = (timerange[0], timerange[1], self.xcoord[0], self.xcoord[-1])
        # exts = (timerange[0], timerange[1], 1, self.numChans + 1)

        fig1 = figure(fig)
        fig1.clear()
        imshow(p[:, t0:t1], aspect='auto', origin='lower', interpolation=interp,
            extent=exts, **kw)
        
        xlabel(r'time (us)')
        ylabel(r'velocity (km/s)')
        titlestr = "IDS - {} - {} nm".format(self.shotname, self.wavelength)
        title(titlestr)
        colorbar()
        # figlabel()
        
        if saveFig:
            if plotLog:
                ll = '-log'
            else:
                ll = ''
            fn = self.shotname + '-ids-raw' + ll + '.' + ext
            fig1.savefig(fn)

    def im3(self, timerange = [20, 80], plotLog = False, saveFig = False,
        fig = 23, ext = 'png', **kw):
        """Plots all IDS channels as a 3D waterfall image.

        Data does NOT need to be processed by processIDS() before being plotted
        by this method."""
        t = self.time
        p = self.data
        if plotLog:
            p = log10(p)
        t0 = timerange[0] + 20
        t1 = timerange[1] + 20

        X, Y = meshgrid(t, arange(16) + 1)

        fig1 = plt.figure(fig)
        fig1.clear()
        plt.subplots_adjust(0,0,1,1)
        ax = fig1.add_subplot(111, projection = '3d', frame_on=False)
        ax.plot_surface(X[:, t0:t1], Y[:, t0:t1], p[:, t0:t1], rstride = 1,
            cstride = 1, cmap=cm.Oranges, **kw)
        ax.view_init(45, 235)
        ax.set_xlabel(r'time (us)')
        ax.set_ylabel(r'channel')
        ax.set_zlabel(r'counts')
        titlestr = "IDS - {} - {} nm".format(self.shotname, self.wavelength)
        ax.set_title(titlestr)

        show()
        if saveFig:
            if plotLog:
                ll = '-log'
            else:
                ll = ''
            fn = self.shotname + '-ids-raw-3d' + ll + '.' + ext
            fig1.savefig(fn)
        return ax
    
    def plotRawIDStp(self, ts):
        """Plots all raw IDS channels at a given time."""
        tp = int((ts - (self.time[0]))*1./self.bintime)
        fig = figure(11, **ssxdef.f)
        fig.clear()
        numChannels = self.data.shape[0]
        plot(self.data[:,tp], 'bo')
        plot(self.data[:,tp], '--', color='gray')
        axis = fig.axes[0]
        text(.05,.95, r"t = %3.1f $\mu$s" % (ts), horizontalalignment='left',
            verticalalignment='top', transform = axis.transAxes)
        ylabel('photons')
        xlabel('channel')

    def _preProcess(self, dv0 = 0):
        """Set up some equipment specific variables."""

        numChan, numPts = self.data.shape
        self.dv0 = dv0
        
        # spectrometer specs
        # [mm] focal length
        fl = 1330.
        # [nm] groove spacing
        d = 1.e6 / 316.
        # [radians] czerny-turner internal angle
        gamma = 4.98 * pi/180.
        # output optics magnification
        mag = 3.7
        # [km/s] speed of light
        c = 3e5

        # instrument temperature pre calculations
        # [eV] measured for CIII line at 229.7
        kTinst = 3.4
        # [nm] CIII line
        lam = 229.687
        # diffraction order for this line
        n = 25.

        thc = arcsin(n * lam / (2. * d * cos(gamma)))
        alpha = thc - gamma
        beta = thc + gamma

        # focal plane calib for CIII
        dvdx = c * (1./mag) * (1./fl) * cos(beta) / (sin(alpha) + sin(beta)) 

        # anamorphic magnification; ie, the ratio of the slit width image (at
        # FP) to the slit width
        anamag = cos(alpha) / cos(beta)

        dxinst = sqrt(kTinst/ ( 12. * 0.940e9)) / dvdx / anamag	# [mm]

        # now onto parameters from this run
        N = self.order
        thc = arcsin(N * self.wavelength/ ( 2. * d * cos(gamma)))
        alpha = thc - gamma
        beta = thc + gamma

        # [(km/s)/mm] focal plane calibration
        dvdx = c * (1./mag) * (1./fl) * cos(beta) / (sin(alpha) + sin(beta))

        # anamorphic magnification; ie, the ratio of the slit width image (at
        # FP) to the slit width
        anamag = cos(alpha) / cos(beta)

        dxinst = dxinst * anamag
        # [eV] instrument temperature for mi
        kTinst = (dxinst * dvdx)**2 * (self.mi * 0.940e9)

        self.kTinst = kTinst

        xscale = dvdx
        xcoord = xscale * (arange(numChan, dtype = 'float') - (numChan/2 -
            .5)) - dv0
        self.xcoord = xcoord
    
    def processIDS(self, dv0 = 0, tracescale = 1, ts = None, plotErr = False,
        showPlot = True, times = (30,81), writeFiles = False, writeData =
        False, printFits = False, forceOne = False):
        """Processes the IDS raw data.

        Applies gaussian and double gaussian fits to the data.  Must be called
        manually.  The parameter dv0 sets the offset of the zero velocity
        channel."""
        
        if dv0:
            self._preProcess(dv0)
        xcoord = self.xcoord
        kTinst = self.kTinst

        # internal switches...
        # plot moment points?
        momPlot = False	
        # plot bad SNR, good fit points?
        maybePts = False #True
        # percent of kTFit kTErr is allowed to be for maybe pts
        maybeThresh = .5
        maybeThreshLo = .1
        # max kTErr is allowed to be for maybe pts
        maybeThreshEV = 20

        numChan, numPts = self.data.shape
        tpts = zeros(numPts)
        strength = ma.zeros(numPts)
        dvMom = ma.zeros(numPts)
        dvFit = ma.zeros(numPts)
        dvErr = ma.zeros(numPts)
        kTMom = ma.zeros(numPts)
        kTFit = ma.zeros(numPts)
        kTErr = ma.zeros(numPts)
        chi1 = ma.zeros(numPts)
        chi2 = ma.zeros(numPts)
        kTFit1 = ma.zeros(numPts)
        kTFit2 = ma.zeros(numPts)
        kTErr1 = ma.zeros(numPts)
        kTErr2 = ma.zeros(numPts)
        dvFit1 = ma.zeros(numPts)
        dvFit2 = ma.zeros(numPts)
        sigQuality = ma.zeros(numPts)
        fitparams = []
        area1 = ma.zeros(numPts)
        area2 = ma.zeros(numPts)
        kurt = ma.zeros(numPts)
        skew = ma.zeros(numPts)

        # set up some masks
        kTMom.mask = ma.masked
        dvMom.mask = ma.masked
        strength.mask = ma.masked
        chi1.mask = ma.masked
        chi2.mask = ma.masked
        kurt.mask = ma.masked
        skew.mask = ma.masked
    	# sigQuality.mask = ma.masked

        peaks = zeros(numPts, dtype='i8')

        # set up some time stuff
        # TODO fix this time shit
        time = arange(0., 2000, self.bintime * 10)/10. - 20
        a = time >= times[0]
        b = time < times[1]
        tq = where(-a-b)[0]

        # plotting stuff
        for i in tq:

            # smaller channel number corresponds to longer wavelengths (with
            # grating angle fixed) this means plasma is moving away from the
            # detector; assign this a negative velocity. the zero of the
            # dispersive coordinate is at the boundary of channels 16 and 17 by
            # default

            tpts[i] = self.time[i]
            # q is the active ids channels

            #channels 3-14	
            # for the scopes
            if not self.dtac:
                q = arange(12)+2
            else:
                # for dtacq
                q = arange(16)
            self.channels = q

            # select the range of channels for use in calculating the width by
            # computing the moment. the range eliminates those channels on the
            # edges of the spectrum with small or no signal. fluctuation at
            # these large distances from the mean cause the temperature to be
            # too big.
            # 6/3/09 - set the limit not as a ratio to the max signal
            #   but to 20 photons
            # 2012-01-25 - set the limit to be based not on photons (which we
            #   can't necessarily trust) but instead base it on anode current.
            #   We used to have the limit set at 20 photons, and the conversion
            #   from the old photon number ot anode current is approx .13 so
            #   we'll just use 20 * 0.13 = 2.6 as our number.
            # 
            #   The .13 comes from 1 / self.photons / self.amp_calib.mean() /
            #   self.ampgain * 0.85.  Where 0.85 is near the lower end of the
            #   anode_calib array.
            q2 = where(self.data[:,i] > 2.6)[0]
            xcoord2 = xcoord[q2]
            # calculate the first three moments 
            m0 = mf.integrate.trapz(self.data[q2,i], xcoord2)
            m1 = mf.integrate.trapz(self.data[q2,i]*xcoord2, xcoord2)
            m2 = mf.integrate.trapz(self.data[q2,i]*xcoord2**2, xcoord2)
            m3 = mf.integrate.trapz(self.data[q2,i]*xcoord2**3, xcoord2)
            m4 = mf.integrate.trapz(self.data[q2,i]*xcoord2**4, xcoord2)

            # 	print m2
            # 	print m0
            strength[i] = m0
            dvMom[i] = m1/m0
            kTMom[i] = (m2/m0 - (m1/m0)**2)/(3e5)**2*(self.mi * .94e9)-kTinst
            # 	print m1/m0, (m2/m0 - (m1/m0)**2)/(3e5)**2*(mi * .94e9)-kTinst
            # 	print
            sigmom = sqrt((m2/m0 - (m1/m0)**2))
            skew[i] = m3/m0/sigmom**3
            kurt[i] = m4/m0/sigmom**4 - 3

            if m0 == 0:
                dvMom[i] = 0.0

            out = self._IDSfitGaussian(i, dvMom[i])
            out['fit'] = 'single'

            out2 = self._IDSfitDoubleGaussian(i, dvMom[i])
            muepsilon = 1
            if ((out['errConst']) > .15) and (out2['errConst'] > .75 *
                out['errConst']):
                muepsilon = muepsilon + 1
                out2 = self._IDSfitDoubleGaussian(i, dvMom[i] + muepsilon)

            chi1[i] = out['errConst']
            chi2[i] = out2['errConst']
            area1[i] = out2['area1']
            area2[i] = out2['area2']


            dvFit[i] = out['dvFit1']
            dvErr[i] = out['dvErr1']
            kTFit[i] = out['kTFit1']
            kTErr[i] = out['kTErr1']

            kTFit1[i] = out2['kTFit1']
            kTFit2[i] = out2['kTFit2']
            kTErr1[i] = out2['kTErr1']
            kTErr2[i] = out2['kTErr2']
            dvFit1[i] = out2['dvFit1']
            dvFit2[i] = out2['dvFit2']
            fitparams.append({'out': out, 'out2': out2})		

            # need to work on error stuff.
            # 	print sigma
                # err = (2 * sigma  *(1/3.e5)**2 *(mi*.94e9)) * sqrt(cov[1][1])
                # * sqrt(chi2/12)
            # 	print err
            if printFits:
                print "time: %i us" % (self.time[i])
                print ("From fit: dv = %.2f +- %.2f km/s and kT = %.2f +- "
                    "%.2f eV" % (dvFit[i], dvErr[i], kTFit[i], kTErr[i]))
                print "From mom: dv = %.2f km/s and kT = %.2f eV" % (dvMom[i],
                    kTMom[i])
                print 

            # this might not be the best way to do this
            try:
                peaks[i] = peakDetect(self.data[q,i])
            except:
                peaks[i] = 0
            # need to redo the conversion factor - don't we want to integrate
            # the signal?
            # converts to #ph/s
    		# sigQuality[i] = max(self.data[q,i]) * (1.e5 * 1.5e-2)
            # converts to #ph/s
            sigQuality[i] = sp.trapz(self.data[q,i]) #* (1.e5 * 1.5e-2)

        if writeData:
            fName3 = ssxutil.ssxPath('', 'output', self.runYear + '/' +
                self.runDate + '/' + run + '/ids/')
            fName3 = os.path.join(fName3, 'idsout.txt')
            j = where(tpts)
            outdata = array([tpts[j], kTFit[j], kTErr[j], dvFit[j], dvErr[j],
                kTMom[j], dvMom[j], strength[j],peaks[j]])
            ssxutil.write_data(fName3, outdata,
                "#time\tkTFit\tkTErr\tdvFit\tdvErr" + 
                "\tkTMom\tdvMom\tstrength\tpeaks")

        # figure out where the fit is good and where it sucks and deterimine
        # the indices
        zp = zeros(len(peaks), dtype = 'i8')

        # This whole section could be done with masked arrays.  In some ways it
        # might be more straightforward to do that. However, in some places it
        # might be tricky and more importantly, this seems to work.  So we are
        # keeping it for now.

        # calc good points
        # where there is one peak
        good = where(kTFit > 0, peaks, 0)
        good = where(kTFit < 200, good, 0)
        # and the error is below the good threshold
    	# good = where(kTErr <= maybeThreshLo*kTFit, good, 0)


        # double fit points
        # more than one peak
        tqDouble1 = where(peaks > 1, 1, 0)
        # where the fit is better than the one peak fit
        tqDouble2 = where((chi2 < .5 * chi1), 1, 0)
        # make sure the error is big on the one gaussian fit points, otherwise
        # drop them out
        tqDouble3 = where((abs(kTErr) > maybeThreshLo * kTFit), 1, 0)
        # or where double fit is WAY better than single fit
        tqDouble4 = where((chi2 < 1 * chi1), 1, 0)
        # take the union of the above sets - 2 and 3 are multiplied together
        # because they both have to be true (think of '+' as an 'or' and '*' as
        # an 'and')
        tqDouble = tqDouble1 + tqDouble2 * tqDouble3 + tqDouble4
        tqDouble = where(tqDouble > 0, peaks, 0)

        # filter out those points where the sum of the two temps is way larger
        # than the one temp fit (and there is a one temp fit)
        # 5/26/09 remove
    	# tqDouble5 = where(chi1 < .12 , peaks, 0) 
        # tqDouble5 = where((abs(kTFit1) + abs(kTFit2) > abs(kTFit)*6),
        #     tqDouble5, 0) 
    	# tqDouble = where(tqDouble5 == 0, tqDouble, 0)


        kTErr2 = ma.masked_invalid(kTErr2)
        # if fit 1 is really good, stick with it.
        # 2012-02-02 - changed chi1 limit to 4
        tqDouble = where(chi1 < 4, 0, tqDouble)
        # make sure our double fit error is small
        tqDouble = where((abs(kTErr1) < .8 * abs(kTFit1)), tqDouble, 0) 
        tqDouble = where((abs(kTErr2) < .8 * abs(kTFit2)), tqDouble, 0) 
        tqDouble = where((abs(kTFit2) < 200), tqDouble, 0) 
        tqDouble = where((abs(kTFit1) < 200), tqDouble, 0) 
        # where the fits are at least positive
        tqDouble = where(kTFit1 > 0, tqDouble, 0)
        tqDouble = where(kTFit2 > 0, tqDouble, 0)
    	# tqDouble = where((abs(kTFit1) < 200), tqDouble, 0) 
    	# tqDouble0 = where((abs(kTFit2) < 200), tqDouble, 0) 
        biga = where(area1 > area2, area1, area2)
        smalla = where(area1 < area2, area1, area2)
        tqDouble0 = where(smalla < .1, 0, tqDouble)
    	# tqDouble0 = tqDouble
        # use fits where there are more than one peak due to bad SNR.
        if maybePts: 
            # where there is one peak
            maybe = where(peaks == 1, peaks, peaks)
            # all points where the error is above the good threshold
            maybe = where(abs(kTErr) > maybeThreshLo * kTFit, peaks, 0)
            # but below the maybe threshold
            maybe = where(abs(kTErr) < maybeThresh * kTFit, maybe, 0)
            maybe = where(abs(kTErr) < maybeThreshEV, maybe, 0)
            # and where the fit is better than the double peaked fit
    		# maybe = maybe - tqDouble0

            maybe = where(peaks == 1, peaks, 0)
            maybe = where(chi1.data > 20, maybe, 0)
            tqMaybe = where(maybe)[0]
        else:
            maybe = zp
            tqMaybe = zp

        # calc bad points
        bad = zp
        # find points where the error is above the error set at the top of the
        # routine
        bad1 = where((kTFit > 5) & ( kTErr > 1 ) & ( kTErr > maybeThresh *
            kTFit), peaks, 0)
        bad2 = where(kTErr1 > maybeThresh * kTFit1, tqDouble0, 0)
        bad3 = where(kTErr2 > maybeThresh * kTFit2, tqDouble0, 0)
    	# bad = bad1 * (bad2 + bad3)
        # filter out no signal points just so as to not clutter up 'bad'
    	# sq = where(sigQuality != 0, sigQuality, zp + 1000)
        # pick points that have too few photons and label as bad.  If there are
        # enough photons, then set it to bad above - which is already 1 if the
        # error is too big or zero if it's ok
    	# bad = where(sq < 200, 1, bad1)
        # #signal quality for each hump.  lets set it to at least 150 photons
        # for each
    	# sq1 = sq * area1
    	# sq2 = sq * area2
    	# bad1 = where(sq1 < 150, 1, 0)
    	# bad2 = where(sq2 < 150, 1, 0)
    	# bad2 = bad2 + bad1
        bad2 = bad2 + bad3
        bad = bad1
        tqBad = where(bad)[0]

        # filter out the bad points
        tqDouble0 = where(bad2==0, tqDouble0, 0)
        maybe = where(bad==0, maybe, 0)
        good = where(bad==0, good, 0)
        # filter out double and maybe points
        if forceOne:
            tqDouble0 = zp
        else:
            maybe = where(tqDouble0==0, maybe, 0)
            good = where(tqDouble0 == 0, good, 0)
            good = where(maybe == 0, good, 0)

        # indices of the points we want
        tqDouble = where(tqDouble0)[0]
        tqGood = where(good)[0]
        tqMaybe = where(maybe)[0]

        # masks are inverted (1 where we don't want it, 0 where we do)
        kTFit.mask = where(good, False, True)
        kTErr.mask = kTFit.mask.copy()
        dvFit.mask = kTFit.mask.copy()
        dvErr.mask = kTFit.mask.copy()

        kTMaybe = kTFit.copy()
        kTMaybeErr = kTErr.copy()
        kTMaybe.mask = where(maybe, False, True)
        kTMaybeErr.mask = kTMaybe.mask.copy()

        kTFit1.mask = where(tqDouble0, False, True)
        kTErr1.mask = kTFit1.mask.copy()
        kTFit2.mask = kTFit1.mask.copy()
        kTErr2.mask = kTFit1.mask.copy()
        dvFit1.mask = kTFit1.mask.copy()
        dvFit2.mask = kTFit1.mask.copy()
        area1.mask = kTFit1.mask.copy()
        area2.mask = kTFit1.mask.copy()

        sigQuality = ma.masked_less_equal(sigQuality, 0)

        # save fits to data
        (self.kTFit, self.kTErr, self.dvFit, self.dvErr, self.kTFit1,
            self.kTErr1, self.kTFit2, self.kTErr2, self.dvMom) = (kTFit, kTErr,
                dvFit, dvErr, kTFit1, kTErr1, kTFit2, kTErr2, dvMom)

        self.kTMaybe, self.kTMaybeErr = kTMaybe, kTMaybeErr
        self.kTMom = kTMom
        self.dvFit1, self.dvFit2 = dvFit1, dvFit2

        self.kurt = kurt
        self.skew = skew

    # 	self.fits = {'kTFit': kTFit, 'kTFit1': kTFit1, 'kTFit2': kTFit2, 'dvFit': dvFit, 'dvMom': dvMom, 'kTMom': kTMom}
    # 	self.error = {'kTErr': kTErr, 'kTErr1': kTErr1, 'kTErr2': kTErr2, 'dvErr': dvErr}
        self.chi1, self.chi2 = chi1, chi2

        # we are going to keep the indices of where things are fitted for two
        # reasons.  One is that it might be useful for plotting of maybe fits
        # so that we don't have to have another array for them.  Second, just
        # in case.  It really doesn't get in the way.
        self.q = {'good': tqGood, 'double': tqDouble, 'maybe': tqMaybe, 'bad':
            tqBad, 'time': tq}

        self.area1, self.area2 = area1,area2
        self.fitparams = fitparams
        self.strength = strength
        self.peaks = peaks
        self.sigQuality = sigQuality
        self.xscale = xscale


    def plotSig(self, writeFiles = False, ext = 'pdf', pubFig = False,
        writeFilesHere = False, fig = 4, **kw):
        """Plots total signal as a function of time."""
        fig = figure(fig, **ssxdef.f)
        subplot(2,1,1)
        semilogy(self.time, self.sigQuality, hold=0, **kw)
        ylabel('summed anode current (uA)')
        title(self.shotname)
        
        subplot(2,1,2)
        semilogy(self.time, self.strength, 'g-', hold=0)
        ylabel('line strength (arb.)')
        xlabel('Time (us)')
        
        if writeFiles:
            dirName = ssxutil.ssxPath('', 'output', self.runYear + '/' +
                self.runDate + '/' + self.shotname + '/ids/')
            os.spawnlp(os.P_WAIT, 'mkdir', 'mkdir', '-p', dirName)
            fName = os.path.join(dirName, 'sigquality.' + ext)
            fig.savefig(fName)
            close(fig)
        if writeFilesHere:
            fName = "%s-sigquality.%s" % (self.shotname, ext)
            fig.savefig(fName)
            close(fig)

    def plotIDStp(self, ts, writeFiles = False, plotErr = True, force = None,
        printError = True, maybePts = True, ext = 'pdf', style = None, pubFig =
        False, dvshift = 0, writeFilesHere = False):
        """Plots IDS with fit at specified timepoint.

        Plots the IDS data with a fit at the specified timepoint.  Inputs are
        as follows:
          - ts - timepoint to plot - required
          - force - 1 or 2.  Determines which fit is plotted.  If empty, the
            'best' fit is used."""
        dv0 = self.dv0

        tf = int((ts - (self.time[self.q['time'][0]]))*1./self.bintime)
        i = self.q['time'][tf]
        tp = self.time[i] # for printing purposes 
        q = arange(self.data.shape[0])
        xscale = self.xscale
        xcoord = self.xcoord

        out = self.fitparams[tf]['out']
        out2 = self.fitparams[tf]['out2']

        kTFit, kTErr, dvFit, dvErr, kTFit1, kTErr1, kTFit2, kTErr2, dvMom = self.kTFit, self.kTErr, self.dvFit, self.dvErr, self.kTFit1, self.kTErr1, self.kTFit2, self.kTErr2, self.dvMom 

        if printError:
            print "chi1 = %f\t chi2 = %f" % ( out['errConst'], out2['errConst'])
            print "peaks = %i" % self.peaks[i]

        if pubFig:
            fig = ssxdef.pubfig(2)
        else:
            fig = figure(1, **ssxdef.f)
            fig.clear()
        if plotErr:
            # doesn't work at all - don't use for now
            plot(xcoord[q] + dvshift,self.data[q,i], linestyle='None',
                color='gray')
            plot(xcoord[q] + dvshift,self.data[q,i], 'wo', ms = 4)
            errorbar(xcoord[q] +
                dvshift,self.data[q,i],self.err[q,i],fmt=None, ecolor =
                'red')
        else:
            if style == 'step':
                step(xcoord[q], self.data[q,i], where = 'mid', color='gray')
            else:
                plot(xcoord[q],self.data[q,i], ':', color='gray')
                plot(xcoord[q],self.data[q,i], 'wo', ms = 4)

        axis = fig.axes[0]

        if force == 0:
            fit = 0
        elif ((out2.has_key('yfit1') and out2.has_key('yfit2') and force != 1)
            and (~self.kTFit1.mask[i] or force == 2)):
            fit = 2
        elif ((out.has_key('yfit') and force != 2) and (~self.kTFit.mask[i] or
            (maybePts and ~self.kTMaybe.mask[i]) or (force == 1))):
            fit = 1
        else:
            fit = 0

        if fit == 1:
            plot(out['xfit'] + dvshift,out['yfit'],'-', color = 'blue')
    # 		ylim(0,)
            text(.62,.95, r"$\Delta$v = %.1f $\pm$ %.1f km/s" % (dvFit.data[i]
                + dvshift, dvErr.data[i]), horizontalalignment='left',
                verticalalignment='top', transform = axis.transAxes)
            text(.62,.9, r"kT = %.1f $\pm$ %.1f eV" % (kTFit.data[i],
                kTErr.data[i]), horizontalalignment='left',
                verticalalignment='top', transform = axis.transAxes)
            if kTFit.mask[i]:
                text(.62, .85, 'forced fit - no good',
                    horizontalalignment='left', verticalalignment='top',
                    transform = axis.transAxes)
            if printError:
                print "photons = %.1f" % (self.sigQuality[i])
        elif fit == 2:
            plot(out2['xfit'] + dvshift,out2['yfit'],'-', color = 'blue')
            plot(out2['xfit'] + dvshift,out2['yfit1'],'-.', color = 'gray')
            plot(out2['xfit'] + dvshift,out2['yfit2'],'-.', color = 'gray')
            text(.62, .95, r"kT1 = %.1f $\pm$ %.1f eV" % (kTFit1.data[i],
                kTErr1.data[i]), horizontalalignment='left',
                verticalalignment='top', transform = axis.transAxes)
            text(.62, .9, r"kT2 = %.1f $\pm$ %.1f eV" % (kTFit2.data[i],
                kTErr2.data[i]), horizontalalignment='left',
                verticalalignment='top', transform = axis.transAxes)	
            if kTFit1.mask[i]:
                text(.62, .85, 'forced fit - no good',
                    horizontalalignment='left', verticalalignment='top',
                    transform = axis.transAxes)
            if printError:
                print "photons = %.1f + %.1f" % ( out2['area1'] *
                    self.sigQuality[i], out2['area2'] * self.sigQuality[i])
        text(.05,.95, r"t = %3.1f $\mu$s" % (tp), horizontalalignment='left',
            verticalalignment='top', transform = axis.transAxes)
        ylim(0,)
        ylabel('photons')
        xlabel('velocity (km/s)')

        title(self.shotname)

        if writeFiles:
            dirName = ssxutil.ssxPath('', 'output', self.runYear + '/' +
                self.runDate + '/' + self.shotname + '/ids/' + ext)
            os.spawnlp(os.P_WAIT, 'mkdir', 'mkdir', '-p', dirName)
            fName = "f%03.1f.%s" % (tp, ext)
            fName = os.path.join(dirName,fName)
            fig.savefig(fName)
            close(fig)
        if writeFilesHere:
            fName = "%sf%03.1f.%s" % (self.shotname, tp, ext)
            fig.savefig(fName)
            close(fig)

    def plotIDS(self, writeFiles = False, plotAll = False, maybePts = False,
        momPlot = False, ms = 4, showLegend = False, ymax = None):
        """Plot fitted IDS data as a function of time."""

        # 	momPlot = False	

        t = self.time

        if writeFiles:
            plotAll = True

        # figures
        figs = {}
        # plot kTi
        f2 = figure(2, **ssxdef.f)
        figs['kTi'] = f2
        f2.clear()

        if momPlot:
            plot(t, self.kTMom, ls = ':', color = 'gray', marker = '.', mfc =
                'black', mec = 'black', label='2nd moment') #
    # 		plot(t, self.kTMom, 'k.', ms = ms) #

        errorbar(t, self.kTFit, self.kTErr, fmt=None, ecolor='gray')
        plot(t, self.kTFit, 'kx', ms = ms, label ='single gaussian')

        errorbar(t, self.kTFit1, self.kTErr1, fmt=None, ecolor='gray')
        errorbar(t, self.kTFit2, self.kTErr2, fmt=None, ecolor='gray')
        plot(t, self.kTFit1, 'rx', ms = ms, label ='double gaussian')
        plot(t, self.kTFit2, 'rx', ms = ms)

        if maybePts:
            errorbar(t, self.kTMaybe, self.kTMaybeErr, fmt=None, ecolor='blue')
            plot(t, self.kTMaybe, 'kx', ms = ms)


        title(self.shotname)
        xlabel('Time (us)')
        ylabel('kT_i (eV)')
        xlim(self.time[self.q['time'][0]], self.time[self.q['time'][-1]]+1)


    # find the max 
        if ymax:
            ylim(0,ymax)
        else:
            ylim(0,)

        if showLegend:
            legend(borderpad=1,handletextpad=1, borderaxespad=1, numpoints=1)


        if plotAll:
            # dv plot
            dv0 = self.dv0
            f3 = figure(3, **ssxdef.f)
            f3.clear()
            figs['dv'] = f3
            plot(t, self.dvMom,  ls = ':', color = 'gray', marker = '.', mfc =
                'black', mec = 'black', label='1st moment')

            errorbar(t, self.dvFit, self.dvErr, fmt=None, ecolor='gray')
            plot(t, self.dvFit, 'ko', ms = ms, label ='single gaussian')
            plot(t, self.dvFit1, 'ro', ms = ms, label ='double gaussian')
            plot(t, self.dvFit2, 'ro', ms = ms)

            xlabel('Time (us)')
            ylabel('dv (km/s)')
            title(self.shotname)
            if showLegend:
                legend(borderpad=1,handletextpad=1, borderaxespad=1,
                    numpoints=1)		


            f4 = figure(4, **ssxdef.f)
            figs['sigquality'] = f4
            semilogy(t, self.sigQuality, hold=0)
            ylabel('Counts (# photons/us)')
            xlabel('Time (us)')
            title(self.shotname)

            f5 = figure(5, **ssxdef.f)
            figs['linestrength'] = f5
            semilogy(t, self.strength, 'g-', hold=0)
            ylabel('line strength (arb.)')
            xlabel('Time (us)')
            title(self.shotname)

        if writeFiles:
            dirName = ssxutil.ssxPath('', 'output', self.runYear + '/' + self.runDate + '/' + self.shotname + '/ids/')
            os.spawnlp(os.P_WAIT, 'mkdir', 'mkdir', '-p', dirName)
            for fig in figs.iterkeys():
                fName = ssxutil.ssxPath(fig+'.pdf', 'output', self.runYear + '/' + self.runDate + '/' + self.shotname +'/ids/')
                figs[fig].savefig(fName)

def gainCalc(voltage, fullgain = 5.46e6):
    """Calculates the gain based on the PMT voltage."""
    # the following numbers (voltage and gain) were taking from the spec sheet
    # for the Hamamatsu PMT array (7260-03).  We will need to scale them
    # however for our PMT's measured gain at 800 V.
    v = array([ 500.8208313, 514.43792725, 530.95874023, 548.79815674,
        566.42248535, 584.61279297, 603.38726807, 622.76470947, 642.7644043,
        664.36029053, 685.69580078, 707.71655273, 730.44445801, 753.90222168,
        778.11334229, 810.765625, 830.08496094, 856.74261475, 884.25640869,
        900.5435791])
    g = array([29282.6328125, 37161.0, 49402.69140625, 66894.7578125,
        88931.4453125, 118227.482, 157174.361, 208951.235, 280347.943,
        376140.261, 504663.372, 664776.993, 891924.858, 1196686.27, 1576358.91,
        2294951.92, 2837655.54, 3807257.41, 5061456.68, 5946480.27])
    p0 = sp.interpolate.splrep(v,g)
    norm = fullgain / sp.interpolate.splev(800, p0)
    gain = sp.interpolate.splev(voltage, p0) * norm
    return gain

def convertPNGs(dir):
    """Convert PNGs to JPGs in directory."""
    os.chdir(dir)
    cmd = "for f in *png ; do convert -quality 100 $f `basename $f png`jpg; done"
    os.popen(cmd)

# def removeImages(dir):
#     """Removes files in a directory"""
#     basedir, trash = os.path.split(dir)
#     os.chdir(dir)
#     for file in os.listdir(dir):
#         os.remove(file)
#     # 	for file in glob.glob(dir + '/*.png'):
#     # 		os.remove(file)
#     # 	for file in glob.glob(dir + '/*.jpg'):
#     # 		os.remove(file)
#     os.chdir(basedir)
#     os.rmdir(dir)

# def gzipPDFs(dir, run):
#     basedir, trash = os.path.split(dir)
#     os.chdir(basedir)
#     cmd = "tar czf %s-pdf.tgz pdf/*" % (run)
#     s = os.popen(cmd)
#     s.close()

def makeMovie(dir, run, fps = 3, ftype = 'jpg'):
    fname = '../'+ run + '.avi'
    os.chdir(dir)
    command = ('mencoder', 'mf://*.'+ftype, '-mf',
        'type='+ftype+':w=800:h=600:fps='+str(fps), '-ovc', 'lavc',
        '-lavcopts', 'vcodec=mjpeg', '-oac', 'copy', '-o', fname)
    s = os.spawnvp(os.P_WAIT, 'mencoder', command)

###############
### For CLI ###
###############

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def main(argv=None):
    """./ids.py [run]		

    Analyzes ids data and saves plots.  [run] should be in the format of ddmmyyr#"""

    # parse command line options

    if argv is None:
        argv = sys.argv
    programName = os.path.basename(argv[0])
    version = "%prog " +  "%s" % (__version__)
    usage = "usage: %prog run-number"
    parser = optparse.OptionParser(version = version, usage = usage)

    parser.add_option('-4', "--he", action="store_true", dest="He",
        help="Process the data for the He line at 468.57.", default=False)

    try:
        opts, args = parser.parse_args()
    except Usage, err:
        print >>sys.stderr, err.msg
        sys.exit(2)

    if len(args) > 0:
        for run in args:
            print "Processing %s" % run
            print "Reading data and generating plots."
            if opts.He:
                out = IDSdata(run, writeFiles = 1, wavelength = 468.57, mi = 4,
                    order = 12, writeData = True)
            else:
                out = IDSdata(run, writeFiles = 1, writeData = True)			
        sys.exit()
    else:
        x = parser.get_usage()
        print x
        sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())
