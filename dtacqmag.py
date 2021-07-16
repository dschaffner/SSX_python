#!/usr/bin/env python
"""Routines for the hi res magnetic probe."""

__author__ = "Tim Gray"
__version__ = "1.7.1"

import os
import sys

#import matplotlib
#matplotlib.use('Agg')
#matplotlib.rc('axes', grid=False)

from pylab import *
import matplotlib.pyplot as plt
from numpy import *
from ssxanalysis import *
import re
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker
import time
#ioff()

class hiresmag_data(sdr.dtac_diag):
    """High resolution magnetic probe data class.

    This probe was built and installed in the first half of 2011.  It uses the
    new ACQ132 D-Tac digitizers for data acquisition.

    Labelling for the probe is as follows: Each channel of data starts with
    'm', followed by the probe number (1), followed by axis (r,t,z), and ending
    in the channel number (1-16).  The channel names should be coded into the
    tgz files - be sure to use the right setup files when taking data with this
    probe."""
    def __str__(self):
        return "SSX hires mag: %s" % (self.shotname)

    def __repr__(self):
        if not self.filestrings:
            fs = None
        else:
            fs = self.filestrings
        return ("hiresmag_data('%s', probe = '%s', filestrings = %s)" %
            (self.shotname, self.probe, fs))

    def _setSpacing(self):
        """Sets the coil winding spacing.

        Called automatically from _processData()."""

        # spacing is in units of cm...
        self.spacing = 0.18*2.54
        # This is the amount of length you want to have between 0 and the first
        # coil
        self.spacing_0 = 0.5
        self.axes_r = ['r', 't', 'z']

        self.x_label = r'radius (cm)'

    def _processData(self):
        if 'simplecalib' in self.settings:
            self.simplecalib = self.settings['simplecalib']
        # This is the amount of length you want to have between 0 and the first
        # coil
        self.spacing_0 = 0
        self._setSpacing()
        self.axes = dict(zip(self.axes_r, arange(3)))
        # radius of probe locs in cm
        self.x = arange(self.numChans/3) * self.spacing + self.spacing_0
        if self.settings.has_key('calibFiles') and self.settings['calibFiles']:
            self.applyCalib()
            self.findClipping()
            self.integrateSignals()
            self.fft()

    def findClipping(self, clipParam = 2.498):
        tmp = np.ma.masked_inside(self.unCalibData, -clipParam, clipParam)
        self.clippedData = tmp.mask


    def integrateSignals(self, pt = 60):
        """Integrate our signals.
        
        Also calculates |B|."""
        self.iUnCalibData = np.ma.zeros(self.unCalibData.shape)
        self.iUnCalibData[:,:,1:] = sp.integrate.cumtrapz(self.unCalibData, dx =
            self.deltat * 1e6, axis = 2)
        
        self.Bdot = self.fullData.copy()
        
        # TODO should make the offset range a variable and write a new offset
        # removal method that takes advantage of it
        # we most definitely need this offset removal here.
        tmp = ma.masked_outside(self.Bdot[:,:,10:pt], -70, 70).mean(2)
             #ma = masked arrays module (numpy.ma)
             #masked_outside masks any element of the array outside of abs(70)
        #rint 'tmp shape = ',tmp.shape
        #print 'self.c_offsets before',self.c_offsets.shape
        self.c_offsets = np.expand_dims(tmp, 2)
           #expand_dims adds a dimension to tmp so that it can be subtracted from Bdot
        #print 'self.c_offsets after= ',self.c_offsets.shape
        self.Bdot_no = self.Bdot
        self.Bdot = self.Bdot - self.c_offsets
        
        self.B = sp.integrate.cumtrapz(self.Bdot, dx = self.deltat * 1e6, axis
            = 2)
        self.B = np.ma.masked_array(self.B)
        self.B_no = sp.integrate.cumtrapz(self.Bdot_no, dx = self.deltat * 1e6, axis
            = 2)
        self.B_no = np.ma.masked_array(self.B_no)
        self.B_simp = sp.integrate.simps(self.Bdot, dx = self.deltat * 1e6, axis
            = 2)
        self.B_simp = np.ma.masked_array(self.B_simp) 
        self.Bmod = sqrt(self.B[0]**2 + self.B[1]**2 + self.B[2]**2)
        self.Bmodpln = sqrt(self.B[0]**2+self.B[1]**2)
        self.BW = self.Bmod**2
        self.BW = self.BW.mean(0)
    

    def applyCalib(self, pth = 'magnetic_calibration_files'):
        """This applies the calibration file.

        This is trickier than other mag data.  We are going to apply the
        calibration we calculated to the non-integrated data, even though the
        calibrations were calculated based on the integrated data.  Then we
        will integrate it up and it should do the right thing (it does - I
        checked)."""

        #print 'Gain is ',self.gain #gain due to use of attenuator
        # find the calibration files
        calibFiles = self.settings['calibFiles']
        if pth == 'magnetic_calibration_files':
            pth = ssxutil.ssxPath('magnetic_calibration_files', 'ssx')

        if type(calibFiles) == type(''):
                calibFiles = [calibFiles,]
        self.calibFiles = calibFiles

        # nominally 16
        numChan = self.numChans / 3
        
        # save the uncalibrated data.  If we have the simplecalib variable,
        # apply it to the calibration data (just corrects for polarity).
        self.unCalibData = self.gain*(self.fullData.copy().reshape((3,numChan,-1)))
        self.unCalibData = np.ma.masked_array(self.unCalibData)
        # self.fullData = np.ma.masked_outside(self.unCalibData, -2.5, 2.5)
        if hasattr(self, 'simplecalib'):
            #print self.unCalibData
            self.unCalibData = (self.unCalibData *
                np.expand_dims(self.simplecalib, 2))
            #print self.unCalibData


        for p, calibFile in enumerate(calibFiles):
            # the p here (and calibfiles) is for the number of probes.  One
            # probe, one calib file.  There is some index math below to stick
            # multiple probes into one array (voltm) for efficiency.
            calibFile = os.path.join(pth, calibFile)
            calibData = loadtxt(calibFile)

            # reshape the array so we have dimensions of (axes, probes, time)
            # instead of (axes * probes, time)
            vdat = self.gain*self.fullData.reshape((3,numChan,-1))
            #print 'Full Data shape: ',self.fullData.copy().shape
            #figure(1)
            #plot(self.fullData[0,0:])
            cdat = calibData.reshape((3,numChan,4))
            # clear out data and get it ready for the calibrated data
            data = ma.zeros(vdat.shape)
            for j in xrange(numChan):
                # read in the calibration data.  Stick the 4th column (the
                # magnitudes) in 3x3 array (cfm) with the magnitudes on the
                # diagonal.  Stick the first three columns in a 3x3 array
                # (mvecm).
                cfm = eye(3) * cdat[:,j,3]
                mvecm = cdat[:,j,:3]
                # next apply it to our data.  Note we have to do this all in a
                # for loop because inv() only works on 2-D arrays.  The
                # dot(cfm, mvecm) essentially undoes the normalization we did
                # in mag_calib (not sure why this is a good idea, but keeping
                # it for historical reasons).  Then we invert the 3x3 array
                # from calibration, and dot that into the data from our probe
                # trio.  Then just stick it back into our data array.
                #print 'cfm shape:',cfm.shape
                #print 'mvecm shape:',mvecm.shape
                #print 'vdat shape',vdat[:,j].shape
                bbm = dot(inv(dot(cfm, mvecm)), vdat[:,j])
                data[:,j] = bbm
        self.channelNames = self.channelNames.reshape((3, numChan))
        self.fullData = data.copy()
        #figure(2)
        #plot(self.fullData[0,0,:])
    
    def rampOffset(self):
        """This function takes the last value of the B-field for as an offset value,
        then divides through by the number of timesteps to find an offset per timestep 
        value. This value is then multiplied by an array which increments by 1 for each
        timestep, starting at a particular time (in this case, 25ms) and continues to 
        remove the increasing offset until the end of the array. Thus, the very last 
        B-field measurement should be what the final B-field value is.
        """
    
    def fft(self, bdot = True, time = None):
        if time:#convert time values into index values
            t0 = time[0]
            t1 = time[1]
            t0 = (t0 - self.delay[1] * 1e6) / (self.deltat * 1e6)
            t0 = int(t0)
            t1 = (t1 - self.delay[1] * 1e6) / (self.deltat * 1e6)
            t1 = int(t1)
            self.fft_t = time
        else:
            t0 = 0
            t1 = self.Bdot.shape[2] 
            t1B = self.B.shape[2]
            self.fft_t = (self.time[0], self.time[-1])
        # save the fft window times - these are indices - to get the actual
        # times back, do self.time[self.fft_t[0]].
        if bdot:
            self.bdotstr = '-dot'
        else:
            self.bdotstr = ''
        Nk = self.numChans/3
        Nw = self.Bdot[:,:,t0:t1].shape[2]
        f = self.Bdot[:,:,t0:t1]

        if time:
            NwB = self.B[:,:,t0:t1].shape[2]
            fB = self.B[:,:,t0:t1]
            fBmod = sqrt((fB[0,:,:]**2)+(fB[1,:,:]**2)*(fB[2,:,:])**2)
        else:
            NwB = self.B[:,:,t0:t1B].shape[2]
            fB = self.B[:,:,t0:t1B]
            fBmod = sqrt((fB[0,:,:]**2)+(fB[1,:,:]**2)*(fB[2,:,:])**2)
        
        k = fft.fftfreq(Nk, (self.spacing))
        ak = fft.fft(f,axis=1)
        k0 = fft.fftshift(k)
        ak = fft.fftshift(ak, axes=(1,))
        if not mod(Nk, 2):
            k0 = np.append(k0, -k0[0])
            ak = np.append(ak, np.expand_dims(-ak[:,0,:], 1), axis = 1)
        Nki = Nk/2
        k2 = k0[Nki:]
        pk = abs(ak[:,Nki:])**2
        self.k = k2
        self.fftk = pk
        
        akB = fft.fft(fB,axis=1)
        k0B = fft.fftshift(k)
        akB = fft.fftshift(akB,axes=(1,))
        if not mod(Nk, 2):
            akB = np.append(akB, np.expand_dims(-akB[:,0,:], 1), axis = 1)
        pkB = abs(akB[:,Nki:])**2
        self.fftkB = pkB

        w = fft.fftfreq(Nw, (self.deltat))
        win = sp.signal.hann(f.shape[2])
        #f=win*f
        aw = fft.fft(f, axis=2)
        w0 = fft.fftshift(w)
        self.w0save = w0
        aw = fft.fftshift(aw, axes=(2,))
        self.awsave = aw
        # remove the extra w since this is B-dot
        tmpw = expand_dims(expand_dims(w0,0),0)
        if not bdot:
            aw = aw/tmpw
        if not mod(Nw, 2):
            w0 = np.append(w0, -w0[0])
            aw = np.append(aw, np.expand_dims(-aw[:,:,0],2), axis=2)
        
        wB = fft.fftfreq(NwB,(self.deltat))
        awB = fft.fft(fB,axis=2)
        w0B = fft.fftshift(wB)
        awB = fft.fftshift(awB,axes=(2,))
        if not mod(NwB,2):
            w0B = np.append(w0B,-w0B[0])
            awB = np.append(awB,np.expand_dims(-awB[:,:,0],2),axis=2)
            
        awBmod = fft.fft(fBmod,axis=1)
        awBmod = fft.fftshift(awBmod,axes=(1,))
        #print 'shape of awBmod', awBmod.shape
        if not mod(NwB,2):
            awBmod = np.append(awBmod,np.expand_dims(-awBmod[:,0],2),axis=1)
        #print 'shape of awBmod', awBmod.shape
        Nwi = Nw/2
        w2 = w0[Nwi:]
        pw = abs(aw[:,:,Nwi:])**2
        # pw = aw
        self.w = w2
        self.fftw = pw

        NwiB = NwB/2
        w2B = w0B[NwiB:]
        pwB = abs(awB[:,:,NwiB:])**2
        self.wB = w2B
        self.fftwB = pwB
        
        pwBmod = abs(awBmod[:,NwiB:])**2
        self.wBmod = w2B
        self.fftwBmod = pwBmod
       
        
        awk = fft.fft2(f, axes=(1, 2))
        if not bdot:
            awk = awk * tmpw
        awk = fftshift(awk, axes=(1,2,))
        if not mod(Nk, 2):
            awk = np.append(awk, np.expand_dims(-awk[:,0,:], 1), axis = 1)
            awk = np.append(awk, np.expand_dims(-awk[:,:,0], 2), axis = 2)
        pwk = abs(awk)**2
        # pwk = real(awk * conj(awk))
        pwk = ma.masked_where(pwk == 0, pwk)
        pwk[:,:,0] += pwk[:,:,0] + 0.01

        self.fftwk = pwk
        self.wk = [w0, k0]

    def background(self, background):
        self.fftwo = self.fftw.copy()
        self.fftko = self.fftk.copy()
        self.fftwko = self.fftwk.copy()
        self.fftw = self.fftwo - background.fftw
        self.fftk = self.fftko - background.fftk
        self.fftwk = self.fftwko - background.fftwk

    def spatialspec(self, axis = 0, plotLog = True, saveFig = False, cont =
        False, interp='nearest'):
        
        if type(axis) == type('s'):
            axis = self.axes[axis]

        fftdat = self.fftk.copy()
        t = self.time
        k = self.k
        ext = (self.fft_t[0], self.fft_t[-1], k[0], k[-1])
        
        if plotLog:
            fftdat = log10(fftdat)
        p = fftdat[axis, :, :]

        fig1 = figure(31)
        fig1.clear()
        
        imshow(p, aspect='auto', origin='lower', interpolation=interp,
            extent=ext)

        ylabel(r'k (m$^{-1}$)')
        xlabel(r'time ($\mu$s)')
        title(r'B%s_%s - %s' % (self.bdotstr, self.axes_r[axis],
            self.shotname))
        # figlabel()

        if saveFig:
            if plotLog:
                ll = '-log'
            else:
                ll = ''
            fig1.savefig(self.shotname + '-b' + self.axes_r[axis] + '-kspec' +
                ll + '.pdf')

    def temporalspec(self, axis = 0, plotLog = True, numconts = 15, saveFig =
        False, interp='nearest', fig = 32):
        if type(axis) == type('s'):
            axis = self.axes[axis]
        fftdat = self.fftw.copy()
        t = self.time
        w = self.w /1e6
        
        ext = (w[0], w[-1], 1, 17)

        if plotLog:
            fftdat = log10(fftdat)
        p = fftdat[axis, :, :]
        
        fig1 = figure(fig)
        fig1.clear()
        imshow(p, aspect='auto', origin='lower', interpolation=interp,
            extent=ext)
        
        ylabel(r'probe channel')
        xlabel(r'f (mhz)')
        title(r'B%s_%s - %s' % (self.bdotstr, self.axes_r[axis], self.shotname))
        # figlabel()
        
        if saveFig:
            if plotLog:
                ll = '-log'
            else:
                ll = ''
            fig1.savefig(self.shotname + '-b' + self.axes_r[axis] + '-wspec' +
                ll + '.pdf')
        
    def omegakplot(self, axis = 0, fig = 34, plotLog = True, saveFig = False,
        interp='nearest'):
        if type(axis) == type('s'):
            axis = self.axes[axis]
       
        fftdat = self.fftwk.copy()
        w, k = self.wk
        w = w/1e6

        ext = (w[0], w[-1], k[0], k[-1])
        
        if plotLog:
            fftdat = log10(fftdat)
        p = fftdat[axis, :, :]

        fig1 = figure(fig)
        fig1.clear()
        imshow(p, aspect='auto', origin='lower', interpolation=interp,
            extent=ext)
        title(r'B%s_%s - %s' % (self.bdotstr, self.axes_r[axis], self.shotname))
        xlabel(r'f (MHz)')
        ylabel(r'k (m$^{-1}$)')
        # figlabel()
        
        if saveFig:
            if plotLog:
                ll = '-log'
            else:
                ll = ''
            fig1.savefig(self.shotname + '-b' + self.axes_r[axis] + '-wkspec' +
                ll + '.pdf')
        
    def eulersum(self, axis = 0, saveFig = False, fig = 12):
        if type(axis) == type('s'):
            axis = self.axes[axis]

        fftdat = self.fftwk.copy()
        w, k = self.wk

        a = fftdat[axis, :, :]
        
        ai = trapz(a, k, axis = 0)
        
        fig = figure(fig)
        fig.clear()

        loglog(w/1e6,ai,'k-')
        
        title(r'B%s_%s - %s' % (self.bdotstr, self.axes_r[axis],
            self.shotname))
        xlabel('f (MHz)')
        ylabel('power (arb)')

        if saveFig:
            fig.savefig(self.shotname + '-b' + self.axes_r[axis] + '-euler' +
                '.pdf')

    def specgram(self, axis = 0, fig = 15):
        pass

    def plotWaves(self, chan = 1, lims = (44, 64), fig = 15, saveFig = False,
        lowh = False, *args, **kw):

        d = self.Bmod[chan - 1, :]

        # make our plots and set the spacing
        if lowh:
            fs = (5.33, 6)
            fig = plt.figure(fig, figsize = fs)
            fig.clear()
            fig, axs = plt.subplots(3, 1, sharex = True, sharey = False, num =
                fig.number, figsize = fs)
        else:
            fs = (5.33, 4)
            fig = plt.figure(fig, figsize = fs)
            fig.clear()
            fig, axs = plt.subplots(2, 1, sharex = True, sharey = False, num =
                fig.number, figsize = fs)


        fig.subplots_adjust(left = .15, right = .9, bottom = .1, hspace = .15)
        t = self.time[1:]

        # estimated density...
        ne = 1e15

        fci = d * 1.52e3
        fce = d * 2.8e6
        fpe = 8.98e3 * sqrt(ne)
        fpi = 2.1e2 * sqrt(ne)

        # from stix
        flh = ( 1 / (fci**2 + fpi**2) + 1 / (fci * fce) )**(-.5)
        # from bellan
        flh2 = sqrt( fci**2 + ( fpi**2 / (1 + fpe**2 / fce**2 )))
        # from stix - high density limit
        flh3 = sqrt( fci * fce )

        axs[0].plot(t, d, *args, **kw)
        axs[1].plot(t, fci / 1e6, *args, **kw)
        
        # puts the ylabels at the same spot
        # box = dict(pad = 5, alpha = 0)
        box = dict()

        if lowh:
            axs[2].plot(t, flh / 1e6, *args, **kw)
            axs[2].set_xlabel('time (us)')
            axs[2].set_ylabel('f$_{LH}$ (MHz)', bbox = box)
            # axs[2].plot(t, flh3 / 1e6, *args, **kw)
        else:
            axs[1].set_xlabel('time (us)')


        # set the xlims
        l1, l2 = lims
        axs[0].set_xlim(l1, l2)

        # set the ylims
        # we need to find the min and max of the three plots only in the time
        # plotted.
        t0 = (l1 - self.delay[1] * 1e6) / (self.deltat * 1e6)
        t0 = int(t0)
        t1 = (l2 - self.delay[1] * 1e6) / (self.deltat * 1e6)
        t1 = int(t1)
        yl = []
        yl.append(d[t0:t1].max())
        yl.append(d[t0:t1].min())
        yl1 = floor(min(yl) / 500) * 500
        yl2 = ceil(max(yl) / 500) * 500
        axs[0].set_ylim(yl1, yl2)

        axs[0].set_title('%s - channel %s' % (self.shotname, chan))
        axs[0].set_ylabel('|B| (G)', bbox = box)
        axs[1].set_ylabel('f$_{ci}$ (MHz)', bbox = box)
        # axs[1].set_ylabel(r'$%s_{\theta}$ (G)' % (dat), bbox = box)
        # axs[2].set_ylabel('$%s_z$ (G)' % (dat), bbox = box)

        fig.show()

    def plotLines(self, chan = 1, dat = 'B', lims = (44, 64), fig = 16, saveFig
        = False, showClip = False, *args, **kw):

        d = getattr(self, dat)
        d = d[:, chan-1, :]

        fs = (5.33, 6)
        fig = plt.figure(fig, figsize = fs)
        fig.clear()
        # make our plots and set the spacing
        fig, axs = plt.subplots(3, 1, sharex = True, sharey = True, num =
            fig.number, figsize = fs)

        fig.subplots_adjust(left = .15, right = .9, bottom = .1, hspace = .15)

        # puts the ylabels at the same spot
        # box = dict(pad = 5, alpha = 0)
        box = dict()

        #clipped data
        cd = d.copy()
        cd.mask = self.clippedData[:, chan-1]

        # fixes the time for integrated quantities
        if dat in ['B', 'Bmod']:
            t = self.time[1:]
        else:
            t = self.time
        
        # plot data
        if showClip:
            axs[0].plot(t, cd[0], *args, **kw)
            axs[1].plot(t, cd[1], *args, **kw)
            axs[2].plot(t, cd[2], *args, **kw)
        else:
            axs[0].plot(t, d[0], *args, **kw)
            axs[1].plot(t, d[1], *args, **kw)
            axs[2].plot(t, d[2], *args, **kw)

        # set the xlims
        l1, l2 = lims
        if l2 > self.time[-1]:
            l2 = self.time[-1]
        axs[0].set_xlim(l1, l2)

        # set the ylims
        # we need to find the min and max of the three plots only in the time
        # plotted.
        t0 = (l1 - self.delay[1] * 1e6) / (self.deltat * 1e6)
        t0 = int(t0)
        t1 = (l2 - self.delay[1] * 1e6) / (self.deltat * 1e6)
        t1 = int(t1)
        yl = []
        for i in xrange(3):
            yl.append(d[i, t0:t1].max())
            yl.append(d[i, t0:t1].min())
        yl1 = floor(min(yl) / 500) * 500
        yl2 = ceil(max(yl) / 500) * 500
        axs[0].set_ylim(yl1, yl2)

        axs[0].set_title('%s - channel %s' % (self.shotname, chan))
        axs[0].set_ylabel('$%s_%s$ (G)' % (dat, self.axes_r[0]), bbox = box)
        axs[1].set_ylabel('$%s_%s$ (G)' % (dat, self.axes_r[1]), bbox = box)
        axs[2].set_ylabel('$%s_%s$ (G)' % (dat, self.axes_r[2]), bbox = box)
        axs[2].set_xlabel('time (us)')
        fig.show()

    def plotChan(self, axis = 0, chan = 1, dat = 'B', fig = 15, showClip =
        False, *args, **kw):
        # if we feed this command an axis like 'r3', we should recognize that
        # it's a string and is both the axes and channel.  If axis = 'r', look
        # for the chan argument for the channel, and lastly, if axis = a
        # number, then just use that axis
        if type(axis) == type('s'):
            if len(axis) > 1:
                chan = int(axis[1:])
                axis = self.axes[axis[0]]
            elif len(axis) == 1:
                axis = self.axes[axis]

        ioff()
        fig = figure(fig)
        fig.clear()
        d = getattr(self, dat)
        if dat == "Bmod":
            d = d[chan-1, :]
        else:
            d = d[axis, chan-1, :]
        a = axes()
        #clipped data
        cd = d.copy()
        cd.mask = self.clippedData[axis, chan-1]
        if dat in ['B', 'Bmod']:
            t = self.time[1:]
        else:
            t = self.time
        a.plot(t, d, *args, **kw)
        if showClip:
            a.plot(t, cd, 'r.', *args, **kw)
        title('channel %s' % chan)
        ylabel('%s_%s' % (dat, self.axes_r[axis]))
        xlabel('time (us)')
        fig.show()
        ion()

    def plotRaw(self, axis = 0, chan = 1, *args, **kw):
        # if we feed this command an axis like 'r3', we should recognize that
        # it's a string and is both the axes and channel.  If axis = 'r', look
        # for the chan argument for the channel, and lastly, if axis = a
        # number, then just use that axis
        if type(axis) == type('s'):
            if len(axis) > 1:
                chan = int(axis[1:])
                axis = self.axes[axis[0]]
            elif len(axis) == 1:
                axis = self.axes[axis]
        ioff()
        fig = figure(16)
        fig.clear()
        d = self.unCalibData
        d = d[axis, chan-1, :]
        #clipped data
        cd = d.copy()
        cd.mask = self.clippedData[axis, chan-1]
        
        a = axes()
        a.plot(self.time, d, *args, **kw)
        a.plot(self.time, cd, 'r-', *args, **kw)
        title('UNCALIBRATED - channel %s' % chan)
        ylabel('%s_%s' % ('uncalib', self.axes_r[axis]))
        xlabel('time (us)')
        fig.show()
        ion()
    
    def plotB(self, t = None, timestep = 7, timerange = (28,60), writeFiles
        = False, pdfOut = False, scale = 2000): 

        if pdfOut:
            writeFiles = True
        if writeFiles:
            # make output directory
            if pdfOut:
                types = ['png', 'pdf']
            else:
                types = ['png']
            subDirs = [''] + types
            for dir in subDirs:
                fName = ssxutil.ssxPath('', 'output', self.runYear + '/' +
                    self.runDate + '/' + self.shotname + '/mag/' +dir)
                os.spawnlp(os.P_WAIT, 'mkdir', 'mkdir', '-p', fName)

        r = self.B[0,:,:]
        th = self.B[1,:,:]
        z = self.B[2,:,:]
        lr = arange(self.numChans/3) * self.spacing + .5
        ly = zeros(lr.shape)

        fig1 = figure(5, figsize=(5.33, 9))
        fig1.clear()
        # time label
        subplot(311)
        kw = {'units': 'inches',
            'scale_units': 'inches',
            'width': .03,
            'headlength': 5,
            'headwidth': 3,
            'zorder': 10,
            'scale': scale}
        p1 = quiver(lr, ly, ly, r[:,0], **kw)
        ax = gca()
        ax.yaxis.set_major_formatter(mpl.ticker.NullFormatter())
        tx = text(.10,.8, r'%.1f $\mu$s' % (0), horizontalalignment='center',
            color = 'gray', transform = ax.transAxes)
        quiverkey(p1, 0.9, 0.8, 1000, '1 kG', coordinates = 'axes', color =
            'gray', labelcolor = 'gray')
        # ylabel(r'B$_r$')
        ylabel(r'B_%s'% self.axes_r[0])
        title(self.shotname)

        subplot(312)
        p2 = quiver(lr, ly, ly, th[:,0], **kw)
        ax = gca()
        ax.yaxis.set_major_formatter(mpl.ticker.NullFormatter())
        # ylabel(r'B$_\theta$')
        ylabel(r'B_%s'% self.axes_r[1])
        
        subplot(313)
        p3 = quiver(lr, ly, ly, z[:,0], **kw)
        ax = gca()
        ax.yaxis.set_major_formatter(mpl.ticker.NullFormatter())
        # ylabel(r'B$_z$')
        ylabel(r'B_%s' % self.axes_r[2])
        xlabel(self.x_label)

        if not timerange:
            t0 = 0
            t1 = self.time.shape[0]
        else:
            t0 = timerange[0]
            t1 = timerange[1]
            t0 = (t0 - self.delay[1] * 1e6) / (self.deltat * 1e6)
            t0 = int(t0)
            t1 = (t1 - self.delay[1] * 1e6) / (self.deltat * 1e6)
            t1 = int(t1)
        if t:
            t0 = (t - self.delay[1] * 1e6) / (self.deltat * 1e6)
            t0 = int(t0)
            t1 = t0 + 1
        for t in xrange(t0, t1, timestep):
        # for t in xrange(self.time[0],self.time[-1],timestep):
            # rt = (t/10. - 20)
            rt = self.time[t]
            V1 = r[:,t]
            V2 = th[:,t]
            V3 = z[:,t]
            p1.set_UVC(lr, V1)
            p2.set_UVC(lr, V2)
            p3.set_UVC(lr, V3)
            tx.set_text( r'%.1f $\mu$s' % (rt))
            draw()


            if writeFiles:
                for t in types:
                    fName = "f%04i.%s" % ((rt*10), t)
                    fName = ssxutil.ssxPath(fName, 'output', self.runYear +
                        '/' + self.runDate + '/' + self.shotname +
                        "/mag/%s" % t)
                    savefig(fName)

    def plotBavg(self, saveFig = False, fig = 18, plotLog = False):
        """Plots mean magnetic field."""
        t = self.time[1:]
        fig1 = plt.figure(fig)
        fig1.clear()
        ax = fig1.add_subplot(111)
        if plotLog:
            semilogy(t[780:], self.Bmod.mean(0)[780:], 'b-')
        else:
            plot(t, self.Bmod.mean(0), 'b-')
        ylabel(r'$|\bar{B}|$')
        xlim(20,100)
        xlabel('time (us)')

        if saveFig:
            if plotLog:
                ll = '-log'
            else:
                ll = ''
            fig1.savefig(self.shotname + '-b-avg' + ll + '.pdf')

    
    def plotBW(self, saveFig = False, fig = 18, plotLog = False):
        """Plots magnetic energy."""
        t = self.time[1:]
        fig1 = plt.figure(fig)
        fig1.clear()
        ax = fig1.add_subplot(111)
        if plotLog:
            semilogy(t[780:], self.BW[780:], 'b-')
            ylabel(r'$\frac{1}{n} \sum_n B^{2}$')
        else:
            plot(t, self.BW/1.e6, 'b-')
            ylabel(r'$\frac{1}{n} \sum_n B^{2} \times 10^{6}$')
        xlim(20,100)
        xlabel('time (us)')

        if saveFig:
            if plotLog:
                ll = '-log'
            else:
                ll = ''
            fig1.savefig(self.shotname + '-b-w' + ll + '.pdf')

    def imB(self, axis = 0, p = 'B', interp = 'bilinear', saveFig =
        False, fig = 22, **kw):
        if type(axis) == type('s'):
            axis = self.axes[axis]
        t = self.time
        if p == 'Bmod':
            p = self.Bmod
            tstr = '|B|'
        else:
            if p == 'B':
                p = self.B
                bdotstr = ''
            elif p == 'Bdot':
                p = self.Bdot
                bdotstr = self.bdotstr
            elif p == 'u':
                p = self.unCalibData
                bdotstr = 'uncalib'
            elif p == 'iu':
                p = self.iUnCalibData
                bdotstr = 'uncalib'
            tstr = r'B{}$_{}$ - {}'.format(bdotstr, self.axes_r[axis],
                self.shotname)
            p = p[axis, :, :]
        # ext = (t[0], t[-1], 1, self.numChans/3 + 1)
        ext = (t[0], t[-1], self.x[0], self.x[-1])

        fig1 = figure(fig)
        fig1.clear()
        imshow(p, aspect='auto', origin='lower', interpolation=interp,
            extent=ext, **kw)
        
        xlabel(r'time (us)')
        ylabel(self.x_label)
        title(tstr)
        colorbar()
        # figlabel()
        
        if saveFig:
            if plotLog:
                ll = '-log'
            else:
                ll = ''
            fig1.savefig(self.shotname + '-b' + axis + '-wspec' + ll + '.pdf')

    def im3(self, time = [20, 80], plotLog = False, saveFig = False, fig = 23,
        ext = 'png', **kw):
        t = self.time
        p = self.Bmod
        if plotLog:
            p = log10(p)
        t0 = time[0]
        t1 = time[1]
        t0 = (t0 - self.delay[1] * 1e6) / (self.deltat * 1e6)
        t0 = int(t0)
        t1 = (t1 - self.delay[1] * 1e6) / (self.deltat * 1e6)
        t1 = int(t1)
        if t1 > len(t):
            t1 = t.shape[0] - 1

        n = self.numChans/3

        X, Y = meshgrid(t, arange(n) + 1)

        fig1 = plt.figure(fig)
        fig1.clear()
        plt.subplots_adjust(0,0,1,1)
        ax = fig1.add_subplot(111, projection = '3d', frame_on=False)
        ax.plot_surface(X[:, t0:t1], Y[:, t0:t1], p[:, t0:t1], rstride = 1,
            cstride = 65, cmap=cm.Oranges, **kw)
        ax.view_init(45, 235)
        ax.set_xlabel(r'time (us)')
        ax.set_ylabel(r'channel')
        ax.set_zlabel('|B|')
        titlestr = "{}".format(self.shotname)
        ax.set_title(titlestr)

        show()
        if saveFig:
            if plotLog:
                ll = '-log'
            else:
                ll = ''
            fn = self.shotname + '-bmod-3d' + ll + '.' + ext
            fig1.savefig(fn)
        return ax

    def lamcalc(self, t = 50, avg = None, showplot = True):
        # put B in Tesla (everything needs to be in MKS)
        B = self.B / 1e4
        t0 = (t - self.delay[1] * 1e6) / (self.deltat * 1e6)
        if not avg:
            Br = B[0, :, t0]
            Bt = B[1, :, t0]
            Bz = B[2, :, t0]
        else:
            Br = B[0, :, t0:t0+avg].mean(1)
            Bt = B[1, :, t0:t0+avg].mean(1)
            Bz = B[2, :, t0:t0+avg].mean(1)
        bmod = sqrt(Br**2 + Bt**2 + Bz**2)
        bmax = bmod.max()
        noise = .0003
        rerr = .002


        r = arange(self.numChans/3) * self.spacing + .5
        r2 = (arange(self.numChans/3 - 1) + .5) * self.spacing + .5
        # put these in meters, not centimeters
        r = r / 100
        r2 = r2 / 100
        rp = arange(r[0], r[-1], .001)

        splr = sp.interpolate.splrep(r, Br)
        splt = sp.interpolate.splrep(r, Bt)
        splz = sp.interpolate.splrep(r, Bz)
        splrt = sp.interpolate.splrep(r, r * Bt)

        br = sp.interpolate.splev(rp, splr)
        bt = sp.interpolate.splev(rp, splt)
        bz = sp.interpolate.splev(rp, splz)
        rbt = sp.interpolate.splev(rp, splrt, 1)

        # we don't really use this calculation, but leave it around
        drrbt = mf.deriv(rp, rp * bt)
        lambp1 = (rbt) / (rp * bz)
        # other orientation
        lambp2 = (rbt - br) / (rp * bz)

        # need to put spacing in meters
        drrbt2 = diff(r * Bt) / (self.spacing / 100)
        lamb1 = (drrbt2) / (r2 * avgnminus1(Bz))
        lamb1 = abs(lamb1)
        lamberr1 = (sqrt((noise/avgnminus1(Bt))**2 + 2 * (rerr/r2)**2 +
            (noise/avgnminus1(Bz))**2 + (rerr/r2)**2 +
            (noise/avgnminus1(Br))**2) * lamb1)
        # other orientation
        lamb2 = (drrbt2 - avgnminus1(Br)) / (r2 * avgnminus1(Bz))
        lamb2 = abs(lamb2)
        lamberr2 = (sqrt((noise/avgnminus1(Bt))**2 + 2 * (rerr/r2)**2 +
            (noise/avgnminus1(Bz))**2 + (rerr/r2)**2 +
            (noise/avgnminus1(Br))**2) * lamb2)

        # fig = figure(16)
        # fig.clear()
        # plot(rp, lamb, 'bo')
        # ylim(0,100)

        if showplot:
            fig = figure(14)
            fig.clear()
            plot(r, Br, 'ro', ms = 3)
            plot(r, Bt, 'go', ms = 3)
            plot(r, Bz, 'bo', ms = 3)

            plot(rp, br, ':', color = 'gray') 
            plot(rp, bt, ':', color = 'gray') 
            plot(rp, bz, ':', color = 'gray') 
            xlabel('Radius (m)')
            ylabel('B (T)')	

            fig = figure(15)
            fig.clear()
            plot(rp, lambp1, 'r:')
            plot(r2, lamb1, 'ko', ms = 3)
            errorbar(r2, lamb1, lamberr1, fmt=None, ecolor = 'gray')
            ylim(0,100)
            title('Lambda - angle = 0')
            xlabel('Radius (m)')
            ylabel(r'$\lambda$ (m$^{-1}$)')

            fig = figure(16)
            fig.clear()
            plot(rp, lambp2, 'r:')
            plot(r2, lamb2, 'ko', ms = 3)
            errorbar(r2, lamb2, lamberr2, fmt=None, ecolor = 'gray')
            ylim(0,100)
            title('Lambda - angle = 90')
            xlabel('Radius (m)')
            ylabel(r'$\lambda$ (m$^{-1}$)')

        return lamb1, lamb2

    def lamt(self):
        t = arange(30, 70, .5)
        lamb1 = zeros(t.shape)
        lamb2 = zeros(t.shape)
        lamberr1 = zeros(t.shape)
        lamberr2 = zeros(t.shape)
        for i,k in enumerate(t):
            a, b = self.lamcalc(t = k, showplot = False)
            lamb1[i] = a.mean()
            lamb2[i] = b.mean()
            # lamberr1[i] = sqrt(a.std(0))/a.shape[0]
            # lamberr2[i] = sqrt(b.std(0))/b.shape[0]
            lamberr1[i] = sqrt(a.std(0))
            lamberr2[i] = sqrt(b.std(0))

        fig = figure(17)
        fig.clear()
        plot(t, lamb1)
        errorbar(t, lamb1, lamberr1, fmt = None, ecolor = 'gray')
        xlabel('time (us)')
        ylabel('lambda (m^-1)')
        ylim(0,100)

        fig = figure(18)
        fig.clear()
        plot(t, lamb2)
        errorbar(t, lamb2, lamberr2, fmt = None, ecolor = 'gray')
        xlabel('time (us)')
        ylabel('lambda (m^-1)')
        ylim(0,100)

    def profile(self, axis = 0, p = 'B', t = 20, saveFig = False, fig = 25):
        time = self.time
        if type(axis) == type('s'):
            axis = self.axes[axis]
        # need to find the right index for the time entered.  If there is no
        # pre delay on the shot, self.delay[1] = 0, so you just divide the time
        # given by deltat in us.
        t0 = (t - self.delay[1] * 1e6) / (self.deltat * 1e6)
        t0 = int(t0)
        dat = getattr(self, p)
        pdat = dat[axis, :, t0]
        x = arange(self.numChans/3) * self.spacing + .5

        fig = figure(fig)
        fig.clear()
        plot(x, pdat, 'b--')
        plot(x, pdat, 'ko')
        xlabel(self.x_label)
        ylabel(p + '_' + self.axes_r[axis])
        ax = gca()
        tx = text(.9,.9, r'%.1f $\mu$s' % self.time[t0],
            horizontalalignment='center', color = 'gray', transform =
            ax.transAxes) 
        if saveFig:
            if plotLog:
                ll = '-log'
            else:
                ll = ''
            fig1.savefig(self.shotname + '-b' + self.axes_r[axis] + '-wspec' +
                ll + '.pdf')

class flexmag_data(hiresmag_data):
    """Flex magnetic probe data class.

    This probe was built and installed in the summer of 2012.  It uses the new
    ACQ132 D-Tac digitizers for data acquisition.  This is for the flexible
    magnetic probes built by Alex Werth.

    Labelling for the probe is as follows: Each channel of data starts with
    'f', followed by the probe number (1), followed by axis (a,b,c), and ending
    in the channel number (1-8).  The channel names should be coded into the
    tgz files - be sure to use the right setup files when taking data with this
    probe.
    
    It should be noted that these probes do not follow the r,t,z nomenclature.
    The way a,b,c matches up is as follows:
        r -> c
        t -> b
        z -> a
    """

    def __str__(self):
        return "SSX flex mag: %s" % (self.shotname)

    def __repr__(self):
        if not self.filestrings:
            fs = None
        else:
            fs = self.filestrings
        return ("flexmag_data('%s', probe = '%s', filestrings = %s)" %
            (self.shotname, self.probe, fs))

    def _setSpacing(self):
        """Sets the coil winding spacing.

        Called automatically from _processData()."""

        self.spacing = 0.375*2.54
        self.axes_r = ['a', 'b', 'c']
        self.x_label = r'length (cm)'
        
def getMagData(shot, probe = 'm2',gain=1.0):
    """Gets mag data for the hires probe."""

    if probe == 'm1':
        filestrings = ['mag1', 'mag2']
        calibFiles = ['calib-061411-hires1.txt',]
        diag = 'hires1'
    elif probe == 'm2':
        shotRe = re.compile(r"""(\d\d)(\d\d)(\d\d)r?(\d*)(.*)""")
        m = shotRe.match(shot)
        if m:
            (runMonth, runDay, runYearShort, runNumber, runSuffix) = m.groups()
            if int(runMonth) < 11 and int(runYearShort) == 11:
                filestrings = ['mag2', 'mag3']
                calibFiles = ['calib-072511-hires2.txt',]
            else:
                filestrings = ['mag1', 'mag2', 'mag3']
                calibFiles = ['calib-121611-hires2.txt',]
                #calibFiles = ['calib-051511-hires2.txt',]
        diag = 'hires2'
        # calibFiles = None
    simplecalib = array([
        [1 ,  1, -1, -1, -1, -1,  1,  1,  1, -1, 1,  1, -1, -1,  1, -1],
        [-1, -1,  1,  1,  1,  1,  1,  1, -1, -1, 1, -1,  1, -1,  1, 1],
        [1 ,  1,  1, -1,  1, -1, -1, -1,  1,  1, 1,  1, -1, -1, -1, 1]])
    magdata = hiresmag_data(shot, probe, gain=gain,filestrings=filestrings,
        diagname = diag, calibFiles=calibFiles, simplecalib = simplecalib)

    return magdata
    
def getMagData2(shot, probe = 'm2',gain=1.0):
    """Gets mag data for the hires probe."""

    if probe == 'm1':
        filestrings = ['mag1', 'mag2']
        calibFiles = ['calib-061411-hires1.txt',]
        diag = 'hires1'
    elif probe == 'm2':
        shotRe = re.compile(r"""(\d\d)(\d\d)(\d\d)r?(\d*)(.*)""")
        m = shotRe.match(shot)
        if m:
            (runMonth, runDay, runYearShort, runNumber, runSuffix) = m.groups()
            if int(runMonth) < 11 and int(runYearShort) == 11:
                filestrings = ['mag2', 'mag3']
                calibFiles = ['calib-072511-hires2.txt',]
            else:
                filestrings = ['mag1', 'mag2', 'mag3']
                #calibFiles = ['calib-121611-hires2.txt',]
                calibFiles = ['calib-051514-hires2.txt',]
        diag = 'hires2'
        # calibFiles = None
    simplecalib = array([
        [1 ,  1, -1, -1, -1, -1,  1,  1,  1, -1, 1,  1, -1, -1,  1, -1],
        [-1, -1,  1,  1,  1,  1,  1,  1, -1, -1, 1, -1,  1, -1,  1, 1],
        [1 ,  1,  1, -1,  1, -1, -1, -1,  1,  1, 1,  1, -1, -1, -1, 1]])
    magdata = hiresmag_data(shot, probe, gain=gain,filestrings=filestrings,
        diagname = diag, calibFiles=calibFiles, simplecalib = simplecalib)

    return magdata
    
def getMJMagData(shot, probe = 'm2',gain=1.0):
    """Gets mag data for the hires probe."""

    if probe == 'm1':
        filestrings = ['mag1', 'mag2']
        calibFiles = ['calib-061411-hires1.txt',]
        diag = 'hires1'
    elif probe == 'm2':
        shotRe = re.compile(r"""(\d\d)(\d\d)(\d\d)r?(\d*)(.*)""")
        m = shotRe.match(shot)
        if m:
            (runMonth, runDay, runYearShort, runNumber, runSuffix) = m.groups()
            if int(runMonth) < 11 and int(runYearShort) == 11:
                filestrings = ['mag2', 'mag3']
                calibFiles = ['calib-072511-hires2.txt',]
            else:
                filestrings = ['mag1', 'mag2', 'mag3']
                #calibFiles = ['calib-121611-hires2.txt',]
                calibFiles = ['calib-123016-mjmag_1.txt',]
        diag = 'hires2'
        # calibFiles = None
    simplecalib = array([
        [1 , 1, 1, 1, 1, 1,  1,  1,  1, 1, 1,  1, 1, 1,  1, 1],
        [1 , 1, 1, 1, 1, 1,  1,  1,  1, 1, 1,  1, 1, 1,  1, 1],
        [1 , 1, 1, 1, 1, 1,  1,  1,  1, 1, 1,  1, 1, 1,  1, 1],])
    magdata = hiresmag_data(shot, probe, gain=gain,filestrings=filestrings,
        diagname = diag, calibFiles=calibFiles, simplecalib = simplecalib)

    return magdata

def getFlexMagData(shot, probe = 'f1', bu = False):
    """Gets mag data for the flex probes.
    
    As of 2012-07-19, flex1 is the backup for flex2.  Likewise, f4 is the
    backup for f3.  This means that 1 and 2 share cables and dtac
    channels and similar for 3 and 4."""

    if probe == 'f1':
        filestrings = ['mag1', 'mag2']
        if not bu:
            calibFiles = ['calib-072512-flex2.txt',]
        else:
            calibFiles = ['calib-072512-flex1.txt',]
        diag = 'flex1'
    elif probe == 'f2':
        filestrings = ['mag2', 'mag3']
        if not bu:
            calibFiles = ['calib-072512-flex3.txt',]
        else:
            calibFiles = ['calib-072612-flex4.txt',]
        diag = 'flex2'

    magdata = flexmag_data(shot, probe, filestrings=filestrings,
        diagname = diag, calibFiles=calibFiles)

    return magdata

def avgnminus1(x):
	y = zeros(len(x)-1)
	for i in xrange(len(x) - 1):
		y[i] = (x[i] + x[i+1] ) / 2
	return y

def processDay(runDate):
    runs = sdr.listRuns(runDate)
    for run in runs:
        try:
            print (run)
            m = getMagData(run)
            m.writeFullData()
            m = getMagData(run, 'm2')
            m.writeFullData()
        except:
            pass

def plotI(dat):
    fig = figure(1)
    fig.clear()
    for i in xrange(3):
        for j in xrange(8):
            plot(dat.time, dat.iUnCalibData[i,j])

    fig = figure(2)
    fig.clear()
    for i in xrange(3):
        for j in xrange(8):
            plot(dat.time, dat.unCalibData[i,j])
