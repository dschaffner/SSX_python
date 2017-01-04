#!/usr/bin/env python
"""SSX data read functions and data classes.

This module contains the read functions for generic data.  The data class
ssx_data is also defined.

Requires matplotlib for plotting."""

__author__ = "Tim Gray"
__version__ = "1.7.1"

# 1.0 - original code
# 1.1 - added in dtac code
# 1.2 - 9/11/08 - added code to use new data locations (on ion) and refactoring
# for better design
# 1.2.1 - 2/9/09 - added code so we can tell if we are on ion or elsewhere so
# we can read data properly
# 1.3 - 2/18/09 - added in rga and oo code.  changed code so it could find the
# rga and oo data on the data server, in gzipped format.  Also now uses the
# ssx_data class.
# 1.3.1 - 2/24/09 - added in some peak detection stuff to oo
# 1.4 - 4/22/09 - read in old format data.  get rid of ssx_data_read_old.
# Actually it was easier to keep sdr_old.  I've just incorporated that as an
# external module.  If you call an 'old' data file, it calls the appropriate
# classes from the old module, and copies the attributes over.  Much less work
# that way.  Retrieving old data that isn't stored locally does not work, since
# it's still all sitting on the G4 hard drive.  At some point, I need to pull
# that out and copy it all over to ion in its own directory, and update the ssh
# fetch routines, but only for old data.
# 1.5 - 2011-05-27 - corresponds to v1.5 of ssx.py.
# 1.7 - 2012-07-12 - started right after 1.6 finished
#       - moved many methods to private (start with _) to cleanup namespace in
#         interactive mode
#       - upgraded file retrieval methods to try curl/http first and then
#         toggle over to ssh if it fails.

import ssxdefaults as ssxdef
import ssxutilities as ssxutil
from string import *
import os, sys
path = os.path
import gzip, zipfile, tarfile
import re
import struct
import subprocess as sb
import shlex
import numpy as np
from numpy import array, transpose, zeros, arange, where, ma
import ssxmathfuncs as mf

#import ssx_data_read_old as sdr_old

from pylab import (plot, figure, close, axes, title, xlabel, ylabel, xlim,
    ylim, legend, show, ion, ioff, loglog)			# For matplotlib plotting
from matplotlib.ticker import MultipleLocator

import datetime as dt
# from matplotlib.dates import *

from ssxutilities import *

# location of data
dataServer = "ion.physics.swarthmore.edu"
httpDir = 'data'

# find our location and look for data in the right spot.  If we are on ion, we
# need to not copy data either.
#location = os.uname()[1]
#if location == 'ion.physics.swarthmore.edu':
#    basedir = '/ssx/data'
#elif location == 'd197.scdc1.swarthmore.edu':
#    basedir = os.path.expanduser('~/Documents/SSX/data')
#else:
#    basedir = '~/Documents/data'

basedir = 'C:\Users\dschaffner\Google Drive\Data Deposit'

# I like to store may data elsewhere.  Others can put it where they want.
#if os.getlogin() != 'tgray':
#    basedir = '~/Documents/data'
#else:
#    basedir = os.path.expanduser('~/Documents/SSX/data')

# set the default fetch method for data
fetchMethod = 'http'

###############
### Classes ###
###############

class ssx_date:
    """Parses shotname and adds a datetime.

    Doesn't work yet."""

    shottime = ''
    numtime = 0
    runDate = ''
    runNumber = ''

    def __init__(self, shotname):
        self.shottime = dt.datetime(int(shotname[4:6]) + 2000,
            int(shotname[0:2]), int(shotname[2:4]), int(shotname[6:8]),
            int(shotname[8:10]))
        self.numtime = date2num(self.shottime)

class ssx_base(object):
    """Base ssx class.

    Provides several methods that are used in different classes."""

    def __init__(self):
        pass

    def runSplit(self):
        """Parses the shotname.

        Parses the shotname into useful information.  Makes python timedate
        object as well as strings for the day, month, and year."""
        shotRe = re.compile(r"""(\d\d)(\d\d)(\d\d)r?(\d*)(.*)""")
        m = shotRe.match(self.shotname)
        if m:
            (self.runMonth, self.runDay, self.runYearShort, self.runNumber,
                self.runSuffix) = m.groups()
            self.runDate = self.runMonth + self.runDay + self.runYearShort
            self.runYear = '20'+self.runYearShort
            self.dateString = "%i/%i/%i" % (int(self.runMonth),
                int(self.runDay), int(self.runYear))
            self.oldFormat = False
            if (int(self.runYear) < 2009) and (int(self.runMonth) < 9):
                self.oldFormat = True
            if self.runNumber != '':		
                self.isShot = True
            else:
                self.isShot = False
            self.date = dt.datetime(int(self.runYear), int(self.runMonth),
                int(self.runDay))
        else:
            self.isShot = False

class ssx_data(ssx_base):
    """Generic SSX data class."""
    # data,time,signal = array([]), array([]), array([])	
    shotname = ''
    xlabel = 'Time (s)'
    ylabel = 'Signal (Arb. Units)'

    def __init__(self, shotname):
        """Sets up instance's title and shotname"""

        self.shotname = shotname

        self.runSplit()
        # set up the dates
# 		self.shottime = ssx_date(shotname)

    def __str__(self):
        return "SSX data %s" % self.shotname

    def __repr__(self):
        return "ssx_data('%s')" % (self.shotname)

    def _setDoc(self, func):
        self.__doc__ = func.__doc__

    def _getLines(self):
        """Reads fileName as file type.

        Automatically decodes gzip or anything else that you might set it up
        for."""
        if self.type == 'gzip':
            gzipper = gzip.GzipFile(self.filename)
            lines = gzipper.readlines()
            gzipper.close()
        elif self.type == 'ascii':
            lines = file(self.filename).readlines()
        return lines

    def _getFile(self):
        """Returns a file object that is decompressed, depending on type."""
        if self.type == 'gzip':
            f = gzip.GzipFile(self.filename)
        elif self.type == 'ascii':
            f = file(self.filename)
        return f

    def _fileType(self):
        """Checks to see if file exists locally, gzipped or not.

        This sets self.type to the file type.  If the file can't be located on
        the local filesystem, then self.type is set to False."""
        fn, ext = os.path.splitext(self.filename)
        #print 'filename =',self.filename
        if os.path.exists(self.filename):
            if ext == ".gz":
                self.type = "gzip"
            elif ext == ".tgz":
                self.type = "tar+gzip"
            elif ext in ['', '.txt']:
                self.type = 'ascii'
            elif ext == ".ProcSpec":
                self.type = 'zip'
        else:
            if os.path.exists(fn):
                self.type = 'ascii'
                self.filename = fn
            else:
                self.type = False

    def renewData(self):
        """Refetch and reread data from server."""
        self.fetchFile()
        self._getData()

    def _getData(self, readData = True):
        """High level get data function."""

        self.title = self.shotname

# 		self.runSplit()

        if self.isShot:
            if hasattr(self, 'fileString'):
                self._processFilename()
                self._processRemFilename()
        else:
            self._altNameProcess()

        self._fileType()
        if not self.type:
            x = self.fetchFile()
            self._fileType()

        # TODO put in exception handling here
        if not self.type:
            print "No file"
        else:
            if readData:
                self._get_data_low()

    def showHeader(self):
        """Prints header strings."""
        if 'header' in dir(self):
            print self.header
        else:
            print 'No header strings here.'

    def _processFilename(self):
        """Construct full filename on local computer."""
        #basename = "%s/%s/%sr%i-%s%s" % (self.runYear, self.runDate,
        #    self.runDate, int(self.runNumber), self.fileString, self.fileExt)
        basename = "%s\%s\%sr%i-%s%s" % (self.runYear, self.runDate,
            self.runDate, int(self.runNumber), self.fileString, self.fileExt)
        self.filename = ssxutil.ssxPath(basename,'data')

    def _processRemFilename(self):
        """Construct full filename of file on data server."""
        basename = "%s/%s/%sr%i-%s%s" % (self.runYear, self.runDate,
            self.runDate, int(self.runNumber), self.fileString, self.fileExt)
        self.remotefilename = os.path.join('/ssx/data/', basename)
        self.remoteserver = dataServer

    def _altNameProcess(self):
        """Construct names for non-shot files."""
        basename = self.shotname + self.fileString + self.fileExt
        self.remotefilename = os.path.join('/ssx/data/', basename)
        self.filename = ssxutil.ssxPath(basename,'data')

    def _processOldFilename(self):
        basename = "%s/%s/r%02i(%s)%s" % (self.runYear, self.runDate,
            int(self.runNumber), self.runDate, self.fileString)
        self.filename = ssxPath(basename,'data')

    def fetchFile(self):
        """Get data off of ion.
        
        Uses HTTP and falls back on an SSH connection.  If the file is a
        gzipped text file (.txt.gz) and the file doesn't appear to be on ion,
        this method will automatically look for the ungzipped version of the
        file.  This is because sometimes labview has problems gzipping the
        files."""
        localDir, fname = os.path.split(self.filename)

        #if not os.path.exists(localDir):
        #    os.spawnlp(os.P_WAIT, 'mkdir', 'mkdir', '-p', localDir)

        # We will try to use curl to fetch the data via http.  This will only
        # work if you are on the same network as ion or inside the swarthmore
        # firewall somehow.  If this fails, we will switch to SSH retrieval.
        # We will also fix the corner case of looking for a .txt.gz file on ion
        # that wasn't gzipped for some reason.
        print "Retrieving %s... " % (fname),
        global fetchMethod
        a = True
        b = True
        if fetchMethod == 'http':
            # print 'Trying http'
            a = self._fetchFileHTTP(1)
            if not a and self.fileExt == '.txt.gz':
                # print 'Trying http without gz'
                a = self._fetchFileHTTP(1, removeGZ = True)
            if not a:
                # print 'Trying ssh'
                b = self._fetchFileSSH()
            if not b and self.fileExt == '.txt.gz':
                # print 'Trying ssh without gz'
                b = self._fetchFileSSH(removeGZ = True)
            if not a and b:
                print "Setting fetch method to SSH."
                fetchMethod = "ssh"
        elif fetchMethod == 'ssh':
            b = self._fetchFileSSH()
        return a or b

    def _fetchFileHTTP(self, timeout = 1, removeGZ = False):
        """Fetch data over HTTP.

        Needs to be on the same network as ion."""
        # do some filename munging
        localFn = self.filename
        # need to correct the file path on the server to the one the webserver
        # presents
        remFn = '/'.join(self.remotefilename.split('/')[2:])
        print 'localFn = ',localFn
        print 'remFn = ',remFn
        if removeGZ:
            localFn = localFn.rsplit('.gz')[0]
            remFn = remFn.rsplit('.gz')[0]

        # check to see if file is on remote system
        #chkCmd = ("/usr/bin/curl --connect-timeout %i -I -s -f http://%s/%s"
        #    % (timeout, dataServer, remFn))
        chkCmd = ("/usr/bin/curl --connect-timeout %i -I -s -f http://%s/%s"
            % (timeout, dataServer, remFn))
        print 'chkCmd = ',chkCmd		
        cmd = shlex.split(chkCmd)
        print 'cmd = ',cmd
        p = sb.Popen(cmd)#, stdout = sb.PIPE, stderr = sb.PIPE)
        stdout, stderr = p.communicate()
        # it's there, so go get it
        if stdout.split('\n')[0].find('200') != -1:
            getCmd = ("/usr/bin/curl --connect-timeout %i -s -f -o %s "
            "http://%s/%s " % (timeout, localFn, dataServer, remFn))
            cmd = shlex.split(getCmd)
            # p = sb.Popen(cmd, stdout = sb.PIPE, stderr = sb.PIPE)
            # stdout, stderr = p.communicate()
            try:
                sb.check_call(cmd)
                print "got via HTTP."
                return True
            except:
                pass
        else:
            return False

    def _fetchFileSSH(self, removeGZ = False):
        """SSH data off of data server.
        
        You will have to install your public SSH key on ion for this to work
        because this method depends on passwordless logins."""
        # do some filename munging
        localFn = self.filename
        remFn = self.remotefilename
        if removeGZ:
            localFn = localFn.rsplit('.gz')[0]
            remFn = remFn.rsplit('.gz')[0]

        #check file existence on server
        chkCmd = """ssh %s ls '%s' 2>&1""" % (self.remoteserver, remFn)
        cmd = shlex.split(chkCmd)
        p = sb.Popen(cmd, stdout = sb.PIPE, stderr = sb.PIPE)
        stdout, stderr = p.communicate()
        # it's there, so go get it
        if stdout.split('\n')[0].find('No such file') == -1:
            # old command
            # getCmd = """ssh %s cat '%s' > %s""" % (self.remoteserver,
            #     self.remotefilename, self.filename)
            getCmd = """scp -q %s:'%s'  %s""" % (self.remoteserver, remFn,
                localFn)
            cmd = shlex.split(getCmd)
            try:
                sb.check_call(cmd)
                print "got via SSH."
                return True
            except:
                pass
        else:
            return False

    def _get_data_low(self):
        """Stub that is called by _getData() so that you can do some extra
        stuff before/after calling _read_data_file if you want."""
        self._read_data_file()

    def _read_data_file(self):
        """Low level data reading function.

        Should work for most/all labview output."""

        class raw_data: pass
        my_data = raw_data()
        my_data.fileName = self.filename

        f = self._getFile()
        tmp = f.readline()
        if tmp[0] == '#':
            my_data.headerstrs = tmp
            my_data.headerCols = tmp[1:].strip().split('\t')

        my_data.bulkdata = np.loadtxt(self.filename, delimiter='\t',unpack =
            True)
        my_data.time = my_data.bulkdata[0]
        my_data.cols, my_data.rows = my_data.bulkdata.shape
        my_data.timesec = my_data.time * 1e-6
        self.rawdata = my_data
        self.header = self.rawdata.headerCols
        self.time = self.rawdata.time

class srs_delays(ssx_data):
    """Parses SRS delay file."""

    shotname = ''

    def __init__(self, shotname):
        self.shotname = shotname
        self.runSplit()
        self.fileString = "srs-delays"
        self.fileExt = '.txt'
        self._getData()

    def __str__(self):
        d = self.data
        out = []
        out.append("SSX SRS delays: %s" % (self.shotname))
        for k in d:
            tmp = "  %s: %s %+.12f" % (k, d[k][0], d[k][1])
            out.append(tmp)
        return '\n'.join(out)    

    def __repr__(self):
        return "srs_delays('%s')" % (self.shotname)

    def _get_data_low(self):
        """Reads the data file.
        
        Reads the simple SRS ASCII file."""

        f = self._getFile()
        lines = f.readlines()
        f.close()

        header = [l for l in lines if l.startswith('#')]
        data = [l for l in lines if not l.startswith('#')]

        self.header = ''.join(header)
        self.filecontents = data
        x = ''.join(data)
        dt = np.dtype([('channel', 'a2'), ('reference', 'a1'), ('delay','f8')])
        self.rawdata = np.loadtxt(self.filename, dtype = dt)
        keys = self.rawdata['channel'].tolist()
        values = self.rawdata[['reference','delay']].tolist()
        d = dict(zip(keys,values))
        self.data = d


class scope_data(ssx_data):
    def __init__(self,shotname,scope=None):
        """Sets up instance's title and shotname.

        Reads data from the Lecroy scopes."""

        self.shotname = shotname

        if scope == 1:
            scope = '1'
        if scope == 2:
            scope = '2'
        if scope == 3:
            scope = '3'

        self.runSplit()
        self.scope = scope

        if self.oldFormat:
            data = sdr_old.scope_data(shotname, scope)
            for attr in dir(data):
                if attr not in ['_setScopeProperties', 'plot']:
                    setattr(self, attr, getattr(data, attr))

        else:
            self.fileExt = '.txt.gz'
            if scope in ('1','2','3'):
                self.fileString = "scope%s" % scope
            self._getData()
            self.deltat = self.time[1] - self.time[0]
            self.fft()

    def __str__(self):
        return "SSX data: %s scope %s" % (self.shotname, self.scope)

    def __repr__(self):
        return "scope_data('%s', scope='%s')" % (self.shotname, self.scope)

    def _dataInit(self):
        for i, k in enumerate(self.names):
            setattr(self, k, self.rawdata.bulkdata[self.channels[k],:] *
                self.mult[k])

    def setScopeProperties(self, names=('ch1', 'ch2', 'ch3', 'ch4'), channels =
        (1,2,3,4), mult = (1., 1., 1., 1.), units = None):
        """Set the channel names, mult factors, and units to the scope."""

        self.names = names
        self.mult = dict(zip(names, mult))
        self.channels = dict(zip(names, channels))

        if not units:
            units = ['arb'] * len(names)
        self.units = dict(zip(names, units))
        for name in names:
            ch = "ch%i" % (self.channels[name])
            if name != ch:
                setattr(self, name, getattr(self, ch) * self.mult[name])

    def _get_data_low(self, names = ['ch1', 'ch2', 'ch3', 'ch4'], channels =
        (1,2,3,4), mult = (1., 1., 1., 1.), units = None):
        """Stub that is called by _getData() so that you can do some extra
        stuff before/after calling _read_data_file if you want."""

        self._read_data_file()

        self.ch1 = self.rawdata.bulkdata[1,:]
        self.ch2 = self.rawdata.bulkdata[2,:]
        self.ch3 = self.rawdata.bulkdata[3,:]
        self.ch4 = self.rawdata.bulkdata[4,:]

        self.setScopeProperties(names, channels, mult, units)
        self._dataInit()

    def plot(self, key = True, fig = 1):
        """Quick plot of all channels."""
        # ioff()
        fig = figure(fig)
        fig.clear()

        plot(self.time, self.ch1, label = "%i - %s" % (1, self.names[0]))
        plot(self.time, self.ch2, label = "%i - %s" % (2, self.names[1]))
        plot(self.time, self.ch3, label = "%i - %s" % (3, self.names[2]))
        plot(self.time, self.ch4, label = "%i - %s" % (4, self.names[3]))
        xlabel(self.xlabel)
        ylabel(self.ylabel)
        
        if key:
            legend()
        show()
        # ion()
    
    def plotChan(self, ch = 1, fig = 2):
        """Plot single channel."""
        d = 'ch' + str(ch)
        fig = figure(fig)
        fig.clear()
        plot(self.time, getattr(self, d))
        
    def plotFFT(self, ch = 1, fig = 3):
        """Plot single channel FFT."""
        d = ch - 1
        fig = figure(fig)
        fig.clear()
        loglog(self.w, self.fftw[d])
        title('%s - scope %s, ch %i' % (self.shotname, self.scope, ch))
        xlabel('frequency (MHz)')
        ylabel('power (arb.)')

    def fft(self, time = None):
        """Calculate fft of scope data."""
        t = self.time
        dat = self.rawdata.bulkdata[1:,:]
        if time:
            t0 = time[0]
            t1 = time[1]
            t0 = (t0 + 20) / self.deltat
            t0 = int(t0)
            t1 = (t1 + 20) / self.deltat
            t1 = int(t1)
        else:
            t0 = 0
            t1 = t.shape[0] 

        Nw = dat[:,t0:t1].shape[1]
        f = dat[:, t0:t1]
        
        w = np.fft.fftfreq(Nw, (self.deltat))
        aw = np.fft.fft(f, axis = 1)
        w0 = np.fft.fftshift(w)
        aw = np.fft.fftshift(aw, axes = (1,))
        # if not mod(Nw, 2):
        #     w0 = np.append(w0, -w0[0])
        #     aw = np.append(aw, np.expand_dims(-aw[:,:,0],2), axis=2)
        Nwi = Nw/2
        w2 = w0[Nwi:]
        pw = abs(aw[:, Nwi:])**2

        self.w = w2
        self.fftw = pw

class dtac_data(ssx_data):
    def __init__(self,shotname, fileString = 'ids', probe = 'ids'):
        """Sets up instance's title and shotname.

        For dtac data."""
        self.shotname = shotname
        
        self.fileString = fileString
        self.probe = probe
        self.fileExt = ".tgz"
        self.runSplit()

        self._getData()

        if self.type:
            self._assignChannelNames()

    def __str__(self):
        return "SSX dtac data: %s %s" % (self.shotname, self.fileString)

    def __repr__(self):
        return "dtac_data('%s', '%s')" % (self.shotname, self.fileString)


    def _read_data_file(self):
        """Get some data."""
        self.channelMask = None
        t = tarfile.open(self.filename)
        f, e = os.path.splitext(os.path.basename(self.filename))
        dats = []
        other = []
        for m in t.getnames():
            if os.path.splitext(m)[1] == '.dat':
                dats.append(m)
            else:
                mf, me = os.path.splitext(m)
                # lets guard against reading of accidentally included shot data
                if me != '.tgz':
                    other.append(m)
# 			fn = "%s-ch%02i.dat" % (f, 1)
# 			fninfo = t.getmember(fn)
# 			fileLen = fninfo.size
        dats.sort()

        # read in other files
        self.info = {}
        for m in other:
            fn = m
            fp = t.extractfile(fn)
            s = fp.read()
            name = m.split(f+'-')[1]
            self.info[name] = s

        if self.info.has_key('capturestats.txt'):
            tmp = self.info['capturestats.txt']
            tmp = tmp.split('\n')
            for tm in tmp:
                if 'getChannelMask' in tm:
                    self.channelMaskstr = tm.split('=')[1]
                    self.channelMask = [int(i) for i in
                        list(self.channelMaskstr)]
                if 'getInternalClock' in tm:
                    k = float(tm.split('=')[1].split()[0]) / 1e6
                    self.rclock = round(k) * 1e6
                if 'decimate_clk' in tm:
                    try:
                        k = float(tm.split('=')[1].split()[0])
                        self.clock = round(k) * 1e6
                    except:
                        self.clock = 0
                if 'pre_post_mode' in tm:
                    self.pre_post = tm.split('=')[1].split()
                if 'SRS delay' in tm:
                    k = tm.split('=')[1]
                    self.delay_channel = k

        # if we have a real clock, but no effective clock, copy the real clock
        # over to the effective clock, which is used to calculate the time base
        if not hasattr(self, 'clock') and hasattr(self, 'rclock'):
            self.clock = self.rclock
        if hasattr(self, 'rclock'):
            if self.rclock > 32000000:
                self.clock = self.rclock
# 		for i in arange(1,17):
# 			fn = "%s-ch%02i.dat" % (f, i)
# 			fninfo = t.getmember(fn)
# 			fileLen = fninfo.size
# 			fp = t.extractfile(fn)
# 			s = fp.read()
# 			a = struct.unpack('h'*(fileLen/2), s)
# 	# 		a = arr.array('h')
# 	# 		a.read(fp, fileLen/2)
# 			fp.close()
# 			a = array(a)
# 			data[i-1, :] = a

        # setup for data read in
        fileLen = t.getmember(dats[0]).size

        if self.channelMask:
            chans = where(self.channelMask)[0]
        else:
            chans = arange(len(dats))
        chansl = len(chans)
        self.numberChannels = chansl
        data = zeros((chansl, fileLen/2))
        off_data = zeros((chansl, fileLen/2))
        cdata = zeros((chansl, fileLen/2))
        offsets = zeros(chansl)
        calib = zeros((chansl, 2))

        # read in calibration values
        fn = "%s-vin.txt" % (f)
        fp = t.extractfile(fn)
        s = fp.read()
        fcalib = array([float(x) for x in s.split(',')])
        fcalib = fcalib.reshape(-1,2)

        # read in data
        for k,i in enumerate(chans):
            d = dats[i]
            myf = t.getmember(d)
            fn = d
            fp = t.extractfile(fn)
            s = fp.read()
            a = struct.unpack('h'*(myf.size/2), s)
            fp.close()
            a = array(a)
            data[k, :] = a
            calib[k, :] = fcalib[i, :]

        # apply calibration factors
        for i in xrange(chansl):
            tmp = calib[i][0] + (data[i,:] + 32768) * (calib[i][1] -
                calib[i][0]) / 65535
            # simple offset removal
            offset = tmp[2:100].mean()
            #print 'Check Offsets',offset
            # complex offset removal
            #offset = ma.masked_outside(tmp[2:122], -0.5, 0.5).mean()
            #print 'Check Offsets',offset
            offsets[i] = offset
            cdata[i, :] = tmp
            off_data[i, :] = tmp - offset
        self.dat_fn = dats
        self.offsets = offsets
        self.signal = cdata
        self.signal_off = off_data
        self.rawdata = data
        self.calib = calib
        self.bitnoise = (calib[:,1] - calib[:,0])/2**14

        if hasattr(self, 'clock'):
            deltat = 1./self.clock
            self.deltat = deltat
            l = cdata.shape[1]
            pre = 0
            if hasattr(self, 'pre_post'):
                pre = int(self.pre_post[0])
            time = (arange(0, l, 1.) - pre) * deltat * 1e6
            self.time = time
        else:
            self.clock = 10000000
            deltat = 1./self.clock
            self.deltat = deltat
            self.time = arange(0,400,.1) -20 # #mark TODO fix this time stuff

    def _assignChannelNames(self):
        """Assign channel names from channel-names.txt."""
        r = range(self.numberChannels)
        chans = ['ch' + str(i+1) for i in r]
        if self.info.has_key('channel-names.txt'):
            tmp = self.info['channel-names.txt']
            tmp = tmp.strip().split('\n')
            ctmp = []
            for l in tmp:
                if not l.startswith('#'):
                    ctmp.append(l)
            j = 0
            for i in where(self.channelMask)[0]:
                if i > len(ctmp) - 1:
                    break
                chans[j] = ctmp[i]
                j += 1
            # for i, k in enumerate(ctmp):
            #     chans[i] = k

        # to patch the fuckup by Jim in wiring the original hires mag probe.
        # We check to see if the date is 6/29/11 or before and that the probe
        # is 'm1'.  If it is, reverse the order of the last 8 theta and z
        # channels.
        problemdate = dt.datetime(2011,6,29, 1)
        if self.probe == 'm1' and self.date < problemdate:
            if self.fileString == 'mag1':
                t = chans[24:32]
                t.reverse()
                chans[24:32] = t
            if self.fileString == 'mag2':
                t = chans[8:16]
                t.reverse()
                chans[8:16] = t
        
        # to patch the swapped cables on probe 2.  The actual channel ordering
        # in the files is 9-16 then 1-8 on ALL channels.  I have fixed this in
        # the channel name files, but for data take before 2011-07-29
        problemdate = dt.datetime(2011,7,29, 1)
        if self.probe == 'm2' and self.date < problemdate:
            if self.fileString == 'mag2':
                t1 = chans[24:32]
                t2 = chans[16:24]
                chans[24:32] = t2
                chans[16:24] = t1
            if self.fileString == 'mag3':
                t1 = chans[0:8]
                t2 = chans[8:16]
                t3 = chans[16:24]
                t4 = chans[24:32]
                chans[0:8] = t2
                chans[8:16] = t1
                chans[16:24] = t4
                chans[24:32] = t3
        
        self.channelNames = array(chans)
        chanDict = dict(zip(self.channelNames, r))
        self.channelDict = chanDict

    def plot(self, chan = None):
        """Quick plot of dtac data.
        
        Can plot specified or all channels."""
        # ioff()
        fig = figure(15)
        fig.clear()
        a = axes()
        numChannels = self.signal.shape[0]
        if chan:
            if self.channelDict.has_key(chan):
                i = self.channelDict[chan]
                chanlabel = chan
            elif type(chan) == type(1):
                i = chan - 1
                chanlabel = str(i + 1)
            else:
                i = 0
                chanlabel = str(i + 1)
            a.plot(self.time, self.signal[i, :])
            title('channel %s' % chanlabel)
        else:
            for i in xrange(numChannels):
                a.plot(self.time, self.signal[i])
        ylabel('voltage (V)')
        xlabel('time (us)')
        fig.show()
        # ion()

class dtac_diag(ssx_base):
    """A diagnostic based on dtac data.

    This class assembles data files from multiple dtac units into a logic unit
    representing one diagnostic.  Subclass this when building a new
    diagnostic."""

    shotname = ''
    xlabel = 'time (us)'
    ylabel = 'voltage (V)'

    def __init__(self, shotname, probe = '', gain=1.0, filestrings = None,
        diagname = 'dtac', **kw):
        """Init method.

        Input parameters:

            'shotname' is a typical SSX shotname

            'probe' is a string.  This is the prefix used by channel names in
            the dtac data files.  It lets you grab only the channels that have
            this prefix, ignoring other dtac channels that might be unused or
            used for other diagnostics.  If left blank, all channels from all
            of the dtac data specified will be used.

            'filestrings' is a list of strings.  This setting determines which
            dtac data files to snag.  The strings entered should correspond to
            the 'diag' string in the dtac configuration files used by the dtac
            command line script."""
        self.shotname = shotname
        self.gain = gain
        #print gain
        self.runSplit()
        self.probe = probe
        self.filestrings = filestrings
        self.diagname = diagname
        self.settings = kw
        try:
            self.delays = srs_delays(shotname)
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

    def __str__(self):
        return "SSX dtac diagnostic: %s" % (self.shotname)

    def __repr__(self):
        if not self.filestrings:
            fs = None
        else:
            fs = self.filestrings
        return ("dtac_diag('%s, probe = '%s', filestrings = %s)" %
            (self.shotname, self.probe, fs))

    def _getData(self):
        """Gets the data files specified by the dtac filestrings."""
        filecontents = {}
        for filestring in self.filestrings:
            filecontents[filestring] = dtac_data(self.shotname, filestring,
                self.probe)
        self.filecontents = filecontents

    def renewData(self):
        """Refetch data from server."""
        for f in self.filecontents:
            self.filecontents[f].renewData()
        self._getData()
        self._makeFullData()
        self._processData()
    
    def _processData(self):
        """Stub method for processing data.

        This is meant to be replaced by a method in the subclass of
        dtac_diag."""
        pass

    def _makeFullData(self):
        """Assembly the full data array.

        Assembles the array out of the filecontents from the separate dtac
        units.  Also makes the time variable and some other simple things, like
        offsets and bitnoise numbers."""
        channelDict = {}
        for d in self.filecontents:
            k = self.filecontents[d]
            chanprefix = 'ch'
            for c in k.channelNames:
                if c.startswith(self.probe):
                    chanprefix = self.probe
            for c in k.channelNames:
                if c.startswith(chanprefix):
                    tmp = [d, k.channelDict[c]]
                    channelDict[c] = tmp
        self.channelDict = channelDict
        self.channelNames = array(sort_nat(self.channelDict.keys()))
        self.numChans = len(self.channelNames)

        # snag timing info from one of the units
        self.deltat = k.deltat
        self.clock = k.clock
        if self.delays:
            self.delay_channel = k.delay_channel
            self.delay = self.delays.data[self.delay_channel]
        self.t0 = self.delay[1]
        self.delay_ref = self.delay[0]
        # if our delay ref is T, then we need to subtract off the 0.1 second
        # delay to get to experiment T_0
        if self.delay_ref == 'T':
            self.time = k.time + (self.t0 - 0.1) * 1e6
        else:
            self.time = k.time + self.t0 * 1e6


        # Start putting together a big array with all the data
        self.fullDataRaw = zeros((self.numChans, self.time.shape[0]))
        self.offsets = zeros(self.numChans)
        self.bitnoise = zeros(self.numChans)

        for i,c in enumerate(self.channelNames):
            filestring, chan = self.channelDict[c]
            dtacdata = self.filecontents[filestring]
            signal = dtacdata.signal[chan]
            self.fullDataRaw[i,:] = signal
            self.offsets[i] = dtacdata.offsets[chan]
            self.bitnoise[i] = dtacdata.bitnoise[chan]
        self.removeOffsets()

    def removeOffsets(self):
        """Remove offsets from the signals."""
        self.fullData = self.fullDataRaw.transpose() - self.offsets
        self.fullData = self.fullData.transpose()
        self.fullData[:,0] = self.fullData[:,2:12].mean(1)
        self.fullData[:,1] = self.fullData[:,0]

    def plotRaw(self, chan = None, data = 'fullData'):
        """Plotting routine to check things.

        Can plot any data array based on fullData."""
        # ioff()
        fig = figure(15)
        fig.clear()
        a = axes()
        dat = getattr(self, data)
        numChannels = dat.shape[0]
        if chan:
            if chan in self.channelNames:
                i = self.channelNames.index(chan)
                chanlabel = chan
            elif type(chan) == type(1):
                i = chan - 1
                chanlabel = str(i + 1)
            else:
                i = 0
                chanlabel = str(i + 1)
            a.plot(self.time, dat[i, :])
            title('channel %s' % chanlabel)
        else:
            for i in xrange(numChannels):
                a.plot(self.time, dat[i])
        ylabel(self.ylabel)
        xlabel(self.xlabel)
        fig.show()
        # ion()

    def writeFullData(self, data = 'fullData', outputDir = None):
        """Writes out fullData (or other) structure.

        Meant to be run *after* the data is calibrated.  This way the data can
        be exported to other people."""

        if outputDir:
            fName = os.path.join(os.path.expanduser(outputDir),
                "%s-%s.gz" % (self.shotname, self.diagname))
            os.spawnlp(os.P_WAIT, 'mkdir', 'mkdir', '-p',
                os.path.expanduser(outputDir))
        else:
            fName = ssxutil.ssxPath(
                "%s-%s.gz" % (self.shotname, self.diagname),
                'output', 
                "%s/%s/processed" % (self.runYear, self.runDate),
                mkdir=True)
        gzipper = gzip.GzipFile(fName, 'w')
        # must flatten channel names if it is not 1D
        cn = self.channelNames.reshape(-1)
        header = '#t\t'+'\t'.join(cn) + '\n'
        filecontents = [header]
        # get proper data
        d = getattr(self, data)
        # must reshape fullData if it is not 2D
        if len(d.shape) != 2:
            data = d.reshape((cn.shape[0], -1))
        else:
            data = d

        cols, rows = data.shape
        for row in xrange(rows):
            tmpline = ["%.6f" % self.time[row]]
            for col in xrange(cols):
                tmpline.append("%.6f" % data[col,row])
            tmpline.append('\n')
            filecontents.append('\t'.join(tmpline))
        gzipper.writelines(filecontents)
        gzipper.close()

class DSPrack(object):
    """DSP rack setup file class.
    
    Used for writing mag setup files LabView.  These are used for the DSP rack
    only."""
    def __init__(self, numcards = 17):
        self.numcards = numcards
        self.chanspercard = 8
        self.labels = ['']*self.numcards*self.chanspercard
        self.cards = []

        self._initHeader()

    def _initHeader(self):
        """sets header to default"""
        self.header = ["", ""]
        header = ("first line is the cards that are enabled.  "
            "second is a note.  third and below are channel names.")
        self.setHeader(header, 1)
        self.setHeader('')


    def setHeader(self, header, line = 2):
        self.header[line - 1] = "# %s" % header

    def insert(self, newRack):
        """Insert an rack of info into the current rack"""
        self.cards.extend(newRack.cards)
        fullset = set(self.cards)
        self.cards = list(fullset)
        self.cards.sort()
        for i,n in enumerate(newRack.labels):
            if n != '':
                self.labels[i] = n

    def insertChan(self, card, chan, label):
        """Insert a single channel into the current rack"""
        self.cards.append(card)
        fullset = set(self.cards)
        self.cards = list(fullset)
        self.cards.sort()
        i = (card - 1) * 8 + chan - 1
        self.labels[i] = label

    def writeMagSetupFile(self, filename):
        """Writes the mag setup file to the python output directory."""
        fn = ssxutil.ssxPath(filename, 'output')

        lines = self.header
        cards  = [str(c) for c in self.cards]
        cards = ' '.join(cards)
        lines.append(cards)
        lines.extend(self.labels)

        f = file(fn, 'w')
        f.writelines('\n'.join(lines))
        f.close()

class magnetics_data(ssx_data):
    def __init__(self,shotname):
        """Sets up instance's title and shotname.

        For magnetics data recorded with the DSPs."""

        self.shotname = shotname

        self.fileString = "dsp"
        self.fileExt = ".txt.gz"
        self.runSplit()

        if self.oldFormat:
            data = sdr_old.magnetics_data(shotname)
            for attr in dir(data):
                setattr(self, attr, getattr(data, attr))
        else:
            self._getData()

    def __str__(self):
        return "SSX magnetics data: %s" % (self.shotname)

    def __repr__(self):
        return "magnetics_data('%s')" % (self.shotname)


    def _get_data_low(self):
        self._read_data_file()
        tmp = self.rawdata.headerstrs.split('\t')
        tmp = ['m'+chan for chan in tmp[1:-1]]
        self.names = tmp
        self.totChan = len(self.names)
        tmp = arange(self.totChan) + 1

        self.channels = dict(zip(self.names, tmp))

        self.offsets = {}

        for probe in self.names:
            tmp = self.rawdata.bulkdata[self.channels[probe]]
            offset = tmp[:100].mean()
            self.offsets[probe] = offset

            setattr(self, probe, self.rawdata.bulkdata[self.channels[probe]] -
                offset)

        self.time = self.rawdata.bulkdata[0]

        self._weedChannels()
        self._groupProbes()
        self._groupAxes()
        self._groupChannels()
        self._makeFullData()

    def _weedChannels(self):
        """Stub function for channel removal.

        Remove any channels that don't belong in this probe.  This is called
        automatically, but should be defined in the probe class that is a
        subclass of magnetics_data."""
        pass

    def _groupChannels(self):
        channels = [x[3] for x in self.names]
        channels = list(set(channels))

        channelGroups = {}

        for channel in channels:
            tmp = []
            for chan in self.names:
                if chan[3] == channel:
                    tmp.append(chan)
            channelGroups[channel] = sorted(tmp)
        self.channelGroups = channelGroups
        self.numChan = len(self.channelGroups)

    def _groupProbes(self):
        probes = [x[1] for x in self.names]
        probes = list(set(probes))
        self.probes = probes
        self.numProbes = len(probes)

        probeGroups = {}

        for probe in probes:
            tmp = []
            for chan in self.names:
                if chan[1] == probe:
                    tmp.append(chan)
            probeGroups[probe] = sorted(tmp)
        self.probeGroups = probeGroups

    def _groupAxes(self):
        axes = [x[2] for x in self.names]
        axes = sorted(list(set(axes)))
        self.probeAxes = axes

        axesGroups = {}

        for axis in axes:
            tmp = []
            for chan in self.names:
                if chan[2] == axis:
                    tmp.append(chan)
            axesGroups[axis] = sorted(tmp)
        self.axesGroups = axesGroups

    def _makeFullData(self):
        chans = sorted(self.names)
        w = zeros((self.totChan, len(self.time)))
        for i, chan in enumerate(chans):
            w[i] = getattr(self, chan)
        self.fullData = w
        self.fullDataOrder = chans

    def plotProbe(self, probe, axis):
        fig = figure(23)
        fig.clear()
        for i in xrange(numChan):
            probeNum = "m%i%s%i" % (int(probe), axis, i+1)
            plot(self.time[5:], getattr(self, probeNum)[5:], label='%s' %
                probeNum)
        ylabel('voltage (V)')
        xlabel('time (us)')
        legend()

    def plotChan(self, probename):
        fig = figure(23)
        fig.clear()
        plot(self.time[5:], getattr(self, probename)[5:], label='%s' %
            probename)
        ylabel('voltage (V)')
        xlabel('time (us)')
        legend()

    def plotAll(self):
        fig = figure(24)
        fig.clear()
        n = self.fullData.shape[0]
        for i in xrange(n):
            plot(self.time[5:], self.fullData[i,5:])
        ylabel('voltage (V)')
        xlabel('time (us)')

    def writeFullData(self, outputDir = None):
        """Writes out fullData structure.

        Meant to be run *after* the magnetics data is calibrated.  This way the
        data can be exported to other people."""

        if outputDir:
            fName = os.path.join(os.path.expanduser(outputDir),
                self.shotname+'mag.gz')
            os.spawnlp(os.P_WAIT, 'mkdir', 'mkdir', '-p',
                os.path.expanduser(outputDir))
        else:
            fName = ssxutil.ssxPath(self.shotname+'-mag.gz', 'output',
                self.runYear + '/' + self.runDate + '/' + self.shotname +
                '/mag', mkdir=True)
        gzipper = gzip.GzipFile(fName, 'w')
        header = '#t\t'+'\t'.join(self.fullDataOrder)+'\n'
        filecontents = [header]
        cols, rows = self.fullData.shape
        for row in xrange(rows):
            tmpline = ["%.6f" % self.time[row]]
            for col in xrange(cols):
                tmpline.append("%.6f" % self.fullData[col,row])
            tmpline.append('\n')
            filecontents.append('\t'.join(tmpline))
        gzipper.writelines(filecontents)
        gzipper.close()

class rga_data(ssx_data):
    """RGA data class."""

    data, mass, pressure = array([]), array([]), array([])
    filename = ''
    xlabel = 'mass (amu)'
    ylabel = 'pressure (torr)'
    header = ''
    dataRe = re.compile(r"""^\s*(\d+\.\d+),\s*(-?\d+\.\d+E[-+]\d+),\s*$""",
        re.X|re.I|re.M)

    def __init__(self, shotname):
        """Sets up data.
        
        Shotname should just be the date."""

        self.fileString = "rga"
        self.fileExt = ".txt.gz"
        self.shotname = shotname
        self.runSplit()

        self._getData()
        self._binData(massCorrection = True)

    def __str__(self):
        return "SSX rga data: %s" % (self.shotname)

    def __repr__(self):
        return "rga_data('%s')" % (self.shotname)


    def _altNameProcess(self):
        self.runSplit()
        basename = "%s/rga/%s%s/%s-%s%s" % (self.runYear, self.runMonth,
            self.runYearShort, self.runDate, self.fileString, self.fileExt)
        self.filename = ssxutil.ssxPath(basename,'data')

        basename = "%s/%s%s/%s-%s%s" % (self.runYear, self.runMonth,
            self.runYearShort, self.runDate, self.fileString, self.fileExt)
        self.remotefilename = os.path.join('/ssx/data/rga/', basename)
        self.remoteserver = dataServer

    def _read_data_file(self):
        """Reads data file.

        Reads RGA program ASCII files."""

        f = self._getFile()
        lines = f.readlines()
        f.close()

        x = ''.join(lines)
        x = x.replace('\r\n','\n')
        header, data = x.split('\n\n\n')
        self.header = header

        g = self.dataRe.findall(data)
        if g:
            y = array(g, dtype='f').transpose()
            self.mass, self.rawdata = y[0], y[1]
        self.pressure = np.where(self.rawdata > 0, self.rawdata, 0)

    def _binData(self, massCorrection = True):
        """Bins data 1 per AMU and calculates total pressure."""
        dx = self.mass[1]-self.mass[0]
        maxx = self.mass.max()
        binningNum = round(1/dx)
        self.pressure2 =  zeros(maxx)

        if massCorrection:
            self.pressure2[0] = max(self.pressure[:binningNum - int(binningNum
                * .2)])
            for i in xrange(1,maxx):
                self.pressure2[i] = max(self.pressure[i * binningNum  -
                    int(binningNum * .2):i * binningNum + binningNum  -
                    int(binningNum * .2)])
        else:
            for i in xrange(1,maxx):
                self.pressure2[i] = max(self.pressure[i * binningNum:i *
                    binningNum + binningNum])
        self.totalPressure = self.pressure2.sum()

    def plot(self, log=False, save=False):
        """Plots rga data."""
        fig = figure(1, **ssxdef.f)
        fig.clear()
        a = axes()
        if log:
            a.semilogy(self.mass, self.pressure)
            ylim(ymin=1e-10)
        else:
            a.plot(self.mass, self.pressure)
            ylim(ymin=0)
        a.text(0.93, 0.92, self.dateString, horizontalalignment='right',
            verticalalignment='top', transform = a.transAxes)
        a.text(0.93, 0.87, "B_p = %.1e" % self.totalPressure,
            horizontalalignment='right', verticalalignment='top', transform =
            a.transAxes)
        ylabel(self.ylabel)
        xlabel(self.xlabel)
        filename = os.path.splitext(self.filename)[0]+'.pdf'
        if save:
            fig.savefig(filename)

    def bar(self, save=False):
        """Plots rga data as a bar chart."""
        fig = figure(1, **ssxdef.f)
        fig.clear()
        a = axes()
        a.bar(arange(1, self.mass.max()+1), self.pressure2)
        ylim(ymin=0)
        a.text(0.93, 0.92, self.dateString, horizontalalignment='right',
            verticalalignment='top', transform = a.transAxes)
        a.text(0.93, 0.87, 'B_p = %.1e' % self.totalPressure,
            horizontalalignment='right', verticalalignment='top', transform =
            a.transAxes)
        ylabel(self.ylabel)
        xlabel(self.xlabel)
        filename = os.path.splitext(self.filename)[0]+'.pdf'
        if save:
            fig.savefig(filename)

class oo_data(ssx_data):
    """OceanOptics data class."""

    data = {}
    filename = ''
    xlabel = 'wavelength (nm)'
    ylabel = 'intensity (arb)'
    header = ''
    pixRe = re.compile(r"""<numberOfPixels>(\d*)</numberOfPixels>""", re.X)
    valuesRe = re.compile(r"""<double>([\d.-]*)</double>""", re.X)

    def __init__(self, shotname):
        """Sets up data.
        
        Shotname should just be the name of the file.  Leave off the trailing
        '-oo.ProcSpec'.  So to read 'argon-gdc-oo.ProcSpec', one should call
        oo_data('argon-gdc')."""

        self.fileString = "oo"
        self.fileExt = ".ProcSpec"
        self.shotname = shotname
        self.runSplit()

        self._getData()

    def __str__(self):
        return "SSX ocean optics data: %s" % (self.shotname)

    def __repr__(self):
        return "oo_data('%s')" % (self.shotname)

    def _altNameProcess(self):
        """Construct names for non-shot files."""
        basename = "%s-%s%s" % (self.shotname, self.fileString, self.fileExt)
        self.filename = ssxutil.ssxPath('spectrometer/' + basename,
            'data')
        self.remoteserver = dataServer
        self.remotefilename = os.path.join('/ssx/data/spectrometer/', basename)

    def _parseChunk(self, txt, splitTxt):
        """parses data chunks from the OO binary files"""
        out = txt.split(splitTxt)
        if len(out) > 1:
            source, rest = out

            results = self.valuesRe.findall(source)
            if results:
                data = array(results, dtype = 'f')
            return data, rest
        else:
            return None

    def _read_data_file(self):
        """reads OO binary files.

        OO binary files are nothing more than zip files with a couple xml files
        in them.  We read the main xml file, parse it for a couple different
        spectrums if they exist, and return a dictionary of the data found in
        the file."""

        z = zipfile.ZipFile(self.filename)
        for i in z.namelist():
            if i[:3] == 'ps_':
                f = z.read(i)

        results = self.pixRe.search(f)
        if results:
            self.numPix = int(results.groups()[0])

        out = self._parseChunk(f, '</sourceSpectra>')
        if out:
            dat, rest = out
            dat = dat.reshape(2, self.numPix)

        self.data['wavelengths'] = dat[1]
        self.data['sourceSpec'] = dat[0]


        out = self._parseChunk(rest, '</processedPixels>')

        if out:
            self.data['processSpec'], rest = out
            out = self._parseChunk(rest, '</darkSpectrum>')
        if out:
            self.data['darkSpec'], rest = out
            out = self._parseChunk(rest, '</referenceSpectrum>')
        if out:
            self.data['refSpec'], rest = out

        for k in self.data.keys():
            setattr(self, k, self.data[k])


    def plot(self, writeFiles = False, spec = 'sourceSpec'):
        """Plots the source spectrum."""
# 		ioff()
# 		data = readOObin(file)
        fig = figure(17, **ssxdef.specsize)
        fig.clear()
        a = axes()
        a.plot(self.wavelengths, self.data[spec], lw='.5')
        a.text(0.93, 0.92, self.shotname, horizontalalignment='right',
            verticalalignment='top', transform = a.transAxes)
        ylabel(self.ylabel)
        xlabel(self.xlabel)
        xlim(200, 1100)
        if writeFiles:
            ssxutil
            fName = ssxutil.ssxPath(self.shotname+'-oo.pdf', 'output',
                self.runYear + '/' + self.runDate + '/' + self.shotname, mkdir
                = True)
            fig.savefig(fName)
        else:
            fig.show()
            # ion()

    def plot2(self, writeFiles = False):
        """Plots the dark subtracted spectrum."""
        fig = figure(17, **ssxdef.specsize)
        fig.clear()
        a = axes()
        a.plot(self.wavelengths, self.data['sourceSpec'] -
            self.data['darkSpec'], lw='1')
        a.text(0.93, 0.92, self.shotname, horizontalalignment='right',
            verticalalignment='top', transform = a.transAxes)
        ylabel(self.ylabel)
        xlabel(self.xlabel)
        xlim(200, 1100)
        if writeFiles:
            ssxutil
            fName = ssxutil.ssxPath(self.shotname+'-oo.pdf', 'output',
                self.runYear + '/' + self.runDate + '/' + self.shotname, mkdir
                = True)
            fig.savefig(fName)
        else:
            fig.show()
            ion()

    def saveData(self):
        """Saves the source spectrum as a text file."""
        head = ("# ocean optics output - file %s\n#wavelength\tpixel value" %
            os.path.basename(self.filename))
        fName = ssxutil.ssxPath(self.shotname+'-oo.txt', 'output', self.runYear
            + '/' + self.runDate + '/' + self.shotname, mkdir = True)
        write_data(fName, array((self.data['wavelengths'],
            self.data['sourceSpec'])), header=head)

    def findPeaks(self, threshold = 1150, showPlot = False):
        """Peak detection"""
        px,py = mf.peakDetect(self.wavelengths, self.sourceSpec,
            threshold=threshold)
        self.peaks = px
        if showPlot:
            self.plot()
            plot(px,py,'ro')
            plot([self.wavelengths[0], self.wavelengths[-1]], [threshold,
                threshold], c = 'gray', ls = '--')
            xlim(200, 1100)

#############
# Functions #
#############

def sort_nat(l):
    """ Sort the given list in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    nat = sorted(l, key=alphanum_key )
    return nat


### file utilities ###

def listRuns(runDate, server = "ion.physics.swarthmore.edu"):
    runYear = '20' + runDate[4:6]
    basedir = '%s/%s' % (runYear, runDate)
    remoteDir = os.path.join('/ssx/data/', basedir)
    cmd = "ssh %s ls %s" % (server, remoteDir)
    output = os.popen(cmd)
    o = output.readlines()
    o = ''.join(o)
    o = o.split()
    o = [n.split('-')[0] for n in o]
    o = list(set(o))
    try:
        o.remove(runDate)
    except:
        pass
    return o

def fetchRun(run, server = "ion.physics.swarthmore.edu"):
    """Download data from a given run."""
    runDate, runNumber = run.split('r')
    runYear = '20' + runDate[4:6]
    basedir = '%s/%s' % (runYear, runDate)
    localDir = ssxutil.ssxPath(basedir, 'data')
    remoteDir = os.path.join('/ssx/data/', basedir)
    if not os.path.exists(localDir):
        os.spawnlp(os.P_WAIT, 'mkdir', 'mkdir', '-p', localDir)
    cmd = "scp %s:%s/%s* %s/" % (server, remoteDir, run, localDir)
    output = os.popen(cmd)

def fetchRunDay(run, server = "ion.physics.swarthmore.edu"):
    """Download data from a given run day."""
    runDate = run.split('r')[0]
    runYear = '20' + runDate[4:6]
    basedir = '%s/%s' % (runYear, runDate)
    localDir = ssxutil.ssxPath(basedir, 'data')
    remoteDir = os.path.join('/ssx/data/', basedir)
    if not os.path.exists(localDir):
        os.spawnlp(os.P_WAIT, 'mkdir', 'mkdir', '-p', localDir)
    cmd = "scp %s:%s/* %s/" % (server, remoteDir, localDir)
    output = os.popen(cmd)

def decomposeFileName(filepath):
    """Split filename into components.

    Shotname, date, run, device"""
    filename = os.path.basename(filepath)
    shotname, tmp = filename.split('-')
    device = tmp.split('.')[0]
    date, run = shotname.split('r')
    keys = ['shotname', 'date', 'run', 'device']
    return dict(zip(keys, (shotname, date, run, device)))

def read_visit_curve(name):
    """Returns curves from a visit curve file."""
    if os.path.exists(name):
        thefile = file(name)
        lines = thefile.readlines()
        thefile.close()

        x = []
        y = []
        xall, yall = [], []
        for k,i in enumerate(lines):
            tmp = i.split()
            if tmp != [] and (tmp[0] not in ('#', 'x', 'y', '%', 'c')) :
                x.append(float(tmp[0]))
                y.append(float(tmp[1]))
            elif tmp[0] == '#' and k != 0:
                xall.append(array(x))
                yall.append(array(y))
                x, y = [], []
        xall.append(array(x))
        yall.append(array(y))

        return xall, yall
    else:
        raise "File does not exist."

def read_graphclick_file(name):
    """Returns 2-d data from graphclick file.

    Simple routine.  Needs the first line to be header."""
    if os.path.exists(name):
        thefile = file(name)
        lines = thefile.readlines()
        thefile.close()

        x = []
        y = []

        for i in lines:
            tmp = i.split()
            if tmp != [] and (tmp[0] not in ('#', 'x', 'y', '%', 'c')) :
                x.append(float(tmp[0]))
                y.append(float(tmp[1]))
        x = array(x)
        y = array(y)

        return x, y
    else:
        raise "File does not exist."

def read_spectrasuite_file(name):
    """Returns 2-d data from spectrasuite file.

    Simple routine.  Needs the first line to be header."""
    if os.path.exists(name):
        thefile = file(name)
        lines = thefile.readlines()
        thefile.close()

        x = []
        y = []

        indata = False
        for i in lines:
            if i[0] == ">":
                indata = not indata
                continue
            if indata:
                tmp = i.split()
                x.append(float(tmp[0]))
                y.append(float(tmp[1]))
        x = array(x)
        y = array(y)

        return x, y
    else:
        raise "File does not exist."

def readfile(filename):
    """Returns lines from specified file."""
    if path.exists(filename):
        thefile = file(filename)
        lines = thefile.readlines()
        thefile.close()
        return lines
    else:
        raise "File does not exist."

def fileLinesToArray(lines):
    """Takes output from readfile and sticks it in a float array."""
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.', '-']
    header = None
    offset = 0
    tmpline = lines[0]
    if tmpline[0] == '#':
        header = tmpline
        header = header[1:].split()
        cols = len(lines[1].split())
        rows = len(lines) - 1
        offset = 1
    else:
        cols = len(tmpline.split())
        rows = len(lines)
    output = zeros((cols, rows), 'float')
    for i in xrange(rows):
        tmpline = lines[i+offset].split()
        if tmpline[0][0] == "#":
            continue
        if tmpline[0][0] not in numbers:
            continue
        for j in xrange(len(tmpline)):
            output[j][i] = float(tmpline[j])
    if header:
        return output, header
    else:
        return output
