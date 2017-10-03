#simple magnetic calibration

import os
import sys

#import matplotlib
#matplotlib.use('Agg')
#matplotlib.rc('axes', grid=False)

from pylab import *
from numpy import *
# from ssx import *
import ssx_data_read_david as sdr
import ssx_py_utils_david as ssxutil
import scipy as sp
from matplotlib.pylab import *
import ssx_functions as ssxf
import hiresmag_david as hdr

#load current
def loadcurrent(shot, scope = '3'):
    data = sdr.scope_data(shot, scope)
    time = data.time    
    eastcurrent = -data.ch2*170000#East gun 170kA/V
    westcurrent = -data.ch4*190000#West gun 190kA/V    
    return time,eastcurrent,westcurrent

def polyPeak(time, data, timerange = [40,80],pretrigger=20):
    """Finds the peak in the specified time range.

    Finds the peak in the data in the specified time range.  You must pass it
    pretrigger and timebase as well as the time and data series.
    - timebase - is an integer equal to (1 us)/(delta t of the digitizer).  For
      example, if capturing data at 10 MHz, timebase = (1e-6/1e-7) = 10.  For
      data captured at 2 MHz, timebase = (1e-6/5e-7) = 2.  In other words, how
      many data points per us.  It's a pretty bad way to code this up, but it
      lets you specify the time range in micro seconds in which to look for the
      peak.
    - pretrigger - how many microseconds of data before t = 0."""

    # Find the indices corresponding to the ends of the time range
    t1 = ssxf.tindex_min(time,timerange[0]+pretrigger)
    t2 = ssxf.tindex_min(time,timerange[1]+pretrigger)
    #print t1
    #print t2
    # generate an array of indices spanning the range
    ti = arange(t1,t2)
    # get the time and data points in the range
    t = time[ti]
    d = data[ti]
    # Fit a 2nd degree polynomial and find the min and max.
    p = polyfit(t,d,2)
    fit = p[0]*t**2 + p[1]*t + p[2]
    dataMax = fit.max()
    dataMin = fit.min()
    plt.plot(t,d/max(d))
    plt.plot(t,fit/max(fit))
    if abs(dataMin) > dataMax:
        dataMax = dataMin
    return dataMax

#compute helmholz coil Bfield
def helmholtz2(r, i = 1.0, coil = 2):
    """Compute B field of Helmholtz coil at (x,y,z)
    
    i is current in amps and coil is the coil selection.
        - 1 for the old wooden coil
        - 2 for the new delrin coil"""
    r1, r2, r3 = r
    
    # a is the radius of the coil in meters, and d is the distance between the
    # two coils.
    if coil == 1:
        # coil 1 is the old wooden coil.  It has two turns per side, a radius
        # of 6.1" and a separation of 5.9".
        a = 6.1 * 0.0254
        d = 5.9 * 0.0254
        turns = 2
    elif coil == 2:
        # Coil 2 is the new delrin and copper pipe tube built in 2011.  Each
        # side has one turn.  The ID of the pipe is 6.125" and the OD is 6.625,
        # so we split the difference to calculate the radius.  Same goes for
        # the separation - the coils are 3/4" high, so we go from the midline.
        # These numbers were come from page 26-28 in Tim's 2011 lab notebook
        # and helmholtz.nb.
        a = 0.0809625
        d = 0.085725
        turns = 1

    b = zeros(3)

    for j in xrange(360):	# compute line integral
        dth = 1./360 * 2 * pi
        th = j * dth

        rho = sqrt( (r1 - a * cos(th))**2 + (r2 - a * sin(th))**2 + (r3 - .5 *
            d)**2 )
        b[0] = b[0] + a * dth * (- r3 * cos(th))/rho**3
        b[1] = b[1] + a * dth * (- r3 * sin(th))/rho**3
        b[2] = b[2] + a * dth * ( (r1 - a * cos(th)) * cos(th) + (r2 - a *
            sin(th)) * sin(th) )/rho**3

        rho = sqrt( (r1 - a * cos(th))**2 + (r2 - a * sin(th))**2 + (r3 + .5 *
            d)**2 )
        b[0] = b[0] + a * dth * (- r3 * cos(th))/rho**3
        b[1] = b[1] + a * dth * (- r3 * sin(th))/rho**3
        b[2] = b[2] + a * dth * ( (r1 - a * cos(th)) * cos(th) + (r2 - a *
            sin(th)) * sin(th) )/rho**3
    

    # calculate the be field.  The 4pi's cancel out, and the 1e4 is to convert
    # to gauss
    b = b * i * 1.0e-7 * turns * 1e4
    return b
    
def calib_by_shot(shot,index_arr=[2,10]):
    magdata = hdr.getMagData(shot)
    #uncalibBdot = magdata.unCalibData[index_arr[0],index_arr[1],:]
    uncalibB=magdata.iUnCalibData[index_arr[0],index_arr[1],:]
    magmax = polyPeak(magdata.time,uncalibB,[20,90],pretrigger=0)
    
    guntime,east,west=loadcurrent(shot)
    currmax = polyPeak(guntime,east,[20,90],pretrigger=20)
    
    helmB=helmholtz2([0,0,0],currmax)
    
    ratio = magmax/helmB
    print 'Mag Max', magmax
    print 'HelmB', helmB
    print 'Ratio', ratio
    
    return magmax,currmax,helmB
    