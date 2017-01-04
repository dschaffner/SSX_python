#!/usr/bin/env python
"""Generic math functions."""

from scipy import *
import scipy.optimize as optimize
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from scipy import signal

import numpy as np

from pylab import *

# $Id$
# $HeadURL$

# 12/5/08 3:24 PM by Tim Gray
# A670DC7C-22ED-4E2D-A0BE-05634F828073

# 1.1 - 2/24/09 - added in a couple functions, double gaussian fit, sliding median filter, and a peak detect function

# 1.1.1 - 5/14/09 - fixing up fitting routines to properly calculate reduced chi2, where chi2 = sum(residuals)/(N-M) - N is number of points and M is the number of fitting parameters

__author__ = "Tim Gray"
__version__ = "1.1"


def shift_left(x):
    """Shifts i=0 to i=len(x)"""

    y = x.tolist()
    y.append(y.pop(0))
    return array(y)

def shift_right(x):
    """Shifts i=len(x) to i=0"""

    y = x.tolist()
    y.insert(0,y.pop())
    return array(y)	

def deriv(x, y):
    """IDL like derivative.

    Takes x and y.  Very similar to diff function, but instead of
    returning an array of length n-1, it returns length n.  Mimics
    IDL's deriv function."""
    n = len(x)
    if n < 3:
        print "Must have at least 3 points"
    else:
        if len(x) != len(y):
            print "Vectors must have same size"
            return
        d = (shift_left(y) - shift_right(y))/(shift_left(x) - shift_right(x))
        d[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (x[2] - x[0])
        d[-1] = (3.0 * y[-1] - 4.0 * y[-2] + y[-3])/(x[-1] - x[-3])
    return d

def smooth(array, avg):
    """Smooth - like IDL's smooth function."""

    a = arange(float(avg))
    a[:] = 1 / float(avg)
    return convolve(array, a, 1)

def flatten_list(thelist):
    """"Flattens" a shot list"""
    newlist = []
    for i in thelist:
        for j in i:
            newlist.append(j)

    return newlist

def sampFreq(time):
    """Calculates sampling frequency from time data."""
    t0 = time[0]
    t1 = time[1]

    return 1/(t1 - t0)

def powerSpec(time, signal, detrend = detrend_none):
    """Calculate a power spectrum.

    Returns f and power with inputs of time, the signal, and detrending."""
    signal = detrend(signal)
    V = fft(signal)
    N = len(signal)
    sf = sampFreq(time)
    power = abs(V[:N/2])**2

    f = arange(0,N+1) / (N /sf)
    f = f[:(len(f)-1)/2]
    return f, power

def gaussian(x, mu, sigma):
    """Gaussian function."""
    sigma = abs(sigma)
    g = (1 / (sigma * sqrt(2 * pi))) * exp((-(x - mu)**2) / (2 * sigma**2))
# 	g = exp((-(x - mu)**2) / (2 * sigma**2))
    return g

def scaled_gaussian(x, mu, sigma, scale, positive = False):
    """Scaled gaussian."""
    if positive:
        scale = abs(scale)
    return gaussian(x, mu, sigma) * scale

def doubleGaussian(x, mu1, sigma1, scale1, mu2, sigma2, scale2, positive = False):
    """Double gaussian."""
    g1 = scaled_gaussian(x, mu1, sigma1, scale1, positive) 
    g2 = scaled_gaussian(x, mu2, sigma2, scale2, positive)
    return g1 + g2

def pGaussian(x, mu, sigma, scale):
    """Scaled gaussian constrained to be positive."""
    return scaled_gaussian(x, mu, sigma, scale, 1)

def pOffsetGaussian(x, mu, sigma, scale, offset):
    """Scaled gaussian constrained to be positive."""
    return scaled_gaussian(x, mu, sigma, scale, 1) + offset

def pDoubleGaussian(x, mu1, sigma1, scale1, mu2, sigma2, scale2):
    """Double gaussian constrained to be positive."""
    return doubleGaussian(x, mu1, sigma1, scale1, mu2, sigma2, scale2, 1)

def curveFit(f, x, y, p0 = None, sigma = None, **kw):
    """Generalized curve fitting function.  

    Uses non-linear least squares to fit a function, f, to data.

    Parameters
    ----------
    f: callable
        The model function, f(x, ...).  It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments - f(x, *parameters).
    x: N-length array
        The independent variable where the data is measured.
    y: N-length array
        The dependent data --- nominally f(xdata, ...)
    p0 : None, scalar, or M-length sequence
        Initial guess for the parameters.  If None, then the initial
        values will all be 1 (if the number of parameters for the function
        can be determined using introspection, otherwise a ValueError
        is raised).
    sigma : None or N-length sequence
        If not None, it represents the standard-deviation of ydata.
        This vector, if given, will be used as weights in the
        least-squares problem.

    Returns
    -------
    p: array
        fitted parameters
    cov: 2d array
        Estimated covariance of p.  The diagonals provide the variance of the
        parameters.
    chi2: number
        Reduced chi^2 for the fit.
    success: number
        Success of the fit

    See Also
    --------
    scipy.optimize.curve_fit
    scipy.optimize.leastsq

    Notes
    -----
    Modelled after scipy.optimize.curve_fit().  I'm guessing we can really just
    remove this function and use curve_fit instead, but I'll keep it here
    anyway.  And probably continue to use it.

    """
    if p0 is None or isscalar(p0):
        # determine number of parameters by inspecting the function
        import inspect
        args, varargs, varkw, defaults = inspect.getargspec(f)
        if len(args) < 2:
            raise ValueError, "p0 not given as a sequence and inspection"\
                " cannot determine the number of fit parameters"
        if p0 is None:
            p0 = 1.0
        p0 = [p0]*(len(args)-1)

    args = (x, y, f)

    def resid(p, x, y, f):
        return y - f(x, *p)

    def wresid(p, x, y, f, weights):
        return weights*(y - f(x, *p))

    if sigma is None:
        r = resid		
    else:
        args += (1.0/ asarray(sigma), )
        r = wresid

    plsqFull = optimize.leastsq(r, p0, args = args, full_output=True, **kw)

    p = plsqFull[0]
    cov = plsqFull[1]
    success = plsqFull[4]
# 	print p, success
    # For no weighting,
    # 	 Weights(i) = 1.0.
    # For instrumental (Gaussian) weighting,
    # 	 Weights(i)=1.0/sigma(i)^2
    # For statistical (Poisson)  weighting,
    # 	 Weights(i) = 1.0/y(i), etc.

    # calculate the error factor according to section 15.2 of Numerical Recipes (which I got from the IDL routine gaussfit)

    # if sigma is passed in, it is put in the denominator of r() and then squared, so we get our (y(x) - y)^2 / sigma^2.  This assumes sigma is std of measurements.  If we want poisson stats, pass in sqrt(y), then we get (y(x) - y)^2/ y (equation 4.33 from bevington).  Otherwise 
    if (len(y) > len(p)) and cov is not None:
        chi2 = sum( r(p, *args)**2 )
        chi2 = chi2 / (len(x) - len(p)) # reduced chi2
        # If no weighting, then we need to boost error estimates by
        # sqrt(chisq).  we are multiplying the errors by sqrt(chi2) - so
        # multiply the covariance matrix by chi2 and when you sqrt it later it
        # works out.
        #
        # I think this also corresponds to equation 6.23 in Bevington.  The
        # basic idea is that we have 'common uncertainties', so sigma_i =
        # sigma.  The sigma_xx (covariance bits) reduce to 6.23.  Notice there
        # is a sigma^2 in both of those that are not in 6.21 and 6.22.  So we
        # need to mult by that.  sigma^2 is defined in 6.14, which is...
        # chi2_reduced.
        if sigma is None:
            cov = cov * chi2
    else:
        chi2 = inf
        cov = inf

    return p, cov, chi2, success

def tempfit(f,x,y,p0, sigma=None):

    p, cov, chi2, success = curveFit(f,x,y,p0, sigma=sigma)
    fig = figure(23)
    fig.clear()
    plot(x,y,'bo')
    if sigma is not None:
        errorbar(x,y,sigma,fmt = None, color='gray')
    if type(x) == type(ma.zeros(0)):
        x = x.compressed()
    x2 = arange(x[0], x[-1], (x[1] - x[0])/10.)
    y2 = f(x2, *p)
    plot(x2,y2)
    print "chi^2 = %f" % chi2
    ee = sqrt(cov.diagonal())
    for i in xrange(len(ee)):
        print "%.2f +/- %.2f" % (p[i], ee[i])
    return p, ee

# A helper function to make histograms
def histplot(data, bins, shift = True, *p, **kw):
    # fix histogram bins
    bindx = bins[1] - bins[0]
    if (len(bins) - 1) == len(data):
        if shift:
            bins = bins[1:] - (bindx)/2
        else:
            bins = bins[1:]
    step(bins,data, *p, **kw)

# don't need these guys anymore now that I discovered *p notation
# def scaled_gaussian2(x, p, positive = False):
# 	mu, sigma, scale = p
# 	return scaled_gaussian(x, mu, sigma, scale, positive)
# 
# def doubleGaussian2(x, p, positive = False):
# 	mu1, sigma1, scale1, mu2, sigma2, scale2 = p
# 	return doubleGaussian(x, mu1, sigma1, scale1, mu2, sigma2, scale2, positive)
# 
# 
# 
# def fitGaussian(x, y, mu = None, sigma = None, scale = None):
# 	if not mu:
# 		mu = (x[len(x)/2] + x[len(x)/2 + 1]) / 2
# 	if not scale:
# 		scale = max(y)
# 	if not sigma:
# 		meanx = mean(x)
# 		sigma = sqrt( sum((x - meanx)**2)/(len(x)-1))
# 	def resid(p, x, y):
# 		(mu, sigma, scale) = p
# 		return y - scaled_gaussian(x, mu, sigma, scale)
# 		
# 	p0 = (mu, sigma, scale)
# 	plsqFull = optimize.leastsq(resid, p0, args = (x, y),full_output=True)
# 	p = plsqFull[0]
# 	cov = plsqFull[1]
# 	success = plsqFull[4]
# 	mu, sigma, scale = p
# 
# 	#If no weighting, then we need to boost error estimates by sqrt(chisq).
# 	# calculate the error factor according to section 15.2 of Numerical Recipes (which I got from the IDL routine gaussfit)
# 	# we are multiplying the errors by sqrt(chi2)
# 	chi2 = sum((resid(p, x, y))**2)/(len(x) - len(p))
# 	cov = cov * chi2
# 
# 	fit = scaled_gaussian(x, mu, sigma, scale)
# 
# 	return fit, p, cov, chi2, success
# 
# def fitDoubleGaussian(x, y, mu = None, sigma = None, scale = None, q = None, positive = False):
# 	if not mu:
# 		mu1 = (x[len(x)/2] + x[len(x)/2 + 1]) / 2
# 		mu2 = mu1
# 	else:
# 		mu1, mu2 = mu
# 
# 	if not scale:
# 		scale1 = max(y)
# 		scale2 = scale1
# 	else:
# 		scale1, scale2 = scale
# 		
# 	if not sigma:
# 		meanx = mean(x)
# 		sigma1 = sqrt( sum((x - meanx)**2)/(len(x)-1))
# 		sigma2 = sigma1 
# 	else:
# 		sigma1, sigma2 = sigma
# 
# 	def resid(p, x, y):
# 		(mu1, sigma1, scale1, mu2, sigma2, scale2) = p
# 		return y - doubleGaussian(x, mu1, sigma1, scale1, mu2, sigma2, scale2, positive)
# 
# 		
# 	p0 = (mu1, sigma1, scale1, mu2, sigma2, scale2)
# 	plsqFull = optimize.leastsq(resid, p0, args = (x, y),full_output=True)	
# 	p = plsqFull[0]
# 	cov = plsqFull[1]
# 	success = plsqFull[4]
# 	(mu1, sigma1, scale1, mu2, sigma2, scale2) = p
# 
# 	#If no weighting, then we need to boost error estimates by sqrt(chisq).
# 	# calculate the error factor according to section 15.2 of Numerical Recipes (which I got from the IDL routine gaussfit)
# 	# we are multiplying the errors by sqrt(chi2)	
# 	chi2 = sum((resid(p, x, y))**2)/(len(x) - len(p))	
# 	cov = cov * chi2
# 	
# 
# 	if (type(q) == type(arange(2))):
# 		x = arange(x[q[0]], x[q[-1]])
# 	
# 	fit = doubleGaussian(x, mu1, sigma1, scale1, mu2, sigma2, scale2, positive)
# 
# 	return fit, p, cov, chi2, success

def test(a,b,c,d,e,f,g):
    x = arange(-10,10,1)
    y1 = doubleGaussian(x,a,b,c,d,e,f) + randn(len(x))*g
# 	y1 = scaled_gaussian(x,a,b,c) + randn(len(x))*g
    y2 = scaled_gaussian(x,a,b,c)
    y3 = scaled_gaussian(x,d,e,f)

    fit = fitGaussian(x, y1)


    g = fitDoubleGaussian(x, y1, (-1,1))
    i = 0
    while ((fit[3] > 0.01) and (g[3] > .75 * fit[3]) and (i < 20)):
        g = fitDoubleGaussian(x, y1)
        i = i + 1
    a,b,c,d,e,f = g[1]
    y4 = scaled_gaussian(x,a,b,c)
    y5 = scaled_gaussian(x,d,e,f)
    f = figure(1)
    f.clear()
    a = axes()
    a.plot(x,y1,'bo')
    a.plot(x,y2,'r-')
    a.plot(x,y3,'g-')
    a.plot(x,y4,'r--')
    a.plot(x,y5,'g--')
    a.plot(x,fit[0],'c-')
    a.plot(x,g[0],'k-')
    show()
    print g[1]
    print fit[1]
    print g[3]
    print fit[3]

def slidingMedian(x, window):
    if hasattr(x, 'ndim'):
        if x.ndim == 1:
            l = x.shape[0]
            z = zeros(l)
            for k in xrange(window - 1, l - window):
                tmp = sort(x[k-(window - 1):k+(window - 1)])
                z[k] = tmp[window-1]
            return z
    print "input must be 1-dim array"
    return 

def peakDetect(x, y, threshold = None, showPlot=False, gauss = False, gsize=3):
    """Detects peaks.

    Fits a spline to the data, evaluates the derivative, and does some counting to find the peaks.  The gauss (and gsize) use a gaussian derivative convolved with the data to find peaks and doesn't work properly."""

    if threshold > 0:
        y2 = where(y>threshold, y, threshold)
    elif threshold < 0:
        y2 = where(y<threshold, y, threshold)
    else:
        y2 = y

    spl = interpolate.splrep(x,y2)
    s = interpolate.splev(x, spl)
    if gauss:
        # experimental for now
        q = arange(-gsize,gsize+1)
        g = - q * exp(-(q**2/float((0.5*gsize)**2)))
        ds = signal.convolve(s, g, mode='same')
        ds[0]=0
    else:
        ds = interpolate.splev(x, spl, 1)

    threshold2 = .1*ds.max()
    t1 = where(ds > threshold2, 1, 0)
    t2 = where(ds < -threshold2, -1, 0)
    t = t1+t2

    flag1 = False
    flag2 = False
    j, k = 0, 0
    peaks = []
    for i in xrange(len(t)):
        if not flag1 and not flag2:
            if t[i] == 1:
                flag1 = True
                j = i
    # 		elif t == -1:
    # 			flag = True
        elif flag1 and not flag2:
            if t[i] == -1:
                flag2 = True
            else:
                pass
        elif flag1 and flag2:
            if t[i] != -1:
                k = i
                peaks.append((k + j) / 2)
                flag1, flag2 = False, False
    px,py = x[peaks], y[peaks]
    if showPlot:
        f1 = figure(1)
        f1.clear()
        plot(x,y, 'b-')	
        plot(x,y2, c = 'gray')
        plot(px,py,'ro')


        f2 = figure(2)
        f2.clear()
# 		plot(x,ds)
        plot(x,t)


    return px, py
