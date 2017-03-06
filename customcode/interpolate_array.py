#interpolate_array

from scipy.interpolate import interp1d
from numpy import linspace

def interpolate_array(newlength,xarr,yarr):
    newx = linspace(xarr[0],xarr[-1],newlength)
    newy = interp1d(xarr,yarr)
    output = newy(newx)
    return newx,output