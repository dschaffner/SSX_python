#Calibration Calculator
import numpy as np



u0 = 4.0*np.pi*1e-7
def BT_to_BG(BT):
    BG = BT*10000.
    return BG

def B_wire_T(I,r): #Current I in Amps, #Distance r in meters
    B = u0*(I)/(2*np.pi*r)
    return B
    
def B_wire_G(I,r): #Current I in Amps, #Distance r in meters
    B = u0*(I)/(2*np.pi*r)
    B = BT_to_BG(B)
    return B    
    
def coilflux_I(I,l,r,nloops=1): #Current in A, l is distance from wire in m, r is radius in m
    flux = nloops*B_wire_T(I,l)*np.pi*r**2
    return flux #in T-m
    
def emf_approx(I,l,r,f,nloops=1): #Current in A, l is distance from wire in m, r is radius in m, frequency in f
    flux = coilflux_I(I,l,r,nloops=nloops)
    deltaflux = 2.0*flux
    halfperiod = (1.0/f)/2
    emf = deltaflux/halfperiod #in Volts
    return emf