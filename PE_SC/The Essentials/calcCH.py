import numpy as np
from math import factorial
from calcS import calcS
from collections import Counter

def calcCH(data,n,delay=1):
    '''
    function Cjs - Returns the normalized Jensen-Shannon statistical complexity
    Input:
        data  - array
        n     - embedding dimension (default=5)
        delay - embedding delay (default=1)
    Output:
        C - Normalized Jensen-Shannon complexity
        H - Normalized Shannon Perumation Entropy
    '''		
    N  = factorial(n)
    S, Se  = calcS(data,n,delay)   
    C = -2.*((Se - 0.5*S - 0.5*np.log2(N))
            /((1 + 1./N)*np.log2(N+1) - 2*np.log2(2*N) 
            + np.log2(N))*(S/np.log2(N)))

    return S/np.log2(N), C