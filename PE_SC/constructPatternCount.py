import numpy as np
from collections import Counter

def constructPatternCount(data,n=5,delay=1):
    '''
    

    Parameters
    ----------
    data : 1-D array
        time-series data.
    n : integer,optional
        embedding dimension. The default is 5.
    delay : integer, optional
        embedday delay. The default is 1.

    Returns
    -------
    A Count occurance of patterns and total number of permutations

    '''
    
    T=np.array(data)
    if len(T.shape)>1:
        raise TypeError('Data must be a 1-D array')
    t = len(T)
    Ptot = t - delay*(n - 1)    #Total number of order n permutations in T
    #print 'Number of permutations = ', Ptot
    A = []			 #Array to store each permutation
    
    for i in range(Ptot):	#Will run through all possible n segments of T
        A.append(''.join(T[i:i+(n-1)*delay+1:delay].argsort().astype(str)))
    #Count occurance of patterns
    count=Counter(A)
    return count,Ptot