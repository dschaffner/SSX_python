import numpy as np
from math import factorial
from collections import Counter

def calc_H(data,n=5,delay=1):		
    '''
    function Cjs - Returns the Normalized Shannon Permutation Energy, H
    Input:
        data  - 1-D array
        n     - permutation number (default=5)
        delay - integeter delay (default=1)
    Output:
        Sp - Shannon permuation entropy
        Se - Shannon + Uniform permutation entropy
    ''' 
    N=factorial(n)
    T=np.array(data)
    if len(T.shape)>1:
        raise TypeError( 'Data must be a 1-D array')
    t = len(T)
    Ptot = t - delay*(n - 1)    #Total number of order n permutations in T
    print('Number of permutations = ', Ptot)
    invPtot=1./Ptot     #Inverse for later calcuations
    A = []			 #Array to store each permutation
    S = 0.
    Se = 0.
    
    for i in range(Ptot):	#Will run through all possible n segments of T
        A.append(''.join(T[i:i+(n-1)*delay+1:delay].argsort().astype(str)))
    #Count occurance of patterns
    count=Counter(A)
    #print len(count)
    #Calculate S from the count
    for q in count.itervalues():
        q*=invPtot #convert to probability
        S += -q * np.log2(q)
        q+=1./N
        q/=2
        Se += -q * np.log2(q)
    for i in range(len(count),N):
        q=1./2/N
        Se += -q * np.log2(q)
    return S/np.log2(N),Se/np.log2(N)