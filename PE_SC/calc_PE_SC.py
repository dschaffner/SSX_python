import numpy as np
from math import factorial
from collections import Counter

def PE_dist(data,n=5,delay=1):
    '''
    function Cjs - Returns the Shannon Permutation Energy
    Input:
        data  - 1-D array
        n     - permutation number (default=5)
        delay - integeter delay (default=1)
    Output:
        Sp - Shannon permuation entropy
        Se - Shannon + Uniform permutation entropy
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
    
def PE_calc_only(count,tot_permutations,n=5):
    Ptot=tot_permutations
    N=factorial(n)
    invPtot=1./Ptot     #Inverse for later calcuations
    S = 0.
    Se = 0.
    for q in iter(count.values()):
        q*=invPtot #convert to probability
        S += -q * np.log2(q)
        q+=1./N
        q/=2
        Se += -q * np.log2(q)
    for i in range(len(count),N):
        q=1./2/N
        Se += -q * np.log2(q)
    return S,Se
    
def PE(data,n=5,delay=1):		
    '''
    function Cjs - Returns the Shannon Permutation Energy
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
        raise TypeError('Data must be a 1-D array')
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
    for q in iter(count.values()):
        q*=invPtot #convert to probability
        S += -q * np.log2(q)
        q+=1./N
        q/=2
        Se += -q * np.log2(q)
    for i in range(len(count),N):
        q=1./2/N
        Se += -q * np.log2(q)
    return S,Se


def CH(data,n,delay=1):
    '''
    function Cjs - Returns the normalized Jensen-Shannon statistical complexity
    Input:
        data  - array
        n     - permutation number
        delay - integeter delay (default=1)
    Output:
        C - Normalized Jensen-Shannon complexity
        H - Normalized Shannon Perumation Entropy
    '''		
    N  = factorial(n)
    S, Se  = PE(data,n,delay)   
    C = -2.*((Se - 0.5*S - 0.5*np.log2(N))
            /((1 + 1./N)*np.log2(N+1) - 2*np.log2(2*N) 
            + np.log2(N))*(S/np.log2(N)))

    return S/np.log2(N), C
