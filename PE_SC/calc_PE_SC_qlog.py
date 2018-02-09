import numpy as np
from math import factorial
from collections import Counter

def qlog(q,x):
    if q==0:
        return "First argument cannot be zero"
    if q>0 and x>0 and q!=1:
        a=x**(1-q)
        b=(1-q)
        return (a-1)/b
    if x==0:
        return 0
    if q==1 and x>0:
        return np.log(x)
        
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
        Su - Uniform permutation entropy
        ''' 
    T=np.array(data)
    if len(T.shape)>1:
        raise TypeError, 'Data must be a 1-D array'
    t = len(T)
    Ptot = t - delay*(n - 1)    #Total number of order n permutations in T
    #print 'Number of permutations = ', Ptot
    A = []			 #Array to store each permutation
    
    for i in range(Ptot):	#Will run through all possible n segments of T
        A.append(''.join(T[i:i+(n-1)*delay+1:delay].argsort().astype(str)))
    #Count occurance of patterns
    count=Counter(A)
    return count,Ptot
    
def q_PE_calc(count,tot_permutations,n=5,qnum=100,stepsize=1e-4):
    Ptot=tot_permutations
    N=factorial(n)
    invPtot=1./Ptot     #Inverse for later calcuations
    S = np.zeros([qnum+1])
    Se = np.zeros([qnum+1])
    Su = np.zeros([qnum+1])
    for k in np.arange(1,qnum+1):   
        kq=k*stepsize
        for q in count.itervalues():
            q*=invPtot #convert to probability, that is p_j
            if q!=0:
                print 'For k=',kq,' and q=',q,' qlog=',qlog(kq,1.0/q)
                S[k] = S[k] + (q * qlog(kq,1.0/q)) #sum of p_j * log(1/p_j) #Recall that log(1/a)=-log(a)
            
            q+=1./N
            q/=2
            Se[k] = Se[k] + (q * qlog(kq,1.0/q))
        for i in xrange(len(count),N):
            q=1./2/N
            Se[k] = Se[k] + (q * qlog(kq,1.0/q))
        Su[k]=qlog(kq,N)
    return S,Se,Su
    
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
        raise TypeError, 'Data must be a 1-D array'
    t = len(T)
    Ptot = t - delay*(n - 1)    #Total number of order n permutations in T
    print 'Number of permutations = ', Ptot
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
    for i in xrange(len(count),N):
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
