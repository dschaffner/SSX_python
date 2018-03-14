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
        
def PE_dist(data,d=5,delay=1):
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
    Ptot = t - delay*(d - 1)    #Total number of order n permutations in T
    #print 'Number of permutations = ', Ptot
    A = []			 #Array to store each permutation
    
    for i in range(Ptot):	#Will run through all possible n segments of T
        A.append(''.join(T[i:i+(d-1)*delay+1:delay].argsort().astype(str)))
    #Count occurance of patterns
    count=Counter(A)
    return count,Ptot
    
def q_PE_calc(count,tot_permutations,d=5,qnum=100,stepsize=1e-4):
    Ptot=tot_permutations
    D=factorial(d)
    invPtot=1./Ptot     #Inverse for later calcuations
    S = np.zeros([qnum+1])
    Su = np.zeros([qnum+1])
    for q_index in np.arange(1,qnum+1):   
        q=q_index*stepsize
        for p_count in count.itervalues():
            pj=p_count/Ptot #propablity of count of permutation over total
            if p_count!=0:
                print 'For q=',q,' and p_count=',p_count,' qlog=',qlog(q,1.0/pj)
                S[q_index] = S[q_index] + (pj * qlog(q,1.0/pj)) #sum of p_j * log(1/p_j) #Recall that log(1/a)=-log(a)
        Su[q_index]=qlog(q,N)
    return S,Su
    
def q_PESC_calc(count,tot_permutations,d=5,qnum=100,stepsize=1e-4):
    Ptot=tot_permutations
    D=factorial(d)
    invPtot=1./Ptot     #Inverse for later calcuations
    S = np.zeros([qnum+1])#Shannon q-entropy
    Su = np.zeros([qnum+1])#Uniform Shannon q-entropy (normalization for H)
    Du = np.zeros([qnum+1])#Disequliubrium
    Dstar = np.zeros([qnum+1])#max disequilibrium (normalization for C)
    for q_index in np.arange(1,qnum+1):   
        q=q_index*stepsize
        sum1=0.0
        sum2=0.0
        for p_count in count.itervalues():
            print 'p_count is ',p_count
            pj=p_count*invPtot #propablity of count of permutation over total
            if p_count!=0:
                
                print 'For q=',q,' and p_count=',p_count,' qlog=',qlog(q,1.0/pj)
                #Shannon Entropy
                S[q_index] = S[q_index] + (pj * qlog(q,1.0/pj)) #sum of p_j * log(1/p_j) #Recall that log(1/a)=-log(a)
                #Disequilibrium
                #sum 1 of disequilibrium
                sum1 = sum1+(-0.5*pj*qlog(q,(pj+(1.0/D))/(2*pj)))
            #sum 2 of disequilibrium
            sum2 = sum2+(-0.5*(1.0/D)*qlog(q,(pj+(1.0/D))/(2.0/D)))
        Su[q_index]=qlog(q,D)
        Du[q_index]=sum1+sum2
        #six terms in Complexity normalization
        t1=D*2**(2-q)
        t2=-(1+D)**(1-q)
        t3=-D*(1+(1.0/D))**(1-q)
        t4=-D
        t5=1.0
        t6=(1-q)*D*2**(2-q)
        Dstar[q_index]=(t1+t2+t3+t4+t5)/t6
    return S,Su,Du,Dstar