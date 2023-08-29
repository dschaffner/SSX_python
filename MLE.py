#Maximum Likelihood Estimation

import numpy as np

def mle_range(fft,w,reweightKS=True,norm_min=False):

    
    #This function finds the alpha using the MLE and then minimizes 
    #the KS statistic over a range of minimum and maximum frequencies
    xmins_indices = np.arange(357,358)#np.arange(1,101)
    xmaxes_indices = np.arange(800,2000)#np.arange(300,601)
    minmax_ks_arr = np.zeros((len(xmins_indices),len(xmaxes_indices)))
    minKS_ind_range = [0,0]
    minKS = 10.0
    fit_alpha = 0.0
    
    for imin in xmins_indices:
        print ('On ',imin)
        for imax in xmaxes_indices:
            if imax % 100 == 0: print ('On ',imax)
    
            alpha = np.arange(0,5.1,0.01)
            L_vals = np.zeros(len(alpha))
    
            #Power Likelihood
            S = fft[imin:imax] #S is power of each frequency
            x = w[imin:imax] #x is the frequency bin
            x = 3.0*x #makes sure x is all integer values
    
            #Power Sum
            Stot = float(S.sum())
            #Log of x
            lnx = np.log(x)
            #Abs Product of Power*Log x
            sumSlnx = np.abs(S*lnx)
            #Sum of above
            sumSlnx = sumSlnx.sum()
    
            for a in np.arange(len(alpha)):
                #print 'Min =',imin,' Max = ',imax,' alpha = ',alpha[a]
                zeta = x**(-alpha[a])
                L_vals_t1 = (-alpha[a]*sumSlnx)
                L_vals_t2 = (-Stot*np.log(zeta.sum()))
                L_vals[a] = (-alpha[a]*sumSlnx) - (Stot*np.log(zeta.sum()))
                #print 'For alpha = ',alpha[a],' L_vals_t1 = ',L_vals_t1,' and L_vals_t2 = ',L_vals_t2
    
                max_alpha_index = np.argmax(L_vals)
                max_alpha = alpha[max_alpha_index]
    
            #calculate CDFs for KS method
            #data
            cdf_dat = np.cumsum(S/Stot)
            pdf_mod = x**(-max_alpha)
            pdf_norm = pdf_mod.sum()
            cdf_mod = np.cumsum(pdf_mod/pdf_norm)
            KSdenom = np.sqrt(np.abs((cdf_mod)*(1-cdf_mod)))
            if reweightKS:
                KS = np.max(np.abs(cdf_dat-cdf_mod)/KSdenom)
            else:
                KS = np.max(np.abs(cdf_dat-cdf_mod))
            
            
            #test for minimum KS
            #print KSdenom
            if KS <= minKS:
                minKS = KS
                minKS_ind_range = [imin,imax]
                fit_alpha = max_alpha
    
    return fit_alpha,minKS,minKS_ind_range

def mle(fft,w,xmin,xmax,norm_min=False):
    #Power Likelihood
    S = fft[xmin:xmax] #S is power of each frequency
    if norm_min:
        S = fft[xmin:xmax]/np.min(fft[xmin:xmax]) #S is power of each frequency
    x = w[xmin:xmax] #x is the frequency bin
    x = 3.0*x #makes sure x is all integer values
    
    #print S
    Stot = float(S.sum())
    #print Stot
    lnx = np.log(x)
    sumSlnx = np.abs(S*lnx)
    sumSlnx = sumSlnx.sum()
    
    alpha = np.arange(0,10.1,0.01)
    L_vals = np.zeros(len(alpha))
    
    for a in np.arange(len(alpha)):
        zeta = x**(-alpha[a])
        L_vals_t1 = (-alpha[a]*sumSlnx)
        L_vals_t2 = (-Stot*np.log(zeta.sum()))
        L_vals[a] = (-alpha[a]*sumSlnx) - (Stot*np.log(zeta.sum()))
        #print 'For alpha = ',alpha[a],' L_vals_t1 = ',L_vals_t1,' and L_vals_t2 = ',L_vals_t2
    
    max_alpha_index = np.argmax(L_vals)
    max_alpha = alpha[max_alpha_index]
    
    #calculate CDFs for KS method
    #data
    cdf_dat = np.cumsum(S/Stot)
    pdf_mod = x**(-max_alpha)
    pdf_norm = pdf_mod.sum()
    cdf_mod = np.cumsum(pdf_mod/pdf_norm)
    KS = np.max(np.abs(cdf_dat-cdf_mod))
    return max_alpha,alpha,L_vals,cdf_dat,cdf_mod,KS
    
def mle_werr(fft,w,xmin,xmax,norm_min=False):
    #Power Likelihood
    S = fft[xmin:xmax] #S is power of each frequency
    if norm_min:
        S = fft[xmin:xmax]/np.min(fft[xmin:xmax]) #S is power of each frequency
    x = w[xmin:xmax] #x is the frequency bin
    n = len(x)
    print ('n is ',n, 'and xmin is ',w[xmin])
    x = 3.0*x #makes sure x is all integer values
    
    #print S
    Stot = float(S.sum())
    #print Stot
    lnx = np.log(x)
    sumSlnx = np.abs(S*lnx)
    sumSlnx = sumSlnx.sum()
    
    alpha = np.arange(0,10.1,0.01)
    L_vals = np.zeros(len(alpha))
    
    for a in np.arange(len(alpha)):
        zeta = x**(-alpha[a])
        L_vals_t1 = (-alpha[a]*sumSlnx)
        L_vals_t2 = (-Stot*np.log(zeta.sum()))
        L_vals[a] = (-alpha[a]*sumSlnx) - (Stot*np.log(zeta.sum()))
        #print 'For alpha = ',alpha[a],' L_vals_t1 = ',L_vals_t1,' and L_vals_t2 = ',L_vals_t2
    
    max_alpha_index = np.argmax(L_vals)
    max_alpha = alpha[max_alpha_index]
    
    #calculate error
    zeta_max = x**(-max_alpha)
    zeta_max = zeta_max.sum()
    d1zeta_max = -1.0*(lnx)*(x**(-max_alpha))
    d1zeta_max = d1zeta_max.sum()
    d2zeta_max = (lnx**2)*(x**(-max_alpha))
    d2zeta_max = d2zeta_max.sum()
    sigma = 1.0/np.sqrt(n*((d2zeta_max/zeta_max)-(d1zeta_max/zeta_max)**2))
    
    #calculate CDFs for KS method
    #data
    cdf_dat = np.cumsum(S/Stot)
    pdf_mod = x**(-max_alpha)
    pdf_norm = pdf_mod.sum()
    cdf_mod = np.cumsum(pdf_mod/pdf_norm)
    KS = np.max(np.abs(cdf_dat-cdf_mod))
    return max_alpha,alpha,L_vals,cdf_dat,cdf_mod,KS,sigma
    
def mle_wexp(fft,w,xmin,xmax,norm_min=False):
    #Power Likelihood
    S = fft[xmin:xmax] #S is power of each frequency
    if norm_min:
        S = fft[xmin:xmax]/np.min(fft[xmin:xmax]) #S is power of each frequency
    x = w[xmin:xmax] #x is the frequency bin
    x = 3.0*x #makes sure x is all integer values
    
    #print S
    Stot = float(S.sum())
    #print Stot
    lnx = np.log(x)
    sumSlnx = np.abs(S*lnx)
    sumSlnx = sumSlnx.sum()
    
    alpha = np.arange(0,2.2,0.1)
    gamma = np.arange(0,2.2,0.1)
    lamb = np.arange(0,2.2,0.1)
    L_vals = np.zeros([len(alpha),len(gamma),len(lamb)])
    
    for a in np.arange(len(alpha)):
        print ('On alpha',a)
        for g in np.arange(len(gamma)):
            for l in np.arange(len(lamb)):
                zeta = (np.exp(-lamb[l]*x**(gamma[g])))*(x**(-alpha[a]))
                L_vals_t1 = (-alpha[a]*sumSlnx)
                print ('a,g,l =',a,g,l)
                print ('t1',L_vals_t1)
                L_vals_t2 = (-lamb[l]*Stot*x**(gamma[g]))
                L_vals_t2 = L_vals_t2.sum()
                print ('t2',L_vals_t2)
                L_vals_t3 = (-Stot*np.log(zeta.sum()))
                print ('t3',L_vals_t3)
                
                L_vals[a,g,l] = L_vals_t1+L_vals_t2+L_vals_t3
                #print 'For alpha = ',alpha[a],' L_vals_t1 = ',L_vals_t1,' and L_vals_t2 = ',L_vals_t2
    
    #max_alpha_index = np.argmax(L_vals)
    #max_alpha = alpha[max_alpha_index]
    
    #calculate CDFs for KS method
    #data
    #cdf_dat = np.cumsum(S/Stot)
    #pdf_mod = x**(-max_alpha)
    #pdf_norm = pdf_mod.sum()
    #cdf_mod = np.cumsum(pdf_mod/pdf_norm)
    #KS = np.max(np.abs(cdf_dat-cdf_mod))
    return L_vals#,alpha,L_vals,cdf_dat,cdf_mod,KS