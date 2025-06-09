##################################################################
# Compute Shannon Entropy for single and multivariate time series
##################################################################
#
import numpy as np
import scipy.stats as scst
from scipy.spatial import cKDTree
from scipy.special import gamma, digamma
import sys
sys.path.append('./')
import statsTools
#
class entropy:
    """
    Shannon Entropy Computed by different methods
    """

    def __init__(self,x,verbose=False):
        self.x = x
        self.verbose = verbose

    def binning(self,nbin=10,tol=1e-14):
        """
        Binning method for estimating entropy of x

        Args:
           `nbin`: int, number of bins
           `iplot`: bool, whether or not plot the histogram
        """
        self.nbin = nbin
        self.tol = tol
        p, binEdge = statsTools.hist(self.x,self.nbin,self.verbose)
        H = -np.sum(p*np.log(p + self.tol))
  
        return H
    
    def kde(self,method='mc',**kwargs):
        """
        Kernel density estimation (KDE)-based method for computing Shannon entropy

        Args:
          `x`: 2d numpy array of shape(n,dim), n: number of samples, dim: number of variates
          `method`: string, integration method: default 'mc' (Monte Carlo)

        Return:
          `H`: float, Shannon entropy estimated by the KL method
        """

        self.method = method


        if self.method == 'mc':   
           #Monte Carlo method 
           kde_ = scst.gaussian_kde(self.x.T, bw_method='scott')
           kde__ = kde_(self.x.T)
           H = -np.mean(np.log(kde__))

        if self.method in ['simpson','romberg']:
           for key, value in kwargs.items(): 
               if key == 'nInteg':
                  self.nInteg = value 

           raise ValueError('Not Implemented')       
           H=10000       

        return H

    def kl(self,k=3):
        """
        Kozachenko-Leonenko entropy estimator using k-th nearest neighbour points.

        Args:
          `x`: 2d numpy array of shape(n,dim), n: number of samples, dim: number of variates
          `k`: int, k-th nearest points to each sample

        Return:
          `H`: float, Shannon entropy estimated by the KL method
        """
        self.k = k
        n = self.x.shape[0]
        if self.x.ndim == 1:
           x = self.x[:,None]
        dim = x.shape[1]

        #find the distance of x[i] from points within the k-th neighbour
        tree = cKDTree(x)
        r, I = tree.query(x, k + 1)  # k+1 instead of k, as the x[i] is within the set (1st closest to itself)

        #comnpute epsilon and cd
        eps = 2.*r[:, -1]  # twice the distance between x[i] and the k-th nearest neighbour
        cd = np.pi**(dim/2.)/gamma(1+dim/2.)/2.**dim

        # KL estimator for entropy
        H = - digamma(k) + digamma(n) + np.log(cd) + dim * np.mean(np.log(eps))
        return H

