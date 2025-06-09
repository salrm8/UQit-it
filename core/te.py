##################################################################
# Compute transfer entropy beween time series
##################################################################
#
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import gamma, digamma
from sklearn.neighbors import NearestNeighbors
import sys
sys.path.append('./')
from entropy import entropy
#
class bi_te:    
    """
    Estimate Transfer Entropy (TE) for a pair of time series
    """

    def __init__(self,x,y,embDim):
        """
        Args: 
           `x`: 1d numpy array of size n, containing samples of source time series
           `y`: 1d numpy array of size n, containing samples of target time series
           `embDim`: int (>0), embedded dimension (number of delayed samples0

        Return:
            `te`: float, transfer entropy (x->y) for embedded dimension `embDim`
        """
        self.x = x
        self.y = y
        self.embDim = embDim

        self.n = self.x.shape[0]
        assert self.y.shape[0] == self.n

    def kde(self,method='mc',**kwargs):
        """
        Estimate transfer entropy using KDE method

        TE(x->y) = H(y,y_past) - H(y_past) - H(y,y_past,x_past) - H(y_past,x_past)

        Args:
           `method`: string, integration method: default 'mc' (Monte Carlo)       
        """
        self.kdeMethod = method

        xPast = self.x[:-self.embDim]
        yPast = self.y[:-self.embDim]
        yFutu = self.y[self.embDim:]

        H_yPast = entropy(yPast).kde(method=self.kdeMethod,**kwargs)
        H_yFutu_yPast = entropy(np.vstack((yFutu,yPast)).T).kde(method=self.kdeMethod,**kwargs)
        H_yPast_xPast = entropy(np.vstack((yPast,xPast)).T).kde(method=self.kdeMethod,**kwargs)
        H_yFutu_yPast_xPast = entropy(np.vstack((yFutu,yPast,xPast)).T).kde(method=self.kdeMethod,**kwargs)

        te = H_yFutu_yPast - H_yPast - H_yFutu_yPast_xPast + H_yPast_xPast
        return te

    def ksg(self,k=3,tol=1e-8):    
        """
        Estimate transfer entropy using the KSG-based method (KNN type). 

        TE is defined as a conditional mutual information:        
        TE(x->y) = MI(y,x_past|y_past)

        Reference:
        J. Witter and C. Houghton, arXiv:2403.00556v3, 2024. 

        Args:
           `k`: int, k-th nearest points to each sample
           `tol`: float, tolerance (small value)
        """
        self.k = k
        self.tol = tol

        #Create delayed time series
        xPast = np.array([self.x[i-self.embDim: i][::-1] for i in range(self.embDim,self.n)])  
        yPast = np.array([self.y[i-self.embDim: i][::-1] for i in range(self.embDim,self.n)])  
        yFutu = self.y[self.embDim:self.n]  

        #Create joint samples
        XZ = np.hstack((yFutu[:,None], yPast))   
        YZ = np.hstack((xPast, yPast))   
        XYZ = np.hstack((yFutu[:,None], xPast, yPast))

        # Create kd trees
        tree_XYZ = cKDTree(XYZ)

        # find distance of each point from its k-th neighbours in the joint set
        r_Z, _ = tree_XYZ.query(XYZ, k + 1, p=np.inf)

        eps_Z = r_Z[:, -1]  - self.tol

        #Count neighbors in marginal spaces (X-space and Y-space)
        knn_YZ = NearestNeighbors(n_neighbors=k+1, p=np.inf,metric='chebyshev').fit(YZ)
        knn_XZ = NearestNeighbors(n_neighbors=k+1, p=np.inf,metric='chebyshev').fit(XZ)
        knn_Z = NearestNeighbors(n_neighbors=k+1, p=np.inf,metric='chebyshev').fit(yPast)

        # collect the index of points falling within the ball of radius epsilon around each point
        nYZ = np.array([knn_YZ.radius_neighbors([YZ[i]], eps_Z[i], 
                        return_distance=False)[0] for i in range(self.n-self.embDim)], dtype=object)
        nXZ = np.array([knn_XZ.radius_neighbors([XZ[i]], eps_Z[i], 
                        return_distance=False)[0] for i in range(self.n-self.embDim)], dtype=object)
        nZ = np.array([knn_Z.radius_neighbors([yPast[i]], eps_Z[i], 
                        return_distance=False)[0] for i in range(self.n-self.embDim)], dtype=object)

        # count the number of points in the vicinity of each point, excluding itself
        nZ = np.array([float(len(i) - 1) for i in nZ])
        nYZ = np.array([float(len(i) - 1) for i in nYZ])
        nXZ = np.array([float(len(i) - 1) for i in nXZ])

        te = digamma(k) + np.mean(digamma(nZ + 1) - digamma(nXZ + 1) - digamma(nYZ + 1))
        #te = digamma(k) + np.mean(digamma(nZ) - digamma(nXZ) - digamma(nYZ))
        #te = digamma(k) - 2./k + np.mean(digamma(nZ) - digamma(nXZ) - digamma(nYZ) - 1./nXZ - 1./nYZ)
        return te
