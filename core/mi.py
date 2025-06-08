##################################################################
# Compute mutual information between two time series
##################################################################
#
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import gamma, digamma
import sys
sys.path.append('./')
from entropy import entropy

class mi:
    """
    Mutual information between two time series x and y

    Args:
       `x`: 1d numpy array of size n
       `y`: 1d numpy array of size n
       `lag`: positive int, lag between the time series; lag==0: for random variables

    Return:
       `Ixy`: mutual information between x and y
    """
    def __init__(self,x,y,lag=0):
        """

        """
        self.x = x
        self.y = y
        self.lag = lag

        if self.lag>0:
           self.x = self.x[:-self.lag] 
           self.y = self.y[self.lag:] 

    def kde(self,method='mc'):
        """
        MI based on entropies estimated by the KDE method
        I(X,Y) = H(X) + H(Y) - H(X,Y)
        """
        self.method = method
        Hx = entropy(self.x).kde(method = self.method)
        Hy = entropy(self.y).kde(method = self.method)
        Hxy = entropy(np.vstack((self.x,self.y)).T).kde(method = self.method)
        Ixy = Hx + Hy - Hxy
        return Ixy

    def ksg(self,k=3):        
        """
        MI based on entropies estimated by the KL (Kozachenko-Leonenko) method following the KSG approach. 
        This method relies on the KNN method. 

        Reference: 
        A. Kraskov, H. Stogbauer, P. Grassberger (KSG), Phys. Rev. E 69, 066138, 2011. 

        Args:
           `k`: int, k-th nearest points to each sample
        """
        if self.x.ndim == 1:
           x = self.x[:,None]
        if self.y.ndim == 1:
           y = self.y[:,None]

        n = x.shape[0]

        xy = np.hstack((x, y))  #joint samples (n,2)

        # Create kd trees
        tree_x = cKDTree(x)
        tree_y = cKDTree(y)
        tree_xy = cKDTree(xy)

        # find distance of each point from its k-th neighbours in the joint set
        r, _ = tree_xy.query(xy, k + 1, p=np.inf)

        eps = r[:, -1]  # distance of each point from farthest in the k-th neighbours

        # collect the index of points falling within the ball of radius epsilon around each point
        nx = np.array([tree_x.query_ball_point(x[i], eps[i], p=np.inf) for i in range(n)],
                       dtype=object)
        ny = np.array([tree_y.query_ball_point(y[i], eps[i], p=np.inf) for i in range(n)],
                       dtype=object)

        # count the number of points in the vicinity of each point, excluding itself
        nx = np.array([float(len(i) - 1) for i in nx])
        ny = np.array([float(len(i) - 1) for i in ny])

        # compute mutual information based on KL method
        Ixy = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)) #-1./k
        return Ixy

class mi_multiLag:
    """
    Estimate MI between x(t) and y(t) at a set of given lags

    Returns:
       `miList`: 1d numpy array, array of MI values associated with the lagList
       `maxMI`: float, maximum MI among the considered lagList
       `maxMIlag`: int, the lag at which max MI is computed
    """
    def __init__(self,x,y,lagList):
        """
        Args:
          `x`: 1d numpy array of size n
          `y`: 1d numpy array of size n
          `lagList`: List of positive int, list of lags between the time series; 
        """  
        self.x = x
        self.y = y
        self.lagList = lagList

    def kde(self,method='mc'):
        """
        MI based on entropies estimated by the KDE method at lags in lagList
        I(X,Y) = H(X) + H(Y) - H(X,Y)
        """
        self.method = method
        miList = []
        maxMI = -100.0

        for lag_ in self.lagList: 
            mi_ = mi(self.x, self.y,lag=lag_).kde(method=self.method)
            if mi_ > maxMI: 
               maxMI = mi_
               maxMIlag = lag_
            miList.append(mi_)
        return np.asarray(miList), maxMI, maxMIlag

    def ksg(self,k=3):
        """
        MI based on entropies estimated by the KL (Kozachenko-Leonenko) method following the KSG approach.
        This method relies on the KNN method.
        """
        self.k = k
        miList = []
        maxMI = -100.0

        for lag_ in self.lagList: 
            mi_ = mi(self.x, self.y,lag=lag_).ksg(k=self.k)
            if mi_ > maxMI: 
               maxMI = mi_
               maxMIlag = lag_
            miList.append(mi_)
        return np.asarray(miList), maxMI, maxMIlag

