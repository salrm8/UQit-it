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

        #Create joint embeddings
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


class mv_te:    
    """
    Esimate Transfer Entropy for multivariate time series
    """
    def __init__(self,X,y,embDim):
       """
       Args:
          `X`: List, list of 1d arrays of source time series, X=[x1,x2,...,xp] where xi has size n
          `y`: 1d numpy array, target time series
          `embDim`: int (>0), embedded dimension (number of delayed samples0
       """
       self.X = X
       self.y = y
       self.embDim = embDim

       self.child = None

    def _create_child(self,X,y,embDim):
        self.child = mv_te(X,y,embDim)    

    def multiSrc_ksg(self,k=3,tol=1e-8):
        """
        Estimate transfer entropy from multi-source X to single target y
        Args:
           `k`: int, k-th nearest points to each sample
           `tol`: float, tolerance (small value)

        Return:
          `te`: float, transfer entropy (X->y) at embedded dimension `embDim`
        """
        self.k = k
        self.tol = tol
        lag = self.embDim

        self.n = self.y.shape[0]
        assert all(x.shape[0] == self.n for x in self.X) 

        # Create delayed time series
        yPast = np.array([self.y[i-lag:i][::-1] for i in range(lag, self.n)])    
        XPast = [np.array([x[i-lag:i][::-1] for i in range(lag, self.n)]) for x in self.X]
        yFutu = self.y[lag:self.n]

        # Create joint embeddings
        XY_past = np.hstack(XPast + [yPast])
        YZ = np.hstack((yFutu[:, None], yPast))
        XYZ = np.hstack((yFutu[:, None], XY_past))

        # Build k-d trees for radius queries
        tree_XYZ = cKDTree(XYZ)
        r_XYZ, _ = tree_XYZ.query(XYZ, k + 1, p=np.inf)
        eps_Z = r_XYZ[:, -1] - self.tol

        # Radius neighbour counts
        # NOTE: The comment-out code below uses NearestNeighbors,
        # which is slower than cKDTree for radius queries.

        # knn_XY = NearestNeighbors(n_neighbors=k+1, p=np.inf,metric='chebyshev').fit(XY_past)
        # knn_YZ = NearestNeighbors(n_neighbors=k+1, p=np.inf,metric='chebyshev').fit(YZ)
        # knn_Y = NearestNeighbors(n_neighbors=k+1, p=np.inf,metric='chebyshev').fit(yPast)
        
        # nXY = np.array([len(knn_XY.radius_neighbors([XY_past[i]], eps_Z[i], 
        #       return_distance=False)[0]) - 1 for i in range(self.n - lag)])
        # nYZ = np.array([len(knn_YZ.radius_neighbors([YZ[i]], eps_Z[i], 
        #       return_distance=False)[0]) - 1 for i in range(self.n - lag)])
        # nY = np.array([len(knn_Y.radius_neighbors([yPast[i]], eps_Z[i], 
        #       return_distance=False)[0]) - 1 for i in range(self.n - lag)])

        # Create kd-trees for neighbour queries
        # NOTE: Using cKDTree for radius queries seems to be more efficient than NearestNeighbors.
        # cKDTree and KDTree are identical as of scipy 1.6, but @salrm8 you used cKDTree so I
        # just stuck with it
        
        # Create kd-trees for radius queries
        tree_XY = cKDTree(XY_past)
        tree_YZ = cKDTree(YZ)
        tree_Y = cKDTree(yPast)
    
        # This returns a list of lists of neighbor indices
        neighbors_XY = tree_XY.query_ball_point(XY_past, eps_Z, p=np.inf)
        neighbors_YZ = tree_YZ.query_ball_point(YZ, eps_Z, p=np.inf)
        neighbors_Y = tree_Y.query_ball_point(yPast, eps_Z, p=np.inf)

        # Get the count of neighbors for each point
        nXY = np.array([len(indices) for indices in neighbors_XY]) - 1
        nYZ = np.array([len(indices) for indices in neighbors_YZ]) - 1
        nY = np.array([len(indices) for indices in neighbors_Y]) - 1


        # KSG estimator
        te = digamma(k) + np.mean(digamma(nY + 1) - digamma(nXY + 1) - digamma(nYZ + 1))
        return te 

    def ntwrk_ksg(self,k=3,tol=1e-8):
        """
        Estimate all TEs from each source to the target consideing the multi sources and 
        single target being in a network.
        
        Return:
          `teDict`: dict, Net transfer entropy from each source within the mult-souces X to single 
                          target y at embedded dimension `embDim`
        """
        self.p = len(self.X)    #number of source variates
        self.k = k
        self.tol = tol

        #TE from all sources to the target
        self._create_child(X=self.X, y=self.y, embDim=self.embDim)
        te_all = self.child.multiSrc_ksg(k=self.k,tol=self.tol)
        teDict={'TE_all->y':te_all}

        for i in range(self.p):

            #TE from all sources except the i-th one
            X_=[]
            for j in range(self.p):
                if j != i:
                   X_.append(self.X[j])   

            self._create_child(X=X_, y=self.y, embDim=self.embDim)
            te_ = self.child.multiSrc_ksg(k=self.k,tol=self.tol)

            #Net TE from i-th source to target y
            teDict.update({'TE'+str(i)+'->y':te_all - te_})

        return teDict
