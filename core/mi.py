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
    I(X,Y) = H(X) + H(Y) - H(X,Y)

    Args:
       `x`: 1d numpy array of size n
       `y`: 1d numpy array of size n

    Return:
       `Ixy`: mutual information between x and y
    """
    def __init__(self,x,y):
        """

        """
        self.x = x
        self.y = y

    def kde(self,method='mc'):
        """
        MI based on entropies estimated by the KDE method
        """
        Hx = entropy(self.x).kde()
        Hy = entropy(self.y).kde()
        Hxy = entropy(np.vstack((self.x,self.y)).T).kde()
        Ixy = Hx + Hy - Hxy
        return Ixy

    def kl(self,k=3):
        """
        MI based on entropies estimated by the KL (Kozachenko-Leonenko) method

        Args:
          `k`: int, k-th nearest points to x[i]
        """
        x = self.x
        y = self.y
        if x.ndim == 1:
           x = x[:,None]
        if y.ndim == 1:
           y = y[:,None]

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

