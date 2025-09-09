##################################################################################
# Phase Slope Index (a linear measure of directionality of information transfer)
##################################################################################
import numpy as np
from scipy.signal import csd

class bi_psi:
    """
    Estimate Phase Slope Index between a pair of time series
    """
    def __init__(self,x,y):
        """
        Args:
           `x`: 1d numpy array of size n, containing samples of source time series
           `y`: 1d numpy array of size n, containing samples of target time series

        Return:
            `psi`: float, linear information direction between x and y
            psi>0: x->y
            psi<0: y->x
        """
        self.x = x
        self.y = y
        self.n = self.x.shape[0]
        assert self.y.shape[0] == self.n
    
    @classmethod
    def csdComp(self,x_,y_):
        """
        Cross spectral desnity between x and y

        Note: we need to swap x_ and y_ when using Scipy.signal.csd. 
        Since, this function computes CSDxy with the conjugate FFT of X multiplied by the FFT of Y:
        CSDxy(f) = E[X*(f)Y(f)]
        """
        freq_, csd_ = csd(y_,x_)  
        return freq_, csd_

    def comp(self):
        """
        Compute bivariate PSI
        Reference: Nolte et al. (2008). https://doi.org/10.1103/PhysRevLett.100.234101

        Note: CSDxy in the reference is assumed to be CSDxy(f) = E[X(f)Y*(f)]
        """
        f, Sxy = self.csdComp(self.x,self.y)
        _, Sxx = self.csdComp(self.x,self.x)
        _, Syy = self.csdComp(self.y,self.y)

        T1 = Sxy[1:]/np.sqrt(Sxx[1:]*Syy[1:])
        T2 = np.conj(Sxy[:-1])/np.sqrt(Sxx[:-1]*Syy[:-1])

        psi = np.sum(T1 * T2).imag
        return psi

class mv_psi:
    """
    Estimate Phase Slope Index between a combination of time series
    """
    def __init__(self,X,Y):
        """
        Args:
           `X`: numpy array of size nX x n, containing n samples of nX source time series
           `Y`: numpy array of size nY x n, containing n samples of nY target time series

        Return:
            `mpsi`: float, linear information direction between X and Y
            mpsi>0: X->Y
            mpsi<0: Y->X
        """
        self.X = X
        self.Y = Y
        self.n = self.X.shape[1]
        assert self.Y.shape[1] == self.n

    def csd_matrix(self,X,Y):
        """
        Matrix of cross spectral density between the elements of X and Y
        """
        nX = X.shape[0]
        nY = Y.shape[0]
        _, S_ = bi_psi.csdComp(X[0,:],Y[0,:])
        nc = S_.shape[0]

        S = np.zeros((nX,nY,nc),dtype=complex)
  
        for i in range(nX):
            for j in range(nY):
               _, S_ = bi_psi.csdComp(X[i,:],Y[j,:])
               S[i,j,:] = S_
        return S

    def comp(self):
        """
        Compute multivariate PSI between X and Y
        Reference: Basti et al. (2018). https://doi.org/10.1016/j.neuroimage.2018.03.004
        """
        SXX = self.csd_matrix(self.X,self.X)
        SXY = self.csd_matrix(self.X,self.Y)
        SYX = self.csd_matrix(self.Y,self.X)
        SYY = self.csd_matrix(self.Y,self.Y)

        TXX = SXX[:,:,:-1].real + SXX[:,:,1:].real
        TYY = SYY[:,:,:-1].real + SYY[:,:,1:].real

        for i in range(TXX.shape[-1]):  #frequency
            T1 = np.linalg.inv(TXX[:,:,i]) @ SXY[:,:,i+1].imag
            T2 = np.linalg.inv(TYY[:,:,i]) @ SYX[:,:,i].real
            T3 = np.linalg.inv(TXX[:,:,i]) @ SXY[:,:,i+1].real
            T4 = np.linalg.inv(TYY[:,:,i]) @ SYX[:,:,i].imag
            T = T1 @ T2 + T3 @ T4
            if i == 0:
               mpsi_ = T
            else:
               mpsi_ += T
        mpsi = 4.*np.sum(np.diag(mpsi_))
        return mpsi
 


