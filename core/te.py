##################################################################
# Compute transfer entropy beween time series
##################################################################
#
import numpy as np
import sys
sys.path.append('./')
from entropy import entropy
#
class bi_te:    
    """
    Estimate Transfer Entropy (TE) for a pair of time series
    """

    def __init__(self,x,y,lag):
        """
        Args: 
           `x`: 1d numpy array of size n, containing samples of source time series
           `y`: 1d numpy array of size n, containing samples of target time series
           `lag`: int (>0), lag when computing TE

        Return:
            `te`: float, transfer entropy (x->y) at embedded delay `lag`
        """
        self.x = x
        self.y = y
        self.lag = lag

    def kde(self,method='mc',**kwargs):
        """
        Estimate transfer entropy using KDE method

        TE(x->y) = H(y,y_past) - H(y_past) - H(y,y_past,x_past) - H(y_past,x_past)

        Args:
           `method`: string, integration method: default 'mc' (Monte Carlo)       
        """
        self.kdeMethod = method

        xPast = self.x[:-self.lag]
        yPast = self.y[:-self.lag]
        yFutu = self.y[self.lag:]

        H_yPast = entropy(yPast).kde(method=self.kdeMethod,**kwargs)
        H_yFutu_yPast = entropy(np.vstack((yFutu,yPast)).T).kde(method=self.kdeMethod,**kwargs)
        H_yPast_xPast = entropy(np.vstack((yPast,xPast)).T).kde(method=self.kdeMethod,**kwargs)
        H_yFutu_yPast_xPast = entropy(np.vstack((yFutu,yPast,xPast)).T).kde(method=self.kdeMethod,**kwargs)

#        H_yin = H_kde(y_past,nInteg)
#        H_y_yin = H_kde(np.vstack((y_futu,y_past)).T,nInteg)
#        H_yin_xim = H_kde(np.vstack((y_past,x_past)).T,nInteg)
#        H_y_yin_xim = H_kde(np.vstack((y_futu,y_past,x_past)).T,nInteg)

#        te = H_y_yin - H_yin - H_y_yin_xim + H_yin_xim
        te = H_yFutu_yPast - H_yPast - H_yFutu_yPast_xPast + H_yPast_xPast

        return te

    def ksg(self,k=3):    
        pass
