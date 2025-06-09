#######################################################################
# General statistical tools
#######################################################################
import numpy as np
#
def hist(x,nbin=10,plot=False):
    """
    Returns histogram of array x

    Args: 
       `x`: numpy array, samples of random variable or time series
       `nbin`: int, number of bins
       `iplot`: bool, whether or not plot the histogram
    """
    p, binEdge = np.histogram(x, bins=nbin, density=True)
    
    if plot:
       plt.figure(figsize=(3,2))
       plt.plot(binEdge[1:],p)
       plt.show()
        
    return p, binEdge
