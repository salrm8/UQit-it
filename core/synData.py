#######################################################
# Synthesize time series data
#######################################################
#
import sys
import numpy as np

#np.random.seed(2000)
def varSampGen(n,order=1):
    """
    Generating ssamples from linear vector autoregression with known coefficients and noise covariance matrix.
    Note that 1. the initial sample is random,
              2. at first 2*`n` samples are generated and then the first `n` samples are discarded.
              3. choose the coefficients of the VAR so that the eigen values of the coefficient matrix is within the unit circle (statically stationarity)

    Args:
       `n`: int, number of samples
       `order`: int, order of the VAR

    Returns:
       `x`: numpy array of size (n,2), samples of x1, x2
       `A`: VAR coefficient matrix
       `covMat_eps`: numpy array, noise covariance matrix
    """
    x=np.zeros((2*n,2))
    if order==1:
       A0=np.asarray([0.0,0.0])  #constant
       A1=np.asarray([[0.4,0.2],[0.5,0.6]])  #multipliers of x_{i-1}
       A = np.vstack((A0,A1))
       noisePar=[.2,.4]
       x[0,:]=np.random.rand(2)  #initial data for x and y

    elif order==2:
       A0=np.asarray([0.1,0.3])  #constant
       A1=np.asarray([[0.5,0.1],[0.4,0.6]])  #multipliers of x_{i-1}
       A2=np.asarray([[0.2,-0.1],[0.3,0.2]])  #multipliers of x_{i-2}
       A = np.vstack((A0,A1.T,A2.T))
       noisePar=[1.5,2]
       x[0,:]=np.random.rand(2)  #initial data for x and y
       x[1,:]=np.random.rand(2)  #initial data for x and y

    elif order==3:
       A0=np.asarray([0.1,0.3])  #constant
       A1=np.asarray([[0.5,0.3],[0.2,0.7]])  #multipliers of x_{i-1}
       A2=np.asarray([[0.2,-0.1],[0.3,0.2]])  #multipliers of x_{i-2}
       A3=np.asarray([[0.1,-0.2],[0.1,-0.5]])  #multipliers of x_{i-2}
       A = np.vstack((A0,A1.T,A2.T,A3.T))
       noisePar=[1,2]
       x[0,:]=np.random.rand(2)  #initial data for x and y
       x[1,:]=np.random.rand(2)  #initial data for x and y
       x[2,:]=np.random.rand(2)  #initial data for x and y
    else:
        raise ValueError("VAR: max order implemented is 3!")


    #Set the correlation between the noise of x and y: 0<=rho<=1
    rho_eps=0.7 #correlation between the noises
    covMat_eps=np.asarray([[noisePar[0]**2,rho_eps*noisePar[0]*noisePar[1]],
                           [rho_eps*noisePar[0]*noisePar[1],noisePar[1]**2]])
    eps=np.random.multivariate_normal(np.zeros(2), covMat_eps, 2*n)

    #sample generator
    for i in range(order,2*n):
        if order==1:
           x[i,:] = A0 + A1@x[i-1,:] + eps[i,:]
        elif order==2:
           x[i,:] = A0 + A1@x[i-1,:] + A2@x[i-2,:] + eps[i,:]
        elif order==3:
           x[i,:] = A0 + A1@x[i-1,:] + A2@x[i-2,:] + A3@x[i-3,:] + eps[i,:]
    return x[n:,:],A,covMat_eps


