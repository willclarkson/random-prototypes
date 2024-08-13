#
# noisemodel2d.py
#

#
# 2024-08-13 WIC - refactored noise vs magnitude methods here, mostly
# from the prototype fittwod.py .
#

import numpy as np
from weightedDeltas import CovarsNx2x2

def noisescale(noisepars=np.array([]), mags=np.array([]), \
               default_a = 0.):

    """Magnitude-dependent scaling for noise. Returns a 1d array of noise
scale factors with same length as the input apparent magnitudes mags[N]. 

Inputs: 

    noisepars = [log10(A), log10(B), C] 
                describing noise model A + B.exp(m C)

    mags = N-element array of apparent magnitudes

    default_a = default value of "a" to use if no parameters supplied. In most cases we want this to be zero.

Returns:

    noisescales = N-element array of noise scale factors

    """

    # Nothing to return if empty input
    if np.size(mags) < 1:
        return np.array([])

    # Initialize the model parameters
    a = default_a
    b = 0.
    c = 0.
    
    # Parse the model parameters
    if np.isscalar(noisepars):
        a = 10.0**(noisepars)

    else:
        sz = np.size(noisepars)
        if sz < 1:
            return mags*0. # + 1.

        if sz > 0:
            a = 10.0**(noisepars[0])
        if sz > 1:
            b = 10.0**(noisepars[1])
        if sz > 2:
            c = noisepars[2]

    # OK now we have the a, b, c for our model. Apply it
    return b * np.exp(mags*c) + a

def parsecorrpars(stdxs=np.array([]), parscov=np.array([]), \
                  unpack=False):

    """Takes stdxs and optional covariance shape parameters and returns a
[3,N] array of [stdx, stdy/stdx, corrxy]. 

Inputs:

    stdxs [N] = array of stddevs in x for the covariance array

    parscov = up to [2,N] array of parameters [stdy/stdx, corrxy] for
    covariance

    unpack = return as stdx, stdy, rxy rather than as [3,N] array

Returns:

    if unpack=True: 
    
    returns stdxs, stdys, corrxy

    if unpack=False:

    returns np.array([stdxs, stdys/stdxs, corrxy])

    """

    if np.size(stdxs) < 1:
        if unpack:
            return np.array([]), np.array([]), np.array([])
        else:
            return np.array([])

    # Initialize the output
    ryxs = stdxs*0. + 1.
    corrs = stdxs*0.

    # Slot in the ratio of stdev(y) / stdev(x) if given
    if np.isscalar(parscov):
        ryxs[:] = parscov
    else:
        sz = np.shape(parscov)[0] # should handle [N,N] input now
        if sz > 0:
            ryxs[:] = parscov[0]
        if sz > 1:
            corrs[:] = parscov[1]

    # Form the [3,N] array of correlation parameters
    if unpack:
        return stdxs, stdxs * ryxs, corrs
    else:
        return np.vstack(( stdxs, ryxs, corrs ))

def mags2noise(parsmag=np.array([]), \
               parscov=np.array([]), mags=np.array([]) ):

    """Returns a CovsNx2x2 object describing noise model covariance. The stdx of each 2x2 plane is computed from the model

    stdx = a + b.exp(c.mags) 

Inputs:

    parsmag = [log10(a), log10(b), c]   in the above model

    mags = vector of magnitudes used to assign stdx

    parscov = [stdy/stdx, corrxy]


Returns:

    Covars = CovarsNx2x2 object including .covars (Nx2x2 covariance array) and methods to draw samples.
    
    """

    # Unpack the model parameters and ensure all the pieces are
    # present
    stdxs = noisescale(parsmag, mags)
    stdx, stdy, corrxy = parsecorrpars(stdxs, parscov, unpack=True)

    return CovarsNx2x2(stdx=stdx, stdy=stdy, corrxy=corrxy)

def unifnoise(npts=1000, stdx=0.1, parscov=np.array([]) ):

    """Returns uniform covariances with shape given. 

Inputs:
    
    npts = number of datapoints to generate

    stdx = stddev in x-direction

    parscov = [stdy/stdx, corrxy] shape parameters

"""
    stdxs = np.repeat(stdx, npts)    
    stdxv, stdy, corrxy = parsecorrpars(stdxs, parscov, unpack=True)

    return CovarsNx2x2(stdx=stdxv, stdy=stdy, corrxy=corrxy)
    
#### SHORT test routines follow

def testnoise(parsnoise = [-4., -20., 2.], parsshape = [0.8, 0.1]):

    """Tests the unpacking of the noise model"""

    mags = np.random.uniform(16., 19.5, 10)
    
    CC = mags2noise(parsnoise, mags, parsshape)

    print(CC.covars)
