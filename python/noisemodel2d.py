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
               default_log10a = -20., mag0=0., islog10_c=False):

    """Magnitude-dependent scaling for noise. Returns a 1d array of noise
scale factors with same length as the input apparent magnitudes
mags[N]. The noise model is

    noise = A + B.((flux/flux_0)**-C) 

          = A + B.exp(+0.921(m-m_0)c)

There is some parsing of the parameters: noisepars is expected to be
one of [a,b,c], or [b,c], or [a], or a. The "c" can be supplied as log10(c). 

mag0 is the "zeropoint" for the noise model. It should not make any
difference physically what values is used, but it might make a
difference numerically. Usually good for this to be something roughly
in the middle of the expected magnitude distribution.

Inputs: 

    noisepars = [log10(A), log10(B), C] 

    mags = N-element array of apparent magnitudes

    default_log10a = default value of log10(a) to use if no parameters supplied. In most cases we want this to be a very small value.

    mag0 = zeropoint for magnitudes. 

    islog10_c = "c" is supplied as log10(c)

Returns:

    noisescales = N-element array of noise scale factors

    """

    # Nothing to return if empty input
    if np.size(mags) < 1:
        return np.array([])

    # slightly awkward: a, b are supplied as log10 while we really
    # want them in ln for logaddexp. So we convert here.
    log10toln = 1.0/np.log10(np.e)

    # Also the conversion for fluxtomag (the 0.921)
    fluxtomag = np.log(10.)/2.5 
    
    # Initialize the model parameters
    loga = default_log10a * log10toln
    logb = -np.inf
    c = 0.

    # Ensure noisepars are treated as an array. Don't modify the
    # inputs in-place.
    npars = np.atleast_1d(noisepars)

    # Parse based on length. Python 3.10 (mid-2021) has the "match"
    # syntax similar to switch/case, but not all my systems are
    # updated yet. There's something of a special case anyway: if
    # input pars have length 2 then they are [b,c]. Otherwise they are
    # a or [a] or [a,b,c].

    # This needs to work for 2D [ncols, nsamples] as well as 1D
    # [ncols] input. Since we're now also looking for special cases
    # (like 2-column input), we bite the bullet and explicitly look
    # for columns, rather than just using the array size and trusting
    # in my understanding of array broadcasting. "npars" is the
    # array-like version of noisepars anyway, so we know it will have
    # at least shape [1] (though that might mean [None]). So:
    
    ncols = np.shape(npars)[0]        
    if ncols < 1:
        return mags*0. # + 1.

    # condition-trap for None supplied.
    if npars[0] is None:
        return mags*0.
    
    # Slope only
    if ncols == 2:
        logb = npars[0] * log10toln
        c = npars[1] # rather than [-1] to support possible future
                     # extension to more complicated models
    else:
        loga = npars[0] * log10toln
        if ncols > 1:
            # b = 10.0**(npars[1])
            logb = npars[1] * log10toln
            c=npars[2]
            
    # if a c value was supplied as log10(c), convert it to c.
    if islog10_c and ncols > 1:
        c = 10.0**c

    # OK now we have the a, b, c for our model. Apply it
    logsum = np.logaddexp(c*(mags-mag0)*fluxtomag + logb, loga)

    return np.exp(logsum)

def parsecorrpars(stdxs=np.array([]), parscov=np.array([]), \
                  unpack=False, islog10_ryx=False):

    """Takes stdxs and optional covariance shape parameters and returns a
[3,N] array of [stdx, stdy/stdx, corrxy]. 

Inputs:

    stdxs [N] = array of stddevs in x for the covariance array

    parscov = up to [2,N] array of parameters [stdy/stdx, corrxy] for
    covariance

    unpack = return as stdx, stdy, rxy rather than as [3,N] array

    islog10_ryx = The stdy/stdx ratio is supplied as log10 instead of
    the value itself. If true, the input is converted to
    10.0**(log10(ryx)) before returning.

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

    ryxs_init = 1.
    if islog10_ryx:
        ryxs_init = 0.

    # Initialize the output
    ryxs = np.atleast_1d(stdxs*0. + ryxs_init)
    corrs = np.atleast_1d(stdxs*0.)

    # Slot in the ratio of stdev(y) / stdev(x) if given
    if np.isscalar(parscov):
        ryxs[:] = parscov
    else:
        sz = np.shape(parscov)[0] # should handle [N,N] input now
        if sz > 0:
            ryxs[:] = parscov[0]
        if sz > 1:
            corrs[:] = parscov[1]

    # Update if supplied as log10
    if islog10_ryx:
        ryxs[:] = 10.0**ryxs

    # Convert back to scalar?
    if np.isscalar(stdxs):
        ryxs = ryxs[0]
        corrs = corrs[0]
        
    # Form the [3,N] array of correlation parameters
    if unpack:
        return stdxs, stdxs * ryxs, corrs  # sic
    else:
        return np.vstack(( stdxs, ryxs, corrs ))

def mags2noise(parsmag=np.array([]), \
               parscov=np.array([]), mags=np.array([]), \
               islog10_ryx=False, mag0=0., \
               returnarrays=False, islog10_c=False):

    """Returns a CovsNx2x2 object describing noise model covariance. The stdx of each 2x2 plane is computed from the model

    stdx = a + b.exp(c.(mags-mag0)) 

Inputs:

    parsmag = [log10(a), log10(b), c]   in the above model

    mags = vector of magnitudes used to assign stdx

    parscov = [stdy/stdx, corrxy]

    islog10_ryx = if provided, stdy/stdx is supplied as
    log10(stdy/stdx)

    mag0 = zeropoint for magnitudes

    islog10_c = the "c" parameter in the noise model is supplied as
    log10

    returnarrays = True if we want the covariance components and not a
    CovarsNx2x2 object

Returns:

    Covars = CovarsNx2x2 object including .covars (Nx2x2 covariance array) and methods to draw samples.

    or, if returnarray = True:

    stdx, stdy, corrxy = covariance components

    """

    # Unpack the model parameters and ensure all the pieces are
    # present
    stdxs = noisescale(parsmag, mags, mag0=mag0, islog10_c=islog10_c)
    stdx, stdy, corrxy = parsecorrpars(stdxs, parscov, unpack=True, \
                                       islog10_ryx=islog10_ryx)
    if not returnarrays:
        return CovarsNx2x2(stdx=stdx, stdy=stdy, corrxy=corrxy)

    return stdx, stdy, corrxy
    
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
    
    CC = mags2noise(parsnoise, parsshape, mags)

    print(CC.covars)
