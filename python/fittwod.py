#
# fittwod.py
#

#
# WIC 2024-07-24 - test-bed to use and fit transformation objects in
# unctytwod.py
# 

import os, time, pickle, copy

# 2024-08-07 commented THIS import out since we don't call
# multiprocessing from within this module any more.
#
# from multiprocessing import cpu_count, Pool 

from scipy import stats

import numpy as np
import matplotlib.pylab as plt
import numpy.ma as ma

import unctytwod
from covstack import CovStack

# we want to draw samples
from weightedDeltas import CovarsNx2x2

# The minimizer
from scipy.optimize import minimize

# For initial guess by linear least squares
from fitpoly2d import Leastsq2d, Patternmatrix

# For sampling and plotting
import emcee
import corner

# For logistic regression on responsibilities
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

def uTVu(u, V):

    """Returns u^T.V.u where
    
    u = [N,m] - N datapoints of dimension m (typically deltas array)

    V = [N,m,m] - N covariances of dimension m x m (an
    inverse-covariance stack)

    This will return an N-element array.

    """

    Vu = np.einsum('ijk,ik -> ij', V, u)
    return np.einsum('ij,ji -> j', u.T, Vu)

def noisescale(noisepars=np.array([]), mags=np.array([]) ):

    """Magnitude-dependent scaling for noise. Returns a 1d array of noise
scale factors with same length as the input apparent magnitudes mags[N]. 

    Inputs: 

    noisepars = [log10(A), log10(B), C] 
                describing noise model A + B.exp(m C)

    mags = N-element array of apparent magnitudes

    Returns:

    noisescales = N-element array of noise scale factors

    """

    # Nothing to return if empty input
    if np.size(mags) < 1:
        return np.array([])

    # Initialize the model parameters
    a = 1.
    b = 0.
    c = 0.
    
    # Parse the model parameters
    if np.isscalar(noisepars):
        a = 10.0**(noisepars)

    else:
        sz = np.size(noisepars)
        if sz < 1:
            return mags*0. + 1.

        if sz > 0:
            a = 10.0**(noisepars[0])
        if sz > 1:
            b = 10.0**(noisepars[1])
        if sz > 2:
            c = noisepars[2]

    # OK now we have the a, b, c for our model. Apply it
    return b * np.exp(mags*c) + a

def parsemixpars(mixpars=np.array([]), \
                 islog10frac=False, \
                 islog10vxx=False, \
                 vxxbg=1.):

    """Parses foreground/background mixture parameters. If no mixpars are
supplied, this defaults to a single-component (foreground-only) model.

Inputs:

    mixpars = [ffg, vbg] = up to 2-element array of mixture model
    parameters: fraction of foreground, variance of background

    islog10frac = foreground fraction ffg supplied as log10

    islog10vxx = background variance vxx supplied as log10

    vxxbg = default variance for background component

Returns:

    ffg = fraction of model component that is foreground. Returns 1.0
    if no mixture parameters were supplied.

    vxx = variance of model. 

    """

    # Defaults
    ffg = 1.
    vxx = 1.
    
    #covbg = np.eye(2) * vxxbg

    if np.size(mixpars) < 1:
        return ffg, vxx

    # Mixture fraction...
    ffg = parsefraction(mixpars, islog10frac)

    # variance...
    if np.size(mixpars) > 1:
        vxx = parsefraction(mixpars[1], islog10vxx, maxval=np.inf, inclusive=False)

    return ffg, vxx

def unpackmixpars(mixpars, islog10frac=False, islog10vxx=False, vxxbg=1.):

    """Given mixture model [ffg, vxx], parses and unpacks the parameters
into ffg, covxy.

Inputs:

    mixpars = [ffg, vxx] - array of fraction and variance

    islog10frac = ffg supplied as log10

    islog10vxx = vxx supplied as log10

    vxxbg = default vxx parameter (if supplied value is bad)

Returns:

    ffg = fraction of foreground. Between 0 and 1, or -np.inf if bad

    covxy = [2,2] covariance array

"""

    # parse the fraction and vxx...
    ffg, vxx = parsemixpars(mixpars, islog10frac, islog10vxx)

    # ... and convert vxx into a covariance matrix
    covxy = var2cov22(vxx)
    
    return ffg, covbg
    
        
def get0thscalar(pars=np.array([]) ):

    """Utility - returns scalar zeroth value of array.

Inputs:

    pars = scalar or 1D array. If an array, the first entry is assessed.

Returns:

    firstval = first value in the array, or (if supplied as scalar)
    *a copy of* the scalar value.

    if nothing was supplied, returns None

    """

    if pars is None:
        return None
    
    if np.isscalar(pars):
        return pars

    if np.size(pars) < 1:
        return None
    
    return pars[0]
    
def parsefraction(frac=None, islog10=False, badval=-np.inf, \
                  minval=0., maxval=1., \
                  minlog10=-50., \
                  inclusive=True):

    """Given a scalar value frac, return the value (if within bounds) or
badval if not. If frac was supplied as log10(frac), then it is
converted to frac BEFORE comparing to the bounds minval/maxval.

Inputs:

    frac = parameter of interest. If supplied as an array, the 0th
    entry is used.

    islog10 = log10(frac) supplied instead of frac

    badval = what to return if the supplied parameter is outside the range

    minval = minimum value for the parameter
    
    maxval = maximum value for the parameter

    minlog10 = minimum value of log10 to supply. Used for bounds
    checking, is ignored if islog10=False.

    inclusive = value can be equal to minval or maxval.

Returns:

    value = parameter value (if within allowed range) or badval if
    outside. Scalar.

    """

    # Ensure we're dealing with a scalar
    fuse = get0thscalar(frac)
    if fuse is None:
        return badval

    # If log10(frac) supplied, convert to frac itself.
    if islog10:
        if fuse < minlog10:
            fuse = 0.
        else:
            fuse = 10.0**fuse

    # return badval if outside the bounds
    if inclusive:
        isbad = fuse < minval or fuse > maxval
    else:
        isbad = fuse <= minval or fuse >= maxval
    if isbad:
        return badval

    # Return the adopted value if it passed all the tests above.
    return fuse
    
def var2cov22(vxx=1., vyy=None, vxy=None, islog10=False):

    """Given one or more components of a covariance matrix, return as
[2,2] array. No checking is done on whether the return matrix is
singular. If vxx or vyy are supplied as negative values, the absolute
value is taken rather than failing (though in that case you may have
supplied log10(vxx)?)

Inputs:

    vxx = variance in x

    vyy = variance in y

    vxy = xy covariance

    islog10 = vxx (and any vyy) supplied as log10(value)

Returns:

    covxy = [2,2] covariance matrix

    """

    if islog10:
        covxy = np.eye(2) * 10.0**vxx
    else:
        covxy = np.eye(2) * np.abs(vxx)
        
    # only vxx supplied?
    if vyy is None:
        return covxy

    if islog10:
        covxy[1,1] = 10.0**vyy
    else:
        covxy[1,1] = np.abs(vyy)


    if vxy is None:
        return covxy

    covxy[0,1] = vxy
    covxy[1,0] = vxy

    return covxy
    
def parsecorrpars(stdxs=np.array([]), parscov=np.array([]) ):

    """Takes stdxs and optional covariance shape parameters and returns a
[3,N] array of [stdx, stdy/stdx, corrxy]. Inputs:

    stdxs [N] = array of stddevs in x for the covariance array

    parscov = up to [2,N] array of parameters [stdy/stdx, corrxy] for
    covariance

    """

    if np.size(stdxs) < 1:
        return np.array([])

    # Initialize the output
    rxys = stdxs*0. + 1.
    corrs = stdxs*0.

    # Slot in the ratio of stdev(y) / stdev(x) if given
    if np.isscalar(parscov):
        rxys[:] = parscov
    else:
        sz = np.shape(parscov)[0] # should handle [N,N] input now
        if sz > 0:
            rxys[:] = parscov[0]
        if sz > 1:
            corrs[:] = parscov[1]

    # Form the [3,N] array of correlation parameters
    return np.vstack(( stdxs, rxys, corrs ))

def corr3n2covn22(corr3xn=np.array([]), Verbose=False):

    """Converts [3,N] array [stdx, stdy/stdx, corrxy] into [N,2,2]
covariance matrix stack. Inputs:

    corr3xn = [3,N] array [stdx, stdy/stdx, corrxy]

    """

    covs3xn = corr2cov1d(corr3xn)      # output has shape 3,N

    # OK this *is* our covariance array, it just needs reshaping into
    # the order we expect. Do so like this:
    covsnx2x2 = np.zeros(( covs3xn.shape[-1], 2, 2 ))
    covsnx2x2[:,0,0] = covs3xn[0]
    covsnx2x2[:,1,1] = covs3xn[1]
    covsnx2x2[:,0,1] = covs3xn[2]
    covsnx2x2[:,1,0] = covs3xn[2]

    # Optionally print debug information
    if Verbose:
        print("mags2cov INFO:", corrpars.shape)
        print("mags2cov INFO:", covs3xn.shape)
        print(covs3xn.T[0:3])
        print(covsnx2x2[0:3])

    return covsnx2x2

def stdxs2covn22(stdxs=np.array([]), parscov=np.array([]) ):

    """Reshapes stdxs and cov shape arrays into Nx2x2 covariance
array. Returns: [N,2,2] covariance array. Inputs:

    stdxs = [N] - element array of stdxs

    parscov = up to [N,2] array of stdy/stdx and corrxy"""

    corr3xn = parsecorrpars(stdxs, parscov)
    return corr3n2covn22(corr3xn)
    
def corr2cov1d(s=np.array([]) ):

    """Utility - given covariance entries as [stdx, stdy/stdx, corrxy],
returns them as [varx, vary, covxy]

    """

    # As usual, much of the syntax is parsing input...
    if np.isscalar(s):
        varx=s**2
        return np.array([varx, varx, 0.])

    # Nothing to do if blank input
    if np.size(s) < 1:
        return np.array([])

    varx = s[0]**2

    # Use shape[0] rather than size so that we correctly handle
    # [N]-element arrays for each input
    sz = np.shape(s)[0]
    
    if sz < 2:
        return np.array([varx, varx, 0.])

    vary = (s[0]*s[1])**2
    covxy = 0.

    if sz > 2:
        covxy = s[0]**2 * s[1] * s[2]  # sic

    return np.array([varx, vary, covxy])

def cov2corr1d(v=np.array([]) ):

    """Utility - given covariance entries as [varx, vary, covxy], return
them as [stdx, stdy/stdx, corrcoef]"""

    if np.isscalar(v):
        stdx = np.sqrt(v)
        return np.array([stdx, 1., 0.])

    if np.size(v) < 1:
        return np.array([])

    stdx = np.sqrt(v[0])
 
    # Use shape[0] rather than size in order to handle [N, N, N] input
    vz = np.shape(v)[0]
    
    if vz < 2:
        return np.array([stdx, 1., 0.])

    # stddev in y, output ratio
    stdy = np.sqrt(v[1])
    ryx = stdy/stdx
    corrcoef = 0.

    if vz > 2:
        corrcoef = v[2]/(stdx * stdy)

    return np.array([stdx, ryx, corrcoef])

def mags2cov(parsmag=np.array([]), mags=np.array([]), \
             parscov=np.array([]), Verbose=False):
    
    """Returns an [N,2,2] covariance matrix set. The stdx of each 2x2
plane is computed from the model 

    stdx = a + b.exp(c.mags) 

    Inputs:

    parsmag = [log10(a), log10(b), c]   in the above model

    mags = vector of magnitudes used to assign stdx

    parscov = [stdy/stdx, corrxy]

    Verbose = print debug messages
    
    """

    # N-element arrays giving stdx, stdy/stdx, corrxy for each plane
    # in the N,2,2 covariance matrix stack
    stdxs = noisescale(parsmag, mags)

    return stdxs2covn22(stdxs, parscov)

def covn222cov3(covn22=np.array([]) ):

    """Utility - given [N,2,2] covariance matrix, returns [vx, vy, vxy] as
[N,3] array.

Inputs:

    covn22 = [N,2,2] covariance matrix stack

Returns:

    covn3 = [N,3] array [vxx, vyy, vxy]

    """

    # Must be a 3d array. For the moment we enforce this rather than
    # reshaping the input to be 3d.
    if np.ndim(covn22) != 3:
        return np.array([])

    nrows = np.shape(covn22)[0]
    covn3 = np.zeros((nrows, 3))

    covn3[:,0] = covn22[:,0,0]
    covn3[:,1] = covn22[:,1,1]
    covn3[:,2] = covn22[:,0,1]

    return covn3

def covn32covn22(covn3=np.array([]) ):

    """Utility - given [N, 3] array of vxx, vyy, vxy values, returns an
N,2,2 covariance mtarix stack.

Inputs:

    covn3 = [N,3] array of [vxx, vyy, vxy values]

Returns:

    covn22 = [N,2,2] array of covariances

    """

    # Possibly redundant with cov32covn22 provided the input is first
    # transposed then the output transposed again, thus taking
    # advantage of numpy broadcasting rules. BUT this present method
    # does do the one thing it needs to do, so we retain it.

    # np.atleast_2d adds the newaxis to the end, I want it at the
    # beginning. So we do this the other way. Not sure we don't want
    # this just to outright fail if our supposed [N,3] array was
    # passed as a scalar... but the below should still work.
    if np.ndim(covn3) < 2:
        if np.isscalar(covn3):
            covn3 = np.array([covn3])
        covn3 = covn3[:, None]        

    # Now finally the operations on our [N, <=3] array can happen.
    nrows, ncols = np.shape(covn3)
    covn22 = np.zeros((nrows, 2, 2))
    covn22[:,0,0] = covn3[:,0]
    if ncols < 2:
        covn22[:,1,1] = covn22[:,0,0]
    else:
        covn22[:,1,1] = covn3[:,1]
        if ncols > 2:
            covn22[:,0,1] = covn3[:,2]
            covn22[:,1,0] = covn22[:,0,1]
            
    return covn22

def cov32covn22(addvars=np.array([]), nrows=1, covn22=np.array([]) ):

    """Utility - populates [N,2,2] covariance matrix from entries of [vx,
vy, vxy] array. Can pass in the N,2,2 covariance matrix to modify
in-place. Inputs:

    addvars [1-3] = [vx, vy, vxy] array, each entry assumed scalar.

    nrows = length of desired covariance matrix. Ignored if covn22 is
    supplied for modification in-place.

    covn22 = [N,2,2] covariance matrix. If supplied, is modified
    in-place, otherwise is created here.

    """

    # if covn22 not already supplied as a 3D array:
    if np.ndim(covn22) != 3:
        covn22 = np.zeros(( nrows, 2, 2 ))

    # now we populate the array depending on what was supplied:
    covn22[:,0,0] = addvars[0]
    covn22[:,1,1] = addvars[0]
    
    sz = np.size(addvars)
    if sz > 1:
        covn22[:,1,1] = addvars[1]
        if sz > 2:
            offdiag = addvars[2]
            covn22[:,0,1] = offdiag
            covn22[:,1,0] = offdiag

    return covn22

def corr32cov3(addcorr=np.array([]), islog10=False):

    """Given all or a subset of [stdx, stdy/stdx, corrxy],
returns array [vx, vy, vxy] for use by cov32covn22. Inputs:

    addcorr = [stdx, stdy/stdx, corrxy], or a subset

    islog10 = [T/F]: If the first entry is log10(stdx) instead of stdx

    Returns:

    addcov = array [vx, vy, vxy]

    cov_ok = [T/F] parameters OK: stdx > 0, stdy/stdx >0, |cxy <= 1|

    """

    # Ensure the correlation parameters are not unphysical
    cov_ok = checkcorrpars(addcorr, islog10)
    if islog10:
        addcorr[0] = 10.0**addcorr[0]

    addcov = corr2cov1d(addcorr)

    return addcov, cov_ok

def cov3d(xy=np.array([]) ):

    """Computes plane-by-plane covariance of [nsamples, 2, ndata] array.

Inputs:

    xy = [nsamples, 2, ndata] array

Returns

    covn22 = [ndata, 2, 2] covariance array


"""

    # 2024-08-09: This really should be an input method for
    # CovarsNx2x2. Then, so probably should many of the methods in
    # this part of the module.

    # Do nothing if input not 3d
    if np.ndim(xy) != 3:
        return

    nsamples, ndim, ndata = xy.shape

    # Get the vxx, vyy, vxy terms
    meanxy = np.mean(xy, axis=0)
    var = np.sum((xy - meanxy[None, :, :])**2, axis=0)/(ndata + 1.)
    
    vxy = np.sum( (xy[:,0,:] - meanxy[None,0,:]) * \
                  (xy[:,1,:] - meanxy[None,1,:]), axis=0 ) /(ndata+1.)

    # assemble the output into an nx2x2 covariance array.
    covn22 = np.zeros(( ndata, ndim, ndim ))
    covn22[:,0,0] = var[0]
    covn22[:,1,1] = var[1]
    covn22[:,0,1] = vxy
    covn22[:,1,0] = vxy

    return covn22
        
def checkcorrpars(addcorr=np.array([]), islog10=False):

    """Utility - given [stdx, stdy/stdx, corrxy], determines if the
supplied parameters violate positivity and other constraints. Inputs:

    addcorr = [3]-element array of parameters

    islog10 [T/F] - first entry of addpars is supplied as
    log10(value), and thus can be negative.

    Returns True if corr pars are OK, otherwise False.

    """

    # stdx must be positive
    if not islog10:
        if addcorr[0] < 0:
            return False

    # stdy/stdx must be positive nonzero
    if addcorr.size > 1:
        if addcorr[1] <= 0.:
            return False

    # if the correlation coefficient was supplied outside the
    # range [-1, +1], flag for the calling routine
    if addcorr.size > 2:
        if np.abs(addcorr[-1]) > 1.:
            return False

    # if we got here then the addcorr array passed the tests.
    return True

def checknoisepars(noisepars=np.array([]), covpars=np.array([]), \
                   increasing=True, maxlogb=10., maxexpon=10.):

    """Checks noise model parameters for validity.

Inputs:

    noisepars = noise model parameters [loga, logb, c]

    covpars = covariance model parameters [stdy/stdx, corrxy]

    increasing = expon must be >= 0

    maxlogb = maximum allowed value of logb

    maxexpon = maximum value of c

Returns:
    
    cov_ok = [T/F]: noise and covariance parameters passed the tests


"""

    if np.size(noisepars) < 1:
        return False

    # The noise model should not have stupidly high constant...
    noise_ok = True
    if np.size(noisepars) > 1:
        if noisepars[1] >= maxlogb:
            noisepars[1] = maxlogb
            noise_ok = False
            
    # If the noise model must increase with magnitude and the exponent
    # doesn't do this, then this noise model parameter set is 'bad'
    if np.size(noisepars) > 2 and increasing:
        if noisepars[2] < -0:
            noise_ok = False

        # Safety check: np.exp(maxexpon) is too large.
        if noisepars[2] >= maxexpon:
            noisepars[2] = maxexpon # WATCHOUT - changing in place
            noise_ok = False

    # If any of the above noise checks failed, return False
    if not noise_ok:
        return False
        
    # the covariance parameters are optional. If supplied, they are in
    # the order [stdy/stdx, corrxy], and we do have requirements on
    # those:
    sz = np.size(covpars)
    if sz > 0:
        if covpars[0] < 0:
            return False
        if sz > 1:
            if np.abs(covpars[1]) > 1.:
                return False

    # If here then the noise model and covariance model parameters
    # both passed the test
    return True

def splitmodel(pars=np.array([]), nnoise=0, nvars=0, nmix=0):

    """Split a 1D parameter array into transformation parameters, noise
model parameters, and covariance parameters. (It's probably better to
use splitpars(), which this method calls and which is more flexible.)

Inputs:

    pars = [M + nnoise + nvars + nmix] 1D array with the parameters

    nnoise = number of parameters describing the variation of noise stdx with apparent magnitude

    nvars = number of parameters describing the shape of the noise
    (stdy/stdx, corrxy)

    nmix = number of parameters describing the mixture model

Returns:

    transf = [M] array of transformation parameters

    noise = [nnoise] array of noise model parameters

    covar = [nvars] array of variance shape parameters

    mixt = [nmix] array of mixture model parameters

    """

    transf, lsplit = splitpars(pars, [nnoise, nvars, nmix])
    return transf, lsplit[0], lsplit[1], lsplit[2]

def splitpars(pars, nsplit=[] ):

    """Splits a 1d array into sub-arrays, skimming off the nsplit entries
from the end at each stage. Like splitpars() but the nsplits are
generalized into a loop. 

Inputs:

    pars = 1d array of parameters

    nsplit = [n0, n1, ... ] list of last-n indices to split off at each stage.

Returns: 

    allbutsplit = all the pars not split off into a subarray

    [p1, p2, ...] = list of pars split off from the far end, *in the same order as the nsplit list*. 

Example:

    x = np.arange(10)
    fittwod.splitpars(x,[3,2])
    
    returns:
             (array([0, 1, 2, 3, 4]), [array([7, 8, 9]), array([5, 6])])

    """

    # handle scalar input for nsplit
    if np.isscalar(nsplit):
        nsplit = [nsplit]
    
    # if no splits, nothing to do
    if len(nsplit) < 1:
        return pars

    lsplit = []
    allbut = np.copy(pars)

    for isplit in range(len(nsplit)):
        allbut, split = splitlastn(allbut, nsplit[isplit])
        lsplit = lsplit + [split]
        
    return allbut, lsplit
        
def splitlastn(pars=np.array([]), nsplit=0):

    """Splits a 1D array into its [0-nsplit] and [-nsplit::] pieces.

    Inputs:

    pars = [M]  array of parameters

    nsplit = number of places from the end of the array that will be
    split off

    Returns:

    first = [M-nsplit] array before the split

    last = [nsplit] array after the split

    """

    # Nothing to do if nothing provided
    if np.size(pars) < 1:
        return np.array([]), np.array([])

    # Cannot do anything if the lengths do not match
    if np.size(pars) < nsplit:
        return pars, np.array([])
    
    # Nothing to do if no split
    if nsplit < 1:
        return pars, np.array([])

    return pars[0:-nsplit], pars[-nsplit::]

def extracovar(noisepars=np.array([]), mags=np.array([]), \
               corrpars=np.array([]), fromcorr=True, islog10=False, \
               nrows=0):

    """Given noise parameters, returns extra covariance to include as part
of the model.

Inputs:

    noisepars = [<=3]-element array of parameters for the noise
    model. Assumed [loga, logb, c] where extra noise stdx = a +
    b.exp(m.c) .

    mags = [N]-element vector of apparent magnitudes to use
    constructing the additive noise.

    corrpars = [<=3] array of parameters describing covariance shape.

    fromcorr [T/F] = corrpars are [stdx, sydy/stdx, corrxy]. If false,
    assumed to be [varx, vary, varxy].

    islog10 [T/F] = corrpars[0] = log10(stdx)

    nrows = number of rows in the dataset. Needed if the noise model
    is not being used. If mags is supplied, its length is used in
    preference to nrows.

Returns

    extracov = [N,2,2] array of covariances. If no parameters were
    supplied, returns 0. (so that this can be added to any existing
    variances without problems).

    cov_ok = result of internal checking on the noise
    parameters. (Should be redundant if priors were used to enforce
    positivity, etc.)

    """

    # initialize to blank
    extracov = 0.
    cov_ok = True

    # cowardly refuse to proceed further if no parameters given
    if noisepars.size < 1 and corrpars.size < 1:
        return extracov, cov_ok
    
    # generating this from noise model and shape parameters?
    if noisepars.size > 0 and mags.size > 0:
        cov_ok = checknoisepars(noisepars, corrpars)
        if cov_ok:
            extracov = mags2cov(noisepars, mags, corrpars)

        return extracov, cov_ok

    # if here then we're using the older model in which a single
    # covariance is used fo the entire dataset.
    if fromcorr:
        corrpars, cov_ok = corr32cov3(corrpars, islog10)

    # If magnitude data were supplied, use them instead of nrows.
    if mags.size > 0:
        nrows = mags.size
        
    extracov = cov32covn22(corrpars, nrows)
    return extracov, cov_ok
            
            
def skimvar(pars, nrows, npars=1, fromcorr=False, islog10=False, \
            nnoise=0, mags=np.array([]), \
            nmix=0):

    """Utility - if an additive variance is included with the parameters,
split it off from the parameters, returning the parameters and an
[N,2,2] covariance matrix from the supplied extra variance. 

Inputs:

    pars = [M+mvars] element array with parameters of transformation
    model and extra variance model

    nrows = number of rows in the dataset

    npars = number of parameters that are covariances.

    fromcorr [T/F]: additive variance is supplied as [stdx, stdy/stdx,
    corrcoef]. If False, is assumed to be [varx, vary, covxy].

    islog10 = stdx is supplied as np.log10(stdx). Applies only if
    fromcorr is True.

    nnoise = number of entries in pars that refer to the noise
    model. These are always the ones at the end.

    mags = [N-element] array with magnitude values. Used to produce the covariance from the noise model

    nmix = number of parameters describing the mixture model

Returns: 

    pars[M] = transformation parameters only

    covextra [N,2,2] = additional covariance computed from the
    variance model parameters.

    cov_ok [T/F] = result of parsing the variance model parameters. If
    False, then the variance model parameters were unphysical
    (e.g. correlation coefficient greater than unity).

NOTE: this was originally developed to both skim off the parameters
and generate the additional covariance. Its functionality has now been
split across splitmodel() and lnprob(), so skimvar() should no longer
be used.

    """

    # split the input parameters into the pieces we want:
    parsmodel, addnoise, addvars, addmix = \
        splitmodel(pars, nnoise, npars, nnmix)
    
    # Status flag. If we're doing additional translation of the input
    # covariance forms, we might violate the prior. Set a flag to
    # report back up if this is happening.
    cov_ok = True

    # Now we generate the covariances. The noise model is always used
    # in preference to the flat model, if parameters were supplied.
    if addnoise.size > 0 and mags.size > 0:
        cov_ok = checknoisepars(addnoise, addvars)
        if cov_ok:
            extracov = mags2cov(addnoise, mags, addvars)
        else:
            # This is OK as long as lnprob returns -np.inf when not
            # cov_ok.
            extracov = np.array([])
    else:
        # If the added variance was supplied as [(log10)stdx, stdy/stdx,
        # corrxy], parse and convert to [vx, vy, vxy]:
        if fromcorr:
            addvars, cov_ok = corr32cov3(addvars, islog10)
        
        # Populate the n22 covariance from this
        extracov = cov32covn22(addvars, nrows)

    return parsmodel, extracov, cov_ok
    
def lnprior_unif(pars):

    """ln uniform prior"""

    return 0.

def lnprior_noisemodel_rect(parsnoise, \
                            logalo=-50., logahi=2., logblo=-50.,
                            logbhi=10., clo=0., chi=10.):

    """Expresses positivity constraints etc. on the noise model as a
(log) prior. Model is assumed to have the form a + b.exp(m.c). Prior
    is uniform within the limits.

Inputs:

    parsnoise = [log10(a), log10(b), c] noise model. 0-3 components
    can be supplied, but always parsed left-right.

    logalo, logahi = min, max allowed values for log(a)

    logblo, logbhi = min, max allowed values for log(b)
    
    clo, chi = min, max allowed values for c.

    """

    # No judgement if no parameters passed
    sza = np.size(parsnoise)
    if sza < 1:
        return 0.

    # If only scalar (assumed log10(a)) was supplied, judge and
    # return.
    if np.isscalar(parsnoise):
        if not (logalo < parsnoise < logahi):
            return -np.inf
        return 0.

    # We only get here if parsnoise has at least one element. So:
    if not (logalo < parsnoise[0] < logahi):
        return -np.inf

    # now look for parameters outside the specified ranges
    if sza > 1:
        if not(logblo < parsnoise[1] < logbhi):
            return -np.inf

    if sza > 2:
        if not(clo < parsnoise[2] < chi):
            return -np.inf

    # If we got here then the params are all within the range we set.
    return 0.

def lnprior_corrmodel2_rect(parscorr, \
                            rlo=0., rhi=10., \
                            corrlo=0., corrhi=1.):

    """Expresses sanity constraints on the correlation shape model
    [stdy/stdx, corrxy] as a prior. The prior is assumed uniform
    within these constraints.

Inputs:

    parscorr = [stdy/stdx, corrxy] parameters of our correlation shape
    model. 0-2 parameters are parsed, from the left.
    
    rlo, rhi = minmax values of stdy/stdx. Evaluated exclusively: rlo < r < rhi

    corrlo, corrhi = minmax values of correlation coefficient rho(x,y). Evaluated inclusively: corrlo <= corr <= corrhi .

    """

    szc = np.size(parscorr) # works on python lists and scalars too
    if szc < 1:
        return 0.

    # scalar input parameter assumed to be stdy/stdx.
    if np.isscalar(parscorr):
        if not (rlo < parscorr[0] < rhi):
            return -np.inf
        return 0.

    # Now we go through the rest of the entries in the correlation
    # shape parameters. In this case there only is one more...
    if szc > 1:
        if not (corrlo <= parscorr[1] <= corrhi):
            return -np.inf

    return 0.

def lnprior_corrmodel3_rect(parscorr, \
                            logslo=-50., logshi=10., \
                            rlo=0., rhi=10., \
                            corrlo=0., corrhi=1., \
                            islog10=False):

    """Expresses sanity constraints on [stdx, stdy/stdx, corrxy] as a
    prior.

Inputs:

    parscorr = [stdx, stdy/stdx, corrxy] parameters of our correlation shape
    model. 0-3 parameters are parsed, from the left.

    logslo, logshi = minmax values on log10(stdx).

    rlo, rhi = minmax values of stdy/stdx. Evaluated exclusively: rlo < r < rhi

    corrlo, corrhi = minmax values of correlation coefficient
    rho(x,y). Evaluated inclusively: corrlo <= corr <= corrhi .

    islog10 = stdx argument is supplied as log10(stdx)

    """

    # Most of the syntax here is handling the fact that other routines
    # have options on the meaning of that first entry, whether it is
    # stdx or log10(stdx).

    # work out a couple of characteristics up front so that we can
    # refer to them rather than recalculate:
    sz = np.size(parscorr)
    scalar = np.isscalar(parscorr)
    
    if sz < 1:
        return 0.

    # Because we're going to do a couple of checks on the stdx or its
    # we isolate it first.
    if scalar:
        ssx = parscorr
    else:
        ssx = parscorr[0]

    # judgement on stdx or log10(stdx). Copy rather than modify
    # in-place
    if not islog10:
        if ssx <= 0.:
            return -np.inf
        logsx = np.log10(ssx)
    else:
        logsx = ssx

    # now we can finally make our judgement on log10(stdx):
    if not (logslo <= logsx <= logshi):
        return -np.inf

    # If only the one entry was provided, and it passed the tests,
    # then there's nothing more to do.
    if sz < 2 or scalar:
        return 0.

    # now apply judgement to the last two entries
    lnprior_remainder = lnprior_corrmodel2_rect(parscorr[1::], \
                                                rlo, rhi, corrlo, corrhi)

    return lnprior_remainder

def lnprior_binomial_mixfrac(pbad=0., ndata=1., islog10=False, \
                             minlog10fbad=-90.):

    """Computes the binomial prior on the mixture fraction pbad, using the
following:

    ln(prior) = sum_i q_i ln(1-pbad) + sum_i (1-q_i) ln(pbad)

which, since q_i = 1 for "good" and q_i=0 for "bad" points, simplifies to

    ln(prior) = ndata x ( (1-pbad) ln(1-pbad) + pbad ln (pbad) )

where ndata is the number of datapoints (so nbad = ndata x pbad).



Inputs:

    pbad = mixture amplitude for the "bad" points. Assumed (0 < pbad < 1.)

    ndata = size of the data sample (defaults to 1. if not set)

    islog10 = log10(pbad) supplied instead of pbad

    minlog10fbad = minimum value of log10(fbad) to count as
    nonzero. Heuristic to avoid bounds problems later on.

Outputs:

    ln_prob_fmix = evaluation of the prior

    """

    # COMMENT 2024-08-07 - this should really return two entries, one
    # for foreground, one for background.
    
    # Just a little parsing (we use fbad not pbad to avoid changing
    # things upstream)
    fbad = np.copy(pbad)
    if not np.isscalar(pbad):
        fbad = pbad[0]

    # Heuristic to get round bounds problems with very small supplied
    # fraction
    if islog10:
        if fbad > 0.:
            return -np.inf
        
        if fbad < minlog10fbad:
            return 0.

        fbad = 10.0**fbad

    # if supplied is outside the range 0-1, return badval
    if fbad < 0. or fbad > 1.:
        return -np.inf
        
    # if supplied is very close to 0 or very close to 1, return the
    # value that ln(prior) would return at those values were it not
    # for bounds issues.
    if np.abs(fbad) < 1.0e-50:
        return 0.

    if np.abs(fbad-1.) < 1.0e-50:
        return 0.

    # If we got here, then we finally have fbad that won't produce
    # bounds violations! Compute and return the actual prior we want...
    
    lnprior = (1.0-fbad)*np.log(1.0-fbad) + fbad * np.log(fbad)
    return lnprior * ndata
        
def lnprior_mixmod_binomial(parsmix=np.array([]), islog10frac=False, ndata=1):

    """ln-prior on mixture model parameters, including a binomial prior on
the mixture fraction (see docstring for lnprior_binomial_mixfrac for
more information on the binomial prior).

Inputs:

    parsmix = [fbad] - model parameters for mixture model. 

    islogfrac = fraction supplied as log10(fraction)

    ndata = number of datapoints. If not supplied, defaults to 1.

Returns:

    lnprior_mix = ln(prior) of mixture model.

    """

    # If no parameters, say nothing about the prior.
    if np.size(parsmix) < 1:
        return 0.

    # Allow scalar input
    if np.isscalar(parsmix):
        fbad = parsmix
    else:
        fbad = parsmix[0]

    # Evaluate the binomial prior on the mixture model fraction
    lnprior_mixfrac = lnprior_binomial_mixfrac(fbad, ndata, islog10frac)

    # currently the mixture model fraction is the only mixture model
    # parameter we have... Others would be added here.
    lnprior = lnprior_mixfrac
    
    return lnprior

def lnprior_mixmod_rect(parsmix=np.array([]), islog10frac=False, islog10vxx=False, \
                        minffg=1.0e-4, maxvxx=2.):

    """Enforces validity constraints on mixture model parameters but nothing else. 

Some sources refer to the mixture fractions as prior components. This
tends to mean evaluating logsumexp of the mixture inside the
"likelihood" function. That's I think the correct way to do things but
it can be confusing. This function applies the validity check to the
mixmod parameters before we do anything to them.

If *any* parameters are bad, this returns -np.inf. Otherwise returns
0.

Inputs:

    parsmix = input mixture parameters. Order [fbad, ...]

    islog10frac = fbad supplied as log10

    islog10vxx = any vxx supplied as log10

    minffg = minimum mixture fraction for foreground

    maxvxx = maximum variance for the background component

Returns:

    lnprior(pars) = ln(prior) enforcing only validity constraints.

    """

    # If no parameters, say nothing about the prior.
    if np.size(parsmix) < 1:
        return 0.

    # Apply the parsing, return values (not log10)
    ffg, vxx = parsemixpars(parsmix, islog10frac, islog10vxx)
    
    # return with badval if either of the parameters are bad
    if not np.isfinite(ffg) or not np.isfinite(vxx):
        return -np.inf

    # If the walker is trying zero foreground or huge vxx, disallow.
    if ffg < minffg or vxx > maxvxx:
        return -np.inf
    
    # If we reached this point then our prior parameters do not
    # violate the conditions we set above.
    return 0.
        
def sumlnlike(pars, transf, xytarg, covtarg, covextra=0.):

    """Returns sum(log-likelihood) for a single-population model"""

    expon, det, piterm = lnlike(pars, transf, xytarg, covtarg, covextra)
    return np.sum(expon) + np.sum(det) + np.sum(piterm)

def sumlnlikefrac(pars, transf, xytarg, covtarg, covextra, \
                  ffg = 1., covbg=np.eye(2) ):

    """Returns sum(ln likelihood) for a foreground-background mixture
model. For the foreground component, the likelihood is computed as

    ln(like)_fg = -ln(2pi) -0.5 ln(|V|) - 0.5( dx^T.V^{-1}.dx ) + ln(P_fg)

where P_fg is the fraction of datapoints assigned to the foreground
component. 

For the background model, the likelihood is computed as

    ln(like)_bg = -ln(2pi) -0.5 ln(|V_b|) - 0.5( dx^T.V_b^{-1}.dx ) + ln(1.-P_fg)

where V_bg is the covariance including the extra variance of the
background component. The returned likelihood is then

    ln(like) = sum(ln( like_fg + like_bg ))

If the supplied foreground fraction is >= 1.0, drops back to a
single-component model (set P_fg=1 in the first expression above).

Inputs:

    pars = [M] - transformation parameters

    transf = transformation object including source data, covariances,
    and methods to transform from source to target plane

    xytarg = [N,2] - xy positions in the target frame

    covtarg = [N,2,2] - xy covariances in the target frame

    covextra = [N,2,2] - extra xy covariances from the model

    fracfg = Scalar, fraction of the population associated with the
    foreground.

    covbg = Scalar or [2,2]. extra covariance to add for the background
    component. Defaults to [[1,0],[0,1]] - so variance of one unit in
    each direction.

Returns:

    sumln(like) = sum of log-likelihood of the mixture model

    lnlike_fg = log-likelihood of the foreground component, per datapoint

    lnlike_bg = log-likelihood of the background component, per datapoint

    """

    # If 100% of the model is foreground, we don't need to do the
    # two-mix model, and the sum is the same. The "0, 0" at the end is
    # for compatibility with emcee blobs if we are using them.
    if ffg >= 1.:
        return sumlnlike(pars, transf, xytarg, covtarg, covextra), 0, 0
    
    # foreground component
    lnlike_fg = lnlikestat(pars, transf, xytarg, covtarg, covextra) + np.log(ffg)

    # For the background component, we add the background extra
    # variance onto covextra. Thanks to numpy broadcasting rules, this
    # works whether the existing covextra is [2,2] or [N,2,2], or
    # zero. (E.g. [N,2,2] + [2,2] has the same effect as [N,2,2] +
    # [None, 2, 2].
    covextra_bg = covextra + covbg
    lnlike_bg =  lnlikestat(pars, transf, xytarg, covtarg, covextra_bg) \
        + np.log(1.0-ffg)

    # return the sum of the foreground and background components. Also
    # return the individual ln(probs) for optional use in computation
    # of log responsibilities.
    sumlnlike_mix = np.sum(np.logaddexp( lnlike_fg , lnlike_bg ))

    return sumlnlike_mix, lnlike_fg, lnlike_bg
    
def lnlikestat(pars, transf, xytarg, covtarg, covextra=0.):

    """Returns the sum of all three terms on a per-object basis."""

    expon, det, piterm = lnlike(pars, transf, xytarg, covtarg, covextra)

    return expon + det + piterm
    
def lnlike(pars, transf, xytarg=np.array([]), covtarg=np.array([]), \
           covextra=0. ):

    """(log-) badness-of-fit statistic for transformation. Evaluates the
logarithm of the gaussian badness-of-fit statistic for each point,
i.e.

    ln(like) = -ln(2pi) -0.5 ln(|V|) - 0.5( dx^T.V^{-1}.dx )

    whose terms here are returned in reverse order:
    
    -0.5(dx^T.V^{-1}.dx), -0.5 ln(|V|), -ln(2pi)

    """

    # evaluate the function for the current params
    xytran, covtran = feval(pars, transf)

    # Now form the deltas array and covariances-sum arrays.
    deltas = xytarg - xytran
    covars = covtran + covtarg + covextra

    # Reminder that there are two pieces in lnlike to evaluate! The
    # exponential term (delta x^T . V^-1. delta x) and the determinant
    # term (|covar|). We do these in turn:
    
    # 1. The exponent: invert the sum covariance array and find the
    # quantity u^T.V^{-1}.u
    invcov = np.linalg.inv(covars)
    expon = uTVu(deltas, invcov)
    term_expon = -0.5 * expon

    # 2. The determinant:
    dets = np.linalg.det(covars)
    term_dets = -0.5 * np.log(dets)

    # 3. the -ln(2pi)
    term_2pi = term_dets * 0. -np.log(2.0*np.pi)
    
    # Return the two terms, but DO NOT SUM THEM YET.
    return term_expon, term_dets, term_2pi

def lnprob(parsIn, transf, xytarg, covtarg=np.array([]), \
           addvar=False, nvar=1, fromcorr=False, islog10=False, \
           nnoise=0, mags=np.array([]), \
           nmix=0, \
           islog10mixfrac=True, \
           islog10mixvxx=True, \
           retblobs=False, \
           methprior=lnprior_unif, \
           methlike=sumlnlike, \
           methprior_noise=lnprior_noisemodel_rect, \
           methprior_mixmod=lnprior_mixmod_rect):

    """Evaluates ln(posterior), summed over the datapoints. Takes the
method to compute the ln(prior) and ln(likelihood) as arguments.

Inputs:

    parsIn = 1D array of model parameters, in the order [transfxy,
    noise, shape, mixture]

    transf = object with methods for performing the transformation and
    source covariances. The actual datapoints in the source frame and
    the covariances in that frame are included in this object, as
    attributes transf.x, transf.y, transf.covxy . 

    xytarg = [N,2] array of positions in the target frame onto which
    we are trying to map our inputs.

    covtarg = [N,2,2] array of covariances associated with the target
    positions.

    addvar [T/F] = interpret the [-1]th parameter as extra variance to
    be added in both target dimensions equally.

    nvar = number of entries corresponding to covariance. Maximum 3,
    in the order [Vxx, Vyy, Vxy]

    fromcorr [T/F] = Any extra variance is supplied as [sx, sy/sx,
    rho] instead of [vx, vy, covxy].

    islog10 [T/F] = sx is supplied as log10(sx). Applies only if
    fromcorr is True.

    nnoise = number of parameters in parsIn that refer to the noise
    model
    
    mags = array of apparent magnitudes

    nmix = number of parameters that refer to the mixture model

    islog10mixfrac = mixture fraction supplied as log10(frac)

    islog10mixvxx = mixture background vxx supplied as log10(vxx)

    methprior = method for evaluating prior on transformation parameters

    methprior_noise = method for evaluating prior on noise vs mag model
    
    methprior_mixmod = method for evaluating prior on mixture model

    retblobs = return information for emcee blobs along with the
    ln(posterior). This is meant for running calculations on samples
    after the fact, if used as part of a sampler run it may lead to
    very large output files and blob sizes. For example: 40,000
    chainlen with 50 datapoints and a 7-parameter model produces blobs
    that require about 500 MB of RAM (and a .h5 sampler output file
    788 MB). Only set this to True if you have enough RAM and storage
    space.
    
Returns:

    lnprob = ln(posterior probability)   (scalar)

    loglike(fg) = per-datapoint log likelihood of foreground model

    loglike(bg) = per-datapoint log likelihood of background model

    """

    # Split the transformation parameters from noise, etc. parameters.
    pars, addnoise, addvars, addmix = \
        splitmodel(parsIn, nnoise, nvar, nmix)

    # The selection of prior method for the correlation shape should
    # probably be promoted to the calling method. For the moment we
    # enforce it here.
    methprior_corr=lnprior_corrmodel2_rect
    if nnoise < 1:
        methprior_corr=lnprior_corrmodel3_rect

    # Evaluate the prior on the transformation parameter and on any
    # noise parameters
    lnprior_transf = methprior(pars)
    lnprior_noise = methprior_noise(addnoise)
    lnprior_corr = methprior_corr(addvars)
    lnprior_mixmod = methprior_mixmod(addmix, islog10mixfrac, islog10mixvxx)

    lnprior = lnprior_transf + lnprior_noise + lnprior_corr + lnprior_mixmod
    if not np.isfinite(lnprior):
        if retblobs:
            return -np.inf, np.array([]), np.array([])
        else:
            return -np.inf
        
    # unpack the mixture model parameters.
    ffg, covbg = parsemixpars(addmix, \
                              islog10frac=islog10mixfrac, \
                              islog10vxx=islog10mixvxx)
    
    # Generate any additional extra covariance.
    cov_ok = True
    covextra, cov_ok = extracovar(addnoise, mags, \
                                  addvars, fromcorr, islog10) 

    # now evaluate the ln likelihood using our mixture-aware model.
    lnlike, ll_fg, ll_bg = \
        sumlnlikefrac(pars, transf, xytarg, covtarg, covextra, ffg, covbg)

    # 2024-08-08 testing the mixture-aware version. NOTE TO SELF -
    # don't delete this until after testing the mixmod with outliers.
    #
    ## evaluate ln likelihood.
    #if np.size(addmix) < 1:

    #    # single-component model
    #    lnlike = methlike(pars, transf, xytarg, covtarg, covextra) 

    #else:
    #    fbackg = addmix[0]
    #    varbg = 1.
    #    if np.size(addmix) > 1:
    #        varbg = addmix[1]

    #    # covariance for the background component. For the moment we
    #    # just add in a very large covariance as an extra component
    #    # (we cannot use outliers to fit covextra anyway).
    #    covextrabg = cov32covn22([varbg], xytarg.shape[0])
            
    #    # Foreground and background models on a per-object basis
    #    ll_fg = lnlikestat(pars, transf, xytarg, covtarg, covextra) + np.log(1.0-fbackg)
    #    ll_bg = lnlikestat(pars, transf, xytarg, covtarg, covextrabg) + np.log(fbackg)

    #    # Sum along the data for the foreground and background component
    #    lnlike = np.sum(np.logaddexp(ll_fg, ll_bg))
        
    # If this is about to return nan, provide a warning and show the
    # parameters. 
    if np.any(np.isnan(lnlike)):
        print("lnprob WARN - at least one NaN entry. Trial params:")
        print(parsIn)

    # returns the sum ln posterior, as well as the individual ln(like)
    # values so that emcee's "blobs" feature can track them
    if retblobs:
        return lnprior + lnlike, ll_fg, ll_bg
    else:
        return lnprior + lnlike
    
    # return the ln posterior
    # return lnprior + lnlike


def feval(pars, transf):

    """Evaluates the transformation and propagates the covariances. Inputs:

    pars = 1D array of parameters

    transf = transformation object from unctytwod.py. Must already be
    initialized.

    returns: transformed xy positions, propagated covariances"""

    transf.updatetransf(pars)
    transf.propagate()

    return transf.xytran, transf.covtran

def lnfom(pars, transf, xytarg):

    """Evaluates sum(fom) where fom is the negative of the sum of
residuals. This is for cases where we don't have or don't trust the
uncertainties (i.e. we're doing least-squares on the deltas).

    """

    xytran = ftran(pars, transf)
    deltas = xytarg - xytran

    return 0. - deltas**2


def ftran(pars, transf):

    """Evaluates only the transformation of points, ignoring the
covariances. 

    """

    transf.updatetransf(pars)
    xtran, ytran = transf.propxy(transf.x, transf.y)

    xytran = np.zeros((np.size(xtran), 2))
    xytran[:,0] = xtran
    xytran[:,1] = ytran
    
    return xytran

#### Example usages of these pieces follow

def makefakexy(npts=2000, \
                 xmin=-10., xmax=10., ymin=-10., ymax=10.):

    """Utility to make unform random sampled datapoints"""

    xy = np.random.uniform(size=(npts,2))
    xy[:,0] = xy[:,0]*(xmax-xmin) + xmin
    xy[:,1] = xy[:,1]*(ymax-ymin) + ymin

    return xy

def makefakemags(npts=2000, expon=2.5, maglo=16., maghi=22., \
                 seed=None):

    """Utility - creates array of apparent magnitudes following a
power-law distribution

    """

    rng = np.random.default_rng(seed)
    sraw = rng.power(expon, npts)

    return sraw*(maghi - maglo) + maglo

def assignoutliers(x=np.array([]), foutly=0., seed=None):

    """Assigns outlier status to a (uniform) random permutation of the objects

Inputs:

    x = [N (,...)] array of datapoints.

    foutly = fraction of outliers.

    seed = random number seed.

Returns:
    
    isoutlier = [N] boolean array indicating whether a datapoint is
    assigned outlier

    """

    if np.size(x) < 1:
        return np.array([])

    # Initialize outlier status
    npts = np.shape(x)[0]
    isoutly = np.repeat(False, npts)
    
    # If fraction of outliers is zero, nothing is an outlier.
    frac = parsefraction(foutly)
    if frac <= 0. or frac > 1.:
        return isoutly

    # rng.choice doesn't do quite what I expect... e.g. if 20 points
    # and 0.99 are outliers, sometimes I get more than 2
    # outliers. Fall back on the sort(random uniform) method instead,
    # we're only doing this once per simulation set.
    rng=np.random.default_rng(seed=seed)
    xdum = rng.uniform(size=npts)
    lbad = np.argsort(xdum)[0:int(npts*frac)]
    isoutly[lbad] = True

    return isoutly

def makeoutliers(x=np.array([]), foutly=0., vxxoutly=1., \
                 islog10frac=False, islog10vxx=False):

    """Makes outliers for input positions.

Inputs:

    x = datapoints (used to determine the length of the outlier array)

    foutly = fraction of points that are outliers.

    vxxoutly = variance (per parameter) for outlier noise

    islog10frac = outlier fraction supplied as log10(frac)

    islog10vxx = vxx supplied as log10(vxx)

Returns:

    nudgexy_outliers = nudges in the target frame for x, y outliers. 

    isoutly = boolean array giving outlier status (True = outlier)

"""

    # Nothing to do if no data passed in
    if np.size(x) < 1:
        return np.array([]), np.array([]) 

    # Initialize the nudges
    npts = np.shape(x)[0]
    nudgexy_outliers = np.zeros((npts, 2))

    # Parse the parameters.
    foutliers, vxx = parsemixpars([foutly, vxxoutly], \
                                  islog10frac, islog10vxx)

    # Do not return if badvals, but continue to generate zero deltas.
    if not np.isfinite(foutliers):
        foutliers = 0.
        
    # Which objects are background?
    isoutly = assignoutliers(x, foutliers)

    # If no objects are background, nothing to do. Return the default
    # outputs.
    if np.sum(isoutly) < 1:
        return nudgexy_outliers, isoutly

    # OK if we're here then we do make the outliers. This isn't quite
    # as convenient yet as I would like it to be, consider a
    # replication argument for CovarsNx2x2...
    covs = np.zeros((npts, 2, 2))
    covs[:,0,0] = vxx
    covs[:,1,1] = vxx
    
    Coutly = CovarsNx2x2(covs)

    # draw samples (for all the points)...
    nudgeall = Coutly.getsamples()

    # and take those that ARE outliers for the nudges
    nudgexy_outliers[isoutly] = nudgeall[isoutly]
    
    return nudgexy_outliers, isoutly
    
def makeunifcovars(npts=2000, sigx=0.1, sigy=0.07, sigr=0.2):

    """Makes fake covariances in form [N,2,2] with the same [2,2]
covariance in each plane. Returns a CovarsNx2x2 object."""

    vstdxi = np.ones(npts)*sigx
    vstdeta = vstdxi * sigy/sigx
    vcorrel = np.ones(npts)*sigr
    CS = CovarsNx2x2(stdx=vstdxi, stdy=vstdeta, corrxy=vcorrel)
    
    return CS

def makemagcovars(parsnoise, mags, parscorr):

    """Makes fake covariances in the form [N,2,2] using the parameters of
a magnitude-dependent noise model and shape parameters [stdy/stdx,
corrxy]. Returns a CovarsNx2x2 object. Inputs:

    parsnoise = up to 3-element array of noise model parameters

    mags = [N]-element array of magnitudes"""

    # Debug lines no longer needed
    #
    #print("=============================")
    #print("makemagcovars DEBUG - inputs:")
    #print(parsnoise, np.shape(parsnoise))
    #print(mags.shape, np.min(mags), np.max(mags))
    #print(parscorr, np.shape(parscorr))
    #print("=============================")
    
    covsxy = mags2cov(parsnoise, mags, parscorr)
    CS = CovarsNx2x2(covars=np.copy(covsxy))

    return CS
    
def wtsfromcovars(covars=np.array([]), scalebydet=True ):

    """Utility - returns inverse covars as weights, optionally scaled bys sqrt(median determinant)

    """

    wraw = np.linalg.inv(covars)
    sfac = 1.
    if scalebydet:
        #sfac = np.median(np.sqrt(np.linalg.det(wraw)))
        sfac = np.sqrt(np.median(np.linalg.det(wraw)))

    return wraw / sfac

def quivresid(xy=np.array([]), dxy=np.array([]),  ax=None, \
              quant=0.9, color='k', \
              stitl='', labelx='x', labely='y'):

    """Utility - creates a residuals quiver plot"""

    # Needs an axis object on which to operate
    if ax is None:
        return

    # Convenience views
    x = xy[:,0]
    y = xy[:,1]

    dx = dxy[:,0]
    dy = dxy[:,1]

    dmag = np.sqrt(dxy[:,0]**2 + dxy[:,1]**2)
    quse = np.min([quant, 1.0])
    ql = np.quantile(dmag, quse)

    # string for quiver label
    qs = r'%.1e (%0.f$^{th}$ percentile)' % (ql, quse*100.)
    
    # now do the quiver plot and key
    blah = ax.quiver(xy[:,0], xy[:,1], dxy[:,0], dxy[:,1], color=color)
    qk = ax.quiverkey(blah, 0.1, 0.95, U=ql, \
                      label=qs, \
                      labelpos='E', fontproperties={'size':8})

    # adjust the axis scale to make room for the quiver key
    ax.set_xlim(ax.get_xlim()*np.repeat(1.1, 2) )
    ax.set_ylim(ax.get_ylim()*np.repeat(1.1, 2) )

    if len(stitl) > 0:
        ax.set_title(stitl)

    if len(labelx) > 0:
        ax.set_xlabel(labelx)
    if len(labely) > 0:
        ax.set_ylabel(labely)
    
########## "Test" routines that use these pieces. Some are messy.

def split1dpars(pars1d=np.array):

    """Utility - split 1D params into 2x1d expected by Poly() objects"""

    npars = int(np.size(pars1d)/2)
    return pars1d[0:npars], pars1d[npars::]

def splitxypars(pars=np.array([]) ):

    """Given an xy pars array, splits it into two arrays for each of the
x, y parameters.

Inputs:

    pars = [A_00, A_10, ... , B_00, B_10, ...] parameters


Returns:

    parsx = [A_00, A_10, ... ] "x" parameters

    parsy = [B_00, B_10, ... ] "y" parameters

    """

    # Cowardly refuse to split a non-even array
    szpars = np.size(pars)
    if szpars % 2 != 0:
        return np.array([]), np.array([])

    halfsz = int(szpars * 0.5)
    return pars[0:halfsz], pars[halfsz::]

    
def scalexyguess(guessxy=np.array([]), simxy=np.array([]) ):

    """Scales initial guesses in x,y parameters by offset from simulation
x,y parameters, accounting for any differences in degree between the
two sets.

Inputs:

    guessxy = [A_00, A_10, ... , B_00, B_10, ...] parameters for guess

    simxy = [a_00, a_10, ... , b_00, b_10, ... ] simulated parameters

    """

    # Partition both arrays into x, y sets
    guessx, guessy = splitxypars(guessxy)
    simx, simy = splitxypars(simxy)

    # If either of the returned x-arrays have zero length, something
    # was wrong with the input. Return.
    if guessx.size < 1 or simx.size < 1:
        return np.array([])
    
    # Because the guessx, simx and guessy, simy arrays now ARE in
    # increasing order of powers of xy, we can validly compare their
    # offsets. So treat the x- and y- parameters separately:
    scalesx = scaleguessbyoffset(guessx, simx)
    scalesy = scaleguessbyoffset(guessy, simy)

    # Now we abut the two scales arrays back into our 1D
    # convention. This should ALWAYS be the same length as guess.
    scalesxy = np.hstack(( scalesx, scalesy ))
    
    return scalesxy
    

def scaleguessbyoffset(guess=np.array([]), pars=np.array([]), \
                       mult=5., defaultscale=1.0e-3, \
                       min_offset=1.0e-50, min_guess=1.0e-50):

    """Given initial-guess parameters and simulation parameters, sets the
range of offsets for walkers from the difference between the guess and
truth parameters. If the guess has more entries than the simulation
(e.g. if we are trying to use the 'wrong' model to fit simulated data)
then argument defaultscale is used.

    WARNING: no parsing at all is done of the entries. They are assumed
    to have the same meaning in both guess and pars, even if the
    lengths are different.

Inputs:

    guess = [M] element estimate for parameters

    pars = [N] element simulated parameters

    mult = multiple of the |guess - sim| offset to use for the scale
    factor for each parameter

    defaultscale = default scale factor to use.

    min_offset = minimum offset between guess and scale. If any offset
    is below this, its scale factor is replaced by defaultscale.

    min_guess = minimum value for input guess (since the offsets are
    scaled by the guess value). For any guesses below this, the
    scalefactor is replaced by defaultscale.

Returns:

    guess_scale = [M-] element array of scale factors for the initial
    guess.

    """

    # Handle possibly scalar input
    guesspars = np.atleast_1d(guess)
    simpars = np.atleast_1d(pars)
    
    # array (or list) lengths; default return value
    szguess = np.size(guesspars)
    szpars = np.size(simpars)
    guess_scale = guess * 0. + defaultscale

    # Nothing to do if guess or pars has zero length
    if szguess < 1 or szpars < 1:
        return guess_scale

    # Abs offset between elements common to both, with nonzero input
    # guess
    lboth = np.arange(min([szguess, szpars]) )
    lboth = lboth[np.abs(guesspars[lboth]) > min_guess] 
    guess_scale[lboth] = np.abs( (guesspars[lboth] - simpars[lboth])\
                                 /guesspars[lboth] ) * mult

    # Do a little checking
    bsmall = (guess_scale < min_offset) | \
        (np.isnan(guess_scale)) | \
        (np.isinf(guess_scale))
    guess_scale[bsmall] = defaultscale

    return guess_scale

def padpars(truths=np.array([]), guess=np.array([]), \
            nnoisetruth=0, nshapetruth=0, \
            nnoiseguess=0, nshapeguess=0, \
            nmixtruth=0, nmixguess=0):

    """Given a truths array and a guess array, ensure the indices match
up. This means unpacking each array into transformation arrays and the
arrays that communicate the noise model (vs mag) and noise shape,
updating the entries, then re-packing the results into the correct
order.

Inputs:

    truths = [transf, noise, shape] array for truths

    guess = [transf, noise, shape] array for guess

    nnoisetruth = number of noise parameters for truth

    nshapetruth = number of shape parameters for truth

    nnoiseguess = number of noise parameters for guess

    nshapeguess = number of shape parameters for guess

    nmixtruth = number of mixture model parameters for truth

    nmixguess = number of mixture model parameters for guess

Returns:

    scalesret = array of scale factors for MCMC initial guesses

    truthsret = array of 'truth' values with same ordering as guess

    """

    # Step 1: unpack the supplied parameter arrays
    truthpars, truthnoise, truthshape, truthmix = \
        splitmodel(truths, nnoisetruth, nshapetruth, nmixtruth)

    guesspars, guessnoise, guessshape, guessmix = \
        splitmodel(guess, nnoiseguess, nshapeguess, nmixguess)

    # 2024-08-07 nothing done with the mixture parameters yet.
    
    # Step 2: find the offsets and the perturbation scales for the
    # MCMC walker start positions
    scalesxy = scalexyguess(guesspars, truthpars)
    scalesnoise = scaleguessbyoffset(guessnoise, truthnoise)
    scalesshape = scaleguessbyoffset(guessshape, truthshape)
    scalesmix = scaleguessbyoffset(guessmix, truthmix)
    
    # Step 3: pad the truths for the individual pieces
    truthsxy = padxytruths(truthpars, guesspars)
    truthsnoise = padtruths(truthnoise, guessnoise)
    truthsshape = padtruths(truthshape, guessshape)
    truthsmix = padtruths(truthmix, guessmix)
    
    # Step 4: gather the pieces for return
    scalesret = np.hstack(( scalesxy, scalesnoise, scalesshape, scalesmix ))
    truthsret = np.hstack(( truthsxy, truthsnoise, truthsshape, truthsmix ))

    return scalesret, truthsret
    
def padxytruths(truthsxy=np.array([]), guessxy=np.array([]) ):

    """Given a 'truthsxy' array of parameters [xpars, ypars] and a guess
array following the same convention, returns a version of the truths
array with each of the x, y parameters having the same length as its
guess counterpart. 

Inputs:

    truthsxy = [A_00, A_10, ..., B_00, B_10, ...] truth parameters

    guessxy  = [a_00, a_10, ..., b_00, b_10, ...] guess parameters

Returns:

    truthsxy_adj = [A_00, A_10, ... , B_00, B_10, ... ] truth
    parameters of same length as guessxy. Any values in guessxy with
    no counterparts in truths are assigned None.

    """

    truthsx, truthsy = splitxypars(truthsxy)
    guessx, guessy = splitxypars(guessxy)

    # If either input has non-even size, return recognizable badvals
    if np.size(truthsx) < 1 or np.size(guessx) < 1:
        return np.array([])

    truthsx_adj = padtruths(truthsx, guessx)
    truthsy_adj = padtruths(truthsy, guessy)

    return np.hstack(( truthsx_adj, truthsy_adj ))

    
def padtruths(truths=np.array([]), guess=np.array([]) ):

    """Given a 'truths' array and a 'guess' array, returns a version of
truths that has the same number of elements as 'guess'. If guess is
larger than truths, the returned truths array is padded with None.

Inputs:

    truths = [M] element array of 'truth' values

    guess = [N] element array of guesses

Returns:

    truths_adj = [N]-element array of truth values, adjusted to the
    same length as guess.

    """

    sztruths = np.size(truths)
    szguess = np.size(guess)

    if sztruths == szguess:
        return np.copy(truths)

    if sztruths > szguess:
        return truths[0:szguess]

    truths_adj = np.hstack(( truths, np.repeat(None, szguess-sztruths) ))
    return truths_adj
    
    
def labelstransf(transf=None, sx='A', sy='B'):

    """Returns a string of labels for plotting.

Inputs:

    transf = Patternmatrix object. Must have method setplotlabels().

    sx = label string, x

    sy = label string, y

Returns:

    slabels = list of labels, in order [x params, y params]

Example:

    Suppose pars2x is a Patternmatrix object describing a polynomial of order 1. Then:

    slabels = labelstransf(pars2x, 'A', 'B')

would return:

    ['$A_{00}$', '$A_{10}$', '$A_{01}$', '$B_{00}$', '$B_{10}$', '$B_{01}$']

for which the transformation is interpreted as

    xi(x,y) = A_00 + A_10 p(x, 0) + A_01 p(0, y)
    eta(x,y) = B_00 + B_10 p(x, 0) + B_01 p(0, y)

where 

    p(x, 0) is the polynomial of order 1 in x, order 0 in y
    p(0, y) is the polynomial of order 0 in x, order 1 in y

    etc.

    """

    if transf is None:
        return []

    slabelsx = transf.setplotlabels(sx) 
    slabelsy = transf.setplotlabels(sy) 

    return slabelsx + slabelsy
    
def labelsnoisemodel(nparsnoise=0, nparscorr=0):

    """Utility - returns the list of additional variable labels in the
order [noise shape, noise model].

Inputs:

    nparsnoise = number of parameters for the noise model

    nparscorr = number of parameters for the covariance shape


    """

    # Correlation labels
    lcorr = []
    if nparscorr > 0:
        lcorr = [r'$s_\eta/s_\xi$']
    if nparscorr > 1:
        lcorr = lcorr + [r'$\rho_{\xi\eta}$']

    # noise model labels
    lnoise = []
    if nparsnoise > 0:
        lnoise = [r'$log_{10}(a)$']
    if nparsnoise > 1:
        lnoise = lnoise + [r'$log_{10}(b)$']
    if nparsnoise > 2:
        lnoise = lnoise + [r'$c$']

    return lnoise + lcorr
    
def labelsaddvar(npars_extravar=0, extra_is_corr=False, std_is_log=False):

    """Utility - returns list of additional variable labels depending on
what we're doing

    """

    if npars_extravar < 1:
        return []
    lextra = [r'$V$']
    if extra_is_corr:
        lextra = [r'$s$']

        if std_is_log:
            lextra = [r'$log_{10}(s)$']

    if npars_extravar < 2:
        return lextra

    lextra = [r'$V_{\xi}$', r'$V_{\eta}$']
    if extra_is_corr:
        lextra = [r'$s_{\xi}$', r'$s_{\eta}/s_{\xi}$']

        if std_is_log:
            lextra[0] = [r'$log_{10}(s_{\xi})$']
        
    if npars_extravar < 3:
        return lextra

    lextra = [r'$V_{\xi}$', r'$V_{\eta}$', r'$V_{\xi \eta}$']
    if extra_is_corr:
        lextra = [r'$s_{\xi}$', r'$s_{\eta}/s_{\xi}$', r'$\rho_{\xi,\eta}$']

        if std_is_log:
            lextra[0] = [r'$log_{10}(s_{\xi})$']

        
    return lextra

def assignnoiseguess(guessnoise=None, guessshape=None, \
                     simnoise=np.array([]), simshape=np.array([]), \
                     nudgenoise=0.01, nudgeshape=0.01, mult=5.):

    """Wrapper to either assign noise from input guess or scale it from
simulated parameters."""

    # assign noise from simulation if guess not given
    if guessnoise is None:
        guessnoise, scalesnoise = \
            guessfromsim(simnoise, nudgenoise, mult)
    else:
        scalesnoise = np.repeat(nudgenoise*mult, np.size(guessnoise))

    # assign shape from simulation if guess not given
    if guessshape is None:
        guessshape, scalesshape = \
            guessfromsim(simshape, nudgeshape, mult)
    else:
        scalesshape = np.repeat(nudgeshape*mult, np.size(guessshape))

    return guessnoise, guessshape, scalesnoise, scalesshape
        
def noiseguessfromsim(parsnoise=np.array([]), \
                      parsshape=np.array([]), \
                      nudgenoise=0.01, \
                      nudgeshape=0.01, \
                      multscale=5.):

    """Generates guess for noise and shape model by perturbing the
simulated parameters.

Inputs:

    parsnoise = 1D array of parameters describing the variation of noise stdx with apparent magnitude.

    parsshape = 1D array of parameters describing [stdy/stdx, corrxy]

    nudgenoise = scale of noise perturbations as fraction of values

    nudgeshape = scale of shape perturbations as fraction of values

    multscale = multiple of nudge* to use when generating scale
    factors for the MCMC initial walker positions

Returns:

    guessnoise = array of 'guesses' for the noise model

    guessshape = array of 'guesses' for the shape model

    scaleguess = array of scale factors for the noise parameters

    """

    # refactored the guess into 1D method again!
    guessnoise, scalesnoise = \
        guessfromsim(parsnoise, nudgenoise, multscale)
    guessshape, scalesshape = \
        guessfromsim(parsshape, nudgeshape, multscale)

    return guessnoise, guessshape, \
        np.hstack(( scalesnoise, scalesshape ))
    
    ## Note to self: this should all work even if no parameters at all
    ## were passed.
    
    ## Set up the labels for plots
    #npars_noise = np.size(parsnoise)
    #npars_extravar = np.size(parsshape)

    ## Set up perturbations on the noise model and shape model
    #pertnoise = np.random.normal(size=npars_noise)*nudgenoise
    #guessnoise = parsnoise + pertnoise

    #pertshape = np.random.normal(size=npars_extravar)*nudgeshape
    #guessshape = parsshape + pertshape

    ## return scale factors for the guesses
    #scalesnoise = np.repeat(nudgenoise*multscale, guessnoise.size)
    #scalesshape = np.repeat(nudgeshape*multscale, guessshape.size)
    #scalesguess = np.hstack(( scalesnoise, scalesshape ))
    
    #return guessnoise, guessshape, scalesguess

def guessfromsim(simpars=np.array([]), nudge=0.01, mult=5.):

    """Produces a guess for noise parametrs from simulated parameters.

Inputs:

    simpars = array of noise parameters

    nudge = fraction of parameter value to use when perturbing

    mult = multiple of nudge to use for scale factors

Returns

    guess = array of perturbed initial guess values

    scales = array of scale factors to use when setting walker initial
    positions using the guess

    """

    sz = np.size(simpars)
    pert = np.random.normal(size=sz)*nudge
    guess = simpars + pert

    scales = np.repeat(nudge*mult, sz)

    return guess, scales
    
def anycovbad(covars=np.array([]) ):

    """Returns True if any of the N,2,2 input covariance planes are
singular, OR if blank input given"""

    if np.size(covars) < 4:
        return True

    nsingular = np.sum(findcovsingular(covars))
    if np.sum(findcovsingular(covars)) > 0:
        return True 

    return False
    
def findcovsingular(covars=np.array([])):

    """Utility - given [N,2,2] covariance array, finds if any planes are
singular. Returns boolean array (True where a plane is singular)"""

    if np.size(covars) < 4:
        return np.array([])

    return np.linalg.det(covars) <= 0.

def plotsamplescolumn(samples, fignum=2, slabels=[]):

    """Utiltity - plots samples"""

    sshape = samples.shape
    sdim = np.size(sshape)

    ssho = samples
    if sdim < 3:
        ssho = samples[:,np.newaxis,:]
    
    fig=plt.figure(fignum)
    fig.clf()
    lsampl = np.arange(ssho.shape[0])
    iplot = 0
    for ipar in range(ssho.shape[-1]):
        iplot += 1        
        ax21 = fig.add_subplot(samples.shape[-1], 1, iplot)        
        for j in range(ssho.shape[1]):
            dum21 = ax21.plot(lsampl, ssho[lsampl,j,ipar], \
                              alpha=0.5)

        if len(slabels) == ssho.shape[-1]:
            ax21.set_ylabel(slabels[ipar])
            
    ax21.set_xlabel('Sample number')

    # try deactivating the axis label offset
    ax21.yaxis.get_major_formatter().set_useOffset(False)
    
    # Ensure there is room for our nice labels
    fig.subplots_adjust(left=0.2)
    
    # return the figure as an obejct we can work with
    return fig

def testpoly(npts=2000, \
             deg=3, degfit=-1, \
             xmin=-1., xmax=1., ymin=-1., ymax=1., \
             sigx=0.001, sigy=0.0007, sigr=0.0, \
             polytransf='Polynomial', \
             polyfit='Polynomial', \
             covtranscale=1., \
             showpoints=True, \
             nouncty=False, \
             tan2sky=False, \
             alpha0=35., delta0=35.):

    """Creates and fits fake data: polynomial or tan2sky

    Example call:

    fittwod.testpoly(deg=3, npts=2000, nouncty=True, tan2sky=False, degfit=3, polyfit='Legendre', polytransf='Legendre')


"""

    # fit degree?
    if degfit < 0:
        degfit = deg
    
    # What object are we using?
    if tan2sky:
        transf=unctytwod.Tan2equ
    else:
        transf=unctytwod.Poly
    
    # Make up some data and covariances in the source plane
    xy = makefakexy(npts, xmin, xmax, ymin, ymax)
    Cxy = makeunifcovars(xy.shape[0], sigx, sigy, sigr)
    
    # make up some parameters, abut them together into the 1D format
    # the various optimizers etc. will expect
    if tan2sky:
        pars1d=np.array([alpha0, delta0])
        PTruth = transf(xy[:,0], xy[:,1], Cxy.covars, pars1d)
    else:
        parsx, parsy = unctytwod.makepars(deg)
        pars1d = np.hstack(( parsx, parsy ))

        # transform source positions and covariances to the target frame
        PTruth = transf(xy[:,0], xy[:,1], Cxy.covars, parsx, parsy, \
                        kind=polytransf)
    PTruth.propagate()

    # Get the unperturbed transformed positions and covariances
    xytran = np.copy(PTruth.xytran)
    covtran = np.copy(PTruth.covtran)

    # Make a covars object from the covtran so that we can draw
    # samples
    Ctran = CovarsNx2x2(PTruth.covtran * covtranscale)

    # Now generate samples from the two sets of covariances, and nudge
    # the positions by these amounts
    if nouncty:
        nudgexy = xy * 0.
        nudgexytran = xytran * 0.
    else:
        nudgexy = Cxy.getsamples()
        nudgexytran = Ctran.getsamples()
    
    xyobs = xy + nudgexy
    xytarg = xytran + nudgexytran

    # covariances in the source and target frame
    covobs = Cxy.covars
    covtarg = Ctran.covars

    # interpret covariances as weights (useful when trying linear
    # least squares)
    W = np.ones(xy.shape[0])
    if not nouncty:

        wobs = wtsfromcovars(covobs)
        wtra = wtsfromcovars(covtran)

        W = np.matmul(wobs, wtra)
        
        
    # initial guess for the fit (should work for tan2sky as long as
    # we're not close to the pole)
    pertpars = np.random.uniform(-0.1, 0.1, size=np.size(pars1d))*pars1d
    parsguess1d = pars1d + pertpars
    
    if not tan2sky:

        # if the fit and truth parameters have different lengths then
        # we need to reformulate our guess:
        if degfit != deg:
            parsxgg, parsygg = unctytwod.makepars(degfit)
            parsgg = np.hstack(( parsxgg, parsygg ))
            pertpars = np.random.uniform(-0.1, 0.1, \
                                         size=np.size(parsgg))*parsgg
            parsguess1d = parsgg + pertpars
            
        parsxg, parsyg = PTruth.splitpars(parsguess1d)

        
    # Arrange things for optimization. For the polynomial, the size of
    # parsxg, parsyg are needed to set the jacobian appropriately for
    # the degree of fitting.
    if tan2sky:
        PFit = transf(xyobs[:,0], xyobs[:,1], covobs, pars1d)
    else:
        PFit = transf(xyobs[:,0], xyobs[:,1], covobs, parsxg, parsyg, \
                      kind=polyfit)

    # Show the points?
    if showpoints:
        fig1=plt.figure(1)
        fig1.clf()
        ax1 = fig1.add_subplot(221)
        ax2 = fig1.add_subplot(222)
        
        blah1  = ax1.scatter(xyobs[:,0], xyobs[:,1], marker='x', c='g', s=1)
        blah2a = ax2.scatter(xytran[:,0], xytran[:,1], marker='x', c='g', s=1)
        blah2b = ax2.scatter(xytarg[:,0], xytarg[:,1], marker='+', c='b', s=1)

        # use the objects to plot
        ax1.set_xlabel(PTruth.labelx)
        ax1.set_ylabel(PTruth.labely)
        ax2.set_xlabel(PTruth.labelxtran)
        ax2.set_ylabel(PTruth.labelytran)

    # Try the linear least squares guess
    xylsq = np.array([]) # default if we're doing tan2sky
    if not tan2sky:
        print("testpoly INFO - trying leastsq2d guess")
        t4 = time.time()
        LSQ = Leastsq2d(xyobs[:,0], xyobs[:,1], W, \
                        deg=degfit, kind=polyfit, \
                        xytarg=xytarg)
        t5 = time.time()
        print("Done in %.2e seconds" % (t5-t4))

        # project the leastsq2d solution for comparison later
        xylsq = LSQ.ev(xy[:,0], xy[:,1])

        # split the 1d params up using the same convention as the Poly objects:
        parsxlsq, parsylsq = PTruth.splitpars(LSQ.pars)
        
    # Now the function for the minimizer. The non-parameter arguments
    # passed in...
    if nouncty:
        args = (PFit, xytarg)
        ufunc = lambda *args: -np.sum(lnfom(*args))
    else:
        args = (PFit, xytarg, covtarg)
        ufunc = lambda *args: -np.sum(lnlikestat(*args))
       
    print("Trying minimization...")
    t0 = time.time()
    soln = minimize(ufunc, parsguess1d, args=args)
    print("... done in %.2e seconds" % (time.time() - t0) )
    
    # once we have a solution, try propagating this forward to see how
    # well we did.
    if tan2sky:
        PCheck = transf(xy[:,0], xy[:,1], Cxy.covars, soln.x)
    else:
        parsxf, parsyf = PTruth.splitpars(soln.x)
        PCheck = transf(xy[:,0], xy[:,1], Cxy.covars, parsxf, parsyf, \
                        kind=polyfit)
    PCheck.propagate()
    
    # Produce residuals plots
    residxy = PCheck.xytran - xytarg

    if showpoints:
        ax3 = fig1.add_subplot(223)
        ax4 = fig1.add_subplot(224)

        blah3 = ax3.scatter(xytarg[:,0], xytarg[:,1], c=residxy[:,0], s=1)
        blah4 = ax4.scatter(xytarg[:,0], xytarg[:,1], c=residxy[:,1], s=1)

        cb3 = fig1.colorbar(blah3, ax=ax3)
        cb4 = fig1.colorbar(blah4, ax=ax4)

        for ax in [ax3, ax4]:
            ax.set_xlabel(PTruth.labelxtran)
            ax.set_ylabel(PTruth.labelytran)

        ax3.set_title(r'$\Delta %s$' % (PTruth.labelxtran.replace('$','')))
        ax4.set_title(r'$\Delta %s$' % (PTruth.labelytran.replace('$','')))

        # Do a few figure labelings
        if tan2sky:
            ssup='Tangent plane to sky'
        else:
            ssup='Gen: %s(%i), Fit:%s(%i)' % \
                (polytransf, deg, polyfit, degfit)

        if nouncty:
            ssup = '%s: no unctys' % (ssup)
        else:
            # show the median det(covar)
            detcov = np.sqrt(np.median(np.linalg.det(covtran)))
            ssup = r'%s, $\sqrt{\langle|V|\rangle}$ = %.2e' % (ssup, detcov)
            
        fig1.suptitle(ssup)
        fig1.subplots_adjust(wspace=0.3, hspace=0.3, left=0.15, bottom=0.15)
        
    # compare the input and fit parameters
    if tan2sky:
        print("Parameters comparison: alpha0, delta0")
        print('alpha_0: %.2f, diffnce %.2e "' % \
              (pars1d[0], (soln.x[0]-pars1d[0])*3.6e3))
        print('delta_0: %.2f, diffnce %.2e "' % \
              (pars1d[1], (soln.x[1]-pars1d[1])*3.6e3))

    else:

        # quiver plots
        if np.size(parsxlsq) > 0:
            fig3 = plt.figure(3, figsize=(7,7))
            fig3.clf()
            ax31 = fig3.add_subplot(221)
            ax32 = fig3.add_subplot(222)

            slsq = 'Leastsq2d'
            smin = 'scipy.optimize.minimize'
            
            quivresid(xyobs, xylsq - xytarg, ax=ax31, color='r', \
                      stitl=slsq)
            quivresid(xyobs, residxy, ax=ax32, color='b', \
                      stitl=smin)

            # Show histograms of the deltas as well
            ax33 = fig3.add_subplot(223)
            ax34 = fig3.add_subplot(224)            

            ## do magnitudes
            #dmaglsq = np.sqrt(np.sum((xylsq-xytarg)**2, axis=1))
            #dmagmin = np.sqrt(np.sum(residxy**2, axis=1))

            # just do a single component for the moment
            dmaglsq = xylsq[:,0] - xytarg[:,0]
            dmagmin = residxy[:,0]

            # labels for histograms - use matplotlib's legen to take
            # care of the positioning
            sleglsq = r'$\sigma_{\Delta \xi} = %.2e$' % (np.std(dmaglsq))
            slegmin = r'$\sigma_{\Delta \xi} = %.2e$' % (np.std(dmagmin))
            
            blah33 = ax33.hist(dmaglsq, \
                               bins=100, color='r', label=sleglsq)
            blah34 = ax34.hist(dmagmin, \
                               bins=100, color='b', label=slegmin)

            #ax33.annotate(r'$\sigma_{\Delta \xi} = %.2e$' % (np.std(dmaglsq)), \
#                          (0.95, 0.95), ha='right', va='top', fontsize=9, \
#                          color='r', xycoords='axes fraction')
#            ax34.annotate(r'$\sigma_{\Delta \xi} = %.2e$' % (np.std(dmagmin)), \
#                          (0.95, 0.95), ha='right', va='top', fontsize=9, \
#                          color='b', xycoords='axes fraction')

            
            ax33.set_title(slsq)
            ax34.set_title(smin)
            for ax in [ax33, ax34]:
                #ax.set_xlabel(r'$|\vec{\Delta \xi}|$')
                ax.set_xlabel(r'$\Delta \xi$')
                leg = ax.legend(fontsize=8)
                
            # use the same supertitle as figure 1
            fig3.suptitle(ssup)
            fig3.subplots_adjust(wspace=0.3, hspace=0.3, \
                                 left=0.15, bottom=0.15, top=0.85)

        # Show the polynomial parameters comparison:
        npars = np.max([np.size(parsx), np.size(parsxf)])
        print("Parameters comparison: X, Y")

        # This is just a little awkward, since we may or may not be
        # using linear least squares as well for a comparison. 

        # avoid typos
        nolsq = np.size(parsxlsq) < 1

        # The Polynom() and Leastsq2d() objects store their parameters
        # in different orderings. We use the index "glsq" to map the
        # Leastsq2d() ordering to the Polynom() ordering.
        i_min = PCheck.pars2x.i
        j_min = PCheck.pars2x.j

        i_lsq = LSQ.pattern.isel
        j_lsq = LSQ.pattern.jsel

        print("INDICES INFO:")
        print("i_min      ", i_min)
        print("j_min      ", j_min)
        
        #print("i_lsq[::-1]", i_lsq[::-1])
        #print("j_lsq[::-1]", j_lsq[::-1])

        print("i_lsq      ", i_lsq)
        print("j_lsq      ", j_lsq)

        # now make these into 2D arrays
        c2dmin = np.zeros(( i_min.max()+1, i_min.max()+1))
        c2dlsq = np.zeros(( i_lsq.max()+1, i_lsq.max()+1), dtype='int')

        lmin = np.arange(i_min.size)
        c2dmin[i_min[lmin], j_min[lmin]] = lmin

        llsq = np.arange(i_lsq.size)
        c2dlsq[i_lsq[llsq], j_lsq[llsq]] = llsq

        print("c2d: min")
        print(c2dmin)
        print("c2d: lsq")
        print(c2dlsq)

        # try our simple method
        #count_lsq = c2dlsq[i_min[lmin], j_min[lmin]]
        count_lsq = c2dlsq[i_min, j_min]
        print(count_lsq)

        # YES this works. We can use this to reorder our lsq pattern
        # matrix into the same order as the polycoeffs() object.
        
        for ipar in range(npars):

            sind = ''
            
            # of course the indices in the two conventions don't line
            # up, so we have to fix that...
            if ipar < np.size(i_min):
                glsq = np.where((i_lsq == i_min[ipar]) & \
                                (j_lsq == j_min[ipar]))[0]

                # try our simple reordering
                print("Reorder check:", glsq, count_lsq[ipar])
                
                # indices track: do these indices line up?
                sind = 'i,j: %i, %i' % (PCheck.pars2x.i[ipar], \
                                        PCheck.pars2x.j[ipar])

                sind = '%s ## %i, %i' % (sind, \
                                         LSQ.pattern.isel[glsq], \
                                         LSQ.pattern.jsel[glsq])
            
            if ipar >= np.size(parsx):
                if nolsq:
                    print("%s - X: ########, %9.2e -- Y: #########, %9.2e" % \
                          (parsxf[ipar], parsyf[ipar], sind))
                else:
                    print("%s - X: ########, %9.2e, %9.2e -- Y: #########, %9.2e, %9.2e" % \
                          (sind, \
                           parsxf[ipar], \
                           parsxlsq[glsq], \
                           parsyf[ipar], \
                           parsylsq[glsq]))
                continue

            if ipar >= np.size(parsxf):
                if nolsq:
                    print("%s - X: %9.2e, ######## -- Y: %9.2e, ######## " % \
                          (sind, parsx[ipar], parsy[ipar]))
                else:
                    print("%s - X: %9.2e, ########,  ######## -- Y: %9.2e, ########, ######## " % (sind, parsx[ipar], parsy[ipar]) )
                continue

            if nolsq:
                print("%s - X: %9.2e, %9.2e -- Y: %9.2e, %9.2e" % \
                      (sind, \
                       parsx[ipar], parsxf[ipar], \
                       parsy[ipar], parsyf[ipar]))
            else:
                print("%s - X: %9.2e, %9.2e, %9.2e -- Y: %9.2e, %9.2e, %9.2e" % \
                      (sind,\
                       parsx[ipar], \
                       parsxf[ipar] - parsx[ipar], \
                       parsxlsq[glsq] - parsx[ipar], \
                       parsy[ipar], \
                       parsyf[ipar] - parsy[ipar], \
                       parsylsq[glsq]-parsy[ipar]))
                

def testmcmc_linear(npts=200, \
                    deg=2, degfit=-1, \
                    xmin=-1., xmax=1., ymin=-1., ymax=1., \
                    sigx=1e-4, sigy=7e-5, sigr=0.0, \
                    polytransf='Polynomial', polyfit=None, \
                    seed=None, expfac=1., scale=1.,\
                    covscale=1., \
                    unctysrc=True, \
                    unctytarg=True, \
                    nchains=-1, chainlen=20000, ntau=50, \
                    checknudge=False, \
                    samplefile='testmcmc.h5', \
                    doruns=False, \
                    domulti=False, \
                    addvar=False, \
                    extravar=5.0e-12, \
                    forgetcovars=False, \
                    guessextra=True, \
                    wtlsq=True, \
                    extra_is_corr=False, stdx_is_log=False, \
                    gen_noise_model=False, \
                    fit_noise_model=False, \
                    add_noise_model=False, \
                    noise_mag_pars=[-4., -26., 2.5], \
                    noise_shape_pars=[0.7, 0.1], \
                    extranoisepars = [-5.], \
                    extranoiseshape = [], \
                    cheat_guess=False, \
                    maglo=16., maghi=20., magexpon=2., \
                    guess_noise_mag=None, \
                    guess_noise_shape=None, \
                    nudgenoise = 0.01, \
                    nudgeshape = 0.01, \
                    useminimizer=False, \
                    minimizermethod='Nelder-Mead', \
                    minimizermaxit=3000, \
                    pathwritedata='test_simdata.xyxy', \
                    fbackg=-1., vxxbackg=-8., \
                    islog10fbackg=True, \
                    islog10vxxoutly=True, \
                    add_outliers=False, \
                    fit_outliers=False, \
                    debug_outliers=False):

    """Tests the MCMC approach on a linear transformation.

    set doruns=True to actually do the runs.

    addvar = 1D variance to add to the datapoints in target space. FOr
    testing, 5.0e-12 seems sensible (it's about 10x the covtran)

    extra_is_corr --> extra covariance is modeled internally as [stdx,
    stdy/stdx, corrcoef]

    stdx_is_log --> stdx is supplied as log10(stdx). Applies only if
    stdx_is_log is True.

    gen_noise_model --> generate uncertainties using noise model.

    fit_noise_model --> use the noise model to "fit" uncertainties

    add_noise_model --> use the noise model for the additive noise.

    noise_mag_pars [3] --> parameters of the magnitude dependence of
    the noise model stdx

    noise_shape_pars [2] --> noise model [stdy/stdx, corrxy].

    extranoisepars [<3] --> parameters of the noise model for added noise

    extranoiseshape [2] --> shape parameters of added noise. Defaults
    to symmetric noise.
    
    cheat_guess -- use the generating parameters as the guess.

    guess_noise_mag = None --> supplied initial guess for magnitude
    parameters of extra noise model

    guess_noise_shape = None --> supplied initial guess for shape
    parameters of extra noise model

    nudgenoise = factor by which to perturb guesses for the noise vs
    mag model

    nudgeshape = factory by which to perturb guesses for the shape
    noise model

    useminimizer = use scipy.optimize.minimize to refine the initial
    guess

    minimizermethod = if minimizing, the method for the minimizer to
    use. Recommended choices: 'TNC' or 'Nelder-Mead'

    minimizermaxit = maximum number of iterations for the minimizer

    pathwritedata = path to export simulated data

    fbackg = fraction of points that are "background,"
    i.e. outliers. Triggers mixture model if >0.

    vxxbackg = variance (in x, assumed symmetric) of background
    component.

    islog10fbackg = fbackg is supplied as log10

    islog10vxxoutly = vxx for outliers supplied as log10

    add_outliers = include outliers in the simulated data?

    fit_outliers = use mixture model when fitting?

    """

    # Use the same fit degree and basis as used to construct the data,
    # unless told otherwise.
    if degfit < 0:
        degfit = deg
        
    if polyfit is None:
        polyfit = polytransf[:]
        
    # What family of transformations are we using?
    transf = unctytwod.Poly

    # Generate positions and apparent magnitudes
    xy = makefakexy(npts, xmin, xmax, ymin, ymax)
    mags = makefakemags(npts, maglo=maglo, maghi=maghi, expon=magexpon)
    isoutly = np.repeat(False, npts) # which component. False = foreground  
    
    # Generate covariances in the observed frame. If gen_noise_model
    # is set, build the covariances using our magnitude-dependent
    # noise model
    if gen_noise_model:
        Cxy = makemagcovars(noise_mag_pars, mags, noise_shape_pars)
    else:
        Cxy = makeunifcovars(xy.shape[0], sigx, sigy, sigr)

    # sanity-check on the generated covariances
    if anycovbad(Cxy.covars):
        print("testmcmc_linear WARN: singular cov planes:", \
              np.sum(findcovsingular(Cxy.covars)) )
        return {}, {}, {}


    # Make fake parameters and propagated covariances:
    PM = Patternmatrix(deg, xy[:,0], xy[:,1], kind=polytransf, \
                       orderbypow=True)
    fpars = PM.getfakeparams(scale=scale, seed=seed, expfac=expfac)
    fparsx, fparsy = split1dpars(fpars)

    PTruth = transf(xy[:,0], xy[:,1], Cxy.covars, fparsx, fparsy, \
                    kind=polytransf)
    PTruth.propagate()
    
    xytran = np.copy(PTruth.xytran)
    covtran = np.copy(PTruth.covtran)*covscale 
    Ctran = CovarsNx2x2(covtran) # to draw samples

    # Set the perturbations in the observed and target frame, before
    # any additional noise
    nudgexy = xy * 0.
    nudgexytran = xytran * 0.
    nudgexyoutly = xytran * 0.
    
    if unctysrc:
        nudgexy = Cxy.getsamples()
    if unctytarg:
        nudgexytran = Ctran.getsamples()

    # Additional scatter to apply to outliers. The makeoutliers()
    # method already checks if fbackg >= 0, but we add the control
    # variable here.
    if add_outliers:
        nudgexyoutly, isoutly = makeoutliers(xytran, fbackg, vxxbackg, \
                                             islog10fbackg, \
                                             islog10vxxoutly)

        print("testmcmc_linear OUTLIERS:")
        print(np.min(nudgexyoutly, axis=0), \
              np.max(nudgexyoutly, axis=0))
        print(np.min(nudgexytran, axis=0), \
              np.max(nudgexytran, axis=0))

    # Some arguments that will be used if fitting the noise model and
    # ignored if not:
    npars_noise = 0
        
    # If we are adding more noise, do so here.
    nudgexyextra = xy * 0.
    npars_extravar = 0 # default, even if not used in the modeling

    # The two methods I have in mind are complicated but have
    # diverged, so in order to maintain readability, I use two
    # conditionals instead!
    CExtra = None # for later
    if addvar and not add_noise_model:

        # Parse the additional variance
        if np.isscalar(extravar):
            var_extra = np.array([extravar])
        else:
            var_extra = np.copy(extravar)

        # Number of extra parameters to include in the modeling
        npars_extravar = np.size(var_extra)

        # entries are: [stdx, stdy, corrxy]

        # At minimum we must have stdx; initialize off-diagonals too.
        stdx = np.repeat(np.sqrt(var_extra[0]), xy.shape[0])
        corrxy = stdx * 0.
        if np.size(var_extra) > 1:
            stdy = np.repeat(np.sqrt(var_extra[1]), xy.shape[0])

            # WATCHOUT - CovarsNx2x2 expects the correlation
            # coefficient, whereas we specify the actual off-diagonal
            # covariance. So:
            if np.size(var_extra) > 2:
                covoff = var_extra[2] / (stdx[0] * stdy[0])
                corrxy = np.repeat(covoff, xy.shape[0])
        else:
            stdy = np.copy(stdx)
        
        # Conditional because we might want to make this more complex later.
        CExtra = CovarsNx2x2(stdx=stdx, stdy=stdy, corrxy=corrxy)
        nudgexyextra = CExtra.getsamples()

        # For information, look at the extra covariance
        print("testmcmc_linear info - additional covariance:")
        print(CExtra.covars[0])
        print(npars_extravar)

    # If we want, we can add noise using the noise model.
    if addvar and add_noise_model:
        CExtra = makemagcovars(extranoisepars, mags, extranoiseshape)
        nudgexyextra = CExtra.getsamples()

    # Apply the nudges in the observed and target frame to the
    # positions.
    xyobs  = xy + nudgexy
    xytarg = xytran + nudgexytran + nudgexyextra + nudgexyoutly

    # At this point we have our perturbed data. Try writing it to disk
    if len(pathwritedata) > 3:
        writesimdata(pathwritedata, xyobs, xytarg, Cxy.covars, covtran, \
                     isoutly*1)
    
    # check the nudges
    if checknudge:
        print("Nudge DEBUG:")
        print(Cxy.covars[0], np.linalg.det(Cxy.covars[0]) )
        CC = np.cov(nudgexy, rowvar=False)
        print(CC, np.linalg.det(CC) )
        print("###")
        print(Ctran.covars[0], np.linalg.det(Ctran.covars[0]) )
        CD = np.cov(nudgexytran, rowvar=False)
        print(CD, np.linalg.det(CD))

    #### NOTE 2024-07-26 - the above syntax could all be refactored
    #### into a separate data-generation method. Come back to that
    #### later.

    # so that we don't lose the variable later on
    hinv = np.array([])

    if cheat_guess:
        guess = np.copy(fpars)
    else:
        # Since our model is linear, we can use linear least squares to
        # get an initial guess for the parameters.

        # Weight by the inverse of the covariances (which we trust to
        # all be nonsingular). 
        wts = np.ones(xyobs.shape[0])
        if wtlsq:
            # do not scale by det so that the formal covariance will
            # have meaning.
            wts = wtsfromcovars(covtran, scalebydet=False)        

            print("testmcmc_linear DEBUG: weights:", wts.shape)            
            detwts = np.linalg.det(wts)
            print("testmcmc_linear DEBUG: det(weights):", \
                  np.min(detwts), np.max(detwts), np.median(detwts) )
            
            
        LSQ = Leastsq2d(xyobs[:,0], xyobs[:,1], deg=degfit, w=wts, \
                        kind=polyfit, \
                        xytarg=xytarg)

        # formal estimate of parameter covariance
        hinv = np.linalg.inv(LSQ.H)
        print("testmcmc_linear INFO - inverse of lsstsq hessian:")
        print(hinv.shape)
        
        guess = LSQ.pars # We may want to modify or abut the guess.

    guessx, guessy = split1dpars(guess)

    # Now we arrange things for our mcmc exploration. The
    # transformation object...
    covsrc = Cxy.covars
    if forgetcovars or not unctysrc:
        covsrc *= 0.
    PFit = transf(xyobs[:,0], xyobs[:,1], covsrc, guessx, guessy, \
                  kind=polyfit)
    PFit.propagate()

    # Take a look at the data we generated... do these look
    # reasonable? [Refactored into new method]
    #
    # 2024-08-12 Moved this farther down so that we can look at the
    # results of the initial guess.
    
    # showsimxy(xy, xyobs, xytran, xytarg, covtran, mags, \
    #          CExtra, PFit, PTruth, fignum=1, isoutly=isoutly)


    # Set up labeling for plots.
    slabels = labelstransf(PFit.pars2x, 'A', 'B')

    # Set the guess offset scale for the fit parameters
    if not cheat_guess:
        scaleguess = scalexyguess(guess, fpars)
    else:
        scaleguess = scalexyguess(fpars, fpars)


    # Ensure the truths array has the same order (and length!) as the
    # guess array.
    truths = padxytruths(fpars, guess)
        
    # Try adjusting the guess scale to cover the offset between our
    # generated parameters and our guess, but not to swamp it. We do
    # this BEFORE we abut any additional noise model parameters onto
    # the guess.
    #if not cheat_guess:
    #    scaleguess = np.abs((guess-fpars)/guess)
    #else:
    #    scaleguess = 1.0e-3

    # If we want to fit for additional variance, ensure the guesses
    # etc. are structured so that the MCMC can use them. For the older
    # method, the simulating and fitting used the same structure, so
    # there is some ad hoc structure here. I therefore do two separate
    # conditionals again.
        
    # If we added 1d variance, accommodate this here.
    if addvar and not fit_noise_model:

        lextra = labelsaddvar(npars_extravar, extra_is_corr, stdx_is_log)
        slabels = slabels + lextra
                    
        # Come up with a guess for the added variance. For the moment
        # use a relatively soft test, where we know the truth going
        # in... Make these vectors from the start so that we can
        # smoothly adjust later.
        vguess = np.random.uniform(low=0.8, high=1.2, \
                                   size=np.size(var_extra)) \
                                   * var_extra

        # Pull the initial guess away harder
        # vguess *= 0.01

        # Now try pretending we don't know the covariances in either
        # frame, leaving *all* the covariance to the model. In that
        # case, a more sensible guess is probably the residuals after
        # fitting. So
        if forgetcovars:
            covtran *= 0.  # is passed to the sampler

            # Ensure the pfit object has forgotten the
            # model covariance
            PFit.covxy *= 0.
            PFit.covtran *= 0.

        # This may get spun out into a separate method
        if guessextra or forgetcovars:
            
            # Estimate the covariance after applying the initial-guess
            # transformation
            gxy = PFit.xytran - xytarg
            cg = np.cov(gxy, rowvar=False)

            # guess the extra covariance
            covguess = cg - covtran - PFit.covtran
            cf = np.mean(covguess, axis=0)

            # We need to ensure that our guess has the right number of
            # entries. If we're assuming scalar or 1d input variance,
            # we only want the first entry.  Here's one way to do the
            # ordering - it's a little stupid, but it works.
            lx = np.array([0,1,0])
            ly = np.array([0,1,1])
            vguess = cf[lx, ly][0:npars_extravar]
            
            # vguess = np.array([cf[0,0], cf[1,1], cf[0,1]] )
            
            # Now slot in the guesses depending on how many covariance
            # guess parameters we want
            print("testmcmc_linear INFO - mean excess covariance:")
            print(np.mean(covguess, axis=0))
            
            #vguess = np.array([cg[0,0], cg[1,1], cg[0,1]])
            print("testmcmc_linear INFO - Initial vars guess:")
            print("testmcmc_linear INFO - ", vguess)

        # If extra covariance will be explored as [sx, sy/sx, rho],
        # update the entries accordingly.
        if extra_is_corr:

            # WATCHOUT - cov2corr1d returns a 3-element array even if
            # <3 entries were supplied. This will mess things up
            # downstream since the labels array knows how many
            # elements were given. So, we enforce the array length
            # here.            
            vguess = cov2corr1d(vguess)[0:npars_extravar]
            var_extra = cov2corr1d(var_extra)[0:npars_extravar]

            # Is the stdx to be explored as log10(stdx)? NOTE the
            # np.abs() is there as a safety in case the guess from
            # data produces a negative stdx...
            sdum = 'stdx'
            if stdx_is_log:
                vguess[0] = np.log10(np.abs(vguess[0]))
                var_extra[0] = np.log10(np.abs(var_extra[0]))
                sdum = 'log10(stdx)'
                
            print("testmcmc_linear INFO -  re-expressed vguess as [%s, stdy/stdx, corrcoef]:" % (sdum))
            print("testmcmc_linear INFO - ", vguess)
            
        # Ensure the "truth" and guess parameters have the right
        # dimensions
        guess = np.hstack(( guess, vguess ))
        fpars = np.hstack(( fpars, var_extra )) 

        scaleguess = np.hstack(( scaleguess, \
                                 np.repeat(0.01, np.size(vguess)) ))

        # Same for the truths array - which now has the same ordering
        # as the guess array even if the degrees do not match:
        truths = np.hstack(( truths, var_extra ))
        
    # Are we going to be fitting with the more general noise model?
    if fit_noise_model:

        # Initial guess from perturbing the simulated parameters OR
        # from initial guess supplied when setting this up
        guessnoise, guessshape, scalesnoise, scalesshape = \
            assignnoiseguess(guess_noise_mag, \
                             guess_noise_shape, \
                             extranoisepars, extranoiseshape, \
                             nudgenoise, nudgeshape, 5.)

        scaleguess = np.hstack(( scaleguess, scalesnoise, scalesshape ))

        # Ensure the "truth" arrays for the noise model make sense as
        # guesses
        # truthsnoise = padtruths(extranoisepars, guessnoise)
        # truthsshape = padtruths(extranoiseshape, guessshape)
        
        # Abut onto our 1D parameter sets for guesses
        npars_noise = np.size(guessnoise)
        npars_extravar = np.size(guessshape)
        labels_extra = labelsnoisemodel(npars_noise, npars_extravar)

        slabels = slabels + labels_extra
        guess = np.hstack(( guess, guessshape, guessnoise ))
        fpars = np.hstack(( fpars, extranoiseshape, extranoisepars ))

        # Ensure our 1D master parameter array has the truth entries
        # in the right places
        truthsnoise = padtruths(extranoisepars, guessnoise)
        truthsshape = padtruths(extranoiseshape, guessshape)
        truths = np.hstack(( truths, truthsnoise, truthsshape ))

    # Ensure the number of "truth" parameters for additional noise are
    # recorded, they will be needed later. WARNING - this will need to
    # be updated if we're going to keep our older noise model.
    ntruths_extranoise = np.size(extranoisepars)
    ntruths_extrashape = np.size(extranoiseshape)

    # are we going to try fitting a mixture model?
    npars_mix = 0
    if fit_outliers:

        # while testing if this actually functions, abut the
        # generating mix parameters to the truths and to the
        # guess. Don't forget that the lnprob uses the FOREGROUND
        # fraction and not the background fraction. This is awkward:
        # consider making this uniform between the generator and MCMC
        # (so that the MCMC uses the background fraction instead of
        # the foreground fraction).
        fbg = parsefraction(fbackg, islog10fbackg)
        ffg = 1.0 - fbg
        if islog10fbackg:
            ffg = np.log10(ffg)

        truthmix = np.array([ffg, vxxbackg])

        npars_mix = np.size(truthmix)
        
        # stack onto truths
        truths = np.hstack(( truths, truthmix ))

        # stack onto fpars (passed into padpars)
        fpars = np.hstack(( fpars, truthmix ))
        
        # stack onto guess
        guess = np.hstack(( guess, truthmix ))

        # stack onto scaleguess
        smguess = np.abs(truthmix) * 0.01
        scaleguess = np.hstack(( scaleguess, smguess ))

        # append plot labels onto the end
        lmix = [r'$log_{10}(f_{fg})$', r'$v_{bg}$']
        slabels = slabels + lmix
        
        print("TEST OUTLIERS DEBUG:")
        print(fbackg, islog10fbackg, fbg, ffg, 10.0**ffg)
        # return {}, {}, {}
    
    # now (drumroll) set up the sampler.
    methpost = lnprob
    args = (PFit, xytarg, covtran, addvar, npars_extravar, \
            extra_is_corr, stdx_is_log, npars_noise, mags, \
            npars_mix, islog10fbackg, islog10vxxoutly)
    ndim = np.size(guess)

    # Use scipy minimizer to refine the initial guess?
    if useminimizer:
        print("testmcmc_linear INFO - refinining initial guess w/optimizer")
        print("testmcmc_linear INFO - method (maxiter:%i): %s ..." \
              % (minimizermaxit, minimizermethod))

        # Try specifying the maximum number of iterations
        options={'maxiter':minimizermaxit}
        if minimizermethod.find('TNC') > -1:
            options={'maxfun':minimizermaxit}
        
        tm0 = time.time()
        ufunc = lambda *args: -np.sum(methpost(*args))
        soln = minimize(ufunc, guess, args=args, \
                        method=minimizermethod, \
                        options=options)
        tm1 = time.time()

        # If the minimizer failed, return and show.
        if not soln.success:
            print("testmcmc_linear WARN - minimizer success flag:", \
                  soln.success)
            print(soln)
            return {}, {}, {}

        print("testmcmc_linear INFO - ... done in %.2e seconds." \
              % (tm1 - tm0) )

        print("testmcmc_linear INFO - pre-minimizer guess:")
        print(guess)
        
        # now update our guess and truths array accordingly.
        guess = np.copy(soln.x)
        scaleguess, truths = padpars(fpars, guess, \
                                     ntruths_extranoise, \
                                     ntruths_extrashape, \
                                     npars_noise, \
                                     npars_extravar, \
                                     npars_mix, \
                                     npars_mix)

    print("testmcmc_linear INFO - fpars:")
    print(fpars)

    print("testmcmc_linear INFO - truths:")
    print(truths)
    
    print("testmcmc_linear INFO - guess:")
    print(guess)

    print("testmcmc_linear INFO - |fractional offset| in guess:")
    print(scaleguess)
    print("^^^^^^")
    # with fake data, those are all VERY small - like 1e-7 to 1e-6
    # off. So our scaling is enormous. Try bringing the guessing way
    # down then.
    ##  scaleguess *= 5. # scale up so we cover the interval and a bit more

    # consider doing one per component, it should work without adjustment.
    
    # adjust the nchains to match ndim
    if nchains < 1:
        nchains = int(ndim*2)+2
        print("testmcmc_linear - scaling nchains from ndim to %i" % (nchains))
    
    # set up the walkers, each with perturbed guesses
    pertn = np.random.randn(nchains, np.size(guess))
    magn  = scaleguess * guess  # was 0.01
    pos = guess + pertn * magn[np.newaxis,:]
    nwalkers, ndim = pos.shape

    print("INFO: pos", pos.shape)
    print("nwalkers, ndim", nwalkers, ndim)

    # WATCHOUT - only uncomment this if you want to use emcee's blobs
    # feature.
    # 
    ## Use emcee's "blobs" feature to track mixture assignment
    ## information. Look into supplying keyword arguments rather than
    ## by-order!
    #args = (PFit, xytarg, covtran, addvar, npars_extravar, \
    #        extra_is_corr, stdx_is_log, npars_noise, mags, True)
    
    ## send the iniital guess through the ln(prob) to test whether it
    ## returns sensible values
    #check, llfg, llbg = methpost(guess, *args)
    #print("testmcmc_linear DEBUG - ln(prob) on initial guess:", check)
    #print("testmcmc_linear DEBUG -  blob info shape:", np.shape(llfg), np.shape(llbg))

    # Since we don't see the plot until we get the interpreter back,
    # we may as well show the samples here. The advantage is that we
    # now have the modified guess transformation so we can check
    # it. So - unpack the guess parameters back into pfit parameters
    # and pass them to the sim plotter:
    pguess, _, _, _ = splitmodel(guess, npars_extravar, \
                                 npars_noise, npars_mix)
    PFit.updatetransf(pguess)
    
    showsimxy(xy, xyobs, xytran, xytarg, covtran, mags, \
              CExtra, PFit, PTruth, fignum=1, isoutly=isoutly)

    
    
    check = methpost(guess, *args)
    print("testmcmc_linear DEBUG - ln(prob) on initial guess:", check)
    
    if np.isnan(check):
        print("testmcmc_linear FATAL - initial guess returns nan. Check it!")
        if not domulti:
            return
        
        return {}, {}, {}
    
    # set up the backend to save the samples
    if os.access(samplefile, os.R_OK):
        os.remove(samplefile)
    backend = emcee.backends.HDFBackend(samplefile)
    backend.reset(nwalkers, ndim)

    if not doruns:
        print("testmcmc_linear INFO - look at the data, then rerun setting doruns=True.")
        print(fpars)
        print(fpars.size)
        
        return

    # 2024-08-09 if we want to preserve esargs (or maybe just the args
    # and log_prob_fn), might consider pickling that here. Try it:

    argskeep = {'args':args, 'log_prob_fn':methpost}
    with open('test_argskeep.pickle', 'wb') as wobj:
        pickle.dump(argskeep, wobj)
    
    # Now we set the arguments for the sampler and for the plotter, so
    # that we can call them from the interpreter if needed
    esargs = {'nwalkers':nwalkers, 'ndim':ndim, 'log_prob_fn':methpost, \
              'args':args, 'backend':backend}

    runargs = {'initial_state':pos, 'nsteps':chainlen, 'progress':True}
    
    showargs = {'slabels':slabels, 'ntau':ntau, 'fpars':fpars, \
                'truths':truths, \
                'guess':guess, 'basis':PFit.kind, 'lsq_hessian_inv':hinv, \
                'basis_gen':PTruth.kind, \
                'degree_gen':PTruth.pars2x.deg, \
                'degree':PFit.pars2x.deg}
    
    # if multiprocessing, then we'll want to run from the python
    # interpreter.
    if domulti:

        # Could wrap the returns into an object for clarity?

        # Watchout - the backend may need to be set at the
        # interpreter. Test this!
        print("Returning arguments for multiprocessing:")
        print("esargs, runargs, showargs")

        print("Now execute:")
        print("with Pool() as pool:")
        print("      sampler = emcee.EnsembleSampler(**esargs, pool=pool)")
        print("      sampler.run_mcmc(**runargs)")
        print("      fittwod.showsamples(sampler, **showargs")
        return esargs, runargs, showargs

    # Run without multiprocessing
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, \
    #                                methpost, \
    #                                args=args, \
    #                                backend=backend)

    sampler = emcee.EnsembleSampler(**esargs)
    
    t0 = time.time()
    # sampler.run_mcmc(pos, chainlen, progress=True);
    sampler.run_mcmc(**runargs);

    t1 = time.time()
        
    print("testmcmc INFO - samples took %.2e seconds" % (t1 - t0))

    # samples = sampler.get_chain()
    #showsamples(sampler, slabels, ntau, fpars, guess)
    showsamples(sampler, **showargs)

    
    
def writesimdata(pathout='test_simdata.dat', \
                 xyobs=np.array([]), xytarg=np.array([]), \
                 covsrc=np.array([]), covtran=np.array([]), \
                 mismatch=np.array([]) ):

    """Writes simulated data pairs to disk. Output columns:

    xobs yobs xtarg ytarg vxxobs vyyobs vxyobs vxxtarg vyytarg vxytarg
    mismatch

    with missing values filled in with zeros.

    Parsing is rather rudimentary: currently the shape of the xyobs
    array is always used to determine how long the output should be.

Inputs:

    pathout = path to output file.

    xyobs = [N,2] array of x, y positions in the source frame

    xytarg = [N,2] array of positions in the target frame

    covsrc = [N,2,2] array of covariances in the source frame

    covtarg = [N,2,2] array of covariances in the target frame

    mismatch = [N] array (1 if mismatch, 0 otherwise)

Returns:

    No return

    """

    # Nothing to do if no valid path given
    if len(pathout) < 3:
        return

    if pathout.find('.') < 0:
        return

    # use xyobs to set the expected size of the output. (Note:
    # np.ndim(np.array([])) = 1.)
    if np.ndim(xyobs) < 2:
        print("writesimdata FATAL - xyobs must be 2D array.")
        return
    nrows = np.shape(xyobs)[0]

    # we put all the input into a single 2D array for output by
    # numpy's savetxt method. 
    aout = np.copy(xyobs)

    # Now we stack on each provided quantity in turn. First the xy
    # positions in the target frame:
    add_xytarg = np.zeros((nrows, 2))
    if np.ndim(xytarg) is 2:
        if np.shape(xytarg)[0] is nrows:
            add_xytarg = xytarg

    # Then the covariance components in the source frame:
    add_covsrc = np.zeros((nrows, 3))
    if np.ndim(covsrc) is 3:
        if np.shape(covsrc)[0] is nrows:
            add_covsrc = covn222cov3(covsrc)

    # then the covariance in the target frame:
    add_covtran = np.zeros((nrows, 3))
    if np.ndim(covtran) is 3:
        if np.shape(covtran)[0] is nrows:
            add_covtran = covn222cov3(covtran)

    # Finally a mismatches flag
    add_mismatch = np.zeros((nrows))
    if np.size(mismatch) is nrows:
        add_mismatch = mismatch

    # stack all at once
    aout = np.column_stack((aout, add_xytarg, \
                            add_covsrc, add_covtran, \
                            add_mismatch))

    # This is only going to be read from within this module, so we
    # don't need to make the output too flexible. For the moment, just
    # output a string with the column names to the header.
    sheader = 'xobs yobs xtarg ytarg vxxobs vyyobs vxyobs vxxtran vyytran vxytran mismatch'
    np.savetxt(pathout, aout, header=sheader)
    
def loadsimdata(pathdata=''):

    """Loads simulation data: positions and covariances in source and
target frames, along with a mismatch flag.

Input:

    pathdata = path to data file

Returns:

    xyobs = [N,2] -- x y positions, observed frame

    xytran = [N,2] -- x,y positions, target frame

    covsrc = [N,2,2] - covariances in observed frame

    covtran = [N,2,2] -- covariances in target frame
 
    mismatch = [N] - 1/0 for mismatch

    """

    blank = np.array([])
    if len(pathdata) < 4:
        return blank, blank, blank, blank, blank

    if not os.access(pathdata, os.R_OK):
        return blank, blank, blank, blank, blank

    adata = np.genfromtxt(pathdata, unpack=False)

    # No parsing is currently done: if the input file doesn't match
    # what we expect - if only partial information is given - then we
    # want this to fail.
    
    # Now we partition this across the pieces. 
    xyobs = adata[:,0:2]
    xytran = adata[:,2:4]
    covsrc  = covn32covn22(adata[:,4:7])
    covtran = covn32covn22(adata[:,7:10]) 
    mismatch = np.asarray(adata[:,-1], 'int')

    return xyobs, xytran, covsrc, covtran, mismatch
    
def showsimxy(xy=np.array([]), xyobs=np.array([]), \
              xytran=np.array([]), xytarg=np.array([]), \
              covtran=np.array([]), \
              mags=np.array([]), CExtra=None, \
              PFit=None, PTruth=None, \
              fignum=1, isoutly=np.array([]), \
              showguess=True, \
              scattarcsec=True):

    """Show some characteristics of the generated data"""

    # refactored out of testmcmc_linear to try to clean up that method

    fig1 = plt.figure(fignum, figsize=(9.5,5))
    fig1.clf()
    ax1=fig1.add_subplot(231)
    ax2=fig1.add_subplot(232)

    if showguess:
        ax3=fig1.add_subplot(235)
        ax4=fig1.add_subplot(234)
    else:
        ax3=fig1.add_subplot(234)
        ax4=fig1.add_subplot(235)
        
    ax5 = None # now set below
    ax6 = None # now set in a conditional below
    
    fig1.subplots_adjust(wspace=0.4, hspace=0.4, \
                         left=0.15, bottom=0.15)

    # generated and perturbed points in original and target frame:
    blah1=ax1.scatter(xy[:,0], xy[:,1], s=1)
    blah2=ax2.scatter(xyobs[:,0], xyobs[:,1], c='g', s=1)
    blah4=ax4.scatter(xytarg[:,0], xytarg[:,1], c='g', s=1)

    # If we're not showing the evaluation of the guess on the
    # PERTURBED datapoints, plot the scatterplot of the transformed
    # objects
    if not showguess:
        blah3=ax3.scatter(xytran[:,0], xytran[:,1], s=1)
        ax3.set_title('Transformed')
        

    # Set titles for the first four axes
    ax1.set_title('Generated')
    ax2.set_title('Perturbed')
    ax4.set_title('Target')

    # Labels for the position plots
    for ax in [ax1, ax2]:
        ax.set_xlabel(r'X')
        ax.set_ylabel(r'Y')

    for ax in [ax3, ax4]:
        ax.set_xlabel(r'$\xi$')
        ax.set_ylabel(r'$\eta$')

    # membership color if we have that information
    cscatt='k'
    shademembs = False
    if np.sum(isoutly) > 0:
        cscatt = isoutly * 1.
        shademembs = True

    # Conversion factor for any scatterplots of deltas
    conv = 1.
    sunit = '' # axis labels units for residuals
    if scattarcsec:
        conv=3600.
        sunit = '(arcsec)'
        
    # If a fit object has been sent in, use it to show the transformed
    # positions using the initial guess (or whatever parameters are in
    # the pfit object):
    if PFit is not None:
        blah5 = ax4.scatter(PFit.xytran[:,0], PFit.xytran[:,1], \
                            c='r', s=1)

        # If here, we are showing what happens when we transform the
        # xy using the parameters in PFit:
        if showguess:
            PFit.propagate()
            dxy = (PFit.xytran - xytarg)*conv
            blah3 = ax3.scatter(dxy[:,0], dxy[:,1], s=9, \
                            c=cscatt, cmap='Greens_r', \
                            edgecolor='k')

            ax3.set_title('guess pars')
            ax3.set_aspect('equal', adjustable='box')
            
    # Show the residuals from truth in the target space on axis 5
    if PTruth is not None:
        ax5=fig1.add_subplot(236)
        fxy = (PTruth.xytran - xytarg)*conv

        # If we have outliers, make a choice about what range to use.
        shadeax5 = False
        cc = np.cov(fxy, rowvar=False)

        if np.sum(isoutly) > 0:
            # cscatt = isoutly * 1.
            # shademembs = True
            ccinly = np.cov(fxy[~isoutly], rowvar=False)
            
        blah5 = ax5.scatter(fxy[:,0], fxy[:,1], s=9, \
                            c=cscatt, cmap='Greens_r', \
                            edgecolor='k')
        sanno = "%.2e, %.2e, %.2e" % (cc[0,0], cc[1,1], cc[0,1])
        anno5 = ax5.annotate(sanno, (0.05,0.05), \
                             xycoords='axes fraction', \
                             ha='left', va='bottom', fontsize=6)

        if shademembs:
            sanno2 = "%.2e, %.2e, %.2e" % (ccinly[0,0], ccinly[1,1], ccinly[0,1])
            anno52 = ax5.annotate(sanno2, (0.05,0.15), \
                                  xycoords='axes fraction', \
                                  ha='left', va='bottom', fontsize=6, color='g')
            
            
        # Enforce aspect ratio for residuals plot(s)
        ax5.set_aspect('equal', adjustable='box')        
        
        # Some cosmetics
        ax5.set_title('Residuals, generated')
            
        axdelts = [ax5]
        if showguess:
            axdelts=[ax5, ax3]

        for ax in axdelts:
            ax.set_xlabel(r'$\Delta \xi$ %s' % (sunit)) # sunit set above
            ax.set_ylabel(r'$\Delta \eta$ %s' % (sunit))

        # Add a colorbar if we have shade information for our scatter
        if shademembs:
            cbar5 = fig1.colorbar(blah5, ax=ax5)
            
            if showguess:
                cbar3 = fig1.colorbar(blah3, ax=ax3)
            
    # If we have apparent magnitudes, compute and show the uncertainty
    # wrt magnitude
    if np.size(mags) > 0:
        
        ax6=fig1.add_subplot(233)
        covres = covtran + covtran
        dettran = np.linalg.det(covres)**0.25
        blah61=ax6.scatter(mags, dettran, s=2, \
                           label='Covtran', c='b', zorder=2)

        # Do we have covariance information for the output frame?
        if CExtra is not None:
            covext = CExtra.covars
            detext = np.linalg.det(covext)**0.25
            detsho = np.linalg.det(covres + covext)**0.25
            blah62=ax6.scatter(mags, detext, s=2, label='Added', \
                               c='r', zorder=4)
            blah63=ax6.scatter(mags, detsho, s=2, label='Total', \
                               c='k', zorder=5)

        # ax6 legend is only meaningful if we have magnitudes...
        leg6 = ax6.legend(fontsize=8)
        ax6.set_yscale('log')
        ax6.set_title(r'Magnitude $m$')
        ax6.set_xlabel(r'$m$')
        ax6.set_ylabel(r'$|V_{\xi\eta}|^{1/4}$')
        
def showsamples(sampler, slabels=[], ntau=10, fpars=np.array([]), \
                guess=np.array([]), \
                basis='', basis_gen='', \
                truths=None, \
                flatfile='test_flatsamples.npy', \
                argsfile='test_flatsamples.pickle', \
                filfig3='test_thinned.png', \
                filfig2='test_allsamp.png', \
                filfig4='test_corner.png', \
                nminclose=20, burnin=-1, \
                lsq_hessian_inv=np.array([]), \
                degree=-1, degree_gen=-1):

    """Ported the methods to use the samples into a separate method so
that we can run this from the interpreter."""

    # Might be better to have this work on samples so that they could
    # be read in from disk. Not sure if sampler() is serialized.
    
    # look at the results
    samples = sampler.get_chain()
    
    print("SAMPLES INFO - SAMPLES:", np.shape(samples))

    # Plot the unthinned samples
    fig2 = plotsamplescolumn(samples, 2, slabels=slabels)
    fig2.savefig(filfig2)

    # close the figure if we have more than, say, 20 params
    if samples.shape[-1] > nminclose:
        plt.close(fig2)
    
    
    # get the autocorrelation time
    try:
        tau = sampler.get_autocorr_time()
        tauto = tau.max()
        print("testmcmc_linear info: autocorrelation time:", tauto)
        ntau = int(ntau*0.5) # maybe a little risky? When this works,
                             # tau typically comes out to about 80.
    except:
        print("testmcmc_linear warn: long autocorrelation time")
        tauto = 50

    nThrow = int(tauto * ntau)
    nThin = int(tauto * 0.5)

    # allow overriding nthrown with burnin
    if burnin > 0:
        nThrow = np.copy(burnin)
    
    flat_samples = sampler.get_chain(discard=nThrow, thin=nThin, flat=True)
    print("FLAT SAMPLES INFO:", flat_samples.shape, nThrow, nThin)

    # ^^^ This is the important part. We now have our flat
    # samples. These should be written to disk or returned to do
    # analysis on. Using np.save because that's supposed to work well
    # for multidimensional numpy arrays. Need to think a bit on how to
    # handle metadata.
    np.save(flatfile, flat_samples)

    # determine the covariance between the parameters
    parscov = np.cov(flat_samples, rowvar=False)
    print("fittwod.showsamples INFO - parameter covariance:", parscov.shape)
    
    # Useful to save run information. For the moment let's just use
    # the labels while I work out what info and how to send...
    with open(argsfile, 'wb') as wobj:
        dwrite={'slabels':slabels, 'fpars':fpars,'guess':guess, \
                'basis':basis, \
                'truths':truths, \
                'covpars':parscov, \
                'lsq_hessian_inv':lsq_hessian_inv}
        pickle.dump(dwrite, wobj)
    
    fig3 = plotsamplescolumn(flat_samples, 3, slabels=slabels)
    fig3.savefig(filfig3)
    if flat_samples.shape[-1] > nminclose:
        plt.close(fig3)

    
    # Try a corner plot
    fig4 = plt.figure(4, figsize=(9,7))
    fig4.clf()
    dum4 = corner.corner(flat_samples, labels=slabels, truths=truths, \
                         truth_color='b', fig=fig4, labelpad=0.7, \
                         use_math_text=True, \
                         label_kwargs={'fontsize':8, \
                                       'rotation':'horizontal'})
    fig4.subplots_adjust(bottom=0.2, left=0.2, top=0.95)

    # Try adjusting the label size externally:
    for ax in fig4.get_axes():
        ax.tick_params(axis='both', labelsize=5)
    
    # Construct supertitle from generation and fit information
    ssup=''
    if len(basis_gen) > 0:
        ssup = 'Generated: %s' % (basis_gen)
    if degree_gen > -1:
        ssup = '%s (degree %i)' % (ssup, degree_gen)
    if len(basis) > 0:
        spad = ''
        if len(ssup) > 0:
            spad = ' -- '
        ssup = '%s%s Fit: %s' % (ssup, spad, basis)
    if degree > -1:
        ssup = '%s (degree %i)' % (ssup, degree)

    if len(ssup) > 0:
        fig4.suptitle(ssup)
        
    fig4.savefig(filfig4)

    # if lots of figure panels, close the figure
    if flat_samples.shape[-1] > nminclose:
        plt.close(fig4)
        
    print("INFO: generated parameters:")
    print(fpars)
    print("INFO: lsq parameters")
    print(guess)
    print("INFO: 50th pctile mcmc samples")
    print(np.percentile(flat_samples, 50., axis=0))

    print("INFO: 16th, 84th pctile MCMC:")
    print(np.percentile(flat_samples, 16., axis=0))
    print(np.percentile(flat_samples, 84., axis=0))
    
    return

    tau = sampler.get_autocorr_time()
    tauto = tau.max()
    nThrow = int(tauto * ntau)
    nThin = int(tauto * 0.5)
    flat_samples = sampler.get_chain(discard=nThrow, thin=nThin, flat=True)

    print("SAMPLES INFO - FLAT:", np.shape(flat_samples))

def getrunsamples(samples=np.array([]), \
                  pathsamples='test_flatsamples.npy', \
                  Verbose=True):

    """Gets run samples and run information.

Inputs:

    samples = np.array of run samples.

    pathsamples = path to samples if samples not supplied.

    Verbose = print screen output if there is a problem loading the path

Returns:

    samples = np.array of run samples

"""

    if np.size(samples) > 0:
        flatsamples = samples
    else:
        try:
            flatsamples = np.load(pathsamples)
        except:
            if Verbose:
                print("getrunsamples FATAL - problem loading flat samples from %s" \
                      % (pathsamples))
            return np.array([])

    return flatsamples

def runargsok(runargs={}, lkeys=['args'], Verbose=False ):

    """Checks if run arguments dictionary contains the needed quantities

Inputs:
    
    runargs = {} dictionary of run arguments

    lkeys = list of keywords that must be present

    Verbose = print screen output if any keywords are missing

Returns:

    argsok = [T/F] - all the requested items are present

"""

    runkeys = runargs.keys()
    if len(runkeys) < 1:
        if Verbose:
            print("checkrunargs WARN - no keywords.")
        return False

    for skey in lkeys:
        if not skey in runkeys:
            if Verbose:
                print("checkrunargs WARN - keyword missing: %s" % (skey))
            return False

    # If we reached here, all the required keys are present.
    return True

def computeprojs(samples=np.array([]), pathsamples='test_flatsamples.npy', \
                 runargs={}, keyargs='args', \
                 xy=np.array([]), \
                 samplesize=-1, \
                 ireport = 1000, \
                 pathprojs='test_flatproj.npy'):

    """Computes projection of input data onto the target space, using the sample of transformation parameters. 

Inputs:

    samples = array of MCMC samples.

    pathsamples = path to file holding MCMC samples. Ignored if samples has nonzero size.

    runargs = dictionary of MCMC run arguments. Must include the
    transformation object used. Currently this structure is not
    terribly transparent.

    keyargs = keyword in runarguments dictionary for the actual run arguments.

    xy = [N,2] array of positions to project. If not supplied, these
    are taken from the transformation object.

    samplesize = number of samples to keep

    ireport = report progress every ireport samples

    pathprojs = file path to write the resulting samples.

Returns:
    
    projxy = [samplesize, 2, N] array of positions.

    """

    # initialize the output to blank
    projxy = np.array([])
    
    # Ensure we have the necessary pieces
    if not runargsok(runargs, [keyargs], Verbose=True):
        print("computeprojs WARN - problem with runargs keywords")
        return projxy

    # Check that the samples are OK
    flatsamples = getrunsamples(samples, pathsamples)
    if np.size(flatsamples) < 1:
        print("computeprojs FATAL - no samples")
        return projxy

    # OK now identify the transformation object. We make a copy
    # because we may be changing things.
    transf = copy.deepcopy(runargs[keyargs][0])

    # we will need to lift out the geometric parameters. To do this,
    # we appeal to the arguments, which are currently not done very
    # transparently...
    argus = runargs[keyargs]
    nnoise_shape = argus[4]
    nnoise_model = argus[7]
    nmix = argus[9]
    
    # If xy positions were supplied, switch them in to the
    # transformation object.
    if np.size(xy) > 1:
        if np.ndim(xy) is 2:
            transf.x = xy[:,0]
            transf.y = xy[:,1]

    # if here then we should have our transformation object ready to
    # go. Since feval() propagates both the positions and the
    # covariances and we want only the covariances, use the
    # self.tranpos() method of the transformation object instead.

    # Convenience views
    nsamples = flatsamples.shape[0]
    ndata = transf.x.size
    
    # Set up the projections array
    imax = samplesize
    if samplesize < 0 or samplesize > nsamples:
        imax = nsamples
    projxy = np.zeros((imax, 2, ndata)) 

    # Split the model parameters as vectors. This actually only buys a
    # small speed improvement, I wonder if this is because the native
    # parallelization was overridden in favor of running the samples.
    ptrans2d, _, _, _ = splitmodel(flatsamples.T, \
                                   nnoise_model, nnoise_shape, nmix)
    ptrans2d = ptrans2d.T
        
    # OK here we go...
    t0 = time.time()
    tremain = 0.
    for isample in range(imax):

        # Update the transformation object's parameters
        #ptransf, _, _, _ = splitmodel(flatsamples[isample], \
        #                              nnoise_model, nnoise_shape, nmix)
        transf.updatetransf(ptrans2d[isample])

        # Apply the transformation and slot into the master array
        transf.tranpos()
        projxy[isample,0] = transf.xtran
        projxy[isample,1] = transf.ytran

        if isample % ireport < 1 and isample > 0:
            telapsed = time.time() - t0
            itpersec = float(isample)/telapsed
            if itpersec > 0.:
                tremain = float(imax)/itpersec
            print("computeprojs INFO - iteration %i of %i after %.2e seconds: %.1f it/sec. Est %.2e sec remain" \
                  % (isample, imax, telapsed, itpersec, tremain), end='\r') 

    print("")
    print("computeprojs INFO - %i loops finished in %.2e sec" % (imax, time.time() - t0) )

    # Write the projections to file
    if len(pathprojs) > 3:
        if pathprojs.find('.') > 0:
            np.save(pathprojs, projxy)
    
    return projxy
    
def computeresps(samples=np.array([]), \
                 pathsamples='test_flatsamples.npy', \
                 runargs={}, \
                 keyargs='args', keyfunc='log_prob_fn', \
                 pathprobs='test_postmaster.npy', \
                 samplesize=-1, \
                 ireport = 1000, \
                 keepmaster = True, \
                 writemaster = True, \
                 returnmaster = True):

    """Given a set of flat samples, computes the (log) responsibilities
and estimates probabilities of foreground/background association.

Example call (after running the emcee sampler):

    psum, postmaster = fittwod.computeresps(runargs=esargs)

    psum = fittwod.computeresps(runargs=esargs, returnmaster=False, keepmaster=False)


Inputs:

    samples = np.array([])
    
    pathsamples = path to the flattened samples, if loading from file

    runargs = {} dictionary of run arguments that were supplied to
    emcee

    keyargs = keyword in runargs that holds the arguments passed to
    log prob function

    keyfunc = keyword in runargs that gives the log prob function
    itself

    pathprobs = path to write the estimated probabilities

    samplesize = how many rows to compute (since this takes a
    while...). If <1, defaults to all the samples.

    ireport = report progress to screen every ireport iterations.

    keepmaster = store and return the array of posterior probability
    samples per datapoint. If False, an empty array is returned for
    postmaster (but not for psum!) and the postmaster array is not
    written to disk.

    writemaster = write the array of posterior samples to disk

    returnmaster = return the array of posterior samples

Outputs:
    
    psum = [N] - probability that each object belongs to the foreground

    postmaster = [samplesize, N] array of membership probabilities

    """

    # Write in plot routine first to get the I/O and plotting working,
    # then refactor the responsibility calculation out into a separate
    # method.

    # Check that the run arguments are OK
    if not runargsok(runargs, [keyargs, keyfunc], Verbose=True):
        print("computeresps WARN - problem with runargs keywords")
        return

    # Check that the samples are OK
    flatsamples = getrunsamples(samples, pathsamples)
    if np.size(flatsamples) < 1:
        print("computeresps FATAL - no samples")
        return
    
    # OK if we're here then we have the ln prob function and its
    # arguments. Convenience-views:
    methprob = runargs[keyfunc]
    probargs = runargs[keyargs]

    print("computeresps DEBUG - flatsamples shape:", flatsamples.shape)
    print("computeresps DEBUG - lnprob function:", runargs[keyfunc])

    # Gather some convenient characteristics
    nsamples = flatsamples.shape[0]
    ndata = runargs[keyargs][1].shape[0] # this is really opaque!!

    # This is taking quite a while to calculate. See how long...
    t0 = time.time()
    print("computeresps INFO - starting responsibility loop:")

    # how far do we have to go?
    if samplesize < 0 or samplesize > nsamples:
        imax = nsamples
    else:
        imax = samplesize

    # Build the responsibilities array from the sample size we are
    # going to compute.
    postmaster = np.array([])
    if keepmaster:
        postmaster = np.zeros((imax, ndata))
    
    # Now do the calculation
    norm = 0.
    psum = np.zeros(ndata)
    for isample in range(imax):

        # evaluate the posterior probability that each point belongs
        # to the foreground
        lnprob, ll_fg, ll_bg = methprob(flatsamples[isample], *probargs, retblobs=True)
        probfg = np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))

        # Do the line-by-line, increment the norm
        psum += probfg
        norm += 1.
        
        # accumulate onto the master probfg array if we are going to store it:
        if keepmaster:
            postmaster[isample] = probfg

        # report out every so often
        if isample % ireport < 1 and isample > 0:
            telapsed = time.time() - t0
            itpersec = float(isample)/telapsed
            tremain = 0.
            if itpersec > 0.:
                tremain = float(imax)/itpersec
            print("computeresps INFO - iteration %i of %i after %.2e seconds: %.1f it/sec. Est %.2e sec remain" \
                  % (isample, imax, telapsed, itpersec, tremain), end='\r')

    t1 = time.time()
    print("") # carriage return so next print starts on the next line
    print("computeresps INFO - loops took %.2e seconds for %.2e samples" \
          % (t1-t0, nsamples))

    # evaluate the probability average
    psum /= norm

    # Now compute the probabilities
    #ncomputed = postmaster.shape[0]
    #probfg = np.sum(postmaster, axis=0) / np.float(ncomputed)
    #print("CHECK:", probfg - psum)

    # Write to disk if we kept the entire sample set
    if keepmaster and writemaster:
        np.save(pathprobs, postmaster)

    # If asked, return the master samples to the calling method
    if returnmaster:
        return psum, postmaster

    # Otherwise just return the average probabilities
    return psum
    
def showresps(postmasterin=np.array([]), \
              fignum=7, loghist=False, pminlog=0.01, \
              pathfig='test_resps.png', \
              wantbg=False):

    """Plots formal assignment probabilities. 

Example call (after getting the postmaster array from computeresps):

    fittwod.showresps(postmaster, loghist=True)
    

Inputs:

    postmasterin = [nsamples, ndata] array of membership probabilities

    loghist = plot histogram as log10 (only nonzero considered)

    pminlog = if log scale, minimum value to show

    wantbg = we want the background assignments not the foreground

    """

    if np.size(postmasterin) < 1:
        return

    nsamples, ndata = postmasterin.shape

    # do we want the reverse?
    postmaster = postmasterin
    if wantbg:
        postmaster = 1.0 - postmasterin
    
    # get the averages
    pformal = np.sum(postmaster, axis=0)/np.float(nsamples)

    bnonzero = pformal > pminlog

    print("showresps INFO:", np.size(bnonzero), np.sum(bnonzero))
    
    fig7=plt.figure(fignum)
    fig7.clf()
    ax71 = fig7.add_subplot(211)
    ax72 = fig7.add_subplot(212, sharex=ax71)

    # show the distribution, distinguishing nonzero
    if np.sum(bnonzero) > 0:
        #dum70 = ax71.hist(pformal[bnonzero], bins=100, log=True, color='#00274C')

        if not loghist:
            dum70 = ax71.hist(pformal[bnonzero], bins=100, log=True, color='#00274C')
        else:
            dum70 = ax71.hist(np.log10(pformal[bnonzero]), bins=100, \
                              log=True, color='#00274C')
            
    if np.sum(~bnonzero) > 0 and not loghist:
        dum71 = ax71.hist(pformal[~bnonzero], bins=100, log=True, color='#D86018')

    # Hack for convenient vertical axis when not showing log10
    ylargest = -1
        
    # start a vanilla histogram plot
    print("showresps INFO - plotting marginal distributions...")
    t0 = time.time()
    for iset in range(ndata):

        # if log hist, skip very small formal probabilites
        if loghist:
            if ~bnonzero[iset]:
                continue
            dum72 = ax72.hist(np.log10(postmaster[:,iset]), \
                              bins=50, alpha=0.2, log=False)
        else:
            n, _, _ = ax72.hist(postmaster[:,iset], \
                                bins=50, alpha=0.2, log=False)

            if bnonzero[iset] and np.max(n) > ylargest:
                ylargest = np.max(n)

    t1 = time.time()
    print("showhist INFO - ... done in %.1f seconds" % (t1-t0))
                
    # If not log hist, use our hack to set maximum y value
    if not loghist and ylargest > 0:
        ax72.set_ylim(top=ylargest*1.1)

    # Foreground/background string
    swhich = 'fg'
    if wantbg:
        swhich = 'bg'
        
    sx = r'$p_{%s}$' % (swhich)
    if loghist:
        sx = r'$log_{10}(p_{%s})$' % (swhich)

    for ax in [ax71, ax72]:
        ax.set_xlabel(sx)
        ax.set_ylabel(r'$N$')

        ax.grid(visible=True, alpha=0.3, which='both')
        
        if loghist:
            ax.annotate(r'$log_{10}(p_{%s}) > %.1f$' % \
                        (swhich, np.log10(pminlog)), \
                        (0.05,0.95), xycoords='axes fraction', \
                        ha='left', va='top', color='#00274C')
        else:
            ax.set_xlim(left=0.)


    # A few cosmetic adjustments
    fig7.subplots_adjust(hspace=0.05)

    # Set the title. Try an f-string approach
    ax71.set_title(f"Probabilities (%s): (nsamples, ndata) = ({nsamples:,}, {ndata:,})" % (swhich))

    # save the figure
    fig7.savefig(pathfig)

def showresptruths(fitresps='test_xyresps.xyxy', \
                   simresps='test_simdata.xyxy', \
                   creg=1.0e5, \
                   fitisfg=True, \
                   logx=True, \
                   fignum=8, \
                   pathfig='test_logreg.png', \
                   showscatter=True, \
                   pathprojs='test_flatproj.npy', \
                   argsscatt={'cmap':'inferno', 's':16} ):

    """Plots assigned responsibilities and truth (simulated)
responsibilities.

Inputs:

    fitresps = path to fitted responsibilities

    simresps = path to simulated identifications

    creg = regularization parameter in the logistic regression. See
    the documentation for sklearn.LogisticRegression for more.

    fitisfg [T/F] - the MCMC interprets "f" as the foreground fraction rather than the background fraction

    logx [T/F] - use log scale for the horizontal axis

    fignum = matplotlib figure number to assign

    showscatter = show scatterplot with outlier ID

    pathprojs = path to projected positions (needed if doing the scatterplot)

    argsscatt = arguments to pass to the scatter command.

    """

    try:
        asim = np.genfromtxt(simresps, unpack=False)
    except:
        print("showresptruths WARN - problem reading %s" % (simresps))
        return

    try:
        afit = np.genfromtxt(fitresps, unpack=False)
    except:
        print("showresptruths WARN - problem reading %s" % (fitresps))
        return

    # trusting the input files for the moment...
    respsim = asim[:,-1]
    respfit = afit[:,-1]

    # If "f" in the MCMC means identification with the foreground:
    if fitisfg:
        respfit = 1.0 - respfit

    # Try logistic regression on the result
    clf = LogisticRegression(C=creg)
    clf.fit(respfit[:, None], respsim)
    xfine = np.linspace(respfit.min(), 1., 100)
    yfine = expit(xfine * clf.coef_ + clf.intercept_).ravel()
    
    # Show the logistic regression
    fig8 = plt.figure(fignum)
    fig8.clf()
    ax80 = fig8.add_subplot(111)
    dum = ax80.scatter(respfit, respsim, alpha=0.5, \
                       label='Responsibilities')

    # overplot the logistic regression model
    dumfine = ax80.plot(xfine, yfine, c='#75988d', zorder=10, alpha=0.7, \
                        ls='--', lw=1, \
                        label='logistic regression')

    # take control over the legend location (don't want it to overly
    # the corners)
    legloc = 'center right'
    if logx:
        legloc = 'center left'
    
    leg = ax80.legend(fontsize=8, loc=legloc)

    if logx:
        ax80.set_xscale('log')
    
    ax80.set_xlabel('p(is background), MCMC')
    ax80.set_ylabel('Simulated as background')

    # Save the figure to disk
    if len(pathfig) > 3:
        fig8.savefig(pathfig)

    # Show the scatterplot
    if not showscatter:
        return

    # load the projected positions
    try:
        projxy = np.load(pathprojs)
    except:
        print("showresptruths WARN - problem loading projections from %s" \
              % (pathprojs))
        return

    # It's possible at this point that the projections are for a
    # different set than the identifications. Guard against that here
    nsim = asim.shape[0]
    nproj = projxy.shape[-1]
    if nsim != nproj:
        print("showresptruths WARN - simulation and projections have different lengths")
        return
        
    # Use the 50th percentile positions (note this is using the FOUND
    # positions and not the SIMULATED positions, so might be biased).
    x50 = np.percentile(projxy[:,0,:], 50., axis=0)
    y50 = np.percentile(projxy[:,1,:], 50., axis=0)

    conv = 3600.
    dx = x50 - asim[:,2]
    dy = y50 - asim[:,3]

    dx *= conv
    dy *= conv
    
    fig9=plt.figure(fignum+1, figsize=(7.5,2.7))
    fig9.clf()
    ax90 = fig9.add_subplot(121)
    ax91 = fig9.add_subplot(122, sharex=ax90, sharey=ax90)

    dum90 = ax90.scatter(dx, dy, c=respsim, edgecolor='k', alpha=0.7, \
                         **argsscatt)
    dum91 = ax91.scatter(dx, dy, c=respfit, edgecolor='k', alpha=0.7, \
                         **argsscatt)

    cb90 = fig9.colorbar(dum90, ax=ax90)
    cb91 = fig9.colorbar(dum91, ax=ax91)

    ax90.set_title('Simulated')
    ax91.set_title('Estimated')
    
    for ax in [ax90, ax91]:
        ax.set_xlabel(r'$\Delta\xi$, arcsec')
        ax.set_ylabel(r'$\Delta\eta$, arcsec')

        
    # Cosmetics
    fig9.subplots_adjust(hspace=0.3, wspace=0.4, left=0.13, bottom=0.2)
        
def calcmoments(projxy = np.array([]), \
                pathprojxy='', \
                methavg=np.mean):

    """Calculates moments of a sample-set of 2d simulations. Example call:

    meds, covs, skews, kurts = fittwod.calcmoments(projxy, methavg=np.median)

Inputs:

    projxy = [nsamples, 2, N] array of xy projected position for
    each sample

    pathprojxy = path to load the projected samples if not supplied

    methavg = method to use to compute the average. Must accept the
    "axis" keyword. Usual choices will be np.mean or np.median.

Returns:

    avgs = [N,2] array of average values

    covs = [N,2,2] array of covariances

    skews = [N,2] array of skewness
    
    kurts = [N,2] array of kurtosis

    """

    # Ensure we have projections to work with
    projxy = getrunsamples(projxy, pathprojxy, Verbose=True)
    
    # Nothing to do if no data supplied
    blank = np.array([])
    if np.size(projxy) < 3:
        return blank, blank, blank

    if np.ndim(projxy) != 3:
        return blank, blank, blank

    covs = cov3d(projxy)
    avgs = methavg(projxy, axis=0).T
    skews = stats.skew(projxy, axis=0).T
    kurts = stats.kurtosis(projxy, axis=0).T
    
    return avgs, covs, skews, kurts
    
def showcovarscomp(pathcovs='test_flatsamples.pickle', dcovs={}, \
                   keymcmc='covpars', keylsq='lsq_hessian_inv', \
                   keylabels='slabels', \
                   fignum=6, \
                   sqrt=True, \
                   log=True, \
                   pathfig=''):
    
    """Visualizes the comparison in parameter covariance between the ltsq
and the mcmc evaluations

Inputs:

    pathcovs = path to pickle file holding the covariances. Ignored if
    dcovs is supplied

    dcovs = dictionary holding the samples.

    keymcmc = dictionary key corresponding to the MCMC covariances
    
    keylsq = dictionary key corresponding to the LSQ covariances

    keylabels = dictionary key corresponding to the labels

    fignum = matplotlib figure number to use

    log = use log scale for the MCMC and LSQ heatmaps

    sqrt = use sqrt scale for the MCMC and LSQ heatmaps

    pathfig = file path to save figure (must contain ".")

    nextramcmc = number of "extra" arguments in mcmc parameters. (For
    example, might be noise parameters that the lsq array doesn't
    have).

Returns:

    No return quantities. See the figure.

Example call:

    fittwod.showcovarscomp(pathcovs='./no_src_uncty_linear/test_flatsamples.pickle', pathfig='lsq_mcmc_covars.png', sqrt=True, log=True)

    """

    # Ensure the dcovs dictionary is populated
    if len(dcovs.keys()) < 1:
        try:
            with open(pathcovs, "rb") as robj:
                dcovs = pickle.load(robj)
        except:
            nopath = True

    # Check to see if all the right entries are present
    lkeys = dcovs.keys()
    for key in [keymcmc, keylsq, keylabels]:
        if not key in lkeys:
            print("showcovarscomp WARN - key not in dictionary: %s" \
                  % (key))
            return

    # convenience views
    covslsq = dcovs[keylsq]
    covsmcmc = np.copy(dcovs[keymcmc])
    slabels = dcovs[keylabels]

    # The MCMC may also be exploring noise parameters or mixture model
    # fractions, which the LSQ approach can't do. In that instance,
    # take just the model parameters
    nmcmc = np.shape(covsmcmc)[0]
    nlsq = np.shape(covslsq)[0]
    if nmcmc > nlsq:
        covsmcmc = covsmcmc[0:nlsq, 0:nlsq]
        slabels = dcovs[keylabels][0:nlsq]
    
    # Showing a heatmap of one of the quantities
    fig6 = plt.figure(6, figsize=(8,6))
    fig6.clf()
    ax61 = fig6.add_subplot(221)
    ax62 = fig6.add_subplot(222)
    ax63 = fig6.add_subplot(224)

    # if log, we can meaningfully show the text. Otherwise
    # don't. (Kept out as a separate quantity in case we want to add
    # more conditions here.)
    showtxt = log

    # fontsize for annotations
    fontsz=5
    if covsmcmc.shape[0] < 10:
        fontsz=6
    
    showheatmap(covsmcmc, slabels, ax=ax61, fig=fig6, \
                log=log, sqrt=sqrt, \
                cmap='viridis_r', title='MCMC', \
                showtext=showtxt, fontsz=fontsz)
    showheatmap(covslsq, slabels, ax=ax62, fig=fig6, \
                log=log, sqrt=sqrt, \
                cmap='viridis_r', title='LSQ', \
                showtext=showtxt, fontsz=fontsz)

    # find the fractional difference. The mcmc has already been cut
    # down to match the lsq length above, so if the arrays still
    # mismatch their lengths then something is wrong with the input.
    fdiff = (covslsq - covsmcmc)/covsmcmc
    titlediff = r'(LSQ - MCMC)/MCMC'
    showheatmap(fdiff, slabels[0:nlsq], ax=ax63, fig=fig6, log=False, \
                cmap='RdBu_r', title=titlediff, showtext=True, \
                symmetriclimits=True, symmquantile=0.99, \
                fontcolor='#D86018', fontsz=fontsz)

    # Warn on the plots if more mcmc parameters were supplied than
    # used. In all the use cases these should be noise parameters that
    # the LSQ covariances don't have, so it's not an "error".
    if nmcmc > nlsq:
        ax61.annotate('MCMC params ignored: %i' % (nmcmc-nlsq), \
                      (0.97,0.97), xycoords='axes fraction', \
                      ha='right', va='top', fontsize=8, \
                      color='#9A3324')
    
    # save figure to disk?
    if len(pathfig) > 0:
        if pathfig.find('.') > 0:
            fig6.savefig(pathfig)
    
def showheatmap(arr=np.array([]), labels=[], \
                ax=None, fig=None, fignum=6, \
                cmap='viridis', \
                showtext=False, fontsz=6, fontcolor='w', \
                addcolorbar=True, \
                sqrt=False, \
                log=False, \
                title='', \
                maskupperright=True, \
                symmetriclimits=False, \
                symmquantile=1.):

    """Plots 2D array as a heatmap on supplied axis. Intended use:
visualizing the covariance matrix output by an MCMC or other parameter
estimation.

Inputs:

    arr = [M,M] array of quantities to plot

    labels = [M] length array or list of quantity labels. If there are
    more labels than datapoints, only 0:M are included. This may not
    be what you want.

    ax = axis on which to draw the plot

    fig = figure object in which to put the axis

    fignum = figure number, if creating a new figure

    cmap = color map for the heatmap

    showtext = annotate each tile with the array value?

    fontsz = font size for tile annotations

    addcolorbar = add colorbar to the axis?

    sqrt = take sqrt(abs(arr)) before plotting

    log = colormap on a log10 scale (Note: if sqrt and log are both
    true, then the quantity plotted is log10(sqrt(abs(arr))).  )

    title = '' -- string for axis title

    maskupperright -- don't plot the duplicate upper-right (off
    diagonal) corner values

    symmetriclimits -- if not logarithmic plot, makes the color limits
    symmetric about zero (useful with diverging colormaps)

    symmquantile -- if using symmetric limits, quantile of the limits
    to use as the max(abs value). Defaults to 1.0

Outputs:

    None

    """

    # Must be given a 2D array
    if np.ndim(arr) != 2:
        return
    
    # Ensure we know where we're plotting
    if fig is None:
        fig = plt.figure(fignum)
        
    if ax is None:
        ax = fig.add_subplot(111)

    # What are we showing?
    labelz = r'$V_{xy}$'
    arrsho = np.copy(arr)

    if sqrt:
        labelz = r'$\sqrt{|V_{xy}|}$'
        arrsho = np.sqrt(np.abs(arr))

    # log10 - notice this happens on ARRSHO (i.e. after we might have
    # taken the square root).
    if log:
        labelz = r'$log_{10} \left(%s\right)$' % \
            (labelz.replace('$',''))
        arrsho = np.log10(np.abs(arrsho))

    # make arrsho a masked array to make things a bit more consistent
    # below
    arrsho = ma.masked_array(arrsho)
        
    # Mask upper-left (duplicate points)?
    if maskupperright:
        iur = np.triu_indices(np.shape(arrsho)[0], 1)
        arrsho[iur] = ma.masked

    # compute symmetric colorbar limits?
    vmin = None
    vmax = None
    if symmetriclimits and not log and not sqrt:
        maxlim = np.quantile(np.abs(arrsho), symmquantile)
        vmin = 0. - maxlim
        vmax = 0. + maxlim
        
    # imshow the dataset
    im = ax.imshow(arrsho, cmap=cmap, vmin=vmin, vmax=vmax)

    # Ensure labels are set, assuming symmetry
    ncov = np.shape(arrsho)[0]
    nlab = np.size(labels)

    if nlab < ncov:
        labls = ['p%i' % (i) for i in range(ncov)]
    else:
        labls = labels[0:ncov]        

    # Now set up the ticks
    ax.set_xticks(np.arange(ncov))
    ax.set_yticks(np.arange(ncov))
    ax.set_xticklabels(labls)
    ax.set_yticklabels(labls)

    # Text annotations (this might be messy)
    if showtext:
        for i in range(len(labls)):
            for j in range(len(labls)):
                if arrsho[i,j] is ma.masked:
                    continue
                
                text = ax.text(j, i, \
                               "%.2f" % (arrsho[i,j]), \
                               ha="center", va="center", \
                               color=fontcolor, \
                               fontsize=fontsz)

    if addcolorbar:
        cbar = fig.colorbar(im, ax=ax, label=labelz)

    # Set title
    if len(title) > 0:
        ax.set_title(title)
        
    # Some cosmetic settings
    fig.tight_layout()
    
def test_mags(npts=200, loga=-5., logb=-26., c=2.5, \
              expon=2.5, maglo=16., maghi=22., \
              parscov=[], showcovs=False):

    """Generate fake magnitudes and show the noise scale factor vs
magnitude. The noise model used is

    stdx = a + b.exp(m c)

    Lots of screen output since this was used to debug development

    """

    # Parameters that correspond reasonably well to the datasets in
    # test linear (examples show different exponents):
    #
    # x,y: loga = -5., logb=-26., c=2.5  
    #
    # xi,eta: loga = -6., logb = -23.5, c = 2.
    
    # Useful to get better intuition about what sort of model
    # parameters are sensible.
    #
    # 

    # convert pars into array
    magpars = [loga, logb, c]
    
    mags = makefakemags(npts, expon, maglo=maglo, maghi=maghi)
    sigm = noisescale(magpars, mags)

    # now try assigning covariance matrices from this
    # covsnx2x2 = mags2cov(magpars, mags, parscov)

    # Let's look at the ingredients:
    corr3xn = parsecorrpars(sigm, parscov)
    print("DBG: corr3xn:", corr3xn.shape)
    print(corr3xn[:,0])
    print(corr3xn[:,1])

    covsnx2x2 = corr3n2covn22(corr3xn)
    print("DBG: covsnx2x2:", covsnx2x2.shape)
    print(covsnx2x2[0])
    print(covsnx2x2[1])

    # is it doing it in one step that's breaking?
    covsdirect = mags2cov(magpars, mags, parscov)
    
    # Are any of the covariances singular?
    detcovs = np.linalg.det(covsnx2x2)
    bbad = detcovs <= 0.
    print("test_mags info: singular planes: ", np.sum(bbad) )

    # Try our simple one-liner to return the object. WTF is going
    # wrong there?
    CC = makemagcovars(magpars, mags, parscov)
    
    # ^^ That works method-by-method. Is something wrong with
    # covarsnx2x2?
    CC = CovarsNx2x2(covars=covsnx2x2)

    print("Object check - method by method:")
    print(covsnx2x2[0])
    print("Object check - makemagcovars:")
    print(CC.covars[0])
    print("Object check - mag2cov:")
    print(covsdirect[0])

    print("########")
    
    #print(covsnx2x2[0], detcovs[0])
    
    # Show the noise covariances
    fig2=plt.figure(2)
    fig2.clf()
    ax21 = fig2.add_subplot(211)
    ax22 = fig2.add_subplot(212)

    blah21 = ax21.hist(mags, bins=25, alpha=0.5)
    blah22 = ax22.scatter(mags, sigm, s=9, alpha=0.5, \
                          color='#00274C', zorder=1, label='generated')

    # Show the model components
    mfine = np.linspace(np.min(mags), np.max(mags), 100)
    ymod1 = np.repeat(10.0**loga, np.size(mfine))
    ymod2 = 10.0**logb * np.exp(mfine*c)

    # Now get the stdxs of the N,2,2 matrices we just produced.
    if showcovs:
        gencov3n = np.vstack(( covsnx2x2[:,0,0], \
                               covsnx2x2[:,1,1], \
                               covsnx2x2[:,0,1] ))
    
        gencor3n = cov2corr1d(gencov3n)            
        blah23 = ax22.scatter(mags, gencor3n[0], s=.5, c='#D86018', \
                              zorder=15, label='Back-converted from Nx2x2')
    
    # Labels for legends
    sleg1 = r'$\log_{10} \sigma = %.1f$' % (loga)
    #sleg2 = r'$\sigma = 10.0^{%.2f} \times e^{%.2f m}$' % (b, c)

    sleg2 = r'$\sigma = b e^{mc}$ w/ $(\log_{10}(b), c) = (%.1f, %.1f)$' \
        % (logb,c)
    
    blah221 = ax22.plot(mfine, ymod1, ls='--', color='b', zorder=5, \
                        label=sleg1, lw=1)
    blah222 = ax22.plot(mfine, ymod2, ls='-', color='r', zorder=5, \
                        label=sleg2, lw=1)

    leg = ax22.legend(fontsize=8)

    # axis carpentry
    ax22.set_xlabel(r'$m$')
    ax21.set_ylabel(r'$N(m)$')
    ax22.set_ylabel(r'$\sigma$')

    ax21.get_xaxis().set_ticklabels([])    
    fig2.subplots_adjust(hspace=0.01)

    ax21.set_title(r'Magnitudes: $N(m) \propto m^{%.1f}$' % (expon))
    
    # vertical axis logarithmic
    ax22.set_yscale('log')

    # The vertical scale tends to go very low at the bright
    # end. Adjust the scale accordingly.
    yscale = np.copy(ax22.get_ylim())
    yadj = np.array([ymod1[0]*0.2, yscale[-1]])
    ax22.set_ylim(yadj)

