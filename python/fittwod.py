#
# fittwod.py
#

#
# WIC 2024-07-24 - test-bed to use and fit transformation objects in
# unctytwod.py
# 

import os, time, pickle
from multiprocessing import cpu_count, Pool

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
    
def checkcorrpars(addcorr=np.array([]), islog10=False):

    """Utility - given [stdx, stdy/stdx, corrxy], determines if the supplied parameters violate positivity and other constraints. Inputs:

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

def splitmodel(pars=np.array([]), nnoise=0, nvars=0):

    """Split a 1D parameter array into transformation parameters, noise
model parameters, and covariance parameters. (It's probably better to
use splitpars(), which this method calls and which is more flexible.)

Inputs:

    pars = [M + nnoise + nvars] 1D array with the parameters

Returns:

    transf = [M] array of transformation parameters

    noise = [nnoise] array of noise model parameters

    covar = [nvars] array of variance shape parameters

    """

    transf, lsplit = splitpars(pars, [nnoise, nvars])
    return transf, lsplit[0], lsplit[1]

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
            nnoise=0, mags=np.array([]) ):

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
    parsmodel, addnoise, addvars = splitmodel(pars, nnoise, npars)
    
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
    
def sumlnlike(pars, transf, xytarg, covtarg, covextra=0. ):

    """Returns sum(log-likelihood) for a single-population model"""

    expon, det, piterm = lnlike(pars, transf, xytarg, covtarg, covextra)
    return np.sum(expon) + np.sum(det) + np.sum(piterm)

def lnlikestat(pars, transf, xytarg, covtarg, covextra=0.):

    """Returns the sum of all three terms on a per-object basis"""

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
           methprior=lnprior_unif, \
           methlike=sumlnlike, \
           methprior_noise=lnprior_noisemodel_rect):

    """Evaluates ln(posterior). Takes the method to compute the ln(prior)
and ln(likelihood) as arguments.

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

    """

    # Split the transformation parameters from noise, etc. parameters.
    pars, addnoise, addvars = splitmodel(parsIn, nnoise, nvar)

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

    lnprior = lnprior_transf + lnprior_noise + lnprior_corr
    if not np.isfinite(lnprior):
        return -np.inf

    # Generate any additional extra covariance.
    cov_ok = True
    covextra, cov_ok = extracovar(addnoise, mags, \
                                  addvars, fromcorr, islog10) 

    ## 2024-08-02 replaced with more targeted routines. Commented out
    ## for now.    
    #pars = parsIn
    #covextra = 0.
    #noisepars = np.array([])
    #if addvar:
    #    pars, covextra, covok = \
    #        skimvar(parsIn, xytarg.shape[0], nvar, fromcorr, islog10, \
    #                nnoise, mags)

    #    # If the supplied parameters led to an improper covariance
    #    # (correlation coefficient outside the range [-1., 1.], say),
    #    # then reject the sample.
    #    if not covok:
    #        return -np.inf
        

    # evaluate ln likelihood
    lnlike = methlike(pars, transf, xytarg, covtarg, covextra) 

    # If this is about to return nan, provide a warning and show the
    # parameters. 
    if np.any(np.isnan(lnlike)):
        print("lnprob WARN - at least one NaN entry. Trial params:")
        print(parsIn)
    
    # return the ln posterior
    return lnprior + lnlike


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
            nnoiseguess=0, nshapeguess=0):

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

Returns:

    scalesret = array of scale factors for MCMC initial guesses

    truthsret = array of 'truth' values with same ordering as guess

    """

    # Step 1: unpack the supplied parameter arrays
    truthpars, truthnoise, truthshape = \
        splitmodel(truths, nnoisetruth, nshapetruth)

    guesspars, guessnoise, guessshape = \
        splitmodel(guess, nnoiseguess, nshapeguess)

    # Step 2: find the offsets and the perturbation scales for the
    # MCMC walker start positions
    scalesxy = scalexyguess(guesspars, truthpars)
    scalesnoise = scaleguessbyoffset(guessnoise, truthnoise)
    scalesshape = scaleguessbyoffset(guessshape, truthshape)
    
    # Step 3: pad the truths for the individual pieces
    truthsxy = padxytruths(truthpars, guesspars)
    truthsnoise = padtruths(truthnoise, guessnoise)
    truthsshape = padtruths(truthshape, guessshape)

    # Step 4: gather the pieces for return
    scalesret = np.hstack(( scalesxy, scalesnoise, scalesshape ))
    truthsret = np.hstack(( truthsxy, truthsnoise, truthsshape ))

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
                    minimizermaxit=3000):

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
    
    if unctysrc:
        nudgexy = Cxy.getsamples()
    if unctytarg:
        nudgexytran = Ctran.getsamples()

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
    xytarg = xytran + nudgexytran + nudgexyextra

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
    showsimxy(xy, xyobs, xytran, xytarg, covtran, mags, \
              CExtra, PFit, PTruth, fignum=1)


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
        
        #lextra = [r'$V_{\xi\eta}$']
        #if npars_extravar > 1:
        #    lextra = [r'$V_{\xi}$', r'$V_{\eta}$']
        #    if npars_extravar > 2:
        #        lextra = [r'$V_{\xi}$', r'$V_{\eta}$', r'$V_{\xi \eta}$']

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
        
    # now (drumroll) set up the sampler.
    methpost = lnprob
    args = (PFit, xytarg, covtran, addvar, npars_extravar, \
            extra_is_corr, stdx_is_log, npars_noise, mags)
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

        # now update our guess and truths array accordingly.
        guess = np.copy(soln.x)
        scaleguess, truths = padpars(fpars, guess, \
                                     ntruths_extranoise, \
                                     ntruths_extrashape, \
                                     npars_noise, \
                                     npars_extravar)

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

    # send the iniital guess through the ln(prob) to test whether it
    # returns sensible values
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

def showsimxy(xy=np.array([]), xyobs=np.array([]), \
              xytran=np.array([]), xytarg=np.array([]), \
              covtran=np.array([]), \
              mags=np.array([]), CExtra=None, \
              PFit=None, PTruth=None, \
              fignum=1):

    """Show some characteristics of the generated data"""

    # refactored out of testmcmc_linear to try to clean up that method

    fig1 = plt.figure(fignum, figsize=(9.5,5))
    fig1.clf()
    ax1=fig1.add_subplot(231)
    ax2=fig1.add_subplot(232)
    ax3=fig1.add_subplot(234)
    ax4=fig1.add_subplot(235)
    ax5 = None # now set below
    ax6 = None # now set in a conditional below
    
    fig1.subplots_adjust(wspace=0.4, hspace=0.4, \
                         left=0.15, bottom=0.15)

    # generated and perturbed points in original and target frame:
    blah1=ax1.scatter(xy[:,0], xy[:,1], s=1)
    blah2=ax2.scatter(xyobs[:,0], xyobs[:,1], c='g', s=1)
    blah3=ax3.scatter(xytran[:,0], xytran[:,1], s=1)
    blah4=ax4.scatter(xytarg[:,0], xytarg[:,1], c='g', s=1)

    # Set titles for the first four axes
    ax1.set_title('Generated')
    ax2.set_title('Perturbed')
    ax3.set_title('Transformed')
    ax4.set_title('Target')

    # Labels for the position plots
    for ax in [ax1, ax2]:
        ax.set_xlabel(r'X')
        ax.set_ylabel(r'Y')

    for ax in [ax3, ax4]:
        ax.set_xlabel(r'$\xi$')
        ax.set_ylabel(r'$\eta$')

    
    # If a fit object has been sent in, use it to show the transformed
    # positions using the initial guess (or whatever parameters are in
    # the pfit object):
    if PFit is not None:
        blah5 = ax4.scatter(PFit.xytran[:,0], PFit.xytran[:,1], \
                            c='r', s=1)

    # Show the residuals from truth in the target space on axis 5
    if PTruth is not None:
        ax5=fig1.add_subplot(236)
        fxy = PTruth.xytran - xytarg
        blah5 = ax5.scatter(fxy[:,0], fxy[:,1], s=.1)
        cc = np.cov(fxy, rowvar=False)
        sanno = "%.2e, %.2e, %.2e" % (cc[0,0], cc[1,1], cc[0,1])
        anno5 = ax5.annotate(sanno, (0.05,0.05), \
                             xycoords='axes fraction', \
                             ha='left', va='bottom', fontsize=6)

        # Enforce aspect ratio for residuals plot
        ax5.set_aspect('equal', adjustable='box')
        ax5.set_title('Residuals, generated')
        ax5.set_xlabel(r'$\Delta \xi$')
        ax5.set_ylabel(r'$\Delta \eta$')
        
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
    # the LSQ covariances don't have.
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

