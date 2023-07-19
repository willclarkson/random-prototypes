#
# likesMCMC.py
#

# WIC 2023-06-19 -- collection of loglikes and priors for use in MCMC
# explorations.
#
# The log-likelihoods return N-dimensional arrays, which then the
# log-posteriors sum over N. This is so that logsumexp will work where needed.

import numpy as np
from scipy.special import logsumexp

def uTVu(u, V):

    """ 
    Returns u^T.V.u where
    
    u = [N,m] - N datapoints of dimension m (typically deltas array)

    V = [N,m,m] - N covariances of dimension m x m (a covariance stack)

    This will return an N-element array.

    """

    Vu = np.einsum('ijk,ik -> ij', V, u)
    return np.einsum('ij,ji -> j', u.T, Vu)

def propagate_covars_abc(covars, abc):

    """Propagates [n,2,2] covariances through 6-term linear transformation
with transformation specified as [a, b, c, d, e, f] array"""

    _, b, c, _, e, f = abc


    return propagate_covars_bcef(covars, b, c, e, f)

def propagate_covars_vectorized(covars, b, c, e, f):

    """Propagates [n,2,2] covariances by b,c,e,f, element by element"""
    
    # Mostly to ensure I'm not misunderstanding np.matmul's broadcast
    # rules

    j11, j12, j21, j22 = b,c,e,f # yes, I know...
    
    v11 = covars[:,0,0]
    v12 = covars[:,0,1]
    v22 = covars[:,1,1]

    s11 = j11**2 * v11 + j12**2 * v22 + 2.0 * j11*j12*v12
    s22 = j21**2 * v11 + j22**2 * v22 + 2.0 * j21*j22*v12
    s12 = j11*j21* v11 + j12*j22 *v22 + (j11*j22 + j12*j21) * v22

    covtran =  np.zeros((v11.size, 2, 2))
    covtran[:,0,0] = s11
    covtran[:,1,1] = s22
    covtran[:,0,1] = s12
    covtran[:,1,0] = s12

    return covtran
    
def propagate_covars_bcef(covars, b,c,e,f):

    """Propagates [n,2,2] covariances through 6-term linear
transformation where the b, c, e, f transformation is specified"""

    # refactor params into 2x2 matrix
    J = np.array([[b,c],[e,f]])

    return np.matmul(J, np.matmul(covars, J.T))
    
def bcefToPars(b,c,e,f):

    """Converts b,c,e,f (from xi = a + bx + cy, eta = d + ex + fy) to the
sx, sy, rotDeg,skewDeg parameters. 

    Returns sx, sy, rotDeg, skewDeg"""

    # Implementation note: in this module, b,c,e,f are SCALARS. 
    
    sx = np.sqrt(b**2 + e**2)
    sy = np.sqrt(c**2 + f**2)

    arctanCF = np.arctan2(c,f)
    arctanEB = np.arctan2(e,b)
    rotDeg = 0.5*np.degrees(arctanCF - arctanEB)
    skewDeg =    np.degrees(arctanCF + arctanEB)
    
    # enforce convention on sx, beta (don't want beta > 45 deg)
    if skewDeg > 90.:
        skewDeg -= 180.
        rotDeg += 90.
        sx  *= -1.

    if skewDeg < -90.:
        skewDeg += 180.
        rotDeg -= 90.
        sx *= -1.

    return sx, sy, rotDeg, skewDeg

def parsToBcef(sx, sy, rotDeg, skewDeg):

    """Coverts (sx, sy, rotdeg, skewdeg) to the b,c,e,f in the 6-term
linear transformation"""

    radX = np.radians(rotDeg - 0.5*skewDeg)
    radY = np.radians(rotDeg + 0.5*skewDeg)

    b =  sx * np.cos(radX)
    c =  sy * np.sin(radY)
    e = -sx * np.sin(radX)
    f =  sy * np.cos(radY)

    return b,c,e,f

# The following utilities accept the 6-term [a,b,c,d,e,f] vector and
# return various things we may want

def abcToPars(abc):

    """Utility - produces [dxi, deta, sx, sy, rotDeg, skewDeg] from
[a,b,c,d,e,f]"""

    a,b,c,d,e,f = abc

    sx, sy, rotDeg, skewDeg = bcefToPars(b,c,e,f)

    return np.array([a, d, sx, sy, rotDeg, skewDeg])

def parsToAbc(pars):

    """Utility - produces [a,b,c,d,e,f] from [a,d,sx,sy,rot,skew]"""

    a, d, sx, sy, rotDeg, skewDeg = pars

    b, c, e, f = parsToBcef(sx, sy, rotDeg, skewDeg)

    return np.array([a,b,c,d,e,f])
    
def abcToJ(abc):

    """Utility: produces Jacobian from 6-term [a,b,c,d,e,f] array"""

    _,b,c,_,e,f = abc

    return np.array([[b,c], [e,f]])

def prepCovForOutliers(covars, scalefac=10, nrows=1):

    """Utility: given a [N,2,2] set of covariances, prepare the scaled
median covariance as a [2,2] output. If 2-dimensional input, we need
to supply the number of planes in the output stack.

    """

    # covars must be 2 or 3 dimensional
    ndimen = covars.ndim
    if not (1 < ndimen < 4):
        print("prepCovForOutliers FATAL - covars must be 2D or 3D.")
        return np.array([])

    # if [N,2,2] take the median and scale, otherwise just use the input
    if ndimen == 3:
        covscal = np.median(covars,axis=0)*scalefac
        nrows = covars.shape[0]
    else:
        covscal = covars * scalefac

    # Now construct the [N,2,2] inverse covariance matrix
    output = np.zeros((nrows,2,2))
    output[:] = covscal

    return output

def lognormal(deltas, covars):

    """Returns the log(normal) parameterized by deltas and
covariances. Includes covariance matrix inversion. DOES NOT SUM along
the datapoints.

    """

    # Invert the covariance matrix and evaluate the exponential term
    invcovars = np.linalg.inv(covars)
    expon = uTVu(deltas, invcovars)
    term_expon = -0.5 * expon

    # Evaluate the determinants term
    lndets = np.log(np.linalg.det(covars))
    term_dets = -0.5 * lndets

    # keep both terms so we can check them separately
    return term_expon, term_dets

def lognormalsum(deltas, covars):

    """Computes the sum along N of the log normal distribution, keeping the terms separate so that we can examine them downstream"""

    term_expon, term_dets = lognormal(deltas, covars)

    return np.sum(term_expon), np.sum(term_dets)
    
################ log-likelihood and priors follow #################

def loglike_linear_fast(pars, xypattern, xi, invcovars):

        """Returns the log-likelihood for the M-term (6-term linear or 4-term
similarity) plane mapping. Covariances are in the (xi, eta) plane and
are assumed independent of the model parameters (i.e. assume x,y
covariances are unimportant). DOES NOT SUM OVER N.

        pars = [M] - parameter array 

        xypattern = [N,2,M] - element pattern array in xy

        xi = [N,2] - N-element target positions (xi, eta)

        invcovars = [N,2,2] stack of covariances in the (eta, xi)
        plane. Note that this is the INVERSE of the covariances, since
        we don't want to perform the matrix inverse each time we
        evaluate this.

        """

        # Project the positions into the xi, eta plane using the
        # current parameters
        proj = np.matmul(xypattern, pars)
        deltas = xi - proj

        # evaluate the N-element array uT.V.u ( where V is the INVERSE
        # covariance)...
        expon = uTVu(deltas, invcovars)

        # ... and use this to evaluate the sum
        return -0.5 * np.sum(expon)

def loglike_linear(pars, xypattern, xi, covars):

    """Returns the log-likelihood for the M-term (6-term linear or 4-term
similarity) plane mapping. Covariances are in the (xi, eta) plane and
are assumed independent of the model parameters (i.e. assume x,y
covariances are unimportant) 

DOES NOT SUM OVER N because we'll need this in logsumexp.

        pars = [M] - parameter array 

        xypattern = [N,2,M] - element pattern array in xy

        xi = [N,2] - N-element target positions (xi, eta)

        covars = [N,2,2] stack of covariances in the (eta, xi)
        plane. NOT the inverse covariances!

    """

    # Project the positions into the xi, eta plane using the
    # current parameters
    proj = np.matmul(xypattern, pars)
    deltas = xi - proj

    term_expon, term_dets = lognormal(deltas, covars)

    return term_expon + term_dets
    
def logprior_unif(pars):

    """Uniform prior. Takes arguments for consistency with other calls"""

    return 0.

def logprior_unif_scalenobias(pars):

    """Enforces uniform prior including the jacobian det correction factor for the scale factor product"""

    # These are absolute values (we are not enforcing the conditions
    # to include the flipping in sf)
    
    sx = np.sqrt(pars[1]**2 + pars[4]**2)
    sy = np.sqrt(pars[2]**2 + pars[5]**2)

    lp = 0.-np.log10(sx)-np.log10(sy)

    if not np.isfinite(lp):
        return -np.inf
    
    return lp
    
def logprior_unif_signs(pars):

    """Enforces a prior on the sign of the parameters."""

    # Hard coded this to match the specific test case. If this works,
    # figure out how to generalize it!

    sgnpriors = np.array([1, -1, 1, -1, 1, 1])
    sgnpars = np.asarray(np.sign(pars), 'int')

    if np.sum(sgnpriors != sgnpars) > 0:
        return -np.inf

    return 0.

def logprior_unif_mixture(pars):

    """Returns uniform prior with conditions on ln(mixture fraction), which is assumed to be the final parameter in pars."""

    if pars[-1] > 0.:
        return -np.inf

    # ugh - this is starting to get nested...
    lp = logprior_unif_scalenobias(pars[0:6])
    
    return 0.
    
def logprob_linear_unif_fast(pars, xypattern, xi, invcovars):

    """ 
    Computes log posterior (to within a constant) for the linear mapping, with a uniform prior on the parameters

"""

    # Thought: we probably could pass the methods in to this object?
    
    lp = logprior_unif(pars)
    if not np.isfinite(lp):
        return -np.inf

    return lp + np.sum(loglike_linear_fast(pars, xypattern, xi, invcovars))

def logprob_linear_unif(pars, xypattern, xi, covars):

    """ 
    Computes log posterior (to within a constant) for the linear mapping, with a uniform prior on the parameters

"""

    # Thought: we probably could pass the methods in to this object?
    
    # lp = logprior_unif(pars)
    lp = logprior_unif_scalenobias(pars)
    if not np.isfinite(lp):
        return -np.inf

    return lp + np.sum(loglike_linear(pars, xypattern, xi, covars))


### Uncertainties in both source and destiation coordinates

def loglike_linear_unctyproj(pars, xypattern, xi, xycovars, xicovars):

    """Returns the log-likelihood for 6-term plane mapping, when both the
(x,y) and (xi, eta) positions have uncertainty covariances. DOES NOT
SUM OVER N. Inputs:

    pars   -    [6] - parameter array

    xypattern = [N,2,6] - element pattern array in xy
    
    xi = [N,2] - N-element target positions (xi, eta)

    xycovars  -  [N,2,2] stack of covariance matrices in (x,y)

    xicovars -   [N,2,2] stack of covariance matrices in (xi, eta)

    """

    # Project xy into the xi, eta plane and compute deltas
    deltas = xi - np.matmul(xypattern, pars)

    # Project the (x,y) covariances into the (xi, eta) plane
    covxy_proj = propagate_covars_abc(xycovars, pars)

    # Total covariance including (xi, eta) and projected (x,y)
    covars = xicovars + covxy_proj

    # REFACTORED into method lognormal()
    ## Now for the two covariance terms in the likelihood. We need the
    ## determinant and the inverse
    #invcovars = np.linalg.inv(covars)
    #expon = uTVu(deltas, invcovars)
    #term_expon = -0.5 * np.sum(expon)
    
    #lndets = np.log(np.linalg.det(covars))
    #term_dets = -0.5 * np.sum(lndets)

    term_expon, term_dets = lognormal(deltas, covars)
    
    return term_expon + term_dets

### log-likelihood for mixture model for outliers

def loglike_linear_unctyproj_outliers(pars, xypattern, xi, \
                                      xycovars, xicovars, \
                                      covout=np.eye(2), \
                                      calcResps=False):

    """Returns the log-likelihood for 6-term plane mapping, when both
(x,y) and (xi,eta) have uncertainty covariances, and we use a mixture
model to account for outliers. DOES NOT SUM ALONG N. Inputs:

    pars - [7] - parameter array, outlier fraction at the end

    xypattern = [N,2,6] - element pattern array in xy
    
    xi = [N,2] - N-element target positions (xi, eta)

    xycovars  -  [N,2,2] stack of covariance matrices in (x,y)

    xicovars -   [N,2,2] stack of covariance matrices in (xi, eta)
    
    covout - [N, 2,2] fixed covariance for the outlier points

    calcResps - if true, returns the log responsibilities for every object

    """

    # Separate out the pars from the mixture fraction
    parsmod = pars[0:-1]
    lnfout = pars[-1]

    fout = np.exp(lnfout)
    
    # Now compute the inlier and outlier likelihoods
    inliers = loglike_linear_unctyproj(parsmod, xypattern, xi, \
                                       xycovars, xicovars)
    outliers = loglike_linear(parsmod, xypattern, xi, \
                              covout)
    
    # prepare the mixture for logsumexp. Should be [k,N]
    b = np.ones( (2,xypattern.shape[0]) )
    b[0] = 1.0 - fout
    b[1] = fout

    loglikemix = logsumexp( np.vstack((inliers, outliers)), b=b, axis=0 )

    if calcResps:
        logResp_inlier = np.log(1.0-fout) + inliers - loglikemix
        return logResp_inlier, np.log(1.0-np.exp(logResp_inlier))    
   
    return loglikemix
    
def logprob_linear_unctyproj_unif(pars, xypattern, xi, xycovars, xicovars):

    """Returns the log posterior (to within a constant) for 6-term plane
mapping when both the (x,y) and (xi, eta) positions have uncertainty
covariances. Assumes uniform prior on the parameters.

    """

    lp = logprior_unif(pars)
    if not np.isfinite(lp):
        return -np.inf

    return lp + np.sum(loglike_linear_unctyproj(pars, xypattern, xi, \
                                         xycovars, xicovars) )
    

def logprob_linear_unctyproj_outliers_unif(pars, xypattern, xi, \
                                           xycovars, xicovars, \
                                           covout):

    """Returns ln(posterior) for 6-term mapping plus mixture fraction,
with uniform prior on the 6 parameters."""

    lp = logprior_unif_mixture(pars)
    if not np.isfinite(lp):
        return -np.inf

    
    ll =  loglike_linear_unctyproj_outliers(pars, xypattern, xi, \
                                            xycovars, xicovars, covout)

    return lp + np.sum(ll)
