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

def parseParsAndCovars(pars):

    """Utility method - separates the parameters array into intrinsic
covariances and everything else.

    INPUT:

    pars -- [m] array of parameters

    OUTPUT:

    pars[0:-3] -- [m-3] model parameters (transformation, mixture)
    
    covint -- [2,2] intrinsic covariance array.

    By convention, the intrinsic covariances take up the last three
    entries of the parameters array, in the strict order
    [xx,yy,xy]. Any interpretation of the three unique covariance
    entries (as major, minor, position angle, for example) is deferred
    to other routines.

    """

    # Used because the calling syntax for lnprior must have the
    # parameters as a 1D array, but what we *do* with these model
    # parameters depends on which are transformation parameters and
    # which are intrinsic covariances that get added to every plane of
    # the [N,2,2] covariance matrix).
    
    covint = np.array( [[pars[-3], pars[-1]], [pars[-1], pars[-2]] ]) 

    return pars[0:-3], covint
    
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

    # The jacobian determinant is 1/(sxsy). So if there is a uniform
    # prior in the parameters (a,d,sx,sy,theta,beta) then the jac det
    # correction on prior(a,b,c,d,e,f) is 1/(sxsy), or on log(prior)
    # it is 0. - log(sx) - log(sy). Since sx = sqrt(b^2 + e^2) and sy
    # = sqrt(c^2 + f^2), we thus have log(prior on sx) - 0.5log(b^2 +
    # e^2) and similar for sy. So:
    
    sx2 = pars[1]**2 + pars[4]**2
    sy2 = pars[2]**2 + pars[5]**2

    lp = 0.-0.5*np.log10(sx2)-0.5*np.log10(sy2)

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

def logprior_unif_mixture(pars, imix=-1):

    """Returns uniform prior with conditions on ln(mixture fraction). That index is assumed the last one in pars unless changed with the argument imix."""

    if pars[imix] > 0.:
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

def loglike_linear_unctyproj(pars, xypattern, xi, xycovars, xicovars, \
                             covintrinsic=np.zeros((2,2))):

    """Returns the log-likelihood for 6-term plane mapping, when both the
(x,y) and (xi, eta) positions have uncertainty covariances. DOES NOT
SUM OVER N. Inputs:

    pars   -    [6] - parameter array

    xypattern = [N,2,6] - element pattern array in xy
    
    xi = [N,2] - N-element target positions (xi, eta)

    xycovars  -  [N,2,2] stack of covariance matrices in (x,y)

    xicovars -   [N,2,2] stack of covariance matrices in (xi, eta)

    covintrinsic - [2,2] intrinsic covariance array (in xi, eta)

    """

    # Project xy into the xi, eta plane and compute deltas
    deltas = xi - np.matmul(xypattern, pars)

    # Project the (x,y) covariances into the (xi, eta) plane
    covxy_proj = propagate_covars_abc(xycovars, pars)

    # Total covariance including (xi, eta) and projected (x,y)
    covars = xicovars + covxy_proj + covintrinsic

    # (Note: in the above line, we take advantage of numpy's broadcast
    # rules to handle the addition of the [2,2] intrinsic covariances
    # to every plane of the covars array (which are [N,2,2]).

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
    

def logprob_linr_unctyproj_intrns(pars, xypattern, xi, xycovars, xicovars):


    """Returns the log posterior (to within a constant) for 6-term plane
mapping when both the (x,y) and (xi, eta) positions have uncertainty
covariances, *and* the parameters include intrinsic variance to be
explored. Assumes uniform prior on the parameters.

    """

    # Comment 2024-02-26 - this may seem really inefficient to write
    # given that the method directly above is identical apart from the
    # method called. BUT: (i) I don't know if specifying the method as
    # an argument from a calling routine slows this down, and (ii) I
    # suspect we're going to want to use priors for the intrinsic
    # variance case anyway. For the moment, let's just get this to
    # work.
    
    lp = logprior_unif(pars)
    if not np.isfinite(lp):
        return -np.inf

    return lp + np.sum(ll_linr_unctyproj_intrns(pars, xypattern, xi, \
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


## 2024-02-26 - I'm not convinced it *is* possible to send in
## intrinsic covariances as a keyword argument to the same loglike
## routine that expects the parameters, since the intrinsic
## covariances come out of the parameters. For the moment, implement a
## separate method in each case (sigh):

def ll_linr_unctyproj_intrns(parsIn, xypattern, xi, xycovars, xicovars):

    """Like loglike_linear_unctyproj but with the last three of the
parameters being the intrinsic variance entries. Inputs as with loglke_linear_unctyproj except parsIn is extended by three elements."""

    pars, covInt = parseParsAndCovars(parsIn)

    return loglike_linear_unctyproj(pars, xypattern, xi, xycovars, xicovars, \
                                    covintrinsic=covInt)
    
    
def loglike_with_intrinsic(parsIn, argsIn, kwargs={}, \
                           methLike=loglike_linear_unctyproj):

    """Determine log-likelihood with intrinsic covariances as input model
parameters. Inputs:

    parsIn --  [m+3] - array of model parameters, including intrinsic covar

    argsIn -- arguments to be sent to the likelihood method

    kwargs -- any existing keyword arguments to send to the
    likelihood method (this method will add one keyword argument)

    methlike -- method used to determine the log-likelihood

    """

    # split the parameters into parameters, intrinsic covariances
    pars, covint = parseParsAndCovars(parsIn)

    # can we access the arguments?
    kwargs['covIntrinsic'] = covint

    
    
class IntrinsicCovar(object):

    """A collection of methods for reparameterization of intrinsic variance
described by Gaussian. Gathered into a class for notational
convenience. By convention, an aligned covariance matrix would have
entries [[a^2, 0], [0, b^2]], i.e. the diagonal entries are
VARIANCES. The position angle is entered and returned as degrees in
each case, converted internally to/from radians within each method.

    """

    
    def __init__(self, \
                 a=np.array([]), b=np.array([]), phideg=np.array([])):

        # The ellipse parameters: a, b, phi in degrees
        self.a = np.copy(a)
        self.b = np.copy(b)
        self.phi = np.copy(phideg)
                
        # The reparameterization: alpha, beta, ln(r)
        self.alpha = np.array([])
        self.beta = np.array([])
        self.lnr = np.array([])
        
        # The entries of the [(N,) 2,2] covariance matrix
        self.xx = np.array([])
        self.yy = np.array([])
        self.xy = np.array([])

    def alpha2ab(self, alpha, beta, lnr):

        """Computes covariance ellipse parameters (a, b, phi) from
reparameterization (alpha, beta, ln(r)). Returns phi in degrees.

        """

        a = np.sqrt(alpha**2 + beta**2)
        b = a * np.exp(lnr)
        phi = np.arctan2(beta, alpha)

        return a, b, np.degrees(phi)

    def alpha2xy(self, alpha, beta, lnr):

        """Computes covariance matrix entries from alpha, beta, ln(r)

        """

        r = np.exp(lnr)
        xx = alpha**2 + r**2 * beta**2
        yy = beta**2 + r**2 * alpha**2
        xy = alpha * beta * (1. - r**2)

        return xx, yy, xy

    def ab2alpha(self, a, b, phideg):

        """Converts ellipse parameters (a,b,phi) to reparameterization (alpha,
        beta, ln(r)). Expects phi in degrees.

        """

        # Convert phi to radians
        phi = np.radians(phideg)
        
        alpha = a * np.cos(phi)
        beta =  a * np.sin(phi)
        lnr = np.log(b/a)

        return alpha, beta, lnr

    def ab2xy(self, a, b, phideg):

        """Converts ellipse parameters (a,b,phi) to covariance matrix entries
(xx, yy, xy). Expects phi in degrees ccw from the x-axis.

"""

        # convert phi in degrees to radians
        phi = np.radians(phideg)
        
        # Might consider rewriting as fewer trig operations?
        xx = (a * np.cos(phi))**2 + (b * np.sin(phi))**2
        yy = (b * np.cos(phi))**2 + (a * np.sin(phi))**2
        xy = (a**2 - b**2) * np.cos(phi) * np.sin(phi)

        return xx, yy, xy

    def xy2ab(self, xxIn, yyIn, xyIn):

        """ 
        Converts covariance matrix entries (xx, yy, xy) to geometric ellipse parameters (a,b,phi)

        """

        # This does in an element-wise fashion the eigenvalue and
        # eigenvector computation for the (Nx)2x2 matrix formed from
        # xx, yy, xy. It might actually be faster to use the build-in
        # eigh, at the cost of programming effort rearranging the input into
        # (Nx)2x2. For the moment, just use the element-wise approach.

        # Ensure scalars --> arrays so we can treat special cases on a
        # uniform basis.
        xx = np.atleast_1d(xxIn)
        yy = np.atleast_1d(yyIn)
        xy = np.atleast_1d(xyIn)

        # Convert this into an [Nx2x2] stack so that we can use the
        # linalg methods we trust. This transposition is NOT a bug,
        # but shuffles the output so that we have our [N,2,2] that we
        # need.
        covar = np.array([[xx, xy],[xy, yy]]).T
        
        # OK now we extract the eigenvalues and eigenvectors of this
        # matrix stack. Input matrix is (by definition) symmetric, so
        # we can use eigh. Notes: 1. eigh returns the eigenvalues in
        # ASCENDING order, and 2. the eigenvalues are the VARIANCES,
        # so we need to take the sqrt to produce the a, b values we
        # want.
        evals, evecs = np.linalg.eigh(covar)
        
        # In each case, the position angle is just that of the
        # eigenvector corresponding to the major axis
        vecsMaj = evecs[:,:,-1]
        phideg = np.degrees(np.arctan2(vecsMaj[:,-1], vecsMaj[:,0]))

        # The major and minor axis lengths are the square roots of the
        # major eigenvalues in each case.
        a = np.sqrt(evals[:,-1])
        b = np.sqrt(evals[:,0])

        # We explicitly test whether input xx was a scalar so that we
        # can preserve any input [1] element inputs
        if np.isscalar(xxIn):
            a = a[0]
            b = b[0]
            phideg = phideg[0]

        return a, b, phideg

    def xy2alpha(self, xx, yy, xy):

        """Converts covariance matrix entries (xx, yy ,xy) to reparameterization (alpha, beta, ln(r). Calls xy2ab to convert to ellipse parameters first."""

        a, b, phideg = self.xy2ab(xx, yy, xy)
        alpha, beta, lnr = self.ab2alpha(a, b, phideg)

        return alpha, beta, lnr

    def populatefrom_ab(self, a, b, phideg):

        """ 
        Populates instance variables when given a, b, phideg. Useful when converting samples all at once after running a simulation.
        """

        self.a = np.copy(a)
        self.b = np.copy(b)
        self.phi = np.copy(phideg)

        # Populate the alpha, beta, ln r
        self.alpha, self.beta, self.lnr = \
            self.ab2alpha(self.a, self.b, self.phi)

        # populate the xx, yy, xy
        self.xx, self.yy, self.xy = \
            self.ab2xy(self.a, self.b, self.phi)

    def populatefrom_alpha(self, alpha, beta, lnr):

        """ 
        Populates covar parameters given alpha, beta, ln r parameterization
        """

        self.alpha = np.copy(alpha)
        self.beta = np.copy(beta)
        self.lnr = np.copy(lnr)

        self.a, self.b, self.phi = \
            self.alpha2ab(self.alpha, self.beta, self.lnr)

        self.xx, self.yy, self.xy = \
            self.alpha2xy(self.alpha, self.beta, self.lnr)

    def populatefrom_xy(self, xx, yy, xy):

        """Populates covar stack from covariance matrix entries"""

        self.xx = np.copy(xx)
        self.yy = np.copy(yy)
        self.xy = np.copy(xy)
        
        self.a, self.b, self.phi = \
            self.xy2ab(self.xx, self.yy, self.xy)

        self.alpha, self.beta, self.lnr = \
            self.ab2alpha(self.a, self.b, self.phi)

    def xy_as_x2(self, xx, yy, xy):

        """Convenience method to return individual covariance XY parameters
        as a(n Nx)2x2 array """

        return np.array([[xx, xy],[xy,yy]]).T

    def x2_as_xy(self, covar):

        """Convenience method - returns (Nx)2x2 as three separate vectors (or scalars if covar is a 2x2 matrix)"""

        if np.ndim(covar) < 3:
            return covar[0,0], covar[1,1], covar[0,1]

        return covar[:,0,0], covar[:,1,1], covar[:,0,1]
        
