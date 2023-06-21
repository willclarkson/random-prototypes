#
# likesMCMC.py
#

# WIC 2023-06-19 -- collection of loglikes and priors for use in MCMC
# explorations.

import numpy as np

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

    print("PROPAG INFO:", J)
    
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

################ log-likelihood and priors follow #################

def loglike_linear(pars, xypattern, xi, invcovars):

        """Returns the log-likelihood for the M-term (6-term linear or 4-term
similarity) plane mapping. Covariances are in the (xi, eta) plane and
are assumed independent of the model parameters (i.e. assume x,y
covariances are unimportant).

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
        
def logprior_unif(pars):

    """Uniform prior. Takes arguments for consistency with other calls"""

    return 0.

def logprob_linear_unif(pars, xypattern, xi, invcovars):

    """ 
    Computes log posterior (to within a constant) for the linear mapping, with a uniform prior on the parameters

"""

    # Thought: we probably could pass the methods in to this object?
    
    lp = logprior_unif(pars)
    if not np.isfinite(lp):
        return -np.inf

    return lp + loglike_linear(pars, xypattern, xi, invcovars)

### Uncertainties in both source and destiation coordinates

# (Will need a routine to project the uncertainties into the target
# frame. Do that next.)
