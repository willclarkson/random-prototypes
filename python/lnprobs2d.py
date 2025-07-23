#
# lnprobs2d.py
#

#
# 2024-08-14 WIC - methods for computing ln(prob), to be used by MCMC
# and minimize.
#

import numpy as np

# Some pieces we don't need to reinvent
import mixfgbg
import noisemodel2d

# For specifying and using an informative (Gaussian) prior on some (or
# all) of the model parameters
from gaussianprior2d import gaussianprior

# For converting [a,b,c,d,e,f] to [a,d,sx,sy,theta,beta] in order to
# apply informative prior on any of those variables
import sixterm2d

def uTVu(u, V):

    """Returns u^T.V.u where
    
    u = [N,m] - N datapoints of dimension m (typically deltas array)

    V = [N,m,m] - N covariances of dimension m x m (an
    inverse-covariance stack)

    This will return an N-element array.

    """

    # Comment: in tests, this is faster than writing out the terms
    # explicitly and computing the (three-term) evaluation.
    
    Vu = np.einsum('ijk,ik -> ij', V, u)
    return np.einsum('ij,ji -> j', u.T, Vu)

class Prior(object):

    """Object and methods for computing ln(prior) for some parameter set"""

    def __init__(self, parset=None, path_informative_priors=''):

        # The parameter-set object
        self.parset=parset

        # The various parameters on which we may have priors
        self.model = np.array([])
        self.noise = np.array([])
        self.asymm = np.array([])
        self.mix = np.array([])

        # Options
        self.islog10_mix_frac = True
        self.islog10_mix_vxx = True
        self.islog10_noise_ryx = False
        self.islog10_noise_c = False

        # Which model parameters correspond to {a,b,c,d,e,f} in the
        # linear transformation?
        self.inds1d_6term = np.array([])

        # Distribute the parameters and options on initialization
        self.distributepars()

        # Informative (Gaussian) prior for zero or more
        # parameters. Because there are so many combinations of
        # possible parameters, we specify the input priors via
        # parameter file. Defaults to zeros (no informative
        # prior). For nuisance parameters like noise parameters, the
        # indices are specified via the "indices" attribute.
        self.gaussprior = \
            gaussianprior(pathpars = path_informative_priors, \
                          indices = self.parset.dindices)
        self.withgauss = len(self.gaussprior.lpars) > 0

        # Attribute to hold [xo, yo, sx, sy, theta, beta] if we are
        # using an informative prior on any of the model components
        self.geom6term = np.array([])
        
        # Some bounds for noise model parameters
        self.noise_min_loga = -50.
        self.noise_max_loga = 2.
        self.noise_min_logb = -50.
        self.noise_max_logb = 10.
        self.noise_min_c = 0.
        self.noise_max_c = 10.
        self.asymm_min_ryx = 0.
        self.asymm_max_ryx = 10.
        self.asymm_min_corr = -1. # WATCHOUT
        self.asymm_max_corr = 1.
        self.mix_min_fbg = 1.0e-5
        self.mix_max_fbg = 0.7 # WATCHOUT
        self.mix_max_vxx = 2.
        
        # the lnpriors for all parts of the model. Initialize to
        # zeros.
        self.lnprior_model = 0.
        self.lnprior_noise = 0.
        self.lnprior_asymm = 0.
        self.lnprior_mix = 0.
        self.sumlnprior = 0.

        # Evaluate the ln priors on initialization
        self.lnprior_transf_rect()
        # self.lnprior_gaussian_6term() # replaces lnprior_model
        self.lnprior_noisemodel_rect()
        self.lnprior_asymm_rect()
        self.lnprior_mixmod_rect()
        self.sumlnpriors()
        
    def distributepars(self):

        """Distributes parameters in our parset among the various subsets for
which we have parameters

        """

        if self.parset is None:
            return

        self.model = self.parset.model
        self.noise = self.parset.noise
        self.asymm = self.parset.symm
        self.mix = self.parset.mix

        self.islog10_mix_frac = self.parset.islog10_mix_frac
        self.islog10_mix_vxx = self.parset.islog10_mix_vxx

        if hasattr(self.parset, 'islog10_noise_c'):
            self.islog10_noise_c = self.parset.islog10_noise_c
        
    def lnprior_transf_rect(self):

        """Rectangular prior for transformation parameters"""

        self.lnprior_model = 0.

    def lnprior_gaussian_6term(self):

        """Applies gaussian prior to specified model parameters"""

        # adds the result to self.sumlnprior
        
        # first off, do we actually have informative priors on any of
        # the parameters?
        if not self.withgauss:
            return

        # if we got here, then we extract the parameters [x0, y0, sx,
        # sy, theta, beta] and send them into our gaussian prior
        # object. That object already knows which of these parameters
        # have valid informative priors (via its lpars attribute). So:
        abc = self.model[self.inds1d_6term]
        self.geom6term = sixterm2d.getpars(abc)

        self.sumlnprior = self.sumlnprior + \
            self.gaussprior.getlnprior(self.geom6term)

    def lnprior_gaussian(self):

        """Applies ln(prior) for the entire set of parameters, trusting the
gaussian prior object to identify which parameters have priors"""

        if not self.withgauss:
            return

        # We still need to convert the abcdef linear parameters to the
        # geometric terms expected by the prior. So:
        abc = self.parset.pars[self.inds1d_6term]
        geom6term = sixterm2d.getpars(abc)

        # now we feed ONLY the parameters on which we have
        # priors. This is still a little awkward, because gaussprior's
        # lpars_6term refers to the reordered [x0, y0, sx, sy, theta,
        # beta] while the lpars[not including 6term] refer to the
        # indices in the original array. For the moment, we just bring
        # them in separately.
        l6term = self.gaussprior.lpars_6term
        lother = self.gaussprior.lpars_nuisance

        # Initialize to blank
        pars4prior_6term = np.array([])
        pars4prior_nuisance = np.array([])

        if np.size(l6term) > 0:
            pars4prior_6term = geom6term[l6term]

        if np.size(lother) > 0:
            pars4prior_nuisance = self.parset.pars[lother]
            
        pars4prior = np.hstack(( pars4prior_6term, pars4prior_nuisance ))

        # Now we evaluate THIS prior:
        lnpriorgauss = self.gaussprior.getlnprior(pars4prior)
        self.sumlnprior = self.sumlnprior + lnpriorgauss
        
    def lnprior_noisemodel_rect(self):

        """Enforces positivity constraints etc. on the noise model. Prior is
uniform within the limits."""

        self.lnprior_noise = 0.
        sz = np.size(self.noise)
        
        if sz < 1:
            return

        loga = self.noise[0]
        if not (self.noise_min_loga < loga < self.noise_max_loga):
            self.lnprior_noise = -np.inf
            return

        if sz < 2:
            return

        logb = self.noise[1]
        if not (self.noise_min_logb < logb < self.noise_max_logb):
            self.lnprior_noise = -np.inf
            return

        if sz < 3:
            return

        c = self.noise[2]
        if self.islog10_noise_c:
            c = 10.0**self.noise[2]
            
        if not (self.noise_min_c < c < self.noise_max_c):
            self.lnprior_noise = -np.inf
            return

    def lnprior_asymm_rect(self):

        """ln(prior) on the noise asymmetry parameters [stdy/stdx, corrxy]"""

        self.lnprior_asymm = 0.
        
        sz = np.size(self.asymm)
        if sz < 1:
            return

        ryx = self.asymm[0]
        if not (self.asymm_min_ryx < ryx < self.asymm_max_ryx):
            self.lnprior_asymm = -np.inf
            return

        if sz < 2:
            return

        # x-y correlation.
        corr = self.asymm[1]
        if not (self.asymm_min_corr < corr < self.asymm_max_corr):
            self.lnprior_asymm = -np.inf
            return
        
    def lnprior_mixmod_rect(self):

        """Enforces positivity constraints on the mixture model parameters"""

        self.lnprior_mix = 0.
        if np.size(self.mix) < 1:
            return

        fbg, vxx = mixfgbg.parsemixpars(self.mix, \
                                        self.islog10_mix_frac, \
                                        self.islog10_mix_vxx, \
                                        vxxbg=0.)

        # Now we enforce the constraints. First, both must be
        # finite...
        if not np.isfinite(fbg) or not np.isfinite(vxx):
            self.lnprior_mix = -np.inf
            return

        # Avoid triggering bounds problems later on
        if fbg < self.mix_min_fbg or vxx > self.mix_max_vxx \
           or fbg > self.mix_max_fbg:
            self.lnprior_mix = -np.inf
            return

    def sumlnpriors(self):

        """Sums the ln priors"""

        self.sumlnprior = self.lnprior_model \
            + self.lnprior_noise \
            + self.lnprior_asymm \
            + self.lnprior_mix

    def updatepriors(self, parset=None):

        """Updates the parameters and recomputes"""

        if parset is None:
            return

        # update and re-distribute the parameters...
        self.parset = parset
        self.distributepars()

        # ... and recompute the priors
        self.lnprior_transf_rect()
        self.lnprior_noisemodel_rect()
        self.lnprior_asymm_rect()
        self.lnprior_mixmod_rect()
        self.sumlnpriors()

        # Incorporate the informative prior
        self.lnprior_gaussian()

        
class Like(object):

    """Object and methods to compute ln(likelihood)"""

    def __init__(self, parset=None, transf=None, obstarg=None):

        self.parset = parset
        self.transf = transf
        self.obstarg = obstarg

        # Transformed positions using the current parameters
        self.xytran = np.array([])

        # Target positions (which could be parameter-dependent)
        self.xytarg = np.copy(self.obstarg.xy)
        
        # Magnitude zeropoint to use when applying the noise model
        self.mag0 = 0.
        
        # Noise components
        self.covtarg = np.copy(self.obstarg.covxy)
        self.covtran = 0.
        self.covextra = 0.
        self.covoutly = self.covtarg * 0.

        # Fraction in foreground
        self.fbg = 0. 
        
        # Sum covariance, observed minus projected positions
        self.covsum = np.array([])
        self.dxy = np.array([])

        # Computed sum ln like and foreground/background components if
        # we're using a mixture model
        self.sumlnlike = -np.inf
        self.lnlike_fg = np.array([0.])
        self.lnlike_bg = np.array([-np.inf])

        # responsibilities (foreground)
        self.resps_fg = np.array([])
        
        # Compute lnlikelihoods on initialization
        self.updatetrans()
        self.unpackmixpars()
        self.calctrans()
        self.calclnlike()
        
        
    def updateparams(self, parset=None):

        """Updates the parameter set"""

        self.parset = parset
        
    def updatetrans(self):

        """Updates the transformation object with the paramset"""

        self.transf.updatetransf(self.parset.model)

    def unpackmixpars(self):

        """Translates the mixture parameters"""

        # Nothing to do if we aren't using a mixture
        if np.size(self.parset.mix) < 1:
            return
    
        # This is maybe a bit inefficient since this is already being
        # done upstram if the prior is evaluated first - which it
        # should be!
        fbg, vxx = mixfgbg.parsemixpars(self.parset.mix, \
                                        self.parset.islog10_mix_frac, \
                                        self.parset.islog10_mix_vxx, \
                                        vxxbg=0.)
        
        # Update the instance
        self.fbg = fbg

        # Slot the extra covariance into the outlier pieces - which
        # are already initialized to [N,2,2] * 0.
        self.covoutly *= 0.
        self.covoutly[:,0,0] = vxx
        self.covoutly[:,1,1] = vxx
        
    def tranposns(self):

        """Transforms positions using the current parameter set"""

        self.transf.tranpos()
        self.transf.xytran[:,0] = self.transf.xtran
        self.transf.xytran[:,1] = self.transf.ytran
        self.xytran = np.copy(self.transf.xytran)

        
    def trancovs(self):

        """Transforms covariances using the current parameter set"""

        self.transf.trancov()
        self.covtran = self.transf.covtran

    def calcextracovar(self):

        """Computes the extra covariance as a result of the noise model

        """

        parsnoise = self.parset.noise
        parsshape = self.parset.symm
        mags = self.obstarg.mags
        
        CC = noisemodel2d.mags2noise(parsnoise, parsshape, mags, \
                                     mag0=self.mag0, \
                                     islog10_c=self.parset.islog10_noise_c)

        self.covextra = CC.covars

    def sumcovars(self):

        """Sums the noise components"""

        self.covsum = self.covtarg + self.covtran + self.covextra

    def calcdeltas(self):

        """Computes the (projected minus observed) deltas

        """

        # self.dxy = self.xytran - self.obstarg.xy
        self.dxy = self.xytran - self.xytarg

    def updatesky(self, parset=None):

        """Updates the parameters, projects the observed positions and
covariances onto the target frame, and recomputes the sum covariance
and positional deltas for the current transformation and noise model
parameter set.

        """

        self.updateparams(parset)
        self.updatetrans()
        self.unpackmixpars()
        self.calctrans()
        
    def calctrans(self):

        """Wrapper - projects quantities onto target frame and computes the
deltas and projected summed covariances."""
        
        self.tranposns()
        self.trancovs()

        # If the transformation object also changes the observed
        # points, we need to propagate that so that the deltas will be
        # correct. For the moment we use a conditional since only one
        # of the transf methods actually does this. Consider adding a
        # dummy method to all the others so that the calls are
        # uniform.
        if hasattr(self.transf,'xytarg'):
            self.xytarg = self.transf.xytarg
        if hasattr(self.transf, 'covtarg'):
            self.covtarg = self.transf.covtarg
        
        self.calcextracovar()
        self.sumcovars()
        self.calcdeltas()

    def getlnlike(self, isbg=False):

        """Calculates ln(likelihood) from the projected positions and
covariances. Evaluates the logarithm of the gaussian badness-of-fit
statistic for each point, i.e.

    ln(like) = -ln(2pi) -0.5 ln(|V|) - 0.5( dx^T.V^{-1}.dx )

Inputs:
        
        isbg = control variable: is this component "background?" (If
        so, the background covariance is used in preference to the
        "data" covariance.)

This *returns* the lnlike vector, since it may be called in a mixture model.

Returns:

        lnlike = vector of ln-likelihood evaluations

        """

        # Which covariances are we using? (We add the outlier
        # covariance rather than switching it in to avoid running into
        # singular outlier covariance if the walker goes that way.)
        if isbg:
            covars = self.covsum + self.covoutly
            #covars = self.covoutly
        else:
            covars = self.covsum

        # 1. The exponent: invert the sum covariance array and find
        # the quantity u^T.V^{-1}.u
        invcov = np.linalg.inv(covars)
        expon = uTVu(self.dxy, invcov)
        term_expon = -0.5 * expon

        # 2. The determinant
        dets = np.linalg.det(covars)
        term_dets = -0.5 * np.log(dets)

        # 3. The -ln(2pi)
        term_2pi = term_dets * 0. - np.log(2.0*np.pi)

        # how large are these terms?
        # print(self.dxy[0], invcov[0][0,0], term_expon[0], term_dets[0], term_2pi[0], self.obstarg.isfg[0], isbg)
        
        return term_expon + term_dets + term_2pi

    def calclnlike(self):

        """Computes the ln(likelihood) in a mixture-aware way.

        See Equation (17) and nearby discussion in Hogg, Bovy & Lang (2010) https://arxiv.org/abs/1008.4686 """

        if self.fbg <= 0:
            lnlike = self.getlnlike(isbg=False)
            self.sumlnlike = np.sum(lnlike)

            # Update the foreground/background quantities
            self.lnlike_fg = lnlike
            self.lnlike_bg = -np.inf
            
            return

        # foreground, background components and their sum.
        self.lnlike_fg = self.getlnlike(isbg=False) + np.log(1.0-self.fbg)
        self.lnlike_bg = self.getlnlike(isbg=True) + np.log(self.fbg)
        self.sumlnlike = np.sum(np.logaddexp(self.lnlike_fg, self.lnlike_bg))

    def calcresps(self, strictlyfinite=True):

        """Calculates formal probability that each object belongs to the
foreground component. If there is no mixture, all points belong to the
foreground.

        Only points with finite lnlike(background) are included in the
        calculation. If strictlyfinite=True, then *all* objects must
        have finite lnlike(bg) for any of the responsibilities to be
        computed. While this is an option in this method, I think it
        should always be set to True so that the number of valid
        points is the same across all parameter-sets for the same
        dataset. That said, if there are any points with lnlike_bg =
        -np.inf, we are likely to see problems elsewhere.

Inputs:

        strictlyfinite = all objects must have finite
        lnlike(background).

        """

        # Initialize as all belonging to the foreground
        self.resps_fg = np.ones(self.obstarg.npts)
        if self.fbg <= 0.:
            return

        binf = np.isinf(self.lnlike_bg)
        if strictlyfinite and np.sum(binf) > 0:
                return

        self.resps_fg[~binf] = \
            np.exp(self.lnlike_fg[~binf] - \
                   np.logaddexp(self.lnlike_fg[~binf], self.lnlike_bg[~binf]) )
        
            
        
    def updatelnlike(self, parset=None):

        """One-liner to update the parameters and compute the sum
ln-likelihood with the new parameters"""

        self.updatesky(parset)
        self.calclnlike()
        
def lnprob(pars, transf, obstarg, parset=None, \
           lnprior=None, lnlike=None, return_blob=False):

    """ln(prob) for MCMC and minimizer.

Inputs:

    pars = [transf, noise, shape, mix] model parameters.

    transf = Transformation object, including the source positions and
    covariances

    obstarg = Obset object in the target frame. Must have at least
    .xy, .covxy and also .mags if magnitudes are going to be used.

    parset = Pars1d object. Contains methods for partitioning pars
    into the transformation, noise, shape, mixture as needed, and also
    is used to send in choices about whether the non-transformation
    parameters are in log10.

    lnprior = Prior object. Created if None is passed in, otherwise
    updated in-place.

    lnlike = Likelihood object. Created if None, otherwise updated
    in-place.

    return_blob = in addition to the lnposterior, returns lnlike and
    lnprior (as per emcee "blob" functionality). 

Returns:

    lnprob = ln(posterior probability)

    """

    # Ensure the current parameter set is propagated
    parset.updatepars(pars)

    # debug - what is the current paramset?
    ## print(pars, parset.pars)
    
    # Update the priors, returning if the parameter-set is out of
    # bounds
    if lnprior is None:
        lnprior = Prior(parset)
    else:
        lnprior.updatepriors(parset)

    if not np.isfinite(lnprior.sumlnprior):
        if return_blob:
            return -np.inf, -np.inf, -np.inf
        else:
            return -np.inf
        
    # compute log-likelihood with the parameters.
    if lnlike is None:
        lnlike = Like(parset, transf, obstarg)
    else:
        lnlike.updatelnlike(parset)

    if not np.isfinite(lnlike.sumlnlike):
        if return_blob:
            return -np.inf, -np.inf, -np.inf
        else:
            return -np.inf

    # WATCHOUT - lnlike.sumlnlike is a sum over the datapoints (taking
    # into account the model components including the mixture
    # fraction), while lnprior.sumlnprior is a sum over the model
    # components.
    # term_lnprior = lnprior.sumlnprior * obstarg.npts ## NO!!!
    term_lnprior = lnprior.sumlnprior
    term_lnlike = lnlike.sumlnlike
    
    # OK if we got here, then both the sum of ln(prior) and ln(like)
    # should be finite. Return it!
    if return_blob:

        # 2025-07-23 return the foreground lnlike separately as a
        # diagnostic
        return term_lnprior + term_lnlike, term_lnlike, \
            np.sum(lnlike.lnlike_fg)

        #return term_lnprior + term_lnlike, term_lnlike, term_lnprior

    
    
    # if here then we are not returning the blob.
    return term_lnprior + term_lnlike
