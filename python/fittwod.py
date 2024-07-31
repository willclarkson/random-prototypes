#
# fittwod.py
#

#
# WIC 2024-07-24 - test-bed to use and fit transformation objects in
# unctytwod.py
# 

import os, time
from multiprocessing import cpu_count, Pool

import numpy as np
import matplotlib.pylab as plt

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
    
def skimvar(pars, nrows, npars=1, fromcorr=False, islog10=False):

    """Utility - if an additive scalar variance is included with the
parameters, split it off from the parameters, returning the parameters
and an [N,2,2] covariance matrix from the supplied extra variance.

    Returns: pars[M], covextra[2,2], cov_is_ok (Boolean)

    npars = number of parameters that are covariances.

    fromcorr [T/F]: additive variance is supplied as [stdx, stdy/stdx,
    corrcoef]. If False, is assumed to be [varx, vary, covxy].

    islog10 = additive variance is supplied as
    log10(variance). (WATCHOUT - this doesn't work as well in tests as
    islog10=False...)

    """

    parsmodel = pars[0:-npars]
    addvars = pars[-npars::]

    # Status flag. If we're doing additional translation of the input
    # covariance forms, we might violate the prior. Enforce that here.
    cov_ok = True
    
    if islog10:
        addvars = 10.0**addvars

    if fromcorr:

        # If we're building our covariance from [stdx, stdy/stdx,
        # corrxy], then we have bounds on all the parameters. If our
        # trial set violates those requirements, ensure the calling
        # routine is informed.

        # stdx must be >= 0 (I think >= and not >).
        if addvars[0] < 0:
            cov_ok = False
        
        # The stdy/stdx must be >0
        if addvars.size > 1:
            if addvars[1] <= 0.:
                cov_ok = False
        
        # if the correlation coefficient was supplied outside the
        # range [-1, +1], flag for the calling routine
        if addvars.size > 2:
            rho = addvars[-1]
            if rho < -1. or rho > +1.:
                cov_ok = False

        # print("skimvar DEBUG 1:", addvars)
                
        addvars = corr2cov1d(addvars)

        # print("skimvar DEBUG 2:", addvars)
        
    extracov = np.zeros((nrows, 2, 2))
    extracov[:,0,0] = addvars[0]

    # Populate the rest of the addvars entries
    if np.size(addvars) > 1:
        extracov[:,1,1] = addvars[1]
        if np.size(addvars) > 2:
            offdiag = addvars[2]
            extracov[:,0,1] = offdiag
            extracov[:,1,0] = offdiag
    else:
        extracov[:,1,1] = addvars[0]

    return parsmodel, extracov, cov_ok
    
def lnprior_unif(pars):

    """ln uniform prior"""

    return 0.

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
           addvar=False, nvar=1, fromcorr=False, \
           methprior=lnprior_unif, \
           methlike=sumlnlike):

    """Evaluates ln(posterior). Takes the method to compute the ln(prior)
and ln(likelihood) as arguments.

    addvar [T/F] = interpret the [-1]th parameter as extra variance to
    be added in both target dimensions equally.

    nvar = number of entries corresponding to covariance. Maximum 3,
    in the order [Vxx, Vyy, Vxy]

    fromcorr [T/F] = Any extra variance is supplied as [sx, sy/sx,
    rho] instead of [vx, vy, covxy].

    """

    pars = parsIn
    covextra = 0. 
    if addvar:
        pars, covextra, covok = \
            skimvar(parsIn, xytarg.shape[0], nvar, fromcorr)

        # If the supplied parameters led to an improper covariance
        # (correlation coefficient outside the range [-1., 1.], say),
        # then reject the sample.
        if not covok:
            return -np.inf
        
    # Evaluate the ln prior
    lnprior = methprior(pars)
    if not np.isfinite(lnprior):
        return -np.inf

    # evaluate ln likelihood
    lnlike = methlike(pars, transf, xytarg, covtarg, covextra) 

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

    print("=============================")
    print("makemagcovars DEBUG - inputs:")
    print(parsnoise, np.shape(parsnoise))
    print(mags.shape, np.min(mags), np.max(mags))
    print(parscorr, np.shape(parscorr))
    print("=============================")
    
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

def labelsaddvar(npars_extravar=0, extra_is_corr=False):

    """Utility - returns list of additional variable labels depending on
what we're doing

    """

    if npars_extravar < 1:
        return []
    lextra = [r'$V$']
    if extra_is_corr:
        lextra = [r'$s$']

    if npars_extravar < 2:
        return lextra

    lextra = [r'$V_{\xi}$', r'$V_{\eta}$']
    if extra_is_corr:
        lextra = [r'$s_{\xi}$', r'$s_{\eta}/s_{\xi}$']

    if npars_extravar < 3:
        return lextra

    lextra = [r'$V_{\xi}$', r'$V_{\eta}$', r'$V_{\xi \eta}$']
    if extra_is_corr:
        lextra = [r'$s_{\xi}$', r'$s_{\eta}/s_{\xi}$', r'$\rho_{\xi,\eta}$']

    return lextra

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
                    unctysrc=True, unctytarg=True, \
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
                    extra_is_corr=False, \
                    gen_noise_model=False, \
                    noise_mag_pars=[-4., -26., 2.5], \
                    noise_shape_pars=[0.7, 0.1], \
                    cheat_guess=False, \
                    maglo=16., maghi=20., magexpon=2.):

    """Tests the MCMC approach on a linear transformation.

    set doruns=True to actually do the runs.

    addvar = 1D variance to add to the datapoints in target space. FOr
    testing, 5.0e-12 seems sensible (it's about 10x the covtran)

    extra_is_corr --> extra covariance is modeled internally as [stdx,
    stdy/stdx, corrcoef]

    gen_noise_model --> generate uncertainties using noise model.

    noise_mag_pars [3] --> parameters of the magnitude dependence of
    the noise model stdx

    noise_shape_pars [2] --> noise model [stdy/stdx, corrxy].

    cheat_guess -- use the generating parameters as the guess.

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
        
    # Use the pattern object to make fake parameters for generating
    PM = Patternmatrix(deg, xy[:,0], xy[:,1], kind=polytransf, \
                       orderbypow=True)
    fpars = PM.getfakeparams(scale=scale, seed=seed, expfac=expfac)
    fparsx, fparsy = split1dpars(fpars)

    # Transform the truth postions and covariances
    PTruth = transf(xy[:,0], xy[:,1], Cxy.covars, fparsx, fparsy, \
                    kind=polytransf)

    PTruth.propagate()
    xytran = np.copy(PTruth.xytran)
    covtran = np.copy(PTruth.covtran)*covscale 

    # Index labeling
    #print("Poly labels: i", PTruth.pars2x.i)
    #print("Poly labels: j", PTruth.pars2x.j)

    #slabels = [r'a_{%i%i}' % \
    #    (PTruth.pars2x.i[count], PTruth.pars2x.j[count]) for count in range(PTruth.pars2x.i.size)]

    #print(slabels)
    
    #return
    
    # create covstack object from the target covariances so that we
    # can draw samples
    Ctran = CovarsNx2x2(covtran)

    # initialise the perturbations to zero
    nudgexy = xy * 0.
    nudgexytran = xytran * 0.

    if unctysrc:
        nudgexy = Cxy.getsamples()
    if unctytarg:
        nudgexytran = Ctran.getsamples()

    # If we are adding more noise, do so here
    nudgexyextra = xy * 0.
    npars_extravar = 1 # default, even if not used in the modeling
    if addvar:

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

    if cheat_guess:
        guess = np.copy(fpars)
    else:
        # Since our model is linear, we can use linear least squares to
        # get an initial guess for the parameters.

        # Weight by the inverse of the covariances (which we trust to
        # all be nonsingular). 
        wts = np.ones(xyobs.shape[0])
        if wtlsq:
            wts = wtsfromcovars(covtran, scalebydet=True)

            print("testmcmc_linear DEBUG: weights:", wts.shape)            
            detwts = np.linalg.det(wts)
            print("testmcmc_linear DEBUG: det(weights):", \
                  np.min(detwts), np.max(detwts), np.median(detwts) )
            
            
        LSQ = Leastsq2d(xyobs[:,0], xyobs[:,1], deg=degfit, w=wts, \
                        kind=polyfit, \
                        xytarg=xytarg)

        guess = LSQ.pars # We may want to modify or abut the guess.

    guessx, guessy = split1dpars(guess)

    # Now we arrange things for our mcmc exploration. The
    # transformation object...
    covsrc = Cxy.covars
    if forgetcovars or not unctysrc:
        covsrc *= 0.
    PFit = transf(xyobs[:,0], xyobs[:,1], covsrc, guessx, guessy, \
                  kind=polyfit)

    # ... and the arguments for ln(prob)
    # args = (PFit, xytarg, covtran)

    # Take a look at the data we generated... do these look
    # reasonable?
    fig1 = plt.figure(1, figsize=(8,5))
    fig1.clf()
    ax1=fig1.add_subplot(231)
    ax2=fig1.add_subplot(232)
    ax3=fig1.add_subplot(234)
    ax4=fig1.add_subplot(235)
    ax5=fig1.add_subplot(236)
    ax6=fig1.add_subplot(233)

    fig1.subplots_adjust(wspace=0.3, hspace=0.3, left=0.15, bottom=0.15)
    
    blah1=ax1.scatter(xy[:,0], xy[:,1], s=1)
    blah2=ax2.scatter(xyobs[:,0], xyobs[:,1], c='g', s=1)
    blah3=ax3.scatter(xytran[:,0], xytran[:,1], s=1)
    blah4=ax4.scatter(xytarg[:,0], xytarg[:,1], c='g', s=1)

    # Show apparent magnitudes. Rather than show the histogram, now
    # show std(x) vs mag, color coded by std(y)
    blah6=ax6.scatter(mags, np.sqrt(covtran[:,0,0]), \
                      c=np.log10(np.sqrt(covtran[:,1,1])), \
                      alpha=0.8, cmap='viridis', \
                      s=2)
    ax6.set_yscale('log')
    cb6 = fig1.colorbar(blah6, ax=ax6)
    #blah6=ax6.hist(mags, bins=25, alpha=0.5)
    
    # how about our initial guess parameters...
    PFit.propagate()
    blah5 = ax4.scatter(PFit.xytran[:,0], PFit.xytran[:,1], \
                        c='r', s=1)
    
    # Since we're simulating, we know what the generated parameters
    # were. Use this to plot the residuals under the truth parameters.
    fxy = PTruth.xytran - xytarg

    blah5 = ax5.scatter(fxy[:,0], fxy[:,1], s=.1)
    cc = np.cov(fxy, rowvar=False)
    sanno = "%.2e, %.2e, %.2e" % (cc[0,0], cc[1,1], cc[0,1])
    anno5 = ax5.annotate(sanno, (0.05,0.05), \
                         xycoords='axes fraction', \
                         ha='left', va='bottom', fontsize=6)

    # Enforce equal aspect ratio for the residuals axes
    ax5.set_aspect('equal', adjustable='box')
    
    ax1.set_title('Generated')
    ax2.set_title('Perturbed')
    ax3.set_title('Transformed')
    ax4.set_title('Target')
    ax5.set_title('Residuals, generated')
    ax6.set_title(r'Magnitude $m$')

    for ax in [ax1, ax2]:
        ax.set_xlabel(r'X')
        ax.set_ylabel(r'Y')

    for ax in [ax3, ax4]:
        ax.set_xlabel(r'$\xi$')
        ax.set_ylabel(r'$\eta$')

    ax5.set_xlabel(r'$\Delta \xi$')
    ax5.set_ylabel(r'$\Delta \eta$')

    ax6.set_xlabel(r'$m$')
    ax6.set_ylabel(r'$N(m)$')
    ax6.set_ylabel(r'$\sigma_\xi$')
    
    # Set up labels for plots
    slabelsx = [r'$a_{%i%i}$' % \
        (PTruth.pars2x.i[count], PTruth.pars2x.j[count]) for count in range(PTruth.pars2x.i.size)]
    slabelsy = [r'$b_{%i%i}$' % \
        (PTruth.pars2x.i[count], PTruth.pars2x.j[count]) for count in range(PTruth.pars2x.i.size)]
    slabels = slabelsx + slabelsy

    # If we added 1d variance, accommodate this here.
    if addvar:

        lextra = labelsaddvar(npars_extravar, extra_is_corr)
        
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

            print("testmcmc_linear INFO -  re-expressed vguess as [stdx, stdy/stdx, corrcoef]:")
            print("testmcmc_linear INFO - ", vguess)
            
        # Ensure the "truth" and guess parameters have the right
        # dimensions
        guess = np.hstack(( guess, vguess ))
        fpars = np.hstack(( fpars, var_extra )) 
        
    # now (drumroll) set up the sampler.
    methpost = lnprob
    args = (PFit, xytarg, covtran, addvar, npars_extravar, extra_is_corr)
    ndim = np.size(guess)

    # Try adjusting the guess scale to cover the offset between our
    # generated parameters and our guess, but not to swamp it
    if not cheat_guess:
        scaleguess = np.abs((guess-fpars)/guess)
    else:
        scaleguess = 1.0e-3
        
    print("testmcmc_linear INFO - |fractional offset| in guess:")
    print(scaleguess)
    print("^^^^^^")
    # with fake data, those are all VERY small - like 1e-7 to 1e-6
    # off. So our scaling is enormous. Try bringing the guessing way
    # down then.
    scaleguess *= 5. # scale up so we cover the interval and a bit more

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
                'guess':guess, 'basis':PFit.kind}
    
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
    
def showsamples(sampler, slabels=[], ntau=10, fpars=np.array([]), \
                guess=np.array([]), basis='', \
                flatfile='test_flatsamples.npy', \
                filfig3='test_thinned.png', \
                filfig2='test_allsamp.png', \
                filfig4='test_corner.png', \
                nminclose=20, burnin=-1):

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
    
    fig3 = plotsamplescolumn(flat_samples, 3, slabels=slabels)
    fig3.savefig(filfig3)
    if flat_samples.shape[-1] > nminclose:
        plt.close(fig3)
    
    # Try a corner plot
    fig4 = plt.figure(4, figsize=(9,7))
    fig4.clf()
    dum4 = corner.corner(flat_samples, labels=slabels, truths=fpars, \
                         truth_color='b', fig=fig4, labelpad=0.7, \
                         use_math_text=True)
    fig4.subplots_adjust(bottom=0.2, left=0.2)

    # set supertitle
    if len(basis) > 0:
        fig4.suptitle('Basis: %s' % (basis))
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

