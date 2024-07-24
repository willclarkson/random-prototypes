#
# fittwod.py
#

#
# WIC 2024-07-24 - test-bed to use and fit transformation objects in
# unctytwod.py
# 

import time

import numpy as np
import matplotlib.pylab as plt

import unctytwod
from covstack import CovStack

# we want to draw samples
from weightedDeltas import CovarsNx2x2

# The minimizer
from scipy.optimize import minimize


def uTVu(u, V):

    """Returns u^T.V.u where
    
    u = [N,m] - N datapoints of dimension m (typically deltas array)

    V = [N,m,m] - N covariances of dimension m x m (an
    inverse-covariance stack)

    This will return an N-element array.

    """

    Vu = np.einsum('ijk,ik -> ij', V, u)
    return np.einsum('ij,ji -> j', u.T, Vu)

def sumlnlike(pars, transf, xytarg, covtarg):

    """Returns sum(log-likelihood) for a single-population model"""

    expon, det, piterm = lnlike(pars, transf, xytarg, covtarg)
    return np.sum(expon) + np.sum(det) + np.sum(piterm)

def lnlikestat(pars, transf, xytarg, covtarg):

    """Returns the sum of all three terms on a per-object basis"""

    expon, det, piterm = lnlike(pars, transf, xytarg, covtarg)

    return expon + det + piterm
    
def lnlike(pars, transf, xytarg=np.array([]), covtarg=np.array([]) ):

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
    covars = covtran + covtarg

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

def makefakedata(npts=2000, \
                 xmin=-10., xmax=10., ymin=-10., ymax=10.):

    """Utility to make unform random sampled datapoints"""

    xy = np.random.uniform(size=(npts,2))
    xy[:,0] = xy[:,0]*(xmax-xmin) + xmin
    xy[:,1] = xy[:,1]*(ymax-ymin) + ymin

    return xy

def makefakecovars(npts=2000, sigx=0.1, sigy=0.07, sigr=0.2):

    """Makes fake covariances"""

    vstdxi = np.ones(npts)*sigx
    vstdeta = vstdxi * sigy/sigx
    vcorrel = np.ones(npts)*sigr
    CS = CovarsNx2x2(stdx=vstdxi, stdy=vstdeta, corrxy=vcorrel)
    
    return CS

def testpoly(npts=2000, \
             deg=3, degfit=-1, \
             xmin=-2., xmax=2., ymin=-2., ymax=2., \
             sigx=0.001, sigy=0.0007, sigr=0.0, \
             polytransf='Polynomial', \
             polyfit='Polynomial', \
             covtranscale=1., \
             showpoints=True, \
             nouncty=False, \
             tan2sky=False, \
             alpha0=35., delta0=35.):

    """Creates and fits fake data: polynomials"""

    # fit degree?
    if degfit < 0:
        degfit = deg
    
    # What object are we using?
    if tan2sky:
        transf=unctytwod.Tan2equ
    else:
        transf=unctytwod.Poly
    
    # Make up some data and covariances in the source plane
    xy = makefakedata(npts, xmin, xmax, ymin, ymax)
    Cxy = makefakecovars(xy.shape[0], sigx, sigy, sigr)
    
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

        fig1.suptitle(ssup)
            
    # compare the input and fit parameters
    if tan2sky:
        print("Parameters comparison: alpha0, delta0")
        print('alpha_0: %.2f, diffnce %.2e "' % \
              (pars1d[0], (soln.x[0]-pars1d[0])*3.6e3))
        print('delta_0: %.2f, diffnce %.2e "' % \
              (pars1d[1], (soln.x[1]-pars1d[1])*3.6e3))

    else:
        # Show the polynomial parameters comparison:
        npars = np.max([np.size(parsx), np.size(parsxf)])
        print("Parameters comparison: X, Y")
        
        for ipar in range(npars):
            if ipar >= np.size(parsx):
                print("X: ########, %.2e -- Y: #########, %.2e" % \
                      (parsxf[ipar], parsyf[ipar]))
                continue

            if ipar >= np.size(parsxf):
                print("X: %.2e, ######## -- Y: %.2e, ######## " % \
                      (parsx[ipar], parsy[ipar]))
                continue
                
            print("X: %.2e, %.2e -- Y: %.2e, %.2e" % \
                  (parsx[ipar], parsxf[ipar], parsy[ipar], parsyf[ipar]))
