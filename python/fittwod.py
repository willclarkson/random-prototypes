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

def lnprior_unif(pars):

    """ln uniform prior"""

    return 0.

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

def lnprob(pars, transf, xytarg, covtarg=np.array([]), \
           methprior=lnprior_unif, \
           methlike=sumlnlike):

    """Evaluates ln(posterior). Takes the method to compute the ln(prior)
and ln(likelihood) as arguments.
    """

    # Evaluate the ln prior
    lnprior = methprior(pars)
    if not np.isfinite(lnprior):
        return -np.inf

    # evaluate ln likelihood
    lnlike = methlike(pars, transf, xytarg, covtarg) 

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

def wtsfromcovars(covars=np.array([]) ):

    """Utility - returns inverse covars as weights, scaled to
median(det)=1

    """

    wraw = np.linalg.inv(covars)
    sfac = np.median(np.sqrt(np.linalg.det(wraw)))

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
                    nchains=32, chainlen=1000, ntau=10, \
                    checknudge=False, \
                    samplefile='testmcmc.h5', \
                    doruns=False, \
                    domulti=False):

    """Tests the MCMC approach on a linear transformation.

set doruns=True to actually do the runs.

    """

    # To check: are the perturbations actually applying the right
    # covariance?
    #
    # Can do this by plotting the deltas and finding the covariance,
    # since we're using the same covariance for all the input frame
    # here.
    
    # Fit degree, type
    if degfit < 0:
        degfit = deg
        
    if polyfit is None:
        polyfit = polytransf[:]
        
    # Transformation object, synthetic data
    transf = unctytwod.Poly
    xy = makefakedata(npts, xmin, xmax, ymin, ymax)
    Cxy = makefakecovars(xy.shape[0], sigx, sigy, sigr)

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

    xyobs  = xy + nudgexy
    xytarg = xytran + nudgexytran

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
    
    # Since our model is linear, we can use linear least squares to
    # get an initial guess for the parameters. Go unweighted.
    LSQ = Leastsq2d(xyobs[:,0], xyobs[:,1], deg=degfit, kind=polyfit, \
                    xytarg=xytarg)

    guess = LSQ.pars # in case we want to do things to guess
    guessx, guessy = split1dpars(guess)

    # Now we arrange things for our mcmc exploration. The
    # transformation object...
    covsrc = Cxy.covars
    #if not unctysrc:  ### WATCHOUT
    #    covsrc *= 0.
    PFit = transf(xyobs[:,0], xyobs[:,1], covsrc, guessx, guessy, \
                  kind=polyfit)

    # ... and the arguments for ln(prob)
    args = (PFit, xytarg, covtran)

    # Take a look at the data we generated... do these look
    # reasonable?
    fig1 = plt.figure(1, figsize=(5,5))
    fig1.clf()
    ax1=fig1.add_subplot(221)
    ax2=fig1.add_subplot(222)
    ax3=fig1.add_subplot(223)
    ax4=fig1.add_subplot(224)

    blah1=ax1.scatter(xy[:,0], xy[:,1], s=1)
    blah2=ax2.scatter(xyobs[:,0], xyobs[:,1], c='g', s=1)
    blah3=ax3.scatter(xytran[:,0], xytran[:,1], s=1)
    blah4=ax4.scatter(xytarg[:,0], xytarg[:,1], c='g', s=1)

    # how about our initial guess parameters...
    PFit.propagate()
    blah5 = ax4.scatter(PFit.xytran[:,0], PFit.xytran[:,1], \
                        c='r', s=1)
    
    ax1.set_title('Generated')
    ax2.set_title('Perturbed')
    ax3.set_title('Transformed')
    ax4.set_title('Target')

    for ax in [ax1, ax2]:
        ax.set_xlabel(r'X')
        ax.set_ylabel(r'Y')

    for ax in [ax3, ax4]:
        ax.set_xlabel(r'$\xi$')
        ax.set_ylabel(r'$\eta$')

    # now (drumroll) set up the sampler:
    methpost = lnprob
    ndim = np.size(guess)

    # set up the walkers, each with perturbed guesses
    pertn = np.random.randn(nchains, np.size(guess))
    magn  = 0.02 * guess  # was 0.01
    pos = guess + pertn * magn[np.newaxis,:]
    nwalkers, ndim = pos.shape

    print("INFO: pos", pos.shape)
    print("nwalkers, ndim", nwalkers, ndim)

    # Set up labels for plots
    slabelsx = [r'$a_{%i%i}$' % \
        (PTruth.pars2x.i[count], PTruth.pars2x.j[count]) for count in range(PTruth.pars2x.i.size)]
    slabelsy = [r'$b_{%i%i}$' % \
        (PTruth.pars2x.i[count], PTruth.pars2x.j[count]) for count in range(PTruth.pars2x.i.size)]
    slabels = slabelsx + slabelsy

    
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

    # if multiprocessing, then we'll want to run from the python
    # interpreter.
    if domulti:

        # Could wrap the returns into an object for clarity?

        # Watchout - the backend may need to be set at the
        # interpreter. Test this!
        print("Returning arguments for multiprocessing:")
        print("nwalkers, ndim, methpost, args, pos, chainlen, slabels, fpars, guess")

        print("Now run:")
        print("sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)")
        print("sampler.run_mcmc(initial, nsteps, progress=True)")
        print("fittwod.showsamples(sampler, slabels, fpars, guess")
        return nwalkers, ndim, methpost, args, pos, chainlen, slabels, fpars, guess

    
    # Run without multiprocessing
    sampler = emcee.EnsembleSampler(nwalkers, ndim, \
                                    methpost, \
                                    args=args, \
                                    backend=backend)

    t0 = time.time()
    sampler.run_mcmc(pos, chainlen, progress=True);
    t1 = time.time()
        
    print("testmcmc INFO - samples took %.2e seconds" % (t1 - t0))

    showsamples(sampler, slabels, ntau, fpars, guess)
    
def showsamples(sampler, slabels=[], ntau=10, fpars=np.array([]), \
                guess=np.array([]) ):

    """Ported the methods to use the samples into a separate method so
that we can run this from the interpreter."""
    
    # look at the results
    samples = sampler.get_chain()
    
    print("SAMPLES INFO - SAMPLES:", np.shape(samples))

    # Plot the unthinned samples
    fig2 = plotsamplescolumn(samples, 2, slabels=slabels)
    
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

    flat_samples = sampler.get_chain(discard=nThrow, thin=nThin, flat=True)
    print("FLAT SAMPLES INFO:", flat_samples.shape, nThrow, nThin)

    fig3 = plotsamplescolumn(flat_samples, 3, slabels=slabels)

    # Try a corner plot
    fig4 = plt.figure(4, figsize=(9,7))
    fig4.clf()
    dum4 = corner.corner(flat_samples, labels=slabels, truths=fpars, \
                         truth_color='b', fig=fig4, labelpad=0.7, \
                         use_math_text=True)
    fig4.subplots_adjust(bottom=0.2, left=0.2)

    # set supertitle
    # fig4.suptitle(polyfit)   # need this to get passed
    
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
