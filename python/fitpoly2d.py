#
# fitpoly2d.py
#

#
# 2024-07-25 WIC - perform linear least squares fitting of optionally
# weighted 2D data pairs, using numpy's polynomial classes for the
# bases
#

import time
import numpy as np
import matplotlib.pylab as plt

from numpy import polynomial

class Patternmatrix(object):

    """Sets up the pattern matrix for linear least squares fitting to
linear model

    """

    def __init__(self, deg=2, x=np.array([]), y=np.array([]), \
                 kind='Polynomial', norescale=False, \
                 xmin=-1., xmax=1., ymin=-1., ymax=1., \
                 seed=None, Verbose=True):

        # degree
        self.deg = deg

        # input points
        self.x = np.copy(x)
        self.y = np.copy(y)
        
        # control variable for rescaling
        self.norescale = norescale
        self.xmin = xmin # default leads to no rescaling
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        # For methods that use randomness, set the seed
        self.seed = seed

        # Control variable for screen output
        self.Verbose = Verbose
        
        # points: x, y --> xr, yr 
        self.setlims()
        self.rescalexy()
        
        # Method selection
        self.kind=kind[:]
        self.methvander = polynomial.polynomial.polyvander2d
        self.meths = \
            { 'Polynomial':polynomial.polynomial.polyvander2d, \
              'Chebyshev':polynomial.chebyshev.chebvander2d, \
              'Legendre':polynomial.legendre.legvander2d, \
              'Hermite':polynomial.hermite.hermvander2d, \
              'HermiteE':polynomial.hermite_e.hermevander2d}

        self.setmethvander()
                
        # powers following the convention of numpy's polynomial model
        self.ipow = np.array([])
        self.jpow = np.array([])
        self.bpow = np.array([]) # boolean for powers <= deg

        # selected indices
        self.isel = np.array([])
        self.jsel = np.array([])

        # selected indices in the vandermonde array
        self.lvander = np.array([])
        
        # vandermonde array, pattern matrix
        self.vander = np.array([])
        self.pattern = np.array([])
        
        # set the indices
        self.buildpowers()
        self.selectpowers()

        # build the vandermonde array and the pattern matrix
        self.buildvander()
        self.buildpattern()

        # it's useful to be able to access the debug figure from
        # outside the instance
        self.fignum = 1

    def setlims(self):

        """Sets the domain limits for rescaling"""

        # Don't reset the limits if we're not rescaling
        if self.norescale:
            return
        
        self.xmin = np.min(self.x)
        self.xmax = np.max(self.x)

        self.ymin = np.min(self.y)
        self.ymax = np.max(self.y)
        
    def rescalexy(self):

        """Utility - rescales x, y to the [-1,1] domain"""

        self.xr = (2.0*self.x - (self.xmax + self.xmin))\
            /(self.xmax - self.xmin)
        self.yr = (2.0*self.y - (self.ymax + self.ymin))\
            /(self.ymax - self.ymin)
        
        
    def setmethvander(self):

        """Selects the method for the vandermonde matrix"""

        # Set a default if the selected method is not in the allowed
        # list
        if not self.kind in self.meths.keys():
            self.kind = 'Polynomial'

        self.methvander = self.meths[self.kind]
            
    def buildpowers(self):

        """Sets up the powers arrays from the degree"""

        vpow = np.arange(self.deg+1)
        self.ipow = np.repeat(vpow, self.deg+1)
        self.jpow = np.tile(vpow, self.deg+1)

    def selectpowers(self):

        """Sets boolean for powers of the proper degree"""

        self.bpow = self.ipow + self.jpow <= self.deg    

        self.isel = self.ipow[self.bpow]
        self.jsel = self.jpow[self.bpow]

        ldum = np.arange(np.size(self.ipow))
        self.lvander = ldum[self.bpow]
        
    def buildvander(self):

        """Makes vandermonde array"""

        # Note that this works on the rescaled x, y
        
        if np.size(self.xr) < 1:
            return
        
        deg2 = (self.deg, self.deg)
        self.vander = self.methvander(self.xr, self.yr, deg2)

    def initpattern(self):

        """Initialize the pattern matrix"""

        npoints = np.size(self.xr)
        ncols = 2*np.sum(self.bpow)
        nrows = 2

        self.pattern = np.zeros(( npoints, nrows, ncols))
        
    def buildpattern(self):

        """Builds the pattern matrix given the selected powers"""

        self.initpattern()
        nbases = np.size(self.lvander)

        self.pattern[:,0, 0:nbases] = self.vander[:,self.lvander]
        self.pattern[:,1, nbases::] = self.vander[:,self.lvander]

    def getfakeparams(self, scale=1., seed=None, expfac=1.):

        """Utility - generate random fake parameters following the conventions
of the pattern matrix. Parameters are returned rather than getting
passed up to the instance to avoid confusion later on.

        """

        # the powers array must be set up first
        if np.size(self.isel) < 1:
            if self.Verbose:
                print("Patternmatrix.getfakeparams WARN - array self.isel not set")                
            return

        # remember, this is TWO dimensional data, so the powers arrays
        # are repeated. tile not repeat!! 
        sumpow = np.tile(self.isel + self.jsel, 2)
        npatt = np.size(sumpow)
        
        # Set up the generator and draw from it. Either inherit or
        # pass up the seed
        if seed is None:
            seed = self.seed
        else:
            self.seed = seed
            
        rng = np.random.default_rng(seed=seed)
        sampls = rng.uniform(-1., 1., npatt)
        
        # find the maximum power for each entry (we'll use this to
        # scale the parameters)
        sfacs = 10.0**(expfac*(0.-sumpow))
        
        pars = sampls * sfacs * scale

        return pars
        
        
    def showbases(self, fignum=1, showcolorbar=False, \
                  showindices=True, fsz=6, cmap='viridis'):

        """Shows the bases"""

        # record the figure number in the instance so that we can
        # access it later
        self.fignum = fignum
        
        # index for subplots
        ldum = np.arange(np.size(self.ipow))
        lplot = ldum[self.bpow]+1

        # scale the fontsize by the degree
        fontsize=9
        if self.deg > 7:
            fontsize=6
        
        fig1=plt.figure(fignum, figsize=(fsz,fsz))
        fig1.clf()
        axes = []
        for iax in range(np.size(lplot)):
            ax = fig1.add_subplot(self.deg+1, self.deg+1, lplot[iax])

            basis = self.pattern[:, 0, iax]
            dum = ax.scatter(self.xr, self.yr, c=basis, s=2, cmap=cmap)

            # hide the vertical axis if j < 1
            if self.isel[iax] < self.deg:            
                ax.get_yaxis().set_ticklabels([])            
            if self.jsel[iax] < 1:
                ax.set_ylabel('Y', rotation=0.)
                
            if self.jsel[iax] + self.isel[iax] < self.deg:
                ax.get_xaxis().set_ticklabels([])
            else:
                ax.set_xlabel('X')
                
            # show which term this is
            if showindices:
                stitl = '%i,%i' % (self.isel[iax], self.jsel[iax])
                dum = ax.annotate(stitl, (0.05,0.95), \
                                  fontsize=fontsize, \
                                  zorder=25, \
                                  ha='left', va='top', \
                                  xycoords='axes fraction', \
                                  color='w')
    
            # add a colorbar
            if showcolorbar:
                cb = fig1.colorbar(dum, ax=ax)

        # tighten up the panels
        fig1.subplots_adjust(hspace=0.02, wspace=0.02, \
                             left=0.1, right=0.9, \
                             bottom=0.1, top=0.9)

        # supertitle
        ssup = '%s (degree %i)' % (self.kind, self.deg)
        fig1.suptitle(ssup)

class Leastsq2d(object):

    """Fits linear least squares transformation to (optionally) weighted
data, 2d input, 2d output"""

    def __init__(self, x=np.array([]), y=np.array([]), w=np.array([]), \
                 kind='Polynomial', deg=2, norescale=False, \
                 xytarg = np.array([])):

        # input data
        self.x = x
        self.y = y

        # Weights
        self.W = np.array([])
        self.initweights()
        self.parseweights(w)

        # x, y in target space
        self.xytarg = np.copy(xytarg)
        
        # Fit function information
        self.fitkind = kind[:]
        self.fitdeg = deg
        self.fitnorescale = norescale

        # Pattern matrix object
        self.P = np.array([])
        self.pattern = None
        self.setpatternmatrix()

        # The Normal Equations terms: beta, hessian
        self.beta = np.array([])  # [npars]
        self.H = np.array([])     # [npars, npars]

        # Sets the hessian from the pattern matrix
        self.H = np.array([])
        self.sethessian()

        # lhs of normal equations
        self.beta = np.array([])
        self.setbeta()

        # fitted parameters. Tries to solve on initialization
        self.pars = np.array([])
        self.solvepars()
        
    def initweights(self):

        """Initializes weights array using input dataset as a guide"""

        # If the input data has zero size, weights have no meaning. We
        # want operations on those weights to break later on (or at
        # least trigger exceptions!). So:
        npts = np.size(self.x)
        if npts < 1:
            return

        self.W = np.zeros((npts, 2, 2))
        self.W[:,0,0] = 1.
        self.W[:,1,1] = 1.
        
    def parseweights(self, wts=np.array([]), Verbose=True):

        """Parses input weights into an [N,2,2] array"""

        # Initialize the weights to [N,2,2] identity
        self.initweights()

        # If weights isn't even [2,2], nothing to use it for.
        if np.size(self.W) < 4:
            return

        # If the weights array is not three-dimensional for some
        # reason, return
        if np.size(np.shape(self.W)) != 3:
            return

        if Verbose:
            print("Leastsq2d.parseweights INFO:", self.W.shape, np.shape(wts))
        
        # Now we handle a few cases:

        # 0. no weights provided. Stick with the identity initialized
        if np.size(wts) < 1:
            return
        
        # 1. wts = 42.        
        if np.isscalar(wts):
            self.W[:,0,0] = wts
            self.W[:,1,1] = wts
            return

        # 2. wts = [1d]
        if np.ndim(wts) == 1:
            # Given [scalar] for some reason
            if np.size(wts) < 2:
                self.W[:,0,0] = wts[0]
                self.W[:,1,1] = wts[0]
                return

            # weights as [wx, wy] scalars
            if np.size(wts) == 2:
                self.W[:,0,0] = wts[0]
                self.W[:,1,1] = wts[1]
                return
                
            # weights as [wx, wy, wxy] scalars
            if np.size(wts) == 3:
                self.W[:,0,0] = wts[0]
                self.W[:,1,1] = wts[1]
                self.W[:,0,1] = wts[2]
                self.W[:,1,0] = wts[2]
                return
                
            # Diagonal weights [N]
            if np.size(wts) == self.W.shape[0]:
                self.W[:,0,0] = wts
                self.W[:,1,1] = wts
                return

        # 3. wts = [N,1], [N,2] or [N,3]
        if np.dim(wts) == 2:
            wshape = wts.shape

            # if 2x2 array
            if wshape[0] == 2 and wshape[-1] == 2:
                self.W[:,0,0] = wts[0,0]
                self.W[:,1,1] = wts[1,1]
                self.W[:,0,1] = wts[0,1]
                self.W[:,1,0] = wts[1,0]
                return
            
            # if N-length, can unpack this if [N,1], [N,2] or [N,3]
            if wshape[0] == self.W.shape[0]:

                # [N,1]
                if wshape[-1] == 1:
                    w1d = wts.squeeze()
                    self.W[:,0,0] = w1d
                    self.W[:,1,1] = w1d
                    return

                # [N,2]
                if wshape[-1] == 2:
                    self.W[:,0,0] = wts[:,0]
                    self.W[:,1,1] = wts[:,1]
                    return

                # [N,3]
                if wshape[-1] == 3:
                    self.W[:,0,0] = wts[:,0]
                    self.W[:,1,1] = wts[:,1]
                    self.W[:,0,1] = wts[:,2]
                    self.W[:,1,0] = wts[:,2]
                    return

        # 4. wts = [N,2,2] - in this case just copy the weights straight in
        #
        # Note we can compare the shapes directly
        if wts.shape == self.W.shape:
            self.W = np.copy(wts)
            return

    def setpatternmatrix(self):

        """Sets up the pattern matrix object"""

        self.pattern = Patternmatrix(self.fitdeg, self.x, self.y, \
                                     self.fitkind, norescale=self.fitnorescale)

        # sets up the P matrix itself
        self.P = self.pattern.pattern

    def sethessian(self):

        """Sets the Hessian matrix H = Sum_i P^T_i W_i P_i

        H has dimensions [npars, npars]."""

        # Some sanity checking
        if np.size(self.P) < 1 or np.size(self.W) < 1:
            return
        
        H3d = self.getH3d()
        self.H = np.sum(H3d, axis=0)

    def setbeta(self):

        """Sets the lhs of the normal equations, beta = Sum_i (P_i^T W_i eps_i)

        beta has dimensions [npars]

        """

        # nothing to do if the required inputs haven't been set yet
        if np.size(self.P) < 1 or np.size(self.W) < 1 or np.size(self.xytarg) < 1:
            return
        
        beta2d = self.getbeta2d()
        self.beta = np.sum(beta2d, axis=0)
        
    def getH3d(self):

        """Returns the right hand side of the normal equations: H_i = P^T_i
W_i P_i . Does not sum along the data axis.

        Output will be [N,npars, npars]

        """

        PT = np.transpose(self.P, axes=(0,2,1))
        WP = np.matmul(self.W, self.P)
        return np.matmul(PT, WP)
        
    def getbeta2d(self):

        """Returns the left hand side of the normal equations: beta_i = P_i^T
W_i eps_i , without summing along the data axis. 

        Output will be [N,npars]

        """

        PT = np.transpose(self.P, axes=(0,2,1))
        WE = np.matmul(self.W, self.xytarg[:,:,np.newaxis]).squeeze()
        return np.matmul(PT, WE[:,:,np.newaxis]).squeeze()

    def solvepars(self):

        """Solves the normal equations beta = H.pars to get pars"""

        # Quantities must be populated
        if np.size(self.H) < 1 or np.size(self.beta) < 1:
            return
        
        self.pars = np.linalg.solve(self.H, self.beta)

    def ev(self, x=np.array([]), y=np.array([]) ):

        """Utility - evaluates the best-fit polynomial"""

        # if no points given, use the instance points
        if np.size(x) < 1 or np.size(y) < 1:
            x = np.copy(self.x)
            y = np.copy(self.y)
        
        # Make the pattern matrix with the input data. Use the same
        # rescale limits as the pattern matrix that was used to fit
        # the data in the first place
        pp = Patternmatrix(self.fitdeg, x, y, \
                           self.fitkind, norescale=self.fitnorescale, \
                           xmin=self.pattern.xmin, \
                           xmax=self.pattern.xmax, \
                           ymin=self.pattern.ymin, \
                           ymax=self.pattern.ymax)

        return np.matmul(pp.pattern, self.pars)
        
        
### utility methods

def gridxy(sidelen=1., nside=31, llzero=False, nfine=None):

    """Constructs regular grid of xy points"""

    # unctytwod.py has a more sophisticated version of this method

    if nfine is None:
        nfine = nside
    
    xv = np.linspace(0.-sidelen, sidelen, nside, endpoint=True)
    yv = np.linspace(0.-sidelen, sidelen, nfine, endpoint=True)

    print("INFO:", xv.shape, yv.shape)
    
    if llzero:
        xv -= np.min(xv)
        yv -= np.min(yv)

    xx, yy = np.meshgrid(xv, yv)
    xi = np.ravel(xx)
    eta = np.ravel(yy)
    
    # double up if nside and nfine are different
    if nfine != nside:
        xi = np.hstack(( xi, np.ravel(yy) ))
        eta = np.hstack(( eta, np.ravel(xx) ))
        
    return xi, eta

### Test methods follow
def testfit(sidelen=1., nside=41, llcorner=False, seed=None, deg=2, kind='Polynomial'):

    """Tests the fitting by linear least squares"""

    # Generate data
    x, y = gridxy(sidelen, nside, llcorner)

    # set up fit object
    LSQ = Leastsq2d(x, y, deg=deg, kind=kind)
    
    # make some fake parameters, use them to generate the target x, y
    # positions
    fpars = LSQ.pattern.getfakeparams(scale=10., seed=seed)
    LSQ.xytarg = np.matmul(LSQ.P, fpars)

    # populate the normal equations and do the fit
    LSQ.setbeta()
    LSQ.solvepars()

    print("Madeup params:", fpars)
    print("Fitted params:", LSQ.pars)

def testoneline(sidelen=1., nside=41, llcorner=False, \
                gendeg=2, genkind='Polynomial', \
                fitdeg=None, fitkind=None, \
                scale=1., seed=None, \
                showbasis=True, cmap='inferno', \
                showproj=True, \
                nfine=None, unweighted=True, \
                expfac=1., quivquant=0.95):

    """Tests one-line version of fitter. Example call:


    fitpoly2d.testoneline(gendeg=3, genkind='Legendre', seed=123123, fitkind='Chebyshev', fitdeg=3, sidelen=1., nside=41, showbasis=True, showproj=False)
    
    fitpoly2d.testoneline(gendeg=3, genkind='Legendre', seed=123123, fitkind='Chebyshev', fitdeg=3, sidelen=1., nside=11, nfine=41, showbasis=False, showproj=True)

    fitpoly2d.testoneline(gendeg=5, genkind='Legendre', showbasis=False, seed=123123, fitkind='Legendre', fitdeg=5, sidelen=5., nside=11, nfine=41, expfac=.7)


    """

    # We generate the pattern matrix first, then do the leastsq
    # fitting on initialization

    # Generate data
    x, y = gridxy(sidelen, nside, llcorner, nfine=nfine)

    # generate pattern matrix object so that we can generate target points
    PM = Patternmatrix(gendeg, x, y, kind=genkind)
    fpars = PM.getfakeparams(scale=scale, seed=seed, expfac=expfac)
    xytarg = np.matmul(PM.pattern, fpars)

    # Allow fitting kind and degree to differ from the generating kind
    # and degree
    if fitdeg is None:
        fitdeg = gendeg

    if fitkind is None:
        fitkind = genkind

    # set up some weights
    wts = np.array([])

    # Try random-weighted data
    if not unweighted:
        wts = np.random.uniform(0., 1., size=x.size)
    
    # OK now we can try our fitter in one line
    LSQ = Leastsq2d(x, y, w=wts, deg=fitdeg, kind=fitkind, xytarg=xytarg)

    print("Generated pars: ", fpars)
    print("LSQ fitted pars:", LSQ.pars)

    # evaluate the best-fit model
    xyeval = LSQ.ev()

    print("LSQ eval:", xyeval.shape)

    # plot the projection to see how well this actually did...
    if showproj:
        fig2=plt.figure(2, figsize=(8,4))
        fig2.clf()
        ax1 = fig2.add_subplot(121)
        ax2 = fig2.add_subplot(122)
        
        # Projected space indicating the generated and fit patterns
        sgen = 'gen: %s(%i)' % (PM.kind, PM.deg)
        sfit = 'fit: %s(%i)' % (LSQ.pattern.kind, LSQ.pattern.deg)        
        blah1a = ax1.scatter(xytarg[:,0], xytarg[:,1], s=1, c='b', label=sgen, marker='s', alpha=0.7)
        blah1b = ax1.scatter(xyeval[:,0], xyeval[:,1], s=1, c='r', label=sfit, marker='o', alpha=0.7)

        ax1.set_xlabel(r'$\xi$')
        ax1.set_ylabel(r'$\eta$')        
        leg = ax1.legend(fontsize=8)
        ax1.set_title('Projected')

        # Generated space indicating the deltas
        dxi = xyeval - xytarg
        dmag = np.sqrt(dxi[:,0]**2 + dxi[:,1]**2)
        ql = np.quantile(dmag, quivquant)
        blah2 = ax2.quiver(x, y, dxi[:,0], dxi[:,1])
        qk = ax2.quiverkey(blah2, 0.05, 0.95, U=ql, \
                           label='%.e (at %.2f)' % (ql, quivquant), \
                           labelpos='E', fontproperties={'size':8})

        # adjust the axis limits to accommodate the quiver key
        ax2.set_xlim(ax2.get_xlim()*np.repeat(1.1, 2) )
        ax2.set_ylim(ax2.get_ylim()*np.repeat(1.1, 2) )
        
        ax2.set_xlabel(r'$x$')
        ax2.set_ylabel(r'$y$')        

        ax2.set_title('Residuals (fit minus gen)')
        
        # figure supertitle
        fig2.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)
        
    # Make plots showing the basis?
    if not showbasis:
        return

    print("testoneline INFO - plotting basis...")
    LSQ.pattern.showbases(1, cmap=cmap)
    
