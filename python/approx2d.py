#
# approx2d.py  - approximate transformations by polynomials
#

import copy
import numpy as np
import matplotlib.pylab as plt
plt.ion()

# methods for simulating and fitting data
import sim2d, fit2d
import fitpoly2d
from weightedDeltas import CovarsNx2x2

class Simset(object):

    """Simulated data with polynomial fits [to be tidied up]"""

    def __init__(self, ndata=10000, \
                 parsim='test_sim_equat_nonoise.ini', \
                 degmin=1, degmax=5, polytype='Chebyshev'):

        # How many points are we generating?
        self.ndata = ndata

        # simulation object
        self.parfile_sim = parsim[:]
        self.setupsim()

        # fit information
        self.llsq = []
        self.degmin = degmin
        self.degmax = degmax
        self.polytype = polytype

        # quantiles for residuals statistics
        self.quantiles = np.array([0.01, 0.1, 0.5, 0.90, 0.99])
        self.resid_quantiles = np.array([])

        # transformation objects for "test" samples
        self.transf_test = None

        # parameters for "test" data (source frame). Include uniform
        # covariances to assess how that propagates.
        self.test_nx = 8
        self.test_ny = 8
        self.test_major = 1.0e-6
        self.test_minor = 7.0e-7
        self.test_rotdeg = 30.

    def setupsim(self):

        """Sets up the simulated data"""

        self.sim = sim2d.Simdata()
        self.sim.loadconfig(self.parfile_sim)

        # Override ndata with input if positive
        if self.ndata > 0:
            self.sim.npts = self.ndata
        else:
            self.ndata = np.copy(self.sim.npts)

    def makedata(self):

        """Generates a new dataset under the simulation parameters"""

        self.sim.generatedata()

    def settransf_used(self):

        """Sets up transformation object for the 'truth' transformation"""

        self.transf_test = copy.copy(self.sim.PTruth)

    def populatetestdata(self):

        """Populates the 'test' dataset"""

        if self.transf_test is None:
            self.settransf_used()

        # if this is STILL None, return
        if self.transf_test is None:
            return

        # Get the positions...
        xs, ys = self.gridxy()

        # ... build the covariances
        majors = np.repeat(self.test_major, xs.size)
        minors = np.repeat(self.test_minor, xs.size)
        rotdegs = np.repeat(self.test_rotdeg, xs.size)

        # Compute the covariances
        Cov = CovarsNx2x2(majors=majors, minors=minors, rotDegs=rotdegs)

        # update the transformation source data...
        self.transf_test.x = xs
        self.transf_test.y = ys
        self.transf_test.covxy = Cov.covars

        # ... update its jacobian...
        try:
            self.transf_test.getjacobian()
        except:
            self.transf_test.setjacobian() ## FIX THIS
            
        # ... and re-propagate positions and uncertainties to the
        # target frame
        self.transf_test.initxytran()
        self.transf_test.propagate()
        
    def gridxy(self):

        """Returns a grid of x, y positions using the instance's grid
settings"""

        if self.transf_test is None:
            return np.array([]), np.array([])

        # Allow transf not to have domain information
        xmin = np.min(self.transf_test.x)
        xmax = np.max(self.transf_test.x)
        ymin = np.min(self.transf_test.y)
        ymax = np.max(self.transf_test.y)

        if hasattr(self.transf_test, 'xmin'):
            xmin = self.transf_test.xmin
        if hasattr(self.transf_test, 'xmax'):
            xmax = self.transf_test.xmax
        if hasattr(self.transf_test, 'ymin'):
            ymin = self.transf_test.ymin
        if hasattr(self.transf_test, 'ymax'):
            ymax = self.transf_testy.ymax
            
        
        vx = np.linspace(xmin, xmax, self.test_nx)
        vy = np.linspace(ymin, ymax, self.test_ny)

        xx, yy = np.meshgrid(vx, vy, indexing='ij')

        return np.ravel(xx), np.ravel(yy)
        
    def performfits(self):

        """Sweeps through the degrees, performing lstsq fits"""

        self.llsq = []
        degs = np.arange(self.degmin, self.degmax+1., 1., 'int')
        wts = np.ones(self.sim.xy.shape[0])

        # Set up the summary statistics array
        self.resid_quantiles = np.array([])
        
        for deg in degs:
            lsq = fitpoly2d.Leastsq2d(self.sim.xy[:,0], \
                                      self.sim.xy[:,1], \
                                      deg=deg, \
                                      w=wts, \
                                      kind=self.polytype, \
                                      xytarg=self.sim.xytarg)

            # Add a residuals attribute to the lsq object
            lsq.xyresid = lsq.xytarg - lsq.ev()

            # Add quantiles information
            lsq.quantiles = self.quantiles
            lsq.resid_quantiles = np.quantile(lsq.xyresid, \
                                              lsq.quantiles, axis=0)

            # quantiles in xy residuals
            if np.size(self.resid_quantiles) < 1:
                self.resid_quantiles = np.copy(lsq.resid_quantiles)
            else:
                self.resid_quantiles = np.dstack(( self.resid_quantiles, \
                                                   lsq.resid_quantiles))
            
            # ... finally, append the fit onto the list
            self.llsq.append(lsq)
            
    def setupfit(self):

        """Sets up least-squares fitting object"""

        deg=1
        kind='Chebyshev'
        ndata = self.sim.xy.shape[0]
        wts = np.ones(ndata)
        self.lsq = fitpoly2d.Leastsq2d(self.sim.xy[:,0], \
                                       self.sim.xy[:,1], \
                                       deg=deg, \
                                       w=wts, \
                                       kind=kind, \
                                       xytarg=self.sim.xytarg)

        # populate residuals
        self.lsq.ev()
        
##### Methods that use this follow

def testsim(ndata=2500, polytype='Chebyshev'):

    """Tests the functionality"""

    SS = Simset(ndata, polytype=polytype)
    SS.makedata()

    # Transformation
    print(SS.sim.transf.__name__)
    
    # Now sweep through the fits
    SS.performfits()

    # Prepare transformation object with test data
    SS.populatetestdata()
    
    # Tell the quantiles
    print("quantiles:", SS.resid_quantiles.shape)
    print(SS.resid_quantiles[0,0,:]*3.6e6) # along N for x
    print(SS.resid_quantiles[0,1,:]*3.6e6) # along N for y
    
    print("approx2d.testsim INFO - plotting...")
    fig1 = plt.figure(1)
    fig1.clf()

    # loop through the orders
    nords = len(SS.llsq)
    for iset in range(nords):
        ax1 = fig1.add_subplot(2, nords, iset+1)
        ax2 = fig1.add_subplot(2, nords, iset+1+nords)

        lsq = SS.llsq[iset]
        dum1 = ax1.scatter(SS.sim.xytran[:,0], \
                           SS.sim.xytran[:,1], s=1, \
                           c=lsq.xyresid[:,0])
        dum2 = ax2.scatter(SS.sim.xytran[:,0], \
                           SS.sim.xytran[:,1], s=1, \
                           c=lsq.xyresid[:,1])

        cbar1 = fig1.colorbar(dum1, ax=ax1)
        cbar2 = fig1.colorbar(dum2, ax=ax2)

        ax1.set_title('deg %i' % (lsq.fitdeg))

        # label the bottom vertical axes
        ax2.set_xlabel(r'$\alpha$')
        
        # hide the vertical tick labels for all but the first plot
        if iset > 0:
            ax1.set_yticklabels([])
            ax2.set_yticklabels([])
        else:
            ax1.set_ylabel(r'$\delta$')
            ax2.set_ylabel(r'$\delta$')
        
    return
        
    ax11 = fig1.add_subplot(221)
    ax12 = fig1.add_subplot(222)
    ax13 = fig1.add_subplot(224)
    
    dum11 = ax11.scatter(SS.sim.xy[:,0], \
                         SS.sim.xy[:,1], s=1, c='g')
    dum12 = ax12.scatter(SS.sim.xytran[:,0], \
                         SS.sim.xytran[:,1], s=1, \
                         c=SS.llsq[ordr-1].xyresid[:,0])

    dum13 = ax13.scatter(SS.sim.xytran[:,0], \
                         SS.sim.xytran[:,1], s=1, \
                         c=SS.llsq[ordr-1].xyresid[:,1])

    
    ax11.set_xlabel(r'$\xi$')
    ax11.set_ylabel(r'$\eta$')
    ax12.set_xlabel(r'$\alpha$')
    ax12.set_ylabel(r'$\delta$')
    ax13.set_xlabel(r'$\alpha$')
    ax13.set_ylabel(r'$\delta$')

    ax12.set_title('Order %i' % (SS.llsq[ordr-1].fitdeg))
    
    # colorbar for axis 2
    cbar2 = fig1.colorbar(dum12, ax=ax12)
    cbar3 = fig1.colorbar(dum13, ax=ax13)
