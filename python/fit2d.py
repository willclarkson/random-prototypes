#
# fit2d.py 
#

#
# 2024-08-14 WIC - OO refactoring of 2d fitting and MCMC exploration
# methods
#

import numpy as np

import unctytwod
from fitpoly2d import Leastsq2d

class Guess(object):

    """Object and methods to set up the guess parameters for a later more
full exploration with MCMC."""

    def __init__(self, obssrc=None, obstarg=None, deg=1):

        # observation object in source frame. Must have at least xy
        # and ideally covxy attributes.
        self.obssrc = obssrc
        self.obstarg = obstarg

        # Measure the number of "rows" in the data
        self.countdata()
        
        # Fit particulars for linear model
        self.deg = deg
        self.transf = unctytwod.Poly
        self.polyfit = 'Chebyshev'

        # Weights for any weighted estimates (e.g. lstsq)
        self.wts = np.array([])

        # Least squares object
        self.LSQ = None
        
    def countdata(self):

        """Utility - measures the number of datapoints"""

        self.nsrc = 0 # initialise
        
        # data must be present
        if not hasattr(self.obssrc, 'xy'):
            return

        self.nsrc = np.shape(self.obssrc.xy)[0]
        
    def initializeweights(self):

        """Initializes the weights"""

        self.wts = np.ones(self.nsrc)
        
    def wtsfromcovtarg(self):

        """Constructs weights from covariances in the target frame"""

        self.initializeweights()

        # Covariances must be populated in the target space
        if not hasattr(self.obstarg,'covxy'):
            return

        covxy = self.obstarg.covxy
        bbad = self.getbadcovs(covxy)

        # If there are NO good planes, return here
        if np.sum(~bbad) < 1:
            return
        
        self.wts = covxy * 0.
        self.wts[~bbad] = np.linalg.inv(covxy[~bbad])
        
    def getbadcovs(self, covars=np.array([]) ):

        """Returns a boolean indicating which if any planes of an input
[N,2,2] covariance matrix stack are singular.

        """

        # CovarsNx2x2 also has simular functionality
        if np.size(covars) < 4:
            return np.array([])

        return np.linalg.det(covars) <= 0.

        
    def fitlsq(self):

        """Does least squares fitting

        """

        # Convenience view
        xyobs  = self.obssrc.xy

        self.LSQ = Leastsq2d(xyobs[:,0], xyobs[:,1], deg=self.deg, \
                             w=self.wts, kind=self.polyfit, \
                             xytarg = self.obstarg.xy)
