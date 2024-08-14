#
# fit2d.py 
#

#
# 2024-08-14 WIC - OO refactoring of 2d fitting and MCMC exploration
# methods
#

import numpy as np

from parset2d import Pars1d
import unctytwod
from fitpoly2d import Leastsq2d
import lnprobs2d

class Guess(object):

    """Object and methods to set up the guess parameters for a later more
full exploration with MCMC."""

    def __init__(self, obssrc=None, obstarg=None, deg=1, Verbose=True):

        # Control variable
        self.Verbose = Verbose
        
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
        self.lsq_nowts = False

        # control variables for the parts of the model other than the
        # transformation parameters, as scalars given that this will
        # likely be written and read using a configuration file. Any
        # value of None is ignored.
        self.guess_noise_loga = None
        self.guess_noise_logb = None
        self.guess_noise_c = None
        self.guess_asymm_ryx = None
        self.guess_asymm_corrxy = None
        self.guess_mixmod_f = None
        self.guess_mixmod_vxx = None

        # Populate the non-transformation guesses from this
        self.populatenontransf()

        # guess for transformation
        self.guess_transf = np.array([])
        
        # Weights for any weighted estimates (e.g. lstsq)
        self.wts = np.array([])

        # Least squares object
        self.LSQ = None

        # Initial guess as Pars1d object
        self.Parset = None

        # Transformation object with guess to pass to minimizer etc.
        self.PGuess = None
        
    def countdata(self):

        """Utility - measures the number of datapoints"""

        self.nsrc = 0 # initialise
        
        # data must be present
        if not hasattr(self.obssrc, 'xy'):
            return

        self.nsrc = np.shape(self.obssrc.xy)[0]

    def initguessesnontransf(self):

        """Initializes guesses for the parts of the model not referring to the tansformation

        """

        self.guess_noise_model = []
        self.guess_noise_asymm = []
        self.guess_mixmod = []

    def populatenontransf(self):

        """Populates guesses for the non-transformation pieces of the model"""

        # (re-) initialize the guess parameters
        self.initguessesnontransf()
        
        # The noise model
        if self.guess_noise_loga is not None:
            self.guess_noise_model = np.array([self.guess_noise_loga])

            # This indent is deliberate. It only makes sense to fit
            # logb and c if all three noise model parameters are
            # present.
            if self.guess_noise_logb is not None and \
               self.guess_noise_c is not None:
                self.guess_noise_model \
                    = np.hstack(( self.guess_noise_model, \
                                  self.guess_noise_logb, \
                                  self.guess_noise_c ))

        # The noise asymmetry model
        if self.guess_asymm_ryx is not None:
            self.guess_asymm = [self.guess_asymm_ryx]

            # Indent intentional, corrxy must be the second parameter
            if self.guess_asymm_corrxy is not None:
                self.guess_asymm.append(self.guess_asymm_corrxy)

        # the mixture model
        if self.guess_mixmod_f is not None:
            self.guess_mixmod.append(self.guess_mixmod_f)

            # Indent intentional, vxx must be the second parameter
            if self.guess_mixmod_vxx is not None:
                self.guess_mixmod.append(self.guess_mixmod_vxx)
                
    def initializeweights(self):

        """Initializes the weights"""

        self.wts = np.ones(self.nsrc)
        
    def wtsfromcovtarg(self):

        """Constructs weights from covariances in the target frame"""

        self.initializeweights()
        
        if self.lsq_nowts:
            return
        
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

        if not hasattr(self.LSQ, 'pars'):
            if self.Verbose:
                print("Guess.fitlsq WARN - LSQ parameters not present")
            return

        self.guess_transf = np.copy(self.LSQ.pars)
        
    def populateparset(self):

        """Attempts to use lsq parameters to populate the full guess model"""

        self.Parset = Pars1d(model=self.guess_transf,\
                             noise=self.guess_noise_model, \
                             symm=self.guess_noise_asymm, \
                             mix=self.guess_mixmod)

    def populateguesstransf(self):

        """Sets up the transformation object for the initial guess"""

        # (Remember, self.transf is a pointer to the kind of
        # transformation object we're using from unctytwod.py)

        xy = self.obssrc.xy
        self.PGuess = self.transf(xy[:,0], xy[:,1], \
                                  self.obssrc.covxy, \
                                  self.Parset.model, \
                                  kind=self.polyfit, \
                                  checkparsy=True)
