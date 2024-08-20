#
# fit2d.py 
#

#
# 2024-08-14 WIC - OO refactoring of 2d fitting and MCMC exploration
# methods
#

import numpy as np

# For writing/reading configurations
import configparser


from parset2d import Pars1d
import unctytwod
from fitpoly2d import Leastsq2d
import lnprobs2d
from obset2d import Obset

class Guess(object):

    """Object and methods to set up the guess parameters for a later more
full exploration with MCMC.

(Some choices can be passed in as arguments, but parameter files will
likely work better.)

    """

    def __init__(self, obssrc=None, obstarg=None, deg=1, \
                 fit_noise_model=True, fit_noise_asymm=True, \
                 fit_mixmod=True, ignore_uncty_obs=False, \
                 Verbose=True):

        # Control variable
        self.Verbose = Verbose
        
        # observation object in source frame. Must have at least xy
        # and ideally covxy attributes.
        self.obssrc = self.copyobset(obssrc)
        self.obstarg = self.copyobset(obstarg)
        
        # Measure the number of "rows" in the data
        self.countdata()
        
        # Fit particulars for linear model
        self.deg = deg
        self.transf = unctytwod.Poly
        self.polyfit = 'Chebyshev'
        self.lsq_nowts = False

        # Ignore the supplied uncertainties in either the source or
        # target frames? (E.g. if we want to model all uncertainty as
        # extra)
        self.ignore_uncty_obs = ignore_uncty_obs
        self.ignore_uncty_targ = False
        
        # control variables for the parts of the model other than the
        # transformation parameters. First explicit choices about what
        # we are fitting, then scalars to supply guess parameters. Any
        # value of None is ignored. The fit choices override the
        # parameters, in that the parameters are still recorded but
        # they are not sent into the guess if we don't want to fit
        # that component.
        self.fit_noise_model = fit_noise_model
        self.fit_noise_asymm = fit_noise_asymm
        self.fit_mixmod = fit_mixmod
        
        self.guess_noise_loga = None
        self.guess_noise_logb = None
        self.guess_noise_c = None
        self.guess_asymm_ryx = None
        self.guess_asymm_corrxy = None
        self.guess_mixmod_f = None
        self.guess_mixmod_vxx = None

        self.guess_islog10_mix_frac = True
        self.guess_islog10_mix_vxx = True
        
        # Configuration parameters we may want to write or
        # read. Again, we need to partition these by type when reading
        # in
        self.conf_section='Guess'
        self.conf_readpath='NA' # path from which configs loaded
        self.conf_bool = ['lsq_nowts', \
                          'ignore_uncty_obs', 'ignore_uncty_targ', \
                          'guess_islog10_mix_frac', \
                          'guess_islog10_mix_vxx', \
                          'fit_noise_model', 'fit_noise_asymm', \
                          'fit_mixmod']
        self.conf_int = ['deg']
        self.conf_flt = ['guess_noise_loga', 'guess_noise_logb', \
                         'guess_noise_c', 'guess_asymm_ryx', \
                         'guess_asymm_corrxy', \
                         'guess_mixmod_f', 'guess_mixmod_vxx']
        self.conf_str = ['polyfit', 'conf_readpath']
        
        # Use this to restrict the attributes that can be set, and to
        # put the configuration file in a human-readable order
        self.confpars=['polyfit', 'deg', 'lsq_nowts', \
                       'ignore_uncty_obs', 'ignore_uncty_targ', \
                       'fit_noise_model', \
                       'guess_noise_loga', 'guess_noise_logb', \
                       'guess_noise_c', \
                       'fit_noise_asymm', \
                       'guess_asymm_ryx', \
                       'guess_asymm_corrxy', \
                       'fit_mixmod', \
                       'guess_mixmod_f', 'guess_mixmod_vxx', \
                       'guess_islog10_mix_frac', \
                       'guess_islog10_mix_vxx', \
                       'conf_readpath']
        
        # Instance quantities that may depend on the above choices and
        # settings follow.
        
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

        # Now we prepare for the fit
        self.populatenontransf()
        self.applyunctyignorance()
        
    def writeconfig(self, pathconfig=''):

        """Writes configuration parameters to supplied file"""

        if len(pathconfig) < 4:
            return

        config = configparser.ConfigParser()
        config[self.conf_section] = {}
        for key in self.confpars:
            if not hasattr(self, key):
                print("fit2d.writeconfig WARN - keyname typo? %s" \
                      % (key))
                continue
            
            config[self.conf_section][key] = str(getattr(self, key))

        with open(pathconfig, 'w') as configout:
            config.write(configout)

    def loadconfig(self, pathconfig=''):

        """Loads configuration parameters from file"""

        # There is a nearly identical method in sim2d.py . Maybe this
        # can be lifted into a separate module (returns attributes to
        # set?)
        
        if len(pathconfig) < 4:
            return

        config = configparser.ConfigParser()
        try:
            config.read(pathconfig)
        except:
            print("Guess.loadconfig WARN - problem reading config file %s" \
                  % (pathconfig))
            return

        # If the file is not found, config.read doesn't throw an
        # exception. So:
        if len(config.sections()) < 1:
            print("Guess.loadconfig WARN - cnofig file not found: %s" \
                  % (pathconfig))
            return
        
        # Can't proceed if the required section is missing
        if not self.conf_section in config.sections():
            print("Guess.loadconfig WARN - section %s missing from %s" \
                  % (self.conf_section, pathconfig))
            return

        # View of the relevant section of the configuration file
        conf = config[self.conf_section]

        # List of keys that failed for some reason
        lfailed = []

        # Now we import the values, type by type. First the booleans:
        for keybool in self.conf_bool:
            try:
                if conf[keybool].find('None') < 0:
                    thisattr = conf.getboolean(keybool)
                else:
                    thisattr = None
                setattr(self, keybool, thisattr)
            except:
                lfailed.append(keybool)

        # The integers
        for key in self.conf_int:
            try:
                # Hack for random number seeds (None or int)
                if conf[key].find('None') < 0:
                    thisattr = conf.getint(key)
                else:
                    thisattr = None
                setattr(self, key, thisattr)
            except:
                lfailed.append(key)

        # ... the floats ...
        for key in self.conf_flt:
            try:
                if conf[key].find('None') < 0:
                    thisattr = conf.getfloat(key)
                else:
                    thisattr = None
                setattr(self, key, thisattr)
            except:
                lfailed.append(key)

        # ... and the strings
        for key in self.conf_str:
            try:
                if conf[key].find('None') < 0:
                    thisattr = conf[key]
                else:
                    thisattr = None
                setattr(self, key, thisattr)
            except:
                lfailed.append(key)

        if len(lfailed) > 0 and self.Verbose:
            print("Guess.loadconfig WARN - parse problems with keywords:", \
                  lfailed)
                
        # Finally, update the config path we just read in.
        self.conf_readpath = pathconfig[:]

        # If this is called outside self.__init__ then we want to do
        # any processing that init would do to the parameters before
        # proceeding.
        self.populatenontransf()
        self.applyunctyignorance()

    def copyobset(self, obset=None):

        """Creates a copy of observation-set object so that we can modify it
in-place without affecting things upstream."""

        if obset is None:
            return None

        obs = Obset(*obset.copycontents())

        return obs
        
    def countdata(self):

        """Utility - measures the number of datapoints"""

        self.nsrc = 0 # initialise
        
        # data must be present
        if not hasattr(self.obssrc, 'xy'):
            return

        self.nsrc = np.shape(self.obssrc.xy)[0]

    def applyunctyignorance(self):

        """If asked, zero out the uncertainties in the (local copies of) the
uncertainty estimates in the source and/or target frame.

        """

        if self.ignore_uncty_obs:
            self.obssrc.covxy *= 0.

        if self.ignore_uncty_targ:
            self.obstarg.covxy *= 0.
        
    def initguessesnontransf(self):

        """Initializes guesses for the parts of the model not referring to the tansformation

        """

        self.guess_noise_model = []
        self.guess_asymm = []
        self.guess_mixmod = []

    def reconcilenontransf(self):

        """Reconciles model guess parameters with fit choices. E.g. if noise model parameters are supplied but self.fit_noise_model = False, the noise model parameters are all set to None."""

        # This doesn't actually need to be called for
        # populatenontransf() to ignore the parameters if not being
        # fit, but it may make things easier depending on what we want
        # to see in the parameter files. It may be useful to store
        # values even if we don't actually use them.
        
        if not self.fit_noise_model:
            self.guess_noise_loga = None
            self.guess_noise_logb = None
            self.guess_noise_c = None

        if not self.fit_noise_asymm:
            self.guess_asymm_ryx = None
            self.guess_asymm_corrxy = None

        if not self.fit_mixmod:
            self.guess_mixmod_f = None
            self.guess_mixmod_vxx = None

        # Also go the other way - don't try to fit a model for which
        # guess = []
        self.preventfittingempty()
            
    def preventfittingempty(self):

        """Prevents attempting to fit when all parameters for a given fit were
supplied as None"""
            
        if all(par is None for par in \
               [self.guess_noise_loga, \
                self.guess_noise_logb, \
                self.guess_noise_c]):
            self.fit_noise_model = False

        if all(par is None for par in \
               [self.guess_asymm_ryx, self.guess_asymm_corrxy]):
            self.fit_noise_asymm = False

        if all(par is None for par in \
               [self.guess_mixmod_f, self.guess_mixmod_vxx]):
            self.fit_mixmod = False
            
    def populatenontransf(self):

        """Populates guesses for the non-transformation pieces of the model. Guesses are not populated if self.fit_[thatmodel] = False"""

        # (re-) initialize the guess parameters
        self.initguessesnontransf()

        # If all parameters for a given model are None, ensure the fit
        # choice is consistent for that model
        self.preventfittingempty()
        
        # The noise model
        if self.guess_noise_loga is not None and self.fit_noise_model:
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
        if self.guess_asymm_ryx is not None and self.fit_noise_asymm:
            self.guess_asymm = [self.guess_asymm_ryx]

            # Indent intentional, corrxy must be the second parameter
            if self.guess_asymm_corrxy is not None:
                self.guess_asymm.append(self.guess_asymm_corrxy)

        # the mixture model
        if self.guess_mixmod_f is not None and self.fit_mixmod:
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

        # If there are NO good planes, return here. This can happen if
        # we already zeroed out the covariances (e.g. if we want to
        # ignore them).
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
                             symm=self.guess_asymm, \
                             mix=self.guess_mixmod)

    def populateguesstransf(self):

        """Sets up the transformation object for the initial guess"""

        # (Remember, self.transf is a pointer to the kind of
        # transformation object we're using from unctytwod.py)

        # Convenience views. Don't forget the domain!
        xy = self.obssrc.xy
        self.PGuess = self.transf(xy[:,0], xy[:,1], \
                                  self.obssrc.covxy, \
                                  self.Parset.model, \
                                  kind=self.polyfit, \
                                  checkparsy=True, \
                                  xmin=self.obssrc.xmin, \
                                  xmax=self.obssrc.xmax, \
                                  ymin=self.obssrc.ymin, \
                                  ymax=self.obssrc.ymax)