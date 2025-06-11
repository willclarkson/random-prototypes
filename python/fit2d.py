#
# fit2d.py 
#

#
# 2024-08-14 WIC - OO refactoring of 2d fitting and MCMC exploration
# methods
#

import time # for timing
import numpy as np
import copy

# For writing/reading configurations
import configparser

# For copying objects
import copy

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

        self.guess_islog10_noise_c = False
        
        # Configuration parameters we may want to write or
        # read. Again, we need to partition these by type when reading
        # in
        self.conf_section='Guess'
        self.conf_readpath='NA' # path from which configs loaded
        self.conf_bool = ['lsq_nowts', \
                          'ignore_uncty_obs', 'ignore_uncty_targ', \
                          'guess_islog10_mix_frac', \
                          'guess_islog10_mix_vxx', \
                          'guess_islog10_noise_c', \
                          'fit_noise_model', 'fit_noise_asymm', \
                          'fit_mixmod']
        self.conf_int = ['deg']
        self.conf_flt = ['guess_noise_loga', 'guess_noise_logb', \
                         'guess_noise_c', 'guess_asymm_ryx', \
                         'guess_asymm_corrxy', \
                         'guess_mixmod_f', 'guess_mixmod_vxx', \
                         'mag0', \
                         'alpha0', 'delta0']
        self.conf_str = ['polyfit', 'conf_readpath']

        # config strings that are classes
        self.conf_class = ['transf']
        
        # Use this to restrict the attributes that can be set, and to
        # put the configuration file in a human-readable order
        self.confpars=['transf', 'polyfit', 'deg', 'lsq_nowts', \
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
                       'guess_islog10_noise_c', \
                       'conf_readpath', \
                       'mag0', \
                       'alpha0', 'delta0']
        
        # Instance quantities that may depend on the above choices and
        # settings follow.

        # data ranges
        self.xmin = -1.
        self.xmax = 1.
        self.ymin = -1.
        self.ymax = 1.
        
        # guess for pointing
        self.alpha0 = 0.
        self.delta0 = 0.

        # Indicator: does the transformation have the tangent point as
        # its first two entries?
        self.transfstponly = ['Equ2tan', 'Tan2equ']
        self.transfswithtp = ['xy2equ', 'Equ2tan', 'Tan2equ', 'TangentPlane']
        self.hastangentpoint = False
        
        # guess for transformation
        self.guess_transf = np.array([])

        # formal covariance estimate of the transformation parameters
        self.guess_uncty_formal = np.array([])
        
        # Weights for any weighted estimates (e.g. lstsq)
        self.wts = np.array([])

        # Least squares object
        self.LSQ = None

        # backup of least squares object (to keep the original if we
        # are shunting updated weights into LSQ)
        self.LSQ_BAK = None
        
        # Non-parametric bootstrap parameters (implemented currently
        # only for the polynomial model).
        self.nboots = 10000 # keep a record (2025-06-06 - was 1,000)
        self.boots_ignoreweights = False
        self.boots_pars = np.array([])
        self.boots_ok = np.array([])
        
        # Initial guess as Pars1d object
        self.Parset = None

        # Transformation object with guess to pass to minimizer etc.
        self.PGuess = None

        # Set xminmax from obssrc by default
        self.xminmaxfromobs()
        
        # Now we prepare for the fit
        self.populatenontransf()
        # self.applyunctyignorance()

        # set our indicator
        self.classifywithtp()
        
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

        # ... classes in unctytwod...
        for key in self.conf_class:
            try:
                sattr = conf[key]
                if sattr.find('None') < 0:
                    if hasattr(unctytwod, sattr):
                        setattr(self, key, getattr(unctytwod, sattr))
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

        # If we are ignoring all the data uncertainties then we must
        # fit the nosie as part of the model. Apply that here.
        if self.ignore_uncty_obs and self.ignore_uncty_targ:
            self.fit_noise_model=True

        # Set the indicator flag (has tangent point or not)
        self.classifywithtp()
            
        # If this is called outside self.__init__ then we want to do
        # any processing that init would do to the parameters before
        # proceeding.
        self.populatenontransf()
        self.applyunctyignorance()

    def classifywithtp(self):

        """Sets the indicator attribute for whether the transformation
parameters include the tangent point (useful for perturbing initial
guesses)

        """

        self.hastangentpoint = self.transf.__name__ in self.transfswithtp
        
        
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

    def xminmaxfromobs(self):

        """Utility - sets the xmin, xmax, ymin, ymax from observation object."""

        # This is probably NOT what we will eventually want. For the
        # moment, this implements how this used to work.

        # default then override if present
        self.xmin = np.min(self.obssrc.xy[:,0])
        self.xmax = np.max(self.obssrc.xy[:,0])
        self.ymin = np.min(self.obssrc.xy[:,1])
        self.ymax = np.max(self.obssrc.xy[:,1])

        for key in ['xmin', 'xmax', 'ymin', 'ymax']:
            if hasattr(self.obssrc, key):
                setattr(self, key, getattr(self.obssrc, key))
            
            
    def applyunctyignorance(self):

        """If asked, zero out the uncertainties in the (local copies of) the
uncertainty estimates in the source and/or target frame.

        Copies the input covariances across to backup quantities so
        that we can "unforget" them later if needed

        """

        self.bak_covxyobs = np.array([])
        self.bak_covxytarg = np.array([])
        
        if self.ignore_uncty_obs:
            self.bak_covxyobs = np.copy(self.obssrc.covxy)
            self.obssrc.covxy *= 0.

        if self.ignore_uncty_targ:
            self.bak_covxytarg = np.copy(self.obstarg.covxy)
            self.obstarg.covxy *= 0.

    def initguesstransf(self):

        """Initializes the transformation to zeros using the transformation degree to set its length."""

        # Default is to get the number of parameters from the degree
        # (thought: this really ought to be an instance
        # attribute). The quadratic below is PER COORDINATE so we
        # avoid the division by two:
        # npars = int((self.deg**2 + 3*self.deg + 2)/2.)
        npars = int(self.deg**2 + 3*self.deg + 2)
        
        # If the transformation params are only tangent plane related,
        # then there are only two parameters.
        if self.transf.__name__ in self.transfstponly:
            npars = 2

        self.guess_transf = np.zeros(npars)
            
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

    def covobs2targ(self):

        """Transforms covariances in the obs frame to the targ frame, using
the fitted parameters (if any).

        """

        # count bad covariance planes, compare to number of obs
        # datapoints
        bbadobs = self.getbadcovs(self.obssrc.covxy)
        nsrc = self.obssrc.xy.shape[0]

        if np.sum(bbadobs) > 0:
            if self.Verbose:
                print("fit2d.Guess.covobs2targ WARN - too few good obs covars: %i, %i" % (np.sum(~bbadobs), self.obssrc.xy.shape[0]) )
            return

        # do we have parameters?
        if np.size(self.PGuess.parsx) < 3:
            if self.Verbose:
                print("fit2d.Guess.covobs2targ WARN - guess parameters not yet populated.")
            return

        # If we are here then our guess has uncertainties to
        # propagate. So propagate them!
        
        self.PGuess.propagate()

        print("PROPAG COVAR DEBUG:")
        print(self.PGuess.covtran[0])
        print(self.obstarg.covxy[0])
        print("==================")

    def updatelsq_covars(self):

        """If we have an initial-guess least-squares fit, this method updates
it with propagated covariances from the obs frame."""

        
        self.covobs2targ()

        # Now add the target-frame covariances, update the weights in
        # the LSQ object, and re-fit.
        if np.shape(self.PGuess.covtran) != np.shape(self.obstarg.covxy):
            if self.Verbose:
                print("fit2d.Guess.updatelsq_covars INFO - shape mismatch between source and target covariances. Not updating.")
            return

        
        covs_targ_sum = self.PGuess.covtran + self.obstarg.covxy
        bbad = self.getbadcovs(covs_targ_sum)
        if np.sum(~bbad) < 1:
            return

        # now update the LSQ object, copying the old one to backup
        self.LSQ_BAK = copy.deepcopy(self.LSQ)
        self.wts_BAK = np.copy(self.wts)

        # Copy the fit parameters and formal uncertainties as well
        self.guess_transf_BAK = np.copy(self.guess_transf)
        self.guess_uncty_formal_BAK = np.copy(self.guess_uncty_formal)
        
        self.wts = self.obstarg.covxy * 0.
        self.wts[~bbad] = np.linalg.inv(covs_targ_sum[~bbad])

        if self.Verbose:
            print("fit2d.Guess.updatelsq_covars INFO - re-fitting the lsq with propagated covars")
        self.fitlsq()
        

        # report results?
        if self.Verbose:
            print("fit2d.Guess.updatelsq_covars INFO - before and after:")
            print(self.wts_BAK[0:2])
            print(self.wts[0:2])
            print("@@@@@@@@@@@@@@@@@@@@@@@@")
            print(self.guess_uncty_formal_BAK)
            print(self.guess_uncty_formal)

            print("^^^^^^^^^^^^^^^^^^^^^^^^^")

                
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

        # populate the formal uncertainty estimate
        self.guess_uncty_formal = np.linalg.inv(self.LSQ.H)

    def bootstraplsq(self, nboots=-1, seed=None):

        """Does nonparametric bootstrap least-squares fitting.

Inputs:

        nboots = number of samples. If <0, defaults to self.nboots

        seed = random number seed

Returns:

        No returns - attribute self.boots_pars is populated.

"""

        # First ensure the whole sample is fit so that we know what
        # shape to build the bootstrap parameters array. We will use
        # self.LSQ as a template.
        if not hasattr(self.LSQ, 'pars'):
            self.fitlsq()
            
        if nboots < 0:
            nboots = self.nboots
        else:
            self.nboots = nboots
            
        # Set up and run the bootstrap trials
        lsq_boots = copy.copy(self.LSQ)
        self.boots_pars = np.zeros(( nboots, self.LSQ.pars.size))

        # Now we do the bootstrap:
        if self.Verbose:
            t0 = time.time()
            print("fit2d.bootstraplsq INFO - starting %i non-parametric bootstrap samples..." % (nboots))

            if self.boots_ignoreweights:
                print("fit2d.bootstraplsq INFO - bootstraps will ignore weights")

        # boolean for OK
        self.boots_ok = np.repeat(True, nboots)
            
        ndata = np.size(self.LSQ.x)
        rng = np.random.default_rng(seed=seed)
        for iboot in range(nboots):
            lsample = rng.integers(low=0, high=ndata, size=ndata)

            # update the lsq object with this sample
            lsq_boots.x = self.LSQ.x[lsample]
            lsq_boots.y = self.LSQ.y[lsample]
            lsq_boots.W = self.LSQ.W[lsample]
            lsq_boots.xytarg = self.LSQ.xytarg[lsample]

            # If ignoring weights, replace with identity matrices
            if self.boots_ignoreweights:
                lsq_boots.initweights()
                
            lsq_boots.setpatternmatrix()
            lsq_boots.sethessian()
            lsq_boots.setbeta()

            # 2026-06-06 put this in try/except to handle small n_data
            try:
                lsq_boots.solvepars()

                # Now slot the parameters into the bootstrap sample
                self.boots_pars[iboot] = np.copy(lsq_boots.pars)
            except:
                self.boots_ok[iboot] = False
                
        if self.Verbose:
            print("fit2d.bootstraplsq INFO - ... done in %.2e seconds" \
                  % (time.time()-t0))

            print("Boots OK: %i of %i" \
                  % (np.sum(self.boots_ok), np.size(self.boots_ok)) )

        # trim down to the non-crashed bootstraps
        self.boots_pars = self.boots_pars[self.boots_ok]
        
    def populateparset(self):

        """Attempts to use lsq parameters to populate the full guess model"""

        self.Parset = Pars1d(model=self.guess_transf,\
                             noise=self.guess_noise_model, \
                             symm=self.guess_asymm, \
                             mix=self.guess_mixmod, \
                             mag0=self.mag0, \
                             islog10_noise_c=self.guess_islog10_noise_c, \
                             islog10_mix_frac=self.guess_islog10_mix_frac, \
                             islog10_mix_vxx=self.guess_islog10_mix_vxx, \
                             xmin=self.xmin, \
                             xmax=self.xmax, \
                             ymin=self.ymin, \
                             ymax=self.ymax, \
                             transfname=self.transf.__name__)

    def ingestparset(self, parset=None):

        """Sets parset attribute from input parameter set"""

        self.Parset = parset
        
    def parsfromparset(self):

        """Populates parameters and needed attributes from parset. The
complement to self.populateparset()."""

        # The transformation and polynomial name
        try:
            self.transf = getattr(unctytwod, self.Parset.transfname)
        except:
            print("parsfromparset WARN - problem with transfname %s" \
                  % (self.Parset.transfname))

        # Choice of polynomial component
        if hasattr(self.Parset, 'polyname'):
            setattr(self, 'polyfit', self.Parset.polyname)
            
        self.guess_transf = self.Parset.model
        self.guess_noise_model = self.Parset.noise
        self.guess_asymm = self.Parset.asymm
        self.guess_mixmod = self.Parset.mix

        self.mag0 = self.Parset.mag0

        self.guess_islog10_noise_c = self.Parset.islog10_noise_c
        self.guess_islog10_mix_frac = self.Parset.islog10_mix_frac
        self.guess_islog10_mix_vxx = self.Parset.islog10_mix_vxx

        # ensure the nuisance parameter choices are consistent with
        # the options set
        self.reconcilenontransf()
        
    def populateguesstransf(self):

        """Sets up the transformation object for the initial guess"""

        # (Remember, self.transf is a pointer to the kind of
        # transformation object we're using from unctytwod.py)

        # Convenience views. Don't forget the domain!
        xy = self.obssrc.xy
        covxy = self.obssrc.covxy

        # position and covariance in the target frame. Only used by a
        # subset of the transformations (which we trust to handle this
        # input).
        radec = self.obstarg.xy
        covradec = self.obstarg.covxy

        # Now we populate the transformation object. This
        # specification should work regardless of whether the
        # transformation actually uses the target positions and
        # covariances.
        self.PGuess = self.transf(xy[:,0], xy[:,1], \
                                  covxy, \
                                  self.Parset.model, \
                                  kindpoly=self.polyfit, \
                                  radec=radec, covradec=covradec, \
                                  xmin=self.xmin, \
                                  xmax=self.xmax, \
                                  ymin=self.ymin, \
                                  ymax=self.ymax, \
                                  checkparsy=True)

        # This is the old way, with the specifications done separately
        # depending on the transformation. Comment out before deletion
        # after testing.
        
        ## this is just a little specialized... if our transformation
        ## also works on observation data, we call that too
        #if self.transf.__name__.find('TangentPlane') < 0:
        #    self.PGuess = self.transf(xy[:,0], xy[:,1], \
        #                              self.obssrc.covxy, \
        #                              self.Parset.model, \
        #                              kind=self.polyfit, \
        #                              checkparsy=True, \
        #                              xmin=self.xmin, \
        #                              xmax=self.xmax, \
        #                              ymin=self.ymin, \
        #                              ymax=self.ymax)
        #else:
        #    # only if transf is 'TangentPlane,' we pass the
        #    # observation data too
        #    radec = self.obstarg.xy
        #    covradec = self.obstarg.covxy
        #    self.PGuess = self.transf(xy[:,0], xy[:,1], \
        #                              self.obssrc.covxy, \
        #                              self.Parset.model, \
        #                              kindpoly=self.polyfit, \
        #                              radec=radec, covradec=covradec, \
        #                              xmin=self.xmin, \
        #                              xmax=self.xmax, \
        #                              ymin=self.ymin, \
        #                              ymax=self.ymax)
