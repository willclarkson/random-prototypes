#
# sim2d.py
#

#
# 2028-08-13 WIC - methods to explore 2D mapping using MCMC. Much of
# this is OO refactored from the prototype fittwod.py.
#

import numpy as np

# Don't reinvent the wheel if it comes with the standard library:
import configparser

# transformations, noise model
from parset2d import Pars1d
import unctytwod
import noisemodel2d
import mixfgbg

from fitpoly2d import Leastsq2d, Patternmatrix

from weightedDeltas import CovarsNx2x2

from obset2d import Obset


class Simdata(object):

    """Methods and fake data for MCMC simulations"""

    def __init__(self, npts=200, deg=1, Verbose=True):

        # Some decision / control variables
        self.gen_noise_model = False
        self.Verbose = Verbose
        self.nouncty_obs = False
        self.nouncty_tran = False
        self.add_outliers = True
        self.add_uncty_extra = True
        
        # Position
        self.npts = npts
        self.seed_data = None
        self.xmin, self.xmax = -1., 1.
        self.ymin, self.ymax = -1., 1.

        # Apparent magnitude
        self.magexpon = 1.5
        self.maglo, self.maghi = 16., 19.5
        self.seed_mag = None
        self.mag0 = 0.
        
        # "Truth" model parameters
        self.transf = unctytwod.Poly
        self.polytransf = 'Chebyshev'
        self.deg = deg
        self.transfexpon = 1.5
        self.transfscale = 1.
        self.pars_transf = np.array([])
        self.seed_params = None

        # Tangent point
        self.alpha0 = 0.
        self.delta0 = 0.
        
        # Non-transformation model parameters. These are all specified
        # here as scalars because the configuration writer/reader
        # understands them that way. The method parsfromscalars() then
        # propagates these into the pars_noise etc. that are needed.
        self.noise_loga = -4. # stdx vs mag
        self.noise_logb = -20.
        self.noise_c = 2.
        self.islog10_noise_c = False
        self.asymm_ryx = 0.8  # noise shape
        self.asymm_corr = 0.1
        
        # Parameters for noise generation in the target frame. By
        # default, noise is propagated from the obs frame to the
        # target frame. If the parameters below are not none, then the
        # noise is generated separately for the target frame
        self.gen_noise_targ = False
        self.noise_targ_loga = None
        self.noise_targ_logb = None
        self.noise_targ_c = None
        self.mag0_targ = None        
        self.asymm_targ_ryx = None
        self.asymm_targ_corr = None

        # Mixture fraction - for outliers
        self.mix_frac = -1.
        self.mix_vxx = -8.5
        self.mix_islog10_frac = True
        self.mix_islog10_vxx = True
        
        # parameters for "extra" (i.e. unmodeled) noise
        self.extra_loga = None
        self.extra_logb = None
        self.extra_c = None
        self.extra_ryx = None
        self.extra_corr = None

        # Initialise pars lists
        self.pars_noise = []
        self.pars_noise_targ = []
        self.pars_asymm = []
        self.pars_asymm_targ = []
        self.pars_mix = []
        self.pars_extra_noise = []
        self.pars_extra_asymm = []
        
        # Populate the noise etc. parameter lists from these
        # scalars. This populates pars_noise, pars_asymm, pars_mix,
        # islog10_mix, extra_noise, extra_asymm.
        self.parstolists()
        
        # Mixture model / outlier parameters
        self.seed_outly = None

        # List of configuration parameters we'll want to write or
        # read. Because configparser needs to know which arguments are
        # what datatypes, we set the list here of types to read/write.
        self.conf_section = 'Simulation' 
        self.conf_int = ['npts', 'deg', \
                         'seed_data', 'seed_mag', 'seed_params', \
                         'seed_outly']
        self.conf_flt = ['xmin', 'xmax', 'ymin', 'ymax', 'magexpon', \
                         'maghi','maglo', 'transfexpon', 'transfscale', \
                         'noise_loga', 'noise_logb', 'noise_c', \
                         'asymm_ryx', 'asymm_corr', \
                         'mix_frac', 'mix_vxx', \
                         'extra_loga', 'extra_logb', 'extra_c', \
                         'extra_ryx', 'extra_corr', \
                         'mag0', \
                         'noise_targ_loga', 'noise_targ_logb', \
                         'noise_targ_c', 'mag0_targ', \
                         'asymm_targ_ryx', 'asymm_targ_corr', \
                         'alpha0', 'delta0']
        self.conf_bool = ['gen_noise_model', 'add_uncty_extra', \
                          'nouncty_obs', 'nouncty_tran', 'add_outliers', \
                          'mix_islog10_frac', 'mix_islog10_vxx', \
                          'islog10_noise_c', \
                          'gen_noise_targ']
        self.conf_class = ['transf']
        self.conf_str = ['polytransf']

        # The configuration file should also be human-readable... Here's an
        # attempt to put this into a sensible order.
        self.confpars = self.conf_bool[0:5] + self.conf_str + ['deg'] + \
            ['npts', 'xmin', 'xmax', 'ymin', 'ymax', 'seed_data'] + \
            ['magexpon', 'maglo', 'maghi', 'seed_mag', 'mag0'] + \
            ['transfexpon', 'transfscale', 'seed_params'] + \
            ['noise_loga', 'noise_logb','noise_c', 'islog10_noise_c'] + \
            ['asymm_ryx', 'asymm_corr'] + \
            ['gen_noise_targ', 'noise_targ_loga', 'noise_targ_logb', \
             'noise_targ_c', 'mag0_targ', \
             'asymm_targ_ryx', 'asymm_targ_corr'] + \
            ['mix_frac', 'mix_vxx','mix_islog10_frac','mix_islog10_vxx'] + \
            ['extra_loga', 'extra_logb', 'extra_c'] + \
            ['extra_ryx','extra_corr'] + \
            ['seed_outly'] + \
            ['alpha0', 'delta0'] + \
            ['transf']
        
        # Number of parameters per non-transformation
        # parameter. Useful when splitting the 1D parameters into
        # transformation and [other] as will be used with the
        # minimizer and MCMC.
        self.countpars()
        
        # Generated data quantities
        self.xy = np.array([])
        self.mags = np.array([])
        self.isoutly = np.array([])
        self.xytran = np.array([])
        self.covtran = np.array([])

        # perturbed positions
        self.xyobs = np.array([])
        self.xytarg = np.array([])
        
        # Generated transformation objects
        self.PTruth = None
        
        # Generated noise objects. Maybe change the name once the
        # refactoring is done.
        self.Cxy = None
        self.Ctran = None
        self.Coutliers = None # for generating outlier samples
        self.CExtra = None # for adding extra "unmodeled" variance
        
        # Nudges to apply to the generated positions in each frame to
        # transform to 'observed' positions. The "outliers" xy nudge
        # is kept separate so that it can be applied in the source or
        # the target frame.
        self.nudgexy = np.array([])
        self.nudgexytran = np.array([])
        self.nudgexyoutly = np.array([])
        self.nudgexyextra = np.array([])

        # 1D array of (transformation + noise etc.) parameters
        self.Parset = Pars1d()
        
        # Object to package the target position to fitting routines
        self.Obssrc = Obset()
        self.Obstarg = Obset()

    def parstolists(self):

        """Converts non-transformation parameters specified as scalars, to
lists. If the ingredients are listed as None, they are not added to
the output list in each case.

This populates:

        self.pars_noise
        
        self.pars_asymm
        
        self.pars_mix
        
        self.islog10_mix
        
        self.pars_extra_noise
        
        self.pars_extra_asymm

from attributes:

        self.noise_loga, self.noise_logb, self.noise_c

        self.asymm_ryx, self.asymm_corr

        self.mix_frac, self.mix_vxx

        self.mix_islog10_frac, self.mix_islog10_vxx

        self.extra_loga, self.extra_logb, self.extra_c

        self.extra_ryx, self.extra_corr

respectively.

        """

        # Noise parameters...
        self.pars_noise = []

        if self.noise_loga is not None:
            self.pars_noise = [self.noise_loga]
        
        if self.noise_logb is not None:
            self.pars_noise.append(self.noise_logb)

        if self.noise_c is not None:
            self.pars_noise.append(self.noise_c)

        # ... noise parameters for the target frame...
        self.pars_noise_targ = []

        if self.noise_targ_loga is not None:
            self.pars_noise_targ = [self.noise_targ_loga]

        if self.noise_targ_logb is not None:
            self.pars_noise_targ.append(self.noise_targ_logb)

        if self.noise_targ_c is not None:
            self.pars_noise_targ.append(self.noise_targ_c)
        
        # Noise symmetry parameters...
        self.pars_asymm = []

        if self.asymm_ryx is not None:
            self.pars_asymm = [self.asymm_ryx]
            
        if self.asymm_corr is not None:
            self.pars_asymm.append(self.asymm_corr)

        # ... noise asymmetry in the target frame...
        self.pars_asymm_targ = []

        if self.asymm_targ_ryx is not None:
            self.pars_asymm_targ = [self.asymm_targ_ryx]

        if self.asymm_targ_corr is not None:
            self.pars_asymm_targ.append(self.asymm_targ_corr)
            
        # Mixture parameters...
        self.pars_mix = []

        if self.mix_frac is not None:
            self.pars_mix = [self.mix_frac]

        if self.mix_vxx is not None:
            self.pars_mix.append(self.mix_vxx)

        # The mixture log10s are control variables, so we need them:
        self.islog10_mix = [True, True]
        if self.mix_islog10_frac is not None:
            self.islog10_mix[0] = self.mix_islog10_frac

        if self.mix_islog10_vxx is not None:
            self.islog10_mix[1] = self.mix_islog10_vxx

        # Now the parameters for extra variance
        self.pars_extra_noise = []

        if self.extra_loga is not None:
            self.pars_extra_noise.append(self.extra_loga)

        if self.extra_logb is not None:
            self.pars_extra_noise.append(self.extra_logb)

        if self.extra_c is not None:
            self.pars_extra_noise.append(self.extra_c)

        self.pars_extra_asymm = []

        if self.extra_ryx is not None:
            self.pars_extra_asymm.append(self.extra_ryx)

        if self.extra_corr is not None:
            self.pars_extra_asymm.append(self.extra_corr)
            
    def writeconfig(self, pathconfig=''):

        """Writes configuration parameters to file"""

        if len(pathconfig) < 4:
            return

        # Set up the object
        config = configparser.ConfigParser()

        # Write to the "simulation" section
        config[self.conf_section] = {}
        for key in self.confpars:
            if not hasattr(self, key):
                print("writeconfig WARN - attribute missing: %s" % (key))
                continue
            config[self.conf_section][key] = str(getattr(self, key))

        with open(pathconfig, 'w') as configfile:
            config.write(configfile)

    def loadconfig(self, pathconfig=''):

        """Loads configuration file. Because configparser treats all input as
strings and must be instructed to parse the different types
separately, any string that matches 'None' is explicitly converted to
None.

        """

        # Do nothing if the configuration file has too few entries
        if len(pathconfig) < 4:
            return

        config = configparser.ConfigParser()
        try:
            config.read(pathconfig)
        except:
            print("Simdata.loadconfig WARN - problem reading config file %s" \
                  % (pathconfig))
            return

        if not self.conf_section in config.sections():
            if self.Verbose:
                print("Simdata.loadconfig WARN - section %s not in file %s" \
                      % (self.conf_section, pathconfig))
            return

        # View of the section of the configuration file we want
        conf = config[self.conf_section]

        # List of keys that failed for some reason
        lfailed = []

        # Now we go through these type by type. The booleans...
        for keybool in self.conf_bool:
            try:
                if conf[keybool].find('None') < 0:
                    thisattr = conf.getboolean(keybool)
                else:
                    thisattr = None
                setattr(self, keybool, thisattr)
            except:
                lfailed.append(keybool)

        # ... the integers...
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

        # ... the attributes of unctytwod...
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

        # Since some of the parameter are used in list form, ensure
        # again that those lists are accurate.
        self.parstolists()
        self.countpars()

        # If points are to be generated on the sky, adjust the xmin,
        # xmax, ymin, ymax accordingly. NOTE that this corresponds to
        # catalog positions selected by rectangle on the sky (rather
        # than from another observation-set projected onto the sky).
        if self.transf.__name__.find('Equ2tan') > -1:
            self.xmin += self.alpha0
            self.xmax += self.alpha0
            self.ymin += self.delta0
            self.ymax += self.delta0
        
        if len(lfailed) > 0 and self.Verbose:
            print("Simdata.loadconfig WARN - parse problems with keywords:", \
                  lfailed)
            
    def countpars(self):

        """Counts the number of (non-transformation) model parameters."""

        self.npars_noise = np.size(self.pars_noise)
        self.npars_asymm = np.size(self.pars_asymm)
        self.npars_mix = np.size(self.pars_mix)
        
    def makefakexy(self):

        """Makes random uniform xy points within bounds set in the instance"""

        rng = np.random.default_rng(self.seed_data)
        self.xy = rng.uniform(size=(self.npts, 2))
        
        # self.xy = np.random.uniform(size=(self.npts,2))
        
        self.xy[:,0] = self.xy[:,0]*(self.xmax-self.xmin) + self.xmin
        self.xy[:,1] = self.xy[:,1]*(self.ymax-self.ymin) + self.ymin

    def makefakemags(self):

        """Makes power-law apparent magnitudes"""

        rng = np.random.default_rng(self.seed_mag)
        sraw = rng.power(self.magexpon, self.npts)
        self.mags = sraw *(self.maghi - self.maglo) + self.maglo

    def makemagcovars(self):

        """Generates magnitude-dependent covariances."""

        self.Cxy = self.getmagcovars(self.pars_noise, self.pars_asymm, \
                                     mag0=self.mag0)

    def maketargcovars(self):

        """Makes covariances in the target frame if parameters supplied"""

        # Allow control variable to override
        if not self.gen_noise_targ:
            return

        if len(self.pars_noise_targ) < 1:
            return
        
        self.Ctran = self.getmagcovars(self.pars_noise_targ, \
                                       self.pars_asymm_targ, \
                                       mag0=self.mag0_targ)

        # Update self.covtran accordingly
        self.covtran = np.copy(self.Ctran.covars)
        
    def makeextracovars(self):

        """Generates (magnitude-dependent) covariance due to unmodeled
noise

        """

        # WATCHOUT - sending blank arrays in DOES NOT stop
        # getmagcovars from doing something! Exercise a little more
        # control...

        self.CExtra = noisemodel2d.mags2noise(self.pars_extra_noise, \
                                              self.pars_extra_asymm, \
                                              self.mags, \
                                              mag0=self.mag0, \
                                              islog10_c=self.islog10_noise_c)
        
    def getmagcovars(self, \
                     pars_noise=np.array([]), \
                     pars_asymm=np.array([]), \
                     mags=np.array([]), mag0=None):

        """Returns magnitude-dependent covariances"""

        if np.size(pars_noise) < 1:
            pars_noise = self.pars_noise
        if np.size(pars_asymm) < 1:
            pars_asymm = self.pars_asymm
        if np.size(mags) < 1:
            mags = self.mags

        # Magnitude zeropoint
        if mag0 is None:
            mag0 = np.copy(self.mag0)
            
        return noisemodel2d.mags2noise(pars_noise, pars_asymm, mags, \
                                       mag0=mag0, \
                                       islog10_c=self.islog10_noise_c)
        
        
    def makeunifcovars(self):

        """Makes uniform covariances"""

        # Identical to makemagcovars except only the first noise
        # parameter is used. 
        
        self.Cxy = noisemodel2d.mags2noise(self.pars_noise[0], \
                                           self.pars_asymm, \
                                           self.mags, \
                                           mag0=self.mag0)

    def assignoutliers(self):

        """Assigns outlier status to a (uniform) random permutation of
objects"""

        self.isoutly = np.repeat(False, self.npts)

        if np.size(self.pars_mix) < 1:
            return
        
        # parse the mixture parameters
        foutliers = mixfgbg.parsefraction(self.pars_mix[0], \
                                          self.islog10_mix[0])
        
        if foutliers <= 0. or foutliers > 1 or not np.isfinite(foutliers):
            return

        # now generate uniform random permuation:
        rng=np.random.default_rng(seed = self.seed_outly)
        xdum = rng.uniform(size = self.npts)
        lbad = np.argsort(xdum)[0:int(self.npts * foutliers)]
        self.isoutly[lbad] = True

    def makeoutliers(self):

        """Creates the xy covariance object due to outliers"""
        
        # Nothing to do if we don't actually have a variance...
        if np.size(self.pars_mix) < 2:
            return
        
        # Parse the variance
        vxx = mixfgbg.parsefraction(self.pars_mix[1], \
                                    self.islog10_mix[1], \
                                    maxval=np.inf, inclusive=False)
        
        # Replicate the variance
        stdxs = np.repeat(np.sqrt(np.abs(vxx)) , self.npts)
        self.Coutliers = CovarsNx2x2(stdx=stdxs)

        # for debug:
        # print("sim2d.makeoutliers INFO:", vxx, stdxs[0], self.Coutliers.covars[0])

    def makepars(self):

        """Generates parameters for the simulation"""

        # 2024-09-03: either a linear model OR a pointing model.
        transfname = self.transf.__name__
        if transfname.find('Tan2equ') < 0 and \
           transfname.find('Equ2tan') < 0:
            self.makepars_linear()

            # A little awkward again... if doing a polynomial-then-sky
            # model, swap in the field center for the simulation
            if transfname.find('xy2equ') > -1:
                print("makepars INFO - swapping in field center:")
                ihalf = int(np.size(self.pars_transf)*0.5)

                pars_poly = np.copy(self.pars_transf)
                pars_point = np.hstack(( self.alpha0, self.delta0 ))

                self.pars_transf = np.hstack(( pars_point, pars_poly[1:ihalf], pars_poly[ihalf+1::] ))
                
            
            return

        # If all we are doing is generating a pointing, then we shunt
        # the pointing information into the pars_transf attribute.
        self.pars_transf = np.array([self.alpha0, self.delta0])
        
    def makepars_linear(self):

        """Generates fake parameters for the linear model"""

        if np.size(self.xy) < 2:
            if self.Verbose:
                print("Simdata.makepars WARN - xy not yet populated.")
            return

        PM = Patternmatrix(self.deg, self.xy[:,0], self.xy[:,1], \
                           kind=self.polytransf, orderbypow=True)

        self.pars_transf = \
            PM.getfakeparams(seed = self.seed_params, \
                             scale = self.transfscale, \
                             expfac = self.transfexpon)

    def setuptransftruth(self):

        """Sets up the 'truth' transformation object"""

        self.PTruth = self.transf(self.xy[:,0], self.xy[:,1], \
                                  self.Cxy.covars, self.pars_transf, \
                                  kindpoly=self.polytransf, \
                                  checkparsy=True, \
                                  xmin=self.xmin, xmax=self.xmax, \
                                  ymin=self.ymin, ymax=self.ymax)
        
    def setupxytran(self):

        """Propagates the truth positions into the target frame"""

        self.PTruth.propagate()
        self.xytran = np.copy(self.PTruth.xytran)
        self.covtran = np.copy(self.PTruth.covtran)

        self.Ctran = CovarsNx2x2(self.PTruth.covtran)

    def initnudges(self):

        """Initialises the nudges in the two frames"""

        self.nudgexy = self.xy * 0.
        self.nudgexytran = self.xy * 0.
        self.nudgexyoutly = self.xy * 0.
        self.nudgexyextra = self.xy * 0.
        
    def makenudges(self):

        """Sets up the x, y nudges depending on the noise choices. The objects
that draw sapmles from the various noise objects must already be
present."""

        self.initnudges()

        if not self.nouncty_obs and self.Cxy is not None:
            self.nudgexy += self.Cxy.getsamples()
            
        if not self.nouncty_tran and self.Ctran is not None:
            self.nudgexytran += self.Ctran.getsamples()

        # any extra unmodeled covariance
        if self.add_uncty_extra:
            if self.CExtra is not None:
                self.nudgexyextra = self.CExtra.getsamples()
                # print("makenudges INFO:", np.sum(self.nudgexyextra))
                
        # Any outliers
        if self.add_outliers:
            if np.sum(self.isoutly) > 0 and self.Coutliers is not None:
                
                nudgeall = self.Coutliers.getsamples()
                self.nudgexyoutly[self.isoutly] = \
                    nudgeall[self.isoutly]

    def applynudges(self):

        """Applies the nudges to the source and target data"""

        # Should think about how we handle a change of frame (e.g. if
        # the extra and/or outliers are to be added in the source
        # rather than the target frame). For the moment, apply all the
        # extras to the target frame.

        self.xyobs = self.xy +  self.nudgexy
        self.xytarg = self.xytran \
            + self.nudgexytran \
            + self.nudgexyextra \
            + self.nudgexyoutly

    def packagemodelpars(self):

        """Packages the transformation and noise parameters into a Pars1d
object"""

        # These are going to serve as truth parameters against which
        # the guess will be compared. So the noise parameters are the
        # EXTRA noise.
        
        #self.Parset = Pars1d(model=self.pars_transf, \
        #                     noise=self.pars_noise, \
        #                     symm=self.pars_asymm, \
        #                     mix=self.pars_mix, \
        #                     mag0=self.mag0, \
        #                     islog10_noise_c=self.islog10_noise_c)

        self.Parset = Pars1d(model=self.pars_transf, \
                             noise=self.pars_extra_noise, \
                             symm=self.pars_extra_asymm, \
                             mix=self.pars_mix, \
                             mag0=self.mag0, \
                             islog10_noise_c=self.islog10_noise_c, \
                             islog10_mix_frac=self.mix_islog10_frac, \
                             islog10_mix_vxx=self.mix_islog10_vxx, \
                             xmin=self.xmin, xmax=self.xmax, \
                             ymin=self.ymin, ymax=self.ymax, \
                             transfname=self.transf.__name__)

    def packagedata(self):

        """Wrapper - packages source/obs and target data into Obset objects

        """

        self.packagesourcedata()
        self.packagetargetdata()
        
    def packagesourcedata(self):

        """Packages the source data into an obset object. Includes the domain
of the simulated data (since we know this going in)

        """

        self.Obssrc = Obset(self.xyobs, self.PTruth.covxy, \
                            self.mags, ~self.isoutly, \
                            xmin=self.xmin, xmax=self.xmax, \
                            ymin=self.ymin, ymax=self.ymax)
        
    def packagetargetdata(self):

        """Packages the target data into an obset object to reduce the number
of arguments to the minimizer"""

        self.Obstarg = Obset(self.xytarg, self.covtran, \
                             self.mags, ~self.isoutly)

    def gendata_source(self):

        """Generates data in the source frame, including apparent magnitudes
and outliers

        """

        # Generate "truth" points and magnitudes in the source frame
        self.makefakexy()
        self.makefakemags()

        # Use the magnitudes to create and apply noise
        if np.size(self.pars_noise) < 1:
            covs = np.zeros(( self.npts, 2, 2))
            self.Cxy = CovarsNx2x2(covs)
        else:
            if self.gen_noise_model:
                self.makemagcovars()
            else:
                self.makeunifcovars()

        # assign outliers
        self.assignoutliers()

    def gendata_targ(self):

        """Generates target-frame data, using the transformation parameters

        """

        # The self.PTruth (a transf object) needs to be populated
        if self.PTruth is None:
            return

        # Propagate source positions into target frame, set up the
        # noise and nudges
        self.setupxytran()
        self.initnudges()

        # If separate noise parameters were provided for the noise in
        # the target frame, generate the noise using those parameters
        # and replace the covariance in the transformed frame with
        # those covariances. This method will do nothing if
        # self.gen_noise_targ is False.
        self.maketargcovars()
        
        # set up any outliers and/or extra unmodeled covariances
        self.makeoutliers()
        self.makeextracovars()

    def applyposuncertainties(self):

        """Applies positional uncertainties to generated data, in both the
source/obs and target frames.

        """
        
        # Apply the above to produce x, y nudges
        self.initnudges()
        self.makenudges()
        self.applynudges()

    def genpars(self, clobber=True):

        """Generates model parameters using simulation hyperparameters

        """

        # Allow refusal to overwrite if already set
        if self.PTruth is not None and not clobber:
            return
        
        self.makepars()
        self.setuptransftruth()
        self.packagemodelpars()
        
    def generatedata(self, clobber_model = False):

        """Wrapper - generates fake data. Model parameters can also be
generated if not set elsewhere (or forced to re-generate using
clobber_model).

        INPUTS

        clobber_model = force regeneration of model parameters, even
        if set elsewhere??

        OUTPUTS

        No outputs - updates instance attributes.

        """

        # Somewhat simplified by the refactoring of the actual
        # generating methods into other methods...

        if self.npts < 1:
            print("Simdata.gendata WARN - simulation npts set to %i" \
                  % (self.npts))
            return
        
        # Source-frame data, magnitudes, outliers
        self.gendata_source()

        # If we are generating params, they need the data
        # characteristics. If we already read them in, don't overwrite
        # them with new parameters:
        self.genpars(clobber = clobber_model)

        # Generate the target data
        self.gendata_targ()
        
        # Apply the positional uncertainties in both frames
        self.applyposuncertainties()

        # Package the source/obs and targ data into forms needed
        # elsewhere
        self.packagedata()
        
        ## Baseline x, y, covars in source frame
        #self.makefakexy()
        #self.makefakemags()


        ## Allow noise to be optional
        #if np.size(self.pars_noise) < 1:
        #    covs = np.zeros(( self.npts, 2, 2))
        #    self.Cxy = CovarsNx2x2(covs)
        #else:
        #    if self.gen_noise_model:
        #        self.makemagcovars()
        #    else:
        #        self.makeunifcovars()
        #self.assignoutliers()
        
        ## Define the transformed frame and propagate to it
        #self.makepars()
        #self.setuptransftruth()

        # Our refactored target-frame data generator
        ## self.gendata_targ()
        
        # To refactor
        ## self.setupxytran()
        ## self.initnudges()

        ## If separate noise parameters were provided for the noise in
        ## the target frame, generate the noise using those parameters
        ## and replace the covariance in the transformed frame with
        ## those covariances. This method will do nothing if
        ## self.gen_noise_targ is False.
        #self.maketargcovars()
        
        ## set up any outliers and/or extra unmodeled covariances
        #self.makeoutliers()
        #self.makeextracovars()

        ## Apply the above to produce x, y nudges
        #self.initnudges()
        #self.makenudges()
        #self.applynudges()

        ## Package the fit parameters into a 1d array. See self.Parset
        ## self.packagemodelpars()
        
        ## Package the target data into object expected by the fitter
        #self.packagesourcedata()
        #self.packagetargetdata()
        
### SHORT test routines come here.

def testsim():

    """Tests wrapper for generating data"""

    SD = Simdata()
    SD.writeconfig('test_config.ini')
    SD.generatedata()

    print(SD.Parset.pars)

    # Now try writing to disk
    SD.Obssrc.writeobs('test_obset_written.dat')
    
def testreadconfig():

    """Tests loading the configuration file"""

    SD = Simdata()

    # Ensure things are being changed
    #print(SD.pars_noise)
    #print(SD.pars_asymm)
    
    SD.loadconfig('test_config_changed.ini')

    #print(SD.pars_noise)
    #print(SD.pars_asymm)

    # OK now generate data using the imported parameters
    SD.generatedata()
