#
# mcmc2d.py
#

#
# WIC 2024-08-16 - use sim2d and fit2d to set up and explore 2d data
# with mcmc
#

import os, time
import numpy as np

import pickle
import copy

from scipy.optimize import minimize

import sim2d
from parset2d import Pars1d, Pairset, loadparset
from fit2d import Guess
import lnprobs2d
from lnprobs2d import Prior, Like

# utilities for converting linear parameters back and forth
import sixterm2d

# For serializing sim info to disk
import configparser

# for handling observations
from obset2d import Obset

class MCMCrun(object):

    """Sets up for an emcee run that would be performed via
multiprocessing.

    """

    def __init__(self, parfile_sim='', parfile_guess='', \
                 chainlen=40000, \
                 parfile_prior='', \
                 pathjitter='', \
                 ignoretruth=False, \
                 doboots_poly=False, \
                 nboots=10000,\
                 lsq_uncty_trick=True,\
                 boots_ignoreweights=False,\
                 npoints_sim=None, \
                 path_obs='', path_targ='', \
                 path_truth = '', \
                 simulating=True, \
                 Verbose=True, \
                 path_config=''):

        # list of object attributes that can be I/O with config
        # file. These will be sections in the config file, and they
        # are strings, booleans, and integers, respectively (config
        # parser needs to know what type of variable it is reading in).
        self.config_attr_strings = ['parfile_sim', 'parfile_guess', \
                                    'pathprior', 'pathjitter', \
                                    'path_obs', 'path_targ', 'path_truth']
        self.config_attr_bools = ['ignoretruth', 'doboots_poly', \
                                    'boots_ignoreweights',\
                                    'lsq_uncty_trick', 'simulating', \
                                    'Verbose']
        self.config_attr_integers = ['chainlen', 'npoints_sim', \
                                     'nboots', 'minimizer_maxiter']

        self.config_attr_floats = []

        # Path to mcmc configuration file itself (useful to keep track
        # of this so we can query it from elsewhere)
        self.path_config = path_config[:]
        
        # Control variables
        self.Verbose = Verbose

        # Are we trying to simulate or to use data?
        self.simulating = simulating 
        
        # If simulating, ignore the truth values when setting up the
        # guess?
        self.ignoretruth = ignoretruth

        # if NOT simulating, we need data! These are the paths to the
        # observation-frame and target-frame datafiles, and the
        # instance-level observation objects populated from them.
        self.path_obs = path_obs[:]
        self.path_targ = path_targ[:]
        self.Obssrc = Obset()
        self.Obstarg = Obset()

        # truth parameters if we are reading them in (will want to
        # think about outputting these too)
        self.path_truth = path_truth[:]
        
        # Do non-parametric bootstrapping on polynomial? If so, ignore
        # weights?
        self.doboots_poly = doboots_poly
        self.nboots = nboots
        self.boots_ignoreweights = boots_ignoreweights
        
        # if doing least-squared fit, project source uncertainty onto
        # target frame after first fit, reweight, and refit? [Default
        # to True while testing]
        self.lsq_uncty_trick=lsq_uncty_trick
        
        # Parameters for simulation and for guess
        self.parfile_sim = parfile_sim[:]
        self.parfile_guess = parfile_guess[:]

        # Path to informative prior, if any
        self.pathprior = parfile_prior[:]

        # If simulating, use this number of stars in preference to the
        # contents o the parameter file
        self.npoints_sim = npoints_sim      
        
        # Simulation and guess objects
        self.sim = None
        self.guess = None

        # Does the transformation have the tangent plane as its first
        # two entries?
        self.classeswithtp = ['xy2equ', 'Equ2tan', 'Tan2equ']
        self.hastangentpoint = False
        
        # Method that will be used for ln(posterior), its arguments
        self.methpost = lnprobs2d.lnprob
        self.argspost = ()
        self.argspost_minimizer = ()
        self.lnlike = None
        self.lnprior = None
        self.guess1d = np.array([])

        # For perturbing the initial guess (1d) for the minimizer
        self.nudgescale_guess1d = 1.0e-2
        self.nudge_guess1d = np.array([])
        self.nudge_seed = None
        self.nudge_pointing_arcsec = 5. # for pointing arguments

        # The minimizer output, and some settings
        self.minimizer_soln = None
        self.minimizer_method = 'Nelder-Mead'
        self.minimizer_maxiter = 5000 # so we can I/O this parameter
        self.minimizer_options = {'maxiter':self.minimizer_maxiter}
        self.guess_parset = None

        # For resampling the data when trying to estimate scatter
        self.resample_seed = None
        self.resample_scalefac = 0.1 # rescale from resample to jitter
        self.resample_fsample = 0.9 # fraction to draw resample
        self.resample_solns = np.array([])
        self.resample_lnprobs = np.array([])
        self.resample_jitter = np.array([])
        
        # Comparison between the guess and the 'truth' parameters
        # (useful for scaling the jitter ball)
        self.fracdiff = None

        # Helpful attributes for plots of the samples
        self.truths = None
        self.labels = None
        
        # Quantities for setting up the mcmc runs
        self.guess1d_refined = np.array([]) # convenience view
        self.scaleguess = np.array([])
        self.pos = np.array([])
        self.ndim = 1
        self.nchains = -1
        self.fjitter = 1. # was 3, but that sends the noise pars off
        self.jitterscale_default = 0.05
        self.chainlen = chainlen

        # walker centers and jitter scales actually used
        self.walkers_centers = np.array([])
        self.walkers_jitters = np.array([])

        # text file we could use to read in the jitter scale and/or
        # walker centers. Default to blank
        self.pathjitter=pathjitter[:]
        
        # Arguments to send to mcmc runs
        self.args_ensemble = {}
        self.args_run = {}

        # Some arguments that will be useful for plotting and
        # examining the results. Use sub-dictionaries for each
        # destination for the arguments, so, e.g. corner plot
        # arguments would go in self.args_show['corner'], etc.
        self.args_show = {}

        # Finally, if the configuration file was supplied, get the
        # attribute values from that file.
        if os.access(self.path_config, os.R_OK):
            self.loadconfig(self.path_config, strict=True)
        
    def dosim(self):

        """Wrapper - imports simulation parameters and generates simulated
dataset"""

        self.setupsim()
        self.runsim()

    def setupsim(self, clobber=False):

        """Sets up the simulation object"""

        if self.sim is None or clobber:
            self.sim = sim2d.Simdata()
            
        self.sim.loadconfig(self.parfile_sim)

        # allow override of npoints with supplied input
        if self.npoints_sim is not None:  # Consider >= 0 rather than None
            self.sim.npts = self.npoints_sim
        else:
            self.npoints_sim = self.sim.npts
        
    def runsim(self):

        """Generates the simulated dataset"""

        self.sim.generatedata()

    def setupguess(self):

        """Sets up the guess object"""

        # Although awkward, this probably is the right place to make
        # this distinction. We can imagine that we might want to
        # control the option to use the simulated or actual data in a
        # run.
        
        if self.simulating:
            self.guess = Guess(self.sim.Obssrc, self.sim.Obstarg)
        else:

            if np.size(self.Obssrc.xy) < 1 or np.size(self.Obstarg.xy) < 1:
                print("setupguess WARN: observations not fully populated.")
                return
            
            self.guess = Guess(self.Obssrc, self.Obstarg)
            
            
        self.guess.loadconfig(self.parfile_guess)

        
    def initguessfromtruth(self):

        """Initializes the initial guess from truth parameters if we have
them"""

        # Do we actually have a truth transformation?
        # if not hasattr(self.sim, 'pars_transf'):
        if not hasattr(self.sim, 'Parset'):
            return

        # Now use a pairset to merge the truth and guess parameters
        Pair = Pairset(self.sim.Parset, self.guess.Parset)
        Psub = Pair.sub1into2()
        
        #print("mcmc2d.initguessfromtruth INFO:")
        #for sattr in ['model', 'noise', 'symm', 'mix']:
        #    print("simul:", getattr(self.sim.Parset, sattr) )
        #    print("guess:", getattr(self.guess.Parset, sattr))
        #print("#############")
        
        # self.guess1d = np.copy(self.sim.pars_transf)
        self.guess1d = np.copy(Psub.model)
        self.scalenudgeguess()
        self.nudgeguess1d()

        self.guess.guess_transf = np.copy(self.guess1d)
        self.guess.populateparset()
        self.guess.populateguesstransf()
        
    def guessfromlstsq(self):

        """Does least-squares estimate for initial guess at transformation
parameters.

        """

        # Find the least-squares fit to the transformation only...
        self.guess.wtsfromcovtarg()
        self.guess.fitlsq()

        # ... and populate the pieces we need
        self.guess.populateparset()
        self.guess.populateguesstransf()

        if self.lsq_uncty_trick:
            self.updatelstsquncty()
        
    def updatelstsquncty(self):

        """Given a least-squares estimate, project the obs uncertainties onto
the target frame, update, and re-weight"""

        if not self.lsq_uncty_trick:
            return

        # Proof of life...
        self.guess.updatelsq_covars()
        
    def setupfitargs(self):

        """Sets up arguments for passing to minimizer and/or emcee"""

        self.setuplnprior()        
        self.setuplnlike()
        self.assembleargs()

    def assembleargs(self):

        """Assembles expected arguments for minimizer and/or emcee"""

        # return_blob set to True for lnprob as will be called by
        # emcee...
        self.argspost = (self.guess.PGuess, self.guess.obstarg, \
                         self.guess.Parset, self.lnprior, self.lnlike, \
                         True)

        # ... but the minimizer expects a scalar return, so
        # return_blob=False
        ldum = list(self.argspost)
        ldum[-1] = False
        self.argspost_minimizer = tuple(ldum)
        
    def getargspost_subset(self, fsampl=0.8):

        """Assembles minimizer arguments for a subsample of the datapoints.

        INPUTS

        fsampl = fraction of data sample to draw (note this will be
        random integers with replacement)

        RETURNS

        argspost = tuple of arguments to feed to the minimizer

        COMMENTS

        Attribute self.argspost_minimizer needs to be set.

        """

        if len(self.argspost_minimizer) < 5:
            return ()

        ndata = self.argspost_minimizer[0].x.size

        # Generate sample indices
        nsim = int(ndata * fsampl)
        rng = np.random.default_rng(self.resample_seed)
        lsam = rng.integers(0,ndata,nsim)

        # OK now we (re)construct the pieces with these samples.
        pguess = copy.deepcopy(self.argspost_minimizer[0])
        pguess.updatedata(pguess.x[lsam], \
                          pguess.y[lsam], \
                          pguess.covxy[lsam])
        pguess.initxytran()
        
        obstarg = copy.deepcopy(self.argspost_minimizer[1])
        for attr in ['xy','covxy','mags','isfg']:
            setattr(obstarg, attr, getattr(obstarg,attr)[lsam])
        obstarg.countpoints()

        # We have to trim the pset object too...
        pset = copy.deepcopy(self.argspost_minimizer[2])

        lnprior = self.argspost_minimizer[3]
        
        # Construct a new lnlike object rather than modifying in place
        lnlike = Like(pset, pguess, obstarg)

        # Retain these debug comments for the moment...
        #print("getargspost_minimizer INFO")
        #print("0:",self.argspost_minimizer[0].x.size)
        #print("0':",pguess.x.size)
        #print("1:", self.argspost_minimizer[1].xy.shape)
        #print("1':", obstarg.xy.shape)
                    
        #print("4:", self.argspost_minimizer[4].transf.x.shape)
        #print("4:", self.argspost_minimizer[4].xytarg.shape)

        #print("4':", lnlike.transf.x.shape)
        #print("4':", lnlike.xytarg.shape)

        return (pguess, obstarg, pset, lnprior, lnlike)
    
    def minimize_on_subset(self, fsampl=0.8, Verbose=True, \
                           dolnprob=False):

        """Draws random sample with replacement and runs the minimizer on
it."""

        if len(self.argspost_minimizer) < 1:
            print("minimize_on_subset WARN - main argspost_minimizer not yet set.")
            return
        
        # Draw the subset and get the arguments for the minimizer
        argssub = self.getargspost_subset(fsampl)
        ufunc = lambda *args: 0.-self.methpost(*args)

        if Verbose:
            print("minimize_on_subset INFO - starting on subset...", \
                  end="")
            t0 = time.time()
            
        soln = minimize(ufunc, \
                        self.guess1d, \
                        args = argssub, \
                        method = self.minimizer_method, \
                        options = self.minimizer_options)

        if Verbose:
            print(" done in %.2e seconds" % (time.time() - t0))

        # Because the objective function *is* lnprob (or more
        # properly, ln(like) if doing non-paramtric bootstraps), we
        # get it for free:
            
        if self.resample_solns.size < 1:
            self.resample_solns = np.copy(soln.x)
            self.resample_lnprobs = np.array([soln.fun])
        else:
            self.resample_solns = np.vstack(( self.resample_solns, soln.x ))
            self.resample_lnprobs = np.hstack(( self.resample_lnprobs, \
                                                soln.fun ))
            
    def estjitter_from_resamples(self, fsample = -1.):

        """Estimates the jitter by re-fitting to resamples from the data.
        
        INPUTS

        fsample = fraction of the data that will be resampled. Set <0
        to accept the already-set attribute

        OUTPUTS

        none - updates attributes.

        """

        # Update the attribute with argument if in the right range.
        if 0 < fsample <= 1.:
            self.resample_fsample = fsample
            
        
        self.resample_solns = np.array([])
        for iset in range(2):
            self.minimize_on_subset(self.resample_fsample)

        diffs = np.abs(self.resample_solns[1] - self.resample_solns[0])

        # just send up the diffs
        self.resample_jitter = diffs * self.resample_scalefac

    def bootstrap_jitter(self, nboots=1000, fsample=1., \
                         pathboots='test_nonparam_full.npy', \
                         pathprobs='test_nonparam_lnprobs.npy'):

        """Uses the machinery of jitter to perform a kind of nonparametric
bootstrap with the full minimizer"""

        # do nothing if we don't want to actually do this...
        if nboots < 1:
            return
        
        self.resample_solns = np.array([])

        t0 = time.time()
        for iset in range(nboots):
            self.minimize_on_subset(fsample, Verbose=False, dolnprob=True)

            if iset % 10 == 1:

                tsofar = time.time()-t0
                secperit = tsofar / (1.0*iset)
                tremain = (nboots - iset)*secperit
                
                print("bootstrap_jitter INFO: set %i of %i after %.2e sec, est %.2e sec remain..." \
                      % (iset, nboots, tsofar, tremain), end="\r")

                np.save(pathboots, self.resample_solns)
                np.save(pathprobs, self.resample_lnprobs)

        # Clear the newline
        print("")
                
        # since we're still developing, write this to disk now
        np.save(pathboots, self.resample_solns)
        np.save(pathprobs, self.resample_lnprobs)
        
        
    def setuplnprior(self):

        """Sets up the ln(prior) object for minimization and/or emcee"""

        self.lnprior = Prior(self.guess.Parset, self.pathprior)

        # Ensure the lnprior object knows which indices correspond to
        # {a,b,c,d,e,f}
        print("setuplnprior INFO - abc indices:", self.guess.PGuess.inds1d_6term)
        if hasattr(self.guess.PGuess,'inds1d_6term'):
            self.lnprior.inds1d_6term = \
                self.guess.PGuess.inds1d_6term
        
    def setuplnlike(self):

        """Sets up the ln(likelihood) object for minimization and/or emcee"""
        
        self.lnlike = Like(self.guess.Parset, self.guess.PGuess, \
                           self.guess.obstarg)

        # Magnitude zeropoint for fitting... consider packaging this
        # with obstarg?
        self.lnlike.mag0 = self.guess.mag0
        
    def guessforminimizer(self):

        """Creates and perturbs initial guess for minimizer"""

        self.setupguess1d()
        self.nudgeguess1d()
        self.guessfromprior()
        
    def setupguess1d(self):

        """Creates 1d initial guess for minimizer"""

        self.guess1d = np.copy(self.guess.Parset.pars)

    def scalenudgeguess(self):

        """Sets up the initial nudges for the first guess"""

        # Initialize
        self.nudge_guess1d = self.nudgescale_guess1d * self.guess1d

        # For any cases where the guess was zero, sub with the scale
        bzer = np.abs(self.guess1d) < 1.0e-30
        self.nudge_guess1d[bzer] = self.nudgescale_guess1d

        # Take absolute values (shouldn't matter later on when this is
        # used, but let's get this right)
        self.nudge_guess1d = np.abs(self.nudge_guess1d)
        
        # for pointing arguments
        if not hasattr(self.guess, 'transf'):
            return

        self.setnudgetp()
        
    def setnudgetp(self):

        """If the guess has a tangent point, set the nudge accordingly"""
        
        if self.guess.hastangentpoint:
            self.nudge_guess1d[0] = self.nudge_pointing_arcsec / 3600.
            self.nudge_guess1d[1] = self.nudge_pointing_arcsec / 3600.
        
    def nudgeguess1d(self, seed=None):

        """Perturbs the 1d initial guess for input (e.g. if truth values have
been given as the guess)

"""

        # Ensure the nudge guess is appropriately scaled
        if np.size(self.nudge_guess1d) != np.size(self.guess1d):
            self.scalenudgeguess()
        
        rng = np.random.default_rng(self.nudge_seed)
        #pertns = rng.normal(size=np.size(self.guess1d)) \
        #    * self.nudgescale_guess1d * self.guess1d

        pertns = rng.normal(size=np.size(self.guess1d)) \
            * self.nudge_guess1d

        
        self.guess1d += pertns

    def guessfromprior(self, Verbose=True):

        """For parameters on which we have an informative prior, swap in
samples from that prior for the initial-guess for the minimizer."""

        # If we do not have (or are ignoring) any gaussian prior in
        # the lnprior, do nothing
        if not self.lnprior.withgauss:
            return

        gp = self.lnprior.gaussprior
        priorsample = gp.drawsample().squeeze()

        if Verbose:
            print("guessfromprior DEBUG:", priorsample)
            print("guessfromprior DEBUG:", gp.lpars)
            print("guessfromprior INFO: original guess1d:", self.guess1d)

        # OK now we have to convert the linear parameters from the
        # guess into goeometric parameters so that the samples from
        # the prior can be swapped in
        inds1d = self.guess.PGuess.inds1d_6term
        geom = sixterm2d.getpars(self.guess1d[inds1d])

        if Verbose:
            print("guessfromprior INFO - original geom pars:", geom)

        print("DEBUG:", gp.lpars_6term)
        print("DEBUG:", priorsample)
        print("DEBUG:", priorsample.shape)
            
        # now slot in the samples from the prior - but only the
        # transformation and not the nuisance parameters. Only do this
        # if we actually have priors on any of the 6terms:
        if np.size(gp.lpars_6term) > 0:
            lsample = np.arange(np.size(gp.lpars_6term))
            geom[gp.lpars_6term[lsample]] = priorsample[lsample]

            if Verbose:
                print("guessfromprior INFO - updated geom pars:", geom)

            # Now we slot these back into the guess array as linear
            # parameters
            abc = sixterm2d.abcfromgeom(geom)
            labc = np.arange(np.size(inds1d))
            self.guess1d[inds1d[labc]] = abc[labc]

        if Verbose:
            print("guessfromprior INFO - updated guess1d:", self.guess1d)
        
    def runminimizer(self, Verbose=True):

        """Runs scipy.optimize.minimize on the simulated dataset"""

        print("Minimizer debug:", self.minimizer_options)
        
        if Verbose:
            print("mcmc2d.runminimizer INFO - starting minimizer...")
        t0 = time.time()
        ufunc = lambda *args: 0.-self.methpost(*args)

        self.minimizer_soln = minimize(ufunc, self.guess1d, \
                                       args=self.argspost_minimizer, \
                                       method=self.minimizer_method, \
                                       options=self.minimizer_options)
        t1 = time.time()
        if Verbose:
            print("mcmc2d INFO - ... done in %.2e sec, status %s" \
                  % (t1-t0, self.minimizer_soln.success))

        # copy the result across to 1d parameter array
        self.getguess_from_minimizer()
            
    def getguess_from_minimizer(self):

        """Copies the minimizer solution into the refined guess"""

        if not hasattr(self.minimizer_soln,'x'):
            print("refinedguess WARN - minimizer solution has no x")
            return

        self.guess1d_refined = np.copy(self.minimizer_soln.x)
        
    def populate_guess_parset(self):

        """Shunts the minimizer-found parameters into a new paramset object
for convenient comparison with the truth parameters"""

        if np.size(self.guess1d_refined) < 1:
            return

        self.guess_parset = Pars1d(self.guess1d_refined, \
                                   self.guess.Parset.nnoise, \
                                   self.guess.Parset.nshape, \
                                   self.guess.Parset.nmix, \
                                   mag0=self.guess.mag0, \
                                   islog10_noise_c = \
                                   self.guess.guess_islog10_noise_c)
        
    def calcfracdiff_truth_guess(self):

        """Compares the guess parameter-set with the truth parameter-set"""

        # Nothing to do if we don't actually have a simulation
        # paramset
        if not hasattr(self.sim, 'Parset'):
            return

        # Also nothing to do if we have sworn off truth for this run
        if self.ignoretruth:
            return
        
        PP = Pairset(self.sim.Parset, self.guess_parset)

        # Find fractional difference (of guess) in matching
        # parameters. Force this to become an np float array
        # (depending on what was set up, there may be None or nan in
        # the answer).
        self.fracdiff = PP.fracdiff()
        self.fracdiff.pars = np.asarray(self.fracdiff.pars, 'float64')

        # debug lines
        print("calcfracdiff_truth_guess DEBUG")
        print("==============================")
        print("Absolute difference:")
        pdiff = PP.arithmetic(PP.set1on2, PP.set2, np.subtract)
        print(pdiff.pars)
        print("Fractional difference:")
        print(self.fracdiff.pars)
        print("==============================")

    def settruthsarray(self):

        """Utility - populates the 'truths' array for future plots, matched to
the guess array that will be plotted."""

        # Nothing to do if there is no sim.Parset (usually this means
        # we don't actually know the truths. Note that
        # self.ignoretruths=True does NOT mean we don't necessarily
        # know the truths...
        if not hasattr(self.sim, 'Parset'):
            return
        
        PP = Pairset(self.sim.Parset, self.guess_parset)
        self.truths = np.asarray(PP.set1on2.pars, 'float64')
        
    def ndimfromguess(self):

        """Gets ndim for the MCMC runs"""

        self.ndim = np.size(self.guess1d_refined)

    def setnchains(self):

        """Sets number of mcmc chains"""

        if self.nchains < 1:
            self.nchains = int(self.ndim * 2)+2
        
    def setjitterscale(self):

        """Sets up the scale factors to multiply the offsets for the initial
walker positions"""

        # Initialize the jitter scale from the guess parameters
        self.scaleguess = self.guess1d_refined * self.jitterscale_default
        
        # If we have truth parameters, use the fractional offset from
        # the truth parameters to set the scale.
        if hasattr(self.fracdiff, 'pars') and not self.ignoretruth:
            self.scaleguess = np.asarray(self.fracdiff.pars) * self.fjitter

        # Finally, if we are ignoring truths, fall back on our arcsec
        # pointing (otherwise we can end up with huge scatters).
        if self.ignoretruth:
            self.scaleguess[0] = self.nudge_pointing_arcsec / 3600.
            self.scaleguess[1] = self.nudge_pointing_arcsec / 3600.
            
    def jitterfromguess(self):

        """Sets walker arguments from guess"""

        self.walkers_centers = np.copy(self.guess1d_refined)
        self.walkers_jitters = np.copy(self.scaleguess)

    def initwalkers(self):

        """Sets up walker initial conditions"""

        pertn = np.random.randn(self.nchains, self.ndim)
        #magn = self.scaleguess * self.guess1d_refined
        #self.pos = self.guess1d_refined + pertn * magn[None, :]

        self.pos = self.walkers_centers + \
            pertn * self.walkers_jitters[None, :]
        
        #print("initwalkers INFO:", self.guess1d_refined.dtype, \
        #      pertn.dtype, magn.dtype, self.scaleguess.dtype)
        
    def setupwalkers(self):

        """Wrapper - sets up the walker characteristics for mcmc"""

        # If we have truth parameters, use them to set the scale for
        # the jitter ball for the mcmc initial state
        self.calcfracdiff_truth_guess()
        self.setjitterscale()
        self.jitterfromguess()

        # This probably needs a bit of refactoring... We probably
        # don't want to be switching in the "empirical" jitter here.
        if np.size(self.resample_jitter) == np.size(self.walkers_jitters):
            print("setupwalkers INFO - switching in resampling-based jitters")
            self.jitters_old = np.copy(self.walkers_jitters)
            self.walkers_jitters = np.copy(self.resample_jitter)
            
        # Override jitter with estimate from file
        if len(self.pathjitter) > 3:
            print("setupwalkers INFO - attempting to read jitters from %s" \
                  % (self.pathjitter))
            self.readjitterball(self.pathjitter, getjitter=True, \
                                getpars=False, getlabels=False)
        
        
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("setupwalkers INFO - walkers_jitters:")
        print(self.walkers_jitters)
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        
        self.ndimfromguess()
        self.setnchains()
        # self.setjitterscale()  # moved up 
        self.initwalkers()

    def setargs_ensemblesampler(self):

        """Slots in the arguments for the ensemble sampler"""

        nwalkers, ndim = self.pos.shape # for consistency
        
        self.args_ensemble = {'nwalkers':nwalkers, 'ndim':ndim, \
                              'log_prob_fn':self.methpost, \
                              'args':self.argspost}

    def setargs_emceerun(self):

        """Sets up run arguments for emcee run"""

        self.args_run = {'initial_state':self.pos, \
                         'nsteps':self.chainlen, 'progress':True}

    def setlabels_corner(self):

        """Sets up plot labels for corner plot"""

        # 1d labels
        self.labels = self.guess_parset.getlabels()
        
        # 2024-08-39 uncomment this to get the labels from the
        # transformation object.
        labelsxy = self.guess.PGuess.getlabels()
        self.labels[0:len(labelsxy)] = labelsxy

        ## Refine the plot labels using the patternmatrix object in
        ## the guess transformation
        #pmatrix = self.guess.PGuess.pars2x
        #labelsx = pmatrix.setplotlabels('A')
        #labelsy = pmatrix.setplotlabels('B')
        #labelsxy = labelsx + labelsy
        #self.labels[0:len(labelsxy)] = labelsxy
        
        
    def setargs_corner(self):

        """Sets up arguments that will help in examining the output
(e.g. truth parameters if simulating, etc.)"""

        # Ensure we have a 'truths' array with the same length as the
        # guess array
        if self.truths is None:
            self.settruthsarray()

        self.args_show['corner'] = {}
            
        self.args_show['corner']['truths'] = self.truths

        # plot labels
        self.setlabels_corner()
        if self.labels is not None:
            self.args_show['corner']['labels'] = self.labels

        # Set the number of model parameters for the nuisance
        # parameter highlighting
        self.args_show['corner']['nmodel'] \
            = np.size(self.guess.Parset.model)

        # Indices that correspond to {a,b,c,d,e,f} in linear
        # transformation. We may want to do this transformation when
        # doing the corner plot.
        self.args_show['corner']['inds_abc'] = []
        if hasattr(self.guess.PGuess,'inds1d_6term'):
            self.args_show['corner']['inds_abc'] = \
                self.guess.PGuess.inds1d_6term

        # Scaling for mcmc jitter. There are several attributes we
        # might use, so at the moment we serialize them all
        self.args_show['scalings'] = {}
        for sattr in ['nudge_guess1d', 'scaleguess', 'fjitter']:
            self.args_show['scalings'][sattr] = getattr(self, sattr)

        # The magnitudes actually used when setting up the jitter
        self.args_show['scalings']['jitter_mag'] = \
            self.walkers_jitters
        
            # self.scaleguess * self.guess1d_refined
        
            
    def setargs_truthset(self):

        """Passes paramset object for the truth parameters as an output
argument in args_show['truths']"""

        # Return at least blank arguments
        self.args_show['truthset'] = {}
        
        # Not much to do if nothing simulated
        if not hasattr(self.sim, 'Parset'):
            return
        
        self.args_show['truthset']['parset'] = self.sim.Parset

        # also show truths forced to same length as guess
        PP = Pairset(self.sim.Parset, self.guess_parset)
        self.args_show['truthset']['parslikeguess'] = PP.set1on2

        # Also send the un-perturbed simulated x, y positions
        self.args_show['truthset']['xy'] = self.sim.xy
        self.args_show['truthset']['xytran'] = self.sim.xytran

    def setargs_guess(self):

        """Smuggles information about the guess and scale back to the
interpreter"""

        self.args_show['guess'] = {}
        if hasattr(self.fracdiff, 'pars'):
            self.args_show['guess']['fracdiff'] = self.fracdiff.pars
        self.args_show['guess']['scaleguess'] = self.scaleguess
        self.args_show['guess']['fjitter'] = self.fjitter
        self.args_show['guess']['guess1d_refined'] \
            = self.guess1d_refined
        self.args_show['guess']['guess1d'] \
            = self.guess1d

        # formal uncertainty in lstsq estimate
        self.args_show['guess']['lstsq_guess_transf'] \
            = self.guess.guess_transf
        self.args_show['guess']['lstsq_uncty_formal'] \
            = self.guess.guess_uncty_formal

        # labels for plotting
        labelsxy = self.guess.PGuess.getlabels()
        self.args_show['guess']['labels_transf'] = labelsxy
        
        #pmatrix = self.guess.PGuess.pars2x
        #labelsx = pmatrix.setplotlabels('A')
        #labelsy = pmatrix.setplotlabels('B')
        #self.args_show['guess']['labels_transf'] = \
        #    labelsx + labelsy
        
    def setargs_emcee(self):

        """Sets up the arguments to send to emcee with multiprocessor"""

        self.setargs_ensemblesampler()
        self.setargs_emceerun()
        self.setargs_corner()
        self.setargs_truthset() # useful for examining the truth (ha!)
        self.setargs_guess() # useful for checking the input guess
        
    def returnargs_emcee(self, Verbose=True):

        """Returns the arguments to the interpreter, prints a helpful message"""

        if Verbose:
            print("Returning arguments: esargs, runargs, showargs")
            print("Now execute:")
            print(" ")
            print("examine2d.showguess(esargs)")
            print(" ")
            print("with Pool() as pool:")
            print("      sampler = emcee.EnsembleSampler(**esargs, pool=pool)")
            print("      sampler.run_mcmc(**runargs)")
            print("      flat_samples, lnprobs = mcmc2d.getflatsamples(sampler)")
            print("      examine2d.showcorner(flat_samples, **showargs['corner'])")

        return self.args_ensemble, self.args_run, self.args_show

    def writeargs_emcee(self, stemargs='test'):

        """Serializes the arguments for emcee to disk"""

        if len(self.args_show.keys()) > 0:
            try:
                with open('%s_showargs.pickle' % (stemargs), 'wb') as f:
                    pickle.dump(self.args_show, f)
            except:
                print("writeargs_emcee WARN - problem pickling args_show")
                    
        if len(self.args_run.keys()) > 0:
            with open('%s_runargs.pickle' % (stemargs), 'wb') as f:
                pickle.dump(self.args_run, f)           
        
        if len(self.args_ensemble.keys()) > 0:
            try:
                with open('%s_esargs.pickle' % (stemargs), 'wb') as f:
                    pickle.dump(self.args_ensemble, f)
            except:
                print("writeargs_emcee WARN - problem pickling args_ensemble")

    def writejitterball(self, pathjitter='test_jitter.txt'):

        """Writes the centroids and jitters used to set up the walkers. Make
this something we can input into an mcmc run on actual data"""

        # Centers and jitters as views, so we can decide later to
        # switch to something else if needed

        # Nothing to do if the jitters aren't populated yet
        if np.size(self.walkers_jitters) < 1:
            return
        
        # centers = self.guess1d_refined
        # jitters = self.scaleguess * self.guess1d_refined

        # Variable names. Still need a good way to do this...
        varnames = self.labels[:]

        # use the standard library config parser again:
        config = configparser.ConfigParser()
        config['Pars'] = {}
        config['Jitter'] = {}

        for ipar in range(len(varnames)):
            skey = varnames[ipar][:]
            sjit = 'j_%s' % (skey)

            config['Pars'][skey] = str(self.walkers_centers[ipar])
            config['Jitter'][sjit] = str(self.walkers_jitters[ipar])

        with open(pathjitter, 'w') as jitfile:
            config.write(jitfile)

        # We now need a way to read this in. It probably makes sense
        # to have self.jitterscale as a separate variable rather than
        # the two pieces we currently have...

    def readjitterball(self, pathjitter='test_jitter.txt', \
                       getjitter=True, getpars=False, getlabels=False, \
                       sectionpars='Pars', sectionjitter='Jitter', \
                       debug=False):

        """Loads jitter ball information. Inputs:

        pathjitter = path to jitter ball file

        getjitter [T] = update instance jitter attribute from file

        getpars [F] = update parameter centers from file

        getlabels [F] = update labels from file

        sectionpars = section name for parameters

        sectionjitter = section name for jitter ball

        debug = print information to screen

        """

        
        # let's try piggybacking on the config parser
        if len(pathjitter) < 4:
            return

        if not os.access(pathjitter, os.R_OK):
            print("readjitterball WARN - cannot read path %s" \
                  % (pathjitter))
            return

        # Let's try piggybacking on the config parser
        config = configparser.ConfigParser()
        try:
            config.read(pathjitter)
        except:
            print("readjitterball WARN - problem reading jitter file %s" \
                  % (pathjitter))
            return

        # Populate the needed items. We go ahead and populate these
        # local objects, sending them up to the instance depending on
        # control variables here
        labels = []
        centers = []
        jitters = []

        # We do the parameters and jitters separately in case the
        # input parfile doesn't have any parameters. Currently there
        # is no parsing for the jitters, we trust the input to follow
        # the same order as the parameters.
        
        # labels and jitter ball centers
        for key in config[sectionpars].keys():
            labels.append(key)
            centers.append(config[sectionpars].getfloat(key))
        
        # jitter distribution
        labelsjit = []
        for keyjit in config[sectionjitter].keys():
            labelsjit.append(keyjit)
            jitters.append(config[sectionjitter].getfloat(keyjit))

        # Enforce numpy arrays...
        centers = np.asarray(centers)
        jitters = np.asarray(jitters)

        # ... and pass up to the instance
        if getjitter:
            self.walkers_jitters = np.copy(jitters)

        if getpars:
            self.walkers_centers = np.copy(centers)
        
        # printing to screen?
        if not debug:
            return

        print("-------------------------------------------------")
        print("readjitterball DEBUG - loaded jitter information:")
        for idum in range(len(jitters)):
            print(labels[idum], centers[idum], labelsjit[idum], \
                  jitters[idum], \
                  jitters[idum]/np.abs(centers[idum]) )
        print("-------------------------------------------------")

    def readdata(self):

        """Reads source/target data if we already have any"""

        # Made this slightly more informative:
        
        if not os.access(self.path_obs, os.R_OK):
            if len(self.path_obs) > 3:
                print("MCMCrun.readdata WARN - cannot load path %s" \
                      % (self.path_obs))
            return

        if not os.access(self.path_targ, os.R_OK):
            if len(self.path_targ) > 3:
                print("MCMCrun.readdata WARN - cannot load path %s" \
                      % (self.path_targ))
            return

        self.Obssrc.readobs(self.path_obs)
        self.Obstarg.readobs(self.path_targ)

    def loadconfig(self, pathconfig='', strict=True):

        """Loads configuration parameters for MCMC run. Overrides any choices that may have been set elsewhere (e.g. by command line). 

        INPUTS

        pathconfig = path to configuration file

        strict = only set attributes already initialized (don't
        arbitrary attributes). If set to False, any new attribtue can
        be sent in via the parameter file.

        OUTPUTS

        none - updates instance attributes.

        """

        # redundant but defensive
        if not os.access(pathconfig, os.R_OK):
            print("MCMCrun.loadconfig WARN - cannot read config path: %s" \
                  % (pathconfig))
            return
        
        config = configparser.ConfigParser()
        try:
            config.read(pathconfig)
        except:
            print("MCMCrun.loadconfig WARN - config parser problem with %s" \
                  % (pathconfig))
            return

        # Set the instance attribute with the argument we supplied
        self.path_config = pathconfig[:]
        
        print("MCMCrun.loadconfig INFO - reading configuration from %s" \
              % (pathconfig))
        
        # now we set attributes from the configuration
        # object. Currently we don't need to split the parameters by
        # different sections (though we might decide later to do so) -
        # for the moment, we just read in everything.
        for section in config:
            thisconf = config[section]

            # let's loop through the attributes actually found,
            # classifying them by membership in the list of strings,
            # booleans, etc.
            for attr in thisconf.keys():

                # if asked, only proceed if the attribute already has
                # been initialized in the object (to prevent arbitrary
                # attributes from slipping in)
                if not hasattr(self, attr) and strict:
                    continue

                # now we use the parse convenience tools. I think the
                # easiest way is to set the method by whether the
                # attribute is in our list of ints, floats, or
                # booleans, defaulting to string if not already caught
                # by the other types. This could be done even more
                # parsimoniously using a dictionary with the method as
                # the key, but I think this way below is easier to
                # trouble-shoot. So:
                methget = None
                
                if attr in self.config_attr_integers:
                    methget = thisconf.getint
                    
                if attr in self.config_attr_bools:
                    methget = thisconf.getboolean
                    
                if attr in self.config_attr_floats:
                    methget = thisconf.getfloat

                if methget is None:
                    methget = thisconf.get

                thisval = methget(attr)

                # Could put in some special cases here...
                if attr.find('minimizer_maxiter') > -1:
                    self.minimizer_options['maxiter'] = thisval
                
                setattr(self, attr, thisval)

    def writeconfig(self, pathout='test_mcmcrun_parsused.ini', \
                    secname='MCMCrun'):

        """Writes the configuration parameters actually used to disk

        INPUTS

        pathout = path to write the parameters

        secname = section name for output parameter file

        """

        if len(pathout) < 3:
            return

        config = configparser.ConfigParser()
        config[secname] = {}

        # We might want to impose an order in the output keys. For the
        # moment, just abut the lists together.
        keys = self.config_attr_strings + \
            self.config_attr_bools + \
            self.config_attr_integers + \
            self.config_attr_floats

        # populate the configuration object with teh attributes we
        # want
        for key in keys:
            if not hasattr(self, key):
                continue

            config[secname][key] = str(getattr(self, key))

        # now write the configuration object to disk
        with open(pathout, 'w') as wobj:
            config.write(wobj)
            
    def loadtruths(self):

        """Loads truth parameters into 'simulation' object"""

        # Nothing to do if the input path is not readable
        if not os.access(self.path_truth, os.R_OK):
            print("MCMCrun.loadtruths WARN - truth path not readable: %s" \
                  % (self.path_truth))

            if not self.simulating:
                self.ignoretruth = True
                print("MCMCrun.loadtruths WARN - setting self.ignoretruth to True")
            
            return

        # Sets up the sim object to transfer truth parameters across
        # to
        if self.sim is None:
            self.sim = sim2d.Simdata()

        # Now include parsing
        self.sim.Parset = loadparset(self.path_truth)
        
    def doguess(self, norun=False):

        """Wrapper - sets up and performs initial fit to the data to serve as
our initial state for MCMC exploration.

        When this is done, self.guess_parset contains the minimizer
        fit parameters.

Inputs:

        norun = set up the minimizer but don't actually run it. Useful
        for debugging.

        """
        
        # Sets up the guess object
        self.setupguess()

        # IF that failed, we can't continue
        if self.guess is None:
            print("doguess WARN - guess cannot be set up. Returning.")
            return
        
        print("mcmc2d.doguess DEBUG: guess transf:", self.guess.guess_transf)
        
        # We populate this with default values before doing any of the
        # clever stuff below. The guess currently handles the noise
        # parameters, it's the guess transformation parameters that
        # need initializing. That's actually quite hard in the general
        # case, come back to this later.
        self.guess.initguesstransf()
        self.guess.populateparset()
        self.guess.populateguesstransf()

        print("mcmc2d.doguess DEBUG - initialized guess parset:")
        print(self.guess.Parset.model)
        print(self.guess.Parset.noise)
        print(self.guess.Parset.symm)
        print(self.guess.Parset.mix)
        print("Guess degree:", self.guess.deg)
        print("mcmc2d.doguess DEBUG =======")

        # read in any pre-cooked jitter ball, if pathjitter has been
        # defined.
        self.readjitterball(pathjitter=self.pathjitter, \
                            getjitter=True, \
                            getpars=False, getlabels=False)
        
        # How we specify the "guess" depends on whether we can do a
        # least-squares fit to arrive at one.
        if self.guess.transf.__name__.find('Poly') < 0:

            # If we don't have leastsq fitting for the model, scale it
            # from the "truth" parameters. Will need to work out how
            # to supply a good guess for the case where we are NOT
            # simulating, later.
            self.initguessfromtruth()
                        
        else:

            # 2025-07-17 WIC - I think this self.ignoretruth is a bug:
            # we always want to be trying a guess this way if we're
            # using the polynomial model! Testing this now...
            
            # Do a linear leastsq fit to the data to serve as the
            # initial guess for the minimizer
            ## # if not self.ignoretruth:
            self.guessfromlstsq()

            # 2024-11-25 - try nonparametric bootstrapping? (Note:
            # not sure this should be in this ignoretruth
            # conditional, since we might want to try this anyway)
            if self.doboots_poly:
                self.guess.boots_ignoreweights = \
                    self.boots_ignoreweights
                self.guess.nboots = self.nboots
                    
                self.guess.bootstraplsq()

        # By this point we should have the parameter set. Check that
        # we do.
        print("doguess DEBUG - parameter indices:")
        ps = self.guess.Parset
        print("transf:", ps.lmodel)
        print("lnoise:", ps.lnoise)
        print("lsymm:", ps.lsymm)
        print("lmix:", ps.lmix)
        print("labels:", ps.getlabels() )
        print("indices:", ps.dindices)

        print("doguess DEBUG - parameter values:")
        print("transf:", ps.model)
        print("lnoise:", ps.noise)
        print("lsymm:", ps.symm)
        print("lmix:", ps.mix)
            
        # Setup and run the minimizer using the lstsq fit as input,
        # and shunt the result across to the guess object
        self.setupfitargs() # includes the prior
        self.guessforminimizer()

        print("doguess DEBUG - parameter nudges:")
        print("######################")
        print(self.nudge_guess1d)
        print("######################")
        
        if norun:
            return
        
        self.runminimizer(Verbose=self.Verbose)
        self.populate_guess_parset()

        # output the fitted params to screen
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print("doguess DEBUG - refined post-fit guess:")
        print(self.guess1d_refined)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        
### Some methods follow that we want to be able to use from the
### interpreter. Once an mcmc run is done, the interpreter will have
### "samples" available to use. So we write the flat samples to disk

def setupmcmc(pathsim='test_sim_mixmod.ini', \
             pathfit='test_guess_input.ini', \
              pathprior='', \
              pathjitter='', \
              ignoretruth=False, \
              chainlen=40000, debug=False, \
              writedata=True, \
              doboots_poly=False, \
              nboots=10000,\
              lsq_uncty_trick=True,\
              boots_ignoreweights=False,\
              pathboots='test_boots.npy', \
              npoints_arg=None, \
              pathobs='', pathtarg='', pathtruth='', \
              jitterfromsamples=False, \
              nonparam_minimizer=0, \
              pathconfig=''):

    """Sets up for mcmc simulations. 

Inputs:

    pathsim = path to parameter file for simulating data

    pathfit = path to parameter file for assembling the guess

    pathprior = path to informative prior terms (if any)

    pathjitter = path to any jitter guess (including scales)

    ignoretruth = ignore truth values

    chainlen = chain length for mcmc

    debug = return after setting up the minimizer (useful for development)

    writedata = write generated data to disk

    doboots_poly = do non-parametric bootstrap for polynomial model?

    lsq_uncty_trick = project src frame uncertainties onto the target
    frame, then re-evaluate the lsq with this combined uncertainty?

    pathboots = path for output non-parameteric bootstrap trials

    npoints_arg = None - number of objects to simulate,
    overriding the choice in the simulation parameter file. Default is
    not to override.

    pathobs = path to canned observation set

    pathtarg = path to canned target set

    pathtruth = path to truth parameters that were used to generate
    the canned observations.

    jitterfromsamples = attempt to estimate the jitter by using the
    minimizer on two samples of the dataset.

    nonparam_minimizer = number of bootstraps to try with the
    minimizer [under development]

    pathconfig = path to mcmc run configuration file (which overrides
    any of the previous arguments)
    
Returns:

    esargs = dictionary of arguments for the ensemble sampler

    runargs = dictionary of arguments for running mcmc

    showargs = dictionary of arguments for analysing and showing the results

    """

    # par files must exist!
    simulating = False  # we could use a more sophisticated test
                       # later... This is now superseded by the
                       # parameter file IF IT EXISTS...

    # defer the decision to the config path if it exists
    if not os.access(pathconfig, os.R_OK):                       
        if not parspaths_exist(pathfit, pathsim):
            if not parspaths_exist(pathobs, pathtarg):
                return None, None, None
            else:
                print("mcmc2d.setupMCMC INFO - found obs files %s, %s" \
                      % (pathobs, pathtarg))
        else:
            simulating = True
            print("mcmc2d.setupmcmc INFO - found sim, guess files %s, %s" \
                  % (pathsim, pathfit))
        
    mc = MCMCrun(pathsim, pathfit, chainlen, pathprior, \
                 pathjitter=pathjitter, ignoretruth=ignoretruth, \
                 doboots_poly=doboots_poly, nboots=nboots, \
                 lsq_uncty_trick=lsq_uncty_trick, \
                 boots_ignoreweights=boots_ignoreweights, \
                 npoints_sim=npoints_arg, \
                 path_obs=pathobs, path_targ=pathtarg, \
                 path_truth=pathtruth, \
                 simulating=simulating, \
                 path_config=pathconfig)

    # whatever we do, a guess file must be present (it controls the
    # way the simulation works). Exit gracefully if not present.
    if not os.access(mc.parfile_guess, os.R_OK):
        print("setupMCMC WARN - guess paramfile not readable: %s" \
              % (mc.parfile_guess))
        return None, None, None
    
    # read canned data if we have any
    mc.readdata()

    # write the configuration parameters used
    mc.writeconfig()

    # If the mc object has a truth path set, read it in here. The mc
    # object will handle the case where no path is set. If simulating,
    # these parameters will be used to simulate the data. If using
    # pre-built data, these parameters will be used in plots against
    # the posteriod distribution.
    print("Reading truth parset from %s" % (mc.path_truth))
    mc.loadtruths()

    # print("mc sim object parset model:", mc.sim.Parset.model)

    # Generate synthetic data if asked
    if mc.simulating:
        mc.dosim()

        if mc.sim.xy.size < 1:
            print("setupmcmc WARN - simulated data zero size. Check your input parameters.")
            return None, None, None


        # This is now only important if we are simulating. Shunt it
        # across.
        if mc.sim is None:
            print("WARN - sim is none. Check input arguments.")
            return None, None, None

    if mc.simulating:
        print("setupMCMC INFO: simulating and ignoretruth is " , \
              mc.ignoretruth)
        
    #print("Imported truthset INFO:", mc.sim.Parset.pars)

    print("setupmcmc INFO - MC parfile_guess:", mc.parfile_guess)
    mc.doguess(norun=debug)

    # condition trap
    if mc.guess is None:
        print("setupmcmc WARN - guess is None. Returning.")
        return None, None, None
    
    print("MC debug:")
    print("==========")
    if mc.sim is not None:
        print("mc.sim.Parset.model:", mc.sim.Parset.model)
        
    print("mc.guess.Parset.model", mc.guess.Parset.model)
    if mc.guess_parset is not None:
        print("MC INFO - guess parset:", mc.guess_parset.model)

        
    print("MCMC prior debug:")
    print("mc.lnprior.withgauss:", mc.lnprior.withgauss)
    print("mc.lnprior.gaussprior.lpars:", mc.lnprior.gaussprior.lpars)
    print("mc.lnprior.gaussprior.center:", mc.lnprior.gaussprior.center)
    print("mc.lnprior.gaussprior.covar:", mc.lnprior.gaussprior.covar)

    if debug:
        return 

    if jitterfromsamples:
        mc.estjitter_from_resamples(0.9)

    # 2025-06-27: try launching nonparamtric bootstraps now.
    mc.bootstrap_jitter(nboots = nonparam_minimizer, \
                        fsample = 1)

    #if nonparam_minimizer > 1:
    #    print("Now check nonparametric bootstrap output")
    #    return None, None, None
    
    mc.setupwalkers()
    mc.setargs_emcee()

    
    
    # Print the jitters
    if hasattr(mc, 'jitters_old'):
        
        print("Jitter comparison:")
        print("==================")
        print("Resampling:")
        print(mc.resample_jitter)
        print("Current method:")
        print(mc.jitters_old)
        
    #print("setupMCMC INFO - fracdiff:", mc.fracdiff.pars)
    #print("setupMCMC INFO - scaleguess:", mc.scaleguess)
    
    # Try serializing the arguments to disk so we can retrieve them
    # later
    mc.writeargs_emcee()

    mc.writejitterball()

    # Write the guess and truth parsets to disk
    mc.guess.Parset.writeparset("test_parset_guess.txt")
    if hasattr(mc.sim,'Parset'):
        mc.sim.Parset.writeparset("test_parset_truth.txt")

    # Now write the data to disk
    if writedata:
        mc.guess.obssrc.writeobs('test_obs_src.dat')
        mc.guess.obstarg.writeobs('test_obs_targ.dat')

    # If nonparametric bootstraps were done, write them to disk
    if np.size(mc.guess.boots_pars) > 0:
        np.save(pathboots, mc.guess.boots_pars)
        print("setupmcmc INFO - written nonparametric bootstraps to %s" \
              % (pathboots))
        stdboots = np.std(mc.guess.boots_pars, axis=0)
        print("setupmcmc INFO - stddev of nonparametric bootstraps:")
        print(stdboots)

    print("setupmcmc INFO: truths", mc.truths)
        
    # Let's see if reading this back in works...
    # mc.readjitterball(debug=True)
    
    # Get the arguments and print a helpful message...
    return mc.returnargs_emcee(Verbose=True)

def parspaths_exist(pathfit='NONE', pathsim='', Verbose=True):

    """Checks that the parameter files exist.

INPUTS

    pathfit = path to fit / guess file. Must exist.

    pathsim = path to simulation parameter file. Only required if we
    are simulating.

OUTPUTS

    pathsok = True if all the required paths are readable.

    """

    if not os.access(pathfit, os.R_OK):
        if Verbose:
            print("mcmc2d.parspaths_exist WARN - guess path not readable: %s" \
                  % (pathfit))
        return False

    # we only look for the simulation parfile if it was specified
    if len(pathsim) > 0:
        if not os.access(pathsim, os.R_OK):
            if Verbose:
                print("mcmc2d.parspaths_exist WARN - sim pars not readable: %s" % (pathsim))
            return False

    # If we got here, then all the par files we need were readable.
    return True
    
def getflatsamples(sampler=None, \
                   pathflat='test_flat_samples.npy', \
                   pathprobs='test_log_probs.npy', \
                   ntau=20, burnin=-1, Verbose=True, \
                   onlyfinite=True, \
                   lnprobmin=-np.inf):
    
    """Gets flat samples and saves them to disk.

Inputs:

    sampler = emcee sampler

    pathflat = path to write flattened samples

    pathprobs = path to write lnprobs

    ntau = multiple of autocorrelation timescale to use

    burnin = user-specified burnin interval (overrides automatic
    choice if >0)

    Verbose = print messages to screen

    onlyfinite = only accept samples with finite ln prob

    lnprobmin = only accept points with lnprob > lnprobmin

Returns:

    flat_samples = [nsamples, npars] array of flattened samples
    
    logprobs = [nsamples] array of log-probabilities from the samples

    """

    if sampler is None:
        print("mcmc2d.getflatsamples WARN - no sampler supplied")
        return

    # A few things we need
    if Verbose:
        print("mcmc2d.getflatsamples INFO - measuring autocorrelation time:")
        
    try:
        tau = sampler.get_autocorr_time()
        tauto = tau.max()

        if Verbose:
            print("mcmc2d.getflatsamples INFO - max autocorr time: %.2e" \
                  % (tauto))

        if np.isnan(tauto):
            print("mcmc2dgetflatsamples WARN - autocorr time is nan")
            tauto = 200.
            
    except:
        if Verbose:
            print("mcmc2d.getflatsamples WARN - long autocorrelation time compared to chain length")
        tauto = 50.

    # Set sample parameters but allow override from arguments
    nthrow = int(tauto * ntau)
    nthin = int(tauto * 0.5)

    if burnin > 0:
        nthrow = np.copy(burnin)

    # Report the acceptance fraction to screen
    print("mcmc2d.getflatsamples INFO - mean acceptance fraction:", \
          np.mean(sampler.acceptance_fraction), \
          np.shape(sampler.acceptance_fraction), \
          np.min(sampler.acceptance_fraction), \
          np.max(sampler.acceptance_fraction) )
        
    # now get the samples
    flat_samples = sampler.get_chain(discard=nthrow, thin=nthin, flat=True)

    # get the log probabilities for this flattened sample.
    log_probs = sampler.get_log_prob(discard=nthrow, thin=nthin, flat=True)

    if onlyfinite:
        bok = np.isfinite(log_probs)
        print("mcmc2d.getflatsamples INFO - retaining samples with finite lnprob", np.sum(bok), np.size(bok))
        bok = np.isfinite(log_probs)
        log_probs = log_probs[bok]
        flat_samples = flat_samples[bok, :]

    # select on lnprobmin. This is best left at default (-np.inf) but 
    bprob = log_probs > lnprobmin
    log_probs = log_probs[bprob]
    flat_samples = flat_samples[bprob,:]
        
    if Verbose:
        print("mcmc2d.getflatsamples INFO - flat samples shape:", \
              flat_samples.shape)
    
    # save the samples to disk...
    if len(pathflat) > 3:
        np.save(pathflat, flat_samples)

    if len(pathprobs) > 3:
        np.save(pathprobs, log_probs)        
        
    # ... and return them to the interpreter
    return flat_samples, log_probs

def loadjitterball(pathjitter='test_jitter.txt'):

    """Test-bed for jitter-ball reader, to be inserted into mcmc object
when ready"""

    # let's try piggybacking on the config parser
    if len(pathjitter) < 4:
        return

    if not os.access(pathjitter, os.R_OK):
        print("readjitterball WARN - cannot read path %s" \
              % (pathjitter))
        return
    
    config = configparser.ConfigParser()
    try:
        config.read(pathjitter)
    except:
        print("readjitterball WARN - problem reading jitter file %s" \
              % (pathjitter))
        return

    # OK now we have the config object populated. Explore it
    labels = []
    centers = []
    jitters = []
    
    for key in config['Pars'].keys():
        labels.append(key)
        centers.append(config['Pars'].getfloat(key))

        keyjit = 'j_%s' % (key)
        jitters.append(config['Jitter'].getfloat(keyjit))

        # for debugging
        #
        # print(key, centers[-1], jitters[-1])

    # Ensure the components read in are numpy arrays
    centers = np.asarray(centers)
    jitters = np.asarray(jitters)

    # For debugging
    for idum in range(len(labels)):
        print(labels[idum], centers[idum], jitters[idum] , \
              jitters[idum]/np.abs(centers[idum]) )
        
    #print(labels)
    #print(centers)
    #print(jitters)

                       
