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

from scipy.optimize import minimize

import sim2d
from parset2d import Pars1d, Pairset
from fit2d import Guess
import lnprobs2d
from lnprobs2d import Prior, Like

# utilities for converting linear parameters back and forth
import sixterm2d

# For serializing sim info to disk
import configparser


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
                 Verbose=True):

        # Control variables
        self.Verbose = Verbose

        # If simulating, ignore the truth values when setting up the
        # guess?
        self.ignoretruth = ignoretruth

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
        self.minimizer_options = {'maxiter':5000}
        self.guess_parset = None

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
        
    def dosim(self):

        """Wrapper - imports simulation parameters and generates simulated
dataset"""

        self.setupsim()
        self.runsim()
        
    def setupsim(self):

        """Sets up the simulation object"""

        self.sim = sim2d.Simdata()
        self.sim.loadconfig(self.parfile_sim)

    def runsim(self):

        """Generates the simulated dataset"""

        self.sim.generatedata()

    def setupguess(self):

        """Sets up the guess object"""

        self.guess = Guess(self.sim.Obssrc, self.sim.Obstarg)
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

        self.argspost = (self.guess.PGuess, self.guess.obstarg, \
                         self.guess.Parset, self.lnprior, self.lnlike)
        
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

        if Verbose:
            print("mcmc2d.runminimizer INFO - starting minimizer...")
        t0 = time.time()
        ufunc = lambda *args: 0.-self.methpost(*args)

        self.minimizer_soln = minimize(ufunc, self.guess1d, \
                                       args=self.argspost, \
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
            with open('%s_showargs.pickle' % (stemargs), 'wb') as f:
                pickle.dump(self.args_show, f)

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
            # Do a linear leastsq fit to the data to serve as the
            # initial guess for the minimizer
            if not self.ignoretruth:
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
              pathboots='test_boots.npy'):

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

Returns:

    esargs = dictionary of arguments for the ensemble sampler

    runargs = dictionary of arguments for running mcmc

    showargs = dictionary of arguments for analysing and showing the results

    """

    mc = MCMCrun(pathsim, pathfit, chainlen, pathprior, \
                 pathjitter=pathjitter, ignoretruth=ignoretruth, \
                 doboots_poly=doboots_poly, nboots=nboots, \
                 lsq_uncty_trick=lsq_uncty_trick, \
                 boots_ignoreweights=boots_ignoreweights)
    mc.dosim()
    mc.doguess(norun=debug)
    
    print("MC debug:")
    print(mc.sim.Parset.model)
    print(mc.guess.Parset.model)
    if mc.guess_parset is not None:
        print(mc.guess_parset.model)

    print("MCMC prior debug:")
    print(mc.lnprior.withgauss)
    print(mc.lnprior.gaussprior.lpars)
    print(mc.lnprior.gaussprior.center)
    print(mc.lnprior.gaussprior.covar)

    if debug:
        return 

    mc.setupwalkers()
    mc.setargs_emcee()

    #print("setupMCMC INFO - fracdiff:", mc.fracdiff.pars)
    #print("setupMCMC INFO - scaleguess:", mc.scaleguess)
    
    # Try serializing the arguments to disk so we can retrieve them
    # later
    mc.writeargs_emcee()

    mc.writejitterball()

    # Write the guess and truth parsets to disk
    mc.guess.Parset.writeparset("test_parset_guess.txt")
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
        
        
    # Let's see if reading this back in works...
    # mc.readjitterball(debug=True)
    
    # Get the arguments and print a helpful message...
    return mc.returnargs_emcee(Verbose=True)
    
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

                       
