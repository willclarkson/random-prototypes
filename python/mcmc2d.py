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
from fit2d import Guess, lnprobs2d
from lnprobs2d import Prior, Like



class MCMCrun(object):

    """Sets up for an emcee run that would be performed via
multiprocessing.

    """

    def __init__(self, parfile_sim='', parfile_guess='', \
                 chainlen=40000, Verbose=True):

        # Contol variables
        self.Verbose = Verbose
        
        # Parameters for simulation and for guess
        self.parfile_sim = parfile_sim[:]
        self.parfile_guess = parfile_guess[:]
    
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
        self.nudgescale_guess1d = 1.0e-3
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

        if not hasattr(self.sim, 'pars_transf'):
            return

        self.guess1d = np.copy(self.sim.pars_transf)
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

        self.lnprior = Prior(self.guess.Parset)

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
        
        # for pointing arguments
        if not hasattr(self.guess, 'transf'):
            return

        # If the guess has a tangent point, set the nudge accordingly
        if self.guess.hastangentpoint:
            self.nudge_guess1d[0] = self.nudge_pointing_arcsec / 3600.
            self.nudge_guess1d[1] = self.nudge_pointing_arcsec / 3600.
        
    def nudgeguess1d(self, seed=None):

        """Perturbs the 1d initial guess for input (e.g. if truth values have
been given as the guess)

"""

        # Ensure the nudge guess is appropriately scaled
        if np.size(self.nudge_guess1d) < 1:
            self.scalenudgeguess()
        
        rng = np.random.default_rng(self.nudge_seed)
        #pertns = rng.normal(size=np.size(self.guess1d)) \
        #    * self.nudgescale_guess1d * self.guess1d

        pertns = rng.normal(size=np.size(self.guess1d)) \
            * self.nudgescale_guess1d

        
        self.guess1d += pertns
        
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
        
        PP = Pairset(self.sim.Parset, self.guess_parset)

        # Find fractional difference (of guess) in matching
        # parameters. Force this to become an np float array while I
        # work out why it isn't that by default...
        self.fracdiff = PP.fracdiff()
        self.fracdiff.pars = np.asarray(self.fracdiff.pars, 'float64')

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
        if not hasattr(self.fracdiff, 'pars'):
            return
        
        self.scaleguess = np.asarray(self.fracdiff.pars) * self.fjitter

    def initwalkers(self):

        """Sets up walker initial conditions"""

        pertn = np.random.randn(self.nchains, self.ndim)
        magn = self.scaleguess * self.guess1d_refined
        self.pos = self.guess1d_refined + pertn * magn[None, :]

        #print("initwalkers INFO:", self.guess1d_refined.dtype, \
        #      pertn.dtype, magn.dtype, self.scaleguess.dtype)
        
    def setupwalkers(self):

        """Wrapper - sets up the walker characteristics for mcmc"""

        # If we have truth parameters, use them to set the scale for
        # the jitter ball for the mcmc initial state
        self.calcfracdiff_truth_guess()
        
        self.ndimfromguess()
        self.setnchains()
        self.setjitterscale()
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
            with open('%s_esargs.pickle' % (stemargs), 'wb') as f:
                pickle.dump(self.args_ensemble, f)
        
    def doguess(self):

        """Wrapper - sets up and performs initial fit to the data to serve as
our initial state for MCMC exploration.

        When this is done, self.guess_parset contains the minimizer
        fit parameters.

        """

        # Sets up the guess object
        self.setupguess()

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
            self.guessfromlstsq()

        # Setup and run the minimizer using the lstsq fit as input,
        # and shunt the result across to the guess object
        self.setupfitargs()
        self.guessforminimizer()
        self.runminimizer(Verbose=self.Verbose)
        self.populate_guess_parset()
        
### Some methods follow that we want to be able to use from the
### interpreter. Once an mcmc run is done, the interpreter will have
### "samples" available to use. So we write the flat samples to disk

def setupmcmc(pathsim='test_sim_mixmod.ini', \
             pathfit='test_guess_input.ini', \
             chainlen=40000):

    """Sets up for mcmc simulations. 

Inputs:

    pathsim = path to parameter file for simulating data

    pathfit = path to parameter file for assembling the guess

    chainlen = chain length for mcmc

Returns:

    esargs = dictionary of arguments for the ensemble sampler

    runargs = dictionary of arguments for running mcmc

    showargs = dictionary of arguments for analysing and showing the results

"""

    mc = MCMCrun(pathsim, pathfit, chainlen)
    mc.dosim()
    mc.doguess()

    print("MC debug:")
    print(mc.sim.Parset.model)
    print(mc.guess.Parset.model)
    print(mc.guess_parset.model)
    
    mc.setupwalkers()
    mc.setargs_emcee()

    #print("setupMCMC INFO - fracdiff:", mc.fracdiff.pars)
    #print("setupMCMC INFO - scaleguess:", mc.scaleguess)
    
    # Try serializing the arguments to disk so we can retrieve them
    # later
    mc.writeargs_emcee()
    
    # Get the arguments and print a helpful message...
    return mc.returnargs_emcee(Verbose=True)
    
def getflatsamples(sampler=None, \
                   pathflat='test_flat_samples.npy', \
                   pathprobs='test_log_probs.npy', \
                   ntau=20, burnin=-1, Verbose=True):
    
    """Gets flat samples and saves them to disk.

Inputs:

    sampler = emcee sampler

    pathflat = path to write flattened samples

    pathprobs = path to write lnprobs

    ntau = multiple of autocorrelation timescale to use

    burnin = user-specified burnin interval (overrides automatic
    choice if >0)

    Verbose = print messages to screen

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
        
    except:
        if Verbose:
            print("mcmc2d.getflatsamples WARN - long autocorrelation time compared to chain length")
        tauto = 50.

    # Set sample parameters but allow override from arguments
    nthrow = int(tauto * ntau)
    nthin = int(tauto * 0.5)

    if burnin > 0:
        nthrow = np.copy(burnin)
        
    # now get the samples
    flat_samples = sampler.get_chain(discard=nthrow, thin=nthin, flat=True)

    # get the log probabilities for this flattened sample.
    log_probs = sampler.get_log_prob(discard=nthrow, thin=nthin, flat=True)
    
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


