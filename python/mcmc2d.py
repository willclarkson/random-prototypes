#
# mcmc2d.py
#

#
# WIC 2024-08-16 - use sim2d and fit2d to set up and explore 2d data
# with mcmc
#

import os, time
import numpy as np

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
                 chainlen=40000):

        # Parameters for simulation and for guess
        self.parfile_sim = parfile_sim[:]
        self.parfile_guess = parfile_guess[:]
    
        # Simulation and guess objects
        self.sim = None
        self.guess = None

        # Method that will be used for ln(posterior), its arguments
        self.methpost = lnprobs2d.lnprob
        self.argspost = ()
        self.lnlike = None
        self.lnprior = None
        self.guess1d = np.array([])

        # For perturbing the initial guess (1d) for the minimizer
        self.nudge_guess1d = 1.0e-2
        self.nudge_seed = None

        # The minimizer output, and some settings
        self.minimizer_soln = None
        self.minimizer_method = 'Nelder-Mead'
        self.minimizer_options = {'maxiter':5000}
        self.guess_parset = None

        # Comparison between the guess and the 'truth' parameters
        # (useful for scaling the jitter ball)
        self.fracdiff = None

        # Quantities for setting up the mcmc runs
        self.guess1d_refined = np.array([]) # convenience view
        self.scaleguess = np.array([])
        self.pos = np.array([])
        self.ndim = 1
        self.nchains = -1
        self.fjitter = 3.
        self.chainlen = chainlen

        # Arguments to send to mcmc runs
        self.args_ensemble = {}
        self.args_run = {}
        
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

        """Assembles expected arguments for minimizert and/or emcee"""

        self.argspost = (self.guess.PGuess, self.guess.obstarg, \
                         self.guess.Parset, self.lnprior, self.lnlike)
        
    def setuplnprior(self):

        """Sets up the ln(prior) object for minimization and/or emcee"""

        self.lnprior = Prior(self.guess.Parset)

    def setuplnlike(self):

        """Sets up the ln(likelihood) object for minimization and/or emcee"""
        
        self.lnlike = Like(self.guess.Parset, self.guess.PGuess, \
                           self.guess.obstarg)

    def doguess1d(self):

        """Creates and perturbs initial guess for minimizer"""

        self.setupguess1d()
        self.nudgeguess1d()
        
    def setupguess1d(self):

        """Creates 1d initial guess for minimizer"""

        self.guess1d = np.copy(self.guess.Parset.pars)

    def nudgeguess1d(self, seed=None):

        """Perturbs the 1d initial guess for input (e.g. if truth values have
been given as the guess)

"""

        rng = np.random.default_rng(self.nudge_seed)
        pertns = rng.normal(size=np.size(self.guess1d)) \
            * self.nudge_guess1d * self.guess1d

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

        self.getguess_from_minimizer()
            
    def getguess_from_minimizer(self):

        """Shunts the minimizer solution into the refined guess"""

        if not hasattr(self.minimizer_soln,'x'):
            print("refinedguess WARN - minimizer solution has no x")
            return

        self.guess1d_refined = np.copy(self.minimizer_soln.x)
        
    def populate_guess_parset(self):

        """Shunts the minimizer-found parameters into a new paramset object
for convenient comparison with the truth parameters"""

        if self.minimizer_soln is None:
            return

        self.guess_parset = Pars1d(self.guess1d_refined, \
                                   self.guess.Parset.nnoise, \
                                   self.guess.Parset.nshape, \
                                   self.guess.Parset.nmix)
        
    def calcfracdiff_truth_guess(self):

        """Compares the guess parameter-set with the truth parameter-set"""

        PP = Pairset(self.sim.Parset, self.guess_parset)

        # Find fractional difference (of guess) in matching parameters
        self.fracdiff = PP.fracdiff()
        self.fracdiff.pars = np.asarray(self.fracdiff.pars, 'float64')
        
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

        self.scaleguess = np.asarray(self.fracdiff.pars) * self.fjitter

    def initwalkers(self):

        """Sets up walker initial conditions"""

        pertn = np.random.randn(self.nchains, self.ndim)
        magn = self.scaleguess * self.guess1d_refined
        self.pos = self.guess1d_refined + pertn * magn[None, :]

        print("initwalkers INFO:", self.guess1d_refined.dtype, \
              pertn.dtype, magn.dtype, self.scaleguess.dtype)
        
    def setupwalkers(self):

        """Wrapper - sets up the walker characteristics for mcmc"""

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
        

    def setargs_emcee(self):

        """Sets up the arguments to send to emcee with multiprocessor"""

        self.setargs_ensemblesampler()
        self.setargs_emceerun()

    def returnargs_emcee(self, Verbose=True):

        """Returns the arguments to the interpreter, prints a helpful message"""

        if Verbose:
            print("Returning emcee arguments:")
            print("esargs, runargs")
            print(" ")
            print("Now execute:")
            print("with Pool() as pool:")
            print("      sampler = emcee.EnsembleSampler(**esargs, pool=pool)")
            print("      sampler.run_mcmc(**runargs)")

        return self.args_ensemble, self.args_run
        
