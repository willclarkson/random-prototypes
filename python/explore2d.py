#
# explore2d.py
#

#
# 2028-08-13 WIC - methods to explore 2D mapping using MCMC. Much of
# this is OO refactored from the prototype fittwod.py.
#

import numpy as np

# transformations, noise model
import unctytwod
import noisemodel2d
import mixfgbg

from fitpoly2d import Leastsq2d, Patternmatrix

from weightedDeltas import CovarsNx2x2

class Pars1d(object):

    """Parameters for transformation and any other 'model' parameters
including noise model and mixture model. A 1d parameter array can be
input (and split) or the separate pieces can be input and fused. If
the 1d array is supplied, any separate components supplied are
ignored.

Example: supply 1D parameter + index counts to split:

    PP = Pars1d(ppars, nnoise, nshape, nmix)

Example: supply separate model components to fuse into a 1D array:

    QQ = Pars1d(model=PP.model, noise=PP.noise, symm=PP.symm, mix=PP.mix)

Relevant attributes:

    PP.pars = 1D array [transformation, noise, shape, mixture]

    PP.model = transformation model

    PP.noise = noise vs mag model

    PP.symm = [stdy/stdx, corrxy] model

    PP.mix = [foutly, vbackg] model

    

    """

    def __init__(self, pars=np.array([]), nnoise=0, nshape=0, nmix=0, \
                 model=np.array([]), noise=np.array([]), symm=np.array([]), \
                 mix=np.array([])):

        # 1D array of parameters as expected by e.g. minimize. Can be
        # a numpy array or a list
        self.pars = pars

        # parameter-splitting quantities
        self.nnoise = nnoise
        self.nshape = nshape
        self.nmix = nmix

        # Partitioned parameters
        self.model = np.copy(model) # the transformation
        self.noise = np.copy(noise) # the noise vs mag model
        self.symm = np.copy(symm) # noise shape [stdy/stdx, corrxy]
        self.mix = np.copy(mix)  # mixture model [ffg, var_backg]

        # Indices for the model pieces (faster than the general
        # loop-based partitioning at the cost of hard-coding. But
        # we're doing that anyway for the model pieces!):
        self.lmodel = np.array([])
        self.lnoise = np.array([])
        self.lsymm = np.array([])
        self.lmix = np.array([])
        
        # partition the input model parameters if 1D supplied...
        if np.size(pars) > 0:
            self.setupindices()
            self.partitionmodel()

        # ... or, if not, fuse any supplied pieces together
        else:
            if np.size(model) > 0:
                self.fusemodel()
            
    def insertpars(self, p=np.array([]) ):

        """Replaces the parameters with input.

Inputs:

        p = [transf, noise, shape, mix] parameters

Returns: None. Updates the following attributes:

        self.pars = 1D array of parameters. Makes a copy of the input.

"""

        self.pars = np.copy(p)

    def updatepars(self, p=np.array([]) ):

        """Inputs and partitions input parameters. 

    Inputs:

        p = [transf, noise, shape, mix] parameters

    Returns: None. Updates the following attributes:

        self.pars = 1D array of parameters. Makes a copy of the input.

        self.model = transformation parameters

        self.noise = noise model parameters

        self.symm = noise shape parameters
        
        self.mix = mixture model parameters

"""

        self.insertpars(p)
        self.partitionmodel()

    def setupindices(self):

        """Sets up the indices corresponding to each model parameter.

Inputs: None. The following must be set:

        self.pars = 1d array of model parameters

        self.nnoise, self.nsymm, self.nmix = number of parameters
        describing the noise mag model, noise shape model and mixture
        model, respectively.

Returns: None. Updates the following:

        self.lmodel, self.lnoise, self.lsymm, self.lmix = index arrays
        describing which parameters in self.pars are transformation,
        noise, shape, mixture, respectively.

        """

        npars = np.size(self.pars)
        lpars = np.arange(npars, dtype='int')
        
        # initialise the non-transformation parameters. Must be
        # integer to be used as indices
        self.lnoise = np.array([], dtype='int')
        self.lsymm = np.copy(self.lnoise)
        self.lmix = np.copy(self.lnoise)

        # Cut down the index array progressively. This is similar to
        # what splitmodel does as a loop, but we lay it out explicitly
        # here since we're hard-coding the different parts of the
        # model anyway.
        if self.nmix > 0:
            self.lmix = lpars[-self.nmix::]
            lpars = lpars[0:-self.nmix]

        if self.nshape > 0:
            self.lsymm = lpars[-self.nshape::]
            lpars = lpars[0:-self.nshape]
            
        if self.nnoise > 0:
            self.lnoise = lpars[-self.nnoise::]
            lpars = lpars[0:-self.nnoise]

        # OK what's left is the transformation parameter index set
        self.lmodel = lpars

    def partitionmodel(self):

        """Partitions model parameters using indices already built by setupindices(). 

Inputs: None. 

Returns: None.


"""

        if np.size(self.lmodel) < 1:
            return
        
        self.model = self.pars[self.lmodel]
        self.noise = self.pars[self.lnoise]
        self.symm = self.pars[self.lsymm]
        self.mix = self.pars[self.lmix]

    def fusemodel(self):

        """If model parameters were provided separately, fuse them together."""

        if np.size(self.model) < 1:
            return

        self.pars = np.copy(self.model)

        self.nnoise = np.size(self.noise)
        self.nshape = np.size(self.symm)
        self.nmix = np.size(self.mix)

        if self.nnoise > 0:
            self.pars = np.hstack(( self.pars, self.noise ))
        if self.nshape > 0:
            self.pars = np.hstack(( self.pars, self.symm ))
        if self.nmix > 0:
            self.pars = np.hstack(( self.pars, self.mix ))

        # Now that we've done this, set up the indices for consistency
        self.setupindices()
            
    def splitmodel(self):

        """Splits the model 1d parameters into transformation and the other parameters.

    Inputs:

        None. Acts on:

        self.pars = 1D array with the parameters: [transformation,
        noise, vars, mixture]

    Returns: None. Updates the following attributes:

        self.model = transformation parameters

        self.noise = [A, B, C] parameters for noise vs magnitude

        self.symm = [stdy/stdx, corrxy] parameters for noise

        self.mix = [ffg, var_bg] mixture model parameters

        """

        lnums = [self.nmix, self.nshape, self.nnoise]
        
        self.model, lsplit = self.splitpars(self.pars, lnums)
        self.noise = lsplit[2]
        self.symm = lsplit[1]
        self.mix = lsplit[0]
        
    def splitpars(self, pars, nsplit=[]):

        """Splits a 1d array into sub-arrays, skimming off the nsplit entries
from the end at each stage. Like splitpars() but the nsplits are
generalized into a loop. 

    Inputs:

        pars = 1d array of parameters

        nsplit = [n0, n1, ... ] list of last-n indices to split off at each stage.

    Returns: 

        allbutsplit = all the pars not split off into a subarray
        
        [p1, p2, ...] = list of pars split off from the far end, *in the same order as the nsplit list*. 

    Example:

        x = np.arange(10)
        fittwod.splitpars(x,[3,2])
    
        returns:

             (array([0, 1, 2, 3, 4]), [array([7, 8, 9]), array([5, 6])])

    """

        # handle scalar input for nsplit
        if np.isscalar(nsplit):
            nsplit = [nsplit]
    
        # if no splits, nothing to do
        if len(nsplit) < 1:
            return pars

        lsplit = []
        allbut = np.copy(pars)

        for isplit in range(len(nsplit)):
            allbut, split = self.splitlastn(allbut, nsplit[isplit])
            lsplit = lsplit + [split]
        

        return allbut, lsplit
            
    def splitlastn(self, pars=np.array([]), nsplit=0):

        """Splits a 1D array into its [0-nsplit] and and [-nsplit::] pieces.

    Inputs:

        pars = [M]  array of parameters

        nsplit = number of places from the end of the array that will be
        split off

    Returns:

        first = [M-nsplit] array before the split

        last = [nsplit] array after the split

        """

        # Nothing to do if nothing provided
        if np.size(pars) < 1:
            return np.array([]), np.array([])

        # Cannot do anything if the lengths do not match
        if np.size(pars) < nsplit:
            return pars, np.array([])
    
        # Nothing to do if no split
        if nsplit < 1:
            return pars, np.array([])

        return pars[0:-nsplit], pars[-nsplit::]


class Obset(object):

    """Convenience-object to hold positions, covariances, and other
information like apparent magnitudes for hypothetical observations."""

    def __init__(self, xy=np.array([]), covxy=np.array([]), \
                 mags=np.array([]), isfg=np.array([]) ):

        self.xy = np.copy(xy)
        self.covxy = np.copy(covxy)
        self.mags = np.copy(mags)
        self.isfg = np.copy(isfg)

    # Self-checking methods could come here.

class Simdata(object):

    """Methods and fake data for MCMC simulations"""

    def __init__(self, npts=200, deg=1, Verbose=False):

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

        # "Truth" model parameters
        self.transf = unctytwod.Poly
        self.polytransf = 'Chebyshev'
        self.deg = deg
        self.transfexpon = 1.5
        self.transfscale = 1.
        self.pars_transf = np.array([])
        self.seed_params = None
        
        # Non-transformation model parameters
        self.pars_noise = [-4., -20., 2.]
        self.pars_asymm = [0.8, 0.1]
        self.pars_mix = [-1., -8.5]
        self.islog10_mix = [True, True]

        # parameters for "extra" (i.e. unmodeled) noise
        self.pars_extra_noise = []
        self.pars_extra_asymm = []
        
        # Mixture model / outlier parameters
        self.seed_outly = None

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
        self.Obstarg = Obset()
        
    def countpars(self):

        """Counts the number of (non-transformation) model parameters."""

        self.npars_noise = np.size(self.pars_noise)
        self.npars_asymm = np.size(self.pars_asymm)
        self.npars_mix = np.size(self.pars_mix)
        
    def makefakexy(self):

        """Makes random uniform xy points within bounds set in the instance"""

        rng = np.random.default_rng(self.seed_data)
        
        self.xy = np.random.uniform(size=(self.npts,2))
        self.xy[:,0] = self.xy[:,0]*(self.xmax-self.xmin) + self.xmin
        self.xy[:,1] = self.xy[:,1]*(self.ymax-self.ymin) + self.ymin

    def makefakemags(self):

        """Makes power-law apparent magnitudes"""

        rng = np.random.default_rng(self.seed_mag)
        sraw = rng.power(self.magexpon, self.npts)
        self.mags = sraw *(self.maghi - self.maglo) + self.maglo

    def makemagcovars(self):

        """Generates magnitude-dependent covariances."""

        self.Cxy = self.getmagcovars(self.pars_noise, self.pars_asymm)

    def makeextracovars(self):

        """Generates (magnitude-dependent) covariance due to unmodeled
noise

        """

        self.CExtra = self.getmagcovars(self.pars_extra_noise, \
                                        self.pars_extra_asymm)

        
    def getmagcovars(self, \
                     pars_noise=np.array([]), \
                     pars_asymm=np.array([]), \
                     mags=np.array([]) ):

        """Returns magnitude-dependent covariances"""

        if np.size(pars_noise) < 1:
            pars_noise = self.pars_noise
        if np.size(pars_asymm) < 1:
            pars_asymm = self.pars_asymm
        if np.size(mags) < 1:
            mags = self.mags

        return noisemodel2d.mags2noise(pars_noise, pars_asymm, mags)
        
        
    def makeunifcovars(self):

        """Makes uniform covariances"""

        # Identical to makemagcovars except only the first noise
        # parameter is used.
        
        self.Cxy = noisemodel2d.mags2noise(self.pars_noise[0], self.mags, \
                                           self.pars_asymm)

    def assignoutliers(self):

        """Assigns outlier status to a (uniform) random permutation of
objects"""

        self.isoutly = np.repeat(False, self.npts)

        # parse the mixture parameters
        foutliers = mixfgbg.parsefraction(self.pars_mix[0], \
                                          self.islog10_mix[0])
        
        if foutliers <= 0. or foutliers > 1:
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

        
    def makepars(self):

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
                                  kind=self.polytransf, \
                                  checkparsy=True)
        
    def setupxytran(self):

        """Propagates the truth positions into the target frame"""

        self.PTruth.propagate()
        self.xytran = np.copy(self.PTruth.xytran)
        self.covtran = np.copy(self.PTruth.covtran)
        
        self.CTran = CovarsNx2x2(self.PTruth.covtran)

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

        self.Parset = Pars1d(model=self.pars_transf, \
                             noise=self.pars_noise, \
                             symm=self.pars_asymm, \
                             mix=self.pars_mix)
        
    def packagetargetdata(self):

        """Packages the target data into an obset object to reduce the number
of arguments to the minimizer"""

        self.Obstarg = Obset(self.xytarg, self.covtran, \
                             self.mags, ~self.isoutly)
        
    def generatedata(self):

        """Wrapper - generates fake data"""

        # Baseline x, y, covars in source frame
        self.makefakexy()
        self.makefakemags()
        self.makemagcovars()
        self.assignoutliers()
        
        # Define the transformed frame and propagate to it
        self.makepars()
        self.setuptransftruth()
        self.setupxytran()
        self.initnudges()

        # set up any outliers and/or extra unmodeled covariances
        self.makeoutliers()
        self.makeextracovars()

        # Apply the above to produce x, y nudges
        self.initnudges()
        self.makenudges()
        self.applynudges()

        # Package the fit parameters into a 1d array. See self.Parset
        self.packagemodelpars()
        
        # Package the target data into object expected by the fitter
        self.packagetargetdata()
        
### SHORT test routines come here.

def testsplit(nnoise=3, nshape=2, nmix=2):

    """Tests the splitting behavior"""

    transf = np.arange(6)
    pnoise = np.arange(nnoise)+10
    pshape = np.arange(nshape)+100
    pmix = np.arange(nmix) + 1000

    ppars = np.hstack(( transf, pnoise, pshape, pmix ))

    PP = Pars1d(ppars, nnoise, nshape, nmix)

    print("Original:")
    print(ppars)

    # Set up the indices - what do we get?

    print(PP.model)
    print(PP.noise)
    print(PP.symm)
    print(PP.mix)

    # Now try fusing these into a separate object
    QQ = Pars1d(model=PP.model, noise=PP.noise, symm=PP.symm, mix=PP.mix)

    print("Fused:")
    print(QQ.pars)
    
    # Now update the parameters
    PP.updatepars(0.-ppars)

    print("Updated:")
    print(PP.pars)
    print(PP.model)
    print(PP.noise)
    print(PP.symm)
    print(PP.mix)
    
def testsim():

    """Tests wrapper for generating data"""

    SD = Simdata()
    SD.generatedata()
