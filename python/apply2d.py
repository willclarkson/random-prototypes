#
# apply2d.py
#

# WIC 2024-11-12 - methods to apply MCMC2d results

import os
import copy
import numpy as np
from scipy import stats

import time

import matplotlib.pylab as plt
from matplotlib.collections import LineCollection
plt.ion()

# While developing, import the parent modules for the transformations
# and data
import unctytwod
import parset2d
import obset2d

# For computing moments
import moments2d

# for occasional polyfitting
from fitpoly2d import Leastsq2d
import sixterm2d

# for drawing samples from the covariances
from weightedDeltas import CovarsNx2x2

# convenient binning statistics
from binstats2d import Binstats

class Evalset(object):

    """Sets of evaluations of the transformation."""

    # Should support both repeated evaluations on the same data and
    # the same parameters on different datasets

    def __init__(self, \
                 pathpset='test_parset_guess.txt', \
                 pathflat='test_flat_fitPoly_500_order1fit1_noprior_run2.npy', \
                 pathobs='', \
                 pathlnprobs='', \
                 neval=100, \
                 transf=None):

        # Paths to any input files
        self.pathpset = pathpset[:]
        self.pathflat = pathflat[:]
        self.pathobs = pathobs[:]
        self.pathlnprobs = pathlnprobs[:]
        
        # The transformation object, flat samples
        self.transf = transf # allow passing input
        self.parsamples = None
        self.pset = None
        self.obset = None

        # evaluations of ln(prob)
        self.lnprob = np.array([])
        
        # The non-nuisance parameters
        self.modelsamples = np.array([])
        self.lmodel = np.array([])
        
        # Number to evaluate, xi, eta samples
        self.neval = np.copy(neval)
        self.initsamples_xieta()

        # Are we keeping the xy samples? (Usually no, unless we want
        # to perform statistics on parametric monte-carlo to test the
        # routines)
        self.keepxysamples = False
        self.initsamples_xy()

        # We might be tracking a third dimension (e.g. logprob) to plot
        self.initsamples_z()
        
        # datapoints for evaluation
        self.xy = None
        self.covxy = None

        # Source-frame covariances as an object with methods we may
        # want
        self.covobj = None
        
        # If producing datapoints by perturbing a "truth" set, use
        # these for the reference points
        self.xyref = np.array([])

        # Propagated covariance from the source frame (to enable
        # checking against the covariance of the propagated positions)
        self.cov_propagated = None
        
        # Covariances in the sample plane, and some other statistics
        # of interest
        self.med_xieta = np.array([])
        self.cov_xieta = None
        self.skew_xieta = np.array([])
        self.kurt_xieta = np.array([])
        
        
        # If generating a grid, use these parameters
        self.grid_nxcoarse = 15 # 5
        self.grid_nycoarse = 15 # 5
        self.grid_nxfine = 10 # 20
        self.grid_nyfine = 10 # 20
        self.grid_whichline = np.array([]) # useful for lineplots
        
    def getsamples(self):

        """Loads the parameter samples"""

        self.transf, self.parsamples, self.pset = \
            loadparsamples(self.pathpset, self.pathflat)

        # which flat samples are model parameters?
        self.lmodel = self.pset.lmodel

    def getlnprobs(self):

        """Loads ln(prob) evaluations for the flat samples"""

        if len(self.pathlnprobs) < 3:
            return
        
        if not os.access(self.pathlnprobs, os.R_OK):
            return

        self.lnprob = np.load(self.pathlnprobs)
        
        # Might want to do some sanity checking on whether the lnprobs
        # match the number of evaluations?
        
    def getobs(self):

        """Loads observations from disk"""

        # Defensive programming
        if len(self.pathobs) < 3:
            return
        
        self.obset = loadobset(self.pathobs)
        
    def covfromobs(self):

        """Passes the covariance and position information from the input
observation file to the self.xy, self.covxy quantities"""

        if not hasattr(self.obset, 'covxy'):
            return

        if np.size(self.obset.covxy) < 3:
            return

        self.covxy = np.copy(self.obset.covxy)
        self.xy = np.copy(self.obset.xy)

        # populate the covariance object to draw samples
        self.covobj = CovarsNx2x2(self.covxy)

    def genposran(self, ngen=100, seed=None):

        """Generates random uniform points over the range defined by the
transformation xmin, xmax, ymin, ymax"""

        # Convenience views for the limits
        xmin = self.transf.xmin
        xmax = self.transf.xmax
        ymin = self.transf.ymin
        ymax = self.transf.ymax
        
        rng = np.random.default_rng(seed)
        x = rng.uniform(xmin, xmax, ngen)
        y = rng.uniform(ymin, ymax, ngen)

        self.xy = np.stack((x, y), axis=1)

    def genposgrid(self, nx=7, ny=7):

        """Generates positions on a simple grid (no dual-scaling).

Inputs:

        nx = number of grid points in x

        ny = number of grid points in y

Returns:

        No returns - updates attribute self.xy

"""

        vx = np.linspace(self.transf.xmin, self.transf.xmax, \
                         nx, endpoint=True)
        vy = np.linspace(self.transf.ymin, self.transf.ymax, \
                         ny, endpoint=True)

        xx, yy = np.meshgrid(vx, vy, indexing='ij')

        self.xy = np.stack( (xx.ravel(), yy.ravel()), axis=1)
        
    def gencovunif(self, major=1.0e-4, minor=0.7e-4, rotdeg=0.):

        """Creates uniform set of covariances for input positions"""

        # We're going to use the positions to propagate the
        # covariances.
        if np.ndim(self.xy) < 1:
            return

        npts = np.shape(self.xy)[0]

        # This time we create the covariance object first and read off
        # the covariances from it:
        majors = np.repeat(major, npts)
        minors = np.repeat(minor, npts)
        posans = np.repeat(rotdeg, npts)

        self.covobj = CovarsNx2x2(majors=majors, \
                                  minors=minors, \
                                  rotDegs=posans)

        self.covxy = np.copy(self.covobj.covars)
        
    def propagate_covar(self):

        """Propagates the covariance using the transformation's
methods. Useful to check against the covariance of tha propagated
samples."""

        if self.transf is None:
            return

        # ensure xytran is populated
        if np.size(self.transf.xytran) < 1:
            self.transf.initxytran()
            self.transf.propagate()
        else:
            self.transf.trancov()

        # now pass this up to the instance...
        self.cov_propagated = CovarsNx2x2(self.transf.covtran)

        # ... and compute the eigen rep
        self.cov_propagated.eigensFromCovars()
        
    def checkneval(self):

        """Ensures neval <= sample size"""

        # May want to get more sophisticated to allow for sampling
        # over the data
        if np.ndim(self.parsamples) < 2:
            return
        
        self.neval = np.min([self.neval, self.parsamples.shape[0] ])
        
    def gengrid(self):

        """Generates grid of test points for evaluation of the
transformation"""

        vxfine = np.linspace(self.transf.xmin, self.transf.xmax, \
                             self.grid_nxfine, endpoint=True)
        vyfine = np.linspace(self.transf.ymin, self.transf.ymax, \
                             self.grid_nyfine, endpoint=True)

        yzer = np.zeros(np.size(vxfine))
        xzer = np.zeros(np.size(vyfine))
        
        vxcoarse = np.linspace(self.transf.xmin, self.transf.xmax, \
                             self.grid_nxcoarse, endpoint=True)

        vycoarse = np.linspace(self.transf.ymin, self.transf.ymax, \
                               self.grid_nycoarse, endpoint=True)

        # build the coarse/fine grid:
        x = np.array([])
        y = np.array([])

        # identify which grid feature (useful when doing lineplots)
        whichg = np.array([])
        
        # horizontal lines...
        whichline = -1
        for icoarse in range(np.size(vycoarse)):
            x = np.hstack(( x, vxfine ))
            y = np.hstack(( y, yzer + vycoarse[icoarse] ))

            # grid line ID (useful for lineplots)
            whichline += 1
            wgrid = np.repeat(whichline, yzer.size)
            whichg = np.hstack((whichg, wgrid))
            
        # ... and vertical
        for jcoarse in range(np.size(vxcoarse)):
            y = np.hstack(( y, vyfine ))
            x = np.hstack(( x, xzer + vxcoarse[jcoarse] ))

            whichline += 1
            wgrid = np.repeat(whichline, xzer.size)
            whichg = np.hstack((whichg, wgrid))
            
        # Now stack them together for the instance
        self.xy = np.vstack(( x, y )).T

        # set the grid line ID marker
        self.grid_whichline = np.asarray(whichg, 'int')

    def initsamples_xy(self):

        """Initialises samples array in source frame"""

        self.samples_x = np.array([])
        self.samples_y = np.array([])

    def setupsamples_xy(self):

        """Populates zeros-arrays with samples in x, y"""

        self.samples_x, self.samples_y = self.blanksamples()
        
    def initsamples_xieta(self):

        """Initializes xi, eta samples arrays"""

        self.samples_xi = np.array([])
        self.samples_eta = np.array([])

    def setupsamples_xieta(self):

        """Sets up arrays to hold the transformed samples"""

        self.samples_xi, self.samples_eta = self.blanksamples()
        
    def blanksamples(self):

        """Creates zero-arrays for samples, shaped [ndata, nsamples]"""
        
        if np.ndim(self.xy) < 2:
            return np.array([]), np.array([]) 
        
        # Separate arrays for each coordinate for now.
        ndata = np.shape(self.xy)[0]

        return np.zeros((self.neval, ndata)), np.zeros((self.neval, ndata))

    def initsamples_z(self):

        """Initializes (scalar) z samples"""

        self.samples_z = np.array([])
        
    def setupsamples_z(self):

        """Sets up scalar z array for samples"""

        self.samples_z, _ = self.blanksamples()
        
    def runsamples_uncty(self):

        """Runs samples under the uncertainty distribution"""

        if self.covobj is None:
            return

        # If we are keeping the xy samples, set up the samples arrays
        # here.
        if self.keepxysamples:
            self.setupsamples_xy()
        
        for isample in range(self.neval):
            xysample = self.covobj.getsamples() + self.xy
            self.setdata(xysample)
            self.applytransf(isample)

            if self.keepxysamples:
                self.samples_x[isample] = xysample[:,0]
                self.samples_y[isample] = xysample[:,1]
            
    def runsamples_pars(self):

        """Runs the parameter samples"""
        
        self.setdata()
        for iset in range(self.neval):
            self.updatetransf(iset)
            self.applytransf(iset)

            # map the lnprob across (could port into an applylnprob)
            if self.lnprob.size > 0:
                self.samples_z[iset] = self.lnprob[None, iset]
            
    def setdata(self, xy=np.array([]) ):

        """Passes the x, y data and any covariances to the transformation
object"""

        if np.size(xy) < 1:
            xy = self.xy
        
        self.transf.updatedata(xy=xy, covxy=self.covxy)

    def updatetransf(self, itransf=0):

        """Applies the i'th transformation to the xy points"""

        # Safety valve
        if itransf >= self.samples_xi.size:
            return

        # update the transformation parameters
        modelpars = self.parsamples[itransf, self.lmodel]
        self.transf.updatetransf(modelpars)

    def applytransf(self, itransf=0):

        """Applies the current transformation, slotting the results into the
itransf'th sample set"""
        
        self.transf.tranpos()
        self.samples_xi[itransf] = self.transf.xtran
        self.samples_eta[itransf] = self.transf.ytran
        
        # If we have covariances, apply those too (Currently has no
        # destination)
        if np.size(self.transf.covxy) > 0:
            self.transf.trancov()

    def samples_stats(self):

        """Computes moments of the xi, eta samples"""

        self.med_xieta, self.cov_xieta, \
            self.skew_xieta, self.kurt_xieta = \
                samples_moments(self.samples_xi, self.samples_eta)

        
## Generally useful utilities follow
            
def samples_moments(samples=np.array([]), \
                    samples_y=np.array([]), \
                    methcent=np.median, \
                    Verbose=True):

        """Computes moments of input samples.

Inputs:

        samples = [nsamples, 2, ndata] set of xy samples, OR [nsamples, ndata] array of x values only

        samples_y = [nsamples, ndata] array of y samples. Optional unless samples_xy is two dimensional

        methcent = method used to compute the centroid. Defaults to median

        Verbose = print screen output if error

Returns:

        med = centroid of the data. Default median

        cov = CovarsNx2x2 object describing the covariance

        skew = [N, 2] - element describing skew of marginal x, y

        kurt = [N, 2] - element describing kurtosis of marginal x, y

"""

        # to avoid typos
        blank = np.array([])
        
        # Just a bit of parsing on the dimensionality of the input
        if np.size(samples) < 1:
            return blank, None, blank, blank

        # allow passing in of separate x, y samples
        ndim = np.ndim(samples)
        if ndim == 3:
            samples_xieta = samples
        else:
            try:
                samples_xieta = np.stack((samples, samples_y), axis=1)
            except:
                if Verbose:
                    print("apply2d.samples_moments WARN - samples problem")
                    print("samples, samples_y shapes:" \
                          , np.shape(samples), np.shape(samples_y))
                return blank, None, blank, blank

        # Now compute the moments
        med = methcent(samples_xieta, axis=0).T
        cov = CovarsNx2x2(xysamples=samples_xieta)
        cov.eigensFromCovars()
        
        # one-dimensional skew and kurtosis for each axis
        skew = stats.skew(samples_xieta, axis=0).T
        kurt = stats.kurtosis(samples_xieta, axis=0).T

        return med, cov, skew, kurt
        
# Utilities to import the samples and parameters follow. These are set
# outside a class in order to be accessible from anywhere.

def blanktransf(transfname='Poly', polyname='Chebyshev', Verbose=True):

    """Returns blank transformation object"""

    # must be supported
    if not transf_supported(transfname, polyname):
        if Verbose:
            print("apply2d.blanktransf WARN - transfname %s and/or polyname %s not both supported." % (transfname, polyname))
        return None

    # Generate the blank transformation object
    objtransf = getattr(unctytwod, transfname)
    transfblank = objtransf(kindpoly=polyname)

    return transfblank
    
def loadtransf(pathpars='', pathobs='', pathtarg=''):

    """Loads transformation objects from its parts on disk, where:

    pathpars = path to (ascii) parameters file

    pathobs = path to (ascii) source data file

    pathtarg = path to (ascii) target data file (optional).

Returns:

    transf = transformation object, with parameters, data, and methods
    populated.

    """

    pset, obset, obstarg = loadparts(pathpars, pathobs, pathtarg)
    transf = buildtransf(pset, obset, obstarg)

    return transf
    
def loadparts(pathpars='', pathobs='', pathtarg=''):

    """Loads transformation parameters, and, optionally, data.

Inputs:

    pathpars = path to saved parset parameters.

    pathobs = path to "source" frame observation data

    pathtarg = path to "target" frame observation data

Returns: 

    pset, obset, obstarg, where:

    pset = parameter-set object for the transformation

    obset = data in the source frame

    obstarg = data in the target frame 

"""

    if len(pathpars) < 3:
        print("apply2d.loadtransf WARN - input path too short")
        return

    pset = parset2d.loadparset(pathpars)
    
    # Now load the source data
    obset = obset2d.Obset()
    obset.readobs(pathobs, strictlimits=False)

    # Load the target data if given. If not, return blank obset object
    obstarg = obset2d.Obset()
    if len(pathtarg) > 3:
        obstarg.readobs(pathtarg)
    
    return pset, obset, obstarg

def buildtransf(pset=None, obset=None, obstarg=None):

    """Builds transformation object from pset and obset

Inputs:

    pset = parameter-set object

    obset = observation-set object

    obstarg = observations in the target frame

Returns:

    transf = transformation object

"""

    # Parse input
    if pset is None or obset is None:
        return None

    # Ensure the input transformation is supported
    if not transfnamesok(pset):
        if Verbose:
            print("buildtransf WARN - problem with transformation names")
        return None
    
    # Implementation note: the data insertion will later be updated
    # once all the transformations can accept a data-update method
    # later. For the moment, lift them out
    objtransf = getattr(unctytwod, pset.transfname)

    # Ugh - still on this
    xsrc = obset.xy[:,0]
    ysrc = obset.xy[:,1]
    covsrc = obset.covxy

    # Since at least one of the transformations expects target
    # coordinates as well, we need at least placeholders for those.
    radec=np.array([])
    covradec=np.array([])
    
    if obstarg is not None:
        if obstarg.xy.size > 0:
            radec = obstarg.xy

        if obstarg.covxy.size > 0:
            covradec = obstarg.covxy
    
    transf = objtransf(xsrc, ysrc, covsrc, \
                       pset.model, checkparsy=True, \
                       kindpoly=pset.polyname, \
                       xmin=pset.xmin, \
                       xmax=pset.xmax, \
                       ymin=pset.ymin, \
                       ymax=pset.ymax, \
                       radec=radec, \
                       covradec=covradec)

    return transf
    
def transfnamesok(pset=None, Verbose=True):

    """Parses transformation names from parameter set object to ensure they are supported by the methods that will use them. 

Inputs:

    pset = parameter set object"

    Verbose = print screen output while parsing

Returns:

    namesok = True if the transformation name is OK, otherwise False.

"""

    # Implementation note: we do this here rather than in parset2d.py
    # because the latter doesn't know about the transf object or
    # uncertaintytwod. This keeps the import chain less tangled.
    
    if pset is None:
        return False

    # Get the transfname and polyname if present
    try:
        transfname = pset.transfname
    except:
        transfname = ''

    try:
        polyname = pset.polyname
    except:
        polyname = ''
        
    return transf_supported(transfname, polyname)

def transf_supported(transfname='', polyname='', \
                     reqpoly = ['Poly', 'xy2equ', 'TangentPlane'], \
                     Verbose=True):

    """Parses transformation and poly names to ensure they are supported.

Inputs:

    transfname = string giving transformation name. Corresponds to
    class name in unctytwod.py.

    polyname = name of numpy polynomial class.

    reqpoly = list of unctytwod.py transformations that actually
    require the polyname to be set.

    Verbose = print screen output

Returns:

    namesok = True if transformation and any needed polyname is supported.

    """
        
    if len(transfname) < 1:
        if Verbose:
            print("apply2d.parsetransf INFO - transfname is blank")
        return False

    if not hasattr(unctytwod, transfname):
        if Verbose:
            print("apply2d.parsetransf INFO - transfname not found: %s" \
                  % (transfname))

        return False

    # If the parse reached here then the transfname is OK. If we don't
    # care about the polynomial name, we can return True here.
    if not transfname in reqpoly:
        return True

    # Parse the polynomial name
    if len(polyname) < 2:
        if Verbose:
            print("apply2d.parsetransf INFO - needed polyname not found")
        return False

    polys_allowed = unctytwod.Poly().polysallowed[:]
    
    if not polyname in polys_allowed:
        if Verbose:
            print("apply2d.parsetransf INFO - polyname %s not in supported polynomials" % (polyname))
            print(polys_allowed)
        return False

    
    # If we got here, then all should be OK.
    return True

def loadobset(pathobs='', Verbose=True):

    """Loads an observation set from disk.

Inputs:

    pathobs = path to text file containing coordinates

    Verbose = print screen output

Returns:

    obset = Obset object. Blank if file not found.

"""

    obset = obset2d.Obset()
    try:
        obset.readobs(pathobs, strictlimits=False)
    except:
        if Verbose:
            print("apply2d.loadobset WARN - problem loading obset path: %s" \
                  % (pathobs))

    return obset

def paths_ok(paths=[]):

    """Returns True if all the paths in a supplied list are readable. Returns False if any of the paths are unreadable, OR if the supplied list has zero length.

    INPUTS

    paths = list of paths to check"""

    if len(paths) < 1:
        return False
    
    for path in paths:
        if not os.access(path, os.R_OK):
            return False

    # If we got here then all the paths were readable.
    return True
        

    
    
####### More involved utilities follow

def loadparsamples(pathpset='', pathflat=''):

    """Loads flat samples from disk, including their paramset, populates a
transformation object ready for evaluation. More involved examination
of an mcmc2d generation run is better done with examine2d.py
methods. The present method aims to support a more streamlined
approach.

Inputs:

    pathpset = path to the file with the paramset (including limits
    and options)

    pathflat = path to flat samples

Returns:

    transf = transformation object with parameters and limits
    populated

    flatsamples = array of flat samples

"""

    # Load the parameter set and parse its transformation names
    pset = parset2d.loadparset(pathpset)
    print(pathpset, pset.transfname)
    if not transfnamesok(pset):
        return None, np.array([]), np.array([])

    # If we're dealing with a pointing and the tangent point is not
    # within the limits, treat the limits as differential. Currently
    # this is a bit hardwired to the one case:
    if pset.transfname.find('Equ2tan') > -1:
        x0 = pset.model[0]
        y0 = pset.model[1]
        if x0 < pset.xmin or x0 > pset.xmax or \
           y0 < pset.ymin or y0 > pset.ymax:
            print("loadparsamples INFO - treating the xy limits as relative")
            print("loadparsamples INFO - xlims %.2f, %.2f <> %.2f" \
                  % (pset.xmin, pset.xmax, x0))
            print("loadparsamples INFO - ylims %.2f, %.2f <> %.2f" \
                  % (pset.ymin, pset.ymax, y0))

            pset.xmin = x0 + pset.xmin
            pset.xmax = x0 + pset.xmax
            pset.ymin = y0 + pset.ymin
            pset.ymax = y0 + pset.ymax
            
    # now create the transformation object
    transf = blanktransf(pset.transfname, pset.polyname)
    transf.updatelimits(pset.xmin, pset.xmax, pset.ymin, pset.ymax)
    transf.updatetransf(pset.model)

    try:
        flatsamples = np.load(pathflat)
    except:
        flatsamples = np.array([])
        
    return transf, flatsamples, pset

##### test routines follow

def sample_transf(neval=250, \
                  pathpset='test_parset_guess_poly_deg2_n100.txt', \
                  pathflat='test_flat_fitPoly_100_order2fit2_noprior_run1.npy',\
                  pathobs='test_obs_src.dat', \
                  pathtarg='test_obs_targ.dat', \
                  pathlnprobs='', scmap='rainbow_r', \
                  alpha=0.5):

    """Maps (selected) observed points onto the target frame, using
samples from the MCMC as transformation parameters

    """

    # Need to think about this a little more, because observational
    # uncertainty can also be large... We probably want to compare to
    # the transformed src observations under the truth transformation,
    # rather than the observed transformations (since the latter has
    # also been perturbed by measurement uncertainty)

    lpaths = [pathpset, pathflat, pathobs]
    if not paths_ok(lpaths):
        print("sample_transf WARN - not all paths readable:", lpaths)
        return

    # evaluation set
    ES = Evalset(pathpset=pathpset, \
                 pathflat=pathflat, pathobs=pathobs, \
                 pathlnprobs=pathlnprobs, neval=neval)
    ES.getobs()
    ES.getsamples()
    ES.getlnprobs()

    # Pass the observed data to the object (this might need one more
    # instance-level method):
    ES.xy = np.copy(ES.obset.xy)

    # Apply the truth transformation to the source positions
    ES.setdata()
    ES.transf.tranpos()
    truth_xi = np.copy(ES.transf.xtran)
    truth_eta = np.copy(ES.transf.ytran)
    
    ES.setupsamples_xieta()
    ES.setupsamples_z()

    print("sample_transf INFO - running samples...")
    ES.runsamples_pars()
    print("sample_transf INFO - ... done.")

    ## Load some "observed" objects to see how predicted and actual
    ## positions compare...
    obstar = None
    if os.access(pathtarg, os.R_OK):
        obstar = obset2d.Obset()
        obstar.readobs(pathtarg, strictlimits=False)

    #print("OBSTARG:", obstar.xy.shape)
    
    # Now we pick a few objects
    lsho = np.arange(4)
    
    # Find a background object
    if np.size(ES.obset.isfg) > 0:
        gbg = np.where(ES.obset.isfg < 1)[0]
        lsho[2] = gbg[0]

    # minmax for uniform color range for all plots
    vmin = np.min(ES.samples_z)
    vmax = np.max(ES.samples_z)
        
    figscatt = plt.figure(3, figsize=(10,10))
    lax = []
    scatts=[]
    for isho in range(len(lsho)):
        ax = figscatt.add_subplot(2,2,isho+1)
        jobj = lsho[isho]
        thisx = ES.samples_xi[:,jobj]
        thisy = ES.samples_eta[:,jobj]
        thisz = ES.samples_z[:,jobj]

        dumscatt = ax.scatter(thisx, thisy, c=thisz, vmin=vmin, vmax=vmax, \
                              cmap=scmap, s=9, alpha=alpha, \
                              label='Samples', zorder=15)

        # Show the target object
        truthx = truth_xi[jobj]
        truthy = truth_eta[jobj]
        dumtruth = ax.scatter(truthx, truthy, marker='x', color='k', \
                              zorder=25, s=81, label='Transformed "truth"')

        if hasattr(obstar, 'xy'):
            targx = obstar.xy[jobj,0]
            targy = obstar.xy[jobj,1]

            dumtar = ax.scatter(targx, targy, marker='+', color='k', \
                              zorder=25, \
                                s=64, label='Observed')

        
        # Show the isfg
        ax.set_title('is foreground: %i' % (ES.obset.isfg[jobj]))

        leg = ax.legend()
        
        # axis handles so we can access them later
        lax.append(ax)
        scatts.append(dumscatt)

    cbar = figscatt.colorbar(scatts[-1], ax=lax[-1])
    cbar.solids.set(alpha=1.0)

    print(lsho, ES.obset.isfg[lsho])

    for i in range(2,4):
        lax[i].set_xlabel(r'$\xi$')

    lax[0].set_ylabel(r'$\eta$')
    lax[2].set_ylabel(r'$\eta$')

    figscatt.subplots_adjust(hspace=0.3, wspace=0.3)
    
def eval_uncty(neval=250, \
               pathpset='test_parset_guess_poly_deg2_n100.txt', \
               pathflat='test_flat_fitPoly_100_order2fit2_noprior_run1.npy', \
               pathobs='test_obs_src.dat', \
               plotmajors=True, sqrtplot=True, nbins=10):

    """Evaluates both the pointing and propagated uncertainty based on
MCMC trial output.

Inputs:

    neval = number of evaluations to draw

    pathpset = parameter guess set (to interpret the flat samples)

    pathflat = path to MCMC flattened samples

    pathobs = path to observation file with positions and covariances
    
    plotmajors = plot debug plot of output vs input major axes
    
    sqrtplot = plot sqrt(major axes)

    nbins = number of magnitude bins to plot (10 seems to be a
    sensible default for most cases)

    """

    # Check that all the paths are present
    lpaths = [pathpset, pathflat, pathobs]
    if not paths_ok(lpaths):
        print("apply2d.eval_uncty WARN - one or more path unreadable:", \
              lpaths)
        return
    
    # Samples from the flattened MCMC:
    ES = Evalset(pathpset=pathpset, \
                 pathflat=pathflat, \
                 pathobs=pathobs, \
                 neval=neval)

    # Populate the source-frame coordinates and covariances
    ES.getsamples()
    ES.getobs()
    ES.covfromobs()
    ES.setdata()
    ES.checkneval()
    ES.covobj.eigensFromCovars()
    
    # now draw transformation uncertainty samples and do statistics on
    # them
    t0=time.time()
    print("sample_uncty INFO - starting %.2e flat samples..." \
          % (ES.neval))
    ES.setupsamples_xieta()

    ES.runsamples_pars()
    ES.samples_stats()
    print("sample_uncty INFO - ... done in %.2e seconds" \
          % (time.time()-t0))
    
    # Copy into a new object to run the samples in the data
    # uncertainty
    t0 = time.time()
    print("sample_uncty INFO - starting %.2e MC samples..." % (ES.neval))
    MS = copy.copy(ES)
    MS.setupsamples_xieta()
    MS.runsamples_uncty()
    MS.samples_stats()
    print("sample_uncty INFO - ... done in %.2e seconds" \
          % (time.time()-t0))

    # try a plot of output covariance against input covariance
    if not plotmajors:
        return

    # length of evaluations
    print(ES.covobj.majors.shape, \
          ES.cov_xieta.majors.shape, \
          MS.cov_xieta.majors.shape)

    # magnitude vector
    xvec = ES.covobj.majors
    try:
        xvec = ES.obset.mags
        labx = r'mag'
    except:
        magprob = True
        labx = r'Major axis (X,Y)'

    # What are we plotting
    ytran = np.copy(ES.cov_xieta.majors)
    ymeas = np.copy(MS.cov_xieta.majors)
        
    # out of curiosity, sqrt the majors
    #ES.cov_xieta.majors = np.sqrt(ES.cov_xieta.majors)
    #MS.cov_xieta.majors = np.sqrt(MS.cov_xieta.majors)

    labely=r'Major axis ($\xi,\eta$)'
    if sqrtplot:
        ytran = np.sqrt(ytran)
        ymeas = np.sqrt(ymeas)
        labely=r'$\sqrt{Major axis (\xi,\eta)}$'
        
        
    # Compute binned trends for plotting:
    bstran = Binstats(xvec, np.atleast_2d(ytran).T, nbins=nbins)
    bsmeas = Binstats(xvec, np.atleast_2d(ymeas).T, nbins=nbins)
        
    fig7 = plt.figure(7)
    fig7.clf()
    ax71 = fig7.add_subplot(111)

    dum711 = ax71.scatter(xvec, \
                          ytran, s=9, \
                          cmap='Blues', \
                          label=r'transformation')
    dum712 = ax71.scatter(xvec, \
                          ymeas, s=16, \
                          marker='s',\
                          cmap='Reds', \
                          label=r'measurement')

    trendtran = ax71.step(bstran.medns, bstran.meansxy, c='b', \
                          where='mid', lw=0.5, alpha=0.5)

    trendmeas = ax71.step(bsmeas.medns, bsmeas.meansxy, c='r', \
                          where='mid', lw=2, alpha=0.5)

    
    # Add bounds?
    stdtran = np.sqrt(bstran.covsxy.squeeze())
    stdmeas = np.sqrt(bstran.covsxy.squeeze())

    #dummeas = ax71.step(bsmeas.medns, \
    #                    bsmeas.meansxy.squeeze() + stdmeas, \
    #                    c='r', \
    #                    where='mid', lw=0.75, alpha=0.5, ls='--')

    #dummeas = ax71.step(bsmeas.medns, \
    #                    bsmeas.meansxy.squeeze() - stdmeas, \
    #                    c='r', \
    #                    where='mid', lw=0.75, alpha=0.5, ls='--')

    
    # print("DBG:", bsmeas.meansxy.shape, bstran.covsxy.shape)
    
    ax71.set_xlabel(labx)
    ax71.set_ylabel(labely)
    ax71.set_yscale('log')
    leg7 = ax71.legend()
    
def unctysamples(nsamples=10, \
                 pathpset='test_parset_guess_poly_deg2_n100.txt', \
                 pathobs='test_obs_src.dat', \
                 plotsamples=True, \
                 unifcovs=False, \
                 unif_major=1.0e-4, unif_minor=0.4e-4, unif_posan=0., \
                 maxplot=-1, \
                 genpos=False, \
                 nx=7, ny=7, \
                 showshifts=True, \
                 deg=1, \
                 returnmedian=False, \
                 parsswap = np.array([]), \
                 nomode=True, \
                 iskew=-1):

    """Performs monte carlo sampling of the source uncertainty,
propagated through to the target frame.

Inputs:

    nsamples = how many samples we want to draw

    pathpset = path to parameter set to be used for the transformation

    pathobs = path to observations including uncertainty covariances

    plotsamples = do a scatter plot of the samples

    unifcovs = assign uniform covariances (useful for testing)

    unif_major, unif_minor, unif_posan = uniform covariance scalars

    maxplot = maximum number of samples to show in the scatterplots

    genpos = generate positions using limits in transformation

    nx, ny = number of positions in x, y to produce on regular grid

    showshifts = show quiver plot with offsets

    deg = polynomial degree for residual approximation

    returnmedian = return median delta? (Useful if wrapping)

    parsswap = parameters to substitute for the transformation
    parameters (useful if calling in a loop)

    nomode = don't compute the sample mode (which is slow)

    iskew = if > -1, plot the eta distribution for this
    object. Otherwise find the one with maximum (Eta) skew.

    """

    lpaths = [pathpset, pathobs]
    if not paths_ok(lpaths):
        print("apply2d.unctysamples WARN - one or more unreadable:", \
              lpaths)
        return
    
    US = Evalset(pathpset=pathpset, neval=nsamples, pathobs=pathobs)
    US.getsamples()
    US.getobs()
    US.covfromobs()

    print("unctysamples INFO - transformation name:", US.pset.transfname)
    
    # Replace transformation parameters with input parameters? (Trust
    # the user to get the parameters right)
    if np.size(parsswap) > 0:
        US.transf.updatetransf(parsswap)
    
    # Replace or generate uniform random positions over transf limits
    if genpos or len(US.pathobs) < 3:
        # US.genposran(50)
        US.genposgrid(nx, ny)
        
    # Replace covariances with uniform covariances?
    if unifcovs or US.xy.shape[0] != US.covxy.shape[0]:
        print("unctysamples INFO - generating uniform covariances")
        US.gencovunif(major=unif_major, \
                      minor=unif_minor, \
                      rotdeg=unif_posan)
    
    # Pass the samples to the transformation and compute the
    # propagated covariance
    US.setdata()
    US.propagate_covar()

    # For computing the bias in midpoints later
    US.med_input_xieta = US.transf.xytran
    
    # Now set up the samples and run the parametric monte carlo
    print("unctysamples INFO - starting %.2e MC samples..." % (nsamples))
    t0 = time.time()
    US.keepxysamples = True
    US.setupsamples_xieta()
    US.runsamples_uncty()
    print("... done in %.2e seconds" % (time.time() - t0))

    # Perform statistics on the samples
    US.samples_stats()

    # Use our new moments calculator
    print("Computing moments...")
    t99 = time.time()
    moments = moments2d.Moments2d(US.samples_xi, US.samples_eta, \
                                  nomode=nomode)
    print("")
    print("... done in %.2e seconds." % (time.time()-t99))

    print(moments.mean.shape)
    print(moments.skew.shape)
    print(moments.covars.shape)
    print(moments.mode.shape)
    
    
    # Find the median of the transformed minus the transformed median
    xieta_means, _, _, _ = samples_moments(US.samples_xi, \
                                           US.samples_eta, \
                                           methcent=np.mean)

    # The difference in medians is a (much) larger signal...
    #dxieta_modes = US.med_xieta - xieta_means
    ### dxieta_modes = xieta_means - US.med_input_xieta
    dxieta_modes = US.med_xieta - US.med_input_xieta    
    dxyeta_asymm = US.med_xieta - xieta_means
    
    if np.size(moments.mode) > 0:
        print("unctysamples INFO - using the median-to-mode as offset")
        #dxieta_modes = moments.median - moments.mode
        dxieta_modes = moments.mode - US.med_input_xieta
        
    # scale delta alpha -> delta alpha* 
    dalpha = dxieta_modes[:,0] * np.cos(np.radians(US.med_xieta[:,1]))
        
    # Try subtracting off the median delta (consider making this an
    # input argument)
    #
    # 
    # dxieta_modes -= np.median(dxieta_modes, axis=0)[None,:]

    # Subtract off the median coordinate so that the fit parameters
    # will be easier to interpret
    coocen = np.zeros(2)
    # coocen = np.median(US.med_input_xieta, axis=0)

    #if US.pset.transfname.find('Tan2equ') > -1:
    #    print("unctysamples INFO - assigning TP as median coord")
    #    coocen = np.copy(US.transf.pars)
    
    US.med_input_xieta -= coocen[None,:]
    US.med_xieta -= coocen[None,:]
    
    if showshifts:

        # magnitudes of the shifts
        mags = np.sqrt(dxieta_modes[:,0]**2 + dxieta_modes[:,1]**2)
        
        fig6 = plt.figure(6)
        fig6.clf()
        ax6 = fig6.add_subplot(111)
        dum6 = ax6.quiver(US.med_input_xieta[:,0], \
                          US.med_input_xieta[:,1], \
                          #dxieta_modes[:,0], \
                          dalpha, \
                          dxieta_modes[:,1], \
                          mags)
        cbar6 = fig6.colorbar(dum6, ax=ax6)
        ax6.set_title(r'Centroid offsets ($\Delta \alpha^{\ast}, \Delta \delta$)')

        # fit these (not strictly correct because euclidean distance
        # isn't the right metric)
        xytarg = US.med_input_xieta + dxieta_modes
        xytarg[:,0] = US.med_input_xieta[:,0] + dalpha

        #xytarg = US.med_xieta

        print("Offset INFO - median offset:", \
              np.median(dxieta_modes, axis=0))
        
        lsq = Leastsq2d(US.med_input_xieta[:,0], \
                        US.med_input_xieta[:,1], \
                        deg=deg, kind='Poly', \
                        xytarg=xytarg, \
                        norescale=True)

        # translate parameters into 6-terms. Note that we need to know
        # which is which...
        geom = sixterm2d.getpars(lsq.pars[0:6])
        
        # evaluate residuals
        xyeval = lsq.ev()
        dxyeval = xyeval - US.med_input_xieta  # Model of offsets
        dxyresid = xyeval - xytarg
        
        fig8 = plt.figure(8)
        fig8.clf()
        ax8=fig8.add_subplot(222)
        ax82=fig8.add_subplot(224)
        ax83=fig8.add_subplot(223)

        dum8 = ax8.quiver(US.med_input_xieta[:,0], \
                          US.med_input_xieta[:,1], \
                          dxyeval[:,0], dxyeval[:,1], \
                          np.sqrt(dxyeval[:,0]**2 + dxyeval[:,1]**2) )
        cbar8 = fig8.colorbar(dum8, ax=ax8)
        ax8.set_title(r'Fit to centroid offsets, degree %i' % (deg))

        dum82 = ax82.quiver(US.med_input_xieta[:,0], \
                            US.med_input_xieta[:,1], \
                            dxyresid[:,0], dxyresid[:,1], \
                            np.sqrt(dxyresid[:,0]**2 + dxyresid[:,1]**2) )
        cbar82 = fig8.colorbar(dum82, ax=ax82)
        ax82.set_title(r'Fit residuals, degree %i' % (deg))

        # median delta subtracted
        medev = np.median(dxyeval, axis=0)
        dxyone = dxyeval - medev[None,:]
        dum83 = ax83.quiver(US.med_input_xieta[:,0], \
                            US.med_input_xieta[:,1], \
                            dxyone[:,0], dxyone[:,1], \
                            np.sqrt(dxyone[:,0]**2 + dxyone[:,1]**2) )
        cbar83 = fig8.colorbar(dum83, ax=ax83)
        ax83.set_title(r'Fit minus median offset')
        
        print("lsq pars:", lsq.pars)
        print("lsq geom:", geom)
        #return
        
        # OK I have to see what this is doing...
        fig7 = plt.figure(7)
        fig7.clf()
        ax71 = fig7.add_subplot(121)
        ax72 = fig7.add_subplot(122)
        axes = [ax71, ax72]
        sdel = [r'$\Delta \xi$', r'$\Delta \eta$']
        for iset in range(len(axes)):
            ss = dxieta_modes[:,iset]
            dum = axes[iset].hist(ss, \
                                  bins=30, alpha=0.5, \
                                  density=True)
            kde = stats.gaussian_kde(ss)
            xfine = np.linspace(np.min(ss), np.max(ss), 1000, \
                                endpoint=True)
            dumf = axes[iset].plot(xfine, kde.pdf(xfine))
            axes[iset].set_xlabel(sdel[iset])
        fig7.suptitle(r'Peak-to-median offsets')

        # Return for the moment to see output
        # return
        
    # Perform statistics on the *generated* samples, just to ensure
    # that part works...
    if unifcovs:
        print("### Generated samples, source frame: ####")

        # What was sent to the generator...
        US.covobj.eigensFromCovars()
        print("Generated parameters:")
        print("unifcovs DBG - majors:", US.covobj.majors[0:4])
        print("unifcovs DBG - minors:", US.covobj.minors[0:4])
        print("unifcovs DBG - posans (deg):", US.covobj.rotDegs[0:4])

        # ... and what was produced
        med, cov, skew, kurt = samples_moments(US.samples_x, \
                                               US.samples_y, \
                                               methcent=np.mean)
    
        print("Covariance of generated data, source frame:")
        print("Generated majors:", cov.majors[0:4])
        print("Generated minors:", cov.minors[0:4])
        print("Generated posans (deg):", cov.rotDegs[0:4])

        # useful to display the check on the stddevs
        print("Checking against stddev squared:")
        print("var(x):", cov.stdx[0:4]**2)
        print("var(y):", cov.stdy[0:4]**2)
        print("np.std(x, ddof=1)**2:", \
              np.std(US.samples_x,axis=0, ddof=1.)[0:4]**2)
        print("np.std(y, ddof=1)**2:", \
              np.std(US.samples_y,axis=0, ddof=1.)[0:4]**2)


        # If we generated all the datapoints from the same
        # distribution, take a look at the skew distributions
        fig4 = plt.figure(4)
        fig4.clf()
        ax41 = fig4.add_subplot(211)
        ax42 = fig4.add_subplot(212)

        ss = np.vstack(( skew, US.skew_xieta))
        maxval = np.max(np.abs(ss), axis=0)
        
        nbins = 25
        #binrange = (-maxval, maxval)

        axes = [ax41, ax42]
        labs_src = [r'$X$', r'$Y$']
        labs_tar = [r'$\xi$', r'$\eta$']
        
        for dim in range(len(axes)):
            ax = axes[dim]
            rang = [-maxval[dim], maxval[dim]]
            n, bins, patches = ax.hist(skew[:,dim], bins=nbins, \
                                       range=rang, \
                                       histtype='step', zorder=6, \
                                       label='source %s' % (labs_src[dim]))
        
            n, bins, patches = ax.hist(US.skew_xieta[:,dim], bins=nbins, \
                                       range=rang, \
                                       histtype='stepfilled', \
                                       zorder=5, \
                                       label='transf %s' % (labs_tar[dim]), \
                                       alpha=0.5)
            leg = ax.legend()

            ax.set_ylabel('N(skew)')
            ax.set_xlabel('skew (%s or %s)' % (labs_src[dim], labs_tar[dim]))
            
        nsamples, ndata = US.samples_x.shape
        fig4.suptitle('Marginal skew (%i objects at %i samples each)' % \
                      (ndata, nsamples))
        fig4.subplots_adjust(hspace=0.3, wspace=0.3)
        
        print("#####")


    print("===== Transformed covariances =====")
    print("Covariances of propagated positions:")
    print("majors computed:", US.cov_xieta.majors[0:4])
    print("minors computed:", US.cov_xieta.minors[0:4])
    print("posans computed:", US.cov_xieta.rotDegs[0:4])

    # cross-check
    print("Propagated covariances of input positions")
    print("majors propag:", US.cov_propagated.majors[0:4])
    print("minors propag:", US.cov_propagated.minors[0:4])
    print("posans propag:", US.cov_propagated.rotDegs[0:4])

    
    # Scatter plot of the samples
    if not plotsamples:
        return

    #### Histogram of one or two "representative" points
    skeweta = US.skew_xieta[:,0]

    if iskew < 0:
        iskew = np.argmax(np.abs(US.skew_xieta[:,1]))
    fig5 = plt.figure(5)
    fig5.clf()
    ax51 = fig5.add_subplot(111)
    dum51 = ax51.hist(US.samples_eta[:,iskew], bins=100, alpha=0.5, \
                      label='Transformed samples', density=True)
    ax51.set_xlabel(r'$\eta$')
    ax51.set_ylabel(r'$N(\eta)$')

    dumv1 = ax51.axvline(US.med_xieta[iskew,1]+coocen[1], \
                         zorder=25, color='k', \
                         label='median of transformed')
    dumv2 = ax51.axvline(US.med_input_xieta[iskew,1]+coocen[1], \
                         zorder=25, \
                         color='m', linestyle='--', \
                         label='transformed input peak')
    dumv3 = ax51.axvline(xieta_means[iskew,1], zorder=25, \
                         color='r', linestyle='--', lw=2, \
                         label='mean of transformed')

    # Try getting the mode (thought: in two dimensions, best to use
    # two marginals so that the grid for the fine samples doesn't get
    # huge)
    distr = stats.gaussian_kde(US.samples_eta[:,iskew])
    xfine = np.linspace(np.min(US.samples_eta[:,iskew]), \
                       np.max(US.samples_eta[:,iskew]), \
                       5000)
    imax = np.argmax(distr.pdf(xfine))
    dumfine = ax51.plot(xfine, distr.pdf(xfine), zorder=50, \
                        color='g', lw=1, label='KDE')
    #dumv4 = ax51.axvline(xfine[imax], zorder=25, color='y', lw=2, \
    #                     label='')
    try:
        dumv4 = ax51.axvline(moments.mode[iskew, 1], color='g', lw=2, \
                             zorder=30, label='Mode of transformed')
    except:
        # mode not computed
        dummy = 4
        
    ax51.set_title(r'Skew ($\eta$): %.2f' % (US.skew_xieta[iskew, 1]) )
    fig5.suptitle('iskew: %i' % (iskew))
    
    leg51 = ax51.legend()
    
    #### Scatterplots of the monte carlo follow.

    # Allow plotting of a subset if there are many samples
    nsamples, ndata = US.samples_x.shape
    if maxplot < 1 or maxplot > nsamples:
        maxplot = nsamples 
    rng = np.random.default_rng()
    lsho = rng.choice(nsamples, maxplot, replace=False)

    # Assign an ID to each row
    lid = np.arange(ndata)
    lid = US.skew_xieta[:,0] # show skewness in eta
    arrid = np.tile(lid, nsamples).reshape(nsamples, ndata)
    
    fig2 = plt.figure(2)
    fig2.clf()
    ax2 = fig2.add_subplot(111)
    dum2 = ax2.scatter(US.samples_xi[lsho,:], \
                       US.samples_eta[lsho,:], s=1, \
                       alpha=0.5, \
                       c=arrid[lsho,:])
    ax2.set_xlabel(r'$\xi$')
    ax2.set_ylabel(r'$\eta$')
    ax2.set_title('%i transformed samples from source-frame uncertainty' \
                  % (np.size(lsho)))

    cbar2=fig2.colorbar(dum2, ax=ax2)
    
    # Plot the original point clouds
    if np.size(US.samples_x) < 1:
        return
    fig3 = plt.figure(3)
    fig3.clf()
    ax3 = fig3.add_subplot(111)
    dum3 = ax3.scatter(US.samples_x[lsho,:], \
                       US.samples_y[lsho,:], s=1, \
                       alpha=0.5, \
                       c=arrid[lsho,:])
    ax3.set_xlabel(r'$X$')
    ax3.set_ylabel(r'$Y$')
    ax3.set_title('%i of %i samples from source-frame uncertainty' \
                  % (np.size(lsho), nsamples))

    cbar3 = fig3.colorbar(dum3, ax=ax3)

    # add a coordinate grid to the scatter plots
    GG = Evalset(pathpset=pathpset)
    GG.grid_nxfine=100
    GG.grid_nyfine=100
    GG.getsamples()

    # if we subtracted the pointing off the points, we need to do it
    # to the grid, too
    if np.size(parsswap) > 0:
        GG.transf.updatetransf(parsswap)
    
    GG.gengrid()
    GG.setdata()
    GG.transf.tranpos()

    for lineid in np.unique(GG.grid_whichline):
        bthisline = GG.grid_whichline == lineid

        dum3g = ax3.plot(GG.transf.x[bthisline], \
                         GG.transf.y[bthisline], \
                         color='0.5', alpha=0.5, zorder=1, \
                         lw=0.5)

        dum2g = ax2.plot(GG.transf.xtran[bthisline], \
                         GG.transf.ytran[bthisline], \
                         color='0.5', alpha=0.5, zorder=1, \
                         lw=0.5)

        # Ditto the quiver plot
        dum6g = ax6.plot(GG.transf.xtran[bthisline] - coocen[0], \
                         GG.transf.ytran[bthisline] - coocen[1], \
                         color='0.5', alpha=0.5, zorder=1, \
                         lw=0.5)

    # return the median delta?
    if returnmedian:
        return np.median(dxieta_modes, axis=0), geom
        
def traceplot(neval=10, \
              pathpset='test_parset_guess_poly_deg2_n100.txt', \
              pathflat='test_flat_fitPoly_100_order2fit2_noprior_run1.npy', \
              pathlnprobs='', scmap='RdBu'):

    """Evaluates the flat samples on a grid of coords"""

    # Comment: also shades by auxiliary (per-sample) quantity. Here
    # it's hardcoded to lnprobs, but we could imagine generalzing that
    # if needed. Samples_z is actually implemented in EvaluateSet, but
    # we drop back to ES.lnprob when actually plotting.
    
    if not paths_ok([pathpset, pathflat]):
        print("traceplot WARN - not all paths readable: ", \
              [pathpset, pathflat])
        return
    
    ES = Evalset(pathpset=pathpset, \
                 pathflat=pathflat, \
                 neval=neval, \
                 pathlnprobs=pathlnprobs)
    ES.getsamples()
    ES.getlnprobs()

    print(ES.pset.lmodel)
    print("traceplot DBG:", ES.parsamples.shape, ES.lnprob.shape)
    print(ES.pathflat)
    print(ES.pathlnprobs)
    
    # Set up evaluation coords
    ES.gengrid()

    print("DBG: grid line IDs:")
    print(np.size(ES.grid_whichline))
    print(np.unique(ES.grid_whichline))
    
    # Set up and run the evaluations
    ES.checkneval()
    ES.setupsamples_xieta()
    if ES.lnprob.size > 0:
        ES.setupsamples_z()

    print("samples_z DEBUG:", ES.samples_z.shape, ES.lnprob.shape)

    #Check to see if number of samples matches neval
    if neval > ES.samples_z.shape[0]:
        neval = ES.samples_z.shape[0]
        
    # run the samples
    ES.runsamples_pars()

    print(ES.samples_z.size)
    print(ES.samples_z.shape)
    #, ES.samples_z[:,-1])
    
    #print(ES.xy.shape)
    #print(ES.parsamples.shape)
    #print(ES.samples_xi.shape)

    #print(ES.samples_xi[:,0])

    # orphan plots under development
    fig1 = plt.figure(1)
    fig1.clf()
    ax1 = fig1.add_subplot(111)
    #dum = ax1.scatter(ES.xy[:,0], ES.xy[:,1], s=1)

    ax1.set_aspect('equal')
    ax1.set_xlabel(r'$\xi$')
    ax1.set_ylabel(r'$\eta$')
    fig1.suptitle('traceplot (%i samples)' % (neval))
    
    # Line collection coordinates and aux values
    linecoos = []
    zvals = []
        
    line_ids = np.unique(ES.grid_whichline)
    for lineid in line_ids:
        bthisline = ES.grid_whichline == lineid

        for isho in range(neval):
            xthis = ES.samples_xi[isho][bthisline]
            ythis = ES.samples_eta[isho][bthisline]

            # We plot the very first one in black
            if isho < 1:
                color = 'k'
                zorder=10
                alpha=0.5

                # Ensure we plot the "black" example
                dum = ax1.plot(xthis, ythis, \
                           color=color, zorder=zorder, \
                           alpha=alpha)
            else:
                linecoos.append(np.column_stack((xthis, ythis)))
                if np.size(ES.lnprob) > 0:
                    #color = cmap(zscal[isho])
                    zvals.append(ES.lnprob[isho])
                    color=None
                else:
                    color='b'
            
    # try the line collection
    linecoll = LineCollection(linecoos, array=np.array(zvals), \
                              cmap=scmap, alpha=0.02, zorder=1, \
                              color=color)
    ax1.add_collection(linecoll)

    if np.size(ES.lnprob) > 0:
        cbar = fig1.colorbar(linecoll, label="lnprob")
        cbar.solids.set(alpha=1.0)
    
    #for isho in range(neval):
        #dum2 = ax1.scatter(ES.samples_xi[isho], ES.samples_eta[isho], s=.5, \
        #                   alpha=0.05, color='b')



##### A few canned analyses follow

def mc_uncertainty_pointings(major=0.5, minor=0.5, posan=0., nsets=10, \
                             alpha0=220., \
                             pathpars='test_parset_tan2equ.txt', \
                             nsamples=100000, \
                             nx=15, ny=15, deg=1, \
                             delta0max=80., \
                             nomode=True, iskew=-1):

    """Does a loop of bias searches for fixed semimajor, minor axes but looping over central pointings. This is a wrapper around unctysamples."""

    # tangent points
    alpha0s = np.repeat(alpha0, nsets)
    delta0s = np.linspace(0., delta0max, nsets, endpoint=True)
    tps = np.stack((alpha0s, delta0s), axis=1)
        
    # results arrays
    medians = np.zeros(tps.shape)
    geoms = np.zeros((tps.shape[0], 6))
    
    # now do everything...
    for iset in range(medians.shape[0]):
        thismed, geom = unctysamples(nsamples, pathpset=pathpars, genpos=True, \
                                     unif_major=major, unif_minor=minor, \
                                     unif_posan=posan, unifcovs=True, \
                                     maxplot=100, nx=nx, ny=ny, deg=deg, \
                                     returnmedian=True, \
                                     parsswap=tps[iset], \
                                     nomode=nomode, \
                                     iskew=iskew)
        medians[iset] = thismed
        geoms[iset] = geom
        print(">>>>>>>> Done %i, %.2e, %.2e" \
              % (iset, tps[iset,0], tps[iset,1]), \
              thismed, "<<<<<<<<<<")
        
    fig10 = plt.figure(10)
    fig10.clf()
    ax101 = fig10.add_subplot(311)
    ax102 = fig10.add_subplot(312, sharex=ax101)
    ax103 = fig10.add_subplot(313, sharex=ax101)

    dum101 = ax101.scatter(tps[:,1], medians[:,0])
    dum102 = ax102.scatter(tps[:,1], medians[:,1])
    dum103 = ax103.scatter(tps[:,1], geoms[:,3])

    ax101.set_ylabel(r'$\Delta \alpha^{\ast}$')
    ax102.set_ylabel(r'$\Delta \delta$')
    ax103.set_ylabel(r'$s_y$')

    # Try a candidate fit to the vertical offsets...
    xfine = np.linspace(0., delta0max, 100, endpoint=True)
    cosdec = np.cos(np.radians(xfine))
    # dum = ax102.plot(xfine, 0.04*(0.98-1./cosdec**0.25), ls='--')
    # dum = ax102.plot(cosdec
    
    for ax in [ax101, ax102, ax103]:
        ax.set_xlabel(r'$\delta_0$')

    fig10.suptitle(r'$\alpha_0 = %.2f$, major=%.2f, minor=%.2f, $\theta$=%.1f' % (alpha0, major, minor, posan))
        
    # Return the statistics so that we can do things with them
    return tps, medians, geoms

def showgrids(pathpset='test_parset_tan2equ.txt', \
              nxfine=100, nyfine=100, \
              alpha0=220., delta0=85., \
              dalphadeg = 0.5, ddeltadeg=0., \
              xlen=2.0, ylen=2.0, \
              tan2equ=True):

    """For eventual paper: show tangent-plane coordinates for two pointings offset by different tangent points"""

    # TODO - feed the transformed grid from GR into GS as a dataset,
    # then transform back
    
    # Currently a canned analysis if going the other way. This could /
    # should be generalized.
    xlabel=r'$\alpha$'
    ylabel=r'$\delta$'
    if not tan2equ:
        pathpset='test_parset_eq2tan.txt'
        xlabel=r'$\xi$'
        ylabel=r'$\eta$'

    # Set up the grid object
    GR = Evalset(pathpset=pathpset)
    GR.grid_nxfine = nxfine
    GR.grid_nyfine = nyfine
    GR.getsamples()

    # Create our altered pointing object
    GS = copy.deepcopy(GR)
    
    # Swap in the model parameters
    tp0 = np.array([alpha0, delta0])
    tp1 = tp0 + np.array([dalphadeg, ddeltadeg])

    # This means we also have to update the limits
    GR.transf.updatelimits(tp0[0] - 0.5*xlen, tp0[0] + 0.5*xlen, \
                           tp0[1] - 0.5*ylen, tp0[1] + 0.5*ylen)
    GS.transf.updatelimits(tp0[0] - 0.5*xlen, tp0[0] + 0.5*xlen, \
                           tp0[1] - 0.5*ylen, tp0[1] + 0.5*ylen)
    
    GR.transf.updatetransf(tp0)
    GS.transf.updatetransf(tp1)

    
    print(GR.transf.pars)
    print(GS.transf.pars)
    
    # Create the grids
    grids = [GR, GS]
    for obj in grids:
        obj.gengrid()
        obj.setdata()
        obj.transf.tranpos()

    # Some debugging
    if not tan2equ:
        print("DBG:", GR.transf.pars)
        print("DBG:", np.max(GR.transf.x), np.min(GR.transf.x))
        print("DBG:", np.max(GR.transf.y), np.min(GR.transf.y))


    # Now create a figure and plot the two grids
    fig10 = plt.figure(10)
    fig10.clf()
    ax10 = fig10.add_subplot(111)

    colors = ['g', 'b']
    lss = ['-', '--']
    syms = ['*', 's']
    for lineid in np.unique(GR.grid_whichline):

        # go thru both grids
        for igrid in range(len(grids)):
            grid = grids[igrid]
            bthisline = grid.grid_whichline == lineid
            dum10g = ax10.plot(grid.transf.xtran[bthisline], \
                               grid.transf.ytran[bthisline], \
                               color=colors[igrid], \
                               ls=lss[igrid], lw=0.5)

            # Plot the tangent points
            if tan2equ:
                dum0 = ax10.plot(grid.transf.pars[0], \
                                 grid.transf.pars[1], \
                                 marker=syms[igrid], \
                                 color=colors[igrid], \
                                 ms=4)
            
    ax10.set_xlabel(xlabel)
    ax10.set_ylabel(ylabel)

    
def focal(alpha0=230., delta0=29., dalpha0=1., ddelta0=1., \
          xlen=40., ylen=40., nxfine=100, nyfine=100, \
          colors=['k','r'], lss=['-','--'], \
          degpoly=1, submedian=False):

    """Canned analysis - create a grid at one tangent point, then
'reobserve' from a second tangent point (in focal plane coordinates)
and compare.

Inputs:

    alpha0, delta0 = tangent point, degrees

    dalpha0, ddelta0 = offset in tangent point in degrees

    xlen, ylen = side-length in degrees in focal plane coordinates

    nxfine, nyfine = fine-grained number of points

    colors = [2] = matplotlib colors for generated and offset

    lss = [2] = matplotlib linestyles for generated and offset

    degpoly = fit degree for the polynomial

    submedian = subtract off the median delta before fitting"""

    # Create the transformation objects
    transf_gen = blanktransf('Tan2equ')
    transf_off = blanktransf('Equ2tan')

    transf_gen.pars = np.array([alpha0, delta0])
    transf_off.pars = np.array([alpha0 + dalpha0, \
                                delta0 + ddelta0])
    
    # limits
    transf_gen.xmin = 0.-0.5*xlen
    transf_gen.xmax = 0.+0.5*xlen
    transf_gen.ymin = 0.-0.5*ylen
    transf_gen.ymax = 0.+0.5*ylen

    # Now generate evaluation-set going from focal plane to sky
    Gen = Evalset('','','', transf=transf_gen)
    Gen.grid_nxfine = nxfine
    Gen.grid_nyfine = nyfine

    # Set up the focal plane grid
    Gen.gengrid()
    Gen.setdata()
    Gen.transf.tranpos()

    # OK that created a set of datapoints on the sky that can now be
    # translated back. So:
    Off = Evalset('','','', transf=transf_off)

    # Set data and limits
    Off.setdata(np.vstack((Gen.transf.xtran, Gen.transf.ytran)).T)
    Off.transf.updatelimits(np.min(Off.transf.x), \
                            np.max(Off.transf.x), \
                            np.min(Off.transf.y), \
                            np.max(Off.transf.y))

    # Copy the "whichgrid" information so that we know which line is
    # which in the plot
    Off.grid_whichline = np.copy(Gen.grid_whichline)
    
    # Now transform these positions back to the focal plane
    Off.transf.tranpos()

    # Now we examine the resulting transformation in the focal plane.
    print(Gen.transf.x.shape, Off.transf.xtran.shape)
    xytarg = np.copy(np.vstack((Off.transf.xtran, Off.transf.ytran)).T)
    if submedian:
        xytarg -= np.median(xytarg,axis=0)[None,:]
    LSQ = Leastsq2d(Gen.transf.x, Gen.transf.y, deg=degpoly, \
                    norescale=True, \
                    xytarg=xytarg)

    # Interpret the pars
    coefs = sixterm2d.getpars(LSQ.pars)
    print("Fit pars:", LSQ.pars)
    print("As coefs:", coefs)

    # Find the residuals and their magnitudes
    resids = LSQ.ev() - xytarg
    rmags = np.sqrt(resids[:,0]**2 + resids[:,1]**2)
    
    # Now show these:
    fig11 = plt.figure(11)
    fig11.clf()
    ax11 = fig11.add_subplot(2,2,1)
    ax12 = fig11.add_subplot(2,2,2)
    
    # Because we're actually plotting different attributes, we lift
    # those out here:
    wline = [Gen.grid_whichline, Off.grid_whichline]
    xshos = [Gen.transf.x, Off.transf.xtran]
    yshos = [Gen.transf.y, Off.transf.ytran]
    labls = [r'$\alpha_0, \delta_0 = (%.2f, %.2f)$' % (alpha0, delta0), \
             r'$\Delta \alpha_0, \Delta \delta_0 = %.2e, %.2e$' \
             % (dalpha0, ddelta0)]

    labelsdone = -1
    for lineid in np.unique(wline[0]):
        labelsdone += 1
        
        for iset in range(len(xshos)):
            bthisline = wline[iset] == lineid
            labl = labls[iset]
            if labelsdone > 0:
                labl=None
            dum11a = ax11.plot(xshos[iset][bthisline], \
                               yshos[iset][bthisline], \
                               color=colors[iset], \
                               ls=lss[iset], \
                               lw=0.5, \
                               label=labl)

    ax11.set_xlabel(r'$\xi$')
    ax11.set_ylabel(r'$\eta$')
    leg = ax11.legend()

            
    # now show the residuals
    dum = ax12.quiver(Off.transf.xtran, \
                      Off.transf.ytran, \
                      resids[:,0], resids[:,1], rmags, cmap='viridis_r')
    cbar2 = fig11.colorbar(dum, ax=ax12)
    
    ax12.set_xlabel(r'$\xi$')
    ax12.set_ylabel(r'$\eta$')
    ax12.set_title('Residuals, polynomial order %i' % (degpoly))

    # For interest, show the residuals after subtracting the median
    # offset. It should look like a rotation.
    dxmed = np.median(Off.transf.xtran - Gen.transf.x)
    dymed = np.median(Off.transf.ytran - Gen.transf.x)
    ax13 = fig11.add_subplot(2,2,3)
    dum13 = ax13.quiver(Gen.transf.x, Gen.transf.y, \
                        Off.transf.xtran - dxmed - Gen.transf.x, \
                        Off.transf.ytran - dymed - Gen.transf.y, \
                        scale=1.0, scale_units='xy', angles='xy')

    ax13.set_title('Offsets minus median')
    ax13.set_xlabel(r'$\xi$')
    ax13.set_ylabel(r'$\eta$')

    # histogram of the residual magnitudes
    ax14 = fig11.add_subplot(2,2,4)
    dum14 = ax14.hist(resids[:,0], bins=50, alpha=0.5, label=r'$\xi$')
    dum15 = ax14.hist(resids[:,1], bins=50, alpha=0.5, label=r'$\eta$')

    ax14.set_title('Distribution of residuals')
    leg14 = ax14.legend()

    fig11.subplots_adjust(wspace=0.4, hspace=0.4)
