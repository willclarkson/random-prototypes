#
# apply2d.py
#

# WIC 2024-11-12 - methods to apply MCMC2d results

import numpy as np
from scipy import stats

import matplotlib.pylab as plt
plt.ion()

# While developing, import the parent modules for the transformations
# and data
import unctytwod
import parset2d
import obset2d

# for drawing samples from the covariances
from weightedDeltas import CovarsNx2x2

class Evalset(object):

    """Sets of evaluations of the transformation."""

    # Should support both repeated evaluations on the same data and
    # the same parameters on different datasets

    def __init__(self, \
                 pathpset='test_parset_guess.txt', \
                 pathflat='test_flat_fitPoly_500_order1fit1_noprior_run2.npy', \
                 pathobs='', \
                 neval=100):

        # Paths to any input files
        self.pathpset = pathpset[:]
        self.pathflat = pathflat[:]
        self.pathobs = pathobs[:]
        
        # The transformation object, flat samples
        self.transf = None
        self.parsamples = None
        self.pset = None
        self.obset = None
        
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
    if not transfnamesok(pset):
        return None, np.array([])

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

def unctysamples(nsamples=10, \
                 pathpset='test_parset_guess_poly_deg2_n100.txt', \
                 pathobs='test_obs_src.dat', \
                 plotsamples=True, \
                 unifcovs=False):

    """Performes monte carlo sampling of the source uncertainty,
propagated through to the target frame.

Inputs:

    nsamples = how many samples we want to draw

    pathpset = path to parameter set to be used for the transformation

    pathobs = path to observations including uncertainty covariances

    plotsamples = do a scatter plot of the samples

    unifcovs = assign uniform covariances (useful for testing)

"""

    US = Evalset(pathpset=pathpset, neval=nsamples, pathobs=pathobs)
    US.getsamples()
    US.getobs()
    US.covfromobs()

    # Replace covariances with uniform covariances?
    if unifcovs:
        US.gencovunif()
    
    # Pass the samples to the transformation and compute the
    # propagated covariance
    US.setdata()
    US.propagate_covar()

    # Now set up the samples and run the parametric monte carlo
    US.keepxysamples = True
    US.setupsamples_xieta()
    US.runsamples_uncty()

    # Perform statistics on the samples
    US.samples_stats()

    # Perform statistics on the *generated* samples, just to ensure
    # that part works...
    if unifcovs:
        print("### Generated samples, source frame: ####")

        # What was sent to the generator...
        US.covobj.eigensFromCovars()
        print("unifcovs DBG - majors:", US.covobj.majors[0:4])
        print("unifcovs DBG - minors:", US.covobj.minors[0:4])

        # ... and what was produced
        print("Samples:", US.samples_x.shape, US.samples_y.shape)
        med, cov, skew, kurt = samples_moments(US.samples_x, \
                                               US.samples_y, \
                                               methcent=np.mean)
    
        print("Computed covxy:")
        print(cov.majors[0:4])
        print(cov.minors[0:4])


        print("#####")


    print("===== Transformed covariances =====")
    #print(US.med_xieta.shape)
    #print(US.cov_xieta.covars.shape)
    print("majors computed:", US.cov_xieta.majors[0:4])
    print("minors computed:", US.cov_xieta.minors[0:4])

    # cross-check
    print("cross-check INFO:")
    print("majors propag:", US.cov_propagated.majors[0:4])
    print("minors propag:", US.cov_propagated.minors[0:4])

    # print(US.cov_propagated.covars[10])
    
    ## stack the xi, eta samples together and do statistics on them
    #samples_xieta = np.stack((US.samples_xi, US.samples_eta), axis=1)
    ### print(np.ndim(samples_xieta), np.shape(samples_xieta))
    #covsamples = CovarsNx2x2(xysamples=samples_xieta)
    
    ## print("DBG:", covsamples.covars[0])
    
    # Scatter plot of the samples
    if not plotsamples:
        return

    # Just plot the point clouds for the moment!
    fig2 = plt.figure(2)
    fig2.clf()
    ax2 = fig2.add_subplot(111)
    dum = ax2.scatter(US.samples_xi, US.samples_eta, s=1)
    ax2.set_xlabel(r'$\xi$')
    ax2.set_ylabel(r'$\eta$')
    ax2.set_title('Transformed samples from source-frame uncertainty')

    # Plot the original point clouds
    if np.size(US.samples_x) < 1:
        return
    fig3 = plt.figure(3)
    fig3.clf()
    ax3 = fig3.add_subplot(111)
    dum = ax3.scatter(US.samples_x, US.samples_y, s=1)
    ax3.set_xlabel(r'$X$')
    ax3.set_ylabel(r'$Y$')
    ax3.set_title('Samples from source-frame uncertainty')
    
    
def traceplot(neval=10, \
              pathpset='test_parset_guess_poly_deg2_n100.txt', \
              pathflat='test_flat_fitPoly_100_order2fit2_noprior_run1.npy'):

    """Evaluates the flat samples on a grid of coords"""

    ES = Evalset(pathpset=pathpset, \
                 pathflat=pathflat, \
                 neval=neval)
    ES.getsamples()

    print(ES.pset.lmodel)
    
    # Set up evaluation coords
    ES.gengrid()

    print("DBG: grid line IDs:")
    print(np.size(ES.grid_whichline))
    print(np.unique(ES.grid_whichline))
    
    # Set up and run the evaluations
    ES.checkneval()
    ES.setupsamples_xieta()
    ES.runsamples_pars()
    
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
    
    # now we use line plots. matplotlib has a linecollection
    # capability that might make this faster. For the moment we do
    # things in a simpler way:

    line_ids = np.unique(ES.grid_whichline)
    for lineid in line_ids:
        bthisline = ES.grid_whichline == lineid

        for isho in range(neval):
            xthis = ES.samples_xi[isho][bthisline]
            ythis = ES.samples_eta[isho][bthisline]

            if isho < 1:
                color = 'k'
                zorder=10
                alpha=0.5
            else:
                color='b'
                zorder=1
                alpha=0.02
            
            dum = ax1.plot(xthis, ythis, \
                           color=color, zorder=zorder, \
                           alpha=alpha)
    
    #for isho in range(neval):
        #dum2 = ax1.scatter(ES.samples_xi[isho], ES.samples_eta[isho], s=.5, \
        #                   alpha=0.05, color='b')

        
        
