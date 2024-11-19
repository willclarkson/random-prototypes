#
# apply2d.py
#

# WIC 2024-11-12 - methods to apply MCMC2d results

import numpy as np

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
        
        # Number to evaluate, xy samples
        self.neval = np.copy(neval)
        self.initsamples_xieta()
        
        # datapoints for evaluation
        self.xy = None
        self.covxy = None

        # Source-frame covariances as an object with methods we may
        # want
        self.covobj = None
        
        # If producing datapoints by perturbing a "truth" set, use
        # these for the reference points
        self.xyref = np.array([])

        # Covariances in the sample plane
        self.med_xieta = np.array([])
        self.cov_xieta = None
        
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
        
    def initsamples_xieta(self):

        """Initializes xy samples arrays"""

        self.samples_xi = np.array([])
        self.samples_eta = np.array([])

    def setupsamples_xieta(self):

        """Sets up arrays to hold the transformed samples"""

        if np.ndim(self.xy) < 2:
            return
        
        # Separate arrays for each coordinate for now.
        ndata = np.shape(self.xy)[0]
        self.samples_xi = np.zeros(( self.neval, ndata ))
        self.samples_eta = np.zeros(( self.neval, ndata ))

    def runsamples_uncty(self):

        """Runs samples under the uncertainty distribution"""

        if self.covobj is None:
            return

        for isample in range(self.neval):
            xypert = self.covobj.getsamples()
            self.setdata(self.xy + xypert)
            
            self.applytransf(isample)
            
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

        """Commputes the medians and covariances in xi, eta of the samples"""

        if np.size(self.samples_xi) < 1:
            return

        samples_xieta = np.stack((self.samples_xi, \
                                  self.samples_eta), axis=1)

        self.med_xieta = np.median(samples_xieta, axis=0).T
        self.cov_xieta = CovarsNx2x2(xysamples=samples_xieta)
        self.cov_xieta.eigensFromCovars()
        
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
                 plotsamples=True):

    """Performes monte carlo sampling of the source uncertainty,
propagated through to the target frame.

Inputs:

    nsamples = how many samples we want to draw

    pathpset = path to parameter set to be used for the transformation

    pathobs = path to observations including uncertainty covariances

    plotsamples = do a scatter plot of the samples

"""

    US = Evalset(pathpset=pathpset, neval=nsamples, pathobs=pathobs)
    US.getsamples()
    US.getobs()
    US.covfromobs()
    
    US.setupsamples_xieta()
    US.runsamples_uncty()

    # Perform statistics on the samples
    US.samples_stats()
    print(US.med_xieta.shape)
    print(US.cov_xieta.covars.shape)
    print(US.cov_xieta.majors[0:4])
    print(US.cov_xieta.minors[0:4])
    
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
    ax2.set_title('Samples from source-frame uncertainty distribution')
    
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

        
        
