#
# weightedDeltas.py
# 

#
# 2020-05-29 WIC - collects some methods for finding the weighted
# positional deltas (on the sphere) and the weighted least squares fit
# between positions.
#

import time, datetime
import numpy as np
import copy
import os

# for plotting
import matplotlib.pylab as plt
from matplotlib.collections import EllipseCollection, LineCollection
from matplotlib.patches import Ellipse
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms

# matplotlib.rc('text', usetex=True) This didn't work well with
# corner... maybe send in as a plot argument to corner??

# for the param plots
import corner

# When exporting the covariance-pairs in the monte carlo, the
# astropy.table functionality is particularly useful. Later we could
# imagine dropping down to numpy structured arrays just to decrease
# the number of dependencies. 
from astropy.table import Table, Column

# for estimating weights in the ''simpler'' approach
from covstack import CovStack

# For experimenting with MCMC
import likesMCMC

import emcee
from scipy.optimize import minimize


class CooSet(object):

    """Set of coordinates with optional weights"""

    def __init__(self, ra=np.array([]), de=np.array([]), \
                     X=np.array([]), Y=np.array([]), Z=np.array([]), \
                     wgts=np.array([]) ):

        # All coordinates are assumed to be input in DEGREES.

        # equatorial
        self.ra = copyAsVec(ra)
        self.de = copyAsVec(de)

        # cartesian
        self.X = copyAsVec(X)
        self.Y = copyAsVec(Y)
        self.Z = copyAsVec(Z)

        # weights
        self.wgts = copyAsVec(wgts)

        # If nonzero input given, ensure the components are populated.
        if np.size(self.ra) > 0:
            if np.size(self.X) != np.size(self.ra):
                self.equat2xyz()
        else:
            if np.size(self.X) > 0:
                self.xyz2equat()

        # By this point, initialize the weights if not already set
        if np.size(self.wgts) != np.size(self.X):
            self.initWeights()

    def initWeights(self):

        """Initializes the weights if size different from data"""

        # An empty array is easier for the calling method to
        # recognize...
        self.wgts = np.array([])

    def equat2xyz(self):

        """Overwrites X,Y,Z with their conversion from ra, dec."""

        cosAlpha = np.cos(np.radians(self.ra))
        sinAlpha = np.sin(np.radians(self.ra))

        cosDelta = np.cos(np.radians(self.de))
        sinDelta = np.sin(np.radians(self.de))

        self.X = cosDelta * cosAlpha
        self.Y = cosDelta * sinAlpha
        self.Z = sinDelta

    def xyz2equat(self):

        """Overwrites ra, de with the conversion from X,Y,Z."""

        # XYZ should already be components of a unit vector, but let's
        # be defensive and insist on normalizing by the length...
        r = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        
        pln = np.sqrt( (self.X*r)**2 + (self.Y*r)**2 )

        self.ra = np.degrees(np.arctan2(self.Y*r, self.X*r))
        self.de = np.degrees(np.arctan2(self.Z*r, pln))

        # wrap the right ascension to be positive
        bNeg = self.ra < 0.
        self.ra[bNeg] += 360.

class CooPair(object):

    """Set of two coordinates that we'll want to map. At the moment we
    trust the user to supply two coo objects with the same number of
    rows, which are to be compared element by element."""

    def __init__(self, coo1=None, coo2=None):

        self.cooOne = coo1
        self.cooTwo = coo2

        # weights for the computation
        self.wts = np.array([])
        
        # some internal variables
        self.nPts = np.size(self.cooOne.X)
        self.sumWgts = 0.
        self.nUsed = np.copy(self.nPts) # convenience variable

        # Anticipating that we may want to try bootstrapping,
        # sigma-clipping, or other selections, we'll set an index
        # array giving the objects to actually use in the comparison.
        self.gUse = np.arange(self.nPts, dtype=np.long)

        # the weighted difference between the sets
        self.initDeltaDiff()

        # convenience variable when calling
        self.dxyz = np.array([])

        # actions on initialization
        self.populateBiweights()

    def initDeltaDiff(self):

        """Initializes the difference in coords"""

        self.avgDX = 0.
        self.avgDY = 0.
        self.avgDZ = 0.
        
        self.dXYZ = np.zeros(3, 'double')

    def populateBiweights(self):

        """Populates the weighting to be used in the comparison. Convention:

         - If BOTH objects have weights then the weights are added.

         - If only one has weights, then its weights are used.
  
         - If neither has weights, then uniform weighting of 1. for
           every object is used."""

        nw1 = np.size(self.cooOne.wgts)
        nw2 = np.size(self.cooTwo.wgts)
        
        if nw1 > 0:
            if nw2 == nw1:
                self.wts = self.cooOne.wgts + self.cooTwo.wgts
            else:
                self.wts = self.cooOne.wgts
        else:
            if nw2 > 0:
                self.wts = self.cooTwo.wgts
            else:
                self.wts = np.ones(self.nPts, 'double')

    def findSumWeights(self):

        """Sums the weights of the objects used"""

        self.sumWgts = np.sum(self.wts[self.gUse])
        self.nUsed = np.size(self.gUse)

    def findDiffXYZ(self):

        """Finds the weighted average difference between the XYZ
        coordinate sets"""
        
        # First find the sum of the weights for this iteration
        #self.findSumWeights()
        
        # A couple of convenience views (yes I know we could do this
        # multidimensionally!)
        wg = self.wts[self.gUse]
        sumWts = np.sum(wg)
        dxg = self.cooTwo.X[self.gUse] - self.cooOne.X[self.gUse]
        dyg = self.cooTwo.Y[self.gUse] - self.cooOne.Y[self.gUse]
        dzg = self.cooTwo.Z[self.gUse] - self.cooOne.Z[self.gUse]

        self.avgDX = np.sum(dxg * wg)/sumWts
        self.avgDY = np.sum(dyg * wg)/sumWts
        self.avgDZ = np.sum(dzg * wg)/sumWts

        # create our convenience-view
        self.mergeDXYZ()

        # pass up the number of used objects too
        self.nUsed = np.size(self.gUse)

    def mergeDXYZ(self):

        """Merges dx, dy, dz into a vector for convenient calling"""

        self.dXYZ = np.array([self.avgDX, self.avgDY, self.avgDZ])

class CooBoots(object):

    """Bootstrap-resampling object"""

    # Implementation note: I'm trying to ensure that most of the
    # details of what is done per cycle is handled by the object read
    # in to this class, that way this should be reasonably easy to
    # adapt to other things (like finding the linear transformation
    # between coordinate sets on the plane).

    def __init__(self, CooPair=None, nBoots=100, \
                     cenRA=0., cenDE=0., \
                     fResample=1., \
                     runOnInit=True, \
                     Verbose=False):

        self.cooPair = CooPair

        # how many resamplings?
        self.nBoots = nBoots
        self.fResample = np.clip(fResample, 0., 1.)
        self.nPerSample = 1

        # We're going to be interested in the distribution of shifts
        # to the pointing, which will be converted from the
        # cartesian. So that we don't have to mess around with the
        # nonlinearity of the conversion, we just accept input central
        # pointings.
        self.cenRA = np.copy(cenRA)
        self.cenDE = np.copy(cenDE)

        # results arrays. We'll grow them with each trial so that we
        # don't have to deal with empty planes later if anything goes
        # wrong...
        self.aRect = np.array([])
        self.aEqua = np.array([])

        # sample statistics
        self.summCenter = 0.
        self.summStddev = 0. 

        # Control variables
        self.Verbose = Verbose
        
        # set things up
        self.calcSampleSize()

        if runOnInit:
            self.doSamples()
            self.calcSummaryStats()

    def calcSampleSize(self):

        """Computes the sample size"""

        self.nSample = int(self.cooPair.nPts * self.fResample)

    def doSamples(self):

        """Runs thru the samples"""

        for iSampl in range(self.nBoots):

            if self.Verbose:
                print("Sample %i:" % (iSampl))

            self.doOneSampling()

    def drawResample(self):

        """Draws a resampling"""

        # do nothing if sample size zero...
        if self.nSample < 1:
            return np.array([])

        # random integers
        iMax = self.cooPair.nPts 
        gSampl = np.random.randint(0, iMax, self.nSample, np.long)

        return gSampl

    def doOneSampling(self):

        """Does a single sampling"""

        # Pass this to the coo pair to use as a sample
        self.cooPair.gUse = self.drawResample()

        # do nothing if this made us produce an empty sample
        if np.size(self.cooPair.gUse) < 1:
            return

        # now have the coord pair compute the statistics for this
        # subset:
        self.cooPair.initDeltaDiff()
        self.cooPair.findDiffXYZ()

        # now we compute the delta for the pointing and pass these up
        # to the bucket of results.
        corrRA, corrDE = shiftAngleByXYZ(self.cenRA, self.cenDE, \
                                             self.cooPair.dXYZ)
        dEQ = np.array([ corrRA - self.cenRA, corrDE-self.cenDE ])[:,0]

        if np.size(self.aRect) < 1:
            self.aRect = np.copy(self.cooPair.dXYZ)
            self.aEqua = np.copy(dEQ)
        else:
            self.aRect = np.vstack(( self.aRect, self.cooPair.dXYZ ))
            self.aEqua = np.vstack(( self.aEqua, dEQ ))

    def calcSummaryStats(self):

        """Calculates summary statistics"""

        if np.size(self.aEqua) < 1:
            return

        # Can't do much with fewer than three simulations...
        if np.shape(self.aEqua)[0] < 3:
            return

        self.summCenter = np.median(self.aEqua, axis=0)
        self.summStddev = np.std(self.aEqua, axis=0)

        # correct the summary standard deviation for the sub-sampling        
        nOrig = self.cooPair.nPts
        if nOrig == self.nSample:
            return

        corrFac = np.sqrt(np.float(self.nSample - 1)/nOrig)
        #print("INFO - correction factor:", corrFac, nOrig, self.nSample)
        self.summStddev *= corrFac
      
class CooTP(object):

    """Class to hold tangent-plane coordinates and related methods."""

    # Separate from CooSet above since in this case the linear least
    # squares methods may be valid.

    def __init__(self, ra=np.array([]), de=np.array([]), \
                     xi=np.array([]), eta=np.array([]), \
                     ra0 = None, de0 = None, \
                     Verbose=False):
                    
        # Vectors of coordinates
        self.ra = np.copy(ra)
        self.de = np.copy(de)
        self.xi = np.copy(xi)
        self.eta = np.copy(eta)

        # tangent point
        self.ra0 = np.copy(ra0)
        self.de0 = np.copy(de0)

        # Control variables
        self.Verbose = Verbose

        # if nonero input, ensure the components are populated
        if np.size(self.ra) > 0:
            if np.size(self.xi) != np.size(self.ra):
                self.sphere2tp()
        else:
            if np.size(self.xi) > 0:
                self.tp2sphere()

    def sphere2tp(self):

        """Populates the tangent plane co-ordinates given the
        co-ordinates on the sphere and the tangent point"""

        if not self.canTransform(s2TP=True):
            return

        dra = self.ra - self.ra0
        dde = self.de - self.de0

        # trig terms
        cosd = np.cos(np.radians(self.de))

        cosd0 = np.cos(np.radians(self.de0))
        sind0 = np.sin(np.radians(self.de0))

        cosdra = np.cos(np.radians(dra))
        sindra = np.sin(np.radians(dra))
        
        cosdde = np.cos(np.radians(dde))
        sindde = np.cos(np.radians(dde))

        # denominator, same for both output coords
        denom = cosdde - cosd0 * cosd*(1.0-cosdra)
        
        self.xi = cosd * sindra / denom
        self.eta = (sindde + sind0*cosd*(1.0-cosdra)) / denom

    def tp2sphere(self):

        """Populates the spherical coords given the tangent plane
        coordinates"""

        if not self.canTransform(s2TP=False):
            return

        # trig terms needed
        cosd0 = np.cos(np.radians(self.de0))
        sind0 = np.cos(np.radians(self.de0))

        xiRad  = np.radians(self.xi)
        etaRad = np.radians(self.eta)

        brack = cosd0 - xiRad * sind0

        # alpha - alpha_0, converting to the range 0 < da < 360
        # deg.
        da = np.degrees(np.arctan2(brack, xiRad))
        da[da < 0] += 360.

        self.ra = da + self.ra0

        # We want the declination to stay in the range (-90, +90) so
        # we use arctan, not arctan2:
        numer = np.sqrt(xiRad**2 + brack**2)
        denom = sind0 + xiRad*cosd0

        self.de = np.arctan(numer / denom)
        

    def canTransform(self, s2TP=True):

        """Utility: checks whether we can transform. The error
        messages are deferred to this method."""

        if not np.isscalar(self.ra0):
            if self.Verbose:
                print("CooTP.canTransform WARN - tangent point ra0 not specified.")
            return False

        if not np.isscalar(self.de0):
            if self.Verbose:
                print("CooTP.canTransform WARN - tangent point de0 not specified.")
            return False

        # Which coordinate sets are we transforming?
        if s2TP:
            sOne = 'ra'
            sTwo = 'de'
        else:
            sOne = 'xi'
            sTwo = 'eta'

        src1 = getattr(self,sOne)
        src2 = getattr(self,sTwo)

        if np.size(src1) < 1:
            if self.Verbose:
                print("CooTP.canTransform WARN - source coord %s not populated" \
                          % (sOne))
            return False
        
        if np.size(src2) < 1:
            if self.Verbose:
                print("CooTP.canTransform WARN - source coord %s not populated" \
                          % (sTwo))
            return False

        if np.size(src1) != np.size(src2):
            if self.Verbose:
                print("CooTP.canTransform WARN - source coords %s, %s have different lengths: (%i, %i)" % (sOne, sTwo, np.size(src1), np.size(src2)))

        return True

    def updatePointing(self, raZero=None, deZero=None, sphere2tp=True):

        """Updates the tangent point with user supplied values, then
        updates the coordinates to ensure consistency. Arguments:

        -- raZero = tangent point RA

        -- deZero = tangent point DEC

        -- sphere2tp -- keep spherical coords fixed and update the
           tangent plane coords using the new tangent point. (If
           False, then the tangent plane coords are held fixed and the
           spherical coords are updated.)

        """

        if not np.isscalar(raZero):
            if self.Verbose:
                print("CooTP.updateSphere WARN - no new ra tangent point given")
            return

        if not np.isscalar(deZero):
            if self.Verbose:
                print("CooTP.updateSphere WARN - no new dec tangent point given")
            return

        # now update the tangent point
        self.ra0 = np.copy(raZero)
        self.de0 = np.copy(deZero)

        if sphere2tp:
            self.sphere2tp()
        else:
            self.tp2sphere()

class CooXY(object):

    """Object holding focal plane coordinates"""

    # I keep the coordinates as 1D arrays for consistency with the
    # other coord objects (rather than having a single M-dimensional
    # vector per coordinate set).

    def __init__(self, x=np.array([]), y=np.array([]), \
                     xRef = 0., yRef = 0.):

        self.x = np.copy(x)
        self.y = np.copy(y)

        # Reference point. Located to 0,0 rather than None, None
        # because we could always update after fitting with 0,0
        # reference.
        self.xRef = np.copy(xRef)
        self.yRef = np.copy(yRef)

        self.dx = np.array([])
        self.dy = np.array([])
        self.calcDx()

    def calcDx(self):

        """Updates x-xref"""

        if np.size(self.x) < 1:
            return

        self.dx = self.x - self.xRef
        self.dy = self.y - self.yRef
        
    def updateXref(self, xR=None, yR=None):

        """Utility: updates the reference position and recalculates
        the offsets for consistency"""

        # Do nothing if no new xRef actually given
        if not np.isscalar(xR) or not np.isscalar(yR):
            return

        self.xRef = np.copy(xR)
        self.yRef = np.copy(yR)
        self.calcDx()
        

class PairPlanes(object):

    """Class to hold and fit transformations between two coord sets,
    with a linear transformation assumed and multivariate weights are
    allowed for (the weights cannot depend on the transformation
    parameters)."""

    def __init__(cooSrc=None, cooTarg=None, \
                     wgts=np.array([]), \
                     colSrcX='dx', colSrcY='dy', \
                     colTargX='xi', colTargY='eta',\
                     nRowsMin=6, \
                     Verbose=True):

        """Arguments: 

        -- cooSrc = coordinate object for the source coords
        
        -- cooTarg = coordinate object for the target coords

        -- wgts = 1- or 2D weights array

        -- colSrcX = attribute name for the 'x' column in the source object

        -- colSrcY = attribute name for the 'y' column in the source object

        -- colTargX = attribute name for the 'x' column in the target object

        -- colTargY = attribute name for the "y" column in the target object

        -- nRowsMin = minimum number of rows

        -- Verbose = print lots of screen output for diagnostic purposes

        """

        # objects
        self.cooSrc = cooSrc
        self.cooTarg = cooTarg
        self.wgts = np.copy(wgts)

        # columns for source, target
        self.colSrcX = colSrcX[:]
        self.colSrcY = colSrcY[:]
        self.colTargX = colTargX[:]
        self.colTargY = colTargY[:]

        # Control variables
        self.nRowsMin = np.max([0, nRowsMin])   # minimum number of sources
        self.Verbose = Verbose

        # some internal variables. For consistency with my notes on
        # this I will call the target vectors "xi" and the input
        # vectors "x", generalizing to "u".
        self.xi = np.array([])
        self.u = np.array([])
        self.W = np.array([]) # multivariate weights
        self.uuT = np.array([]) # the matrix u.u^T

        # the pieces of the solution
        self.beta = np.array([])
        self.C = np.array([])
        self.alpha = np.array([]) # the vectorized solution...
        self.A = np.array([]) # ... and its matrix form

        # 2020-06-04 breaking into reference + linear transformation
        # comes here??
        #
        # (No, we create a separate class to deal with the parameter
        # matrix. Thoughts about that object: (i). refactor the "A"
        # into "1x2xM" so that the same object can be used when
        # constructing or interpreting artificial data; (ii) enforce a
        # convention such that a,d are ALWAYS the reference point and
        # b,c,e,f are ALWAYS the general linear transformation.) To be
        # implemented.


        # populate the source and target vectors
        self.populateSrcTarg()
        self.parseWeights()
        self.buildUUT()
        self.buildCmatrix()
        self.buildBeta()

        # now solve for the alpha vector
        self.solveForAlpha()

    def rowsAreOK(self):

        """Checks that both coordinate objects have the right
        quantities and that their sizes are the same."""

        # This takes a somewhat verbose view: if self.Verbose is set,
        # *all* error messages are printed to screen.

        canProceed = True

        for attr in [self.colSrcX, self.colSrcY]:
            if not hasattr(self.cooSrc, attr):
                canProceed = False
                if self.Verbose:
                    print("PairPlanes.rowsAreOK WARN - src has no attr %s" \
                              % (attr))

        for attr in [self.colTargX, self.colTargY]:
            if not hasattr(self.cooTarg, attr):
                canProceed = False
                if self.Verbose:
                    print("PairPlanes.rowsAreOK WARN - targ has no attr %s" \
                              % (self.colTargX))
        
        nSrc = np.size(getattr(self.cooSrcX, self.colSrcX))
        nTar = np.size(getattr(self.cooTargX, self.colTargX))

        if nSrc < self.nRowsMin:
            canProceed = False
            if self.Verbose:
                print("PairPlanes.rowsAreOK WARN - src object has < %i rows" \
                          % (self.nRowsMin))
        
        if nSrc != nTar:
            canProceed = False
            if self.Verbose:
                print("PairPlanes.rowsAreOK WARN - src, targ different n(rows).")

        return canProceed

    def populateSrcTarg(self):

        """Populates the source and target rows from the coord
        objects. We keep the vectors xi_i, u_i because we want to be
        able to easily evaluate the fom for each row. So:"""

        if not rowsAreOK():
            return

        xTarg = getattr(self.cooTarg, self.colTargY)
        yTarg = getattr(self.cooTarg, self.colTargY)

        xSrc = getattr(self.cooSrc, self.colSrcX)
        ySrc = getattr(self.cooSrc, self.colSrcY)

        oSrc = np.ones(np.size(xSrc), dtype='double')

        # Construct the [Nx2] target array and src array...
        self.xi = np.vstack(( xTarg, yTarg ))
        self.u = np.vstack(( oSrc, xSrc, ySrc )).T

    def parseWeights(self):

        """Checks that the weights are consistent with the coordinates"""

        shData = np.shape(self.xi)
        nRows = shData[0]
        if nRows < self.nRowsMin or np.size(shData) > 2:
            if self.Verbose:
                print("PairPlanes.parseWeights WARN - input data unusual shape:", shData)
                return
        nCols = shData[1] # for the moment this should always be 2.
        
        # initialize the Nx2x2 weights array to the identity stack.
        eyePlane = np.eye(nCols, dtype='double')
        eyeStack = np.repeat(eyePlane[np.newaxis,:,:], nRows, axis=0)
        self.W = np.copy(eyeStack)

        # if no weights were input, use the identity as weights.
        if np.size(self.wgts) < 1:
            if self.Verbose:
                print("PairPlanes.parseWeights INFO - no weights supplied. Using unity.")
            return

        # In the simplest case, we have been supplied weights with
        # identical shape to the target data. In that instance, just
        # copy and we're done.
        shWgts = np.shape(self.wgts)
        if shWgts == shData:
            self.W = np.copy(self.wgts)
            return

        # If the shapes do not agree, one possibility is that the
        # weights have the wrong length. Check that here:
        nWgts = shWgts[0]        
        if nWgts != nRows:
            if self.Verbose:
                print("PairPlanes.parseWeights WARN - data and wgts array different lengths: data %i, weights %i" % (nRows, nWgts))
            return

        # If we got to here, then we have the correct number N of
        # weights, but they are not the same dimensions as the N x 2 x
        # 2 we expect. Handle the cases:
        dimenWgts = np.size(shWgts)

        # For scalar weights, every plane is the identity x the scalar
        # weight:
        if dimenWgts < 2:
            self.W = eyeStack * self.wgts[:, np.newaxis, np.newaxis]
            return

        # In the two dimensional weight cases, we have been given
        # diagonal weights (e.g. in xi[0], xi[1] separately but with
        # no covariances).
        if dimenWgts == 2:
            colsWgts = shWgts[-1]

            # Catch the annoying condition when the weights are passed
            # as N x 1 array:
            if colsWgts == 1:
                self.W = eyeStack * self.wgts[:, np.newaxis, np.newaxis]
                return
                
            # Now deal with the case where we actually do have N x M
            # weight-arrays
            self.W = np.copy(eyeStack)
            for kDim in shWgts[-1]:
                self.W[:,kDim, kDim] *= self.wgts[:,kDim]
            return

        # If we got to here, then the weights have the right number of
        # rows, but are not each scalars, vectors or MxM matrices.
        if self.Verbose:
            print("PairPlanes.parseWeights WARN - weights size mismatch: Data, weights:", shData, shWgts)

    def buildUUt(self):

        """Builds the u.uT matrix"""

        # ... which I assume has a standard name in this context, I
        # just don't know what it is (calling it the "covariance" is
        # confusing.)
        self.uuT = np.einsum('ij,ik->ijk', self.u, self.u)
        
    def buildCmatrix(self):

        """Builds the C-matrix for the fitting."""

        # The following trick does the plane-by-plane Kronecker
        # product that we want, using np.einsum:
        i,j,k = np.shape(self.W)
        i,l,m = np.shape(self.uuT)

        # This is a two-step solution...
        #Cstack = np.einsum("ijk,ilm->ijlkm",self.W,self.uuT).reshape(i,j*l,k*m)
        #self.C = np.sum(Cstack, axis=0)

        # ... here it is in one step
        self.C = np.einsum("ijk,ilm->jlkm",self.W,self.uuT).reshape(j*l,k*m)

    def buildBeta(self):

        """Builds the array of outer((wgts x target),u^T"""

        # I'm not confident enough with np's einsum to do this all in
        # one go, so I break it into pieces. First the dot(W_i,
        # xi_i)...
        Wxi = np.einsum("ijk,ij->ik",self.W, self.xi)
        
        # ... now the outer products of  (Wxi)_i and u_T...
        WxiuT = np.einsum('ij,ik->ijk', Wxi, self.u)

        # ... now sum along the N datapoints...
        self.beta = np.ravel(np.sum(Wxiut, axis=0))

    def solveForAlpha(self):

        """Solves the expression beta = C.alpha for alpha"""

        if not self.canSolve():
            return

        self.alpha = np.linalg.solve(self.C, self.beta)

        # Let's try to be clever and get the dimensions of the matrix
        # from the quantities we are fitting:
        dimenTarg  = np.shape(self.xi)[-1]
        dimenModel = np.shape(self.u)[-1]

        self.A = self.alpha.reshape(dimenTarg, dimenModel)

    def canSolve(self):

        """Checks if we've populated the pieces we need yet to solve
        the linlstq"""

        okSolve = True
        if np.size(self.beta) < 1:
            if self.Verbose:
                print("PlanePair.canSolve WARN - beta not populated")
            okSolve = False

        if np.size(self.C) < 1:
            if self.Verbose:
                print("PlanePair.canSolve WARN - C not populated")
            okSolve = False
        
        return okSolve

#####  useful generally-found methods

def copyAsVec(x):

    """Returns a copy of the input as a vector, even if scalar"""

    if np.ndim(x) == 0:
        return np.array([np.copy(x)])
    return np.copy(x)
        
def shiftAngleByXYZ(ra=0., de=0., dxyz = np.zeros(3) ):

    """Utility routine - given an ra and a dec, converts to XYZ,
    shifts by dX, dY, dZ, and converts back"""

    coo = CooSet(ra, de)
    coo.X += dxyz[0]
    coo.Y += dxyz[1]
    coo.Z += dxyz[2]
    coo.xyz2equat()

    # get the converted coords
    raCorrected = coo.ra
    deCorrected = coo.de

    # if the input was passed as a scalar, ensure we also return a
    # scalar:
    if np.isscalar(ra):
        raCorrected = coo.ra[0]
        deCorrected = coo.de[0]

    return raCorrected, deCorrected

class LinearMapping(object):

    """Class of linear transformations, implemented as a stack of N x
    K x M arrays where N is the number of planes, M the number of
    terms in each dimension and K the dimension of the data space. If
    passed a K x M matrix, the array is converted into a 1 x K x M
    array for consistency"""

    # ... and so that I don't have to write basically the same
    # routines twice. This is written out this way so that I can use
    # the same framework to handle higher-order terms including
    # distortion, although that's likely to still require a bit of
    # tweaking.

    def __init__(self, Ain = np.array([]), \
                     Verbose=False):

        # Control variable
        self.Verbose = Verbose

        # the GIVEN matrix or stack, which we convert to an N x K x M
        # array
        self.Ain = np.copy(Ain)
        self.A = np.array([])
        self.parseInputStack()

        # Some special partitions for interpretation in our use-case,
        # where we will want to know linear transformations for 2D
        # parameters.
        self.consts = np.array([])  # will be N x 2
        self.squares = np.array([]) # will be N x 2 x 2

        # Object with the squares and converted parameters
        self.stackSquares = None

        # partition the stack of transformations in to reference
        # coords and planes
        self.partitionRefsAndPlanes()

        # Parse the array into the N x 2 x 2 stack with interpreted
        # parameters
        self.populateStack()

    def parseInputStack(self):

        """Ensures the stack is three dimensional or empty"""

        self.A = np.array([])

        # if the input stack is empty, there's nothing to do.
        if np.size(self.Ain) < 1:
            return

        nDim = np.size(np.shape(self.Ain))
        if nDim == 3:
            self.A = np.copy(self.Ain)
            return

        # If a two-dimensional matrix was sent in, convert it to 3D. I
        # think the most pythonic way to do it is to use the
        # np.newaxis. WHICH way we do this depends on what the user is
        # sending in. If they have found an array of N x 2 pairs
        # (i.e. if each transformation is just the offsets) then we
        # want to end up with N x 2 x 1. However if they've sent in a
        # single plane (e.g. 2 x 3) then we want to end up with 1 x 2
        # x 3. For the moment we use an ansatz:
        if nDim == 2:
            l1, l2 = self.Ain.shape
            if l2 > l1 or l2 == 1:
                self.A = np.copy(self.Ain[np.newaxis,:,:]) # given K x M
            else:
                self.A = np.copy(self.Ain[:,:,np.newaxis]) # given N x K
            return

        # If we got here then the dimension is 1. That's not one of
        # the use cases for which I wrote this, so we trigger a
        # warning if verbose is set.
        if nDim == 1:
            nRows = np.size(self.Ain)
            eyeStack = np.repeat(eyePlane[np.newaxis,:,:], nRows, axis=0)
            self.A = eyeStack * self.Ain[:,np.newaxis, np.newaxis]

            # This could fail if the user has passed a float as the
            # input array. In that case the routine SHOULD faill.
            if self.Verbose:
                print("LinearMapping.parseInputStack WARN - supplied array has unusual dimension %i. Replicated into scaled identities" % (nDim))

        # if we got here then we have nonzero size array but the
        # dimension is not 1,2 or 3. Throw a warning message and
        # return without populating the master array
        if self.Verbose:
            print("LinearMapping.parseInputStack WARN - unusual shape array:" \
                      % (np.shape(self.Ain)))
            
    def partitionRefsAndPlanes(self):

        """Separates out the components in the transformation
        corresponding to the xi_ref and the 4-term linear
        transformation for 2D coordinates"""

        if np.size(self.A) < 2:
            return

        n,k,m = self.A.shape

        # A few cases to think about here. If there is not room for a
        # constants array then we start the squares from column 0,
        # otherwise we start at column 1.
        if m != 2:
            self.consts = self.A[:,:,0]
            colSq = 1
        else:
            self.consts = np.array([])
            colSq = 0

        # If there is room for a separate squares array, populate it
        # here. Examples: (1x2x3) or (1x2x2) or (1x2x5).
        if m > 1 + colSq:
            self.squares = self.A[:,:,colSq:colSq+2]
        else:
            self.squares = np.array([])
        
    def populateStack(self):

        """Populates the N x 2 x 2 stack of linear transformation
        matrices, using the object's methods to convert the
        transformations into geometric parameters (sx, sy, rotation,
        skew)"""

        self.stackSquares = Stack2x2(self.squares)

    # 2020-06-10 I think it's better to have two methods rather than
    # one method with a conditional inside it. Trust the calling
    # method to know which it needs.

    # applyLinearPlane() is somewhat bare-bones, and I want it to fail
    # if given improper input. applyLinear() is a bit more
    # feature-rich and I want it to apply some level of defensiveness.

    def applyLinearPlane(self, x=np.array([]), y=np.array([])):

        """Applies the 0'th plane matrix transformation to every
        element in the given x, y arrays. Returns transformed x,
        transformed y."""

        Bx = np.matmul(self.squares[0], np.vstack((x,y)) )

        # This generates a (2, N) array that can be unpacked
        return Bx + self.consts.T

    def applyLinear(self, x=np.array([]), y=np.array([]) ):

        """Applies every planar transformation to the input points
        list, one plane per input point. Returns transformed x,
        transformed y."""

        if np.size(x) < 1 or np.size(y) < 1 or np.size(x) != np.size(y):
            return np.array([]), np.array([])

        if np.size(x) != self.squares.shape[0]:
            return np.array([]), np.array([])

        # now we want the matrices to be applied plane-by-plane
        aXY = np.vstack(( x, y )).T[:,:,np.newaxis]

        BxPlanes = np.matmul(self.squares, aXY)
        
        aTransf = BxPlanes[:,:,0] + self.consts
        
        #print("HERE:", np.shape(aXY), np.shape(self.squares), \
        #    np.shape(BxPlanes), np.shape(self.consts))

        # This generates a (2, N) array that can be unpacked.
        return aTransf.T
    
# Now for the operation A.V.A^T , which will come in handy when
# generating covariance matrices with axes that are not aligned
# with the co-ordinate system. This is useful enough that I have
# promoted it ouf the LinearModels object
def AVAt(A=np.array([]), V=np.array([])):

    """Applies the matrix expression A. V. A^T to input matrix (or
    stack of matrices) V. Returns an [N x K x K] matrix stack, where N
    is the depth of the largest out of A and V."""

    AA = ensureStack(A)
    VV = ensureStack(V)

    # Do some simple checking on the 3D stacks. Must be nonzero
    # size...
    if np.size(AA) < 1 or np.size(VV) < 1:
        return np.array([])

    # If both have >1 plane, then the number of planes must be equal.
    if AA.shape[0] > 1 and VV.shape[0] > 1:
        if AA.shape[0] != VV.shape[0]:
            return np.array([])

    # Once we get here, the operation is straightforward:

    # Plane-by-plane transpose
    AT = np.transpose(AA,(0,2,1))

    return np.matmul(AA, np.matmul(VV,AT))

def ensureStack(A=np.array([])):
    
    """Utility: returns input 2D or 3D array as 3D stack"""

    if np.size(A) < 1:
        return np.array([])

    dimA = np.size(np.shape(A))
    if dimA < 2 or dimA > 3:
        return np.array([])
    
    if dimA == 2:
        return A[np.newaxis,:]
    return A
    

class NormalEqs(object):

    """Finds the linear mapping from one set to the other, optionally
    finding the formal covariance estimate on the returned
    parameters. Initially written with 2D data in mind. Initialization
    arguments:

    x, y    = length-N arrays of source points

    xi, eta = length-N arrays of target points

    W       = N x 2 x 2 weight matrix. Easily generated by CovarsNx2x2
              object

    xref, yref = coords of source reference point (scalars)

    fitChoice = string, choice of transformation. '6term, similarity'

    flipx = flip the x-coordinate? (Default False)

    Verbose = print lots of screen output
    
    """

    # Work on generalizing to N-dimensional data later.

    def __init__(self, x=np.array([]), y=np.array([]), \
                     xi=np.array([]), eta=np.array([]), \
                     W=np.array([]), \
                     xref=0., yref=0., \
                     fitChoice='6term', flipx=False, \
                     Verbose=False):

        # the coords to be transformed 
        self.x = np.column_stack(( x-xref, y-yref ))

        # The target coords
        self.xi = np.column_stack(( xi, eta ))

        # The N x 2 x 2 weight array
        self.W = W

        # Keep track of the reference input coords
        self.xref = np.copy(xref)
        self.yref = np.copy(yref)

        # control variables
        self.fitChoice = fitChoice[:]
        self.flipx = flipx

        self.Verbose = Verbose

        # Internal variables
        self.pattern = np.array([]) # pattern matrix
        self.patternT = np.array([]) # its plane-by-plane transpose
        self.H = np.array([])  # The hessian (M x M)
        self.beta = np.array([]) # beta (M)

        # pattern object containing the sample to be used when fitting
        self.patnObj = None 

        # row-by-row fit statistic - for all objects
        self.s2All = np.array([])
        self.outlyNsigma = 4.

        # Which planes do we trust?
        self.gPlanes = np.array([])
        self.initGplanes()

        # The results
        self.pars = np.array([])  # M
        self.formalCov = np.array([]) # the formal covariance estimate

        # Results decomposed into reference point and 2x2 transformation
        self.xiRef = np.array([])   # 2-element vector
        self.BMatrix = np.array([]) # 2x2 matrix

        # The tangent point on the source system (the point
        # corresponding to xi = [0., 0.])
        self.xZero = np.array([])

    def initGplanes(self):

        """Initializes the selection index for planes we trust"""

        self.gPlanes = np.arange(self.x.shape[0], dtype='int')

        #if np.size(self.bPlanes) < 1:
        #    self.bPlanes = np.isfinite(self.x[:,0])

    def buildPattern(self):

        """Creates the pattern matrix using the pattern class"""

        scaleX = 1.
        if self.flipx:
            scaleX = -1.

        self.patnObj = ptheta2d(self.x[:,0], self.x[:,1], self.fitChoice, \
                                    xSign=scaleX)

        # Worth duplicating the pattern array for clarity elsewhere in
        # THIS object.
        self.pattern = self.patnObj.pattern 
        self.transposePattern()

    def transposePattern(self):

        """We'll need the plane-by-plane transpose of the pattern
        array at least twice. So we populate it here."""

        self.patternT = np.transpose(self.pattern, (0,2,1))

    def makeBeta(self):

        """Populates the beta array sum_i P^T W_i xi """

        PWxi = np.matmul(self.patternT, \
                             np.matmul(self.W, self.xi[:,:,np.newaxis]))

        # sum along the i dimension, but only the planes we trust
        self.beta = np.sum(PWxi[self.gPlanes], axis=0)

    def makeHessian(self):

        """Populates the Hessian matrix: sum_i P^T W_i P"""

        PWP = np.matmul(self.patternT, np.matmul(self.W, self.pattern))

        # Sum along the i dimension, but only the planes we trust
        self.H = np.sum(PWP[self.gPlanes], axis=0)
        
    def solveParams(self):

        """Solves the normal equations to find the parameters theta"""

        self.pars = np.linalg.solve(self.H, self.beta)

    def invertHessian(self):

        """Inverts the hessian to estimate the formal covariance of
        the parameters"""

        # We probably won't need to do this every time through a monte
        # carlo simulation, so I assign a separate variable for the
        # inverse of the Hessian.

        self.formalCov = np.linalg.inv(self.H)

    def estTangentPoint(self):

        """Inverts the linear part of the transformation to estimate
        the location of the tangent point (where \vec{xi} = \vec{0})
        on the detector."""

        if np.size(self.pars) < 1:
            return

        # Decompose the transformation into xi_ref, B...
        self.interpretBmatrix()

        vXref = np.array([self.xref, self.yref])
        BInv = np.linalg.inv(self.BMatrix)

        self.xZero = vXref - np.dot(BInv, self.xiRef)


    def interpretBmatrix(self):

        """Interprets the linear part of the transformation as the
        xi_ref and the linear part of the transformation so that we
        can invert it to estimate the location of the tangent
        point."""

        # The reference point, transformed onto the target coords
        self.xiRef = self.pars[[0,3]][:,0]

        # Initialize the 2x2 matrix to the identity, filling in the
        # pieces appropriately for the choice of transformation. 
        self.BMatrix = np.eye(2)
        self.BMatrix[0,0] = self.pars[1]
        self.BMatrix[0,1] = self.pars[2]

        if self.fitChoice.find('similarity') > -1:
            self.BMatrix[1,0] = 0.-self.BMatrix[0,1]
            self.BMatrix[1,1] = self.BMatrix[0,0]
            return

        # If we're here then the first six terms are the same as for
        # the 6-term linear transformation
        self.BMatrix[1,0] = self.pars[4]
        self.BMatrix[1,1] = self.pars[5]

    def doFit(self):

        """Wrapper that sets up and does the fitting"""

        self.buildPattern()
        self.makeBeta()
        self.makeHessian()
        self.solveParams()

    def applyTransfToFitSample(self):

        """Applies the transformation to the fit sample"""

        # Note that it does this to the entire sample. This is so that
        # the booleans always have the same length and I don't go nuts
        # trying to trace the booleans round an ever-shrinking sample
        # when sigma-clipping...

        self.patnObj.pars = np.copy(self.pars)
        self.xiPred = self.patnObj.evaluatePtheta()

    def evaluateS2all(self):

        """Evaluates the row-by-row fit statistic s^2 = (xi -
        xi_pred)^T W (xi - xi_pred)"""

        self.applyTransfToFitSample()
        deltAll = self.xi - self.xiPred
        self.s2All = np.matmul(deltAll.T, np.matmul(self.W, deltAll))

    def findOutliers(self):

        """Finds strong outliers"""

        # ( if W is accurately giving inverse variance weights, then
        # s^2 should be chi-square distributed. If we really believe
        # this is chi-square distributed then we could use the upper
        # tail to find and remove outliers.)
        peak = np.median(self.s2All)
        
        # 2020-06-15: come back to this because I'm not sure I have a
        # way to ID outliers yet that I trust for the general weighted
        # average.

class ptheta2d(object):

    """(Needs a better name...) Object to hold the pattern matrix and
    the parameters encoding a linear transformation"""

    def __init__(self, x=np.array([]), y=np.array([]), \
                     fitChoice='6term', pars=np.array([]), xSign=1.):

        # xSign - set to -1 to flip the x on initialization

        self.x = x * xSign
        self.y = y
        self.pars = pars
        self.fitChoice = fitChoice[:]

        self.pattern = np.array([])

        # Populate the pattern on initialization
        self.populatePattern()

    def populatePattern(self):

        """Applies the fit choice by populating the pattern matrix."""
        
        if self.fitChoice.find('similarity') > -1:
            self.patternSimilarity()
            return

        # default to the six-term case
        self.pattern6term()

    def pattern6term(self):

        """Populates the pattern matrix for the 6-term linear
        transformation"""

        if np.size(self.x) < 1:
            return

        # Do this the hardcoded way (there's probably an np.einsum way
        # to do this in a couple of lines...)
        nRows = self.x.shape[0]
        self.pattern = np.zeros(( nRows, 2, 6 ))

        self.pattern[:,0,0] = 1.
        self.pattern[:,0,1] = self.x
        self.pattern[:,0,2] = self.y
        self.pattern[:,1,3] = 1.
        self.pattern[:,1,4] = self.x
        self.pattern[:,1,5] = self.y

    def patternSimilarity(self):

        """Builds the pattern matrix for the 4-term similarity
        transformation"""

        if np.size(self.x) < 1:
            return

        # We drop this in term by term
        nRows = self.x.shape[0]
        self.pattern = np.zeros(( nRows, 2, 4 ))

        self.pattern[:,0,0] = 1.
        self.pattern[:,0,1] = self.x
        self.pattern[:,0,2] = self.y
        self.pattern[:,1,1] = self.y
        self.pattern[:,1,2] = 0.-self.x
        self.pattern[:,1,3] = 1.

    def evaluatePtheta(self):

        """Evaluates P.theta, returning the NxK array of evaluates"""

        if np.size(self.pars) < 1:
            return np.array([])

        return np.matmul(self.pattern, self.pars)

class Stack2x2(object):

    """Stack of N x 2 x 2 transformation matrices. Intended use:
    conversion between a stack of 2x2 matrices and vectors of the
    geometric parameters (sx, sy, rotation, skew) corresponding to the
    transformation.

    If the matrix stack is supplied, the geometric parameters are
    generated on initialization. 

    If the geometric parameters are supplied, the matrix stack is generated on
    initialization."""

    def __init__(self, Asup=np.array([]), \
                     sx=np.array([]), sy=np.array([]), \
                     rotDeg=np.array([]), skewDeg=np.array([]), \
                     Verbose=False):

        # control keywords
        self.Verbose = Verbose

        self.Asup = np.copy(Asup)
        self.A = np.array([])
        self.parseInputStack()

        # Now the parameters in human-readable form...
        self.sx = copyAsVec(sx)
        self.sy = copyAsVec(sy)
        self.rotDeg = copyAsVec(rotDeg)
        self.skewDeg = copyAsVec(skewDeg)

        # ... and the inverse matrix
        self.AINV = np.array([])

        # Convert the matrix stack to human readable params or from
        # human readable params, depending on what was sent in
        if np.size(self.A) > 0:
            self.parsFromStack()
        else:
            self.stackFromPars()

        self.invertA()

    def parseInputStack(self):

        """Restricted parsing of the input stack. Must be N x 2 x 2 or
        can be made N x 2 x 2."""

        # if no input stack was given then silently return (e.g. if we
        # supplied human-readable parameters to convert to a matrix
        # stack)
        if np.size(self.Asup) < 1:
            return

        nDim = np.size(np.shape(self.Asup))
        if nDim < 2 or nDim > 3:
            if self.Verbose:
                print("Stack2x2.parseInputStack WARN: input not 2D or 3D", \
                          np.shape(self.Asup))
            return
        
        # This object is specialized to N x 2 x 2 arrays.
        if self.Asup.shape[-1] != 2 or self.Asup.shape[-2] != 2:
            if self.Verbose:
                print("Stack2x2.parseInputStack WARN: input not [ * x 2 x 2]", \
                          np.shape(self.Asup))
            return

        # if we got here then we must have either an N x 2 x 2 or a 2
        # x 2 array. Work accordingly.
        if nDim == 3:
            self.A = np.copy(self.Asup)
            return
        
        # The only way we reach this line is if we have a 2 x 2
        # matrix. So, convert it.
        self.A = self.Asup[np.newaxis, :, :]


    def allParsArePresent(self):

        """Utility - returns True if ALL of sx, sy, theta, skew have
        >0 rows and are all the same size. Otherwise returns False.

        """

        # There must be nonzero rows.
        nRows = np.size(self.sx)
        if nRows < 1:
            return False

        # The number of rows of all of the quantities must match.
        if np.size(self.sy) != nRows or \
                np.size(self.rotDeg) != nRows or \
                np.size(self.skewDeg) != nRows:
            return False

        return True

    def parsFromStack(self):

        """Interprets the N x 2 x 2 stack into transformation
        parameters"""

        if np.size(self.A) < 1:
            return

        # views to try to mitigate typos. Each plane of "A" is {b,c,e,f}
        b = self.A[:,0,0]
        c = self.A[:,0,1]
        e = self.A[:,1,0]
        f = self.A[:,1,1]

        self.sx = np.sqrt(b**2 + e**2)
        self.sy = np.sqrt(c**2 + f**2)
        
        # the arctan addition formula has limited validity. Do this
        # the quadrant-correct way...
        arctanCF = np.arctan2(c,f)
        arctanEB = np.arctan2(e,b)
        self.rotDeg = 0.5*np.degrees(arctanCF - arctanEB)
        self.skewDeg =    np.degrees(arctanCF + arctanEB)

        # In testing the enforcement of the parameter convention seems
        # to work well. Call it here.
        self.enforceParsConvention()

    def enforceParsConvention(self):

        """The intuitive parameters sx, sy, theta, skew are
        ambiguous. This method enforces a convention (prefer sx < 0 to
        |skew| > 45 degrees"""

        # Nothing to do if there are no parameters
        if np.size(self.sx) < 1:
            return

        # Skew angle high extreme
        bHiSkew = self.skewDeg > 90.
        self.skewDeg[bHiSkew] -= 180.
        self.rotDeg[bHiSkew] += 90.
        self.sx[bHiSkew] *= -1.

        # skew angle low extreme
        bLoSkew = self.skewDeg < -90.
        self.skewDeg[bLoSkew] += 180.
        self.rotDeg[bLoSkew] -= 90.
        self.sx[bLoSkew] *= -1.

    def invertA(self):

        """Find the matrix inverse of the stack"""

        self.AINV = np.linalg.inv(self.A)

    def stackFromPars(self):

        """Produces the stack of transformations from the input
        params"""

        # Exit gracefully if the parameters are not set or have
        # different row lengths
        if not self.allParsArePresent():
            return

        # convenience views again
        radX = np.radians(self.rotDeg - 0.5*self.skewDeg)
        radY = np.radians(self.rotDeg + 0.5*self.skewDeg)
        ssx = self.sx 
        ssy = self.sy

        self.A = np.zeros((np.size(self.sx),2,2))
        self.A[:,0,0] = ssx * np.cos(radX)
        self.A[:,0,1] = ssy * np.sin(radY)
        self.A[:,1,0] = 0.-ssx * np.sin(radX)
        self.A[:,1,1] = ssy * np.cos(radY)
      

### Utility class to generate XY points with different covariance
### matrices

class CovarsNx2x2(object):

    """Given a set of covariances as an Nx2x2 stack, or as the
    individual components (as might be read from an astrometric
    catalog), populates all the following forms: {covariance stack},
    {x-y components}, {major and minor axes and rotation angles}. Any
    of the three forms can be supplied: the stack is populated in the
    following order of preference: {covars, xy components,
    abtheta}. Computes various useful intermediate attributes that are
    useful when plotting: my coverrplot uses this fairly extensively.

    Also contains methods to generate datapoints described by the
    covariance stack. 

    Initialization arguments, all optional:

    covars = N x 2 x 2 stack of covariances. 

    --- If supplying the covariances in x, y, and correlations:

    stdx = N-element sqrt(Var) in xx 

    stdy = N-element sqrt(Var) in yy
        
    corrxy = N-element xy correlation coefficients

    --- If supplying the covariances as a, b, theta:

    majors = N-element array of major axis lengths [If supplying axes]

    minors = N-element array of minor axis lengths

    rotDegs = N-element array of rotation angles

    --- arguments for generating data follow ---

    nPts = number of points to generate
    
    rotDeg = scalar, typical rotation angle

    aLo, aHi = bounds on the major and minor axes
    
    ratLo, ratHi = bounds on the minor:major axis ratios

    genStripe = The covariances of the back half of the generated
                sample are flipped in the x-axis

    stripeFrac = fraction of the sample that will be the `special`
                 set. Default 0.5 .

    stripeCovRatio = ratio of the axis-lengths for the second and
                     first stripe

    """


    def __init__(self, covars=np.array([]), \
                     stdx=np.array([]), stdy=np.array([]), \
                     corrxy=np.array([]), \
                     rotDegs=np.array([]), \
                     majors=np.array([]), \
                     minors=np.array([]), \
                     nPts=100, rotDeg=30., \
                     aLo=1.0, aHi=1.0, \
                     ratLo=0.1, ratHi=0.3, \
                     # ratLo=1., ratHi=1., \
                     genStripe=True, \
                     stripeFrac=0.5, \
                     stripeCovRatio=1.):

        # The covariance stack (which could be input)
        self.covars = np.copy(covars)

        # Another form - the coord-aligned components of the stack
        self.stdx = np.copy(stdx)
        self.stdy = np.copy(stdy)
        self.corrxy = np.copy(corrxy)

        # Or, the covariances could be supplied as major, minor axes
        # and rotation angles
        self.majors = np.copy(majors)
        self.minors = np.copy(minors)
        self.rotDegs = np.copy(rotDegs)

        # Quantities we'll need when generating synthetic data
        self.nPts = nPts
        self.rotDeg = rotDeg
        self.aLo = aLo
        self.aHi = aHi
        self.ratLo = ratLo
        self.ratHi = ratHi

        # Options for some particular patterns can come here.
        self.genStripe = genStripe
        self.stripeFrac = np.clip(stripeFrac, 0., 1.)

        #print("INFO: stripeFrac", self.stripeFrac)

        # ratio between the axis lengths of the two stripes
        self.stripeCovRatio = stripeCovRatio

        # Initialize some internal variables that will be useful for
        # anything that needs abtheta:
        self.VV = np.array([]) # the diagonal covar matrix
        self.RR = np.array([]) # the rotation (+ skew?) matrix
        self.TT = np.array([]) # the transformation matrix 

        # Populate the covariance stack from inputs if any were given
        self.populateCovarsFromInputs()

        # Override npts with the size of the input stack
        if np.size(self.covars) > 0:
            self.nPts = self.covars.shape[0]

        # The sample of deltas (about 0,0)
        self.deltaTransf = np.array([])

        # Labels for the planes. Useful if plotting or outputting, but
        # will remain unused in most applications. This is initialized
        # so that we know what to call this attribute if we decide
        # elsewhere that we do need it after all.
        self.planeLabels = []

    def populateCovarsFromInputs(self):

        """Populates the covariance stack from input arguments. In
        order of preference: covars --> xycomp --> abtheta"""

        # if the covariances already have nonzero shape, override nPts
        # with their leading dimension
        if np.size(self.covars) > 0:

            # if passed a plane, turn into a 1x2x2 array:
            if np.size(np.shape(self.covars)) == 2:
                self.covars = self.covars[np.newaxis, :, :]

            self.nPts = np.shape(self.covars)[0]

            # populate the xy components
            self.populateXYcomponents()

        # Or, if we have no covariance but we DO have the XY
        # components, build it that way. covStackFromXY checks that
        # the arrays are nonzero so we don't need to do it here)
        else:
            self.covStackFromXY()  

        # If we still don't have the covariance stack yet, but we do
        # have major axes, then populate the covariance stack from the
        # abtheta form.
        if np.size(self.covars) < 1 and np.size(self.majors) > 0:
            self.covarFromABtheta()

    def populateXYcomponents(self):

        """Populates the X, Y covariance vectors from the covariance
        stack"""

        if np.size(self.covars) < 1:
            return

        # if the covariance is already set, then we just read off the
        # three components. 
        self.stdx = np.sqrt(self.covars[:,0,0])
        self.stdy = np.sqrt(self.covars[:,1,1])
        self.corrxy = self.covars[:,0,1] / (self.stdx * self.stdy)

    def covStackFromXY(self):

        """Populates the covariance stack from the XY components"""

        nPts = np.size(self.stdx)
        if nPts < 1:
            return

        self.nPts = nPts
        self.covars = np.zeros((self.nPts, 2, 2))
        
        # Now populate the parts. The xx variance must always be
        # populated
        self.covars[:,0,0] = self.stdx**2

        # If the y-component is not given, duplicate the xx part
        if np.size(self.stdy) == self.nPts:
            self.covars[:,1,1] = self.stdy**2
        else:
            self.covars[:,1,1] = self.stdx**2

        # Populate the off-diagonal elements
        if np.size(self.corrxy) != self.nPts:
            return

        covxy = self.corrxy * self.stdx * self.stdy
        self.covars[:,0,1] = covxy
        self.covars[:,1,0] = covxy

    def eigensFromCovars(self):

        """Finds the eigenvalues, eigenvectors and angles from the
        covariance stack"""

        # Get the stacks of eigenvalues and eigenvectors
        w, v = np.linalg.eigh(self.covars)

        # identify the major and minor axes (squared)
        self.majors = w[:,1]
        self.minors = w[:,0]

        # the eigenvectors are already normalized. We'll keep them so
        # that we can use them in plots
        self.axMajors = v[:,:,1]
        self.axMinors = v[:,:,0]  # Not needed?
        
        # enforce a convention: if the major axis points in the -x
        # direction, flip both eigenvectors
        bNeg = self.axMajors[:,0] < 0
        self.axMajors[bNeg] *= -1.
        self.axMinors[bNeg] *= -1.

        # the rotation angle of the major axis
        self.rotDegs = np.degrees(np.arctan(\
                self.axMajors[:,1]/self.axMajors[:,0]))

        # Having done this, we can now generate the diagonal, rotation
        # and transformation matrix should we wish to generate samples
        # from the deltas.


    def genEigens(self):

        """Generates the eigenvectors of the diagonal covariance matrix"""

        ratios = np.random.uniform(self.ratLo, self.ratHi, self.nPts)

        self.majors = np.random.uniform(self.aLo, self.aHi, self.nPts)
        ratios = np.random.uniform(self.ratLo, self.ratHi, self.nPts)
        self.minors = self.majors * ratios

    def genRotns(self, stripe=True):

        """Generates the rotation angles for the transformation"""

        # 2020-06-12 currently that's only rotation
        self.rotDegs = np.repeat(self.rotDeg, self.nPts)
        
        # stipe the rotation angles? 
        if self.genStripe:
            iPartition = int(self.stripeFrac*np.size(self.rotDegs))
            self.rotDegs[iPartition::] *= -1.

            # Scale the covar axes of the stripe
            self.majors[iPartition::] *= self.stripeCovRatio 
            self.minors[iPartition::] *= self.stripeCovRatio

    def populateDiagCovar(self):

        """Populates diagonal matrix stack with major and minor axes"""

        self.VV = np.array([])

        nm = np.size(self.majors)
        if nm < 1:
            return

        self.VV = np.zeros(( nm, 2, 2 ))
        self.VV[:,0,0] = self.asVector(self.majors)

        if np.size(self.minors) == np.size(self.majors):
            self.VV[:,1,1] = self.asVector(self.minors)
        else:
            self.VV[:,1,1] = self.VV[:,0,0]

    def populateRotationMatrix(self, rotateAxes=False):

        """Populates rotation matrix stack using rotations.

        rotateAxes = rotate the axes instead of the points?"""

        self.RR = np.array([])
        nR = np.size(self.rotDegs)
        nMaj = np.size(self.majors)
        if nR < 1:
            # If we DO have major array, use the identity matrix stack
            if nMaj > 0:
                i2 = np.eye(2, dtype='double')
                self.RR = np.repeat(i2[np.newaxis,:,:], nMaj, axis=0)
            return

        cc = np.cos(np.radians(self.rotDegs))
        ss = np.sin(np.radians(self.rotDegs))
        sgn = 1.
        if not rotateAxes:
            sgn = -1.

        self.RR = np.zeros(( nR, 2, 2 ))
        self.RR[:,0,0] = cc
        self.RR[:,1,1] = cc
        self.RR[:,0,1] = sgn * ss
        self.RR[:,1,0] = 0. - sgn * ss
        
    def populateTransformation(self):

        """Populates the transformation matrix by doing RR.VV
        plane-by-plane"""

        self.TT = np.matmul(self.RR, self.VV)

    def asVector(self, x=np.array([])):

        """Returns a copy of the input object if a scalar, and a
        reference to the input if already an array"""

        if np.isscalar(x):
            return np.array([x])
        return x
    
    def populateCovarStack(self):

        """Populates the stack of covariance matrices"""

        self.covars = AVAt(self.RR, self.VV)

    def generateCovarStack(self):

        """Wrapper that generates a stack of transformation
        matrices"""

        self.genEigens()
        self.genRotns()
        self.covarFromABtheta()
        #self.populateDiagCovar()
        #self.populateRotationMatrix(rotateAxes=False)
        #self.populateTransformation()
        #self.populateCovarStack()
        #self.populateXYcomponents()

    def covarFromABtheta(self):

        """If the a, b, theta components have been populated, use them
        to populate the covariance stack"""

        if np.size(self.majors) < 1:
            return

        self.populateDiagCovar()
        self.populateRotationMatrix(rotateAxes=False)
        self.populateTransformation()
        self.populateCovarStack()
        self.populateXYcomponents()
        

    def populateTransfsFromCovar(self):

        """Wrapper - populates the transformation matrix from the
        covariance stack. Useful if we want to draw samples from a set
        of covariance stacks"""

        if np.size(self.covars) < 1:
            return

        # If we haven't already got the rotation angles from the
        # covariance matrices, get them!
        if np.size(self.rotDegs) < 1:
            self.eigensFromCovars()

        # Now we can populate the other pieces
        self.populateDiagCovar()
        self.populateRotationMatrix(rotateAxes=False)
        self.populateTransformation()

    def generateSamples(self):

        """Generate samples from the distributions"""

        # Creates [nPts, 2] array

        # this whole thing is 2x2 so we'll do it piece by piece
        xr = np.random.normal(size=self.nPts)
        yr = np.random.normal(size=self.nPts)
        xxr = np.vstack(( xr, yr )).T[:,:,np.newaxis]

        self.deltaTransf = np.matmul(self.TT, xxr)[:,:,0].T
        #self.deltaTransf = np.dot(self.TT, xxr)

        # self.deltaTransf = np.einsum('ij,ik->ijk', self.TT, xxr)

    def showDeltas(self, figNum=1):

        """Utility: scatterplots the deltas we have generated"""
        
        if np.size(self.deltaTransf) < 1:
            return

        dx = self.deltaTransf[0]
        dy = self.deltaTransf[1]

        fig = plt.figure(figNum, figsize=(7,6))
        fig.clf()
        ax1 = fig.add_subplot(111)

        dumScatt = ax1.scatter(dx, dy, s=1, c=self.rotDegs, \
                                   cmap='inferno', zorder=5)
        
        ax1.set_xlabel(r'$\Delta X$')
        ax1.set_ylabel(r'$\Delta Y$')

        # enforce uniform axes
        dm = np.max(np.abs(np.hstack(( dx, dy ))))
        ax1.set_xlim(-dm, dm)
        ax1.set_ylim(-dm, dm)

        ax1.set_title('Deltas before shifting')

        ax1.grid(which='both', visible=True, alpha=0.5, zorder=1)

        cDum = fig.colorbar(dumScatt)

### SOme generically useful (and possibly unused) methods follow

def diagStack2x2(ul=np.array([]), lr=np.array([])):

    """Given a vector of upper-left and lower-right entries, construct
    an N x 2 x 2 diagonal matrix stack"""

    nUL = np.size(ul)
    if nUL < 1:
        return np.array([])

    # If the lower-right entry is not the same size as the upper-left,
    # ignore it and create identical lower-right as upper-left
    if np.size(lr) != nUL:
        lr = np.copy(ul)
    
    # gracefully handle scalars
    UL = copyAsVec(ul)
    LR = copyAsVec(lr)

    VV = np.zeros((nUL, 2, 2))
    VV[:,0,0] = UL
    VV[:,1,1] = LR

    return VV

def rotStack2x2(rotDeg=np.array([]), rotateAxes=False):

    """Given a vector of rotation angles, generates a stack of
    rotation matrices

    rotDeg = scalar or array of counter-clockwise rotation angles in degrees

    rotateAxes = rotates the axes about the points (if false, rotates the points)
    """

    nRot = np.size(rotDeg)
    if nRot < 1:
        return np.array([])
    
    # handle scalar input
    rotVec = copyAsVec(rotDeg)

    # Some pieces
    cc = np.cos(np.radians(rotDeg))
    ss = np.sin(np.radians(rotDeg))
    sgn = 1.
    if not rotateAxes:
        sgn = -1.

    RR = np.zeros(( nRot, 2, 2 ))
    RR[:,0,0] = cc
    RR[:,1,1] = cc
    RR[:,0,1] = sgn * ss
    RR[:,1,0] = 0. - sgn * ss
    
    return RR

### Utility - covariant error plot

def coverrplot(x=np.array([]), y=np.array([]), \
                   covars=None, \
                   errx=np.array([]), \
                   erry=np.array([]), \
                   corrxy=np.array([]), \
                   errSF = 1., \
                   showMajors = True, \
                   showMinors = True, \
                   showEllipses = True, \
                   shadeEllipses = True, \
                   colorMajors = 'c', \
                   colorMinors = 'c', \
                   edgecolorEllipse = 'c', \
                   alphaEllipse = 0.5, \
                   cmapEllipse = 'inferno', \
                   showColorbarEllipse = True, \
                   labelColorbarEllipse = r'$\theta$ ($^\circ$)', \
                   crossStyle=True, \
                   lwMajor = 0.5, \
                   lwMinor=0.3, \
                   showCaps=True, \
                   capScale=0.1, \
                   enforceUniformAxes=True, \
                   ax=None, fig=None, figNum=1, \
                   xLabel='', yLabel='', adjustMargins=True):

    """Utility - given points with covariant uncertainties, plot
    them. Arguments:

    x, y = positions
    
    covars = Nx2x2 stack of uncertainty covariances

    errx = std devns in x (ignored if covars set)

    erry = std devns in y (ignored if covars set)

    corrxy = correlation coefficients for errors in xy (ignored if
             covars set)

    errSF = scale factor to multiply the uncertainties for display

    showMajors = draw major axes?

    showMinors = draw minor axes?

    showEllipses = draw ellipses? 

    shadeEllipses = shade the ellipses by an array value?

    colorMajors, colorMinors = colors for the major and minor axes

    edgecolorEllipse = edgecolor for the ellipse edge (by default the
                       ellipse is color coded by the position angle of
                       its major axis)

    alphaEllipse = transparency for ellipses

    cmapEllipse = colormap use for ellipse shading

    crossStyle = plot the major and minor axes as crosses rather than
                 half-width vectors

    lwMajor = if crossStyle, linewidth for major axis and its endcap

    lwMinor = if crossStyle, linewidth for minor axis and its endcap

    showCaps = if crossStyle, show the endcaps for the crosses?

    capScale = fraction of major/minor axis length for the endcaps

    enforceUniformAxes = ensure the data ranges of the two axes are
                         the same (otherwise the major and minor axes
                         will not be perpendicular in the plot)
                      
    ax = axis on which to draw the plot

    fig = figure in which to draw the axis (needed for the colorbar)

    figNum = if we are making a new figure, the figure number

    xLabel = string for x-axis label. Ignored if zero length

    yLabel = string for y-axis label. Ignored if zero length

    adjustMargins - set the figure margins to reveal the labels"""

    if np.size(x) < 1:
        return

    if np.size(y) != np.size(x):
        return

    # construct the covariance object if not given
    if covars is None:
        covars = CovarsNx2x2(stdx=errx, stdy=erry, corrxy=corrxy)
        if covars.nPts < 1:
            return

        covars.eigensFromCovars()
        covars.populateTransfsFromCovar()

    # 2023-06-20 some condition-trapping depending on what was passed
    # in. It may be better to explicitly do this by checking if the
    # input covariance object is an instance of the covariance object,
    # or instead is a numpy array.
    if not hasattr(covars, 'axMajors'):
        covars.eigensFromCovars()
        # covars.populateTransfsFromCovar()
        
    # Expects to be given an axis, but generates a new figure if none
    # is passed.
    if not fig:
        fig = plt.figure(figNum, figsize=(5,4))
        fig.clf()

    if not ax:
        ax = fig.add_subplot(111)

    # A few convenience views. (covars.axMajors are the normalized
    # major eigenvector of each plane of the covariance stack,
    # covars.majors are the major eigenvalues. Similar for the minor
    # axes.)

    # 2023-06-20 take the sqrt() of the covars.majors and
    # covars.minors when used to plot. Remember those are variances.
    
    xMajors = covars.axMajors[:,0]*errSF*covars.majors**0.5
    yMajors = covars.axMajors[:,1]*errSF*covars.majors**0.5

    xMinors = covars.axMinors[:,0]*errSF*covars.minors**0.5
    yMinors = covars.axMinors[:,1]*errSF*covars.minors**0.5

    # For the moment, we use the covariance-stacks to plot. Later we
    # will give this method the ability to do its own covar
    # interpretation.
    if showMajors:
        if not crossStyle:
            dumMaj = ax.quiver(x,y, xMajors, yMajors, zorder=6, \
                                   color=colorMajors, \
                                   units='xy', angles='xy', scale_units='xy', \
                                   scale=1., \
                                   width=0.05*np.median(xMajors), headwidth=2)
        else:
            lcMaj = LineCollection(lineSetFromVectors(x, xMajors, y, yMajors), \
                                       zorder=6, color=colorMajors, \
                                       lw=lwMajor)
            ax.add_collection(lcMaj)

            # try the endcaps
            if showCaps:
                capsMaj = endcapsFromVectors(x,xMajors,y,yMajors, capScale)
                lcCapsMaj = LineCollection(capsMaj, zorder=6, \
                                               color=colorMajors, \
                                               lw=lwMajor)
                ax.add_collection(lcCapsMaj)

    if showMinors:
        if not crossStyle:
            dumMin = ax.quiver(x,y, xMinors, yMinors, zorder=6, \
                                   color=colorMinors, \
                                   units='xy', angles='xy', scale_units='xy', \
                                   scale=1., \
                                   width=0.05*np.median(xMajors), headwidth=2)
        else:
            lcMin=LineCollection( lineSetFromVectors(x,xMinors,y,yMinors), \
                                      zorder=6, color=colorMinors, \
                                      lw=lwMinor)
            ax.add_collection(lcMin)

            # show the endcaps
            if showCaps:
                capsMin = endcapsFromVectors(x, xMinors, y, yMinors, capScale)
                lcCapsMin = LineCollection(capsMin, zorder=6, \
                                               color=colorMinors, \
                                               lw=lwMinor)
                ax.add_collection(lcCapsMin)

    # Do the ellipse plot (Currently color-coded by rotation
    # angle. The choice of array to use as a color-coding could be
    # passed in as an argument.)
    if showEllipses:
        # (EllipseCollection wants the full widths not the half widths)
        xy = np.column_stack(( x, y ))

        ec = EllipseCollection(covars.majors**0.5*errSF*2., \
                                   covars.minors**0.5*errSF*2., \
                                   covars.rotDegs, \
                                   units='xy', offsets=xy, \
                                   transOffset=ax.transData, \
                                   alpha=alphaEllipse, \
                                   edgecolor=edgecolorEllipse, \
                                   cmap=cmapEllipse, \
                                   zorder=5)
        if shadeEllipses:
            ec.set_array(covars.rotDegs)

        ax.add_collection(ec)
    
        if showColorbarEllipse and shadeEllipses:
            cbar = fig.colorbar(ec)
            cbar.set_label(labelColorbarEllipse)
            
    # Ensure the axis autoscales with the data
    ax.autoscale(enable=True, axis='both')

    # enforce uniform axes?
    if enforceUniformAxes:
        unifAxisLengths(ax)
        ax.set_aspect('equal')

    # label the axes?
    if len(xLabel) > 0:
        ax.set_xlabel(xLabel)
    if len(yLabel) > 0:
        ax.set_ylabel(yLabel)

    if adjustMargins:
        fig.subplots_adjust(left=0.15, bottom=0.15)

def lineSetFromVectors(x=np.array([]), dx=np.array([]), \
                           y=np.array([]), dy=np.array([]) ):

    """Utility - returns an [Nx2x2] set of lines corresponding to
    point +/- vector, for use setting up a line collection in
    coverrplot."""

    if np.size(x) < 1:
        return np.array([])

    lms = np.zeros((np.size(x), 2, 2))
    lms[:,0,0] = x-dx
    lms[:,0,1] = y-dy

    lms[:,1,0] = x+dx
    lms[:,1,1] = y+dy

    return lms

def endcapsFromVectors(x=np.array([]), dx=np.array([]), \
                           y=np.array([]), dy=np.array([]), \
                           capscale=0.1):

    """Given centroids and vectors, returns a set of endcap lines for
    use as a line collection."""

    # Same ideas as lineSetFromVectors() except this time we focus on
    # the ends of the lines
    if np.size(x) < 1:
        return np.array([])

    # To construct the deltas, simply rotate the dx, dy vector by 90
    # degrees. In other words:
    dxCap = dy*capscale
    dyCap = -dx*capscale

    # We build the caps at either end of the vector then stack them
    emsLo = np.zeros((np.size(x),2,2))
    emsHi = np.copy(emsLo) 
    emsLo[:,0,0] = x-dx - dxCap
    emsLo[:,0,1] = y-dy - dyCap
    emsLo[:,1,0] = x-dx + dxCap
    emsLo[:,1,1] = y-dy + dyCap

    emsHi[:,0,0] = x+dx - dxCap
    emsHi[:,0,1] = y+dy - dyCap
    emsHi[:,1,0] = x+dx + dxCap
    emsHi[:,1,1] = y+dy + dyCap
    
    return np.vstack(( emsLo, emsHi ))

def unifAxisLengths(ax=None):

    """Utility - ensure axis has the same axis lengths"""

    # Used by coverrplot

    if not ax:
        return

    xax = np.copy(ax.get_xlim())
    yax = np.copy(ax.get_ylim())
        
    xc = 0.5*(xax[0] + xax[1])
    yc = 0.5*(yax[0] + yax[1])

    xdelt = xax - xc
    ydelt = yax - yc

    # maximum abs delta
    dmax = np.max(np.abs( np.hstack(( xdelt, ydelt )) ))

    xnew = xc + dmax*np.array([-1., 1.])
    ynew = yc + dmax*np.array([-1., 1.])

    # handle the sign
    if xdelt[0] > 0:
        xnew *= -1.

    if ydelt[0] > 0:
        ynew *= -1.

    ax.set_xlim(xnew)
    ax.set_ylim(ynew)

#### Monte Carlo framework for NE Fitting comes here.

class CovarsMxM(object):

    """Handy methods for partitioning an MxM covariance matrix"""

    # Requires CovarsNx2x2 class defined above.

    # 2020-07-20 started out as a method within SimResultsStack, but I
    # discovered this is going to be useful to package the
    # covariance-pairs in the standard fit anyway.

    def __init__(self, cov=np.array([]), labels=[], pairsOnInit=True):

        # MxM covariance and labels per quantity
        self.cov = cov
        self.labels = labels[:]
        self.nPars = np.shape(self.cov)[0]
        self.ensureLabelsOK()

        # CovarsNx2x2 object for the covariance pairs
        self.covPairs = None

        if pairsOnInit:
            self.populateCovPairs()

    def ensureLabelsOK(self):

        """Ensures the labels have the same length as the dimension of
        the covariance matrix"""

        if np.size(self.cov) < 2:
            return

        self.nPars = np.shape(self.cov)[0]
        if self.nPars != len(self.labels):
            self.labels = ['s%i' % (i) for i in range(self.nPars)]

    def populateCovPairs(self):

        """Populates the covariance pairs"""

        # We keep the array local and make the object the
        # instance-level quantity.
        
        # Exit gracefully if the parameters are not yet set
        nPars = np.shape(self.cov)[0]
        if nPars < 1:
            return

        nPlanes = int(nPars*(nPars+1)/2)        
        covnx2x2 = np.zeros((nPlanes, 2, 2))
        keyPairs = []
        
        if np.size(covnx2x2) < 1:
            self.initializeCovPairs()

        # List of iPar, jPar created here because will be useful to
        # construct the pair-plot later.
        iPars = []
        jPars = []

        thisPlane = 0
        for iPar in range(nPars):
            for jPar in range(iPar, nPars):
                sKey = r'%s_vs_%s' % \
                    (self.labels[iPar], self.labels[jPar])
                keyPairs.append(sKey)
                
                # Populate the appropriate plane
                covnx2x2[thisPlane,0,0] = self.cov[iPar, iPar]
                covnx2x2[thisPlane,1,1] = self.cov[jPar, jPar]
                if iPar != jPar:
                    covnx2x2[thisPlane,0,1] = self.cov[iPar, jPar]
                    covnx2x2[thisPlane,1,0] = self.cov[jPar, iPar]

                iPars.append(iPar)
                jPars.append(jPar)

                # increment the plane
                thisPlane += 1

        # Now that we've populated the Nx2x2 covariance array and the
        # labels, create the covPairs Object
        # Now generate the covariance stack
        self.covPairs = CovarsNx2x2(covnx2x2)
        self.covPairs.eigensFromCovars()

        # attach the plane labels as an attribute to the stack object
        self.covPairs.planeLabels = keyPairs      

        # add a nonstandard pair of arguments to the object for plotting
        self.covPairs.iParsPlot = iPars
        self.covPairs.jParsPlot = jPars

class NormWithMonteCarlo(object):

    """Performs a normal equations-based fit to data, whether supplied
    or generated. Includes various ways to do Monte Carlo on the
    results. Optional inputs:

    x, y = source coords

    xi, eta = target coords

    stdxi, stdeta = uncertainties in the target coords

    corrxieta = correlation coefficient for uncertainty in target coords

    xref, yref = reference point in source coords

    --- simulation variables - parameters 

    simNpts = number of objects to simulate

    simSx, simSy = "true" scale factors

    simRotDeg, simSkewDeg = "true" rotation and skew angles, degrees

    simXiRef, simEtaRef = "true" reference xi, eta

    simParsVec = parameters as [a,b,c,d,e,f] or [a,b,c,d] 

    simXmin, simYmin = minmax source X coords

    simYmin, simYmax = minmax source Y coords

    simXcen, simYcen = offset for the source X, Y coords

    -- If simulating a gaussian field rather than uniform

    simMakeGauss = simulate a gaussian field?

    simGauMajor = major axis of gaussian component

    simGauMinor = minor axis of gaussian component
    
    simGauTheta = position angle of gaussian component

    --- simulation variables - covariances 

    simAlo, simAhi = min, max semimajor axes of Xi, Eta covariance
                   matrices
    
    simRotCov = rotation angle (degrees, ccw) of canned covariance
                   major axis

    simBAlo    =   minimum covariance axis ratio (minor:major)

    simBAhi    =   maximum covariance axis ratio (minor:major)

    genStripe = flip the covariances for the back half of the samples
                in the xi-axis
   
    stripeFrac = fraction of objects assigned to the second stripe

    stripeCovRatio = covariance axis-length ratio for the second to
                     first stripe

    posnSortCol = if nonzero length, the attribute on which positions
                  are to be sorted (Useful if preparing two
                  populations with differing covariance properties.)

    --- Generating outliers -----

    fOutly = fraction of objects to be outliers

    rOutly_min_arcsec = minimum separation for outliers (arcsec, xi plane)

    rOutly_max_arcsec = maximum separation for outliers (arcsec, xi plane)

    --- Choices for the simulations -----

    nTrials = number of Monte Carlo trials to do

    resetPositions = populate the positions each simulation?

    doFewWeightings = do diagonal and unweighted monte carlo as well?

    fNonparam = fraction of original data size to use in nonparametric
                resampling (set to 0.5 for half-sample
                bootstrap). Will be clipped to 1.

    doNonparam = Bootstrap trials will be non-parametric if True

    paramUsesTrueTransf = parametric monte carlo uses the ``true''
                        transformation parameters when producing new
                        position-sets. If False, will use the
                        estimated parameters from the first round of
                        fitting. Ignored if nonparametric bootstraps
                        or if the parametric bootstraps do not reset
                        the positions for each trial.

    whichFittedTransf = 'full' - if using one of the fitted
                         transformations rather than the true
                         transformation, which one do we use?

    --- Choices for the fitting ---

    fitChoice = string, choice of model fitting between the frames

    flipx = flip the x-coordinate in the fit? Ignored if similarity
            transformation is used.

    --- Misc control variables ---

    Verbose = Print some screen output?

    parFile = input parameter file (parameters read from this file
              supersede arguments passed on the command line.)

    """

    def __init__(self, x=np.array([]), y=np.array([]), \
                     xi=np.array([]), eta=np.array([]), \
                     stdxi=np.array([]), stdeta=np.array([]), \
                     corrxieta=np.array([]), xref=0., yref=0., \
                     simNpts = 20, \
                     simSx = -5.0e-4, simSy = 4.0e-4, \
                     simRotDeg = 30., simSkewDeg = 5., \
                     simXiRef = 0.05, simEtaRef = -0.06, \
                     simParsVec = np.array([]), \
                     simXmin = 0., simXmax = 500., \
                     simYmin = 0., simYmax = 500., \
                     simXcen = 0., simYcen = 0., \
                     simMakeGauss = False, \
                     simGauMajor = 100., simGauMinor=60., \
                     simGauTheta = -15., \
                     simAlo = 1.0e-4, simAhi=2.0e-3, \
                 simBAlo = 0.1, simBAhi = 0.3, simRotCov=30., \
                     genStripe=True, \
                     stripeFrac=0.5, \
                     stripeCovRatio=1., \
                     posnSortCol='', \
                 fOutly=0., rOutly_min_arcsec=0., \
                 rOutly_max_arcsec=1., \
                     nTrials = 3, \
                     resetPositions=False, \
                     doFewWeightings=True, \
                     fNonparam=1., \
                     fitChoice='6term', \
                     flipx=False, \
                     doNonparam=True, \
                     paramUsesTrueTransf=True, \
                     whichFittedTransf='full', \
                     Verbose=False, \
                     parFile=''):

        # Control variables
        self.Verbose = Verbose

        # dictionary of parameters we'd dump to or read from disk
        self.ddump = {}
        self.lkeys = self.ddump.keys()

        # coords, if given
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.xi = np.copy(xi)
        self.eta = np.copy(eta)
        self.stdxi = np.copy(stdxi)
        self.stdeta = np.copy(stdeta)
        self.corrxieta = np.copy(corrxieta)
        self.xRef = np.copy(xref)
        self.yRef = np.copy(yref)

        # For parametric monte carlo, the simulation parameters as
        # geometric params
        self.simSx = np.copy(simSx)
        self.simSy = np.copy(simSy)
        self.simRotDeg = np.copy(simRotDeg)
        self.simSkewDeg = np.copy(simSkewDeg)
        self.simXiRef = np.copy(simXiRef)
        self.simEtaRef = np.copy(simEtaRef)
        self.simNpts = int(np.copy(simNpts)) # scalars copy to array!

        # or, we might already have the 6-term vector [a,b,c,d,e,f]:
        self.simTheta = np.copy(simParsVec)

        # Field of view limits for X, Y (later, we could be clever and
        # get these from the input dataset)
        self.simXmin = np.copy(simXmin)
        self.simXmax = np.copy(simXmax)
        self.simYmin = np.copy(simYmin)
        self.simYmax = np.copy(simYmax)

        # Offset for the simulated points
        self.simXcen = float(np.copy(simXcen))
        self.simYcen = float(np.copy(simYcen))

        # Parameters for gaussian simulation (if we're doing that)
        self.simMakeGauss = simMakeGauss # True/False
        self.simGauMajor = simGauMajor
        self.simGauMinor = simGauMinor
        self.simGauTheta = simGauTheta

        # Variables for canned covariance matrices in xi, eta
        self.simAlo = simAlo
        self.simAhi = simAhi
        self.simRotCov = simRotCov
        self.simBAlo = simBAlo
        self.simBAhi = simBAhi
        
        # Variables to do with the special stripe (the back stripeFrac
        # of the samples)
        self.genStripe = genStripe
        self.stripeFrac = np.clip(stripeFrac, 0., 1.)
        self.stripeCovRatio = stripeCovRatio
        self.posnSortCol = posnSortCol[:] # column to sort positions

        # Outlier generation
        self.fOutly = fOutly
        self.rOutly_min_arcsec = rOutly_min_arcsec
        self.rOutly_max_arcsec = rOutly_max_arcsec
        
        # Number of trials we will be using
        self.nTrials = np.copy(nTrials)
        self.resetPositions = resetPositions

        # choice for the fitting 
        self.fitChoice = fitChoice[:]

        # Do we try a few different weightings in the monte carlo?
        self.doFewWeightings = doFewWeightings

        # Will the bootstrap trials be non-parametric?
        self.doNonparam = doNonparam
        
        # If doing parametric bootstraps, do we reset the positions
        # using the true parameters, or using the parameters
        # determined from the previous fit?
        self.paramUsesTrueTransf = paramUsesTrueTransf
        self.whichFittedTransf = whichFittedTransf[:]

        # Fraction of sample size to draw if doing non-parametric
        # bootstraps
        self.fNonparam = np.clip(fNonparam, 0.01, 1.)

        # filename stem for corner plot
        self.stemCornerFil = 'tmp_corner'
        self.stemSummStats = 'tmp_summ'
        self.tailSummStats = '.csv'

        # filename for output parameter dump if we're writing it
        self.filParamsOut = 'tmp_mcparams.txt'

        self.filParamsIn = parFile[:]
        if len(self.filParamsIn) > 0:
            self.readPars(self.filParamsIn)

        # Flip the (x-xref) coordinate when fitting? (Ignored if
        # 6-term transformation used)
        if self.fitChoice.find('6term') > -1:
            self.flipx = False
            
        # latex-friendly labels for the terms. Used for plotting and
        # possibly outputting to file.
        self.paramLabels = [r'$a$', r'$d$', r'$s_x$', r'$s_y$', \
                                r'$\theta$', r'$\beta$']

        if self.fitChoice.lower().find('similarity') > -1:
            self.paramLabels = [r'$a$', r'$d$', r'$s$', r'$\theta$']

        # There are three fitting objects. FitData holds the info and
        # fit for the actual data. FitUnperturbed holds the points,
        # covariances, etc. for the unperturbed samples. Finally,
        # FitSample holds the values actually used, which in the case
        # of parametric Monte Carlo, will be drawn by perturbing the
        # values of FitUnperturbed.
        #
        # This way, FitUnperturbed is not changed after first
        # populated. This also allows us to play with the covariances
        # in the FitSample object, for example changing the estimate
        # of the covariances after the perturbations have been done
        # (to simulate cases where the covariances are inaccurate).
        self.FitData = None
        self.FitUnperturbed = None
        self.FitSample = None
        
        self.FitResample = None  # for nonparametric resampling 

        # Covariance stack (will be used to draw perturbation samples
        # from the covariances if doing parametric monte carlo)
        self.CF = None

        # Some internal variables for sample-generation. 
        self.xRaw = np.array([])
        self.yRaw = np.array([])
        self.xiRaw = np.array([])
        self.etaRaw = np.array([])

        # Will the object be an outlier?
        self.isOutlier = np.array([])  # identifier...
        self.dxiOutlier = np.array([])
        
        # Internal variables: generated points (in practice I think
        # it'll be better to initialize the fit object and manipulate
        # it directly, but having the parameters here will keep things
        # clearer while writing this the first time through). It also
        # depends a bit on what kind of monte carlo we're doing:
        # nonparametric bootstrap (or jackknife) or parametric
        # bootstrap.
        self.xGen = np.array([])
        self.yGen = np.array([])
        self.xiGen = np.array([])
        self.etaGen = np.array([])

        # The parameters that were fit from the data, the tangent
        # point corresponding to these params, and the formal
        # covariance matrix
        self.parsData = np.array([])
        self.tpData = np.array([])
        self.formalCovData = np.array([])

        # Fitted parameters for the other weightings (if we want to
        # use them in the monte carlo)
        self.parsDataDiag = np.array([])
        self.parsDataUnif = np.array([])
        self.formalCovDataDiag = np.array([])
        self.formalCovDataUnif = np.array([])

        # Set of fitted parameters from the monte carlo trials
        self.stackTrials = None

        # 2020-06-21: We may also want to try several different types
        # of fit for the same sample. For ease of debugging later (and
        # NOT minimization of lines of code) I give the trial types
        # different named objects (rather than, say, dictionary
        # entries). So:
        self.stackTrialsDiag = None
        self.stackTrialsUnif = None

        # Are we doing non-parametric Monte Carlo? (Status attribute,
        # will be updated when doMonteCarlo is run.)
        self.bootsAreNonparam = True

        # if doing parametric bootstrapping, use the following array
        # to allow passing in a different transformation than the
        # 'true' transformation (e.g. simulating the investigator
        # doing parametric MC based on their first round of fitting)
        self.parsForParamBoots = np.array([])
        self.choiceForParamBoots = '6term'

        # If doing a 4-term transformation, we might want to ensure
        # the x coords are flipped even if the transformation doesn't
        # include a term that does this. The flipx argument
        # accommodates this.
        self.xSign = 1.
        if self.flipx:
            self.xSign = -1.

        # Some utility variables for plotting
        self.LLaxes = []
        self.axesCovarPairs = {}
        self.figCovarPairs = None

        # refactored into StackTrials object
        # 
        #self.parsTrials = np.array([])
        #self.tpTrials = np.array([]) # tangent points
        #self.nUseTrials = np.array([]) # number of objects fit

        # Transformation N x 2 x 2 stack for conversion to geometric
        # parameters
        #self.transfs2x2 = None
        #self.transfs2x2rev = None # going the other way 
        
    def transfParsAsVecs(self):

        """Translates the geometric simulation parameters into
        [a,b,c,d,e,f] vector"""

        Transf = Stack2x2(None, self.simSx, self.simSy, \
                              self.simRotDeg, self.simSkewDeg)
        self.simTheta = np.hstack(( self.simXiRef, Transf.A[0,0,:], \
                                        self.simEtaRef, Transf.A[0,1,:] ))

        # Here we determine the parameters to be used for parametric
        # monte carlo
        self.parsForParamBoots = np.copy(self.simTheta)
        self.choiceForParamBoots = '6term'

    def setSimRangesFromData(self):

        """If the input data are populated, get the relevant
        simulation quantities (like minmax X, npts) from the data"""

        if np.size(self.x) < 1:
            return

        # Bounds are rounded to the nearest 10
        self.simXmin = np.round(np.min(self.x), -1)
        self.simXmax = np.round(np.max(self.x), -1)
        self.simYmin = np.round(np.min(self.y), -1)
        self.simYmax = np.round(np.max(self.y), -1)
        self.simNpts = np.size(self.x)

    def generateRawXY(self):

        """Generates a set of unperturbed X, Y coords"""

        if self.simMakeGauss:
            self.generateXYgauss()

            #print("INFO - xRaw, yRaw:", \
            #          np.shape(self.xRaw), np.shape(self.yRaw))
            return

        # If we got here, then we do the rectangular field
        self.generateXYuniform()

    def generateXYuniform(self):

        """Generates raw X, Y from rectangular uniform distribution"""

        self.xRaw = np.random.uniform(self.simXmin, self.simXmax, \
                                          size=self.simNpts) + self.simXcen
        self.yRaw = np.random.uniform(self.simYmin, self.simYmax, \
                                          size=self.simNpts) + self.simYcen

        ## print("INFO::", self.simNpts, self.simYmin, self.simYmax, np.shape(self.simTheta))

    def generateXYgauss(self):

        """Generates raw X, raw Y from gaussian"""

        CG = CovarsNx2x2(nPts = 1, \
                             majors=self.simGauMajor, \
                             minors=self.simGauMinor, \
                             rotDegs=self.simGauTheta, \
                             genStripe=False)

        # This is a little ugly since CovarsNx2x2 expects one
        # covariance matrix per plane, rather than N objects all of
        # which use the same covariance matrix. Rather than trying to
        # alter CovarsNx2x2, let's just generate normal random
        # variables using the GC.covar[0] as the covariance matrix. As
        # a ``plus'', we can naturally handle a zeropoint offset too.

        vCen = np.array([self.simXcen, self.simYcen])
        cova = CG.covars[0]
        normPts = np.random.multivariate_normal(vCen, cova, self.simNpts)

        self.xRaw = normPts[:,0]
        self.yRaw = normPts[:,1]

    def sortRawPositions(self, sortCol=''):

        """Arg-sorts the raw points by raw x (useful when imposing
        systematics on the covariances). How this works: if the
        `stripe` argument in the Covars2D object is true, then the
        first half of the rows have a different covariance matrix
        (angle) than the second half. *This* routine argsorts the
        positions by sortCol, thus inserting a spatial correlation
        between covariance and location."""

        # Either use the input argument or the instance sort column,
        # ensuring both are consistent (i.e. passing the argument into
        # this method causes the instance-wide attribute to be
        # changed).
        if len(sortCol) < 1:
            sortCol = self.posnSortCol[:]
        else:
            self.posnSortCol = sortCol[:]

        # silently exit if there's no column to sort on
        if len(sortCol) < 1:
            return

        if not hasattr(self, sortCol):
            if self.Verbose:
                print("sortPositions WARN - attribute not populated: %s" \
                          % (sortCol))
            return

        vSort = getattr(self, sortCol)        
        if np.size(vSort) < 1:
            if self.Verbose:
                print("sortPositions WARN - column %s has zero length" \
                          % (sortCol))
            return

        lSor = np.argsort(vSort)

        if np.size(self.xRaw) > 0:
            self.xRaw = self.xRaw[lSor]
            self.yRaw = self.yRaw[lSor]

        # if the xi, eta raw have already been populated, sort them
        # too
        if np.size(self.xiRaw) > 0:
            self.xiRaw = self.xiRaw[lSor]
            self.etaRaw = self.etaRaw[lSor]

    def populateRawXiEta(self):

        """Populates simulated Xi, Eta by transforming the unperturbed
        X, Y to the Xi, Eta system using the 'true' transformation"""

        # The instance-variables "parsForParamBoots" and
        # "choiceForParamBoots" determine the parameters to use and
        # the fit choice.

        # Note that the true transformation will usually be 6-term
        # even if we are fitting with 4-term. So - don't pass the
        # fitChoice through to this object.

        #parsTransf = np.copy(self.simTheta)
        #transfChoice = '6term'

        PT = ptheta2d(self.xRaw-self.xRef, self.yRaw-self.yRef, \
                          pars=self.parsForParamBoots, \
                          fitChoice=self.choiceForParamBoots, \
                          xSign = self.xSign)

        aXiTrue = PT.evaluatePtheta()

        ## print("INFO:-", self.simTheta)

        self.xiRaw = aXiTrue[:,0]
        self.etaRaw = aXiTrue[:,1]

    def initOutliers(self):

        """Initialises the array of outliers"""

        if np.size(self.xiRaw) < 1:
            return

        nraw = self.xiRaw.size
        
        self.isOutlier = np.repeat(False, nraw)
        self.dxiOutlier = np.zeros((nraw, 2))
        
    def assignOutliers(self):

        """Assigns outliers"""
        
        # How many outliers?
        nOutly = int(self.xiRaw.size * self.fOutly)
        if nOutly < 1:
            return
        
        rng = np.random.default_rng()
        badinds = rng.integers(0, self.xiRaw.size, nOutly)
        self.isOutlier[badinds] = True

    def populateOutliers(self):

        """Populates outliers"""

        # Nothing to do if there are no outliers
        noutliers = np.int(np.sum(self.isOutlier))
        if noutliers < 1:
            return

        # generate deltas for the outliers
        outliers = Outliers2d(noutliers, \
                              self.rOutly_max_arcsec/3600., \
                              self.rOutly_min_arcsec/3600.)
        dxi, deta = outliers.unifInDisk()

        print("OUTLIERS INFO:", outliers.rInner, outliers.rOuter, outliers.nOutliers, dxi.min(), dxi.max() )
        
        self.dxiOutlier[self.isOutlier,0] = dxi
        self.dxiOutlier[self.isOutlier,1] = deta

    def setupOutliers(self):

        """One-liner to set up the outliers"""

        self.initOutliers()
        self.assignOutliers()
        self.populateOutliers()
        
    def populateCovarsFromSim(self):

        """Populates the covariances object with simulated parameters"""

        CN = CovarsNx2x2(nPts=self.simNpts, rotDeg=self.simRotCov, \
                             aLo=self.simAlo, aHi=self.simAhi, \
                         ratLo=self.simBAlo, ratHi = self.simBAhi, \
                             genStripe=self.genStripe, \
                             stripeCovRatio=self.stripeCovRatio, \
                             stripeFrac=self.stripeFrac)
        CN.generateCovarStack()

        self.CF = CovarsNx2x2(CN.covars)
        self.CF.populateTransfsFromCovar()

    def populateCovarsFromData(self):

        """Populates the covariances object from the data"""

        if np.size(self.stdxi) < 1:
            return

        # (note to self: None has nonzero size)
        self.CF = CovarsNx2x2(np.array([]), \
                                  self.stdxi, self.stdeta, self.corrxieta)
        self.CF.populateTransfsFromCovar()

    def populateUnperturbedFitObj(self):

        """Populates the unperturbed fit object for Monte Carlo
        simulations"""

        self.FitUnperturbed = FitNormEq(self.xRaw, self.yRaw, \
                                            self.xiRaw, self.etaRaw, \
                                            covars=self.CF.covars, \
                                            xRef=self.xRef, yRef=self.yRef, \
                                            fitChoice=self.fitChoice, \
                                            flipx=self.flipx, \
                                            runOnInit=False)

    def resetUnperturbedPositions(self):

        """Generates new unperturbed positions in-place"""
        
        # Notice that the covariance matrices are NOT touched by this
        # method. So the row-ordering of the covariance matrices is
        # still useful to do the covariance-striping.

        self.generateRawXY()
        self.populateRawXiEta()
        self.sortRawPositions() # apply position-striping if attribute set
        self.FitUnperturbed.x = self.xRaw
        self.FitUnperturbed.y = self.yRaw
        self.FitUnperturbed.xi = self.xiRaw
        self.FitUnperturbed.eta = self.etaRaw

    def populatePerturbedFitObj(self):

        """Populates the 'perturbed' fit object out of the unperturbed
        fit object"""
        
        # Initialize the perturbed fit object out of the unperturbed
        # fit object. NOTE: uncommenting the conditional breaks the
        # monte carlo (i.e. the fit parameters come out random) when
        # resetting positions, but NOT when perturbing from the
        # unperturbed.

        if self.FitSample is None:
            self.FitSample = FitNormEq(self.xRaw, self.yRaw, \
                                           self.xiRaw, self.etaRaw, \
                                           covars=self.CF.covars, \
                                           xRef=self.xRef, yRef=self.yRef, \
                                           fitChoice=self.fitChoice, \
                                           flipx=self.flipx, \
                                           runOnInit=False)

        # ensure the raw positions are populated (we could perturb
        # these too to simulate 'realistic' measurement uncertainties
        # in the DETX, DETY frames). We could do a covariance stack in
        # XY just like we do for xi, eta. TO BE ADDED.
        self.FitSample.x = self.FitUnperturbed.x
        self.FitSample.y = self.FitUnperturbed.y

        # Generate the perturbations in xi, eta
        self.CF.generateSamples()
        self.FitSample.xi  = self.FitUnperturbed.xi  + self.CF.deltaTransf[0]
        self.FitSample.eta = self.FitUnperturbed.eta + self.CF.deltaTransf[1]

        # Other steps (like messing with the covariance) could come
        # here.

    def makeSampleWeightsScalar(self, fitTemplate=None):

        """Changes the weights array in the FitSample object to be
        scalars. This returns a copy of the input template object."""

        if fitTemplate is None:
            return None

        # Get the covariance matrix in the FitSample object, find the
        # eigenvalues, and use the major axes to populate the new
        # weights in the fit object.
        FitSampleCopy = copy.deepcopy(fitTemplate)

        CV = CovarsNx2x2(FitSampleCopy.covars)
        CV.eigensFromCovars()

        # Create a new covariance object out of the major axes only
        CS = CovarsNx2x2(np.array([]), majors=np.copy(CV.majors))
        
        # Now update the perturbed fit object with the uniform weights
        FitSampleCopy.covars = CS.covars
        FitSampleCopy.weightsFromCovars()
        FitSampleCopy.initNE()

        return FitSampleCopy

    def makeSampleWeightsUniform(self, fitTemplate=None):

        """Creates a copy of the fitsample object, this time with
        uniform weighting. Takes the object passed in as a
        fitTemplate, makes a copy, and adjusts the weights. 

        The adjusted object is returned (rather than set at an
        instance level) since this may be called in different
        circumstances depending on what monte carlo we're doing."""

        if fitTemplate is None:
            return None

        FitSampleCopy = copy.deepcopy(fitTemplate)
        
        nCovs = FitSampleCopy.covars.shape[0]
        eyePlane = np.eye(2, dtype='double')
        eyeStack = np.repeat(eyePlane[np.newaxis,:,:], nCovs, axis=0)

        FitSampleCopy.covars = eyeStack # this might be redundant
        FitSampleCopy.W = eyeStack
        FitSampleCopy.initNE()

        return FitSampleCopy

    def drawNonparamSample(self):

        """Draws a sample for non-parametric bootstrap using options
        set at the instance level."""

        # Ensure the nonparametric bootstrap object is initialized
        # *with the full dataset* (we'll use an index array to select
        # the objects we want to use in the fitting
        if self.FitResample is None:
            if self.FitData is not None:
                self.FitResample = copy.deepcopy(self.FitData)
            else:
                return

        # Initialize the normal eq object in FitResample if not
        # already set
        if self.FitResample.NE is None:
            self.FitResample.initNE()

        # Draw the sample
        nAll = np.size(self.FitResample.x)
        nDraw = int(nAll * self.fNonparam)
        self.FitResample.NE.gPlanes = np.random.randint(0, nAll, size=nDraw)
        
    def fitPerturbed(self):

        """Does the fit for the perturbed object"""

        self.FitSample.performFit()

    def copyPerturbedSimToData(self):

        """Utility - copies the current simulation object into the
        data arrays"""

        # This is useful when using this class to generate the
        # hypothetical data as well as doing the simulations.

        self.x = np.copy(self.FitSample.x)
        self.y = np.copy(self.FitSample.y)
        self.xi = np.copy(self.FitSample.xi)
        self.eta = np.copy(self.FitSample.eta)
        self.stdxi = np.copy(self.CF.stdx)
        self.stdeta = np.copy(self.CF.stdy)
        self.corrxieta = np.copy(self.CF.corrxy)
        self.nPts = np.size(self.x)

    def fitData(self):

        """Initializes the fitting object for data. If we already have
        data, then the fit will be performed on initialization."""

        self.FitData = FitNormEq(self.x, self.y, self.xi, self.eta, \
                                     stdxi=self.stdxi, stdeta=self.stdeta, \
                                     corrxieta=self.corrxieta, \
                                     xRef=self.xRef, yRef=self.yRef, \
                                     fitChoice=self.fitChoice, \
                                     flipx=self.flipx, \
                                     invertHessian=True, Verbose=True)

        self.parsData = np.copy(self.FitData.NE.pars[:,0])
        self.tpData = np.copy(self.FitData.NE.xZero)
        self.formalCovData = np.copy(self.FitData.NE.formalCov)

        if not self.doFewWeightings:
            return

        # if we're trying multiple weightings, do the scalar and
        # uniform weightings.
        self.FitDataDiag = self.makeSampleWeightsScalar(self.FitData)
        self.FitDataUnif = self.makeSampleWeightsUniform(self.FitDataDiag)

        self.FitDataDiag.performFit()
        self.FitDataUnif.performFit()

        # Special objects for the fitted data with diagonal and
        # uniform weightings
        self.parsDataDiag = self.FitDataDiag.NE.pars[:,0]
        self.parsDataUnif = self.FitDataDiag.NE.pars[:,0]
        self.formalCovDataDiag = np.copy(self.FitDataDiag.NE.formalCov)
        self.formalCovDataUnif = np.copy(self.FitDataUnif.NE.formalCov)

    def accumulateTrialParams(self):

        """Abuts the trial parameters onto the master-array"""

        thesePars = np.copy(self.FitSample.NE.pars[:,0])
        thisTP = np.copy(self.FitSample.NE.xZero)
        nFitted = np.size(self.FitSample.NE.gPlanes)

        # It seems like there should be a more efficient way to do
        # this. This procedure will result in an Ntrials x nPars array

#        # If this is the first...
        if np.size(self.parsTrials) < 1:
            self.parsTrials = thesePars
            self.tpTrials = thisTP
            self.nUseTrials = nFitted
            return

        self.parsTrials = np.vstack(( self.parsTrials, thesePars ))
        self.tpTrials   = np.vstack(( self.tpTrials, thisTP ))
        self.nUseTrials = np.hstack(( self.nUseTrials, nFitted ))

    def initTrials(self):

        """Initializes the trials object"""

        # refactored into StackTrials object
        #self.parsTrials = np.array([])
        #self.tpTrials = np.array([])
        #self.nUseTrials = np.array([])

        # Also initialize the results object
        self.stackTrials = SimResultsStack()
        self.stackTrialsDiag = SimResultsStack()
        self.stackTrialsUnif = SimResultsStack()
        
    def setupForNonparam(self):

        """Sets up the various objects for nonparametric
        boostrapping"""

        self.stackTrials = SimResultsStack()
        self.drawNonparamSample() # We do this once to initialize the object

        # Set the list of objects and stacks
        lFit = [self.FitResample]
        lSta = [self.stackTrials]

        return lFit, lSta

    def doNonparamBootstrap(self):

        """Perform non-parametric bootstrapping"""

        if self.nTrials < 1:
            return

        lFit, lSta = self.setupForNonparam()

        if self.doFewWeightings:
            self.stackTrialsDiag = SimResultsStack()
            self.stackTrialsUnif = SimResultsStack()
        
            # Create copies with different weighting schemes
            FitScalar  = self.makeSampleWeightsScalar(self.FitData)
            FitUnif = self.makeSampleWeightsUniform(self.FitData)
            
            # Wrap our non-standard weight cases into lists 
            lFit = [self.FitResample, FitScalar, FitUnif]
            lSta = [self.stackTrials, \
                        self.stackTrialsDiag, \
                        self.stackTrialsUnif]

            # 2020-07-20 WIC we may want to switch off inversion of
            # the Hessian for the monte carlo objects since we're
            # unlikely to use it

        # Set the instance status flag for which type of bootstrap
        # we're doing
        self.bootsAreNonparam = True

        for iBoot in range(self.nTrials):
            
            # draw nonparam sample
            self.drawNonparamSample()

            # Propagate into the different-weighting objects (we must
            # do this BEFORE fitting (in case sigma-clipping in the
            # .NE. object cuts down the index array as part of the
            # fitting).
            if self.doFewWeightings:
                planesSel = self.FitResample.NE.gPlanes
                FitScalar.NE.gPlanes = planesSel
                FitUnif.NE.gPlanes = planesSel

            # Fit the samples
            for Fit, Stack in zip(lFit, lSta):
                Fit.performFit(reInit=False)
                Stack.appendNewResults(Fit.NE)

        # Now convert the b,d,e,f to sx, sy, rotDeg, skewDeg
        for Stack in lSta:
            Stack.prepareResults()
            #Stack.convertParsToGeom()
            #Stack.assembleParamSet()


    def doMonteCarlo(self):

        """Wrapper - does the monte carlo trials"""

        if self.nTrials < 1:
            return

        self.initTrials()

        # Set the instance status flag to indicate these are parameric
        # Monte Carlo
        self.bootsAreNonparam = False

        # determine what transformation we'll use for the monte
        # carlo. Will be defaulted to the true parameters and 6term
        # transformation.
        if not self.paramUsesTrueTransf:

            # Which transformation we used is user-selectable. This
            # makes the choice
            parsTransf = self.parsData
            if self.whichFittedTransf.lower().find('diag') > -1:
                if np.size(self.parsDataDiag) > 0:
                    parsTransf = self.parsDataDiag
                else:
                    self.whichFittedTransf = 'full'
            if self.whichFittedTransf.lower().find('unif') > -1:
                if np.size(self.parsDataUnif) > 0:
                    parsTransf = self.parsDataUnif
                else:
                    self.whichFittedTransf = 'full'

            self.parsForParamBoots = np.copy(parsTransf)
            self.choiceForParamBoots = self.fitChoice[:]
        

        for iTrial in range(self.nTrials):
            
            # If we are resetting the positions as well, then we need
            # to update the positions in the unperturbed object.
            if self.resetPositions:
                self.resetUnperturbedPositions()

            # now we create the perturbed positions out of the
            # unperturbed positions
            self.populatePerturbedFitObj()
            self.fitPerturbed()

            # add the results to the stack of fit parameters
            #self.accumulateTrialParams()
            self.stackTrials.appendNewResults(self.FitSample.NE)

            # If we're only doing the one weighting, break off this
            # loop
            if not self.doFewWeightings:
                continue

            # Now that's done, try scalar weights...
            self.FitSampleCopy = self.makeSampleWeightsScalar(self.FitSample)
            self.FitSampleCopy.performFit()
            self.stackTrialsDiag.appendNewResults(self.FitSampleCopy.NE)

            # ... and uniform weights
            self.FitSampleCopy = self.makeSampleWeightsUniform(self.FitSample)
            self.FitSampleCopy.performFit()
            self.stackTrialsUnif.appendNewResults(self.FitSampleCopy.NE)

        # Once the monte carlo is done, convert the parameters to
        # geometric parameters
        self.stackTrials.prepareResults()
        #self.stackTrials.convertParsToGeom()
        #self.stackTrials.assembleParamSet()

        if not self.doFewWeightings:
            return

        for Stack in [self.stackTrialsDiag, self.stackTrialsUnif]:
            Stack.prepareResults()
            #Stack.convertParsToGeom()
            #Stack.assembleParamSet()

        ##self.stackTrialsDiag.convertParsToGeom()
        ##self.stackTrialsUnif.convertParsToGeom()

    def showPositions(self, fignam='tmp_deltas.jpg'):

        """Plots the deltas"""

        fig = plt.figure(1)
        coverrplot(self.xiRaw, self.etaRaw, self.CF, \
                       xLabel=r'$\xi$, deg', yLabel=r'$\eta$, deg', \
                       fig=fig)
        fig.savefig(fignam, rasterized=True)

    def populateTruthsForPlots(self):

        """Utility - populates the 'truth' values for corner-type
        plots"""

        self.truthsPlot = [self.simTheta[0], self.simTheta[3], \
                               self.simSx, self.simSy, \
                               self.simRotDeg, self.simSkewDeg]

        if self.fitChoice.find('similarity') > -1:
            self.truthsPlot = \
                [self.simTheta[0], self.simTheta[3], \
                     0.5*(np.abs(self.simSx) + np.abs(self.simSy)), \
                     self.simRotDeg]

    def showCornerPlot(self, sStack='stackTrials', \
                           doAnnotations=True, \
                           stackLabel='', \
                           truthColor='#4682b4', fignum=2, \
                       tellVars=True):

        """Shows a corner plot of the monte carlo trials."""

        
        # for the moment let's hardcode one example just to see how
        # they look
        stackSho = getattr(self, sStack)
        
        stackArr = getattr(stackSho, 'paramSet')
        labels = getattr(stackSho, 'paramLabels')

        # truth values for the simulation (for corner)
        # self.simParsVec[0], self.simParsVec[1]
        self.populateTruthsForPlots()
        
        #truths = [self.simTheta[0], self.simTheta[3], \
        #              self.simSx, self.simSy, \
        #              self.simRotDeg, self.simSkewDeg]

        #if self.fitChoice.find('similarity') > -1:
        #    truths = [self.simTheta[0], self.simTheta[3], \
        #                  0.5*(np.abs(self.simSx) + np.abs(self.simSy)), \
        #                  self.simRotDeg]

        # OK now try the corner plot. Passing in a blank figure
        # doesn't seem to work, so we just generate 
        print("Plotting corner for %s..." % (sStack))

        # Report std devs to screen if asked
        if tellVars:
            for iPar in range(stackArr.shape[-1]):
                print("%s, %.2e" % (labels[iPar], np.std(stackArr[:,iPar])))

        corner.corner(stackArr, labels=labels, \
                          label_kwargs={'labelpad':50}, \
                          truths=self.truthsPlot, \
                          truth_color=truthColor, \
                      fig=plt.figure(fignum))
        fig = plt.gcf()
        fig.subplots_adjust(left=0.15, bottom=0.15)
        fig.set_size_inches(8.,6., forward=True)
        for ax in fig.get_axes():
            ax.tick_params(axis='both', labelsize=6)        
            ax.xaxis.labelpad = 500

        # Add annnotations
        if doAnnotations:
            fszAnno = 10
            dyAnno = 0.03 # step for annotation lines
            yFirst = 0.95
            if len(stackLabel) < 1:
                stackLabel = sStack[:]

            # Construct the strings for annotations
            sUncty = 'Full covariance in fit'
            if sStack.find('TrialsDiag') > -1:
                sUncty = 'Diagonal covariance in fit'
            if sStack.find('TrialsUnif') > -1:
                sUncty = 'Covariances ignored in fit'

            sChoice = r'Fit with %s transformation' % (self.fitChoice)

            # Type of monte carlo
            sTyp = '%i Non-parametric bootstrap trials' % (self.nTrials)
            if not self.bootsAreNonparam:
                sTyp = '%i Parametric Monte Carlo trials' % (self.nTrials)

            # More info about monte carlo
            if self.bootsAreNonparam:
                sMC = 'Bootstrap fraction %.2f' % (self.fNonparam)
            else:
                if self.resetPositions:
                    if self.paramUsesTrueTransf:
                        sMC = 'using true transformation'
                    else:
                        sMC = 'using "%s" fitted transformation' \
                            % (self.whichFittedTransf)
                else:
                    sMC = 'Unperturbed positions not reset'
            

            # number of datapoints
            # what kind of positional distribution?
            sDist = 'uniformly distributed'
            if self.simMakeGauss:
                sDist = 'Gaussian-distributed'

            sNum = '%i %s datapoints' % (self.simNpts, sDist)

            lAnno = [stackLabel, sUncty, sChoice, sNum, sTyp, sMC]

            if self.flipx:
                lAnno.append(r'$\Delta x$ inverted')

            for iAnno in range(len(lAnno)):
                yAnno = yFirst - iAnno*dyAnno
                ax.annotate(lAnno[iAnno], \
                                (0.95, yAnno), \
                                xycoords='figure fraction', \
                                va='top', ha='right', fontsize=fszAnno)

            # Now also show the 6-term transformation actually used...
            lSeco = [' ']
            #lAnno.append(' ')
            lSeco.append('"True" transformation:')
            lSeco.append(r'a = %.3e' % (self.simTheta[0]))
            lSeco.append(r'd = %.3e' % (self.simTheta[3]))
            lSeco.append(r'$s_x$ = %.3e' % (self.simSx))
            lSeco.append(r'$s_y$ = %.3e' % (self.simSy))
            lSeco.append(r'$\theta = %.2f^{\circ}$' % (self.simRotDeg))
            lSeco.append(r'$\beta = %.2f^{\circ}$' % (self.simSkewDeg))
                         
            for iSeco in range(len(lSeco)):
                ax.annotate(lSeco[iSeco], \
                                (0.95, yAnno - (iSeco+1)*dyAnno), \
                                xycoords='figure fraction', \
                                va='top', ha='right', fontsize=fszAnno, \
                                color=truthColor)
              
    def plotCorners(self):

        """Does the corner plots for each of the fit-methods simulated"""
      
        # On my laptop, the matplotlib window holding the corner plot
        # sometimes triggeres a segfault if clicked on with the
        # mouse. This method does corner plots for all three
        # simulations in turn, saving the plots to jpeg and closing
        # the plot window after each plot. The matplotlib state
        # machine is queried to determine the figure object used to
        # save the figure to disk.

        # what do we call the corner plot in the various cases?
        dKeys = {'stackTrials':'full', 'stackTrialsDiag':'diag', \
                     'stackTrialsUnif':'unif'} 

        for sTyp in dKeys.keys():
            self.showCornerPlot(sTyp)

            # now we construct the filename, save the figure and close
            # it so that the next iteration uses the same figure number
            thisFig = plt.gcf()

            # Save the figure if the filename stem is long enough
            if len(self.stemCornerFil) > 3:
                fnam = '%s_%s.jpg' % (self.stemCornerFil, dKeys[sTyp])
                thisFig.savefig(fnam, rasterized=True)

            # free up the figure handle
            plt.close(thisFig)
            
    def setStackPlotInfo(self):

        """Utility - sets the plot information for the stacks"""
    
        # Set up some properties which will come in handy later
        self.stackTrials.plotColor='k'
        self.stackTrials.plotLabel = 'Full'
        self.stackTrials.plotLS = '-'
        self.stackTrials.plotLW = 1
        self.stackTrials.plotZorder = 10

        self.stackTrialsDiag.plotColor='b'
        self.stackTrialsDiag.plotLabel = 'Diag'
        self.stackTrialsDiag.plotLS = '--'
        self.stackTrialsDiag.plotLW = 1
        self.stackTrialsDiag.plotZorder = 8

        self.stackTrialsUnif.plotColor='r'
        self.stackTrialsUnif.plotLabel='Unif'
        self.stackTrialsUnif.plotLS='-.'
        self.stackTrialsUnif.plotLW=1
        self.stackTrialsUnif.plotZorder = 6.

    def plotCovarPairs(self):

        """Wrapper - plots covariance-pair visualizations"""

        self.setStackPlotInfo()
        self.showCovarPairs(self.stackTrials, newFig=True, \
                                reportMarginals=True, plotTruths=True)
        for stack in [self.stackTrialsDiag, self.stackTrialsUnif]:
            self.showCovarPairs(stack, newFig=False)

    def showCovarPairs(self, stackObj=None, figNum=2, newFig=True, \
                           reportMarginals=False, plotTruths=False, \
                           truthColor='0.5', truthAlpha=0.5, \
                           truthZorder=12):

        """Visualizes the covariance matrices of multiple sim stacks
        in the same Corner-like plot"""

        # 2020-07-20 - at the moment, the fit parameters are all in
        # terms of a,b,c,d,e,f rather than the geometric
        # parameters. For the moment, cross-compare the simulations
        # since those are already in the format we want. So, start
        # with that.
        
        # The formal prediction from the data with uncertainties
        #covPairsData = CovarsMxM(self.formalCovData, self.paramLabels, \
        #                             pairsOnInit=True)

        if stackObj is None:
            stackShow = self.stackTrials
        else:
            stackShow = stackObj

        # at the moment this is just a renaming...
        covPairsFull = stackShow.covPairs

        # determine the figure object
        self.figCovarPairs = plt.figure(figNum, constrained_layout=False)

        nPars = np.shape(stackShow.paramCov)[0]

        # Set up the axes plane-by-plane, using the special extra
        # arrays we added to covPairs.
        cols = covPairsFull.iParsPlot
        rows = covPairsFull.jParsPlot
        iMax = np.max(cols)
        jMax = np.max(rows)

        # If we will be plotting the 'truths', ensure they are set
        if plotTruths:
            self.populateTruthsForPlots()

        # We'll also set up parameter midpoint arrays here, for
        # convenience of passing to the plot method later.
        parsData = np.zeros((np.size(cols), 2))

        # I think the "easiest" way to have objects as a grid is to
        # set up a list of lists. So:
        #LLaxes = [[None for i in range(nPars)] for j in range(nPars)]
        #LLlabl = [[None for i in range(nPars)] for j in range(nPars)]

        # set up the gridspec
        # Dictionary of accessible axis objects, keyed by the pair
        # label. Because this is interacting with the matplotlib state
        # machine, we should be able to get away with reconstituting
        # this each time.
        if newFig:
            self.figCovarPairs.clf()
            self.figCovarPairs.subplots_adjust(\
                hspace=0.05, wspace=0.05, left=0.2,bottom=0.2)

            self.axesCovarPairs = {}
            gs = gridspec.GridSpec(nrows=iMax+1, ncols=jMax+1) 

            # Populates the LLaxes [[]] axes object
            self.setupCovAxesWithSharing(nPars)

            for iPlane in range(np.size(cols)):
                thisLabel = covPairsFull.planeLabels[iPlane]
                thisCol = cols[iPlane]
                thisRow = rows[iPlane]
            
                self.axesCovarPairs[thisLabel] = self.LLaxes[thisRow][thisCol]

        for iPlane in range(np.size(cols)):
            thisLabel = covPairsFull.planeLabels[iPlane]
            thisCol = cols[iPlane]  # note transposed in ax call.
            thisRow = rows[iPlane]
            
            # slot in to the parsData object so we can easily plot
            # below
            parsData[iPlane,0] = stackShow.paramMed[thisCol]
            parsData[iPlane,1] = stackShow.paramMed[thisRow]

            # which axis are we using?
            thisAx = self.axesCovarPairs[thisLabel]

            # What we plot depends on whether this is a 2d or a 1d
            # covariance... Start with the nice simple 1D case
            # first...
            self.addCovObjectToAxis(covPairsFull, iPlane,\
                                        parsData, \
                                        is2d=thisRow != thisCol, \
                                        ax=thisAx, \
                                        color=stackShow.plotColor, \
                                        lw=stackShow.plotLW, \
                                        ls=stackShow.plotLS, \
                                        label=stackShow.plotLabel, \
                                        zorder=stackShow.plotZorder)

            # Annotate with marginals?
            if reportMarginals and thisRow == thisCol:
                stdx = covPairsFull.stdx[iPlane]
                labl = stackShow.paramLabels[thisCol] # not sure why not working
                sTitl=r'$\sigma$ = %.2e' % (stdx)
                thisAx.set_title(sTitl, fontsize=7, color=stackShow.plotColor)

            # Overplot truth values?
            if plotTruths:                
                xThis = self.truthsPlot[thisCol]
                yLimCurrent = np.copy(thisAx.get_ylim())
                yLimNew = yLimCurrent + \
                    10.0 * (yLimCurrent - np.mean(yLimCurrent))
                dum = thisAx.plot([xThis, xThis], yLimNew, \
                                      alpha=truthAlpha, color=truthColor, \
                                      zorder=truthZorder)
                thisAx.set_ylim(yLimCurrent)

                if thisCol != thisRow:
                    yThis = self.truthsPlot[thisRow]
                    xLimCurrent = np.copy(thisAx.get_xlim())
                    xLimNew = xLimCurrent + \
                        10.0 * (xLimCurrent - np.mean(xLimCurrent))
                    dum = thisAx.plot(xLimNew, [yThis, yThis], \
                                          zorder=truthZorder, \
                                          alpha=truthAlpha, color=truthColor)
                    thisAx.set_xlim(xLimCurrent)

            # labels - IF new figure!
            if newFig:
                if thisCol < 1:
                    thisAx.set_ylabel(\
                        stackShow.paramLabels[thisRow], \
                            rotation=0)
                    thisAx.tick_params(axis='y', labelsize=7)


            # otherwise, hide the axis
                else:
                    thisAx.set_yticklabels([])

                if thisRow == jMax:
                    thisAx.set_xlabel(\
                        stackShow.paramLabels[thisCol])
                    thisAx.tick_params(axis='x', labelsize=7, \
                                                    rotation=90)


                # otherwise hide the labels
                else:
                    thisAx.set_xticklabels([])

        # Despite our overthinking the share axis behavior, the
        # lower-right panel doesn't have axes scaled with anything
        # else yet. Start by matching it to the lower-left Y range.
        axlr = self.LLaxes[nPars-1][nPars-1]
        axll = self.LLaxes[nPars-1][0]
        axlr.set_xlim(np.copy(axll.get_ylim()))

    def addCovObjectToAxis(self, cov=None, iPlane=0, \
                               centers=np.array([]), is2d=True, ax=None, \
                               color='g', label='', ls='-', lw=1, \
                               nFine=100, nSigRange=2., zorder=5):

        """Adds a covariance slice to axis. cov = covarsNx2x2 object,
        iPlane=plane in the covariance object"""

        if cov is None:
            return

        if ax is None:
            return

        nSigRange = np.float(nSigRange)

        # Median locatin
        medn = centers[iPlane]

        if not is2d:

            # use a broader range for 1d
            stdx = cov.stdx[iPlane]
            medx = medn[0]
            xFine = np.linspace(medx-3.0*nSigRange*stdx, \
                                    medx+3.0*nSigRange*stdx, 
                                nFine, endpoint=True)

            #print("INFO:", np.min(xFine), np.max(xFine))

            powr = 0.-0.5*((xFine - medx)/stdx)**2
            norm = 1.0 / (np.sqrt(2.0*np.pi*stdx))

            yFine = norm * np.exp(powr)
            dum = ax.plot(xFine, yFine, color=color, label=label, \
                              ls=ls, lw=lw, zorder=zorder)
            return

        # if the object IS two-dimensional, then we add the ellipse,
        # centered on the midpoint. 
        width = 2.0*cov.majors[iPlane]**0.5
        height = 2.0*cov.minors[iPlane]**0.5
        theta = cov.rotDegs[iPlane]
        xCen = medn[0]
        yCen = medn[1]

        # Do we have to actually plot the corners for the ellipse to work?
        xMin = xCen - nSigRange*cov.stdx[iPlane]
        xMax = xCen + nSigRange*cov.stdx[iPlane]
        yMin = yCen - nSigRange*cov.stdy[iPlane]
        yMax = yCen + nSigRange*cov.stdy[iPlane]

        # don't shrink either axis limit if already exists
        if len(ax.patches) > 0:
            xLimCurrent = np.copy(ax.get_xlim())
            yLimCurrent = np.copy(ax.get_ylim())
            if xMin < xLimCurrent[0]:
                xLimCurrent[0] = xMin
            if xMax > xLimCurrent[1]:
                xLimCurrent[1] = xMax
            if yMin < yLimCurrent[0]:
                yLimCurrent[0] = yMin
            if yMax > yLimCurrent[1]:
                yLimCurrent[1] = yMax

            ax.set_xlim(xLimCurrent)
            ax.set_ylim(yLimCurrent)
        else:
            ax.set_xlim(xMin, xMax)
            ax.set_ylim(yMin, yMax)

        # let's try that for an ellipse
        thisEll = Ellipse((xCen, yCen), width, height, theta, \
                              facecolor='none', edgecolor=color, \
                              zorder=zorder, label=label, ls=ls, lw=lw)
        fillEll = Ellipse((xCen, yCen), width, height, theta, \
                              facecolor=color, edgecolor='none', \
                              label=label, ls=ls, lw=lw, \
                              alpha=0.3, zorder=np.max([zorder-1,1]) )


        ax.add_patch(thisEll)
        ax.add_patch(fillEll)
        #ax.autoscale()

    def setupCovAxesWithSharing(self, nPars=6):

        """Sets up the axis list of lists required by showCovarPairs"""

        # Needs a figure to work on
        if self.figCovarPairs is None:
            return 

        LLaxes = [[None for i in range(nPars)] for j in range(nPars)]
        gs = gridspec.GridSpec(nrows=nPars, ncols=nPars) 
        
        # set up the nPars x nPars grid of axes, lower-left + diagonal
        for iCol in range(nPars):
            for jRow in range(iCol, nPars):
                LLaxes[jRow][iCol] = self.figCovarPairs.add_subplot(\
                    gs[jRow, iCol])

        # set all columns to share the bottom row x axis
        for iCol in range(nPars-1):
            refAxCol = LLaxes[nPars-1][iCol]
            #refAxCol.set_title('%i, %i' % (jRow, iCol), color='b')
            for jRow in range(nPars-1)[::-1]:
                thisAxCol = LLaxes[jRow][iCol]
                if thisAxCol is None:
                    continue
                #thisAxCol.set_title('%i, %i' % (jRow, iCol), color='r')
                thisAxCol.get_shared_x_axes().join(thisAxCol, refAxCol)
        
        # Set all rows to share the left column y
        for jRow in range(1, nPars):
            refAxRow = LLaxes[jRow][0]
            #refAxRow.set_title('REF', color='g')
            for iCol in range(1,nPars-1):
                thisAxRow = LLaxes[jRow][iCol]
                if thisAxRow is None:
                    continue

                # Don't mess up the histograms
                if iCol == jRow:
                    continue

                thisAxRow.get_shared_y_axes().join(thisAxRow, refAxRow)

        self.LLaxes = LLaxes

    def writeSummaryStats(self):

        """Wrapper - dumps the summary statistics to disk"""

        if len(self.stemSummStats) < 1:
            return

        # file ending. Do a bit of parsing here to save angst later
        if len(self.tailSummStats) < 1:
            self.tailSummStats = '.csv'
        if self.tailSummStats.find('.') < 0:
            self.tailSummStats = '.%s' % (self.tailSummStats)
        

        for sObj in ['stackTrials', 'stackTrialsDiag', 'stackTrialsUnif']:
            
            # The info and methods for writing are attached to these
            # object names. So can only proceed if the object is
            # populated.
            if not hasattr(self, sObj):
                continue

            stack = getattr(self, sObj)
            if stack is None:
                continue

            if len(stack.tCovPairs) < 1:
                continue

            stack.filCovPairs = '%s_%s%s' % \
                (self.stemSummStats, sObj, self.tailSummStats)
            stack.writeSummaryTable()

    def writeParfile(self):

        """Dumps the parameters to disk"""

        if len(self.filParamsOut) < 3:
            if self.Verbose:
                print("writeParfile WARN - output parfils < 3 chars. Not writing.")
            return

        if len(self.ddump.keys()) < 1:
            self.setParsToDump()

        with open(self.filParamsOut, 'w') as wObj:
            wObj.write('# Pars for NormWithMonteCarlo\n')
            wObj.write('# %s\n' % (datetime.datetime.now()))
            wObj.write('#\n')
            wObj.write('# attribute_name value\n')

            # Now write the variables.
            for sKey in self.ldump:
                if not sKey in self.ddump.keys():
                    print("writeParfile WARN - param %s not in list: %s" \
                              % (sKey))
                    continue 
                
                #print("INFO - %s" % (sKey))
                # If the attribute doesn't exist...
                if not hasattr(self, sKey):
                    print("writeParfile WARN - object has no attribute: %s" \
                              % (sKey))
                    continue

                # if there's a spacer/comment, write it
                sCommen = self.ddump[sKey]['comment']
                if len(sCommen) > 0:
                    # Ensure the comment starts with '#' if I forgot
                    # to set that elsewhere.
                    if sCommen.find('#') != 0:
                        sCommen = '#%s' % (sCommen)
                    wObj.write('\n')
                    wObj.write('%s\n' % (sCommen))

                # now write the keyword and value. Since my laptop is
                # still on python 2 and I don't know how to pass the
                # format code as a variable, we'll do this the
                # explicit way.
                valu = getattr(self, sKey)
                styp = self.ddump[sKey]['dtype']

                if styp.find('int') > -1 or styp.find('bool') > -1:
                    wObj.write('%s %i\n' % (sKey, valu))
                    continue

                if styp.find('str') > -1:
                    # strings can have zero length...
                    if len(valu) < 1:
                        valu = 'BLANK'
                    wObj.write('%s %s\n' % (sKey, valu))
                    continue

                if styp.find('float') > -1:
                    wObj.write('%s %f\n' % (sKey, valu))

    def setParsToDump(self):

        """Sets the list of attributes to dump to disk"""

        # This probably should be called from __init__()

        # We first build up the list of keywords so that we can write
        # them out in order (as opposed to whatever order python 

        # We'll build this up as keyword / dtype so that it'll be
        # easier to read the parameters correctly. Written line by
        # line in the source code here for ease of debugging later.

        self.ldump = ['simNpts', 'simSx', 'simSy', 'simRotDeg', \
                          'simSkewDeg', 'simXiRef', 'simEtaRef', \
                          'simXmin', 'simXmax', 'simYmin', 'simYmax', \
                      'xRef', 'yRef', \
                          'simXcen', 'simYcen', 'simMakeGauss', \
                          'simGauMajor', 'simGauMinor', 'simGauTheta', \
                          'simAlo', 'simAhi', 'simRotCov', \
                      'simBAlo', 'simBAhi', \
                      'fOutly', 'rOutly_min_arcsec', 'rOutly_max_arcsec', \
                          'genStripe', 'stripeFrac', 'stripeCovRatio', \
                          'posnSortCol', \
                          'nTrials', 'resetPositions', 'doFewWeightings', \
                          'doNonparam', 'paramUsesTrueTransf', \
                          'whichFittedTransf', \
                          'fNonparam', \
                          'filParamsIn', 'filParamsOut', \
                          'fitChoice', \
                          'flipx', \
                          'stemCornerFil', 'stemSummStats', 'tailSummStats']

        # Now we set up the parameter dtype dictionary. This also
        # allows us to add a pre-comment should we wish. 
        self.ddump = {}
        for skey in self.ldump:
            self.ddump[skey] = {'dtype':'float', 'comment':''}

        # Now we add the customizations for each of the non-float
        # datatypes
        for sInt in ['nTrials', 'simNpts']:
            self.ddump[sInt]['dtype'] = 'int'
            
        for sBool in ['simMakeGauss', 'genStripe', 'doFewWeightings', \
                          'doNonparam', 'paramUsesTrueTransf', \
                          'flipx', \
                          'resetPositions']:
            self.ddump[sBool]['dtype'] = 'bool'

        for sStr in ['posnSortCol', 'fitChoice', \
                         'filParamsOut', 'filParamsIn', 'stemCornerFil', \
                         'stemSummStats', 'tailSummStats', \
                         'whichFittedTransf']:
            self.ddump[sStr]['dtype'] = 'str'

        # Now we put some comments in to separate the outputs in the
        # param file to make it human-readable
        self.ddump['simNpts']['comment'] = '# True transformation'
        self.ddump['simXmin']['comment'] = '# Focal plane field of view'
        self.ddump['simMakeGauss']['comment'] = '# Gaussian component params'
        self.ddump['simAlo']['comment'] = '# Simulated covariances parameters'
        self.ddump['fOutly']['comment'] = '# Outlier generation'
        self.ddump['genStripe']['comment'] = '# Stripe parameters'
        self.ddump['posnSortCol']['comment'] = '# Sort positionally?'
        self.ddump['nTrials']['comment'] = '# Monte carlo settings'
        self.ddump['fitChoice']['comment'] = '# Settings for fit'
        self.ddump['stemCornerFil']['comment'] = '# summary statistics settings'

            
    def readPars(self, parFile='NONE.NONE'):

        """Utility - sets simulation attributes from input file"""

        if len(parFile) < 1:
            return

        if not os.access(parFile, os.R_OK):
            return

        if self.Verbose:
            print("readPars: reading attributes from file %s" \
                      % (parFile))

        if len(self.ddump.keys()) < 1:
            self.setParsToDump()

        # Written for clarity of approach rather than speed of
        # execution. The datatypes expected are not included in the
        # parameter file (just because that might be annoying) but
        # instead are pre-built in dictionary self.ddump. Attributes
        # that don't have an entry in self.ddump are thus ignored on
        # input. If Verbose, write when skipping so that I can work
        # out if I need to add more objects to self.ddump.
        with open(parFile, 'r') as rObj:
            for inline in rObj:
                if inline.find('#') == 0:
                    continue

                if len(inline.strip()) < 1:
                    continue

                lLine = inline.strip().split()
                if len(lLine) == 1:
                    if self.Verbose:
                        print("readPars WARN - no value given for %s" \
                                  % (lLine[0]))
                    continue

                # (Note that we DO allow > 2 entries, e.g. the user
                # might have put a comment after the values.

                # OK now how we interpret this depends on what we
                # think this type of quantity should be. We might
                # decide later to set default behavior (e.g. assume
                # string unless told otherwise). For the moment,
                # though, we will be stricter and ignore anything not
                # in self.ddump.keys().
                sAttr = lLine[0]
                sValu = lLine[1]
        
                if not sAttr in self.ddump.keys():
                    print("readPars WARN - unexpected attr: %s" \
                              % (sAttr))
                    continue

                # Now interpret the quantity
                sTyp = self.ddump[sAttr]['dtype']
                
                # Integers
                if sTyp.find('int') > -1:
                    setattr(self, sAttr, int(sValu))
                    continue

                # Booleans require conversion from integers (I think!!)
                if sTyp.find('bool') > -1:
                    setattr(self, sAttr, bool(int(sValu)) )
                    continue

                # floats (probably the majority)
                if sTyp.find('float') > -1:
                    setattr(self, sAttr, float(sValu))
                    continue
                            
                # Strings are the simplest. Goes at the end since this
                # may well end up being default behavior.
                if sTyp.find('str') > -1:
                    setattr(self, sAttr, sValu.strip())
                    continue

    ### A couple of wrappers follow to achieve common tasks. 

    def prepSimData(self):

        """Sets up the source and target data using sim parameters"""

        self.transfParsAsVecs()
        self.generateRawXY()
        self.populateRawXiEta()
        self.sortRawPositions()

        self.populateCovarsFromSim()

        # Populate the outlier deltas, but don't add them to
        # self.xiRaw and self.etaRaw (so that we can choose at what
        # stage to apply them, in the calling methods).
        self.setupOutliers()

        # 2023-06-20 WIC - commenting this out. This method is used in
        # an MCMC sense anyway, when we don't use the unperturbed fit
        # object.
        self.populateUnperturbedFitObj()
        
        
    def setupAndFit(self):

        """Wrapper - Sets up the simulation, and does the fit. (The
        parameters should have been loaded on initialization.)"""

        self.transfParsAsVecs()
        self.generateRawXY()
        self.populateRawXiEta()
        self.sortRawPositions()

        self.populateCovarsFromSim()
        self.populateUnperturbedFitObj()

        # now populate the perturbed fit object and fit the data
        self.populatePerturbedFitObj()
        self.copyPerturbedSimToData()
        self.fitData()

    def setupAndBootstrap(self):

        """Wrapper - sets up the bootstraps (parametric or otherwise)
        and runs them."""

        if self.nTrials < 3:
            if self.Verbose:
                print("setupAndBootstrap INFO - nTrials < 3. Not doing bootstraps")
            return

        # string for annotations
        sPar = 'parametric'
        if self.doNonparam:
            sPar = 'non-parametric'

        print("Starting %i %s trials..." % (self.nTrials, sPar))

        # Actually do the simulation!
        t0 = time.time()
        if not self.doNonparam:
            self.doMonteCarlo()
        else:
            self.doNonparamBootstrap()

        t1 = time.time()
        print("%i %s bootstrap trials took %.2e seconds" \
                  % (self.nTrials, sPar, t1 - t0))

        # for the moment let's show the corner plot
        # self.showCornerPlot()

#    def convertParsToGeom(self):

#        """Converts the [[b,c],[e,f]] transformations into geometric
#        parameters sx, sy, rotation, skew"""

#        # REFACTORED into StackTrials object

#        if np.size(self.parsTrials) < 1:
#            return

#        # There's probably a clever pythonic way to construct the
#        # Nx2x2 matrix stack out of the Nx4 or Nx6 stack of fitted
#        # parameters (like reshaping and then removing a column). For
#        # now, I build this in a way I understand...
#        nDone, nTerms = np.shape(self.parsTrials)
#        stackTransfs = np.zeros((nDone, 2, 2))

#        stackTransfs[:,0,0] = self.parsTrials[:,1]
#        stackTransfs[:,0,1] = self.parsTrials[:,2]

#        print("INFO::", nDone, nTerms)

#        if nTerms < 6:
#            stackTransfs[:,1,0] = 0 - self.parsTrials[:,2]
#            stackTransfs[:,1,1] = self.parsTrials[:,1]
#        else:
#            stackTransfs[:,1,0] = self.parsTrials[:,4]
#            stackTransfs[:,1,1] = self.parsTrials[:,5]

#        # Create the object
#        self.transfs2x2 = Stack2x2(stackTransfs)

#        # Also do the inverse transformation
#        self.transfs2x2rev = Stack2x2(self.transfs2x2.AINV)
        
#        #print(stackTransfs[2])
#        #print(np.linalg.inv(stackTransfs[2]))
#        #print(self.transfs2x2.AINV[2])

#### Normal Equations Fitting class

class SimResultsStack(object):

    """N x K stack of K-length results from N monte carlo trials.

    Written for NormWithMonteCarlo, so includes input parameters,
    tangent points estimated from those parameters, and the number of
    points kept.

    """

    def __init__(self, doRev=True):

        # Do the reverse transformation as well?
        self.doRev = doRev

        # The parameters, tangent points, and number kept
        self.parsTrials = np.array([])
        self.tpTrials = np.array([])
        self.nUseTrials = None

        # linear transformation including geometric parameters
        self.transfs2x2 = None
        self.transfs2x2rev = None

        # N x 6 array of all the parameters, labels - for sending to a
        # plotter like Corner
        self.paramSet = np.array([]) # (needs a better name)
        self.paramLabels = []

        # Since we have the paramset, we can store the covariance and
        # median here.
        self.paramCov = np.array([])
        self.paramMed = np.array([])

        # astropy table of summary statistics (as covariance pairs)
        # for convenient writing to disk
        self.tCovPairs = []
        self.filCovPairs = '' # output file for covariance pairs

        # Some labels which will be of use when plotting
        self.plotColor = 'g'
        self.plotLabel = 'Full'
        self.plotLS = '-'
        self.plotLW = 1
        self.plotZorder = 10

    def appendNewResults(self, NE=None):

        """Appends the results onto the stack"""

        # Uses as input a NormalEqs object
        thesePars = np.copy(NE.pars[:,0])
        thisTP = np.copy(NE.xZero)
        nFitted = np.size(NE.gPlanes)

        if np.size(self.parsTrials) < 1:
            self.parsTrials = thesePars
            self.tpTrials = thisTP
            self.nUseTrials = nFitted
            return

        self.parsTrials = np.vstack(( self.parsTrials, thesePars ))
        self.tpTrials   = np.vstack(( self.tpTrials, thisTP ))
        self.nUseTrials = np.hstack(( self.nUseTrials, nFitted ))

    def convertParsToGeom(self):

        """Converts the [[b,c],[e,f]] transformations into geometric
        parameters sx, sy, rotation, skew"""

        if np.size(self.parsTrials) < 1:
            return

        # There's probably a clever pythonic way to construct the
        # Nx2x2 matrix stack out of the Nx4 or Nx6 stack of fitted
        # parameters (like reshaping and then removing a column). For
        # now, I build this in a way I understand...
        nDone, nTerms = np.shape(self.parsTrials)
        stackTransfs = np.zeros((nDone, 2, 2))

        stackTransfs[:,0,0] = self.parsTrials[:,1]
        stackTransfs[:,0,1] = self.parsTrials[:,2]

        if nTerms < 6:
            stackTransfs[:,1,0] = 0 - self.parsTrials[:,2]
            stackTransfs[:,1,1] = self.parsTrials[:,1]
        else:
            stackTransfs[:,1,0] = self.parsTrials[:,4]
            stackTransfs[:,1,1] = self.parsTrials[:,5]

        # Create the object
        self.transfs2x2 = Stack2x2(stackTransfs)

        # Also do the inverse transformation
        if self.doRev:
            self.transfs2x2rev = Stack2x2(self.transfs2x2.AINV)
        
    def assembleParamSet(self):

        """Utility: arranges the master set of parameters into an Nx6
        array (e.g. for use in corner plot)"""

        # Judge from the array shape whether 6 or 4 term
        nDone, nTerms = np.shape(self.parsTrials)

        if nTerms == 4:
            # for 4-term, the sx and sy are both identical (uncomment
            # the line below to confirm this).
            #print("INFO - 4term:", \
            #          self.transfs2x2.sx[0:3], \
            #          self.transfs2x2.sy[0:3] )

            self.paramSet = np.column_stack(( \
                    self.parsTrials[:,0], self.parsTrials[:,3], \
                        self.transfs2x2.sx, self.transfs2x2.rotDeg ))
            self.paramLabels = [r'$a$', r'$d$', r'$s$', r'$\theta$']
            return

        self.paramSet = np.column_stack(( \
           #self.tpTrials[:,0], self.tpTrials[:,1], \
                self.parsTrials[:,0], self.parsTrials[:,3], \
                    self.transfs2x2.sx, self.transfs2x2.sy, \
                    self.transfs2x2.rotDeg, self.transfs2x2.skewDeg))

        self.paramLabels = [r'$a$', r'$d$', \
                                r'$s_x$', r'$s_y$', \
                                r'$\theta$', r'$\beta$']

    def getParamStats(self):

        """Utility - computes the covariance and medians of the
        parameter stack"""

        # Don't bother if fewer than three parameters
        if self.paramSet.shape[0] < 3:
            return

        self.paramCov = np.cov(self.paramSet, rowvar=False)
        self.paramMed = np.median(self.paramSet, axis=0)

    def prepareResults(self):

        """Wrapper - converts the results to geometric parameters,
        prepares the stack (for corner plots, say) and computes
        summary statistics"""

        self.convertParsToGeom()
        self.assembleParamSet()
        self.getParamStats()
        self.assembleCovarPairs()
        self.tabulateCovarPairs() # repackage as table

    def assembleCovarPairs(self):

        """Construct covariance matrices of parameter-pairs to
        summarize the bootstrap parameters"""

        # I have a feeling corner.py probably does something like this
        # under the hood, but am not sure how to access the relevant
        # quantities. Let's do it here.
        if np.size(self.paramCov) < 1:
            self.getParamStats()
        if np.size(self.paramCov) < 1:
            return

        # 2020-07-20 refactored the syntax that follows into CovarsMxM
        # class. Consider removing this enclosing method entirely.
        CMM = CovarsMxM(self.paramCov, self.paramLabels, pairsOnInit=True)
        self.covPairs = CMM.covPairs
    
    def tabulateCovarPairs(self):

        """Packages the covariance-pairs into a table and writes to disk"""

        self.tCovPairs = Table()
        self.tCovPairs['pair'] = self.covPairs.planeLabels[:]

        # Can we loop through attribute names?
        for sAttr in ['majors', 'minors', 'rotDegs', \
                          'stdx', 'stdy', 'corrxy']:

            if hasattr(self.covPairs, sAttr):
                self.tCovPairs[sAttr] = getattr(self.covPairs, sAttr)

    def writeSummaryTable(self):

        """Writes the summary table to disk"""

        if len(self.tCovPairs) < 1:
            return

        if len(self.filCovPairs) < 3:
            return

        self.tCovPairs.write(self.filCovPairs, overwrite=True)

class FitNormEq(object):

    """Fits the transformation between two datasets using the normal
    equations approach, allowing for possibly covariant
    uncertainties. Runs on initialization. The weights can be
    specified in the following ways:

    W = Nx2x2 array of weights. If specified, then covariances are
        ignored.

    covars = Nx2x2 array of covariances. If weights not specified,
             they are constructed by inversion of the covariances
             matrix-stack.

    fitChoice = string, choice of fitting method

    flipx = flip x-coords in the fit object?

    stdxi, stdeta, corrxieta = covariances in xi, eta specified as
             stddev in xi, stddev in eta, correlation coefficient
             between xi and eta.

    """

    def __init__(self, x=np.array([]), y=np.array([]), \
                     xi=np.array([]), eta=np.array([]), \
                     covars=np.array([]), \
                     stdxi=np.array([]), stdeta=np.array([]), \
                     corrxieta=np.array([]), \
                     W=np.array([]), \
                     xRef=0., yRef=0., \
                     invertHessian=False, \
                     fitChoice='6term', \
                     flipx=False, \
                     runOnInit=True, \
                     anglesInDegrees=False, \
                     Verbose=False):

        self.Verbose = Verbose

        # Datapoints in each frame, ref points, covariances
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.xi = np.copy(xi)
        self.eta = np.copy(eta)
        self.xRef = np.copy(xRef)
        self.yRef = np.copy(yRef)
        
        # Choice of fitter
        self.fitChoice = fitChoice[:]
        self.flipx = flipx
        
        # Covariances in the target frame, if given
        self.covars = np.copy(covars)

        # Covariances as stddevs & correlation, if given
        self.stdxi = np.copy(stdxi)
        self.stdeta = np.copy(stdeta)
        self.corrxieta = np.copy(corrxieta)

        # Weights (can be supplied)
        self.W = np.copy(W)        
        self.ensureWeightsPopulated()

        # Jacobian for linear parameters to geometric parameters
        self.jacLinToGeom = np.array([])

        # Angular conversion factor for jacobian
        self.angJacFac = 1.
        if anglesInDegrees:
            self.angJacFac = 180./np.pi

        # Normal equations object, formal covariance estimate (if
        # desired)
        self.NE = None

        # perform the fit on initialization unless told not to
        if runOnInit:
            self.performFit()
        
            if invertHessian:
                self.NE.invertHessian()

                self.populateJacobian() 
                self.propagateCovarianceToGeom()

    def ensureWeightsPopulated(self):

        """Ensures weights are populated"""

        nRows = np.size(self.x)

        # Weights should be nRows x 2 x 2 so size 4Nrows.
        if np.size(self.W) == 4*nRows:
            return

        if np.size(self.covars) < 1 and np.size(self.stdxi) == nRows:
            CD=CovarsNx2x2(stdx=self.stdxi, stdy=self.stdeta, \
                               corrxy=self.corrxieta)
            self.covars = CD.covars

        if np.size(self.covars) == 4*nRows:
            self.weightsFromCovars()
            return

        # If we got here then neither the weights, nor a covariance
        # stack, nor the information needed to build the covariance
        # stack, were supplied. In that instance, just use uniform
        # weights.
        if self.Verbose:
            print("FitNormEq.ensureWeightsPopulated WARN - using uniform weights")
        self.W = np.repeat(np.eye(2)[np.newaxis,:,:], nRows, axis=0)

    def weightsFromCovars(self):

        """Populates the weights given the covariances"""

        self.W = np.linalg.inv(self.covars)

    def populateJacobian(self):

        """Populates the Jacobian taking linear parameters to
        geometric parameters"""

        # Requires the fit values to be populated
        if self.fitChoice.find('6term') > -1:
            self.fillJacobian6term()
            return

        if self.fitChoice.lower().find('similarity') > -1:
            self.fillJacobianSimilarity()
            return

    def fillJacobian6term(self):

        """Fills in the values in the Jacobian for the 6-term
        transformation"""
            
        #print("INFO:", np.shape(self.NE.pars))
        a,b,c,d,e,f = self.NE.pars[:,0]

        sx = np.sqrt(b**2 + e**2)
        sy = np.sqrt(c**2 + f**2)

        self.jacLinToGeom = np.zeros((6,6))
        self.jacLinToGeom[0,0] = 1.
        self.jacLinToGeom[1,3] = 1.
        self.jacLinToGeom[2,1] = b/sx
        self.jacLinToGeom[2,4] = e/sx
        self.jacLinToGeom[3,2] = c/sy
        self.jacLinToGeom[3,5] = f/sy
        self.jacLinToGeom[4,1] =  e*self.angJacFac/(2.0*sx**2)
        self.jacLinToGeom[4,2] =  f*self.angJacFac/(2.0*sy**2)
        self.jacLinToGeom[4,4] = -b*self.angJacFac/(2.0*sx**2)
        self.jacLinToGeom[4,5] = -c*self.angJacFac/(2.0*sy**2)
        self.jacLinToGeom[5,1] = -e*self.angJacFac/sx**2
        self.jacLinToGeom[5,2] =  f*self.angJacFac/sy**2
        self.jacLinToGeom[5,4] =  b*self.angJacFac/sx**2
        self.jacLinToGeom[5,5] = -c*self.angJacFac/sy**2

    def fillJacobianSimilarity(self):

        """Fills in the terms in the Jacobian for similarity
        transform"""

        a,b,c,d = self.NE.pars[:,0]

        s = np.sqrt(b**2 + c**2)

        self.jacLinToGeom = np.zeros((4,4))
        self.jacLinToGeom[0,0] = 1.
        self.jacLinToGeom[1,3] = 1.
        self.jacLinToGeom[2,1] = b/s
        self.jacLinToGeom[2,2] = c/s
        self.jacLinToGeom[3,1] = -c*self.angJacFac/s**2
        self.jacLinToGeom[3,2] =  b*self.angJacFac/s**2

    def propagateCovarianceToGeom(self):

        """Propagates the formal covariance estimate from linear to
        geometric parameters"""

        J = self.jacLinToGeom
        V = self.NE.formalCov

        JVJt = np.dot(J,np.dot(V, np.transpose(J)) )

        # use this to populate the propagated uncertainty of the
        # normal-equations object
        self.NE.formalCovGeom = JVJt

    def initNE(self):

        """(Re-) initializes the Normal-Equations object for this Fit
        object"""

        #scaleX = 1.
        #if self.flipx:
        #    scaleX = -1.

        self.NE = NormalEqs(self.x, self.y, self.xi, self.eta, W=self.W, \
                                fitChoice=self.fitChoice, \
                                flipx=self.flipx, \
                                xref=self.xRef, yref=self.yRef)

    def performFit(self, reInit=True):

        """Sets up the normal equations object and performs the fit,
        as well as operations we are likely to want for every trial
        (such as finding the tangent point from the fit parameters)"""

        # Exit gracefully if data not set yet
        if np.size(self.x) < 1:
            return

        if reInit:
            self.initNE()

        self.NE.doFit()
        self.NE.estTangentPoint()

    def findFormalCovar(self):

        """Finds the formal covariance estimate for the parameters"""

        self.NE.invertHessian()

class Outliers2d(object):

    """Convenience methods for computing dx, dy for outliers"""

    def __init__(self, nOutliers=10, rOuter=0.1, rInner=0.):

        self.nOutliers = nOutliers

        # Enforce rmax > rmin, otherwise strange things happen...
        self.rInner = np.min([rInner, rOuter])
        self.rOuter = np.max([rInner, rOuter])
        
    def unifInDisk(self):
        
        """

        Generates uniformly-distributed random xy points within the disk,
        between radii rInner and rOuter
        
        """

        # radii, uniform in sqrt(r)
        rng = np.random.default_rng()
        runif = np.sqrt(rng.uniform(size=self.nOutliers))
        
        r = self.rInner + (self.rOuter - self.rInner)*runif
        theta = rng.uniform(size=self.nOutliers) * 2.0 * np.pi
        
        dx = r * np.cos(theta)
        dy = r * np.sin(theta)

        return dx, dy
        
#### Some routines that use this follow

def sixtermToPars(b,c,e,f):

    """Utility - given the scaling parameters in a 6-term linear
relationship, interpret them as the parameters sx, sy, rotation,
skew."""

    Amat = np.zeros((b.size, 2, 2))
    Amat[:,0,0] = b
    Amat[:,0,1] = c
    Amat[:,1,0] = e
    Amat[:,1,1] = f

    astack = Stack2x2(Amat)

    return astack.sx, astack.sy, astack.rotDeg, astack.skewDeg
    
    
    
def testSet(nPts = 100, fBad=0.6, useWeights=True, unctyFac=10., \
                nBoots=100, fResample=1., showFig=True):

    """Try using these objects. We assume the two star catalogs are
    matched. This can be made more verbose by adding print statements
    but for the moment seems to work pretty well. Arguments:

    -- nPts = number of objects to simulate
    
    -- fBad = fraction of points with high-scatter measurements. (To
       make the fit squeak, set fBad to 0.8 or so.)

    -- unctyFac = factor by which the "bad" points have larger scatter
       than the "good" points.

    -- useWeights = use weights when fitting the deltas? (In the
       simulation, weights are inverse variance weights.)

    -- nBoots = number of bootstrap trials to do. (Set to 0 to not do
       bootstraps.)

    -- fResample = fraction resampling when doing bootstraps. (Set to
       0.5 to do half-sample bootstrap.)

    -- showFig = plot up a vector point errorbar of the simulation

    """

    # field center
    raCen1 = 270.
    deCen1 = -29.
    fovDE = 0.05
    fovRA = fovDE / np.cos(np.radians(deCen1))

    # separations, noise
    deltaRA = 10./3600.
    deltaDE = 5. /3600.

    # try only noise
    #deltaRA = 0.
    #deltaDE = 0.

    # create an array of uncertainties. We add a few with very large
    # uncertainties to see what this does
    sigm2Lo = 0.1/3600.
    sigm2Hi = sigm2Lo * unctyFac
    sig2 = np.repeat(sigm2Lo, nPts)
    nBad = int(fBad * nPts)
    sig2[0:nBad] = sigm2Hi
    
    # use the uncertainties to estimate weights
    CS = CovStack(sig2, sig2)
    wts = 1.0/CS.maxEigvals
    if not useWeights:
        wts = np.ones(nPts)

    # first list...
    ra1 = (np.random.uniform(size=nPts) - 0.5)*fovRA + raCen1
    de1 = (np.random.uniform(size=nPts) - 0.5)*fovDE + deCen1

    # ... and the second
    ra2 = ra1 + deltaRA + np.random.normal(size=nPts)*sig2
    de2 = de1 + deltaDE + np.random.normal(size=nPts)*sig2

    # OK now we set up our coordinate objects (unweighted for the
    # moment)
    coo1 = CooSet(ra1, de1)
    coo2 = CooSet(ra2, de2, wgts=wts)

    # and find the differences...
    CP = CooPair(coo1, coo2)
    CP.findDiffXYZ()

    cooOrig = CooSet(raCen1, deCen1)

    # now try correcting the original coords
    cooOrig.X += CP.avgDX
    cooOrig.Y += CP.avgDY
    cooOrig.Z += CP.avgDZ

    cooOrig.xyz2equat()
    
    print("Scalar orig pointing:", cooOrig.ra, cooOrig.de)

    # Now let's try our one-liner:
    predRA2, predDE2 = shiftAngleByXYZ(raCen1, deCen1, CP.dXYZ)

    print("One-liner result:", predRA2, predDE2)

    # compute the delta in arcsec
    dRA = (predRA2 - raCen1)*3600.
    dDE = (predDE2 - deCen1)*3600.

    if showFig:
        fig1 = plt.figure(1)
        fig1.clf()

        # Show the quantities as Delta ra. cos(delta) 
        #
        # (This is a nice feature of our doing things in cartesian
        # coordinates: we *never* have to worry whether we got the cos
        # delta right because we converted to cartesian coordinates
        # the first time out.)
        cd = np.cos(np.radians(deCen1))

        ax1 = fig1.add_subplot(111)
        dum1 = ax1.errorbar((ra2-ra1)*3600.*cd, (de2-de1)*3600., \
                                xerr=sig2*3600.*cd, yerr=sig2*3600., \
                                alpha=0.3, ls='None', marker='o', ms=2, \
                                ecolor='0.6')
        dum2 = ax1.scatter((ra2-ra1)*3600.*cd, (de2-de1)*3600., \
                               c=sig2*3600, zorder=5, alpha=0.3, cmap='viridis')
    
        xlim = np.copy(ax1.get_xlim())
        ylim = np.copy(ax1.get_ylim())
    
        cax = '#9D685B'
        chx = ax1.plot(np.repeat(dRA,2)*cd, ylim, color=cax, zorder=10, \
                           alpha=0.6)
        chy = ax1.plot(xlim, np.repeat(dDE,2), color=cax, zorder=10, \
                           alpha=0.6)

        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)

        cb = fig1.colorbar(dum2)

        ax1.set_xlabel(r'$\cos(\delta_0) \Delta \alpha$, arcsec')
        ax1.set_ylabel(r'$\Delta \delta$, arcsec')

    # Now try doing a bootstrap resampling...
    tZer = time.time()
    CB = CooBoots(CP, nBoots=nBoots, cenRA=raCen1, cenDE=deCen1, \
                      runOnInit=True, fResample=fResample)

    tElapsed = time.time() - tZer

    print("Full-set results:", dRA, dDE)
    print("Frac %.2f Bootstrap results:" % (CB.fResample), \
              CB.summCenter*3600., CB.summStddev*3600.)
    print("Did %i bootstraps in %.2e seconds for %i datapoints" \
              % (nBoots, tElapsed, nPts))


def testTP(sxIn=1.1, syIn=0.7, thetaDeg = 45., skewDeg=0., nPlanes=3, \
               asScalar=False):

    """Routine to test the tangent plane - spherical class"""

    sx = np.repeat(sxIn, nPlanes)
    sy = np.repeat(syIn, nPlanes)
    th = np.repeat(thetaDeg, nPlanes)
    sk = np.repeat(skewDeg, nPlanes)

    # testing what happens when the stack is fed single numbers?
    if asScalar:
        sx = sxIn
        sy = syIn
        th = thetaDeg
        sk = skewDeg

    ST = Stack2x2(None, sx, sy, th, sk)

    # now try converting from the stack to the params, see if it
    # worked...

    print(ST.A[0])

    b = ST.A[0,0,0]
    c = ST.A[0,0,1]
    e = ST.A[0,1,0]
    f = ST.A[0,1,1]

    ST.parsFromStack()

    # we need to make the supplied values vectors for printing if
    # asScalar. We deliberately do NOT use copyVec() here.
    if asScalar:
        sx = np.array([sx])
        sy = np.array([sy])
        th = np.array([th])
        sk = np.array([sk])
        

    for i in range(np.size(sx)):
        print("SX: %.3f, %.3f -- SY: %.3f, %.3f -- TH: %.3f, %.3f -- SK: %.3f, %.3f"  \
                  % (sx[i], ST.sx[i], sy[i], ST.sy[i], th[i], ST.rotDeg[i], sk[i], ST.skewDeg[i]))


def testPlane(nPlanes=5, dxIn=0., dyIn=0., sxIn=1.1, syIn=0.7, \
                  thetaDeg = 45., skewDeg=0., asScalar=False, nPoints=5, \
                  scrambleRotations=False, scrambleOffsets=False):

    """Constructs source and transformed catalogs to test the various
    pieces for plane-to-plane transformations"""

    # Written initially to support sending in one transformation at a
    # time.

    # We'll start simple... generate a set of transformations and make
    # sure our LinearMapping object handles it correctly. This borrows
    # methods from testTP().
    
    sx = np.repeat(sxIn, nPlanes)
    sy = np.repeat(syIn, nPlanes)
    th = np.repeat(thetaDeg, nPlanes)
    sk = np.repeat(skewDeg, nPlanes)

    dx = np.repeat(dxIn, nPlanes)
    dy = np.repeat(dyIn, nPlanes)

    if scrambleRotations:
        th = np.random.uniform(size=nPlanes)*360.-180.

    if scrambleOffsets:
        dx = np.random.uniform(size=nPlanes)*10.-5.
        dy = np.random.uniform(size=nPlanes)*2.-1.

    if asScalar:
        nPlanes = 1
        sx = sxIn
        sy = syIn
        th = thetaDeg
        sk = skewDeg
        dx = dxIn
        dy = dyIn
        nPoints = 1

    # create the 2x2 stack...
    ST = Stack2x2(None, sx, sy, th, sk)

    # ... and the dx array...
    aNx2x2 = ST.A
    vXx2 = np.vstack(( dx, dy)).T

    aNx2x3 = np.dstack(( vXx2, aNx2x2 ))
    
    print(aNx2x3)
    print(np.shape(aNx2x3))

    # OK now send this as an input into our LinearMapping object
    LM = LinearMapping(aNx2x3)

    print(LM.squares)
    print(LM.consts, np.shape(LM.consts))
    print(LM.stackSquares.rotDeg)
    print(LM.stackSquares.skewDeg)

    # Now generate a set of positions and apply the linear
    # transformation to them
    vX = np.random.uniform(size=nPoints)
    vY = np.random.uniform(size=nPoints)

    if asScalar:
        vX = vX[0]
        vY = vY[0]

    # return the Nx2x3 stack so that other test routines can use them
    #return LM.stackSquares
    vXi, vEta = LM.applyLinearPlane(vX, vY)

    # slight fudge - if we chose asScalar, we put vX, vY back into an
    # array form so that the test will work
    if asScalar:
        vX = np.array([vX])
        vY = np.array([vY])

    # Now we check whether the broadcasting did what we think it did
    Aplane = LM.squares[0]
    vOff = LM.consts
    for i in range(np.min([5,nPoints])):
        xiDir = Aplane[0,0]*vX[i] + Aplane[0,1]*vY[i] + vOff[0,0]
        etDir = Aplane[1,0]*vX[i] + Aplane[1,1]*vY[i] + vOff[0,1]
        print("PLANE test: %.3f, %.3f --> %.3f, %.3f and %.3f, %.3f" % \
                  (vX[i], vY[i], vXi[i], vEta[i], xiDir, etDir))

    # Now we try a different test: we apply the transformation
    # plane-by-plane and view the results of sending the input data
    # thru each plane at a time. 
    planeXi, planeEta = LM.applyLinear(vX, vY)
    
    # Multiplying plane-by-plane only makes sense if there are the
    # same number of datapoints as there are planes. So check that
    # here.
    if nPlanes == nPoints:
        print("===============")
        for i in range(nPlanes):
            Athis = LM.squares[i]
            vThis = LM.consts[i]

            xiThis = Athis[0,0]*vX[i] + Athis[0,1]*vY[i] + vThis[0]
            etThis = Athis[1,0]*vX[i] + Athis[1,1]*vY[i] + vThis[1]

            print("STACK test: %.3f, %.3f --> %.3f, %.3f and %.3f, %.3f" % \
                      (vX[i], vY[i], planeXi[i], planeEta[i], xiThis, etThis))


def testMakingSample(nPts = 20, rotDeg=30., sf=1., nReplicas=2000, \
                         testComponents=True):

    """Tests generating a sample with non-aligned covariances"""

    # Many of the methods have been moved into the class CovarsNx2x2

    CN = CovarsNx2x2(nPts=nPts, rotDeg=rotDeg)
    CN.generateCovarStack()
    CN.generateSamples()
    # CN.showDeltas()
    
    # OK now we set up a second covarstack object, this time with the
    # covariances
    CF = CovarsNx2x2(CN.covars)
    CF.eigensFromCovars()

    # Populate the transformation matrices so that we can generate
    # points that follow these ellipses
    CF.populateTransfsFromCovar()

    # Now we generate points at these locations...
    nPts = np.size(CF.rotDegs)
    xGen = np.random.uniform(-10., 10., size=nPts)
    yGen = np.random.uniform(-10., 10., size=nPts)

    # now we generate replicas of the points but offset by these
    # covariances.
    xRep = np.array([])
    yRep = np.array([])
    cRep = np.array([])
    
    for iRep in range(nReplicas):

        # Generate a replica set
        CF.generateSamples()
        xRep = np.hstack(( xRep, CF.deltaTransf[0] + xGen)) 
        yRep = np.hstack(( yRep, CF.deltaTransf[1] + yGen )) 
        cRep = np.hstack(( cRep, CF.rotDegs ))

    # TEST THE REFACTORING IF OUR COVAR ERRPLOT
    fig1 = plt.figure(1, figsize=(5,4))
    fig1.clf()
    ax1 = fig1.add_subplot(111)

    # We add a scatterplot 
    alph = 10.0**(3.-np.log10(np.size(xRep)) )
    alph = np.float(np.clip(alph, 0.06, 1.0))
    dumScat = ax1.scatter(xRep, yRep, c=cRep, zorder=1, s=1, \
                              alpha=alph, \
                              edgecolor='0.3')

    # uncomment this to test the sign-retention when coverrplot's
    # uniform axis length preservation is used.
    # ax1.set_xlim(3,-3)
    
    # Plot arguments moved into new method:
    if not testComponents:
        coverrplot(xGen, yGen, CF, errSF=sf, ax=ax1, fig=fig1)
    else:
        # Try this with the xerr, yerr, corrxy. To ignore one or both
        # of those, set the erry, corrxy to None.
        coverrplot(xGen, yGen, None, errx=CF.stdx, \
                       erry=CF.stdy, \
                       corrxy=CF.corrxy, \
                       errSF=sf, ax=ax1, fig=fig1)
    

    # We add a scatterplot 
    #alph = 10.0**(3.-np.log10(np.size(xRep)) )
    #alph = np.float(np.clip(alph, 0.06, 1.0))
    #dumScat = ax1.scatter(xRep, yRep, c=cRep, zorder=1, s=1, \
    #                          alpha=alph, \
    #                          edgecolor='0.3')

    # label the axes
    ax1.set_xlabel(r'X')
    ax1.set_ylabel(r'Y')

    ax1.set_title('Covariances and samples: %i sets of %i points' \
                      % (nReplicas, np.size(xGen)), fontsize=10)

    ax1.grid(which='both', visible=True, zorder=0, alpha=0.5)

    return


    #print(np.shape(CN.deltaTransf))
    return
    
    # Covariances: First we generate a set of major and minor axes
    aLo = 1.0
    aHi = 4.0
    ratLo = 0.6  # ratio of minor:major axes (so that the axes don't
                 # swap)
    ratHi = 1.0
    ratios = np.random.uniform(ratLo, ratHi, size=nPts)

    vMajors = np.random.uniform(aLo, aHi, size=nPts)
    vMinors = vMajors * ratios

    # This allows us to generate the diagonal-covariances array...
    VV = diagStack2x2(vMajors, vMinors)

    # Now we generate rotation angles for the covariance matrices
    rotDegs = np.repeat(rotDeg, np.size(vMajors))
    RR = rotStack2x2(rotDegs, rotateAxes=False)

    # Now we combine the two to get the stack of covariance matrices
    CC = AVAt(RR,VV)

    print(np.matmul(RR,VV))
    #print(CC)

def testFitting(nPts = 20, rotDegCovar=30., \
                    rotDeg = 30., \
                    genStripe=True, \
                    xDetMax=500., yDetMax=500., \
                    xRef = 250., yRef=250.):

    """End-to-end test of sample generation and fitting."""

    # Generate the centroids in the DETECTOR plane
    xGen = np.random.uniform(0., xDetMax, size=nPts)
    yGen = np.random.uniform(0., yDetMax, size=nPts)

    # Set up the transformation. For the moment this is still arranged
    # as the A-matrix
    sx = -5.0e-4
    sy = 4.0e-4
    #rotDeg = 30.
    skewDeg = 5.
    xiRef  = 0.05
    etaRef = -0.06

    # Convert the transformation into abcdef pars and slot into the
    # 1D order [a,b,c,d,e,f]
    Transf = Stack2x2(None, sx, sy, rotDeg, skewDeg)
    vTheta = np.hstack(( xiRef, Transf.A[0,0,:], etaRef, Transf.A[0,1,:] )) 

    # Armed with this, populate the true parameters array in the order
    # in which it is expected by the P.theta object
    PT = ptheta2d(xGen-xRef, yGen-yRef, pars=vTheta)

    # Populate the target objects
    xiTrue = PT.evaluatePtheta()

    # Now, we give these objects measurement uncertainty. We'll start
    # off assuming that our uncertainties are correct.
    CN = CovarsNx2x2(nPts=nPts, rotDeg=rotDegCovar, \
                         genStripe=genStripe, \
                         aLo=1.0e-3, aHi=2.0e-2)
    CN.generateCovarStack()

    # It will be useful to produce catalog-like uncertainties
    stdXiGen = CN.stdx
    stdEtaGen = CN.stdy
    corrXiEtaGen = CN.corrxy

    # Now we create a second object, this time with the covars as an
    # input, and use this to draw samples
    CF = CovarsNx2x2(CN.covars)
    CF.populateTransfsFromCovar()

    t0 = time.time()
    CF.generateSamples()

    # tweak the generated points in the target frame by adding the
    # deltas to them
    xiNudged  = xiTrue[:,0] + CF.deltaTransf[0] # Yes, mismatched...
    etaNudged = xiTrue[:,1] + CF.deltaTransf[1]

    # To test the different forms of covariance input, uncomment the
    # relevant lines in the FNE call.
    FNE = FitNormEq(xGen, yGen, xiNudged, etaNudged, \
                        covars=CF.covars, W=np.array([]), \
                        #covars=np.array([]), W=np.array([]), \
                        #stdxi=stdXiGen, stdeta=stdEtaGen, \
                        #corrxieta=corrXiEtaGen, \
                        xRef=xRef, yRef=yRef, invertHessian=True, \
                        Verbose=True)

    # OK now we try the normal equations to see how the fit goes...
    #NE = NormalEqs(xGen, yGen, xiNudged, etaNudged, \
    #                   W=np.linalg.inv(CF.covars), \
    #                   xref=xRef, yref=yRef)

    #NE.doFit()
    print("elapsed in simulation + fit: %.2e sec" % (time.time()-t0)) 

    # Estimate the tangent point in the original coordinates, invert
    # the Hessian to get the formal covariance estimate
    #NE.estTangentPoint()
    #NE.invertHessian()

    print("Tangent point:", FNE.NE.xZero)

    # what does the covariance matrix of just the positions look like?
    covPos = np.array([ FNE.NE.formalCov[0,[0,3]], FNE.NE.formalCov[3,[0,3]] ])

    # Let's try putting this into a 2x2 stack for easy (and
    # convention-following) retrieval of the covariance parameters for
    # the position
    CovPos2x2 = CovarsNx2x2(covPos[np.newaxis,:])
    CovPos2x2.eigensFromCovars()

    print("Formal covariance for xiRef: %.3e, %.3e, %.2f" % \
              (np.sqrt(CovPos2x2.majors), \
                   np.sqrt(CovPos2x2.minors), \
                   CovPos2x2.rotDegs))

    #print(NE.formalCov)
    #print(covPos)

    # Debug plots for our sample follow below    

    fig2=plt.figure(2)
    fig2.clf()
    ax1 = fig2.add_subplot(121)
    ax2 = fig2.add_subplot(122)

    dum1 = ax1.scatter(xGen, yGen, c='r', s=3, edgecolor='0.5')
    ##dum2 = ax2.scatter(xiTrue[:,0], xiTrue[:,1], c='b', s=3)

    # Now show the target coords as perturbed
    dum3 = ax2.scatter(xiNudged, etaNudged, c='0.5', s=2)

    # Now try an errorplot of this
    coverrplot(xiNudged, etaNudged, covars=CF, fig=fig2, ax=ax2, \
                   showColorbarEllipse=True, \
                   cmapEllipse='viridis', edgecolorEllipse='k', \
                   colorMajors='k', colorMinors='0.1', \
                   shadeEllipses=False, crossStyle=True, \
                   showEllipses=True, \
                   alphaEllipse=0.15)

    # Show the tangent point on the plots
    dum4 = ax1.plot(FNE.NE.xZero[0], FNE.NE.xZero[1], 'go')
    dum5 = ax2.plot(0., 0., 'go')

    ax1.set_xlabel('X, pix')
    ax1.set_ylabel('Y, pix')
    ax2.set_xlabel(r'$\xi$, degrees')
    ax2.set_ylabel(r'$\eta$, degrees')

    fig2.subplots_adjust(wspace=0.3, bottom=0.15)

    # add the grid?
    for ax in [ax1, ax2]:
        ax.grid(which='both', visible=True, alpha=0.3)

def testFitOO(nPts=50, resetPositions=False, nTrials=3, skewDeg=5., \
                  testNonparam=True, fNonparam=1.0, showPoints=False, \
                  genStripe=True, \
                  stripeAxRatio=1.0, \
                  stripeFrac = 0.5, posnSortCol='', \
                  fitChoice='6term', \
                  simGauss=False, \
                  simSx = -5.0e-4, simSy=4.0e-4, \
                  parFile='', errscaleplot=1.):

    """Tests fitting with the class NormWithMC"""

    # close any matplotlib figures that are open (because the version
    # of corner on my laptop doesn't allow passing a blank matplotlib
    # figure).
    plt.close('all')

    NMC = NormWithMonteCarlo(simNpts=nPts, nTrials=nTrials, \
                                 simSkewDeg=skewDeg, \
                                 stripeCovRatio=stripeAxRatio, \
                                 genStripe=genStripe, \
                                 stripeFrac=stripeFrac, \
                                 posnSortCol=posnSortCol, \
                                 simMakeGauss=simGauss,  \
                                 fitChoice=fitChoice, \
                                 simSx=simSx, simSy=simSy, \
                                 Verbose=True, \
                                 parFile=parFile)
    
    # Create a synthetic dataset
    NMC.transfParsAsVecs()
    NMC.generateRawXY()
    NMC.populateRawXiEta()
    NMC.sortRawPositions() # note that the sorting does nothing if no
                        # position sort column has been supplied.

    NMC.populateCovarsFromSim()
    NMC.populateUnperturbedFitObj()

    # show the points?
    if showPoints:
        coverrplot(NMC.xiRaw, NMC.etaRaw, NMC.CF, \
                   errSF=errscaleplot, \
                       xLabel=r'$\xi$, deg', yLabel=r'$\eta$, deg')

    # The first time, we assume the perturbed fit object constitutes
    # the "data". This mimics the case of feeding the object our
    # matched catalogs
    NMC.populatePerturbedFitObj()
    NMC.copyPerturbedSimToData()
    NMC.fitData()

    # 2023-06-15 were the uncertainties populated?
    #print("First plane INFO:", NMC.FitData.covars.shape)
    #print("First plane INFO:", NMC.FitData.W)
    
    # dump the parameters to disk
    NMC.Verbose = True

    # uncomment the following to test writing the parameters to disk...
    # print("Dumping parameters to disk")
    NMC.writeParfile()
    
    # uncomment the following to test reading parameters FROM disk
    # NMC.readPars('tmp_mcparams.txt')

    ### print("Round 1 - Pars fit from data:", NMC.parsData)
    ### print("Round 1 - Pars simulated:", NMC.simTheta)

    # If we are doing nonparametric bootstrapping, then we can
    # simulate with the same object
    if testNonparam:
        # now set in the input param file
        #NMC.nTrials = nTrials  
        #NMC.doFewWeightings=True
        #NMC.fNonparam = fNonparam

        t1 = time.time()
        NMC.doNonparamBootstrap()
        print("%i Nonparam bootstraps took %.2e sec" \
                  % (NMC.nTrials, time.time()-t1))

        if nTrials < 10:
            print(NMC.stackTrials.transfs2x2.rotDeg)
            print(NMC.stackTrialsDiag.transfs2x2.rotDeg)
            print(NMC.stackTrialsUnif.transfs2x2.rotDeg)
            return

        # show corner plot?
        ### print("INFO:", NMC.simTheta)
        NMC.showCornerPlot('stackTrials', fignum=2)
        NMC.showCornerPlot('stackTrialsUnif', fignum=3)
        return
    

    # OK now we have our "data", and parameters we have got from
    # fitting the data, let's test the case of setting up a second
    # object with the data as input. We'll try a parametric monte
    # carlo case, in which the model is fit to the data, then those
    # parameters are used to generate draws from the same model.
    
    # 2020-07-10 WATCHOUT - this is now somewhat different frmo the
    # NMC object in some of the arguments. COME BACK TO THIS.
    MC = NormWithMonteCarlo(NMC.x, NMC.y, NMC.xi, NMC.eta, \
                                NMC.stdxi, NMC.stdeta, NMC.corrxieta, \
                                xref=NMC.xRef, yref=NMC.yRef, \
                                simParsVec=NMC.simTheta, \
                                nTrials=nTrials, \
                                fitChoice=NMC.fitChoice, \
                                resetPositions=resetPositions)
    MC.setSimRangesFromData()
    MC.populateCovarsFromData()

    # OK now we set up the unperturbed object and perturb it for the
    # monte carlo
    MC.generateRawXY()
    MC.populateRawXiEta()
    MC.populateUnperturbedFitObj()

    # now do the monte carlo
    t1 = time.time()
    MC.doMonteCarlo()
    print("%i parametric bootstraps took %.2e sec" \
              % (nTrials, time.time()-t1))

    # MC.stackTrials.convertParsToGeom()

    ##print(MC.stackTrials.parsTrials[:,1])
    if nTrials < 10:
        print(MC.stackTrials.transfs2x2.rotDeg)
        print(MC.stackTrialsDiag.transfs2x2.rotDeg)
        print(MC.stackTrialsUnif.transfs2x2.rotDeg)
        return

    # show corner plot?
    print("INFO:", np.size(MC.stackTrials.transfs2x2.rotDeg))
    MC.showCornerPlot('stackTrials')
    return


    #print(MC.stackTrials.transfs2x2rev.rotDeg)

    ##print(MC.stackTrialsResampled.transfs2x2.rotDeg)


def demoTranslatedGaussians(nPts=1000, skewDeg=20., stdx=6., stdy=2., \
                                plotLim=20., rotDeg=0., sx=1., sy=1.):

    """Simulates a single Gaussian in one frame, performs a frame
    transformation, and shows the distributions in the two frames."""

    # Example call:
    # 
    # weightedDeltas.demoTranslatedGaussians(skewDeg=-30., stdy=4, nPts=10000)

    # Create the covariance matrix in the original frame
    C1 = CovarsNx2x2(stdx=stdx, stdy=stdy, corrxy=0.)

    # Now generate a large sample of these points
    sampl = np.random.multivariate_normal(mean=np.zeros(2), cov=C1.covars[0], \
                                              size=nPts)

    # Now we generate the transformation
    TRANSF = Stack2x2(sx=sx, sy=sy, rotDeg=rotDeg, skewDeg=skewDeg)

    # Now apply the transformation to produce the transformed
    # positions
    newSampl = np.matmul(TRANSF.A[0], sampl.T)

    # numpy of course has a convenient method to compute the
    # covariance matrix from data... So we'll insert this into our
    # covarsNx2x2 object and uses its methods to find the eigenvalues
    # and eigenvectors.
    covUV = np.cov(newSampl[0], newSampl[1])
    covUV = covUV[np.newaxis, :, :]

    CT = CovarsNx2x2(covars=covUV)
    CT.eigensFromCovars()
    
    # Consider moving this into Covars2x2 as a method. WATCHOUT - why
    # are the eigenvectors coming out squared here but not in
    # coverrplot?
    uMajors = CT.axMajors[:,0]*CT.majors**0.5
    vMajors = CT.axMajors[:,1]*CT.majors**0.5

    uMinors = CT.axMinors[:,0]*CT.minors**0.5
    vMinors = CT.axMinors[:,1]*CT.minors**0.5 

    fig = plt.figure(1)
    fig.clf()
    ax1=fig.add_subplot(121)
    ax2=fig.add_subplot(122)

    dum1 = ax1.scatter(sampl[:,0], sampl[:,1], alpha=0.5, s=.1, c='0.3', zorder=2)
    dum2 = ax2.scatter(newSampl[0], newSampl[1], alpha=0.5, s=.1, c='0.4', zorder=2)

    # now show the eigenvectors

    ## ORIGINAL
    C1.eigensFromCovars()
    xMajors = C1.axMajors[:,0]*C1.majors**0.5
    yMajors = C1.axMajors[:,1]*C1.majors**0.5

    xMinors = C1.axMinors[:,0]*C1.minors**0.5
    yMinors = C1.axMinors[:,1]*C1.minors**0.5

    dumMaj1 = ax1.quiver(0., 0., xMajors, yMajors, zorder=8, \
                            units='xy', angles='xy', scale_units='xy', \
                            scale=1., \
                            width=0.05*np.median(uMajors), headwidth=2)
    dumMin1 = ax1.quiver(0., 0., xMinors, yMinors, zorder=8, \
                            units='xy', angles='xy', scale_units='xy', \
                            scale=1., \
                            width=0.05*np.median(uMajors), headwidth=2)


    ecXY = EllipseCollection(C1.majors**0.5*2., \
                                 C1.minors**0.5*2, \
                                 C1.rotDegs, \
                                 units='xy', offsets=[0.,0.], \
                                 transOffset=ax1.transData, \
                                 alpha=0.5, \
                                 edgecolor='k', \
                                 facecolor='r', \
                                 cmap='Reds', \
                                 zorder=5)

    ax1.add_collection(ecXY)
    

    ## TRANSFORMED
    dumMaj = ax2.quiver(0., 0., uMajors, vMajors, zorder=8, \
                            units='xy', angles='xy', scale_units='xy', \
                            scale=1., \
                            width=0.05*np.median(uMajors), headwidth=2, \
                            label='U,V major, minor axes')

    dumMin = ax2.quiver(0., 0., uMinors, vMinors, zorder=8, \
                            units='xy', angles='xy', scale_units='xy', \
                            scale=1., \
                            width=0.05*np.median(uMajors), headwidth=2)
    

    ecUV = EllipseCollection(CT.majors**0.5*2., \
                                 CT.minors**0.5*2., \
                                 CT.rotDegs, \
                                 units='xy', offsets=[0.,0.], \
                                 transOffset=ax2.transData, \
                                 alpha=0.5, \
                                 edgecolor='k', \
                                 cmap='gray', \
                                 zorder=5)

    ax2.add_collection(ecUV)

    # Now try transforming the eigenvectors to the destination frame
    xyMajors = np.column_stack(( xMajors, yMajors ))
    xyMinors = np.column_stack(( xMinors, yMinors ))
    
    uvMajors = np.matmul(TRANSF.A, xyMajors.T)[0,:,:]
    uvMinors = np.matmul(TRANSF.A, xyMinors.T)[0,:,:]

    ## print np.shape(uvMajors)

    dumMajTransf = ax2.quiver(0., 0., uvMajors[0], uvMajors[1], zorder=7, \
                                  units='xy', angles='xy', scale_units='xy', \
                                  scale=1., \
                                  width=0.05*np.median(uMajors), headwidth=2, \
                                  color='r', label='Transformed X, Y major, minor axes')

    dumMinTransf = ax2.quiver(0., 0., uvMinors[0], uvMinors[1], zorder=7, \
                                  units='xy', angles='xy', scale_units='xy', \
                                  scale=1., \
                                  width=0.05*np.median(uMajors), headwidth=2, \
                                  color='r')

    # under-plot the axes
    xyHorizX = np.array([-plotLim*2., plotLim*2.])
    xyHorizY = np.array([0., 0.])
    xyVertX = np.array([0., 0.])
    xyVertY = np.array([-plotLim*2., plotLim*2.])

    # stack the coords together for transformation
    xyHorizXY = np.column_stack(( xyHorizX, xyHorizY ))
    xyVertXY = np.column_stack(( xyVertX, xyVertY ))

    # transform them
    uvHorizXY = np.matmul(TRANSF.AINV[0], xyHorizXY.T)
    uvVertXY = np.matmul(TRANSF.AINV[0], xyVertXY.T)

    uvHorizX = uvHorizXY[0]
    uvHorizY = uvHorizXY[1]
    uvVertX = uvVertXY[0]
    uvVertY = uvVertXY[1]
    
    # original axes
    ax1.plot(xyHorizX, xyHorizY, 'r-', lw=1, zorder=1)
    ax1.plot(xyVertX, xyVertY, 'r-', lw=1, zorder=1)

    ax1.plot([0.,xyHorizX[-1]], [0., xyHorizY[-1]], 'r-', lw=2, zorder=1)
    ax1.plot([0.,xyVertX[-1]], [0.,xyVertY[-1]], 'r-', lw=2, zorder=1)

    ax1.plot(uvHorizX, uvHorizY, 'b-', lw=1, ls='--', zorder=1)
    ax1.plot(uvVertX, uvVertY, 'b-', lw=1, ls='--', zorder=1)

    # do the bold upper-right quadrant
    ax1.plot([0., uvHorizX[-1]], [0.,uvHorizY[-1]], 'b-', lw=2, ls='--', zorder=1)
    ax1.plot([0.,uvVertX[-1]], [0., uvVertY[-1]], 'b-', lw=2, ls='--', zorder=1)

    # Now do the transformed axes in the transformed space
    ax2.plot(xyHorizX, xyHorizY, 'b-', lw=1, ls='--', zorder=1)
    ax2.plot(xyVertX, xyVertY, 'b-', lw=1, ls='--', zorder=1)

    ax2.plot([0.,xyHorizX[-1]], [0., xyHorizY[-1]], 'b-', lw=2, ls='--', zorder=1)
    ax2.plot([0.,xyVertX[-1]], [0.,xyVertY[-1]], 'b-', lw=2, ls='--', zorder=1)

    for ax in [ax1, ax2]:
        ax.set_xlim(-plotLim, plotLim)
        ax.set_ylim(-plotLim, plotLim)
        ax.set_aspect('equal')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax2.set_xlabel('U')
    ax2.set_ylabel('V')

    # show the legend
    leg2=ax2.legend(loc=0, fontsize=8)
    
    ax1.set_title(r'X,Y data ("1$\sigma$" ellipse, eigenvectors shown)', fontsize=10)
    ax2.set_title('U,V = A.(X, Y)', fontsize=10)

    sOtitl = 'X,Y -->  U,V by skew matrix only: each axis rotated (in opposite directions) by %i degrees' % (skewDeg/2.)

    fig.suptitle(sOtitl, fontsize=10)
    fig.subplots_adjust(top=0.80)

    fig.savefig('2020-06-27_exampleSkewOnly.jpg', rasterized=True)

def fitAndBootstrap(parFile='inp_mcparams.txt'):

    """Simulates, fits, and does bootstrap trials, using input
    parameter file to control everything"""

    # Close all open plot windows (I find this plays better with the
    # version of corner.py on my laptop):
    plt.close('all')

    MC = NormWithMonteCarlo(parFile=parFile)    

    if len(parFile) < 1:
        print("fitAndBootstrap INFO - dumping parfile to disk")
        MC.writeParfile()
        return

    MC.setupAndFit()
    print("INFO - formal Hessian^{-1}:", MC.FitData.NE.formalCov)


    MC.setupAndBootstrap()
    MC.writeSummaryStats()

    # Try a comparison in terms of the linear params
    compCov = np.cov(MC.stackTrials.parsTrials, rowvar=False)
    print("INFO:", np.shape(MC.stackTrials.parsTrials), np.shape(compCov))
    print("INFO - recovered covariance:", compCov)

    MC.showPositions()

    MC.plotCovarPairs()

    #MC.setStackPlotInfo()

    # Try looping thru:
    #clobberFig = True
    #MC.showCovarPairs(MC.stackTrials, newFig=True)

    #for sObj in [MC.stackTrialsDiag, MC.stackTrialsUnif]:
    #    MC.showCovarPairs(sObj, newFig=False)


    # 2020-07-20 remove this line for speed while developing
    # MC.plotCorners()
    

def testMCMC(parFile='inp_mcparams.txt', showPoints=True, nchains=32, \
             errscalefac=10., showDataTransf=True):

    """Sets up input and output datasets and uses MCMC"""

    # This follows the model fitting example in the emcee manual:
    #
    # https://emcee.readthedocs.io/en/stable/tutorials/line/
    
    # Close all open plot windows (I find this plays better with the
    # version of corner.py on my laptop):
    plt.close('all')
    
    # set up the datasets
    MC = NormWithMonteCarlo(parFile=parFile)

    MC.prepSimData()
    
    # dump parameter file to disk (allowing some level of lookback
    # when tinkering)
    MC.writeParfile()

    # Create covariance stack for measurement uncertainties in the XY
    # plane. Later we'll want this to track in some sense the xieta
    # uncertainties on the grounds that brighter objects would be
    # better-measured in both catalogs. For now, we make them random.
    CVD = CovarsNx2x2(nPts=MC.xRaw.size, aLo=1., aHi=5.0, \
                      ratLo=0.7, ratHi=0.7, genStripe=False, rotDeg=0.)
    CVD.generateCovarStack()

    # Try transforming these into covariances in (xi, eta)
    covsDataProj = likesMCMC.propagate_covars_abc(CVD.covars, MC.simTheta)

    print("SIM PARS:", MC.simTheta)
    
    # invert the covariance matrix stack for convenient call in MCMC
    covinv = np.linalg.inv(MC.CF.covars)

    # Abut the positions into [N,2] element arrays
    xy = np.column_stack((MC.xRaw, MC.yRaw))
    xieta = np.column_stack((MC.xiRaw, MC.etaRaw))

    # Create perturbed xi, eta out of the uncertainties
    MC.CF.generateSamples()
    xiObs = xieta + MC.CF.deltaTransf.T
    #print("Perturbations:", MC.CF.deltaTransf.shape)

    # NOW apply outliers
    xiObs += MC.dxiOutlier
    
    # Create the pattern matrix for the linear transformation
    PATT = ptheta2d(MC.xRaw - MC.xRef, \
                    MC.yRaw - MC.yRef, \
                    MC.fitChoice, \
                    xSign = 1.0-MC.flipx)

    # 2023-06-20 Transform the data points into xi, eta and do a debug
    # plot showing the transformed datapoints and their uncertainties
    xiDataProj = np.matmul(PATT.pattern, MC.simTheta)
    if showDataTransf:
        fig7 = plt.figure(7, figsize=(10,5))
        fig7.clf()
        ax7a = fig7.add_subplot(121)
        ax7b = fig7.add_subplot(122)

        coverrplot(MC.xRaw, MC.yRaw, CVD, errSF=errscalefac, \
                   ax=ax7a, fig=fig7, showColorbarEllipse=False)

        # Create a covars stack out of the covars array (really should
        # streamline this within coverrplot!):
        CVP = CovarsNx2x2(covsDataProj)
        w, v = np.linalg.eigh(CVP.covars)
        print("INFO:", w[:,1].min(), w[:,1].max())
        #CVP.eigensFromCovars()
        #print("INFO:", CVP.majors.min(), CVP.majors.max(), MC.CF.majors.max())

        ## try eigens before and after eigensFromCovars
        #w, v = np.linalg.eigh(CVD.covars)
        #print("PRE: ", w[:,1].min(), w[1,:].max())
        #print("PRE: ", CVD.majors.min(), CVD.majors.max())
        #CVD.eigensFromCovars()
        #CVD.populateTransfsFromCovar()
        #print("POST:", CVD.majors.min(), CVD.majors.max())

        CVP.eigensFromCovars()
        CVP.populateTransfsFromCovar()

        # What are the eigenvalues of the transformed covariances? how
        # do they compare to the xieta covariances?
        print("RAW:", np.linalg.eigvals(CVD.covars[0])**0.5)
        print("TRANSFORMED:", np.linalg.eigvals(CVP.covars[0])**0.5)
        print("GENERATED:", np.linalg.eigvals(MC.CF.covars[0])**0.5)

        # major, minor axes are indeed the sf**2
        print("RAW majors:", CVD.majors.max(), CVD.majors.min() )
        print("TRANSFORMED majors:", CVP.majors.max(), CVP.majors.min() )
        
        print("abcdef", MC.simTheta)
        print("pars", MC.simXiRef, MC.simEtaRef, MC.simSx, MC.simSy, MC.simRotDeg, MC.simSkewDeg)
        
        dum = ax7b.scatter(xiDataProj[:,0], xiDataProj[:,1], s=3, zorder=25)

        # MC.simSx appears to be being applied twice - as squared not
        # as factor. Dividing errSF by the simSx (for symmetric scale
        # factors) makes the error ellipses scale correctly in the
        # plot of the transformed coordinates. So the relative
        # positions are being scaled correctly, the covariances
        # aren't.

        print(CVP.stdx.max(), CVP.stdx.min())
        
        coverrplot(xiDataProj[:,0], xiDataProj[:,1], CVP, \
                   errSF=errscalefac,\
                   ax=ax7b, fig=fig7, showColorbarEllipse=False)
        
        # Commented out the return so we'll have a VPD at the end for comparison
        return
    
    print("MINMAX INFO:", MC.xRaw.min(), MC.xRaw.max(), MC.yRaw.min(), MC.yRaw.max(), MC.xRef, MC.yRef)
    
    # Try evaluating loglike for a random set of parameters, just to
    # ensure the syntax works... For starters we use the true
    # parameters since we know that the deltas should then be small...
    parsTruth = MC.simTheta
    parsTruthInp = np.array([MC.simXiRef, MC.simEtaRef, \
                             MC.simSx, MC.simSy, MC.simRotDeg, MC.simSkewDeg])
    

    # perturb parsTruth into an initial guess
    parsGuess = parsTruth * np.random.uniform(0.8, 1.2, parsTruth.size)

    # Now we try finding the point estimate, pretending we don't know
    # how to do this test case by linear algebra methods...
    ufunc = lambda *args: -likesMCMC.loglike_linear(*args)
    soln = minimize(ufunc, parsGuess, args=(PATT.pattern, xiObs, covinv))

    print(soln.x)
    print(parsTruth)

    print("log(post): ", likesMCMC.logprob_linear_unif(parsGuess, PATT.pattern, xiObs, covinv))

    # follow dfm's example of "initializing the walkers in a tiny
    # Gaussian ball around the maximum likelihood result"
    pos = soln.x + 1.0e-3 * np.random.randn(nchains, soln.x.size)
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, \
                                    likesMCMC.logprob_linear_unif, \
                                    args=(PATT.pattern, xiObs, covinv))

    sampler.run_mcmc(pos, 5000, progress=True);

    # Determine the autocorrelation time and get "flat" samples
    tau = sampler.get_autocorr_time()
    tauto = tau.max()
    nThrow = int(tauto * 3)
    nThin = int(tauto * 0.5)
    flat_samples = sampler.get_chain(discard=nThrow, thin=nThin, flat=True)

    # Now convert these samples into "human" parameters
    sx, sy, theta, beta = \
        sixtermToPars(flat_samples[:,1], flat_samples[:,2], \
                      flat_samples[:,4], flat_samples[:,5])
    #print("FLAT SAMPLES:", flat_samples.shape, sx.shape)

    flat_human = np.column_stack(( flat_samples[:,0], flat_samples[:,3], \
                                   sx, sy, theta, beta ))

    print(flat_samples.shape, flat_human.shape, parsTruthInp.shape)
    
    labelsh = [r"$\Delta\xi$", r"$\Delta \eta$", r"$s_x$", r"$s_y$", \
               r"$\theta$", r"$\beta$"]

    # labelsh = ['dx', 'dy', 'sx', 'sy', 'theta', 'beta']
    
    # Let's take a look at the samples...
    samples = sampler.get_chain()
    labels = ["a", "b", "c", "d", "e", "f"]
    
    print(samples.shape)

    fig3, axes = plt.subplots(ndim, sharex=True, num=3)
    
    for iax in range(ndim):
        ax = axes[iax]
        dum = ax.plot(samples[:,:,iax], "k", alpha=0.2)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[iax])
        dum2 = ax.axvline(nThrow, color='g', ls='--', lw=2)
    axes[-1].set_xlabel("step number")


    # Now do the corner plot with the "truth" values overplotted
    fig4 = plt.figure(4, figsize=(9,7))
    fig4.clf()

    # Parameters as abcdef
    #dum = corner.corner(flat_samples, labels=labels, truths=parsTruth, \
    #                    truth_color='b', fig=fig4,
    #                    labelpad=0.7, use_math_test=True )

    # parameters as [dxi, deta, sx, sy, theta, beta]
    dum = corner.corner(flat_human, labels=labelsh, truths=parsTruthInp, \
                        truth_color='b', fig=fig4,
                        labelpad=0.7, use_math_test=True )
    
    # some cosmetic adjusting
    fig4.subplots_adjust(left=0.2, bottom=0.2)
    
    #lnlike = likesMCMC.loglike_linear(parsTruth, PATT.pattern, xiObs, covinv)
    #print(lnlike, likesMCMC.loglike_linear(parsGuess, PATT.pattern, xiObs, covinv))

    
    # Show a few array shapes to help understand what's going on here
    #print("INFO:", np.shape(MC.xRaw), np.shape(MC.xGen), \
    #      np.shape(MC.CF.covars), np.shape(covinv))

    # Convert the xy positions into a pattern matrix for the
    # MCMC. 
    
    
    # Show the points, just to see if we have something sensible...
    if showPoints:
        fig1 = plt.figure(1, figsize=(5,4))
        ax1 = fig1.add_subplot(111)
        coverrplot(xiObs[:,0], xiObs[:,1], MC.CF, \
                   errSF=errscalefac, \
                   xLabel=r'$\xi$, deg', yLabel=r'$\eta$, deg', \
                   ax=ax1, fig=fig1)

        dum = ax1.set_title('Uncertainties x%.1f' % (errscalefac))

        # It's useful to do a vector point diagram so that we can test
        # our outlier generation and recovery. Do that here.

        # overplot the transformed xy positions
        fig2, ax2 = plt.subplots(1, num=2, figsize=(5,4))

        dxi  = xiObs[:,0] - MC.xiRaw
        deta = xiObs[:,1] - MC.etaRaw

        # Color for outliers
        coutly = np.asarray(MC.isOutlier,'float')
        if np.sum(MC.isOutlier) < 1:
            coutly = 'k'
        
        blah = ax2.scatter(dxi*3600., deta*3600., alpha=0.5, s=3, c=coutly, \
                           cmap='RdBu')
        ax2.set_xlabel(r'$\Delta \xi$, arcsec')
        ax2.set_xlabel(r'$\Delta \eta$, arcsec')
        ax2.set_title(r'Observed minus "Truth"')

        if MC.isOutlier.sum() > 0:
            cbar = fig2.colorbar(blah, ax=ax2)
        # dum2 = ax1.scatter(MC.xiRaw, MC.etaRaw, marker='o', zorder=25, s=2)


def testRandomDisk(n=100, rmax=1., rmin=0.):

    """Test the method for uniform random xy points generation within an
annulus

    """

    OU = Outliers2d(n, rmax, rmin)
    dx, dy = OU.unifInDisk()

    fig6 = plt.figure(6, figsize=(4,4))
    fig6.clf()
    ax6 = fig6.add_subplot(111)
    
    dum = ax6.scatter(dx, dy, alpha=0.5, s=2, c='g')
    ax6.set_xlabel('dx')
    ax6.set_ylabel('dy')
