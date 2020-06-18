#
# weightedDeltas.py
# 

#
# 2020-05-29 WIC - collects some methods for finding the weighted
# positional deltas (on the sphere) and the weighted least squares fit
# between positions.
#

import time

import numpy as np

# for plotting
import matplotlib.pylab as plt
from matplotlib.collections import EllipseCollection, LineCollection

# for estimating weights
from covstack import CovStack

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
        if AA.shape[0] <> VV.shape[0]:
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

    Verbose = print lots of screen output
    
    """

    # Work on generalizing to N-dimensional data later.

    def __init__(self, x=np.array([]), y=np.array([]), \
                     xi=np.array([]), eta=np.array([]), \
                     W=np.array([]), \
                     xref=0., yref=0., \
                     fitChoice='6term', Verbose=False):

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
        self.bPlanes = np.array([])
        self.initBplanes()

        # The results
        self.pars = np.array([])  # M
        self.formalCov = np.array([]) # the formal covariance estimate

        # Results decomposed into reference point and 2x2 transformation
        self.xiRef = np.array([])   # 2-element vector
        self.BMatrix = np.array([]) # 2x2 matrix

        # The tangent point on the source system (the point
        # corresponding to xi = [0., 0.])
        self.xZero = np.array([])

    def initBplanes(self):

        """Ensure the b-planes object is populated"""

        if np.size(self.bPlanes) < 1:
            self.bPlanes = np.isfinite(self.x[:,0])

    def buildPattern(self):

        """Creates the pattern matrix using the pattern class"""

        self.patnObj = ptheta2d(self.x[:,0], self.x[:,1], self.fitChoice)

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
        self.beta = np.sum(PWxi[self.bPlanes], axis=0)

    def makeHessian(self):

        """Populates the Hessian matrix: sum_i P^T W_i P"""

        PWP = np.matmul(self.patternT, np.matmul(self.W, self.pattern))

        # Sum along the i dimension, but only the planes we trust
        self.H = np.sum(PWP[self.bPlanes], axis=0)
        
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
                     fitChoice='6term', pars=np.array([]) ):

        self.x = x
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

    If the parameters are supplied, the matrix stack is generated on
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
                     genStripe=True):

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

    def populateCovarsFromInputs(self):

        """Populates the covariance stack from input arguments. In
        order of preference: covars --> xycomp --> abtheta"""

        # if the covariances already have nonzero shape, override nPts
        # with their leading dimension
        if np.size(self.covars) > 0:
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
            iHalf = int(0.5*np.size(self.rotDegs))
            self.rotDegs [iHalf::] *= -1.

    def populateDiagCovar(self):

        """Populates diagonal matrix stack with major and minor axes"""

        self.VV = np.array([])

        nm = np.size(self.majors)
        if nm < 1:
            return

        self.VV = np.zeros(( nm, 2, 2 ))
        self.VV[:,0,0] = self.asVector(self.majors)
        self.VV[:,1,1] = self.asVector(self.minors)

    def populateRotationMatrix(self, rotateAxes=False):

        """Populates rotation matrix stack using rotations.

        rotateAxes = rotate the axes instead of the points?"""

        self.RR = np.array([])
        nR = np.size(self.rotDegs)
        if nR < 1:
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
                   ax=None, fig=None, figNum=1):

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

    figNum = if we are making a new figure, the figure number"""

    if np.size(x) < 1:
        return

    if np.size(y) != np.size(x):
        return

    # construct the covariance object if not given
    if not covars:
        covars = CovarsNx2x2(stdx=errx, stdy=erry, corrxy=corrxy)
        if covars.nPts < 1:
            return

        covars.eigensFromCovars()
        covars.populateTransfsFromCovar()

    # Expects to be given an axis, but generates a new figure if none
    # is passed.
    if not fig:
        fig = plt.figure(figNum, figsize=(5,4))
    
    if not ax:
        ax = fig.add_subplot(111)

    # A few convenience views. (covars.axMajors are the normalized
    # major eigenvector of each plane of the covariance stack,
    # covars.majors are the major eigenvalues. Similar for the minor
    # axes.)
    xMajors = covars.axMajors[:,0]*errSF*covars.majors
    yMajors = covars.axMajors[:,1]*errSF*covars.majors

    xMinors = covars.axMinors[:,0]*errSF*covars.minors
    yMinors = covars.axMinors[:,1]*errSF*covars.minors

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

        ec = EllipseCollection(covars.majors*errSF*2., \
                                   covars.minors*errSF*2., \
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
        
    # enforce uniform axes?
    if enforceUniformAxes:
        unifAxisLengths(ax)
        ax.set_aspect('equal')

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



#### Normal Equations Fitting class

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
                     Verbose=False):

        self.Verbose = Verbose

        # Datapoints in each frame, ref points, covariances
        self.x = np.copy(x)
        self.y = np.copy(y)
        self.xi = np.copy(xi)
        self.eta = np.copy(eta)
        self.xRef = np.copy(xRef)
        self.yRef = np.copy(yRef)
        
        # Covariances in the target frame, if given
        self.covars = np.copy(covars)

        # Covariances as stddevs & correlation, if given
        self.stdxi = np.copy(stdxi)
        self.stdeta = np.copy(stdeta)
        self.corrxieta = np.copy(corrxieta)

        # Weights (can be supplied)
        self.W = np.copy(W)        
        self.ensureWeightsPopulated()

        # Normal equations object, formal covariance estimate (if
        # desired)
        self.NE = None

        # perform the fit on initialization
        self.performFit()
        
        if invertHessian:
            self.NE.invertHessian()


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

    def performFit(self):

        """Sets up the normal equations object and performs the fit,
        as well as operations we are likely to want for every trial
        (such as finding the tangent point from the fit parameters)"""

        self.NE = NormalEqs(self.x, self.y, self.xi, self.eta, W=self.W, \
                                xref=self.xRef, yref=self.yRef)

        self.NE.doFit()
        self.NE.estTangentPoint()

    def findFormalCovar(self):

        """Finds the formal covariance estimate for the parameters"""

        self.NE.invertHessian()

#### Some routines that use this follow

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
