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
                          
# useful generally-found methods
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


