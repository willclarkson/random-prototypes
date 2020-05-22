#
# straightLineInvar.py 
#
 
#
# WIC 2020-05-21
#

# Class LinearInvar() performs inverse variance-weighted straight line
# fit. Naming convention: x(t) = alpha + beta*(t-tbar)
#
# Currently this assumes the times, values and uncertainties are all
# 1-d arrays with the same length.
#
# We take a bare-bones approach, since the straight line case is just
# a set of formula evaluations.

# method testStraightLine() generates fake data, fits it, and plots
# it.

import numpy as np
import matplotlib.pylab as plt
from matplotlib import ticker
import os

# use stylesheets?
try:
    plt.style.use('default')
    # plt.style.use('ggplot')
except:
    notUsingStylesheet = True  # dummy branch

class LinearInvar(object):

    """Methods to fit the inverse variance-weighted straight line fit
    x(t) = alpha + beta(t-t0) where t0 is the center of mass from the
    data (computed using the input uncertainties).

    Populates the attributes self.alpha, self.beta, self.alphaVar,
    self.betaVar, self.tBar"""

    def __init__(self, times=np.array([]), \
                     vals=np.array([]), \
                     unctys=np.array([]), \
                     runOnInit=True):

        # time, value, uncty
        self.times = times
        self.vals = vals
        self.unctys = unctys

        # The results
        self.tBar = 0.
        self.alpha = 0.  # center of mass vertical value
        self.beta = 0.   # gradient
        self.alphaVar = 0. # formal variance of the CM value
        self.betaVar = 0. # formal variance of the gradient
        
        # some statistics
        self.sumChisq = 0.
        self.nDof = np.size(self.times) - 2

        # Convenience views
        self.wgts = np.array([])    # weights
        self.tDiffs = np.array([])   # t-tbar
        
        # Internal quantities
        self.sumWgts = 1.

        # array of chisq per object
        self.valsPred = np.array([]) 
        self.chisqs = np.array([])

        # if runOnInit is set and the dimensions of the data allow it,
        # do everything.
        if runOnInit and self.dimensAgree():
            self.calcWeights()
            self.calcSumWgts()
            self.calcTbar()
            self.calcTdiffs()
            self.calcOptAlpha()
            self.calcOptBeta()
            self.calcOptAlphaVar()
            self.calcOptBetaVar()
            self.calcValsPred()
            self.calcChisq()
            self.calcSumChisq()

    def dimensAgree(self):

        """Returns True if the dimensions of the times, values and
        uncertainties are compatible."""

        if np.size(self.times) < 1:
            return False

        if np.size(self.times) != np.size(self.vals):
            return False

        # We take the strict view and require the uncertainties to be
        # populated.
        if np.size(self.times) != np.size(self.unctys):
            return False

        # if we didn't fail any of the conditions, return True
        return True

    def calcWeights(self):

        """Computes the weights"""

        self.wgts = 1.0/self.unctys**2
        
    def calcSumWgts(self):

        """Computes the sum of the inverse variances"""

        self.sumWgts = np.sum(self.wgts)

    def calcTbar(self):

        """Computes the inverse variance-weighted mean time"""

        self.tBar = np.sum(self.times * self.wgts) / self.sumWgts

    def calcTdiffs(self):

        """Computes t-tbar"""

        self.tDiffs = self.times - self.tBar

    def calcOptAlpha(self):

        """Computes the weighted CM value (the best-fit line with zero
        slope)"""

        self.alpha = np.sum(self.vals*self.wgts) / self.sumWgts

    def calcOptAlphaVar(self):

        """Computes the formal variance of the weighted CM value"""

        self.alphaVar = 1.0/self.sumWgts

    def calcOptBeta(self):

        """Computes the weighted gradient"""
        
        self.beta = np.sum(self.vals * self.tDiffs * self.wgts) \
            / np.sum(self.tDiffs**2 * self.wgts)

    def calcOptBetaVar(self):

        """Computes the formal variance of the weighted gradient"""

        self.betaVar = 1.0/np.sum(self.tDiffs**2 * self.wgts)

    def evalFunc(self, tIn=np.array([]) ):

        """Evaluates the function x(t) = alpha + beta*(t-t0) given
        input times"""

        return self.alpha + (tIn - self.tBar)*self.beta

    def evalOneSig(self, tIn=np.array([]) ):

        """Evaluates the '1-sigma' uncertainty in x(t) given input
        times. Returns std not variance."""

        xVar = self.alphaVar + (tIn - self.tBar)**2 * self.betaVar

        return np.sqrt(xVar)
        
    def calcValsPred(self):

        """Calculates predicted locations using the best-fit model"""

        self.valsPred = self.evalFunc(self.times)

    def calcChisq(self):

        """Calculates chisq"""
        
        # weights are assumed to be 1/sigma^2

        self.chisqs = (self.vals - self.valsPred)**2 * self.wgts

    def calcSumChisq(self):

        """Calculates the sum of the chisq"""

        self.sumChisq = np.sum(self.chisqs)
        self.nDof = np.size(self.chisqs) - 2

class TimeSeries(object):

    """Time-series (t,x,uncty). Methods to read the data can go in
    here. One file each for time, position, uncty assumed."""

    def __init__(self, fTimes='egTimesX.txt', \
                     fVals='egPosX.txt', \
                     fUncts='egUnctyX.txt', \
                     runOnInit=True, \
                     Verbose=False):

        # filenames 
        self.filTimes = fTimes[:]
        self.filPosns = fVals[:]
        self.filUnctys = fUncts[:]

        # arrays for times, values, uncertainties
        self.aTimes = np.array([])
        self.aVals = np.array([])
        self.aUnctys = np.array([])

        # status flag for read OK
        self.readOK = False

        # cotrol variable
        self.Verbose = Verbose
        
        if runOnInit:
            self.loadData()

    def loadData(self):

        """Loads the input data"""

        # if any of the paths are missing, do nothing
        if not self.allPathsOK():
            return

        self.aTimes = np.genfromtxt(self.filTimes)
        self.aVals  = np.genfromtxt(self.filPosns)
        self.aUnctys = np.genfromtxt(self.filUnctys)

        self.readOK = True

    def allPathsOK(self):

        """Checks to see if all paths are OK"""

        for thisPath in [self.filTimes, self.filPosns, self.filUnctys]:
            if not os.access(thisPath, os.R_OK):
                if self.Verbose:
                    print("TimeSeries.allPathsOK WARN - cannot read path %s" \
                              % (thisPath))
                return False

        return True

def testLoadFitPlot(objID='Star 7847', \
                        filTimesX='egTimesX.txt', \
                        filValsX='egPosX.txt', \
                        filUncsX='egUnctyX.txt', \
                        filTimesY='egTimesY.txt', \
                        filValsY='egPosY.txt', \
                        filUncsY='egUnctyY.txt', \
                        masPerPix=50., \
                        figFilename='testFromFiles.png'):

    """Tests loading, fitting and plotting. Arguments:

    objID -- string with the object ID

    filTimesX, filTimesY -- files with times for X, Y measurements

    filValsX, filValsY -- files with X, Y measurements
    
    filUncsX, filUncsY -- files with Y, Y uncertainties

    masPerPix -- milliarcsec per pixel for data (to convert the
    gradient in pix/yr to mas/yr)

    figFilename -- filename for the output figure"""

    # load the data for this star for X, Y
    TSx = TimeSeries(filTimesX, filValsX, filUncsX, runOnInit=True)
    TSy = TimeSeries(filTimesY, filValsY, filUncsY, runOnInit=True)

    # Warn and exit if there was a problem reading the second dataset
    if not TSx.readOK or not TSy.readOK:
        print("testLoadFitPlot WARN - problem reading data")
        return

    # Do the linear fit
    FITx = LinearInvar(TSx.aTimes, TSx.aVals, TSx.aUnctys, runOnInit=True)
    FITy = LinearInvar(TSy.aTimes, TSy.aVals, TSy.aUnctys, runOnInit=True)

    # plot the results
    plotFits(FITx, FITy, objID, masPerPix, figFilename=figFilename)

def testStraightLine(nOne=5, nTwo=6, yrOne=2000.0, yrTwo=2007.0, \
                         medOneX=1393.0, betaX=0.0, \
                         medOneY=755.3, betaY=-5., \
                         nFine=200, \
                         plotTimeBuf=1., \
                         masPerPix=50., \
                         objID='Star 123 (fake data)', \
                         enforceSameAxes=False, \
                         figFilename='testFigFake.jpg'):

    """Tests the straight line fitting and plotting for fake data. Arguments:

    nOne, nTwo, yrOne, yrTwo = number and epoch of the two datasets

    medOneX, medOneY = median X, Y position at epoch 1

    betaX, betaY = gradient in mas per year in  X, Y

    masPerPix = plate scale of the camera

    enforceSameAxes -- ensure the vertical scales of the two subplots
    have the same range

    figFilename -- filename for saved figure

    """

    # generate fake time-series
    betaInX = betaX / masPerPix
    betaInY = betaY / masPerPix

    tSimX, ySimX, uSimX = genFake1D(nOne, nTwo, yrOne, yrTwo, medOneX, betaInX)
    tSimY, ySimY, uSimY = genFake1D(nOne, nTwo, yrOne, yrTwo, medOneY, betaInY)

    # compute inverse variance-weighted average in X, Y
    LIx = LinearInvar(tSimX, ySimX, uSimX, runOnInit=True)
    LIy = LinearInvar(tSimY, ySimY, uSimY, runOnInit=True)

    plotFits(LIx, LIy, objID, masPerPix, nFine, plotTimeBuf, \
                 enforceSameAxes, figFilename)

def plotFits(LIx=None, LIy=None, objID='123', \
                 masPerPix=50, nFine=100, plotTimeBuf=2., \
                 enforceSameAxes=True, figFilename='Test.jpg'):

    """Does the two-panel plot of the straight line fits"""

    # set up the figure
    fig = plt.figure(1)
    fig.clf()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    # now add the panels for x and y
    showBestFit(LIx, ax1, nFine, plotTimeBuf)
    showBestFit(LIy, ax2, nFine, plotTimeBuf)

    # Annotate the top panel with the object ID
    ax1.annotate(objID, (0.50, 0.93), xycoords='axes fraction', \
                     ha='center', va='top', fontsize=14)

    # Annotate both panels with the proper motion and uncertainty
    for sCoo, fitObj, ax in zip(['X', 'Y'], [LIx, LIy], [ax1, ax2]):
        sAnno = r'$\mu(%s) = %.2f \pm %.3f$ mas yr$^{-1}$' \
            % (sCoo, fitObj.beta*masPerPix, \
                   np.sqrt(fitObj.betaVar)*masPerPix)
        
        ax.annotate(sAnno, (0.50,0.02), xycoords='axes fraction', \
                        ha='center', va='bottom', fontsize=12)

    # Do we want to enforce the same vertical axis scale?
    if enforceSameAxes:
        vertLimsX = np.copy(ax1.get_ylim())
        vertLimsY = np.copy(ax2.get_ylim())

        # axis range and midpoint
        diffX = np.max(vertLimsX) - np.min(vertLimsX)
        diffY = np.max(vertLimsY) - np.min(vertLimsY)

        midX = 0.5 * diffX + np.min(vertLimsX)
        midY = 0.5 * diffY + np.min(vertLimsY)

        if diffX > diffY:
            newLimsY = diffX * np.array([-0.5, 0.5]) + midY
            ax2.set_ylim(newLimsY)
        else:
            newLimsX = diffY * np.array([-0.5, 0.5]) + midX
            ax1.set_ylim(newLimsX)

    # Now for a few pieces of decoration...

    # bring the panes together and move the vertical axis of the top
    # pane to the top
    fig.subplots_adjust(hspace=0.03)
    ax1.xaxis.set_ticks_position('top')

    for ax in [ax1, ax2]:
        ax.grid(which='both', visible=True, alpha=0.3)

    ax2.set_xlabel('time (years)')
    ax1.set_ylabel('Pixel position X')
    ax2.set_ylabel('Pixel position Y')

    # finally, save the figure to disk
    fig.savefig(figFilename)

    
def showBestFit(LI=None, ax=None, nFine=100, plotTimeBuf=0.5, \
                    showChisq=True):

    """Adds the panel with the best-fit to the input axis."""
    
    # Arguments for the plots
    cData = 'darkred'
    cErro = '0.3'
    szData = 3
    errCapsz = 3
    markerData = 'o'
    cBestfit = 'k'
    cBounds = '0.6'
    cFill = '0.5'
    # cFill = 'b'
    alphaFill = 0.3
    alphaErrorbar = 0.6  # we want to get an idea of point-concentration

    # We'll want a fine-grained array to overplot the predictions and
    # 1-sigma uncertainties
    tFine = np.linspace(np.min(LI.times)-plotTimeBuf, \
                            np.max(LI.times)+plotTimeBuf, nFine)

    # evaluate the best-fit straight line and its 1-sigma curve
    predFineX = LI.evalFunc(tFine)
    oneSigFineX = LI.evalOneSig(tFine)

    # generate the +/- one-sigma polygons to fill here
    tPolX, fPolX = retPolyFill(tFine, predFineX + oneSigFineX, \
                                   predFineX - oneSigFineX)

    # When annotating with chi-squared the annotation sometimes
    # overlays the data. To avoid this, we use the chisq as the label
    # and use matplotlib's positioning of the label to put it
    # somewhere that does not overlay the data. That means moving the
    # chisq determination up to here.
    sChisX = r'$\chi^2_\nu = %.2f / %i$' % (LI.sumChisq, LI.nDof)


    # Now do the plotting incantations for each co-ordinate
    dumErrX = ax.errorbar(LI.times, LI.vals, yerr=LI.unctys, \
                              ls='none', c=cData, \
                              zorder=10, \
                              marker='o', ms=szData, \
                              capsize=errCapsz, \
                              ecolor=cErro, \
                              alpha=alphaErrorbar)

    # The best-fit line is labeled with the chisq/dof
    dumBestX = ax.plot(tFine, predFineX, ls='-', color=cBestfit, zorder=6, \
                           label=sChisX)
    dumUpper = ax.plot(tFine, predFineX + oneSigFineX, ls='-', \
                           color=cBounds, \
                           lw=0.5)
    dumLower = ax.plot(tFine, predFineX - oneSigFineX, ls='-', \
                           color=cBounds, \
                            lw=0.5)

    dumPolX = ax.fill(tPolX, fPolX, zorder=2, color=cFill, alpha=alphaFill)
    
    # now we use the legend to draw the chisq so it doesn't collide
    if showChisq:
        leg=ax.legend(loc=0, frameon=False)
        
        # uncomment the following to hide the marker with the legend
        #for item in leg.legendHandles:
        #    item.set_visible(False)

    # switch off the axis offset
    y_formatter = ticker.ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(y_formatter)

def retPolyFill(tFine, yLo, yHi):

    """Utility - given tFine and the upper and lower bounds, returns a
    cyclic polygon for matplotlib's fill command""" 

    tRet = np.hstack(( tFine, tFine[::-1] ))
    yRet = np.hstack(( yLo, yHi[::-1] ))

    return tRet, yRet

def genFake1D(nOne=5, nTwo=6, yrOne=2000.0, yrTwo=2007.0, \
                  medOne=1393.0, betaIn=0.1):

    """Generates toy fake data from two epochs"""

    # uncertainty distribution per epoch
    sigmOne = np.random.uniform(size=nOne)*0.08 + +0.2
    sigmTwo = np.random.uniform(size=nTwo)*0.05 + +0.1

    # generate the "true" times, positions and uncertainties
    tOne = np.repeat(yrOne, nOne)
    tTwo = np.repeat(yrTwo, nTwo)
    
    genOne = (tOne-yrOne) * betaIn + medOne
    genTwo = (tTwo-yrOne) * betaIn + medOne

    yOne = genOne + np.random.normal(size=np.size(genOne))*sigmOne
    yTwo = genTwo + np.random.normal(size=np.size(genTwo))*sigmTwo

    # append the time-series together and return
    tRet = np.hstack(( tOne, tTwo ))
    yRet = np.hstack(( yOne, yTwo ))
    sigmRet = np.hstack(( sigmOne, sigmTwo ))

    return tRet, yRet, sigmRet
    
