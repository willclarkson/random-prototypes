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


    # compute optimal average
    LIx = LinearInvar(tSimX, ySimX, uSimX, runOnInit=True)
    LIy = LinearInvar(tSimY, ySimY, uSimY, runOnInit=True)


    # now set up the plot. We'll want a fine-grained array to overplot
    # the predictions and 1-sigma uncertainties
    tFine = np.linspace(np.min(tSimX)-plotTimeBuf, \
                            np.max(tSimX)+plotTimeBuf, nFine)

    # evaluate the best-fit straight line and its 1-sigma curve
    predFineX = LIx.evalFunc(tFine)
    oneSigFineX = LIx.evalOneSig(tFine)

    # ditto for Y
    predFineY = LIy.evalFunc(tFine)
    oneSigFineY = LIy.evalOneSig(tFine)

    # generate the +/- one-sigma polygons to fill here
    tPolX, fPolX = retPolyFill(tFine, predFineX + oneSigFineX, \
                                   predFineX - oneSigFineX)

    tPolY, fPolY = retPolyFill(tFine, predFineY + oneSigFineY, \
                                   predFineY - oneSigFineY)


    # a few arguments for all the plots
    cData = 'darkred'
    cErro = '0.4'
    szData = 4
    markerData = 'o'
    cBestfit = 'k'
    cBounds = '0.3'
    cFill = '0.7'
    # cFill = 'b'
    alphaFill = 0.3

    fig = plt.figure(1)
    fig.clf()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    # Now do the plotting incantations for each co-ordinate
    dumErrX = ax1.errorbar(tSimX, ySimX, yerr=uSimX, ls='none', c=cData, \
                               zorder=5, \
                               marker='o', ms=szData, \
                               capsize=4, \
                               ecolor=cErro)

    dumBestX = ax1.plot(tFine, predFineX, ls='-', color=cBestfit, zorder=6)
    dumUpper = ax1.plot(tFine, predFineX + oneSigFineX, ls='-', \
                            color=cBounds, \
                            lw=0.5)
    dumLower = ax1.plot(tFine, predFineX - oneSigFineX, ls='-', \
                            color=cBounds, \
                            lw=0.5)

    dumPolX = ax1.fill(tPolX, fPolX, zorder=2, color=cFill, alpha=alphaFill)

    dumErrX = ax1.errorbar(tSimX, ySimX, yerr=uSimX, ls='none', c=cData, \
                               zorder=5, \
                               marker='o', ms=szData, \
                               capsize=6, \
                               ecolor=cErro)

    #... and now for the Y-positions
    dumErrY = ax2.errorbar(tSimY, ySimY, yerr=uSimX, ls='none', c=cData, \
                               zorder=5, \
                               marker='o', ms=szData, \
                               capsize=4, \
                               ecolor=cErro)

    dumBestY = ax2.plot(tFine, predFineY, ls='-', color=cBestfit, zorder=6)
    dumUpper = ax2.plot(tFine, predFineY + oneSigFineY, ls='-', \
                            color=cBounds, \
                            lw=0.5)
    dumLower = ax2.plot(tFine, predFineY - oneSigFineY, ls='-', \
                            color=cBounds, \
                            lw=0.5)

    dumPolY = ax2.fill(tPolY, fPolY, zorder=2, color=cFill, alpha=alphaFill)


    # Annotate the plots with the proper motions and object IDs

    # String for the annotations themselves...
    sPMx = r'$\mu(X) = %.2f \pm %.2f$ mas yr$^{-1}$' % \
        (LIx.beta*masPerPix, np.sqrt(LIx.betaVar)*masPerPix)
    sPMy = r'$\mu(Y) = %.2f \pm %.2f$ mas yr$^{-1}$' % \
        (LIy.beta*masPerPix, np.sqrt(LIy.betaVar)*masPerPix)

    ax1.annotate(sPMx, (0.50,0.02), xycoords='axes fraction', \
                     ha='center', va='bottom', fontsize=12)
    ax2.annotate(sPMy, (0.50,0.02), xycoords='axes fraction', \
                     ha='center', va='bottom', fontsize=12)

    ax1.annotate(objID, (0.50, 0.97), xycoords='axes fraction', \
                     ha='center', va='top', fontsize=14)

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
    
