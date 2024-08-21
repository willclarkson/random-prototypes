#
# binstats2d.py
#

#
# 2024-08-21 WIC - perform statistics binned by a quantity. Refactored
# out of examine2d.py
#

import numpy as np

class Binstats(object):

    """Performs statistics on data binned by a quantity. Similar to
astropy's binned_statistic.

Inputs:
    
    mags = 1d array used to set the bins.

    xy = [N, 2+] array of data on which to compute stats

    nperbin = number of objects per bin

    nbins = if set, the number of bins. Overrides nperbin.

"""

    def __init__(self, mags=np.array([]), xy=np.array([]), nperbin=15., \
                 nbins=-1):

        self.mags = mags
        self.xy = xy
        self.nperbin = nperbin

        if nbins > 0:
            self.calcnperbin(nbins)

        # Bin information
        self.lsor = np.array([])
        self.xleft = np.array([])
        self.binid = np.array([])
        self.initbins()

        # statistics
        self.counts = np.array([])
        self.medns = np.array([])
        self.meansxy = np.array([])
        self.covsxy = np.array([])
        
        # Set the bin boundaries and identify points with each bin
        self.setbins()
        self.initstats()
        self.assignbins()
        self.calcstats()

    def calcnperbin(self, nbins=-1):

        """Computes the number per bin given desired number of bins"""

        if nbins < 1:
            return

        # Nothing to do if we have no data
        ndata = np.size(self.mags)
        if ndata < 1:
            return

        self.nperbin = int(ndata/float(nbins))
        
    def initbins(self):

        """(Re-)initializes the bins"""

        self.ileft = np.array([])
        self.xleft = np.array([])
        
    def initstats(self):

        """(Re)-initializes the statistics arrays"""

        nbins = self.ileft.size
        ndim = self.xy.shape[-1]
        
        self.counts = np.zeros(nbins, 'int') 
        self.medns = np.zeros(nbins)
        self.meansxy = np.zeros((nbins, ndim))
        self.covsxy = np.zeros((nbins, ndim, ndim))

    def setbins(self):

        """Sets bins of equal number of points"""

        self.lsor = np.argsort(self.mags)

        if np.size(self.lsor) < 1:
            return
        
        self.ileft = np.asarray(np.arange(0, np.size(self.lsor), self.nperbin), 'int')

        # edge-case: nperbin a factor of size(lsor)
        if self.ileft[-1] == np.size(self.lsor):
            self.ileft = self.ileft[0:-1]

        # Ensure the rightmost bin has enough points
        if self.lsor.size - self.ileft[-1] < self.nperbin:
            self.ileft = self.ileft[0:-1]

        # array of magnitude values at left edges of bins
        self.xleft = self.mags[self.lsor[self.ileft]]
        self.xleft[0] -= 0.01 # left edge < min datapoint
        
    def assignbins(self):

        """Assigns bins to input datapoints and counts the number of points
per bin"""

        # Nothing to do if arrays not set up yet
        if self.xleft.size < 1:
            return

        if self.mags.size < 1:
            return

        self.binid = np.digitize(self.mags, self.xleft, right=False)
        self.counts = np.bincount(self.binid)[1::] # sic
        self.binid -= 1 # sic

    def calcstats(self):

        """Computes the binned statistics"""

        nbins = self.counts.size
        if nbins < 1:
            return
        
        # Examine bins that are actually populated
        bocc = self.counts > 0
        if np.sum(bocc) < 1:
            return

        lbin = np.arange(nbins, dtype='int')[bocc]
        for ibin in lbin:
            bthis = self.binid == ibin

            # Defensive programming - this should never happen
            if np.sum(bthis) < 1:            
                continue

            self.medns[ibin] = np.median(self.mags[bthis])

            if self.xy.size < 2:
                continue

            self.meansxy[ibin] = np.mean(self.xy[bthis], axis=0)
            self.covsxy[ibin] = np.cov(self.xy[bthis], rowvar=False)

    def getstats(self):

        """Returns the binned statistics to the calling method.

Inputs: None.

Returns:

        mmed  =  [nbins]: median "magnitude" in the bin

        meansxy = [nbins, dim] = mean position

        covsxy = [nbins, ndim, ndim] = positional covariance

        counts = [nbins] = number of points within the bin

Example call:
        
        BC = Binstats(mags, xy, 15)

        midpt, means, covs, counts = BC.getstats()

        """

        return self.medns, self.meansxy, self.covsxy, self.counts

            
