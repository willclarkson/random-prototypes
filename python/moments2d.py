#
# moments2d.py
#

# WIC - utility object to compute and store moments of [nsamples,
# ndata] sample sets

import numpy as np
from scipy import stats
from weightedDeltas import CovarsNx2x2

class Moments2d(object):

    """Computes moments of input 2d data. The samples are NOT stored in
the instance (since the arrays might be large), only the summary
statistics"""

    def __init__(self, x=np.array([]), y=np.array([]), \
                 nfine=10000, nomode=True):

        # Quantities we want
        self.median = np.array([])
        self.mean = np.array([])
        self.mode = np.array([])

        self.covars = np.array([])
        self.skew = np.array([])
        self.kurtosis = np.array([])

        # Covariance object with methods
        self.cov = None

        # number of fine-grained samples to use when estimating the
        # mode
        self.nfine = nfine

        # The mode is currently estimated by looping through the data
        # rows and dimensions, computing the marginal mode in each
        # case. It may thus be slow. Allow it to be deactivated
        self.nomode = nomode

        # Compute the moments on initialization
        self.calcmoments(x, y)

        
    def calcmoments(self, x=np.array([]), y=np.array([]) ):

        """Computes moments of input data.

Inputs:

        x = [samples, 2, ndata] set of 2d samples; OR, [nsamples,
        ndata] sets of x-values only

        y = [nsamples, ndata] sets of y-values, if x is also two
        dimensional

"""

        # Ensure samples is [nsamples, 2, ndata]
        samples = self.parsesamples(x, y)
        if np.size(samples) < 1:
            return

        self.median = np.median(samples, axis=0).T
        self.mean = np.mean(samples, axis=0).T

        # The covariance object and the actual covariances 
        self.cov = CovarsNx2x2(xysamples=samples)
        self.cov.eigensFromCovars()
        self.covars = self.cov.covars

        # Marginal higher moments
        self.skew = stats.skew(samples, axis=0).T
        self.kurtosis = stats.kurtosis(samples, axis=0).T

        # Compute the mode
        self.calcmodes(samples)
        
    def parsesamples(self, x=np.array([]), y=np.array([]) ):

        """Returns [nsamples, 2, ndata] array from inputs.

Inputs:

        x = [nsamples, ndata] or [nsample, 2, ndata] samples

        y = [nsamples, ndata] . Ignored if x is already three-dimensional

Returns:

        samples = [nsamples, 2, ndata] array

"""

        if np.size(x) < 1:
            return np.array([])

        if np.ndim(x) == 3:
            return x

        # If x was not 3 dimensional, construct samples from the
        # inputs
        try:
            samples = np.stack((x, y), axis=1)
        except:
            samples = np.array([])

        return samples

    def calcmodes(self, x=np.array([]) ):

        """Computes the marginal modes. 

Inputs:

        x = [nsamples, 2, ndata] set of samples

Returns:

        modes = [ndata, 2] array of marginal modes of the samples

"""

        # don't actually do anything if mode computation is switched
        # off
        if self.nomode:
            return
        
        if np.ndim(x) != 3:
            return
        
        nsamples, ndim, ndata = x.shape
        self.mode = np.zeros(( ndata, ndim ))

        # try transposing the samples array
        sam = np.transpose(x, axes=(1,2,0))
        
        for idim in range(ndim):
            for jdata in range(ndata):

                sampl = sam[idim, jdata]
                print("  ",jdata, idim, sampl.shape, end="\r")
                
                self.mode[jdata, idim] = \
                    self.calcmode1d(sampl)
        
    def calcmode1d(self, x=np.array([]) ):

        """Estimates the mode of the distribution of (1d) samples. 

Inputs:

        x = [nsamples] array of samples

Returns:

        mode = scalar mode of the samples

"""

        if np.size(x) < 1:
            return 0.

        kde = stats.gaussian_kde(x)
        xfine = np.linspace(np.min(x), np.max(x), self.nfine)
        imax = np.argmax(kde.pdf(xfine))
        return xfine[imax]
        
