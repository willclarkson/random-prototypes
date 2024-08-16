#
# obset2d.py
#

#
# 2024-08-16 WIC - refactored out of sim2d.py
#

import numpy as np

class Obset(object):

    """Convenience-object to hold positions, covariances, and other
information like apparent magnitudes for hypothetical observations."""

    def __init__(self, xy=np.array([]), covxy=np.array([]), \
                 mags=np.array([]), isfg=np.array([]), \
                 xmin=None, xmax=None, ymin=None, ymax=None):

        self.xy = np.copy(xy)
        self.covxy = np.copy(covxy)
        self.mags = np.copy(mags)
        self.isfg = np.copy(isfg)

        # Domain of the source data (assumed detector)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        
        # Number of datapoints
        self.countpoints()

    def countpoints(self):

        """Updates internal attribute self.npts with the number of datapoints"""
        
        self.npts = np.shape(self.xy)[0]
        
    def copycontents(self):

        """Utility - returns copies of all the attributes, in the same order as in initialization"""

        return np.copy(self.xy), np.copy(self.covxy), np.copy(self.mags), \
            np.copy(self.isfg), np.copy(self.xmin), np.copy(self.xmax), \
            np.copy(self.ymin), np.copy(self.ymax)
