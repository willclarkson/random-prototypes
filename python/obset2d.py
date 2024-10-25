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

        """Utility - returns copies of all the attributes, in the same order as in initialization. Useful if we want to ensure that modifications do not change an original. 

Example call:

        # obset is an instance we wish to copy
        
        mycopy = Obset(*obset.copycontents())
        

"""

        return np.copy(self.xy), np.copy(self.covxy), np.copy(self.mags), \
            np.copy(self.isfg), np.copy(self.xmin), np.copy(self.xmax), \
            np.copy(self.ymin), np.copy(self.ymax)


    def writeobs(self, pathwrite='test_writeobs.dat'):

        """Writes observations object to disk"""

        # Nothing to do if no data
        if np.size(self.xy) < 1:
            return

        # Nothing to do if path under-specified
        if len(pathwrite) < 4:
            return

        labels_xy = ['x', 'y']
        labels_cov = ['c_xx', 'c_yy', 'c_xy']
        labels_mag = ['mag']
        labels_isfg = ['isfg']

        # build up the output array and labels depending on what we
        # have
        adata = np.copy(self.xy)
        cnames = labels_xy
        
        # Ensure the relevant pieces are output if we have them
        if np.size(self.covxy) > 0:
            cov3 = np.array([self.covxy[:,0,0], \
                             self.covxy[:,0,1], \
                             self.covxy[:,1,1]]).T
            print(np.shape(adata), np.shape(cov3) )
            adata = np.hstack(( adata, cov3 ))
            cnames = cnames + labels_cov

        if np.size(self.mags) > 0:
            adata = np.hstack(( adata, self.mags[:,None] ))
            cnames = cnames + labels_mag

        if np.size(self.isfg) > 0:
            adata = np.hstack(( adata, self.isfg[:,None] ))
            cnames = cnames + labels_isfg

        # Now we've built our array and colnames, write them out
        np.savetxt(pathwrite, adata, header=" ".join(cnames))

    def readobs(self, pathobs='test_writeobs.dat'):

        """Loads observations from disk.

Currently REQUIRES the following order of columns:

        x y c00 c01 c11 mags isfg

"""

        try:
            ain = np.genfromtxt(pathobs, unpack=False)
        except:
            print("obset2d.readobs WARN - problem reading path %s" \
                  % (pathobs))

            return

        # "Initialise" the attributes
        xy = np.array([])
        covxy = np.array([])
        mags = np.array([])
        isfg = np.array([])
        
        
        # Read in the quantities
        nrows, ncols = np.shape(ain)
        xy = ain[:,0:2]

        # Covariance
        if ncols > 2:
            covxy = np.zeros((nrows, 2, 2))
            covxy[:,0,0] = ain[:,2]
            covxy[:,0,1] = ain[:,3]
            covxy[:,1,0] = ain[:,3]
            covxy[:,1,1] = ain[:,4]

        # mag
        if ncols > 5:
            mags = ain[:,5]

        # isfg
        if ncols > 6:
            isfg = ain[:,6]
            

        # Pass up to the instance
        self.xy = np.copy(xy)
        self.covxy = np.copy(covxy)
        self.mags = np.copy(mags)
        self.isfg = np.copy(isfg)
        
