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

        # we DO want to output the adopted xmin, xmax, ymin, ymax
        # since at laest fit2d.py gets them from the observation
        # object (and not the parset). May want to fix that later.

        # String for limits
        slims = 'limits (x then y): '
        for attr in ['xmin', 'xmax', 'ymin', 'ymax']:
            valu = getattr(self, attr)
            if valu is not None:
                slims = '%s %e' % (slims, valu)
            else:
                slims = '%s None' % (slims)

        # Column names
        snames = " ".join(cnames)

        # make the header two lines:
        sheader = '%s \n %s' % (slims, snames)
        
        # Now we've built our array and colnames, write them out
        np.savetxt(pathwrite, adata, header=sheader)

    def readobs(self, pathobs='test_writeobs.dat', strictlimits=True):

        """Loads observations from disk.

Currently REQUIRES the following order of columns:

        x y c00 c01 c11 mags isfg

Inputs:

        pathobs = path to observations

        strictlimits = require all four limits to be OK to use any of them.

"""

        # Could do some parsing based on the column names, but for the
        # moment we assume we are only reading output from writobs.
        
        try:
            ain = np.genfromtxt(pathobs, unpack=False)
        except:
            print("obset2d.readobs WARN - problem reading path %s" \
                  % (pathobs))

            return

        # I find numpy's use of data formats to smuggle in the column
        # names to be non-transparent, and I don't think it handles
        # header information at all... For the moment, then, use the
        # standard python library to read in the "header", like so:
        header = []
        ilims = -1
        with open(pathobs, 'r') as robj:
            for line in robj:
                line = line.strip()
                if line.find('#') < 0:
                    break

                header.append(line)
                if line.find('limit') > -1:
                    ilims = len(header)-1
                
        # now get the limits. The slightly weird syntax here is so
        # that we can split on "limits" and not "limits:"
        badlimits = True
        attrs = ['xmin', 'xmax', 'ymin', 'ymax']
        if ilims > -1:
            badlimits = False
            slims = header[ilims].split("limits")[-1]
            vlims = slims.split(' ')[-4::]

            # Now we interpret the results
            for iattr in range(len(attrs)):
                try:
                    valu = float(vlims[iattr])
                except:
                    valu = None

                    badlimits=True # any of the limits are bad
                    
                setattr(self, attrs[iattr], valu)

        # Replace all four limits with None if ANY of them are bad?
        if strictlimits and badlimits:
            for attr in attrs:
                setattr(self, attr, None)
            
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
        
#######

def testread(pathobs='test_obset_written.dat', strictlims=True):

    """Tests loading obset from text"""

    dum = Obset()
    dum.readobs(pathobs, strictlims)

    print("INFO:", np.shape(dum.covxy))
    print(dum.xmin, dum.xmax, dum.ymin, dum.ymax)
