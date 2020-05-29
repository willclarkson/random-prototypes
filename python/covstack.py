#
# covstack.py
#

#
# 2020-05-29 WIC - refactoring into a new module of various
# convenience methods to deal with N x M x M covariance matrices,
# where N is the number of datapoints and we generally want to do
# operations on all the planes (each M x M) at once.

# 
# The application I have in mind here is to assign weights to Gaia
# entries using the maximum eigenvalue of the covariance matrix for
# each object. 
#


# Note: for the special case N x 2 x 2 I have written out the terms
# line by line, for M != 2 we use for loops to populate the
# stack. Tests with timeit (see testPopulateStack below) suggest this
# buys a factor 2 in speed for small (N ~ 1000) arrays, but once N >
# 5000 or so, the difference is fairly negligible.

import numpy as np
import timeit

class CovStack(object):

    """Stack of covariance matrices. Initially developed with N x 2 x
    2 in mind, but I keep this as flexible as I know how..."""

    # This was initially written for Gaia data, which means we know
    # the sigmas and the correlation coefficients (NOT the
    # covariances). We need an ordering principle for parsing the
    # input. I have invented one for the 2x2 case where the matrix is
    # positive-definite.

    def __init__(self, \
                     s11=np.array([]), \
                     s22=np.array([]), \
                     s33=np.array([]), \
                     s44=np.array([]), \
                     r12=np.array([]), \
                     r13=np.array([]), \
                     r14=np.array([]), \
                     r23=np.array([]), \
                     r24=np.array([]), \
                     r34=np.array([]), \
                     crossTermsAreCorrel=True, \
                     Verbose=True, \
                     runOnInit=True):

        """Sets up the covariance stack. Arguments:

        crossTermsAreCorrel -- r?? terms are correlation coefficients,
        not covariances"""

        self.covars = np.array([])

        # Input pieces for the covariance matrices. We hardcode up to
        # N x 4 x 4, anything larger can be dealt with in separate I/O
        # methods.
        self.s11 = np.copy(s11)
        self.s22 = np.copy(s22)
        self.s33 = np.copy(s33)
        self.s44 = np.copy(s44)
        self.r12 = np.copy(r12)
        self.r13 = np.copy(r13)
        self.r14 = np.copy(r14)
        self.r23 = np.copy(r23)
        self.r24 = np.copy(r24)
        self.r34 = np.copy(r34)

        # initialize the dimensions (can use this to see if stack
        # populated yet)
        self.n = 0
        self.m = 0

        # A few internal variables we'll find helpful
        self.eigvals = np.array([])

        # maximum eigenvalue, average eigenvalue (useful as weights)
        self.maxEigvals = np.array([])
        self.meanEigvals = np.array([])

        # Some control variables
        self.crossTermsAreCorrel = crossTermsAreCorrel
        self.Verbose = Verbose

        # populate the covar stack form the inputs
        if runOnInit:
            self.populateStackFromEntries()
            self.getEigenvalues()
            self.findMaxEigenvalues()
            self.findMeanEigenvalues()

    def populateStackFromEntries(self):
        
        """Populates the covariance stack from input entries."""

        self.stackDimens()
        if self.m == 2:
            self.covarsFromEntries2x2()
            return

        self.covarsFromEntries()

    def stackDimens(self):

        """Gets the stack dimensions from the inputs""" 

        self.m = 0
        self.n = np.size(self.s11)
        if self.n < 1:
            return

        # I write out the dimension-determining here to not use
        # FOR-loops... the version that does use a for loop is
        # commented below.
        self.m = 1
        if np.size(self.s22) == self.n:
            self.m = 2
        if np.size(self.s33) == self.n:
            self.m = 3
        if np.size(self.s44) == self.n:
            self.m = 4

        #for kDim in range(1,5):
        #    thisVec= getattr(self, 's%i%i' % (kDim, kDim))
        #    if np.size(thisVec) == self.n:
        #        self.m = kDim
            
    def covarsFromEntries2x2(self):

        """Populates the covariance matrix from input entries,
        assuming 2x2"""

        # ... which is small enough that we can do the interpreter's
        # work for it and just write out the lines. 

        self.n = np.size(self.s11)
        self.covars = np.zeros(shape=(self.n,2,2))
        self.covars[:,0,0] = self.s11**2
        self.covars[:,1,1] = self.s22**2
        if np.size(self.r12) > 0:

            diagMult = np.ones(self.n)
            if self.crossTermsAreCorrel:
                diagMult = self.s11 * self.s22

            self.covars[:,0,1] = ( self.r12 * diagMult )**2
            self.covars[:,1,0] = self.covars[:,0,1]

    def covarsFromEntries(self):

        """Populates the covariance matrix from the input entries."""

        # Determine the dimensions of the covariance stack
        self.stackDimens()
        
        if self.n < 1 or self.m < 1:
            if self.Verbose:
                print("CovStack.covarsFromEntries WARN - (n,m) = (%i,%i)" \
                          % (self.n, self.m))
            return

        # OK now that we know the dimensions, populate the stack. 
        self.covars = np.zeros(shape=(self.n, self.m, self.m))

        # Now we go thru the entries piece by piece. Rather than write
        # out all the conditionals, I think it's worth taking the
        # execution-time hit to populate this by a loop:
        for j in range(self.m):
            self.covars[:,j,j] = getattr(self, 's%i%i' % (j+1,j+1))**2
            
            # now the non-diagonals. 
            for k in range(j+1,self.m):
#                print(j+1,k+1)
                thisOffDiag = getattr(self, 'r%i%i' % (j+1, k+1))
                if np.size(thisOffDiag) < 1:
                    continue

                # thisOffDiag is the correlation coefficient. To get
                # the covariance from this, we multiply by the
                # relevant diagonals and square:
                thisCovarSqrt = thisOffDiag

                if self.crossTermsAreCorrel:
                    thisDiag11 = getattr(self, 's%i%i' % (j+1, j+1))
                    thisDiag22 = getattr(self, 's%i%i' % (k+1, k+1))
                    thisCovarSqrt = thisOffDiag * thisDiag11 * thisDiag22

                self.covars[:,j,k] = thisCovarSqrt**2
                self.covars[:,k,j] = self.covars[:,j,k]


    def getEigenvalues(self):

        """Finds the eigenvalues of each plane in the stack"""
        
        self.eigvals = np.linalg.eigvals(self.covars)

    def findMaxEigenvalues(self):

        """Utility - finds the maximum eigenvalue for each plane"""

        self.maxEigvals = np.max(self.eigvals, axis=1)

    def findMeanEigenvalues(self):

        """Utility - finds the mean eigenvalue for each plane"""

        self.meanEigvals = np.mean(self.eigvals, axis=1)

###############

def testEigens(nRows=1000):

    """Tests getting the eigenvalues"""

    vOnes = np.ones(nRows)*3.
    vRand = np.random.uniform(size=nRows)
    CS2 = CovStack(vOnes, vOnes*0.7, r12=vRand)
    
    print(np.shape(CS2.eigvals))
    print(CS2.covars[0])
    print(CS2.eigvals[0])
    print(CS2.maxEigvals[0:4])
    print(CS2.meanEigvals[0:4])
    print(CS2.eigvals[0:4])

def testPopulateStack(nRows = 1000, nTest=1000):

    """Test routine to see if our methods for populating the (n, m, m)
    stack are working..."""

    vOnes = np.ones(nRows)
    vRand = np.random.uniform(size=nRows)

    CSN = CovStack(vOnes, vOnes, r12=vRand, runOnInit=False)
    CS2 = CovStack(vOnes, vOnes, r12=vRand, runOnInit=False)

    # try timing...
    tLoopN = timeit.Timer(CSN.covarsFromEntries)
    tLoop2 = timeit.Timer(CS2.covarsFromEntries2x2)

    print("Loop: %.2e sec per iteration" % (tLoopN.timeit(nTest)/nTest))
    print("2x2: %.2e sec per iteration" % (tLoop2.timeit(nTest)/nTest))

    CSN.covarsFromEntries()
    CS2.covarsFromEntries2x2()

    print(CSN.covars[0:2])
    print("=====")
    print(CS2.covars[0:2])
