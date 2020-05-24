#
# stackmult.py
#

# 2020-05-24 WIC - try different methods of multiplying N x m xm
# matrices with N x m vectors

# Aim: use only np or pure python methods

import numpy as np
import timeit

class StackMult(object):

    def __init__(self, nRows=100, nSquare=2):

        self.n = nRows
        self.m = nSquare

        self.aStack = np.array([])
        self.vStack = np.array([])

        self.enforcePositiveDefinite = True

    def makeRandomVectors(self):

        """Constructs random 'vectors' to multiply by the matrix
        stack"""

        self.vStack = np.random.uniform(size=(self.n, self.m))

    def makeRandomStack(self):

        """Populates the 'stack' of m x m matrices"""

        self.aStack = np.random.uniform(size=(self.n, self.m, self.m))

        if self.enforcePositiveDefinite:
            self.enforcePosDef()

    def enforcePosDef(self):

        """Enforces each plane of the stack as positive definite"""

        # I don't think this will be very slow unless each plane is
        # huge...
        for j in range(self.m):            
            for k in range(self.m):
                self.aStack[:,j,k] = np.abs(self.aStack[:,j,k])
                if k > j:
                    self.aStack[:,j,k] = self.aStack[:,k,j]

    def multStackVecLoop(self):
        
        """Multiplies each plane of the stack in a loop"""
                                        
        aMult = self.vStack*0.
        for iPlane in range(np.shape(self.aStack)[0]):
            aMult[iPlane] = np.dot(self.aStack[iPlane], self.vStack[iPlane])

            #if iPlane < 3:
            #    print("aPlane:", self.aStack[iPlane])
            #    print("vPlane:", self.vStack[iPlane])
            #    print("result:", aMult[iPlane])
                
        return aMult
            
    def multStackVecIter(self):

        """Multiplies each plane of the stack by its corresponding
        vector, using an iterator"""

        nPlanes = np.shape(self.aStack)[0]
        vMult = np.array([np.dot(self.aStack[i], self.vStack[i]) \
                              for i in range(nPlanes)] )

        return vMult

    def multStackVecEinsum(self):

        """Uses np's ensum function to multuply"""

        # Looked this up at
        # https://stackoverflow.com/questions/41190287/how-to-do-matrix-vector-inner-products-for-each-pair-separate-in-python

        vEin = np.einsum('ijk,ik -> ij', self.aStack, self.vStack)
        return vEin

    def multStackVecDummy(self):

        """Adds a dummy axis to the vector stack then uses numpy
        broadcasting to do the elementwise multiplication"""

        # Found this at https://stackoverflow.com/questions/51479148/how-to-perform-a-stacked-element-wise-matrix-vector-multiplication-in-numpy

        vDummy = np.matmul(self.aStack, self.vStack[:,:,None])
        return vDummy

    def vSvLoop(self):

        """Computes vT.S.v plane-by-plane using a loop"""

        # This recalculates all the pieces per iteration to avoid
        # double-looping.

        nPlanes = np.shape(self.vStack)[0]
        vSv = np.zeros(nPlanes)
        for iPlane in range(nPlanes):
            Sv = np.dot(self.aStack[iPlane], self.vStack[iPlane])
            vSv[iPlane] = np.dot(self.vStack[iPlane].T, Sv) 

        return vSv

    def vSvIter(self):

        """Does vT.S.v using an iterator"""

        nPlanes = np.shape(self.vStack)[0]
        vSv = np.array([np.dot(self.vStack[i].T, \
                                  np.dot(self.aStack[i], self.vStack[i])) \
                           for i in range(nPlanes)])

        return vSv

    def vSvDummy(self):

        """Uses dummy indices to do vT.S.v"""

        Sv = self.multStackVecDummy()
        vT = self.vStack.T

        print("vSvDummy INFO:", np.shape(Sv), np.shape(vT[None,:,:]))
        #vSv = np.matmul(self.vStack.T[:,:,None], Sv)
        vSv = np.dot(vT[None,:,:], Sv)
        return vSv

    def vSvEinsum(self):

        """Use the Einstein-summation idiom to do vT.S.v"""

        Sv = self.multStackVecEinsum()
        vSv = np.einsum('ij,ji -> j', self.vStack.T, Sv)
        return vSv

def testTimeStack(n=1000,m=2, nTest=50):

    """Tests the stack"""

    SM = StackMult(n,m)
    SM.makeRandomStack()
    SM.makeRandomVectors()

    vLoop = SM.multStackVecLoop()
    vIter = SM.multStackVecIter()
    vDummy  = SM.multStackVecDummy()
    vEin  = SM.multStackVecEinsum()

    #print vLoop[50]
    #print vIter[50]
    #print vDummy[50].T
    #print vEin[50]

    # OK now we try timing this
    tLoop = timeit.Timer(SM.multStackVecLoop)
    tIter = timeit.Timer(SM.multStackVecIter)
    tDumm = timeit.Timer(SM.multStackVecDummy)
    tEins = timeit.Timer(SM.multStackVecEinsum)

    print("==========")
    print("Timing S*.v with %i iterations @ (%i x %i x %i):" \
              % (nTest, SM.n, SM.m, SM.m))
    print("--------------------------------------------------")
    print("Loop:      %.2e sec per iteration" % ( tLoop.timeit(nTest)/nTest) )
    print("Iterator:  %.2e sec per iteration" % ( tIter.timeit(nTest)/nTest) )
    print("Add dummy: %.2e sec per iteration" % ( tDumm.timeit(nTest)/nTest) )
    print("einsum:    %.2e sec per iteration" % ( tEins.timeit(nTest)/nTest) )

    vSvLoop = SM.vSvLoop()
    vSvIter = SM.vSvIter()
    # vSvDummy = SM.vSvDummy()  ## DOESN'T WORK
    vSvEin = SM.vSvEinsum()

    #print vSvLoop[50]
    #print vSvIter[50]
    ##print vSvDummy[50]
    #print vSvEin[50]

    # (I'm actually not entirely sure how to do the dummy-axis
    # version...)

    # OK now we try timing *this*
    t2Loop = timeit.Timer(SM.vSvLoop)
    t2Iter = timeit.Timer(SM.vSvIter)
    #t2Dumm = timeit.Timer(SM.multStackVecDummy)
    t2Eins = timeit.Timer(SM.vSvEinsum)

    print("==========")
    print("Timing vT.S*.v with %i iterations @ (%i x %i x %i):" \
              % (nTest, SM.n, SM.m, SM.m))
    print("-----------------------------------------------------")
    print("Loop:      %.2e sec per iteration" % ( t2Loop.timeit(nTest)/nTest) )
    print("Iterator:  %.2e sec per iteration" % ( t2Iter.timeit(nTest)/nTest) )
    #print("Add dummy: %.2e sec per iteration" % ( tDumm.timeit(nTest)/nTest) )
    print("einsum:    %.2e sec per iteration" % ( t2Eins.timeit(nTest)/nTest) )
