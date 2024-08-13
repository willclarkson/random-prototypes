#
# explore2d.py
#

#
# 2028-08-13 WIC - methods to explore 2D mapping using MCMC. Much of
# this is OO refactored from the prototype fittwod.py.
#

import numpy as np

class Pars1d(object):

    """Parameters for transformation and any other 'model' parameters
including noise model and mixture model"""

    def __init__(self, pars=np.array([]), nnoise=0, nshape=0, nmix=0):

        # 1D array of parameters as expected by e.g. minimize. Can be
        # a numpy array or a list
        self.pars = pars

        # parameter-splitting quantities
        self.nnoise = nnoise
        self.nshape = nshape
        self.nmix = nmix

        # Partitioned parameters
        self.model = np.array([]) # the transformation
        self.noise = np.array([]) # the noise vs mag model
        self.symm = np.array([]) # noise shape [stdy/stdx, corrxy]
        self.mix = np.array([])   # mixture model [ffg, var_backg]

        # Indices for the model pieces (faster than the general
        # loop-based partitioning at the cost of hard-coding. But
        # we're doing that anyway for the model pieces!):
        self.lmodel = np.array([])
        self.lnoise = np.array([])
        self.lsymm = np.array([])
        self.lmix = np.array([])
        
        # partition the input model parameters
        self.setupindices()
        self.partitionmodel()
        
    def insertpars(self, p=np.array([]) ):

        """Replaces the parameters with input.

Inputs:

        p = [transf, noise, shape, mix] parameters

Returns: None. Updates the following attributes:

        self.pars = 1D array of parameters. Makes a copy of the input.

"""

        self.pars = np.copy(p)

    def updatepars(self, p=np.array([]) ):

        """Inputs and partitions input parameters. 

    Inputs:

        p = [transf, noise, shape, mix] parameters

    Returns: None. Updates the following attributes:

        self.pars = 1D array of parameters. Makes a copy of the input.

        self.model = transformation parameters

        self.noise = noise model parameters

        self.symm = noise shape parameters
        
        self.mix = mixture model parameters

"""

        self.insertpars(p)
        self.partitionmodel()

    def setupindices(self):

        """Sets up the indices corresponding to each model parameter.

Inputs: None. The following must be set:

        self.pars = 1d array of model parameters

        self.nnoise, self.nsymm, self.nmix = number of parameters
        describing the noise mag model, noise shape model and mixture
        model, respectively.

Returns: None. Updates the following:

        self.lmodel, self.lnoise, self.lsymm, self.lmix = index arrays
        describing which parameters in self.pars are transformation,
        noise, shape, mixture, respectively.

        """

        npars = np.size(self.pars)
        lpars = np.arange(npars, dtype='int')
        
        # initialise the non-transformation parameters. Must be
        # integer to be used as indices
        self.lnoise = np.array([], dtype='int')
        self.lsymm = np.copy(self.lnoise)
        self.lmix = np.copy(self.lnoise)

        # Cut down the index array progressively. This is similar to
        # what splitmodel does as a loop, but we lay it out explicitly
        # here since we're hard-coding the different parts of the
        # model anyway.
        if self.nmix > 0:
            self.lmix = lpars[-self.nmix::]
            lpars = lpars[0:-self.nmix]

        if self.nshape > 0:
            self.lsymm = lpars[-self.nshape::]
            lpars = lpars[0:-self.nshape]
            
        if self.nnoise > 0:
            self.lnoise = lpars[-self.nnoise::]
            lpars = lpars[0:-self.nnoise]

        # OK what's left is the transformation parameter index set
        self.lmodel = lpars

    def partitionmodel(self):

        """Partitions model parameters using indices already built by setupindices(). 

Inputs: None. 

Returns: None.


"""

        self.model = self.pars[self.lmodel]
        self.noise = self.pars[self.lnoise]
        self.symm = self.pars[self.lsymm]
        self.mix = self.pars[self.lmix]
        
    def splitmodel(self):

        """Splits the model 1d parameters into transformation and the other parameters.

    Inputs:

        None. Acts on:

        self.pars = 1D array with the parameters: [transformation,
        noise, vars, mixture]

    Returns: None. Updates the following attributes:

        self.model = transformation parameters

        self.noise = [A, B, C] parameters for noise vs magnitude

        self.symm = [stdy/stdx, corrxy] parameters for noise

        self.mix = [ffg, var_bg] mixture model parameters

        """

        lnums = [self.nmix, self.nshape, self.nnoise]
        
        self.model, lsplit = self.splitpars(self.pars, lnums)
        self.noise = lsplit[2]
        self.symm = lsplit[1]
        self.mix = lsplit[0]
        
    def splitpars(self, pars, nsplit=[]):

        """Splits a 1d array into sub-arrays, skimming off the nsplit entries
from the end at each stage. Like splitpars() but the nsplits are
generalized into a loop. 

    Inputs:

        pars = 1d array of parameters

        nsplit = [n0, n1, ... ] list of last-n indices to split off at each stage.

    Returns: 

        allbutsplit = all the pars not split off into a subarray
        
        [p1, p2, ...] = list of pars split off from the far end, *in the same order as the nsplit list*. 

    Example:

        x = np.arange(10)
        fittwod.splitpars(x,[3,2])
    
        returns:

             (array([0, 1, 2, 3, 4]), [array([7, 8, 9]), array([5, 6])])

    """

        # handle scalar input for nsplit
        if np.isscalar(nsplit):
            nsplit = [nsplit]
    
        # if no splits, nothing to do
        if len(nsplit) < 1:
            return pars

        lsplit = []
        allbut = np.copy(pars)

        for isplit in range(len(nsplit)):
            allbut, split = self.splitlastn(allbut, nsplit[isplit])
            lsplit = lsplit + [split]
        

        return allbut, lsplit
            
    def splitlastn(self, pars=np.array([]), nsplit=0):

        """Splits a 1D array into its [0-nsplit] and and [-nsplit::] pieces.

    Inputs:

        pars = [M]  array of parameters

        nsplit = number of places from the end of the array that will be
        split off

    Returns:

        first = [M-nsplit] array before the split

        last = [nsplit] array after the split

        """

        # Nothing to do if nothing provided
        if np.size(pars) < 1:
            return np.array([]), np.array([])

        # Cannot do anything if the lengths do not match
        if np.size(pars) < nsplit:
            return pars, np.array([])
    
        # Nothing to do if no split
        if nsplit < 1:
            return pars, np.array([])

        return pars[0:-nsplit], pars[-nsplit::]


### SHORT test routines come here.

def testsplit(nnoise=3, nshape=2, nmix=2):

    """Tests the splitting behavior"""

    transf = np.arange(6)
    pnoise = np.arange(nnoise)+10
    pshape = np.arange(nshape)+100
    pmix = np.arange(nmix) + 1000

    ppars = np.hstack(( transf, pnoise, pshape, pmix ))

    PP = Pars1d(ppars, nnoise, nshape, nmix)

    print(ppars)

    # Set up the indices - what do we get?
    
    print(PP.model)
    print(PP.noise)
    print(PP.symm)
    print(PP.mix)

    # Now update the parameters
    PP.updatepars(0.-ppars)

    print(PP.pars)
    print(PP.model)
    print(PP.noise)
    print(PP.symm)
    print(PP.mix)
    
