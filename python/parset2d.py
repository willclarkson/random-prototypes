#
# parset2d.py
#

#
# 2024-08-14 WIC - refactored out of sim2d.py so that other modules
# can use it without importing sim2d
#

import numpy as np

class Pars1d(object):

    """Parameters for transformation and any other 'model' parameters
including noise model and mixture model. A 1d parameter array can be
input (and split) or the separate pieces can be input and fused. If
the 1d array is supplied, any separate components supplied are
ignored.

Example: supply 1D parameter + index counts to split:

    PP = Pars1d(ppars, nnoise, nshape, nmix)

Example: supply separate model components to fuse into a 1D array:

    QQ = Pars1d(model=PP.model, noise=PP.noise, symm=PP.symm, mix=PP.mix)

Relevant attributes:

    PP.pars = 1D array [transformation, noise, shape, mixture]

    PP.model = transformation model

    PP.noise = noise vs mag model

    PP.symm = [stdy/stdx, corrxy] model

    PP.mix = [foutly, vbackg] model

    

    """

    def __init__(self, pars=np.array([]), nnoise=0, nshape=0, nmix=0, \
                 model=np.array([]), noise=np.array([]), symm=np.array([]), \
                 mix=np.array([])):

        # 1D array of parameters as expected by e.g. minimize. Can be
        # a numpy array or a list
        self.pars = pars

        # parameter-splitting quantities
        self.nnoise = nnoise
        self.nshape = nshape
        self.nmix = nmix

        # Partitioned parameters
        self.model = np.copy(model) # the transformation
        self.noise = np.copy(noise) # the noise vs mag model
        self.symm = np.copy(symm) # noise shape [stdy/stdx, corrxy]
        self.mix = np.copy(mix)  # mixture model [ffg, var_backg]

        # Indices for the model pieces (faster than the general
        # loop-based partitioning at the cost of hard-coding. But
        # we're doing that anyway for the model pieces!):
        self.lmodel = np.array([])
        self.lnoise = np.array([])
        self.lsymm = np.array([])
        self.lmix = np.array([])
        
        # partition the input model parameters if 1D supplied...
        if np.size(pars) > 0:
            self.setupindices()
            self.partitionmodel()

        # ... or, if not, fuse any supplied pieces together
        else:
            if np.size(model) > 0:
                self.fusemodel()
            
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

        """Partitions model parameters using indices already built by
setupindices().

Inputs: None. 

Returns: None.

        """

        if np.size(self.lmodel) < 1:
            return
        
        self.model = self.pars[self.lmodel]
        self.noise = self.pars[self.lnoise]
        self.symm = self.pars[self.lsymm]
        self.mix = self.pars[self.lmix]

    def fusemodel(self):

        """If model parameters were provided separately, fuse them together."""

        if np.size(self.model) < 1:
            return

        self.pars = np.copy(self.model)

        self.nnoise = np.size(self.noise)
        self.nshape = np.size(self.symm)
        self.nmix = np.size(self.mix)

        if self.nnoise > 0:
            self.pars = np.hstack(( self.pars, self.noise ))
        if self.nshape > 0:
            self.pars = np.hstack(( self.pars, self.symm ))
        if self.nmix > 0:
            self.pars = np.hstack(( self.pars, self.mix ))

        # Now that we've done this, set up the indices for consistency
        self.setupindices()
            
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
