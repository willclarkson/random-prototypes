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

This object is also used to smuggle options for the eventual use by lnprob(). Currently that just means:

    islog10_mix_frac = mixture fraction specified as log10

    islog10_mix_vxx = mixture variance vxx specified as log10

    islog10_noise_c = noise model parameter c specified as log10

    mag0 = magnitude zeropoint for the noise model

    """

    def __init__(self, pars=np.array([]), nnoise=0, nshape=0, nmix=0, \
                 model=np.array([]), noise=np.array([]), symm=np.array([]), \
                 mix=np.array([]), \
                 islog10_mix_frac=True, islog10_mix_vxx=True, \
                 islog10_noise_c=False, \
                 mag0=0.):

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
        
        # Descriptors for some of the non-transformation model
        # parameters
        self.islog10_mix_frac = islog10_mix_frac
        self.islog10_mix_vxx = islog10_mix_vxx
        self.islog10_noise_c = islog10_noise_c
        
        # Some other quantities we need but which are not model
        # parameters
        self.mag0 = mag0 # magnitude zeropoint
        
        # stems for labels
        self.labelstem_transf = 'A'
        self.labels_noise = [r'$log_{10}(a)$', r'$log_{10}(b)$', r'$c$']
        self.labels_asymm = [r'$\sigma_y/\sigma_x$', r'$\rho_{xy}$']
        self.labels_mix = [r'$f_{bg}$', r'$V_{bg}$']
        self.fixlabelstems()
        
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

        self.insertpars(np.copy(p))
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

    def getmodelasxy(self):

        """Utility - returns self.model as two arrays of half the length of the original, *if* self.model has an even number of entries."""

        nmodel = np.size(self.model)
        if nmodel < 0:
            return np.array([]), np.array([])

        if nmodel % 2 > 0:
            return np.array([]), np.array([])

        nhalf = int(nmodel*0.5)
        return self.model[0:nhalf], self.model[nhalf::]
        
        
        
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

    def fixlabelstems(self):

        """Ensures the noise parts of the plot labels are consistent with
whether the quantities are being used as log_10"""

        if self.islog10_mix_frac:
            self.labels_mix[0] = r'$log_{10}(f_{bg})$'

        if self.islog10_mix_vxx:
            self.labels_mix[1] = r'$log_{10}(V_{bg})$'

        if self.islog10_noise_c:
            self.labels_noise[2] = r'$log_{10}(c)$'
            
    def getlabels(self):

        """Utility - returns labels for use in plots"""

        # Transformation parameters
        labels_model = [r'$%s_{%i}$' % (self.labelstem_transf, i) \
                        for i in range(np.size(self.model)) ]

        labels_model += self.labels_noise[0:self.nnoise]
        labels_model += self.labels_asymm[0:self.nshape]
        labels_model += self.labels_mix[0:self.nmix]

        return labels_model
        
class Pairset(object):

    """Pair of two Pars1d objects, with comparison method(s)"""

    def __init__(self, set1=None, set2=None, padval=None):

        self.set1 = set1
        self.set2 = set2

        # The transformation itself usually has parameters for x, y
        # separately:
        self.modelx1, self.modely1 = self.set1.getmodelasxy()
        self.modelx2, self.modely2 = self.set2.getmodelasxy()

        # value to use when padding
        self.padval = padval

        # Set 1 on set 2
        self.set1on2 = None

        # Since we need to do the padding before doing any kind of
        # arithmetic, go ahead and do the padding now.
        self.padset1toset2()
        
    def padset1toset2(self, retval=False):

        """Produces a copy of set 1 with the same lengths in all model
parameters as set 2. If set 2 contains entries not present in set 1,
they are padded with self.padval in set 1.

        if retval=True, returns the set rather than updating the instance

        """

        # First the transformation model...
        modelx = self.pad1to2(self.modelx1, self.modelx2)
        modely = self.pad1to2(self.modely1, self.modely2)
        model = np.hstack(( modelx, modely ))

        # ... then the non-transformation model pieces
        noise = self.pad1to2(self.set1.noise, self.set2.noise)
        asymm = self.pad1to2(self.set1.symm, self.set2.symm)
        mix = self.pad1to2(self.set1.mix, self.set2.mix)

        # Now construct the padded set out of this
        set1on2 = Pars1d(model=model, noise=noise, symm=asymm, mix=mix, \
                         mag0=self.set1.mag0)

        if retval:
            return set1on2

        self.set1on2 = set1on2

    def pad1to2(self, arr1=np.array([]), arr2=np.array([]), padval=None):

        """Utility - given two 1D arrays, returns a version of the first, cut
or padded to the same length as set 2.

        """

        len1 = np.size(arr1)
        len2 = np.size(arr2)

        if len2 < 1:
            return np.array([])

        if len2 <= len1:
            return arr1[0:len2]

        arrpad = np.repeat(padval, len2-len1)
        
        return np.hstack(( arr1, arrpad ))

    def add(self):

        """Adds the two parameter sets along their common elements"""

        # The sets must have the same length
        if self.set1on2 is None:
            self.padset1toset2()

        # Initialize to array of None, filling in the values for which
        # non-Nones are in both.
        sum = np.repeat(None, np.size(self.set1on2.pars))
        
        bok = self.set1on2.pars != None        
        sum[bok] = self.set1on2.pars[bok] + self.set2.pars[bok]

        # Create a new parameter set with these values filled in.
        Psum = Pars1d(sum, \
                      self.set1on2.nnoise, \
                      self.set1on2.nshape, \
                      self.set1on2.nmix)

        return Psum

    def arithmetic(self, set1=None, set2=None, op=np.add, divzeroval=None):

        """Does arithmetic on two input sets, whose parameters must already be
of the same length (see padset1toset2). If not provided, returns None.

Inputs:

        set1 = Pars1d object for the first set

        set2 = Pars1d object for the second set

        op = operation. Usually np.add, np.multiply, np.subtract,
        np.divide

        divzeroval = If dividing, replace any div by zero entries with
        this value.

Returns:

        set3 = set1 (+, -, *, /) set 2 = Pars1d object.

        """

        # The paramsets must have the same length
        if set1 is None or set2 is None:
            return None

        # Get the parameter arrays, those are the parts we will want
        # to use
        pars1 = set1.pars
        pars2 = set2.pars

        res = np.repeat(None, np.size(pars1))

        # Now fill in the values. Can only do this if both entries in
        # each matching piece are valid.
        bok = (pars1 != None) & (pars2 != None)

        # If we're dividing, don't divide by zero
        if op is np.divide:
            divok = pars2 != 0
            res[~divok] = divzeroval
            
            bok = (bok) & (divok)
        
        res[bok] = op(pars1[bok], pars2[bok])

        # Create a new parameter set with these values filled in
        Pres = Pars1d(res, set1.nnoise, set1.nshape, set1.nmix)

        return Pres

    def fracdiff(self):

        """Utility - finds the fractional difference between self.set1 and self.set2, in the sense abs(set1-set2)/set2

Returns:

        Pfrac = Pars1d object with the fractional difference between
        parameters returned.

        """

        if self.set1on2 is None:
            self.padset1toset2()
        
        Pdiff = self.arithmetic(self.set1on2, self.set2, np.subtract)
        bok = Pdiff.pars != None
        Pdiff.pars[bok] = np.abs(Pdiff.pars[bok])

        return self.arithmetic(Pdiff, self.set2, np.divide)
    
## SHORT test routines come here

def testsplit(nnoise=3, nshape=2, nmix=2):

    """Tests the splitting behavior"""

    transf = np.arange(6)
    pnoise = np.arange(nnoise)+10
    pshape = np.arange(nshape)+100
    pmix = np.arange(nmix) + 1000

    ppars = np.hstack(( transf, pnoise, pshape, pmix ))

    PP = Pars1d(ppars, nnoise, nshape, nmix)

    print("Original:")
    print(ppars)

    # Set up the indices - what do we get?

    print(PP.model)
    print(PP.noise)
    print(PP.symm)
    print(PP.mix)

    # Now try fusing these into a separate object
    QQ = Pars1d(model=PP.model, noise=PP.noise, symm=PP.symm, mix=PP.mix)

    print("Fused:")
    print(QQ.pars)
    
    # Now update the parameters
    PP.updatepars(0.-ppars)

    print("Updated:")
    print(PP.pars)
    print(PP.model)
    print(PP.noise)
    print(PP.symm)
    print(PP.mix)

def testcompare(ntransf1=6, nnoise1=3, nshape1=2, nmix1=2, \
                ntransf2=6, nnoise2=3, nshape2=2, nmix2=2):

    """Compare two parameter sets. Useful when e.g. comparing truth to fit
parameters, where the two parameter sets can have differnet
configurations.

    Currently just does a lot of screen output while I think of ways
    to test this.

    """

    PP = Pars1d(model=np.arange(ntransf1), \
                noise=np.arange(nnoise1)+10., \
                symm=np.arange(nshape1)+100., \
                mix=np.arange(nmix1)+1000.)

    QQ = Pars1d(model=np.arange(ntransf2), \
                noise=np.arange(nnoise2)+10., \
                symm=np.arange(nshape2)+100., \
                mix=np.arange(nmix2)+1000.)

    print(PP.pars)
    print(QQ.pars)
    
    # try merging the two
    Pair = Pairset(PP, QQ)
    # Pair.padset1toset2()

    #print(PP.pars)
    #print(QQ.pars)

    # Try adding them
    Psum = Pair.add()

    # Try our more general arithmetic
    Padd = Pair.arithmetic(Pair.set1on2, Pair.set2, np.add)
    
    print(Psum.pars)
    print(Padd.pars)

    # now try subtracting and dividing
    print("Subtraction:")
    print(Pair.set1on2.pars)
    print(Pair.set2.pars)
    Psub = Pair.arithmetic(Pair.set2, Pair.set1on2, np.subtract)
    print(Psub.pars)

    # try a ratio
    Prat = Pair.arithmetic(Psub, Pair.set2, np.divide)
    print(Prat.pars)

    print(Prat.model)

    # Try fractional difference
    Pfd = Pair.fracdiff()
    print(Pfd.pars)

    print(Pfd.model)
