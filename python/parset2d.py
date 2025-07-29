#
# parset2d.py
#

#
# 2024-08-14 WIC - refactored out of sim2d.py so that other modules
# can use it without importing sim2d
#

import numpy as np
import copy
import configparser

# our methods for converting {abcdef} <--> {a,d,sx,sy,theta, beta}
import sixterm2d

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

    xmin, xmax, ymin, ymax = data limits (used to rescale to [-1,1]
    for any polynomial model)

    transfname = name of transformation

    polyname = name of any polynomial transformation

    """

    def __init__(self, pars=np.array([]), nnoise=0, nshape=0, nmix=0, \
                 model=np.array([]), noise=np.array([]), symm=np.array([]), \
                 mix=np.array([]), \
                 islog10_mix_frac=True, islog10_mix_vxx=True, \
                 islog10_noise_c=False, \
                 mag0=0., \
                 xmin=None, xmax=None, ymin=None, ymax=None, \
                 transfname='', polyname='Chebyshev'):

        # 1D array of parameters as expected by e.g. minimize. Can be
        # a numpy array or a list
        self.pars = pars

        # parameter-splitting quantities
        self.nmodel = np.size(model)
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

        # dictionary matching parameter labels to indices
        self.dindices = {}
        
        # Descriptors for some of the non-transformation model
        # parameters
        self.islog10_mix_frac = islog10_mix_frac
        self.islog10_mix_vxx = islog10_mix_vxx
        self.islog10_noise_c = islog10_noise_c
        
        # Some other quantities we need but which are not model
        # parameters
        self.mag0 = mag0 # magnitude zeropoint

        # Some other quantities some of the transforamtions need
        self.updatedatarange(xmin, xmax, ymin, ymax)
        self.updatetransfname(transfname)
        self.updatepolyname(polyname)
        
        # stems for latex labels
        self.labelstem_transf = 'A'
        self.labels_noise = [r'$log_{10}(a)$', r'$log_{10}(b)$', r'$c$']
        self.labels_asymm = [r'$\sigma_y/\sigma_x$', r'$\rho_{xy}$']
        self.labels_mix = [r'$f_{bg}$', r'$V_{bg}$']

        # parameter names for parameter files
        self.parnames_noise = ['noise_log10a', 'noise_log10b', 'noise_c']
        self.parnames_asymm = ['symm_yx', 'symm_rhoxy']
        self.parnames_mix = ['mix_fbg', 'mix_vbg']

        # Covariance matrix between all the parameters. Expected to be
        # populated after an MCMC run.
        self.covar = np.array([])
        
        # Update the label stems and parnames depending on whether the
        # noise and mix parameters are log10.
        self.fixlabelstems()
        
        # partition the input model parameters if 1D supplied...
        if np.size(pars) > 0:
            self.setupindices()
            self.partitionmodel()

        # ... or, if not, fuse any supplied pieces together
        else:
            if np.size(model) > 0:
                self.fusemodel()

        # Prepare the dictionary giving 1d indices in the eventual
        # vector output
        self.makeindexmap()
                
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

    def updatedatarange(self, xmin=None, xmax=None, ymin=None, ymax=None):

        """Updates data range attributes.

Inputs:

        xmin, xmax, ymin, ymax = input data ranges, scalars"""

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def updatetransfname(self, transfname=''):

        """Updates the transformation name.

Inputs:

        transfname = string, the new name for the transformation

"""

        self.transfname = transfname[:]

    def updatepolyname(self, polyname='Chebyshev'):

        """Updates the polynomial transformation name.

Inputs:

        polyname = name of the polynomial piece of the transformation

        """

        self.polyname = polyname[:]
        
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

        # Update the count for the model params
        self.nmodel = np.size(lpars)
        
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
        self.nmodel = np.size(self.model)
        
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
            self.parnames_mix[0] = 'mix_log10fbg'
            
        if self.islog10_mix_vxx:
            self.labels_mix[1] = r'$log_{10}(V_{bg})$'
            self.parnames_mix[1] = 'mix_log10vbg'

        # 2025-07-21 allow for length-2 noise
        if self.islog10_noise_c:
            self.labels_noise[-1] = r'$log_{10}(c)$'
            self.parnames_noise[-1] = 'noise_log10c'
            
    def getlabels(self):

        """Utility - returns labels for use in plots"""

        # Transformation parameters
        labels_model = [r'$%s_{%i}$' % (self.labelstem_transf, i) \
                        for i in range(np.size(self.model)) ]

        # Hack for two-element noise model again (whence we count from
        # 1 and not 0 since we're using [b,c] and not [a,b,c]).
        ioffset_noise = 0
        if self.nnoise == 2:
            ioffset_noise = 1
        
        labels_model += self.labels_noise[ioffset_noise:self.nnoise+ioffset_noise]


        labels_model += self.labels_asymm[0:self.nshape]
        labels_model += self.labels_mix[0:self.nmix]

        return labels_model

    def makeindexmap(self):

        """Utility - makes a dictionary giving the parameter index of each
model component, keyed by the model component label as would be read
in by a parameter file"""

        self.dindices={}

        # Fill in this indices. Rather than using the full latex
        # labels, which are inconvenient for parameter files, we use a
        # more easily-typed shorthand, from the 'parnames_noise'
        # attributes. For the "nuisance parameters," we already know
        # what the names are. But the transformation parameter labels
        # are generated programmatically. We generate them again here
        # and pass them in. So:

        # Transformation parameters
        labls = self.getlabels()
        for imodl in range(np.size(self.model)):
            skey = labls[imodl].replace('$','').replace('{','').replace('}','')
            self.dindices[skey] = imodl
        
        # Now for the nuisance parameters. There needs to be a slight
        # hack for the noise model, since if size==2 then we are
        # fitting [b,c] instead of [a,b,c] or [a]. One way is just to
        # start counting from 1 in the list of labels. (Another might
        # just be to cut down the instance-level list of parameter
        # names and labels.)
        inoise_offset = 0
        if np.size(self.lnoise) == 2:
            inoise_offset = 1
            
        for inoise in range(np.size(self.lnoise)):
            self.dindices[self.parnames_noise[inoise + inoise_offset]] \
                = self.lnoise[inoise]

        for isymm in range(np.size(self.lsymm)):
            self.dindices[self.parnames_asymm[isymm]] = self.lsymm[isymm]

        for imix in range(np.size(self.lmix)):
            self.dindices[self.parnames_mix[imix]] = self.lmix[imix]

    def writeparset(self, pathpars='test_parset.txt'):

        """Writes parameter set to disk"""

        if len(pathpars) < 4:
            return

        # set up the configuration object
        config = configparser.ConfigParser()

        # Configuration section...
        config['config'] = {}
        conf = config['config'] # save on typos

        for key in ['transfname', 'polyname', \
                    'xmin', 'xmax', 'ymin', 'ymax', \
                    'mag0', 'islog10_mix_frac', 'islog10_mix_vxx', \
                    'islog10_noise_c', \
                    'nmodel', 'nnoise', 'nshape', 'nmix']:
            conf[key] = str(getattr(self, key))

        # Now for the model and nuisance sections
        config['model'] = {}
        config['indices'] = {}
        modl = config['model']
        inds = config['indices']

        # Populate both model and index values
        for key in self.dindices.keys():
            indx = self.dindices[key]
            valu = self.pars[indx]

            modl[key] = str(valu)
            inds[key] = str(indx)

        # Write the resulting file to disk
        with open(pathpars, 'w') as wobj:
            config.write(wobj)

    def readparset(self, pathpars='test_parset.txt'):

        """Sets parameter attributes from text file.

Inputs:

        pathpars = path to input text file

"""

        if len(pathpars) < 1:
            return

        config = configparser.ConfigParser()
        try:
            config.read(pathpars)
        except:
            print("parset2d.readparset WARN - problem loading path %s" \
                  % (pathpars))
            return

        # Load the configuration information
        if 'config' in config.sections():
            conf = config['config']

            # Configparser distinguishes between datatypes. So:
            keys_str = ['transfname', 'polyname']
            keys_flt = ['xmin', 'xmax', 'ymin', 'ymax', 'mag0']
            keys_int = ['nmodel', 'nnoise', 'nshape', 'nmix']
            keys_boo = ['islog10_mix_frac', 'islog10_mix_vxx', \
                        'islog10_noise_c']

            # Now we read them in, type by type
            self.getconfvalues(conf, keys_str, conf.get, '')
            self.getconfvalues(conf, keys_flt, conf.getfloat)
            self.getconfvalues(conf, keys_int, conf.getint)
            self.getconfvalues(conf, keys_boo, conf.getboolean)

        # now indices if they are present. Currently this REQUIRES the
        # [indices] section to be present. Would be good to allow this
        # to work without that section. Come back to that later.
        self.dindices = {}
        if 'indices' in config.sections():
            inds = config['indices']
            for item in inds.items():
                key = item[0]
                ind = inds.getint(key)
                self.dindices[key] = ind
                
        # We get the number of parameters from the header (rather than
        # the dictionary, so that we need only specify a subset)
        npars = self.nmodel + self.nnoise + self.nshape + self.nmix
        # npars = len(self.dindices.keys() )

        # Here we re-initialize the parameters.
        self.pars = np.zeros(npars)
        
        # Now we read in the parameters section
        if 'model' in config.sections():
            modl = config['model']
            for parname in self.dindices.keys():
                valu = 0.
                try:
                    valu = modl.getfloat(parname)
                except:
                    absent = True

                # Now we slot the quantity into the proper position in
                # the parameter array:
                indx = self.dindices[parname]
                self.pars[indx] = valu

        # Now that we have the parameters filled in (or at least
        # "harmless" default values), ensure the partition attributes
        # are consistent:
        self.setupindices()
        self.partitionmodel()

        # (re-) fix latex label stems if needed
        self.fixlabelstems()
        
    def getconfvalues(self, conf=None, keys=[], methget=None, default=None):

        """One-liner to get config values depending on type.

Inputs:

        conf = configuration object section

        keys = lists of keywords to get from the object

        methget = method to use to get the value (depends on target
        type). 

        default = default value for attribute if absent from config

        """

        if conf is None:
            return

        if len(keys) < 1:
            return

        if methget is None:
            return
        
        for key in keys:
            valu = default
            try:
                valu = methget(key)
            except:
                absent = True
            setattr(self, key, valu)

    def blankmodel(self, transfname='Poly', deg=1, \
                   nnoise=3, nshape=2, nmix=2, \
                   polyname='Chebyshev'):

        """Utility - populates blank parset (mostly for serializing to disk so
that we can create a guess)

Inputs:

        transfname = name of the transformation (as per unctytwod
        class names)

        deg = degree of any polynomial included

        nnoise = number of noise parameters

        nshape = number of shape parameters

        nmix = number of mixture parameters

        polyname = polynomial type (if any) to be used in the model

        """

        # Implementation comment: prefer not to need to import
        # uncertaintytwod.py here, instead going for heuristics to
        # determine the length of the model array.

        # Number of polynomial coefficients, whether or not we will
        # use them.
        npoly = int(deg**2 + 3*deg + 2)

        # Populate the number of model parameters depending on which
        # option we have. This approach should allow us some
        # flexibility if we later decide to allow both the tangent
        # point and the location of the detector on the focal plane to
        # vary.
        dmodl = {'Poly':npoly, 'Tan2equ':2, 'Equ2tan':2, \
                 'xy2equ':npoly, 'TangentPlane':npoly}

        # Now we know how large to make the model parameter vector,
        # build it and ensure the object is self-consistent:
        self.nmodel = npoly
        if transfname in dmodl.keys():
            self.nmodel = dmodl[transfname]
        self.nnoise = nnoise
        self.nshape = nshape
        self.nmix = nmix

        # By this point we know how large an array we are making. So:
        self.model = np.zeros(self.nmodel)
        self.noise = np.zeros(self.nnoise)
        self.symm = np.zeros(self.nshape)
        self.mix = np.zeros(self.nmix)

        self.pars = np.hstack(( self.model, self.noise, \
                                self.symm, self.mix ))
        
        # now (re-) set up the various indices to ensure consistency
        self.updatetransfname(transfname)
        self.updatepolyname(polyname)
        self.fixlabelstems()
        self.setupindices()
        self.partitionmodel()
        self.makeindexmap()
        
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
        modelx = self.pad1to2(self.modelx1, self.modelx2, padval=self.padval)
        modely = self.pad1to2(self.modely1, self.modely2, padval=self.padval)
        model = np.hstack(( modelx, modely ))

        # ... then the non-transformation model pieces
        noise = self.pad1to2(self.set1.noise, self.set2.noise)
        asymm = self.pad1to2(self.set1.symm, self.set2.symm)
        mix = self.pad1to2(self.set1.mix, self.set2.mix)

        # Now construct the padded set out of this.

        # If returning a new object, bring across the metadata...
        if retval:
            return Pars1d(model=model, noise=noise, \
                          symm=asymm, mix=mix, \
                          mag0=self.set1.mag0, \
                          islog10_mix_frac = self.set1.islog10_mix_frac, \
                          islog10_mix_vxx = self.set1.islog10_mix_vxx, \
                          islog10_noise_c = self.set1.islog10_noise_c, \
                          xmin = self.set1.xmin, \
                          xmax = self.set1.xmax, \
                          ymin = self.set1.ymin, \
                          ymax = self.set1.ymax, \
                          transfname = self.set1.transfname, \
                          polyname = self.set1.polyname)

        # ... otherwise re-gen in place        
        self.set1on2 = Pars1d(model=model, noise=noise, symm=asymm, mix=mix, \
                              mag0=self.set1.mag0)


        
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

    def sub1into2(self, subvalue=0.):

        """For any entries in set2 that are subvalue (default to zero),
substitute any non-None values from set1on2. 

Inputs:

        subvalue = value in set2 that indicates which items need
        substituting

Returns:

        Psub = parset object with the substitutions carried out

        """

        # The sets must have the same length
        if self.set1on2 is None:
            self.padset1toset2()

        # Which entries in set 2 can be substituted?
        bmissing2 = np.abs(self.set2.pars-subvalue) == 0.
        bcansub1 = self.set1on2.pars != None
        bsub = bmissing2 * bcansub1

        # Now we set up the parameter object to substitute where
        # needed
        subs = np.copy(self.set2.pars)
        subs[bsub] = self.set1on2.pars[bsub]

        # We create a parset object from set2 and update its
        # parameters in place.
        Psub = copy.deepcopy(self.set2)
        Psub.updatepars(subs)

        return Psub
        
    def fracdiff(self, preventnans=True, preventzeros=False):

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

        Pdiv = self.arithmetic(Pdiff, self.set2, np.divide)

        if preventnans:
            print("parset2d.fracdiff INFO - preventing NaNs:")
            bbad = Pdiv.pars == None

            # We do this in two steps:
            Pdiv.pars[bbad] = 0.

        # If we want to prevent zeros as well, do the following
        if preventzeros:
            bnz = np.abs(Pdiv.pars) > 0.
            minval = np.min(np.abs(Pdiv.pars[bnz]))
            Pdiv.pars[~bnz] = minval
                
            Pdiv.partitionmodel() # ensure that propagated in

        return Pdiv

def loadparset(pathpars='', parse6term=True):

    """One-liner to return parset object from disk.


Inputs:

    pathpars = path to paramset

    parse6term = if True, looks for the parameter "theta" and, if
    present, converts the 6-term {a,d,sx, sy, theta, beta} into
    {a,b,c,d,e,f} before returning the pset.

Returns:

    parset = parset object. If pathpars not found, returns blank
    parset object

    """
    
    pset = Pars1d()
    try:
        pset.readparset(pathpars)
    except:
        return pset
        
        
    if parse6term:
        if 'theta' in pset.dindices.keys():
            print("parset2d.loadparset INFO - converting 6-term geometric to {abcdef} on input")
            
            pset = convert_linear(pset, fromabc=False)
            
    return pset

def convert_linear(pset=None, fromabc=False):

    """Utility - converts the linear part of a parameter set from {abcdef}
to {a,d,sx,sy,theta, beta} or vice versa. No parsing is done on the
input here, the user is expected to know what parameters they are
sending in.

    INPUTS

    pset = input parameter set object

    fromabc = assumed convention for input [a,b,c,d,e,f]. If False,
    {a,d,sx,sy,theta, beta} is assumed.

    OUTPUTS

    pout = pset object with the linear parameters converted. (Note:
    updated in-place)

    """

    # some defensive programming
    if pset is None:
        return pset

    if len(pset.pars) < 6:
        return pset
    
    # convenience view of model indices
    l6 = np.arange(6)  # we do use this later on
    #lmod = pset.lmodel[l6]

    # Get the indices corresponding to the {a,b,c,d,e,f} from our
    # model parameters
    lmod = sixterm2d.labcfrompars(pset.lmodel.size)
    
    # For the output, if we are converting [abcdef] then the output
    # will be reordered into [a,d,sx,sy,theta,beta], so we need to do
    # a little bit of index carpentry. Set that out here
    
    labels_out = ['xi_0', 'eta_0','s_x','s_y','theta','beta']
    methconv = sixterm2d.getpars
    if not fromabc:
        labels_out = ['a_%i' % (i) for i in lmod]
        methconv = sixterm2d.abcfromgeom

    # I think the least error-prone way to implement this is to
    # maintain side-by-side lists of dictionary keys, like so:
    keys_new = list(pset.dindices.keys() )
    for ireplace in l6:
        jrep = lmod[ireplace]
        keys_new[jrep] = labels_out[ireplace]

    # ... and now we can rebuild the indices dictionary:
    inds_new = {}
    for ikey in range(len(keys_new)):
        inds_new[keys_new[ikey]] = ikey

    pset.dindices = inds_new

    # ... ok now that we have the indices rebuilt, actually do the
    # conversion.
    parsconv = methconv(pset.pars[lmod])
    
    pset.pars[lmod] = parsconv[l6]

    # note we also need to ensure the model parameters are
    # appropriately populated.
    pset.partitionmodel()

    # ... and return
    return pset
    
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
                ntransf2=6, nnoise2=3, nshape2=2, nmix2=2, \
                testsub=True):

    """Compare two parameter sets. Useful when e.g. comparing truth to fit
parameters, where the two parameter sets can have differnet
configurations.

    Currently just does a lot of screen output while I think of ways
    to test this.

    """

    modelpp = np.arange(ntransf1)
    modelqq = np.arange(ntransf2)
    if testsub:
        ldum = np.arange(ntransf2, dtype='int')
        modelqq[ldum % 3 == 1] = 0.
        modelpp = np.arange(ntransf1, dtype=object) # to add None
        modelpp[1] = None
        
    PP = Pars1d(model=modelpp, \
                noise=np.arange(nnoise1)+10., \
                symm=np.arange(nshape1)+100., \
                mix=np.arange(nmix1)+1000.)

    QQ = Pars1d(modelqq, \
                noise=np.arange(nnoise2)+10., \
                symm=np.arange(nshape2)+100., \
                mix=np.arange(nmix2)+1000.)

    print(PP.pars)
    print(QQ.pars)
    
    # try merging the two
    Pair = Pairset(PP, QQ)
    # Pair.padset1toset2()

    if testsub:
        print("testcompare INFO - testing substitution:")
        Psub = Pair.sub1into2()

        print("set1 on 2: ", Pair.set1on2.model)
        print("set2 model:", Pair.set2.model)
        print("substitute:", Psub.model)
        
        # return
    
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
    print("set1on2:", Pair.set1on2.pars)
    print("set2:   ", Pair.set2.pars)
    Psub = Pair.arithmetic(Pair.set2, Pair.set1on2, np.subtract)
    print("2minus1:", Psub.pars)

    # try a ratio
    Prat = Pair.arithmetic(Psub, Pair.set2, np.divide)
    print(Prat.pars)

    print(Prat.model)

    # Try fractional difference
    Pfd = Pair.fracdiff()
    print(Pfd.pars)

    print(Pfd.model)

def testio(npars=6, nnoise=3, nshape=2, nmix=2):

    """Test routine for input/output to text"""

    # create a dummy parset
    transf = np.arange(6)
    pnoise = np.arange(nnoise)+10
    pshape = np.arange(nshape)+100
    pmix = np.arange(nmix) + 1000

    ppars = np.hstack(( transf, pnoise, pshape, pmix ))

    PP = Pars1d(ppars, nnoise, nshape, nmix)

    # write to disk
    PP.writeparset()

    # Now try generating a blank parset and reading in
    QQ = Pars1d()
    QQ.readparset()

    # Print some values
    print("model:", QQ.model)
    print("noise:", QQ.noise)
    print("symm:", QQ.symm)
    print("mix:", QQ.mix)
    print("===")
    print("lmodel:", QQ.lmodel)
    print("lnoise:", QQ.lnoise)
    print("lsymm:", QQ.lsymm)
    print("lmix:", QQ.lmix)

def testblank(transfname='Poly', deg=1, pathwrite='test_blankpars.txt', \
              nnoise=3, nshape=2, nmix=2):

    """Test generating and writing blank parameter set"""

    PP = Pars1d()
    PP.blankmodel(transfname, deg, nnoise, nshape, nmix)

    print("testblank INFO:")
    print(PP.pars)
    print(PP.model)
    print(PP.noise)
    print(PP.dindices)
    
    PP.writeparset(pathwrite)
    
def testconvert(pathin='test_parset_truths.txt', \
                pathout='test_conv.pars', fromabc=True):

    """Tests loading in a parameter set in one form, outputting into the
other.

    INPUTS

    pathin = parfile to read in

    pathout = converted parfile

    fromabc = converting FROM {abcdef}?

"""

    # To test abc --> geom:
    #
    # parsetd.testconvert('test_parset_truths.txt','test_conv.pars')

    # To test geom --> abc:
    #
    # parset.testconvert('test_6pars_geom.txt','test_6pars_abc.txt', False)
    
    # My original solution to this was too dependent on sixterm2d
    # plumbing. We know what the parameters and their orderings are
    # going to be, so keep it simple here.
    
    # Load, convert, write:
    parsin = loadparset(pathin, parse6term=False)
    parsin = convert_linear(parsin, fromabc)
    parsin.writeparset(pathout)

    return
    
def testloadandparse(pathpars='test_parset_truths.txt', \
                     pathout='test_parsed_truths.txt'):

    """Tests loading and parsing of parameter set that may include
{a,d,sx,sy,theta, beta} rather than {a,b,c,d,e,f}

    INPUTS

    pathpars = path to input parameters

    OUTPUTS

    pathout = path to possibly-converted parset file

    """

    pset = loadparset(pathpars, parse6term=True)
    pset.writeparset(pathout)
    
