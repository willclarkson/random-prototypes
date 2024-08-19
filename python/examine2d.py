#
# examine2d.py
#

#
# WIC 2024-08-19 - methods for examining the output from MCMC runs
#

import os, time
import numpy as np
import copy

# for corner plots
import matplotlib.pylab as plt
plt.ion()
import corner

class Flatsamples(object):

    """Convenience object to hold flattened samples from MCMC, and various
methods to examine them"""

    def __init__(self, flat_samples=np.array([]), path_samples='NA', \
                 esargs={}):

        self.flat_samples = np.copy(flat_samples)
        self.path_samples = path_samples[:]

        # if no samples passed in, tries to load them from disk
        if np.size(self.flat_samples) < 1:
            self.loadsamples()

        # Populate the shape attributes that we will need to access
        self.nsamples = 0
        self.npars = 0
        self.countsamples()
        
        # Dictionary of arguments that were passed to emcee.
        self.esargs = esargs

        # Quantities we get from this dictionary
        self.inp_pguess = None
        self.inp_obstarg = None
        self.inp_parset = None
        self.inp_lnprior = None
        self.inp_lnlike = None

        # Now some things we can compute on the samples
        self.ndata = 0.
        self.resps_samples = np.array([])   # (nsamples, ndata) - could be large
        self.resps_avg = np.array([]) 

        # Unpack the arguments passed in
        self.unpack_esargs()
        self.countdata()
        
    def loadsamples(self):

        """Loads the flattened samples"""

        if len(self.path_samples) < 4:
            return

        try:
            self.flat_samples = np.load(self.path_samples())
        except:
            nosamples = True

    def countsamples(self):

        """Gets the samples shape"""

        if np.ndim(self.flat_samples) < 2:
            return

        self.nsamples, self.npars = np.shape(self.flat_samples)

    def unpack_esargs(self):

        """Unpacks the parts of the emcee arguments we will need to compute
various things

        """

        # What quantitities do we have?
        ekeys = self.esargs.keys()
        
        if len(ekeys) < 1:
            return

        # Views of the objects used in the emcee run
        if not 'args' in ekeys:
            return

        nargs = len(self.esargs['args'])
        
        if nargs > 0:
            self.inp_pguess = self.esargs['args'][0]
        if nargs > 1:
            self.inp_obstarg = self.esargs['args'][1]
        if nargs > 2:
            self.inp_parset = self.esargs['args'][2]
        if nargs > 3:
            self.inp_lnprior = self.esargs['args'][3]
        if nargs > 4:
            self.inp_lnlike = self.esargs['args'][4]

    def countdata(self):

        """Gets the data dimensions from the lnlike object"""

        # run this without try/except first
        self.ndata = self.inp_lnlike.obstarg.xy.shape[0]
        
    def computeresps(self, samplesize=-1, keepmaster=True, \
                     ireport=1000, Verbose=True, \
                     pathresps='test_resps_samples.npy'):

        """Computes foreground probabilities for every sample.

Inputs:

        samplesize = number of samples to compute

        keepmaster = store all the samples in memory? [nsamples, ndata]
        
        ireport = report every this many rows

        Verbose = report to screen

        pathresps = paths to write master samples responsibilities file

Returns: 

        nothing

Example call:

        FS = examine2d.Flatsamples(flat_samples, esargs=esargs)
        FS.computeresps()

"""

        # The lnlike object must be populated
        if self.inp_lnlike is None:
            return

        # local copy of the ln(like) object that we can modify
        # in-place
        lnlike = copy.deepcopy(self.inp_lnlike)

        # how many datapoints?
        self.countdata()
        if self.ndata < 1:
            return
        
        # How many of these do we want to do now?
        if samplesize < 1 or samplesize >= self.nsamples:
            imax = self.nsamples
        else:
            imax = samplesize

        norm = 0.
        self.resps_avg = np.zeros(self.ndata)
            
        if keepmaster:
            self.resps_samples = np.zeros(( imax, self.ndata ))

        t0 = time.time()
        if Verbose:
            print("examine2d.computeresps INFO - starting responsibility loop...")
            
        for isample in range(imax):

            # update the parameter-set and recompute everything
            lnlike.parset.updatepars(self.flat_samples[isample])
            lnlike.updatelnlike(lnlike.parset)
            lnlike.calcresps()

            # increment the responsibilities and the norm
            norm += 1.
            self.resps_avg += lnlike.resps_fg
            
            # If we want to store all the responsibilities per sample
            if keepmaster:
                self.resps_samples[isample] = lnlike.resps_fg

            # report out every so often
            if isample % ireport < 1 and isample > 0 and Verbose:
                telapsed = time.time() - t0
                itpersec = float(isample)/telapsed
                tremain = 0.
                if itpersec > 0.:
                    tremain = float(imax)/itpersec
                print("examine2d.computeresps INFO - iteration %i of %i after %.2e seconds: %.1f it/sec. Est %.2e sec remain" \
                      % (isample, imax, telapsed, itpersec, tremain), end='\r')

        # For reporting
        t1 = time.time()
        if Verbose:
            print("")
            print("examine2d.computeresps INFO - loops took %.2e seconds for %.2e samples" % (t1-t0, imax))
        
        # Evaluate the average
        self.resps_avg /= norm

        # Since those loops can take a while, write the
        # responsibilities to disk by default
        self.writeresps(pathresps)
        
    def writeresps(self, pathresps='test_resps_samples.npy'):

        """Utility - write responsibilities array to disk (can be large).

Inputs:
        
        test_resps_samples.npy  =  path to write to

Returns: Nothing

        """

        if len(pathresps) < 1:
            return

        if np.size(self.resps_samples) < 1:
            return

        np.save(pathresps, self.resps_samples)
        
def showcorner(flat_samples=np.array([]), \
               labels=None, truths=None, \
               fignum=4, pathfig='test_corner_oo.png', \
               minaxesclose=20):

    """Corner plot of flattened samples from mcmc run.

Inputs:

    flat_samples = [nsamples, npars] array of flattened samples

    labels = [npars] array of string labels for each parameter

    truths = [npars] array of "truth" values (if known)

    fignum = matplotlib figure number to use 

    pathfig = path to save the corner plot image

    minaxesclose = closes the figure (after saving to disk) if there
    are >= minaxesclose quantities to plot. Useful to free up memory.

Returns:
    
    None.

Example call:

    examine2d.showcorner(flat_samples, **showargs['corner'])

    """

    if np.size(flat_samples) < 1:
        return

    # Label keyword arguments
    label_kwargs = {'fontsize':8, 'rotation':'horizontal'}
    
    # I prefer to set the figure up and then make the corner plot in
    # the figure:
    fig4 = plt.figure(4, figsize=(9,7))
    fig4.clf()

    print("examine2d.showcorner INFO - plotting corner plot...")
    corn = corner.corner(flat_samples, labels=labels, truths=truths,\
                         truth_color='b', fig=fig4, labelpad=0.7, \
                         use_math_text=True, \
                         label_kwargs = label_kwargs)
    fig4.subplots_adjust(bottom=0.2, left=0.2, top=0.95)

    # Adjust the label size
    for ax in fig4.get_axes():
        ax.tick_params(axis='both', labelsize=5)

    # save figure to disk?
    if len(pathfig) > 3:
        fig4.savefig(pathfig)

    # If the samples have "high" dimension then the corner plot may
    # slow the interpreter. So we close the figure if "large":
    if flat_samples.shape[-1] > minaxesclose:
        plt.close(fig4)
