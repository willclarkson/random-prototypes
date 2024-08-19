#
# examine2d.py
#

#
# WIC 2024-08-19 - methods for examining the output from MCMC runs
#

import os, time
import numpy as np

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

        # Populate shape attributes that we will need to access
        self.nsamples = 0
        self.npars = 0
        self.countsamples()

        # Dictionary of arguments that were passed to emcee.
        self.esargs = esargs
        self.pguess = None
        self.obstarg = None
        self.parset = None
        self.lnprior = None
        self.lnlike = None
        
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
        if not 'args' in self.esargs:
            return

        nargs = len(self.esargs['args'])
        
        if nargs > 0:
            self.pguess = self.esargs['args'][0]
        if nargs > 1:
            self.obstarg = self.esargs['args'][1]
        if nargs > 2:
            self.parset = self.esargs['args'][2]
        if nargs > 3:
            self.lnprior = self.esargs['args'][3]
        if nargs > 4:
            self.lnlike = self.esargs['args'][4]

        
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
