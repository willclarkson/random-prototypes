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

# For logistic regression on responsibilities
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

class Flatsamples(object):

    """Convenience object to hold flattened samples from MCMC, and various
methods to examine them. Optionally also comparisons with the 'truth'
parameters, if we are doing this on simulated data

Inputs:

    flat_samples = [nsamples, npars] array of flattened samples

    path_samples = path to load flattened samples

    esargs = {} = dictionary of mcmc run arguments

    ptruths = Pars1d object containing the truth parameters, if known

    """

    def __init__(self, flat_samples=np.array([]), path_samples='NA', \
                 esargs={}, ptruths=None):

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

        # truth parameters, if known
        self.ptruths = ptruths
        
        # Now some things we can compute on the samples
        self.ndata = 0.
        self.resps_samples = np.array([])   # (nsamples, ndata) - could be large
        self.resps_avg = np.array([])
        self.isfg = np.array([])

        # transformation object for the truth parameters
        self.transftruth = None

        # Some attributes relating to comparison with truth
        # parameters, if we know them
        self.regress_on_bg = False
        self.clf = None
        self.creg = 1.0e5 # logistic regression constant
        self.resp_sim = np.array([])
        self.resp_post = np.array([]) 
        self.dxyproj_truthpars = np.array([])
        
        # Unpack the arguments passed in
        self.unpack_esargs()
        self.countdata()
        self.getsimisfg()
        self.unpacktruths()
        
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

    def getsimisfg(self):

        """Gets the foreground/background IDs from the simulation"""

        self.isfg = self.inp_lnlike.obstarg.isfg

    def unpacktruths(self):

        """Creates transformation object from the 'truth' parameters so that
we can project the observed xy positions onto the target frame using
them"""

        # This makes a copy of the transformation object from the mcmc
        # runs, and swaps in the transformation parameters from the
        # truth parameters.

        # We have to actually have truth parameters for the
        # transformation to do something here.
        if not hasattr(self.ptruths, 'model'):
            return
        
        if self.ptruths is None:
            return

        if self.inp_pguess is None:
            return

        self.transftruth = copy.deepcopy(self.inp_pguess)
        self.transftruth.updatetransf(self.ptruths.model)

        # We ensure the positions in the input and target frames for
        # this object are consistent with the parameters
        self.transftruth.propagate()

        # Set instance attribute for the xy observations projected
        # onto the target frame using these truth parameters
        self.setobjontruth()
        
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

        print("examine2d.writeresps INFO - writing responsibilities:", \
              self.resps_samples.shape)
        
        np.save(pathresps, self.resps_samples)

    def loadresps(self, pathresps=''):

        """Load per-sample responsibilities from file.
    
Inputs:

    pathresps = path to responsibilities file

Returns: nothing


    """

        if len(pathresps) < 4:
            return

        self.resps_samples = np.load(pathresps)

        # recompute the averages along the samples
        self.resps_avg = np.mean(self.resps_samples, axis=0)

    def regressresps(self):

        """Runs a logistic regression on the model and simulated membership probabilities"""

        # Promoted this from showresps since it's useful to compute
        # anyway.
        if np.size(self.isfg) < 1 or np.size(self.resps_avg) < 1:
            return

        if np.size(self.isfg) != np.size(self.resps_avg):
            return

        # what do we regress on? (This seems stupidly duplicative,
        # except the plotting routine is outside the class)
        if self.regress_on_bg:
            self.resp_sim = 1.0 - self.isfg
            self.resp_post = 1.0 - self.resps_avg
        else:
            self.resp_sim = self.isfg
            self.resp_post = self.resps_avg

        self.clf = LogisticRegression(C=self.creg)
        self.clf.fit(self.resp_post[:, None], self.resp_sim)

    def setobjontruth(self):

        """If the truth parameters are present, extracts the positional differences between the target positions and the observed positions projected onto the target space using the truth parameters."""

        if not hasattr(self.transftruth, 'xytran'):
            return

        if not hasattr(self.inp_lnlike, 'obstarg'):
            return
        
        xyproj_truth = self.transftruth.xytran
        xytarg = self.inp_lnlike.obstarg.xy

        self.dxyproj_truthpars = xyproj_truth - xytarg

def showguess(esargs={}, fignum=2):

    """Plots up the guess transformation, noise, etc., before running mcmc.

Inputs:

    esargs = {} = dictionary of ensemble sampler arguments.

    fignum = matplotlib figure number to use

    """

    # Parse the keywords in the input arguments
    try:
        llike = esargs['args'][4]
        obstarg = llike.obstarg
        transf = llike.transf
    except:
        print("examine2d.showguess WARN - problem parsing ensemble sampler arguments")
        return

    # Views of necessary pieces: target frame...
    xytarg = obstarg.xy
    mags = obstarg.mags
    covtarg = obstarg.covxy
    isfg = obstarg.isfg

    # ... and observation frame
    xyobs = np.column_stack(( transf.x, transf.y ))
    covobs = transf.covxy

    # Positions transformed using the guess parameters: their deltas
    dxytran = transf.xytran - xytarg

    # Sum of the covariances transformed into the target frame
    covassume = covtarg + transf.covtran

    # Now a few more pieces depending on what the model includes.
    covextra = llike.covextra

    # Total covariance in the target frame from the lnlike model
    # params (unless everything is being fit, this will overlap with
    # one of the noise curves)
    covsum = llike.covsum

    # Compute the responsibilities according to the guess
    if np.size(llike.resps_fg) < 1:
        llike.calcresps()

    # Now set up the figure
    fig2=plt.figure(fignum, figsize=(8.25, 6.00))
    fig2.clf()

    # Positional residual plots
    ax37=fig2.add_subplot(337)
    ax38=fig2.add_subplot(338, sharey=ax37)
    ax39=fig2.add_subplot(339, sharey=ax37)
    ax34=fig2.add_subplot(334, sharex=ax37)
    ax31=fig2.add_subplot(331, sharex=ax37)

    # noise vs mag plots
    ax32 = fig2.add_subplot(332)
    ax33 = fig2.add_subplot(333)
    
    # Do the positional scatter plots...
    resid = ax37.scatter(dxytran[:,0], dxytran[:,1], c=mags, s=1)

    # try a stripe plot - dy vs x, y
    residyx = ax38.scatter(xytarg[:,0], dxytran[:,1], c=mags, s=1)
    residyy = ax39.scatter(xytarg[:,1], dxytran[:,1], c=mags, s=1)

    # dx vs x, y (rotated)
    residxx = ax34.scatter(dxytran[:,0], xytarg[:,0], c=mags, s=1)
    residxy = ax31.scatter(dxytran[:,0], xytarg[:,1], c=mags, s=1)

    # Some axis label carpentry
    ax37.set_xlabel(r'$\Delta \xi$')
    ax37.set_ylabel(r'$\Delta \eta$')

    ax38.set_xlabel(r'$\xi$')
    ax39.set_xlabel(r'$\eta$')
    ax34.set_ylabel(r'$\xi$')
    ax31.set_ylabel(r'$\eta$')

    # How do our responsibilities look?
    ax35 = fig2.add_subplot(335)
    dum35 = ax35.scatter(dxytran[:,0], dxytran[:,1], c=llike.resps_fg, \
                         cmap='inferno_r', s=16, edgecolor='k', vmax=1.)
    cbar35 = fig2.colorbar(dum35, ax=ax35, label=r'$f_fg$')
    
    # Colorbars
    for obj, ax in zip([resid, residyx, residyy, residxx, residxy], \
                       [ax37, ax38, ax39, ax34, ax31]):
        cbar = fig2.colorbar(obj, ax=ax, label='mag')

    # ... now do the vs magnitude plots
    for ax, quan, label in zip([ax32, ax33], \
                               [covobs[:,0,0], covtarg[:,0,0]], \
                               ['assumed (src)', 'assumed (target)']):
        dumobs = ax.scatter(mags, quan, c='#00274C', s=2, label=label)

    # On the "target frame" mag plot, show the quad sum of the target
    # assumed covariance and the covariance projected from the source
    # frame. This is what a hypothetical observer might adopt as the
    # "measured" covariance.
    cassume = ax33.scatter(mags, covassume[:,0,0], c='#00B2A9', \
                          label='assumed (total)', s=4)

    # Any extra noise under the current guess model parameters. KEEP
    # THIS IN even if zero, it's useful to ensure we're not adding
    # noise when we shouldn't be.
    if np.size(covextra) > 1:
        cextra = ax33.scatter(mags, covextra[:,0,0], c='#702082', \
                              label='Model extra', s=4)

    # Finally, the sum covariance assumed by the guess
    if np.size(covsum) > 1:
        csum = ax33.scatter(mags, covsum[:,0,0], c='#75988d', \
                            label='covsum', s=4)

    print("INFO:", np.size(covsum), np.size(covobs), np.size(covsum) is np.size(covobs))
        
    for ax in [ax32, ax33]:
        ax.set_xlabel('mag')
        ax.set_yscale('log')
    ax32.set_ylabel(r'$V_{xx}$')
    ax33.set_ylabel(r'$V_{\xi\xi}$')

    leg = ax33.legend()
    
    
def showresps(flatsamples=None, fignum=8, logx=False, creg=1.0e5, wantbg=True, \
              clobber=True):

    """Plots the simulated and mcmc responsibilities after an mcmc run.

Inputs:

    flatsamples = Flatsamples object including responsibilities and
    other data characteristics

    fignum = matplotlib figure number

    logx = plot horizontal on log scale

    creg = regularization parameter for logistic regression

    wantbg = we want to plot the probabilities of being part of the background

    clobber = redo the logistic regression even if previously done

Example call:

    esargs, runargs, showargs = mcmc2d.setupmcmc()

    # (Then run the mcmc sampler to produce flat_samples)

    FS = examine2d.Flatsamples(flat_samples, esargs=esargs, ptruths=showargs['truthset']['parset'])
    FS.loadresps('test_resps_samples.npy')

    showresps(FS)

    """

    # If the regression hasn't been done yet, do it now
    if flatsamples.clf is None or wantbg != flatsamples.regress_on_bg or clobber:
        flatsamples.creg = creg
        flatsamples.regress_on_bg = wantbg
        flatsamples.regressresps()

    # convenience views
    clf = flatsamples.clf
    resp_sim = flatsamples.resp_sim
    resp_post = flatsamples.resp_post

    # For overplotting the regression:
    xfine = np.linspace(np.min(resp_post), 1., 100)
    yfine = expit(xfine * clf.coef_ + clf.intercept_).ravel()
    
    # sim vs recovered responsibilities
    fig8 = plt.figure(fignum)
    fig8.clf()
    ax8 = fig8.add_subplot(111)

    dum8 = ax8.scatter(resp_post, resp_sim, alpha=0.5, label='responsibilities')

    reg8 = ax8.plot(xfine, yfine, c='#75988d', zorder=10, alpha=0.7, \
                    ls='--', lw=1, \
                    label='logistic regression')
    
    # Legend position control
    legloc = 'center right'
    if logx:
        legloc = 'center left'
        ax8.set_xscale('log')
    
    leg = ax8.legend(fontsize=8, loc=legloc)

    # hack for labels
    slabel = 'foreground'
    if wantbg:
        slabel = 'background'
    
    ax8.set_xlabel('p(is %s), MCMC' % (slabel))
    ax8.set_ylabel('Simulated as %s' % (slabel))

    if np.size(flatsamples.dxyproj_truthpars) < 1:
        return

    # now set up the scatter plot
    fig9 = plt.figure(fignum+1, figsize=(4,6))
    fig9.clf()
    ax91 = fig9.add_subplot(211)
    ax92 = fig9.add_subplot(212, sharex=ax91, sharey=ax91)
    dsim = ax91.scatter(flatsamples.dxyproj_truthpars[:,0], \
                        flatsamples.dxyproj_truthpars[:,1], \
                            c=resp_sim, alpha=0.7, s=2, vmax=1.0)
    
    dpost = ax92.scatter(flatsamples.dxyproj_truthpars[:,0], \
                         flatsamples.dxyproj_truthpars[:,1], \
                         c=resp_post, alpha=0.7, s=2, vmax=1.0)

    for obj, ax, label in zip([dsim, dpost], [ax91, ax92], \
                              ['Generated', 'Avg(samples)']):
        cbar = fig9.colorbar(obj, ax=ax, label=label)

        ax.set_xlabel(r'$\Delta \xi$')
        ax.set_ylabel(r'$\Delta \eta$')

    fig9.subplots_adjust(left=0.3, bottom=0.11, hspace=0.4)
        
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
