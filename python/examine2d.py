#
# examine2d.py
#

#
# WIC 2024-08-19 - methods for examining the output from MCMC runs
#

import os, time
import numpy as np
import numpy.ma as ma

import copy

# For computing some needed pieces
from binstats2d import Binstats
import noisemodel2d

# For reordering and processing flat_samples
import sixterm2d

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

    ptruths = Pars1d object containing the truth parameters, if
    known. Overridden by showargs if ptruths is present there.

    log_probs = [nsamples] array of log-probabilities returned by the sampler

    path_log_probs = path to log_probs

    showargs = {} = dictionary of arguments that show routines may
    need (not strict yet)

    """

    def __init__(self, flat_samples=np.array([]), path_samples='NA', \
                 esargs={}, ptruths=None, log_probs=np.array([]), \
                 path_log_probs='NA', showargs={}):

        self.flat_samples = np.copy(flat_samples)
        self.path_samples = path_samples[:]

        # if no samples passed in, tries to load them from disk
        if np.size(self.flat_samples) < 1:
            self.loadsamples()

        # log probs
        self.log_probs = np.copy(log_probs)
        self.path_log_probs = path_log_probs[:]
        if np.size(self.log_probs) < 1:
            self.loadlogprobs()
            
        # Populate the shape attributes that we will need to access
        self.nsamples = 0
        self.npars = 0
        self.countsamples()
        
        # Dictionary of arguments that were passed to emcee.
        self.esargs = esargs

        # Dictionary of arguments sent to "show" routines
        self.showargs = showargs
        
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

        # covariance among flat samples parameters
        self.param_covars = np.array([])
        self.lstsq_covars = np.array([])
        self.labels_transf = None

        # Statistics on projected deltas, binned by magnitude
        self.binstats_fg = None
        self.binstats_bg = None
        
        # Percentiles of the computed uncertainties, options about
        # what those entries are to be
        self.pctiles = [50., 15.9, 84.1, 0.135, 99.865]
        self.covsum_samples_pctiles = np.array([])
        self.covsum_samples_has_corrxy = True
        self.covsum_samples_has_ryx = True
        
        # Unpack the arguments passed in
        self.unpack_showargs()
        self.unpack_esargs()
        self.countdata()
        self.getsimisfg()
        self.unpacktruths()
        
        # Compute the parameter covariances among the flat samples
        self.computecovars()

        # Compute binned statistics
        self.computebinnedstats()
        
    def loadsamples(self):

        """Loads the flattened samples"""

        if len(self.path_samples) < 4:
            return

        try:
            self.flat_samples = np.load(self.path_samples)
        except:
            nosamples = True

    def loadlogprobs(self):

        """Loads log probabilities"""

        if len(self.path_log_probs) < 4:
            return

        try:
            self.log_probs = np.load(self.path_log_probs)
        except:
            nologprobs = True
        
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

    def unpack_showargs(self):

        """Unpacks arguments passed in"""

        if not 'guess' in self.showargs.keys():
            return

        if 'labels_transf' in self.showargs['guess'].keys():
            self.labels_transf = self.showargs['guess']['labels_transf']

        if 'lstsq_uncty_formal' in self.showargs['guess'].keys():
            self.lstsq_covars = self.showargs['guess']['lstsq_uncty_formal']

        # Imports ptruths if present in input showargs
        if 'truthset' in self.showargs.keys():
            if 'parset' in self.showargs['truthset'].keys():
                self.ptruths = self.showargs['truthset']['parset']
            
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

        # Safety valve - if the model doesn't have a mixture, there's
        # no need to recalculate everything:
        nmix = np.size(self.inp_lnlike.parset.mix)
        if nmix < 1:
            self.resps_avg += 1.
            if keepmaster:
                self.resps_samples += 1.
            print("examine2d.computeresps INFO - no mixture, all responsibilities 1.0")
            return

            
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
                    tremain = float(imax-isample)/itpersec
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

        xyproj_truth = self.transftruth.xytran
        
        # 2024-10-02 UPDATE - now use the likelihood object's xytarg,
        # since we might be comparing in a frame other than the
        # observation frame.        
        #if not hasattr(self.inp_lnlike, 'obstarg'):
        #    return
        
        # xytarg = self.inp_lnlike.obstarg.xy
        
        xytarg = self.inp_lnlike.xytarg
        
        self.dxyproj_truthpars = xyproj_truth - xytarg

    def computecovars(self):

        """Computes covariances between the parameter samples"""

        if np.size(self.flat_samples) < 1:
            return

        self.param_covars = np.cov(self.flat_samples, rowvar=False)
        
    def computesamplescovars(self, ireport=1000, Verbose=True, samplesize=-1):

        """Computes the uncertainties (as covariances) in the target plane for
the samples.

Inputs:

        ireport = report every this many loops

        Verbose = produce screen output

        samplesize = number of loops to do (useful in development)

Returns:

        None - updates class attribute self.covsum_samples_pctiles , a
        [3, ndata, 2, 2] array that contains the [16., 50., 84]'th
        percentile covariance for each datapoint.

        """

        # Do this the simple (if slow) way: loop through the samples
        # and use the exact same methods that were used to evaluate
        # these quantities. Currently these methods are inside the
        # Like() object.

        # First we make a copy of the Like() object that was used in
        # the simulation
        if not hasattr(self.inp_lnlike,'obstarg'):
            return

        llike = copy.deepcopy(self.inp_lnlike)

        # How far are we going to go?
        imax = np.copy(self.nsamples)
        if 0 < samplesize < self.nsamples:
            imax = samplesize
        
        # Now create a master array in which to store the covariance
        # estimates 
        covs_samples = np.zeros((self.nsamples, self.ndata, 2, 2))
        if Verbose:
            print("computesamplescovars INFO - covs shape:", covs_samples.shape)

            
        # Loop through the samples comes here.
        t0 = time.time()
        for isample in range(imax):
            llike.parset.updatepars(self.flat_samples[isample])
            llike.updatesky(llike.parset)

            covs_samples[isample] = np.copy(llike.covsum)

            # Report out every so often
            if isample % ireport < 1 and isample > 0:
                telapsed = time.time() - t0
                itpersec = float(isample)/telapsed
                tremain = 0.
                if itpersec > 0.:
                    tremain = float(imax-isample)/itpersec
                print("computesamplescovars INFO - iteration %i of %i after %.2e seconds: %.1f it/sec. Est %.2e sec remain" \
                      % (isample, imax, telapsed, itpersec, tremain), end='\r')

        # are off-diagonals to be correlation coefficients:
        if self.covsum_samples_has_corrxy:
            rho = covs_samples[:,:,0,1] / \
                np.sqrt(covs_samples[:,:,0,0] * covs_samples[:,:,1,1])
            covs_samples[:,:,0,1] = rho
            covs_samples[:,:,1,0] = rho

        # Do we want stdy/stdx instead of vy
        if self.covsum_samples_has_ryx:
            covs_samples[:,:,1,1] = np.sqrt(covs_samples[:,:,1,1] / covs_samples[:,:,0,0])

        # compute the median and percentiles here
        if Verbose:
            print("")
            print("computesamplescovars INFO - computing percentiles...")
            
        self.covsum_samples_pctiles = \
            np.percentile(covs_samples, self.pctiles, axis=0)

        if Verbose:
            print("computesamplescovars INFO - ... done:", \
                  self.covsum_samples_pctiles.shape)
        
        # Force free the memory
        covs_samples = None

    def computebinnedstats(self, threshfg=0.9, threshbg=0.2, nbins=10, \
                           minperbin=16):

        """Computes binned statistics on datapoints identified as foreground
and background"""

        # Read the needed pieces from the lnlike object
        if not hasattr(self.inp_lnlike,'obstarg'):
            return

        if not hasattr(self.inp_lnlike,'transf'):
            return

        # We now may be performing the comparison in a frame other
        # than the observation frame. So:
        xytarg = self.inp_lnlike.xytarg
        
        mags = self.inp_lnlike.obstarg.mags
        # xytarg = self.inp_lnlike.obstarg.xy
        resps_fg = self.inp_lnlike.resps_fg
        xytran = self.inp_lnlike.transf.xytran

        dxytran = xytran - xytarg

        # The responsibilities might not be present in the lnlike
        # object but in the flatsamples object. Look for them here.
        if np.size(resps_fg) < 1:
            resps_fg = np.copy(self.resps_avg)
        
        # Now identify foreground and background objects. Assume all
        # objects are foreground unless we have already computed the
        # responsibilities.
        bfg = np.isfinite(mags)
        bbg = np.array([])
        if np.size(resps_fg) > 0:
            bbg = (bfg) & (resps_fg < threshbg)
            bfg = (bfg) & (resps_fg > threshfg)

        # Nothing to do if there are no foreground objects (neat trick...)
        if np.sum(bfg) < 1:
            print("computebinnedstats WARN - no foreground objects")
            return
            
        # Set up binstats objects and compute
        if np.sum(bfg) > 1:
            self.binstats_fg = Binstats(mags[bfg], dxytran[bfg], \
                                        minperbin, nbins=nbins)

        if np.sum(bbg) > 1:
            self.binstats_bg = Binstats(mags[bbg], dxytran[bbg], \
                                        minperbin)
            
        
            
def showguess(esargs={}, fignum=2, npermagbin=36, respfg=0.8, nmagbins=10, \
              pathfig='test_guess_deltas.png', showargs={}, \
              usetruths=False, showquiver=True):

    """Plots up the guess transformation, noise, etc., before running mcmc.

Inputs:

    esargs = {} = dictionary of ensemble sampler arguments.

    fignum = matplotlib figure number to use

    npermagbin = number of points per magnitude bin (when estimating
    delta vs mag for foreground objects)

    respfg = threshold for identifying objects as "foreground" when
    estimating the running statistics

    nmagbins = number of magnitude bins (for running statistics). If >0, overrides npermagbin.

    showargs = {} = arguments returned from setupmcmc that include
    truth parameters

    usetruths [T/F] - use truth parameters for plots if we have them
    
    showquiver [T/F] - plot quiver plot showing residuals

    """

    # DO NOT REFACTOR the binning into Flatsamples: we want this to
    # work before we have flatsamples populated!
    
    # Parse the keywords in the input arguments
    try:
        llike = copy.deepcopy(esargs['args'][4])
        obstarg = llike.obstarg
        transf = llike.transf
    except:
        print("examine2d.showguess WARN - problem parsing ensemble sampler arguments")
        return

    # it would be very useful to plot the predictions of the truth
    # parameters if we have them, particularly for the vs-mag
    # graph...
    ptruth = None
    if 'truthset' in showargs.keys():
        if 'parset' in showargs['truthset'].keys():
            ptruth = showargs['truthset']['parset']

            # If asked to use the truth parameters, copy them in to
            # the llike object and update
            if usetruths:
                print("showguess INFO - using truth parameters for plots")
                llike.updatesky(ptruth)
                transf = llike.transf

                
    # Views of necessary pieces: target frame...
    #xytarg = obstarg.xy
    #covtarg = obstarg.covxy
    xytarg = llike.xytarg
    covtarg = llike.covtarg
    mags = obstarg.mags
    isfg = obstarg.isfg

    # ... and observation frame
    xyobs = np.column_stack(( transf.x, transf.y ))
    covobs = transf.covxy

    # labels for coordinates
    labelxsrc = r'$X$'
    labelysrc = r'$Y$'
    labelxtran = r'$\xi$'
    labelytran = r'$\eta$'
    if hasattr(transf,'labelxtran'):
        labelxtran = transf.labelxtran[:]
    if hasattr(transf,'labelytran'):
        labelytran = transf.labelytran[:]
    if hasattr(transf,'labelx'):
        labelxsrc = transf.labelx
    if hasattr(transf,'labely'):
        labelysrc = transf.labely
        
    labeldxtran = r'$\Delta %s$' % (labelxtran.replace('$',''))
    labeldytran = r'$\Delta %s$' % (labelytran.replace('$',''))

    labelvxxsrc = r'$V_{%s%s}$' % (labelxsrc.replace('$',''), \
                                    labelxsrc.replace('$',''))

    labelvxxtran = r'$V_{%s%s}$' % (labelxtran.replace('$',''), \
                                    labelxtran.replace('$',''))
    
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

    # Compute binned statistics on "foreground" objects
    bfg = llike.resps_fg > respfg
    BG = Binstats(mags[bfg], dxytran[bfg], npermagbin, nbins=nmagbins)
    magbins, dxymeans, dxycovs, counts = BG.getstats()

    # Try objects identified as background
    bbg = llike.resps_fg < 0.2 # was 0.2
    print("showguess DEBUG - bbg:", np.sum(bbg))
    magbins_bg = np.array([])
    if np.sum(bbg) > 20:
        BB = Binstats(mags[bbg], dxytran[bbg], nbins=int(nmagbins*0.7))
        magbins_bg, dxymeans_bg, dxycovs_bg, counts_bg = BB.getstats()
    
    # A couple of things useful to standardize
    fontsz=10
    
    # Now set up the figure
    fig2=plt.figure(fignum, figsize=(9., 7.))
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

    # truth parameters if we have them
    ax36=None
    magbins_fg_truth = np.array([])                            
    if ptruth is not None:
        ax36 = fig2.add_subplot(336)
        ltruth = copy.deepcopy(llike)
        ltruth.updatesky(ptruth)

        # Note that the binned statistics for the "target" plot are
        # post-fit responsibilities. But if we have truths, then we
        # also know which *are* the foreground objects. So show them
        # too.
        bfg_truth = isfg > 0
        BT = Binstats(mags[bfg_truth], dxytran[bfg_truth], \
                      npermagbin, nbins=nmagbins)
        magbins_fg_truth, _, dxycovs_fg_truth, counts_fg_truth = BT.getstats()
        
    # Do the positional scatter plots...
    resid = ax37.scatter(dxytran[:,0], dxytran[:,1], c=mags, s=1)

    # try a stripe plot - dy vs x, y
    residyx = ax38.scatter(xytarg[:,0], dxytran[:,1], c=mags, s=1)
    residyy = ax39.scatter(xytarg[:,1], dxytran[:,1], c=mags, s=1)

    # dx vs x, y (rotated)
    residxx = ax34.scatter(dxytran[:,0], xytarg[:,0], c=mags, s=1)
    residxy = ax31.scatter(dxytran[:,0], xytarg[:,1], c=mags, s=1)

    # Some axis label carpentry
    for ax in [ax37, ax31, ax34]:
        ax.set_xlabel(labeldxtran) # now that there's room
    ax37.set_ylabel(labeldytran)

    ax38.set_xlabel(labelxtran)
    ax39.set_xlabel(labelytran)
    ax34.set_ylabel(labelxtran)
    ax31.set_ylabel(labelytran)

    # How do our responsibilities look?
    ax35 = fig2.add_subplot(335)
    dum35 = ax35.scatter(dxytran[:,0], dxytran[:,1], c=llike.resps_fg, \
                         cmap='inferno', s=4, edgecolor=None, vmax=1., \
                         alpha=0.7)
    cbar35 = fig2.colorbar(dum35, ax=ax35, label=r'$f_fg$')
    ax35.set_xlabel(labeldxtran)
    
    # Colorbars
    for obj, ax in zip([resid, residyx, residyy, residxx, residxy], \
                       [ax37, ax38, ax39, ax34, ax31]):
        cbar = fig2.colorbar(obj, ax=ax)

    # label where we can fit it in
    ax31.set_title('Colors: mag')
        
    # ... now do the vs magnitude plots
    for ax, quan, label, color in zip([ax32, ax33], \
                               [covobs[:,0,0], covtarg[:,0,0]], \
                                       ['assumed (src)', 'assumed (target)'], \
                                       ['#702082','#00274C']):
        dumobs = ax.scatter(mags, quan, c=color, s=2, label=label)

    # On the "target frame" mag plot, show the source frame
    # uncertainty, propagated out to the target frame.
    ctransf = ax33.scatter(mags, transf.covtran[:,0,0], c='#702082', \
                           label='source transformed', s=2, alpha=0.6)
        
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
        cextra = ax33.scatter(mags, covextra[:,0,0], c='#D86018', \
                              label='Model extra', s=4)

    # Show, the sum covariance assumed by the "guess"
    if np.size(covsum) > 1:
        csum = ax33.scatter(mags, covsum[:,0,0], c='#75988d', \
                            label='covsum', s=4)

    # Show the statistics for the covariance against magnitude, for
    # "foreground" objects
    cemp = ax33.scatter(magbins, dxycovs[:,0,0], c='#9A3324', \
                        label='fg, %i / bin' % (counts[0]), s=9)

    # try the same for background objects
    if np.size(magbins_bg) > 1:
        cbg = ax33.scatter(magbins_bg, dxycovs_bg[:,0,0], c='#D86018', \
                           label='bg, %i / bin' % (counts_bg[0]), s=9, marker='s')

    # Truth parameters if we have them
    if ax36 is not None:
        # show the data again...
        dum = ax36.scatter(magbins, dxycovs[:,0,0], c='#9A3324', \
                           label='fg, %i / bin' % (counts[0]), s=9, \
                           zorder=25)

        ctarg2 = ax36.scatter(mags, ltruth.covtarg[:,0,0], c='#00274C', \
                              label='target (truth)', s=2)
        
        ctransf2 = ax36.scatter(mags, ltruth.covtran[:,0,0], c='#702082', \
                           label='source transformed', s=2, alpha=0.6)
        
        cextra2 = ax36.scatter(mags, ltruth.covextra[:,0,0], c='#D86018', \
                              label='Model extra', s=4)
        
        # ... now overplot the covariances using the truth model
        dum2 = ax36.scatter(mags, ltruth.covsum[:,0,0], c='#75988d', \
                            label='covsum, truth model', zorder=10, s=4)

        # ... and the covariances using those objects simulated as foreground
        dumt = ax36.scatter(magbins_fg_truth, \
                            dxycovs_fg_truth[:,0,0], \
                            c='r', \
                            label='fg (sim), %i / bin' % (counts_fg_truth[0]), \
                            s=9, \
                            zorder=25)
        
        ax36.set_title('Truth model')
        ax36.set_yscale('log')
        
    # now label the vs-mag plots
        
    for ax in [ax32, ax33]:
        ax.set_xlabel('mag')

    # We always want to log-scale the target frame, we only want to
    # log-scale the source frame if it has nonzero values
    bpos = covobs[:,0,0] > 0.
    if np.sum(bpos) > 2:
        ax32.set_yscale('log')
    ax33.set_yscale('log')
    ax32.set_ylabel(labelvxxsrc)
    ax33.set_ylabel(labelvxxtran)

    leg = ax33.legend(fontsize=5)

    ax32.set_title('Source frame', fontsize=fontsz)
    ax33.set_title('Target frame', fontsize=fontsz)

    # a few cosmetic things
    fig2.subplots_adjust(hspace=0.4, wspace=0.4)

    # save to disk
    if len(pathfig) > 3:
        fig2.savefig(pathfig)

    if not showquiver:
        return

    fig3 = plt.figure(fignum+1, figsize=(8., 6.))
    fig3.clf()
    ax31 = fig3.add_subplot(223)

    # Ranges for quiver plot
    # mmin = np.min(mags)
    # mmax = np.max(mags)
    
    quiv_fg = ax31.quiver(xytarg[bfg, 0], xytarg[bfg, 1], \
                          dxytran[bfg,0], dxytran[bfg,1], \
                          mags[bfg], cmap='viridis_r')
    cbar1 = fig3.colorbar(quiv_fg, ax=ax31)

    # Show a marginal plot for the foreground objects
    ax33 = fig3.add_subplot(221)
    blah33 = ax33.scatter(xytarg[bfg, 0], dxytran[bfg,0], \
                          c=mags[bfg], s=9, cmap='viridis_r')

    ax34 = fig3.add_subplot(224)
    blah34 = ax34.scatter(xytarg[bfg, 1], dxytran[bfg,1], \
                          c=mags[bfg], s=9, cmap='viridis_r')

    cbar3 = fig3.colorbar(blah33, ax=ax33)
    cbar4 = fig3.colorbar(blah34, ax=ax34)
    
    if np.sum(bbg) > 0:
        ax32 = fig3.add_subplot(222, sharex=ax31, sharey=ax31)

        quiv_bg = ax32.quiver(xytarg[bbg, 0], xytarg[bbg, 1], \
                              dxytran[bbg,0], dxytran[bbg,1], \
                              mags[bbg], cmap='viridis_r')
        cbar2 = fig3.colorbar(quiv_bg, ax=ax32)

    for ax in [ax31, ax32]:
        ax.set_xlabel(labelxtran)
        ax.set_ylabel(labelytran)

    ax33.set_xlabel(labelxtran)
    ax33.set_ylabel(labeldxtran)
    ax34.set_xlabel(labelytran)
    ax34.set_ylabel(labeldytran)
        

    
    for ax in [ax31, ax33, ax34]:
        ax.set_title('foreground')
    ax32.set_title('outliers')
        
    fig3.subplots_adjust(left=0.18, bottom=0.17, hspace=0.4, wspace=0.49)
    
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

    # Return if no responsibilities yet
    if np.size(flatsamples.resps_avg) < 1:
        print("examine2d.showresps WARN - responsibilities not populated yet")
        print("examine2d.showresps WARN - suggest: FS.computeresps() .")
        return
    
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
        print("dxyproj_truthpars:", np.shape(flatsamples.dxyproj_truthpars))
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

        ax.set_xlabel(labeldxtran)
        ax.set_ylabel(labeldytran)

    fig9.subplots_adjust(left=0.3, bottom=0.11, hspace=0.4)

def showunctysamples(flatsamples=None, fignum=7):

    """Shows the range of uncertainty (variance) samples as percentiles.

    Example call:

    FS = examine2d.Flatsamples(flat_samples, esargs=esargs, log_probs=lnprobs, showargs=showargs)
    FS.computebinnedstats()
    FS.computesamplescovars()
    examine2d.showunctysamples(FS)

    """

    if flatsamples is None:
        return

    if np.size(flatsamples.covsum_samples_pctiles) < 1:
        print("showunctysamples WARN - sample percentiles not populated: covsum_samples_pcctiles")
        return

    if np.size(flatsamples.pctiles) < 1:
        print("showunctysamples WARN - percentile levels not populated: pctiles")
        return

    # must have input magnitudes to show
    if not hasattr(flatsamples, 'inp_lnlike'):
        return

    # magnitudes, along with sorting index
    mags = flatsamples.inp_lnlike.obstarg.mags
    lmag = np.argsort(mags)
    
    # Convenience views
    pctiles = flatsamples.pctiles
    levels = flatsamples.covsum_samples_pctiles

    print("showunctysamples INFO - mags, pctiles, levels:", \
          mags.shape, np.shape(pctiles), levels.shape)

    # compute binned statistics if not already done:
    if flatsamples.binstats_fg is None:
        flatsamples.computebinnedstats()

    binmag, _, bincov, bincounts = flatsamples.binstats_fg.getstats()
    
    # Now set up the figure:
    fig7 = plt.figure(fignum)
    fig7.clf()
    ax7xx = fig7.add_subplot(221)
    ax7yy = fig7.add_subplot(224)
    ax7xy = fig7.add_subplot(222)

    # entry in the [npctile, ndata,2,2] covariance matrices
    icov = [0,1,0]
    jcov = [0,1,1]
    lax = [ax7xx, ax7yy, ax7xy]

    # linestyles for each percentile
    lls = ['-','--','--', '-.', '-.']
    lws = [2,1,1,1,1]
    colos = ['k', '0.3', '0.3', '0.3', '0.3']
    labls = ['median', r'"$1\sigma$"','',r'"$3\sigma$"','']
    
    # Plot the percentiles for the covsums
    for ax, i, j in zip(lax, icov, jcov):
        for k in range(levels.shape[0]):
            var = levels[k,:,i,j]
            dum = ax.plot(mags[lmag], var[lmag], \
                          ls=lls[k], lw=lws[k], \
                          color=colos[k], label=labls[k])

    # Overplot the binned datapoints
    sbin = 'fg, %i / bin' % (bincounts[0])
    ax7xx.scatter(binmag, bincov[:,0,0], c='#9A3324', s=9, \
                  label=sbin)

    overyy = np.copy(bincov[:,1,1])
    if flatsamples.covsum_samples_has_ryx:
        overyy = np.sqrt(bincov[:,1,1]/bincov[:,0,0])

    overxy = np.copy(bincov[:,0,1])
    if flatsamples.covsum_samples_has_corrxy:
        overxy = bincov[:,0,1]/np.sqrt(bincov[:,0,0]*bincov[:,1,1])

    ax7yy.scatter(binmag, overyy, c='#9A3324', s=9, \
                  label=sbin)

    ax7xy.scatter(binmag, overxy, c='#9A3324', s=9, \
                  label=sbin)

    
    # legends for the axes
    leg7xx = ax7xx.legend(fontsize=8)
    
    # The vxx axis label
    ax7xx.set_yscale('log')

    # The vyy axis label
    ax7yy.set_ylabel(r'$\sqrt{V_{\eta \eta} / V_{\xi \xi}}$')
    if not flatsamples.covsum_samples_has_ryx:
        ax7yy.set_yscale('log')
        ax7yy.set_ylabel(r'$V_{\eta \eta}$')

    # the vxy axis label
    ax7xy.set_ylabel(r'$\rho_{\xi \eta}$')
    if not flatsamples.covsum_samples_has_corrxy:
        ax7xy.set_ylabel(r'$V_{\xi \eta}$')
        
    # Control labels out of the loop
    for ax in [ax7xx, ax7yy, ax7xy]:
        ax.set_xlabel('mag')

    # Cosmetics
    fig7.subplots_adjust(wspace=0.3, hspace=0.3)
                        
                         
def shownoisesamples(flatsamples=None, nshow=100, fignum=9, \
                     logy=True, showvar=True, \
                     cmap='inferno_r', jaux=2, \
                     alpha=0.1, \
                     showlogprobs=True, \
                     pathfig='test_noisemags.png', \
                     closeaftersave=False):

    """Shows the covariances corresponding to the noise model samples.

Inputs:

    flatsamples = Flatsamples object including flattened samples

    nshow = number of samples to show

    fignum = matpotlib figure number to use

    logy = use log10 scale for y axis

    showvar = show variance (instead of stddev)

    cmap = colormap to use

    jaux = (0,1,2) - index of noise model parameter to use for colors

    alpha = opacity for (noise vs mag) plots

    showlogprobs = color-code flat samples plot by log probability

    closeaftersave = Close the figure after saving

Example call:

    FS = examine2d.Flatsamples(flat_samples, esargs=esargs, log_probs=log_probs)
    shownoisesamples(FS)

    """

    if flatsamples is None:
        return

    # Ensure the number of sets to show makes sense
    if flatsamples.nsamples < 1:
        return
    
    nshow = min([flatsamples.nsamples, nshow])    

    # log probabilities from the Flatsamples object
    logprobs = np.array([])
    if hasattr(flatsamples, 'log_probs'):
        logprobs = flatsamples.log_probs
    
    # This will be using noisemodel directly, so will need the parset
    if not hasattr(flatsamples, 'inp_parset'):
        print("examine2d.shownoisesamples WARN - flatsamples has no parset")
        return

    # Which indices are the ones we want in the flat samples?
    lnoise = flatsamples.inp_parset.lnoise
    lsymm = flatsamples.inp_parset.lsymm
    
    # If the flat samples don't actually have a noise model, there's
    # nothing much to do.
    if np.size(lnoise) < 1:
        print("examine2d.shownoisesamples WARN - no indices for noise model")
        return

    # Zeropoint magnitude. This SHOULD come across in the parset, if
    # not, look at the likelihood object
    if not hasattr(flatsamples.inp_parset, 'mag0'):
        mag0 = flatsamples.inp_lnlike.mag0
    else:
        mag0 = flatsamples.inp_parset.mag0

    # Noise model option
    islog10_c = flatsamples.inp_lnlike.parset.islog10_noise_c
        
    # Magnitude ranges
    mags = flatsamples.inp_lnlike.obstarg.mags
    mshow = np.linspace(mags.min(), mags.max(), 100, endpoint=True)
        
    # Try computing the stdx, stdy, corrxy components from the
    # flatsamples.
    parsnoise = flatsamples.flat_samples[:,lnoise]
    parssymm = flatsamples.flat_samples[:,lsymm]

    stdxs, stdys, corrxys = noisemodel2d.mags2noise(parsnoise.T, parssymm.T, \
                                                    mshow[:,None], \
                                                    mag0=mag0, returnarrays=True, \
                                                    islog10_c=islog10_c)

    
    # pick a random sample to show
    ldum = np.argsort(np.random.uniform(size=parsnoise.shape[0]))
    lsho = ldum[0:nshow]
    
    # now, finally, plot the figure
    fig9=plt.figure(fignum)
    fig9.clf()
    ax90 = fig9.add_subplot(222)
    ax92 = fig9.add_subplot(224)

    # axes for the flat samples themselves
    axf0 = fig9.add_subplot(321)
    axf1 = fig9.add_subplot(323)
    axf2 = fig9.add_subplot(325)

    for ax in axf0, axf1:
        ax.tick_params(labelbottom=False)
    
    # showing variance or stddev?
    powr = 1.
    if showvar:
        powr = 2.

    # Color-code our plots by auxiliary quantity
    zmin = np.min(parsnoise[:,jaux])
    zmax = np.max(parsnoise[:,jaux])
    if jaux == 2:
        zmin = 0. 
    aux = (parsnoise[:,jaux] - zmin)/(zmax - zmin)
    Cmap = plt.get_cmap(cmap)

    # make a scalar mappable that will correspond to this
    sm = plt.cm.ScalarMappable(cmap=Cmap, \
                               norm=plt.Normalize(vmin=zmin, vmax=zmax))
    
    for ishow in lsho:
        dum = ax90.plot(mshow, stdxs[:,ishow]**powr, alpha=alpha, \
                        color=Cmap(aux[ishow]) )

        dum2 = ax92.plot(mshow, stdys[:,ishow]**powr/stdxs[:,ishow]**powr, \
                         alpha=alpha, \
                         color=Cmap(aux[ishow]) )
        
    # Does this understand colorbars?
    pset = flatsamples.inp_parset
    noiselabels =pset.labels_noise[0:pset.nnoise]
    
    cbar = fig9.colorbar(sm, ax=ax90, label=noiselabels[jaux] )
    cbar.solids.set(alpha=1)

    cbar2 = fig9.colorbar(sm, ax=ax92, label=noiselabels[jaux] )
    cbar2.solids.set(alpha=1)

    
    if logy:
        ax90.set_yscale('log')
        #ax92.set_yscale('log')

    # Axis label
    squan = r'$s_{\xi}$'
    if showvar:
        squan = r'$%s^2$' % (squan.replace('$',''))
    ax90.set_xlabel('mag')
    ax90.set_ylabel(squan)
    ax92.set_xlabel('mag')
    ax92.set_ylabel(r'covar ratio')

    # Annotation for sample shown
    nsamples = parsnoise.shape[0]
    sanno = f"{nshow:,} of {nsamples:,} shown"
    ax90.annotate(sanno, (0.04,0.97), xycoords='axes fraction', \
                  fontsize=8, \
                  ha='left', va='top')
    
    # Now show the flat samples.
    lsam = np.arange(parsnoise.shape[0], dtype='int')

    # Colors to use
    caux = parsnoise[lsam, jaux]
    vmin = zmin
    vmax = zmax
    cmapaux = cmap
    ok_logprobs = False
    
    if showlogprobs:
        if np.size(logprobs) == parsnoise.shape[0]:
            caux = logprobs[lsam]
            vmin = None
            vmax = None
            cmapaux = 'viridis'
            ok_logprobs = True
        else:
            print("shownoisesamples WARN - logprobs the wrong shape or not provided. Not showing.")
            
    for ax, j in zip([axf0, axf1, axf2], [0,1,2]):
        dumflat = ax.scatter(lsam, parsnoise[lsam, j], \
                             c=caux, \
                             alpha=0.25, \
                             cmap=cmapaux, \
                             vmin=vmin, vmax=vmax, \
                             s=1)

        ax.set_ylabel(noiselabels[j])

        # Only show the colorbar if not the same as we're already
        # plotting on the larger panel
        if ok_logprobs:
            clabel = ''
            if j < 1:
                clabel = r'ln(prob)'
            cbar = fig9.colorbar(dumflat, ax=ax, label=clabel)
            cbar.solids.set(alpha=1)
            
    axf2.set_xlabel('Flat sample number')

    # A few cosmetics
    fig9.subplots_adjust(hspace=0.05, wspace=0.4)

    if len(pathfig) > 3:
        fig9.savefig(pathfig)

        if closeaftersave:
            plt.close(fig9)
    
def showcorner(flat_samples=np.array([]), \
               labels=None, truths=None, \
               fignum=4, pathfig='test_corner_oo.png', \
               minaxesclose=20, \
               nmodel=-1, colornuisance='#9A3324', \
               inds_abc=[], convert_linear=False, \
               tellsummary=False):

    """Corner plot of flattened samples from mcmc run.

Inputs:

    flat_samples = [nsamples, npars] array of flattened samples

    labels = [npars] array of string labels for each parameter

    truths = [npars] array of "truth" values (if known)

    fignum = matplotlib figure number to use 

    pathfig = path to save the corner plot image

    minaxesclose = closes the figure (after saving to disk) if there
    are >= minaxesclose quantities to plot. Useful to free up memory.

    nmodel = number of parameters that constitute the non-nuisance
    parameters. Defaults to no selection

    tellsummary = print (1d) summary statistics for the parameters

Returns:
    
    None.

Example call:

    examine2d.showcorner(flat_samples, **showargs['corner'])

    """

    if np.size(flat_samples) < 1:
        return

    # convert {b,c,e,f} to {sx, sy, theta, beta} and reorder?
    if convert_linear and np.size(inds_abc) > 5:
        flat_samples, labels, truths = \
            sixterm2d.flatpars(flat_samples, inds_abc, labels, truths)

        print("showcorner DEBUG - truths:", truths)
        print("showcorner DEBUG - labels:", labels)
        
    # number of model parameters that are non-nuisance (default to no
    # selection)
    nsamples, ndim = flat_samples.shape
    if nmodel < 1 or nmodel > ndim:
        nmodel = ndim


    if tellsummary:
        pctiles = [50., 15.9, 84.1]
        quants = np.percentile(flat_samples, pctiles, axis=0)
        print("INFO:", flat_samples.shape, quants.shape, ndim)
        print("examine2d.showcorner INFO - 1D parameter ranges:")
        for ipar in range(ndim):
            srange = r"Fit: %.6e - %.2e + %.2e" \
                % (quants[0,ipar], \
                   quants[0,ipar]-quants[1,ipar], \
                   quants[2,ipar]-quants[0,ipar])
            struth = ''
            if np.size(truths) > 0:
                struth = r" - Truth: %.6e" % (truths[ipar])

            # (Using very high precision because some of these
            # quantities are coming out with very small distributions)
                
            print("%s -- %s %s" % (labels[ipar], srange, struth))

        print("##########")
        
    # Label keyword arguments
    label_kwargs = {'fontsize':8, 'rotation':'horizontal'}
    
    # I prefer to set the figure up and then make the corner plot in
    # the figure:
    fig4 = plt.figure(fignum, figsize=(9,7))
    fig4.clf()

    print("examine2d.showcorner INFO - plotting corner plot...")
    corn = corner.corner(flat_samples, labels=labels, truths=truths,\
                         truth_color='b', fig=fig4, labelpad=0.7, \
                         use_math_text=True, \
                         label_kwargs = label_kwargs)
    fig4.subplots_adjust(bottom=0.2, left=0.2, top=0.95)

    # If we have nuisance parameters, highlight them. Follows the API
    # documentation for corner.corner:
    # https://corner.readthedocs.io/en/latest/pages/custom/
    if nmodel < ndim:
        print("examine2d.showcorner INFO - highlighting nuisance parameters")
        axes = np.array(fig4.axes).reshape((ndim, ndim))
        for yi in range(nmodel, ndim):
            for xi in range(ndim):
                ax = axes[yi, xi]

                # Change the spine color
                for spine in ['bottom','top','left','right']:
                    ax.spines[spine].set_color(colornuisance)
                
                # Change the label color
                ax.yaxis.label.set_color(colornuisance)
                if xi >= nmodel:
                    ax.xaxis.label.set_color(colornuisance)
                
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


def showcovarscomp(flatsamples=None, \
                   pathcovs='test_flatsamples.pickle', dcovs={}, \
                   keymcmc='covpars', keylsq='lsq_hessian_inv', \
                   keylabels='slabels', \
                   fignum=6, \
                   sqrt=True, \
                   log=True, \
                   pathfig=''):
    
    """Visualizes the comparison in parameter covariance between the ltsq
and the mcmc evaluations

Inputs:

    flatsamples = Flatsamples object - [TO BE UPDATED]

    pathcovs = path to pickle file holding the covariances. Ignored if
    dcovs is supplied

    dcovs = dictionary holding the samples.

    keymcmc = dictionary key corresponding to the MCMC covariances
    
    keylsq = dictionary key corresponding to the LSQ covariances

    keylabels = dictionary key corresponding to the labels

    fignum = matplotlib figure number to use

    log = use log scale for the MCMC and LSQ heatmaps

    sqrt = use sqrt scale for the MCMC and LSQ heatmaps

    pathfig = file path to save figure (must contain ".")

    nextramcmc = number of "extra" arguments in mcmc parameters. (For
    example, might be noise parameters that the lsq array doesn't
    have).

Returns:

    No return quantities. See the figure.

Example call:

    fittwod.showcovarscomp(pathcovs='./no_src_uncty_linear/test_flatsamples.pickle', pathfig='lsq_mcmc_covars.png', sqrt=True, log=True)

    """

    # 2024-08-26: allow reading in of pickle file as an option. May
    # get rid of this option in future.
    if len(pathcovs) > 3:
        try:
            with open(pathcovs, "rb") as robj:
                dcovs = pickle.load(robj)
        except:
            nopath = True

        # Check to see if all the right entries are present
        lkeys = dcovs.keys()
        for key in [keymcmc, keylsq, keylabels]:
            if not key in lkeys:
                print("showcovarscomp WARN - key not in dictionary: %s" \
                      % (key))
                return

        # convenience views
        covslsq = dcovs[keylsq]
        covsmcmc = np.copy(dcovs[keymcmc])
        slabels = dcovs[keylabels]

    # 2024-08-26 updating to read these quantities from a Flatsamples
    # instance instead of a pickle file
    if flatsamples is None:
        return

    covsmcmc = flatsamples.param_covars
    covslsq = flatsamples.lstsq_covars
    slabels = flatsamples.labels_transf
    
    # The MCMC may also be exploring noise parameters or mixture model
    # fractions, which the LSQ approach can't do. In that instance,
    # take just the model parameters
    nmcmc = np.shape(covsmcmc)[0]
    nlsq = np.shape(covslsq)[0]
    if nmcmc > nlsq and nlsq > 0:
        covsmcmc = covsmcmc[0:nlsq, 0:nlsq]

        # not sure this is still needed - comment out for the moment
        # slabels = dcovs[keylabels][0:nlsq]

    # Showing a heatmap of one of the quantities
    fig6 = plt.figure(6, figsize=(8,6))
    fig6.clf()
    if nlsq > 0:
        ax61 = fig6.add_subplot(221)
        ax62 = fig6.add_subplot(222)
        ax63 = fig6.add_subplot(224)
    else:
        ax61 = fig6.add_subplot(111)
        ax62 = None
        ax63 = None
        
    # if log, we can meaningfully show the text. Otherwise
    # don't. (Kept out as a separate quantity in case we want to add
    # more conditions here.)
    showtxt = log

    # fontsize for annotations
    fontsz=5
    if covsmcmc.shape[0] < 10:
        fontsz=6
    
    showheatmap(covsmcmc, slabels, ax=ax61, fig=fig6, \
                log=log, sqrt=sqrt, \
                cmap='viridis_r', title='MCMC', \
                showtext=showtxt, fontsz=fontsz)

    if nlsq > 0:
        showheatmap(covslsq, slabels, ax=ax62, fig=fig6, \
                    log=log, sqrt=sqrt, \
                    cmap='viridis_r', title='LSQ', \
                    showtext=showtxt, fontsz=fontsz)

        # find the fractional difference. The mcmc has already been
        # cut down to match the lsq length above, so if the arrays
        # still mismatch their lengths then something is wrong with
        # the input.
        fdiff = (covslsq - covsmcmc)/covsmcmc
        titlediff = r'(LSQ - MCMC)/MCMC'
        showheatmap(fdiff, slabels[0:nlsq], ax=ax63, fig=fig6, log=False, \
                    cmap='RdBu_r', title=titlediff, showtext=True, \
                    symmetriclimits=True, symmquantile=0.99, \
                    fontcolor='#D86018', fontsz=fontsz)

    # Warn on the plots if more mcmc parameters were supplied than
    # used. In all the use cases these should be noise parameters that
    # the LSQ covariances don't have, so it's not an "error".
    if nmcmc > nlsq and nlsq > 0:
        ax61.annotate('MCMC params ignored: %i' % (nmcmc-nlsq), \
                      (0.97,0.97), xycoords='axes fraction', \
                      ha='right', va='top', fontsize=8, \
                      color='#9A3324')
    
    # save figure to disk?
    if len(pathfig) > 0:
        if pathfig.find('.') > 0:
            fig6.savefig(pathfig)
    
def showheatmap(arr=np.array([]), labels=[], \
                ax=None, fig=None, fignum=6, \
                cmap='viridis', \
                showtext=False, fontsz=6, fontcolor='w', \
                addcolorbar=True, \
                sqrt=False, \
                log=False, \
                title='', \
                maskupperright=True, \
                symmetriclimits=False, \
                symmquantile=1.):

    """Plots 2D array as a heatmap on supplied axis. Intended use:
visualizing the covariance matrix output by an MCMC or other parameter
estimation.

Inputs:

    arr = [M,M] array of quantities to plot

    labels = [M] length array or list of quantity labels. If there are
    more labels than datapoints, only 0:M are included. This may not
    be what you want.

    ax = axis on which to draw the plot

    fig = figure object in which to put the axis

    fignum = figure number, if creating a new figure

    cmap = color map for the heatmap

    showtext = annotate each tile with the array value?

    fontsz = font size for tile annotations

    addcolorbar = add colorbar to the axis?

    sqrt = take sqrt(abs(arr)) before plotting

    log = colormap on a log10 scale (Note: if sqrt and log are both
    true, then the quantity plotted is log10(sqrt(abs(arr))).  )

    title = '' -- string for axis title

    maskupperright -- don't plot the duplicate upper-right (off
    diagonal) corner values

    symmetriclimits -- if not logarithmic plot, makes the color limits
    symmetric about zero (useful with diverging colormaps)

    symmquantile -- if using symmetric limits, quantile of the limits
    to use as the max(abs value). Defaults to 1.0

Outputs:

    None

    """

    # Must be given a 2D array
    if np.ndim(arr) != 2:
        return
    
    # Ensure we know where we're plotting
    if fig is None:
        fig = plt.figure(fignum)
        
    if ax is None:
        ax = fig.add_subplot(111)

    # What are we showing?
    labelz = r'$V_{xy}$'
    arrsho = np.copy(arr)

    if sqrt:
        labelz = r'$\sqrt{|V_{xy}|}$'
        arrsho = np.sqrt(np.abs(arr))

    # log10 - notice this happens on ARRSHO (i.e. after we might have
    # taken the square root).
    if log:
        labelz = r'$log_{10} \left(%s\right)$' % \
            (labelz.replace('$',''))
        arrsho = np.log10(np.abs(arrsho))

    # make arrsho a masked array to make things a bit more consistent
    # below
    arrsho = ma.masked_array(arrsho)
        
    # Mask upper-left (duplicate points)?
    if maskupperright:
        iur = np.triu_indices(np.shape(arrsho)[0], 1)
        arrsho[iur] = ma.masked

    # compute symmetric colorbar limits?
    vmin = None
    vmax = None
    if symmetriclimits and not log and not sqrt:
        maxlim = np.quantile(np.abs(arrsho), symmquantile)
        vmin = 0. - maxlim
        vmax = 0. + maxlim
        
    # imshow the dataset
    im = ax.imshow(arrsho, cmap=cmap, vmin=vmin, vmax=vmax)

    # Ensure labels are set, assuming symmetry
    ncov = np.shape(arrsho)[0]
    nlab = np.size(labels)

    if nlab < ncov:
        labls = ['p%i' % (i) for i in range(ncov)]
    else:
        labls = labels[0:ncov]        

    # Now set up the ticks
    ax.set_xticks(np.arange(ncov))
    ax.set_yticks(np.arange(ncov))
    ax.set_xticklabels(labls)
    ax.set_yticklabels(labls)

    # Text annotations (this might be messy)
    if showtext:
        for i in range(len(labls)):
            for j in range(len(labls)):
                if arrsho[i,j] is ma.masked:
                    continue
                
                text = ax.text(j, i, \
                               "%.2f" % (arrsho[i,j]), \
                               ha="center", va="center", \
                               color=fontcolor, \
                               fontsize=fontsz)

    if addcolorbar:
        cbar = fig.colorbar(im, ax=ax, label=labelz)

    # Set title
    if len(title) > 0:
        ax.set_title(title)
        
    # Some cosmetic settings
    fig.tight_layout()
