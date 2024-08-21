#
# examine2d.py
#

#
# WIC 2024-08-19 - methods for examining the output from MCMC runs
#

import os, time
import numpy as np
import copy

# For computing some needed pieces
from binstats2d import Binstats
import noisemodel2d


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
                 esargs={}, ptruths=None, log_probs=np.array([]), \
                 path_log_probs='NA'):

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


            
            
def showguess(esargs={}, fignum=2, npermagbin=36, respfg=0.8, nmagbins=10, \
              pathfig='test_guess_deltas.png'):

    """Plots up the guess transformation, noise, etc., before running mcmc.

Inputs:

    esargs = {} = dictionary of ensemble sampler arguments.

    fignum = matplotlib figure number to use

    npermagbin = number of points per magnitude bin (when estimating
    delta vs mag for foreground objects)

    respfg = threshold for identifying objects as "foreground" when
    estimating the running statistics

    nmagbins = number of magnitude bins (for running statistics). If >0, overrides npermagbin.

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

    # Compute binned statistics on "foreground" objects
    bfg = llike.resps_fg > respfg
    BG = Binstats(mags[bfg], dxytran[bfg], npermagbin, nbins=nmagbins)
    magbins, dxymeans, dxycovs, counts = BG.getstats()

    # Try objects identified as background
    bbg = llike.resps_fg < 0.2
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
        ax.set_xlabel(r'$\Delta \xi$') # now that there's room
    ax37.set_ylabel(r'$\Delta \eta$')

    ax38.set_xlabel(r'$\xi$')
    ax39.set_xlabel(r'$\eta$')
    ax34.set_ylabel(r'$\xi$')
    ax31.set_ylabel(r'$\eta$')

    # How do our responsibilities look?
    ax35 = fig2.add_subplot(335)
    dum35 = ax35.scatter(dxytran[:,0], dxytran[:,1], c=llike.resps_fg, \
                         cmap='inferno', s=4, edgecolor=None, vmax=1., \
                         alpha=0.7)
    cbar35 = fig2.colorbar(dum35, ax=ax35, label=r'$f_fg$')
    ax35.set_xlabel(r'$\Delta \xi$')
    
    # Colorbars
    for obj, ax in zip([resid, residyx, residyy, residxx, residxy], \
                       [ax37, ax38, ax39, ax34, ax31]):
        cbar = fig2.colorbar(obj, ax=ax)

    # label where we can fit it in
    ax31.set_title('Colors: mag')
        
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
    
    for ax in [ax32, ax33]:
        ax.set_xlabel('mag')
        ax.set_yscale('log')
    ax32.set_ylabel(r'$V_{xx}$')
    ax33.set_ylabel(r'$V_{\xi\xi}$')

    leg = ax33.legend(fontsize=5)

    ax32.set_title('Source frame', fontsize=fontsz)
    ax33.set_title('Target frame', fontsize=fontsz)
    
    # a few cosmetic things
    fig2.subplots_adjust(hspace=0.4, wspace=0.4)

    # save to disk
    if len(pathfig) > 3:
        fig2.savefig(pathfig)
    
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

    # Magnitude ranges
    mags = flatsamples.inp_lnlike.obstarg.mags
    mshow = np.linspace(mags.min(), mags.max(), 100, endpoint=True)
        
    # Try computing the stdx, stdy, corrxy components from the
    # flatsamples.
    parsnoise = flatsamples.flat_samples[:,lnoise]
    parssymm = flatsamples.flat_samples[:,lsymm]

    stdxs, stdys, corrxys = noisemodel2d.mags2noise(parsnoise.T, parssymm.T, \
                                                    mshow[:,None], \
                                                    mag0=mag0, returnarrays=True)

    # pick a random sample to show
    ldum = np.argsort(np.random.uniform(size=parsnoise.shape[0]))
    lsho = ldum[0:nshow]
    
    # now, finally, plot the figure
    fig9=plt.figure(fignum)
    fig9.clf()
    ax90 = fig9.add_subplot(122)

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
        
    # Does this understand colorbars?
    labels = [r'$log_{10}(a)$', r'$log_{10}(b)$', r'$c$']
    cbar = fig9.colorbar(sm, ax=ax90, label=labels[jaux] )
    cbar.solids.set(alpha=1)
    
    if logy:
        ax90.set_yscale('log')

    # Axis label
    squan = r'$s_{\xi}$'
    if showvar:
        squan = r'$%s^2$' % (squan.replace('$',''))
    ax90.set_xlabel('mag')
    ax90.set_ylabel(squan)

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

        ax.set_ylabel(labels[j])

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
    fig4 = plt.figure(fignum, figsize=(9,7))
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


