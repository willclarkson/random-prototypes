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
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter
plt.ion()
import corner

# For logistic regression on responsibilities
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

# For identifying rogue burn-in chains
from sklearn.cluster import DBSCAN

# For serializing the parameters and their covariances
import pickle

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

    cluster_eps = cluster bandwidth for partitioning

    """

    def __init__(self, flat_samples=np.array([]), path_samples='NA', \
                 esargs={}, ptruths=None, log_probs=np.array([]), \
                 path_log_probs='NA', showargs={}, path_showargs='',\
                 path_esargs='', cluster_eps=0.3):

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

        # show arguments. Input in preference to path
        self.path_showargs = path_showargs[:]
        if len(showargs.keys()) < 1:
            self.loadshowargs()
        else:
            self.showargs = showargs
            
        # emcee arguments
        self.path_esargs = path_esargs[:]
        if len(esargs.keys()) < 1:
            self.loadesargs()
        else:
            self.esargs = esargs
            
        # Populate the shape attributes that we will need to access
        self.nsamples = 0
        self.npars = 0
        self.countsamples()
        
        # Dictionary of arguments that were passed to emcee.
        # self.esargs = esargs ### replaced by syntax above

        # Dictionary of arguments sent to "show" routines
        # self.showargs = showargs ## replaced by load syntax above.
        
        # Quantities we get from this dictionary
        self.inp_pguess = None
        self.inp_obstarg = None
        self.inp_parset = None
        self.inp_lnprior = None
        self.inp_lnlike = None

        # truth parameters, if known
        self.ptruths = ptruths
        self.lnlike_truth = None  # Like() object
        
        # Now some things we can compute on the samples
        self.ndata = 0.
        self.resps_samples = np.array([])   # (nsamples, ndata) - could be large
        self.resps_avg = np.array([])
        self.isfg = np.array([])

        # ln(likelihoods), useful for post-hoc summary statistics
        self.lnlikevec = np.array([])
        self.sumlnlike = np.array([])
        
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

        # Median of flat samples parameters
        self.param_medians = np.array([])
        
        # covariance among flat samples parameters
        self.param_covars = np.array([])
        self.lstsq_covars = np.array([])
        self.labels_transf = None

        # Clusters by likelihood values (sometimes rogue burn-in
        # chains can slip in). Default is that all objects belong to
        # the main cluster. We retain the labels (rather than creating
        # a new instance per cluster) because we might be interested
        # in the comparison, and retaining the labels allows this at
        # the cost of more instance attributes.
        self.cluster_eps = cluster_eps # clustering bandwidth
        self.clusterid = np.array([])
        self.clustermeds = np.array([])
        self.ismain = np.array([])
        self.initclusters()                                        
        
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

        # Classify by lnprob cluster
        self.clusterbylogprob()
        
        # Compute the parameter medians and covariances among the flat
        # samples
        self.computemedians()
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

    def loadshowargs(self):

        """Loads show arguments from disk if found"""

        if len(self.path_showargs) < 4:
            return

        try:
            self.showargs = pickle.load(open(self.path_showargs,'rb'))
        except:
            noshowargs = True

    def loadesargs(self):

        """Loads emcee arguments from disk"""

        if len(self.path_esargs) < 4:
            return

        try:
            self.esargs = pickle.load(open(self.path_esargs, 'rb'))
        except:
            noesargs = True
        
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

            # If we passed in an lnlike object for the truths object,
            # unpack it here.
            if 'lnlike' in self.showargs['truthset'].keys():
                self.lnlike_truth = self.showargs['truthset']['lnlike']
                self.lnlike_truth.updatesky(self.lnlike_truth.parset)
                
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

        # UPDATE - if a transformation object was already passed in,
        # use it! (This helps account for the case where npars differ)
        if hasattr(self.lnlike_truth,'transf'):
            self.transftruth = copy.deepcopy(self.lnlike_truth.transf)
            self.transftruth.propagate()
            self.setobjontruth()
            return
            
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

    def initclusters(self):

        """Initializes the cluster IDs"""

        if self.nsamples < 1:
            self.clusterid = np.array([])
            self.clustermeds = np.array([])
            self.ismain = np.array([])
            return

        self.clusterid = np.zeros(self.nsamples)
        self.ismain = np.repeat(True, self.nsamples)
        self.clustermeds = np.array([0.])
        
    def clusterbylogprob(self):

        """Identifies clusters by lnprob"""

        if self.nsamples < 2:
            return

        if np.size(self.log_probs) < 2:
            return

        # Find the clusters
        self.clusterid, self.clustermeds, ulabs = \
            splitclusters(self.log_probs, self.cluster_eps)

        # which has the maximum median lnprob
        imax = np.argmax(self.clustermeds)
        self.ismain = self.clusterid == ulabs[imax]
        
    def computeresps(self, samplesize=-1, keepmaster=True, \
                     ireport=1000, Verbose=True, \
                     pathresps='test_resps_samples.npy', \
                     pathlikes='test_likes_samples.npy'):

        """Computes ln(like) and foreground probabilities for every sample.

Inputs:

        samplesize = number of samples to compute

        keepmaster = store all the samples in memory? [nsamples, ndata]
        
        ireport = report every this many rows

        Verbose = report to screen

        pathresps = paths to write master samples responsibilities file

        pathlikes = paths to write [nsamples, ndata] log-likelihood file

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

        # sum statistics
        norm = 0.
        self.resps_avg = np.zeros(self.ndata)
        self.sumlnlike = np.zeros(self.nsamples)
        
        if keepmaster:
            self.resps_samples = np.zeros(( imax, self.ndata ))
            self.lnlikevec = np.copy(self.resps_samples)
            
        # Safety valve - if the model doesn't have a mixture, there's
        # no need to recalculate everything:
        nmix = np.size(self.inp_lnlike.parset.mix)
        if nmix < 1:
            self.resps_avg += 1.
            if keepmaster:
                self.resps_samples += 1.
            print("examine2d.computeresps INFO - no mixture, all responsibilities 1.0")

            # We no longer want to return since we also need the
            # lnlikes, so we probably should NOT return here.
            return
            
        t0 = time.time()
        if Verbose:
            print("examine2d.computeresps INFO - starting responsibility loop...")
            
        for isample in range(imax):

            # update the parameter-set and recompute everything
            lnlike.parset.updatepars(self.flat_samples[isample])
            lnlike.updatelnlike(lnlike.parset)
            lnlike.calcresps()

            # The log-like object already computes the sum of
            # lnlikes. Slot it in here.
            self.sumlnlike[isample] = lnlike.sumlnlike
            
            # increment the responsibilities and the norm
            norm += 1.
            self.resps_avg += lnlike.resps_fg
            
            # If we want to store all the responsibilities per sample
            if keepmaster:
                self.resps_samples[isample] = lnlike.resps_fg

                # populate the lnlike per-datapoint
                self.lnlikevec[isample] = \
                    np.logaddexp(lnlike.lnlike_fg, lnlike.lnlike_bg)
                
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
        self.writelnlikevecs(pathlikes)
        
    def writeresps(self, pathresps='test_resps_samples.npy'):

        """Utility - write responsibilities array to disk (can be large).

Inputs:
        
        pathresps  =  path to write to

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

    def writelnlikevecs(self, pathlnlikes='test_lnlikes_samples.npy'):

        """Writes [nsamples, ndata] ln-likelihood vectors to .npy format

        Inputs:
        
        pathlnlikes =  path to write to

        Returns: Nothing

        """

        if len(pathlnlikes) < 1:
            return

        if np.size(self.lnlikevec) < 1:
            return

        print("examine2d.writelnlikevecs INFO - writing log-like vectors:", \
              self.lnlikevec.shape)

        np.save(pathlnlikes, self.lnlikevec)


        
    def loadlnlikevecs(self, pathlnlikes='test_lnlikes_samples.npy'):

        """Reads log-likelihood evaluations from disk, and computes the sum
along the dataset for each sample.

        Inputs

        pathlnlikes = path to .npy file holding [nsamples, ndata] log likelihood evaluations

        Outputs

        None. Internal attributes self.lnlikevec and self.sumlnlike are updated.

        """

        if len(pathlnlikes) < 4:
            return

        self.lnlikevec = np.load(pathlnlikes)

        self.sumlnlike = np.sum(self.lnlikevec, axis=0)
        
        
        
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

    def computemedians(self):

        """Computes median parameter samples"""

        if np.size(self.flat_samples) < 1:
            return

        self.param_medians = np.median(self.flat_samples, axis=0)
        
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

            try:
                self.binstats_fg = Binstats(mags[bfg], dxytran[bfg], \
                                            minperbin, nbins=nbins)
            except:
                print("Flatsamples.computebinnedstats WARN - problem binning fg by magnitude (likely too few fg objects)")
                
        if np.sum(bbg) > 1:
            self.binstats_bg = Binstats(mags[bbg], dxytran[bbg], \
                                        minperbin)
            

    def saveparset(self, outpath='test_parset_found.pickle'):

        """Writes the parameters summary to disk as a Parset1d object"""

        # Nothing to do if no file given, or if we do not have
        # parameters (neat trick)
        if len(outpath) < 4:
            return
        
        if np.size(self.param_medians) < 1:
            return
        
        # We populate the parset as a copy of the input paramset,
        # updating the various pieces we need with the median and the
        # covariances, both calculated here. This helps ensure that
        # all the baggage (like labels) is carried in.
        if self.inp_parset is None:
            return

        # Copy the input parset and replace its values with the median
        # values determined in this object
        parset = copy.deepcopy(self.inp_parset)
        parset.updatepars(self.param_medians)

        # Populate the covariance attribute
        parset.covar = np.copy(self.param_covars)

        # now serialize this to disk
        with open(outpath, 'wb') as wobj:
            pickle.dump(parset, wobj)

def splitclusters(logprob=np.array([]), eps=1.):

    """
    Clusters by ln(prob).

    Inputs:

    logprob = [N] - array on which to cluster

    eps = minimum separation parameter for sklearn's DBSCAN

    Returns:

    labels - [N] array of cluster labels

    medns - median logprob for each cluster

    ulabs - unique labels in the same order as medns

    """

    # ... or we could just return the cluster itself?

    if np.size(logprob) < 1:
        return np.array([]), np.array([])

    dbs = DBSCAN(eps=eps)

    # Get input into the right shape
    if np.ndim(logprob) < 2:
        arr2d = logprob.reshape(-1,1)
    else:
        arr2d = logprob

    # fit the dbscan to the data
    dbs.fit(arr2d)

    # now set the labels and find the medians
    medians = []
    sizes = []
    ulabs = np.unique(dbs.labels_)
    for lab in ulabs:
        bthis = dbs.labels_ == lab
        medians.append(np.median(logprob[bthis]))
        sizes.append(np.sum(bthis))

    return dbs.labels_, np.array(medians), ulabs

def ismaincluster(logprobs=np.array([]), eps=2.):

    """Returns boolean for objects that lie within the main cluster of
lnprobs. 

    Inputs:

    logprobs = [N] - element array that will be split into clusters

    eps = minimum-distance parameter for the dbscan cluster

    Returns:

    bmain = [N] - element boolean array: whether the object is part of
    the cluster with the largest median logprob (NOT necessarily the
    largest cluster)?

    """

    # Written as a method here because I'm still not sure if this is
    # best encapsulated into FlatSamples instances...

    if np.size(logprobs) < 1:
        return np.array([])

    labels, medns, ulabs = splitclusters(logprobs, eps)
    imax = np.argmax(medns)
    
    return labels == ulabs[imax]


def showguess(esargs={}, fignum=2, npermagbin=36, respfg=0.8, nmagbins=10, \
              pathfig='test_guess_deltas.png', \
              pathfignoise='test_guess_noise.png', \
              showargs={}, \
              usetruths=False, showquiver=True, \
              sqrtx=False):

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

    sqrtx = in the source/meas uncertainty, show sqrt(Vxx) since this
    is easier to visualize

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
    # graph... (WATCHOUT - this duplicates some methods farther down.)
    ptruth = None
    ltruth = None
    if 'truthset' in showargs.keys():
        if 'parset' in showargs['truthset'].keys():
            ptruth = showargs['truthset']['parset']

        if 'lnlike' in showargs['truthset'].keys():
            ltruth = copy.deepcopy(showargs['truthset']['lnlike'])
            if ltruth is not None:
                ltruth.updatesky(ltruth.parset)
        else:
            ltruth = copy.deepcopy(llike)        
            ltruth.updatesky(ptruth) # fails if number of params differ

    # If asked to use the truth parameters, copy them in to
    # the llike object and update
    if usetruths and ltruth is not None:
        print("showguess INFO - using truth parameters for plots")
        llike = ltruth
        transf = ltruth.transf

                
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

    # in case we need them, sqrt(vxx)
    labelsxxsrc = r'$\sqrt{V_{%s%s}}$' % (labelxsrc.replace('$',''), \
                                          labelxsrc.replace('$',''))

    labelsxxtran = r'$\sqrt{V_{%s%s}}$' % (labelxtran.replace('$',''), \
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
    if np.sum(bfg) < 10:
        print("examine2d.showguess WARN - few objects identified as foreground. Defaulting to all for plots (probably not what you want)")
        bfg = np.isfinite(mags)
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

    # 2025-06-13 split the big figure into two figures so that the
    # axes will be easier to read.
    
    # Now set up the figures
    fig2=plt.figure(fignum, figsize=(9., 7.))
    fig2.clf()

    fig2b=plt.figure(fignum+1, figsize=(9., 5.))
    fig2b.clf()

    # 2025-06-13 INFO - the axis variable names (ax37 etc) are
    # artefacts of how I originally threw together the plot. The
    # numbers have no relation to the current position of the graphs,
    # and some day sohuld be updated with more accurate descriptive
    # variable names.
    
    # Positional residual plots
    # THIS WAS WITH 3x3 bigfigure:
    #ax37=fig2.add_subplot(337)
    #ax38=fig2.add_subplot(338, sharey=ax37)
    #ax39=fig2.add_subplot(339, sharey=ax37)
    #ax34=fig2.add_subplot(334, sharex=ax37)
    #ax31=fig2.add_subplot(331, sharex=ax37)

    # THIS IS WITH 3x2:
    ax37=fig2b.add_subplot(232) # depsilon, dxi
    ax38=fig2b.add_subplot(235, sharey=ax37) # deta vs xi
    ax39=fig2b.add_subplot(236, sharey=ax37) # deta vs eta
    ax34=fig2b.add_subplot(234, sharex=ax37) # xi vs dxi
    ax31=fig2b.add_subplot(231, sharex=ax37) # eta vs dxi

    # noise vs mag plots
    ax32 = fig2.add_subplot(221) # was 332
    ax33 = fig2.add_subplot(222)
    ax30 = fig2.add_subplot(223)
    
    # truth parameters if we have them
    ax36=None
    magbins_fg_truth = np.array([])

    # check for truths now done above.

    # ok by this point we should have our ltruth if the ingredients
    # were present.
    if ltruth is not None:
        ax36 = fig2.add_subplot(224)

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
    # ax35 = fig2.add_subplot(335) # old BIGFIG
    ax35 = fig2b.add_subplot(233)
    dum35 = ax35.scatter(dxytran[:,0], dxytran[:,1], c=llike.resps_fg, \
                         cmap='inferno', s=4, edgecolor=None, vmax=1., \
                         alpha=0.7)
    cbar35 = fig2.colorbar(dum35, ax=ax35, label=r'$f_{fg}$')
    ax35.set_xlabel(labeldxtran)
    ax35.set_title(r'Colors: $f_{fg}$')
    
    # Colorbars
    for obj, ax in zip([resid, residyx, residyy, residxx, residxy], \
                       [ax37, ax38, ax39, ax34, ax31]):
        cbar = fig2.colorbar(obj, ax=ax)

    # label where we can fit it in
    ax31.set_title('Colors: mag')
        
    # ... now do the vs magnitude plots. Little bit of a fudge to
    # duplicate the target plot in two axes
    laxmag = [ax32, ax33, ax30]
    if sqrtx:
        lquan = [covobs[:,0,0]**0.5, covtarg[:,0,0], covtarg[:,0,0]**0.5 ]
    else:
        lquan = [covobs[:,0,0], covtarg[:,0,0], covtarg[:,0,0] ]
        
    llabl = ['assumed (src)', 'assumed (target)', 'target']
    lcolo = ['#702082','#00274C', '#00274C']
    for ax, quan, label, color in zip(laxmag, lquan, llabl, lcolo):
        dumobs = ax.scatter(mags, quan, c=color, s=6, \
                            label=label, marker='s', zorder=1)
        
        
    # On the "target frame" mag plot, show the source frame
    # uncertainty, propagated out to the target frame.
    ctransf = ax33.scatter(mags, transf.covtran[:,0,0], c='#702082', \
                           label='source transformed', s=6, alpha=0.6, \
                           zorder=1)
        
    # On the "target frame" mag plot, show the quad sum of the target
    # assumed covariance and the covariance projected from the source
    # frame. This is what a hypothetical observer might adopt as the
    # "measured" covariance.
    cassume = ax33.scatter(mags, covassume[:,0,0], c='#00B2A9', \
                           label='assumed (total)', s=2, zorder=4)

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
        
        ax36.set_title('Target frame, Truth model')
        ax36.set_yscale('log')
        
    # now label the vs-mag plots
        
    for ax in [ax32, ax33, ax30]:
        ax.set_xlabel('mag')

    # We always want to log-scale the target frame, we only want to
    # log-scale the source frame if it has nonzero values
    bpos = covobs[:,0,0] > 0.
    if np.sum(bpos) > 2:
        ax32.set_yscale('log')

    ax32.set_ylabel(labelvxxsrc)
        
    btarg = covtarg[:,0,0] > 0.
    for ax in [ax33, ax30]:

        if np.sum(btarg) > 0:
            ax.set_yscale('log')
    
        ax.set_ylabel(labelvxxtran)

    # replace the vertical label for sqrt(vxx) if needed
    if sqrtx:
        ax32.set_ylabel(labelsxxsrc)
        ax30.set_ylabel(labelsxxtran)
        
    leg = ax33.legend(fontsize=8)

    ax32.set_title('Source frame measurement uncertainty', fontsize=fontsz)
    ax33.set_title('Target frame, guess pars', fontsize=fontsz)
    ax30.set_title('Target frame measurement uncertainty', fontsize=fontsz)

    
    # a few cosmetic things
    fig2.subplots_adjust(hspace=0.4, wspace=0.4)
    fig2b.subplots_adjust(hspace=0.4, wspace=0.4)

    # save to disk
    if len(pathfig) > 3:
        fig2b.savefig(pathfig)

    if len(pathfignoise) > 3:
        fig2.savefig(pathfignoise)
        
    if not showquiver:
        return

    fig3 = plt.figure(fignum+2, figsize=(8., 6.))
    fig3.clf()
    axquiv = fig3.add_subplot(223)

    # Ranges for quiver plot
    # mmin = np.min(mags)
    # mmax = np.max(mags)
    
    quiv_fg = axquiv.quiver(xytarg[bfg, 0], xytarg[bfg, 1], \
                          dxytran[bfg,0], dxytran[bfg,1], \
                          mags[bfg], cmap='viridis_r')
    cbar1 = fig3.colorbar(quiv_fg, ax=axquiv)

    # Show a marginal plot for the foreground objects
    ax33 = fig3.add_subplot(221)
    #blah33 = ax33.scatter(xytarg[bfg, 0], dxytran[bfg,0], \
    #                      c=mags[bfg], s=9, cmap='viridis_r')

    blah33 = ax33.scatter(xytarg[bfg, 0], xytarg[bfg,1], \
                          c=mags[bfg], s=9, cmap='viridis_r')

    # 2025-06-06 I think it's more informative to show the fg original
    # (so we can assess the source frame uncertainties); we already
    # have the displacement figure.
    ax34 = fig3.add_subplot(222)
    #blah34 = ax34.scatter(transf.xytran[bfg, 0], transf.xytran[bfg,1], \
    #                      c=mags[bfg], s=9, cmap='viridis_r')
    blah34 = ax34.scatter(xyobs[bfg, 0], xyobs[bfg,1], \
                          c=mags[bfg], s=9, cmap='viridis_r')

    ##blah34 = ax34.scatter(xytarg[bfg, 1], dxytran[bfg,1], \
    ##                      c=mags[bfg], s=9, cmap='viridis_r')

    
    cbar3 = fig3.colorbar(blah33, ax=ax33)
    cbar4 = fig3.colorbar(blah34, ax=ax34)
    
    if np.sum(bbg) > 0:
        ax32 = fig3.add_subplot(224, sharex=axquiv, sharey=axquiv)

        quiv_bg = ax32.quiver(xytarg[bbg, 0], xytarg[bbg, 1], \
                              dxytran[bbg,0], dxytran[bbg,1], \
                              mags[bbg], cmap='viridis_r')
        cbar2 = fig3.colorbar(quiv_bg, ax=ax32)

        ax32.set_xlabel(labelxtran)
        ax32.set_ylabel(labelytran)

    # This repurposing of axis handler is problematic. For the moment,
    # split into the pieces.
    axquiv.set_xlabel(labelxtran)
    axquiv.set_ylabel(labelytran)

    ax33.set_xlabel(labelxtran)
    #ax33.set_ylabel(labeldxtran)
    ax33.set_ylabel(labelytran)
    
    #ax34.set_xlabel(labelytran)  # 2025-06-06
    ##ax34.set_ylabel(labeldytran)
    #ax34.set_ylabel(labelytran)
    ax34.set_xlabel(labelxsrc)
    ax34.set_ylabel(labelysrc)

    # plot labels
    axquiv.set_title('foreground')
    ax33.set_title('fg, target')    
    #ax34.set_title('fg, transformed') # 2025-06-06
    ax34.set_title('fg, source')
    
    #for ax in [axquiv, ax33, ax34]:
    #    ax.set_title('foreground')
    if np.sum(bbg) > 0:
        ax32.set_title('outliers')
        
    fig3.subplots_adjust(left=0.18, bottom=0.17, hspace=0.4, wspace=0.49)
    
def showresps(flatsamples=None, fignum=8, logx=False, creg=1.0e5, wantbg=True, \
              clobber=True, \
              labeldxtran=r'$\Delta \xi$', labeldytran=r'$\Delta \eta$', \
              cmap='RdBu_r'):

    """Plots the simulated and mcmc responsibilities after an mcmc run.

Inputs:

    flatsamples = Flatsamples object including responsibilities and
    other data characteristics

    fignum = matplotlib figure number

    logx = plot horizontal on log scale

    creg = regularization parameter for logistic regression

    wantbg = we want to plot the probabilities of being part of the background

    clobber = redo the logistic regression even if previously done

    labeldxtran, labeldytran = labels for delta plots

    cmap = colormap for delta plot

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
                            c=resp_sim, alpha=0.7, s=2, vmax=1.0, \
                        cmap=cmap)
    
    dpost = ax92.scatter(flatsamples.dxyproj_truthpars[:,0], \
                         flatsamples.dxyproj_truthpars[:,1], \
                         c=resp_post, alpha=0.7, s=2, vmax=1.0, \
                         cmap=cmap)

    
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

def showparsamples(flatsamples=None, fignum=8, cmap='inferno', \
                   alpha=0.5, pathfig='test_parsamples.png', \
                   showpointing=True, onlyfinite=True, \
                   histlog=False, shownuisance=False, \
                   mainclust=False, minsample=-1):

    """Shows the parameter flat-samples, color-coded by lnprob

Inputs:

    flatsamples = Flatsamples object including samples and lnprobs

    fignum = matplotlib figure number to use

    cmap = color map to use for plots

    alpha = transparency for scatterplot

    pathfig = path for saved image

    showpointing = show par[0,1] covariance

    onlyfinite = show only the samples for which lnprob is finite 

    histlog = use log-y for histogram

    shownuisance = plot nuisance parameters instead of model
    parameters

    mainclust = show only the "main" cluster (using the FS.ismain
    attribute if present)

    minsample = minimum sample number to show (negative for all)

    """

    # Ensure inputs are present
    if flatsamples is None:
        return

    if flatsamples.nsamples < 1:
        return

    # We need to know which indices correspond to which pieces
    if not hasattr(flatsamples, 'inp_parset'):
        print("examine2d.showparsamples WARN - flatsamples.inp_parset absent")
        return

    # Go ahead and get all the pieces so that we can plot whatever we want
    pset = flatsamples.inp_parset
    lpars = np.arange(np.size(pset.model))
    lnoise = pset.lnoise
    lsymm = pset.lsymm
    lmix = pset.lmix

    # Hack to ensure consistent dimension later
    if shownuisance:
        lpars = np.hstack((lnoise, lsymm, lmix))
    
    # Labels
    labels_pars = flatsamples.labels_transf

    # Quantities to actually plot
    nsam = np.shape(flatsamples.flat_samples)[0]
    lsam = np.arange(nsam, dtype='int')

    if not shownuisance:
        pars = flatsamples.flat_samples[:,lpars]
    else:
        # build the parameters and labels depending on what we
        # have. This is a little inelegant because we distingiush the
        # different noise parameter sets by attribute. So:
        labels_all = np.asarray(pset.getlabels())
        pars = np.array([])
        labels_pars = []
        
        if np.size(lnoise) > 0:
            pars = np.copy(flatsamples.flat_samples[:,lnoise])
            labels_pars = list(labels_all[lnoise])[:]
            
        if np.size(lsymm) > 0:
            pars_symm = np.copy(flatsamples.flat_samples[:,lsymm])
            labels_symm = list(labels_all[lsymm])

            if np.size(pars) < 1:
                pars = np.copy(pars_symm)
                labels_pars = labels_symm[:]
            else:
                pars = np.hstack((pars, pars_symm))
                labels_pars = labels_pars + labels_symm[:]
            
        if np.size(lmix) > 0:
            pars_mix = np.copy(flatsamples.flat_samples[:,lmix])
            labels_mix = list(labels_all[lmix])

            if np.size(pars) < 1:
                pars = np.copy(pars_mix)
                labels_pars = labels_mix[:]
            else:
                pars = np.hstack(( pars, pars_mix ))
                labels_pars = labels_pars + labels_mix[:]
                
    # lnprobs recognizably absent if not present
    bok = np.isfinite(pars[:,0])
    logprobs = None
    if hasattr(flatsamples, 'log_probs'):
        logprobs = flatsamples.log_probs

        if onlyfinite:
            bok = (bok) & (np.isfinite(flatsamples.log_probs))

    if mainclust:
        if hasattr(flatsamples,'ismain'):
            print("showparsamples INFO - showing flatsamples.ismain=True")
            bok = (bok) & (flatsamples.ismain)

    # trim off first minsample samples
    if minsample > 0:
        print("showparsamples INFO - showing samples > %i" \
              % (minsample))
        bok[0:minsample] = False
            
    # Now set up the figure panels
    npars = np.size(lpars)
    ncols = 2
    nrows = int(npars/2)

    # Allow for odd number of parameters (useful if nuisance
    # parameters)
    if npars % 2 > 0:
        nrows += 1
    
    # matplotlib counts left-right, but we may want parameters to
    # count vertically after the centroid. That's annoying, but we can
    # deal with it here:
    if not shownuisance:
        lplot = np.hstack(( np.arange(2)+1, \
                            np.arange(3, npars, 2), \
                            np.arange(3, npars, 2)+1 ))
    else:
        lplot = np.arange(npars)+1
        
    print("showparsamples DEBUG:", lplot, np.size(lplot), npars)
    
    fig8 = plt.figure(fignum)
    fig8.clf()
    axes = []
    for iax in range(npars):

        # Which way are we adding here...
        iplot = lplot[iax]

        if iax < 1:
            thisax = fig8.add_subplot(nrows, ncols, iplot)
        else:
            thisax = fig8.add_subplot(nrows, ncols, iplot, sharex=axes[0])
        thisax.set_ylabel(labels_pars[iax])
        if iplot > npars-2:
            thisax.set_xlabel('Flat sample number')
        axes.append(thisax)

    # Now populate the axes
    sz=1
    for iax in range(len(axes)):
        xsho = lsam[bok]
        ysho = pars[bok,iax]
        ax = axes[iax]
        
        if np.size(logprobs) > 0:
            dum = ax.scatter(xsho, ysho, c=logprobs[bok], cmap=cmap, \
                             alpha=alpha, s=sz)
            cbar = fig8.colorbar(dum, ax=ax, label='ln(prob)')
            cbar.solids.set(alpha=1)
        else:
            dum = ax.scatter(xsho, ysho, alpha=alpha, s=sz)

    # Ensure the panel labels are readable
    fig8.subplots_adjust(hspace=0.4, wspace=0.6)
            
    # Save the figure to disk
    if len(pathfig) > 3:
        fig8.savefig(pathfig)

    if not showpointing:
        return

    fig11 = plt.figure(11, figsize=(8,3))
    fig11.clf()
    ax1 = fig11.add_subplot(121)

    c = None
    
    if np.size(logprobs) > 0:
        c = logprobs

    # UPDATE - show bok only
        
    dum = ax1.scatter(pars[bok,0], pars[bok,1], c=c[bok], s=sz, cmap=cmap, \
                      alpha=alpha)
    ax1.set_xlabel(labels_pars[0])
    ax1.set_ylabel(labels_pars[1])

    if np.size(logprobs) > 0:
        cbar = fig11.colorbar(dum, ax=ax1, label='lnprob')
        cbar.solids.set(alpha=1)
        
        ax2 = fig11.add_subplot(122)
        dum, _, _ = ax2.hist(logprobs[bok], bins=100, log=histlog)
        ax2.set_xlabel('logprob')

    fig11.subplots_adjust(hspace=0.4, wspace=0.6, left=0.2, bottom=0.2)

    # save this figure to disk
    if len(pathfig) > 4:
        pathsup = '%s_pointing.png' % (os.path.splitext(pathfig)[0])

        fig11.savefig(pathsup)
    
def shownoisesamples(flatsamples=None, nshow=100, fignum=9, \
                     logy=True, showvar=True, \
                     cmap='inferno_r', jaux=2, \
                     alpha=0.1, \
                     showlogprobs=True, \
                     pathfig='test_noisemags.png', \
                     closeaftersave=False, \
                     mainclust=False, minsample=-1):

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

    mainclust = plot the "main" cluster only? (uses flatsamples.bmain
    attribute. Use this ONLY if you are sure the outliers really are
    rogue burn-in samples.)

    minsample = minimum sample to show (negative for all samples)

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

    # 2025-06-14: nothing much to do if the noise isn't already >1
    # dimensional.
    if parsnoise.ndim > 1:
        if parsnoise.shape[-1] < 2:
            print("examine2d.shownoisesamples INFO - parsnoise 1D. Nothing to show:", parsnoise.shape, parssymm.shape)
            return
    
        
    stdxs, stdys, corrxys = noisemodel2d.mags2noise(parsnoise.T, parssymm.T, \
                                                    mshow[:,None], \
                                                    mag0=mag0, \
                                                    returnarrays=True, \
                                                    islog10_c=islog10_c)


    # pick a random sample to show
    bok = np.isfinite(parsnoise[:,0])
    if mainclust:
        if hasattr(flatsamples, 'ismain'):
            print("shownoisesamples INFO - showing flatsamples.ismain=True")
            bok = np.copy(flatsamples.ismain)

            # update nshow if needed
            nshow = np.min([nshow, np.sum(bok)])

    lok = np.where(bok)[0]

    # trim for first sample
    if minsample > -1:
        print("shownoisesamples INFO - showing samples > %i" % (minsample))
    lok = lok[lok > minsample]
    
    # ldum = np.argsort(np.random.uniform(size=parsnoise.shape[0]))
    ldum = np.argsort(np.random.uniform(size=np.size(lok) ))
    lsho = lok[ldum[0:nshow]]
    
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

    # condition-trap
    jaux = np.min([jaux, parsnoise.shape[-1]-1 ])
        
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

    # only show those in the main cluster?
    if mainclust:
        if hasattr(flatsamples,'ismain'):
            lsam = np.where(flatsamples.ismain)[0]

    # enforce minimum sample number
    lsam = lsam[lsam > minsample]
            
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

        # condition trap for two-parameter noise 
        if j > parsnoise.shape[-1]-1:
            continue
        
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
               nmodel=-1, \
               colornuisance='#9A3324', \
               facecolornuisance='mistyrose', \
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
            if np.size(truths) > 0 and truths is not None:
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

                # change the face color (if specified)
                try:
                    ax.set_facecolor(facecolornuisance)
                except:
                    badcolor = True
                    
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
    fig6 = plt.figure(6, figsize=(4,7))
    fig6.clf()
    if nlsq > 0:
        ax61 = fig6.add_subplot(311)
        ax62 = fig6.add_subplot(312)
        ax63 = fig6.add_subplot(313)
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

##### Methods to overlay multiple sets follow

def multicorner(lsamples=['eg10_mix_twoframe_flatsamples_n100_noobs.npy', \
                          'eg10_mix_twoframe_flatsamples_n100_nobc.npy'], \
                lprobs=['eg10_mix_twoframe_lnprobs_n100_noobs.npy', \
                        'test_log_probs.npy'], \
                lsho=['eg10_noobs_test_showargs.pickle', \
                      'test_eg10_nobc_showargs.pickle'], \
                lesa=['eg10_noobs_test_esargs.pickle', \
                      'test_eg10_nobc_esargs.pickle'], \
                fignum=5, \
                llabels=['no obs', 'no bc'], \
                alpha=0.75, \
                lFS = [], \
                pathfig='test_multicorner.png', \
                convert_linear=True, \
                linestyles=['solid', 'dashed', 'dashdot', 'dotted'], \
                linewidths=[1, 1., 1., 1.], \
                levels=[0.393, 0.864], \
                annotate_levels = True, \
                fill_contours=False, \
                lnprob_log=True, \
                alpha_fill=0.25, \
                scmap='', cmapmax=0.9, \
                zorders=[], \
                ticklabelsize=6, \
                rescale=True, round3=False, \
                deg2arcsec=True, \
                arcsecperpix=True, \
                usemaincluster=True, \
                redoclusters=True, \
                pathlis=''):

    """Exoerimental method - show two sets of corner plots.

    INPUTS:

    lFS = list of paths to Flatsamples pickle files. Used in preference of
    the ingredients. 

    llabels = list of dataset labels (for figure legend)

    alpha = transparency parameter for corner, hist, legend

    pathfig = path to output figure

    convert_linear = convert 6term parameters to sxale, rotation, etc.

    If reading in the pieces of the flatsamples, the following are used:

    lsamples = list of .npy flat samples files

    lprobs = list of .npy lnprobs files

    lsho = list of "show" arguments files

    lesa = list of "esargs" arguments files

    linestyles = linestyles for plots

    linewidths = linewidths for color plots

    levels = levels for contours. The defaults [0.393, 0.864]
    correspond to 1, 2 sigma in a 2D gaussian. For 0.5, 1, 2 sigma use
    [0.118, 0.393, 0.864]. (The calculation is 1.0-np.exp(-0.5*n**2)
    where n is the number of "sigma".)

    annotate_levels = annotate the contour levels on the plot

    fill_contours = fill the contour plots

    lnprob_log = histogram of lnprob is logarithmic (vertically)

    alpha_fill = transparency for fill histograms

    scmap = colormap name to use when sampling colors (if zero length,
    ignored)

    cmapmax = max value (0 - 1) to use for the drawing of cmap
    values. (Useful if we don't want the extreme right edge of the
    colormap)
    
    zorders = list of vertical orders for plots

    ticklabelsize = fontsize for corner tick labels

    rescale = rescale the points for tidy plotting?

    round3 = if rescaling, group powers of ten by 3 (e.g. 1e-6, 1e-3, etc.)

    deg2arcsec = if convert to linear AND rescaling, also convert
    deltas in degrees to deltas in arcsec.

    arcsecperpix = scale factors sx, sy in arcsec per pixel (assumes
    target frame in degrees, source frame in pixels)

    usemaincluster = if present, use the samples present in the "main"
    cluster by likelihood (this is for the case where some very small
    number of rogue burn-in samples have slipped in).

    redoclusters = find the lnprob clusters if not already present in
    the flatsamples object

    pathlis = path to list of flat samples and labels (can be quicker
    to edit a parameter file than to use the cursor on the terminal)

    OUTPUTS 

    None - figure is drawn

    """

    # This will probably duplicate some of the functionality of
    # showcorner() above, since we want to do something fairly
    # specific here. Go for readability now, the two could perhaps be
    # merged later on.
    
    # Currently we reconstruct new Flatsamples objects from the
    # outputs of previous runs. This is inefficient (and indeed we
    # probably could just pickle the FS objects from the run...) but
    # hopefully should be easier to split apart when considering the
    # fitting and simulation separately. Input arguments are in list
    # form so that their length can be made flexible as easy as
    # possible.
    FSS = []

    # Allow list of samples and labels to be passed in from parameter
    # file
    lFS_in, llabels_in = loadsampleslist(pathlis, checkpaths=True)

    if len(lFS_in) > 0:
        lFS = lFS_in[:]
        llabels = llabels_in[:]

        print("examine2d.multicorner INFO - loading flatsamples from paths:")
        for path in lFS:
            print("examine2d.multicorner INFO - %s" % (path))
        
    # If a list of flatsamples pickles was passed, use that in
    # preference to the piece-by-piece
    if len(lFS) > 1:
        for iset in range(len(lFS)):
            if os.access(lFS[iset], os.R_OK):
                FSS.append(pickle.load(open(lFS[iset], 'rb'))) 
    else:
        for iset in range(len(lsamples)):
            FSS.append(Flatsamples(path_samples=lsamples[iset], \
                                   path_log_probs=lprobs[iset], \
                                   path_showargs=lsho[iset], \
                                   path_esargs=lesa[iset]) )

    # Do some checking of the Flatsamples objects we just loaded:
        
    # Let's assume we are only interested in the model parameters
    # (since the number of nuisance parameters may vary between
    # comparison cases). Trust the user to try to compare two obsets
    # that have the same number (and meaning) of model parameters, for
    # now...
    for iset in range(len(FSS)):

        # Do we know which are the model parameters?
        if not 'inds_abc' in FSS[iset].showargs['corner'].keys():
            print("multicorner WARN - inds_abc not in showargs keys for FS %i")
            return

        # because we have read these in as files, the flat_samples,
        # log_probs and nsamples are all attributes of the local
        # instance, which are not currently returned or used by
        # anything else. So we can safely modify them in-place. So -
        # here we cut out the rogue burn-ins if asked.
        if not usemaincluster:
            continue

        sstart="" # For reporting status to screen later on
        if not hasattr(FSS[iset],'ismain'):
            if not redoclusters:
                continue
            else:
                print("examine2d.multicorner INFO - isolating main logprob cluster for set %i (%s)..." % (iset, llabels[iset] ), end="")
                FSS[iset].clusterbylogprob()
        else:

            # what to put at the start of the status line
            sstart="examine2d.multicorner INFO - trimmed set %i (%s):" \
                % (iset, llabels[iset])
                
        ismain = FSS[iset].ismain
        if np.size(ismain) != FSS[iset].nsamples:
            continue

        if np.sum(~ismain) < 1:
            continue
        
        # ok now we can trim!
        print("%s %i of %i" \
              % (sstart, np.sum(ismain), np.size(ismain) ))
        FSS[iset].flat_samples = FSS[iset].flat_samples[ismain,:]
        FSS[iset].log_probs = FSS[iset].log_probs[ismain]
        FSS[iset].countsamples()
        
        
    # now we can proceed! Set up the figure:
    figc = plt.figure(fignum, figsize=(8,7))
    figc.clf()

    # plot colors
    colos = ['b','r','g','m']

    # umich marketing colors (why not...)
    colos = ['#00274C', '#D86018', '#702082', '#9B9A6D']

    # or try this...
    try:
        cmap = plt.get_cmap(scmap)
        colos = cmap(np.linspace(0., cmapmax, 4, endpoint=True))
    except:
        nocmap=True
        
    # for legend
    legends = []

    # histogram arguments, which we will want to replicate across the
    # corner histograns, the lnprob histogram, and also the legends
    lhistargs = []

    # Initialize a few things so that conditionals later on won't
    # break if the user asks for pieces that may or may not require
    # each other
    gangl=np.array([])  # which (if any) columns are angles
    gscale=np.array([]) # which (if any) are s_x or s_y
    
    # If we are going to do the rescaling and shifting, we need to
    # initialize those arguments (which will be determined for the
    # first pass through, then kept for the remaining passes. So:
    midpts=np.array([])
    scales=np.array([])
    labelpad=0.7
    
    # Create zorders for plot
    if len(zorders) < len(FSS):
        zorders = list(10 - np.arange(len(FSS)))
    
    # start with the zeroth case only while debugging
    for iset in range(len(FSS)):
        inds_abc = FSS[iset].showargs['corner']['inds_abc']
        labels = np.array(FSS[iset].showargs['corner']['labels'])[inds_abc]
        sampls = FSS[iset].flat_samples[:,inds_abc]

        truths = None
        if iset < 1:
            truths = FSS[iset].showargs['corner']['truths'][inds_abc]
            
        # Convert 6term to linear?
        if convert_linear:
            samplsall, labelsall, truthsall = \
                sixterm2d.flatpars(FSS[iset].flat_samples, \
                                   inds_abc, \
                                   FSS[iset].showargs['corner']['labels'], \
                                   FSS[iset].showargs['corner']['truths'])

            sampls = samplsall[:,inds_abc]
            labels = np.array(labelsall)[inds_abc]

            if arcsecperpix:
                
                # do our hack for the string matching
                isscale=stringsmatch(labels, ['s_x','s_y'])
                gscale = np.array(np.where(isscale),'int').squeeze()
                sampls[:,gscale] *= 3600. # arcsec per degree
                                                
            # switch across to the truths
            if iset < 1:
                truths = np.array(truthsall)[inds_abc]

                if arcsecperpix:
                    truths[gscale] *= 3600. #arcsec per degree

        # If we are rescaling, our "midpoints" should be initialized
        # to the truths, however they were processed above.
        if iset < 1:
            midpts = np.copy(truths)

            # The version of the labels for the title and for the axis
            # label may end up differing, so we split off a copy here.
            labelstitl=labels[:]

            
        # reweight for the histograms
        wts = np.ones(FSS[iset].nsamples)/FSS[iset].nsamples

        # rescale the points so that matplotlib's autoscaling doesn't
        # put text all over the axes
        if rescale:
            sampls, midpts, scales = \
                scaleforcorner(sampls, midpts, scales, 1., \
                               round3=round3)

            if deg2arcsec:

                # Hack to determine which labels contain theta or beta
                isangl = stringsmatch(labels, ['theta','beta'])
                gangl = np.asarray(np.where(isangl), 'int').squeeze()

                # now rescale the deltas for the impacted columns
                if np.sum(isangl) > 0:
                    sampls[:,gangl] *= 3600.*scales[gangl]
                    scales[gangl] = 1.
                    
            # Update the plot labels. We have to actually rebuild the
            # labels because the np.array trick above imposes a
            # uniform length. There is a difference between "" and ''
            # here, so we use the join trick below to enforce the kind
            # of string we want.
            labelsnew = []
            for ilabel in range(len(labels)):                
                powten = int(np.log10(scales[ilabel]))
                slab = r"$\Delta$%s" % (str(labels[ilabel]))

                # Hack to add the units to the axis label
                if deg2arcsec:
                    if isangl[ilabel]:
                        slab = r'%s (")' % (slab)

                # hack to avoid 10^0
                if np.abs(powten) > 0:
                    spow = r"$\times 10^{%i}$" % (powten)
                    slabel = '\n'.join([slab, spow])
                else:
                    slabel = slab[:]
                    
                labelsnew.append(slabel)

            # hack to keep access to the original labels
            labels_orig = labels[:]
            labels = labelsnew[:]
            labelpad = 0.25
            
            # Ensure the truths are offset once
            if iset < 1:
                truths = truths - midpts
            
                
        print("examine2d.multicorner INFO - plotting set %i: %i..." \
              % (iset, FSS[iset].nsamples))

        # Some customization
        iline = iset % len(linewidths)
        lw = linewidths[iline]
        ls = linestyles[iline]
        contour_kwargs = {'linewidths':lw, 'linestyles':ls, \
                          'zorder':zorders[iset]}

        # try sending the fill and edge colors including the
        # transparency
        color_edge = colorConverter.to_rgba(colos[iset], alpha=1.)
        color_fill = colorConverter.to_rgba(colos[iset], alpha=alpha_fill)

        # label_kwargs={'fontsize':8}
        
        # Construct the histogram arguments and keep them so we can
        # use them later
        hist_kwargs = {'linewidth':lw, 'linestyle':ls, \
                       'histtype':'stepfilled', \
                       'edgecolor':color_edge, \
                       'color':color_fill, \
                       'zorder':zorders[iset]}

        lhistargs.append(hist_kwargs)
        
        cornr = corner.corner(sampls, labels=labels, truths=truths, \
                              weights=wts, \
                              truth_color='0.4', \
                              fig=figc, \
                              labelpad=labelpad, \
                              use_math_text=True, \
                              plot_datapoints=False, color=colos[iset], \
                              alpha=alpha, \
                              contour_kwargs=contour_kwargs, \
                              hist_kwargs=hist_kwargs, \
                              levels=levels, \
                              fill_contours=fill_contours)

        # accumulate for legend
        legends.append(mpatches.Patch(label=llabels[iset], \
                                      facecolor=hist_kwargs['color'], \
                                      edgecolor=hist_kwargs['edgecolor'], \
                                      linewidth=hist_kwargs['linewidth'], \
                                      linestyle=hist_kwargs['linestyle'] ))

        # If we rescaled, then show the midpoints on the plot
        if rescale:
            npar = midpts.size
            axes = np.array(figc.axes).reshape((npar, npar))

            # print("INFO2 - labelstitl:", labelstitl)
            
            # loop through the model dimensions
            for jpar in range(npar):
                axh = axes[jpar, jpar]

                # I don't see a way to use matplotlib's useMathText
                # for the title, so we put in a few special cases
                # here.
                stitl='%s: %s' % (labelstitl[jpar], \
                                  scalarstring(midpts[jpar]) )

                # hack for degrees
                if deg2arcsec:
                    if jpar in gangl:
                        stitl=r'%s$^o$' % (stitl)

                # hack for arcsec per pix:
                if arcsecperpix:
                    if jpar in gscale:
                        stitl=r'%s "/pix' % (stitl)
                        
                axh.set_title(stitl, fontsize=8)
                
        
    # Update the tick fontsize using the same trick as showcorner above
    for ax in figc.get_axes():
        ax.tick_params(axis='both', labelsize=ticklabelsize)
        
    # activate the legends for the legends axis
    axleg = figc.axes[2]
    leg = axleg.legend(handles=legends, fontsize=6, frameon=False)

    # Annotate the levels?
    if annotate_levels:
        slevs = "Contour levels:\n%s" % ", ".join("%.3f" % (lev) for lev in levels)

        dumlevs = axleg.annotate(slevs, (0.05,0.05), \
                                 xycoords='axes fraction', \
                                 va='bottom', ha='left', \
                                 fontsize=6)
        
    # the likelihoods plot requires a separate loop after the corner
    # plot has been made, because we want to override some of its
    # settings
    for iset in range(len(FSS)):
        
        # likelihoods plot - consider making ncoarse for this
        npars = sampls.shape[-1]
        if iset < 1:
            axlike = figc.add_subplot(npars-2, npars-2, npars-2)

        hist = axlike.hist(FSS[iset].log_probs, \
                           density=False, \
                           log=lnprob_log,\
                           **lhistargs[iset])

    axlike.set_xlabel('ln(prob)')
    axlike.set_ylabel('N')
        
    # Do the figure subplots adjust grumble grumble
    figc.subplots_adjust(bottom=0.2, left=0.2, top=0.95)

    # Save the figure to disk
    if len(pathfig) > 3:
        figc.savefig(pathfig)

def scaleforcorner(samples=np.array([]), middles=np.array([]), \
                   sfacs=np.array([]), \
                   logquant=1., telldebug=False, \
                   round3=False, log10noscale=[-2,1] ):

    """Returns samples as scaled deltas from supplied (or calculated)
middle values, and the scale factors. This duplicates some of the
functionality of matplotlib's auto scaling of labels, but gives us
full control of the placement etc. of the results.

    INPUTS

    samples = [nsamples, ndim] array of flat samples

    middles = [ndim] array of anchor values for the deltas. If not
    supplied, the median along the samples is calculated.

    sfacs = [ndim] array of scale factors. Determined if not supplied.

    logquant = quantile used when evaluating the range of powers of
    ten of the (abs) deltas. Set 1.00 for max.

    round3 = round to the next power of 3 down? (Like engineering
    notation)

    telldebug = print debug messages to screen

    log10noscale = [-2,1] = don't do any rescaling if log10(quant) is
    between these values (useful if the values are already in a
    human-sensible range)

    OUTPUTS

    deltas = [nsamples, ndim] scaled samples

    midpts = [ndim] midpoints for the deltas

    scales = [ndim] scale factors (to be reported in the plotting)

    """

    # initialize return values (in case bad)
    deltas = np.array([])
    midpts = np.array([])
    scales = np.array([])

    if np.size(samples) < 1 or np.ndim(samples) < 2:
        return deltas, midpts, scales
    nsamples, npars = samples.shape

    # the midpoints (copy rather than ref to allow changing in-place)
    if middles.size != npars:
        midpts = np.median(samples, axis=0)
    else:
        midpts = np.copy(middles)

    # Subtract off the midpoints
    deltas_raw = samples - midpts[None,:]

    # allow passing in of the scale factors
    if np.size(sfacs) == npars:
        scales = np.copy(sfacs)
    else:
        # now find the indicator powers of ten for the deltas, go to
        # the next power of ten down (so that the max will go to about
        # ten in those units)
        pow10 = np.quantile(np.log10(np.abs(deltas_raw)), logquant, axis=0)
        pow10floor = np.floor(pow10)
        if round3:
            pow10floor = 3.*np.floor(pow10/3)

        # set pow10floor to zero for cases within log10noscale
        bnoscale = (pow10floor > log10noscale[0]) & \
            (pow10floor < log10noscale[-1])

        #print("RESCALE INFO:")
        #print(pow10floor)
        #print(bnoscale)
        
        pow10floor[bnoscale]=0.
        
        scales = 10.0**(pow10floor)

        
    # Now convert the deltas into multiples of 10**(these powers)
    deltas = deltas_raw  /scales[None, :]

    if telldebug:
        print(samples.shape)
        print(midpts.shape)
        print(deltas.shape)
        print(pow10)
        print(pow10floor)
        print(deltas_raw[0])
        print(deltas[0])

    return deltas, midpts, scales

def scalarstring(valu=0.):

    """Returns literal string of float for plotting

    INPUT

    valu = value to return

    RETURNS

    svalu = string version of value, nicely formatted for printing."""

    pow10 = np.log10(np.abs(valu))

    # we make a few decisions here about what kind of output we
    # want. There must be a more flexible way to o this, but for the
    # moment let's just hand-implement them...
    if pow10 > -2:
        if pow10 < -1:
            return r'%.4f' % (valu)
        if pow10 < 0:
            return r'%.3f' % (valu)
        if pow10 < 3:
            return r'%.2f' % (valu)

    # if here, then we're in the regime in which we do want scientific
    # notation. So:
    
    ssci = '%.3e' % (valu)
    arg, expon = ssci.split('e')
    expon = '%s%s' % (expon[0], expon[1::].strip('0'))
           
    return r'$%s\times 10^{%s}$' % (arg, expon)
        
def stringsmatch(src=[], targ=[]):

    """Returns boolean array for list entries missing any of the input
strings

    INPUTS

    src = [] list of source terms

    targ = [] list of search terms

    OUTPUTS 

    bmatch = boolean giving whether any of the src terms contain any
    of the srch terms

"""

    bany = np.repeat(False,len(src))

    if len(src) < 1 or len(targ) < 1:
        return bany

    for sstr in targ:
        bthis = [sthis.find(sstr) > -1 for sthis in src]
        bany = bany + bthis

    return bany

def loadsampleslist(pathlist='', checkpaths=True):

    """Loads list of flat samples for multicorner.

    INPUT:

    pathlist = ascii file containing the paths and labels for the flat
    samples to go into multicorner. 

    checkpaths = only include paths that are readable

    The path list must have the following format:

           ./path/to/samples.pickle
           label

    i.e. the path and label are on alternating lines (to make the
    files easier to edit from the command line). Lines beginning with
    "#" are ignored.

    Output:

    lpaths = list of paths to flat samples

    labels = list of labels for plot

    """

    # Written because it's faster to manipulate parameter files than
    # it is to drag the cursor back and forth on the ipython command
    # line.
    
    if not os.access(pathlist, os.R_OK):
        return [], []

    paths = []
    labels = []
    
    with open(pathlist, 'r') as robj:
        count = 0
        for line in robj:

            # ignore comments and blank lines
            if line[0].find('#') > -1 or len(line.strip()) < 1:
                continue

            line = line.strip()
            
            if count % 2 == 0:
                paths.append(line)
            else:
                labels.append(line)

            count = count + 1

    if checkpaths:
        pout = []
        lout = []

        for ipath in range(len(paths)):
            if os.access(paths[ipath], os.R_OK):
                pout.append(paths[ipath])
                lout.append(labels[ipath])

        return pout, lout

    # If we are NOT checking readability, just return what we read in
    return paths, labels
