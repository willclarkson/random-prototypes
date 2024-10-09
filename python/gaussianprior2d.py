#
# gaussianprior2d.py
#

# WIC 2024-10-07: object and methods to handle informative Gaussian
# prior. Written with the 6-term geometric parameter-set [xo, yo, sx,
# sy, theta, beta] in mind, where 0-6 of these quantities may have
# informative priors set.

import numpy as np
import numpy.ma as ma
import configparser

class gaussianprior(object):

    def __init__(self, pathpars='', Verbose=True):

        # The main attributes we need
        self.center = np.array([])
        self.precis = np.array([]) # the inverse covariance matrix
        self.lpars = np.array([])
        
        self.covar = np.array([])
        
        # control variable
        self.Verbose = Verbose
        
        # parameter labels (useful to show the ordering convention)
        self.labels = [r'$x_0$', r'$y_0$', r'$s_x$', r'$s_y$', \
                       r'$\theta$', r'$\beta$']
        
        # Prior parameters specified in input file
        self.pathpars=pathpars[:]
        self.conf_section='Prior'

        # the up-front constant in the prior
        self.constant = 0.
        
        # load parameters on initialization
        if len(self.pathpars) > 3:
            self.loadpars(self.pathpars)
            self.invertcov()
            self.calcnorm()
            
    def loadpars(self, pathconfig=''):

        """Loads the prior parameters"""

        # Refuse to load too short a parameter path
        if len(pathconfig) < 3:
            return

        # parameter names for the means
        names_center = ['x0', 'y0', 'sx', 'sy', 'theta', 'beta']

        # Now open the configuration file and read it in
        config = configparser.ConfigParser()
        try:
            config.read(pathconfig)
        except:
            print("gaussianprior.loadpars WARN - problem loading config %s "\
                  % (pathconfig))
            return

        if not self.conf_section in config.sections():
            print("gaussianprior2d.loadconfig WARN - section %s not in file %s" % (self.conf_section, pathconfig))
            return

        
        conf = config[self.conf_section]

        # For which geometric parameters do we have means?
        notfound_centers = []
        found_centers = []
        lcenters = []
        vcenters = []

        # Dictionary of covariance components that also have supplied
        # entries in the location parameters. (Done as a dictionary
        # because it's easy to find any cases for which a mean but no
        # covariance was supplied).
        dvars = {}
        
        for icenter in range(len(names_center)):
            attrib = names_center[icenter]
            try:
                # Only pay attention to non-None floats
                if conf[attrib].find('None') < 0:
                    thisfloat = conf.getfloat(attrib)
                    vcenters.append(thisfloat)
                    lcenters.append(icenter)
                    found_centers.append(attrib)
                    dvars[attrib] = {}
            except:
                notfound_centers.append(attrib)

        # boolean - are the off-diagonals actually correlation
        # coefficients?
        offdiag_are_rho = True
        try:
            offdiag_are_rho = conf.getboolean('offdiag_are_rho')
        except:
            notoffdiag = True
                
        # Now pass these up to the instance as arrays
        self.lpars = np.asarray(lcenters, dtype='int')
        self.center = np.asarray(vcenters)

        if len(found_centers) < 1:
            return

        # Any missing covariances?
        notfound_covs = []

        # We populate a covariance matrix here, with the same side
        # length as the number of parameters for which a central
        # location was given
        ncen = len(found_centers)
        cov = np.zeros((ncen, ncen))
        bok = np.zeros((ncen, ncen), dtype='bool')
        
        # Now we look for covariance components corresponding to the
        # means we have read in.
        for icen in range(len(found_centers)):
            scen = found_centers[icen]
            for jcen in range(len(found_centers)):

                # only need e.g. x0beta and not betax0
                if icen > jcen:
                    continue
                
                scov = found_centers[jcen]

                # Construct covariance component name. We'll go with a
                # convention where the user inputs the standard
                # deviation (of an object with itself) and the
                # off-diagonal covariance term (since it's probably
                # taken from a covariance matrix anyway). This might
                # be a bad choice...
                if icen != jcen:
                    covcomp = 'r_%s%s' % (scen, scov)
                else:
                    covcomp = 's_%s' % (scen)

                # If this component is actually in the parameter file,
                # read it in. Otherwise, move on
                try:
                    if conf[covcomp].find('None') < 0:
                        thisfloat = conf.getfloat(covcomp)
                except:
                    notfound_covs.append(covcomp)
                    continue

                # Populate the covariance matrix with the given entries
                if icen != jcen:
                    thisentry = thisfloat
                else:
                    thisentry = thisfloat**2
                    
                cov[icen, jcen] = thisentry
                cov[jcen, icen] = thisentry # matrix is square
                bok[icen, jcen] = True
                bok[jcen, icen] = True
                       
                # Update our record of which quantities had covariance
                # entries. Record the indices in the output covariance
                # matrix to make things easier later on
                dvars[scen][scov] = [icen, jcen, thisfloat]
                    
        # if the off-diagonals are supplied as rho (correlation
        # coefficients) rather than the straight covariance entries,
        # correct for that here.
        if offdiag_are_rho:
            cov = self.offdiag_as_rho(cov, True)

        # If there are any cases for which the center and at least one
        # covariance component was not supplied, the entire row of the
        # covariance matrix will be zero (and thus the matrix will be
        # singular). Therefore, the covariance matrix (and center and
        # index arrays) must be trimmed for validity. Here's one way
        # to do this:
        bbad = np.sum(bok, axis=1) < 1
        if np.sum(bbad) < 1:
            self.covar = cov
            return

        # There must be a clever pythonic way to extract only the
        # non-masked elements from a 2d square array into a new 2d
        # square array, but I'm not aware of it... since we only have
        # to do this once, we can afford a loop:
        ngood = np.sum([~bbad])
        lgood = np.arange(np.size(bbad))[~bbad]

        # The trimmed arrays
        centtrim = np.zeros((ngood))
        covtrim = np.zeros((ngood, ngood))
        for igood in range(ngood):
            for jgood in range(ngood):
                covtrim[igood, jgood] = cov[lgood[igood], lgood[jgood]]

        # Don't forget to trim the center and indices arrays as well
        self.covar = covtrim
        self.center = self.center[lgood]
        self.lpars = self.lpars[lgood]
                
        # How does our covariance matrix look?
        #print(cov)
        #print("###")
        #print(covtrim)
        #print(mask)
        print("###")
        print(self.center)
        print(self.lpars)
        print(cov)

    def offdiag_as_rho(self,cov=np.array([]), diag_is_variance=True):

        """Interprets off-diagonal components in the covariance matrix as
correlation coefficients rho, populating the off-diagonal comopnents as

        v_ij = rho_ij * var[i] * var[j]

"""

        if np.size(cov) < 1:
            return

        if self.Verbose:
            print("offdiag_as_rho INFO - interpreting offdiags as correlation coefficients")
        
        diag = np.diag(np.diag(cov))
        if diag_is_variance:
            diag = np.sqrt(diag)


        offdiag = np.copy(cov)
        offdiag[np.diag_indices(offdiag.shape[0])]=1.

        return np.dot(diag, np.dot(offdiag, diag))

    def invertcov(self):

        """Produces the precision matrix from the covariance matrix"""

        # Nothing to do if the covariance is not populated
        if np.size(self.covar) < 1:
            return
        
        npars = np.shape(self.covar)[0]
        self.precis = np.zeros((npars, npars))

        try:
            self.precis = np.linalg.inv(self.covar)
        except:
            print("gaussianprior.invertcov WARN - problem inverting covariance matrix")

    def calcnorm(self):

        """Computes the up-front constant in the prior"""

        if np.size(self.covar) < 1:
            self.constant = 0.
            return

        ndim = np.shape(self.covar)[0]

        norm1 = -0.5 * ndim * np.log(2.0*np.pi)
        norm2 = -0.5 * np.log(np.linalg.det(self.covar))

        self.constant = norm1 + norm2
        
        
    def getlnprior(self, testpars=np.array([]) ):

        """Evaluates the prior on a set of input parameters and returns
ln(prior) as a single scalar.

        """

        # We might not actually have a prior after all
        if np.size(self.center) < 1:
            return 0.
        
        # Might be feeding all six parameters, or just the subset.
        if testpars.size > self.center.size:
            delta = testpars[self.lpars] - self.center
        else:
            delta = testpars - self.center

        utVu = np.dot(np.transpose(delta), np.dot(self.precis, delta))
            
        return 0. -0.5 * utVu + self.constant
            

    def drawsample(self, nsamples=1, seed=None):

        """Draws a sample from the prior distribution"""

        if np.size(self.center) < 1:
            return np.array([])

        rng = np.random.default_rng(seed)
        parsran = np.random.multivariate_normal(self.center, self.covar, \
                                                size=nsamples)

        return parsran
        
#######

def testpriorpars():

    """Try loading prior parameters"""

    GG = gaussianprior('test_priorparams.ini')

    print("=====")
    print(GG.center)
    print(GG.lpars)
    print(GG.covar)
    print(np.asarray(GG.labels)[GG.lpars])
    print(GG.precis)

    # Test evaluation, on a single set of generated parameters
    parsran = np.array([])
    if np.ndim(GG.covar) is 2:
        parsran = np.random.multivariate_normal(GG.center, GG.covar)
        w, v = np.linalg.eig(GG.covar)
        print("testpriorpars INFO - GG.cov eig:")
        print("     eigenvalues: ", w)
        print("     eigenvectors:", v)

    print("testpriorpars INFO: test params:", parsran.squeeze())

    # Evaluate ln(prior) of this paramset
    lnprob = GG.getlnprior(parsran.squeeze())

    print("testpriorpars INFO: lnprob:", lnprob)
    
def testblank():

    """Tests behavior if no informative prior is needed"""

    GP = gaussianprior()

    # Generate some test parameters
    npar = 6
    cmean = np.zeros(npar)
    ccov = np.eye(npar)
    #parsran = np.random.multivariate_normal(cmean, ccov)

    parsran = GP.drawsample()
    
    print(cmean)
    print(ccov)
    print(parsran)

    lnprob = GP.getlnprior(parsran)
    print("gaussianprior2d.testblank INFO - ln(prior):", lnprob)
    print(len(GP.lpars))
