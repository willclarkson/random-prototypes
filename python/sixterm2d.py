#
# sixterm2d.py
#

# Utilities to convert between {a,b,c,d,e,f} and {sx, sy, theta, beta}
# - to be callable from other methods.

# Inputs: flat_samples, indices, maybe labels
#
# outputs: flat_samples, updated labels

import numpy as np

class sixterm(object):

    """Convenience class for six-term transformation"""

    def __init__(self, pars=np.array([]), inds=np.array([]), labels=[], \
                 reorder=True):

        # control variables
        reorder=reorder
        
        # Inputs
        self.ndim = np.ndim(pars)
        self.parsin = np.atleast_2d(pars)
        self.inds = np.asarray(inds, dtype='int') # force integer array
        self.labelsin = labels[:]

        if np.size(self.inds) < 1:
            self.inds=np.arange(self.parsin.shape[-1], dtype='int')
        
        # coefficients abcdef
        self.a = np.array([])
        self.b = np.array([])
        self.c = np.array([])
        self.d = np.array([])
        self.e = np.array([])
        self.f = np.array([])

        # in terms of transformations
        self.sx = np.array([])
        self.sy = np.array([])
        self.theta = np.array([])
        self.beta = np.array([])
    
        # Labels for the outputs. The blank strings are there because
        # we probably don't want to replace the labels for the central
        # point. (That central point is mostly coming along for the
        # ride anyway.
        self.labels = ['', r'$s_x$', r'$s_y$', '', r'$\theta$', r'$\beta$']

        # output quantities
        self.parsout = np.array([])
        self.labelsout = []
        
        # Run on initialization
        self.populateabc()
        if np.size(self.b) > 0:
            self.computecoeffs()
            self.enforceconvention()
            self.angles2deg()
            
    def populateabc(self):

        """Populates the abcdef parameters from the input set"""

        if np.size(self.parsin) < 1:
            return

        if np.size(self.inds) < 6:
            return

        # Ensure at least [1, npars] so that we can apply conditions
        # uniformly
        self.a = self.parsin[:, self.inds[0] ]
        self.b = self.parsin[:, self.inds[1] ]
        self.c = self.parsin[:, self.inds[2] ]
        self.d = self.parsin[:, self.inds[3] ]
        self.e = self.parsin[:, self.inds[4] ]
        self.f = self.parsin[:, self.inds[5] ]
        
    def computecoeffs(self):

        """Computes the coefficients sx, sy, theta, beta from abcdef"""

        # If abcdef not populated yet, do nothing.
        if np.size(self.b) < 1:
            return

        self.sx = np.sqrt(self.b**2 + self.e**2)
        self.sy = np.sqrt(self.c**2 + self.f**2)

        cf = np.arctan2(self.c, self.f)
        eb = np.arctan2(self.e, self.b)

        self.theta = 0.5*(cf - eb)
        self.beta = (cf + eb)

    def enforceconvention(self):

        """Enforces convention on the computed parameters"""

        # if beta > pi/2, theta -> theta + pi/2 and sx is the negative
        # sqrt
        halfpi = np.pi * 0.5
        betahi = self.beta > halfpi
        self.theta[betahi] += halfpi
        self.beta[betahi] -= np.pi
        self.sx[betahi] = 0. - np.abs(self.sx[betahi])

        betalo = self.beta < -halfpi
        self.theta[betalo] -= halfpi
        self.beta[betalo] += np.pi
        self.sx[betalo] = 0. - np.abs(self.sx[betalo])

    def angles2deg(self):

        """Converts the beta, theta to degrees"""

        self.theta = np.degrees(self.theta)
        self.beta = np.degrees(self.beta)

    def getoutput(self):

        """Populates and returns the output parameters and labels.

Returns:

        parsout = [nsamples, npars] array of output params

        labels = [npars] list of plot-ready labels"""

        self.buildoutputpars()
        self.reorderoutput()
        self.trimoutput()

        return self.parsout, self.labelsout
        
    def buildoutputpars(self):

        """Builds output parameter array"""

        # Initialize by copy of the input parameters (including extra
        # parameters, nuisance parameters, etc.)
        self.parsout = np.copy(self.parsin)

        # Also labels
        self.labelsout = self.labelsin[:]
        
        # now slot in the updated values
        self.parsout[:,self.inds[0]] = self.a
        self.parsout[:,self.inds[1]] = self.sx
        self.parsout[:,self.inds[2]] = self.sy
        self.parsout[:,self.inds[3]] = self.d
        self.parsout[:,self.inds[4]] = self.theta
        self.parsout[:,self.inds[5]] = self.beta
        
        # now the labels. Unless there's a clever pythonic way to
        # slice lists by arbitrary indices, we just build this. Do
        # nothing if we haven't bothered to set the output labels
        # array
        if np.size(self.labelsout) < 1:
            return
        
        linds = np.array([1,2,4,5], dtype='int')
        for iind in linds:
            self.labelsout[self.inds[iind]] = self.labels[iind]

    def reorderoutput(self):

        """Reorders the output (and labels) into the order [xo, yo, sx, sy,
theta, beta, others]"""

        is6term = np.zeros(self.parsin.shape[-1], dtype='bool')
        is6term[self.inds] = True

        # Counters for [xo, yo, sx, sy, theta, beta] in the output
        # array
        ldum = [0,3,1,2,4,5]
        l6term = self.inds[ldum]

        lall = np.arange(np.size(is6term), dtype='int')
        lrest = lall[~is6term]

        lreorder = np.hstack(( l6term, lrest ))

        # now actually reorder the output
        self.parsout = self.parsout[:,lreorder]

        # Return here if we don't care about the labels
        if np.size(self.labelsout) < 1:
            return
        
        alabels = np.asarray(self.labelsout)
        self.labelsout = list(alabels[lreorder])

    def trimoutput(self):

        """If input ndim=1, trims the output down to this size"""

        # Nothing to do if originally given 2d input
        if self.ndim > 1:
            return

        self.parsout = self.parsout.squeeze()
        
####

def flatpars(flatsamples=np.array([]), inds=np.array([]), labels=[], \
             truths=np.array([]) ):

    """One-liner to convert the linear parameters in flat samples to sx,
sy, theta, beta, including reordering and labeling"""

    ST = sixterm(flatsamples, inds, labels)
    samples_out, labels_out = ST.getoutput()

    # try this on the truths array
    truths_out=None # same default as examine2d.showcorner() 
    if np.size(truths) > 0:
        TT = sixterm(truths, inds, labels)
        truths_out, _ = TT.getoutput()

    return samples_out, labels_out, truths_out

def abcfromgeom(pars=np.array([]), degrees=True):
    
    """Converts [xo, yo, sx, sy, theta, beta] into [a,b,c,d,e,f]
parameters, in that order.

Inputs:

   pars = [x0, y0, sx, sy, theta, beta] .  If length 4, interpreted as
   [sx, sy, theta, beta]

    degrees = angles are in degrees, otherwise assumed to be radians

Returns:

   abc = [a,b,c,d,e,f] - linear parameters

    """

    if np.size(pars) < 4:
        return np.array([])

    # If 4-element parameters passed, interpret as [sx, sy, theta,
    # beta]. Otherwise interpret as [x0, y0, sx, sy, theta, beta]
    imin = 0.
    a = 0.
    d = 0.
    if np.size(pars) > 5:
        imin = 2
        a = pars[0]
        d = pars[1]
        
    # Ensure angles are interpreted as radians
    angleconv = 1.
    if degrees:
        angleconv = np.pi / 180.

    sx = pars[imin]
    sy = pars[imin+1]
    thetarad = pars[imin+2] * angleconv
    betarad = pars[imin+3] * angleconv
    
    # Compute the linear parameters...
    b =  sx * np.cos(thetarad - betarad*0.5)
    c =  sy * np.sin(thetarad + betarad*0.5)
    e = -sx * np.sin(thetarad - betarad*0.5)
    f =  sy * np.cos(thetarad + betarad*0.5)

    # ... and slot into return array
    return np.array([ a, b, c, d, e, f ])
    

def getpars(abc=np.array([]), \
            hasprior=np.ones(6, dtype='bool') ):

    """One-liner to convert [a,b,c,d,e,f] parameters into geometric parameters, optionally selecting on priors. Returns the subset of parameters for which hasprior=True, in the order given below.

Inputs:

    abc = [a,b,c,d,e,f] array of linear parameters

    hasprior = optional boolean array indicating which of the OUTPUT
    parameters [a,d,sx, sy, theta, beta] we want back. This is useful
    when extracting only the parameters on which we might have
    informative priors.

Returns:

    [a,d,sx,sy,theta,beta][haspriors] = array of selected geometric
    parameters

"""
    
    ST = sixterm(abc)#, inds=np.arange(6))
    ST.buildoutputpars()
    ST.reorderoutput()
    ST.trimoutput()

    return ST.parsout[hasprior]
