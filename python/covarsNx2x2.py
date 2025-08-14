#
# covarsNx2x2.py
#

#
# holds class CovarsNx2x2 so that we can import this without requiring
# astropy. This was copied out of weightedDeltas.py on Tues Aug 14,
# 2025.
#

import numpy as np
import matplotlib.pylab as plt

class CovarsNx2x2(object):

    """Given a set of covariances as an Nx2x2 stack, or as the
    individual components (as might be read from an astrometric
    catalog), populates all the following forms: {covariance stack},
    {x-y components}, {major and minor axes and rotation angles}. Any
    of the three forms can be supplied: the stack is populated in the
    following order of preference: {covars, xy components,
    abtheta}. Computes various useful intermediate attributes that are
    useful when plotting: my coverrplot uses this fairly extensively.

    Also contains methods to generate datapoints described by the
    covariance stack. 

    Initialization arguments, all optional:

    covars = N x 2 x 2 stack of covariances. 

    --- If supplying the covariances in x, y, and correlations:

    stdx = N-element sqrt(Var) in xx 

    stdy = N-element sqrt(Var) in yy
        
    corrxy = N-element xy correlation coefficients

    --- If supplying the covariances as a, b, theta:

    majors = N-element array of major axis lengths [If supplying axes]

    minors = N-element array of minor axis lengths

    rotDegs = N-element array of rotation angles

    --- If supplying xy samples:

    xysamples = [N,2, nsamples] samples in X, Y for which to compute
    the covariances

    --- arguments for generating data follow ---

    nPts = number of points to generate
    
    rotDeg = scalar, typical rotation angle

    aLo, aHi = bounds on the major and minor axes
    
    ratLo, ratHi = bounds on the minor:major axis ratios

    genStripe = The covariances of the back half of the generated
                sample are flipped in the x-axis

    stripeFrac = fraction of the sample that will be the `special`
                 set. Default 0.5 .

    stripeCovRatio = ratio of the axis-lengths for the second and
                     first stripe

    """


    def __init__(self, covars=np.array([]), \
                     stdx=np.array([]), stdy=np.array([]), \
                     corrxy=np.array([]), \
                     rotDegs=np.array([]), \
                 xysamples=np.array([]), \
                     majors=np.array([]), \
                     minors=np.array([]), \
                     nPts=100, rotDeg=30., \
                     aLo=1.0, aHi=1.0, \
                     ratLo=0.1, ratHi=0.3, \
                     # ratLo=1., ratHi=1., \
                     genStripe=True, \
                     stripeFrac=0.5, \
                     stripeCovRatio=1.):

        # The covariance stack (which could be input)
        self.covars = np.copy(covars)

        # Another form - the coord-aligned components of the stack
        self.stdx = np.copy(stdx)
        self.stdy = np.copy(stdy)
        self.corrxy = np.copy(corrxy)

        # Or, the covariances could be supplied as major, minor axes
        # and rotation angles
        self.majors = np.copy(majors)
        self.minors = np.copy(minors)
        self.rotDegs = np.copy(rotDegs)

        # Quantities we'll need when generating synthetic data
        self.nPts = nPts
        self.rotDeg = rotDeg
        self.aLo = aLo
        self.aHi = aHi
        self.ratLo = ratLo
        self.ratHi = ratHi

        # Options for some particular patterns can come here.
        self.genStripe = genStripe
        self.stripeFrac = np.clip(stripeFrac, 0., 1.)

        #print("INFO: stripeFrac", self.stripeFrac)

        # ratio between the axis lengths of the two stripes
        self.stripeCovRatio = stripeCovRatio

        # Initialize some internal variables that will be useful for
        # anything that needs abtheta:
        self.VV = np.array([]) # the diagonal covar matrix
        self.RR = np.array([]) # the rotation (+ skew?) matrix
        self.TT = np.array([]) # the transformation matrix 

        # If xy samples were passed in, use them to populate the
        # covariance array
        if np.size(xysamples) > 1:
            self.computeSampleCovariance(xysamples)
        
        # Populate the covariance stack from inputs if any were given
        self.populateCovarsFromInputs()

        # Override npts with the size of the input stack
        if np.size(self.covars) > 0:
            self.nPts = self.covars.shape[0]

        # The sample of deltas (about 0,0)
        self.deltaTransf = np.array([])

        # Labels for the planes. Useful if plotting or outputting, but
        # will remain unused in most applications. This is initialized
        # so that we know what to call this attribute if we decide
        # elsewhere that we do need it after all.
        self.planeLabels = []

    def computeSampleCovariance(self, xy=np.array([]) ):

        """Populates the covars array by computation from input samples.

Inputs:

        xy = [nsamples, 2, ndata] array 

Outputs:

        None - attribute covars is updated.

        """

        # Do nothing if input not 3d
        if np.ndim(xy) != 3:
            return

        nsamples, ndim, ndata = xy.shape

        # calculate the vxx, vyy, vxy terms
        meanxy = np.mean(xy, axis=0)
        var = np.sum((xy - meanxy[None, :, :])**2, axis=0)/(nsamples - 1.)
    
        vxy = np.sum( (xy[:,0,:] - meanxy[None,0,:]) * \
                     (xy[:,1,:] - meanxy[None,1,:]), axis=0 ) /(nsamples-1.)

        ##print("WD DEBUG:")
        ##print(xy[0:4,0,0], np.std(xy[:,0,0]), var[0,0]**0.5)
        ##print(xy[0:4,1,0], np.std(xy[:,1,0]), var[1,0]**0.5)
        
        # assemble the output into an nx2x2 covariance array.
        self.covars = np.zeros(( ndata, ndim, ndim ))
        self.covars[:,0,0] = var[0]
        self.covars[:,1,1] = var[1]
        self.covars[:,0,1] = vxy
        self.covars[:,1,0] = vxy

    def populateCovarsFromInputs(self):

        """Populates the covariance stack from input arguments. In
        order of preference: covars --> xycomp --> abtheta"""

        # if the covariances already have nonzero shape, override nPts
        # with their leading dimension
        if np.size(self.covars) > 0:

            # if passed a plane, turn into a 1x2x2 array:
            if np.size(np.shape(self.covars)) == 2:
                self.covars = self.covars[np.newaxis, :, :]

            self.nPts = np.shape(self.covars)[0]

            # populate the xy components
            self.populateXYcomponents()

        # Or, if we have no covariance but we DO have the XY
        # components, build it that way. covStackFromXY checks that
        # the arrays are nonzero so we don't need to do it here)
        else:
            self.covStackFromXY()  

        # If we still don't have the covariance stack yet, but we do
        # have major axes, then populate the covariance stack from the
        # abtheta form.
        if np.size(self.covars) < 1 and np.size(self.majors) > 0:
            self.covarFromABtheta()

    def populateXYcomponents(self):

        """Populates the X, Y covariance vectors from the covariance
        stack"""

        if np.size(self.covars) < 1:
            return

        # if the covariance is already set, then we just read off the
        # three components. 
        self.stdx = np.sqrt(self.covars[:,0,0])
        self.stdy = np.sqrt(self.covars[:,1,1])

        # Initialize correlation to zero...
        self.corrxy = self.stdx * 0.

        # ... and apply corrxy to nonzero elements
        bok = self.stdx * self.stdy > 0.
        self.corrxy[bok] = \
            self.covars[bok,0,1] / (self.stdx[bok] * self.stdy[bok])

        #self.corrxy = self.covars[:,0,1] / (self.stdx * self.stdy)

    def covStackFromXY(self):

        """Populates the covariance stack from the XY components"""

        nPts = np.size(self.stdx)
        if nPts < 1:
            return

        self.nPts = nPts
        self.covars = np.zeros((self.nPts, 2, 2))
        
        # Now populate the parts. The xx variance must always be
        # populated
        self.covars[:,0,0] = self.stdx**2

        # If the y-component is not given, duplicate the xx part
        if np.size(self.stdy) == self.nPts:
            self.covars[:,1,1] = self.stdy**2
        else:
            self.covars[:,1,1] = self.stdx**2

        # Populate the off-diagonal elements
        if np.size(self.corrxy) != self.nPts:
            return

        covxy = self.corrxy * self.stdx * self.stdy
        self.covars[:,0,1] = covxy
        self.covars[:,1,0] = covxy

    def eigensFromCovars(self):

        """Finds the eigenvalues, eigenvectors and angles from the
        covariance stack"""

        # Get the stacks of eigenvalues and eigenvectors
        w, v = np.linalg.eigh(self.covars)

        # identify the major and minor axes (squared)
        self.majors = w[:,1]
        self.minors = w[:,0]

        # the eigenvectors are already normalized. We'll keep them so
        # that we can use them in plots # WATCHOUT SIGNS
        self.axMajors = v[:,:,1]
        self.axMinors = v[:,:,0]  # Not needed?

        # For cases where the major and minor axes are equal (to
        # rounding error), choose the major axis to point along the
        # x-axis
        bEqu = ~(np.abs(self.majors - self.minors) > 0.)
        if np.sum(bEqu) > 0:
            majorsCopy = np.copy(self.axMajors)
            self.axMajors[bEqu] = self.axMinors[bEqu]
            self.axMinors[bEqu] = majorsCopy[bEqu]
        
        # enforce a convention: if the major axis points in the -x
        # direction, flip both eigenvectors
        bNeg = self.axMajors[:,0] < 0
        self.axMajors[bNeg] *= -1.
        self.axMinors[bNeg] *= -1.

        # the rotation angle of the major axis, avoiding annoying
        # warnings for cases where the major axis points along the
        # y-axis
        self.rotDegs = np.zeros(np.size(self.majors))
        bDeg = self.axMajors[:,0] > 0
        self.rotDegs[bDeg] = \
            np.degrees(np.arctan(self.axMajors[bDeg,1]/self.axMajors[bDeg,0]))
        self.rotDegs[~bDeg] = 90.

        # Having done this, we can now generate the diagonal, rotation
        # and transformation matrix should we wish to generate samples
        # from the deltas.


    def genEigens(self):

        """Generates the eigenvectors of the diagonal covariance matrix"""

        ratios = np.random.uniform(self.ratLo, self.ratHi, self.nPts)

        self.majors = np.random.uniform(self.aLo, self.aHi, self.nPts)
        ratios = np.random.uniform(self.ratLo, self.ratHi, self.nPts)
        self.minors = self.majors * ratios

    def genRotns(self, stripe=True):

        """Generates the rotation angles for the transformation"""

        # 2020-06-12 currently that's only rotation
        self.rotDegs = np.repeat(self.rotDeg, self.nPts)
        
        # stipe the rotation angles? 
        if self.genStripe:
            iPartition = int(self.stripeFrac*np.size(self.rotDegs))
            self.rotDegs[iPartition::] *= -1.

            # Scale the covar axes of the stripe
            self.majors[iPartition::] *= self.stripeCovRatio 
            self.minors[iPartition::] *= self.stripeCovRatio

    def populateDiagCovar(self):

        """Populates diagonal matrix stack with major and minor axes"""

        self.VV = np.array([])

        nm = np.size(self.majors)
        if nm < 1:
            return

        self.VV = np.zeros(( nm, 2, 2 ))
        self.VV[:,0,0] = self.asVector(self.majors)

        if np.size(self.minors) == np.size(self.majors):
            self.VV[:,1,1] = self.asVector(self.minors)
        else:
            self.VV[:,1,1] = self.VV[:,0,0]
            
    def populateRotationMatrix(self, rotateAxes=False):

        """Populates rotation matrix stack using rotations.

        rotateAxes = rotate the axes instead of the points?"""

        self.RR = np.array([])
        nR = np.size(self.rotDegs)
        nMaj = np.size(self.majors)
        if nR < 1:
            # If we DO have major array, use the identity matrix stack
            if nMaj > 0:
                i2 = np.eye(2, dtype='double')
                self.RR = np.repeat(i2[np.newaxis,:,:], nMaj, axis=0)
            return

        cc = np.cos(np.radians(self.rotDegs))
        ss = np.sin(np.radians(self.rotDegs))
        sgn = 1.
        if not rotateAxes:
            sgn = -1.

        self.RR = np.zeros(( nR, 2, 2 ))
        self.RR[:,0,0] = cc
        self.RR[:,1,1] = cc
        self.RR[:,0,1] = sgn * ss
        self.RR[:,1,0] = 0. - sgn * ss
        
    def populateTransformation(self):

        """Populates the transformation matrix by doing RR.VV
        plane-by-plane"""

        # 2023-06-21 this should be sqrt(self.VV) when interpreting
        # the covariance matrix as a geometric transformation of
        # vectors. Apply.
        
        self.TT = np.matmul(self.RR, self.VV**0.5)

    def asVector(self, x=np.array([])):

        """Returns a copy of the input object if a scalar, and a
        reference to the input if already an array"""

        if np.isscalar(x):
            return np.array([x])
        return x
    
    def populateCovarStack(self):

        """Populates the stack of covariance matrices"""

        self.covars = AVAt(self.RR, self.VV)

    def generateCovarStack(self):

        """Wrapper that generates a stack of transformation
        matrices"""

        self.genEigens()
        self.genRotns()
        self.covarFromABtheta()
        #self.populateDiagCovar()
        #self.populateRotationMatrix(rotateAxes=False)
        #self.populateTransformation()
        #self.populateCovarStack()
        #self.populateXYcomponents()

    def covarFromABtheta(self):

        """If the a, b, theta components have been populated, use them
        to populate the covariance stack"""

        if np.size(self.majors) < 1:
            return

        self.populateDiagCovar()
        self.populateRotationMatrix(rotateAxes=False)
        self.populateTransformation()
        self.populateCovarStack()
        self.populateXYcomponents()
        

    def populateTransfsFromCovar(self):

        """Wrapper - populates the transformation matrix from the
        covariance stack. Useful if we want to draw samples from a set
        of covariance stacks"""

        if np.size(self.covars) < 1:
            return

        # If we haven't already got the rotation angles from the
        # covariance matrices, get them!
        if np.size(self.rotDegs) < 1:
            self.eigensFromCovars()

        # Now we can populate the other pieces
        self.populateDiagCovar()
        self.populateRotationMatrix(rotateAxes=False)
        self.populateTransformation()

    def generateSamples(self):

        """Generate samples from the distributions"""

        # safety check: if transformation not yet populated, populate it!
        if np.size(self.TT) < 2:
            self.populateTransfsFromCovar()

        # Creates [nPts, 2] array

        # this whole thing is 2x2 so we'll do it piece by piece
        xr = np.random.normal(size=self.nPts)
        yr = np.random.normal(size=self.nPts)
        xxr = np.vstack(( xr, yr )).T[:,:,np.newaxis]

        self.deltaTransf = np.matmul(self.TT, xxr)[:,:,0].T
        #self.deltaTransf = np.dot(self.TT, xxr)

        # self.deltaTransf = np.einsum('ij,ik->ijk', self.TT, xxr)

    def getsamples(self):

        """Utility - returns samples using the N,2,2 covariance array, in the form [N,2]"""

        if np.size(self.TT) < 1:
            self.populateTransfsFromCovar()
        self.generateSamples()
        return self.deltaTransf.T

    def anyplanesbad(self, covs=np.array([]) ):

        """Utilty - returns True if any of the planes of the Nx2x2 covariance
stack is singular or has negative determinant. While this will take a
covarianc matrix as input, it will act on self.covs if no input is
supplied. The user / calling method is trusted to get the input
correct.

Inputs:

        covs = [(N), 2, 2] covariance matrix. If none supplied, uses
        self.covs

Returns:

        issingular = True if any of the planes of the covariance
        arrays are singular.

        """

        if np.size(covs) < 1:
            covs = self.covars
        
        # Return False if we don't actually have covariances...
        if np.size(covs) < 1:
            return False

        bbad = np.linalg.det(covs) <= 0.

        return np.sum(bbad) > 0
        
        
    def showDeltas(self, figNum=1):

        """Utility: scatterplots the deltas we have generated"""
        
        if np.size(self.deltaTransf) < 1:
            return

        dx = self.deltaTransf[0]
        dy = self.deltaTransf[1]

        fig = plt.figure(figNum, figsize=(7,6))
        fig.clf()
        ax1 = fig.add_subplot(111)

        dumScatt = ax1.scatter(dx, dy, s=1, c=self.rotDegs, \
                                   cmap='inferno', zorder=5)
        
        ax1.set_xlabel(r'$\Delta X$')
        ax1.set_ylabel(r'$\Delta Y$')

        # enforce uniform axes
        dm = np.max(np.abs(np.hstack(( dx, dy ))))
        ax1.set_xlim(-dm, dm)
        ax1.set_ylim(-dm, dm)

        ax1.set_title('Deltas before shifting')

        ax1.grid(which='both', visible=True, alpha=0.5, zorder=1)

        cDum = fig.colorbar(dumScatt)
