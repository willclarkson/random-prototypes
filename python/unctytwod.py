#
# unctytwod.py
#

#
# Methods to transform astrometric uncertainties between frames
#

import numpy as np
from covstack import CovStack

# for replicating instances
import copy

# for debug plotting
from matplotlib.pylab import plt
plt.ion()

class Polycoeffs(object):

    """Object and methods to translate between flat array of polynomial
coefficients and the 2D convention expected by numpy methods"""

    def __init__(self, p=np.array([]), Verbose=True):

        # flat coefficients
        self.p = p

        # Print warnings?
        self.Verbose = Verbose

        # coefficients arranged as 2D array
        self.p2d = np.array([])
        
        # Degree of the corresponding polynomial (could make this -1
        # to indicate inconsistency or unfilled)
        self.deg = -1

        # arrays of x indices, y indices
        self.i = np.array([])
        self.j = np.array([])

        # Populate on initialization
        self.assigndeg()
        self.assignij()

        self.initcoeffs2d()
        self.updatecoeffs2d()
        
    def degfromcoeffs(self, m=1):

        """Returns the degree given the number of coefficients"""

        d = (-3. + np.sqrt(9.0 + 8.*(m-1.) ))/2.

        return d

    def ijfromdeg(self, deg=2):

        """Returns the i, j indices for the coefficients given the degree"""

        # useful to have as a method that returns values
        
        iarr = np.array([], dtype='int')
        jarr = np.array([], dtype='int')

        for iterm in range(deg+1):
            count = np.arange(iterm+1, dtype='int')
            jarr = np.hstack(( jarr, count ))
            iarr = np.hstack(( iarr, count[::-1] ))

        return iarr, jarr

    def assigndeg(self):

        """Assigns the degree from the length of the coefficients"""

        degr = self.degfromcoeffs(np.size(self.p))

        if np.abs(degr - np.int(degr)) > 1.0e-3:
            if self.Verbose:
                print("Polycoeffs.assigndeg WARN - parameters do not correspond to a degree: %i" % (np.size(self.p)))
            self.deg = -1
            return

        self.deg = int(degr)

    def assignij(self):

        """Assigns i- and j-arrays using the degree of the polynomial"""

        self.i, self.j = self.ijfromdeg(self.deg)

    def initcoeffs2d(self):

        """Sets up empty 2d array of coefficients"""

        self.p2d = np.zeros(( self.deg+1, self.deg+1 ))

    def updatecoeffs2d(self):

        """Fills in the 2D coefficients array"""

        # do nothing if there is an inconsistency
        if self.deg < 1:
            if self.Verbose:
                print("Polycoeffs.updatecoeffs2d WARN - degree < 0. Check length of parameter-set")
            return
        
        l = np.arange(np.size(self.p))
        self.p2d[self.i[l],self.j[l]] = self.p[l]

    def getcoeffs2d(self, p=np.array([]), clobber=False):

        """Updates and returns the 2D coefficients for supplied parameters"""

        # If this instance has already had the degree and indices
        # arrays assigned, don't do them again.
        if np.size(p) > 0:
            self.p = p

        # (re-) assign the degree and indices arrays if not already
        # set, OR if input keyword "clobber" is set.
        if self.deg < 0 or clobber:
            self.assigndeg()
            self.assignij()
            
        self.updatecoeffs2d()
        return self.p2d
        
class Poly(object):

    """Methods to transform positions and uncertainties using numpy's
polynomial objects and methods. Should allow polynomials, legendre,
chebyshev and hermite depending on which of numpy's methods we
choose."""

    def __init__(self, x=np.array([]), y=np.array([]), covxy=np.array([]), \
                 parsx=np.array([]), parsy=np.array([]), degrees=True):

        # Inputs
        self.x = x
        self.y = y
        self.covxy = covxy
        self.parsx = parsx
        self.parsy = parsy

        # control variable (for scaling deltas)
        self.degrees = degrees

        # The jacobian, transformed coords, transformed covariances
        self.xtran = np.array([])
        self.ytran = np.array([])
        self.covtran = np.array([])

        
class Polynom(object):

    """Methods to transform positions and uncertainties via
polynomial. Note this is not invertible, so we only provide methods in
the one direction."""

    def __init__(self, posxy=np.array([]), covxy=np.array([]), \
                 parsx=np.array([]), parsy=np.array([]), \
                 degrees=True):

        self.x = posxy[:,0]
        self.y = posxy[:,1]

        self.covxy = covxy

        # transformation parameters for x, y coords
        self.parsx = parsx
        self.parsy = parsy

        # Jacobian for the transformation
        self.jac = np.array([])

        # Transformed coordinates, covariances
        self.xtran = np.array([])
        self.ytran = np.array([])
        self.covtran = np.array([])

        # control variable - original coords are in degrees?
        self.degrees = degrees

        # Labels for the transformed quantities
        self.labelxtran = r'$X$'
        self.labelytran = r'$Y$'

    def polyval2d(self, pars=np.array([])):

        """Evaluates the polynomial for the instance-level coordinates"""

        # This is written out by hand for readability and for ease
        # translating to the jacobians later on. Might be better to
        # use rules to construct all the powers, but that requires
        # cleverer list comprehension chops than I currently have.
        
        # This really serves to establish our convention for the
        # coefficients throughout this object.
        z = self.x * 0. + pars[0] # inherit the shape of x
        if np.size(pars) < 2: # want this to work on scalar pars
            return z

        # add linear terms...
        z += self.x * pars[1] + self.y * pars[2]
        if np.size(pars) < 6:
            return z

        # second-order...
        z += self.x**2 * pars[3] + self.x*self.y*pars[4] + self.y**2 * pars[5]
        if np.size(pars) < 10:
            return z

        # third order...
        z += self.x**3 * pars[6] + \
            self.x**2 * self.y    * pars[7] + \
            self.x    * self.y**2 * pars[8] + \
            self.y**3 * pars[9]
        if np.size(pars) < 15:
            return z
        
        # fourth order...
        z += self.x**4            * pars[10] + \
            self.x**3 * self.y    * pars[11] + \
            self.x**2 * self.y**2 * pars[12] + \
            self.x    * self.y**3 * pars[13] + \
                        self.y**4 * pars[14]
        if np.size(pars) < 21:
            return z
        
        # fifth-order
        z += self.x**5 * pars[15] + \
            self.x**4 * self.y    * pars[16] + \
            self.x**3 * self.y**2 * pars[17] + \
            self.x**2 * self.y**3 * pars[18] + \
            self.x * self.y**4 * pars[19] + \
            self.y**5 * pars[20]

        return z
        
        # ... beyond fifth order, consider iterations (would need
        # something clever for the covariances)

    def jac2d(self, pars=np.array([]) ):

        """Returns the Jacobian terms dz/dx, dz/dy when z=polyval2d(pars,
x,y)). Coordinates are taken from the instance, pars are passed in as
arguments
        """

        #dz/dx, dz/dy
        zx = self.x * 0.
        zy = self.y * 0.

        if np.size(pars) < 2:
            return zx, zy

        # first order
        zx += self.x * 0. + pars[1]
        zy += self.y * 0. + pars[2]
        if np.size(pars) < 6:
            return zx, zy

        # second order - now the coordinates finally get involved
        zx += 2.0*self.x * pars[3] + self.y*pars[4]
        zy += 2.0*self.y * pars[5] + self.x*pars[4]
        if np.size(pars) < 10:
            return zx, zy

        # third order
        zx += \
            3.0*self.x**2 * pars[6] + \
            2.0*self.x*self.y*pars[7] + \
            self.y**2 * pars[8]

        zy += \
            self.x**2 * pars[7] + \
            2.0*self.y * self.x * pars[8] + \
            3.0*self.y**2 * pars[9]
        if np.size(pars) < 15:
            return zx, zy

        # fourth order
        zx += \
            4.0*self.x**3 * pars[10] + \
            3.0*self.x**2 * self.y * pars[11] + \
            2.0*self.x * self.y**2 * pars[12] + \
            self.y**3 * pars[13]

        zy += \
            self.x**3 * pars[11] + \
            2.0 * self.y * self.x**2 * pars[12] + \
            3.0 * self.y**2 * self.x * pars[13] + \
            4.0 * self.y**3 * pars[14]
        if np.size(pars) < 21:
            return zx, zy

        # fifth-order
        zx += \
            5.0*self.x**4 * pars[15] + \
            4.0*self.x**3 * self.y * pars[16] + \
            3.0*self.x**2 * self.y**2 * pars[17] + \
            2.0*self.x    * self.y**3 * pars[18] +\
            self.y**4 * pars[19]

        zy += \
            self.x**4 * pars[16] + \
            2.0*self.y    * self.x**3 * pars[17] + \
            3.0*self.y**2 * self.x**2 * pars[18] + \
            4.0*self.y**3 * self.x    * pars[19] + \
            5.0*self.y**4 * pars[20]

        # fifth-order is probably enough for now!
        return zx, zy

    def tranpos(self):

        """Transforms the positions by the polyomials"""

        self.xtran = self.polyval2d(self.parsx)
        self.ytran = self.polyval2d(self.parsy)

    def getjacobian(self):

        """Populates the jacobian associated with the polynomial
transformations"""

        self.jac = np.zeros(( np.size(self.x), 2, 2 ))

        jxix, jxiy = self.jac2d(self.parsx)
        jetax, jetay = self.jac2d(self.parsy)

        self.jac[:,0,0] = jxix
        self.jac[:,0,1] = jxiy
        self.jac[:,1,0] = jetax
        self.jac[:,1,1] = jetay

    def trancov(self):

        """Transforms the covariance via the jacobian"""

        if np.size(self.jac) < 1:
            self.getjacobian()

        J = self.jac
        Jt = np.transpose(J, axes=(0,2,1))
        C = self.covxy

        self.covtran = np.matmul(J, np.matmul(C, Jt) )

    def propagate(self):

        """One-liner to propagate positions and covariances"""

        self.tranpos()
        self.getjacobian()
        self.trancov()

    def nudgepos(self, dxarcsec=10., dyarcsec=10.):

        """Nudges the input positions by input amounts"""

        conv = 206265.
        if self.degrees:
            conv = 3600.

        self.x += dxarcsec / conv
        self.y += dyarcsec / conv
            
    def calcdeltas(self, dxarcsec=10., dyarcsec=10.):

        """Estimates deltas in the projected frame from Jacobian.dx"""

        if np.size(self.jac) < 4:
            return np.array([])

        # Produce delta-array in the same unit as the original
        # coordinates, assuming input in arcsec
        conv = 206265.
        if self.degrees:
            conv = 3600.
            
        dx = np.array([dxarcsec, dyarcsec])/conv
        dv = np.matmul(self.jac, dx)
        
        return dv
        
class Sky(object):

    def __init__(self, possky=np.array([]), covsky=np.array([]), \
                 tpoint=np.array([]), \
                 postan=np.array([]), covtan=np.array([]) ):

        # positions, covariances on the sky
        self.possky = possky  # Nx2
        self.covsky = covsky # Nx2x2

        # tangent point, degrees
        self.tpoint=tpoint  # [2]

        # positions, covariances on the tangent plane
        self.postan = postan
        self.covtan = covtan

        # Jacobians for transforming uncertainties
        self.j2sky = np.array([])    # dalpha/dxi, etc.
        self.j2tan = np.array([])    # dxi/dalpha, etc.

        # Labels for transformed positions (when plotting)
        self.labelxtran = r'$\alpha$'
        self.labelytran = r'$\delta$'
        
    def sky2tan(self):

        """Converts sky coordinates to tangent plane coordinates. Input and output all in DEGREES."""

        # Unpack everything for readability later
        alpha0 = np.radians(self.tpoint[0])
        delta0 = np.radians(self.tpoint[1])

        alpha = np.radians(self.possky[:,0])
        delta = np.radians(self.possky[:,1])

        denom = np.cos(alpha-alpha0) * np.cos(delta)*np.cos(delta0) \
            + np.sin(delta)*np.sin(delta0)

        xi = np.cos(delta)*np.sin(alpha-alpha0) / denom
        
        eta = (np.cos(delta0)*np.sin(delta) - \
            np.cos(alpha-alpha0)*np.sin(delta)*np.sin(delta0)) / denom

        self.postan = self.possky*0.
        self.postan[:,0] = np.degrees(xi)
        self.postan[:,1] = np.degrees(eta)
        
    def tan2sky(self):

        """Converts tangent plane to sky coordinates. Input output all in DEGREES."""

        # Again, unpack everything for readability
        xi  = np.radians(self.postan[:,0])
        eta = np.radians(self.postan[:,1])

        alpha0 = np.radians(self.tpoint[0])
        delta0 = np.radians(self.tpoint[1])

        gamma = np.cos(delta0) - eta*np.sin(delta0)
        alphaf = alpha0 + np.arctan(xi/gamma)
        deltaf = np.arctan(
            (eta*np.cos(delta0) + np.sin(delta0) ) /
            np.sqrt(xi**2 + gamma**2) )

        # populate the instance
        self.possky = self.postan*0.
        self.possky[:,0] = np.degrees(alphaf)
        self.possky[:,1] = np.degrees(deltaf)

    def jac2sky(self):

        """Populates the Nx2x2 jacobian d(alpha, delta)/d(xi, eta) , which is
stored in object self.j2sky"""
        
        # unpack for readability
        xi  = np.radians(self.postan[:,0])
        eta = np.radians(self.postan[:,1])

        alpha0 = np.radians(self.tpoint[0])
        delta0 = np.radians(self.tpoint[1])

        # dalpha / dxi
        denom00 = (np.cos(delta0) - eta*np.sin(delta0)) * \
            (1.0 + (xi/(np.cos(delta0) - eta*np.sin(delta0) ))**2 )
        J_ax = 1.0/denom00

        # dalpha / deta
        denom01 = xi**2 + (np.cos(delta0))**2 \
            - 2.0 * eta * np.cos(delta0) * np.sin(delta0) \
            + eta**2 * (np.sin(delta0))**2
        J_ay = xi*np.sin(delta0) / denom01

        # ddelta / dxi
        denom10 = (1.0 + xi**2 + eta**2) \
            * np.sqrt(xi**2 + (np.cos(delta0) - eta * np.sin(delta0) )**2 )

        J_dx = 0. - xi*( eta*np.cos(delta0) + np.sin(delta0)) / denom10

        # ddelta / deta
        J_dy = ((1.0 + xi**2)*np.cos(delta0) - eta*np.sin(delta0)) / denom10

        # Populate the stack
        self.j2sky = np.zeros(( np.size(J_dx), 2, 2 ))
        self.j2sky[:,0,0] = J_ax
        self.j2sky[:,0,1] = J_ay
        self.j2sky[:,1,0] = J_dx
        self.j2sky[:,1,1] = J_dy
        
    def jac2tan(self):

        """Populates the Nx2x2 jacobian d(xi,eta)/d(alpha, delta), which is
stored in object self.j2tan"""

        # unpack for readability
        alpha = np.radians(self.possky[:,0])
        delta = np.radians(self.possky[:,1])

        alpha0 = np.radians(self.tpoint[0])
        delta0 = np.radians(self.tpoint[1])

        # We have the same denominator for all four terms
        denom = ( np.cos(alpha-alpha0) * np.cos(delta) * np.cos(delta0) \
            + np.sin(delta)*np.sin(delta0) )**2

        # dxi/dalpha
        J_xia = np.cos(delta) * ( np.cos(delta) * np.cos(delta0) \
            + np.cos(alpha-alpha0) * np.sin(delta)*np.sin(delta0)) / denom

        # dxi/ddelta
        J_xid = 0. -np.sin(alpha-alpha0) * np.sin(delta0) / denom

        # deta/dalpha
        J_etaa = 0.5 * np.sin(alpha-alpha0) * np.sin(2.0*delta) / denom

        # deta/ddelta
        J_etad = np.cos(alpha-alpha0) / denom

        # populate the stack
        self.j2tan = np.zeros(( np.size(J_xia),2,2 ))
        self.j2tan[:,0,0] = J_xia
        self.j2tan[:,0,1] = J_xid
        self.j2tan[:,1,0] = J_etaa
        self.j2tan[:,1,1] = J_etad

    def cov2sky(self):

        """Propagates the covariance matrices from tangent plane to sky"""

        J = self.j2sky
        Jt = np.transpose(J, axes=(0,2,1) )
        C = self.covtan

        JCJt = np.matmul(J, np.matmul(C, Jt) )
        self.covsky = JCJt

    def cov2tan(self):

        """Propagates the covariance matrices from sky to tangent plane"""

        J = self.j2tan
        Jt = np.transpose(J, axes=(0,2,1) )
        C = self.covsky

        JCJt = np.matmul(J, np.matmul(C, Jt) )
        self.covtan = JCJt

    def propag2sky(self, alpha0deg, delta0deg, \
                   postan=np.array([]), covtan=np.array([]), \
                   retvals=False):

        """One-liner to propagate tangent plane coordinates and covariances onto the sky, given input pointing. If retvals is True, the transformed positions and covariances are returned. Otherwise they are just updated in the instance."""

        # update the tangent point in radians
        self.tpoint=np.array([alpha0deg, delta0deg])

        # update the coords and covariances if they were supplied here
        # and if their lengths match
        if np.size(postan) > 0:
            self.postan = np.copy(postan)

        if np.abs(np.shape(covtan)[0] - np.shape(self.postan)[0]) < 1:
            self.covtan = np.copy(covtan)
        
        # Propagate the positions
        self.tan2sky()
        
        # Propagate the uncertainties
        self.jac2sky()
        self.cov2sky()

        if retvals:
            return self.possky, self.covsky

    def propag2tan(self, alpha0deg, delta0deg, \
                   possky=np.array([]), covsky=np.array([]), \
                   retvals=False):

        """One-liner to propagate equatorial coordinates and uncertainties onto the tangent plane. If retvals is True, the transformed positions and covariances are returned. Otherwise they are just updated in the instance."""

        # update the tangent point in radians
        self.tpoint=np.array([alpha0deg, delta0deg])

        # update the coords and covariances if they were supplied here
        # and if their lengths match
        if np.size(possky) > 0:
            self.possky = np.copy(possky)

        if np.abs(np.shape(covsky)[0] - np.shape(self.possky)[0]) < 1:
            self.covsky = np.copy(covsky)

        # Propagate the positions
        self.sky2tan()
        
        # Propagate the uncertainties
        self.jac2tan()
        self.cov2tan()

        if retvals:
            return self.postan, self.covtan

    def nudgepos(self, dxarcsec=10., dyarcsec=10.):

        """Nudges the tangent plane positions by input offsets"""

        self.postan[:,0] += dxarcsec / 3600.
        self.postan[:,1] += dyarcsec / 3600.

    def calcdeltas(self, dxarcsec=10., dyarcsec=10.):

        """Estimates deltas on the sky from the jacobian.dxi"""

        if np.size(self.j2sky) < 4:
            return

        # For this instance the Jacobian expects everything in
        # radians.
        dxi = np.array([dxarcsec, dyarcsec])/206265.
        dv = np.matmul(self.j2sky, dxi)

        # Converts back to degrees, since the sky coords are in degrees
        return np.degrees(dv)

    def tranpos(self):

        """Transforms tangent plane positions onto the sky using the same
naming convention as the Polynom() object"""

        self.tan2sky()
        self.xtran = self.possky[:,0]
        self.ytran = self.possky[:,1]
        
# utility - return a grid of xi, eta points
def gridxieta(sidelen=2.1, ncoarse=11, nfine=41):

    """Returns a grid of points in xi, eta"""
    
    xv = np.linspace(0.-sidelen, sidelen, ncoarse, endpoint=True)
    yv = np.linspace(0.-sidelen, sidelen, nfine, endpoint=True)
    xx, yy = np.meshgrid(xv, yv)
    xi = np.ravel(xx)
    eta = np.ravel(yy)

    xi = np.hstack(( xi, np.ravel(yy) ))
    eta = np.hstack(( eta, np.ravel(xx) ))

    return xi, eta
    
####### Methods that use the above follow

def checkdeltas(transf=None, dxarcsec=10., dyarcsec=10., showPlots=True, \
                cmap='viridis', symm=False, showpct=True):

    """Given a transformation object, checks the differences between the brute-force deltas and the Jacobian-obtained deltas"""

    if transf is None:
        return

    dv = transf.calcdeltas(dxarcsec, dyarcsec)

    # Now compute the deltas directly
    nudged = copy.deepcopy(transf)
    nudged.nudgepos(dxarcsec, dyarcsec)
    nudged.tranpos()

    # Hack to ensure the transformation object has the xtran, ytran
    # coordinates we expect here
    if not hasattr(transf, 'xtran'):
        transf.xtran = transf.possky[:,0]
        transf.ytran = transf.possky[:,1]
        transf.x = transf.postan[:,0]
        transf.y = transf.postan[:,1]
        detj = np.linalg.det(transf.j2sky)
    else:
        detj = np.linalg.det(transf.jac)
        
    dxbrute = nudged.xtran - transf.xtran
    dybrute = nudged.ytran - transf.ytran
    dvbrute = np.vstack(( dxbrute, dybrute )).T

    # Create delta of deltas array
    ddv = dv - dvbrute
    dmag = np.sqrt(dvbrute[:,0]**2 + dvbrute[:,1]**2)

    # views - our figure of merit
    sx = ddv[:,0]/dmag
    sy = ddv[:,1]/dmag
    
    if not showPlots:
        return

    # Are we showing as percent?
    if showpct:
        sx *= 100.
        sy *= 100.
    
    # symmetric limits for colorbars
    if symm:
        vminx = 0.-np.max(np.abs(sx))
        vmaxx = 0.+np.max(np.abs(sx))
        vminy = 0.-np.max(np.abs(sy))
        vmaxy = 0.+np.max(np.abs(sy))
    else:
        vminx = None
        vmaxx = None
        vminy = None
        vmaxy = None
        

    fig2=plt.figure(2)
    fig2.clf()
    ax1=fig2.add_subplot(223)
    ax2=fig2.add_subplot(224)
    ax0=fig2.add_subplot(221)

    # raw offsets?
    ax4=fig2.add_subplot(222)

    # Show the original positions. One more feature: if we are dealing
    # with outoput in equatorial coordinates, convert the magnitude
    # displayed here to arcsec and adjust the reporting accordingly
    magconv = 1.
    if hasattr(transf,'tpoint'):
        magconv = 3600.
    blah0=ax0.scatter(transf.x, transf.y, c=dmag*magconv, \
                      cmap=cmap, s=1)
    
    blah1=ax1.scatter(transf.xtran, transf.ytran, c=sx, \
                      cmap=cmap, s=1, \
                      vmin=vminx, vmax=vmaxx)

    blah2=ax2.scatter(transf.xtran, transf.ytran, c=sy, \
                      cmap=cmap, s=1, \
                      vmin=vminy, vmax=vmaxy)

    blah41 = ax4.scatter(transf.xtran, transf.ytran, s=1, \
                         c=detj,\
                         zorder=10)
    #blah42 = ax4.scatter(nudged.xtran, nudged.ytran, s=1, \
    #                     c='k', \
    #                     alpha=0.5, zorder=5)

    # colorbars
    cb0 = fig2.colorbar(blah0, ax=ax0)
    cb1 = fig2.colorbar(blah1, ax=ax1)
    cb2 = fig2.colorbar(blah2, ax=ax2)
    cb4 = fig2.colorbar(blah41, ax=ax4)

    
    ax0.set_xlabel(r'$\xi$, degrees')
    ax0.set_ylabel(r'$\eta$, degrees')

    # Some plot label carpentry
    labelx = r'$X$'
    labely = r'$Y$'
    if hasattr(transf, 'labelxtran'):
        labelx = transf.labelxtran
    if hasattr(transf, 'labelytran'):
        labely = transf.labelytran

    # For concatenation within latex strings
    labelxr = labelx.replace('$','')
    labelyr = labely.replace('$','')

    for ax in [ax1, ax2, ax4]:
        ax.set_xlabel(labelx)
        ax.set_ylabel(labely)

    # titles
    ax0.set_title(r"$|d\vec{%s}|$" % (labelxr) )
    if magconv > 1:
        ax0.set_title(r"$|d\vec{%s}|$, arcsec" % (labelxr) )
        
    ax1.set_title(r"$(d%s - d%s_{\rm J}) / |d\vec{%s}|$" \
                  % (labelxr, labelxr, labelxr))
    ax2.set_title(r"$(d%s - d%s_{\rm J}) / |d\vec{%s}|$" \
                  % (labelyr, labelyr, labelyr)) 

    if showpct:
        ax1.set_title(r"$100\times (d%s - d%s_{\rm J}) / |d\vec{%s}|$" \
                      % (labelxr, labelxr, labelxr))
        ax2.set_title(r"$100\times (d%s - d%s_{\rm J}) / |d\vec{%s}|$" \
                      % (labelyr, labelyr, labelxr))

    ax4.set_title('det(J)')
        
    # Show the input nudge
    ssup = r"$(\Delta \xi, \Delta\eta) = (%.1f, %.1f)$ arcsec" \
        % (dxarcsec, dyarcsec)

    # If the transformation object has a tangent point, show this too
    if hasattr(transf,'tpoint'):
        ssup = r"$(\Delta \xi, \Delta\eta) = (%.1f'', %.1f'')$, $(\alpha_0, \delta_0) = (%.1f, %.1f)$" %  (dxarcsec, dyarcsec, transf.tpoint[0], transf.tpoint[1])
    
    fig2.suptitle(ssup)
    fig2.subplots_adjust(hspace=0.5, wspace=0.5, top=0.85)
    
def testTransf(nobjs=5000, alpha0=35., delta0=35., sidelen=2.1, \
               showplots=True, \
               sigx=1.0, sigy=0.7, sigr=0.2, \
               usegrid=True, \
               dxarcsec=10., dyarcsec=10., showpct=True):


    # Example call:
    #
    # unctytwod.testTransf(usegrid=True, sidelen=2., dxarcsec=10., dyarcsec=0., delta0=57.9, showpct=False)

    
    # Construct a random set of xi, eta points for our
    # transformations. Use a square detector for convenience
    # halfrad = np.radians(sidelen)
    xieta = np.random.uniform(0.-sidelen, sidelen, (nobjs,2)) 

    if usegrid:
        xi, eta = gridxieta(sidelen, 11, 41)
        xieta = np.vstack((xi, eta)).T
        nobjs = np.size(xi)
        
    # construct our coordinate object
    SS = Sky(postan=xieta, tpoint=np.array([alpha0, delta0]) )

    # generate some covariances in the tangent plane. For testing,
    # default to uniform so that we can see how the transformation
    # impacts the covariances
    vstdxi = np.ones(nobjs)*sigx
    vstdeta = vstdxi * sigy/sigx
    vcorrel = np.ones(nobjs)*sigr
    CS = CovStack(vstdxi, vstdeta, r12=vcorrel, runOnInit=True)

    # pass the covariances arrays to the uncty2d object
    SS.covtan = np.copy(CS.covars)
    
    # convert tp to sky
    SS.tan2sky()

    # populate the jacobians
    SS.jac2sky()
    SS.jac2tan()

    # Now convert the covariance matrices from the tangent plane to the sky
    SS.cov2sky()

    # By this point we should have the Jacobian to the sky
    # populated. Run our checker to see how the deltas compare to each
    # other.
    checkdeltas(SS, dxarcsec, dyarcsec, showpct=showpct)
    
    ### Check whether the jacobians really are the inverses of each
    ### other...
    Jsky = SS.j2sky
    Jinv = np.linalg.inv(SS.j2tan)

    print("Inversion check - sky vs inv(tan):")
    print(Jsky[0])
    print(Jinv[0])

    Jtan = SS.j2tan
    Jsin = np.linalg.inv(SS.j2sky)
    
    print("Inversion check - tan vs inv(sky):")
    print(Jtan[0])
    print(Jsin[0])

    print("============")
    
    print("Covariances on the sky:", SS.covsky.shape)

    # Try converting back again... do we get the same as the input?
    SS.cov2tan()
    
    print("INFO: input row 0:", CS.covars[0])
    print("INFO: conv row 0:", SS.covsky[0])
    print("INFO: back row 0:", SS.covtan[0])


    ### Now try the one-liner
    TT = Sky()
    TT.propag2sky(alpha0, delta0, xieta, CS.covars)

    print("One-liner check:")
    print(SS.covsky[0])
    print(TT.covsky[0])
    print("============")

    ### Try the one-liner in the other direction
    RR = Sky()
    RR.propag2tan(alpha0, delta0, SS.possky, SS.covsky)

    print("One-liner check, other direction:")
    print(SS.covtan[0])
    print(RR.covtan[0])
    print("============")

    
    # compute the determinants
    det2sky = np.linalg.det(SS.j2sky)
    det2tan = np.linalg.det(SS.j2tan)

    # so that we can conveniently divide out the cos(delta) when
    # plotting
    cosdec = np.cos(np.radians( SS.possky[:,1] ))

    print("testTransf INFO -- covtan shape:",SS.covtan.shape)
    print(SS.covtan[0])
    print(SS.j2sky.shape)
    print(SS.j2tan.shape)
    print(SS.j2sky[0])
    
    if not showplots:
        return
    
    fig1 = plt.figure(1)
    fig1.clf()
    ax1 = fig1.add_subplot(221)
    ax2 = fig1.add_subplot(222)

    blah1 = ax1.scatter(SS.postan[:,0], SS.postan[:,1], s=2, \
#                        c=SS.j2sky[:,1,1] )
                        c = det2sky * cosdec ) 
    blah2 = ax2.scatter(SS.possky[:,0], SS.possky[:,1], s=2, \
#                        c=SS.j2tan[:,1,1] )
                        c = det2tan / cosdec )
    cb1 = fig1.colorbar(blah1, ax=ax1)
    cb2 = fig1.colorbar(blah2, ax=ax2)

    ax1.set_xlabel(r'$\xi$, degrees')
    ax1.set_ylabel(r'$\eta$, degrees')
    ax2.set_xlabel(r'$\alpha$, degrees')
    ax2.set_ylabel(r'$\delta$, degrees')

    ax1.set_title(r'$\left|\frac{\partial(\alpha,\delta)}{\partial(\xi, \eta)}\right|\cos(\delta)$')
    ax2.set_title(r'$|\frac{\partial(\xi,\eta)}{\partial(\alpha, \delta)}|/\cos(\delta)$')
    
    fig1.subplots_adjust(hspace=0.4, wspace=0.4)


def testpoly(sidelen=2.1, ncoarse=15, nfine=51, \
             showplots=True, \
             sigx=1.0, sigy=0.7, sigr=0.2, \
             symm=False, cmap='viridis', \
             dxarcsec=10., dyarcsec=10.):

    """Test the propagation through a polynomial"""

    # Example call:
    #
    # unctytwod.testpoly(dxarcsec=10., dyarcsec=-10.)
    
    # Create the grid of points
    xi, eta = gridxieta(sidelen, ncoarse, nfine)

    # concatenate these into the N,2 array we expect
    xieta = np.vstack((xi, eta)).T
    
    # transformation parameters. While testing I'll just write out
    # some examples here. Consider making this more systematic later
    # on...
    parsx = [ 10., 10., 2.]
    parsy = [-5., -1., 9.]

    # add some curvature via quadratic
    parsx = parsx + [1.5, 0.2, 0.1]
    parsy = parsy + [0.7, 0.05, -0.4]

    # Now make this really curved...
    parsx = parsx +  [0.1, 0.2, 0.3, 0.4]
    parsy = parsy +  [-0.4, -0.3, -0.2, -0.1]
    
    # Covariances in the original frame
    vstdxi = np.ones(np.size(xi))*sigx
    vstdeta = vstdxi * sigy/sigx
    vcorrel = np.ones(np.size(xi))*sigr
    CS = CovStack(vstdxi, vstdeta, r12=vcorrel, runOnInit=True)
    
    # Create the instance and use it
    PP = Polynom(xieta, CS.covars, parsx, parsy)
    PP.propagate()

    # try our deltas-checker
    checkdeltas(PP, dxarcsec, dyarcsec, cmap=cmap, symm=symm)
    
    if not showplots:
        return
    
    fig1 = plt.figure(1)
    fig1.clf()
    ax1 = fig1.add_subplot(221)
    ax2 = fig1.add_subplot(222)

    # plot the original coordinates
    blah1 = ax1.scatter(xi, eta, s=1)
    ax1.set_xlabel(r'$\xi$')
    ax1.set_ylabel(r'$\eta$')
    ax1.set_title('raw')
    
    # plot the transformed coordinates
    blah2 = ax2.scatter(PP.xtran, PP.ytran, s=1)
    ax2.set_xlabel(r'$X$')
    ax2.set_ylabel(r'$Y$')
    ax2.set_title('transformed')

    # did that produce sensible output?
    print(PP.covxy[0])
    print(PP.covtran[0])

    # because np.matmul works plane-by-plane, the test of how delta-x
    # compares with jac x delta xi is pretty easy to do once you have
    # the jacobian in place. Return to this tomorrow.


def testpolycoefs(nterms=10, Verbose=True):

    """Tests the polycoeffs functionality"""

    p = np.arange(nterms)
    PC = Polycoeffs(p, Verbose=Verbose)
    print("Input params:", p)
    print("Degree:", PC.deg)
    print("i-indices:", PC.i)
    print("j-indices:", PC.j)
    print("2D coeffs array:")
    print(PC.p2d) # gets a separate line for nice printing

    # try the one-liner
    dum = PC.getcoeffs2d(p+1)

    print("One-liner call-return with p + 1 as input:")
    print(dum)
