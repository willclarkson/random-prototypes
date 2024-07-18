#
# unctytwod.py
#

#
# Methods to transform astrometric uncertainties between frames
#

import numpy as np
from covstack import CovStack

# for debug plotting
from matplotlib.pylab import plt
plt.ion()

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
        self.postan[:,1] = np.degreees(eta)
        
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
        denom01 = xi**2 + np.cos(delta0)**2 \
            - 2.0 * eta * np.cos(delta0) * np.sin(delta0) \
            + eta**2 * np.sin(delta0)**2
        J_ay = xi*np.sin(delta0) / denom01

        # ddelta / dxi
        denom10 = (1.0 + xi**2 + eta**2) \
            * np.sqrt(xi**2 + (np.cos(delta0) - eta * np.sin(delta0) )**2 )

        J_dx = 0. - xi*( eta*np.cos(delta0) + np.sin(delta0)) / denom10

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
        denom = ( np.cos(alpha-alpha0)*np.cos(delta)*np.cos(delta0) \
            + np.sin(delta)*np.sin(delta0) )**2

        # dxi/dalpha
        J_xia = ( np.cos(delta) * np.cos(delta) * np.cos(delta0) \
            + np.cos(alpha-alpha0) * np.sin(delta)*np.sin(delta0)) / denom

        # dxi/ddelta
        J_xid = 0. -np.sin(alpha-alpha0) * np.sin(2.0*delta) / denom

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

####### Methods that use the above follow

def testTransf(nobjs=5000, alpha0=35., delta0=35., sidelen=2.1):

    # Construct a random set of xi, eta points for our
    # transformations. Use a square detector for convenience
    # halfrad = np.radians(sidelen)
    xieta = np.random.uniform(0.-sidelen, sidelen, (nobjs,2)) 
    
    # construct our coordinate object
    SS = Sky(postan=xieta, tpoint=np.array([alpha0, delta0]) )

    # convert tp to sky
    SS.tan2sky()

    # populate the jacobians
    SS.jac2sky()
    SS.jac2tan()

    # compute the determinants
    det2sky = np.linalg.det(SS.j2sky)
    det2tan = np.linalg.det(SS.j2tan)

    # so that we can conveniently divide out the cos(delta) when
    # plotting
    cosdec = np.cos(np.radians( SS.possky[:,1] ))
    
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
