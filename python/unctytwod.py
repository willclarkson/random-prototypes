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

        
####### Methods that use the above follow

def testTransf(nobjs=5000, alpha0=35., delta0=35., sidelen=2.1, \
               showplots=True, \
               sigx=1.0, sigy=0.7, sigr=0.2):

    # Construct a random set of xi, eta points for our
    # transformations. Use a square detector for convenience
    # halfrad = np.radians(sidelen)
    xieta = np.random.uniform(0.-sidelen, sidelen, (nobjs,2)) 
    
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
