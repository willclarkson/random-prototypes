#
# deltapointing.py
#

# Toy field of view to evaluate the change in focal plane coordinates
# caused by a small shift in tangent point

import numpy as np
import matplotlib.pylab as plt

def buildfov(arcminx=10., arcminy=10., nx=101, ny=101, posdeg=0.):

    """Builds a grid of focal plane coordinates in radians"""

    vx = np.linspace(-arcminx/2., arcminx/2., nx)
    vy = np.linspace(-arcminy/2., arcminy/2., nx)

    yg, xg = np.meshgrid(vx, vy)

    xg = np.radians(xg/60.)
    yg = np.radians(yg/60.)

    return xg, yg

def rotatefov(x,y,posdeg=0):

    """Rotates the focal plane by posdeg degrees"""

    posrad = np.radians(posdeg)
    cc = np.cos(posrad)
    ss = np.sin(posrad)

    xr = x * cc - y*ss
    yr = x * ss + y*cc

    return xr, yr
    
def fov2radec(x,y, ra0deg=0., dec0deg=0.):

    """Converts focal plane coordinates to ra, dec specified in degrees"""

    alpha0 = np.radians(ra0deg)
    delta0 = np.radians(dec0deg)

    gamma = np.cos(delta0) - y*np.sin(delta0)

    # RA is straightforward...
    alpha = alpha0 + np.arctan2(x, gamma)

    # ... dec is a bit more involved
    numer = np.sin(delta0) + y*np.cos(delta0)
    denom = np.sqrt(x**2 + gamma**2)

    delta = np.arctan2(numer , denom)

    return alpha, delta

def radec2fov(ra, dec, alpha0deg, delta0deg):

    """Converts ra, dec to focal plane x, y. RA, DEC in radians,
alpha0deg, dec0deg in degrees."""

    dec0rad = np.radians(delta0deg)
    ra0rad = np.radians(alpha0deg)

    dra = ra - ra0rad
    ddec = dec - dec0rad

    denom = np.cos(ddec) - np.cos(dec0rad)*np.cos(dec)*(1.0-np.cos(dra))

    xrad = np.cos(dec)*np.sin(dra) / denom
    yrad = (np.sin(ddec)+np.sin(dec0rad)*np.cos(dec)*(1.0-np.cos(dra))) / denom

    return xrad, yrad
    
    
def jacobianpointing(alpha, delta, alpha0deg=0., delta0deg=0.):

    """Computes the terms in the jacobian delta(x,y)/delta(alpha0,
delta0). All the inputs are in radians. Returns the four components of the jacobian."""

    alpha0 = np.radians(alpha0deg)
    delta0 = np.radians(delta0deg)
    
    dra = alpha - alpha0
    dde = delta - delta0

    gamma_xi = np.cos(dde) - np.cos(delta)*np.cos(delta0)*(1.-np.cos(dra))

    J_x_a0 = 0. -(np.cos(delta)/gamma_xi**2) * \
        (np.cos(dde) - np.sin(delta)*np.sin(delta0)*(1.0-np.cos(dra)) )

    J_x_d0 = (np.cos(delta)/gamma_xi**2) * \
        (np.sin(dde) + np.sin(delta0)*np.cos(delta)*(1.-np.cos(dra)) ) \
        * np.sin(dra)

    J_y_a0 = -(0.5 * np.sin(2.*delta)/gamma_xi**2) * \
        np.sin(dra)

    J_y_d0 = -(1.0/gamma_xi**2) * \
        (1.-np.cos(delta)**2 * (1.-np.cos(dra)**2) )

    return J_x_a0, J_x_d0, J_y_a0, J_y_d0

 ####### Test routines follow ########


def testfov(arcminx=10., arcminy=10., alpha0deg = 0., delta0deg=0., \
            damas=1000., ddmas=1000., posdeg=0., xoffax=0., yoffax=0.):

     """End-to-end test"""

     xg, yg = buildfov(arcminx, arcminy)
     xr, yr = rotatefov(xg, yg, posdeg)

     # Now allow an off-axis offset in arcmin
     dxax = np.radians(xoffax/60.)
     dyax = np.radians(yoffax/60.)
     
     ra, dec = fov2radec(xr+dxax, yr+dyax, alpha0deg, delta0deg)

     # Compute the reprojected positions directly.
     cosdec0 = np.cos(np.radians(delta0deg))

     alpha1deg = alpha0deg + damas/(3.6e6 * cosdec0)
     delta1deg = delta0deg + ddmas/3.6e6

     # Now we undo the transformations one by one
     xf1, yf1 = radec2fov(ra, dec, alpha1deg, delta1deg)
     xr1 = xf1 - dxax
     yr1 = yf1 - dyax

     print("POSDEG", posdeg)
     
     x1, y1 = rotatefov(xr1, yr1, -posdeg)
     
     # Compute the jacobian
     J_xa, J_xd, J_ya, J_yd = jacobianpointing(ra, dec, alpha0deg, delta0deg)

     # Convert the focal plane coordinates to arcmin for plotting
     xgarcmin = np.degrees(xg)*60.
     ygarcmin = np.degrees(yg)*60.
     
     fig = plt.figure(1, figsize=(8,7))
     fig.clf()

     ax1 = fig.add_subplot(2,2,1)
     ax2 = fig.add_subplot(2,2,2)
     ax3 = fig.add_subplot(2,2,3)
     ax4 = fig.add_subplot(2,2,4)
     axes = [ax1, ax2, ax3, ax4]

     # Correction factor for projection
     cosdec0 = np.cos(np.radians(delta0deg))
     # cosdec0 = np.cos(dec)

     # correct the cols and deltas for cos(dec_0)
     
     cols = [J_xa, J_xd, J_ya, J_yd]
     labels = [r'$(\partial x/\partial \alpha_0) \Delta \alpha_0$', \
               r'$(\partial x/\partial \delta_0) \Delta \delta_0$', \
               r'$(\partial y/\partial \alpha_0) \Delta \alpha_0$', \
               r'$(\partial y/\partial \delta_0) \Delta \delta_0$']

     deltas = [damas/cosdec0, ddmas, damas/cosdec0, ddmas]
     
     for iax in range(len(axes)):
         ax = axes[iax]
         dum = ax.scatter(xgarcmin, ygarcmin, c=cols[iax]*deltas[iax])
         ax.set_title('%s , mas' % (labels[iax]))

         ax.set_xlabel('x (arcmin)')
         ax.set_ylabel('y (arcmin)')
         
         cbar = fig.colorbar(dum, ax=ax)
     


     fig.subplots_adjust(hspace=0.3, wspace=0.3)
     ssuptitle=r'$(\cos{\delta_0} \Delta \alpha_0, \Delta \delta_0) = (%.1f$",$ %.1f$"$)$' \
         % (damas/1000., ddmas/1000.)

     # add the pointing
     ssuptitle = r'%s , ($\alpha_0, \delta_0, \phi) = (%.1f, %.1f, %.2f)$' \
         % (ssuptitle, alpha0deg, delta0deg, posdeg)
     
     fig.suptitle(ssuptitle)

     ## FOr the moment, repurpose the top right figure
     #ax2.cla()
     #dum99 = ax2.scatter(xgarcmin, ygarcmin, c=np.degrees(xr1-xr)*3.6e6)
     #cbar99 = fig.colorbar(dum99, ax=ax2)
