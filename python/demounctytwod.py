#
# demounctytwod.py
#

#
# WIC 2024-07-23 - demo uncertainty2d methods by making nice plots of
# input and transformed covariances
#


# We want the methods as well as the objects, so import the entire
# module for now.
import unctytwod
from weightedDeltas import coverrplot, CovarsNx2x2

# for other operations we'l need
import numpy as np
import matplotlib.pylab as plt

def showtan2equ(sidelen=25., alpha0=35., delta0=35., \
                ncoarse=15, nfine=101, \
                sigx=0.1, sigy=0.05, sigr=0.0, \
                fignum=4, Verbose=True, \
                errsf=50000., edgecolor='0.3', \
                reverse=False):

    """Demos tangent plane to equatorial, optionally showing the reverse
direction"""

    # Create fine grid and covariances at each cross-point
    xigrid, etagrid, xicross, etacross, covxieta = \
        setupxieta(sidelen, ncoarse, nfine, sigx, sigy, sigr)
    
    # Now transform the coordinates and the uncertainties. First we
    # set up the transformation object, then we apply it.
    tpoint = np.array([alpha0, delta0])
    T2E = unctytwod.Tan2equ(xicross, etacross, covxieta.covars, \
                            tpoint, Verbose=Verbose)

    T2E.propagate()

    # Show the forward direction anyway.
    showellipses(T2E, xigrid, etagrid, errsf, fignum=fignum)
        
    # if we're here, then we are starting at equatorial and projecting
    # back. So:
    E2T = unctytwod.Equ2tan(T2E.xtran, T2E.ytran, T2E.covtran, \
                            tpoint, Verbose=Verbose)
    E2T.propagate()

    # propagate the grid from equatorial back to tangent plane
    ragrid, degrid = T2E.propxy(xigrid, etagrid)

    showellipses(E2T, ragrid, degrid, errsf, fignum=fignum+1)
    return

def showpoly(sidelen=25., deg=3, kind='Polynomial', \
             ncoarse=15, nfine=101, \
             sigx=0.1, sigy=0.04, sigr=0.0, \
             fignum=4, Verbose=True, \
             errsf=50000., edgecolor='0.3', \
             reverse=False, projsky=True, \
             alpha0=35., delta0=35., \
             scale=1., rotdeg=0.):

    """Demos tangent plane to polynomial

    scale = scale factor from [-1,1] to xi, eta (convenient way to
    inflate the output)

    Example calls:

    demounctytwod.showpoly(kind='Chebyshev', reverse=False, sidelen=2., errsf=4000, delta0=85., deg=3)

    demounctytwod.showpoly(kind='Legendre', reverse=True, sidelen=500., errsf=1000000, scale=2., delta0=85., deg=3, rotdeg=20.)

    """
    
    # Set up the xi, eta points
    xigrid, etagrid, xicross, etacross, covxieta = \
        setupxieta(sidelen, ncoarse, nfine, sigx, sigy, sigr, \
                   pix=reverse)

    # Make up some parameters at the given degree
    parsx, parsy = unctytwod.makepars(deg=deg, reverse=reverse, \
                                      scale=scale, rotdeg=rotdeg)

    # now create the polynomial object
    PP = unctytwod.Poly(xicross, etacross, covxieta.covars, parsx, parsy, \
                        kind=kind, Verbose=Verbose, \
                        degrees=not(reverse), xisxi=not(reverse))
    PP.propagate()

    # Show the ellipses
    showellipses(PP, xigrid, etagrid, errsf, fignum=fignum)

    # bonus - project onto the sky (assuming the target frame is xi,
    # eta)?
    if not projsky or not reverse:
        return

    T2E = unctytwod.Tan2equ(PP.xtran, PP.ytran, PP.covtran, \
                            np.array([alpha0, delta0]), Verbose=Verbose)
    T2E.propagate()
    xg, yg = PP.propxy(xigrid, etagrid) # unfortunate naming
    
    showellipses(T2E, xg, yg, errsf, fignum=fignum+1, \
                 enforceuniformaxes=False, showmajors=False)
    
    
def setupxieta(sidelen=25., ncoarse=15, nfine=101, \
               sigx=0.1, sigy=0.05, sigr=0.0, pix=False):
    

    """Utility - sets up xi, eta points for the plots"""

    # create grid of positions for plot
    xigrid, etagrid = unctytwod.gridxieta(sidelen, ncoarse, nfine)

    # create grid of positions at the meeting points of the fine grid
    xicross, etacross = unctytwod.gridxieta(sidelen, ncoarse, ncoarse)

    # If we want the lower-left to be 0,0 then we subtract the minimum
    # grid point from both outputs
    if pix:
        ximin = np.min(xigrid)
        etamin = np.min(etagrid)

        xigrid -= ximin
        xicross -= ximin
        etagrid -= etamin
        etacross -= etamin
    
    # Create covariances stack in the tangent plane
    npts = np.size(xicross)
    covxieta = CovarsNx2x2(stdx=np.repeat(sigx/3600.,npts), \
                           stdy=np.repeat(sigy/3600.,npts), \
                           corrxy=np.repeat(sigr, npts))

    return xigrid, etagrid, xicross, etacross, covxieta
    
def showellipses(transf=None, xigrid=np.array([]), etagrid=np.array([]),\
                 errsf=10000, edgecolor='0.3', \
                 fignum=4, enforceuniformaxes=True, showmajors=True):

    """Refactored our input/output plotter into a new method for
repeatability. 

    transf = object that propagates positions and covariances.


"""

    if transf is None:
        return
    
    # (Re-)create CovarsNx2x2 objects from the covariance arrays
    covxieta = CovarsNx2x2(transf.covxy)
    covradec = CovarsNx2x2(transf.covtran)

    # Use this object to propagate the grid positions as well (which
    # does NOT require the jacobian)
    ragrid, degrid = transf.propxy(xigrid, etagrid)
    
    # set the figure
    fig = plt.figure(fignum, figsize=(8,4))
    fig.clf()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # show the grid
    blah1g = ax1.scatter(xigrid, etagrid, s=.1, c='0.5')
        
    # Now show the ellipses
    ell1 = coverrplot(transf.x, transf.y, covxieta, errSF=errsf, \
                      ax=ax1, fig=fig, showColorbarEllipse=False, \
                      colorMajors=edgecolor, colorMinors=edgecolor, \
                      edgecolorEllipse=edgecolor, \
                      shadeEllipses=False, \
                      enforceUniformAxes=enforceuniformaxes, \
                      showMajors=showmajors, showMinors=showmajors)

    # Show the transformed positions on the second axis
    blah2g = ax2.scatter(ragrid, degrid, s=.1, c='0.5')
    ell2 = coverrplot(transf.xtran, transf.ytran, covradec, \
                      errSF=errsf, \
                      ax=ax2, fig=fig, showColorbarEllipse=False, \
                      colorMajors=edgecolor, colorMinors=edgecolor, \
                      edgecolorEllipse=edgecolor, \
                      shadeEllipses=False, \
                      enforceUniformAxes=enforceuniformaxes, \
                      showMajors=showmajors, showMinors=showmajors)

    # Set the labels
    ax1.set_xlabel(transf.labelx)
    ax1.set_ylabel(transf.labely)
    ax2.set_xlabel(transf.labelxtran)
    ax2.set_ylabel(transf.labelytran)

    # indicate which is which
    ax1.set_title('Original frame')
    ax2.set_title('Transformed frame')
    
    # Allow some flexibility with the kind of object. If two
    # parameters, assume those are pointing center.
    ssup = 'ellipses scaled by %.2e' % (errsf)

    if hasattr(transf, 'pars'):
        skind = r'$(\alpha_0, \delta_0)=(%.1f^{\circ}, %.1f^{\circ})$' \
            % (transf.pars[0], transf.pars[1])

        ssup = r'%s, %s' % (skind, ssup)

    # if we're showing the results of a polynomial transformation:
    if hasattr(transf, 'parsx'):
        ssup="%s (degree %i), %s" % (transf.kind, transf.pars2x.deg, ssup)

        
    fig.suptitle(ssup)
    
    # adjust the figure to allow a good interval between panes
    fig.subplots_adjust(hspace=0.3, wspace=0.3, top=0.85)
