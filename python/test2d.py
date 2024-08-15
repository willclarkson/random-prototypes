#
# test2d.py
#

#
# 2024-08-15 WIC - test routines for various pieces of 2d mcmc, which
# I think may make useful figures.
#

import numpy as np
import os, time
import matplotlib.pylab as plt
plt.ion()

# The various pieces we want to examine
from weightedDeltas import CovarsNx2x2
import noisemodel2d

def shownoisemodel(parsnoise=[-4., -20., 2.], \
                   parsshape=[], islog10_ryx=False, \
                   maglo=16., maghi=19.5, npts=100, \
                   ylog=False):

    """Tests noise model. 

Example calls:

(1) Demo the behavior of the covariance major and minor axes with
shape params:

    test2d.shownoisemodel(parsshape=[], ylog=True)

    test2d.shownoisemodel(parsshape=[0.6], ylog=True)

    test2d.shownoisemodel(parsshape=[0.9, -0.4], ylog=True)

    """

    # 1. Evaluate the noise vs magnitude model
    mags = np.random.uniform(maglo, maghi, npts)
    stdxs = noisemodel2d.noisescale(parsnoise, mags)

    # For plotting things against magnitude as a function
    mfine = np.linspace(maglo, maghi, 100)

    # Constant and full model
    parsslope = np.hstack(( -99., parsnoise[1::] ))
    yconst = noisemodel2d.noisescale(parsnoise[0], mfine)
    yslope = noisemodel2d.noisescale(parsslope, mfine)
    yfine = noisemodel2d.noisescale(parsnoise, mfine)

    # 2. Shape model
    stdx, stdy, corrxy = \
        noisemodel2d.parsecorrpars(stdxs, parsshape, unpack=True, \
                                   islog10_ryx=islog10_ryx)

    # 3. Generate covariance matrices using these parameters
    CC = CovarsNx2x2(stdx=stdx, stdy=stdy, corrxy=corrxy)
    covars = CC.covars

    # 4. Find determinants of the covariance matrices, and the
    # major/minor axes. Convert these to stddev equivalents for
    # comparison with the noise model
    detcovs = np.linalg.det(covars)
    CC.populateTransfsFromCovar()

    det_stds = detcovs**0.25 # det has units variance squared
    cov_stds = np.sqrt(CC.majors)
    cov_minors = np.sqrt(CC.minors)

    # 5. Draw samples from the covariance matrix and determine whether
    # they match our expectations.
    xysamples = CC.getsamples()

    # Statistics vs magnitude for the samples
    magmid, medmid, covmid = statsvsmag(mags, xysamples)

    # rotation angles
    cov_rotans = CC.rotDegs
    print(cov_rotans[0:3])
    
    # Colors for pieces
    color_model = '#9A3324'
    color_cc = '#2F65A7'
    color_cc_minor = '#575294'
    color_det = '#A5A508'
    color_comp = 'k'
    
    fig2=plt.figure(2)
    fig2.clf()
    ax21 = fig2.add_subplot(111)

    # It's annoying to plot very large numbers of points. Show a
    # subsample
    nshow = np.min([100, np.size(mags)])
    xdum = np.random.uniform(size=np.size(mags))
    ldum = np.argsort(xdum)
    lshow = ldum[0:nshow]

    if nshow < np.size(mags):
        snum = '%i of %s shown' % (nshow, f"{np.size(mags):,}")
        ax21.annotate(snum, (0.98, 0.98), xycoords='axes fraction', \
                      ha='right', va='top', \
                      fontsize=10, color=color_cc)
    
    dumfine = ax21.plot(mfine, yfine, color=color_model, ls='-', \
                        label='Model: stddev(x)', \
                        zorder=20, alpha=0.8)
    dumconst = ax21.plot(mfine, yconst, color=color_model, ls='--', \
                         zorder=2, alpha=0.8)
    dumslope = ax21.plot(mfine, yslope, color=color_model, ls='-.', \
                         zorder=2, alpha=0.8)

    # Now overplot the major axes and the determinant factors
    dummajors = ax21.scatter(mags[lshow], cov_stds[lshow], \
                             color=color_cc, marker='o', \
                             alpha=0.5, s=16, zorder=10, \
                             label='Major axis (as sqrt(var) )')
    dumminors = ax21.scatter(mags[lshow], cov_minors[lshow], \
                             color=color_cc_minor, \
                             marker='x', \
                             alpha=0.5, s=16, zorder=10, \
                             label='Minor axis (as sqrt(var) )')

    
    dumfacts = ax21.scatter(mags[lshow], det_stds[lshow], \
                            color=color_det, marker='s', \
                            alpha=0.5, s=25, zorder=5, \
                            label=r'$|V|^{1/4}$')

    # If we have them, show the results of our test sample
    if np.size(covmid) > 0:
        scomp = r'$\sqrt{s^2_x}$ from sample'
        dumcomp = ax21.scatter(magmid, covmid[:,0,0]**0.5, \
                               alpha=0.95, color=color_comp, \
                               zorder=50, \
                               label=scomp)
    
    # legend
    leg = ax21.legend()
    
    # cosmetics
    ax21.set_xlabel('mag')
    ax21.set_ylabel(r'$\sigma$')

    # Hack to convert the shape parameters into a single string for title
    sshp = ['%.2f' % (parsshape[i]) for i in range(len(parsshape)) ]
    strshape = '[%s]' % (', '.join(sshp))

    # ditto for noise parameters
    snoise = ['%.1f' % (parsnoise[i]) for i in range(len(parsnoise)) ]
    strnoise = '[%s]' % (', '.join(snoise))

    ssup = 'Noise pars: %s, Shape pars: %s' \
                   % (strnoise, strshape)
    
    ax21.set_title(ssup)
        
    if ylog:
        ax21.set_yscale('log')

        # Hack to fit the range nicely on the plot
        ilo = np.argmin(mags)
        ymin = np.min([yfine[0], cov_minors[ilo], \
                       cov_stds[ilo], det_stds[ilo]])

        print("%.2e, %.2e, %.1e" % (ymin, yslope[0], ymin / yslope[0]))
        
        if np.abs(ymin / yslope[0]) > 10.:
            ax21.set_ylim(bottom=ymin*0.7)

    # Now a figure 3, showing draws from our samples
    fig3=plt.figure(3, figsize=(5.5, 6.5))
    fig3.clf()
    ax30 = fig3.add_subplot(321)
    
    ax31=fig3.add_subplot(323)
    ax32=fig3.add_subplot(324)

    # scatter vs mag a slightly different way...
    ax33=fig3.add_subplot(325)
    ax34=fig3.add_subplot(326)

    # Show the deltas color-coded by mag
    p_scatt = ax30.scatter(xysamples[:,0], xysamples[:,1], \
                           c=mags, alpha=0.5, s=2, cmap='viridis_r')
    cb30 = fig3.colorbar(p_scatt, ax=ax30)
    
    # First off show deltas vs mag
    p_dxvsmag = ax31.scatter(mags, xysamples[:,0], alpha=0.15, s=2)
    p_dyvsmag = ax32.scatter(mags, xysamples[:,1], alpha=0.15, s=2)

    p_ldx = ax33.scatter(mags, np.abs(xysamples[:,0]), alpha=0.15, s=2)
    p_ldy = ax34.scatter(mags, np.abs(xysamples[:,1]), alpha=0.15, s=2)
    
    for ax in [ax33, ax34]:
        ax.set_yscale('log')
    
    # How are our bin statistics looking?
    # dum_bin = ax31.scatter(magmid, medmid[:,0], zorder=10, s=1)

    # show the std devs from the fake data
    for sigfac, ls in zip([1., 3.], ['-', '--']):
        for ax in [ax31, ax33]:
            p_envx_hi = ax.plot(magmid, covmid[:,0,0]**0.5 * sigfac, \
                                alpha=1., zorder=25, \
                                color=color_comp, lw=2, ls=ls)
        p_envx_lo = ax31.plot(magmid, 0.-covmid[:,0,0]**0.5 * sigfac, \
                              alpha=1., zorder=25, \
                              color=color_comp, lw=2, ls=ls)

        for ax in [ax32, ax34]:
            p_envy_hi = ax.plot(magmid, covmid[:,1,1]**0.5 * sigfac, \
                                alpha=1., zorder=25, \
                                color=color_comp, lw=2, ls=ls)
        p_envy_lo = ax32.plot(magmid, 0.-covmid[:,1,1]**0.5 * sigfac, \
                              alpha=1., zorder=25, \
                              color=color_comp, lw=2, ls=ls)

    
    # annotations
    for ax in [ax31, ax32, ax33, ax34]:
        ax.set_xlabel('mag')

    ax31.set_ylabel(r'$\Delta x$')
    ax33.set_ylabel(r'$|\Delta x|$')

    ax32.set_ylabel(r'$\Delta y$')
    ax34.set_ylabel(r'$|\Delta y|$')
    for ax in [ax32, ax34]:
        ax.yaxis.set_label_position('right')
        
    ax30.set_xlabel(r'$\Delta x$')
    ax30.set_ylabel(r'$\Delta y$')
    
    fig3.subplots_adjust(left=0.2, wspace=0.4, hspace=0.4)
    
    # Show the supertitle
    fig3.suptitle(ssup, fontsize=10)

def statsvsmag(mags=np.array([]), xy=np.array([]), nbins=15):

    """Utility - evaluates statistics per bin"""

    # Uniform binning by cdf
    lsor = np.argsort(mags)
    ileft = np.asarray(np.linspace(0, np.size(lsor), \
                                   nbins, endpoint=False), 'int')
    iright = np.hstack(( ileft[1::], np.size(lsor) ))

    # Bin statistics
    mag_mid = np.array([])
    med_bin = np.array([])
    cov_bin = np.array([])
    
    for ibin in range(np.size(ileft)):

        lthis = lsor[ileft[ibin]:iright[ibin]]

        # Statistics
        med_dxy = np.median(xy[lthis], axis=0)
        if np.size(med_bin) < 1:
            med_bin = np.copy(med_dxy)
        else:
            med_bin = np.vstack(( med_bin, med_dxy ))

        # Find the covariance of this bin
        cov_dxy = np.cov(xy[lthis], rowvar=False)
        if np.size(cov_bin) < 1:
            cov_bin = np.copy(cov_dxy)
        else:
            cov_bin = np.dstack(( cov_bin, cov_dxy ))
            
        # magnitude midpoint, just to ensure we're doing something
        # roughly sensible
        mag_mid = np.hstack((mag_mid, np.median(mags[lthis]) ))

    # I prefer covariances in [N,2,2] not [2,2,N]
        
    return mag_mid, med_bin, np.transpose(cov_bin, axes=(2,0,1))
