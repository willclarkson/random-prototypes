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
    
    # rotation angles
    cov_rotans = CC.rotDegs
    print(cov_rotans[0:3])
    
    # Colors for pieces
    color_model = '#9A3324'
    color_cc = '#2F65A7'
    color_cc_minor = '#575294'
    color_det = '#A5A508'
    
    fig2=plt.figure(2)
    fig2.clf()
    ax21 = fig2.add_subplot(111)

    dumfine = ax21.plot(mfine, yfine, color=color_model, ls='-', \
                        label='Model: stddev(x)', \
                        zorder=20, alpha=0.8)
    dumconst = ax21.plot(mfine, yconst, color=color_model, ls='--', \
                         zorder=2, alpha=0.8)
    dumslope = ax21.plot(mfine, yslope, color=color_model, ls='-.', \
                         zorder=2, alpha=0.8)

    # Now overplot the major axes and the determinant factors
    dummajors = ax21.scatter(mags, cov_stds, color=color_cc, marker='o', \
                             alpha=0.5, s=16, zorder=10, \
                             label='Major axis (as sqrt(var) )')
    dumminors = ax21.scatter(mags, cov_minors, color=color_cc_minor, \
                             marker='x', \
                             alpha=0.5, s=16, zorder=10, \
                             label='Minor axis (as sqrt(var) )')

    
    dumfacts = ax21.scatter(mags, det_stds, color=color_det, marker='s', \
                            alpha=0.5, s=25, zorder=5, \
                            label=r'$|V|^{1/4}$')

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
    
    ax21.set_title('Noise pars: %s, Shape pars: %s' \
                   % (strnoise, strshape))

        
    if ylog:
        ax21.set_yscale('log')

        # Hack to fit the range nicely on the plot
        ilo = np.argmin(mags)
        ymin = np.min([yfine[0], cov_minors[ilo], \
                       cov_stds[ilo], det_stds[ilo]])

        print("%.2e, %.2e, %.1e" % (ymin, yslope[0], ymin / yslope[0]))
        
        if np.abs(ymin / yslope[0]) > 10.:
            ax21.set_ylim(bottom=ymin*0.7)
