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

from parset2d import Pars1d
import unctytwod

import lnprobs2d

import sim2d

def shownoisemodel(parsnoise=[-4., -20., 2.], \
                   parsshape=[], islog10_ryx=False, \
                   maglo=16., maghi=19.5, npts=1000, \
                   ylog=False):

    """Tests noise model. 

Example calls:

(1) Demo the behavior of the covariance major and minor axes with
shape params:

    test2d.shownoisemodel(parsshape=[], ylog=True)

    test2d.shownoisemodel(parsshape=[0.6], ylog=True)

    test2d.shownoisemodel(parsshape=[0.9, -0.4], ylog=True)

This also compares the running stddevx vs magnitude against the model. To do so, it's usually a good idea to make npts fairly large, e.g.:

    test2d.shownoisemodel(parsshape=[.5], ylog=True, islog10_ryx=False, npts=500)

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
    #stdx, stdy, corrxy = \
    #    noisemodel2d.parsecorrpars(stdxs, parsshape, unpack=True, \
    #                               islog10_ryx=islog10_ryx)

    # 3. Generate covariance matrices using these parameters
    #CC = CovarsNx2x2(stdx=stdx, stdy=stdy, corrxy=corrxy)

    CC = noisemodel2d.mags2noise(parsnoise, parsshape, mags, islog10_ryx)
    
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
    magmid, medmid, covmid, countbin = statsvsmag(mags, xysamples)

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
        inum = int(countbin[0])
        scomp = r'$\sqrt{s^2_x}$ at %s / bin' \
            % (f"{inum:,}")
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
    fig3=plt.figure(3, figsize=(6.6, 6.5))
    fig3.clf()
    ax30 = fig3.add_subplot(321)
    
    ax31=fig3.add_subplot(323)
    ax32=fig3.add_subplot(324)

    # scatter vs mag a slightly different way...
    ax33=fig3.add_subplot(325)
    ax34=fig3.add_subplot(326)

    # Show the deltas color-coded by mag
    p_scatt = ax30.scatter(xysamples[:,0], xysamples[:,1], \
                           c=mags, alpha=0.5, s=2, cmap='gray_r')
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

def testcovtran(npts=1000, parsnoise=[-4.], \
                xmin=-1., xmax=1., ymin=-1., ymax=1.):

    """Tests transformation of covariance from source to target frame.

Inputs:

    npts = number of points to simulate

    parsnoise = [loga, logb, c] parameters for the covariance.

    xmin, xmax, ymin, ymax = domain limits for observed data

"""

    # We set up to do the various things in lnprob.py so that we can
    # test if that's actually producing the output we expect.
    
    # Stick in a "known" transformation
    parsx = np.array([0.2, 3., 0])
    parsy = np.array([0.35, 0., 2.])
    parsvec = np.hstack(( parsx, parsy ))

    # covariance
    parsshape=[]

    
    # Set up a transformation object
    Pset = Pars1d(parsvec)

    # Set up random positions and covars
    x = np.random.uniform(xmin, xmax, size=npts)
    y = np.random.uniform(ymin, ymax, size=npts)
    xy = np.column_stack((x,y))

    xy = np.random.uniform(low=-1., high=1., size=(npts, 2))
    mags = np.random.uniform(low=16., high=19.5, size=npts)

    CC = noisemodel2d.mags2noise(parsnoise, parsshape, mags)
    covsxy = np.copy(CC.covars)

    # Populate an observation-set object with the data and domain info
    targset = sim2d.Obset(xy, covsxy, mags, \
                          xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    
    # OK first look at the transformation. Set up the object:
    transf = unctytwod.Poly(xy[:,0], xy[:,1], np.copy(covsxy), \
                            parsx, parsy, kind='Polynomial', \
                            xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    t0 = time.time()
    transf.propagate()
    print("Propagated in %.2e seconds" % (time.time() - t0))

    # The following syntax tests various parts of lnprob
    ll = lnprobs2d.Like(Pset, transf, targset)

    # What do the results look like:
    print("lnprobs2d.Like() test for covariance consistency:")
    print("-------------------------------------------------")
    print("covtarg:", ll.covtarg[0])
    print("covtran:", ll.covtran[0])
    print("covextra:", ll.covextra[0])
    print("covoutly:", ll.covoutly[0])

    # Sum the covariances manually, compare with the LIke sum
    covsum = ll.covtarg + ll.covtran + ll.covextra + ll.covoutly
    
    print("ll.covsum:",ll.covsum[0])
    print("covsum   :", covsum[0])
    
    # The syntax below compares the propagation of the covariance from
    # source to target frame, against direct calculation outside the
    # Poly() object.
    
    # datatypes?
    print("")
    print("Covariance propagation from source to target:")
    print("---------------------------------------------")

    print(covsxy.dtype)
    print(transf.covtran.dtype)
    
    # Now we populate the jacobian to compare directly.
    jacorig = covsxy * 0.
    jacorig[:,0,0] = parsx[1]
    jacorig[:,0,1] = parsx[2]
    jacorig[:,1,0] = parsy[1]
    jacorig[:,1,1] = parsy[2]

    # There's also a jacobian handling the rescaling from the input
    # positions to the [-1,1] interval for the polynomials. Handle
    # that here. Notice it's the DOMAIN we are rescaling, not the area
    # actually covered by the data
    jacrescale = covsxy*0.
    jacrescale[:,0,0] = 2.0/(xmax-xmin)
    jacrescale[:,1,1] = 2.0/(ymax-ymin)

    jac = np.matmul(jacorig, jacrescale)
    
    print(jacrescale[0])
    
    t0 = time.time()
    VJt = np.matmul(covsxy, np.transpose(jac, axes=(0,2,1)))
    JVJt = np.matmul(jac, VJt)
    print("Evaluated matmul (twice) in %.2e seconds" % (time.time() - t0))
    
    print("Propagation comparison:")
    print("Original:", transf.covxy[0])
    print("Jac:", jac[0])
    print("covtran:", transf.covtran[0])
    print("JVJt:", JVJt[0])

    

    # Residual
    print("JVJt - covtran:", JVJt[1] - transf.covtran[1])
    # print("Fractional residual:", \
    #      (JVJt[1]-transf.covtran[1])/transf.covtran[1])

    print(transf.parsx, transf.parsy)

    # Appeal to vectorized version
    t0 = time.time()
    jvjt_vec = JVJt_vectorized(jac, transf.covxy)
    print("Evaluated vectorized in %.2e seconds" \
          % (time.time() - t0))
    
    print(jvjt_vec[0])

def mixmodvals(nfrac=20, nvar=20, logvarpadlo=1.5, logvarpadhi=2.5, \
               doloops=True, showlog10resps=False):

    """Sets up a mixture model and plots the variation of fit statistic with trial mixture parameters. 

Inputs:

    nfrac, nvar = number of grid points to use along log10(fbg),
    log10(vxx), respectively

    logvarpadlo, logvarpadhi = lower and upper distances from truth log10(vxx) to use for the grid

    doloops = do the loop through the grid (takes about a minute)

    showlog10resps = plot the responsibilities as log10(resps)

Example call:

    test2d.mixmodvals(51,51, -10.,-7.)

Currently this is a complete mess. To be cleaned up!

    """

    # Simulate a dataset with a mixture model
    SD = sim2d.Simdata()
    SD.loadconfig('test_config_mixmod.ini')
    SD.generatedata()

    # We're going to want to look at the scatterplot with identified
    # objects. Get those parameters here.
    SD.PTruth.propagate()
    dxy = SD.PTruth.xytran - SD.Obstarg.xy
    mags = SD.Obstarg.mags
    isfg = SD.Obstarg.isfg

    # what IS the covariance of the outliers? Find it and use our
    # covariance object to interpret it
    cov_all = np.cov(dxy, rowvar=False)
    cov_inliers = np.cov(dxy[isfg], rowvar=False)
    cov_outliers = np.cov(dxy[~isfg], rowvar=False)

    CA = CovarsNx2x2(cov_all)
    CA.eigensFromCovars()
    
    CI = CovarsNx2x2(cov_inliers)
    CI.eigensFromCovars()

    CB = CovarsNx2x2(cov_outliers)
    CB.eigensFromCovars()

    print(" ")
    print("Data covariance check:")
    print("--------------------------------------------------------------")
    print("Truth set: covariance, all:")
    print(cov_inliers)
    print(CA.majors, CA.stdx, CA.stdy, np.log10(CA.majors))

    print("--------------------------------------------------------------")
    print("Truth set: covariance, inliers:")
    print(cov_inliers)
    print(CI.majors, CI.stdx, CI.stdy, np.log10(CI.majors))

    print(" ")
    print("Outlier set: covariance, outliers:")
    print(cov_outliers)
    print(CB.majors, CB.stdx, CB.stdy, np.log10(CB.majors))
    print("--------------------------------------------------------------")

    # We have the truth parameters for everything. Try varying only
    # the mixture fraction and the (log-) covariance and trace the
    # variation of fom as those vary. Like so:

    truthmix = np.copy(SD.Parset.mix)
    print("Truth mixture parameters:", truthmix)
    
    vlogfrac = np.linspace(-3., -0.1, nfrac, endpoint=True)
    vlogvar = np.linspace(truthmix[1]-logvarpadlo, \
                          truthmix[1]+logvarpadhi, \
                          nvar, endpoint=True)

    ff, vv = np.meshgrid(vlogfrac, vlogvar, indexing='ij')
    ll = ff * 0. - np.inf

    # the truth parset and fit parset are different if we are not
    # fitting for extra noise.
    print(SD.Parset.model)
    print(SD.Parset.noise)
    print(SD.Parset.symm)
    print(SD.Parset.mix)

    # Create paramset object corresponding to the parameters we
    # actually want to explore, create lnlike object to (re-)compute
    # the likelihood for each mixture-variance pair
    Parsho = Pars1d(model=SD.Parset.model, noise=[], symm=[], \
                    mix=SD.Parset.mix)

    llike = lnprobs2d.Like(Parsho, SD.PTruth, SD.Obstarg)

    # compute the responsibilities given the truth mixture values. NOT
    # done as part of each function evaluation for speed and storage
    # issues.
    llike.calcresps()
    
    # Test uTvu
    precis = np.linalg.inv(llike.covsum)
    t0 = time.time()
    utvu_vec = utVu_vectorized(dxy, precis)
    t1 = time.time()
    utvu = lnprobs2d.uTVu(dxy, precis)
    t2 = time.time()

    print(" ")
    print("utVu Test:")
    print("----------------------------------------------------------")
    print("utVu INFO - vectorized %.2e sec; einsum %.2e sec" \
          % (t2-t1, t1-t0))
    
    for isho in range(5):
        print("uTVu: %i, %.5f, %.5f, %.2e, %.2e, %.2e" \
              % (isfg[isho], utvu[isho], utvu_vec[isho], \
                 dxy[isho,0], dxy[isho,1], llike.covsum[isho,0,0]) )
    print("----------------------------------------------------------")

    # return

    # what do the outliers look like...
    #print(llike.covsum[isfg][0:3])
    #print(llike.covsum[~isfg][0:3])  

    # Now take a look...
    fig4 = plt.figure(4, figsize=(8.25, 4.8))
    fig4.clf()
    fig4.subplots_adjust(hspace=0.5)
    # ax4 = fig4.add_subplot(121) # deferred to later
    ax42 = fig4.add_subplot(233)    
    ax43 = fig4.add_subplot(236)    
    
    dumdxy = ax42.scatter(dxy[:,0], dxy[:,1], c=isfg, cmap='viridis', s=4)
    cb42 = fig4.colorbar(dumdxy, ax=ax42, label='foreground (1/0)')

    # same thing, this time coded by uncertainty
    stdx = np.log10(llike.covsum[:,0,0])

    respsho = llike.resps_fg
    sresp = r'$\pi_{fg}$'
    if showlog10resps:
        respsho = np.log10(llike.resps_fg)
        sresp = r'$log_{10}(\pi_{fg})$'
        
    dumss = ax43.scatter(dxy[:,0], dxy[:,1], \
                         c=respsho, \
                         cmap='viridis', s=4)
    cb43 = fig4.colorbar(dumss, ax=ax43, label=sresp)

    for ax in [ax42, ax43]:
        ax.set_xlabel(r'$\Delta \xi$')
        ax.set_ylabel(r'$\Delta \eta$')
    
    #ax42.set_title('Colors: isfg', fontsize=10)
    #ax43.set_title(sresp, fontsize=10)

    # option to return without doing all the loops
    if not doloops:
        return

    # add the axis if we're going to use it
    ax4 = fig4.add_subplot(121)

    print(" ")
    print("Loops through [log10(fbg), log10(vxx_bg)]:")
    print("------------------------------------------------------------------------------")
    
    # Now populate the trial values. Do as meshgrid so that we can
    # easily contour the results.
    parsvec = np.copy(Parsho.pars) # as might be sent in
    for ifrac in range(np.size(vlogfrac)):
        for jvxx in range(np.size(vlogvar)):
            parsvec[-2] = ff[ifrac, jvxx]
            parsvec[-1] = vv[ifrac, jvxx]

            # Update the source parset object (because when used for
            # real, the same parset is referenced by both Like and
            # Prior. So we want to update it OUTSIDE Like).
            Parsho.updatepars(parsvec)
            
            llike.updatelnlike(Parsho)

            ll[ifrac, jvxx] = np.copy(llike.sumlnlike)

            # Print screen output
            if ifrac is 40:
                itell=3
                print("%.2e, %.2e:: %.2e, %.2e %.2e ##, %.2e, %.2e, %.2e -- %.2e, %.2e, %.2e >> %i, %.1f" \
                      % (llike.fbg, parsvec[-1], \
                      llike.lnlike_fg[itell], llike.lnlike_bg[itell], \
                         llike.lnlike_fg[itell] +  llike.lnlike_bg[itell], \
                         llike.covsum[itell][0,0], \
                         llike.covoutly[itell][0,0], \
                         llike.covoutly[itell][0,0] + \
                         llike.covsum[itell][0,0], \
                         llike.covtarg[itell][0,0], \
                         llike.covtran[itell][0,0], \
                         llike.covextra[itell][0,0], \
                         isfg[itell], mags[itell]))
            
    # Now look at the result of our loops
    dum = ax4.contour(ff, vv, ll, levels=20, zorder=10)

    # show a scatterplot as well
    dumscatt = ax4.scatter(np.ravel(ff), np.ravel(vv), c=np.ravel(ll),
                           s=4, zorder=1, alpha=0.3)
    
    ax4.set_xlabel(r'$log_{10}$(mixture fraction)')
    ax4.set_ylabel(r'$log_{10}(V_{xx})$')

    cb = fig4.colorbar(dumscatt, ax=ax4)

    # where is the minimum value?
    f1d = np.ravel(ff)
    v1d = np.ravel(vv)
    l1d = np.ravel(ll)
    imax = np.argmax(l1d)

    # I specified the background fraction not the foreground
    dummax = ax4.scatter([f1d[imax]], [v1d[imax]], \
                         c='m', zorder=25)
    
    print(imax, f1d[imax], v1d[imax], l1d[imax])
    
    # We know what the generated values were! Plot them
    #
    # I think I flipped foreground/background when generating. Handle
    # that...
    #truth_f = np.log10(1.0 - 10.0**truthmix[0])
    truth_f = truthmix[0]
    
    dumtruth = ax4.scatter([truth_f], [truthmix[1]], \
                           c='m', marker='*', zorder=20) 
    
    # We should have the truth parameters - for everything - now. Take
    # a look:
    print(SD.Obssrc.xmin)
    print(SD.Parset.pars)
    
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
    count_bin = np.array([])
    
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

        count_bin = np.hstack(( count_bin, np.size(lthis) ))
        
    # I prefer covariances in [N,2,2] not [2,2,N]
        
    return mag_mid, med_bin, np.transpose(cov_bin, axes=(2,0,1)), \
        count_bin

def JVJt_vectorized(jac=np.array([]), cov=np.array([])):

    """Element-by-element evaluation of JVJt"""

    # Datatypes?
    print(jac.dtype, cov.dtype)

    # must both be Nx2x2. This is to check where along the chain the
    # multiplications might be losing accuracy. Subscripts as per
    # manuscript draft
    j11 = jac[:,0,0]
    j12 = jac[:,0,1]
    j21 = jac[:,1,0]
    j22 = jac[:,1,1]

    s1 = cov[:,0,0]
    s12 = cov[:,0,1]
    s21 = cov[:,1,0]
    s2 = cov[:,1,1]

    var11 = j11**2 * s1    + j12**2 * s2 + 2. * j11 * j12 * s12
    var22 = j21**2 * s1    + j22**2 * s2 + 2. * j21 * j22 * s12
    var12 = j11 * j21 * s1 + j12*j22* s2 + (j11*j22 + j12*j21)*s12

    jvjt = jac*0.

    jvjt[:,0,0] = var11
    jvjt[:,1,1] = var22
    jvjt[:,0,1] = var12
    jvjt[:,1,0] = var12

    return jvjt

def utVu_vectorized(u=np.array([]), V=np.array([]) ):

    """Vectorized version of (x-mu)^T V (x-mu) just to check that I haven't done something idiotic setting that up in lnprobs2d. 

Inputs:

    u = [N,2] array of deltas

    V = [N,2,2] precision array

Returns:

    uT.V.u = [N] array of evaluates

"""

    s11 = V[:,0,0]
    s22 = V[:,1,1]
    s12 = V[:,0,1]

    x = u[:,0]
    y = u[:,1]

    return s11*x**2 + s22*y**2 + 2.0*s12*x*y
    
