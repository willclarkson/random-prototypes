#
# test_numpyro2d.py
#

# 2026-05-15: Implement 2d mapping version of linear regression using
# HMCMC. This should show which aspects can be simply extended from
# the 1d case of test_numpyro1d.py .

# For *this* prototype, we can lay things out in the individual
# methods. If this works, the generalization into something a little
# more modular and flexible will be a separate process. But here we
# can build the modules around the models we want to fit.

# Our usual imports
import time

import numpy as np
import matplotlib.pylab as plt
plt.ion()

import jax
# jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpyro
from numpyro import distributions as dist, infer

# CPU count
numpyro.set_host_device_count(2)

# for examining the output
import arviz as az
import corner

# for constructing the covariances
import covarsNx2x2

# some visualization
from arviz_plots import plot_trace_dist, style


# For this, we adopt "x" as the "input" positions, and "u" as the
# "output". This allows us to use (x,y) and (u,v) later on if that is
# clearer.

def model_scalerot(x,uerr, u=None):

    """Two-parameter scale and rotation model. 

    INPUTS:

    x = [N,2] = input positions

    uerr = [N,2,2] = input uncertainties as covariances

    u = [N,2] optional output positions

    """

    # Define the priors as numpyro distributions. 
    theta = numpyro.sample("theta", dist.Uniform(-1.0*jnp.pi, 1.0*jnp.pi))
    s = numpyro.sample("s", dist.LogUniform(1e-5,1.))

    # Convert the theta, scale parameters into the CDMATRIX
    # parameters. These will be tracked. The CDMATRIX is {{b,c}.{e,f}}
    # following the conventions below. Redundant here but easy to
    # adjust into 6-term version later.
    b = numpyro.deterministic("b",  s * jnp.cos(theta))
    c = numpyro.deterministic("c",  s * jnp.sin(theta))
    e = numpyro.deterministic("e", -s * jnp.sin(theta))
    f = numpyro.deterministic("f",  s * jnp.cos(theta))

    # Create the transformation matrix A out of this
    A = jnp.array([[b,c],[e,f]])

    # and produce predicted u_i = A.x_i from each row
    upred = jnp.einsum('jk,ik -> ij', A, x)
    
    # At this point we evaluate the sampling distribution for the
    # multivariate normal. I am trusting the syntax to handle the
    # dimensions appropriately because I have not found documentation
    # that states explicitly how this works. I *think* the covariances
    # must be [N,D,D] if the data are [N,D], that is:
    pred_dist = dist.MultivariateNormal(upred, uerr)

    # and the sampling then works as follows (not using plates since I
    # am worried that it might break with multidimensional data)

    # what about plates - does it just take the left-most dimension?
    with numpyro.plate("data", x.shape[0]):    
        numpyro.sample("u", pred_dist, obs=u)

def model_6term(x, uerr, u=None, xerr=None, fitvar=False):

    """Offset and general linear transformation, parameterized in human
terms

INPUTS:

    x = [N,2] = input positions

    uerr = [N,2,2] = input uncertainties as covariances

    u = [N,2] optional output positions

    xerr = [N,2,2] optional xy uncertainties as covariances

    """

    # Define the priors as numpyro distributions. 
    theta = numpyro.sample("theta", dist.Uniform(-1.0*jnp.pi, 1.0*jnp.pi))
    s = numpyro.sample("s", dist.LogUniform(1e-5,1.))
    beta = numpyro.sample("beta", dist.Uniform(-0.5*jnp.pi, 0.5*jnp.pi))
    r = numpyro.sample("r", dist.Normal(1,1))
    u0= numpyro.sample("u0", dist.Uniform(-1.0, 1.0))
    v0= numpyro.sample("v0", dist.Uniform(-1.0, 1.0))

    # The conversion to cdmatrix entries, which will be tracked during
    # the sampling
    sx = s 
    sy = s * r
    
    b = numpyro.deterministic("b",  sx  * jnp.cos(theta - 0.5*beta) )
    c = numpyro.deterministic("c",  sy  * jnp.sin(theta + 0.5*beta) )
    e = numpyro.deterministic("e", -sx  * jnp.sin(theta - 0.5*beta) )
    f = numpyro.deterministic("f",  sy  * jnp.cos(theta + 0.5*beta) )

    # cdmatrix from b,c,e,f
    A = jnp.array([[b,c],[e,f]])

    # predicted u,v, including the offsets (note that jnp arrays
    # cannot be modified in place, so we need to construct the offset
    # as a separate quantity to add)
    upred = jnp.einsum('jk,ik -> ij', A, x) + jnp.array([u0,v0])[None,:]
    
    # upred[:,0] += u0
    # upred[:,1] += v0

    # Here we transform the supplied xy-frame covariances using the
    # transformation parameters. For the 6-term rtransformation, the
    # cdmatrix *is* the Jacobian. For more complicated models, things
    # will not be so simple (but I don't think e.g. chebyshev
    # polynomials have yet been implemented in jax in the same way
    # they are in numpy, so that's going to be some downstream effort
    # anyway).
    xycov_tran = 0.
    if xerr is not None:
        xycov_tran = A @ xerr @ A.T

    # additional covariance in target frame
    cov_extra = jnp.zeros((2,2))
    if fitvar:
        v_add = numpyro.sample("v_add", dist.LogUniform(1e-12,1e-3))
        cov_extra = jnp.array([[v_add,0.],[0., v_add]])

    cov_total = uerr + xycov_tran + cov_extra[None,:,:]
                          
    #pred_dist = dist.MultivariateNormal(upred, uerr + xycov_tran)
    pred_dist = dist.MultivariateNormal(upred, cov_total)

    # now the deltas
    with numpyro.plate("data", x.shape[0]):    
        numpyro.sample("u", pred_dist, obs=u)
    
def cdmatrix_from_pars(sx=1., rotdeg=0., skewdeg=0., r=1., ):

    """Utility - returns the cdmatrix from human-readable pars

    INPUTS

    sx = scale factor in x

    rotdeg = ccw rotation angle in degrees

    skewdeg = deviation from perpendicular of axes (0 means axes
    perpendicular)

    r = scale factor ratio: r = s_y / s_x

    """

    # The pieces
    sy = sx * r
    thetarad = np.radians(rotdeg)
    betarad = np.radians(skewdeg)

    b =  sx * np.cos(thetarad - 0.5 * betarad)
    c =  sy * np.sin(thetarad + 0.5 * betarad)
    e = -sx * np.sin(thetarad - 0.5 * betarad)
    f =  sy * np.cos(thetarad + 0.5 * betarad)

    return np.array([[b,c], [e,f]] )
    
def gendata(ndata=25, xsz=2., ysz=2., \
            thetadeg_true = 30., s_true=1.0e-2, \
            r_true = 1.0, betadeg_true = 0., \
            u0=0., v0=0., \
            sigu=1e-4, sigv=1e-4, \
            sigx=0.01, sigy=0.01, \
            perturb_xy=False, \
            showdata=True):

    """Generate the data.

    INPUTS

    ndata = number of points to generate

    xsz = full-range in X of generated data

    ysz = full-range in Y of generated data

    thetadeg_true = true rotation of transformation

    s_true = true scale factor (x -> u)

    r_true = true sy/sx ratio

    betadeg_true = true deviation from axis perpendicular

    u0 = offset in target frame, u[:,0]

    v0 = offset in target frame, u[:.1]

    sigu = uncertainty in u[:,0] as stddev

    sigv = uncertainty in u[:,1] as stddev

    sigx = uncertainty in x[:,0] as stddev

    sigy = uncertainty in x[:,1] as stddev

    perturbxy = perturb x as well as u

    showdata = plot the data before returning

    RETURNS

    x = [N,2] array of 'input' datapoints

    u = [N,2] array of transformed datapoints

    ucovs = [N,2,2] array of uncertainty covariances in the u frame

    xcovs = None, or [N, 2, 2] array of uncertainty covariances in the
    x frame

    """
    
    # uniform-random positions over the domain
    xgen = np.random.uniform(size=(ndata,2))-0.5
    xgen[:,0] *= xsz
    xgen[:,1] *= ysz

    # Transform these into the target frame. First build the cdmatrix
    Atrue = cdmatrix_from_pars(s_true, thetadeg_true, \
                               betadeg_true, r_true)

    ugen = np.einsum('jk,ik -> ij', Atrue, xgen)

    ugen[:,0] += u0
    ugen[:,1] += v0
    
    # now produce uncertainties and perturb with them. Refactored into
    # method getcovs() since we will likely use this more than
    # once.

    ucovs, pertn = getcovs(sigu, sigv, ndata)                              
    uobs = ugen + pertn

    # Initialize the output if not perturbing in the xy plane
    xobs = np.copy(xgen)
    xcovs = None

    # if perturbing in the xy plane, set up the perturbations and the
    # covariances (done differently from above. Oh well.
    if perturb_xy:
        xcovs, xpertn = getcovs(sigx, sigy, ndata)
        xobs = xgen + xpertn

        # check that the perturbation is working...
        xcovs_tran = Atrue @ xcovs @ Atrue.T

        print("DEBUG - TRANSF CHECK:")
        print(xcovs_tran[0])
        print(ucovs[0])

        # to check later: what happens when we draw samples from this
        # and overplot it? Does the result look sensibile?
        
    if not showdata:
        return xobs, uobs, ucovs, xcovs
    
    # it helps to show the actual data at this point...
    fig1 = plt.figure(1, figsize=(5,5))
    fig1.subplots_adjust(hspace=0.5, wspace=0.5, bottom=0.25)
    fig1.clf()
    ax1a = fig1.add_subplot(221)
    ax1b = fig1.add_subplot(222)
    ax1d = fig1.add_subplot(224)

    dum1a = ax1a.scatter(xobs[:,0], xobs[:,1], color='k', marker='o', \
                         label='Observed', s=2)
    dum1b = ax1b.scatter(ugen[:,0], ugen[:,1], color='b', marker='s', \
                         label='Target', s=2)

    dum1b2 = ax1b.scatter(uobs[:,0], uobs[:,1], color='b', marker='x', \
                         label='Perturbed', s=2)

    dum1d = ax1d.scatter(uobs[:,0]-ugen[:,0], uobs[:,1]-ugen[:,1], \
                         color='b', marker='s', s=1, alpha=0.5, \
                         label='Perturbations')

    # ditto observed frame if we perturbed them
    if xcovs is not None:
        ax1c = fig1.add_subplot(223)
        dum1c = ax1c.scatter(xobs[:,0]-xgen[:,0], xobs[:,1]-xgen[:,1], \
                             color='k', marker='s', s=1, alpha=0.5, \
                             label='Perturbations (obs)')
        ax1c.set_xlabel(r'$\Delta x$')
        ax1c.set_ylabel(r'$\Delta y$')
        leg1c = ax1c.legend()
        
        
    ax1a.set_xlabel('x')
    ax1a.set_ylabel('y')
    
    ax1b.set_xlabel('u')
    ax1b.set_ylabel('v')

    ax1d.set_xlabel(r'$\Delta u$')
    ax1d.set_ylabel(r'$\Delta v$')
    
    leg1a = ax1a.legend()
    leg1b = ax1b.legend()
    leg1d = ax1d.legend()

    return xobs, uobs, ucovs, xcovs


def getcovs(sigx=0., sigy=0., ndata=10, corxy=0.):

    """Utility - returns covariances given sigmas

    INPUTS

    sigx, sigy = scalars for stddev in x, y

    ndata = number of datapoints

    corxy = correlation between x, y

    RETURNS

    xycovs = [ndata, 2, 2] array of covariances

    xysamples = [ndata, 2] array of samples from the covariances

    """

    # This could be fed as arrays. For the moment let's just repeat
    # the single plane.
    xdev = np.repeat(sigx,  ndata)
    ydev = np.repeat(sigy,  ndata)
    rdev = np.repeat(corxy, ndata)

    # Generate covariances object out of this...
    Covs = covarsNx2x2.CovarsNx2x2(stdx=xdev, stdy=ydev, corrxy=rdev)

    # ... draw perturbations and get the covariances
    xypertns = Covs.getsamples()
    xycovs = Covs.covars

    return xycovs, xypertns
    
######## test routines follow

def test2par(ndata=25, true_params=[1.0e-2, 30.]):

    """Test the 2D version of our fitter"""

    # Generate data, accepting the defaults for the moment
    x, u, ucov, xcov = gendata(ndata, \
                               s_true=true_params[0],
                               thetadeg_true=true_params[1], \
                               showdata=False)

    # set up the sampler
    sampler = infer.MCMC(
        infer.NUTS(model_scalerot),
        num_warmup=2000,
        num_samples=2000,
        num_chains=2,
        progress_bar=True)

    t0=time.time()
    sampler.run(jax.random.PRNGKey(0), x, ucov, u=u)
    t1=time.time()

    print("Time elapsed sampling: %.2e seconds" % (time.time()-t0 ) )
    inf_data = az.from_numpyro(sampler)
    print(az.summary(inf_data))

    # We try our incantations for the corner plot as before
    samples = sampler.get_samples()
    chainz = np.vstack(( samples["s"], samples["theta"] )).T

    # angle computed in radians but specified in degrees. So convert
    # it...
    truths = np.copy(true_params)
    truths[1] = np.radians(true_params[1])
    
    fig2 = plt.figure(2, figsize=(5,5))
    fig2.clf()
    dum = corner.corner(chainz, labels=["s", "theta"], \
                        truths=truths[:], \
                        fig=fig2)


def test6term(ndata=25, \
              s_true=1.0e-2, rotdeg_true=30., \
              betadeg_true=2.0, r_true=1.0, \
              u0_true = 2.0e-3, \
              v0_true = 1.0e-3, \
              perturb_xy=False, \
              sigx=0.02, sigy=0.02, \
              fit_xy=False, \
              fit_var=False):

    """Tests the 6-term transformation sampler"""

    # generate the data
    x, u, ucov, xcov = gendata(ndata, \
                               s_true=s_true, \
                               thetadeg_true=rotdeg_true,\
                               r_true=r_true, \
                               betadeg_true=betadeg_true, \
                               u0=u0_true, \
                               v0=v0_true,\
                               perturb_xy=perturb_xy, \
                               sigx=sigx, sigy=sigy, \
                               showdata=True)   

    # set up the sampler
    sampler = infer.MCMC(
        infer.NUTS(model_6term),
        num_warmup=2000,
        num_samples=2000,
        num_chains=2,
        progress_bar=True)

    # run the sampler
    t0=time.time()

    # transform x uncertainty as well?
    xsend = None
    if fit_xy:
        xsend = jnp.array(xcov)
    
    sampler.run(jax.random.PRNGKey(1), x, ucov, u=u, xerr=xsend, \
                fitvar=fit_var)
    t1=time.time()

    print("Time elapsed sampling: %.2e seconds" % (time.time()-t0 ) )
    inf_data = az.from_numpyro(sampler)
    print(az.summary(inf_data))

    samples = sampler.get_samples()
    chainz = np.vstack(( samples["s"], samples["theta"], \
                         samples["r"], samples["beta"], \
                         samples["u0"], samples["v0"] )).T

    # some particulars for the corner plot
    corner_labels = ["s", r"$\theta$", r"$s_y/s_x$", r"$\beta$", \
                     r"u_0", r"v_0"]

    corner_truths = [s_true, np.radians(rotdeg_true), \
              r_true, np.radians(betadeg_true), \
                     u0_true, v0_true]
    
    # ugh
    if fit_var:
        chainz = np.vstack(( samples["s"], samples["theta"], \
                             samples["r"], samples["beta"], \
                             samples["u0"], samples["v0"], \
                             np.log10(samples["v_add"]) )).T
        
        corner_labels = ["s", r"$\theta$", r"$s_y/s_x$", r"$\beta$", \
                         r"u_0", r"v_0", r"$log_{10}(v_{add})$"]

        corner_truths.append(None)
    
    fig2 = plt.figure(2, figsize=(6,6))
    fig2.clf()
    dum = corner.corner(chainz, labels=corner_labels, \
                        truths=corner_truths, \
                        fig=fig2)

    fig2.subplots_adjust(bottom=0.15, left=0.15)

    #blah = plot_trace_dist(inf_data, var_names=["s","theta", \
    #                                            "r", "beta", \
    #                                            "u0", "v0" ])
