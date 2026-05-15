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
    theta = numpyro.sample("theta", dist.Uniform(-0.5*jnp.pi, 0.5*jnp.pi))
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
            sigu=1e-3, sigv=1e-3, \
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

    sigu = uncertainty in u[:,0] as stddev

    sigv = uncertainty in u[:,1] as stddev

    showdata = plot the data before returning

    RETURNS

    x = [N,2] array of 'input' datapoints

    u = [N,2] array of transformed datapoints

    ucovs = [N,2,2] array of uncertainty covariances in the u frame"""

    # uniform-random positions over the domain
    x = np.random.uniform(size=(ndata,2))-0.5
    x[:,0] *= xsz
    x[:,1] *= ysz

    # Transform these into the target frame. First build the cdmatrix
    Atrue = cdmatrix_from_pars(s_true, thetadeg_true, \
                               betadeg_true, r_true)

    ugen = np.einsum('jk,ik -> ij', Atrue, x)

    # now produce uncertainties and perturb with them. For starters,
    # we will assume diagonal uniformu uncertainties, to be relaxed
    # later if things actually work...
    pertn=np.random.normal(size=(ndata, 2))
    pertn[:,0] *= sigu
    pertn[:,1] *= sigv
                           
    uobs = ugen + pertn

    # now turn the uncertaintis into [N,2,2] covariance matrix
    Covs = covarsNx2x2.CovarsNx2x2(stdx=np.repeat(sigu, ndata), \
                                   stdy=np.repeat(sigv, ndata), \
                                   corrxy=np.zeros(ndata))

    ucovs = Covs.covars

    if not showdata:
        return x, uobs, ucovs
    
    # it helps to show the actual data at this point...
    fig1 = plt.figure(1, figsize=(7,2))
    fig1.subplots_adjust(hspace=0.5, wspace=0.5, bottom=0.25)
    fig1.clf()
    ax1a = fig1.add_subplot(131)
    ax1b = fig1.add_subplot(132)
    ax1c = fig1.add_subplot(133)

    dum1a = ax1a.scatter(x[:,0], x[:,1], color='k', marker='o', \
                         label='Observed', s=2)
    dum1b = ax1b.scatter(ugen[:,0], ugen[:,1], color='b', marker='s', \
                         label='Target', s=2)

    dum1b2 = ax1b.scatter(uobs[:,0], uobs[:,1], color='b', marker='x', \
                         label='Perturbed', s=2)

    dum1c = ax1c.scatter(uobs[:,0]-ugen[:,0], uobs[:,1]-ugen[:,1], \
                         color='b', marker='o', s=1, alpha=0.5, \
                         label='Perturbations')
    
    ax1a.set_xlabel('x')
    ax1a.set_ylabel('y')
    
    ax1b.set_xlabel('u')
    ax1b.set_ylabel('v')

    ax1c.set_xlabel(r'$\Delta u$')
    ax1c.set_ylabel(r'$\Delta v$')
    
    leg1a = ax1a.legend()
    leg1b = ax1b.legend()
    leg1c = ax1c.legend()

    return x, uobs, ucovs


######## test routines follow

def test2par(ndata=25, true_params=[1.0e-2, 30.]):

    """Test the 2D version of our fitter"""

    # Generate data, accepting the defaults for the moment
    x, u, ucov = gendata(ndata, \
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
