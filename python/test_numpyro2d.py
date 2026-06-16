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

# There will likely be quite a lot of duplication in these prototypes
# as features are added one by one. I think it will be more robust to
# develop a few methods rather than one with lots of options.

# Our usual imports
import time

import numpy as np
import matplotlib.pylab as plt
plt.ion()

import jax
jax.config.update("jax_enable_x64", True)
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

# useful for mixture model visualization
from matplotlib.collections import EllipseCollection, LineCollection
from matplotlib import colors as mpl_colors

# For dumping samples to disk while developing
import pickle

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

def model_2term_bells(x, uerr, u=None, xerr=None, fitvar=False):

    """Scale, rotation model but with some bells and whistles to test.

    INPUTS

    x = [N,2] = input positions

    uerr = [N,2,2] = input uncertainties as covariances

    u = [N,2] optional output positions

    xerr = [N,2,2] optional xy uncertainties as covariances

    fitvar = include diagonal covariance in model parameters
    

"""

    # Define the priors as numpyro distributions. 
    theta = numpyro.sample("theta", dist.Uniform(-1.0*jnp.pi, 1.0*jnp.pi))
    s = numpyro.sample("s", dist.LogUniform(1e-5,1.))

    # cdmatrix components, produce transformed positions
    b = numpyro.deterministic("b",  s * jnp.cos(theta))
    c = numpyro.deterministic("c",  s * jnp.sin(theta))
    e = numpyro.deterministic("e", -s * jnp.sin(theta))
    f = numpyro.deterministic("f",  s * jnp.cos(theta))
    
    A = jnp.array([[b,c],[e,f]])
    
    upred = jnp.einsum('jk,ik -> ij', A, x)

    xycov_tran = A * 0.
    if xerr is not None:
        xycov_tran = jnp.matmul(A, jnp.matmul(xerr, A.T))

    # additional covariance in target frame
    cov_extra = jnp.zeros((2,2))
    if fitvar:
        v_add = numpyro.sample("v_add", dist.LogUniform(1e-12,1e-3))
        cov_extra = jnp.array([[v_add,0.],[0., v_add]])
#        cov_total = uerr + cov_extra
#    else:
#        cov_total = uerr
        
    # Total covariance 
    cov_total = uerr + xycov_tran + cov_extra[None,:,:]


    # now the deltas
    with numpyro.plate("data", x.shape[0]):
        
        pred_dist = dist.MultivariateNormal(upred, cov_total)
        numpyro.sample("u", pred_dist, obs=u)

def model_2term_mixmod(x, uerr, u=None, xerr=None, fitvar=False, wid_u0=2.):

    """Fits mixture model to the positions, but does not fit individual star-by-star shifts. 


    INPUTS:

    x = [N,2] = input positions

    uerr = [N,2,2] = input uncertainties as covariances

    u = [N,2] optional output positions

    xerr = [N,2,2] optional xy uncertainties as covariances   

    fitvar = include diagonal covariance in model parameters

    wid_u0 = prior width for u0 parameters

"""

    # our two-term model again:
    theta = numpyro.sample("theta", dist.Uniform(-1.0*jnp.pi, 1.0*jnp.pi))
    s = numpyro.sample("s", dist.LogUniform(1e-5,1.))

    # 2026-06-11 widen the priors
    u0= numpyro.sample("u0", dist.Uniform(-wid_u0, wid_u0))
    v0= numpyro.sample("v0", dist.Uniform(-wid_u0, wid_u0))
    
    # transform the sampled parameters into matrix components
    b = numpyro.deterministic("b",  s * jnp.cos(theta))
    c = numpyro.deterministic("c",  s * jnp.sin(theta))
    e = numpyro.deterministic("e", -s * jnp.sin(theta))
    f = numpyro.deterministic("f",  s * jnp.cos(theta))
    
    A = jnp.array([[b,c],[e,f]])

    # We are going to have different offsets for the "foreground" and
    # "background" components. So:
    utran = jnp.einsum('jk,ik -> ij', A, x)

    upred_fg = utran + jnp.array([u0,v0])[None,:]

    # Propagate input-frame covariances via the model paramters
    xycov_tran = A * 0.
    if xerr is not None:
        xycov_tran = jnp.matmul(A, jnp.matmul(xerr, A.T))

    # allow an optional additional covariance to be applied to all the
    # objects
    cov_extra = jnp.zeros((2,2))
    if fitvar:
        v_add = numpyro.sample("v_add", dist.LogUniform(1e-12,1e-3))
        cov_extra = jnp.array([[v_add,0.],[0., v_add]])


    # star-by-star total covariance
    cov_total = uerr + xycov_tran + cov_extra[None,:,:]
    
    # Now for the mixture-relevant pieces. The "background" model
    # components (can do this with vectors later, which would allow
    # covariant priors)

    # foreground component. What we used to call "cov_extra" is now
    # the covariance of the foreground component. This is identified
    # with u_0, v_0 to avoid any identifiability problem.
    var_fg = numpyro.sample("var_fg", dist.LogUniform(1e-12,1e-3))
    cov_mixmod_fg =  jnp.array([[var_fg,0], [0,var_fg]])
    cov_total_fg = cov_total + cov_mixmod_fg[None,:,:]
    
    # background component (2026-06-13 narrowed prior and made this a
    # delta over u0, v0)
    u0_bg = numpyro.sample("u0_bg", dist.Uniform(-0.1, 0.1))
    v0_bg = numpyro.sample("v0_bg", dist.Uniform(-0.1, 0.1))

    # maybe this prior should be tightened to avoid piling up on one object
    #var_bg = numpyro.sample("var_bg", dist.LogUniform(1e-12,1e-3) )
    var_bg = numpyro.sample("var_bg", dist.LogUniform(1e-12,1e-2) )

    # Model background covariance. Try making this ALWAYS bigger than
    # the fg variance to assist with component identification...
    cov_mixmod_bg = jnp.array([[var_bg,0], [0,var_bg]])
    # cov_total_bg = cov_total  + cov_mixmod_bg[None,:,:]
    cov_total_bg = cov_total_fg  + cov_mixmod_bg[None,:,:]
    
    # predicted positions assuming assigned to bg component. NOTE that these are now smaller deltas over the foreground component.
    upred_bg = utran + jnp.array([u0_bg + u0,v0_bg + v0])[None,:]
    
    # The mixture components. Let Q be the outlier probability. Not
    # sure yet how to stop this going to 1/Npoints... try imposing a
    # lower and upper limit.
    #
    # 2026-06-12 try also enforcing Q as the majority component. This
    # has been the case for the simulations I have tried, but WATCHOUT...
    Q = numpyro.sample("Q", dist.Uniform(0.05, 0.95))
    mix = dist.Categorical(probs=jnp.array([Q, 1.0 - Q]))

    # The usual plate incantation
    with numpyro.plate("data", x.shape[0]):

        # foreground and background distances:
        dist_fg = dist.MultivariateNormal(upred_fg, cov_total_fg)
        dist_bg = dist.MultivariateNormal(upred_bg, cov_total_bg)

        # Create the mixture...
        mixture = dist.MixtureGeneral(mix, [dist_fg, dist_bg])

        # actually do the samples:
        y_ = numpyro.sample("obs", mixture, obs=u)

        # print("DEBUG: y_", y_.shape)
        
        # track the membership probabilities
        log_probs = mixture.component_log_probs(y_)
        p = numpyro.deterministic(
            "p", log_probs - \
            jax.nn.logsumexp(log_probs, axis=-1, keepdims=True) \
        )

        # It turns out numpyro lets you do this without filling the
        # screen with debug statements every time through. Some magic
        # to do with suppressing screen output (probably when the
        # progress bar is switched on)
        print("DEBUG: y_, log_probs, dist_fg.batch, dist_fg.event, dist_bg.batch, dist_bg.event, p", \
              y_.shape, log_probs.shape, \
              dist_fg.batch_shape, dist_fg.event_shape, \
              dist_bg.batch_shape, dist_bg.event_shape, \
              p.shape)
        #, \
        #      log_probs[0])

        # NOW FOR THE NEXT TEST - DO THE SAME FOR THE 1D CASE. THIS
        # WILL SHOW US WHAT DIMENSION LOG_PROBS TAKES.
        
        # jax.debug.print does access the values cleanly but is not
        # suppressed by the progress bar. Use with caution.
        # jax.debug.print("{x}", x=log_probs[0])
        
        
def model_2term_moves(x, uerr, u=None, xerr=None, fitvar=False):

    """Scale and rotation, plus object-by-object moves

    INPUTS

    x = [N,2] = input positions

    uerr = [N,2,2] = input uncertainties as covariances

    u = [N,2] optional output positions

    xerr = [N,2,2] optional xy uncertainties as covariances   

    fitvar = include diagonal covariance in model parameters


    """

    # Define the priors as numpyro distributions. 
    theta = numpyro.sample("theta", dist.Uniform(-1.0*jnp.pi, 1.0*jnp.pi))
    s = numpyro.sample("s", dist.LogUniform(1e-5,1.))

    # cdmatrix components, produce transformed positions
    b = numpyro.deterministic("b",  s * jnp.cos(theta))
    c = numpyro.deterministic("c",  s * jnp.sin(theta))
    e = numpyro.deterministic("e", -s * jnp.sin(theta))
    f = numpyro.deterministic("f",  s * jnp.cos(theta))
    
    A = jnp.array([[b,c],[e,f]])
    
    upred = jnp.einsum('jk,ik -> ij', A, x)
    
    # Propagating uncertainty from input frame as well?
    xycov_tran = A * 0.
    if xerr is not None:
        xycov_tran = jnp.matmul(A, jnp.matmul(xerr, A.T))
    
    # additional covariance in target frame
    cov_extra = jnp.zeros((2,2))
    if fitvar:
        v_add = numpyro.sample("v_add", dist.LogUniform(1e-12,1e-3))
        cov_extra = jnp.array([[v_add,0.],[0., v_add]])

    # Total covariance 
    cov_total = uerr + xycov_tran + cov_extra[None,:,:]

    # Hyper-parameters for the star-by-star shifts. Try a tight prior
    shift_centers = x * 0
    #shift_covars = jnp.array([[1.0e-5,0.], [0., 1.0e-5] ])
    shift_covars = jnp.array([[1.0e-6,0.], [0., 1.0e-6] ])

    
    # priors broader than about 1e-5 tend to run into problems,
    # possibly because that's about the same breadth as the entire
    # delta distribution.
    
    with numpyro.plate("data", x.shape[0]):
        # Now for the star-by-star shifts
        du = numpyro.sample("du", \
                            dist.MultivariateNormal(shift_centers, \
                                                    shift_covars) )

        utot = upred + du
        
        pred_dist = dist.MultivariateNormal(utot, cov_total)
        numpyro.sample("u", pred_dist, obs=u)

def model_2term_shift(x, uerr, u=None, xerr=None, fitvar=False):

    """Rotation, scale, and offset, plus (optionally) individual object
shifts as residuals. 

    INPUTS

    x = [N,2] = input positions

    uerr = [N,2,2] = input uncertainties as covariances

    u = [N,2] optional output positions

    xerr = [N,2,2] optional xy uncertainties as covariances   

    fitvar = include diagonal covariance in model parameters


    """

    # priors on bulk parameters as numpyro distributions
    theta = numpyro.sample("theta", dist.Uniform(-1.0*jnp.pi, 1.0*jnp.pi))
    s = numpyro.sample("s", dist.LogUniform(1e-5,1.))
    #u0= numpyro.sample("u0", dist.Uniform(-1.0, 1.0))
    #v0= numpyro.sample("v0", dist.Uniform(-1.0, 1.0))

    # 2026-06-11 try broadening the priors. Tried making the shift
    # [+1. +1] and the (mixture) trials pegged at +1, +1. Nice!! Let's
    # loosen this somewhat...
    u0= numpyro.sample("u0", dist.Uniform(-10.0, 10.0))
    v0= numpyro.sample("v0", dist.Uniform(-10.0, 10.0))

    
    # cdmatrix components, produce transformed positions
    b = numpyro.deterministic("b",  s * jnp.cos(theta))
    c = numpyro.deterministic("c",  s * jnp.sin(theta))
    e = numpyro.deterministic("e", -s * jnp.sin(theta))
    f = numpyro.deterministic("f",  s * jnp.cos(theta))
    
    A = jnp.array([[b,c],[e,f]])

    # transformation model prediction, now including offset
    upred = jnp.einsum('jk,ik -> ij', A, x) \
        + jnp.array([u0,v0])[None,:]

    # Now for the covariances
    xycov_tran = A * 0.
    if xerr is not None:
        xycov_tran = jnp.matmul(A, jnp.matmul(xerr, A.T))
    
    # additional covariance in target frame
    cov_extra = jnp.zeros((2,2))
    if fitvar:
        v_add = numpyro.sample("v_add", dist.LogUniform(1e-12,1e-3))
        cov_extra = jnp.array([[v_add,0.],[0., v_add]])

    # Total covariance 
    cov_total = uerr + xycov_tran + cov_extra[None,:,:]

    # Hyper-parameters for the star-by-star shifts. Try a tight prior
    shift_centers = x * 0
    shift_covars = jnp.array([[1.0e-5,0.], [0., 1.0e-5] ])

    # now the actual distribution
    with numpyro.plate("data", x.shape[0]):
        # Now for the star-by-star shifts
        du = numpyro.sample("du", \
                            dist.MultivariateNormal(shift_centers, \
                                                    shift_covars) )

        utot = upred + du
        
        pred_dist = dist.MultivariateNormal(utot, cov_total)
        numpyro.sample("u", pred_dist, obs=u)

def model_2term_mix(x, uerr, u=None, xerr=None, fitvar=False):

    """Scale, rotation, offset, mixture, individual moves

    INPUTS

    x = [N,2] = input positions

    uerr = [N,2,2] = input uncertainties as covariances

    u = [N,2] optional output positions

    xerr = [N,2,2] optional xy uncertainties as covariances   
    

    """

    # fitvar no longer does anything because there's already a
    # covariance parameter in the background component
    
    # priors on bulk parameters as numpyro distributions
    theta = numpyro.sample("theta", dist.Uniform(-1.0*jnp.pi, 1.0*jnp.pi))
    s = numpyro.sample("s", dist.LogUniform(1e-5,1.))
    #u0= numpyro.sample("u0", dist.Uniform(-1.0, 1.0))
    #v0= numpyro.sample("v0", dist.Uniform(-1.0, 1.0))

    # 2026-06-11 update the prior to make more flexible...
    u0= numpyro.sample("u0", dist.Uniform(-10.0, 10.0))
    v0= numpyro.sample("v0", dist.Uniform(-10.0, 10.0))

    
    # hyper-parameters for the star-by-star shifts
    shift_centers = x * 0
    shift_covars = jnp.array([[1.0e-5,0.], [0., 1.0e-5] ])
    
    # Prior on mixture fraction and "background" covariance scale. For
    # the moment we will keep all these things as scalars and wrap
    # them into jnp arrays, where indicated, below.
    Q = numpyro.sample("Q", dist.Uniform(0.0, 1.0))
    # var_bg = numpyro.sample("var_bg", dist.LogUniform(1e-12,1e-3))
    var_bg = 0.
    u0_bg = numpyro.sample("u0_bg", dist.Uniform(-1.0, 1.0))
    #v0_bg = numpyro.sample("v0_bg", dist.Uniform(-1.0, 1.0))

    v0_bg = 0.
    
    # 2026-05-21: set the offsets to zero for the moment
    #u0_bg = 0.
    #v0_bg = 0.
    
    ## COMPUTED MODEL PARAMETERS
    # cdmatrix components, produce transformed positions
    b = numpyro.deterministic("b",  s * jnp.cos(theta))
    c = numpyro.deterministic("c",  s * jnp.sin(theta))
    e = numpyro.deterministic("e", -s * jnp.sin(theta))
    f = numpyro.deterministic("f",  s * jnp.cos(theta))
    
    A = jnp.array([[b,c],[e,f]])

    # transformation model prediction, now including offset
    utran = jnp.einsum('jk,ik -> ij', A, x)
    upred = utran + jnp.array([u0,v0])[None,:]

    # Now for the covariances
    xycov_tran = A * 0.
    if xerr is not None:
        xycov_tran = jnp.matmul(A, jnp.matmul(xerr, A.T))
    
    # Total covariance (measurement in both frames, optionally)
    cov_fg = uerr + xycov_tran

    # Model parameters for background component
    cov_bg = cov_fg + \
        jnp.array([[var_bg,0],[0,var_bg]])[None,:,:]

    u_bg = utran + jnp.array([u0_bg, v0_bg])[None,:]

    # compute the distributions
    with numpyro.plate("data", x.shape[0]):

        # the star-by-star offsets
        du = numpyro.sample("du", \
                            dist.MultivariateNormal(shift_centers, \
                                                    shift_covars) )

        # the foreground and background predictions...
        u_fg = upred + du        
        u_bg = u_bg + du

        dist_fg = dist.MultivariateNormal(u_fg, cov_fg)
        dist_bg = dist.MultivariateNormal(u_bg, cov_bg)
        
        # Now form the mixture
        mix = dist.Categorical(probs=jnp.array([Q, 1.0 - Q]))
        mixture = dist.MixtureGeneral(mix, [dist_fg, dist_bg] )

        # This I think *should* compute the appropriate mixture model
        y_ = numpyro.sample("u", mixture, obs=u)

        # track the membership probabilities
        log_probs = mixture.component_log_probs(y_)
        numpyro.deterministic(
            "p", log_probs - \
            jax.nn.logsumexp(log_probs, axis=-1, keepdims=True) \
        )
        
def model_6term(x, uerr, u=None, xerr=None, fitvar=False):

    """Offset and general linear transformation, parameterized in human
terms

INPUTS:

    x = [N,2] = input positions

    uerr = [N,2,2] = input uncertainties as covariances

    u = [N,2] optional output positions

    xerr = [N,2,2] optional xy uncertainties as covariances

    fitvar = include diagonal covariance in model parameters

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
        # xycov_tran = A @ xerr @ A.T

        # 2026-05-17: using jnp.matmul because I think this forces the
        # output to be jnp arrays (what was here before should be OK
        # as long as the inputs are already jnp arrays).
        xycov_tran = jnp.matmul(A, jnp.matmul(xerr, A.T))
        
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

def vortex_matrices(uv=None, \
                    rotdeg_in=0., rotdeg_out=0., xc=0., yc=0., \
                    r_out=None, rpow=1., scal=1.):

    """Radius-dependent rotation, by offset from centroid.

    INPUTS

    uv = [N,2] array of positions to transform

    rotdeg_in = position angle at r=0 from center

    rotdeg_out = position angle at maximum radius rmax

    xc, yc = x, y coordinates of center

    r_out = outer radius (for calibrating the rotation: taken from the
    data if not specified)

    rpow = power for position angle growth with radius, in the sense:

         rotdeg = rotdeg_in + (rotdeg_out - rotdeg_in) * (r/r_out)**rpow

    scal = 1. = any scaling we want to apply here

    RETURNS
   
    AMat = [N, 2, 2] stack of transformation matrices

    """

    # Initialize the return
    amat = None

    if np.size(uv) < 1:
        return amat

    npoints = uv.shape[0]
    if npoints < 1:
        return amat
    
    # displacements from rotation center
    uv_cen = np.array([xc, yc])
    r = np.sqrt(np.sum(( uv - uv_cen[None,:])**2, axis=1))

    if r_out is None:
        r_out = np.max(r)

    # now compute the position angles in degrees
    if rpow >= 0:
        rotdeg = rotdeg_in + (rotdeg_out - rotdeg_in) * \
            (r/r_out)**rpow
    else:
        # Slightly more of a hassle if the power is negative. We are
        # finding alpha, x0 in
        #
        # y = alpha*(x-x0)**beta
        f = (rotdeg_out/rotdeg_in)**(1/rpow)
        x0 = r_out/(1-f)
        lnalpha = np.log(rotdeg_out) - rpow * np.log(r_out - x0)
        alpha = np.exp(lnalpha)

        rotdeg = alpha * (r-x0)**rpow

    # now form the rotation matrix as appropriate for each object
    rotrad = np.radians(rotdeg)
    b =  scal * np.cos(rotrad)
    c =  scal * np.sin(rotrad)
    e = -scal * np.sin(rotrad)
    f =  scal * np.cos(rotrad)

    # form the matrix stack out of this
    amat = np.zeros((npoints, 2, 2))

    amat[:,0,0] = b
    amat[:,0,1] = c
    amat[:,1,0] = e
    amat[:,1,1] = f

    return amat, r, rotdeg

def apply_vortex(uv=None, amat=None, uv_cen=None):

    """Applies the offset-dependent rotation to input positions.

    INPUTS

    uv = [N,2] array of input positions

    amat = [N,2,2] array of transformation matrices

    uv_cen = [2] = rotation center about which to apply

    RETURNS

    uv_transf = transformed points."""

    if uv is None:
        return None

    # Returns a copy of the input if no rotation specified
    if amat is None:
        return np.copy(uv)

    if uv_cen is None:
        uv_cen = np.zeros(2)

    # Offset from rotation center
    du = uv - uv_cen[None,:]
    
    uv_eval =  np.einsum('ijk,ik->ij',amat, du) + uv_cen[None,:]

    return uv_eval
    
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

    ugen = transformed xy positions before addition of perturbations
    in the uv frame

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
        return xobs, uobs, ucovs, xcovs, ugen
    
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
        #leg1c = ax1c.legend()
        ax1c.set_title('Inp perturb')
        
    ax1a.set_xlabel('x')
    ax1a.set_ylabel('y')
    
    ax1b.set_xlabel('u')
    ax1b.set_ylabel('v')

    ax1d.set_xlabel(r'$\Delta u$')
    ax1d.set_ylabel(r'$\Delta v$')

    # Legends are annoying here. Show the titles instead.
    #leg1a = ax1a.legend()
    #leg1b = ax1b.legend()
    #leg1d = ax1d.legend()

    ax1a.set_title('Input')
    ax1b.set_title('Output')
    ax1d.set_title('Out perturbed')

    return xobs, uobs, ucovs, xcovs, ugen

def clumps_du(uv=None, fracs=None, cens_u=None, cens_v=None, \
              sigs_u=None, sigs_v=None, corxy_uv=None):

    """Generates clumps in [du, dv] space, inheriting uv from input.

    INPUTS

    uv = [N,2] unperturbed datapoints

    fracs = [M] - fractions of the sample in each clump. E.g. for two
    clumps, at 15% and 20% of the sample each, this would be [0.15,
    0.20].

    cens_u = [M] - centroids in u

    cens_v = [M] - centroids in v

    sigs_u = [M] - stddevs in u of clumps

    sigs_v = [M] - stddevs in v of clumps. Defaults to sigs_u if not
    set

    corxy_uv = [M] - xy correlations in u, v of clumps.

    RETURNS

    perts_clumps = [N, 2] array of perturbations

    which_clump = [N] array giving ID of each clump (-1 means not assigned to any clump)

    """
    
    if uv is None or fracs is None or sigs_u is None:
        print("clumps_du WARN - at least one of uv, fracs, sigs_u is None")
        return None, None

    mclumps = len(fracs)
    if mclumps < 1:
        return None, None

    # Accept scalar covariance info for the clump (and replicate to
    # the other entries if so). Note that np.isscalar(None) = False.
    if np.isscalar(sigs_u):
        sigs_u = np.repeat(sigs_u, mclumps)
    if np.isscalar(sigs_v):
        sigs_v = np.repeat(sigs_v, mclumps)
    if np.isscalar(corxy_uv):
        corxy_uv = np.repeat(corxy_uv, mclumps)

    if np.size(sigs_v) < np.size(sigs_u) or sigs_v is None:
        sigs_v = np.copy(sigs_u)

    if np.size(corxy_uv) < np.size(sigs_u) or corxy_uv is None:
        corxy_uv = np.zeros(np.size(sigs_u))
        
    # Ensure the clump centers and sigmas are filled. We might
    # concievably use this to put clumps on top of each other, so
    # deficient input dimensions is not actually an error. But at
    # least tell the user what is going on...
    if np.size(cens_u) < mclumps or cens_u is None:
        print("clumps_du INFO: %i centers supplied < %i fractions. All clumps defaulting to [0,0]." % (np.size(cens_u), mclumps) )

        duv = np.zeros((mclumps, 2))

    else:
        duv = np.column_stack(( cens_u, cens_v ))
    
    # convert clump fractions into array indices in input
    # data. Calling lsor [0:linds[0], linds[0]:linds[1],
    # linds[1]:linds[2], etc., should pull out a random sample by
    # fraction as we want.
    ndata = uv.shape[0]
    lsor = np.argsort(np.random.uniform(size=ndata))
    linds = np.asarray(np.cumsum(fracs)*uv.shape[0],'int')

    # Initialize the perturbations...
    perts_clumps = uv*0.

    # ... and populate them
    which_clump = np.zeros(ndata)-1
    bclump = np.repeat(False, ndata)
    
    ilo = 0
    for iclump in range(mclumps):
        ihi = np.copy(linds[iclump])

        # indices to assign to this clump
        lthis = lsor[ilo:ihi]
        
        which_clump[lthis] = iclump
        bclump[lthis] = True
        
        # Generate perturbations about 0, 0...
        covs, perts = getcovs(sigs_u[iclump], sigs_v[iclump], \
                              ndata, corxy_uv[iclump])

        # ... add the centroid onto the perturbations
        perts += duv[iclump][None,:]
        perts_clumps[lthis] = perts[lthis]
        
        ilo = np.copy(ihi)

    return perts_clumps, which_clump
        
def getcovs(sigx=0., sigy=0., ndata=10, corxy=0.):

    """Utility - returns covariances and samples from the covariances,
given sigmas

    INPUTS

    sigx, sigy = scalars or 1D arrays for stddev in x, y

    ndata = number of datapoints. If sigx is 1D array, input ndata is
    ignored in favor of the size of sigx.

    corxy = correlation between x, y (scalar or 1d array)

    RETURNS

    xycovs = [ndata, 2, 2] array of covariances

    xysamples = [ndata, 2] array of samples from the covariances

    """

    # Accept scalar or vector input. This may do something unexpected
    # if the inputs are a mixture of scalar and vectors.
    if np.size(sigx) < 2:
        xdev = np.repeat(sigx,  ndata)
        ydev = np.repeat(sigy,  ndata)
        rdev = np.repeat(corxy, ndata)
    else:

        # The lines below assume sigx is a 1D array.
        ndata = np.shape(sigx)[0]
        xdev = sigx
        if np.size(sigy) != np.size(xdev):
            ydev = np.copy(xdev)
        else:
            ydev = sigy

        if np.size(corxy) != np.size(xdev):
            rdev = np.zeros(ndata)
        else:
            rdev = corxy
    

    # Generate covariances object out of this...
    Covs = covarsNx2x2.CovarsNx2x2(stdx=xdev, stdy=ydev, corrxy=rdev)

    # ... draw perturbations and get the covariances
    xypertns = Covs.getsamples()
    xycovs = Covs.covars

    return xycovs, xypertns

def cdmatrices_from_samples(dsamples={}):

    """Returns cdmatrix from dsamples {b,c,e,f}

    INPUTS:

    dsamples = dictionary of samples.

    RETURNS:

    A = [N,2,2] array of cdmatrices"""

    if dsamples is None:
        return None
    
    if not 'b' in dsamples.keys():
        print("cdmatrices_from_samples WARN - key not in samples: b.")
        return None

    npoints = dsamples['b'].shape[0]
    A = np.zeros((npoints,2,2))
    A[:,0,0] = dsamples['b']
    A[:,0,1] = dsamples['c']
    A[:,1,0] = dsamples['e']
    A[:,1,1] = dsamples['f']

    return A

def centroids_from_samples(dsamples={}, key_u0="u0", key_v0="v0"):

    """Returns [N,2] array of centroids from samples dictionary

    INPUTS

    dsamples = dictionary of samples

    key_u0 = key for centroid u

    key_v0 = key for centroid v

    RETURNS

    u = [N,2] = array of centroids

    """

    for key in [key_u0, key_v0]:
        if not key in dsamples.keys():
            return None

    return np.column_stack(( dsamples[key_u0], dsamples[key_v0] ))
    
    
def correction_matrices(A=None, dsamples={}):

    """Returns transformation correction matrices to undo the median frame
transformation and apply the individual transformation for each
sample, to support assessment of mixture models.

    INPUTS:

    A = [N,2,2] stack of transformation matrices

    dsamples = dictionary of samples. Used if A=None.


    RETURNS:

    AAinv = [N,2,2] transformation A.A_med^-1

    A_med = [2,2] median transformation matrix

    """

    # Implementation comment: being a rank-ordering, the 1D median is
    # invariant under monotonic reparameterization. So I think it
    # makes sense to take the median {b,c,e,f} rather than reaching
    # back to the {s, theta, etc.}. The latter is correct, the former
    # is more flexible and I think OK in practice (the median is not
    # always the same as the mode in either case).

    # Can supply the [N,2,2] stack of matrices or re-lift it from the
    # dictionary
    if A is None:
        A = cdmatrices_from_samples(dsamples)

    # If by this point we still have no transformations, we cannot
    # proceed
    if A is None:
        return None, None

    # Take the median and invert it
    A_med = np.median(A, axis=0)
    Ainv_med = np.linalg.inv(A_med)

    # compute A.<A>^{-1}, plane by plane.
    AAinv = np.matmul(A, Ainv_med[None,:,:])
   
    return AAinv, A_med



def ellipsepars_from_covars(covars=None, exagfac=1.):

    """Utility - given a CovarsNx2x2 object, returns the parameters needed
for a matplotlib ellipse collection.

    INPUTS
    ======

    covars = covarsNx2x2 object

    exagfac = exaggeration factor. Widths are scaled by this for
    visual display.

    RETURNS
    =======
    
    ww = [N] widths for ellipses

    hh = [N] heights for ellipses

    posans = [N] position angles for ellipses

    """

    if covars is None:
        return None, None, None

    # ensure the eigenvalues and vectors are populated
    covars.eigensFromCovars()

    ww = covars.majors**0.5 * exagfac * 2.0
    hh = covars.minors**0.5 * exagfac * 2.0
    return ww, hh, covars.rotDegs

def show_du(samples={}, keypos='u_tran', \
            ucolor='k', errcolor='0.25', \
            pcolor='g', alpha=0.4, \
            show_std=True, fshow = 1.0, \
            subset_name=None, \
            u0_truth=0.25, v0_truth=0.25, \
            debug=False):

    """Utility: shows the samples in du

    subset = keyname of subset

    u0_truth, v0_truth = input values of pointing

    debug = print helpful messages to screen"""

    if len(samples.keys()) < 1:
        return

    # The star-by-star samples: [nsamples stars, 2]
    du = samples['du']

    # should be a [nsamples, nstars, 2] array.
    du_med = np.median(du, axis=0)
    du_std = np.std(du, axis=0)

    # To put du in context, we also need to see the rest of the
    # model. So get that here.
    u0 = centroids_from_samples(samples, 'u0', 'v0')
    A = cdmatrices_from_samples(samples)
    x = samples['x']
    u_obs = samples['u_obs']
    
    # Predicted position set for every sample
    if debug:
        print("show_du INFO - sample shapes:")
        print("show_du INFO - x:", x.shape)
        print("show_du INFO - A:", A.shape)
        print("show_du INFO - u0:", u0.shape)
        print("show_du INFO - du:", du.shape)
        print("show_du INFO - u_obs:", u_obs.shape)

    # multiply row by row
    print("show_du INFO - re-projecting predictions...")
    t00 = time.time()
    upred_samples = np.einsum('ijk,lk->ilj',A, x)
    print("show_du INFO - done einsum in %.2e seconds" % (time.time()-t00))

    # now add on the deltas
    upred_total = upred_samples + du + u0[:,None,:]

    # Form statistics on THESE. Our covsNx2x2 object does this,
    # but expects [nsamples 2, ndata]. So we transpose first.
    print("show_du INFO - computing covars...")
    t01 = time.time()
    sampls = np.transpose(upred_total, axes=(0,2,1))
    Covs = covarsNx2x2.CovarsNx2x2(xysamples=sampls)
    print("show_du INFO - ... done in %.2e seconds" % \
          (time.time() - t01))
    if debug:
        print("show_du INFO - upred_total:", upred_total.shape)
        print("show_du INFO - sampls shape:", sampls.shape)
        print("show_du INFO - covars shape:", Covs.covars.shape)


    # median along the samples of the upred
    upred_med = np.median(upred_total, axis=0)

    # The deltas about predictions...
    delt_u = u_obs[None,:,:] - upred_total

    dutotal_med = np.median(delt_u, axis=0)
    dutotal_std = np.std(delt_u, axis=0)

    if debug:
        print("show_du INFO: delt_u:", delt_u.shape)
        print("DEBUG:", np.median(dutotal_std, axis=0) )
    
        # is that at all sensible??
        isho = 10
        lsho = 100
        udum = np.matmul(A[isho],x[lsho])
        print("MULTIUPLY DEBUG:")    
        print("MULTIPLY DEBUG: einsum: samples[%i,%i]:" % \
              (isho, lsho), upred_samples[isho, lsho])
        print("MULTIPLY DEBUG: direct: A[%i].x[%i]:   " % \
              (isho, lsho), udum)

        print("show_du DBG: upred_total[%i,%i]: " % \
              (isho, lsho), upred_total[isho, lsho])
    
        print("show_du INFO - upred_samples:", upred_samples.shape)
        print("show_du INFO - upred_total:", upred_total.shape)
    
        print("show_du INFO - upred_med:", upred_med.shape)
    
    
    # The bulk-offset samples: [nsamples, 2]
    #
    # update 2026-06-11 : NO - THIS IS ALREADY INCLUDED IN THE MODEL!!
    #if 'u0' in samples.keys() and 'v0' in samples.keys():
    #    shift = np.vstack(( samples['u0'], samples['v0'] )).T
    #    du_all = du + shift[:,None,:]

    # print("Including frame shift:")
    #    du_med = np.median(du_all, axis=0)
    #    du_std = np.std(du_all, axis=0)

    #    # consider the u0, v0 model parameters

    # set up the figure
    fig4 = plt.figure(4, figsize=(10,6))
    fig4.clf()

    # allow plotting a subset so that we can get into the dense areas
    bsho = np.repeat(True, du_med.shape[0])
    if fshow < 1.0:
        bsho = np.random.rand(du_med.shape[0]) <= fshow

    # ... or we can quote by subset
    if subset_name in samples.keys():
        if subset_name.find('b_') == 0:
            bsho = samples[subset_name]

            print("using subset %s: %i" % (subset_name, np.sum(bsho)))

            if np.sum(bsho) <1:
                print("No entries in this subset! Nothing to plot...")
                return

    # 2026-06-15 I have got tangled in definitions of which delta is
    # which... Go back and overplot the predictions against the
    # observations. That will show if the model plus delta is bringing
    # about an offset!
            
    ax41 = fig4.add_subplot(231)
    if show_std:

        # 2026-06-15 commented out while testing overplot of positions
        #dum41 = ax41.errorbar(du_med[bsho,0], du_med[bsho,1], \
        #                      #yerr=du_std[bsho,1], xerr=du_std[bsho,0], \
        #                      xerr=dutotal_std[bsho,0], \
        #                      yerr=dutotal_std[bsho,1], \
        #                      fmt='.', alpha=alpha, ms=4, capsize=2, \
        #                      color=ucolor, ecolor=errcolor, zorder=10)

        dum41 = ax41.errorbar(upred_med[bsho,0], upred_med[bsho,1], \
                              #yerr=du_std[bsho,1], xerr=du_std[bsho,0], \
                              xerr=dutotal_std[bsho,0], \
                              yerr=dutotal_std[bsho,1], \
                              fmt='.', alpha=alpha, ms=4, capsize=2, \
                              color=ucolor, ecolor=errcolor, zorder=10)

        
    else:
        dum41 = ax41.scatter(dutotal_med[bsho,0], \
                             dutotal_med[bsho,1], \
                             marker='.', alpha=alpha, s=6,\
                             c=ucolor, zorder=10)


    # test our ellipses
    ww, hh, posans = ellipsepars_from_covars(Covs, 1.0)
    facecolorEllipse='b'
    alphaEllipse=0.05
    facergbaEllipse = mpl_colors.to_rgba(c=facecolorEllipse, \
                                         alpha=alphaEllipse)
    ecc = EllipseCollection(ww[bsho], hh[bsho], posans[bsho], \
                            units='xy', \
                            #offsets=dutotal_med[bsho], \
                            offsets=upred_med[bsho], \
                            transOffset=ax41.transData, \
                            edgecolor='b', \
                            #facecolor=None, \
                            facecolor=facergbaEllipse, \
                            zorder=5)
    ax41.add_collection(ecc)

    if debug:
        print("show_du DEBUG - SANITY CHECK:")
        dumdelt = samples['u_obs'] - upred_med
        vardelt = np.cov(dumdelt, rowvar=False)
        w,v = np.linalg.eig(vardelt)
        print("show_du DEBUG - obs - pred:", vardelt)
        print("show_du DEBUG - eigenvalues:", w)
    
    # If we have the commanded perturbations, show them too
    pert = None
    if 'u_obs' in samples.keys() and 'u_tran' in samples.keys():
        pert = samples['u_obs'] - samples['u_tran']
        # pert = u_obs - upred_med # THIS matches du very well. 
        #pert = samples['perts_u']
        pert = samples['u_obs']
        dum41_2 = ax41.scatter(pert[bsho,0], pert[bsho,1], \
                               alpha=alpha, c=pcolor, \
                               zorder=7, s=16)

        print("show_du INFO - median offset in plotted shifts: %.2e, %.2e" \
              % (np.median(pert[bsho,0]-du_med[bsho,0]), \
                 np.median(pert[bsho,1]-du_med[bsho,1]) ) )

        meanshift = np.mean(samples['u_obs'] - samples['u_tran'], axis=0)
        print("show_du INFO - mean offset in simulated shifts: %.2e, %.2e" \
              % (meanshift[0], meanshift[1]))
        
    #ax41.set_xlabel(r"$\Delta u$")
    #ax41.set_ylabel(r"$\Delta v$")

    ax41.set_xlabel(r"$u$")
    ax41.set_ylabel(r"$v$")
    ax41.set_title(r'"Observed" and sampled')
    
    # Add an entire third axis to show the deltas. Note that we
    # subtract u[tran] from BOTH (these are deltas from the
    # obs). Currently written in the order this occurred to me, to be
    # cleaned up later. So:
    ax43 = fig4.add_subplot(234)
    dum43_1 = ax43.errorbar(upred_med[bsho,0]-samples['u_tran'][bsho,0], \
                            upred_med[bsho,1]-samples['u_tran'][bsho,1], \
                            #yerr=du_std[bsho,1], xerr=du_std[bsho,0], \
                            xerr=dutotal_std[bsho,0], \
                            yerr=dutotal_std[bsho,1], \
                            fmt='.', alpha=alpha, ms=4, capsize=2, \
                            color=ucolor, ecolor=errcolor, zorder=10)

    dum3_2 = ax43.scatter(pert[bsho,0] - samples['u_tran'][bsho,0], \
                           pert[bsho,1] - samples['u_tran'][bsho,1], \
                           alpha=alpha, c=pcolor, \
                           zorder=7, s=16)

    
    ec3 = EllipseCollection(ww[bsho], hh[bsho], posans[bsho], \
                            units='xy', \
                            #offsets=dutotal_med[bsho], \
                            offsets=upred_med[bsho] - samples['u_tran'][bsho], \
                            transOffset=ax43.transData, \
                            edgecolor='b', \
                            #facecolor=None, \
                            facecolor=facergbaEllipse, \
                            zorder=5)
    ax43.add_collection(ec3)
    

    ax43.set_xlabel(r"$u - u_{tran}$")
    ax43.set_ylabel(r"$v - v_{tran}$")
    ax43.set_title(r"Sampled model incl. $\Delta \vec{u}$")
    
    # Now, how do just the straight deltas look?
    ax44 = fig4.add_subplot(235)
    dum44_1 = ax44.errorbar(du_med[bsho,0], du_med[bsho,1],\
                            yerr=du_std[bsho,1], xerr=du_std[bsho,0], \
                            fmt='.', alpha=alpha, ms=4, capsize=2, \
                            color=ucolor, ecolor=errcolor, zorder=10)

    for ax in [ax43, ax44]:
        blah44 = ax.axvline(0., zorder=20, color='#FFCB05', alpha=0.7)
        blah44 = ax.axhline(0., zorder=20, color='#FFCB05', alpha=0.7)
    
    ax44.set_title(r'Sampled $\Delta \vec{u}$ only')
    ax44.set_xlabel(r'$\Delta u$')
    ax44.set_ylabel(r'$\Delta v$')

    # median
    med_du_sho = np.median(du_med, axis=0)
    smed_sho_du = r'$<\Delta \vec{u}>$=[%.2e, %.2e]' \
        % (med_du_sho[0], med_du_sho[1])
    dum44 = ax44.annotate(smed_sho_du, (0.96,0.96), \
                          xycoords='axes fraction', \
                          ha='right', va='top', fontsize=8, \
                          backgroundcolor='w', zorder=50, \
                          alpha=0.8)
    
    # Show the u0, v0. We will want to bring in the truth parameters
    # in the samples, but atm that's a task for later...
    u0v_truth = np.array([u0_truth, v0_truth])
    med_du0_sho = np.median(u0 - u0v_truth[None,:], axis=0)
    smed_sho_du0 = r'$<\Delta \vec{u}_0>$=[%.2e, %.2e]' \
        % (med_du0_sho[0], med_du0_sho[1])
    
    fsampl = 0.1 # show this frawcwtion of the samples
    bsampl = np.random.rand(u0.shape[0]) <= fsampl
    ax45 = fig4.add_subplot(236)
    dum45_1 = ax45.scatter(u0[bsampl,0], u0[bsampl,1], \
                           s=1, alpha=0.5, color=ucolor, zorder=10)

    dum45_2 = ax45.axvline(u0_truth, zorder=20, color='#FFCB05', alpha=0.7)
    dum45_2 = ax45.axhline(v0_truth, zorder=20, color='#FFCB05', alpha=0.7)

    dum45 = ax45.annotate(smed_sho_du0, (0.96,0.96), \
                          xycoords='axes fraction', \
                          ha='right', va='top', fontsize=8, \
                          backgroundcolor='w', zorder=50, \
                          alpha=0.8)

    
    ax45.set_xlabel(r'$u_0$')
    ax45.set_ylabel(r'$v_0$')
    ax45.set_title(r'Sampled [$u_0, v_0$]')
    
    # Do we have the base points for quiver plot?
    if not keypos in samples.keys():
        return

    uo = samples[keypos]
    
    ax42 = fig4.add_subplot(232)
    dum_42 = ax42.quiver(uo[:,0], uo[:,1], \
                         du_med[:,0], du_med[:,1], \
                         color=ucolor, zorder=10, alpha=alpha)

    # if we have it, overplot the commanded perturbations
    if pert is not None:
        dum_42_b = ax42.quiver(uo[:,0], uo[:,1], \
                               pert[:,0] - samples['u_tran'][:,0], \
                               pert[:,1] - samples['u_tran'][:,1], \
                               color=pcolor, zorder=20, \
                               alpha=alpha)

    
    ax42.set_xlabel(r"$u$")
    ax42.set_ylabel(r"$v$")
    ax42.set_title(r'"Observed" and sampled')

    
    # cosmetics
    fig4.subplots_adjust(bottom=0.15, left=0.12, hspace=0.35, wspace=0.4)

    fig4.savefig('test_postdeltas.png')
    
def show_samples(dsamples={}, ellipses=True, n_ellipses=50, \
                 cmap='plasma_r', extralog=False):

    """One-liner to show some of the results from an MCMC run.

    INPUTS

    dsamples = dictionary of samples

    ellipses = call our prototype ellipse plotter

    n_ellipses = number of ellipses to draw

    cmap = colormap for quiver plot color coded by membership
    probability

    extralog = do log10(log(pmem))?

    """

    # Construct the linear transformation from the samples
    A = cdmatrices_from_samples(dsamples)
    if A is None:
        return

    # Correction matrices for use later when plotting ellipses
    AAinv, Amed = correction_matrices(A)

    # Centroids
    u0 = centroids_from_samples(dsamples, 'u0', 'v0')
    u0_bg = centroids_from_samples(dsamples, 'u0_bg', 'v0_bg')
    
    var_bg = None
    if "var_bg" in dsamples.keys():
        var_bg = dsamples["var_bg"]

    # Mixture fractions...
    Q = None
    if "Q" in dsamples.keys():
        Q = dsamples['Q']

    # ... and membership probabilities
    p = None
    if 'p' in dsamples.keys():
        p = dsamples['p']
        
    # Simulated data
    x = None
    if "x" in dsamples.keys():
        x = dsamples['x']

    u_obs = None
    if 'u_obs' in dsamples.keys():
        u_obs = dsamples['u_obs']

    u_tran = None
    if 'u_tran' in dsamples.keys():
        u_tran = dsamples['u_tran']

    # Number of points in the dataset
    ndata = u_obs.shape[0]

    #print("Membership probabilities:",p.shape)
    #print("Mixture fractions:", Q.shape)
    #print("Ndata:", ndata)
    #print("CDMATRIX shape:", A.shape)

    # Apply the median transformation here INCLUDING THE OFFSET
    u0med = np.median(u0, axis=0)
    Amed = np.median(A, axis=0)
    # upred_med = np.einsum('jk,ik -> ij', Amed, x)

    upred_med = np.einsum('jk,ik -> ij', Amed, x) + u0med[None,:]

    print("show_samples INFO:", u0med)
    
    # deltas
    uresid_med = u_obs - upred_med

    print("show_samples DBG:")
    print(A.shape, Amed.shape, upred_med.shape, u_obs.shape)
    
    # Mean probabilities
    pmem = np.median(p[...,0],axis=0)
    ## print("pmem", pmem.shape)

    #### FOUR-PANEL PLOT WITH RESIDUALS AND ELLIPSES
    fig6 = plt.figure(6)
    fig6.clf()
    ax61 = fig6.add_subplot(221)
    ax62 = fig6.add_subplot(222)
    ax63 = fig6.add_subplot(223)
    ax64 = fig6.add_subplot(224)

    dum = ax61.hist(np.log10(Q), bins=100, alpha=0.5, zorder=10)
    #blah = ax61.axvline(np.log10(1.0/ndata), ls='--', color='k', \
    #                    zorder=20, label=r'$\log_{10}(N_{\rm data}^{-1})$')
    ax61.set_xlabel(r'$\log_{10}Q$')
    # leg = ax61.legend()

    # predicted
    dum62 = ax62.scatter(upred_med[:,0], upred_med[:,1], \
                         s=16, c='b')

    dum63 = ax63.hist(pmem, alpha=0.5, color='g')
    
    # residuals
    dum64 = ax64.scatter(uresid_med[:,0], uresid_med[:,1], \
                         s=4, c='k', zorder=30)

    
    fig6.subplots_adjust(hspace=0.3, wspace=0.3)

    # Draw a list of indices to send to the ellipse plotter, to be the
    # same for both ellipse sets
    npoints = A.shape[0]
    ndraw = min([n_ellipses, npoints])
    rng = np.random.default_rng()
    lellipse = rng.choice(npoints, ndraw, replace=False)
    
    # foreground
    show_ellipses(dsamples, ax=ax64, fig=fig6, \
                  key_cen_u='u0', key_cen_v='v0', \
                  key_var_u='var_fg', \
                  edgecolorEllipse='#00274C', \
                  facecolorEllipse='#FFCB05', \
                  zorder=25, \
                  which_samples=lellipse, \
                  AAinv=AAinv)

    show_ellipses(dsamples, ax=ax64, fig=fig6, \
                  key_cen_u='u0_bg', key_cen_v='v0_bg', \
                  key_var_u='var_bg', \
                  edgecolorEllipse='#9A3324', \
                  facecolorEllipse='#702082', \
                  alphaEllipse=0.01, \
                  zorder=15, \
                  which_samples=lellipse, \
                  AAinv=AAinv)

    fig6.savefig('test_mixmod_ellipses.png')

    #### QUIVER PLOT COLOR CODED BY MEDIAN MEMBERSHIP PROBABILITY
    fig8=plt.figure(8, figsize=(10,4))
    fig8.clf()
    ax81=fig8.add_subplot(121)    
    ax82=fig8.add_subplot(122)

    # plot u_tran unless not present, in which case use u_obs
    if u_tran is not None:
        u_sho = u_tran
    else:
        u_sho = u_obs

    # shading
    shade = np.copy(pmem)
    label_pmem = 'ln(pmem)'

    # formal membership probabilities can be REALLY small.
    if extralog:
        shade = np.sign(pmem) * np.log10(np.abs(pmem-1.))
        label_pmem = r'log$_{10}$(|ln(pmem)-1.|)'

    # cmap = 'plasma'
        
    dum_82 = ax82.quiver(u_sho[:,0], u_sho[:,1], \
                         uresid_med[:,0], uresid_med[:,1], \
                         shade, cmap=cmap)

    # scatterplot of deltas
    dum_81 = ax81.scatter(uresid_med[:,0], uresid_med[:,1], \
                          c=shade, cmap=cmap, s=16, \
                          edgecolor='0.5')
    
    cbar81 = fig8.colorbar(dum_81, ax=ax81)
    cbar82 = fig8.colorbar(dum_82, ax=ax82, label=label_pmem)

    ax81.set_xlabel(r'$\Delta u$')
    ax81.set_ylabel(r'$\Delta v$')
    
    ax82.set_xlabel('u')
    ax82.set_ylabel('v')

    fig8.savefig('test_mixmod_quiver.png')

    
def show_ellipses(dsamples={}, ax=None, fig=None, \
                  key_cen_u='u0', key_cen_v='v0', \
                  key_var_u='var_fg', key_var_v=None, \
                  key_corr_uv=None, \
                  errSF=1., \
                  nshow=25, \
                  which_samples=None, \
                  alphaEllipse=0.02, \
                  cmapEllipse='viridis', \
                  edgecolorEllipse='#00274C', \
                  facecolorEllipse='#FFCB05', \
                  edgealphaEllipse=0.05, \
                  zorder=5, \
                  plotMedian=True, \
                  AAinv=None, \
                  residuals_include_offsets=True):

    """Overplots samples from covariance ellipses on the current
axes. Specify the keys for the model component to overplot on the
current axes. Currently every model component is assumed to be scalar (so two keys for the [u,v] components of u0, etc.

    INPUTS

    ax = current axis instance to use

    fig = current figure instance to use

    dsamples = dictionary of samples to use

    key_cen_u = key for centroid in u

    key_cen_v = key for centroid in v

    key_var_u = key for variance in u
    
    key_var_v = key for variance in v (if None, variance in u is copied)

    key_corr_uv = key for u,v covariance (if None, defaults to zero)

    errSF = scale factor by which to exaggerate the ellipse axis
    lengths for visualization

    nshow = how many to actually plot

    which_samples = integer array of indices to show (optional)

    alphaEllipse = face color alpha

    cmapEllipse = color map if using dynamic scaling (not implemented
    for this yet)

    edgecolorEllipse = edge color for the ellipse

    facecolorEllipse = face color for the ellipse

    edgeAlphaEllipse = alpha for ellipse edge color

    zorder = vertical order for this ellipse collection

    plotMedian = overplot the median (of the entire sample) ellipse

    AAinv = [N,2,2] stack of A.<A>^{-1} correction matrices to apply
    the differential frame transformation when plotting each
    ellipse. Ignored if None.
    
    residuals_include_offsets - True if the residuals include the offsets [u0, v0] in the model.

    """

    # WISHLIST:
    #

    # (i) compute and plot median -- DONE.

    # (ii) include frame transformations -- DONE UPSTREAM.

    # (iii) accept optional list of indices of entries to plot (so
    # that a deterministic random sample can be plotted, but the SAME
    # sample for both foreground and background) DONE.
    
    # Comment: much of this is borrowed from my repo
    # weighteddeltas.coverrplot(), which uses some other stuff I
    # wrote. For the moment, bring the main pieces here to try to
    # minimize dependencies...
    
    # Views of the samples for convenience later on. At a minimum, we
    # need both components of the centroid, and one component of the
    # variance. So:
    for key in [key_cen_u, key_cen_v, key_var_u]:
        if not key in dsamples.keys():
            print("show_ellipses WARN - key not present: %s" \
                  % (key))
            return
            
    # Having established that our minimum keys are present, populate
    # the components we need:
    u0 = dsamples[key_cen_u]
    v0 = dsamples[key_cen_v]
    var_u = dsamples[key_var_u]

    # Now for the other variance components. Populate if not given.
    if key_var_v in dsamples.keys():
        var_v = dsamples[key_var_v]
    else:
        var_v = np.copy(var_u)

    if key_corr_uv in dsamples.keys():
        corr_uv = dsamples[key_corr_uv]
    else:
        corr_uv = var_u * 0.

    # If we get here then the input has been correctly parsed. Whether
    # it actually makes sense to use the input is established here.
    if np.size(u0) < 1:
        print("show_samples WARN - u0 has zero size")
        return

    # Wrap the covariance in to an [N,2,2] stack. That method can take
    # stddev_u, stdev_v, corruv as inputs. We defer taking the sqrt to
    # here (rather than specifying earlier on) so that we can decide
    # later to use other inputs if we wish.
    covars = covarsNx2x2.CovarsNx2x2(stdx=np.sqrt(var_u), \
                                     stdy=np.sqrt(var_v), \
                                     corrxy=corr_uv)

    # the centroids. Note that because the offset is now included in
    # the model, we subtract off the median offset first. This is
    # probably most easily done in the following way:
    uv = np.column_stack(( u0, v0 ))
    if residuals_include_offsets:
        uv -= np.median(uv, axis=0)[None,:]

    # Apply corrections here if supplied. Written out for transparency
    if AAinv is not None:
        U = AAinv
        UT = np.transpose(AAinv,axes=(0,2,1))

        # Create copies of the un-updated objects for checking
        covs_old = np.copy(covars.covars)
        uv_old = np.copy(uv)
        
        # Update the covariance array, plane-by-plane, and
        # re-initialize the covariances object using THIS as input.
        V_cor = UT @ covars.covars @ U
        covars = None
        covars = covarsNx2x2.CovarsNx2x2(V_cor)

        # update the offset, plane-by-plane
        uv = np.einsum('ijk,ik->ij',U,uv_old)

        print("show_ellipses INFO - updated covariances and offsets with transformation corrections")
        
    
    # the full-widths wanted by the ellipse collection
    covars.eigensFromCovars()
    ww = covars.majors**0.5 * errSF * 2.0
    hh = covars.minors**0.5 * errSF * 2.0
    posans = covars.rotDegs

    
    #print("show_ellipses INFO:")
    #print(uv.shape, ww.shape, hh.shape, posans.shape, var_u.shape, covars.nPts)
    
    # Supply a figure if none was given, and override the input choice
    # of axis (so that the axis and figure do not point to separate
    # objects). Could probably do the following a little more
    # intelligently, but this will do for now.
    if fig is None:
        fig = plt.figure(14, figsize=(5,5))
        ax = fig.add_subplot(111)

    # ensure we have an axis to work with if none was supplied
    if ax is None:
        ax = fig.add_subplot(111)

    # Some color fun
    edgergbaEllipse = mpl_colors.to_rgba(c=edgecolorEllipse, \
                                         alpha=edgealphaEllipse)
    facergbaEllipse = mpl_colors.to_rgba(c=facecolorEllipse, \
                                         alpha=alphaEllipse)

    # Which ones to show?
    if which_samples is None:
        lsho = np.arange(nshow, dtype='int')
    else:
        lsho = np.copy(which_samples)

    # ensure size OK
    if lsho.size > np.size(ww):
        lsho = lsho[0:np.size(ww)]
        
    # Now construct and plot the ellipse collection
    nshow = min([nshow, np.size(ww)])
    ec = EllipseCollection(ww[lsho], hh[lsho], posans[lsho], \
                           units='xy', offsets=uv[lsho], \
                           transOffset=ax.transData, \
                           #alpha=alphaEllipse, \
                           edgecolor=edgergbaEllipse, \
                           facecolor=facergbaEllipse, \
                           cmap=cmapEllipse, \
                           zorder=zorder)
    
    # add the collection to the current axes
    ax.add_collection(ec)

    # Are we going to plot the median?
    if not plotMedian:
        return

    med_uv = np.median(uv,axis=0)
    
    # do the median covariance BEFORE conversion to the ellipse
    # full-width
    med_ww = np.median(covars.majors, axis=0)**0.5 * errSF * 2.0
    med_hh = np.median(covars.minors, axis=0)**0.5 * errSF * 2.0
    med_posan = np.median(posans)

    med_rgba = mpl_colors.to_rgba(c=edgecolorEllipse, alpha=0.8)
    med_rgba_face = mpl_colors.to_rgba(c=facecolorEllipse, \
                                       alpha=0.0)
    
    med_ec = EllipseCollection(med_ww, med_hh, med_posan, \
                               units='xy', offsets=med_uv, \
                               transOffset = ax.transData, \
                               edgecolor = med_rgba, \
                               facecolor = med_rgba_face, \
                               zorder = zorder+1)

    ax.add_collection(med_ec)
    
######## test routines follow

def test2par(ndata=25, true_params=[1.0e-2, 30.]):

    """Test the 2D version of our fitter"""

    # Generate data, accepting the defaults for the moment
    x, u, ucov, xcov, _ = gendata(ndata, \
                                  s_true=true_params[0],
                                  thetadeg_true=true_params[1], \
                                  showdata=True)

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

def test2term_bells(ndata=25, s=1.0e-2, theta=30., \
                    sigu=1e-4, sigv=1e-4, \
                    sigx=0.01, sigy=0.01, perturb_xy=False, \
                    fit_xy=False, fit_var=False, num_chains=2):

    """Test 2-parameter model with various bells and whistles"""

    x, u, ucov, xcov, _ = gendata(ndata, \
                                  s_true=s, thetadeg_true=theta, \
                                  sigu=sigu, sigv=sigv, \
                                  perturb_xy=perturb_xy, \
                                  sigx=sigx, sigy=sigy, \
                                  showdata=True)

    # set up the sampler...
    sampler = infer.MCMC(
        infer.NUTS(model_2term_bells),
        num_warmup=2000,
        num_samples=2000,
        num_chains=num_chains,
        progress_bar=True)

    # run the sampler
    t0=time.time()

    # transform x uncertainty as well?
    xsend = None
    if fit_xy:
        xsend = jnp.array(xcov)

    sampler.run(jax.random.key(123), x, ucov, u=u, xerr=xsend, \
                fitvar=fit_var)

    
    t1=time.time()
                    
    print("Time elapsed sampling 2term: %.2e seconds" \
          % (time.time()-t0 ) )
    inf_data = az.from_numpyro(sampler)
    print(az.summary(inf_data))

    samples = sampler.get_samples()
    chainz = np.vstack(( samples["s"], samples["theta"] )).T

    # some particulars for the corner plot
    corner_labels = ["s", r"$\theta$"]
    corner_truths = [s, np.radians(theta)]
    
    if fit_var:
        chainz = np.vstack(( samples["s"], samples["theta"], \
                             samples["v_add"] )).T

        corner_labels.append([r"$v_{add}$"])
        corner_truths.append(None)

    fig2 = plt.figure(2, figsize=(6,6))
    fig2.clf()
    dum = corner.corner(chainz, labels=corner_labels, \
                        truths=corner_truths, \
                        fig=fig2)

        
def test6term(ndata=25, \
              s_true=1.0e-2, rotdeg_true=30., \
              betadeg_true=2.0, r_true=1.0, \
              u0_true = 2.0e-3, \
              v0_true = 1.0e-3, \
              perturb_xy=False, \
              sigx=0.02, sigy=0.02, \
              fit_xy=False, \
              fit_var=False, \
              num_chains=2):

    """Tests the 6-term transformation sampler"""

    # generate the data
    x, u, ucov, xcov, _ = gendata(ndata, \
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
        num_chains=num_chains,
        progress_bar=True)

    # run the sampler
    t0=time.time()

    # transform x uncertainty as well?
    xsend = None
    if fit_xy:
        xsend = jnp.array(xcov)
    
    sampler.run(jax.random.PRNGKey(0), x, ucov, u=u, xerr=xsend, \
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
                             #samples["v_add"] )).T
                             np.log10(samples["v_add"]) )).T
        
        corner_labels = ["s", r"$\theta$", r"$s_y/s_x$", r"$\beta$", \
                         r"u_0", r"v_0", \
                         #r"$v_{add}$"]
                         r"$log_{10}(v_{add})$"]

        corner_truths.append(None)
    
    fig2 = plt.figure(2, figsize=(6,6))
    fig2.clf()
    dum = corner.corner(chainz, labels=corner_labels, \
                        truths=corner_truths, \
                        fig=fig2)

    fig2.subplots_adjust(bottom=0.15, left=0.15)

    for ax in fig2.get_axes():
        ax.tick_params(axis="both",labelsize=7)
    
    #blah = plot_trace_dist(inf_data, var_names=["s","theta", \
    #                                            "r", "beta", \
    #                                            "u0", "v0" ])


def test2term_moves(ndata=25, s=1.0e-2, theta=30., \
                    u0=0., v0=0., \
                    sigu=1e-4, sigv=1e-4, \
                    du_lo=1e-4, du_hi=1e-3, \
                    num_chains=2, \
                    num_samples=2000, \
                    fit_var=True, \
                    test_moves=False, \
                    seed=123, \
                    shift_u=0., shift_v=0., \
                    xsz=400., ysz=400., \
                    frac_shift=1., \
                    test_shift=False, \
                    frac_outly=0., \
                    sigm_outly=0.01, \
                    shift_outly=True, \
                    test_mix=False, \
                    test_popmix=False, \
                    show_gen=True, \
                    only_show=False, \
                    add_covar=False, \
                    add_contam=False, \
                    frac_contam=0.2, \
                    rot_in=1., rot_ou=0.05, rot_pow=-3., \
                    rtrue=1., betadeg=0., \
                    add_clumps=False, \
                    tell_perts=True):

    """Sets up 2-term mapping where the objects can move after the
transformation. Main aim: see if we can track star-by-star movements
as part of the transformation fitting. Lots of optional tweaks to the input to test the parameter exploration under various confounding situations.

    INPUTS
    ======

    ndata = number of datapoints to generate

    s = model scale

    theta = model position angle, degrees

    u0 = model center, u

    v0 = model center, v

    sigu = standard deviation ("measurement uncertainty") in u

    sigv = standard deviation in v

    du_lo = additional variance in u, lower bound

    du_hi = additional variance in v, lower bound

    num_chains = number of MCMC chains

    num_samples = number of samples per chain

    fit_var = fit additional variance as a model parameter

    seed = random number seed

    shift_u0 = shift to add to SOME of the points, u

    shift_v0 = shift to add to SOME of the points, v

    xsz = dataset x side length

    ysz = dataset y side length

    frac_shift = fraction of points to shift

    test_moves = test 2-term model plus star-by-star moves but without
    centroid shift

    test_shift = test a model in which both the model and star-by-star
    include a shift.

    test_mix = test the mixture model plus individual star motions

    test_popmix = test mixture model applied to populations, but NOT
    allowing individualstars to move.

    show_gen = show generated data and save figure (usually want to do
    this)

    only_show = return after showing the figure (i.e. don't actually
    do the samples)

    add_covar = add diagonal variances described in the target frame
    by uniform(du_lo**2, du_hi**2)

    add_contam = add systematic (e.g. vortex) contaminants

    frac_contam = fraction of objects to label as contaminants

    rot_in = position angle for contaminants, in degrees, at the
    contaminant rotation center

    rot_ou = as rot_in but for the maximum radius (which defaults to
    the greatest separation from the center of rotation)

    rot_pow = power law exponent for the rotation angle of contaminants

    rtrue = scale ratio sy/sx in the transformation (note that the
    2-term scale-rotation model assumes rtrue=1).

    add_clumps = test multiple clumps in at least du, dv space

    betadeg = axis deviation from perpendicular, in degrees (note that
    the 2-term scale rotation model assumes betadeg=0.)

    tell_perts = report to screen which perturbations are being generated

    RETURNS
    =======

    dret = {} = dictionary containing the MCMC samples, with arguments that depend on the options that were sent in

    """

    # 2026-05-20 testing note: the old defaults were:
    #
    # xsz=2., ysz=2., s=1.0e-2
    
    # try shifting u, v to see if the model recovers it

    if tell_perts:
        print("test2term_moves INFO - truth parameters:")
        print("test2term_moves INFO - s=%.2e, theta=%.2e (%.3f rad), shift_u=%.2e, shift_v=%.2e, u0=%.2e, v0=%.2e" \
              % (s, theta, np.radians(theta), shift_u, shift_v, u0, v0))
        print("test2term_moves INFO - sy/sx=%.2f, betadeg=%.2f" \
              % (rtrue, betadeg))
        
    # Transformation plus measurement uncertainty...
    x, utran, ucov, xcov, ugen = gendata(ndata, xsz, ysz, \
                                         s_true=s, thetadeg_true=theta, \
                                         r_true=rtrue, \
                                         betadeg_true=betadeg, \
                                         u0=u0, v0=v0, \
                                         sigu=sigu, sigv=sigv, \
                                         showdata=True)

    # Note: ucov, xcov are what the experimenter "thinks" the
    # measurement uncertainties are in the input and target frame,
    # respectively. "ugen" is the transformed xy positions BEFORE any
    # addition of measurement uncertainty in the uv frame within
    # gendata().

    # Add one or more clumps in du, dv space (later we can add in u, v
    # space too).
    bclumps = np.repeat(False, utran.shape[0])
    perts_clumps = utran * 0.
    if add_clumps:

        # For the moment, let's come up with some parameters
        fclumps = [0.15, 0.1, 0.15]
        du_clumps = [0.009, 0.004, -0.003]
        dv_clumps = [-0.006, 0.004, 0.003]

        sigu_clumps = [2e-4, 1e-4, 5e-4]
        sigv_clumps = [2e-4, 1e-5, 8e-4]
        corrxy_clumps = [0.0, 0.2, -0.2]
        
        perts_clumps, which_clump \
            = clumps_du(utran, fclumps, \
                        np.array(du_clumps), np.array(dv_clumps), \
                        np.array(sigu_clumps), np.array(sigv_clumps), \
                        np.array(corrxy_clumps))

        # which objects are part of clumps?
        bclumps = which_clump >= 0

        if tell_perts:
            print("test2term_moves INFO - added %i clumps:" \
                  % (np.max(which_clump)+1), np.sum(bclumps) )
        
    # Assign a population of "contaminants" undergoing something
    # systematic in the [u,v] frame. Since we want to allow nonlinear
    # transforamtions (in the coordinates) without incurring
    # measurement uncertainty more than once, the offsets are
    # generated using the positions before measurement uncertainty is
    # applied. They can then be added in any order. Currently the
    # actual model for the contamination is our separation-dependent
    # "vortex" model, which we apply as a rotation on top of the [x,y]
    # to [u,v] transformation. So:
    bcontam = np.repeat(False, ugen.shape[0])
    perts_contam = ugen * 0.
    if add_contam and frac_contam > 0:

        # Generate a rotation center slightly off-axis,
        # programammatically for the moment (we aren't fitting this,
        # but are using it to challenge the sampler). Set some nasty
        # contamination parameters.
        rot_cen = ugen.min(axis=0) \
            + 0.35*(ugen.max(axis=0)-ugen.min(axis=0))

        # this was 0.45
        
        #rot_cen=np.zeros(2)

        ## moved up to arguments
        #rot_in = 1.
        #rot_ou = 0.05
        #rot_pow = -3.

        ## try a *really* big spurious rotation. What happens?
        #rot_in = 2.
        #rot_ou = 0.05

        ## 2026-06-08 9:06pm - try a really large contaminating
        ## rotation. Does this pull off the fitted parameters?
        #rot_in=10.
        #rot_ou=0.5
        
        # Generate contamination shift for everything...
        AA, contam_dr, contam_rotdeg \
            = vortex_matrices(ugen, rot_in, rot_ou, *rot_cen, rpow=rot_pow)

        duv_contam = apply_vortex(ugen, AA, rot_cen) - ugen

        # ... and select out a sample to apply
        bcontam = np.random.rand(ugen.shape[0]) <= frac_contam
        perts_contam[bcontam] = duv_contam[bcontam]

        if tell_perts:
            print("test2term_moves INFO - adding contam: %.2f, %i, rotn center [%.2e, %.2e], rotdegs (%.2f, %.2f), power %.1f" \
                  % (frac_contam, np.sum(bcontam), rot_cen[0], rot_cen[1], \
                     rot_in, rot_ou, rot_pow))
        
    # Now move the objects, *after* the transformation and
    # measurement. Depending on the input arguments, these might be by
    # more or less than the measurement uncertainty. For the moment
    # we'll make these diagonal for ease of specification, can relax
    # later. So - first we generate the arrays of low, hi
    # perturbations...
    if add_covar:
        stdds_u = np.random.uniform(du_lo, du_hi, size=x.shape[0])
        stdds_v = np.copy(stdds_u)

        # now generate covariances and samples from these perturbations
        covs, perts_u = getcovs(stdds_u, stdds_v)

        if tell_perts:
            print("test2term_moves INFO - added (uv) covariances: %.2e to %.2e" \
                  % (du_lo, du_hi))
        
    else:
        # Otherwise there are no perturbations to add HERE.
        covs = np.copy(ucov)
        perts_u = utran * 0.

    # Add the "contaminants" (complicated systematic in [u,v] defined
    # above)
    perts_u[bcontam] += perts_contam[bcontam]
        
    # add outliers (gaussian with big deviations) here
    covs_outly, perts_outly = getcovs(np.repeat(sigm_outly, x.shape[0]))
    boutly = np.random.rand(perts_u.shape[0]) <= frac_outly
    perts_u[boutly] += perts_outly[boutly]

    # Report to screen that the outliers were apploed?
    if np.sum(boutly) > 0 and tell_perts:
        print("test2term_moves INFO - added outliers: s=%.2e, f=%.2f" \
              % (sigm_outly, frac_outly))
    
    # add the shifts here. This is a little awkward at the moment: if
    # we are only shifting the outliers, we apply the shift to
    # them. Otherwise we draw a *different* random sample and shift
    # those.
    bshif = np.repeat(False, perts_u.shape[0])
    if shift_outly:
        perts_u[boutly,0] += shift_u
        perts_u[boutly,1] += shift_v

        if tell_perts:
            print("test2term_moves INFO - shifting outliers [%.2e, %.2e]" \
                  % (shift_u, shift_v))
        
    else:
        bshif = np.random.rand(perts_u.shape[0]) <= frac_shift
        perts_u[bshif,0] += shift_u
        perts_u[bshif,1] += shift_v

        if tell_perts and frac_shift > 0:
            print("test2term_moves INFO - shifted subset %.2f by [%.2e, %.2e]" \
                  % (frac_shift, shift_u, shift_v))

    # Add the clumps
    perts_u += perts_clumps
            
    # report the fraction assigned outlier or shift or contam
    bbg = boutly + bshif + bcontam + bclumps
    if tell_perts:
        print("test2term_moves INFO - assigned perturbed: %i of %i = %.2e" \
              % (np.sum(bbg), np.size(bbg), 1.0*np.sum(bbg)/np.size(bbg)))
            
    # ok now THIS is our observed sample
    u_obs = utran + perts_u

    # We want to keep track of the total perturbation applied in
    # (u,v), which means we need to know what was added within
    # gendata() after transformation from nominal (x,y). This is what
    # we plot in the VPD below.
    perts_total = perts_u + utran - ugen
    
    # since the generation has become more complicated now, do a plot
    # here.
    if show_gen:
        fig5=plt.figure(5, figsize=(6,6))
        fig5.clf()

        ax5_1 = fig5.add_subplot(221)
        ax5_2 = fig5.add_subplot(222)
        ax5_4 = fig5.add_subplot(224)

        # vs-coord plots
        fig9=plt.figure(9, figsize=(6,4))
        fig9.clf()
        ax9_1 = fig9.add_subplot(321)
        ax9_2 = fig9.add_subplot(323, sharex=ax9_1)
        ax9_3 = fig9.add_subplot(322, sharey=ax9_1)
        ax9_4 = fig9.add_subplot(324, sharey=ax9_2, sharex=ax9_3)
        
        # colors
        cinp = 'k'
        ctar = 'b'
        coutly = '#00B2A9'
        cshift = '#FFCB05'
        ccontam = '#D86018'
        cclumps = '#D86018'
        
        # scatterplots of the datapoints
        ball = np.isfinite(x[:,0])

        for bset, col, zord, sz, marker, label in \
            zip(\
                [ball, bshif, boutly, bcontam, bclumps], \
                [ctar, cshift, coutly, ccontam, cclumps], \
                [10,14,15, 13, 14], \
                [16,9,16, 25, 9], \
                ['o','s','+','x', 's'], \
                ['all', 'shift','outlier', 'contam', 'clumps']):

            # don't plot if this is an empty set (relax this if we do
            # want to see in the plot that there are empty sets)
            if np.sum(bset) < 1:
                continue
            
            dum51 = ax5_1.scatter(x[bset,0], x[bset,1], \
                                  c=col, zorder=zord, s=sz, marker=marker)
            dum52 = ax5_2.scatter(u_obs[bset,0], u_obs[bset,1], \
                                  c=col, zorder=zord, s=sz, marker=marker)
            dum54 = ax5_4.scatter(perts_total[bset,0], perts_total[bset,1], \
                                  c=col, zorder=zord, s=sz, marker=marker, \
                                  label=label)
        

            # Now for the marginals. Make what we plot for the
            # coordinates a choice, we can promote this to an argument
            # (or refactor this all out into a method) later.
            marginals_u=False
            
            coosho = x
            labx='X'
            laby='Y'
            if marginals_u:
                coosho = u_obs
                labx='u'
                laby='v'
                
            dum9xx = ax9_1.scatter(coosho[bset,0], perts_total[bset,0], \
                                   c=col, zorder=zord, s=sz*0.5, \
                                   marker=marker)
            dum9xy = ax9_2.scatter(coosho[bset,0], perts_total[bset,1], \
                                   c=col, zorder=zord, s=sz*0.5, \
                                   marker=marker)
            dum9yx = ax9_3.scatter(coosho[bset,1], perts_total[bset,0], \
                                   c=col, zorder=zord, s=sz*0.5, \
                                   marker=marker)
            dum9yy = ax9_4.scatter(coosho[bset,1], perts_total[bset,1], \
                                   c=col, zorder=zord, s=sz*0.5, \
                                   marker=marker, \
                                   label=label)

            
        ax5_1.set_xlabel(r'$X$')
        ax5_1.set_ylabel(r'$Y$')

        ax5_2.set_xlabel(r'$u$')
        ax5_2.set_ylabel(r'$v$')

        ax5_4.set_xlabel(r'$\Delta u$')
        ax5_4.set_ylabel(r'$\Delta v$')

        leg = fig5.legend(loc=3)
        
        fig5.subplots_adjust(wspace=0.25, hspace=0.25)

        # save the figure so we can conveniently view it...
        fig5.savefig('simulated_scatterplot.png')

        # NOw for the marginals
        for ax in [ax9_1, ax9_2]:
            ax.set_xlabel(labx)
        for ax in [ax9_3, ax9_4]:
            ax.set_xlabel(laby)
        for ax in [ax9_1]:#, ax9_3]:
            ax.set_ylabel(r'$\Delta u$')
        for ax in [ax9_2]:#, ax9_4]:
            ax.set_ylabel(r'$\Delta v$')

        for ax in [ax9_3, ax9_4]:
            ax.yaxis.tick_right()
            
        leg9 = fig9.legend(loc=3)
        fig9.subplots_adjust(wspace=0., hspace=0., left=0.2)

        fig9.savefig('simulated_deltas.png')
            
        # Stop here if we're tweaking our simulated datasets before
        # sampling
        if only_show:
            return {}
    
    # For the moment, try our "working" method, just to make sure our
    # syntax is sensible.
    
    methmodel = model_2term_bells
    if test_moves:
        print("TESTING MOVES MODEL")
        methmodel = model_2term_moves

    if test_shift:
        print("TESTING SHIFT MODEL")
        methmodel = model_2term_shift

    if test_mix:
        print("TESTING MIX MODEL")
        methmodel = model_2term_mix

    if test_popmix:
        print("TESTING MIXMOD")
        methmodel = model_2term_mixmod
        
    sampler = infer.MCMC(
        infer.NUTS(methmodel),
        num_warmup=2000,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True)

    t0 = time.time()
    
    sampler.run(jax.random.key(seed), x, ucov, u=u_obs, xerr=None, \
                fitvar=fit_var)

    # for screen printing
    var_names=["s", "theta"]
    if fit_var:
        var_names.append("v_add")
    inf_data = az.from_numpyro(sampler)
    #print(az.summary(inf_data, var_names=var_names))
    print(az.summary(inf_data))

    t1 = time.time()
    print("Time sampling and printing summary: %.2e sec" % (t1 - t0))
    
    # Set up the corner plot as usual
    samples = sampler.get_samples()
    chainz = np.vstack(( samples["s"], np.degrees(samples["theta"]) )).T
    
    # some particulars for the corner plot
    corner_labels = ["s", r"$\theta$"]
    corner_truths = [s, theta]  # keep as degrees

    # Was a shift part of the model?
    if "u0" in samples.keys():
        chainz = np.vstack(( chainz.T, \
                             samples["u0"], \
                             samples["v0"], )).T
        corner_labels.append(r'$u_0$')
        corner_labels.append(r'$v_0$')
        corner_truths.append(u0) # won't always be right
        corner_truths.append(v0)

    if "Q" in samples.keys():
        chainz = np.vstack(( chainz.T, \
                             samples["var_fg"], \
                             samples["u0_bg"], \
                             samples["v0_bg"], \
                             samples["var_bg"], \
                             samples["Q"] )).T

        corner_labels.append(r'var(fg)')
        corner_labels.append(r'$u_0(bg)$')
        corner_labels.append(r'$v_0(bg)$')
        corner_labels.append('var(bg)')
        corner_labels.append(r'$Q$')
        
        corner_truths.append(du_hi**2) # won't always be right
        corner_truths.append(shift_u)
        corner_truths.append(shift_v)
        corner_truths.append(sigm_outly**2)
        corner_truths.append(1.0-frac_outly) # Q

    
    # additional deltas? (This is a bit awkward, since will be ignored
    # if we fit extra variance. For the moment that's OK, but watchout
    # later.)
    if 'du' in samples.keys():
        chainz = np.vstack(( chainz.T, \
#            samples["s"], \
#                             np.degrees(samples["theta"]), \
                             samples['du'][:,0,0], \
                             samples['du'][:,0,1]
                            )).T
        corner_labels.append('du[:,0,0]')
        corner_labels.append('du[:,0,1]')

        # append the actual perturbation not the shift
        corner_truths.append(perts_u[0,0])
        corner_truths.append(perts_u[0,1])
        #corner_truths.append(shift_u)
        #corner_truths.append(shift_v)
        
    if fit_var:
        chainz = np.vstack(( chainz.T, \
                             np.log10(samples["v_add"]) )).T
        corner_labels.append(r"$log_{10}(v_{add})$")
        corner_truths.append(None)
        
    fig3 = plt.figure(3, figsize=(6,6))
    fig3.clf()
    dum = corner.corner(chainz, labels=corner_labels, \
                        truths=corner_truths, \
                        fig=fig3, \
                        show_titles=True, \
                        #titles=corner_labels[:], \
                        title_fmt=None, \
                        title_kwargs={"fontsize":9}, \
                        label_kwargs={"fontsize":9} )

    fig3.savefig('simulated_cornerplot.png')
    
    # return the samples so that we can play with them. Smuggle the
    # transformed positions in the samples as well
    dret = samples.copy()
    dret['u_gen'] = ugen
    dret['u_tran'] = utran
    dret['u_obs'] = u_obs
    dret['perts_u'] = perts_u # I've lost track of what is what
    dret['x'] = x

    dret['methmodel'] = methmodel.__name__
    
    # for the simulations, we know which ones are (not) outliers,
    # which is useful to know while diagnosing performance and
    # bugfixing. So include this information too.
    dret['b_inly'] = ~bbg
    dret['b_outly'] = boutly
    dret['b_shif'] = bshif
    dret['b_contam'] = bcontam
    dret['b_clumps'] = bclumps
    
    fig3.subplots_adjust(left=0.15, bottom=0.15)

    # dump the samples to disk for the moment
    with open('test_samples.pickle', 'wb') as wobj:
        pickle.dump(dret, wobj)
    
    return dret

def test_vortex(npoints=50, uv_cen = np.array([0.25, 0.25]),\
                rpow=-2, rot_in=20., rot_ou=1.):

    """Tests the vortex model"""

    uv = np.random.uniform(size=(npoints,2))

    AA, r, rotdeg = vortex_matrices(uv, rot_in, rot_ou, \
                                    *uv_cen, rpow=rpow)

    uv_eval = apply_vortex(uv, AA, uv_cen)
    
    # shift
    uv_shif = uv_eval - uv

    # All the rest is debug plot
    
    fig7 = plt.figure(7)
    fig7.clf()
    ax71 = fig7.add_subplot(224)
    ax72 = fig7.add_subplot(221)
    ax73 = fig7.add_subplot(223)

    # Label for the theta plot
    sthet = r'$\phi_i=%.1f^o, \phi_{ou}=%.1f^o, rpow=%.1f, [%.1f, %.1f]$' \
        % (rot_in, rot_ou, rpow, uv_cen[0], uv_cen[1])
    
    dum1 = ax71.scatter(uv[:,0], uv[:,1], \
                        s=1, color='k', zorder=10, alpha=0.4)
    dum2 = ax71.scatter(uv_eval[:,0], uv_eval[:,1], \
                        s=2, color='r', zorder=15, alpha=0.4)

    dum3 = ax72.scatter(r, rotdeg, s=4, alpha=0.5, label=sthet)

    dum4 = ax73.quiver(uv[:,0], uv[:,1], uv_shif[:,0], uv_shif[:,1],\
                       scale_units='xy')

    for ax in [ax71, ax73]:
        ax.set_xlabel('u')
        ax.set_ylabel('v')

    ax72.set_xlabel('r')
    ax72.set_ylabel(r'$\phi$')

    ax72.set_title(sthet, fontsize=7)

    #leg2 = ax72.legend(fontsize=8)
    
    fig7.subplots_adjust(hspace=0.30, wspace=0.30)
    
    fig7.savefig('test_vortexplot.png')

def test_clumps(ndata=100):

    """Tests the clump adder"""

    uvdum = np.zeros((ndata, 2))

    fracs = [0.2,0.15,0.1]

    sigs_u = np.array([0.01,0.005,0.02])
    sigs_v = np.array([0.01,0.009,0.04])
    corrxy = np.array([0.0,0.2, -0.6])

    cens_u = np.array([0.1, -0.1, -0.1])
    cens_v = np.array([0.0,  0.0,  0.1])

    perts, whichclump = clumps_du(uvdum, fracs, cens_u, cens_v, \
                                  sigs_u, sigs_v, corrxy)

    fig7 = plt.figure(7,figsize=(4,4))
    fig7.clf()
    ax7=fig7.add_subplot(111)

    dum = ax7.scatter(perts[:,0], perts[:,1], c=whichclump, s=9, alpha=0.5)
    ax7.set_xlabel(r'$\Delta u$')
    ax7.set_xlabel(r'$\Delta v$')
