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

def model_2term_mixmod(x, uerr, u=None, xerr=None, fitvar=False):

    """Fits mixture model to the positions, but does not fit individual star-by-star shifts. 


    INPUTS:

    x = [N,2] = input positions

    uerr = [N,2,2] = input uncertainties as covariances

    u = [N,2] optional output positions

    xerr = [N,2,2] optional xy uncertainties as covariances   

    fitvar = include diagonal covariance in model parameters


"""

    # our two-term model again:
    theta = numpyro.sample("theta", dist.Uniform(-1.0*jnp.pi, 1.0*jnp.pi))
    s = numpyro.sample("s", dist.LogUniform(1e-5,1.))
    u0= numpyro.sample("u0", dist.Uniform(-1.0, 1.0))
    v0= numpyro.sample("v0", dist.Uniform(-1.0, 1.0))
    
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
    
    # background component
    u0_bg = numpyro.sample("u0_bg", dist.Uniform(-1.0, 1.0))
    v0_bg = numpyro.sample("v0_bg", dist.Uniform(-1.0, 1.0))
    var_bg = numpyro.sample("var_bg", dist.LogUniform(1e-12,1e-3) )

    # Model background covariance. Currently this is always greater
    # than the "foreground" covariance:
    cov_mixmod_bg = jnp.array([[var_bg,0], [0,var_bg]])
    cov_total_bg = cov_total + cov_mixmod_bg[None,:,:]
    
    # predicted positions assuming assigned to bg component
    upred_bg = utran + jnp.array([u0_bg,v0_bg])[None,:]
    
    # The mixture components. Let Q be the outlier probability. Not
    # sure yet how to stop this going to 1/Npoints... try imposing a
    # lower and upper limit
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
    u0= numpyro.sample("u0", dist.Uniform(-1.0, 1.0))
    v0= numpyro.sample("v0", dist.Uniform(-1.0, 1.0))

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
    u0= numpyro.sample("u0", dist.Uniform(-1.0, 1.0))
    v0= numpyro.sample("v0", dist.Uniform(-1.0, 1.0))

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

def show_du(samples={}, keypos='u_tran', \
            ucolor='k', pcolor='g', alpha=0.4, \
            show_std=True):

    """Utility: shows the samples in du"""

    if len(samples.keys()) < 1:
        return

    # The star-by-star samples: [nsamples stars, 2]
    du = samples['du']

    # should be a [nsamples, nstars, 2] array.
    du_med = np.median(du, axis=0)
    du_std = np.std(du, axis=0)

    # The bulk-offset samples: [nsamples, 2]
    if 'u0' in samples.keys() and 'v0' in samples.keys():
        shift = np.vstack(( samples['u0'], samples['v0'] )).T
        du_all = du + shift[:,None,:]

        print("Including frame shift:")
        du_med = np.median(du_all, axis=0)
        du_std = np.std(du_all, axis=0)
        
    
    # set up the figure
    fig4 = plt.figure(4, figsize=(9,4))
    fig4.clf()

    ax41 = fig4.add_subplot(121)
    if show_std:
        dum41 = ax41.errorbar(du_med[:,0], du_med[:,1], \
                              yerr=du_std[:,0], xerr=du_std[:,1], \
                              fmt='.', alpha=alpha, ms=6, capsize=2, \
                              color=ucolor, ecolor=ucolor, zorder=10)

    else:
        dum41 = ax41.scatter(du_med[:,0], du_med[:,1], \
                             marker='.', alpha=alpha, s=6,\
                             c=ucolor, zorder=10)

        
    # If we have the commanded perturbations, show them too
    pert = None
    if 'u_obs' in samples.keys() and 'u_tran' in samples.keys():
        pert = samples['u_obs'] - samples['u_tran']
        dum41_2 = ax41.scatter(pert[:,0], pert[:,1], \
                               alpha=alpha, c=pcolor, \
                               zorder=20, s=16)
    
    ax41.set_xlabel(r"$\Delta u$")
    ax41.set_ylabel(r"$\Delta v$")
    
    # Do we have the base points for quiver plot?
    if not keypos in samples.keys():
        return

    uo = samples[keypos]
    
    ax42 = fig4.add_subplot(122)
    dum_42 = ax42.quiver(uo[:,0], uo[:,1], \
                         du_med[:,0], du_med[:,1], \
                         color=ucolor, zorder=10, alpha=alpha)

    # if we have it, overplot the commanded perturbations
    if pert is not None:
        dum_42_b = ax42.quiver(uo[:,0], uo[:,1], \
                               pert[:,0], pert[:,1], \
                               color=pcolor, zorder=20, \
                               alpha=alpha)

    
    ax42.set_xlabel(r"$u$")
    ax42.set_ylabel(r"$v$")
    
    # cosmetics
    fig4.subplots_adjust(bottom=0.15, left=0.15, hspace=0.30, wspace=0.30)

def show_samples(dsamples={}, ellipses=True):

    """One-liner to show some of the results from an MCMC run

    INPUTS

    dsamples = dictionary of samples

    ellipses = call our prototype ellipse plotter

    """

    # Construct the linear transformation from the samples
    if not 'b' in dsamples.keys():
        print("show_samples INFO - key not in input: b. Returning")
        return

    # Parse the input dictionary. In "production" we would probably
    # setattr, getattr, etc., but for the moment we'll spell them
    # out. This is a prototype after all...
    
    # build the cdmatrix from the samples
    npoints = dsamples['b'].shape[0]
    A = np.zeros((npoints,2,2))
    A[:,0,0] = dsamples['b']
    A[:,0,1] = dsamples['c']
    A[:,1,0] = dsamples['e']
    A[:,1,1] = dsamples['f']

    u0 = np.zeros((npoints,2))
    if "u0" in dsamples.keys():
        u0[:,0] = dsamples['u0']
        u0[:,1] = dsamples['v0']

    u0_bg = np.zeros((npoints,2))
    if "u0_bg" in dsamples.keys():
        u0_bg[:,0] = dsamples["u0_bg"]
        u0_bg[:,1] = dsamples["v0_bg"]

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

    print("Membership probabilities:",p.shape)
    print("Mixture fractions:", Q.shape)
    print("Ndata:", ndata)
    print("CDMATRIX shape:", A.shape)

    # Apply the transformation here
    Amed = np.median(A, axis=0)
    upred_med = np.einsum('jk,ik -> ij', Amed, x)

    # deltas
    uresid_med = u_obs - upred_med

    # Mean probabilities
    pmem = np.median(p[...,0],axis=0)
    print("pmem", pmem.shape)
    
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
                         s=16, c='b')

    fig6.subplots_adjust(hspace=0.3, wspace=0.3)

    show_ellipses(dsamples, ax=ax64, fig=fig6)
    
def show_ellipses(dsamples={}, ax=None, fig=None, \
                  key_cen_u='u0_bg', key_cen_v='v0_bg', \
                  key_var_u='var_bg', key_var_v=None, \
                  key_corr_uv=None, \
                  errSF=1.):

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

    """

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

    # the full-widths wanted by the ellipse collection
    ww = covars.majors**0.5 * errSF * 2.0
    hh = covars.minors**0.5 * errSF * 2.0
    posans = covars.rotDegs
    # TO BE CONTINUED - SEE LINE 2457 of weightedDeltas
    
    print("show_ellipses INFO:")
    print(u0.shape, v0.shape, var_u.shape, var_v.shape, corr_uv.shape)
    
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
                    add_covar=False):

    """Sets up 2-term mapping where the objects can move after the
transformation. Main aim: see if we can track star-by-star movements
as part of the transformation fitting.

    Some special inputs:

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

    """

    # 2026-05-20 testing note: the old defaults were:
    #
    # xsz=2., ysz=2., s=1.0e-2
    
    # try shifting u, v to see if the model recovers it
    
    # Transformation plus measurement uncertainty...
    x, utran, ucov, xcov, ugen = gendata(ndata, xsz, ysz, \
                                         s_true=s, thetadeg_true=theta, \
                                         sigu=sigu, sigv=sigv, \
                                         showdata=True)

    # Note: ucov, xcov are what the experimenter "thinks" the
    # measurement uncertainties are in the input and target frame,
    # respectively. "ugen" is the transformed xy positions BEFORE any
    # addition of measurement uncertainty in the uv frame within
    # gendata().
    
    
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
    else:
        # Otherwise there are no perturbations to add HERE.
        covs = np.copy(ucov)
        perts_u = utran * 0.
        
    # add outliers here
    covs_outly, perts_outly = getcovs(np.repeat(sigm_outly, x.shape[0]))
    boutly = np.random.rand(perts_u.shape[0]) <= frac_outly
    perts_u[boutly] += perts_outly[boutly]
    
    # add the shifts here. This is a little awkward at the moment: if
    # we are only shifting the outliers, we apply the shift to
    # them. Otherwise we draw a *different* random sample and shift
    # those.
    bshif = np.repeat(False, perts_u.shape[0])
    if shift_outly:
        perts_u[boutly,0] += shift_u
        perts_u[boutly,1] += shift_v

    else:
        bshif = np.random.rand(perts_u.shape[0]) <= frac_shift
        perts_u[bshif,0] += shift_u
        perts_u[bshif,1] += shift_v

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

        # colors
        cinp = 'k'
        ctar = 'b'
        coutly = 'r'
        cshift = 'g'
        
        # scatterplots of the datapoints
        ball = np.isfinite(x[:,0])

        for bset, col, zord, sz, label in \
            zip(\
                [ball, bshif, boutly], \
                [ctar, cshift, coutly], \
                [10,11,12], \
                [16,6,2], \
                ['all', 'shift','outlier']):
        
            dum51 = ax5_1.scatter(x[bset,0], x[bset,1], \
                                  c=col, zorder=zord, s=sz)
            dum52 = ax5_2.scatter(u_obs[bset,0], u_obs[bset,1], \
                                  c=col, zorder=zord, s=sz)
            dum54 = ax5_4.scatter(perts_total[bset,0], perts_total[bset,1], \
                                  c=col, zorder=zord, s=sz, \
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

        # Stop here if we're tweaking our simulated datasets before
        # sampling
        if only_show:
            return
    
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
        corner_truths.append(0.) # won't always be right
        corner_truths.append(0.)

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
    dret['x'] = x

    dret['methmodel'] = methmodel.__name__
    
    fig3.subplots_adjust(left=0.15, bottom=0.15)

    # dump the samples to disk for the moment
    with open('test_samples.pickle', 'wb') as wobj:
        pickle.dump(dret, wobj)
    
    return dret
