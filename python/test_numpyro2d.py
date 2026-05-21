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

    """Scale, rotation, offset, mixture

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
    var_bg = numpyro.sample("var_bg", dist.LogUniform(1e-12,1e-3))
    u0_bg = numpyro.sample("u0_bg", dist.Uniform(-1.0, 1.0))
    v0_bg = numpyro.sample("v0_bg", dist.Uniform(-1.0, 1.0))

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

    return xobs, uobs, ucovs, xcovs


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
            ucolor='k', pcolor='g', alpha=0.4):

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
    dum41 = ax41.errorbar(du_med[:,0], du_med[:,1], \
                          yerr=du_std[:,0], xerr=du_std[:,1], \
                          fmt='.', alpha=alpha, ms=6, capsize=2, \
                          color=ucolor, ecolor=ucolor, zorder=10)

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
    
######## test routines follow

def test2par(ndata=25, true_params=[1.0e-2, 30.]):

    """Test the 2D version of our fitter"""

    # Generate data, accepting the defaults for the moment
    x, u, ucov, xcov = gendata(ndata, \
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

    x, u, ucov, xcov = gendata(ndata, \
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
                    test_mix=False):

    """Sets up 2-term mapping where the objects can move after the
transformation. Main aim: see if we can track star-by-star movements
as part of the transformation fitting.

    Some special inputs:

    test_shift = test a model in which both the model and star-by-star
    include a shift.

    """

    # 2026-05-20 testing note: the old defaults were:
    #
    # xsz=2., ysz=2., s=1.0e-2
    
    # try shifting u, v to see if the model recovers it
    
    # Transformation plus measurement uncertainty...
    x, utran, ucov, xcov = gendata(ndata, xsz, ysz, \
                                   s_true=s, thetadeg_true=theta, \
                                   sigu=sigu, sigv=sigv, \
                                   showdata=True)

    # Now move the objects, *after* the transformation and
    # measurement. Depending on the input arguments, these might be by
    # more or less than the measurement uncertainty. For the moment
    # we'll make these diagonal for ease of specification, can relax
    # later. So - first we generate the arrays of low, hi
    # perturbations...
    stdds_u = np.random.uniform(du_lo, du_hi, size=x.shape[0])
    stdds_v = np.copy(stdds_u)

    # now generate covariances and samples from these perturbations
    covs, perts_u = getcovs(stdds_u, stdds_v)

    # add outliers here
    covs_outly, perts_outly = getcovs(np.repeat(sigm_outly, x.shape[0]))
    boutly = np.random.rand(perts_u.shape[0]) <= frac_outly
    perts_u[boutly] += perts_outly[boutly]
    
    # add the shifts here
    bshif = np.random.rand(perts_u.shape[0]) <= frac_shift
    perts_u[bshif,0] += shift_u
    perts_u[bshif,1] += shift_v
    
    # ok now THIS is our observed sample
    u_obs = utran + perts_u

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
        corner_truths.append(shift_u) # won't always be right
        corner_truths.append(shift_v)

    if "u0_bg" in samples.keys():
        chainz = np.vstack(( chainz.T, \
                             samples["u0_bg"], \
                             samples["v0_bg"], \
                             samples["Q"] )).T
        corner_labels.append(r'$u_0(bg)$')
        corner_labels.append(r'$v_0(bg)$')
        corner_labels.append(r'$Q$')
        corner_truths.append(None) # won't always be right
        corner_truths.append(None)
        corner_truths.append(None)

    
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

    # return the samples so that we can play with them. Smuggle the
    # transformed positions in the samples as well
    dret = samples.copy()
    dret['u_tran'] = utran
    dret['u_obs'] = u_obs

    fig3.subplots_adjust(left=0.15, bottom=0.15)
    
    return dret
