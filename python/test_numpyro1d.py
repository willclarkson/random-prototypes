#
# test_numpyro1d.py
#

#
# 2026-04-14: Test dfm's numpyro one-dimensional example from 2022 on
# my system. Do this in a .py module so that I "understand" variable
# scope...
#
#
# This follows very heavily Dan Foreman-Mackey's excellent blog post
# "Introduction to numpyro for astronomers", which can be found here:
# https://dfm.io/posts/intro-to-numpyro/

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

# Some visualization options
import arviz as az
import corner

def gendata(true_frac=0.8, \
            true_params=[1.0,0.0], \
            true_outliers=[0.0,1.0], \
            ranseed=12, ndata=15, ysigma=0.2, \
            xmin=-2., xmax=2.):

    """Generates fake data including outliers. 

    INPUTS

    true_frac = fraction of points that are outliers

    true_params = truth parameters of the model

    true_outliers = [mean, variance] of outlier model. Added in quadrature to measurement uncertainty.

    ranseed = random number seed
    
    ndata = number of datapoints to generate

    ysigma = stddev uncertainty of y points
    
    xmin, xmax = data limits

    RETURNS

    x, y, yerr = observed x, y, uncty, all 1D arrays

    m_bkg = boolean: which are background points

    x0, y0 = fine-grained "truth" array for plotting (also 1d) """

    # set the random seed for reproducibility, generate positions and
    # perturbations
    np.random.seed(ranseed)
    x = np.sort(np.random.uniform(xmin,xmax, ndata))
    yerr = ysigma * np.ones_like(x)

    y = true_params[0] * x + true_params[1] \
        + yerr * np.random.randn(len(x))

    # replace a few with outliers
    m_bkg = np.random.rand(len(x)) > true_frac
    y[m_bkg] = true_outliers[0]
    y[m_bkg] += np.sqrt( true_outliers[1]+yerr[m_bkg]**2 ) \
        * np.random.randn(sum(m_bkg))

    # compute the ``true'' line (for plotting)
    x0 = np.linspace(xmin-0.1, xmax+0.1, 200)
    y0 = np.dot(np.vander(x0,2),true_params)

    return x, y, yerr, m_bkg, x0, y0

def linear_model(x, yerr, y=None):

    """Linear model for numpyro."""

    # Define the priors in mumpyro-native format
    theta = numpyro.sample("theta", dist.Uniform(-0.5*jnp.pi, 0.5*jnp.pi))
    b_perp = numpyro.sample("b_perp", dist.Normal(0,1))

    # Get intermediate parameters from these generated parameters,
    # again in numpyro implementations
    m = numpyro.deterministic("m", jnp.tan(theta))
    b = numpyro.deterministic("b", b_perp / jnp.cos(theta) )

    # Now we set up the sample, using the following incantation:
    with numpyro.plate("data", len(x)):
        numpyro.sample("y", dist.Normal(m * x + b, yerr), obs=y)

def linear_mixture_model(x, yerr, y=None):

    """1D linear model with mixture"""

    # Foreground model and priors
    theta = numpyro.sample("theta", dist.Uniform(-0.5*jnp.pi, 0.5*jnp.pi))
    b_perp = numpyro.sample("b_perp", dist.Normal(0,1))

    # Again, we compute intermediate parameters. Note that we use jax
    # to perform operations other than the simple arithmetic ones:
    m = numpyro.deterministic("m", jnp.tan(theta))
    b = numpyro.deterministic("b", b_perp / jnp.cos(theta) )

    # and now we compute the distribution model of the "foreground"
    # component explicitly:
    fg_dist = dist.Normal(m * x + b, yerr)

    # That's the foreground. The background component is another
    # gaussian, whose location and scale parameter are model
    # parameters with their own priors. So:
    bg_mean = numpyro.sample("bg_mean", dist.Normal(0.0, 1.0))
    bg_sigma = numpyro.sample("bg_sigma", dist.HalfNormal(3.0))

    # ... and these are also wrapped into a distribution:
    bg_dist = dist.Normal(bg_mean, jnp.sqrt(bg_sigma**2 + yerr**2))

    # Now for the mixture. The parameter "Q" indicates the probability
    # that any object is a member of the foreground population. We
    # assign it a uniform prior from 0-1. Our mixture then has a
    # categorical distribution:
    Q = numpyro.sample("Q", dist.Uniform(0.0, 1.0))
    mix = dist.Categorical(probs=jnp.array([Q, 1.0 - Q]))

    # Updated with DFM's tweak to keep track of membership
    # probabilities. We move the mixture definition up here so we can
    # access the properties later.
    mixture = dist.MixtureGeneral(mix, [fg_dist, bg_dist] )
    
    # And now the sample state definition. Notice that the fg_dist and
    # bg_dist are both distributions we defined farther up, but in
    # terms of numpyro and jax entities where necessary.
    with numpyro.plate("data", len(x)):
        #numpyro.sample("obs", dist.MixtureGeneral(mix, \
        #                                          [fg_dist, bg_dist]), \
        #               obs=y)

        # This is the same as the previous version, except now the
        # state (I think?) is attached to a view so that it can be
        # interrogated in the next line:
        y_ = numpyro.sample("obs", mixture, obs=y)

        # and here the membership probabilities are tracked
        log_probs = mixture.component_log_probs(y_)
        numpyro.deterministic(
            "p", log_probs - \
            jax.nn.logsumexp(log_probs, axis=-1, keepdims=True) \
        )

        # The above line attaches column "p" to the samples, I think
        # one per object per sample. It also results in the trace for
        # every one of the membership probabilities being printed -->
        # lots of screen output.
        
        
def showdata(x, y, yerr, m_bkg, x0, y0, \
             post_pred_y=np.array([]), color_pred="C0", \
             prob_fg=np.array([]), cmap='gray_r', \
             samplepars=np.array([]), npars=50, colorsample='g', \
             fignum=1):

    """Plots the data"""

    fig1=plt.figure(fignum)
    plt.clf()
    ax1 = fig1.add_subplot(111)

    dum = ax1.errorbar(x, y, yerr=yerr, fmt=',k', capsize=0, ms=0, zorder=10)
    dum2 = ax1.scatter(x[m_bkg], y[m_bkg], marker='s', s=22, c='w',\
                       edgecolor='k', zorder=15)
    dum3 = ax1.scatter(x[~m_bkg], y[~m_bkg], marker='s', s=22, c='k',\
                       edgecolor='k', zorder=15, label='data')
    dum4 = ax1.plot(x0, y0, alpha=0.5, c='0.1', lw=2, zorder=20, \
                    label='generating model')

    # if we have post-prediction plots, show a subset
    if np.size(post_pred_y) > 0:
        label = "posterior predictive samples"
        for n in np.random.default_rng(0).integers(len(post_pred_y),size=100):
            dum = ax1.plot(x, post_pred_y[n],".", color=color_pred, alpha=0.1,\
                           label=label)
            label = None

    if np.size(prob_fg) > 0:
        dum1 = ax1.scatter(x,y,marker='s',s=22, c=prob_fg, \
                          edgecolor="k", zorder=50, cmap=cmap)
        cbar = fig1.colorbar(dum1, label='inlier probability', ax=ax1)


    # if sample params were passed, show them
    if np.size(samplepars) > 0:
        label2 = 'Posterior model sample'
        for ipar in range(npars):
            mthis = samplepars[ipar][0]
            bthis = samplepars[ipar][1]

            ythis = mthis * x0 + bthis
            dumx = ax1.plot(x0, ythis, zorder=10, alpha=0.05, \
                            color=colorsample, \
                            label=label2)
            label2=None
            
        
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    leg = ax1.legend()
    
###### TEST ROUTINES FOLLOW

def testshow(true_frac=0.8, dosample=True, \
             true_params=[1.0,0.0]):

    """Simple test with defaults"""

    x, y, yerr, bbkg, x0, y0 = gendata(true_frac=true_frac, \
                                       true_params=true_params)
    if not dosample:
        showdata(x,y,yerr, bbkg, x0, y0)
        return

    # Set up the sampler...
    sampler = infer.MCMC(
        infer.NUTS(linear_model),
        num_warmup=2000,
        num_samples=2000,
        num_chains=2,
        progress_bar=True)

    t0 = time.time()
    sampler.run(jax.random.PRNGKey(0), x, yerr, y=y)

    print("Time elapsed sampling: %.2e seconds" % (time.time()-t0 ) )
    inf_data = az.from_numpyro(sampler)
    print(az.summary(inf_data))

    # Now draw the posterior predictive samples. The example in the
    # blog sets up the object, calls it, and passes the output to an
    # argument in one go. Referring to the documentation, let's try
    # this step by step.
    predictive = infer.Predictive(model=linear_model, \
                                  posterior_samples = sampler.get_samples())
    post_pred_samples = predictive(jax.random.PRNGKey(1), x, yerr)
    post_pred_y = post_pred_samples["y"]
    print(post_pred_samples.keys())
    
    print("post-pred:", post_pred_y.shape)
    
    # Re-extract the samples so that we can corner plot them. My version of
    # corner doesn't understand my version of arviz, so we build them
    # here.
    samples = sampler.get_samples()
    chainz = np.vstack(( samples["m"], samples["b"] )).T
    
    # ... and the usual corner plot
    fig2 = plt.figure(2, figsize=(5,5))
    fig2.clf()
    dum = corner.corner(chainz, labels=["m", "b"], truths=true_params, \
                        fig=fig2)

    # Do the plot down here, allowing now for post-prediction samples
    showdata(x,y,yerr, bbkg, x0, y0, post_pred_y, \
             samplepars=chainz, colorsample='b')


def testmixture(true_frac=0.8, true_params=[1.0,0.0]):

    """Tests the mixture model version of the 1d setup"""

    # Same data generation as before...
    x, y, yerr, bbkg, x0, y0 = gendata(true_frac=true_frac, \
                                       true_params=true_params)

    # Set up a sampler using the mixture model
    sampler_mix = infer.MCMC(
        infer.NUTS(linear_mixture_model),
        num_warmup=2000,
        num_samples=2000,
        num_chains=2,
        progress_bar=True,
    )

    t0 = time.time()
    sampler_mix.run(jax.random.PRNGKey(3), x, yerr, y=y)
    t1 = time.time()

    print("Time elapsed in mixture model: %.2e seconds" % (t1-t0 ) )

    inf_data_mix = az.from_numpyro(sampler_mix)
    # print(az.summary(inf_data_mix))

    # lift out the samples again
    samples_mix = sampler_mix.get_samples()
    chainz_mix = np.vstack(( samples_mix["m"], \
                             samples_mix["b"], \
                             samples_mix["Q"] )).T

    fig3 = plt.figure(3, figsize=(5,5))
    fig3.clf()
    dum = corner.corner(chainz_mix, labels=["m", "b", "Q"], \
                        truths = [true_params[0], true_params[1], true_frac], \
                        fig=fig3)

    # Generate post-predictive samples
    predictive_mix = infer.Predictive(model=linear_mixture_model, \
                                      posterior_samples = sampler_mix.get_samples())
    post_pred_samples = predictive_mix(jax.random.PRNGKey(4), x, yerr)
    post_pred_y = post_pred_samples["obs"]
    print(post_pred_samples.keys(), post_pred_y.shape)

    # collapse the per-object membership probability down into mean
    # per object
    p_fg = jnp.mean(jnp.exp(samples_mix["p"][...,0]),axis=0)
    p_sd = jnp.std(jnp.exp(samples_mix["p"][...,0]),axis=0)

    #p_fg = jnp.mean(samples_mix["p"][...,0],axis=0)
    #p_sd = jnp.std(jnp.exp(samples_mix["p"][...,0]),axis=0)

    print("Chains shape:", chainz_mix.shape)
    
    # show the data along with any post-predictive samples
    showdata(x,y,yerr, bbkg, x0, y0, \
             #post_pred_y, color_pred="green", \
             prob_fg = p_fg, cmap='Greens', \
             samplepars=chainz_mix, \
             fignum=4)

