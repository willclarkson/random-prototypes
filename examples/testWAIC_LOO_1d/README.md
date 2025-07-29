## testWAIC_LOO ##

Does Monte Carlo runs for one-dimensional mixture model using MCMC, converts the output to **arviz** format, and uses **arviz**' methods 
to compute the WAIC and LOO statistics.

Uses Dan Foreman-Mackey's 2014 post on mixture models with emcee ([10.5281/zenodo.15856](http://dx.doi.org/10.5281/zenodo.15856)) as a 
starting point, altered slightly to do the run against polynomial degree and output the result in arviz format for comparison.

Main notebooks:

* 2025-07-29_emcee_1d_mixmod_example-withloops.ipynb
* 2025-07-29_compare_emcee.ipynb


