## no-obsnoise-sim

Simulates and does MCMC on hypothetical case where 40 points are simulated, transformed under the full 
noise model, but for which the observation uncertainties (in the source frame) are assumed unknown. So:

* Data are simulated in both frames;
* The full noise model is used to generate the data;
* but the observer is assumed not to know the observational uncertainty;
* The additional uncertainty and outlier mixture model is included in the fit and MCMC exploration.
* The simulation parameter file includes some random number seeds. This will make much (but not yet all!) 
of the simulation reproducible (I think the particulars of the outliers are not yet controlled by random 
number seed - we could add this feature!)

This example doesn't yet include the use of a paramter file for mcmc2d.setupmcmc() early in the notebook; 
for the moment, the relevant entries are specified as arguments to that method. (The "fullnoise-canned" 
subdirectory shows an example where an mcmc2d parameter file is used.)

