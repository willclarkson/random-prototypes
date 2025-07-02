## fullnoise_canned

Example plane-to-plane MCMC investigation, using prevously-generated data, and with input paramter files 
controlling the MCMC options and the options for the model used in the guess and MCMC exploration. In 
particular:

*three-parameter noise model included for semimajor axis (mag)
* (the "c" parameter is NOT specified as log10(c) )
*astrometric uncertainy has major, minor axes and position angle
*additional uncertainty included over the assumed measurement uncertainties in source and target frames
*mixture model fit to include outliers

In this case, the model fit ("fullnoise" is the same as the model that was used to generate the data.
