#
# mixfgbg.py - methods to parse foreground/background mixture
# parameters
#

#
# 2028-08-13 WIC - refactored out of fittwod.py
#

import numpy as np

def parsemixpars(mixpars=np.array([]), \
                 islog10frac=False, \
                 islog10vxx=False, \
                 vxxbg=1.):

    """Parses foreground/background mixture parameters. If no mixpars are
supplied, this defaults to a single-component (foreground-only) model.

Inputs:

    mixpars = [fbg, vbg] = up to 2-element array of mixture model
    parameters: fraction of background, variance of background

    islog10frac = background fraction fbg supplied as log10

    islog10vxx = background variance vxx supplied as log10

    vxxbg = default variance for background component

Returns:

    fbg = fraction of model component that is background. Returns 0.0
    if no mixture parameters were supplied.

    vxx = variance of model. 

    """

    # Defaults
    fbg = 0.
    vxx = vxxbg
    
    #covbg = np.eye(2) * vxxbg

    if np.size(mixpars) < 1:
        return fbg, vxx

    # Mixture fraction...
    fbg = parsefraction(mixpars, islog10frac)

    # variance...
    if np.size(mixpars) > 1:
        vxx = parsefraction(mixpars[1], islog10vxx, maxval=np.inf, inclusive=False)

    return fbg, vxx


def parsefraction(frac=None, islog10=False, badval=-np.inf, \
                  minval=0., maxval=1., \
                  minlog10=-50., \
                  inclusive=True):

    """Given a scalar value frac, return the value (if within bounds) or
badval if not. If frac was supplied as log10(frac), then it is
converted to frac BEFORE comparing to the bounds minval/maxval.

Inputs:

    frac = parameter of interest. If supplied as an array, the 0th
    entry is used.

    islog10 = log10(frac) supplied instead of frac

    badval = what to return if the supplied parameter is outside the range

    minval = minimum value for the parameter
    
    maxval = maximum value for the parameter

    minlog10 = minimum value of log10 to supply. Used for bounds
    checking, is ignored if islog10=False.

    inclusive = value can be equal to minval or maxval.

Returns:

    value = parameter value (if within allowed range) or badval if
    outside. Scalar.

    """

    # Ensure we're dealing with a scalar
    fuse = get0thscalar(frac)
    if fuse is None:
        return badval

    # If log10(frac) supplied, convert to frac itself.
    if islog10:
        if fuse < minlog10:
            fuse = 0.
        else:
            fuse = 10.0**fuse

    # return badval if outside the bounds
    if inclusive:
        isbad = fuse < minval or fuse > maxval
    else:
        isbad = fuse <= minval or fuse >= maxval
    if isbad:
        return badval

    # Return the adopted value if it passed all the tests above.
    return fuse

def get0thscalar(pars=np.array([]) ):

    """Utility - returns scalar zeroth value of array.

Inputs:

    pars = scalar or 1D array. If an array, the first entry is assessed.

Returns:

    firstval = first value in the array, or (if supplied as scalar)
    *a copy of* the scalar value.

    if nothing was supplied, returns None

    """

    if pars is None:
        return None
    
    if np.isscalar(pars):
        return pars

    if np.size(pars) < 1:
        return None
    
    return pars[0]
