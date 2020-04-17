#
# cooDemo.py
#
# 2020-04-16 WIC - demo coord conversions using astropy

from astropy.coordinates import ICRS, Galactic, LSR, Galactocentric
import astropy.units as u
import astropy.coordinates as coord
from astropy.utils.exceptions import AstropyDeprecationWarning
from astropy.table import Table
import numpy as np
import warnings
import os

#warnings.simplefilter('ignore', category=AstropyDeprecationWarning)
coord.galactocentric_frame_defaults.set('v4.0')

def obs2xyz(cooFil='cowleyLine.txt', Verbose=True, outFil='test.fits'):

    """Uses astropy to convert observed ICRS coordinates to Galactic coordinates (with proper motions) and (Galactocentric) Cartesian coordinates. Arguments:

    cooFil = text file giving input coordinates. Column names are read from this file.

    Verbose=True -- set this to print to the terminal a selection of the converted data

    outFil -- filename for the output. (I like .fits format because it preserves metadata and units)

    """
 
    # First read in the data
    try:
        tCoo = Table.read(cooFil, format='ascii')
    except:
        print("cooDemo.obs2xyz WARN - problem reading file %s" % (cooFil))
        return

    # we attach our guess for units in-place to the table (moved to a
    # separate method for clarity here).
    guessUnits(tCoo)

    # ... convert parallax to distance
    tCoo['distance'] = tCoo['parallax'].to(u.parsec, equivalencies=u.parallax())

    # set up a proper motion (RA) scaled by cos(delta)
    tCoo['mu_RAcosDec'] = tCoo['mu_RA']*np.cos(tCoo['DEC'].to(u.radian))

    # print the table, including units
    #if Verbose:
    #    print(tCoo)

    
    # ... and now we are in business. Set up our frame object...
    icrs = ICRS(ra=tCoo['RA'], dec=tCoo['DEC'], \
                    distance=tCoo['distance'], \
                    pm_ra_cosdec=tCoo['mu_RAcosDec'], pm_dec=tCoo['mu_dec'], \
                    radial_velocity=tCoo['RV=rho'])

    # Now, to find the coordinates and velocities in different formats
    # (Cartesian, say), we just look up the relevant attributes for
    # the ICRS object we've created. Here we shunt the cartesian
    # coordinates and velocities into our output table:
    tCoo['X'] = icrs.cartesian.x
    tCoo['Y'] = icrs.cartesian.y
    tCoo['Z'] = icrs.cartesian.z

    # ... then the velocities
    tCoo['vX'] = icrs.velocity.d_x
    tCoo['vY'] = icrs.velocity.d_y
    tCoo['vZ'] = icrs.velocity.d_z

    # We can also transform these to different frames. Here are two
    lsr = icrs.transform_to(LSR)
    galactic = icrs.transform_to(Galactic)

    
    galcen = icrs.transform_to(Galactocentric)
    
    # Now, we populate the output table with our chosen coords and
    # velocities in these frames. Here are the outputs in Galactic
    # coordinates...
    tCoo['l'] = galactic.l
    tCoo['b'] = galactic.b
    tCoo['pm_l_cosb'] = galactic.pm_l_cosb
    tCoo['pm_b'] = galactic.pm_b

    #print(icrs.velocity)
    
    # ... and here are the cartesian coords and velocities in galactocentric
    # coordinates
    tCoo['X'] = galcen.cartesian.x
    tCoo['Y'] = galcen.cartesian.y
    tCoo['Z'] = galcen.cartesian.z
    tCoo['U'] = galcen.velocity.d_x
    tCoo['V'] = galcen.velocity.d_y
    tCoo['W'] = galcen.velocity.d_z
    
    tCoo.write('test.fits',format='fits', overwrite=True)
    
    # if verbose, we write some columns of interest
    if Verbose:

        # make the formats for certain columns a bit more reasonable
        for sCoo in ['l','b','distance', 'X','Y','Z','U','V','W', \
                     'pm_l_cosb', 'pm_b']:
            tCoo[sCoo].format='%7.4f'

        for sCoo in ['l','b']:
            tCoo[sCoo].format='%.6f'
            
        print(tCoo['RV=rho','parallax','RA','DEC','mu_RA','mu_dec', \
                   'l','b','pm_l_cosb','pm_b','distance',\
                   'X','Y','Z','U','V','W'])

        print("####")
        print("INFO - Quantities used:")
        print("Input assumed ICRS, so J2000.0 equinox")
        print("Galactic coords use the IAU 1958 definition (see https://docs.astropy.org/en/stable/api/astropy.coordinates.Galactic.html)")
        print("Galactic center distance:", galcen.galcen_distance, \
              "Solar motion:", galcen.galcen_v_sun, \
              ", z_sun:", galcen.z_sun)
        print("(Column units for input data guessed.)")
        
def guessUnits(tDum=Table):

    """Guess the units of our table"""
             
    # we apply astropy units to our table

    tDum['RV=rho'].unit = u.km / u.second
    tDum['parallax'].unit = u.mas 
    tDum['RA'].unit = u.degree
    tDum['DEC'].unit = u.degree
    tDum['mu_RA'].unit = u.mas / u.year
    tDum['mu_dec'].unit = u.mas / u.year

