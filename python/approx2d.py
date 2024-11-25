#
# approx2d.py  - approximate transformations by polynomials
#

import numpy as np
import matplotlib.pylab as plt
plt.ion()

# methods for simulating and fitting data
import sim2d, fit2d
import fitpoly2d


class Simset(object):

    """Simulated data with polynomial fits [to be tidied up]"""

    def __init__(self, ndata=2500, \
                 parsim='test_sim_equat_nonoise.ini'):

        # How many points are we generating?
        self.ndata = ndata

        # simulation object
        self.parfile_sim = parsim[:]
        self.setupsim()

        # fit object
        self.lsq = None
        
    def setupsim(self):

        """Sets up the simulated data"""

        self.sim = sim2d.Simdata()
        self.sim.loadconfig(self.parfile_sim)

        # Override ndata with input if positive
        if self.ndata > 0:
            self.sim.npts = self.ndata
        else:
            self.ndata = np.copy(self.sim.npts)

    def makedata(self):

        """Generates a new dataset under the simulation parameters"""

        self.sim.generatedata()

    def setupfit(self):

        """Sets up least-squares fitting object"""

        deg=1
        kind='Chebyshev'
        ndata = self.sim.xy.shape[0]
        wts = np.ones(ndata)
        self.lsq = fitpoly2d.Leastsq2d(self.sim.xy[:,0], \
                                       self.sim.xy[:,1], \
                                       deg=deg, \
                                       w=wts, \
                                       kind=kind, \
                                       xytarg=self.sim.xytarg)

        # populate residuals
        self.lsq.ev()
        
##### Methods that use this follow

def testsim():

    """Tests the functionality"""

    SS = Simset()
    SS.makedata()

    # Now do the fit for degree 1
    SS.setupfit()

    print(SS.lsq.pars)
    
    fig1 = plt.figure(1)
    fig1.clf()
    ax11 = fig1.add_subplot(121)
    ax12 = fig1.add_subplot(122)
    
    dum11 = ax11.scatter(SS.sim.xy[:,0], \
                         SS.sim.xy[:,1], s=1, c='g')
    dum12 = ax12.scatter(SS.sim.xytran[:,0], \
                         SS.sim.xytran[:,1], s=1, c='b')
    
    ax11.set_xlabel(r'$\xi$')
    ax11.set_ylabel(r'$\eta$')
    ax12.set_xlabel(r'$\alpha$')
    ax12.set_ylabel(r'$\delta$')
