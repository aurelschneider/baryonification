"""

FUNCTIONS TO CALCULATE BIAS AND CORRELATION FUNCTION 
(2-HALO TERM)

"""

#from __future__ import print_function
#from __future__ import division

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import splrep,splev
from .constants import *



def wf(y):
    """
    Tophat window function
    """
    w = 3.0*(np.sin(y) - y*np.cos(y))/y**3.0
    if (y>100.0):
        w = 0.0
    return w


def siny_ov_y(y):
    s = np.sin(y)/y
    if (y>100):
        s = 0.0
    return s

def rhoc_of_z(param):
    """
    Redshift dependence of critical density
    (in comoving units where rho_b=const; same as in AHF)
    """
    Om = param.cosmo.Om
    z  = param.cosmo.z
    
    return RHOC*(Om*(1.0+z)**3.0 + (1.0-Om))/(1.0+z)**3.0


def hubble(a,param):
    """
    Hubble parameter
    """
    Om = param.cosmo.Om
    Ol = 1.0-Om
    H0 = 100.0*param.cosmo.h0
    H  = H0 * (Om/(a**3.0) + (1.0 - Om - Ol)/(a**2.0) + Ol)**0.5
    return H


def growth_factor(a, param):
    """
    Growth factor from Longair textbook (Eq. 11.56)
    """
    Om = param.cosmo.Om
    itd = lambda aa: 1.0/(aa*hubble(aa,param))**3.0
    itl = quad(itd, 0.0, a, epsrel=5e-3, limit=100)
    return hubble(a,param)*(5.0*Om/2.0)*itl[0]


def bias(var,dcz):
    """
    bias function from Cooray&Sheth Eq.68
    """
    q  = 0.707
    p  = 0.3
    nu = dcz**2.0/var
    e1 = (q*nu - 1.0)/dcz
    E1 = 2.0*p/dcz/(1.0 + (q*nu)**p)
    b1 = 1.0 + e1 + E1
    return b1


def variance(r,TF_tck,Anorm,param):
    """
    variance of density perturbations at z=0
    """
    ns = param.cosmo.ns
    kmin = param.code.kmin
    kmax = param.code.kmax
    itd = lambda logk: np.exp((3.0+ns)*logk) * splev(np.exp(logk),TF_tck)**2.0 * wf(np.exp(logk)*r)**2.0
    itl = quad(itd, np.log(kmin), np.log(kmax), epsrel=5e-3, limit=100)
    var = Anorm*itl[0]/(2.0*np.pi**2.0)
    return var


def correlation(r,TF_tck,Anorm,param):
    """
    Correlation function at z=0
    """
    ns = param.cosmo.ns
    kmin = param.code.kmin
    kmax = param.code.kmax
    itd = lambda logk: np.exp((3.0+ns)*logk) * splev(np.exp(logk),TF_tck)**2.0 * siny_ov_y(np.exp(logk)*r)
    itl = quad(itd, np.log(kmin), np.log(kmax), epsrel=5e-3, limit=100)
    corr = Anorm*itl[0]/(2.0*np.pi**2.0)
    return corr


def cosmo(param):
    """
    Calculate bias and correlation function
    write results to temporary file
    Input:  transfer fucntion from CAMB
    """
    #parameters
    h0 = param.cosmo.h0
    Om = param.cosmo.Om
    Ob = param.cosmo.Ob
    ns = param.cosmo.ns
    s8 = param.cosmo.s8
    dc = param.cosmo.dc
    a  = 1.0/(1.0+param.cosmo.z)
    fb = Ob/Om
    kmin = param.code.kmin
    kmax = param.code.kmax
    rmin = param.code.rmin
    rmax = param.code.rmax

    #growth function
    Da = growth_factor(a,param)
    D0 = growth_factor(1.0,param)
    Da = Da/D0

    #load transfer function
    TFfile = param.files.transfct
    try:
        names  = "k, Ttot"
        TF     = np.genfromtxt(TFfile,usecols=(0,6),comments='#',dtype=None, names=names)
    except IOError:
        print('IOERROR: Cannot read transfct. Try: par.files.transfct = "/path/to/file"')
        exit()
    TF_tck = splrep(TF['k'], TF['Ttot'])

    #Normalize power spectrum
    R = 8.0
    itd = lambda logk: np.exp((3.0+ns)*logk) * splev(np.exp(logk),TF_tck)**2.0 * wf(np.exp(logk)*R)**2.0
    itl = quad(itd, np.log(kmin), np.log(kmax), epsrel=5e-3, limit=100)
    A_NORM = 2.0 * np.pi**2.0 * s8**2.0 / itl[0]
    print('Normalizing power-spectrum done!')

    bin_N = 100
    bin_r = np.logspace(np.log(rmin),np.log(rmax),bin_N,base=np.e)
    #bin_m = 4.0*np.pi*Om*rhoc_of_z(param)*bin_r**3.0/3.0
    bin_m = 4.0*np.pi*Om*RHOC*bin_r**3.0/3.0  #Om*RHOC is constant in comoving coords 
    
    bin_var  = []
    bin_bias = []
    bin_corr = []
    for i in range(bin_N):
        bin_var  += [variance(bin_r[i],TF_tck,A_NORM,param)]
        #bin_bias += [bias(bin_var[i],dc)]
        #bin_bias += [bias(bin_var[i],dc/Da)]
        bin_bias += [bias(bin_var[i]*Da**2.0,dc)]
        bin_corr += [correlation(bin_r[i],TF_tck,A_NORM,param)]
    bin_var  = np.array(bin_var)*Da**2.0
    bin_bias = np.array(bin_bias)
    bin_corr = np.array(bin_corr)*Da**2.0

    COSMOfile = param.files.cosmofct
    try:
        np.savetxt(COSMOfile,np.transpose([bin_r,bin_m,bin_var,bin_bias,bin_corr]))
    except IOError:
        print('IOERROR: cannot write Cosmofct file in a non-existing directory!')
        exit()
    return bin_r, bin_m, bin_var, bin_bias, bin_corr
