import numpy as np
from scipy.integrate import quad
from scipy.interpolate import splrep,splev

from .constants import *




class CosmoCalculator:
    """
    Class that defines functions for cosmology. Mainly used for the two-halo term.
    """
    def __init__(self, param):
        self.param = param
        self.Om = param.cosmo.Om
        self.Ob = param.cosmo.Ob
        self.h0 = param.cosmo.h0
        self.ns = param.cosmo.ns
        self.s8 = param.cosmo.s8
        self.z  = param.cosmo.z
        self.dc = param.cosmo.dc

        self.kmin = param.code.kmin
        self.kmax = param.code.kmax
        self.rmin = param.code.rmin
        self.rmax = param.code.rmax
        
    def wf(self, y):
        #Tophat window function
        if y > 100.0:
            return 0.0
        return 3.0 * (np.sin(y) - y * np.cos(y)) / y**3

    def siny_ov_y(self, y):
        #sin(y)/y with truncation
        if y > 100.0:
            return 0.0
        return np.sin(y) / y
    
    def rhoc_of_z(self):
        #redshift dependence of critical density (in comoving units where rho_b=const)
        return RHOC * (self.Om * (1.0 + self.z)**3.0 + (1.0 - self.Om)) / (1.0 + self.z)**3.0

    def hubble(self, a):
        #Hubble parameter
        Om = self.Om
        Ol = 1.0 - Om
        H0 = 100.0 * self.h0
        H = H0 * (Om / (a**3.0) + (1.0 - Om - Ol) / (a**2.0) + Ol)**0.5
        return H

    def growth_factor(self, a):
        #Growth factor from Longair textbook (Eq. 11.56)
        Om = self.Om
        
        def integrand(aa):
            return 1.0 / (aa * self.hubble(aa))**3.0

        integral_result, _ = quad(integrand, 0.0, a, epsrel=5e-3, limit=100)
        return self.hubble(a) * (5.0 * Om / 2.0) * integral_result

    def bias(self, var, dcz):
        #bias function from Cooray&Sheth Eq.68
        q  = 0.707
        p  = 0.3
        nu = dcz**2.0/var
        e1 = (q*nu - 1.0)/dcz
        E1 = 2.0*p/dcz/(1.0 + (q*nu)**p)
        b1 = 1.0 + e1 + E1
        return b1

    def variance(self, r, TF_tck, Anorm):
        #calculate variance from transfer function
        if self.wf is None:
            raise ValueError("wf function must be provided for variance calculation.")
        
        ns = self.ns
        kmin = self.kmin
        kmax = self.kmax
        
        def integrand(logk):
            k = np.exp(logk)
            return np.exp((3.0 + ns) * logk) * splev(k, TF_tck)**2 * self.wf(k * r)**2
        
        integral_result, _ = quad(integrand, np.log(kmin), np.log(kmax), epsrel=5e-3, limit=100)
        var = Anorm * integral_result / (2.0 * np.pi**2)
        return var

    def correlation(self, r, TF_tck, Anorm):
        #calculate linear correlation function from transfer function
        if self.siny_ov_y is None:
            raise ValueError("siny_ov_y function must be provided for correlation calculation.")
        
        ns = self.ns
        kmin = self.kmin
        kmax = self.kmax
        
        def integrand(logk):
            k = np.exp(logk)
            return np.exp((3.0 + ns) * logk) * splev(k, TF_tck)**2 * self.siny_ov_y(k * r)
        
        integral_result, _ = quad(integrand, np.log(kmin), np.log(kmax), epsrel=5e-3, limit=100)
        corr = Anorm * integral_result / (2.0 * np.pi**2)
        return corr


    def compute_cosmology(self):
        #compute all relevant quantities, write them to file and pass them 
        a = 1.0 / (1.0 + self.z)
        fb = self.Ob / self.Om
        
        # Growth factor normalized
        Da = self.growth_factor(a)
        D0 = self.growth_factor(1.0)
        Da /= D0

        # Load transfer function from file
        TFfile = self.param.files.transfct
        try:
            names = "k, Ttot"
            TF = np.genfromtxt(TFfile, usecols=(0,6), comments='#', dtype=None, names=names)
        except IOError:
            print('IOERROR: Cannot read transfct. Try: param.files.transfct = "/path/to/file"')
            exit()
        
        TF_tck = splrep(TF['k'], TF['Ttot'])
        
        # Normalize power spectrum
        R = 8.0
        ns = self.ns
        
        def integrand(logk):
            k = np.exp(logk)
            return np.exp((3.0 + ns) * logk) * splev(k, TF_tck)**2 * self.wf(k * R)**2
        
        integral_result, _ = quad(integrand, np.log(self.kmin), np.log(self.kmax), epsrel=5e-3, limit=100)
        A_NORM = 2.0 * np.pi**2 * self.s8**2 / integral_result
        print('Normalizing power-spectrum done!')

        bin_N = 100
        bin_r = np.logspace(np.log(self.rmin), np.log(self.rmax), bin_N, base=np.e)
        bin_m = 4.0 * np.pi * self.Om * RHOC * bin_r**3 / 3.0  # Om * RHOC constant in comoving coords

        bin_var = []
        bin_bias = []
        bin_corr = []
        
        for r_val in bin_r:
            var_val = self.variance(r_val, TF_tck, A_NORM)
            bin_var.append(var_val)
            # bias uses var scaled by Da^2 as in your original code
            bin_bias.append(self.bias(var_val * Da**2, self.dc))
            bin_corr.append(self.correlation(r_val, TF_tck, A_NORM))

        bin_var = np.array(bin_var) * Da**2
        bin_bias = np.array(bin_bias)
        bin_corr = np.array(bin_corr) * Da**2

        COSMOfile = self.param.files.cosmofct
        try:
            np.savetxt(COSMOfile, np.transpose([bin_r, bin_m, bin_var, bin_bias, bin_corr]))
        except IOError:
            print('IOERROR: cannot write Cosmofct file in a non-existing directory!')
            exit()

        return bin_r, bin_m, bin_var, bin_bias, bin_corr

