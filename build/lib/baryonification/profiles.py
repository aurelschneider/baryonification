import numpy as np
from scipy.special import erf
from scipy.integrate import simpson
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.optimize import fsolve
from scipy.interpolate import splrep,splev

from .constants import *
from .cosmo import CosmoCalculator


def fstar_fct(Mvir, param, eta=0.3):
    """
    Total stellar fraction (central and satellite galaxies).
    Free model parameter eta.
    (Function inspired by Moster+2013, Eq.2)
    """
    NN = param.baryon.Nstar
    M1 = param.baryon.Mstar
    zeta = 1.376
    #return NN/(Mvir/M1)**(eta)
    return NN*((Mvir/M1)**(-zeta)+(Mvir/M1)**(eta))**(-1.0)


class Profiles:
    """
    Class that defines functions for profiles and fractions.
    Used to defined the displacement of particles (in displ.py).
    """
    def __init__(self, rbin, Mvir, cvir, cosmo_corr, cosmo_bias, cosmo_var, param):
        self.param = param
        self.cosmo = CosmoCalculator(param)
        self.rbin  = rbin
        self.Mvir  = Mvir
        self.cvir  = cvir
        self.cosmo_corr = cosmo_corr
        self.cosmo_bias = cosmo_bias
        self.cosmo_var  = cosmo_var
        self.rvir  = (3.0 * Mvir / (4.0 * np.pi * DELTAVIR * self.cosmo.rhoc_of_z())) ** (1.0 / 3.0)
        self.z     = self.param.cosmo.z
        
    def uNFWtr_fct(self, t):
        """
        Truncated NFW density profile. Normalised.
        """
        x = self.cvir * self.rbin / self.rvir
        return 1.0 / (x * (1.0 + x) ** 2.0 * (1.0 + x ** 2.0 / t ** 2.0) ** 2.0)
   
    def rhoNFW_fct(self):
        """
        NFW density profile.
        """
        rho0 = DELTAVIR * self.cosmo.rhoc_of_z() * self.cvir ** 3.0 / (3.0 * np.log(1.0 + self.cvir) - 3.0 * self.cvir / (1.0 + self.cvir))
        x = self.cvir * self.rbin / self.rvir
        return rho0 / (x * (1.0 + x) ** 2.0)
    
    def mNFWtr_fct(self, x, t):
        """
        Truncated NFW mass profile. Normalised.
        """
        pref = t ** 2.0 / (1.0 + t ** 2.0) ** 3.0 / 2.0
        first = x / ((1.0 + x) * (t ** 2.0 + x ** 2.0)) * (x - 2.0 * t ** 6.0 + t ** 4.0 * x * (1.0 - 3.0 * x) + x ** 2.0 + 2.0 * t ** 2.0 * (1.0 + x - x ** 2.0))
        second = t * ((6.0 * t ** 2.0 - 2.0) * np.arctan(x / t) + t * (t ** 2.0 - 3.0) * np.log(t ** 2.0 * (1.0 + x) ** 2.0 / (t ** 2.0 + x ** 2.0)))
        return pref * (first + second)
    
    def mNFW_fct(self, x):
        """
        NFW mass profile. Normalised.
        """
        return (np.log(1.0 + x) - x / (1.0 + x))
    
    def mTOTtr_fct(self, t):
        """
        Normalised total mass (from truncated NFW)
        """
        pref = t ** 2.0 / (1.0 + t ** 2.0) ** 3.0 / 2.0
        first = (3.0 * t ** 2.0 - 1.0) * (np.pi * t - t ** 2.0 - 1.0)
        second = 2.0 * t ** 2.0 * (t ** 2.0 - 3.0) * np.log(t)
        return pref * (first + second)
    
    def MNFWtr_fct(self, r, t):
        """
        Truncated NFW mass profile.
        """
        return self.Mvir * self.mNFWtr_fct(self.cvir * r / self.rvir, t) / self.mNFWtr_fct(self.cvir, t)
    
    def MNFW_fct(self):
        """
        NFW mass profile
        """
        x = self.cvir * self.rbin / self.rvir
        return (np.log(1.0 + x) - x / (1.0 + x)) / (np.log(1.0 + self.cvir) - self.cvir / (1.0 + self.cvir)) * self.Mvir
    
    def beta_fct(self):
        """
        Parametrises slope of gas profile
        Two models (0), (1)
        """
        Mc = self.param.baryon.Mc
        mu = self.param.baryon.mu
        nu = self.param.baryon.nu
        Mc_of_z = Mc * (1 + self.z) ** nu
        
        if (self.param.code.beta_model == 0):
            dslope = 3.0
            beta = dslope - (Mc_of_z / self.Mvir) ** mu
            if (beta < -10.0):
                beta = -10.0

        elif (self.param.code.beta_model == 1):
            dslope = 3.0
            beta = dslope * (self.Mvir / Mc) ** mu / (1 + (self.Mvir / Mc) ** mu)

        else:
            print('ERROR: beta model not defined!')
            exit()
        
        return beta
    
    def uHGA_fct(self, eps):
        """
        Normalised gas density profile
        """
        thco = self.param.baryon.thco
        al = self.param.baryon.alpha
        be = self.beta_fct()
        ga = self.param.baryon.gamma
        de = self.param.baryon.delta

        rsc = self.rvir / self.cvir
        rco = thco * self.rvir
        rej = eps * self.rvir

        w = self.rbin / rco
        x = self.rbin / rsc
        y = self.rbin / rej
        
        #return 1.0 / ((1.0+w**(al/3))*(1.0+x**(2*al/3)))**(be/al)/(1.0+y**ga)**(de/ga)
        #return 1.0 / (w**al*(1.0+w))**(be/(1+al))/(1.0+y**ga)**(de/ga)
        #return 1.0 / (w**al(Mvir)*(1.0+w**(2/3))*(1.0+x**(4/3)))**(be/(2+al(Mvir)))/(1.0+y**ga)**(de/ga) 
        return 1.0 / (1.0 + w ** al) ** (be / al) / (1.0 + y ** ga) ** (de / ga)
    
    def uIGA_fct(self):
        """
        Inner (cold) density profile. Truncated at the virial radius
        """
        rco  = self.rvir
        w    = self.rbin/rco
        uIGA = np.exp(-w) / self.rbin ** 3.0
        rmin = 0.005 #Mpc/h
        uIGA[self.rbin<rmin] = 1.0
        return uIGA
        #return np.exp(-w)/self.rbin**3.0
    
    def uCGA_fct(self):
        """
        Normalised density profile of central galaxy
        """
        rvir = (3.0 * self.Mvir / (4.0 * np.pi * DELTAVIR * self.cosmo.rhoc_of_z())) ** (1.0 / 3.0)
        R12 = self.param.baryon.rcga * rvir
        return np.exp(-self.rbin / R12) / self.rbin ** 2.0
    
    def MCGA_fct(self):
        """
        Normalised mass profile of central galaxy
        (needs to be multiplied with fcga*Mtot)
        """
        R12 = self.param.baryon.rcga * self.rvir
        return 1 - np.exp(-self.rbin / R12)
    
    def P_thermal(self, rho_gas, mass_tot, al_nth, n_nth):
        """
        Thermal pressure (following arXiv:2009.05558, Eq. 13)
        in units of [(Msun/h)/(Mpc/h)^3 * (km/s)^2]
        """
        Pintegral = cumtrapz(G * rho_gas * mass_tot / self.rbin ** 2, self.rbin, initial=0.0) 
        Pth = (1 - al_nth * (self.rbin / self.rvir) ** n_nth) * (Pintegral[-1] - Pintegral)
        Pth[np.where(Pth < 0)] = 0
        return Pth
    
    def calc_profiles(self):
        """
        Calculates fractions, density and mass profiles as a function of radius
        Returns a dictionary
        """

        #Cosmological params
        Om = self.param.cosmo.Om
        Ob = self.param.cosmo.Ob
        h0 = self.param.cosmo.h0

        #DMO profile params
        eps0 = self.param.code.eps0
        eps1 = self.param.code.eps1

        #Stellar params
        eta = self.param.baryon.eta
        deta = self.param.baryon.deta

        #Adiabatic contraction/relaxation params
        ACM_q0     = self.param.code.q0
        ACM_q0_exp = self.param.code.q0_exp
        ACM_q1     = self.param.code.q1
        ACM_q1_exp = self.param.code.q1_exp
        ACM_q2     = self.param.code.q2
        ACM_q2_exp = self.param.code.q2_exp
        
        #nonthermal pressure parms
        n_nth = self.param.baryon.n_nth
        a_nth = self.param.baryon.a_nth
        b_nth = self.param.baryon.b_nth
        
        #eps
        aa = 1 / (1 + self.z)
        Da = self.cosmo.growth_factor(aa)
        D0 = self.cosmo.growth_factor(1.0)
        Da /= D0
        peak_height = 1.686 / self.cosmo_var ** 0.5
        eps  = (eps0 - eps1 * peak_height)

        if (eps < 0):
            print("ERROR: eps<0. Abort")
            exit()
        
        tau = eps * self.cvir
                
        #Total dark-matter-only mass
        Mtot = self.Mvir*self.mTOTtr_fct(tau)/self.mNFWtr_fct(self.cvir,tau)
        
        #total fractions
        fbar  = Ob/Om
        fcdm  = (Om-Ob)/Om
        fstar = fstar_fct(self.Mvir, self.param, eta)
        fcga  = fstar_fct(self.Mvir, self.param, eta+deta) #Moster13
        fsga  = fstar-fcga #satellites and intracluster light

        figa  = self.param.baryon.ciga*fcga #param.baryon.ciga*1e12/Mtot #param.baryon.ciga*fcga
        fhga  = fbar-fcga-fsga-figa #gas fraction
    
        if(fsga<0):
            fsga = 0.0
            print('WARNING: negative fraction of satellite galaxies. Set to 0')
     
        #Initial density and mass profiles
        rho0NFWtr = DELTAVIR*self.cosmo.rhoc_of_z()*self.cvir**3.0/(3.0*self.mNFWtr_fct(self.cvir, tau))
        rhoNFW = rho0NFWtr*self.uNFWtr_fct(tau)

        exclscale = self.param.code.halo_excl
        rho2h  = (1-np.exp(-exclscale*self.rbin/self.rvir)) * (self.cosmo_bias*self.cosmo_corr + 1.0)*Om*RHOC #rho_b=const in comoving coord.
        
        rhoDMO = rhoNFW + rho2h
        MNFW   = self.MNFWtr_fct(self.rbin, tau)
        M2h    = cumtrapz(4.0*np.pi*self.rbin**2.0*rho2h,self.rbin,initial=0.0)
        M2h_tck = splrep(self.rbin, M2h, s=0, k=3)

        #Final HGA density and mass profile
        uHGA     = self.uHGA_fct(eps)
        rho0HGA  = Mtot/(4.0*np.pi*simpson(self.rbin**2.0*uHGA,self.rbin))
        rhoHGA   = rho0HGA*uHGA
        MHGA     = cumtrapz(4.0*np.pi*self.rbin**2.0*rhoHGA,self.rbin,initial=0.0)
        MHGA_tck = splrep(self.rbin, MHGA, s=0, k=3)

        #FINAL IGA density and mass profile
        uIGA     = self.uIGA_fct()
        rho0IGA  = Mtot / (4.0*np.pi*simpson(self.rbin**2.0*uIGA,self.rbin))
        rhoIGA   = rho0IGA * uIGA
        MIGA     = cumtrapz(4.0*np.pi*self.rbin**2.0*rhoIGA,self.rbin,initial=0.0)
        MIGA_tck = splrep(self.rbin, MIGA, s=0, k=3)  

        #FINAL CGA density and mass profile
        uCGA     = self.uCGA_fct()
        rho0CGA  = Mtot/(4.0*np.pi*simpson(self.rbin**2.0*uCGA,self.rbin))
        rhoCGA   = rho0CGA*uCGA
        MCGA     = cumtrapz(4.0*np.pi*self.rbin**2.0*rhoCGA,self.rbin,initial=0.0)
        MCGA_tck = splrep(self.rbin, MCGA, s=0, k=3)

        #Adiabatic Correction Model 0 (Abadi et al 2010)
        if (self.param.code.AC_model==0):
            nn = self.ACM_q0 * (1+self.z)**self.ACM_q0_exp #nn = 1 corresponds to Gnedin 2004
            aa = ACM_q1 * (1+self.z)**ACM_q1_exp
            func = lambda x: (x-1.0) - aa*(((MNFW + M2h)/((fcdm+fsga)*MNFW + fcga*splev(x*self.rbin,MCGA_tck,der=0,ext=3) + fhga*splev(x*self.rbin,MHGA_tck,der=0,ext=3) + figa*splev(x*self.rbin,MIGA_tck,der=0,ext=3) + splev(x*self.rbin,M2h_tck,der=0,ext=3)))**nn - 1.0)

            if (isinstance(self.rbin, float)):
                xi = 1.0
            else:
                xi = np.empty(len(self.rbin)); xi.fill(1.0)
            xx = fsolve(func, xi, fprime=None)

        #Adiabatic Correction Model 1 (Velmani and Paranjape 2023)
        elif (self.param.code.AC_model == 1):

            Q0 = ACM_q0 * (1+self.z)**ACM_q0_exp  # Q0=0 corresponds to Gnedin 2004, Teyssier et al 2011 
            Q1 = ACM_q1 * (1+self.z)**ACM_q1_exp  # Q1 corresponds to q10 in Velmani and Paranjape 2023
            Q2 = ACM_q2 * (1+self.z)**ACM_q2_exp  # Q2 corresponds to q11 in Velmani and Paranjape 2023
            Q1fct = lambda rf: Q1 + Q2*np.log(rf/self.rvir)
            func = lambda x: (x-1.0) - Q1fct(x*self.rbin)*((MNFW + M2h)/((fcdm+fsga)*MNFW + fcga*splev(x*self.rbin,MCGA_tck,der=0) + fhga*splev(x*self.rbin,MHGA_tck,der=0) + figa*splev(x*self.rbin,MIGA_tck,der=0,ext=3) + splev(x*self.rbin,M2h_tck,der=0,ext=3)) - 1.0) - Q0

            if (isinstance(self.rbin, float)):
                xi = 1.0
            else:
                xi = np.empty(len(self.rbin)); xi.fill(1.0)
            xx = fsolve(func, xi, fprime=None)
        
        #Adiabatic Correcion Model 2 (Velmani and Paranjape 2023 with step function instead of Q0)
        elif (self.param.code.AC_model == 2):

            Q0 = ACM_q0*(1+self.z)**ACM_q0_exp
            Q1 = ACM_q1*(1+self.z)**ACM_q1_exp
            Q2 = ACM_q2*(1+self.z)**ACM_q2_exp #not in use

            #Smooth step function
            nn = 1.5
            rc = eps*self.rvir/eps0
            fstep = lambda rr: Q0/(1+(rr/rc)**nn)

            func = lambda x: (x-1.0) - Q1*((MNFW + M2h)/((fcdm+fsga)*MNFW + fcga*splev(x*self.rbin,MCGA_tck,der=0) + fhga*splev(x*self.rbin,MHGA_tck,der=0) + figa*splev(x*self.rbin,MIGA_tck,der=0,ext=3) + splev(x*self.rbin,M2h_tck,der=0,ext=3)) - 1.0) - fstep(x*self.rbin)
        
            if (isinstance(self.rbin, float)):
                xi = 1.0
            else:
                xi = np.empty(len(self.rbin)); xi.fill(1.0)
            xx = fsolve(func, xi, fprime=None)

        #Model 3 (smooth step function follwoed by AC model)
        elif (self.param.code.AC_model == 3):
            Q0 = ACM_q0*(1+self.z)**ACM_q0_exp
            Q1 = ACM_q1*(1+self.z)**ACM_q1_exp
            Q2 = ACM_q2*(1+self.z)**ACM_q2_exp

            #Smooth step function
            nn = 1.5
            rc = eps*self.rvir/eps0
            fstep = lambda rr: 1 - Q0/(1+(rr/rc)**nn)

            r1 = self.rbin*fstep(self.rbin)
            MNFW1 = self.MNFWtr_fct(r1, tau)

            func = lambda x: (x-1.0) - Q2*((MNFW1+splev(r1,M2h_tck,der=0,ext=0))/((fcdm+fsga+Q1*fcga+Q1*figa)*MNFW1 + (1-Q1)*(fcga*splev(x*r1,MCGA_tck,der=0,ext=0) + figa*splev(x*r1,MIGA_tck,der=0,ext=0)) + fhga*splev(x*r1,MHGA_tck,der=0,ext=0) + splev(x*r1,M2h_tck,der=0,ext=0)) - 1.0)

            if (isinstance(self.rbin, float)):
                xi = 1.0
            else:
                xi = np.empty(len(self.rbin)); xi.fill(1.0)
            x2 = fsolve(func, xi, fprime=None)
            xx = x2*fstep(self.rbin)

        #Model 4 (attempt with two stage AC model)
        elif (self.param.code.AC_model == 4):

            Q0 = ACM_q0*(1+self.z)**ACM_q0_exp
            Q1 = ACM_q1*(1+self.z)**ACM_q1_exp
            Q2 = ACM_q2*(1+self.z)**ACM_q2_exp

            #Smooth step function
            nn = 1.5
            rc = eps*self.rvir/eps0
            fstep = lambda rr: 1 - Q0/(1+(rr/rc)**nn)

            r1 = self.rbin*fstep(self.rbin)
            MNFW1 = self.MNFWtr_fct(r1, tau)

            func = lambda x: x - (MNFW1+splev(r1,M2h_tck,der=0,ext=0))/((fcdm + fsga + (1-Q1)*(fcga+figa) + (1-Q2)*fhga)*MNFW1 + Q1*fcga*splev(x*r1,MCGA_tck,der=0,ext=0) + Q1*figa*splev(x*r1,MIGA_tck,der=0,ext=0) + Q2*fhga*splev(x*r1,MHGA_tck,der=0,ext=0) + splev(x*r1,M2h_tck,der=0,ext=0))

            if (isinstance(self.rbin, float)):
                xi = 1.0
            else:
                xi = np.empty(len(self.rbin)); xi.fill(1.0)
            x2 = fsolve(func, xi, fprime=None)
            xx = x2*fstep(self.rbin)

        #Model 5 (explicit function for AC)
        elif (self.param.code.AC_model == 5):

            #Q0 = ACM_q0*(1+self.z)**ACM_q0_exp
            #Q1 = ACM_q1*(1+self.z)**ACM_q1_exp
            #Q2 = ACM_q2*(1+self.z)**ACM_q2_exp
            Q0 = ACM_q0 + ACM_q0_exp*self.z
            Q1 = ACM_q1 + ACM_q1_exp*self.z
            Q2 = ACM_q2 + ACM_q2_exp*self.z

        #Smooth step function
            nn = 1.5
            rc = eps * self.rvir / eps0
            fstep    = lambda rr: 1 + Q0/(1+(rr/rc)**nn)
            ri_ov_rf = lambda rr: fstep(rr) + Q1*fcga*(splev(rr,MCGA_tck,der=0,ext=0)/self.MNFWtr_fct(rr, tau) - 1) + Q1*figa*(splev(rr,MIGA_tck,der=0,ext=0)/self.MNFWtr_fct(rr, tau) - 1) + Q2*fhga*(splev(rr,MHGA_tck,der=0,ext=0)/self.MNFWtr_fct(rr, tau) - 1)

            xx = 1.0 / ri_ov_rf(self.rbin)
            xx = np.nan_to_num(xx,nan=1.0)

        #Model 6
        elif (self.param.code.AC_model == 6):

            Q0 = ACM_q0*(1+self.z)**ACM_q0_exp
            Q1 = ACM_q1*(1+self.z)**ACM_q1_exp
            Q2 = ACM_q2*(1+self.z)**ACM_q2_exp

        #Smooth step function                                                                                   
            nn = 1.5
            rc = eps*self.rvir/eps0
            fstep    = lambda rr: 1 + Q0/(1+(rr/rc)**nn)

            ri_ov_rf = lambda rr: fstep(rr) + fcga*((splev(rr,MCGA_tck,der=0,ext=0)/self.MNFWtr_fct(rr, tau))**Q1 - 1) + figa*((splev(rr,MIGA_tck,der=0,ext=0)/self.MNFWtr_fct(rr, tau))**Q1 - 1) + fhga*((splev(rr,MHGA_tck,der=0,ext=0)/self.MNFWtr_fct(rr, tau))**Q2 - 1)
            
            #prevent zeros
            inv_xx = ri_ov_rf(self.rbin)
            inv_xx[inv_xx == 0] = 1e-6

            xx = 1/inv_xx
            xx = np.nan_to_num(xx,nan=1.0)
            
        else:
            print("ERROR: Adiabatic Contraction model not known. Exit")
            exit()

        #no adiabatic expansion
        if (self.param.code.adiab_exp==False):
            xx[xx>1] = 1
        
        MACM     = self.MNFWtr_fct(self.rbin/xx, tau)
        MACM_tck = splrep(self.rbin, MACM, s=0, k=3)
        rhoACM   = splev(self.rbin,MACM_tck,der=1)/(4.0*np.pi*self.rbin**2.0)
        MACM     = MACM

        if (np.any(np.diff(MACM) <= 0)):
            print("Warning MACM not monotonically rising! Force monotonic rise")
            MACM = np.maximum.accumulate(MACM)

        #Total profile
        rhoDMB   = (fcdm+fsga)*rhoACM + fhga*rhoHGA + figa*rhoIGA + fcga*rhoCGA
        MDMB     = (fcdm+fsga)*MACM + fhga*MHGA + figa*MIGA + fcga*MCGA

        #Nonthermal pressure, Shaw2010 model (1006.1945 Eq16, see also 2009.05558, Eq14)
        if (a_nth>0):
            fmax = 4**(-n_nth) / a_nth
            f1   = (1 + self.z)**b_nth
            f2   = (fmax-1) * np.tanh(b_nth*self.z) + 1.0
            f_z  = min(f1,f2)
        else:
            f_z = 0
        a_nthz = a_nth * f_z

        #Thermal pressure [(km/s)^2*(Msun/h)/(Mpc/h)^3]
        P_th = self.P_thermal(fhga*rhoHGA,MDMB,a_nthz,n_nth)

        #Electron pressure [(km/s)^2*(Msun/h)/(Mpc/h)^3]
        P_e = ((2+2*Xe)/(3+5*Xe))*P_th

        #Gas and electron number density [1/(Mpc/h)^3]
        mean_mol_weight = 4/(3+5*Xe)
        n_gas = fhga*rhoHGA/(m_atom*h0*mean_mol_weight)
        n_e   = (1 + Xe)/2 * fhga*rhoHGA/(m_atom*h0)

        #gas and electron temperature
        T_gas = P_th/(n_gas * kB)  #[(km/s)^2*(Msun/h) * K/erg]
        T_e   = P_e/(n_e * kB)     #[(km/s)^2*(Msun/h) * K/erg]

        #Joule = kg (m/s)^2
        #change of units
        T_gas = 1e6 * T_gas * Msun_per_kg   #[(m/s)^2*(kg/h) * K/erg]
        T_e   = 1e6 * T_e * Msun_per_kg     #[(m/s)^2*(kg/h) * K/erg]

        T_gas = T_gas/h0  #[J * K/erg]
        T_e   = T_e/h0    #[J * K/erg]    
        T_gas = T_gas / erg_per_Joule  # [K]
        T_e   = T_e / erg_per_Joule    # [K]

        #fraction 
        frac = { 'CDM':fcdm, 'CGA':fcga, 'SGA':fsga, 'HGA':fhga, 'IGA':figa }

        #Normalised density without 2-halo term (true density = fX*rhoX)
        dens = { 'NFW':rhoNFW, 'BG':rho2h, 'DMO':rhoDMO, 'ACM':rhoACM, 'CDM':rhoACM, 'SGA':rhoACM, 'CGA':rhoCGA, 'HGA':rhoHGA, 'IGA':rhoIGA, 'DMB':rhoDMB }


        #True mass with 2-halo term
        mass = { 'NFW':MNFW, 'BG':M2h, 'CDM':fcdm*MACM, 'SGA':fsga*MACM, 'CGA':fcga*MCGA, 'HGA':fhga*MHGA, 'IGA':figa*MIGA, 'DMB':MDMB }

        #Pressure and temperature
        pres = { 'thermal': P_th, 'electron': P_e }
        temp = { 'gas': T_gas, 'electron': T_e }
    
        return frac, dens, mass, pres, temp
