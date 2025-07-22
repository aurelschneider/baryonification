"""

PROFILES AND FRACTIONS FOR BARIONIC CORRECTIONS

"""
#from __future__ import print_function
#from __future__ import division

import numpy as np
from scipy.special import erf
from scipy.integrate import simpson
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.optimize import fsolve
from scipy.interpolate import splrep,splev
from .constants import *
from .cosmo import *



"""
GENERAL FUNCTIONS REALTED TO THE NFW PROFILE
"""

def r500_fct(r200,c):
    """
    From r200 to r500 assuming a NFW profile
    """
    f = lambda y: np.log(1.0+c*y) - c*y/(1.0+c*y) - 5.0/2.0*(np.log(1.0+c)-c/(1.0+c))*y**3.0
    y0 = fsolve(f,1.0)
    return y0*r200


def rvir_fct(r200,c):
    """
    From r500 to r200 assuming a NFW profile
    """
    f = lambda y: np.log(1.0+c*y) - c*y/(1.0+c*y) - 96.0/200.0*(np.log(1.0+c)-c/(1.0+c))*y**3.0
    y0 = fsolve(f,1.0)
    return y0*r200


def M500_fct(M200,c):
    """
    From M200 to M500 assuming a NFW profiles
    """
    f = lambda y: np.log(1.0+c*y) - c*y/(1.0+c*y) - 5.0/2.0*(np.log(1.0+c)-c/(1.0+c))*y**3.0
    y0 = fsolve(f,1.0)
    return 5.0/2.0*M200*y0**3.0


#def cvir_fct(mvir,param):
#    """
#    Concentrations form Dutton+Maccio (2014)
#    c200 (200 times RHOC)
#    Assumes PLANCK cosmology
#    """
#    z = param.cosmo.z
#    
#    A = 0.520 + (0.905-0.520)*np.exp(-0.617*z**1.21)
#    B = -0.101 + 0.026*z
#    return 10.0**A*(mvir/1.0e12)**(B)


"""
STELLAR FRACTIONS
"""

#def fSTAR_fct(Mvir,param,eta=0.3):
#    NN = param.baryon.Nstar
#    M1 = param.baryon.Mstar
#    return NN/(Mvir/M1)**(eta)


def fSTAR_fct(Mvir,param,eta=0.3):
    """
    Total stellar fraction (central and satellite galaxies).
    Free model parameter eta.
    (Function inspired by Moster+2013, Eq.2)
    """
    NN = param.baryon.Nstar
    M1 = param.baryon.Mstar
    zeta = 1.376
    return NN*((Mvir/M1)**(-zeta)+(Mvir/M1)**(eta))**(-1.0)



"""
Generalised (truncated) NFW profiles
"""

def uNFWtr_fct(rbin,cvir,t,Mvir,param):
    """
    Truncated NFW density profile. Normalised.
    """

    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
    x = cvir*rbin/rvir
    return 1.0/(x * (1.0+x)**2.0 * (1.0+x**2.0/t**2.0)**2.0)


def rhoNFW_fct(rbin,cvir,Mvir,param):
    """
    NFW density profile.
    """
    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
    rho0 = DELTAVIR*rhoc_of_z(param)*cvir**3.0/(3.0*np.log(1.0+cvir)-3.0*cvir/(1.0+cvir))
    x = cvir*rbin/rvir
    return rho0/(x * (1.0+x)**2.0)


def mNFWtr_fct(x,t):
    """
    Truncated NFW mass profile. Normalised.
    """
    pref   = t**2.0/(1.0+t**2.0)**3.0/2.0
    first  = x/((1.0+x)*(t**2.0+x**2.0))*(x-2.0*t**6.0+t**4.0*x*(1.0-3.0*x)+x**2.0+2.0*t**2.0*(1.0+x-x**2.0))
    second = t*((6.0*t**2.0-2.0)*np.arctan(x/t)+t*(t**2.0-3.0)*np.log(t**2.0*(1.0+x)**2.0/(t**2.0+x**2.0)))
    return pref*(first+second)


def mNFW_fct(x):
    """
    NFW mass profile. Normalised.
    """
    return (np.log(1.0+x)-x/(1.0+x))


def mTOTtr_fct(t):
    """
    Normalised total mass (from truncated NFW)
    """
    pref   = t**2.0/(1.0+t**2.0)**3.0/2.0
    first  = (3.0*t**2.0-1.0)*(np.pi*t-t**2.0-1.0)
    second = 2.0*t**2.0*(t**2.0-3.0)*np.log(t)
    return pref*(first+second)


def MNFWtr_fct(rbin,cvir,t,Mvir,param):
    """
    Truncateed NFW mass profile.
    """
    
    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
    return Mvir*mNFWtr_fct(cvir*rbin/rvir,t)/mNFWtr_fct(cvir,t)


def MNFW_fct(rbin,cvir,Mvir,param):
    """
    NFW mass profile
    """
    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
    x = cvir*rbin/rvir
    return (np.log(1.0+x) - x/(1.0+x))/(np.log(1.0+cvir)-cvir/(1.0+cvir))*Mvir



"""
GAS PROFILE
"""

def beta_fct(Mvir,param):
    """
    Parametrises slope of gas profile
    Two models (0), (1)
    """
    z  = param.cosmo.z
    Mc = param.baryon.Mc
    mu = param.baryon.mu
    nu = param.baryon.nu
    Mc_of_z = Mc*(1+z)**nu
    
    if (param.code.beta_model==0):

        dslope = 3.0
        beta = dslope - (Mc_of_z/Mvir)**mu
        if (beta<-10.0):
            beta = -10.0

    elif (param.code.beta_model==1):

        dslope = 3.0
        beta = dslope*(Mvir/Mc)**mu/(1+(Mvir/Mc)**mu)

        #if (Mvir<1e12):
        #    beta = 3.0

    else:
        print('ERROR: beta model not defined!')
        exit()
        
    return beta


#def uHGA_fct(rbin,Mvir,param):
#    """
#    Normalised gas density profile
#    """
#    thej = param.baryon.thej
#    thco = param.baryon.thco
#    al   = param.baryon.alpha
#    be   = beta_fct(Mvir,param)
#    ga   = param.baryon.gamma
#    de   = param.baryon.delta 
#    
#    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
#    rco  = thco*rvir
#    rej  = thej*rvir
#    
#    x = rbin/rco
#    y = rbin/rej
#    return 1.0/(1.0+x**al)**(be)/(1.0+y**ga)**((de-al*be)/ga)

'''
def uHGA_fct(rbin,Mvir,eps,param):
    """
    Normalised gas density profile
    """
    thco = param.baryon.thco
    al   = param.baryon.alpha
    be   = beta_fct(Mvir,param)
    ga   = param.baryon.gamma
    de   = param.baryon.delta

    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
    rco  = thco*rvir
    rej  = eps*rvir

    x = rbin/rco
    y = rbin/rej

    return 1.0/(1.0+x**al)**(be/al)/(1.0+y**ga)**(de/ga)
'''


def uHGA_fct(rbin,cvir,Mvir,eps,param):
    """
    Normalised gas density profile
    """
    thco = param.baryon.thco
    #thco = thco_fct(Mvir,param)
    al   = param.baryon.alpha
    #al   = alpha_fct(Mvir,param)
    be   = beta_fct(Mvir,param)
    ga   = param.baryon.gamma
    de   = param.baryon.delta

    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
    rsc  = rvir/cvir
    rco  = thco*rvir
    rej  = eps*rvir

    w = rbin/rco
    x = rbin/rsc
    y = rbin/rej

    #return 1.0/((1.0+w**(al/3))*(1.0+x**(2*al/3)))**(be/al)/(1.0+y**ga)**(de/ga)
    #return 1.0/(w**al*(1.0+w))**(be/(1+al))/(1.0+y**ga)**(de/ga)
    return 1.0/(1.0+w**al)**(be/al)/(1.0+y**ga)**(de/ga)
    #return 1.0/(w**al(Mvir)*(1.0+w**(2/3))*(1.0+x**(4/3)))**(be/(2+al(Mvir)))/(1.0+y**ga)**(de/ga) 

    
def uIGA_fct(rbin,Mvir,param):
    """
    Inner (cold) density profile. Truncated at the virial radius
    """
    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
    thco = param.baryon.thco
    rco  = rvir #thco*rvir
    w    = rbin/rco
    return np.exp(-w)/rbin**3.0


def uBAR_noF_fct(rbin,cvir,Mvir,eps,param):
    """
    Normalised gas+stellar density profile assuming 
    no feeedback (just cooling)
    """
    thco = 0.1 #0.05 #0.0001
    al   = 1.0
    be   = 3.0
    ga   = param.baryon.gamma
    de   = param.baryon.delta

    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
    rsc  = rvir/cvir
    rco  = thco*rvir
    rej  = eps*rvir

    w = rbin/rco
    x = rbin/rsc
    y = rbin/rej

    return 1.0/(1.0+w**al)**(be/al)/(1.0+y**ga)**(de/ga)
    #return 1.0/((1.0+w**(al/3))*(1.0+x**(2*al/3)))**(be/al)/(1.0+y**ga)**(de/ga)


"""
STELLAR PROFILE
"""


#def uCGA_fct(rbin,Mvir,param):
#    """
#    Normalised density profile of central galaxy
#    """
#    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
#    R12 = param.baryon.rcga*rvir
#    return np.exp(-(rbin/R12/2.0)**2.0)/rbin**2.0

def uCGA_fct(rbin,Mvir,param):
    """
    Normalised density profile of central galaxy
    """
    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
    R12 = param.baryon.rcga*rvir
    return np.exp(-rbin/R12)/rbin**2.0


#def MCGA_fct(rbin,Mvir,param):
#    """
#    Normalised mass profile of central galaxy
#    (needs to be multiplied with Mtot)
#    """
#    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
#    R12 = param.baryon.rcga*rvir
#    return erf(rbin/R12/2.0)


def MCGA_fct(rbin,Mvir,param):
    """
    Normalised mass profile of central galaxy
    (needs to be multiplied with fcga*Mtot)
    """
    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
    R12 = param.baryon.rcga*rvir
    return 1 - np.exp(-rbin/R12)



"""
Thermal pressure
"""

def P_thermal(rbin,rho_gas,mass_tot,rv,al_nth,n_nth,param):
    """
    Thermal pressure (following arXiv:2009.05558, Eq. 13)
    in units of [(Msun/h)/(Mpc/h)^3 * (km/s)^2]
    """
    Pintegral = cumtrapz(G*rho_gas*mass_tot/rbin**2,rbin, initial=0.0) 
    Pth = (1 - al_nth*(rbin/rv)**n_nth)*(Pintegral[-1] - Pintegral)
    Pth[np.where(Pth<0)]=0

    return Pth


"""
TOTAL PROFILE
"""

def profiles(rbin,Mvir,cvir,cosmo_corr,cosmo_bias,cosmo_var,param):

    """
    Calculates fractions, density and mass profiles as a function of radius
    Returns a dictionary
    """
    
    #redshift
    zz = param.cosmo.z
    
    #Cosmological params
    Om      = param.cosmo.Om
    Ob      = param.cosmo.Ob
    h0      = param.cosmo.h0

    #DMO profile params
    eps0    = param.code.eps0
    eps1    = param.code.eps1
    #eps     = param.code.eps

    #Stellar params
    eta     = param.baryon.eta
    deta    = param.baryon.deta

    #Adiabatic contraction/relaxation params
    ACM_q0     = param.code.q0
    ACM_q0_exp = param.code.q0_exp
    ACM_q1     = param.code.q1
    ACM_q1_exp = param.code.q1_exp
    ACM_q2     = param.code.q2
    ACM_q2_exp = param.code.q2_exp
    

    #nonthermal pressure parms
    n_nth   = param.baryon.n_nth
    a_nth   = param.baryon.a_nth
    b_nth   = param.baryon.b_nth
    
    #eps
    aa = 1/(1+zz)
    Da = growth_factor(aa,param)
    D0 = growth_factor(1.0,param)
    Da = Da/D0
    peak_height = 1.686/cosmo_var**0.5
    eps  = (eps0 - eps1*peak_height)

    if (eps<0):
        print("ERROR: eps<0. Abort")
        exit()
        
    tau = eps*cvir
    rvir = (3.0*Mvir/(4.0*np.pi*DELTAVIR*rhoc_of_z(param)))**(1.0/3.0)
    r500 = r500_fct(rvir,cvir)
    M500 = MNFW_fct(r500,cvir,Mvir,param)

    #Total dark-matter-only mass
    Mtot = Mvir*mTOTtr_fct(tau)/mNFWtr_fct(cvir,tau)
    
    #total fractions
    fbar  = Ob/Om
    fcdm  = (Om-Ob)/Om
    fstar = fSTAR_fct(Mvir,param,eta)
    fcga  = fSTAR_fct(Mvir,param,eta+deta) #Moster13
    fsga  = fstar-fcga #satellites and intracluster light

    figa  = param.baryon.ciga*fcga #param.baryon.ciga*1e12/Mtot #param.baryon.ciga*fcga
    fhga  = fbar-fcga-fsga-figa
    
    if(fsga<0):
        fsga = 0.0
        print('WARNING: negative fraction of satellite galaxies. Set to 0')
        #exit()

    #Initial density and mass profiles
    rho0NFWtr = DELTAVIR*rhoc_of_z(param)*cvir**3.0/(3.0*mNFWtr_fct(cvir,tau))
    rhoNFW = rho0NFWtr*uNFWtr_fct(rbin,cvir,tau,Mvir,param)

    #rho2h  = (cosmo_bias*cosmo_corr + 1.0)*Om*RHOC #rho_b=const in comoving coord.
    
    #  be = 1.0 #1.5
    #  se = 1.2 #1.5
    #  rho2h  = (be*(rbin/(5*rvir/0.55))**(-se)*cosmo_bias*cosmo_corr + 1.0)*Om*RHOC
    #  rho2h  = (be*(rbin/(5*rvir/0.55))**(-se) + 1.0)*Om*RHOC

    #exclscale = 0.5 #0.45 #0.35
    exclscale = param.code.halo_excl
    #rho2h  = (cosmo_bias*cosmo_corr + 1.0)*Om*RHOC #rho_b=const in comoving coord.
    rho2h  = (1-np.exp(-exclscale*rbin/rvir)) * (cosmo_bias*cosmo_corr + 1.0)*Om*RHOC #rho_b=const in comoving coord.
    
    rhoDMO = rhoNFW + rho2h
    MNFW   = MNFWtr_fct(rbin,cvir,tau,Mvir,param)
    M2h    = cumtrapz(4.0*np.pi*rbin**2.0*rho2h,rbin,initial=0.0)
    M2h_tck = splrep(rbin, M2h, s=0, k=3)
    #MDMO   = MNFW + M2h
    #MDMO_tck = splrep(rbin, MDMO, s=0, k=3)
    
    #Final HGA density and mass profile
    #uHGA =  uHGA_fct(rbin,Mvir,eps,param)
    uHGA     = uHGA_fct(rbin,cvir,Mvir,eps,param)
    rho0HGA  = Mtot/(4.0*np.pi*simpson(rbin**2.0*uHGA,rbin))
    rhoHGA   = rho0HGA*uHGA
    MHGA     = cumtrapz(4.0*np.pi*rbin**2.0*rhoHGA,rbin,initial=0.0)
    MHGA_tck = splrep(rbin, MHGA, s=0, k=3)

    #FINAL IGA density and mass profile
    uIGA     = uIGA_fct(rbin,Mvir,param)
    rho0IGA  = Mtot/(4.0*np.pi*simpson(rbin**2.0*uIGA,rbin))
    rhoIGA   = rho0IGA*uIGA
    MIGA     = cumtrapz(4.0*np.pi*rbin**2.0*rhoIGA,rbin,initial=0.0)
    MIGA_tck = splrep(rbin, MIGA, s=0, k=3)

    #Final CGA density and mass profile
    #R12      = param.baryon.rcga*rvir
    #rho0CGA  = Mtot/(4.0*np.pi*R12)
    #rhoCGA   = rho0CGA*uCGA_fct(rbin,Mvir,param)
    #MCGA     = Mtot*MCGA_fct(rbin,Mvir,param) #+ M2h
    #MCGA_tck = splrep(rbin, MCGA, s=0, k=3)
    uCGA     = uCGA_fct(rbin,Mvir,param)
    rho0CGA  = Mtot/(4.0*np.pi*simpson(rbin**2.0*uCGA,rbin))
    rhoCGA   = rho0CGA*uCGA
    MCGA     = cumtrapz(4.0*np.pi*rbin**2.0*rhoCGA,rbin,initial=0.0)
    #MCGA     = cumtrapz(4.0*np.pi*rbin**2.0*rhoCGA,rbin,initial=0.0) + 0.1*fcga*Mtot
    MCGA_tck = splrep(rbin, MCGA, s=0, k=3)
    
    
    ###
    ##Final SGA density and mass profile (cvir=cvir/2, tau=1)
    #taucga  = 1
    #rho0SGA = DELTAVIR*rhoc_of_z(param)*(cvir/2)**3.0/(3.0*mNFWtr_fct(cvir/2,taucga))
    #rhoSGA  = rho0SGA*uNFWtr_fct(rbin,cvir/2,taucga,Mvir,param)
    #MSGA    = MNFWtr_fct(rbin,cvir/2,taucga,Mvir,param) + M2h
    #MSGA_tck = splrep(rbin, MSGA, s=0, k=3)
    ###

    
    #Adiabatic Correction Model 0 (Abadi et al 2010)
    if (param.code.AC_model == 0):
        nn = ACM_q0*(1+zz)**ACM_q0_exp #nn = 1 corresponds to Gnedin 2004
        aa = ACM_q1*(1+zz)**ACM_q1_exp
        func = lambda x: (x-1.0) - aa*(((MNFW + M2h)/((fcdm+fsga)*MNFW + fcga*splev(x*rbin,MCGA_tck,der=0,ext=3) + fhga*splev(x*rbin,MHGA_tck,der=0,ext=3) + figa*splev(x*rbin,MIGA_tck,der=0,ext=3) + splev(x*rbin,M2h_tck,der=0,ext=3)))**nn - 1.0)

        if (isinstance(rbin, float)):
            xi = 1.0
        else:
            xi = np.empty(len(rbin)); xi.fill(1.0)
        xx = fsolve(func, xi, fprime=None)
    
    #Adiabatic Correction Model 1 (Velmani and Paranjape 2023)
    elif (param.code.AC_model == 1):

        Q0 = ACM_q0*(1+zz)**ACM_q0_exp  # Q0=0 corresponds to Gnedin 2004, Teyssier et al 2011 
        Q1 = ACM_q1*(1+zz)**ACM_q1_exp  # Q1 corresponds to q10 in Velmani and Paranjape 2023
        Q2 = ACM_q2*(1+zz)**ACM_q2_exp  # Q2 corresponds to q11 in Velmani and Paranjape 2023
        Q1fct = lambda rf: Q1 + Q2*np.log(rf/rvir)
        func = lambda x: (x-1.0) - Q1fct(x*rbin)*((MNFW + M2h)/((fcdm+fsga)*MNFW + fcga*splev(x*rbin,MCGA_tck,der=0) + fhga*splev(x*rbin,MHGA_tck,der=0) + figa*splev(x*rbin,MIGA_tck,der=0,ext=3) + splev(x*rbin,M2h_tck,der=0,ext=3)) - 1.0) - Q0

        if (isinstance(rbin, float)):
            xi = 1.0
        else:
            xi = np.empty(len(rbin)); xi.fill(1.0)
        xx = fsolve(func, xi, fprime=None)
        
    #Adiabatic Correcion Model 2 (Velmani and Paranjape 2023 with step function instead of Q0)
    elif (param.code.AC_model == 2):

        Q0 = ACM_q0*(1+zz)**ACM_q0_exp
        Q1 = ACM_q1*(1+zz)**ACM_q1_exp
        Q2 = ACM_q2*(1+zz)**ACM_q2_exp #not in use

        #Smooth step function
        nn = 1.5
        rc = eps*rvir/eps0
        fstep = lambda rr: Q0/(1+(rr/rc)**nn)

        func = lambda x: (x-1.0) - Q1*((MNFW + M2h)/((fcdm+fsga)*MNFW + fcga*splev(x*rbin,MCGA_tck,der=0) + fhga*splev(x*rbin,MHGA_tck,der=0) + figa*splev(x*rbin,MIGA_tck,der=0,ext=3) + splev(x*rbin,M2h_tck,der=0,ext=3)) - 1.0) - fstep(x*rbin)
    
        if (isinstance(rbin, float)):
            xi = 1.0
        else:
            xi = np.empty(len(rbin)); xi.fill(1.0)
        xx = fsolve(func, xi, fprime=None)
        
    #Model 3 (smooth step function follwoed by AC model)
    elif (param.code.AC_model == 3):
        Q0 = ACM_q0*(1+zz)**ACM_q0_exp
        Q1 = ACM_q1*(1+zz)**ACM_q1_exp
        Q2 = ACM_q2*(1+zz)**ACM_q2_exp

        #Smooth step function
        nn = 1.5
        rc = eps*rvir/eps0
        fstep = lambda rr: 1 - Q0/(1+(rr/rc)**nn)

        r1 = rbin*fstep(rbin)
        MNFW1 = MNFWtr_fct(r1,cvir,tau,Mvir,param)

        func = lambda x: (x-1.0) - Q2*((MNFW1+splev(r1,M2h_tck,der=0,ext=0))/((fcdm+fsga+Q1*fcga+Q1*figa)*MNFW1 + (1-Q1)*(fcga*splev(x*r1,MCGA_tck,der=0,ext=0) + figa*splev(x*r1,MIGA_tck,der=0,ext=0)) + fhga*splev(x*r1,MHGA_tck,der=0,ext=0) + splev(x*r1,M2h_tck,der=0,ext=0)) - 1.0)

        if (isinstance(rbin, float)):
            xi = 1.0
        else:
            xi = np.empty(len(rbin)); xi.fill(1.0)
        x2 = fsolve(func, xi, fprime=None)
        xx = x2*fstep(rbin)

    #Model 4 (attempt with two stage AC model)
    elif (param.code.AC_model == 4):

        Q0 = ACM_q0*(1+zz)**ACM_q0_exp
        Q1 = ACM_q1*(1+zz)**ACM_q1_exp
        Q2 = ACM_q2*(1+zz)**ACM_q2_exp

        #Smooth step function
        nn = 1.5
        rc = eps*rvir/eps0
        fstep = lambda rr: 1 - Q0/(1+(rr/rc)**nn)

        r1 = rbin*fstep(rbin)
        MNFW1 = MNFWtr_fct(r1,cvir,tau,Mvir,param)

        func = lambda x: x - (MNFW1+splev(r1,M2h_tck,der=0,ext=0))/((fcdm + fsga + (1-Q1)*(fcga+figa) + (1-Q2)*fhga)*MNFW1 + Q1*fcga*splev(x*r1,MCGA_tck,der=0,ext=0) + Q1*figa*splev(x*r1,MIGA_tck,der=0,ext=0) + Q2*fhga*splev(x*r1,MHGA_tck,der=0,ext=0) + splev(x*r1,M2h_tck,der=0,ext=0))

        if (isinstance(rbin, float)):
            xi = 1.0
        else:
            xi = np.empty(len(rbin)); xi.fill(1.0)
        x2 = fsolve(func, xi, fprime=None)
        xx = x2*fstep(rbin)

    #Model 5 (explicit function for AC)
    elif (param.code.AC_model == 5):

        #Q0 = ACM_q0*(1+zz)**ACM_q0_exp
        #Q1 = ACM_q1*(1+zz)**ACM_q1_exp
        #Q2 = ACM_q2*(1+zz)**ACM_q2_exp
        Q0 = ACM_q0 + ACM_q0_exp*zz
        Q1 = ACM_q1 + ACM_q1_exp*zz
        Q2 = ACM_q2 + ACM_q2_exp*zz

	#Smooth step function
        nn = 1.5
        rc = eps*rvir/eps0
        fstep    = lambda rr: 1 + Q0/(1+(rr/rc)**nn)
        ri_ov_rf = lambda rr: fstep(rr) + Q1*fcga*(splev(rr,MCGA_tck,der=0,ext=0)/MNFWtr_fct(rr,cvir,tau,Mvir,param) - 1) + Q1*figa*(splev(rr,MIGA_tck,der=0,ext=0)/MNFWtr_fct(rr,cvir,tau,Mvir,param) - 1) + Q2*fhga*(splev(rr,MHGA_tck,der=0,ext=0)/MNFWtr_fct(rr,cvir,tau,Mvir,param) - 1)

        xx = 1/ri_ov_rf(rbin)
        xx = np.nan_to_num(xx,nan=1.0)

    #Model 6
    elif (param.code.AC_model == 6):

        Q0 = ACM_q0*(1+zz)**ACM_q0_exp
        Q1 = ACM_q1*(1+zz)**ACM_q1_exp
        Q2 = ACM_q2*(1+zz)**ACM_q2_exp

	#Smooth step function                                                                                   
        nn = 1.5
        rc = eps*rvir/eps0
        fstep    = lambda rr: 1 + Q0/(1+(rr/rc)**nn)

        ri_ov_rf = lambda rr: fstep(rr) + fcga*((splev(rr,MCGA_tck,der=0,ext=0)/MNFWtr_fct(rr,cvir,tau,Mvir,param))**Q1 - 1) + figa*((splev(rr,MIGA_tck,der=0,ext=0)/MNFWtr_fct(rr,cvir,tau,Mvir,param))**Q1 - 1) + fhga*((splev(rr,MHGA_tck,der=0,ext=0)/MNFWtr_fct(rr,cvir,tau,Mvir,param))**Q2 - 1)
        
        #prevent zeros
        inv_xx = ri_ov_rf(rbin)
        inv_xx[inv_xx == 0] = 1e-6

        xx = 1/inv_xx
        xx = np.nan_to_num(xx,nan=1.0)
        
    else:
        print("ERROR: Adiabatic Contraction model not known. Exit")
        exit()

    #no adiabatic expansion
    if (param.code.adiab_exp==False):
        xx[xx>1] = 1

    #here rbin means r_final
    MACM     = MNFWtr_fct(rbin/xx,cvir,tau,Mvir,param)
    #hydro sims have lower concentration (2212.05964, Fig.2)
    #MACM     = MNFWtr_fct(rbin/xx,cvir/1.1,tau,Mvir,param)
    MACM_tck = splrep(rbin, MACM, s=0, k=3)
    rhoACM   = splev(rbin,MACM_tck,der=1)/(4.0*np.pi*rbin**2.0)
    MACM     = MACM #+ M2h
    
    if (np.any(np.diff(MACM) <= 0)):
        print("Warning MACM not monotonically rising! Force monotonic rise")
        MACM = np.maximum.accumulate(MACM)

    
    #total profile
    rhoDMB   = (fcdm+fsga)*rhoACM + fhga*rhoHGA + figa*rhoIGA + fcga*rhoCGA
    MDMB     = (fcdm+fsga)*MACM + fhga*MHGA + figa*MIGA + fcga*MCGA
    ###
    #rhoDMB   = fcdm*rhoACM + fsga*rhoSGA + fhga*rhoHGA + fcga*rhoCGA
    #MDMB     = fcdm*MACM + fsga*MSGA + fhga*MHGA + fcga*MCGA
    ###
    #MDMB_tck = splrep(rbin, MDMB, s=0, k=3)
    #MDMBinv_tck = splrep(MDMB, rbin, s=0, k=3)

    
    #Total pressure assuming hydrostatic equilibrium (dP/dr = -rho_gas * G * M/r^2)
    #itg = G * fhga * rhoHGA * (MDMB+M2h) / rbin**2  # Mpc/Msun*(km/s)^2
    #itg = G * fhga * rhoHGA * MDMB / rbin**2  # Mpc/Msun*(km/s)^2
    #Pcumint = cumtrapz(itg,rbin,initial=0.0)
    #P_tot = Pcumint[-1] - Pcumint  # [(km/s)^2*(Msun/h)/(Mpc/h)^3]

    #Nonthermal pressure, Shaw2010 model (1006.1945 Eq16, see also 2009.05558, Eq14)
    if (a_nth>0):
        fmax = 4**(-n_nth)/a_nth
        f1   = (1 + param.cosmo.z)**b_nth
        f2   = (fmax-1)*np.tanh(b_nth*param.cosmo.z) + 1.0
        f_z  = min(f1,f2)
    else:
        f_z = 0
    a_nthz = a_nth * f_z
    
    #Thermal pressure [(km/s)^2*(Msun/h)/(Mpc/h)^3]
    #P_th = P_thermal(rbin,fhga*rhoHGA,(MDMB+M2h),rvir,a_nthz,n_nth,param)
    P_th = P_thermal(rbin,fhga*rhoHGA,MDMB,rvir,a_nthz,n_nth,param)

    #Electron pressure [(km/s)^2*(Msun/h)/(Mpc/h)^3]
    P_e = ((2+2*Xe)/(3+5*Xe))*P_th

    #Gas and electron number density [1/(Mpc/h)^3]
    mean_mol_weight = 0.6125
    n_gas = fhga*rhoHGA/(m_atom*h0*mean_mol_weight)
    n_e   = (1 + Xe)/2 * fhga*rhoHGA/(m_atom*h0*mean_mol_weight)
    
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
    ###
    #dens = { 'NFW':rhoNFW, 'BG':rho2h, 'DMO':rhoDMO, 'ACM':rhoACM, 'CDM':rhoACM, 'SGA':rhoSGA, 'CGA':rhoCGA, 'HGA':rhoHGA, 'DMB':rhoDMB }
    ###

    #True mass with 2-halo term
    mass = { 'NFW':MNFW, 'BG':M2h, 'CDM':fcdm*MACM, 'SGA':fsga*MACM, 'CGA':fcga*MCGA, 'HGA':fhga*MHGA, 'IGA':figa*MIGA, 'DMB':MDMB }
    #mass = { 'NFW':MNFW, 'BG':M2h, 'DMO':MDMO, 'ACM':(fcdm+fsga)*MACM, 'CDM':fcdm*MACM, 'SGA':fsga*MACM, 'CGA':fcga*MCGA, 'HGA':fhga*MHGA, 'DMB':MDMB }
    ###
    pres = { 'thermal': P_th, 'electron': P_e }
    temp = { 'gas': T_gas, 'electron': T_e }
   
    return frac, dens, mass, pres, temp
