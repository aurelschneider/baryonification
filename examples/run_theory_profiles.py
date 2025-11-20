import baryonification as bfc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

print('Set up the paths. Currently working directory is assumed to be "examples"')

# initialise parameters                                                                            
par = bfc.par()

#set paths to input-files
par.files.transfct     = '../baryonification/files/CDM_PLANCK_tk.dat'

#radial bins
rbin = np.logspace(np.log10(0.005),np.log10(50),100,base=10)

#mass and concetration of profile
Mvir = 1e13
cvir = 10

#calculate thing related to 2-halo term
CosmoCalculator = bfc.cosmo.CosmoCalculator(par)
vc_r, vc_m, vc_var, vc_bias, vc_corr = CosmoCalculator.compute_cosmology()
var_tck  = splrep(vc_m, vc_var, s=0)
bias_tck = splrep(vc_m, vc_bias, s=0)
corr_tck = splrep(vc_r, vc_corr, s=0)
cosmo_var  = splev(Mvir,var_tck)
cosmo_bias = splev(Mvir,bias_tck)
cosmo_corr = splev(rbin,corr_tck)

#calcualte fractions and density, mass, pressure, temperature profiles
Profiles = bfc.Profiles(rbin,Mvir,cvir,cosmo_corr,cosmo_bias,cosmo_var,par)
frac, den, mass, press, temp = Profiles.calc_profiles()

#plot density profiles
fig = plt.figure(dpi=120)
plt.loglog(rbin,frac['HGA']*den['HGA'], color='blue', label='Hot gas profile')
plt.loglog(rbin,den['DMO'], color='red', label='Dark matter profile')
plt.loglog(rbin,frac['CGA']*den['CGA']+frac['SGA']*den['SGA'], color='yellow', label='Stellar profile (central + satellites)')
plt.loglog(rbin,den['DMB']+den['BG'], color='black', label='Total profile (incl 2-halo term)')
plt.xlabel('r [cMpc/h]')
plt.ylabel('density [h^2 Msun/cMpc^3]')
plt.legend()
plt.show()