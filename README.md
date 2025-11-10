# Baryonification

Code to modify gravity-only N-body simulations accounting for effects from gas, stars, and baryonic feedback on the density distribution. Component-wise output for gas stars and dark matter. See https://arxiv.org/abs/2507.07892 for more information.
An earlier version of the code can be found on Bitbucket (https://bitbucket.org/aurelschneider/baryonification/src/master/ with corresponding references https://arxiv.org/abs/1510.06034, https://arxiv.org/abs/1810.08629).

## Installation

To download and install the package, type

    git clone https://github.com/aurelschneider/baryonification.git
    cd baryonification
    pip install .
    
## Quickstart

The script below provides a minimal example to use the code:
```python
from __future__ import annotations
import os, sys, math, pathlib, requests, subprocess
from typing import Optional
import baryonification as bfc

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
RECORD_ID = "17525781"  # your Zenodo_ID for required files
CACHE_DIR = pathlib.Path(RECORD_ID).resolve()

BOX_FILE  = CACHE_DIR / "results.00100"                   # simulation name on Zenodo (DO NOT CHANGE)
HALO_FILE = CACHE_DIR / "AHF_halos.00100.0000.z0.000.AHF_halos"      # Halofile name on Zenodo (DO NOT CHANGE)

# ----------------------------------------------------------------------
# 1. Ensure data exist (run zenodo_dl.py if missing)
# ----------------------------------------------------------------------
if not BOX_FILE.exists() or not HALO_FILE.exists():
    print("\n🔹 Required data not found locally. Downloading from Zenodo...\n")
    subprocess.run(["python", "zenodo_dl.py", RECORD_ID], check=True)
else:
    print("✅ All required data already downloaded.\n")

#initialise parameters                                                                                                                                                                                                                                                                     
par = bfc.par()

#set paths to input-files
par.files.transfct     = './baryonification/baryonification/files/CDM_PLANCK_tk.dat'
par.files.partfile_in  = str(BOX_FILE)
par.files.halofile_in  = str(HALO_FILE)

#set path to output-file
par.files.partfile_out = "BFC_CDM_L128_N256.00100"

#modify model parameters (all default values listed in baryonification/params.py)
par.baryon.Mc    = 1e13
par.baryon.mu    = 0.5
par.baryon.delta = 5.0

#set simulation box and redshift
par.sim.Lbox = 128
par.cosmo.z = 0.0

#Set number of chunks for paralleisation (N_chunk=i means 2^i chunks)
par.sim.N_chunk = 1

#baryonify
part_displ = bfc.ParticleDisplacer(par)
part_displ.displace()
```
    
All model parameters are defined in baryonification/params.py.

Analytical profiles can be obtained directly without the need to baryonify. Here is a minimal example:

```python
import baryonification as bfc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev

# initialise parameters                                                                            
par = bfc.par()

#set paths to input-files
par.files.transfct     = "path_to_transfer_function/CDM_PLANCK_tk.dat"

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
```

    
## Contact

Please contact Aurel Schneider (aurel.schneider@uzh.ch) for questions, remarks, bugs etc.


