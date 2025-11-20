# Baryonification

Code to modify gravity-only N-body simulations accounting for effects from gas, stars, and baryonic feedback on the density distribution. Component-wise output for gas stars and dark matter. See https://arxiv.org/abs/2507.07892 for more information.
An earlier version of the code can be found on Bitbucket (https://bitbucket.org/aurelschneider/baryonification/src/master/ with corresponding references https://arxiv.org/abs/1510.06034, https://arxiv.org/abs/1810.08629).

## Installation

To download and install the package, type

    git clone https://github.com/aurelschneider/baryonification.git
    cd baryonification
    pip install .
    
## Quickstart

The scripts below provide minimal examples to use the code (scripts can also be found in `examples` folder). The default values of all the model parameters are defined in `baryonification/params.py`. 

Analytical profiles can be obtained directly without the need to baryonify. Here is a minimal example:
```python
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
```

Example to baryonify the dark matter-only simulation snapshot (data necessary for this example will be downloaded automatically, ~2.6G):

```python
from __future__ import annotations
import pathlib, subprocess
import baryonification as bfc

print('Set up the paths. Currently working directory is assumed to be "examples"')

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
RECORD_ID = "17660327"  # your Zenodo_ID for required files
CACHE_DIR = pathlib.Path(RECORD_ID).resolve()

BOX_FILE  = CACHE_DIR / "results.00100"                   # simulation name on Zenodo (DO NOT CHANGE)
HALO_FILE = CACHE_DIR / "AHF_halos.00100.0000.z0.000.AHF_halos"      # Halofile name on Zenodo (DO NOT CHANGE)
PATH = './'
PATH_tk = '../baryonification/files/CDM_PLANCK_tk.dat'
# ----------------------------------------------------------------------
# 1. Ensure data exist (run zenodo_dl.py if missing)
# ----------------------------------------------------------------------
if not BOX_FILE.exists() or not HALO_FILE.exists():
    print("\n🔹 Required data not found locally. Downloading from Zenodo...\n")
    subprocess.run(["python", PATH+"zenodo_dl.py", RECORD_ID], check=True)
else:
    print("✅ All required data already downloaded.\n")

#initialise parameters                                                                                                                                                                                                                                                                     
par = bfc.par()

#set paths to input-files
par.files.transfct     = PATH_tk
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

#Set number of chunks for parallelisation (N_chunk=i means i^3 divisions of the box = number of CPUs requested)
par.sim.N_chunk = 1

#baryonify
part_displ = bfc.ParticleDisplacer(par)
part_displ.displace()
```

And finally, the example to perform the baryonification at the level of lightcone dark matter density shells (data necessary for this example will be downloaded automatically, ~2.6G):
```python
from __future__ import annotations
import pathlib, subprocess
import baryonification as bfc
from time import time

print('Set up the paths. Currently working directory is assumed to be "examples"')

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
RECORD_ID = "17660327"  # your Zenodo_ID for required files
CACHE_DIR = pathlib.Path(RECORD_ID).resolve()

SHELL_FILE  = CACHE_DIR / "baryonified_shells_v11.h5"                   # simulation name on Zenodo (DO NOT CHANGE)
HALO_FILE = CACHE_DIR / "profiled_halos_v11.h5"      # Halofile name on Zenodo (DO NOT CHANGE)
PATH = './'
PATH_tk = '../baryonification/files/CDM_PLANCK_tk.dat'
# ----------------------------------------------------------------------
# 1. Ensure data exist (run zenodo_dl.py if missing)
# ----------------------------------------------------------------------
if not SHELL_FILE.exists() or not HALO_FILE.exists():
    print("\n🔹 Required data not found locally. Downloading from Zenodo...\n")
    subprocess.run(["python", PATH+"zenodo_dl.py", RECORD_ID], check=True)
else:
    print("✅ All required data already downloaded.\n")

# ----------------------------------------------------------------------
# RUN SHELL-BFC example
# ----------------------------------------------------------------------

# setup - baryonify shells between min_shell and max_shell.
# Available shell IDs in this example: 0-68
min_shell = 4 # ID for low-z lightcone shell
max_shell = 5 # ID for high-z lightcone shell

# initialize bfc params
par = bfc.par()

# cosmo params
par.cosmo.Om = 0.26
par.cosmo.Ob = 0.0493
par.cosmo.h0 = 0.673

# baryon params
thco = 0.31899051
mc = 10**13.18646411
mu = 0.84270811
de = 5.20774146
et = 0.04284131
dt = 0.21069236
Ns = 0.0277226
cg = 0.05961302
par.baryon.Mc = mc
par.baryon.mu = mu
par.baryon.delta = de
par.baryon.thco = thco
par.baryon.eta  = et
par.baryon.deta = dt
par.baryon.Nstar = Ns
par.baryon.ciga = cg

# shell bfc params
par.shell.N_cpu = 6
par.shell.max_shell = max_shell
par.shell.min_shell = min_shell
par.shell.nside = 1024
par.shell.nside_out = 1024

# file params
par.files.halolc_format = "CosmoGrid_nersc"
par.files.shellfile_format = "CosmoGrid_nersc"
par.files.shellfile_in  = str(SHELL_FILE)
par.files.shellfile_out = f"SHELL_BFC_cosmogrid_fiducial_run_0000_shell_{min_shell}_{max_shell}.h5"
par.files.halolc_in  = str(HALO_FILE)
par.files.transfct = PATH_tk


# run the shell baryonification
t1 = time()
shell_displacer = bfc.ShellDisplacer(par)
shell_displacer.perform_shell_displacement()
print(f"\nShell baryonification completed in {time()-t1} seconds.")
```

    
## Contact

Please contact Aurel Schneider (aurel.schneider@uzh.ch) for questions, remarks, bugs etc.


