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

