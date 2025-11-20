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