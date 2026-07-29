import h5py
import numpy as np
import healpy as hp
from time import time

#i_shell_min = 3
#i_shell_max = 33
# i_shell_min = 10
# i_shell_max = 11
i_shell_min = 0
i_shell_max = 69

hydro_sim = "fgas_8sigma" # use BCM params from Flamingo fits (Schneider et al. 2025) can be "chexmate" | "fiducial" | "fgas_8sigma" | "tng300"

filename_shells_out = f"/cluster/work/refregier/jbucko/shell_baryonification/data/cosmogrid/grid_cosmo_111246_run0/baryonified_shell_{i_shell_min}_{i_shell_max}__cosmogrid_nersc_full_lightcone_FS2_branch_{hydro_sim}.h5"
filename_halos = "/cluster/work/refregier/jbucko/shell_baryonification/data/cosmogrid/grid_cosmo_111246_run0/profiled_halos_v11.h5"
filename_shells = "/cluster/work/refregier/jbucko/shell_baryonification/data/cosmogrid/grid_cosmo_111246_run0/compressed_shells.npz"

# with h5py.File(filename_halos,'r') as f:
#     shell_data = f["/shell_data"][:]

# z_min = shell_data[i_shell_min][1]
# z_max = shell_data[i_shell_max][2]
# # z_shell = 0.5*(z_min + z_max)
# print('z_min, z_max:', z_min, z_max)

# import sys
# sys.path.append('/cluster/home/jbucko/shell_baryonification/')
import baryonification as bfc

par = bfc.par()


par.shell.max_shell = i_shell_max
par.shell.min_shell = i_shell_min
par.shell.nside = 2048
par.shell.nside_out = 2048
par.shell.N_cpu = 128

par.cosmo.Om = 0.3309967
par.cosmo.Ob = 0.04712517
par.cosmo.h0 = 0.67933
par.cosmo.ns = 0.979819
par.cosmo.s8 = 0.85249176



par.code.Mhalo_min = 1e13
par.code.beta_model = 1
par.code.multicomp = True
par.code.eps1 = 0.5             # eps1=0 corresponds to the old case
par.code.q1 =  0.25              # Adiabatic contraction model param Q1 = q1*(1+z)*q1_exp
par.code.q2 =  0.7
par.code.halo_excl = 0.4


if hydro_sim == "chexmate":
    par.baryon.ciga = 0.1
    par.baryon.gamma = 1.5
    par.baryon.nu = -0.38372423 # Mc = Mc_0*10 ** (-nu * z) as in Schneider et al 2025
    par.baryon.Mc = 10**13.99795619 # Mc_0
    par.baryon.mu = 0.33539002
    par.baryon.thco = 0.08383643 # will be * (1+z)**0.5
    par.baryon.delta = 4.23574897

elif hydro_sim == "fiducial":
    par.baryon.Mc          = 10**13.08268916
    par.baryon.thco        = 0.31867105
    par.baryon.mu          = 0.553033
    par.baryon.delta       = 5.76996342
    par.baryon.eta         = 0.0378558
    par.baryon.deta        = 0.22472725
    par.baryon.Nstar       = 0.02718398
    par.baryon.ciga        = 0.1037475
elif hydro_sim == "fgas_8sigma":
    # 13.45878413 0.51973802 4.71687645 0.31916443 0.06044311 0.23963549 0.02687355 0.25362647
    # logMc, mu, delta, thco, eta, deta, Nstar, ciga
    par.baryon.Mc          = 10**13.45878413
    par.baryon.thco        = 0.31916443
    par.baryon.mu          = 0.51973802
    par.baryon.delta       = 4.71687645
    par.baryon.eta         = 0.06044311
    par.baryon.deta        = 0.23963549
    par.baryon.Nstar       = 0.02687355
    par.baryon.ciga        = 0.25362647
elif hydro_sim == "tng300":
    # 1.28761345e+01 6.44209621e-01 5.93059309e+00 2.50067063e-01 4.04978805e-02 1.70204134e-01 1.27288377e-02 9.83909386e-02
    # logMc, mu, delta, thco, eta, deta, Nstar, ciga
    par.baryon.Mc          = 10**12.8761345
    par.baryon.thco        = 0.250067063
    par.baryon.mu          = 0.644209621
    par.baryon.delta       = 5.93059309
    par.baryon.eta         = 0.0404978805
    par.baryon.deta        = 0.170204134
    par.baryon.Nstar       = 0.0127288377
    par.baryon.ciga        = 0.0983909386


par.files.halolc_format = "CosmoGrid_nersc"
par.files.shellfile_format = "CosmoGrid"
par.files.shellfile_in  = filename_shells
par.files.shellfile_out = filename_shells_out
par.files.halolc_in  = filename_halos
par.files.transfct = "/cluster/home/jbucko/baryonification/baryonification/files/CDM_PLANCK_tk.dat"
par.files.output_pixelparticle_file = True
par.files.tmp_files = "/cluster/scratch/jbucko/tmp/grid_cosmo_111246_run0"


# h_list, thickness_list, redshift_list = bfc.read_halo_lc_file(par)
# print('read_halo_lc_file finished. Output:',h_list, thickness_list, redshift_list)
# shell_id, map_list = bfc.read_healpix_file(par)
# print('read_healpix_file finished. Output:',shell_id, map_list[0],map_list[0].shape)
print('entering the perform_shell_displacement function')
# run the shell baryonification
t1 = time()
shell_displacer = bfc.ShellDisplacer(par)
shell_displacer.perform_shell_displacement()
print(f"\nShell baryonification completed in {time()-t1} seconds.")
