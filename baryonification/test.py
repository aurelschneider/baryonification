from params import par
from cosmo import cosmo
from displ import displace, displ_file, displace_from_displ_file

#initialise parameters
par = par()
par.files.transfct = "files/CDM_PLANCK_tk.dat"
par.files.cosmofct = "files/cosmofct.dat"
par.files.displfct = "files/displfct.dat"
par.files.partfile_in = "../../BarCorr9/BESTMATCH2OBS/CDM_L128_N256/out/CDM_L128_N256.00010"
par.files.parfile_out = "/cluster/home/scaurel/dcdcdc/out.std"
par.files.halofile_in = "../../BarCorr9/BESTMATCH2OBS/CDM_L128_N256/ahf_rhoc/files/CDM_L128_N256.00010.z0.000.AHF_halos"

#cosmo(par)


#displ_file(par)

#displace_from_displ_file(par)

displace(par)
