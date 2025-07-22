import baryonification as bfc

#initialise parameters
par = bfc.par()


#path to transfer function (e.g. from class/camb)
par.files.transfct = "TF/CDM_PLANCK_tk.dat"

#path to N-body file (e.g. tipsy/gadget)
par.files.partfile_in  = "sim/output.00100"

#path to halo file (e.g. AHF/ Rockstar)
par.files.halofile_in  = "sim/halofiles/haloes.00100"

#path to baryonified output file
par.files.partfile_out = "sim/baryonified.output.00100"

#change parameters from their default values (see params.py)
par.cosmo.Om = 0.315
par.baryon.Mc    = 1e13
par.baryon.mu    = 0.7
par.baryon.delta = 5.0


#single (all matter) or multi-component (gas, stars, dark matter)
par.code.multicomp = True

#redshift
par.cosmo.z  = 0.0

#calculate 2-halo term
bfc.cosmo(par)

#baryonify
bfc.displace(par)
