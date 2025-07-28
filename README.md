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

    #import module
    import baryonification as bfc

    #initialise parameters                                                                                                                                                                                                                                                                     
    par = bfc.par()

    #set paths to input-files
    par.files.transfct     = "path_to_transfer_function/CDM_PLANCK_tk.dat"
    par.files.partfile_in  = "path_to_nbody_file/CDM_L128_N256.00100
    par.files.halofile_in  = "path_to_halo_file/CDM_L128_N256.00100.z0.000_halos

    #set path to output-file
    par.files.partfile_out = "path_to_output_file/BFC-CDM_L128_N256.00100

    #modify model parameters (all default values listed in baryonification/params.py)
    par.baryon.Mc    = 1e13
    par.baryon.mu    = 0.5
    par.baryon.delta = 5.0

    #set simulation box and redshift
    par.sim.Lbox = 128
    par.cosmo.redshift = 0.0

    #Set number of chunks for paralleisation (N_chunk=i means 2^i chunks)
    par.sim.N_chunk = 2

    #baryonify
    part_displ = bfc.ParticleDisplacer(par)
    part_displ.displace()
    
All model parameters are defined in baryonification/params.py.

Analytical profiles can be obtained directly without the need to baryonify. Here is a minimal example:

    #import modules
    import baryonification as bfc
    import numpy as np
    import matplotlib.pyplot as plt

    #initialise parameters                                                                                                                                                                                                                                                                     
    par = bfc.par()

    #set paths to input-files
    par.files.transfct     = "path_to_transfer_function/CDM_PLANCK_tk.dat"

    #radial bins
    rbin = np.logspace(np.log10(0.005),np.log10(50),100,base=10)
    
    #mass and concetration of profile
    Mvir = 1e13
    cvir = 10

    cosmo = bfc.CosmoCalculator(par)
    vc_r, vc_m, vc_var, vc_bias, vc_corr = cosmo.compute_cosmology()

    var_tck  = splrep(vc_m, vc_var, s=0)
    bias_tck = splrep(vc_m, vc_bias, s=0)
    corr_tck = splrep(vc_r, vc_corr, s=0)

    cosmo_var  = splev(Mvir,var_tck)
    cosmo_bias = splev(Mvir,bias_tck)
    cosmo_corr = splev(rbin,corr_tck)

    #calcualte fractions and density, mass, pressure, temperature profiles
    profiles = bfc.Profiles(rbin, Mvir, cvir, cosmo_bias, cosmo_corr, cosmo_var, par)
    frac, den, mass, pres, temp = profiles.calc_profiles()

    #plot density profiles
    fig = plt.figure(dpi=120)
    plt.loglog(rbin,frac['HGA']*den['HGA'], color='blue', label='Hot gas profile')
    plt.loglog(rbin,den['DMO'], color='red', label='Dark matter profile')
    plt.loglog(rbin,frac['CGA']*den['CGA'] + frac['SGA']*den['SGA'], color='yellow', label='Stellar profile (central + satellites)')
    plt.loglog(rbin,den['DMB'], color='black', label='Total profile (incl 2-halo term)')
    plt.legend()
    plt.show()

    
## Contact

Please contact Aurel Schneider (aurel.schneider@uzh.ch) for questions, remarks, bugs etc.


