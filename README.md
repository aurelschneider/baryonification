# Baryonification

Code to modify gravity-only N-body simulations accounting for effects from gas, stars, and baryonic feedback on the density distribution. Component wise output for gas stars and dark matter. See https://arxiv.org/abs/2507.07892 for more information.
An earlier version of the code can be found on Bitbucket (https://bitbucket.org/aurelschneider/baryonification/src/master/ with corresponding references https://arxiv.org/abs/1510.06034, https://arxiv.org/abs/1810.08629).

## Installation

To download and install the package, type

    git clone https://github.com/aurelschneider/baryonification.git
    cd baryonification
    pip install .
    
## Quickstart

The script below shows an easy way to use the code:

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
    bfc.displace(par)

All model parameters are defined in baryonification/params.py

##Contact

Please contact Aurel Schneoider (aurel.schneider@uzh.ch) for questions, remarks, bugs etc
