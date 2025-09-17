"""
External Parameters
"""

class Bunch(object):

    #def __init__(self, data):
    #    #translate dic['name'] into dic.name
    #    self.__dict__.update(data)

    def __init__(self, data):
        # Store the allowed keys
        self._allowed_keys = set(data.keys())
        # Use __dict__ to avoid recursion in __setattr__
        self.__dict__.update(data)

    def __setattr__(self, key, value):
        # Allow only predefined keys
        if key != "_allowed_keys" and key not in self._allowed_keys:
            raise AttributeError(f"Cannot set undefined parameter '{key}'.")
        # Directly update __dict__ to prevent recursion
        self.__dict__[key] = value

    #def __getattr__(self, key):
    #    # Raise an error if accessing an undefined key
    #    if key not in self._allowed_keys:
    #        raise AttributeError(f"Cannot access undefined parameter '{key}'.")
    #    # Use __dict__ to get the value directly
    #    return self.__dict__[key]



def cosmo_par():
    par = {
        "z": 0.0,
        "Om": 0.315,
        "Ob": 0.049,
        "s8": 0.83,
        "h0": 0.673,
        "ns": 0.963,
        "dc": 1.675,
        }
    return Bunch(par)

def baryon_par():
    par = {
        "Mc": 3.0e13,     # beta(M,z): critical mass scale
        "mu": 0.3,        # beta(M,z): critical mass scale
        "ciga": 0.1,      # fraction of cold gas
        "nu": 0.0,        # beta(M,c): redshift dependence
        "thco": 0.1,      # core factor thco=rco/rvir
        "alpha": 1.0,     # index in gas profile [default: 1.0]
        "gamma": 1.5,     # index in gas profile [default: 2.0]
        "delta": 7.0,     # index in gas profile [default: 7.0 -> same asympt. behav. than NFWtrunc profile]  
        "rcga": 0.03,     #0.015 # half-light radius of central galaxy (ratio to rvir)
        "Nstar": 0.04,    # Stellar normalisation param [fstar = Nstar*(Mstar/Mvir)**eta]
        "Mstar": 2.5e11,  # Stellar critical mass [fstar = Nstar*(Mstar/Mvir)**eta]
        "eta": 0.32,      # exponent of total stellar fraction [fstar = Nstar*(Mstar/Mvir)**eta]
        "deta": 0.28,     # exponent of central stellar fraction [fstar = Nstar*(Mstar/Mvir)**(eta+deta)]
        "a_nth": 0.18,    #Non-thermal pressure profile (P_nth = a_nth(b_nth(z))*(r/rvir)^n_nth): normalisation
        "n_nth": 0.8,     #Non-thermal pressure profile: power-law index
        "b_nth": 0.5,     #Non-thermal pressure profile: redshift evolution (b_nth = 0 means no z-evolution)
        }
    return Bunch(par)

def io_files():
    par = {
        "transfct": 'CDM_PLANCK_tk.dat',
        "cosmofct": 'cosmofct.dat',
        "displfct": 'displfct.dat',
        "partfile_in": 'partfile_in.std',
        "partfile_out": 'partfile_out.std',
        "partfile_format": 'tipsy',
        "halofile_in": 'halofile_in.dat',
        "halofile_out": 'halofile_out.dat',
        "halofile_format": 'AHF-ASCII',
        "TNGnumber": 99,   #number of TNG files
    }
    return Bunch(par)

def code_par():
    par = {
        "multicomp": True,       #individual displacement of collisionless matter and stars/gas?
        "satgal": True,          #satellite galaxies treated explicitely (only used in multicomp model).
        "adiab_exp": True,       #Adiabatic expansion (turned on in default model)
        "spher_corr": False,      #Sphericity correction (currently only in multicomp model)
        "spher_amplitude": 1.0,  #Amplitude of the sphericity correciotn A=[0,1]
        "spher_scale": 1.0,      #Scale of sphericity correction (r_scale = spher_scale*rvir)
        "spher_powerlaw": 10.0,   #Power-law of the transition 
        "kmin": 0.01,
        "kmax": 100.0,
        "rmin": 0.005,
        "rmax": 50.0,
        "Nrbin": 100,
        "rbuffer": 10.0,        # buffer size to take care of boundary conditions
        "eps0": 4.0,            # truncation factor: eps=rtr/rvir with eps = eps0 - eps1*nu
        "eps1": 0.5,            # eps1=0 corresponds to the old case
        "beta_model": 1,        # 0: old model from Schneider+18 1: new model
        "AC_model": 5,          # 0: Abadi2010, 1: Velmani&Paranjape2023, 2:stepfct followed by AC, 5: empirical model (default)
        "q0": 0.075,            # Adiabatic contraction model param Q0 = q0*(1+z)*q0_exp
        "q0_exp": 0.0,          # Exponent of adiabatic contraction param Q0
        "q1": 0.25,              # Adiabatic contraction model param Q1 = q1*(1+z)*q1_exp
        "q1_exp": 0.0,          # Exponent of adiabatic contraction param Q1
        "q2": 0.8,              
        "q2_exp": 0.0,
        "Mhalo_min": 2.5e11,    # Minimum halo mass [Msun/h]
        "disp_trunc": 0.01,     # Truncation of displacment funct (disp=0 if disp<disp_trunc) [Mpc/h]
        "halo_excl": 0.4        # halo exclusion parameter (no exclusion = large number e.g. 1000)
        }
    return Bunch(par)

def sim_par():
    par = {
        "Lbox": 128.0,   #box size of partfile_in
        "rbuffer": 10.0, #buffer size to take care of boundary conditions
        #"Nmin_per_halo": 100,
        "N_chunk": 1      #number of chunks (for multiprocesser: n_core = N_chunk^3)
        }
    return Bunch(par)

def par():
    par = Bunch({"cosmo": cosmo_par(),
        "baryon": baryon_par(),
        "files": io_files(),
        "code": code_par(),
        "sim": sim_par(),
        })
    return par
