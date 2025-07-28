import numpy as np
from scipy import spatial
from scipy.interpolate import splrep,splev
from numpy.lib.recfunctions import append_fields

import schwimmbad

from .constants import *
from .profiles import Profiles
from .cosmo import CosmoCalculator
from .io_utils import IO_nbody, IO_halo


class HaloDisplacer:
    """
    Class to handle halo displacement.
    """
    def __init__(self,param):
        self.param = param

    def displ(self, rbin, MINITIAL, MFINAL):
        """
        Calculates the displacement of all particles as a function
        of the radial distance from the halo centre
        MFINAL = MDMB or MBAR, ..
        """

        MFINAL_tck = splrep(rbin, MFINAL, s=0, k=3)
        MFINALinv_tck=splrep(MFINAL, rbin, s=0, k=3)
        rFINAL = splev(MINITIAL,MFINALinv_tck,der=0)
        DFINAL = rFINAL - rbin

        return DFINAL

    def displace_haloes(self):
        """
        Reading halo files looping over haloes,
        dispalcing haloes
        """
        #Force serial mode
        self.param.sim.N_chunk = 1
        
        #Read in halo file, build chunks and buffer
        halo_io = IO_halo(self.param)
        h_list = halo_io.read()
        h_list = h_list[0]

        h_displ = self.displace_halo_chunk(h_list)

        print("h_displ", min(h_displ['x']),max(h_displ['x']))

        #hack
        halo_io.write(h_displ)

        return

    def displace_halo_chunk(self, h_chunk):
        """
        Reading halo file, looping over haloes, calculating
        displacements, and dispalcing haloes.
        """
        #relevant parameters
        Lbox = self.param.sim.Lbox

        #Read cosmic variance/nu/correlation and interpolate
        cosmo = CosmoCalculator(self.param)
        vc_r, vc_m, vc_var, vc_bias, vc_corr = cosmo.compute_cosmology()
        var_tck  = splrep(vc_m, vc_var, s=0)
        bias_tck = splrep(vc_m, vc_bias, s=0)
        corr_tck = splrep(vc_r, vc_corr, s=0)

        if (self.param.code.multicomp==True):

            #Copy into p_temp
            H_dt = np.dtype([("x",'>f'),("y",'>f'),("z",'>f'),("id",'>i4')])
            Hp   = np.zeros(len(h_chunk),dtype=H_dt)
            
            #Build tree for dm and bar
            print('building tree..')
            h_tree = spatial.cKDTree(list(zip(h_chunk['x'],h_chunk['y'],h_chunk['z'])), leafsize=100)
            print('...done!')


            #Loop over haloes, calculate displacement, and displace neighbouring haloes
            for i in range(len(h_chunk['Mvir'])):

                #select host haloes (subhaloes >= 1)
                if (h_chunk['IDhost'][i] < 0.1):
                    
                    print('start: ', i)

                    #range where we consider displacement
                    rmax = self.param.code.rmax
                    rmin = (0.001*h_chunk['rvir'][i] if 0.001*h_chunk['rvir'][i]>self.param.code.rmin else self.param.code.rmin)
                    rmax = (20.0*h_chunk['rvir'][i] if 20.0*h_chunk['rvir'][i]<self.param.code.rmax else self.param.code.rmax)
                    rbin = np.logspace(np.log10(rmin),np.log10(rmax),100,base=10)

                    #initialse profiles
                    cosmo_var  = splev(h_chunk['Mvir'][i],var_tck)
                    cosmo_bias = splev(h_chunk['Mvir'][i],bias_tck)
                    cosmo_corr = splev(rbin,corr_tck)
                    profiles = Profiles(rbin, h_chunk['Mvir'][i], h_chunk['cvir'][i], cosmo_corr, cosmo_bias, cosmo_var, self.param)

                    #calculate profiles for displacement
                    frac, dens, mass, pres, temp = profiles.calc_profiles()

                    #collisional (final dark) matter displacement 
                    DFDM = self.displ(rbin, frac['CDM']*(mass['NFW'] + mass['BG']), mass['CDM'] + frac['CDM']*mass['BG'])
                    DFDM_tck = splrep(rbin, DFDM,s=0,k=3)

                    #define minimum displacement
                    smallestD = 0.01 #Mpc/h

                    #array of idx with DFDM > Dsmallest
                    idx_FDM = np.where(abs(DFDM) > smallestD)
                    idx_FDM = idx_FDM[:][0]
                    if (len(idx_FDM)>1):
                        idx_largest = idx_FDM[-1]
                        rball = rbin[idx_largest]
                    else:
                        rball = 0.0
                
                    #consistency check:
                    print('rball/rvir = ', rball/h_chunk['rvir'][i])

                    print('Mvir, cvir = ', h_chunk['Mvir'][i], h_chunk['cvir'][i])
                    if (rball>Lbox/2.0):
                        print('rball = ', rball)
                        print('ERROR: REDUCE RBALL!')
                        exit()

                    #halo ids within rball
                    ihbool = np.array(h_tree.query_ball_point((h_chunk['x'][i],h_chunk['y'][i],h_chunk['z'][i]),rball))

                    #remove the main halo i
                    ihbool = np.setdiff1d(ihbool,i)

                
                #calculating radii of FDM particles around halo i
                if (rball>0.0 and len(ihbool) > 0):
                    rhFDM = ((h_chunk['x'][ihbool]-h_chunk['x'][i])**2.0 +
                            (h_chunk['y'][ihbool]-h_chunk['y'][i])**2.0 +
                            (h_chunk['z'][ihbool]-h_chunk['z'][i])**2.0)**0.5

                    DrhFDM = splev(rhFDM,DFDM_tck,der=0,ext=1)
                    Hp['x'][ihbool] += (h_chunk['x'][ihbool]-h_chunk['x'][i])*DrhFDM/rhFDM
                    Hp['y'][ihbool] += (h_chunk['y'][ihbool]-h_chunk['y'][i])*DrhFDM/rhFDM
                    Hp['z'][ihbool] += (h_chunk['z'][ihbool]-h_chunk['z'][i])*DrhFDM/rhFDM

            h_chunk['x'] += Hp['x']
            h_chunk['y'] += Hp['y']
            h_chunk['z'] += Hp['z']

        elif (self.param.code.multicomp==False):

            #Copy into p_temp
            H_dt = np.dtype([("x",'>f'),("y",'>f'),("z",'>f'),("id",'>i4')])
            Hp   = np.zeros(len(h_chunk),dtype=H_dt)

            #Build tree for dm and bar
            print('building tree..')
            h_tree = spatial.cKDTree(list(zip(h_chunk['x'],h_chunk['y'],h_chunk['z'])), leafsize=100)
            print('...done!')


            #Loop over haloes, calculate displacement, and displace neighbouring haloes
            for i in range(len(h_chunk['Mvir'])):

                #select host haloes (subhaloes >= 1)
                if (h_chunk['IDhost'][i] < 0.1):

                    print('start: ', i)

                    #range where we consider displacement
                    rmax = self.param.code.rmax
                    rmin = (0.001*h_chunk['rvir'][i] if 0.001*h_chunk['rvir'][i]>self.param.code.rmin else self.param.code.rmin)
                    rmax = (20.0*h_chunk['rvir'][i] if 20.0*h_chunk['rvir'][i]<self.param.code.rmax else self.param.code.rmax)
                    rbin = np.logspace(np.log10(rmin),np.log10(rmax),100,base=10)

                    #calculate displacement
                    cosmo_var  = splev(h_chunk['Mvir'][i],var_tck)
                    cosmo_bias = splev(h_chunk['Mvir'][i],bias_tck)
                    cosmo_corr = splev(rbin,corr_tck)
                    frac, dens, mass, pres, temp = profiles(rbin,h_chunk['Mvir'][i],h_chunk['cvir'][i],cosmo_corr,cosmo_bias,cosmo_var,self.param)

                    #collisional (final) matter displacement
                    #DDMB = displ(rbin,mass['DMO'],mass['DMB'])
                    DDMB = self.displ(rbin,mass['NFW']+mass['BG'],mass['DMB']+mass['BG'])
                    DDMB_tck = splrep(rbin, DDMB,s=0,k=3)
                    
                    #define minimum displacement
                    smallestD = 0.01 #Mpc/h

            #array of idx with D>Dsmallest
                    idx = np.where(abs(DDMB) > smallestD)
                    idx = idx[:][0]
                    if (len(idx)>1):
                        idx_largest = idx[-1]
                        rball = rbin[idx_largest]
                    else:
                        rball = 0.0

                    #consistency check:
                    print('rball/rvir = ', rball/h_chunk['rvir'][i])

                    print('Mvir, cvir = ', h_chunk['Mvir'][i], h_chunk['cvir'][i])
                    if (rball>Lbox/2.0):
                        print('rball = ', rball)
                        print('ERROR: REDUCE RBALL!')
                        exit()

                    #halo ids within rball
                    ihbool = np.array(h_tree.query_ball_point((h_chunk['x'][i],h_chunk['y'][i],h_chunk['z'][i]),rball))

                    #remove the main halo i
                    ihbool = np.setdiff1d(ihbool,i)
                    
                #calculating radii
                if (rball>0.0 and len(ihbool) > 0):
                    rhDMB = ((h_chunk['x'][ihbool]-h_chunk['x'][i])**2.0 +
                            (h_chunk['y'][ihbool]-h_chunk['y'][i])**2.0 +
                            (h_chunk['z'][ihbool]-h_chunk['z'][i])**2.0)**0.5

                    DrhDMB = splev(rhDMB,DDMB_tck,der=0,ext=1)
                    Hp['x'][ihbool] += (h_chunk['x'][ihbool]-h_chunk['x'][i])*DrhDMB/rhDMB
                    Hp['y'][ihbool] += (h_chunk['y'][ihbool]-h_chunk['y'][i])*DrhDMB/rhDMB
                    Hp['z'][ihbool] += (h_chunk['z'][ihbool]-h_chunk['z'][i])*DrhDMB/rhDMB


            h_chunk['x'] += Hp['x']
            h_chunk['y'] += Hp['y']
            h_chunk['z'] += Hp['z']
            
        else:
            print("param.code.multicomp must be either True or False. Abort")
            exit()

        return h_chunk
