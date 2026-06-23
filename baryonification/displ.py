import numpy as np
from scipy import spatial
import pickle as pkl
from scipy.interpolate import splrep,splev

import schwimmbad

# Force tqdm to use threading-based locks (no semaphores)
from tqdm import tqdm
import gc, sys
from time import time
from cosmic_toolbox import logger

from .constants import *
from .cosmo import CosmoCalculator
from .profiles import Profiles, fstar_fct
from .io_utils import IO_nbody, IO_halo, IO_shell
from .shell_utils import *

LOGGER = logger.get_logger(__name__)

class ParticleDisplacer:
    """
    Class to handle particle displacement.
    """
    def __init__(self,param):
        self.param = param

    def displ(self, rbin, MINITIAL, MFINAL):
        """
        Calculates the displacement 
        """
        MFINAL_tck = splrep(rbin, MFINAL, s=0, k=3)
        MFINALinv_tck = splrep(MFINAL, rbin, s=0, k=3)
        rFINAL = splev(MINITIAL, MFINALinv_tck, der=0)
        DFINAL = rFINAL - rbin
        return DFINAL
    
    def displace(self):
        """
        Main function to do the displacement.
        Reading in N-body and halo files, defining chunks, 
        calling displace_chunk()
        combining chunks, writing N-body file
        """
        nbody_io = IO_nbody(self.param)
        halo_io = IO_halo(self.param)
        p_header, p_list = nbody_io.read()
        h_list = halo_io.read()
        N_chunk = self.param.sim.N_chunk
        N_cpu = int(N_chunk**3)
        print('N_cpu = ', N_cpu)

        if self.param.code.multicomp:
            p_header['Ngas'] = p_header['Npart']
            p_header['Npart'] = int(2 * p_header['Npart'])

        if N_cpu == 1:
            displ_p = self.displace_chunk(p_header, p_list[0], h_list[0])
            p_gas_displ = [displ_p[0]]
            p_dm_displ = [displ_p[1]]
            p_star_displ = [displ_p[2]]
        elif N_cpu > 1:
            pool = schwimmbad.choose_pool(mpi=False, processes=N_cpu)
            tasks = list(zip(p_list, h_list, np.repeat(p_header, N_cpu)))
            displ_p = np.array(pool.map(self.worker, tasks))
            p_gas_displ = displ_p[:, 0]
            p_dm_displ = displ_p[:, 1]
            p_star_displ = displ_p[:, 2]
            pool.close()
        
        #combine chunks
        p_gas = np.concatenate(p_gas_displ)
        p_dm = np.concatenate(p_dm_displ)
        p_star = np.concatenate(p_star_displ)

        #correct header
        p_header['Ngas'] = len(p_gas)
        p_header['Nstar'] = len(p_star)

        #write output
        nbody_io.write(p_header, p_gas, p_dm, p_star)

        return
    
    def worker(self, task):
        """
        Worker for multi-processing
        """
        p_chunk, h_chunk, p_header = task
        p_gas_chunk, p_dm_chunk, p_star_chunk = self.displace_chunk(p_header, p_chunk, h_chunk)
        return np.array([p_gas_chunk, p_dm_chunk, p_star_chunk], dtype=object)


    def particle_separation(self, p_chunk):
        """
        Separate DMO particles into DM and baryonic particles.
        """
        Ob = self.param.cosmo.Ob
        Om = self.param.cosmo.Om

        p_darkmatter = p_chunk.copy()
        p_baryons = p_chunk.copy()

        f_baryons = Ob / Om
        p_darkmatter['mass'] = ((1 - f_baryons) * p_darkmatter['mass']).astype(np.float32)
        p_baryons['mass'] = (f_baryons * p_baryons['mass']).astype(np.float32)

        return p_darkmatter, p_baryons

    def rotation_matrix(self, vec, axis='z-axis'):
        """
        Find rotation matrix for coordinate transformation that puts vec
        along axis.
        """
        a = (vec / np.linalg.norm(vec)).reshape(3)
        if axis == 'z-axis':
            unitvec = np.array([0, 0, 1])
        elif axis == 'y-axis':
            unitvec = np.array([0, 1, 0])
        elif axis == 'x-axis':
            unitvec = np.array([1, 0, 0])
        else:
            print("axis needs to be either x,y,or z. Exit.")
            exit()
        v = np.cross(a, unitvec)
        c = np.dot(a, unitvec)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    def sph_corr(self, rbin, rvir, A, n, th):
        """
        Amplitude and transition of the sphericity correction.
        sph_corr = A for r<<th*rvir and 0 for r>>th*rvir
        Sphericity correction (inspired by Fig3 of https://arxiv.org/pdf/2109.00012.pdf)
        """
        #return 1/((1 + A*(th*rvir/rbin)**n) / (1+((th*rvir/rbin)**n)))
        return A * (th * rvir / rbin) ** n / (1 + (th * rvir / rbin) ** n)

    def displace_chunk(self, p_header, p_chunk, h_chunk):
        """
        Reading in N-body and halo files, looping over haloes, calculating
        displacements, and dispalcing particles.
        Combines functions displ_file() and displace_from_displ_file().
        """
        #relevant parameters
        Lbox = self.param.sim.Lbox
        
        #calculate cosmo for 2-halo term
        cosmo = CosmoCalculator(self.param)
        vc_r, vc_m, vc_var, vc_bias, vc_corr = cosmo.compute_cosmology()
        var_tck  = splrep(vc_m, vc_var, s=0)
        bias_tck = splrep(vc_m, vc_bias, s=0)
        corr_tck = splrep(vc_r, vc_corr, s=0)    
        
        if (self.param.code.multicomp==True):

            #Copy into p_temp
            #Dp_type = np.dtype([("x",'>f'),("y",'>f'),("z",'>f'),("id",'>i4'),("temp",'>f')])
            Dp_type = np.dtype([("x",'>f'),("y",'>f'),("z",'>f'),("id",'>i4'),("temp",'>f'),("pres",'>f')])
            DpBAR = np.zeros(len(p_chunk),dtype=Dp_type)
            DpFDM = np.zeros(len(p_chunk),dtype=Dp_type)
            DpFDM_sph = np.zeros(len(p_chunk),dtype=Dp_type)

            
            #Build tree for dm and bar
            print('building tree..')
            p_tree = spatial.cKDTree(list(zip(p_chunk['x'],p_chunk['y'],p_chunk['z'])), leafsize=100)
            print('...done!')
            
            #id of all halo particle (ihalo=0 means field particles)
            iphalo = np.zeros(len(p_chunk))
            for i in range(len(h_chunk['Mvir'])):
                ip   = np.array(p_tree.query_ball_point((h_chunk['x'][i],h_chunk['y'][i],h_chunk['z'][i]), h_chunk['rvir'][i]))
                if len(ip)>0:
                    iphalo[ip] = i

            #separating particles
            p_darkmatter, p_baryons = self.particle_separation(p_chunk)
            del p_chunk
            
            print("Nhalo_chunk = ", len(h_chunk['Mvir']))

            #Loop over haloes, calculate displacement, and displace partricles
            
            for i in range(len(h_chunk['Mvir'])):
            # for i in tqdm(idx_local):
                #select host haloes (subhaloes >= 1)
                if (h_chunk['IDhost'][i] < 0):
                    
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

                    #baryon displacement
                    DBAR = self.displ(rbin, (1-frac['CDM'])*(mass['NFW'] + mass['BG']), mass['HGA'] + mass['IGA'] + mass['CGA'] + mass['SGA'] + (1-frac['CDM'])*mass['BG'])
                    DBAR_tck = splrep(rbin, DBAR, s=0, k=3)
                    #DBAR_tck = make_interp_spline(rbin, DBAR, k=1)
                    
                    #collisionless (final dark) matter displacement 
                    DFDM = self.displ(rbin, frac['CDM']*(mass['NFW'] + mass['BG']), mass['CDM'] + frac['CDM']*mass['BG'])
                    DFDM_tck = splrep(rbin, DFDM,s=0,k=3)

                    #gas temperature
                    TGAS = temp['electron']
                    TGAS_tck = splrep(rbin,TGAS,s=0,k=1)

                    #gas pressure
                    PGAS = pres['electron']
                    PGAS_tck = splrep(rbin,PGAS,s=0,k=1)
                    
                    #define minimum displacement
                    smallestD = self.param.code.disp_trunc #Mpc/h

                    #array of idx with DBAR > Dsmallest
                    idx_BAR = np.where(abs(DBAR) > smallestD)
                    idx_BAR = idx_BAR[:][0]
                    if (len(idx_BAR)>1):
                        idx_largest = idx_BAR[-1]
                        rball_BAR = rbin[idx_largest]
                    else:
                        rball_BAR = 0.0

                    #array of idx with DFDM > Dsmallest
                    idx_FDM = np.where(abs(DFDM) > smallestD)
                    idx_FDM = idx_FDM[:][0]
                    if (len(idx_FDM)>1):
                        idx_largest = idx_FDM[-1]
                        rball_FDM = rbin[idx_largest]
                    else:
                        rball_FDM = 0.0

                    #largest rball
                    rball = max(rball_BAR,rball_FDM)
                
                    #consistency check:
                    print('rvir = ', h_chunk['rvir'][i])
                    print('rball/rvir = ', rball/h_chunk['rvir'][i])

                    print('Mvir, cvir = ', h_chunk['Mvir'][i], h_chunk['cvir'][i])
                    if (rball>Lbox/2.0):
                        print('rball = ', rball)
                        print('ERROR: REDUCE RBALL!')
                        exit()

                    #particle ids within rball
                    ipbool = np.array(p_tree.query_ball_point((h_chunk['x'][i],h_chunk['y'][i],h_chunk['z'][i]),rball))
                    #print("Halo centre, surrounding particle number = ", h_chunk['x'][i],h_chunk['y'][i],h_chunk['z'][i], len(ipbool))
                
                    if (len(ipbool) > 0):
                        #calculating radii of FDM particles around halo i
                        rpFDM  = ((p_darkmatter['x'][ipbool]-h_chunk['x'][i])**2.0 +
                            (p_darkmatter['y'][ipbool]-h_chunk['y'][i])**2.0 +
                            (p_darkmatter['z'][ipbool]-h_chunk['z'][i])**2.0)**0.5

                        #calculating radii of BAR particles around halo i
                        rpBAR = ((p_baryons['x'][ipbool]-h_chunk['x'][i])**2.0 +
                            (p_baryons['y'][ipbool]-h_chunk['y'][i])**2.0 +
                            (p_baryons['z'][ipbool]-h_chunk['z'][i])**2.0)**0.5

                        #ids of baryonic particles that are in neighbouring haloes 
                        ipbool_nbrhaloes = ipbool[np.where(rpBAR > h_chunk['rvir'][i])]
                        ipbool_nbrhaloes = ipbool_nbrhaloes[np.where(iphalo[ipbool_nbrhaloes]>0)]
                        ipbool_wo_nbrhaloes = np.setdiff1d(ipbool,ipbool_nbrhaloes)

                        #calculating radii of BAR particles around halo i
                        rpBAR_nbrhaloes = ((p_baryons['x'][ipbool_nbrhaloes]-h_chunk['x'][i])**2.0 +
                                (p_baryons['y'][ipbool_nbrhaloes]-h_chunk['y'][i])**2.0 +
                                (p_baryons['z'][ipbool_nbrhaloes]-h_chunk['z'][i])**2.0)**0.5
                        rpBAR_wo_nbrhaloes = ((p_baryons['x'][ipbool_wo_nbrhaloes]-h_chunk['x'][i])**2.0 +
                                (p_baryons['y'][ipbool_wo_nbrhaloes]-h_chunk['y'][i])**2.0 +
                                (p_baryons['z'][ipbool_wo_nbrhaloes]-h_chunk['z'][i])**2.0)**0.5
                        #print("lengthtiffffffffffffffffff",i,len(rpBAR_nbrhaloes),len(rpBAR_wo_nbrhaloes),len(rpBAR_nbrhaloes)+len(rpBAR_wo_nbrhaloes),len(rpBAR))

                        #if (rball>0.0 and len(rpFDM)):
                        DrpFDM = splev(rpFDM,DFDM_tck,der=0,ext=1)
                        DpFDM['x'][ipbool] += (p_darkmatter['x'][ipbool]-h_chunk['x'][i])*DrpFDM/rpFDM
                        DpFDM['y'][ipbool] += (p_darkmatter['y'][ipbool]-h_chunk['y'][i])*DrpFDM/rpFDM
                        DpFDM['z'][ipbool] += (p_darkmatter['z'][ipbool]-h_chunk['z'][i])*DrpFDM/rpFDM

                        #if (rball>0.0 and len(rpBAR)):
                        #if/else there are neighbouring haloes within rball
                        if(len(rpBAR_nbrhaloes)>0 and len(rpBAR_wo_nbrhaloes)>0):
                            #print("AAAA",splev([np.min(rpBAR_wo_nbrhaloes),np.max(rpBAR_wo_nbrhaloes)],DBAR_tck,der=0,ext=1))
                            DrpBAR_nbrhaloes    = splev(rpBAR_nbrhaloes,DFDM_tck,der=0,ext=1)
                            DrpBAR_wo_nbrhaloes = splev(rpBAR_wo_nbrhaloes,DBAR_tck,der=0,ext=1)
                            #DrpBAR_wo_nbrhaloes = DBAR_tck(rpBAR_wo_nbrhaloes, extrapolate=False)
                            return_dict_DpBAR['x'][ipbool_nbrhaloes] += (p_baryons['x'][ipbool_nbrhaloes]-h_chunk['x'][i])*DrpBAR_nbrhaloes/rpBAR_nbrhaloes
                            return_dict_DpBAR['y'][ipbool_nbrhaloes] += (p_baryons['y'][ipbool_nbrhaloes]-h_chunk['y'][i])*DrpBAR_nbrhaloes/rpBAR_nbrhaloes
                            return_dict_DpBAR['z'][ipbool_nbrhaloes] += (p_baryons['z'][ipbool_nbrhaloes]-h_chunk['z'][i])*DrpBAR_nbrhaloes/rpBAR_nbrhaloes
                            return_dict_DpBAR['x'][ipbool_wo_nbrhaloes] += (p_baryons['x'][ipbool_wo_nbrhaloes]-h_chunk['x'][i])*DrpBAR_wo_nbrhaloes/rpBAR_wo_nbrhaloes
                            return_dict_DpBAR['y'][ipbool_wo_nbrhaloes] += (p_baryons['y'][ipbool_wo_nbrhaloes]-h_chunk['y'][i])*DrpBAR_wo_nbrhaloes/rpBAR_wo_nbrhaloes
                            return_dict_DpBAR['z'][ipbool_wo_nbrhaloes] += (p_baryons['z'][ipbool_wo_nbrhaloes]-h_chunk['z'][i])*DrpBAR_wo_nbrhaloes/rpBAR_wo_nbrhaloes
                        else:
                            DrpBAR = splev(rpBAR,DBAR_tck,der=0,ext=1)
                            #DrpBAR = DBAR_tck(rpBAR, extrapolate=False)
                            return_dict_DpBAR['x'][ipbool] += (p_baryons['x'][ipbool]-h_chunk['x'][i])*DrpBAR/rpBAR
                            return_dict_DpBAR['y'][ipbool] += (p_baryons['y'][ipbool]-h_chunk['y'][i])*DrpBAR/rpBAR
                            return_dict_DpBAR['z'][ipbool] += (p_baryons['z'][ipbool]-h_chunk['z'][i])*DrpBAR/rpBAR

                        if (self.param.code.spher_corr==True):

                            # INERTIA TENSOR FROM AHF (what about rockstar?)
                            #define relative particle position (should be done further up)                                                                                                                                                                                            
                            rel_dm_part_pos_x = p_darkmatter['x'][ipbool]-h_chunk['x'][i]
                            rel_dm_part_pos_y = p_darkmatter['y'][ipbool]-h_chunk['y'][i]
                            rel_dm_part_pos_z = p_darkmatter['z'][ipbool]-h_chunk['z'][i]
                            rel_dm_part_pos = [rel_dm_part_pos_x,rel_dm_part_pos_y,rel_dm_part_pos_z]

                            #rotation amtrix
                            Ea = [h_chunk['Eax'][i],h_chunk['Eay'][i],h_chunk['Eaz'][i]]
                            Eb = [h_chunk['Ebx'][i],h_chunk['Eby'][i],h_chunk['Ebz'][i]]
                            Ec = [h_chunk['Ecx'][i],h_chunk['Ecy'][i],h_chunk['Ecz'][i]]
                            Mrot = np.column_stack((Ea,Eb,Ec))

                            #rotate particles
                            rel_dm_part_pos_rot = np.linalg.inv(Mrot).dot(rel_dm_part_pos)

                            #Displacement correction
                            re_ov_ci = (h_chunk['c_ov_a'][i]*h_chunk['b_ov_a'][i])**(1/3)/h_chunk['c_ov_a'][i]
                            re_ov_bi = (h_chunk['c_ov_a'][i]*h_chunk['b_ov_a'][i])**(1/3)/h_chunk['b_ov_a'][i]
                            re_ov_ai = (h_chunk['c_ov_a'][i]*h_chunk['b_ov_a'][i])**(1/3)

                            #calculate displacement
                            DpFDM_sph_rot_x = np.zeros(len(rel_dm_part_pos_rot[0]))
                            DpFDM_sph_rot_y = np.zeros(len(rel_dm_part_pos_rot[1]))
                            DpFDM_sph_rot_z = np.zeros(len(rel_dm_part_pos_rot[2]))

                            AA = self.param.code.spher_amplitude
                            th = self.param.code.spher_scale
                            nn = self.param.code.spher_powerlaw
                            DpFDM_sph_rot_x = (re_ov_ai - 1) * rel_dm_part_pos_rot[0] * self.sph_corr(rpFDM,h_chunk['rvir'][i],AA,nn,th)
                            DpFDM_sph_rot_y = (re_ov_bi - 1) * rel_dm_part_pos_rot[1] * self.sph_corr(rpFDM,h_chunk['rvir'][i],AA,nn,th)
                            DpFDM_sph_rot_z = (re_ov_ci - 1) * rel_dm_part_pos_rot[2] * self.sph_corr(rpFDM,h_chunk['rvir'][i],AA,nn,th)
                            DpFDM_sph_rot = [DpFDM_sph_rot_x,DpFDM_sph_rot_y,DpFDM_sph_rot_z]
                            
                            #Rotate back
                            DpFDM_sph_rotback = Mrot.dot(DpFDM_sph_rot)

                            DpFDM_sph['x'][ipbool] += DpFDM_sph_rotback[0]
                            DpFDM_sph['y'][ipbool] += DpFDM_sph_rotback[1]
                            DpFDM_sph['z'][ipbool] += DpFDM_sph_rotback[2]

                            
                            Earot = np.linalg.inv(Mrot).dot(Ea)
                            Ebrot = np.linalg.inv(Mrot).dot(Eb)
                            Ecrot = np.linalg.inv(Mrot).dot(Ec)

                        
                        #Add temperature
                        TrpBAR = splev(rpBAR,TGAS_tck,der=0,ext=1)
                        return_dict_DpBAR['temp'][ipbool] += TrpBAR

                        #Add electron pressure
                        PrpBAR = splev(rpBAR,PGAS_tck,der=0,ext=1)
                        return_dict_DpBAR['pres'][ipbool] += PrpBAR

                        
                        #separate baryons into gas and stars
                        
                        #probabilities
                        probHGA = np.round(frac['HGA']*dens['HGA']/(frac['HGA']*dens['HGA'] + frac['IGA']*dens['IGA'] + frac['CGA']*dens['CGA'] + frac['SGA']*dens['SGA']),8)
                        probIGA = np.round(frac['IGA']*dens['IGA']/(frac['HGA']*dens['HGA'] + frac['IGA']*dens['IGA'] + frac['CGA']*dens['CGA'] + frac['SGA']*dens['SGA']),8)
                        probCGA = np.round(frac['CGA']*dens['CGA']/(frac['HGA']*dens['HGA'] + frac['IGA']*dens['IGA'] + frac['CGA']*dens['CGA'] + frac['SGA']*dens['SGA']),8)
                        probSGA = np.round(frac['SGA']*dens['SGA']/(frac['HGA']*dens['HGA'] + frac['IGA']*dens['IGA'] + frac['CGA']*dens['CGA'] + frac['SGA']*dens['SGA']),8)

                        #Only HGA outside of virial radius
                        probHGA[np.where(rbin>=h_chunk['rvir'][i])] = 1.0
                        probIGA[np.where(rbin>=h_chunk['rvir'][i])] = 0.0
                        probCGA[np.where(rbin>=h_chunk['rvir'][i])] = 0.0
                        probSGA[np.where(rbin>=h_chunk['rvir'][i])] = 0.0

                        probHGA_tck = splrep(rbin, probHGA, s=0, k=3)
                        probIGA_tck = splrep(rbin, probIGA, s=0, k=3)
                        probCGA_tck = splrep(rbin, probCGA, s=0, k=3)
                        probSGA_tck = splrep(rbin, probSGA, s=0, k=3)

                        #calculate the probabilities per particle
                        if(len(rpBAR_nbrhaloes)>0 and len(rpBAR_wo_nbrhaloes)>0):
                            rpBAR_wo_nbrhaloes_displ = rpBAR_wo_nbrhaloes + DrpBAR_wo_nbrhaloes
                            pHGA = splev(rpBAR_wo_nbrhaloes_displ, probHGA_tck,der=0,ext=3)
                            pIGA = splev(rpBAR_wo_nbrhaloes_displ, probIGA_tck,der=0,ext=3)
                            pCGA = splev(rpBAR_wo_nbrhaloes_displ, probCGA_tck,der=0,ext=3)
                            pSGA = splev(rpBAR_wo_nbrhaloes_displ, probSGA_tck,der=0,ext=3)
                        else:
                            rpBAR_displ = rpBAR + DrpBAR
                            pHGA = splev(rpBAR_displ, probHGA_tck,der=0,ext=3)
                            pIGA = splev(rpBAR_displ, probIGA_tck,der=0,ext=3)
                            pCGA = splev(rpBAR_displ, probCGA_tck,der=0,ext=3)
                            pSGA = splev(rpBAR_displ, probSGA_tck,der=0,ext=3)
                            
                        #Set ID: 0=gas, 1=star
                        if (self.param.code.satgal==False):
                            
                            #random number between 0 and 1
                            RN = np.random.rand(len(pHGA))
                            if (len(rpBAR_wo_nbrhaloes)):

                                setID = np.zeros_like(RN).astype(int)
                                setID[RN>(pHGA+pIGA)] = 1
                                
                                if(len(rpBAR_nbrhaloes)>0):
                                    DpBAR['id'][ipbool_wo_nbrhaloes] += setID
                                else:
                                    DpBAR['id'][ipbool] += setID

                        #Set ID: 0=gas, 1=star
                        elif (self.param.code.satgal==True):

                            #subhaloes
                            shi = h_chunk[h_chunk['IDhost']==i]
                            Nsh = len(shi['Mvir'])

                            if (Nsh<1):
                                #If there is no satellite, sga-stars are distributed as radial profile
                                print("No satellites detected. SGA stars (f_sga) distributed as radial profile.")
                                RN = np.random.rand(len(pHGA))
                                setID = np.zeros_like(RN).astype(int)
                                setID[RN<(pCGA+pSGA)] = 1
                                if(len(rpBAR_nbrhaloes)>0 and len(rpBAR_wo_nbrhaloes)>0):
                                    DpBAR['id'][ipbool_wo_nbrhaloes] += setID
                                    #DpBAR['id'][ipbool] += setID
                                else:
                                    DpBAR['id'][ipbool] += setID
                                    
                            else:
                                #If there is at least one satellite, the sga-stars are dumped around its/their centre
                                print("Satellites detected. SGA stars (f_sga) added to satellite positions.")
                                print("N_sat = ", Nsh)

                                #Check if total stellite stellar mass from (fstar-fcga)*Mvir_host agrees with the sum over satellites (each contributing with fcga_sat*Mvir_sat)
                                #particle_mass = self.param.cosmo.Ob*rhoc_of_z(self.param)*Lbox**3/(p_header['Npart']/2)
                                particle_mass = self.param.cosmo.Ob*RHOC*Lbox**3/(p_header['Npart']/2)
                                fstar_halo_tot = fstar_fct(h_chunk['Mvir'][i],self.param,self.param.baryon.eta)
                                fstar_halo_cga = fstar_fct(h_chunk['Mvir'][i],self.param,self.param.baryon.eta+self.param.baryon.deta)
                                Mstar_halo_sat = (fstar_halo_tot-fstar_halo_cga)*h_chunk['Mvir'][i] #expected total satellite stellar mass (fstar-fcga)*Mvir_host
                                Mstar_halo_sat_from_sh_cga = np.sum(fstar_fct(shi['Mvir'], self.param, self.param.baryon.eta + self.param.baryon.deta)*shi['Mvir']) #expected total sat stellar mass from fcga_sat*Mvir_sat
                                Sat_boost = Mstar_halo_sat/Mstar_halo_sat_from_sh_cga
                                print("Sat_boost = ", Sat_boost)

                                #Star particles
                                RN = np.random.rand(len(pHGA))
                                setID = np.zeros_like(RN).astype(int)

                                #Total number of satellite stars is smaller than expected from satellite numbers
                                #We reduce the numbers to obtain the total f_sga (multiplying with Sat_boost)
                                if (Sat_boost<1):
                                    setID[RN<pCGA] = 1
                                    if(len(rpBAR_nbrhaloes)>0 and len(rpBAR_wo_nbrhaloes)>0):
                                        DpBAR['id'][ipbool_wo_nbrhaloes] += setID
                                    else:
                                        DpBAR['id'][ipbool] += setID

                                    #loop over satellites
                                    for j in range(Nsh):
                                        #reduce satellite stellar mass to make it agree with fsat = (fstar-fcga)*Mvir_host
                                        fstar_sh_cga = Sat_boost*fstar_fct(shi['Mvir'][j], self.param, self.param.baryon.eta + self.param.baryon.deta)
                                        Mstar_sh_cga = fstar_sh_cga*shi['Mvir'][j]
                                        Npart = np.round(Mstar_sh_cga/particle_mass)
                                        if (Npart == 0):
                                            Npart = 1    
                                        print("Npart_per_sat = ", Npart)
                                    
                                        dist_, ip_sh_stars = p_tree.query((shi['x'][j],shi['y'][j],shi['z'][j]), Npart, distance_upper_bound=shi['rvir'][j])

                                        #Remove subhaloes that are outside of boundary region (no particles witin rvir)
                                        ip_sh_stars_clean = np.delete(ip_sh_stars, np.where(ip_sh_stars == len(DpBAR['id'])))
                                        DpBAR['id'][ip_sh_stars_clean] = 1
                                        
                                #Total number of satellite stars is larger than expected from satellite numbers
                                #We split f_sga into satellites and halo stars
                                else:

                                    #loop over satellites and add stars
                                    for j in range(Nsh):
                                        
                                        fstar_sh_cga = fstar_fct(shi['Mvir'][j], self.param, self.param.baryon.eta + self.param.baryon.deta)
                                        Mstar_sh_cga = fstar_sh_cga*shi['Mvir'][j]
                                        Npart = np.round(Mstar_sh_cga/particle_mass)
                                        if (Npart == 0):
                                            Npart = 1
                                        print("Npart_per_sat = ", Npart)
                                        
                                        dist_, ip_sh_stars = p_tree.query((shi['x'][j],shi['y'][j],shi['z'][j]), Npart, distance_upper_bound=shi['rvir'][j])

                                        #Remove subhaloes that are outside of boundary region (no particles witin rvir)
                                        ip_sh_stars_clean = np.delete(ip_sh_stars, np.where(ip_sh_stars == len(DpBAR['id'])))
                                        DpBAR['id'][ip_sh_stars_clean] = 1

                                    #prob of halo stars (excess stars not in satellites)  
                                    p_halostars = pSGA - pSGA/Sat_boost

                                    #add central and halo stars SOME IDs MAY OVERLAP WITH SATELLITE GALAXIES! This is ignored
                                    setID[RN<(pCGA+p_halostars)] = 1
                                    if(len(rpBAR_nbrhaloes)>0 and len(rpBAR_wo_nbrhaloes)>0):
                                        DpBAR['id'][ipbool_wo_nbrhaloes] += setID
                                    else:
                                        DpBAR['id'][ipbool] += setID
            

            #Displace particles and separarte BAR into HGA, CGA, SGA
            p_baryons['x'] += DpBAR['x']
            p_baryons['y'] += DpBAR['y']
            p_baryons['z'] += DpBAR['z']

            p_darkmatter['x'] += DpFDM['x'] + DpFDM_sph['x']
            p_darkmatter['y'] += DpFDM['y'] + DpFDM_sph['y']
            p_darkmatter['z'] += DpFDM['z'] + DpFDM_sph['z']
            #p_darkmatter['x'] += DpFDM['x']
            #p_darkmatter['y'] += DpFDM['y']
            #p_darkmatter['z'] += DpFDM['z']
            
            #Separate baryons into gas and star
            p_gas_temporary_chunk  = p_baryons[DpBAR['id']<0.5]
            p_dm_chunk   = p_darkmatter
            p_star_chunk = p_baryons[DpBAR['id']>0.5]

            #copy to new structured array and add temperature/pressure
            #p_gas_type = np.dtype(p_gas_temporary_chunk.dtype.descr + [('temp', '>f4')])
            p_gas_type = np.dtype(p_gas_temporary_chunk.dtype.descr + [('temp', '>f4')] + [('pres', '>f4')])
            p_gas_chunk = np.zeros(p_gas_temporary_chunk.shape, dtype=p_gas_type)
            p_gas_chunk['x'] = p_gas_temporary_chunk['x']
            p_gas_chunk['y'] = p_gas_temporary_chunk['y']
            p_gas_chunk['z'] = p_gas_temporary_chunk['z']
            p_gas_chunk['vx'] = p_gas_temporary_chunk['x']
            p_gas_chunk['vy'] = p_gas_temporary_chunk['y']
            p_gas_chunk['vz'] = p_gas_temporary_chunk['z']
            p_gas_chunk['mass'] = p_gas_temporary_chunk['mass']
            p_gas_chunk['phi'] = p_gas_temporary_chunk['phi']
        
            p_gas_chunk['temp'] = DpBAR['temp'][DpBAR['id']<0.5]
            p_gas_chunk['pres'] = DpBAR['pres'][DpBAR['id']<0.5]

            del p_gas_temporary_chunk 
            del p_baryons
            del p_darkmatter
                
        elif (self.param.code.multicomp==False):

            #Copy into p_temp
            Dp_dt  = np.dtype([("x",'>f'),("y",'>f'),("z",'>f')])
            Dp     = np.zeros(len(p_chunk),dtype=Dp_dt)
            Dp_sph = np.zeros(len(p_chunk),dtype=Dp_dt)
            
            #Build tree
            print('building tree..')
            p_tree = spatial.cKDTree(list(zip(p_chunk['x'],p_chunk['y'],p_chunk['z'])), leafsize=100)
            print('...done!')

            #Loop over haloes, calculate displacement, and displace partricles
            for i in range(len(h_chunk['Mvir'])):

                #select host haloes
                if (h_chunk['IDhost'][i] < 0.0):
                    
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

                    #DDMB = displ(rbin,mass['DMO'],mass['DMB'])
                    DDMB = self.displ(rbin,mass['NFW'] + mass['BG'], mass['CDM'] + mass['BG'])

                    DDMB_tck = splrep(rbin, DDMB,s=0,k=3)

                    #define minimum displacement
                    smallestD = self.param.code.disp_trunc #Mpc/h

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

                    #particle ids within rball
                    ipbool = p_tree.query_ball_point((h_chunk['x'][i],h_chunk['y'][i],h_chunk['z'][i]),rball)

                    #update displacement
                    rpDMB  = ((p_chunk['x'][ipbool]-h_chunk['x'][i])**2.0 + 
                        (p_chunk['y'][ipbool]-h_chunk['y'][i])**2.0 + 
                        (p_chunk['z'][ipbool]-h_chunk['z'][i])**2.0)**0.5

                    if (rball>0.0 and len(rpDMB)):
                        DrpDMB = splev(rpDMB,DDMB_tck,der=0,ext=1)
                        Dp['x'][ipbool] += (p_chunk['x'][ipbool]-h_chunk['x'][i])*DrpDMB/rpDMB
                        Dp['y'][ipbool] += (p_chunk['y'][ipbool]-h_chunk['y'][i])*DrpDMB/rpDMB
                        Dp['z'][ipbool] += (p_chunk['z'][ipbool]-h_chunk['z'][i])*DrpDMB/rpDMB

            #Displace 
            p_chunk['x'] += Dp['x'] + Dp_sph['x']
            p_chunk['y'] += Dp['y'] + Dp_sph['y']
            p_chunk['z'] += Dp['z'] + Dp_sph['z']

            #only dm, rest zero
            p_dm_chunk   = p_chunk

            p_gas_type   = np.dtype([('mass','>f4'),('x', '>f4'),('y', '>f4'),('z', '>f4'),('vx', '>f4'),('vy', '>f4'),\
                                ('vz', '>f4'),('rho','>f4'),('temp','>f4'),('hsmooth','>f4'),('metals','>f4'),('phi','>f4')])
            p_star_type  = np.dtype([('mass','>f4'),('x', '>f4'),('y', '>f4'),('z', '>f4'),('vx', '>f4'),('vy', '>f4'),\
                                    ('vz', '>f4'),('metals','>f4'),('tform','>f4'),('eps','>f4'),('phi','>f4')])
            p_gas_chunk  = np.zeros(0,dtype=p_gas_type)
            p_star_chunk = np.zeros(0,dtype=p_star_type)

            del p_chunk
            
        else:
            print("param.code.multicomp must be either True or False. Abort")
            exit()


        return p_gas_chunk, p_dm_chunk, p_star_chunk


def worker_method(args):
    """
    wrapper function of the loop_halo_chunks
    to make it the top-level function, important for parallelization
    """
    obj, process, idx, task, args_for_loop_halo_chunks = args
    return obj.loop_halo_chunks(process, idx, task, args_for_loop_halo_chunks)

def wrapper(args):
    return loop_halo_chunks_worker(*args)

class ShellDisplacer:
    """
    Class to handle shell displacement.
    """
    def __init__(self,param):
        self.param = param

    def perform_shell_displacement(self):
        """
        Reading in pixel and halo files
        dispalcing particles, writing healpix file
        """
        # create MPI pool
        with schwimmbad.MPIPool() as pool:
        # with schwimmbad.SerialPool() as pool:
            
            if not pool.is_master():
                pool.wait()
                sys.exit(0)

            try:
                LOGGER.info(f"Performing shell baryonification for {self.param.shell.max_shell - self.param.shell.min_shell} shells ({self.param.shell.min_shell}-{self.param.shell.max_shell}).\n")

                io_shell = IO_shell(self.param)
                h_list, thickness_list, redshift_list, shell_cov_list = io_shell.read_halo_lc_file()
                shell_id, map_list = io_shell.read_healpix_file()

                p_list = self.perform_get_particle(map_list, h_list, shell_cov_list, pool)
                del map_list

                if (self.param.code.multicomp == True):
                    gas_shell, dm_shell, star_shell = self.displace_shell(shell_id, p_list, redshift_list, h_list, thickness_list, shell_cov_list, pool)
                    io_shell.write_shell_file_multicomp(gas_shell,dm_shell,star_shell)
                elif (self.param.code.multicomp == False):
                    dmb_shell = self.displace_shell(shell_id, p_list, redshift_list, h_list, thickness_list, shell_cov_list, pool)
                    io_shell.write_shell_file_singlecomp(dmb_shell)
                else:
                    LOGGER.critical(f"param.code.multicomp must be either True or False. Abort")
                    exit()
            finally:
                # safety net: removes every temp file tagged with this job's SLURM_JOB_ID,
                # even if an exception above skipped the normal per-phase cleanup
                LOGGER.info(f"Cleaning up any remaining temporary files for this job...")
                cleanup_job_tmp_files(self.param)

        return 0

    def perform_get_particle(self, map_list, h_list, shell_cov_list, pool):
        """
        Convert map to particles, parallelized.
        """
        num_shells = int(self.param.shell.max_shell - self.param.shell.min_shell)
        if num_shells != len(map_list):
            raise ValueError(f"Mismatch: {len(map_list)} shells but num_shells={num_shells}")

        tasks = [(i, map_list[i], h_list[i], self.param, shell_cov_list[i]) for i in range(num_shells)]
        
        output_dir = self.param.files.tmp_files
        results = []
        for i_proc in range(num_shells):
            LOGGER.info(f"Sampling particles for shell {i_proc+1}/{num_shells}")
            t1 = time()
            # if subsampled particles for a given DM shell are stored, read the file instead of rerunning
            filename_pixleparticle = f"{output_dir}/pixel_particles__{self.param.files.shellfile_in.replace('/','_').replace('.','_')}___shell_{self.param.shell.min_shell+i_proc}.pkl"
            if os.path.exists(filename_pixleparticle):
                LOGGER.info(f"......Found existing pixel particles file for shell {self.param.shell.min_shell+i_proc}, loading file {filename_pixleparticle}")
                with open(filename_pixleparticle, "rb") as pkl_file:
                    result = pkl.load(pkl_file)
            else:
                LOGGER.info(f"......No existing pixel particles file found for shell {self.param.shell.min_shell+i_proc}, entering particle subsampling")
                result = particle_worker(tasks[i_proc],pool)
            
                # store subsampled particles to disk (if output_pixelparticle_file=TRUE) 
                if self.param.files.output_pixelparticle_file:
                    with open(filename_pixleparticle, "wb") as pkl_file:
                        pkl.dump(result, pkl_file)
            
            results.append(result)
            t2 = time()
            LOGGER.info(f"Sampling particles for shell {i_proc+1}/{num_shells} done ✅. Ellapsed time: {t2 - t1:.3f} seconds\n")

        particle_shell = {i: p for i, p in results}
        return [particle_shell[i] for i in sorted(particle_shell.keys())]

    def displace_shell(self, shell_id, p_list, redshift_list, h_list, thickness_list, shell_cov_list, pool, test=False):
        '''
        displace particles on the shell with the halo file
        '''
        LOGGER.info(f"Displacing shells...")
        num_shells = int(self.param.shell.max_shell - self.param.shell.min_shell)

        tasks = list(zip(shell_id, h_list, thickness_list, p_list, redshift_list, shell_cov_list, np.repeat(self.param,num_shells)))

        if (self.param.code.multicomp == True):
            
            gasdata = {}
            dmdata = {}
            stardata = {}

            for i_proc in range(num_shells):
                LOGGER.info(f"......Shell {i_proc+1}/{num_shells}")
                t1 = time()
                result = self.loop_halos(tasks[i_proc], pool)
                i_shell = result[0]
                gasdata[i_shell] = result[1]
                dmdata[i_shell] = result[2]
                stardata[i_shell] = result[3]
                t2 = time()
                LOGGER.info(f"......Shell {i_proc+1}/{num_shells} done. Ellapsed time: {t2 - t1:.3f} seconds")
            LOGGER.info(f"Displacing shells done ✅\n")
            return gasdata, dmdata, stardata
        
        elif (self.param.code.multicomp == False):

            dmbdata = {}

            for i_proc in range(num_shells):
                LOGGER.info(f"......Shell {i_proc+1}/{num_shells}")
                t1 = time()
                result = self.loop_halos(tasks[i_proc], pool)
                i_shell = result[0]
                dmbdata[i_shell] = result[1]
                t2 = time()
                LOGGER.info(f"......Shell {i_proc+1}/{num_shells} done. Ellapsed time: {t2 - t1:.3f} seconds")
            LOGGER.info(f"Displacing shells done ✅\n")
            return dmbdata
        
        else:
            LOGGER.critical(f"param.code.multicomp must be either True or False. Abort")
            exit()

    def loop_halos(self, task, pool):
        '''
        loop over halos and displace particles
        '''
        cosmo_calculator = CosmoCalculator(self.param)

        shell_id, h, thickness, p, redshift, shell_cov, param = task

        param.cosmo.z = redshift

        vc_r, vc_m, vc_var, vc_bias, vc_corr = cosmo_calculator.compute_cosmology()
        var_tck  = splrep(vc_m, vc_var, s=0)
        bias_tck = splrep(vc_m, vc_bias, s=0)
        corr_tck = splrep(vc_r, vc_corr, s=0)

        #build tree for dm and baryons, separate particles
        #id of all halo particle (ihalo=0 means field particles)
        
        t1 = time()
        # p_tree = spatial.cKDTree(list(zip(p['x'],p['y'],p['z'])), leafsize=100)
        pos = np.empty((len(p), 3), dtype=np.float32)
        pos[:, 0] = p['x']
        pos[:, 1] = p['y']
        pos[:, 2] = p['z']
        LOGGER.info(f"......KD-tree building for {pos.shape} array...")
        p_tree = spatial.cKDTree(pos[:,:], leafsize=100)
        del pos
        t2 = time()
        LOGGER.info(f"......KD-tree building done✅. Ellapsed time: {t2 - t1:.3f} seconds\n")

        
        """
        parallelization
        """
        gl_start = time()

        nproc = min(param.shell.N_cpu - 1, len(h["Mvir"]))
        if nproc == 0:
            # the above condition is valid for parallel jobs
            nproc += 1
        
        LOGGER.info(f"......Looping over halos {len(h)} halos, using {nproc} CPUs.")

        idx = np.arange(len(h["Mvir"]))

        output_dir = self.param.files.tmp_files
        run_tag = make_run_tag(shell_id)
        # store px,py,pz to be used by MPI tasks
        fields = ['x','y','z']
        new_dtype = np.dtype([(f, p[f].dtype) for f in fields])
        # Allocate new structured array
        p_coords = np.zeros(len(p), dtype=new_dtype)
        # Copy the requested fields
        for f in fields:
            p_coords[f] = p[f]
        p_fn = os.path.join(output_dir,f"p_{run_tag}.npy")
        h_fn = os.path.join(output_dir,f"h_{run_tag}.npy")
        np.save(p_fn, p_coords)  # save p to a file to avoid pickling issues
        np.save(h_fn, h)  # save p to a file to avoid pickling issues
        # save the tree's large internal arrays as separate mmap-able files so workers
        # share the same physical pages instead of each unpickling a private full copy
        p_tree_files = save_cKDTree_shared(p_tree, output_dir, prefix=f"p_tree_{run_tag}")

        # prepare argument list
        args_for_loop_halo_chunks = [shell_cov, var_tck, bias_tck, corr_tck]
        task_mpi = (shell_id, thickness, redshift, param)
        iterable_args = [
            (i_cpu, idx[i_cpu::nproc], task_mpi, args_for_loop_halo_chunks, output_dir)
            for i_cpu in range(nproc)
        ]

        # ---- parallel execution ----
        results = list(pool.map(wrapper, iterable_args))
        # ----------------------------


        gl_end = time()
        LOGGER.info(f"......Looping over halos done. Ellapsed time: {gl_end - gl_start:.3f}")

        n_host_total = int(np.count_nonzero(h["IDhost"] < 0))
        n_displaced_total = sum(r[2] for r in results)
        LOGGER.info(f"......{n_displaced_total}/{n_host_total} host halos were displaced (had particles within rball and contributed to the displacement maps).")
        if param.code.multicomp:
            n_bar_superpixel_total = sum(r[3] for r in results)
            LOGGER.info(f"......{n_bar_superpixel_total}/{n_host_total} host halos had max|DBAR| larger than one output pixel (the rest displace baryons by less than a pixel, so the gas/dm difference map can't show them).")

        LOGGER.info(f"......Cleaning up temporary files (tree, particles, h)")
        os.remove(p_fn)
        os.remove(h_fn)
        for fn in p_tree_files:
            os.remove(fn)

        LOGGER.info(f"......Splitting particles to dm and baryons...")

        # results = [(DpBAR_part, DpFDM_part, n_displaced), ...]
        bar_filenames = [r[0] for r in results]
        dm_filenames = [r[1] for r in results]

        p_darkmatter = p.copy() # becomes dmb for multicomp = False
        if param.code.multicomp:
            p_baryons = p.copy()
        else:
            # do not treat baryons, darkmatter becomes dmb
            p_baryons = None
        n_p = len(p)
        del p


        t_split = time()
        LOGGER.info(f"......Splitting particles to dm and baryons done. Ellapsed time: {t_split - gl_end:.3f}s")
        
        LOGGER.info(f"......Summing displacements...")
        
        if (self.param.code.multicomp == True):
            DpBAR = sum_structured_arrays_from_files_multicomp(bar_filenames, n_p)
            DpFDM = sum_structured_arrays_from_files_multicomp(dm_filenames, n_p)
            
            t_sum = time()
            LOGGER.info(f"......Summing displacements done. Ellapsed time: {t_sum - t_split:.3f}s")

            gc.collect()

            LOGGER.info(f"......Applying displacements around halos...")
            t = time()
            
            #Displace DM particles
            DpFDM_c = arcdisplace(np.column_stack((DpFDM['x'], DpFDM['y'], DpFDM['z'])),
                                np.column_stack((p_darkmatter['x'], p_darkmatter['y'], p_darkmatter['z'])),shell_cov,param)
            p_darkmatter['x'] += DpFDM_c[:,0]
            p_darkmatter['y'] += DpFDM_c[:,1]
            p_darkmatter['z'] += DpFDM_c[:,2]
            del DpFDM_c

            #Displace particles and separarte BAR into HGA, CGA, SGA
            DpBAR_c = arcdisplace(np.column_stack((DpBAR['x'], DpBAR['y'], DpBAR['z'])),
                                np.column_stack((p_baryons['x'], p_baryons['y'], p_baryons['z'])),shell_cov,param)
            p_baryons['x'] += DpBAR_c[:,0]
            p_baryons['y'] += DpBAR_c[:,1]
            p_baryons['z'] += DpBAR_c[:,2]
            del  DpBAR_c
            
            LOGGER.info(f"......Applying displacements around halos done. Ellapsed time: {time()-t:.3f} seconds")
            t = time()
            #convert position to healpix index and store the data
            LOGGER.info(f"......Converting particles to healpix maps...")
            shell_gas = get_healpix_map(p_baryons,param, star_fraction=DpBAR['id'], pool = pool, shell_id=shell_id)
            shell_dm = get_healpix_map(p_darkmatter,param, star_fraction=None, pool = pool, shell_id=shell_id)
            shell_star = get_healpix_map(p_baryons,param, star_fraction=1-DpBAR['id'], pool = pool, shell_id=shell_id)
            LOGGER.info(f"......Converting particles to healpix maps done. Ellapsed time: {time()-t:.3f} seconds")
            return shell_id, shell_gas, shell_dm, shell_star
        
        elif (self.param.code.multicomp == False):
            Dp = sum_structured_arrays_from_files_singlecomp(dm_filenames, n_p)
            
            t_sum = time()
            LOGGER.info(f"......Summing displacements done. Ellapsed time: {t_sum - t_split:.3f}s")

            gc.collect()

            LOGGER.info(f"......Applying displacements around halos...")
            t = time()
            
            #Displace DM particles
            Dp_c = arcdisplace(np.column_stack((Dp['x'], Dp['y'], Dp['z'])),
                                np.column_stack((p_darkmatter['x'], p_darkmatter['y'], p_darkmatter['z'])),shell_cov,param)
            p_darkmatter['x'] += Dp_c[:,0]
            p_darkmatter['y'] += Dp_c[:,1]
            p_darkmatter['z'] += Dp_c[:,2]
            del Dp_c

            LOGGER.info(f"......Applying displacements around halos done. Ellapsed time: {time()-t:.3f} seconds")
            t = time()
            #convert position to healpix index and store the data
            LOGGER.info(f"......Converting particles to healpix maps...")
            shell_dmb = get_healpix_map(p_darkmatter,param, star_fraction=None, pool = pool, shell_id=shell_id)
            LOGGER.info(f"......Converting particles to healpix maps done. Ellapsed time: {time()-t:.3f} seconds")
            return shell_id, shell_dmb
        
        else:
            LOGGER.critical(f"param.code.multicomp must be either True or False. Abort")
            exit()
        
            
