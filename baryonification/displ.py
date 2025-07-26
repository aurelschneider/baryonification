"""
CALCULATE DISPLACEMENT FUNCTION FOR A GRID OF M AND C
PRINT INFORMATION INTO TEMPOßRARY FILE

"""

import numpy as np
from scipy import spatial
from scipy.interpolate import splrep,splev
from numpy.lib.recfunctions import append_fields

import schwimmbad

from .cosmo import CosmoCalculator
from .constants import *
from .profiles import Profiles
from .io import *

#from memory_profiler import profile

"""
DISPLACEMENT FUNCTION
"""

def displ(rbin,MINITIAL,MFINAL):

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



def displace(param):

    """
    Reading in N-body and halo files, defining chunks, 
    looping over haloes in single or multi-processor mode,
    dispalcing particles, combining chunks, 
    writing N-body file
    """

    #Read in N-body particle file and build chunks
    p_header, p_list = read_nbody_file(param)

    #Read in halo file, build chunks and buffer
    h_list = read_halo_file(param)
    
    #split work on cpus and perform displacement
    N_chunk = param.sim.N_chunk
    N_cpu   = int(N_chunk**3)
    print('N_cpu = ',N_cpu)

    #Modify header for multicomp
    if (param.code.multicomp==True):
        p_header['Ngas'] = p_header['Npart']
        p_header['Npart'] = int(2*p_header['Npart'])

    if (N_cpu == 1):

        displ_p = displace_chunk(p_header,p_list[0],h_list[0],param)
        
        p_gas_displ  = [displ_p[0]]
        p_dm_displ   = [displ_p[1]]
        p_star_displ = [displ_p[2]]
        
    elif (N_cpu > 1):

        pool = schwimmbad.choose_pool(mpi=False, processes=N_cpu)
        tasks= list(zip(p_list,h_list,np.repeat(p_header,N_cpu),np.repeat(param,N_cpu)))

        displ_p = np.array(pool.map(worker, tasks))
        
        p_gas_displ  = displ_p[:,0]
        p_dm_displ   = displ_p[:,1]
        p_star_displ = displ_p[:,2]

        pool.close()

    p_gas  = np.concatenate(p_gas_displ)
    p_dm   = np.concatenate(p_dm_displ)
    p_star = np.concatenate(p_star_displ)
    
    #correct header
    p_header['Ngas']  = len(p_gas)
    p_header['Nstar'] = len(p_star)

    #combine chunks and write output
    write_nbody_file(p_header,p_gas,p_dm,p_star,param)

    return


def worker(task):
    """
    Worker for multi-processing
    """
    #import os
    #import sys
    #pid = os.getpid()
    #log_filename = f"core_{pid}_log.txt"
    #sys.stdout = open(log_filename, "a")
        
    p_chunk, h_chunk, p_header, param = task
    p_gas_chunk, p_dm_chunk, p_star_chunk = displace_chunk(p_header,p_chunk,h_chunk,param)
    return np.array([p_gas_chunk, p_dm_chunk, p_star_chunk],dtype=object)


#def worker(task):
#
#    """
#    Worker for multi-processing
#    """
#    try:
#        import os
#        import sys
#        pid = os.getpid()  # Get the process ID of the current worker
#        log_filename = f"core_{pid}_log.txt"
#        sys.stdout = open(log_filename, "a")
#    
#        p_chunk, h_chunk, p_header, param = task
#
#        p_gas_chunk, p_dm_chunk, p_star_chunk = displace_chunk(p_header,p_chunk,h_chunk,param)
#    
#        return np.array([p_gas_chunk, p_dm_chunk, p_star_chunk],dtype=object)
#
#    except Exception as e:
#        print(f"Error in worker with task: {task}")
#        #return None
#        exit()


def particle_separation(p_chunk,param):

    """
    Separate DMO particles into DM and BARYONS 
    """
    Ob = param.cosmo.Ob
    Om = param.cosmo.Om

    p_darkmatter = p_chunk.copy()
    p_baryons = p_chunk.copy()

    f_baryons = Ob/Om
    p_darkmatter['mass'] = ((1-f_baryons)*p_darkmatter['mass']).astype(np.float32)
    p_baryons['mass'] = (f_baryons*p_baryons['mass']).astype(np.float32)

    return p_darkmatter, p_baryons


def rotation_matrix(vec, axis='z-axis'):
    """
    Find rotation matrix for coordinate transformation that puts vec
    along axis
    """
    a = (vec / np.linalg.norm(vec)).reshape(3)

    if (axis=='z-axis'):
        unitvec = np.array([0,0,1])
    elif(axis=='y-axis'):
        unitvec = np.array([0,1,0])
    elif(axis=='x-axis'):
        unitvec = np.array([1,0,0])
    else:
        print("axis needs to be either x,y,or z. Exit.")
        exit()

    v = np.cross(a, unitvec)
    c = np.dot(a, unitvec)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def sph_corr(rbin,rvir,A,n,th):
    """
    Amplitude and transition of the sphericity correction.
    sph_corr = A for r<<th*rvir and 0 for r>>th*rvir
    Sphericity correction (inspired by Fig3 of https://arxiv.org/pdf/2109.00012.pdf)
    """
    #return 1/((1 + A*(th*rvir/rbin)**n) / (1+((th*rvir/rbin)**n)))
    return A*(th*rvir/rbin)**n / (1+(th*rvir/rbin)**n)


def displace_chunk(p_header,p_chunk,h_chunk,param):

    """
    Reading in N-body and halo files, looping over haloes, calculating
    displacements, and dispalcing particles.
    Combines functions displ_file() and displace_from_displ_file()
    """

    #relevant parameters
    Lbox = param.sim.Lbox
    
    #calculate cosmo for 2-halo term
    cosmo = CosmoCalculator(param)
    vc_r, vc_m, vc_var, vc_bias, vc_corr = cosmo.compute_cosmology()

    
    #Read cosmic variance/nu/correlation and interpolate
    #cosmofile = param.files.cosmofct
    #try:
    #    vc_r, vc_m, vc_var, vc_bias, vc_corr = np.loadtxt(cosmofile, usecols=(0,1,2,3,4), unpack=True)
    #    var_tck  = splrep(vc_m, vc_var, s=0)
    #    bias_tck = splrep(vc_m, vc_bias, s=0)
    #    corr_tck = splrep(vc_r, vc_corr, s=0)
    #except IOError:
    #    print('IOERROR: Cosmofct file does not exist!')
    #    print('Define par.files.cosmofct = "/path/to/file"')
    #    print('Run: cosmo = Cosmology(par) and cosmo.compute_cosmology() to create file')
    #    exit()

    if (param.code.multicomp==True):

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
        p_darkmatter, p_baryons = particle_separation(p_chunk,param)
        del p_chunk
        
        print("Nhalo_chunk = ", len(h_chunk['Mvir']))

        #Loop over haloes, calculate displacement, and displace partricles
        for i in range(len(h_chunk['Mvir'])):

            #select host haloes (subhaloes >= 1)
            if (h_chunk['IDhost'][i] < 0):
                
                print('start: ', i)

                #range where we consider displacement
                rmax = param.code.rmax
                rmin = (0.001*h_chunk['rvir'][i] if 0.001*h_chunk['rvir'][i]>param.code.rmin else param.code.rmin)
                rmax = (20.0*h_chunk['rvir'][i] if 20.0*h_chunk['rvir'][i]<param.code.rmax else param.code.rmax)
                rbin = np.logspace(np.log10(rmin),np.log10(rmax),100,base=10)

                #initialse profiles
                profiles = Profiles(rbin, h_chunk['Mvir'][i], h_chunk['cvir'][i], param)

                #calculate profiles for displacement
                frac, dens, mass, pres, temp = profiles.calc_profiles()

                #baryon displacement
                DBAR = displ(rbin, (1-frac['CDM'])*(mass['NFW'] + mass['BG']), mass['HGA'] + mass['IGA'] + mass['CGA'] + mass['SGA'] + (1-frac['CDM'])*mass['BG'])
                DBAR_tck = splrep(rbin, DBAR, s=0, k=3)
                #DBAR_tck = make_interp_spline(rbin, DBAR, k=1)
                
                #collisionless (final dark) matter displacement 
                DFDM = displ(rbin, frac['CDM']*(mass['NFW'] + mass['BG']), mass['CDM'] + frac['CDM']*mass['BG'])
                DFDM_tck = splrep(rbin, DFDM,s=0,k=3)

                #gas temperature
                TGAS = temp['electron']
                TGAS_tck = splrep(rbin,TGAS,s=0,k=1)

                #gas pressure
                PGAS = pres['electron']
                PGAS_tck = splrep(rbin,PGAS,s=0,k=1)
                
                #define minimum displacement
                smallestD = param.code.disp_trunc #Mpc/h

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
                print('rvir = ',h_chunk['rvir'][i])
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
                        DpBAR['x'][ipbool_nbrhaloes] += (p_baryons['x'][ipbool_nbrhaloes]-h_chunk['x'][i])*DrpBAR_nbrhaloes/rpBAR_nbrhaloes
                        DpBAR['y'][ipbool_nbrhaloes] += (p_baryons['y'][ipbool_nbrhaloes]-h_chunk['y'][i])*DrpBAR_nbrhaloes/rpBAR_nbrhaloes
                        DpBAR['z'][ipbool_nbrhaloes] += (p_baryons['z'][ipbool_nbrhaloes]-h_chunk['z'][i])*DrpBAR_nbrhaloes/rpBAR_nbrhaloes
                        DpBAR['x'][ipbool_wo_nbrhaloes] += (p_baryons['x'][ipbool_wo_nbrhaloes]-h_chunk['x'][i])*DrpBAR_wo_nbrhaloes/rpBAR_wo_nbrhaloes
                        DpBAR['y'][ipbool_wo_nbrhaloes] += (p_baryons['y'][ipbool_wo_nbrhaloes]-h_chunk['y'][i])*DrpBAR_wo_nbrhaloes/rpBAR_wo_nbrhaloes
                        DpBAR['z'][ipbool_wo_nbrhaloes] += (p_baryons['z'][ipbool_wo_nbrhaloes]-h_chunk['z'][i])*DrpBAR_wo_nbrhaloes/rpBAR_wo_nbrhaloes
                    else:
                        DrpBAR = splev(rpBAR,DBAR_tck,der=0,ext=1)
                        #DrpBAR = DBAR_tck(rpBAR, extrapolate=False)
                        DpBAR['x'][ipbool] += (p_baryons['x'][ipbool]-h_chunk['x'][i])*DrpBAR/rpBAR
                        DpBAR['y'][ipbool] += (p_baryons['y'][ipbool]-h_chunk['y'][i])*DrpBAR/rpBAR
                        DpBAR['z'][ipbool] += (p_baryons['z'][ipbool]-h_chunk['z'][i])*DrpBAR/rpBAR

                    if (param.code.spher_corr==True):

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

                        '''
                        #CALCULATE INERTIA TENSOR (does not work at boundaries of chuncks, provides same results than ahf if inertia tensor is evaluated at rvir)
                        if (h_chunk['x'][i] > 0 and h_chunk['x'][i] < Lbox and h_chunk['y'][i] > 0 and h_chunk['y'][i] < Lbox and h_chunk['z'][i] > 0 and h_chunk['z'][i] < Lbox):
                            #define relative particle position
                            rel_dm_part_pos_x = p_darkmatter['x'][ipbool]-h_chunk['x'][i]
                            rel_dm_part_pos_y = p_darkmatter['y'][ipbool]-h_chunk['y'][i]
                            rel_dm_part_pos_z = p_darkmatter['z'][ipbool]-h_chunk['z'][i]
                            rel_dm_part_pos = [rel_dm_part_pos_x,rel_dm_part_pos_y,rel_dm_part_pos_z]

                            relx_red = rel_dm_part_pos_x[rpFDM<1.0*h_chunk['rvir'][i]]
                            rely_red = rel_dm_part_pos_y[rpFDM<1.0*h_chunk['rvir'][i]]
                            relz_red = rel_dm_part_pos_z[rpFDM<1.0*h_chunk['rvir'][i]]

                            #inertia tensor
                            Sxx = np.sum(relx_red*relx_red)/len(relx_red)
                            Sxy = np.sum(relx_red*rely_red)/len(relx_red)
                            Sxz = np.sum(relx_red*relz_red)/len(relx_red)
                            Syx = np.sum(rely_red*relx_red)/len(rely_red)
                            Syy = np.sum(rely_red*rely_red)/len(rely_red)
                            Syz = np.sum(rely_red*relz_red)/len(rely_red)
                            Szx = np.sum(relz_red*relx_red)/len(relz_red)
                            Szy = np.sum(relz_red*rely_red)/len(relz_red)
                            Szz = np.sum(relz_red*relz_red)/len(relz_red)

                            SS = np.array([[Sxx,Sxy,Sxz],[Syx,Syy,Syz],[Szx,Szy,Szz]])

                            #eigenvalues, eigenvectors
                            eigenval, eigenvec = np.linalg.eig(SS)

                            ai, bi, ci = np.real(eigenval)**0.5
                            re = (ai*bi*ci)**(1/3)
                            print("a, b, c, re = ",ai,bi,ci,re)

                            #rotation amtrix
                            Mrot = np.column_stack((np.real(eigenvec[:,2]),np.real(eigenvec[:,1]),np.real(eigenvec[:,0])))

                            #rotate particles
                            rel_dm_part_pos_rot = np.linalg.inv(Mrot).dot(rel_dm_part_pos)
                        '''

                        #calculate displacement
                        DpFDM_sph_rot_x = np.zeros(len(rel_dm_part_pos_rot[0]))
                        DpFDM_sph_rot_y = np.zeros(len(rel_dm_part_pos_rot[1]))
                        DpFDM_sph_rot_z = np.zeros(len(rel_dm_part_pos_rot[2]))

                        AA = param.code.spher_amplitude
                        th = param.code.spher_scale
                        nn = param.code.spher_powerlaw
                        DpFDM_sph_rot_x = (re_ov_ai - 1) * rel_dm_part_pos_rot[0] * sph_corr(rpFDM,h_chunk['rvir'][i],AA,nn,th)
                        DpFDM_sph_rot_y = (re_ov_bi - 1) * rel_dm_part_pos_rot[1] * sph_corr(rpFDM,h_chunk['rvir'][i],AA,nn,th)
                        DpFDM_sph_rot_z = (re_ov_ci - 1) * rel_dm_part_pos_rot[2] * sph_corr(rpFDM,h_chunk['rvir'][i],AA,nn,th)
                        DpFDM_sph_rot = [DpFDM_sph_rot_x,DpFDM_sph_rot_y,DpFDM_sph_rot_z]
                        
                        
                        #DpFDM_sph_rot_x[rpFDM<1.0*h_chunk['rvir'][i]] = (re_ov_ai - 1) * rel_dm_part_pos_rot[0][rpFDM<1.0*h_chunk['rvir'][i]]
                        #DpFDM_sph_rot_y[rpFDM<1.0*h_chunk['rvir'][i]] = (re_ov_bi - 1) * rel_dm_part_pos_rot[1][rpFDM<1.0*h_chunk['rvir'][i]]
                        #DpFDM_sph_rot_z[rpFDM<1.0*h_chunk['rvir'][i]] = (re_ov_ci - 1) * rel_dm_part_pos_rot[2][rpFDM<1.0*h_chunk['rvir'][i]]
                        #DpFDM_sph_rot = [DpFDM_sph_rot_x,DpFDM_sph_rot_y,DpFDM_sph_rot_z]

                        #Rotate back
                        DpFDM_sph_rotback = Mrot.dot(DpFDM_sph_rot)

                        DpFDM_sph['x'][ipbool] += DpFDM_sph_rotback[0]
                        DpFDM_sph['y'][ipbool] += DpFDM_sph_rotback[1]
                        DpFDM_sph['z'][ipbool] += DpFDM_sph_rotback[2]

                        
                        Earot = np.linalg.inv(Mrot).dot(Ea)
                        Ebrot = np.linalg.inv(Mrot).dot(Eb)
                        Ecrot = np.linalg.inv(Mrot).dot(Ec)

                        '''
                        import matplotlib
                        import matplotlib.pyplot as plt
                        import matplotlib.colors as colors
                        from matplotlib import cm
                        f,ax  = plt.subplots(1, 6)
                        ax[0].plot(p_darkmatter['x'][ipbool][rpFDM<2*h_chunk['rvir'][i]], p_darkmatter['y'][ipbool][rpFDM<2*h_chunk['rvir'][i]], 'bo',ms=0.1)
                        ax[0].plot([h_chunk['x'][i],h_chunk['x'][i] + Ea[0]], [h_chunk['y'][i], h_chunk['y'][i] + Ea[1]], ls='-',color='red')
                        ax[0].plot([h_chunk['x'][i],h_chunk['x'][i] + Eb[0]], [h_chunk['y'][i], h_chunk['y'][i] + Eb[1]], ls='-',color='cyan')
                        ax[0].plot([h_chunk['x'][i],h_chunk['x'][i] + Ec[0]], [h_chunk['y'][i], h_chunk['y'][i] + Ec[1]], ls='-',color='magenta')
                    
                        c1 = plt.Circle((h_chunk['x'][i], h_chunk['y'][i]), h_chunk['rvir'][i], color='red',fc='None',ls='--')
                        ax[0].add_patch(c1)
                        ax[0].set_xlim([h_chunk['x'][i]-2*h_chunk['rvir'][i], h_chunk['x'][i]+2*h_chunk['rvir'][i]])
                        ax[0].set_ylim([h_chunk['y'][i]-2*h_chunk['rvir'][i], h_chunk['y'][i]+2*h_chunk['rvir'][i]])
                        ax[0].set_aspect(1)

                        ax[1].plot(p_darkmatter['x'][ipbool][rpFDM<2*h_chunk['rvir'][i]], p_darkmatter['z'][ipbool][rpFDM<2*h_chunk['rvir'][i]], 'bo',ms=0.1)
                        ax[1].plot([h_chunk['x'][i],h_chunk['x'][i] + Ea[0]], [h_chunk['z'][i], h_chunk['z'][i] + Ea[2]], ls='-',color='red')
                        ax[1].plot([h_chunk['x'][i],h_chunk['x'][i] + Eb[0]], [h_chunk['z'][i], h_chunk['z'][i] + Eb[2]], ls='-',color='cyan')
                        ax[1].plot([h_chunk['x'][i],h_chunk['x'][i] + Ec[0]], [h_chunk['z'][i], h_chunk['z'][i] + Ec[2]], ls='-',color='magenta')
                    
                        c1 = plt.Circle((h_chunk['x'][i], h_chunk['z'][i]), h_chunk['rvir'][i], color='red',fc='None',ls='--')
                        ax[1].add_patch(c1)
                        ax[1].set_xlim([h_chunk['x'][i]-2*h_chunk['rvir'][i], h_chunk['x'][i]+2*h_chunk['rvir'][i]])
                        ax[1].set_ylim([h_chunk['z'][i]-2*h_chunk['rvir'][i], h_chunk['z'][i]+2*h_chunk['rvir'][i]])
                        ax[1].set_aspect(1)

                        ax[2].plot(rel_dm_part_pos_rot[0][rpFDM<2*h_chunk['rvir'][i]], rel_dm_part_pos_rot[1][rpFDM<2*h_chunk['rvir'][i]], 'go',ms=0.1)
                        ax[2].plot([0, Earot[0]], [0, Earot[1]], ls='-',color='red')
                        ax[2].plot([0, Ebrot[0]], [0, Ebrot[1]], ls='-',color='cyan')
                        ax[2].plot([0, Ecrot[0]], [0, Ecrot[1]], ls='-',color='magenta')
                        c1 = plt.Circle((0, 0), h_chunk['rvir'][i], color='red',fc='None',ls='--')
                        ax[2].add_patch(c1)
                        ax[2].set_xlim([-2*h_chunk['rvir'][i], 2*h_chunk['rvir'][i]])
                        ax[2].set_ylim([-2*h_chunk['rvir'][i], 2*h_chunk['rvir'][i]])
                        ax[2].set_aspect(1)

                        ax[3].plot(rel_dm_part_pos_rot[0][rpFDM<2*h_chunk['rvir'][i]], rel_dm_part_pos_rot[2][rpFDM<2*h_chunk['rvir'][i]], 'go',ms=0.1)
                        ax[3].plot([0, Earot[0]], [0, Earot[2]], ls='-',color='red')
                        ax[3].plot([0, Ebrot[0]], [0, Ebrot[2]], ls='-',color='cyan')
                        ax[3].plot([0, Ecrot[0]], [0, Ecrot[2]], ls='-',color='magenta')
                        c1 = plt.Circle((0, 0), h_chunk['rvir'][i], color='red',fc='None',ls='--')
                        ax[3].add_patch(c1)
                        ax[3].set_xlim([-2*h_chunk['rvir'][i], 2*h_chunk['rvir'][i]])
                        ax[3].set_ylim([-2*h_chunk['rvir'][i], 2*h_chunk['rvir'][i]])
                        ax[3].set_aspect(1)
                    
                        ax[4].plot(rel_dm_part_pos_rot[0][rpFDM<2*h_chunk['rvir'][i]]+DpFDM_sph_rot_x[rpFDM<2*h_chunk['rvir'][i]], rel_dm_part_pos_rot[1][rpFDM<2*h_chunk['rvir'][i]]+DpFDM_sph_rot_y[rpFDM<2*h_chunk['rvir'][i]], 'go',ms=0.1)
                        ax[4].plot([0, Earot[0]], [0, Earot[1]], ls='-',color='red')
                        ax[4].plot([0, Ebrot[0]], [0, Ebrot[1]], ls='-',color='cyan')
                        ax[4].plot([0, Ecrot[0]], [0, Ecrot[1]], ls='-',color='magenta')
                        c1 = plt.Circle((0, 0), h_chunk['rvir'][i], color='red',fc='None',ls='--')
                        ax[4].add_patch(c1)
                        ax[4].set_xlim([-2*h_chunk['rvir'][i], 2*h_chunk['rvir'][i]])
                        ax[4].set_ylim([-2*h_chunk['rvir'][i], 2*h_chunk['rvir'][i]])
                        ax[4].set_aspect(1)

                        ax[5].plot(rel_dm_part_pos_rot[0][rpFDM<2*h_chunk['rvir'][i]]+DpFDM_sph_rot_x[rpFDM<2*h_chunk['rvir'][i]], rel_dm_part_pos_rot[2][rpFDM<2*h_chunk['rvir'][i]]+DpFDM_sph_rot_z[rpFDM<2*h_chunk['rvir'][i]], 'go',ms=0.1)
                        ax[5].plot([0, Earot[0]], [0, Earot[2]], ls='-',color='red')
                        ax[5].plot([0, Ebrot[0]], [0, Ebrot[2]], ls='-',color='cyan')
                        ax[5].plot([0, Ecrot[0]], [0, Ecrot[2]], ls='-',color='magenta')
                        c1 = plt.Circle((0, 0), h_chunk['rvir'][i], color='red',fc='None',ls='--')
                        ax[5].add_patch(c1)
                        ax[5].set_xlim([-2*h_chunk['rvir'][i], 2*h_chunk['rvir'][i]])
                        ax[5].set_ylim([-2*h_chunk['rvir'][i], 2*h_chunk['rvir'][i]])
                        ax[5].set_aspect(1)

                        plt.show()
                        '''

                    
                    #Add temperature
                    TrpBAR = splev(rpBAR,TGAS_tck,der=0,ext=1)
                    DpBAR['temp'][ipbool] += TrpBAR

                    #Add electron pressure
                    PrpBAR = splev(rpBAR,PGAS_tck,der=0,ext=1)
                    DpBAR['pres'][ipbool] += PrpBAR

                    
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
                    if (param.code.satgal==False):
                        
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
                    elif (param.code.satgal==True):

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
                            #particle_mass = param.cosmo.Ob*rhoc_of_z(param)*Lbox**3/(p_header['Npart']/2)
                            particle_mass = param.cosmo.Ob*RHOC*Lbox**3/(p_header['Npart']/2)
                            fstar_halo_tot = fSTAR_fct(h_chunk['Mvir'][i],param,param.baryon.eta)
                            fstar_halo_cga = fSTAR_fct(h_chunk['Mvir'][i],param,param.baryon.eta+param.baryon.deta)
                            Mstar_halo_sat = (fstar_halo_tot-fstar_halo_cga)*h_chunk['Mvir'][i] #expected total satellite stellar mass (fstar-fcga)*Mvir_host
                            Mstar_halo_sat_from_sh_cga = np.sum(fSTAR_fct(shi['Mvir'],param,param.baryon.eta+param.baryon.deta)*shi['Mvir']) #expected total sat stellar mass from fcga_sat*Mvir_sat 
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
                                    fstar_sh_cga = Sat_boost*fSTAR_fct(shi['Mvir'][j],param,param.baryon.eta+param.baryon.deta)
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
                                    
                                    fstar_sh_cga = fSTAR_fct(shi['Mvir'][j],param,param.baryon.eta+param.baryon.deta)
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
                            
                            ##stars in satellite galaxy
                            #particle_mass = param.cosmo.Ob*rhoc_of_z(param)*Lbox**3/(p_header['Npart']/2)
                            #fstar_halo_tot = fSTAR_fct(h_chunk['Mvir'][i],param,param.baryon.eta)
                            #fstar_halo_cga = fSTAR_fct(h_chunk['Mvir'][i],param,param.baryon.eta+param.baryon.deta)
                            #Mstar_halo_sat = (fstar_halo_tot-fstar_halo_cga)*h_chunk['Mvir'][i] #expected total satellite stellar mass (fstar-fcga)*Mvir_host                                                                                                                                                                                                                            
                            #Mstar_halo_sat_from_sh_cga = np.sum(fSTAR_fct(shi['Mvir'],param,param.baryon.eta+param.baryon.deta)*shi['Mvir']) #expected total sat stellar mass from fcga_sat*Mvir_sat                                                                                                                                                                                     
                            #Sat_boost = Mstar_halo_sat/Mstar_halo_sat_from_sh_cga
                            #print("Sat_boost = ", Sat_boost)
                            #print("Npart_tot = ", Mstar_halo_sat/particle_mass)

                            #Stars in central galaxy
                            #RN = np.random.rand(len(pHGA))
                            #setID = np.zeros_like(RN).astype(int)
                            #setID[RN<pCGA] = 1
                            #if(len(rpBAR_nbrhaloes)>0):
                            #    DpBAR['id'][ipbool_wo_nbrhaloes] += setID
                            #else:
                            #    DpBAR['id'][ipbool] += setID
                                
                            #for j in range(Nsh):
                            #    fstar_sh_cga = Sat_boost*fSTAR_fct(shi['Mvir'][j],param,param.baryon.eta+param.baryon.deta)
                            #    Mstar_sh_cga = fstar_sh_cga*shi['Mvir'][j]
                            #    Npart = np.round(Mstar_sh_cga/particle_mass)
                            #    if (Npart == 0):
                            #        Npart = 1
                            #    print("Npart_per_sat = ", Npart)
			    #    dist_, ip_sh_stars = p_tree.query((shi['x'][j],shi['y'][j],shi['z'][j]), Npart, distance_upper_bound=shi['rvir'][j])
			    #    #Set ID: 0=gas+sga, 1=stars
                            #    ip_sh_stars_clean = np.delete(ip_sh_stars, np.where(ip_sh_stars == len(DpBAR['id'])))
                            #    #if (len(ip_sh_stars)):
                            #    DpBAR['id'][ip_sh_stars_clean] = 1
                                
                '''
                ##################################################################
                #import matplotlib

                if (i % 100==0):
                    print(i)
                    #calculate profiles
                    rbin_edges = np.logspace(np.log10(0.001),np.log10(10),100,base=10)
                    Vol_edges  = 4*np.pi/3*rbin_edges**3
                    dVol       = Vol_edges[1:] - Vol_edges[:-1]

                    DrpBAR = splev(rpBAR,DBAR_tck,der=0,ext=1)
                    DrpFDM = splev(rpFDM,DFDM_tck,der=0,ext=1)
                    profiles_dmo, bin_edges = np.histogram(rpFDM,bins=rbin_edges)
                    profiles_bar, bin_edges = np.histogram(rpBAR+DrpBAR,bins=rbin_edges)
                    profiles_fdm, bin_edges = np.histogram(rpFDM+DrpFDM,bins=rbin_edges)

                    dmo_mass_per_part = param.cosmo.Om*rhoc_of_z(param)*Lbox**3/(p_header['Npart']/2)
                    bar_mass_per_part = gas_mass_per_part = star_mass_per_part = param.cosmo.Ob*rhoc_of_z(param)*Lbox**3/(p_header['Npart']/2)
                    fdm_mass_per_part = (param.cosmo.Om-param.cosmo.Ob)*rhoc_of_z(param)*Lbox**3/(p_header['Npart']/2)

                    profiles_dmo = dmo_mass_per_part*profiles_dmo
                    profiles_bar = bar_mass_per_part*profiles_bar
                    profiles_fdm = fdm_mass_per_part*profiles_fdm

                    profiles_dmo = np.cumsum(profiles_dmo)
                    profiles_bar = np.cumsum(profiles_bar)
                    profiles_fdm = np.cumsum(profiles_fdm)
                    
                    print("halo position = ", h_chunk['x'][i],h_chunk['y'][i],h_chunk['z'][i])
                    print("model0: Mc, mu thej, ga, de, eta, deta = ", param.baryon.Mc,param.baryon.mu,param.baryon.thej,param.baryon.gamma,param.baryon.delta, param.baryon.eta, param.baryon.deta)
                    print("nu,thco,al,rcga,eps,betamodel = ", param.baryon.nu, param.baryon.thco, param.baryon.alpha, param.baryon.rcga, param.code.eps, param.code.beta_model)

                    import matplotlib.pyplot as plt
                    fig, ax  = plt.subplots(1,2,figsize=(12,6))

                    ax[0].hlines(h_chunk['Mvir'][i],0.001,10,ls=':',color='grey')
                    ax[0].vlines(h_chunk['rvir'][i],1e5,1e16,ls=':',color='grey')
                    ax[0].loglog(rbin, mass['DMO'], ls='--', c='black')

                    ax[0].loglog(rbin_edges[1:],profiles_dmo,  ls='-', lw=5,c='black',alpha=0.5)
                    ax[0].grid()
                    ax[0].set_xlabel('rb[Mpc/h]')
                    ax[0].set_ylabel('Mass [Msun/h]')
                    ax[0].set_xlim([0.005, 9])
                    ax[0].set_ylim([5e8,2e15])

                    ax[1].hlines(h_chunk['Mvir'][i],0.001,10,ls=':',color='grey')
                    ax[1].vlines(h_chunk['rvir'][i],1e5,1e16,ls=':',color='grey')
                    ax[1].loglog(rbin, mass['HGA'] + mass['CGA'] + mass['SGA'], ls='--', c='blue')
                    ax[1].loglog(rbin, mass['CDM'], ls='--', c='red')
                    ax[1].loglog(rbin, mass['DMB'], ls='--', c='black')
                    ax[1].loglog(rbin, mass['DMO'], ls='--', c='grey')

                    ax[1].loglog(rbin_edges[1:],profiles_bar,  ls='-', lw=5,c='blue',alpha=0.5)
                    ax[1].loglog(rbin_edges[1:],profiles_fdm,  ls='-', lw=5, c='red',alpha=0.5)
                    ax[1].loglog(rbin_edges[1:],profiles_fdm + profiles_bar,  ls='-', lw=5, c='black',alpha=0.5)
                    ax[1].grid()
                    ax[1].set_xlabel('rb[Mpc/h]')
                    ax[1].set_ylabel('Mass [Msun/h]')
                    ax[1].set_xlim([0.005, 9])
                    ax[1].set_ylim([5e8,2e15])
                
                    plt.show()
                '''

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

        ##Add temperature floor
        #zz = param.cosmo.z
        #T0 = 1e4*(1+zz)/2 # [K] McQuinn (1512.00086,below Eq. 6)
        #p_gas_chunk['temp'] = p_gas_chunk['temp'] + T0
        
        del p_gas_temporary_chunk 
        
        del p_baryons
        del p_darkmatter
            
    elif (param.code.multicomp==False):

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
                rmax = param.code.rmax
                rmin = (0.001*h_chunk['rvir'][i] if 0.001*h_chunk['rvir'][i]>param.code.rmin else param.code.rmin)
                rmax = (20.0*h_chunk['rvir'][i] if 20.0*h_chunk['rvir'][i]<param.code.rmax else param.code.rmax)
                rbin = np.logspace(np.log10(rmin),np.log10(rmax),100,base=10)

                #initialse profiles
                profiles = Profiles(rbin, h_chunk['Mvir'][i], h_chunk['cvir'][i], param)

                #calculate profiles for displacement
                frac, dens, mass, pres, temp = profiles.calc_profiles()

                #DDMB = displ(rbin,mass['DMO'],mass['DMB'])
                DDMB = displ(rbin,mass['NFW'] + mass['BG'], mass['CDM'] + mass['BG'])

                DDMB_tck = splrep(rbin, DDMB,s=0,k=3)

                #define minimum displacement
                smallestD = param.code.disp_trunc #Mpc/h

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

                    '''
                    #Sphericity correction (for singlecomp / not tested!)
                    if (param.code.spher_corr==True):

                        # INERTIA TENSOR FROM AHF (what about rockstar?)
                        #define relative particle position (should be done further up)                                                                                                                                                                                            
                        rel_part_pos_x = p_chunk['x'][ipbool]-h_chunk['x'][i]
                        rel_part_pos_y = p_chunk['y'][ipbool]-h_chunk['y'][i]
                        rel_part_pos_z = p_chunk['z'][ipbool]-h_chunk['z'][i]
                        rel_part_pos = [rel_part_pos_x,rel_part_pos_y,rel_part_pos_z]

                        #rotation amtrix
                        Ea = [h_chunk['Eax'][i],h_chunk['Eay'][i],h_chunk['Eaz'][i]]
                        Eb = [h_chunk['Ebx'][i],h_chunk['Eby'][i],h_chunk['Ebz'][i]]
                        Ec = [h_chunk['Ecx'][i],h_chunk['Ecy'][i],h_chunk['Ecz'][i]]
                        Mrot = np.column_stack((Ea,Eb,Ec))

                        #rotate particles
                        rel_part_pos_rot = np.linalg.inv(Mrot).dot(rel_part_pos)

                        #Displacement correction
                        re_ov_ci = (h_chunk['c_ov_a'][i]*h_chunk['b_ov_a'][i])**(1/3)/h_chunk['c_ov_a'][i]
                        re_ov_bi = (h_chunk['c_ov_a'][i]*h_chunk['b_ov_a'][i])**(1/3)/h_chunk['b_ov_a'][i]
                        re_ov_ai = (h_chunk['c_ov_a'][i]*h_chunk['b_ov_a'][i])**(1/3)

                        #calculate displacement
                        Dp_sph_rot_x = np.zeros(len(rel_dm_part_pos_rot[0]))
                        Dp_sph_rot_y = np.zeros(len(rel_dm_part_pos_rot[1]))
                        Dp_sph_rot_z = np.zeros(len(rel_dm_part_pos_rot[2]))
                        
                        Dp_sph_rot_x[rpDMB<1.0*h_chunk['rvir'][i]] = (re_ov_ai - 1) * rel_part_pos_rot[0][rpDMB<1.0*h_chunk['rvir'][i]]
                        Dp_sph_rot_y[rpDMB<1.0*h_chunk['rvir'][i]] = (re_ov_bi - 1) * rel_part_pos_rot[1][rpDMB<1.0*h_chunk['rvir'][i]]
                        Dp_sph_rot_z[rpDMB<1.0*h_chunk['rvir'][i]] = (re_ov_ci - 1) * rel_part_pos_rot[2][rpDMB<1.0*h_chunk['rvir'][i]]
                        Dp_sph_rot = [Dp_sph_rot_x,Dp_sph_rot_y,Dp_sph_rot_z]

                        #Rotate back
                        Dp_sph_rotback = Mrot.dot(Dp_sph_rot)

                        Dp_sph['x'][ipbool] += DpFDM_sph_rotback[0]
                        Dp_sph['y'][ipbool] += DpFDM_sph_rotback[1]
                        Dp_sph['z'][ipbool] += DpFDM_sph_rotback[2]
                    '''    

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
