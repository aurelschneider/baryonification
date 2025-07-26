"""
CALCULATE DISPLACEMENT FUNCTION FOR A GRID OF M AND C
PRINT INFORMATION INTO TEMPORARY FILE

"""

import numpy as np
from scipy import spatial
from scipy.interpolate import splrep,splev
from numpy.lib.recfunctions import append_fields

import schwimmbad

from .constants import *
from .profiles import *
from .io import *


"""
MODIFY MASS FUNCTION
"""


def modify_mass(param):

    """
    Reading in N-body and halo files, defining chunks, 
    looping over haloes in single or multi-processor mode,
    modifying particle masses, combining chunks, 
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

        displ_p = modify_mass_chunk(p_header,p_list[0],h_list[0],param)
        
        p_gas_displ  = [displ_p[0]]
        p_dm_displ   = [displ_p[1]]
        p_star_displ = [displ_p[2]]
        
    elif (N_cpu > 1):

        pool = schwimmbad.choose_pool(mpi=False, processes=N_cpu)
        tasks = list(zip(p_list,h_list,np.repeat(p_header,N_cpu),np.repeat(param,N_cpu)))
        
        modmass_p = np.array(pool.map(worker, tasks))
        
        p_gas_modmass  = modmass_p[:,0]
        p_dm_modmass   = modmass_p[:,1]
        p_star_modmass = modmass_p[:,2]

        pool.close()

    p_gas  = np.concatenate(p_gas_modmass)
    p_dm   = np.concatenate(p_dm_modmass)
    p_star = np.concatenate(p_star_modmass)
    
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

    p_chunk, h_chunk, p_header, param = task

    p_gas_chunk, p_dm_chunk, p_star_chunk = modify_mass_chunk(p_header,p_chunk,h_chunk,param)
    
    return np.array([p_gas_chunk, p_dm_chunk, p_star_chunk],dtype=object)


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


def modify_mass_chunk(p_header,p_chunk,h_chunk,param):

    """
    Reading in N-body and halo files, looping over haloes, calculating
    density fractions, modify masses..
    """

    #relevant parameters
    Mc   = param.baryon.Mc
    mu   = param.baryon.mu
    nu   = param.baryon.nu
    thej = param.baryon.thej
    Lbox = param.sim.Lbox
    Om   = param.cosmo.Om
    
    #Read cosmic variance/nu/correlation and interpolate
    cosmofile = param.files.cosmofct
    try:
        vc_r, vc_m, vc_bias, vc_corr = np.loadtxt(cosmofile, usecols=(0,1,2,3), unpack=True)
        bias_tck = splrep(vc_m, vc_bias, s=0)
        corr_tck = splrep(vc_r, vc_corr, s=0)
    except IOError:
        print('IOERROR: Cosmofct file does not exist!')
        print('Define par.files.cosmofct = "/path/to/file"')
        print('Run: cosmo(params) to create file')
        exit()

    if (param.code.multicomp==True):

        #Copy into p_temp
        dmp_type = np.dtype([("mass",'>f'),("id",'>i4')])
        dmpBAR   = np.ones(len(p_chunk),dtype=dmp_type)
        dmpFDM   = np.ones(len(p_chunk),dtype=dmp_type)
        dmpBAR['id'] = 0
        dmpFDM['id'] = 0
        
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

        #Loop over haloes, calculate density fractions, and modify particle masses
        for i in range(len(h_chunk['Mvir'])):

            #select host haloes (subhaloes >= 1)
            if (h_chunk['IDhost'][i] < 0.1):
                
                print('start: ', i)

                #range where we consider changes of mass
                rmax = param.code.rmax
                rmin = (0.001*h_chunk['rvir'][i] if 0.001*h_chunk['rvir'][i]>param.code.rmin else param.code.rmin)
                rmax = (20.0*h_chunk['rvir'][i] if 20.0*h_chunk['rvir'][i]<param.code.rmax else param.code.rmax)
                rbin = np.logspace(np.log10(rmin),np.log10(rmax),100,base=10)

                #calculate displacement
                cosmo_bias = splev(h_chunk['Mvir'][i],bias_tck)
                cosmo_corr = splev(rbin,corr_tck)
                frac, dens, mass, press, temp = profiles(rbin,h_chunk['Mvir'][i],h_chunk['cvir'][i],cosmo_corr,cosmo_bias,param)

                #EMPIRICAL NUMBER (1 does not work). IT has toi be ~4 to make the PS agree on alrge scales
                EMP = 4.0
                
                #baryon density fractions
                rhoBAR_initial = (frac['HGA'] + frac['CGA'] + frac['SGA'])*dens['NFW']
                rhoBAR_final   = frac['HGA']*dens['HGA'] + frac['CGA']*dens['CGA'] + frac['SGA']*dens['SGA']
                rhoBAR_2H      = (frac['HGA'] + frac['CGA'] + frac['SGA'])*dens['BG']
                dmBAR          = (rhoBAR_final + EMP*rhoBAR_2H)/(rhoBAR_initial + EMP*rhoBAR_2H)
                dmBAR_tck      = splrep(rbin, dmBAR,s=0,k=3)

                print('BAR = ', np.trapz(rbin**2*rhoBAR_final,rbin)/np.trapz(rbin**2*rhoBAR_initial,rbin))
                print(np.trapz(rbin**2*(rhoBAR_final+rhoBAR_2H),rbin)/np.trapz(rbin**2*(rhoBAR_initial+rhoBAR_2H),rbin))

                #collisionless (final dark) matter density fraction
                rhoFDM_initial = (frac['CDM']+frac['SGA'])*dens['NFW']
                rhoFDM_final   = (frac['CDM']+frac['SGA'])*dens['ACM']
                rhoFDM_2H      = (frac['CDM']+frac['SGA'])*dens['BG']
                dmFDM          = (rhoFDM_final + EMP*rhoFDM_2H)/(rhoFDM_initial + EMP*rhoFDM_2H)
                dmFDM_tck      = splrep(rbin, dmFDM,s=0,k=3)

                print('FDM = ', np.trapz(rbin**2*rhoFDM_final,rbin)/np.trapz(rbin**2*rhoFDM_initial,rbin))
                print(np.trapz(rbin**2*(rhoFDM_final+rhoFDM_2H),rbin)/np.trapz(rbin**2*(rhoFDM_initial+rhoFDM_2H),rbin))
                
                #if (i % 100 == 0):
                #    print("mass i=20", h_chunk['Mvir'][i])
                #    import matplotlib.pyplot as plt
                #    plt.loglog(rbin,(dmBAR),c='b')
                #    plt.loglog(rbin,(dmFDM),c='red')
                #    #plt.axis([rmin,rmax,0,5])
                #    plt.show()
                #    #exit()
                    
                #define minimum density cut
                rho_min = 0.001*rhoc_of_z(param)
                #dm_min = 1e-5
                
                #array of idx with DBAR > Dsmallest
                idx_BAR = np.where(rhoBAR_final > rho_min)
                #idx_BAR = np.where(rhoBAR_final > 0.1*rhoBAR_2H)
                idx_BAR = idx_BAR[:][0]
                if (len(idx_BAR)>1):
                    idx_largest = idx_BAR[-1]
                    rball_BAR = rbin[idx_largest]
                else:
                    rball_BAR = 0.0

                #array of idx with DFDM > Dsmallest
                idx_FDM = np.where(rhoFDM_final > rho_min)
                #idx_FDM = np.where(rhoFDM_final > 0.1*rhoFDM_2H)
                idx_FDM = idx_FDM[:][0]
                if (len(idx_FDM)>1):
                    idx_largest = idx_FDM[-1]
                    rball_FDM = rbin[idx_largest]
                else:
                    rball_FDM = 0.0
                
                #largest rball
                #rball = 2*h_chunk['rvir'][i]
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
            
                #calculating radii of FDM particles around halo i
                if (len(ipbool) > 0):
                    rpFDM  = ((p_darkmatter['x'][ipbool]-h_chunk['x'][i])**2.0 +
                             (p_darkmatter['y'][ipbool]-h_chunk['y'][i])**2.0 +
                             (p_darkmatter['z'][ipbool]-h_chunk['z'][i])**2.0)**0.5

                    #calculating radii of BAR particles around halo i
                    rpBAR = ((p_baryons['x'][ipbool]-h_chunk['x'][i])**2.0 +
                            (p_baryons['y'][ipbool]-h_chunk['y'][i])**2.0 +
                            (p_baryons['z'][ipbool]-h_chunk['z'][i])**2.0)**0.5

                    #change mass
                    dmpBAR['mass'][ipbool] *= splev(rpBAR,dmBAR_tck,der=0,ext=3)
                    dmpFDM['mass'][ipbool] *= splev(rpFDM,dmFDM_tck,der=0,ext=3)

                    #probabilities
                    probHGA = np.round(frac['HGA']*dens['HGA']/(frac['HGA']*dens['HGA'] + frac['CGA']*dens['CGA'] + frac['SGA']*dens['SGA']),8)
                    probCGA = np.round(frac['CGA']*dens['CGA']/(frac['HGA']*dens['HGA'] + frac['CGA']*dens['CGA'] + frac['SGA']*dens['SGA']),8)
                    probSGA = np.round(frac['SGA']*dens['SGA']/(frac['HGA']*dens['HGA'] + frac['CGA']*dens['CGA'] + frac['SGA']*dens['SGA']),8)

                    #Only HGA outside of virial radius
                    probHGA[np.where(rbin>=h_chunk['rvir'][i])] = 1.0
                    probCGA[np.where(rbin>=h_chunk['rvir'][i])] = 0.0
                    probSGA[np.where(rbin>=h_chunk['rvir'][i])] = 0.0

                    probHGA_tck = splrep(rbin, probHGA, s=0, k=3)
                    probCGA_tck = splrep(rbin, probCGA, s=0, k=3)
                    probSGA_tck = splrep(rbin, probSGA, s=0, k=3)

                    #calculate the probabilities per particle
                    pHGA = splev(rpBAR, probHGA_tck,der=0,ext=3)
                    pCGA = splev(rpBAR, probCGA_tck,der=0,ext=3)
                    pSGA = splev(rpBAR, probSGA_tck,der=0,ext=3)
                        
                    #Set ID: 0=gas, 1=star
                    if (param.code.satgal==False):
                        #random number between 0 and 1
                        RN = np.random.rand(len(pHGA))
                        setID = np.zeros_like(RN).astype(int)
                        setID[RN>pHGA] = 1
                        dmpBAR['id'][ipbool] += setID

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
                            dmpBAR['id'][ipbool] += setID
                                
                        else:
                            #If there is at least one satellite, the sga-stars are dumped around its/their centre
                            print("Satellites detected. SGA stars (f_sga) added to satellite positions.")
                            print("N_sat = ", Nsh)

                            #Stars in central galaxy
                            RN = np.random.rand(len(pHGA))
                            setID = np.zeros_like(RN).astype(int)
                            setID[RN<pCGA] = 1
                            dmpBAR['id'][ipbool] += setID

                            #stars in satellite galaxy
                            particle_mass = param.cosmo.Ob*rhoc_of_z(param)*Lbox**3/(p_header['Npart']/2)
                            fstar_halo_tot = fSTAR_fct(h_chunk['Mvir'][i],param,param.baryon.eta)
                            fstar_halo_cga = fSTAR_fct(h_chunk['Mvir'][i],param,param.baryon.eta+param.baryon.deta)
                            Mstar_halo_sat = (fstar_halo_tot-fstar_halo_cga)*h_chunk['Mvir'][i]
                            Mstar_halo_sat_from_sh_cga = np.sum(fSTAR_fct(shi['Mvir'],param,param.baryon.eta+param.baryon.deta)*shi['Mvir'])
                            Sat_boost = Mstar_halo_sat/Mstar_halo_sat_from_sh_cga
                            print("Sat_boost = ", Sat_boost)
                            print("Npart_tot = ", Mstar_halo_sat/particle_mass)

                            for i in range(Nsh):
                                fstar_sh_cga = Sat_boost*fSTAR_fct(shi['Mvir'][i],param,param.baryon.eta+param.baryon.deta)
                                Mstar_sh_cga = fstar_sh_cga*shi['Mvir'][i]
                                Npart = int(Mstar_sh_cga/particle_mass)
                                print("Npart_per_sat = ", Npart)
                                dist_, ip_sh_stars = p_tree.query((shi['x'][i],shi['y'][i],shi['z'][i]), Npart, distance_upper_bound=shi['rvir'][i])

                                #Set ID: 0=gas+sga, 1=stars
                                ip_sh_stars_clean = np.delete(ip_sh_stars, np.where(ip_sh_stars == len(dmpBAR['id'])))
                                #if (len(ip_sh_stars)):
                                dmpBAR['id'][ip_sh_stars_clean] = 1

        print("max fractional mass change:")
        print(np.min(dmpBAR['mass']+1))
        print(np.min(dmpFDM['mass']+1))

        print("minmax")
        print(np.min(p_baryons['mass']),np.max(p_baryons['mass']))
        print(np.min(p_darkmatter['mass']),np.max(p_darkmatter['mass']))

        MTOTi = np.sum(p_baryons['mass'])+np.sum(p_darkmatter['mass'])
        print("M_TOTi", MTOTi)
        
        #Displace particles and separarte BAR into HGA, CGA, SGA
        p_baryons['mass'] = dmpBAR['mass']*p_baryons['mass']
        p_darkmatter['mass'] = dmpFDM['mass']*p_darkmatter['mass']
        
        #Separate baryons into gas and star
        p_gas_chunk  = p_baryons[dmpBAR['id']<0.5]
        p_dm_chunk   = p_darkmatter
        p_star_chunk = p_baryons[dmpBAR['id']>0.5]

        MTOTf = np.sum(p_baryons['mass'])+np.sum(p_darkmatter['mass'])
        print("M_TOTf", MTOTf)

        p_baryons['mass'] = MTOTi/MTOTf * p_baryons['mass']
        p_darkmatter['mass'] = MTOTi/MTOTf * p_darkmatter['mass']
        MTOTff = np.sum(p_baryons['mass'])+np.sum(p_darkmatter['mass'])
        print("M_TOTff", MTOTff)

        del p_baryons
        del p_darkmatter
            
    elif (param.code.multicomp==False):

        #Copy into p_temp
        dmp_dt = np.dtype([("mass",'>f')])
        #dmp    = np.zeros(len(p_chunk),dtype=dmp_dt)
        dmp    = np.ones(len(p_chunk),dtype=dmp_dt)

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
                rbin = np.logspace(np.log10(rmin),np.log10(rmax),param.code.Nrbin,base=10)

                #calculate displacement
                cosmo_bias = splev(h_chunk['Mvir'][i],bias_tck)
                cosmo_corr = splev(rbin,corr_tck)
                frac, dens, mass, press, temp = profiles(rbin,h_chunk['Mvir'][i],h_chunk['cvir'][i],cosmo_corr,cosmo_bias,param)

                #density fraction
                rhoDMB_initial = dens['NFW']
                rhoDMB_final   = dens['DMB']
                rhoDMB_2H      = dens['BG']
                #dmDMB          = (rhoDMB_final - rhoDMB_initial)/(rhoDMB_initial + rho2H) * Om*rhoc_of_z(param)*Lbox**3/p_header['Npart']

                #delta_m = (m_new-m_old)/m_old
                dmDMB          = (rhoDMB_final + rhoDMB_2H)/(rhoDMB_initial + rhoDMB_2H)
                #dmDMB          = (rhoDMB_final - rhoDMB_initial)/(rhoDMB_initial + rho2H)
                dmDMB_tck      = splrep(rbin, dmDMB,s=0,k=3)

                #define minimum density
                rho_min = 0.001*rhoc_of_z(param)
                    
                #array of idx with rho>rho_min
                idx = np.where(rhoDMB_final > rho_min)
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

                #change mass
                if (rball>0.0 and len(rpDMB)):
                    dmp['mass'][ipbool] *= splev(rpDMB,dmDMB_tck,der=0,ext=1)
                    #dmp['mass'][ipbool] += splev(rpDMB,dmDMB_tck,der=0,ext=1)
                    
        #Displace
        p_chunk['mass'] = dmp['mass']*p_chunk['mass']
        
        print("minmax",np.min(p_chunk['mass']),np.max(p_chunk['mass']))
        
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
