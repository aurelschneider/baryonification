import healpy as hp
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import splrep, splev
from collections import defaultdict

from .params import *

def arcdistance(distance,radius,param):
    '''Correct distance calculation for low redshift shells'''
    if radius > param.code.cur_radi:
        return distance
    else:
        angle = 2*np.arcsin(distance/(2*radius))
        return radius*angle

def euclidean_distance(arcdistance,radius,param):
    '''Correct distance calculation for low redshift shells'''
    if radius > param.code.cur_radi:
        return arcdistance
    else:
        angle = arcdistance/radius
        return 2*radius*np.sin(angle/2)

def arcdisplace(displace,position,radius,param):
    '''Correct displacement calculation for low redshift shells'''
    if radius > param.code.cur_radi:
        return displace
    else:
        norm_position = position / radius
        tangent = displace - np.sum(displace * norm_position, axis=1, keepdims=True) * norm_position
        displace_norm = np.linalg.norm(displace, axis=1, keepdims=True)
        tangent_norm = np.linalg.norm(tangent, axis=1)
        nonzero = tangent_norm > 1e-10
        corr_displace = np.copy(displace)
        corr_displace[nonzero] = tangent[nonzero] * displace_norm[nonzero] / tangent_norm[nonzero][:,np.newaxis]
        return corr_displace  

def get_particles(nside, pixels, shell_r, mass_threshold, mass_threshold_2, mass_threshold_3):
    pixel_indices = np.where(pixels > 0)[0]
    pixel_masses = pixels[pixel_indices]
    mask0 = pixel_masses <= mass_threshold
    mask1 = (pixel_masses > mass_threshold) & (pixel_masses <= mass_threshold_2)
    if mass_threshold_3 == None:
        mask2 = pixel_masses > mass_threshold_2
    elif mass_threshold_3 != None:
        mask2 = (pixel_masses > mass_threshold_2) & (pixel_masses <= mass_threshold_3)
        mask3 = pixel_masses > mass_threshold_3
    
    idx0 = pixel_indices[mask0]
    m0 = pixel_masses[mask0]
    dirs0 = np.array(hp.pix2vec(nside, idx0, nest=False))
    pos0 = dirs0.T * shell_r
    order0 = np.zeros(len(pos0), dtype=int)

    idx1 = pixel_indices[mask1]
    m1 = pixel_masses[mask1] / 4.0
    sub2 = 2 * nside
    idx1_n = hp.ring2nest(nside, idx1)
    children1 = hp.nest2ring(sub2, (idx1_n[:, None] * 4 + np.arange(4)).ravel())
    dirs1 = np.array(hp.pix2vec(sub2, children1, nest=False))
    pos1 = dirs1.T * shell_r
    m1_rep = np.repeat(m1, 4)
    order1 = np.ones(len(pos1), dtype=int)

    idx2 = pixel_indices[mask2]
    m2 = pixel_masses[mask2] / 16.0
    sub4 = 4 * nside
    idx2_n = hp.ring2nest(nside, idx2)
    grandchildren = hp.nest2ring(sub4, (idx2_n[:, None] * 16 + np.arange(16)).ravel())
    dirs2 = np.array(hp.pix2vec(sub4, grandchildren, nest=False))
    pos2 = dirs2.T * shell_r
    m2_rep = np.repeat(m2, 16)
    order2 = np.full(len(pos2), 2, dtype=int)

    if mass_threshold3 != None:
        idx3 = pixel_indices[mask3]
        m3 = pixel_masses[mask3] / 64.0
        sub8 = 8 * nside
        idx3_n = hp.ring2nest(nside, idx3)
        ggrandchildren = hp.nest2ring(sub8, (idx3_n[:, None] * 64 + np.arange(64)).ravel())
        dirs3 = np.array(hp.pix2vec(sub8, ggrandchildren, nest=False))
        pos3 = dirs3.T * shell_r
        m3_rep = np.repeat(m3, 64)
        order3 = np.full(len(pos3), 3, dtype=int)

    positions = np.vstack([pos0, pos1, pos2])
    masses = np.concatenate([m0, m1_rep, m2_rep])
    orders = np.concatenate([order0, order1, order2]).astype(np.int32)

    p_list = [(positions[i], masses[i], orders[i]) for i in range(len(masses))]
    return p_list

def get_particles_hmap(nside, pixels, shell_r, h):
    pixel_indices = np.where(pixels > 0)[0]
    npix = hp.nside2npix(nside)
    halo_map = np.zeros(npix, dtype=np.float64)
    xyz = np.column_stack((h['x'], h['y'], h['z']))
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)
    uv = xyz / norms 
    pix_ids = hp.vec2pix(nside, uv[:, 0], uv[:, 1], uv[:, 2], nest=False)
    np.add.at(halo_map, pix_ids, 1)
    del xyz, uv, norm
    
    halo_pixels = np.nonzero(h_map > 0)[0]
    neighbor_set = set()
    p_list = []
    for pix in halo_pixels:
        nbrs = hp.get_all_neighbours(nside, pix, nest=False)
        for nbr in nbrs:
            if nbr >= 0:
                neighbor_set.add(int(nbr))
    neighbor_set.difference_update(halo_pixels)
    for pix in pixel_indices:
        pix_mass = pixels[pix]

        # Case A: split into 16 sub-pixels (order=2)
        if pix in halo_pixels:
            sub_nside = 4 * nside
            m_sub = pix_mass / 16.0
            idx_n = hp.ring2nest(nside, np.array([pix]))
            grandchildren = hp.nest2ring(sub_nside,(idx_n[:, None] * 16 + np.arange(16)).ravel())
            dirs2 = np.array(hp.pix2vec(sub_nside, grandchildren, nest=False))
            pos2 = dirs2.T * shell_r
            for j in range(len(grandchildren)):
                p_list.append((pos2[j], m_sub, 2))

        # Case B: split into 4 sub-pixels (order=1)
        elif pix in neighbor_set:
            sub_nside = 2 * nside
            m_sub = pix_mass / 4.0
            idx_n = hp.ring2nest(nside, np.array([pix]))
            children = hp.nest2ring(sub_nside,(idx_n[:, None] * 4 + np.arange(4)).ravel())
            dirs1 = np.array(hp.pix2vec(sub_nside, children, nest=False))
            pos1 = dirs1.T * shell_r
            for j in range(len(children)):
                p_list.append((pos1[j], m_sub, 1))

        # Case C: no splitting (order=0)
        else:
            dirs0 = np.array(hp.pix2vec(nside, np.array([pix]), nest=False))
            pos0 = dirs0.T * shell_r
            p_list.append((pos0[0], pix_mass, 0))
    return p_list

def get_particles_uv(nside, pixels, shell_r, halos):
    npix = hp.nside2npix(nside)
    halo_pixels_dict = defaultdict(list)
    neighbor_map = np.zeros(npix, dtype=bool)
    h = halos[halos['IDhost']==-1]
    
    # Process each halo to mark its pixel and neighbors
    for i in range(len(h)):
        x, y, z = h['x'][i], h['y'][i], h['z'][i]
        norm = np.sqrt(x*x + y*y + z*z)
        ux, uy, uz = x/norm, y/norm, z/norm
        pix = hp.vec2pix(nside, ux, uy, uz, nest=False)
        halo_pixels_dict[pix].append(i)
        
        neighbors = hp.get_all_neighbours(nside, pix, nest=False)
        for nbr in neighbors:
            if nbr >= 0:
                neighbor_map[nbr] = True
    
    halo_map = np.array([pix in halo_pixels_dict for pix in range(npix)])
    neighbor_map &= ~halo_map  # Exclude halo pixels from neighbors
    
    # Precompute adjacent halos for neighbor pixels
    adjacent_halos_dict = defaultdict(list)
    for pix in np.where(neighbor_map)[0]:
        neighbors = hp.get_all_neighbours(nside, pix, nest=False)
        for nbr in neighbors:
            if nbr >= 0 and halo_map[nbr]:
                adjacent_halos_dict[pix].extend(halo_pixels_dict.get(nbr, []))
    
    # Process particle pixels
    particle_pixels = np.where(pixels > 0)[0]
    p_list = []

    def assign_weight(dists, h):
        x = dists / h['rvir']
        weight = np.zeros_like(x)

        mask1 = x < 1
        sqrt1 = np.sqrt((1 - x[mask1]) / (1 + x[mask1]))
        weight[mask1] = (1 - (2 / np.sqrt(1 - x[mask1]**2)) * np.arctanh(sqrt1)) / (x[mask1]**2 - 1)

        mask2 = np.isclose(x, 1)
        weight[mask2] = 1.0 / 3.0  # Value at x = 1

        mask3 = x > 1
        sqrt2 = np.sqrt((x[mask3] - 1) / (1 + x[mask3]))
        weight[mask3] = (1 - (2 / np.sqrt(x[mask3]**2 - 1)) * np.arctan(sqrt2)) / (x[mask3]**2 - 1)

        return weight
    
    for pix in particle_pixels:
        pix_mass = pixels[pix]
        
        if halo_map[pix]:
            # Case A: Halo pixel
            halos_in_pixel = halo_pixels_dict.get(pix, [])
            sub_nside = 4 * nside
            idx_n = hp.ring2nest(nside, pix)
            grandchildren = hp.nest2ring(sub_nside, idx_n * 16 + np.arange(16))
            dirs = np.array(hp.pix2vec(sub_nside, grandchildren, nest=False)).T
            sub_positions = dirs * shell_r
            
            mass_weights = np.ones(16)
            if halos_in_pixel:
                for halo_idx in halos_in_pixel:
                    halo_pos = np.array([h['x'][halo_idx], h['y'][halo_idx], h['z'][halo_idx]])
                    dists = np.linalg.norm(sub_positions - halo_pos, axis=1)
                    mass_weights += assign_weight(dists,h[halo_idx])
                mass_weights /= np.sum(mass_weights)
            
            for j in range(16):
                mass = pix_mass * mass_weights[j]
                p_list.append((sub_positions[j], mass, 2))
                
        elif neighbor_map[pix]:
            # Case B: Neighbor pixel 
            adjacent_halos = adjacent_halos_dict.get(pix, [])
            sub_nside = 2 * nside
            idx_n = hp.ring2nest(nside, pix)
            children = hp.nest2ring(sub_nside, idx_n * 4 + np.arange(4))
            dirs = np.array(hp.pix2vec(sub_nside, children, nest=False)).T
            sub_positions = dirs * shell_r
            
            mass_weights = np.ones(4)
            if adjacent_halos:
                for halo_idx in adjacent_halos:
                    halo_pos = np.array([h['x'][halo_idx], h['y'][halo_idx], h['z'][halo_idx]])
                    dists = np.linalg.norm(sub_positions - halo_pos, axis=1)
                    mass_weights += assign_weight(dists,h[halo_idx])
                mass_weights /= np.sum(mass_weights)
            
            for j in range(4):
                mass = pix_mass * mass_weights[j]
                p_list.append((sub_positions[j], mass, 1))
                
        else:
            # Case C: Regular pixel
            dirs = np.array(hp.pix2vec(nside, pix, nest=False))
            pos = dirs * shell_r
            p_list.append((pos, pix_mass, 0))
            
    return p_list

def get_child_pixels(parent_nside, pix, child_nside):
    """Get child pixels for hierarchical pixel subdivision"""
    order_diff = int(np.log2(child_nside / parent_nside))
    idx_n = hp.ring2nest(parent_nside, pix)
    return hp.nest2ring(child_nside, idx_n * (4**order_diff) + np.arange(4**order_diff))

def particle_worker(task):
    '''
    mesh_ref == 0: No any subgrid sampling
    mesh_ref == 1: Uniform subgrid sampling based on mass threshold
    mesh_ref == 2: Uniform subgrid sampling based on distance to halo centers
    mesh_ref == 3: Subgrid sampling following projected NFW profile.
    '''
    i, pixels, h, param = task
    shell_cov = np.sqrt(h['x'][0]**2 + h['y'][0]**2 + h['z'][0]**2)

    if param.code.mass_threshold_3 is None:
        p_dt = np.dtype([("x", '>f8'), ("y", '>f8'), ("z", '>f8'), ("M", '>f4'), ('ref_order', np.uint8)])
    else:
        p_dt = np.dtype([("x", '>f8'), ("y", '>f8'), ("z", '>f8'), ("M", '>f4'), ('ref_order', np.int32)])
    mesh_ref = param.code.mesh_ref
    nside = param.code.nside

    if mesh_ref == 0:
        idx = np.where(pixels > 0)[0]
        px, py, pz = hp.pix2vec(nside, idx)
        px *= shell_cov; py *= shell_cov; pz *= shell_cov
        p = np.zeros(len(px), dtype=p_dt)
        p['x'], p['y'], p['z'] = px, py, pz
        p['M'] = pixels[idx]
        p['ref_order'] = 0

    elif mesh_ref == 1:
        avg = np.mean(pixels)
        m1 = param.code.mass_threshold   * avg
        m2 = param.code.mass_threshold_2 * avg
        m3 = param.code.mass_threshold_3 * avg
        sub = get_particles(nside, pixels, shell_cov, m1, m2, m3)
        p = np.zeros(len(sub), dtype=p_dt)
        for j, (pos, mm, order) in enumerate(sub):
            p[j]['x'], p[j]['y'], p[j]['z'] = pos
            p[j]['M'] = mm
            p[j]['ref_order'] = order

    elif mesh_ref == 2:
        sub = get_particles_hmap(nside, pixels, shell_cov, h)
        p = np.zeros(len(sub), dtype=p_dt)
        for j, (pos, mm, order) in enumerate(sub):
            p[j]['x'], p[j]['y'], p[j]['z'] = pos
            p[j]['M'] = mm
            p[j]['ref_order'] = order

    elif mesh_ref == 3:
        sub = get_particles_uv(nside, pixels, shell_cov, h)
        p = np.zeros(len(sub), dtype=p_dt)
        for j, (pos, mm, order) in enumerate(sub):
            p[j]['x'], p[j]['y'], p[j]['z'] = pos
            p[j]['M'] = mm
            p[j]['ref_order'] = order
    else:
        raise ValueError(f"Unsupported mesh_ref: {mesh_ref}")
    return i, p

def get_healpix_map(p, param, star_fraction=None):
    nside      = param.code.nside
    nside_out  = param.code.nside_out
    interp     = param.code.interp
    x = p['x']; y = p['y']; z = p['z']
    masses = p['M'] * (1 - star_fraction) if star_fraction is not None else p['M']
    
    if not interp:
        # direct binning at final resolution: splitting order irrelevant
        r = np.linalg.norm(np.stack((x, y, z), axis=0), axis=0)
        pix_out = hp.vec2pix(nside_out,x/r, y/r, z/r)
        healpix_map = np.zeros(hp.nside2npix(nside_out))
        np.add.at(healpix_map, pix_out, masses)
        return healpix_map

    max_order = int(p['ref_order'].max())
    final_map = np.zeros(hp.nside2npix(nside_out))
    r = np.linalg.norm(np.stack((x, y, z), axis=0), axis=0)
    theta = np.arccos(z / r)
    phi   = np.mod(np.arctan2(y, x), 2*np.pi)

    for order in range(max_order + 1):
        mask = (p['ref_order'] == order)
        this_nside = nside * (2 ** order)
        this_map   = np.zeros(hp.nside2npix(this_nside))
        this_mass = masses[mask]
        th        = theta[mask]
        ph        = phi[mask]
        
        pix_ids, weights = hp.get_interp_weights(this_nside, th, ph)
        weights = weights.T
        pix_ids = pix_ids.T

        flat_pix = pix_ids.ravel()
        flat_m   = (this_mass[:,None] * weights).ravel()
        np.add.at(this_map, flat_pix, flat_m)

        # downgrade and accumulate
        final_map += hp.ud_grade(this_map,nside_out=nside_out,power=-2)

    return final_map.astype(np.float16)

def save_dict_to_hdf5(file, group_name, data):
    """
    Save a dictionary of HEALPix maps to an HDF5 group.
    """
    group = file.create_group(group_name)
    for shell_id, healpix_map in data.items():
        group.create_dataset(f'shell_{shell_id}', data=healpix_map)
    return

def projection(rho,rbin,rvir,thickness,param, output='mass', star=False):
    #create distance grid
    rmin = (0.001*rvir if 0.001*rvir>param.code.rmin else param.code.rmin)
    rmax = thickness/2
    if star == True:
        rmax = rvir
    
    r_int = np.logspace(np.log10(rmin),np.log10(rmax),200,base=10)
    R, Z = np.meshgrid(rbin, r_int, indexing='ij')
    r_3d = np.sqrt(R**2 + Z**2)

    tck_rho = splrep(rbin, rho, s=0, k=3)
    rho_3d = splev(r_3d.ravel(), tck_rho, ext=3).reshape(r_3d.shape)
    projected_rho = 2 * np.trapz(rho_3d, x=r_int, axis=1)
    if output == 'density':
        return projected_rho
        
    integrand = 2.0 * np.pi * rbin * projected_rho
    projected_Mass = cumulative_trapezoid(integrand, x=rbin, initial=0.0)

    return projected_Mass

def sphere_intersection_volume(r1, r2, d):

    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        R_min = min(r1, r2)
        return (4.0/3.0) * np.pi * R_min**3

    h1 = (r1 - (d**2 - r2**2 + r1**2) / (2.0 * d))
    h2 = (r2 - (d**2 - r1**2 + r2**2) / (2.0 * d))
    Vcap1 = (1.0/3.0) * np.pi * h1**2 * (3*r1 - h1)
    Vcap2 = (1.0/3.0) * np.pi * h2**2 * (3*r2 - h2)

    return Vcap1 + Vcap2

def impact_factor(h_cov, shell_cov, thickness, rball):

    r_in = shell_cov - thickness / 2
    r_out = shell_cov + thickness / 2
    r_in = max(r_in, 0.0)
    
    V_halo = (4/3) * np.pi * rball**3
    V_outer = sphere_intersection_volume(r_out, rball, h_cov)
    V_inner = sphere_intersection_volume(r_in, rball, h_cov)
    V_overlap = V_outer - V_inner
    V_overlap = max(0.0, V_overlap) 

    return V_overlap / V_halo

#The following functions are under testing:

def get_healpix_map_gaussian(p, param, star_fraction=None):

    nside     = param.code.nside
    nside_out = param.code.nside_out

    x = p['x']; y = p['y']; z = p['z']
    masses = p['M'] * (1 - star_fraction) if star_fraction is not None else p['M']
    coords = np.vstack((x, y, z)) 
    norms  = np.linalg.norm(coords, axis=0)
    unit_v = coords / norms

    sigma0 = 1.0 / (np.sqrt(6) * nside)
    max_order = int(p['ref_order'].max())
    final_map = np.zeros(hp.nside2npix(nside_out), dtype=np.float16)

    for order in range(max_order + 1):
        mask = (p['ref_order'] == order)
        if not np.any(mask):
            continue

        this_nside = nside * (2 ** order)
        this_sigma = sigma0 / (2 ** order)
        vecs_part = unit_v[:, mask]           
        this_mass = masses[mask]
        this_map = np.zeros(hp.nside2npix(this_nside), dtype=np.float64)

        for m, vec in zip(this_mass, vecs_part.T):
            radius   = 3.0 * this_sigma
            neighbors= hp.query_disc(this_nside, vec, radius=radius)
            if neighbors.size == 0:
                continue

            pix_vecs = np.vstack(hp.pix2vec(this_nside, neighbors)).T  
            ang = hp.rotator.angdist(vec, pix_vecs)
            w = np.exp(-0.5 * (ang / this_sigma)**2)
            w_sum = w.sum()
            if w_sum <= 0:
                print("Ops!")
                continue
                
            np.add.at(this_map, neighbors, m * w / w_sum)

        final_map += hp.ud_grade(this_map, nside_out=nside_out, power=-2)
    return final_map.astype(np.float16)

def get_particles_uv_full(nside, pixels, shell_r, h):
    npix = hp.nside2npix(nside)
    particle_pixels = np.where(pixels > 0)[0]
    p_list = []
    
    halo_pos_mpc = np.vstack((h['x'], h['y'], h['z'])).T
    halo_norms = np.linalg.norm(halo_pos_mpc, axis=1)
    halo_vectors = halo_pos_mpc / halo_norms[:, None]
    
    rvir_mpc = h['rvir']
    halo_theta_vir = np.arctan(rvir_mpc / halo_norms)
    
    halo_influence_map = np.zeros(npix, dtype=bool)
    adjacent_halos_dict = defaultdict(list)
    
    for i, (vec, theta_vir) in enumerate(zip(halo_vectors, halo_theta_vir)):
        if theta_vir <= 0:
            continue
        pix_list = hp.query_disc(nside, vec, theta_vir, nest=False, inclusive=False)
        for pix in pix_list:
            adjacent_halos_dict[pix].append(i)
            halo_influence_map[pix] = True
    
    halo_map = np.zeros(npix, dtype=bool)
    halo_pixels_dict = defaultdict(list)
    for i, vec in enumerate(halo_vectors):
        pix_center = hp.vec2pix(nside, vec[0], vec[1], vec[2], nest=False)
        halo_pixels_dict[pix_center].append(i)
        halo_map[pix_center] = True
    
    neighbor_map = halo_influence_map & ~halo_map

    def assign_weight(dists, rvir_mpc):
        x = dists / rvir_mpc
        weight = np.zeros_like(x)
        
        mask1 = x < 1
        if np.any(mask1):
            sqrt1 = np.sqrt((1 - x[mask1]) / (1 + x[mask1]))
            weight[mask1] = (1 - (2 / np.sqrt(1 - x[mask1]**2)) * np.arctanh(sqrt1)) / (x[mask1]**2 - 1)
        
        mask2 = np.isclose(x, 1, atol=1e-6)
        if np.any(mask2):
            weight[mask2] = 1.0 / 3.0
        
        mask3 = x > 1
        if np.any(mask3):
            sqrt2 = np.sqrt((x[mask3] - 1) / (1 + x[mask3]))
            weight[mask3] = (1 - (2 / np.sqrt(x[mask3]**2 - 1)) * np.arctan(sqrt2)) / (x[mask3]**2 - 1)
        
        return weight

    for pix in particle_pixels:
        pix_mass = pixels[pix]
        
        if halo_map[pix]:
            # Case A: Exact halo pixel
            halos_in_pixel = halo_pixels_dict.get(pix, [])
            sub_nside = 4 * nside
            grandchildren = get_child_pixels(nside, pix, sub_nside)
            dirs = np.array(hp.pix2vec(sub_nside, grandchildren, nest=False)).T
            sub_positions = dirs * shell_r
            
            mass_weights = np.ones(16)
            for halo_idx in halos_in_pixel:
                dists = np.linalg.norm(sub_positions - halo_pos_mpc[halo_idx], axis=1)
                mass_weights += assign_weight(dists, rvir_mpc[halo_idx])
            if halos_in_pixel:
                mass_weights /= np.sum(mass_weights)
            
            for j in range(16):
                mass = pix_mass * mass_weights[j]
                p_list.append((sub_positions[j], mass, 2))
                
        elif neighbor_map[pix]:
            # Case B: Pixel within virial radius
            covering_halos = adjacent_halos_dict.get(pix, [])
            sub_nside = 2 * nside
            children = get_child_pixels(nside, pix, sub_nside)
            dirs = np.array(hp.pix2vec(sub_nside, children, nest=False)).T
            sub_positions = dirs * shell_r
            
            mass_weights = np.ones(4)
            for halo_idx in covering_halos:
                dists = np.linalg.norm(sub_positions - halo_pos_mpc[halo_idx], axis=1)
                mass_weights += assign_weight(dists, rvir_mpc[halo_idx])
            if covering_halos:
                mass_weights /= np.sum(mass_weights)
            
            for j in range(4):
                mass = pix_mass * mass_weights[j]
                p_list.append((sub_positions[j], mass, 1))
                
        else:
            # Case C: Regular pixel
            dirs = np.array(hp.pix2vec(nside, pix, nest=False))
            pos = dirs * shell_r
            p_list.append((pos, pix_mass, 0))
            
    return p_list
