import os
import healpy as hp
import numpy as np
from time import time
# Force tqdm to use threading-based locks (no semaphores)
from tqdm import tqdm
import schwimmbad
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import splrep, splev
from collections import defaultdict
import gc
from cosmic_toolbox import logger
from .params import *

LOGGER = logger.get_logger(__name__)

def arcdistance(distance,radius,param):
    '''Correct distance calculation for low redshift shells'''
    if radius > param.shell.curv_radius:
        return distance
    else:
        angle = 2*np.arcsin(distance/(2*radius))
        return radius*angle

def euclidean_distance(arcdistance,radius,param):
    '''Correct distance calculation for low redshift shells'''
    if radius > param.shell.curv_radius:
        return arcdistance
    else:
        angle = arcdistance/radius
        return 2*radius*np.sin(angle/2)

def arcdisplace(displace,position,radius,param):
    '''Correct displacement calculation for low redshift shells'''
    if radius > param.shell.curv_radius:
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


def loop_cpus_subsample_particles(pid, pix_subset, pixels, nside, shell_r, halo_map, halo_pixels_dict,
              neighbor_map, adjacent_halos_dict, h, output_dir):

    t0 = time()
    local_list = []

    n_pix_subset = len(pix_subset)
    for pix in maybe_progressbar(pix_subset ,total = n_pix_subset, desc = f"Process {pid}: Loop over pixel subset"):
        pix_mass = pixels[pix]

        if halo_map[pix]:
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
                    # assign_weight must be visible at module level; reuse existing function
                    rvir = h['rvir'][halo_idx]
                    mass_weights += assign_weight(dists, rvir)
                mass_weights /= np.sum(mass_weights)

            for j in range(16):
                mass = pix_mass * mass_weights[j]
                local_list.append((pix, sub_positions[j], mass, 2))

        elif neighbor_map[pix]:
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
                    mass_weights += assign_weight(dists, h['rvir'][halo_idx])
                mass_weights /= np.sum(mass_weights)

            for j in range(4):
                mass = pix_mass * mass_weights[j]
                local_list.append((pix, sub_positions[j], mass, 1))

        else:
            dirs = np.array(hp.pix2vec(nside, pix, nest=False))
            pos = dirs * shell_r
            local_list.append((pix, pos, pix_mass, 0))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"particles_local_{pid}.npy")
    arr = np.zeros(len(local_list), dtype=[('pix', 'i4'), ('pos', '3f4'), ('mass', 'f4'), ('tag', 'i4')])
    for i, (pix, pos, mass, tag) in enumerate(local_list):
        arr[i]['pix'] = pix
        arr[i]['pos'] = pos
        arr[i]['mass'] = mass
        arr[i]['tag'] = tag
    np.save(filename, arr)

    del local_list
    del arr
    gc.collect()

    LOGGER.debug(f"......Process {pid} looped over {len(pix_subset)} pixels. Ellapsed time: {time()-t0:.2f}s.")
    return pid, filename


def assign_weight(dists, rvir):
    """
    Projected-NFW-like weight used for subpixel mass splitting.
    dists : array-like distances (Mpc)
    rvir  : scalar or array-like virial radii matching dists
    """
    x = dists / rvir
    weight = np.zeros_like(x)

    mask1 = x < 1
    if np.any(mask1):
        sqrt1 = np.sqrt((1 - x[mask1]) / (1 + x[mask1]))
        weight[mask1] = (1 - (2 / np.sqrt(1 - x[mask1]**2)) * np.arctanh(sqrt1)) / (x[mask1]**2 - 1)

    mask2 = np.isclose(x, 1)
    if np.any(mask2):
        weight[mask2] = 1.0 / 3.0

    mask3 = x > 1
    if np.any(mask3):
        sqrt2 = np.sqrt((x[mask3] - 1) / (1 + x[mask3]))
        weight[mask3] = (1 - (2 / np.sqrt(x[mask3]**2 - 1)) * np.arctan(sqrt2)) / (x[mask3]**2 - 1)

    return weight


def subsample_pixels(nside, pixels, shell_r, halos, param):
    """
    Subsampling pixels around halo centres to improve resolution.
    """
    npix = hp.nside2npix(nside)
    halo_pixels_dict = defaultdict(list)
    neighbor_map = np.zeros(npix, dtype=bool)
    h = halos[halos['IDhost']==-1]
    
    # Process each halo to mark its pixel and neighbors
    t = time()
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
    LOGGER.info(f"......Pre-processing of the subsampling done. Ellapsed time: {time() - t:.3f} seconds.")
    
    """
    parallelization
    """
    t = time()
    p_list = []

    
    t_global = time()

    output_dir = param.files.tmp_files

    # ---- prepare arguments for MultiPool ----
    nproc = param.shell.N_cpu
    iterable_args = [
        (p,
        particle_pixels[p::nproc],
        pixels,
        nside,
        shell_r,
        halo_map,
        halo_pixels_dict,
        neighbor_map,
        adjacent_halos_dict,
        h,
        output_dir)
        for p in range(nproc)
    ]

    LOGGER.info(f"......Launching subsampling with {nproc} processes...")

    # ---- Run in parallel ----
    with schwimmbad.MultiPool(processes=nproc) as pool:
        results = list(pool.starmap(loop_cpus_subsample_particles, iterable_args))

    LOGGER.info(f"......Subsampling with {nproc} processes done. Ellapsed time {time()-t_global:.2f} seconds")

    # results = [(pid, filename), ...]
    results = sorted(results, key=lambda x: x[0])
    file_list = [fn for (_, fn) in results]

    # Load data
    all_data = [np.load(fn) for fn in file_list]
    p_all = np.concatenate(all_data)

    # remove temp files
    for fn in file_list:
        os.remove(fn)

    keep_fields = ['pos', 'mass', 'tag']
    p_all_filtered = p_all[keep_fields]

    p_list = p_all_filtered

    return p_list



def get_child_pixels(parent_nside, pix, child_nside):
    """Get child pixels for hierarchical pixel subdivision"""
    order_diff = int(np.log2(child_nside / parent_nside))
    idx_n = hp.ring2nest(parent_nside, pix)
    return hp.nest2ring(child_nside, idx_n * (4**order_diff) + np.arange(4**order_diff))


def particle_worker(task):
    '''
    mesh_ref == 0: No any subgrid sampling
    mesh_ref == 1: Subgrid sampling following projected NFW profile.
    '''
    i, pixels, h, param = task
    mesh_ref = param.shell.mesh_ref
    nside = param.shell.nside
    
    LOGGER.info(f"......Subsampling with mesh_ref={mesh_ref}")
    
    shell_cov = np.sqrt(h['x'][0]**2 + h['y'][0]**2 + h['z'][0]**2)
    p_dt = np.dtype([("x", '>f8'), ("y", '>f8'), ("z", '>f8'), ("M", '>f4'), ('ref_order', np.uint8)])

    if mesh_ref == 0:
        idx = np.where(pixels > 0)[0]
        px, py, pz = hp.pix2vec(nside, idx)
        px *= shell_cov; py *= shell_cov; pz *= shell_cov
        p = np.zeros(len(px), dtype=p_dt)
        p['x'], p['y'], p['z'] = px, py, pz
        p['M'] = pixels[idx]
        p['ref_order'] = 0

    elif mesh_ref == 1:
        sub = subsample_pixels(nside, pixels, shell_cov, h, param)
        p = np.zeros(len(sub), dtype=p_dt)
        p[:]['x'], p[:]['y'], p[:]['z'] = sub['pos'][:,0], sub['pos'][:,1], sub['pos'][:,2]
        p[:]['M'] = sub['mass']
        p[:]['ref_order'] = sub['tag']
    else:
        raise ValueError(f"Unsupported mesh_ref: {mesh_ref}")
    return i, p


def get_healpix_map(p, param, star_fraction=None):
    nside      = param.shell.nside
    nside_out  = param.shell.nside_out
    interp     = param.shell.interp
    x = p['x']; y = p['y']; z = p['z']
    masses = p['M'] * (1 - star_fraction) if star_fraction is not None else p['M']
    print("test::::::",np.min(masses),np.max(masses))#,np.min((1 - star_fraction)),np.max((1 - star_fraction)))
    
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

    nside     = param.shell.nside
    nside_out = param.shell.nside_out

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

def maybe_progressbar(iterable, total, desc):
    level = os.getenv("PYTHON_LOGGER_LEVEL", "info")
    if level == "debug":
        return LOGGER.progressbar(iterable, total=total, desc=desc, at_level="debug")
    return iterable