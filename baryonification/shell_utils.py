import os, sys
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
from .profiles import Profiles

from mpi4py import MPI
import pickle as pkl
import psutil

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


def loop_cpus_subsample_particles(pid, pix_subset, nside, shell_r, halo_pixels_dict, adjacent_halos_dict, h, output_dir):

    # read in large healpix arrays - no need to send them via MPI
    pixels = np.load(os.path.join(output_dir,"pixels.npy"), mmap_mode="r")
    halo_map = np.load(os.path.join(output_dir,"halo_map.npy"), mmap_mode="r")
    neighbor_map = np.load(os.path.join(output_dir,"neighbor_map.npy"), mmap_mode="r")

    t0 = time()
    local_list = []
    n_pix_subset = len(pix_subset)
    for pix in maybe_progressbar(pix_subset ,total = n_pix_subset, desc = f"Process {pid}: Loop over pixel subset"):
        pix_mass = pixels[pix]

        if halo_map[pix]:
            halos_in_pixel = halo_pixels_dict.get(pix, [])
            Ngrandchildren_per_dim = 4
            sub_nside = Ngrandchildren_per_dim * nside
            idx_n = hp.ring2nest(nside, pix)
            grandchildren = hp.nest2ring(sub_nside, idx_n * Ngrandchildren_per_dim**2 + np.arange(Ngrandchildren_per_dim**2))
            dirs = np.array(hp.pix2vec(sub_nside, grandchildren, nest=False)).T
            sub_positions = dirs * shell_r

            mass_weights = np.ones(Ngrandchildren_per_dim**2)
            if halos_in_pixel:
                for halo_idx in halos_in_pixel:
                    halo_pos = np.array([h['x'][halo_idx], h['y'][halo_idx], h['z'][halo_idx]])
                    dists = np.linalg.norm(sub_positions - halo_pos, axis=1)
                    # assign_weight must be visible at module level; reuse existing function
                    rvir = h['rvir'][halo_idx]
                    mass_weights += assign_weight(dists, rvir)
                mass_weights /= np.sum(mass_weights)

            for j in range(Ngrandchildren_per_dim**2):
                mass = pix_mass * mass_weights[j]
                local_list.append((pix, sub_positions[j], mass, 2))

        elif neighbor_map[pix]:
            adjacent_halos = adjacent_halos_dict.get(pix, [])
            Nchildren_per_dim = 2
            sub_nside = Nchildren_per_dim * nside
            idx_n = hp.ring2nest(nside, pix)
            children = hp.nest2ring(sub_nside, idx_n * Nchildren_per_dim**2 + np.arange(Nchildren_per_dim**2))
            dirs = np.array(hp.pix2vec(sub_nside, children, nest=False)).T
            sub_positions = dirs * shell_r

            mass_weights = np.ones(Nchildren_per_dim**2)
            if adjacent_halos:
                for halo_idx in adjacent_halos:
                    halo_pos = np.array([h['x'][halo_idx], h['y'][halo_idx], h['z'][halo_idx]])
                    dists = np.linalg.norm(sub_positions - halo_pos, axis=1)
                    mass_weights += assign_weight(dists, h['rvir'][halo_idx])
                mass_weights /= np.sum(mass_weights)

            for j in range(Nchildren_per_dim**2):
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

def unpacked_loop(args):
    return loop_cpus_subsample_particles(*args)

def subsample_pixels(nside, pixels, shell_r, halos, param, pool):
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir,"pixels.npy"), pixels)
    np.save(os.path.join(output_dir,"halo_map.npy"), halo_map)
    np.save(os.path.join(output_dir,"neighbor_map.npy"), neighbor_map)

    # ---- prepare arguments for MultiPool ----
    nproc = param.shell.N_cpu
    iterable_args = [
        (p,
        particle_pixels[p::nproc],
        nside,
        shell_r,
        halo_pixels_dict,
        adjacent_halos_dict,
        h,
        output_dir)
        for p in range(nproc)
    ]
 
    LOGGER.info(f"......Launching subsampling with {nproc} processes...")

    # ---- Run in parallel ----
    # with schwimmbad.MultiPool(processes=nproc) as pool:
        # results = list(pool.starmap(loop_cpus_subsample_particles, iterable_args))
    results = list(pool.map(unpacked_loop, iterable_args))

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


def particle_worker(task,pool):
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
        sub = subsample_pixels(nside, pixels, shell_cov, h, param, pool)
        p = np.zeros(len(sub), dtype=p_dt)
        p[:]['x'], p[:]['y'], p[:]['z'] = sub['pos'][:,0], sub['pos'][:,1], sub['pos'][:,2]
        p[:]['M'] = sub['mass']
        p[:]['ref_order'] = sub['tag']
    else:
        raise ValueError(f"Unsupported mesh_ref: {mesh_ref}")
    return i, p


def get_healpix_map(p, param, star_fraction=None,pool=None):
    nside      = param.shell.nside
    nside_out  = param.shell.nside_out
    interp     = param.shell.interp
    x = p['x']; y = p['y']; z = p['z']
    masses = p['M'] * (1 - star_fraction) if star_fraction is not None else p['M']
    
    if not interp:
        # direct binning at final resolution: splitting order irrelevant
        r = np.linalg.norm(np.stack((x, y, z), axis=0), axis=0)
        pix_out = hp.vec2pix(nside_out,x/r, y/r, z/r)
        healpix_map = np.zeros(hp.nside2npix(nside_out))
        np.add.at(healpix_map, pix_out, masses)
        return healpix_map

    # LOGGER.debug("......Using interpolation-based HEALPix mapping.")
    # max_order = int(p['ref_order'].max())
    # final_map = np.zeros(hp.nside2npix(nside_out))
    # r = np.linalg.norm(np.stack((x, y, z), axis=0), axis=0)
    # theta = np.arccos(z / r)
    # phi   = np.mod(np.arctan2(y, x), 2*np.pi)

    # LOGGER.debug(f"......Max refinement order: {max_order}")
    # for order in range(max_order + 1):
    #     LOGGER.debug(f"......Processing refinement order {order}...")
    #     mask = (p['ref_order'] == order)
    #     this_nside = nside * (2 ** order)
    #     this_map   = np.zeros(hp.nside2npix(this_nside))
    #     this_mass = masses[mask]
    #     th        = theta[mask]
    #     ph        = phi[mask]
        
    #     pix_ids, weights = hp.get_interp_weights(this_nside, th, ph)
    #     weights = weights.T
    #     pix_ids = pix_ids.T

    #     flat_pix = pix_ids.ravel()
    #     flat_m   = (this_mass[:,None] * weights).ravel()
    #     np.add.at(this_map, flat_pix, flat_m)

    #     # downgrade and accumulate
    #     LOGGER.debug(f"......Processing refinement order {order}...downgrade and accumulate...")
    #     if hp.get_nside(this_map) == nside_out:
    #         final_map += this_map
    #         LOGGER.debug(f"......nside match, performing a simple addition")
    #     else:
    #         final_map += hp.ud_grade(this_map, nside_out=nside_out, power=-2)

    #     # final_map += hp.ud_grade(this_map,nside_out=nside_out,power=-2)

    # return final_map.astype(np.float16)
    LOGGER.debug("......Using interpolation-based HEALPix mapping.")
    max_order = int(p['ref_order'].max())
    final_map = np.zeros(hp.nside2npix(nside_out))
    r = np.linalg.norm(np.stack((x, y, z), axis=0), axis=0)
    theta = np.arccos(z / r)
    phi   = np.mod(np.arctan2(y, x), 2*np.pi)

    # LOGGER.debug(f"......Max refinement order: {max_order}")
    # for order in range(max_order + 1):
    #     mask = (p['ref_order'] == order)
    #     if not np.any(mask):
    #         continue

    #     this_nside = nside * (2 ** order)
    #     k = int(np.log2(this_nside // nside_out))

    #     th = theta[mask]
    #     ph = phi[mask]
    #     this_mass = masses[mask]

    #     pix_ids, weights = hp.get_interp_weights(
    #         this_nside, th, ph, nest=True
    #     )
    #     pix_ids = pix_ids.T
    #     weights = weights.T

    #     flat_child = pix_ids.ravel()
    #     flat_mass  = (this_mass[:, None] * weights).ravel()

    #     # DOWNGRADE ON THE FLY
    #     flat_parent = flat_child >> (2 * k)

    #     np.add.at(final_map, flat_parent, flat_mass)

    # return final_map.astype(np.float16)
    
    import itertools
    

    N = len(theta)
    n_workers = pool.size  # number of MPI ranks

    # Divide particles evenly among workers
    chunk_size = int(np.ceil(N / n_workers))
    
    chunks = [
        (i, min(i + chunk_size, N))
        for i in range(0, N, chunk_size)
    ]

    # store arrays such that I do not need to pass them as arguments
    LOGGER.debug(f"Storing tmp files for parallel pixelization...")
    output_dir = param.files.tmp_files
    np.save(os.path.join(output_dir,"theta.npy"), theta)
    np.save(os.path.join(output_dir,"phi.npy"), phi)
    np.save(os.path.join(output_dir,"masses.npy"), masses)
    np.save(os.path.join(output_dir,"ref_order.npy"), p['ref_order'])


    tasks = [
    (idx_start, idx_end, nside, nside_out, output_dir)
    for idx_start, idx_end in chunks
    ]

    results = pool.map(
        process_particle_chunk, tasks
    ) # returns list of filenames of the local maps saved by each worker

    # Reduce on master
    final_map = np.zeros(hp.nside2npix(nside_out), dtype=np.float32)

    LOGGER.debug(f"summing up local maps from {n_workers} workers...")
    for fname in tqdm(results):
        m = np.load(fname, mmap_mode='r')
        final_map += m
        del m

    LOGGER.debug(f"Cleaning maps...")
    for fname in results:
        os.remove(fname)

    LOGGER.debug(f"Cleaning tmp files...")
    os.remove(os.path.join(output_dir,"theta.npy"))
    os.remove(os.path.join(output_dir,"phi.npy"))
    os.remove(os.path.join(output_dir,"masses.npy"))
    os.remove(os.path.join(output_dir,"ref_order.npy"))
    
    return final_map.astype(np.float16)

def process_particle_chunk(args):
    try:
        
        # memory profiling
        rank = MPI.COMM_WORLD.Get_rank()
        
        idx_start, idx_end, nside, nside_out, output_dir = args
        # LOGGER.debug(f"[process_particle_chunk] pixelizing particles from {idx_start} to {idx_end}")

        npix_out = hp.nside2npix(nside_out)
        local_map = np.zeros(npix_out, dtype=np.float32)

        # Slice particles
        theta = np.load(os.path.join(output_dir, "theta.npy"), mmap_mode="r")
        phi = np.load(os.path.join(output_dir, "phi.npy"), mmap_mode="r")
        masses = np.load(os.path.join(output_dir, "masses.npy"), mmap_mode="r")
        pref = np.load(os.path.join(output_dir, "ref_order.npy"), mmap_mode="r")
        ref_order = pref[idx_start:idx_end]
        th_all = theta[idx_start:idx_end]
        ph_all = phi[idx_start:idx_end]
        mass_all = masses[idx_start:idx_end]

        # LOGGER.debug(f"[process_particle_chunk {idx_start}-{idx_end}] loaded data")

        # Loop only over orders PRESENT in this chunk
        for order in np.unique(ref_order):
            # LOGGER.debug(f"[process_particle_chunk {idx_start}-{idx_end}] ref {order}")
            mask = (ref_order == order)
            if not np.any(mask):
                continue
            
            # LOGGER.debug(f"rank {rank}, stamp 1")

            this_nside = int(nside) * (2 ** int(order))
            k = int(np.log2(this_nside // nside_out))

            th = th_all[mask]
            ph = ph_all[mask]
            this_mass = mass_all[mask]

            # LOGGER.debug(f"rank {rank}, stamp 2")

            pix_ids, weights = hp.get_interp_weights(
                this_nside, th, ph, nest=True
            )

            # LOGGER.debug(f"rank {rank}, stamp 3")

            pix_ids = pix_ids.T
            weights = weights.T

            # LOGGER.debug(f"rank {rank}, stamp 4")

            flat_child = pix_ids.ravel()
            flat_mass = (this_mass[:, None] * weights).ravel()

            # LOGGER.debug(f"rank {rank}, stamp 5")

            # downgrade-on-the-fly
            flat_parent = flat_child >> (2 * k)

            np.add.at(local_map, flat_parent, flat_mass)

            # LOGGER.debug(f"rank {rank}, stamp 6")

            # LOGGER.debug(f"[process_particle_chunk {idx_start}-{idx_end}] ref {order} done")
            del mask, th, ph, this_mass, pix_ids, weights, flat_child, flat_mass, flat_parent
            gc.collect()
            
            proc = psutil.Process(os.getpid())
            rss_mb = proc.memory_info().rss / 1024**2

            vmem = psutil.virtual_memory()
            total_mb = vmem.total / 1024**2

            # LOGGER.debug(f"rank {rank}: {rss_mb:.1f} MB / {total_mb:.0f} MB available")

            # LOGGER.debug(f"rank {rank}, stamp 7")

        fname = os.path.join(output_dir, f"local_map_{rank}.npy")
        np.save(fname, local_map.astype(np.float32))
        del local_map
        gc.collect()

        return fname
    except:
        rank = MPI.COMM_WORLD.Get_rank()
        print(f"\n❌ Rank {rank} crashed in process_particle_chunk:", file=sys.stderr)
        traceback.print_exc()
        sys.stderr.flush()
        raise



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

'''
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
'''

def sphere_intersection_volume(r1, r2, d):

    #deal with overlaps
    h1 = (r1 - (d**2 - r2**2 + r1**2) / (2.0 * d))
    h2 = (r2 - (d**2 - r1**2 + r2**2) / (2.0 * d))
    Vcap1 = (1.0/3.0) * np.pi * h1**2 * (3*r1 - h1)
    Vcap2 = (1.0/3.0) * np.pi * h2**2 * (3*r2 - h2)

    Vexcl = Vcap1 + Vcap2

    #deal with case where there is no overlap
    Vexcl[d>=(r1 + r2)] = 0
    Vexcl[d<=abs(r1 - r2)] = (4.0/3.0) * np.pi * r2[d<=abs(r1 - r2)]**3

    return Vexcl


def impact_factor(rbin, h_cov, shell_cov, thickness):

    r_in = shell_cov - thickness / 2
    r_out = shell_cov + thickness / 2
    r_in = max(r_in, 0.0)
    
    #V_halo = (4/3) * np.pi * rball**3
    #V_halo = (4/3) * np.pi * rbin[-1]**3
    V_halo = (4/3) * np.pi * rbin**3

    #array of r between 0 and rball 
    V_outer = sphere_intersection_volume(r_out, rbin, h_cov)
    V_inner = sphere_intersection_volume(r_in, rbin, h_cov)
    V_overlap = V_outer - V_inner
    
    #V_overlap = max(0.0, V_overlap)
    V_overlap[V_overlap<0] = 0 

    impact_fac = V_overlap / V_halo
    #impact_fac[impact_fac<0] = 0

    return impact_fac


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

def loop_halo_chunks_worker(i_cpu, idx_local, task, args_for_loop_halo_chunks, output_dir):
    """
    Top-level function version of the original class method 'loop_halo_chunks'
    Args:
        i_cpu : int
            CPU index for logging / file naming
        idx_local : array-like
            Subset of halo indices for this CPU
        task : tuple
            (shell_id, h, thickness, p, redshift, param)
        args_for_loop_halo_chunks : tuple
            Precomputed data (shell_cov, var_tck, bias_tck, corr_tck, p_tree, etc.)
        param : object
            Parameter object containing code, shell, files, etc.
        h : halo data (dict or structured array)
        p_tree : spatial tree for particle queries
        p_darkmatter, p_baryons : particle data
        output_dir : str
            Directory to save temporary output files
    Returns:
        tuple of filenames: (DpBAR_file, DrpFDM_file) or similar
    """

    ts = time()
    LOGGER.debug(f'......process {i_cpu} starting with {len(idx_local)} halos...')

    shell_id, thickness, redshift, param = task
    shell_cov, var_tck, bias_tck, corr_tck = args_for_loop_halo_chunks
    
    LOGGER.debug(f'......loading precomputed data: p...')
    p = np.load(os.path.join(output_dir,"p.npy"), mmap_mode='r')  # load p to a file to avoid pickling issues
    LOGGER.debug(f'......loading precomputed data: h...')
    h = np.load(os.path.join(output_dir,"h.npy"), mmap_mode='r')  # load p to a file to avoid pickling issues
    LOGGER.debug(f'......loading precomputed data: p_tree...')
    with open(os.path.join(output_dir,"p_tree.pkl"), "rb") as f:
        p_tree = pkl.load(f)
    # sys.exit()
    profiles = Profiles(None, 1e13, None, None, None, None, param)

    # Multiple names for the same memory-mapped array
    # we do not modify p here, jsut read out the coordinates - so we do not need to copy
    p_darkmatter = p
    p_baryons = p if param.code.multicomp else None
    n_p = len(p)
    LOGGER.debug(f'......process {i_cpu} loaded all precomputed data.')

    if param.code.multicomp:
        # Setup arrays for baryons and DM
        Dp_type = np.dtype([("x",'>f'),("y",'>f'),("z",'>f'),
                            ("id",'>f4'),("rho2D_star_at_xyz",'>f4'),("rho2D_bar_at_xyz",'>f4')])
        DpBAR = np.zeros(n_p, dtype=Dp_type)
        DpFDM = np.zeros(n_p, dtype=Dp_type)

        for j in maybe_progressbar(idx_local, total=len(idx_local), desc=f"Process {i_cpu}: Loop over halo subset"):
            if h['IDhost'][j] < 0:
                # Extract halo data
                hx, hy, hz = h['x'][j], h['y'][j], h['z'][j]
                h_cov = h['cov'][j]
                Mvir, rvir, cvir = h['Mvir'][j], h['rvir'][j], h['cvir'][j]

                # Compute bins
                rmin = max(0.001*rvir, param.code.rmin)
                rmax = min(20.0*rvir, param.code.rmax)
                rbin = np.logspace(np.log10(rmin), np.log10(rmax), 100)

                # Load 3D profiles
                cosmo_var  = splev(Mvir, var_tck)
                cosmo_bias = splev(Mvir, bias_tck)
                cosmo_corr = splev(rbin, corr_tck)
                profiles._update_params({'rbin': rbin, 'Mvir': Mvir, 'cvir': cvir,
                                         'cosmo_corr': cosmo_corr, 'cosmo_bias': cosmo_bias, 'cosmo_var': cosmo_var})
                frac, dens, mass, press, temp = profiles.calc_profiles()

                # Projected densities
                rhoBAR_i = (1-frac['CDM'])*(dens['NFW'] + dens['BG'])
                rhoBAR_f = frac['HGA']*dens['HGA'] + frac['IGA']*dens['IGA'] + frac['CGA']*dens['CGA'] + frac['SGA']*dens['SGA'] + (1-frac['CDM'])*dens['BG']
                rhoDM_i = frac['CDM']*(dens['NFW'] + dens['BG'])
                rhoDM_f = frac['CDM']*(dens['CDM'] + dens['BG'])

                projected_MDM_i = projection(rhoDM_i, rbin, rvir, thickness, param, output='mass')
                projected_MDM_f = projection(rhoDM_f, rbin, rvir, thickness, param, output='mass')
                projected_MBAR_i = projection(rhoBAR_i, rbin, rvir, thickness, param, output='mass')
                projected_MBAR_f = projection(rhoBAR_f, rbin, rvir, thickness, param, output='mass')

                # Displacement functions
                DBAR = displ(rbin, projected_MBAR_i, projected_MBAR_f)
                DFDM = displ(rbin, projected_MDM_i, projected_MDM_f)

                # Compute impact factors
                V_overlap_ov_tot = impact_factor(rbin, h_cov, shell_cov, thickness)

                # Apply corrections
                rhoCDM = dens['CDM']
                rhoBAR = frac['HGA']*dens['HGA'] + frac['IGA']*dens['IGA'] + frac['CGA']*dens['CGA'] + frac['SGA']*dens['SGA']
                corrFDM = np.trapz(rbin**2 * V_overlap_ov_tot * rhoCDM, rbin)/np.trapz(rbin**2 * rhoCDM, rbin)
                corrBAR = np.trapz(rbin**2 * V_overlap_ov_tot * rhoBAR, rbin)/np.trapz(rbin**2 * rhoBAR, rbin)
                DBAR *= corrBAR
                DFDM *= corrFDM

                # print(DBAR, DFDM,imf)   
                DBAR_tck = splrep(rbin, DBAR,s=0,k=3)
                DFDM_tck = splrep(rbin, DFDM,s=0,k=3)
                    
                smallestD = param.code.disp_trunc #Mpc/h
                # print(DBAR, DFDM, smallestD)   
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
                # print('rball before arc = ', rball)
                rball = euclidean_distance(rball,shell_cov,param)
            
                #particle ids within rball
                ipbool = np.array(p_tree.query_ball_point((hx,hy,hz),rball))
                # print("Halo centre, surrounding particle number = ", hx,hy,hz, len(ipbool))

                if (len(ipbool) > 0):
                    #calculating radii of FDM particles around halo j
                    rpFDM  = ((p_darkmatter['x'][ipbool]-hx)**2.0 +
                            (p_darkmatter['y'][ipbool]-hy)**2.0 +
                            (p_darkmatter['z'][ipbool]-hz)**2.0)**0.5
                    rpFDM = arcdistance(rpFDM,shell_cov,param)

                    #calculating radii of BAR particles around halo j
                    rpBAR = ((p_baryons['x'][ipbool]-hx)**2.0 +
                            (p_baryons['y'][ipbool]-hy)**2.0 +
                            (p_baryons['z'][ipbool]-hz)**2.0)**0.5
                    rpBAR = arcdistance(rpBAR,shell_cov,param)

                    if param.shell.nbrhalo == 1:
                        
                        iphalo = np.zeros(len(p))
                        multi_halo = np.zeros(len(p), dtype=bool)
                        for i in range(len(h['Mvir'])):
                            ip   = np.array(p_tree.query_ball_point((h['x'][i], h['y'][i], h['z'][i]), h['rvir'][i]))
                            if len(ip)>0:
                                previously = iphalo[ip]
                                collided  = (previously != 0) & (previously != i)
                                multi_halo[ip[collided]] = True
                                iphalo[ip] = i
                
                        
                        mask_out = (rpBAR > rvir) & (iphalo[ipbool] > 0)    
                        mask = mask_out | multi_halo[ipbool]
                        ipbool_nbrhaloes    = ipbool[mask]
                        ipbool_wo_nbrhaloes = ipbool[~mask]

                        #calculating radii of BAR particles around halo j
                        rpBAR_nbrhaloes = ((p_baryons['x'][ipbool_nbrhaloes]-hx)**2.0 +
                                    (p_baryons['y'][ipbool_nbrhaloes]-hy)**2.0 +
                                    (p_baryons['z'][ipbool_nbrhaloes]-hz)**2.0)**0.5
                        rpBAR_nbrhaloes = arcdistance(rpBAR_nbrhaloes,shell_cov,param)
                        rpBAR_wo_nbrhaloes = ((p_baryons['x'][ipbool_wo_nbrhaloes]-hx)**2.0 +
                                    (p_baryons['y'][ipbool_wo_nbrhaloes]-hy)**2.0 +
                                    (p_baryons['z'][ipbool_wo_nbrhaloes]-hz)**2.0)**0.5
                        rpBAR_wo_nbrhaloes = arcdistance(rpBAR_wo_nbrhaloes,shell_cov,param)

                        DrpFDM = splev(rpFDM,DFDM_tck,der=0,ext=1)
                        DpFDM['x'][ipbool] += (p_darkmatter['x'][ipbool]-hx)*DrpFDM/rpFDM
                        DpFDM['y'][ipbool] += (p_darkmatter['y'][ipbool]-hy)*DrpFDM/rpFDM
                        DpFDM['z'][ipbool] += (p_darkmatter['z'][ipbool]-hz)*DrpFDM/rpFDM

                        if(len(rpBAR_nbrhaloes)>0):
                            DrpBAR_nbrhaloes    = splev(rpBAR_nbrhaloes,DFDM_tck,der=0,ext=1)
                            DrpBAR_wo_nbrhaloes = splev(rpBAR_wo_nbrhaloes,DBAR_tck,der=0,ext=1)
                            DpBAR['x'][ipbool_nbrhaloes] += (p_baryons['x'][ipbool_nbrhaloes]-hx)*DrpBAR_nbrhaloes/rpBAR_nbrhaloes
                            DpBAR['y'][ipbool_nbrhaloes] += (p_baryons['y'][ipbool_nbrhaloes]-hy)*DrpBAR_nbrhaloes/rpBAR_nbrhaloes
                            DpBAR['z'][ipbool_nbrhaloes] += (p_baryons['z'][ipbool_nbrhaloes]-hz)*DrpBAR_nbrhaloes/rpBAR_nbrhaloes
                            DpBAR['x'][ipbool_wo_nbrhaloes] += (p_baryons['x'][ipbool_wo_nbrhaloes]-hx)*DrpBAR_wo_nbrhaloes/rpBAR_wo_nbrhaloes
                            DpBAR['y'][ipbool_wo_nbrhaloes] += (p_baryons['y'][ipbool_wo_nbrhaloes]-hy)*DrpBAR_wo_nbrhaloes/rpBAR_wo_nbrhaloes
                            DpBAR['z'][ipbool_wo_nbrhaloes] += (p_baryons['z'][ipbool_wo_nbrhaloes]-hz)*DrpBAR_wo_nbrhaloes/rpBAR_wo_nbrhaloes
                        else:
                            DrpBAR = splev(rpBAR,DBAR_tck,der=0,ext=1)
                            DpBAR['x'][ipbool] += (p_baryons['x'][ipbool]-hx)*DrpBAR/rpBAR
                            DpBAR['y'][ipbool] += (p_baryons['y'][ipbool]-hy)*DrpBAR/rpBAR
                            DpBAR['z'][ipbool] += (p_baryons['z'][ipbool]-hz)*DrpBAR/rpBAR
                            
                    elif param.shell.nbrhalo == 0:
                        
                        DrpBAR = splev(rpBAR,DBAR_tck,der=0,ext=1)
                        DpBAR['x'][ipbool] += (p_baryons['x'][ipbool]-hx)*DrpBAR/rpBAR
                        DpBAR['y'][ipbool] += (p_baryons['y'][ipbool]-hy)*DrpBAR/rpBAR
                        DpBAR['z'][ipbool] += (p_baryons['z'][ipbool]-hz)*DrpBAR/rpBAR

                        DrpFDM = splev(rpFDM,DFDM_tck,der=0,ext=1)
                        DpFDM['x'][ipbool] += (p_darkmatter['x'][ipbool]-hx)*DrpFDM/rpFDM
                        DpFDM['y'][ipbool] += (p_darkmatter['y'][ipbool]-hy)*DrpFDM/rpFDM
                        DpFDM['z'][ipbool] += (p_darkmatter['z'][ipbool]-hz)*DrpFDM/rpFDM

                    #separate baryons into gas and stars                  
                    #probabilities
                    proj_HGA = projection(frac['HGA']*(dens['HGA']+dens['BG']), rbin, rvir, thickness, param, output='density')
                    proj_IGA = projection(frac['IGA']*(dens['IGA']+dens['BG']), rbin, rvir, thickness, param, output='density')
                    proj_CGA = projection(frac['CGA']*(dens['CGA']), rbin, rvir, thickness, param, output='density', star=True)
                    proj_SGA = projection(frac['SGA']*(dens['SGA']), rbin, rvir, thickness, param, output='density', star=True)
                    
                    #make sure no stars are outside virial radius
                    proj_CGA[np.where(rbin>=h['rvir'][j])] = 0.0 
                    proj_SGA[np.where(rbin>=h['rvir'][j])] = 0.0 

                    #interpolate projected densities
                    rho2D_HGA_tck = splrep(rbin, proj_HGA, s=0, k=1)
                    rho2D_IGA_tck = splrep(rbin, proj_IGA, s=0, k=1)
                    rho2D_CGA_tck = splrep(rbin, proj_CGA, s=0, k=1)
                    rho2D_SGA_tck = splrep(rbin, proj_SGA, s=0, k=1)

                    if param.shell.nbrhalo == 1:
                        if(len(rpBAR_nbrhaloes)>0):
                            rpBAR_wo_nbrhaloes_displ = rpBAR_wo_nbrhaloes + DrpBAR_wo_nbrhaloes
                            rho2D_HGA = splev(rpBAR_wo_nbrhaloes_displ, rho2D_HGA_tck,der=0,ext=3)
                            rho2D_IGA = splev(rpBAR_wo_nbrhaloes_displ, rho2D_IGA_tck,der=0,ext=3)
                            rho2D_CGA = splev(rpBAR_wo_nbrhaloes_displ, rho2D_CGA_tck,der=0,ext=3)
                            rho2D_SGA = splev(rpBAR_wo_nbrhaloes_displ, rho2D_SGA_tck,der=0,ext=3)
                        else:
                            rpBAR_displ = rpBAR + DrpBAR
                            rho2D_HGA = splev(rpBAR_displ, rho2D_HGA_tck,der=0,ext=3)
                            rho2D_IGA = splev(rpBAR_displ, rho2D_IGA_tck,der=0,ext=3)
                            rho2D_CGA = splev(rpBAR_displ, rho2D_CGA_tck,der=0,ext=3)
                            rho2D_SGA = splev(rpBAR_displ, rho2D_SGA_tck,der=0,ext=3)
                            
                    elif param.shell.nbrhalo == 0:
                        rpBAR_displ = rpBAR + DrpBAR
                        rho2D_HGA = splev(rpBAR_displ, rho2D_HGA_tck,der=0,ext=3)
                        rho2D_IGA = splev(rpBAR_displ, rho2D_IGA_tck,der=0,ext=3)
                        rho2D_CGA = splev(rpBAR_displ, rho2D_CGA_tck,der=0,ext=3)
                        rho2D_SGA = splev(rpBAR_displ, rho2D_SGA_tck,der=0,ext=3)
                    
                    #we record how likely particles in this healpix is a star with a float id
                    #id=0.0 for full gas, id=1.0 for full star
                    #imf_star   = impact_factor(h_cov, shell_cov, thickness, 1.0*rvir)
                    #rho2D_star = imf_star*(rho2D_CGA + rho2D_SGA)
                    #rho2D_bar  = imf_star*(rho2D_HGA + rho2D_IGA + rho2D_CGA + rho2D_SGA)
                    
                    V_overlap_ov_tot   = impact_factor(rbin, h_cov, shell_cov, thickness)
                    rhoSTAR = frac['CGA']*dens['CGA'] + frac['SGA']*dens['SGA']
                    corrSTAR = np.trapz(rbin**2 * V_overlap_ov_tot * rhoSTAR, rbin)/np.trapz(rbin**2 * rhoSTAR, rbin)

                    rho2D_star = corrSTAR*(rho2D_CGA + rho2D_SGA)
                    rho2D_bar  = corrSTAR*(rho2D_HGA + rho2D_IGA + rho2D_CGA + rho2D_SGA)


                    if param.shell.nbrhalo==1:
                        if (len(rpBAR_nbrhaloes) > 0):
                            DpBAR['rho2D_star_at_xyz'][ipbool_wo_nbrhaloes] = rho2D_star
                            DpBAR['rho2D_bar_at_xyz'][ipbool_wo_nbrhaloes]  = rho2D_bar
                        else:
                            DpBAR['rho2D_star_at_xyz'][ipbool] += rho2D_star
                            DpBAR['rho2D_bar_at_xyz'][ipbool]  += rho2D_bar
                            #DpBAR['rho2D_bar_at_xyz'][ipbool] = np.clip(DpBAR['rho2D_bar_at_xyz'][ipbool], a_min=0.0, a_max=1.0)
                    elif param.shell.nbrhalo == 0:
                        DpBAR['rho2D_star_at_xyz'][ipbool] += rho2D_star
                        DpBAR['rho2D_bar_at_xyz'][ipbool]  += rho2D_bar

        # Save temporary results
        filenameDpBAR = f'{output_dir}/DpBAR_shell_{shell_id}_cpu_{i_cpu}.npy'
        filenameDrpFDM = f'{output_dir}/DrpFDM_shell_{shell_id}_cpu_{i_cpu}.npy'
        np.save(filenameDpBAR, DpBAR)
        np.save(filenameDrpFDM, DpFDM)

        LOGGER.debug(f'Process {i_cpu} done. Elapsed time: {time()-ts:.2f} s')
        return filenameDpBAR, filenameDrpFDM

    else:
        # Single-component routine (similar)
        Dp_type = np.dtype([("x",'>f'),("y",'>f'),("z",'>f')])
        Dp = np.zeros(n_p, dtype=Dp_type)

        for j in maybe_progressbar(idx_local, total=len(idx_local), desc=f"Process {i_cpu}: Loop over halo subset"):
            #select host haloes (subhaloes >= 1)
            if (h['IDhost'][j] < 0):
                hx, hy, hz = h['x'][j], h['y'][j], h['z'][j]
                h_cov = h['cov'][j]
                Mvir, rvir, cvir = h['Mvir'][j], h['rvir'][j], h['cvir'][j]

                # print('start halo: ', j)

                #range where we consider displacement
                rmax = param.code.rmax
                rmin = (0.001*rvir if 0.001*rvir>param.code.rmin else param.code.rmin)
                rmax = (20.0*rvir if 20.0*rvir<param.code.rmax else param.code.rmax)
                rbin = np.logspace(np.log10(rmin),np.log10(rmax),100,base=10)

                #load 3D profiles
                cosmo_var  = splev(Mvir,var_tck)
                cosmo_bias = splev(Mvir,bias_tck)
                cosmo_corr = splev(rbin,corr_tck)
                profiles._update_params({'rbin': rbin, 'Mvir': Mvir, 'cvir': cvir, 'cosmo_corr': cosmo_corr, 'cosmo_bias': cosmo_bias, 'cosmo_var': cosmo_var})
                frac, dens, mass, press, temp = profiles.calc_profiles()

                #project 3D profiles
                rhoDMB_i = (dens['NFW'] + dens['BG'])
                rhoDMB_f = (dens['DMB'] + dens['BG'])
                
                #line of sight integration
                projected_MDM_i = projection(rhoDMB_i,rbin,rvir,thickness,param, output='mass')
                projected_MDM_f = projection(rhoDMB_f,rbin,rvir,thickness,param, output='mass')

                #displacement functions
                DDMB = displ(rbin, projected_MDM_i, projected_MDM_f)
                
                #r_boundary = param.shell.boundary_factor * rvir
                #imf = impact_factor(h_cov, shell_cov, thickness, r_boundary)
                #DDMB *= imf
                
                #V_overlap_ov_tot = relative volume (as a function of rbin)
                V_overlap_ov_tot = impact_factor(rbin, h_cov, shell_cov, thickness)

                #correction = [int dr r^2 V_rel(r) rho(r)]/[int dr r^2 rho(r)]
                rhoDMB = dens['DMB']
                corrDMB = np.trapz(rbin**2 * V_overlap_ov_tot * rhoDMB, rbin)/np.trapz(rbin**2 * rhoDMB, rbin)
                DDMB *= corrDMB

                DDMB_tck = splrep(rbin, DDMB,s=0,k=3)
                    
                smallestD = param.code.disp_trunc #Mpc/h
                #array of idx with DBAR > Dsmallest

                #array of idx with DDMB > Dsmallest
                idx_DMB = np.where(abs(DDMB) > smallestD)
                idx_DMB = idx_DMB[:][0]
                if (len(idx_DMB)>1):
                    idx_largest = idx_DMB[-1]
                    rball = rbin[idx_largest]
                else:
                    rball = 0.0

                rball = euclidean_distance(rball,shell_cov,param)
            
                #particle ids within rball
                ipbool = np.array(p_tree.query_ball_point((hx,hy,hz),rball))
                # print("Halo centre, surrounding particle number = ", hx,hy,hz, len(ipbool))

                if (len(ipbool) > 0):
                    #calculating radii of FDM particles around halo j
                    rpDMB  = ((p_darkmatter['x'][ipbool]-hx)**2.0 +
                            (p_darkmatter['y'][ipbool]-hy)**2.0 +
                            (p_darkmatter['z'][ipbool]-hz)**2.0)**0.5
                    rpDMB = arcdistance(rpDMB,shell_cov,param)

                    DrpDMB = splev(rpDMB,DDMB_tck,der=0,ext=1)
                    Dp['x'][ipbool] += (p_darkmatter['x'][ipbool]-hx)*DrpDMB/rpDMB
                    Dp['y'][ipbool] += (p_darkmatter['y'][ipbool]-hy)*DrpDMB/rpDMB
                    Dp['z'][ipbool] += (p_darkmatter['z'][ipbool]-hz)*DrpDMB/rpDMB

        filenameDrpDMB = f'{output_dir}/DrpDMB_shell_{shell_id}_cpu_{i_cpu}.npy'
        np.save(filenameDrpDMB, Dp)
        return "None", filenameDrpDMB


def displ(rbin, MINITIAL, MFINAL):
    """
    Calculates the displacement 
    """
    MFINAL_tck = splrep(rbin, MFINAL, s=0, k=3)
    MFINALinv_tck = splrep(MFINAL, rbin, s=0, k=3)
    rFINAL = splev(MINITIAL, MFINALinv_tck, der=0)
    DFINAL = rFINAL - rbin
    return DFINAL

def sum_structured_arrays_from_files_singlecomp(filenames):
    """
    Sum 'x', 'y', 'z', 'id' fields from a list of .npy structured arrays,
    opening one file at a time to minimize open file count and memory use.
    """
    if not filenames:
        raise ValueError("Empty file list")

    # Initialize accumulator with zeros like the first file
    first = np.load(filenames[0])
    out = np.zeros_like(first)
    out["x"] += first["x"]
    out["y"] += first["y"]
    out["z"] += first["z"]
    del first

    # Loop through remaining files one by one
    for fn in filenames[1:]:
        arr = np.load(fn)
        out["x"] += arr["x"]
        out["y"] += arr["y"]
        out["z"] += arr["z"]
        del arr

    for fn in filenames:
        os.remove(fn)
    return out

def sum_structured_arrays_from_files_multicomp(filenames):
    """
    Sum 'x', 'y', 'z', 'id' fields from a list of .npy structured arrays,
    opening one file at a time to minimize open file count and memory use.
    """
    if not filenames:
        raise ValueError("Empty file list")

    # Initialize accumulator with zeros like the first file
    first = np.load(filenames[0])
    out = np.zeros_like(first)
    out["x"] += first["x"]
    out["y"] += first["y"]
    out["z"] += first["z"]
    out["rho2D_star_at_xyz"] += first["rho2D_star_at_xyz"]
    out["rho2D_bar_at_xyz"]  += first["rho2D_bar_at_xyz"]
    del first

    # Loop through remaining files one by one
    
    # for fn in filenames[1:]:
    for fn in maybe_progressbar(filenames[1:] ,total = len(filenames[1:]), desc = f"Loop over displacement files"):
        arr = np.load(fn)
        out["x"] += arr["x"]
        out["y"] += arr["y"]
        out["z"] += arr["z"]
        out["rho2D_star_at_xyz"] += arr["rho2D_star_at_xyz"]
        out["rho2D_bar_at_xyz"]  += arr["rho2D_bar_at_xyz"]
        del arr

    # LOGGER.debug(f"minmax {np.min(out['rho2D_bar_at_xyz'])}, {np.max(out['rho2D_bar_at_xyz'])}")
    #calculate stellar fraction for each pixelparticle
    mask = (out["rho2D_bar_at_xyz"] != 0)
    out["id"][mask] = out["rho2D_star_at_xyz"][mask]/out["rho2D_bar_at_xyz"][mask]

    for fn in filenames:
        os.remove(fn)
    return out