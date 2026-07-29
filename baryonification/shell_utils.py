import os, sys
import traceback
import healpy as hp
import numpy as np
from scipy import spatial
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

def make_run_tag(shell_id):
    """
    Unique tag for this SLURM job + shell combination, used to suffix temp
    filenames in the shared tmp_files directory. Without this, two jobs (or
    two shells processed by the same job) writing fixed filenames like
    "p.npy"/"theta.npy" into the same directory would silently stomp on each
    other if run concurrently. Computed independently by both the writer
    (master rank) and readers (worker ranks) from the same job env var and
    shell_id, so no extra value needs to be passed/pickled around.
    """
    job_id = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
    return f"job{job_id}_shell{shell_id}"

def cleanup_job_tmp_files(param):
    """
    Remove every temp file this job tagged via make_run_tag(), regardless of
    which function created it. Intended to run in a finally-block around the
    whole job so a crash partway through (which skips the normal per-phase
    cleanup) doesn't leave large files behind in the shared scratch dir.
    Does not touch the persistent pixel_particles cache (it's untagged by
    design, meant to be reused across runs).
    """
    import glob
    job_id = os.environ.get('SLURM_JOB_ID', str(os.getpid()))
    pattern = os.path.join(param.files.tmp_files, f"*job{job_id}*")
    for fn in glob.glob(pattern):
        try:
            os.remove(fn)
        except OSError as e:
            LOGGER.warning(f"Could not remove temp file {fn}: {e}")

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


def loop_cpus_subsample_particles(pid, nproc, nside, shell_r, output_dir, run_tag):

    # diagnostic: when did this task actually start running, and on which
    # rank? Compared against the "[startup] rank N ready" timestamps logged
    # once per rank at job launch, this tells us whether tasks are slow to
    # *arrive* (rank was ready early but waited a long time for dispatch -
    # points to a slow per-task comm.send/large payload) or whether ranks
    # themselves only became ready late (points to a startup/import-storm
    # bottleneck instead).
    LOGGER.info(f"......[task-start] pid={pid} rank={MPI.COMM_WORLD.Get_rank()} starting")

    # read in large healpix arrays and the halo lookup structures - no need to
    # send them via MPI (every one of the nproc tasks used to receive its own
    # freshly-pickled copy of particle_pixels[p::nproc] (tens of MB),
    # halo_pixels_dict, adjacent_halos_dict and h directly in the task tuple,
    # turning each dispatch's comm.send into a multi-second pickle+transfer -
    # measured as a ~13s/task bottleneck serializing the whole dispatch loop.
    # Reading them from shared files instead means only the tiny (pid, nproc)
    # pair travels through MPI.
    #
    # pixels/halo_map/neighbor_map stay mmap'd: each worker only ever does
    # sparse, scattered single-pixel lookups into them, so the touched
    # footprint per worker is small. particle_pixels is different - every
    # worker iterates over its *entire* pid::nproc slice, so mmap'ing the one
    # shared ~nside^2-sized file meant all nproc workers collectively touched
    # the whole multi-GB file as page cache; under this cluster's per-task
    # cgroup memory accounting those shared pages get charged against every
    # task that maps them rather than deduplicated, which exhausted the job's
    # memory budget and caused thrashing (observed as an iteration rate
    # collapsing from ~50000 it/s to a fraction of an it/s mid-run). Fixed by
    # pre-splitting particle_pixels into nproc small *private* per-worker
    # files (see subsample_pixels) so each worker only ever touches its own
    # slice - no cross-task sharing, no double accounting. h is small enough
    # that a full private load per worker is cheap, so it no longer needs to
    # be mmap'd either.
    pixels = np.load(os.path.join(output_dir,f"pixels_{run_tag}.npy"), mmap_mode="r")
    halo_map = np.load(os.path.join(output_dir,f"halo_map_{run_tag}.npy"), mmap_mode="r")
    neighbor_map = np.load(os.path.join(output_dir,f"neighbor_map_{run_tag}.npy"), mmap_mode="r")
    pix_subset = np.load(os.path.join(output_dir,f"particle_pixels_chunk_{pid}_{run_tag}.npy"))
    h = np.load(os.path.join(output_dir,f"h_subsample_{run_tag}.npy"))
    with open(os.path.join(output_dir,f"halo_pixels_dict_{run_tag}.pkl"), "rb") as fpkl:
        halo_pixels_dict = pkl.load(fpkl)
    with open(os.path.join(output_dir,f"adjacent_halos_dict_{run_tag}.pkl"), "rb") as fpkl:
        adjacent_halos_dict = pkl.load(fpkl)

    n_pix_subset = len(pix_subset)

    # A plain Python list of (pix, pos, mass, tag) tuples - one tuple (or up
    # to 16, for halo pixels) per entry - used to cost ~10x the memory of the
    # final packed array below: each tuple plus its embedded 3-element numpy
    # array carries CPython object overhead on top of the 24 useful bytes/row
    # (i4+3f4+f4+i4). For a dense low-z shell with hundreds of millions of
    # output rows that overhead alone was enough to exhaust the job's memory
    # budget partway through the loop (observed as it/s collapsing from
    # ~50000 to a fraction of an it/s as the list grew) - independent of how
    # many worker processes the work is split across, since the aggregate
    # entry count, and therefore the aggregate Python-object overhead, is the
    # same either way. Fixed by writing directly into a preallocated packed
    # array instead of building this list and converting it at the end.
    Ngrandchildren_per_dim = 4
    Nchildren_per_dim = 2
    halo_sel = halo_map[pix_subset]
    neighbor_sel = neighbor_map[pix_subset]
    n_halo_pix = int(np.count_nonzero(halo_sel))
    n_neighbor_pix = int(np.count_nonzero(neighbor_sel))
    n_plain_pix = n_pix_subset - n_halo_pix - n_neighbor_pix
    n_total = (n_halo_pix * Ngrandchildren_per_dim**2
               + n_neighbor_pix * Nchildren_per_dim**2
               + n_plain_pix)

    arr = np.zeros(n_total, dtype=[('pix', 'i4'), ('pos', '3f4'), ('mass', 'f4'), ('tag', 'i4')])
    w = 0  # write cursor into arr

    t0 = time()
    for k, pix in enumerate(maybe_progressbar(pix_subset, total=n_pix_subset, desc=f"Process {pid}: Loop over pixel subset")):
        pix_mass = pixels[pix]

        if halo_sel[k]:
            halos_in_pixel = halo_pixels_dict.get(pix, [])
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

            n = Ngrandchildren_per_dim**2
            arr['pix'][w:w+n] = pix
            arr['pos'][w:w+n] = sub_positions
            arr['mass'][w:w+n] = pix_mass * mass_weights
            arr['tag'][w:w+n] = 2
            w += n

        elif neighbor_sel[k]:
            adjacent_halos = adjacent_halos_dict.get(pix, [])
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

            n = Nchildren_per_dim**2
            arr['pix'][w:w+n] = pix
            arr['pos'][w:w+n] = sub_positions
            arr['mass'][w:w+n] = pix_mass * mass_weights
            arr['tag'][w:w+n] = 1
            w += n

        else:
            dirs = np.array(hp.pix2vec(nside, pix, nest=False))
            arr['pix'][w] = pix
            arr['pos'][w] = dirs * shell_r
            arr['mass'][w] = pix_mass
            arr['tag'][w] = 0
            w += 1

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"particles_local_{pid}_{run_tag}.npy")
    np.save(filename, arr[:w])

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

def subsample_pixels(nside, pixels, shell_r, halos, param, pool, shell_id):
    """
    Subsampling pixels around halo centres to improve resolution.
    """
    run_tag = make_run_tag(shell_id)
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
    # Stage these on node-local disk (same reasoning/pattern as loop_halos in
    # displ.py): pixels_fn alone is multi-GB, and with up to N_cpu workers all
    # mmap'ing it (plus halo_map_fn/neighbor_map_fn) from network-mounted
    # tmp_files (Lustre), per-worker throughput drops noticeably as N_cpu goes
    # up - measured ~8-10x slower end-to-end going from 63 to 127 concurrent
    # workers on the same shared files. Falls back to tmp_files if TMPDIR
    # isn't set; only correct for single-node jobs (see loop_halos).
    local_dir = os.environ.get("TMPDIR", output_dir)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    pixels_fn = os.path.join(local_dir,f"pixels_{run_tag}.npy")
    halo_map_fn = os.path.join(local_dir,f"halo_map_{run_tag}.npy")
    neighbor_map_fn = os.path.join(local_dir,f"neighbor_map_{run_tag}.npy")
    h_subsample_fn = os.path.join(local_dir,f"h_subsample_{run_tag}.npy")
    halo_pixels_dict_fn = os.path.join(local_dir,f"halo_pixels_dict_{run_tag}.pkl")
    adjacent_halos_dict_fn = os.path.join(local_dir,f"adjacent_halos_dict_{run_tag}.pkl")
    np.save(pixels_fn, pixels)
    np.save(halo_map_fn, halo_map)
    np.save(neighbor_map_fn, neighbor_map)
    np.save(h_subsample_fn, h)
    with open(halo_pixels_dict_fn, "wb") as fpkl:
        pkl.dump(dict(halo_pixels_dict), fpkl)
    with open(adjacent_halos_dict_fn, "wb") as fpkl:
        pkl.dump(dict(adjacent_halos_dict), fpkl)

    # ---- prepare arguments for MultiPool ----
    # only the tiny (pid, nproc, nside, shell_r, output_dir, run_tag) tuple
    # travels through MPI now - everything large is read from the shared
    # files above instead of being re-pickled into every one of the nproc
    # task dispatches (see loop_cpus_subsample_particles for why).
    #
    # particle_pixels itself is pre-split into nproc private per-worker chunk
    # files here (instead of one shared file every worker mmaps and scans in
    # full) - see loop_cpus_subsample_particles for why a single shared mmap
    # of this particular array caused a catastrophic memory-accounting
    # regression.
    nproc = param.shell.N_cpu
    particle_pixels_chunk_fns = []
    for p in range(nproc):
        chunk_fn = os.path.join(local_dir, f"particle_pixels_chunk_{p}_{run_tag}.npy")
        np.save(chunk_fn, particle_pixels[p::nproc])
        particle_pixels_chunk_fns.append(chunk_fn)

    iterable_args = [
        (p, nproc, nside, shell_r, local_dir, run_tag)
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
    os.remove(pixels_fn)
    os.remove(halo_map_fn)
    os.remove(neighbor_map_fn)
    for chunk_fn in particle_pixels_chunk_fns:
        os.remove(chunk_fn)
    os.remove(h_subsample_fn)
    os.remove(halo_pixels_dict_fn)
    os.remove(adjacent_halos_dict_fn)

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
    i, pixels, h, param, shell_cov = task
    shell_id = param.shell.min_shell + i
    mesh_ref = param.shell.mesh_ref
    nside = param.shell.nside

    LOGGER.info(f"......Subsampling with mesh_ref={mesh_ref}")

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
        sub = subsample_pixels(nside, pixels, shell_cov, h, param, pool, shell_id)
        p = np.zeros(len(sub), dtype=p_dt)
        p[:]['x'], p[:]['y'], p[:]['z'] = sub['pos'][:,0], sub['pos'][:,1], sub['pos'][:,2]
        p[:]['M'] = sub['mass']
        p[:]['ref_order'] = sub['tag']
    else:
        raise ValueError(f"Unsupported mesh_ref: {mesh_ref}")
    return i, p


def get_healpix_map(p, param, star_fraction=None, pool=None, shell_id=None):
    run_tag = make_run_tag(shell_id)
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
    theta_fn = os.path.join(output_dir,f"theta_{run_tag}.npy")
    phi_fn = os.path.join(output_dir,f"phi_{run_tag}.npy")
    masses_fn = os.path.join(output_dir,f"masses_{run_tag}.npy")
    ref_order_fn = os.path.join(output_dir,f"ref_order_{run_tag}.npy")
    np.save(theta_fn, theta)
    np.save(phi_fn, phi)
    np.save(masses_fn, masses)
    np.save(ref_order_fn, p['ref_order'])


    tasks = [
    (idx_start, idx_end, nside, nside_out, output_dir, run_tag)
    for idx_start, idx_end in chunks
    ]

    results = pool.map(
        process_particle_chunk, tasks
    ) # returns list of filenames of the sparse (idx, mass) contributions saved by each worker

    # Reduce on master - only one dense npix_out-sized array needed here (on the
    # master only), workers themselves never materialize one (see
    # process_particle_chunk: each worker keeps only the sparse pixels it touches)
    final_map = np.zeros(hp.nside2npix(nside_out), dtype=np.float32)

    LOGGER.debug(f"summing up sparse contributions from {n_workers} workers...")
    for fname in tqdm(results):
        with np.load(fname) as npz:
            np.add.at(final_map, npz['idx'], npz['mass'])

    LOGGER.debug(f"Cleaning maps...")
    for fname in results:
        os.remove(fname)

    LOGGER.debug(f"Cleaning tmp files...")
    os.remove(theta_fn)
    os.remove(phi_fn)
    os.remove(masses_fn)
    os.remove(ref_order_fn)

    return final_map.astype(np.float16)

def process_particle_chunk(args):
    try:

        # memory profiling
        rank = MPI.COMM_WORLD.Get_rank()

        idx_start, idx_end, nside, nside_out, output_dir, run_tag = args
        # LOGGER.debug(f"[process_particle_chunk] pixelizing particles from {idx_start} to {idx_end}")

        # Slice particles
        theta = np.load(os.path.join(output_dir, f"theta_{run_tag}.npy"), mmap_mode="r")
        phi = np.load(os.path.join(output_dir, f"phi_{run_tag}.npy"), mmap_mode="r")
        masses = np.load(os.path.join(output_dir, f"masses_{run_tag}.npy"), mmap_mode="r")
        pref = np.load(os.path.join(output_dir, f"ref_order_{run_tag}.npy"), mmap_mode="r")
        ref_order = pref[idx_start:idx_end]
        th_all = theta[idx_start:idx_end]
        ph_all = phi[idx_start:idx_end]
        mass_all = masses[idx_start:idx_end]

        # LOGGER.debug(f"[process_particle_chunk {idx_start}-{idx_end}] loaded data")

        # Accumulate sparsely (idx, mass pairs) instead of a dense npix_out-sized
        # array per worker - a worker only ever needs to "own" the output pixels
        # its own particle chunk actually touches, not the full output grid
        # (a dense float32 array at nside_out=8192 is ~3.2 GB *per worker*,
        # regardless of chunk size; with 63 workers that's >200 GB just for this).
        idx_chunks = []
        mass_chunks = []

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

            idx_chunks.append(flat_parent)
            mass_chunks.append(flat_mass)

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

        # collapse duplicate target pixels (e.g. shared interpolation neighbors
        # between particles) within this worker's own chunk before writing out
        unique_idx, summed = consolidate_sparse_contributions(idx_chunks, {'mass': mass_chunks})
        del idx_chunks, mass_chunks
        gc.collect()

        fname = os.path.join(output_dir, f"local_map_{rank}_{run_tag}.npz")
        np.savez(fname, idx=unique_idx.astype(np.int64), mass=summed['mass'].astype(np.float32))
        del unique_idx, summed
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
    Save a dictionary of HEALPix maps to an HDF5 group. Uses require_group (not
    create_group) so this can be called once per shell, across repeated opens of
    the same output file, without erroring on shells already written by a
    previous call.
    """
    group = file.require_group(group_name)
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
    """
    rbin
    """

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

def save_cKDTree_shared(tree, output_dir, prefix="p_tree", large_array_threshold=1_000_000):
    """
    Save a cKDTree's pickle state as one file per large internal array (data,
    indices, the packed node buffer, ...) plus one small pickle for the rest,
    instead of a single pickle blob. Lets workers mmap the large arrays (see
    load_cKDTree_shared) so they share physical pages on the same node rather
    than each unpickling a private full-size copy.
    Returns the list of files written, for cleanup later.
    """
    state = tree.__getstate__()
    meta = []
    written = []
    for i, item in enumerate(state):
        if isinstance(item, np.ndarray) and item.nbytes > large_array_threshold:
            fname = os.path.join(output_dir, f"{prefix}_arr_{i}.npy")
            np.save(fname, item)
            written.append(fname)
            meta.append(('arr', i))
        else:
            meta.append(('val', item))
    meta_fname = os.path.join(output_dir, f"{prefix}_meta.pkl")
    with open(meta_fname, "wb") as f:
        pkl.dump(meta, f)
    written.append(meta_fname)
    return written

def load_cKDTree_shared(output_dir, prefix="p_tree"):
    """
    Reconstruct a cKDTree saved by save_cKDTree_shared, with its large internal
    arrays memory-mapped read-only rather than copied into private memory.
    """
    meta_fname = os.path.join(output_dir, f"{prefix}_meta.pkl")
    with open(meta_fname, "rb") as f:
        meta = pkl.load(f)
    state = []
    for kind, payload in meta:
        if kind == 'arr':
            fname = os.path.join(output_dir, f"{prefix}_arr_{payload}.npy")
            state.append(np.load(fname, mmap_mode='r'))
        else:
            state.append(payload)
    tree = spatial.cKDTree.__new__(spatial.cKDTree)
    tree.__setstate__(tuple(state))
    return tree

def consolidate_sparse_contributions(idx_chunks, value_chunks_dict):
    """
    Combine many small (idx_array, value_array) contributions - e.g. one pair per
    halo touching a handful of particles - into a single set of unique indices with
    summed values per field, without ever materializing a full shell-sized array.
    idx_chunks: list of 1D int arrays
    value_chunks_dict: dict field_name -> list of 1D arrays, aligned with idx_chunks
    """
    if not idx_chunks:
        empty_idx = np.array([], dtype=np.int64)
        return empty_idx, {k: np.array([], dtype=np.float64) for k in value_chunks_dict}
    all_idx = np.concatenate(idx_chunks)
    unique_idx, inverse = np.unique(all_idx, return_inverse=True)
    summed = {}
    for field, chunks in value_chunks_dict.items():
        all_val = np.concatenate(chunks)
        summed[field] = np.bincount(inverse, weights=all_val, minlength=len(unique_idx))
    return unique_idx, summed

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
    run_tag = make_run_tag(shell_id)

    LOGGER.debug(f'......loading precomputed data: p...')
    p = np.load(os.path.join(output_dir,f"p_{run_tag}.npy"), mmap_mode='r')  # load p to a file to avoid pickling issues
    LOGGER.debug(f'......loading precomputed data: h...')
    h = np.load(os.path.join(output_dir,f"h_{run_tag}.npy"), mmap_mode='r')  # load p to a file to avoid pickling issues
    LOGGER.debug(f'......loading precomputed data: p_tree...')
    p_tree = load_cKDTree_shared(output_dir, prefix=f"p_tree_{run_tag}")
    # sys.exit()
    profiles = Profiles(None, 1e13, None, None, None, None, param)

    # Multiple names for the same memory-mapped array
    # we do not modify p here, jsut read out the coordinates - so we do not need to copy
    p_darkmatter = p
    p_baryons = p if param.code.multicomp else None
    LOGGER.debug(f'......process {i_cpu} loaded all precomputed data.')

    if param.code.multicomp:
        # Sparse accumulators: each halo appends an (idx, value) event instead of
        # writing into a dense n_p-sized array, so a worker's memory scales with the
        # particles its own halo subset actually touches, not with the whole shell.
        fdm_idx_chunks, fdm_x_chunks, fdm_y_chunks, fdm_z_chunks = [], [], [], []
        bar_idx_chunks, bar_x_chunks, bar_y_chunks, bar_z_chunks = [], [], [], []
        star_idx_chunks, star_val_chunks, bar2D_val_chunks = [], [], []
        n_host_halos = 0
        n_displaced = 0  # host halos that actually found particles within rball, i.e.
                          # halos that contributed any displacement to DpBAR/DpFDM
        n_bar_superpixel = 0  # host halos whose max|DBAR| exceeds one output pixel's
                               # physical size, i.e. the baryon displacement is large
                               # enough to actually move particles into a different pixel
        # physical size of one output pixel at this shell's distance - constant for the
        # whole shell, so compute it once instead of inside the per-halo loop
        pixel_size = hp.nside2resol(param.shell.nside_out) * shell_cov

        for j in maybe_progressbar(idx_local, total=len(idx_local), desc=f"Process {i_cpu}: Loop over halo subset"):
            if h['IDhost'][j] < 0:
                n_host_halos += 1
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

                # tally every halo (not just the j%10 sample below) so the final count
                # is exact: is this halo's baryon displacement even large enough to move
                # particles into a different pixel? (sub-pixel displacement => no visible
                # change in the gas/dm difference map regardless of how different DBAR/DFDM are)
                max_DBAR = np.max(np.abs(DBAR))
                max_DFDM = np.max(np.abs(DFDM))
                if max_DBAR > pixel_size:
                    n_bar_superpixel += 1

                do_diag = False #(j % 10 == 0)
                if do_diag:
                    LOGGER.debug(
                        f"......[diag] halo j={j} Mvir={Mvir:.3e} rvir={rvir:.4f} cvir={cvir:.3f} "
                        f"cosmo_var={cosmo_var:.4e} cosmo_bias={cosmo_bias:.4e} "
                        f"fcdm={frac['CDM']:.4f} fhga={frac['HGA']:.4f} fcga={frac['CGA']:.6f} "
                        f"fsga={frac['SGA']:.6f} figa={frac['IGA']:.6f} "
                        f"pixel_size={pixel_size:.6e} Mpc/h "
                        f"max|DBAR|={max_DBAR:.6e} (>pixel? {max_DBAR > pixel_size}) "
                        f"max|DFDM|={max_DFDM:.6e} (>pixel? {max_DFDM > pixel_size}) "
                        f"max|DBAR-DFDM|={np.max(np.abs(DBAR-DFDM)):.6e}"
                    )

                # Compute impact factors
                V_overlap_ov_tot = impact_factor(rbin, h_cov, shell_cov, thickness)

                # Apply corrections
                rhoCDM = dens['CDM']
                rhoBAR = frac['HGA']*dens['HGA'] + frac['IGA']*dens['IGA'] + frac['CGA']*dens['CGA'] + frac['SGA']*dens['SGA']
                corrFDM = np.trapz(rbin**2 * V_overlap_ov_tot * rhoCDM, rbin)/np.trapz(rbin**2 * rhoCDM, rbin)
                corrBAR = np.trapz(rbin**2 * V_overlap_ov_tot * rhoBAR, rbin)/np.trapz(rbin**2 * rhoBAR, rbin)

                if do_diag:
                    LOGGER.debug(
                        f"......[diag-corr] halo j={j} h_cov={h_cov:.6e} shell_cov={shell_cov:.6e} "
                        f"thickness={thickness:.6e} |h_cov-shell_cov|={abs(h_cov-shell_cov):.6e} Mpc/h "
                        f"corrBAR={corrBAR:.6e} corrFDM={corrFDM:.6e} "
                        f"max|DBAR|_postcorr={np.max(np.abs(DBAR*corrBAR)):.6e}"
                    )

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

                if do_diag:
                    LOGGER.debug(
                        f"......[diag-rball] halo j={j} smallestD={smallestD:.4e} Mpc/h "
                        f"len(idx_BAR)={len(idx_BAR)} rball_BAR={rball_BAR:.6e} "
                        f"len(idx_FDM)={len(idx_FDM)} rball_FDM={rball_FDM:.6e} "
                        f"rball(euclid)={rball:.6e} Mpc/h n_particles_in_ball={len(ipbool)}"
                    )

                if (len(ipbool) > 0):
                    n_displaced += 1
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
                        fdm_idx_chunks.append(ipbool)
                        fdm_x_chunks.append((p_darkmatter['x'][ipbool]-hx)*DrpFDM/rpFDM)
                        fdm_y_chunks.append((p_darkmatter['y'][ipbool]-hy)*DrpFDM/rpFDM)
                        fdm_z_chunks.append((p_darkmatter['z'][ipbool]-hz)*DrpFDM/rpFDM)

                        if(len(rpBAR_nbrhaloes)>0):
                            DrpBAR_nbrhaloes    = splev(rpBAR_nbrhaloes,DFDM_tck,der=0,ext=1)
                            DrpBAR_wo_nbrhaloes = splev(rpBAR_wo_nbrhaloes,DBAR_tck,der=0,ext=1)
                            bar_idx_chunks.append(ipbool_nbrhaloes)
                            bar_x_chunks.append((p_baryons['x'][ipbool_nbrhaloes]-hx)*DrpBAR_nbrhaloes/rpBAR_nbrhaloes)
                            bar_y_chunks.append((p_baryons['y'][ipbool_nbrhaloes]-hy)*DrpBAR_nbrhaloes/rpBAR_nbrhaloes)
                            bar_z_chunks.append((p_baryons['z'][ipbool_nbrhaloes]-hz)*DrpBAR_nbrhaloes/rpBAR_nbrhaloes)
                            bar_idx_chunks.append(ipbool_wo_nbrhaloes)
                            bar_x_chunks.append((p_baryons['x'][ipbool_wo_nbrhaloes]-hx)*DrpBAR_wo_nbrhaloes/rpBAR_wo_nbrhaloes)
                            bar_y_chunks.append((p_baryons['y'][ipbool_wo_nbrhaloes]-hy)*DrpBAR_wo_nbrhaloes/rpBAR_wo_nbrhaloes)
                            bar_z_chunks.append((p_baryons['z'][ipbool_wo_nbrhaloes]-hz)*DrpBAR_wo_nbrhaloes/rpBAR_wo_nbrhaloes)
                        else:
                            DrpBAR = splev(rpBAR,DBAR_tck,der=0,ext=1)
                            bar_idx_chunks.append(ipbool)
                            bar_x_chunks.append((p_baryons['x'][ipbool]-hx)*DrpBAR/rpBAR)
                            bar_y_chunks.append((p_baryons['y'][ipbool]-hy)*DrpBAR/rpBAR)
                            bar_z_chunks.append((p_baryons['z'][ipbool]-hz)*DrpBAR/rpBAR)

                    elif param.shell.nbrhalo == 0:

                        DrpBAR = splev(rpBAR,DBAR_tck,der=0,ext=1)
                        bar_idx_chunks.append(ipbool)
                        bar_x_chunks.append((p_baryons['x'][ipbool]-hx)*DrpBAR/rpBAR)
                        bar_y_chunks.append((p_baryons['y'][ipbool]-hy)*DrpBAR/rpBAR)
                        bar_z_chunks.append((p_baryons['z'][ipbool]-hz)*DrpBAR/rpBAR)

                        DrpFDM = splev(rpFDM,DFDM_tck,der=0,ext=1)
                        fdm_idx_chunks.append(ipbool)
                        fdm_x_chunks.append((p_darkmatter['x'][ipbool]-hx)*DrpFDM/rpFDM)
                        fdm_y_chunks.append((p_darkmatter['y'][ipbool]-hy)*DrpFDM/rpFDM)
                        fdm_z_chunks.append((p_darkmatter['z'][ipbool]-hz)*DrpFDM/rpFDM)

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
                            star_idx_chunks.append(ipbool_wo_nbrhaloes)
                            star_val_chunks.append(rho2D_star)
                            bar2D_val_chunks.append(rho2D_bar)
                        else:
                            star_idx_chunks.append(ipbool)
                            star_val_chunks.append(rho2D_star)
                            bar2D_val_chunks.append(rho2D_bar)
                    elif param.shell.nbrhalo == 0:
                        star_idx_chunks.append(ipbool)
                        star_val_chunks.append(rho2D_star)
                        bar2D_val_chunks.append(rho2D_bar)

        # Consolidate this worker's sparse contributions (sized to the particles its
        # own halo subset touched, not to the full shell) before writing to disk.
        fdm_idx, fdm_vals = consolidate_sparse_contributions(
            fdm_idx_chunks, {'x': fdm_x_chunks, 'y': fdm_y_chunks, 'z': fdm_z_chunks})
        bar_idx, bar_vals = consolidate_sparse_contributions(
            bar_idx_chunks, {'x': bar_x_chunks, 'y': bar_y_chunks, 'z': bar_z_chunks})
        star_idx, star_vals = consolidate_sparse_contributions(
            star_idx_chunks, {'rho2D_star_at_xyz': star_val_chunks, 'rho2D_bar_at_xyz': bar2D_val_chunks})

        # DpBAR needs both the xyz displacement and the star/gas split, which can be
        # keyed to slightly different particle sets (nbrhalo==1) - merge onto their union.
        bar_union_idx = np.union1d(bar_idx, star_idx)
        DpBAR_sparse_type = np.dtype([("idx",'<i8'),("x",'>f'),("y",'>f'),("z",'>f'),
                                       ("rho2D_star_at_xyz",'>f4'),("rho2D_bar_at_xyz",'>f4')])
        DpBAR = np.zeros(len(bar_union_idx), dtype=DpBAR_sparse_type)
        DpBAR['idx'] = bar_union_idx
        pos_bar = np.searchsorted(bar_union_idx, bar_idx)
        DpBAR['x'][pos_bar] = bar_vals['x']
        DpBAR['y'][pos_bar] = bar_vals['y']
        DpBAR['z'][pos_bar] = bar_vals['z']
        pos_star = np.searchsorted(bar_union_idx, star_idx)
        DpBAR['rho2D_star_at_xyz'][pos_star] = star_vals['rho2D_star_at_xyz']
        DpBAR['rho2D_bar_at_xyz'][pos_star] = star_vals['rho2D_bar_at_xyz']

        DpFDM_sparse_type = np.dtype([("idx",'<i8'),("x",'>f'),("y",'>f'),("z",'>f')])
        DpFDM = np.zeros(len(fdm_idx), dtype=DpFDM_sparse_type)
        DpFDM['idx'] = fdm_idx
        DpFDM['x'] = fdm_vals['x']
        DpFDM['y'] = fdm_vals['y']
        DpFDM['z'] = fdm_vals['z']

        # Save temporary results
        filenameDpBAR = f'{output_dir}/DpBAR_{run_tag}_cpu_{i_cpu}.npy'
        filenameDrpFDM = f'{output_dir}/DrpFDM_{run_tag}_cpu_{i_cpu}.npy'
        np.save(filenameDpBAR, DpBAR)
        np.save(filenameDrpFDM, DpFDM)

        LOGGER.info(f'......process {i_cpu}: {n_displaced}/{n_host_halos} host halos had particles within rball and were displaced (contributed to DpBAR/DpFDM).')
        LOGGER.info(f'......process {i_cpu}: {n_bar_superpixel}/{n_host_halos} host halos had max|DBAR| larger than one output pixel.')
        LOGGER.debug(f'Process {i_cpu} done. Elapsed time: {time()-ts:.2f} s')
        return filenameDpBAR, filenameDrpFDM, n_displaced, n_bar_superpixel

    else:
        # Single-component routine (similar), sparse accumulation as above.
        #
        # Restructured into 3 phases instead of one per-halo loop that both
        # computes each halo's physics AND queries the (potentially huge,
        # network-mmap'd) particle KDTree in the same iteration:
        #   1) per-halo profile/displacement physics (cheap, ~10ms/halo, no
        #      shared-blob access) - produces centers/rballs/DDMB splines.
        #   2) ONE batched p_tree.query_ball_point() call across all host
        #      halos at once, using cKDTree's native per-point radius array +
        #      workers=-1 multithreading, instead of one Python-level call per
        #      halo. Verified equivalent to the per-halo loop on synthetic
        #      data. This replaces thousands of sequential small queries
        #      (each paying Python call overhead and, under network-mounted
        #      tmp storage, its own scattered random-access I/O) with a
        #      single internally-parallel C-level batch query.
        #   3) per-halo displacement using the precomputed spline + the
        #      batched query's result - the only remaining per-halo touch of
        #      the shared particle array.
        idx_local_arr = np.asarray(idx_local)
        host_mask = h['IDhost'][idx_local_arr] < 0
        host_idx = idx_local_arr[host_mask]
        n_host_halos = len(host_idx)

        centers = np.empty((n_host_halos, 3))
        rballs = np.empty(n_host_halos)
        DDMB_tcks = [None] * n_host_halos

        for i, j in enumerate(maybe_progressbar(host_idx, total=n_host_halos, desc=f"Process {i_cpu}: Loop over halo subset (physics)")):
            hx, hy, hz = h['x'][j], h['y'][j], h['z'][j]
            h_cov = h['cov'][j]
            Mvir, rvir, cvir = h['Mvir'][j], h['rvir'][j], h['cvir'][j]

            #range where we consider displacement
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

            #V_overlap_ov_tot = relative volume (as a function of rbin)
            V_overlap_ov_tot = impact_factor(rbin, h_cov, shell_cov, thickness)

            #correction = [int dr r^2 V_rel(r) rho(r)]/[int dr r^2 rho(r)]
            rhoDMB = dens['DMB']
            corrDMB = np.trapz(rbin**2 * V_overlap_ov_tot * rhoDMB, rbin)/np.trapz(rbin**2 * rhoDMB, rbin)
            DDMB *= corrDMB

            smallestD = param.code.disp_trunc #Mpc/h
            #array of idx with DDMB > Dsmallest
            idx_DMB = np.where(abs(DDMB) > smallestD)
            idx_DMB = idx_DMB[:][0]
            if (len(idx_DMB)>1):
                idx_largest = idx_DMB[-1]
                rball = rbin[idx_largest]
            else:
                rball = 0.0
            rball = euclidean_distance(rball,shell_cov,param)

            centers[i] = (hx, hy, hz)
            rballs[i] = rball
            DDMB_tcks[i] = splrep(rbin, DDMB, s=0, k=3)

        # One batched, internally-parallel query instead of n_host_halos
        # separate Python-level calls.
        all_ipbool = p_tree.query_ball_point(centers, r=rballs, workers=-1)

        dmb_idx_chunks, dmb_x_chunks, dmb_y_chunks, dmb_z_chunks = [], [], [], []
        n_displaced = 0  # host halos that actually found particles within rball

        for i in range(n_host_halos):
            ipbool = np.array(all_ipbool[i])
            if (len(ipbool) > 0):
                n_displaced += 1
                hx, hy, hz = centers[i]
                #calculating radii of FDM particles around halo j
                rpDMB  = ((p_darkmatter['x'][ipbool]-hx)**2.0 +
                        (p_darkmatter['y'][ipbool]-hy)**2.0 +
                        (p_darkmatter['z'][ipbool]-hz)**2.0)**0.5
                rpDMB = arcdistance(rpDMB,shell_cov,param)

                DrpDMB = splev(rpDMB,DDMB_tcks[i],der=0,ext=1)
                dmb_idx_chunks.append(ipbool)
                dmb_x_chunks.append((p_darkmatter['x'][ipbool]-hx)*DrpDMB/rpDMB)
                dmb_y_chunks.append((p_darkmatter['y'][ipbool]-hy)*DrpDMB/rpDMB)
                dmb_z_chunks.append((p_darkmatter['z'][ipbool]-hz)*DrpDMB/rpDMB)

        dmb_idx, dmb_vals = consolidate_sparse_contributions(
            dmb_idx_chunks, {'x': dmb_x_chunks, 'y': dmb_y_chunks, 'z': dmb_z_chunks})
        Dp_sparse_type = np.dtype([("idx",'<i8'),("x",'>f'),("y",'>f'),("z",'>f')])
        Dp = np.zeros(len(dmb_idx), dtype=Dp_sparse_type)
        Dp['idx'] = dmb_idx
        Dp['x'] = dmb_vals['x']
        Dp['y'] = dmb_vals['y']
        Dp['z'] = dmb_vals['z']

        filenameDrpDMB = f'{output_dir}/DrpDMB_{run_tag}_cpu_{i_cpu}.npy'
        np.save(filenameDrpDMB, Dp)
        LOGGER.info(f'......process {i_cpu}: {n_displaced}/{n_host_halos} host halos had particles within rball and were displaced.')
        return "None", filenameDrpDMB, n_displaced


def displ(rbin, MINITIAL, MFINAL):
    """
    Calculates the displacement 
    """
    MFINAL_tck = splrep(rbin, MFINAL, s=0, k=3)
    MFINALinv_tck = splrep(MFINAL, rbin, s=0, k=3)
    rFINAL = splev(MINITIAL, MFINALinv_tck, der=0)
    DFINAL = rFINAL - rbin
    return DFINAL

def sum_structured_arrays_from_files_singlecomp(filenames, n_p):
    """
    Scatter-add the sparse (idx, x, y, z) contributions from each worker's file into
    a single n_p-sized dense array, opening one (now small, sparse) file at a time.
    """
    if not filenames:
        raise ValueError("Empty file list")

    Dp_type = np.dtype([("x",'>f'),("y",'>f'),("z",'>f')])
    out = np.zeros(n_p, dtype=Dp_type)

    for fn in filenames:
        arr = np.load(fn)
        idx = arr["idx"]
        np.add.at(out["x"], idx, arr["x"])
        np.add.at(out["y"], idx, arr["y"])
        np.add.at(out["z"], idx, arr["z"])
        del arr

    for fn in filenames:
        os.remove(fn)
    return out

def sum_structured_arrays_from_files_multicomp(filenames, n_p):
    """
    Scatter-add the sparse (idx, x, y, z, rho2D_star_at_xyz, rho2D_bar_at_xyz)
    contributions from each worker's file into a single n_p-sized dense array,
    opening one (now small, sparse) file at a time.
    """
    if not filenames:
        raise ValueError("Empty file list")

    Dp_type = np.dtype([("x",'>f'),("y",'>f'),("z",'>f'),
                         ("id",'>f4'),("rho2D_star_at_xyz",'>f4'),("rho2D_bar_at_xyz",'>f4')])
    out = np.zeros(n_p, dtype=Dp_type)

    for fn in maybe_progressbar(filenames ,total = len(filenames), desc = f"Loop over displacement files"):
        arr = np.load(fn)
        idx = arr["idx"]
        np.add.at(out["x"], idx, arr["x"])
        np.add.at(out["y"], idx, arr["y"])
        np.add.at(out["z"], idx, arr["z"])
        # DpFDM's sparse files (DpFDM_sparse_type) carry no rho2D_* fields - only
        # DpBAR's do, so only scatter-add them when present (this function is shared
        # between both bar_filenames and dm_filenames callers).
        if "rho2D_star_at_xyz" in arr.dtype.names:
            np.add.at(out["rho2D_star_at_xyz"], idx, arr["rho2D_star_at_xyz"])
            np.add.at(out["rho2D_bar_at_xyz"], idx, arr["rho2D_bar_at_xyz"])
        del arr

    # LOGGER.debug(f"minmax {np.min(out['rho2D_bar_at_xyz'])}, {np.max(out['rho2D_bar_at_xyz'])}")
    #calculate stellar fraction for each pixelparticle
    mask = (out["rho2D_bar_at_xyz"] != 0)
    out["id"][mask] = out["rho2D_star_at_xyz"][mask]/out["rho2D_bar_at_xyz"][mask]

    for fn in filenames:
        os.remove(fn)
    return out