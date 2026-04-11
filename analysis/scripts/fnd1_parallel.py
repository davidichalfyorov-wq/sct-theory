"""FND-1 Parallel Computation Engine."""

from __future__ import annotations

# CRITICAL: Set BLAS threads BEFORE any numpy/scipy import.
# On Linux fork(), children inherit parent's BLAS state. If numpy is imported
# first with default threads (=all cores), each of 140 workers spawns 176
# BLAS threads -> 24640 threads on 176 cores -> catastrophic contention.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count, Value, Lock
from functools import partial

import numpy as np

# ---------------------------------------------------------------------------
# System-adaptive worker count
# ---------------------------------------------------------------------------

_n_cores = cpu_count()

if _n_cores >= 64:
    # Server / cloud VM (e.g. 176-core H100 node, 1.4 TB RAM)
    # Use 80% of cores, 1 BLAS thread per worker (dense eigendecomp is single-thread anyway)
    N_WORKERS = max(1, int(_n_cores * 0.8))
    OPENBLAS_THREADS_PER_WORKER = 1
elif _n_cores >= 16:
    # Workstation (e.g. i9-12900KS: 24 threads, 64 GB)
    # 2 BLAS threads per worker, leave headroom for user
    N_WORKERS = max(1, int(_n_cores * 0.8) // 2)
    OPENBLAS_THREADS_PER_WORKER = 2
else:
    # Laptop / small machine
    N_WORKERS = max(1, _n_cores - 1)
    OPENBLAS_THREADS_PER_WORKER = 1


# Detect GPUs without initializing CUDA (safe before fork)
try:
    from fnd1_gpu import detect_gpus
    N_GPUS = detect_gpus()
except ImportError:
    N_GPUS = 0

# Shared counter for round-robin GPU assignment across Pool workers
from multiprocessing import Value as _Value
_gpu_worker_counter = _Value("i", 0)


def _init_worker():
    """Initialize each worker process with optimal thread settings and GPU."""
    os.environ["OPENBLAS_NUM_THREADS"] = str(OPENBLAS_THREADS_PER_WORKER)
    os.environ["OMP_NUM_THREADS"] = str(OPENBLAS_THREADS_PER_WORKER)
    os.environ["MKL_NUM_THREADS"] = str(OPENBLAS_THREADS_PER_WORKER)

    # Assign GPU round-robin and enable GPU mode (only if GPUs detected)
    if N_GPUS > 0:
        with _gpu_worker_counter.get_lock():
            wid = _gpu_worker_counter.value
            _gpu_worker_counter.value += 1
        os.environ["CUDA_VISIBLE_DEVICES"] = str(wid % N_GPUS)
        try:
            from fnd1_gpu import enable_worker_mode
            enable_worker_mode()
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

_counter = None
_lock = None
_total = None
_t0 = None
_desc = None


def _init_progress(counter, lock, total, t0, desc):
    """Initialize shared progress state in each worker."""
    global _counter, _lock, _total, _t0, _desc
    _counter = counter
    _lock = lock
    _total = total
    _t0 = t0
    _desc = desc
    _init_worker()


def _report_progress():
    """Increment counter and print progress."""
    global _counter, _lock, _total, _t0, _desc
    with _lock:
        _counter.value += 1
        done = _counter.value
        elapsed = time.perf_counter() - _t0.value
        if done > 0:
            eta = elapsed / done * (_total.value - done)
        else:
            eta = 0
        if done % max(1, _total.value // 20) == 0 or done == _total.value:
            print(f"    [{_desc.value.decode()}] {done}/{_total.value} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)",
                  flush=True)


# ---------------------------------------------------------------------------
# Core parallel runner
# ---------------------------------------------------------------------------

def parallel_map(func, args_list, desc="computing", n_workers=N_WORKERS):
    """
    Execute func(args) in parallel across n_workers processes.

    Parameters
    ----------
    func : callable
        Function to execute. Must be picklable (top-level, no lambdas).
    args_list : list of tuples
        Each tuple is unpacked as func(*args).
    desc : str
        Description for progress output.
    n_workers : int
        Number of worker processes.

    Returns
    -------
    list of results, in same order as args_list.
    """
    total = len(args_list)
    if total == 0:
        return []

    print(f"  Parallel: {total} tasks on {n_workers} workers [{desc}]", flush=True)
    t0_wall = time.perf_counter()

    counter = Value("i", 0)
    lock = Lock()
    total_val = Value("i", total)
    t0_val = Value("d", t0_wall)
    # Note: shared strings are hard in multiprocessing; progress printed from main only

    if n_workers <= 1 or total <= 2:
        # Sequential fallback
        results = []
        for i, args in enumerate(args_list):
            if (i + 1) % max(1, total // 20) == 0 or i == 0:
                elapsed = time.perf_counter() - t0_wall
                eta = elapsed / (i + 1) * (total - i - 1) if i > 0 else 0
                print(f"    [{desc}] {i+1}/{total} "
                      f"({elapsed:.0f}s, ~{eta:.0f}s left)", flush=True)
            results.append(func(*args) if isinstance(args, tuple) else func(args))
        elapsed = time.perf_counter() - t0_wall
        print(f"    [{desc}] Done: {elapsed:.1f}s", flush=True)
        return results

    # Parallel execution
    with Pool(processes=n_workers, initializer=_init_worker) as pool:
        if isinstance(args_list[0], tuple):
            results = pool.starmap(func, args_list)
        else:
            results = pool.map(func, args_list)

    elapsed = time.perf_counter() - t0_wall
    print(f"    [{desc}] Done: {total} tasks in {elapsed:.1f}s "
          f"({elapsed/total:.2f}s/task, {n_workers}w)", flush=True)
    return results


# ---------------------------------------------------------------------------
# Sprinkling-specific parallel helpers
# ---------------------------------------------------------------------------

def _compute_single_sprinkling(seed_int, N, T, eps, rho, compute_func_name):
    """
    Worker function for one sprinkling. Must be top-level for pickle.

    Parameters
    ----------
    seed_int : int
        Random seed.
    N, T, eps, rho : float
        Sprinkling parameters.
    compute_func_name : str
        Name of the function to call: 'observables', 'family_b_eig', 'commutator', etc.
    """
    # Import here to ensure each worker has its own module state
    from fnd1_ensemble_runner import (
        compute_interval_cardinalities, build_bd_L,
        compute_family_B_eigenvalues,
    )
    from fnd1_gate5_runner import sprinkle_curved, _sprinkle_flat

    rng = np.random.default_rng(seed_int)

    if eps == 0.0:
        pts, C = _sprinkle_flat(N, T, rng)
    else:
        pts, C = sprinkle_curved(N, eps, T, rng)

    n_mat = compute_interval_cardinalities(C)
    L = build_bd_L(C, n_mat, rho)

    if compute_func_name == "commutator":
        return _compute_commutator_observables(L, C)
    elif compute_func_name == "family_b_eig":
        eig = compute_family_B_eigenvalues(L)
        return eig
    elif compute_func_name == "svd":
        sv = np.linalg.svd(L, compute_uv=False)
        return sv[sv > 1e-10]
    else:
        return _compute_commutator_observables(L, C)


def _compute_commutator_observables(L, C):
    """Compute all observables for the commutator experiment.

    Uses optimized formula: [H,M] = (L^TL - LL^T)/2
    This is 1.8x faster than H@M - M@H (2 matmuls instead of 4).
    Verified: max diff < 1e-12.
    """
    # Optimized: sparse L^TL - LL^T (6.4x faster than dense H@M-M@H)
    import scipy.sparse as sp
    L_sp = sp.csr_matrix(L)
    comm = ((L_sp.T @ L_sp - L_sp @ L_sp.T) / 2.0).toarray()
    comm = (comm + comm.T) / 2.0  # enforce exact symmetry (sparse rounding)
    comm_eigs = np.linalg.eigvalsh(comm)

    # Summary stats
    a = np.abs(comm_eigs)
    s = np.sum(a)
    entropy = float(-np.sum((a/s) * np.log(a/s + 1e-300))) if s > 0 else 0.0

    total_causal = float(np.sum(C))

    # Layer counts
    past = C.T
    n_mat = C @ C  # recompute for layer info
    n0 = float(np.sum((past > 0) & (n_mat == 0)))

    # SVD entropy for comparison
    sv = np.linalg.svd(L, compute_uv=False)
    sv = sv[sv > 1e-10]
    if len(sv) > 0 and np.sum(sv) > 0:
        p_sv = sv / np.sum(sv)
        svd_entropy = float(-np.sum(p_sv * np.log(p_sv + 1e-300)))
    else:
        svd_entropy = 0.0

    # H eigenvalue entropy (eigvalsh: values only, 2.8x faster)
    H = (L + L.T) / 2.0
    h_eigs = np.linalg.eigvalsh(H)
    a_h = np.abs(h_eigs)
    s_h = np.sum(a_h)
    h_entropy = float(-np.sum((a_h/s_h) * np.log(a_h/s_h + 1e-300))) if s_h > 0 else 0.0

    return {
        "comm_entropy": entropy,
        "comm_mean_abs": float(np.mean(np.abs(comm_eigs))),
        "comm_max_abs": float(np.max(np.abs(comm_eigs))),
        "comm_std": float(np.std(comm_eigs)),
        "comm_frobenius": float(np.sqrt(np.sum(comm_eigs**2))),
        "total_causal": total_causal,
        "n0": n0,
        "svd_entropy": svd_entropy,
        "h_entropy": h_entropy,
    }


def parallel_sprinkle(N, T, eps, M, base_seed, compute="commutator",
                      desc="sprinkle", n_workers=N_WORKERS):
    """
    Run M sprinklings in parallel and return list of result dicts.

    Parameters
    ----------
    N : int, causal set size
    T : float, diamond time extent
    eps : float, curvature parameter
    M : int, number of sprinklings
    base_seed : int, base seed (spawns M children)
    compute : str, what to compute ('commutator', 'family_b_eig', 'svd')
    desc : str, description for progress
    n_workers : int

    Returns
    -------
    list of M result dicts/arrays
    """
    V = T**2 / 2.0
    rho = N / V

    ss = np.random.SeedSequence(base_seed)
    child_seeds = ss.spawn(M)
    seed_ints = [int(cs.generate_state(1)[0]) for cs in child_seeds]

    args_list = [
        (seed_int, N, T, eps, rho, compute)
        for seed_int in seed_ints
    ]

    return parallel_map(_compute_single_sprinkling, args_list,
                        desc=desc, n_workers=n_workers)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(N=2000, M=24):
    """Quick benchmark: sequential vs parallel."""
    print(f"Benchmark: N={N}, M={M}", flush=True)
    T = 1.0
    V = T**2 / 2.0
    rho = N / V

    ss = np.random.SeedSequence(42)
    seeds = ss.spawn(M)
    seed_ints = [int(s.generate_state(1)[0]) for s in seeds]

    # Sequential
    t0 = time.perf_counter()
    for i, si in enumerate(seed_ints):
        _compute_single_sprinkling(si, N, T, 0.0, rho, "commutator")
    seq_time = time.perf_counter() - t0
    print(f"  Sequential: {seq_time:.1f}s ({seq_time/M:.2f}s/task)", flush=True)

    # Parallel
    args = [(si, N, T, 0.0, rho, "commutator") for si in seed_ints]
    t0 = time.perf_counter()
    with Pool(processes=N_WORKERS, initializer=_init_worker) as pool:
        pool.starmap(_compute_single_sprinkling, args)
    par_time = time.perf_counter() - t0
    print(f"  Parallel ({N_WORKERS}w): {par_time:.1f}s ({par_time/M:.2f}s/task)",
          flush=True)
    print(f"  Speedup: {seq_time/par_time:.1f}x", flush=True)


if __name__ == "__main__":
    print(f"System: {cpu_count()} logical cores, {N_WORKERS} workers")
    print(f"OPENBLAS threads per worker: {OPENBLAS_THREADS_PER_WORKER}")
    benchmark(N=2000, M=24)
    print()
    benchmark(N=3000, M=12)
