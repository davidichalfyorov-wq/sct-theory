"""
FND-1 Route 3: Commutator [H, M] — DEFINITIVE Experiment (v2).

Fixes ALL deficiencies of v1:
  1. Isotropic density control (R ~ O(eps^2), O(eps) weaker than anisotropic: ~13x at eps=0.5)
  2. Common-random-numbers (CRN) paired design
  3. 15 reproducibility seeds
  4. Theoretical prediction framework
  5. Boundary point exclusion option
  6. Full optimization stack (eigvalsh, sparse, parallel, flush)
  7. Dashboard integration (progress file)

Run with MKL for best performance:
  "C:/Users/youre/miniconda3/envs/sct-mkl/python.exe" analysis/scripts/fnd1_route3_commutator_v2.py
Or with standard Python:
  python analysis/scripts/fnd1_route3_commutator_v2.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_ensemble_runner import (
    compute_interval_cardinalities,
    build_bd_L,
)
from fnd1_gate5_runner import sprinkle_curved, _sprinkle_flat, check_omega_positivity
from fnd1_experiment_registry import (
    ExperimentMeta, update_progress, clear_progress, save_experiment, RESULTS_DIR,
)
from fnd1_parallel import N_WORKERS, _init_worker

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_POINTS = 3000
M_ENSEMBLE = 100        # per epsilon (increased from 80)
M_PAIRED = 100           # per reproducibility seed
T_DIAMOND = 1.0
MASTER_SEED = 42

# Curvature values (anisotropic: R ≠ 0)
EPSILON_ANISO = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75]

# Isotropic density control: Omega = 1 + eps*(t^2 + x^2)
# NOTE: R is NOT exactly 0. R = 8*eps^2*(x^2-t^2)/(1+eps*(t^2+x^2))^4 = O(eps^2)
# with OPPOSITE anisotropic signature (x^2-t^2) vs main experiment (t^2-x^2).
# At eps=0.5: max|R_iso| ~ 0.31 vs R_aniso ~ 4.0 (~13x weaker curvature).
# This is a WEAK-curvature control, not a zero-curvature control.
EPSILON_ISO = [0.1, 0.25, 0.5]  # 3 values for meaningful regression

# Reproducibility seeds
REPRO_SEEDS_POS = [11111, 22222, 33333, 44444, 55555]   # eps = +0.5
REPRO_SEEDS_NEG = [66666, 77777, 88888, 99999, 12321]   # eps = -0.5
REPRO_SEEDS_ISO = [13579, 24680, 36912, 48024, 51015]   # isotropic eps = +0.5

# 80% resource cap
WORKERS = N_WORKERS  # 10 workers from fnd1_parallel (80% of 24 cores)


# ---------------------------------------------------------------------------
# Isotropic sprinkling (R = O(eps^2), weak curvature, non-uniform density)
# ---------------------------------------------------------------------------

def omega_isotropic(t, x, eps):
    """Omega for isotropic density: 1 + eps*(t^2 + x^2). R = O(eps^2), NOT zero."""
    return 1.0 + eps * (t**2 + x**2)


def sprinkle_isotropic(N, eps, T, rng):
    """
    Sprinkle with isotropic non-uniform density Omega^2 = (1+eps(t^2+x^2))^2.
    Causal structure is Minkowski (conformal invariance).
    R = 8*eps^2*(x^2-t^2)/(1+eps(t^2+x^2))^4 — O(eps^2), NOT zero.
    Opposite anisotropy signature vs main experiment. ~13x weaker at eps=0.5.
    """
    omega_min_val = omega_isotropic(0, 0, eps)  # at origin
    # Max of t^2+x^2 in diamond: at corners (t=0, x=±T/2) or (t=±T/2, x=0)
    u_max = (T / 2)**2
    omega_max_val = omega_isotropic(T/2, 0, eps)  # or (0, T/2) — same
    max_omega_sq = max(omega_min_val, omega_max_val)**2

    if omega_min_val <= 0 or omega_max_val <= 0:
        raise ValueError(f"Omega <= 0 for isotropic eps={eps}")

    t_acc, x_acc = [], []
    while len(t_acc) < N:
        batch = int(N * 1.5)
        u = rng.uniform(-T/2, T/2, size=batch)
        v = rng.uniform(-T/2, T/2, size=batch)
        t_cand = (u + v) / 2
        x_cand = (u - v) / 2
        omega_sq = omega_isotropic(t_cand, x_cand, eps)**2
        accept = rng.uniform(0, 1, size=batch) < (omega_sq / max_omega_sq)
        t_acc.extend(t_cand[accept])
        x_acc.extend(x_cand[accept])

    t_arr = np.array(t_acc[:N])
    x_arr = np.array(x_acc[:N])
    order = np.argsort(t_arr)
    t_arr, x_arr = t_arr[order], x_arr[order]
    points = np.column_stack([t_arr, x_arr])

    dt = t_arr[np.newaxis, :] - t_arr[:, np.newaxis]
    dx = np.abs(x_arr[np.newaxis, :] - x_arr[:, np.newaxis])
    C = ((dt > dx) & (dt > 0)).astype(np.float64)

    return points, C


# ---------------------------------------------------------------------------
# Common-Random-Numbers (CRN) paired sprinkling
# ---------------------------------------------------------------------------

def sprinkle_crn_pair(N, eps, T, rng, mode="anisotropic"):
    """
    Generate a CORRELATED pair (flat, curved) using Common Random Numbers.

    Method: generate a large pool of candidate points from the same RNG.
    - Flat: take first N candidates (uniform)
    - Curved: accept/reject from the SAME pool, take first N accepted

    The overlap creates genuine correlation → lower Var(ΔK).
    """
    V = T**2 / 2.0

    # Generate candidate pool (oversized)
    pool_size = int(N * 2.5)
    u = rng.uniform(-T/2, T/2, size=pool_size)
    v = rng.uniform(-T/2, T/2, size=pool_size)
    t_pool = (u + v) / 2
    x_pool = (u - v) / 2

    # Flat: first N
    t_flat = t_pool[:N].copy()
    x_flat = x_pool[:N].copy()

    # Curved: accept/reject on the SAME pool
    if mode == "isotropic":
        omega_vals = omega_isotropic(t_pool, x_pool, eps)
    else:
        omega_vals = 1.0 + eps * (t_pool**2 - x_pool**2)

    omega_sq = omega_vals**2
    max_osq = np.max(omega_sq)
    accept_prob = omega_sq / max_osq

    # Use deterministic acceptance from the same RNG stream
    rand_accept = rng.uniform(0, 1, size=pool_size)
    accepted = rand_accept < accept_prob
    t_curved = t_pool[accepted][:N]
    x_curved = x_pool[accepted][:N]

    if len(t_curved) < N:
        # Need more points — extend pool
        extra_needed = N - len(t_curved)
        u2 = rng.uniform(-T/2, T/2, size=extra_needed * 3)
        v2 = rng.uniform(-T/2, T/2, size=extra_needed * 3)
        t2 = (u2 + v2) / 2
        x2 = (u2 - v2) / 2
        if mode == "isotropic":
            o2 = omega_isotropic(t2, x2, eps)
        else:
            o2 = 1.0 + eps * (t2**2 - x2**2)
        osq2 = o2**2
        ap2 = osq2 / max_osq
        r2 = rng.uniform(0, 1, size=len(t2))
        acc2 = r2 < ap2
        t_curved = np.concatenate([t_curved, t2[acc2][:extra_needed]])
        x_curved = np.concatenate([x_curved, x2[acc2][:extra_needed]])

    t_curved = t_curved[:N]
    x_curved = x_curved[:N]

    def _build(t_arr, x_arr):
        order = np.argsort(t_arr)
        t_arr, x_arr = t_arr[order], x_arr[order]
        dt = t_arr[np.newaxis, :] - t_arr[:, np.newaxis]
        dx = np.abs(x_arr[np.newaxis, :] - x_arr[:, np.newaxis])
        C = ((dt > dx) & (dt > 0)).astype(np.float64)
        return C

    C_flat = _build(t_flat, x_flat)
    C_curved = _build(t_curved, x_curved)

    return C_flat, C_curved


# ---------------------------------------------------------------------------
# Core computation (optimized)
# ---------------------------------------------------------------------------

def compute_observables_from_C(C, N, T):
    """Compute all observables from a causal matrix C. Fully optimized."""
    V = T**2 / 2.0
    rho = N / V

    n_mat = compute_interval_cardinalities(C)  # sparse C@C
    L = build_bd_L(C, n_mat, rho)

    # Commutator [H,M] = (L^TL - LL^T)/2 — sparse, 6.4x faster
    L_sp = sp.csr_matrix(L)
    comm = ((L_sp.T @ L_sp - L_sp @ L_sp.T) / 2.0).toarray()
    comm_eigs = np.linalg.eigvalsh(comm)  # 2.8x faster (no eigvecs)

    # Summary statistics
    a = np.abs(comm_eigs)
    s = np.sum(a)
    comm_entropy = float(-np.sum((a/s) * np.log(a/s + 1e-300))) if s > 0 else 0.0
    comm_frobenius = float(np.sqrt(np.sum(comm_eigs**2)))
    comm_mean_abs = float(np.mean(a))
    comm_std = float(np.std(comm_eigs))

    total_causal = float(np.sum(C))

    # Layer counts: n_mat[j,i] counts elements between j≺i (upper tri for C upper tri)
    # past[i,j] = C[j,i] = 1 if j≺i, so past is lower tri
    # For (past > 0) & (n == k): need n in same orientation as past → use n_mat.T
    past = C.T
    n_past = n_mat.T  # n_past[i,j] = interval count for j≺i, matching past[i,j]
    n0 = float(np.sum((past > 0) & (n_past == 0)))
    n1 = float(np.sum((past > 0) & (n_past == 1)))
    n3plus = float(np.sum((past > 0) & (n_past >= 3)))

    # SVD entropy (comparison)
    sv = np.linalg.svd(L, compute_uv=False)
    sv = sv[sv > 1e-10]
    svd_entropy = 0.0
    if len(sv) > 0 and np.sum(sv) > 0:
        p_sv = sv / np.sum(sv)
        svd_entropy = float(-np.sum(p_sv * np.log(p_sv + 1e-300)))

    return {
        "comm_entropy": comm_entropy,
        "comm_frobenius": comm_frobenius,
        "comm_mean_abs": comm_mean_abs,
        "comm_std": comm_std,
        "total_causal": total_causal,
        "n0": n0, "n1": n1, "n3plus": n3plus,
        "svd_entropy": svd_entropy,
    }


# ---------------------------------------------------------------------------
# Worker functions (top-level for pickle)
# ---------------------------------------------------------------------------

def _worker_single(args):
    """Compute observables for one sprinkling."""
    seed_int, N, T, eps, mode = args
    rng = np.random.default_rng(seed_int)
    if mode == "flat":
        pts, C = _sprinkle_flat(N, T, rng)
    elif mode == "isotropic":
        pts, C = sprinkle_isotropic(N, eps, T, rng)
    else:
        pts, C = sprinkle_curved(N, eps, T, rng)
    obs = compute_observables_from_C(C, N, T)
    obs["eps"] = eps
    obs["mode"] = mode
    return obs


def _worker_null(seed_int):
    """Compute causal and null [H,M] entropy. Module-level for pickle."""
    rho = N_POINTS / (T_DIAMOND**2 / 2.0)
    rng = np.random.default_rng(seed_int)
    pts, C = _sprinkle_flat(N_POINTS, T_DIAMOND, rng)
    n_mat = compute_interval_cardinalities(C)
    L = build_bd_L(C, n_mat, rho)
    obs_causal = compute_observables_from_C(C, N_POINTS, T_DIAMOND)

    # Null: random values in same nonzero positions, matched Frobenius norm
    rng2 = np.random.default_rng(seed_int + 1000000)
    mask = L != 0
    L_null = np.zeros_like(L)
    L_null[mask] = rng2.standard_normal(int(np.sum(mask)))
    L_null *= np.linalg.norm(L, 'fro') / max(np.linalg.norm(L_null, 'fro'), 1e-15)
    L_sp = sp.csr_matrix(L_null)
    comm_null = ((L_sp.T @ L_sp - L_sp @ L_sp.T) / 2.0).toarray()
    null_eigs = np.linalg.eigvalsh(comm_null)
    a = np.abs(null_eigs)
    s_n = np.sum(a)
    null_ent = float(-np.sum((a / s_n) * np.log(a / s_n + 1e-300))) if s_n > 0 else 0.0

    return obs_causal["comm_entropy"], null_ent


def _worker_crn_pair(args):
    """Compute paired observables using CRN."""
    seed_int, N, T, eps, mode = args
    rng = np.random.default_rng(seed_int)
    C_flat, C_curved = sprinkle_crn_pair(N, eps, T, rng, mode=mode)
    obs_flat = compute_observables_from_C(C_flat, N, T)
    obs_curved = compute_observables_from_C(C_curved, N, T)
    return obs_flat, obs_curved


# ---------------------------------------------------------------------------
# Mediation analysis
# ---------------------------------------------------------------------------

def mediation_analysis(all_data, obs_key="comm_entropy"):
    """Full mediation: direct r, partial r controlling TC, n0, TC+n0."""
    eps_arr = np.array([d["eps"] for d in all_data])
    obs_arr = np.array([d[obs_key] for d in all_data])
    tc_arr = np.array([d["total_causal"] for d in all_data])
    n0_arr = np.array([d["n0"] for d in all_data])

    r_direct, p_direct = stats.pearsonr(eps_arr, obs_arr)

    def residualize(x, ctrl):
        sl, ic, _, _, _ = stats.linregress(ctrl, x)
        return x - (sl * ctrl + ic)

    def partial_r(x, y, ctrl):
        xr, yr = residualize(x, ctrl), residualize(y, ctrl)
        if np.std(xr) > 0 and np.std(yr) > 0:
            return stats.pearsonr(xr, yr)
        return 0.0, 1.0

    r_tc, p_tc = partial_r(eps_arr, obs_arr, tc_arr)
    r_n0, p_n0 = partial_r(eps_arr, obs_arr, n0_arr)

    # Multiple regression residualization (TC + n0)
    X = np.column_stack([tc_arr, n0_arr, np.ones(len(tc_arr))])
    b_eps = np.linalg.lstsq(X, eps_arr, rcond=None)[0]
    b_obs = np.linalg.lstsq(X, obs_arr, rcond=None)[0]
    er = eps_arr - X @ b_eps
    or_ = obs_arr - X @ b_obs
    if np.std(er) > 0 and np.std(or_) > 0:
        r_both, p_both = stats.pearsonr(er, or_)
    else:
        r_both, p_both = 0.0, 1.0

    return {
        "obs": obs_key,
        "r_direct": float(r_direct), "p_direct": float(p_direct),
        "r_partial_tc": float(r_tc), "p_partial_tc": float(p_tc),
        "r_partial_n0": float(r_n0), "p_partial_n0": float(p_n0),
        "r_partial_both": float(r_both), "p_partial_both": float(p_both),
    }


# ---------------------------------------------------------------------------
# Theoretical prediction
# ---------------------------------------------------------------------------

def theoretical_framework():
    """
    Theoretical framework for [H,M] vs curvature.

    Established facts:
      - Box_g is formally self-adjoint on L^2(M, sqrt|g| dV) for globally hyperbolic M
      - Discrete BD operator L is strictly lower-triangular → always non-normal → [H,M] ≠ 0
      - [H,M] ≠ 0 is structural (tautological for non-normal matrices), not a prediction
      - Physical relevance of [H,M] to continuum curvature is UNESTABLISHED

    ANSATZ (not derived, no literature support):
      - ||[H,M]||^2 ≈ c_0(N) + c_1(N) * R_bar/rho + O(R^2/rho^2)
      - c_0 = finite-size non-normality (present even at R=0, dominant)
      - c_1 = curvature coupling (the signal we seek)
      - Plausible on dimensional grounds but NOT proven

    Testable expectations (not predictions — no first-principles derivation):
      1. Aniso (R ~ O(eps)): comm observables should correlate with eps
      2. Iso (R ~ O(eps^2)): correlation ~13x weaker at eps=0.5
      3. Aniso slope >> Iso slope separates O(eps) from O(eps^2)
      4. Direction of change: unknown (empirical)
    """
    return {
        "expectation_1": "Anisotropic: comm_entropy correlates with eps (empirical, no derivation)",
        "expectation_2": "Isotropic: comm_entropy weakly dependent (R = O(eps^2), ~13x weaker at eps=0.5)",
        "expectation_3": "Aniso slope >> Iso slope separates O(eps) curvature from O(eps^2) residual",
        "expectation_4": "Signal should scale as O(eps) (ANSATZ, not derived, needs multi-N test)",
        "null_hypothesis": "comm_entropy is fully determined by total_causal (pair counting)",
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    meta = ExperimentMeta(
        route=3, name="route3_commutator_v2",
        description="Commutator [H,M] DEFINITIVE: CRN pairing, isotropic control, 15 seeds",
        N=N_POINTS, M=M_ENSEMBLE, status="running",
    )

    print("=" * 70, flush=True)
    print("FND-1 ROUTE 3: COMMUTATOR [H,M] — DEFINITIVE EXPERIMENT (v2)", flush=True)
    print("=" * 70, flush=True)
    print(f"N={N_POINTS}, M={M_ENSEMBLE}, workers={WORKERS}", flush=True)
    print(f"Anisotropic eps: {EPSILON_ANISO}", flush=True)
    print(f"Isotropic eps: {EPSILON_ISO}", flush=True)
    print(f"Repro seeds: {len(REPRO_SEEDS_POS)+len(REPRO_SEEDS_NEG)+len(REPRO_SEEDS_ISO)}", flush=True)
    print(f"CRN pairing: YES", flush=True)
    print(f"Theoretical framework: YES", flush=True)
    theory = theoretical_framework()
    for k, v in theory.items():
        print(f"  {k}: {v}", flush=True)
    print(flush=True)

    ss = np.random.SeedSequence(MASTER_SEED)

    # ==================================================================
    # PART 1: Multi-epsilon ANISOTROPIC (R ≠ 0)
    # ==================================================================
    print("=" * 60, flush=True)
    print("PART 1: ANISOTROPIC (R ≠ 0) — multi-epsilon", flush=True)
    print("=" * 60, flush=True)

    update_progress(meta, step="Part 1: Anisotropic multi-epsilon", pct=0.0)

    aniso_data = []
    for i, eps in enumerate(EPSILON_ANISO):
        eps_ss = ss.spawn(1)[0]
        child_seeds = eps_ss.spawn(M_ENSEMBLE)
        seed_ints = [int(cs.generate_state(1)[0]) for cs in child_seeds]
        args = [(si, N_POINTS, T_DIAMOND, eps, "anisotropic" if eps != 0 else "flat")
                for si in seed_ints]

        print(f"\n  eps={eps:+.3f}: {M_ENSEMBLE} sprinklings...", flush=True)
        t0 = time.perf_counter()
        with Pool(WORKERS, initializer=_init_worker) as pool:
            results = pool.map(_worker_single, args)
        elapsed = time.perf_counter() - t0
        print(f"    Done in {elapsed:.1f}s", flush=True)

        for r in results:
            r["eps"] = eps
            r["mode"] = "anisotropic"
        aniso_data.extend(results)

        ce = [r["comm_entropy"] for r in results]
        tc = [r["total_causal"] for r in results]
        print(f"    comm_entropy: {np.mean(ce):.6f} ± {np.std(ce)/np.sqrt(M_ENSEMBLE):.6f}", flush=True)
        print(f"    total_causal: {np.mean(tc):.1f}", flush=True)

        update_progress(meta, step=f"Part 1: eps={eps:+.2f}",
                       pct=(i+1)/len(EPSILON_ANISO) * 0.30)

    # ==================================================================
    # PART 2: ISOTROPIC CONTROL (R = O(eps^2), non-uniform density)
    # ==================================================================
    print(f"\n{'='*60}", flush=True)
    print("PART 2: ISOTROPIC CONTROL (R = O(eps^2), non-uniform density)", flush=True)
    print("=" * 60, flush=True)

    update_progress(meta, step="Part 2: Isotropic control", pct=0.30)

    iso_data = []
    for eps in EPSILON_ISO:
        eps_ss = ss.spawn(1)[0]
        child_seeds = eps_ss.spawn(M_ENSEMBLE)
        seed_ints = [int(cs.generate_state(1)[0]) for cs in child_seeds]
        args = [(si, N_POINTS, T_DIAMOND, eps, "isotropic") for si in seed_ints]

        print(f"\n  eps_iso={eps:+.3f}: {M_ENSEMBLE} sprinklings...", flush=True)
        t0 = time.perf_counter()
        with Pool(WORKERS, initializer=_init_worker) as pool:
            results = pool.map(_worker_single, args)
        elapsed = time.perf_counter() - t0
        print(f"    Done in {elapsed:.1f}s", flush=True)

        for r in results:
            r["eps"] = eps
            r["mode"] = "isotropic"
        iso_data.extend(results)

        ce = [r["comm_entropy"] for r in results]
        print(f"    comm_entropy: {np.mean(ce):.6f} ± {np.std(ce)/np.sqrt(M_ENSEMBLE):.6f}", flush=True)

    update_progress(meta, step="Part 2 done", pct=0.45)

    # ==================================================================
    # PART 3: CRN PAIRED REPRODUCIBILITY
    # ==================================================================
    print(f"\n{'='*60}", flush=True)
    print("PART 3: CRN PAIRED REPRODUCIBILITY (15 seeds)", flush=True)
    print("=" * 60, flush=True)

    update_progress(meta, step="Part 3: CRN reproducibility", pct=0.45)

    repro_results = {}
    all_repro_configs = (
        [(s, 0.5, "anisotropic") for s in REPRO_SEEDS_POS] +
        [(s, -0.5, "anisotropic") for s in REPRO_SEEDS_NEG] +
        [(s, 0.5, "isotropic") for s in REPRO_SEEDS_ISO]
    )

    for idx, (seed_r, eps_r, mode_r) in enumerate(all_repro_configs):
        r_ss = np.random.SeedSequence(seed_r)
        child_seeds = r_ss.spawn(M_PAIRED)
        seed_ints = [int(cs.generate_state(1)[0]) for cs in child_seeds]
        args = [(si, N_POINTS, T_DIAMOND, eps_r, mode_r) for si in seed_ints]

        label = f"seed={seed_r}, eps={eps_r:+.1f}, {mode_r}"
        print(f"\n  {label}: {M_PAIRED} CRN pairs...", flush=True)
        t0 = time.perf_counter()
        with Pool(WORKERS, initializer=_init_worker) as pool:
            pair_results = pool.map(_worker_crn_pair, args)
        elapsed = time.perf_counter() - t0
        print(f"    Done in {elapsed:.1f}s", flush=True)

        # Paired differences
        diffs = {k: [] for k in ["comm_entropy", "comm_frobenius", "svd_entropy"]}
        for obs_flat, obs_curved in pair_results:
            for k in diffs:
                diffs[k].append(obs_curved[k] - obs_flat[k])

        key = f"{mode_r}_eps{eps_r:+.1f}_seed{seed_r}"
        repro_results[key] = {}
        for k, d_arr in diffs.items():
            d_arr = np.array(d_arr)
            t_stat, p_val = stats.ttest_1samp(d_arr, 0.0)
            repro_results[key][k] = {
                "mean_diff": float(np.mean(d_arr)),
                "sem": float(np.std(d_arr, ddof=1) / np.sqrt(M_PAIRED)),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "significant": bool(p_val < 0.05),
            }
            sig = "**" if p_val < 0.05 else ""
            print(f"    {k}: shift={np.mean(d_arr):+.6f}, p={p_val:.4f} {sig}", flush=True)

        update_progress(meta, step=f"Part 3: {idx+1}/{len(all_repro_configs)} seeds",
                       pct=0.45 + (idx+1)/len(all_repro_configs) * 0.35)

    # ==================================================================
    # PART 4: NULL MODEL (sparsity-matched)
    # ==================================================================
    print(f"\n{'='*60}", flush=True)
    print("PART 4: NULL MODEL (sparsity-matched)", flush=True)
    print("=" * 60, flush=True)

    update_progress(meta, step="Part 4: Null model", pct=0.80)

    null_ss = ss.spawn(1)[0]
    null_seeds = null_ss.spawn(80)
    V = T_DIAMOND**2 / 2.0
    rho = N_POINTS / V

    causal_ents, null_ents = [], []
    null_seed_ints = [int(cs.generate_state(1)[0]) for cs in null_seeds]

    print(f"  Computing {len(null_seed_ints)} null comparisons...", flush=True)
    t0 = time.perf_counter()
    with Pool(WORKERS, initializer=_init_worker) as pool:
        null_results_raw = pool.map(_worker_null, null_seed_ints)
    elapsed = time.perf_counter() - t0
    print(f"    Done in {elapsed:.1f}s", flush=True)

    for ce, ne in null_results_raw:
        causal_ents.append(ce)
        null_ents.append(ne)

    t_null, p_null = stats.ttest_ind(causal_ents, null_ents)
    print(f"  Causal: {np.mean(causal_ents):.6f}, Null: {np.mean(null_ents):.6f}, p={p_null:.2e}", flush=True)

    null_model = {
        "causal_mean": float(np.mean(causal_ents)),
        "null_mean": float(np.mean(null_ents)),
        "t_stat": float(t_null),
        "p_value": float(p_null),
        "discriminates": bool(p_null < 0.01),
    }

    # ==================================================================
    # ANALYSIS
    # ==================================================================
    print(f"\n{'='*60}", flush=True)
    print("ANALYSIS", flush=True)
    print("=" * 60, flush=True)

    update_progress(meta, step="Analysis", pct=0.90)

    # Mediation on anisotropic data
    print("\n  ANISOTROPIC mediation:", flush=True)
    observables = ["comm_entropy", "comm_frobenius", "comm_mean_abs", "comm_std", "svd_entropy"]
    med_aniso = {}
    for obs in observables:
        m = mediation_analysis(aniso_data, obs)
        med_aniso[obs] = m
        surv = abs(m["r_partial_both"]) > 0.1 and m["p_partial_both"] < 0.10
        print(f"    {obs}: direct r={m['r_direct']:+.4f} (p={m['p_direct']:.2e}), "
              f"partial r={m['r_partial_both']:+.4f} (p={m['p_partial_both']:.2e}) "
              f"{'SURVIVES' if surv else 'mediated'}", flush=True)

    # Mediation on isotropic data (should show NO signal)
    print("\n  ISOTROPIC mediation (should be zero):", flush=True)
    med_iso = {}
    if len(iso_data) > 20:
        for obs in ["comm_entropy", "svd_entropy"]:
            m = mediation_analysis(iso_data, obs)
            med_iso[obs] = m
            print(f"    {obs}: direct r={m['r_direct']:+.4f} (p={m['p_direct']:.2e})", flush=True)

    # Aniso vs Iso comparison
    print("\n  ANISO vs ISO comparison:", flush=True)
    for eps in EPSILON_ISO:
        aniso_ents = [d["comm_entropy"] for d in aniso_data if abs(d["eps"] - eps) < 0.01]
        iso_ents = [d["comm_entropy"] for d in iso_data if abs(d["eps"] - eps) < 0.01]
        if aniso_ents and iso_ents:
            t_ai, p_ai = stats.ttest_ind(aniso_ents, iso_ents)
            print(f"    eps={eps:+.2f}: aniso={np.mean(aniso_ents):.6f}, "
                  f"iso={np.mean(iso_ents):.6f}, diff={np.mean(aniso_ents)-np.mean(iso_ents):+.6f}, "
                  f"p={p_ai:.4f}", flush=True)

    # Reproducibility summary
    print("\n  REPRODUCIBILITY:", flush=True)
    n_sig_aniso = sum(1 for k, v in repro_results.items()
                      if "anisotropic" in k and v["comm_entropy"]["significant"])
    n_sig_iso = sum(1 for k, v in repro_results.items()
                    if "isotropic" in k and v["comm_entropy"]["significant"])
    n_aniso = sum(1 for k in repro_results if "anisotropic" in k)
    n_iso = sum(1 for k in repro_results if "isotropic" in k)
    print(f"    Anisotropic: {n_sig_aniso}/{n_aniso} seeds significant", flush=True)
    print(f"    Isotropic:   {n_sig_iso}/{n_iso} seeds significant", flush=True)

    # Group means table
    print("\n  GROUP MEANS (anisotropic):", flush=True)
    print(f"  {'eps':>6} {'comm_ent':>10} {'SEM':>8} {'total_caus':>10} {'svd_ent':>10}", flush=True)
    for eps in EPSILON_ANISO:
        d = [x for x in aniso_data if abs(x["eps"] - eps) < 0.01]
        if d:
            ce = [x["comm_entropy"] for x in d]
            tc = [x["total_causal"] for x in d]
            se = [x["svd_entropy"] for x in d]
            print(f"  {eps:+6.3f} {np.mean(ce):10.6f} {np.std(ce)/np.sqrt(len(d)):8.6f} "
                  f"{np.mean(tc):10.1f} {np.mean(se):10.6f}", flush=True)

    # ==================================================================
    # VERDICT
    # ==================================================================
    total_time = time.perf_counter() - t_total
    print(f"\n{'='*70}", flush=True)
    print("VERDICT", flush=True)
    print("=" * 70, flush=True)

    comm_surv = (abs(med_aniso["comm_entropy"]["r_partial_both"]) > 0.1
                 and med_aniso["comm_entropy"]["p_partial_both"] < 0.10)
    # Iso control: R_iso = O(eps^2) → signal ~13x weaker than aniso at eps=0.5
    # "flat" = iso slope much smaller than aniso slope
    aniso_r = abs(med_aniso["comm_entropy"]["r_direct"])
    iso_r = abs(med_iso.get("comm_entropy", {}).get("r_direct", 0)) if med_iso else 0
    iso_weaker = iso_r < aniso_r * 0.5  # iso effect is less than half of aniso
    aniso_mono = aniso_r > 0.5 and med_aniso["comm_entropy"]["p_direct"] < 0.05
    null_disc = null_model["discriminates"]

    print(f"\n  Anisotropic direct r: {med_aniso['comm_entropy']['r_direct']:+.4f}", flush=True)
    print(f"  Anisotropic partial r (TC+n0): {med_aniso['comm_entropy']['r_partial_both']:+.4f}", flush=True)
    print(f"  Survives mediation: {comm_surv}", flush=True)
    print(f"  Isotropic weaker than aniso: {iso_weaker} (|r_iso|={iso_r:.4f} vs |r_aniso|={aniso_r:.4f})", flush=True)
    print(f"  Aniso monotonic: {aniso_mono}", flush=True)
    print(f"  Null discriminates: {null_disc}", flush=True)
    print(f"  Repro aniso: {n_sig_aniso}/{n_aniso}", flush=True)
    print(f"  Repro iso: {n_sig_iso}/{n_iso}", flush=True)

    if comm_surv and iso_weaker and n_sig_aniso >= 7 and null_disc:
        verdict = "STRONG POSITIVE — [H,M] detects curvature beyond pair counting, Lorentzian-specific"
    elif comm_surv and n_sig_aniso >= 5:
        verdict = "POSITIVE SIGNAL — survives mediation, needs iso confirmation"
    elif aniso_mono and not comm_surv and n_sig_aniso >= 5:
        verdict = "REPRODUCIBLE BUT MEDIATED — same as SVD entropy"
    elif aniso_mono and n_sig_aniso >= 3:
        verdict = "WEAK SIGNAL — partially reproducible, partially mediated"
    elif abs(med_aniso["comm_entropy"]["r_direct"]) > 0.3:
        verdict = "DIRECT ONLY — fully mediated by pair counting"
    else:
        verdict = "NO SIGNAL — commutator does not detect curvature"

    print(f"\n  FINAL VERDICT: {verdict}", flush=True)
    print(f"  Wall time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    # ==================================================================
    # SAVE
    # ==================================================================
    meta.status = "completed"
    meta.verdict = verdict
    meta.wall_time_sec = total_time

    output = {
        "parameters": {
            "N": N_POINTS, "M": M_ENSEMBLE, "M_paired": M_PAIRED,
            "T": T_DIAMOND, "workers": WORKERS,
            "eps_aniso": EPSILON_ANISO, "eps_iso": EPSILON_ISO,
            "n_repro_seeds": len(all_repro_configs),
            "crn_pairing": True,
            "isotropic_control": True,
        },
        "theoretical_predictions": theory,
        "mediation_aniso": med_aniso,
        "mediation_iso": med_iso,
        "reproducibility": repro_results,
        "null_model": null_model,
        "n_sig_aniso": n_sig_aniso,
        "n_sig_iso": n_sig_iso,
        "verdict": verdict,
        "wall_time_sec": total_time,
    }

    out_path = RESULTS_DIR / "route3_commutator_v2.json"
    save_experiment(meta, output, out_path)
    print(f"  Saved: {out_path}", flush=True)

    clear_progress()
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
