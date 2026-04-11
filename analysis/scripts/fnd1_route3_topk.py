"""
FND-1 Route 3: Top-k Eigenvalue Analysis of [H, M].

Motivated by v2 finding: comm_frobenius (= sqrt(sum lambda^2)) detects
curvature (p < 0.0001, 4/4 seeds) while comm_entropy does not.
Frobenius is dominated by large eigenvalues. This script identifies
WHICH eigenvalues carry the curvature signal.

Design principles (learned from v2 mistakes):
  - NO SVD computation (saves 35% time)
  - Uses MKL Python for eigendecomp (2.3x faster)
  - Saves RAW eigenvalue distributions for each sprinkling
  - Benchmark BEFORE estimating time
  - Sparse commutator + eigvalsh (verified optimizations)
  - flush=True on all output
  - 80% CPU cap (10 workers)

Run with MKL:
  "C:/Users/youre/miniconda3/envs/sct-mkl/python.exe" analysis/scripts/fnd1_route3_topk.py

Or benchmark only:
  ... analysis/scripts/fnd1_route3_topk.py --benchmark
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
from fnd1_gate5_runner import sprinkle_curved, _sprinkle_flat
from fnd1_experiment_registry import (
    ExperimentMeta, update_progress, clear_progress, save_experiment, RESULTS_DIR,
)
from fnd1_parallel import N_WORKERS, _init_worker

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_VALUES = [1000, 2000, 3000, 5000]  # finite-size scaling: 4 N values
M_ENSEMBLE = 80                      # per epsilon per N (increased)
T_DIAMOND = 1.0
MASTER_SEED = 42
EPSILON_VALUES = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
TOP_K_VALUES = [10, 25, 50, 100, 200, 500]
N_PERMUTATIONS = 500             # permutation test for non-parametric p
WORKERS = N_WORKERS              # 10 (80% cap)

# Pre-registered primary hypothesis (stated BEFORE seeing data):
# "The normalized Frobenius norm (frobenius/total_causal) of [H,M]
#  shows a quadratic response to eps (r(eps^2, norm_frob) significant)
#  that survives partial correlation controlling for total_causal."
# Everything else is secondary/exploratory.
PRIMARY_METRIC = "norm_frobenius"
PRIMARY_MODE = "quadratic"


# ---------------------------------------------------------------------------
# Worker: compute [H,M] eigenvalues and return RAW (no compression)
# ---------------------------------------------------------------------------

def _worker_eigenvalues(args):
    """Compute [H,M] eigenvalues for one sprinkling. Returns full array."""
    seed_int, N, T, eps = args[:4]
    V = T**2 / 2.0
    rho = N / V

    rng = np.random.default_rng(seed_int)
    if eps == 0.0:
        pts, C = _sprinkle_flat(N, T, rng)
    else:
        pts, C = sprinkle_curved(N, eps, T, rng)

    n_mat = compute_interval_cardinalities(C)
    L = build_bd_L(C, n_mat, rho)

    # [H,M] = (L^TL - LL^T)/2, sparse (6.4x faster)
    L_sp = sp.csr_matrix(L)
    comm = ((L_sp.T @ L_sp - L_sp @ L_sp.T) / 2.0).toarray()

    # eigvalsh: values only, 2.8x faster, sorted ascending
    eigs = np.linalg.eigvalsh(comm)

    # NO SVD (saves 35% time)
    total_causal = float(np.sum(C))

    # BD action: S_BD = N - 2*N1 + 4*N2 - 2*N3 (known curvature estimator)
    past = C.T
    n_past = n_mat.T
    N1 = int(np.sum((past > 0) & (n_past == 0)))
    N2 = int(np.sum((past > 0) & (n_past == 1)))
    N3 = int(np.sum((past > 0) & (n_past == 2)))
    bd_action = N - 2*N1 + 4*N2 - 2*N3

    # Graph Laplacian (Route 5): undirected causal graph, L_G = D - A
    # Zero extra cost: uses same C matrix, eigvalsh already optimized
    A_undir = ((C + C.T) > 0).astype(np.float64)  # undirected adjacency
    degrees = np.sum(A_undir, axis=1)
    L_graph = np.diag(degrees) - A_undir
    graph_eigs = np.linalg.eigvalsh(L_graph)
    # Fiedler value (2nd smallest eigenvalue): connectivity measure
    sorted_ge = np.sort(graph_eigs)
    fiedler = float(sorted_ge[1]) if len(sorted_ge) > 1 else 0.0
    graph_spectral_gap = float(sorted_ge[-1] - sorted_ge[1]) if len(sorted_ge) > 1 else 0.0

    return {
        "eigs": eigs,  # [H,M] eigenvalues
        "graph_eigs": graph_eigs,  # graph Laplacian eigenvalues
        "fiedler": fiedler,
        "graph_spectral_gap": graph_spectral_gap,
        "total_causal": total_causal,
        "bd_action": float(bd_action),
        "eps": eps,
    }


# ---------------------------------------------------------------------------
# Top-k analysis functions
# ---------------------------------------------------------------------------

def topk_stats(eigs, k):
    """Compute statistics on top-k eigenvalues (by absolute value)."""
    abs_eigs = np.abs(eigs)
    idx = np.argsort(abs_eigs)[::-1]  # descending by |lambda|

    topk = eigs[idx[:k]]
    abs_topk = np.abs(topk)
    sum_topk = float(np.sum(abs_topk))
    sum_total = float(np.sum(abs_eigs))

    return {
        "topk_sum": sum_topk,
        "topk_fraction": sum_topk / sum_total if sum_total > 0 else 0,
        "topk_max": float(np.max(abs_topk)),
        "topk_mean": float(np.mean(abs_topk)),
        "topk_frobenius": float(np.sqrt(np.sum(topk**2))),
    }


def full_spectrum_stats(eigs, total_causal):
    """
    Additional observables on the FULL eigenvalue spectrum.
    Zero extra eigendecomp cost — uses the same eigenvalues.
    """
    abs_eigs = np.abs(eigs)
    N = len(eigs)

    # Participation ratio: (sum|l|^2)^2 / (N * sum|l|^4)
    # Measures how many eigenvalues contribute. More sensitive than entropy.
    sum2 = np.sum(abs_eigs**2)
    sum4 = np.sum(abs_eigs**4)
    ipr = (sum2**2) / (N * sum4) if sum4 > 0 else 0.0

    # Signed sums: positive and negative eigenvalues separately
    pos_sum = float(np.sum(eigs[eigs > 0]))
    neg_sum = float(np.sum(eigs[eigs < 0]))
    # Asymmetry ratio: how much + vs - differ (beyond tracelessness)
    pos_abs = float(np.sum(np.abs(eigs[eigs > 0])))
    neg_abs = float(np.sum(np.abs(eigs[eigs < 0])))
    sign_asymmetry = (pos_abs - neg_abs) / (pos_abs + neg_abs) if (pos_abs + neg_abs) > 0 else 0

    # Spectral gap
    spectral_gap = float(np.max(abs_eigs) - np.min(abs_eigs))

    # Normalized frobenius: frobenius / total_causal
    frob = float(np.sqrt(sum2))
    norm_frob = frob / total_causal if total_causal > 0 else 0

    # Entropy (for comparison with v2)
    s = np.sum(abs_eigs)
    if s > 0:
        p = abs_eigs / s
        entropy = float(-np.sum(p * np.log(p + 1e-300)))
    else:
        entropy = 0.0

    # Full frobenius (unnormalized)
    frobenius = frob

    return {
        "ipr": float(ipr),
        "sign_asymmetry": float(sign_asymmetry),
        "spectral_gap": float(spectral_gap),
        "norm_frobenius": float(norm_frob),
        "entropy": float(entropy),
        "frobenius": float(frobenius),
    }


def mediation_analysis(eps_arr, obs_arr, tc_arr, bd_arr=None, n_perm=0):
    """
    Partial correlation: obs vs eps, controlling total_causal (and optionally BD action).
    Tests BOTH linear (r vs eps) and quadratic (r vs eps^2) responses.
    Includes: effect size (Cohen's d), permutation p-value, cross-validation.
    If bd_arr is provided, also controls for BD action (the known curvature estimator).
    """
    eps2_arr = eps_arr**2

    def resid(x, ctrl):
        sl, ic, _, _, _ = stats.linregress(ctrl, x)
        return x - (sl * ctrl + ic)

    def partial_r(x, y, ctrl):
        xr, yr = resid(x, ctrl), resid(y, ctrl)
        if np.std(xr) > 0 and np.std(yr) > 0:
            return stats.pearsonr(xr, yr)
        return 0.0, 1.0

    def partial_r_multi(x, y, ctrls):
        """Partial r controlling for multiple variables via OLS residualization."""
        X_ctrl = np.column_stack([*ctrls, np.ones(len(x))])
        bx = np.linalg.lstsq(X_ctrl, x, rcond=None)[0]
        by = np.linalg.lstsq(X_ctrl, y, rcond=None)[0]
        xr = x - X_ctrl @ bx
        yr = y - X_ctrl @ by
        if np.std(xr) > 0 and np.std(yr) > 0:
            return stats.pearsonr(xr, yr)
        return 0.0, 1.0

    # Direct correlations
    r_direct, p_direct = stats.pearsonr(eps_arr, obs_arr)
    r_quad, p_quad = stats.pearsonr(eps2_arr, obs_arr)

    # Partial correlations controlling TC
    r_partial, p_partial = partial_r(eps_arr, obs_arr, tc_arr)
    r_partial_quad, p_partial_quad = partial_r(eps2_arr, obs_arr, tc_arr)

    # Effect size: Cohen's d between extreme groups
    obs_flat = obs_arr[np.abs(eps_arr) < 0.01]
    obs_curved = obs_arr[np.abs(eps_arr) > 0.4]
    if len(obs_flat) > 2 and len(obs_curved) > 2:
        pooled_std = np.sqrt((np.var(obs_flat, ddof=1) + np.var(obs_curved, ddof=1)) / 2)
        cohens_d = (np.mean(obs_curved) - np.mean(obs_flat)) / pooled_std if pooled_std > 0 else 0
    else:
        cohens_d = 0.0

    # Cross-validation: split in half, check consistency
    n = len(eps_arr)
    half = n // 2
    r_cv1, _ = partial_r(eps2_arr[:half], obs_arr[:half], tc_arr[:half])
    r_cv2, _ = partial_r(eps2_arr[half:], obs_arr[half:], tc_arr[half:])
    cv_consistent = (r_cv1 * r_cv2 > 0)  # same sign in both halves

    # Permutation test (non-parametric p-value)
    p_perm_quad = 1.0
    if n_perm > 0:
        observed_r = abs(r_partial_quad)
        rng = np.random.default_rng(42)
        count_ge = 0
        for _ in range(n_perm):
            perm_idx = rng.permutation(n)
            eps2_perm = eps2_arr[perm_idx]
            r_perm, _ = partial_r(eps2_perm, obs_arr, tc_arr)
            if abs(r_perm) >= observed_r:
                count_ge += 1
        p_perm_quad = (count_ge + 1) / (n_perm + 1)

    # BD action control: does signal survive beyond the KNOWN curvature estimator?
    r_partial_quad_bd = 0.0
    p_partial_quad_bd = 1.0
    if bd_arr is not None and len(bd_arr) == n:
        r_partial_quad_bd, p_partial_quad_bd = partial_r_multi(
            eps2_arr, obs_arr, [tc_arr, bd_arr]
        )

    return {
        "r_direct": float(r_direct), "p_direct": float(p_direct),
        "r_quad": float(r_quad), "p_quad": float(p_quad),
        "r_partial": float(r_partial), "p_partial": float(p_partial),
        "r_partial_quad": float(r_partial_quad), "p_partial_quad": float(p_partial_quad),
        "r_partial_quad_bd": float(r_partial_quad_bd), "p_partial_quad_bd": float(p_partial_quad_bd),
        "cohens_d": float(cohens_d),
        "cv_r1": float(r_cv1), "cv_r2": float(r_cv2), "cv_consistent": bool(cv_consistent),
        "p_perm_quad": float(p_perm_quad),
    }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    """Benchmark one sprinkling at each N to give accurate estimates."""
    print("=== BENCHMARK ===", flush=True)
    times_by_N = {}
    for N in N_VALUES:
        rng = np.random.default_rng(42)
        V = T_DIAMOND**2 / 2.0
        rho = N / V

        t0 = time.perf_counter()
        pts, C = _sprinkle_flat(N, T_DIAMOND, rng)
        n_mat = compute_interval_cardinalities(C)
        L = build_bd_L(C, n_mat, rho)
        L_sp = sp.csr_matrix(L)
        comm = ((L_sp.T @ L_sp - L_sp @ L_sp.T) / 2.0).toarray()
        eigs = np.linalg.eigvalsh(comm)
        total = time.perf_counter() - t0

        n_tasks = len(EPSILON_VALUES) * M_ENSEMBLE
        par_time = n_tasks * total / WORKERS
        print(f"  N={N}: {total:.3f}s/task, {n_tasks} tasks, "
              f"parallel: {par_time/60:.1f} min, with margin: {par_time*1.5/60:.1f} min", flush=True)
        times_by_N[N] = total

    total_all = sum(len(EPSILON_VALUES) * M_ENSEMBLE * t / WORKERS for t in times_by_N.values())
    print(f"\n  TOTAL all N values: {total_all/60:.1f} min (with 1.5x: {total_all*1.5/60:.1f} min)", flush=True)
    print(flush=True)
    return times_by_N


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    N_PRIMARY = max(N_VALUES)  # primary analysis at largest N

    meta = ExperimentMeta(
        route=3, name="route3_topk",
        description=f"Top-k eigenvalue analysis + finite-size scaling. Primary: {PRIMARY_METRIC} ({PRIMARY_MODE})",
        N=N_PRIMARY, M=M_ENSEMBLE, status="running",
    )

    print("=" * 70, flush=True)
    print("FND-1 ROUTE 3: TOP-K EIGENVALUE ANALYSIS", flush=True)
    print("=" * 70, flush=True)
    print(f"N values: {N_VALUES} (primary: {N_PRIMARY})", flush=True)
    print(f"M={M_ENSEMBLE}, workers={WORKERS}", flush=True)
    print(f"Epsilon: {EPSILON_VALUES}", flush=True)
    print(f"Top-k values: {TOP_K_VALUES}", flush=True)
    print(f"Pre-registered primary: {PRIMARY_METRIC} ({PRIMARY_MODE})", flush=True)
    print(f"Permutation tests: {N_PERMUTATIONS}", flush=True)
    print(f"NO SVD", flush=True)
    print(flush=True)

    # Benchmark first
    times_by_N = run_benchmark()

    ss = np.random.SeedSequence(MASTER_SEED)
    N_POINTS = N_PRIMARY  # for backward compatibility with analysis sections

    # ==================================================================
    # COMPUTE: eigenvalues at each epsilon
    # ==================================================================

    all_data = {}  # eps -> list of {eigs, total_causal}

    for eps in EPSILON_VALUES:
        eps_ss = ss.spawn(1)[0]
        child_seeds = eps_ss.spawn(M_ENSEMBLE)
        seed_ints = [int(cs.generate_state(1)[0]) for cs in child_seeds]
        args = [(si, N_POINTS, T_DIAMOND, eps) for si in seed_ints]

        print(f"  eps={eps:+.3f}: {M_ENSEMBLE} sprinklings...", flush=True)
        t0 = time.perf_counter()
        with Pool(WORKERS, initializer=_init_worker) as pool:
            results = pool.map(_worker_eigenvalues, args)
        elapsed = time.perf_counter() - t0
        print(f"    Done in {elapsed:.1f}s", flush=True)

        all_data[eps] = results

        # Quick summary
        tc = [r["total_causal"] for r in results]
        frobs = [np.sqrt(np.sum(r["eigs"]**2)) for r in results]
        print(f"    total_causal: {np.mean(tc):.1f}", flush=True)
        print(f"    frobenius: {np.mean(frobs):.2f} +/- {np.std(frobs)/np.sqrt(M_ENSEMBLE):.2f}", flush=True)

        update_progress(meta, step=f"eps={eps:+.2f}",
                       pct=(EPSILON_VALUES.index(eps)+1)/len(EPSILON_VALUES) * 0.6)

    # ==================================================================
    # FULL SPECTRUM STATS (IPR, signed, gap, normalized frobenius)
    # ==================================================================

    print(f"\n{'='*60}", flush=True)
    print("FULL SPECTRUM ANALYSIS", flush=True)
    print("=" * 60, flush=True)

    update_progress(meta, step="Full spectrum analysis", pct=0.65)

    full_metrics = ["ipr", "sign_asymmetry", "spectral_gap", "norm_frobenius", "entropy", "frobenius",
                     "fiedler", "graph_spectral_gap"]
    full_eps_all = []
    full_tc_all = []
    full_bd_all = []
    full_stats_all = {m: [] for m in full_metrics}

    for eps in EPSILON_VALUES:
        for r in all_data[eps]:
            fs = full_spectrum_stats(r["eigs"], r["total_causal"])
            full_eps_all.append(eps)
            full_tc_all.append(r["total_causal"])
            full_bd_all.append(r["bd_action"])
            for m in full_metrics:
                if m in fs:
                    full_stats_all[m].append(fs[m])
                elif m in r:
                    full_stats_all[m].append(r[m])
                else:
                    full_stats_all[m].append(0.0)

    full_eps = np.array(full_eps_all)
    full_tc = np.array(full_tc_all)
    full_bd = np.array(full_bd_all)

    full_med_results = {}
    print(f"\n  {'metric':>18} {'q_r':>6} {'p_q|TC':>9} {'p_q|TC+BD':>10} {'d':>6} {'cv':>3} {'perm':>8} {'surv':>6}", flush=True)
    print(f"  {'-'*18} {'-'*6} {'-'*9} {'-'*10} {'-'*6} {'-'*3} {'-'*8} {'-'*6}", flush=True)

    for metric in full_metrics:
        obs = np.array(full_stats_all[metric])
        n_p = N_PERMUTATIONS if metric == PRIMARY_METRIC else 0
        med = mediation_analysis(full_eps, obs, full_tc, bd_arr=full_bd, n_perm=n_p)
        full_med_results[metric] = med
        surv_tc = abs(med["r_partial_quad"]) > 0.1 and med["p_partial_quad"] < 0.10
        surv_bd = abs(med["r_partial_quad_bd"]) > 0.1 and med["p_partial_quad_bd"] < 0.10
        cv = "Y" if med["cv_consistent"] else "N"
        perm = f"{med['p_perm_quad']:.4f}" if med["p_perm_quad"] < 1.0 else ""
        label = "TC+BD" if surv_bd else ("TC" if surv_tc else "")
        primary = " <--" if metric == PRIMARY_METRIC else ""
        print(f"  {metric:>18} {med['r_quad']:+6.3f} {med['p_partial_quad']:9.2e} "
              f"{med['p_partial_quad_bd']:10.2e} {med['cohens_d']:+6.2f} {cv:>3} {perm:>8} "
              f"{label:>6}{primary}", flush=True)

    # Group means table
    print(f"\n  Group means:", flush=True)
    print(f"  {'eps':>6} {'entropy':>10} {'frobenius':>12} {'IPR':>8} {'norm_frob':>10} {'sign_asym':>10} {'gap':>10} {'TC':>10}", flush=True)
    for eps in EPSILON_VALUES:
        d = [i for i, e in enumerate(full_eps_all) if abs(e - eps) < 0.01]
        if d:
            print(f"  {eps:+6.3f} "
                  f"{np.mean([full_stats_all['entropy'][i] for i in d]):10.6f} "
                  f"{np.mean([full_stats_all['frobenius'][i] for i in d]):12.2f} "
                  f"{np.mean([full_stats_all['ipr'][i] for i in d]):8.5f} "
                  f"{np.mean([full_stats_all['norm_frobenius'][i] for i in d]):10.6f} "
                  f"{np.mean([full_stats_all['sign_asymmetry'][i] for i in d]):10.6f} "
                  f"{np.mean([full_stats_all['spectral_gap'][i] for i in d]):10.2f} "
                  f"{np.mean([full_tc_all[i] for i in d]):10.0f}", flush=True)

    # ==================================================================
    # ANALYSIS: top-k at each k
    # ==================================================================

    print(f"\n{'='*60}", flush=True)
    print("TOP-K ANALYSIS", flush=True)
    print("=" * 60, flush=True)

    update_progress(meta, step="Top-k analysis", pct=0.7)

    topk_results = {}

    for k in TOP_K_VALUES:
        print(f"\n  k = {k}:", flush=True)

        # Compute top-k stats for each sprinkling
        eps_all = []
        tc_all = []
        stats_by_metric = {m: [] for m in ["topk_sum", "topk_fraction", "topk_max",
                                             "topk_mean", "topk_frobenius"]}

        for eps in EPSILON_VALUES:
            for r in all_data[eps]:
                eps_all.append(eps)
                tc_all.append(r["total_causal"])
                tk = topk_stats(r["eigs"], k)
                for m in stats_by_metric:
                    stats_by_metric[m].append(tk[m])

        eps_arr = np.array(eps_all)
        tc_arr = np.array(tc_all)

        # Mediation for each top-k metric
        k_med = {}
        for metric_name, vals in stats_by_metric.items():
            obs_arr = np.array(vals)
            med = mediation_analysis(eps_arr, obs_arr, tc_arr)
            k_med[metric_name] = med
            surv_lin = abs(med["r_partial"]) > 0.1 and med["p_partial"] < 0.10
            surv_quad = abs(med["r_partial_quad"]) > 0.1 and med["p_partial_quad"] < 0.10
            surv = surv_lin or surv_quad
            marker = "LIN-SURVIVES" if surv_lin else ("QUAD-SURVIVES" if surv_quad else "mediated")
            print(f"    {metric_name:20s}: lin r={med['r_direct']:+.4f}, quad r={med['r_quad']:+.4f}, "
                  f"partial_lin={med['r_partial']:+.4f}, partial_quad={med['r_partial_quad']:+.4f} "
                  f"[{marker}]", flush=True)

        topk_results[k] = k_med

    # ==================================================================
    # KEY COMPARISON: at which k does signal survive mediation?
    # ==================================================================

    print(f"\n{'='*60}", flush=True)
    print("SIGNAL SURVIVAL BY K", flush=True)
    print("=" * 60, flush=True)

    print(f"\n  {'k':>5} {'metric':>20} {'lin_r':>8} {'quad_r':>8} {'part_lin':>9} {'part_quad':>10} {'survives':>12}", flush=True)
    print(f"  {'-'*5} {'-'*20} {'-'*8} {'-'*8} {'-'*9} {'-'*10} {'-'*12}", flush=True)

    any_survives = False
    best_k = None
    best_metric = None
    best_partial_r = 0
    best_mode = None  # "linear" or "quadratic"

    for k in TOP_K_VALUES:
        for metric_name, med in topk_results[k].items():
            surv_lin = abs(med["r_partial"]) > 0.1 and med["p_partial"] < 0.10
            surv_quad = abs(med["r_partial_quad"]) > 0.1 and med["p_partial_quad"] < 0.10
            surv = surv_lin or surv_quad

            if surv:
                any_survives = True
                # Track best across both modes
                for r_val, mode in [(med["r_partial"], "linear"),
                                    (med["r_partial_quad"], "quadratic")]:
                    if abs(r_val) > abs(best_partial_r):
                        best_partial_r = r_val
                        best_k = k
                        best_metric = metric_name
                        best_mode = mode

            surv_label = "LIN" if surv_lin else ("QUAD" if surv_quad else "")
            print(f"  {k:5d} {metric_name:>20} {med['r_direct']:+8.4f} {med['r_quad']:+8.4f} "
                  f"{med['r_partial']:+9.4f} {med['r_partial_quad']:+10.4f} {surv_label:>12}", flush=True)

    # ==================================================================
    # WASSERSTEIN DISTANCE: full distribution comparison
    # ==================================================================

    print(f"\n{'='*60}", flush=True)
    print("WASSERSTEIN DISTANCE: full eigenvalue distribution comparison", flush=True)
    print("=" * 60, flush=True)

    update_progress(meta, step="Wasserstein analysis", pct=0.85)

    # Compare each curved epsilon against flat (eps=0)
    flat_eigs_all = [r["eigs"] for r in all_data[0.0]]
    # Pool flat eigenvalues into one reference distribution
    flat_pool = np.sort(np.concatenate(flat_eigs_all))

    wasserstein_results = {}
    print(f"\n  {'eps':>6} {'W1 distance':>12} {'KS stat':>10} {'KS p':>10} {'significant':>12}", flush=True)
    print(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*10} {'-'*12}", flush=True)

    for eps in EPSILON_VALUES:
        curved_eigs_all = [r["eigs"] for r in all_data[eps]]
        curved_pool = np.sort(np.concatenate(curved_eigs_all))

        w1 = float(stats.wasserstein_distance(flat_pool, curved_pool))
        ks_stat, ks_p = stats.ks_2samp(flat_pool, curved_pool)

        sig = "**" if ks_p < 0.01 else ""
        print(f"  {eps:+6.3f} {w1:12.4f} {ks_stat:10.6f} {ks_p:10.2e} {sig:>12}", flush=True)

        wasserstein_results[str(eps)] = {
            "w1": w1,
            "ks_stat": float(ks_stat),
            "ks_p": float(ks_p),
        }

    # Per-sprinkling Wasserstein: paired comparison (each curved vs random flat)
    # This gives a distribution of W1 values for mediation
    print(f"\n  Per-sprinkling Wasserstein (paired, for mediation):", flush=True)

    eps_w1_all = []
    tc_w1_all = []
    w1_all = []

    for eps in EPSILON_VALUES:
        for i, r in enumerate(all_data[eps]):
            # Compare this sprinkling's eigenvalues against a random flat sprinkling
            flat_ref = flat_eigs_all[i % len(flat_eigs_all)]
            w1_i = float(stats.wasserstein_distance(r["eigs"], flat_ref))
            eps_w1_all.append(eps)
            tc_w1_all.append(r["total_causal"])
            w1_all.append(w1_i)

    eps_w1 = np.array(eps_w1_all)
    tc_w1 = np.array(tc_w1_all)
    w1_arr = np.array(w1_all)

    w1_med = mediation_analysis(eps_w1, w1_arr, tc_w1)
    w1_surv_lin = abs(w1_med["r_partial"]) > 0.1 and w1_med["p_partial"] < 0.10
    w1_surv_quad = abs(w1_med["r_partial_quad"]) > 0.1 and w1_med["p_partial_quad"] < 0.10
    w1_surv = w1_surv_lin or w1_surv_quad
    w1_label = "LIN-SURVIVES" if w1_surv_lin else ("QUAD-SURVIVES" if w1_surv_quad else "mediated")
    print(f"    W1: lin r={w1_med['r_direct']:+.4f}, quad r={w1_med['r_quad']:+.4f}, "
          f"partial_lin={w1_med['r_partial']:+.4f}, partial_quad={w1_med['r_partial_quad']:+.4f} "
          f"[{w1_label}]", flush=True)

    # ==================================================================
    # FINITE-SIZE SCALING (run at smaller N values)
    # ==================================================================

    print(f"\n{'='*60}", flush=True)
    print("FINITE-SIZE SCALING", flush=True)
    print("=" * 60, flush=True)

    scaling_results = {}
    for N_test in N_VALUES:
        if N_test == N_POINTS:
            # Already computed above, extract primary metric
            primary_data = full_med_results.get(PRIMARY_METRIC, {})
            scaling_results[N_test] = {
                "r_quad": primary_data.get("r_quad", 0),
                "r_partial_quad": primary_data.get("r_partial_quad", 0),
                "cohens_d": primary_data.get("cohens_d", 0),
            }
            continue

        print(f"\n  N={N_test}: {M_ENSEMBLE} sprinklings × {len(EPSILON_VALUES)} eps...", flush=True)
        t0 = time.perf_counter()

        n_ss = ss.spawn(1)[0]
        sc_data = []
        for eps in EPSILON_VALUES:
            e_ss = n_ss.spawn(1)[0]
            child_seeds = e_ss.spawn(M_ENSEMBLE)
            seed_ints = [int(cs.generate_state(1)[0]) for cs in child_seeds]
            args = [(si, N_test, T_DIAMOND, eps) for si in seed_ints]
            with Pool(WORKERS, initializer=_init_worker) as pool:
                results = pool.map(_worker_eigenvalues, args)
            for r in results:
                fs = full_spectrum_stats(r["eigs"], r["total_causal"])
                sc_data.append({"eps": eps, "obs": fs[PRIMARY_METRIC],
                               "tc": r["total_causal"], "bd": r["bd_action"]})

        sc_eps = np.array([d["eps"] for d in sc_data])
        sc_obs = np.array([d["obs"] for d in sc_data])
        sc_tc = np.array([d["tc"] for d in sc_data])
        sc_bd = np.array([d["bd"] for d in sc_data])
        sc_med = mediation_analysis(sc_eps, sc_obs, sc_tc, bd_arr=sc_bd)

        elapsed = time.perf_counter() - t0
        print(f"    Done in {elapsed:.1f}s: quad r={sc_med['r_quad']:+.4f}, "
              f"partial_quad={sc_med['r_partial_quad']:+.4f}, d={sc_med['cohens_d']:+.4f}", flush=True)

        scaling_results[N_test] = {
            "r_quad": sc_med["r_quad"],
            "r_partial_quad": sc_med["r_partial_quad"],
            "cohens_d": sc_med["cohens_d"],
        }

    print(f"\n  Scaling summary ({PRIMARY_METRIC}, {PRIMARY_MODE}):", flush=True)
    print(f"  {'N':>6} {'r_quad':>8} {'partial_quad':>12} {'cohens_d':>10}", flush=True)
    for N_test in sorted(scaling_results.keys()):
        sr = scaling_results[N_test]
        print(f"  {N_test:6d} {sr['r_quad']:+8.4f} {sr['r_partial_quad']:+12.4f} {sr['cohens_d']:+10.4f}", flush=True)

    signal_grows = False
    if len(scaling_results) >= 2:
        Ns = sorted(scaling_results.keys())
        pqs = [abs(scaling_results[n]["r_partial_quad"]) for n in Ns]
        signal_grows = all(pqs[i] <= pqs[i+1] for i in range(len(pqs)-1))

    # ==================================================================
    # MULTIPLE TESTING CORRECTION (Benjamini-Hochberg FDR)
    # ==================================================================

    print(f"\n{'='*60}", flush=True)
    print("MULTIPLE TESTING CORRECTION (BH FDR)", flush=True)
    print("=" * 60, flush=True)

    all_pvals = []
    all_labels = []
    # Collect all p-values from full spectrum + top-k + wasserstein
    for metric, med in full_med_results.items():
        all_pvals.extend([med["p_partial"], med["p_partial_quad"]])
        all_labels.extend([f"full_{metric}_lin", f"full_{metric}_quad"])
    for k, k_meds in topk_results.items():
        for metric, med in k_meds.items():
            all_pvals.extend([med["p_partial"], med["p_partial_quad"]])
            all_labels.extend([f"k{k}_{metric}_lin", f"k{k}_{metric}_quad"])
    all_pvals.extend([w1_med["p_partial"], w1_med["p_partial_quad"]])
    all_labels.extend(["wasserstein_lin", "wasserstein_quad"])

    # BH correction
    n_tests = len(all_pvals)
    sorted_idx = np.argsort(all_pvals)
    bh_threshold = np.array([(i+1) / n_tests * 0.05 for i in range(n_tests)])
    sorted_pvals = np.array(all_pvals)[sorted_idx]
    bh_reject = sorted_pvals <= bh_threshold
    # Find the largest k where p_(k) <= k/m * alpha
    last_reject = -1
    for i in range(n_tests):
        if bh_reject[i]:
            last_reject = i
    bh_significant = np.zeros(n_tests, dtype=bool)
    if last_reject >= 0:
        bh_significant[sorted_idx[:last_reject+1]] = True

    n_bh_sig = int(np.sum(bh_significant))
    print(f"\n  Total tests: {n_tests}", flush=True)
    print(f"  Expected false positives at alpha=0.05: {n_tests*0.05:.1f}", flush=True)
    print(f"  BH-significant at FDR=0.05: {n_bh_sig}", flush=True)
    if n_bh_sig > 0:
        print(f"  Significant tests:", flush=True)
        for i in range(n_tests):
            if bh_significant[i]:
                print(f"    {all_labels[i]}: p={all_pvals[i]:.2e}", flush=True)

    # ==================================================================
    # VERDICT
    # ==================================================================

    total_time = time.perf_counter() - t_total

    print(f"\n{'='*70}", flush=True)
    print("VERDICT", flush=True)
    print("=" * 70, flush=True)

    if any_survives:
        verdict = (f"SIGNAL FOUND at k={best_k}, metric={best_metric}, "
                   f"mode={best_mode}, partial r={best_partial_r:+.4f}")
        print(f"\n  {verdict}", flush=True)
        if best_mode == "quadratic":
            print(f"  QUADRATIC response: |curvature| affects top-{best_k} eigenvalues (even function of eps).", flush=True)
        else:
            print(f"  LINEAR response: curvature direction affects top-{best_k} eigenvalues (odd function of eps).", flush=True)
    elif w1_surv:
        verdict = "WASSERSTEIN SIGNAL: full distribution shifts with curvature beyond pair counting"
        print(f"\n  {verdict}", flush=True)
    else:
        verdict = "NO SIGNAL at any k, distribution level, or functional form (linear or quadratic)"
        print(f"\n  {verdict}", flush=True)

    print(f"  Wall time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    # ==================================================================
    # SAVE (including raw eigenvalue distributions for future use)
    # ==================================================================

    meta.status = "completed"
    meta.verdict = verdict
    meta.wall_time_sec = total_time

    # Save eigenvalue distributions as separate file (can be large)
    eig_save = {}
    for eps in EPSILON_VALUES:
        eig_save[str(eps)] = {
            "eigenvalues": [r["eigs"].tolist() for r in all_data[eps]],
            "graph_eigenvalues": [r["graph_eigs"].tolist() for r in all_data[eps]],
            "total_causal": [r["total_causal"] for r in all_data[eps]],
            "bd_action": [r["bd_action"] for r in all_data[eps]],
        }

    eig_path = RESULTS_DIR / "route3_topk_eigenvalues.json"
    with open(eig_path, "w") as f:
        json.dump(eig_save, f)
    print(f"  Raw eigenvalues saved: {eig_path} ({eig_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    # Save analysis results
    output = {
        "parameters": {
            "N": N_POINTS, "M": M_ENSEMBLE, "T": T_DIAMOND,
            "eps_values": EPSILON_VALUES, "top_k_values": TOP_K_VALUES,
            "workers": WORKERS, "svd": False, "mkl": "mkl" in sys.executable.lower(),
        },
        "full_spectrum_mediation": full_med_results,
        "topk_mediation": {
            str(k): {m: v for m, v in med.items()}
            for k, med in topk_results.items()
        },
        "best_k": best_k,
        "best_metric": best_metric,
        "best_partial_r": best_partial_r,
        "best_mode": best_mode,
        "any_survives": any_survives,
        "wasserstein": wasserstein_results,
        "wasserstein_mediation": w1_med,
        "wasserstein_survives": w1_surv,
        "verdict": verdict,
        "wall_time_sec": total_time,
        "benchmark_sec_by_N": {str(n): t for n, t in times_by_N.items()},
    }

    out_path = RESULTS_DIR / "route3_topk.json"
    save_experiment(meta, output, out_path)
    print(f"  Results saved: {out_path}", flush=True)

    clear_progress()
    print("\nDone.", flush=True)


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        run_benchmark()
    else:
        main()
