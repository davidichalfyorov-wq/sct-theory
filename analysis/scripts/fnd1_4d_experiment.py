"""
FND-1 d=4 Prototype: Curvature Detection Beyond Conformal Invariance.

In d=2, ALL routes failed because causal structure is conformally invariant.
In d=4, the Weyl tensor is nontrivial. The tidal quadrupole metric
g_00 = -(1 + eps*(x^2-y^2)), g_ij = delta_ij is Ricci-flat with pure Weyl
curvature. The causal structure CHANGES with eps.

Tests: Route 2 (link-graph Laplacian), Route 7 (magnetic Laplacian),
basic link statistics. Conformally flat control to separate Weyl from density.

Run with MKL:
  "C:/Users/youre/miniconda3/envs/sct-mkl/python.exe" analysis/scripts/fnd1_4d_experiment.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_experiment_registry import (
    ExperimentMeta, save_experiment, RESULTS_DIR,
)
from fnd1_parallel import N_WORKERS, _init_worker
try:
    from fnd1_gpu import gpu_eigvalsh, gpu_eigh
except ImportError:
    gpu_eigvalsh, gpu_eigh = np.linalg.eigvalsh, np.linalg.eigh

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_VALUES = [500, 1000, 2000]
N_PRIMARY = 2000
M_ENSEMBLE = 120
T_DIAMOND = 1.0
MASTER_SEED = 42
K_VALUES = [2, 5, 10, 20]
EPSILON_TIDAL = [0.0, 0.1, 0.2, 0.3, 0.5]   # pp-wave: TC changes 0-29%
EPSILON_CONFORMAL = [0.2, 0.5]               # density-only control
N_DISTANCE_PAIRS = 8000
N_PERMUTATIONS = 200
WORKERS = N_WORKERS


# ---------------------------------------------------------------------------
# 4D Causal Diamond Sprinkling
# ---------------------------------------------------------------------------

def sprinkle_4d_flat(N, T, rng):
    """Sprinkle N points into 4D Minkowski causal diamond via rejection.

    Diamond: {(t,x,y,z) : |t| + sqrt(x^2+y^2+z^2) < T/2}
    """
    pts = np.empty((N, 4))
    count = 0
    half = T / 2.0
    while count < N:
        batch = max(N - count, 1000) * 10  # oversample for ~13% acceptance
        candidates = rng.uniform(-half, half, size=(batch, 4))
        r = np.sqrt(candidates[:, 1]**2 + candidates[:, 2]**2 + candidates[:, 3]**2)
        inside = np.abs(candidates[:, 0]) + r < half
        valid = candidates[inside]
        n_take = min(len(valid), N - count)
        pts[count:count + n_take] = valid[:n_take]
        count += n_take

    # Sort by time coordinate
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    return pts


def _ppwave_profile(x, y):
    """Harmonic profile: f(x,y) = cos(pi*x)*cosh(pi*y). nabla^2 f = 0."""
    return np.cos(np.pi * x) * np.cosh(np.pi * y)


def sprinkle_4d_ppwave(N, T, rng):
    """Sprinkle into 4D Minkowski diamond. PP-wave has det(g)=-1, uniform density."""
    return sprinkle_4d_flat(N, T, rng)


def sprinkle_4d_conformal(N, T, eps, rng):
    """Sprinkle with pp-wave-like density but flat causal structure.

    Density proportional to (1 + eps*f(x,y)/max_f) to roughly match the
    number of points affected by the pp-wave. Causal structure: flat Minkowski.
    """
    pts = np.empty((N, 4))
    count = 0
    half = T / 2.0
    max_f = float(np.cosh(np.pi * half))  # max of f in diamond

    while count < N:
        batch = max(N - count, 500) * 12
        candidates = rng.uniform(-half, half, size=(batch, 4))
        r = np.sqrt(candidates[:, 1]**2 + candidates[:, 2]**2 + candidates[:, 3]**2)
        inside = np.abs(candidates[:, 0]) + r < half

        x, y = candidates[:, 1], candidates[:, 2]
        f_val = _ppwave_profile(x, y)
        density = 1.0 + 0.1 * eps * f_val / max_f  # mild density modulation
        density = np.maximum(density, 0.1)
        accept_prob = density / (1.0 + 0.1 * abs(eps))
        accepted = inside & (rng.random(batch) < accept_prob)

        valid = candidates[accepted]
        n_take = min(len(valid), N - count)
        pts[count:count + n_take] = valid[:n_take]
        count += n_take

    order = np.argsort(pts[:, 0])
    pts = pts[order]
    return pts


# ---------------------------------------------------------------------------
# 4D Causal Matrix
# ---------------------------------------------------------------------------

def causal_matrix_4d(points, eps=0.0, metric="flat"):
    """Build causal matrix for 4D metric.

    Tidal: g_00 = -(1+eps*(x^2-y^2)), g_ij = delta_ij
    Flat/conformal: standard Minkowski causal condition
    """
    N = len(points)
    t = points[:, 0]
    x = points[:, 1]
    y = points[:, 2]
    z = points[:, 3]

    dt = t[np.newaxis, :] - t[:, np.newaxis]  # dt[i,j] = t_j - t_i
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx**2 + dy**2 + dz**2

    if metric == "tidal" and abs(eps) > 1e-12:
        # PP-wave: ds^2 = eps*f*du^2 - 2*du*dv + dx^2 + dy^2
        # In (t,x,y,z): causal iff dt^2 - dr^2 > eps*f(x_m,y_m)*(dt+dz)^2 / 2
        xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
        ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
        f_mid = _ppwave_profile(xm, ym)
        mink_interval = dt**2 - dr2  # > 0 for Minkowski-timelike
        ppwave_correction = eps * f_mid * (dt + dz)**2 / 2.0
        C = ((mink_interval > ppwave_correction) & (dt > 0)).astype(np.float64)
    else:
        # Minkowski causal condition
        C = ((dt**2 > dr2) & (dt > 0)).astype(np.float64)

    return C


# ---------------------------------------------------------------------------
# Layer Counts and BD Action (d=4)
# ---------------------------------------------------------------------------

def compute_layers_4d(C):
    """Compute interval cardinalities and layer counts for d=4 BD.

    Returns: n_matrix (C@C), n3_matrix (C@C@C), layer counts N0..N3.
    """
    C_sp = sp.csr_matrix(C)
    n_matrix = (C_sp @ C_sp).toarray()

    past = C.T
    n_past = n_matrix.T

    N0 = int(np.sum((past > 0) & (n_past == 0)))  # links
    N1 = int(np.sum((past > 0) & (n_past == 1)))
    N2 = int(np.sum((past > 0) & (n_past == 2)))

    N3 = int(np.sum((past > 0) & (n_past == 3)))

    return n_matrix, N0, N1, N2, N3


def bd_action_4d(N_pts, N0, N1, N2, N3):
    """Compute d=4 BD action: (-4N + 4N0 - 36N1 + 64N2 - 32N3) / sqrt(6)."""
    return (-4 * N_pts + 4 * N0 - 36 * N1 + 64 * N2 - 32 * N3) / np.sqrt(6.0)


# ---------------------------------------------------------------------------
# Link Graph and Spectral Embedding
# ---------------------------------------------------------------------------

def build_link_graph(C, n_matrix):
    """Build link adjacency (Hasse diagram) from causal matrix."""
    past = C.T
    n_past = n_matrix.T
    link_mask = ((past > 0) & (n_past == 0)).astype(np.float64)
    A = link_mask + link_mask.T
    A = (A > 0).astype(np.float64)
    return sp.csr_matrix(A)


def link_spectral_embedding(A_link, k, N):
    """Spectral embedding from link-graph Laplacian."""
    degrees = np.array(A_link.sum(axis=1)).ravel()
    L = sp.diags(degrees) - A_link

    # Dense eigh: robust for d=4 link graphs with near-degenerate spectrum
    L_dense = L.toarray() if sp.issparse(L) else L
    all_evals, all_evecs = gpu_eigh(L_dense)

    # Skip zero eigenvalue(s)
    start = 0
    while start < len(all_evals) and all_evals[start] < 1e-8:
        start += 1
    end = min(start + k, len(all_evals))
    if end <= start:
        return np.zeros((N, 1)), np.array([0.0])

    lams = all_evals[start:end]
    vecs = all_evecs[:, start:end]
    embedding = vecs * np.sqrt(np.maximum(lams, 0))[np.newaxis, :]
    return embedding, lams


# ---------------------------------------------------------------------------
# Magnetic Laplacian
# ---------------------------------------------------------------------------

def build_magnetic_laplacian(C):
    """Build Hermitian magnetic Laplacian: L_mag = D - A_mag."""
    A_mag = 1j * C - 1j * C.T
    A_undir = ((C + C.T) > 0).astype(np.float64)
    degrees = np.sum(A_undir, axis=1)
    L_mag = np.diag(degrees) - A_mag
    return L_mag, A_undir, degrees


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker_4d(args):
    """Compute all observables for one 4D sprinkling."""
    seed_int, N, T, eps, metric, k_values, n_pairs = args

    rng = np.random.default_rng(seed_int)

    # Sprinkle
    if metric == "flat" or (metric == "tidal" and abs(eps) < 1e-12):
        pts = sprinkle_4d_flat(N, T, rng)
        C = causal_matrix_4d(pts, 0.0, "flat")
    elif metric == "tidal":
        pts = sprinkle_4d_ppwave(N, T, rng)  # PP-wave: det=-1, uniform density
        C = causal_matrix_4d(pts, eps, "tidal")
    elif metric == "conformal":
        pts = sprinkle_4d_conformal(N, T, eps, rng)
        C = causal_matrix_4d(pts, 0.0, "flat")

    total_causal = float(np.sum(C))

    # Layers and BD action
    n_matrix, N0, N1, N2, N3 = compute_layers_4d(C)
    V = np.pi * T**4 / 24.0
    rho = N / V
    bd = bd_action_4d(N, N0, N1, N2, N3)

    # Link graph
    A_link = build_link_graph(C, n_matrix)
    mean_link_deg = float(A_link.sum() / N)

    # Route 2: spectral embedding distances
    max_k = max(k_values)
    embedding, link_lams = link_spectral_embedding(A_link, max_k, N)

    rng2 = np.random.default_rng(seed_int + 999)
    idx_i = rng2.integers(0, N, size=n_pairs)
    idx_j = rng2.integers(0, N, size=n_pairs)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    dt = pts[idx_i, 0] - pts[idx_j, 0]
    dx = pts[idx_i, 1] - pts[idx_j, 1]
    dy = pts[idx_i, 2] - pts[idx_j, 2]
    dz = pts[idx_i, 3] - pts[idx_j, 3]
    d_eucl = np.sqrt(dt**2 + dx**2 + dy**2 + dz**2)

    r2_results = {}
    for k in k_values:
        k_eff = min(k, embedding.shape[1])
        emb = embedding[:, :k_eff]
        d_emb = np.sqrt(np.sum((emb[idx_i] - emb[idx_j])**2, axis=1))
        if np.std(d_emb) > 0 and np.std(d_eucl) > 0:
            r_e, p_e = stats.pearsonr(d_emb, d_eucl)
            rho_sp, _ = stats.spearmanr(d_emb, d_eucl)
        else:
            r_e, p_e, rho_sp = float('nan'), 1.0, float('nan')
        r2_results[k] = {"r_eucl": float(r_e), "rho_sp": float(rho_sp)}

    # Route 7: magnetic Laplacian
    L_mag, A_undir, degrees = build_magnetic_laplacian(C)
    eigs_mag = gpu_eigvalsh(L_mag)
    eigs_undir = gpu_eigvalsh(np.diag(degrees) - A_undir)

    sorted_mag = np.sort(eigs_mag)
    sorted_undir = np.sort(eigs_undir)
    fiedler_mag = float(sorted_mag[1]) if len(sorted_mag) > 1 else 0.0
    fiedler_undir = float(sorted_undir[1]) if len(sorted_undir) > 1 else 0.0
    mag_surplus = fiedler_mag - fiedler_undir

    abs_eigs = np.abs(eigs_mag)
    s = np.sum(abs_eigs)
    entropy = float(-np.sum((abs_eigs / s) * np.log(abs_eigs / s + 1e-300))) if s > 0 else 0

    return {
        "eps": eps, "metric": metric,
        "total_causal": total_causal,
        "n_links": N0,
        "mean_link_deg": mean_link_deg,
        "bd_action": float(bd),
        "link_fiedler": float(link_lams[0]) if len(link_lams) > 0 else 0.0,
        "r2_results": r2_results,
        "mag_surplus": float(mag_surplus),
        "mag_fiedler": fiedler_mag,
        "mag_entropy": entropy,
        "mag_frobenius": float(np.sqrt(np.sum(eigs_mag**2))),
    }


# ---------------------------------------------------------------------------
# Mediation
# ---------------------------------------------------------------------------

def mediation_analysis(eps_arr, obs_arr, tc_arr, nlink_arr, bd_arr, n_perm=0):
    """Mediation: linear + quadratic, TC + N_link + BD controls."""
    eps2 = eps_arr**2
    n = len(eps_arr)

    def resid(x, c):
        s, i, _, _, _ = stats.linregress(c, x)
        return x - (s * c + i)

    def partial_r(x, y, c):
        xr, yr = resid(x, c), resid(y, c)
        if np.std(xr) > 0 and np.std(yr) > 0:
            return stats.pearsonr(xr, yr)
        return 0.0, 1.0

    def partial_r_multi(x, y, ctrls):
        X = np.column_stack([*ctrls, np.ones(n)])
        bx = np.linalg.lstsq(X, x, rcond=None)[0]
        by = np.linalg.lstsq(X, y, rcond=None)[0]
        xr, yr = x - X @ bx, y - X @ by
        if np.std(xr) > 0 and np.std(yr) > 0:
            return stats.pearsonr(xr, yr)
        return 0.0, 1.0

    r_lin, p_lin = stats.pearsonr(eps_arr, obs_arr)
    r_quad, p_quad = stats.pearsonr(eps2, obs_arr)

    # Partial: TC only
    r_pl_tc, p_pl_tc = partial_r(eps_arr, obs_arr, tc_arr)
    # Partial: TC + N_link
    r_pl_tn, p_pl_tn = partial_r_multi(eps_arr, obs_arr, [tc_arr, nlink_arr])
    # Partial: TC + N_link + BD
    r_pl_full, p_pl_full = partial_r_multi(eps_arr, obs_arr, [tc_arr, nlink_arr, bd_arr])

    # Same for quadratic
    r_pq_full, p_pq_full = partial_r_multi(eps2, obs_arr, [tc_arr, nlink_arr, bd_arr])

    # Cohen's d
    flat = obs_arr[np.abs(eps_arr) < 0.01]
    curved = obs_arr[np.abs(eps_arr) > 0.3]
    d = 0.0
    if len(flat) > 2 and len(curved) > 2:
        ps = np.sqrt((np.var(flat, ddof=1) + np.var(curved, ddof=1)) / 2)
        d = (np.mean(curved) - np.mean(flat)) / ps if ps > 0 else 0

    # Permutation test
    p_perm = 1.0
    if n_perm > 0:
        obs_r = abs(r_pl_full)
        rng = np.random.default_rng(42)
        cnt = sum(1 for _ in range(n_perm)
                  if abs(partial_r_multi(eps_arr[rng.permutation(n)], obs_arr,
                                         [tc_arr, nlink_arr, bd_arr])[0]) >= obs_r)
        p_perm = (cnt + 1) / (n_perm + 1)

    return {
        "r_lin": float(r_lin), "p_lin": float(p_lin),
        "r_quad": float(r_quad),
        "r_partial_lin_tc": float(r_pl_tc), "p_partial_lin_tc": float(p_pl_tc),
        "r_partial_lin_full": float(r_pl_full), "p_partial_lin_full": float(p_pl_full),
        "r_partial_quad_full": float(r_pq_full), "p_partial_quad_full": float(p_pq_full),
        "cohens_d": float(d), "p_perm": float(p_perm),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    print("=" * 70, flush=True)
    print("FND-1 d=4: CURVATURE DETECTION BEYOND CONFORMAL INVARIANCE", flush=True)
    print("=" * 70, flush=True)
    print(f"Metric: PP-wave, f=cos(pi*x)cosh(pi*y) [exact vacuum, Ricci=0, Weyl!=0]", flush=True)
    print(f"N values: {N_VALUES} (primary: {N_PRIMARY})", flush=True)
    print(f"M={M_ENSEMBLE}, k={K_VALUES}", flush=True)
    print(f"eps_tidal: {EPSILON_TIDAL}", flush=True)
    print(f"eps_conformal: {EPSILON_CONFORMAL} (density-only control)", flush=True)
    print(flush=True)

    # Benchmark
    print("=== BENCHMARK ===", flush=True)
    for N in N_VALUES:
        t0 = time.perf_counter()
        _worker_4d((42, N, T_DIAMOND, 0.0, "flat", K_VALUES, 1000))
        elapsed = time.perf_counter() - t0
        n_tasks = len(EPSILON_TIDAL) * M_ENSEMBLE
        print(f"  N={N}: {elapsed:.3f}s/task, parallel: {n_tasks*elapsed/WORKERS/60:.1f} min",
              flush=True)

    ss = np.random.SeedSequence(MASTER_SEED)

    # ==================================================================
    # PART 1: TIDAL METRIC (Weyl != 0)
    # ==================================================================

    print(f"\n{'='*60}", flush=True)
    print("PART 1: TIDAL METRIC (Weyl != 0, changes causal structure)", flush=True)
    print("=" * 60, flush=True)

    tidal_data = {}
    for eps in EPSILON_TIDAL:
        eps_ss = ss.spawn(1)[0]
        seeds = [int(cs.generate_state(1)[0]) for cs in eps_ss.spawn(M_ENSEMBLE)]
        args = [(si, N_PRIMARY, T_DIAMOND, eps, "tidal", K_VALUES, N_DISTANCE_PAIRS)
                for si in seeds]
        print(f"\n  eps={eps:+.3f}: {M_ENSEMBLE} sprinklings...", flush=True)
        t0 = time.perf_counter()
        with Pool(WORKERS, initializer=_init_worker) as pool:
            results = pool.map(_worker_4d, args)
        print(f"    Done in {time.perf_counter()-t0:.1f}s  "
              f"TC={np.mean([r['total_causal'] for r in results]):.0f}  "
              f"deg={np.mean([r['mean_link_deg'] for r in results]):.1f}  "
              f"surplus={np.mean([r['mag_surplus'] for r in results]):.4f}", flush=True)
        tidal_data[eps] = results

    # ==================================================================
    # PART 2: CONFORMALLY FLAT CONTROL (Weyl = 0)
    # ==================================================================

    print(f"\n{'='*60}", flush=True)
    print("PART 2: CONFORMALLY FLAT CONTROL (Weyl = 0, density only)", flush=True)
    print("=" * 60, flush=True)

    conf_data = {}
    for eps in EPSILON_CONFORMAL:
        eps_ss = ss.spawn(1)[0]
        seeds = [int(cs.generate_state(1)[0]) for cs in eps_ss.spawn(M_ENSEMBLE)]
        args = [(si, N_PRIMARY, T_DIAMOND, eps, "conformal", K_VALUES, N_DISTANCE_PAIRS)
                for si in seeds]
        print(f"\n  eps_conf={eps:+.3f}: {M_ENSEMBLE} sprinklings...", flush=True)
        t0 = time.perf_counter()
        with Pool(WORKERS, initializer=_init_worker) as pool:
            results = pool.map(_worker_4d, args)
        print(f"    Done in {time.perf_counter()-t0:.1f}s", flush=True)
        conf_data[eps] = results

    # ==================================================================
    # ANALYSIS
    # ==================================================================

    print(f"\n{'='*60}", flush=True)
    print("ANALYSIS: TIDAL METRIC", flush=True)
    print("=" * 60, flush=True)

    # Build arrays for mediation
    eps_a = np.array([r["eps"] for eps in EPSILON_TIDAL for r in tidal_data[eps]])
    tc_a = np.array([r["total_causal"] for eps in EPSILON_TIDAL for r in tidal_data[eps]])
    nl_a = np.array([r["n_links"] for eps in EPSILON_TIDAL for r in tidal_data[eps]])
    bd_a = np.array([r["bd_action"] for eps in EPSILON_TIDAL for r in tidal_data[eps]])

    # Route 2: geometry recovery by k
    print(f"\n  Route 2 (link Laplacian): r_euclidean by k", flush=True)
    print(f"  {'eps':>6} {'k':>3} {'r_eucl':>8} {'rho_sp':>8}", flush=True)
    for eps in EPSILON_TIDAL:
        for k in K_VALUES:
            rs = [r["r2_results"][k]["r_eucl"] for r in tidal_data[eps]]
            rhos = [r["r2_results"][k]["rho_sp"] for r in tidal_data[eps]]
            print(f"  {eps:+6.3f} {k:3d} {np.mean(rs):+8.4f} {np.mean(rhos):+8.4f}", flush=True)

    # Route 7 + link stats: mediation
    metrics = {
        "mag_surplus": [r["mag_surplus"] for eps in EPSILON_TIDAL for r in tidal_data[eps]],
        "mag_fiedler": [r["mag_fiedler"] for eps in EPSILON_TIDAL for r in tidal_data[eps]],
        "mag_entropy": [r["mag_entropy"] for eps in EPSILON_TIDAL for r in tidal_data[eps]],
        "mag_frobenius": [r["mag_frobenius"] for eps in EPSILON_TIDAL for r in tidal_data[eps]],
        "link_fiedler": [r["link_fiedler"] for eps in EPSILON_TIDAL for r in tidal_data[eps]],
        "mean_link_deg": [r["mean_link_deg"] for eps in EPSILON_TIDAL for r in tidal_data[eps]],
    }

    # Also add r_eucl at k=2 as a metric
    r_k2_list = [r["r2_results"][2]["r_eucl"] for eps in EPSILON_TIDAL
                 for r in tidal_data[eps]]
    # Replace NaN with 0 for mediation (NaN from constant embedding)
    metrics["r_eucl_k2"] = [0.0 if np.isnan(v) else v for v in r_k2_list]

    PRIMARY = "mag_surplus"
    all_pvals, all_labels = [], []
    med_results = {}

    print(f"\n  {'metric':>15} {'r_lin':>7} {'r|TC':>7} {'r|full':>7} {'p|full':>10} "
          f"{'d':>6} {'perm':>6} {'surv':>5}", flush=True)
    print(f"  {'-'*15} {'-'*7} {'-'*7} {'-'*7} {'-'*10} {'-'*6} {'-'*6} {'-'*5}", flush=True)

    for metric_name, obs_list in metrics.items():
        obs = np.array(obs_list)
        n_p = N_PERMUTATIONS if metric_name == PRIMARY else 0
        med = mediation_analysis(eps_a, obs, tc_a, nl_a, bd_a, n_perm=n_p)
        med_results[metric_name] = med

        all_pvals.extend([med["p_partial_lin_full"], med["p_partial_quad_full"]])
        all_labels.extend([f"{metric_name}_lin", f"{metric_name}_quad"])

        surv = abs(med["r_partial_lin_full"]) > 0.10 and med["p_partial_lin_full"] < 0.10
        perm_s = f"{med['p_perm']:.3f}" if med["p_perm"] < 1.0 else ""
        mark = "YES" if surv else ""
        print(f"  {metric_name:>15} {med['r_lin']:+7.3f} {med['r_partial_lin_tc']:+7.3f} "
              f"{med['r_partial_lin_full']:+7.3f} {med['p_partial_lin_full']:10.2e} "
              f"{med['cohens_d']:+6.2f} {perm_s:>6} {mark:>5}", flush=True)

    # Group means
    print(f"\n  Group means:", flush=True)
    print(f"  {'eps':>6} {'TC':>10} {'links':>8} {'BD':>10} {'surplus':>8} {'deg':>6}", flush=True)
    for eps in EPSILON_TIDAL:
        d = tidal_data[eps]
        print(f"  {eps:+6.3f} {np.mean([r['total_causal'] for r in d]):10.0f} "
              f"{np.mean([r['n_links'] for r in d]):8.0f} "
              f"{np.mean([r['bd_action'] for r in d]):10.1f} "
              f"{np.mean([r['mag_surplus'] for r in d]):+8.4f} "
              f"{np.mean([r['mean_link_deg'] for r in d]):6.1f}", flush=True)

    # Conformal control
    print(f"\n  Conformally flat control:", flush=True)
    for eps in EPSILON_CONFORMAL:
        d = conf_data[eps]
        surplus_c = np.mean([r["mag_surplus"] for r in d])
        tc_c = np.mean([r["total_causal"] for r in d])
        print(f"    eps_conf={eps:+.3f}: TC={tc_c:.0f}, surplus={surplus_c:+.4f}", flush=True)

    # BH correction
    n_tests = len(all_pvals)
    sorted_idx = np.argsort(all_pvals)
    bh_sig = 0
    for i, si in enumerate(sorted_idx):
        if all_pvals[si] <= (i + 1) / n_tests * 0.05:
            bh_sig = i + 1
    print(f"\n  BH correction: {n_tests} tests, {bh_sig} significant at FDR=0.05", flush=True)

    # ==================================================================
    # FINITE-SIZE SCALING
    # ==================================================================

    print(f"\n{'='*60}", flush=True)
    print("FINITE-SIZE SCALING", flush=True)
    print("=" * 60, flush=True)

    scaling = {}
    for N_test in N_VALUES:
        if N_test == N_PRIMARY:
            scaling[N_test] = med_results.get(PRIMARY, {})
            continue

        n_ss = ss.spawn(1)[0]
        sc_data = []
        for eps in EPSILON_TIDAL:
            e_ss = n_ss.spawn(1)[0]
            seeds = [int(cs.generate_state(1)[0]) for cs in e_ss.spawn(M_ENSEMBLE // 2)]
            args = [(si, N_test, T_DIAMOND, eps, "tidal", [2], N_DISTANCE_PAIRS // 2)
                    for si in seeds]
            with Pool(WORKERS, initializer=_init_worker) as pool:
                sc_data.extend(pool.map(_worker_4d, args))

        sc_eps = np.array([r["eps"] for r in sc_data])
        sc_obs = np.array([r["mag_surplus"] for r in sc_data])
        sc_tc = np.array([r["total_causal"] for r in sc_data])
        sc_nl = np.array([r["n_links"] for r in sc_data])
        sc_bd = np.array([r["bd_action"] for r in sc_data])
        sc_med = mediation_analysis(sc_eps, sc_obs, sc_tc, sc_nl, sc_bd)
        scaling[N_test] = sc_med
        print(f"  N={N_test}: r_lin={sc_med['r_lin']:+.4f}, "
              f"r_partial_full={sc_med['r_partial_lin_full']:+.4f}", flush=True)

    # ==================================================================
    # VERDICT
    # ==================================================================

    total_time = time.perf_counter() - t_total

    print(f"\n{'='*70}", flush=True)
    print("VERDICT", flush=True)
    print("=" * 70, flush=True)

    best = med_results.get(PRIMARY, {})
    sig = abs(best.get("r_partial_lin_full", 0)) > 0.10 and best.get("p_partial_lin_full", 1) < 0.10
    conf_surplus = [np.mean([r["mag_surplus"] for r in conf_data[e]]) for e in EPSILON_CONFORMAL]
    tidal_surplus = [np.mean([r["mag_surplus"] for r in tidal_data[e]])
                     for e in EPSILON_TIDAL if abs(e) > 0.3]

    if sig and all(abs(cs) < abs(ts) * 0.5 for cs, ts in zip(conf_surplus, tidal_surplus)):
        verdict = "GENUINE CURVATURE SIGNAL (survives mediation, weaker in conformal control)"
    elif sig:
        verdict = "SIGNAL (survives mediation, conformal control inconclusive)"
    elif abs(best.get("r_lin", 0)) > 0.3:
        verdict = "DIRECT ONLY (mediated by TC+links+BD)"
    else:
        verdict = "NO SIGNAL"

    print(f"\n  {verdict}", flush=True)
    print(f"  Primary ({PRIMARY}): r_lin={best.get('r_lin',0):+.4f}, "
          f"r_partial_full={best.get('r_partial_lin_full',0):+.4f}, "
          f"p={best.get('p_partial_lin_full',1):.2e}", flush=True)
    print(f"  Wall time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    # Save
    meta = ExperimentMeta(route=2, name="d4_prototype",
                          description="d=4 tidal metric, Routes 2+7, conformal control",
                          N=N_PRIMARY, M=M_ENSEMBLE, status="completed", verdict=verdict)
    meta.wall_time_sec = total_time

    output = {
        "parameters": {"N_primary": N_PRIMARY, "M": M_ENSEMBLE,
                       "metric": "tidal: g00=-(1+eps*(x^2-y^2))",
                       "eps_tidal": EPSILON_TIDAL, "eps_conformal": EPSILON_CONFORMAL},
        "mediation": med_results,
        "scaling": {str(n): v for n, v in scaling.items()},
        "verdict": verdict,
        "wall_time_sec": total_time,
    }
    out_path = RESULTS_DIR / "d4_prototype.json"
    save_experiment(meta, output, out_path)
    print(f"  Saved: {out_path}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        for N in N_VALUES:
            t0 = time.perf_counter()
            _worker_4d((42, N, T_DIAMOND, 0.0, "flat", K_VALUES, 1000))
            print(f"N={N}: {time.perf_counter()-t0:.3f}s/task")
    else:
        main()
