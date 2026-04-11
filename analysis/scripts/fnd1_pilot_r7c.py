#!/usr/bin/env python3
"""
COMP-SCAN v7: Topological + algebraic + temporal observables.
N=500, M=20 CRN pairs, pp-wave eps=5 + Schwarzschild eps=0.005.

New observables from SEED agents:
  - DFEE: Directed Flag Euler Excess (chi_flag - N + TC)
  - IPC: Interval Profile Coherence (cosine similarity of C2 columns)
  - TDAT: Temporal Degree Autocorrelation (ordering-based)
  - C4G: 4-Cycle Participation Gini (exploits triangle-free structure)
  - S_vN: Link-Graph Spectral Entropy (von Neumann entropy of Laplacian)
  - SI2: IPR of C2 singular vectors (from incubator)
  - SC2: Chain decay exponent (from incubator)
  - link_degree_kurtosis, link_degree_gini, link_degree_entropy
  - cv_interval_sizes, skew_interval_sizes

Plus positive controls: column_gini_C2, LVA, column_gini_C, link_degree_skew, fan_kurtosis
Plus baselines: tc, link_count, degree_cv

Output: docs/analysis_runs/run_20260326_114955/pilot_results.json

Author: David Alfyorov
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.linalg import norm, svd
from scipy import linalg as la
from scipy.stats import wilcoxon, skew, kurtosis

# ── Parameters ──────────────────────────────────────────────────────
N = 2000
M = 20
T_DIAMOND = 1.0
MASTER_SEED = 77707  # distinct from all prior runs
METRICS = {
    "ppwave_quad": 5.0,
    "schwarzschild": 0.005,
}

RUN_DIR = Path("docs/analysis_runs/run_20260326_114955")
RESULTS_FILE = RUN_DIR / "pilot_results.json"


# ── Sprinkling ──────────────────────────────────────────────────────
def sprinkle(N_target, T, rng):
    pts = []
    while len(pts) < N_target:
        batch = rng.uniform(-T / 2, T / 2, size=(N_target * 8, 4))
        r = np.sqrt(batch[:, 1]**2 + batch[:, 2]**2 + batch[:, 3]**2)
        inside = np.abs(batch[:, 0]) + r < T / 2
        pts.extend(batch[inside].tolist())
    pts = np.array(pts[:N_target])
    return pts[np.argsort(pts[:, 0])]


# ── Causal matrices ────────────────────────────────────────────────
def causal_flat(pts, _eps=0.0):
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        dt = pts[i + 1:, 0] - pts[i, 0]
        dr2 = np.sum((pts[i + 1:, 1:] - pts[i, 1:])**2, axis=1)
        C[i, i + 1:] = (dt**2 > dr2).astype(np.int8)
    return C


def causal_ppwave_quad(pts, eps):
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        dt = pts[i + 1:, 0] - pts[i, 0]
        dx = pts[i + 1:, 1:] - pts[i, 1:]
        dr2 = np.sum(dx**2, axis=1)
        xm = (pts[i + 1:, 1] + pts[i, 1]) / 2
        ym = (pts[i + 1:, 2] + pts[i, 2]) / 2
        dz = dx[:, 2]
        du = dt + dz
        f = xm**2 - ym**2
        interval = dt**2 - dr2 - eps * f * du**2 / 2
        C[i, i + 1:] = (interval > 0).astype(np.int8)
    return C


def causal_schwarzschild(pts, eps):
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        dt = pts[i + 1:, 0] - pts[i, 0]
        dx = pts[i + 1:, 1:] - pts[i, 1:]
        dr2 = np.sum(dx**2, axis=1)
        rm = np.sqrt(np.sum(((pts[i + 1:, 1:] + pts[i, 1:]) / 2)**2, axis=1))
        phi = -eps / (rm + 0.3)
        interval = (1 + 2 * phi) * dt**2 - (1 - 2 * phi) * dr2
        C[i, i + 1:] = (interval > 0).astype(np.int8)
    return C


METRIC_FNS = {
    "ppwave_quad": causal_ppwave_quad,
    "schwarzschild": causal_schwarzschild,
}


# ── Helpers ─────────────────────────────────────────────────────────
def gini_coefficient(values):
    v = np.sort(np.abs(values.astype(np.float64)))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v)) / (n * np.sum(v)) - (n + 1) / n)


def shannon_entropy(arr):
    p = arr / arr.sum() if arr.sum() > 0 else arr
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def build_link_mask(C, C2):
    return (C > 0) & (C2 == 0)


# ── Observable computation ──────────────────────────────────────────
def compute_all_observables(C, pts):
    n = len(C)
    obs = {}

    # ── Pre-computation ──
    Cf = C.astype(np.float64)
    C2 = Cf @ Cf
    tc = int(C.sum())
    obs["tc"] = tc
    obs["n2"] = int(C2.sum())

    # Link graph
    link_mask = build_link_mask(C, C2)
    A_link = link_mask.astype(np.float32) + link_mask.astype(np.float32).T
    degrees = A_link.sum(axis=1).astype(np.float64)
    link_count = int(degrees.sum()) // 2
    obs["link_count"] = link_count
    obs["degree_cv"] = float(degrees.std() / max(degrees.mean(), 1e-10))
    obs["sum_deg_sq"] = float(np.sum(degrees**2))

    # Interval abundances (baselines)
    causal_pairs = np.argwhere(C > 0)
    if len(causal_pairs) > 0:
        interval_sizes = C2[causal_pairs[:, 0], causal_pairs[:, 1]].astype(int)
    else:
        interval_sizes = np.array([0])
    for k in range(4):
        obs[f"I_{k}"] = int(np.sum(interval_sizes == k))
    obs["I_3plus"] = int(np.sum(interval_sizes >= 3))

    # Past and future cone cardinalities
    past_sizes = Cf.sum(axis=0)   # |past(j)| = sum of column j
    future_sizes = Cf.sum(axis=1)  # |future(i)| = sum of row i

    # ══════════════════════════════════════════════════════════════
    # POSITIVE CONTROLS
    # ══════════════════════════════════════════════════════════════

    # column_gini_C2
    col_norms_C2 = np.sqrt(np.sum(C2**2, axis=0))
    obs["column_gini_C2"] = gini_coefficient(col_norms_C2)

    # column_gini_C
    obs["column_gini_C"] = gini_coefficient(np.sqrt(past_sizes))

    # LVA (Link Valence Anisotropy)
    CCT = Cf @ Cf.T
    lva_vals = []
    for x in range(n):
        future_links = np.where(link_mask[x, :])[0]
        if len(future_links) < 2:
            continue
        kplus = CCT[x, future_links]
        mu = kplus.mean()
        if mu > 0:
            lva_vals.append(float(kplus.var() / mu**2))
    obs["lva"] = float(np.mean(lva_vals)) if lva_vals else 0.0

    # link_degree_skew
    obs["link_degree_skew"] = float(skew(degrees)) if len(degrees) > 2 else 0.0

    # fan_kurtosis
    fan_kurt_vals = []
    for x in range(n):
        future_links = np.where(link_mask[x, :])[0]
        if len(future_links) < 4:
            continue
        fan_sizes = future_sizes[future_links]
        if fan_sizes.std() > 0:
            fan_kurt_vals.append(float(kurtosis(fan_sizes, fisher=True)))
    obs["fan_kurtosis"] = float(np.mean(fan_kurt_vals)) if fan_kurt_vals else 0.0

    # ══════════════════════════════════════════════════════════════
    # NEW RUN #007: DFEE (Directed Flag Euler Excess)
    # chi_flag = 1^T (I + C)^{-1} 1. DFEE = chi_flag - N + TC
    # ══════════════════════════════════════════════════════════════
    try:
        IpC = np.eye(n) + Cf
        x_vec = la.solve_triangular(IpC, np.ones(n), lower=False)
        chi_flag = float(x_vec.sum())
        obs["dfee"] = chi_flag - n + tc
    except Exception:
        obs["dfee"] = 0.0

    # ══════════════════════════════════════════════════════════════
    # NEW RUN #007: IPC (Interval Profile Coherence)
    # Mean cosine similarity of C2 column direction vectors
    # ══════════════════════════════════════════════════════════════
    try:
        col_norms = np.sqrt(np.sum(C2**2, axis=0))
        valid = col_norms > 0
        if valid.sum() >= 10:
            C2_valid = C2[:, valid]
            norms_valid = col_norms[valid]
            C2_normed = C2_valid / norms_valid[np.newaxis, :]
            # Gram matrix of normalized columns
            G = C2_normed.T @ C2_normed
            r = G.shape[0]
            # Mean off-diagonal cosine
            total_cos = (G.sum() - r) / (r * (r - 1))
            obs["ipc"] = float(total_cos)
        else:
            obs["ipc"] = 0.0
    except Exception:
        obs["ipc"] = 0.0

    # ══════════════════════════════════════════════════════════════
    # NEW RUN #007: TDAT (Temporal Degree Autocorrelation)
    # ACF of degree sequence in natural labeling
    # ══════════════════════════════════════════════════════════════
    try:
        # Natural labeling = time-ordering (already sorted by t)
        deg_seq = degrees.copy()
        deg_seq -= deg_seq.mean()
        var_d = deg_seq.var()
        if var_d > 0:
            max_lag = min(n // 4, 50)
            acf_energy = 0.0
            for lag in range(1, max_lag + 1):
                corr = np.mean(deg_seq[:-lag] * deg_seq[lag:]) / var_d
                acf_energy += corr ** 2
            obs["tdat"] = float(acf_energy)
        else:
            obs["tdat"] = 0.0
    except Exception:
        obs["tdat"] = 0.0

    # ══════════════════════════════════════════════════════════════
    # NEW RUN #007: C4G (4-Cycle Participation Gini)
    # Per-vertex 4-cycle count in link graph, then Gini
    # ══════════════════════════════════════════════════════════════
    try:
        c4_counts = np.zeros(n)
        A_link_bool = A_link > 0
        for x in range(min(n, 500)):
            neighbors = np.where(A_link_bool[x, :])[0]
            k = len(neighbors)
            if k < 2:
                continue
            # Count common neighbors for each pair of x's neighbors (excl x)
            for ii in range(k):
                a = neighbors[ii]
                for jj in range(ii + 1, k):
                    b = neighbors[jj]
                    # Don't count if a-b are directly connected (that would be a triangle)
                    if A_link_bool[a, b]:
                        continue
                    # Common neighbors of a and b excluding x
                    common = np.sum(A_link_bool[a, :] & A_link_bool[b, :]) - (1 if A_link_bool[a, x] and A_link_bool[b, x] else 0)
                    c4_counts[x] += max(common, 0)
        valid_c4 = c4_counts[degrees >= 2]
        obs["c4g"] = gini_coefficient(valid_c4) if len(valid_c4) >= 5 else 0.0
    except Exception:
        obs["c4g"] = 0.0

    # ══════════════════════════════════════════════════════════════
    # NEW RUN #007: S_vN (Link Spectral Entropy)
    # Von Neumann entropy of normalized link-graph Laplacian
    # ══════════════════════════════════════════════════════════════
    try:
        D_diag = np.diag(degrees)
        L_link = D_diag - A_link.astype(np.float64)
        eigvals = la.eigvalsh(L_link)
        eigvals = eigvals[eigvals > 1e-10]  # nonzero eigenvalues
        trace_L = eigvals.sum()
        if trace_L > 0:
            p = eigvals / trace_L
            s_vn = -np.sum(p * np.log(p))
            obs["spectral_entropy"] = float(s_vn / np.log(max(len(eigvals), 2)))
        else:
            obs["spectral_entropy"] = 0.0
    except Exception:
        obs["spectral_entropy"] = 0.0

    # ══════════════════════════════════════════════════════════════
    # NEW RUN #007: SI-2 (IPR of C2 singular vectors)
    # Average IPR of top-k left singular vectors of C2
    # ══════════════════════════════════════════════════════════════
    try:
        U, S_vals, _ = svd(C2, full_matrices=False)
        # Top 10 singular vectors (or fewer)
        k_top = min(10, len(S_vals))
        S_nonzero = S_vals[:k_top]
        if len(S_nonzero) >= 2:
            iprs = np.array([float(np.sum(U[:, i]**4)) * n for i in range(k_top)])
            obs["ipr_c2_mean"] = float(iprs.mean())
            obs["ipr_c2_gini"] = gini_coefficient(iprs)
        else:
            obs["ipr_c2_mean"] = 0.0
            obs["ipr_c2_gini"] = 0.0
    except Exception:
        obs["ipr_c2_mean"] = 0.0
        obs["ipr_c2_gini"] = 0.0

    # ══════════════════════════════════════════════════════════════
    # NEW RUN #007: SC-2 (Chain decay exponent)
    # Fit log(N_k) vs k for k=2..5
    # ══════════════════════════════════════════════════════════════
    try:
        # N_k = number of k-chains = 1^T C^{k-1} 1
        chain_counts = [tc]  # k=2: number of 2-chains = TC
        Ck = Cf.copy()
        for k in range(3, 7):
            Ck = Ck @ Cf
            chain_counts.append(float(np.sum(Ck)))
            if chain_counts[-1] < 1:
                break
        valid_cc = [(k + 2, c) for k, c in enumerate(chain_counts) if c > 0]
        if len(valid_cc) >= 3:
            ks = np.array([v[0] for v in valid_cc], dtype=float)
            log_nk = np.log(np.array([v[1] for v in valid_cc]))
            # Linear fit: log(N_k) = a + b*k
            coeffs = np.polyfit(ks, log_nk, 1)
            obs["chain_decay_slope"] = float(coeffs[0])
        else:
            obs["chain_decay_slope"] = 0.0
    except Exception:
        obs["chain_decay_slope"] = 0.0

    # ══════════════════════════════════════════════════════════════
    # NEW RUN #007: Degree distribution statistics
    # ══════════════════════════════════════════════════════════════
    obs["link_degree_kurtosis"] = float(kurtosis(degrees, fisher=True)) if len(degrees) > 3 else 0.0
    obs["link_degree_gini"] = gini_coefficient(degrees)

    # Degree entropy
    if degrees.sum() > 0:
        deg_hist, _ = np.histogram(degrees, bins=np.arange(degrees.max() + 2) - 0.5)
        obs["link_degree_entropy"] = shannon_entropy(deg_hist[deg_hist > 0].astype(float))
    else:
        obs["link_degree_entropy"] = 0.0

    # ══════════════════════════════════════════════════════════════
    # NEW RUN #007: Interval size distribution statistics
    # ══════════════════════════════════════════════════════════════
    if len(interval_sizes) > 5 and interval_sizes.mean() > 0:
        obs["cv_interval_sizes"] = float(interval_sizes.std() / max(interval_sizes.mean(), 1e-10))
        obs["skew_interval_sizes"] = float(skew(interval_sizes.astype(float)))
        obs["kurtosis_interval_sizes"] = float(kurtosis(interval_sizes.astype(float), fisher=True))
        # Interval Fano factor
        obs["interval_fano"] = float(interval_sizes.var() / max(interval_sizes.mean(), 1e-10))
    else:
        obs["cv_interval_sizes"] = 0.0
        obs["skew_interval_sizes"] = 0.0
        obs["kurtosis_interval_sizes"] = 0.0
        obs["interval_fano"] = 0.0

    # ══════════════════════════════════════════════════════════════
    # NEW RUN #007: Assortativity (degree correlation)
    # ══════════════════════════════════════════════════════════════
    try:
        edges = np.argwhere(np.triu(A_link > 0, k=1))
        if len(edges) > 5:
            d_i = degrees[edges[:, 0]]
            d_j = degrees[edges[:, 1]]
            mu_d = (d_i + d_j).mean() / 2
            var_d = ((d_i - mu_d)**2 + (d_j - mu_d)**2).mean() / 2
            if var_d > 0:
                obs["assortativity"] = float(np.mean((d_i - mu_d) * (d_j - mu_d)) / var_d)
            else:
                obs["assortativity"] = 0.0
        else:
            obs["assortativity"] = 0.0
    except Exception:
        obs["assortativity"] = 0.0

    # ══════════════════════════════════════════════════════════════
    # NEW RUN #007: Average neighbor degree
    # ══════════════════════════════════════════════════════════════
    try:
        avg_neigh_deg = np.zeros(n)
        for x in range(n):
            nbrs = np.where(A_link_bool[x, :])[0] if 'A_link_bool' in dir() else np.where(A_link[x, :] > 0)[0]
            if len(nbrs) > 0:
                avg_neigh_deg[x] = degrees[nbrs].mean()
        obs["avg_neighbor_degree_cv"] = float(avg_neigh_deg[degrees > 0].std() / max(avg_neigh_deg[degrees > 0].mean(), 1e-10)) if (degrees > 0).sum() > 5 else 0.0
    except Exception:
        obs["avg_neighbor_degree_cv"] = 0.0

    # ══════════════════════════════════════════════════════════════
    # NEW RUN #007: Spectral zeta function at s=2
    # ══════════════════════════════════════════════════════════════
    try:
        eigvals_pos = eigvals if 'eigvals' in dir() and len(eigvals) > 0 else la.eigvalsh(L_link)[la.eigvalsh(L_link) > 1e-10]
        obs["spectral_zeta_2"] = float(np.sum(1.0 / eigvals_pos**2)) if len(eigvals_pos) > 0 else 0.0
    except Exception:
        obs["spectral_zeta_2"] = 0.0

    # ══════════════════════════════════════════════════════════════
    # NEW RUN #007: Past-future size correlation (Spearman-like)
    # ══════════════════════════════════════════════════════════════
    try:
        # Interior elements only (both past and future > 0)
        interior = (past_sizes > 0) & (future_sizes > 0)
        if interior.sum() > 20:
            from scipy.stats import spearmanr
            rho, _ = spearmanr(past_sizes[interior], future_sizes[interior])
            obs["pf_spearman"] = float(rho)
        else:
            obs["pf_spearman"] = 0.0
    except Exception:
        obs["pf_spearman"] = 0.0

    # ══════════════════════════════════════════════════════════════
    # NEW RUN #007: Row-norm Gini of C (future cone inequality)
    # ══════════════════════════════════════════════════════════════
    obs["row_gini_C"] = gini_coefficient(np.sqrt(future_sizes))

    # ══════════════════════════════════════════════════════════════
    # NEW RUN #007: Past-future Gini DIFFERENCE
    # ══════════════════════════════════════════════════════════════
    obs["pf_gini_diff"] = obs["column_gini_C"] - obs["row_gini_C"]

    return obs


# ── CRN Engine ──────────────────────────────────────────────────────
def run_crn_scan():
    rng = np.random.default_rng(MASTER_SEED)
    results = {}

    for metric_name, eps in METRICS.items():
        metric_fn = METRIC_FNS[metric_name]
        print(f"\n{'='*60}")
        print(f"Metric: {metric_name}, eps={eps}")
        print(f"{'='*60}")

        deltas = {}

        for trial in range(M):
            seed_trial = rng.integers(0, 2**31)
            rng_trial = np.random.default_rng(seed_trial)

            # Sprinkle points
            pts = sprinkle(N, T_DIAMOND, rng_trial)

            # Flat causal matrix
            C_flat = causal_flat(pts)
            obs_flat = compute_all_observables(C_flat, pts)

            # Curved causal matrix (same points!)
            C_curved = metric_fn(pts, eps)
            obs_curved = compute_all_observables(C_curved, pts)

            # CRN delta
            for key in obs_flat:
                if key not in deltas:
                    deltas[key] = []
                deltas[key].append(obs_curved[key] - obs_flat[key])

            if (trial + 1) % 5 == 0:
                print(f"  Trial {trial + 1}/{M} done")

        # Compute Cohen's d and Wilcoxon p for each observable
        metric_results = {}
        for key, delta_list in deltas.items():
            d_arr = np.array(delta_list)
            mean_d = d_arr.mean()
            std_d = d_arr.std(ddof=1) if len(d_arr) > 1 else 1e-10
            cohen_d = mean_d / max(std_d, 1e-10)

            # Wilcoxon signed-rank test
            try:
                if np.all(d_arr == 0):
                    p_val = 1.0
                else:
                    _, p_val = wilcoxon(d_arr, alternative='two-sided')
            except Exception:
                p_val = 1.0

            metric_results[key] = {
                "cohen_d": round(float(cohen_d), 4),
                "mean_delta": round(float(mean_d), 6),
                "std_delta": round(float(std_d), 6),
                "p_wilcoxon": round(float(p_val), 6),
            }

        results[metric_name] = metric_results

    return results


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"COMP-SCAN v7 — Run #007")
    print(f"N={N}, M={M}, metrics: {list(METRICS.keys())}")
    t0 = time.time()

    results = run_crn_scan()

    elapsed = time.time() - t0

    # ── Print ranked results ──
    print(f"\n{'='*80}")
    print(f"COMP-SCAN v7 RESULTS (elapsed: {elapsed:.1f}s)")
    print(f"{'='*80}")

    BASELINES = {"tc", "n2", "link_count", "degree_cv", "sum_deg_sq",
                 "max_height", "I_0", "I_1", "I_2", "I_3", "I_3plus"}
    CONTROLS = {"column_gini_C2", "lva", "column_gini_C", "link_degree_skew", "fan_kurtosis"}

    for metric_name, metric_results in results.items():
        print(f"\n--- {metric_name} ---")
        # Sort by |d|
        ranked = sorted(metric_results.items(),
                       key=lambda x: abs(x[1]["cohen_d"]), reverse=True)
        for key, vals in ranked:
            tag = ""
            if key in BASELINES:
                tag = " [BASELINE]"
            elif key in CONTROLS:
                tag = " [CONTROL]"
            elif abs(vals["cohen_d"]) >= 0.5:
                tag = " *"
            p_str = f"p={vals['p_wilcoxon']:.4f}"
            if vals["p_wilcoxon"] < 0.01:
                p_str = f"p={vals['p_wilcoxon']:.2e}"
            print(f"  {key:30s}  d={vals['cohen_d']:+8.3f}  {p_str}{tag}")

    # ── Save ──
    output = {
        "run": "007",
        "version": "comp_scan_v7",
        "N": N,
        "M": M,
        "master_seed": MASTER_SEED,
        "metrics": METRICS,
        "elapsed_seconds": round(elapsed, 1),
        "results": results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {RESULTS_FILE}")
