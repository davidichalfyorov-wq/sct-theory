#!/usr/bin/env python3
"""
COMP-SCAN v5: conditional observables + spatial structure.
N=500, M=20 CRN pairs, pp-wave eps=5 + Schwarzschild eps=0.005.

Key innovations:
  - Conditional observables (condition on interval size s, TC-free by construction)
  - Spatial autocorrelation of column norms (CNMI)
  - Column cosine coherence (CCC)
  - Height-stratified gradient (HCNG)
  - LVA (positive control, known d~-0.97 ppwave, d~+2.37 Schwarzschild)
  - column_gini_C2 (positive control, known d~-1.6 ppwave, d~+3.4 Schwarzschild)

Output: docs/analysis_runs/run_20260325_200948/comp_scan_results.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.linalg import norm
from scipy import sparse
from scipy.stats import pearsonr, spearmanr

# ── Parameters ──────────────────────────────────────────────────────
N = 500
M = 20
T_DIAMOND = 1.0
MASTER_SEED = 55555  # distinct from all prior runs
METRICS = {
    "ppwave_quad": 5.0,
    "schwarzschild": 0.005,
}

RUN_DIR = Path("docs/analysis_runs/run_20260325_200948")
RESULTS_FILE = RUN_DIR / "comp_scan_results.json"


# ── Sprinkling ──────────────────────────────────────────────────────
def sprinkle(N_target: int, T: float, rng: np.random.Generator) -> np.ndarray:
    """Sprinkle N_target points into a 4D causal diamond |t|+r < T/2."""
    pts = []
    while len(pts) < N_target:
        batch = rng.uniform(-T / 2, T / 2, size=(N_target * 8, 4))
        r = np.sqrt(batch[:, 1]**2 + batch[:, 2]**2 + batch[:, 3]**2)
        inside = np.abs(batch[:, 0]) + r < T / 2
        pts.extend(batch[inside].tolist())
    pts = np.array(pts[:N_target])
    order = np.argsort(pts[:, 0])
    return pts[order]


# ── Causal matrices ─────────────────────────────────────────────────
def causal_flat(pts: np.ndarray, _eps: float = 0.0) -> np.ndarray:
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        dt = pts[i + 1:, 0] - pts[i, 0]
        dr2 = np.sum((pts[i + 1:, 1:] - pts[i, 1:])**2, axis=1)
        C[i, i + 1:] = (dt**2 > dr2).astype(np.int8)
    return C


def causal_ppwave_quad(pts: np.ndarray, eps: float) -> np.ndarray:
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


def causal_schwarzschild(pts: np.ndarray, eps: float) -> np.ndarray:
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
def gini_coefficient(values: np.ndarray) -> float:
    v = np.sort(np.abs(values))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * v)) / (n * np.sum(v)) - (n + 1) / n


def shannon_entropy_from_array(arr: np.ndarray) -> float:
    """Shannon entropy of a non-negative array (treated as distribution)."""
    p = arr / arr.sum() if arr.sum() > 0 else arr
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def build_link_graph(C: np.ndarray, C2: np.ndarray):
    """Link graph: i link j iff C[i,j]=1 and C2[i,j]=0."""
    link_mask = (C > 0) & (C2 == 0)
    A = link_mask.astype(np.float32)
    return A + A.T  # undirected


def compute_heights(C: np.ndarray) -> np.ndarray:
    """Longest chain from any minimal element to each element (topological DP)."""
    n = len(C)
    h = np.zeros(n, dtype=np.int32)
    # Points sorted by time, so forward pass works
    for j in range(n):
        predecessors = np.where(C[:j, j] > 0)[0]
        if len(predecessors) > 0:
            h[j] = h[predecessors].max() + 1
    return h


# ── Observable computation (NEW v5) ────────────────────────────────
def compute_all_observables(C: np.ndarray, pts: np.ndarray) -> dict:
    """Compute ~35 observables from causal matrix C and point coords."""
    n = len(C)
    obs = {}

    # ── Pre-computation ──
    Cf = C.astype(np.float64)
    C2 = Cf @ Cf  # C2[i,j] = interval cardinality |I(i,j)|
    tc = int(C.sum())
    obs["tc"] = tc
    obs["n2"] = int(C2.sum())

    # Link graph
    A_link = build_link_graph(C, C2)
    degrees = A_link.sum(axis=1)
    link_count = int(degrees.sum()) // 2
    obs["link_count"] = link_count
    obs["degree_cv"] = float(degrees.std() / max(degrees.mean(), 1e-10))

    # Heights
    heights = compute_heights(C)
    obs["max_height"] = int(heights.max())

    # Layer counts (baselines)
    # layer_k: number of pairs with exactly k elements between them
    for k in range(4):
        obs[f"layer_{k}"] = int(np.sum(C2 == k) - (n if k == 0 else 0))
    obs["layer_3plus"] = int(np.sum(C2 >= 3))

    # Interval abundances (baselines)
    causal_pairs = np.argwhere((C > 0))
    interval_sizes = C2[causal_pairs[:, 0], causal_pairs[:, 1]].astype(int)
    for k in range(4):
        obs[f"I_{k}"] = int(np.sum(interval_sizes == k))
    obs["I_3plus"] = int(np.sum(interval_sizes >= 3))

    # sum_deg_sq baseline
    obs["sum_deg_sq"] = float(np.sum(degrees**2))

    # ═══════════════════════════════════════════════════════════════
    # POSITIVE CONTROL: column_gini_C2
    # ═══════════════════════════════════════════════════════════════
    col_norms_C2 = np.sqrt(np.sum(C2**2, axis=0))  # ||col_j(C²)||_2
    obs["column_gini_C2"] = gini_coefficient(col_norms_C2)

    # ═══════════════════════════════════════════════════════════════
    # POSITIVE CONTROL: LVA (Link Valence Anisotropy)
    # ═══════════════════════════════════════════════════════════════
    link_mask = (C > 0) & (C2 == 0)
    d_plus = link_mask.sum(axis=1)  # future link count for each element
    # For each element x with d_plus[x] >= 2:
    # Compute k_+(x, y_i) = |J+(x) ∩ J+(y_i)| for each future link y_i
    # Using proxy: (C @ C^T)[x, y_i] = |common futures|
    CCT = Cf @ Cf.T  # common futures matrix
    lva_vals = []
    for x in range(n):
        future_links = np.where(link_mask[x, :])[0]
        if len(future_links) < 2:
            continue
        # k_+(x, y_i) = CCT[x, y_i] - accounts for common futures
        kplus = CCT[x, future_links]
        mu = kplus.mean()
        if mu > 0:
            lva_vals.append(float(kplus.var() / mu**2))
    obs["lva"] = float(np.mean(lva_vals)) if lva_vals else 0.0

    # LVA past version
    CTC = Cf.T @ Cf  # common pasts matrix
    lva_past_vals = []
    # Past links: y_i -> x where link_mask[y_i, x] = 1
    for x in range(n):
        past_links = np.where(link_mask[:, x])[0]
        if len(past_links) < 2:
            continue
        kminus = CTC[x, past_links]
        mu = kminus.mean()
        if mu > 0:
            lva_past_vals.append(float(kminus.var() / mu**2))
    obs["lva_past"] = float(np.mean(lva_past_vals)) if lva_past_vals else 0.0

    # ═══════════════════════════════════════════════════════════════
    # NEW: CONDITIONAL OBSERVABLES (TC-free by construction)
    # ═══════════════════════════════════════════════════════════════

    # Collect intervals by size
    # Only consider pairs where C[i,j]=1 (causal relation exists)
    # and C2[i,j] = s > 0 (nonzero interval)
    size_bins = [3, 5, 10, 20]

    for s_target in size_bins:
        # Find all pairs (i,j) with C[i,j]=1 and C2[i,j]=s_target
        pairs = []
        for i, j in zip(causal_pairs[:, 0], causal_pairs[:, 1]):
            if int(C2[i, j]) == s_target:
                pairs.append((i, j))

        if len(pairs) < 5:
            # Not enough intervals of this size
            obs[f"cond_f_s{s_target}"] = np.nan
            obs[f"cond_r3_s{s_target}"] = np.nan
            obs[f"cond_amid_s{s_target}"] = np.nan
            obs[f"cond_rho_pf_s{s_target}"] = np.nan
            continue

        f_vals = []
        r3_vals = []
        amid_vals = []
        rho_pf_vals = []

        for i, j in pairs[:200]:  # cap at 200 intervals per bin
            # Extract interval elements: z such that C[i,z]=1 and C[z,j]=1
            interval_elems = np.where((C[i, :] > 0) & (C[:, j] > 0))[0]
            # Exclude i and j themselves
            interval_elems = interval_elems[(interval_elems != i) & (interval_elems != j)]
            s = len(interval_elems)
            if s < 2:
                continue

            # Sub-causal matrix within interval
            C_I = Cf[np.ix_(interval_elems, interval_elems)]

            # 1. Ordering fraction f(s) = relations / C(s,2)
            relations = int(C_I.sum())
            f_val = relations / max(s * (s - 1) / 2, 1)
            f_vals.append(f_val)

            # 2. Chain ratio r_3(s) = 3-chains / 2-chains
            C_I2 = C_I @ C_I
            c3 = C_I2.sum()  # number of 3-chains = sum of (C_I)^2
            c2 = relations  # number of 2-chains = relations
            r3_vals.append(float(c3 / max(c2, 1)))

            # 3. Midpoint layer fraction A_mid(s)
            # a(z) = C2[i,z], b(z) = C2[z,j]
            a = C2[i, interval_elems]
            b = C2[interval_elems, j]
            # "Midpoint" = elements where |a-b| < 0.25*s
            delta = 0.25
            midpoint_mask = np.abs(a - b) < delta * s
            amid_vals.append(float(midpoint_mask.sum() / s))

            # 4. Past-future correlation rho_PF(s)
            if np.std(a) > 0 and np.std(b) > 0:
                rho, _ = pearsonr(a, b)
                rho_pf_vals.append(float(rho))

        obs[f"cond_f_s{s_target}"] = float(np.mean(f_vals)) if f_vals else np.nan
        obs[f"cond_r3_s{s_target}"] = float(np.mean(r3_vals)) if r3_vals else np.nan
        obs[f"cond_amid_s{s_target}"] = float(np.mean(amid_vals)) if amid_vals else np.nan
        obs[f"cond_rho_pf_s{s_target}"] = float(np.mean(rho_pf_vals)) if rho_pf_vals else np.nan

    # Aggregate conditional: mean over all s-bins
    for prefix in ["cond_f", "cond_r3", "cond_amid", "cond_rho_pf"]:
        vals = [obs[f"{prefix}_s{s}"] for s in size_bins if not np.isnan(obs.get(f"{prefix}_s{s}", np.nan))]
        obs[f"{prefix}_mean"] = float(np.mean(vals)) if vals else np.nan

    # ═══════════════════════════════════════════════════════════════
    # NEW: SPATIAL STRUCTURE OBSERVABLES (SEED-INVERSION)
    # ═══════════════════════════════════════════════════════════════

    # --- CNMI: Column Norm Moran's I ---
    # Spacelike neighbors: not causally related, height difference <= 1
    cn = col_norms_C2
    cn_mean = cn.mean()
    cn_dev = cn - cn_mean
    ss_total = np.sum(cn_dev**2)

    if ss_total > 0:
        # Build spacelike weight matrix (sparse approach for efficiency)
        W_sum = 0.0
        W_cross = 0.0
        # Sample for efficiency at N=500: check all pairs within height band
        for h_val in range(int(heights.max()) + 1):
            h_mask = heights == h_val
            h_indices = np.where(h_mask)[0]
            # Also include height+1
            h1_mask = heights == h_val + 1
            h1_indices = np.where(h1_mask)[0]
            combined = np.concatenate([h_indices, h1_indices])
            if len(combined) < 2:
                continue
            for ii in range(len(combined)):
                for jj in range(ii + 1, len(combined)):
                    a_idx, b_idx = combined[ii], combined[jj]
                    # Spacelike = not causally related
                    if C[min(a_idx, b_idx), max(a_idx, b_idx)] == 0:
                        W_sum += 1.0
                        W_cross += cn_dev[a_idx] * cn_dev[b_idx]
        if W_sum > 0:
            obs["cnmi"] = float(n * W_cross / (W_sum * ss_total))
        else:
            obs["cnmi"] = 0.0
    else:
        obs["cnmi"] = 0.0

    # --- CCC: Column Cosine Coherence of C² ---
    # Mean pairwise cosine similarity of C² columns
    # For efficiency, sample 2000 random column pairs
    rng_ccc = np.random.default_rng(12345)
    n_sample = min(2000, n * (n - 1) // 2)
    cos_vals = []
    col_norms_for_cos = np.sqrt(np.sum(C2**2, axis=0))
    nonzero_cols = np.where(col_norms_for_cos > 0)[0]
    if len(nonzero_cols) >= 2:
        for _ in range(n_sample):
            i_idx, j_idx = rng_ccc.choice(nonzero_cols, 2, replace=False)
            dot = np.dot(C2[:, i_idx], C2[:, j_idx])
            cos_sim = dot / (col_norms_for_cos[i_idx] * col_norms_for_cos[j_idx])
            cos_vals.append(cos_sim)
        obs["ccc"] = float(np.mean(cos_vals))
    else:
        obs["ccc"] = 0.0

    # --- HCNG: Height-Stratified Column Norm Gradient ---
    K_bins = max(5, int(np.sqrt(n)))
    h_min, h_max = int(heights.min()), int(heights.max())
    bin_edges = np.linspace(h_min, h_max + 1, K_bins + 1)
    bin_means = []
    for b in range(K_bins):
        mask = (heights >= bin_edges[b]) & (heights < bin_edges[b + 1])
        if mask.sum() > 0:
            bin_means.append(float(cn[mask].mean()))
    if len(bin_means) >= 2:
        bm = np.array(bin_means)
        mu_bar = bm.mean()
        if mu_bar > 0:
            obs["hcng"] = float(np.var(bm) / mu_bar**2)
        else:
            obs["hcng"] = 0.0
    else:
        obs["hcng"] = 0.0

    # --- ISHI: Interval Size Heterogeneity at fixed causal distance ---
    # Group causal pairs by height difference
    dh = heights[causal_pairs[:, 1]] - heights[causal_pairs[:, 0]]
    unique_dh = np.unique(dh)
    ishi_vals = []
    for d in unique_dh:
        if d <= 0:
            continue
        mask = dh == d
        sizes = interval_sizes[mask]
        if len(sizes) >= 30:
            mu = sizes.mean()
            if mu > 0:
                ishi_vals.append(float(sizes.var() / mu**2))
    obs["ishi"] = float(np.mean(ishi_vals)) if ishi_vals else 0.0

    # ═══════════════════════════════════════════════════════════════
    # Additional new observables from STRATEGY
    # ═══════════════════════════════════════════════════════════════

    # Link degree entropy
    deg_counts = np.bincount(degrees.astype(int))
    obs["link_degree_entropy"] = shannon_entropy_from_array(deg_counts.astype(float))

    # Link degree skewness & kurtosis
    from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis
    obs["link_degree_skew"] = float(sp_skew(degrees))
    obs["link_degree_kurtosis"] = float(sp_kurtosis(degrees))

    # Column gini of C (not C²)
    col_norms_C = np.sqrt(np.sum(Cf**2, axis=0))
    obs["column_gini_C"] = gini_coefficient(col_norms_C)

    # Row gini of C²
    row_norms_C2 = np.sqrt(np.sum(C2**2, axis=1))
    obs["row_gini_C2"] = gini_coefficient(row_norms_C2)

    # Column norm entropy of C²
    if col_norms_C2.sum() > 0:
        obs["col_norm_entropy_C2"] = shannon_entropy_from_array(col_norms_C2)
    else:
        obs["col_norm_entropy_C2"] = 0.0

    # Interval size CV (coefficient of variation across all intervals)
    nonzero_sizes = interval_sizes[interval_sizes > 0]
    if len(nonzero_sizes) > 0:
        obs["interval_size_cv"] = float(nonzero_sizes.std() / max(nonzero_sizes.mean(), 1e-10))
    else:
        obs["interval_size_cv"] = 0.0

    # Degree assortativity (do high-degree link-graph nodes connect to high-degree?)
    A_sp = sparse.csr_matrix(A_link)
    rows, cols = A_sp.nonzero()
    if len(rows) > 10:
        d_src = degrees[rows]
        d_dst = degrees[cols]
        if np.std(d_src) > 0 and np.std(d_dst) > 0:
            obs["degree_assortativity"] = float(np.corrcoef(d_src, d_dst)[0, 1])
        else:
            obs["degree_assortativity"] = 0.0
    else:
        obs["degree_assortativity"] = 0.0

    # Past-future size correlation (Spearman)
    # For each element x: |past(x)| = sum(C[:,x]), |future(x)| = sum(C[x,:])
    past_sizes = Cf.sum(axis=0)  # column sums
    future_sizes = Cf.sum(axis=1)  # row sums
    interior = (past_sizes > 0) & (future_sizes > 0)
    if interior.sum() > 10:
        rho_pf, _ = spearmanr(past_sizes[interior], future_sizes[interior])
        obs["pf_spearman"] = float(rho_pf)
    else:
        obs["pf_spearman"] = 0.0

    # Longest chain
    obs["longest_chain"] = int(heights.max())

    return obs


# ── CRN trial ──────────────────────────────────────────────────────
def crn_trial(seed: int, metric_name: str, eps: float) -> dict:
    """One CRN trial: sprinkle once, compute flat & curved observables."""
    rng = np.random.default_rng(seed)
    pts = sprinkle(N, T_DIAMOND, rng)

    C_flat = causal_flat(pts)
    C_curved = METRIC_FNS[metric_name](pts, eps)

    obs_flat = compute_all_observables(C_flat, pts)
    obs_curved = compute_all_observables(C_curved, pts)

    delta = {}
    for key in obs_flat:
        vf = obs_flat[key]
        vc = obs_curved[key]
        if isinstance(vf, (int, float)) and isinstance(vc, (int, float)):
            if np.isnan(vf) or np.isnan(vc):
                delta[key] = np.nan
            else:
                delta[key] = float(vc - vf)
        else:
            delta[key] = np.nan
    return {"flat": obs_flat, "curved": obs_curved, "delta": delta}


# ── Statistical summary ────────────────────────────────────────────
def summarize(results: list, obs_names: list) -> dict:
    """Cohen's d, p-value, mean delta for each observable."""
    summary = {}
    for name in obs_names:
        deltas = [r["delta"].get(name, np.nan) for r in results]
        deltas = [d for d in deltas if not np.isnan(d)]
        if len(deltas) < 5:
            summary[name] = {"d": np.nan, "p": np.nan, "mean_delta": np.nan, "n": len(deltas)}
            continue
        darr = np.array(deltas)
        sd = darr.std(ddof=1)
        d_cohen = float(darr.mean() / sd) if sd > 0 else 0.0
        from scipy.stats import wilcoxon, ttest_rel
        try:
            _, p_val = wilcoxon(darr)
        except Exception:
            p_val = 1.0
        summary[name] = {
            "d": round(d_cohen, 3),
            "p": round(float(p_val), 6),
            "mean_delta": round(float(darr.mean()), 6),
            "n": len(deltas),
        }
    return summary


# ── Main ───────────────────────────────────────────────────────────
def main():
    print(f"COMP-SCAN v5: N={N}, M={M}, metrics={list(METRICS.keys())}")
    from numpy.random import SeedSequence
    ss = SeedSequence(MASTER_SEED)

    all_results = {}
    t0 = time.time()

    for metric_idx, (metric_name, eps) in enumerate(METRICS.items()):
        print(f"\n{'='*60}")
        print(f"Metric: {metric_name}, eps={eps}")
        # Use distinct integer seeds for each trial (avoid SeedSequence.entropy bug)
        seeds = [MASTER_SEED * 1000 + metric_idx * 100 + i for i in range(M)]
        results = []
        for trial_idx in range(M):
            t_trial = time.time()
            res = crn_trial(seeds[trial_idx], metric_name, eps)
            results.append(res)
            dt = time.time() - t_trial
            if trial_idx < 3 or trial_idx == M - 1:
                print(f"  Trial {trial_idx+1}/{M}: {dt:.1f}s")

        # Get observable names from first result
        obs_names = [k for k in results[0]["delta"].keys()]
        summary = summarize(results, obs_names)

        # Sort by |d|
        ranked = sorted(summary.items(), key=lambda x: abs(x[1]["d"]) if not np.isnan(x[1]["d"]) else 0, reverse=True)

        print(f"\n  Top 15 by |d|:")
        for name, stats in ranked[:15]:
            d_val = stats["d"]
            p_val = stats["p"]
            flag = ""
            if abs(d_val) >= 0.5 and p_val < 0.05:
                flag = " ***"
            elif abs(d_val) >= 0.5:
                flag = " **"
            print(f"    {name:30s}  d={d_val:+.3f}  p={p_val:.4f}  n={stats['n']}{flag}")

        all_results[metric_name] = {
            "eps": eps,
            "M": M,
            "N": N,
            "summary": summary,
            "raw_deltas": {name: [r["delta"].get(name, None) for r in results] for name in obs_names},
        }

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Total time: {elapsed:.1f}s")

    # Identify survivors: |d| >= 0.5 on BOTH metrics
    ppw_summary = all_results.get("ppwave_quad", {}).get("summary", {})
    sch_summary = all_results.get("schwarzschild", {}).get("summary", {})
    common_obs = set(ppw_summary.keys()) & set(sch_summary.keys())

    survivors = []
    for name in sorted(common_obs):
        d_ppw = ppw_summary[name]["d"]
        d_sch = sch_summary[name]["d"]
        if np.isnan(d_ppw) or np.isnan(d_sch):
            continue
        if abs(d_ppw) >= 0.5 or abs(d_sch) >= 0.5:
            survivors.append({
                "name": name,
                "d_ppwave": d_ppw,
                "d_schwarzschild": d_sch,
                "multi_metric": abs(d_ppw) >= 0.5 and abs(d_sch) >= 0.5,
            })

    survivors.sort(key=lambda x: abs(x["d_ppwave"]) + abs(x["d_schwarzschild"]), reverse=True)
    all_results["survivors"] = survivors

    print(f"\nSurvivors (|d| >= 0.5 on at least one metric): {len(survivors)}")
    for s in survivors[:20]:
        mm = " MULTI" if s["multi_metric"] else ""
        print(f"  {s['name']:30s}  pp={s['d_ppwave']:+.3f}  sch={s['d_schwarzschild']:+.3f}{mm}")

    # Save
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Convert nan to null for JSON
    def clean_for_json(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    with open(RESULTS_FILE, "w") as f:
        json.dump(clean_for_json(all_results), f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
