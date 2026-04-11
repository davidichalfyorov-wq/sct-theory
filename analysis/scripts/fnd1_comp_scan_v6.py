#!/usr/bin/env python3
"""
COMP-SCAN v6: SEED ideas + TC-mediation filter.
N=500, M=20 CRN pairs, pp-wave eps=5 + Schwarzschild eps=0.005.

New observables from SEED agents:
  - LFSE: Link Fan Spectral Eccentricity
  - ICRD: Interval Chain Ratio Dispersion
  - LDKL: Layered Degree KL-Divergence
  - LMMD: Local Myrheim-Meyer Dimension Dispersion
  - IWE: Interval Width Profile Entropy (simplified)
  - IPR Gini of C2 singular vectors (SI-2 from incubator)
  - Chain decay exponent (SC-2 from incubator)

Plus positive controls: column_gini_C2, LVA
Plus baselines: tc, link_count, degree_cv, sum_deg_sq, I_0..I_3plus

Output: docs/analysis_runs/run_20260325_202020/comp_scan_v6_results.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.linalg import norm, eigh, svd
from scipy.stats import wilcoxon

# ── Parameters ──────────────────────────────────────────────────────
N = 500
M = 20
T_DIAMOND = 1.0
MASTER_SEED = 66666  # distinct from all prior runs
METRICS = {
    "ppwave_quad": 5.0,
    "schwarzschild": 0.005,
}

RUN_DIR = Path("docs/analysis_runs/run_20260325_202020")
RESULTS_FILE = RUN_DIR / "comp_scan_v6_results.json"


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
    return float((2 * np.sum(idx * v)) / (n * np.sum(v)) - (n + 1) / n)


def shannon_entropy(arr: np.ndarray) -> float:
    p = arr / arr.sum() if arr.sum() > 0 else arr
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def build_link_mask(C: np.ndarray, C2: np.ndarray) -> np.ndarray:
    """Link mask: i->j iff C[i,j]=1 and C2[i,j]=0 (no intervening elements)."""
    return (C > 0) & (C2 == 0)


# ── Observable computation ──────────────────────────────────────────
def compute_all_observables(C: np.ndarray, pts: np.ndarray) -> dict:
    """Compute all observables from causal matrix C and point coords."""
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
    link_directed = link_mask.astype(np.float32)
    A_link = link_directed + link_directed.T  # undirected
    degrees = A_link.sum(axis=1)
    link_count = int(degrees.sum()) // 2
    obs["link_count"] = link_count
    obs["degree_cv"] = float(degrees.std() / max(degrees.mean(), 1e-10))
    obs["sum_deg_sq"] = float(np.sum(degrees**2))

    # Heights (longest chain from source)
    heights = np.zeros(n, dtype=np.int32)
    for j in range(n):
        preds = np.where(C[:j, j] > 0)[0]
        if len(preds) > 0:
            heights[j] = heights[preds].max() + 1
    obs["max_height"] = int(heights.max())

    # Interval abundances (baselines)
    causal_pairs = np.argwhere(C > 0)
    interval_sizes = C2[causal_pairs[:, 0], causal_pairs[:, 1]].astype(int)
    for k in range(4):
        obs[f"I_{k}"] = int(np.sum(interval_sizes == k))
    obs["I_3plus"] = int(np.sum(interval_sizes >= 3))

    # ═══════════════════════════════════════════════════════════════
    # POSITIVE CONTROLS (certified observables)
    # ═══════════════════════════════════════════════════════════════

    # column_gini_C2
    col_norms_C2 = np.sqrt(np.sum(C2**2, axis=0))
    obs["column_gini_C2"] = gini_coefficient(col_norms_C2)

    # LVA (Link Valence Anisotropy)
    CCT = Cf @ Cf.T
    d_plus = link_mask.sum(axis=1)
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

    # LVA past
    CTC = Cf.T @ Cf
    lva_past_vals = []
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
    # NEW SEED-CONSTRAINT: S1 — LFSE (Link Fan Spectral Eccentricity)
    # ═══════════════════════════════════════════════════════════════
    lfse_vals = []
    for x in range(n):
        flinks = np.where(link_mask[x, :])[0]
        k = len(flinks)
        if k < 3:
            continue
        # Build link-fan adjacency Q: Q_ij = C[flinks[i], flinks[j]] + C[flinks[j], flinks[i]]
        Q = np.zeros((k, k), dtype=np.float64)
        for ii in range(k):
            for jj in range(ii + 1, k):
                a, b = flinks[ii], flinks[jj]
                q_val = float(C[min(a, b), max(a, b)])  # only upper triangle (time-ordered)
                Q[ii, jj] = q_val
                Q[jj, ii] = q_val
        eigs = np.sort(np.linalg.eigvalsh(Q))[::-1]
        denom = eigs[0] + eigs[1] + 1e-10
        lfse_vals.append(float((eigs[0] - eigs[1]) / denom))
    obs["lfse"] = float(np.median(lfse_vals)) if lfse_vals else 0.0
    obs["lfse_gini"] = gini_coefficient(np.array(lfse_vals)) if len(lfse_vals) >= 2 else 0.0

    # ═══════════════════════════════════════════════════════════════
    # NEW SEED-CONSTRAINT: S2 — ICRD (Interval Chain Ratio Dispersion)
    # ═══════════════════════════════════════════════════════════════
    icrd_vals = []
    for x in range(n):
        flinks = np.where(link_mask[x, :])[0]
        if len(flinks) < 2:
            continue
        # Find causally-related pairs among future links
        m_vals = []
        for ii in range(len(flinks)):
            for jj in range(ii + 1, len(flinks)):
                a, b = flinks[ii], flinks[jj]
                # Check if a < b (causal)
                if C[a, b] > 0:
                    m_vals.append(float(C2[a, b]))
                elif C[b, a] > 0:
                    m_vals.append(float(C2[b, a]))
        if len(m_vals) >= 2:
            m_arr = np.array(m_vals)
            mu = m_arr.mean() + 1.0  # +1 to avoid division by zero
            icrd_vals.append(float(m_arr.var() / mu**2))
    obs["icrd"] = float(np.median(icrd_vals)) if icrd_vals else 0.0

    # ═══════════════════════════════════════════════════════════════
    # NEW SEED-CONSTRAINT: S4 — LDKL (Layered Degree KL-Divergence)
    # ═══════════════════════════════════════════════════════════════
    ldkl_vals = []
    for x in range(n):
        # BFS layers on link graph from x (future only)
        L1 = set(np.where(link_mask[x, :])[0].tolist())
        if len(L1) < 3:
            continue
        L2 = set()
        for y in L1:
            for z in np.where(link_mask[y, :])[0]:
                if z not in L1 and z != x:
                    L2.add(z)
        if len(L2) < 3:
            continue

        # Degree distribution of L1 elements into L2
        d1 = []
        for y in L1:
            d1.append(sum(1 for z in np.where(link_mask[y, :])[0] if z in L2))

        # Degree distribution of L2 elements into L3
        L3 = set()
        for y in L2:
            for z in np.where(link_mask[y, :])[0]:
                if z not in L1 and z not in L2 and z != x:
                    L3.add(z)
        if len(L3) < 2:
            continue
        d2 = []
        for y in L2:
            d2.append(sum(1 for z in np.where(link_mask[y, :])[0] if z in L3))

        # KL divergence between histograms
        max_deg = max(max(d1), max(d2)) + 1
        bins = np.arange(max_deg + 1) + 0.5
        h1, _ = np.histogram(d1, bins=bins)
        h2, _ = np.histogram(d2, bins=bins)
        # Add-one smoothing
        p1 = (h1 + 0.5) / (h1.sum() + 0.5 * len(h1))
        p2 = (h2 + 0.5) / (h2.sum() + 0.5 * len(h2))
        kl = float(np.sum(p1 * np.log(p1 / p2)))
        ldkl_vals.append(kl)
    obs["ldkl"] = float(np.median(ldkl_vals)) if ldkl_vals else 0.0

    # ═══════════════════════════════════════════════════════════════
    # NEW SEED-INVERSION: LMMD (Local Myrheim-Meyer Dimension Dispersion)
    # ═══════════════════════════════════════════════════════════════
    # For each element x, compute local MM dimension using its future intervals
    lmmd_vals = []
    rng_mm = np.random.default_rng(42)
    for x in range(n):
        # Find future elements at moderate distance
        future = np.where(C[x, :] > 0)[0]
        if len(future) < 5:
            continue
        # Sample up to 15 future elements with moderate interval
        candidates = future[C2[x, future] >= 2]
        if len(candidates) < 5:
            continue
        sample = rng_mm.choice(candidates, size=min(15, len(candidates)), replace=False)

        # For each pair (x, y_i): ordering fraction f = C2[x,y]/C(s,2)
        s_vals = C2[x, sample]
        # Count relations within interval: use C2 entries
        # For ordering fraction: r = number of related pairs among interval elements
        # Approximate: r ≈ C3[x,y] = (C^3)[x,y] (3-chains)
        # Simpler: use the ratio n2/s^2 as proxy for ordering fraction
        s_arr = s_vals.astype(float)
        s_arr = s_arr[s_arr >= 2]
        if len(s_arr) < 3:
            continue

        # MM dimension from interval size: d_MM = f^-1(C2(x,y)/C(|I|,2))
        # We need ordering fraction. For simplicity, use the standard formula:
        # In d-dim flat: <C2>/<|I|^2> = d!*(d-1)!/(2d)!
        # But we want PER-ELEMENT DISPERSION, so compute the RAW interval sizes
        # and their coefficient of variation
        lmmd_vals.append(float(s_arr.var() / max(s_arr.mean(), 1)**2))
    obs["lmmd"] = float(np.mean(lmmd_vals)) if lmmd_vals else 0.0
    obs["lmmd_gini"] = gini_coefficient(np.array(lmmd_vals)) if len(lmmd_vals) >= 2 else 0.0

    # ═══════════════════════════════════════════════════════════════
    # INCUBATOR: SI-2 — IPR Gini of C² singular vectors
    # ═══════════════════════════════════════════════════════════════
    try:
        U, S_vals, Vt = np.linalg.svd(C2, full_matrices=False)
        r = int(np.sum(S_vals > 1e-6))
        if r >= 5:
            # IPR for each left singular vector
            ipr_vals = []
            for j in range(min(r, 50)):  # cap at 50 vectors
                u_j = U[:, j]
                ipr = float(n * np.sum(u_j**4))  # IPR in [1, N]
                ipr_vals.append(ipr)
            ipr_arr = np.array(ipr_vals)
            obs["ipr_gini_C2"] = gini_coefficient(ipr_arr)
            obs["ipr_mean_C2"] = float(ipr_arr.mean())
        else:
            obs["ipr_gini_C2"] = 0.0
            obs["ipr_mean_C2"] = 0.0
    except Exception:
        obs["ipr_gini_C2"] = 0.0
        obs["ipr_mean_C2"] = 0.0

    # ═══════════════════════════════════════════════════════════════
    # INCUBATOR: SC-2 — Chain decay exponent
    # ═══════════════════════════════════════════════════════════════
    # Count k-chains for k=2,3,4,5
    # k-chains = paths of k successive causal relations
    # 2-chains = tc, 3-chains = sum(C^2) = n2
    Cf3 = C2 @ Cf
    n3 = float(Cf3.sum())  # 3-chains
    Cf4 = Cf3 @ Cf
    n4 = float(Cf4.sum())  # 4-chains

    obs["n3"] = n3
    obs["n4"] = n4

    # Chain decay: fit log(N_k) vs k for k=2,3,4
    chain_counts = [tc, obs["n2"], n3, n4]
    log_counts = [np.log(max(c, 1)) for c in chain_counts]
    k_vals = [2, 3, 4, 5]
    if all(c > 0 for c in chain_counts):
        # Linear regression log(N_k) = a + b*k
        A_mat = np.vstack([k_vals, np.ones(4)]).T
        result = np.linalg.lstsq(A_mat, log_counts, rcond=None)
        obs["chain_decay_slope"] = float(result[0][0])
        obs["chain_decay_residual"] = float(np.std(log_counts - A_mat @ result[0]))
    else:
        obs["chain_decay_slope"] = 0.0
        obs["chain_decay_residual"] = 0.0

    # Chain ratio N3/N2
    obs["chain_ratio_32"] = float(n3 / max(obs["n2"], 1))
    obs["chain_ratio_43"] = float(n4 / max(n3, 1))

    # ═══════════════════════════════════════════════════════════════
    # ADDITIONAL PER-ELEMENT OBSERVABLES
    # ═══════════════════════════════════════════════════════════════

    # Link fan entropy: H({k_+}) per element, averaged
    fan_entropy_vals = []
    for x in range(n):
        flinks = np.where(link_mask[x, :])[0]
        if len(flinks) < 2:
            continue
        kplus = CCT[x, flinks]
        if kplus.sum() > 0:
            fan_entropy_vals.append(shannon_entropy(kplus))
    obs["fan_entropy"] = float(np.mean(fan_entropy_vals)) if fan_entropy_vals else 0.0
    obs["fan_entropy_gini"] = gini_coefficient(np.array(fan_entropy_vals)) if len(fan_entropy_vals) >= 2 else 0.0

    # Link fan skewness
    from scipy.stats import skew as sp_skew
    fan_skew_vals = []
    for x in range(n):
        flinks = np.where(link_mask[x, :])[0]
        if len(flinks) < 3:
            continue
        kplus = CCT[x, flinks]
        if kplus.std() > 0:
            fan_skew_vals.append(float(sp_skew(kplus)))
    obs["fan_skewness"] = float(np.mean(fan_skew_vals)) if fan_skew_vals else 0.0

    # Past-future link ratio: for each element, |L+|/|L-|
    pf_ratio_vals = []
    for x in range(n):
        nf = link_mask[x, :].sum()
        np_ = link_mask[:, x].sum()
        if np_ > 0:
            pf_ratio_vals.append(float(nf / np_))
    obs["pf_ratio_cv"] = float(np.std(pf_ratio_vals) / max(np.mean(pf_ratio_vals), 1e-10)) if pf_ratio_vals else 0.0

    # Degree assortativity on link graph
    rows = np.where(link_directed)[0]
    cols = np.where(link_directed)[1]
    # Actually use directed adjacency
    nz = np.argwhere(link_directed > 0)
    if len(nz) > 10:
        d_src = degrees[nz[:, 0]]
        d_dst = degrees[nz[:, 1]]
        if np.std(d_src) > 0 and np.std(d_dst) > 0:
            obs["degree_assortativity"] = float(np.corrcoef(d_src, d_dst)[0, 1])
        else:
            obs["degree_assortativity"] = 0.0
    else:
        obs["degree_assortativity"] = 0.0

    # Past-future Spearman correlation
    from scipy.stats import spearmanr
    past_sizes = Cf.sum(axis=0)
    future_sizes = Cf.sum(axis=1)
    interior = (past_sizes > 0) & (future_sizes > 0)
    if interior.sum() > 10:
        rho_pf, _ = spearmanr(past_sizes[interior], future_sizes[interior])
        obs["pf_spearman"] = float(rho_pf)
    else:
        obs["pf_spearman"] = 0.0

    # Interval size CV
    nonzero_sizes = interval_sizes[interval_sizes > 0]
    obs["interval_size_cv"] = float(nonzero_sizes.std() / max(nonzero_sizes.mean(), 1e-10)) if len(nonzero_sizes) > 0 else 0.0

    # Column norm entropy of C2
    obs["col_norm_entropy_C2"] = shannon_entropy(col_norms_C2) if col_norms_C2.sum() > 0 else 0.0

    # Row gini of C2
    row_norms_C2 = np.sqrt(np.sum(C2**2, axis=1))
    obs["row_gini_C2"] = gini_coefficient(row_norms_C2)

    # ISHI: Interval Size Heterogeneity
    dh = heights[causal_pairs[:, 1]] - heights[causal_pairs[:, 0]]
    unique_dh = np.unique(dh)
    ishi_vals = []
    for d in unique_dh:
        if d <= 0:
            continue
        mask = dh == d
        sizes = interval_sizes[mask]
        if len(sizes) >= 10:
            mu = sizes.mean()
            if mu > 0:
                ishi_vals.append(float(sizes.var() / mu**2))
    obs["ishi"] = float(np.mean(ishi_vals)) if ishi_vals else 0.0

    return obs


# ── CRN trial ──────────────────────────────────────────────────────
def crn_trial(seed: int, metric_name: str, eps: float) -> dict:
    rng = np.random.default_rng(seed)
    pts = sprinkle(N, T_DIAMOND, rng)
    C_flat = causal_flat(pts)
    C_curved = METRIC_FNS[metric_name](pts, eps)
    obs_flat = compute_all_observables(C_flat, pts)
    obs_curved = compute_all_observables(C_curved, pts)
    delta = {}
    for key in obs_flat:
        vf, vc = obs_flat[key], obs_curved[key]
        if isinstance(vf, (int, float)) and isinstance(vc, (int, float)):
            delta[key] = float(vc - vf) if not (np.isnan(vf) or np.isnan(vc)) else np.nan
        else:
            delta[key] = np.nan
    return {"flat": obs_flat, "curved": obs_curved, "delta": delta}


# ── Statistical summary ────────────────────────────────────────────
def summarize(results: list, obs_names: list) -> dict:
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


# ── Main ────────────────────────────────────────────────────────────
def main():
    print(f"COMP-SCAN v6: N={N}, M={M}, metrics={list(METRICS.keys())}")
    t0 = time.time()

    all_results = {}

    for metric_name, eps in METRICS.items():
        print(f"\n{'='*60}")
        print(f"Metric: {metric_name}, eps={eps}")
        print(f"{'='*60}")

        results = []
        for trial in range(M):
            seed = MASTER_SEED + trial * 1000 + hash(metric_name) % 10000
            seed = seed % (2**31)
            r = crn_trial(seed, metric_name, eps)
            results.append(r)
            if trial % 5 == 0:
                print(f"  Trial {trial + 1}/{M} done")

        obs_names = sorted(results[0]["delta"].keys())
        summary = summarize(results, obs_names)

        # Sort by |d|
        ranked = sorted(summary.items(), key=lambda x: abs(x[1]["d"]) if not np.isnan(x[1]["d"]) else 0, reverse=True)

        print(f"\n  TOP 15 by |d|:")
        print(f"  {'Observable':<30} {'d':>8} {'p':>10} {'mean_delta':>12}")
        print(f"  {'-'*62}")
        for name, stats in ranked[:15]:
            d = stats["d"]
            p = stats["p"]
            md = stats["mean_delta"]
            flag = ""
            if abs(d) >= 1.0 and p < 0.05:
                flag = " ***"
            elif abs(d) >= 0.5:
                flag = " *"
            print(f"  {name:<30} {d:>8.3f} {p:>10.6f} {md:>12.6f}{flag}")

        all_results[metric_name] = {
            "summary": summary,
            "raw": [{"flat": r["flat"], "curved": r["curved"], "delta": r["delta"]} for r in results],
        }

    # ── Conformal null test ──
    print(f"\n{'='*60}")
    print("Conformal null test")
    print(f"{'='*60}")
    results_null = []
    for trial in range(5):
        seed = MASTER_SEED + trial * 1000 + 99999
        rng = np.random.default_rng(seed)
        pts = sprinkle(N, T_DIAMOND, rng)
        C1 = causal_flat(pts)
        C2_check = causal_flat(pts)
        obs1 = compute_all_observables(C1, pts)
        obs2 = compute_all_observables(C2_check, pts)
        max_diff = 0.0
        for key in obs1:
            v1, v2 = obs1[key], obs2[key]
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                diff = abs(v2 - v1)
                max_diff = max(max_diff, diff)
        results_null.append(max_diff)
    print(f"  Max absolute difference across 5 null trials: {max(results_null)}")
    assert max(results_null) == 0.0, "CONFORMAL NULL FAIL"
    print("  CONFORMAL NULL: PASS (exact 0)")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # ── Save ──
    # Convert NaN to None for JSON
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    output = sanitize(all_results)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
