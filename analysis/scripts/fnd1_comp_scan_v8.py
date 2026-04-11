#!/usr/bin/env python3
"""
COMP-SCAN v8: Interval-internal geometry observables.
N=500, M=20 CRN pairs, pp-wave eps=5 + Schwarzschild eps=0.005.

NEW observables (interval-internal heterogeneity):
  - OFV: Ordering fraction variance (MAD of f(I) at fixed n_0)
  - IBDV: Bipartite density variance (MAD of cross-layer density D(I))
  - SFV: Spacelike fraction profile variance (MAD of Var(s(a)) within I)
  - WPEV: Width profile entropy variance (MAD of H(I))
  - ISF: Sideways fraction (Raychaudhuri detector)
  - IDS_kurt: Kurtosis of Myrheim-Meyer dimension across intervals

Path-based (from SEED-INVERSION):
  - path_excess_skew: Skewness of relative path standing
  - path_divergence_cv: CV of path divergence per element
  - chain_slope_no_k2: log(N_4/N_3) excluding TC-contaminated k=2

SVD vector structure:
  - ipr_gini_C2: Gini of IPR of C^2 singular vectors

Controls: column_gini_C2 (positive), tc (baseline)
Output: docs/analysis_runs/run_20260326_125402/comp_scan_v8_results.json

Author: David Alfyorov
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.linalg import norm, svd
from scipy import stats
from scipy.stats import wilcoxon

# ── Parameters ──────────────────────────────────────────────────────
N = 500
M = 20
T_DIAMOND = 1.0
MASTER_SEED = 88888  # distinct from all prior runs
N0_INTERVAL = 15  # minimum interval size for internal statistics
MAX_INTERVALS = 300  # max intervals to sample per sprinkling
BOUNDARY_FRAC = 0.15  # exclude boundary 15% on each end

METRICS = {
    "ppwave_quad": 5.0,
    "schwarzschild": 0.005,
}

RUN_DIR = Path("docs/analysis_runs/run_20260326_125402")
RESULTS_FILE = RUN_DIR / "comp_scan_v8_results.json"


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
        f = xm**2 - ym**2
        dz = pts[i + 1:, 3] - pts[i, 3]
        mink = dt**2 - dr2
        corr = eps * f * (dt + dz)**2 / 2.0
        C[i, i + 1:] = ((mink > corr) & (dt > 0)).astype(np.int8)
    return C


def causal_schwarzschild(pts, eps):
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        dt = pts[i + 1:, 0] - pts[i, 0]
        dr2 = np.sum((pts[i + 1:, 1:] - pts[i, 1:])**2, axis=1)
        xm = (pts[i + 1:, 1] + pts[i, 1]) / 2
        ym = (pts[i + 1:, 2] + pts[i, 2]) / 2
        zm = (pts[i + 1:, 3] + pts[i, 3]) / 2
        rm = np.sqrt(xm**2 + ym**2 + zm**2) + 0.3
        Phi = -eps / rm
        C[i, i + 1:] = (((1 + 2*Phi) * dt**2 > (1 - 2*Phi) * dr2) & (dt > 0)).astype(np.int8)
    return C


METRIC_FNS = {
    "ppwave_quad": causal_ppwave_quad,
    "schwarzschild": causal_schwarzschild,
}


# ── Interval extraction ─────────────────────────────────────────────
def extract_intervals(C, pts, n0_min, max_count, boundary_frac):
    """Extract intervals I(x,y) with |I| >= n0_min.
    Returns list of (i, j, elements_indices) tuples.
    """
    n = len(C)
    C2 = C.astype(np.int16) @ C.astype(np.int16)

    # Boundary exclusion
    t_vals = pts[:, 0]
    t_min, t_max = t_vals.min(), t_vals.max()
    t_range = t_max - t_min
    t_lo = t_min + boundary_frac * t_range
    t_hi = t_max - boundary_frac * t_range
    interior = (t_vals >= t_lo) & (t_vals <= t_hi)

    # Find pairs with |I| >= n0_min
    intervals = []
    interior_idx = np.where(interior)[0]
    for i in interior_idx:
        for j in range(i + 1, n):
            if C[i, j] == 0:
                continue
            if not interior[j]:
                continue
            sz = int(C2[i, j])
            if sz < n0_min:
                continue
            # Elements in interval: z such that C[i,z]=1 and C[z,j]=1
            elems = np.where((C[i, :] == 1) & (C[:, j] == 1))[0]
            if len(elems) >= n0_min:
                intervals.append((i, j, elems))
            if len(intervals) >= max_count:
                return intervals
    return intervals


# ── Interval-internal statistics ─────────────────────────────────────
def ordering_fraction(C, elems):
    """Ordering fraction f = relations / C(n,2) within interval."""
    sub = C[np.ix_(elems, elems)]
    n = len(elems)
    pairs = n * (n - 1) // 2
    if pairs == 0:
        return 0.0
    rels = int(np.sum(sub))
    return rels / pairs


def bipartite_density(C, elems):
    """Cross-density between early and late halves of interval."""
    sub = C[np.ix_(elems, elems)]
    n = len(elems)
    # Past count for each element within interval
    past_counts = np.sum(sub, axis=0)
    ranks = np.argsort(np.argsort(past_counts))
    half = n // 2
    lo = elems[ranks < half]
    hi = elems[ranks >= half]
    cross = 0
    for a in lo:
        for b in hi:
            if C[a, b] == 1 or C[b, a] == 1:
                cross += 1
    return cross / max(len(lo) * len(hi), 1)


def spacelike_fraction_var(C, elems):
    """Variance of per-element spacelike fraction within interval."""
    sub = C[np.ix_(elems, elems)]
    n = len(elems)
    if n < 3:
        return 0.0
    # Causal matrix (including transitive)
    causal = sub + sub.T
    causal = (causal > 0).astype(int)
    np.fill_diagonal(causal, 1)
    spacelike_fracs = []
    for k in range(n):
        spacelike_count = np.sum(causal[k, :] == 0)
        spacelike_fracs.append(spacelike_count / max(n - 1, 1))
    return float(np.var(spacelike_fracs))


def width_profile_entropy(C, elems):
    """Shannon entropy of the width profile (depth histogram)."""
    sub = C[np.ix_(elems, elems)]
    n = len(elems)
    # Depth = number of ancestors within interval
    depths = np.sum(sub, axis=0)
    # Histogram of depths
    max_d = int(np.max(depths)) + 1
    hist = np.zeros(max_d)
    for d in depths:
        hist[int(d)] += 1
    hist = hist / n
    # Shannon entropy
    mask = hist > 0
    return -float(np.sum(hist[mask] * np.log(hist[mask])))


def sideways_fraction(C, i, j, elems):
    """Fraction of interval elements spacelike to midpoint of chain i->j.
    Pick the median element by depth as midpoint proxy.
    """
    sub = C[np.ix_(elems, elems)]
    n = len(elems)
    if n < 3:
        return 0.0
    depths = np.sum(sub, axis=0)
    mid_idx = np.argsort(depths)[n // 2]
    mid_elem = elems[mid_idx]
    # Count elements spacelike to mid_elem within the interval
    spacelike = 0
    for e in elems:
        if e == mid_elem:
            continue
        if C[e, mid_elem] == 0 and C[mid_elem, e] == 0:
            spacelike += 1
    return spacelike / max(n - 1, 1)


def myrheim_meyer_dim(f):
    """Estimate dimension from ordering fraction via MM formula."""
    if f <= 0 or f >= 1:
        return 4.0  # default
    # Approximate inverse: d ~ 2*log(2)/log(1/f) for large d
    # More precise: use the d=4 value 1/20 = 0.05 as reference
    # For generic d: f = Gamma(d+1)*Gamma(d/2) / (4*Gamma(3*d/2))
    # Simplified Newton iteration around d=4
    from math import gamma, log
    def ff(d):
        try:
            return gamma(d+1) * gamma(d/2) / (4 * gamma(3*d/2))
        except (ValueError, OverflowError):
            return 0.05
    # Newton's method from d=4
    d_est = 4.0
    for _ in range(10):
        fd = ff(d_est)
        if abs(fd) < 1e-15:
            break
        # Numerical derivative
        h = 0.01
        fp = (ff(d_est + h) - ff(d_est - h)) / (2 * h)
        if abs(fp) < 1e-15:
            break
        d_est -= (fd - f) / fp
        d_est = max(1.0, min(d_est, 10.0))
    return d_est


# ── Global observables ──────────────────────────────────────────────
def gini(arr):
    """Standard Gini coefficient."""
    a = np.sort(np.abs(arr))
    n = len(a)
    if n == 0 or np.sum(a) == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * a) - (n + 1) * np.sum(a)) / (n * np.sum(a)))


def column_gini_C2(C):
    """Gini of C^2 column norms."""
    C16 = C.astype(np.int16)
    C2 = C16 @ C16
    col_norms = np.sqrt(np.sum(C2.astype(np.float64)**2, axis=0))
    return gini(col_norms)


def ipr_gini_C2(C):
    """Gini of Inverse Participation Ratio of C^2 singular vectors."""
    C16 = C.astype(np.int16)
    C2 = C16 @ C16
    C2f = C2.astype(np.float64)
    try:
        U, s, Vt = np.linalg.svd(C2f, full_matrices=False)
    except np.linalg.LinAlgError:
        return 0.0
    # Keep only nonzero singular values
    mask = s > 1e-10
    if np.sum(mask) < 2:
        return 0.0
    U_active = U[:, mask]
    n = U_active.shape[0]
    # IPR for each singular vector
    iprs = n * np.sum(U_active**4, axis=0)
    return gini(iprs)


def path_counts_dp(C):
    """Compute path counts through each element via DP on link matrix.
    Returns (p_down, p_up) arrays.
    """
    n = len(C)
    # Build link matrix (Hasse diagram)
    C16 = C.astype(np.int16)
    C2 = C16 @ C16
    # L[i,j] = 1 iff C[i,j]=1 and C2[i,j]=0 (link = direct relation with no intervening)
    L = ((C == 1) & (C2 == 0)).astype(np.int8)

    # p_down[i] = number of directed paths from any source to i
    p_down = np.ones(n, dtype=np.float64)
    for j in range(n):
        for i in range(j):
            if L[i, j]:
                p_down[j] += p_down[i]

    # p_up[i] = number of directed paths from i to any sink
    p_up = np.ones(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            if L[i, j]:
                p_up[i] += p_up[j]

    return p_down, p_up


def path_excess_skew(C, pts):
    """Skewness of relative path standing per height layer."""
    p_down, p_up = path_counts_dp(C)
    n = len(C)
    # Height = number of links below (simple proxy: rank by time)
    t_vals = pts[:, 0]
    n_layers = 10
    layer_edges = np.linspace(t_vals.min(), t_vals.max(), n_layers + 1)
    layers = np.digitize(t_vals, layer_edges[1:-1])

    log_products = np.log2(p_down * p_up + 1)
    # Relative standing: R(x) = product(x) / mean(product at same layer)
    R = np.ones(n)
    for h in range(n_layers):
        mask = layers == h
        if np.sum(mask) < 3:
            continue
        layer_mean = np.mean(log_products[mask])
        if layer_mean > 0:
            R[mask] = log_products[mask] / layer_mean

    # Boundary exclusion
    interior = (layers >= 1) & (layers <= n_layers - 2)
    R_int = R[interior]
    if len(R_int) < 10:
        return 0.0
    return float(stats.skew(np.log(R_int + 1e-15)))


def path_divergence_cv(C, pts):
    """Mean CV of future path counts across children of each element."""
    n = len(C)
    C16 = C.astype(np.int16)
    C2 = C16 @ C16
    L = ((C == 1) & (C2 == 0)).astype(np.int8)
    _, p_up = path_counts_dp(C)

    cvs = []
    t_vals = pts[:, 0]
    t_min, t_max = t_vals.min(), t_vals.max()
    t_range = t_max - t_min
    interior = (t_vals >= t_min + 0.15 * t_range) & (t_vals <= t_max - 0.15 * t_range)

    for i in range(n):
        if not interior[i]:
            continue
        children = np.where(L[i, :] == 1)[0]
        if len(children) < 2:
            continue
        child_paths = np.log2(p_up[children] + 1)
        m = np.mean(child_paths)
        if m > 0:
            cvs.append(float(np.std(child_paths) / m))
    if len(cvs) < 5:
        return 0.0
    return float(np.mean(cvs))


def chain_counts(C):
    """Count chains of length 2, 3, 4."""
    C16 = C.astype(np.int16)
    C2 = C16 @ C16
    C3 = C16 @ C2
    C4 = C16 @ C3
    n2 = int(np.sum(C2)) // 2 if np.sum(C2) > 0 else int(np.sum(C))
    n3 = int(np.sum(C3))
    n4 = int(np.sum(C4))
    return n2, n3, n4


def chain_slope_no_k2(C):
    """log(N_4/N_3) — chain decay excluding TC-contaminated k=2."""
    _, n3, n4 = chain_counts(C)
    if n3 <= 0 or n4 <= 0:
        return 0.0
    return float(np.log(n4) - np.log(n3))


# ── Master observable computation ────────────────────────────────────
def compute_all_observables(C, pts):
    """Compute all observables for one sprinkling."""
    results = {}

    # 1. TC (baseline)
    results["tc"] = float(np.sum(C))

    # 2. column_gini_C2 (positive control)
    results["column_gini_C2"] = column_gini_C2(C)

    # 3. IPR Gini of C^2 singular vectors
    results["ipr_gini_C2"] = ipr_gini_C2(C)

    # 4. Path-based observables
    results["path_excess_skew"] = path_excess_skew(C, pts)
    results["path_divergence_cv"] = path_divergence_cv(C, pts)

    # 5. Chain slope (no k=2)
    results["chain_slope_no_k2"] = chain_slope_no_k2(C)

    # 6. Link degree kurtosis
    C16 = C.astype(np.int16)
    C2_mat = C16 @ C16
    L = ((C == 1) & (C2_mat == 0)).astype(np.int8)
    A = L + L.T
    degrees = np.sum(A, axis=1)
    results["link_degree_kurtosis"] = float(stats.kurtosis(degrees))

    # 7. Interval-internal observables
    intervals = extract_intervals(C, pts, N0_INTERVAL, MAX_INTERVALS, BOUNDARY_FRAC)
    n_intervals = len(intervals)
    results["n_intervals"] = n_intervals

    if n_intervals >= 5:
        # OFV: ordering fraction variance
        of_values = []
        for (i, j, elems) in intervals:
            of_values.append(ordering_fraction(C, elems))

        results["ofv_mad"] = float(np.median(np.abs(np.array(of_values) - np.median(of_values))))
        results["ofv_iqr"] = float(np.percentile(of_values, 75) - np.percentile(of_values, 25))

        # IDS: MM dimension scatter
        mm_dims = [myrheim_meyer_dim(f) for f in of_values]
        results["ids_mad"] = float(np.median(np.abs(np.array(mm_dims) - np.median(mm_dims))))
        results["ids_kurtosis"] = float(stats.kurtosis(mm_dims)) if len(mm_dims) >= 4 else 0.0

        # IBDV: bipartite density variance
        bd_values = []
        for (i, j, elems) in intervals[:100]:  # cap at 100 for speed
            bd_values.append(bipartite_density(C, elems))
        results["ibdv_mad"] = float(np.median(np.abs(np.array(bd_values) - np.median(bd_values))))

        # SFV: spacelike fraction variance (within-interval)
        sfv_values = []
        for (i, j, elems) in intervals[:100]:
            sfv_values.append(spacelike_fraction_var(C, elems))
        results["sfv_mad"] = float(np.median(np.abs(np.array(sfv_values) - np.median(sfv_values))))

        # WPEV: width profile entropy variance
        wpe_values = []
        for (i, j, elems) in intervals[:100]:
            wpe_values.append(width_profile_entropy(C, elems))
        results["wpev_mad"] = float(np.median(np.abs(np.array(wpe_values) - np.median(wpe_values))))

        # ISF: sideways fraction
        isf_values = []
        for (i, j, elems) in intervals[:100]:
            isf_values.append(sideways_fraction(C, i, j, elems))
        results["isf_mean"] = float(np.mean(isf_values))
        results["isf_mad"] = float(np.median(np.abs(np.array(isf_values) - np.median(isf_values))))
    else:
        for key in ["ofv_mad", "ofv_iqr", "ids_mad", "ids_kurtosis",
                     "ibdv_mad", "sfv_mad", "wpev_mad", "isf_mean", "isf_mad"]:
            results[key] = float("nan")

    return results


# ── CRN Trial ───────────────────────────────────────────────────────
def crn_trial(seed, metric_name, eps):
    rng = np.random.default_rng(seed)
    pts = sprinkle(N, T_DIAMOND, rng)

    C_flat = causal_flat(pts)
    fn_curved = METRIC_FNS[metric_name]
    C_curved = fn_curved(pts, eps)

    obs_flat = compute_all_observables(C_flat, pts)
    obs_curved = compute_all_observables(C_curved, pts)

    return obs_flat, obs_curved


# ── Main ────────────────────────────────────────────────────────────
def main():
    print(f"COMP-SCAN v8: N={N}, M={M}, metrics={list(METRICS.keys())}")
    t0 = time.time()

    all_results = {}

    for metric_name, eps in METRICS.items():
        print(f"\n=== Metric: {metric_name}, eps={eps} ===")
        flat_data = {}
        curved_data = {}

        for trial in range(M):
            seed = MASTER_SEED + trial * 1000 + (100 if metric_name == "ppwave_quad" else 300)
            obs_f, obs_c = crn_trial(seed, metric_name, eps)

            for key in obs_f:
                if key not in flat_data:
                    flat_data[key] = []
                    curved_data[key] = []
                flat_data[key].append(obs_f[key])
                curved_data[key].append(obs_c[key])

            if (trial + 1) % 5 == 0:
                print(f"  Trial {trial + 1}/{M} done ({time.time() - t0:.1f}s)")

        # Compute Cohen's d and Wilcoxon p for each observable
        metric_results = {}
        for key in sorted(flat_data.keys()):
            f_arr = np.array(flat_data[key])
            c_arr = np.array(curved_data[key])
            deltas = c_arr - f_arr

            # Skip if any NaN
            valid = ~(np.isnan(f_arr) | np.isnan(c_arr))
            if np.sum(valid) < 5:
                metric_results[key] = {"d": float("nan"), "p": 1.0, "n_valid": int(np.sum(valid))}
                continue

            deltas_v = deltas[valid]
            if np.std(deltas_v) == 0:
                d_val = 0.0
                p_val = 1.0
            else:
                d_val = float(np.mean(deltas_v) / np.std(deltas_v))
                try:
                    _, p_val = wilcoxon(deltas_v)
                    p_val = float(p_val)
                except ValueError:
                    p_val = 1.0

            metric_results[key] = {
                "d": round(d_val, 3),
                "p": round(p_val, 6),
                "mean_flat": round(float(np.mean(f_arr[valid])), 6),
                "mean_curved": round(float(np.mean(c_arr[valid])), 6),
                "n_valid": int(np.sum(valid)),
            }

        all_results[metric_name] = metric_results

    # Summary
    print(f"\n{'='*70}")
    print(f"COMP-SCAN v8 RESULTS (elapsed: {time.time()-t0:.1f}s)")
    print(f"{'='*70}")

    bonferroni_alpha = 0.01 / (len(METRICS) * 15)  # ~15 observables, 2 metrics
    print(f"Bonferroni alpha = {bonferroni_alpha:.6f}")

    for metric_name in METRICS:
        print(f"\n--- {metric_name} ---")
        print(f"{'Observable':<25} {'d':>8} {'p':>12} {'Status':<12}")
        for key, val in sorted(all_results[metric_name].items(),
                               key=lambda x: abs(x[1].get("d", 0) if not np.isnan(x[1].get("d", 0)) else 0),
                               reverse=True):
            if key in ("tc", "n_intervals"):
                continue
            d = val["d"]
            p = val["p"]
            if np.isnan(d):
                status = "NaN"
            elif abs(d) >= 0.5 and p < bonferroni_alpha:
                status = "SIGNAL ***"
            elif abs(d) >= 0.5 and p < 0.05:
                status = "SIGNAL *"
            elif abs(d) >= 0.3:
                status = "WEAK"
            else:
                status = "NULL"
            print(f"  {key:<25} {d:>8.3f} {p:>12.6f} {status:<12}")

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
