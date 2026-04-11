#!/usr/bin/env python3
"""
COMP-SCAN r8: Interval-internal + product-amplification probes.
N=500, M=20 CRN pairs, pp-wave eps=5 + Schwarzschild eps=0.005.

Observable groups:
  A (PRIORITY 1): Interval-internal probes — ordering_fraction_variance,
      interval_dim_scatter, interval_sideways_frac, interval_width_entropy_var,
      interval_causal_reach_product_gini
  B (PRIORITY 2): Product amplification — pf_product_gini,
      depth_height_product_gini, inout_degree_product_gini,
      cone_expansion_rate_product_gini
  C (PRIORITY 3): Chain/path extensions — chain_slope_no_k2, fan_entropy,
      path_count_entropy
  D: Proxy baselines — link_count, degree_cv, mean_degree, degree_var,
      degree_skew, degree_kurt, edge_count, max_degree, TC
  E: Positive controls — column_gini_C2, LVA

Conformal null: flat vs Omega^2=4 rescaled flat (must give d=0).
Bonferroni correction: alpha=0.01 / num_observables.

Output: docs/analysis_runs/run_20260326_125411/comp_scan_r8_results.json

Author: David Alfyorov
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import linalg as la
from scipy.stats import wilcoxon, skew, kurtosis, ttest_1samp

# ── Parameters ──────────────────────────────────────────────────────
N = 500
M = 20
T_DIAMOND = 1.0
MASTER_SEED = 88888  # distinct from all prior runs
METRICS = {
    "ppwave_quad": 5.0,
    "schwarzschild": 0.005,
}

RUN_DIR = Path("docs/analysis_runs/run_20260326_125411")
RESULTS_FILE = RUN_DIR / "comp_scan_r8_results.json"


# ── Sprinkling ──────────────────────────────────────────────────────
def sprinkle(N_target, T, rng):
    """Poisson sprinkling into Alexandrov diamond in d=4 Minkowski.
    Diamond: |t| + r < T/2.  Points sorted by t."""
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
    """Flat Minkowski causal matrix.  int16 for C^2 safety at N=500."""
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int16)
    for i in range(n):
        dt = pts[i + 1:, 0] - pts[i, 0]
        dr2 = np.sum((pts[i + 1:, 1:] - pts[i, 1:])**2, axis=1)
        C[i, i + 1:] = (dt**2 > dr2).astype(np.int16)
    return C


def causal_ppwave_quad(pts, eps):
    """pp-wave quadrupole metric.
    ds^2 = -du dv + (1 + eps*(x^2 - y^2))(dx^2 + dy^2).
    Causal: interval dt^2 - dr^2 - eps*f*du^2/2 > 0 with midpoint f."""
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int16)
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
        C[i, i + 1:] = (interval > 0).astype(np.int16)
    return C


def causal_schwarzschild(pts, eps):
    """Linearized Schwarzschild in isotropic coordinates.
    h_00 = -2*eps/r, h_ij = -2*eps/r * delta_ij.
    Softened r -> r + 0.3 to avoid 1/r singularity."""
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int16)
    for i in range(n):
        dt = pts[i + 1:, 0] - pts[i, 0]
        dx = pts[i + 1:, 1:] - pts[i, 1:]
        dr2 = np.sum(dx**2, axis=1)
        rm = np.sqrt(np.sum(((pts[i + 1:, 1:] + pts[i, 1:]) / 2)**2, axis=1))
        phi = -eps / (rm + 0.3)
        interval = (1 + 2 * phi) * dt**2 - (1 - 2 * phi) * dr2
        C[i, i + 1:] = (interval > 0).astype(np.int16)
    return C


def causal_conformal(pts, _omega_sq=4.0):
    """Conformally rescaled flat metric: ds^2 = Omega^2 * eta.
    Causal structure is IDENTICAL to flat — null cone unchanged.
    Used as null control: all deltas must be exactly 0."""
    return causal_flat(pts)


METRIC_FNS = {
    "ppwave_quad": causal_ppwave_quad,
    "schwarzschild": causal_schwarzschild,
    "conformal_null": causal_conformal,
}


# ── Helpers ─────────────────────────────────────────────────────────
def gini_coefficient(values):
    """Standard Gini coefficient on absolute values."""
    v = np.sort(np.abs(np.asarray(values, dtype=np.float64)))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v)) / (n * np.sum(v)) - (n + 1) / n)


def shannon_entropy(arr):
    """Shannon entropy of a probability-like array."""
    a = np.asarray(arr, dtype=np.float64)
    s = a.sum()
    if s <= 0:
        return 0.0
    p = a / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def mad(arr):
    """Median absolute deviation."""
    a = np.asarray(arr, dtype=np.float64)
    if len(a) == 0:
        return 0.0
    med = np.median(a)
    return float(np.median(np.abs(a - med)))


def build_link_mask(C, C2):
    """Link = causal relation with no mediating element."""
    return (C > 0) & (C2 == 0)


def longest_chains_from_set(C_bool, start_set):
    """BFS longest-chain distances from a set of starting elements.
    C_bool[i,j] = True iff i < j causally.  Returns array of distances."""
    n = C_bool.shape[0]
    dist = np.full(n, -1, dtype=np.int32)
    dist[list(start_set)] = 0
    # Process in topological order (already sorted by time)
    for j in range(n):
        if dist[j] < 0:
            continue
        # j's future successors
        futures = np.where(C_bool[j, :])[0]
        for k in futures:
            if dist[j] + 1 > dist[k]:
                dist[k] = dist[j] + 1
    return dist


# ── Observable computation ──────────────────────────────────────────
def compute_all_observables(C, pts):
    """Compute all observables for one causal set (C, pts)."""
    n = len(C)
    obs = {}

    # ── Pre-computation ──
    Cf = C.astype(np.float64)
    C2f = Cf @ Cf                        # float64 for algebra
    C2 = C.astype(np.int32) @ C.astype(np.int32)  # int32 for exact interval sizes
    tc = int(C.sum())
    obs["TC"] = tc

    # Link graph
    link_mask = build_link_mask(C, C2)
    A_link = link_mask.astype(np.float32) + link_mask.astype(np.float32).T
    degrees = A_link.sum(axis=1).astype(np.float64)
    in_degrees = link_mask.astype(np.float64).sum(axis=0)   # past links
    out_degrees = link_mask.astype(np.float64).sum(axis=1)  # future links

    # Past and future cone cardinalities (from C)
    past_sizes = Cf.sum(axis=0)    # |past(j)| = column sum
    future_sizes = Cf.sum(axis=1)  # |future(i)| = row sum

    # Interior: elements with both past and future
    interior = (past_sizes > 0) & (future_sizes > 0)
    n_int = int(interior.sum())

    # ══════════════════════════════════════════════════════════════
    # GROUP D: PROXY BASELINES
    # ══════════════════════════════════════════════════════════════
    link_count = int(degrees.sum()) // 2
    obs["link_count"] = link_count
    obs["degree_cv"] = float(degrees.std() / max(degrees.mean(), 1e-10))
    obs["mean_degree"] = float(degrees.mean())
    obs["degree_var"] = float(degrees.var())
    obs["degree_skew"] = float(skew(degrees)) if len(degrees) > 2 else 0.0
    obs["degree_kurt"] = float(kurtosis(degrees, fisher=True)) if len(degrees) > 3 else 0.0
    obs["edge_count"] = link_count
    obs["max_degree"] = float(degrees.max()) if len(degrees) > 0 else 0.0

    # ══════════════════════════════════════════════════════════════
    # GROUP E: POSITIVE CONTROLS
    # ══════════════════════════════════════════════════════════════

    # E1: column_gini_C2
    col_norms_C2 = np.sqrt(np.sum(C2f**2, axis=0))
    obs["column_gini_C2"] = gini_coefficient(col_norms_C2)

    # E2: LVA (Link Valence Anisotropy)
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

    # ══════════════════════════════════════════════════════════════
    # GROUP A: INTERVAL-INTERNAL PROBES
    # ══════════════════════════════════════════════════════════════

    # Sample causal pairs and their interval sizes
    causal_pairs = np.argwhere(C > 0)
    if len(causal_pairs) > 0:
        interval_sizes_all = C2[causal_pairs[:, 0], causal_pairs[:, 1]]
    else:
        interval_sizes_all = np.array([0], dtype=np.int32)

    # Use P75 of interval sizes for band selection (median is typically 0-1
    # at N=500 in d=4, too small for meaningful internal structure).
    n_p75 = int(np.percentile(interval_sizes_all, 75)) if len(interval_sizes_all) > 10 else 3
    n_band_lo = max(n_p75 - 2, 3)
    n_band_hi = max(n_p75 + 2, n_band_lo + 2)

    # ── A1: ordering_fraction_variance ──
    # For intervals with n elements in [n_band_lo, n_band_hi],
    # f(x,y) = 2*relations_in_I / (n*(n-1)).  Obs = Var(f).
    try:
        of_vals = []
        # Filter pairs in the P75 band
        band_mask = (interval_sizes_all >= n_band_lo) & \
                    (interval_sizes_all <= n_band_hi)
        band_pairs = causal_pairs[band_mask]
        # Sample up to 300 pairs for speed
        rng_local = np.random.default_rng(42)
        if len(band_pairs) > 300:
            idx_sample = rng_local.choice(len(band_pairs), 300, replace=False)
            band_pairs = band_pairs[idx_sample]
        for pair_idx in range(len(band_pairs)):
            xi, yi = band_pairs[pair_idx]
            # Interval I(x,y) = {z : x < z < y}
            above_x = C[xi, :] > 0  # x < z
            below_y = C[:, yi] > 0  # z < y
            interval_elts = np.where(above_x & below_y)[0]
            ni = len(interval_elts)
            if ni < 2:
                continue
            # Count relations within interval
            C_sub = C[np.ix_(interval_elts, interval_elts)]
            rels = int(C_sub.sum())
            f = 2.0 * rels / (ni * (ni - 1))
            of_vals.append(f)
        obs["ordering_fraction_variance"] = float(np.var(of_vals)) if len(of_vals) >= 5 else 0.0
    except Exception:
        obs["ordering_fraction_variance"] = 0.0

    # ── A2: interval_dim_scatter ──
    # For intervals with n >= 8, d_MM = log(2)/log(n/rel_norm).
    # In d=4 causal sets, ordering fractions ~0.1 give d_MM ~ 0.25-0.4.
    # Obs = MAD(d_MM).
    try:
        dim_vals = []
        large_mask = interval_sizes_all >= 8
        large_pairs = causal_pairs[large_mask]
        if len(large_pairs) > 300:
            idx_sample = rng_local.choice(len(large_pairs), 300, replace=False)
            large_pairs = large_pairs[idx_sample]
        for pair_idx in range(len(large_pairs)):
            xi, yi = large_pairs[pair_idx]
            above_x = C[xi, :] > 0
            below_y = C[:, yi] > 0
            interval_elts = np.where(above_x & below_y)[0]
            ni = len(interval_elts)
            if ni < 5:
                continue
            C_sub = C[np.ix_(interval_elts, interval_elts)]
            rels = int(C_sub.sum())
            if rels < 1:
                continue
            # Normalized relations per pair
            rel_norm = rels / max(ni * (ni - 1) / 2, 1)
            if rel_norm > 0 and rel_norm < 1:
                d_mm = np.log(2) / np.log(1.0 / rel_norm)
                if d_mm > 0 and d_mm < 20:  # permissive filter
                    dim_vals.append(d_mm)
        obs["interval_dim_scatter"] = mad(dim_vals) if len(dim_vals) >= 5 else 0.0
    except Exception:
        obs["interval_dim_scatter"] = 0.0

    # ── A3: interval_sideways_frac ──
    # For link triples x→y link, y→z link, S = |I(x,z)| - |I(x,y)| - |I(y,z)| - 2.
    # Obs = mean(S / max(1, |I(x,z)|-2)).
    try:
        sf_vals = []
        # Sample link triples: pick x, take a future link y, then a future link of y
        link_rows, link_cols = np.where(link_mask)
        if len(link_rows) > 0:
            # Build adjacency list of future links
            future_link_list = defaultdict(list)
            for r, c in zip(link_rows, link_cols):
                future_link_list[int(r)].append(int(c))
            # Sample triples
            triple_count = 0
            for x in rng_local.permutation(list(future_link_list.keys()))[:200]:
                y_list = future_link_list[x]
                for y in y_list:
                    if y not in future_link_list:
                        continue
                    z_list = future_link_list[y]
                    for z in z_list[:3]:  # limit per y
                        if C[x, z] <= 0:
                            continue
                        I_xz = int(C2[x, z])
                        I_xy = int(C2[x, y])
                        I_yz = int(C2[y, z])
                        S = I_xz - I_xy - I_yz - 2
                        denom = max(1, I_xz - 2)
                        sf_vals.append(S / denom)
                        triple_count += 1
                        if triple_count >= 500:
                            break
                    if triple_count >= 500:
                        break
                if triple_count >= 500:
                    break
        obs["interval_sideways_frac"] = float(np.mean(sf_vals)) if len(sf_vals) >= 5 else 0.0
    except Exception:
        obs["interval_sideways_frac"] = 0.0

    # ── A4: interval_width_entropy_var ──
    # For intervals in [n_band_lo, n_band_hi], BFS layer sizes from bottom,
    # Shannon entropy H of layer sizes.  Obs = Var(H).
    try:
        ent_vals = []
        band_mask2 = (interval_sizes_all >= n_band_lo) & \
                     (interval_sizes_all <= n_band_hi)
        band_pairs2 = causal_pairs[band_mask2]
        if len(band_pairs2) > 200:
            idx_sample = rng_local.choice(len(band_pairs2), 200, replace=False)
            band_pairs2 = band_pairs2[idx_sample]
        for pair_idx in range(len(band_pairs2)):
            xi, yi = band_pairs2[pair_idx]
            above_x = C[xi, :] > 0
            below_y = C[:, yi] > 0
            interval_elts = np.where(above_x & below_y)[0]
            ni = len(interval_elts)
            if ni < 3:
                continue
            # BFS layers from bottom of interval
            # "bottom" = elements that have no past within the interval
            C_sub = C[np.ix_(interval_elts, interval_elts)]
            has_past = (C_sub.sum(axis=0) > 0)
            current_layer = set(np.where(~has_past)[0])
            visited = set(current_layer)
            layer_sizes = [len(current_layer)]
            while current_layer:
                next_layer = set()
                for el in current_layer:
                    futures = np.where(C_sub[el, :] > 0)[0]
                    for f in futures:
                        if f not in visited:
                            next_layer.add(f)
                            visited.add(f)
                if not next_layer:
                    break
                layer_sizes.append(len(next_layer))
                current_layer = next_layer
            if len(layer_sizes) >= 2:
                H = shannon_entropy(np.array(layer_sizes, dtype=float))
                ent_vals.append(H)
        obs["interval_width_entropy_var"] = float(np.var(ent_vals)) if len(ent_vals) >= 5 else 0.0
    except Exception:
        obs["interval_width_entropy_var"] = 0.0

    # ── A5: interval_causal_reach_product_gini ──
    # Within intervals of size n >= 8, for each w: past_reach(w)*future_reach(w).
    # Gini of products.
    try:
        crpg_vals = []
        large_mask2 = interval_sizes_all >= 8
        large_pairs2 = causal_pairs[large_mask2]
        if len(large_pairs2) > 150:
            idx_sample = rng_local.choice(len(large_pairs2), 150, replace=False)
            large_pairs2 = large_pairs2[idx_sample]
        for pair_idx in range(len(large_pairs2)):
            xi, yi = large_pairs2[pair_idx]
            above_x = C[xi, :] > 0
            below_y = C[:, yi] > 0
            interval_elts = np.where(above_x & below_y)[0]
            ni = len(interval_elts)
            if ni < 10:
                continue
            C_sub = C[np.ix_(interval_elts, interval_elts)]
            past_reach = C_sub.astype(np.float64).sum(axis=0)   # column sums
            future_reach = C_sub.astype(np.float64).sum(axis=1) # row sums
            products = past_reach * future_reach
            products = products[products > 0]
            if len(products) >= 5:
                crpg_vals.append(gini_coefficient(products))
        obs["interval_causal_reach_product_gini"] = float(np.mean(crpg_vals)) if len(crpg_vals) >= 3 else 0.0
    except Exception:
        obs["interval_causal_reach_product_gini"] = 0.0

    # ══════════════════════════════════════════════════════════════
    # GROUP B: PRODUCT AMPLIFICATION
    # ══════════════════════════════════════════════════════════════

    # ── B1: pf_product_gini ──
    # Gini({|past(x)| * |future(x)| : x in interior})
    try:
        if n_int > 10:
            pf_prod = past_sizes[interior] * future_sizes[interior]
            obs["pf_product_gini"] = gini_coefficient(pf_prod)
        else:
            obs["pf_product_gini"] = 0.0
    except Exception:
        obs["pf_product_gini"] = 0.0

    # ── B2: depth_height_product_gini ──
    # depth(x) = longest chain from any minimal to x
    # height(x) = longest chain from x to any maximal
    # Gini({depth*height : interior})
    try:
        C_bool = C > 0
        # Minimals: no past in C
        minimals = set(np.where(Cf.sum(axis=0) == 0)[0])
        # Maximals: no future in C
        maximals = set(np.where(Cf.sum(axis=1) == 0)[0])

        # Depth: longest chain from minimals (forward pass)
        depth = np.zeros(n, dtype=np.int32)
        for j in range(n):
            # predecessors
            preds = np.where(C_bool[:, j])[0]
            if len(preds) > 0:
                depth[j] = depth[preds].max() + 1

        # Height: longest chain to maximals (backward pass)
        height = np.zeros(n, dtype=np.int32)
        for j in range(n - 1, -1, -1):
            succs = np.where(C_bool[j, :])[0]
            if len(succs) > 0:
                height[j] = height[succs].max() + 1

        if n_int > 10:
            dh_prod = depth[interior].astype(np.float64) * height[interior].astype(np.float64)
            obs["depth_height_product_gini"] = gini_coefficient(dh_prod)
        else:
            obs["depth_height_product_gini"] = 0.0
    except Exception:
        obs["depth_height_product_gini"] = 0.0

    # ── B3: inout_degree_product_gini ──
    # Gini({k_in(x) * k_out(x) : x in link graph interior})
    try:
        interior_link = (in_degrees > 0) & (out_degrees > 0)
        if interior_link.sum() > 10:
            io_prod = in_degrees[interior_link] * out_degrees[interior_link]
            obs["inout_degree_product_gini"] = gini_coefficient(io_prod)
        else:
            obs["inout_degree_product_gini"] = 0.0
    except Exception:
        obs["inout_degree_product_gini"] = 0.0

    # ── B4: cone_expansion_rate_product_gini ──
    # expansion = |past_at_dist_2(x)| / |past_at_dist_1(x)|, future analog.
    # Obs = Gini(r_past * r_future).
    try:
        # past_at_dist_1 = direct predecessors in C = past links
        # past_at_dist_2 = predecessors at exactly distance 2 (via C2 > 0 and C == 0)
        # For speed: past_1(x) = C[:,x].sum (column), future_1(x) = C[x,:].sum (row)
        # past_2(x) = number of z with C2[z,x]>0 and C[z,x]==0 (not directly linked)
        # But C2[z,x] counts paths of length 2, not distance.
        # Simpler: past_at_dist_1 = column sum of link_mask
        # past_at_dist_2 = column sum of (C - link_mask) restricted to transitives at distance 2
        # Use C2 approach: past2(x) = #{z: exists w with C[z,w]=1, C[w,x]=1, C[z,x]=0}
        #                            = (link^T @ link)[z,x] minus direct links
        link_f = link_mask.astype(np.float64)
        link2 = link_f.T @ link_f  # link2[z,x] = #(2-link-paths from z to x)
        # past_1[x] = number of past links
        past_1 = link_f.sum(axis=0)      # column sums of link_mask
        future_1 = link_f.sum(axis=1)    # row sums of link_mask
        # past_2[x] = number of elements at link-distance 2 in past
        past_2 = (link2 > 0).sum(axis=0).astype(np.float64) - past_1  # subtract dist-1
        past_2 = np.maximum(past_2, 0)
        # future direction: link @ link
        link2_fut = link_f @ link_f  # link2_fut[x,z] = 2-link-paths from x to z
        future_2 = (link2_fut > 0).sum(axis=1).astype(np.float64) - future_1
        future_2 = np.maximum(future_2, 0)

        r_past = past_2 / np.maximum(past_1, 1)
        r_future = future_2 / np.maximum(future_1, 1)
        valid_exp = (past_1 > 0) & (future_1 > 0) & (past_2 > 0) & (future_2 > 0)
        if valid_exp.sum() > 10:
            products = r_past[valid_exp] * r_future[valid_exp]
            obs["cone_expansion_rate_product_gini"] = gini_coefficient(products)
        else:
            obs["cone_expansion_rate_product_gini"] = 0.0
    except Exception:
        obs["cone_expansion_rate_product_gini"] = 0.0

    # ══════════════════════════════════════════════════════════════
    # GROUP C: CHAIN/PATH EXTENSIONS
    # ══════════════════════════════════════════════════════════════

    # ── C1: chain_slope_no_k2 ──
    # N_k for k=3,4.  Obs = log(N_4 / N_3).  Excludes N_2=TC.
    try:
        # N_3 = 1^T C^2 1
        N3 = float(C2f.sum())
        # N_4 = 1^T C^3 1
        C3f = C2f @ Cf
        N4 = float(C3f.sum())
        if N3 > 0 and N4 > 0:
            obs["chain_slope_no_k2"] = float(np.log(N4 / N3))
        else:
            obs["chain_slope_no_k2"] = 0.0
    except Exception:
        obs["chain_slope_no_k2"] = 0.0

    # ── C2: fan_entropy ──
    # For each x with >= 4 future links y_i, Shannon entropy of {|J+(y_i)|}.
    # Obs = mean entropy.
    try:
        fan_ent_vals = []
        for x in range(n):
            future_links = np.where(link_mask[x, :])[0]
            if len(future_links) < 4:
                continue
            # |J+(y_i)| = future cone size of each future link
            fsizes = future_sizes[future_links]
            fsizes = fsizes[fsizes > 0]
            if len(fsizes) >= 4:
                H = shannon_entropy(fsizes)
                fan_ent_vals.append(H)
        obs["fan_entropy"] = float(np.mean(fan_ent_vals)) if len(fan_ent_vals) >= 5 else 0.0
    except Exception:
        obs["fan_entropy"] = 0.0

    # ── C3: path_count_entropy ──
    # P(x) = total number of directed paths through x (via DP).
    # Shannon entropy of {log2(P(x)+1)}.
    try:
        # DP for path counts: forward pass for number of paths ending at x,
        # backward pass for number of paths starting at x.
        # paths_to[x] = sum over predecessors of paths_to[pred] (links only for efficiency)
        # paths_from[x] = sum over successors of paths_from[succ]
        paths_to = np.ones(n, dtype=np.float64)   # each element is a path of length 0
        for j in range(n):
            preds = np.where(link_mask[:, j])[0]  # past links to j
            if len(preds) > 0:
                paths_to[j] += paths_to[preds].sum()

        paths_from = np.ones(n, dtype=np.float64)
        for j in range(n - 1, -1, -1):
            succs = np.where(link_mask[j, :])[0]  # future links from j
            if len(succs) > 0:
                paths_from[j] += paths_from[succs].sum()

        path_through = paths_to * paths_from
        log_path = np.log2(path_through + 1)
        log_path_int = log_path[interior]
        if len(log_path_int) > 10:
            obs["path_count_entropy"] = shannon_entropy(log_path_int)
        else:
            obs["path_count_entropy"] = 0.0
    except Exception:
        obs["path_count_entropy"] = 0.0

    return obs


# ── CRN Engine ──────────────────────────────────────────────────────
def run_crn_scan():
    rng = np.random.default_rng(MASTER_SEED)
    results = {}

    # Add conformal null as a metric with eps=0
    all_metrics = dict(METRICS)
    all_metrics["conformal_null"] = 0.0

    for metric_name, eps in all_metrics.items():
        metric_fn = METRIC_FNS[metric_name]
        print(f"\n{'='*60}")
        print(f"Metric: {metric_name}, eps={eps}")
        print(f"{'='*60}")

        deltas = {}
        flat_means = {}
        curved_means = {}

        t_metric = time.time()
        for trial in range(M):
            seed_trial = rng.integers(0, 2**31)
            rng_trial = np.random.default_rng(seed_trial)

            # Sprinkle points (same for flat and curved — CRN)
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
                    flat_means[key] = []
                    curved_means[key] = []
                deltas[key].append(obs_curved[key] - obs_flat[key])
                flat_means[key].append(obs_flat[key])
                curved_means[key].append(obs_curved[key])

            if (trial + 1) % 5 == 0:
                elapsed_trial = time.time() - t_metric
                rate = (trial + 1) / elapsed_trial
                remaining = (M - trial - 1) / max(rate, 0.01)
                print(f"  Trial {trial + 1}/{M} done ({elapsed_trial:.1f}s, ~{remaining:.0f}s remaining)")

        # Compute statistics
        metric_results = {}
        num_obs = len(deltas)
        alpha_bonferroni = 0.01 / max(num_obs, 1)

        for key, delta_list in deltas.items():
            d_arr = np.array(delta_list)
            mean_d = d_arr.mean()
            std_d = d_arr.std(ddof=1) if len(d_arr) > 1 else 1e-10
            cohen_d = mean_d / max(std_d, 1e-10)

            # Wilcoxon signed-rank test
            try:
                if np.all(d_arr == 0):
                    p_wilc = 1.0
                else:
                    _, p_wilc = wilcoxon(d_arr, alternative='two-sided')
            except Exception:
                p_wilc = 1.0

            # t-test for paired samples
            try:
                if std_d > 1e-10:
                    _, p_t = ttest_1samp(d_arr, 0)
                else:
                    p_t = 1.0 if abs(mean_d) < 1e-10 else 0.0
            except Exception:
                p_t = 1.0

            metric_results[key] = {
                "cohen_d": round(float(cohen_d), 4),
                "mean_delta": round(float(mean_d), 8),
                "std_delta": round(float(std_d), 8),
                "p_wilcoxon": round(float(p_wilc), 8),
                "p_ttest": round(float(p_t), 8),
                "significant_bonferroni": bool(min(p_wilc, p_t) < alpha_bonferroni),
                "mean_flat": round(float(np.mean(flat_means[key])), 8),
                "mean_curved": round(float(np.mean(curved_means[key])), 8),
                "deltas": [round(float(d), 8) for d in delta_list],
            }

        results[metric_name] = metric_results

        elapsed_metric = time.time() - t_metric
        print(f"  {metric_name} complete in {elapsed_metric:.1f}s")

    return results


# ── Main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"COMP-SCAN r8 — Run #008")
    print(f"N={N}, M={M}, metrics: {list(METRICS.keys())} + conformal_null")
    print(f"Master seed: {MASTER_SEED}")
    print(f"Output: {RESULTS_FILE}")
    t0 = time.time()

    results = run_crn_scan()

    elapsed = time.time() - t0

    # ── Bonferroni alpha ──
    sample_key = list(results.keys())[0]
    num_obs = len(results[sample_key])
    alpha_bonf = 0.01 / num_obs

    # ── Print ranked results ──
    print(f"\n{'='*80}")
    print(f"COMP-SCAN r8 RESULTS (elapsed: {elapsed:.1f}s)")
    print(f"Bonferroni alpha = 0.01/{num_obs} = {alpha_bonf:.6f}")
    print(f"{'='*80}")

    BASELINES = {"TC", "link_count", "degree_cv", "mean_degree", "degree_var",
                 "degree_skew", "degree_kurt", "edge_count", "max_degree"}
    CONTROLS = {"column_gini_C2", "lva"}
    GROUP_A = {"ordering_fraction_variance", "interval_dim_scatter",
               "interval_sideways_frac", "interval_width_entropy_var",
               "interval_causal_reach_product_gini"}
    GROUP_B = {"pf_product_gini", "depth_height_product_gini",
               "inout_degree_product_gini", "cone_expansion_rate_product_gini"}
    GROUP_C = {"chain_slope_no_k2", "fan_entropy", "path_count_entropy"}

    for metric_name, metric_results in results.items():
        print(f"\n--- {metric_name} ---")
        print(f"  {'Observable':42s}  {'d':>8s}  {'p_wilc':>10s}  {'p_t':>10s}  {'Tag'}")
        print(f"  {'-'*42}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}")

        ranked = sorted(metric_results.items(),
                       key=lambda x: abs(x[1]["cohen_d"]), reverse=True)
        for key, vals in ranked:
            tag = ""
            if key in BASELINES:
                tag = "BASELINE"
            elif key in CONTROLS:
                tag = "CONTROL+"
            elif key in GROUP_A:
                tag = "A:interval"
            elif key in GROUP_B:
                tag = "B:product"
            elif key in GROUP_C:
                tag = "C:chain"

            sig = ""
            if vals["significant_bonferroni"]:
                sig = " ***"
            elif min(vals["p_wilcoxon"], vals["p_ttest"]) < 0.01:
                sig = " **"
            elif min(vals["p_wilcoxon"], vals["p_ttest"]) < 0.05:
                sig = " *"

            p_w = vals["p_wilcoxon"]
            p_t = vals["p_ttest"]
            p_w_str = f"{p_w:.2e}" if p_w < 0.01 else f"{p_w:.4f}"
            p_t_str = f"{p_t:.2e}" if p_t < 0.01 else f"{p_t:.4f}"

            print(f"  {key:42s}  {vals['cohen_d']:+8.3f}  {p_w_str:>10s}  {p_t_str:>10s}  {tag}{sig}")

    # ── Conformal null validation ──
    if "conformal_null" in results:
        print(f"\n{'='*60}")
        print("CONFORMAL NULL VALIDATION")
        print(f"{'='*60}")
        null_results = results["conformal_null"]
        max_null_d = max(abs(v["cohen_d"]) for v in null_results.values())
        null_failures = [k for k, v in null_results.items() if abs(v["cohen_d"]) > 0.3]
        if null_failures:
            print(f"  WARNING: {len(null_failures)} observables have |d| > 0.3 on conformal null!")
            for k in null_failures:
                print(f"    {k}: d = {null_results[k]['cohen_d']:+.4f}")
        else:
            print(f"  PASS: All observables have |d| <= 0.3 (max |d| = {max_null_d:.4f})")

    # ── Summary counts ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for metric_name, metric_results in results.items():
        if metric_name == "conformal_null":
            continue
        sig_count = sum(1 for v in metric_results.values() if v["significant_bonferroni"])
        strong = sum(1 for v in metric_results.values()
                     if abs(v["cohen_d"]) >= 0.8 and v["significant_bonferroni"])
        new_sig = sum(1 for k, v in metric_results.items()
                      if v["significant_bonferroni"] and k not in BASELINES and k not in CONTROLS)
        print(f"  {metric_name}: {sig_count} significant (Bonferroni), "
              f"{strong} strong (|d|>=0.8), {new_sig} new (non-baseline/control)")

    # ── Save ──
    output = {
        "run": "008",
        "version": "comp_scan_r8",
        "N": N,
        "M": M,
        "master_seed": MASTER_SEED,
        "metrics": {**METRICS, "conformal_null": 0.0},
        "bonferroni_alpha": round(alpha_bonf, 8),
        "elapsed_seconds": round(elapsed, 1),
        "observable_groups": {
            "A_interval_internal": sorted(GROUP_A),
            "B_product_amplification": sorted(GROUP_B),
            "C_chain_path": sorted(GROUP_C),
            "D_baselines": sorted(BASELINES),
            "E_positive_controls": sorted(CONTROLS),
        },
        "results": results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Total elapsed: {elapsed:.1f}s")
