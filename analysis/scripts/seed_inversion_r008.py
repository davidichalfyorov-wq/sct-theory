#!/usr/bin/env python3
"""
SEED INVERSION: Novel Observable Candidates.

Structural inversion: existing certified observables all measure INEQUALITY
of scalar distributions (Gini, skewness, kurtosis, variance ratios).
These 6 candidates target DIFFERENT information channels:

  SI-1: fan_moran_I         — Spatial autocorrelation of detrended fan sizes
  SI-2: diamond_aspect_cv   — CV of causal diamond height/width ratios
  SI-3: level_spacing_ratio — Mean eigenvalue spacing ratio (RMT universality)
  SI-4: pf_residual_var     — Past-future curve residual variance
  SI-5: chain_direction_cv  — Directional anisotropy of future chains
  SI-6: eff_resistance_cv   — Effective resistance variation on link graph

None is "Gini/kurtosis of a scalar distribution." Each taps a
structurally different information channel.

NOTE: IPR of C² singular vectors (svd_vec_ipr) was ALREADY tested in
COMP-SCAN v8 and FAILED (NOT_SIG for both metrics). Excluded.

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
import time, gc

# ═══════════════════════════════════════════════════════════════════════
# OBSERVABLE SI-1: FAN MORAN'S I (Spatial Autocorrelation)
# ═══════════════════════════════════════════════════════════════════════
#
# MOTIVATION: All certified observables measure marginal distribution
# shape. Moran's I measures PAIRWISE spatial correlation: do nearby
# elements have correlated fan anomalies? Curvature creates coherent
# anisotropy fields — Moran's I detects the spatial coherence.
#
# FORMULA:
#   f_i = |past(i)|. Detrend by height: δf_i = f_i - mean(f | height=h(i)).
#   Weight: w_{ij} = 1 if i,j linked AND |h(i)-h(j)| ≤ 1.
#   I = (N/W) × Σ_{ij} w_{ij} δf_i δf_j / Σ_i δf_i²
#
# EXPECTED:
#   Flat: I ≈ 0 (no spatial coherence in Poisson noise)
#   Schwarzschild: I > 0 (radial curvature gradient creates coherent fan modulation)
#   pp-wave: I > 0 but smaller (transverse gradient is weaker)
#
# CONFIDENCE: MEDIUM
# REFERENCE: Moran (1950); spatial statistics on graphs: Griffith (2003)
# ═══════════════════════════════════════════════════════════════════════

def compute_fan_moran_I(C, L_dir_csc, L_dir_csr, heights, A_link):
    """Compute detrended Moran's I of past-sizes on the link graph."""
    N = C.shape[0]
    past_sizes = np.array(C.sum(axis=0)).ravel().astype(np.float64)

    # Detrend by height
    delta_f = np.zeros(N)
    for h in range(int(np.max(heights)) + 1):
        mask = heights == h
        if np.sum(mask) >= 2:
            mu_h = np.mean(past_sizes[mask])
            delta_f[mask] = past_sizes[mask] - mu_h
        else:
            delta_f[mask] = 0.0

    ss_total = np.sum(delta_f ** 2)
    if ss_total < 1e-15:
        return 0.0

    # Weight matrix: linked elements within 1 height step
    A_sp = sp.csr_matrix(A_link, dtype=np.float64)
    rows, cols = A_sp.nonzero()
    height_close = np.abs(heights[rows] - heights[cols]) <= 1
    w_sum = 0.0
    numerator = 0.0
    for idx in range(len(rows)):
        if height_close[idx]:
            i, j = rows[idx], cols[idx]
            numerator += delta_f[i] * delta_f[j]
            w_sum += 1.0

    if w_sum < 1:
        return 0.0

    I = (N / w_sum) * numerator / ss_total
    return float(I)


# ═══════════════════════════════════════════════════════════════════════
# OBSERVABLE SI-2: DIAMOND ASPECT RATIO CV
# ═══════════════════════════════════════════════════════════════════════
#
# MOTIVATION: Measures the SHAPE of causal intervals (diamonds), not
# just per-element scalars. Curvature causes geodesic focusing/defocusing
# (Raychaudhuri equation), which stretches or compresses diamonds.
#
# FORMULA:
#   For sampled pairs (x,y) with x ≺ y and 5 ≤ C²[x,y] ≤ 60:
#     height(x,y) = longest chain from x to y within I(x,y)
#     width(x,y) = max antichain size in I(x,y) [via Dilworth/matching]
#     α(x,y) = height / width
#   Observable = Std(α) / Mean(α)  [CV of aspect ratios]
#
# EXPECTED:
#   Flat: Low CV (diamonds are statistically similar)
#   Schwarzschild: High CV (diamonds near horizon are stretched)
#   pp-wave: Moderate CV (transverse compression)
#
# CONFIDENCE: MEDIUM-HIGH
# REFERENCE: Myrheim (1978); Brightwell & Gregory (1991)
# ═══════════════════════════════════════════════════════════════════════

def _longest_chain_in_interval(C_dense, members):
    """Longest chain in sub-DAG defined by 'members' index array.
    Uses DP on the sub-DAG. Returns chain length (number of relations)."""
    n = len(members)
    if n <= 1:
        return 0
    # Build sub-DAG adjacency
    idx_map = {m: i for i, m in enumerate(members)}
    # DP: dp[i] = longest chain ending at sub-element i
    dp = np.zeros(n, dtype=int)
    # Elements are already sorted by natural order (C is upper triangular)
    for ii in range(n):
        for jj in range(ii + 1, n):
            if C_dense[members[ii], members[jj]] > 0:
                dp[jj] = max(dp[jj], dp[ii] + 1)
    return int(np.max(dp))


def _max_antichain_size(C_dense, members):
    """Max antichain size via Dilworth: width = |S| - max_matching.
    For small intervals (|S| < 80), use bipartite matching."""
    n = len(members)
    if n <= 1:
        return n
    # Build comparability bipartite graph: left=sources, right=targets
    # Edge (i,j) if members[i] ≺ members[j]
    from scipy.optimize import linear_sum_assignment
    adj = np.zeros((n, n), dtype=int)
    for ii in range(n):
        for jj in range(ii + 1, n):
            if C_dense[members[ii], members[jj]] > 0:
                adj[ii, jj] = 1
    # Max matching in bipartite graph = min vertex cover (Konig)
    # Width = n - max_matching
    # Use Hungarian on cost = -adj
    cost = 1 - adj[:, :]  # 0 where edge exists, 1 where not
    row_ind, col_ind = linear_sum_assignment(cost)
    matching = np.sum(adj[row_ind, col_ind])
    return n - matching


def compute_diamond_aspect_cv(C, C2, rng, n_samples=200):
    """Compute CV of causal diamond aspect ratios."""
    N = C.shape[0]
    C_dense = C if isinstance(C, np.ndarray) else C.toarray()
    C2_dense = C2 if isinstance(C2, np.ndarray) else C2.toarray()

    # Find pairs with interval size in [5, 60]
    ii, jj = np.where((C_dense > 0) & (C2_dense >= 5) & (C2_dense <= 60))
    if len(ii) < 20:
        return 0.0

    # Sample
    n_use = min(len(ii), n_samples)
    idx = rng.choice(len(ii), n_use, replace=False)
    ii, jj = ii[idx], jj[idx]

    aspects = []
    for a, b in zip(ii, jj):
        # Members of interval I(a,b): elements k with a ≺ k ≺ b
        between = np.where((C_dense[a, :] > 0) & (C_dense[:, b] > 0))[0]
        between = between[(between != a) & (between != b)]
        if len(between) < 3:
            continue
        members = np.sort(between)
        h = _longest_chain_in_interval(C_dense, members)
        if h < 1:
            continue
        w = _max_antichain_size(C_dense, members)
        if w < 1:
            continue
        aspects.append(h / w)

    if len(aspects) < 10:
        return 0.0

    aspects = np.array(aspects)
    mu = np.mean(aspects)
    if mu < 1e-10:
        return 0.0
    return float(np.std(aspects) / mu)


# ═══════════════════════════════════════════════════════════════════════
# OBSERVABLE SI-3: LEVEL SPACING RATIO (Random Matrix Theory)
# ═══════════════════════════════════════════════════════════════════════
#
# MOTIVATION: Eigenvalue SPACING statistics encode symmetry structure.
# GOE (chaotic, symmetric): <r> ≈ 0.5307.
# Poisson (integrable, no repulsion): <r> ≈ 0.3863.
# Curvature breaks the statistical symmetry of the causal set →
# the spectral statistics of the link Laplacian should shift.
#
# FORMULA:
#   L = D - A (link graph Laplacian). Eigenvalues λ_1 ≤ ... ≤ λ_N.
#   Spacings: δ_n = λ_{n+1} - λ_n.
#   Ratio: r_n = min(δ_n, δ_{n+1}) / max(δ_n, δ_{n+1}).
#   Observable = mean(r_n) over bulk spectrum (n ∈ [N/4, 3N/4]).
#
# EXPECTED:
#   Flat: Near GOE (~0.53) due to maximal symmetry
#   Schwarzschild: Shift toward Poisson (symmetry breaking)
#   pp-wave: Smaller shift (partial symmetry breaking)
#
# CONFIDENCE: MEDIUM
# REFERENCE: Oganesyan & Huse (2007); Atas et al. (2013)
# ═══════════════════════════════════════════════════════════════════════

def compute_level_spacing_ratio(A_link, degrees):
    """Mean level spacing ratio of the link Laplacian bulk spectrum."""
    N = A_link.shape[0]
    A_dense = A_link.toarray() if sp.issparse(A_link) else A_link
    L = np.diag(degrees.astype(np.float64)) - A_dense

    eigs = np.sort(np.linalg.eigvalsh(L))

    # Use bulk: indices from N//4 to 3*N//4
    lo, hi = N // 4, 3 * N // 4
    if hi - lo < 10:
        return 0.0

    bulk = eigs[lo:hi]
    spacings = np.diff(bulk)
    if len(spacings) < 3:
        return 0.0

    # Remove zero spacings (degeneracies)
    spacings = spacings[spacings > 1e-14]
    if len(spacings) < 3:
        return 0.0

    ratios = []
    for i in range(len(spacings) - 1):
        s1, s2 = spacings[i], spacings[i + 1]
        r = min(s1, s2) / max(s1, s2)
        ratios.append(r)

    return float(np.mean(ratios)) if ratios else 0.0


# ═══════════════════════════════════════════════════════════════════════
# OBSERVABLE SI-4: PAST-FUTURE RESIDUAL VARIANCE
# ═══════════════════════════════════════════════════════════════════════
#
# MOTIVATION: In flat d=4 Minkowski, |past(x)|/N and |future(x)|/N
# lie on a specific curve determined by the dimension. Curvature
# distorts this curve. The EXCESS variance of future-sizes relative
# to the flat-space prediction measures the integrated curvature effect.
#
# FORMULA (CRN version):
#   For each element x: p_x = |past(x)|/N, f_x = |future(x)|/N.
#   From the FLAT reference, fit LOESS: f_flat(p).
#   In curved sample, residuals: r_x = f_x^{curved} - f_flat(p_x^{curved}).
#   Observable = Var(r) / Var(f^{curved})
#
# SIMPLE VERSION (no CRN pairing, just one sample):
#   Fit f = (1 - p^{1/4})^4 (d=4 model). Measure R²_flat_model.
#   Observable = 1 - R² (fraction of unexplained variance).
#
# EXPECTED:
#   Flat: ~0 (model fits well, only Poisson noise)
#   Schwarzschild: >> 0 (curvature distorts p-f locus)
#   pp-wave: > 0 but smaller
#
# CONFIDENCE: MEDIUM-HIGH
# REFERENCE: Myrheim (1978); Bombelli, Lee, Meyer, Sorkin (1987)
# ═══════════════════════════════════════════════════════════════════════

def compute_pf_residual_var(C):
    """Past-future residual variance: how well does a smooth f(p) fit?

    Uses isotonic regression (monotone decreasing) to find the best
    monotone fit f_hat(p), then measures residual scatter. In flat space
    the fit is tight (low residual). Curvature creates scatter.

    Returns the coefficient of variation of residuals from isotonic fit.
    """
    N = C.shape[0]
    past_sizes = np.array(C.sum(axis=0)).ravel().astype(np.float64)
    future_sizes = np.array(C.sum(axis=1)).ravel().astype(np.float64)

    p = past_sizes / N
    f = future_sizes / N

    # Exclude extreme boundary elements
    mask = (p > 0.03) & (p < 0.97) & (f > 0.005)
    if np.sum(mask) < 30:
        return 0.0

    p_bulk = p[mask]
    f_bulk = f[mask]

    # Empirical approach: bin p into 20 bins, compute mean f per bin,
    # then measure scatter WITHIN bins. This is the unexplained variance.
    n_bins = 20
    bin_edges = np.linspace(np.min(p_bulk), np.max(p_bulk), n_bins + 1)
    bin_idx = np.digitize(p_bulk, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    residuals = np.zeros_like(f_bulk)
    for b in range(n_bins):
        bm = bin_idx == b
        if np.sum(bm) >= 3:
            residuals[bm] = f_bulk[bm] - np.mean(f_bulk[bm])

    # Fraction of variance not explained by the smooth p→f relationship
    var_resid = np.var(residuals)
    var_total = np.var(f_bulk)
    if var_total < 1e-15:
        return 0.0

    return float(var_resid / var_total)


# ═══════════════════════════════════════════════════════════════════════
# OBSERVABLE SI-5: CHAIN DIRECTION CV (Directional Anisotropy)
# ═══════════════════════════════════════════════════════════════════════
#
# MOTIVATION: For each element x, its outgoing links define "directions"
# into the future. The longest chain reachable from each linked successor
# measures the "reach" in that direction. In flat space, all directions
# have similar reach. Curvature creates directional asymmetry
# (geodesic focusing/defocusing).
#
# FORMULA:
#   For each element x with ≥ 3 out-links:
#     For each out-link y: L(y) = longest chain from y to any maximal element.
#     cv(x) = Std({L(y)}) / Mean({L(y)}).
#   Observable = Median(cv(x)) over all qualifying elements.
#
# EXPECTED:
#   Flat: Low median cv (isotropic future cones)
#   Schwarzschild: High median cv (radial vs tangential asymmetry)
#   pp-wave: Moderate median cv (transverse compression)
#
# CONFIDENCE: MEDIUM
# REFERENCE: Raychaudhuri equation; geodesic deviation in causal sets
# ═══════════════════════════════════════════════════════════════════════

def compute_chain_direction_cv(C, L_dir_csr, heights):
    """Median CV of longest-chain-to-max from each link direction."""
    N = C.shape[0]
    h_max = int(np.max(heights))

    # Longest chain from each element to any maximal element.
    # Compute via reverse DP on the link DAG.
    chain_to_max = np.zeros(N, dtype=int)
    for i in range(N - 1, -1, -1):
        children = L_dir_csr.getrow(i).indices
        if len(children) > 0:
            chain_to_max[i] = 1 + int(np.max(chain_to_max[children]))
        # else: chain_to_max[i] = 0 (maximal element)

    # For each element, compute CV of chain_to_max of its out-links
    cvs = []
    for x in range(N):
        children = L_dir_csr.getrow(x).indices
        if len(children) < 3:
            continue
        reaches = chain_to_max[children].astype(np.float64)
        mu = np.mean(reaches)
        if mu < 1.0:
            continue
        cv = np.std(reaches) / mu
        cvs.append(cv)

    if len(cvs) < 10:
        return 0.0

    return float(np.median(cvs))


# ═══════════════════════════════════════════════════════════════════════
# OBSERVABLE SI-6: EFFECTIVE RESISTANCE CV
# ═══════════════════════════════════════════════════════════════════════
#
# MOTIVATION: Effective resistance between linked pairs measures the
# CONNECTIVITY GEOMETRY of the graph — how many independent paths
# connect two nodes. In flat space, this is roughly uniform. Curvature
# creates regions of sparse connectivity (high resistance) and dense
# connectivity (low resistance).
#
# FORMULA:
#   L = D - A (Laplacian). L⁺ = pseudoinverse.
#   For each link (i,j): R_ij = L⁺_{ii} + L⁺_{jj} - 2 L⁺_{ij}.
#   Observable = CV(R_ij) = Std(R) / Mean(R) over all links.
#
# EXPECTED:
#   Flat: Low CV (uniform connectivity)
#   Schwarzschild: High CV (connectivity varies radially)
#   pp-wave: Moderate CV
#
# CONFIDENCE: MEDIUM
# REFERENCE: Klein & Randic (1993); Ellens et al. (2011)
# ═══════════════════════════════════════════════════════════════════════

def compute_eff_resistance_cv(A_link, degrees):
    """CV of per-link effective resistance on the link graph."""
    N = A_link.shape[0]
    A_dense = A_link.toarray() if sp.issparse(A_link) else A_link
    L = np.diag(degrees.astype(np.float64)) - A_dense

    # Pseudoinverse
    try:
        L_pinv = np.linalg.pinv(L)
    except Exception:
        return 0.0

    # Effective resistance for each link
    rows, cols = sp.triu(sp.csr_matrix(A_link), k=1).nonzero()
    if len(rows) < 10:
        return 0.0

    R_eff = (L_pinv[rows, rows] + L_pinv[cols, cols]
             - L_pinv[rows, cols] - L_pinv[cols, rows])

    R_eff = np.abs(R_eff)  # numerical noise protection
    mu = np.mean(R_eff)
    if mu < 1e-15:
        return 0.0

    return float(np.std(R_eff) / mu)


# ═══════════════════════════════════════════════════════════════════════
# MASTER COMPUTE FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def compute_seed_inversion_observables(C, A_link, degrees, pts, C2, rng):
    """Compute all 6 seed-inversion observables.

    Parameters
    ----------
    C : ndarray or sparse, shape (N,N) — causal matrix
    A_link : sparse, shape (N,N) — symmetric link graph adjacency
    degrees : ndarray, shape (N,) — link degrees
    pts : ndarray, shape (N,4) — spacetime coordinates
    C2 : ndarray, shape (N,N) — 2-chain matrix (C @ C)
    rng : np.random.Generator — for diamond sampling

    Returns
    -------
    dict with keys: fan_moran_I, diamond_aspect_cv, level_spacing_ratio,
                    pf_residual_var, chain_direction_cv, eff_resistance_cv
    """
    N = C.shape[0]
    C_dense = C if isinstance(C, np.ndarray) else C.toarray()
    C_sp = sp.csr_matrix(C)

    # Build directed link graph and heights
    C2_bool_sp = sp.csr_matrix(C2 > 0, dtype=np.float64)
    C_bool = (C_sp > 0).astype(np.float64)
    L_dir = C_bool - C_bool.multiply(C2_bool_sp)
    L_dir.eliminate_zeros()
    L_dir_csc = L_dir.tocsc()
    L_dir_csr = L_dir.tocsr()

    heights = np.zeros(N, dtype=int)
    for j in range(N):
        parents = L_dir_csc.getcol(j).indices
        if len(parents) > 0:
            heights[j] = np.max(heights[parents]) + 1

    obs = {}

    # SI-1: Fan Moran's I
    t0 = time.perf_counter()
    obs['fan_moran_I'] = compute_fan_moran_I(
        C_sp, L_dir_csc, L_dir_csr, heights, A_link)
    obs['_time_moran'] = time.perf_counter() - t0

    # SI-2: Diamond aspect ratio CV
    t0 = time.perf_counter()
    obs['diamond_aspect_cv'] = compute_diamond_aspect_cv(
        C_dense, C2, rng, n_samples=150)
    obs['_time_diamond'] = time.perf_counter() - t0

    # SI-3: Level spacing ratio
    t0 = time.perf_counter()
    obs['level_spacing_ratio'] = compute_level_spacing_ratio(A_link, degrees)
    obs['_time_lsr'] = time.perf_counter() - t0

    # SI-4: Past-future residual variance
    t0 = time.perf_counter()
    obs['pf_residual_var'] = compute_pf_residual_var(C_sp)
    obs['_time_pfresid'] = time.perf_counter() - t0

    # SI-5: Chain direction CV
    t0 = time.perf_counter()
    obs['chain_direction_cv'] = compute_chain_direction_cv(
        C_dense, L_dir_csr, heights)
    obs['_time_chaindir'] = time.perf_counter() - t0

    # SI-6: Effective resistance CV
    t0 = time.perf_counter()
    obs['eff_resistance_cv'] = compute_eff_resistance_cv(A_link, degrees)
    obs['_time_effres'] = time.perf_counter() - t0

    del L_dir, L_dir_csc, L_dir_csr, C2_bool_sp
    gc.collect()
    return obs


# ═══════════════════════════════════════════════════════════════════════
# STANDALONE TEST (sanity check at N=200)
# ═══════════════════════════════════════════════════════════════════════

def _test_standalone():
    """Quick sanity test: compute all observables on a tiny flat causal set."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from scripts.discovery_common import (
        sprinkle_4d, causal_flat, build_link_graph, graph_statistics
    )

    N, T = 200, 1.0
    rng = np.random.default_rng(42)
    pts = sprinkle_4d(N, T, rng)
    C = causal_flat(pts)
    A_link = build_link_graph(C)
    _, degrees = graph_statistics(A_link)
    C_sp = sp.csr_matrix(C)
    C2 = (C_sp @ C_sp).toarray()

    print(f"Sanity test: N={N}, flat Minkowski")
    print(f"  TC = {np.sum(C):.0f}, links = {int(np.sum(A_link))//2}")

    obs = compute_seed_inversion_observables(C, A_link, degrees, pts, C2, rng)

    print(f"\n  Seed-inversion observables:")
    for k, v in sorted(obs.items()):
        if not k.startswith('_'):
            print(f"    {k:25s} = {v:.6f}")
    print(f"\n  Timings:")
    for k, v in sorted(obs.items()):
        if k.startswith('_time'):
            print(f"    {k:25s} = {v:.3f}s")

    total = sum(v for k, v in obs.items() if k.startswith('_time'))
    print(f"    {'TOTAL':25s} = {total:.3f}s")

    # Basic sanity checks
    assert -1 <= obs['fan_moran_I'] <= 1, f"Moran I out of range: {obs['fan_moran_I']}"
    assert 0 <= obs['level_spacing_ratio'] <= 1, f"LSR out of range: {obs['level_spacing_ratio']}"
    assert obs['pf_residual_var'] >= 0, f"pf_resid negative: {obs['pf_residual_var']}"
    assert obs['chain_direction_cv'] >= 0, f"chain_dir negative: {obs['chain_direction_cv']}"
    assert obs['eff_resistance_cv'] >= 0, f"eff_res negative: {obs['eff_resistance_cv']}"
    print("\n  All sanity checks PASSED.")


if __name__ == '__main__':
    _test_standalone()
