#!/usr/bin/env python3
"""
COMP-SCAN v3: blind scan of ~50 observables on causal sets.
N=500, M=20 CRN pairs, pp-wave quadrupole eps=5.

v3 additions: interval abundances, degree skewness/kurtosis, triangle count,
clustering coefficient, nuclear norm residual, directed entropy asymmetry,
longest chain, trace(C²)/rank(C²).

Positive control: rank_C2 (expected d ~ -2.0 from Run #001).
Negative control: conformal (expected d ~ 0).

Output: docs/analysis_runs/run_20260325_133158/comp_scan_results.json
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from numpy.linalg import svd, matrix_rank, norm, eigh
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.stats import wilcoxon, ttest_rel, entropy as sp_entropy, skew, kurtosis
from scipy.special import gammaln  # for stable Poisson log-probs

# ── Parameters ──────────────────────────────────────────────────────
N = 500
M = 20
T_DIAMOND = 1.0
EPS_PPWAVE = 5.0
EPS_CONFORMAL = 1.0  # for null test
MASTER_SEED = 42424
HEAT_T_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0]
K_EIGS = min(N - 2, 80)  # (unused — now using dense eigh for full spectrum)

RUN_DIR = Path(__file__).resolve().parents[2] / "docs" / "analysis_runs" / "run_20260325_133158"
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
    """Flat Minkowski causal matrix."""
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        dt = pts[i + 1:, 0] - pts[i, 0]  # always > 0 (sorted by t)
        dr2 = np.sum((pts[i + 1:, 1:] - pts[i, 1:])**2, axis=1)
        causal = dt**2 > dr2
        C[i, i + 1:] = causal.astype(np.int8)
    return C


def causal_ppwave_quad(pts: np.ndarray, eps: float) -> np.ndarray:
    """PP-wave quadrupole: ds^2 = -du dv + (1 + eps*(x^2-y^2))(dx^2+dy^2)."""
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        dt = pts[i + 1:, 0] - pts[i, 0]
        dx = pts[i + 1:, 1:] - pts[i, 1:]
        dr2 = np.sum(dx**2, axis=1)
        # Midpoint for metric evaluation
        xm = (pts[i + 1:, 1] + pts[i, 1]) / 2
        ym = (pts[i + 1:, 2] + pts[i, 2]) / 2
        dz = dx[:, 2]
        du = dt + dz
        f = xm**2 - ym**2
        interval = dt**2 - dr2 - eps * f * du**2 / 2
        causal = interval > 0
        C[i, i + 1:] = causal.astype(np.int8)
    return C


def causal_conformal(pts: np.ndarray, eps: float) -> np.ndarray:
    """Conformal rescaling — must give IDENTICAL causal matrix as flat."""
    return causal_flat(pts)


# ── Link graph ──────────────────────────────────────────────────────
def build_link_graph(C: np.ndarray):
    """Build undirected link graph adjacency (sparse CSR).
    Link: i≺j AND no k with i≺k≺j, i.e. C[i,j]=1 AND (C²)[i,j]=0.
    """
    C_sp = sparse.csr_matrix(C.astype(np.float32))
    C2 = C_sp @ C_sp
    # Link = causal AND zero intervening
    link_mask = (C > 0) & (C2.toarray() == 0)
    A_link = link_mask.astype(np.float32)
    A_sym = A_link + A_link.T
    return sparse.csr_matrix(A_sym)


def build_laplacian(A_sp):
    """Combinatorial graph Laplacian L = D - A."""
    degrees = np.array(A_sp.sum(axis=1)).flatten()
    D = sparse.diags(degrees)
    return D - A_sp


# ── Helper functions ────────────────────────────────────────────────
def gini_coefficient(values: np.ndarray) -> float:
    """Gini coefficient of a distribution."""
    v = np.sort(np.abs(values))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * v)) / (n * np.sum(v)) - (n + 1) / n


def shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a discrete distribution (from counts)."""
    p = counts / counts.sum()
    p = p[p > 0]
    return -np.sum(p * np.log(p))


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Paired Cohen's d: mean(x-y) / std(x-y)."""
    diff = x - y
    s = diff.std(ddof=1)
    if s == 0:
        return 0.0
    return diff.mean() / s


# ── Observable computation ──────────────────────────────────────────
def compute_all_observables(C: np.ndarray, pts: np.ndarray) -> dict:
    """Compute all ~36 observables from causal matrix C and point coords."""
    n = len(C)
    obs = {}

    # ── A. Counting baselines ──
    tc = int(C.sum())
    obs["tc"] = tc

    C2 = C.astype(np.float64) @ C.astype(np.float64)
    n2 = int(C2.sum())
    obs["n2"] = n2

    C3 = C2 @ C.astype(np.float64)
    n3 = int(C3.sum())
    obs["n3"] = n3

    C4 = C3 @ C.astype(np.float64)
    n4 = int(C4.sum())
    obs["n4"] = n4

    obs["chain_ratio_32"] = n3 / max(n2, 1)
    obs["chain_ratio_43"] = n4 / max(n3, 1)

    # ── B. Link graph ──
    A_link = build_link_graph(C)
    degrees = np.array(A_link.sum(axis=1)).flatten()
    link_count = int(degrees.sum()) // 2
    obs["link_count"] = link_count
    obs["mean_degree"] = degrees.mean()
    obs["degree_cv"] = degrees.std() / max(degrees.mean(), 1e-10)
    obs["degree_gini"] = gini_coefficient(degrees)
    obs["degree_entropy"] = shannon_entropy(np.bincount(degrees.astype(int)))

    # KL divergence from Poisson (stable computation using log-space)
    mean_deg = degrees.mean()
    deg_counts = np.bincount(degrees.astype(int))
    deg_probs = deg_counts / deg_counts.sum()
    max_k = len(deg_counts)
    # Use gammaln to avoid factorial overflow: log(Poisson(k)) = -mu + k*log(mu) - gammaln(k+1)
    ks = np.arange(max_k, dtype=np.float64)
    log_poisson = -mean_deg + ks * np.log(max(mean_deg, 1e-30)) - gammaln(ks + 1)
    poisson_probs = np.exp(log_poisson - log_poisson.max())  # numerically stable
    poisson_probs = poisson_probs / poisson_probs.sum()
    obs["degree_kl_poisson"] = float(sp_entropy(deg_probs + 1e-12, poisson_probs + 1e-12))

    # ── C. Directed asymmetry (FPLAV) ──
    # Future links: i→j where i≺j (lower to higher row index, original C direction)
    # Past links: j→i where j≺i
    # For each element i: d_i^+ = future link count, d_i^- = past link count
    link_mask = (C > 0) & (C2 == 0)  # C2 was computed as float64
    d_plus = link_mask.sum(axis=1)   # future links from i (row sums)
    d_minus = link_mask.sum(axis=0)  # past links to i (column sums -> transposes)
    d_total = d_plus + d_minus
    # Exclude boundary (elements with d_total < 3)
    interior = d_total >= 3
    if interior.sum() > 10:
        beta = d_plus[interior] / d_total[interior]
        obs["fplav"] = float(np.var(beta))
        obs["fplav_mean"] = float(np.mean(beta))
    else:
        obs["fplav"] = 0.0
        obs["fplav_mean"] = 0.5

    # ── D. Matrix-algebraic (from C and C²) ──
    rank_C = int(matrix_rank(C))
    obs["rank_C"] = rank_C

    rank_C2 = int(matrix_rank(C2))
    obs["rank_C2"] = rank_C2

    rank_C3 = int(matrix_rank(C3))
    obs["rank_C3"] = rank_C3

    obs["rank_ratio_21"] = rank_C2 / max(rank_C, 1)
    obs["rank_ratio_32"] = rank_C3 / max(rank_C2, 1)

    # Frobenius norms
    frob_C = float(norm(C, 'fro'))
    frob_C2 = float(norm(C2, 'fro'))
    obs["frobenius_C"] = frob_C
    obs["frobenius_C2"] = frob_C2
    obs["frobenius_ratio"] = frob_C2 / max(frob_C**2, 1e-10)

    # SVD of C² (truncated for speed)
    k_svd = min(rank_C2 + 5, n - 1)
    try:
        sigmas = svd(C2, compute_uv=False)
        sigmas_nz = sigmas[sigmas > 1e-10]
    except Exception:
        sigmas_nz = np.array([1.0])

    # Nuclear norm of C
    try:
        sigmas_C = svd(C.astype(np.float64), compute_uv=False)
        obs["nuclear_norm"] = float(sigmas_C.sum())
    except Exception:
        obs["nuclear_norm"] = 0.0

    # Stable rank of C²
    if len(sigmas_nz) > 0 and sigmas_nz[0] > 0:
        obs["stable_rank_C2"] = float(np.sum(sigmas_nz**2)) / float(sigmas_nz[0]**2)
    else:
        obs["stable_rank_C2"] = 0.0

    # SVD participation ratio
    if len(sigmas_nz) > 0:
        obs["svd_participation_ratio"] = float(np.sum(sigmas_nz))**2 / (n * float(np.sum(sigmas_nz**2)))
    else:
        obs["svd_participation_ratio"] = 0.0

    # SV Gini coefficient
    obs["sv_gini"] = gini_coefficient(sigmas_nz)

    # SV entropy (normalized)
    if len(sigmas_nz) > 1:
        sv_probs = sigmas_nz / sigmas_nz.sum()
        obs["sv_entropy"] = float(-np.sum(sv_probs * np.log(sv_probs))) / np.log(len(sigmas_nz))
    else:
        obs["sv_entropy"] = 0.0

    # ISFR = ||C²||_F² / (sum C²)²
    obs["isfr"] = float(np.sum(C2**2)) / max(float(n2)**2, 1)

    # Column-norm Gini of C²
    col_norms_C2 = np.sqrt(np.sum(C2**2, axis=0))
    obs["column_gini_C2"] = gini_coefficient(col_norms_C2)

    # C² column sum entropy
    col_sums_C2 = C2.sum(axis=0)
    col_sums_C2 = np.asarray(col_sums_C2).flatten()
    col_sums_pos = col_sums_C2[col_sums_C2 > 0]
    if len(col_sums_pos) > 1:
        p_col = col_sums_pos / col_sums_pos.sum()
        obs["ccne"] = float(-np.sum(p_col * np.log(p_col))) / np.log(len(col_sums_pos))
    else:
        obs["ccne"] = 0.0

    # Jordan Grade Entropy: H({rank(C^{k-1}) - rank(C^k)}) for k=1..5
    ranks = [n, rank_C, rank_C2, rank_C3]
    # Compute rank(C⁴) — C4 already computed
    rank_C4 = int(matrix_rank(C4))
    ranks.append(rank_C4)
    mu = [ranks[k] - ranks[k + 1] for k in range(len(ranks) - 1)]
    mu = np.array([m for m in mu if m > 0], dtype=np.float64)
    if len(mu) > 1:
        mu_p = mu / mu.sum()
        obs["jordan_grade_entropy"] = float(-np.sum(mu_p * np.log(mu_p)))
    else:
        obs["jordan_grade_entropy"] = 0.0

    # ── E. Link-graph spectral (dense eigh for N=300 — all eigenvalues) ──
    L = build_laplacian(A_link)
    L_dense = L.toarray().astype(np.float64)
    all_eigs = np.linalg.eigh(L_dense)[0]  # sorted ascending
    lambdas = all_eigs[all_eigs > 1e-10]  # remove zero eigenvalue(s)

    if len(lambdas) >= 2:
        obs["fiedler"] = float(lambdas[0])
        obs["lambda_3"] = float(lambdas[1])
        obs["lambda_ratio_23"] = float(lambdas[0]) / float(lambdas[1])
    else:
        obs["fiedler"] = float(lambdas[0]) if len(lambdas) > 0 else 0.0
        obs["lambda_3"] = 0.0
        obs["lambda_ratio_23"] = 0.0

    if len(lambdas) > 0:
        lam_max = lambdas[-1]
        obs["lambda_ratio_2max"] = float(lambdas[0]) / max(float(lam_max), 1e-10)
    else:
        obs["lambda_ratio_2max"] = 0.0

    # Spectral gap
    if len(lambdas) >= 2:
        obs["spectral_gap_23"] = float(lambdas[1] - lambdas[0])
    else:
        obs["spectral_gap_23"] = 0.0

    # Spectral entropy of Laplacian
    if len(lambdas) > 1:
        lam_p = lambdas / lambdas.sum()
        obs["spectral_entropy"] = float(-np.sum(lam_p * np.log(lam_p + 1e-30)))
    else:
        obs["spectral_entropy"] = 0.0

    # Heat traces at specific t values
    for t_val in HEAT_T_VALUES:
        ht = float(np.sum(np.exp(-lambdas * t_val)))
        obs[f"heat_trace_{t_val}"] = ht

    # ── F. Graph topology ──
    # 4-cycle density: Tr(A⁴) computation
    A_dense = A_link.toarray().astype(np.float64)
    A2 = A_dense @ A_dense
    trA4 = float(np.trace(A2 @ A2))
    # For triangle-free graph: 8*c4 = Tr(A⁴) - 2m - 2*Σ d_i(d_i-1)
    # where m = number of undirected edges
    sum_d_d_minus_1 = float(np.sum(degrees * (degrees - 1)))
    four_cycle_raw = trA4 - 2 * link_count - 2 * sum_d_d_minus_1
    obs["four_cycle_count"] = max(four_cycle_raw / 8, 0)
    # Normalize by edge pairs
    obs["four_cycle_density"] = obs["four_cycle_count"] / max(link_count * (link_count - 1) / 2, 1)

    # Number of connected components
    n_components = sparse.csgraph.connected_components(A_link, directed=False)[0]
    obs["n_components"] = n_components

    # Triangle count: Tr(A³)/6
    trA3 = float(np.trace(A_dense @ A2))
    obs["triangle_count"] = max(trA3 / 6, 0)
    obs["triangle_density"] = obs["triangle_count"] / max(link_count, 1)

    # Clustering coefficient (global)
    # C_global = 3 * triangles / (number of connected triples)
    n_triples = float(np.sum(degrees * (degrees - 1))) / 2
    obs["clustering_coeff"] = 3 * obs["triangle_count"] / max(n_triples, 1)

    # ── G. Interval abundances (from C and C²) ──
    # I_k = #{pairs (i,j): i≺j AND |interval(i,j)| = k internal elements}
    # C²[i,j] gives the number of intermediate elements between i and j
    C2_int = C2.astype(int)
    C_bool = C > 0
    # I_0 = links (no intervening element)
    I_0 = int(np.sum(C_bool & (C2_int == 0)))  # = link_count
    # I_1 = relations with exactly 1 intervening element
    I_1 = int(np.sum(C_bool & (C2_int == 1)))
    # I_2 = relations with exactly 2 intervening elements
    I_2 = int(np.sum(C_bool & (C2_int == 2)))
    # I_3+
    I_3plus = int(np.sum(C_bool & (C2_int >= 3)))
    obs["interval_I0"] = I_0
    obs["interval_I1"] = I_1
    obs["interval_I2"] = I_2
    obs["interval_I3plus"] = I_3plus
    # Ratios (NOT obs/TC — these are interval-to-interval ratios)
    obs["interval_ratio_10"] = I_1 / max(I_0, 1)
    obs["interval_ratio_21"] = I_2 / max(I_1, 1)
    # Interval entropy
    interval_counts = np.array([I_0, I_1, I_2, I_3plus], dtype=np.float64)
    interval_counts = interval_counts[interval_counts > 0]
    if len(interval_counts) > 1:
        ip = interval_counts / interval_counts.sum()
        obs["interval_entropy"] = float(-np.sum(ip * np.log(ip)))
    else:
        obs["interval_entropy"] = 0.0

    # ── H. Degree distribution higher moments ──
    if len(degrees) > 3 and degrees.std() > 0:
        obs["degree_skewness"] = float(skew(degrees))
        obs["degree_kurtosis"] = float(kurtosis(degrees))
    else:
        obs["degree_skewness"] = 0.0
        obs["degree_kurtosis"] = 0.0

    # ── I. Directed link graph entropy asymmetry ──
    # In-degree and out-degree of directed links
    link_mask_dir = (C > 0) & (C2 == 0)
    d_out = np.asarray(link_mask_dir.sum(axis=1)).flatten()  # future links
    d_in = np.asarray(link_mask_dir.sum(axis=0)).flatten()   # past links
    d_out_counts = np.bincount(d_out.astype(int))
    d_in_counts = np.bincount(d_in.astype(int))
    H_out = shannon_entropy(d_out_counts[d_out_counts > 0])
    H_in = shannon_entropy(d_in_counts[d_in_counts > 0])
    obs["H_out_degree"] = H_out
    obs["H_in_degree"] = H_in
    obs["H_asymmetry"] = H_out - H_in

    # ── J. Nuclear norm residual ──
    # KB fact: ||C||_* ≈ 6.6 × √TC. Residual = signal?
    obs["nuclear_norm_residual"] = obs["nuclear_norm"] - 6.6 * math.sqrt(max(tc, 0))

    # ── K. Average SV squared of C² ──
    if rank_C2 > 0 and len(sigmas_nz) > 0:
        obs["mean_sv_sq_C2"] = float(np.mean(sigmas_nz**2))
        obs["trace_C2_over_rank"] = float(np.sum(sigmas_nz**2)) / rank_C2
    else:
        obs["mean_sv_sq_C2"] = 0.0
        obs["trace_C2_over_rank"] = 0.0

    # ── L. Longest chain (height of poset) ──
    # Compute via longest path in DAG using dynamic programming
    # C is upper-triangular → topological order = row order
    chain_len = np.zeros(n, dtype=int)
    for i in range(n - 1, -1, -1):
        successors = np.where(C[i, :] > 0)[0]
        if len(successors) > 0:
            chain_len[i] = 1 + chain_len[successors].max()
    obs["longest_chain"] = int(chain_len.max())

    # ── M. Spectral dimension from link-graph Laplacian ──
    # d_S(t) = -2 * d(log K)/d(log t) at t=1.0
    if len(lambdas) > 0:
        t_a, t_b = 0.9, 1.1
        K_a = float(np.sum(np.exp(-lambdas * t_a)))
        K_b = float(np.sum(np.exp(-lambdas * t_b)))
        if K_a > 0 and K_b > 0:
            obs["spectral_dim_t1"] = -2 * (math.log(K_b) - math.log(K_a)) / (math.log(t_b) - math.log(t_a))
        else:
            obs["spectral_dim_t1"] = 0.0
    else:
        obs["spectral_dim_t1"] = 0.0

    return obs


# ── CRN main loop ───────────────────────────────────────────────────
def run_comp_scan():
    """Run COMP-SCAN with CRN: flat vs pp-wave on same points."""
    print(f"COMP-SCAN: N={N}, M={M}, eps={EPS_PPWAVE}, K_EIGS={K_EIGS}")
    print(f"Output: {RESULTS_FILE}")

    ss = np.random.SeedSequence(MASTER_SEED)
    seeds = ss.spawn(M)

    all_results = []
    obs_names = None
    t_start = time.time()

    for trial in range(M):
        rng = np.random.default_rng(seeds[trial])
        pts = sprinkle(N, T_DIAMOND, rng)

        # CRN: same points, different metrics
        C_flat = causal_flat(pts)
        C_curved = causal_ppwave_quad(pts, EPS_PPWAVE)

        obs_flat = compute_all_observables(C_flat, pts)
        obs_curved = compute_all_observables(C_curved, pts)

        if obs_names is None:
            obs_names = sorted(obs_flat.keys())

        record = {"trial": trial}
        for name in obs_names:
            record[f"{name}_flat"] = obs_flat[name]
            record[f"{name}_curved"] = obs_curved[name]
            record[f"{name}_delta"] = obs_curved[name] - obs_flat[name]

        all_results.append(record)

        elapsed = time.time() - t_start
        eta = elapsed / (trial + 1) * (M - trial - 1)
        print(f"  Trial {trial + 1}/{M} done ({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)")

    # ── Analysis ────────────────────────────────────────────────────
    print("\n=== COMP-SCAN RESULTS ===")
    print(f"{'Observable':<30} {'Cohen d':>10} {'Wilcoxon p':>12} {'Mean delta':>12} {'VERDICT':>10}")
    print("-" * 80)

    analysis = {}
    for name in obs_names:
        flat_vals = np.array([r[f"{name}_flat"] for r in all_results])
        curved_vals = np.array([r[f"{name}_curved"] for r in all_results])
        deltas = curved_vals - flat_vals

        d = cohen_d(curved_vals, flat_vals)
        try:
            _, p_wilcox = wilcoxon(deltas, alternative='two-sided')
        except Exception:
            p_wilcox = 1.0
        try:
            _, p_ttest = ttest_rel(curved_vals, flat_vals)
        except Exception:
            p_ttest = 1.0

        # Bonferroni correction: α = 0.05 / n_tests
        n_tests = len(obs_names)
        alpha_bonf = 0.05 / n_tests
        verdict = "PASS" if abs(d) > 0.5 and p_wilcox < alpha_bonf else "FAIL"

        analysis[name] = {
            "cohen_d": round(d, 3),
            "p_wilcoxon": float(p_wilcox),
            "p_ttest": float(p_ttest),
            "mean_delta": float(deltas.mean()),
            "std_delta": float(deltas.std()),
            "mean_flat": float(flat_vals.mean()),
            "mean_curved": float(curved_vals.mean()),
            "verdict": verdict,
        }

        star = "***" if p_wilcox < 0.001 else ("**" if p_wilcox < 0.01 else ("*" if p_wilcox < 0.05 else ""))
        print(f"  {name:<28} {d:>10.3f} {p_wilcox:>12.2e} {deltas.mean():>12.4f}  {verdict} {star}")

    # ── Baseline R² (simple: each obs vs TC delta) ──────────────────
    print("\n=== BASELINE R² (delta_obs vs delta_TC) ===")
    tc_deltas = np.array([r["tc_delta"] for r in all_results])
    tc_var = np.var(tc_deltas)

    for name in obs_names:
        if name == "tc":
            continue
        deltas = np.array([r[f"{name}_delta"] for r in all_results])
        if tc_var > 0 and np.var(deltas) > 0:
            r_pearson = np.corrcoef(tc_deltas, deltas)[0, 1]
            r2 = r_pearson**2
        else:
            r2 = 0.0
        analysis[name]["r2_vs_tc"] = round(r2, 3)

        if abs(analysis[name]["cohen_d"]) > 0.5:
            color = "GREEN" if r2 < 0.3 else ("YELLOW" if r2 < 0.5 else "RED")
            print(f"  {name:<28} R²={r2:.3f}  [{color}]")

    # ── Multi-baseline R² (link_count, degree_cv, tc, n2) ───────────
    print("\n=== MULTI-BASELINE R² (4 baselines) ===")
    baselines = {}
    for bname in ["tc", "link_count", "degree_cv", "n2"]:
        baselines[bname] = np.array([r[f"{bname}_delta"] for r in all_results])

    X_base = np.column_stack(list(baselines.values()))
    X_base_c = X_base - X_base.mean(axis=0)

    for name in obs_names:
        if name in baselines or abs(analysis[name].get("cohen_d", 0)) < 0.5:
            continue
        y = np.array([r[f"{name}_delta"] for r in all_results])
        y_c = y - y.mean()
        # OLS R²
        try:
            beta, res, rank_ols, _ = np.linalg.lstsq(X_base_c, y_c, rcond=None)
            ss_res = float(np.sum((y_c - X_base_c @ beta)**2))
            ss_tot = float(np.sum(y_c**2))
            r2_multi = 1 - ss_res / max(ss_tot, 1e-30)
            r2_multi = max(r2_multi, 0.0)
        except Exception:
            r2_multi = 0.0

        analysis[name]["r2_multi_baseline"] = round(r2_multi, 3)
        color = "GREEN" if r2_multi < 0.3 else ("YELLOW" if r2_multi < 0.5 else "RED")
        print(f"  {name:<28} R²={r2_multi:.3f}  [{color}]")

    # ── Save results ────────────────────────────────────────────────
    output = {
        "parameters": {"N": N, "M": M, "eps": EPS_PPWAVE, "T": T_DIAMOND,
                       "seed": MASTER_SEED, "metric": "ppwave_quad"},
        "analysis": analysis,
        "raw": all_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_FILE}")

    # ── Summary: top candidates ─────────────────────────────────────
    print("\n=== TOP CANDIDATES (|d|>0.5, R²_multi<0.5, p<0.05) ===")
    candidates = []
    for name, a in analysis.items():
        if (abs(a.get("cohen_d", 0)) > 0.5
                and a.get("r2_multi_baseline", a.get("r2_vs_tc", 1.0)) < 0.5
                and a.get("p_wilcoxon", 1.0) < 0.05):
            candidates.append((name, a))
    candidates.sort(key=lambda x: abs(x[1]["cohen_d"]), reverse=True)
    for name, a in candidates:
        r2 = a.get("r2_multi_baseline", a.get("r2_vs_tc", -1))
        print(f"  {name:<28} d={a['cohen_d']:+.3f}  R²={r2:.3f}  p={a['p_wilcoxon']:.2e}")

    if not candidates:
        print("  (none)")

    return output


if __name__ == "__main__":
    result = run_comp_scan()
