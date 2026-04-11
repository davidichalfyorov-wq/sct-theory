#!/usr/bin/env python3
"""
SEED INVERSION (PATH): Observable Candidates from
path_kurtosis TC-independence mechanism.

INSIGHT: path_kurtosis (kurtosis of log-path-counts through Hasse DAG) has
TC-independent signal because:
  1. It is a PER-ELEMENT statistic (each x has its own P(x))
  2. Kurtosis is a SHAPE statistic (location/scale invariant)
  3. TC shifts the mean of log P(x) but not the shape

chain_decay_slope FAILS because it measures GLOBAL chain counts N_k,
where N_2 = TC directly contaminates the slope.

MECHANISM: Per-element path statistics capture LOCAL causal geometry.
Higher-order shape statistics (kurtosis, entropy, normalized ratios)
are invariant under the TC-induced shift/scale transformation.

CANDIDATES (4):
  SP-1: evenness_kurtosis    — branching entropy evenness per element
  SP-2: path_asymmetry_kurt  — past-future path imbalance (height-normalized)
  SP-3: path_excess_skew     — relative path standing at same height
  SP-4: path_divergence_cv   — CV of path-spread among future children

N=500, M=30, CRN protocol. pp-wave eps=5, Schwarzschild eps=0.005.
Conformal null control.

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
import json, time, sys, os, gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.discovery_common import (
    sprinkle_4d, causal_flat, causal_ppwave_quad, causal_schwarzschild,
    causal_conformal, build_link_graph, graph_statistics
)

# ═══════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════
N = 500
M = 30
T_DIAMOND = 1.0
MASTER_SEED = 808_008  # unique for this SEED-INVERSION-PATH run
METRICS = {
    "ppwave_quad":   (causal_ppwave_quad,   5.0),
    "schwarzschild": (causal_schwarzschild, 0.005),
    "conformal":     (causal_conformal,     0.0),   # NULL control
}
SEED_OFFSETS = {"ppwave_quad": 100, "schwarzschild": 300, "conformal": 500}

# Certified controls (must detect signal)
CTRL_KEYS = ['column_gini_C2', 'path_kurtosis']
# New candidates
CAND_KEYS = ['evenness_kurtosis', 'path_asymmetry_kurt',
             'path_excess_skew', 'path_divergence_cv']
# Proxy fields for decontamination
PROXY_KEYS = ['tc', 'mean_degree', 'degree_var', 'degree_skew',
              'degree_kurt', 'max_degree']


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n < 2 or np.sum(np.abs(x)) == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(2.0 * np.sum(idx * x) / (n * np.sum(x)) - (n + 1) / n)


# ═══════════════════════════════════════════════════════════════════════
# Path infrastructure: p_down, p_up via DP on Hasse DAG
# ═══════════════════════════════════════════════════════════════════════
def compute_path_arrays(L_dir_csr, L_dir_csc, N_pts):
    """Compute per-element path counts via DP on the Hasse DAG.

    Returns:
        p_down: array of shape (N_pts,), number of directed paths from
                minimal elements to each element
        p_up:   array of shape (N_pts,), number of directed paths from
                each element to maximal elements
        heights: array of shape (N_pts,), longest-chain height of each element
    """
    # p_down: number of maximal paths from past boundary to x
    p_down = np.ones(N_pts, dtype=np.float64)
    heights = np.zeros(N_pts, dtype=np.int32)
    for j in range(N_pts):
        parents = L_dir_csc.getcol(j).indices
        if len(parents) > 0:
            p_down[j] = np.sum(p_down[parents])
            heights[j] = np.max(heights[parents]) + 1
        if p_down[j] == 0:
            p_down[j] = 1.0

    # p_up: number of maximal paths from x to future boundary
    p_up = np.ones(N_pts, dtype=np.float64)
    for i in range(N_pts - 1, -1, -1):
        children = L_dir_csr.getrow(i).indices
        if len(children) > 0:
            p_up[i] = np.sum(p_up[children])
        if p_up[i] == 0:
            p_up[i] = 1.0

    return p_down, p_up, heights


# ═══════════════════════════════════════════════════════════════════════
# Observable SP-1: evenness_kurtosis
# ═══════════════════════════════════════════════════════════════════════
# For each element x with k_out >= 2, compute the branching entropy:
#   H(x) = -sum_{child y} [p_up(y)/p_up(x)] * log2[p_up(y)/p_up(x)]
# Normalize by max entropy: E(x) = H(x) / log2(k_out(x))
# E(x) in [0,1], measures how evenly paths branch through x.
# TC-immunity: E(x) is a ratio of entropies, independent of scale.
# Then: evenness_kurtosis = Kurt({E(x) : bulk, k_out >= 2})
def compute_evenness_kurtosis(L_dir_csr, p_up, N_pts, bulk_mask):
    """Kurtosis of per-element path branching evenness."""
    evenness = []
    for i in range(N_pts):
        if not bulk_mask[i]:
            continue
        children = L_dir_csr.getrow(i).indices
        k_out = len(children)
        if k_out < 2:
            continue
        # Probability of each child branch
        child_p = p_up[children]
        total = np.sum(child_p)
        if total <= 0:
            continue
        probs = child_p / total
        # Entropy
        mask_nz = probs > 0
        H = -np.sum(probs[mask_nz] * np.log2(probs[mask_nz]))
        H_max = np.log2(k_out)
        if H_max > 0:
            evenness.append(H / H_max)

    if len(evenness) < 10:
        return 0.0
    return float(stats.kurtosis(evenness, fisher=True))


# ═══════════════════════════════════════════════════════════════════════
# Observable SP-2: path_asymmetry_kurt
# ═══════════════════════════════════════════════════════════════════════
# For each element x, A(x) = log2(p_down(x)) - log2(p_up(x)).
# Height-normalize: A_norm(x) = A(x) - median(A at same height).
# TC-immunity: A(x) is a difference of logs, TC shift cancels.
# Height normalization removes trivial position dependence.
# Then: path_asymmetry_kurt = Kurt({A_norm(x) : bulk})
def compute_path_asymmetry_kurt(p_down, p_up, heights, bulk_mask, N_pts):
    """Kurtosis of height-normalized past-future path asymmetry."""
    log_down = np.log2(p_down + 1)
    log_up = np.log2(p_up + 1)
    A = log_down - log_up

    # Height-normalize: subtract median per height layer
    unique_heights = np.unique(heights[bulk_mask])
    A_norm = np.full(N_pts, np.nan)
    for h in unique_heights:
        h_mask = bulk_mask & (heights == h)
        if np.sum(h_mask) < 3:
            continue
        med = np.median(A[h_mask])
        A_norm[h_mask] = A[h_mask] - med

    vals = A_norm[~np.isnan(A_norm)]
    if len(vals) < 10:
        return 0.0
    return float(stats.kurtosis(vals, fisher=True))


# ═══════════════════════════════════════════════════════════════════════
# Observable SP-3: path_excess_skew
# ═══════════════════════════════════════════════════════════════════════
# For each element x at height h, define:
#   R(x) = [p_down(x) * p_up(x)] / [mean(p_down at h) * mean(p_up at h)]
# R(x) measures excess path load relative to height-peers.
# TC-immunity: R(x) is a double ratio; TC cancels in num and denom.
# Then: path_excess_skew = Skew({log R(x) : bulk})
def compute_path_excess_skew(p_down, p_up, heights, bulk_mask, N_pts):
    """Skewness of log excess path load relative to height peers."""
    log_R = np.full(N_pts, np.nan)

    unique_heights = np.unique(heights[bulk_mask])
    for h in unique_heights:
        h_mask = bulk_mask & (heights == h)
        n_h = np.sum(h_mask)
        if n_h < 3:
            continue
        mean_down = np.mean(p_down[h_mask])
        mean_up = np.mean(p_up[h_mask])
        if mean_down <= 0 or mean_up <= 0:
            continue
        R = (p_down[h_mask] * p_up[h_mask]) / (mean_down * mean_up)
        log_R[h_mask] = np.log2(R + 1e-30)

    vals = log_R[~np.isnan(log_R)]
    if len(vals) < 10:
        return 0.0
    return float(stats.skew(vals))


# ═══════════════════════════════════════════════════════════════════════
# Observable SP-4: path_divergence_cv
# ═══════════════════════════════════════════════════════════════════════
# For each element x with k_out >= 2, define:
#   D(x) = std({log2(p_up(y)+1) : y child of x}) / mean({log2(p_up(y)+1)})
# This is the CV of log-path-up among children.
# D(x) measures how much the future path structure DIVERGES from x.
# TC-immunity: log2(p_up(y)+1) shifts by ~const under TC change; CV is
# scale-invariant but NOT shift-invariant. However the shift from TC is
# the SAME for all children (same depth), so it inflates both numerator
# and denominator equally → D(x) approximately TC-invariant.
# Then: path_divergence_cv = mean({D(x) : bulk, k_out >= 2})
# We also report its kurtosis for shape information.
def compute_path_divergence_cv(L_dir_csr, p_up, N_pts, bulk_mask):
    """Mean CV of future path-up among children (path divergence)."""
    cvs = []
    for i in range(N_pts):
        if not bulk_mask[i]:
            continue
        children = L_dir_csr.getrow(i).indices
        if len(children) < 2:
            continue
        log_child_up = np.log2(p_up[children] + 1)
        m = np.mean(log_child_up)
        if m > 0.5:  # require non-trivial path counts
            cv = np.std(log_child_up) / m
            cvs.append(cv)

    if len(cvs) < 10:
        return 0.0, 0.0
    return float(np.mean(cvs)), float(stats.kurtosis(cvs, fisher=True))


# ═══════════════════════════════════════════════════════════════════════
# Control observables
# ═══════════════════════════════════════════════════════════════════════
def compute_controls(C_sp, p_down, p_up, N_pts):
    """Compute certified control observables: column_gini_C2, path_kurtosis."""
    C2 = (C_sp @ C_sp).toarray()
    col_norms_C2 = np.sqrt(np.sum(C2 ** 2, axis=0).astype(np.float64))
    cgc2 = gini(col_norms_C2)
    del C2

    log_p = np.log2(p_down * p_up + 1)
    pk = float(stats.kurtosis(log_p, fisher=True)) if N_pts > 10 else 0.0

    return {'column_gini_C2': cgc2, 'path_kurtosis': pk}


# ═══════════════════════════════════════════════════════════════════════
# Full observable computation for one causal matrix
# ═══════════════════════════════════════════════════════════════════════
def compute_all_obs(C, pts):
    """Compute all candidates, controls, and proxy statistics."""
    N_pts = C.shape[0]
    C_sp = sp.csr_matrix(C) if not sp.issparse(C) else C
    tc = float(C_sp.sum())

    obs = {'tc': tc, 'N': N_pts}

    # Degree statistics (proxy)
    A_link = build_link_graph(C)
    degrees = np.array(A_link.sum(axis=1)).ravel()
    obs['mean_degree'] = float(np.mean(degrees))
    obs['degree_var'] = float(np.var(degrees))
    obs['degree_skew'] = float(stats.skew(degrees))
    obs['degree_kurt'] = float(stats.kurtosis(degrees))
    obs['max_degree'] = int(np.max(degrees))

    # Directed link graph (Hasse DAG)
    C_bool = (C_sp > 0).astype(np.float64)
    C2_sp = sp.csr_matrix((C_sp @ C_sp).toarray())
    C2_bool = (C2_sp > 0).astype(np.float64)
    L_dir = C_bool - C_bool.multiply(C2_bool)
    L_dir.eliminate_zeros()
    L_dir_csr = L_dir.tocsr()
    L_dir_csc = L_dir.tocsc()

    del C2_sp, C2_bool, C_bool
    gc.collect()

    # Path arrays
    p_down, p_up, heights = compute_path_arrays(L_dir_csr, L_dir_csc, N_pts)

    # Bulk mask (interior 80% by time coordinate)
    t_coords = pts[:, 0]
    t_lo, t_hi = np.percentile(t_coords, 10), np.percentile(t_coords, 90)
    bulk_mask = (t_coords >= t_lo) & (t_coords <= t_hi)

    # Controls
    ctrls = compute_controls(C_sp, p_down, p_up, N_pts)
    obs.update(ctrls)

    # SP-1: evenness_kurtosis
    obs['evenness_kurtosis'] = compute_evenness_kurtosis(
        L_dir_csr, p_up, N_pts, bulk_mask)

    # SP-2: path_asymmetry_kurt
    obs['path_asymmetry_kurt'] = compute_path_asymmetry_kurt(
        p_down, p_up, heights, bulk_mask, N_pts)

    # SP-3: path_excess_skew
    obs['path_excess_skew'] = compute_path_excess_skew(
        p_down, p_up, heights, bulk_mask, N_pts)

    # SP-4: path_divergence_cv (returns mean_cv, kurt_cv)
    div_cv, div_kurt = compute_path_divergence_cv(
        L_dir_csr, p_up, N_pts, bulk_mask)
    obs['path_divergence_cv'] = div_cv
    obs['path_divergence_cv_kurt'] = div_kurt

    del L_dir_csr, L_dir_csc, L_dir, A_link
    gc.collect()

    return obs


# ═══════════════════════════════════════════════════════════════════════
# CRN Trial: flat + curved on same sprinkling
# ═══════════════════════════════════════════════════════════════════════
def crn_trial(seed, metric_name, eps):
    """One CRN trial: flat + curved, compute all observables, return delta."""
    offset = SEED_OFFSETS[metric_name]
    rng = np.random.default_rng(seed + offset)
    pts = sprinkle_4d(N, T_DIAMOND, rng)

    # Flat
    C_flat = causal_flat(pts)
    obs_flat = compute_all_obs(C_flat, pts)
    del C_flat; gc.collect()

    # Curved
    if eps == 0.0:
        # Conformal null = identical to flat
        C_curv = causal_conformal(pts, eps)
    else:
        C_curv = METRICS[metric_name][0](pts, eps)
    obs_curv = compute_all_obs(C_curv, pts)
    del C_curv; gc.collect()

    # Build delta record
    result = {'seed': seed, 'metric': metric_name, 'eps': eps, 'N': N}
    all_keys = set(obs_flat.keys()) & set(obs_curv.keys())
    for key in all_keys:
        if isinstance(obs_flat[key], (int, float)):
            result[f'{key}_flat'] = obs_flat[key]
            result[f'{key}_curv'] = obs_curv[key]
            result[f'{key}_delta'] = obs_curv[key] - obs_flat[key]

    return result


# ═══════════════════════════════════════════════════════════════════════
# Analysis: CRN delta statistics + TC decontamination + proxy checks
# ═══════════════════════════════════════════════════════════════════════
def analyze_candidate(results, key, label=""):
    """Full analysis of one candidate: effect size, TC residual, proxy R2."""
    M_eff = len(results)
    deltas = [r.get(f'{key}_delta', 0) for r in results]
    tc_deltas = [r.get('tc_delta', 0) for r in results]

    if all(d == 0 for d in deltas):
        return {'key': key, 'verdict': 'NULL (all zeros)', 'd_raw': 0,
                'd_tc_resid': 0, 'p_raw': 1.0, 'p_tc_resid': 1.0,
                'max_r2_proxy': 0.0}

    arr = np.array(deltas, dtype=np.float64)
    tc_arr = np.array(tc_deltas, dtype=np.float64)

    # Raw effect size
    d_raw = np.mean(arr) / np.std(arr) if np.std(arr) > 0 else 0.0
    _, p_raw = stats.ttest_1samp(arr, 0.0) if M_eff >= 5 else (0, 1.0)

    # TC-decontaminated residual
    if np.std(tc_arr) > 1e-15 and np.std(arr) > 1e-15:
        slope_tc = np.polyfit(tc_arr, arr, 1)[0]
        residuals = arr - slope_tc * tc_arr
    else:
        residuals = arr.copy()
    d_tc = np.mean(residuals) / np.std(residuals) if np.std(residuals) > 0 else 0.0
    _, p_tc = stats.ttest_1samp(residuals, 0.0) if M_eff >= 5 else (0, 1.0)

    # Proxy R2: check against all simple graph statistics
    proxy_stats = ['tc_delta', 'mean_degree_delta', 'degree_var_delta',
                   'degree_skew_delta', 'degree_kurt_delta', 'max_degree_delta']
    max_r2 = 0.0
    max_r2_name = ""
    for ps in proxy_stats:
        vals = np.array([r.get(ps, 0) for r in results], dtype=np.float64)
        if np.std(vals) > 1e-15 and np.std(arr) > 1e-15:
            r2 = np.corrcoef(arr, vals)[0, 1] ** 2
        else:
            r2 = 0.0
        if r2 > max_r2:
            max_r2 = r2
            max_r2_name = ps

    # Cross-candidate leakage
    cross_r2 = {}
    for ck in CTRL_KEYS:
        cv = np.array([r.get(f'{ck}_delta', 0) for r in results], dtype=np.float64)
        if np.std(cv) > 1e-15 and np.std(arr) > 1e-15:
            cross_r2[ck] = float(np.corrcoef(arr, cv)[0, 1] ** 2)
        else:
            cross_r2[ck] = 0.0

    # Verdict
    ALPHA_BONF = 0.01 / 20
    if p_tc < ALPHA_BONF and max_r2 < 0.50:
        verdict = "DETECTED (genuine)"
    elif p_tc < ALPHA_BONF and max_r2 >= 0.80:
        verdict = "PROXY (explains >80%)"
    elif p_tc < ALPHA_BONF and 0.50 <= max_r2 < 0.80:
        verdict = "AMBIGUOUS (50-80% proxy)"
    elif p_tc < 0.05:
        verdict = "WEAK (p<0.05, not Bonferroni)"
    else:
        verdict = "NULL"

    report = {
        'key': key,
        'verdict': verdict,
        'd_raw': float(d_raw),
        'p_raw': float(p_raw),
        'd_tc_resid': float(d_tc),
        'p_tc_resid': float(p_tc),
        'max_r2_proxy': float(max_r2),
        'max_r2_proxy_name': max_r2_name,
        'cross_r2': cross_r2,
        'mean_delta': float(np.mean(arr)),
        'mean_tc_resid': float(np.mean(residuals)),
    }

    print(f"\n  {key} ({label}):")
    print(f"    d_raw = {d_raw:+.3f}, p = {p_raw:.2e}")
    print(f"    d_tc_resid = {d_tc:+.3f}, p = {p_tc:.2e}")
    print(f"    max R2(proxy) = {max_r2:.3f} ({max_r2_name})")
    for ck, rv in cross_r2.items():
        if rv > 0.3:
            print(f"    R2({ck}) = {rv:.3f} {'!!!' if rv > 0.7 else ''}")
    print(f"    VERDICT: {verdict}")

    return report


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("SEED INVERSION (PATH)")
    print("Path-kurtosis TC-independence mechanism: per-element shape stats")
    print(f"N={N}, M={M}, CRN protocol, master_seed={MASTER_SEED}")
    print("=" * 72)

    all_results = {}
    t0_total = time.time()

    for metric_name, (metric_fn, eps) in METRICS.items():
        print(f"\n{'─' * 60}")
        print(f"  METRIC: {metric_name}, eps={eps}")
        print(f"{'─' * 60}")

        results = []
        t0 = time.time()
        for trial in range(M):
            seed = MASTER_SEED + trial
            r = crn_trial(seed, metric_name, eps)
            results.append(r)
            if (trial + 1) % 10 == 0 or trial == 0:
                elapsed = time.time() - t0
                rate = (trial + 1) / elapsed if elapsed > 0 else 0
                print(f"    trial {trial+1}/{M}  "
                      f"({elapsed:.1f}s, {rate:.1f} trials/s)")
                gc.collect()

        elapsed = time.time() - t0
        print(f"  Completed {M} trials in {elapsed:.1f}s")

        # Analyze controls
        for ck in CTRL_KEYS:
            analyze_candidate(results, ck, label=f"CONTROL on {metric_name}")

        # Analyze candidates
        metric_reports = {}
        for cand in CAND_KEYS:
            report = analyze_candidate(results, cand,
                                       label=f"CANDIDATE on {metric_name}")
            metric_reports[cand] = report

        # Also analyze path_divergence_cv_kurt as bonus
        analyze_candidate(results, 'path_divergence_cv_kurt',
                          label=f"BONUS on {metric_name}")

        all_results[metric_name] = {
            'raw_trials': results,
            'candidate_reports': metric_reports,
        }

    total_time = time.time() - t0_total
    print(f"\n{'=' * 72}")
    print(f"TOTAL TIME: {total_time:.1f}s")

    # ─── SUMMARY TABLE ───────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("SUMMARY TABLE")
    print(f"{'=' * 72}")
    header = f"{'Observable':<25s} {'ppwave_quad':<18s} {'schwarzschild':<18s} {'conformal':<18s}"
    print(header)
    print("─" * 79)

    for key in CAND_KEYS + CTRL_KEYS + ['path_divergence_cv_kurt']:
        row = f"{key:<25s}"
        for mn in METRICS:
            trials = all_results[mn]['raw_trials']
            deltas = [r.get(f'{key}_delta', 0) for r in trials]
            arr = np.array(deltas)
            if np.std(arr) > 0:
                d = np.mean(arr) / np.std(arr)
            else:
                d = 0.0
            row += f" d={d:+6.2f}"
        print(row)

    # ─── PAIRWISE SPEARMAN ───────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("PAIRWISE SPEARMAN (pp-wave deltas)")
    print(f"{'=' * 72}")
    pp_trials = all_results.get('ppwave_quad', {}).get('raw_trials', [])
    if pp_trials:
        all_obs_keys = CAND_KEYS + CTRL_KEYS
        data = {}
        for k in all_obs_keys:
            data[k] = np.array([r.get(f'{k}_delta', 0) for r in pp_trials])

        print(f"{'':>25s}", end="")
        for k in all_obs_keys:
            print(f" {k[:10]:>10s}", end="")
        print()
        for k1 in all_obs_keys:
            print(f"{k1:>25s}", end="")
            for k2 in all_obs_keys:
                if np.std(data[k1]) > 1e-15 and np.std(data[k2]) > 1e-15:
                    rho, _ = stats.spearmanr(data[k1], data[k2])
                else:
                    rho = 0.0
                print(f" {rho:+10.3f}", end="")
            print()

    # ─── SAVE ─────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'seed_inversion_r008_path.json')

    # Serialize: strip raw trials for compactness, keep reports
    save_data = {}
    for mn in METRICS:
        save_data[mn] = {
            'candidate_reports': all_results[mn]['candidate_reports'],
            # Store just delta arrays for reproducibility
            'delta_arrays': {}
        }
        for k in CAND_KEYS + CTRL_KEYS:
            save_data[mn]['delta_arrays'][k] = [
                r.get(f'{k}_delta', 0) for r in all_results[mn]['raw_trials']
            ]

    save_data['params'] = {'N': N, 'M': M, 'T': T_DIAMOND,
                           'master_seed': MASTER_SEED}
    save_data['total_time_s'] = total_time

    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'=' * 72}")
    print("SEED INVERSION COMPLETE")
    print(f"{'=' * 72}")


if __name__ == '__main__':
    main()
