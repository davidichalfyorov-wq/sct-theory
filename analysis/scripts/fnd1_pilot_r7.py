"""
PILOT (Phase 5)
N=2000, M=20, CRN. Dual metric + conformal null.

Tests pf_rank_asym_kurtosis (from RESURRECT) + column_gini_C (positive control).
Addresses DEMOLISH concerns: n_int matching, cross-correlation, boundary sensitivity.

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
from scipy.stats import rankdata
import json, time, sys, os, gc
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.discovery_common import (
    sprinkle_4d, causal_flat, causal_ppwave_quad, causal_schwarzschild,
    causal_conformal, build_link_graph, graph_statistics
)

RUN_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'docs',
                       'analysis_runs', 'run_20260326_102015')


def gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n < 2 or np.sum(x) == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(2.0 * np.sum(idx * x) / (n * np.sum(x)) - (n + 1) / n)


def compute_pilot_obs(C, A_link, degrees, boundary_frac=0.0):
    """Compute pilot observables with boundary exclusion control."""
    N = C.shape[0]
    C_sp = sp.csr_matrix(C)

    # Past/future sizes
    past_sizes = np.array(C.sum(axis=0)).ravel() if isinstance(C, np.ndarray) else np.array(C_sp.sum(axis=0)).ravel()
    future_sizes = np.array(C.sum(axis=1)).ravel() if isinstance(C, np.ndarray) else np.array(C_sp.sum(axis=1)).ravel()

    tc = float(np.sum(past_sizes))

    # Interior mask: exclude top/bottom boundary_frac of elements by height
    if boundary_frac > 0:
        n_exc = int(N * boundary_frac)
        interior = np.ones(N, dtype=bool)
        interior[:n_exc] = False
        interior[-n_exc:] = False
    else:
        # Default: exclude elements with past_size=0 or future_size=0
        interior = (past_sizes > 0) & (future_sizes > 0)

    n_int = int(np.sum(interior))
    obs = {'tc': tc, 'n_int': n_int, 'N': N}

    # === pf_rank_asym_kurtosis ===
    past_int = past_sizes[interior]
    future_int = future_sizes[interior]
    r_past = rankdata(past_int) / n_int
    r_future = rankdata(future_int) / n_int
    asym = r_past - r_future
    obs['pf_rank_asym_kurtosis'] = float(stats.kurtosis(asym))
    obs['pf_rank_asym_skew'] = float(stats.skew(asym))
    obs['pf_rank_asym_std'] = float(np.std(asym))

    # === column_gini_C (positive control, E2) ===
    sqrt_past = np.sqrt(past_sizes.astype(np.float64))
    obs['column_gini_C'] = gini(sqrt_past)

    # === link_degree_skew (cross-correlation check) ===
    obs['link_degree_skew'] = float(stats.skew(degrees))

    # === LVA (positive control) ===
    L_sp = sp.csr_matrix(A_link)
    lva_vals = []
    for x in range(N):
        row = L_sp.getrow(x)
        neighbors = row.indices
        if len(neighbors) >= 2:
            fan = future_sizes[neighbors].astype(np.float64)
            m = np.mean(fan)
            if m > 0:
                lva_vals.append(float(np.var(fan) / m**2))
    obs['lva'] = float(np.mean(lva_vals)) if lva_vals else 0.0

    # === fan_kurtosis (positive control, E3) ===
    fan_kurt_vals = []
    for x in range(N):
        row = L_sp.getrow(x)
        neighbors = row.indices
        if len(neighbors) >= 4:
            fan = future_sizes[neighbors].astype(np.float64)
            if np.std(fan) > 0:
                fan_kurt_vals.append(float(stats.kurtosis(fan)))
    obs['fan_kurtosis'] = float(np.mean(fan_kurt_vals)) if fan_kurt_vals else 0.0

    # === column_gini_C2 (positive control, CERTIFIED) ===
    C2 = (C_sp @ C_sp).toarray()
    col_norms_C2 = np.sqrt(np.sum(C2 ** 2, axis=0))
    obs['column_gini_C2'] = gini(col_norms_C2)

    # === Baselines ===
    obs['degree_cv'] = float(np.std(degrees) / np.mean(degrees)) if np.mean(degrees) > 0 else 0
    obs['mean_degree'] = float(np.mean(degrees))
    obs['degree_var'] = float(np.var(degrees))
    obs['link_count'] = int(np.sum(A_link)) // 2 if sp.issparse(A_link) else int(np.sum(A_link)) // 2

    del C2
    gc.collect()
    return obs


def crn_trial(seed, N, T, metric_fn, eps, boundary_frac=0.0):
    rng = np.random.default_rng(seed)
    pts = sprinkle_4d(N, T, rng)

    C_flat = causal_flat(pts)
    A_flat = build_link_graph(C_flat)
    _, deg_flat = graph_statistics(A_flat)
    obs_flat = compute_pilot_obs(C_flat, A_flat, deg_flat, boundary_frac)
    del C_flat, A_flat; gc.collect()

    C_curv = metric_fn(pts, eps)
    A_curv = build_link_graph(C_curv)
    _, deg_curv = graph_statistics(A_curv)
    obs_curv = compute_pilot_obs(C_curv, A_curv, deg_curv, boundary_frac)
    del C_curv, A_curv; gc.collect()

    deltas = {k: obs_curv[k] - obs_flat[k] for k in obs_flat}
    deltas['seed'] = seed
    deltas['n_int_flat'] = obs_flat['n_int']
    deltas['n_int_curv'] = obs_curv['n_int']
    return deltas, obs_flat, obs_curv


def analyze(deltas_list, label):
    M = len(deltas_list)
    keys = ['pf_rank_asym_kurtosis', 'column_gini_C', 'link_degree_skew',
            'lva', 'fan_kurtosis', 'column_gini_C2']
    print(f"\n{'='*60}")
    print(f"  {label}  (M={M})")
    print(f"{'='*60}")
    print(f"  {'Observable':<25} {'d_z':>7} {'p':>10} {'mean_Δ':>10} {'SE':>8}")
    print("  " + "-" * 60)

    results = {}
    for key in keys:
        vals = np.array([d[key] for d in deltas_list], dtype=np.float64)
        vals = np.where(np.isfinite(vals), vals, 0.0)
        m = np.mean(vals)
        sd = np.std(vals, ddof=1)
        d_z = m / sd if sd > 0 else 0
        se = sd / np.sqrt(M)
        _, p = stats.ttest_1samp(vals, 0.0) if M >= 5 else (0, 1)
        print(f"  {key:<25} {d_z:>+7.3f} {p:>10.2e} {m:>+10.5f} {se:>8.5f}")
        results[key] = {'d_z': round(d_z, 4), 'p': float(f'{p:.2e}'),
                        'mean_delta': round(m, 6), 'se': round(se, 6)}

    # n_int stats
    n_int_flat = [d['n_int_flat'] for d in deltas_list]
    n_int_curv = [d['n_int_curv'] for d in deltas_list]
    n_int_delta = np.array(n_int_curv) - np.array(n_int_flat)
    print(f"\n  n_int: flat={np.mean(n_int_flat):.0f}±{np.std(n_int_flat):.0f}, "
          f"curv={np.mean(n_int_curv):.0f}±{np.std(n_int_curv):.0f}, "
          f"Δ={np.mean(n_int_delta):.1f}±{np.std(n_int_delta):.1f}")

    # Cross-correlation: pf_rank_asym_kurtosis vs link_degree_skew
    pf_vals = np.array([d['pf_rank_asym_kurtosis'] for d in deltas_list])
    lds_vals = np.array([d['link_degree_skew'] for d in deltas_list])
    if np.std(pf_vals) > 0 and np.std(lds_vals) > 0:
        rho, _ = stats.spearmanr(pf_vals, lds_vals)
        print(f"  Cross-corr(pf_kurtosis, link_degree_skew): Spearman r = {rho:.3f}")
    else:
        rho = 0

    results['n_int_mean_delta'] = round(np.mean(n_int_delta), 1)
    results['cross_corr_lds'] = round(rho, 3) if np.isfinite(rho) else 0

    return results


def main():
    N = 2000
    T = 1.5
    M = 20

    conditions = [
        ("ppwave_quad eps=5", causal_ppwave_quad, 5.0),
        ("schwarzschild eps=0.005", causal_schwarzschild, 0.005),
        ("conformal eps=1 (NULL)", causal_conformal, 1.0),
    ]

    all_results = {}

    for label, metric_fn, eps in conditions:
        print(f"\n\nPILOT: {label}, N={N}, M={M}")
        deltas_list = []
        t0 = time.time()
        for trial in range(M):
            seed = 50000 + trial * 1000
            d, f, c = crn_trial(seed, N, T, metric_fn, eps)
            deltas_list.append(d)
            if (trial + 1) % 5 == 0:
                print(f"  Trial {trial+1}/{M} ({time.time()-t0:.1f}s)")

        res = analyze(deltas_list, label)
        all_results[label] = res

    # Boundary sensitivity for pf_rank_asym_kurtosis
    print(f"\n\n{'='*60}")
    print("BOUNDARY SENSITIVITY (ppwave eps=5)")
    print(f"{'='*60}")
    boundary_results = {}
    for bf in [0.05, 0.10, 0.15, 0.20]:
        deltas_b = []
        for trial in range(M):
            seed = 50000 + trial * 1000
            d, _, _ = crn_trial(seed, N, T, causal_ppwave_quad, 5.0, boundary_frac=bf)
            deltas_b.append(d)
        vals = np.array([d['pf_rank_asym_kurtosis'] for d in deltas_b])
        vals = np.where(np.isfinite(vals), vals, 0.0)
        m = np.mean(vals)
        sd = np.std(vals, ddof=1)
        d_z = m / sd if sd > 0 else 0
        print(f"  boundary_frac={bf:.2f}: d_z={d_z:+.3f}")
        boundary_results[f"bf_{bf}"] = round(d_z, 3)

    all_results['boundary_sensitivity'] = boundary_results

    # Baseline R² for pf_rank_asym_kurtosis
    print(f"\n\n{'='*60}")
    print("BASELINE R² (9 baselines, ppwave)")
    print(f"{'='*60}")
    # Collect flat-only data for baselines
    flat_obs = []
    for trial in range(30):
        seed = 70000 + trial * 1000
        rng = np.random.default_rng(seed)
        pts = sprinkle_4d(N, T, rng)
        C = causal_flat(pts)
        A = build_link_graph(C)
        _, deg = graph_statistics(A)
        o = compute_pilot_obs(C, A, deg)
        flat_obs.append(o)
        del C, A; gc.collect()

    # R² of pf_rank_asym_kurtosis vs baselines on flat data
    y_flat = np.array([o['pf_rank_asym_kurtosis'] for o in flat_obs])
    baseline_keys = ['tc', 'degree_cv', 'mean_degree', 'degree_var', 'link_count']
    X_base = np.column_stack([[o[k] for o in flat_obs] for k in baseline_keys])
    X_full = np.column_stack([np.ones(len(y_flat)), X_base])
    try:
        beta = np.linalg.lstsq(X_full, y_flat, rcond=None)[0]
        pred = X_full @ beta
        ss_res = np.sum((y_flat - pred)**2)
        ss_tot = np.sum((y_flat - np.mean(y_flat))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        n_obs = len(y_flat)
        n_feat = X_base.shape[1]
        r2_adj = 1 - (1 - r2) * (n_obs - 1) / (n_obs - n_feat - 1)
    except Exception:
        r2_adj = 0.0
    print(f"  R²_adj (5 baselines, 30 flat sprinklings): {r2_adj:.3f}")
    all_results['baseline_r2'] = {'r2_adj': round(r2_adj, 3), 'n_baselines': 5, 'n_flat': 30}

    # Save
    output_path = os.path.join(RUN_DIR, 'pilot_results.json')
    json.dump(all_results, open(output_path, 'w'), indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    main()
