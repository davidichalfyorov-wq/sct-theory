"""
RED-TEAM Phase 9.5
6 attacks on column_gini_C.

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
import json, time, sys, os, gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.discovery_common import (
    sprinkle_4d, causal_flat, causal_ppwave_quad, causal_schwarzschild,
    causal_ppwave_cross, causal_flrw,
    build_link_graph, graph_statistics
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


def compute_gini_C(C):
    if sp.issparse(C):
        past_sizes = np.array(C.sum(axis=0)).ravel()
    else:
        past_sizes = np.sum(C, axis=0)
    return gini(np.sqrt(past_sizes.astype(np.float64)))


def crn_delta(seed, N, T, metric_fn, eps):
    rng = np.random.default_rng(seed)
    pts = sprinkle_4d(N, T, rng)
    C_flat = causal_flat(pts)
    C_curv = metric_fn(pts, eps)
    d = compute_gini_C(C_curv) - compute_gini_C(C_flat)
    del C_flat, C_curv; gc.collect()
    return d


def main():
    N = 2000
    T = 1.5
    M = 15
    results = {}

    # ═══════════════════════════════════════════════════════════════
    # ATTACK 1: BOUNDARY EXCLUSION {5,10,20,30}%
    # ═══════════════════════════════════════════════════════════════
    print("ATTACK 1: Boundary exclusion sweep (Schwarzschild)")
    boundary = {}
    for bf_pct in [5, 10, 20, 30]:
        bf = bf_pct / 100.0
        deltas = []
        for trial in range(M):
            seed = 80000 + trial * 1000
            rng = np.random.default_rng(seed)
            pts = sprinkle_4d(N, T, rng)
            C_flat = causal_flat(pts)
            C_curv = causal_schwarzschild(pts, 0.005)
            ps_flat = np.sum(C_flat, axis=0)
            n_exc = int(N * bf)
            order = np.argsort(ps_flat)
            keep = order[n_exc:-n_exc] if n_exc > 0 else order
            ps_curv = np.sum(C_curv, axis=0)
            g_flat = gini(np.sqrt(ps_flat[keep].astype(np.float64)))
            g_curv = gini(np.sqrt(ps_curv[keep].astype(np.float64)))
            deltas.append(g_curv - g_flat)
            del C_flat, C_curv; gc.collect()

        arr = np.array(deltas)
        d_z = np.mean(arr) / np.std(arr, ddof=1) if np.std(arr, ddof=1) > 0 else 0
        boundary[f"{bf_pct}%"] = round(d_z, 3)
        print(f"  {bf_pct}%: d_z={d_z:+.3f}")

    # Kill criterion: |d| monotonically decreases AND d_30% < 0.5 * d_5%
    d5 = abs(boundary.get("5%", 0))
    d30 = abs(boundary.get("30%", 0))
    mono_decrease = all(abs(boundary.get(f"{p}%", 0)) >= abs(boundary.get(f"{q}%", 0))
                        for p, q in [(5, 10), (10, 20), (20, 30)])
    kill_boundary = mono_decrease and d30 < 0.5 * d5
    print(f"  KILL criterion: mono_decrease={mono_decrease}, d30/d5={d30/d5 if d5>0 else 'inf':.2f}")
    print(f"  VERDICT: {'KILL' if kill_boundary else 'PASS'}")
    results['attack_1_boundary'] = {
        'values': boundary, 'mono_decrease': mono_decrease,
        'd30_over_d5': round(d30 / d5, 3) if d5 > 0 else None,
        'verdict': 'KILL' if kill_boundary else 'PASS'
    }

    # ═══════════════════════════════════════════════════════════════
    # ATTACK 2: RANDOM DAG (matched density, no geometry)
    # ═══════════════════════════════════════════════════════════════
    print("\nATTACK 2: Random DAG (matched density)")
    rdag_deltas = []
    geo_deltas = []
    for trial in range(M):
        seed = 80000 + trial * 1000
        # Geometric
        geo_d = crn_delta(seed, N, T, causal_schwarzschild, 0.005)
        geo_deltas.append(geo_d)
        # Random DAG: CRN between flat geo and random DAG with same edge count
        rng = np.random.default_rng(seed)
        pts = sprinkle_4d(N, T, rng)
        C_flat = causal_flat(pts)
        tc = int(np.sum(C_flat))
        g_flat = compute_gini_C(C_flat)
        # Random upper triangular with same tc
        rng2 = np.random.default_rng(seed + 777)
        rows_u, cols_u = np.triu_indices(N, k=1)
        idx = rng2.choice(len(rows_u), size=tc, replace=False)
        C_rng = np.zeros((N, N))
        C_rng[rows_u[idx], cols_u[idx]] = 1.0
        g_rng = compute_gini_C(C_rng)
        rdag_deltas.append(g_rng - g_flat)
        del C_flat, C_rng; gc.collect()

    d_geo = np.mean(geo_deltas)
    d_rdag = np.mean(rdag_deltas)
    ratio = abs(d_rdag) / abs(d_geo) if abs(d_geo) > 0 else float('inf')
    kill_rdag = ratio > 0.3
    print(f"  d_geo = {d_geo:+.6f}, d_rdag = {d_rdag:+.6f}")
    print(f"  |d_rdag/d_geo| = {ratio:.3f}")
    print(f"  VERDICT: {'KILL' if kill_rdag else 'PASS'}")
    results['attack_2_random_dag'] = {
        'd_geo': round(d_geo, 6), 'd_rdag': round(d_rdag, 6),
        'ratio': round(ratio, 3),
        'verdict': 'KILL' if kill_rdag else 'PASS'
    }

    # ═══════════════════════════════════════════════════════════════
    # ATTACK 3: THRESHOLDING ±10%
    # ═══════════════════════════════════════════════════════════════
    print("\nATTACK 3: Thresholding sensitivity ±10% (Schwarzschild)")
    # Perturb eps by ±10%
    thresh_results = {}
    for eps_mult in [0.9, 1.0, 1.1]:
        eps = 0.005 * eps_mult
        deltas = []
        for trial in range(M):
            seed = 80000 + trial * 1000
            d = crn_delta(seed, N, T, causal_schwarzschild, eps)
            deltas.append(d)
        arr = np.array(deltas)
        d_z = np.mean(arr) / np.std(arr, ddof=1) if np.std(arr, ddof=1) > 0 else 0
        thresh_results[f"eps={eps:.4f}"] = round(d_z, 3)
        print(f"  eps={eps:.4f}: d_z={d_z:+.3f}")

    # Check: sign flip or |d| change > 50%
    d_base = abs(thresh_results.get("eps=0.0050", 1))
    d_low = abs(thresh_results.get("eps=0.0045", 0))
    d_high = abs(thresh_results.get("eps=0.0055", 0))
    max_change = max(abs(d_low - d_base), abs(d_high - d_base)) / d_base if d_base > 0 else 0
    sign_flip = any(thresh_results.get(k, 0) * thresh_results.get("eps=0.0050", 1) < 0
                    for k in thresh_results if k != "eps=0.0050")
    print(f"  Max relative change: {max_change:.2f}")
    print(f"  Sign flip: {sign_flip}")
    print(f"  VERDICT: {'FLAG' if max_change > 0.5 or sign_flip else 'PASS'}")
    results['attack_3_threshold'] = {
        'values': thresh_results,
        'max_relative_change': round(max_change, 3),
        'sign_flip': sign_flip,
        'verdict': 'FLAG' if max_change > 0.5 or sign_flip else 'PASS'
    }

    # ═══════════════════════════════════════════════════════════════
    # ATTACK 4: NEW METRIC FAMILY (pp-wave cross, NOT in original set)
    # ═══════════════════════════════════════════════════════════════
    print("\nATTACK 4: New metric family (ppwave_cross eps=5)")
    deltas_cross = []
    for trial in range(M):
        seed = 80000 + trial * 1000
        d = crn_delta(seed, N, T, causal_ppwave_cross, 5.0)
        deltas_cross.append(d)
    arr_cross = np.array(deltas_cross)
    d_z_cross = np.mean(arr_cross) / np.std(arr_cross, ddof=1) if np.std(arr_cross, ddof=1) > 0 else 0
    _, p_cross = stats.ttest_1samp(arr_cross, 0.0) if M >= 5 else (0, 1)
    print(f"  d_z = {d_z_cross:+.3f}, p = {p_cross:.2e}")
    print(f"  VERDICT: {'FAIL' if p_cross > 0.05 else 'PASS'}")
    results['attack_4_new_metric'] = {
        'd_z': round(d_z_cross, 3), 'p': float(f"{p_cross:.2e}"),
        'metric': 'ppwave_cross_eps5',
        'verdict': 'FAIL' if p_cross > 0.05 else 'PASS'
    }

    # ═══════════════════════════════════════════════════════════════
    # ATTACK 5: DESIGN LEAKAGE
    # ═══════════════════════════════════════════════════════════════
    print("\nATTACK 5: Design leakage (Spearman with simpler observables)")
    # Compute column_gini_C and simpler observables on flat data
    from scripts.discovery_common import graph_statistics
    flat_gini = []
    flat_tc = []
    flat_link = []
    flat_degcv = []
    flat_degvar = []
    flat_colg_c2 = []
    flat_lva = []
    for trial in range(30):
        seed = 70000 + trial * 1000
        rng = np.random.default_rng(seed)
        pts = sprinkle_4d(N, T, rng)
        C = causal_flat(pts)
        C_sp = sp.csr_matrix(C)
        past_sizes = np.array(C_sp.sum(axis=0)).ravel()
        future_sizes = np.array(C_sp.sum(axis=1)).ravel()
        flat_gini.append(gini(np.sqrt(past_sizes.astype(np.float64))))
        flat_tc.append(float(np.sum(C)))
        A = build_link_graph(C)
        _, deg = graph_statistics(A)
        flat_link.append(int(np.sum(A)) // 2)
        flat_degcv.append(float(np.std(deg) / np.mean(deg)) if np.mean(deg) > 0 else 0)
        flat_degvar.append(float(np.var(deg)))
        C2 = (C_sp @ C_sp).toarray()
        col_norms_c2 = np.sqrt(np.sum(C2 ** 2, axis=0))
        flat_colg_c2.append(gini(col_norms_c2))
        # LVA
        L_sp = sp.csr_matrix(A)
        lva_vals = []
        for x in range(N):
            row = L_sp.getrow(x)
            neighbors = row.indices
            if len(neighbors) >= 2:
                fan = future_sizes[neighbors].astype(np.float64)
                m = np.mean(fan)
                if m > 0:
                    lva_vals.append(float(np.var(fan) / m**2))
        flat_lva.append(float(np.mean(lva_vals)) if lva_vals else 0)
        del C, A, C2; gc.collect()

    # Spearman correlations
    leakage = {}
    for name, vals in [('tc', flat_tc), ('link_count', flat_link),
                       ('degree_cv', flat_degcv), ('degree_var', flat_degvar),
                       ('column_gini_C2', flat_colg_c2), ('LVA', flat_lva)]:
        rho, _ = stats.spearmanr(flat_gini, vals)
        leakage[name] = round(float(rho), 3) if np.isfinite(rho) else 0
        print(f"  Spearman(column_gini_C, {name}): {rho:+.3f}")

    max_rho = max(abs(v) for v in leakage.values())
    kill_leakage = max_rho > 0.8
    print(f"  Max |Spearman|: {max_rho:.3f}")
    print(f"  VERDICT: {'KILL' if kill_leakage else 'PASS'}")
    results['attack_5_leakage'] = {
        'correlations': leakage,
        'max_abs_spearman': round(max_rho, 3),
        'verdict': 'KILL' if kill_leakage else 'PASS'
    }

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("RED-TEAM SUMMARY")
    print(f"{'='*60}")
    for i, (attack, res) in enumerate(results.items(), 1):
        print(f"  Attack {i}: {res['verdict']}")

    overall = 'KILL' if any(r['verdict'] == 'KILL' for r in results.values()) else \
              'FLAG' if any(r['verdict'] in ('FLAG', 'FAIL') for r in results.values()) else 'PASS'
    results['overall_verdict'] = overall
    print(f"\n  OVERALL: {overall}")

    output_path = os.path.join(RUN_DIR, 'redteam_results.json')
    json.dump(results, open(output_path, 'w'), indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
