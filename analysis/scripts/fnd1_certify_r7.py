"""
CERTIFICATION STACK (Phase 7-9.5)
Observable: column_gini_C = Gini({sqrt(|past(j)|)})
Tests: N=5000, holdout de Sitter, graph-theory adversary, boundary sweep,
       convergence d(N), dose-response, random DAG control.

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
from scipy.stats import rankdata
import json, time, sys, os, gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.discovery_common import (
    sprinkle_4d, causal_flat, causal_ppwave_quad, causal_schwarzschild,
    causal_flrw, build_link_graph, graph_statistics
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


def causal_desitter(pts, H):
    """de Sitter metric: ds² = -dt² + exp(2Ht)(dx² + dy² + dz²)"""
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dr2 = ((x[np.newaxis, :] - x[:, np.newaxis])**2 +
           (y[np.newaxis, :] - y[:, np.newaxis])**2 +
           (z[np.newaxis, :] - z[:, np.newaxis])**2)
    tm = (t[np.newaxis, :] + t[:, np.newaxis]) / 2.0
    a2 = np.exp(2.0 * H * tm)
    return ((dt**2 > a2 * dr2) & (dt > 0)).astype(np.float64)


def compute_gini_C(C):
    """column_gini_C = Gini(sqrt(past_sizes))"""
    if sp.issparse(C):
        past_sizes = np.array(C.sum(axis=0)).ravel()
    else:
        past_sizes = np.sum(C, axis=0)
    return gini(np.sqrt(past_sizes.astype(np.float64)))


def crn_trial_gini(seed, N, T, metric_fn, eps, boundary_frac=0.0):
    rng = np.random.default_rng(seed)
    pts = sprinkle_4d(N, T, rng)

    C_flat = causal_flat(pts)
    g_flat = compute_gini_C(C_flat)
    tc_flat = float(np.sum(C_flat))
    del C_flat; gc.collect()

    C_curv = metric_fn(pts, eps)
    g_curv = compute_gini_C(C_curv)
    tc_curv = float(np.sum(C_curv))
    del C_curv; gc.collect()

    return {
        'gini_flat': g_flat, 'gini_curv': g_curv,
        'delta': g_curv - g_flat,
        'tc_flat': tc_flat, 'tc_curv': tc_curv,
        'tc_delta': tc_curv - tc_flat,
        'seed': seed
    }


def random_dag_trial(seed, N, T):
    """Random DAG with matched density: permute off-diagonal entries of C."""
    rng = np.random.default_rng(seed)
    pts = sprinkle_4d(N, T, rng)
    C = causal_flat(pts)
    g_geo = compute_gini_C(C)

    # Random DAG: preserve total edges, randomize upper triangle
    tc = int(np.sum(C))
    n_upper = N * (N - 1) // 2
    rng2 = np.random.default_rng(seed + 999)
    idx = rng2.choice(n_upper, size=tc, replace=False)
    C_rng = np.zeros((N, N))
    rows_idx, cols_idx = np.triu_indices(N, k=1)
    C_rng[rows_idx[idx], cols_idx[idx]] = 1.0
    g_rng = compute_gini_C(C_rng)

    return {'gini_geo': g_geo, 'gini_rng': g_rng}


def adversary_trial(seed, N, T, metric_fn, eps):
    """Graph-theory adversary: random perturbation with same n_diff."""
    rng = np.random.default_rng(seed)
    pts = sprinkle_4d(N, T, rng)
    C_flat = causal_flat(pts)
    C_curv = metric_fn(pts, eps)

    diff = C_curv - C_flat
    n_diff = int(np.sum(np.abs(diff)))
    g_geo = compute_gini_C(C_curv) - compute_gini_C(C_flat)

    # Random perturbation: flip n_diff random upper-triangle entries
    rng2 = np.random.default_rng(seed + 888)
    rows_u, cols_u = np.triu_indices(N, k=1)
    flip_idx = rng2.choice(len(rows_u), size=min(n_diff, len(rows_u)), replace=False)
    C_rng = C_flat.copy()
    for idx in flip_idx:
        r, c = rows_u[idx], cols_u[idx]
        C_rng[r, c] = 1.0 - C_rng[r, c]
    g_rng = compute_gini_C(C_rng) - compute_gini_C(C_flat)

    del C_flat, C_curv, C_rng; gc.collect()
    return {'d_geo': g_geo, 'd_rng': g_rng, 'n_diff': n_diff}


def run_stat(vals, label):
    arr = np.array(vals)
    m = np.mean(arr)
    sd = np.std(arr, ddof=1)
    d_z = m / sd if sd > 0 else 0
    _, p = stats.ttest_1samp(arr, 0.0) if len(arr) >= 5 else (0, 1)
    print(f"  {label}: d_z={d_z:+.3f}, p={p:.2e}, mean={m:+.6f}, M={len(arr)}")
    return {'d_z': round(d_z, 4), 'p': float(f'{p:.2e}'), 'mean': round(m, 6), 'M': len(arr)}


def main():
    results = {}
    T = 1.5

    # ═══════════════════════════════════════════════════════════════
    # 1. N-SCALING CONVERGENCE
    # ═══════════════════════════════════════════════════════════════
    print("=" * 60)
    print("1. N-SCALING CONVERGENCE (ppwave eps=5)")
    print("=" * 60)
    n_scaling = {}
    for N in [500, 1000, 2000, 5000]:
        M = max(10, 20 if N <= 2000 else 15)
        deltas = []
        t0 = time.time()
        for trial in range(M):
            seed = 80000 + trial * 1000
            r = crn_trial_gini(seed, N, T, causal_ppwave_quad, 5.0)
            deltas.append(r['delta'])
        elapsed = time.time() - t0
        res = run_stat(deltas, f"N={N}")
        res['elapsed_s'] = round(elapsed, 1)
        n_scaling[f"N={N}"] = res
    results['n_scaling_ppwave'] = n_scaling

    print("\n" + "=" * 60)
    print("1b. N-SCALING (Schwarzschild eps=0.005)")
    print("=" * 60)
    n_scaling_sch = {}
    for N in [500, 1000, 2000, 5000]:
        M = max(10, 20 if N <= 2000 else 15)
        deltas = []
        t0 = time.time()
        for trial in range(M):
            seed = 80000 + trial * 1000
            r = crn_trial_gini(seed, N, T, causal_schwarzschild, 0.005)
            deltas.append(r['delta'])
        elapsed = time.time() - t0
        res = run_stat(deltas, f"N={N}")
        res['elapsed_s'] = round(elapsed, 1)
        n_scaling_sch[f"N={N}"] = res
    results['n_scaling_schwarzschild'] = n_scaling_sch

    # ═══════════════════════════════════════════════════════════════
    # 2. HOLDOUT: de Sitter H=0.5
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("2. HOLDOUT: de Sitter H=0.5, N=2000, M=20")
    print("=" * 60)
    deltas_ds = []
    for trial in range(20):
        seed = 90000 + trial * 1000
        r = crn_trial_gini(seed, 2000, T, causal_desitter, 0.5)
        deltas_ds.append(r['delta'])
    results['holdout_desitter'] = run_stat(deltas_ds, "de Sitter H=0.5")

    # ═══════════════════════════════════════════════════════════════
    # 3. GRAPH-THEORY ADVERSARY (Schwarzschild)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("3. GRAPH-THEORY ADVERSARY (Schwarzschild eps=0.005, N=2000)")
    print("=" * 60)
    adv_geo = []
    adv_rng = []
    for trial in range(15):
        seed = 95000 + trial * 1000
        r = adversary_trial(seed, 2000, T, causal_schwarzschild, 0.005)
        adv_geo.append(r['d_geo'])
        adv_rng.append(r['d_rng'])

    geo_arr = np.array(adv_geo)
    rng_arr = np.array(adv_rng)
    print(f"  Geometric: mean={np.mean(geo_arr):+.6f}, sign={'POS' if np.mean(geo_arr) > 0 else 'NEG'}")
    print(f"  Random:    mean={np.mean(rng_arr):+.6f}, sign={'POS' if np.mean(rng_arr) > 0 else 'NEG'}")
    print(f"  SIGN FLIP: {'YES' if np.sign(np.mean(geo_arr)) != np.sign(np.mean(rng_arr)) else 'NO'}")
    results['adversary_schwarzschild'] = {
        'geo_mean': round(float(np.mean(geo_arr)), 6),
        'rng_mean': round(float(np.mean(rng_arr)), 6),
        'sign_flip': bool(np.sign(np.mean(geo_arr)) != np.sign(np.mean(rng_arr))),
        'M': 15
    }

    # Same for ppwave
    print("\n3b. ADVERSARY (ppwave eps=5, N=2000)")
    adv_geo_pp = []
    adv_rng_pp = []
    for trial in range(15):
        seed = 95000 + trial * 1000
        r = adversary_trial(seed, 2000, T, causal_ppwave_quad, 5.0)
        adv_geo_pp.append(r['d_geo'])
        adv_rng_pp.append(r['d_rng'])

    geo_pp = np.array(adv_geo_pp)
    rng_pp = np.array(adv_rng_pp)
    print(f"  Geometric: mean={np.mean(geo_pp):+.6f}")
    print(f"  Random:    mean={np.mean(rng_pp):+.6f}")
    print(f"  SIGN FLIP: {'YES' if np.sign(np.mean(geo_pp)) != np.sign(np.mean(rng_pp)) else 'NO'}")
    results['adversary_ppwave'] = {
        'geo_mean': round(float(np.mean(geo_pp)), 6),
        'rng_mean': round(float(np.mean(rng_pp)), 6),
        'sign_flip': bool(np.sign(np.mean(geo_pp)) != np.sign(np.mean(rng_pp))),
        'M': 15
    }

    # ═══════════════════════════════════════════════════════════════
    # 4. BOUNDARY SWEEP
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("4. BOUNDARY SWEEP (ppwave eps=5, N=2000)")
    print("=" * 60)
    # column_gini_C is a GLOBAL observable (uses all elements)
    # Boundary test: exclude boundary elements and recompute
    # Since Gini uses all past_sizes, "boundary exclusion" means
    # removing elements with smallest/largest past_sizes
    boundary_results = {}
    for bf in [0.05, 0.10, 0.20, 0.30]:
        deltas_b = []
        for trial in range(15):
            seed = 80000 + trial * 1000
            rng = np.random.default_rng(seed)
            pts = sprinkle_4d(2000, T, rng)
            C_flat = causal_flat(pts)
            C_curv = causal_ppwave_quad(pts, 5.0)

            # Exclude boundary
            ps_flat = np.sum(C_flat, axis=0)
            n_exc = int(2000 * bf)
            # Exclude elements with smallest past_sizes (bottom boundary)
            # and largest past_sizes (top boundary)
            order = np.argsort(ps_flat)
            keep = order[n_exc:-n_exc] if n_exc > 0 else order

            g_flat = gini(np.sqrt(ps_flat[keep].astype(np.float64)))
            ps_curv = np.sum(C_curv, axis=0)
            g_curv = gini(np.sqrt(ps_curv[keep].astype(np.float64)))
            deltas_b.append(g_curv - g_flat)
            del C_flat, C_curv; gc.collect()

        arr_b = np.array(deltas_b)
        m = np.mean(arr_b)
        sd = np.std(arr_b, ddof=1)
        d_z = m / sd if sd > 0 else 0
        print(f"  bf={bf:.2f}: d_z={d_z:+.3f}")
        boundary_results[f"bf_{bf}"] = round(d_z, 3)

    results['boundary_sweep_ppwave'] = boundary_results

    # ═══════════════════════════════════════════════════════════════
    # 5. DOSE-RESPONSE
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("5. DOSE-RESPONSE (ppwave, N=2000, M=10)")
    print("=" * 60)
    dose_pp = {}
    for eps in [1.0, 2.0, 5.0, 10.0]:
        deltas_d = []
        for trial in range(10):
            seed = 80000 + trial * 1000
            r = crn_trial_gini(seed, 2000, T, causal_ppwave_quad, eps)
            deltas_d.append(r['delta'])
        res = run_stat(deltas_d, f"eps={eps}")
        dose_pp[f"eps={eps}"] = res
    results['dose_response_ppwave'] = dose_pp

    print("\n5b. DOSE-RESPONSE (Schwarzschild, N=2000, M=10)")
    dose_sch = {}
    for eps in [0.001, 0.002, 0.005, 0.01, 0.02]:
        deltas_d = []
        for trial in range(10):
            seed = 80000 + trial * 1000
            r = crn_trial_gini(seed, 2000, T, causal_schwarzschild, eps)
            deltas_d.append(r['delta'])
        res = run_stat(deltas_d, f"eps={eps}")
        dose_sch[f"eps={eps}"] = res
    results['dose_response_schwarzschild'] = dose_sch

    # ═══════════════════════════════════════════════════════════════
    # 6. RANDOM DAG CONTROL
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("6. RANDOM DAG CONTROL (N=2000, M=15)")
    print("=" * 60)
    rdag_geo = []
    rdag_rng = []
    for trial in range(15):
        seed = 80000 + trial * 1000
        r = random_dag_trial(seed, 2000, T)
        rdag_geo.append(r['gini_geo'])
        rdag_rng.append(r['gini_rng'])
    print(f"  Geometric flat: mean={np.mean(rdag_geo):.4f}")
    print(f"  Random DAG:     mean={np.mean(rdag_rng):.4f}")
    print(f"  d = {(np.mean(rdag_geo) - np.mean(rdag_rng)) / np.std(rdag_rng):+.1f}")
    results['random_dag'] = {
        'geo_mean': round(float(np.mean(rdag_geo)), 4),
        'rng_mean': round(float(np.mean(rdag_rng)), 4),
        'd_separation': round(float((np.mean(rdag_geo) - np.mean(rdag_rng)) / np.std(rdag_rng)), 1)
    }

    # Save
    output_path = os.path.join(RUN_DIR, 'certification_results.json')
    json.dump(results, open(output_path, 'w'), indent=2, default=str)
    print(f"\nAll results saved to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("CERTIFICATION SUMMARY: column_gini_C")
    print("=" * 60)
    pp_dz = [f"d={v['d_z']:+.2f}" for v in n_scaling.values()]
    sch_dz = [f"d={v['d_z']:+.2f}" for v in n_scaling_sch.values()]
    print(f"  N-scaling ppwave:  {' -> '.join(pp_dz)}")
    print(f"  N-scaling Sch:     {' -> '.join(sch_dz)}")
    print(f"  Holdout de Sitter: d_z={results['holdout_desitter']['d_z']:+.3f}")
    print(f"  Adversary Sch:     sign_flip={results['adversary_schwarzschild']['sign_flip']}")
    print(f"  Adversary ppw:     sign_flip={results['adversary_ppwave']['sign_flip']}")
    print(f"  Boundary sweep:    {boundary_results}")
    print(f"  Random DAG:        d={results['random_dag']['d_separation']:+.1f}")


if __name__ == "__main__":
    main()
