"""N=40000 pp-wave exact — standalone Task 3 from E4 closure package.
Interior 80% window. eps={2,3}, 10 seeds, 8 workers.
Estimated: ~5 min/trial, 20 trials / 8 workers = ~13 min total.
"""
import sys, time, json, os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, 'analysis')
from sct_tools.hasse import (sprinkle_diamond, build_hasse_bitset,
                               path_counts)

N_WORKERS = 8

def boundary_slack(pts, T=1.0):
    t = pts[:, 0]
    r = np.sqrt(pts[:, 1]**2 + pts[:, 2]**2 + pts[:, 3]**2)
    return T / 2.0 - (np.abs(t) + r)

def windowed_kurtosis(Y, slack, alpha=0.80):
    threshold = np.quantile(slack, 1.0 - alpha)
    mask = slack >= threshold
    Yw = Y[mask]
    if len(Yw) < 10:
        return 0.0
    X = Yw - Yw.mean()
    s2 = np.var(Yw)
    if s2 < 1e-12:
        return 0.0
    return float(np.mean(X**4) / (s2**2) - 3.0)

def worker(args):
    N, eps, seed = args
    T = 1.0
    t0 = time.time()

    pts = sprinkle_diamond(N, T=T, seed=seed)
    slack = boundary_slack(pts, T)

    # Flat
    p0, c0 = build_hasse_bitset(pts, eps=None)
    pd0, pu0 = path_counts(p0, c0)
    Y0 = np.log2(pd0 * pu0 + 1.0)

    # PP-wave exact
    pe, ce = build_hasse_bitset(pts, eps=eps, exact=True)
    pde, pue = path_counts(pe, ce)
    Ye = np.log2(pde * pue + 1.0)

    # Multiple windows
    results = {}
    for alpha in [0.60, 0.70, 0.80]:
        k0 = windowed_kurtosis(Y0, slack, alpha)
        ke = windowed_kurtosis(Ye, slack, alpha)
        results[f'dk_{alpha:.2f}'] = ke - k0

    # Whole
    k0w = float(np.mean((Y0 - Y0.mean())**4) / np.var(Y0)**2 - 3.0) if np.var(Y0) > 0 else 0.0
    kew = float(np.mean((Ye - Ye.mean())**4) / np.var(Ye)**2 - 3.0) if np.var(Ye) > 0 else 0.0
    results['dk_whole'] = kew - k0w

    elapsed = time.time() - t0
    return {'N': N, 'eps': eps, 'seed': seed, 'elapsed': elapsed, **results}

def main():
    N = 40000
    EPS_VALUES = [2, 3]
    M_SEEDS = 10

    print(f"N=40000 PP-WAVE EXACT — {len(EPS_VALUES)} eps × {M_SEEDS} seeds = {len(EPS_VALUES)*M_SEEDS} trials")
    print(f"Workers: {N_WORKERS}")
    print(f"Estimated: ~5 min/trial → ~{5 * len(EPS_VALUES) * M_SEEDS / N_WORKERS:.0f} min total")
    print()

    tasks = []
    for eps in EPS_VALUES:
        for m in range(M_SEEDS):
            tasks.append((N, eps, 40000000 + eps * 1000 + m))

    t_start = time.time()
    all_results = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(worker, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            all_results.append(r)
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(tasks)}] eps={r['eps']} "
                  f"dk_0.80={r['dk_0.80']:+.5f} dk_0.60={r['dk_0.60']:+.5f} "
                  f"({r['elapsed']:.0f}s) [total {elapsed:.0f}s]", flush=True)

    total_time = time.time() - t_start
    print(f"\nTotal: {total_time:.0f}s = {total_time/60:.1f}min\n")

    # N=20000 reference values (from decisive package and window robustness)
    ref_20k = {
        2: {'0.60': 0.1768, '0.70': 0.1836, '0.80': 0.1021},
        3: {'0.60': 0.1768, '0.70': 0.1836, '0.80': 0.2192},
    }
    # More precise from window_robustness for eps=3
    ref_20k_wr = {3: {'0.60': 0.17681, '0.70': 0.18356, '0.80': 0.21695}}

    for eps in EPS_VALUES:
        trials = [r for r in all_results if r['eps'] == eps]
        print(f"eps={eps} (N=40000, {len(trials)} seeds):")

        for alpha_str in ['0.60', '0.70', '0.80']:
            key = f'dk_{alpha_str}'
            dks = [t[key] for t in trials]
            mn = np.mean(dks)
            se = np.std(dks, ddof=1) / np.sqrt(len(dks))
            d = abs(mn) / se if se > 0 else 0

            # Ratio check
            ref = ref_20k_wr.get(eps, {}).get(alpha_str, ref_20k.get(eps, {}).get(alpha_str, 0))
            ratio = mn / ref if ref > 0 else 0
            alpha_local = np.log2(ratio) if ratio > 0 else 0

            status = "✓ PLATEAU" if ratio < 1.07 else ("⚠ tension" if ratio < 1.15 else "✗ running")
            print(f"  α={alpha_str}: dk={mn:+.5f}±{se:.5f} d={d:.1f} | R(40k/20k)={ratio:.3f} α_loc={alpha_local:.3f} {status}")

        dks_w = [t['dk_whole'] for t in trials]
        mn_w = np.mean(dks_w)
        se_w = np.std(dks_w, ddof=1) / np.sqrt(len(dks_w))
        print(f"  whole: dk={mn_w:+.5f}±{se_w:.5f}")
        print()

    # Save
    outpath = os.path.join('analysis', 'discovery_runs', 'run_001', 'n40k_ppwave.json')
    save_data = {
        'N': N, 'seeds': M_SEEDS, 'total_time_s': total_time,
        'trials': all_results,
    }
    with open(outpath, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved to {outpath}")

if __name__ == '__main__':
    main()
