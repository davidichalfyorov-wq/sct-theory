"""Window robustness test: does the plateau depend on window fraction alpha?

Tests alpha = {0.60, 0.70, 0.80, 0.90, 1.00} for pp-wave exact at N=5000,10000.
Fast: N=5000 ~4s/trial, 10 seeds = 40s per (N,eps) combo.

This is the "most dangerous untested failure mode" per independent analysis adversarial review.
If plateau only exists at alpha=0.80, the continuum claim is fragile.
"""
import sys, os, time, json
import numpy as np
from scipy.stats import kurtosis as sp_kurtosis

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sct_tools.hasse import (sprinkle_diamond, build_hasse_bitset,
                               path_counts)

def boundary_slack_diamond(pts, T=1.0):
    """Compute boundary slack = T/2 - (|t| + r) for each point."""
    t = pts[:, 0]
    r = np.sqrt(pts[:, 1]**2 + pts[:, 2]**2 + pts[:, 3]**2)
    return T / 2.0 - (np.abs(t) + r)

def windowed_kurtosis(Y, slack, alpha):
    """Compute excess kurtosis on top alpha-fraction by slack."""
    if alpha >= 1.0:
        X = Y - Y.mean()
        s2 = np.var(Y)
        if s2 < 1e-12:
            return 0.0
        return float(np.mean(X**4) / (s2**2) - 3.0)

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

def crn_trial_windowed(N, eps, seed, alphas, T=1.0):
    """CRN trial returning dk for each window fraction alpha."""
    pts = sprinkle_diamond(N, T=T, seed=seed)
    slack = boundary_slack_diamond(pts, T)

    # Flat
    p0, c0 = build_hasse_bitset(pts, eps=None)
    pd0, pu0 = path_counts(p0, c0)
    Y0 = np.log2(pd0 * pu0 + 1.0)

    # PP-wave exact
    pe, ce = build_hasse_bitset(pts, eps=eps, exact=True)
    pde, pue = path_counts(pe, ce)
    Ye = np.log2(pde * pue + 1.0)

    results = {}
    for a in alphas:
        k0 = windowed_kurtosis(Y0, slack, a)
        ke = windowed_kurtosis(Ye, slack, a)
        results[f'{a:.2f}'] = ke - k0

    return results

def main():
    alphas = [0.60, 0.70, 0.80, 0.90, 1.00]
    configs = [
        (5000, 3, 15),
        (10000, 3, 12),
        (20000, 3, 8),
    ]

    all_results = {}

    for N, eps, M_seeds in configs:
        key = f'N{N}_eps{eps}'
        print(f'\n{"="*60}')
        print(f'N={N}, eps={eps}, M_seeds={M_seeds}')
        print(f'{"="*60}')

        # Collect per-seed results
        seed_data = {f'{a:.2f}': [] for a in alphas}

        for s in range(M_seeds):
            t0 = time.time()
            res = crn_trial_windowed(N, eps, s + 1000, alphas)
            dt = time.time() - t0

            for a_str, dk in res.items():
                seed_data[a_str].append(dk)

            if (s + 1) % 3 == 0:
                print(f'  [{s+1}/{M_seeds}] {dt:.1f}s | ' +
                      ' | '.join(f'a={a:.2f}: {res[f"{a:.2f}"]:+.4f}' for a in alphas))

        # Compute stats
        stats = {}
        for a in alphas:
            a_str = f'{a:.2f}'
            dks = np.array(seed_data[a_str])
            mean = float(np.mean(dks))
            se = float(np.std(dks, ddof=1) / np.sqrt(len(dks)))
            d = abs(mean) / se if se > 0 else 0
            stats[a_str] = {'mean': mean, 'se': se, 'd': d, 'n': len(dks)}

        all_results[key] = stats

        print(f'\nResults for N={N}, eps={eps}:')
        print(f'  {"alpha":>6s}  {"dk_mean":>10s}  {"SE":>8s}  {"d":>6s}')
        for a in alphas:
            a_str = f'{a:.2f}'
            s = stats[a_str]
            print(f'  {a_str:>6s}  {s["mean"]:+10.5f}  {s["se"]:8.5f}  {s["d"]:6.1f}')

    # Cross-N ratios for each alpha (main test)
    print(f'\n{"="*60}')
    print('WINDOW ROBUSTNESS: ratios across N for eps=3')
    print(f'{"="*60}')

    ns = [cfg[0] for cfg in configs]
    for a in alphas:
        a_str = f'{a:.2f}'
        vals = []
        for N in ns:
            key = f'N{N}_eps3'
            if key in all_results and a_str in all_results[key]:
                vals.append(all_results[key][a_str]['mean'])
            else:
                vals.append(None)

        ratios = []
        for i in range(1, len(vals)):
            if vals[i] is not None and vals[i-1] is not None and vals[i-1] != 0:
                ratios.append(vals[i] / vals[i-1])
            else:
                ratios.append(None)

        r_strs = [f'{r:.3f}' if r else 'N/A' for r in ratios]
        v_strs = [f'{v:+.5f}' if v else 'N/A' for v in vals]
        print(f'  alpha={a_str}: dk = [{", ".join(v_strs)}]  ratios = [{", ".join(r_strs)}]')

    # Save
    outpath = os.path.join(os.path.dirname(__file__),
                           '..', 'discovery_runs', 'run_001', 'window_robustness.json')
    with open(outpath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\nSaved to {outpath}')

if __name__ == '__main__':
    main()
