"""CRN Bin Test: Scenario A' convergence in 2D continuum bins.

Protocol from independent analysis (2026-03-27, §3.2-3.3):

For each seed, compute per-element Y values on flat and pp-wave.
Bin elements by 2D continuum coordinates (u, s):
    u = tau_minus / (tau_minus + tau_plus)   — normalized temporal location
    s = (tau_minus + tau_plus) / T           — proper-time scale

Measure:
    (A) V_N^mean(B) = Var_seeds[mean(Y/lambda | bin B)]          — raw self-averaging
    (B) V_N^Delta(B) = Var_seeds[mean(delta_Y / (eps*lambda) | bin B)]  — CRN-stabilized

Scenario A:  V_mean → 0, V_Delta → 0
Scenario A': V_mean may not → 0, but V_Delta → 0  (SUFFICIENT for Δκ)
Scenario B:  V_mean → const, V_Delta → 0
Scenario C:  V_Delta does not → 0  (BAD)

Also test lambda = N^{1/4} vs N^{1/4}*log(N) per independent analysis suggestion.
"""
import sys, os, time, json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from sct_tools.hasse import (sprinkle_diamond, build_hasse_bitset, path_counts)

N_WORKERS = 8
T = 1.0

def compute_tau(pts, T=1.0):
    """Compute tau_minus and tau_plus for each point in diamond."""
    t = pts[:, 0]
    r = np.sqrt(pts[:, 1]**2 + pts[:, 2]**2 + pts[:, 3]**2)
    # tau_minus = proper time from past tip: sqrt((t + T/2)^2 - r^2)
    arg_m = (t + T/2)**2 - r**2
    tau_m = np.sqrt(np.maximum(arg_m, 0.0))
    # tau_plus = proper time to future tip: sqrt((T/2 - t)^2 - r^2)
    arg_p = (T/2 - t)**2 - r**2
    tau_p = np.sqrt(np.maximum(arg_p, 0.0))
    return tau_m, tau_p

def worker(args):
    """Single CRN trial: return per-element Y values for flat and pp-wave."""
    N, eps, seed = args
    t0 = time.time()

    pts = sprinkle_diamond(N, T=T, seed=seed)

    # Boundary slack for interior window
    t_arr = pts[:, 0]
    r_arr = np.sqrt(pts[:, 1]**2 + pts[:, 2]**2 + pts[:, 3]**2)
    slack = T/2 - (np.abs(t_arr) + r_arr)

    # Continuum coordinates
    tau_m, tau_p = compute_tau(pts, T)
    tau_sum = tau_m + tau_p
    # u = tau_m / tau_sum (0 = past tip, 1 = future tip)
    u = np.where(tau_sum > 1e-12, tau_m / tau_sum, 0.5)
    # s = tau_sum / T (0 = boundary, ~1 = deep interior)
    s = tau_sum / T

    # Flat Hasse + path counts
    p0, c0 = build_hasse_bitset(pts, eps=None)
    pd0, pu0 = path_counts(p0, c0)
    Y0 = np.log2(pd0 * pu0 + 1.0)

    # PP-wave exact Hasse + path counts
    pe, ce = build_hasse_bitset(pts, eps=eps, exact=True)
    pde, pue = path_counts(pe, ce)
    Ye = np.log2(pde * pue + 1.0)

    elapsed = time.time() - t0
    return {
        'N': N, 'eps': eps, 'seed': seed, 'elapsed': elapsed,
        'Y0': Y0.tolist(),
        'Ye': Ye.tolist(),
        'u': u.tolist(),
        's': s.tolist(),
        'slack': slack.tolist(),
    }

def analyze_bins(all_trials, N, eps, n_ubins=5, n_sbins=5, alpha_win=0.70):
    """Analyze convergence in 2D continuum bins."""
    lambda_N = N**(1/4)
    lambda_N_log = N**(1/4) * np.log(N)

    # Collect per-seed bin data
    seeds = [t for t in all_trials if t['N'] == N and t['eps'] == eps]
    M = len(seeds)

    if M < 5:
        return None

    # Use first seed to define bin edges (same geometry for all)
    u_all = np.concatenate([np.array(t['u']) for t in seeds])
    s_all = np.concatenate([np.array(t['s']) for t in seeds])
    slack_all = np.concatenate([np.array(t['slack']) for t in seeds])

    # Interior mask: top alpha_win fraction by slack
    slack_thresh = np.quantile(slack_all, 1.0 - alpha_win)

    # Bin edges in (u, s) space, only for interior points
    int_mask_all = slack_all >= slack_thresh
    u_int = u_all[int_mask_all]
    s_int = s_all[int_mask_all]

    u_edges = np.linspace(np.percentile(u_int, 2), np.percentile(u_int, 98), n_ubins + 1)
    s_edges = np.linspace(np.percentile(s_int, 2), np.percentile(s_int, 98), n_sbins + 1)

    bin_results = {}

    for iu in range(n_ubins):
        for js in range(n_sbins):
            bin_key = f'u{iu}_s{js}'
            raw_means = []      # mean(Y0 / lambda) per seed
            raw_means_log = []  # mean(Y0 / lambda_log) per seed
            delta_means = []    # mean(deltaY / (eps * lambda)) per seed
            delta_means_log = []
            fourth_moments = [] # mean((Y0 - mean)^4 / lambda^4) per seed
            bin_sizes = []

            for t in seeds:
                u_arr = np.array(t['u'])
                s_arr = np.array(t['s'])
                slack_arr = np.array(t['slack'])
                Y0 = np.array(t['Y0'])
                Ye = np.array(t['Ye'])

                # Interior + bin mask
                int_mask = slack_arr >= slack_thresh
                bin_mask = (int_mask &
                           (u_arr >= u_edges[iu]) & (u_arr < u_edges[iu+1]) &
                           (s_arr >= s_edges[js]) & (s_arr < s_edges[js+1]))

                n_in_bin = np.sum(bin_mask)
                if n_in_bin < 20:
                    continue

                Y0_bin = Y0[bin_mask]
                Ye_bin = Ye[bin_mask]
                dY_bin = Ye_bin - Y0_bin

                # Raw self-averaging: mean(Y0 / lambda)
                raw_means.append(float(np.mean(Y0_bin / lambda_N)))
                raw_means_log.append(float(np.mean(Y0_bin / lambda_N_log)))

                # CRN-stabilized: mean(deltaY / (eps * lambda))
                delta_means.append(float(np.mean(dY_bin / (eps * lambda_N))))
                delta_means_log.append(float(np.mean(dY_bin / (eps * lambda_N_log))))

                # Fourth moment of normalized raw field
                Y0_centered = Y0_bin - np.mean(Y0_bin)
                if np.var(Y0_bin) > 1e-12:
                    m4 = float(np.mean(Y0_centered**4) / lambda_N**4)
                else:
                    m4 = 0.0
                fourth_moments.append(m4)
                bin_sizes.append(int(n_in_bin))

            if len(raw_means) < 5:
                continue

            raw_means = np.array(raw_means)
            raw_means_log = np.array(raw_means_log)
            delta_means = np.array(delta_means)
            delta_means_log = np.array(delta_means_log)
            fourth_moments = np.array(fourth_moments)

            bin_results[bin_key] = {
                'u_range': [float(u_edges[iu]), float(u_edges[iu+1])],
                's_range': [float(s_edges[js]), float(s_edges[js+1])],
                'n_seeds': len(raw_means),
                'median_bin_size': int(np.median(bin_sizes)),
                # Raw self-averaging (V_N^mean)
                'V_mean': float(np.var(raw_means, ddof=1)),
                'V_mean_log': float(np.var(raw_means_log, ddof=1)),
                'mean_of_means': float(np.mean(raw_means)),
                # CRN-stabilized (V_N^Delta) — THE KEY TEST
                'V_delta': float(np.var(delta_means, ddof=1)),
                'V_delta_log': float(np.var(delta_means_log, ddof=1)),
                'mean_of_deltas': float(np.mean(delta_means)),
                # Fourth moment stability
                'V_m4': float(np.var(fourth_moments, ddof=1)),
                'mean_m4': float(np.mean(fourth_moments)),
            }

    return bin_results

def main():
    NS = [5000, 10000, 20000]
    EPS = 3
    M_SEEDS = 25
    N_UBINS, N_SBINS = 4, 4
    ALPHA_WIN = 0.70  # Use 70% window (shown to plateau better)

    print("="*70)
    print(f"CRN BIN TEST: Scenario A' convergence")
    print(f"N = {NS}, eps = {EPS}, seeds = {M_SEEDS}")
    print(f"Bins: {N_UBINS}×{N_SBINS} in (u, s), window α={ALPHA_WIN}")
    print(f"Workers: {N_WORKERS}")
    print("="*70)

    tasks = []
    for N in NS:
        for m in range(M_SEEDS):
            tasks.append((N, EPS, 7770000 + N + m))

    print(f"Total trials: {len(tasks)}")
    t_start = time.time()

    all_trials = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(worker, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures)):
            r = fut.result()
            # Don't store full Y arrays in memory — analyze on the fly
            all_trials.append(r)
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t_start
                print(f"  [{i+1}/{len(tasks)}] N={r['N']} ({r['elapsed']:.1f}s) [total {elapsed:.0f}s]",
                      flush=True)

    total_time = time.time() - t_start
    print(f"\nTotal computation: {total_time:.0f}s = {total_time/60:.1f}min\n")

    # Analyze each N
    results = {}
    for N in NS:
        print(f"\n{'='*70}")
        print(f"N = {N}")
        print(f"{'='*70}")

        bins = analyze_bins(all_trials, N, EPS, N_UBINS, N_SBINS, ALPHA_WIN)
        if bins is None:
            print("  Not enough seeds")
            continue

        results[str(N)] = bins

        # Summary statistics
        V_means = [b['V_mean'] for b in bins.values()]
        V_deltas = [b['V_delta'] for b in bins.values()]
        V_m4s = [b['V_m4'] for b in bins.values()]

        print(f"  Bins with data: {len(bins)}")
        print(f"  V_mean  (raw self-averaging):  median={np.median(V_means):.6f}  max={np.max(V_means):.6f}")
        print(f"  V_delta (CRN-stabilized):      median={np.median(V_deltas):.6f}  max={np.max(V_deltas):.6f}")
        print(f"  V_m4    (4th moment stability): median={np.median(V_m4s):.6f}  max={np.max(V_m4s):.6f}")

        print(f"\n  Per-bin details:")
        print(f"  {'Bin':>8s}  {'u_mid':>6s}  {'s_mid':>6s}  {'n_pts':>5s}  {'V_mean':>10s}  {'V_delta':>10s}  {'mean_δ':>10s}")
        for bk in sorted(bins.keys()):
            b = bins[bk]
            u_mid = (b['u_range'][0] + b['u_range'][1]) / 2
            s_mid = (b['s_range'][0] + b['s_range'][1]) / 2
            print(f"  {bk:>8s}  {u_mid:6.3f}  {s_mid:6.3f}  {b['median_bin_size']:5d}  "
                  f"{b['V_mean']:10.6f}  {b['V_delta']:10.6f}  {b['mean_of_deltas']:+10.6f}")

    # Cross-N convergence check
    print(f"\n{'='*70}")
    print("CONVERGENCE CHECK: V_delta across N")
    print(f"{'='*70}")

    common_bins = set.intersection(*[set(results[str(N)].keys()) for N in NS if str(N) in results])
    if common_bins:
        print(f"  Common bins: {len(common_bins)}")
        print(f"\n  {'Bin':>8s}", end='')
        for N in NS:
            print(f"  {'V_d('+str(N//1000)+'k)':>12s}", end='')
        print(f"  {'Ratio':>8s}  {'Verdict':>10s}")

        verdicts = []
        for bk in sorted(common_bins):
            print(f"  {bk:>8s}", end='')
            vds = []
            for N in NS:
                vd = results[str(N)][bk]['V_delta']
                vds.append(vd)
                print(f"  {vd:12.6f}", end='')

            if len(vds) >= 2 and vds[-2] > 0:
                ratio = vds[-1] / vds[-2]
                verdict = "A'" if ratio < 0.8 else ("stable" if ratio < 1.2 else "BAD")
                verdicts.append(verdict)
                print(f"  {ratio:8.3f}  {verdict:>10s}")
            else:
                print(f"  {'N/A':>8s}  {'N/A':>10s}")

        n_good = sum(1 for v in verdicts if v in ["A'", "stable"])
        n_bad = sum(1 for v in verdicts if v == "BAD")
        print(f"\n  Summary: {n_good}/{len(verdicts)} bins converging, {n_bad} diverging")

        if n_bad == 0:
            print(f"\n  *** SCENARIO A' SUPPORTED: CRN response field converges in all bins ***")
        elif n_bad <= 2:
            print(f"\n  *** SCENARIO A' MOSTLY SUPPORTED: {n_bad} bins need investigation ***")
        else:
            print(f"\n  *** WARNING: {n_bad} bins show non-convergence ***")

    # Save (without per-element Y — too large)
    save_data = {
        'config': {
            'NS': NS, 'eps': EPS, 'M_seeds': M_SEEDS,
            'n_ubins': N_UBINS, 'n_sbins': N_SBINS,
            'alpha_win': ALPHA_WIN,
        },
        'results': results,
        'total_time_s': total_time,
    }

    outpath = os.path.join(os.path.dirname(__file__),
                           '..', 'discovery_runs', 'run_001', 'crn_bin_test.json')
    # Save summary only (no per-element data)
    with open(outpath, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to {outpath}")

if __name__ == '__main__':
    main()
