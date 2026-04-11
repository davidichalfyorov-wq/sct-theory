"""
E4 CLOSURE PACKAGE: Three computations to close continuum limit gaps
====================================================================

Based on independent analysis analytical derivation (2026-03-27).

Task 1: nu_N measurement (eq 6.1 from independent analysis)
    nu_N = <delta_Y * f>_W / (eps * lambda_N * <f^2>_W)
    where f(x,y) = (x^2 - y^2)/2, lambda_N ~ N^{1/4}
    On interior 80% window. N = {5000, 10000, 20000}, eps = {2, 3}.
    If nu_N stabilizes -> A_inf formula becomes a prediction.

Task 2: Per-N twosided Schwarzschild mass fits
    Full mass grid M = {0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02}
    at N = {10000, 20000} with twosided 80% window.
    Fit alpha(N), beta(N) separately at each N.
    Explains non-monotonic dk(N) at fixed M.

Task 3: N=40000 pp-wave exact on interior window
    One large N point to nail the plateau.
    eps = {2, 3}, M_seeds = 12, outer_trim 80%.
    Compare with N=20000 ratio -> should be ~1.04 if plateau.

Config: 8 parallel workers, RTX 3090 Ti not needed (CPU-bound Hasse).
"""
import sys, time, json, os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, 'analysis')
from sct_tools.hasse import (
    sprinkle_diamond, sprinkle_shell, build_hasse_bitset,
    path_counts,
)

T = 1.0
R_MIN = 0.10
INTERIOR_FRAC = 0.80
N_WORKERS = 8

# ─── UTILITY FUNCTIONS ──────────────────────────────────────────

def outer_slack(pts, T=1.0):
    r = np.sqrt(np.sum(pts[:, 1:]**2, axis=1))
    return T / 2 - (np.abs(pts[:, 0]) + r)

def inner_slack(pts, r_min=0.10):
    r = np.sqrt(np.sum(pts[:, 1:]**2, axis=1))
    return r - r_min

def twosided_slack(pts, T=1.0, r_min=0.10):
    return np.minimum(outer_slack(pts, T), inner_slack(pts, r_min))

def kurtosis_on_subset(Y, mask=None):
    if mask is not None:
        Y = Y[mask]
    if len(Y) < 4:
        return 0.0
    X = Y - Y.mean()
    s2 = np.var(Y)
    if s2 < 1e-12:
        return 0.0
    return float(np.mean(X ** 4) / (s2 * s2) - 3.0)

def get_interior_mask(pts, T=1.0, frac=0.80):
    """Interior 80% mask for diamond (outer_trim only)."""
    os_ = outer_slack(pts, T)
    thr = np.quantile(os_, 1.0 - frac)
    return os_ >= thr

def get_twosided_mask(pts, T=1.0, r_min=0.10, frac=0.80):
    """Twosided 80% mask for shell."""
    ts_ = twosided_slack(pts, T, r_min)
    thr = np.quantile(ts_, 1.0 - frac)
    return ts_ >= thr

def summarize(arr):
    a = np.array(arr)
    mn = a.mean()
    se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    d = mn / se if se > 1e-15 else 0.0
    return float(mn), float(se), float(d)


# ═════════════════════════════════════════════════════════════════
# TASK 1: nu_N MEASUREMENT
# ═════════════════════════════════════════════════════════════════

def nu_trial(N, eps, seed):
    """Compute nu_N for one seed.

    nu_N = <delta_Y * f>_W / (eps * lambda_N * <f^2>_W)

    where:
      delta_Y = Y_ppw - Y_flat (per-element)
      f(x,y) = (x^2 - y^2)/2 (pp-wave profile)
      lambda_N = N^{1/4} (longest chain scale)
      <...>_W = average over interior 80% window
    """
    t0 = time.time()
    pts = sprinkle_diamond(N, T=T, seed=seed)

    # Build flat and pp-wave Hasse diagrams
    p0, c0 = build_hasse_bitset(pts, eps=None)
    pE, cE = build_hasse_bitset(pts, eps=eps)

    # Path counts
    pd0, pu0 = path_counts(p0, c0)
    pdE, puE = path_counts(pE, cE)

    Y0 = np.log2(pd0 * pu0 + 1.0)
    YE = np.log2(pdE * puE + 1.0)
    delta_Y = YE - Y0

    # Interior 80% mask
    mask = get_interior_mask(pts, T, INTERIOR_FRAC)

    # pp-wave profile f(x,y) = (x^2 - y^2)/2
    x = pts[:, 1]
    y = pts[:, 2]
    f = (x**2 - y**2) / 2.0

    # Compute nu_N on window
    dY_w = delta_Y[mask]
    f_w = f[mask]

    lambda_N = N**(1.0/4.0)

    # <delta_Y * f>_W and <f^2>_W
    dYf_mean = np.mean(dY_w * f_w)
    f2_mean = np.mean(f_w**2)

    nu = dYf_mean / (eps * lambda_N * f2_mean)

    # Also compute kurtosis readouts for cross-check
    k0_int = kurtosis_on_subset(Y0, mask)
    kE_int = kurtosis_on_subset(YE, mask)
    dk_int = kE_int - k0_int

    k0_whole = kurtosis_on_subset(Y0)
    kE_whole = kurtosis_on_subset(YE)
    dk_whole = kE_whole - k0_whole

    elapsed = time.time() - t0
    return {
        'N': N, 'eps': eps, 'seed': seed,
        'nu': float(nu),
        'dYf_mean': float(dYf_mean),
        'f2_mean': float(f2_mean),
        'lambda_N': float(lambda_N),
        'dk_int': float(dk_int),
        'dk_whole': float(dk_whole),
        'n_window': int(mask.sum()),
        'elapsed': float(elapsed),
    }

def _nu_worker(args):
    N, eps, seed = args
    return nu_trial(N, eps, seed)

def run_task1():
    """Task 1: nu_N measurement across N and eps."""
    N_VALUES = [5000, 10000, 20000]
    EPS_VALUES = [2, 3]
    M_SEEDS = 15

    print("=" * 80, flush=True)
    print("TASK 1: nu_N MEASUREMENT (analytical eq 6.1)", flush=True)
    print(f"N = {N_VALUES}, eps = {EPS_VALUES}, seeds = {M_SEEDS}", flush=True)
    print("=" * 80, flush=True)

    tasks = []
    for N in N_VALUES:
        for eps in EPS_VALUES:
            for m in range(M_SEEDS):
                seed = 7000000 + N + int(eps * 1000) + m
                tasks.append((N, eps, seed))

    print(f"Total trials: {len(tasks)}", flush=True)

    results_raw = {}
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_nu_worker, t): t for t in tasks}
        done_count = 0
        for fut in as_completed(futures):
            r = fut.result()
            key = f"{r['N']}_{r['eps']}"
            if key not in results_raw:
                results_raw[key] = []
            results_raw[key].append(r)
            done_count += 1
            if done_count % 10 == 0 or done_count == len(tasks):
                elapsed = time.time() - t_start
                print(f"  [{done_count}/{len(tasks)}] N={r['N']} eps={r['eps']} "
                      f"nu={r['nu']:.4f} dk_int={r['dk_int']:+.5f} "
                      f"({r['elapsed']:.1f}s) [total {elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t_start
    print(f"\nTask 1 total: {elapsed:.0f}s = {elapsed/60:.1f}min", flush=True)

    # Summary
    print(f"\n{'='*80}", flush=True)
    print("RESULTS: nu_N by N and eps", flush=True)
    print("=" * 80, flush=True)

    C2 = T**4 / 1120.0

    results = {}
    for N in N_VALUES:
        for eps in EPS_VALUES:
            key = f"{N}_{eps}"
            trials = results_raw[key]
            nus = [t['nu'] for t in trials]
            dks = [t['dk_int'] for t in trials]
            dk_wholes = [t['dk_whole'] for t in trials]

            nu_mean, nu_se, _ = summarize(nus)
            dk_mean, dk_se, dk_d = summarize(dks)
            dkw_mean, dkw_se, dkw_d = summarize(dk_wholes)

            # Predicted A_inf from nu
            # K_{2,alpha} ~ nu^2 * integral_factor
            # For now just report nu

            results[key] = {
                'N': N, 'eps': eps,
                'nu_mean': nu_mean, 'nu_se': nu_se,
                'dk_int_mean': dk_mean, 'dk_int_se': dk_se, 'dk_int_d': dk_d,
                'dk_whole_mean': dkw_mean, 'dk_whole_se': dkw_se, 'dk_whole_d': dkw_d,
                'nus': nus, 'dks_int': dks, 'dks_whole': dk_wholes,
                'K2_empirical': dk_mean / eps**2,  # K_{2,alpha} = Dkappa / eps^2
            }

            print(f"  N={N:5d} eps={eps}: nu = {nu_mean:.4f} +/- {nu_se:.4f} | "
                  f"dk_int = {dk_mean:+.5f} +/- {dk_se:.5f} (d={dk_d:.1f}) | "
                  f"K2 = {dk_mean/eps**2:.5f}", flush=True)

    # Check convergence of nu
    print(f"\nNU CONVERGENCE CHECK:", flush=True)
    for eps in EPS_VALUES:
        print(f"  eps={eps}:", flush=True)
        for N in N_VALUES:
            key = f"{N}_{eps}"
            r = results[key]
            print(f"    N={N:5d}: nu = {r['nu_mean']:.4f} +/- {r['nu_se']:.4f}", flush=True)

    # K2 convergence (should stabilize = continuum limit)
    print(f"\nK2 = Dkappa/eps^2 CONVERGENCE (continuum limit test):", flush=True)
    for eps in EPS_VALUES:
        print(f"  eps={eps}:", flush=True)
        prev_K2 = None
        for N in N_VALUES:
            key = f"{N}_{eps}"
            K2 = results[key]['K2_empirical']
            ratio = K2 / prev_K2 if prev_K2 is not None else float('nan')
            print(f"    N={N:5d}: K2 = {K2:.5f}" +
                  (f"  ratio = {ratio:.3f}" if prev_K2 is not None else ""),
                  flush=True)
            prev_K2 = K2

    return results


# ═════════════════════════════════════════════════════════════════
# TASK 2: PER-N TWOSIDED SCHWARZSCHILD MASS FITS
# ═════════════════════════════════════════════════════════════════

def sch_twosided_trial(N, M_sch, seed):
    """One Schwarzschild CRN trial with twosided window readout."""
    t0 = time.time()
    pts = sprinkle_shell(N, T=T, r_min=R_MIN, seed=seed)

    p0, c0 = build_hasse_bitset(pts, eps=None)
    pS, cS = build_hasse_bitset(pts, M_sch=M_sch)

    pd0, pu0 = path_counts(p0, c0)
    pdS, puS = path_counts(pS, cS)

    Y0 = np.log2(pd0 * pu0 + 1.0)
    YS = np.log2(pdS * puS + 1.0)

    # Twosided mask
    mask_ts = get_twosided_mask(pts, T, R_MIN, INTERIOR_FRAC)

    dk_whole = kurtosis_on_subset(YS) - kurtosis_on_subset(Y0)
    dk_twosided = kurtosis_on_subset(YS, mask_ts) - kurtosis_on_subset(Y0, mask_ts)

    elapsed = time.time() - t0
    return {
        'N': N, 'M_sch': M_sch, 'seed': seed,
        'dk_whole': float(dk_whole),
        'dk_twosided': float(dk_twosided),
        'elapsed': float(elapsed),
    }

def _sch_worker(args):
    N, M_sch, seed = args
    return sch_twosided_trial(N, M_sch, seed)

def run_task2():
    """Task 2: Per-N twosided mass fits at N=10000, 20000."""
    N_VALUES = [10000, 20000]
    MASSES = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]
    M_SEEDS = 15

    print("\n" + "=" * 80, flush=True)
    print("TASK 2: PER-N TWOSIDED SCHWARZSCHILD MASS FITS", flush=True)
    print(f"N = {N_VALUES}, M = {MASSES}, seeds = {M_SEEDS}", flush=True)
    print("=" * 80, flush=True)

    tasks = []
    for N in N_VALUES:
        for M_sch in MASSES:
            for m in range(M_SEEDS):
                seed = 8000000 + N + int(M_sch * 100000) + m
                tasks.append((N, M_sch, seed))

    print(f"Total trials: {len(tasks)}", flush=True)

    results_raw = {}
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_sch_worker, t): t for t in tasks}
        done_count = 0
        for fut in as_completed(futures):
            r = fut.result()
            key = f"{r['N']}_{r['M_sch']}"
            if key not in results_raw:
                results_raw[key] = []
            results_raw[key].append(r)
            done_count += 1
            if done_count % 10 == 0 or done_count == len(tasks):
                elapsed = time.time() - t_start
                print(f"  [{done_count}/{len(tasks)}] N={r['N']} M={r['M_sch']} "
                      f"dk_2s={r['dk_twosided']:+.5f} dk_w={r['dk_whole']:+.5f} "
                      f"({r['elapsed']:.1f}s) [total {elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t_start
    print(f"\nTask 2 total: {elapsed:.0f}s = {elapsed/60:.1f}min", flush=True)

    # Summary + WLS fits per N
    from scipy.optimize import curve_fit

    print(f"\n{'='*80}", flush=True)
    print("RESULTS: Twosided dk by N and M", flush=True)
    print("=" * 80, flush=True)

    results = {}
    for N in N_VALUES:
        results[N] = {}
        print(f"\n  N = {N}:", flush=True)

        M_arr = []
        dk_means = []
        dk_ses = []

        for M_sch in MASSES:
            key = f"{N}_{M_sch}"
            trials = results_raw[key]
            dks = [t['dk_twosided'] for t in trials]
            dks_w = [t['dk_whole'] for t in trials]

            mn, se, d = summarize(dks)
            mnw, sew, dw = summarize(dks_w)

            results[N][M_sch] = {
                'dk_twosided_mean': mn, 'dk_twosided_se': se, 'dk_twosided_d': d,
                'dk_whole_mean': mnw, 'dk_whole_se': sew, 'dk_whole_d': dw,
                'dks_twosided': dks, 'dks_whole': dks_w,
            }

            M_arr.append(M_sch)
            dk_means.append(mn)
            dk_ses.append(se)

            print(f"    M={M_sch:.4f}: dk_2s = {mn:+.5f} +/- {se:.5f} (d={d:.1f}) | "
                  f"dk_w = {mnw:+.5f} +/- {sew:.5f}", flush=True)

        # WLS fit: dk = alpha*M + beta*M^2
        M_arr = np.array(M_arr)
        dk_means = np.array(dk_means)
        dk_ses = np.array(dk_ses)

        weights = 1.0 / (dk_ses**2 + 1e-20)

        def linear(M, alpha):
            return alpha * M

        def quadratic(M, alpha, beta):
            return alpha * M + beta * M**2

        try:
            popt_lin, pcov_lin = curve_fit(linear, M_arr, dk_means, sigma=dk_ses, absolute_sigma=True)
            resid_lin = dk_means - linear(M_arr, *popt_lin)
            chi2_lin = np.sum((resid_lin / dk_ses)**2)

            popt_q, pcov_q = curve_fit(quadratic, M_arr, dk_means, sigma=dk_ses, absolute_sigma=True)
            perr_q = np.sqrt(np.diag(pcov_q))
            resid_q = dk_means - quadratic(M_arr, *popt_q)
            chi2_q = np.sum((resid_q / dk_ses)**2)

            ndof_lin = len(M_arr) - 1
            ndof_q = len(M_arr) - 2

            results[N]['fit'] = {
                'alpha': float(popt_q[0]), 'alpha_se': float(perr_q[0]),
                'beta': float(popt_q[1]), 'beta_se': float(perr_q[1]),
                'chi2_lin': float(chi2_lin), 'ndof_lin': ndof_lin,
                'chi2_quad': float(chi2_q), 'ndof_quad': ndof_q,
                'delta_chi2': float(chi2_lin - chi2_q),
            }

            print(f"\n    WLS fit dk = alpha*M + beta*M^2:", flush=True)
            print(f"      alpha = {popt_q[0]:+.2f} +/- {perr_q[0]:.2f}", flush=True)
            print(f"      beta  = {popt_q[1]:+.1f} +/- {perr_q[1]:.1f}", flush=True)
            print(f"      chi2/ndof: linear = {chi2_lin/ndof_lin:.2f}, "
                  f"quadratic = {chi2_q/ndof_q:.2f}, "
                  f"delta_chi2 = {chi2_lin - chi2_q:.1f}", flush=True)

            # Turnover
            if popt_q[1] != 0:
                M_star = -popt_q[0] / popt_q[1]
                print(f"      M* = -alpha/beta = {M_star:.4f}", flush=True)
        except Exception as e:
            print(f"    FIT FAILED: {e}", flush=True)

    # Compare alpha(N), beta(N) across N
    print(f"\n{'='*80}", flush=True)
    print("ALPHA(N), BETA(N) CONVERGENCE", flush=True)
    print("=" * 80, flush=True)

    # Include N=5000 from previous run
    print("  (N=5000 from sch_twosided_window.py: alpha=+15.53, beta=-810.7)", flush=True)
    for N in N_VALUES:
        if 'fit' in results[N]:
            f = results[N]['fit']
            print(f"  N={N:5d}: alpha = {f['alpha']:+.2f} +/- {f['alpha_se']:.2f}, "
                  f"beta = {f['beta']:+.1f} +/- {f['beta_se']:.1f}", flush=True)

    return results


# ═════════════════════════════════════════════════════════════════
# TASK 3: N=40000 PP-WAVE EXACT
# ═════════════════════════════════════════════════════════════════

def ppwave_40k_trial(N, eps, seed):
    """One exact pp-wave CRN trial with interior 80% readout."""
    t0 = time.time()
    pts = sprinkle_diamond(N, T=T, seed=seed)

    p0, c0 = build_hasse_bitset(pts, eps=None)
    pE, cE = build_hasse_bitset(pts, eps=eps)

    pd0, pu0 = path_counts(p0, c0)
    pdE, puE = path_counts(pE, cE)

    Y0 = np.log2(pd0 * pu0 + 1.0)
    YE = np.log2(pdE * puE + 1.0)

    mask = get_interior_mask(pts, T, INTERIOR_FRAC)

    dk_int = kurtosis_on_subset(YE, mask) - kurtosis_on_subset(Y0, mask)
    dk_whole = kurtosis_on_subset(YE) - kurtosis_on_subset(Y0)

    elapsed = time.time() - t0
    return {
        'N': N, 'eps': eps, 'seed': seed,
        'dk_int': float(dk_int),
        'dk_whole': float(dk_whole),
        'elapsed': float(elapsed),
    }

def _ppw_worker(args):
    N, eps, seed = args
    return ppwave_40k_trial(N, eps, seed)

def run_task3():
    """Task 3: N=40000 pp-wave exact with interior window."""
    N = 40000
    EPS_VALUES = [2, 3]
    M_SEEDS = 12  # fewer seeds due to high cost

    print("\n" + "=" * 80, flush=True)
    print(f"TASK 3: N={N} PP-WAVE EXACT (interior 80%)", flush=True)
    print(f"eps = {EPS_VALUES}, seeds = {M_SEEDS}", flush=True)
    print(f"Estimated time: ~120s/trial × {len(EPS_VALUES) * M_SEEDS} = "
          f"~{120 * len(EPS_VALUES) * M_SEEDS / 60:.0f} min (parallelized)", flush=True)
    print("=" * 80, flush=True)

    tasks = []
    for eps in EPS_VALUES:
        for m in range(M_SEEDS):
            seed = 9000000 + N + int(eps * 1000) + m
            tasks.append((N, eps, seed))

    print(f"Total trials: {len(tasks)}", flush=True)

    results_raw = {}
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_ppw_worker, t): t for t in tasks}
        done_count = 0
        for fut in as_completed(futures):
            r = fut.result()
            key = f"{r['eps']}"
            if key not in results_raw:
                results_raw[key] = []
            results_raw[key].append(r)
            done_count += 1
            elapsed = time.time() - t_start
            print(f"  [{done_count}/{len(tasks)}] eps={r['eps']} "
                  f"dk_int={r['dk_int']:+.5f} dk_whole={r['dk_whole']:+.5f} "
                  f"({r['elapsed']:.1f}s) [total {elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t_start
    print(f"\nTask 3 total: {elapsed:.0f}s = {elapsed/60:.1f}min", flush=True)

    # Summary
    print(f"\n{'='*80}", flush=True)
    print(f"RESULTS: N={N} PP-WAVE", flush=True)
    print("=" * 80, flush=True)

    results = {}
    for eps in EPS_VALUES:
        key = f"{eps}"
        trials = results_raw[key]
        dks_int = [t['dk_int'] for t in trials]
        dks_whole = [t['dk_whole'] for t in trials]

        mn_i, se_i, d_i = summarize(dks_int)
        mn_w, se_w, d_w = summarize(dks_whole)

        results[eps] = {
            'N': N, 'eps': eps,
            'dk_int_mean': mn_i, 'dk_int_se': se_i, 'dk_int_d': d_i,
            'dk_whole_mean': mn_w, 'dk_whole_se': se_w, 'dk_whole_d': d_w,
            'dks_int': dks_int, 'dks_whole': dks_whole,
            'K2': mn_i / eps**2,
        }

        print(f"  eps={eps}: dk_int = {mn_i:+.5f} +/- {se_i:.5f} (d={d_i:.1f}) | "
              f"dk_whole = {mn_w:+.5f} +/- {se_w:.5f} (d={d_w:.1f})", flush=True)

    # Plateau check: compare with N=20000
    print(f"\nPLATEAU CHECK (N=20000 -> N=40000):", flush=True)
    # N=20000 values from decisive package
    n20k = {2: 0.1021, 3: 0.2192}
    for eps in EPS_VALUES:
        r = results[eps]
        ratio = r['dk_int_mean'] / n20k[eps]
        alpha_local = np.log2(ratio)
        print(f"  eps={eps}: dk_int(40k)/dk_int(20k) = {r['dk_int_mean']:.5f}/{n20k[eps]:.5f} "
              f"= {ratio:.3f}, local alpha = {alpha_local:.3f}", flush=True)

    # K2 convergence full table
    print(f"\nK2 = Dkappa_int/eps^2 FULL TABLE:", flush=True)
    print(f"  (from decisive + this run)", flush=True)
    for eps in EPS_VALUES:
        print(f"  eps={eps}:", flush=True)
        prev = {5000: {2: 0.0824/4, 3: 0.1618/9},
                10000: {2: 0.0949/4, 3: 0.2023/9},
                20000: {2: 0.1021/4, 3: 0.2192/9}}
        for n_, k2 in sorted(prev.items()):
            print(f"    N={n_:5d}: K2 = {k2[eps]:.5f}", flush=True)
        print(f"    N={N:5d}: K2 = {results[eps]['K2']:.5f}", flush=True)

    return results


# ═════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t_global = time.time()

    # Task 1: nu_N (fastest, ~20 min)
    results1 = run_task1()

    # Task 2: per-N Sch mass fits (~45 min)
    results2 = run_task2()

    # Task 3: N=40000 pp-wave (~60 min)
    results3 = run_task3()

    total = time.time() - t_global

    # ─── SAVE ALL RESULTS ──────────────────────────────────────
    out_dir = 'analysis/discovery_runs/run_001'
    os.makedirs(out_dir, exist_ok=True)

    output = {
        'task1_nu': results1,
        'task2_sch_perN': {str(k): v for k, v in results2.items()},
        'task3_ppw40k': {str(k): v for k, v in results3.items()},
        'total_time_s': total,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    out_path = os.path.join(out_dir, 'e4_closure_package.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*80}", flush=True)
    print(f"ALL TASKS COMPLETE. Total: {total:.0f}s = {total/60:.1f}min", flush=True)
    print(f"Saved to: {out_path}", flush=True)
    print("=" * 80, flush=True)

    # ─── FINAL SUMMARY ─────────────────────────────────────────
    print(f"\n{'='*80}", flush=True)
    print("E4 CLOSURE SUMMARY", flush=True)
    print("=" * 80, flush=True)

    print("\n1. NU_N CONVERGENCE:", flush=True)
    for eps in [2, 3]:
        print(f"   eps={eps}:", flush=True)
        for N in [5000, 10000, 20000]:
            key = f"{N}_{eps}"
            if key in results1:
                r = results1[key]
                print(f"     N={N:5d}: nu = {r['nu_mean']:.4f} +/- {r['nu_se']:.4f}", flush=True)

    print("\n2. ALPHA(N), BETA(N) RUNNING:", flush=True)
    print(f"   N= 5000: alpha = +15.53, beta = -810.7 (previous)", flush=True)
    for N in [10000, 20000]:
        if N in results2 and 'fit' in results2[N]:
            f = results2[N]['fit']
            print(f"   N={N:5d}: alpha = {f['alpha']:+.2f} +/- {f['alpha_se']:.2f}, "
                  f"beta = {f['beta']:+.1f} +/- {f['beta_se']:.1f}", flush=True)

    print("\n3. PP-WAVE PLATEAU (K2 = dk_int / eps^2):", flush=True)
    for eps in [2, 3]:
        prev_vals = {5000: {2: 0.0206, 3: 0.01798},
                     10000: {2: 0.02373, 3: 0.02248},
                     20000: {2: 0.02553, 3: 0.02436}}
        k2_40k = results3[eps]['K2'] if eps in results3 else None
        print(f"   eps={eps}: N=5k→10k→20k→40k: ", end="", flush=True)
        vals = [prev_vals[n][eps] for n in [5000, 10000, 20000]]
        if k2_40k is not None:
            vals.append(k2_40k)
        print(" → ".join(f"{v:.5f}" for v in vals), flush=True)
        if len(vals) >= 4:
            ratio = vals[3] / vals[2]
            print(f"         20k→40k ratio = {ratio:.3f} "
                  f"(alpha_local = {np.log2(ratio):.3f})", flush=True)
