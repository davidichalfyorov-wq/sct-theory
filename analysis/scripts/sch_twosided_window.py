"""
SCHWARZSCHILD TWO-SIDED SHELL WINDOW PACKAGE
=============================================
Resolves sign reversal question from independent analysis analysis.

For shell geometry (r >= r_min), the current "interior 80%" only trims
the OUTER diamond boundary. This script tests FOUR window definitions:

  1. whole       — all N elements
  2. outer_trim  — 80% by outer boundary slack: T/2 - (|t| + r)
  3. inner_trim  — 80% by inner shell slack: r - r_min
  4. twosided    — 80% by min(outer_slack, inner_slack)

If sign reversal survives in two-sided window → real physical effect.
If it disappears → artifact of one-sided trimming.

Also runs interior O(M²) fit at N=5000 with 6 masses.

Config: N = {5000, 10000, 20000}, M_sch = {0.005, 0.01}, M_seeds = 15
        + O(M²): N=5000, M = {0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02}, M_seeds=15
"""
import sys, time, json, os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, 'analysis')
from sct_tools.hasse import (
    sprinkle_shell, build_hasse_bitset, path_counts,
)

# ─── CONFIGURATION ────────────────────────────────────────────
N_VALUES = [5000, 10000, 20000]
M_SCH_VALUES = [0.005, 0.01]
M_SEEDS = 15
R_MIN = 0.10
T = 1.0
INTERIOR_FRAC = 0.80
N_WORKERS = 8

# O(M²) fit configuration
OM2_N = 5000
OM2_MASSES = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]
OM2_SEEDS = 15

# ─── SLACK FUNCTIONS ──────────────────────────────────────────

def outer_slack(pts, T=1.0):
    """Distance from outer diamond boundary: T/2 - (|t| + r)."""
    r = np.sqrt(np.sum(pts[:, 1:]**2, axis=1))
    return T / 2 - (np.abs(pts[:, 0]) + r)


def inner_slack(pts, r_min=0.10):
    """Distance from inner shell boundary: r - r_min."""
    r = np.sqrt(np.sum(pts[:, 1:]**2, axis=1))
    return r - r_min


def twosided_slack(pts, T=1.0, r_min=0.10):
    """Two-sided shell slack: min(outer, inner)."""
    return np.minimum(outer_slack(pts, T), inner_slack(pts, r_min))


# ─── WINDOWED KURTOSIS ───────────────────────────────────────

def kurtosis_on_subset(Y, mask=None):
    """Compute excess kurtosis of Y on a subset defined by mask."""
    if mask is not None:
        Y = Y[mask]
    if len(Y) < 4:
        return 0.0
    X = Y - Y.mean()
    s2 = np.var(Y)
    if s2 < 1e-12:
        return 0.0
    return float(np.mean(X ** 4) / (s2 * s2) - 3.0)


def compute_Y(parents, children):
    """Compute Y = log₂(p_down × p_up + 1) from Hasse lists."""
    pd, pu = path_counts(parents, children)
    return np.log2(pd * pu + 1.0)


def get_window_masks(pts, T=1.0, r_min=0.10, frac=0.80):
    """Return dict of masks for 4 window types.

    Returns:
        dict with keys: 'whole', 'outer_trim', 'inner_trim', 'twosided'
        Each value is a boolean mask of length N.
    """
    n = len(pts)
    os_ = outer_slack(pts, T)
    is_ = inner_slack(pts, r_min)
    ts_ = twosided_slack(pts, T, r_min)

    masks = {
        'whole': np.ones(n, dtype=bool),
    }

    # outer_trim: keep top frac% by outer_slack
    thr_out = np.quantile(os_, 1.0 - frac)
    masks['outer_trim'] = os_ >= thr_out

    # inner_trim: keep top frac% by inner_slack
    thr_in = np.quantile(is_, 1.0 - frac)
    masks['inner_trim'] = is_ >= thr_in

    # twosided: keep top frac% by twosided_slack
    thr_ts = np.quantile(ts_, 1.0 - frac)
    masks['twosided'] = ts_ >= thr_ts

    return masks


# ─── CRN TRIAL ────────────────────────────────────────────────

def crn_sch_4windows(N, M_sch, seed, T=1.0, r_min=0.10, frac=0.80):
    """One Schwarzschild CRN trial with 4 window readouts.

    Returns:
        dict: {window_name: dk} for each of 4 windows
    """
    pts = sprinkle_shell(N, T=T, r_min=r_min, seed=seed)

    p0, c0 = build_hasse_bitset(pts, eps=None)
    pS, cS = build_hasse_bitset(pts, M_sch=M_sch)

    Y0 = compute_Y(p0, c0)
    YS = compute_Y(pS, cS)

    masks = get_window_masks(pts, T, r_min, frac)

    result = {}
    for wname, mask in masks.items():
        k0 = kurtosis_on_subset(Y0, mask if wname != 'whole' else None)
        kS = kurtosis_on_subset(YS, mask if wname != 'whole' else None)
        result[wname] = kS - k0

    return result


# ─── WORKER ───────────────────────────────────────────────────

def _worker(args):
    N, M_sch, seed, T_, r_min_, frac_ = args
    t0 = time.time()
    dks = crn_sch_4windows(N, M_sch, seed, T_, r_min_, frac_)
    dt = time.time() - t0
    return N, M_sch, seed, dks, dt


def summarize(arr):
    """Mean, SE, Cohen's d."""
    a = np.array(arr)
    mn = a.mean()
    se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    d = mn / se if se > 1e-15 else 0.0
    return float(mn), float(se), float(d)


# ─── MAIN N-SCALING RUN ──────────────────────────────────────

def run_n_scaling():
    print("=" * 80, flush=True)
    print("SCHWARZSCHILD TWO-SIDED SHELL WINDOW PACKAGE", flush=True)
    print(f"N = {N_VALUES}, M_sch = {M_SCH_VALUES}, seeds = {M_SEEDS}", flush=True)
    print(f"r_min = {R_MIN}, interior_frac = {INTERIOR_FRAC}, workers = {N_WORKERS}", flush=True)
    print("=" * 80, flush=True)

    tasks = []
    for N in N_VALUES:
        for M_sch in M_SCH_VALUES:
            for m in range(M_SEEDS):
                seed = 6000000 + N + int(M_sch * 100000) + m
                tasks.append((N, M_sch, seed, T, R_MIN, INTERIOR_FRAC))

    print(f"Total trials: {len(tasks)}", flush=True)

    results_raw = {}
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        done_count = 0
        for fut in as_completed(futures):
            N, M_sch, seed, dks, dt = fut.result()
            key = f"{N}_{M_sch}"
            if key not in results_raw:
                results_raw[key] = {w: [] for w in ['whole', 'outer_trim', 'inner_trim', 'twosided']}
            for w, dk in dks.items():
                results_raw[key][w].append(dk)
            done_count += 1
            if done_count % 5 == 0 or done_count == len(tasks):
                elapsed = time.time() - t_start
                print(f"  [{done_count}/{len(tasks)}] N={N} M={M_sch} "
                      f"whole={dks['whole']:+.5f} outer={dks['outer_trim']:+.5f} "
                      f"inner={dks['inner_trim']:+.5f} 2side={dks['twosided']:+.5f} "
                      f"({dt:.1f}s) [total {elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t_start
    print(f"\nN-scaling total: {elapsed:.0f}s = {elapsed/60:.1f}min", flush=True)

    # Summary table
    WINDOWS = ['whole', 'outer_trim', 'inner_trim', 'twosided']

    print(f"\n{'='*80}", flush=True)
    print("RESULTS: 4 WINDOWS × N × M", flush=True)
    print("=" * 80, flush=True)

    results = {}
    for N in N_VALUES:
        for M_sch in M_SCH_VALUES:
            key = f"{N}_{M_sch}"
            results[key] = {'N': N, 'M_sch': M_sch}
            for w in WINDOWS:
                mn, se, d = summarize(results_raw[key][w])
                results[key][f'dk_{w}'] = mn
                results[key][f'se_{w}'] = se
                results[key][f'd_{w}'] = d
                results[key][f'dks_{w}'] = results_raw[key][w]

    # Print compact table
    print(f"\n{'N':>6s} {'M':>7s} | {'whole':>12s} {'SE':>8s} | "
          f"{'outer_80':>12s} {'SE':>8s} | {'inner_80':>12s} {'SE':>8s} | "
          f"{'twosided':>12s} {'SE':>8s}", flush=True)
    print("-" * 120, flush=True)

    for N in N_VALUES:
        for M_sch in M_SCH_VALUES:
            key = f"{N}_{M_sch}"
            r = results[key]
            print(f"{N:6d} {M_sch:7.4f} | "
                  f"{r['dk_whole']:+12.6f} {r['se_whole']:8.6f} | "
                  f"{r['dk_outer_trim']:+12.6f} {r['se_outer_trim']:8.6f} | "
                  f"{r['dk_inner_trim']:+12.6f} {r['se_inner_trim']:8.6f} | "
                  f"{r['dk_twosided']:+12.6f} {r['se_twosided']:8.6f}",
                  flush=True)

    # Sign analysis
    print(f"\n{'='*80}", flush=True)
    print("SIGN ANALYSIS: Which windows are positive vs negative?", flush=True)
    print("=" * 80, flush=True)
    for N in N_VALUES:
        for M_sch in M_SCH_VALUES:
            key = f"{N}_{M_sch}"
            r = results[key]
            signs = {w: '+' if r[f'dk_{w}'] > 0 else '−' for w in WINDOWS}
            print(f"  N={N:5d} M={M_sch}: "
                  + " | ".join(f"{w}={signs[w]}" for w in WINDOWS), flush=True)

    # Cohen's d table
    print(f"\n{'='*80}", flush=True)
    print("COHEN'S d (significance)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'N':>6s} {'M':>7s} | {'whole':>8s} | {'outer':>8s} | {'inner':>8s} | {'2side':>8s}", flush=True)
    print("-" * 60, flush=True)
    for N in N_VALUES:
        for M_sch in M_SCH_VALUES:
            key = f"{N}_{M_sch}"
            r = results[key]
            print(f"{N:6d} {M_sch:7.4f} | "
                  f"{r['d_whole']:8.1f} | {r['d_outer_trim']:8.1f} | "
                  f"{r['d_inner_trim']:8.1f} | {r['d_twosided']:8.1f}", flush=True)

    return results, elapsed


# ─── O(M²) FIT WITH 4 WINDOWS ────────────────────────────────

def run_om2_fit():
    print(f"\n{'='*80}", flush=True)
    print(f"O(M²) FIT: N={OM2_N}, 6 masses, 4 windows", flush=True)
    print(f"Masses: {OM2_MASSES}", flush=True)
    print(f"Seeds: {OM2_SEEDS}, r_min={R_MIN}", flush=True)
    print("=" * 80, flush=True)

    tasks = []
    for M_sch in OM2_MASSES:
        for m in range(OM2_SEEDS):
            seed = 7000000 + OM2_N + int(M_sch * 1000000) + m
            tasks.append((OM2_N, M_sch, seed, T, R_MIN, INTERIOR_FRAC))

    print(f"Total trials: {len(tasks)}", flush=True)

    results_raw = {}
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        done_count = 0
        for fut in as_completed(futures):
            N, M_sch, seed, dks, dt = fut.result()
            key = f"{M_sch}"
            if key not in results_raw:
                results_raw[key] = {w: [] for w in ['whole', 'outer_trim', 'inner_trim', 'twosided']}
            for w, dk in dks.items():
                results_raw[key][w].append(dk)
            done_count += 1
            if done_count % 10 == 0 or done_count == len(tasks):
                elapsed = time.time() - t_start
                print(f"  [{done_count}/{len(tasks)}] M={M_sch} ({dt:.1f}s) "
                      f"[total {elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t_start
    print(f"\nO(M²) total: {elapsed:.0f}s = {elapsed/60:.1f}min", flush=True)

    WINDOWS = ['whole', 'outer_trim', 'inner_trim', 'twosided']

    # Summarize
    om2_results = {}
    for M_sch in OM2_MASSES:
        key = f"{M_sch}"
        om2_results[key] = {'M_sch': M_sch}
        for w in WINDOWS:
            mn, se, d = summarize(results_raw[key][w])
            om2_results[key][f'dk_{w}'] = mn
            om2_results[key][f'se_{w}'] = se
            om2_results[key][f'd_{w}'] = d
            om2_results[key][f'dks_{w}'] = results_raw[key][w]

    # Print
    print(f"\n{'M':>8s} | {'whole':>12s} {'SE':>8s} | {'outer':>12s} {'SE':>8s} | "
          f"{'inner':>12s} {'SE':>8s} | {'twosided':>12s} {'SE':>8s}", flush=True)
    print("-" * 120, flush=True)

    for M_sch in OM2_MASSES:
        key = f"{M_sch}"
        r = om2_results[key]
        print(f"{M_sch:8.4f} | "
              f"{r['dk_whole']:+12.6f} {r['se_whole']:8.6f} | "
              f"{r['dk_outer_trim']:+12.6f} {r['se_outer_trim']:8.6f} | "
              f"{r['dk_inner_trim']:+12.6f} {r['se_inner_trim']:8.6f} | "
              f"{r['dk_twosided']:+12.6f} {r['se_twosided']:8.6f}",
              flush=True)

    # WLS fit: Δκ(M) = α·M + β·M² for each window
    from scipy.optimize import curve_fit

    print(f"\n{'='*80}", flush=True)
    print("WLS FIT: Δκ(M) = α·M + β·M² for each window", flush=True)
    print("=" * 80, flush=True)

    fit_results = {}
    for w in WINDOWS:
        M_arr = np.array(OM2_MASSES)
        dk_arr = np.array([om2_results[f"{M}"][f'dk_{w}'] for M in OM2_MASSES])
        se_arr = np.array([om2_results[f"{M}"][f'se_{w}'] for M in OM2_MASSES])

        # Linear + quadratic: dk = α·M + β·M²
        def quad_model(M, alpha, beta):
            return alpha * M + beta * M**2

        # Pure linear: dk = α·M
        def lin_model(M, alpha):
            return alpha * M

        try:
            p_quad, cov_quad = curve_fit(quad_model, M_arr, dk_arr,
                                         p0=[-8.0, -100.0], sigma=se_arr, absolute_sigma=True)
            alpha_q, beta_q = p_quad
            se_alpha = np.sqrt(cov_quad[0, 0])
            se_beta = np.sqrt(cov_quad[1, 1])

            resid_q = dk_arr - quad_model(M_arr, *p_quad)
            chi2_q = float(np.sum((resid_q / se_arr)**2))

            p_lin, cov_lin = curve_fit(lin_model, M_arr, dk_arr,
                                       p0=[-8.0], sigma=se_arr, absolute_sigma=True)
            resid_l = dk_arr - lin_model(M_arr, *p_lin)
            chi2_l = float(np.sum((resid_l / se_arr)**2))

            # Bootstrap CI for β
            n_boot = 5000
            rng = np.random.default_rng(42)
            alphas_boot = []
            betas_boot = []
            for _ in range(n_boot):
                dk_boot = dk_arr + rng.normal(0, se_arr)
                try:
                    pb, _ = curve_fit(quad_model, M_arr, dk_boot,
                                     p0=p_quad, sigma=se_arr, absolute_sigma=True)
                    alphas_boot.append(pb[0])
                    betas_boot.append(pb[1])
                except:
                    pass
            alphas_boot = np.array(alphas_boot)
            betas_boot = np.array(betas_boot)
            alpha_ci = np.percentile(alphas_boot, [2.5, 97.5])
            beta_ci = np.percentile(betas_boot, [2.5, 97.5])
            p_beta_pos = float(np.mean(betas_boot > 0))

            dchi2 = chi2_l - chi2_q

            print(f"\n  --- {w} ---", flush=True)
            print(f"  Quad fit: α = {alpha_q:.2f} ± {se_alpha:.2f}, "
                  f"95%CI [{alpha_ci[0]:.2f}, {alpha_ci[1]:.2f}]", flush=True)
            print(f"           β = {beta_q:.1f} ± {se_beta:.1f}, "
                  f"95%CI [{beta_ci[0]:.1f}, {beta_ci[1]:.1f}]", flush=True)
            print(f"           P(β>0) = {p_beta_pos:.4f}", flush=True)
            print(f"  Quad: χ²/ndof = {chi2_q:.2f}/{len(OM2_MASSES)-2} = "
                  f"{chi2_q/(len(OM2_MASSES)-2):.2f}", flush=True)
            print(f"  Linear: χ²/ndof = {chi2_l:.2f}/{len(OM2_MASSES)-1} = "
                  f"{chi2_l/(len(OM2_MASSES)-1):.2f}", flush=True)
            print(f"  Δχ² = {dchi2:.2f} (quad vs linear)", flush=True)

            fit_results[w] = {
                'alpha': float(alpha_q), 'alpha_se': float(se_alpha),
                'alpha_ci': [float(alpha_ci[0]), float(alpha_ci[1])],
                'beta': float(beta_q), 'beta_se': float(se_beta),
                'beta_ci': [float(beta_ci[0]), float(beta_ci[1])],
                'p_beta_pos': float(p_beta_pos),
                'chi2_quad': float(chi2_q), 'chi2_lin': float(chi2_l),
                'dchi2': float(dchi2),
            }
        except Exception as e:
            print(f"\n  --- {w} --- FIT FAILED: {e}", flush=True)
            fit_results[w] = {'error': str(e)}

    return om2_results, fit_results, elapsed


# ==============================================================
# MAIN
# ==============================================================
if __name__ == '__main__':
    t_global = time.time()

    nsc_results, nsc_time = run_n_scaling()
    om2_results, fit_results, om2_time = run_om2_fit()

    total = time.time() - t_global
    print(f"\n{'='*80}", flush=True)
    print(f"TOTAL: {total:.0f}s = {total/60:.1f}min", flush=True)

    # ─── CRITICAL DIAGNOSTIC ──────────────────────────────────
    print(f"\n{'='*80}", flush=True)
    print("CRITICAL DIAGNOSTIC: Sign reversal in two-sided window?", flush=True)
    print("=" * 80, flush=True)
    for N in N_VALUES:
        for M_sch in M_SCH_VALUES:
            key = f"{N}_{M_sch}"
            r = nsc_results[key]
            print(f"\n  N={N}, M={M_sch}:", flush=True)
            for w in ['whole', 'outer_trim', 'inner_trim', 'twosided']:
                dk = r[f'dk_{w}']
                se = r[f'se_{w}']
                d = r[f'd_{w}']
                sign = '✓ POSITIVE' if dk > 0 else '✗ NEGATIVE'
                sig = '***' if abs(d) > 3.0 else ('**' if abs(d) > 2.0 else ('*' if abs(d) > 1.5 else ''))
                print(f"    {w:>12s}: Δκ = {dk:+.6f} ± {se:.6f}  d={d:+.1f}{sig}  {sign}",
                      flush=True)

    # ─── SAVE ─────────────────────────────────────────────────
    # Strip per-seed dks for compact summary, keep raw separately
    nsc_compact = {}
    nsc_raw = {}
    for key, r in nsc_results.items():
        nsc_compact[key] = {k: v for k, v in r.items() if not k.startswith('dks_')}
        nsc_raw[key] = {k: v for k, v in r.items() if k.startswith('dks_')}

    om2_compact = {}
    om2_raw_out = {}
    for key, r in om2_results.items():
        om2_compact[key] = {k: v for k, v in r.items() if not k.startswith('dks_')}
        om2_raw_out[key] = {k: v for k, v in r.items() if k.startswith('dks_')}

    out = {
        'config': {
            'N_values': N_VALUES, 'M_sch_values': M_SCH_VALUES,
            'M_seeds': M_SEEDS, 'r_min': R_MIN, 'T': T,
            'interior_frac': INTERIOR_FRAC,
            'om2_N': OM2_N, 'om2_masses': OM2_MASSES, 'om2_seeds': OM2_SEEDS,
            'windows': ['whole', 'outer_trim', 'inner_trim', 'twosided'],
            'slack_definitions': {
                'outer_trim': 'T/2 - (|t| + r)',
                'inner_trim': 'r - r_min',
                'twosided': 'min(T/2 - (|t| + r), r - r_min)',
            },
        },
        'n_scaling': nsc_compact,
        'n_scaling_raw': nsc_raw,
        'om2_fit': om2_compact,
        'om2_fit_raw': om2_raw_out,
        'om2_wls_results': fit_results,
        'total_time_s': total,
    }

    outpath = 'analysis/discovery_runs/run_001/sch_twosided_window.json'
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {outpath}", flush=True)
    print("DONE", flush=True)
