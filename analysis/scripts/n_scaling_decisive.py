"""
N-SCALING DECISIVE PACKAGE
==========================
Resolves the N-scaling anomaly for path_kurtosis.

For each (N, eps/M) pair, computes BOTH:
  - whole-diamond Δκ
  - 80% interior window Δκ (boundary_slack >= 20th percentile)

Uses multiprocessing for parallel seeds (8 P-core workers).

PP-WAVE:  N = {5000, 10000, 20000}, eps = {2, 3}, M_seeds = 15
SCHWARZSCHILD: N = {5000, 10000, 20000}, M_sch = {0.005, 0.01}, M_seeds = 15

Three rival models fit:
  Model 1: Δκ = c × ε² × C₂ × N^{1/2}            (original)
  Model 2: Δκ = c × ε² × C₂ × N^{1/2} × (1 + a/N^{1/4})  (finite-size correction)
  Model 3: Δκ = c × ε² × C₂ × N^α                 (free exponent)
"""
import sys, time, json, os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, 'analysis')
from sct_tools.hasse import (
    sprinkle_diamond, sprinkle_shell, build_hasse_bitset,
    path_counts, path_kurtosis_from_lists,
)

# ─── CONFIGURATION ────────────────────────────────────────────
PP_N_VALUES = [5000, 10000, 20000]
PP_EPS_VALUES = [2.0, 3.0]
PP_M_SEEDS = 15

SCH_N_VALUES = [5000, 10000, 20000]
SCH_M_VALUES = [0.005, 0.01]
SCH_M_SEEDS = 15
SCH_R_MIN = 0.10

INTERIOR_FRAC = 0.80
N_WORKERS = 8
T = 1.0

# ─── HELPERS ──────────────────────────────────────────────────

def boundary_slack(pts, T=1.0):
    """Distance from causal diamond boundary: T/2 - (|t| + r)."""
    r = np.sqrt(np.sum(pts[:, 1:]**2, axis=1))
    return T / 2 - (np.abs(pts[:, 0]) + r)


def kurtosis_windowed(parents, children, pts, T=1.0, frac=1.0):
    """Compute path_kurtosis on a boundary-interior window.

    frac=1.0: whole diamond
    frac=0.80: 80% most interior elements
    """
    pd, pu = path_counts(parents, children)
    Y = np.log2(pd * pu + 1.0)

    if frac < 1.0:
        slack = boundary_slack(pts, T)
        thr = np.quantile(slack, 1.0 - frac)
        mask = slack >= thr
        Y = Y[mask]

    X = Y - Y.mean()
    s2 = np.var(Y)
    if s2 < 1e-12:
        return 0.0
    return float(np.mean(X ** 4) / (s2 * s2) - 3.0)


def crn_ppwave_windowed(N, eps, seed, T=1.0, frac=0.80):
    """One CRN trial returning (dk_whole, dk_interior)."""
    pts = sprinkle_diamond(N, T=T, seed=seed)

    p0, c0 = build_hasse_bitset(pts, eps=None)
    pE, cE = build_hasse_bitset(pts, eps=eps)

    k0_whole = kurtosis_windowed(p0, c0, pts, T, frac=1.0)
    kE_whole = kurtosis_windowed(pE, cE, pts, T, frac=1.0)

    k0_int = kurtosis_windowed(p0, c0, pts, T, frac=frac)
    kE_int = kurtosis_windowed(pE, cE, pts, T, frac=frac)

    return kE_whole - k0_whole, kE_int - k0_int


def crn_sch_windowed(N, M_sch, seed, T=1.0, r_min=0.10, frac=0.80):
    """One Schwarzschild CRN trial returning (dk_whole, dk_interior)."""
    pts = sprinkle_shell(N, T=T, r_min=r_min, seed=seed)

    p0, c0 = build_hasse_bitset(pts, eps=None)
    pS, cS = build_hasse_bitset(pts, M_sch=M_sch)

    k0_whole = kurtosis_windowed(p0, c0, pts, T, frac=1.0)
    kS_whole = kurtosis_windowed(pS, cS, pts, T, frac=1.0)

    k0_int = kurtosis_windowed(p0, c0, pts, T, frac=frac)
    kS_int = kurtosis_windowed(pS, cS, pts, T, frac=frac)

    return kS_whole - k0_whole, kS_int - k0_int


# Wrapper for multiprocessing (top-level functions for pickle)
def _ppw_worker(args):
    N, eps, seed, T, frac = args
    t0 = time.time()
    dk_w, dk_i = crn_ppwave_windowed(N, eps, seed, T, frac)
    dt = time.time() - t0
    return N, eps, seed, dk_w, dk_i, dt


def _sch_worker(args):
    N, M_sch, seed, T, r_min, frac = args
    t0 = time.time()
    dk_w, dk_i = crn_sch_windowed(N, M_sch, seed, T, r_min, frac)
    dt = time.time() - t0
    return N, M_sch, seed, dk_w, dk_i, dt


def summarize(dks, label=""):
    """Mean, SE, Cohen's d."""
    a = np.array(dks)
    mn = a.mean()
    se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    d = mn / se if se > 1e-15 else 0.0
    return mn, se, d


# ─── CONSTANTS ────────────────────────────────────────────────
C2 = 24.0 / (T**2)  # for diamond (a=0)
C1_rms_fn = lambda a: np.sqrt(8) * (T - 2*a) / (T**2 + 4*T*a + 12*a**2)


# ==============================================================
# PP-WAVE RUNS
# ==============================================================
def run_ppwave():
    print("=" * 70, flush=True)
    print("PP-WAVE DECISIVE PACKAGE", flush=True)
    print(f"N = {PP_N_VALUES}, eps = {PP_EPS_VALUES}, M = {PP_M_SEEDS}", flush=True)
    print(f"Interior fraction = {INTERIOR_FRAC}", flush=True)
    print(f"Workers = {N_WORKERS}", flush=True)
    print("=" * 70, flush=True)

    # Build task list
    tasks = []
    for N in PP_N_VALUES:
        for eps in PP_EPS_VALUES:
            for m in range(PP_M_SEEDS):
                seed = 4000000 + N + int(eps * 100) + m
                tasks.append((N, eps, seed, T, INTERIOR_FRAC))

    print(f"Total trials: {len(tasks)}", flush=True)

    # Run in parallel
    results_raw = {}
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_ppw_worker, t): t for t in tasks}
        done_count = 0
        for fut in as_completed(futures):
            N, eps, seed, dk_w, dk_i, dt = fut.result()
            key = f"{N}_{eps}"
            if key not in results_raw:
                results_raw[key] = {'whole': [], 'interior': []}
            results_raw[key]['whole'].append(dk_w)
            results_raw[key]['interior'].append(dk_i)
            done_count += 1
            if done_count % 10 == 0 or done_count == len(tasks):
                elapsed = time.time() - t_start
                print(f"  [{done_count}/{len(tasks)}] N={N} eps={eps} "
                      f"dk_w={dk_w:+.5f} dk_i={dk_i:+.5f} ({dt:.1f}s) "
                      f"[total {elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t_start
    print(f"\nPP-wave total: {elapsed:.0f}s = {elapsed/60:.1f}min", flush=True)

    # Summary table
    print(f"\n{'='*70}", flush=True)
    print(f"PP-WAVE RESULTS", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'N':>6s} {'eps':>5s} | {'dk_whole':>12s} {'SE':>8s} {'d':>6s} | "
          f"{'dk_int80':>12s} {'SE':>8s} {'d':>6s} | "
          f"{'A_whole':>8s} {'A_int80':>8s} {'ratio':>6s}", flush=True)
    print("-" * 110, flush=True)

    pp_results = {}
    for N in PP_N_VALUES:
        for eps in PP_EPS_VALUES:
            key = f"{N}_{eps}"
            dw = np.array(results_raw[key]['whole'])
            di = np.array(results_raw[key]['interior'])

            mn_w, se_w, d_w = summarize(dw)
            mn_i, se_i, d_i = summarize(di)

            # A_eff = dk / (eps² × √N × C₂)
            norm = eps**2 * np.sqrt(N) * C2
            A_w = mn_w / norm
            A_i = mn_i / norm
            ratio = A_i / A_w if abs(A_w) > 1e-15 else float('inf')

            pp_results[key] = {
                'N': N, 'eps': eps,
                'dk_whole_mean': float(mn_w), 'dk_whole_se': float(se_w),
                'dk_whole_d': float(d_w),
                'dk_int_mean': float(mn_i), 'dk_int_se': float(se_i),
                'dk_int_d': float(d_i),
                'A_eff_whole': float(A_w), 'A_eff_int': float(A_i),
                'ratio_int_whole': float(ratio),
                'dks_whole': dw.tolist(), 'dks_int': di.tolist(),
            }

            print(f"{N:6d} {eps:5.1f} | {mn_w:+12.6f} {se_w:8.6f} {d_w:6.1f} | "
                  f"{mn_i:+12.6f} {se_i:8.6f} {d_i:6.1f} | "
                  f"{A_w:8.4f} {A_i:8.4f} {ratio:6.2f}", flush=True)

    # A_eff(N) trend
    print(f"\n--- A_eff(N) trend (whole vs interior) ---", flush=True)
    for eps in PP_EPS_VALUES:
        print(f"\n  eps={eps}:", flush=True)
        print(f"  {'N':>6s} {'A_whole':>10s} {'A_int80':>10s}", flush=True)
        for N in PP_N_VALUES:
            key = f"{N}_{eps}"
            print(f"  {N:6d} {pp_results[key]['A_eff_whole']:10.4f} "
                  f"{pp_results[key]['A_eff_int']:10.4f}", flush=True)

    return pp_results, elapsed


# ==============================================================
# SCHWARZSCHILD RUNS
# ==============================================================
def run_schwarzschild():
    print(f"\n{'='*70}", flush=True)
    print("SCHWARZSCHILD DECISIVE PACKAGE", flush=True)
    print(f"N = {SCH_N_VALUES}, M_sch = {SCH_M_VALUES}, M_seeds = {SCH_M_SEEDS}", flush=True)
    print(f"Interior fraction = {INTERIOR_FRAC}, r_min = {SCH_R_MIN}", flush=True)
    print(f"Workers = {N_WORKERS}", flush=True)
    print("=" * 70, flush=True)

    tasks = []
    for N in SCH_N_VALUES:
        for M_sch in SCH_M_VALUES:
            for m in range(SCH_M_SEEDS):
                seed = 5000000 + N + int(M_sch * 100000) + m
                tasks.append((N, M_sch, seed, T, SCH_R_MIN, INTERIOR_FRAC))

    print(f"Total trials: {len(tasks)}", flush=True)

    results_raw = {}
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_sch_worker, t): t for t in tasks}
        done_count = 0
        for fut in as_completed(futures):
            N, M_sch, seed, dk_w, dk_i, dt = fut.result()
            key = f"{N}_{M_sch}"
            if key not in results_raw:
                results_raw[key] = {'whole': [], 'interior': []}
            results_raw[key]['whole'].append(dk_w)
            results_raw[key]['interior'].append(dk_i)
            done_count += 1
            if done_count % 10 == 0 or done_count == len(tasks):
                elapsed = time.time() - t_start
                print(f"  [{done_count}/{len(tasks)}] N={N} M={M_sch} "
                      f"dk_w={dk_w:+.5f} dk_i={dk_i:+.5f} ({dt:.1f}s) "
                      f"[total {elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t_start
    print(f"\nSchwarzschild total: {elapsed:.0f}s = {elapsed/60:.1f}min", flush=True)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print(f"SCHWARZSCHILD RESULTS", flush=True)
    print(f"{'='*70}", flush=True)

    a = SCH_R_MIN
    C1_rms = C1_rms_fn(a)
    print(f"C1_rms(r_min={a}) = {C1_rms:.4f}", flush=True)

    print(f"\n{'N':>6s} {'M':>7s} | {'dk_whole':>12s} {'SE':>8s} {'d':>6s} | "
          f"{'dk_int80':>12s} {'SE':>8s} {'d':>6s} | "
          f"{'B_whole':>8s} {'B_int80':>8s}", flush=True)
    print("-" * 100, flush=True)

    sch_results = {}
    for N in SCH_N_VALUES:
        for M_sch in SCH_M_VALUES:
            key = f"{N}_{M_sch}"
            dw = np.array(results_raw[key]['whole'])
            di = np.array(results_raw[key]['interior'])

            mn_w, se_w, d_w = summarize(dw)
            mn_i, se_i, d_i = summarize(di)

            # B_eff = dk / (M × N^{1/4} × C1_rms)
            norm = M_sch * N**0.25 * C1_rms
            B_w = mn_w / norm
            B_i = mn_i / norm

            sch_results[key] = {
                'N': N, 'M_sch': M_sch,
                'dk_whole_mean': float(mn_w), 'dk_whole_se': float(se_w),
                'dk_whole_d': float(d_w),
                'dk_int_mean': float(mn_i), 'dk_int_se': float(se_i),
                'dk_int_d': float(d_i),
                'B_eff_whole': float(B_w), 'B_eff_int': float(B_i),
                'dks_whole': dw.tolist(), 'dks_int': di.tolist(),
            }

            print(f"{N:6d} {M_sch:7.4f} | {mn_w:+12.6f} {se_w:8.6f} {d_w:6.1f} | "
                  f"{mn_i:+12.6f} {se_i:8.6f} {d_i:6.1f} | "
                  f"{B_w:8.4f} {B_i:8.4f}", flush=True)

    # B_eff(N) trend
    print(f"\n--- B_eff(N) trend (whole vs interior) ---", flush=True)
    for M_sch in SCH_M_VALUES:
        print(f"\n  M={M_sch}:", flush=True)
        print(f"  {'N':>6s} {'B_whole':>10s} {'B_int80':>10s}", flush=True)
        for N in SCH_N_VALUES:
            key = f"{N}_{M_sch}"
            print(f"  {N:6d} {sch_results[key]['B_eff_whole']:10.4f} "
                  f"{sch_results[key]['B_eff_int']:10.4f}", flush=True)

    return sch_results, elapsed


# ==============================================================
# MODEL FITTING
# ==============================================================
def fit_models(pp_results):
    """Fit three rival N-scaling models for pp-wave."""
    print(f"\n{'='*70}", flush=True)
    print("MODEL FITTING: A_eff(N) for pp-wave", flush=True)
    print("=" * 70, flush=True)

    from scipy.optimize import curve_fit

    for window in ['whole', 'int']:
        for eps in PP_EPS_VALUES:
            Ns = np.array(PP_N_VALUES, dtype=float)
            dks = []
            ses = []
            for N in PP_N_VALUES:
                key = f"{N}_{eps}"
                r = pp_results[key]
                if window == 'whole':
                    dks.append(r['dk_whole_mean'])
                    ses.append(r['dk_whole_se'])
                else:
                    dks.append(r['dk_int_mean'])
                    ses.append(r['dk_int_se'])
            dks = np.array(dks)
            ses = np.array(ses)

            norm_eps = eps**2 * C2

            # Model 1: dk = c × eps² × C₂ × N^{1/2}
            def m1(N, c): return c * norm_eps * N**0.5
            # Model 2: dk = c × eps² × C₂ × N^{1/2} × (1 + a/N^{1/4})
            def m2(N, c, a): return c * norm_eps * N**0.5 * (1 + a / N**0.25)
            # Model 3: dk = c × eps² × C₂ × N^α
            def m3(N, c, alpha): return c * norm_eps * N**alpha

            label = f"eps={eps}, {window}-diamond"
            print(f"\n  --- {label} ---", flush=True)
            print(f"  Data: {list(zip(PP_N_VALUES, [f'{d:+.5f}' for d in dks]))}", flush=True)

            # Fit Model 1
            try:
                p1, cov1 = curve_fit(m1, Ns, dks, p0=[0.07], sigma=ses, absolute_sigma=True)
                resid1 = dks - m1(Ns, *p1)
                chi2_1 = float(np.sum((resid1 / ses)**2))
                print(f"  M1 (√N):   c={p1[0]:.5f}  χ²={chi2_1:.2f}/{len(Ns)-1}", flush=True)
            except Exception as e:
                print(f"  M1 failed: {e}", flush=True)
                chi2_1 = 1e9

            # Fit Model 2
            try:
                p2, cov2 = curve_fit(m2, Ns, dks, p0=[0.07, -5.0], sigma=ses, absolute_sigma=True)
                resid2 = dks - m2(Ns, *p2)
                chi2_2 = float(np.sum((resid2 / ses)**2))
                print(f"  M2 (√N+corr): c={p2[0]:.5f} a={p2[1]:.2f}  χ²={chi2_2:.2f}/{len(Ns)-2}", flush=True)
            except Exception as e:
                print(f"  M2 failed: {e}", flush=True)
                chi2_2 = 1e9

            # Fit Model 3
            try:
                p3, cov3 = curve_fit(m3, Ns, dks, p0=[0.001, 0.7], sigma=ses, absolute_sigma=True)
                resid3 = dks - m3(Ns, *p3)
                chi2_3 = float(np.sum((resid3 / ses)**2))
                print(f"  M3 (N^α):  c={p3[0]:.6f} α={p3[1]:.3f}  χ²={chi2_3:.2f}/{len(Ns)-2}", flush=True)
            except Exception as e:
                print(f"  M3 failed: {e}", flush=True)
                chi2_3 = 1e9

            # AICc comparison (small sample)
            n_data = len(Ns)
            for name, chi2, k in [("M1(√N)", chi2_1, 1), ("M2(corr)", chi2_2, 2), ("M3(N^α)", chi2_3, 2)]:
                aic = chi2 + 2*k
                if n_data - k - 1 > 0:
                    aicc = aic + 2*k*(k+1)/(n_data - k - 1)
                else:
                    aicc = float('inf')
                print(f"    {name}: AICc = {aicc:.2f}", flush=True)


# ==============================================================
# MAIN
# ==============================================================
if __name__ == '__main__':
    t_global = time.time()

    pp_results, pp_time = run_ppwave()
    sch_results, sch_time = run_schwarzschild()
    fit_models(pp_results)

    total = time.time() - t_global
    print(f"\n{'='*70}", flush=True)
    print(f"TOTAL: {total:.0f}s = {total/60:.1f}min", flush=True)
    print(f"  PP-wave: {pp_time:.0f}s, Schwarzschild: {sch_time:.0f}s", flush=True)

    # ─── KEY DIAGNOSTIC ───────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print("KEY DIAGNOSTIC: Does interior window stabilize A_eff?", flush=True)
    print("=" * 70, flush=True)
    for eps in PP_EPS_VALUES:
        print(f"\n  eps={eps}:", flush=True)
        print(f"  {'N':>6s} {'A_whole':>10s} {'A_int80':>10s} {'stable?':>10s}", flush=True)
        A_prev_w = None
        A_prev_i = None
        for N in PP_N_VALUES:
            key = f"{N}_{eps}"
            Aw = pp_results[key]['A_eff_whole']
            Ai = pp_results[key]['A_eff_int']
            stable = ""
            if A_prev_i is not None:
                ratio_w = Aw / A_prev_w if A_prev_w else 0
                ratio_i = Ai / A_prev_i if A_prev_i else 0
                stable = f"w:{ratio_w:.2f} i:{ratio_i:.2f}"
            print(f"  {N:6d} {Aw:10.4f} {Ai:10.4f} {stable:>10s}", flush=True)
            A_prev_w = Aw
            A_prev_i = Ai

    # ─── SAVE ─────────────────────────────────────────────────
    out = {
        'config': {
            'pp_N': PP_N_VALUES, 'pp_eps': PP_EPS_VALUES, 'pp_M_seeds': PP_M_SEEDS,
            'sch_N': SCH_N_VALUES, 'sch_M': SCH_M_VALUES, 'sch_M_seeds': SCH_M_SEEDS,
            'interior_frac': INTERIOR_FRAC, 'r_min': SCH_R_MIN, 'T': T,
        },
        'ppwave': {k: {kk: vv for kk, vv in v.items() if kk != 'dks_whole' and kk != 'dks_int'}
                   for k, v in pp_results.items()},
        'ppwave_raw': {k: {'whole': v['dks_whole'], 'int': v['dks_int']}
                       for k, v in pp_results.items()},
        'schwarzschild': {k: {kk: vv for kk, vv in v.items() if kk != 'dks_whole' and kk != 'dks_int'}
                          for k, v in sch_results.items()},
        'schwarzschild_raw': {k: {'whole': v['dks_whole'], 'int': v['dks_int']}
                              for k, v in sch_results.items()},
        'total_time_s': total,
    }
    outpath = 'analysis/discovery_runs/run_001/n_scaling_decisive.json'
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {outpath}", flush=True)
    print("DONE", flush=True)
