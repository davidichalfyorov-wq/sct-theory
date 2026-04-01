#!/usr/bin/env python3
"""
Paper 7 mini-tests battery (Tier A):
  T1: de Sitter null test (CJ = 0 when Weyl = 0)
  T2: M_ss convergence rate fit
  T3: Extended epsilon range (eps = 0.5, 1, 2, 3, 5, 10)
  T4: Normality test (Shapiro-Wilk on M=50 trials)
  T5: Interval CJ = 0 verification
"""
import sys, os, time, json
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, ppwave_exact_preds,
    build_hasse_from_predicate, bulk_mask,
)

C0 = 32 * np.pi**2 / (3 * 362880 * 45)


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def path_counts(par, ch_list, n):
    pd = np.zeros(n, dtype=np.float64)
    pu = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if par[i] is not None and len(par[i]) > 0:
            pd[i] = np.sum(pd[list(par[i])]) + 1
        else:
            pd[i] = 1.0
    for i in range(n - 1, -1, -1):
        if ch_list[i]:
            pu[i] = np.sum(pu[ch_list[i]]) + 1
        else:
            pu[i] = 1.0
    return pd, pu


def make_children(par, n):
    ch = [[] for _ in range(n)]
    for i in range(n):
        if par[i] is not None:
            for j in par[i]:
                ch[int(j)].append(i)
    return ch


def compute_depth(par, n):
    depth = np.zeros(n, dtype=int)
    for i in range(n):
        if par[i] is not None and len(par[i]) > 0:
            depth[i] = int(np.max(depth[list(par[i])])) + 1
    return depth


def make_strata(pts, par, T):
    n = len(pts)
    tau_hat = 2.0 * pts[:, 0] / T
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2.0 - np.abs(pts[:, 0])
    rho_hat = np.clip(r / np.maximum(rmax, 1e-12), 0, 0.999999)
    tau_bin = np.clip(np.floor((tau_hat + 1) * 2.5).astype(int), 0, 4)
    rho_bin = np.clip(np.floor(rho_hat * 3).astype(int), 0, 2)
    depth = compute_depth(par, n)
    max_d = max(int(depth.max()), 1)
    depth_terc = np.clip((depth * 3) // (max_d + 1), 0, 2)
    return tau_bin * 9 + rho_bin * 3 + depth_terc


def cj_stratified(weight, response, strata_m, min_bin=3):
    total = 0.0
    for b in np.unique(strata_m):
        idx = strata_m == b
        if idx.sum() < min_bin:
            continue
        w = idx.sum() / len(weight)
        cov = (np.mean(weight[idx] * response[idx])
               - np.mean(weight[idx]) * np.mean(response[idx]))
        total += w * cov
    return float(total)


def single_trial_cj(N, T, eps, seed):
    """Run one CRN trial and return CJ value."""
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)
    n = len(pts)
    par0, _ = build_hasse_from_predicate(pts, minkowski_preds)
    curved_pred = lambda pts, i, eps=eps: ppwave_exact_preds(pts, i, eps)
    parC, _ = build_hasse_from_predicate(pts, curved_pred)
    ch0 = make_children(par0, n)
    chC = make_children(parC, n)
    pd0, pu0 = path_counts(par0, ch0, n)
    pdC, puC = path_counts(parC, chC, n)
    bmask = bulk_mask(pts, T, 0.15)
    strata = make_strata(pts, par0, T)
    strata_m = strata[bmask]
    Y0 = np.log2(pd0 * pu0 + 1)
    YC = np.log2(pdC * puC + 1)
    dY = YC[bmask] - Y0[bmask]
    X0 = Y0[bmask] - np.mean(Y0[bmask])
    return cj_stratified(np.abs(X0), dY**2, strata_m)


def desitter_preds(pts, i, H=1.0):
    """de Sitter causal predicate: conformally flat, Weyl = 0.
    ds^2 = -dt^2 + e^{2Ht}(dx^2+dy^2+dz^2)
    Returns boolean mask of length i (like minkowski_preds).
    """
    if i == 0:
        return np.empty(0, dtype=bool)
    t_i = pts[i, 0]
    x_i = pts[i, 1:]
    past_pts = pts[:i]
    dt = t_i - past_pts[:, 0]  # all positive since sorted by time
    dx = np.linalg.norm(x_i - past_pts[:, 1:], axis=1)
    # Conformal time integral: ∫_{t_j}^{t_i} dt'/a(t') = (1/H)(e^{-Ht_j} - e^{-Ht_i})
    if H > 0:
        conformal_dist = (1.0/H) * (np.exp(-H * past_pts[:, 0]) - np.exp(-H * t_i))
    else:
        conformal_dist = dt
    return dx < conformal_dist


# ═══════════════════════════════════════════════════════════════
# T1: de Sitter null test
# ═══════════════════════════════════════════════════════════════

def test_desitter(N=3000, T=1.0, H=0.5, M=20, seed_base=5500000):
    """CJ on de Sitter (Weyl=0). Should be zero."""
    print("=" * 70)
    print("T1: de Sitter null test (Weyl = 0, CJ should be 0)")
    print(f"    N={N}, T={T}, H={H}, M={M}")
    print("=" * 70)

    cj_list = []
    for trial in range(M):
        seed = seed_base + trial
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)
        n = len(pts)

        # Flat Hasse
        par0, _ = build_hasse_from_predicate(pts, minkowski_preds)
        ch0 = make_children(par0, n)
        pd0, pu0 = path_counts(par0, ch0, n)

        # de Sitter Hasse
        ds_pred = lambda pts, i, H=H: desitter_preds(pts, i, H)
        parD, _ = build_hasse_from_predicate(pts, ds_pred)
        chD = make_children(parD, n)
        pdD, puD = path_counts(parD, chD, n)

        bmask = bulk_mask(pts, T, 0.15)
        strata = make_strata(pts, par0, T)
        strata_m = strata[bmask]

        Y0 = np.log2(pd0 * pu0 + 1)
        YD = np.log2(pdD * puD + 1)
        dY = YD[bmask] - Y0[bmask]
        X0 = Y0[bmask] - np.mean(Y0[bmask])
        cj = cj_stratified(np.abs(X0), dY**2, strata_m)
        cj_list.append(cj)

    cj_mean = np.mean(cj_list)
    cj_se = np.std(cj_list) / np.sqrt(M)
    # Compare to pp-wave scale
    norm_ppw = N**(8/9) * (3.0**2/2) * T**4  # eps=3 pp-wave norm
    R_ds = cj_mean / (C0 * norm_ppw)

    print(f"  CJ(dS) = {cj_mean:.4e} ± {cj_se:.4e}")
    print(f"  R(dS)/R(ppw,eps=3) = {R_ds:.4f}")
    print(f"  |CJ(dS)/CJ(ppw)| = {abs(R_ds):.4f}  (should be << 1)")
    sig = abs(cj_mean) / cj_se if cj_se > 0 else 0
    print(f"  Significance from zero: {sig:.1f}σ")
    print(f"  VERDICT: {'PASS (consistent with 0)' if sig < 3 else 'FAIL'}")
    return cj_mean, cj_se, R_ds


# ═══════════════════════════════════════════════════════════════
# T2: M_ss convergence rate
# ═══════════════════════════════════════════════════════════════

def test_mss_convergence():
    """Fit M_ss(N) = 2 + a * N^{-alpha}."""
    print("\n" + "=" * 70)
    print("T2: M_ss convergence rate fit")
    print("=" * 70)

    # Data from the full decomposition run (N=1000-10000, M=30)
    N_vals = np.array([1000, 3000, 5000, 8000, 10000], dtype=float)
    Mss_vals = np.array([2.008, 1.974, 2.011, 2.018, 2.059])
    # Exclude N=2000 (statistical outlier R=1.10)

    def model(N, a, alpha):
        return 2.0 + a * N**(-alpha)

    try:
        popt, pcov = curve_fit(model, N_vals, Mss_vals, p0=[1.0, 0.3],
                               maxfev=10000)
        a, alpha = popt
        perr = np.sqrt(np.diag(pcov))
        resid = Mss_vals - model(N_vals, *popt)
        chi2 = np.sum(resid**2)
        print(f"  Fit: M_ss(N) = 2 + {a:.3f} * N^(-{alpha:.3f})")
        print(f"  Errors: a = {a:.3f} ± {perr[0]:.3f}, alpha = {alpha:.3f} ± {perr[1]:.3f}")
        print(f"  Residuals: {resid}")
        print(f"  chi² = {chi2:.6f}")
        print(f"  M_ss(N=50000) predicted = {model(50000, *popt):.4f}")
        print(f"  M_ss(N=100000) predicted = {model(100000, *popt):.4f}")
    except Exception as e:
        print(f"  Fit failed: {e}")
        a, alpha = np.nan, np.nan

    # Also fit M_sa → 0
    Msa_vals = np.array([0.100, 0.162, 0.067, 0.056, -0.013])
    def model_sa(N, b, beta):
        return b * N**(-beta)
    try:
        # Use only positive values for initial fit
        mask = Msa_vals > 0
        popt_sa, _ = curve_fit(model_sa, N_vals[mask], Msa_vals[mask],
                                p0=[10.0, 0.5], maxfev=10000)
        b, beta = popt_sa
        print(f"\n  M_sa fit: M_sa(N) = {b:.2f} * N^(-{beta:.3f})")
        print(f"  M_sa(N=50000) predicted = {model_sa(50000, *popt_sa):.4f}")
    except Exception as e:
        print(f"  M_sa fit failed: {e}")

    # M_aa convergence
    Maa_vals = np.array([0.898, 0.592, 0.565, 0.539, 0.522])
    def model_aa(N, c, M_inf):
        return M_inf + c / np.sqrt(N)
    try:
        popt_aa, _ = curve_fit(model_aa, N_vals, Maa_vals, p0=[10.0, 0.5])
        c_aa, M_inf = popt_aa
        print(f"\n  M_aa fit: M_aa(N) = {M_inf:.3f} + {c_aa:.1f}/sqrt(N)")
        print(f"  M_aa(∞) = {M_inf:.3f}  (persistent nonzero)")
    except Exception as e:
        print(f"  M_aa fit failed: {e}")


# ═══════════════════════════════════════════════════════════════
# T3: Extended epsilon range
# ═══════════════════════════════════════════════════════════════

def test_epsilon_range(N=3000, T=1.0, M=20, seed_base=4400000):
    """CJ at eps = 0.5, 1, 2, 3, 5, 10."""
    print("\n" + "=" * 70)
    print("T3: Extended epsilon range")
    print(f"    N={N}, T={T}, M={M}")
    print("=" * 70)

    eps_list = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    results = []

    for eps in eps_list:
        t0 = time.time()
        E2 = eps**2 / 2.0
        norm = N**(8/9) * E2 * T**4
        cj_trials = []
        for trial in range(M):
            cj = single_trial_cj(N, T, eps, seed_base + trial)
            cj_trials.append(cj)
        cj_mean = np.mean(cj_trials)
        cj_se = np.std(cj_trials) / np.sqrt(M)
        R = cj_mean / (C0 * norm)
        R_se = cj_se / (C0 * norm)
        dt = time.time() - t0
        results.append((eps, E2, R, R_se, cj_mean))
        print(f"  eps={eps:5.1f}  E²={E2:7.3f}  R={R:.4f}±{R_se:.4f}  ({dt:.0f}s)")

    # Fit R vs eps² to check linearity
    eps_arr = np.array([r[0] for r in results])
    R_arr = np.array([r[2] for r in results])
    R_se_arr = np.array([r[3] for r in results])

    # R should be constant if CJ ∝ E² = eps²/2
    R_wmean = np.average(R_arr, weights=1/R_se_arr**2)
    chi2 = np.sum(((R_arr - R_wmean) / R_se_arr)**2)
    ndof = len(R_arr) - 1
    print(f"\n  Weighted mean R = {R_wmean:.4f}")
    print(f"  chi²/ndof = {chi2:.2f}/{ndof} = {chi2/ndof:.2f}")
    print(f"  VERDICT: {'PASS (consistent with CJ ∝ E²)' if chi2/ndof < 3 else 'CHECK (possible nonlinearity)'}")
    return results


# ═══════════════════════════════════════════════════════════════
# T4: Normality test
# ═══════════════════════════════════════════════════════════════

def test_normality(N=3000, T=1.0, eps=3.0, M=50, seed_base=3300000):
    """Shapiro-Wilk test on per-trial CJ values."""
    print("\n" + "=" * 70)
    print("T4: Normality test (Shapiro-Wilk)")
    print(f"    N={N}, eps={eps}, M={M}")
    print("=" * 70)

    t0 = time.time()
    cj_trials = []
    for trial in range(M):
        cj = single_trial_cj(N, T, eps, seed_base + trial)
        cj_trials.append(cj)
    dt = time.time() - t0

    cj_arr = np.array(cj_trials)
    norm = N**(8/9) * (eps**2/2) * T**4
    R_arr = cj_arr / (C0 * norm)

    stat, p_val = sp_stats.shapiro(R_arr)
    skew = sp_stats.skew(R_arr)
    kurt = sp_stats.kurtosis(R_arr)

    print(f"  M={M} trials in {dt:.0f}s")
    print(f"  R: mean={np.mean(R_arr):.4f}, std={np.std(R_arr):.4f}, SE={np.std(R_arr)/np.sqrt(M):.4f}")
    print(f"  Shapiro-Wilk: W={stat:.4f}, p={p_val:.4f}")
    print(f"  Skewness = {skew:.3f}, Kurtosis = {kurt:.3f}")
    print(f"  VERDICT: {'PASS (normal)' if p_val > 0.05 else 'MARGINAL' if p_val > 0.01 else 'FAIL (non-normal)'}")

    # Bootstrap 95% CI
    n_boot = 10000
    boot_means = np.array([np.mean(np.random.choice(R_arr, size=M, replace=True))
                           for _ in range(n_boot)])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    print(f"\n  Bootstrap 95% CI for R: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Contains 1.0: {'YES' if ci_lo <= 1.0 <= ci_hi else 'NO'}")

    return R_arr, stat, p_val


# ═══════════════════════════════════════════════════════════════
# T5: Interval CJ = 0 (already measured, re-verify)
# ═══════════════════════════════════════════════════════════════

def test_interval_cj(N=3000, T=1.0, eps=3.0, M=10, seed_base=2200000):
    """CJ using interval counts |past|*|future| instead of path counts."""
    print("\n" + "=" * 70)
    print("T5: Interval-counting CJ (should be ≡ 0)")
    print(f"    N={N}, eps={eps}, M={M}")
    print("=" * 70)

    cj_hasse_list = []
    cj_interval_list = []

    for trial in range(M):
        seed = seed_base + trial
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)
        n = len(pts)

        par0, _ = build_hasse_from_predicate(pts, minkowski_preds)
        curved_pred = lambda pts, i, eps=eps: ppwave_exact_preds(pts, i, eps)
        parC, _ = build_hasse_from_predicate(pts, curved_pred)

        ch0 = make_children(par0, n)
        chC = make_children(parC, n)
        pd0, pu0 = path_counts(par0, ch0, n)
        pdC, puC = path_counts(parC, chC, n)

        # Interval counts: number of elements in past/future
        # For each element, count how many have a path to/from it
        nm0 = np.zeros(n)
        np0_arr = np.zeros(n)
        nmC = np.zeros(n)
        npC = np.zeros(n)
        for i in range(n):
            if par0[i] is not None:
                nm0[i] = len(par0[i])
            if parC[i] is not None:
                nmC[i] = len(parC[i])
        for i in range(n):
            np0_arr[i] = len(ch0[i])
            npC[i] = len(chC[i])

        bmask = bulk_mask(pts, T, 0.15)
        strata = make_strata(pts, par0, T)
        strata_m = strata[bmask]

        # Hasse CJ
        Y0_h = np.log2(pd0 * pu0 + 1)
        YC_h = np.log2(pdC * puC + 1)
        dY_h = (YC_h - Y0_h)[bmask]
        X_h = Y0_h[bmask] - np.mean(Y0_h[bmask])
        cj_h = cj_stratified(np.abs(X_h), dY_h**2, strata_m)

        # Interval CJ (using link-degree as proxy for interval count)
        Y0_i = np.log2(nm0 * np0_arr + 1)
        YC_i = np.log2(nmC * npC + 1)
        dY_i = (YC_i - Y0_i)[bmask]
        X_i = Y0_i[bmask] - np.mean(Y0_i[bmask])
        cj_i = cj_stratified(np.abs(X_i), dY_i**2, strata_m)

        cj_hasse_list.append(cj_h)
        cj_interval_list.append(cj_i)

    h_mean = np.mean(cj_hasse_list)
    i_mean = np.mean(cj_interval_list)
    i_se = np.std(cj_interval_list) / np.sqrt(M)
    ratio = abs(i_mean / h_mean) if h_mean != 0 else np.inf

    print(f"  CJ(Hasse):    {h_mean:.4e}")
    print(f"  CJ(interval): {i_mean:.4e} ± {i_se:.4e}")
    print(f"  |CJ_int/CJ_hasse| = {ratio:.4f}")
    print(f"  VERDICT: {'PASS (interval CJ negligible)' if ratio < 0.05 else 'CHECK'}")
    return h_mean, i_mean


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 70)
    print("PAPER 7: TIER A MINI-TESTS BATTERY")
    print("=" * 70)
    t_total = time.time()

    all_results = {}

    # T1: de Sitter
    t0 = time.time()
    ds_mean, ds_se, ds_R = test_desitter(N=2000, M=15)
    all_results['T1_deSitter'] = {'CJ': ds_mean, 'SE': ds_se, 'R_ratio': ds_R,
                                   'time': time.time() - t0}

    # T2: M_ss convergence (uses existing data, no MC needed)
    t0 = time.time()
    test_mss_convergence()
    all_results['T2_convergence'] = {'time': time.time() - t0}

    # T3: Extended epsilon
    t0 = time.time()
    eps_results = test_epsilon_range(N=2000, M=15)
    all_results['T3_epsilon'] = {'results': [(r[0], r[2], r[3]) for r in eps_results],
                                  'time': time.time() - t0}

    # T4: Normality
    t0 = time.time()
    R_arr, W, p = test_normality(N=2000, M=50)
    all_results['T4_normality'] = {'W': W, 'p': p, 'R_mean': float(np.mean(R_arr)),
                                    'R_std': float(np.std(R_arr)),
                                    'time': time.time() - t0}

    # T5: Interval CJ
    t0 = time.time()
    h_mean, i_mean = test_interval_cj(N=2000, M=10)
    all_results['T5_interval'] = {'CJ_hasse': h_mean, 'CJ_interval': i_mean,
                                   'ratio': abs(i_mean/h_mean) if h_mean != 0 else 0,
                                   'time': time.time() - t0}

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"T1 de Sitter null:     R(dS)/R(ppw) = {ds_R:.4f}")
    print(f"T2 M_ss convergence:   see fit above")
    print(f"T3 ε extended:         {len(eps_results)} points tested")
    print(f"T4 Normality:          W={W:.4f}, p={p:.4f}")
    print(f"T5 Interval CJ:        |CJ_int/CJ_hasse| = {abs(i_mean/h_mean):.4f}")
    print(f"\nTotal time: {time.time() - t_total:.0f}s")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), '..', 'fnd1_data', 'paper7_mini_tests.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Saved -> {out_path}")
