#!/usr/bin/env python3
"""
DECISIVE TEST for the factor-4 problem.

Key question: Does the |X|-weighted past-future correlation ρ_eff equal 1.0?

If ρ_eff = Cov_B(|X|, g₋g₊) / Cov_B(|X|, g₋²) ≈ 1.0,
then factor 4 = 2(1 + ρ_eff) = 4 exactly, even though
the unweighted ρ = ⟨g₋g₊⟩/⟨g₋²⟩ ≈ 0.51.

This would mean: the |X| weight selects elements where
past and future responses are perfectly correlated.
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, ppwave_exact_preds,
    build_hasse_from_predicate, bulk_mask,
)

C0 = 32 * np.pi**2 / (3 * 362880 * 45)
C_AN = C0 / 4


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


def make_strata(pts, par, T, K_time=5, K_rad=3, K_depth=3):
    n = len(pts)
    tau_hat = 2.0 * pts[:, 0] / T
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2.0 - np.abs(pts[:, 0])
    rho_hat = np.clip(r / np.maximum(rmax, 1e-12), 0, 0.999999)
    tau_bin = np.clip(np.floor((tau_hat + 1) * K_time / 2).astype(int), 0, K_time - 1)
    rho_bin = np.clip(np.floor(rho_hat * K_rad).astype(int), 0, K_rad - 1)
    depth = compute_depth(par, n)
    max_d = max(int(depth.max()), 1)
    depth_terc = np.clip((depth * K_depth) // (max_d + 1), 0, K_depth - 1)
    return tau_bin * (K_rad * K_depth) + rho_bin * K_depth + depth_terc


def cj_stratified(weight_arr, response_arr, strata_m, min_bin=3):
    total = 0.0
    for b in np.unique(strata_m):
        idx = strata_m == b
        if idx.sum() < min_bin:
            continue
        w = idx.sum() / len(weight_arr)
        cov = (np.mean(weight_arr[idx] * response_arr[idx])
               - np.mean(weight_arr[idx]) * np.mean(response_arr[idx]))
        total += w * cov
    return float(total)


def run_test(N, T=1.0, eps=3.0, M=30, seed_base=7700000):
    E2 = eps**2 / 2.0
    zeta = 0.15

    cj_full_list = []
    cj_mm_list = []
    cj_pp_list = []
    cj_mp_list = []
    rho_eff_list = []
    rho_unw_list = []

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

        bmask = bulk_mask(pts, T, zeta)
        strata = make_strata(pts, par0, T)
        strata_m = strata[bmask]

        # Full Y
        Y0 = np.log2(pd0 * pu0 + 1)
        YC = np.log2(pdC * puC + 1)
        dY = YC[bmask] - Y0[bmask]
        X0 = Y0[bmask] - np.mean(Y0[bmask])
        absX = np.abs(X0)
        dY2 = dY**2

        # Decomposed: g_minus = delta(log2(p_down)), g_plus = delta(log2(p_up))
        # Careful: avoid log(0) by adding small eps to path counts
        safe = 1e-30
        g_minus = np.log2(pdC[bmask] + safe) - np.log2(pd0[bmask] + safe)
        g_plus = np.log2(puC[bmask] + safe) - np.log2(pu0[bmask] + safe)

        # Full CJ
        cj_full = cj_stratified(absX, dY2, strata_m)

        # Three kernel components
        cj_mm = cj_stratified(absX, g_minus**2, strata_m)
        cj_pp = cj_stratified(absX, g_plus**2, strata_m)
        cj_mp = cj_stratified(absX, g_minus * g_plus, strata_m)

        # Effective |X|-weighted correlation
        if cj_mm > 0:
            rho_eff = cj_mp / cj_mm
        else:
            rho_eff = np.nan

        # Unweighted correlation
        gm2 = np.mean(g_minus**2)
        gmp = np.mean(g_minus * g_plus)
        rho_unw = gmp / gm2 if gm2 > 0 else np.nan

        cj_full_list.append(cj_full)
        cj_mm_list.append(cj_mm)
        cj_pp_list.append(cj_pp)
        cj_mp_list.append(cj_mp)
        rho_eff_list.append(rho_eff)
        rho_unw_list.append(rho_unw)

    # Averages
    cj_full_mean = np.mean(cj_full_list)
    cj_mm_mean = np.mean(cj_mm_list)
    cj_pp_mean = np.mean(cj_pp_list)
    cj_mp_mean = np.mean(cj_mp_list)
    rho_eff_mean = np.nanmean(rho_eff_list)
    rho_eff_std = np.nanstd(rho_eff_list) / np.sqrt(M)
    rho_unw_mean = np.nanmean(rho_unw_list)

    # Normalization
    norm = N**(8/9) * E2 * T**4
    R_full = cj_full_mean / (C0 * norm)

    # Decomposed sum and ratio to C_AN
    decomp_sum = cj_mm_mean + 2 * cj_mp_mean + cj_pp_mean
    R_decomp = decomp_sum / (C0 * norm)

    # The effective factor
    # CJ_full = Cov(|X|, g-^2) + 2*Cov(|X|,g-g+) + Cov(|X|,g+^2)
    #         = Cov(|X|,g-^2) * (1 + 2*rho_eff + sigma_ratio)
    sigma_ratio = cj_pp_mean / cj_mm_mean if cj_mm_mean != 0 else np.nan
    effective_factor = cj_full_mean / cj_mm_mean if cj_mm_mean != 0 else np.nan

    return {
        'N': N, 'M': M,
        'R_full': R_full,
        'R_decomp': R_decomp,
        'cj_mm': cj_mm_mean, 'cj_pp': cj_pp_mean, 'cj_mp': cj_mp_mean,
        'rho_eff': rho_eff_mean, 'rho_eff_std': rho_eff_std,
        'rho_unw': rho_unw_mean,
        'sigma_ratio': sigma_ratio,
        'effective_factor': effective_factor,
        'sum_over_CAN': (cj_mm_mean + 2*cj_mp_mean + cj_pp_mean) / (C_AN * norm),
    }


if __name__ == '__main__':
    print("=" * 80)
    print("DECISIVE TEST: Factor-4 via |X|-weighted correlation")
    print("=" * 80)
    print(f"C_0 = {C0:.6e}, C_AN = C_0/4 = {C_AN:.6e}")
    print()

    results = []
    for N in [1000, 2000, 3000, 5000]:
        t0 = time.time()
        M = 30 if N <= 3000 else 20
        r = run_test(N, M=M)
        dt = time.time() - t0
        results.append(r)
        print(f"N={N:5d} ({dt:.0f}s):")
        print(f"  R_full     = {r['R_full']:.4f}")
        print(f"  R_decomp   = {r['R_decomp']:.4f}")
        print(f"  rho_eff    = {r['rho_eff']:.4f} +/- {r['rho_eff_std']:.4f}  (|X|-weighted)")
        print(f"  rho_unw    = {r['rho_unw']:.4f}  (unweighted)")
        print(f"  sigma_r    = {r['sigma_ratio']:.4f}  (Cov(|X|,g+^2)/Cov(|X|,g-^2))")
        print(f"  eff_factor = {r['effective_factor']:.4f}  (CJ_full/Cov(|X|,g-^2))")
        print(f"  sum/C_AN   = {r['sum_over_CAN']:.4f}  (K--+2K-++K++, target=4)")
        print()

    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'N':>6}  {'R_full':>7}  {'rho_eff':>8}  {'rho_unw':>8}  {'eff_fac':>8}  {'sum/CAN':>8}")
    for r in results:
        print(f"{r['N']:>6}  {r['R_full']:>7.3f}  {r['rho_eff']:>8.4f}  {r['rho_unw']:>8.4f}  {r['effective_factor']:>8.3f}  {r['sum_over_CAN']:>8.3f}")

    print()
    print("KEY PREDICTION: If rho_eff -> 1.0, factor 4 is explained by")
    print("|X|-weighted perfect correlation of past and future responses.")
    print(f"Factor = 1 + 2*rho_eff + sigma_ratio")
    for r in results:
        pred = 1 + 2 * r['rho_eff'] + r['sigma_ratio']
        print(f"  N={r['N']}: 1 + 2*{r['rho_eff']:.3f} + {r['sigma_ratio']:.3f} = {pred:.3f} (actual: {r['effective_factor']:.3f})")
