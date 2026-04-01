#!/usr/bin/env python3
"""
Measure the CJ kernel profile K(s) directly.

K(s) = contribution to CJ from elements at normalized position s.

If CJ = ∫₀¹ K(s) ds × (8π/15)E² × T⁴ × N^{8/9},
and K(s) = c₄² × k₄(s)k₄(1-s) × F(s),
then F(s) encodes the "factor 4" as ∫F(s)·k₄k₄ ds / ∫k₄k₄ ds = 4.

Also test:
1. BD-like CJ using interval counts n₋, n₊ instead of path counts
2. The ratio CJ_hasse / CJ_interval to see if factor 4 is Hasse vs interval
3. Profile of the effective weight Cov(|X|, δY²|s) vs s
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
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


def interval_counts(C, n):
    """Count elements in past and future of each element using causal matrix."""
    n_minus = np.sum(C, axis=1).astype(float)  # number of elements in past
    n_plus = np.sum(C, axis=0).astype(float)   # number of elements in future
    return n_minus, n_plus


def build_causal_matrix(pts, pred_func):
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        preds = pred_func(pts, i)
        if preds is not None:
            for j in preds:
                C[j, i] = 1  # j is in past of i
    return C


def make_children(par, n):
    ch = [[] for _ in range(n)]
    for i in range(n):
        if par[i] is not None:
            for j in par[i]:
                ch[int(j)].append(i)
    return ch


def cj_by_sbin(weight, response, s_vals, n_bins=20, s_range=(0.05, 0.95)):
    """Compute CJ contribution from each s-bin."""
    edges = np.linspace(s_range[0], s_range[1], n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    contributions = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = (s_vals >= edges[b]) & (s_vals < edges[b + 1])
        nb = mask.sum()
        counts[b] = nb
        if nb < 5:
            continue
        w_b = nb / len(weight)
        cov = (np.mean(weight[mask] * response[mask])
               - np.mean(weight[mask]) * np.mean(response[mask]))
        contributions[b] = w_b * cov

    return centers, contributions, counts


def run_profile(N=3000, T=1.0, eps=3.0, M=20, seed_base=8800000):
    E2 = eps**2 / 2.0
    zeta = 0.15
    norm = N**(8/9) * E2 * T**4

    all_profiles_hasse = []
    all_profiles_interval = []
    cj_hasse_list = []
    cj_interval_list = []

    for trial in range(M):
        seed = seed_base + trial
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)
        n = len(pts)

        # Build Hasse
        par0, _ = build_hasse_from_predicate(pts, minkowski_preds)
        curved_pred = lambda pts, i, eps=eps: ppwave_exact_preds(pts, i, eps)
        parC, _ = build_hasse_from_predicate(pts, curved_pred)

        ch0 = make_children(par0, n)
        chC = make_children(parC, n)

        # Hasse path counts
        pd0, pu0 = path_counts(par0, ch0, n)
        pdC, puC = path_counts(parC, chC, n)

        # Build causal matrices for interval counts
        C0_mat = build_causal_matrix(pts, minkowski_preds)
        CC_mat = build_causal_matrix(pts, curved_pred)

        # Interval counts
        nm0, np0 = interval_counts(C0_mat, n)
        nmC, npC = interval_counts(CC_mat, n)

        bmask = bulk_mask(pts, T, zeta)

        # Normalized position s = (t + T/2) / T
        s_vals = (pts[bmask, 0] + T / 2) / T

        # === Hasse-based CJ ===
        Y0_h = np.log2(pd0 * pu0 + 1)
        YC_h = np.log2(pdC * puC + 1)
        dY_h = YC_h[bmask] - Y0_h[bmask]
        X0_h = Y0_h[bmask] - np.mean(Y0_h[bmask])
        absX_h = np.abs(X0_h)

        # === Interval-based CJ ===
        Y0_i = np.log2(nm0 * np0 + 1)
        YC_i = np.log2(nmC * npC + 1)
        dY_i = YC_i[bmask] - Y0_i[bmask]
        X0_i = Y0_i[bmask] - np.mean(Y0_i[bmask])
        absX_i = np.abs(X0_i)

        # Full CJ
        cj_h = np.mean(absX_h * dY_h**2) - np.mean(absX_h) * np.mean(dY_h**2)
        cj_i = np.mean(absX_i * dY_i**2) - np.mean(absX_i) * np.mean(dY_i**2)
        cj_hasse_list.append(cj_h)
        cj_interval_list.append(cj_i)

        # Kernel profiles
        centers, prof_h, _ = cj_by_sbin(absX_h, dY_h**2, s_vals)
        _, prof_i, _ = cj_by_sbin(absX_i, dY_i**2, s_vals)
        all_profiles_hasse.append(prof_h)
        all_profiles_interval.append(prof_i)

    # Average results
    cj_h_mean = np.mean(cj_hasse_list)
    cj_i_mean = np.mean(cj_interval_list)
    R_hasse = cj_h_mean / (C0 * norm)
    R_interval = cj_i_mean / (C0 * norm)
    ratio_hi = cj_h_mean / cj_i_mean if cj_i_mean != 0 else np.nan

    prof_h_mean = np.mean(all_profiles_hasse, axis=0)
    prof_i_mean = np.mean(all_profiles_interval, axis=0)

    # Theoretical kernel for comparison
    k4k4 = centers**4 * (1 - centers)**4 / (24**2)  # k₄(s)k₄(1-s)

    print(f"N={N}, M={M}, eps={eps}")
    print(f"  CJ (Hasse paths):   R = {R_hasse:.4f}")
    print(f"  CJ (interval count): R = {R_interval:.4f}")
    print(f"  Ratio Hasse/Interval = {ratio_hi:.4f}")
    print()

    # Profile comparison
    print(f"{'s':>6}  {'K_hasse':>10}  {'K_interval':>10}  {'ratio':>8}  {'k4k4':>10}  {'K_h/k4k4':>10}")
    for i, s in enumerate(centers):
        if k4k4[i] > 1e-15:
            r_h = prof_h_mean[i] / k4k4[i] if k4k4[i] > 1e-15 else 0
        else:
            r_h = 0
        r_hi = prof_h_mean[i] / prof_i_mean[i] if abs(prof_i_mean[i]) > 1e-20 else 0
        print(f"{s:>6.3f}  {prof_h_mean[i]:>10.3e}  {prof_i_mean[i]:>10.3e}  {r_hi:>8.3f}  {k4k4[i]:>10.3e}  {r_h:>10.2f}")

    # Integral comparison
    ds = centers[1] - centers[0]
    int_h = np.sum(prof_h_mean) * ds
    int_i = np.sum(prof_i_mean) * ds
    int_k4k4 = np.sum(k4k4) * ds
    print(f"\nIntegrals (ds={ds:.4f}):")
    print(f"  ∫K_hasse ds   = {int_h:.6e}")
    print(f"  ∫K_interval ds = {int_i:.6e}")
    print(f"  ∫k4k4 ds       = {int_k4k4:.6e} (theory: 1/9! = {1/362880:.6e})")
    print(f"  ∫K_hasse/∫k4k4 = {int_h/int_k4k4:.4f} (should be ~factor_4 * c4^2 * angular * vol)")

    return centers, prof_h_mean, prof_i_mean, k4k4


if __name__ == '__main__':
    print("=" * 80)
    print("CJ KERNEL PROFILE: Hasse paths vs Interval counts")
    print("=" * 80)

    for N in [2000, 3000]:
        print(f"\n{'='*80}")
        t0 = time.time()
        run_profile(N=N, M=20)
        print(f"Time: {time.time()-t0:.0f}s")
