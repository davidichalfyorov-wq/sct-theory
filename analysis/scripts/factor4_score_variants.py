#!/usr/bin/env python3
"""
Test whether factor 4 depends on the link score definition.

Variants:
  V0: Y = log2(pd*pu + 1)              [standard]
  V1: Y = log2(pd + 1) + log2(pu + 1)  [separated log]
  V2: Y = (pd*pu)^(1/4)                [4th root of product]
  V3: Y = log2(pd + 1)                 [one-leg only, past]
  V4: Y = log2(pu + 1)                 [one-leg only, future]

For each variant, measure CJ and compute R = CJ / (C0 * N^{8/9} * E^2 * T^4).
The ratios between variants reveal what depends on the score definition.
"""
import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, ppwave_exact_preds,
    build_hasse_from_predicate, bulk_mask,
)

C0 = 32 * np.pi**2 / (3 * 362880 * 45)


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


def run_test(N=3000, T=1.0, eps=3.0, M=20, seed_base=6600000):
    E2 = eps**2 / 2.0
    zeta = 0.15
    norm = N**(8/9) * E2 * T**4

    results = {f'V{i}': [] for i in range(5)}

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

        # V0: Standard Y = log2(pd*pu + 1)
        Y0_v0 = np.log2(pd0 * pu0 + 1)
        YC_v0 = np.log2(pdC * puC + 1)
        dY_v0 = (YC_v0 - Y0_v0)[bmask]
        X_v0 = Y0_v0[bmask] - np.mean(Y0_v0[bmask])
        cj_v0 = cj_stratified(np.abs(X_v0), dY_v0**2, strata_m)
        results['V0'].append(cj_v0)

        # V1: Y = log2(pd+1) + log2(pu+1) (separated logs)
        Y0_v1 = np.log2(pd0 + 1) + np.log2(pu0 + 1)
        YC_v1 = np.log2(pdC + 1) + np.log2(puC + 1)
        dY_v1 = (YC_v1 - Y0_v1)[bmask]
        X_v1 = Y0_v1[bmask] - np.mean(Y0_v1[bmask])
        cj_v1 = cj_stratified(np.abs(X_v1), dY_v1**2, strata_m)
        results['V1'].append(cj_v1)

        # V2: Y = (pd*pu)^(1/4)
        Y0_v2 = (pd0 * pu0)**0.25
        YC_v2 = (pdC * puC)**0.25
        dY_v2 = (YC_v2 - Y0_v2)[bmask]
        X_v2 = Y0_v2[bmask] - np.mean(Y0_v2[bmask])
        cj_v2 = cj_stratified(np.abs(X_v2), dY_v2**2, strata_m)
        results['V2'].append(cj_v2)

        # V3: Y = log2(pd + 1) [one-leg past only]
        Y0_v3 = np.log2(pd0 + 1)
        YC_v3 = np.log2(pdC + 1)
        dY_v3 = (YC_v3 - Y0_v3)[bmask]
        X_v3 = Y0_v3[bmask] - np.mean(Y0_v3[bmask])
        cj_v3 = cj_stratified(np.abs(X_v3), dY_v3**2, strata_m)
        results['V3'].append(cj_v3)

        # V4: Y = log2(pu + 1) [one-leg future only]
        Y0_v4 = np.log2(pu0 + 1)
        YC_v4 = np.log2(puC + 1)
        dY_v4 = (YC_v4 - Y0_v4)[bmask]
        X_v4 = Y0_v4[bmask] - np.mean(Y0_v4[bmask])
        cj_v4 = cj_stratified(np.abs(X_v4), dY_v4**2, strata_m)
        results['V4'].append(cj_v4)

    print(f"N={N}, M={M}, eps={eps}")
    print(f"{'Variant':>10}  {'mean_CJ':>12}  {'R=CJ/C0norm':>12}  {'CJ/CJ_V0':>10}")
    for v in ['V0', 'V1', 'V2', 'V3', 'V4']:
        mean_cj = np.mean(results[v])
        R = mean_cj / (C0 * norm)
        ratio = mean_cj / np.mean(results['V0']) if np.mean(results['V0']) != 0 else 0
        labels = {
            'V0': 'log2(pd*pu+1)',
            'V1': 'log2(pd+1)+log2(pu+1)',
            'V2': '(pd*pu)^{1/4}',
            'V3': 'log2(pd+1) [past]',
            'V4': 'log2(pu+1) [future]',
        }
        print(f"{v:>10}  {mean_cj:>12.4e}  {R:>12.4f}  {ratio:>10.4f}  {labels[v]}")

    # Key test: V0 vs V1
    v0 = np.mean(results['V0'])
    v1 = np.mean(results['V1'])
    print(f"\nV0 vs V1 ratio: {v1/v0:.4f}")
    print("  If ~1.0: log2(pd*pu+1) ≈ log2(pd+1)+log2(pu+1) [+1 negligible]")

    # Key test: V0 vs (V3 + V4)
    v3 = np.mean(results['V3'])
    v4 = np.mean(results['V4'])
    print(f"\nV3 (past-only CJ): {v3:.4e}, V4 (future-only CJ): {v4:.4e}")
    print(f"V3+V4 = {v3+v4:.4e}, V0 = {v0:.4e}")
    print(f"V0/(V3+V4) = {v0/(v3+v4):.4f}" if (v3+v4) != 0 else "V3+V4 = 0")
    print(f"V0/V3 = {v0/v3:.4f}" if v3 != 0 else "V3 = 0")

    return results


if __name__ == '__main__':
    print("=" * 80)
    print("FACTOR-4: SCORE VARIANT TEST")
    print("Does the factor depend on the link score definition?")
    print("=" * 80)
    for N in [2000, 3000]:
        print()
        t0 = time.time()
        run_test(N=N, M=25)
        print(f"Time: {time.time()-t0:.0f}s")
