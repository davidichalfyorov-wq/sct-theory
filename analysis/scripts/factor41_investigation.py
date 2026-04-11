#!/usr/bin/env python3
"""
Factor 4.1 Investigation: Why does the analytical prefactor
C_an = 8pi^2/(3*9!*45) differ from data-fitted C_0 by ~4.1x?

Strategy: systematically modify CJ definition to isolate the source.

Variants tested:
  V0: Standard CJ = sum w_B Cov_B(|X|, dY^2)  [baseline]
  V1: Replace Y=log2(pd*pu+1) with Y=pd*pu     [remove log]
  V2: Replace Y=log2(pd*pu+1) with Y=pd*pu+1   [remove log, keep +1]
  V3: No centering: use |Y0| instead of |Y0-mean(Y0)|
  V4: No absolute value: use X^2 instead of |X|
  V5: No stratification: single global bin
  V6: Raw product: mean(pd_flat * pu_flat * dY^2) [direct kernel test]
  V7: Proper-time kernel: weight by (tau-^4 * tau+^4) instead of |X|
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, ppwave_exact_preds,
    build_hasse_from_predicate, bulk_mask,
)

C_AN = 8 * np.pi**2 / (3 * 362880 * 45)  # analytical prefactor


def path_counts_raw(par, ch_list, n):
    """Path counts WITHOUT +1 (pure Hasse path count)."""
    pd = np.zeros(n, dtype=np.float64)
    pu = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if par[i] is not None and len(par[i]) > 0:
            pd[i] = np.sum(pd[list(par[i])]) + 1  # +1 convention
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


def cj_stratified(weight_arr, response_arr, strata_m, min_bin=3):
    """Generic stratified covariance: sum w_B Cov_B(weight, response)."""
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


def run_investigation(N=2000, T=1.0, eps=3.0, M=20, seed_base=9800000):
    """Run all variants and compare prefactors."""
    print("=" * 70)
    print(f"FACTOR 4.1 INVESTIGATION: N={N}, eps={eps}, T={T}, M={M}")
    print(f"Analytical prefactor C_an = {C_AN:.6e}")
    print("=" * 70)

    E2 = eps**2 / 2.0
    zeta = 0.15
    results = {f"V{i}": [] for i in range(8)}

    for trial in range(M):
        seed = seed_base + trial
        t0 = time.time()

        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)
        n = len(pts)

        # Build Hasse diagrams
        par0, _ = build_hasse_from_predicate(pts, minkowski_preds)
        curved_pred = lambda pts, i, eps=eps: ppwave_exact_preds(pts, i, eps)
        parC, _ = build_hasse_from_predicate(pts, curved_pred)

        ch0 = make_children(par0, n)
        chC = make_children(parC, n)

        # Path counts (with +1 convention)
        pd0, pu0 = path_counts_raw(par0, ch0, n)
        pdC, puC = path_counts_raw(parC, chC, n)

        # Various Y definitions
        Y0_standard = np.log2(pd0 * pu0 + 1)       # standard
        YC_standard = np.log2(pdC * puC + 1)
        Y0_nolog = pd0 * pu0                         # no log
        YC_nolog = pdC * puC
        Y0_nolog_p1 = pd0 * pu0 + 1                  # no log, keep +1
        YC_nolog_p1 = pdC * puC + 1

        bmask = bulk_mask(pts, T, zeta)
        strata = make_strata(pts, par0, T)
        strata_m = strata[bmask]

        # Proper times for V7
        t_vals = pts[bmask, 0]
        r_vals = np.linalg.norm(pts[bmask, 1:], axis=1)
        tau_m_sq = np.maximum((t_vals + T / 2)**2 - r_vals**2, 0)
        tau_p_sq = np.maximum((T / 2 - t_vals)**2 - r_vals**2, 0)
        tau_kernel = (tau_m_sq / T**2)**2 * (tau_p_sq / T**2)**2  # (tau-/T)^4*(tau+/T)^4

        # === V0: Standard CJ ===
        delta0 = YC_standard - Y0_standard
        X0 = Y0_standard[bmask] - np.mean(Y0_standard[bmask])
        dY2_0 = delta0[bmask]**2
        cj_v0 = cj_stratified(np.abs(X0), dY2_0, strata_m)

        # === V1: No log (Y = pd*pu) ===
        delta1 = YC_nolog - Y0_nolog
        X1 = Y0_nolog[bmask] - np.mean(Y0_nolog[bmask])
        dY2_1 = delta1[bmask]**2
        cj_v1 = cj_stratified(np.abs(X1), dY2_1, strata_m)

        # === V2: No log, keep +1 ===
        delta2 = YC_nolog_p1 - Y0_nolog_p1
        X2 = Y0_nolog_p1[bmask] - np.mean(Y0_nolog_p1[bmask])
        dY2_2 = delta2[bmask]**2
        cj_v2 = cj_stratified(np.abs(X2), dY2_2, strata_m)

        # === V3: No centering (|Y0| instead of |Y0-mean|) ===
        cj_v3 = cj_stratified(Y0_standard[bmask], dY2_0, strata_m)

        # === V4: X^2 instead of |X| ===
        cj_v4 = cj_stratified(X0**2, dY2_0, strata_m)

        # === V5: No stratification (single bin) ===
        cj_v5 = float(np.mean(np.abs(X0) * dY2_0)
                       - np.mean(np.abs(X0)) * np.mean(dY2_0))

        # === V6: Raw product kernel (no log, no centering, no abs) ===
        # Weight by pd0*pu0 (the raw product, flat)
        # Response: (pdC*puC - pd0*pu0)^2
        raw_weight = (pd0 * pu0)[bmask]
        raw_response = ((pdC * puC - pd0 * pu0)[bmask])**2
        cj_v6 = cj_stratified(raw_weight, raw_response, strata_m)

        # === V7: Proper-time kernel weight ===
        # Weight by (tau-/T)^4*(tau+/T)^4 instead of |X|
        cj_v7 = cj_stratified(tau_kernel, dY2_0, strata_m)

        elapsed = time.time() - t0

        for vi, val in enumerate([cj_v0, cj_v1, cj_v2, cj_v3,
                                  cj_v4, cj_v5, cj_v6, cj_v7]):
            results[f"V{vi}"].append(val)

        if trial < 3 or trial == M - 1:
            print(f"  [{trial+1}/{M}] V0={cj_v0:.5e} V1={cj_v1:.3e} "
                  f"V5={cj_v5:.5e} V7={cj_v7:.5e}  ({elapsed:.1f}s)")

    # Analysis
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    N89 = N**(8/9)
    labels = [
        "V0: Standard CJ (log2, |X-mean|, strat)",
        "V1: No log (Y=pd*pu, |X-mean|, strat)",
        "V2: No log keep +1 (Y=pd*pu+1)",
        "V3: No centering (Y0, not |Y0-mean|)",
        "V4: X^2 instead of |X|",
        "V5: No stratification (single bin)",
        "V6: Raw product kernel (pd*pu weight)",
        "V7: Proper-time kernel weight",
    ]

    print(f"\n{'Variant':<45} {'mean CJ':>12} {'C_eff':>12} {'C/C_an':>8}")
    print("-" * 80)
    for i in range(8):
        vals = np.array(results[f"V{i}"])
        mean_cj = np.mean(vals)
        se = np.std(vals, ddof=1) / np.sqrt(M)
        # C_eff = CJ / (N^{8/9} * E^2 * T^4)
        c_eff = mean_cj / (N89 * E2 * T**4) if abs(mean_cj) > 1e-30 else 0
        ratio = c_eff / C_AN if C_AN > 0 else 0
        print(f"  {labels[i]:<43} {mean_cj:>12.5e} {c_eff:>12.5e} {ratio:>8.3f}")

    # Detailed V0 analysis
    v0_arr = np.array(results["V0"])
    c0_eff = np.mean(v0_arr) / (N89 * E2)
    print(f"\nV0 detail: C_0 = {c0_eff:.4e} ± {np.std(v0_arr,ddof=1)/np.sqrt(M)/(N89*E2):.4e}")
    print(f"           C_an = {C_AN:.4e}")
    print(f"           C_0/C_an = {c0_eff/C_AN:.3f}")

    # Save results
    output = {
        "N": N, "eps": eps, "T": T, "M": M, "E2": E2,
        "C_AN": C_AN, "N89": N89,
    }
    for i in range(8):
        vals = np.array(results[f"V{i}"])
        c_eff = np.mean(vals) / (N89 * E2)
        output[f"V{i}"] = {
            "label": labels[i],
            "mean_cj": float(np.mean(vals)),
            "se": float(np.std(vals, ddof=1) / np.sqrt(M)),
            "C_eff": float(c_eff),
            "ratio_to_Can": float(c_eff / C_AN),
        }
    return output


if __name__ == "__main__":
    t_start = time.time()

    res = run_investigation(N=2000, T=1.0, eps=3.0, M=20)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "fnd1_data")
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "factor41_investigation.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved -> {path}")
    print(f"Total: {time.time()-t_start:.0f}s")
