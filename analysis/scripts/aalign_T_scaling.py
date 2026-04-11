#!/usr/bin/env python3
"""A_align T-scaling: does the alignment channel follow T⁴?

A_align = Σ_B w_B Cov_B(X², δY²)
- R_scr = 0.001 (purely spatial)
- d = 6.2 (strong)
- T-scaling: UNKNOWN — this test answers it

Also computes Moran's I and kurtosis for comparison at each T.
"""
import sys, os, time, json
import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, excess_kurtosis, ppwave_exact_preds, bulk_mask
)

N = 10000
ZETA = 0.15
EPS = 3.0
M_SEEDS = 15


def compute_A_align(Y0, delta, mask, strata):
    X = Y0[mask] - np.mean(Y0[mask])
    X2 = X ** 2
    dY2 = delta[mask] ** 2
    strata_m = strata[mask]
    total_cov = 0.0
    for label in np.unique(strata_m):
        idx = strata_m == label
        n_b = idx.sum()
        if n_b < 3:
            continue
        w_b = n_b / len(X)
        cov_b = np.mean(X2[idx] * dY2[idx]) - np.mean(X2[idx]) * np.mean(dY2[idx])
        total_cov += w_b * cov_b
    return float(total_cov)


def compute_morans_I(delta, parents, children, mask):
    idx_map = np.where(mask)[0]
    n = len(idx_map)
    if n < 10:
        return 0.0
    idx_to_local = {g: l for l, g in enumerate(idx_map)}
    dY = delta[idx_map]
    dY_c = dY - np.mean(dY)
    num = 0.0
    W = 0
    for li, gi in enumerate(idx_map):
        for j in parents[gi]:
            j = int(j)
            if j in idx_to_local:
                num += dY_c[li] * dY_c[idx_to_local[j]]
                W += 1
        for j in children[gi]:
            j = int(j)
            if j in idx_to_local:
                num += dY_c[li] * dY_c[idx_to_local[j]]
                W += 1
    den = np.sum(dY_c ** 2)
    if W == 0 or den < 1e-15:
        return 0.0
    return float((n / W) * (num / den))


def make_strata(pts, parents0, T):
    tau_hat = 2.0 * pts[:, 0] / T
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2.0 - np.abs(pts[:, 0])
    rho_hat = np.clip(r / np.maximum(rmax, 1e-12), 0.0, 0.999999)
    tau_bin = np.clip(np.floor((tau_hat + 1.0) * 2.5).astype(int), 0, 4)
    rho_bin = np.clip(np.floor(rho_hat * 3).astype(int), 0, 2)
    depth = np.zeros(len(pts), dtype=int)
    for i in range(len(pts)):
        if parents0[i].size > 0:
            depth[i] = int(np.max(depth[parents0[i]])) + 1
    max_d = max(int(depth.max()), 1)
    depth_terc = np.clip((depth * 3) // (max_d + 1), 0, 2)
    return tau_bin * 9 + rho_bin * 3 + depth_terc


if __name__ == "__main__":
    print(f"=== A_ALIGN T-SCALING: eps={EPS} fixed, N={N}, M={M_SEEDS} ===", flush=True)
    print(f"Three observables: kurtosis, A_align, Moran's I", flush=True)
    print(flush=True)

    t_total = time.time()
    results = {}

    for T in [1.0, 0.70, 0.50]:
        kurt_list, aalign_list, moran_list = [], [], []
        t0 = time.time()

        for si in range(M_SEEDS):
            seed = 2000000 + int(T * 1000) * 100 + si
            rng = np.random.default_rng(seed)
            pts = sprinkle_local_diamond(N, T, rng)

            par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
            Y0 = Y_from_graph(par0, ch0)

            parF, chF = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS))
            YF = Y_from_graph(parF, chF)

            mask = bulk_mask(pts, T, ZETA)
            delta = YF - Y0
            strata = make_strata(pts, par0, T)

            kurt_list.append(excess_kurtosis(YF[mask]) - excess_kurtosis(Y0[mask]))
            aalign_list.append(compute_A_align(Y0, delta, mask, strata))
            moran_list.append(compute_morans_I(delta, par0, ch0, mask))

            if (si + 1) % 5 == 0:
                print(f"  T={T:.2f}: {si+1}/{M_SEEDS} ({time.time()-t0:.0f}s)", flush=True)

        for name, arr in [("kurtosis", kurt_list), ("A_align", aalign_list), ("morans_I", moran_list)]:
            a = np.array(arr)
            m = np.mean(a)
            se = np.std(a, ddof=1) / np.sqrt(len(a))
            d = m / np.std(a, ddof=1) if np.std(a, ddof=1) > 0 else 0
            results[f"{name}_T{T}"] = {"mean": float(m), "se": float(se), "d": float(d)}

        print(f"  T={T:.2f}: kurt={np.mean(kurt_list):+.6f}, "
              f"A_align={np.mean(aalign_list):+.6f}, "
              f"Moran={np.mean(moran_list):+.6f}", flush=True)

    # T-scaling ratios
    print(f"\n=== T-SCALING RATIOS (normalized to T=1.0) ===", flush=True)
    print(f"{'Observable':12s} {'T=1.0':>10s} {'T=0.70':>10s} {'T=0.50':>10s} "
          f"{'r(0.70)':>10s} {'r(0.50)':>10s} {'r/T⁴(0.70)':>12s} {'r/T⁴(0.50)':>12s}", flush=True)

    for name in ["kurtosis", "A_align", "morans_I"]:
        v1 = results[f"{name}_T1.0"]["mean"]
        v07 = results[f"{name}_T0.7"]["mean"]
        v05 = results[f"{name}_T0.5"]["mean"]
        r07 = v07 / v1 if abs(v1) > 1e-15 else 0
        r05 = v05 / v1 if abs(v1) > 1e-15 else 0
        rt4_07 = r07 / 0.70**4 if abs(r07) > 0 else 0
        rt4_05 = r05 / 0.50**4 if abs(r05) > 0 else 0
        print(f"  {name:12s} {v1:+10.6f} {v07:+10.6f} {v05:+10.6f} "
              f"{r07:10.4f} {r05:10.4f} {rt4_07:12.4f} {rt4_05:12.4f}", flush=True)

    total = time.time() - t_total
    print(f"\nTotal: {total:.0f}s = {total/60:.1f}min", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "universal_runs_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "aalign_T_scaling.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved.", flush=True)
