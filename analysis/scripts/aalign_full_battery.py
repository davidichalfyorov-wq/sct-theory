#!/usr/bin/env python3
"""A_align full battery: NC-2 (matched-ΔTC) + Sch local + dS subtraction.

A_align = Σ_B w_B Cov_B(X², δY²)
Already confirmed: T⁴ ✅, R_scr=0.001 ✅, d=6.2 ✅
Missing: NC-2, Sch, dS
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, excess_kurtosis, ppwave_exact_preds, ds_preds,
    jet_preds, bulk_mask, riemann_schwarzschild_local
)

N = 10000
ZETA = 0.15


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


def count_causal_pairs(pts, pred_fn):
    total = 0
    for i in range(len(pts)):
        total += int(pred_fn(pts, i).sum())
    return total


if __name__ == "__main__":
    print("=== A_ALIGN FULL BATTERY ===", flush=True)
    t_total = time.time()
    out = {}

    # ═══════════════════════════════════════════════════════
    # TEST 1: Sch local (T=1.0, M_sch=0.05)
    # ═══════════════════════════════════════════════════════
    print("=== TEST 1: Sch local ===", flush=True)
    M_SCH, R0 = 0.05, 0.50
    R_SCH = riemann_schwarzschild_local(M_SCH, R0)
    M_S = 30

    aalign_sch = []
    dk_sch = []
    for si in range(M_S):
        seed = 2100000 + si
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, 1.0, rng)
        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)
        parF, chF = build_hasse_from_predicate(pts, lambda P, i: jet_preds(P, i, R_abcd=R_SCH))
        YF = Y_from_graph(parF, chF)
        mask = bulk_mask(pts, 1.0, ZETA)
        delta = YF - Y0
        strata = make_strata(pts, par0, 1.0)
        aalign_sch.append(compute_A_align(Y0, delta, mask, strata))
        dk_sch.append(excess_kurtosis(YF[mask]) - excess_kurtosis(Y0[mask]))
        if (si + 1) % 10 == 0:
            print(f"  Sch: {si+1}/{M_S} ({time.time()-t_total:.0f}s)", flush=True)

    arr = np.array(aalign_sch)
    d = np.mean(arr) / np.std(arr, ddof=1) if np.std(arr, ddof=1) > 0 else 0
    print(f"  Sch A_align: {np.mean(arr):+.6f}±{np.std(arr,ddof=1)/np.sqrt(len(arr)):.6f} (d={d:+.3f})", flush=True)
    print(f"  Sch kurtosis: {np.mean(dk_sch):+.6f}", flush=True)
    out["sch_aalign"] = {"mean": float(np.mean(arr)), "d": float(d)}
    out["sch_kurtosis"] = {"mean": float(np.mean(dk_sch))}

    # ═══════════════════════════════════════════════════════
    # TEST 2: dS raw A_align
    # ═══════════════════════════════════════════════════════
    print("\n=== TEST 2: dS ===", flush=True)
    H_DS = 0.707
    M_DS = 15
    aalign_ds = []
    for si in range(M_DS):
        seed = 2200000 + si
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, 1.0, rng)
        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)
        parD, chD = build_hasse_from_predicate(pts, lambda P, i: ds_preds(P, i, H=H_DS))
        YdS = Y_from_graph(parD, chD)
        mask = bulk_mask(pts, 1.0, ZETA)
        delta = YdS - Y0
        strata = make_strata(pts, par0, 1.0)
        aalign_ds.append(compute_A_align(Y0, delta, mask, strata))
        if (si + 1) % 5 == 0:
            print(f"  dS: {si+1}/{M_DS}", flush=True)

    print(f"  dS A_align: {np.mean(aalign_ds):+.6f}±{np.std(aalign_ds,ddof=1)/np.sqrt(len(aalign_ds)):.6f}", flush=True)
    out["ds_aalign"] = {"mean": float(np.mean(aalign_ds))}

    # ═══════════════════════════════════════════════════════
    # TEST 3: NC-2 matched-ΔTC (A_align)
    # ═══════════════════════════════════════════════════════
    print("\n=== TEST 3: NC-2 matched-ΔTC ===", flush=True)
    EPS = 3.0
    M_NC2 = 10

    aalign_ppw = []
    aalign_dS_matched = []
    for si in range(M_NC2):
        seed = 2300000 + si
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, 1.0, rng)
        mask = bulk_mask(pts, 1.0, ZETA)

        TC0 = count_causal_pairs(pts, lambda P, i: minkowski_preds(P, i))
        TCA = count_causal_pairs(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS))
        dTC_target = TCA - TC0

        # Bisect dS
        lo, hi = 0.01, 5.0
        for step in range(15):
            mid = 0.5 * (lo + hi)
            TCB = count_causal_pairs(pts, lambda P, i: ds_preds(P, i, H=mid))
            if TCB - TC0 < dTC_target:
                lo = mid
            else:
                hi = mid
        H_matched = 0.5 * (lo + hi)

        # Build Hasse
        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)
        strata = make_strata(pts, par0, 1.0)

        parA, chA = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS))
        YA = Y_from_graph(parA, chA)
        deltaA = YA - Y0
        aalign_ppw.append(compute_A_align(Y0, deltaA, mask, strata))

        parB, chB = build_hasse_from_predicate(pts, lambda P, i: ds_preds(P, i, H=H_matched))
        YB = Y_from_graph(parB, chB)
        deltaB = YB - Y0
        aalign_dS_matched.append(compute_A_align(Y0, deltaB, mask, strata))

        if (si + 1) % 5 == 0:
            print(f"  NC-2: {si+1}/{M_NC2} H={H_matched:.4f}", flush=True)

    arr_ppw = np.array(aalign_ppw)
    arr_ds = np.array(aalign_dS_matched)
    R_TC = abs(np.mean(arr_ds)) / max(abs(np.mean(arr_ppw)), 1e-15)

    print(f"\n  NC-2 A_align:", flush=True)
    print(f"  ppw:  {np.mean(arr_ppw):+.6f}±{np.std(arr_ppw,ddof=1)/np.sqrt(len(arr_ppw)):.6f}", flush=True)
    print(f"  dS:   {np.mean(arr_ds):+.6f}±{np.std(arr_ds,ddof=1)/np.sqrt(len(arr_ds)):.6f}", flush=True)
    print(f"  R_TC (A_align) = {R_TC:.4f}", flush=True)

    if R_TC < 0.5:
        verdict = "STRONG"
    elif R_TC < 0.8:
        verdict = "MODERATE"
    else:
        verdict = "WEAK/BAD"
    print(f"  VERDICT: {verdict}", flush=True)

    out["nc2_aalign"] = {
        "ppw_mean": float(np.mean(arr_ppw)),
        "ds_mean": float(np.mean(arr_ds)),
        "R_TC": float(R_TC),
        "verdict": verdict,
    }

    total = time.time() - t_total
    print(f"\nTotal: {total:.0f}s = {total/60:.1f}min", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "universal_runs_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "aalign_full_battery.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved.", flush=True)
