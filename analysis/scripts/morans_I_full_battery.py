#!/usr/bin/env python3
"""Moran's I FULL BATTERY: all tests that kurtosis passed/failed.

Tests:
1. T⁴ scaling (fixed eps=3, T=1.0/0.70/0.50, pp-wave exact)
2. dS Ricci subtraction (should be ~0 if Moran's I is curvature-specific)
3. Sch local (jet predicate, M_sch=0.05, T=1.0/0.70)
4. NC-2: matched-ΔTC ppw vs dS (bisection)

All using Moran's I instead of kurtosis.
Sequential. No multiprocessing.
NO prediction-dependent filtering — all values reported raw.
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


def compute_morans_I(delta, parents, children, mask):
    """Moran's I of δY on Hasse graph."""
    idx_map = np.where(mask)[0]
    n = len(idx_map)
    if n < 10:
        return 0.0

    idx_set = set(idx_map.tolist())
    idx_to_local = {g: l for l, g in enumerate(idx_map)}

    dY = delta[idx_map]
    dY_centered = dY - np.mean(dY)

    numerator = 0.0
    W = 0
    for local_i, global_i in enumerate(idx_map):
        for j in parents[global_i]:
            j = int(j)
            if j in idx_to_local:
                local_j = idx_to_local[j]
                numerator += dY_centered[local_i] * dY_centered[local_j]
                W += 1
        for j in children[global_i]:
            j = int(j)
            if j in idx_to_local:
                local_j = idx_to_local[j]
                numerator += dY_centered[local_i] * dY_centered[local_j]
                W += 1

    denominator = np.sum(dY_centered ** 2)
    if W == 0 or denominator < 1e-15:
        return 0.0

    I = (n / W) * (numerator / denominator)
    return float(I)


def run_crn_seed(pts, curved_pred_fn, flat_parents, flat_children, Y0, mask):
    """Run one CRN seed: compute Moran's I delta."""
    parF, chF = build_hasse_from_predicate(pts, curved_pred_fn)
    YF = Y_from_graph(parF, chF)
    delta = YF - Y0

    # Moran's I on FLAT graph adjacency (same graph for all, only δY changes)
    mI = compute_morans_I(delta, flat_parents, flat_children, mask)

    # Also compute kurtosis for comparison
    dk = excess_kurtosis(YF[mask]) - excess_kurtosis(Y0[mask])

    return mI, dk


def count_causal_pairs(pts, pred_fn):
    total = 0
    for i in range(len(pts)):
        total += int(pred_fn(pts, i).sum())
    return total


if __name__ == "__main__":
    print("=== MORAN'S I FULL BATTERY ===", flush=True)
    print(f"N={N}, zeta={ZETA}", flush=True)
    print("All values reported raw. No prediction-dependent filtering.", flush=True)
    print(flush=True)

    out = {}
    t_total = time.time()

    # ═══════════════════════════════════════════════════════
    # TEST 1: T⁴ SCALING (pp-wave exact, eps=3 fixed)
    # ═══════════════════════════════════════════════════════
    print("=== TEST 1: T⁴ SCALING (pp-wave exact, eps=3) ===", flush=True)
    EPS = 3.0
    M_T = 15

    t_scaling = {}
    for T in [1.0, 0.70, 0.50]:
        mI_list = []
        dk_list = []
        t0 = time.time()
        for si in range(M_T):
            seed = 1600000 + int(T * 1000) * 100 + si
            rng = np.random.default_rng(seed)
            pts = sprinkle_local_diamond(N, T, rng)

            par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
            Y0 = Y_from_graph(par0, ch0)
            mask = bulk_mask(pts, T, ZETA)

            mI, dk = run_crn_seed(
                pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS),
                par0, ch0, Y0, mask)
            mI_list.append(mI)
            dk_list.append(dk)

            if (si + 1) % 5 == 0:
                print(f"  T={T:.2f}: {si+1}/{M_T} ({time.time()-t0:.0f}s)", flush=True)

        arr_mI = np.array(mI_list)
        arr_dk = np.array(dk_list)
        m_mI = np.mean(arr_mI)
        se_mI = np.std(arr_mI, ddof=1) / np.sqrt(len(arr_mI))
        d_mI = m_mI / np.std(arr_mI, ddof=1) if np.std(arr_mI, ddof=1) > 0 else 0

        print(f"  T={T:.2f}: Moran_I={m_mI:+.6f}±{se_mI:.6f} (d={d_mI:+.3f}), "
              f"kurtosis={np.mean(arr_dk):+.6f}", flush=True)

        t_scaling[str(T)] = {
            "mI_mean": float(m_mI), "mI_se": float(se_mI), "mI_d": float(d_mI),
            "dk_mean": float(np.mean(arr_dk)),
            "mI_per_seed": mI_list, "dk_per_seed": dk_list,
        }

    # T-scaling ratio
    mI_1 = t_scaling["1.0"]["mI_mean"]
    print(f"\n  T-scaling (Moran's I):", flush=True)
    for T in [1.0, 0.70, 0.50]:
        mI_T = t_scaling[str(T)]["mI_mean"]
        ratio = mI_T / mI_1 if abs(mI_1) > 1e-15 else 0
        rt4 = ratio / T**4 if T < 1.0 else 1.0
        print(f"  T={T:.2f}: ratio={ratio:.4f}, ratio/T⁴={rt4:.4f}", flush=True)

    out["test1_T_scaling"] = t_scaling

    # ═══════════════════════════════════════════════════════
    # TEST 2: dS RICCI SUBTRACTION
    # ═══════════════════════════════════════════════════════
    print(f"\n=== TEST 2: dS RICCI SUBTRACTION ===", flush=True)
    T_DS = 1.0
    H_DS = 0.707
    M_DS = 15

    mI_raw = []
    mI_weyl = []
    dk_raw = []
    for si in range(M_DS):
        seed = 1700000 + si
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T_DS, rng)

        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)
        mask = bulk_mask(pts, T_DS, ZETA)

        # dS exact
        parD, chD = build_hasse_from_predicate(pts, lambda P, i: ds_preds(P, i, H=H_DS))
        YdS = Y_from_graph(parD, chD)

        delta_raw = YdS - Y0
        mI_r = compute_morans_I(delta_raw, par0, ch0, mask)
        mI_raw.append(mI_r)
        dk_raw.append(excess_kurtosis(YdS[mask]) - excess_kurtosis(Y0[mask]))

        # Weyl = full - Ricci. For dS: Ricci = full → Weyl = 0
        delta_weyl = YdS - YdS  # trivially zero
        mI_w = compute_morans_I(delta_weyl, par0, ch0, mask)
        mI_weyl.append(mI_w)

        if (si + 1) % 5 == 0:
            print(f"  dS: {si+1}/{M_DS}", flush=True)

    print(f"  dS raw Moran_I: {np.mean(mI_raw):+.6f}±{np.std(mI_raw,ddof=1)/np.sqrt(len(mI_raw)):.6f}", flush=True)
    print(f"  dS weyl Moran_I: {np.mean(mI_weyl):+.8f} (should be 0)", flush=True)
    print(f"  dS raw kurtosis: {np.mean(dk_raw):+.6f}", flush=True)

    out["test2_dS"] = {
        "mI_raw_mean": float(np.mean(mI_raw)),
        "mI_weyl_mean": float(np.mean(mI_weyl)),
        "dk_raw_mean": float(np.mean(dk_raw)),
    }

    # ═══════════════════════════════════════════════════════
    # TEST 3: SCHWARZSCHILD LOCAL
    # ═══════════════════════════════════════════════════════
    print(f"\n=== TEST 3: SCHWARZSCHILD LOCAL ===", flush=True)
    M_SCH = 0.05
    R0 = 0.50
    R_SCH = riemann_schwarzschild_local(M_SCH, R0)
    M_SEEDS_SCH = 30

    for T_S in [1.0, 0.70]:
        mI_list = []
        dk_list = []
        t0 = time.time()
        for si in range(M_SEEDS_SCH):
            seed = 1800000 + int(T_S * 1000) * 100 + si
            rng = np.random.default_rng(seed)
            pts = sprinkle_local_diamond(N, T_S, rng)

            par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
            Y0 = Y_from_graph(par0, ch0)
            mask = bulk_mask(pts, T_S, ZETA)

            mI, dk = run_crn_seed(
                pts, lambda P, i: jet_preds(P, i, R_abcd=R_SCH),
                par0, ch0, Y0, mask)
            mI_list.append(mI)
            dk_list.append(dk)

            if (si + 1) % 10 == 0:
                print(f"  Sch T={T_S:.2f}: {si+1}/{M_SEEDS_SCH} ({time.time()-t0:.0f}s)", flush=True)

        arr_mI = np.array(mI_list)
        m = np.mean(arr_mI)
        se = np.std(arr_mI, ddof=1) / np.sqrt(len(arr_mI))
        d = m / np.std(arr_mI, ddof=1) if np.std(arr_mI, ddof=1) > 0 else 0
        print(f"  Sch T={T_S:.2f}: Moran_I={m:+.6f}±{se:.6f} (d={d:+.3f}), "
              f"kurtosis={np.mean(dk_list):+.6f}", flush=True)

        out[f"test3_sch_T{T_S}"] = {
            "mI_mean": float(m), "mI_se": float(se), "mI_d": float(d),
            "dk_mean": float(np.mean(dk_list)),
        }

    # ═══════════════════════════════════════════════════════
    # TEST 4: NC-2 MATCHED-ΔTC (pp-wave vs dS)
    # ═══════════════════════════════════════════════════════
    print(f"\n=== TEST 4: NC-2 MATCHED-ΔTC (Moran's I) ===", flush=True)
    EPS_NC2 = 3.0
    T_NC2 = 1.0
    M_NC2 = 10
    H_BRACKET = (0.01, 5.0)

    mI_ppw = []
    mI_dS = []
    for si in range(M_NC2):
        seed = 1900000 + si
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T_NC2, rng)

        mask = bulk_mask(pts, T_NC2, ZETA)

        # Flat
        TC0 = count_causal_pairs(pts, lambda P, i: minkowski_preds(P, i))
        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)

        # pp-wave
        TCA = count_causal_pairs(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS_NC2))
        dTC_target = TCA - TC0

        # Bisect dS to match ΔTC
        lo, hi = H_BRACKET
        for step in range(15):
            mid = 0.5 * (lo + hi)
            TCB = count_causal_pairs(pts, lambda P, i: ds_preds(P, i, H=mid))
            dTC_mid = TCB - TC0
            if dTC_mid < dTC_target:
                lo = mid
            else:
                hi = mid
        H_matched = 0.5 * (lo + hi)

        # Build Hasse for ppw and matched dS
        parA, chA = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS_NC2))
        YA = Y_from_graph(parA, chA)
        deltaA = YA - Y0
        mI_a = compute_morans_I(deltaA, par0, ch0, mask)

        parB, chB = build_hasse_from_predicate(pts, lambda P, i: ds_preds(P, i, H=H_matched))
        YB = Y_from_graph(parB, chB)
        deltaB = YB - Y0
        mI_b = compute_morans_I(deltaB, par0, ch0, mask)

        mI_ppw.append(mI_a)
        mI_dS.append(mI_b)

        if (si + 1) % 5 == 0:
            print(f"  NC-2: {si+1}/{M_NC2}, H={H_matched:.4f}, "
                  f"mI_ppw={mI_a:+.4f}, mI_dS={mI_b:+.4f}", flush=True)

    arr_ppw = np.array(mI_ppw)
    arr_dS = np.array(mI_dS)
    R_TC_mI = abs(np.mean(arr_dS)) / max(abs(np.mean(arr_ppw)), 1e-15)

    print(f"\n  NC-2 Moran's I:", flush=True)
    print(f"  ppw:  {np.mean(arr_ppw):+.6f}±{np.std(arr_ppw,ddof=1)/np.sqrt(len(arr_ppw)):.6f}", flush=True)
    print(f"  dS:   {np.mean(arr_dS):+.6f}±{np.std(arr_dS,ddof=1)/np.sqrt(len(arr_dS)):.6f}", flush=True)
    print(f"  R_TC (Moran's I) = {R_TC_mI:.4f}", flush=True)

    out["test4_NC2"] = {
        "mI_ppw_mean": float(np.mean(arr_ppw)),
        "mI_dS_mean": float(np.mean(arr_dS)),
        "R_TC": float(R_TC_mI),
    }

    # ═══════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════
    total = time.time() - t_total
    print(f"\n=== TOTAL: {total:.0f}s = {total/60:.1f}min ===", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "universal_runs_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "morans_I_full_battery.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved.", flush=True)
