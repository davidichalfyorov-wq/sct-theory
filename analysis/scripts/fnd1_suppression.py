#!/usr/bin/env python3
"""FND-1 TEST 4.5: T/r₀ suppression test.

Does the Sch/ppw ratio improve as T/r₀ → 0 (weaker-field regime)?
Keep q_W = M·T²/r₀³ = 0.4 fixed. Vary r₀.

Points:
  r₀=0.5, M=0.05,  T/r₀=2.0  (current baseline)
  r₀=1.0, M=0.40,  T/r₀=1.0
  r₀=1.5, M=1.35,  T/r₀=0.67

analytical PASS: calibrated ratio increases toward 1 as T/r₀ decreases.
analytical FAIL: ratio plateaus near 0.5.
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, ppwave_exact_preds, jet_preds,
    bulk_mask, riemann_schwarzschild_local,
)
from schwarzschild_exact_local_tools import (
    map_rnc_to_schwarzschild_expmap,
    schwarzschild_exact_midpoint_preds_from_mapped
)

N = 10000
M_SEEDS = 20
SEED_BASE = 6000000
T = 1.0
EPS = 3.0
ZETA = 0.15
Q_W_TARGET = 0.4  # fixed tidal parameter

# r₀ values → compute M to keep q_W = M/r₀³ = 0.4
# Constraint: r₀ > 2M = 2·q_W·r₀³ → r₀ < 1/√(2·q_W) ≈ 1.118 for q_W=0.4
R0_VALUES = [0.50, 0.70, 1.00]
# M = q_W · r₀³
CONFIGS = [{"r0": r0, "M_sch": Q_W_TARGET * r0**3, "T_over_r0": T / r0}
           for r0 in R0_VALUES]

E2_PPW = EPS**2 / 2.0
# E²_Sch = 6q² = 6·0.4² = 0.96 for all configs (same q_W)
E2_SCH = 6.0 * Q_W_TARGET**2


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
    total = 0.0
    for label in np.unique(strata_m):
        idx = strata_m == label
        if idx.sum() < 3:
            continue
        w = idx.sum() / len(X)
        cov = np.mean(X2[idx] * dY2[idx]) - np.mean(X2[idx]) * np.mean(dY2[idx])
        total += w * cov
    return float(total)


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("FND-1 TEST 4.5: T/r₀ SUPPRESSION", flush=True)
    print(f"q_W={Q_W_TARGET} fixed, r₀={R0_VALUES}", flush=True)
    print("=" * 60, flush=True)

    t_total = time.time()
    results = {}

    # PPW reference (same for all configs since ppw doesn't depend on r₀/M)
    print("\n─── PPW REFERENCE ───", flush=True)
    a_ppw_list = []
    for si in range(M_SEEDS):
        seed = SEED_BASE + si
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)
        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)
        mask = bulk_mask(pts, T, ZETA)
        strata = make_strata(pts, par0, T)
        parP, chP = build_hasse_from_predicate(
            pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS))
        YP = Y_from_graph(parP, chP)
        a_ppw_list.append(compute_A_align(Y0, YP - Y0, mask, strata))
        if (si + 1) % 10 == 0:
            print(f"  {si+1}/{M_SEEDS}", flush=True)

    AE_ppw = np.mean(a_ppw_list) / E2_PPW
    print(f"  AE_ppw = {AE_ppw:.6f}", flush=True)
    results["ppw"] = {"AE": AE_ppw, "n_seeds": M_SEEDS}

    # SCH at each r₀
    for cfg in CONFIGS:
        r0 = cfg["r0"]
        M_sch = cfg["M_sch"]
        T_r0 = cfg["T_over_r0"]
        R_SCH = riemann_schwarzschild_local(M_sch, r0)

        print(f"\n─── r₀={r0}, M={M_sch:.4f}, T/r₀={T_r0:.2f} ───", flush=True)

        a_sch_jet_list = []
        a_sch_exp_list = []

        for si in range(M_SEEDS):
            seed = SEED_BASE + si
            rng = np.random.default_rng(seed)
            pts = sprinkle_local_diamond(N, T, rng)
            par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
            Y0 = Y_from_graph(par0, ch0)
            mask = bulk_mask(pts, T, ZETA)
            strata = make_strata(pts, par0, T)

            # Sch jet
            parS, chS = build_hasse_from_predicate(
                pts, lambda P, i: jet_preds(P, i, R_abcd=R_SCH))
            YS = Y_from_graph(parS, chS)
            a_sch_jet_list.append(compute_A_align(Y0, YS - Y0, mask, strata))

            # Sch expmap
            mapped = map_rnc_to_schwarzschild_expmap(pts, M_sch, r0)
            parX, chX = build_hasse_from_predicate(
                pts, lambda P, i: schwarzschild_exact_midpoint_preds_from_mapped(
                    mapped, i, M_sch))
            YX = Y_from_graph(parX, chX)
            a_sch_exp_list.append(compute_A_align(Y0, YX - Y0, mask, strata))

            if (si + 1) % 5 == 0:
                elapsed = time.time() - t_total
                print(f"  {si+1}/{M_SEEDS} ({elapsed/60:.0f}min)", flush=True)

        AE_jet = np.mean(a_sch_jet_list) / E2_SCH
        AE_exp = np.mean(a_sch_exp_list) / E2_SCH
        ratio_jet = AE_jet / AE_ppw if abs(AE_ppw) > 1e-15 else 0
        ratio_exp = AE_exp / AE_ppw if abs(AE_ppw) > 1e-15 else 0

        print(f"  AE_sch_jet = {AE_jet:.6f}, ratio = {ratio_jet:.3f}", flush=True)
        print(f"  AE_sch_exp = {AE_exp:.6f}, ratio = {ratio_exp:.3f}", flush=True)

        results[f"r0_{r0}"] = {
            "r0": r0, "M_sch": M_sch, "T_over_r0": T_r0,
            "AE_jet": AE_jet, "AE_exp": AE_exp,
            "ratio_jet": ratio_jet, "ratio_exp": ratio_exp,
        }

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("SUMMARY: T/r₀ → ratio", flush=True)
    for cfg in CONFIGS:
        r0 = cfg["r0"]
        key = f"r0_{r0}"
        r = results[key]
        print(f"  T/r₀={r['T_over_r0']:.2f}: ratio_exp={r['ratio_exp']:.3f}, "
              f"ratio_jet={r['ratio_jet']:.3f}", flush=True)

    total_time = time.time() - t_total
    results["total_time_min"] = total_time / 60
    print(f"\nTotal: {total_time/60:.0f}min", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "fnd1_data")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "suppression_test.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Saved.", flush=True)
