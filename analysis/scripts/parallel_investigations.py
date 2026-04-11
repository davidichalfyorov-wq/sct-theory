#!/usr/bin/env python3
"""4 parallel investigations while analytical thinks.

1. Stratum-1 NC-3: scramble δY only on depth 3-7, check R_scr
2. Multi-statistic NC-3: R_scr for variance, skewness, gini, entropy
3. dk vs ΔTC correlation across seeds
4. Absolute calibration: κ_flat(N) curve
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
T = 1.0
M_SEEDS = 10
K_PERM = 100


def gini(x):
    x = np.sort(np.abs(np.asarray(x, dtype=np.float64)))
    n = len(x)
    if n < 2 or np.sum(x) < 1e-15:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(2.0 * np.sum(idx * x) / (n * np.sum(x)) - (n + 1) / n)


def entropy_Y(Y):
    Y = np.asarray(Y, dtype=np.float64)
    if Y.size < 10:
        return 0.0
    counts, _ = np.histogram(Y, bins=20)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


if __name__ == "__main__":
    print("=== PARALLEL INVESTIGATIONS ===", flush=True)
    t_total = time.time()
    out = {}

    # Precompute flat and curved for all seeds
    all_data = []
    for si in range(M_SEEDS):
        seed = 1300000 + si
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)

        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)

        parF, chF = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS))
        YF = Y_from_graph(parF, chF)

        mask = bulk_mask(pts, T, ZETA)
        delta = YF - Y0

        # Depth
        depth = np.zeros(N, dtype=int)
        for i in range(N):
            if par0[i].size > 0:
                depth[i] = int(np.max(depth[par0[i]])) + 1

        # Strata (5τ × 3ρ × 3depth)
        tau_hat = 2.0 * pts[:, 0] / T
        r = np.linalg.norm(pts[:, 1:], axis=1)
        rmax = T / 2.0 - np.abs(pts[:, 0])
        rho_hat = np.clip(r / np.maximum(rmax, 1e-12), 0.0, 0.999999)
        tau_bin = np.clip(np.floor((tau_hat + 1.0) * 2.5).astype(int), 0, 4)
        rho_bin = np.clip(np.floor(rho_hat * 3).astype(int), 0, 2)
        depth_terc = np.clip((depth * 3) // max(int(depth.max()) + 1, 1), 0, 2)
        strata = tau_bin * 9 + rho_bin * 3 + depth_terc

        # TC (total causal pairs)
        tc_flat = sum(len(p) for p in par0)  # approximation: link count ≈ proportional to TC
        tc_curv = sum(len(p) for p in parF)
        dtc = tc_curv - tc_flat

        all_data.append({
            "pts": pts, "Y0": Y0, "YF": YF, "mask": mask, "delta": delta,
            "depth": depth, "strata": strata, "seed": seed, "dtc": dtc,
        })
        print(f"  Precomputed seed {si+1}/{M_SEEDS} ({time.time()-t_total:.0f}s)", flush=True)

    # ═══════════════════════════════════════════════════════
    # INVESTIGATION 1: Stratum-1 NC-3 (depth 3-7 only)
    # ═══════════════════════════════════════════════════════
    print("\n=== INV 1: STRATUM-1 NC-3 (depth 3-7) ===", flush=True)
    dk_curv_s1 = []
    dk_scr_s1 = []
    for d in all_data:
        mask_s1 = d["mask"] & (d["depth"] >= 3) & (d["depth"] <= 7)
        if mask_s1.sum() < 20:
            continue
        dk_c = excess_kurtosis(d["YF"][mask_s1]) - excess_kurtosis(d["Y0"][mask_s1])
        dk_curv_s1.append(dk_c)

        scr_vals = []
        rng = np.random.default_rng(d["seed"] + 60000)
        for k in range(K_PERM):
            delta_scr = d["delta"].copy()
            for label in np.unique(d["strata"][mask_s1]):
                idx = np.where((d["strata"] == label) & mask_s1)[0]
                if len(idx) > 1:
                    delta_scr[idx] = d["delta"][rng.permutation(idx)]
            Yscr = d["Y0"] + delta_scr
            scr_vals.append(excess_kurtosis(Yscr[mask_s1]) - excess_kurtosis(d["Y0"][mask_s1]))
        dk_scr_s1.append(np.mean(scr_vals))

    R_scr_s1 = abs(np.mean(dk_scr_s1)) / max(abs(np.mean(dk_curv_s1)), 1e-15)
    print(f"  dk_curv (depth 3-7): {np.mean(dk_curv_s1):+.6f}", flush=True)
    print(f"  dk_scr (depth 3-7):  {np.mean(dk_scr_s1):+.6f}", flush=True)
    print(f"  R_scr (stratum 1) = {R_scr_s1:.4f}  (global was 0.504)", flush=True)
    out["inv1_stratum1_nc3"] = {"R_scr": float(R_scr_s1),
                                 "dk_curv": float(np.mean(dk_curv_s1)),
                                 "dk_scr": float(np.mean(dk_scr_s1))}

    # ═══════════════════════════════════════════════════════
    # INVESTIGATION 2: Multi-statistic NC-3
    # ═══════════════════════════════════════════════════════
    print("\n=== INV 2: MULTI-STATISTIC NC-3 ===", flush=True)
    stat_funcs = {
        "kurtosis": lambda y: excess_kurtosis(y),
        "variance": lambda y: float(np.var(y)),
        "skewness": lambda y: float(sp_stats.skew(y)),
        "gini": lambda y: gini(y),
        "entropy": lambda y: entropy_Y(y),
    }

    for stat_name, stat_fn in stat_funcs.items():
        dk_curv_list = []
        dk_scr_list = []
        for d in all_data:
            dk_c = stat_fn(d["YF"][d["mask"]]) - stat_fn(d["Y0"][d["mask"]])
            dk_curv_list.append(dk_c)

            scr_vals = []
            rng = np.random.default_rng(d["seed"] + 70000)
            for k in range(50):  # fewer perms for speed
                delta_scr = d["delta"].copy()
                for label in np.unique(d["strata"][d["mask"]]):
                    idx = np.where((d["strata"] == label) & d["mask"])[0]
                    if len(idx) > 1:
                        delta_scr[idx] = d["delta"][rng.permutation(idx)]
                Yscr = d["Y0"] + delta_scr
                scr_vals.append(stat_fn(Yscr[d["mask"]]) - stat_fn(d["Y0"][d["mask"]]))
            dk_scr_list.append(np.mean(scr_vals))

        R = abs(np.mean(dk_scr_list)) / max(abs(np.mean(dk_curv_list)), 1e-15)
        print(f"  {stat_name:12s}: dk_curv={np.mean(dk_curv_list):+.6f}, R_scr={R:.4f}", flush=True)
        out[f"inv2_{stat_name}"] = {"R_scr": float(R), "dk_curv": float(np.mean(dk_curv_list))}

    # ═══════════════════════════════════════════════════════
    # INVESTIGATION 3: dk vs ΔTC correlation
    # ═══════════════════════════════════════════════════════
    print("\n=== INV 3: dk vs ΔTC CORRELATION ===", flush=True)
    dks = [excess_kurtosis(d["YF"][d["mask"]]) - excess_kurtosis(d["Y0"][d["mask"]]) for d in all_data]
    dtcs = [d["dtc"] for d in all_data]
    rho_dk_tc, p_dk_tc = sp_stats.spearmanr(dks, dtcs)
    print(f"  Spearman(dk, ΔTC): ρ = {rho_dk_tc:+.4f}, p = {p_dk_tc:.4f}", flush=True)
    print(f"  If |ρ| < 0.3 → dk independent of pair-count change", flush=True)
    out["inv3_dk_vs_dtc"] = {"spearman_rho": float(rho_dk_tc), "p_value": float(p_dk_tc)}

    # ═══════════════════════════════════════════════════════
    # INVESTIGATION 4: Absolute calibration κ_flat(N)
    # ═══════════════════════════════════════════════════════
    print("\n=== INV 4: ABSOLUTE CALIBRATION κ_flat(N) ===", flush=True)
    for N_cal in [2000, 5000, 10000]:
        k_vals = []
        for si in range(20):
            seed = 1400000 + N_cal + si
            rng = np.random.default_rng(seed)
            pts = sprinkle_local_diamond(N_cal, T, rng)
            par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
            Y0 = Y_from_graph(par0, ch0)
            mask = bulk_mask(pts, T, ZETA)
            k_vals.append(excess_kurtosis(Y0[mask]))
        arr = np.array(k_vals)
        print(f"  N={N_cal}: κ_flat = {np.mean(arr):+.4f} ± {np.std(arr,ddof=1)/np.sqrt(len(arr)):.4f} "
              f"(std={np.std(arr,ddof=1):.4f})", flush=True)
        out[f"inv4_kflat_N{N_cal}"] = {"mean": float(np.mean(arr)), "std": float(np.std(arr, ddof=1)),
                                        "se": float(np.std(arr,ddof=1)/np.sqrt(len(arr)))}

    total = time.time() - t_total
    print(f"\n=== TOTAL: {total:.0f}s = {total/60:.1f}min ===", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "universal_runs_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "parallel_investigations.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved.", flush=True)
