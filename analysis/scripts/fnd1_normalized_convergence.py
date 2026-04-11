#!/usr/bin/env python3
"""FND-1: Re-normalized N-convergence using BITSET Hasse (20-30x faster).

independent analysis prescription (2026-03-29):
  Â_{p,2} = A_{p,2} / (σ₀^{p+2} · T⁴ · E²)
  where σ₀² = Var(Y₀) from flat branch.

For (1,2): divide by σ₀³·T⁴·E²
For (2,2): divide by σ₀⁴·T⁴·E²

Uses build_hasse_bitset_generic from sct_tools/hasse.py for ALL predicates.
"""
import sys, os, time, json
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, 'analysis')
from sct_tools.hasse import (
    sprinkle_diamond, build_hasse_bitset, build_hasse_bitset_generic,
    path_kurtosis_from_lists, path_counts,
)

sys.path.insert(0, os.path.join('analysis', 'scripts'))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, ppwave_exact_preds,
    jet_preds, bulk_mask, riemann_schwarzschild_local, Y_from_graph,
)

N_VALUES = [3000, 5000, 10000, 15000, 20000]
M_SEEDS = 20
SEED_BASE = 5000000
T = 1.0
EPS = 3.0
M_SCH = 0.05
R0 = 0.50
ZETA = 0.15

E2_PPW = EPS**2 / 2.0
E2_SCH = 6.0 * (M_SCH / R0**3)**2
R_SCH = riemann_schwarzschild_local(M_SCH, R0)


def make_strata(pts, parents0, T):
    tau_hat = 2.0 * pts[:, 0] / T
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2.0 - np.abs(pts[:, 0])
    rho_hat = np.clip(r / np.maximum(rmax, 1e-12), 0.0, 0.999999)
    tau_bin = np.clip(np.floor((tau_hat + 1.0) * 2.5).astype(int), 0, 4)
    rho_bin = np.clip(np.floor(rho_hat * 3).astype(int), 0, 2)
    depth = np.zeros(len(pts), dtype=int)
    for i in range(len(pts)):
        if parents0[i] is not None and len(parents0[i]) > 0:
            depth[i] = int(np.max(depth[parents0[i]])) + 1
    max_d = max(int(depth.max()), 1)
    depth_terc = np.clip((depth * 3) // (max_d + 1), 0, 2)
    return tau_bin * 9 + rho_bin * 3 + depth_terc


def compute_A_pq(Y0, delta, mask, strata, p, q):
    X = Y0[mask] - np.mean(Y0[mask])
    Xp = np.abs(X) ** p
    dYq = np.abs(delta[mask]) ** q
    strata_m = strata[mask]
    total = 0.0
    for label in np.unique(strata_m):
        idx = strata_m == label
        if idx.sum() < 3:
            continue
        w = idx.sum() / len(Xp)
        cov = np.mean(Xp[idx] * dYq[idx]) - np.mean(Xp[idx]) * np.mean(dYq[idx])
        total += w * cov
    return float(total)


def Y_from_hasse(parents, children):
    """Compute Y = log2(pd * pu + 1) from Hasse parents/children."""
    pd, pu = path_counts(parents, children)
    return np.log2(pd * pu + 1.0)


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("FND-1: NORMALIZED N-CONVERGENCE (BITSET)", flush=True)
    print(f"Â_{{p,2}} = A_{{p,2}} / (σ₀^{{p+2}} · T⁴ · E²)", flush=True)
    print(f"N={N_VALUES}, M={M_SEEDS}", flush=True)
    print("=" * 60, flush=True)

    t_total = time.time()
    results = {}

    for N in N_VALUES:
        print(f"\n─── N = {N} ───", flush=True)
        t_n = time.time()

        raw_12 = []
        raw_22 = []
        sigma0_list = []
        raw_sch_12 = []
        raw_sch_22 = []

        for si in range(M_SEEDS):
            seed = SEED_BASE + si
            rng = np.random.default_rng(seed)
            pts = sprinkle_local_diamond(N, T, rng)

            # Flat — bitset generic with Minkowski predicate
            par0, ch0 = build_hasse_bitset_generic(
                pts, lambda P, i: minkowski_preds(P, i))
            Y0 = Y_from_hasse(par0, ch0)
            mask = bulk_mask(pts, T, ZETA)
            strata = make_strata(pts, par0, T)

            sigma0 = float(np.std(Y0[mask]))
            sigma0_list.append(sigma0)

            # PPW exact — bitset generic
            parP, chP = build_hasse_bitset_generic(
                pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS))
            YP = Y_from_hasse(parP, chP)
            delta_ppw = YP - Y0

            raw_12.append(compute_A_pq(Y0, delta_ppw, mask, strata, 1, 2))
            raw_22.append(compute_A_pq(Y0, delta_ppw, mask, strata, 2, 2))

            # Sch jet — bitset generic
            parS, chS = build_hasse_bitset_generic(
                pts, lambda P, i: jet_preds(P, i, R_abcd=R_SCH))
            YS = Y_from_hasse(parS, chS)
            delta_sch = YS - Y0

            raw_sch_12.append(compute_A_pq(Y0, delta_sch, mask, strata, 1, 2))
            raw_sch_22.append(compute_A_pq(Y0, delta_sch, mask, strata, 2, 2))

            if (si + 1) % 5 == 0:
                elapsed = time.time() - t_n
                rate = elapsed / (si + 1)
                eta = rate * (M_SEEDS - si - 1)
                print(f"  {si+1}/{M_SEEDS} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

        sigma0_mean = np.mean(sigma0_list)

        # Raw (old normalization)
        AE_12_raw = np.mean(raw_12) / E2_PPW
        AE_22_raw = np.mean(raw_22) / E2_PPW

        # Normalized: Â = A / (σ₀^{p+2} · T⁴ · E²)
        norm_12 = [raw_12[i] / (sigma0_list[i]**3 * T**4 * E2_PPW) for i in range(M_SEEDS)]
        norm_22 = [raw_22[i] / (sigma0_list[i]**4 * T**4 * E2_PPW) for i in range(M_SEEDS)]

        Ahat_12 = np.mean(norm_12)
        Ahat_22 = np.mean(norm_22)
        Ahat_12_se = np.std(norm_12, ddof=1) / np.sqrt(M_SEEDS)
        Ahat_22_se = np.std(norm_22, ddof=1) / np.sqrt(M_SEEDS)

        # Sch normalized
        norm_sch_12 = [raw_sch_12[i] / (sigma0_list[i]**3 * T**4 * E2_SCH) for i in range(M_SEEDS)]
        norm_sch_22 = [raw_sch_22[i] / (sigma0_list[i]**4 * T**4 * E2_SCH) for i in range(M_SEEDS)]

        Ahat_sch_12 = np.mean(norm_sch_12)
        Ahat_sch_22 = np.mean(norm_sch_22)

        ratio_12 = Ahat_sch_12 / Ahat_12 if abs(Ahat_12) > 1e-15 else 0
        ratio_22 = Ahat_sch_22 / Ahat_22 if abs(Ahat_22) > 1e-15 else 0

        elapsed_n = time.time() - t_n
        print(f"  σ₀ = {sigma0_mean:.3f} ({elapsed_n:.0f}s)", flush=True)
        print(f"  (1,2) raw AE = {AE_12_raw:.6f}  |  Â = {Ahat_12:.6f} ± {Ahat_12_se:.6f}  |  Â_sch = {Ahat_sch_12:.6f}  |  ratio = {ratio_12:.3f}", flush=True)
        print(f"  (2,2) raw AE = {AE_22_raw:.6f}  |  Â = {Ahat_22:.6f} ± {Ahat_22_se:.6f}  |  Â_sch = {Ahat_sch_22:.6f}  |  ratio = {ratio_22:.3f}", flush=True)

        results[N] = {
            "sigma0_mean": sigma0_mean,
            "raw_AE_12": AE_12_raw, "raw_AE_22": AE_22_raw,
            "Ahat_12": Ahat_12, "Ahat_22": Ahat_22,
            "Ahat_12_se": float(Ahat_12_se), "Ahat_22_se": float(Ahat_22_se),
            "Ahat_sch_12": Ahat_sch_12, "Ahat_sch_22": Ahat_sch_22,
            "ratio_12": ratio_12, "ratio_22": ratio_22,
            "elapsed_s": elapsed_n,
        }

    # Convergence analysis
    print(f"\n{'='*60}", flush=True)
    print("CONVERGENCE ANALYSIS", flush=True)
    print(f"{'='*60}", flush=True)

    Ns = np.array(sorted(results.keys()), dtype=float)

    for label, key in [("(1,2) Â", "Ahat_12"), ("(2,2) Â", "Ahat_22"),
                        ("(1,2) raw", "raw_AE_12"), ("(2,2) raw", "raw_AE_22")]:
        vals = np.array([results[int(n)][key] for n in Ns])
        logN = np.log(Ns)
        logV = np.log(np.maximum(np.abs(vals), 1e-20))
        slope, _ = np.polyfit(logN, logV, 1)

        if len(vals) >= 2:
            last_change = abs(vals[-1] - vals[-2]) / abs(vals[-2]) * 100 if abs(vals[-2]) > 1e-15 else 0
        else:
            last_change = 0

        converging = "✅ CONVERGING" if last_change < 15 else "❌ GROWING" if slope > 0.1 else "⚠️ UNCLEAR"
        print(f"  {label}: slope={slope:.3f}, last_step={last_change:.1f}%  {converging}", flush=True)
        print(f"    values: {' → '.join(f'{v:.6f}' for v in vals)}", flush=True)

    print(f"\n  Ratio (1,2) jet:", flush=True)
    for n in Ns:
        r = results[int(n)]["ratio_12"]
        print(f"    N={int(n)}: {r:.3f}", flush=True)

    total_time = time.time() - t_total
    results["total_time_min"] = total_time / 60
    print(f"\nTotal: {total_time/60:.1f}min", flush=True)

    out_dir = os.path.join("analysis", "fnd1_data")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "normalized_convergence.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Saved to analysis/fnd1_data/normalized_convergence.json", flush=True)
