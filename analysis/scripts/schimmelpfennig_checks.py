#!/usr/bin/env python3
"""All Schimmelpfennig pre-emptive checks.

Tasks:
1. Var(D₁) per realization (recompute Sch M=50 with per-seed storage)
2. Power-law fit T^α with bootstrap CI
3. Bootstrap CI on Sch T=0.70 ratio/T⁴
4. Depth-stratified kurtosis
5. P₂ quadrupole power of δY
6. FLRW correct Ricci subtraction

Uses sequential computation only (no multiprocessing).
"""
import sys, os, time, json, math
import numpy as np
from scipy import stats as sp_stats
from scipy.special import logsumexp

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, excess_kurtosis, ppwave_exact_preds, jet_preds,
    bulk_mask, riemann_schwarzschild_local, riemann_ds, ricci_built_part,
    flrw_preds, riemann_flrw, ds_preds, project_l0_grid
)


def recompute_per_seed(geometry, pred_fn_factory, T_vals, M_seeds, N, zeta, seed_base):
    """Recompute dk per seed for given geometry."""
    results = {}
    for T in T_vals:
        dks = []
        for si in range(M_seeds):
            seed = seed_base + int(T * 1000) * 100 + si
            rng = np.random.default_rng(seed)
            pts = sprinkle_local_diamond(N, T, rng)
            par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
            Y0 = Y_from_graph(par0, ch0)
            pred_fn = pred_fn_factory(T)
            parF, chF = build_hasse_from_predicate(pts, pred_fn)
            YF = Y_from_graph(parF, chF)
            mask = bulk_mask(pts, T, zeta)
            dk = excess_kurtosis(YF[mask]) - excess_kurtosis(Y0[mask])
            dks.append(dk)
        results[T] = np.array(dks)
    return results


if __name__ == "__main__":
    N = 10000
    ZETA = 0.15
    t_total = time.time()

    out = {}

    # ═════════════════════════════════════════════════════════
    # TASK 1 + 3: Sch per-seed recompute (M=30, enough for CI)
    # ═════════════════════════════════════════════════════════
    print("=== TASK 1+3: Sch per-seed recompute (M=30) ===", flush=True)
    M_SCH = 0.05
    R0 = 0.50
    R_SCH = riemann_schwarzschild_local(M_SCH, R0)
    T_VALS_SCH = [1.0, 0.70, 0.50]

    sch_seeds = {}
    for T in T_VALS_SCH:
        dks = []
        t0 = time.time()
        for si in range(30):
            seed = 300000 + int(T * 1000) * 100 + si
            rng = np.random.default_rng(seed)
            pts = sprinkle_local_diamond(N, T, rng)
            par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
            Y0 = Y_from_graph(par0, ch0)
            parF, chF = build_hasse_from_predicate(pts, lambda P, i: jet_preds(P, i, R_abcd=R_SCH))
            YF = Y_from_graph(parF, chF)
            mask = bulk_mask(pts, T, ZETA)
            dk = excess_kurtosis(YF[mask]) - excess_kurtosis(Y0[mask])
            dks.append(dk)
            if (si + 1) % 10 == 0:
                print(f"  Sch T={T:.2f}: {si+1}/30 ({time.time()-t0:.0f}s)", flush=True)
        sch_seeds[T] = np.array(dks)
        m = np.mean(dks)
        std = np.std(dks, ddof=1)
        print(f"  T={T:.2f}: dk={m:+.6f}±{std/np.sqrt(len(dks)):.6f}, std={std:.6f}", flush=True)

    # TASK 1: Var(D₁)
    print("\n--- TASK 1: Per-seed fluctuation ---", flush=True)
    for T in T_VALS_SCH:
        arr = sch_seeds[T]
        print(f"  T={T:.2f}: mean={np.mean(arr):+.6f}, std={np.std(arr,ddof=1):.6f}, "
              f"CV={np.std(arr,ddof=1)/max(abs(np.mean(arr)),1e-15):.2f}, "
              f"min={np.min(arr):+.6f}, max={np.max(arr):+.6f}", flush=True)

    # TASK 3: Bootstrap CI on ratio dk(0.70)/dk(1.0) / T⁴
    print("\n--- TASK 3: Bootstrap CI on Sch T=0.70 ratio/T⁴ ---", flush=True)
    dk1 = sch_seeds[1.0]
    dk07 = sch_seeds[0.70]
    n_boot = 10000
    rng_boot = np.random.default_rng(42)
    boot_ratios = []
    for _ in range(n_boot):
        idx1 = rng_boot.integers(0, len(dk1), len(dk1))
        idx07 = rng_boot.integers(0, len(dk07), len(dk07))
        r = np.mean(dk07[idx07]) / max(np.mean(dk1[idx1]), 1e-15) / 0.70**4
        boot_ratios.append(r)
    boot_ratios = np.array(boot_ratios)
    ci_lo, ci_hi = np.percentile(boot_ratios, [2.5, 97.5])
    print(f"  ratio/T⁴ at T=0.70: median={np.median(boot_ratios):.3f}, "
          f"95% CI=[{ci_lo:.3f}, {ci_hi:.3f}]", flush=True)
    print(f"  Contains 1.0: {ci_lo <= 1.0 <= ci_hi}", flush=True)

    out["task1_sch_per_seed"] = {
        T: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)),
            "per_seed": v.tolist()}
        for T, v in sch_seeds.items()
    }
    out["task3_bootstrap_ratio"] = {
        "median": float(np.median(boot_ratios)),
        "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
        "contains_1": bool(ci_lo <= 1.0 <= ci_hi),
    }

    # ═════════════════════════════════════════════════════════
    # TASK 2: Power-law fit T^α with bootstrap CI
    # ═════════════════════════════════════════════════════════
    print("\n=== TASK 2: Power-law fit T^α ===", flush=True)

    # pp-wave fixed eps=3 data (from fixed_eps_T_scaling.json)
    ppw_data = {1.0: 0.1006, 0.70: 0.0370, 0.50: 0.0084, 0.35: 0.0026}
    ppw_se = {1.0: 0.0107, 0.70: 0.0053, 0.50: 0.0035, 0.35: 0.0018}

    # Fit dk = C * T^α using log-linear regression
    Ts = np.array(sorted(ppw_data.keys()))
    dks = np.array([ppw_data[T] for T in Ts])
    log_T = np.log(Ts)
    log_dk = np.log(dks)

    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(log_T, log_dk)
    print(f"  pp-wave (all 4 points): α = {slope:.3f} ± {std_err:.3f}, R² = {r_value**2:.4f}", flush=True)

    # Exclude T=1.0 (most contaminated by q⁴)
    Ts3 = Ts[1:]
    dks3 = dks[1:]
    slope3, intercept3, r3, p3, se3 = sp_stats.linregress(np.log(Ts3), np.log(dks3))
    print(f"  pp-wave (T≤0.70 only): α = {slope3:.3f} ± {se3:.3f}, R² = {r3**2:.4f}", flush=True)

    # Bootstrap CI on α
    n_boot = 10000
    alphas_boot = []
    for _ in range(n_boot):
        idx = rng_boot.integers(0, len(Ts), len(Ts))
        if len(set(idx)) < 2:
            continue
        s, _, _, _, _ = sp_stats.linregress(log_T[idx], log_dk[idx])
        alphas_boot.append(s)
    alphas_boot = np.array(alphas_boot)
    a_lo, a_hi = np.percentile(alphas_boot, [2.5, 97.5])
    print(f"  pp-wave α bootstrap 95% CI: [{a_lo:.2f}, {a_hi:.2f}]", flush=True)
    print(f"  Contains 4.0: {a_lo <= 4.0 <= a_hi}", flush=True)
    print(f"  Contains 2.0: {a_lo <= 2.0 <= a_hi}", flush=True)

    # Sch local (from recomputed per-seed)
    sch_means = {T: float(np.mean(sch_seeds[T])) for T in T_VALS_SCH}
    Ts_s = np.array(sorted(sch_means.keys()))
    dks_s = np.array([sch_means[T] for T in Ts_s])
    # Only positive values for log fit
    pos = dks_s > 0
    if np.sum(pos) >= 2:
        s_s, i_s, r_s, p_s, se_s = sp_stats.linregress(np.log(Ts_s[pos]), np.log(dks_s[pos]))
        print(f"  Sch local (positive dk only): α = {s_s:.3f} ± {se_s:.3f}", flush=True)
    else:
        print(f"  Sch local: not enough positive points for fit", flush=True)

    out["task2_powerlaw"] = {
        "ppw_all4": {"alpha": float(slope), "se": float(std_err), "R2": float(r_value**2)},
        "ppw_3pt": {"alpha": float(slope3), "se": float(se3), "R2": float(r3**2)},
        "ppw_bootstrap_CI": [float(a_lo), float(a_hi)],
    }

    # ═════════════════════════════════════════════════════════
    # TASK 4: Depth-stratified kurtosis
    # ═════════════════════════════════════════════════════════
    print("\n=== TASK 4: Depth-stratified kurtosis ===", flush=True)
    # Use pp-wave at T=1.0, eps=3, N=10000, 10 seeds
    EPS_PPW = 3.0
    T_STRAT = 1.0
    M_STRAT = 10

    strat_results = []
    for si in range(M_STRAT):
        seed = 500000 + int(T_STRAT * 1000) * 100 + si
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T_STRAT, rng)

        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)

        parF, chF = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS_PPW))
        YF = Y_from_graph(parF, chF)

        # Compute Hasse depth for each element
        depth = np.zeros(N, dtype=int)
        for i in range(N):
            if par0[i].size > 0:
                depth[i] = max(depth[par0[i]]) + 1

        max_depth = int(depth.max())
        n_strata = min(5, max_depth)
        boundaries = np.linspace(0, max_depth, n_strata + 1).astype(int)

        dk_per_stratum = []
        for s in range(n_strata):
            lo, hi = boundaries[s], boundaries[s + 1]
            mask_s = (depth >= lo) & (depth < hi)
            if mask_s.sum() < 20:
                dk_per_stratum.append(float('nan'))
                continue
            k0 = excess_kurtosis(Y0[mask_s])
            kF = excess_kurtosis(YF[mask_s])
            dk_per_stratum.append(kF - k0)

        strat_results.append(dk_per_stratum)
        if (si + 1) % 5 == 0:
            print(f"  Stratified: {si+1}/{M_STRAT} done", flush=True)

    # Average across seeds
    strat_arr = np.array(strat_results)
    print(f"\n  Depth strata (pp-wave T=1, eps=3, {M_STRAT} seeds):", flush=True)
    print(f"  max_depth={max_depth}, n_strata={n_strata}", flush=True)
    for s in range(n_strata):
        vals = strat_arr[:, s]
        vals_clean = vals[~np.isnan(vals)]
        if len(vals_clean) > 0:
            m = np.mean(vals_clean)
            se = np.std(vals_clean, ddof=1) / np.sqrt(len(vals_clean)) if len(vals_clean) > 1 else 0
            lo, hi = boundaries[s], boundaries[s + 1]
            print(f"  stratum {s} (depth {lo}-{hi}): dk={m:+.6f}±{se:.6f}, n_elements~{int(np.mean([(depth>=lo)&(depth<hi)]))}",
                  flush=True)

    out["task4_stratified"] = {
        "n_strata": n_strata,
        "boundaries": boundaries.tolist(),
        "per_seed": strat_arr.tolist(),
    }

    # ═════════════════════════════════════════════════════════
    # TASK 5: P₂ quadrupole power of δY
    # ═════════════════════════════════════════════════════════
    print("\n=== TASK 5: P₂ quadrupole power ===", flush=True)
    # For each element, compute angular position (θ,φ) from spatial coords
    # Project δY onto Y_{2m} and compute Σ|a_{2m}|²

    P2_results = []
    for si in range(M_STRAT):  # reuse same seeds as task 4
        seed = 500000 + int(T_STRAT * 1000) * 100 + si
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T_STRAT, rng)

        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)
        parF, chF = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS_PPW))
        YF = Y_from_graph(parF, chF)

        mask = bulk_mask(pts, T_STRAT, ZETA)
        deltaY = (YF - Y0)[mask]
        pts_bulk = pts[mask]

        # Spherical coords
        r = np.linalg.norm(pts_bulk[:, 1:], axis=1)
        r_safe = np.maximum(r, 1e-15)
        cos_theta = pts_bulk[:, 3] / r_safe  # z/r
        phi = np.arctan2(pts_bulk[:, 2], pts_bulk[:, 1])  # atan2(y,x)

        # Real spherical harmonics Y_{2m}
        # Y_20 = (1/4)√(5/π)(3cos²θ - 1)
        # Y_21c = -(1/2)√(15/π) sinθ cosθ cosφ
        # Y_21s = -(1/2)√(15/π) sinθ cosθ sinφ
        # Y_22c = (1/4)√(15/π) sin²θ cos2φ
        # Y_22s = (1/4)√(15/π) sin²θ sin2φ
        sin_theta = np.sqrt(1 - cos_theta**2)

        Y20 = 0.25 * np.sqrt(5/np.pi) * (3*cos_theta**2 - 1)
        Y21c = -0.5 * np.sqrt(15/np.pi) * sin_theta * cos_theta * np.cos(phi)
        Y21s = -0.5 * np.sqrt(15/np.pi) * sin_theta * cos_theta * np.sin(phi)
        Y22c = 0.25 * np.sqrt(15/np.pi) * sin_theta**2 * np.cos(2*phi)
        Y22s = 0.25 * np.sqrt(15/np.pi) * sin_theta**2 * np.sin(2*phi)

        harmonics = [Y20, Y21c, Y21s, Y22c, Y22s]

        # Project: a_{2m} = <δY, Y_{2m}> / <Y_{2m}, Y_{2m}>
        # (discrete inner product, not orthonormalized on Poisson set)
        a2m = [np.mean(deltaY * h) for h in harmonics]
        P2 = sum(a**2 for a in a2m)

        # Compare with global kurtosis
        dk_global = excess_kurtosis(YF[mask]) - excess_kurtosis(Y0[mask])
        var_deltaY = np.var(deltaY)

        P2_results.append({
            "P2": float(P2),
            "dk_global": float(dk_global),
            "var_deltaY": float(var_deltaY),
            "a2m": [float(a) for a in a2m],
        })

    P2_vals = [r["P2"] for r in P2_results]
    dk_vals = [r["dk_global"] for r in P2_results]
    print(f"  P₂ mean: {np.mean(P2_vals):.6f} ± {np.std(P2_vals,ddof=1)/np.sqrt(len(P2_vals)):.6f}", flush=True)
    print(f"  dk_global mean: {np.mean(dk_vals):+.6f}", flush=True)
    print(f"  Correlation P₂ vs dk: {np.corrcoef(P2_vals, dk_vals)[0,1]:.3f}", flush=True)
    print(f"  P₂/var(δY): {np.mean(P2_vals)/np.mean([r['var_deltaY'] for r in P2_results]):.4f}", flush=True)

    out["task5_P2"] = P2_results

    # ═════════════════════════════════════════════════════════
    # TASK 6: FLRW correct Ricci subtraction
    # ═════════════════════════════════════════════════════════
    print("\n=== TASK 6: FLRW Ricci subtraction ===", flush=True)
    # FLRW: conformally flat → Weyl=0. Same logic as dS fix:
    # full_mode = flrw_exact, ric_mode = flrw_exact (same predicate)
    # → delta_weyl = 0 exactly.
    # Also check: flrw_exact vs jet (buggy) residual.

    GAMMA = 2.0  # a(t) = 1 + γt²/2, R = 6γ
    T_FLRW = 0.70
    M_FLRW = 15

    dk_raw_list = []
    dk_weyl_fixed_list = []
    dk_weyl_buggy_list = []

    R_flrw = riemann_flrw(GAMMA)
    R_ric_flrw = ricci_built_part(R_flrw)

    for si in range(M_FLRW):
        seed = 450000 + int(T_FLRW * 1000) * 100 + si
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T_FLRW, rng)

        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)

        # FLRW exact
        parF, chF = build_hasse_from_predicate(pts, lambda P, i: flrw_preds(P, i, gamma=GAMMA))
        YF = Y_from_graph(parF, chF)

        # Ricci control: ALSO flrw_exact (FIXED: same predicate → delta=0)
        YRic_exact = YF  # trivially identical

        # Ricci control: jet (BUGGY)
        parJ, chJ = build_hasse_from_predicate(pts, lambda P, i: jet_preds(P, i, R_abcd=R_ric_flrw))
        YRic_jet = Y_from_graph(parJ, chJ)

        mask = bulk_mask(pts, T_FLRW, ZETA)
        k0 = excess_kurtosis(Y0[mask])

        dk_raw = excess_kurtosis(YF[mask]) - k0
        dk_weyl_fixed = excess_kurtosis((Y0 + YF - YRic_exact)[mask]) - k0  # = dk_raw trivially
        # Actually: weyl = full - ric. If ric=full: weyl=0 → Y0+0 → kurtosis(Y0) - k0 = 0
        delta_weyl_fixed = YF - YRic_exact  # = 0 vector
        dk_wf = excess_kurtosis((Y0 + delta_weyl_fixed)[mask]) - k0

        delta_weyl_buggy = YF - YRic_jet
        dk_wb = excess_kurtosis((Y0 + delta_weyl_buggy)[mask]) - k0

        dk_raw_list.append(dk_raw)
        dk_weyl_fixed_list.append(dk_wf)
        dk_weyl_buggy_list.append(dk_wb)

        if (si + 1) % 5 == 0:
            print(f"  FLRW: {si+1}/{M_FLRW}", flush=True)

    print(f"\n  FLRW γ={GAMMA}, T={T_FLRW}, M={M_FLRW}:", flush=True)
    for label, arr in [("raw", dk_raw_list), ("weyl(FIXED)", dk_weyl_fixed_list), ("weyl(BUGGY)", dk_weyl_buggy_list)]:
        a = np.array(arr)
        m = np.mean(a)
        se = np.std(a, ddof=1) / np.sqrt(len(a))
        print(f"  {label:15s}: dk={m:+.8f} ± {se:.8f}", flush=True)

    out["task6_flrw"] = {
        "gamma": GAMMA, "T": T_FLRW, "M": M_FLRW,
        "raw": {"mean": float(np.mean(dk_raw_list)), "se": float(np.std(dk_raw_list,ddof=1)/np.sqrt(M_FLRW))},
        "weyl_fixed": {"mean": float(np.mean(dk_weyl_fixed_list)), "se": float(np.std(dk_weyl_fixed_list,ddof=1)/np.sqrt(M_FLRW))},
        "weyl_buggy": {"mean": float(np.mean(dk_weyl_buggy_list)), "se": float(np.std(dk_weyl_buggy_list,ddof=1)/np.sqrt(M_FLRW))},
    }

    # ═════════════════════════════════════════════════════════
    # SAVE
    # ═════════════════════════════════════════════════════════
    total = time.time() - t_total
    print(f"\n=== TOTAL: {total:.0f}s = {total/60:.1f}min ===", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "universal_runs_v2")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "schimmelpfennig_checks.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, (np.integer,)) else o)
    print(f"Saved to {out_path}", flush=True)
