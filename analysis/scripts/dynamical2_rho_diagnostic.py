#!/usr/bin/env python3
"""
DYN-2 Phase 2: Detailed rho_eff diagnostic.

Key insight from Phase 1 benchmarks: rho_eff ~ 0.6-0.7 at N=10k-15k, NOT near 1.
The dynamical 2 arises from M_1leg_norm * (1 + rho_eff) = 2, with BOTH factors moving.

This script diagnoses:
  1. Per-element pointwise g_- vs g_+ correlation and spatial structure
  2. M_1leg_norm and rho_eff as separate convergence tracks
  3. Spatial profile rho(zeta) — where in the diamond do g_-,g_+ correlate?
  4. Time-reversal asymmetry: quantify |g_-(x) - g_+(x)| as function of position
  5. Constraint identity: does L*(1+R) = 2 hold exactly or approximately?

Uses N = 5000 (fast, sufficient for spatial diagnostics) with M = 50 trials.
"""

import os, sys, time, json, math
import numpy as np

# GPU preamble
_c = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.isdir(_c):
    os.add_dll_directory(_c)

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond,
    minkowski_preds,
    ppwave_exact_preds,
    build_hasse_from_predicate,
    bulk_mask,
    log_path_counts,
    Y_from_graph,
)
from factor41_investigation import (
    compute_depth,
    make_strata,
    cj_stratified,
)

C0_FULL = 32 * np.pi**2 / (3 * math.factorial(9) * 45)
C_AN    = C0_FULL / 4
EPS     = 3.0
E2      = EPS**2 / 2.0
T       = 1.0
ZETA    = 0.15
N_ZETA_BINS = 20


def run_diagnostic_trial(N, seed, eps=EPS):
    """Detailed per-element and per-bin diagnostic for one trial."""
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)
    n = len(pts)

    par_f, ch_f = build_hasse_from_predicate(pts, minkowski_preds)
    curved_pred = lambda pts, i, eps=eps: ppwave_exact_preds(pts, i, eps)
    par_c, ch_c = build_hasse_from_predicate(pts, curved_pred)

    log_pd_f, log_pu_f = log_path_counts(par_f, ch_f)
    log_pd_c, log_pu_c = log_path_counts(par_c, ch_c)

    LN2 = math.log(2.0)
    g_minus = (log_pd_c - log_pd_f) / LN2
    g_plus  = (log_pu_c - log_pu_f) / LN2

    # Full CJ for normalization check
    Y_flat = Y_from_graph(par_f, ch_f)
    Y_curv = Y_from_graph(par_c, ch_c)
    delta_Y = Y_curv - Y_flat

    bmask = bulk_mask(pts, T, ZETA)
    strata = make_strata(pts, par_f, T)
    strata_m = strata[bmask]

    Y0_bulk = Y_flat[bmask]
    X = Y0_bulk - np.mean(Y0_bulk)
    abs_X = np.abs(X)

    gm = g_minus[bmask]
    gp = g_plus[bmask]

    # Channel covariances
    M_mm = cj_stratified(abs_X, gm**2, strata_m)
    M_pp = cj_stratified(abs_X, gp**2, strata_m)
    M_mp = cj_stratified(abs_X, gm * gp, strata_m)

    M_ss = (M_mm + 2 * M_mp + M_pp) / 2.0
    norm = C_AN * N**(8.0/9.0) * E2 * T**4

    # Derived quantities
    M_mm_norm = M_mm / norm
    M_pp_norm = M_pp / norm
    M_mp_norm = M_mp / norm
    M_ss_norm = M_ss / norm
    M_1leg_norm = (M_mm_norm + M_pp_norm) / 2.0
    rho_eff = M_mp / math.sqrt(abs(M_mm * M_pp)) if M_mm * M_pp > 0 else 0.0

    # Check constraint: L*(1+R) = 2
    constraint_product = M_1leg_norm * (1 + rho_eff)

    # === Spatial profile ===
    depth = compute_depth(par_f, n)
    depth_bulk = depth[bmask]
    max_d = max(int(depth_bulk.max()), 1)
    zeta_rank = depth_bulk / max_d

    rho_profile = []
    gm_var_profile = []
    gp_var_profile = []
    asymmetry_profile = []
    n_per_bin = []

    for ib in range(N_ZETA_BINS):
        lo = ib / N_ZETA_BINS
        hi = (ib + 1) / N_ZETA_BINS
        sel = (zeta_rank >= lo) & (zeta_rank < hi)
        cnt = int(sel.sum())
        n_per_bin.append(cnt)

        if cnt < 10:
            rho_profile.append(None)
            gm_var_profile.append(None)
            gp_var_profile.append(None)
            asymmetry_profile.append(None)
            continue

        gm_sel = gm[sel]
        gp_sel = gp[sel]

        # Pointwise Pearson correlation in this zeta bin
        if np.std(gm_sel) > 1e-12 and np.std(gp_sel) > 1e-12:
            rho_local = float(np.corrcoef(gm_sel, gp_sel)[0, 1])
        else:
            rho_local = 0.0
        rho_profile.append(rho_local)

        # Variance of g_- and g_+ in this bin
        gm_var_profile.append(float(np.var(gm_sel)))
        gp_var_profile.append(float(np.var(gp_sel)))

        # Time-reversal asymmetry: mean |g_- - g_+| / mean(|g_-| + |g_+|)
        diff = np.abs(gm_sel - gp_sel)
        total = np.abs(gm_sel) + np.abs(gp_sel)
        asym = float(np.mean(diff) / np.mean(total)) if np.mean(total) > 1e-12 else 0.0
        asymmetry_profile.append(asym)

    # === Pointwise correlation statistics ===
    # How often is g_-(x) close to g_+(x)?
    ratio_gm_gp = gp / np.where(np.abs(gm) > 1e-8, gm, 1e-8)
    mean_ratio = float(np.mean(ratio_gm_gp[(np.abs(gm) > 0.01) & (np.abs(gp) > 0.01)]))
    median_ratio = float(np.median(ratio_gm_gp[(np.abs(gm) > 0.01) & (np.abs(gp) > 0.01)]))

    # Cosine similarity of (g_-, g_+) vectors
    dot = float(np.dot(gm, gp))
    norm_gm = float(np.linalg.norm(gm))
    norm_gp = float(np.linalg.norm(gp))
    cos_sim = dot / (norm_gm * norm_gp) if norm_gm * norm_gp > 0 else 0.0

    # Global Pearson correlation (unweighted)
    global_corr = float(np.corrcoef(gm, gp)[0, 1]) if len(gm) > 2 else 0.0

    return {
        "N": N,
        "n_bulk": int(bmask.sum()),
        "M_mm_norm": float(M_mm_norm),
        "M_pp_norm": float(M_pp_norm),
        "M_mp_norm": float(M_mp_norm),
        "M_ss_norm": float(M_ss_norm),
        "M_1leg_norm": float(M_1leg_norm),
        "rho_eff": float(rho_eff),
        "constraint_product": float(constraint_product),
        "global_corr_gm_gp": global_corr,
        "cos_sim_gm_gp": cos_sim,
        "mean_ratio_gp_gm": mean_ratio,
        "median_ratio_gp_gm": median_ratio,
        "rho_profile": rho_profile,
        "gm_var_profile": gm_var_profile,
        "gp_var_profile": gp_var_profile,
        "asymmetry_profile": asymmetry_profile,
        "n_per_bin": n_per_bin,
    }


def main():
    N_VALUES = [2000, 5000, 10000]
    M = 40
    SEED_BASE = 9900000

    print("=" * 90)
    print("DYN-2 PHASE 2: RHO-EFF DIAGNOSTIC")
    print(f"N values: {N_VALUES}, M = {M}")
    print("=" * 90)

    all_results = {}

    for N in N_VALUES:
        print(f"\n{'='*70}")
        print(f"N = {N}")
        print(f"{'='*70}")

        trials = []
        t0 = time.time()

        for trial in range(M):
            seed = SEED_BASE + N * 100 + trial
            t1 = time.time()
            res = run_diagnostic_trial(N, seed)
            dt = time.time() - t1
            trials.append(res)

            if trial == 0 or (trial + 1) % 10 == 0:
                print(f"  trial {trial+1}/{M}: M_ss_norm={res['M_ss_norm']:.4f}, "
                      f"rho_eff={res['rho_eff']:.4f}, "
                      f"L*(1+R)={res['constraint_product']:.4f}, "
                      f"corr={res['global_corr_gm_gp']:.4f}, "
                      f"cos={res['cos_sim_gm_gp']:.4f}, dt={dt:.1f}s")

        elapsed = time.time() - t0

        # Aggregate
        keys = ["M_ss_norm", "M_1leg_norm", "rho_eff", "constraint_product",
                "global_corr_gm_gp", "cos_sim_gm_gp",
                "mean_ratio_gp_gm", "median_ratio_gp_gm",
                "M_mm_norm", "M_pp_norm", "M_mp_norm"]

        agg = {}
        for k in keys:
            vals = [t[k] for t in trials]
            agg[k + "_mean"] = float(np.mean(vals))
            agg[k + "_std"] = float(np.std(vals))
            agg[k + "_se"] = float(np.std(vals) / math.sqrt(M))

        # Average profiles
        for profile_key in ["rho_profile", "gm_var_profile", "gp_var_profile", "asymmetry_profile"]:
            avg = []
            for ib in range(N_ZETA_BINS):
                vals = [t[profile_key][ib] for t in trials if t[profile_key][ib] is not None]
                avg.append(float(np.mean(vals)) if vals else None)
            agg[profile_key + "_mean"] = avg

        agg["N"] = N
        agg["M"] = M
        agg["elapsed_s"] = elapsed
        all_results[str(N)] = agg

        print(f"\n  SUMMARY N={N}:")
        print(f"    M_ss_norm       = {agg['M_ss_norm_mean']:.4f} +/- {agg['M_ss_norm_se']:.4f}")
        print(f"    M_1leg_norm     = {agg['M_1leg_norm_mean']:.4f} +/- {agg['M_1leg_norm_se']:.4f}")
        print(f"    rho_eff         = {agg['rho_eff_mean']:.4f} +/- {agg['rho_eff_se']:.4f}")
        print(f"    L*(1+R)         = {agg['constraint_product_mean']:.4f} +/- {agg['constraint_product_se']:.4f}")
        print(f"    corr(g-,g+)     = {agg['global_corr_gm_gp_mean']:.4f}")
        print(f"    cos_sim(g-,g+)  = {agg['cos_sim_gm_gp_mean']:.4f}")
        print(f"    mean(g+/g-)     = {agg['mean_ratio_gp_gm_mean']:.4f}")
        print(f"    median(g+/g-)   = {agg['median_ratio_gp_gm_mean']:.4f}")
        print(f"    rho_profile     = {agg['rho_profile_mean']}")
        print(f"    asymmetry       = {agg['asymmetry_profile_mean']}")
        print(f"    time = {elapsed:.1f}s")

    # ---- Key Analysis ----
    print("\n" + "=" * 90)
    print("KEY ANALYSIS: CONSTRAINT PRODUCT L*(1+R) vs 2")
    print("=" * 90)

    for N in N_VALUES:
        agg = all_results[str(N)]
        L = agg["M_1leg_norm_mean"]
        R = agg["rho_eff_mean"]
        product = agg["constraint_product_mean"]
        print(f"  N={N:6d}: L={L:.4f}, R={R:.4f}, L*(1+R)={product:.4f}, "
              f"deviation from 2: {product - 2:.4f}")

    print("\nIf L*(1+R) = 2 holds exactly, the dynamical 2 is a CONSTRAINT identity.")
    print("If it only holds approximately, the 2 is an ASYMPTOTIC coincidence.")

    # ---- Save ----
    output = {
        "description": "DYN-2 Phase 2: rho_eff spatial diagnostic",
        "eps": EPS, "E2": E2, "T": T,
        "C0_FULL": C0_FULL, "C_AN": C_AN,
        "n_zeta_bins": N_ZETA_BINS,
        "results": all_results,
    }

    out_path = os.path.join(os.path.dirname(__file__),
                            "..", "fnd1_data", "dynamical2_rho_diagnostic.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
