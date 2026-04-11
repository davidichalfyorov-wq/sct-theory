#!/usr/bin/env python3
"""
DYN-2: High-N attack on the dynamical normalization factor 2.

Target: M_ss / (C_AN * N^{8/9} * E^2 * T^4) -> 2 as N -> infinity.

Strategy:
  1. Extend channel decomposition to N = 15000, 20000, 30000 (GPU)
  2. Measure rho_eff = M_mp / sqrt(M_mm * M_pp)  -- the key diagnostic
  3. Per-bin M_ss decomposition to identify spatial structure
  4. Convergence fit: M_ss_norm(N) = 2 + a * N^{-alpha}
  5. Richardson extrapolation

Uses CRN (common random numbers) throughout.
GPU used for causal matrix construction at all N.
"""

import os, sys, time, json, math
import numpy as np
from scipy.optimize import curve_fit

# GPU preamble (mandatory on Windows)
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

# ---- Constants ----
C0_FULL = 32 * np.pi**2 / (3 * math.factorial(9) * 45)
C_AN    = C0_FULL / 4
EPS     = 3.0
E2      = EPS**2 / 2.0
T       = 1.0
ZETA    = 0.15


def run_trial_extended(N, seed, eps=EPS):
    """Single CRN trial with extended diagnostics.

    Returns channel decomposition + per-bin M_ss + per-element g data.
    """
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)
    n = len(pts)

    # Build Hasse diagrams (flat and curved)
    par_f, ch_f = build_hasse_from_predicate(pts, minkowski_preds)
    curved_pred = lambda pts, i, eps=eps: ppwave_exact_preds(pts, i, eps)
    par_c, ch_c = build_hasse_from_predicate(pts, curved_pred)

    # Log path counts
    log_pd_f, log_pu_f = log_path_counts(par_f, ch_f)
    log_pd_c, log_pu_c = log_path_counts(par_c, ch_c)

    LN2 = math.log(2.0)
    g_minus = (log_pd_c - log_pd_f) / LN2
    g_plus  = (log_pu_c - log_pu_f) / LN2

    # Full CJ
    Y_flat = Y_from_graph(par_f, ch_f)
    Y_curv = Y_from_graph(par_c, ch_c)
    delta_Y = Y_curv - Y_flat

    # Bulk mask and strata
    bmask = bulk_mask(pts, T, ZETA)
    strata = make_strata(pts, par_f, T)
    strata_m = strata[bmask]

    # Weight: |X|
    Y0_bulk = Y_flat[bmask]
    X = Y0_bulk - np.mean(Y0_bulk)
    abs_X = np.abs(X)

    # Bulk g values
    gm = g_minus[bmask]
    gp = g_plus[bmask]

    # --- Channel covariances ---
    M_mm = cj_stratified(abs_X, gm**2, strata_m)
    M_pp = cj_stratified(abs_X, gp**2, strata_m)
    M_mp = cj_stratified(abs_X, gm * gp, strata_m)

    M_ss = (M_mm + 2 * M_mp + M_pp) / 2.0
    M_aa = (M_mm - 2 * M_mp + M_pp) / 2.0
    M_sa = (M_mm - M_pp) / 2.0

    # Full CJ
    dY2 = delta_Y[bmask]**2
    CJ_full = cj_stratified(abs_X, dY2, strata_m)

    # --- rho_eff: correlation coefficient ---
    denom = math.sqrt(abs(M_mm * M_pp)) if M_mm * M_pp > 0 else 1e-30
    rho_eff = M_mp / denom

    # --- Per-bin M_ss decomposition ---
    unique_bins = np.unique(strata_m)
    per_bin = {}
    for b in unique_bins:
        idx = strata_m == b
        if idx.sum() < 3:
            continue
        w = idx.sum() / len(abs_X)
        cov_mm = (np.mean(abs_X[idx] * gm[idx]**2)
                  - np.mean(abs_X[idx]) * np.mean(gm[idx]**2))
        cov_pp = (np.mean(abs_X[idx] * gp[idx]**2)
                  - np.mean(abs_X[idx]) * np.mean(gp[idx]**2))
        cov_mp = (np.mean(abs_X[idx] * gm[idx] * gp[idx])
                  - np.mean(abs_X[idx]) * np.mean(gm[idx] * gp[idx]))
        mss_b = (cov_mm + 2 * cov_mp + cov_pp) / 2.0
        per_bin[int(b)] = {
            "w": float(w),
            "n": int(idx.sum()),
            "M_ss_contrib": float(w * mss_b),
            "M_mm_contrib": float(w * cov_mm),
            "rho_local": float(cov_mp / math.sqrt(abs(cov_mm * cov_pp)))
                         if cov_mm * cov_pp > 0 else 0.0,
        }

    # --- Per-element g_- vs g_+ correlation (pointwise) ---
    # Pearson correlation in bulk
    if len(gm) > 10:
        corr_gm_gp = float(np.corrcoef(gm, gp)[0, 1])
    else:
        corr_gm_gp = 0.0

    # --- Normalized rank for spatial profile ---
    # Rank = fractional depth position
    depth_all = compute_depth(par_f, n)
    depth_bulk = depth_all[bmask]
    max_d = max(int(depth_bulk.max()), 1)
    zeta_rank = depth_bulk / max_d  # 0 = bottom, 1 = top

    # Binned rho_eff(zeta) profile
    n_zeta_bins = 10
    rho_profile = []
    for ib in range(n_zeta_bins):
        lo = ib / n_zeta_bins
        hi = (ib + 1) / n_zeta_bins
        sel = (zeta_rank >= lo) & (zeta_rank < hi)
        if sel.sum() < 10:
            rho_profile.append(None)
            continue
        gm_sel = gm[sel]
        gp_sel = gp[sel]
        c = float(np.corrcoef(gm_sel, gp_sel)[0, 1]) if len(gm_sel) > 2 else 0.0
        rho_profile.append(c)

    # Normalization
    norm = C_AN * N**(8.0/9.0) * E2 * T**4
    M_ss_norm = M_ss / norm if norm > 0 else 0.0
    M_aa_norm = M_aa / norm if norm > 0 else 0.0
    M_sa_norm = M_sa / norm if norm > 0 else 0.0
    R_full = CJ_full / (C0_FULL * N**(8.0/9.0) * E2 * T**4) if norm > 0 else 0.0

    return {
        "N": N,
        "n_bulk": int(bmask.sum()),
        "CJ_full": float(CJ_full),
        "M_mm": float(M_mm),
        "M_pp": float(M_pp),
        "M_mp": float(M_mp),
        "M_ss": float(M_ss),
        "M_aa": float(M_aa),
        "M_sa": float(M_sa),
        "M_ss_norm": float(M_ss_norm),
        "M_aa_norm": float(M_aa_norm),
        "M_sa_norm": float(M_sa_norm),
        "R_full": float(R_full),
        "rho_eff": float(rho_eff),
        "corr_gm_gp_pointwise": corr_gm_gp,
        "rho_profile": rho_profile,
        "per_bin_count": len(per_bin),
    }


def convergence_fit(N_vals, M_ss_norms):
    """Fit M_ss_norm(N) = 2 + a * N^{-alpha}."""
    def model(N, a, alpha):
        return 2.0 + a * N**(-alpha)

    try:
        popt, pcov = curve_fit(model, N_vals, M_ss_norms,
                               p0=[1.0, 0.3], maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        return {
            "a": float(popt[0]),
            "alpha": float(popt[1]),
            "a_err": float(perr[0]),
            "alpha_err": float(perr[1]),
            "residual_rms": float(np.sqrt(np.mean(
                (M_ss_norms - model(N_vals, *popt))**2))),
        }
    except Exception as e:
        return {"error": str(e)}


def rho_convergence_fit(N_vals, rho_vals):
    """Fit rho_eff(N) = rho_inf + b * N^{-beta}."""
    def model(N, rho_inf, b, beta):
        return rho_inf + b * N**(-beta)

    try:
        popt, pcov = curve_fit(model, N_vals, rho_vals,
                               p0=[1.0, -1.0, 0.3], maxfev=5000)
        perr = np.sqrt(np.diag(pcov))
        return {
            "rho_inf": float(popt[0]),
            "b": float(popt[1]),
            "beta": float(popt[2]),
            "rho_inf_err": float(perr[0]),
            "residual_rms": float(np.sqrt(np.mean(
                (rho_vals - model(N_vals, *popt))**2))),
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    # Include existing data points + new high-N
    N_VALUES = [1000, 3000, 5000, 10000, 15000, 20000]
    M = 20
    SEED_BASE = 8800000

    print("=" * 90)
    print("DYN-2: DYNAMICAL NORMALIZATION HIGH-N ATTACK")
    print(f"N values: {N_VALUES}")
    print(f"M = {M} CRN trials per N, eps = {EPS}, T = {T}")
    print(f"C_0 = {C0_FULL:.6e},  C_AN = {C_AN:.6e}")
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
            res = run_trial_extended(N, seed)
            dt = time.time() - t1
            trials.append(res)

            if trial == 0 or (trial + 1) % 10 == 0:
                print(f"  trial {trial+1}/{M}: M_ss_norm={res['M_ss_norm']:.4f}, "
                      f"rho_eff={res['rho_eff']:.4f}, "
                      f"corr(g-,g+)={res['corr_gm_gp_pointwise']:.4f}, "
                      f"dt={dt:.1f}s")

        elapsed = time.time() - t0

        # Aggregate
        keys = ["M_ss_norm", "M_aa_norm", "M_sa_norm", "R_full",
                "rho_eff", "corr_gm_gp_pointwise",
                "M_mm", "M_pp", "M_mp", "M_ss", "CJ_full"]

        agg = {}
        for k in keys:
            vals = [t[k] for t in trials]
            agg[k + "_mean"] = float(np.mean(vals))
            agg[k + "_std"] = float(np.std(vals))
            agg[k + "_se"] = float(np.std(vals) / math.sqrt(M))

        # Average rho profile
        profiles = [t["rho_profile"] for t in trials]
        avg_profile = []
        for ib in range(10):
            vals = [p[ib] for p in profiles if p[ib] is not None]
            if vals:
                avg_profile.append(float(np.mean(vals)))
            else:
                avg_profile.append(None)

        agg["rho_profile_mean"] = avg_profile
        agg["N"] = N
        agg["M"] = M
        agg["elapsed_s"] = elapsed

        all_results[str(N)] = agg

        print(f"\n  SUMMARY N={N}:")
        print(f"    M_ss_norm = {agg['M_ss_norm_mean']:.4f} +/- {agg['M_ss_norm_se']:.4f}")
        print(f"    rho_eff   = {agg['rho_eff_mean']:.4f} +/- {agg['rho_eff_se']:.4f}")
        print(f"    corr(g-,g+) = {agg['corr_gm_gp_pointwise_mean']:.4f}")
        print(f"    M_aa_norm = {agg['M_aa_norm_mean']:.4f}")
        print(f"    M_sa_norm = {agg['M_sa_norm_mean']:.4f}")
        print(f"    R_full    = {agg['R_full_mean']:.4f}")
        print(f"    rho_profile = {avg_profile}")
        print(f"    time = {elapsed:.1f}s")

    # ---- Convergence fits ----
    N_arr = np.array([int(k) for k in all_results.keys()])
    M_ss_arr = np.array([all_results[str(n)]["M_ss_norm_mean"] for n in N_arr])
    rho_arr = np.array([all_results[str(n)]["rho_eff_mean"] for n in N_arr])

    print("\n" + "=" * 70)
    print("CONVERGENCE FITS")
    print("=" * 70)

    mss_fit = convergence_fit(N_arr, M_ss_arr)
    print(f"\nM_ss_norm(N) = 2 + a * N^{{-alpha}}:")
    print(f"  {mss_fit}")

    rho_fit = rho_convergence_fit(N_arr, rho_arr)
    print(f"\nrho_eff(N) = rho_inf + b * N^{{-beta}}:")
    print(f"  {rho_fit}")

    # ---- Key diagnostic ----
    print("\n" + "=" * 70)
    print("KEY DIAGNOSTICS")
    print("=" * 70)

    rho_inf = rho_fit.get("rho_inf", "?")
    print(f"\nrho_inf = {rho_inf}")
    if isinstance(rho_inf, float):
        if abs(rho_inf - 1.0) < 0.05:
            print("  --> rho_eff -> 1: DYNAMICAL 2 EXPLAINED by g_- ~ g_+ correlation")
            print("  --> Factor 4 = 2_alg * 2(1+rho_eff) / 2 where 2(1+1)/2 = 2")
        else:
            print(f"  --> rho_eff -> {rho_inf:.4f} != 1: mechanism is DIFFERENT")

    # ---- Save ----
    output = {
        "description": "DYN-2 high-N dynamical normalization attack",
        "eps": EPS,
        "E2": E2,
        "T": T,
        "C0_FULL": C0_FULL,
        "C_AN": C_AN,
        "zeta_cut": ZETA,
        "M_per_N": M,
        "results": all_results,
        "convergence_fit_Mss": mss_fit,
        "convergence_fit_rho": rho_fit,
    }

    out_path = os.path.join(os.path.dirname(__file__),
                            "..", "fnd1_data", "dynamical2_highN.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
