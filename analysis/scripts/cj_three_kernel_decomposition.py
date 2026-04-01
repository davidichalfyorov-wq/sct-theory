#!/usr/bin/env python3
"""
Three-kernel decomposition of CJ: measuring the "+1" correction and
identifying the source of the ~4% excess in M_-- + 2*M_-+ + M_++.

For each N, we build flat and curved (pp-wave, eps=3) Hasse diagrams
on the SAME point set (CRN), compute:

  Y_flat  = log2(p_down_flat * p_up_flat + 1)
  Y_curv  = log2(p_down_curv * p_up_curv + 1)
  delta_Y = Y_curv - Y_flat

  g_minus = log2(p_down_curv) - log2(p_down_flat)
  g_plus  = log2(p_up_curv)   - log2(p_up_flat)

  delta_Y_approx = g_minus + g_plus   (from log product rule, ignoring "+1")

Full CJ:
  CJ_full = sum_B w_B Cov_B(|X|, delta_Y^2)

Decomposed (using delta_Y_approx = g_- + g_+):
  M_--  = sum_B w_B Cov_B(|X|, g_-^2)
  M_++  = sum_B w_B Cov_B(|X|, g_+^2)
  M_-+  = sum_B w_B Cov_B(|X|, g_- * g_+)
  CJ_decomp = M_-- + 2*M_-+ + M_++

Correction from "+1":
  delta_correction = delta_Y_exact - delta_Y_approx
  frac_correction  = mean(delta_correction^2) / mean(delta_Y_exact^2)

GPU is used for N >= 5000 (causal matrix construction).
"""

import os, sys, time, json, math
import numpy as np
from scipy.special import logsumexp

# GPU preamble (mandatory on Windows)
_c = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.isdir(_c):
    os.add_dll_directory(_c)

# Import from existing infrastructure
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
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
C0_FULL = 32 * np.pi**2 / (3 * math.factorial(9) * 45)  # = 6.4469e-6
C_AN    = C0_FULL / 4                                     # = 1.6117e-6
EPS     = 3.0
E2      = EPS**2 / 2.0   # = 4.5
T       = 1.0
ZETA    = 0.15


def log2_safe(x):
    """log2 of positive values, returns -inf for x<=0."""
    return np.where(x > 0, np.log2(x), -100.0)


def run_trial(N, seed, use_gpu=False):
    """Single CRN trial: same points, flat vs curved Hasse."""
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)
    n = len(pts)

    # Build Hasse diagrams (flat and curved)
    par_f, ch_f = build_hasse_from_predicate(pts, minkowski_preds)
    curved_pred = lambda pts, i, eps=EPS: ppwave_exact_preds(pts, i, eps)
    par_c, ch_c = build_hasse_from_predicate(pts, curved_pred)

    # Log path counts (natural log)
    log_pd_f, log_pu_f = log_path_counts(par_f, ch_f)
    log_pd_c, log_pu_c = log_path_counts(par_c, ch_c)

    # Y values (log2 of product + 1)
    Y_flat = Y_from_graph(par_f, ch_f)
    Y_curv = Y_from_graph(par_c, ch_c)
    delta_Y_exact = Y_curv - Y_flat

    # Decomposition: g_minus, g_plus (in log2)
    LN2 = math.log(2.0)
    log2_pd_f = log_pd_f / LN2
    log2_pu_f = log_pu_f / LN2
    log2_pd_c = log_pd_c / LN2
    log2_pu_c = log_pu_c / LN2

    g_minus = log2_pd_c - log2_pd_f   # change in log2(p_down)
    g_plus  = log2_pu_c - log2_pu_f   # change in log2(p_up)

    delta_Y_approx = g_minus + g_plus  # log product approximation

    # The "+1" correction
    delta_correction = delta_Y_exact - delta_Y_approx

    # Bulk mask and strata
    bmask = bulk_mask(pts, T, ZETA)
    strata = make_strata(pts, par_f, T)
    strata_m = strata[bmask]

    # Weight: |X| where X = Y_flat - mean(Y_flat) in bulk
    Y0_bulk = Y_flat[bmask]
    X = Y0_bulk - np.mean(Y0_bulk)
    abs_X = np.abs(X)

    # --- Full CJ ---
    dY2_exact = delta_Y_exact[bmask]**2
    CJ_full = cj_stratified(abs_X, dY2_exact, strata_m)

    # --- Decomposed kernels ---
    gm = g_minus[bmask]
    gp = g_plus[bmask]

    M_mm = cj_stratified(abs_X, gm**2, strata_m)
    M_pp = cj_stratified(abs_X, gp**2, strata_m)
    M_mp = cj_stratified(abs_X, gm * gp, strata_m)
    CJ_decomp = M_mm + 2 * M_mp + M_pp

    # --- Correction diagnostics ---
    dc = delta_correction[bmask]
    dY_ex = delta_Y_exact[bmask]

    # Mean squared correction relative to mean squared exact delta
    mse_correction = float(np.mean(dc**2))
    mse_exact = float(np.mean(dY_ex**2))
    frac_correction = mse_correction / mse_exact if mse_exact > 0 else 0.0

    # Cross term: 2 * mean(delta_approx * delta_correction) in bulk
    dY_ap = delta_Y_approx[bmask]
    cross_term = float(np.mean(dY_ap * dc))

    # Also compute CJ using approx delta (to double-check decomposition)
    dY2_approx = delta_Y_approx[bmask]**2
    CJ_approx = cj_stratified(abs_X, dY2_approx, strata_m)

    # Symmetric/antisymmetric decomposition
    M_ss = (M_mm + 2 * M_mp + M_pp) / 2.0
    M_aa = (M_mm - 2 * M_mp + M_pp) / 2.0
    M_sa = (M_mm - M_pp) / 2.0

    return {
        "CJ_full": CJ_full,
        "CJ_decomp": CJ_decomp,
        "CJ_approx": CJ_approx,
        "M_mm": M_mm,
        "M_pp": M_pp,
        "M_mp": M_mp,
        "M_ss": M_ss,
        "M_aa": M_aa,
        "M_sa": M_sa,
        "frac_correction": frac_correction,
        "mse_correction": mse_correction,
        "mse_exact": mse_exact,
        "cross_term": cross_term,
        "n_bulk": int(bmask.sum()),
        "n_total": n,
    }


def run_N_scan(N_values, M=30, seed_base=7700000):
    """Run full scan across N values."""
    print("=" * 90)
    print("THREE-KERNEL DECOMPOSITION OF CJ")
    print(f"eps={EPS}, E^2={E2}, T={T}, M={M} trials per N")
    print(f"C_0(full) = {C0_FULL:.6e},  C_AN = C_0/4 = {C_AN:.6e}")
    print("=" * 90)

    all_results = {}

    for N in N_values:
        print(f"\n{'='*80}")
        print(f"  N = {N}   (M={M} trials)")
        print(f"{'='*80}")

        trial_results = []
        t_start = time.time()

        for trial in range(M):
            seed = seed_base + N * 100 + trial
            t0 = time.time()
            res = run_trial(N, seed, use_gpu=(N >= 5000))
            elapsed = time.time() - t0

            trial_results.append(res)

            if trial < 3 or trial == M - 1 or (trial + 1) % 10 == 0:
                print(f"  [{trial+1:2d}/{M}] CJ_full={res['CJ_full']:.4e}  "
                      f"CJ_decomp={res['CJ_decomp']:.4e}  "
                      f"frac_corr={res['frac_correction']:.4e}  "
                      f"({elapsed:.1f}s)")

        total_time = time.time() - t_start

        # Aggregate
        N89 = N**(8/9)
        norm = C0_FULL * N89 * E2 * T**4
        norm_AN = C_AN * N89 * E2 * T**4

        keys = ["CJ_full", "CJ_decomp", "CJ_approx",
                "M_mm", "M_pp", "M_mp", "M_ss", "M_aa", "M_sa",
                "frac_correction", "cross_term"]

        agg = {}
        for k in keys:
            vals = np.array([r[k] for r in trial_results])
            agg[k] = {
                "mean": float(np.mean(vals)),
                "se": float(np.std(vals, ddof=1) / np.sqrt(M)),
                "std": float(np.std(vals, ddof=1)),
            }

        # Ratios
        R_full   = agg["CJ_full"]["mean"] / norm
        R_decomp = agg["CJ_decomp"]["mean"] / norm
        R_approx = agg["CJ_approx"]["mean"] / norm
        diff     = R_decomp - R_full
        frac_c   = agg["frac_correction"]["mean"]

        # Per-kernel normalized by C_AN * N^{8/9} * E^2
        M_mm_n = agg["M_mm"]["mean"] / norm_AN
        M_pp_n = agg["M_pp"]["mean"] / norm_AN
        M_mp_n = agg["M_mp"]["mean"] / norm_AN
        M_ss_n = agg["M_ss"]["mean"] / norm_AN
        M_aa_n = agg["M_aa"]["mean"] / norm_AN
        M_sa_n = agg["M_sa"]["mean"] / norm_AN

        print(f"\n  --- N={N} SUMMARY ({total_time:.0f}s total) ---")
        print(f"  R_full    = CJ_full  / (C0*N^{{8/9}}*E2*T4) = {R_full:.4f} +/- {agg['CJ_full']['se']/norm:.4f}")
        print(f"  R_decomp  = (M--+2M-++M++) / (C0*...)       = {R_decomp:.4f} +/- {agg['CJ_decomp']['se']/norm:.4f}")
        print(f"  R_approx  = CJ(dY_approx) / (C0*...)        = {R_approx:.4f} +/- {agg['CJ_approx']['se']/norm:.4f}")
        print(f"  DIFF(decomp-full) = {diff:.5f}  ({100*diff/R_full:.2f}% of R_full)")
        print(f"  mean frac_correction = {frac_c:.5e}")
        print(f"  mean cross_term      = {agg['cross_term']['mean']:.5e}")
        print(f"")
        print(f"  Per-kernel (normalized by C_AN*N^{{8/9}}*E2):")
        print(f"    M_-- / C_AN_norm = {M_mm_n:.4f}")
        print(f"    M_++ / C_AN_norm = {M_pp_n:.4f}")
        print(f"    M_-+ / C_AN_norm = {M_mp_n:.4f}")
        print(f"    M_ss = (M--+2M-++M++)/2 / C_AN_norm = {M_ss_n:.4f}")
        print(f"    M_aa = (M---2M-++M++)/2 / C_AN_norm = {M_aa_n:.4f}")
        print(f"    M_sa = (M---M++)/2      / C_AN_norm = {M_sa_n:.4f}")

        all_results[N] = {
            "N": N, "M": M, "time_s": total_time,
            "R_full": R_full, "R_decomp": R_decomp, "R_approx": R_approx,
            "diff_decomp_full": diff,
            "frac_correction": frac_c,
            "agg": {k: agg[k] for k in keys},
            "M_mm_norm": M_mm_n, "M_pp_norm": M_pp_n, "M_mp_norm": M_mp_n,
            "M_ss_norm": M_ss_n, "M_aa_norm": M_aa_n, "M_sa_norm": M_sa_n,
        }

    # ---- Final summary table ----
    print("\n\n" + "=" * 110)
    print("FINAL SUMMARY TABLE")
    print("=" * 110)
    header = (f"{'N':>6} | {'R_full':>8} | {'R_decomp':>9} | {'R_approx':>9} | "
              f"{'diff':>8} | {'%':>6} | {'frac_corr':>10} | "
              f"{'M--/C_AN':>9} | {'M++/C_AN':>9} | {'M-+/C_AN':>9} | {'M_ss':>8} | {'M_aa':>8}")
    print(header)
    print("-" * 110)
    for N in N_values:
        r = all_results[N]
        print(f"{N:>6} | {r['R_full']:>8.4f} | {r['R_decomp']:>9.4f} | {r['R_approx']:>9.4f} | "
              f"{r['diff_decomp_full']:>8.5f} | {100*r['diff_decomp_full']/r['R_full']:>5.2f}% | "
              f"{r['frac_correction']:>10.2e} | "
              f"{r['M_mm_norm']:>9.4f} | {r['M_pp_norm']:>9.4f} | {r['M_mp_norm']:>9.4f} | "
              f"{r['M_ss_norm']:>8.4f} | {r['M_aa_norm']:>8.4f}")
    print("=" * 110)

    # Key diagnostic: does frac_correction scale with N?
    print("\n  +1 CORRECTION DIAGNOSTIC:")
    print(f"  {'N':>6} | {'frac_corr':>12} | {'sqrt(frac)':>12} | {'1/N':>12}")
    print("  " + "-" * 55)
    for N in N_values:
        fc = all_results[N]["frac_correction"]
        print(f"  {N:>6} | {fc:>12.4e} | {math.sqrt(fc):>12.4e} | {1/N:>12.4e}")

    # Identity check: CJ_full = CJ_approx + correction terms
    print("\n  IDENTITY CHECK: CJ_full vs CJ_approx:")
    print(f"  {'N':>6} | {'CJ_full':>12} | {'CJ_approx':>12} | {'ratio':>8} | {'diff%':>8}")
    print("  " + "-" * 60)
    for N in N_values:
        r = all_results[N]
        ratio = r['R_approx'] / r['R_full'] if r['R_full'] != 0 else 0
        diff_pct = 100 * (r['R_approx'] - r['R_full']) / r['R_full']
        print(f"  {N:>6} | {r['R_full']:>12.6f} | {r['R_approx']:>12.6f} | {ratio:>8.5f} | {diff_pct:>+7.3f}%")

    # Save
    outdir = os.path.join(os.path.dirname(__file__), "..", "fnd1_data")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "cj_three_kernel_decomposition.json")
    with open(outpath, "w") as f:
        json.dump({str(N): all_results[N] for N in N_values}, f, indent=2, default=float)
    print(f"\nSaved -> {outpath}")

    return all_results


if __name__ == "__main__":
    t_total = time.time()

    # Full scan: 1000, 2000, 3000, 5000, 8000, 10000
    N_values = [1000, 2000, 3000, 5000, 8000, 10000]
    results = run_N_scan(N_values, M=30)

    print(f"\nTotal runtime: {time.time()-t_total:.0f}s")
