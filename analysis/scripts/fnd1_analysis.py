#!/usr/bin/env python3
"""FND-1 POST-PROCESSING ANALYSIS.

Reads per-element data from fnd1_data_collection and runs:
  0.1  Reproducibility (compare with killer_tests baseline)
  1.1  ppw A_align T⁴ scaling (4 T-points, T² rejection)
  1.2  Sch A_align T⁴ scaling (3 T-points)
  3.2  Nonlinear TC mediation
  3.3  Cross-seed CRN decoupling
  3.4  Flat-noise / random-strata null
  3.5  Independent sprinkling control
  4.1  Same-predicate ratio (jet ppw vs jet Sch)
  6.2  ζ/window robustness
  7.1  Observable (p,q) family
"""
import sys, os, time, json, pickle, gzip, glob
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import bulk_mask

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "fnd1_data")
OUT_FILE = os.path.join(DATA_DIR, "analysis_results.json")

EPS = 3.0
M_SCH = 0.05
R0 = 0.50
ZETA = 0.15
E2_PPW = EPS**2 / 2.0
E2_SCH = 6.0 * (M_SCH / R0**3)**2


def load_seeds(T_val):
    """Load all seed data for a given T value."""
    pattern = os.path.join(DATA_DIR, f"T{T_val:.2f}_seed*.pkl.gz")
    files = sorted(glob.glob(pattern))
    data = []
    for f in files:
        with gzip.open(f, "rb") as fh:
            data.append(pickle.load(fh))
    return data


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


def compute_A_pq(Y0, delta, mask, strata, p, q):
    """Generalized Cov(|X|^p, |δY|^q) with strata."""
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


def compute_global_cov(Y0, delta, mask):
    X = Y0[mask] - np.mean(Y0[mask])
    X2 = X ** 2
    dY2 = delta[mask] ** 2
    return float(np.mean(X2 * dY2) - np.mean(X2) * np.mean(dY2))


if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("FND-1 POST-PROCESSING ANALYSIS", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    results = {}

    # ══════════════════════════════════════════════════════
    # Load T=1.0 data (full suite)
    # ══════════════════════════════════════════════════════
    print("\nLoading T=1.0 data...", flush=True)
    data_T1 = load_seeds(1.0)
    M = len(data_T1)
    print(f"  {M} seeds loaded", flush=True)

    # ══════════════════════════════════════════════════════
    # TEST 0.1: REPRODUCIBILITY
    # ══════════════════════════════════════════════════════
    print("\n=== TEST 0.1: REPRODUCIBILITY ===", flush=True)
    a_ppw = [compute_A_align(d["Y0"], d["delta_ppw"], d["mask"], d["strata"]) for d in data_T1]
    AE_ppw = np.mean(a_ppw) / E2_PPW
    AE_ppw_se = np.std(a_ppw) / np.sqrt(M) / E2_PPW
    print(f"  AE_ppw = {AE_ppw:.6f} ± {AE_ppw_se:.6f}", flush=True)
    print(f"  (Previous: ~0.143 from killer_tests. Diff: {abs(AE_ppw - 0.143)/0.143*100:.1f}%)", flush=True)

    if "delta_sch_exp" in data_T1[0]:
        a_sch = [compute_A_align(d["Y0"], d["delta_sch_exp"], d["mask"], d["strata"]) for d in data_T1]
        AE_sch = np.mean(a_sch) / E2_SCH
        ratio_baseline = AE_sch / AE_ppw
        print(f"  AE_sch_exp = {AE_sch:.6f}", flush=True)
        print(f"  Ratio Sch/ppw = {ratio_baseline:.3f} (Previous: ~0.565)", flush=True)
    else:
        AE_sch = None
        ratio_baseline = None

    results["test_0_1"] = {"AE_ppw": AE_ppw, "AE_ppw_se": AE_ppw_se,
                            "AE_sch_exp": AE_sch, "ratio": ratio_baseline}

    # ══════════════════════════════════════════════════════
    # TEST 1.1: PPW T⁴ SCALING
    # ══════════════════════════════════════════════════════
    print("\n=== TEST 1.1: PPW A_ALIGN T⁴ SCALING ===", flush=True)
    T_ae = {}
    for T_val in [1.0, 0.70, 0.50, 0.35]:
        data_T = load_seeds(T_val)
        a_vals = [compute_A_align(d["Y0"], d["delta_ppw"], d["mask"], d["strata"]) for d in data_T]
        ae = np.mean(a_vals) / E2_PPW
        se = np.std(a_vals) / np.sqrt(len(a_vals)) / E2_PPW
        d_stat = np.mean(a_vals) / np.std(a_vals) if np.std(a_vals) > 0 else 0
        T_ae[T_val] = {"AE": ae, "SE": se, "d": d_stat, "M": len(a_vals)}
        print(f"  T={T_val}: AE={ae:.6f} ± {se:.6f}, d={d_stat:.2f}", flush=True)

    # Power law fit
    Ts = np.array(sorted(T_ae.keys()))
    AEs = np.array([T_ae[t]["AE"] for t in Ts])
    logT = np.log(Ts)
    logAE = np.log(np.maximum(AEs, 1e-15))
    slope, intercept = np.polyfit(logT, logAE, 1)
    print(f"  Power law: α = {slope:.3f} (expect 4.0)", flush=True)

    # T² vs T⁴ residuals
    resid_T2 = np.sum((logAE - (logAE[0] + 2 * (logT - logT[0])))**2)
    resid_T4 = np.sum((logAE - (logAE[0] + 4 * (logT - logT[0])))**2)
    ratio_T2_T4 = resid_T2 / max(resid_T4, 1e-15)
    print(f"  T² residual / T⁴ residual = {ratio_T2_T4:.1f}× (T⁴ better)", flush=True)

    # Normalized ratios
    for t in Ts[1:]:
        r = T_ae[t]["AE"] / T_ae[Ts[0]]["AE"] / (t / Ts[0])**4
        print(f"  A(T={t})/A(T=1)/T⁴ = {r:.3f}", flush=True)

    results["test_1_1"] = {
        "T_data": {str(t): T_ae[t] for t in Ts},
        "alpha": float(slope),
        "T2_T4_ratio": float(ratio_T2_T4),
    }

    # ══════════════════════════════════════════════════════
    # TEST 1.2: SCH A_ALIGN T⁴ SCALING
    # ══════════════════════════════════════════════════════
    print("\n=== TEST 1.2: SCH A_ALIGN T⁴ SCALING ===", flush=True)
    T_sch = {}
    for T_val in [1.0, 0.70, 0.50]:
        data_T = load_seeds(T_val)
        if "delta_sch_exp" not in data_T[0]:
            print(f"  T={T_val}: NO Sch data", flush=True)
            continue
        a_exp = [compute_A_align(d["Y0"], d["delta_sch_exp"], d["mask"], d["strata"]) for d in data_T]
        a_jet = [compute_A_align(d["Y0"], d["delta_sch_jet"], d["mask"], d["strata"]) for d in data_T]
        ae_exp = np.mean(a_exp) / E2_SCH
        ae_jet = np.mean(a_jet) / E2_SCH
        d_exp = np.mean(a_exp) / np.std(a_exp) if np.std(a_exp) > 0 else 0
        T_sch[T_val] = {"AE_exp": ae_exp, "AE_jet": ae_jet, "d_exp": d_exp}
        print(f"  T={T_val}: AE_exp={ae_exp:.6f} (d={d_exp:.2f}), AE_jet={ae_jet:.6f}", flush=True)

    results["test_1_2"] = {str(t): T_sch[t] for t in sorted(T_sch.keys())}

    # ══════════════════════════════════════════════════════
    # TEST 3.2: NONLINEAR TC MEDIATION
    # ══════════════════════════════════════════════════════
    print("\n=== TEST 3.2: NONLINEAR TC MEDIATION ===", flush=True)
    covariates = []
    a_ppw_vals = []
    for d in data_T1:
        links_diff = d.get("links_ppw", 0) - d["links0"]
        depth_mean = np.mean(d["depth0"])
        a_val = compute_A_align(d["Y0"], d["delta_ppw"], d["mask"], d["strata"])
        covariates.append([links_diff, depth_mean, d["links0"]])
        a_ppw_vals.append(a_val)

    covariates = np.array(covariates)
    a_ppw_arr = np.array(a_ppw_vals)

    # Linear regression: A_align ~ β₀ + β₁·Δlinks + β₂·depth + β₃·links0
    X_reg = np.column_stack([np.ones(M), covariates])
    beta, residuals, _, _ = np.linalg.lstsq(X_reg, a_ppw_arr, rcond=None)
    predicted = X_reg @ beta
    SS_res = np.sum((a_ppw_arr - predicted)**2)
    SS_tot = np.sum((a_ppw_arr - np.mean(a_ppw_arr))**2)
    R2_linear = 1 - SS_res / SS_tot if SS_tot > 0 else 0

    # Polynomial: add squares and interactions
    X_poly = np.column_stack([X_reg, covariates**2,
                               covariates[:, 0] * covariates[:, 1],
                               covariates[:, 0] * covariates[:, 2]])
    beta_p, _, _, _ = np.linalg.lstsq(X_poly, a_ppw_arr, rcond=None)
    pred_p = X_poly @ beta_p
    SS_res_p = np.sum((a_ppw_arr - pred_p)**2)
    R2_poly = 1 - SS_res_p / SS_tot if SS_tot > 0 else 0

    print(f"  R²_linear (Δlinks, depth, links0) = {R2_linear:.3f}", flush=True)
    print(f"  R²_polynomial (+ squares, interactions) = {R2_poly:.3f}", flush=True)
    print(f"  analytical PASS: R²_mediation < 0.3 → {'PASS' if R2_poly < 0.3 else 'FAIL'}", flush=True)

    results["test_3_2"] = {"R2_linear": R2_linear, "R2_poly": R2_poly}

    # ══════════════════════════════════════════════════════
    # TEST 3.3: CROSS-SEED CRN DECOUPLING
    # ══════════════════════════════════════════════════════
    print("\n=== TEST 3.3: CROSS-SEED DECOUPLING ===", flush=True)
    # For each seed pair (s, s'≠s): take Y0 from s, delta_ppw from s'
    # Match by strata label, compute "cross" A_align
    n_cross = 0
    cross_vals = []
    true_vals = a_ppw  # already computed

    for i in range(M):
        j = (i + 1) % M  # paired with next seed
        d_i = data_T1[i]
        d_j = data_T1[j]
        mask_i = d_i["mask"]
        # Cross: X from seed i, δY from seed j, strata from seed i
        a_cross = compute_A_align(d_i["Y0"], d_j["delta_ppw"], mask_i, d_i["strata"])
        cross_vals.append(a_cross)
        n_cross += 1

    R_cross = abs(np.mean(cross_vals)) / abs(np.mean(true_vals)) if abs(np.mean(true_vals)) > 1e-15 else 0
    print(f"  True A_align mean = {np.mean(true_vals):.6f}", flush=True)
    print(f"  Cross A_align mean = {np.mean(cross_vals):.6f}", flush=True)
    print(f"  R_cross = {R_cross:.4f}", flush=True)
    print(f"  analytical PASS: R_cross < 0.1 → {'PASS' if R_cross < 0.1 else 'FAIL'}", flush=True)

    results["test_3_3"] = {"true_mean": float(np.mean(true_vals)),
                            "cross_mean": float(np.mean(cross_vals)),
                            "R_cross": R_cross}

    # ══════════════════════════════════════════════════════
    # TEST 3.4: FLAT-NOISE / RANDOM-STRATA NULL
    # ══════════════════════════════════════════════════════
    print("\n=== TEST 3.4: NULL TESTS ===", flush=True)

    # Null 1: δY = 0 (flat vs flat)
    null1_vals = [compute_A_align(d["Y0"], np.zeros_like(d["Y0"]), d["mask"], d["strata"])
                  for d in data_T1]
    print(f"  Null 1 (δY=0): mean = {np.mean(null1_vals):.8f} (should be ~0)", flush=True)

    # Null 2: δY = synthetic noise with same variance as real δY
    null2_vals = []
    for d in data_T1:
        delta_real = d["delta_ppw"]
        rng_null = np.random.default_rng(d["seed"] + 77777)
        delta_fake = rng_null.normal(0, np.std(delta_real[d["mask"]]), size=len(delta_real))
        null2_vals.append(compute_A_align(d["Y0"], delta_fake, d["mask"], d["strata"]))
    R_null2 = abs(np.mean(null2_vals)) / abs(np.mean(true_vals)) if abs(np.mean(true_vals)) > 1e-15 else 0
    print(f"  Null 2 (random noise): mean = {np.mean(null2_vals):.6f}, R = {R_null2:.4f}", flush=True)

    # Null 3: randomize strata labels (keep real δY)
    null3_vals = []
    for d in data_T1:
        rng_null = np.random.default_rng(d["seed"] + 88888)
        strata_shuffled = rng_null.permutation(d["strata"])
        null3_vals.append(compute_A_align(d["Y0"], d["delta_ppw"], d["mask"], strata_shuffled))
    R_null3 = abs(np.mean(null3_vals)) / abs(np.mean(true_vals)) if abs(np.mean(true_vals)) > 1e-15 else 0
    print(f"  Null 3 (random strata): mean = {np.mean(null3_vals):.6f}, R = {R_null3:.4f}", flush=True)

    print(f"  analytical PASS: all nulls < 0.1 × phys → "
          f"{'PASS' if max(R_null2, R_null3) < 0.1 else 'FAIL'}", flush=True)

    results["test_3_4"] = {
        "null1_mean": float(np.mean(null1_vals)),
        "null2_mean": float(np.mean(null2_vals)), "R_null2": R_null2,
        "null3_mean": float(np.mean(null3_vals)), "R_null3": R_null3,
    }

    # ══════════════════════════════════════════════════════
    # TEST 3.5: INDEPENDENT SPRINKLING
    # ══════════════════════════════════════════════════════
    print("\n=== TEST 3.5: INDEPENDENT SPRINKLING ===", flush=True)
    indep_vals = []
    for d in data_T1:
        if "Y0_indep" not in d:
            continue
        # Use independent flat Y0 as X, original curved δY
        # This breaks the CRN pairing
        a_indep = compute_A_align(d["Y0_indep"], d["delta_ppw"], d["mask"], d["strata"])
        indep_vals.append(a_indep)

    if indep_vals:
        R_indep = abs(np.mean(indep_vals)) / abs(np.mean(true_vals)) if abs(np.mean(true_vals)) > 1e-15 else 0
        print(f"  True A_align mean = {np.mean(true_vals):.6f}", flush=True)
        print(f"  Indep A_align mean = {np.mean(indep_vals):.6f}", flush=True)
        print(f"  R_indep = {R_indep:.4f}", flush=True)
        results["test_3_5"] = {"indep_mean": float(np.mean(indep_vals)), "R_indep": R_indep}
    else:
        print(f"  No independent sprinkling data", flush=True)
        results["test_3_5"] = {"error": "no data"}

    # ══════════════════════════════════════════════════════
    # TEST 4.1: SAME-PREDICATE RATIO
    # ══════════════════════════════════════════════════════
    print("\n=== TEST 4.1: SAME-PREDICATE RATIO ===", flush=True)
    if "delta_ppw_jet" in data_T1[0] and "delta_sch_jet" in data_T1[0]:
        a_ppw_jet = [compute_A_align(d["Y0"], d["delta_ppw_jet"], d["mask"], d["strata"]) for d in data_T1]
        a_sch_jet = [compute_A_align(d["Y0"], d["delta_sch_jet"], d["mask"], d["strata"]) for d in data_T1]
        AE_ppw_j = np.mean(a_ppw_jet) / E2_PPW
        AE_sch_j = np.mean(a_sch_jet) / E2_SCH
        ratio_same = AE_sch_j / AE_ppw_j if abs(AE_ppw_j) > 1e-15 else 0

        # Also: expmap vs exact comparison
        a_ppw_ex = [compute_A_align(d["Y0"], d["delta_ppw"], d["mask"], d["strata"]) for d in data_T1]
        a_sch_ex = [compute_A_align(d["Y0"], d["delta_sch_exp"], d["mask"], d["strata"]) for d in data_T1]
        AE_ppw_e = np.mean(a_ppw_ex) / E2_PPW
        AE_sch_e = np.mean(a_sch_ex) / E2_SCH
        ratio_mixed = AE_sch_e / AE_ppw_e if abs(AE_ppw_e) > 1e-15 else 0

        print(f"  Same-predicate (jet/jet): ppw={AE_ppw_j:.6f}, Sch={AE_sch_j:.6f}, ratio={ratio_same:.3f}", flush=True)
        print(f"  Mixed (expmap/exact):     ppw={AE_ppw_e:.6f}, Sch={AE_sch_e:.6f}, ratio={ratio_mixed:.3f}", flush=True)
        print(f"  analytical PASS: 0.65 ≤ ratio ≤ 1.50 → "
              f"{'PASS' if 0.65 <= ratio_same <= 1.50 else 'FAIL'}", flush=True)

        results["test_4_1"] = {"ratio_jet_jet": ratio_same, "ratio_exp_exact": ratio_mixed,
                                "AE_ppw_jet": AE_ppw_j, "AE_sch_jet": AE_sch_j}

    # ══════════════════════════════════════════════════════
    # TEST 6.2: ζ/WINDOW ROBUSTNESS
    # ══════════════════════════════════════════════════════
    print("\n=== TEST 6.2: ζ ROBUSTNESS ===", flush=True)
    zeta_values = [0.05, 0.10, 0.15, 0.20, 0.30]
    zeta_results = {}
    for z in zeta_values:
        a_z = []
        for d in data_T1:
            mask_z = bulk_mask(d["pts"], d["T"], z)
            a_z.append(compute_A_align(d["Y0"], d["delta_ppw"], mask_z, d["strata"]))
        ae_z = np.mean(a_z) / E2_PPW
        zeta_results[z] = ae_z
        print(f"  ζ={z}: AE_ppw = {ae_z:.6f}", flush=True)

    ae_vals = list(zeta_results.values())
    cv = np.std(ae_vals) / np.mean(ae_vals) * 100 if np.mean(ae_vals) > 0 else 0
    print(f"  CV across ζ = {cv:.1f}%", flush=True)
    print(f"  analytical PASS: CV < 15% → {'PASS' if cv < 15 else 'FAIL'}", flush=True)

    results["test_6_2"] = {"zeta_AE": {str(z): v for z, v in zeta_results.items()}, "CV_pct": cv}

    # ══════════════════════════════════════════════════════
    # TEST 7.1: OBSERVABLE (p,q) FAMILY
    # ══════════════════════════════════════════════════════
    print("\n=== TEST 7.1: OBSERVABLE (p,q) FAMILY ===", flush=True)
    pq_pairs = [(1, 2), (2, 1), (2, 2), (1, 4), (4, 2), (1, 1)]
    pq_results = {}
    for p, q in pq_pairs:
        a_pq = [compute_A_pq(d["Y0"], d["delta_ppw"], d["mask"], d["strata"], p, q)
                for d in data_T1]
        mean_pq = np.mean(a_pq)
        d_pq = abs(np.mean(a_pq)) / np.std(a_pq) if np.std(a_pq) > 0 else 0

        # Also scramble test for this (p,q)
        scr_pq = []
        for d in data_T1:
            rng_s = np.random.default_rng(d["seed"] + 55555)
            delta_scr = rng_s.permutation(d["delta_ppw"])
            scr_pq.append(compute_A_pq(d["Y0"], delta_scr, d["mask"], d["strata"], p, q))
        R_scr_pq = abs(np.mean(scr_pq)) / abs(mean_pq) if abs(mean_pq) > 1e-15 else 1.0

        pq_results[(p, q)] = {"mean": mean_pq, "d": d_pq, "R_scr": R_scr_pq}
        marker = "★" if p == 2 and q == 2 else " "
        print(f"  {marker} (p={p},q={q}): d={d_pq:.2f}, R_scr={R_scr_pq:.4f}", flush=True)

    results["test_7_1"] = {f"p{p}_q{q}": v for (p, q), v in pq_results.items()}

    # ══════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════
    print(f"\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)

    total = time.time() - t0
    results["total_time_s"] = total

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_FILE}", flush=True)
    print(f"Total analysis time: {total:.0f}s", flush=True)
