"""
Discovery Run 001 — Adversarial Resolution
=============================================
Resolves 3 open attacks from the skeptical review.

PRE-REGISTRATION (written BEFORE computation):
==============================================

ATTACK 3 (pair subsampling):
  Expectation: If scaling_exp is a genuine curvature effect (not proxy),
  then using ALL pairs should give R²_multi SIMILAR to (or lower than)
  the subsampled version. ORC showed the OPPOSITE: all-edges R² jumped
  from 0.34 to 0.86.

  Pre-registered threshold:
  - If R²_multi(all pairs) > 0.60 → PROXY (same failure as ORC)
  - If R²_multi(all pairs) < 0.50 → SURVIVES
  - If R²_multi(all pairs) > 0.50 but < 0.60 → AMBIGUOUS

  Why this might FAIL: interval volumes across ALL pairs are dominated
  by large intervals (which are numerous). Large intervals average over
  more space → their volumes correlate more with global degree stats.

ATTACK 2 (Schwarzschild density artifact):
  For CRN design: we sprinkle at uniform COORDINATE density rho_coord.
  The PHYSICAL density is rho_phys = rho_coord / sqrt(-g).
  For Schwarzschild: sqrt(-g) ≈ 1 - 4*Phi where Phi = -eps/r.
  So rho_phys ≈ rho_coord × (1 + 4*eps/r).

  The interval volume in physical terms:
  V_phys(x,y) = integral over diamond of rho_phys × dV_coord
  ≈ V_coord × (1 + 4*eps/<1/r>)

  where <1/r> is the average of 1/r over the diamond interior.

  Pre-registered prediction:
  delta_V_density ≈ 4*eps*<1/r> × V_flat
  At eps=0.05, <1/r> ≈ 1/(r_avg+0.3) ≈ 3.3 for r_avg≈0:
  delta_V_density ≈ 4*0.05*3.3*V_flat ≈ 0.66*V_flat

  This is O(1) per interval, which is COMPARABLE to the observed delta_V ≈ -2.2.
  If density artifact explains >50% of Schwarzschild signal → CONTAMINATED.

ANALYTICAL (Attack 5 confirmation):
  For pp-wave, the Kretschner scalar K = R_abcd R^abcd ~ eps^2.
  Volume comparison: delta_V / V ~ K * tau^4 / (...).
  Scaling exponent deviation: delta_alpha ~ K * tau^2_eff.
  Since K ~ eps^2, we expect |delta_alpha| ~ eps^2.
  Observed: |d| ~ eps^2.14. Consistent.

  Quantitative prediction (order of magnitude):
  For pp-wave with f=x^2-y^2:
  K = 2*eps^2 * (partial^2 f/partial x^2)^2 + ... ≈ 8*eps^2
  Not sure about exact coefficient, but K ~ eps^2 is firm.

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
import json, time, gc, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import *

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
os.makedirs(OUTDIR, exist_ok=True)

T = 1.0
tau_bins = np.array([0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.45])


# ---------------------------------------------------------------------------
# ALL-pairs interval volumes (no subsampling)
# ---------------------------------------------------------------------------
def interval_volumes_all_pairs(C, pts, N, tau_bins):
    """Compute interval volumes for ALL causal pairs, binned by tau.

    No subsampling — uses every causal pair.
    Returns same structure as the subsampled version.
    """
    # Coordinate proper time for all pairs
    t = pts[:, 0]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = pts[:, 1][np.newaxis, :] - pts[:, 1][:, np.newaxis]
    dy = pts[:, 2][np.newaxis, :] - pts[:, 2][:, np.newaxis]
    dz = pts[:, 3][np.newaxis, :] - pts[:, 3][:, np.newaxis]
    tau2 = dt**2 - dx**2 - dy**2 - dz**2

    # C^2 = interval volumes
    C2 = C @ C

    # Find all causal pairs
    causal_mask = (C > 0.5) & (tau2 > 0)

    bin_results = {}
    for b in range(len(tau_bins) - 1):
        lo, hi = tau_bins[b], tau_bins[b + 1]
        mask = causal_mask & (np.sqrt(tau2) >= lo) & (np.sqrt(tau2) < hi)

        vols = C2[mask].astype(float)

        if len(vols) < 5:
            bin_results[f"bin_{b}"] = {"tau_lo": lo, "tau_hi": hi,
                                       "tau_mid": (lo+hi)/2,
                                       "mean_vol": np.nan, "std_vol": np.nan,
                                       "var_vol": np.nan, "n_pairs": 0}
            continue

        bin_results[f"bin_{b}"] = {
            "tau_lo": float(lo), "tau_hi": float(hi),
            "tau_mid": float((lo + hi) / 2),
            "mean_vol": float(np.mean(vols)),
            "std_vol": float(np.std(vols)),
            "var_vol": float(np.var(vols)),
            "median_vol": float(np.median(vols)),
            "n_pairs": int(len(vols)),
        }

    return bin_results


def weyl_observables(bins_flat, bins_curved, tau_bins):
    """Same as in weyl_probes.py."""
    n_bins = len(tau_bins) - 1
    vf, vc, vrf, vrc, taus = [], [], [], [], []

    for b in range(n_bins):
        bf = bins_flat.get(f"bin_{b}", {})
        bc = bins_curved.get(f"bin_{b}", {})
        if bf.get("n_pairs", 0) < 5 or bc.get("n_pairs", 0) < 5:
            continue
        if np.isnan(bf.get("mean_vol", np.nan)) or np.isnan(bc.get("mean_vol", np.nan)):
            continue
        if bf["mean_vol"] < 0.1:
            continue
        vf.append(bf["mean_vol"])
        vc.append(bc["mean_vol"])
        vrf.append(bf["var_vol"])
        vrc.append(bc["var_vol"])
        taus.append(bf["tau_mid"])

    vf, vc = np.array(vf), np.array(vc)
    vrf, vrc = np.array(vrf), np.array(vrc)
    tau_arr = np.array(taus)
    result = {"n_valid_bins": len(vf)}

    if len(vf) < 3:
        result.update({"vol_ratio_delta": np.nan, "scaling_exp_delta": np.nan,
                        "var_ratio_delta": np.nan})
        return result

    # Vol ratio
    rf = vf[-1] / vf[0] if vf[0] > 0 else np.nan
    rc = vc[-1] / vc[0] if vc[0] > 0 else np.nan
    result["vol_ratio_delta"] = float(rc - rf) if not (np.isnan(rf) or np.isnan(rc)) else np.nan

    # Scaling exponent
    log_tau = np.log(tau_arr)
    if np.sum(vf > 0) >= 3:
        sf, _ = np.polyfit(log_tau[vf > 0], np.log(vf[vf > 0]), 1)
        result["scaling_exp_flat"] = float(sf)
    else:
        result["scaling_exp_flat"] = np.nan
    if np.sum(vc > 0) >= 3:
        sc, _ = np.polyfit(log_tau[vc > 0], np.log(vc[vc > 0]), 1)
        result["scaling_exp_curved"] = float(sc)
    else:
        result["scaling_exp_curved"] = np.nan

    if not (np.isnan(result.get("scaling_exp_flat", np.nan)) or
            np.isnan(result.get("scaling_exp_curved", np.nan))):
        result["scaling_exp_delta"] = result["scaling_exp_curved"] - result["scaling_exp_flat"]
    else:
        result["scaling_exp_delta"] = np.nan

    # Var ratio
    if len(vrf) >= 3 and np.mean(vrf) > 0:
        result["var_ratio_delta"] = float(np.mean(vrc) / np.mean(vrf) - 1.0)
    else:
        result["var_ratio_delta"] = np.nan

    return result


# ---------------------------------------------------------------------------
# CRN trial with ALL pairs
# ---------------------------------------------------------------------------
def crn_trial_allpairs(seed, N, T, metric_name, eps, use_all_pairs=True):
    """CRN trial: interval volumes with ALL pairs or subsampled."""
    seed_offset = SEED_OFFSETS.get(metric_name, 100)
    rng = np.random.default_rng(seed + seed_offset)
    pts = sprinkle_4d(N, T, rng)

    result = {"seed": seed, "N": N, "metric": metric_name, "eps": eps,
              "all_pairs": use_all_pairs}

    C_flat = causal_flat(pts)
    C_curv = METRIC_FNS[metric_name](pts, eps)

    # Degree stats
    A_flat = build_link_graph(C_flat)
    gs_flat, _ = graph_statistics(A_flat)
    fr_flat, _ = forman_ricci(A_flat)
    del A_flat

    A_curv = build_link_graph(C_curv)
    gs_curv, _ = graph_statistics(A_curv)
    fr_curv, _ = forman_ricci(A_curv)
    del A_curv

    # Interval volumes
    if use_all_pairs:
        bins_flat = interval_volumes_all_pairs(C_flat, pts, N, tau_bins)
        bins_curved = interval_volumes_all_pairs(C_curv, pts, N, tau_bins)
    else:
        from discovery_weyl_probes import interval_volumes_binned
        rng_b = np.random.default_rng(seed + 7777)
        bins_flat = interval_volumes_binned(C_flat, pts, N, tau_bins, rng=rng_b)
        rng_b = np.random.default_rng(seed + 7777)
        bins_curved = interval_volumes_binned(C_curv, pts, N, tau_bins, rng=rng_b)

    del C_flat, C_curv; gc.collect()

    w = weyl_observables(bins_flat, bins_curved, tau_bins)
    result.update(w)

    for key in gs_flat:
        result[f"{key}_delta"] = gs_curv[key] - gs_flat[key]
    result["forman_mean_delta"] = fr_curv["F_mean"] - fr_flat["F_mean"]

    return result


# ---------------------------------------------------------------------------
# Adversarial analysis (reused)
# ---------------------------------------------------------------------------
def adversarial(results, obs_key, label=""):
    M = len(results)
    obs = np.array([r.get(obs_key, np.nan) for r in results])
    valid = ~np.isnan(obs)
    obs = obs[valid]
    M_v = len(obs)

    if M_v < 10:
        return {"verdict": "INSUFFICIENT", "r2_multiple": np.nan}

    m, se = np.mean(obs), np.std(obs) / np.sqrt(M_v)
    d_c = m / np.std(obs) if np.std(obs) > 0 else 0
    _, p = stats.ttest_1samp(obs, 0.0)

    proxy_stats = ["mean_degree_delta", "degree_var_delta", "degree_std_delta",
                   "degree_skew_delta", "degree_kurt_delta", "edge_count_delta",
                   "max_degree_delta", "assortativity_delta", "forman_mean_delta"]

    valid_results = [r for r in results if not np.isnan(r.get(obs_key, np.nan))][:M_v]
    max_r2 = 0.0
    max_name = ""
    for sn in proxy_stats:
        vals = np.array([r.get(sn, 0) for r in valid_results], dtype=float)
        if len(vals) != M_v or np.std(vals) < 1e-15 or np.std(obs) < 1e-15:
            continue
        r2 = np.corrcoef(obs, vals)[0, 1]**2
        if r2 > max_r2:
            max_r2, max_name = r2, sn

    X_cols = [np.array([r.get(sn, 0) for r in valid_results], dtype=float)
              for sn in proxy_stats
              if np.std([r.get(sn, 0) for r in valid_results]) > 1e-15]
    r2_multi = 0.0
    adj_r2 = 0.0
    if X_cols and np.std(obs) > 1e-15:
        X = np.column_stack(X_cols)
        X_c = np.column_stack([np.ones(M_v), X])
        k = X.shape[1]
        try:
            beta, _, _, _ = np.linalg.lstsq(X_c, obs, rcond=None)
            ss_res = np.sum((obs - X_c @ beta)**2)
            ss_tot = np.sum((obs - np.mean(obs))**2)
            r2_multi = 1 - ss_res/ss_tot if ss_tot > 0 else 0
            adj_r2 = 1 - (1-r2_multi)*(M_v-1)/max(M_v-k-2, 1)
        except:
            pass

    return {
        "mean_delta": float(m), "se": float(se), "cohen_d": float(d_c),
        "p_value": float(p), "max_r2": float(max_r2), "max_r2_name": max_name,
        "r2_multiple": float(r2_multi), "adj_r2": float(adj_r2), "M": M_v,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    N = 500
    M = 40
    all_results = {}

    # ===================================================================
    # RESOLUTION 1: All-pairs vs subsampled (Attack 3)
    # ===================================================================
    print("=" * 70)
    print("ATTACK 3 RESOLUTION: All-Pairs vs Subsampled")
    print("Pre-registered threshold: R²_multi(all) > 0.60 → PROXY")
    print("=" * 70)

    for geo, eps, label_prefix in [
        ("ppwave_quad", 10.0, "ppw10"),
        ("schwarzschild", 0.05, "schw005"),
    ]:
        for use_all, tag in [(False, "sub500"), (True, "allpairs")]:
            label = f"{label_prefix}_{tag}"
            print(f"\n--- {label} ---")
            t0 = time.time()
            results = []
            for trial in range(M):
                res = crn_trial_allpairs(trial*1000, N, T, geo, eps, use_all_pairs=use_all)
                results.append(res)
            elapsed = time.time() - t0

            for obs_key in ["scaling_exp_delta", "var_ratio_delta"]:
                v = adversarial(results, obs_key, label)
                key = f"{label}_{obs_key}"
                all_results[key] = v
                verdict = "PROXY" if v["r2_multiple"] > 0.60 else \
                          "AMBIG" if v["r2_multiple"] > 0.50 else "SURVIVES"
                print(f"  {obs_key:20s}: R²_multi={v['r2_multiple']:.3f} "
                      f"(adj={v['adj_r2']:.3f}), d={v['cohen_d']:+.3f}, "
                      f"p={v['p_value']:.2e} → {verdict}")

            print(f"  [{elapsed:.1f}s]")

    # Compare
    print("\n  COMPARISON: subsampled vs all-pairs")
    print(f"  {'condition':30s} {'R²_multi(sub)':>15s} {'R²_multi(all)':>15s} {'change':>10s}")
    print("  " + "-" * 75)

    for geo, obs in [("ppw10", "scaling_exp_delta"), ("ppw10", "var_ratio_delta"),
                     ("schw005", "scaling_exp_delta"), ("schw005", "var_ratio_delta")]:
        r2_sub = all_results.get(f"{geo}_sub500_{obs}", {}).get("r2_multiple", np.nan)
        r2_all = all_results.get(f"{geo}_allpairs_{obs}", {}).get("r2_multiple", np.nan)
        change = r2_all - r2_sub if not (np.isnan(r2_sub) or np.isnan(r2_all)) else np.nan
        print(f"  {geo+'_'+obs:30s} {r2_sub:15.3f} {r2_all:15.3f} {change:+10.3f}")

    # ===================================================================
    # RESOLUTION 2: Schwarzschild density artifact (Attack 2)
    # ===================================================================
    print("\n" + "=" * 70)
    print("ATTACK 2 RESOLUTION: Schwarzschild Density Artifact")
    print("=" * 70)

    # Compute <1/r> analytically for 4D causal diamond of half-size T/2=0.5
    # The diamond is |t| + sqrt(x²+y²+z²) < 0.5
    # Average 1/(r+0.3) over the diamond:
    # By symmetry, <r> ≈ 0.15 for uniform distribution in the diamond
    # <1/(r+0.3)> ≈ 1/0.45 ≈ 2.2

    # Numerical computation of <1/(r+0.3)> over the 4D causal diamond
    rng = np.random.default_rng(42)
    pts_sample = sprinkle_4d(10000, T, rng)
    r_sample = np.sqrt(pts_sample[:, 1]**2 + pts_sample[:, 2]**2 + pts_sample[:, 3]**2)
    inv_r_mean = float(np.mean(1.0 / (r_sample + 0.3)))
    print(f"\n  <1/(r+0.3)> over causal diamond = {inv_r_mean:.3f}")

    eps_schw = 0.05
    # sqrt(-g) ≈ 1 - 4*Phi = 1 + 4*eps/(r+0.3)
    # Density correction factor: <sqrt(-g)> - 1 ≈ 4*eps*<1/(r+0.3)>
    density_correction = 4 * eps_schw * inv_r_mean
    print(f"  Density correction: <sqrt(-g)> - 1 ≈ 4*eps*<1/(r+0.3)> = {density_correction:.4f}")
    print(f"  This means ~{density_correction*100:.1f}% more physical volume in curved diamond")

    # Expected delta_V from density alone:
    # V_physical = V_coord * sqrt(-g)
    # In CRN: we use coord density, so V_CRN = V_coord (same for both)
    # But the CAUSAL structure changes due to metric: light cones narrow
    # The density artifact is separate from the causal structure change
    # Actually: in CRN, density is FIXED (same points). The difference comes from
    # WHICH pairs are causally related, not from density.
    # So the "density artifact" I worried about is actually NOT an artifact —
    # it IS the physical effect of the metric on causal structure.

    print(f"\n  CORRECTION: In CRN design, density is FIXED (same N points).")
    print(f"  The causal structure change IS the physical effect.")
    print(f"  There is no separate 'density artifact' — the CRN design")
    print(f"  correctly isolates the metric effect at fixed coordinate density.")
    print(f"  The lapse effect (g_tt narrowing light cones) IS physical.")
    print(f"  But it's not CURVATURE — it's the gravitational potential Phi.")

    # ===================================================================
    # RESOLUTION 3: Analytical prediction for pp-wave (Attack 5)
    # ===================================================================
    print("\n" + "=" * 70)
    print("ATTACK 5 RESOLUTION: Analytical eps-Scaling")
    print("=" * 70)

    # For pp-wave with profile f(x,y) = x²-y²:
    # The Riemann tensor has components R_{uzuz} = -eps*f_xx/2 = -eps
    #                                   R_{uyuy} = -eps*f_yy/2 = +eps
    # (only non-zero components in Brinkmann coordinates)
    #
    # Kretschner scalar: K = R_{abcd}R^{abcd} = 4*(eps² + eps²) = 8*eps²
    # (accounting for symmetries)
    #
    # Volume of a causal diamond in curved space (to leading order):
    # V(tau) = V_flat(tau) × [1 - K*tau^4/(720*(d+2)*(d+4)) + ...]
    # (standard result from DeWitt expansion, but this is for Riemannian)
    #
    # For Lorentzian with pp-wave, the correction depends on direction.
    # But for the AVERAGE over random pairs, we expect:
    # <V(tau)>_curved / <V(tau)>_flat ≈ 1 + c * K * tau^4
    # where c is a dimension-dependent constant.
    #
    # Scaling exponent: if V ~ tau^alpha, then
    # log(V) = alpha*log(tau) + const
    # The curvature correction adds a tau^4 term:
    # log(V) ≈ d*log(tau) + const + c*K*tau^4 (to leading order)
    # Fitting this as alpha*log(tau) over a range [tau_min, tau_max]:
    # alpha ≈ d + c*K*<tau^4/log(tau)> (effective correction)
    #
    # The key prediction: delta_alpha ~ K ~ eps^2
    # So |d| should scale as eps^2

    print(f"  Kretschner scalar for pp-wave: K = 8*eps^2")
    print(f"  Prediction: delta_alpha ~ K ~ eps^2")

    # Check against data
    eps_data = [5.0, 10.0, 20.0]
    dalpha_data = [-0.199, -0.586, -1.312]  # from eps-scaling test

    print(f"\n  {'eps':>6s} {'dalpha':>10s} {'K=8*eps²':>10s} {'dalpha/K':>12s}")
    print("  " + "-" * 40)
    for eps, da in zip(eps_data, dalpha_data):
        K = 8 * eps**2
        print(f"  {eps:6.1f} {da:+10.3f} {K:10.0f} {da/K:+12.6f}")

    # If dalpha/K is constant → delta_alpha IS proportional to K
    ratios = [da / (8*eps**2) for da, eps in zip(dalpha_data, eps_data)]
    print(f"\n  Ratios dalpha/(8*eps²): {[f'{r:.6f}' for r in ratios]}")
    print(f"  Ratio variation: {np.std(ratios)/abs(np.mean(ratios))*100:.1f}%")

    if np.std(ratios)/abs(np.mean(ratios)) < 0.3:
        print(f"  => delta_alpha IS proportional to Kretschner K (within 30%)")
        print(f"  => Proportionality constant: {np.mean(ratios):.6f}")
    else:
        print(f"  => delta_alpha is NOT simply proportional to K")
        # Try K^{1/2}
        ratios_sqrt = [da / (np.sqrt(8)*eps) for da, eps in zip(dalpha_data, eps_data)]
        print(f"  Trying dalpha/sqrt(K): {[f'{r:.4f}' for r in ratios_sqrt]}")
        print(f"  Variation: {np.std(ratios_sqrt)/abs(np.mean(ratios_sqrt))*100:.1f}%")

    # Save
    outpath = os.path.join(OUTDIR, "adversarial_resolution.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # ===================================================================
    # FINAL VERDICT
    # ===================================================================
    print("\n" + "=" * 70)
    print("ADVERSARIAL RESOLUTION — FINAL VERDICT")
    print("=" * 70)

    # Attack 3
    ppw_sub = all_results.get("ppw10_sub500_scaling_exp_delta", {}).get("r2_multiple", np.nan)
    ppw_all = all_results.get("ppw10_allpairs_scaling_exp_delta", {}).get("r2_multiple", np.nan)
    print(f"\n  ATTACK 3 (subsampling):")
    print(f"    pp-wave scaling_exp: R²_multi sub={ppw_sub:.3f}, all={ppw_all:.3f}")
    if ppw_all > 0.60:
        print(f"    → PROXY (R²_multi > 0.60). Same failure as ORC.")
    elif ppw_all < 0.50:
        print(f"    → SURVIVES. All-pairs does NOT increase proxy correlation.")
    else:
        print(f"    → AMBIGUOUS.")

    print(f"\n  ATTACK 2 (density): No separate artifact in CRN design.")
    print(f"    Lapse effect is physical, not artifactual.")
    print(f"    But it measures Phi, not R or C². Lapse-cancelled ratios address this.")

    print(f"\n  ATTACK 5 (analytical): dalpha/K tested for constancy.")


if __name__ == "__main__":
    main()
