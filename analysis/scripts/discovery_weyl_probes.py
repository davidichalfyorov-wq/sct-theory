"""
Discovery Run 001 — Weyl Probes: Lapse-Cancelling Curvature Observables
=========================================================================

PROBLEM: vol_mean measures lapse (g_tt), not curvature. 130x overcounting.
SOLUTION: Take RATIOS of volumes at different scales. Lapse cancels.

TWO CANDIDATES:

W-C: Volume Ratio R(tau_1, tau_2) = <V(tau_2)> / <V(tau_1)>
  In flat space: R = (tau_2/tau_1)^d = const for fixed tau ratio.
  Lapse multiplies all V by same factor → cancels in ratio.
  Curvature modifies the tau-dependence → changes the ratio.
  CRN delta: Delta_R = R_curved - R_flat.

W-D: Scaling Exponent alpha from V(tau) ~ tau^alpha
  In flat d=4: alpha = 4. Lapse shifts intercept, not slope.
  Curvature modifies alpha. CRN delta: Delta_alpha.

BINNING: Use coordinate proper time tau_coord = sqrt(dt^2 - dr^2)
between pairs. This is the SAME for flat and curved (CRN: same coords).
Bin pairs by tau_coord, compute V in each bin for flat and curved.

Also test: PAIR DISPARITY (variance of V at fixed tau).
In flat space: V depends only on tau (isotropic).
Weyl curvature makes V position-dependent → higher variance.
CRN delta: Delta_Var = Var_curved(V|tau) - Var_flat(V|tau).

PRE-REGISTERED:
  Bonferroni alpha = 0.01/20 = 0.0005
  Same adversarial protocol (degree stats + forman)
  Null control: conformal

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
import json, time, gc, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import (
    sprinkle_4d, causal_flat, causal_ppwave_quad, causal_schwarzschild,
    causal_conformal, build_link_graph, graph_statistics, forman_ricci,
    SEED_OFFSETS, METRIC_FNS
)

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
os.makedirs(OUTDIR, exist_ok=True)

T = 1.0


# ---------------------------------------------------------------------------
# Compute interval volumes binned by coordinate proper time
# ---------------------------------------------------------------------------
def interval_volumes_binned(C, pts, N, tau_bins, max_pairs_per_bin=200, rng=None):
    """Compute mean interval volume in each tau bin.

    Args:
        C: causal matrix (N x N)
        pts: coordinate array (N x 4)
        tau_bins: array of bin edges for coordinate proper time
        max_pairs_per_bin: max pairs to sample per bin
        rng: random generator

    Returns:
        dict with per-bin: mean_vol, std_vol, n_pairs
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Compute coordinate proper time for all causal pairs
    t = pts[:, 0]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = pts[:, 1][np.newaxis, :] - pts[:, 1][:, np.newaxis]
    dy = pts[:, 2][np.newaxis, :] - pts[:, 2][:, np.newaxis]
    dz = pts[:, 3][np.newaxis, :] - pts[:, 3][:, np.newaxis]
    tau2 = dt**2 - dx**2 - dy**2 - dz**2

    # Find causal pairs
    causal_mask = C > 0.5
    rows, cols = np.where(causal_mask & (tau2 > 0))
    tau_vals = np.sqrt(tau2[rows, cols])

    # For each pair, count interval size
    # V(i,j) = |{k : C[i,k]=1 and C[k,j]=1}| = (C @ C)[i,j]
    # But we need this for specific pairs, not all
    # Precompute C^2 once
    C2 = C @ C  # C2[i,j] = number of elements between i and j

    bin_results = {}
    for b in range(len(tau_bins) - 1):
        lo, hi = tau_bins[b], tau_bins[b + 1]
        mask = (tau_vals >= lo) & (tau_vals < hi)
        bin_rows = rows[mask]
        bin_cols = cols[mask]

        if len(bin_rows) == 0:
            bin_results[f"bin_{b}"] = {"tau_lo": lo, "tau_hi": hi,
                                       "mean_vol": np.nan, "std_vol": np.nan,
                                       "var_vol": np.nan, "n_pairs": 0}
            continue

        # Sample if too many
        if len(bin_rows) > max_pairs_per_bin:
            idx = rng.choice(len(bin_rows), size=max_pairs_per_bin, replace=False)
            bin_rows = bin_rows[idx]
            bin_cols = bin_cols[idx]

        # Interval volumes
        vols = C2[bin_rows, bin_cols].astype(float)

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


# ---------------------------------------------------------------------------
# Weyl observables from binned volumes
# ---------------------------------------------------------------------------
def weyl_observables(bins_flat, bins_curved, tau_bins):
    """Compute lapse-cancelling observables from binned interval volumes."""
    n_bins = len(tau_bins) - 1

    # Collect valid bins
    vols_flat = []
    vols_curved = []
    vars_flat = []
    vars_curved = []
    taus = []

    for b in range(n_bins):
        bf = bins_flat.get(f"bin_{b}", {})
        bc = bins_curved.get(f"bin_{b}", {})
        if bf.get("n_pairs", 0) < 10 or bc.get("n_pairs", 0) < 10:
            continue
        if np.isnan(bf["mean_vol"]) or np.isnan(bc["mean_vol"]):
            continue
        if bf["mean_vol"] < 0.1:  # too small intervals
            continue

        vols_flat.append(bf["mean_vol"])
        vols_curved.append(bc["mean_vol"])
        vars_flat.append(bf["var_vol"])
        vars_curved.append(bc["var_vol"])
        taus.append(bf["tau_mid"])

    vf = np.array(vols_flat)
    vc = np.array(vols_curved)
    vrf = np.array(vars_flat)
    vrc = np.array(vars_curved)
    tau_arr = np.array(taus)

    result = {"n_valid_bins": len(vf)}

    if len(vf) < 3:
        result.update({
            "vol_ratio_delta": np.nan, "scaling_exp_flat": np.nan,
            "scaling_exp_curved": np.nan, "scaling_exp_delta": np.nan,
            "var_ratio_delta": np.nan,
        })
        return result

    # W-C: Volume ratio (large tau / small tau)
    # Use first and last valid bins
    ratio_flat = vf[-1] / vf[0] if vf[0] > 0 else np.nan
    ratio_curved = vc[-1] / vc[0] if vc[0] > 0 else np.nan
    if not (np.isnan(ratio_flat) or np.isnan(ratio_curved)):
        result["vol_ratio_flat"] = float(ratio_flat)
        result["vol_ratio_curved"] = float(ratio_curved)
        result["vol_ratio_delta"] = float(ratio_curved - ratio_flat)
    else:
        result["vol_ratio_delta"] = np.nan

    # W-D: Scaling exponent alpha from log(V) ~ alpha*log(tau)
    log_tau = np.log(tau_arr)
    mask_f = vf > 0
    mask_c = vc > 0

    if np.sum(mask_f) >= 3:
        slope_f, intercept_f = np.polyfit(log_tau[mask_f], np.log(vf[mask_f]), 1)
        result["scaling_exp_flat"] = float(slope_f)
    else:
        result["scaling_exp_flat"] = np.nan

    if np.sum(mask_c) >= 3:
        slope_c, intercept_c = np.polyfit(log_tau[mask_c], np.log(vc[mask_c]), 1)
        result["scaling_exp_curved"] = float(slope_c)
    else:
        result["scaling_exp_curved"] = np.nan

    if not (np.isnan(result.get("scaling_exp_flat", np.nan)) or
            np.isnan(result.get("scaling_exp_curved", np.nan))):
        result["scaling_exp_delta"] = result["scaling_exp_curved"] - result["scaling_exp_flat"]
    else:
        result["scaling_exp_delta"] = np.nan

    # Pair disparity: variance ratio
    if len(vrf) >= 3 and np.mean(vrf) > 0:
        result["mean_var_flat"] = float(np.mean(vrf))
        result["mean_var_curved"] = float(np.mean(vrc))
        result["var_ratio_delta"] = float(np.mean(vrc) / np.mean(vrf) - 1.0) \
            if np.mean(vrf) > 0 else np.nan
    else:
        result["var_ratio_delta"] = np.nan

    return result


# ---------------------------------------------------------------------------
# CRN trial
# ---------------------------------------------------------------------------
def crn_trial_weyl(seed, N, T, metric_name, eps, tau_bins):
    """CRN trial for Weyl probes."""
    seed_offset = SEED_OFFSETS.get(metric_name, 100)
    rng = np.random.default_rng(seed + seed_offset)
    pts = sprinkle_4d(N, T, rng)

    result = {"seed": seed, "N": N, "metric": metric_name, "eps": eps}

    # Build causal matrices
    C_flat = causal_flat(pts)
    C_curv = METRIC_FNS[metric_name](pts, eps)

    # Degree stats (for adversarial check)
    A_flat = build_link_graph(C_flat)
    gs_flat, deg_flat = graph_statistics(A_flat)
    fr_flat, _ = forman_ricci(A_flat, deg_flat)
    del A_flat

    A_curv = build_link_graph(C_curv)
    gs_curv, deg_curv = graph_statistics(A_curv)
    fr_curv, _ = forman_ricci(A_curv, deg_curv)
    del A_curv

    # Binned interval volumes (same RNG for pair sampling)
    rng_bins = np.random.default_rng(seed + 7777)
    bins_flat = interval_volumes_binned(C_flat, pts, N, tau_bins, rng=rng_bins)
    rng_bins = np.random.default_rng(seed + 7777)  # reset for same pairs
    bins_curved = interval_volumes_binned(C_curv, pts, N, tau_bins, rng=rng_bins)

    del C_flat, C_curv; gc.collect()

    # Weyl observables
    w = weyl_observables(bins_flat, bins_curved, tau_bins)

    result.update(w)

    # Degree stat deltas for adversarial check
    for key in gs_flat:
        result[f"{key}_delta"] = gs_curv[key] - gs_flat[key]
    result["forman_mean_delta"] = fr_curv["F_mean"] - fr_flat["F_mean"]

    return result


# ---------------------------------------------------------------------------
# Adversarial analysis
# ---------------------------------------------------------------------------
def adversarial_weyl(results, obs_key, label=""):
    """Adversarial proxy check for Weyl observable."""
    M = len(results)
    obs = np.array([r.get(obs_key, np.nan) for r in results])
    valid = ~np.isnan(obs)
    obs = obs[valid]
    M_valid = len(obs)

    print(f"\n  === {label}: {obs_key} (M={M_valid}/{M}) ===")
    if M_valid < 10:
        print(f"  TOO FEW valid trials")
        return {"verdict": "INSUFFICIENT DATA", "mean_delta": np.nan}

    m, se = np.mean(obs), np.std(obs) / np.sqrt(M_valid)
    d_cohen = m / np.std(obs) if np.std(obs) > 0 else 0
    _, p_obs = stats.ttest_1samp(obs, 0.0)
    print(f"  delta = {m:+.4f} +/- {se:.4f}, d={d_cohen:+.3f}, p={p_obs:.2e}")

    # Adversarial
    proxy_stats = ["mean_degree_delta", "degree_var_delta", "degree_std_delta",
                   "degree_skew_delta", "degree_kurt_delta", "edge_count_delta",
                   "max_degree_delta", "assortativity_delta", "forman_mean_delta"]

    valid_results = [r for i, r in enumerate(results) if valid[i] if i < len(valid)]
    if len(valid_results) < M_valid:
        valid_results = [r for r in results if not np.isnan(r.get(obs_key, np.nan))]

    max_r2 = 0.0
    max_r2_name = ""
    for stat_name in proxy_stats:
        vals = np.array([r.get(stat_name, 0) for r in valid_results[:M_valid]], dtype=float)
        if len(vals) != M_valid:
            continue
        if np.std(vals) < 1e-15 or np.std(obs) < 1e-15:
            continue
        corr = np.corrcoef(obs, vals)[0, 1]
        r2 = corr**2
        if r2 > max_r2:
            max_r2 = r2
            max_r2_name = stat_name
        if r2 > 0.15:
            flag = " PROXY!!!" if r2 > 0.80 else (" AMBIG" if r2 > 0.50 else "")
            print(f"    R2({obs_key}, {stat_name}) = {r2:.3f}{flag}")

    # Multiple regression
    X_cols = []
    for stat_name in proxy_stats:
        vals = np.array([r.get(stat_name, 0) for r in valid_results[:M_valid]], dtype=float)
        if len(vals) == M_valid and np.std(vals) > 1e-15:
            X_cols.append(vals)

    r2_multi = 0.0
    if X_cols and np.std(obs) > 1e-15:
        X = np.column_stack(X_cols)
        X_c = np.column_stack([np.ones(M_valid), X])
        try:
            beta, _, _, _ = np.linalg.lstsq(X_c, obs, rcond=None)
            y_pred = X_c @ beta
            ss_res = np.sum((obs - y_pred)**2)
            ss_tot = np.sum((obs - np.mean(obs))**2)
            r2_multi = 1 - ss_res/ss_tot if ss_tot > 0 else 0
        except:
            pass

    print(f"    MAX R2 = {max_r2:.3f} ({max_r2_name})")
    print(f"    Multi R2 = {r2_multi:.3f}")

    ALPHA_BONF = 0.01 / 20
    if p_obs < ALPHA_BONF and max_r2 < 0.50 and r2_multi < 0.50:
        verdict = "DETECTED (genuine)"
    elif max_r2 > 0.80 or r2_multi > 0.80:
        verdict = "PROXY"
    elif p_obs < ALPHA_BONF and max(max_r2, r2_multi) >= 0.50:
        verdict = "AMBIGUOUS"
    elif p_obs < 0.05:
        verdict = "WEAK"
    else:
        verdict = "NULL"

    print(f"  VERDICT: {verdict}")

    return {
        "observable": obs_key, "mean_delta": float(m), "se_delta": float(se),
        "cohen_d": float(d_cohen), "p_value": float(p_obs),
        "max_r2_single": float(max_r2), "r2_multiple": float(r2_multi),
        "verdict": verdict, "M_valid": M_valid,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    N = 500
    M = 40

    # Tau bins (coordinate proper time)
    # At N=500 in 4D, typical tau range is 0 to ~0.5
    tau_bins = np.array([0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.45])

    TESTS = [
        ("conformal",     5.0),    # NULL CONTROL
        ("ppwave_quad",   10.0),   # Weyl curvature, no lapse
        ("ppwave_quad",   20.0),   # stronger Weyl
        ("schwarzschild", 0.05),   # lapse + curvature
    ]

    OBSERVABLES = ["vol_ratio_delta", "scaling_exp_delta", "var_ratio_delta"]

    print("=" * 70)
    print("WEYL PROBES: Lapse-Cancelling Curvature Observables")
    print(f"N={N}, M={M}, {len(tau_bins)-1} tau bins")
    print(f"Bonferroni alpha = {0.01/20:.5f}")
    print("=" * 70)

    all_results = {}

    for geo, eps in TESTS:
        label = f"{geo}_eps{eps}"
        is_null = (geo == "conformal")
        note = " [NULL]" if is_null else ""
        print(f"\n{'='*50}")
        print(f"  {label}{note}")
        print(f"{'='*50}")

        t0 = time.time()
        results = []
        for trial in range(M):
            res = crn_trial_weyl(trial * 1000, N, T, geo, eps, tau_bins)
            results.append(res)
            if (trial + 1) % 10 == 0:
                elapsed = time.time() - t0
                vr = [r.get("vol_ratio_delta", np.nan) for r in results]
                vr_valid = [v for v in vr if not np.isnan(v)]
                se = [r.get("scaling_exp_delta", np.nan) for r in results]
                se_valid = [v for v in se if not np.isnan(v)]
                print(f"  trial {trial+1}/{M}: vol_ratio={np.mean(vr_valid):+.3f} "
                      f"scale_exp={np.mean(se_valid):+.4f} [{elapsed:.1f}s]")

        elapsed = time.time() - t0

        geo_results = {"geometry": geo, "eps": eps, "is_null": is_null,
                       "elapsed_sec": elapsed}

        for obs_key in OBSERVABLES:
            v = adversarial_weyl(results, obs_key, label)
            geo_results[obs_key] = v

        geo_results["trials"] = results
        all_results[label] = geo_results

    # Save
    outpath = os.path.join(OUTDIR, "weyl_probes.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # Summary
    print("\n" + "=" * 70)
    print("WEYL PROBES SUMMARY")
    print("=" * 70)
    print(f"{'geometry':25s} {'observable':20s} {'delta':>10s} {'d':>8s} "
          f"{'p':>12s} {'R2max':>8s} {'R2multi':>8s} {'verdict':>22s}")
    print("-" * 110)

    genuine = 0
    for label, res in all_results.items():
        if res["is_null"]:
            continue
        for obs_key in OBSERVABLES:
            v = res.get(obs_key, {})
            if not isinstance(v, dict) or "verdict" not in v:
                continue
            print(f"{label:25s} {obs_key:20s} {v.get('mean_delta',0):+10.4f} "
                  f"{v.get('cohen_d',0):+8.3f} {v.get('p_value',1):12.2e} "
                  f"{v.get('max_r2_single',0):8.3f} {v.get('r2_multiple',0):8.3f} "
                  f"{v['verdict']:>22s}")
            if v["verdict"] == "DETECTED (genuine)":
                genuine += 1

    print(f"\nGenuine detections: {genuine}")

    # Key question: does pp-wave now work?
    ppw10 = all_results.get("ppwave_quad_eps10.0", {})
    ppw20 = all_results.get("ppwave_quad_eps20.0", {})
    schw = all_results.get("schwarzschild_eps0.05", {})

    print("\n  KEY QUESTION: Does lapse cancellation reveal Weyl curvature?")
    for name, res in [("ppw_10", ppw10), ("ppw_20", ppw20), ("schw", schw)]:
        for obs in OBSERVABLES:
            v = res.get(obs, {})
            if isinstance(v, dict) and "verdict" in v:
                if "GENUINE" in v["verdict"]:
                    print(f"    {name} {obs}: YES! {v['verdict']}")
                elif "WEAK" in v["verdict"] or "NULL" in v["verdict"]:
                    pass  # only print positive


if __name__ == "__main__":
    main()
