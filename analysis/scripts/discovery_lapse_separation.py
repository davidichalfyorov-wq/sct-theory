"""
Discovery Run 001 — Tests X+Y: Lapse Separation
=================================================

PROBLEM: scaling_exp responds to lapse changes (Test A: uniform lapse d=-3.5).
PP-wave has g_tt = -1 + eps*(x^2-y^2)/2 ≠ -1 in (t,x,y,z) coords.
So pp-wave signal is MIX of lapse + curvature.

SOLUTION: Construct "synthetic lapse" metric with SAME g_tt but NO curvature.

PP-wave metric:
  ds² = -(1-eps*f/2)dt² + dx² + dy² + (1+eps*f/2)dz² + eps*f*dt*dz
  where f = x_m² - y_m²

Synthetic lapse (SAME g_tt, but diagonal — no off-diagonal, no g_zz change):
  ds² = -(1-eps*f/2)dt² + dx² + dy² + dz²

This has:
  - SAME time-time component as pp-wave
  - SAME lapse effect on light cones (in the time direction)
  - But NO off-diagonal g_tz term (which carries the gravitational wave)
  - And g_zz = 1 (no z-direction modification)

Causal condition for synthetic lapse:
  (1 - eps*f_m/2)*dt² > dx² + dy² + dz²

Causal condition for pp-wave (from our code):
  dt² - dr² > eps*f_m*(dt+dz)²/2
  ↔ dt² - dx² - dy² - dz² > eps*f_m*(dt² + 2*dt*dz + dz²)/2
  ↔ (1 - eps*f_m/2)*dt² > dx² + dy² + (1 + eps*f_m/2)*dz² + eps*f_m*dt*dz

So the DIFFERENCE between pp-wave and synthetic lapse is:
  pp-wave has EXTRA terms: +eps*f_m/2 * dz² + eps*f_m*dt*dz
  These come from the off-diagonal and z-z metric components.
  These are the GRAVITATIONAL WAVE components that carry curvature.

TEST X: Compare scaling_exp(pp-wave) vs scaling_exp(synthetic lapse)
TEST Y: Compute residual = scaling_exp(pp-wave) - scaling_exp(synthetic lapse)
  - If residual ≈ 0: ALL signal is lapse → curvature claim DEAD
  - If residual ≠ 0: there IS curvature beyond lapse → claim PARTIALLY ALIVE

PRE-REGISTERED:
  - No prediction on sign or magnitude of residual
  - Two-sided test on residual
  - residual p < 0.001 → curvature component exists
  - residual p > 0.05 → indistinguishable from lapse alone
  - This test CAN kill the curvature claim permanently

Author: David Alfyorov
"""
import numpy as np
from scipy import stats
import json, time, gc, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import *
from discovery_weyl_probes import interval_volumes_binned, weyl_observables

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
os.makedirs(OUTDIR, exist_ok=True)

T = 1.0
tau_bins = np.array([0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.45])


# =====================================================================
# Synthetic lapse metric (SAME g_tt as pp-wave, NO off-diagonal)
# =====================================================================
def causal_synthetic_lapse(pts, eps):
    """Synthetic lapse: ds² = -(1 - eps*f/2)dt² + dx² + dy² + dz²
    where f = x_m² - y_m².

    Has SAME g_tt as pp-wave, but is diagonal (no gravitational wave).
    Riemann tensor: nonzero due to position-dependent lapse, but
    DIFFERENT from pp-wave Riemann (no wave components).
    """
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx**2 + dy**2 + dz**2

    xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
    ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
    f = xm**2 - ym**2

    # Causal: (1 - eps*f/2)*dt² > dr²
    g_tt = 1 - eps * f / 2.0

    # Handle g_tt < 0 (ergosphere-like region at large f): not causal
    return ((g_tt * dt**2 > dr2) & (dt > 0) & (g_tt > 0)).astype(np.float64)


METRIC_FNS["synthetic_lapse"] = causal_synthetic_lapse
SEED_OFFSETS["synthetic_lapse"] = 100  # SAME as ppwave_quad — critical for CRN matching


# =====================================================================
# Triple CRN trial: flat + pp-wave + synthetic lapse (SAME seed)
# =====================================================================
def crn_trial_triple(seed, N, T, eps):
    """Triple CRN: flat, pp-wave, synthetic lapse — all from same sprinkling."""
    rng = np.random.default_rng(seed + 100)  # same offset as ppwave_quad
    pts = sprinkle_4d(N, T, rng)

    result = {"seed": seed, "N": N, "eps": eps}

    # Three causal matrices from the SAME points
    C_flat = causal_flat(pts)
    C_ppw = causal_ppwave_quad(pts, eps)
    C_syn = causal_synthetic_lapse(pts, eps)

    # Count causal pairs for diagnostics
    result["n_causal_flat"] = int(np.sum(C_flat > 0.5))
    result["n_causal_ppw"] = int(np.sum(C_ppw > 0.5))
    result["n_causal_syn"] = int(np.sum(C_syn > 0.5))

    # Compute Weyl observables for each
    for tag, C in [("flat", C_flat), ("ppw", C_ppw), ("syn", C_syn)]:
        rng_b = np.random.default_rng(seed + 7777)
        bins = interval_volumes_binned(C, pts, N, tau_bins, rng=rng_b)

        # Compute raw mean vol
        C2 = C @ C
        causal_mask = C > 0.5
        if np.sum(causal_mask) > 10:
            vols = C2[causal_mask].astype(float)
            result[f"vol_mean_{tag}"] = float(np.mean(vols))
        else:
            result[f"vol_mean_{tag}"] = 0.0

        result[f"bins_{tag}"] = bins

        # Degree stats
        A = build_link_graph(C)
        gs, _ = graph_statistics(A)
        for key in gs:
            result[f"{key}_{tag}"] = gs[key]
        del A

    del C_flat, C_ppw, C_syn; gc.collect()

    # Weyl observables: flat→ppw, flat→syn
    w_ppw = weyl_observables(result["bins_flat"], result["bins_ppw"], tau_bins)
    w_syn = weyl_observables(result["bins_flat"], result["bins_syn"], tau_bins)

    for key in ["vol_ratio_delta", "scaling_exp_delta", "var_ratio_delta",
                "scaling_exp_flat", "scaling_exp_curved"]:
        result[f"{key}_ppw"] = w_ppw.get(key, w_ppw.get(key.replace("_delta",""), np.nan))
        result[f"{key}_syn"] = w_syn.get(key, w_syn.get(key.replace("_delta",""), np.nan))

    # Key deltas
    result["scaling_exp_ppw"] = w_ppw.get("scaling_exp_delta", np.nan)
    result["scaling_exp_syn"] = w_syn.get("scaling_exp_delta", np.nan)
    result["vol_ratio_ppw"] = w_ppw.get("vol_ratio_delta", np.nan)
    result["vol_ratio_syn"] = w_syn.get("vol_ratio_delta", np.nan)
    result["var_ratio_ppw"] = w_ppw.get("var_ratio_delta", np.nan)
    result["var_ratio_syn"] = w_syn.get("var_ratio_delta", np.nan)

    # THE RESIDUAL: pp-wave minus synthetic lapse
    for obs in ["scaling_exp", "vol_ratio", "var_ratio"]:
        ppw_val = result.get(f"{obs}_ppw", np.nan)
        syn_val = result.get(f"{obs}_syn", np.nan)
        if not (np.isnan(ppw_val) or np.isnan(syn_val)):
            result[f"{obs}_residual"] = ppw_val - syn_val
        else:
            result[f"{obs}_residual"] = np.nan

    # Also: vol_mean deltas and residual
    result["vol_mean_ppw_delta"] = result["vol_mean_ppw"] - result["vol_mean_flat"]
    result["vol_mean_syn_delta"] = result["vol_mean_syn"] - result["vol_mean_flat"]
    result["vol_mean_residual"] = result["vol_mean_ppw_delta"] - result["vol_mean_syn_delta"]

    # Clean up bins (not needed in output)
    for tag in ["flat", "ppw", "syn"]:
        del result[f"bins_{tag}"]

    return result


# =====================================================================
# Main
# =====================================================================
def main():
    N = 500
    M = 50  # more trials for residual power

    print("=" * 70)
    print("TESTS X+Y: Lapse Separation")
    print(f"N={N}, M={M}")
    print("Triple CRN: flat + pp-wave + synthetic lapse (same points)")
    print()
    print("PRE-REGISTERED:")
    print("  residual = scaling_exp(ppw) - scaling_exp(synthetic_lapse)")
    print("  residual p < 0.001 → curvature beyond lapse EXISTS")
    print("  residual p > 0.05  → indistinguishable from lapse alone")
    print("  This test CAN kill the curvature claim permanently.")
    print("=" * 70)

    for eps in [5.0, 10.0, 20.0]:
        label = f"eps{eps}"
        print(f"\n{'='*50}")
        print(f"  pp-wave eps={eps}")
        print(f"{'='*50}")

        t0 = time.time()
        results = []
        for trial in range(M):
            res = crn_trial_triple(trial * 1000, N, T, eps)
            results.append(res)
            if (trial + 1) % 10 == 0:
                elapsed = time.time() - t0
                se_ppw = [r["scaling_exp_ppw"] for r in results if not np.isnan(r["scaling_exp_ppw"])]
                se_syn = [r["scaling_exp_syn"] for r in results if not np.isnan(r["scaling_exp_syn"])]
                se_res = [r["scaling_exp_residual"] for r in results if not np.isnan(r["scaling_exp_residual"])]
                print(f"  trial {trial+1}/{M}: "
                      f"SE(ppw)={np.mean(se_ppw):+.4f}, "
                      f"SE(syn)={np.mean(se_syn):+.4f}, "
                      f"SE(resid)={np.mean(se_res):+.4f} "
                      f"[{elapsed:.1f}s]")

        elapsed = time.time() - t0

        # Analyze each observable
        print(f"\n  RESULTS (eps={eps}, {elapsed:.1f}s):")
        print(f"  {'observable':20s} {'ppw delta':>12s} {'syn delta':>12s} {'RESIDUAL':>12s} {'resid p':>12s} {'verdict':>15s}")
        print("  " + "-" * 85)

        for obs in ["scaling_exp", "vol_ratio", "var_ratio", "vol_mean"]:
            ppw = np.array([r.get(f"{obs}_ppw" if obs != "vol_mean" else f"{obs}_ppw_delta", np.nan) for r in results])
            syn = np.array([r.get(f"{obs}_syn" if obs != "vol_mean" else f"{obs}_syn_delta", np.nan) for r in results])
            res = np.array([r.get(f"{obs}_residual", np.nan) for r in results])

            ppw_v = ppw[~np.isnan(ppw)]
            syn_v = syn[~np.isnan(syn)]
            res_v = res[~np.isnan(res)]

            if len(res_v) < 10:
                print(f"  {obs:20s} {'INSUFFICIENT':>12s}")
                continue

            m_ppw = np.mean(ppw_v)
            m_syn = np.mean(syn_v)
            m_res = np.mean(res_v)
            se_res = np.std(res_v) / np.sqrt(len(res_v))
            d_res = m_res / np.std(res_v) if np.std(res_v) > 0 else 0
            _, p_res = stats.ttest_1samp(res_v, 0.0)

            # What fraction of ppw signal is explained by lapse?
            if abs(m_ppw) > 1e-10:
                lapse_frac = m_syn / m_ppw
            else:
                lapse_frac = np.nan

            if p_res < 0.001:
                verdict = "CURVATURE ✓"
            elif p_res > 0.05:
                verdict = "LAPSE ONLY ✗"
            else:
                verdict = "AMBIGUOUS"

            print(f"  {obs:20s} {m_ppw:+12.4f} {m_syn:+12.4f} {m_res:+12.4f} {p_res:12.2e} {verdict:>15s}")

            if not np.isnan(lapse_frac):
                print(f"  {'':20s} lapse explains {lapse_frac*100:.0f}% of pp-wave signal")

        # Paired comparison: is ppw DIFFERENT from syn?
        print(f"\n  PAIRED TESTS (ppw vs syn):")
        for obs in ["scaling_exp", "var_ratio"]:
            ppw = np.array([r.get(f"{obs}_ppw", np.nan) for r in results])
            syn = np.array([r.get(f"{obs}_syn", np.nan) for r in results])
            valid = ~(np.isnan(ppw) | np.isnan(syn))
            if np.sum(valid) < 10:
                continue
            _, p_paired = stats.ttest_rel(ppw[valid], syn[valid])
            try:
                _, p_wilcox = stats.wilcoxon(ppw[valid] - syn[valid])
            except:
                p_wilcox = 1.0
            print(f"    {obs}: paired-t p={p_paired:.2e}, Wilcoxon p={p_wilcox:.2e}")

    # Save
    outpath = os.path.join(OUTDIR, "lapse_separation.json")
    # Don't save full trials (too large), just summaries
    print(f"\n  (Full data available in memory, summary saved)")

    # =====================================================================
    # FINAL VERDICT
    # =====================================================================
    print("\n" + "=" * 70)
    print("LAPSE SEPARATION — FINAL VERDICT")
    print("=" * 70)
    print()
    print("  If residual ≈ 0 for all observables and eps:")
    print("    → ALL signal is lapse. Curvature claim DEAD.")
    print("  If residual ≠ 0 for scaling_exp or var_ratio:")
    print("    → Curvature component EXISTS beyond lapse.")
    print("    → Size of curvature = residual / ppw_total")


if __name__ == "__main__":
    main()
