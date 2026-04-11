"""
Test 3: Schwarzschild Lapse Separation
=======================================
Schwarzschild: g_tt = -(1+2Phi) where Phi = -eps/r.
Synthetic lapse: ds² = -(1+2Phi)dt² + dx² + dy² + dz²
Full Schwarzschild: ds² = -(1+2Phi)dt² + (1-2Phi)(dx²+dy²+dz²)

Difference: Schwarzschild has (1-2Phi) in the spatial part.
The residual captures the effect of spatial metric modification.

For weak-field: spatial curvature from (1-2Phi) = Newtonian tidal forces.

PRE-REGISTERED:
  - Residual p < 0.001 → spatial curvature component exists
  - Residual p > 0.05 → signal is pure lapse
  - Adversarial proxy on residual

Author: David Alfyorov
"""
import numpy as np
from scipy import stats
import json, time, gc, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import *
from discovery_weyl_probes import interval_volumes_binned, weyl_observables

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
T = 1.0
tau_bins = np.array([0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.45])


def causal_schw_lapse_only(pts, eps):
    """Schwarzschild lapse only: ds² = -(1+2Phi)dt² + dx²+dy²+dz²
    Same g_tt as Schwarzschild, but spatial part = flat.
    """
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dr2 = ((x[np.newaxis, :] - x[:, np.newaxis])**2 +
           (y[np.newaxis, :] - y[:, np.newaxis])**2 +
           (z[np.newaxis, :] - z[:, np.newaxis])**2)
    xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
    ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
    zm = (z[np.newaxis, :] + z[:, np.newaxis]) / 2.0
    rm = np.sqrt(xm**2 + ym**2 + zm**2) + 0.3
    Phi = -eps / rm
    # Only lapse: (1+2Phi)*dt² > dr² (no spatial metric modification)
    return (((1 + 2*Phi) * dt**2 > dr2) & (dt > 0) & ((1 + 2*Phi) > 0)).astype(np.float64)


METRIC_FNS["schw_lapse"] = causal_schw_lapse_only
SEED_OFFSETS["schw_lapse"] = 300  # same as schwarzschild


def crn_trial_schw_triple(seed, N, T, eps):
    rng = np.random.default_rng(seed + 300)
    pts = sprinkle_4d(N, T, rng)
    result = {"seed": seed, "N": N, "eps": eps}

    C_flat = causal_flat(pts)
    C_schw = causal_schwarzschild(pts, eps)
    C_lapse = causal_schw_lapse_only(pts, eps)

    for tag, C in [("flat", C_flat), ("schw", C_schw), ("lapse", C_lapse)]:
        rng_b = np.random.default_rng(seed + 7777)
        bins = interval_volumes_binned(C, pts, N, tau_bins, rng=rng_b)
        result[f"bins_{tag}"] = bins

        A = build_link_graph(C)
        gs, _ = graph_statistics(A)
        for key in gs:
            result[f"{key}_{tag}"] = gs[key]
        del A

    w_schw = weyl_observables(result["bins_flat"], result["bins_schw"], tau_bins)
    w_lapse = weyl_observables(result["bins_flat"], result["bins_lapse"], tau_bins)

    for obs in ["scaling_exp_delta", "var_ratio_delta", "vol_ratio_delta"]:
        s_v = w_schw.get(obs, np.nan)
        l_v = w_lapse.get(obs, np.nan)
        result[f"{obs}_schw"] = s_v
        result[f"{obs}_lapse"] = l_v
        result[f"{obs}_residual"] = (s_v - l_v) if not (np.isnan(s_v) or np.isnan(l_v)) else np.nan

    for tag in ["flat", "schw", "lapse"]:
        del result[f"bins_{tag}"]

    del C_flat, C_schw, C_lapse; gc.collect()
    return result


def main():
    N = 500
    M = 50
    eps = 0.05

    print("=" * 70)
    print(f"TEST 3: Schwarzschild Lapse Separation")
    print(f"N={N}, M={M}, eps={eps}")
    print("Schwarzschild = lapse(1+2Phi) + spatial(1-2Phi)")
    print("Synthetic lapse = lapse(1+2Phi) only")
    print("Residual = spatial curvature contribution")
    print("=" * 70)

    t0 = time.time()
    results = []
    for trial in range(M):
        res = crn_trial_schw_triple(trial * 1000, N, T, eps)
        results.append(res)
        if (trial + 1) % 10 == 0:
            elapsed = time.time() - t0
            s = [r.get("scaling_exp_delta_schw", np.nan) for r in results]
            l = [r.get("scaling_exp_delta_lapse", np.nan) for r in results]
            r_val = [r.get("scaling_exp_delta_residual", np.nan) for r in results]
            s_v = [v for v in s if not np.isnan(v)]
            l_v = [v for v in l if not np.isnan(v)]
            r_v = [v for v in r_val if not np.isnan(v)]
            print(f"  trial {trial+1}/{M}: "
                  f"schw={np.mean(s_v):+.4f}, lapse={np.mean(l_v):+.4f}, "
                  f"resid={np.mean(r_v):+.4f} [{elapsed:.1f}s]")

    elapsed = time.time() - t0

    print(f"\n  RESULTS ({elapsed:.1f}s):")
    print(f"  {'observable':20s} {'schw delta':>12s} {'lapse delta':>12s} {'RESIDUAL':>12s} {'resid p':>12s} {'lapse %':>10s}")
    print("  " + "-" * 80)

    for obs in ["scaling_exp_delta", "var_ratio_delta", "vol_ratio_delta"]:
        schw = np.array([r.get(f"{obs}_schw", np.nan) for r in results])
        lapse = np.array([r.get(f"{obs}_lapse", np.nan) for r in results])
        resid = np.array([r.get(f"{obs}_residual", np.nan) for r in results])

        valid = ~(np.isnan(schw) | np.isnan(lapse) | np.isnan(resid))
        if np.sum(valid) < 10:
            print(f"  {obs:20s} INSUFFICIENT")
            continue

        m_s = np.mean(schw[valid])
        m_l = np.mean(lapse[valid])
        m_r = np.mean(resid[valid])
        _, p_r = stats.ttest_1samp(resid[valid], 0.0)
        lapse_pct = m_l / m_s * 100 if abs(m_s) > 1e-10 else np.nan

        verdict = "SPATIAL CURV ✓" if p_r < 0.001 else ("AMBIG" if p_r < 0.05 else "LAPSE ONLY ✗")
        print(f"  {obs:20s} {m_s:+12.4f} {m_l:+12.4f} {m_r:+12.4f} {p_r:12.2e} {lapse_pct:9.0f}%  {verdict}")

    outpath = os.path.join(OUTDIR, "schw_lapse_sep.json")
    with open(outpath, "w") as f:
        json.dump({"N": N, "M": M, "eps": eps, "elapsed": elapsed}, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
