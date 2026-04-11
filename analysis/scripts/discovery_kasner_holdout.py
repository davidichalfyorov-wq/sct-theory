"""
Discovery Run 001 — Kasner Holdout Geometry
=============================================

Kasner metric: ds² = -dt² + t^{2p₁}dx² + t^{2p₂}dy² + t^{2p₃}dz²
with Kasner conditions: p₁+p₂+p₃ = 1, p₁²+p₂²+p₃² = 1

Properties:
  - Vacuum solution (R=0, Ric=0)
  - Anisotropic (C² ≠ 0 for generic p_i)
  - NOT a pp-wave — genuinely different geometry
  - Has a singularity at t=0

For our CRN: sprinkle in causal diamond centered at t=t₀ (avoiding t=0),
apply Kasner causal condition.

Kasner causal condition (midpoint approximation):
  dt² > t_m^{2p₁} dx² + t_m^{2p₂} dy² + t_m^{2p₃} dz²

CRN design: same coordinates, different metric (flat vs Kasner).

We use p₁=-1/3, p₂=2/3, p₃=2/3 (one of the standard Kasner solutions).
To control strength: use p_i(ε) = δ_i + ε*(p_i - δ_i) where δ_i=1/3.
At ε=0: isotropic (flat). At ε=1: full Kasner.

Actually simpler: shift the diamond center to t₀ = 1.0 (far from singularity)
and use eps to control the Kasner exponents.

Even simpler: implement as a perturbation around flat:
  ds² = -dt² + (1 + ε·h₁(t))dx² + (1 + ε·h₂(t))dy² + (1 + ε·h₃(t))dz²
where h_i(t) = 2p_i·ln(t/t₀) for Kasner.

For the causal condition (midpoint):
  dt² > (1 + ε·h₁(t_m))dx² + (1 + ε·h₂(t_m))dy² + (1 + ε·h₃(t_m))dz²

But this is only valid for small ε. For a true Kasner, need t^{2p_i}.

Let me implement the EXACT Kasner with shifted center.

Author: David Alfyorov
"""
import numpy as np
from scipy import stats
import json, time, gc, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import *
from discovery_weyl_probes import (
    interval_volumes_binned, weyl_observables, adversarial_weyl,
)

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
os.makedirs(OUTDIR, exist_ok=True)

T = 1.0
tau_bins = np.array([0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.45])

# Kasner exponents: p1+p2+p3=1, p1²+p2²+p3²=1
# Standard choice: p1=-1/3, p2=p3=2/3
P_KASNER = (-1.0/3, 2.0/3, 2.0/3)


def causal_kasner(pts, eps):
    """Kasner metric with controlled anisotropy.

    Interpolate between flat (eps=0) and full Kasner (eps=1):
    a_i(t) = |t + t0|^{eps * p_i} where t0 = 1.5 (shift away from singularity)

    Causal condition (midpoint approx):
    dt² > a₁(t_m)² dx² + a₂(t_m)² dy² + a₃(t_m)² dz²

    At eps=0: a_i = 1 → flat.
    At eps=1: a_i = |t+1.5|^{p_i} → Kasner near t+1.5.
    """
    p1, p2, p3 = P_KASNER
    t0 = 1.5  # shift to avoid singularity

    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]

    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]

    tm = (t[np.newaxis, :] + t[:, np.newaxis]) / 2.0 + t0  # shifted midpoint

    # Scale factors
    a1_sq = np.abs(tm)**(2 * eps * p1)
    a2_sq = np.abs(tm)**(2 * eps * p2)
    a3_sq = np.abs(tm)**(2 * eps * p3)

    ds2_spatial = a1_sq * dx**2 + a2_sq * dy**2 + a3_sq * dz**2

    return ((dt**2 > ds2_spatial) & (dt > 0)).astype(np.float64)


METRIC_FNS["kasner"] = causal_kasner
SEED_OFFSETS["kasner"] = 700


def crn_trial_kasner(seed, N, T, eps):
    """CRN trial for Kasner geometry with Weyl probes."""
    rng = np.random.default_rng(seed + SEED_OFFSETS["kasner"])
    pts = sprinkle_4d(N, T, rng)

    result = {"seed": seed, "N": N, "eps": eps}

    C_flat = causal_flat(pts)
    C_curv = causal_kasner(pts, eps)

    # Weyl probes
    rng_b = np.random.default_rng(seed + 7777)
    bins_flat = interval_volumes_binned(C_flat, pts, N, tau_bins, rng=rng_b)
    rng_b = np.random.default_rng(seed + 7777)
    bins_curved = interval_volumes_binned(C_curv, pts, N, tau_bins, rng=rng_b)

    w = weyl_observables(bins_flat, bins_curved, tau_bins)
    result.update(w)

    # Degree stats
    A_flat = build_link_graph(C_flat)
    gs_flat, _ = graph_statistics(A_flat)
    A_curv = build_link_graph(C_curv)
    gs_curv, _ = graph_statistics(A_curv)
    for key in gs_flat:
        result[f"{key}_delta"] = gs_curv[key] - gs_flat[key]

    fr_f, _ = forman_ricci(A_flat)
    fr_c, _ = forman_ricci(A_curv)
    result["forman_mean_delta"] = fr_c["F_mean"] - fr_f["F_mean"]

    del C_flat, C_curv, A_flat, A_curv; gc.collect()
    return result


def main():
    N = 500
    M = 40

    print("=" * 70)
    print("KASNER HOLDOUT GEOMETRY")
    print(f"N={N}, M={M}")
    print(f"Kasner exponents: p = {P_KASNER}")
    print("Properties: vacuum (R=0), anisotropic (C²≠0), NOT pp-wave")
    print("=" * 70)

    all_results = {}

    for eps in [0.5, 1.0, 2.0, 5.0]:
        label = f"kasner_eps{eps}"
        print(f"\n--- {label} ---")
        t0 = time.time()

        results = []
        for trial in range(M):
            res = crn_trial_kasner(trial * 1000, N, T, eps)
            results.append(res)

        elapsed = time.time() - t0

        for obs_key in ["scaling_exp_delta", "var_ratio_delta", "vol_ratio_delta"]:
            v = adversarial_weyl(results, obs_key, label)
            all_results[f"{label}_{obs_key}"] = v

        all_results[f"{label}_meta"] = {"eps": eps, "elapsed_sec": elapsed}
        print(f"  [{elapsed:.1f}s]")

    # Save
    outpath = os.path.join(OUTDIR, "kasner_holdout.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # Summary
    print("\n" + "=" * 70)
    print("KASNER HOLDOUT SUMMARY")
    print("=" * 70)
    print(f"{'eps':>6s} {'scaling_exp d':>14s} {'p':>12s} {'R²_multi':>10s} {'verdict':>22s}")
    print("-" * 70)

    for eps in [0.5, 1.0, 2.0, 5.0]:
        v = all_results.get(f"kasner_eps{eps}_scaling_exp_delta", {})
        if isinstance(v, dict) and "verdict" in v:
            print(f"{eps:6.1f} {v.get('cohen_d',0):+14.3f} {v.get('p_value',1):12.2e} "
                  f"{v.get('r2_multiple',0):10.3f} {v['verdict']:>22s}")

    # Verdict
    genuine = sum(1 for eps in [0.5, 1.0, 2.0, 5.0]
                  for obs in ["scaling_exp_delta", "var_ratio_delta"]
                  if all_results.get(f"kasner_eps{eps}_{obs}", {}).get("verdict") == "DETECTED (genuine)")

    if genuine > 0:
        print(f"\n*** KASNER HOLDOUT: {genuine} GENUINE detections ***")
        print("*** Weyl probes generalize beyond pp-wave! ***")
    else:
        print(f"\n*** KASNER HOLDOUT: No genuine detections ***")
        print("*** Weyl probes may be pp-wave specific ***")


if __name__ == "__main__":
    main()
