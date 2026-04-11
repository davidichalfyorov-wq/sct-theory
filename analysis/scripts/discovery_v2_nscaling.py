"""
Discovery Run 001 — Variant 2: N-Scaling for Interval Volume
==============================================================
Does vol_mean GENUINE survive at N=1000? Does the effect SIZE grow?

Also: analytical prediction check.
In flat d=4 Minkowski, the expected number of elements in a causal
interval of proper time tau is:
    E[V] = rho * V_d(tau) = rho * pi * tau^4 / 24  (d=4)
where rho = N / V_diamond is the sprinkling density.

In weakly curved Schwarzschild (weak field, Phi = -eps/r):
    V_curved / V_flat ~ 1 + (R/6) * tau^2 / (d+2) + O(tau^4)
where R = Ricci scalar.

For our Schwarzschild: R ≈ 0 (vacuum solution, Ric=0!). So the
volume deficit should come from HIGHER ORDER terms (Weyl, not Ricci).

Wait — our "Schwarzschild" is actually an isotropic weak-field metric:
    ds^2 = -(1+2Phi)dt^2 + (1-2Phi)(dx^2+dy^2+dz^2)
with Phi = -eps/r. This is NOT vacuum! It has R = -2*nabla^2(Phi) ≠ 0
in general (Laplacian of 1/r is -4pi*delta, but our r has an offset +0.3
so Phi is smooth and nabla^2(Phi) ≠ 0 everywhere).

So the Ricci scalar is nonzero, and the volume deficit should be
proportional to the integrated R.

Author: David Alfyorov
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import *
from discovery_v2_intervals import interval_observables, crn_trial_v2, adversarial_v2
import json, time

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
os.makedirs(OUTDIR, exist_ok=True)

T = 1.0

def main():
    all_results = {}

    # ===================================================================
    # PART 1: N-scaling (N=500, 1000)
    # ===================================================================
    print("=" * 70)
    print("PART 1: N-Scaling of Interval Volume")
    print("=" * 70)

    TESTS = [
        # (geo, eps, N, M, n_pairs)
        ("schwarzschild", 0.05, 500,  40, 500),    # replicate pilot
        ("schwarzschild", 0.05, 1000, 30, 1000),   # scale up
        ("ppwave_quad",   10.0, 500,  40, 500),     # replicate
        ("ppwave_quad",   10.0, 1000, 30, 1000),    # scale up
        ("ppwave_quad",   20.0, 500,  30, 500),     # stronger eps
    ]

    for geo, eps, N, M, n_pairs in TESTS:
        label = f"{geo}_eps{eps}_N{N}"
        print(f"\n--- {label} (M={M}) ---")
        t0 = time.time()

        results = []
        for trial in range(M):
            res = crn_trial_v2(trial * 1000, N, T, geo, eps, n_pairs=n_pairs)
            results.append(res)
            if (trial + 1) % 10 == 0:
                elapsed = time.time() - t0
                vd = [r["vol_mean_delta"] for r in results]
                cd = [r["chain_mean_delta"] for r in results]
                print(f"  trial {trial+1}/{M}: vol={np.mean(vd):+.3f}, "
                      f"chain={np.mean(cd):+.4f} [{elapsed:.1f}s]")

        elapsed = time.time() - t0

        obs_results = {}
        for obs_key in ["vol_mean_delta", "chain_mean_delta"]:
            v = adversarial_v2(results, obs_key, label)
            obs_results[obs_key] = v

        obs_results["elapsed_sec"] = elapsed
        obs_results["N"] = N
        obs_results["M"] = M
        obs_results["trials"] = results
        all_results[label] = obs_results

    # ===================================================================
    # PART 2: Analytical check — does delta-V scale with eps?
    # ===================================================================
    print("\n" + "=" * 70)
    print("PART 2: Epsilon Scaling of vol_mean (Schwarzschild)")
    print("=" * 70)

    N, M = 500, 30
    eps_values = [0.01, 0.02, 0.05, 0.1, 0.2]
    eps_results = []

    for eps in eps_values:
        results = []
        for trial in range(M):
            res = crn_trial_v2(trial * 1000, N, T, "schwarzschild", eps, n_pairs=500)
            results.append(res)
        vd = [r["vol_mean_delta"] for r in results]
        mean_vd = np.mean(vd)
        se_vd = np.std(vd) / np.sqrt(M)
        eps_results.append({"eps": eps, "vol_delta": mean_vd, "se": se_vd})
        print(f"  eps={eps:.3f}: vol_delta={mean_vd:+.3f} +/- {se_vd:.3f}")

    # Fit power law: vol_delta ~ eps^alpha
    eps_arr = np.array([r["eps"] for r in eps_results])
    vd_arr = np.array([r["vol_delta"] for r in eps_results])
    # All should be negative
    log_eps = np.log(eps_arr)
    log_vd = np.log(np.abs(vd_arr))
    slope, intercept = np.polyfit(log_eps, log_vd, 1)
    r2 = 1 - np.sum((log_vd - (slope*log_eps + intercept))**2) / np.sum((log_vd - np.mean(log_vd))**2)
    print(f"\n  Power law fit: |vol_delta| ~ eps^{slope:.2f} (R²={r2:.4f})")
    print(f"  If slope≈1: V deficit linear in Phi (Newtonian potential) — expected")
    print(f"  If slope≈2: V deficit quadratic — curvature (R ~ nabla²Phi ~ eps)")

    all_results["eps_scaling"] = {
        "eps_values": eps_results,
        "power_law_slope": float(slope),
        "power_law_r2": float(r2),
    }

    # ===================================================================
    # PART 3: Analytical prediction
    # ===================================================================
    print("\n" + "=" * 70)
    print("PART 3: Analytical Prediction")
    print("=" * 70)

    # Our Schwarzschild metric: Phi = -eps/(r+0.3) where r = sqrt(x²+y²+z²)
    # ds² = -(1+2Phi)dt² + (1-2Phi)(dx²+dy²+dz²)
    #
    # Ricci scalar: R = -2*nabla²(Phi) (weak-field, linearized gravity)
    # Phi = -eps/(r+0.3)
    # nabla²(1/r) = -4*pi*delta³(r), but with offset r+0.3:
    # nabla²(1/(r+0.3)) = 2/(r+0.3)³  (for r>0, smooth)
    # So: nabla²(Phi) = eps * 2/(r+0.3)³
    # And: R = -2 * eps * 2/(r+0.3)³ = -4*eps/(r+0.3)³
    #
    # For the causal diamond centered at origin with half-size T/2=0.5:
    # Average r ~ 0.2 (rough), so (r+0.3)³ ~ 0.125
    # R ~ -4*eps/0.125 = -32*eps
    #
    # Volume deficit from R:
    # delta_V / V ~ (R/6) * tau² / 10  (leading order for d=4)
    # With tau ~ T/2 = 0.5: delta_V/V ~ R * 0.25/60 ~ R/240
    # With R ~ -32*eps: delta_V/V ~ -32*eps/240 ~ -0.133*eps
    #
    # At eps=0.05: delta_V/V ~ -0.0067
    # Mean V_flat ~ 2-3 (typical interval size at N=500 in 4D is small)
    # delta_V ~ -0.0067 * 2.5 ~ -0.017
    #
    # But observed: delta_V ~ -2.2. That's 100x larger!
    # This means the volume deficit is NOT from R (linearized).
    # It's from the DIRECT metric change: changing (1±2Phi) changes which
    # pairs are causally related, changing interval sizes dramatically.
    #
    # The dominant effect is NOT curvature but LAPSE and SHIFT:
    # When Phi < 0 (attractive gravity): light cones narrow,
    # fewer pairs are causally related, intervals shrink.
    # This is a coordinate/potential effect, not curvature.

    print("  Analytical estimate for delta_V from Ricci scalar R:")
    print(f"  R ~ -4*eps/(r+0.3)^3 ~ -32*eps (at r~0.2)")
    print(f"  delta_V/V ~ R/(6*10) * tau^2 ~ -0.133*eps")
    print(f"  At eps=0.05: delta_V/V ~ -0.007")
    print(f"  Expected |delta_V| ~ 0.017 (for V~2.5)")
    print(f"  Observed |delta_V| = 2.21")
    print(f"  RATIO: observed/predicted ~ 130")
    print()
    print("  INTERPRETATION: The observed volume deficit is NOT from curvature (R).")
    print("  It is from the LAPSE effect: Phi<0 narrows light cones,")
    print("  reducing the number of causally related pairs.")
    print("  This is a coordinate/potential effect, not a curvature invariant.")
    print()
    print("  CRITICAL QUESTION: Is vol_mean_delta measuring CURVATURE or LAPSE?")
    print("  If lapse → it's a gravitational redshift detector, not curvature.")
    print("  Test: pp-wave has NO lapse (g_tt = -1). If vol_mean works on")
    print("  pp-wave, it's curvature. If only on Schwarzschild, it's lapse.")
    print()

    # Check pp-wave results
    ppw_results = all_results.get("ppwave_quad_eps10.0_N500", {})
    ppw_v = ppw_results.get("vol_mean_delta", {})
    ppw_verdict = ppw_v.get("verdict", "?") if isinstance(ppw_v, dict) else "?"

    schw_results = all_results.get("schwarzschild_eps0.05_N500", {})
    schw_v = schw_results.get("vol_mean_delta", {})
    schw_verdict = schw_v.get("verdict", "?") if isinstance(schw_v, dict) else "?"

    print(f"  Schwarzschild vol_mean: {schw_verdict}")
    print(f"  pp-wave vol_mean:      {ppw_verdict}")

    if "GENUINE" in str(schw_verdict) and "GENUINE" not in str(ppw_verdict):
        print()
        print("  *** vol_mean detects Schwarzschild but NOT pp-wave. ***")
        print("  *** This is consistent with LAPSE detection, not curvature. ***")
        print("  *** The 'GENUINE' result may be measuring g_tt, not R. ***")
        all_results["analytical_verdict"] = "LAPSE_DETECTOR"
    elif "GENUINE" in str(schw_verdict) and "GENUINE" in str(ppw_verdict):
        print("  *** vol_mean detects BOTH → genuine curvature sensitivity ***")
        all_results["analytical_verdict"] = "CURVATURE_DETECTOR"
    else:
        all_results["analytical_verdict"] = "INCONCLUSIVE"

    # Save
    outpath = os.path.join(OUTDIR, "v2_nscaling.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n  N-SCALING:")
    for key, v in all_results.items():
        if key.startswith(("schwarz", "ppwave")):
            vm = v.get("vol_mean_delta", {})
            cm = v.get("chain_mean_delta", {})
            if isinstance(vm, dict) and isinstance(cm, dict):
                print(f"    {key:35s}: vol={vm.get('verdict','?'):>22s}, "
                      f"chain={cm.get('verdict','?'):>22s}")

    print(f"\n  EPS-SCALING: |vol_delta| ~ eps^{slope:.2f} (R²={r2:.4f})")
    print(f"  ANALYTICAL: {all_results.get('analytical_verdict', '?')}")


if __name__ == "__main__":
    main()
