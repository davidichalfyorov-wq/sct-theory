"""
Discovery Run 001 — Analytical Derivation
===========================================

Goal: Derive the expected scaling exponent deviation Δα for pp-wave
from first principles, compare with numerical data.

THEORY:
The volume of a causal diamond of proper time τ in d dimensions:

  V(τ) = V_flat(τ) × [1 + Σ a_k × I_k × τ^{2k}]

where I_k are curvature invariants:
  k=1: I_1 = R (Ricci scalar) → coefficient from Gibbons-Solodukhin (2007)
  k=2: I_2 involves R², Ric², Riem² → coefficient less known for Lorentzian

For pp-wave: R = 0, Ric = 0. Leading correction at O(τ⁴):
  V(τ) = V_flat(τ) × [1 + a₂ × K × τ⁴ + ...]
  where K = Kretschner scalar = R_{μνρσ}R^{μνρσ}

For our pp-wave f(x,y) = x²-y², the nonzero Riemann components are:
  R_{txtx} = -ε∂²f/∂x² /2 = -ε
  R_{tyty} = -ε∂²f/∂y² /2 = +ε
  (in Brinkmann coordinates; our coordinates differ slightly)

K = 8ε² (for this specific profile)

APPROACH: Instead of deriving a₂ from first principles (hard for Lorentzian),
we EXTRACT it from numerical data at one ε value, then PREDICT at other ε.

If Δα = c × K × τ_eff⁴, then:
  c × K = Δα / τ_eff⁴
Extracted at ε=10: c × 800 = Δα₁₀ / τ_eff⁴
Predicted at ε=5: Δα₅ = c × 200 × τ_eff⁴ = Δα₁₀ × (200/800) = Δα₁₀ / 4
Predicted at ε=20: Δα₂₀ = Δα₁₀ × (3200/800) = Δα₁₀ × 4

This is a PARAMETER-FREE prediction (extracted at one point, tested at others).

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
from scipy.optimize import curve_fit
import json, time, gc, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import *

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
os.makedirs(OUTDIR, exist_ok=True)

T = 1.0
# Finer tau bins for accurate V(τ) profile
TAU_FINE = np.linspace(0.02, 0.40, 20)


def volume_profile(C, pts, N, tau_edges):
    """Compute mean interval volume V(τ) in fine tau bins.

    Uses ALL causal pairs (no subsampling).
    Returns tau_mid array and V array.
    """
    t = pts[:, 0]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = pts[:, 1][np.newaxis, :] - pts[:, 1][:, np.newaxis]
    dy = pts[:, 2][np.newaxis, :] - pts[:, 2][:, np.newaxis]
    dz = pts[:, 3][np.newaxis, :] - pts[:, 3][:, np.newaxis]
    tau2 = dt**2 - dx**2 - dy**2 - dz**2

    C2 = C @ C  # interval volumes

    causal = C > 0.5
    tau_vals = np.where(causal & (tau2 > 0), np.sqrt(tau2), 0)

    tau_mid = []
    v_mean = []
    v_count = []

    for i in range(len(tau_edges) - 1):
        lo, hi = tau_edges[i], tau_edges[i+1]
        mask = causal & (tau_vals >= lo) & (tau_vals < hi)
        vols = C2[mask].astype(float)
        if len(vols) >= 10:
            tau_mid.append((lo + hi) / 2)
            v_mean.append(float(np.mean(vols)))
            v_count.append(len(vols))

    return np.array(tau_mid), np.array(v_mean), np.array(v_count)


def main():
    N = 500
    M = 30

    print("=" * 70)
    print("ANALYTICAL DERIVATION: Δα from Kretschner scalar")
    print("=" * 70)

    # ===================================================================
    # STEP 1: Extract V(τ) profile for flat and pp-wave at ε=10
    # ===================================================================
    print("\nSTEP 1: Extract V(τ) profiles (pp-wave ε=10, M=30)")

    ratios_by_tau = {}  # tau -> list of V_curved/V_flat ratios

    for trial in range(M):
        rng = np.random.default_rng(trial * 1000 + 100)
        pts = sprinkle_4d(N, T, rng)

        C_flat = causal_flat(pts)
        C_curv = causal_ppwave_quad(pts, 10.0)

        tau_f, vf, _ = volume_profile(C_flat, pts, N, TAU_FINE)
        tau_c, vc, _ = volume_profile(C_curv, pts, N, TAU_FINE)

        del C_flat, C_curv; gc.collect()

        # Align bins
        for i, t in enumerate(tau_f):
            t_key = f"{t:.4f}"
            if t_key not in ratios_by_tau:
                ratios_by_tau[t_key] = {"tau": t, "ratios": [], "vf": [], "vc": []}
            # Find matching tau in curved
            idx_c = np.argmin(np.abs(tau_c - t))
            if abs(tau_c[idx_c] - t) < 0.005 and vf[i] > 0.1:
                ratios_by_tau[t_key]["ratios"].append(vc[idx_c] / vf[i])
                ratios_by_tau[t_key]["vf"].append(vf[i])
                ratios_by_tau[t_key]["vc"].append(vc[idx_c])

    # Compute mean ratio at each tau
    tau_arr = []
    ratio_arr = []
    ratio_se = []
    print(f"\n  {'tau':>8s} {'V_flat':>10s} {'V_curv':>10s} {'ratio':>10s} {'ratio_se':>10s} {'n':>5s}")
    print("  " + "-" * 55)

    for t_key in sorted(ratios_by_tau.keys()):
        d = ratios_by_tau[t_key]
        if len(d["ratios"]) < 10:
            continue
        tau = d["tau"]
        r = np.mean(d["ratios"])
        r_se = np.std(d["ratios"]) / np.sqrt(len(d["ratios"]))
        vf_mean = np.mean(d["vf"])
        vc_mean = np.mean(d["vc"])
        tau_arr.append(tau)
        ratio_arr.append(r)
        ratio_se.append(r_se)
        print(f"  {tau:8.4f} {vf_mean:10.2f} {vc_mean:10.2f} {r:10.4f} {r_se:10.4f} {len(d['ratios']):5d}")

    tau_arr = np.array(tau_arr)
    ratio_arr = np.array(ratio_arr)

    # ===================================================================
    # STEP 2: Fit ratio = 1 + a₂ × K × τ⁴
    # ===================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Fit V_curved/V_flat = 1 + a × τ^n")
    print("=" * 70)

    # Model 1: ratio = 1 + a × τ⁴ (theory prediction for Weyl)
    def model_tau4(tau, a):
        return 1 + a * tau**4

    # Model 2: ratio = 1 + a × τ^n (free exponent)
    def model_taun(tau, a, n):
        return 1 + a * tau**n

    # Model 3: ratio = 1 + a × τ² + b × τ⁴ (with possible Ricci + Weyl)
    def model_tau24(tau, a, b):
        return 1 + a * tau**2 + b * tau**4

    if len(tau_arr) >= 5:
        # Fit model 1
        try:
            popt1, pcov1 = curve_fit(model_tau4, tau_arr, ratio_arr, p0=[-100])
            a4 = popt1[0]
            pred1 = model_tau4(tau_arr, *popt1)
            ss_res1 = np.sum((ratio_arr - pred1)**2)
            ss_tot = np.sum((ratio_arr - np.mean(ratio_arr))**2)
            r2_1 = 1 - ss_res1/ss_tot if ss_tot > 0 else 0
            print(f"\n  Model 1: ratio = 1 + a×τ⁴")
            print(f"    a = {a4:.2f}, R² = {r2_1:.4f}")
            K_10 = 8 * 10**2  # K at eps=10
            a2_coeff = a4 / K_10
            print(f"    a₂ = a/K = {a4:.2f}/{K_10} = {a2_coeff:.6f}")
        except:
            a4 = np.nan; r2_1 = 0; a2_coeff = 0
            print("  Model 1 fit failed")

        # Fit model 2
        try:
            popt2, pcov2 = curve_fit(model_taun, tau_arr, ratio_arr, p0=[-100, 4])
            a_n, n_fit = popt2
            pred2 = model_taun(tau_arr, *popt2)
            ss_res2 = np.sum((ratio_arr - pred2)**2)
            r2_2 = 1 - ss_res2/ss_tot if ss_tot > 0 else 0
            print(f"\n  Model 2: ratio = 1 + a×τ^n (free n)")
            print(f"    a = {a_n:.2f}, n = {n_fit:.2f}, R² = {r2_2:.4f}")
        except:
            n_fit = np.nan; r2_2 = 0
            print("  Model 2 fit failed")

        # Fit model 3
        try:
            popt3, pcov3 = curve_fit(model_tau24, tau_arr, ratio_arr, p0=[0, -100])
            a2_fit, b4_fit = popt3
            pred3 = model_tau24(tau_arr, *popt3)
            ss_res3 = np.sum((ratio_arr - pred3)**2)
            r2_3 = 1 - ss_res3/ss_tot if ss_tot > 0 else 0
            print(f"\n  Model 3: ratio = 1 + a×τ² + b×τ⁴")
            print(f"    a = {a2_fit:.2f}, b = {b4_fit:.2f}, R² = {r2_3:.4f}")
            if abs(a2_fit) > 0.1:
                print(f"    WARNING: τ² term nonzero (a={a2_fit:.2f})")
                print(f"    This should be 0 for vacuum (R=0). Possible:")
                print(f"    - Nonlinear pp-wave effects at ε=10")
                print(f"    - CRN midpoint approximation artifact")
                print(f"    - Higher-order Ricci contribution")
        except:
            r2_3 = 0; a2_fit = 0; b4_fit = 0
            print("  Model 3 fit failed")

    # ===================================================================
    # STEP 3: PREDICTION — use a₂ from ε=10 to predict ε=5 and ε=20
    # ===================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Parameter-free predictions")
    print("=" * 70)

    # Extract coefficient at eps=10
    # Run eps=5 and eps=20 for comparison
    eps_test = [2.0, 5.0, 20.0]
    observed_dalpha = {}

    for eps in eps_test:
        dalphas = []
        for trial in range(M):
            rng = np.random.default_rng(trial * 1000 + 100)
            pts = sprinkle_4d(N, T, rng)

            C_flat = causal_flat(pts)
            C_curv = causal_ppwave_quad(pts, eps)

            tau_f, vf, _ = volume_profile(C_flat, pts, N, TAU_FINE)
            tau_c, vc, _ = volume_profile(C_curv, pts, N, TAU_FINE)

            del C_flat, C_curv; gc.collect()

            # Compute scaling exponents
            mask_f = (vf > 0.1) & (tau_f > 0.03)
            mask_c = (vc > 0.1) & (tau_c > 0.03)
            if np.sum(mask_f) >= 3 and np.sum(mask_c) >= 3:
                alpha_f, _ = np.polyfit(np.log(tau_f[mask_f]), np.log(vf[mask_f]), 1)
                alpha_c, _ = np.polyfit(np.log(tau_c[mask_c]), np.log(vc[mask_c]), 1)
                dalphas.append(alpha_c - alpha_f)

        observed_dalpha[eps] = {
            "mean": float(np.mean(dalphas)),
            "se": float(np.std(dalphas) / np.sqrt(len(dalphas))),
        }
        print(f"  eps={eps:5.1f}: Δα = {np.mean(dalphas):+.4f} +/- {np.std(dalphas)/np.sqrt(len(dalphas)):.4f}")

    # Also get eps=10 result (already computed above)
    dalpha_10 = []
    for trial in range(M):
        rng = np.random.default_rng(trial * 1000 + 100)
        pts = sprinkle_4d(N, T, rng)
        C_flat = causal_flat(pts)
        C_curv = causal_ppwave_quad(pts, 10.0)
        tau_f, vf, _ = volume_profile(C_flat, pts, N, TAU_FINE)
        tau_c, vc, _ = volume_profile(C_curv, pts, N, TAU_FINE)
        del C_flat, C_curv; gc.collect()
        mask_f = (vf > 0.1) & (tau_f > 0.03)
        mask_c = (vc > 0.1) & (tau_c > 0.03)
        if np.sum(mask_f) >= 3 and np.sum(mask_c) >= 3:
            af, _ = np.polyfit(np.log(tau_f[mask_f]), np.log(vf[mask_f]), 1)
            ac, _ = np.polyfit(np.log(tau_c[mask_c]), np.log(vc[mask_c]), 1)
            dalpha_10.append(ac - af)

    da10_mean = np.mean(dalpha_10)
    observed_dalpha[10.0] = {"mean": da10_mean, "se": float(np.std(dalpha_10)/np.sqrt(len(dalpha_10)))}

    # Prediction: Δα(ε) = Δα(10) × (K(ε)/K(10)) = Δα(10) × (ε/10)²
    print(f"\n  Reference: Δα(ε=10) = {da10_mean:+.4f}")
    print(f"\n  {'eps':>6s} {'K/K₁₀':>8s} {'predicted':>12s} {'observed':>12s} {'obs_se':>10s} {'ratio':>10s}")
    print("  " + "-" * 60)

    predictions = {}
    for eps in [2.0, 5.0, 10.0, 20.0]:
        K_ratio = (eps / 10.0)**2
        predicted = da10_mean * K_ratio
        obs = observed_dalpha.get(eps, {})
        obs_mean = obs.get("mean", da10_mean if eps == 10.0 else np.nan)
        obs_se = obs.get("se", 0)
        ratio = obs_mean / predicted if abs(predicted) > 1e-10 else np.nan
        print(f"  {eps:6.1f} {K_ratio:8.2f} {predicted:+12.4f} {obs_mean:+12.4f} {obs_se:10.4f} {ratio:10.3f}")
        predictions[eps] = {"predicted": predicted, "observed": obs_mean, "ratio": ratio}

    # Assess prediction quality
    print("\n  PREDICTION QUALITY:")
    ratios = [v["ratio"] for v in predictions.values() if not np.isnan(v["ratio"]) and v["ratio"] != 1.0]
    if ratios:
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        print(f"    Mean obs/pred ratio (excluding calibration): {mean_ratio:.3f} +/- {std_ratio:.3f}")
        if abs(mean_ratio - 1.0) < 0.3 and std_ratio < 0.3:
            print(f"    => GOOD: Δα ~ K prediction works (within 30%)")
            analytical_verdict = "CONFIRMED"
        elif abs(mean_ratio - 1.0) < 0.5:
            print(f"    => FAIR: Δα ~ K prediction approximate (within 50%)")
            analytical_verdict = "APPROXIMATE"
        else:
            print(f"    => POOR: Δα ~ K prediction fails")
            analytical_verdict = "FAILED"
    else:
        analytical_verdict = "INSUFFICIENT"

    # ===================================================================
    # STEP 4: Best-fit scaling law
    # ===================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Best-fit Δα = c × ε^β")
    print("=" * 70)

    all_eps = np.array([2.0, 5.0, 10.0, 20.0])
    all_dalpha = np.array([observed_dalpha[e]["mean"] for e in all_eps])

    # Only use eps where signal is nonzero
    mask = np.abs(all_dalpha) > 0.01
    if np.sum(mask) >= 3:
        log_eps = np.log(all_eps[mask])
        log_da = np.log(np.abs(all_dalpha[mask]))
        beta, log_c = np.polyfit(log_eps, log_da, 1)
        c = np.exp(log_c)

        print(f"  Best fit: |Δα| = {c:.4f} × ε^{beta:.2f}")
        print(f"  If β≈2: Δα ~ ε² ~ K (Kretschner scaling)")
        print(f"  If β≈1: Δα ~ ε ~ |Riem| (Riemann scaling)")
        print(f"  Observed: β = {beta:.2f}")

        if abs(beta - 2.0) < 0.3:
            print(f"  => CONSISTENT with Kretschner (K ~ ε²) scaling")
        elif abs(beta - 1.0) < 0.3:
            print(f"  => CONSISTENT with Riemann (|Riem| ~ ε) scaling")
        else:
            print(f"  => INTERMEDIATE: β={beta:.2f} between Riemann and Kretschner")

    # Save
    output = {
        "volume_ratio_fit": {
            "model1_r2": float(r2_1) if 'r2_1' in dir() else None,
            "model2_n": float(n_fit) if 'n_fit' in dir() else None,
            "model2_r2": float(r2_2) if 'r2_2' in dir() else None,
        },
        "observed_dalpha": {str(k): v for k, v in observed_dalpha.items()},
        "predictions": {str(k): v for k, v in predictions.items()},
        "best_fit_beta": float(beta) if 'beta' in dir() else None,
        "analytical_verdict": analytical_verdict,
    }

    outpath = os.path.join(OUTDIR, "analytical.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    print("\n" + "=" * 70)
    print(f"ANALYTICAL VERDICT: {analytical_verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
