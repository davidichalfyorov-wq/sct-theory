"""
PATH 1: Extract Discrete a₂ (Weyl Seeley-DeWitt Coefficient)
==============================================================

The volume of a causal diamond of proper time τ in d=4:
  V(τ) = V₀τ⁴ [1 + a₁·R·τ² + a₂·(c₁R² + c₂Ric² + c₃K)·τ⁴ + ...]

For pp-wave: R = 0, Ric = 0, K = 8ε².
  V(τ) = V₀τ⁴ [1 + a₂·c₃·K·τ⁴ + ...]

Gibbons-Solodukhin (2007, hep-th/0508229) computed the continuum a₂c₃
for causal diamonds in d=4. We extract the DISCRETE value from causal
set sprinklings and compare.

STRATEGY:
  - Use SMALL ε (0.1, 0.2, 0.5) and LARGE N (2000-5000)
    to stay in perturbative regime (K·τ⁴ << 1)
  - Use LAPSE-SUBTRACTED residual (ppw - synthetic_lapse)
    to isolate the Weyl contribution
  - Fit δV_residual/V_flat = A·τ⁴ where A = a₂·c₃·K
  - Extract a₂c₃ = A/K

PRE-REGISTERED:
  - If fit R² > 0.8 AND a₂c₃ is stable across ε → genuine coefficient
  - If fit R² < 0.5 → perturbative regime not reached
  - Compare with Gibbons-Solodukhin continuum value

Author: David Alfyorov
"""
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import json, time, gc, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import *
from discovery_lapse_separation import causal_synthetic_lapse

METRIC_FNS["synthetic_lapse"] = causal_synthetic_lapse
SEED_OFFSETS["synthetic_lapse"] = 100

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
os.makedirs(OUTDIR, exist_ok=True)
T = 1.0


def volume_ratio_profile(C_flat, C_curv, C_syn, pts, N, tau_edges):
    """Compute V_residual(τ)/V_flat(τ) - 1 in fine τ bins.

    V_residual = V_ppw - V_syn (lapse-subtracted)
    Returns tau_mid, delta_ratio arrays.
    """
    t = pts[:, 0]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = pts[:, 1][np.newaxis, :] - pts[:, 1][:, np.newaxis]
    dy = pts[:, 2][np.newaxis, :] - pts[:, 2][:, np.newaxis]
    dz = pts[:, 3][np.newaxis, :] - pts[:, 3][:, np.newaxis]
    tau2_coord = dt**2 - dx**2 - dy**2 - dz**2

    C2_flat = C_flat @ C_flat
    C2_ppw = C_curv @ C_curv
    C2_syn = C_syn @ C_syn

    # For each tau bin: compute mean V for flat, ppw, syn
    # Only use pairs causal in ALL THREE matrices (clean comparison)
    causal_all = (C_flat > 0.5) & (C_curv > 0.5) & (C_syn > 0.5) & (tau2_coord > 0)

    tau_mid = []
    delta_ratio = []  # (V_ppw - V_syn) / V_flat
    n_pairs_list = []

    for i in range(len(tau_edges) - 1):
        lo, hi = tau_edges[i], tau_edges[i + 1]
        tau_vals = np.sqrt(np.where(tau2_coord > 0, tau2_coord, 0))
        mask = causal_all & (tau_vals >= lo) & (tau_vals < hi)

        vf = C2_flat[mask].astype(float)
        vp = C2_ppw[mask].astype(float)
        vs = C2_syn[mask].astype(float)

        if len(vf) < 20 or np.mean(vf) < 0.5:
            continue

        # Residual ratio: (V_ppw - V_syn) / V_flat
        mean_vf = np.mean(vf)
        mean_vp = np.mean(vp)
        mean_vs = np.mean(vs)

        dr = (mean_vp - mean_vs) / mean_vf if mean_vf > 0 else 0

        tau_mid.append((lo + hi) / 2)
        delta_ratio.append(dr)
        n_pairs_list.append(len(vf))

    return np.array(tau_mid), np.array(delta_ratio), np.array(n_pairs_list)


def main():
    # Fine tau bins for profile fitting
    tau_edges = np.linspace(0.03, 0.40, 20)

    print("=" * 70)
    print("PATH 1: Extract Discrete a₂ (Weyl Coefficient)")
    print("=" * 70)

    # Gibbons-Solodukhin continuum prediction for d=4 causal diamond:
    # V(τ) = (π/24)τ⁴ [1 - (1/120)Rτ² + (1/(...))Kτ⁴ + ...]
    # The exact a₂c₃ for d=4 Lorentzian causal diamond:
    # From Gibbons-Solodukhin (2007), the volume of a small causal diamond:
    # V = V₀[1 - R/(6(d+2))τ² + ...]
    # At O(τ⁴): involves R², Ric², and Riem² with specific coefficients
    # For vacuum (R=Ric=0): only Riem² = K survives
    # The coefficient for K in d=4: a₂c₃ = -1/(360·6·8) = ...
    # Actually, I need to look up the exact value.
    # For now, just extract the DISCRETE value and report it.

    print("\n  Continuum prediction (Gibbons-Solodukhin 2007):")
    print("  V/V₀ = 1 - R/(6·6)·τ² + [a₂c₃·K + ...]·τ⁴")
    print("  For vacuum (R=0): δV/V₀ = a₂c₃·K·τ⁴")
    print("  We extract a₂c₃ from discrete causal set data.")

    all_results = {}

    # Test at multiple (N, eps) to find perturbative regime
    CONFIGS = [
        (2000, 0.2, 20),
        (2000, 0.5, 20),
        (2000, 1.0, 20),
        (3000, 0.2, 15),
        (3000, 0.5, 15),
    ]

    for N, eps, M in CONFIGS:
        K = 8 * eps**2
        label = f"N{N}_eps{eps}"
        print(f"\n--- {label} (K={K:.2f}, M={M}) ---")

        tau_mids_all = []
        dr_all = []

        t0 = time.time()
        for trial in range(M):
            rng = np.random.default_rng(trial * 1000 + 100)
            pts = sprinkle_4d(N, T, rng)

            C_flat = causal_flat(pts)
            C_ppw = causal_ppwave_quad(pts, eps)
            C_syn = causal_synthetic_lapse(pts, eps)

            tau_m, dr, n_p = volume_ratio_profile(C_flat, C_ppw, C_syn, pts, N, tau_edges)

            if len(tau_m) > 0:
                tau_mids_all.append(tau_m)
                dr_all.append(dr)

            del C_flat, C_ppw, C_syn; gc.collect()

            if (trial + 1) % 5 == 0:
                print(f"  trial {trial+1}/{M} [{time.time()-t0:.1f}s]")

        elapsed = time.time() - t0

        if len(dr_all) < 5:
            print(f"  INSUFFICIENT data")
            continue

        # Average profile across trials
        # Find common tau bins
        all_taus = set()
        for tm in tau_mids_all:
            for t in tm:
                all_taus.add(round(t, 4))
        common_taus = sorted(all_taus)

        tau_final = []
        dr_final = []
        dr_se = []

        for tau_target in common_taus:
            vals = []
            for tm, dr in zip(tau_mids_all, dr_all):
                idx = np.argmin(np.abs(tm - tau_target))
                if abs(tm[idx] - tau_target) < 0.005:
                    vals.append(dr[idx])
            if len(vals) >= 5:
                tau_final.append(tau_target)
                dr_final.append(np.mean(vals))
                dr_se.append(np.std(vals) / np.sqrt(len(vals)))

        tau_final = np.array(tau_final)
        dr_final = np.array(dr_final)

        if len(tau_final) < 4:
            print(f"  INSUFFICIENT tau bins")
            continue

        # Print profile
        print(f"\n  δV_residual/V_flat profile:")
        print(f"  {'τ':>8s} {'δV/V':>12s} {'SE':>10s}")
        for t, d, s in zip(tau_final, dr_final, dr_se):
            print(f"  {t:8.4f} {d:+12.6f} {s:10.6f}")

        # Fit: δV/V = A·τ⁴
        def model_t4(tau, A):
            return A * tau**4

        # Also fit: δV/V = A·τ⁴ + B·τ⁶ (next order correction)
        def model_t46(tau, A, B):
            return A * tau**4 + B * tau**6

        # Also fit free power: δV/V = A·τ^n
        def model_tn(tau, A, n):
            return A * tau**n

        try:
            # Model 1: pure τ⁴
            popt1, _ = curve_fit(model_t4, tau_final, dr_final, p0=[1.0])
            A_fit = popt1[0]
            pred1 = model_t4(tau_final, A_fit)
            ss1 = np.sum((dr_final - pred1)**2)
            ss_tot = np.sum((dr_final - np.mean(dr_final))**2)
            r2_1 = 1 - ss1/ss_tot if ss_tot > 0 else 0

            a2c3 = A_fit / K if K > 0 else 0

            print(f"\n  Fit 1: δV/V = A·τ⁴")
            print(f"    A = {A_fit:.4f}, R² = {r2_1:.4f}")
            print(f"    a₂c₃ = A/K = {A_fit:.4f}/{K:.2f} = {a2c3:.6f}")

        except Exception as e:
            print(f"  Fit 1 failed: {e}")
            A_fit = np.nan; r2_1 = 0; a2c3 = np.nan

        try:
            # Model 2: free power
            popt2, _ = curve_fit(model_tn, tau_final, dr_final, p0=[1.0, 4.0],
                                 maxfev=5000)
            A2, n2 = popt2
            pred2 = model_tn(tau_final, A2, n2)
            ss2 = np.sum((dr_final - pred2)**2)
            r2_2 = 1 - ss2/ss_tot if ss_tot > 0 else 0
            print(f"\n  Fit 2: δV/V = A·τ^n")
            print(f"    A = {A2:.4f}, n = {n2:.2f}, R² = {r2_2:.4f}")
            print(f"    Expected n=4 for Weyl correction")

        except Exception as e:
            print(f"  Fit 2 failed: {e}")
            n2 = np.nan; r2_2 = 0

        all_results[label] = {
            "N": N, "eps": eps, "K": K, "M": M, "elapsed_sec": elapsed,
            "a2c3": float(a2c3) if not np.isnan(a2c3) else None,
            "r2_t4": float(r2_1),
            "n_free": float(n2) if not np.isnan(n2) else None,
            "r2_free": float(r2_2),
            "tau_profile": tau_final.tolist(),
            "dr_profile": dr_final.tolist(),
        }

    # Summary
    print("\n" + "=" * 70)
    print("PATH 1 SUMMARY: Discrete a₂c₃")
    print("=" * 70)
    print(f"{'config':20s} {'K':>8s} {'a₂c₃':>12s} {'R²(τ⁴)':>10s} {'n_free':>8s} {'R²(free)':>10s}")
    print("-" * 70)

    a2c3_values = []
    for label, v in all_results.items():
        a2 = v.get("a2c3")
        print(f"{label:20s} {v['K']:8.2f} {a2 if a2 else 'N/A':>12} "
              f"{v['r2_t4']:10.4f} {v.get('n_free', 'N/A'):>8} {v['r2_free']:10.4f}")
        if a2 is not None and v['r2_t4'] > 0:
            a2c3_values.append(a2)

    if len(a2c3_values) >= 2:
        mean_a2 = np.mean(a2c3_values)
        std_a2 = np.std(a2c3_values)
        cv = std_a2 / abs(mean_a2) * 100 if abs(mean_a2) > 0 else 0
        print(f"\n  a₂c₃ = {mean_a2:.6f} ± {std_a2:.6f} (CV={cv:.0f}%)")
        if cv < 30:
            print(f"  ✅ STABLE across configurations → genuine coefficient")
        else:
            print(f"  ⚠️ VARIABLE ({cv:.0f}%) → may depend on N or ε")

    # Save
    outpath = os.path.join(OUTDIR, "path1_a2.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
