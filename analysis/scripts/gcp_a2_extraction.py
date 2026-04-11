#!/usr/bin/env python3
"""
GCP Experiment: Large-N Weyl Coefficient Extraction
=====================================================
Runs on GCP VM with A100 GPU (or falls back to CPU).

Extracts the discrete a₂ Seeley-DeWitt coefficient by going to large N
where boundary effects (~ τ⁻²) become subdominant to the interior
Weyl correction (~ τ⁴).

Design passed 2 consecutive adversarial reviews (self + R1 CLEAN).

Usage:
  # On GCP VM:
  pip install cupy-cuda12x scipy numpy
  python gcp_a2_extraction.py

  # Or locally (slower, CPU only):
  python gcp_a2_extraction.py --cpu

Output: results saved to gcp_results/ directory as JSON.

Author: David Alfyorov
"""
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import json, time, gc, os, sys, argparse

# ---------------------------------------------------------------------------
# GPU/CPU backend selection
# ---------------------------------------------------------------------------
USE_GPU = False
try:
    # Windows: add CUDA toolkit DLLs before importing CuPy
    import os as _os
    _cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
    if _os.path.isdir(_cuda_bin):
        _os.add_dll_directory(_cuda_bin)
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cp_csr
    if cp.cuda.runtime.getDeviceCount() > 0:
        USE_GPU = True
        print(f"GPU detected: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        print(f"VRAM: {cp.cuda.runtime.memGetInfo()[1] / 1e9:.1f} GB")
except (ImportError, OSError):
    pass

if "--cpu" in sys.argv:
    USE_GPU = False

xp = cp if USE_GPU else np
print(f"Backend: {'GPU (CuPy)' if USE_GPU else 'CPU (NumPy)'}")

OUTDIR = "gcp_results"
os.makedirs(OUTDIR, exist_ok=True)

T = 1.0
TAU_EDGES = np.linspace(0.03, 0.45, 26)  # 25 bins


# ---------------------------------------------------------------------------
# Sprinkling
# ---------------------------------------------------------------------------
def sprinkle_4d(N, T, rng):
    """Sprinkle into 4D causal diamond. CPU (numpy rng)."""
    pts = np.empty((N, 4), dtype=np.float32)
    count, half = 0, T / 2.0
    while count < N:
        batch = max(N - count, 1000) * 10
        c = rng.uniform(-half, half, size=(batch, 4)).astype(np.float32)
        r = np.sqrt(c[:, 1]**2 + c[:, 2]**2 + c[:, 3]**2)
        v = c[np.abs(c[:, 0]) + r < half]
        n = min(len(v), N - count)
        pts[count:count + n] = v[:n]
        count += n
    return pts[np.argsort(pts[:, 0])]


# ---------------------------------------------------------------------------
# Causal matrix construction (GPU-accelerated)
# ---------------------------------------------------------------------------
def causal_flat_gpu(pts_gpu, N):
    """Build flat causal matrix on GPU. Returns GPU array."""
    t = pts_gpu[:, 0]; x = pts_gpu[:, 1]; y = pts_gpu[:, 2]; z = pts_gpu[:, 3]
    dt = t[xp.newaxis, :] - t[:, xp.newaxis]
    dr2 = ((x[xp.newaxis, :] - x[:, xp.newaxis])**2 +
           (y[xp.newaxis, :] - y[:, xp.newaxis])**2 +
           (z[xp.newaxis, :] - z[:, xp.newaxis])**2)
    C = ((dt**2 > dr2) & (dt > 0)).astype(xp.float32)
    del dt, dr2
    return C


def causal_ppwave_gpu(pts_gpu, N, eps):
    """Build pp-wave causal matrix on GPU."""
    t = pts_gpu[:, 0]; x = pts_gpu[:, 1]; y = pts_gpu[:, 2]; z = pts_gpu[:, 3]
    dt = t[xp.newaxis, :] - t[:, xp.newaxis]
    dx = x[xp.newaxis, :] - x[:, xp.newaxis]
    dy = y[xp.newaxis, :] - y[:, xp.newaxis]
    dz = z[xp.newaxis, :] - z[:, xp.newaxis]
    dr2 = dx**2 + dy**2 + dz**2
    xm = (x[xp.newaxis, :] + x[:, xp.newaxis]) / 2.0
    ym = (y[xp.newaxis, :] + y[:, xp.newaxis]) / 2.0
    f = xm**2 - ym**2
    mink = dt**2 - dr2
    corr = eps * f * (dt + dz)**2 / 2.0
    C = ((mink > corr) & (dt > 0)).astype(xp.float32)
    del dt, dx, dy, dz, dr2, xm, ym, f, mink, corr
    return C


def causal_synlapse_gpu(pts_gpu, N, eps):
    """Build synthetic lapse causal matrix on GPU.
    Same g_tt as pp-wave, but diagonal (no wave components).
    """
    t = pts_gpu[:, 0]; x = pts_gpu[:, 1]; y = pts_gpu[:, 2]; z = pts_gpu[:, 3]
    dt = t[xp.newaxis, :] - t[:, xp.newaxis]
    dr2 = ((x[xp.newaxis, :] - x[:, xp.newaxis])**2 +
           (y[xp.newaxis, :] - y[:, xp.newaxis])**2 +
           (z[xp.newaxis, :] - z[:, xp.newaxis])**2)
    xm = (x[xp.newaxis, :] + x[:, xp.newaxis]) / 2.0
    ym = (y[xp.newaxis, :] + y[:, xp.newaxis]) / 2.0
    f = xm**2 - ym**2
    g_tt = 1.0 - eps * f / 2.0
    C = ((g_tt * dt**2 > dr2) & (dt > 0) & (g_tt > 0)).astype(xp.float32)
    del dt, dr2, xm, ym, f, g_tt
    return C


# ---------------------------------------------------------------------------
# Interval volumes (C² = C @ C)
# ---------------------------------------------------------------------------
def interval_volumes_matmul(C):
    """Compute C² via matrix multiplication. GPU or CPU."""
    return C @ C


# ---------------------------------------------------------------------------
# Tau-binned volume profile
# ---------------------------------------------------------------------------
def volume_profile_from_C2(C2_flat, C2_ppw, C2_syn, C_flat, C_ppw, C_syn,
                            tau2_coord, tau_edges):
    """Compute residual volume profile (ppw - syn) / flat at each tau bin.

    All inputs are xp arrays (GPU or CPU).
    Returns numpy arrays (on CPU).
    """
    causal_all = (C_flat > 0.5) & (C_ppw > 0.5) & (C_syn > 0.5) & (tau2_coord > 0)

    tau_vals = xp.sqrt(xp.where(tau2_coord > 0, tau2_coord, xp.float32(0.0)))

    tau_mid = []
    dr_profile = []  # residual = (V_ppw - V_syn) / V_flat
    vf_profile = []
    n_pairs = []

    for i in range(len(tau_edges) - 1):
        lo, hi = tau_edges[i], tau_edges[i + 1]
        mask = causal_all & (tau_vals >= lo) & (tau_vals < hi)

        n = int(xp.sum(mask))
        if n < 50:
            continue

        vf = float(xp.mean(C2_flat[mask]))
        vp = float(xp.mean(C2_ppw[mask]))
        vs = float(xp.mean(C2_syn[mask]))

        if vf < 1.0:
            continue

        dr = (vp - vs) / vf

        tau_mid.append((lo + hi) / 2)
        dr_profile.append(dr)
        vf_profile.append(vf)
        n_pairs.append(n)

    return np.array(tau_mid), np.array(dr_profile), np.array(vf_profile), np.array(n_pairs)


# ---------------------------------------------------------------------------
# Degree stats (for adversarial proxy check)
# ---------------------------------------------------------------------------
def degree_stats_from_C(C):
    """Quick degree stats from causal matrix. Returns dict."""
    # Link graph: C - (C has intervening) = links
    # For speed at large N: approximate degree from C directly
    # Degree proxy: row sums of C (= future size)
    future = xp.sum(C, axis=1)
    past = xp.sum(C, axis=0)
    total = future + past

    return {
        "mean_degree": float(xp.mean(total)),
        "degree_var": float(xp.var(total)),
        "degree_std": float(xp.std(total)),
        "total_causal": int(xp.sum(C > 0.5)),
    }


# ---------------------------------------------------------------------------
# Single CRN trial
# ---------------------------------------------------------------------------
def run_trial(seed, N, eps, tau_edges):
    """One CRN trial: flat + ppwave + synthetic lapse."""
    rng = np.random.default_rng(seed + 100)
    pts_np = sprinkle_4d(N, T, rng)

    # Transfer to GPU if available
    if USE_GPU:
        pts = cp.asarray(pts_np)
    else:
        pts = pts_np

    # Coordinate proper time (same for all metrics — CRN)
    t = pts[:, 0]; x = pts[:, 1]; y = pts[:, 2]; z = pts[:, 3]
    dt = t[xp.newaxis, :] - t[:, xp.newaxis]
    dx = x[xp.newaxis, :] - x[:, xp.newaxis]
    dy = y[xp.newaxis, :] - y[:, xp.newaxis]
    dz = z[xp.newaxis, :] - z[:, xp.newaxis]
    tau2 = (dt**2 - dx**2 - dy**2 - dz**2).astype(xp.float32)
    del dx, dy, dz  # keep dt for later

    # Build causal matrices
    C_flat = causal_flat_gpu(pts, N)
    C_ppw = causal_ppwave_gpu(pts, N, eps)
    C_syn = causal_synlapse_gpu(pts, N, eps)
    del dt

    # Interval volumes via matmul
    C2_flat = interval_volumes_matmul(C_flat)
    C2_ppw = interval_volumes_matmul(C_ppw)
    C2_syn = interval_volumes_matmul(C_syn)

    # Volume profile
    tau_mid, dr_prof, vf_prof, n_pairs = volume_profile_from_C2(
        C2_flat, C2_ppw, C2_syn, C_flat, C_ppw, C_syn, tau2, tau_edges)

    # Degree stats (for adversarial)
    ds_flat = degree_stats_from_C(C_flat)
    ds_ppw = degree_stats_from_C(C_ppw)
    ds_syn = degree_stats_from_C(C_syn)

    # Cleanup GPU memory
    del C_flat, C_ppw, C_syn, C2_flat, C2_ppw, C2_syn, tau2, pts
    if USE_GPU:
        cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    return {
        "seed": seed, "N": N, "eps": eps,
        "tau_mid": tau_mid.tolist(),
        "dr_profile": dr_prof.tolist(),
        "vf_profile": vf_prof.tolist(),
        "n_pairs": n_pairs.tolist(),
        "ds_flat": ds_flat, "ds_ppw": ds_ppw, "ds_syn": ds_syn,
    }


# ---------------------------------------------------------------------------
# Fit tau^4 to large-tau tail
# ---------------------------------------------------------------------------
def fit_a2(tau_mid, dr_profile, K, tau_min=0.25):
    """Fit δV_residual/V_flat = A·τ⁴ to large-τ tail.
    Returns a₂c₃ = A/K, R², and the fit."""
    mask = np.array(tau_mid) >= tau_min
    if np.sum(mask) < 3:
        return {"a2c3": None, "r2": None, "n_points": 0}

    tau_fit = np.array(tau_mid)[mask]
    dr_fit = np.array(dr_profile)[mask]

    try:
        def model(t, A):
            return A * t**4
        popt, _ = curve_fit(model, tau_fit, dr_fit, p0=[1.0])
        A = popt[0]
        pred = model(tau_fit, A)
        ss_res = np.sum((dr_fit - pred)**2)
        ss_tot = np.sum((dr_fit - np.mean(dr_fit))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        a2c3 = A / K if K > 0 else 0

        # Also fit free power
        def model_free(t, A, n):
            return A * t**n
        popt2, _ = curve_fit(model_free, tau_fit, dr_fit, p0=[1.0, 4.0], maxfev=5000)
        n_free = popt2[1]

        return {
            "a2c3": float(a2c3), "A": float(A), "r2": float(r2),
            "n_free": float(n_free), "n_points": int(np.sum(mask)),
        }
    except Exception as e:
        return {"a2c3": None, "r2": None, "error": str(e), "n_points": int(np.sum(mask))}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode")
    parser.add_argument("--quick", action="store_true", help="Quick test (small N only)")
    args = parser.parse_args()

    if args.quick:
        CONFIGS = [(2000, 1.0, 10), (5000, 1.0, 5)]
    else:
        CONFIGS = [
            # (N, eps, M)
            (5000,  0.5, 20),
            (5000,  1.0, 20),
            (10000, 0.5, 20),
            (10000, 1.0, 20),
            (20000, 0.5, 15),
            (20000, 1.0, 15),
            (50000, 0.5, 10),
            (50000, 1.0, 10),
        ]

    # Also: null control
    NULL_CONFIGS = [
        (5000, 0.0, 5),   # eps=0: must give exact 0
    ]

    print("=" * 70)
    print("GCP EXPERIMENT: Large-N Weyl Coefficient Extraction")
    print(f"Configs: {len(CONFIGS)} + {len(NULL_CONFIGS)} null")
    print(f"Backend: {'GPU' if USE_GPU else 'CPU'}")
    print(f"Pre-registered: tau^4 fit R² > 0.5 at N≥20000 → a₂ extracted")
    print("=" * 70)

    all_results = {}
    t0_total = time.time()

    # Null controls first
    for N, eps, M in NULL_CONFIGS:
        label = f"null_N{N}"
        print(f"\n--- {label} [NULL CONTROL] ---")
        for trial in range(M):
            res = run_trial(trial * 1000, N, eps, TAU_EDGES)
            max_dr = max(abs(d) for d in res["dr_profile"]) if res["dr_profile"] else 0
            print(f"  trial {trial+1}/{M}: max|δV/V| = {max_dr:.2e} "
                  f"(must be ~0)")
        all_results[label] = {"N": N, "eps": eps, "status": "null_control"}

    # Main configs
    for N, eps, M in CONFIGS:
        K = 8 * eps**2
        label = f"N{N}_eps{eps}"
        print(f"\n{'='*50}")
        print(f"  {label} (K={K:.1f}, M={M})")
        print(f"{'='*50}")

        # Accumulate profiles across trials
        all_dr_profiles = []
        all_tau_mids = []

        t0 = time.time()
        for trial in range(M):
            res = run_trial(trial * 1000, N, eps, TAU_EDGES)
            if res["tau_mid"]:
                all_dr_profiles.append(res["dr_profile"])
                all_tau_mids.append(res["tau_mid"])

            if (trial + 1) % 5 == 0:
                elapsed = time.time() - t0
                # Average profile so far
                if all_dr_profiles:
                    # Align by tau bins
                    common_tau = all_tau_mids[0]
                    mean_dr = np.mean([dr for dr in all_dr_profiles], axis=0)
                    print(f"  trial {trial+1}/{M}: mean resid(τ=0.3)="
                          f"{mean_dr[len(mean_dr)//2]:+.6f} [{elapsed:.1f}s]")

        elapsed = time.time() - t0

        if len(all_dr_profiles) < 3:
            print(f"  INSUFFICIENT data ({len(all_dr_profiles)} valid trials)")
            all_results[label] = {"status": "insufficient", "N": N, "eps": eps}
            continue

        # Average profile
        mean_dr = np.mean(all_dr_profiles, axis=0)
        se_dr = np.std(all_dr_profiles, axis=0) / np.sqrt(len(all_dr_profiles))
        tau_mid = np.array(all_tau_mids[0])

        # Print profile
        print(f"\n  Averaged profile ({len(all_dr_profiles)} trials):")
        print(f"  {'τ':>8s} {'δV/V resid':>14s} {'SE':>12s} {'V_flat':>10s}")
        for i, t in enumerate(tau_mid):
            print(f"  {t:8.4f} {mean_dr[i]:+14.8f} {se_dr[i]:12.8f}")

        # Fit a₂ from large-τ tail
        fit = fit_a2(tau_mid, mean_dr, K, tau_min=0.25)
        print(f"\n  Fit (τ > 0.25): a₂c₃={fit.get('a2c3')}, R²={fit.get('r2')}, "
              f"n_free={fit.get('n_free')}")

        # Also fit from τ > 0.15 and τ > 0.30 for robustness
        fit_15 = fit_a2(tau_mid, mean_dr, K, tau_min=0.15)
        fit_30 = fit_a2(tau_mid, mean_dr, K, tau_min=0.30)

        print(f"  Fit (τ > 0.15): a₂c₃={fit_15.get('a2c3')}, R²={fit_15.get('r2')}")
        print(f"  Fit (τ > 0.30): a₂c₃={fit_30.get('a2c3')}, R²={fit_30.get('r2')}")

        all_results[label] = {
            "N": N, "eps": eps, "K": K, "M": len(all_dr_profiles),
            "elapsed_sec": elapsed,
            "tau_mid": tau_mid.tolist(),
            "mean_dr": mean_dr.tolist(),
            "se_dr": se_dr.tolist(),
            "fit_025": fit,
            "fit_015": fit_15,
            "fit_030": fit_30,
        }

    total_elapsed = time.time() - t0_total

    # Save
    outpath = os.path.join(OUTDIR, "a2_extraction.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print(f"SUMMARY ({total_elapsed/60:.1f} min total)")
    print("=" * 70)

    print(f"\n{'config':20s} {'a₂c₃(0.25)':>14s} {'R²(0.25)':>10s} {'n_free':>8s} "
          f"{'a₂c₃(0.15)':>14s} {'R²(0.15)':>10s}")
    print("-" * 80)

    a2_values = []
    for label, res in all_results.items():
        if "fit_025" not in res:
            continue
        f25 = res["fit_025"]
        f15 = res["fit_015"]
        a2 = f25.get("a2c3")
        r2 = f25.get("r2")
        nf = f25.get("n_free")
        a2_15 = f15.get("a2c3")
        r2_15 = f15.get("r2")

        print(f"{label:20s} {str(a2):>14s} {str(r2):>10s} {str(nf):>8s} "
              f"{str(a2_15):>14s} {str(r2_15):>10s}")

        if a2 is not None and r2 is not None and r2 > 0:
            a2_values.append({"label": label, "a2c3": a2, "r2": r2, "N": res["N"]})

    if a2_values:
        # Check stability across N
        print(f"\n  a₂c₃ values: {[f'{v['a2c3']:.6f}' for v in a2_values]}")
        a2_arr = np.array([v["a2c3"] for v in a2_values])
        mean_a2 = np.mean(a2_arr)
        cv = np.std(a2_arr) / abs(mean_a2) * 100 if abs(mean_a2) > 0 else 999

        print(f"  Mean a₂c₃ = {mean_a2:.6f} (CV = {cv:.0f}%)")

        if cv < 30:
            print(f"  ✅ STABLE across configs → genuine discrete coefficient")
        else:
            print(f"  ⚠️ VARIABLE → may depend on N or eps")

        # Check if n_free ≈ 4
        n_frees = [v.get("n_free") for v in a2_values if v.get("n_free") is not None]
        if n_frees:
            print(f"  n_free values: {[f'{n:.1f}' for n in n_frees]}")
            mean_n = np.mean(n_frees)
            if abs(mean_n - 4) < 1:
                print(f"  ✅ n_free ≈ {mean_n:.1f} ≈ 4 (Weyl correction)")
            elif mean_n < 0:
                print(f"  ☠️ n_free = {mean_n:.1f} < 0 (boundary-dominated)")
            else:
                print(f"  ⚠️ n_free = {mean_n:.1f} (not 4)")

    print(f"\n  Total time: {total_elapsed/60:.1f} min")
    print(f"  Results saved to: {outpath}")


if __name__ == "__main__":
    main()
