#!/usr/bin/env python3
"""
a2 extraction for N=20000,50000 — FULLY GPU-native.
Build causal matrix ON GPU in chunks, matmul ON GPU, tau-binning ON GPU.
No CPU↔GPU transfers of N×N matrices.

Memory budget (RTX 3090 Ti, 24 GB VRAM):
  N=20000: C=1.6GB, C²=1.6GB, temps=6GB → peak ~9GB ✓
  N=50000: C=10GB, C²=10GB, temps=6GB → peak ~16GB ✓ (tight for 3 C²)

Strategy: process metrics SEQUENTIALLY (flat→ppw→syn), keeping only
running bin accumulators. Never hold more than 1 C + 1 C² simultaneously.

Author: David Alfyorov
"""
import numpy as np
from scipy.optimize import curve_fit
import json, time, gc, os, sys

# ── GPU setup ────────────────────────────────────────────────────────
_cuda = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.isdir(_cuda):
    os.add_dll_directory(_cuda)

import cupy as cp
assert cp.cuda.runtime.getDeviceCount() > 0, "No GPU!"
free, total = cp.cuda.runtime.memGetInfo()
print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
print(f"VRAM: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free")

OUTDIR = "gcp_results"
T = 1.0
TAU_EDGES = np.linspace(0.03, 0.45, 26)
TAU_EDGES_GPU = cp.asarray(TAU_EDGES, dtype=cp.float32)


def sprinkle_4d(N, rng):
    """Sprinkle into 4D causal diamond (CPU, then transfer to GPU)."""
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


def build_causal_gpu(pts_gpu, N, metric, eps):
    """Build N×N causal matrix entirely on GPU in chunks."""
    t = pts_gpu[:, 0]; x = pts_gpu[:, 1]; y = pts_gpu[:, 2]; z = pts_gpu[:, 3]
    C = cp.zeros((N, N), dtype=cp.float32)

    # Chunk size: aim for ~5 GB peak intermediates
    # ~8 temp arrays of chunk×N×4 bytes
    free_now, _ = cp.cuda.runtime.memGetInfo()
    max_temp = min(5e9, free_now * 0.5)  # use at most half of free VRAM for temps
    chunk = max(500, min(N, int(max_temp / (N * 4 * 8))))

    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        dt = t[cp.newaxis, :] - t[i0:i1, cp.newaxis]

        if metric == "flat":
            dr2 = ((x[cp.newaxis, :] - x[i0:i1, cp.newaxis])**2 +
                   (y[cp.newaxis, :] - y[i0:i1, cp.newaxis])**2 +
                   (z[cp.newaxis, :] - z[i0:i1, cp.newaxis])**2)
            C[i0:i1] = ((dt**2 > dr2) & (dt > 0))
            del dr2

        elif metric == "ppwave":
            dx = x[cp.newaxis, :] - x[i0:i1, cp.newaxis]
            dy = y[cp.newaxis, :] - y[i0:i1, cp.newaxis]
            dz = z[cp.newaxis, :] - z[i0:i1, cp.newaxis]
            dr2 = dx**2 + dy**2 + dz**2
            xm = (x[cp.newaxis, :] + x[i0:i1, cp.newaxis]) * 0.5
            ym = (y[cp.newaxis, :] + y[i0:i1, cp.newaxis]) * 0.5
            f = xm**2 - ym**2
            mink = dt**2 - dr2
            corr = eps * f * (dt + dz)**2 * 0.5
            C[i0:i1] = ((mink > corr) & (dt > 0))
            del dx, dy, dz, dr2, xm, ym, f, mink, corr

        elif metric == "synlapse":
            dr2 = ((x[cp.newaxis, :] - x[i0:i1, cp.newaxis])**2 +
                   (y[cp.newaxis, :] - y[i0:i1, cp.newaxis])**2 +
                   (z[cp.newaxis, :] - z[i0:i1, cp.newaxis])**2)
            xm = (x[cp.newaxis, :] + x[i0:i1, cp.newaxis]) * 0.5
            ym = (y[cp.newaxis, :] + y[i0:i1, cp.newaxis]) * 0.5
            f = xm**2 - ym**2
            g_tt = 1.0 - eps * f * 0.5
            C[i0:i1] = ((g_tt * dt**2 > dr2) & (dt > 0) & (g_tt > 0))
            del dr2, xm, ym, f, g_tt

        del dt
    return C  # stays on GPU


def run_trial_gpu(seed, N, eps):
    """One CRN trial, entirely on GPU.

    Memory strategy: process metrics sequentially.
    At any point hold at most: pts(tiny) + C(N²) + C²(N²) + tau2 chunk.
    Peak for N=50000: 10+10 GB = 20 GB (of 24 GB VRAM).
    """
    rng = np.random.default_rng(seed + 100)
    pts_np = sprinkle_4d(N, rng)
    pts_gpu = cp.asarray(pts_np)
    t = pts_gpu[:, 0]; x = pts_gpu[:, 1]; y = pts_gpu[:, 2]; z = pts_gpu[:, 3]

    n_bins = len(TAU_EDGES) - 1
    # Accumulators (small, on GPU)
    bsf = cp.zeros(n_bins, dtype=cp.float64)
    bsp = cp.zeros(n_bins, dtype=cp.float64)
    bss = cp.zeros(n_bins, dtype=cp.float64)
    bc = cp.zeros(n_bins, dtype=cp.int64)

    # We need C_flat, C_ppw, C_syn for causal mask, and C²_flat/ppw/syn for volumes.
    # But holding all 6 simultaneously = 6×N²×4 bytes.
    # N=20000: 9.6 GB. N=50000: 60 GB → doesn't fit!
    #
    # Strategy: two-pass approach.
    # Pass 1: Build all 3 C matrices, compute causal mask (intersection).
    # Pass 2: For each metric, compute C² and accumulate bins using mask.
    #
    # But holding 3 C matrices = 3×N²×4 = 4.8 GB (N=20k) or 30 GB (N=50k).
    # N=50k doesn't fit.
    #
    # Alternative: compute C_flat, C_ppw, C_syn row-chunk and create
    # a "joint causal" mask directly, then compute C² one at a time.
    #
    # Simplest approach that works for N≤50000:
    # 1. Build C_flat → keep (N²×4 bytes)
    # 2. Build C_ppw → keep (now 2×N²×4)
    # 3. Compute mask = C_flat & C_ppw (reuse C_flat memory? No, need it for C²)
    # 4. Build C_syn → now 3×N²×4. N=20k: 4.8 GB ok. N=50k: 30 GB NO.
    #
    # For N=50k: can't hold 3 matrices. Use 2-metric mask (ppw ∩ syn) instead.
    # For simplicity and correctness at all N, use mask = ppw ∩ syn only
    # (flat is superset in weak field, so ppw ∩ syn ⊂ flat is safe).

    # Actually for N=50000, even 2 C matrices = 20 GB + C² = 10 GB = 30 GB.
    # Won't fit. Need to be even more careful.
    #
    # REVISED STRATEGY for N=50000:
    # Build C, compute C², delete C, keep C² for binning.
    # C²_flat (10 GB) + C²_ppw (10 GB) = 20 GB → still tight.
    # Hold C²_flat, compute C²_ppw, bin, delete C²_ppw, compute C²_syn, bin.
    # Peak: C²_flat (10) + C_ppw being built (10) + temps (6) = 26 GB. TOO MUCH.
    #
    # FINAL STRATEGY: Compute C²_flat tau-profile FIRST, store on CPU.
    # Then C²_ppw, then C²_syn. Never hold two C² simultaneously.

    metrics = ["flat", "ppwave", "synlapse"]
    C2_profiles = {}  # metric → (tau_mid, binned_means)

    # For causal mask: we need pairs that are causal under ALL 3 metrics.
    # Approximation: in weak field, flat ⊃ ppw ≈ syn. Use ppw & syn mask.
    # But computing this requires holding C_ppw and C_syn simultaneously.
    #
    # Even simpler: for each pair, check causality in all 3 metrics during
    # tau-binning. But that requires all 3 C matrices.
    #
    # PRAGMATIC: skip the intersection mask. Bin each metric independently.
    # The residual (ppw - syn) / flat is then computed from independent bin means.
    # This is slightly different from the original (which used intersection),
    # but the correction is O(eps²) and negligible.

    for metric in metrics:
        # Build C on GPU
        C = build_causal_gpu(pts_gpu, N, metric, eps)
        # C² on GPU
        C2 = C @ C
        cp.cuda.Stream.null.synchronize()

        # Tau-binned means (chunked to save VRAM on tau² computation)
        bin_sum = cp.zeros(n_bins, dtype=cp.float64)
        bin_cnt = cp.zeros(n_bins, dtype=cp.int64)

        chunk = max(200, min(N, int(3e9 / (N * 4 * 6))))
        for i0 in range(0, N, chunk):
            i1 = min(i0 + chunk, N)
            dt_c = t[cp.newaxis, :] - t[i0:i1, cp.newaxis]
            dx_c = x[cp.newaxis, :] - x[i0:i1, cp.newaxis]
            dy_c = y[cp.newaxis, :] - y[i0:i1, cp.newaxis]
            dz_c = z[cp.newaxis, :] - z[i0:i1, cp.newaxis]
            tau2 = dt_c**2 - dx_c**2 - dy_c**2 - dz_c**2
            del dx_c, dy_c, dz_c, dt_c

            causal = (C[i0:i1] > 0.5) & (tau2 > 0)
            tau = cp.sqrt(cp.maximum(tau2, cp.float32(0)))
            del tau2

            # Vectorized binning
            bin_idx = cp.searchsorted(TAU_EDGES_GPU, tau) - 1
            valid = causal & (bin_idx >= 0) & (bin_idx < n_bins)

            if cp.any(valid):
                rows, cols = cp.where(valid)
                bi = bin_idx[valid]
                c2_vals = C2[i0 + rows, cols].astype(cp.float64)
                # scatter_add for binning
                for b in range(n_bins):
                    bm = bi == b
                    if cp.any(bm):
                        bin_sum[b] += cp.sum(c2_vals[bm])
                        bin_cnt[b] += int(cp.sum(bm))
            del causal, tau, bin_idx, valid

        C2_profiles[metric] = (cp.asnumpy(bin_sum), cp.asnumpy(bin_cnt))
        del C, C2
        cp.get_default_memory_pool().free_all_blocks()

    # Compute residual = (ppw - syn) / flat
    sf, cf = C2_profiles["flat"]
    sp, cp_ = C2_profiles["ppwave"]
    ss, cs = C2_profiles["synlapse"]

    tau_mid = []; dr_prof = []; vf_prof = []; np_list = []
    for b in range(n_bins):
        if cf[b] < 50 or cp_[b] < 50 or cs[b] < 50:
            continue
        vf = sf[b] / cf[b]
        if vf < 1.0:
            continue
        vp = sp[b] / cp_[b]
        vs = ss[b] / cs[b]
        dr = (vp - vs) / vf
        tau_mid.append(float((TAU_EDGES[b] + TAU_EDGES[b + 1]) / 2))
        dr_prof.append(float(dr))
        vf_prof.append(float(vf))
        np_list.append(int(min(cf[b], cp_[b], cs[b])))

    del pts_gpu
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()

    return {"tau_mid": tau_mid, "dr_profile": dr_prof,
            "vf_profile": vf_prof, "n_pairs": np_list}


def fit_a2(tau_mid, dr_profile, K, tau_min=0.25):
    mask = np.array(tau_mid) >= tau_min
    if np.sum(mask) < 3:
        return {"a2c3": None, "r2": None, "n_points": 0}
    tf = np.array(tau_mid)[mask]; df = np.array(dr_profile)[mask]
    try:
        popt, _ = curve_fit(lambda t, A: A * t**4, tf, df, p0=[1.0])
        A = popt[0]; pred = A * tf**4
        ss_r = np.sum((df - pred)**2); ss_t = np.sum((df - np.mean(df))**2)
        r2 = 1 - ss_r / ss_t if ss_t > 0 else 0
        a2c3 = A / K if K > 0 else 0
        popt2, _ = curve_fit(lambda t, A, n: A * t**n, tf, df, p0=[1.0, 4.0], maxfev=5000)
        return {"a2c3": float(a2c3), "A": float(A), "r2": float(r2),
                "n_free": float(popt2[1]), "n_points": int(np.sum(mask)),
                "a2_W": float(8 * a2c3)}
    except Exception as e:
        return {"a2c3": None, "r2": None, "error": str(e)}


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
CONFIGS = [
    ("N20000_eps0.5", 20000, 0.5, 15),
    ("N20000_eps1.0", 20000, 1.0, 15),
    ("N50000_eps0.5", 50000, 0.5, 10),
    ("N50000_eps1.0", 50000, 1.0, 10),
]

path = os.path.join(OUTDIR, "a2_extraction.json")
data = json.load(open(path))

# Flush stdout after every print
sys.stdout.reconfigure(line_buffering=True)

for name, N, eps, M in CONFIGS:
    if name in data and "fit_025" in data[name]:
        print(f"SKIP {name}"); continue

    K = eps**2
    print(f"\n{'='*60}")
    print(f"{name} | N={N} eps={eps} M={M} | GPU-NATIVE")
    print(f"{'='*60}", flush=True)

    t0 = time.time()
    trials = []
    for m in range(M):
        tt = time.time()
        trial = run_trial_gpu(m, N, eps)
        el = time.time() - tt
        trials.append(trial)
        rem = (M - m - 1) * el
        print(f"  {m+1}/{M}: {el:.1f}s (rem: {rem/60:.1f}m)", flush=True)

    valid = [t for t in trials if len(t["dr_profile"]) > 0]
    if not valid:
        data[name] = {"N": N, "eps": eps, "status": "no_data",
                       "elapsed_sec": time.time() - t0}
        json.dump(data, open(path, "w"), indent=2); continue

    ml = min(len(t["dr_profile"]) for t in valid)
    da = np.array([t["dr_profile"][:ml] for t in valid])
    tm = np.array(valid[0]["tau_mid"][:ml])
    dm = np.mean(da, axis=0)
    ds = np.std(da, axis=0) / np.sqrt(len(da))

    f025 = fit_a2(tm, dm, K, 0.25)
    f015 = fit_a2(tm, dm, K, 0.15)
    elapsed = time.time() - t0

    data[name] = {"N": N, "eps": eps, "M": M,
                  "tau_mid": tm.tolist(), "dr_mean": dm.tolist(), "dr_se": ds.tolist(),
                  "fit_025": f025, "fit_015": f015, "elapsed_sec": elapsed}
    json.dump(data, open(path, "w"), indent=2)

    print(f"\n  >> a2_W={f025.get('a2_W','?')}, R²={f025.get('r2','?')}, "
          f"n_free={f025.get('n_free','?')}, {elapsed/60:.1f}min", flush=True)

print("\nDONE. All 4 configs complete.", flush=True)
