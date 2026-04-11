"""
FND-1 GPU Calibration — Measure per-step pipeline speed.

Runs 1 sprinkling at each test N, times every step individually.
Detects GPU (CuPy) and compares CPU vs GPU eigvalsh.
Estimates total time for full research program.

Works on: local (3090 Ti), GCP (PRO 6000), Vast.ai (H200), etc.

Run:
    python analysis/scripts/fnd1_calibrate_gpu.py
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# GPU detection (safe — no CUDA init if CuPy missing)
# ---------------------------------------------------------------------------
GPU_OK = False
GPU_NAME = "none"
GPU_VRAM_GB = 0.0

try:
    import cupy as cp

    props = cp.cuda.runtime.getDeviceProperties(0)
    GPU_NAME = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    GPU_VRAM_GB = round(cp.cuda.Device(0).mem_info[1] / 1e9, 1)
    GPU_OK = True
except Exception:
    cp = None

RAM_GB = 0.0
try:
    import psutil
    RAM_GB = round(psutil.virtual_memory().total / 1e9, 1)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Pipeline functions (self-contained)
# ---------------------------------------------------------------------------

def sprinkle_4d_flat(N: int, T: float, rng) -> np.ndarray:
    pts = np.empty((N, 4))
    count = 0
    half = T / 2.0
    while count < N:
        batch = max(N - count, 1000) * 10
        cands = rng.uniform(-half, half, size=(batch, 4))
        r = np.sqrt(cands[:, 1] ** 2 + cands[:, 2] ** 2 + cands[:, 3] ** 2)
        valid = cands[np.abs(cands[:, 0]) + r < half]
        n_take = min(len(valid), N - count)
        pts[count : count + n_take] = valid[:n_take]
        count += n_take
    return pts[np.argsort(pts[:, 0])]


def build_causal_matrix(pts: np.ndarray, eps: float) -> np.ndarray:
    """Memory-optimised dense causal matrix for pp-wave quadrupole."""
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]

    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx ** 2
    dr2 += dy ** 2
    dr2 += dz ** 2
    del dx, dy  # keep dt, dz, dr2

    if abs(eps) > 1e-12:
        xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
        np.square(xm, out=xm)
        ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
        np.square(ym, out=ym)
        xm -= ym  # xm = x_m^2 - y_m^2 = f_mid
        del ym
        f_mid = xm

        mink = dt ** 2
        mink -= dr2
        del dr2
        corr = dt + dz
        np.square(corr, out=corr)
        corr *= f_mid
        corr *= eps / 2.0
        del f_mid
        C = ((mink > corr) & (dt > 0)).astype(np.float64)
        del mink, corr
    else:
        mink = dt ** 2
        mink -= dr2
        del dr2
        C = ((mink > 0) & (dt > 0)).astype(np.float64)
        del mink

    del dt, dz
    return C


def build_L_and_commutator(C: np.ndarray, N: int, T: float):
    """Build BD operator L and dense commutator. Returns (comm_dense, nnz_L)."""
    V = np.pi * T ** 4 / 24.0
    rho = N / V
    scale = np.sqrt(rho)

    C_sp = sp.csr_matrix(C)
    n_sp = C_sp @ C_sp
    n_arr = n_sp.toarray()
    del C_sp, n_sp

    past = C.T
    n_past = n_arr.T
    del C, n_arr
    gc.collect()

    n_int = np.rint(n_past).astype(np.int64)
    causal_mask = past > 0.5
    del past

    L = np.zeros((N, N), dtype=np.float64)
    L[causal_mask & (n_int == 0)] = 4.0 * scale
    L[causal_mask & (n_int == 1)] = -36.0 * scale
    L[causal_mask & (n_int == 2)] = 64.0 * scale
    L[causal_mask & (n_int == 3)] = -32.0 * scale
    del causal_mask, n_int, n_past
    gc.collect()

    L_sp = sp.csr_matrix(L)
    nnz_L = L_sp.nnz
    del L

    comm_sp = (L_sp.T @ L_sp - L_sp @ L_sp.T) / 2.0
    comm_sp = (comm_sp + comm_sp.T) / 2.0
    del L_sp

    comm_dense = comm_sp.toarray()
    del comm_sp
    gc.collect()

    return comm_dense, nnz_L


def eigvalsh_gpu(comm: np.ndarray):
    """GPU eigvalsh via CuPy. Returns (eigenvalues, True) or (None, False)."""
    try:
        comm_gpu = cp.asarray(comm)
        evals = cp.linalg.eigvalsh(comm_gpu)
        cp.cuda.Stream.null.synchronize()
        result = cp.asnumpy(evals)
        del comm_gpu, evals
        cp.get_default_memory_pool().free_all_blocks()
        return result, True
    except Exception as e:
        if cp is not None:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
        return None, False


def eigvalsh_cpu(comm: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(comm)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate_one_N(N: int, eps: float = 5.0, T: float = 1.0) -> dict:
    """Time each pipeline step for a single N. Returns timing dict."""
    res = {"N": N}
    rng = np.random.default_rng(42)

    # 1. Sprinkle
    t0 = time.perf_counter()
    pts = sprinkle_4d_flat(N, T, rng)
    res["sprinkle"] = round(time.perf_counter() - t0, 3)

    # 2. Causal matrix
    t0 = time.perf_counter()
    C = build_causal_matrix(pts, eps)
    res["causal"] = round(time.perf_counter() - t0, 3)
    res["mem_C_GB"] = round(C.nbytes / 1e9, 2)
    del pts

    # 3. L + commutator
    t0 = time.perf_counter()
    comm, nnz_L = build_L_and_commutator(C, N, T)
    res["L_and_comm"] = round(time.perf_counter() - t0, 3)
    res["mem_comm_GB"] = round(comm.nbytes / 1e9, 2)
    res["nnz_L"] = nnz_L

    # 4. eigvalsh — GPU
    if GPU_OK:
        needed = comm.nbytes * 2.5 / 1e9  # matrix + workspace
        if needed < GPU_VRAM_GB * 0.95:
            t0 = time.perf_counter()
            evals_result, gpu_worked = eigvalsh_gpu(comm)
            elapsed = round(time.perf_counter() - t0, 3)
            if gpu_worked:
                res["eigvalsh_gpu"] = elapsed
            else:
                res["eigvalsh_gpu"] = None
                res["eigvalsh_gpu_skip"] = "cusolver failed (DLL or driver)"
        else:
            res["eigvalsh_gpu"] = None
            res["eigvalsh_gpu_skip"] = f"need {needed:.1f} GB > {GPU_VRAM_GB} GB VRAM"

    # 5. eigvalsh — CPU
    t0 = time.perf_counter()
    evals_cpu = eigvalsh_cpu(comm)
    res["eigvalsh_cpu"] = round(time.perf_counter() - t0, 3)

    # Total
    gpu_time = res.get("eigvalsh_gpu")
    cpu_parts = res["sprinkle"] + res["causal"] + res["L_and_comm"]
    res["total_gpu"] = round(cpu_parts + gpu_time, 1) if gpu_time else None
    res["total_cpu"] = round(cpu_parts + res["eigvalsh_cpu"], 1)

    del comm, evals_cpu
    gc.collect()
    if GPU_OK:
        cp.get_default_memory_pool().free_all_blocks()

    return res


def estimate_program(results: list[dict]):
    """Estimate full research program time from calibration data."""
    print("\n" + "=" * 70)
    print("ESTIMATED RESEARCH PROGRAM")
    print("=" * 70)

    # Find best measured times for key N values
    by_N = {r["N"]: r for r in results}

    program = [
        ("N=30k dense, pp-wave, 400 spr", 30000, 400),
        ("N=50k dense, pp-wave, 400 spr", 50000, 400),
        ("N=30k dense, conformally flat", 30000, 400),
        ("N=30k dense, FLRW-type", 30000, 400),
        ("N=200k matrix-free, 400 spr", 200000, 400),
    ]

    total_hours = 0.0
    print(f"\n  {'Task':<40} {'per spr':>8} {'total':>8}")
    print(f"  {'─' * 40} {'─' * 8} {'─' * 8}")

    for label, target_N, n_spr in program:
        # Extrapolate from closest measured N
        measured_Ns = sorted(by_N.keys())
        closest = min(measured_Ns, key=lambda n: abs(n - target_N))
        r = by_N[closest]

        if target_N == 200000:
            # Matrix-free: only L sparse + eigsh(k=1). ~seconds.
            per_spr = 5.0  # conservative estimate
        else:
            t_ref = r.get("total_gpu") or r["total_cpu"]
            # Scale as ~N^2.5 (causal=N², eigvalsh=N³, mix)
            per_spr = t_ref * (target_N / closest) ** 2.5

        hours = per_spr * n_spr / 3600
        total_hours += hours
        per_str = f"{per_spr:.0f}s" if per_spr < 300 else f"{per_spr / 60:.1f}m"
        print(f"  {label:<40} {per_str:>8} {hours:>7.1f}h")

    print(f"  {'─' * 40} {'─' * 8} {'─' * 8}")
    print(f"  {'TOTAL':<40} {'':>8} {total_hours:>7.1f}h")

    costs = {"$1.50/hr (spot)": 1.50, "$4.50/hr (on-demand)": 4.50}
    print()
    for label, rate in costs.items():
        cost = total_hours * rate
        print(f"  {label}: ${cost:.0f}  (from $160 credits: {'OK' if cost <= 160 else 'OVER'})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("FND-1 PIPELINE CALIBRATION")
    print("=" * 70)
    print(f"GPU:  {GPU_NAME} ({GPU_VRAM_GB:.1f} GB VRAM)" if GPU_OK else "GPU:  none")
    print(f"RAM:  {RAM_GB:.1f} GB")
    print(f"CPU:  {os.cpu_count()} cores")
    print()

    # Determine safe N values
    N_TEST = [5000, 10000]

    avail_ram = RAM_GB * 0.7  # leave 30% for OS
    # N=20k needs ~5 × 3.2 GB = 16 GB peak for causal matrix (optimised)
    if avail_ram > 20:
        N_TEST.append(20000)
    # N=30k needs ~5 × 7.2 GB = 36 GB peak
    if avail_ram > 40:
        N_TEST.append(30000)
    # N=50k needs ~5 × 20 GB = 100 GB peak — only on high-RAM machines
    if avail_ram > 110:
        N_TEST.append(50000)

    print(f"Test N: {N_TEST}")
    print(f"(Higher N skipped if insufficient RAM)")
    print()

    results = []
    t_total = time.perf_counter()

    for N in N_TEST:
        print(f"─── N = {N:,} ───")
        try:
            r = calibrate_one_N(N)
            results.append(r)

            line = (
                f"  sprinkle={r['sprinkle']:.1f}s  "
                f"causal={r['causal']:.1f}s  "
                f"L+comm={r['L_and_comm']:.1f}s  "
            )
            if r.get("eigvalsh_gpu") is not None:
                line += f"eigvalsh_GPU={r['eigvalsh_gpu']:.1f}s  "
            line += f"eigvalsh_CPU={r['eigvalsh_cpu']:.1f}s"
            print(line)

            if r.get("total_gpu"):
                print(f"  TOTAL: GPU={r['total_gpu']:.1f}s  CPU={r['total_cpu']:.1f}s  "
                      f"speedup={r['total_cpu'] / r['total_gpu']:.1f}x")
            else:
                skip = r.get("eigvalsh_gpu_skip", "")
                print(f"  TOTAL: CPU={r['total_cpu']:.1f}s  (GPU: {skip})")

            print(f"  Memory: C={r['mem_C_GB']:.1f} GB  comm={r['mem_comm_GB']:.1f} GB")
            print()
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            print()

    wall = time.perf_counter() - t_total

    if results:
        # Summary table
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\n  {'N':>7}  {'GPU eigvalsh':>14}  {'CPU eigvalsh':>14}  {'Total (GPU)':>12}  {'comm GB':>8}")
        print(f"  {'─' * 7}  {'─' * 14}  {'─' * 14}  {'─' * 12}  {'─' * 8}")
        for r in results:
            gpu_str = f"{r['eigvalsh_gpu']:.1f}s" if r.get("eigvalsh_gpu") else "—"
            tot_str = f"{r['total_gpu']:.1f}s" if r.get("total_gpu") else "—"
            print(f"  {r['N']:>7,}  {gpu_str:>14}  {r['eigvalsh_cpu']:.1f}s{' ' * 8}  {tot_str:>12}  {r['mem_comm_GB']:>7.1f}")

        estimate_program(results)

    # Save
    out = {
        "hardware": {"gpu": GPU_NAME, "vram_gb": GPU_VRAM_GB, "ram_gb": RAM_GB, "cpu_cores": os.cpu_count()},
        "results": results,
        "wall_time_s": round(wall, 1),
    }
    out_dir = Path(__file__).resolve().parent.parent.parent / "speculative" / "numerics" / "ensemble_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "calibrate_gpu.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path.name}")
    print(f"Calibration time: {wall:.0f}s ({wall / 60:.1f} min)")


if __name__ == "__main__":
    main()
