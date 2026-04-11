"""
FND-1: SJ Vacuum N=5000 on GPU (CuPy via WSL2).

GPU-accelerated eigendecomposition of iDelta (5000x5000 complex Hermitian).
CuPy cuSOLVER heevd: ~3s vs CPU LAPACK zheev: ~20s at N=5000.

The heavy part (eigendecomposition) runs on GPU. Everything else on CPU.

Run from Windows (calls WSL internally):
  python analysis/scripts/fnd1_sj_gpu.py
"""

from __future__ import annotations
import sys, time, json, subprocess
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_4d_experiment import (
    sprinkle_4d_flat, sprinkle_4d_ppwave,
    causal_matrix_4d, compute_layers_4d, bd_action_4d,
    build_link_graph, _ppwave_profile,
)
from fnd1_sj_vacuum import sj_observables, mediation_poly
from fnd1_experiment_registry import ExperimentMeta, save_experiment, RESULTS_DIR

N = 5000
M = 40
T = 1.0
SEED = 555
EPS_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5]


def build_sj_gpu(C, n_matrix, rho):
    """Build SJ Wightman function with GPU-accelerated eigendecomposition.

    Falls back to CPU if CuPy is not available.
    """
    N_pts = C.shape[0]
    a = np.sqrt(rho) / (2.0 * np.pi * np.sqrt(6.0))

    past = C.T
    n_past = n_matrix.T
    link_lower = ((past > 0) & (n_past == 0)).astype(np.float64)
    L_upper = link_lower.T

    Delta = a * (L_upper - L_upper.T)
    iDelta = 1j * Delta  # complex Hermitian

    # Try GPU eigendecomposition
    try:
        import cupy as cp
        iDelta_gpu = cp.asarray(iDelta)
        evals_gpu = cp.linalg.eigvalsh(iDelta_gpu)
        evals = cp.asnumpy(evals_gpu)
        del iDelta_gpu, evals_gpu
        cp.get_default_memory_pool().free_all_blocks()
    except (ImportError, Exception):
        # Fallback to CPU
        evals = np.linalg.eigvalsh(iDelta)

    pos_mask = evals > 1e-15
    pos_evals = evals[pos_mask]

    return pos_evals, evals


def worker(seed_int, eps):
    rng = np.random.default_rng(seed_int)
    if abs(eps) < 1e-12:
        pts = sprinkle_4d_flat(N, T, rng)
        C = causal_matrix_4d(pts, 0.0, "flat")
    else:
        pts = sprinkle_4d_ppwave(N, T, rng)
        C = causal_matrix_4d(pts, eps, "tidal")

    tc = float(np.sum(C))
    n_mat, N0, N1, N2, N3 = compute_layers_4d(C)
    V = np.pi * T**4 / 24.0
    rho = N / V
    bd = bd_action_4d(N, N0, N1, N2, N3)

    pos_evals, all_evals = build_sj_gpu(C, n_mat, rho)
    obs = sj_observables(pos_evals, all_evals, N)

    A_link = build_link_graph(C, n_mat)
    degrees = np.array(A_link.sum(axis=1)).ravel()
    deg_cv = float(np.std(degrees) / np.mean(degrees)) if np.mean(degrees) > 0 else 0

    return {"eps": eps, "total_causal": tc, "n_links": N0,
            "bd_action": float(bd), "deg_cv": deg_cv, **obs}


def main():
    t0 = time.perf_counter()
    print(f"FND-1 SJ N={N} (GPU): M={M} x {len(EPS_VALUES)} eps", flush=True)

    # Check GPU
    try:
        import cupy as cp
        print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}, "
              f"VRAM: {cp.cuda.runtime.memGetInfo()[1]/1e9:.0f} GB", flush=True)
        gpu_ok = True
    except Exception as e:
        print(f"GPU not available ({e}), using CPU fallback", flush=True)
        gpu_ok = False

    # Benchmark
    tb = time.perf_counter()
    worker(42, 0.0)
    per_task = time.perf_counter() - tb
    total_est = M * len(EPS_VALUES) * per_task
    print(f"Benchmark: {per_task:.1f}s/task ({'GPU' if gpu_ok else 'CPU'}), "
          f"total est: {total_est/60:.0f} min", flush=True)

    ss = np.random.SeedSequence(SEED)
    data = []

    for eps in EPS_VALUES:
        seeds = [int(cs.generate_state(1)[0]) for cs in ss.spawn(1)[0].spawn(M)]
        print(f"\n  eps={eps:+.2f}:", end=" ", flush=True)
        t1 = time.perf_counter()
        for i, s in enumerate(seeds):
            data.append(worker(s, eps))
            if (i + 1) % 10 == 0:
                print(f"{i+1}", end=" ", flush=True)
        d_eps = [d for d in data if abs(d["eps"] - eps) < 0.01]
        tc_m = np.mean([d["total_causal"] for d in d_eps])
        tr_m = np.mean([d["trace_W"] for d in d_eps])
        print(f"  ({time.perf_counter()-t1:.0f}s, TC={tc_m:.0f}, trace={tr_m:.0f})", flush=True)

    # Analysis
    eps_a = np.array([d["eps"] for d in data])
    tc_a = np.array([d["total_causal"] for d in data])
    nl_a = np.array([d["n_links"] for d in data])
    bd_a = np.array([d["bd_action"] for d in data])
    dcv_a = np.array([d["deg_cv"] for d in data])

    metrics = ["trace_W", "spectral_width", "trace_trunc", "lambda_median",
               "spectral_gap_ratio", "entropy_trunc", "ssee_proxy"]

    print(f"\n{'='*70}", flush=True)
    print(f"MEDIATION (N={N})", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'metric':>20} {'r_dir':>7} {'r|TC':>7} {'r|poly3':>7} {'p|poly3':>10}", flush=True)

    results = {}
    for metric in metrics:
        obs = np.array([d[metric] for d in data])
        if np.std(obs) < 1e-15:
            continue
        med = mediation_poly(eps_a, obs, tc_a, nl_a, bd_a)
        results[metric] = med
        print(f"  {metric:>20} {med['r_direct']:+7.3f} {med['r_tc']:+7.3f} "
              f"{med['r_poly3']:+7.3f} {med['p_poly3']:10.2e}", flush=True)

    # spectral_width with deg_cv
    sw = np.array([d["spectral_width"] for d in data])
    tc2 = tc_a**2; tc3 = tc_a**3
    X = np.column_stack([tc_a, tc2, tc3, nl_a, bd_a, dcv_a, np.ones(len(tc_a))])
    try:
        bx = np.linalg.lstsq(X, eps_a, rcond=None)[0]
        by = np.linalg.lstsq(X, sw, rcond=None)[0]
        xr = eps_a - X @ bx; yr = sw - X @ by
        r_sw_dcv, p_sw_dcv = stats.pearsonr(xr, yr)
    except:
        r_sw_dcv, p_sw_dcv = 0.0, 1.0
    print(f"\n  spectral_width + deg_cv: r={r_sw_dcv:+.4f}, p={p_sw_dcv:.2e}", flush=True)

    # Scaling
    print(f"\n{'='*70}", flush=True)
    print("SCALING: trace_W r|poly3", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  N=1000: -0.054", flush=True)
    print(f"  N=2000: -0.285", flush=True)
    print(f"  N=3000: -0.405", flush=True)
    tw = results.get("trace_W", {})
    print(f"  N=5000: {tw.get('r_poly3',0):+.3f}  <-- THIS", flush=True)

    grows = abs(tw.get('r_poly3', 0)) > 0.40
    print(f"\n  Signal {'GROWS' if grows else 'DOES NOT GROW'} beyond N=3000.", flush=True)

    total_time = time.perf_counter() - t0
    print(f"  Wall time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    output = {"N": N, "M": M, "gpu": gpu_ok, "results": results,
              "spectral_width_degcv": {"r": float(r_sw_dcv), "p": float(p_sw_dcv)},
              "wall_time_sec": total_time}
    save_experiment(
        ExperimentMeta(route=2, name="sj_n5000_gpu", N=N, M=M, status="completed",
                       description=f"SJ vacuum N={N} scaling test (GPU)" if gpu_ok else f"SJ N={N} CPU"),
        output, RESULTS_DIR / "sj_n5000.json")
    print("Done.", flush=True)

if __name__ == "__main__":
    main()
