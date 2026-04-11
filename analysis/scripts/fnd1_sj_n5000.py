"""
FND-1: SJ Vacuum at N=5000 — scaling test.

Does trace_W structural residual grow beyond N=3000?
Previous: r|poly3 = -0.054 (N=1000), -0.285 (N=2000), -0.405 (N=3000).
If continues growing at N=5000 → genuine signal. If flattens/reverses → artifact.

CPU only (no GPU lag). Serial execution, ~60-70 min (benchmark at start).

Run:
  "C:/Users/youre/miniconda3/envs/sct-mkl/python.exe" analysis/scripts/fnd1_sj_n5000.py
"""

from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_4d_experiment import (
    sprinkle_4d_flat, sprinkle_4d_ppwave,
    causal_matrix_4d, compute_layers_4d, bd_action_4d,
    build_link_graph, _ppwave_profile,
)
from fnd1_sj_vacuum import build_sj_wightman, sj_observables, mediation_poly
from fnd1_experiment_registry import ExperimentMeta, save_experiment, RESULTS_DIR

N = 5000
M = 40  # per eps (less due to larger N)
T = 1.0
SEED = 555
EPS_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5]

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

    pos_evals, all_evals = build_sj_wightman(C, n_mat, rho)
    obs = sj_observables(pos_evals, all_evals, N)

    # Degree CV (the confounder that killed spectral_width)
    A_link = build_link_graph(C, n_mat)
    degrees = np.array(A_link.sum(axis=1)).ravel()
    deg_cv = float(np.std(degrees) / np.mean(degrees)) if np.mean(degrees) > 0 else 0

    return {"eps": eps, "total_causal": tc, "n_links": N0,
            "bd_action": float(bd), "deg_cv": deg_cv, **obs}


def main():
    t0 = time.perf_counter()
    print(f"FND-1 SJ N={N}: M={M} x {len(EPS_VALUES)} eps = {M*len(EPS_VALUES)} sprinklings", flush=True)

    # Benchmark
    tb = time.perf_counter()
    worker(42, 0.0)
    per_task = time.perf_counter() - tb
    total_est = M * len(EPS_VALUES) * per_task
    print(f"Benchmark: {per_task:.1f}s/task, total est: {total_est/60:.0f} min", flush=True)

    ss = np.random.SeedSequence(SEED)
    data = []

    for eps in EPS_VALUES:
        seeds = [int(cs.generate_state(1)[0]) for cs in ss.spawn(1)[0].spawn(M)]
        print(f"\n  eps={eps:+.2f}:", end=" ", flush=True)
        t1 = time.perf_counter()
        for i, s in enumerate(seeds):
            data.append(worker(s, eps))
            if (i+1) % 10 == 0:
                print(f"{i+1}", end=" ", flush=True)
        tc_mean = np.mean([d["total_causal"] for d in data if abs(d["eps"]-eps)<0.01])
        tr_mean = np.mean([d["trace_W"] for d in data if abs(d["eps"]-eps)<0.01])
        print(f"  ({time.perf_counter()-t1:.0f}s, TC={tc_mean:.0f}, trace={tr_mean:.0f})", flush=True)

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

    # Also test spectral_width with deg_cv control
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
    print(f"\n  spectral_width with deg_cv control: r={r_sw_dcv:+.4f}, p={p_sw_dcv:.2e}", flush=True)

    # Scaling comparison
    print(f"\n{'='*70}", flush=True)
    print("SCALING: trace_W r|poly3 across N", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  N=1000: r|poly3=-0.054  (previous)", flush=True)
    print(f"  N=2000: r|poly3=-0.285  (previous)", flush=True)
    print(f"  N=3000: r|poly3=-0.405  (previous)", flush=True)
    tw_med = results.get("trace_W", {})
    print(f"  N=5000: r|poly3={tw_med.get('r_poly3',0):+.3f}  (THIS RUN)", flush=True)

    total_time = time.perf_counter() - t0
    print(f"\n  Wall time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    # Save
    output = {"N": N, "M": M, "results": results,
              "spectral_width_degcv": {"r": float(r_sw_dcv), "p": float(p_sw_dcv)},
              "wall_time_sec": total_time}
    save_experiment(
        ExperimentMeta(route=2, name="sj_n5000", N=N, M=M, status="completed",
                       description="SJ vacuum N=5000 scaling test"),
        output, RESULTS_DIR / "sj_n5000.json")
    print("Done.", flush=True)

if __name__ == "__main__":
    main()
