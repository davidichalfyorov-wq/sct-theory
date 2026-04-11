"""
FND-1 EXP-9: d=2 Large Ensemble — a_1 Curvature Extraction.

EXP-7 showed the DW a_0 prediction fails (plateau = 0.047 vs target 0.159).
But the SUBLEADING term a_1 ~ -R_bar/(12*pi*rho) might still be detectable
with a sufficiently large ensemble (M=10000).

At N=500: curvature signal a_1 ~ R/(12*pi*rho) ~ eps^2 / (12*pi*2000) ~ 10^{-6}.
The ensemble SEM at M=10000 is ~std/100. If std ~ 10^{-4}, SEM ~ 10^{-6}.
So a_1 is MARGINALLY detectable.

Uses the LINK-GRAPH Laplacian (not BD operator, per EXP-12 findings)
for the heat trace computation.

Run:
  python analysis/scripts/fnd1_exp9_d2_large_ensemble.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_ensemble_runner import (
    sprinkle_diamond,
    compute_interval_cardinalities,
)
from fnd1_gate5_runner import sprinkle_curved
from fnd1_route2_link_geometry import build_link_adjacency, build_link_laplacian
from fnd1_experiment_registry import (
    ExperimentMeta, save_experiment, RESULTS_DIR,
)
from fnd1_parallel import N_WORKERS, _init_worker
try:
    from fnd1_gpu import gpu_eigvalsh
except ImportError:
    gpu_eigvalsh = np.linalg.eigvalsh

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_VALUES = [500, 1000]
M_ENSEMBLE = 5000        # large ensemble for subleading extraction
T_DIAMOND = 1.0
MASTER_SEED = 9977
WORKERS = N_WORKERS

# Curvature values: need large eps for detectable a_1
EPS_VALUES = [0.0, 0.5]

# Heat trace grid
N_TAU = 100
TAU_MIN = 1e-2
TAU_MAX = 50.0
TAU_GRID = np.logspace(np.log10(TAU_MIN), np.log10(TAU_MAX), N_TAU)


# ---------------------------------------------------------------------------
# Worker: link-graph Laplacian heat trace in d=2
# ---------------------------------------------------------------------------

def _worker(args):
    """Compute link-graph Laplacian eigenvalues and heat trace for one d=2 sprinkling."""
    seed_int, N, T, eps = args
    import scipy.sparse as sp

    rng = np.random.default_rng(seed_int)
    if abs(eps) < 1e-12:
        pts, C = sprinkle_diamond(N, T, rng)
    else:
        pts, C = sprinkle_curved(N, eps, T, rng)

    n_mat = compute_interval_cardinalities(C)
    A_link = build_link_adjacency(C, n_mat)
    L_link = build_link_laplacian(A_link)
    L_dense = L_link.toarray()

    eigenvalues = gpu_eigvalsh(L_dense)
    lam_nz = eigenvalues[eigenvalues > 1e-8]
    n_zero = N - len(lam_nz)

    # Heat trace: K(tau) = (1/N) * sum exp(-tau * lam_k)  (over nonzero evals)
    if len(lam_nz) > 0:
        K = np.sum(np.exp(-TAU_GRID[:, None] * lam_nz[None, :]), axis=1) / N
    else:
        K = np.zeros(N_TAU)

    # tau*K(tau) for a_0 comparison with EXP-7
    tK = TAU_GRID * K

    # For a_1 extraction: K(tau) - a_0/tau should -> a_1 constant
    # But we don't know a_0 exactly for the link Laplacian.
    # Instead, compare flat vs curved: Delta_K = K_curved - K_flat
    # The difference should be proportional to integrated curvature.

    # Return compact summary (not full K array — too large for M=5000)
    # Sample at 20 tau values for curve comparison
    sample_idx = np.linspace(0, N_TAU - 1, 20, dtype=int)

    return {
        "tK_samples": {f"{TAU_GRID[i]:.4f}": float(tK[i]) for i in sample_idx},
        "K_samples": {f"{TAU_GRID[i]:.4f}": float(K[i]) for i in sample_idx},
        "tK_max": float(np.max(tK)) if len(tK) > 0 else 0.0,
        "n_zero": n_zero,
        "lam_max": float(np.max(lam_nz)) if len(lam_nz) > 0 else 0.0,
        "total_causal": float(np.sum(C)),
        "eps": eps,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    meta = ExperimentMeta(
        route=2, name="exp9_d2_large_ensemble",
        description="d=2 large ensemble: M=5000 link-graph heat trace for a_1 extraction",
        N=max(N_VALUES), M=M_ENSEMBLE, status="running",
    )

    print("=" * 70, flush=True)
    print("FND-1 EXP-9: d=2 LARGE ENSEMBLE (M=5000)", flush=True)
    print("=" * 70, flush=True)
    print(f"N values: {N_VALUES}, M={M_ENSEMBLE}", flush=True)
    print(f"Eps: {EPS_VALUES}", flush=True)
    print(f"Using LINK-GRAPH Laplacian (not BD, per EXP-12)", flush=True)
    print(flush=True)

    # Benchmark
    print("=== BENCHMARK ===", flush=True)
    for N in N_VALUES:
        t0 = time.perf_counter()
        _worker((42, N, T_DIAMOND, 0.0))
        dt = time.perf_counter() - t0
        tasks = M_ENSEMBLE * len(EPS_VALUES)
        par = tasks * dt / WORKERS
        print(f"  N={N:5d}: {dt:.3f}s/task, M={M_ENSEMBLE}*{len(EPS_VALUES)} tasks,"
              f" parallel({WORKERS}w): {par / 60:.1f} min", flush=True)

    ss = np.random.SeedSequence(MASTER_SEED)
    results_by_N = {}

    for N in N_VALUES:
        print(f"\n{'=' * 60}", flush=True)
        print(f"N = {N}", flush=True)
        print("=" * 60, flush=True)

        data_by_eps = {}
        for eps in EPS_VALUES:
            eps_ss = ss.spawn(1)[0]
            seeds = eps_ss.spawn(M_ENSEMBLE)
            seed_ints = [int(s.generate_state(1)[0]) for s in seeds]
            args = [(si, N, T_DIAMOND, eps) for si in seed_ints]

            t0 = time.perf_counter()
            with Pool(WORKERS, initializer=_init_worker) as pool:
                raw = pool.map(_worker, args)
            elapsed = time.perf_counter() - t0

            # Ensemble average of tK at sampled tau values
            tau_keys = list(raw[0]["tK_samples"].keys())
            tK_means = {}
            tK_sems = {}
            for tk in tau_keys:
                vals = [r["tK_samples"][tk] for r in raw]
                tK_means[tk] = float(np.mean(vals))
                tK_sems[tk] = float(np.std(vals, ddof=1) / np.sqrt(M_ENSEMBLE))

            tK_max_mean = float(np.mean([r["tK_max"] for r in raw]))
            tK_max_sem = float(np.std([r["tK_max"] for r in raw], ddof=1) / np.sqrt(M_ENSEMBLE))

            data_by_eps[str(eps)] = {
                "tK_means": tK_means,
                "tK_sems": tK_sems,
                "tK_max_mean": tK_max_mean,
                "tK_max_sem": tK_max_sem,
                "mean_n_zero": float(np.mean([r["n_zero"] for r in raw])),
                "mean_lam_max": float(np.mean([r["lam_max"] for r in raw])),
            }

            print(f"  eps={eps:+.2f}: tK_max={tK_max_mean:.6f}+-{tK_max_sem:.6f}"
                  f"  n_zero={data_by_eps[str(eps)]['mean_n_zero']:.1f}"
                  f"  [{elapsed:.1f}s]", flush=True)

        # Curvature test: paired difference at each tau
        if "0.0" in data_by_eps and "0.5" in data_by_eps:
            flat_tK = data_by_eps["0.0"]["tK_means"]
            curv_tK = data_by_eps["0.5"]["tK_means"]
            flat_sem = data_by_eps["0.0"]["tK_sems"]
            curv_sem = data_by_eps["0.5"]["tK_sems"]

            # Compute Delta(tK) and its significance at each tau
            deltas = {}
            sig_count = 0
            for tk in tau_keys:
                d = curv_tK[tk] - flat_tK[tk]
                se = np.sqrt(flat_sem[tk] ** 2 + curv_sem[tk] ** 2)
                z = d / se if se > 0 else 0.0
                sig = abs(z) > 2.576  # p < 0.01
                deltas[tk] = {
                    "delta": float(d), "se": float(se),
                    "z": float(z), "significant": bool(sig),
                }
                if sig:
                    sig_count += 1

            frac_sig = sig_count / len(tau_keys)
            print(f"  Delta(tK) significant at {sig_count}/{len(tau_keys)}"
                  f" tau values ({frac_sig * 100:.0f}%)", flush=True)

            # tK_max difference
            d_max = data_by_eps["0.5"]["tK_max_mean"] - data_by_eps["0.0"]["tK_max_mean"]
            se_max = np.sqrt(data_by_eps["0.0"]["tK_max_sem"] ** 2 +
                             data_by_eps["0.5"]["tK_max_sem"] ** 2)
            z_max = d_max / se_max if se_max > 0 else 0.0
            print(f"  Delta(tK_max) = {d_max:+.6f} (z={z_max:+.2f})", flush=True)
        else:
            deltas = {}
            frac_sig = 0.0
            z_max = 0.0

        results_by_N[str(N)] = {
            "by_eps": data_by_eps,
            "deltas": deltas,
            "frac_significant": frac_sig,
            "tK_max_z": float(z_max),
        }

    # ==================================================================
    # VERDICT
    # ==================================================================

    total_time = time.perf_counter() - t_total

    best_N = str(N_VALUES[-1])
    rb = results_by_N[best_N]

    if rb["frac_significant"] > 0.3:
        verdict = (f"a_1 DETECTED: {rb['frac_significant'] * 100:.0f}% of tau values"
                   f" show significant flat-vs-curved difference (z_max={rb['tK_max_z']:+.2f})")
    elif abs(rb["tK_max_z"]) > 2.576:
        verdict = (f"tK_max DIFFERS: z={rb['tK_max_z']:+.2f} (but sparse tau significance)")
    else:
        verdict = (f"a_1 NOT DETECTED: {rb['frac_significant'] * 100:.0f}% significant,"
                   f" z_max={rb['tK_max_z']:+.2f}")

    print(f"\n{'=' * 70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"Wall time: {total_time:.0f}s ({total_time / 60:.1f} min)", flush=True)
    print("=" * 70, flush=True)

    meta.status = "completed"
    meta.verdict = verdict
    meta.wall_time_sec = total_time

    output = {
        "parameters": {
            "N_values": N_VALUES, "M": M_ENSEMBLE, "T": T_DIAMOND,
            "eps_values": EPS_VALUES,
            "tau_range": [TAU_MIN, TAU_MAX, N_TAU],
            "operator": "link-graph Laplacian (not BD)",
        },
        "results_by_N": results_by_N,
        "verdict": verdict,
        "wall_time_sec": total_time,
    }

    out_path = RESULTS_DIR / "exp9_d2_large_ensemble.json"
    save_experiment(meta, output, out_path)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
