"""
FND-1 EXP-13: d=3 Commutator [H,M] — Weyl Null Test.

In d=3, the Weyl tensor vanishes identically. All curvature is Ricci.
If [H,M] shows signal in d=3 → it detects Ricci (not Weyl-specific).
If [H,M] shows NO signal in d=3 → it is Weyl-specific (d=4 only).

This is the single most informative experiment for interpreting the
EXP-3 result (r=-0.64 in d=4 quadrupole).

Uses d=3 BD coefficients from Dowker-Glaser arXiv:1305.2588:
  n_d=3 layers, C = {1, -27/8, 9/4}, rho scaling = rho^{2/3}.

Run:
    python analysis/scripts/fnd1_exp13_d3_commutator.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import math
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from scipy import stats
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_exp11_d3_intermediate import (
    sprinkle_3d_flat, causal_matrix_3d, build_link_graph_3d, bd_action_3d,
)
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

N_VALUES = [500, 1000, 2000]
N_PRIMARY = 2000          # match EXP-3 d=4 for direct comparison
M_ENSEMBLE = 100          # match EXP-3 d=4
T_DIAMOND = 1.0
MASTER_SEED = 1313
WORKERS = N_WORKERS

# Same eps values as EXP-11 d=3 Fiedler test
EPS_VALUES = [0.0, 0.5, 1.0, 2.0, 4.0]


# ---------------------------------------------------------------------------
# d=3 BD off-diagonal operator (lower-triangular)
# ---------------------------------------------------------------------------

def build_bd_L_3d(C, n_matrix, rho):
    """Build d=3 BD off-diagonal operator.

    Dowker-Glaser arXiv:1305.2588, Table 1:
    n_d=3 layers. C_k = {1, -27/8, 9/4}.
    Scale: rho^{2/d} = rho^{2/3} for d=3.
    """
    past = C.T
    n_past = n_matrix.T
    n_int = np.rint(n_past).astype(np.int64)
    causal_mask = past > 0.5

    scale = rho ** (2.0 / 3.0)
    N = C.shape[0]
    L = np.zeros((N, N), dtype=np.float64)
    L[causal_mask & (n_int == 0)] = 1.0 * scale      # C_1 = 1
    L[causal_mask & (n_int == 1)] = -27.0 / 8.0 * scale  # C_2 = -27/8
    L[causal_mask & (n_int == 2)] = 9.0 / 4.0 * scale     # C_3 = 9/4

    return L


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(args):
    """Compute [H,M] commutator observables for one d=3 sprinkling."""
    seed_int, N, T, eps = args

    rng = np.random.default_rng(seed_int)
    pts = sprinkle_3d_flat(N, T, rng)
    C = causal_matrix_3d(pts, eps)

    total_causal = float(np.sum(C))

    # d=3 diamond volume: V = pi*T^3 / 12 (d=3 Alexandrov set)
    V = np.pi * T ** 3 / 12.0
    rho = N / V

    # Link graph + layers
    A_link, n_links, n_matrix = build_link_graph_3d(C)
    bd = bd_action_3d(N, n_matrix, C)

    # d=3 BD operator
    L = build_bd_L_3d(C, n_matrix, rho)

    # Commutator: [H,M] = (L^T L - L L^T) / 2
    L_sp = sp.csr_matrix(L)
    comm = ((L_sp.T @ L_sp - L_sp @ L_sp.T) / 2.0).toarray()
    comm = (comm + comm.T) / 2.0  # enforce exact symmetry
    comm_eigs = gpu_eigvalsh(comm)

    # Observables (same as EXP-3 for direct comparison)
    a = np.abs(comm_eigs)
    s_total = float(np.sum(a))
    if s_total > 0:
        p_comm = a / s_total
        comm_entropy = float(-np.sum(p_comm * np.log(p_comm + 1e-300)))
    else:
        comm_entropy = 0.0
    comm_frobenius = float(np.sqrt(np.sum(comm_eigs ** 2)))
    comm_max = float(np.max(a))

    return {
        "comm_entropy": comm_entropy,
        "comm_frobenius": comm_frobenius,
        "comm_max": comm_max,
        "total_causal": total_causal,
        "bd_action": bd,
        "n_links": n_links,
        "eps": eps,
    }


# ---------------------------------------------------------------------------
# Mediation (same as EXP-3)
# ---------------------------------------------------------------------------

def partial_corr(x, y, controls):
    """Partial correlation: r(x, y | controls)."""
    from numpy.linalg import lstsq
    cx, _, _, _ = lstsq(controls, x, rcond=None)
    cy, _, _, _ = lstsq(controls, y, rcond=None)
    rx = x - controls @ cx
    ry = y - controls @ cy
    if np.std(rx) < 1e-15 or np.std(ry) < 1e-15:
        return 0.0, 1.0
    return stats.pearsonr(rx, ry)


def mediation_one_obs(eps_arr, obs_arr, tc_arr, bd_arr):
    """Partial r for one observable vs eps (linear + quadratic)."""
    controls = np.column_stack([tc_arr, tc_arr ** 2, bd_arr, np.ones(len(tc_arr))])
    out = {}
    for pred_name, pred in [("linear", eps_arr), ("quadratic", eps_arr ** 2)]:
        r_d, p_d = stats.pearsonr(pred, obs_arr)
        r_p, p_p = partial_corr(pred, obs_arr, controls)
        out[f"{pred_name}_r_direct"] = float(r_d)
        out[f"{pred_name}_r_partial"] = float(r_p)
        out[f"{pred_name}_p_partial"] = float(p_p)

    # Best predictor
    if abs(out["linear_r_partial"]) >= abs(out["quadratic_r_partial"]):
        out["best"] = "linear"
        out["best_r_partial"] = out["linear_r_partial"]
        out["best_p_partial"] = out["linear_p_partial"]
    else:
        out["best"] = "quadratic"
        out["best_r_partial"] = out["quadratic_r_partial"]
        out["best_p_partial"] = out["quadratic_p_partial"]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    meta = ExperimentMeta(
        route=3, name="exp13_d3_commutator",
        description="d=3 commutator [H,M]: Weyl null test (Weyl=0 in d=3)",
        N=N_PRIMARY, M=M_ENSEMBLE, status="running",
    )

    print("=" * 70, flush=True)
    print("FND-1 EXP-13: d=3 COMMUTATOR [H,M] — WEYL NULL TEST", flush=True)
    print("=" * 70, flush=True)
    print(f"N values: {N_VALUES}, M={M_ENSEMBLE}", flush=True)
    print(f"Eps: {EPS_VALUES}", flush=True)
    print(f"d=3: Weyl=0, all curvature is Ricci", flush=True)
    print(f"Workers: {WORKERS}", flush=True)
    print(flush=True)

    # Benchmark
    print("=== BENCHMARK ===", flush=True)
    for N in N_VALUES:
        t0 = time.perf_counter()
        _worker((42, N, T_DIAMOND, 0.0))
        dt = time.perf_counter() - t0
        tasks = M_ENSEMBLE * len(EPS_VALUES)
        par = tasks * dt / WORKERS
        print(f"  N={N:5d}: {dt:.3f}s/task, {tasks} tasks, parallel: {par / 60:.1f} min",
              flush=True)

    ss = np.random.SeedSequence(MASTER_SEED)
    results_by_N = {}

    for N in N_VALUES:
        print(f"\n{'=' * 60}", flush=True)
        print(f"N = {N}", flush=True)
        print("=" * 60, flush=True)

        all_results = []
        for eps in EPS_VALUES:
            eps_ss = ss.spawn(1)[0]
            seeds = eps_ss.spawn(M_ENSEMBLE)
            seed_ints = [int(s.generate_state(1)[0]) for s in seeds]
            args = [(si, N, T_DIAMOND, eps) for si in seed_ints]

            t0 = time.perf_counter()
            with Pool(WORKERS, initializer=_init_worker) as pool:
                raw = pool.map(_worker, args)
            elapsed = time.perf_counter() - t0

            ent = [r["comm_entropy"] for r in raw]
            frob = [r["comm_frobenius"] for r in raw]
            print(f"  eps={eps:+.2f}: entropy={np.mean(ent):.4f}+-{np.std(ent):.4f}"
                  f"  frob={np.mean(frob):.1f}  [{elapsed:.1f}s]", flush=True)
            all_results.extend(raw)

        # Mediation for each observable
        eps_arr = np.array([r["eps"] for r in all_results])
        tc_arr = np.array([r["total_causal"] for r in all_results])
        bd_arr = np.array([r["bd_action"] for r in all_results])

        mediation = {"n": len(all_results)}
        best_r = 0
        best_obs = "comm_entropy"
        best_p = 1.0
        best_pred = "linear"
        for obs_name in ["comm_entropy", "comm_frobenius", "comm_max"]:
            obs_arr = np.array([r[obs_name] for r in all_results])
            med = mediation_one_obs(eps_arr, obs_arr, tc_arr, bd_arr)
            mediation[obs_name] = med
            if abs(med["best_r_partial"]) > abs(best_r):
                best_r = med["best_r_partial"]
                best_obs = obs_name
                best_p = med["best_p_partial"]
                best_pred = med["best"]

        mediation["best_observable"] = best_obs
        mediation["best_r_partial"] = float(best_r)
        mediation["best_p_partial"] = float(best_p)
        mediation["best"] = best_pred

        results_by_N[str(N)] = {
            "mediation": mediation,
            "n_sprinklings": len(all_results),
        }

        print(f"  Best: {best_obs} ({best_pred}) r_partial={best_r:+.4f} p={best_p:.2e}",
              flush=True)

    # ==================================================================
    # VERDICT: Compare with EXP-3 d=4 commutator
    # ==================================================================

    total_time = time.perf_counter() - t_total

    r_best = results_by_N[str(N_PRIMARY)]
    rp = r_best["mediation"]["best_r_partial"]
    pp = r_best["mediation"]["best_p_partial"]

    # EXP-3 d=4 reference: r=-0.6394 on quadrupole at N=2000
    # If d=3 |r| < 0.10 → Weyl-specific
    # If d=3 |r| > 0.30 → detects Ricci too
    if abs(rp) < 0.10:
        verdict = (f"d=3 NULL: |r_partial|={abs(rp):.4f} (p={pp:.2e}). "
                   f"[H,M] does NOT detect Ricci curvature → WEYL-SPECIFIC in d=4.")
    elif abs(rp) < 0.30 and pp > 0.01:
        verdict = (f"d=3 WEAK: r_partial={rp:+.4f} (p={pp:.2e}). "
                   f"[H,M] has marginal Ricci sensitivity.")
    else:
        verdict = (f"d=3 SIGNAL: r_partial={rp:+.4f} (p={pp:.2e}). "
                   f"[H,M] detects Ricci curvature → NOT Weyl-specific.")

    print(f"\n{'=' * 70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"Wall time: {total_time:.0f}s ({total_time / 60:.1f} min)", flush=True)
    print("=" * 70, flush=True)

    meta.status = "completed"
    meta.verdict = verdict
    meta.wall_time_sec = total_time

    output = {
        "parameters": {
            "N_values": N_VALUES, "N_primary": N_PRIMARY,
            "M": M_ENSEMBLE, "T": T_DIAMOND,
            "eps_values": EPS_VALUES,
            "d": 3,
            "bd_coefficients": "Dowker-Glaser 1305.2588: {1, -27/8, 9/4}",
            "metric": "-(1+eps*r_m^2)*dt^2 + dx^2 + dy^2",
            "weyl_tensor": "identically zero in d=3",
        },
        "results_by_N": results_by_N,
        "comparison_with_d4": {
            "exp3_d4_r_partial": -0.6394,
            "exp3_d4_p_partial": 8.1e-59,
            "exp3_d4_observable": "comm_frobenius (quadrupole)",
        },
        "verdict": verdict,
        "wall_time_sec": total_time,
    }

    out_path = RESULTS_DIR / "exp13_d3_commutator.json"
    save_experiment(meta, output, out_path)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
