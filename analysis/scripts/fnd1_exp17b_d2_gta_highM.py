"""
FND-1 EXP-17b: GTA d=2 High-M — Resolve Borderline Result.

EXP-17 gave r=+0.10, p=0.07 (borderline). Expected: null (conformal invariance).
M=200 (double EXP-17) to resolve: either p<0.05 (weak signal exists) or p>0.1 (noise).

Uses same code as EXP-17 but M=200 and different seed.

Run:
    python analysis/scripts/fnd1_exp17b_d2_gta_highM.py
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy import stats
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fnd1_ensemble_runner import sprinkle_diamond, compute_interval_cardinalities, build_bd_L
from fnd1_gate5_runner import sprinkle_curved
from fnd1_experiment_registry import ExperimentMeta, save_experiment, RESULTS_DIR
from fnd1_parallel import N_WORKERS, _init_worker

try:
    from fnd1_gpu import gpu_eigvalsh
except ImportError:
    gpu_eigvalsh = np.linalg.eigvalsh

N_VALUES = [500, 1000, 2000]
M_ENSEMBLE = 200  # double EXP-17 for resolution
T_DIAMOND = 1.0
MASTER_SEED = 17170  # different from EXP-17 (1717)
WORKERS = N_WORKERS
EPS_VALUES = [0.0, 0.25, 0.5]

def _worker(args):
    seed_int, N, T, eps = args
    rng = np.random.default_rng(seed_int)
    if abs(eps) < 1e-12:
        pts, C = sprinkle_diamond(N, T, rng)
    else:
        pts, C = sprinkle_curved(N, eps, T, rng)

    total_causal = float(np.sum(C))
    V = T ** 2 / 2.0
    rho = N / V
    n_mat = compute_interval_cardinalities(C)
    L = build_bd_L(C, n_mat, rho)

    L_sp = sp.csr_matrix(L)
    comm = ((L_sp.T @ L_sp - L_sp @ L_sp.T) / 2.0).toarray()
    comm = (comm + comm.T) / 2.0
    comm_eigs = gpu_eigvalsh(comm)

    a = np.abs(comm_eigs)
    s = float(np.sum(a))
    entropy = float(-np.sum((a/s) * np.log(a/s + 1e-300))) if s > 0 else 0.0

    # Correct d=2 BD action: N - 2*N0 + 4*N1 - 2*N2
    past = C.T; n_past = n_mat.T
    N0 = int(np.sum((past > 0) & (n_past == 0)))
    N1 = int(np.sum((past > 0) & (n_past == 1)))
    N2 = int(np.sum((past > 0) & (n_past == 2)))
    bd = float(N - 2 * N0 + 4 * N1 - 2 * N2)

    return {
        "comm_entropy": entropy,
        "comm_frobenius": float(np.sqrt(np.sum(comm_eigs ** 2))),
        "comm_max_abs": float(np.max(a)),
        "total_causal": total_causal,
        "bd_action": bd,
        "eps": eps,
    }

def partial_corr(x, y, controls):
    from numpy.linalg import lstsq
    cx, _, _, _ = lstsq(controls, x, rcond=None)
    cy, _, _, _ = lstsq(controls, y, rcond=None)
    rx = x - controls @ cx; ry = y - controls @ cy
    if np.std(rx) < 1e-15 or np.std(ry) < 1e-15: return 0.0, 1.0
    return stats.pearsonr(rx, ry)

def main():
    t_total = time.perf_counter()
    meta = ExperimentMeta(route=3, name="exp17b_d2_gta_highM",
        description="d=2 GTA [H,M] high-M: resolve EXP-17 borderline (r=0.10, p=0.07)",
        N=max(N_VALUES), M=M_ENSEMBLE, status="running")

    print("=" * 70)
    print("FND-1 EXP-17b: d=2 GTA HIGH-M (resolve borderline)")
    print("=" * 70)
    print(f"N: {N_VALUES}, M={M_ENSEMBLE} (2x EXP-17), eps: {EPS_VALUES}")
    print(f"EXP-17 result: r=+0.10, p=0.07. If real signal → p<0.05 here.")
    print()

    ss = np.random.SeedSequence(MASTER_SEED)
    all_children = ss.spawn(len(N_VALUES) * len(EPS_VALUES) * M_ENSEMBLE)
    child_idx = 0
    results_by_N = {}

    for N in N_VALUES:
        print(f"\nN = {N}")
        all_results = []
        for eps in EPS_VALUES:
            seed_ints = [int(all_children[child_idx + i].generate_state(1)[0])
                         for i in range(M_ENSEMBLE)]
            child_idx += M_ENSEMBLE
            args = [(si, N, T_DIAMOND, eps) for si in seed_ints]
            t0 = time.perf_counter()
            with Pool(WORKERS, initializer=_init_worker) as pool:
                raw = pool.map(_worker, args)
            elapsed = time.perf_counter() - t0
            ent = [r["comm_entropy"] for r in raw]
            print(f"  eps={eps:+.2f}: entropy={np.mean(ent):.4f}+-{np.std(ent):.4f} [{elapsed:.1f}s]")
            all_results.extend(raw)

        eps_arr = np.array([r["eps"] for r in all_results])
        tc_arr = np.array([r["total_causal"] for r in all_results])
        bd_arr = np.array([r["bd_action"] for r in all_results])
        controls = np.column_stack([tc_arr, tc_arr ** 2, bd_arr, np.ones(len(all_results))])

        mediation = {"n": len(all_results)}
        best_r = 0.0; best_obs = "comm_entropy"; best_p = 1.0; best_pred = "linear"
        for obs_name in ["comm_entropy", "comm_frobenius", "comm_max_abs"]:
            obs_arr = np.array([r[obs_name] for r in all_results])
            out = {}
            for pn, pred in [("linear", eps_arr), ("quadratic", eps_arr ** 2)]:
                r_d, _ = stats.pearsonr(pred, obs_arr)
                r_p, p_p = partial_corr(pred, obs_arr, controls)
                out[f"{pn}_r_direct"] = float(r_d)
                out[f"{pn}_r_partial"] = float(r_p)
                out[f"{pn}_p_partial"] = float(p_p)
            out["best"] = "linear" if abs(out["linear_r_partial"]) >= abs(out["quadratic_r_partial"]) else "quadratic"
            out["best_r_partial"] = out[f"{out['best']}_r_partial"]
            out["best_p_partial"] = out[f"{out['best']}_p_partial"]
            mediation[obs_name] = out
            if abs(out["best_r_partial"]) > abs(best_r):
                best_r = out["best_r_partial"]; best_obs = obs_name
                best_p = out["best_p_partial"]; best_pred = out["best"]

        mediation["best_observable"] = best_obs
        mediation["best_r_partial"] = float(best_r)
        mediation["best_p_partial"] = float(best_p)
        mediation["best"] = best_pred
        results_by_N[str(N)] = {"mediation": mediation, "n_sprinklings": len(all_results)}
        print(f"  Best: {best_obs} ({best_pred}) r_partial={best_r:+.4f} p={best_p:.2e}")

    total_time = time.perf_counter() - t_total
    rp = results_by_N[str(max(N_VALUES))]["mediation"]["best_r_partial"]
    pp = results_by_N[str(max(N_VALUES))]["mediation"]["best_p_partial"]

    # Resolution verdict
    if abs(rp) < 0.05 or pp > 0.10:
        verdict = f"RESOLVED: NULL confirmed. r={rp:+.4f} (p={pp:.2e}). EXP-17 was noise."
    elif abs(rp) > 0.10 and pp < 0.01:
        verdict = f"RESOLVED: WEAK SIGNAL exists. r={rp:+.4f} (p={pp:.2e}). Density leakage in d=2."
    elif pp < 0.05:
        verdict = f"MARGINAL: r={rp:+.4f} (p={pp:.2e}). Borderline persists. Need M=500."
    else:
        verdict = f"INCONCLUSIVE: r={rp:+.4f} (p={pp:.2e}). Cannot resolve."

    print(f"\nVERDICT: {verdict}")
    print(f"Wall time: {total_time:.0f}s ({total_time/60:.1f} min)")

    meta.status = "completed"; meta.verdict = verdict; meta.wall_time_sec = total_time
    output = {
        "parameters": {"N_values": N_VALUES, "M": M_ENSEMBLE, "eps_values": EPS_VALUES,
                        "d": 2, "reference": "EXP-17 r=+0.10 p=0.07"},
        "results_by_N": results_by_N, "verdict": verdict, "wall_time_sec": total_time,
    }
    save_experiment(meta, output, RESULTS_DIR / "exp17b_d2_gta_highM.json")
    print(f"Saved: exp17b_d2_gta_highM.json")

if __name__ == "__main__":
    main()
