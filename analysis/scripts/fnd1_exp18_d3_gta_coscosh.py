"""
FND-1 EXP-18: GTA Coscosh Control in d=3.

EXP-13 showed GTA detects Ricci curvature in d=3 (r=+0.51) with metric
-(1+eps*r²)dt²+dx²+dy². This metric changes both curvature AND density.

This experiment tests CJ (Conformal Juncture) property in d=3:
does GTA distinguish curvature from density in d=3 as it does in d=4?

Uses SAME d=3 metric but with coscosh-like density variation for comparison.
Actually: uses the SAME metric at SAME eps but tests whether the mediation
controls (TC+TC²+BD) are sufficient by checking r_direct vs r_partial ratio.

Run:
    python analysis/scripts/fnd1_exp18_d3_gta_coscosh.py
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys, time, json, math
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy import stats
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fnd1_exp11_d3_intermediate import sprinkle_3d_flat, causal_matrix_3d, build_link_graph_3d, bd_action_3d
from fnd1_experiment_registry import ExperimentMeta, save_experiment, RESULTS_DIR
from fnd1_parallel import N_WORKERS, _init_worker

try:
    from fnd1_gpu import gpu_eigvalsh
except ImportError:
    gpu_eigvalsh = np.linalg.eigvalsh

N_VALUES = [500, 1000, 2000]
M_ENSEMBLE = 100
T_DIAMOND = 1.0
MASTER_SEED = 1818
WORKERS = N_WORKERS
# Same eps as EXP-13 for direct comparison
EPS_VALUES = [0.0, 0.5, 1.0, 2.0, 4.0]

def build_bd_L_3d(C, n_matrix, rho):
    """d=3 BD operator. Dowker-Glaser: {1, -27/8, 9/4}, scale=rho^{2/3}."""
    past = C.T; n_past = n_matrix.T
    n_int = np.rint(n_past).astype(np.int64)
    causal_mask = past > 0.5
    scale = rho ** (2.0 / 3.0)
    N = C.shape[0]
    L = np.zeros((N, N), dtype=np.float64)
    L[causal_mask & (n_int == 0)] = 1.0 * scale
    L[causal_mask & (n_int == 1)] = -27.0 / 8.0 * scale
    L[causal_mask & (n_int == 2)] = 9.0 / 4.0 * scale
    return L

def _worker(args):
    seed_int, N, T, eps = args
    rng = np.random.default_rng(seed_int)
    pts = sprinkle_3d_flat(N, T, rng)
    C = causal_matrix_3d(pts, eps)

    total_causal = float(np.sum(C))
    V = np.pi * T ** 3 / 12.0
    rho = N / V

    A_link, n_links, n_matrix = build_link_graph_3d(C)
    bd = bd_action_3d(N, n_matrix, C)

    L = build_bd_L_3d(C, n_matrix, rho)
    del C, n_matrix  # free memory

    L_sp = sp.csr_matrix(L)
    comm = ((L_sp.T @ L_sp - L_sp @ L_sp.T) / 2.0).toarray()
    comm = (comm + comm.T) / 2.0
    comm_eigs = gpu_eigvalsh(comm)

    a = np.abs(comm_eigs)
    s = float(np.sum(a))
    entropy = float(-np.sum((a/s) * np.log(a/s + 1e-300))) if s > 0 else 0.0
    frobenius = float(np.sqrt(np.sum(comm_eigs ** 2)))
    max_abs = float(np.max(a))

    # Additional: TC change fraction from flat (density indicator)
    return {
        "comm_entropy": entropy, "comm_frobenius": frobenius, "comm_max_abs": max_abs,
        "total_causal": total_causal, "bd_action": bd, "n_links": n_links, "eps": eps,
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
    meta = ExperimentMeta(route=3, name="exp18_d3_gta_coscosh",
        description="d=3 GTA CJ test: purity of curvature signal (r_partial/r_direct ratio)",
        N=max(N_VALUES), M=M_ENSEMBLE, status="running")

    print("=" * 70)
    print("FND-1 EXP-18: d=3 GTA CJ (Conformal Juncture) TEST")
    print("=" * 70)
    print(f"N: {N_VALUES}, M={M_ENSEMBLE}, eps: {EPS_VALUES}")
    print(f"Metric: -(1+eps*r²)dt²+dx²+dy² (Ricci, changes both curvature and density)")
    print(f"Test: r_partial/r_direct ratio = purity of curvature signal")
    print()

    ss = np.random.SeedSequence(MASTER_SEED)
    results_by_N = {}

    for N in N_VALUES:
        print(f"\nN = {N}")
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
            tc_mean = np.mean([r["total_causal"] for r in raw])
            print(f"  eps={eps:+.2f}: entropy={np.mean(ent):.4f} TC={tc_mean:.0f} [{elapsed:.1f}s]")
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
            out["best_r_direct"] = out[f"{out['best']}_r_direct"]
            # CJ purity: what fraction of signal survives mediation
            out["purity"] = abs(out["best_r_partial"]) / abs(out["best_r_direct"]) if abs(out["best_r_direct"]) > 1e-10 else 0
            mediation[obs_name] = out
            if abs(out["best_r_partial"]) > abs(best_r):
                best_r = out["best_r_partial"]; best_obs = obs_name
                best_p = out["best_p_partial"]; best_pred = out["best"]

        mediation["best_observable"] = best_obs
        mediation["best_r_partial"] = float(best_r)
        mediation["best_p_partial"] = float(best_p)
        mediation["best"] = best_pred
        # Overall purity
        best_med = mediation[best_obs]
        purity = best_med["purity"]
        mediation["purity"] = float(purity)

        results_by_N[str(N)] = {"mediation": mediation, "n_sprinklings": len(all_results)}
        print(f"  Best: {best_obs} r_direct={best_med['best_r_direct']:+.4f} "
              f"r_partial={best_r:+.4f} purity={purity:.1%}")

    total_time = time.perf_counter() - t_total
    best_N = results_by_N[str(max(N_VALUES))]
    rp = best_N["mediation"]["best_r_partial"]
    pp = best_N["mediation"]["best_p_partial"]
    pur = best_N["mediation"]["purity"]

    # Compare with EXP-13 and d=4 EXP-3
    verdict = (f"d=3 GTA purity={pur:.1%} (r_partial={rp:+.4f}, p={pp:.2e}). "
               f"Compare: d=4 quadrupole purity~100%, d=4 coscosh purity~8%.")

    print(f"\nVERDICT: {verdict}")
    print(f"Wall time: {total_time:.0f}s ({total_time/60:.1f} min)")

    meta.status = "completed"; meta.verdict = verdict; meta.wall_time_sec = total_time
    output = {
        "parameters": {"N_values": N_VALUES, "M": M_ENSEMBLE, "eps_values": EPS_VALUES,
                        "d": 3, "metric": "-(1+eps*r²)dt²+dx²+dy²"},
        "results_by_N": results_by_N, "verdict": verdict, "wall_time_sec": total_time,
    }
    save_experiment(meta, output, RESULTS_DIR / "exp18_d3_gta_coscosh.json")
    print(f"Saved: exp18_d3_gta_coscosh.json")

if __name__ == "__main__":
    main()
