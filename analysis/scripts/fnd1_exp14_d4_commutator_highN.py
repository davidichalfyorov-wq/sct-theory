"""
FND-1 EXP-14: d=4 Commutator [H,M] — High N Extension.

Tests whether the commutator signal (r=-0.64 at N=2000 in EXP-3)
survives at N=5000 and N=10000. If it does, [H,M] is viable at
continuum-limit scales. If not, it is a finite-size effect.

Critical comparison:
- Fiedler eigenvalue: DIES at large N (delta=0 in EXP-6)
- Heat trace: GROWS at large N (r=0.84 in EXP-5b)
- Commutator: ??? (this experiment)

Uses quadrupole profile only (pure Weyl, TC-stable, density-blind).

Run:
    python analysis/scripts/fnd1_exp14_d4_commutator_highN.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from scipy import stats
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_4d_experiment import (
    sprinkle_4d_flat, causal_matrix_4d, compute_layers_4d, bd_action_4d,
)
from fnd1_4d_followup import quadrupole_profile
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

N_VALUES = [2000, 5000, 10000]  # 2000 for overlap with EXP-3, then high N
N_PRIMARY = 10000
M_ENSEMBLE = 80               # balance between power and compute
T_DIAMOND = 1.0
MASTER_SEED = 1414
WORKERS = N_WORKERS

# Quadrupole only (density-blind, per EXP-3/EXP-13 findings)
EPS_VALUES = [0.0, 2.0, 5.0, 10.0, 20.0]


# ---------------------------------------------------------------------------
# d=4 BD operator (same as EXP-3)
# ---------------------------------------------------------------------------

def build_bd_L_4d(C, n_matrix, rho):
    """Build d=4 BD off-diagonal operator."""
    past = C.T
    n_past = n_matrix.T
    n_int = np.rint(n_past).astype(np.int64)
    causal_mask = past > 0.5

    scale = np.sqrt(rho)
    N = C.shape[0]
    L = np.zeros((N, N), dtype=np.float64)
    L[causal_mask & (n_int == 0)] = 4.0 * scale
    L[causal_mask & (n_int == 1)] = -36.0 * scale
    L[causal_mask & (n_int == 2)] = 64.0 * scale
    L[causal_mask & (n_int == 3)] = -32.0 * scale
    return L


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(args):
    """Compute [H,M] commutator for one d=4 sprinkling at high N."""
    seed_int, N, T, eps = args

    rng = np.random.default_rng(seed_int)
    pts = sprinkle_4d_flat(N, T, rng)

    # Quadrupole causal matrix
    if abs(eps) < 1e-12:
        C = causal_matrix_4d(pts, 0.0, "flat")
    else:
        t = pts[:, 0]; x = pts[:, 1]; y = pts[:, 2]; z = pts[:, 3]
        dt = t[np.newaxis, :] - t[:, np.newaxis]
        dx = x[np.newaxis, :] - x[:, np.newaxis]
        dy = y[np.newaxis, :] - y[:, np.newaxis]
        dz = z[np.newaxis, :] - z[:, np.newaxis]
        dr2 = dx ** 2 + dy ** 2 + dz ** 2
        xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
        ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
        f_mid = quadrupole_profile(xm, ym)
        mink = dt ** 2 - dr2
        corr = eps * f_mid * (dt + dz) ** 2 / 2.0
        C = ((mink > corr) & (dt > 0)).astype(np.float64)
        del dx, dy, xm, ym, f_mid, mink, corr  # free memory

    total_causal = float(np.sum(C))
    V = np.pi * T ** 4 / 24.0
    rho = N / V

    n_matrix, N0, N1, N2, N3 = compute_layers_4d(C)
    bd = bd_action_4d(N, N0, N1, N2, N3)

    # BD operator + commutator
    L = build_bd_L_4d(C, n_matrix, rho)
    del C, n_matrix  # free ~1.6 GB at N=10000
    L_sp = sp.csr_matrix(L)
    comm = ((L_sp.T @ L_sp - L_sp @ L_sp.T) / 2.0).toarray()
    comm = (comm + comm.T) / 2.0  # enforce symmetry
    comm_eigs = gpu_eigvalsh(comm)

    # Observables
    a = np.abs(comm_eigs)
    s_total = float(np.sum(a))
    if s_total > 0:
        p_comm = a / s_total
        comm_entropy = float(-np.sum(p_comm * np.log(p_comm + 1e-300)))
    else:
        comm_entropy = 0.0
    comm_frobenius = float(np.sqrt(np.sum(comm_eigs ** 2)))
    comm_max_abs = float(np.max(a))

    return {
        "comm_entropy": comm_entropy,
        "comm_frobenius": comm_frobenius,
        "comm_max_abs": comm_max_abs,
        "total_causal": total_causal,
        "bd_action": bd,
        "n_links": int(N0),
        "eps": eps,
    }


# ---------------------------------------------------------------------------
# Mediation
# ---------------------------------------------------------------------------

def partial_corr(x, y, controls):
    from numpy.linalg import lstsq
    cx, _, _, _ = lstsq(controls, x, rcond=None)
    cy, _, _, _ = lstsq(controls, y, rcond=None)
    rx = x - controls @ cx
    ry = y - controls @ cy
    if np.std(rx) < 1e-15 or np.std(ry) < 1e-15:
        return 0.0, 1.0
    return stats.pearsonr(rx, ry)


def mediation_one_obs(eps_arr, obs_arr, tc_arr, bd_arr):
    controls = np.column_stack([tc_arr, tc_arr ** 2, bd_arr, np.ones(len(tc_arr))])
    out = {}
    for pred_name, pred in [("linear", eps_arr), ("quadratic", eps_arr ** 2)]:
        r_d, p_d = stats.pearsonr(pred, obs_arr)
        r_p, p_p = partial_corr(pred, obs_arr, controls)
        out[f"{pred_name}_r_direct"] = float(r_d)
        out[f"{pred_name}_r_partial"] = float(r_p)
        out[f"{pred_name}_p_partial"] = float(p_p)
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
        route=3, name="exp14_d4_commutator_highN",
        description="d=4 commutator [H,M] high N: scaling test to N=10000",
        N=N_PRIMARY, M=M_ENSEMBLE, status="running",
    )

    print("=" * 70, flush=True)
    print("FND-1 EXP-14: d=4 COMMUTATOR [H,M] — HIGH N SCALING", flush=True)
    print("=" * 70, flush=True)
    print(f"N values: {N_VALUES}, M={M_ENSEMBLE}", flush=True)
    print(f"Eps (quadrupole only): {EPS_VALUES}", flush=True)
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
        print(f"  N={N:6d}: {dt:.1f}s/task, {tasks} tasks, parallel: {par / 60:.1f} min",
              flush=True)

    ss = np.random.SeedSequence(MASTER_SEED)
    results_by_N = {}
    scaling_r = []

    for N in N_VALUES:
        # Per-N worker cap: ~10 GB/worker at N=10000 (C + n_matrix + L + comm)
        mem_per_worker = max(0.5, 10.0 * (N / 10000) ** 2)
        workers_N = min(WORKERS, max(1, int(60 / mem_per_worker)))  # 60 GB safe limit
        print(f"\n{'=' * 60}", flush=True)
        print(f"N = {N} (workers={workers_N})", flush=True)
        print("=" * 60, flush=True)

        all_results = []
        for eps in EPS_VALUES:
            eps_ss = ss.spawn(1)[0]
            seeds = eps_ss.spawn(M_ENSEMBLE)
            seed_ints = [int(s.generate_state(1)[0]) for s in seeds]
            args = [(si, N, T_DIAMOND, eps) for si in seed_ints]

            t0 = time.perf_counter()
            with Pool(workers_N, initializer=_init_worker) as pool:
                raw = pool.map(_worker, args)
            elapsed = time.perf_counter() - t0

            ent = [r["comm_entropy"] for r in raw]
            print(f"  eps={eps:+5.1f}: entropy={np.mean(ent):.4f}+-{np.std(ent):.4f}"
                  f"  [{elapsed:.1f}s]", flush=True)
            all_results.extend(raw)

        # Mediation
        eps_arr = np.array([r["eps"] for r in all_results])
        tc_arr = np.array([r["total_causal"] for r in all_results])
        bd_arr = np.array([r["bd_action"] for r in all_results])

        mediation = {"n": len(all_results)}
        best_r = 0.0
        best_obs = "comm_entropy"
        best_p = 1.0
        best_pred = "linear"

        for obs_name in ["comm_entropy", "comm_frobenius", "comm_max_abs"]:
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
        scaling_r.append(float(best_r))

        print(f"  Best: {best_obs} ({best_pred}) r_partial={best_r:+.4f} p={best_p:.2e}",
              flush=True)

        # Consistency check at N=2000 against EXP-3
        if N == 2000:
            exp3_ref = -0.6394
            frob_r = mediation.get("comm_frobenius", {}).get("best_r_partial", 0)
            if abs(abs(frob_r) - abs(exp3_ref)) > 0.25:
                print(f"  WARNING: comm_frobenius r={frob_r:+.4f} vs EXP-3 r={exp3_ref:+.4f} "
                      f"(diff={abs(abs(frob_r)-abs(exp3_ref)):.3f})", flush=True)
            else:
                print(f"  Consistency with EXP-3: comm_frobenius r={frob_r:+.4f} "
                      f"(EXP-3={exp3_ref:+.4f}, diff={abs(abs(frob_r)-abs(exp3_ref)):.3f})", flush=True)

    # Scaling analysis — PER OBSERVABLE (not best-of-3)
    total_time = time.perf_counter() - t_total

    print(f"\n{'=' * 70}", flush=True)
    print("SCALING (per observable)", flush=True)
    print("=" * 70, flush=True)

    Ns = np.array(N_VALUES, dtype=float)
    per_obs_scaling = {}
    for obs_name in ["comm_entropy", "comm_frobenius", "comm_max_abs"]:
        obs_r = []
        for N in N_VALUES:
            med = results_by_N[str(N)]["mediation"].get(obs_name, {})
            obs_r.append(abs(med.get("best_r_partial", 0)))
        per_obs_scaling[obs_name] = obs_r
        print(f"  {obs_name}:", flush=True)
        for i, N in enumerate(N_VALUES):
            print(f"    N={N:6d}: |r|={obs_r[i]:.4f}", flush=True)

        abs_r = np.array(obs_r)
        if all(x > 1e-10 for x in abs_r) and len(abs_r) >= 3:
            lr = stats.linregress(np.log(Ns), np.log(abs_r))
            print(f"    Power law: |r| ~ N^{{{lr.slope:+.3f}}}, R^2={lr.rvalue**2:.3f}")

    # Overall best-of-3 scaling (for verdict)
    for i, N in enumerate(N_VALUES):
        print(f"  Best-of-3 N={N}: |r|={abs(scaling_r[i]):.4f}", flush=True)

    abs_r_best = np.array([abs(x) for x in scaling_r])
    if all(x > 1e-10 for x in abs_r_best):
        lr = stats.linregress(np.log(Ns), np.log(abs_r_best))
        signal_grows = lr.slope > -0.1
    else:
        signal_grows = False

    # Verdict
    r_best = results_by_N[str(N_PRIMARY)]
    rp = r_best["mediation"]["best_r_partial"]
    pp = r_best["mediation"]["best_p_partial"]

    # Compare with EXP-3 at N=2000 (r=-0.64) and EXP-6 Fiedler at N=10000 (r=-0.24)
    if abs(rp) > 0.30 and pp < 0.01:
        verdict = (f"SURVIVES at N={N_PRIMARY}: r_partial={rp:+.4f} (p={pp:.2e}). "
                   f"[H,M] viable at large N.")
    elif abs(rp) > 0.10 and pp < 0.05:
        verdict = (f"WEAKENED at N={N_PRIMARY}: r_partial={rp:+.4f} (p={pp:.2e}). "
                   f"Signal present but weaker than N=2000.")
    else:
        verdict = (f"DIES at N={N_PRIMARY}: r_partial={rp:+.4f} (p={pp:.2e}). "
                   f"[H,M] is a finite-size effect like Fiedler.")

    if signal_grows:
        verdict += " | Signal stable/growing with N."

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
            "profile": "quadrupole only",
        },
        "results_by_N": results_by_N,
        "scaling": {
            "N": N_VALUES,
            "r_partial_best_of_3": scaling_r,
            "per_observable": {k: [float(x) for x in v] for k, v in per_obs_scaling.items()},
            "signal_grows": bool(signal_grows),
        },
        "comparison": {
            "exp3_N2000_r": -0.6394,
            "exp6_fiedler_N10000_r": -0.2363,
            "exp5b_heat_trace_N10000_r": 0.8384,
        },
        "verdict": verdict,
        "wall_time_sec": total_time,
    }

    out_path = RESULTS_DIR / "exp14_d4_commutator_highN.json"
    save_experiment(meta, output, out_path)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
