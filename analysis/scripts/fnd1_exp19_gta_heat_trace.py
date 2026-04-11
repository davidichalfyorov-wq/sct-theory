"""
FND-1 EXP-19: GTA Heat Trace — Tr(e^{-τ·|[H,M]|}).

Instead of eigenvalue statistics (entropy, Frobenius), compute the heat trace
of the absolute commutator matrix. This may converge better than individual
eigenvalue statistics and connect directly to the spectral action.

The GTA heat trace K_GTA(τ) = (1/N) Tr(e^{-τ·|[H,M]|}) should show:
- τ^{d/2} · K_GTA → plateau (Weyl law for |[H,M]|)
- Plateau curvature dependence (if GTA encodes geometry)

Uses d=4 quadrupole only (density-blind).

Run:
    python analysis/scripts/fnd1_exp19_gta_heat_trace.py
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys, time, json
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
from fnd1_experiment_registry import ExperimentMeta, save_experiment, RESULTS_DIR
from fnd1_parallel import N_WORKERS, _init_worker

try:
    from fnd1_gpu import gpu_eigvalsh
except ImportError:
    gpu_eigvalsh = np.linalg.eigvalsh

N_VALUES = [500, 1000, 2000]
M_ENSEMBLE = 60
T_DIAMOND = 1.0
MASTER_SEED = 1919
WORKERS = N_WORKERS
EPS_VALUES = [0.0, 2.0, 5.0, 10.0]

# Heat trace tau grid
N_TAU = 50
TAU_MIN = 1e-6
TAU_MAX = 1e-2  # small tau for commutator (eigenvalues are large)
TAU_GRID = np.logspace(np.log10(TAU_MIN), np.log10(TAU_MAX), N_TAU)

def build_bd_L_4d(C, n_matrix, rho):
    past = C.T; n_past = n_matrix.T
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

def _worker(args):
    seed_int, N, T, eps = args
    rng = np.random.default_rng(seed_int)
    pts = sprinkle_4d_flat(N, T, rng)

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
        del dx, dy, dz, dr2, xm, ym, f_mid, mink, corr

    total_causal = float(np.sum(C))
    V = np.pi * T ** 4 / 24.0
    rho = N / V
    n_matrix, N0, N1, N2, N3 = compute_layers_4d(C)
    bd = bd_action_4d(N, N0, N1, N2, N3)

    L = build_bd_L_4d(C, n_matrix, rho)
    del C, n_matrix

    L_sp = sp.csr_matrix(L)
    comm = ((L_sp.T @ L_sp - L_sp @ L_sp.T) / 2.0).toarray()
    comm = (comm + comm.T) / 2.0
    del L, L_sp
    comm_eigs = gpu_eigvalsh(comm)
    del comm

    # Absolute eigenvalues for heat trace
    abs_eigs = np.abs(comm_eigs)
    nz = abs_eigs[abs_eigs > 1e-10]

    # GTA heat trace: K(τ) = (1/N) Σ exp(-τ·|λ_k|)
    if len(nz) > 0:
        K = np.sum(np.exp(-TAU_GRID[:, None] * nz[None, :]), axis=1) / N
    else:
        K = np.zeros(N_TAU)

    # τ² · K (d=4 Weyl law analogue)
    t2K = TAU_GRID ** 2 * K

    # Sample 10 tau points for compact storage
    sample_idx = np.linspace(0, N_TAU - 1, 10, dtype=int)
    t2K_samples = {f"{TAU_GRID[i]:.2e}": float(t2K[i]) for i in sample_idx}

    # GTA eigenvalue statistics
    comm_entropy = 0.0
    if np.sum(abs_eigs) > 0:
        p = abs_eigs / np.sum(abs_eigs)
        comm_entropy = float(-np.sum(p * np.log(p + 1e-300)))

    return {
        "t2K_samples": t2K_samples,
        "t2K_max": float(np.max(t2K)) if len(t2K) > 0 else 0.0,
        "comm_entropy": comm_entropy,
        "n_nonzero_eigs": int(len(nz)),
        "max_eig": float(np.max(abs_eigs)) if len(abs_eigs) > 0 else 0.0,
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
    meta = ExperimentMeta(route=3, name="exp19_gta_heat_trace",
        description="d=4 GTA heat trace: Tr(e^{-τ|[H,M]|}), exploratory spectral probe",
        N=max(N_VALUES), M=M_ENSEMBLE, status="running")

    print("=" * 70)
    print("FND-1 EXP-19: GTA HEAT TRACE — Tr(e^{-τ|[H,M]|})")
    print("=" * 70)
    print(f"N: {N_VALUES}, M={M_ENSEMBLE}, eps: {EPS_VALUES}")
    print(f"τ range: [{TAU_MIN}, {TAU_MAX}], {N_TAU} points")
    print(f"Quadrupole only. Per-sprinkling mediation (not aggregate).")
    print()

    # Pre-spawn all seeds deterministically
    ss = np.random.SeedSequence(MASTER_SEED)
    all_children = ss.spawn(len(N_VALUES) * len(EPS_VALUES) * M_ENSEMBLE)
    child_idx = 0

    results_by_N = {}

    for N in N_VALUES:
        print(f"\nN = {N}")
        all_sprinklings = []  # collect per-sprinkling data for mediation

        for eps in EPS_VALUES:
            seed_ints = [int(all_children[child_idx + i].generate_state(1)[0])
                         for i in range(M_ENSEMBLE)]
            child_idx += M_ENSEMBLE
            args = [(si, N, T_DIAMOND, eps) for si in seed_ints]

            t0 = time.perf_counter()
            with Pool(WORKERS, initializer=_init_worker) as pool:
                raw = pool.map(_worker, args)
            elapsed = time.perf_counter() - t0

            t2K_max_mean = float(np.mean([r["t2K_max"] for r in raw]))
            print(f"  eps={eps:+5.1f}: t2K_max={t2K_max_mean:.6e}  [{elapsed:.1f}s]")
            all_sprinklings.extend(raw)

        # Per-sprinkling mediation: t2K_max and comm_entropy vs eps
        eps_arr = np.array([r["eps"] for r in all_sprinklings])
        tc_arr = np.array([r["total_causal"] for r in all_sprinklings])
        bd_arr = np.array([r["bd_action"] for r in all_sprinklings])
        controls = np.column_stack([tc_arr, tc_arr ** 2, bd_arr, np.ones(len(all_sprinklings))])

        mediation = {"n": len(all_sprinklings)}
        best_r = 0.0; best_obs = "t2K_max"; best_p = 1.0

        for obs_name in ["t2K_max", "comm_entropy"]:
            obs_arr = np.array([r[obs_name] for r in all_sprinklings])
            r_d, _ = stats.pearsonr(eps_arr, obs_arr)
            r_p, p_p = partial_corr(eps_arr, obs_arr, controls)
            mediation[obs_name] = {
                "r_direct": round(float(r_d), 4),
                "r_partial": round(float(r_p), 4),
                "p_partial": float(p_p),
            }
            if abs(r_p) > abs(best_r):
                best_r = float(r_p); best_obs = obs_name; best_p = float(p_p)
            print(f"  {obs_name}: r_direct={r_d:+.4f}, r_partial={r_p:+.4f} (p={p_p:.2e})")

        mediation["best_observable"] = best_obs
        mediation["best_r_partial"] = best_r
        mediation["best_p_partial"] = best_p

        results_by_N[str(N)] = {"mediation": mediation, "n_sprinklings": len(all_sprinklings)}

    total_time = time.perf_counter() - t_total
    best_N = results_by_N[str(max(N_VALUES))]
    rp = best_N["mediation"]["best_r_partial"]
    pp = best_N["mediation"]["best_p_partial"]
    obs = best_N["mediation"]["best_observable"]

    if abs(rp) > 0.30 and pp < 0.01:
        verdict = f"CURVATURE SENSITIVE: {obs} r_partial={rp:+.4f} (p={pp:.2e})"
    elif abs(rp) > 0.10 and pp < 0.05:
        verdict = f"WEAK SIGNAL: {obs} r_partial={rp:+.4f} (p={pp:.2e})"
    else:
        verdict = f"INCONCLUSIVE: {obs} r_partial={rp:+.4f} (p={pp:.2e})"

    print(f"\nVERDICT: {verdict}")
    print(f"Wall time: {total_time:.0f}s ({total_time/60:.1f} min)")

    meta.status = "completed"; meta.verdict = verdict; meta.wall_time_sec = total_time
    output = {
        "parameters": {"N_values": N_VALUES, "M": M_ENSEMBLE, "eps_values": EPS_VALUES,
                        "tau_range": [TAU_MIN, TAU_MAX, N_TAU], "profile": "quadrupole"},
        "results_by_N": results_by_N, "verdict": verdict, "wall_time_sec": total_time,
    }
    save_experiment(meta, output, RESULTS_DIR / "exp19_gta_heat_trace.json")
    print(f"Saved: exp19_gta_heat_trace.json")

if __name__ == "__main__":
    main()
