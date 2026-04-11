"""
FND-1 EXP-8: d=2 Ollivier-Ricci on Hasse Diagram + Extended Density Controls.

Route 4 tested OR curvature on the FULL causal graph (degree ~125).
The Hasse diagram (links only, degree ~8) is geometrically more appropriate.

Also adds higher density controls (density variance, spatial gradient)
to definitively close d=2: if partial r ~ 0 with extended controls,
no spectral method extracts curvature beyond BD in d=2.

Run:
  python analysis/scripts/fnd1_exp8_d2_hasse_ricci.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from scipy import stats
from scipy.optimize import linprog
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_ensemble_runner import (
    sprinkle_diamond,
    compute_interval_cardinalities,
    build_bd_L,
)
from fnd1_gate5_runner import sprinkle_curved
from fnd1_route2_link_geometry import build_link_adjacency, build_link_laplacian
from fnd1_experiment_registry import (
    ExperimentMeta, save_experiment, RESULTS_DIR,
)
from fnd1_parallel import N_WORKERS, _init_worker

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_VALUES = [500, 1000, 2000]
N_PRIMARY = 1000
M_ENSEMBLE = 80
T_DIAMOND = 1.0
MASTER_SEED = 5599
WORKERS = N_WORKERS

EPS_VALUES = [-0.5, -0.25, 0.0, 0.25, 0.5]
N_EDGES_SAMPLE = 200   # edges to sample for OR curvature (link graph has ~8N/2 edges)


# ---------------------------------------------------------------------------
# Ollivier-Ricci on Hasse diagram
# ---------------------------------------------------------------------------

def ollivier_ricci_hasse(A_link_dense, n_edges, rng):
    """Compute OR curvature on sampled edges of the LINK graph (Hasse diagram).

    Much faster than full causal graph because link degree ~ 8 (not ~125).
    LP cost ~ support_size^4, with support ~16 instead of ~250.
    """
    N = A_link_dense.shape[0]

    # Shortest path on link graph
    dist = shortest_path(A_link_dense, method='D', unweighted=True)
    dist[dist == np.inf] = N

    edge_i, edge_j = np.where(np.triu(A_link_dense, k=1) > 0)
    n_total = len(edge_i)
    if n_total == 0:
        return np.array([])

    n_sample = min(n_edges, n_total)
    idx = rng.choice(n_total, size=n_sample, replace=False)

    curvatures = []
    for ei, ej in zip(edge_i[idx], edge_j[idx]):
        d_ij = dist[ei, ej]
        if d_ij <= 0 or d_ij >= N:
            continue

        nbr_i = np.where(A_link_dense[ei] > 0)[0]
        nbr_j = np.where(A_link_dense[ej] > 0)[0]
        if len(nbr_i) == 0 or len(nbr_j) == 0:
            continue

        mu_i = np.zeros(N); mu_i[nbr_i] = 1.0 / len(nbr_i)
        mu_j = np.zeros(N); mu_j[nbr_j] = 1.0 / len(nbr_j)

        support = np.unique(np.concatenate([nbr_i, nbr_j]))
        n_s = len(support)
        if n_s <= 1:
            curvatures.append(1.0)
            continue

        cost = dist[np.ix_(support, support)]
        p, q = mu_i[support], mu_j[support]
        n_var = n_s * n_s

        A_eq = np.zeros((2 * n_s, n_var))
        for k in range(n_s):
            A_eq[k, k * n_s:(k + 1) * n_s] = 1
            A_eq[n_s + k, k::n_s] = 1
        b_eq = np.concatenate([p, q])

        try:
            result = linprog(cost.flatten(), A_eq=A_eq, b_eq=b_eq,
                             bounds=[(0, None)] * n_var, method='highs')
            if result.success:
                curvatures.append(float(1.0 - result.fun / d_ij))
        except Exception:
            pass

    return np.array(curvatures)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(args):
    """Compute Hasse OR curvature + density controls for one d=2 sprinkling."""
    seed_int, N, T, eps = args
    V = T ** 2 / 2.0
    rho = N / V

    rng = np.random.default_rng(seed_int)
    if abs(eps) < 1e-12:
        pts, C = sprinkle_diamond(N, T, rng)
    else:
        pts, C = sprinkle_curved(N, eps, T, rng)

    total_causal = float(np.sum(C))
    n_mat = compute_interval_cardinalities(C)

    # BD action (d=2)
    past = C.T; n_past = n_mat.T
    N1 = int(np.sum((past > 0) & (n_past == 0)))
    N2 = int(np.sum((past > 0) & (n_past == 1)))
    N3 = int(np.sum((past > 0) & (n_past == 2)))
    bd_action = float(N - 2 * N1 + 4 * N2 - 2 * N3)

    # Link graph
    A_link = build_link_adjacency(C, n_mat)
    A_link_dense = A_link.toarray()
    mean_link_deg = float(A_link.sum() / N)

    # Ollivier-Ricci on Hasse diagram
    rng_or = np.random.default_rng(seed_int + 3333)
    kappas = ollivier_ricci_hasse(A_link_dense, N_EDGES_SAMPLE, rng_or)

    if len(kappas) > 0:
        mean_kappa = float(np.mean(kappas))
        std_kappa = float(np.std(kappas, ddof=1))
        median_kappa = float(np.median(kappas))
    else:
        mean_kappa = std_kappa = median_kappa = 0.0

    # Extended density controls: spatial statistics
    t_coords = pts[:, 0]
    x_coords = pts[:, 1]
    # Density variance: divide diamond into 4 quadrants, count points
    n_q1 = np.sum((t_coords > 0) & (x_coords > 0))
    n_q2 = np.sum((t_coords > 0) & (x_coords <= 0))
    n_q3 = np.sum((t_coords <= 0) & (x_coords > 0))
    n_q4 = np.sum((t_coords <= 0) & (x_coords <= 0))
    density_var = float(np.var([n_q1, n_q2, n_q3, n_q4]))
    # Spatial gradient: mean x-coordinate (nonzero for asymmetric density)
    mean_x = float(np.mean(x_coords))
    mean_t = float(np.mean(t_coords))

    return {
        "mean_kappa": mean_kappa,
        "median_kappa": median_kappa,
        "std_kappa": std_kappa,
        "n_valid_edges": len(kappas),
        "total_causal": total_causal,
        "bd_action": bd_action,
        "mean_link_deg": mean_link_deg,
        "density_var": density_var,
        "mean_x": mean_x,
        "mean_t": mean_t,
        "eps": eps,
    }


# ---------------------------------------------------------------------------
# Mediation
# ---------------------------------------------------------------------------

def partial_corr(x, y, controls):
    from numpy.linalg import lstsq
    if controls.shape[1] == 0:
        return stats.pearsonr(x, y)
    cx, _, _, _ = lstsq(controls, x, rcond=None)
    cy, _, _, _ = lstsq(controls, y, rcond=None)
    rx = x - controls @ cx
    ry = y - controls @ cy
    if np.std(rx) < 1e-15 or np.std(ry) < 1e-15:
        return 0.0, 1.0
    return stats.pearsonr(rx, ry)


def mediation(results):
    """Test mean_kappa vs eps with standard AND extended density controls."""
    eps_arr = np.array([r["eps"] for r in results])
    kappa_arr = np.array([r["mean_kappa"] for r in results])
    tc_arr = np.array([r["total_causal"] for r in results])
    bd_arr = np.array([r["bd_action"] for r in results])
    dv_arr = np.array([r["density_var"] for r in results])
    mx_arr = np.array([r["mean_x"] for r in results])

    out = {"n": len(results)}

    # Standard controls: TC + TC^2 + BD
    ctrl_std = np.column_stack([tc_arr, tc_arr ** 2, bd_arr, np.ones(len(tc_arr))])
    r_d, p_d = stats.pearsonr(eps_arr, kappa_arr)
    r_std, p_std = partial_corr(eps_arr, kappa_arr, ctrl_std)
    out["r_direct"] = float(r_d)
    out["r_partial_standard"] = float(r_std)
    out["p_partial_standard"] = float(p_std)

    # Extended controls: TC + TC^2 + BD + density_var + mean_x
    ctrl_ext = np.column_stack([tc_arr, tc_arr ** 2, bd_arr, dv_arr, mx_arr, np.ones(len(tc_arr))])
    r_ext, p_ext = partial_corr(eps_arr, kappa_arr, ctrl_ext)
    out["r_partial_extended"] = float(r_ext)
    out["p_partial_extended"] = float(p_ext)

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    meta = ExperimentMeta(
        route=4, name="exp8_d2_hasse_ricci",
        description="d=2 Ollivier-Ricci on Hasse diagram + extended density controls",
        N=N_PRIMARY, M=M_ENSEMBLE, status="running",
    )

    print("=" * 70, flush=True)
    print("FND-1 EXP-8: d=2 HASSE RICCI + DENSITY CONTROLS", flush=True)
    print("=" * 70, flush=True)
    print(f"N values: {N_VALUES}, M={M_ENSEMBLE}", flush=True)
    print(f"Eps: {EPS_VALUES}, edges sampled: {N_EDGES_SAMPLE}", flush=True)
    print(flush=True)

    # Benchmark
    print("=== BENCHMARK ===", flush=True)
    for N in N_VALUES:
        t0 = time.perf_counter()
        _worker((42, N, T_DIAMOND, 0.0))
        dt = time.perf_counter() - t0
        tasks = M_ENSEMBLE * len(EPS_VALUES)
        par = tasks * dt / WORKERS
        print(f"  N={N:5d}: {dt:.3f}s/task, parallel: {par / 60:.1f} min", flush=True)

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

            kappas = [r["mean_kappa"] for r in raw]
            print(f"  eps={eps:+.3f}: kappa={np.mean(kappas):+.4f}+-{np.std(kappas):.4f}"
                  f"  [{elapsed:.1f}s]", flush=True)
            all_results.extend(raw)

        med = mediation(all_results)
        results_by_N[str(N)] = {
            "mediation": med,
            "n_sprinklings": len(all_results),
        }
        print(f"  Standard:  r_partial={med['r_partial_standard']:+.4f}"
              f" p={med['p_partial_standard']:.3f}", flush=True)
        print(f"  Extended:  r_partial={med['r_partial_extended']:+.4f}"
              f" p={med['p_partial_extended']:.3f}", flush=True)

    # ==================================================================
    # VERDICT
    # ==================================================================

    total_time = time.perf_counter() - t_total
    med_p = results_by_N[str(N_PRIMARY)]["mediation"]

    if abs(med_p["r_partial_extended"]) > 0.10 and med_p["p_partial_extended"] < 0.01:
        verdict = (f"SIGNAL: r_partial_ext={med_p['r_partial_extended']:+.4f}"
                   f" (Hasse Ricci detects curvature beyond density)")
    elif abs(med_p["r_partial_standard"]) > 0.10 and med_p["p_partial_standard"] < 0.01:
        verdict = (f"STANDARD ONLY: r_std={med_p['r_partial_standard']:+.4f}"
                   f" but r_ext={med_p['r_partial_extended']:+.4f} (density explains)")
    else:
        verdict = (f"d=2 CLOSED: r_std={med_p['r_partial_standard']:+.4f},"
                   f" r_ext={med_p['r_partial_extended']:+.4f} (both ~0)")

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
            "n_edges_sample": N_EDGES_SAMPLE,
        },
        "results_by_N": results_by_N,
        "verdict": verdict,
        "wall_time_sec": total_time,
    }

    out_path = RESULTS_DIR / "exp8_d2_hasse_ricci.json"
    save_experiment(meta, output, out_path)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
