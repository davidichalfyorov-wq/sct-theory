"""
FND-1 Route 4: Ollivier-Ricci Curvature on Causal Sets.

Discrete Ricci curvature via optimal transport between neighborhoods.
No BD operator, no eigenvalues. Pure graph-theoretic curvature.
Never applied to causal sets before.

Scientific rigor (parity with top-k):
  - Pre-registered primary: mean_kappa (linear response)
  - Finite-size scaling: N = 300, 500, 800
  - BH multiple testing correction
  - Permutation test (200 permutations)
  - Cross-validation (split-half)
  - Cohen's d
  - BD action as mediator
  - Linear + quadratic response
  - Isotropic density control
  - Benchmark before estimates

Run with MKL:
  "C:/Users/youre/miniconda3/envs/sct-mkl/python.exe" analysis/scripts/fnd1_route4_ricci.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from scipy import stats
from scipy.optimize import linprog
from scipy.sparse.csgraph import shortest_path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_ensemble_runner import compute_interval_cardinalities, build_bd_L
from fnd1_gate5_runner import sprinkle_curved, _sprinkle_flat
from fnd1_route3_commutator_v2 import sprinkle_isotropic
from fnd1_experiment_registry import (
    ExperimentMeta, update_progress, clear_progress, save_experiment, RESULTS_DIR,
)
from fnd1_parallel import N_WORKERS, _init_worker

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_VALUES = [150, 250, 400]
N_PRIMARY = 250
M_ENSEMBLE = 60
T_DIAMOND = 1.0
MASTER_SEED = 42
EPSILON_ANISO = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
EPSILON_ISO = [0.25, 0.5]
N_EDGES_SAMPLE = 100       # reduced: LP cost ~ n_s^4 per edge
N_PERMUTATIONS = 200
MAX_SUPPORT = 200           # cap support size for LP feasibility
WORKERS = N_WORKERS

PRIMARY_METRIC = "mean_kappa"
PRIMARY_MODE = "linear"


# ---------------------------------------------------------------------------
# Ollivier-Ricci curvature
# ---------------------------------------------------------------------------

def compute_ollivier_ricci(C, n_edges=150, rng=None):
    """Compute Ollivier-Ricci curvature on sampled edges."""
    if rng is None:
        rng = np.random.default_rng(42)

    N = C.shape[0]
    A = ((C + C.T) > 0).astype(np.float64)
    dist = shortest_path(A, method='D', unweighted=True)
    dist[dist == np.inf] = N

    edge_i, edge_j = np.where(np.triu(A, k=1) > 0)
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

        nbr_i = np.where(A[ei] > 0)[0]
        nbr_j = np.where(A[ej] > 0)[0]
        if len(nbr_i) == 0 or len(nbr_j) == 0:
            continue

        mu_i = np.zeros(N); mu_i[nbr_i] = 1.0 / len(nbr_i)
        mu_j = np.zeros(N); mu_j[nbr_j] = 1.0 / len(nbr_j)

        support = np.unique(np.concatenate([nbr_i, nbr_j]))
        n_s = len(support)
        if n_s <= 1:
            curvatures.append(1.0)
            continue
        # Cap support size for LP feasibility (LP cost ~ n_s^4)
        if n_s > MAX_SUPPORT:
            keep = np.sort(rng.choice(n_s, MAX_SUPPORT, replace=False))
            support = support[keep]
            n_s = MAX_SUPPORT

        cost = dist[np.ix_(support, support)]
        p, q = mu_i[support], mu_j[support]
        n_var = n_s * n_s

        A_eq = np.zeros((2 * n_s, n_var))
        for k in range(n_s):
            A_eq[k, k*n_s:(k+1)*n_s] = 1
            A_eq[n_s+k, k::n_s] = 1
        b_eq = np.concatenate([p, q])

        try:
            result = linprog(cost.flatten(), A_eq=A_eq, b_eq=b_eq,
                           bounds=[(0, None)]*n_var, method='highs')
            if result.success:
                curvatures.append(float(1.0 - result.fun / d_ij))
        except Exception:
            pass

    return np.array(curvatures)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker_ricci(args):
    """Compute Ollivier-Ricci for one sprinkling."""
    seed_int, N, T, eps, n_edges, mode = args
    rng = np.random.default_rng(seed_int)

    if mode == "isotropic":
        pts, C = sprinkle_isotropic(N, eps, T, rng)
    elif eps == 0.0:
        pts, C = _sprinkle_flat(N, T, rng)
    else:
        pts, C = sprinkle_curved(N, eps, T, rng)

    curvatures = compute_ollivier_ricci(C, n_edges=n_edges, rng=rng)
    total_causal = float(np.sum(C))

    # BD action
    n_mat = compute_interval_cardinalities(C)
    past = C.T; n_past = n_mat.T
    N1 = int(np.sum((past > 0) & (n_past == 0)))
    N2 = int(np.sum((past > 0) & (n_past == 1)))
    N3 = int(np.sum((past > 0) & (n_past == 2)))
    bd_action = float(N - 2*N1 + 4*N2 - 2*N3)

    result = {"total_causal": total_causal, "bd_action": bd_action,
              "eps": eps, "mode": mode, "n_computed": len(curvatures)}
    if len(curvatures) > 0:
        result["mean_kappa"] = float(np.mean(curvatures))
        result["std_kappa"] = float(np.std(curvatures))
        result["median_kappa"] = float(np.median(curvatures))
    else:
        result["mean_kappa"] = 0.0
        result["std_kappa"] = 0.0
        result["median_kappa"] = 0.0
    return result


# ---------------------------------------------------------------------------
# Mediation (shared with top-k)
# ---------------------------------------------------------------------------

def mediation_analysis(eps_arr, obs_arr, tc_arr, bd_arr=None, n_perm=0):
    """Full mediation: linear + quadratic, BD control, permutation, CV, Cohen's d."""
    eps2 = eps_arr**2
    n = len(eps_arr)

    def resid(x, c):
        s, i, _, _, _ = stats.linregress(c, x)
        return x - (s*c + i)

    def partial_r(x, y, c):
        xr, yr = resid(x, c), resid(y, c)
        if np.std(xr) > 0 and np.std(yr) > 0:
            return stats.pearsonr(xr, yr)
        return 0.0, 1.0

    def partial_r_multi(x, y, ctrls):
        X = np.column_stack([*ctrls, np.ones(len(x))])
        bx = np.linalg.lstsq(X, x, rcond=None)[0]
        by = np.linalg.lstsq(X, y, rcond=None)[0]
        xr, yr = x - X @ bx, y - X @ by
        if np.std(xr) > 0 and np.std(yr) > 0:
            return stats.pearsonr(xr, yr)
        return 0.0, 1.0

    r_lin, p_lin = stats.pearsonr(eps_arr, obs_arr)
    r_quad, p_quad = stats.pearsonr(eps2, obs_arr)
    r_partial_lin, p_partial_lin = partial_r(eps_arr, obs_arr, tc_arr)
    r_partial_quad, p_partial_quad = partial_r(eps2, obs_arr, tc_arr)

    r_pq_bd, p_pq_bd = (0.0, 1.0)
    if bd_arr is not None:
        r_pq_bd, p_pq_bd = partial_r_multi(eps2, obs_arr, [tc_arr, bd_arr])

    # Cohen's d
    flat = obs_arr[np.abs(eps_arr) < 0.01]
    curved = obs_arr[np.abs(eps_arr) > 0.4]
    if len(flat) > 2 and len(curved) > 2:
        ps = np.sqrt((np.var(flat, ddof=1) + np.var(curved, ddof=1)) / 2)
        d = (np.mean(curved) - np.mean(flat)) / ps if ps > 0 else 0
    else:
        d = 0.0

    # Cross-validation
    half = n // 2
    cv1, _ = partial_r(eps_arr[:half], obs_arr[:half], tc_arr[:half])
    cv2, _ = partial_r(eps_arr[half:], obs_arr[half:], tc_arr[half:])
    cv_ok = (cv1 * cv2 > 0)

    # Permutation
    p_perm = 1.0
    if n_perm > 0:
        obs_r = abs(r_partial_lin)
        rng = np.random.default_rng(42)
        cnt = sum(1 for _ in range(n_perm)
                  if abs(partial_r(eps_arr[rng.permutation(n)], obs_arr, tc_arr)[0]) >= obs_r)
        p_perm = (cnt + 1) / (n_perm + 1)

    return {
        "r_lin": float(r_lin), "p_lin": float(p_lin),
        "r_quad": float(r_quad), "p_quad": float(p_quad),
        "r_partial_lin": float(r_partial_lin), "p_partial_lin": float(p_partial_lin),
        "r_partial_quad": float(r_partial_quad), "p_partial_quad": float(p_partial_quad),
        "r_pq_bd": float(r_pq_bd), "p_pq_bd": float(p_pq_bd),
        "cohens_d": float(d), "cv_consistent": bool(cv_ok), "p_perm": float(p_perm),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    print("=" * 70, flush=True)
    print("FND-1 ROUTE 4: OLLIVIER-RICCI CURVATURE", flush=True)
    print("=" * 70, flush=True)
    print(f"N values: {N_VALUES} (primary: {N_PRIMARY})", flush=True)
    print(f"Pre-registered: {PRIMARY_METRIC} ({PRIMARY_MODE})", flush=True)
    print(f"Permutations: {N_PERMUTATIONS}", flush=True)
    print(flush=True)

    # Benchmark
    t0 = time.perf_counter()
    _worker_ricci((42, N_PRIMARY, T_DIAMOND, 0.0, N_EDGES_SAMPLE, "anisotropic"))
    per_task = time.perf_counter() - t0
    n_tasks = (len(EPSILON_ANISO) + len(EPSILON_ISO)) * M_ENSEMBLE
    print(f"Benchmark: {per_task:.3f}s/task, {n_tasks} tasks, "
          f"parallel: {n_tasks*per_task/WORKERS/60:.1f} min", flush=True)

    ss = np.random.SeedSequence(MASTER_SEED)

    # PART 1: Anisotropic
    print(f"\n{'='*60}\nPART 1: ANISOTROPIC\n{'='*60}", flush=True)
    aniso_data = []
    for eps in EPSILON_ANISO:
        seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(1)[0].spawn(M_ENSEMBLE)]
        args = [(si, N_PRIMARY, T_DIAMOND, eps, N_EDGES_SAMPLE, "anisotropic") for si in seeds]
        print(f"  eps={eps:+.3f}...", flush=True)
        t0 = time.perf_counter()
        with Pool(WORKERS, initializer=_init_worker) as pool:
            results = pool.map(_worker_ricci, args)
        print(f"    Done {time.perf_counter()-t0:.1f}s, "
              f"kappa={np.mean([r['mean_kappa'] for r in results]):.6f}", flush=True)
        aniso_data.extend(results)

    # PART 2: Isotropic control
    print(f"\n{'='*60}\nPART 2: ISOTROPIC CONTROL\n{'='*60}", flush=True)
    iso_data = []
    for eps in EPSILON_ISO:
        seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(1)[0].spawn(M_ENSEMBLE)]
        args = [(si, N_PRIMARY, T_DIAMOND, eps, N_EDGES_SAMPLE, "isotropic") for si in seeds]
        print(f"  eps_iso={eps:+.3f}...", flush=True)
        t0 = time.perf_counter()
        with Pool(WORKERS, initializer=_init_worker) as pool:
            results = pool.map(_worker_ricci, args)
        print(f"    Done {time.perf_counter()-t0:.1f}s, "
              f"kappa={np.mean([r['mean_kappa'] for r in results]):.6f}", flush=True)
        iso_data.extend(results)

    # ANALYSIS
    print(f"\n{'='*60}\nANALYSIS\n{'='*60}", flush=True)

    metrics = ["mean_kappa", "median_kappa", "std_kappa"]
    eps_a = np.array([d["eps"] for d in aniso_data])
    tc_a = np.array([d["total_causal"] for d in aniso_data])
    bd_a = np.array([d["bd_action"] for d in aniso_data])

    med_results = {}
    all_pvals = []
    all_labels = []

    for metric in metrics:
        obs = np.array([d[metric] for d in aniso_data])
        n_p = N_PERMUTATIONS if metric == PRIMARY_METRIC else 0
        med = mediation_analysis(eps_a, obs, tc_a, bd_arr=bd_a, n_perm=n_p)
        med_results[metric] = med
        all_pvals.extend([med["p_partial_lin"], med["p_partial_quad"], med["p_pq_bd"]])
        all_labels.extend([f"{metric}_lin", f"{metric}_quad", f"{metric}_bd"])
        surv = abs(med["r_pq_bd"]) > 0.1 and med["p_pq_bd"] < 0.10
        print(f"  {metric:>15}: lin={med['r_lin']:+.4f}, quad={med['r_quad']:+.4f}, "
              f"p_q|BD={med['p_pq_bd']:.2e}, d={med['cohens_d']:+.3f}, "
              f"cv={med['cv_consistent']}, perm={med['p_perm']:.3f} "
              f"[{'SURV' if surv else ''}]", flush=True)

    # Iso comparison
    eps_iso = np.array([d["eps"] for d in iso_data])
    obs_iso = np.array([d[PRIMARY_METRIC] for d in iso_data])
    r_iso, p_iso = stats.pearsonr(eps_iso, obs_iso) if len(eps_iso) > 5 else (0, 1)
    print(f"\n  Isotropic: r_lin={r_iso:+.4f}, p={p_iso:.2e}", flush=True)

    # BH correction
    n_tests = len(all_pvals)
    sorted_idx = np.argsort(all_pvals)
    bh_thresh = np.array([(i+1)/n_tests*0.05 for i in range(n_tests)])
    bh_sig = sum(1 for i, si in enumerate(sorted_idx) if all_pvals[si] <= bh_thresh[i])
    print(f"\n  BH correction: {n_tests} tests, {bh_sig} significant at FDR=0.05", flush=True)

    # FINITE-SIZE SCALING
    print(f"\n{'='*60}\nFINITE-SIZE SCALING\n{'='*60}", flush=True)
    scaling = {}
    for N_test in N_VALUES:
        sc_data = []
        for eps in EPSILON_ANISO:
            seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(1)[0].spawn(M_ENSEMBLE//2)]
            args = [(si, N_test, T_DIAMOND, eps, N_EDGES_SAMPLE, "anisotropic") for si in seeds]
            with Pool(WORKERS, initializer=_init_worker) as pool:
                sc_data.extend(pool.map(_worker_ricci, args))
        sc_eps = np.array([d["eps"] for d in sc_data])
        sc_obs = np.array([d[PRIMARY_METRIC] for d in sc_data])
        sc_tc = np.array([d["total_causal"] for d in sc_data])
        sc_bd = np.array([d["bd_action"] for d in sc_data])
        sc_med = mediation_analysis(sc_eps, sc_obs, sc_tc, bd_arr=sc_bd)
        scaling[N_test] = {"r_lin": sc_med["r_lin"], "r_partial_lin": sc_med["r_partial_lin"]}
        print(f"  N={N_test}: r_lin={sc_med['r_lin']:+.4f}, partial={sc_med['r_partial_lin']:+.4f}", flush=True)

    # VERDICT
    total_time = time.perf_counter() - t_total
    best_med = med_results[PRIMARY_METRIC]
    aniso_sig = abs(best_med["r_pq_bd"]) > 0.1 and best_med["p_pq_bd"] < 0.10
    iso_weak = abs(r_iso) < abs(best_med["r_lin"]) * 0.5

    if aniso_sig and iso_weak:
        verdict = f"OLLIVIER-RICCI DETECTS CURVATURE (survives TC+BD, iso weaker)"
    elif aniso_sig:
        verdict = f"SIGNAL (survives mediation, iso not confirmed)"
    elif abs(best_med["r_lin"]) > 0.3:
        verdict = "DIRECT ONLY (mediated)"
    else:
        verdict = "NO SIGNAL"

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  Wall time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    meta = ExperimentMeta(route=4, name="route4_ricci", N=N_PRIMARY, M=M_ENSEMBLE,
                          status="completed", verdict=verdict)
    meta.wall_time_sec = total_time
    output = {"parameters": {"N_values": N_VALUES, "M": M_ENSEMBLE, "n_edges": N_EDGES_SAMPLE},
              "mediation": med_results, "iso_r": float(r_iso),
              "scaling": scaling, "verdict": verdict, "wall_time_sec": total_time}
    save_experiment(meta, output, RESULTS_DIR / "route4_ollivier_ricci.json")
    clear_progress()


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        t0 = time.perf_counter()
        _worker_ricci((42, N_PRIMARY, T_DIAMOND, 0.0, N_EDGES_SAMPLE, "anisotropic"))
        print(f"Benchmark: {time.perf_counter()-t0:.3f}s/task")
    else:
        main()
