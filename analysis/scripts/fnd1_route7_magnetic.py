"""
FND-1 Route 7: Magnetic (Hermitian) Laplacian on Causal Sets.

Preserves edge DIRECTION via phase: A_mag[i,j] = i if x_i < x_j.
L_mag = D - A_mag is Hermitian with real eigenvalues.

Scientific rigor (parity with top-k):
  - Pre-registered primary: mag_surplus (linear response)
  - Finite-size scaling: N = 500, 1000, 1500
  - BH multiple testing correction
  - Permutation test (200 permutations)
  - Cross-validation, Cohen's d
  - BD action as mediator
  - Linear + quadratic
  - Isotropic density control
  - Raw eigenvalues saved

Run with MKL:
  "C:/Users/youre/miniconda3/envs/sct-mkl/python.exe" analysis/scripts/fnd1_route7_magnetic.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from scipy import stats

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

N_VALUES = [500, 1000, 2000, 3000]
N_PRIMARY = 2000
M_ENSEMBLE = 80
T_DIAMOND = 1.0
MASTER_SEED = 42
EPSILON_ANISO = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
EPSILON_ISO = [0.25, 0.5]
N_PERMUTATIONS = 200
WORKERS = N_WORKERS

PRIMARY_METRIC = "mag_surplus"
PRIMARY_MODE = "linear"


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker_magnetic(args):
    """Compute magnetic Laplacian eigenvalues for one sprinkling."""
    seed_int, N, T, eps, mode = args
    V = T**2 / 2.0
    rho = N / V
    rng = np.random.default_rng(seed_int)

    if mode == "isotropic":
        pts, C = sprinkle_isotropic(N, eps, T, rng)
    elif eps == 0.0:
        pts, C = _sprinkle_flat(N, T, rng)
    else:
        pts, C = sprinkle_curved(N, eps, T, rng)

    total_causal = float(np.sum(C))

    # Magnetic adjacency
    A_mag = 1j * C - 1j * C.T
    A_undir = ((C + C.T) > 0).astype(np.float64)
    degrees = np.sum(A_undir, axis=1)

    # Magnetic and undirected Laplacians
    L_mag = np.diag(degrees) - A_mag
    L_undir = np.diag(degrees) - A_undir

    eigs_mag = np.linalg.eigvalsh(L_mag)
    eigs_undir = np.linalg.eigvalsh(L_undir)

    sorted_mag = np.sort(eigs_mag)
    sorted_undir = np.sort(eigs_undir)

    fiedler_mag = float(sorted_mag[1]) if len(sorted_mag) > 1 else 0
    fiedler_undir = float(sorted_undir[1]) if len(sorted_undir) > 1 else 0
    mag_surplus = fiedler_mag - fiedler_undir

    abs_eigs = np.abs(eigs_mag)
    s = np.sum(abs_eigs)
    entropy = float(-np.sum((abs_eigs/s)*np.log(abs_eigs/s + 1e-300))) if s > 0 else 0

    # BD action
    n_mat = compute_interval_cardinalities(C)
    past = C.T; n_past = n_mat.T
    N1 = int(np.sum((past > 0) & (n_past == 0)))
    N2 = int(np.sum((past > 0) & (n_past == 1)))
    N3 = int(np.sum((past > 0) & (n_past == 2)))
    bd_action = float(N - 2*N1 + 4*N2 - 2*N3)

    return {
        "eigs_mag": eigs_mag,
        "fiedler": fiedler_mag,
        "fiedler_undir": fiedler_undir,
        "mag_surplus": float(mag_surplus),
        "entropy": entropy,
        "frobenius": float(np.sqrt(np.sum(eigs_mag**2))),
        "spectral_gap": float(sorted_mag[-1] - sorted_mag[1]) if len(sorted_mag) > 1 else 0,
        "total_causal": total_causal,
        "bd_action": bd_action,
        "eps": eps, "mode": mode,
    }


# ---------------------------------------------------------------------------
# Mediation (same as top-k, full rigor)
# ---------------------------------------------------------------------------

def mediation_analysis(eps_arr, obs_arr, tc_arr, bd_arr=None, n_perm=0):
    """Full mediation with BD control, permutation, CV, Cohen's d."""
    eps2 = eps_arr**2
    n = len(eps_arr)

    def resid(x, c):
        s, i, _, _, _ = stats.linregress(c, x)
        return x - (s*c + i)

    def partial_r(x, y, c):
        xr, yr = resid(x, c), resid(y, c)
        if np.std(xr) > 0 and np.std(yr) > 0: return stats.pearsonr(xr, yr)
        return 0.0, 1.0

    def partial_r_multi(x, y, ctrls):
        X = np.column_stack([*ctrls, np.ones(n)])
        bx = np.linalg.lstsq(X, x, rcond=None)[0]
        by = np.linalg.lstsq(X, y, rcond=None)[0]
        xr, yr = x - X @ bx, y - X @ by
        if np.std(xr) > 0 and np.std(yr) > 0: return stats.pearsonr(xr, yr)
        return 0.0, 1.0

    r_lin, p_lin = stats.pearsonr(eps_arr, obs_arr)
    r_quad, p_quad = stats.pearsonr(eps2, obs_arr)
    r_pl, p_pl = partial_r(eps_arr, obs_arr, tc_arr)
    r_pq, p_pq = partial_r(eps2, obs_arr, tc_arr)
    r_bd, p_bd = partial_r_multi(eps_arr, obs_arr, [tc_arr, bd_arr]) if bd_arr is not None else (0, 1)
    r_qbd, p_qbd = partial_r_multi(eps2, obs_arr, [tc_arr, bd_arr]) if bd_arr is not None else (0, 1)

    flat = obs_arr[np.abs(eps_arr) < 0.01]
    curved = obs_arr[np.abs(eps_arr) > 0.4]
    d = 0.0
    if len(flat) > 2 and len(curved) > 2:
        ps = np.sqrt((np.var(flat, ddof=1) + np.var(curved, ddof=1))/2)
        d = (np.mean(curved) - np.mean(flat)) / ps if ps > 0 else 0

    half = n // 2
    cv1, _ = partial_r(eps_arr[:half], obs_arr[:half], tc_arr[:half])
    cv2, _ = partial_r(eps_arr[half:], obs_arr[half:], tc_arr[half:])

    p_perm = 1.0
    if n_perm > 0:
        obs_r = abs(r_pl)
        rng = np.random.default_rng(42)
        cnt = sum(1 for _ in range(n_perm)
                  if abs(partial_r(eps_arr[rng.permutation(n)], obs_arr, tc_arr)[0]) >= obs_r)
        p_perm = (cnt + 1) / (n_perm + 1)

    return {
        "r_lin": float(r_lin), "r_quad": float(r_quad),
        "r_partial_lin": float(r_pl), "p_partial_lin": float(p_pl),
        "r_partial_quad": float(r_pq), "p_partial_quad": float(p_pq),
        "r_partial_lin_bd": float(r_bd), "p_partial_lin_bd": float(p_bd),
        "r_partial_quad_bd": float(r_qbd), "p_partial_quad_bd": float(p_qbd),
        "cohens_d": float(d), "cv_consistent": bool(cv1*cv2 > 0), "p_perm": float(p_perm),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    print("=" * 70, flush=True)
    print("FND-1 ROUTE 7: MAGNETIC LAPLACIAN", flush=True)
    print("=" * 70, flush=True)
    print(f"N values: {N_VALUES} (primary: {N_PRIMARY})", flush=True)
    print(f"Pre-registered: {PRIMARY_METRIC} ({PRIMARY_MODE})", flush=True)
    print(flush=True)

    # Benchmark
    t0 = time.perf_counter()
    _worker_magnetic((42, N_PRIMARY, T_DIAMOND, 0.0, "anisotropic"))
    per_task = time.perf_counter() - t0
    n_tasks = (len(EPSILON_ANISO) + len(EPSILON_ISO)) * M_ENSEMBLE
    print(f"Benchmark: {per_task:.3f}s/task, {n_tasks} tasks, "
          f"~{n_tasks*per_task/WORKERS/60:.1f} min", flush=True)

    ss = np.random.SeedSequence(MASTER_SEED)

    # PART 1: Anisotropic
    print(f"\n{'='*60}\nPART 1: ANISOTROPIC\n{'='*60}", flush=True)
    aniso_data = []
    for eps in EPSILON_ANISO:
        seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(1)[0].spawn(M_ENSEMBLE)]
        args = [(si, N_PRIMARY, T_DIAMOND, eps, "anisotropic") for si in seeds]
        print(f"  eps={eps:+.3f}...", flush=True)
        t0 = time.perf_counter()
        with Pool(WORKERS, initializer=_init_worker) as pool:
            results = pool.map(_worker_magnetic, args)
        print(f"    Done {time.perf_counter()-t0:.1f}s, surplus={np.mean([r['mag_surplus'] for r in results]):.4f}", flush=True)
        aniso_data.extend(results)

    # PART 2: Isotropic
    print(f"\n{'='*60}\nPART 2: ISOTROPIC CONTROL\n{'='*60}", flush=True)
    iso_data = []
    for eps in EPSILON_ISO:
        seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(1)[0].spawn(M_ENSEMBLE)]
        args = [(si, N_PRIMARY, T_DIAMOND, eps, "isotropic") for si in seeds]
        print(f"  eps_iso={eps:+.3f}...", flush=True)
        t0 = time.perf_counter()
        with Pool(WORKERS, initializer=_init_worker) as pool:
            results = pool.map(_worker_magnetic, args)
        print(f"    Done {time.perf_counter()-t0:.1f}s", flush=True)
        iso_data.extend(results)

    # ANALYSIS
    print(f"\n{'='*60}\nANALYSIS\n{'='*60}", flush=True)
    metrics = ["fiedler", "mag_surplus", "entropy", "frobenius", "spectral_gap"]
    eps_a = np.array([d["eps"] for d in aniso_data])
    tc_a = np.array([d["total_causal"] for d in aniso_data])
    bd_a = np.array([d["bd_action"] for d in aniso_data])

    med_results = {}
    all_pvals, all_labels = [], []

    for metric in metrics:
        obs = np.array([d[metric] for d in aniso_data])
        n_p = N_PERMUTATIONS if metric == PRIMARY_METRIC else 0
        med = mediation_analysis(eps_a, obs, tc_a, bd_arr=bd_a, n_perm=n_p)
        med_results[metric] = med
        all_pvals.extend([med["p_partial_lin"], med["p_partial_quad"], med["p_partial_lin_bd"], med["p_partial_quad_bd"]])
        all_labels.extend([f"{metric}_lin", f"{metric}_quad", f"{metric}_lin_bd", f"{metric}_quad_bd"])
        surv = abs(med["r_partial_lin_bd"]) > 0.1 and med["p_partial_lin_bd"] < 0.10
        print(f"  {metric:>15}: lin={med['r_lin']:+.4f}, quad={med['r_quad']:+.4f}, "
              f"p_l|BD={med['p_partial_lin_bd']:.2e}, d={med['cohens_d']:+.3f}, "
              f"cv={med['cv_consistent']}, perm={med['p_perm']:.3f} "
              f"[{'SURV' if surv else ''}]", flush=True)

    # Iso
    eps_iso = np.array([d["eps"] for d in iso_data])
    obs_iso = np.array([d[PRIMARY_METRIC] for d in iso_data])
    r_iso, p_iso = stats.pearsonr(eps_iso, obs_iso) if len(eps_iso) > 5 else (0, 1)
    print(f"\n  Isotropic {PRIMARY_METRIC}: r={r_iso:+.4f}, p={p_iso:.2e}", flush=True)

    # BH
    n_t = len(all_pvals)
    si = np.argsort(all_pvals)
    bh = sum(1 for i, s in enumerate(si) if all_pvals[s] <= (i+1)/n_t*0.05)
    print(f"  BH: {n_t} tests, {bh} significant", flush=True)

    # SCALING
    print(f"\n{'='*60}\nFINITE-SIZE SCALING\n{'='*60}", flush=True)
    scaling = {}
    for N_test in N_VALUES:
        sc_data = []
        for eps in EPSILON_ANISO:
            seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(1)[0].spawn(M_ENSEMBLE//2)]
            args = [(si, N_test, T_DIAMOND, eps, "anisotropic") for si in seeds]
            with Pool(WORKERS, initializer=_init_worker) as pool:
                sc_data.extend(pool.map(_worker_magnetic, args))
        sc_e = np.array([d["eps"] for d in sc_data])
        sc_o = np.array([d[PRIMARY_METRIC] for d in sc_data])
        sc_t = np.array([d["total_causal"] for d in sc_data])
        sc_b = np.array([d["bd_action"] for d in sc_data])
        sc_m = mediation_analysis(sc_e, sc_o, sc_t, bd_arr=sc_b)
        scaling[N_test] = {"r_lin": sc_m["r_lin"], "r_partial_lin_bd": sc_m["r_partial_lin_bd"]}
        print(f"  N={N_test}: r={sc_m['r_lin']:+.4f}, partial_bd={sc_m['r_partial_lin_bd']:+.4f}", flush=True)

    # VERDICT
    total_time = time.perf_counter() - t_total
    best = med_results[PRIMARY_METRIC]
    sig = abs(best["r_partial_lin_bd"]) > 0.1 and best["p_partial_lin_bd"] < 0.10
    iso_weak = abs(r_iso) < abs(best["r_lin"]) * 0.5

    if sig and iso_weak:
        verdict = "MAGNETIC LAPLACIAN DETECTS CURVATURE (survives TC+BD, iso weaker)"
    elif sig:
        verdict = "SIGNAL (survives mediation)"
    elif abs(best["r_lin"]) > 0.3:
        verdict = "DIRECT ONLY (mediated)"
    else:
        verdict = "NO SIGNAL"

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  Wall time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    meta = ExperimentMeta(route=7, name="route7_magnetic", N=N_PRIMARY, M=M_ENSEMBLE,
                          status="completed", verdict=verdict)
    meta.wall_time_sec = total_time

    eig_save = {}
    for eps in EPSILON_ANISO:
        d = [x for x in aniso_data if abs(x["eps"] - eps) < 0.01]
        eig_save[str(eps)] = {"mag_eigenvalues": [x["eigs_mag"].tolist() for x in d],
                              "total_causal": [x["total_causal"] for x in d],
                              "bd_action": [x["bd_action"] for x in d]}
    with open(RESULTS_DIR / "route7_eigenvalues.json", "w") as f:
        json.dump(eig_save, f)

    output = {"parameters": {"N_values": N_VALUES, "M": M_ENSEMBLE},
              "mediation": med_results, "iso_r": float(r_iso), "scaling": scaling,
              "verdict": verdict, "wall_time_sec": total_time}
    save_experiment(meta, output, RESULTS_DIR / "route7_magnetic_laplacian.json")
    clear_progress()


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        t0 = time.perf_counter()
        _worker_magnetic((42, N_PRIMARY, T_DIAMOND, 0.0, "anisotropic"))
        print(f"Benchmark: {time.perf_counter()-t0:.3f}s/task")
    else:
        main()
