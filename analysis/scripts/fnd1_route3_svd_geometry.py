"""
FND-1 Route 2+3 Synthesis: SVD Truncation -> Emergent Geometry.

Tests whether truncating the retarded BD operator L to its top-k singular
modes recovers spacetime geometry. If the SVD embedding distance between
points correlates with actual spacetime distance, then SVD truncation IS
the reconstruction map that Route 2 was looking for.

Method:
  1. Sprinkle N points, build BD operator L
  2. SVD: L = U S V^T
  3. Truncate to top-k: embed each point x_i -> phi(x_i) = (s_1*u_1(i), ..., s_k*u_k(i))
  4. Compute embedding distances: d_SVD(i,j) = ||phi(x_i) - phi(x_j)||
  5. Compute actual distances: d_true(i,j) (Minkowski or Euclidean after Wick rotation)
  6. Compare: Pearson r(d_SVD, d_true)? Does the embedding recover geometry?

If r > 0.5 at some k, SVD truncation produces a meaningful metric.

Run with MKL:
  "C:/Users/youre/miniconda3/envs/sct-mkl/python.exe" analysis/scripts/fnd1_route3_svd_geometry.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_ensemble_runner import (
    compute_interval_cardinalities,
    build_bd_L,
)
from fnd1_gate5_runner import sprinkle_curved, _sprinkle_flat
from fnd1_experiment_registry import (
    ExperimentMeta, update_progress, clear_progress, save_experiment, RESULTS_DIR,
)
from fnd1_parallel import N_WORKERS, _init_worker

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_POINTS = 2000        # increased for precision (SVD is O(N^3))
M_ENSEMBLE = 50        # sprinklings per condition (increased)
T_DIAMOND = 1.0
MASTER_SEED = 42
K_VALUES = [5, 10, 20, 50, 100, 200, 500]
EPSILON_VALUES = [-0.5, 0.0, 0.25, 0.5]  # negative, flat, and curved
WORKERS = N_WORKERS
N_DISTANCE_PAIRS = 5000       # random pairs for distance comparison (increased)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker_svd(args):
    """Compute SVD of L and distance correlations for one sprinkling."""
    seed_int, N, T, eps, k_values, n_pairs = args
    V = T**2 / 2.0
    rho = N / V

    rng = np.random.default_rng(seed_int)
    if eps == 0.0:
        pts, C = _sprinkle_flat(N, T, rng)
    else:
        pts, C = sprinkle_curved(N, eps, T, rng)

    n_mat = compute_interval_cardinalities(C)
    L = build_bd_L(C, n_mat, rho)

    # Full SVD (need U and S, not just S)
    U, S, Vt = np.linalg.svd(L, full_matrices=False)

    # True spacetime distances between random pairs
    rng2 = np.random.default_rng(seed_int + 999)
    idx_i = rng2.integers(0, N, size=n_pairs)
    idx_j = rng2.integers(0, N, size=n_pairs)
    # Avoid self-pairs
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    t_coords = pts[:, 0]
    x_coords = pts[:, 1]

    # Euclidean distance (proxy for geometry after Wick rotation)
    dt = t_coords[idx_i] - t_coords[idx_j]
    dx = x_coords[idx_i] - x_coords[idx_j]
    d_euclidean = np.sqrt(dt**2 + dx**2)

    # Lorentzian interval (can be imaginary for spacelike, take abs)
    d_lorentzian = np.sqrt(np.abs(dt**2 - dx**2))

    # Causal: are they causally related?
    is_causal = (np.abs(dt) > np.abs(dx)).astype(float)

    results_by_k = {}
    for k in k_values:
        if k > len(S):
            k = len(S)

        # SVD embedding: phi(x_i) = (s_1*u_1(i), ..., s_k*u_k(i))
        embedding = U[:, :k] * S[:k][np.newaxis, :]  # (N, k)

        # Embedding distances between the same pairs
        d_svd = np.sqrt(np.sum((embedding[idx_i] - embedding[idx_j])**2, axis=1))

        # Correlations
        r_eucl, p_eucl = stats.pearsonr(d_svd, d_euclidean)
        r_lor, p_lor = stats.pearsonr(d_svd, d_lorentzian)

        # Rank correlation (more robust)
        rho_eucl, p_rho_eucl = stats.spearmanr(d_svd, d_euclidean)

        # Does SVD distance distinguish causal from non-causal?
        d_causal = d_svd[is_causal == 1]
        d_noncausal = d_svd[is_causal == 0]
        if len(d_causal) > 10 and len(d_noncausal) > 10:
            t_cn, p_cn = stats.ttest_ind(d_causal, d_noncausal)
        else:
            t_cn, p_cn = 0.0, 1.0

        # Fraction of variance explained
        total_var = np.sum(S**2)
        topk_var = np.sum(S[:k]**2)
        var_explained = topk_var / total_var if total_var > 0 else 0

        results_by_k[k] = {
            "r_euclidean": float(r_eucl),
            "p_euclidean": float(p_eucl),
            "r_lorentzian": float(r_lor),
            "rho_spearman": float(rho_eucl),
            "causal_noncausal_p": float(p_cn),
            "variance_explained": float(var_explained),
            "mean_d_svd": float(np.mean(d_svd)),
        }

    return {
        "eps": eps,
        "results_by_k": results_by_k,
        "n_singular_values": len(S),
        "top10_sv": S[:10].tolist(),
    }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    """Benchmark one sprinkling with full SVD."""
    print("=== BENCHMARK ===", flush=True)
    N = N_POINTS
    rng = np.random.default_rng(42)
    V = T_DIAMOND**2 / 2.0
    rho = N / V

    t0 = time.perf_counter()
    pts, C = _sprinkle_flat(N, T_DIAMOND, rng)
    n_mat = compute_interval_cardinalities(C)
    L = build_bd_L(C, n_mat, rho)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    U, S, Vt = np.linalg.svd(L, full_matrices=False)
    t_svd = time.perf_counter() - t0

    total = t_build + t_svd
    n_tasks = len(EPSILON_VALUES) * M_ENSEMBLE
    print(f"  Build L: {t_build:.3f}s", flush=True)
    print(f"  SVD:     {t_svd:.3f}s", flush=True)
    print(f"  Total:   {total:.3f}s per sprinkling", flush=True)
    print(f"  Tasks:   {n_tasks}", flush=True)
    print(f"  Parallel ({WORKERS}w): {n_tasks * total / WORKERS / 60:.1f} min", flush=True)
    print(f"  With 1.5x margin: {n_tasks * total * 1.5 / WORKERS / 60:.1f} min", flush=True)
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    meta = ExperimentMeta(
        route=3, name="route3_svd_geometry",
        description="SVD truncation -> emergent geometry: does SVD embedding recover spacetime distances?",
        N=N_POINTS, M=M_ENSEMBLE, status="running",
    )

    print("=" * 70, flush=True)
    print("FND-1 ROUTE 2+3 SYNTHESIS: SVD TRUNCATION -> EMERGENT GEOMETRY", flush=True)
    print("=" * 70, flush=True)
    print(f"N={N_POINTS}, M={M_ENSEMBLE}, workers={WORKERS}", flush=True)
    print(f"k values: {K_VALUES}", flush=True)
    print(f"Epsilon: {EPSILON_VALUES}", flush=True)
    print(f"Distance pairs per sprinkling: {N_DISTANCE_PAIRS}", flush=True)
    print(flush=True)

    per_task = run_benchmark()

    ss = np.random.SeedSequence(MASTER_SEED)

    all_results = {}

    for eps in EPSILON_VALUES:
        eps_ss = ss.spawn(1)[0]
        child_seeds = eps_ss.spawn(M_ENSEMBLE)
        seed_ints = [int(cs.generate_state(1)[0]) for cs in child_seeds]
        args = [(si, N_POINTS, T_DIAMOND, eps, K_VALUES, N_DISTANCE_PAIRS)
                for si in seed_ints]

        print(f"\n  eps={eps:+.3f}: {M_ENSEMBLE} sprinklings...", flush=True)
        t0 = time.perf_counter()
        with Pool(WORKERS, initializer=_init_worker) as pool:
            results = pool.map(_worker_svd, args)
        elapsed = time.perf_counter() - t0
        print(f"    Done in {elapsed:.1f}s", flush=True)

        all_results[eps] = results

    # ==================================================================
    # ANALYSIS
    # ==================================================================

    print(f"\n{'='*60}", flush=True)
    print("SVD EMBEDDING: DISTANCE CORRELATION BY K", flush=True)
    print("=" * 60, flush=True)

    print(f"\n  {'eps':>6} {'k':>5} {'r_eucl':>8} {'r_lor':>8} {'rho_sp':>8} "
          f"{'var_expl':>10} {'causal_p':>10}", flush=True)
    print(f"  {'-'*6} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}", flush=True)

    summary = {}
    best_r = 0
    best_config = None

    for eps in EPSILON_VALUES:
        for k in K_VALUES:
            r_eucls = [r["results_by_k"][k]["r_euclidean"] for r in all_results[eps]
                       if k in r["results_by_k"]]
            r_lors = [r["results_by_k"][k]["r_lorentzian"] for r in all_results[eps]
                      if k in r["results_by_k"]]
            rho_sps = [r["results_by_k"][k]["rho_spearman"] for r in all_results[eps]
                       if k in r["results_by_k"]]
            var_exps = [r["results_by_k"][k]["variance_explained"] for r in all_results[eps]
                        if k in r["results_by_k"]]
            causal_ps = [r["results_by_k"][k]["causal_noncausal_p"] for r in all_results[eps]
                         if k in r["results_by_k"]]

            mean_r_e = float(np.mean(r_eucls))
            mean_r_l = float(np.mean(r_lors))
            mean_rho = float(np.mean(rho_sps))
            mean_var = float(np.mean(var_exps))
            mean_cp = float(np.mean(causal_ps))

            print(f"  {eps:+6.3f} {k:5d} {mean_r_e:+8.4f} {mean_r_l:+8.4f} {mean_rho:+8.4f} "
                  f"{mean_var:10.4f} {mean_cp:10.2e}", flush=True)

            key = f"eps{eps:+.1f}_k{k}"
            summary[key] = {
                "eps": eps, "k": k,
                "r_euclidean_mean": mean_r_e,
                "r_lorentzian_mean": mean_r_l,
                "rho_spearman_mean": mean_rho,
                "variance_explained": mean_var,
            }

            if abs(mean_r_e) > abs(best_r):
                best_r = mean_r_e
                best_config = (eps, k)

    # ==================================================================
    # VERDICT
    # ==================================================================

    total_time = time.perf_counter() - t_total

    print(f"\n{'='*70}", flush=True)
    print("VERDICT", flush=True)
    print("=" * 70, flush=True)

    if abs(best_r) > 0.5:
        verdict = (f"GEOMETRY RECOVERED: SVD embedding at k={best_config[1]} (eps={best_config[0]}) "
                   f"correlates with spacetime distance (r={best_r:.4f})")
    elif abs(best_r) > 0.3:
        verdict = (f"WEAK GEOMETRY: partial correlation r={best_r:.4f} at k={best_config[1]}")
    else:
        verdict = f"NO GEOMETRY: best r={best_r:.4f}, SVD embedding does not recover distances"

    print(f"\n  {verdict}", flush=True)
    print(f"  Best: eps={best_config[0]}, k={best_config[1]}, r={best_r:.4f}", flush=True)
    print(f"  Wall time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    # Save
    meta.status = "completed"
    meta.verdict = verdict
    meta.wall_time_sec = total_time

    output = {
        "parameters": {
            "N": N_POINTS, "M": M_ENSEMBLE, "T": T_DIAMOND,
            "k_values": K_VALUES, "eps_values": EPSILON_VALUES,
            "n_distance_pairs": N_DISTANCE_PAIRS,
            "workers": WORKERS,
        },
        "summary": summary,
        "best_r": best_r,
        "best_config": {"eps": best_config[0], "k": best_config[1]} if best_config else None,
        "verdict": verdict,
        "wall_time_sec": total_time,
        "per_task_benchmark_sec": per_task,
    }

    out_path = RESULTS_DIR / "route3_svd_geometry.json"
    save_experiment(meta, output, out_path)
    print(f"  Saved: {out_path}", flush=True)

    clear_progress()
    print("\nDone.", flush=True)


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        run_benchmark()
    else:
        main()
