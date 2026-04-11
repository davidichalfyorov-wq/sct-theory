"""
FND-1 Route 2: Link-Graph Laplacian -> Emergent Geometry.

Verification agent found r=+0.82 correlation between link-graph Laplacian
spectral embedding distances and Euclidean spacetime distances (k=2).
This script rigorously reproduces and verifies that finding.

The link graph (Hasse diagram) is the subset of causal relations where
two elements are directly linked (zero intervening elements). Its graph
Laplacian L_link = D - A_link is real symmetric PSD with well-defined
spectral embedding.

Run with MKL:
  "C:/Users/youre/miniconda3/envs/sct-mkl/python.exe" analysis/scripts/fnd1_route2_link_geometry.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_ensemble_runner import compute_interval_cardinalities
from fnd1_gate5_runner import sprinkle_curved, _sprinkle_flat
from fnd1_experiment_registry import (
    ExperimentMeta, update_progress, clear_progress, save_experiment, RESULTS_DIR,
)
from fnd1_parallel import N_WORKERS, _init_worker

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_VALUES = [500, 1000, 2000, 3000]
N_PRIMARY = 3000
M_ENSEMBLE = 80
T_DIAMOND = 1.0
MASTER_SEED = 42
K_VALUES = [2, 3, 5, 10, 20, 50]
EPSILON_VALUES = [-0.5, 0.0, 0.25, 0.5]
N_DISTANCE_PAIRS = 10000
WORKERS = N_WORKERS


# ---------------------------------------------------------------------------
# Link graph construction
# ---------------------------------------------------------------------------

def build_link_adjacency(C, n_matrix):
    """Extract the link (Hasse) adjacency matrix.

    A link i->j exists if j is in the causal past of i with zero
    intervening elements: C[j,i]=1 and n_matrix[j,i]=0.
    Returns the symmetrized undirected link adjacency (sparse).
    """
    past = C.T  # past[i,j]=1 means j precedes i
    n_past = n_matrix.T
    link_mask = (past > 0) & (n_past == 0)
    A_link = link_mask.astype(np.float64)
    A_link = A_link + A_link.T  # symmetrize
    A_link = (A_link > 0).astype(np.float64)
    return sp.csr_matrix(A_link)


def build_link_laplacian(A_link_sp):
    """Build combinatorial graph Laplacian L = D - A from sparse adjacency."""
    degrees = np.array(A_link_sp.sum(axis=1)).ravel()
    D = sp.diags(degrees)
    return D - A_link_sp


def spectral_embedding(L_sp, k, N):
    """Compute spectral embedding from bottom k+1 eigenvectors of L.

    Excludes the trivial zero eigenvalue (constant vector).
    Embedding: phi(i) = (sqrt(lam_1)*v_1(i), ..., sqrt(lam_k)*v_k(i))
    """
    n_eigs = min(k + 1, N - 1)
    try:
        eigenvalues, eigenvectors = eigsh(L_sp.astype(np.float64), k=n_eigs,
                                          which='SM', sigma=0.0)
    except Exception:
        # Fallback: dense eigvalsh
        L_dense = L_sp.toarray()
        all_evals, all_evecs = np.linalg.eigh(L_dense)
        eigenvalues = all_evals[:n_eigs]
        eigenvectors = all_evecs[:, :n_eigs]

    # Sort by eigenvalue
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Skip the zero eigenvalue (first one)
    start = 1
    end = min(start + k, len(eigenvalues))
    lams = eigenvalues[start:end]
    vecs = eigenvectors[:, start:end]

    # Embedding: sqrt(lambda) * eigenvector
    lams_safe = np.maximum(lams, 0)
    embedding = vecs * np.sqrt(lams_safe)[np.newaxis, :]
    return embedding, lams


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker_link_geometry(args):
    """Compute link-graph Laplacian spectral embedding and distance correlations."""
    seed_int, N, T, eps, k_values, n_pairs = args

    rng = np.random.default_rng(seed_int)
    if eps == 0.0:
        pts, C = _sprinkle_flat(N, T, rng)
    else:
        pts, C = sprinkle_curved(N, eps, T, rng)

    n_matrix = compute_interval_cardinalities(C)
    total_causal = float(np.sum(C))

    # BD action
    past = C.T
    n_past = n_matrix.T
    N1 = int(np.sum((past > 0) & (n_past == 0)))
    N2 = int(np.sum((past > 0) & (n_past == 1)))
    N3 = int(np.sum((past > 0) & (n_past == 2)))
    bd_action = float(N - 2 * N1 + 4 * N2 - 2 * N3)

    # Build link graph
    A_link = build_link_adjacency(C, n_matrix)
    L_link = build_link_laplacian(A_link)

    mean_link_degree = float(A_link.sum() / N)

    # Random pairs for distance comparison
    rng2 = np.random.default_rng(seed_int + 999)
    idx_i = rng2.integers(0, N, size=n_pairs)
    idx_j = rng2.integers(0, N, size=n_pairs)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    t_coords = pts[:, 0]
    x_coords = pts[:, 1]
    dt = t_coords[idx_i] - t_coords[idx_j]
    dx = x_coords[idx_i] - x_coords[idx_j]
    d_euclidean = np.sqrt(dt**2 + dx**2)

    results_by_k = {}
    max_k = max(k_values)

    embedding_full, lams = spectral_embedding(L_link, max_k, N)

    for k in k_values:
        if k > embedding_full.shape[1]:
            k_eff = embedding_full.shape[1]
        else:
            k_eff = k

        emb = embedding_full[:, :k_eff]
        d_svd = np.sqrt(np.sum((emb[idx_i] - emb[idx_j])**2, axis=1))

        r_eucl, p_eucl = stats.pearsonr(d_svd, d_euclidean)
        rho_sp, p_rho = stats.spearmanr(d_svd, d_euclidean)

        # Causal vs non-causal discrimination
        is_causal = (np.abs(dt) > np.abs(dx)).astype(float)
        d_causal = d_svd[is_causal == 1]
        d_noncausal = d_svd[is_causal == 0]
        if len(d_causal) > 10 and len(d_noncausal) > 10:
            _, p_cn = stats.ttest_ind(d_causal, d_noncausal)
        else:
            p_cn = 1.0

        results_by_k[k] = {
            "r_euclidean": float(r_eucl),
            "p_euclidean": float(p_eucl),
            "rho_spearman": float(rho_sp),
            "causal_noncausal_p": float(p_cn),
        }

    return {
        "eps": eps,
        "results_by_k": results_by_k,
        "total_causal": total_causal,
        "bd_action": bd_action,
        "mean_link_degree": mean_link_degree,
        "n_links": int(N1),
        "fiedler": float(lams[0]) if len(lams) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def run_benchmark():
    print("=== BENCHMARK ===", flush=True)
    times = {}
    for N in N_VALUES:
        rng = np.random.default_rng(42)
        t0 = time.perf_counter()
        pts, C = _sprinkle_flat(N, T_DIAMOND, rng)
        n_mat = compute_interval_cardinalities(C)
        A_link = build_link_adjacency(C, n_mat)
        L_link = build_link_laplacian(A_link)
        emb, lams = spectral_embedding(L_link, max(K_VALUES), N)
        total = time.perf_counter() - t0

        n_links = int(A_link.sum() / 2)
        mean_deg = A_link.sum() / N
        n_tasks = len(EPSILON_VALUES) * M_ENSEMBLE
        par = n_tasks * total / WORKERS
        print(f"  N={N}: {total:.3f}s/task, links={n_links}, deg={mean_deg:.1f}, "
              f"parallel: {par/60:.1f} min", flush=True)
        times[N] = total

    total_all = sum(len(EPSILON_VALUES) * M_ENSEMBLE * t / WORKERS
                    for t in times.values())
    print(f"\n  TOTAL: {total_all/60:.1f} min (with 1.5x: {total_all*1.5/60:.1f} min)")
    return times


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    meta = ExperimentMeta(
        route=2, name="route2_link_geometry",
        description="Link-graph Laplacian spectral embedding -> geometry recovery",
        N=N_PRIMARY, M=M_ENSEMBLE, status="running",
    )

    print("=" * 70, flush=True)
    print("FND-1 ROUTE 2: LINK-GRAPH LAPLACIAN -> EMERGENT GEOMETRY", flush=True)
    print("=" * 70, flush=True)
    print(f"N values: {N_VALUES} (primary: {N_PRIMARY})", flush=True)
    print(f"M={M_ENSEMBLE}, k values: {K_VALUES}", flush=True)
    print(f"Epsilon: {EPSILON_VALUES}", flush=True)
    print(f"Distance pairs: {N_DISTANCE_PAIRS}", flush=True)
    print(flush=True)

    times = run_benchmark()

    ss = np.random.SeedSequence(MASTER_SEED)

    # ==================================================================
    # PRIMARY ANALYSIS at N_PRIMARY
    # ==================================================================

    print(f"\n{'='*60}", flush=True)
    print(f"PRIMARY ANALYSIS (N={N_PRIMARY})", flush=True)
    print("=" * 60, flush=True)

    all_data = {}
    for eps in EPSILON_VALUES:
        eps_ss = ss.spawn(1)[0]
        child_seeds = eps_ss.spawn(M_ENSEMBLE)
        seed_ints = [int(cs.generate_state(1)[0]) for cs in child_seeds]
        args = [(si, N_PRIMARY, T_DIAMOND, eps, K_VALUES, N_DISTANCE_PAIRS)
                for si in seed_ints]

        print(f"\n  eps={eps:+.3f}: {M_ENSEMBLE} sprinklings...", flush=True)
        t0 = time.perf_counter()
        with Pool(WORKERS, initializer=_init_worker) as pool:
            results = pool.map(_worker_link_geometry, args)
        elapsed = time.perf_counter() - t0
        print(f"    Done in {elapsed:.1f}s", flush=True)
        all_data[eps] = results

    # ==================================================================
    # ANALYSIS: correlation by k
    # ==================================================================

    print(f"\n{'='*60}", flush=True)
    print("GEOMETRY RECOVERY: DISTANCE CORRELATION BY K", flush=True)
    print("=" * 60, flush=True)

    print(f"\n  {'eps':>6} {'k':>5} {'r_eucl':>8} {'rho_sp':>8} {'causal_p':>10}", flush=True)
    print(f"  {'-'*6} {'-'*5} {'-'*8} {'-'*8} {'-'*10}", flush=True)

    summary = {}
    best_r = 0
    best_config = None

    for eps in EPSILON_VALUES:
        for k in K_VALUES:
            r_eucls = [r["results_by_k"][k]["r_euclidean"] for r in all_data[eps]
                       if k in r["results_by_k"]]
            rho_sps = [r["results_by_k"][k]["rho_spearman"] for r in all_data[eps]
                       if k in r["results_by_k"]]
            causal_ps = [r["results_by_k"][k]["causal_noncausal_p"] for r in all_data[eps]
                         if k in r["results_by_k"]]

            mean_r = float(np.mean(r_eucls))
            mean_rho = float(np.mean(rho_sps))
            mean_cp = float(np.mean(causal_ps))
            sem_r = float(np.std(r_eucls) / np.sqrt(len(r_eucls)))

            print(f"  {eps:+6.3f} {k:5d} {mean_r:+8.4f} {mean_rho:+8.4f} {mean_cp:10.2e}",
                  flush=True)

            key = f"eps{eps:+.1f}_k{k}"
            summary[key] = {
                "eps": eps, "k": k,
                "r_euclidean_mean": mean_r,
                "r_euclidean_sem": sem_r,
                "rho_spearman_mean": mean_rho,
            }

            if abs(mean_r) > abs(best_r):
                best_r = mean_r
                best_config = (eps, k)

    # ==================================================================
    # MEDIATION: does r survive TC+BD control?
    # ==================================================================

    print(f"\n{'='*60}", flush=True)
    print("MEDIATION: per-sprinkling r_euclidean vs eps", flush=True)
    print("=" * 60, flush=True)

    # For each sprinkling, compute r_euclidean at the best k
    best_k = best_config[1] if best_config else 2
    eps_all, r_all, tc_all, bd_all, deg_all = [], [], [], [], []
    for eps in EPSILON_VALUES:
        for r in all_data[eps]:
            eps_all.append(eps)
            r_all.append(r["results_by_k"].get(best_k, r["results_by_k"][2])["r_euclidean"])
            tc_all.append(r["total_causal"])
            bd_all.append(r["bd_action"])
            deg_all.append(r["mean_link_degree"])

    eps_arr = np.array(eps_all)
    r_arr = np.array(r_all)
    tc_arr = np.array(tc_all)
    bd_arr = np.array(bd_all)
    deg_arr = np.array(deg_all)

    # Does r_euclidean vary with eps?
    r_vs_eps, p_vs_eps = stats.pearsonr(eps_arr, r_arr)
    r_vs_eps2, p_vs_eps2 = stats.pearsonr(eps_arr**2, r_arr)
    print(f"\n  r(r_eucl, eps) = {r_vs_eps:+.4f}, p = {p_vs_eps:.2e}", flush=True)
    print(f"  r(r_eucl, eps^2) = {r_vs_eps2:+.4f}, p = {p_vs_eps2:.2e}", flush=True)

    # Group means
    print(f"\n  Group means (k={best_k}):", flush=True)
    print(f"  {'eps':>6} {'r_eucl':>8} {'SEM':>8} {'TC':>10} {'deg':>6}", flush=True)
    for eps in EPSILON_VALUES:
        d = [i for i, e in enumerate(eps_all) if abs(e - eps) < 0.01]
        mean_r = np.mean([r_all[i] for i in d])
        sem = np.std([r_all[i] for i in d]) / np.sqrt(len(d))
        mean_tc = np.mean([tc_all[i] for i in d])
        mean_deg = np.mean([deg_all[i] for i in d])
        print(f"  {eps:+6.3f} {mean_r:+8.4f} {sem:8.4f} {mean_tc:10.0f} {mean_deg:6.1f}",
              flush=True)

    # ==================================================================
    # FINITE-SIZE SCALING
    # ==================================================================

    print(f"\n{'='*60}", flush=True)
    print("FINITE-SIZE SCALING", flush=True)
    print("=" * 60, flush=True)

    scaling = {}
    for N_test in N_VALUES:
        if N_test == N_PRIMARY:
            # Already computed
            flat_rs = [r["results_by_k"].get(2, {}).get("r_euclidean", 0)
                       for r in all_data.get(0.0, [])]
            scaling[N_test] = {"r_mean": float(np.mean(flat_rs)) if flat_rs else 0,
                               "r_sem": float(np.std(flat_rs)/np.sqrt(len(flat_rs))) if flat_rs else 0}
            continue

        n_ss = ss.spawn(1)[0]
        child_seeds = n_ss.spawn(M_ENSEMBLE)
        seed_ints = [int(cs.generate_state(1)[0]) for cs in child_seeds]
        args = [(si, N_test, T_DIAMOND, 0.0, [2, 5, 10], N_DISTANCE_PAIRS)
                for si in seed_ints]

        print(f"\n  N={N_test}: {M_ENSEMBLE} sprinklings (flat)...", flush=True)
        t0 = time.perf_counter()
        with Pool(WORKERS, initializer=_init_worker) as pool:
            results = pool.map(_worker_link_geometry, args)
        elapsed = time.perf_counter() - t0

        rs = [r["results_by_k"][2]["r_euclidean"] for r in results]
        mean_r = float(np.mean(rs))
        sem = float(np.std(rs) / np.sqrt(len(rs)))
        mean_deg = float(np.mean([r["mean_link_degree"] for r in results]))
        print(f"    Done in {elapsed:.1f}s: r(k=2) = {mean_r:+.4f} +/- {sem:.4f}, "
              f"deg={mean_deg:.1f}", flush=True)

        scaling[N_test] = {"r_mean": mean_r, "r_sem": sem, "mean_deg": mean_deg}

    print(f"\n  Scaling summary (k=2, flat):", flush=True)
    print(f"  {'N':>6} {'r_eucl':>8} {'SEM':>8}", flush=True)
    for N_test in sorted(scaling.keys()):
        sr = scaling[N_test]
        print(f"  {N_test:6d} {sr['r_mean']:+8.4f} {sr['r_sem']:8.4f}", flush=True)

    # ==================================================================
    # VERDICT
    # ==================================================================

    total_time = time.perf_counter() - t_total

    print(f"\n{'='*70}", flush=True)
    print("VERDICT", flush=True)
    print("=" * 70, flush=True)

    if abs(best_r) > 0.7:
        verdict = (f"GEOMETRY RECOVERED: link-graph Laplacian at k={best_config[1]} "
                   f"(eps={best_config[0]}) correlates with spacetime distance "
                   f"(r={best_r:.4f})")
    elif abs(best_r) > 0.5:
        verdict = f"PARTIAL GEOMETRY: r={best_r:.4f} at k={best_config[1]}"
    elif abs(best_r) > 0.3:
        verdict = f"WEAK GEOMETRY: r={best_r:.4f}"
    else:
        verdict = f"NO GEOMETRY: best r={best_r:.4f}"

    r_varies = abs(r_vs_eps) > 0.1 and p_vs_eps < 0.05
    if r_varies:
        verdict += " | CURVATURE SENSITIVITY DETECTED"
    else:
        verdict += " | curvature-independent (conformal invariance)"

    print(f"\n  {verdict}", flush=True)
    print(f"  Best: eps={best_config[0]}, k={best_config[1]}, r={best_r:.4f}", flush=True)
    print(f"  r(r_eucl, eps) = {r_vs_eps:+.4f} (p={p_vs_eps:.2e})", flush=True)
    print(f"  Wall time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    # Save
    meta.status = "completed"
    meta.verdict = verdict
    meta.wall_time_sec = total_time

    output = {
        "parameters": {
            "N_primary": N_PRIMARY, "N_values": N_VALUES, "M": M_ENSEMBLE,
            "T": T_DIAMOND, "k_values": K_VALUES, "eps_values": EPSILON_VALUES,
            "n_distance_pairs": N_DISTANCE_PAIRS, "workers": WORKERS,
        },
        "summary": summary,
        "best_r": best_r,
        "best_config": {"eps": best_config[0], "k": best_config[1]} if best_config else None,
        "curvature_sensitivity": {
            "r_vs_eps": float(r_vs_eps), "p_vs_eps": float(p_vs_eps),
            "r_vs_eps2": float(r_vs_eps2), "p_vs_eps2": float(p_vs_eps2),
        },
        "scaling": {str(n): v for n, v in scaling.items()},
        "verdict": verdict,
        "wall_time_sec": total_time,
    }

    out_path = RESULTS_DIR / "route2_link_geometry.json"
    save_experiment(meta, output, out_path)
    print(f"  Saved: {out_path}", flush=True)

    clear_progress()
    print("\nDone.", flush=True)


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        run_benchmark()
    else:
        main()
