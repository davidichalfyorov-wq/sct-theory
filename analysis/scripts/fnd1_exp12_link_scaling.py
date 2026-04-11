"""
FND-1 EXP-12: Link-Graph Laplacian Embedding — Finite-Size Scaling.

The route2_link_geometry experiment and verification agents found that
the link-graph (Hasse diagram) Laplacian spectral embedding recovers
Euclidean spacetime distances. This experiment measures how r(N) scales
with causal set size.

If r(N) -> 1 as N -> inf: perfect geometry reconstruction (Route 2 viable)
If r(N) plateaus < 1: fundamental information loss
If r(N) decreases: embedding fails at large N

Finite-size scaling: N = 200, 500, 1000, 2000, 3000, 5000.
Flat d=2 Minkowski (primary) + curved eps=0.5 (secondary).

Run with MKL:
  "C:/Users/youre/miniconda3/envs/sct-mkl/python.exe" analysis/scripts/fnd1_exp12_link_scaling.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_ensemble_runner import (
    sprinkle_diamond,
    compute_interval_cardinalities,
)
from fnd1_gate5_runner import sprinkle_curved
from fnd1_route2_link_geometry import (
    build_link_adjacency,
    build_link_laplacian,
)
from fnd1_experiment_registry import (
    ExperimentMeta, save_experiment, RESULTS_DIR,
)
from fnd1_parallel import N_WORKERS, _init_worker

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_VALUES = [200, 500, 1000, 2000, 3000, 5000]
M_ENSEMBLE = 80
T_DIAMOND = 1.0
MASTER_SEED = 42
K_VALUES = [2, 3, 5, 10, 20, 50]
N_DISTANCE_PAIRS = 10000
WORKERS = N_WORKERS

# Secondary: curved spacetime for comparison
EPS_CURVED = 0.5


# ---------------------------------------------------------------------------
# Spectral embedding (dense, robust)
# ---------------------------------------------------------------------------

def spectral_embedding_dense(L_dense, k):
    """Spectral embedding from dense graph Laplacian.

    Returns embedding (N, k) and eigenvalues[1:k+1].
    Dense eigh is more robust than sparse eigsh for this application.
    """
    N = L_dense.shape[0]
    evals, evecs = np.linalg.eigh(L_dense)

    # evals are ascending; evals[0] ~ 0 (constant vector)
    start = 1
    end = min(start + k, N)
    lams = evals[start:end]
    vecs = evecs[:, start:end]

    lams_safe = np.maximum(lams, 0)
    embedding = vecs * np.sqrt(lams_safe)[np.newaxis, :]
    return embedding, lams


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(args):
    """Compute link-graph embedding and distance correlations + null model."""
    seed_int, N, T, eps, k_values, n_pairs = args

    rng = np.random.default_rng(seed_int)
    if eps == 0.0:
        pts, C = sprinkle_diamond(N, T, rng)
    else:
        pts, C = sprinkle_curved(N, eps, T, rng)

    n_mat = compute_interval_cardinalities(C)

    # Build link graph (Hasse diagram)
    A_link = build_link_adjacency(C, n_mat)
    L_link = build_link_laplacian(A_link)

    n_links = int(A_link.sum() / 2)
    mean_degree = float(A_link.sum() / N)

    # Dense Laplacian for robust eigendecomposition
    L_dense = L_link.toarray() if sp.issparse(L_link) else np.array(L_link)

    max_k = max(k_values)
    embedding_full, lams_full = spectral_embedding_dense(L_dense, max_k)

    # Random distance pairs
    rng2 = np.random.default_rng(seed_int + 999)
    idx_i = rng2.integers(0, N, size=n_pairs)
    idx_j = rng2.integers(0, N, size=n_pairs)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    t_coords = pts[:, 0]
    x_coords = pts[:, 1]
    dt = t_coords[idx_i] - t_coords[idx_j]
    dx = x_coords[idx_i] - x_coords[idx_j]
    d_euclidean = np.sqrt(dt ** 2 + dx ** 2)

    # Lorentzian (proper) interval
    s2 = dt ** 2 - dx ** 2
    d_lorentzian = np.sqrt(np.abs(s2))

    # Causal flag
    is_causal = (np.abs(dt) > np.abs(dx)).astype(float)

    # Null model: shuffled coordinates (breaks geometry-graph correspondence)
    rng_null = np.random.default_rng(seed_int + 55555)
    perm = rng_null.permutation(N)
    pts_shuf = pts[perm]
    dt_s = pts_shuf[idx_i, 0] - pts_shuf[idx_j, 0]
    dx_s = pts_shuf[idx_i, 1] - pts_shuf[idx_j, 1]
    d_eucl_shuf = np.sqrt(dt_s ** 2 + dx_s ** 2)

    results_by_k = {}
    for k in k_values:
        k_eff = min(k, embedding_full.shape[1])
        if k_eff == 0:
            results_by_k[k] = {
                "r_euclidean": 0.0, "rho_spearman": 0.0,
                "r_lorentzian": 0.0, "causal_disc_p": 1.0,
                "r_null": 0.0,
            }
            continue

        emb = embedding_full[:, :k_eff]
        d_emb = np.sqrt(np.sum((emb[idx_i] - emb[idx_j]) ** 2, axis=1))

        r_eucl, _ = stats.pearsonr(d_emb, d_euclidean)
        rho_sp, _ = stats.spearmanr(d_emb, d_euclidean)
        r_lor, _ = stats.pearsonr(d_emb, d_lorentzian)

        # Null model: correlation with shuffled coordinates
        r_null, _ = stats.pearsonr(d_emb, d_eucl_shuf)

        # Does embedding distance distinguish causal from non-causal?
        d_causal = d_emb[is_causal == 1]
        d_noncausal = d_emb[is_causal == 0]
        if len(d_causal) > 10 and len(d_noncausal) > 10:
            _, p_cn = stats.ttest_ind(d_causal, d_noncausal)
        else:
            p_cn = 1.0

        results_by_k[k] = {
            "r_euclidean": float(r_eucl),
            "rho_spearman": float(rho_sp),
            "r_lorentzian": float(r_lor),
            "causal_disc_p": float(p_cn),
            "r_null": float(r_null),
        }

    return {
        "results_by_k": results_by_k,
        "n_links": n_links,
        "mean_degree": mean_degree,
        "fiedler": float(lams_full[0]) if len(lams_full) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Run one N value
# ---------------------------------------------------------------------------

def run_one_N(N, eps, M, ss_parent, label):
    """Run M sprinklings at one N and return aggregated results."""
    print(f"\n  {label} (N={N}, eps={eps:+.2f}): {M} sprinklings...", flush=True)

    n_ss = ss_parent.spawn(1)[0]
    child_seeds = n_ss.spawn(M)
    seed_ints = [int(cs.generate_state(1)[0]) for cs in child_seeds]
    args = [(si, N, T_DIAMOND, eps, K_VALUES, N_DISTANCE_PAIRS)
            for si in seed_ints]

    t0 = time.perf_counter()
    with Pool(WORKERS, initializer=_init_worker) as pool:
        raw = pool.map(_worker, args)
    elapsed = time.perf_counter() - t0
    print(f"    Done: {elapsed:.1f}s ({elapsed / M:.3f}s/task)", flush=True)

    # Aggregate by k
    summary_by_k = {}
    for k in K_VALUES:
        r_eucls = [r["results_by_k"][k]["r_euclidean"] for r in raw
                   if k in r["results_by_k"]]
        rho_sps = [r["results_by_k"][k]["rho_spearman"] for r in raw
                   if k in r["results_by_k"]]
        r_lors = [r["results_by_k"][k]["r_lorentzian"] for r in raw
                  if k in r["results_by_k"]]
        causal_ps = [r["results_by_k"][k]["causal_disc_p"] for r in raw
                     if k in r["results_by_k"]]
        r_nulls = [r["results_by_k"][k]["r_null"] for r in raw
                   if k in r["results_by_k"]]

        summary_by_k[str(k)] = {
            "r_euclidean_mean": float(np.mean(r_eucls)),
            "r_euclidean_std": float(np.std(r_eucls, ddof=1)),
            "r_euclidean_sem": float(np.std(r_eucls, ddof=1) / np.sqrt(len(r_eucls))),
            "rho_spearman_mean": float(np.mean(rho_sps)),
            "rho_spearman_std": float(np.std(rho_sps, ddof=1)),
            "rho_spearman_sem": float(np.std(rho_sps, ddof=1) / np.sqrt(len(rho_sps))),
            "r_lorentzian_mean": float(np.mean(r_lors)),
            "causal_disc_frac_sig": float(np.mean(np.array(causal_ps) < 0.05)),
            "r_null_mean": float(np.mean(r_nulls)),
            "r_null_std": float(np.std(r_nulls, ddof=1)),
        }

    # Print table (Spearman as primary)
    print(f"\n    {'k':>4} {'rho_S':>10} {'+-':>1} {'sem':>8} "
          f"{'r_P':>10} {'r_null':>10} {'causal%':>8}", flush=True)
    print(f"    {'-' * 4} {'-' * 10} {'-' * 1} {'-' * 8} "
          f"{'-' * 10} {'-' * 10} {'-' * 8}", flush=True)
    for k in K_VALUES:
        s = summary_by_k[str(k)]
        print(f"    {k:4d} {s['rho_spearman_mean']:+10.4f} +- {s['rho_spearman_sem']:8.4f} "
              f"{s['r_euclidean_mean']:+10.4f} {s['r_null_mean']:+10.4f} "
              f"{s['causal_disc_frac_sig'] * 100:7.1f}%", flush=True)

    mean_deg = float(np.mean([r["mean_degree"] for r in raw]))
    mean_fiedler = float(np.mean([r["fiedler"] for r in raw]))
    print(f"    Mean link degree: {mean_deg:.2f}, Fiedler: {mean_fiedler:.4f}",
          flush=True)

    return {
        "summary_by_k": summary_by_k,
        "mean_degree": mean_deg,
        "mean_fiedler": mean_fiedler,
        "n_sprinklings": M,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    meta = ExperimentMeta(
        route=2, name="exp12_link_scaling",
        description="Link-graph Laplacian embedding: r(N) finite-size scaling",
        N=max(N_VALUES), M=M_ENSEMBLE, status="running",
    )

    print("=" * 70, flush=True)
    print("FND-1 EXP-12: LINK EMBEDDING FINITE-SIZE SCALING", flush=True)
    print("=" * 70, flush=True)
    print(f"N values: {N_VALUES}", flush=True)
    print(f"M = {M_ENSEMBLE}, k values: {K_VALUES}", flush=True)
    print(f"Distance pairs: {N_DISTANCE_PAIRS}, workers: {WORKERS}", flush=True)
    print(flush=True)

    # Benchmark (5 seeds to capture variance — seed 42 is a known outlier)
    print("=== BENCHMARK (5 seeds) ===", flush=True)
    bench_seeds = [100, 200, 300, 400, 500]
    for N in N_VALUES:
        times = []
        rs = []
        for seed in bench_seeds:
            t0 = time.perf_counter()
            r = _worker((seed, N, T_DIAMOND, 0.0, K_VALUES, N_DISTANCE_PAIRS))
            times.append(time.perf_counter() - t0)
            rs.append(r["results_by_k"][2]["rho_spearman"])
        mean_t = np.mean(times)
        par = M_ENSEMBLE * mean_t / WORKERS
        print(f"  N={N:5d}: {mean_t:.3f}s/task, rho_S(k=2)="
              f"{np.mean(rs):+.3f}+-{np.std(rs):.3f}, "
              f"parallel({WORKERS}w): {par / 60:.1f} min", flush=True)

    ss = np.random.SeedSequence(MASTER_SEED)

    # PRIMARY: flat spacetime, all N values
    results_flat = {}
    print(f"\n{'=' * 70}", flush=True)
    print("PRIMARY: FLAT SPACETIME (eps=0)", flush=True)
    print("=" * 70, flush=True)

    for N in N_VALUES:
        results_flat[str(N)] = run_one_N(N, 0.0, M_ENSEMBLE, ss, f"Flat N={N}")

    # SECONDARY: curved spacetime at N_primary
    N_curved = 2000
    print(f"\n{'=' * 70}", flush=True)
    print(f"SECONDARY: CURVED SPACETIME (eps={EPS_CURVED}, N={N_curved})", flush=True)
    print("=" * 70, flush=True)

    results_curved = run_one_N(N_curved, EPS_CURVED, M_ENSEMBLE, ss,
                               f"Curved N={N_curved}")

    # ==================================================================
    # SCALING ANALYSIS
    # ==================================================================

    print(f"\n{'=' * 70}", flush=True)
    print("SCALING ANALYSIS: r(N) at each k", flush=True)
    print("=" * 70, flush=True)

    Ns = np.array(N_VALUES, dtype=float)

    # Table: N vs k (Spearman rho as primary)
    header_ks = [2, 5, 10, 20]
    print(f"\n  {'N':>6}", end="", flush=True)
    for k in header_ks:
        print(f" {'rho_k=' + str(k):>10}", end="")
    print(f" {'null_k2':>10} {'degree':>8}", flush=True)
    print(f"  {'-' * 6}" + f" {'-' * 10}" * len(header_ks)
          + f" {'-' * 10} {'-' * 8}", flush=True)

    for N in N_VALUES:
        s = results_flat[str(N)]
        print(f"  {N:6d}", end="", flush=True)
        for k in header_ks:
            sk = s["summary_by_k"].get(str(k), {})
            r_val = sk.get("rho_spearman_mean", float("nan"))
            print(f" {r_val:+10.4f}", end="")
        null_k2 = s["summary_by_k"].get("2", {}).get("r_null_mean", float("nan"))
        print(f" {null_k2:+10.4f} {s['mean_degree']:8.2f}", flush=True)

    # Scaling fit for k=2 — SPEARMAN as primary (more robust than Pearson)
    rho_k2 = np.array([
        results_flat[str(N)]["summary_by_k"]["2"]["rho_spearman_mean"]
        for N in N_VALUES
    ])
    # Also extract Pearson for comparison
    rs_k2 = np.array([
        results_flat[str(N)]["summary_by_k"]["2"]["r_euclidean_mean"]
        for N in N_VALUES
    ])

    # Log-linear fit: rho vs log(N)
    lr = stats.linregress(np.log(Ns), rho_k2)
    log_slope = float(lr.slope)
    log_r2 = float(lr.rvalue ** 2)
    r_extrap_1e6 = float(lr.slope * np.log(1e6) + lr.intercept)

    print(f"\n  Spearman rho(k=2) vs log(N): slope = {log_slope:+.4f}, "
          f"R^2 = {log_r2:.4f}", flush=True)
    print(f"  Extrapolation to N=10^6: rho ~ {min(r_extrap_1e6, 1.0):.4f}",
          flush=True)

    # Pearson for comparison
    lr_p = stats.linregress(np.log(Ns), rs_k2)
    print(f"  Pearson r(k=2) vs log(N): slope = {lr_p.slope:+.4f}, "
          f"R^2 = {lr_p.rvalue**2:.4f}", flush=True)

    # Power law fit: 1 - rho = a * N^{-b}
    pwr_exp = pwr_r2 = float("nan")
    if np.all(rho_k2 > 0) and np.all(rho_k2 < 1):
        deficit = 1.0 - rho_k2
        if np.all(deficit > 0):
            lr_pwr = stats.linregress(np.log(Ns), np.log(deficit))
            pwr_exp = float(lr_pwr.slope)
            pwr_r2 = float(lr_pwr.rvalue ** 2)
            print(f"  (1-rho) ~ N^{pwr_exp:.3f} (R^2 = {pwr_r2:.4f})", flush=True)

    is_increasing = all(rho_k2[i + 1] >= rho_k2[i] - 0.01
                        for i in range(len(rho_k2) - 1))
    is_positive = all(r > 0 for r in rho_k2)

    # Null model check: r_null should be ~0 at all N
    null_k2 = np.array([
        results_flat[str(N)]["summary_by_k"]["2"]["r_null_mean"]
        for N in N_VALUES
    ])
    print(f"\n  Null model r(k=2): {['%.4f' % x for x in null_k2]}", flush=True)
    null_ok = all(abs(x) < 0.05 for x in null_k2)
    print(f"  Null model clean (all |r_null| < 0.05): {null_ok}", flush=True)

    # Curved vs flat comparison (Spearman)
    rho_flat_2k = results_flat[str(N_curved)]["summary_by_k"]["2"]["rho_spearman_mean"]
    rho_curv_2k = results_curved["summary_by_k"]["2"]["rho_spearman_mean"]
    print(f"\n  Flat vs Curved at N={N_curved}, k=2 (Spearman): "
          f"flat={rho_flat_2k:+.4f}, curved={rho_curv_2k:+.4f}, "
          f"diff={rho_curv_2k - rho_flat_2k:+.4f}", flush=True)

    # ==================================================================
    # VERDICT
    # ==================================================================

    total_time = time.perf_counter() - t_total
    best_rho = float(rho_k2[-1])
    best_r_p = float(rs_k2[-1])
    best_N = N_VALUES[-1]

    if best_rho > 0.8:
        verdict = (f"STRONG GEOMETRY: rho_S(k=2, N={best_N}) = {best_rho:.4f}"
                   f" [Pearson={best_r_p:.4f}]")
    elif best_rho > 0.5:
        verdict = (f"MODERATE GEOMETRY: rho_S(k=2, N={best_N}) = {best_rho:.4f}"
                   f" [Pearson={best_r_p:.4f}]")
    elif best_rho > 0.3:
        verdict = (f"WEAK GEOMETRY: rho_S(k=2, N={best_N}) = {best_rho:.4f}"
                   f" [Pearson={best_r_p:.4f}]")
    elif is_increasing and is_positive:
        verdict = (f"GROWING: rho_S(k=2) increasing ({rho_k2[0]:.3f} -> "
                   f"{best_rho:.3f}), needs larger N")
    else:
        verdict = f"NO GEOMETRY: rho_S(k=2, N={best_N}) = {best_rho:.4f}"

    if is_increasing:
        verdict += " | MONOTONIC"
    if null_ok:
        verdict += " | NULL CLEAN"

    print(f"\n{'=' * 70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"Wall time: {total_time:.0f}s ({total_time / 60:.1f} min)", flush=True)
    print("=" * 70, flush=True)

    # Save
    meta.status = "completed"
    meta.verdict = verdict
    meta.wall_time_sec = total_time

    output = {
        "parameters": {
            "N_values": N_VALUES, "M": M_ENSEMBLE, "T": T_DIAMOND,
            "k_values": K_VALUES, "n_distance_pairs": N_DISTANCE_PAIRS,
            "eps_curved": EPS_CURVED,
        },
        "results_flat": results_flat,
        "results_curved": {str(N_curved): results_curved},
        "scaling_k2": {
            "N": N_VALUES,
            "rho_spearman": rho_k2.tolist(),
            "r_pearson": rs_k2.tolist(),
            "r_null": null_k2.tolist(),
            "log_slope_spearman": log_slope,
            "log_r2_spearman": log_r2,
            "rho_extrap_1e6": min(r_extrap_1e6, 1.0),
            "power_exponent": pwr_exp,
            "power_r2": pwr_r2,
            "is_increasing": is_increasing,
            "null_clean": null_ok,
        },
        "verdict": verdict,
        "wall_time_sec": total_time,
    }

    out_path = RESULTS_DIR / "exp12_link_scaling.json"
    save_experiment(meta, output, out_path)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
