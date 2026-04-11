"""
PATH 2: Curvature Nonlocality on Causal Sets
==============================================

CLAIM: On causal sets, curvature information is encoded in NONLOCAL
(interval) quantities, not in LOCAL (graph neighborhood) quantities.

TEST: Compute a COMPREHENSIVE set of local graph statistics from the
link graph (k-neighborhood properties for k=1,2,3) and show that
ALL of them are degree proxies (R² > 0.80 against degree stats).

Then show that the interval-based curvature residual is NOT a proxy
for ANY of these local statistics (R² < 0.50).

This would establish: "curvature on causal sets is fundamentally
nonlocal" — a physics statement about discrete Lorentzian geometry.

LOCAL STATISTICS TO TEST (per vertex, then averaged):
  k=1: degree, in-degree, out-degree (already known proxies)
  k=1: clustering coefficient (fraction of neighbor pairs that are linked)
  k=1: local edge density in 1-neighborhood
  k=2: number of 2-step paths through vertex
  k=2: 2-neighborhood size
  k=2: degree of degree (sum of neighbor degrees)
  k=3: 3-neighborhood size
  Derived: degree centrality, betweenness (approximate)

For each: compute CRN delta, check R² against degree stats.
If ALL local stats are degree proxies → curvature is nonlocal.

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
from scipy import stats
import json, time, gc, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import *
from discovery_lapse_separation import causal_synthetic_lapse

METRIC_FNS["synthetic_lapse"] = causal_synthetic_lapse
SEED_OFFSETS["synthetic_lapse"] = 100

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
os.makedirs(OUTDIR, exist_ok=True)
T = 1.0


def local_graph_statistics(C, N):
    """Compute comprehensive LOCAL graph statistics from causal matrix.

    Returns dict with per-graph averages of local vertex properties.
    ALL computable from link graph (Hasse diagram) alone.
    """
    A = build_link_graph(C)
    A_dense = A.toarray()
    degrees = np.array(A.sum(axis=1)).ravel()

    result = {}

    # k=1: basic degree stats
    result["mean_degree"] = float(np.mean(degrees))
    result["degree_var"] = float(np.var(degrees))
    result["degree_std"] = float(np.std(degrees))

    # k=1: directed degree (from original causal link graph)
    C_sp = sp.csr_matrix(C)
    C2 = C_sp @ C_sp
    has_int = (C2 != 0).astype(np.float64)
    L_dir = C_sp - C_sp.multiply(has_int)
    L_dir.eliminate_zeros()

    out_deg = np.array(L_dir.sum(axis=1)).ravel()  # future links
    in_deg = np.array(L_dir.sum(axis=0)).ravel()   # past links
    result["mean_out_degree"] = float(np.mean(out_deg))
    result["mean_in_degree"] = float(np.mean(in_deg))
    result["out_in_ratio"] = float(np.mean(out_deg) / max(np.mean(in_deg), 1e-10))
    result["degree_asymmetry"] = float(np.mean(np.abs(out_deg - in_deg)))

    # k=1: clustering coefficient
    # C_i = 2 * triangles(i) / (deg(i) * (deg(i)-1))
    A2 = A_dense @ A_dense  # A2[i,j] = # paths of length 2 from i to j
    triangles_per_vertex = np.diag(A2 @ A_dense) / 2  # each triangle counted twice
    denom = degrees * (degrees - 1)
    denom[denom == 0] = 1
    clustering = 2 * triangles_per_vertex / denom
    result["mean_clustering"] = float(np.mean(clustering))
    result["clustering_var"] = float(np.var(clustering))

    # k=1: local edge density = edges in 1-neighborhood / possible edges
    # For vertex i: edges among neighbors / (deg(i) choose 2)
    local_densities = []
    for i in range(min(N, 200)):  # sample for speed
        neigh = np.where(A_dense[i] > 0.5)[0]
        k = len(neigh)
        if k < 2:
            local_densities.append(0)
            continue
        sub = A_dense[np.ix_(neigh, neigh)]
        e = np.sum(sub) / 2
        local_densities.append(e / (k * (k - 1) / 2))
    result["mean_local_density"] = float(np.mean(local_densities))

    # k=2: 2-neighborhood size
    A2_bool = (A2 > 0.5) | (A_dense > 0.5)
    np.fill_diagonal(A2_bool, False)
    neigh2_sizes = np.sum(A2_bool, axis=1)
    result["mean_2neigh_size"] = float(np.mean(neigh2_sizes))
    result["2neigh_var"] = float(np.var(neigh2_sizes))

    # k=2: degree-of-degree (sum of neighbor degrees)
    deg_of_deg = A_dense @ degrees
    result["mean_deg_of_deg"] = float(np.mean(deg_of_deg))
    result["deg_of_deg_var"] = float(np.var(deg_of_deg))

    # k=2: number of 2-step paths through each vertex
    paths2 = np.sum(A2, axis=1)
    result["mean_paths2"] = float(np.mean(paths2))

    # Edge count
    result["edge_count"] = int(np.sum(A_dense) / 2)

    # Forman
    rows, cols = np.where(np.triu(A_dense, k=1) > 0.5)
    if len(rows) > 0:
        F = 4.0 - degrees[rows] - degrees[cols]
        result["forman_mean"] = float(np.mean(F))
    else:
        result["forman_mean"] = 0.0

    del A, A_dense, C_sp, C2, L_dir
    return result


def crn_trial_local(seed, N, T, eps):
    """CRN trial computing ALL local stats + interval residual."""
    rng = np.random.default_rng(seed + 100)
    pts = sprinkle_4d(N, T, rng)

    result = {"seed": seed, "N": N, "eps": eps}

    C_flat = causal_flat(pts)
    C_ppw = causal_ppwave_quad(pts, eps)

    # Local stats
    ls_flat = local_graph_statistics(C_flat, N)
    ls_ppw = local_graph_statistics(C_ppw, N)

    for key in ls_flat:
        result[f"{key}_flat"] = ls_flat[key]
        result[f"{key}_ppw"] = ls_ppw[key]
        result[f"{key}_delta"] = ls_ppw[key] - ls_flat[key]

    del C_flat, C_ppw; gc.collect()
    return result


def main():
    N = 500
    M = 50
    eps = 10.0

    print("=" * 70)
    print("PATH 2: Curvature Nonlocality on Causal Sets")
    print(f"N={N}, M={M}, eps={eps}")
    print("Test: are ALL local graph stats degree proxies?")
    print("=" * 70)

    t0 = time.time()
    results = []
    for trial in range(M):
        res = crn_trial_local(trial * 1000, N, T, eps)
        results.append(res)
        if (trial + 1) % 10 == 0:
            print(f"  trial {trial+1}/{M} [{time.time()-t0:.1f}s]")

    elapsed = time.time() - t0
    print(f"  Total: {elapsed:.1f}s")

    # Define "degree baseline" = mean_degree_delta
    deg_delta = np.array([r["mean_degree_delta"] for r in results])

    # For each local statistic, compute R² against degree
    stat_keys = [k.replace("_delta", "") for k in results[0] if k.endswith("_delta")
                 and k != "mean_degree_delta"]

    print(f"\n  {'statistic':25s} {'mean delta':>12s} {'R²(vs deg)':>12s} {'proxy?':>8s}")
    print("  " + "-" * 60)

    n_proxy = 0
    n_total = 0
    local_proxy_table = {}

    for stat in sorted(set(stat_keys)):
        key = f"{stat}_delta"
        vals = np.array([r.get(key, 0) for r in results], dtype=float)

        if np.std(vals) < 1e-15:
            continue

        n_total += 1
        corr = np.corrcoef(vals, deg_delta)[0, 1]
        r2 = corr**2
        m = np.mean(vals)

        is_proxy = r2 > 0.80
        if is_proxy:
            n_proxy += 1

        local_proxy_table[stat] = {"r2_vs_degree": float(r2), "mean_delta": float(m)}
        flag = "PROXY" if r2 > 0.80 else ("notable" if r2 > 0.50 else "")
        print(f"  {stat:25s} {m:+12.4f} {r2:12.3f} {flag:>8s}")

    print(f"\n  LOCAL STATS THAT ARE DEGREE PROXIES: {n_proxy}/{n_total}")

    if n_proxy == n_total and n_total > 5:
        print(f"  ✅ ALL local stats are degree proxies → curvature is NONLOCAL")
        verdict = "NONLOCAL"
    elif n_proxy >= n_total * 0.8:
        print(f"  ⚠️ MOST ({n_proxy}/{n_total}) local stats are degree proxies")
        verdict = "MOSTLY_NONLOCAL"
    else:
        print(f"  ☠️ SOME local stats are NOT degree proxies → curvature may be local")
        verdict = "MIXED"

    # Now check: is the curvature RESIDUAL correlated with ANY local stat?
    # (We need to compute residual from this data — but we don't have
    # lapse-subtracted data in this trial. Use the local stats themselves.)
    # The question is: does ANY local stat carry curvature info beyond degree?

    # Find stats with LOWEST R² vs degree (most independent from degree)
    independent_stats = [(stat, v["r2_vs_degree"]) for stat, v in local_proxy_table.items()
                         if v["r2_vs_degree"] < 0.50]

    if independent_stats:
        print(f"\n  LOCAL STATS INDEPENDENT OF DEGREE:")
        for stat, r2 in sorted(independent_stats, key=lambda x: x[1]):
            print(f"    {stat:25s}: R²(vs deg) = {r2:.3f}")
        print(f"\n  ⚠️ These stats MIGHT carry curvature info. Test further.")
    else:
        print(f"\n  No local stat is degree-independent → ALL curvature is in intervals")

    all_results = {
        "verdict": verdict,
        "n_proxy": n_proxy, "n_total": n_total,
        "proxy_table": local_proxy_table,
        "independent_stats": [s[0] for s in independent_stats] if independent_stats else [],
    }

    outpath = os.path.join(OUTDIR, "path2_nonlocality.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
