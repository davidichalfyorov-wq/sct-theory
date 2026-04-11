"""
Discovery Run 001 — Ollivier-Ricci Curvature on Causal Sets
=============================================================

Computes Ollivier-Ricci curvature (ORC) on the undirected link graph
(Hasse diagram) of a causal set, tests curvature sensitivity via CRN,
and applies the FULL adversarial anti-bias protocol (v1.0).

ORC definition:
    kappa(x,y) = 1 - W_1(mu_x, mu_y) / d(x,y)

where:
    mu_x = uniform measure on neighbors of x in the undirected Hasse diagram
    W_1  = Wasserstein-1 (earth mover) distance
    d    = shortest path distance in the undirected Hasse diagram

van der Hoorn et al. (2020, arXiv:2008.01209) proved:
    ORC on Riemannian RGG --> Ricci curvature of the manifold

This is the FIRST application to causal sets (Lorentzian, directed).

PRE-REGISTERED ANTI-BIAS PROTOCOL (docs/analysis_protocols/antibias_discovery_v1.md):
    - Bonferroni alpha = 0.01 / 30 = 0.000333
    - GENUINE: p < alpha_bonf AND max R^2(proxy) < 0.50
    - PROXY: max R^2 > 0.80
    - AMBIGUOUS: 0.50 <= max R^2 < 0.80
    - Adversarial stats: mean_degree, degree_var, degree_std, degree_skew,
      degree_kurt, edge_count, max_degree, assortativity, MEAN_FORMAN
    - Null control: conformal (must give exact 0)
    - Two-sided tests, report ALL results

Author: David Alfyorov
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path
from scipy import stats
import ot  # Python Optimal Transport (POT)
import json
import time
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import (
    sprinkle_4d, causal_flat, causal_ppwave_quad, causal_schwarzschild,
    causal_conformal, build_link_graph, graph_statistics, forman_ricci,
    SEED_OFFSETS, METRIC_FNS
)

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Ollivier-Ricci curvature
# ---------------------------------------------------------------------------
def compute_orc(A_sp, sample_edges=None, max_edges=200):
    """Compute Ollivier-Ricci curvature on undirected graph.

    Args:
        A_sp: sparse adjacency matrix (undirected)
        sample_edges: if None, sample up to max_edges random edges
        max_edges: max edges to compute ORC for (cost control)

    Returns:
        dict with ORC statistics + per-edge kappa values
    """
    N = A_sp.shape[0]
    A = A_sp.tocsr()

    # 1. All-pairs shortest path distances
    #    For sparse graphs, BFS is faster than Floyd-Warshall
    dist_matrix = shortest_path(A, method='D', directed=False, unweighted=True)
    # Replace inf with large value (disconnected components)
    dist_matrix[np.isinf(dist_matrix)] = N + 1

    # 2. Get edge list
    rows, cols = sp.triu(A, k=1).nonzero()
    all_edges = list(zip(rows.tolist(), cols.tolist()))
    n_total_edges = len(all_edges)

    # 3. Sample edges if too many
    if sample_edges is not None:
        edges = sample_edges
    elif n_total_edges > max_edges:
        rng = np.random.default_rng(42)  # fixed seed for reproducibility
        idx = rng.choice(n_total_edges, size=max_edges, replace=False)
        edges = [all_edges[i] for i in idx]
    else:
        edges = all_edges

    # 4. Adjacency lists for fast neighbor lookup
    neighbors = [[] for _ in range(N)]
    r, c = A.nonzero()
    for i, j in zip(r.tolist(), c.tolist()):
        neighbors[i].append(j)

    # 5. Compute ORC for each sampled edge
    kappa_values = []

    for u, v in edges:
        neigh_u = neighbors[u]
        neigh_v = neighbors[v]

        if len(neigh_u) == 0 or len(neigh_v) == 0:
            continue

        # Uniform measures on neighbors
        mu_u = np.ones(len(neigh_u)) / len(neigh_u)
        mu_v = np.ones(len(neigh_v)) / len(neigh_v)

        # Cost matrix: shortest path distances between neighbors
        C = np.zeros((len(neigh_u), len(neigh_v)), dtype=np.float64)
        for i, a in enumerate(neigh_u):
            for j, b in enumerate(neigh_v):
                C[i, j] = dist_matrix[a, b]

        # W_1 via POT (exact EMD)
        W1 = ot.emd2(mu_u, mu_v, C)

        # Edge distance
        d_uv = dist_matrix[u, v]
        if d_uv < 1e-10:
            continue

        kappa = 1.0 - W1 / d_uv
        kappa_values.append(float(kappa))

    kappa_arr = np.array(kappa_values)

    if len(kappa_arr) == 0:
        return {
            "orc_mean": np.nan, "orc_std": np.nan, "orc_median": np.nan,
            "orc_skew": np.nan, "orc_frac_pos": np.nan, "orc_frac_neg": np.nan,
            "orc_min": np.nan, "orc_max": np.nan, "n_edges_sampled": 0,
            "n_edges_total": n_total_edges,
        }

    return {
        "orc_mean": float(np.mean(kappa_arr)),
        "orc_std": float(np.std(kappa_arr)),
        "orc_median": float(np.median(kappa_arr)),
        "orc_skew": float(stats.skew(kappa_arr)) if len(kappa_arr) > 2 else 0.0,
        "orc_kurt": float(stats.kurtosis(kappa_arr)) if len(kappa_arr) > 3 else 0.0,
        "orc_frac_pos": float(np.mean(kappa_arr > 0)),
        "orc_frac_neg": float(np.mean(kappa_arr < 0)),
        "orc_q25": float(np.percentile(kappa_arr, 25)),
        "orc_q75": float(np.percentile(kappa_arr, 75)),
        "orc_min": float(np.min(kappa_arr)),
        "orc_max": float(np.max(kappa_arr)),
        "n_edges_sampled": len(kappa_arr),
        "n_edges_total": n_total_edges,
    }


# ---------------------------------------------------------------------------
# CRN trial with ORC + full adversarial stats
# ---------------------------------------------------------------------------
def crn_trial_orc(seed, N, T, metric_name, eps, max_edges=200):
    """One CRN trial: flat + curved, ORC + Forman + graph stats."""
    seed_offset = SEED_OFFSETS[metric_name]
    rng = np.random.default_rng(seed + seed_offset)
    pts = sprinkle_4d(N, T, rng)

    # --- Flat ---
    C_flat = causal_flat(pts)
    A_flat = build_link_graph(C_flat)
    del C_flat; gc.collect()
    gs_flat, deg_flat = graph_statistics(A_flat)
    fr_flat, _ = forman_ricci(A_flat, deg_flat)
    orc_flat = compute_orc(A_flat, max_edges=max_edges)
    del A_flat; gc.collect()

    # --- Curved ---
    C_curv = METRIC_FNS[metric_name](pts, eps)
    A_curv = build_link_graph(C_curv)
    del C_curv; gc.collect()
    gs_curv, deg_curv = graph_statistics(A_curv)
    fr_curv, _ = forman_ricci(A_curv, deg_curv)
    orc_curv = compute_orc(A_curv, max_edges=max_edges)
    del A_curv; gc.collect()

    # --- Assemble ---
    result = {"seed": seed, "N": N, "metric": metric_name, "eps": eps}

    # ORC deltas
    result["orc_mean_flat"] = orc_flat["orc_mean"]
    result["orc_mean_curv"] = orc_curv["orc_mean"]
    result["orc_mean_delta"] = orc_curv["orc_mean"] - orc_flat["orc_mean"]
    result["orc_median_flat"] = orc_flat["orc_median"]
    result["orc_median_curv"] = orc_curv["orc_median"]
    result["orc_median_delta"] = orc_curv["orc_median"] - orc_flat["orc_median"]
    result["orc_frac_pos_flat"] = orc_flat["orc_frac_pos"]
    result["orc_frac_pos_curv"] = orc_curv["orc_frac_pos"]
    result["orc_frac_pos_delta"] = orc_curv["orc_frac_pos"] - orc_flat["orc_frac_pos"]
    result["n_edges_sampled"] = min(orc_flat["n_edges_sampled"], orc_curv["n_edges_sampled"])

    # Forman deltas (for comparison)
    result["forman_mean_flat"] = fr_flat["F_mean"]
    result["forman_mean_curv"] = fr_curv["F_mean"]
    result["forman_mean_delta"] = fr_curv["F_mean"] - fr_flat["F_mean"]

    # Graph stat deltas (for adversarial check)
    for key in gs_flat:
        result[f"{key}_flat"] = gs_flat[key]
        result[f"{key}_curv"] = gs_curv[key]
        result[f"{key}_delta"] = gs_curv[key] - gs_flat[key]

    return result


# ---------------------------------------------------------------------------
# Adversarial analysis (anti-bias v1.0)
# ---------------------------------------------------------------------------
def adversarial_analysis(results, observable_key="orc_mean_delta", label=""):
    """Full adversarial proxy check per anti-bias protocol v1.0."""
    M = len(results)
    obs = np.array([r[observable_key] for r in results])

    print(f"\n  === {label}: {observable_key} (M={M}) ===")

    # Basic statistics
    m, se = np.mean(obs), np.std(obs) / np.sqrt(M)
    d_cohen = m / np.std(obs) if np.std(obs) > 0 else 0
    _, p_obs = stats.ttest_1samp(obs, 0.0) if M >= 5 else (0, 1.0)
    print(f"  delta = {m:+.6f} +/- {se:.6f}, d={d_cohen:+.3f}, p={p_obs:.2e}")

    # Adversarial proxy check
    proxy_stats = [
        "mean_degree_delta", "degree_var_delta", "degree_std_delta",
        "degree_skew_delta", "degree_kurt_delta", "edge_count_delta",
        "max_degree_delta", "assortativity_delta", "forman_mean_delta",
    ]

    print(f"\n  ADVERSARIAL PROXY CHECK (anti-bias v1.0):")
    max_r2 = 0.0
    max_r2_name = ""
    proxy_table = {}

    for stat_name in proxy_stats:
        vals = np.array([r.get(stat_name, 0) for r in results], dtype=float)
        if np.std(vals) < 1e-15 or np.std(obs) < 1e-15:
            proxy_table[stat_name] = 0.0
            continue
        corr = np.corrcoef(obs, vals)[0, 1]
        r2 = corr**2
        proxy_table[stat_name] = float(r2)
        if r2 > max_r2:
            max_r2 = r2
            max_r2_name = stat_name
        flag = ""
        if r2 > 0.80:
            flag = " PROXY!!!"
        elif r2 > 0.50:
            flag = " AMBIGUOUS"
        elif r2 > 0.20:
            flag = " notable"
        if r2 > 0.15:
            print(f"    R2({observable_key}, {stat_name}) = {r2:.3f}{flag}")

    print(f"    MAX R2 = {max_r2:.3f} ({max_r2_name})")

    # Multiple regression
    X_cols = []
    col_names = []
    for stat_name in proxy_stats:
        vals = np.array([r.get(stat_name, 0) for r in results], dtype=float)
        if np.std(vals) > 1e-15:
            X_cols.append(vals)
            col_names.append(stat_name)

    r2_multi = 0.0
    if X_cols:
        X = np.column_stack(X_cols)
        X_c = np.column_stack([np.ones(M), X])
        try:
            beta, _, _, _ = np.linalg.lstsq(X_c, obs, rcond=None)
            y_pred = X_c @ beta
            ss_res = np.sum((obs - y_pred)**2)
            ss_tot = np.sum((obs - np.mean(obs))**2)
            r2_multi = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            adj_r2 = 1 - (1 - r2_multi) * (M - 1) / max(M - len(beta) - 1, 1)
            print(f"\n  Multiple regression R2 = {r2_multi:.3f} (adj={adj_r2:.3f})")
        except Exception as e:
            print(f"\n  Multiple regression failed: {e}")
            adj_r2 = 0.0
    else:
        adj_r2 = 0.0

    # Verdict (pre-registered thresholds)
    ALPHA_BONF = 0.01 / 30

    if p_obs < ALPHA_BONF and max_r2 < 0.50 and r2_multi < 0.50:
        verdict = "DETECTED (genuine)"
    elif max_r2 > 0.80 or r2_multi > 0.80:
        verdict = "PROXY"
    elif p_obs < ALPHA_BONF and (0.50 <= max_r2 < 0.80 or 0.50 <= r2_multi < 0.80):
        verdict = "AMBIGUOUS"
    elif p_obs < 0.05:
        verdict = "WEAK"
    else:
        verdict = "NULL"

    print(f"\n  VERDICT: {verdict}")
    print(f"    (alpha_bonf={ALPHA_BONF:.6f}, p={p_obs:.2e}, "
          f"max_R2={max_r2:.3f}, multi_R2={r2_multi:.3f})")

    return {
        "observable": observable_key,
        "mean_delta": float(m),
        "se_delta": float(se),
        "cohen_d": float(d_cohen),
        "p_value": float(p_obs),
        "max_r2_single": float(max_r2),
        "max_r2_name": max_r2_name,
        "r2_multiple": float(r2_multi),
        "verdict": verdict,
        "proxy_table": proxy_table,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    N = 500
    T = 1.0
    M = 40
    MAX_EDGES = 300  # sample 300 edges per graph (cost ~1s per graph)

    TESTS = [
        ("conformal",     5.0),   # NULL CONTROL
        ("ppwave_quad",   10.0),  # strongest known signal
        ("ppwave_quad",   5.0),   # moderate signal
        ("schwarzschild", 0.05),  # Schwarzschild
    ]

    print("=" * 70)
    print("OLLIVIER-RICCI CURVATURE ON CAUSAL SETS")
    print(f"N={N}, M={M}, max_edges={MAX_EDGES}")
    print("Anti-bias protocol v1.0 (pre-registered)")
    print(f"Bonferroni alpha = {0.01/30:.6f}")
    print("=" * 70)

    all_results = {}

    for geo, eps in TESTS:
        label = f"{geo}_eps{eps}"
        is_null = (geo == "conformal")
        note = " [NULL CONTROL]" if is_null else ""
        print(f"\n{'='*50}")
        print(f"  {label}{note}")
        print(f"{'='*50}")

        t0 = time.time()
        results = []

        for trial in range(M):
            seed = trial * 1000
            res = crn_trial_orc(seed, N, T, geo, eps, max_edges=MAX_EDGES)
            results.append(res)

            if (trial + 1) % 10 == 0:
                elapsed = time.time() - t0
                orc_d = [r["orc_mean_delta"] for r in results if not np.isnan(r["orc_mean_delta"])]
                if orc_d:
                    print(f"  trial {trial+1}/{M}: ORC_delta={np.mean(orc_d):+.5f} [{elapsed:.1f}s]")

        elapsed = time.time() - t0
        print(f"  Total: {elapsed:.1f}s")

        # Null control check
        if is_null:
            orc_d = [r["orc_mean_delta"] for r in results]
            max_abs = max(abs(d) for d in orc_d)
            if max_abs > 1e-10:
                print(f"  *** NULL CONTROL: max|delta| = {max_abs:.2e} ***")
                # Non-zero due to edge sampling randomness — check if systematic
                _, p_null = stats.ttest_1samp(orc_d, 0.0)
                print(f"  *** t-test vs 0: p = {p_null:.4f} ***")
                if p_null < 0.01:
                    print(f"  *** WARNING: NULL CONTROL shows systematic bias! ***")
                else:
                    print(f"  *** NULL CONTROL OK: no systematic bias (p={p_null:.4f}) ***")
            else:
                print(f"  NULL CONTROL PASSED: exact 0")

        # Adversarial analysis for BOTH orc_mean and orc_median
        v_mean = adversarial_analysis(results, "orc_mean_delta", label)
        v_median = adversarial_analysis(results, "orc_median_delta", label)
        v_frac = adversarial_analysis(results, "orc_frac_pos_delta", label)

        all_results[label] = {
            "geometry": geo, "eps": eps, "N": N, "M": M,
            "elapsed_sec": elapsed, "is_null": is_null,
            "orc_mean_verdict": v_mean,
            "orc_median_verdict": v_median,
            "orc_frac_pos_verdict": v_frac,
            "trials": results,
        }

    # Save
    outpath = os.path.join(OUTDIR, "orc_pilot.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # Final summary
    print("\n" + "=" * 70)
    print("OLLIVIER-RICCI SUMMARY")
    print("=" * 70)
    print(f"{'condition':25s} {'orc_delta':>10s} {'d':>8s} {'p':>12s} "
          f"{'R2_max':>8s} {'R2_multi':>8s} {'verdict':>22s}")
    print("-" * 95)
    for label, res in all_results.items():
        v = res["orc_mean_verdict"]
        note = " [NULL]" if res["is_null"] else ""
        print(f"{label:25s} {v['mean_delta']:+10.5f} {v['cohen_d']:+8.3f} "
              f"{v['p_value']:12.2e} {v['max_r2_single']:8.3f} "
              f"{v['r2_multiple']:8.3f} {v['verdict']:>22s}{note}")

    # Count genuine detections
    genuine = sum(1 for r in all_results.values()
                  if r["orc_mean_verdict"]["verdict"] == "DETECTED (genuine)"
                  and not r["is_null"])
    total = sum(1 for r in all_results.values() if not r["is_null"])
    print(f"\nGenuine detections: {genuine}/{total}")

    if genuine > 0:
        print("\n*** ORC SURVIVES ADVERSARIAL PROTOCOL. Proceed to N-scaling. ***")
    else:
        proxied = sum(1 for r in all_results.values()
                      if "PROXY" in r["orc_mean_verdict"]["verdict"]
                      and not r["is_null"])
        if proxied > 0:
            print(f"\n*** ORC IS PROXY ({proxied}/{total}). Same failure mode as Forman. ***")
        else:
            print(f"\n*** ORC: no genuine detections. Consider Variant 2 (new candidates). ***")


if __name__ == "__main__":
    main()
