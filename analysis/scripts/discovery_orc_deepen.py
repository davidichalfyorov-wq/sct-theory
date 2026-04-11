"""
Discovery Run 001 — ORC Deepening: 3 Diagnostic Tests
======================================================

Test A: Edge sampling effect
  - Does sampling 300/20000 edges at N=1000 introduce bias?
  - Run N=500 with ALL edges vs 300 edges → compare R²_multi
  - Run N=1000 with 1000 edges vs 300 edges

Test B: Holdout geometries (NEVER used in development)
  - ppwave_cross at eps=20 (different polarization, high eps)
  - "tidal field" metric: Phi = -eps*(x²+y²)/2, anisotropic potential

Test C: Residual analysis at N=1000
  - Regress ORC_delta on ALL degree stats
  - Is the residual still significant?
  - This directly answers: does ORC carry info beyond degree stats?

Author: David Alfyorov
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import *
from discovery_orc_pilot import compute_orc, adversarial_analysis
import json, time
import ot

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
os.makedirs(OUTDIR, exist_ok=True)

T = 1.0
M = 30

# ---------------------------------------------------------------------------
# New holdout metric: tidal field (quadrupole)
# ---------------------------------------------------------------------------
def causal_tidal(pts, eps):
    """Tidal field metric: Phi = -eps*(x^2+y^2)/2 (cylindrical potential).

    ds^2 = -(1+2*Phi)dt^2 + (1-2*Phi)(dx^2+dy^2+dz^2)
    Causal: (1+2*Phi)*dt^2 > (1-2*Phi)*dr^2

    This is Ricci-nonzero (unlike pp-wave which is Ricci-flat).
    NOT used in any previous test.
    """
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dr2 = ((x[np.newaxis, :] - x[:, np.newaxis])**2 +
           (y[np.newaxis, :] - y[:, np.newaxis])**2 +
           (z[np.newaxis, :] - z[:, np.newaxis])**2)
    xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
    ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
    Phi = -eps * (xm**2 + ym**2) / 2.0
    return (((1 + 2*Phi) * dt**2 > (1 - 2*Phi) * dr2) & (dt > 0)).astype(np.float64)


# Add to metric functions
METRIC_FNS["tidal"] = causal_tidal
SEED_OFFSETS["tidal"] = 600

# Also need ppwave_cross
from discovery_common import causal_ppwave_cross
METRIC_FNS["ppwave_cross"] = causal_ppwave_cross


# ---------------------------------------------------------------------------
# CRN trial (same as ORC pilot but with configurable max_edges)
# ---------------------------------------------------------------------------
def crn_trial_orc_full(seed, N, T, metric_name, eps, max_edges):
    """ORC CRN trial with configurable edge sampling."""
    from scipy.sparse.csgraph import shortest_path

    seed_offset = SEED_OFFSETS.get(metric_name, 700)
    rng = np.random.default_rng(seed + seed_offset)
    pts = sprinkle_4d(N, T, rng)

    result = {"seed": seed, "N": N, "metric": metric_name, "eps": eps,
              "max_edges": max_edges}

    for tag, C_fn in [("flat", lambda: causal_flat(pts)),
                       ("curv", lambda: METRIC_FNS[metric_name](pts, eps))]:
        C = C_fn()
        A = build_link_graph(C)
        del C; gc.collect()

        gs, deg = graph_statistics(A)
        fr, _ = forman_ricci(A, deg)
        orc = compute_orc(A, max_edges=max_edges)
        del A; gc.collect()

        result[f"orc_mean_{tag}"] = orc["orc_mean"]
        result[f"orc_median_{tag}"] = orc["orc_median"]
        result[f"forman_mean_{tag}"] = fr["F_mean"]
        result[f"n_edges_sampled_{tag}"] = orc["n_edges_sampled"]
        for key in gs:
            result[f"{key}_{tag}"] = gs[key]

    # Deltas
    result["orc_mean_delta"] = result["orc_mean_curv"] - result["orc_mean_flat"]
    result["orc_median_delta"] = result["orc_median_curv"] - result["orc_median_flat"]
    result["forman_mean_delta"] = result["forman_mean_curv"] - result["forman_mean_flat"]
    for key in ["mean_degree", "degree_var", "degree_std", "degree_skew",
                "degree_kurt", "edge_count", "max_degree", "min_degree",
                "assortativity"]:
        result[f"{key}_delta"] = result.get(f"{key}_curv", 0) - result.get(f"{key}_flat", 0)

    return result


# ---------------------------------------------------------------------------
# Residual analysis (regress out all stats, test remainder)
# ---------------------------------------------------------------------------
def residual_analysis(results, label=""):
    """After regressing ORC on all proxy stats, is the residual significant?"""
    M = len(results)
    orc = np.array([r["orc_mean_delta"] for r in results])

    proxy_names = ["mean_degree_delta", "degree_var_delta", "degree_std_delta",
                   "degree_skew_delta", "degree_kurt_delta", "edge_count_delta",
                   "max_degree_delta", "assortativity_delta", "forman_mean_delta"]

    X_cols = []
    col_names = []
    for name in proxy_names:
        vals = np.array([r.get(name, 0) for r in results], dtype=float)
        if np.std(vals) > 1e-15:
            X_cols.append(vals)
            col_names.append(name)

    if not X_cols:
        print(f"  {label}: no valid proxy stats")
        return {"residual_mean": float(np.mean(orc)), "residual_p": 0.0}

    X = np.column_stack(X_cols)
    X_c = np.column_stack([np.ones(M), X])

    beta, _, _, _ = np.linalg.lstsq(X_c, orc, rcond=None)
    predicted = X_c @ beta
    residual = orc - predicted

    mean_r = float(np.mean(residual))
    se_r = float(np.std(residual) / np.sqrt(M))
    _, p_r = stats.ttest_1samp(residual, 0.0) if M >= 5 else (0, 1)

    print(f"\n  === {label}: Residual after regressing out ALL proxy stats ===")
    print(f"  ORC original:   mean={np.mean(orc):+.5f} +/- {np.std(orc)/np.sqrt(M):.5f}")
    print(f"  ORC residual:   mean={mean_r:+.5f} +/- {se_r:.5f}, p={p_r:.2e}")
    print(f"  Residual std:   {np.std(residual):.5f} (vs original {np.std(orc):.5f})")
    print(f"  Variance explained: {1 - np.var(residual)/np.var(orc):.1%}")
    print(f"  Variance REMAINING: {np.var(residual)/np.var(orc):.1%}")

    if p_r < 0.01:
        print(f"  => RESIDUAL SIGNIFICANT (p={p_r:.2e}): ORC carries info beyond ALL proxies")
    elif p_r < 0.05:
        print(f"  => RESIDUAL MARGINALLY SIGNIFICANT (p={p_r:.2e})")
    else:
        print(f"  => RESIDUAL NOT SIGNIFICANT (p={p_r:.2e}): all ORC info is in proxy stats")

    return {
        "residual_mean": mean_r, "residual_se": se_r, "residual_p": float(p_r),
        "variance_explained": float(1 - np.var(residual)/np.var(orc)),
        "variance_remaining": float(np.var(residual)/np.var(orc)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    all_results = {}

    # =======================================================================
    # TEST A: Edge sampling effect
    # =======================================================================
    print("=" * 70)
    print("TEST A: Edge Sampling Effect (does sampling bias cause AMBIGUOUS?)")
    print("=" * 70)

    for N, max_e_list in [(500, [100, 300, 5000]), (1000, [300, 1000, 5000])]:
        for max_e in max_e_list:
            label = f"schw_N{N}_edges{max_e}"
            print(f"\n--- {label} ---")
            t0 = time.time()
            results = []
            for trial in range(M):
                res = crn_trial_orc_full(trial*1000, N, T, "schwarzschild", 0.05, max_e)
                results.append(res)
            elapsed = time.time() - t0

            v = adversarial_analysis(results, "orc_mean_delta", label)
            ra = residual_analysis(results, label)
            v["residual_analysis"] = ra
            v["elapsed_sec"] = elapsed
            all_results[f"testA_{label}"] = v
            print(f"  [{elapsed:.1f}s]")

    # =======================================================================
    # TEST B: Holdout geometries
    # =======================================================================
    print("\n" + "=" * 70)
    print("TEST B: Holdout Geometries (NEVER used in development)")
    print("=" * 70)

    holdouts = [
        ("ppwave_cross", 20.0, 500, 300),
        ("tidal",        0.05, 500, 300),
    ]

    for geo, eps, N, max_e in holdouts:
        label = f"{geo}_eps{eps}_N{N}"
        print(f"\n--- {label} [HOLDOUT] ---")
        t0 = time.time()
        results = []
        for trial in range(M):
            res = crn_trial_orc_full(trial*1000, N, T, geo, eps, max_e)
            results.append(res)
        elapsed = time.time() - t0

        v = adversarial_analysis(results, "orc_mean_delta", label)
        ra = residual_analysis(results, label)
        v["residual_analysis"] = ra
        v["elapsed_sec"] = elapsed
        v["is_holdout"] = True
        all_results[f"testB_{label}"] = v
        print(f"  [{elapsed:.1f}s]")

    # =======================================================================
    # TEST C: Residual analysis at N=1000 (more trials for power)
    # =======================================================================
    print("\n" + "=" * 70)
    print("TEST C: Residual Analysis (N=1000, M=50, Schwarzschild)")
    print("=" * 70)

    M_C = 50
    label = "schw_N1000_M50_residual"
    print(f"\n--- {label} ---")
    t0 = time.time()
    results = []
    for trial in range(M_C):
        res = crn_trial_orc_full(trial*1000, 1000, T, "schwarzschild", 0.05, 500)
        results.append(res)
        if (trial+1) % 10 == 0:
            print(f"  trial {trial+1}/{M_C} [{time.time()-t0:.1f}s]")
    elapsed = time.time() - t0

    v = adversarial_analysis(results, "orc_mean_delta", label)
    ra = residual_analysis(results, label)
    v["residual_analysis"] = ra
    v["elapsed_sec"] = elapsed
    all_results["testC_residual"] = v

    # Save
    outpath = os.path.join(OUTDIR, "orc_deepen.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # =======================================================================
    # SUMMARY
    # =======================================================================
    print("\n" + "=" * 70)
    print("DEEPENING SUMMARY")
    print("=" * 70)

    print("\n  TEST A: Edge sampling effect on R²_multi")
    for key, v in all_results.items():
        if key.startswith("testA_"):
            label = key[6:]
            print(f"    {label:30s}: R²_multi={v['r2_multiple']:.3f}, "
                  f"verdict={v['verdict']}, "
                  f"resid_p={v['residual_analysis']['residual_p']:.2e}")

    print("\n  TEST B: Holdout geometries")
    for key, v in all_results.items():
        if key.startswith("testB_"):
            label = key[6:]
            print(f"    {label:30s}: delta={v['mean_delta']:+.5f}, d={v['cohen_d']:+.3f}, "
                  f"R²_multi={v['r2_multiple']:.3f}, verdict={v['verdict']}")

    print("\n  TEST C: N=1000 residual after removing ALL proxies")
    v = all_results["testC_residual"]
    ra = v["residual_analysis"]
    print(f"    Variance explained by proxies: {ra['variance_explained']:.1%}")
    print(f"    Variance remaining (ORC-only): {ra['variance_remaining']:.1%}")
    print(f"    Residual p-value: {ra['residual_p']:.2e}")
    if ra["residual_p"] < 0.01:
        print(f"    => ORC CARRIES INDEPENDENT INFO even at N=1000")
    else:
        print(f"    => ORC info fully captured by proxies at N=1000")


if __name__ == "__main__":
    main()
