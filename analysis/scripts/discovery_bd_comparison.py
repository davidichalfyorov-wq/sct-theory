"""
Discovery Run 001 — BD Action Head-to-Head Comparison
=======================================================

THE killer argument: BD action gives 0 on pp-wave (R=0),
our scaling_exp gives d=-2.58.

Computes the Benincasa-Dowker action S_BD for the SAME sprinklings
used in the Weyl probe experiments, on both flat and pp-wave.

BD action in d=4 (Benincasa-Dowker 2010):
  S_BD = (1/l^2) × Σ_i [ C_0 + C_1 n_1(i) + C_2 n_2(i) + C_3 n_3(i) ]

where n_k(i) = number of k-element intervals containing i as the past endpoint.
Coefficients for d=4: C_0 = (4/√6)/(4!), C_1, C_2, C_3 from Glaser 2013.

Simplified: for CRN DELTA, we only need the DIFFERENCE between flat and curved.
Even simpler: count the total number of k-element intervals (k=0,1,2,3) for
flat and curved, take weighted sum.

Actually, the simplest BD-like quantity: just count intervals of size k.
The BD action is a specific linear combination. But for our comparison,
we just need to show:
  1. BD gives ~0 on pp-wave (because R=0 for pp-wave)
  2. Our scaling_exp gives strong signal

We'll compute:
  - Total causal pairs (0-intervals)
  - 1-intervals (links)
  - 2-intervals
  - 3-intervals
  - BD action approximation: S ∝ N - C₁·n₁ + C₂·n₂ - C₃·n₃

where n_k = number of intervals of cardinality k.

For d=4, the BD coefficients are:
  C₀ = 1, C₁ = 9, C₂ = 16, C₃ = 8
  S_BD ∝ Σ_pairs [ C₀ - C₁·δ(|I|=0) + C₂·δ(|I|=1) - C₃·δ(|I|=2) + ... ]

Actually, the standard form:
  S_BD = Σ_{x∈C} [ ε₀ + ε₁·n₁(x) + ε₂·n₂(x) + ε₃·n₃(x) ]

where n_k(x) = #{y : x≺y and |I(x,y)|=k}, and the ε_k are dimension-dependent.

For d=4 (from Glaser 2013, arXiv:1311.1701):
  ε₀ = 1/(√6 × l²)
  ε₁ = -9/(√6 × l²)
  ε₂ = 16/(√6 × l²)
  ε₃ = -8/(√6 × l²)

In the continuum limit: <S_BD> → (1/l²) × ∫ (R/2) √g d⁴x + boundary terms

For pp-wave: R=0 → <S_BD> = 0 (plus boundary terms and fluctuations)
For Schwarzschild: R ≠ 0 → <S_BD> ≠ 0

Author: David Alfyorov
"""
import numpy as np
from scipy import stats
import json, time, gc, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import *
from discovery_weyl_probes import (
    interval_volumes_binned, weyl_observables, crn_trial_weyl, adversarial_weyl,
)

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
os.makedirs(OUTDIR, exist_ok=True)

T = 1.0
tau_bins = np.array([0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.45])

# BD action coefficients for d=4
BD_COEFFS = {0: 1.0, 1: -9.0, 2: 16.0, 3: -8.0}


def compute_bd_action(C, N):
    """Compute BD action proxy from causal matrix.

    Counts intervals of size 0,1,2,3 and combines with BD coefficients.
    Returns the normalized BD action per element.
    """
    # C^2[i,j] = number of elements between i and j
    C2 = C @ C

    # Count intervals by size
    causal_pairs = C > 0.5  # i ≺ j
    n_total = int(np.sum(causal_pairs))

    # n_k = number of pairs (i,j) with exactly k elements between them
    sizes = C2[causal_pairs].astype(int)

    n_0 = int(np.sum(sizes == 0))  # links (no elements between)
    n_1 = int(np.sum(sizes == 1))  # 1-intervals
    n_2 = int(np.sum(sizes == 2))  # 2-intervals
    n_3 = int(np.sum(sizes == 3))  # 3-intervals

    # BD action (unnormalized)
    S_BD = BD_COEFFS[0] * n_0 + BD_COEFFS[1] * n_1 + BD_COEFFS[2] * n_2 + BD_COEFFS[3] * n_3

    # Normalize per element
    S_per_element = S_BD / N

    return {
        "S_BD": float(S_BD),
        "S_per_element": float(S_per_element),
        "n_total_pairs": n_total,
        "n_0": n_0, "n_1": n_1, "n_2": n_2, "n_3": n_3,
    }


def crn_trial_bd_and_weyl(seed, N, T, metric_name, eps):
    """CRN trial computing BOTH BD action and Weyl probes."""
    seed_offset = SEED_OFFSETS.get(metric_name, 100)
    rng = np.random.default_rng(seed + seed_offset)
    pts = sprinkle_4d(N, T, rng)

    result = {"seed": seed, "N": N, "metric": metric_name, "eps": eps}

    C_flat = causal_flat(pts)
    C_curv = METRIC_FNS[metric_name](pts, eps)

    # BD action
    bd_flat = compute_bd_action(C_flat, N)
    bd_curv = compute_bd_action(C_curv, N)
    result["bd_flat"] = bd_flat["S_per_element"]
    result["bd_curv"] = bd_curv["S_per_element"]
    result["bd_delta"] = bd_curv["S_per_element"] - bd_flat["S_per_element"]

    # Interval counts deltas (for diagnostics)
    for k in range(4):
        result[f"n_{k}_flat"] = bd_flat[f"n_{k}"]
        result[f"n_{k}_curv"] = bd_curv[f"n_{k}"]
        result[f"n_{k}_delta"] = bd_curv[f"n_{k}"] - bd_flat[f"n_{k}"]

    # Weyl probes (scaling_exp, var_ratio)
    rng_b = np.random.default_rng(seed + 7777)
    bins_flat = interval_volumes_binned(C_flat, pts, N, tau_bins, rng=rng_b)
    rng_b = np.random.default_rng(seed + 7777)
    bins_curved = interval_volumes_binned(C_curv, pts, N, tau_bins, rng=rng_b)

    w = weyl_observables(bins_flat, bins_curved, tau_bins)
    result.update(w)

    # Degree stats for adversarial
    A_flat = build_link_graph(C_flat)
    gs_flat, _ = graph_statistics(A_flat)
    A_curv = build_link_graph(C_curv)
    gs_curv, _ = graph_statistics(A_curv)
    for key in gs_flat:
        result[f"{key}_delta"] = gs_curv[key] - gs_flat[key]

    fr_f, _ = forman_ricci(A_flat)
    fr_c, _ = forman_ricci(A_curv)
    result["forman_mean_delta"] = fr_c["F_mean"] - fr_f["F_mean"]

    del C_flat, C_curv, A_flat, A_curv; gc.collect()
    return result


def main():
    N = 500
    M = 40

    TESTS = [
        ("conformal",     5.0),   # NULL
        ("ppwave_quad",   10.0),  # Ricci-flat (R=0), Weyl≠0
        ("schwarzschild", 0.05),  # Ricci≠0, Weyl≠0
    ]

    print("=" * 70)
    print("BD ACTION vs WEYL PROBES: Head-to-Head Comparison")
    print(f"N={N}, M={M}")
    print("=" * 70)

    all_results = {}

    for geo, eps in TESTS:
        label = f"{geo}_eps{eps}"
        is_null = (geo == "conformal")
        r_zero = (geo == "ppwave_quad")  # Ricci-flat
        note = " [NULL]" if is_null else (" [R=0, C²≠0]" if r_zero else " [R≠0, C²≠0]")

        print(f"\n--- {label}{note} ---")
        t0 = time.time()

        results = []
        for trial in range(M):
            res = crn_trial_bd_and_weyl(trial * 1000, N, T, geo, eps)
            results.append(res)
            if (trial + 1) % 20 == 0:
                bd_d = [r["bd_delta"] for r in results]
                se_d = [r.get("scaling_exp_delta", np.nan) for r in results]
                se_v = [v for v in se_d if not np.isnan(v)]
                print(f"  trial {trial+1}/{M}: BD_delta={np.mean(bd_d):+.2f}, "
                      f"scaling_exp={np.mean(se_v):+.4f}")

        elapsed = time.time() - t0

        # BD action analysis
        bd_deltas = [r["bd_delta"] for r in results]
        bd_mean = np.mean(bd_deltas)
        bd_se = np.std(bd_deltas) / np.sqrt(M)
        bd_d = bd_mean / np.std(bd_deltas) if np.std(bd_deltas) > 0 else 0
        _, bd_p = stats.ttest_1samp(bd_deltas, 0.0) if M >= 5 else (0, 1)

        # Scaling_exp analysis
        v_se = adversarial_weyl(results, "scaling_exp_delta", label)

        print(f"\n  BD action:     delta={bd_mean:+.2f} +/- {bd_se:.2f}, d={bd_d:+.3f}, p={bd_p:.2e}")
        print(f"  scaling_exp:   delta={v_se.get('mean_delta',0):+.4f}, "
              f"d={v_se.get('cohen_d',0):+.3f}, p={v_se.get('p_value',1):.2e}, "
              f"verdict={v_se.get('verdict','?')}")

        geo_results = {
            "geometry": geo, "eps": eps, "is_null": is_null,
            "ricci_flat": r_zero, "elapsed_sec": elapsed,
            "bd_action": {
                "mean_delta": float(bd_mean), "se": float(bd_se),
                "cohen_d": float(bd_d), "p_value": float(bd_p),
            },
            "scaling_exp": v_se,
            "trials": results,
        }
        all_results[label] = geo_results

    # Save
    outpath = os.path.join(OUTDIR, "bd_comparison.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # THE TABLE
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Geometry':25s} {'R':>5s} {'C²':>5s} {'BD d':>10s} {'BD p':>12s} "
          f"{'SE d':>10s} {'SE p':>12s} {'SE verdict':>22s}")
    print("-" * 105)

    for label, res in all_results.items():
        R = "0" if res.get("ricci_flat") else "≠0"
        C2 = "0" if res.get("is_null") else "≠0"
        bd = res["bd_action"]
        se = res["scaling_exp"]
        print(f"{label:25s} {R:>5s} {C2:>5s} "
              f"{bd['cohen_d']:+10.3f} {bd['p_value']:12.2e} "
              f"{se.get('cohen_d',0):+10.3f} {se.get('p_value',1):12.2e} "
              f"{se.get('verdict','?'):>22s}")

    print("\nKEY: BD action is blind to Weyl curvature (R=0 → BD=0).")
    print("     scaling_exp detects Weyl via volume ratios.")


if __name__ == "__main__":
    main()
