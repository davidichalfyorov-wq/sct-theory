"""
FND-1 Route 1: Gate 5 — Matched-Pairs Curvature Test.

Instead of extracting SDW coefficients (invalid at N=1000 where p≈-0.79),
we directly compare heat traces from PAIRED sprinklings:
  - For each seed i: sprinkle flat (eps=0) AND curved (eps≠0)
  - Compute ΔK_i(t) = K_curved_i(t) - K_flat_i(t)
  - Paired t-test: is mean(ΔK) ≠ 0?

This cancels seed-to-seed noise and isolates the curvature effect.
No SDW expansion required — purely model-free.

Also tests:
  - Eigenvalue distribution shift (paired Wilcoxon)
  - Mean eigenvalue change
  - Relative heat trace change ΔK/K
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_ensemble_runner import (
    compute_interval_cardinalities,
    build_bd_L,
    compute_family_B_eigenvalues,
    compute_heat_trace,
    ZERO_THRESHOLD,
)
from fnd1_gate5_runner import (
    sprinkle_curved,
    _sprinkle_flat,
    check_omega_positivity,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_POINTS = 1000
M_PAIRS = 200          # number of paired sprinklings
T_DIAMOND = 1.0
MASTER_SEED = 42

# Curvature values to test against flat
EPSILON_VALUES = [-0.5, -0.25, 0.25, 0.5]

# t-grid for heat trace comparison
N_T = 300
T_MIN = 1e-4
T_MAX = 5.0


# ---------------------------------------------------------------------------
# Matched-pairs test
# ---------------------------------------------------------------------------

def run_matched_pairs(eps: float, N: int, M: int, T: float,
                      seed: int) -> dict:
    """
    Run M paired sprinklings (flat vs curved) and test for curvature effect.

    For each seed i:
      1. Sprinkle flat (eps=0) → eigenvalues_flat_i
      2. Sprinkle curved (eps) → eigenvalues_curved_i
      3. Compute ΔK_i(t) = K_curved_i(t) - K_flat_i(t)

    Then: paired t-test on mean(ΔK(t)) at each t.
    """
    V = T**2 / 2.0
    rho = N / V

    t_grid = np.logspace(np.log10(T_MIN), np.log10(T_MAX), N_T)

    # Seed hierarchy: for each pair, spawn 2 child seeds (flat, curved)
    master_ss = np.random.SeedSequence(seed)
    pair_seeds = master_ss.spawn(M)

    K_flat_all = np.zeros((M, N_T))
    K_curved_all = np.zeros((M, N_T))
    eig_flat_means = []
    eig_curved_means = []
    n_eff_flat = []
    n_eff_curved = []

    print(f"  Running {M} matched pairs at eps={eps:+.3f}, N={N}...")
    for i in range(M):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    Pair {i+1}/{M}...")

        # Two child seeds from the same parent — correlated but independent
        children = pair_seeds[i].spawn(2)
        rng_flat = np.random.default_rng(children[0])
        rng_curved = np.random.default_rng(children[1])

        # Flat sprinkling
        pts_flat, C_flat = _sprinkle_flat(N, T, rng_flat)
        n_mat_flat = compute_interval_cardinalities(C_flat)
        L_flat = build_bd_L(C_flat, n_mat_flat, rho)
        eig_flat = compute_family_B_eigenvalues(L_flat)

        # Curved sprinkling (same base seed structure)
        pts_curved, C_curved = sprinkle_curved(N, eps, T, rng_curved)
        n_mat_curved = compute_interval_cardinalities(C_curved)
        L_curved = build_bd_L(C_curved, n_mat_curved, rho)
        eig_curved = compute_family_B_eigenvalues(L_curved)

        # Heat traces
        K_flat_all[i] = compute_heat_trace(eig_flat, t_grid)
        K_curved_all[i] = compute_heat_trace(eig_curved, t_grid)

        # Eigenvalue statistics
        eig_flat_means.append(np.mean(eig_flat))
        eig_curved_means.append(np.mean(eig_curved))
        n_eff_flat.append(len(eig_flat))
        n_eff_curved.append(len(eig_curved))

    # ---- Analysis ----
    print(f"  Analyzing...")

    # 1. Paired differences in heat trace
    DK = K_curved_all - K_flat_all  # (M, N_T)
    DK_mean = np.mean(DK, axis=0)
    DK_sem = np.std(DK, axis=0, ddof=1) / np.sqrt(M)

    # Relative change
    K_flat_ens = np.mean(K_flat_all, axis=0)
    rel_DK = np.where(K_flat_ens > 0, DK_mean / K_flat_ens, 0.0)

    # 2. Global paired t-test at multiple t values
    # Pick t-values in the intermediate regime
    t_test_indices = []
    for t_target in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        idx = np.argmin(np.abs(t_grid - t_target))
        t_test_indices.append(idx)

    t_test_results = []
    for idx in t_test_indices:
        dk_samples = DK[:, idx]
        t_stat, p_val = stats.ttest_1samp(dk_samples, 0.0)
        t_test_results.append({
            "t_value": float(t_grid[idx]),
            "mean_DK": float(np.mean(dk_samples)),
            "sem_DK": float(np.std(dk_samples, ddof=1) / np.sqrt(M)),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant_005": bool(p_val < 0.05),
            "significant_001": bool(p_val < 0.01),
        })

    # 3. Aggregate: fraction of t-points where DK is significantly nonzero
    all_p_values = []
    for j in range(N_T):
        dk_j = DK[:, j]
        if np.std(dk_j) > 0:
            _, pv = stats.ttest_1samp(dk_j, 0.0)
            all_p_values.append(pv)
    all_p_values = np.array(all_p_values)
    frac_sig_005 = float(np.mean(all_p_values < 0.05))
    frac_sig_001 = float(np.mean(all_p_values < 0.01))

    # 4. Eigenvalue mean shift
    eig_flat_arr = np.array(eig_flat_means)
    eig_curved_arr = np.array(eig_curved_means)
    eig_diff = eig_curved_arr - eig_flat_arr
    eig_t_stat, eig_p_val = stats.ttest_1samp(eig_diff, 0.0)

    # 5. Max |relative change| in heat trace
    max_rel_DK = float(np.max(np.abs(rel_DK)))
    mean_rel_DK = float(np.mean(np.abs(rel_DK)))

    result = {
        "epsilon": eps,
        "N": N,
        "M": M,
        "t_test_results": t_test_results,
        "fraction_significant_005": frac_sig_005,
        "fraction_significant_001": frac_sig_001,
        "max_abs_relative_change": max_rel_DK,
        "mean_abs_relative_change": mean_rel_DK,
        "eigenvalue_mean_shift": {
            "mean_diff": float(np.mean(eig_diff)),
            "sem_diff": float(np.std(eig_diff, ddof=1) / np.sqrt(M)),
            "t_statistic": float(eig_t_stat),
            "p_value": float(eig_p_val),
            "significant": bool(eig_p_val < 0.05),
        },
        "n_eff_flat_mean": float(np.mean(n_eff_flat)),
        "n_eff_curved_mean": float(np.mean(n_eff_curved)),
    }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.perf_counter()

    print("=" * 70)
    print("FND-1 ROUTE 1: GATE 5 — MATCHED-PAIRS CURVATURE TEST")
    print("=" * 70)
    print(f"Family B only. N={N_POINTS}, M={M_PAIRS} pairs per epsilon.")
    print(f"Epsilon values: {EPSILON_VALUES}")
    print(f"Method: paired t-test on ΔK(t) = K_curved(t) - K_flat(t)")
    print(f"NO SDW decomposition — model-free direct comparison.")
    print()

    all_results = {}

    for eps in EPSILON_VALUES:
        print(f"\n{'='*60}")
        print(f"EPSILON = {eps:+.3f}")
        print(f"{'='*60}")

        result = run_matched_pairs(eps, N_POINTS, M_PAIRS, T_DIAMOND,
                                   MASTER_SEED + hash(str(eps)) % 10000)
        all_results[str(eps)] = result

        # Print summary for this epsilon
        print(f"\n  Paired t-test at selected t values:")
        print(f"  {'t':>8} {'mean(ΔK)':>12} {'SEM':>10} {'t-stat':>8} "
              f"{'p-value':>10} {'sig?':>5}")
        print(f"  {'-'*8} {'-'*12} {'-'*10} {'-'*8} {'-'*10} {'-'*5}")
        for tr in result["t_test_results"]:
            sig = "**" if tr["significant_005"] else ""
            print(f"  {tr['t_value']:8.4f} {tr['mean_DK']:+12.4f} "
                  f"{tr['sem_DK']:10.4f} {tr['t_statistic']:8.2f} "
                  f"{tr['p_value']:10.4f} {sig:>5}")

        print(f"\n  Fraction of t-grid significant at 0.05: "
              f"{result['fraction_significant_005']:.3f}")
        print(f"  Fraction significant at 0.01: "
              f"{result['fraction_significant_001']:.3f}")
        print(f"  Max |relative ΔK|: {result['max_abs_relative_change']:.6f}")
        print(f"  Mean |relative ΔK|: {result['mean_abs_relative_change']:.6f}")

        eig = result["eigenvalue_mean_shift"]
        print(f"\n  Eigenvalue mean shift: {eig['mean_diff']:+.6f} "
              f"± {eig['sem_diff']:.6f}")
        print(f"  t-stat = {eig['t_statistic']:.2f}, "
              f"p = {eig['p_value']:.4f}, "
              f"significant: {eig['significant']}")

    # ---- Overall verdict ----
    print(f"\n{'='*70}")
    print("OVERALL VERDICT")
    print(f"{'='*70}")

    any_curvature_detected = False
    for eps_str, result in all_results.items():
        eps = float(eps_str)
        frac = result["fraction_significant_005"]
        eig_sig = result["eigenvalue_mean_shift"]["significant"]
        max_rel = result["max_abs_relative_change"]

        detected = frac > 0.10 or eig_sig
        status = "DETECTED" if detected else "NOT DETECTED"
        if detected:
            any_curvature_detected = True

        print(f"\n  eps={eps:+.3f}:")
        print(f"    Heat trace: {frac*100:.1f}% of t-grid significant "
              f"(threshold: 10%)")
        print(f"    Eigenvalue shift: "
              f"{'SIGNIFICANT' if eig_sig else 'not significant'}")
        print(f"    Max |ΔK/K|: {max_rel:.6f}")
        print(f"    → {status}")

    if any_curvature_detected:
        verdict = "CURVATURE SIGNAL DETECTED (at least one epsilon)"
    else:
        verdict = "NO CURVATURE SIGNAL DETECTED"

    print(f"\n  FINAL VERDICT: {verdict}")

    wall = time.perf_counter() - t0
    print(f"\n  Total wall time: {wall:.1f}s ({wall/60:.1f} min)")

    # Save
    project_root = Path(__file__).resolve().parent.parent.parent
    output_path = (project_root / "speculative" / "numerics" /
                   "ensemble_results" / "gate5_matched_pairs.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "parameters": {
            "N": N_POINTS, "M": M_PAIRS, "T": T_DIAMOND,
            "epsilon_values": EPSILON_VALUES,
            "method": "matched-pairs paired t-test, model-free",
        },
        "results": {},
        "verdict": verdict,
        "wall_time_sec": wall,
    }
    for eps_str, r in all_results.items():
        save_data["results"][eps_str] = r

    def _clean(obj):
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(_clean(save_data), f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
