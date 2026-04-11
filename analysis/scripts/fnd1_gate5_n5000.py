"""
FND-1 Route 1: Gate 5 — Matched-Pairs at N=5000.

Follow-up on borderline signal at eps=-0.5 (17.7% of t-grid significant).
Tests only eps = -0.5 and +0.5 with N=5000, M=50 pairs.

If signal strengthens at BOTH eps values → genuine curvature detection.
If signal disappears or stays asymmetric → noise, Route 1 closed.
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
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_POINTS = 5000
M_PAIRS = 50
T_DIAMOND = 1.0
MASTER_SEED = 42
EPSILON_VALUES = [-0.5, 0.5]

N_T = 300
T_MIN = 1e-5
T_MAX = 5.0


def run_paired_test(eps, N, M, T, seed):
    """Run M matched pairs and return detailed results."""
    V = T**2 / 2.0
    rho = N / V
    t_grid = np.logspace(np.log10(T_MIN), np.log10(T_MAX), N_T)

    master_ss = np.random.SeedSequence(seed)
    pair_seeds = master_ss.spawn(M)

    K_flat_all = np.zeros((M, N_T))
    K_curved_all = np.zeros((M, N_T))
    eig_flat_means = []
    eig_curved_means = []

    print(f"  Running {M} matched pairs at eps={eps:+.3f}, N={N}...")
    for i in range(M):
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.perf_counter() - t0_pair
            eta = elapsed / (i + 1) * M - elapsed if i > 0 else 0
            print(f"    Pair {i+1}/{M}... "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        children = pair_seeds[i].spawn(2)
        rng_flat = np.random.default_rng(children[0])
        rng_curved = np.random.default_rng(children[1])

        # Flat
        pts_flat, C_flat = _sprinkle_flat(N, T, rng_flat)
        n_mat_flat = compute_interval_cardinalities(C_flat)
        L_flat = build_bd_L(C_flat, n_mat_flat, rho)
        eig_flat = compute_family_B_eigenvalues(L_flat)

        # Curved
        pts_curved, C_curved = sprinkle_curved(N, eps, T, rng_curved)
        n_mat_curved = compute_interval_cardinalities(C_curved)
        L_curved = build_bd_L(C_curved, n_mat_curved, rho)
        eig_curved = compute_family_B_eigenvalues(L_curved)

        K_flat_all[i] = compute_heat_trace(eig_flat, t_grid)
        K_curved_all[i] = compute_heat_trace(eig_curved, t_grid)
        eig_flat_means.append(np.mean(eig_flat))
        eig_curved_means.append(np.mean(eig_curved))

    # Analysis
    DK = K_curved_all - K_flat_all
    DK_mean = np.mean(DK, axis=0)
    K_flat_ens = np.mean(K_flat_all, axis=0)
    rel_DK = np.where(K_flat_ens > 0, DK_mean / K_flat_ens, 0.0)

    # Paired t-test at each t
    p_values = []
    for j in range(N_T):
        dk_j = DK[:, j]
        if np.std(dk_j) > 0:
            _, pv = stats.ttest_1samp(dk_j, 0.0)
            p_values.append(pv)
        else:
            p_values.append(1.0)
    p_values = np.array(p_values)

    frac_sig_005 = float(np.mean(p_values < 0.05))
    frac_sig_001 = float(np.mean(p_values < 0.01))

    # Selected t-values
    print(f"\n  Paired t-test at selected t values:")
    print(f"  {'t':>10} {'mean(ΔK)':>12} {'SEM':>10} {'t-stat':>8} "
          f"{'p-value':>10} {'sig?':>5}")
    print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*8} {'-'*10} {'-'*5}")
    for t_target in [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 3.0]:
        idx = np.argmin(np.abs(t_grid - t_target))
        dk_s = DK[:, idx]
        t_stat, pv = stats.ttest_1samp(dk_s, 0.0)
        sig = "**" if pv < 0.05 else ("*" if pv < 0.10 else "")
        print(f"  {t_grid[idx]:10.5f} {np.mean(dk_s):+12.4f} "
              f"{np.std(dk_s, ddof=1)/np.sqrt(M):10.4f} {t_stat:8.2f} "
              f"{pv:10.4f} {sig:>5}")

    # Eigenvalue shift
    eig_diff = np.array(eig_curved_means) - np.array(eig_flat_means)
    eig_t, eig_p = stats.ttest_1samp(eig_diff, 0.0)

    # UV exponent comparison
    from fnd1_ensemble_runner import fit_uv_exponent, determine_uv_window
    eig_flat_list = [compute_family_B_eigenvalues(
        build_bd_L(C, compute_interval_cardinalities(C), rho))
        for C in []]  # can't recompute easily, skip

    print(f"\n  Fraction of t-grid significant at 0.05: {frac_sig_005:.3f}")
    print(f"  Fraction significant at 0.01: {frac_sig_001:.3f}")
    print(f"  Max |relative ΔK|: {np.max(np.abs(rel_DK)):.6f}")
    print(f"  Mean |relative ΔK|: {np.mean(np.abs(rel_DK)):.6f}")
    print(f"  Eigenvalue mean shift: {np.mean(eig_diff):+.2f} "
          f"± {np.std(eig_diff, ddof=1)/np.sqrt(M):.2f}, "
          f"p = {eig_p:.4f}")

    return {
        "epsilon": eps,
        "N": N,
        "M": M,
        "frac_sig_005": frac_sig_005,
        "frac_sig_001": frac_sig_001,
        "max_rel_DK": float(np.max(np.abs(rel_DK))),
        "mean_rel_DK": float(np.mean(np.abs(rel_DK))),
        "eig_shift_p": float(eig_p),
    }


if __name__ == "__main__":
    t0_total = time.perf_counter()
    t0_pair = t0_total

    print("=" * 70)
    print("FND-1 ROUTE 1: GATE 5 — N=5000 FOLLOW-UP")
    print("=" * 70)
    print(f"N={N_POINTS}, M={M_PAIRS} pairs, eps = {EPSILON_VALUES}")
    print(f"Testing: does borderline signal at eps=-0.5 survive at larger N?")
    print()

    all_results = {}
    for eps in EPSILON_VALUES:
        print(f"\n{'='*60}")
        print(f"EPSILON = {eps:+.3f}")
        print(f"{'='*60}")
        t0_pair = time.perf_counter()

        result = run_paired_test(
            eps, N_POINTS, M_PAIRS, T_DIAMOND,
            MASTER_SEED + abs(hash(str(eps))) % 10000
        )
        all_results[str(eps)] = result

        wall = time.perf_counter() - t0_pair
        print(f"  Wall time: {wall:.1f}s ({wall/60:.1f} min)")

    # Verdict
    total = time.perf_counter() - t0_total
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    r_neg = all_results["-0.5"]
    r_pos = all_results["0.5"]

    print(f"\n  eps=-0.5: {r_neg['frac_sig_005']*100:.1f}% significant "
          f"(was 17.7% at N=1000)")
    print(f"  eps=+0.5: {r_pos['frac_sig_005']*100:.1f}% significant "
          f"(was 0.0% at N=1000)")

    both_detect = (r_neg["frac_sig_005"] > 0.10 and
                   r_pos["frac_sig_005"] > 0.10)
    neither = (r_neg["frac_sig_005"] < 0.10 and
               r_pos["frac_sig_005"] < 0.10)
    asymmetric = not both_detect and not neither

    if both_detect:
        verdict = "GENUINE CURVATURE SIGNAL — both eps show detection"
    elif neither:
        verdict = "NO SIGNAL — borderline at N=1000 was noise"
    else:
        verdict = "ASYMMETRIC — inconclusive (one eps only)"

    print(f"\n  FINAL VERDICT: {verdict}")
    print(f"  Total wall time: {total:.1f}s ({total/60:.1f} min)")

    # Save
    project_root = Path(__file__).resolve().parent.parent.parent
    out = project_root / "speculative" / "numerics" / "ensemble_results"
    out.mkdir(parents=True, exist_ok=True)

    save = {
        "parameters": {"N": N_POINTS, "M": M_PAIRS, "eps": EPSILON_VALUES},
        "results": all_results,
        "verdict": verdict,
        "wall_time_sec": total,
    }

    def _cl(o):
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None
        if isinstance(o, dict):
            return {k: _cl(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_cl(v) for v in o]
        return o

    with open(out / "gate5_n5000_followup.json", "w") as f:
        json.dump(_cl(save), f, indent=2)
    print(f"  Saved to: {out / 'gate5_n5000_followup.json'}")
