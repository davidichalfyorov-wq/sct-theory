"""
FND-1 Route 3: Verification of SVD curvature signal.

Three tests to resolve the ambiguity from the quick kill:

Test A: SPARSITY-MATCHED NULL — Random values in the EXACT SAME nonzero
        positions as L. Does SVD still discriminate?

Test B: MULTI-EPSILON REGRESSION — Spectral entropy vs curvature at
        eps = {-0.5, -0.25, 0, 0.25, 0.5, 0.75}. Monotonic?

Test C: REPRODUCIBILITY — Different seed for the eps=+0.5 entropy test.
        Does p = 0.0047 survive?
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
)
from fnd1_gate5_runner import sprinkle_curved, _sprinkle_flat
from fnd1_route3_quickkill import compute_sv_of_L, SV_ZERO

# ---------------------------------------------------------------------------
N_POINTS = 1000
T_DIAMOND = 1.0
MASTER_SEED = 42


def spectral_entropy(sv):
    """Compute spectral entropy H = -sum p_k ln p_k, p_k = sigma_k / sum."""
    s = np.sum(sv)
    if s <= 0 or len(sv) == 0:
        return np.nan
    p = sv / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def get_L_and_sv(N, T, rng, eps=0.0):
    """Sprinkle, build L, return L and its singular values."""
    V = T**2 / 2.0
    rho = N / V
    if eps == 0.0:
        pts, C = _sprinkle_flat(N, T, rng)
    else:
        pts, C = sprinkle_curved(N, eps, T, rng)
    n_mat = compute_interval_cardinalities(C)
    L = build_bd_L(C, n_mat, rho)
    sv = compute_sv_of_L(L)
    return L, sv


# ---------------------------------------------------------------------------
# Test A: Sparsity-Matched Null
# ---------------------------------------------------------------------------

def test_a_sparsity_null(N, M, T, seed):
    """
    Compare causal L vs random-values-in-same-positions null.
    This is the CORRECT null model that controls for sparsity.
    """
    print("\n" + "=" * 60)
    print("TEST A: SPARSITY-MATCHED NULL MODEL")
    print("=" * 60)

    V = T**2 / 2.0
    rho = N / V
    ss = np.random.SeedSequence(seed)
    seeds = ss.spawn(M)

    causal_ents = []
    matched_ents = []
    causal_mean_svs = []
    matched_mean_svs = []

    print(f"  Computing {M} causal + sparsity-matched pairs at N={N}...")
    t0 = time.perf_counter()
    for i in range(M):
        if (i + 1) % 25 == 0 or i == 0:
            el = time.perf_counter() - t0
            print(f"    {i+1}/{M} ({el:.0f}s)")

        rng = np.random.default_rng(seeds[i])
        L, sv_causal = get_L_and_sv(N, T, rng)

        # Sparsity-matched null: same nonzero pattern, random Gaussian values
        rng2 = np.random.default_rng(seeds[i].spawn(1)[0])
        mask = L != 0
        L_null = np.zeros_like(L)
        n_nonzero = int(np.sum(mask))
        L_null[mask] = rng2.standard_normal(n_nonzero)
        # Match Frobenius norm
        frob_L = np.linalg.norm(L, 'fro')
        frob_null = np.linalg.norm(L_null, 'fro')
        if frob_null > 0:
            L_null *= frob_L / frob_null
        sv_null = compute_sv_of_L(L_null)

        causal_ents.append(spectral_entropy(sv_causal))
        matched_ents.append(spectral_entropy(sv_null))
        causal_mean_svs.append(np.mean(sv_causal))
        matched_mean_svs.append(np.mean(sv_null))

    # Permutation-matched null: same VALUES, random positions within lower tri
    perm_ents = []
    print(f"  Computing {M} permutation-matched nulls...")
    ss2 = np.random.SeedSequence(seed + 5555)
    seeds2 = ss2.spawn(M)
    for i in range(M):
        rng = np.random.default_rng(seeds[i])
        L, sv_causal_2 = get_L_and_sv(N, T, rng)

        # Extract nonzero values, put them in random lower-tri positions
        rng3 = np.random.default_rng(seeds2[i])
        values = L[L != 0].copy()
        rng3.shuffle(values)
        L_perm = np.zeros_like(L)
        mask = L != 0
        L_perm[mask] = values
        sv_perm = compute_sv_of_L(L_perm)
        perm_ents.append(spectral_entropy(sv_perm))

    # Statistics
    causal_ents = np.array(causal_ents)
    matched_ents = np.array(matched_ents)
    perm_ents = np.array(perm_ents)

    # Causal vs sparsity-matched (random values, same positions)
    t_sm, p_sm = stats.ttest_ind(causal_ents, matched_ents)
    ks_sm, ksp_sm = stats.ks_2samp(causal_ents, matched_ents)

    # Causal vs permutation-matched (same values, shuffled within same positions)
    t_pm, p_pm = stats.ttest_ind(causal_ents, perm_ents)

    # Causal vs sparsity-matched: mean SV
    t_mv, p_mv = stats.ttest_ind(causal_mean_svs, matched_mean_svs)

    print(f"\n  Results:")
    print(f"    Causal entropy:   {np.mean(causal_ents):.6f} ± {np.std(causal_ents):.6f}")
    print(f"    Sparsity-null:    {np.mean(matched_ents):.6f} ± {np.std(matched_ents):.6f}")
    print(f"    Permutation-null: {np.mean(perm_ents):.6f} ± {np.std(perm_ents):.6f}")
    print(f"\n    Causal vs Sparsity-null (entropy): t={t_sm:.2f}, p={p_sm:.2e}")
    print(f"    Causal vs Sparsity-null (KS):      D={ks_sm:.4f}, p={ksp_sm:.2e}")
    print(f"    Causal vs Permutation-null:         t={t_pm:.2f}, p={p_pm:.2e}")
    print(f"    Causal vs Sparsity-null (mean SV):  t={t_mv:.2f}, p={p_mv:.2e}")

    # What's actually being detected?
    discriminates_sparsity = p_sm < 0.01
    discriminates_perm = p_pm < 0.01
    print(f"\n    Discriminates from sparsity-null: {discriminates_sparsity}")
    print(f"    Discriminates from permutation-null: {discriminates_perm}")

    if discriminates_perm:
        verdict = "GENUINE STRUCTURAL SIGNAL (survives permutation test)"
    elif discriminates_sparsity:
        verdict = "VALUE-DEPENDENT SIGNAL (values matter, not just positions)"
    else:
        verdict = "NO SIGNAL beyond sparsity pattern"

    print(f"    VERDICT: {verdict}")
    return {
        "p_sparsity_entropy": float(p_sm),
        "p_sparsity_ks": float(ksp_sm),
        "p_permutation": float(p_pm),
        "p_sparsity_meansv": float(p_mv),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Test B: Multi-Epsilon Regression
# ---------------------------------------------------------------------------

def test_b_multi_epsilon(N, M, T, seed):
    """
    Compute spectral entropy at multiple curvature values.
    Test: is entropy proportional to curvature?
    """
    print("\n" + "=" * 60)
    print("TEST B: MULTI-EPSILON REGRESSION (entropy vs curvature)")
    print("=" * 60)

    eps_values = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
    V = T**2 / 2.0
    rho = N / V

    ss = np.random.SeedSequence(seed + 3333)

    results_by_eps = {}
    for eps in eps_values:
        eps_seed = ss.spawn(1)[0]
        pair_seeds = eps_seed.spawn(M)

        ents = []
        print(f"\n  eps={eps:+.3f}: computing {M} sprinklings...")
        t0 = time.perf_counter()
        for i in range(M):
            if (i + 1) % 25 == 0 or i == 0:
                print(f"    {i+1}/{M} ({time.perf_counter()-t0:.0f}s)")
            rng = np.random.default_rng(pair_seeds[i])
            _, sv = get_L_and_sv(N, T, rng, eps=eps)
            ents.append(spectral_entropy(sv))

        ents = np.array(ents)
        results_by_eps[eps] = {
            "mean": float(np.mean(ents)),
            "std": float(np.std(ents, ddof=1)),
            "sem": float(np.std(ents, ddof=1) / np.sqrt(M)),
        }
        print(f"    Entropy: {np.mean(ents):.6f} ± {np.std(ents, ddof=1)/np.sqrt(M):.6f}")

    # Regression: entropy vs epsilon
    eps_arr = np.array(eps_values)
    ent_means = np.array([results_by_eps[e]["mean"] for e in eps_values])
    ent_sems = np.array([results_by_eps[e]["sem"] for e in eps_values])

    # OLS
    from scipy.stats import linregress, pearsonr
    lr = linregress(eps_arr, ent_means)
    r_pearson, p_pearson = pearsonr(eps_arr, ent_means)

    # Monotonicity check
    diffs = np.diff(ent_means)
    monotonic_increasing = all(d > 0 for d in diffs)
    monotonic_decreasing = all(d < 0 for d in diffs)
    monotonic = monotonic_increasing or monotonic_decreasing

    print(f"\n  Summary:")
    print(f"    {'eps':>6} {'entropy':>12} {'SEM':>10}")
    for eps in eps_values:
        r = results_by_eps[eps]
        print(f"    {eps:+6.3f} {r['mean']:12.6f} {r['sem']:10.6f}")

    print(f"\n    Linear fit: entropy = {lr.slope:.6f} * eps + {lr.intercept:.6f}")
    print(f"    Pearson r = {r_pearson:.4f}, p = {p_pearson:.4f}")
    print(f"    R² = {lr.rvalue**2:.4f}")
    print(f"    Monotonic: {monotonic}")

    if abs(r_pearson) > 0.8 and p_pearson < 0.05 and monotonic:
        verdict = "STRONG CORRELATION — entropy tracks curvature"
    elif abs(r_pearson) > 0.6 and p_pearson < 0.10:
        verdict = "MODERATE CORRELATION"
    elif p_pearson < 0.10:
        verdict = "WEAK TREND"
    else:
        verdict = "NO CORRELATION"

    print(f"    VERDICT: {verdict}")
    return {
        "results_by_eps": {str(e): results_by_eps[e] for e in eps_values},
        "pearson_r": float(r_pearson),
        "pearson_p": float(p_pearson),
        "slope": float(lr.slope),
        "r_squared": float(lr.rvalue**2),
        "monotonic": bool(monotonic),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Test C: Reproducibility
# ---------------------------------------------------------------------------

def test_c_reproducibility(N, M, T, eps, seed):
    """
    Re-run the entropy shift test with a completely different seed.
    """
    print("\n" + "=" * 60)
    print(f"TEST C: REPRODUCIBILITY (eps={eps:+.3f}, seed={seed})")
    print("=" * 60)

    V = T**2 / 2.0
    rho = N / V
    ss = np.random.SeedSequence(seed)
    pair_seeds = ss.spawn(M)

    ent_flat = []
    ent_curved = []

    print(f"  Computing {M} matched pairs...")
    t0 = time.perf_counter()
    for i in range(M):
        if (i + 1) % 25 == 0 or i == 0:
            el = time.perf_counter() - t0
            print(f"    {i+1}/{M} ({el:.0f}s)")

        ch = pair_seeds[i].spawn(2)
        _, sv_f = get_L_and_sv(N, T, np.random.default_rng(ch[0]), eps=0.0)
        _, sv_c = get_L_and_sv(N, T, np.random.default_rng(ch[1]), eps=eps)

        ent_flat.append(spectral_entropy(sv_f))
        ent_curved.append(spectral_entropy(sv_c))

    ent_flat = np.array(ent_flat)
    ent_curved = np.array(ent_curved)
    diff = ent_curved - ent_flat

    t_stat, p_val = stats.ttest_1samp(diff, 0.0)
    mean_diff = float(np.mean(diff))
    sem_diff = float(np.std(diff, ddof=1) / np.sqrt(M))

    print(f"\n  Results:")
    print(f"    Flat entropy:   {np.mean(ent_flat):.6f}")
    print(f"    Curved entropy: {np.mean(ent_curved):.6f}")
    print(f"    Shift: {mean_diff:+.6f} ± {sem_diff:.6f}")
    print(f"    t = {t_stat:.2f}, p = {p_val:.4f}")

    reproduced = p_val < 0.05
    print(f"    VERDICT: {'REPRODUCED' if reproduced else 'NOT REPRODUCED'}")
    return {
        "mean_diff": mean_diff,
        "sem_diff": sem_diff,
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "reproduced": reproduced,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_total = time.perf_counter()

    print("=" * 70)
    print("FND-1 ROUTE 3: SIGNAL VERIFICATION")
    print("=" * 70)
    print(f"N={N_POINTS}, T={T_DIAMOND}")
    print()

    M_null = 100     # for null model test
    M_eps = 80       # per epsilon value
    M_repro = 100    # for reproducibility

    ra = test_a_sparsity_null(N_POINTS, M_null, T_DIAMOND, MASTER_SEED)
    rb = test_b_multi_epsilon(N_POINTS, M_eps, T_DIAMOND, MASTER_SEED)
    rc1 = test_c_reproducibility(N_POINTS, M_repro, T_DIAMOND, 0.5,
                                  seed=12345)
    rc2 = test_c_reproducibility(N_POINTS, M_repro, T_DIAMOND, 0.5,
                                  seed=99999)
    rc3 = test_c_reproducibility(N_POINTS, M_repro, T_DIAMOND, -0.5,
                                  seed=77777)

    total = time.perf_counter() - t_total

    print(f"\n\n{'='*70}")
    print("OVERALL VERDICT")
    print(f"{'='*70}")
    print(f"\n  Test A (Sparsity-matched null): {ra['verdict']}")
    print(f"  Test B (Multi-epsilon):          {rb['verdict']}")
    print(f"    Pearson r = {rb['pearson_r']:.4f}, p = {rb['pearson_p']:.4f}")
    print(f"    Monotonic: {rb['monotonic']}")
    print(f"  Test C (Reproducibility):")
    print(f"    seed=12345, eps=+0.5: p={rc1['p_value']:.4f} "
          f"({'REPRO' if rc1['reproduced'] else 'NO'})")
    print(f"    seed=99999, eps=+0.5: p={rc2['p_value']:.4f} "
          f"({'REPRO' if rc2['reproduced'] else 'NO'})")
    print(f"    seed=77777, eps=-0.5: p={rc3['p_value']:.4f} "
          f"({'REPRO' if rc3['reproduced'] else 'NO'})")

    n_repro = sum(1 for r in [rc1, rc2, rc3] if r['reproduced'])
    b_pass = rb['pearson_p'] < 0.05 and rb['monotonic']

    if n_repro >= 2 and b_pass:
        final = "SIGNAL CONFIRMED — reproducible, monotonic, curvature-correlated"
    elif n_repro >= 2:
        final = "SIGNAL PARTIALLY CONFIRMED — reproducible but not monotonic"
    elif b_pass:
        final = "SIGNAL PARTIALLY CONFIRMED — monotonic but not all seeds reproduce"
    elif n_repro >= 1 or rb['pearson_p'] < 0.10:
        final = "WEAK SIGNAL — marginal evidence, needs larger N or M"
    else:
        final = "NO SIGNAL — Route 3 SVD entropy approach does not detect curvature"

    print(f"\n  FINAL: {final}")
    print(f"  Wall time: {total:.0f}s ({total/60:.1f} min)")

    # Save
    project_root = Path(__file__).resolve().parent.parent.parent
    out_path = (project_root / "speculative" / "numerics" /
                "ensemble_results" / "route3_verification.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save = {
        "parameters": {"N": N_POINTS, "T": T_DIAMOND},
        "test_a": ra, "test_b": rb,
        "test_c": {"seed_12345": rc1, "seed_99999": rc2, "seed_77777_neg": rc3},
        "n_reproduced": n_repro,
        "multi_eps_pass": b_pass,
        "verdict": final,
        "wall_time_sec": total,
    }

    def _cl(o):
        if isinstance(o, (float, np.floating)):
            v = float(o)
            return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(o, (bool, np.bool_)):
            return bool(o)
        if isinstance(o, dict):
            return {k: _cl(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_cl(v) for v in o]
        return o

    with open(out_path, "w") as f:
        json.dump(_cl(save), f, indent=2)
    print(f"  Saved: {out_path}")
