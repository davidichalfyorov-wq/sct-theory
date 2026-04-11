"""
FND-1 Route 3: Lorentzian/Krein — Quick Kill Tests.

Three fast tests to determine if Route 3 is worth pursuing:

Test 1: SVD DISCRIMINATION — Do singular values of the retarded BD operator L
        distinguish causal sets from random lower-triangular matrices?

Test 2: SVD CURVATURE — Do singular values of L change between flat and curved
        spacetime? (Matched-pairs test, same framework as Route 1 Gate 5.)

Test 3: DANG-WROCHNA ZETA — Does f(alpha) = sum_k sigma_k^{-2*alpha} show
        non-trivial structure that differs between flat and curved?

If ALL three fail → Route 3 is likely CLOSED.
If ANY passes → proceed to full Route 3 investigation.

Key principle: use L AS-IS (retarded, non-Hermitian). Do NOT symmetrize.
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
    compute_heat_trace,
    ZERO_THRESHOLD,
)
from fnd1_gate5_runner import sprinkle_curved, _sprinkle_flat

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_POINTS = 1000
M_ENSEMBLE = 100       # ensemble size per test
T_DIAMOND = 1.0
MASTER_SEED = 42
EPS_CURVED = 0.5       # curvature parameter for Tests 2-3

# Singular value heat trace grid
N_T = 300
T_MIN = 1e-5
T_MAX = 10.0

# Dang-Wrochna zeta: alpha grid
ALPHA_MIN = 0.1
ALPHA_MAX = 3.0
N_ALPHA = 50

# Zero threshold for singular values
SV_ZERO = 1e-10


# ---------------------------------------------------------------------------
# Core: compute singular values of L
# ---------------------------------------------------------------------------

def compute_sv_of_L(L: np.ndarray) -> np.ndarray:
    """
    Compute singular values of the strictly lower-triangular BD operator L.
    Returns sorted array of positive singular values (excluding near-zeros).
    """
    sv = np.linalg.svd(L, compute_uv=False)
    sv = sv[sv > SV_ZERO]
    return np.sort(sv)[::-1]  # descending order


def generate_null_lower_triangular(N: int, target_frob: float,
                                   rng: np.random.Generator) -> np.ndarray:
    """
    Generate a random strictly lower-triangular matrix with matched
    Frobenius norm. This is the null model for Test 1.
    """
    # Random lower-triangular with Gaussian entries
    M = np.zeros((N, N))
    for i in range(1, N):
        M[i, :i] = rng.standard_normal(i)

    # Rescale to match Frobenius norm
    current_frob = np.linalg.norm(M, 'fro')
    if current_frob > 0:
        M *= target_frob / current_frob

    return M


# ---------------------------------------------------------------------------
# Test 1: SVD Discrimination (causal vs random)
# ---------------------------------------------------------------------------

def test1_svd_discrimination(N, M, T, seed):
    """
    Do singular values of L distinguish causal sets from random
    lower-triangular matrices?
    """
    print("\n" + "=" * 60)
    print("TEST 1: SVD DISCRIMINATION (causal vs random)")
    print("=" * 60)

    V = T**2 / 2.0
    rho = N / V
    ss = np.random.SeedSequence(seed)
    seeds = ss.spawn(M + M)  # M for causal, M for null

    # Compute singular values for causal sprinklings
    sv_causal_all = []
    frob_all = []
    print(f"  Computing {M} causal SVDs at N={N}...")
    for i in range(M):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"    Sprinkling {i+1}/{M}...")
        rng = np.random.default_rng(seeds[i])
        pts, C = _sprinkle_flat(N, T, rng)
        n_mat = compute_interval_cardinalities(C)
        L = build_bd_L(C, n_mat, rho)
        sv = compute_sv_of_L(L)
        sv_causal_all.append(sv)
        frob_all.append(np.linalg.norm(L, 'fro'))

    mean_frob = float(np.mean(frob_all))
    mean_n_sv = float(np.mean([len(sv) for sv in sv_causal_all]))

    # Compute singular values for null (random lower-triangular)
    sv_null_all = []
    print(f"  Computing {M} null SVDs...")
    for i in range(M):
        rng = np.random.default_rng(seeds[M + i])
        L_null = generate_null_lower_triangular(N, mean_frob, rng)
        sv = compute_sv_of_L(L_null)
        sv_null_all.append(sv)

    # Compare distributions
    # Method 1: Compare mean singular value profiles
    # Pad to same length
    max_len = max(max(len(sv) for sv in sv_causal_all),
                  max(len(sv) for sv in sv_null_all))

    def pad_sv(sv_list, length):
        result = np.zeros((len(sv_list), length))
        for i, sv in enumerate(sv_list):
            result[i, :len(sv)] = sv
        return result

    sv_causal_mat = pad_sv(sv_causal_all, max_len)
    sv_null_mat = pad_sv(sv_null_all, max_len)

    sv_causal_mean = np.mean(sv_causal_mat, axis=0)
    sv_null_mean = np.mean(sv_null_mat, axis=0)

    # Method 2: KS test on pooled singular values
    sv_causal_pooled = np.concatenate(sv_causal_all)
    sv_null_pooled = np.concatenate(sv_null_all)
    ks_stat, ks_p = stats.ks_2samp(sv_causal_pooled, sv_null_pooled)

    # Method 3: Compare summary statistics
    causal_means = [np.mean(sv) for sv in sv_causal_all]
    null_means = [np.mean(sv) for sv in sv_null_all]
    t_stat_mean, p_mean = stats.ttest_ind(causal_means, null_means)

    causal_maxs = [sv[0] for sv in sv_causal_all]  # largest SV
    null_maxs = [sv[0] for sv in sv_null_all]
    t_stat_max, p_max = stats.ttest_ind(causal_maxs, null_maxs)

    # Method 4: Compare spectral entropy
    def spectral_entropy(sv):
        p = sv / np.sum(sv)
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    causal_ent = [spectral_entropy(sv) for sv in sv_causal_all]
    null_ent = [spectral_entropy(sv) for sv in sv_null_all]
    t_stat_ent, p_ent = stats.ttest_ind(causal_ent, null_ent)

    print(f"\n  Results:")
    print(f"    Mean Frobenius norm: {mean_frob:.2f}")
    print(f"    Mean # singular values: {mean_n_sv:.1f}")
    print(f"    KS test (pooled SVs): stat={ks_stat:.4f}, p={ks_p:.2e}")
    print(f"    Mean SV: causal={np.mean(causal_means):.2f}, "
          f"null={np.mean(null_means):.2f}, p={p_mean:.2e}")
    print(f"    Max SV: causal={np.mean(causal_maxs):.2f}, "
          f"null={np.mean(null_maxs):.2f}, p={p_max:.2e}")
    print(f"    Spectral entropy: causal={np.mean(causal_ent):.4f}, "
          f"null={np.mean(null_ent):.4f}, p={p_ent:.2e}")

    discriminates = ks_p < 0.01 or p_mean < 0.01 or p_ent < 0.01
    print(f"\n  VERDICT: {'DISCRIMINATES' if discriminates else 'DOES NOT DISCRIMINATE'}")

    return {
        "ks_stat": float(ks_stat), "ks_p": float(ks_p),
        "mean_sv_p": float(p_mean), "max_sv_p": float(p_max),
        "entropy_p": float(p_ent),
        "causal_mean_sv": float(np.mean(causal_means)),
        "null_mean_sv": float(np.mean(null_means)),
        "causal_entropy": float(np.mean(causal_ent)),
        "null_entropy": float(np.mean(null_ent)),
        "discriminates": discriminates,
    }


# ---------------------------------------------------------------------------
# Test 2: SVD Curvature (flat vs curved)
# ---------------------------------------------------------------------------

def test2_svd_curvature(N, M, T, eps, seed):
    """
    Matched-pairs test: do singular values of L change between
    flat and curved spacetime?
    """
    print("\n" + "=" * 60)
    print(f"TEST 2: SVD CURVATURE (eps={eps:+.3f})")
    print("=" * 60)

    V = T**2 / 2.0
    rho = N / V
    ss = np.random.SeedSequence(seed)
    pair_seeds = ss.spawn(M)

    sv_flat_means = []
    sv_curved_means = []
    sv_flat_maxs = []
    sv_curved_maxs = []
    sv_flat_ents = []
    sv_curved_ents = []

    # Also compute SV heat traces for a more sensitive test
    t_grid = np.logspace(np.log10(T_MIN), np.log10(T_MAX), N_T)
    K_flat_all = np.zeros((M, N_T))
    K_curved_all = np.zeros((M, N_T))

    print(f"  Computing {M} matched pairs...")
    t0 = time.perf_counter()
    for i in range(M):
        if (i + 1) % 25 == 0 or i == 0:
            el = time.perf_counter() - t0
            eta = el / (i + 1) * M - el if i > 0 else 0
            print(f"    Pair {i+1}/{M} ({el:.0f}s, ~{eta:.0f}s left)")

        ch = pair_seeds[i].spawn(2)

        # Flat
        pts_f, C_f = _sprinkle_flat(N, T, np.random.default_rng(ch[0]))
        L_f = build_bd_L(C_f, compute_interval_cardinalities(C_f), rho)
        sv_f = compute_sv_of_L(L_f)

        # Curved
        pts_c, C_c = sprinkle_curved(N, eps, T, np.random.default_rng(ch[1]))
        L_c = build_bd_L(C_c, compute_interval_cardinalities(C_c), rho)
        sv_c = compute_sv_of_L(L_c)

        sv_flat_means.append(np.mean(sv_f))
        sv_curved_means.append(np.mean(sv_c))
        sv_flat_maxs.append(sv_f[0] if len(sv_f) > 0 else 0)
        sv_curved_maxs.append(sv_c[0] if len(sv_c) > 0 else 0)

        def _ent(sv):
            p = sv / np.sum(sv) if np.sum(sv) > 0 else sv
            p = p[p > 0]
            return -np.sum(p * np.log(p))
        sv_flat_ents.append(_ent(sv_f))
        sv_curved_ents.append(_ent(sv_c))

        # SV heat trace: K(t) = sum exp(-t * sigma_k^2)
        K_flat_all[i] = np.sum(np.exp(-t_grid[:, None] * sv_f[None, :]**2),
                               axis=1)
        K_curved_all[i] = np.sum(np.exp(-t_grid[:, None] * sv_c[None, :]**2),
                                 axis=1)

    # Paired tests on summary statistics
    diff_means = np.array(sv_curved_means) - np.array(sv_flat_means)
    t_mean, p_mean = stats.ttest_1samp(diff_means, 0.0)

    diff_maxs = np.array(sv_curved_maxs) - np.array(sv_flat_maxs)
    t_max, p_max = stats.ttest_1samp(diff_maxs, 0.0)

    diff_ents = np.array(sv_curved_ents) - np.array(sv_flat_ents)
    t_ent, p_ent = stats.ttest_1samp(diff_ents, 0.0)

    # Paired test on SV heat trace
    DK = K_curved_all - K_flat_all
    p_heat = np.ones(N_T)
    for j in range(N_T):
        dk_j = DK[:, j]
        if np.std(dk_j, ddof=1) > 0:
            _, p_heat[j] = stats.ttest_1samp(dk_j, 0.0)

    frac_sig_005 = float(np.mean(p_heat < 0.05))
    frac_sig_001 = float(np.mean(p_heat < 0.01))

    print(f"\n  Results:")
    print(f"    Mean SV shift: {np.mean(diff_means):+.4f} ± "
          f"{np.std(diff_means, ddof=1)/np.sqrt(M):.4f}, p={p_mean:.4f}")
    print(f"    Max SV shift: {np.mean(diff_maxs):+.2f} ± "
          f"{np.std(diff_maxs, ddof=1)/np.sqrt(M):.2f}, p={p_max:.4f}")
    print(f"    Entropy shift: {np.mean(diff_ents):+.6f} ± "
          f"{np.std(diff_ents, ddof=1)/np.sqrt(M):.6f}, p={p_ent:.4f}")
    print(f"    SV heat trace: {frac_sig_005*100:.1f}% of t-grid sig at 0.05, "
          f"{frac_sig_001*100:.1f}% at 0.01")

    curvature_detected = (p_mean < 0.05 or p_ent < 0.05 or
                          frac_sig_005 > 0.10)
    print(f"\n  VERDICT: {'CURVATURE DETECTED' if curvature_detected else 'NO CURVATURE SIGNAL'}")

    return {
        "mean_sv_shift_p": float(p_mean),
        "max_sv_shift_p": float(p_max),
        "entropy_shift_p": float(p_ent),
        "frac_heat_005": frac_sig_005,
        "frac_heat_001": frac_sig_001,
        "curvature_detected": curvature_detected,
    }


# ---------------------------------------------------------------------------
# Test 3: Dang-Wrochna Zeta
# ---------------------------------------------------------------------------

def test3_dw_zeta(N, M, T, eps, seed):
    """
    Compute f(alpha) = sum_k sigma_k^{-2*alpha} for causal set SVs.
    Check if f(alpha) has non-trivial structure and is curvature-sensitive.
    """
    print("\n" + "=" * 60)
    print(f"TEST 3: DANG-WROCHNA ZETA (eps={eps:+.3f} vs flat)")
    print("=" * 60)

    V = T**2 / 2.0
    rho = N / V
    alpha_grid = np.linspace(ALPHA_MIN, ALPHA_MAX, N_ALPHA)

    ss = np.random.SeedSequence(seed + 7777)
    pair_seeds = ss.spawn(M)

    zeta_flat_all = np.zeros((M, N_ALPHA))
    zeta_curved_all = np.zeros((M, N_ALPHA))

    print(f"  Computing {M} paired DW-zeta values...")
    t0 = time.perf_counter()
    for i in range(M):
        if (i + 1) % 25 == 0 or i == 0:
            el = time.perf_counter() - t0
            eta = el / (i + 1) * M - el if i > 0 else 0
            print(f"    Pair {i+1}/{M} ({el:.0f}s, ~{eta:.0f}s left)")

        ch = pair_seeds[i].spawn(2)

        # Flat
        pts_f, C_f = _sprinkle_flat(N, T, np.random.default_rng(ch[0]))
        L_f = build_bd_L(C_f, compute_interval_cardinalities(C_f), rho)
        sv_f = compute_sv_of_L(L_f)
        sv_f = sv_f[sv_f > SV_ZERO]

        # Curved
        pts_c, C_c = sprinkle_curved(N, eps, T, np.random.default_rng(ch[1]))
        L_c = build_bd_L(C_c, compute_interval_cardinalities(C_c), rho)
        sv_c = compute_sv_of_L(L_c)
        sv_c = sv_c[sv_c > SV_ZERO]

        # Compute zeta: f(alpha) = sum_k sigma_k^{-2*alpha}
        for j, alpha in enumerate(alpha_grid):
            if len(sv_f) > 0:
                zeta_flat_all[i, j] = np.sum(sv_f**(-2.0 * alpha))
            if len(sv_c) > 0:
                zeta_curved_all[i, j] = np.sum(sv_c**(-2.0 * alpha))

    # Ensemble averages
    zeta_flat_ens = np.mean(zeta_flat_all, axis=0)
    zeta_curved_ens = np.mean(zeta_curved_all, axis=0)
    zeta_ratio = np.where(zeta_flat_ens > 0,
                          zeta_curved_ens / zeta_flat_ens, 1.0)

    # Paired t-test at each alpha
    Dzeta = zeta_curved_all - zeta_flat_all
    p_zeta = np.ones(N_ALPHA)
    for j in range(N_ALPHA):
        dz_j = Dzeta[:, j]
        if np.std(dz_j, ddof=1) > 0:
            _, p_zeta[j] = stats.ttest_1samp(dz_j, 0.0)

    frac_sig_005 = float(np.mean(p_zeta < 0.05))
    frac_sig_001 = float(np.mean(p_zeta < 0.01))

    # Check if zeta ratio varies with alpha (non-trivial structure)
    ratio_std = float(np.std(zeta_ratio))
    ratio_range = float(np.max(zeta_ratio) - np.min(zeta_ratio))

    # Log-log slope of zeta_flat: f(alpha) ~ alpha^slope?
    valid = zeta_flat_ens > 0
    if np.sum(valid) > 5:
        from scipy.stats import linregress
        lr = linregress(np.log(alpha_grid[valid]), np.log(zeta_flat_ens[valid]))
        zeta_slope = lr.slope
        zeta_r2 = lr.rvalue**2
    else:
        zeta_slope = np.nan
        zeta_r2 = np.nan

    print(f"\n  Results:")
    print(f"    Zeta log-log slope: {zeta_slope:.4f} (R²={zeta_r2:.4f})")
    print(f"    Curved/flat ratio: mean={np.mean(zeta_ratio):.6f}, "
          f"std={ratio_std:.6f}, range={ratio_range:.6f}")
    print(f"    Fraction of alpha-grid significant at 0.05: {frac_sig_005*100:.1f}%")
    print(f"    Fraction significant at 0.01: {frac_sig_001*100:.1f}%")

    # Print selected alpha values
    print(f"\n    {'alpha':>6} {'zeta_flat':>12} {'zeta_curved':>12} "
          f"{'ratio':>8} {'p-val':>8} {'sig':>4}")
    for idx in [0, N_ALPHA//4, N_ALPHA//2, 3*N_ALPHA//4, N_ALPHA-1]:
        a = alpha_grid[idx]
        sig = "**" if p_zeta[idx] < 0.05 else ("*" if p_zeta[idx] < 0.1 else "")
        print(f"    {a:6.2f} {zeta_flat_ens[idx]:12.4f} "
              f"{zeta_curved_ens[idx]:12.4f} "
              f"{zeta_ratio[idx]:8.6f} {p_zeta[idx]:8.4f} {sig:>4}")

    has_structure = ratio_range > 0.01 and frac_sig_005 > 0.10
    print(f"\n  VERDICT: {'STRUCTURE DETECTED' if has_structure else 'NO STRUCTURE'}")

    return {
        "zeta_slope": float(zeta_slope),
        "zeta_r2": float(zeta_r2),
        "ratio_mean": float(np.mean(zeta_ratio)),
        "ratio_std": ratio_std,
        "ratio_range": ratio_range,
        "frac_sig_005": frac_sig_005,
        "frac_sig_001": frac_sig_001,
        "has_structure": has_structure,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t_total = time.perf_counter()

    print("=" * 70)
    print("FND-1 ROUTE 3: LORENTZIAN/KREIN — QUICK KILL TESTS")
    print("=" * 70)
    print(f"N={N_POINTS}, M={M_ENSEMBLE}, eps={EPS_CURVED}")
    print(f"Using RETARDED BD operator L (not symmetrized)")
    print(f"Observable: singular values of L")
    print()

    r1 = test1_svd_discrimination(N_POINTS, M_ENSEMBLE, T_DIAMOND, MASTER_SEED)
    r2 = test2_svd_curvature(N_POINTS, M_ENSEMBLE // 2, T_DIAMOND,
                              EPS_CURVED, MASTER_SEED + 1000)
    r3 = test3_dw_zeta(N_POINTS, M_ENSEMBLE // 2, T_DIAMOND,
                        EPS_CURVED, MASTER_SEED + 2000)

    # Overall
    total = time.perf_counter() - t_total
    print(f"\n\n{'='*70}")
    print("OVERALL VERDICT")
    print(f"{'='*70}")
    print(f"\n  Test 1 (SVD discrimination): "
          f"{'PASS' if r1['discriminates'] else 'FAIL'}")
    print(f"  Test 2 (SVD curvature):      "
          f"{'PASS' if r2['curvature_detected'] else 'FAIL'}")
    print(f"  Test 3 (DW-zeta structure):   "
          f"{'PASS' if r3['has_structure'] else 'FAIL'}")

    any_pass = r1['discriminates'] or r2['curvature_detected'] or r3['has_structure']
    if any_pass:
        verdict = "PROCEED — at least one test shows signal"
    else:
        verdict = "ROUTE 3 LIKELY CLOSED — no signal in any test"

    print(f"\n  FINAL: {verdict}")
    print(f"  Wall time: {total:.0f}s ({total/60:.1f} min)")

    # Save
    project_root = Path(__file__).resolve().parent.parent.parent
    out_path = (project_root / "speculative" / "numerics" /
                "ensemble_results" / "route3_quickkill.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save = {
        "parameters": {"N": N_POINTS, "M": M_ENSEMBLE, "eps": EPS_CURVED},
        "test1": r1, "test2": r2, "test3": r3,
        "verdict": verdict,
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
