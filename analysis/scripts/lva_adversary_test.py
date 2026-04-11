#!/usr/bin/env python3
"""
LVA Adversary Test — E2 → E3 Promotion Gate
==========================================

Tests whether Link Valence Anisotropy (LVA) responds to genuine curvature
or merely to combinatorial perturbation of the causal matrix.

LVA DEFINITION:
  For each element x with future links {y_1,...,y_k} (k >= 2):
    k_+(x, y_i) = |J+(x) ∩ J+(y_i)|  (number of common future elements)
                 = (C @ C^T)[x, y_i]
  Then A(x) = Var(k_+) / Mean(k_+)^2.
  LVA = mean of A(x) over all interior elements with k >= 2.

NOTE: The original incubator definition used (C·C^T) proxy for J+∩J+.
Links in a causal set are an antichain (no two links from x are causally
related), so the "direct causal relation among links" version is trivially
zero. We use common-future counting instead, which is the standard
geometric probe.

ADVERSARY PROTOCOL:
  1. Generate flat causal set (N=2000, d=4) -> C_flat, compute LVA_flat
  2. Generate curved (pp-wave eps=5) from SAME points -> C_curved, compute LVA_curved
  3. Count n_diff = entries differing between C_flat and C_curved
  4. Create C_random: start from C_flat, randomly flip n_diff entries
     in the upper triangle (time-ordered, maintaining DAG structure)
  5. Compute LVA_random
  6. Repeat M=20 times
  7. Compare: |d_random| vs |d_curved|

KILL CRITERIA:
  - |d_random| > 0.3 * |d_curved| -> KILL (random flip reproduces signal)
  - d_random same sign as d_curved -> FLAG (sign is combinatorial)

Output: docs/analysis_runs/run_20260325_192720/lva_adversary.json
"""

import json
import time
import sys
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import wilcoxon, ttest_rel

# ── Parameters ──────────────────────────────────────────────────────
N = 2000
M = 20
T_DIAMOND = 1.0
EPS_PPWAVE = 5.0
MASTER_SEED = 77701
N_RANDOM_REPEATS = 5  # per trial: average over 5 random flips

RUN_DIR = Path(__file__).resolve().parents[2] / "docs" / "analysis_runs" / "run_20260325_192720"
RESULTS_FILE = RUN_DIR / "lva_adversary.json"


# ── Sprinkling ──────────────────────────────────────────────────────
def sprinkle(N_target: int, T: float, rng: np.random.Generator) -> np.ndarray:
    """Sprinkle N_target points into a 4D causal diamond |t|+r < T/2."""
    pts = []
    while len(pts) < N_target:
        batch = rng.uniform(-T / 2, T / 2, size=(N_target * 8, 4))
        r = np.sqrt(batch[:, 1]**2 + batch[:, 2]**2 + batch[:, 3]**2)
        inside = np.abs(batch[:, 0]) + r < T / 2
        pts.extend(batch[inside].tolist())
    pts = np.array(pts[:N_target])
    order = np.argsort(pts[:, 0])
    return pts[order]


# ── Causal matrices ─────────────────────────────────────────────────
def causal_flat(pts: np.ndarray) -> np.ndarray:
    """Flat Minkowski causal matrix. Upper triangular (time-ordered)."""
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        dt = pts[i + 1:, 0] - pts[i, 0]
        dr2 = np.sum((pts[i + 1:, 1:] - pts[i, 1:])**2, axis=1)
        causal = dt**2 > dr2
        C[i, i + 1:] = causal.astype(np.int8)
    return C


def causal_ppwave_quad(pts: np.ndarray, eps: float) -> np.ndarray:
    """PP-wave quadrupole: ds^2 = -du dv + (1 + eps*(x^2-y^2))(dx^2+dy^2)."""
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        dt = pts[i + 1:, 0] - pts[i, 0]
        dx = pts[i + 1:, 1:] - pts[i, 1:]
        dr2 = np.sum(dx**2, axis=1)
        xm = (pts[i + 1:, 1] + pts[i, 1]) / 2
        ym = (pts[i + 1:, 2] + pts[i, 2]) / 2
        dz = dx[:, 2]
        du = dt + dz
        f = xm**2 - ym**2
        interval = dt**2 - dr2 - eps * f * du**2 / 2
        causal = interval > 0
        C[i, i + 1:] = causal.astype(np.int8)
    return C


# ── Link matrix ─────────────────────────────────────────────────────
def build_link_matrix(C: np.ndarray) -> np.ndarray:
    """Link matrix L[i,j] = 1 iff i≺j and no k with i≺k≺j.
    Link = C[i,j]=1 AND (C^2)[i,j]=0."""
    C_f = C.astype(np.float32)
    C2 = C_f @ C_f
    L = ((C > 0) & (C2 == 0)).astype(np.int8)
    return L


# ── LVA computation ─────────────────────────────────────────────────
def compute_lva(C: np.ndarray, L: np.ndarray) -> tuple:
    """
    Link Valence Anisotropy using common-future counting.

    For each element x with future links {y_1,...,y_k} (k >= 2):
      k_+(x, y_i) = |J+(x) ∩ J+(y_i)| = (C @ C^T)[x, y_i]
    A(x) = Var(k_+) / Mean(k_+)^2
    LVA = mean of A(x) over qualifying elements.

    Returns: (lva_value, n_qualifying_elements, mean_fan_size)
    """
    C_f = C.astype(np.float32)
    CCT = C_f @ C_f.T  # CCT[i,j] = |J+(i) ∩ J+(j)|

    n = len(C)
    A_values = []
    fan_sizes = []

    for x in range(n):
        fl = np.where(L[x, :] == 1)[0]
        k = len(fl)
        if k < 2:
            continue

        fan_sizes.append(k)
        kplus = CCT[x, fl]  # array of |J+(x) ∩ J+(y_i)| for each link y_i

        mean_kp = kplus.mean()
        if mean_kp < 1e-12:
            continue
        var_kp = kplus.var()
        A_x = var_kp / (mean_kp ** 2)
        A_values.append(A_x)

    if len(A_values) == 0:
        return 0.0, 0, 0.0

    lva = float(np.mean(A_values))
    n_qual = len(A_values)
    mean_fan = float(np.mean(fan_sizes)) if fan_sizes else 0.0
    return lva, n_qual, mean_fan


# ── Random flip ─────────────────────────────────────────────────────
def random_flip(C_base: np.ndarray, n_diff: int, rng: np.random.Generator) -> np.ndarray:
    """
    Create a random perturbation of C_base by flipping n_diff entries
    in the upper triangle. Maintains upper-triangular (DAG) structure.

    Strategy: uniformly sample n_diff positions in the upper triangle
    and flip them (0->1 or 1->0).
    """
    n = len(C_base)
    C_rand = C_base.copy()

    # Get all upper-triangle indices
    rows, cols = np.triu_indices(n, k=1)
    total_upper = len(rows)

    # Sample n_diff positions uniformly without replacement
    n_flip = min(n_diff, total_upper)
    chosen = rng.choice(total_upper, size=n_flip, replace=False)

    for idx in chosen:
        i, j = rows[idx], cols[idx]
        C_rand[i, j] = 1 - C_rand[i, j]

    return C_rand


# ── Main experiment ─────────────────────────────────────────────────
def run_adversary_test():
    print("=" * 70)
    print("LVA ADVERSARY TEST — E2 -> E3 Promotion Gate")
    print("=" * 70)
    print(f"N={N}, M={M}, eps={EPS_PPWAVE}, T={T_DIAMOND}")
    print(f"Random repeats per trial: {N_RANDOM_REPEATS}")
    print(f"LVA definition: k_+(x,y_i) = |J+(x) cap J+(y_i)| = (C @ C^T)[x,y_i]")
    print()

    rng_master = np.random.default_rng(MASTER_SEED)

    lva_flat_arr = np.zeros(M)
    lva_curved_arr = np.zeros(M)
    lva_random_arr = np.zeros(M)
    n_diff_arr = np.zeros(M, dtype=int)
    n_qual_flat_arr = np.zeros(M, dtype=int)
    n_qual_curved_arr = np.zeros(M, dtype=int)

    trial_details = []
    t_total_start = time.time()

    for trial in range(M):
        t_start = time.time()
        seed = int(rng_master.integers(0, 2**31))
        rng = np.random.default_rng(seed)

        # 1. Sprinkle points
        pts = sprinkle(N, T_DIAMOND, rng)

        # 2. Flat causal matrix
        C_flat = causal_flat(pts)

        # 3. Curved causal matrix (same points, CRN)
        C_curved = causal_ppwave_quad(pts, EPS_PPWAVE)

        # 4. Count differences
        n_diff = int(np.sum(C_flat != C_curved))
        n_diff_arr[trial] = n_diff

        # 5. Build link matrices
        L_flat = build_link_matrix(C_flat)
        L_curved = build_link_matrix(C_curved)

        # 6. Compute LVA for flat and curved
        lva_flat, nq_flat, mf_flat = compute_lva(C_flat, L_flat)
        lva_curved, nq_curved, mf_curved = compute_lva(C_curved, L_curved)
        lva_flat_arr[trial] = lva_flat
        lva_curved_arr[trial] = lva_curved
        n_qual_flat_arr[trial] = nq_flat
        n_qual_curved_arr[trial] = nq_curved

        # 7. Random flips: average over N_RANDOM_REPEATS
        rng_rand = np.random.default_rng(seed + 1000000)
        lva_randoms = []
        for _ in range(N_RANDOM_REPEATS):
            C_rand = random_flip(C_flat, n_diff, rng_rand)
            L_rand = build_link_matrix(C_rand)
            lva_r, _, _ = compute_lva(C_rand, L_rand)
            lva_randoms.append(lva_r)
        lva_random_mean = float(np.mean(lva_randoms))
        lva_random_arr[trial] = lva_random_mean

        t_elapsed = time.time() - t_start

        detail = {
            "trial": trial,
            "seed": seed,
            "lva_flat": round(lva_flat, 8),
            "lva_curved": round(lva_curved, 8),
            "lva_random_mean": round(lva_random_mean, 8),
            "lva_random_all": [round(v, 8) for v in lva_randoms],
            "n_diff": n_diff,
            "n_diff_frac": round(n_diff / (N * (N - 1) / 2), 6),
            "n_links_flat": int(L_flat.sum()),
            "n_links_curved": int(L_curved.sum()),
            "n_qualifying_flat": nq_flat,
            "n_qualifying_curved": nq_curved,
            "mean_fan_flat": round(mf_flat, 2),
            "mean_fan_curved": round(mf_curved, 2),
            "delta_curved": round(lva_curved - lva_flat, 8),
            "delta_random": round(lva_random_mean - lva_flat, 8),
            "time_s": round(t_elapsed, 1),
        }
        trial_details.append(detail)

        d_c = lva_curved - lva_flat
        d_r = lva_random_mean - lva_flat
        print(f"  Trial {trial+1:2d}/{M}: "
              f"flat={lva_flat:.6f} "
              f"curved={lva_curved:.6f} "
              f"random={lva_random_mean:.6f} "
              f"d_c={d_c:+.6f} "
              f"d_r={d_r:+.6f} "
              f"ndiff={n_diff:5d} "
              f"({t_elapsed:.1f}s)")

    # ── Analysis ────────────────────────────────────────────────────
    t_total = time.time() - t_total_start
    print(f"\nTotal time: {t_total:.0f}s")
    print()

    d_curved = lva_curved_arr - lva_flat_arr
    d_random = lva_random_arr - lva_flat_arr

    mean_d_curved = float(d_curved.mean())
    mean_d_random = float(d_random.mean())
    std_d_curved = float(d_curved.std(ddof=1))
    std_d_random = float(d_random.std(ddof=1))

    # Cohen's d
    cohen_d_curved = mean_d_curved / std_d_curved if std_d_curved > 0 else 0.0
    cohen_d_random = mean_d_random / std_d_random if std_d_random > 0 else 0.0

    # Paired t-tests
    if std_d_curved > 0:
        t_curved, p_curved = ttest_rel(lva_curved_arr, lva_flat_arr)
    else:
        t_curved, p_curved = 0.0, 1.0

    if std_d_random > 0:
        t_random, p_random = ttest_rel(lva_random_arr, lva_flat_arr)
    else:
        t_random, p_random = 0.0, 1.0

    # Wilcoxon signed-rank
    try:
        _, p_wilc_curved = wilcoxon(d_curved)
    except ValueError:
        p_wilc_curved = 1.0
    try:
        _, p_wilc_random = wilcoxon(d_random)
    except ValueError:
        p_wilc_random = 1.0

    # ── Kill criteria ───────────────────────────────────────────────
    abs_d_curved = abs(mean_d_curved)
    abs_d_random = abs(mean_d_random)

    if abs_d_curved < 1e-10:
        ratio = float('inf')
    else:
        ratio = abs_d_random / abs_d_curved

    same_sign = (mean_d_curved > 0 and mean_d_random > 0) or \
                (mean_d_curved < 0 and mean_d_random < 0)

    kill = ratio > 0.3
    flag = same_sign

    # Also check: is the curved signal itself significant?
    curved_significant = (p_curved < 0.01)

    if abs_d_curved < 1e-10:
        verdict = "KILL_NO_SIGNAL"
        verdict_detail = "Curved signal is essentially zero — LVA does not detect curvature."
    elif not curved_significant:
        verdict = "KILL_WEAK_SIGNAL"
        verdict_detail = (f"Curved delta p={p_curved:.3e} > 0.01. "
                          f"LVA curvature response is not statistically significant.")
    elif kill and flag:
        verdict = "KILL"
        verdict_detail = (f"|d_random|/|d_curved| = {ratio:.3f} > 0.3 AND same sign. "
                          f"Random flips reproduce {ratio:.1%} of curved signal with same sign.")
    elif kill:
        verdict = "KILL"
        verdict_detail = (f"|d_random|/|d_curved| = {ratio:.3f} > 0.3. "
                          f"Random flips reproduce {ratio:.1%} of curved signal.")
    elif flag:
        verdict = "FLAG"
        verdict_detail = (f"|d_random|/|d_curved| = {ratio:.3f} < 0.3 (good), "
                          f"but same sign — possible combinatorial component.")
    else:
        verdict = "PASS"
        verdict_detail = (f"|d_random|/|d_curved| = {ratio:.3f} < 0.3, different sign. "
                          f"Curved signal is {1/ratio:.1f}x stronger than random noise.")

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  mean(LVA_flat)    = {lva_flat_arr.mean():.8f}")
    print(f"  mean(LVA_curved)  = {lva_curved_arr.mean():.8f}")
    print(f"  mean(LVA_random)  = {lva_random_arr.mean():.8f}")
    print()
    print(f"  mean(d_curved)  = {mean_d_curved:+.8f}  (std={std_d_curved:.8f})")
    print(f"  mean(d_random)  = {mean_d_random:+.8f}  (std={std_d_random:.8f})")
    print(f"  |d_random|/|d_curved| = {ratio:.6f}")
    print(f"  Same sign: {same_sign}")
    print()
    print(f"  Cohen's d (curved): {cohen_d_curved:.4f}")
    print(f"  Cohen's d (random): {cohen_d_random:.4f}")
    print(f"  t-test curved: t={t_curved:.4f}, p={p_curved:.2e}")
    print(f"  t-test random: t={t_random:.4f}, p={p_random:.2e}")
    print(f"  Wilcoxon curved: p={p_wilc_curved:.2e}")
    print(f"  Wilcoxon random: p={p_wilc_random:.2e}")
    print(f"  Curved significant (p<0.01): {curved_significant}")
    print()
    print(f"  Mean n_diff: {n_diff_arr.mean():.0f} ({n_diff_arr.mean()/(N*(N-1)/2)*100:.2f}% of upper triangle)")
    print(f"  Mean qualifying elements (flat): {n_qual_flat_arr.mean():.0f}")
    print(f"  Mean qualifying elements (curved): {n_qual_curved_arr.mean():.0f}")
    print()
    print(f"  VERDICT: {verdict}")
    print(f"  {verdict_detail}")
    print("=" * 70)

    # ── Save results ────────────────────────────────────────────────
    results = {
        "test": "LVA_adversary_E2_to_E3",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "lva_definition": "k_+(x,y_i) = |J+(x) cap J+(y_i)| = (C @ C^T)[x,y_i]; A(x) = Var/Mean^2; LVA = mean(A)",
        "parameters": {
            "N": N,
            "M": M,
            "d": 4,
            "T_diamond": T_DIAMOND,
            "eps_ppwave": EPS_PPWAVE,
            "n_random_repeats": N_RANDOM_REPEATS,
            "master_seed": MASTER_SEED,
        },
        "summary": {
            "mean_lva_flat": round(float(lva_flat_arr.mean()), 8),
            "std_lva_flat": round(float(lva_flat_arr.std(ddof=1)), 8),
            "mean_lva_curved": round(float(lva_curved_arr.mean()), 8),
            "std_lva_curved": round(float(lva_curved_arr.std(ddof=1)), 8),
            "mean_lva_random": round(float(lva_random_arr.mean()), 8),
            "std_lva_random": round(float(lva_random_arr.std(ddof=1)), 8),
            "mean_d_curved": round(mean_d_curved, 8),
            "mean_d_random": round(mean_d_random, 8),
            "std_d_curved": round(std_d_curved, 8),
            "std_d_random": round(std_d_random, 8),
            "abs_ratio": round(ratio, 6) if ratio != float('inf') else "inf",
            "same_sign": bool(same_sign),
            "cohen_d_curved": round(cohen_d_curved, 4),
            "cohen_d_random": round(cohen_d_random, 4),
            "ttest_curved_t": round(float(t_curved), 4),
            "ttest_curved_p": float(p_curved),
            "ttest_random_t": round(float(t_random), 4),
            "ttest_random_p": float(p_random),
            "wilcoxon_curved_p": float(p_wilc_curved),
            "wilcoxon_random_p": float(p_wilc_random),
            "curved_significant": bool(curved_significant),
            "mean_n_diff": round(float(n_diff_arr.mean()), 1),
            "mean_n_diff_frac": round(float(n_diff_arr.mean()) / (N * (N - 1) / 2), 6),
            "mean_qualifying_flat": round(float(n_qual_flat_arr.mean()), 1),
            "mean_qualifying_curved": round(float(n_qual_curved_arr.mean()), 1),
        },
        "kill_criteria": {
            "threshold": 0.3,
            "abs_d_random_over_abs_d_curved": round(ratio, 6) if ratio != float('inf') else "inf",
            "kill": bool(kill),
            "flag_same_sign": bool(flag),
        },
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "total_time_s": round(t_total, 1),
        "trials": trial_details,
    }

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    run_adversary_test()
