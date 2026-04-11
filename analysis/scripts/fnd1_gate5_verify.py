"""
FND-1 Gate 5: Triple verification of the N=5000 eps=+0.5 signal.

Test 1: REPRODUCIBILITY — different seed, same eps=+0.5, N=5000, M=50
Test 2: NULL CONTROL — flat vs flat (eps=0 vs eps=0), N=5000, M=50
Test 3: DOSE-RESPONSE — eps=+0.25, +0.50, +0.75, N=5000, M=30

If Test 1 reproduces AND Test 2 shows ~5% false positives AND Test 3
shows monotonic scaling → signal is GENUINE.
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
)
from fnd1_gate5_runner import sprinkle_curved, _sprinkle_flat

N_POINTS = 5000
T_DIAMOND = 1.0
N_T = 300
T_MIN = 1e-5
T_MAX = 5.0


def run_paired(eps_curved, N, M, T, seed, label=""):
    """Run M paired (flat, curved) sprinklings. Return fraction significant."""
    V = T**2 / 2.0
    rho = N / V
    t_grid = np.logspace(np.log10(T_MIN), np.log10(T_MAX), N_T)

    ss = np.random.SeedSequence(seed)
    pair_seeds = ss.spawn(M)

    K_flat_all = np.zeros((M, N_T))
    K_curved_all = np.zeros((M, N_T))

    t0 = time.perf_counter()
    for i in range(M):
        if (i + 1) % 10 == 0 or i == 0:
            el = time.perf_counter() - t0
            eta = el / (i + 1) * M - el if i > 0 else 0
            print(f"    [{label}] Pair {i+1}/{M} ({el:.0f}s, ~{eta:.0f}s left)")

        ch = pair_seeds[i].spawn(2)

        # Flat
        pts_f, C_f = _sprinkle_flat(N, T, np.random.default_rng(ch[0]))
        L_f = build_bd_L(C_f, compute_interval_cardinalities(C_f), rho)
        eig_f = compute_family_B_eigenvalues(L_f)
        K_flat_all[i] = compute_heat_trace(eig_f, t_grid)

        # Curved (or second flat for null control)
        if eps_curved == 0.0:
            # Null control: second independent flat sprinkling
            pts_c, C_c = _sprinkle_flat(N, T, np.random.default_rng(ch[1]))
        else:
            pts_c, C_c = sprinkle_curved(N, eps_curved, T,
                                         np.random.default_rng(ch[1]))
        L_c = build_bd_L(C_c, compute_interval_cardinalities(C_c), rho)
        eig_c = compute_family_B_eigenvalues(L_c)
        K_curved_all[i] = compute_heat_trace(eig_c, t_grid)

    DK = K_curved_all - K_flat_all
    DK_mean = np.mean(DK, axis=0)

    # Paired t-test at each t
    p_values = np.ones(N_T)
    t_stats = np.zeros(N_T)
    for j in range(N_T):
        dk_j = DK[:, j]
        if np.std(dk_j, ddof=1) > 0:
            t_stats[j], p_values[j] = stats.ttest_1samp(dk_j, 0.0)

    frac_005 = float(np.mean(p_values < 0.05))
    frac_001 = float(np.mean(p_values < 0.01))

    # Key t-values
    key_results = []
    for t_target in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        idx = np.argmin(np.abs(t_grid - t_target))
        dk_s = DK[:, idx]
        ts, pv = stats.ttest_1samp(dk_s, 0.0)
        key_results.append((t_grid[idx], float(np.mean(dk_s)),
                           float(np.std(dk_s, ddof=1)/np.sqrt(M)),
                           float(ts), float(pv)))

    # Mean ΔK sign in intermediate range (t=0.005 to 0.1)
    mask_mid = (t_grid >= 0.005) & (t_grid <= 0.1)
    mean_DK_mid = float(np.mean(DK_mean[mask_mid]))

    wall = time.perf_counter() - t0
    return {
        "frac_005": frac_005,
        "frac_001": frac_001,
        "key_results": key_results,
        "mean_DK_mid": mean_DK_mid,
        "wall_sec": wall,
    }


def print_result(label, r, eps):
    """Pretty-print one test result."""
    print(f"\n  --- {label} (eps={eps:+.3f}) ---")
    print(f"  Fraction significant: {r['frac_005']*100:.1f}% (p<0.05), "
          f"{r['frac_001']*100:.1f}% (p<0.01)")
    print(f"  Mean ΔK in mid-range: {r['mean_DK_mid']:+.4f}")
    print(f"  {'t':>8} {'mean(ΔK)':>10} {'SEM':>8} {'t-stat':>7} "
          f"{'p-val':>8} {'sig':>4}")
    for (t, dk, sem, ts, pv) in r["key_results"]:
        s = "**" if pv < 0.05 else ("*" if pv < 0.1 else "")
        print(f"  {t:8.4f} {dk:+10.4f} {sem:8.4f} {ts:7.2f} {pv:8.4f} {s:>4}")
    print(f"  Wall time: {r['wall_sec']:.0f}s ({r['wall_sec']/60:.1f} min)")


if __name__ == "__main__":
    t_total = time.perf_counter()

    print("=" * 70)
    print("FND-1 GATE 5: TRIPLE VERIFICATION")
    print("=" * 70)
    print(f"N={N_POINTS}, T={T_DIAMOND}")
    print()

    # === TEST 1: REPRODUCIBILITY (different seed) ===
    print("TEST 1: REPRODUCIBILITY — eps=+0.5, seed=12345 (original was 42)")
    r1 = run_paired(0.5, N_POINTS, 50, T_DIAMOND, 12345, "REPRO")
    print_result("TEST 1: REPRODUCIBILITY", r1, 0.5)

    # === TEST 2: NULL CONTROL (flat vs flat) ===
    print("\n\nTEST 2: NULL CONTROL — flat vs flat, seed=99999")
    r2 = run_paired(0.0, N_POINTS, 50, T_DIAMOND, 99999, "NULL")
    print_result("TEST 2: NULL CONTROL", r2, 0.0)

    # === TEST 3: DOSE-RESPONSE ===
    print("\n\nTEST 3: DOSE-RESPONSE — eps=+0.25, +0.50, +0.75, M=30")
    dose_results = {}
    for eps in [0.25, 0.50, 0.75]:
        print(f"\n  eps={eps:+.3f}:")
        r = run_paired(eps, N_POINTS, 30, T_DIAMOND,
                       77777 + int(eps * 1000), f"DOSE-{eps}")
        dose_results[eps] = r
        print_result(f"TEST 3: DOSE eps={eps}", r, eps)

    # === VERDICT ===
    total_wall = time.perf_counter() - t_total
    print(f"\n\n{'='*70}")
    print("TRIPLE VERIFICATION VERDICT")
    print(f"{'='*70}")

    # Test 1: reproducibility
    repro_pass = r1["frac_005"] > 0.10
    print(f"\n  TEST 1 (Reproducibility):")
    print(f"    Original (seed=42): 27.7% significant")
    print(f"    New (seed=12345):   {r1['frac_005']*100:.1f}% significant")
    print(f"    → {'REPRODUCED' if repro_pass else 'NOT REPRODUCED'}")

    # Test 2: null control
    null_ok = r2["frac_005"] < 0.10
    print(f"\n  TEST 2 (Null control):")
    print(f"    Flat vs flat: {r2['frac_005']*100:.1f}% significant "
          f"(expected ~5%)")
    print(f"    → {'CLEAN' if null_ok else 'CONTAMINATED (too many false positives)'}")

    # Test 3: dose-response
    fracs = [dose_results[e]["frac_005"] for e in [0.25, 0.50, 0.75]]
    monotonic = fracs[0] <= fracs[1] <= fracs[2]
    dks = [dose_results[e]["mean_DK_mid"] for e in [0.25, 0.50, 0.75]]
    dk_monotonic = (dks[0] >= dks[1] >= dks[2]) or (dks[0] <= dks[1] <= dks[2])

    print(f"\n  TEST 3 (Dose-response):")
    for eps in [0.25, 0.50, 0.75]:
        r = dose_results[eps]
        print(f"    eps={eps:+.2f}: {r['frac_005']*100:.1f}% sig, "
              f"mean ΔK_mid = {r['mean_DK_mid']:+.4f}")
    print(f"    Fraction monotonic: {monotonic}")
    print(f"    ΔK monotonic: {dk_monotonic}")
    print(f"    → {'MONOTONIC' if monotonic and dk_monotonic else 'NOT MONOTONIC'}")

    # Overall
    if repro_pass and null_ok and monotonic:
        verdict = "SIGNAL VERIFIED — genuine curvature sensitivity"
    elif repro_pass and null_ok:
        verdict = "SIGNAL PARTIALLY VERIFIED — reproducible but not monotonic"
    elif not repro_pass:
        verdict = "SIGNAL NOT REPRODUCED — likely noise"
    elif not null_ok:
        verdict = "NULL CONTROL FAILED — methodology issue"
    else:
        verdict = "INCONCLUSIVE"

    print(f"\n  OVERALL: {verdict}")
    print(f"  Total wall time: {total_wall:.0f}s ({total_wall/60:.1f} min)")

    # Save
    project_root = Path(__file__).resolve().parent.parent.parent
    out_path = (project_root / "speculative" / "numerics" /
                "ensemble_results" / "gate5_triple_verification.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save = {
        "test1_reproducibility": {
            "seed": 12345, "eps": 0.5, "M": 50,
            "frac_005": r1["frac_005"], "frac_001": r1["frac_001"],
            "mean_DK_mid": r1["mean_DK_mid"],
            "reproduced": repro_pass,
        },
        "test2_null_control": {
            "seed": 99999, "eps": 0.0, "M": 50,
            "frac_005": r2["frac_005"], "frac_001": r2["frac_001"],
            "clean": null_ok,
        },
        "test3_dose_response": {
            eps: {"frac_005": dose_results[eps]["frac_005"],
                  "frac_001": dose_results[eps]["frac_001"],
                  "mean_DK_mid": dose_results[eps]["mean_DK_mid"]}
            for eps in [0.25, 0.50, 0.75]
        },
        "verdict": verdict,
        "wall_time_sec": total_wall,
    }
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"  Saved: {out_path}")
