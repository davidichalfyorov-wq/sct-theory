#!/usr/bin/env python3
"""NC-2: Matched-ΔTC geometric control.

Tests whether path_kurtosis distinguishes curvature channels
beyond generic causal-volume change.

Method:
1. Target: pp-wave exact at eps
2. Control: dS exact at H (bisected to match ΔTC)
3. Compare dk_target vs dk_control at matched ΔTC

If dk_target >> dk_control at same ΔTC → curvature-channel-specific (good!)
If dk_target ≈ dk_control → just pair-count detector (bad)
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, excess_kurtosis, ppwave_exact_preds, ds_preds, bulk_mask
)

N = 10000
ZETA = 0.15
T = 1.0
EPS_TARGET = 3.0
M_SEEDS = 15
N_BISECT = 15
H_BRACKET = (0.01, 5.0)


def count_causal_pairs(pts, pred_fn):
    total = 0
    for i in range(len(pts)):
        total += int(pred_fn(pts, i).sum())
    return total


if __name__ == "__main__":
    print(f"=== NC-2: MATCHED-ΔTC GEOMETRIC CONTROL ===", flush=True)
    print(f"Target: pp-wave exact eps={EPS_TARGET}, T={T}", flush=True)
    print(f"Control: dS exact (H bisected to match ΔTC)", flush=True)
    print(f"N={N}, M_seeds={M_SEEDS}, zeta={ZETA}", flush=True)
    print(flush=True)

    results = []
    t_total = time.time()

    for si in range(M_SEEDS):
        seed = 1000000 + si
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)

        t0 = time.time()

        # Flat TC
        TC0 = count_causal_pairs(pts, lambda P, i: minkowski_preds(P, i))

        # Target TC (pp-wave)
        TCA = count_causal_pairs(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS_TARGET))
        dTC_target = TCA - TC0

        # Bisect on dS H to match ΔTC
        lo, hi = H_BRACKET
        for step in range(N_BISECT):
            mid = 0.5 * (lo + hi)
            TCB = count_causal_pairs(pts, lambda P, i: ds_preds(P, i, H=mid))
            dTC_mid = TCB - TC0
            if abs(dTC_target) < 1:
                break
            if dTC_mid < dTC_target:
                lo = mid
            else:
                hi = mid

        H_matched = 0.5 * (lo + hi)
        TCB_final = count_causal_pairs(pts, lambda P, i: ds_preds(P, i, H=H_matched))
        dTC_match = TCB_final - TC0
        match_err = abs(dTC_match - dTC_target) / max(abs(dTC_target), 1)

        # Now build Hasse for flat, target, control
        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)

        parA, chA = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS_TARGET))
        YA = Y_from_graph(parA, chA)

        parB, chB = build_hasse_from_predicate(pts, lambda P, i: ds_preds(P, i, H=H_matched))
        YB = Y_from_graph(parB, chB)

        mask = bulk_mask(pts, T, ZETA)
        k0 = excess_kurtosis(Y0[mask])
        dkA = excess_kurtosis(YA[mask]) - k0
        dkB = excess_kurtosis(YB[mask]) - k0

        elapsed = time.time() - t0
        print(f"  seed {si}: dTC_target={dTC_target}, dTC_match={dTC_match}, "
              f"match_err={match_err:.3f}, H={H_matched:.4f}, "
              f"dk_ppw={dkA:+.6f}, dk_dS={dkB:+.6f} ({elapsed:.0f}s)", flush=True)

        results.append({
            "seed": seed,
            "dTC_target": int(dTC_target),
            "dTC_match": int(dTC_match),
            "match_relerr": float(match_err),
            "H_matched": float(H_matched),
            "dk_target": float(dkA),
            "dk_control": float(dkB),
        })

    # Summary
    arr_A = np.array([r["dk_target"] for r in results])
    arr_B = np.array([r["dk_control"] for r in results])
    match_errs = np.array([r["match_relerr"] for r in results])

    R_TC = abs(np.mean(arr_B)) / max(abs(np.mean(arr_A)), 1e-15)
    diff = np.mean(arr_A) - np.mean(arr_B)
    se_diff = np.std(arr_A - arr_B, ddof=1) / np.sqrt(len(arr_A))

    print(f"\n=== RESULTS ===", flush=True)
    print(f"  Mean match error: {np.mean(match_errs):.3f}", flush=True)
    print(f"  dk_target (ppw): {np.mean(arr_A):+.6f} ± {np.std(arr_A,ddof=1)/np.sqrt(len(arr_A)):.6f}", flush=True)
    print(f"  dk_control (dS): {np.mean(arr_B):+.6f} ± {np.std(arr_B,ddof=1)/np.sqrt(len(arr_B)):.6f}", flush=True)
    print(f"  R_TC = |dk_control|/|dk_target| = {R_TC:.4f}", flush=True)
    print(f"  Diff dk_A−dk_B = {diff:+.6f} ± {se_diff:.6f}", flush=True)

    if R_TC < 0.5 and abs(diff) > 2 * se_diff:
        verdict = "STRONG: beyond pair-count, curvature-channel-specific"
    elif R_TC < 0.8:
        verdict = "MODERATE: partial channel specificity"
    else:
        verdict = "WEAK/BAD: not clearly beyond pair-count"
    print(f"  VERDICT: {verdict}", flush=True)

    total = time.time() - t_total
    print(f"\n  Total: {total:.0f}s = {total/60:.1f}min", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "universal_runs_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "nc2_matched_tc.json"), "w") as f:
        json.dump({
            "per_seed": results,
            "R_TC": float(R_TC),
            "verdict": verdict,
        }, f, indent=2)
    print("Saved.", flush=True)
