"""
Discovery Run 001 — Validation 1: N-Scaling Test
==================================================
Does the Forman-Ricci degree-corrected residual persist at N=1000, 2000?

PRE-REGISTERED ANALYSIS:
- Test pp-wave (eps=5, 10) and Schwarzschild (eps=0.02) at N=500, 1000, 2000
- Bonferroni alpha = 0.01/30 = 0.000333
- "Genuine" requires: p < alpha_bonf AND max_R2(proxy) < 0.50
- Report ALL results including nulls

NULL CONTROLS:
- conformal (eps=5): must give exactly 0 (identical causal matrix)

Author: David Alfyorov
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import *
import json, time

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
os.makedirs(OUTDIR, exist_ok=True)

T = 1.0
M = 30  # trials per condition

TESTS = [
    # (geometry, eps, N_values)
    ("conformal",     5.0,   [500]),           # NULL CONTROL
    ("ppwave_quad",   5.0,   [500, 1000, 2000]),
    ("ppwave_quad",   10.0,  [500, 1000, 2000]),
    ("schwarzschild", 0.02,  [500, 1000]),      # N=2000 Schw is slow
]

def main():
    print("=" * 70)
    print("VALIDATION 1: N-Scaling of Forman-Ricci Residual")
    print(f"M={M} trials per condition. Bonferroni alpha = {0.01/30:.6f}")
    print("=" * 70)

    all_results = {}

    for geo, eps, N_list in TESTS:
        for N in N_list:
            label = f"{geo}_eps{eps}_N{N}"
            print(f"\n--- {label} ---")
            t0 = time.time()

            results = []
            for trial in range(M):
                seed = trial * 1000
                res = crn_trial_full(seed, N, T, geo, eps)
                results.append(res)
                if (trial + 1) % 10 == 0:
                    elapsed = time.time() - t0
                    resids = [r["F_residual"] for r in results]
                    print(f"  trial {trial+1}/{M}: resid={np.mean(resids):+.4f} [{elapsed:.1f}s]")

            elapsed = time.time() - t0
            verdict = analyze_deltas(results, label)
            verdict["elapsed_sec"] = elapsed
            verdict["trials"] = results
            all_results[label] = verdict

    # Save
    outpath = os.path.join(OUTDIR, "val1_nscaling.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # Summary table
    print("\n" + "=" * 70)
    print("N-SCALING SUMMARY")
    print(f"{'condition':35s} {'residual':>10s} {'d':>8s} {'p':>12s} {'R2_max':>8s} {'verdict':>20s}")
    print("-" * 95)
    for label, v in all_results.items():
        print(f"{label:35s} {v['mean_residual']:+10.4f} {v['cohen_d_residual']:+8.3f} "
              f"{v['p_residual']:12.2e} {v['max_r2_proxy']:8.3f} {v['verdict']:>20s}")


if __name__ == "__main__":
    main()
