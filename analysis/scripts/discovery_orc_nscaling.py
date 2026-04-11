"""
Discovery Run 001 — ORC N-Scaling Validation
==============================================
Tests whether ORC GENUINE detection on Schwarzschild persists at N=1000.
Also tests pp-wave at stronger eps=20 (weak at eps=10, N=500).

PRE-REGISTERED (anti-bias v1.0):
- Bonferroni alpha = 0.01/10 = 0.001
- Same adversarial protocol as ORC pilot

Author: David Alfyorov
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import *
from discovery_orc_pilot import compute_orc, crn_trial_orc, adversarial_analysis
import json, time

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
os.makedirs(OUTDIR, exist_ok=True)

T = 1.0
M = 30
MAX_EDGES = 300

TESTS = [
    # (geo, eps, N)
    ("schwarzschild", 0.05, 500),    # replicate pilot
    ("schwarzschild", 0.05, 1000),   # N-scaling
    ("ppwave_quad",   20.0, 500),    # stronger eps
    ("ppwave_quad",   10.0, 1000),   # N-scaling for pp-wave
]

def main():
    print("=" * 70)
    print("ORC N-SCALING VALIDATION")
    print(f"M={M}, max_edges={MAX_EDGES}")
    print(f"Bonferroni alpha = {0.01/10:.4f}")
    print("=" * 70)

    all_results = {}

    for geo, eps, N in TESTS:
        label = f"{geo}_eps{eps}_N{N}"
        print(f"\n--- {label} ---")
        t0 = time.time()

        results = []
        for trial in range(M):
            seed = trial * 1000
            res = crn_trial_orc(seed, N, T, geo, eps, max_edges=MAX_EDGES)
            results.append(res)
            if (trial + 1) % 10 == 0:
                elapsed = time.time() - t0
                orc_d = [r["orc_mean_delta"] for r in results if not np.isnan(r["orc_mean_delta"])]
                print(f"  trial {trial+1}/{M}: ORC_delta={np.mean(orc_d):+.5f} [{elapsed:.1f}s]")

        elapsed = time.time() - t0

        v = adversarial_analysis(results, "orc_mean_delta", label)
        v["elapsed_sec"] = elapsed
        v["trials"] = results
        all_results[label] = v

    # Save
    outpath = os.path.join(OUTDIR, "orc_nscaling.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # Summary
    print("\n" + "=" * 70)
    print("ORC N-SCALING SUMMARY")
    print(f"{'condition':35s} {'delta':>10s} {'d':>8s} {'p':>12s} {'R2max':>8s} {'R2multi':>8s} {'verdict':>22s}")
    print("-" * 105)
    for label, v in all_results.items():
        print(f"{label:35s} {v['mean_delta']:+10.5f} {v['cohen_d']:+8.3f} "
              f"{v['p_value']:12.2e} {v['max_r2_single']:8.3f} {v['r2_multiple']:8.3f} "
              f"{v['verdict']:>22s}")

    # Check if Schwarzschild GENUINE persists at N=1000
    schw_1000 = all_results.get("schwarzschild_eps0.05_N1000", {})
    if schw_1000.get("verdict") == "DETECTED (genuine)":
        print("\n*** SCHWARZSCHILD GENUINE CONFIRMED AT N=1000! ***")
        print("*** ORC is a genuine curvature probe on causal sets. ***")
    elif "PROXY" in schw_1000.get("verdict", ""):
        print("\n*** SCHWARZSCHILD COLLAPSED TO PROXY AT N=1000. ***")
    else:
        v = schw_1000.get("verdict", "N/A")
        print(f"\n*** SCHWARZSCHILD AT N=1000: {v} ***")


if __name__ == "__main__":
    main()
