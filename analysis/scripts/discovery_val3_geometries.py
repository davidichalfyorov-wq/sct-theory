"""
Discovery Run 001 — Validation 3: Multiple Geometries
=======================================================
Does Forman-Ricci residual detect curvature across different spacetimes?

TESTS (N=500, M=40):
1. conformal (eps=5)     — NULL CONTROL (must give 0)
2. ppwave_quad (eps=5)   — known positive from pilot
3. ppwave_cross (eps=5)  — symmetric profile, may cancel
4. schwarzschild (eps=0.05) — higher eps than pilot (0.02 was proxy-dominated)
5. flrw (eps=1.0)        — KNOWN ARTIFACT (midpoint approx, NOT curvature)

PRE-REGISTERED:
- conformal MUST be null. If not → implementation bug.
- flrw signal (if any) is midpoint artifact, NOT curvature. Flag explicitly.
- Bonferroni alpha = 0.01/30 = 0.000333

Author: David Alfyorov
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import *
import json, time

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
os.makedirs(OUTDIR, exist_ok=True)

N = 500
T = 1.0
M = 40

GEOMETRIES = [
    ("conformal",     5.0),   # NULL CONTROL
    ("ppwave_quad",   5.0),   # positive control
    ("ppwave_cross",  5.0),   # cross-polarization
    ("schwarzschild", 0.05),  # stronger than pilot
    ("flrw",          1.0),   # midpoint artifact control
]

def main():
    print("=" * 70)
    print("VALIDATION 3: Multiple Geometries")
    print(f"N={N}, M={M}")
    print("Bonferroni alpha = 0.000333")
    print("=" * 70)

    all_results = {}

    for geo, eps in GEOMETRIES:
        label = f"{geo}_eps{eps}"
        is_null = (geo == "conformal")
        is_artifact = (geo == "flrw")

        note = ""
        if is_null:
            note = " [NULL CONTROL — must be 0]"
        elif is_artifact:
            note = " [MIDPOINT ARTIFACT — not curvature]"

        print(f"\n--- {label}{note} ---")
        t0 = time.time()

        results = []
        for trial in range(M):
            seed = trial * 1000
            res = crn_trial_full(seed, N, T, geo, eps)
            results.append(res)
            if (trial + 1) % 20 == 0:
                elapsed = time.time() - t0
                resids = [r["F_residual"] for r in results]
                print(f"  trial {trial+1}/{M}: resid={np.mean(resids):+.4f} [{elapsed:.1f}s]")

        elapsed = time.time() - t0
        verdict = analyze_deltas(results, label)
        verdict["elapsed_sec"] = elapsed
        verdict["is_null_control"] = is_null
        verdict["is_artifact_control"] = is_artifact
        verdict["trials"] = results
        all_results[label] = verdict

        # Null control check
        if is_null:
            resids = [r["F_residual"] for r in results]
            if abs(np.mean(resids)) > 1e-10:
                print(f"  *** NULL CONTROL FAILED: residual = {np.mean(resids):.2e} ***")
                print(f"  *** This indicates an implementation bug! ***")
            else:
                print(f"  NULL CONTROL PASSED: residual = {np.mean(resids):.2e} (exact 0)")

    # Save
    outpath = os.path.join(OUTDIR, "val3_geometries.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # Summary table
    print("\n" + "=" * 70)
    print("MULTI-GEOMETRY SUMMARY")
    print(f"{'geometry':25s} {'residual':>10s} {'d':>8s} {'p':>12s} {'R2_max':>8s} {'verdict':>20s} {'note':>15s}")
    print("-" * 100)
    for label, v in all_results.items():
        note = ""
        if v.get("is_null_control"):
            note = "NULL CTRL"
        elif v.get("is_artifact_control"):
            note = "ARTIFACT"
        print(f"{label:25s} {v['mean_residual']:+10.4f} {v['cohen_d_residual']:+8.3f} "
              f"{v['p_residual']:12.2e} {v['max_r2_proxy']:8.3f} {v['verdict']:>20s} {note:>15s}")

    # Count detections
    genuine = sum(1 for v in all_results.values()
                  if v["verdict"] == "DETECTED (genuine)"
                  and not v.get("is_null_control")
                  and not v.get("is_artifact_control"))
    total_real = sum(1 for v in all_results.values()
                     if not v.get("is_null_control")
                     and not v.get("is_artifact_control"))
    print(f"\n  Genuine detections: {genuine}/{total_real}")


if __name__ == "__main__":
    main()
