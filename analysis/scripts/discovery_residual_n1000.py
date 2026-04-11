"""
Test 2: Curvature residual at N=1000
=====================================
Does the lapse-separated curvature residual persist at N=1000?
ORC collapsed at N=1000 (R²_multi went from 0.34 to 0.91).
The residual must survive the same test.

PRE-REGISTERED:
  - Run triple CRN at N=1000, pp-wave eps=10, M=30
  - Adversarial proxy check on residual with ALL degree stats
  - PROXY if R²_multi > 0.50 (same threshold as before)

Author: David Alfyorov
"""
import numpy as np
from scipy import stats
import json, time, gc, os, sys

sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import *
from discovery_lapse_separation import causal_synthetic_lapse
from discovery_weyl_probes import interval_volumes_binned, weyl_observables
from discovery_residual_proxy import crn_trial_residual_full, adversarial_on_residual

METRIC_FNS["synthetic_lapse"] = causal_synthetic_lapse
SEED_OFFSETS["synthetic_lapse"] = 100

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
T = 1.0

def main():
    N = 1000
    M = 30
    eps = 10.0

    print("=" * 70)
    print(f"TEST 2: Curvature Residual at N={N}")
    print(f"pp-wave eps={eps}, M={M}")
    print("PRE-REGISTERED: R²_multi > 0.50 → PROXY → N-scaling FAILS")
    print("=" * 70)

    t0 = time.time()
    results = []
    for trial in range(M):
        res = crn_trial_residual_full(trial * 1000, N, T, eps)
        results.append(res)
        if (trial + 1) % 10 == 0:
            elapsed = time.time() - t0
            r = [x.get("scaling_exp_delta_residual", np.nan) for x in results]
            r_v = [v for v in r if not np.isnan(v)]
            print(f"  trial {trial+1}/{M}: resid={np.mean(r_v):+.4f} [{elapsed:.1f}s]")

    elapsed = time.time() - t0
    print(f"  Total: {elapsed:.1f}s")

    all_results = {}
    for obs in ["scaling_exp_delta_residual", "var_ratio_delta_residual"]:
        for proxy_suffix in ["ppw_flat_delta", "ppw_syn_delta"]:
            v = adversarial_on_residual(results, obs, proxy_suffix, f"N{N}")
            all_results[f"{obs}_{proxy_suffix}"] = v

    # Also compare with N=500 effect sizes
    print(f"\n  N-SCALING COMPARISON:")
    print(f"    N=500  scaling_exp residual: d=-1.82 (from Test 1)")
    r500 = [x.get("scaling_exp_delta_residual", np.nan) for x in results]
    r_v = [v for v in r500 if not np.isnan(v)]
    d_1000 = np.mean(r_v) / np.std(r_v) if np.std(r_v) > 0 else 0
    print(f"    N=1000 scaling_exp residual: d={d_1000:+.2f}")
    if abs(d_1000) > abs(-1.82):
        print(f"    Signal STRONGER at N=1000 ✓")
    elif abs(d_1000) > 0.5:
        print(f"    Signal persists at N=1000 ✓ (weaker)")
    else:
        print(f"    Signal VANISHES at N=1000 ✗")

    outpath = os.path.join(OUTDIR, "residual_n1000.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # Summary
    print(f"\n{'='*70}")
    print("TEST 2 SUMMARY")
    print(f"{'='*70}")
    any_proxy = False
    for key, v in all_results.items():
        print(f"  {key:50s}: R²m={v.get('r2_multiple',0):.3f}, "
              f"adj={v.get('adj_r2',0):.3f}, {v.get('verdict','?')}")
        if "PROXY" in v.get("verdict", ""):
            any_proxy = True

    if any_proxy:
        print("\n  ☠️ RESIDUAL COLLAPSES AT N=1000 (like ORC)")
    else:
        print("\n  ✅ RESIDUAL SURVIVES AT N=1000")

if __name__ == "__main__":
    main()
