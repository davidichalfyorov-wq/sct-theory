"""
Discovery Run 001 — Weyl Probes: Validation Battery
=====================================================
Three tests to move from 5/10 to 6/10:

1. N-SCALING: scaling_exp + var_ratio at N=1000
2. HOLDOUT: tidal metric (never seen) + ppwave_cross (different polarization)
3. EPS-SCALING: pp-wave at eps=2,5,10,20 — where does signal emerge?

Author: David Alfyorov
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from discovery_common import *
from discovery_weyl_probes import (
    interval_volumes_binned, weyl_observables, crn_trial_weyl, adversarial_weyl,
)
from discovery_orc_deepen import causal_tidal
import json, time

# Make sure tidal and ppwave_cross are registered
METRIC_FNS["tidal"] = causal_tidal
SEED_OFFSETS["tidal"] = 600
from discovery_common import causal_ppwave_cross
METRIC_FNS["ppwave_cross"] = causal_ppwave_cross

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "discovery_runs", "run_001")
os.makedirs(OUTDIR, exist_ok=True)

T = 1.0
OBSERVABLES = ["vol_ratio_delta", "scaling_exp_delta", "var_ratio_delta"]
tau_bins = np.array([0.01, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.45])


def run_condition(geo, eps, N, M, label, all_results):
    """Run one CRN condition with adversarial analysis."""
    print(f"\n--- {label} ---")
    t0 = time.time()
    results = []
    for trial in range(M):
        res = crn_trial_weyl(trial * 1000, N, T, geo, eps, tau_bins)
        results.append(res)
        if (trial + 1) % 10 == 0:
            elapsed = time.time() - t0
            se = [r.get("scaling_exp_delta", np.nan) for r in results]
            se_v = [v for v in se if not np.isnan(v)]
            print(f"  trial {trial+1}/{M}: scale_exp={np.mean(se_v):+.4f} [{elapsed:.1f}s]")

    elapsed = time.time() - t0
    cond_results = {"geometry": geo, "eps": eps, "N": N, "M": M, "elapsed_sec": elapsed}

    for obs_key in OBSERVABLES:
        v = adversarial_weyl(results, obs_key, label)
        cond_results[obs_key] = v

    cond_results["trials"] = results
    all_results[label] = cond_results
    return cond_results


def main():
    all_results = {}

    # ===================================================================
    # TEST 1: N-SCALING (pp-wave ε=10 at N=1000)
    # ===================================================================
    print("=" * 70)
    print("TEST 1: N-SCALING")
    print("=" * 70)

    run_condition("ppwave_quad", 10.0, 500, 30, "ppw10_N500", all_results)
    run_condition("ppwave_quad", 10.0, 1000, 25, "ppw10_N1000", all_results)
    run_condition("schwarzschild", 0.05, 1000, 25, "schw005_N1000", all_results)

    # ===================================================================
    # TEST 2: HOLDOUT GEOMETRIES
    # ===================================================================
    print("\n" + "=" * 70)
    print("TEST 2: HOLDOUT GEOMETRIES (never used in development)")
    print("=" * 70)

    run_condition("ppwave_cross", 20.0, 500, 30, "holdout_ppwcross20", all_results)
    run_condition("tidal", 0.1, 500, 30, "holdout_tidal01", all_results)

    # ===================================================================
    # TEST 3: EPS-SCALING (pp-wave, where does Weyl signal emerge?)
    # ===================================================================
    print("\n" + "=" * 70)
    print("TEST 3: EPS-SCALING (pp-wave)")
    print("=" * 70)

    for eps in [2.0, 5.0, 10.0, 20.0]:
        run_condition("ppwave_quad", eps, 500, 25, f"ppw_eps{eps}", all_results)

    # Save
    outpath = os.path.join(OUTDIR, "weyl_validate.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\n{'label':25s} {'scaling_exp':>12s} {'d':>8s} {'p':>12s} {'R2m':>6s} {'verdict':>22s}")
    print("-" * 90)
    for label, res in all_results.items():
        v = res.get("scaling_exp_delta", {})
        if isinstance(v, dict) and "verdict" in v:
            print(f"{label:25s} {v.get('mean_delta',0):+12.4f} {v.get('cohen_d',0):+8.3f} "
                  f"{v.get('p_value',1):12.2e} {v.get('r2_multiple',0):6.3f} "
                  f"{v['verdict']:>22s}")

    # Count
    genuine = sum(1 for res in all_results.values()
                  for obs in OBSERVABLES
                  if isinstance(res.get(obs, {}), dict)
                  and res[obs].get("verdict") == "DETECTED (genuine)")
    total = sum(1 for res in all_results.values()
                for obs in OBSERVABLES
                if isinstance(res.get(obs, {}), dict)
                and "verdict" in res[obs])
    print(f"\nGenuine: {genuine}/{total}")

    # Eps-scaling analysis
    print("\n  EPS-SCALING (scaling_exp on pp-wave):")
    eps_list = []
    d_list = []
    for label, res in all_results.items():
        if label.startswith("ppw_eps"):
            eps = res["eps"]
            v = res.get("scaling_exp_delta", {})
            if isinstance(v, dict) and not np.isnan(v.get("cohen_d", np.nan)):
                eps_list.append(eps)
                d_list.append(v["cohen_d"])
                print(f"    eps={eps:5.1f}: d={v['cohen_d']:+.3f}, p={v['p_value']:.2e}, "
                      f"verdict={v['verdict']}")

    if len(eps_list) >= 3:
        eps_arr = np.array(eps_list)
        d_arr = np.abs(np.array(d_list))
        slope, _ = np.polyfit(np.log(eps_arr), np.log(d_arr + 0.01), 1)
        print(f"    |d| ~ eps^{slope:.2f}")


if __name__ == "__main__":
    main()
