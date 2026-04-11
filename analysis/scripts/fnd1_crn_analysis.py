"""
FND-1: CRN Post-Analysis.

Reads the CRN results (sj_crn.json) and answers the key question:
Is delta(observable) / delta(TC) constant across eps values?

If constant -> observable is just counting pairs (no curvature info beyond TC)
If varies with eps -> observable carries information beyond pair counting

Also fits: delta(obs) = a * delta(TC) + b * delta(TC)^2 + residual
If the residual correlates with eps -> nonlinear TC doesn't explain everything.

Run after CRN completes:
  python analysis/scripts/fnd1_crn_analysis.py
"""

import json, sys
import numpy as np
from scipy import stats
from pathlib import Path

RESULTS_PATH = Path("speculative/numerics/ensemble_results/sj_crn.json")


def main():
    if not RESULTS_PATH.exists():
        print(f"Waiting for {RESULTS_PATH}...", flush=True)
        return

    with open(RESULTS_PATH) as f:
        data = json.load(f)

    eps_values = sorted([float(k) for k in data["results"].keys()])
    obs_names = ["trace_W", "n_modes", "spectral_gap_ratio", "entropy_spectral",
                 "spectral_width", "trace_trunc", "lambda_median"]

    print("=" * 70, flush=True)
    print("FND-1 CRN POST-ANALYSIS", flush=True)
    print("=" * 70, flush=True)
    print(f"Question: is delta(obs)/delta(TC) constant across eps?", flush=True)
    print(f"If constant -> obs just counts pairs (NEGATIVE for FND-1)", flush=True)
    print(f"If varies -> obs carries info beyond TC (POSITIVE for FND-1)", flush=True)
    print(flush=True)

    # Extract mean deltas
    dtc = []
    dobs = {k: [] for k in obs_names}

    print(f"{'eps':>5} {'delta_TC':>10}", end="", flush=True)
    for obs in obs_names:
        print(f" {obs[:8]:>10}", end="")
    print(flush=True)

    for eps in eps_values:
        r = data["results"][str(eps)]
        tc_mean = r["delta_TC"]["mean"]
        dtc.append(tc_mean)
        print(f"{eps:5.2f} {tc_mean:+10.0f}", end="", flush=True)
        for obs in obs_names:
            if obs in r:
                m = r[obs]["mean"]
                dobs[obs].append(m)
                print(f" {m:+10.4f}", end="")
            else:
                dobs[obs].append(0)
                print(f" {'n/a':>10}", end="")
        print(flush=True)

    dtc = np.array(dtc)

    # TEST 1: Is delta(obs)/delta(TC) constant?
    print(f"\n{'='*70}", flush=True)
    print("TEST 1: RATIO delta(obs) / delta(TC)", flush=True)
    print("=" * 70, flush=True)
    print(f"  If CV < 0.10 -> constant (obs ~ TC, NEGATIVE)", flush=True)
    print(f"  If CV > 0.10 -> varies (obs carries info beyond TC, needs further test)", flush=True)
    print(flush=True)

    print(f"  {'obs':>15} {'ratio_mean':>12} {'ratio_std':>10} {'CV':>8} {'verdict':>10}", flush=True)

    for obs in obs_names:
        d = np.array(dobs[obs])
        if np.all(np.abs(dtc) > 1):
            ratios = d / dtc
            mean_r = np.mean(ratios)
            std_r = np.std(ratios)
            cv = std_r / abs(mean_r) if abs(mean_r) > 1e-20 else float('inf')
            verdict = "CONSTANT" if cv < 0.10 else ("VARIES" if cv < 0.30 else "UNSTABLE")
            print(f"  {obs:>15} {mean_r:+12.6f} {std_r:10.6f} {cv:8.4f} {verdict:>10}",
                  flush=True)
        else:
            print(f"  {obs:>15} {'n/a':>12}", flush=True)

    # TEST 2: Residual after linear TC fit
    print(f"\n{'='*70}", flush=True)
    print("TEST 2: RESIDUAL after delta(obs) = a * delta(TC) + b", flush=True)
    print("=" * 70, flush=True)
    print(f"  If residual correlates with eps -> info beyond linear TC", flush=True)
    print(flush=True)

    print(f"  {'obs':>15} {'a':>10} {'R^2':>8} {'resid_vs_eps':>12} {'p':>10}", flush=True)

    for obs in obs_names:
        d = np.array(dobs[obs])
        if len(d) >= 3 and np.std(dtc) > 0:
            slope, intercept, r_val, p_val, _ = stats.linregress(dtc, d)
            r_sq = r_val**2
            resid = d - (slope * dtc + intercept)
            if np.std(resid) > 1e-20:
                r_resid, p_resid = stats.pearsonr(np.array(eps_values), resid)
            else:
                r_resid, p_resid = 0, 1
            print(f"  {obs:>15} {slope:+10.6f} {r_sq:8.4f} {r_resid:+12.4f} {p_resid:10.2e}",
                  flush=True)

    # TEST 3: Quadratic TC fit residual
    print(f"\n{'='*70}", flush=True)
    print("TEST 3: RESIDUAL after delta(obs) = a*dTC + b*dTC^2 + c", flush=True)
    print("=" * 70, flush=True)

    print(f"  {'obs':>15} {'R^2_quad':>10} {'resid_vs_eps':>12} {'p':>10}", flush=True)

    for obs in obs_names:
        d = np.array(dobs[obs])
        if len(d) >= 4:
            X = np.column_stack([dtc, dtc**2, np.ones(len(dtc))])
            beta = np.linalg.lstsq(X, d, rcond=None)[0]
            pred = X @ beta
            resid = d - pred
            ss_res = np.sum(resid**2)
            ss_tot = np.sum((d - np.mean(d))**2)
            r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            if np.std(resid) > 1e-20 and len(resid) >= 3:
                r_resid, p_resid = stats.pearsonr(np.array(eps_values), resid)
            else:
                r_resid, p_resid = 0, 1
            print(f"  {obs:>15} {r_sq:10.6f} {r_resid:+12.4f} {p_resid:10.2e}", flush=True)

    # VERDICT
    print(f"\n{'='*70}", flush=True)
    print("VERDICT", flush=True)
    print("=" * 70, flush=True)

    # Count observables with non-constant ratio
    n_varies = 0
    n_resid = 0
    for obs in obs_names:
        d = np.array(dobs[obs])
        if len(d) >= 3 and np.all(np.abs(dtc) > 1):
            ratios = d / dtc
            cv = np.std(ratios) / abs(np.mean(ratios)) if abs(np.mean(ratios)) > 1e-20 else float('inf')
            if cv > 0.10:
                n_varies += 1

        if len(d) >= 4:
            X = np.column_stack([dtc, dtc**2, np.ones(len(dtc))])
            beta = np.linalg.lstsq(X, d, rcond=None)[0]
            resid = d - X @ beta
            if np.std(resid) > 1e-20 and len(resid) >= 3:
                _, p = stats.pearsonr(np.array(eps_values), resid)
                if p < 0.05:
                    n_resid += 1

    print(f"\n  Ratio test: {n_varies}/7 observables have non-constant delta/dTC (CV>0.10)", flush=True)
    print(f"  Residual test: {n_resid}/7 have significant residual after quadratic dTC fit", flush=True)

    if n_resid >= 3:
        print(f"\n  EVIDENCE FOR: SJ spectrum carries information beyond TC at N=5000", flush=True)
    elif n_varies >= 3:
        print(f"\n  WEAK EVIDENCE: ratios vary, but quadratic TC may explain it", flush=True)
    else:
        print(f"\n  EVIDENCE AGAINST: SJ spectrum changes are consistent with TC counting", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
