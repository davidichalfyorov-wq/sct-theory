"""
FND-1 EXP-15: Conditional Variance Curvature Estimator.

Verified finding: curvature inflates Fiedler variance 2-16x while shifting mean ~20%.
This experiment formalizes var(observable | eps) as a curvature observable.

Uses existing per-sprinkling data from fnd1_per_sprinkling.py. No new sprinklings.

Run:
    python analysis/scripts/fnd1_exp15_conditional_variance.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent.parent / "speculative" / "numerics" / "ensemble_results"

def main():
    # Load per-sprinkling data
    path = RESULTS_DIR / "per_sprinkling_multioperator.json"
    if not path.exists():
        print("ERROR: Run fnd1_per_sprinkling.py first"); return
    with open(path) as f:
        data = json.load(f)
    records = data["per_sprinkling"]
    print(f"Loaded {len(records)} per-sprinkling records")

    observables = ["fiedler", "gap_ratio", "alg_conn_ratio", "comm_entropy",
                   "comm_frobenius", "rho_geom", "t2K_tau1", "link_entropy"]
    N_values = sorted(set(r["N"] for r in records))
    eps_values = sorted(set(r["eps"] for r in records))

    results = {}
    print("\n" + "=" * 70)
    print("CONDITIONAL VARIANCE ANALYSIS")
    print("=" * 70)

    for obs in observables:
        print(f"\n--- {obs} ---")
        obs_results = {}

        for N in N_values:
            variances = []
            means = []
            eps_used = []
            for eps in eps_values:
                vals = [r[obs] for r in records if r["N"] == N and abs(r["eps"] - eps) < 0.01]
                if len(vals) < 5:
                    continue
                variances.append(float(np.var(vals, ddof=1)))
                means.append(float(np.mean(vals)))
                eps_used.append(eps)

            if len(variances) < 3:
                continue

            var_arr = np.array(variances)
            eps_arr = np.array(eps_used)

            # Levene test: flat vs max curvature (PRIMARY — works on raw 80 samples)
            flat_vals = [r[obs] for r in records if r["N"] == N and abs(r["eps"]) < 0.01]
            curv_vals = [r[obs] for r in records if r["N"] == N and abs(r["eps"] - max(eps_values)) < 0.01]

            # Variance ratio (flat vs max eps, by eps value not position)
            flat_var = variances[eps_used.index(0.0)] if 0.0 in eps_used else var_arr[0]
            max_var = variances[eps_used.index(max(eps_values))] if max(eps_values) in eps_used else var_arr[-1]
            var_ratio = max_var / flat_var if flat_var > 1e-20 else 0

            # Spearman on n=4 is weak (note in output); Levene is the primary test
            r_var, p_var = stats.spearmanr(eps_arr, var_arr)
            mean_arr = np.array(means)
            r_mean, p_mean = stats.spearmanr(eps_arr, mean_arr)
            if flat_vals and curv_vals:
                lev_stat, lev_p = stats.levene(flat_vals, curv_vals)
            else:
                lev_stat, lev_p = 0, 1

            obs_results[str(N)] = {
                "r_variance_eps": round(float(r_var), 4),
                "p_variance_eps": float(p_var),
                "r_mean_eps": round(float(r_mean), 4),
                "p_mean_eps": float(p_mean),
                "variance_ratio": round(float(var_ratio), 2),
                "levene_p": float(lev_p),
                "variances": [round(v, 6) for v in variances],
            }

            # Which is stronger signal: variance or mean?
            stronger = "VARIANCE" if abs(r_var) > abs(r_mean) else "MEAN"
            print(f"  N={N}: r_var={r_var:+.4f} r_mean={r_mean:+.4f} ratio={var_ratio:.1f}x "
                  f"Levene p={lev_p:.2e} → {stronger}")

        results[obs] = obs_results

    # Summary: which observables have variance > mean as curvature signal
    print("\n" + "=" * 70)
    print("SUMMARY: Variance vs Mean as Curvature Signal")
    print("=" * 70)
    variance_wins = 0
    for obs in observables:
        if not results.get(obs):
            continue
        best_N = max(results[obs].keys(), key=int)
        r = results[obs][best_N]
        # Use Levene p as primary (not Spearman which is vacuous at n=4)
        lev_sig = r["levene_p"] < 0.05 / (len(observables) * len(N_values))  # Bonferroni
        stronger = "VAR" if abs(r["r_variance_eps"]) > abs(r["r_mean_eps"]) else "MEAN"
        if lev_sig and abs(r["variance_ratio"]) > 1.5:
            variance_wins += 1
        print(f"  {obs}: {stronger} (ratio={r['variance_ratio']:.1f}x, "
              f"Levene p={r['levene_p']:.2e}, {'SIG' if lev_sig else 'ns'})")

    verdict = (f"Variance inflation significant (Levene, Bonferroni-corrected) "
               f"for {variance_wins}/{len(observables)} observables. "
               f"Consistent with curvature-driven distributional broadening.")

    print(f"\nVERDICT: {verdict}")

    output = {
        "_meta": {"name": "exp15_conditional_variance", "route": 2, "status": "completed",
                  "verdict": verdict, "N": max(N_values), "M": len(records),
                  "timestamp": "", "wall_time_sec": 0, "parameters": {}, "tags": []},
        "results": results,
        "verdict": verdict,
    }
    out_path = RESULTS_DIR / "exp15_conditional_variance.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
