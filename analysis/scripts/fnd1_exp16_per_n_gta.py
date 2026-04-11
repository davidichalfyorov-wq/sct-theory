"""
FND-1 EXP-16: Per-N GTA Analysis.

Analyze GTA (commutator) observables separately at each N from per-sprinkling data.
Checks whether GTA-eps correlation changes with N (scaling preview without new sprinklings).

Uses existing per-sprinkling data. No new sprinklings.

Run:
    python analysis/scripts/fnd1_exp16_per_n_gta.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy import stats
from numpy.linalg import lstsq

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent.parent / "speculative" / "numerics" / "ensemble_results"

def partial_corr(x, y, controls):
    cx, _, _, _ = lstsq(controls, x, rcond=None)
    cy, _, _, _ = lstsq(controls, y, rcond=None)
    rx = x - controls @ cx
    ry = y - controls @ cy
    if np.std(rx) < 1e-15 or np.std(ry) < 1e-15:
        return 0.0, 1.0
    return stats.pearsonr(rx, ry)

def main():
    path = RESULTS_DIR / "per_sprinkling_multioperator.json"
    if not path.exists():
        print("ERROR: Run fnd1_per_sprinkling.py first"); return
    with open(path) as f:
        data = json.load(f)
    records = data["per_sprinkling"]
    if not records:
        print("ERROR: per_sprinkling data is empty"); return
    print(f"Loaded {len(records)} records")

    N_values = sorted(set(r["N"] for r in records))
    max_key = "comm_max_abs" if "comm_max_abs" in records[0] else "comm_max"
    gta_obs = ["comm_entropy", "comm_frobenius", max_key]

    results = {}
    print("\n" + "=" * 70)
    print("PER-N GTA (Geometric Temporal Asymmetry) ANALYSIS")
    print("=" * 70)

    for N in N_values:
        recs_N = [r for r in records if r["N"] == N]
        eps_arr = np.array([r["eps"] for r in recs_N])
        tc_arr = np.array([r["total_causal"] for r in recs_N])
        bd_arr = np.array([r["bd_action"] for r in recs_N])
        controls = np.column_stack([tc_arr, tc_arr ** 2, bd_arr, np.ones(len(recs_N))])

        print(f"\nN = {N} (n={len(recs_N)})")
        n_results = {}

        for obs in gta_obs:
            obs_arr = np.array([r[obs] for r in recs_N])

            # Raw correlation
            r_raw, p_raw = stats.spearmanr(eps_arr, obs_arr)

            # Partial correlation (mediated)
            r_part, p_part = partial_corr(eps_arr, obs_arr, controls)

            # Purity: r_partial / r_direct
            r_dir, _ = stats.pearsonr(eps_arr, obs_arr)
            purity = min(abs(r_part) / abs(r_dir), 1.0) if abs(r_dir) > 0.05 else 0.0

            n_results[obs] = {
                "r_spearman": round(float(r_raw), 4),
                "r_partial": round(float(r_part), 4),
                "p_partial": float(p_part),
                "purity": round(float(purity), 3),
            }
            print(f"  {obs}: r_raw={r_raw:+.4f}, r_partial={r_part:+.4f} "
                  f"(p={p_part:.2e}), purity={purity:.1%}")

        results[str(N)] = n_results

    # Scaling summary
    print("\n" + "=" * 70)
    print("GTA SCALING PREVIEW (from per-sprinkling data)")
    print("=" * 70)

    for obs in gta_obs:
        r_values = [results[str(N)][obs]["r_partial"] for N in N_values]
        purity_values = [results[str(N)][obs]["purity"] for N in N_values]
        print(f"  {obs}:")
        for i, N in enumerate(N_values):
            print(f"    N={N}: |r|={abs(r_values[i]):.4f}, purity={purity_values[i]:.1%}")

        # Trend
        if len(r_values) >= 3:
            abs_r = [abs(x) for x in r_values]
            lr = stats.linregress(np.log(N_values), abs_r)
            print(f"    slope={lr.slope:+.4f} (p={lr.pvalue:.2f}, se={lr.stderr:.4f}) "
                  f"— {len(r_values)} points, interpret with caution")

    verdict = "Per-N GTA analysis complete. See scaling preview above."
    print(f"\nVERDICT: {verdict}")

    output = {
        "_meta": {"name": "exp16_per_n_gta", "route": 3, "status": "completed",
                  "verdict": verdict, "N": max(N_values), "M": len(records),
                  "timestamp": "", "wall_time_sec": 0, "parameters": {}, "tags": []},
        "results_by_N": results,
        "N_values": N_values,
        "verdict": verdict,
    }
    out_path = RESULTS_DIR / "exp16_per_n_gta.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
