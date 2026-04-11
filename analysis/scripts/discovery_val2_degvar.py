"""
Discovery Run 001 — Validation 2: Degree-Variance Proxy Control
=================================================================
Is the Forman-Ricci residual secretly a proxy for degree variance,
degree skewness, assortativity, or another simple graph statistic?

This is the CRITICAL adversarial test. If the residual is a proxy
for any simple statistic, the candidate is downgraded.

PRE-REGISTERED:
- Compute R^2 of F_residual against 8 simple graph statistics
- If ALL R^2 < 0.50 → "independent"
- If ANY R^2 > 0.80 → "proxy"
- If max R^2 in [0.50, 0.80] → "ambiguous"

Also tests: partial correlation (F_residual | degree_var, degree_skew)

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
M = 60  # more trials for proxy detection (need reliable R^2)

# Use pp-wave eps=10 (strongest signal) for maximum power
GEO = "ppwave_quad"
EPS = 10.0

def partial_correlation(x, y, z):
    """Partial correlation of x and y controlling for z.

    r_xy.z = (r_xy - r_xz * r_yz) / sqrt((1-r_xz^2)(1-r_yz^2))
    """
    if np.std(x) < 1e-15 or np.std(y) < 1e-15 or np.std(z) < 1e-15:
        return 0.0
    r_xy = np.corrcoef(x, y)[0, 1]
    r_xz = np.corrcoef(x, z)[0, 1]
    r_yz = np.corrcoef(y, z)[0, 1]
    denom = np.sqrt(max(1 - r_xz**2, 1e-15) * max(1 - r_yz**2, 1e-15))
    return (r_xy - r_xz * r_yz) / denom


def main():
    print("=" * 70)
    print("VALIDATION 2: Degree-Variance Proxy Control")
    print(f"N={N}, M={M}, {GEO} eps={EPS}")
    print("=" * 70)

    t0 = time.time()
    results = []
    for trial in range(M):
        seed = trial * 1000
        res = crn_trial_full(seed, N, T, GEO, EPS)
        results.append(res)
        if (trial + 1) % 20 == 0:
            elapsed = time.time() - t0
            resids = [r["F_residual"] for r in results]
            print(f"  trial {trial+1}/{M}: resid={np.mean(resids):+.4f} [{elapsed:.1f}s]")

    elapsed = time.time() - t0
    print(f"\n  Computation: {elapsed:.1f}s")

    # Standard analysis
    print("\n  --- Standard Analysis ---")
    verdict = analyze_deltas(results, f"{GEO} eps={EPS}")

    # Detailed proxy analysis
    print("\n  --- Detailed Proxy Analysis (M=60 for reliable R^2) ---")

    residuals = np.array([r["F_residual"] for r in results])
    F_deltas = np.array([r["F_mean_delta"] for r in results])

    stats_to_check = [
        "mean_degree_delta", "degree_var_delta", "degree_std_delta",
        "degree_skew_delta", "degree_kurt_delta", "edge_count_delta",
        "max_degree_delta", "min_degree_delta", "assortativity_delta",
    ]

    print(f"\n  {'statistic':25s} {'R2 vs F_delta':>14s} {'R2 vs residual':>14s} {'corr':>8s}")
    print("  " + "-" * 65)

    proxy_table = {}
    for stat_name in stats_to_check:
        vals = np.array([r.get(stat_name, 0) for r in results], dtype=float)
        if np.std(vals) < 1e-15:
            print(f"  {stat_name:25s} {'const':>14s} {'const':>14s}")
            proxy_table[stat_name] = {"r2_F": 0, "r2_resid": 0, "corr": 0}
            continue

        corr_F = np.corrcoef(F_deltas, vals)[0, 1]
        r2_F = corr_F**2
        corr_R = np.corrcoef(residuals, vals)[0, 1]
        r2_R = corr_R**2

        flag = ""
        if r2_R > 0.80:
            flag = " PROXY!!!"
        elif r2_R > 0.50:
            flag = " AMBIGUOUS"
        elif r2_R > 0.20:
            flag = " notable"

        print(f"  {stat_name:25s} {r2_F:14.3f} {r2_R:14.3f} {corr_R:+8.3f}{flag}")
        proxy_table[stat_name] = {"r2_F": float(r2_F), "r2_resid": float(r2_R), "corr": float(corr_R)}

    # Multiple regression: residual ~ all simple stats
    print("\n  --- Multiple Regression: residual ~ all stats ---")
    X_cols = []
    col_names = []
    for stat_name in stats_to_check:
        vals = np.array([r.get(stat_name, 0) for r in results], dtype=float)
        if np.std(vals) > 1e-15:
            X_cols.append(vals)
            col_names.append(stat_name)

    if X_cols:
        X = np.column_stack(X_cols)
        # Add intercept
        X_with_const = np.column_stack([np.ones(len(residuals)), X])
        # OLS
        try:
            beta, res_ss, rank, sv = np.linalg.lstsq(X_with_const, residuals, rcond=None)
            y_pred = X_with_const @ beta
            ss_res = np.sum((residuals - y_pred)**2)
            ss_tot = np.sum((residuals - np.mean(residuals))**2)
            r2_multi = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            adj_r2 = 1 - (1-r2_multi) * (M-1) / (M-len(beta)-1) if M > len(beta)+1 else r2_multi
            print(f"  R^2 (all stats combined) = {r2_multi:.3f}")
            print(f"  Adjusted R^2             = {adj_r2:.3f}")

            if r2_multi > 0.80:
                print(f"  => PROXY: simple stats explain {r2_multi*100:.0f}% of residual")
            elif r2_multi > 0.50:
                print(f"  => AMBIGUOUS: simple stats explain {r2_multi*100:.0f}% of residual")
            else:
                print(f"  => INDEPENDENT: simple stats explain only {r2_multi*100:.0f}% of residual")
        except Exception as e:
            r2_multi = -1
            adj_r2 = -1
            print(f"  OLS failed: {e}")
    else:
        r2_multi = 0
        adj_r2 = 0

    # Partial correlations: residual with F_mean_delta, controlling for degree_var
    print("\n  --- Partial Correlations ---")
    degvar = np.array([r["degree_var_delta"] for r in results])
    degskew = np.array([r["degree_skew_delta"] for r in results])

    pc_degvar = partial_correlation(residuals, F_deltas, degvar)
    print(f"  r(residual, F_delta | degree_var) = {pc_degvar:.3f}")

    if np.std(degskew) > 1e-15:
        pc_degskew = partial_correlation(residuals, F_deltas, degskew)
        print(f"  r(residual, F_delta | degree_skew) = {pc_degskew:.3f}")
    else:
        pc_degskew = 0.0

    # Save
    output = {
        "geometry": GEO, "eps": EPS, "N": N, "M": M,
        "elapsed_sec": elapsed,
        "verdict": verdict,
        "proxy_table": proxy_table,
        "r2_multiple_regression": float(r2_multi),
        "adj_r2_multiple_regression": float(adj_r2),
        "partial_corr_degvar": float(pc_degvar),
        "partial_corr_degskew": float(pc_degskew),
        "trials": results,
    }

    outpath = os.path.join(OUTDIR, "val2_degvar.json")
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {outpath}")

    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL PROXY VERDICT")
    print("=" * 70)
    max_r2 = max((v["r2_resid"] for v in proxy_table.values()), default=0)
    max_name = max(proxy_table, key=lambda k: proxy_table[k]["r2_resid"]) if proxy_table else "none"

    if r2_multi > 0.80:
        print(f"  PROXY: multiple regression R2={r2_multi:.3f}. Residual is explained by simple stats.")
    elif max_r2 > 0.80:
        print(f"  PROXY: {max_name} alone explains R2={max_r2:.3f} of residual.")
    elif r2_multi > 0.50 or max_r2 > 0.50:
        print(f"  AMBIGUOUS: max single R2={max_r2:.3f} ({max_name}), multi R2={r2_multi:.3f}.")
    else:
        print(f"  INDEPENDENT: max single R2={max_r2:.3f}, multi R2={r2_multi:.3f}.")
        print(f"  Residual carries information BEYOND all simple graph statistics.")


if __name__ == "__main__":
    main()
