"""
FND-1 Route 1: Gate 3 (Finite-Size Scaling) + Gate 4 (Ensemble Stability).

Bypass path: Gate 0 NON-DISCRIMINATING for all families → skip Gate 1/2,
go directly to Gate 3 + Gate 4.

Gate 3: N in {100, 200, 500, 1000, 2000}, M=100 each.
  Fit p(N) = p_inf + c * N^{-alpha}  (3-param, alpha free).
  PASS: |p_inf - p_target| < 0.20.

Gate 4: M in {10, 50, 100, 200}, N=500 fixed.
  Fit R_K ~ 1/M^beta.
  PASS: beta in [0.8, 1.2].

Reference: speculative/FND1_ENSEMBLE_SPEC.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Import building blocks from the Gate 0/1 runner
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_ensemble_runner import (
    sprinkle_diamond,
    compute_interval_cardinalities,
    build_bd_L,
    compute_family_A_eigenvalues,
    compute_family_B_eigenvalues,
    compute_family_C_eigenvalues,
    compute_heat_trace,
    fit_uv_exponent,
    determine_uv_window,
    generate_null_A,
    generate_null_B,
    generate_null_C,
    bootstrap_ci,
    P_TARGETS,
    ZERO_THRESHOLD,
    N_T_GRID,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GATE3_N_VALUES = [100, 200, 500, 1000, 2000]
GATE3_M = 100
GATE4_N = 500
GATE4_M_VALUES = [10, 50, 100, 200]
GATE4_M_MAX = max(GATE4_M_VALUES)
T_DIAMOND = 1.0
MASTER_SEED = 42

FAMILY_NAMES = ["A", "B", "C"]
FAMILY_EIG_FUNCS = {
    "A": lambda L, C: compute_family_A_eigenvalues(L),
    "B": lambda L, C: compute_family_B_eigenvalues(L),
    "C": lambda L, C: compute_family_C_eigenvalues(C),
}
FAMILY_NULL_FUNCS = {
    "A": generate_null_A,
    "B": generate_null_B,
    "C": generate_null_C,
}


# ---------------------------------------------------------------------------
# 3-parameter scaling model for Gate 3
# ---------------------------------------------------------------------------

def scaling_model(N, p_inf, c, alpha):
    """p(N) = p_inf + c * N^{-alpha}."""
    return p_inf + c * np.power(N, -alpha)


def fit_scaling(N_arr, p_arr, p_target):
    """
    Fit p(N) = p_inf + c * N^{-alpha} via curve_fit.

    Returns (p_inf, c, alpha, p_inf_95ci_low, p_inf_95ci_high) or Nones on failure.
    """
    try:
        popt, pcov = curve_fit(
            scaling_model,
            N_arr,
            p_arr,
            p0=[p_target, 1.0, 0.5],
            bounds=([-5.0, -10.0, 0.1], [0.0, 10.0, 3.0]),
            maxfev=10000,
        )
        p_inf, c, alpha = popt
        p_inf_err = np.sqrt(pcov[0, 0]) if pcov[0, 0] >= 0 else np.nan
        ci_lo = p_inf - 1.96 * p_inf_err
        ci_hi = p_inf + 1.96 * p_inf_err
        return p_inf, c, alpha, ci_lo, ci_hi
    except Exception as e:
        print(f"    curve_fit failed: {e}")
        return None, None, None, None, None


# ---------------------------------------------------------------------------
# Gate 3: single-N computation
# ---------------------------------------------------------------------------

def run_gate3_single_N(N: int, M: int, T: float,
                       seed_seq: np.random.SeedSequence):
    """
    Run M sprinklings at size N. Compute p_ens and p_null for each family.

    Returns dict:
      { "A": {"p_ens": ..., "p_null": ..., "ci_low": ..., "ci_high": ...,
              "frob_mean": ...}, ... }
    """
    V = T ** 2 / 2.0
    rho = N / V

    # Spawn M+1 seeds: M for sprinklings, 1 for null/bootstrap
    child_seeds = seed_seq.spawn(M + 1)
    sprinkle_rngs = [np.random.default_rng(s) for s in child_seeds[:M]]
    aux_rng = np.random.default_rng(child_seeds[M])

    # Storage per family
    eig_all = {f: [] for f in FAMILY_NAMES}
    frob_all = {f: [] for f in FAMILY_NAMES}

    # Phase 1: sprinkle and compute eigenvalues
    for i in range(M):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"    Sprinkling {i + 1}/{M} (N={N})...")

        points, C = sprinkle_diamond(N, T, sprinkle_rngs[i])
        n_matrix = compute_interval_cardinalities(C)
        L = build_bd_L(C, n_matrix, rho)

        for fname in FAMILY_NAMES:
            eig = FAMILY_EIG_FUNCS[fname](L, C)
            eig_all[fname].append(eig)
            frob_all[fname].append(
                float(np.sqrt(np.sum(eig ** 2))) if len(eig) > 0 else 0.0
            )

    # Phase 2: heat traces, fits, null models
    results = {}
    for fname in FAMILY_NAMES:
        # Determine UV window
        t_grid, t_min, t_max, lam_max, lam_min = determine_uv_window(
            eig_all[fname]
        )

        # Ensemble heat traces
        K_all = np.zeros((M, len(t_grid)))
        for i in range(M):
            K_all[i] = compute_heat_trace(eig_all[fname][i], t_grid)

        K_ens = np.mean(K_all, axis=0)
        p_ens, _ = fit_uv_exponent(t_grid, K_ens)

        # Bootstrap CI
        ci_low, ci_high = bootstrap_ci(K_all, t_grid, n_boot=500, rng=aux_rng)

        # Null model
        target_frob = float(np.mean(frob_all[fname]))
        null_gen = FAMILY_NULL_FUNCS[fname]
        K_null_all = np.zeros((M, len(t_grid)))
        for j in range(M):
            null_eig = null_gen(N, target_frob, aux_rng)
            K_null_all[j] = compute_heat_trace(null_eig, t_grid)

        K_null_ens = np.mean(K_null_all, axis=0)
        p_null, _ = fit_uv_exponent(t_grid, K_null_ens)

        results[fname] = {
            "p_ens": float(p_ens) if np.isfinite(p_ens) else None,
            "p_null": float(p_null) if np.isfinite(p_null) else None,
            "ci_low": float(ci_low) if np.isfinite(ci_low) else None,
            "ci_high": float(ci_high) if np.isfinite(ci_high) else None,
            "frob_mean": target_frob,
        }

    return results


# ---------------------------------------------------------------------------
# Gate 3: full finite-size scaling
# ---------------------------------------------------------------------------

def run_gate3():
    """Run Gate 3 for all N values and all families."""
    print("=" * 70)
    print("GATE 3: FINITE-SIZE SCALING")
    print(f"N values: {GATE3_N_VALUES}, M={GATE3_M}, T={T_DIAMOND}")
    print("=" * 70)

    t0 = time.perf_counter()

    # Master seed hierarchy
    master_ss = np.random.SeedSequence(MASTER_SEED)
    gate3_ss, _ = master_ss.spawn(2)  # [0] for gate3, [1] reserved for gate4
    n_seeds = gate3_ss.spawn(len(GATE3_N_VALUES))

    # Collect p_ens(N) and p_null(N) for each family
    p_ens_by_family = {f: [] for f in FAMILY_NAMES}
    p_null_by_family = {f: [] for f in FAMILY_NAMES}
    ci_by_family = {f: [] for f in FAMILY_NAMES}
    raw_by_N = {}

    for idx, N in enumerate(GATE3_N_VALUES):
        print(f"\n--- N = {N}, M = {GATE3_M} ---")
        t_n = time.perf_counter()

        single_results = run_gate3_single_N(N, GATE3_M, T_DIAMOND, n_seeds[idx])
        raw_by_N[N] = single_results

        dt = time.perf_counter() - t_n
        print(f"  N={N} done in {dt:.1f}s")

        for fname in FAMILY_NAMES:
            r = single_results[fname]
            p_ens_by_family[fname].append(r["p_ens"])
            p_null_by_family[fname].append(r["p_null"])
            ci_by_family[fname].append((r["ci_low"], r["ci_high"]))
            print(f"    {fname}: p_ens={r['p_ens']:.4f}, p_null={r['p_null']:.4f}, "
                  f"95%CI=[{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
                  if r["p_ens"] is not None else f"    {fname}: FAILED")

    # Fit scaling models
    N_arr = np.array(GATE3_N_VALUES, dtype=float)
    gate3_results = {"parameters": {}, "families": {}}
    gate3_results["parameters"] = {
        "N_values": GATE3_N_VALUES,
        "M": GATE3_M,
        "T": T_DIAMOND,
        "seed": MASTER_SEED,
    }

    print("\n" + "=" * 70)
    print("GATE 3: SCALING FITS")
    print("=" * 70)

    for fname in FAMILY_NAMES:
        p_target = P_TARGETS[fname]
        p_arr = np.array(p_ens_by_family[fname])
        p_null_arr = np.array(p_null_by_family[fname])

        # Check for None values
        valid_ens = np.array([x is not None for x in p_ens_by_family[fname]])
        valid_null = np.array([x is not None for x in p_null_by_family[fname]])

        print(f"\n--- Family {fname} (p_target = {p_target}) ---")
        print(f"  p_ens(N):  {dict(zip(GATE3_N_VALUES, p_ens_by_family[fname]))}")
        print(f"  p_null(N): {dict(zip(GATE3_N_VALUES, p_null_by_family[fname]))}")

        # Fit ensemble scaling
        fit_ens = {"p_inf": None, "c": None, "alpha": None,
                   "p_inf_95ci": [None, None]}
        if np.all(valid_ens) and len(p_arr) >= 3:
            print(f"  Fitting p_ens(N) = p_inf + c * N^{{-alpha}}...")
            p_inf, c, alpha, ci_lo, ci_hi = fit_scaling(N_arr, p_arr, p_target)
            if p_inf is not None:
                fit_ens = {
                    "p_inf": float(p_inf), "c": float(c), "alpha": float(alpha),
                    "p_inf_95ci": [float(ci_lo), float(ci_hi)],
                }
                print(f"    p_inf = {p_inf:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
                print(f"    c = {c:.4f}, alpha = {alpha:.4f}")

        # Fit null scaling (to show N-independence)
        fit_null = {"p_inf": None, "c": None, "alpha": None}
        if np.all(valid_null) and len(p_null_arr) >= 3:
            print(f"  Fitting p_null(N)...")
            p_inf_n, c_n, alpha_n, _, _ = fit_scaling(
                N_arr, p_null_arr, p_target
            )
            if p_inf_n is not None:
                fit_null = {
                    "p_inf": float(p_inf_n), "c": float(c_n),
                    "alpha": float(alpha_n),
                }
                print(f"    p_inf_null = {p_inf_n:.4f}, c = {c_n:.4f}, "
                      f"alpha = {alpha_n:.4f}")

        # Check improvement: |p(N) - p_target| should decrease with N
        errors = [abs(p - p_target) if p is not None else np.inf
                  for p in p_ens_by_family[fname]]
        improving = all(errors[i] >= errors[i + 1] - 0.05
                        for i in range(len(errors) - 1))
        # More lenient: just check if last is better than first
        monotone_improving = (errors[-1] < errors[0] - 0.02
                              if errors[0] < np.inf and errors[-1] < np.inf
                              else False)

        # Verdict
        verdict = "INCONCLUSIVE"
        if fit_ens["p_inf"] is not None:
            gap = abs(fit_ens["p_inf"] - p_target)
            if gap < 0.20:
                verdict = "PASS"
            elif not monotone_improving:
                verdict = "FAIL (no improvement with N)"
            elif gap >= 0.20:
                verdict = f"FAIL (|p_inf - p_target| = {gap:.3f} >= 0.20)"
        else:
            if not monotone_improving:
                verdict = "FAIL (no improvement with N, fit failed)"
            else:
                verdict = "INCONCLUSIVE (fit failed)"

        print(f"  Improving with N: {monotone_improving}")
        print(f"  VERDICT: {verdict}")

        gate3_results["families"][fname] = {
            "p_target": p_target,
            "p_ens_by_N": {str(n): p_ens_by_family[fname][i]
                           for i, n in enumerate(GATE3_N_VALUES)},
            "p_null_by_N": {str(n): p_null_by_family[fname][i]
                            for i, n in enumerate(GATE3_N_VALUES)},
            "ci_by_N": {str(n): list(ci_by_family[fname][i])
                        for i, n in enumerate(GATE3_N_VALUES)},
            "fit_ens": fit_ens,
            "fit_null": fit_null,
            "errors_by_N": {str(n): errors[i]
                            for i, n in enumerate(GATE3_N_VALUES)},
            "improving": monotone_improving,
            "verdict": verdict,
        }

    gate3_results["wall_time_sec"] = time.perf_counter() - t0
    print(f"\nGate 3 wall time: {gate3_results['wall_time_sec']:.1f}s")

    return gate3_results


# ---------------------------------------------------------------------------
# Gate 4: ensemble stability
# ---------------------------------------------------------------------------

def run_gate4():
    """Run Gate 4: R_K vs M at fixed N=500."""
    print("\n" + "=" * 70)
    print("GATE 4: ENSEMBLE STABILITY")
    print(f"N={GATE4_N}, M values: {GATE4_M_VALUES}, T={T_DIAMOND}")
    print("=" * 70)

    t0 = time.perf_counter()

    V = T_DIAMOND ** 2 / 2.0
    rho = GATE4_N / V

    # Seed hierarchy: use child [1] from master
    master_ss = np.random.SeedSequence(MASTER_SEED)
    _, gate4_ss = master_ss.spawn(2)
    child_seeds = gate4_ss.spawn(GATE4_M_MAX + 1)
    sprinkle_rngs = [np.random.default_rng(s) for s in child_seeds[:GATE4_M_MAX]]
    aux_rng = np.random.default_rng(child_seeds[GATE4_M_MAX])

    # Generate all M_max sprinklings
    eig_all = {f: [] for f in FAMILY_NAMES}

    print(f"\nGenerating {GATE4_M_MAX} sprinklings at N={GATE4_N}...")
    for i in range(GATE4_M_MAX):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Sprinkling {i + 1}/{GATE4_M_MAX}...")

        points, C = sprinkle_diamond(GATE4_N, T_DIAMOND, sprinkle_rngs[i])
        n_matrix = compute_interval_cardinalities(C)
        L = build_bd_L(C, n_matrix, rho)

        for fname in FAMILY_NAMES:
            eig = FAMILY_EIG_FUNCS[fname](L, C)
            eig_all[fname].append(eig)

    print("  Sprinklings done.")

    # For each family: determine UV window, compute K_i at t_eval
    gate4_results = {"parameters": {}, "families": {}}
    gate4_results["parameters"] = {
        "N": GATE4_N,
        "M_values": GATE4_M_VALUES,
        "T": T_DIAMOND,
        "seed": MASTER_SEED,
    }

    for fname in FAMILY_NAMES:
        print(f"\n--- Family {fname} ---")

        # UV window from all eigenvalues
        t_grid, t_min, t_max, lam_max, lam_min = determine_uv_window(
            eig_all[fname]
        )

        # Evaluation point: geometric mean of UV window bounds
        t_eval = np.sqrt(t_min * t_max)
        t_eval_arr = np.array([t_eval])
        print(f"  UV window: [{t_min:.6f}, {t_max:.6f}], t_eval={t_eval:.6f}")

        # Compute K_i(t_eval) for all M_max sprinklings
        K_values = np.zeros(GATE4_M_MAX)
        for i in range(GATE4_M_MAX):
            K_i = compute_heat_trace(eig_all[fname][i], t_eval_arr)
            K_values[i] = K_i[0]

        # For each M, compute R_K
        R_K_by_M = {}
        for M in GATE4_M_VALUES:
            K_sub = K_values[:M]
            K_ens = np.mean(K_sub)
            Var_K = np.mean((K_sub - K_ens) ** 2)
            R_K = Var_K / K_ens ** 2 if K_ens > 0 else np.inf
            R_K_by_M[M] = float(R_K)
            print(f"  M={M:4d}: K_ens={K_ens:.4e}, Var_K={Var_K:.4e}, "
                  f"R_K={R_K:.6f}")

        # Fit log(R_K) = -beta * log(M) + const
        M_arr = np.array(GATE4_M_VALUES, dtype=float)
        R_arr = np.array([R_K_by_M[m] for m in GATE4_M_VALUES])

        valid = (R_arr > 0) & np.isfinite(R_arr)
        if np.sum(valid) >= 2:
            log_M = np.log(M_arr[valid])
            log_R = np.log(R_arr[valid])
            result = linregress(log_M, log_R)
            beta = -result.slope
            beta_stderr = result.stderr
            r_squared = result.rvalue ** 2

            print(f"  Fit: beta = {beta:.4f} +/- {beta_stderr:.4f}, "
                  f"R^2 = {r_squared:.4f}")

            # Verdict
            if 0.8 <= beta <= 1.2:
                verdict = "PASS"
            elif beta < 0.8:
                verdict = "FLAG (beta < 0.8, possible methodology issue)"
            elif beta > 1.2:
                verdict = "PASS (super-CLT concentration)"
            else:
                verdict = "FAIL (R_K does not decrease with M)"
        else:
            beta = None
            beta_stderr = None
            r_squared = None
            verdict = "INCONCLUSIVE (insufficient valid R_K values)"
            print(f"  Fit failed: insufficient valid data points.")

        # Check monotonicity: R_K should decrease
        R_vals = [R_K_by_M[m] for m in GATE4_M_VALUES]
        decreasing = all(R_vals[i] >= R_vals[i + 1] * 0.8
                         for i in range(len(R_vals) - 1))
        if not decreasing and verdict != "INCONCLUSIVE":
            verdict = "FAIL (R_K not decreasing with M)"

        print(f"  Decreasing: {decreasing}")
        print(f"  VERDICT: {verdict}")

        gate4_results["families"][fname] = {
            "t_eval": float(t_eval),
            "R_K_by_M": {str(m): R_K_by_M[m] for m in GATE4_M_VALUES},
            "beta": float(beta) if beta is not None else None,
            "beta_stderr": float(beta_stderr) if beta_stderr is not None else None,
            "r_squared": float(r_squared) if r_squared is not None else None,
            "decreasing": decreasing,
            "verdict": verdict,
        }

    gate4_results["wall_time_sec"] = time.perf_counter() - t0
    print(f"\nGate 4 wall time: {gate4_results['wall_time_sec']:.1f}s")

    return gate4_results


# ---------------------------------------------------------------------------
# Overall verdict
# ---------------------------------------------------------------------------

def overall_verdict(gate3, gate4):
    """Determine overall Gate 3 + Gate 4 verdict."""
    g3_verdicts = [gate3["families"][f]["verdict"] for f in FAMILY_NAMES]
    g4_verdicts = [gate4["families"][f]["verdict"] for f in FAMILY_NAMES]

    g3_pass = sum(1 for v in g3_verdicts if v == "PASS")
    g4_pass = sum(1 for v in g4_verdicts if "PASS" in v)
    g3_fail = sum(1 for v in g3_verdicts if "FAIL" in v)

    lines = []
    lines.append(f"Gate 3: {g3_pass}/3 PASS, {g3_fail}/3 FAIL")
    lines.append(f"Gate 4: {g4_pass}/3 PASS")

    if g3_pass >= 1 and g4_pass >= 2:
        lines.append("OVERALL: PROCEED to Gate 5 (curvature sensitivity)")
    elif g3_fail == 3:
        lines.append("OVERALL: STOP — finite-size scaling fails for all families")
    else:
        lines.append("OVERALL: MIXED — partial evidence, Gate 5 optional")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------

def save_results(gate3, gate4, verdict_str, output_path: Path):
    """Save combined results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "gate3": gate3,
        "gate4": gate4,
        "overall_verdict": verdict_str,
    }

    # Clean NaN/inf for JSON
    def _clean(obj):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    data = _clean(data)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    print("FND-1 ENSEMBLE EXPERIMENT: GATE 3 + GATE 4")
    print(f"Bypass path: Gate 0 NON-DISCRIMINATING → direct to Gate 3+4")
    print(f"Master seed: {MASTER_SEED}")
    print()

    gate3 = run_gate3()
    gate4 = run_gate4()

    # Overall verdict
    verdict_str = overall_verdict(gate3, gate4)

    total_time = time.perf_counter() - t_total

    # Summary table
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print("\n--- GATE 3: Finite-Size Scaling ---")
    print(f"{'Family':<8} {'p_target':<10} {'p_inf':<10} "
          f"{'95% CI':<22} {'alpha':<8} {'Improving':<10} {'Verdict'}")
    print("-" * 90)
    for fname in FAMILY_NAMES:
        r = gate3["families"][fname]
        fe = r["fit_ens"]
        p_inf_str = f"{fe['p_inf']:.4f}" if fe['p_inf'] is not None else "N/A"
        ci_str = (f"[{fe['p_inf_95ci'][0]:.4f}, {fe['p_inf_95ci'][1]:.4f}]"
                  if fe['p_inf_95ci'][0] is not None else "N/A")
        alpha_str = f"{fe['alpha']:.4f}" if fe['alpha'] is not None else "N/A"
        print(f"{fname:<8} {r['p_target']:<10.1f} {p_inf_str:<10} "
              f"{ci_str:<22} {alpha_str:<8} {str(r['improving']):<10} "
              f"{r['verdict']}")

    print(f"\n  p_ens(N) data:")
    for fname in FAMILY_NAMES:
        r = gate3["families"][fname]
        vals = "  ".join(f"N={n}: {r['p_ens_by_N'][str(n)]:.4f}"
                         if r['p_ens_by_N'][str(n)] is not None else f"N={n}: N/A"
                         for n in GATE3_N_VALUES)
        print(f"    {fname}: {vals}")

    print(f"\n  p_null(N) data:")
    for fname in FAMILY_NAMES:
        r = gate3["families"][fname]
        vals = "  ".join(f"N={n}: {r['p_null_by_N'][str(n)]:.4f}"
                         if r['p_null_by_N'][str(n)] is not None else f"N={n}: N/A"
                         for n in GATE3_N_VALUES)
        print(f"    {fname}: {vals}")

    print("\n--- GATE 4: Ensemble Stability ---")
    print(f"{'Family':<8} {'beta':<12} {'stderr':<10} {'R^2':<8} "
          f"{'Decreasing':<12} {'Verdict'}")
    print("-" * 70)
    for fname in FAMILY_NAMES:
        r = gate4["families"][fname]
        beta_str = f"{r['beta']:.4f}" if r['beta'] is not None else "N/A"
        se_str = f"{r['beta_stderr']:.4f}" if r['beta_stderr'] is not None else "N/A"
        r2_str = f"{r['r_squared']:.4f}" if r['r_squared'] is not None else "N/A"
        print(f"{fname:<8} {beta_str:<12} {se_str:<10} {r2_str:<8} "
              f"{str(r['decreasing']):<12} {r['verdict']}")

    print(f"\n  R_K(M) data:")
    for fname in FAMILY_NAMES:
        r = gate4["families"][fname]
        vals = "  ".join(f"M={m}: {r['R_K_by_M'][str(m)]:.6f}"
                         for m in GATE4_M_VALUES)
        print(f"    {fname}: {vals}")

    print(f"\n--- OVERALL ---")
    print(verdict_str)
    print(f"\nTotal wall time: {total_time:.1f}s")

    # Save results
    project_root = Path(__file__).resolve().parent.parent.parent
    output_path = (project_root / "speculative" / "numerics" / "ensemble_results"
                   / "gate3_gate4_results.json")
    save_results(gate3, gate4, verdict_str, output_path)


if __name__ == "__main__":
    main()
