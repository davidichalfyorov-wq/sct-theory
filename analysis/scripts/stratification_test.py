#!/usr/bin/env python3
"""
CJ Bridge Formula: Stratification Verification (3 Experiments)

Tests three analytical predictions from the RSY gap closure derivation:

Exp 1: The 10/3 stratification factor
  I_bulk = (3/10) V_4/9!  =>  CJ_strat / CJ_geo ≈ 10/3

Exp 2: Lemma A — stratification → uniform split measure
  f_pop / f_uniform ≈ 0.3 (geometric measure)
  f_strat / f_uniform → 1.0 (uniform measure, if Lemma A holds)

Exp 3: Lemma B — factorization Cov_B(|X|, δY²) = flat_kernel × tidal_variance
  CJ/E² should be ε-independent
  Per-bin factor_b = Cov_B / mean_B(dY²) should be ε-independent

Usage:
  python stratification_test.py --experiment 1
  python stratification_test.py --experiment 2
  python stratification_test.py --experiment 3
  python stratification_test.py --experiment all
"""
import sys, os, time, json, argparse
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, ppwave_exact_preds,
    build_hasse_from_predicate, bulk_mask,
)

# ── Shared utilities ─────────────────────────────────────────────────

def Y_from_graph(par, n):
    """Compute Y = log2(p_down * p_up + 1) with +1 convention."""
    ch = [[] for _ in range(n)]
    for i in range(n):
        if par[i] is not None:
            for j in par[i]:
                ch[int(j)].append(i)
    p_down = np.ones(n, dtype=np.float64)
    p_up = np.ones(n, dtype=np.float64)
    for i in range(n):
        if par[i] is not None and len(par[i]) > 0:
            p_down[i] = np.sum(p_down[list(par[i])]) + 1
    for i in range(n - 1, -1, -1):
        if ch[i]:
            p_up[i] = np.sum(p_up[ch[i]]) + 1
    return np.log2(p_down * p_up + 1)


def compute_depth(par, n):
    """Longest-chain depth from sources."""
    depth = np.zeros(n, dtype=int)
    for i in range(n):
        if par[i] is not None and len(par[i]) > 0:
            depth[i] = int(np.max(depth[list(par[i])])) + 1
    return depth


def make_strata(pts, par, T, n_tau=5, n_rho=3, n_depth=3):
    """Standard 5τ × 3ρ × 3depth = 45 strata."""
    n = len(pts)
    tau_hat = 2.0 * pts[:, 0] / T
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2.0 - np.abs(pts[:, 0])
    rho_hat = np.clip(r / np.maximum(rmax, 1e-12), 0, 0.999999)
    tau_bin = np.clip(np.floor((tau_hat + 1) * (n_tau / 2.0)).astype(int), 0, n_tau - 1)
    rho_bin = np.clip(np.floor(rho_hat * n_rho).astype(int), 0, n_rho - 1)
    depth = compute_depth(par, n)
    max_d = max(int(depth.max()), 1)
    depth_terc = np.clip((depth * n_depth) // (max_d + 1), 0, n_depth - 1)
    return tau_bin * (n_rho * n_depth) + rho_bin * n_depth + depth_terc


def compute_CJ_stratified(X, dY2, strata_m, min_bin=3):
    """Standard CJ = Σ_B w_B Cov_B(|X|, δY²)."""
    total = 0.0
    for b in np.unique(strata_m):
        idx = strata_m == b
        if idx.sum() < min_bin:
            continue
        w = idx.sum() / len(X)
        absX = np.abs(X[idx])
        dy2 = dY2[idx]
        cov = np.mean(absX * dy2) - np.mean(absX) * np.mean(dy2)
        total += w * cov
    return float(total)


def build_crn_data(N, T, eps, seed):
    """Build flat + curved Hasse with CRN, return Y0, YC, pts, par0."""
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)

    # Predicates: pred_fn(pts, i) → boolean mask of length i
    par0, _ = build_hasse_from_predicate(pts, minkowski_preds)

    curved_pred = lambda pts, i, eps=eps: ppwave_exact_preds(pts, i, eps)
    parC, _ = build_hasse_from_predicate(pts, curved_pred)

    n = len(pts)
    Y0 = Y_from_graph(par0, n)
    YC = Y_from_graph(parC, n)
    return pts, par0, Y0, YC


# ── Experiment 1: 10/3 stratification factor ─────────────────────────

def run_experiment_1(N=2000, T=1.0, eps=3.0, M=30, seed_base=9500000):
    """Test prediction: CJ_strat / CJ_geo ≈ 10/3."""
    print("=" * 70)
    print("EXPERIMENT 1: 10/3 Stratification Factor Test")
    print(f"  N={N}, T={T}, eps={eps}, M={M} seeds")
    print("=" * 70)

    results = []
    zeta = 0.15

    for trial in range(M):
        seed = seed_base + trial
        t0 = time.time()

        pts, par0, Y0, YC = build_crn_data(N, T, eps, seed)
        n = len(pts)
        bmask = bulk_mask(pts, T, zeta)
        delta = YC - Y0

        X = Y0[bmask] - np.mean(Y0[bmask])
        dY2 = delta[bmask] ** 2
        absX = np.abs(X)
        strata = make_strata(pts, par0, T)
        strata_m = strata[bmask]

        # 1) CJ_strat — standard
        CJ_strat = compute_CJ_stratified(X, dY2, strata_m)

        # 2) CJ_global — no stratification (single bin)
        CJ_global = float(np.mean(absX * dY2) - np.mean(absX) * np.mean(dY2))

        # 3) CJ_geo — geometric measure weight τ₋⁴ × τ₊⁴
        t_vals = pts[bmask, 0]
        tau_minus = t_vals + T / 2.0  # proper time from p to element
        tau_plus = T / 2.0 - t_vals   # proper time from element to q
        geo_w_raw = tau_minus**4 * tau_plus**4
        geo_w = geo_w_raw / geo_w_raw.sum()  # normalize

        CJ_geo = float(
            np.sum(geo_w * absX * dY2)
            - np.sum(geo_w * absX) * np.sum(geo_w * dY2)
        )

        # Ratios
        R_strat_geo = CJ_strat / CJ_geo if abs(CJ_geo) > 1e-30 else np.nan
        R_strat_global = CJ_strat / CJ_global if abs(CJ_global) > 1e-30 else np.nan

        elapsed = time.time() - t0
        results.append({
            "seed": seed, "CJ_strat": CJ_strat, "CJ_global": CJ_global,
            "CJ_geo": CJ_geo, "R_strat_geo": R_strat_geo,
            "R_strat_global": R_strat_global, "elapsed": elapsed,
        })
        print(f"  [{trial+1}/{M}] seed={seed}: CJ_strat={CJ_strat:.6f}, "
              f"CJ_geo={CJ_geo:.6f}, R={R_strat_geo:.3f}  ({elapsed:.1f}s)")

    # Summary
    R_vals = np.array([r["R_strat_geo"] for r in results if np.isfinite(r["R_strat_geo"])])
    R_mean = np.mean(R_vals)
    R_se = np.std(R_vals, ddof=1) / np.sqrt(len(R_vals))
    R_cv = np.std(R_vals, ddof=1) / R_mean * 100

    # t-test against 10/3
    t_stat, p_val = stats.ttest_1samp(R_vals, 10.0 / 3.0)
    ci_lo = R_mean - 1.96 * R_se
    ci_hi = R_mean + 1.96 * R_se

    # Also test strat/global
    Rg_vals = np.array([r["R_strat_global"] for r in results if np.isfinite(r["R_strat_global"])])

    print("\n" + "─" * 50)
    print(f"R_strat/geo:  {R_mean:.4f} ± {R_se:.4f}  (CV={R_cv:.1f}%)")
    print(f"95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"Prediction 10/3 = {10/3:.4f}")
    print(f"t-test vs 10/3: t={t_stat:.3f}, p={p_val:.4f}")
    print(f"{'✅ PASS' if p_val > 0.05 else '❌ FAIL'}: 10/3 {'IN' if ci_lo <= 10/3 <= ci_hi else 'NOT in'} 95% CI")
    print(f"\nR_strat/global: {np.mean(Rg_vals):.4f} ± {np.std(Rg_vals,ddof=1)/np.sqrt(len(Rg_vals)):.4f}")
    print("─" * 50)

    return {"experiment": 1, "N": N, "eps": eps, "M": M, "results": results,
            "R_mean": R_mean, "R_se": R_se, "R_cv": R_cv,
            "t_stat": t_stat, "p_val": p_val, "ci": [ci_lo, ci_hi],
            "prediction": 10.0/3.0}


# ── Experiment 2: Lemma A (uniform split measure) ───────────────────

def run_experiment_2(N_values=(2000, 3000, 5000), T=1.0, M=20, seed_base=9600000):
    """Test Lemma A: stratification → uniform split measure."""
    print("=" * 70)
    print("EXPERIMENT 2: Lemma A — Stratification → Uniform Split Measure")
    print(f"  N_values={N_values}, M={M} seeds each")
    print("=" * 70)

    f_uniform = 1.0 / 630.0  # B(5,5) = (4!)²/9! = 1/630
    all_results = {}

    for N in N_values:
        print(f"\n  --- N = {N} ---")
        pop_ratios = []
        strat_ratios = []
        equalw_ratios = []

        for trial in range(M):
            seed = seed_base + trial + N  # avoid collision
            rng = np.random.default_rng(seed)
            pts = sprinkle_local_diamond(N, T, rng)
            par0, _ = build_hasse_from_predicate(pts, minkowski_preds)
            n = len(pts)

            # Normalized time position s ∈ [0, 1]
            s = (pts[:, 0] + T / 2.0) / T

            # Test function: f(s) = s⁴(1-s)⁴
            f_vals = s**4 * (1 - s)**4

            # 1) Population mean (= geometric measure average)
            f_pop = np.mean(f_vals)

            # 2) Stratified equal-weight mean (what stratification does)
            strata = make_strata(pts, par0, T)
            bin_means = []
            for b in np.unique(strata):
                idx = strata == b
                if idx.sum() >= 3:
                    bin_means.append(np.mean(f_vals[idx]))
            f_strat_equal = np.mean(bin_means) if bin_means else f_pop

            # 3) CJ-style population-weighted strata mean (w_B = n_B/N)
            f_strat_popw = 0.0
            for b in np.unique(strata):
                idx = strata == b
                if idx.sum() >= 3:
                    f_strat_popw += (idx.sum() / n) * np.mean(f_vals[idx])

            pop_ratios.append(f_pop / f_uniform)
            strat_ratios.append(f_strat_equal / f_uniform)
            equalw_ratios.append(f_strat_popw / f_uniform)

            if trial < 3 or trial == M - 1:
                print(f"    [{trial+1}/{M}] f_pop={f_pop:.6e}, f_strat_eq={f_strat_equal:.6e}, "
                      f"ratio_pop={f_pop/f_uniform:.4f}, ratio_strat={f_strat_equal/f_uniform:.4f}")

        pop_arr = np.array(pop_ratios)
        strat_arr = np.array(strat_ratios)
        eqw_arr = np.array(equalw_ratios)

        print(f"\n  N={N} Summary:")
        print(f"    ratio_pop (geo measure):    {pop_arr.mean():.4f} ± {pop_arr.std(ddof=1)/np.sqrt(M):.4f}  "
              f"(prediction: 0.300)")
        print(f"    ratio_strat (equal weight):  {strat_arr.mean():.4f} ± {strat_arr.std(ddof=1)/np.sqrt(M):.4f}  "
              f"(prediction: → 1.0)")
        print(f"    ratio_strat (pop weight):    {eqw_arr.mean():.4f} ± {eqw_arr.std(ddof=1)/np.sqrt(M):.4f}  "
              f"(should ≈ ratio_pop)")

        # t-tests
        _, p_pop = stats.ttest_1samp(pop_arr, 0.3)
        _, p_strat = stats.ttest_1samp(strat_arr, 1.0)

        print(f"    t-test ratio_pop vs 0.3: p={p_pop:.4f}")
        print(f"    t-test ratio_strat vs 1.0: p={p_strat:.4f}")

        all_results[N] = {
            "ratio_pop_mean": float(pop_arr.mean()),
            "ratio_pop_se": float(pop_arr.std(ddof=1) / np.sqrt(M)),
            "ratio_strat_mean": float(strat_arr.mean()),
            "ratio_strat_se": float(strat_arr.std(ddof=1) / np.sqrt(M)),
            "ratio_popw_mean": float(eqw_arr.mean()),
            "p_pop_vs_03": float(p_pop),
            "p_strat_vs_10": float(p_strat),
        }

    print("\n" + "─" * 50)
    print("N-convergence table:")
    print(f"{'N':>6}  {'ratio_pop':>10}  {'ratio_strat':>12}  {'ratio_popw':>11}")
    for N in N_values:
        r = all_results[N]
        print(f"{N:>6}  {r['ratio_pop_mean']:>10.4f}  {r['ratio_strat_mean']:>12.4f}  "
              f"{r['ratio_popw_mean']:>11.4f}")
    print("─" * 50)

    return {"experiment": 2, "N_values": list(N_values), "M": M,
            "f_uniform": f_uniform, "results": all_results}


# ── Experiment 3: Lemma B (factorization) ────────────────────────────

def run_experiment_3(N=2000, T=1.0, eps_values=(1, 2, 3, 4, 5, 6, 8),
                     M=20, seed_base=9700000):
    """Test Lemma B: factorization of Cov_B into flat kernel × tidal variance."""
    print("=" * 70)
    print("EXPERIMENT 3: Lemma B — Factorization at Leading Weyl Order")
    print(f"  N={N}, T={T}, eps_values={eps_values}, M={M} seeds")
    print("=" * 70)

    zeta = 0.15
    eps_results = {eps: [] for eps in eps_values}

    for trial in range(M):
        seed = seed_base + trial
        t0 = time.time()

        # Build flat Hasse ONCE
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)
        n = len(pts)
        par0, _ = build_hasse_from_predicate(pts, minkowski_preds)
        Y0 = Y_from_graph(par0, n)
        bmask = bulk_mask(pts, T, zeta)
        strata = make_strata(pts, par0, T)
        strata_m = strata[bmask]
        X = Y0[bmask] - np.mean(Y0[bmask])

        for eps in eps_values:
            curved_pred = lambda pts, i, eps=eps: ppwave_exact_preds(pts, i, eps)
            parC, _ = build_hasse_from_predicate(pts, curved_pred)
            YC = Y_from_graph(parC, n)
            delta = YC - Y0
            dY2 = delta[bmask] ** 2

            CJ = compute_CJ_stratified(X, dY2, strata_m)
            E2 = eps**2 / 2.0
            CJ_over_E2 = CJ / E2 if E2 > 0 else np.nan

            # Per-bin decomposition
            bin_factors = {}
            for b in np.unique(strata_m):
                idx = strata_m == b
                if idx.sum() < 5:
                    continue
                absX_b = np.abs(X[idx])
                dy2_b = dY2[idx]
                cov_b = np.mean(absX_b * dy2_b) - np.mean(absX_b) * np.mean(dy2_b)
                mean_dy2_b = np.mean(dy2_b)
                factor_b = cov_b / mean_dy2_b if mean_dy2_b > 1e-30 else np.nan
                bin_factors[int(b)] = {
                    "cov": float(cov_b), "mean_dY2": float(mean_dy2_b),
                    "factor": float(factor_b), "count": int(idx.sum()),
                }

            eps_results[eps].append({
                "seed": seed, "CJ": CJ, "E2": E2, "CJ_over_E2": CJ_over_E2,
                "bin_factors": bin_factors,
            })

        elapsed = time.time() - t0
        if trial < 3 or trial == M - 1:
            row = "  [{:2d}/{:2d}] ".format(trial + 1, M)
            for eps in eps_values[:4]:
                cj = eps_results[eps][-1]["CJ"]
                row += f"ε={eps}:CJ={cj:.5f} "
            print(row + f" ({elapsed:.1f}s)")

    # ── Analysis ──
    print("\n" + "─" * 60)
    print(f"{'eps':>4}  {'E2':>6}  {'mean CJ':>10}  {'SE':>8}  {'CJ/E2':>8}  {'SE':>8}")
    CJ_E2_means = []
    for eps in eps_values:
        cjs = np.array([r["CJ"] for r in eps_results[eps]])
        e2 = eps**2 / 2.0
        cj_e2 = cjs / e2
        CJ_E2_means.append(np.mean(cj_e2))
        print(f"{eps:>4}  {e2:>6.1f}  {np.mean(cjs):>10.6f}  {np.std(cjs,ddof=1)/np.sqrt(M):>8.6f}  "
              f"{np.mean(cj_e2):>8.5f}  {np.std(cj_e2,ddof=1)/np.sqrt(M):>8.5f}")

    CJ_E2_arr = np.array(CJ_E2_means)
    cv_global = np.std(CJ_E2_arr, ddof=1) / np.mean(CJ_E2_arr) * 100

    # Power-law fit: log(CJ) = a + b*log(eps)
    log_eps = np.log(np.array(eps_values, dtype=float))
    log_cj_means = []
    for eps in eps_values:
        cjs = np.array([r["CJ"] for r in eps_results[eps]])
        log_cj_means.append(np.log(np.mean(cjs)))
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_eps, log_cj_means)

    print(f"\nGlobal factorization:")
    print(f"  CV(CJ/E²) across ε: {cv_global:.2f}%  ({'✅ <5%' if cv_global < 5 else '⚠️  ≥5%'})")
    print(f"  Power law: CJ ∝ ε^{slope:.3f} ± {std_err:.3f}  (R²={r_value**2:.4f})")
    print(f"  {'✅ PASS' if abs(slope - 2.0) < 2*std_err else '❌ FAIL'}: slope = {slope:.3f} vs prediction 2.0")

    # Per-bin analysis: find top-10 bins by average population
    all_bins = set()
    for eps in eps_values:
        for r in eps_results[eps]:
            all_bins.update(r["bin_factors"].keys())

    bin_avg_pop = {}
    for b in all_bins:
        counts = []
        for eps in eps_values:
            for r in eps_results[eps]:
                if b in r["bin_factors"]:
                    counts.append(r["bin_factors"][b]["count"])
        bin_avg_pop[b] = np.mean(counts) if counts else 0

    top_bins = sorted(bin_avg_pop, key=bin_avg_pop.get, reverse=True)[:10]

    print(f"\nPer-bin factorization (top-{len(top_bins)} bins by population):")
    print(f"{'bin':>4}  {'pop':>5}  ", end="")
    for eps in eps_values:
        print(f"{'f(ε='+str(eps)+')':>10}", end="")
    print(f"  {'CV%':>6}")

    for b in top_bins:
        factors = []
        print(f"{b:>4}  {bin_avg_pop[b]:>5.0f}  ", end="")
        for eps in eps_values:
            f_vals = [r["bin_factors"][b]["factor"]
                      for r in eps_results[eps]
                      if b in r["bin_factors"] and np.isfinite(r["bin_factors"][b]["factor"])]
            if f_vals:
                f_mean = np.mean(f_vals)
                factors.append(f_mean)
                print(f"{f_mean:>10.5f}", end="")
            else:
                print(f"{'N/A':>10}", end="")
        if len(factors) >= 3:
            cv = np.std(factors, ddof=1) / abs(np.mean(factors)) * 100 if np.mean(factors) != 0 else np.inf
            print(f"  {cv:>6.1f}{'%' if cv < 100 else ''}")
        else:
            print(f"  {'N/A':>6}")

    print("─" * 60)

    return {"experiment": 3, "N": N, "eps_values": list(eps_values), "M": M,
            "cv_global": cv_global, "slope": slope, "slope_se": std_err,
            "CJ_E2_means": [float(x) for x in CJ_E2_arr],
            "per_seed_data": {str(eps): [
                {"CJ": r["CJ"], "CJ_over_E2": r["CJ_over_E2"]}
                for r in eps_results[eps]
            ] for eps in eps_values}}


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CJ Bridge Stratification Tests")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["1", "2", "3", "all"])
    parser.add_argument("--N", type=int, default=2000)
    parser.add_argument("--M", type=int, default=None)
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "fnd1_data")
    os.makedirs(data_dir, exist_ok=True)

    t_start = time.time()

    if args.experiment in ("1", "all"):
        M1 = args.M or 30
        res1 = run_experiment_1(N=args.N, M=M1)
        path1 = os.path.join(data_dir, "stratification_test_exp1.json")
        with open(path1, "w") as f:
            json.dump(res1, f, indent=2, default=str)
        print(f"\nSaved → {path1}")

    if args.experiment in ("2", "all"):
        M2 = args.M or 20
        N_vals = (2000, 3000, 5000) if args.N == 2000 else (args.N,)
        res2 = run_experiment_2(N_values=N_vals, M=M2)
        path2 = os.path.join(data_dir, "stratification_test_exp2.json")
        with open(path2, "w") as f:
            json.dump(res2, f, indent=2, default=str)
        print(f"\nSaved → {path2}")

    if args.experiment in ("3", "all"):
        M3 = args.M or 20
        res3 = run_experiment_3(N=args.N, M=M3)
        path3 = os.path.join(data_dir, "stratification_test_exp3.json")
        with open(path3, "w") as f:
            json.dump(res3, f, indent=2, default=str)
        print(f"\nSaved → {path3}")

    total = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Total runtime: {total:.0f}s ({total/60:.1f} min)")
    print(f"{'='*50}")
