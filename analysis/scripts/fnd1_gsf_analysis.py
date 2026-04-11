"""
FND-1: GSF Post-Hoc Analysis.

Reads sj_gsf.json (per-sprinkling data from the GSF experiment)
and performs all diagnostics that the in-script analysis missed:

1. Permutation test (10,000 shuffles) — non-parametric p-value for d(eps)
2. Scatter plots delta_obs vs eps colored by seed
3. Residual diagnostics (vs fitted, vs eps, normality)
4. Heteroscedasticity test (Breusch-Pagan)
5. Confidence intervals for d(eps)
6. Quadratic eps term (d2*eps^2) — even vs odd curvature response
7. One-sample t-test on 40 per-seed slopes (mixed-effects shortcut)
8. Outlier detection (Cook's distance)
9. Ratio delta_obs/delta_TC constancy across eps

Run after GSF completes:
  python analysis/scripts/fnd1_gsf_analysis.py
"""

import json, sys
import numpy as np
from scipy import stats
from pathlib import Path

RESULTS = Path("speculative/numerics/ensemble_results/sj_gsf.json")
OBS_NAMES = ["trace_W", "spectral_gap_ratio", "entropy_spectral",
             "spectral_width", "trace_trunc", "lambda_median"]


def load_data():
    with open(RESULTS) as f:
        data = json.load(f)
    rows = data["per_sprinkling"]
    n = len(rows)
    eps = np.array([r["eps"] for r in rows])
    dtc = np.array([r["delta_TC"] for r in rows])
    dtl = np.array([r["delta_TL"] for r in rows])
    seeds = np.array([r["seed"] for r in rows])
    dobs = {k: np.array([r[f"delta_{k}"] for r in rows]) for k in OBS_NAMES}
    return n, eps, dtc, dtl, seeds, dobs


def regression(eps, dtc, dobs, dtl):
    """OLS: dobs = a + b*dTC + b2*dTC^2 + c*dTL + d*eps. Returns beta, se, resid."""
    n = len(eps)
    X = np.column_stack([dtc, dtc**2, dtl, eps, np.ones(n)])
    beta = np.linalg.lstsq(X, dobs, rcond=None)[0]
    pred = X @ beta
    resid = dobs - pred
    ss_res = np.sum(resid**2)
    s2 = ss_res / (n - 5)
    try:
        cov = np.linalg.inv(X.T @ X) * s2
        se = np.sqrt(np.diag(cov))
    except Exception:
        se = np.zeros(5)
    return beta, se, resid, X


def main():
    if not RESULTS.exists():
        print(f"File not found: {RESULTS}")
        return

    n, eps, dtc, dtl, seeds, dobs = load_data()
    unique_seeds = np.unique(seeds)
    unique_eps = np.sort(np.unique(eps))
    print(f"GSF Post-Hoc Analysis: {n} rows, {len(unique_seeds)} seeds, "
          f"{len(unique_eps)} eps values", flush=True)

    for obs in OBS_NAMES:
        print(f"\n{'='*70}", flush=True)
        print(f"OBSERVABLE: {obs}", flush=True)
        print(f"{'='*70}", flush=True)

        d = dobs[obs]
        beta, se, resid, X = regression(eps, dtc, d, dtl)
        d_eps = beta[3]
        se_eps = se[3]
        t_eps = d_eps / se_eps if se_eps > 0 else 0
        p_eps = 2 * (1 - stats.t.cdf(abs(t_eps), n - 5))
        ci_lo = d_eps - 1.96 * se_eps
        ci_hi = d_eps + 1.96 * se_eps

        # --- 1. Permutation test ---
        n_perm = 10000
        rng = np.random.default_rng(42)
        t_null = np.zeros(n_perm)
        for i in range(n_perm):
            eps_shuf = eps[rng.permutation(n)]
            X_p = np.column_stack([dtc, dtc**2, dtl, eps_shuf, np.ones(n)])
            b_p = np.linalg.lstsq(X_p, d, rcond=None)[0]
            res_p = d - X_p @ b_p
            ss_p = np.sum(res_p**2)
            s2_p = ss_p / (n - 5)
            try:
                cov_p = np.linalg.inv(X_p.T @ X_p) * s2_p
                se_p = np.sqrt(cov_p[3, 3])
                t_null[i] = b_p[3] / se_p if se_p > 0 else 0
            except Exception:
                t_null[i] = 0
        p_perm = float(np.mean(np.abs(t_null) >= abs(t_eps)))
        print(f"\n  1. PERMUTATION TEST (n={n_perm}):", flush=True)
        print(f"     d(eps) = {d_eps:+.4f}, t = {t_eps:+.2f}", flush=True)
        print(f"     Parametric p = {p_eps:.4e}", flush=True)
        print(f"     Permutation p = {p_perm:.4f}", flush=True)
        print(f"     {'AGREE' if (p_eps<0.05) == (p_perm<0.05) else 'DISAGREE'}", flush=True)

        # --- 2. Confidence interval ---
        print(f"\n  2. CONFIDENCE INTERVAL:", flush=True)
        print(f"     d(eps) = {d_eps:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}]", flush=True)

        # --- 3. Residual diagnostics ---
        sw_stat, sw_p = stats.shapiro(resid) if len(resid) <= 5000 else (0, 0)
        r_resid_eps, p_resid_eps = stats.pearsonr(eps, resid)
        r_resid_fit, p_resid_fit = stats.pearsonr(X @ beta, resid)
        print(f"\n  3. RESIDUAL DIAGNOSTICS:", flush=True)
        print(f"     Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4e} "
              f"({'normal' if sw_p > 0.05 else 'NON-NORMAL'})", flush=True)
        print(f"     r(resid, eps) = {r_resid_eps:+.4f} (p={p_resid_eps:.4e}) "
              f"— should be ~0", flush=True)
        print(f"     r(resid, fitted) = {r_resid_fit:+.4f} (p={p_resid_fit:.4e}) "
              f"— should be ~0", flush=True)

        # --- 4. Heteroscedasticity (Breusch-Pagan) ---
        resid2 = resid**2
        bp_r, bp_p = stats.pearsonr(eps**2, resid2)
        print(f"\n  4. HETEROSCEDASTICITY (Breusch-Pagan proxy):", flush=True)
        print(f"     r(eps^2, resid^2) = {bp_r:+.4f} (p={bp_p:.4e})", flush=True)
        if bp_p < 0.05:
            print(f"     WARNING: heteroscedastic. OLS SE may be wrong.", flush=True)

        # --- 5. Outlier detection (Cook's distance) ---
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        h = np.diag(H)
        mse = np.sum(resid**2) / (n - 5)
        cooks = resid**2 * h / (5 * mse * (1 - h)**2 + 1e-300)
        n_outliers = int(np.sum(cooks > 4 / n))
        max_cook = float(np.max(cooks))
        print(f"\n  5. OUTLIERS (Cook's distance):", flush=True)
        print(f"     Max Cook's d = {max_cook:.4f} (threshold = {4/n:.4f})", flush=True)
        print(f"     Outliers: {n_outliers}/{n}", flush=True)

        # --- 6. Quadratic eps term ---
        X_q = np.column_stack([dtc, dtc**2, dtl, eps, eps**2, np.ones(n)])
        beta_q = np.linalg.lstsq(X_q, d, rcond=None)[0]
        res_q = d - X_q @ beta_q
        ss_q = np.sum(res_q**2)
        s2_q = ss_q / (n - 6)
        try:
            cov_q = np.linalg.inv(X_q.T @ X_q) * s2_q
            se_q = np.sqrt(np.diag(cov_q))
            t_lin = beta_q[3] / se_q[3] if se_q[3] > 0 else 0
            t_quad = beta_q[4] / se_q[4] if se_q[4] > 0 else 0
            p_lin = 2 * (1 - stats.t.cdf(abs(t_lin), n - 6))
            p_quad = 2 * (1 - stats.t.cdf(abs(t_quad), n - 6))
        except Exception:
            t_lin, t_quad, p_lin, p_quad = 0, 0, 1, 1
        print(f"\n  6. QUADRATIC EPS (even vs odd curvature):", flush=True)
        print(f"     d1(eps):  {beta_q[3]:+.4f} (t={t_lin:+.2f}, p={p_lin:.4e})", flush=True)
        print(f"     d2(eps^2): {beta_q[4]:+.4f} (t={t_quad:+.2f}, p={p_quad:.4e})", flush=True)
        if p_quad < 0.05 and p_lin > 0.05:
            print(f"     -> EVEN response (curvature magnitude, not sign)", flush=True)
        elif p_lin < 0.05 and p_quad > 0.05:
            print(f"     -> ODD response (curvature sign matters)", flush=True)
        elif p_lin < 0.05 and p_quad < 0.05:
            print(f"     -> BOTH linear and quadratic", flush=True)

        # --- 7. Per-seed slope t-test ---
        slopes = []
        for s in unique_seeds:
            mask = seeds == s
            if np.sum(mask) >= 3:
                sl, _, _, _, _ = stats.linregress(eps[mask], d[mask])
                slopes.append(sl)
        if len(slopes) >= 5:
            t_slope, p_slope = stats.ttest_1samp(slopes, 0.0)
            print(f"\n  7. PER-SEED SLOPES (mixed-effects shortcut):", flush=True)
            print(f"     Mean slope = {np.mean(slopes):+.4f} +/- {np.std(slopes)/np.sqrt(len(slopes)):.4f}",
                  flush=True)
            print(f"     t = {t_slope:+.2f}, p = {p_slope:.4e} (n_seeds={len(slopes)})", flush=True)

        # --- 8. Ratio constancy ---
        print(f"\n  8. RATIO delta_obs / delta_TC by eps:", flush=True)
        ratios_by_eps = []
        for e in unique_eps:
            mask = np.abs(eps - e) < 0.01
            if np.sum(mask) > 0:
                r_vals = d[mask] / dtc[mask]
                r_vals = r_vals[np.isfinite(r_vals)]
                if len(r_vals) > 0:
                    mean_r = float(np.mean(r_vals))
                    ratios_by_eps.append(mean_r)
                    print(f"     eps={e:+.2f}: ratio = {mean_r:.6f}", flush=True)
        if len(ratios_by_eps) >= 3:
            cv = np.std(ratios_by_eps) / abs(np.mean(ratios_by_eps)) if abs(np.mean(ratios_by_eps)) > 1e-20 else float('inf')
            print(f"     CV = {cv:.4f} ({'CONSTANT' if cv < 0.05 else 'VARIES'})", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
