#!/usr/bin/env python3
"""
FND-1: SJ Vacuum via GSF (Geometry-Switched Fixed-sprinkling).

Fixes vs v1:
  1. Saves PER-SPRINKLING delta arrays (not just group means)
  2. Same 40 seeds for ALL eps values (cross-eps pairing)
  3. Saves delta_TL (link count change) alongside delta_TC
  4. Post-analysis done INSIDE the script on 160 per-sprinkling data points
  5. Verdict is a sanity check, not a conclusion

Run via WSL (GPU):
  wsl.exe -d Ubuntu bash -c '. ~/.sct_env && source ~/sct-wsl/bin/activate && \
  cd "/mnt/f/Black Mesa Research Facility/Main Facility/Physics department/SCT Theory" && \
  python3 analysis/scripts/fnd1_sj_gsf.py'
"""

import time, json
import numpy as np
from scipy import stats

N = 5000
M = 40
T = 1.0
SEED = 888
EPS_VALUES = [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5]  # both signs: tests symmetry


def sprinkle_4d(N, T, rng):
    pts = np.empty((N, 4))
    count = 0
    half = T / 2.0
    while count < N:
        batch = max(N - count, 500) * 10
        c = rng.uniform(-half, half, size=(batch, 4))
        r = np.sqrt(c[:, 1]**2 + c[:, 2]**2 + c[:, 3]**2)
        valid = c[np.abs(c[:, 0]) + r < half]
        n_take = min(len(valid), N - count)
        pts[count:count + n_take] = valid[:n_take]
        count += n_take
    return pts[np.argsort(pts[:, 0])]


def causal_matrix(pts, eps):
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[None, :] - t[:, None]
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    dz = z[None, :] - z[:, None]
    dr2 = dx**2 + dy**2 + dz**2
    mink = dt**2 - dr2
    if abs(eps) > 1e-12:
        xm = (x[None, :] + x[:, None]) / 2
        ym = (y[None, :] + y[:, None]) / 2
        f = np.cos(np.pi * xm) * np.cosh(np.pi * ym)
        corr = eps * f * (dt + dz)**2 / 2.0  # /2 from du²=(dt+dz)²/2 in Brinkmann coords
        return ((mink > corr) & (dt > 0)).astype(np.float64)
    return ((mink > 0) & (dt > 0)).astype(np.float64)


def sj_eigenvalues(C, rho):
    import scipy.sparse as sp
    n_mat = (sp.csr_matrix(C) @ sp.csr_matrix(C)).toarray()
    past = C.T
    n_past = n_mat.T
    link_lower = ((past > 0) & (n_past == 0)).astype(np.float64)
    L = link_lower.T
    a = np.sqrt(rho) / (2.0 * np.pi * np.sqrt(6.0))
    iDelta = 1j * a * (L - L.T)

    import cupy as cp
    evals = cp.asnumpy(cp.linalg.eigvalsh(cp.asarray(iDelta)))
    cp.get_default_memory_pool().free_all_blocks()

    pos = evals[evals > 1e-15]
    n_links = int(np.sum((past > 0) & (n_past == 0)))
    return pos, n_links


def extract_obs(pos_evals, N_pts):
    if len(pos_evals) == 0:
        return {k: 0.0 for k in ["trace_W", "spectral_gap_ratio",
                "entropy_spectral", "spectral_width", "trace_trunc", "lambda_median"]}
    sp = np.sort(pos_evals)[::-1]
    trace = float(np.sum(sp))
    gap = float(sp[0] / sp[1]) if len(sp) >= 2 and sp[1] > 1e-20 else 0
    p = sp / trace
    entropy = float(-np.sum(p * np.log(p + 1e-300)))
    le = np.log(sp[sp > 1e-20])
    width = float(np.std(le)) if len(le) > 1 else 0
    n_max = max(1, int(N_pts**0.75))
    trace_trunc = float(np.sum(sp[:n_max]))
    lmed = float(np.median(sp))
    return {"trace_W": trace, "spectral_gap_ratio": gap,
            "entropy_spectral": entropy, "spectral_width": width,
            "trace_trunc": trace_trunc, "lambda_median": lmed}


def main():
    t0 = time.perf_counter()
    import cupy as cp
    dev = cp.cuda.runtime.getDeviceProperties(0)
    print(f"FND-1 SJ GSF v2: N={N}, M={M}, GPU={dev['name'].decode()}", flush=True)
    print(f"Fixes: per-sprinkling data, same seeds all eps, delta_TL, in-script analysis",
          flush=True)

    V = np.pi * T**4 / 24.0
    rho = N / V

    # Generate seeds ONCE (same 40 sprinklings for all eps)
    ss = np.random.SeedSequence(SEED)
    seeds = [int(cs.generate_state(1)[0]) for cs in ss.spawn(M)]

    # Benchmark
    rng = np.random.default_rng(seeds[0])
    pts = sprinkle_4d(N, T, rng)
    tb = time.perf_counter()
    C0 = causal_matrix(pts, 0.0)
    C1 = causal_matrix(pts, 0.3)
    ev0, nl0 = sj_eigenvalues(C0, rho)
    ev1, nl1 = sj_eigenvalues(C1, rho)
    bench = time.perf_counter() - tb
    print(f"Benchmark: {bench:.1f}s/pair", flush=True)
    print(f"Est total: {M * len(EPS_VALUES) * bench / 60:.0f} min", flush=True)

    obs_names = ["trace_W", "spectral_gap_ratio", "entropy_spectral",
                 "spectral_width", "trace_trunc", "lambda_median"]

    # Storage: per-sprinkling paired differences
    all_rows = []  # list of dicts: {seed, eps, delta_TC, delta_TL, delta_obs...}

    for eps in EPS_VALUES:
        print(f"\n  eps={eps:+.2f}:", end=" ", flush=True)
        t1 = time.perf_counter()

        for i, s in enumerate(seeds):
            rng = np.random.default_rng(s)
            pts = sprinkle_4d(N, T, rng)

            C_flat = causal_matrix(pts, 0.0)
            C_curved = causal_matrix(pts, eps)

            tc_flat = float(np.sum(C_flat))
            tc_curved = float(np.sum(C_curved))

            ev_flat, nl_flat = sj_eigenvalues(C_flat, rho)
            ev_curved, nl_curved = sj_eigenvalues(C_curved, rho)

            obs_flat = extract_obs(ev_flat, N)
            obs_curved = extract_obs(ev_curved, N)

            row = {"seed": s, "eps": eps,
                   "delta_TC": tc_curved - tc_flat,
                   "delta_TL": nl_curved - nl_flat}
            for k in obs_names:
                row[f"delta_{k}"] = obs_curved[k] - obs_flat[k]
            all_rows.append(row)

            if (i + 1) % 10 == 0:
                print(f"{i+1}", end=" ", flush=True)

        print(f" ({time.perf_counter()-t1:.0f}s)", flush=True)

    # ==========================================================
    # IN-SCRIPT ANALYSIS (on all M*len(EPS) = 160 per-sprinkling rows)
    # ==========================================================

    n_total = len(all_rows)
    eps_a = np.array([r["eps"] for r in all_rows])
    dtc_a = np.array([r["delta_TC"] for r in all_rows])
    dtl_a = np.array([r["delta_TL"] for r in all_rows])

    print(f"\n{'='*70}", flush=True)
    print(f"ANALYSIS: {n_total} per-sprinkling paired differences", flush=True)
    print(f"{'='*70}", flush=True)

    # Sanity: paired t-test (is delta != 0?)
    print(f"\n  SANITY CHECK: paired t-test (delta != 0, trivially expected)", flush=True)
    print(f"  {'obs':>20} {'mean_delta':>12} {'t':>8} {'p':>10}", flush=True)
    for obs in obs_names:
        d = np.array([r[f"delta_{obs}"] for r in all_rows])
        t_stat, p_val = stats.ttest_1samp(d, 0.0)
        print(f"  {obs:>20} {np.mean(d):+12.4f} {t_stat:+8.1f} {p_val:10.2e}", flush=True)

    # KEY TEST: regression delta_obs = a + b*dTC + b2*dTC^2 + c*dTL + d*eps
    # dTC^2 catches nonlinear TC confound. If d significant -> info beyond TC+TL.
    print(f"\n  KEY TEST: delta_obs = a + b*dTC + b2*dTC^2 + c*dTL + d*eps", flush=True)
    print(f"  d != 0 means: SJ spectrum carries info beyond pair/link counting", flush=True)
    print(f"  dTC^2 included to catch nonlinear TC confound", flush=True)
    print(f"  n={n_total}, params=5, DOF={n_total-5}", flush=True)
    print(f"\n  {'obs':>20} {'b(dTC)':>10} {'b2(dTC2)':>10} {'d(eps)':>10} "
          f"{'t(eps)':>8} {'p(eps)':>10} {'R^2':>6} {'sig':>4}", flush=True)

    dtc2_a = dtc_a**2  # quadratic TC to catch nonlinear TC confound

    results = {}
    for obs in obs_names:
        dobs = np.array([r[f"delta_{obs}"] for r in all_rows])
        X = np.column_stack([dtc_a, dtc2_a, dtl_a, eps_a, np.ones(n_total)])

        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X, dobs, rcond=None)
        except Exception:
            continue

        pred = X @ beta
        resid = dobs - pred
        n_params = X.shape[1]  # 5: dTC, dTC^2, dTL, eps, intercept
        ss_res = np.sum(resid**2)
        ss_tot = np.sum((dobs - np.mean(dobs))**2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Standard errors via (X^T X)^{-1} * s^2
        s2 = ss_res / (n_total - n_params)
        try:
            cov = np.linalg.inv(X.T @ X) * s2
            se = np.sqrt(np.diag(cov))
            # beta indices: [0]=dTC, [1]=dTC^2, [2]=dTL, [3]=eps, [4]=intercept
            t_eps = beta[3] / se[3] if se[3] > 0 else 0
            p_eps = 2 * (1 - stats.t.cdf(abs(t_eps), n_total - n_params))
        except Exception:
            t_eps, p_eps, se = 0, 1, np.zeros(n_params)

        sig = "***" if p_eps < 0.001 else ("**" if p_eps < 0.01 else ("*" if p_eps < 0.05 else ""))
        print(f"  {obs:>20} {beta[0]:+10.4e} {beta[1]:+10.4e} {beta[3]:+10.4f} "
              f"{t_eps:+8.2f} {p_eps:10.2e} {r_sq:6.4f} {sig:>4}", flush=True)

        results[obs] = {"b_dTC": float(beta[0]), "b_dTC2": float(beta[1]),
                        "c_dTL": float(beta[2]), "d_eps": float(beta[3]),
                        "t_eps": float(t_eps), "p_eps": float(p_eps), "R2": float(r_sq)}

    # Cross-eps consistency: same seeds across eps -> can check per-seed slopes
    print(f"\n  CROSS-EPS: per-seed delta(obs) vs eps (40 mini-regressions)", flush=True)
    print(f"  {'obs':>20} {'mean_slope':>12} {'frac_sig':>10}", flush=True)

    for obs in obs_names:
        slopes = []
        n_sig = 0
        for s in seeds:
            rows_s = [r for r in all_rows if r["seed"] == s]
            if len(rows_s) < 3:
                continue
            eps_s = np.array([r["eps"] for r in rows_s])
            dobs_s = np.array([r[f"delta_{obs}"] for r in rows_s])
            sl, _, r_val, p_val, _ = stats.linregress(eps_s, dobs_s)
            slopes.append(sl)
            if p_val < 0.05:
                n_sig += 1
        if slopes:
            print(f"  {obs:>20} {np.mean(slopes):+12.4f} {n_sig/len(slopes):10.1%}", flush=True)

    # VERDICT (sanity check, not conclusion)
    print(f"\n{'='*70}", flush=True)
    print("VERDICT (evidence summary, not conclusion)", flush=True)
    print(f"{'='*70}", flush=True)

    # VIF diagnostic
    X_reg = np.column_stack([dtc_a, dtc2_a, dtl_a, eps_a])
    print(f"\n  COLLINEARITY DIAGNOSTIC (VIF > 10 = problematic):", flush=True)
    for j, name in enumerate(["delta_TC", "delta_TC^2", "delta_TL", "eps"]):
        others = np.delete(X_reg, j, axis=1)
        others_c = np.column_stack([others, np.ones(n_total)])
        beta_v = np.linalg.lstsq(others_c, X_reg[:, j], rcond=None)[0]
        r2_v = 1 - np.sum((X_reg[:, j] - others_c @ beta_v)**2) / np.sum((X_reg[:, j] - np.mean(X_reg[:, j]))**2)
        vif = 1 / (1 - r2_v) if r2_v < 1 else float('inf')
        print(f"    VIF({name}) = {vif:.1f}", flush=True)

    n_sig_eps = sum(1 for r in results.values() if r.get("p_eps", 1) < 0.05)
    n_sig_bonf = sum(1 for r in results.values() if r.get("p_eps", 1) < 0.05 / 6)
    print(f"\n  {n_sig_eps}/6 observables: p(eps) < 0.05 (uncorrected)", flush=True)
    print(f"  {n_sig_bonf}/6 observables: p(eps) < {0.05/6:.4f} (Bonferroni)", flush=True)
    print(f"  after controlling for delta_TC and delta_TL.", flush=True)

    if n_sig_bonf >= 3:
        print(f"  -> Evidence consistent with SJ spectrum encoding info beyond TC+TL", flush=True)
    elif n_sig_eps >= 1:
        print(f"  -> Weak/exploratory evidence, needs confirmation at larger N", flush=True)
    else:
        print(f"  -> No evidence at this N. Note: high collinearity (VIF) limits power;", flush=True)
        print(f"     null result does not exclude moderate effects.", flush=True)

    total_time = time.perf_counter() - t0
    print(f"\n  Wall time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    # Save EVERYTHING (per-sprinkling data for external analysis)
    output = {
        "N": N, "M": M, "eps_values": EPS_VALUES, "n_total": n_total,
        "per_sprinkling": all_rows,
        "regression_results": results,
        "n_sig_eps": n_sig_eps,
        "wall_time": total_time,
    }
    with open("speculative/numerics/ensemble_results/sj_gsf.json", "w") as f:
        json.dump(output, f)
    print("Saved sj_gsf.json. Done.", flush=True)


if __name__ == "__main__":
    main()
