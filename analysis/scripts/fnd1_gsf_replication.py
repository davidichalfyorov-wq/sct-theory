#!/usr/bin/env python3
"""
INDEPENDENT REPLICATION of the GSF experiment (FND-1 SJ Vacuum).

Claim to test: "At N=5000 in d=4, the SJ Wightman function eigenvalue
spectrum changes with pp-wave curvature (eps) in a way NOT explained
by the change in total causal pairs (dTC) and links (dTL), even after
quadratic dTC control."

Evidence from original (seed=888, M=40, 6 eps values):
  - d(eps) significant at p < 1e-7 for 5/6 observables
  - R^2 > 0.86 for all regressions

This replication uses:
  - DIFFERENT seed (999 instead of 888)
  - DIFFERENT eps values (-0.3, 0.0, +0.3) instead of (-0.5,-0.3,-0.1,+0.1,+0.3,+0.5)
  - FEWER sprinklings (M=20 instead of 40)
  - COMPLETELY INDEPENDENT code (no imports from fnd1_*.py)

Test: is delta(trace_W) / delta(TC) constant across eps?
      If constant -> original finding REFUTED (TC explains everything)
      If varies   -> original finding REPLICATED (spectrum has info beyond TC)

Run:
  wsl.exe -d Ubuntu bash -c '. ~/.sct_env && source ~/sct-wsl/bin/activate && \
  cd "/mnt/f/Black Mesa Research Facility/Main Facility/Physics department/SCT Theory" && \
  python3 analysis/scripts/fnd1_gsf_replication.py'
"""

import time
import json
import numpy as np
from scipy import stats

# ============================================================
# PARAMETERS — deliberately different from original
# ============================================================
N = 5000          # same N (the claim is about this N)
M = 20            # fewer sprinklings (half of original)
T = 1.0           # diamond half-extent
MASTER_SEED = 999 # different seed
EPS_VALUES = [-0.3, 0.0, 0.3]  # 3 values (original used 6)

OBS_NAMES = ["trace_W", "spectral_gap_ratio", "entropy_spectral",
             "spectral_width", "trace_trunc", "lambda_median"]


# ============================================================
# 1. SPRINKLING: Poisson into 4D causal diamond
# ============================================================
def sprinkle_diamond_4d(n_pts, half_T, rng):
    """
    Uniform Poisson sprinkling into a 4D causal diamond
    {(t,x,y,z) : |t| + sqrt(x^2+y^2+z^2) < half_T}.

    Uses rejection sampling from the bounding box [-T/2, T/2]^4.
    Returns (n_pts, 4) array sorted by time coordinate.
    """
    pts = np.empty((n_pts, 4), dtype=np.float64)
    filled = 0
    while filled < n_pts:
        # Over-sample to reduce loop iterations
        batch_size = max(500, (n_pts - filled) * 12)
        candidates = rng.uniform(-half_T, half_T, size=(batch_size, 4))
        spatial_r = np.sqrt(candidates[:, 1]**2 +
                            candidates[:, 2]**2 +
                            candidates[:, 3]**2)
        inside = np.abs(candidates[:, 0]) + spatial_r < half_T
        valid = candidates[inside]
        take = min(len(valid), n_pts - filled)
        pts[filled:filled + take] = valid[:take]
        filled += take
    # Sort by time for causal ordering
    return pts[np.argsort(pts[:, 0])]


# ============================================================
# 2. CAUSAL MATRIX: flat and pp-wave
# ============================================================
def build_causal_matrix(pts, eps_val):
    """
    Build the upper-triangular causal matrix C[i,j] = 1 iff i < j causally.

    Flat case (eps=0): ds^2 = -dt^2 + dx^2 + dy^2 + dz^2
      Causal iff dt > 0 and dt^2 - (dx^2+dy^2+dz^2) > 0

    PP-wave case (eps != 0): perturbation to the metric.
      The pp-wave adds a correction term involving the transverse profile.
      Using Brinkmann-type coordinates:
        ds^2 = -dt^2 + dx^2 + dy^2 + dz^2 + eps * f(x_mid, y_mid) * (dt+dz)^2 / 2
      where f = cos(pi*x_mid) * cosh(pi*y_mid).

      Two points are causally related if their Minkowski interval exceeds
      the pp-wave correction (maintaining past-to-future ordering).
    """
    n = len(pts)
    t = pts[:, 0]
    x = pts[:, 1]
    y = pts[:, 2]
    z = pts[:, 3]

    # Pairwise differences (j - i): broadcasting [i, j]
    dt = t[None, :] - t[:, None]  # (n, n)
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    dz = z[None, :] - z[:, None]

    dr_sq = dx**2 + dy**2 + dz**2
    mink_interval = dt**2 - dr_sq  # positive = timelike

    if abs(eps_val) < 1e-14:
        # Pure Minkowski
        C = ((mink_interval > 0) & (dt > 0)).astype(np.float64)
    else:
        # PP-wave correction
        x_mid = (x[None, :] + x[:, None]) / 2.0
        y_mid = (y[None, :] + y[:, None]) / 2.0
        profile = np.cos(np.pi * x_mid) * np.cosh(np.pi * y_mid)
        # The /2 factor from du^2 = (dt+dz)^2/2 in Brinkmann coords
        correction = eps_val * profile * (dt + dz)**2 / 2.0
        C = ((mink_interval > correction) & (dt > 0)).astype(np.float64)

    return C


# ============================================================
# 3. LINK MATRIX and SJ WIGHTMAN FUNCTION
# ============================================================
def compute_links_and_sj(C, density):
    """
    From causal matrix C, compute:
    1. Link matrix L[i,j] = 1 iff i->j is a link (no intermediary)
       A link means: i < j, no k with i < k < j.
       Equivalently: C[i,j]=1 and (C @ C)[i,j]=0.
    2. Johnston SJ Wightman function: K_R = a * L
       where a = sqrt(rho) / (2*pi*sqrt(6)) in d=4.
    3. Eigenvalues of i*Delta = i*a*(L - L^T) (Hermitian).

    Returns (positive_eigenvalues, n_links).
    """
    import cupy as cp

    # Interval order matrix: n[i,j] = number of paths of length 2 from i to j
    # = number of intermediary points between i and j
    C_sp = C  # dense is fine for moderate N
    n_mat = C @ C  # (C^2)[i,j] = sum_k C[i,k]*C[k,j]

    # Link matrix: L[i,j] = 1 iff C[i,j]=1 and n_mat[i,j]=0
    L = ((C > 0) & (n_mat == 0)).astype(np.float64)
    n_links = int(np.sum(L))

    # SJ prefactor in d=4: a = sqrt(rho) / (2*pi*sqrt(6))
    a = np.sqrt(density) / (2.0 * np.pi * np.sqrt(6.0))

    # Retarded-minus-advanced: i * a * (L - L^T) is Hermitian
    iDelta = 1j * a * (L - L.T)

    # GPU eigendecomposition (Hermitian -> real eigenvalues)
    iDelta_gpu = cp.asarray(iDelta)
    evals_gpu = cp.linalg.eigvalsh(iDelta_gpu)
    evals = cp.asnumpy(evals_gpu)
    del iDelta_gpu
    cp.get_default_memory_pool().free_all_blocks()

    # Positive eigenvalues = SJ vacuum modes
    pos_evals = evals[evals > 1e-15]

    return pos_evals, n_links


# ============================================================
# 4. OBSERVABLE EXTRACTION
# ============================================================
def compute_observables(pos_evals, n_pts):
    """
    Extract 6 scalar observables from positive SJ eigenvalues.
    These are the SAME observables as the original experiment.
    """
    if len(pos_evals) == 0:
        return {k: 0.0 for k in OBS_NAMES}

    sorted_ev = np.sort(pos_evals)[::-1]  # descending
    trace = float(np.sum(sorted_ev))

    # Spectral gap ratio: largest / second-largest
    gap_ratio = float(sorted_ev[0] / sorted_ev[1]) if (
        len(sorted_ev) >= 2 and sorted_ev[1] > 1e-20) else 0.0

    # Spectral entropy: -sum(p * log(p))
    p = sorted_ev / trace
    entropy = float(-np.sum(p * np.log(p + 1e-300)))

    # Log-spectral width: std of log(lambda)
    log_ev = np.log(sorted_ev[sorted_ev > 1e-20])
    width = float(np.std(log_ev)) if len(log_ev) > 1 else 0.0

    # Truncated trace: sum of top N^{3/4} eigenvalues
    n_trunc = max(1, int(n_pts**0.75))
    trace_trunc = float(np.sum(sorted_ev[:n_trunc]))

    # Median eigenvalue
    median_ev = float(np.median(sorted_ev))

    return {
        "trace_W": trace,
        "spectral_gap_ratio": gap_ratio,
        "entropy_spectral": entropy,
        "spectral_width": width,
        "trace_trunc": trace_trunc,
        "lambda_median": median_ev,
    }


# ============================================================
# 5. MAIN EXPERIMENT
# ============================================================
def main():
    t_start = time.perf_counter()

    # GPU info
    import cupy as cp
    props = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = props['name'].decode()
    print(f"GSF REPLICATION: N={N}, M={M}, seed={MASTER_SEED}, GPU={gpu_name}",
          flush=True)
    print(f"eps values: {EPS_VALUES}", flush=True)
    print(f"This is INDEPENDENT code with DIFFERENT parameters.", flush=True)
    print(f"{'='*70}", flush=True)

    # Volume and density
    V = np.pi * T**4 / 24.0  # volume of 4D diamond with half-extent T
    rho = N / V

    # Generate M seeds (deterministic from MASTER_SEED)
    ss = np.random.SeedSequence(MASTER_SEED)
    child_seeds = [int(cs.generate_state(1)[0]) for cs in ss.spawn(M)]
    print(f"Density rho = {rho:.2f}, V = {V:.6f}", flush=True)
    print(f"First 5 seeds: {child_seeds[:5]}", flush=True)

    # Benchmark one pair
    print(f"\nBenchmarking one (flat, curved) pair...", flush=True)
    rng_bench = np.random.default_rng(child_seeds[0])
    pts_bench = sprinkle_diamond_4d(N, T / 2.0, rng_bench)
    t_bench = time.perf_counter()
    C_flat_b = build_causal_matrix(pts_bench, 0.0)
    C_curv_b = build_causal_matrix(pts_bench, 0.3)
    ev_flat_b, nl_flat_b = compute_links_and_sj(C_flat_b, rho)
    ev_curv_b, nl_curv_b = compute_links_and_sj(C_curv_b, rho)
    bench_time = time.perf_counter() - t_bench
    tc_flat_b = float(np.sum(C_flat_b))
    tc_curv_b = float(np.sum(C_curv_b))
    print(f"  Benchmark: {bench_time:.1f}s per pair", flush=True)
    print(f"  TC_flat={tc_flat_b:.0f}, TC_curved={tc_curv_b:.0f}, "
          f"delta_TC={tc_curv_b - tc_flat_b:.0f}", flush=True)
    print(f"  Links_flat={nl_flat_b}, Links_curved={nl_curv_b}", flush=True)
    print(f"  Pos evals: flat={len(ev_flat_b)}, curved={len(ev_curv_b)}", flush=True)
    total_pairs = M * len(EPS_VALUES)
    est_min = total_pairs * bench_time / 60.0
    print(f"  Estimated total: {total_pairs} pairs x {bench_time:.1f}s = "
          f"{est_min:.0f} min", flush=True)

    # ============================================================
    # RUN THE EXPERIMENT
    # ============================================================
    all_rows = []

    for eps in EPS_VALUES:
        print(f"\n  eps={eps:+.2f}: ", end="", flush=True)
        t_eps_start = time.perf_counter()

        for i, seed in enumerate(child_seeds):
            rng = np.random.default_rng(seed)
            pts = sprinkle_diamond_4d(N, T / 2.0, rng)

            # Flat baseline
            C_flat = build_causal_matrix(pts, 0.0)
            tc_flat = float(np.sum(C_flat))
            ev_flat, nl_flat = compute_links_and_sj(C_flat, rho)
            obs_flat = compute_observables(ev_flat, N)

            # Curved (same points!)
            C_curved = build_causal_matrix(pts, eps)
            tc_curved = float(np.sum(C_curved))
            ev_curved, nl_curved = compute_links_and_sj(C_curved, rho)
            obs_curved = compute_observables(ev_curved, N)

            # Paired differences
            row = {
                "seed": seed,
                "eps": eps,
                "delta_TC": tc_curved - tc_flat,
                "delta_TL": nl_curved - nl_flat,
            }
            for k in OBS_NAMES:
                row[f"delta_{k}"] = obs_curved[k] - obs_flat[k]
            all_rows.append(row)

            if (i + 1) % 5 == 0:
                print(f"{i+1}", end=" ", flush=True)

        elapsed = time.perf_counter() - t_eps_start
        print(f" ({elapsed:.0f}s)", flush=True)

    # ============================================================
    # ANALYSIS
    # ============================================================
    n_total = len(all_rows)
    eps_arr = np.array([r["eps"] for r in all_rows])
    dtc_arr = np.array([r["delta_TC"] for r in all_rows])
    dtl_arr = np.array([r["delta_TL"] for r in all_rows])
    seeds_arr = np.array([r["seed"] for r in all_rows])

    print(f"\n{'='*70}", flush=True)
    print(f"ANALYSIS: {n_total} paired differences "
          f"({M} seeds x {len(EPS_VALUES)} eps)", flush=True)
    print(f"{'='*70}", flush=True)

    # --- A. Basic statistics ---
    print(f"\n  A. BASIC STATS:", flush=True)
    print(f"  {'eps':>6} {'mean_dTC':>12} {'std_dTC':>12} {'mean_dTL':>12}", flush=True)
    for e in EPS_VALUES:
        mask = np.abs(eps_arr - e) < 0.01
        print(f"  {e:+6.2f} {np.mean(dtc_arr[mask]):+12.1f} "
              f"{np.std(dtc_arr[mask]):12.1f} {np.mean(dtl_arr[mask]):+12.1f}",
              flush=True)

    # --- B. KEY TEST: regression with quadratic TC control ---
    print(f"\n  B. KEY REGRESSION: delta_obs = a + b*dTC + b2*dTC^2 + c*dTL + d*eps",
          flush=True)
    print(f"     If d(eps) != 0 -> spectrum carries info beyond TC+TL", flush=True)
    print(f"     n={n_total}, params=5, DOF={n_total-5}", flush=True)
    print(f"\n  {'obs':>22} {'d(eps)':>12} {'t(eps)':>8} {'p(eps)':>12} "
          f"{'R^2':>8} {'sig':>5}", flush=True)
    print(f"  {'-'*67}", flush=True)

    regression_results = {}
    for obs in OBS_NAMES:
        dobs = np.array([r[f"delta_{obs}"] for r in all_rows])

        # Design matrix: [dTC, dTC^2, dTL, eps, intercept]
        X = np.column_stack([
            dtc_arr,
            dtc_arr**2,
            dtl_arr,
            eps_arr,
            np.ones(n_total),
        ])

        beta = np.linalg.lstsq(X, dobs, rcond=None)[0]
        predicted = X @ beta
        residuals = dobs - predicted

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((dobs - np.mean(dobs))**2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 1e-30 else 0.0

        s2 = ss_res / (n_total - 5)
        try:
            cov = np.linalg.inv(X.T @ X) * s2
            se = np.sqrt(np.diag(cov))
            t_eps = beta[3] / se[3] if se[3] > 1e-30 else 0.0
            p_eps = 2 * (1 - stats.t.cdf(abs(t_eps), n_total - 5))
        except np.linalg.LinAlgError:
            se = np.zeros(5)
            t_eps = 0.0
            p_eps = 1.0

        sig = ("***" if p_eps < 0.001
               else ("**" if p_eps < 0.01
                     else ("*" if p_eps < 0.05 else "")))

        print(f"  {obs:>22} {beta[3]:+12.4f} {t_eps:+8.2f} {p_eps:12.4e} "
              f"{r_sq:8.4f} {sig:>5}", flush=True)

        regression_results[obs] = {
            "d_eps": float(beta[3]),
            "se_eps": float(se[3]) if se[3] > 0 else 0.0,
            "t_eps": float(t_eps),
            "p_eps": float(p_eps),
            "R2": float(r_sq),
            "b_dTC": float(beta[0]),
            "b_dTC2": float(beta[1]),
            "c_dTL": float(beta[2]),
        }

    # --- C. PERMUTATION TEST (non-parametric confirmation) ---
    print(f"\n  C. PERMUTATION TEST (5000 shuffles of eps column):", flush=True)
    N_PERM = 5000
    perm_rng = np.random.default_rng(42)
    print(f"  {'obs':>22} {'t_obs':>8} {'p_perm':>10} {'agree':>6}", flush=True)
    print(f"  {'-'*50}", flush=True)

    for obs in OBS_NAMES:
        dobs = np.array([r[f"delta_{obs}"] for r in all_rows])
        t_obs = regression_results[obs]["t_eps"]

        t_perm = np.zeros(N_PERM)
        for p_idx in range(N_PERM):
            eps_shuffled = eps_arr[perm_rng.permutation(n_total)]
            X_p = np.column_stack([dtc_arr, dtc_arr**2, dtl_arr,
                                   eps_shuffled, np.ones(n_total)])
            beta_p = np.linalg.lstsq(X_p, dobs, rcond=None)[0]
            res_p = dobs - X_p @ beta_p
            ss_p = np.sum(res_p**2)
            s2_p = ss_p / (n_total - 5)
            try:
                cov_p = np.linalg.inv(X_p.T @ X_p) * s2_p
                se_p = np.sqrt(cov_p[3, 3])
                t_perm[p_idx] = beta_p[3] / se_p if se_p > 1e-30 else 0.0
            except Exception:
                t_perm[p_idx] = 0.0

        p_perm = float(np.mean(np.abs(t_perm) >= abs(t_obs)))
        p_param = regression_results[obs]["p_eps"]
        agree = "YES" if (p_param < 0.05) == (p_perm < 0.05) else "NO"
        print(f"  {obs:>22} {t_obs:+8.2f} {p_perm:10.4f} {agree:>6}", flush=True)
        regression_results[obs]["p_perm"] = p_perm

    # --- D. RATIO TEST: delta_obs / delta_TC constancy ---
    print(f"\n  D. RATIO TEST: delta_obs / delta_TC by eps", flush=True)
    print(f"     If ratio CONSTANT -> TC explains everything (REFUTES claim)", flush=True)
    print(f"     If ratio VARIES  -> spectrum has extra info (SUPPORTS claim)", flush=True)

    for obs in OBS_NAMES:
        print(f"\n  {obs}:", flush=True)
        ratios_per_eps = []
        for e in EPS_VALUES:
            mask = np.abs(eps_arr - e) < 0.01
            dobs_e = np.array([r[f"delta_{obs}"] for r in all_rows
                               if abs(r["eps"] - e) < 0.01])
            dtc_e = dtc_arr[mask]
            # Only compute ratio where dtc != 0
            valid = np.abs(dtc_e) > 1e-6
            if np.sum(valid) > 0:
                ratios = dobs_e[valid] / dtc_e[valid]
                mean_r = float(np.mean(ratios))
                std_r = float(np.std(ratios))
                ratios_per_eps.append((e, mean_r, std_r))
                print(f"    eps={e:+.2f}: ratio = {mean_r:.8f} +/- {std_r:.8f}",
                      flush=True)

        if len(ratios_per_eps) >= 2:
            means = [r[1] for r in ratios_per_eps]
            cv = np.std(means) / abs(np.mean(means)) if abs(np.mean(means)) > 1e-30 else float('inf')
            # Test monotonicity
            diffs = [means[i+1] - means[i] for i in range(len(means)-1)]
            monotonic = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
            print(f"    CV = {cv:.6f} -> {'CONSTANT' if cv < 0.02 else 'VARIES'}",
                  flush=True)
            if monotonic:
                print(f"    Monotonic: YES (stronger evidence of eps dependence)",
                      flush=True)

    # --- E. VIF (collinearity diagnostic) ---
    print(f"\n  E. COLLINEARITY (VIF):", flush=True)
    X_reg = np.column_stack([dtc_arr, dtc_arr**2, dtl_arr, eps_arr])
    for j, name in enumerate(["dTC", "dTC^2", "dTL", "eps"]):
        others = np.delete(X_reg, j, axis=1)
        others_aug = np.column_stack([others, np.ones(n_total)])
        b_v = np.linalg.lstsq(others_aug, X_reg[:, j], rcond=None)[0]
        r2_v = 1 - (np.sum((X_reg[:, j] - others_aug @ b_v)**2) /
                     np.sum((X_reg[:, j] - np.mean(X_reg[:, j]))**2))
        vif = 1 / (1 - r2_v) if r2_v < 1 else float('inf')
        flag = " <-- HIGH" if vif > 10 else ""
        print(f"    VIF({name}) = {vif:.1f}{flag}", flush=True)

    # --- F. Per-seed slope analysis ---
    print(f"\n  F. PER-SEED SLOPES (mixed-effects shortcut):", flush=True)
    unique_seeds = np.unique(seeds_arr)
    print(f"  {'obs':>22} {'mean_slope':>12} {'se':>10} {'t':>8} {'p':>12}",
          flush=True)
    print(f"  {'-'*66}", flush=True)

    for obs in OBS_NAMES:
        slopes = []
        for s in unique_seeds:
            rows_s = [r for r in all_rows if r["seed"] == s]
            if len(rows_s) < 2:
                continue
            eps_s = np.array([r["eps"] for r in rows_s])
            dobs_s = np.array([r[f"delta_{obs}"] for r in rows_s])
            # Simple linear regression of dobs on eps for this seed
            if np.std(eps_s) > 1e-10:
                sl, _, _, _, _ = stats.linregress(eps_s, dobs_s)
                slopes.append(sl)
        if len(slopes) >= 5:
            slopes = np.array(slopes)
            t_val, p_val = stats.ttest_1samp(slopes, 0.0)
            se_slope = np.std(slopes) / np.sqrt(len(slopes))
            print(f"  {obs:>22} {np.mean(slopes):+12.4f} {se_slope:10.4f} "
                  f"{t_val:+8.2f} {p_val:12.4e}", flush=True)

    # ============================================================
    # VERDICT
    # ============================================================
    print(f"\n{'='*70}", flush=True)
    print(f"REPLICATION VERDICT", flush=True)
    print(f"{'='*70}", flush=True)

    n_sig = sum(1 for r in regression_results.values() if r["p_eps"] < 0.05)
    n_sig_bonf = sum(1 for r in regression_results.values()
                     if r["p_eps"] < 0.05 / 6)
    n_perm_sig = sum(1 for r in regression_results.values()
                     if r.get("p_perm", 1) < 0.05)

    print(f"\n  Observables with p(eps) < 0.05:           {n_sig}/6", flush=True)
    print(f"  Observables with p(eps) < {0.05/6:.4f} (Bonf): {n_sig_bonf}/6",
          flush=True)
    print(f"  Observables with p_perm < 0.05:           {n_perm_sig}/6", flush=True)

    print(f"\n  Original experiment (seed=888, M=40, 6 eps):", flush=True)
    print(f"    5/6 observables significant at p < 1e-7", flush=True)
    print(f"\n  This replication (seed=999, M=20, 3 eps):", flush=True)

    if n_sig_bonf >= 3:
        verdict = "REPLICATED"
        print(f"    {n_sig_bonf}/6 survive Bonferroni correction", flush=True)
        print(f"    -> REPLICATED: SJ spectrum carries info beyond TC+TL", flush=True)
    elif n_sig >= 3:
        verdict = "PARTIALLY_REPLICATED"
        print(f"    {n_sig}/6 nominally significant (uncorrected)", flush=True)
        print(f"    -> PARTIALLY REPLICATED: direction consistent, weaker signal",
              flush=True)
    elif n_sig >= 1:
        verdict = "WEAK_EVIDENCE"
        print(f"    {n_sig}/6 significant (uncorrected)", flush=True)
        print(f"    -> WEAK EVIDENCE: reduced power may explain", flush=True)
    else:
        verdict = "REFUTED"
        print(f"    0/6 significant", flush=True)
        print(f"    -> REFUTED: no evidence of eps effect beyond TC+TL", flush=True)

    total_time = time.perf_counter() - t_start
    print(f"\n  Wall time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    # Save results
    output = {
        "experiment": "GSF_REPLICATION",
        "N": N, "M": M, "master_seed": MASTER_SEED,
        "eps_values": EPS_VALUES,
        "n_total": n_total,
        "per_sprinkling": all_rows,
        "regression_results": regression_results,
        "n_sig": n_sig,
        "n_sig_bonferroni": n_sig_bonf,
        "n_perm_sig": n_perm_sig,
        "verdict": verdict,
        "wall_time": total_time,
    }
    out_path = "speculative/numerics/ensemble_results/sj_gsf_replication.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}", flush=True)
    print(f"  DONE.", flush=True)


if __name__ == "__main__":
    main()
