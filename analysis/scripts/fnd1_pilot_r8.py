#!/usr/bin/env python3
"""
PILOT Phase 5.
N=2000, M=30, CRN.
Tests: chain_slope_no_k2 (TC-mediation), interval probes (N=2000 retest).
Metrics: ppwave eps=5, Schwarzschild eps=0.005.

Author: David Alfyorov
"""

import json
import time
import numpy as np
from scipy.stats import ttest_1samp, wilcoxon

N = 2000
M = 30
MASTER_SEED = 99999
RUN_DIR = "docs/analysis_runs/run_20260326_125411"


def sprinkle(N_target, T, rng):
    pts = []
    while len(pts) < N_target:
        batch = rng.uniform(-T / 2, T / 2, size=(N_target * 8, 4))
        r = np.sqrt(batch[:, 1] ** 2 + batch[:, 2] ** 2 + batch[:, 3] ** 2)
        inside = np.abs(batch[:, 0]) + r < T / 2
        pts.extend(batch[inside].tolist())
    pts = np.array(pts[:N_target])
    return pts[np.argsort(pts[:, 0])]


def causal_flat(pts):
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int16)
    for i in range(n):
        dt = pts[i + 1:, 0] - pts[i, 0]
        dr2 = np.sum((pts[i + 1:, 1:] - pts[i, 1:]) ** 2, axis=1)
        C[i, i + 1:] = (dt ** 2 > dr2).astype(np.int16)
    return C


def causal_ppwave(pts, eps):
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int16)
    for i in range(n):
        dt = pts[i + 1:, 0] - pts[i, 0]
        dx = pts[i + 1:, 1:] - pts[i, 1:]
        dr2 = np.sum(dx ** 2, axis=1)
        xm = (pts[i + 1:, 1] + pts[i, 1]) / 2
        ym = (pts[i + 1:, 2] + pts[i, 2]) / 2
        du = dt + dx[:, 2]
        f = xm ** 2 - ym ** 2
        interval = dt ** 2 - dr2 - eps * f * du ** 2 / 2
        C[i, i + 1:] = (interval > 0).astype(np.int16)
    return C


def causal_schw(pts, eps):
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int16)
    for i in range(n):
        dt = pts[i + 1:, 0] - pts[i, 0]
        dx = pts[i + 1:, 1:] - pts[i, 1:]
        dr2 = np.sum(dx ** 2, axis=1)
        rm = np.sqrt(np.sum(((pts[i + 1:, 1:] + pts[i, 1:]) / 2) ** 2, axis=1))
        phi = -eps / (rm + 0.3)
        interval = (1 + 2 * phi) * dt ** 2 - (1 - 2 * phi) * dr2
        C[i, i + 1:] = (interval > 0).astype(np.int16)
    return C


def gini(values):
    v = np.sort(np.abs(np.asarray(values, dtype=np.float64)))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * v)) / (n * np.sum(v)) - (n + 1) / n)


def compute_obs(C):
    Cf = C.astype(np.float64)
    C2 = C.astype(np.int32) @ C.astype(np.int32)
    C3 = Cf @ Cf @ Cf
    tc = int(C.sum())

    N3 = int(C2.astype(np.float64).sum())
    N4 = int(C3.sum())
    chain_slope_no_k2 = np.log(max(N4, 1)) - np.log(max(N3, 1))

    causal_pairs = np.argwhere(C > 0)
    interval_sizes = C2[causal_pairs[:, 0], causal_pairs[:, 1]]
    rng_local = np.random.default_rng(42)

    # ordering_fraction_variance (fixed n band)
    n_p75 = int(np.percentile(interval_sizes, 75)) if len(interval_sizes) > 10 else 3
    n_lo, n_hi = max(n_p75 - 2, 5), max(n_p75 + 2, 8)
    of_vals = []
    band_mask = (interval_sizes >= n_lo) & (interval_sizes <= n_hi)
    band_pairs = causal_pairs[band_mask]
    if len(band_pairs) > 500:
        band_pairs = band_pairs[rng_local.choice(len(band_pairs), 500, replace=False)]
    for pi in range(len(band_pairs)):
        xi, yi = band_pairs[pi]
        above_x = C[xi, :] > 0
        below_y = C[:, yi] > 0
        ielts = np.where(above_x & below_y)[0]
        ni = len(ielts)
        if ni < 3:
            continue
        C_sub = C[np.ix_(ielts, ielts)]
        rels = int(C_sub.sum())
        f = 2.0 * rels / (ni * (ni - 1))
        of_vals.append(f)
    ofv = float(np.var(of_vals)) if len(of_vals) >= 10 else 0.0

    # interval_dim_scatter (n >= 15)
    dim_vals = []
    large_mask = interval_sizes >= 15
    large_pairs = causal_pairs[large_mask]
    if len(large_pairs) > 500:
        large_pairs = large_pairs[rng_local.choice(len(large_pairs), 500, replace=False)]
    for pi in range(len(large_pairs)):
        xi, yi = large_pairs[pi]
        above_x = C[xi, :] > 0
        below_y = C[:, yi] > 0
        ielts = np.where(above_x & below_y)[0]
        ni = len(ielts)
        if ni < 10:
            continue
        C_sub = C[np.ix_(ielts, ielts)]
        rels = int(C_sub.sum())
        rel_frac = rels / max(ni * (ni - 1) / 2, 1)
        if 0 < rel_frac < 1:
            d_mm = np.log(2) / np.log(1.0 / rel_frac)
            if 0 < d_mm < 20:
                dim_vals.append(d_mm)
    ids = (
        float(np.median(np.abs(np.array(dim_vals) - np.median(dim_vals))))
        if len(dim_vals) >= 10
        else 0.0
    )

    # column_gini_C2 (positive control)
    C2f = Cf @ Cf
    col_norms = np.sqrt(np.sum(C2f ** 2, axis=0))
    cgc2 = gini(col_norms)

    # LVA (positive control)
    link_mask = (C > 0) & (C2 == 0)
    CCT = Cf @ Cf.T
    lva_vals = []
    for x in range(len(C)):
        fl = np.where(link_mask[x, :])[0]
        if len(fl) < 2:
            continue
        kplus = CCT[x, fl]
        mu = kplus.mean()
        if mu > 0:
            lva_vals.append(float(kplus.var() / mu ** 2))
    lva = float(np.mean(lva_vals)) if lva_vals else 0.0

    return {
        "TC": tc,
        "chain_slope_no_k2": chain_slope_no_k2,
        "ordering_fraction_variance": ofv,
        "interval_dim_scatter": ids,
        "column_gini_C2": cgc2,
        "lva": lva,
    }


def main():
    print(f"PILOT r8 -- N={N}, M={M}, 2 metrics")
    print("Testing: chain_slope_no_k2 (TC-mediation), interval probes (N=2000)")
    print()

    obs_keys = [
        "chain_slope_no_k2",
        "ordering_fraction_variance",
        "interval_dim_scatter",
        "column_gini_C2",
        "lva",
    ]

    results = {}
    for metric_name, causal_fn, eps in [
        ("ppwave_quad", causal_ppwave, 5.0),
        ("schwarzschild", causal_schw, 0.005),
    ]:
        print(f"=== {metric_name} (eps={eps}) ===")
        t0 = time.time()
        deltas = {k: [] for k in obs_keys}
        tc_flat_list = []
        tc_curved_list = []
        flat_vals = {k: [] for k in obs_keys}
        curved_vals = {k: [] for k in obs_keys}

        for trial in range(M):
            seed = MASTER_SEED + trial * 1000
            rng = np.random.default_rng(seed)
            pts = sprinkle(N, 1.0, rng)

            C_flat = causal_flat(pts)
            C_curved = causal_fn(pts, eps)

            obs_f = compute_obs(C_flat)
            obs_c = compute_obs(C_curved)

            tc_flat_list.append(obs_f["TC"])
            tc_curved_list.append(obs_c["TC"])

            for k in obs_keys:
                deltas[k].append(obs_c[k] - obs_f[k])
                flat_vals[k].append(obs_f[k])
                curved_vals[k].append(obs_c[k])

            if (trial + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  Trial {trial+1}/{M} ({elapsed:.1f}s)")

        metric_results = {}
        for k in obs_keys:
            d_arr = np.array(deltas[k])
            mean_d = d_arr.mean()
            std_d = d_arr.std(ddof=1)
            cohen_d = mean_d / std_d if std_d > 0 else 0
            _, p_t = ttest_1samp(d_arr, 0)
            try:
                _, p_w = wilcoxon(d_arr)
            except Exception:
                p_w = 1.0
            metric_results[k] = {
                "d": round(float(cohen_d), 3),
                "p_t": round(float(p_t), 8),
                "p_w": round(float(p_w), 8),
                "mean_flat": round(float(np.mean(flat_vals[k])), 6),
                "mean_curved": round(float(np.mean(curved_vals[k])), 6),
            }

        # TC-mediation tests
        tc_deltas = np.array(tc_curved_list) - np.array(tc_flat_list)
        for k in ["chain_slope_no_k2", "ordering_fraction_variance", "interval_dim_scatter"]:
            k_deltas = np.array(deltas[k])
            if np.std(tc_deltas) > 0 and np.std(k_deltas) > 0:
                coeffs = np.polyfit(tc_deltas.astype(float), k_deltas, 1)
                predicted = np.polyval(coeffs, tc_deltas.astype(float))
                residuals = k_deltas - predicted
                resid_mean = residuals.mean()
                resid_std = residuals.std(ddof=1)
                d_tc_r = resid_mean / resid_std if resid_std > 0 else 0
                r2 = (
                    1 - np.var(residuals) / np.var(k_deltas)
                    if np.var(k_deltas) > 0
                    else 0
                )
            else:
                d_tc_r = 0
                r2 = 0
            metric_results[f"{k}_tc_mediation"] = {
                "d_tc_resid": round(float(d_tc_r), 3),
                "r2_tc": round(float(r2), 3),
            }

        results[metric_name] = metric_results
        elapsed = time.time() - t0
        print(f"  {metric_name} complete ({elapsed:.1f}s)")
        for k in obs_keys:
            r = metric_results[k]
            tag = "***" if r["p_t"] < 0.001 else ("**" if r["p_t"] < 0.01 else "")
            print(f"  {k:40s} d={r['d']:+.3f}  p={r['p_t']:.8f} {tag}")
        for k in ["chain_slope_no_k2", "ordering_fraction_variance", "interval_dim_scatter"]:
            r_tc = metric_results[f"{k}_tc_mediation"]
            short = k[:25]
            print(
                f"  TC-med {short:25s}: d_resid={r_tc['d_tc_resid']:+.3f}, "
                f"R2(TC)={r_tc['r2_tc']:.3f}"
            )
        print()

    json.dump(results, open(f"{RUN_DIR}/pilot_r8_results.json", "w"), indent=2)
    print("Results saved to pilot_r8_results.json")
    print()
    print("=== VERDICT ===")
    for metric in ["ppwave_quad", "schwarzschild"]:
        r = results[metric]
        chn = r["chain_slope_no_k2"]
        tc_m = r["chain_slope_no_k2_tc_mediation"]
        print(f"{metric}:")
        print(f"  chain_slope_no_k2: d={chn['d']:+.3f}, p={chn['p_t']:.8f}")
        print(
            f"  TC-mediation: d_tc_resid={tc_m['d_tc_resid']:+.3f}, "
            f"R2(TC)={tc_m['r2_tc']:.3f}"
        )
        if abs(tc_m["d_tc_resid"]) < 0.3:
            print("  >>> TC-MEDIATED: d_tc_resid < 0.3 -> KILL")
        else:
            print("  >>> TC-INDEPENDENT: d_tc_resid >= 0.3 -> ADVANCE")


if __name__ == "__main__":
    main()
