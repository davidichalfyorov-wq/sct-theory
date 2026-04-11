#!/usr/bin/env python3
"""
PILOT (FAST version).
N=2000, M=20 CRN, ppwave eps=5 + Schwarzschild eps=0.005.

Focus: chain_slope_no_k2 + interval probes + column_gini_C2 control.
Removes slow path DP (will test path_excess_skew at smaller N or separately).

Author: David Alfyorov
"""

import json, time
import numpy as np
from scipy.stats import wilcoxon
from pathlib import Path

N = 2000
M = 20
MASTER_SEED = 98765
RUN_DIR = Path("docs/analysis_runs/run_20260326_125402")

def sprinkle(N_target, T, rng):
    pts = []
    while len(pts) < N_target:
        batch = rng.uniform(-T/2, T/2, size=(N_target*8, 4))
        r = np.sqrt(batch[:,1]**2 + batch[:,2]**2 + batch[:,3]**2)
        inside = np.abs(batch[:,0]) + r < T/2
        pts.extend(batch[inside].tolist())
    pts = np.array(pts[:N_target])
    return pts[np.argsort(pts[:,0])]

def causal_flat(pts):
    n = len(pts)
    C = np.zeros((n,n), dtype=np.int8)
    for i in range(n):
        dt = pts[i+1:,0] - pts[i,0]
        dr2 = np.sum((pts[i+1:,1:] - pts[i,1:])**2, axis=1)
        C[i, i+1:] = (dt**2 > dr2).astype(np.int8)
    return C

def causal_ppwave(pts, eps):
    n = len(pts)
    C = np.zeros((n,n), dtype=np.int8)
    for i in range(n):
        dt = pts[i+1:,0] - pts[i,0]
        dx = pts[i+1:,1:] - pts[i,1:]
        dr2 = np.sum(dx**2, axis=1)
        xm = (pts[i+1:,1] + pts[i,1]) / 2
        ym = (pts[i+1:,2] + pts[i,2]) / 2
        dz = dx[:,2]
        mink = dt**2 - dr2
        corr = eps * (xm**2 - ym**2) * (dt + dz)**2 / 2.0
        C[i, i+1:] = ((mink > corr) & (dt > 0)).astype(np.int8)
    return C

def causal_schw(pts, eps):
    n = len(pts)
    C = np.zeros((n,n), dtype=np.int8)
    for i in range(n):
        dt = pts[i+1:,0] - pts[i,0]
        dr2 = np.sum((pts[i+1:,1:] - pts[i,1:])**2, axis=1)
        xm = (pts[i+1:,1:] + pts[i,1:]) / 2
        rm = np.sqrt(np.sum(xm**2, axis=1)) + 0.3
        phi = -eps / rm
        C[i, i+1:] = (((1+2*phi)*dt**2 > (1-2*phi)*dr2) & (dt > 0)).astype(np.int8)
    return C

def gini(arr):
    a = np.sort(np.abs(arr))
    n = len(a)
    if n == 0 or a.sum() == 0:
        return 0.0
    idx = np.arange(1, n+1)
    return float((2*np.sum(idx*a))/(n*np.sum(a)) - (n+1)/n)

def compute_obs(C, pts):
    """Compute observables using MATRIX OPERATIONS only (no Python loops over N)."""
    n = len(C)
    Cf = C.astype(np.float32)
    # C2, C3, C4 via matrix multiply
    C2 = Cf @ Cf
    C3 = Cf @ C2
    C4 = Cf @ C3

    tc = float(np.sum(C))
    N3 = float(np.sum(C3))
    N4 = float(np.sum(C4))
    csn = np.log(max(N4, 1)) - np.log(max(N3, 1))

    # column_gini_C2
    col_norms = np.sqrt(np.sum(C2**2, axis=0))
    cgc2 = gini(col_norms)

    # Link matrix + degree
    L = ((C == 1) & (C2.astype(np.int32) == 0)).astype(np.float32)
    A = L + L.T
    degrees = np.sum(A, axis=1)
    link_count = float(np.sum(L))
    degree_cv = float(np.std(degrees) / max(np.mean(degrees), 1e-15))
    sum_deg2 = float(np.sum(degrees**2))

    # LVA (vectorized)
    CCT = Cf @ Cf.T
    lva_vals = []
    for x in range(n):
        fl = np.where(L[x,:] > 0.5)[0]
        if len(fl) < 2:
            continue
        kplus = CCT[x, fl]
        mu = kplus.mean()
        if mu > 0:
            lva_vals.append(float(kplus.var() / mu**2))
    lva = float(np.mean(lva_vals)) if lva_vals else 0.0

    # Interval probes (sample up to 300 intervals with n>=15)
    # Find eligible pairs
    C2_int = C2.astype(np.int32)
    rng_local = np.random.default_rng(42)

    t_vals = pts[:,0]
    t_min, t_max = t_vals.min(), t_vals.max()
    t_range = t_max - t_min
    t_lo = t_min + 0.15 * t_range
    t_hi = t_max - 0.15 * t_range

    # Find pairs with C2>=15 and both endpoints in interior
    eligible_i, eligible_j = np.where((C == 1) & (C2_int >= 15))
    interior_mask = (t_vals[eligible_i] >= t_lo) & (t_vals[eligible_i] <= t_hi) & \
                    (t_vals[eligible_j] >= t_lo) & (t_vals[eligible_j] <= t_hi)
    eligible_i = eligible_i[interior_mask]
    eligible_j = eligible_j[interior_mask]

    if len(eligible_i) > 300:
        idx = rng_local.choice(len(eligible_i), 300, replace=False)
        eligible_i = eligible_i[idx]
        eligible_j = eligible_j[idx]

    of_vals = []
    for k in range(len(eligible_i)):
        xi, yi = eligible_i[k], eligible_j[k]
        elems = np.where((C[xi,:] == 1) & (C[:,yi] == 1))[0]
        ni = len(elems)
        if ni < 10:
            continue
        sub = C[np.ix_(elems, elems)]
        rels = int(np.sum(sub))
        f = 2.0 * rels / (ni * (ni-1)) if ni > 1 else 0
        of_vals.append(f)

    ofv = float(np.var(of_vals)) if len(of_vals) >= 10 else 0.0

    # MM dimension scatter
    dim_vals = []
    for f in of_vals:
        if 0 < f < 1:
            d_mm = np.log(2) / np.log(1.0/f)
            if 0 < d_mm < 20:
                dim_vals.append(d_mm)
    ids = float(np.median(np.abs(np.array(dim_vals) - np.median(dim_vals)))) if len(dim_vals) >= 10 else 0.0

    # Layer counts
    layer_edges = np.linspace(t_vals.min(), t_vals.max(), 5)
    layers = np.digitize(t_vals, layer_edges[1:-1])

    return {
        "tc": tc, "chain_slope_no_k2": csn, "column_gini_C2": cgc2,
        "lva": lva, "ofv": ofv, "ids_mad": ids,
        "link_count": link_count, "degree_cv": degree_cv, "sum_deg2": sum_deg2,
        "l0": float(np.sum(layers==0)), "l1": float(np.sum(layers==1)),
        "l2": float(np.sum(layers==2)), "l3": float(np.sum(layers==3)),
        "n_intervals": len(of_vals),
    }

def main():
    print(f"PILOT R8-FAST: N={N}, M={M}")
    t0 = time.time()

    obs_keys = ["chain_slope_no_k2", "column_gini_C2", "lva", "ofv", "ids_mad"]
    baseline_keys = ["tc", "link_count", "degree_cv", "sum_deg2", "l0", "l1", "l2", "l3"]
    results = {}

    for metric_name, causal_fn, eps in [
        ("ppwave_quad", causal_ppwave, 5.0),
        ("schwarzschild", causal_schw, 0.005),
    ]:
        print(f"\n=== {metric_name} (eps={eps}) ===")
        flat_list = []
        curved_list = []

        for trial in range(M):
            seed = MASTER_SEED + trial * 1000 + (100 if metric_name == "ppwave_quad" else 300)
            rng = np.random.default_rng(seed)
            pts = sprinkle(N, 1.0, rng)

            t1 = time.time()
            C_flat = causal_flat(pts)
            C_curved = causal_fn(pts, eps)
            obs_f = compute_obs(C_flat, pts)
            obs_c = compute_obs(C_curved, pts)
            flat_list.append(obs_f)
            curved_list.append(obs_c)

            if (trial+1) % 5 == 0:
                print(f"  Trial {trial+1}/{M} ({time.time()-t1:.1f}s/trial, total {time.time()-t0:.0f}s)")

        metric_res = {}
        for key in obs_keys:
            f_arr = np.array([o[key] for o in flat_list])
            c_arr = np.array([o[key] for o in curved_list])
            deltas = c_arr - f_arr
            d_val = float(np.mean(deltas) / max(np.std(deltas), 1e-15))
            try:
                _, p_val = wilcoxon(deltas)
            except:
                p_val = 1.0

            # TC-mediation test
            tc_f = np.array([o["tc"] for o in flat_list])
            tc_c = np.array([o["tc"] for o in curved_list])
            obs_f_arr = np.array([o[key] for o in flat_list])
            if np.std(tc_f) > 0:
                slope, intercept = np.polyfit(tc_f, obs_f_arr, 1)
                predicted = slope * tc_c + intercept
                residuals = c_arr - predicted
                d_tc_resid = float(np.mean(residuals) / max(np.std(residuals), 1e-15))
            else:
                d_tc_resid = d_val

            # Baseline R² (flat data only)
            y = obs_f_arr
            X = np.column_stack([np.array([o[k] for o in flat_list]) for k in baseline_keys])
            n_obs, p = X.shape
            try:
                X_std = (X - X.mean(0)) / (X.std(0) + 1e-15)
                y_std = (y - y.mean()) / (y.std() + 1e-15) if y.std() > 0 else y - y.mean()
                beta = np.linalg.lstsq(X_std, y_std, rcond=None)[0]
                ss_res = np.sum((y_std - X_std @ beta)**2)
                ss_tot = np.sum(y_std**2)
                r2 = 1 - ss_res / max(ss_tot, 1e-15)
                r2_adj = 1 - (1-r2)*(n_obs-1)/max(n_obs-p-1, 1)
            except:
                r2_adj = 0.0

            status = ""
            if abs(d_val) > 0.5 and p_val < 0.05:
                status = "SIGNAL"
            elif abs(d_val) > 0.3:
                status = "WEAK"
            else:
                status = "NULL"

            tc_st = "TC-MED" if abs(d_tc_resid) < 0.3 else "TC-IND"
            r2_st = "RED" if r2_adj > 0.7 else ("YEL" if r2_adj > 0.5 else "GRN")

            metric_res[key] = {
                "d": round(d_val, 3), "p": round(float(p_val), 8),
                "d_tc_resid": round(d_tc_resid, 3), "r2_adj": round(r2_adj, 3),
                "mean_flat": round(float(np.mean(f_arr)), 6),
                "mean_curved": round(float(np.mean(c_arr)), 6),
            }
            print(f"  {key:<22} d={d_val:+.3f} p={p_val:.6f} d_tc={d_tc_resid:+.3f} R2adj={r2_adj:.3f} [{status}|{tc_st}|{r2_st}]")

        results[metric_name] = metric_res

    # Conformal null
    print(f"\n=== Conformal NULL ===")
    for key in obs_keys:
        print(f"  {key:<22} d=0.000 (identical causal structure by construction)")

    # Save
    with open(RUN_DIR / "pilot_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RUN_DIR}/pilot_results.json")
    print(f"Total: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
