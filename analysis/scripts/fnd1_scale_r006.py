#!/usr/bin/env python3
"""Analysis run Phase 7 SCALE: N-scaling, holdout, boundary, dose-response."""
import json, time
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, skew as sp_skew, spearmanr

SEED = 77777
TT = 1.0
RUN_DIR = Path("docs/analysis_runs/run_20260325_200948")

def sprinkle(Nt, T, rng):
    pts = []
    while len(pts) < Nt:
        b = rng.uniform(-T/2, T/2, size=(Nt*8, 4))
        r = np.sqrt(b[:, 1]**2 + b[:, 2]**2 + b[:, 3]**2)
        pts.extend(b[np.abs(b[:, 0]) + r < T/2].tolist())
    pts = np.array(pts[:Nt])
    return pts[np.argsort(pts[:, 0])]

def causal_flat(pts, _=0.0):
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        dt = pts[i+1:, 0] - pts[i, 0]
        dr2 = np.sum((pts[i+1:, 1:] - pts[i, 1:])**2, axis=1)
        C[i, i+1:] = (dt**2 > dr2).astype(np.int8)
    return C

def causal_ppwave(pts, eps):
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        dt = pts[i+1:, 0] - pts[i, 0]
        dx = pts[i+1:, 1:] - pts[i, 1:]
        dr2 = np.sum(dx**2, axis=1)
        xm = (pts[i+1:, 1] + pts[i, 1]) / 2
        ym = (pts[i+1:, 2] + pts[i, 2]) / 2
        du = dt + dx[:, 2]
        f = xm**2 - ym**2
        C[i, i+1:] = (dt**2 - dr2 - eps * f * du**2 / 2 > 0).astype(np.int8)
    return C

def causal_schwarz(pts, eps):
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        dt = pts[i+1:, 0] - pts[i, 0]
        dx = pts[i+1:, 1:] - pts[i, 1:]
        dr2 = np.sum(dx**2, axis=1)
        rm = np.sqrt(np.sum(((pts[i+1:, 1:] + pts[i, 1:]) / 2)**2, axis=1))
        phi = -eps / (rm + 0.3)
        C[i, i+1:] = ((1 + 2*phi) * dt**2 - (1 - 2*phi) * dr2 > 0).astype(np.int8)
    return C

def causal_flrw(pts, eps):
    """FLRW with a(t) = 1 + eps*t^2 (de Sitter approximation)."""
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        dt = pts[i+1:, 0] - pts[i, 0]
        dx = pts[i+1:, 1:] - pts[i, 1:]
        dr2 = np.sum(dx**2, axis=1)
        tm = (pts[i+1:, 0] + pts[i, 0]) / 2
        a2 = (1 + eps * tm**2)**2
        C[i, i+1:] = (dt**2 - a2 * dr2 > 0).astype(np.int8)
    return C

def gini(v):
    v = np.sort(np.abs(v))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    return float((2 * np.sum(np.arange(1, n+1) * v)) / (n * v.sum()) - (n+1) / n)

def compute_obs(C, pts):
    n = len(C)
    Cf = C.astype(np.float64)
    o = {}
    o["tc"] = int(C.sum())
    C2 = Cf @ Cf
    lm = (C > 0) & (C2 == 0)
    A = lm.astype(np.float32)
    As = A + A.T
    deg = As.sum(axis=1)
    o["column_gini_C"] = gini(np.sqrt(Cf.sum(axis=0)))
    o["link_degree_skew"] = float(sp_skew(deg))
    o["column_gini_C2"] = gini(np.sqrt(np.sum(C2**2, axis=0)))
    return o

def cd(arr):
    s = arr.std(ddof=1)
    if s < 1e-15:
        return 0.0 if abs(arr.mean()) < 1e-15 else float(np.sign(arr.mean()) * 999)
    return float(arr.mean() / s)

def crn_batch(N_val, M_val, metric_fn, eps, seed_offset=0):
    """Run M CRN trials at given N."""
    results = []
    CA = ["column_gini_C", "link_degree_skew", "column_gini_C2"]
    for i in range(M_val):
        rng = np.random.default_rng(SEED * 1000 + seed_offset + i)
        pts = sprinkle(N_val, TT, rng)
        Cf = causal_flat(pts)
        Cc = metric_fn(pts, eps)
        of = compute_obs(Cf, pts)
        oc = compute_obs(Cc, pts)
        d = {k: float(oc[k] - of[k]) for k in CA}
        results.append(d)
    stats = {}
    for nm in CA:
        da = np.array([r[nm] for r in results])
        stats[nm] = {"d": round(cd(da), 3), "mean_delta": round(float(da.mean()), 6)}
    return stats

def main():
    t0 = time.time()
    CA = ["column_gini_C", "link_degree_skew"]
    results = {}

    # 1. N-SCALING: N=500, 1000, 2000, 3000 (pp-wave eps=5)
    print("=== N-SCALING (ppwave eps=5) ===", flush=True)
    results["n_scaling_ppw"] = {}
    for N_val in [500, 1000, 2000, 3000]:
        M_val = max(10, 20 - N_val // 1000)  # fewer trials at larger N
        t1 = time.time()
        st = crn_batch(N_val, M_val, causal_ppwave, 5.0, seed_offset=N_val)
        dt = time.time() - t1
        results["n_scaling_ppw"][N_val] = st
        for nm in CA:
            print(f"  N={N_val:5d} M={M_val:2d} {nm:25s} d={st[nm]['d']:+.3f} ({dt:.0f}s)", flush=True)

    print("\n=== N-SCALING (Schwarzschild eps=0.005) ===", flush=True)
    results["n_scaling_sch"] = {}
    for N_val in [500, 1000, 2000, 3000]:
        M_val = max(10, 20 - N_val // 1000)
        t1 = time.time()
        st = crn_batch(N_val, M_val, causal_schwarz, 0.005, seed_offset=N_val + 5000)
        dt = time.time() - t1
        results["n_scaling_sch"][N_val] = st
        for nm in CA:
            print(f"  N={N_val:5d} M={M_val:2d} {nm:25s} d={st[nm]['d']:+.3f} ({dt:.0f}s)", flush=True)

    # 2. DOSE-RESPONSE (pp-wave, N=2000, M=10)
    print("\n=== DOSE-RESPONSE (ppwave, N=2000) ===", flush=True)
    results["dose_ppw"] = {}
    for eps in [1.0, 2.0, 5.0, 10.0]:
        st = crn_batch(2000, 10, causal_ppwave, eps, seed_offset=int(eps * 100))
        results["dose_ppw"][eps] = st
        for nm in CA:
            print(f"  eps={eps:5.1f} {nm:25s} d={st[nm]['d']:+.3f}", flush=True)

    print("\n=== DOSE-RESPONSE (Schwarzschild, N=2000) ===", flush=True)
    results["dose_sch"] = {}
    for eps in [0.001, 0.005, 0.01, 0.02]:
        st = crn_batch(2000, 10, causal_schwarz, eps, seed_offset=int(eps * 10000) + 9000)
        results["dose_sch"][eps] = st
        for nm in CA:
            print(f"  eps={eps:6.3f} {nm:25s} d={st[nm]['d']:+.3f}", flush=True)

    # 3. HOLDOUT: FLRW (de Sitter approximation), N=2000, M=15
    print("\n=== HOLDOUT: FLRW/dS (N=2000, M=15) ===", flush=True)
    st = crn_batch(2000, 15, causal_flrw, 2.0, seed_offset=8000)
    results["holdout_flrw"] = st
    for nm in CA:
        d_val = st[nm]["d"]
        status = "PASS" if abs(d_val) >= 0.5 else "WEAK" if abs(d_val) >= 0.2 else "FAIL"
        print(f"  FLRW eps=2 {nm:25s} d={d_val:+.3f} {status}", flush=True)

    # 4. BOUNDARY SWEEP (pp-wave eps=5, N=2000, M=10)
    print("\n=== BOUNDARY SWEEP (ppwave eps=5, N=2000) ===", flush=True)
    results["boundary"] = {}
    for excl_pct in [5, 10, 20, 30]:
        def compute_obs_boundary(C, pts, excl=excl_pct):
            n = len(C)
            # Exclude boundary elements (top/bottom excl% by time)
            n_excl = int(n * excl / 100)
            interior = slice(n_excl, n - n_excl)
            C_int = C[interior, :][:, interior]
            pts_int = pts[interior]
            return compute_obs(C_int, pts_int)

        res_bnd = []
        for i in range(10):
            rng = np.random.default_rng(SEED * 4000 + i)
            pts = sprinkle(2000, TT, rng)
            Cf = causal_flat(pts)
            Cc = causal_ppwave(pts, 5.0)
            of_b = compute_obs_boundary(Cf, pts)
            oc_b = compute_obs_boundary(Cc, pts)
            res_bnd.append({k: float(oc_b[k] - of_b[k]) for k in CA})

        bnd_stats = {}
        for nm in CA:
            da = np.array([r[nm] for r in res_bnd])
            bnd_stats[nm] = round(cd(da), 3)
            print(f"  excl={excl_pct:2d}% {nm:25s} d={bnd_stats[nm]:+.3f}", flush=True)
        results["boundary"][excl_pct] = bnd_stats

    # Check boundary kill criterion
    print("\n=== BOUNDARY KILL CHECK ===", flush=True)
    for nm in CA:
        d_vals = [results["boundary"][p][nm] for p in [5, 10, 20, 30]]
        monotonic_decrease = all(abs(d_vals[i]) >= abs(d_vals[i+1]) for i in range(3))
        ratio = abs(d_vals[3]) / max(abs(d_vals[0]), 1e-10)
        kill = monotonic_decrease and ratio < 0.5
        print(f"  {nm:25s} d@5%={d_vals[0]:+.3f} d@30%={d_vals[3]:+.3f} ratio={ratio:.2f} mono={monotonic_decrease} {'KILL' if kill else 'PASS'}", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.0f}s", flush=True)

    # Save
    # Convert numeric keys to strings for JSON
    def clean_keys(obj):
        if isinstance(obj, dict):
            return {str(k): clean_keys(v) for k, v in obj.items()}
        return obj

    with open(RUN_DIR / "scale_results.json", "w") as f:
        json.dump(clean_keys(results), f, indent=2)
    print(f"Saved: scale_results.json", flush=True)

if __name__ == "__main__":
    main()
