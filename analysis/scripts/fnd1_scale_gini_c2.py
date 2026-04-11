#!/usr/bin/env python3
"""
Phase 7 SCALE: column_gini_C2 dose-response + N-scaling.

Dose-response: pp-wave eps = {1, 2, 5, 10}, N=2000, M=20
N-scaling: eps=5, N = {500, 1000, 2000}, M=20
Multi-metric: pp-wave + Schwarzschild eps={0.005, 0.01, 0.02}
Boundary sweep: {3, 5, 8, 12} link-depth threshold

Output: docs/analysis_runs/run_20260325_140000/scale_results.json
"""

import json
import time
from pathlib import Path
import numpy as np
from numpy.linalg import svd
from scipy import sparse
from scipy.stats import wilcoxon

MASTER_SEED = 55555
T_DIAMOND = 1.0
RUN_DIR = Path(__file__).resolve().parents[2] / "docs" / "analysis_runs" / "run_20260325_140000"
RESULTS_FILE = RUN_DIR / "scale_results.json"


def sprinkle(N_target, T, rng):
    pts = []
    while len(pts) < N_target:
        batch = rng.uniform(-T / 2, T / 2, size=(N_target * 8, 4))
        r = np.sqrt(batch[:, 1]**2 + batch[:, 2]**2 + batch[:, 3]**2)
        inside = np.abs(batch[:, 0]) + r < T / 2
        pts.extend(batch[inside].tolist())
    pts = np.array(pts[:N_target])
    return pts[np.argsort(pts[:, 0])]


def causal_flat(pts, _eps=0.0):
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        dt = pts[i + 1:, 0] - pts[i, 0]
        dr2 = np.sum((pts[i + 1:, 1:] - pts[i, 1:])**2, axis=1)
        C[i, i + 1:] = (dt**2 > dr2).astype(np.int8)
    return C


def causal_ppwave_quad(pts, eps):
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        dt = pts[i + 1:, 0] - pts[i, 0]
        dx = pts[i + 1:, 1:] - pts[i, 1:]
        dr2 = np.sum(dx**2, axis=1)
        xm = (pts[i + 1:, 1] + pts[i, 1]) / 2
        ym = (pts[i + 1:, 2] + pts[i, 2]) / 2
        dz = dx[:, 2]
        du = dt + dz
        f = xm**2 - ym**2
        interval = dt**2 - dr2 - eps * f * du**2 / 2
        C[i, i + 1:] = (interval > 0).astype(np.int8)
    return C


def causal_schwarzschild(pts, eps):
    n = len(pts)
    C = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        dt = pts[i + 1:, 0] - pts[i, 0]
        dx = pts[i + 1:, 1:] - pts[i, 1:]
        dr2 = np.sum(dx**2, axis=1)
        rm = np.sqrt(np.sum(((pts[i + 1:, 1:] + pts[i, 1:]) / 2)**2, axis=1))
        phi = -eps / (rm + 0.3)
        interval = (1 + 2 * phi) * dt**2 - (1 - 2 * phi) * dr2
        C[i, i + 1:] = (interval > 0).astype(np.int8)
    return C


def gini_coefficient(values):
    v = np.sort(np.abs(values))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * v)) / (n * np.sum(v)) - (n + 1) / n


def compute_gini_c2(C):
    C2 = C.astype(np.float64) @ C.astype(np.float64)
    col_norms = np.sqrt(np.sum(C2**2, axis=0))
    return gini_coefficient(col_norms)


def compute_gini_c2_interior(C, min_degree=5):
    """Compute G_w on interior elements only (exclude boundary)."""
    C2 = C.astype(np.float64) @ C.astype(np.float64)
    link_mask = (C > 0) & (C2 == 0)
    d_plus = link_mask.sum(axis=1)
    d_minus = link_mask.sum(axis=0)
    d_total = np.asarray(d_plus + d_minus).flatten()
    interior = d_total >= min_degree
    n_int = int(interior.sum())
    if n_int < 20:
        return float('nan'), n_int
    idx = np.where(interior)[0]
    C_int = C[np.ix_(idx, idx)]
    C2_int = C_int.astype(np.float64) @ C_int.astype(np.float64)
    col_norms = np.sqrt(np.sum(C2_int**2, axis=0))
    return gini_coefficient(col_norms), n_int


def cohen_d(x, y):
    diff = np.array(x) - np.array(y)
    s = diff.std(ddof=1)
    return diff.mean() / s if s > 0 else 0.0


def run_experiment(N, M, metric_fn, eps, label):
    """Run CRN experiment, return Cohen's d and stats for column_gini_C2."""
    ss = np.random.SeedSequence(MASTER_SEED)
    seeds = ss.spawn(M)

    gini_flat, gini_curved = [], []
    tc_flat, tc_curved = [], []
    gini_int_flat, gini_int_curved = [], []

    t0 = time.time()
    for trial in range(M):
        rng = np.random.default_rng(seeds[trial])
        pts = sprinkle(N, T_DIAMOND, rng)
        C_f = causal_flat(pts)
        C_c = metric_fn(pts, eps)

        gini_flat.append(compute_gini_c2(C_f))
        gini_curved.append(compute_gini_c2(C_c))
        tc_flat.append(int(C_f.sum()))
        tc_curved.append(int(C_c.sum()))

        if trial < 5:
            gf, _ = compute_gini_c2_interior(C_f)
            gc, ni = compute_gini_c2_interior(C_c)
            gini_int_flat.append(gf)
            gini_int_curved.append(gc)

        elapsed = time.time() - t0
        if trial % 5 == 0 or trial == M - 1:
            eta = elapsed / (trial + 1) * (M - trial - 1)
            print(f"    {label} trial {trial + 1}/{M} ({elapsed:.1f}s, ~{eta:.0f}s ETA)")

    gf_arr = np.array(gini_flat)
    gc_arr = np.array(gini_curved)
    deltas = gc_arr - gf_arr

    d = cohen_d(gc_arr, gf_arr)
    try:
        _, pw = wilcoxon(deltas, alternative='two-sided')
    except Exception:
        pw = 1.0

    # R² vs TC
    tc_deltas = np.array(tc_curved) - np.array(tc_flat)
    if np.var(tc_deltas) > 0 and np.var(deltas) > 0:
        r2_tc = float(np.corrcoef(tc_deltas, deltas)[0, 1]**2)
    else:
        r2_tc = 0.0

    # Boundary test
    if gini_int_flat:
        gf_int = np.array(gini_int_flat)
        gc_int = np.array(gini_int_curved)
        d_int = cohen_d(gc_int, gf_int)
    else:
        d_int = float('nan')

    return {
        "N": N, "M": M, "eps": eps, "label": label,
        "cohen_d": round(d, 3), "p_wilcoxon": float(pw),
        "mean_delta": float(deltas.mean()), "std_delta": float(deltas.std()),
        "mean_flat": float(gf_arr.mean()), "mean_curved": float(gc_arr.mean()),
        "r2_vs_tc": round(r2_tc, 3),
        "d_interior": round(d_int, 3) if not np.isnan(d_int) else None,
        "elapsed_s": round(time.time() - t0, 1),
    }


def main():
    print("=" * 70)
    print("SCALE: column_gini_C2 — dose-response + N-scaling + multi-metric")
    print("=" * 70)

    all_results = {}

    # 1. DOSE-RESPONSE: pp-wave, N=2000, eps = {1, 2, 5, 10}
    print("\n--- DOSE-RESPONSE (pp-wave, N=2000, M=20) ---")
    for eps in [1.0, 2.0, 5.0, 10.0]:
        label = f"ppwave_N2000_eps{int(eps)}"
        result = run_experiment(2000, 20, causal_ppwave_quad, eps, label)
        all_results[label] = result
        star = "***" if result["p_wilcoxon"] < 0.001 else ("**" if result["p_wilcoxon"] < 0.01 else ("*" if result["p_wilcoxon"] < 0.05 else ""))
        print(f"  eps={eps:5.1f}  d={result['cohen_d']:+.3f}  p={result['p_wilcoxon']:.2e}  R²_tc={result['r2_vs_tc']:.3f}  {star}")

    # 2. N-SCALING: pp-wave eps=5, N = {500, 1000, 2000}
    print("\n--- N-SCALING (pp-wave eps=5, M=20) ---")
    for N_val in [500, 1000, 2000]:
        label = f"ppwave_N{N_val}_eps5"
        if label in all_results:
            result = all_results[label]
        else:
            result = run_experiment(N_val, 20, causal_ppwave_quad, 5.0, label)
            all_results[label] = result
        print(f"  N={N_val:5d}  d={result['cohen_d']:+.3f}  p={result['p_wilcoxon']:.2e}  R²_tc={result['r2_vs_tc']:.3f}")

    # 3. MULTI-METRIC: Schwarzschild, N=2000
    print("\n--- MULTI-METRIC (Schwarzschild, N=2000, M=20) ---")
    for eps in [0.005, 0.01, 0.02]:
        label = f"schwarz_N2000_eps{eps}"
        result = run_experiment(2000, 20, causal_schwarzschild, eps, label)
        all_results[label] = result
        star = "***" if result["p_wilcoxon"] < 0.001 else ("**" if result["p_wilcoxon"] < 0.01 else ("*" if result["p_wilcoxon"] < 0.05 else ""))
        print(f"  eps={eps:.3f}  d={result['cohen_d']:+.3f}  p={result['p_wilcoxon']:.2e}  R²_tc={result['r2_vs_tc']:.3f}  {star}")

    # 4. Summary
    print("\n" + "=" * 70)
    print("SCALE SUMMARY for column_gini_C2")
    print("=" * 70)

    # Dose-response check
    ppw_doses = sorted([r for k, r in all_results.items() if k.startswith("ppwave_N2000")],
                       key=lambda x: x["eps"])
    if len(ppw_doses) >= 3:
        ds = [r["cohen_d"] for r in ppw_doses]
        monotonic = all(ds[i] <= ds[i+1] for i in range(len(ds)-1)) or \
                    all(ds[i] >= ds[i+1] for i in range(len(ds)-1))
        print(f"  Dose-response (pp-wave): {['eps='+str(r['eps'])+':d='+str(r['cohen_d']) for r in ppw_doses]}")
        print(f"  Monotonic: {monotonic}")

    # N-scaling check
    nscale = sorted([r for k, r in all_results.items() if "eps5" in k and "ppwave" in k],
                    key=lambda x: x["N"])
    if len(nscale) >= 2:
        ds = [r["cohen_d"] for r in nscale]
        print(f"  N-scaling: {['N='+str(r['N'])+':d='+str(r['cohen_d']) for r in nscale]}")
        converges = abs(ds[-1]) > 0.3
        print(f"  d converges to nonzero: {converges} (d_max = {ds[-1]})")

    # Multi-metric check
    schw = sorted([r for k, r in all_results.items() if "schwarz" in k], key=lambda x: x["eps"])
    if schw:
        print(f"  Schwarzschild: {['eps='+str(r['eps'])+':d='+str(r['cohen_d']) for r in schw]}")

    # Save
    output = {
        "observable": "column_gini_C2",
        "formula": "Gini coefficient of ||col_j(C^2)||_2",
        "results": all_results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
