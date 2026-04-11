#!/usr/bin/env python3
"""Schwarzschild LOCAL T-scaling: M=200 seeds, 8 workers."""
import sys, time, json, os
import numpy as np
import concurrent.futures as cf

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, excess_kurtosis, jet_preds, bulk_mask,
    riemann_schwarzschild_local
)

M_SCH = 0.05
R0 = 0.50
N = 10000
M_SEEDS = 50
ZETA = 0.15
T_VALS = [1.00, 0.70, 0.50, 0.35]
R_SCH = riemann_schwarzschild_local(M_SCH, R0)


def run_one(args):
    T, si = args
    seed = 300000 + int(T * 1000) * 100 + si
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)
    par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
    Y0 = Y_from_graph(par0, ch0)
    parF, chF = build_hasse_from_predicate(pts, lambda P, i: jet_preds(P, i, R_abcd=R_SCH))
    YF = Y_from_graph(parF, chF)
    mask = bulk_mask(pts, T, ZETA)
    dk = excess_kurtosis(YF[mask]) - excess_kurtosis(Y0[mask])
    return (T, si, dk)


if __name__ == "__main__":
    print(f"=== SCHWARZSCHILD LOCAL M={M_SEEDS} seeds, N={N} ===")
    print(f"M_sch={M_SCH}, r0={R0}, M/r0³={M_SCH / R0**3:.3f}")
    print()

    tasks = [(T, si) for T in T_VALS for si in range(M_SEEDS)]
    print(f"Total tasks: {len(tasks)} ({len(T_VALS)}T x {M_SEEDS} seeds)")

    t0 = time.time()
    with cf.ProcessPoolExecutor(max_workers=8) as ex:
        results_raw = list(ex.map(run_one, tasks))
    elapsed = time.time() - t0

    results = {}
    for T in T_VALS:
        dks = [dk for t, si, dk in results_raw if t == T]
        arr = np.array(dks)
        m = float(np.mean(arr))
        se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
        d = float(m / np.std(arr, ddof=1)) if np.std(arr, ddof=1) > 0 else 0.0
        q_W = M_SCH / (R0**3) * T**2
        results[T] = {"mean": m, "se": se, "d": d, "q_W": q_W, "n": len(dks)}
        sig = "***" if abs(d) > 3 else "**" if abs(d) > 2 else "*" if abs(d) > 1 else ""
        print(f"T={T:.2f}: q_W={q_W:.4f}, dk={m:+.8f}±{se:.8f}, d={d:+.4f} {sig}  (n={len(dks)})")

    print()
    dk1 = results[1.0]["mean"]
    if abs(dk1) > 1e-15:
        print("=== T-SCALING ===")
        for T in T_VALS:
            dk = results[T]["mean"]
            ratio = dk / dk1
            print(
                f"T={T:.2f}: dk/dk(1)={ratio:+.4f}, T²={T**2:.4f}, T⁴={T**4:.4f}, "
                f"ratio/T²={ratio / T**2:+.4f}, ratio/T⁴={ratio / T**4:+.4f}"
            )

    print(f"\nTotal: {elapsed:.0f}s = {elapsed / 60:.1f}min")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "universal_runs_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sch_local_M200.json"), "w") as f:
        json.dump(
            {"M_sch": M_SCH, "r0": R0, "N": N, "M_seeds": M_SEEDS,
             "results": {str(k): v for k, v in results.items()}, "elapsed_s": elapsed},
            f, indent=2,
        )
    print("Saved.")
