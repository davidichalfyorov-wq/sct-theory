#!/usr/bin/env python3
"""Schwarzschild local T-scaling — production run.

Fixes:
- 4 workers (not 8) to reduce memory pressure
- Stronger signal: M_sch=0.10 (q_W=0.80 at T=1) per independent analysis recommendation
- Error handling per-worker
- Progress reporting
- Two amplitudes: M=0.10 (strong) and M=0.05 (weak, for comparison)
"""
import sys, os, time, json, traceback
import numpy as np
import concurrent.futures as cf

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, excess_kurtosis, jet_preds, bulk_mask,
    riemann_schwarzschild_local
)

R0 = 0.50
N = 10000
ZETA = 0.15
WORKERS = 4

# Two amplitude lines
CONFIGS = [
    {"label": "strong", "M_sch": 0.10, "M_seeds": 50, "T_vals": [1.00, 0.70, 0.50]},
    {"label": "weak",   "M_sch": 0.05, "M_seeds": 50, "T_vals": [1.00, 0.70]},
]

# Precompute Riemann tensors at module level (pickleable numpy arrays)
R_STRONG = riemann_schwarzschild_local(0.10, R0)
R_WEAK = riemann_schwarzschild_local(0.05, R0)


def worker(args):
    try:
        label, T, seed = args
        R_sch = R_STRONG if label == "strong" else R_WEAK
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)
        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)
        parF, chF = build_hasse_from_predicate(pts, lambda P, i: jet_preds(P, i, R_abcd=R_sch))
        YF = Y_from_graph(parF, chF)
        mask = bulk_mask(pts, T, ZETA)
        dk = excess_kurtosis(YF[mask]) - excess_kurtosis(Y0[mask])
        return (label, T, seed, float(dk), None)
    except Exception as e:
        return (label, T, seed, 0.0, str(e))


if __name__ == "__main__":
    print(f"=== SCHWARZSCHILD LOCAL PRODUCTION RUN ===")
    print(f"r0={R0}, N={N}, zeta={ZETA}, workers={WORKERS}")
    print()

    # analytical predictions: dk_cons ≈ 0.031 * q_W²
    # Strong M=0.10: q_W(T=1) = 0.10/0.125 = 0.80, dk ≈ 0.031*0.64 = 0.020
    # Weak M=0.05: q_W(T=1) = 0.40, dk ≈ 0.031*0.16 = 0.005

    all_results = {}
    t0_total = time.time()

    for cfg in CONFIGS:
        label = cfg["label"]
        M_sch = cfg["M_sch"]
        M_seeds = cfg["M_seeds"]
        q_base = M_sch / (R0**3)

        print(f"--- {label}: M_sch={M_sch}, M/r0³={q_base:.3f} ---")

        for T in cfg["T_vals"]:
            q_W = q_base * T**2
            pred_dk = 0.031 * q_W**2

            tasks = [(label, T, 200000 + int(M_sch*10000) + int(T*1000)*100 + si)
                     for si in range(M_seeds)]

            t0 = time.time()
            with cf.ProcessPoolExecutor(max_workers=WORKERS) as ex:
                raw = list(ex.map(worker, tasks))
            elapsed = time.time() - t0

            errors = [r[4] for r in raw if r[4] is not None]
            dks = [r[3] for r in raw if r[4] is None]

            if errors:
                print(f"  T={T:.2f}: {len(errors)} ERRORS: {errors[0]}")

            arr = np.array(dks)
            m = float(np.mean(arr))
            se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
            std = float(np.std(arr, ddof=1))
            d = float(m / std) if std > 0 else 0.0
            t_stat = float(m / se) if se > 0 else 0.0
            ratio = m / pred_dk if abs(pred_dk) > 1e-15 else float("inf")

            sig = "***" if abs(t_stat) > 3 else "**" if abs(t_stat) > 2 else "*" if abs(t_stat) > 1 else ""
            print(f"  T={T:.2f}: q_W={q_W:.4f}")
            print(f"    predicted:  dk={pred_dk:+.6f}")
            print(f"    observed:   dk={m:+.6f} ± {se:.6f}")
            print(f"    d_cohen={d:+.3f}, t={t_stat:+.3f} {sig}, ratio={ratio:.3f}")
            print(f"    ({elapsed:.0f}s, n={len(dks)})")

            key = f"{label}_T{T}"
            all_results[key] = {
                "label": label, "M_sch": M_sch, "T": T, "q_W": q_W,
                "predicted_dk": pred_dk,
                "observed_dk": m, "se": se, "std": std,
                "d_cohen": d, "t_stat": t_stat, "ratio": ratio,
                "n": len(dks), "n_errors": len(errors),
                "per_seed": dks, "elapsed_s": elapsed,
            }

        print()

    # T-scaling for strong line
    print("=== T-SCALING (strong M=0.10) ===")
    strong_keys = [k for k in all_results if k.startswith("strong")]
    if strong_keys:
        dk1 = all_results["strong_T1.0"]["observed_dk"]
        for key in sorted(strong_keys):
            r = all_results[key]
            T = r["T"]
            dk = r["observed_dk"]
            ratio_dk = dk / dk1 if abs(dk1) > 1e-15 else 0
            print(f"T={T:.2f}: dk={dk:+.6f}, dk/dk(1)={ratio_dk:+.4f}, "
                  f"T²={T**2:.4f}, T⁴={T**4:.4f}, "
                  f"ratio/T²={ratio_dk/T**2:+.4f}, ratio/T⁴={ratio_dk/T**4:+.4f}")

    total = time.time() - t0_total
    print(f"\nTotal: {total:.0f}s = {total/60:.1f}min")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "universal_runs_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sch_local_production.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved to sch_local_production.json")
