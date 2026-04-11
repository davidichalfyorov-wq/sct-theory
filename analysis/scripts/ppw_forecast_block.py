#!/usr/bin/env python3
"""pp-wave forecast validation block: 4 conditions from independent analysis signal prediction.

Run 1: T=0.50 eps=5 M=15  (q=1.25, predicted dk=+0.027, d~1.2)
Run 2: T=0.35 eps=5 M=30  (q=0.61, predicted dk=+0.007, d~0.6)
Run 3: T=1.00 eps=2 M=15  (q=2.00, predicted dk=+0.061, d~1.9)  -- purity anchor
Run 4: T=0.70 eps=5 M=15  (q=2.45, predicted dk=+0.082, d~2.3)  -- bonus strong signal
"""
import sys, os, time, json
import numpy as np
import concurrent.futures as cf

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, excess_kurtosis, ppwave_exact_preds, bulk_mask
)

ZETA = 0.15
N = 10000

CONDITIONS = [
    {"label": "T0.50_eps5", "T": 0.50, "eps": 5.0, "M": 15, "seed_base": 610000},
    {"label": "T0.35_eps5", "T": 0.35, "eps": 5.0, "M": 30, "seed_base": 620000},
    {"label": "T1.00_eps2", "T": 1.00, "eps": 2.0, "M": 15, "seed_base": 630000},
    {"label": "T0.70_eps5", "T": 0.70, "eps": 5.0, "M": 15, "seed_base": 640000},
]

# analytical predictions: dk ≈ 0.0186·q² − 8.3e-4·q⁴
def predict_dk(eps, T):
    q = eps * T * T
    return 0.0186 * q**2 - 8.3e-4 * q**4


def run_one(args):
    T, eps, seed = args
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)
    par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
    Y0 = Y_from_graph(par0, ch0)
    parF, chF = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=eps))
    YF = Y_from_graph(parF, chF)
    mask = bulk_mask(pts, T, ZETA)
    dk = excess_kurtosis(YF[mask]) - excess_kurtosis(Y0[mask])
    return dk


if __name__ == "__main__":
    print("=== PP-WAVE FORECAST VALIDATION BLOCK ===")
    print(f"N={N}, zeta={ZETA}, exact pp-wave")
    print(f"analytical model: dk ≈ 0.0186·q² − 8.3e-4·q⁴")
    print()

    all_results = {}
    t0_total = time.time()

    for cond in CONDITIONS:
        T = cond["T"]
        eps = cond["eps"]
        M = cond["M"]
        q = eps * T * T
        pred = predict_dk(eps, T)

        tasks = [(T, eps, cond["seed_base"] + si) for si in range(M)]

        t0 = time.time()
        with cf.ProcessPoolExecutor(max_workers=8) as ex:
            dks = list(ex.map(run_one, tasks))
        elapsed = time.time() - t0

        arr = np.array(dks)
        m = float(np.mean(arr))
        se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
        std = float(np.std(arr, ddof=1))
        d = float(m / std) if std > 0 else 0.0

        ratio = m / pred if abs(pred) > 1e-15 else float("inf")

        sig = "***" if abs(d) > 3 else "**" if abs(d) > 2 else "*" if abs(d) > 1 else ""
        print(f"{cond['label']}: q={q:.3f}, eps={eps}, T={T}")
        print(f"  predicted: dk={pred:+.6f}")
        print(f"  observed:  dk={m:+.6f} ± {se:.6f}, d={d:+.3f} {sig}")
        print(f"  ratio obs/pred: {ratio:.3f}")
        print(f"  ({elapsed:.0f}s, M={M})")
        print()

        all_results[cond["label"]] = {
            "T": T, "eps": eps, "q": q, "M": M,
            "predicted_dk": pred,
            "observed_dk": m, "se": se, "std": std, "d": d,
            "ratio_obs_pred": ratio,
            "per_seed": dks,
            "elapsed_s": elapsed,
        }

    total_elapsed = time.time() - t0_total
    print(f"Total: {total_elapsed:.0f}s = {total_elapsed/60:.1f}min")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "universal_runs_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ppw_forecast_block.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("Saved to ppw_forecast_block.json")
