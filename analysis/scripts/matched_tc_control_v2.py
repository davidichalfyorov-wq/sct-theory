#!/usr/bin/env python3
"""
matched_tc_control.py

Negative Control NC-2:
Matched-ΔTC geometric control for path_kurtosis.

Default experiment:
- target geometry: exact pp-wave, eps = 3.0, T = 1.0
- control geometry: exact de Sitter, H tuned by bracket search + bisection
- N = 10000
- M_seeds = 15
- zeta = 0.15
- sequential only

The script imports core primitives from run_universal.py.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Make local imports robust when launched from another working directory.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_universal import (
    sprinkle_local_diamond,
    minkowski_preds,
    ppwave_exact_preds,
    ds_preds,
    flrw_preds,
    build_hasse_from_predicate,
    Y_from_graph,
    excess_kurtosis,
    bulk_mask,
)


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def summarize(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "se": 0.0, "std": 0.0, "d": 0.0, "n": 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    se = float(std / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    d = float(mean / std) if std > 1e-15 else 0.0
    return {"mean": mean, "se": se, "std": std, "d": d, "n": int(arr.size)}


def count_causal_pairs(pts: np.ndarray, pred_fn) -> int:
    total = 0
    for i in range(len(pts)):
        total += int(np.asarray(pred_fn(pts, i), dtype=bool).sum())
    return total


def build_Y_on_mask(pts: np.ndarray, pred_fn, mask: np.ndarray) -> float:
    parents, children = build_hasse_from_predicate(pts, pred_fn)
    Y = Y_from_graph(parents, children)
    return float(excess_kurtosis(Y[mask])), Y


def make_control_pred(control: str, amp: float):
    if control == "ds":
        return lambda P, i: ds_preds(P, i, H=amp)
    if control == "flrw":
        return lambda P, i: flrw_preds(P, i, gamma=amp)
    raise ValueError(f"Unknown control geometry: {control}")


def coarse_bracket_match(
    pts: np.ndarray,
    dTC_target: int,
    control: str,
    lo: float,
    hi: float,
    n_grid: int = 17,
) -> Tuple[Tuple[float, float] | None, float, int]:
    """
    Search a log-spaced grid to locate a bracket for dTC_control - dTC_target = 0.
    Returns:
        (bracket or None, best_amp, best_dTC)
    """
    amps = np.geomspace(lo, hi, num=n_grid)
    vals = []
    for amp in amps:
        pred = make_control_pred(control, float(amp))
        dTC = count_causal_pairs(pts, pred) - count_causal_pairs(pts, lambda P, i: minkowski_preds(P, i))
        vals.append(int(dTC))
    vals = np.asarray(vals, dtype=np.int64)
    diff = vals - int(dTC_target)

    # Best point even if no bracket.
    idx_best = int(np.argmin(np.abs(diff)))
    best_amp = float(amps[idx_best])
    best_dTC = int(vals[idx_best])

    # Try to find adjacent sign change.
    for k in range(len(amps) - 1):
        if diff[k] == 0:
            return ((float(amps[k]), float(amps[k])), float(amps[k]), int(vals[k]))
        if diff[k] == 0 or diff[k + 1] == 0 or diff[k] * diff[k + 1] < 0:
            return ((float(amps[k]), float(amps[k + 1])), best_amp, best_dTC)
    return (None, best_amp, best_dTC)


def bisect_match(
    pts: np.ndarray,
    dTC_target: int,
    control: str,
    bracket: Tuple[float, float],
    n_bisect: int,
) -> Tuple[float, int]:
    lo, hi = bracket
    if lo == hi:
        amp = lo
        pred = make_control_pred(control, amp)
        dTC = count_causal_pairs(pts, pred) - count_causal_pairs(pts, lambda P, i: minkowski_preds(P, i))
        return float(amp), int(dTC)

    flat_TC = count_causal_pairs(pts, lambda P, i: minkowski_preds(P, i))

    def f(amp: float) -> int:
        pred = make_control_pred(control, amp)
        return count_causal_pairs(pts, pred) - flat_TC

    f_lo = f(lo) - dTC_target
    f_hi = f(hi) - dTC_target

    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid) - dTC_target
        if f_mid == 0:
            lo = hi = mid
            break
        if f_lo == 0:
            hi = lo
            break
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    amp = 0.5 * (lo + hi)
    dTC = f(amp)
    return float(amp), int(dTC)


def run_single_seed(
    seed: int,
    N: int,
    T: float,
    zeta: float,
    target_eps: float,
    control: str,
    control_lo: float,
    control_hi: float,
    n_bisect: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)
    mask = bulk_mask(pts, T, zeta)

    flat_pred = lambda P, i: minkowski_preds(P, i)
    ppw_pred = lambda P, i: ppwave_exact_preds(P, i, eps=target_eps)

    TC0 = count_causal_pairs(pts, flat_pred)
    TCA = count_causal_pairs(pts, ppw_pred)
    dTC_A = int(TCA - TC0)

    # Match control by coarse log-grid + bisection.
    bracket, best_amp, best_dTC = coarse_bracket_match(
        pts, dTC_A, control, control_lo, control_hi, n_grid=17
    )
    if bracket is not None:
        amp_B, dTC_B = bisect_match(
            pts, dTC_A, control, bracket, n_bisect=n_bisect
        )
        matched_by = "bisection"
    else:
        amp_B, dTC_B = best_amp, best_dTC
        matched_by = "nearest-grid"

    control_pred = make_control_pred(control, amp_B)

    # Build Hasse only for final matched amplitudes.
    k0, Y0 = build_Y_on_mask(pts, flat_pred, mask)
    kA, YA = build_Y_on_mask(pts, ppw_pred, mask)
    kB, YB = build_Y_on_mask(pts, control_pred, mask)

    dk_A = float(kA - k0)
    dk_B = float(kB - k0)
    match_relerr = float(abs(dTC_B - dTC_A) / max(abs(dTC_A), 1))

    return {
        "seed": int(seed),
        "TC0": int(TC0),
        "TCA": int(TCA),
        "dTC_target": int(dTC_A),
        "control_amp": float(amp_B),
        "control_geometry": control,
        "dTC_control": int(dTC_B),
        "match_relerr": match_relerr,
        "matched_by": matched_by,
        "dk_target": dk_A,
        "dk_control": dk_B,
        "delta_dk": float(dk_A - dk_B),
    }


def aggregate_results(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    dkA = np.array([r["dk_target"] for r in records], dtype=np.float64)
    dkB = np.array([r["dk_control"] for r in records], dtype=np.float64)
    ddk = dkA - dkB
    dTCA = np.array([r["dTC_target"] for r in records], dtype=np.float64)
    dTCB = np.array([r["dTC_control"] for r in records], dtype=np.float64)
    rel = np.array([r["match_relerr"] for r in records], dtype=np.float64)
    amps = np.array([r["control_amp"] for r in records], dtype=np.float64)

    stats_A = summarize(dkA)
    stats_B = summarize(dkB)
    stats_D = summarize(ddk)
    match_stats = summarize(rel)

    R_TC = float(abs(stats_B["mean"]) / max(abs(stats_A["mean"]), 1e-15))
    strong = (match_stats["mean"] < 0.05) and (R_TC < 0.5) and (abs(stats_D["mean"]) > 2.0 * stats_D["se"])
    moderate = (match_stats["mean"] < 0.05) and (0.5 <= R_TC < 0.8)

    verdict = "BAD"
    if strong:
        verdict = "STRONG_SUPPORT"
    elif moderate:
        verdict = "MODERATE_SUPPORT"
    elif R_TC < 0.8:
        verdict = "WEAK_SUPPORT"

    return {
        "target_dk": stats_A,
        "control_dk": stats_B,
        "delta_dk": stats_D,
        "dTC_target": summarize(dTCA),
        "dTC_control": summarize(dTCB),
        "match_relerr": match_stats,
        "control_amp": summarize(amps),
        "R_TC": R_TC,
        "verdict": verdict,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Matched-ΔTC geometric control for path_kurtosis.")
    p.add_argument("--N", type=int, default=10000)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=3.0, help="Target pp-wave epsilon.")
    p.add_argument("--control", choices=["ds", "flrw"], default="ds")
    p.add_argument("--M-seeds", type=int, default=15)
    p.add_argument("--zeta", type=float, default=0.15)
    p.add_argument("--seed-base", type=int, default=1000000)
    p.add_argument("--control-lo", type=float, default=0.01)
    p.add_argument("--control-hi", type=float, default=5.0)
    p.add_argument("--n-bisect", type=int, default=15)
    p.add_argument("--out", type=str, default="matched_tc_control_result.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)

    print(
        f"[matched_tc_control] N={args.N}, T={args.T}, eps={args.eps}, control={args.control}, "
        f"M_seeds={args.M_seeds}, zeta={args.zeta}",
        flush=True,
    )

    records: List[Dict[str, Any]] = []
    for si in range(args.M_seeds):
        seed = int(args.seed_base + si)
        print(f"[matched_tc_control] seed {si+1}/{args.M_seeds} (seed={seed}) ...", flush=True)
        rec = run_single_seed(
            seed=seed,
            N=args.N,
            T=args.T,
            zeta=args.zeta,
            target_eps=args.eps,
            control=args.control,
            control_lo=args.control_lo,
            control_hi=args.control_hi,
            n_bisect=args.n_bisect,
        )
        print(
            f"  dTC_target={rec['dTC_target']}, {args.control}_amp={rec['control_amp']:.6f}, "
            f"dTC_control={rec['dTC_control']}, relerr={rec['match_relerr']:.4f}, "
            f"dk_target={rec['dk_target']:+.6f}, dk_control={rec['dk_control']:+.6f}",
            flush=True,
        )
        records.append(rec)

    summary = aggregate_results(records)
    result = {
        "config": {
            "target_geometry": "ppwave_exact",
            "target_eps": float(args.eps),
            "control_geometry": args.control,
            "N": int(args.N),
            "T": float(args.T),
            "M_seeds": int(args.M_seeds),
            "zeta": float(args.zeta),
            "seed_base": int(args.seed_base),
            "control_bracket": [float(args.control_lo), float(args.control_hi)],
            "n_bisect": int(args.n_bisect),
        },
        "per_seed": records,
        "summary": summary,
    }

    out_path.write_text(json.dumps(result, indent=2, default=json_default), encoding="utf-8")
    print(f"[matched_tc_control] wrote {out_path}", flush=True)
    print(json.dumps(summary, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
