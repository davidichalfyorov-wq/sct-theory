#!/usr/bin/env python3
"""
deltaY_scramble_control.py

Negative Control NC-3:
Stratified deltaY scrambling control for path_kurtosis.

Default experiment:
- geometry: exact pp-wave, eps = 3.0, T = 1.0
- N = 10000
- M_seeds = 10
- K_perm = 100
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
from typing import Any, Dict, List

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_universal import (
    sprinkle_local_diamond,
    minkowski_preds,
    ppwave_exact_preds,
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


def compute_depth(parents: List[np.ndarray]) -> np.ndarray:
    n = len(parents)
    depth = np.zeros(n, dtype=np.int32)
    for i in range(n):
        p = parents[i]
        if p.size > 0:
            depth[i] = int(np.max(depth[p]) + 1)
    return depth


def make_strata_labels(
    pts: np.ndarray,
    T: float,
    depth: np.ndarray,
    n_tau: int = 5,
    n_rho: int = 3,
    n_depth: int = 3,
) -> np.ndarray:
    tau_hat = 2.0 * pts[:, 0] / T
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2.0 - np.abs(pts[:, 0])
    rho_hat = r / np.maximum(rmax, 1e-12)

    tau_bin = np.clip(np.floor((tau_hat + 1.0) * 0.5 * n_tau).astype(int), 0, n_tau - 1)
    rho_bin = np.clip(np.floor(rho_hat * n_rho).astype(int), 0, n_rho - 1)
    depth_tercile = np.clip((depth * n_depth) // (int(depth.max()) + 1), 0, n_depth - 1)

    labels = tau_bin * n_rho * n_depth + rho_bin * n_depth + depth_tercile
    return labels.astype(np.int32)


def scramble_deltaY(
    Y0: np.ndarray,
    Ycurv: np.ndarray,
    window_mask: np.ndarray,
    strata_labels: np.ndarray,
    rng: np.random.Generator,
) -> float:
    delta = Ycurv - Y0
    delta_scr = delta.copy()

    for label in np.unique(strata_labels[window_mask]):
        idx = np.where((strata_labels == label) & window_mask)[0]
        if len(idx) > 1:
            delta_scr[idx] = delta[rng.permutation(idx)]

    Yscr = Y0 + delta_scr
    dk_scr = excess_kurtosis(Yscr[window_mask]) - excess_kurtosis(Y0[window_mask])
    return float(dk_scr)


def run_single_seed(
    seed: int,
    N: int,
    T: float,
    eps: float,
    zeta: float,
    K_perm: int,
    n_tau: int,
    n_rho: int,
    n_depth: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    pts = sprinkle_local_diamond(N, T, rng)
    mask = bulk_mask(pts, T, zeta)

    flat_pred = lambda P, i: minkowski_preds(P, i)
    curv_pred = lambda P, i: ppwave_exact_preds(P, i, eps=eps)

    parents0, children0 = build_hasse_from_predicate(pts, flat_pred)
    Y0 = Y_from_graph(parents0, children0)

    parentsC, childrenC = build_hasse_from_predicate(pts, curv_pred)
    Ycurv = Y_from_graph(parentsC, childrenC)

    dk_curv = float(excess_kurtosis(Ycurv[mask]) - excess_kurtosis(Y0[mask]))

    depth = compute_depth(parents0)
    strata_labels = make_strata_labels(
        pts, T, depth, n_tau=n_tau, n_rho=n_rho, n_depth=n_depth
    )

    dk_scr = []
    for k in range(K_perm):
        dk_scr.append(scramble_deltaY(Y0, Ycurv, mask, strata_labels, rng))
    dk_scr = np.asarray(dk_scr, dtype=np.float64)

    return {
        "seed": int(seed),
        "dk_curv": dk_curv,
        "dk_scr_mean": float(np.mean(dk_scr)),
        "dk_scr_std": float(np.std(dk_scr, ddof=1)) if dk_scr.size > 1 else 0.0,
        "dk_scr_all": dk_scr,
    }


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    dk_curv = np.array([r["dk_curv"] for r in records], dtype=np.float64)
    dk_scr_seedmean = np.array([r["dk_scr_mean"] for r in records], dtype=np.float64)
    dk_scr_all = np.concatenate([r["dk_scr_all"] for r in records]).astype(np.float64)

    stats_curv = summarize(dk_curv)
    stats_scr_seed = summarize(dk_scr_seedmean)
    stats_scr_all = summarize(dk_scr_all)

    R_scr = float(abs(stats_scr_all["mean"]) / max(abs(stats_curv["mean"]), 1e-15))
    if R_scr < 0.25:
        verdict = "STRONG_SUPPORT"
    elif R_scr < 0.50:
        verdict = "MODERATE_SUPPORT"
    elif R_scr < 0.75:
        verdict = "WEAK_SUPPORT"
    else:
        verdict = "BAD"

    return {
        "dk_curv": stats_curv,
        "dk_scr_seedmean": stats_scr_seed,
        "dk_scr_all": stats_scr_all,
        "R_scr": R_scr,
        "verdict": verdict,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stratified deltaY scrambling control for path_kurtosis.")
    p.add_argument("--N", type=int, default=10000)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=3.0)
    p.add_argument("--M-seeds", type=int, default=10)
    p.add_argument("--K-perm", type=int, default=100)
    p.add_argument("--zeta", type=float, default=0.15)
    p.add_argument("--seed-base", type=int, default=1100000)
    p.add_argument("--n-tau", type=int, default=5)
    p.add_argument("--n-rho", type=int, default=3)
    p.add_argument("--n-depth", type=int, default=3)
    p.add_argument("--out", type=str, default="deltaY_scramble_control_result.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)

    print(
        f"[deltaY_scramble_control] N={args.N}, T={args.T}, eps={args.eps}, "
        f"M_seeds={args.M_seeds}, K_perm={args.K_perm}, zeta={args.zeta}",
        flush=True,
    )

    records: List[Dict[str, Any]] = []
    for si in range(args.M_seeds):
        seed = int(args.seed_base + si)
        print(f"[deltaY_scramble_control] seed {si+1}/{args.M_seeds} (seed={seed}) ...", flush=True)
        rec = run_single_seed(
            seed=seed,
            N=args.N,
            T=args.T,
            eps=args.eps,
            zeta=args.zeta,
            K_perm=args.K_perm,
            n_tau=args.n_tau,
            n_rho=args.n_rho,
            n_depth=args.n_depth,
        )
        print(
            f"  dk_curv={rec['dk_curv']:+.6f}, dk_scr_mean={rec['dk_scr_mean']:+.6f}, "
            f"dk_scr_std={rec['dk_scr_std']:.6f}",
            flush=True,
        )
        records.append(rec)

    summary = aggregate(records)
    result = {
        "config": {
            "geometry": "ppwave_exact",
            "eps": float(args.eps),
            "N": int(args.N),
            "T": float(args.T),
            "M_seeds": int(args.M_seeds),
            "K_perm": int(args.K_perm),
            "zeta": float(args.zeta),
            "seed_base": int(args.seed_base),
            "n_tau": int(args.n_tau),
            "n_rho": int(args.n_rho),
            "n_depth": int(args.n_depth),
        },
        "per_seed": [
            {
                "seed": r["seed"],
                "dk_curv": r["dk_curv"],
                "dk_scr_mean": r["dk_scr_mean"],
                "dk_scr_std": r["dk_scr_std"],
                "dk_scr_all": r["dk_scr_all"],
            }
            for r in records
        ],
        "summary": summary,
    }

    out_path.write_text(json.dumps(result, indent=2, default=json_default), encoding="utf-8")
    print(f"[deltaY_scramble_control] wrote {out_path}", flush=True)
    print(json.dumps(summary, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
