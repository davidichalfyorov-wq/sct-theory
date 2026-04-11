#!/usr/bin/env python3
"""
PILOT: N=2000 CRN test of top COMP-SCAN survivors.

Observables:
  S1: svd_participation_ratio of C²
  S2: column_gini_C2
  S3: nuclear_norm_residual
  S4: heat_trace_0.1 (link-graph Laplacian)
  S5: sv_entropy of C²

Metrics: ppwave_quad (eps=2,5,10), schwarzschild (eps=0.005,0.01,0.02)
Controls: conformal (null, expected d=0)

CRN: same points, different metrics. Adaptive M: start 10, extend to 20.

Output: docs/analysis_runs/run_20260325_133158/pilot_results.json
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from numpy.linalg import svd, norm
from scipy import sparse
from scipy.stats import wilcoxon, ttest_rel

# ── Parameters ──────────────────────────────────────────────────────
N = 2000
M_INITIAL = 10
M_EXTEND = 20
T_DIAMOND = 1.0
MASTER_SEED = 99887

EPS_CONFIGS = {
    "ppwave_quad_eps2": ("ppwave_quad", 2.0),
    "ppwave_quad_eps5": ("ppwave_quad", 5.0),
    "ppwave_quad_eps10": ("ppwave_quad", 10.0),
    "schwarzschild_eps005": ("schwarzschild", 0.005),
    "schwarzschild_eps01": ("schwarzschild", 0.01),
    "schwarzschild_eps02": ("schwarzschild", 0.02),
    "conformal": ("conformal", 1.0),
}

RUN_DIR = Path(__file__).resolve().parents[2] / "docs" / "analysis_runs" / "run_20260325_133158"
RESULTS_FILE = RUN_DIR / "pilot_results.json"


# ── Sprinkling ──────────────────────────────────────────────────────
def sprinkle(N_target: int, T: float, rng: np.random.Generator) -> np.ndarray:
    """Sprinkle N_target points into a 4D causal diamond |t|+r < T/2."""
    pts = []
    while len(pts) < N_target:
        batch = rng.uniform(-T / 2, T / 2, size=(N_target * 8, 4))
        r = np.sqrt(batch[:, 1]**2 + batch[:, 2]**2 + batch[:, 3]**2)
        inside = np.abs(batch[:, 0]) + r < T / 2
        pts.extend(batch[inside].tolist())
    pts = np.array(pts[:N_target])
    return pts[np.argsort(pts[:, 0])]


# ── Causal matrices (block-vectorized for N=2000) ──────────────────
def causal_flat(pts: np.ndarray) -> np.ndarray:
    n = len(pts)
    C = np.zeros((n, n), dtype=np.float32)
    block = 400
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        dt = pts[np.newaxis, :, 0] - pts[i0:i1, np.newaxis, 0]
        dr2 = np.sum((pts[np.newaxis, :, 1:] - pts[i0:i1, np.newaxis, 1:])**2, axis=2)
        mask = (dt > 0) & (dt**2 > dr2)
        C[i0:i1, :] = mask.astype(np.float32)
    np.fill_diagonal(C, 0)
    return C


def causal_ppwave_quad(pts: np.ndarray, eps: float) -> np.ndarray:
    n = len(pts)
    C = np.zeros((n, n), dtype=np.float32)
    block = 400
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        dt = pts[np.newaxis, :, 0] - pts[i0:i1, np.newaxis, 0]
        dx = pts[np.newaxis, :, 1:] - pts[i0:i1, np.newaxis, 1:]
        dr2 = np.sum(dx**2, axis=2)
        xm = (pts[np.newaxis, :, 1] + pts[i0:i1, np.newaxis, 1]) / 2
        ym = (pts[np.newaxis, :, 2] + pts[i0:i1, np.newaxis, 2]) / 2
        dz = dx[:, :, 2]
        du = dt + dz
        f = xm**2 - ym**2
        interval = dt**2 - dr2 - eps * f * du**2 / 2
        mask = (dt > 0) & (interval > 0)
        C[i0:i1, :] = mask.astype(np.float32)
    np.fill_diagonal(C, 0)
    return C


def causal_schwarzschild(pts: np.ndarray, eps: float) -> np.ndarray:
    n = len(pts)
    C = np.zeros((n, n), dtype=np.float32)
    block = 400
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        dt = pts[np.newaxis, :, 0] - pts[i0:i1, np.newaxis, 0]
        dx = pts[np.newaxis, :, 1:] - pts[i0:i1, np.newaxis, 1:]
        dr2 = np.sum(dx**2, axis=2)
        mid = (pts[np.newaxis, :, 1:] + pts[i0:i1, np.newaxis, 1:]) / 2
        rm = np.sqrt(np.sum(mid**2, axis=2))
        Phi = -eps / (rm + 0.3)
        interval = (1 + 2 * Phi) * dt**2 - (1 - 2 * Phi) * dr2
        mask = (dt > 0) & (interval > 0)
        C[i0:i1, :] = mask.astype(np.float32)
    np.fill_diagonal(C, 0)
    return C


def causal_conformal(pts: np.ndarray, _eps: float) -> np.ndarray:
    return causal_flat(pts)


METRIC_FNS = {
    "ppwave_quad": causal_ppwave_quad,
    "schwarzschild": causal_schwarzschild,
    "conformal": causal_conformal,
}


# ── Link graph ──────────────────────────────────────────────────────
def build_link_laplacian_eigs(C: np.ndarray):
    """Build link graph and return Laplacian eigenvalues."""
    C_sp = sparse.csr_matrix(C)
    C2_sp = C_sp @ C_sp
    has_interv = (C2_sp != 0)
    link_sp = C_sp - C_sp.multiply(has_interv)
    A_sym = link_sp + link_sp.T
    degrees = np.array(A_sym.sum(axis=1)).flatten()
    L = sparse.diags(degrees) - A_sym
    eigs = np.linalg.eigh(L.toarray().astype(np.float64))[0]
    return eigs[eigs > 1e-10]


# ── Helpers ─────────────────────────────────────────────────────────
def gini_coefficient(values: np.ndarray) -> float:
    v = np.sort(np.abs(values))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * v)) / (n * np.sum(v)) - (n + 1) / n


def cohen_d_paired(x, y):
    diff = np.array(x) - np.array(y)
    s = diff.std(ddof=1)
    return diff.mean() / s if s > 0 else 0.0


# ── Observable computation ──────────────────────────────────────────
def compute_pilot_observables(C: np.ndarray) -> dict:
    n = len(C)
    obs = {}

    tc = float(C.sum())
    obs["tc"] = tc

    # C²
    C64 = C.astype(np.float64)
    C2 = C64 @ C64
    obs["n2"] = float(C2.sum())

    # SVD of C²
    sigmas_c2 = svd(C2, compute_uv=False)
    sigmas_nz = sigmas_c2[sigmas_c2 > 1e-10]

    # S1: SVD participation ratio
    if len(sigmas_nz) > 0:
        obs["svd_participation_ratio"] = float(np.sum(sigmas_nz))**2 / (n * float(np.sum(sigmas_nz**2)))
    else:
        obs["svd_participation_ratio"] = 0.0

    # S2: Column-norm Gini of C²
    col_norms = np.sqrt(np.sum(C2**2, axis=0))
    obs["column_gini_C2"] = gini_coefficient(col_norms)

    # S3: Nuclear norm residual
    sigmas_c = svd(C64, compute_uv=False)
    nuclear = float(sigmas_c.sum())
    obs["nuclear_norm"] = nuclear
    obs["nuclear_norm_residual"] = nuclear - 6.6 * math.sqrt(max(tc, 0))

    # S4: Heat trace
    lambdas = build_link_laplacian_eigs(C)
    obs["heat_trace_0.1"] = float(np.sum(np.exp(-lambdas * 0.1)))
    obs["heat_trace_0.5"] = float(np.sum(np.exp(-lambdas * 0.5)))

    # S5: SV entropy
    if len(sigmas_nz) > 1:
        sv_p = sigmas_nz / sigmas_nz.sum()
        obs["sv_entropy"] = float(-np.sum(sv_p * np.log(sv_p))) / np.log(len(sigmas_nz))
    else:
        obs["sv_entropy"] = 0.0

    # Monitoring
    obs["rank_C2"] = int(np.sum(sigmas_c2 > 1e-10))
    obs["link_count"] = 0  # placeholder, computed in link graph

    return obs


# ── Main ────────────────────────────────────────────────────────────
def run_pilot():
    print(f"PILOT: N={N}, M_init={M_INITIAL}, M_ext={M_EXTEND}")
    print(f"Metrics: {list(EPS_CONFIGS.keys())}")

    ss = np.random.SeedSequence(MASTER_SEED)
    seeds = ss.spawn(M_EXTEND)

    all_results = {}
    obs_names = ["svd_participation_ratio", "column_gini_C2", "nuclear_norm_residual",
                 "heat_trace_0.1", "heat_trace_0.5", "sv_entropy", "rank_C2"]

    for config_name, (metric_type, eps) in EPS_CONFIGS.items():
        print(f"\n--- {config_name} (eps={eps}) ---")
        metric_fn = METRIC_FNS[metric_type]
        trials = []
        t_start = time.time()
        M_run = M_INITIAL

        for trial in range(M_run):
            rng = np.random.default_rng(seeds[trial])
            pts = sprinkle(N, T_DIAMOND, rng)

            C_flat = causal_flat(pts)
            C_curved = metric_fn(pts, eps)

            obs_flat = compute_pilot_observables(C_flat)
            obs_curved = compute_pilot_observables(C_curved)

            record = {"trial": trial}
            for key in obs_flat:
                record[f"{key}_flat"] = obs_flat[key]
                record[f"{key}_curved"] = obs_curved[key]
                record[f"{key}_delta"] = obs_curved[key] - obs_flat[key]
            trials.append(record)

            elapsed = time.time() - t_start
            per = elapsed / (trial + 1)
            eta = per * (M_run - trial - 1)
            print(f"  Trial {trial+1}/{M_run} ({per:.1f}s/trial, ~{eta:.0f}s left)")

        # Adaptive extension
        any_promising = False
        for name in obs_names[:5]:
            deltas = [t[f"{name}_delta"] for t in trials]
            s = np.std(deltas, ddof=1)
            d = np.mean(deltas) / s if s > 0 else 0
            if abs(d) > 0.3:
                any_promising = True
                break

        if any_promising and M_run < M_EXTEND:
            print(f"  ** Extending to M={M_EXTEND}")
            for trial in range(M_run, M_EXTEND):
                rng = np.random.default_rng(seeds[trial])
                pts = sprinkle(N, T_DIAMOND, rng)
                C_flat = causal_flat(pts)
                C_curved = metric_fn(pts, eps)
                obs_flat = compute_pilot_observables(C_flat)
                obs_curved = compute_pilot_observables(C_curved)
                record = {"trial": trial}
                for key in obs_flat:
                    record[f"{key}_flat"] = obs_flat[key]
                    record[f"{key}_curved"] = obs_curved[key]
                    record[f"{key}_delta"] = obs_curved[key] - obs_flat[key]
                trials.append(record)
                elapsed = time.time() - t_start
                per = elapsed / (trial + 1)
                print(f"  Trial {trial+1}/{M_EXTEND} ({per:.1f}s/trial)")

        # Analysis
        analysis = {}
        for name in obs_names + ["tc", "n2"]:
            flat_v = np.array([t.get(f"{name}_flat", 0) for t in trials])
            curv_v = np.array([t.get(f"{name}_curved", 0) for t in trials])
            deltas = curv_v - flat_v
            d = cohen_d_paired(curv_v, flat_v)
            try:
                _, pw = wilcoxon(deltas, alternative='two-sided')
            except Exception:
                pw = 1.0
            analysis[name] = {
                "cohen_d": round(d, 4),
                "p_wilcoxon": float(pw),
                "mean_delta": float(deltas.mean()),
                "std_delta": float(deltas.std()),
                "mean_flat": float(flat_v.mean()),
                "n_trials": len(trials),
            }
            star = "***" if pw < 0.001 else ("**" if pw < 0.01 else ("*" if pw < 0.05 else ""))
            print(f"  {name:<28} d={d:+.4f}  p={pw:.2e} {star}")

        all_results[config_name] = {"metric": metric_type, "eps": eps,
                                     "n_trials": len(trials), "analysis": analysis}

    # Save
    output = {"parameters": {"N": N, "T": T_DIAMOND, "seed": MASTER_SEED},
              "results": all_results, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {RESULTS_FILE}")

    # Dose-response summary
    print("\n=== DOSE-RESPONSE ===")
    for name in ["svd_participation_ratio", "column_gini_C2", "nuclear_norm_residual", "heat_trace_0.1"]:
        print(f"\n{name}:")
        for cn in EPS_CONFIGS:
            a = all_results[cn]["analysis"].get(name, {})
            print(f"  {cn:<30} d={a.get('cohen_d',0):+.4f}  p={a.get('p_wilcoxon',1):.2e}")

    return output


if __name__ == "__main__":
    run_pilot()
