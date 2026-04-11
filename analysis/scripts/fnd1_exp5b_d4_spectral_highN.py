"""
FND-1 EXP-5b: d=4 Spectral Action Heat Trace — High N Extension.

Extends EXP-5 to N=5000, 8000, 10000 for spectral dimension convergence.
Uses the SAME code as EXP-5 but with larger N values only.
EXP-5 results at N=500-3000 remain valid; this adds the high-N data.

At N=3000, d_S ~ 2.5 (expected 4). Need N→10000+ to check convergence.

Run:
    python analysis/scripts/fnd1_exp5b_d4_spectral_highN.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from scipy import stats
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_4d_experiment import (
    sprinkle_4d_flat, causal_matrix_4d, compute_layers_4d,
    bd_action_4d, build_link_graph, _ppwave_profile,
)
from fnd1_4d_followup import quadrupole_profile
from fnd1_experiment_registry import (
    ExperimentMeta, save_experiment, RESULTS_DIR,
)
from fnd1_parallel import N_WORKERS, _init_worker

try:
    from fnd1_gpu import gpu_eigvalsh
except ImportError:
    gpu_eigvalsh = np.linalg.eigvalsh

# ---------------------------------------------------------------------------
# Parameters — same as EXP-5 except N values
# ---------------------------------------------------------------------------

N_VALUES = [5000, 8000, 10000]
M_ENSEMBLE = 60
T_DIAMOND = 1.0
MASTER_SEED = 5770  # different from EXP-5 (577) for independence
WORKERS = N_WORKERS

# Heat trace — same grid as EXP-5
N_TAU = 200
TAU_MIN = 1e-4
TAU_MAX = 100.0
TAU_GRID = np.logspace(np.log10(TAU_MIN), np.log10(TAU_MAX), N_TAU)
D = 4
POWER = D / 2.0

# Both profiles for comparison
EPS_COSCOSH = [0.0, 0.2, 0.5]
EPS_QUADRUPOLE = [0.0, 5.0, 10.0]


# ---------------------------------------------------------------------------
# Worker — identical to EXP-5 worker
# ---------------------------------------------------------------------------

def _worker(args):
    """Compute link-graph Laplacian heat trace for one d=4 sprinkling at high N."""
    seed_int, N, T, eps, profile = args

    rng = np.random.default_rng(seed_int)
    pts = sprinkle_4d_flat(N, T, rng)

    # Causal matrix
    if abs(eps) < 1e-12:
        C = causal_matrix_4d(pts, 0.0, "flat")
    else:
        t = pts[:, 0]; x = pts[:, 1]; y = pts[:, 2]; z = pts[:, 3]
        dt = t[np.newaxis, :] - t[:, np.newaxis]
        dx = x[np.newaxis, :] - x[:, np.newaxis]
        dy = y[np.newaxis, :] - y[:, np.newaxis]
        dz = z[np.newaxis, :] - z[:, np.newaxis]
        dr2 = dx ** 2 + dy ** 2 + dz ** 2
        xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
        ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
        if profile == "quadrupole":
            f_mid = quadrupole_profile(xm, ym)
        else:
            f_mid = _ppwave_profile(xm, ym)
        mink = dt ** 2 - dr2
        corr = eps * f_mid * (dt + dz) ** 2 / 2.0
        C = ((mink > corr) & (dt > 0)).astype(np.float64)
        del dx, dy, xm, ym, f_mid, mink, corr  # free memory

    n_matrix, N0, N1, N2, N3 = compute_layers_4d(C)
    total_causal = float(np.sum(C))
    bd = bd_action_4d(N, N0, N1, N2, N3)

    # Link graph + Laplacian
    A_link = build_link_graph(C, n_matrix)
    degrees = np.array(A_link.sum(axis=1)).ravel()
    L = sp.diags(degrees) - A_link
    L_dense = L.toarray()

    # Full eigendecomposition
    eigenvalues = gpu_eigvalsh(L_dense)
    lam_nz = eigenvalues[eigenvalues > 1e-8]
    n_zero = N - len(lam_nz)

    # Heat trace K(tau) — normalized by N
    if len(lam_nz) == 0:
        K = np.zeros(N_TAU)
    else:
        K = np.sum(np.exp(-TAU_GRID[:, None] * lam_nz[None, :]), axis=1) / N

    # GOE null (matched spectral range)
    rng_null = np.random.default_rng(seed_int + 77777)
    G = rng_null.standard_normal((N, N))
    G = (G + G.T) / 2.0
    eigs_goe = gpu_eigvalsh(G)
    eigs_goe = eigs_goe - eigs_goe[0]
    if np.max(eigs_goe) > 0 and len(lam_nz) > 0:
        eigs_goe = eigs_goe * (np.max(lam_nz) / np.max(eigs_goe))
    goe_nz = eigs_goe[eigs_goe > 1e-8]
    if len(goe_nz) == 0:
        K_goe = np.zeros(N_TAU)
    else:
        K_goe = np.sum(np.exp(-TAU_GRID[:, None] * goe_nz[None, :]), axis=1) / N

    # Per-sprinkling max
    t2K = TAU_GRID ** POWER * K
    intermed = (TAU_GRID > 0.01) & (TAU_GRID < 10)
    t2K_max_single = float(np.max(t2K[intermed])) if np.any(intermed) else 0.0

    return {
        "K": K.tolist(),
        "K_goe": K_goe.tolist(),
        "n_zero": n_zero,
        "lam_max": float(np.max(lam_nz)) if len(lam_nz) > 0 else 0.0,
        "lam_min_nz": float(np.min(lam_nz)) if len(lam_nz) > 0 else 0.0,
        "total_causal": total_causal,
        "bd_action": bd,
        "n_links": int(N0),
        "eps": eps,
        "t2K_max_single": t2K_max_single,
    }


# ---------------------------------------------------------------------------
# Plateau detection (from EXP-5)
# ---------------------------------------------------------------------------

def detect_plateau(t_power_K, tau_grid, deriv_thresh=0.02, min_decades=0.3):
    """Find plateau in tau^(d/2)*K(tau) using derivative threshold."""
    log_tau = np.log10(tau_grid)
    # Numerical derivative d(t^p*K)/d(log tau)
    deriv = np.gradient(t_power_K, log_tau)
    rel_deriv = np.abs(deriv) / (np.abs(t_power_K) + 1e-30)

    plateau_mask = rel_deriv < deriv_thresh
    if not np.any(plateau_mask):
        return None

    # Find longest contiguous plateau region
    indices = np.where(plateau_mask)[0]
    if len(indices) < 3:
        return None

    diffs = np.diff(indices)
    breaks = np.where(diffs > 1)[0]
    segments = np.split(indices, breaks + 1)
    longest = max(segments, key=len)

    if len(longest) < 3:
        return None

    tau_lo = tau_grid[longest[0]]
    tau_hi = tau_grid[longest[-1]]
    decades = np.log10(tau_hi / tau_lo)

    if decades < min_decades:
        return None

    values = t_power_K[longest]
    return {
        "value": float(np.mean(values)),
        "sem": float(np.std(values, ddof=1) / np.sqrt(len(values))),
        "ci_lo": float(np.mean(values) - 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))),
        "ci_hi": float(np.mean(values) + 1.96 * np.std(values, ddof=1) / np.sqrt(len(values))),
        "tau_lo": float(tau_lo),
        "tau_hi": float(tau_hi),
        "decades": float(decades),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    meta = ExperimentMeta(
        route=2, name="exp5b_d4_spectral_highN",
        description="d=4 spectral action heat trace: high-N extension (5000-10000)",
        N=max(N_VALUES), M=M_ENSEMBLE, status="running",
    )

    print("=" * 70, flush=True)
    print("FND-1 EXP-5b: d=4 SPECTRAL ACTION — HIGH N", flush=True)
    print("=" * 70, flush=True)
    print(f"N values: {N_VALUES}, M={M_ENSEMBLE}", flush=True)
    print(f"Coscosh eps: {EPS_COSCOSH}", flush=True)
    print(f"Quadrupole eps: {EPS_QUADRUPOLE}", flush=True)
    print(f"tau: [{TAU_MIN}, {TAU_MAX}], {N_TAU} points", flush=True)
    print(f"Workers: {WORKERS}", flush=True)
    print(flush=True)

    # Benchmark
    print("=== BENCHMARK ===", flush=True)
    for N in N_VALUES:
        t0 = time.perf_counter()
        _worker((42, N, T_DIAMOND, 0.0, "flat"))
        dt = time.perf_counter() - t0
        tasks = M_ENSEMBLE * (len(EPS_COSCOSH) + len(EPS_QUADRUPOLE))
        par = tasks * dt / WORKERS
        print(f"  N={N:6d}: {dt:.1f}s/task, {tasks} tasks, parallel: {par / 60:.1f} min",
              flush=True)

    ss = np.random.SeedSequence(MASTER_SEED)
    results_by_N = {}

    for N in N_VALUES:
        print(f"\n{'=' * 60}", flush=True)
        print(f"N = {N}", flush=True)
        print("=" * 60, flush=True)

        results_by_eps = {}

        for profile, eps_list in [("coscosh", EPS_COSCOSH), ("quadrupole", EPS_QUADRUPOLE)]:
            for eps in eps_list:
                eps_key = f"{profile}_{eps}"
                eps_ss = ss.spawn(1)[0]
                seeds = eps_ss.spawn(M_ENSEMBLE)
                seed_ints = [int(s.generate_state(1)[0]) for s in seeds]
                args = [(si, N, T_DIAMOND, eps, profile) for si in seed_ints]

                t0 = time.perf_counter()
                with Pool(WORKERS, initializer=_init_worker) as pool:
                    raw = pool.map(_worker, args)
                elapsed = time.perf_counter() - t0

                # Ensemble average
                K_mean = np.mean([r["K"] for r in raw], axis=0)
                K_goe_mean = np.mean([r["K_goe"] for r in raw], axis=0)

                t2K = TAU_GRID ** POWER * K_mean
                t2K_goe = TAU_GRID ** POWER * K_goe_mean

                # Plateau detection
                p_bd = detect_plateau(t2K, TAU_GRID)
                p_goe = detect_plateau(t2K_goe, TAU_GRID)

                intermed = (TAU_GRID > 0.01) & (TAU_GRID < 10)
                t2K_max = float(np.max(t2K[intermed])) if np.any(intermed) else 0.0

                # t2K curve for plotting (50 sampled points)
                plot_idx = np.linspace(0, N_TAU - 1, 50, dtype=int)
                t2K_curve = {f"{TAU_GRID[i]:.6g}": float(t2K[i]) for i in plot_idx}

                results_by_eps[eps_key] = {
                    "plateau_bd": p_bd,
                    "plateau_goe": p_goe,
                    "t2K_max_intermediate": t2K_max,
                    "t2K_curve": t2K_curve,
                    "mean_n_zero": float(np.mean([r["n_zero"] for r in raw])),
                    "mean_lam_max": float(np.mean([r["lam_max"] for r in raw])),
                    "_raw_t2K": [{"t2K_max": r["t2K_max_single"], "tc": r["total_causal"],
                                  "bd": r["bd_action"]} for r in raw],
                }

                p_val = p_bd["value"] if p_bd else 0
                print(f"  {eps_key}: plateau={p_val:.6f}  [{elapsed:.1f}s]", flush=True)

        # BD/GOE ratio at flat
        flat_key = "coscosh_0.0"
        if flat_key in results_by_eps:
            p_flat = results_by_eps[flat_key].get("plateau_bd")
            p_goe_flat = results_by_eps[flat_key].get("plateau_goe")
            if p_flat and p_goe_flat and p_goe_flat["value"] > 0:
                bd_goe = p_flat["value"] / p_goe_flat["value"]
            else:
                bd_goe = 0
        else:
            bd_goe = 0

        # Per-sprinkling curvature test (quadrupole only)
        all_eps_ps = []
        all_t2K_ps = []
        all_tc_ps = []
        all_bd_ps = []
        for ek, ev in results_by_eps.items():
            if not ek.startswith("quadrupole"):
                continue
            eps_val = float(ek.split("_")[1])
            for r_single in ev.get("_raw_t2K", []):
                all_eps_ps.append(eps_val)
                all_t2K_ps.append(r_single["t2K_max"])
                all_tc_ps.append(r_single["tc"])
                all_bd_ps.append(r_single["bd"])

        if len(all_eps_ps) > 10:
            eps_arr = np.array(all_eps_ps)
            t2K_arr = np.array(all_t2K_ps)
            tc_arr = np.array(all_tc_ps)
            bd_arr = np.array(all_bd_ps)

            from numpy.linalg import lstsq
            ctrl = np.column_stack([tc_arr, tc_arr ** 2, bd_arr, np.ones(len(tc_arr))])
            r_d, p_d = stats.pearsonr(eps_arr, t2K_arr)
            cx, _, _, _ = lstsq(ctrl, eps_arr, rcond=None)
            cy, _, _, _ = lstsq(ctrl, t2K_arr, rcond=None)
            rx = eps_arr - ctrl @ cx
            ry = t2K_arr - ctrl @ cy
            if np.std(rx) > 1e-15 and np.std(ry) > 1e-15:
                r_part, p_part = stats.pearsonr(rx, ry)
            else:
                r_part, p_part = 0.0, 1.0
        else:
            r_d = p_d = r_part = p_part = 0.0

        results_by_N[str(N)] = {
            "by_eps": results_by_eps,
            "bd_goe_ratio": float(bd_goe),
            "curvature_r_direct": float(r_d),
            "curvature_p_direct": float(p_d),
            "curvature_r_partial": float(r_part),
            "curvature_p_partial": float(p_part),
            "curvature_n_samples": len(all_eps_ps),
        }

        print(f"  BD/GOE ratio (flat): {bd_goe:.3f}", flush=True)
        print(f"  Curvature (quadrupole, n={len(all_eps_ps)}): "
              f"r_direct={r_d:+.4f} r_partial={r_part:+.4f} p={p_part:.2e}", flush=True)

    total_time = time.perf_counter() - t_total

    # Spectral dimension from highest N
    N_best = str(N_VALUES[-1])
    by_eps = results_by_N[N_best]["by_eps"]
    flat_k = [k for k in by_eps if k.endswith("_0.0") or k.endswith("_0")]
    if flat_k:
        curve = by_eps[flat_k[0]].get("t2K_curve", {})
        if curve:
            pairs = sorted([(float(k), v) for k, v in curve.items()])
            taus_c = np.array([p[0] for p in pairs])
            vals_c = np.array([p[1] for p in pairs])
            K_c = vals_c / np.maximum(taus_c ** POWER, 1e-30)
            mask = K_c > 1e-20
            if np.sum(mask) > 5:
                lt = np.log(taus_c[mask])
                lK = np.log(K_c[mask])
                d_S = np.zeros(len(lt))
                for i in range(1, len(lt) - 1):
                    d_S[i] = -2 * (lK[i + 1] - lK[i - 1]) / (lt[i + 1] - lt[i - 1])
                mid = (taus_c[mask] > 0.1) & (taus_c[mask] < 10)
                d_S_plateau = float(np.median(d_S[1:-1][mid[1:-1]])) if np.any(mid[1:-1]) else float(np.median(d_S[2:-2]))
            else:
                d_S_plateau = 0.0
        else:
            d_S_plateau = 0.0
    else:
        d_S_plateau = 0.0

    # Verdict
    r_best = results_by_N[N_best]
    verdict = (f"d_S={d_S_plateau:.1f} at N={N_VALUES[-1]} "
               f"(expected 4.0). "
               f"Curvature r_partial={r_best['curvature_r_partial']:+.4f}. "
               f"BD/GOE={r_best['bd_goe_ratio']:.2f}x.")

    print(f"\n{'=' * 70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"Wall time: {total_time:.0f}s ({total_time / 60:.1f} min)", flush=True)
    print("=" * 70, flush=True)

    meta.status = "completed"
    meta.verdict = verdict
    meta.wall_time_sec = total_time

    output = {
        "parameters": {
            "N_values": N_VALUES, "M": M_ENSEMBLE, "T": T_DIAMOND,
            "eps_coscosh": EPS_COSCOSH, "eps_quadrupole": EPS_QUADRUPOLE,
            "tau_range": [TAU_MIN, TAU_MAX, N_TAU],
            "d": D, "power": POWER,
        },
        "results_by_N": results_by_N,
        "spectral_dimension_at_Nmax": d_S_plateau,
        "verdict": verdict,
        "wall_time_sec": total_time,
    }

    out_path = RESULTS_DIR / "exp5b_d4_spectral_highN.json"
    save_experiment(meta, output, out_path)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
