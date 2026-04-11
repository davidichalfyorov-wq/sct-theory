"""
FND-1 EXP-5: d=4 Spectral Action from Link-Graph Laplacian.

Computes the heat trace K(tau) from the link-graph Laplacian eigenvalues
in d=4. In d=4, the Weyl law predicts:

  tau^2 * K(tau) -> C_4  (plateau)

where C_4 = V/(4*pi)^2 per-eigenvalue, or some dimension-dependent constant.

Also tests: does the subleading term in K(tau) change with pp-wave curvature?
If yes, the link-graph Laplacian heat trace encodes SDW coefficients in d=4.

Uses the same plateau detection and GOE null as EXP-7.

Run:
  python analysis/scripts/fnd1_exp5_d4_spectral_action.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_4d_experiment import (
    sprinkle_4d_flat,
    causal_matrix_4d,
    compute_layers_4d,
    bd_action_4d,
    build_link_graph,
    _ppwave_profile,
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
# Parameters
# ---------------------------------------------------------------------------

N_VALUES = [500, 1000, 2000, 3000]
M_ENSEMBLE = 60
T_DIAMOND = 1.0
MASTER_SEED = 577
WORKERS = N_WORKERS

# Heat trace
N_TAU = 200
TAU_MIN = 1e-4
TAU_MAX = 100.0
TAU_GRID = np.logspace(np.log10(TAU_MIN), np.log10(TAU_MAX), N_TAU)

# d=4 leading behavior: K(tau) ~ C/tau^2, so tau^2 * K should plateau
D = 4
POWER = D / 2.0  # tau^2 * K(tau) -> const

# Coscosh (Weyl + monopole) and quadrupole (pure Weyl, TC-stable at eps<=10)
EPS_COSCOSH = [0.0, 0.2, 0.5]
EPS_QUADRUPOLE = [0.0, 5.0, 10.0]

# Plateau detection
PLATEAU_DERIV_THRESHOLD = 0.03
PLATEAU_MIN_DECADES = 0.3


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(args):
    """Compute link-graph Laplacian eigenvalues and heat trace in d=4."""
    seed_int, N, T, eps, profile = args
    import scipy.sparse as sp

    rng = np.random.default_rng(seed_int)
    pts = sprinkle_4d_flat(N, T, rng)

    if abs(eps) < 1e-12:
        C = causal_matrix_4d(pts, 0.0, "flat")
    else:
        t = pts[:, 0]; x = pts[:, 1]; y = pts[:, 2]; z = pts[:, 3]
        dt = t[np.newaxis, :] - t[:, np.newaxis]
        dx = x[np.newaxis, :] - x[:, np.newaxis]
        dy = y[np.newaxis, :] - y[:, np.newaxis]
        dz = z[np.newaxis, :] - z[:, np.newaxis]
        dr2 = dx ** 2 + dy ** 2 + dz ** 2
        mink = dt ** 2 - dr2
        xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
        ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
        if profile == "quadrupole":
            f_mid = quadrupole_profile(xm, ym)
        else:
            f_mid = _ppwave_profile(xm, ym)
        corr = eps * f_mid * (dt + dz) ** 2 / 2.0
        C = ((mink > corr) & (dt > 0)).astype(np.float64)

    n_matrix, N0, N1, N2, N3 = compute_layers_4d(C)
    total_causal = float(np.sum(C))

    # Link graph + Laplacian
    A_link = build_link_graph(C, n_matrix)
    degrees = np.array(A_link.sum(axis=1)).ravel()
    L = sp.diags(degrees) - A_link
    L_dense = L.toarray()

    # Full eigendecomposition
    eigenvalues = gpu_eigvalsh(L_dense)

    # Filter nonzero eigenvalues (skip near-zero modes)
    lam_nz = eigenvalues[eigenvalues > 1e-8]
    n_zero = N - len(lam_nz)

    if len(lam_nz) == 0:
        K = np.zeros(N_TAU)
    else:
        # Normalize by N (total eigenvalues), not len(lam_nz) (nonzero only)
        K = np.sum(np.exp(-TAU_GRID[:, None] * lam_nz[None, :]), axis=1) / N

    # GOE null (matched spectral range)
    rng_null = np.random.default_rng(seed_int + 77777)
    G = rng_null.standard_normal((N, N))
    G = (G + G.T) / 2.0
    eigs_goe = gpu_eigvalsh(G)
    # Make PSD like Laplacian: shift so min eigenvalue = 0
    eigs_goe = eigs_goe - eigs_goe[0]
    # Rescale to match link Laplacian spectral range
    if np.max(eigs_goe) > 0 and len(lam_nz) > 0:
        eigs_goe = eigs_goe * (np.max(lam_nz) / np.max(eigs_goe))
    goe_nz = eigs_goe[eigs_goe > 1e-8]
    n_zero_goe = N - len(goe_nz)
    if len(goe_nz) == 0:
        K_goe = np.zeros(N_TAU)
    else:
        K_goe = np.sum(np.exp(-TAU_GRID[:, None] * goe_nz[None, :]), axis=1) / N

    # Per-sprinkling tau^2*K max in intermediate range (for curvature test)
    t2K_single = TAU_GRID ** POWER * K
    intermed_mask = (TAU_GRID > 0.01) & (TAU_GRID < 10)
    t2K_max_single = float(np.max(t2K_single[intermed_mask])) if np.any(intermed_mask) else 0.0

    bd = bd_action_4d(N, N0, N1, N2, N3)

    return {
        "K": K.tolist(),
        "K_goe": K_goe.tolist(),
        "n_zero": n_zero,
        "n_zero_goe": n_zero_goe,
        "lam_max": float(np.max(lam_nz)) if len(lam_nz) > 0 else 0.0,
        "lam_min_nz": float(np.min(lam_nz)) if len(lam_nz) > 0 else 0.0,
        "total_causal": total_causal,
        "bd_action": bd,
        "n_links": int(N0),
        "eps": eps,
        "t2K_max_single": t2K_max_single,
    }


# ---------------------------------------------------------------------------
# Plateau detection (adapted from EXP-7 for tau^2 * K)
# ---------------------------------------------------------------------------

def find_plateau_d4(tau_grid, tpK_mean, tpK_sem):
    """Find plateau in tau^p * K(tau) where p = d/2 = 2."""
    log_tau = np.log10(tau_grid)
    d_tpK = np.gradient(tpK_mean, log_tau)

    is_flat = (np.abs(d_tpK) < PLATEAU_DERIV_THRESHOLD) & (tpK_mean > 1e-4)
    if not np.any(is_flat):
        return None

    changes = np.diff(is_flat.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1
    if is_flat[0]:
        starts = np.concatenate([[0], starts])
    if is_flat[-1]:
        ends = np.concatenate([ends, [len(is_flat)]])
    if len(starts) == 0 or len(ends) == 0:
        return None

    best_width = 0
    best_start = best_end = 0
    for s, e in zip(starts, ends):
        width = log_tau[min(e, len(log_tau) - 1)] - log_tau[s]
        if width > best_width:
            best_width = width
            best_start, best_end = s, e

    if best_width < PLATEAU_MIN_DECADES:
        return None

    sl = slice(best_start, best_end)
    plateau_vals = tpK_mean[sl]
    plateau_sems = tpK_sem[sl]

    value = float(np.mean(plateau_vals))
    sem = float(np.sqrt(np.mean(plateau_sems ** 2) + np.var(plateau_vals, ddof=1)))

    return {
        "value": value,
        "sem": sem,
        "ci_lo": value - 2.0 * sem,
        "ci_hi": value + 2.0 * sem,
        "tau_lo": float(tau_grid[best_start]),
        "tau_hi": float(tau_grid[min(best_end, len(tau_grid) - 1)]),
        "decades": best_width,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    meta = ExperimentMeta(
        route=2, name="exp5_d4_spectral_action",
        description="d=4 spectral action: heat trace from link-graph Laplacian",
        N=max(N_VALUES), M=M_ENSEMBLE, status="running",
    )

    print("=" * 70, flush=True)
    print("FND-1 EXP-5: d=4 SPECTRAL ACTION (LINK-GRAPH LAPLACIAN)", flush=True)
    print("=" * 70, flush=True)
    print(f"N values: {N_VALUES}, M={M_ENSEMBLE}", flush=True)
    print(f"Coscosh eps: {EPS_COSCOSH}", flush=True)
    print(f"Quadrupole eps: {EPS_QUADRUPOLE}", flush=True)
    print(f"Testing: tau^2 * K(tau) plateau (Weyl law for d=4)", flush=True)
    print(flush=True)

    # Benchmark
    print("=== BENCHMARK ===", flush=True)
    for N in N_VALUES:
        t0 = time.perf_counter()
        _worker((42, N, T_DIAMOND, 0.0, "coscosh"))
        dt = time.perf_counter() - t0
        tasks = M_ENSEMBLE * (len(EPS_COSCOSH) + len(EPS_QUADRUPOLE))
        par = tasks * dt / WORKERS
        print(f"  N={N:5d}: {dt:.3f}s/task, total {tasks} tasks,"
              f" parallel: {par / 60:.1f} min", flush=True)

    ss = np.random.SeedSequence(MASTER_SEED)
    results_by_N = {}

    for N in N_VALUES:
        print(f"\n{'=' * 60}", flush=True)
        print(f"N = {N}", flush=True)
        print("=" * 60, flush=True)

        results_by_eps = {}
        all_eps_list = ([(e, "coscosh") for e in EPS_COSCOSH] +
                        [(e, "quadrupole") for e in EPS_QUADRUPOLE if e > 0])
        for eps, profile in all_eps_list:
            eps_ss = ss.spawn(1)[0]
            child_seeds = eps_ss.spawn(M_ENSEMBLE)
            seed_ints = [int(cs.generate_state(1)[0]) for cs in child_seeds]
            args = [(si, N, T_DIAMOND, eps, profile) for si in seed_ints]

            t0 = time.perf_counter()
            with Pool(WORKERS, initializer=_init_worker) as pool:
                raw = pool.map(_worker, args)
            elapsed = time.perf_counter() - t0

            # Ensemble average
            K_arr = np.array([r["K"] for r in raw])
            K_mean = np.mean(K_arr, axis=0)
            K_sem = np.std(K_arr, axis=0, ddof=1) / np.sqrt(M_ENSEMBLE)

            K_goe_arr = np.array([r["K_goe"] for r in raw])
            K_goe_mean = np.mean(K_goe_arr, axis=0)
            K_goe_sem = np.std(K_goe_arr, axis=0, ddof=1) / np.sqrt(M_ENSEMBLE)

            # tau^2 * K
            t2K = TAU_GRID ** POWER * K_mean
            t2K_sem = TAU_GRID ** POWER * K_sem
            t2K_goe = TAU_GRID ** POWER * K_goe_mean
            t2K_goe_sem = TAU_GRID ** POWER * K_goe_sem

            p_bd = find_plateau_d4(TAU_GRID, t2K, t2K_sem)
            p_goe = find_plateau_d4(TAU_GRID, t2K_goe, t2K_goe_sem)

            # Intermediate max
            intermed = (TAU_GRID > 0.01) & (TAU_GRID < 10)
            t2K_max = float(np.max(t2K[intermed])) if np.any(intermed) else 0.0

            bd_str = (f"plateau={p_bd['value']:.6f}" if p_bd
                      else f"max={t2K_max:.6f}")
            goe_str = (f"{p_goe['value']:.6f}" if p_goe
                       else f"{float(np.max(t2K_goe[intermed])) if np.any(intermed) else 0:.6f}")

            eps_key = f"{profile}_{eps}"
            print(f"  {profile} eps={eps:+.2f}: BD {bd_str}  GOE {goe_str}"
                  f"  [{elapsed:.1f}s]", flush=True)

            # Save 50-point curves
            plot_idx = np.unique(np.linspace(0, N_TAU - 1, 50, dtype=int))
            t2K_curve = {f"{TAU_GRID[i]:.6g}": round(float(t2K[i]), 8)
                         for i in plot_idx}

            # GOE intermediate max (stored per-eps to avoid stale variable bug)
            t2K_goe_max = float(np.max(t2K_goe[intermed])) if np.any(intermed) else 0.0

            # Per-sprinkling data for curvature test
            raw_t2K = [{"t2K_max": r["t2K_max_single"], "tc": r["total_causal"],
                        "bd": r["bd_action"]} for r in raw]

            results_by_eps[eps_key] = {
                "plateau_bd": p_bd,
                "plateau_goe": p_goe,
                "t2K_max_intermediate": t2K_max,
                "t2K_goe_max_intermediate": t2K_goe_max,
                "t2K_curve": t2K_curve,
                "mean_n_zero": float(np.mean([r["n_zero"] for r in raw])),
                "mean_lam_max": float(np.mean([r["lam_max"] for r in raw])),
                "_raw_t2K": raw_t2K,  # for per-sprinkling curvature test
            }

        # BD/GOE ratio at flat — use stored per-eps data (no stale variables)
        flat_r = results_by_eps.get("coscosh_0.0", results_by_eps.get("coscosh_0", {}))
        if flat_r["plateau_bd"] and flat_r["plateau_goe"]:
            ratio = flat_r["plateau_bd"]["value"] / flat_r["plateau_goe"]["value"]
        else:
            goe_ref = flat_r["t2K_goe_max_intermediate"]
            ratio = flat_r["t2K_max_intermediate"] / max(goe_ref, 1e-10)

        # Curvature test: per-sprinkling t2K_max vs eps (coscosh only for clean comparison)
        all_eps_per_sprinkling = []
        all_t2K_per_sprinkling = []
        all_tc_per_sprinkling = []
        all_bd_per_sprinkling = []
        for ek, ev in results_by_eps.items():
            if not ek.startswith("coscosh"):
                continue
            eps_val = float(ek.split("_")[1])
            for r_single in ev.get("_raw_t2K", []):
                all_eps_per_sprinkling.append(eps_val)
                all_t2K_per_sprinkling.append(r_single["t2K_max"])
                all_tc_per_sprinkling.append(r_single["tc"])
                all_bd_per_sprinkling.append(r_single["bd"])

        if len(all_eps_per_sprinkling) > 10:
            eps_arr_ps = np.array(all_eps_per_sprinkling)
            t2K_arr_ps = np.array(all_t2K_per_sprinkling)
            tc_arr_ps = np.array(all_tc_per_sprinkling)
            bd_arr_ps = np.array(all_bd_per_sprinkling)

            # Test BOTH linear (eps) and quadratic (eps^2) because
            # pp-wave Riemann ~ eps (linear), Kretschner ~ eps^2
            from numpy.linalg import lstsq
            ctrl = np.column_stack([tc_arr_ps, tc_arr_ps ** 2, bd_arr_ps, np.ones(len(tc_arr_ps))])

            curv_tests = {}
            for pred_name, predictor in [("linear", eps_arr_ps),
                                         ("quadratic", eps_arr_ps ** 2)]:
                r_d, p_d = stats.pearsonr(predictor, t2K_arr_ps)
                cx, _, _, _ = lstsq(ctrl, predictor, rcond=None)
                cy, _, _, _ = lstsq(ctrl, t2K_arr_ps, rcond=None)
                rx = predictor - ctrl @ cx
                ry = t2K_arr_ps - ctrl @ cy
                if np.std(rx) > 1e-15 and np.std(ry) > 1e-15:
                    r_p, p_p = stats.pearsonr(rx, ry)
                else:
                    r_p = p_p = float("nan")
                curv_tests[pred_name] = {
                    "r_direct": float(r_d), "p_direct": float(p_d),
                    "r_partial": float(r_p), "p_partial": float(p_p),
                }

            # Best predictor
            if abs(curv_tests["linear"]["r_partial"]) >= abs(curv_tests["quadratic"]["r_partial"]):
                r_curv_partial = curv_tests["linear"]["r_partial"]
                p_curv_partial = curv_tests["linear"]["p_partial"]
            else:
                r_curv_partial = curv_tests["quadratic"]["r_partial"]
                p_curv_partial = curv_tests["quadratic"]["p_partial"]
            r_curv = curv_tests["linear"]["r_direct"]
            p_curv = curv_tests["linear"]["p_direct"]
        else:
            r_curv = p_curv = r_curv_partial = p_curv_partial = float("nan")
            curv_tests = {}

        print(f"  BD/GOE ratio (flat): {ratio:.3f}", flush=True)
        print(f"  Curvature (per-sprinkling, n={len(all_eps_per_sprinkling)}):"
              f" r_direct={r_curv:+.4f} p={p_curv:.3f}"
              f" r_partial={r_curv_partial:+.4f} p={p_curv_partial:.3f}",
              flush=True)

        # Remove _raw_t2K from saved JSON (too large)
        save_eps = {}
        for ek, ev in results_by_eps.items():
            save_eps[ek] = {k: v for k, v in ev.items() if k != "_raw_t2K"}

        results_by_N[str(N)] = {
            "by_eps": save_eps,
            "bd_goe_ratio": ratio,
            "curvature_r_direct": float(r_curv),
            "curvature_p_direct": float(p_curv),
            "curvature_r_partial": float(r_curv_partial),
            "curvature_p_partial": float(p_curv_partial),
            "curvature_n_samples": len(all_eps_per_sprinkling),
        }

    # ==================================================================
    # VERDICT
    # ==================================================================

    total_time = time.perf_counter() - t_total
    best_N = str(N_VALUES[-1])
    rb = results_by_N[best_N]
    flat_p = rb["by_eps"].get("coscosh_0.0", rb["by_eps"].get("coscosh_0", {})).get("plateau_bd")

    if flat_p and flat_p["value"] > 1e-4:
        verdict = (f"PLATEAU FOUND: tau^2*K = {flat_p['value']:.6f} at N={N_VALUES[-1]}"
                   f" | BD/GOE={rb['bd_goe_ratio']:.2f}x")
    else:
        verdict = f"NO PLATEAU at N={N_VALUES[-1]}"

    if abs(rb["curvature_r_partial"]) > 0.10 and rb["curvature_p_partial"] < 0.01:
        verdict += (f" | CURVATURE SENSITIVE (r_partial={rb['curvature_r_partial']:+.3f},"
                    f" n={rb['curvature_n_samples']})")
    elif abs(rb["curvature_r_direct"]) > 0.3 and rb["curvature_p_direct"] < 0.01:
        verdict += (f" | DIRECT ONLY (r={rb['curvature_r_direct']:+.3f},"
                    f" partial={rb['curvature_r_partial']:+.3f})")
    else:
        verdict += f" | curvature r_partial={rb['curvature_r_partial']:+.3f} (not significant)"

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
            "eps_coscosh": EPS_COSCOSH,
            "eps_quadrupole": EPS_QUADRUPOLE,
            "tau_range": [TAU_MIN, TAU_MAX, N_TAU],
            "d": D, "power": POWER,
        },
        "results_by_N": results_by_N,
        "verdict": verdict,
        "wall_time_sec": total_time,
    }

    out_path = RESULTS_DIR / "exp5_d4_spectral_action.json"
    save_experiment(meta, output, out_path)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
