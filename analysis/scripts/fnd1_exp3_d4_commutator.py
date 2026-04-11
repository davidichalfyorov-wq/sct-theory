"""
FND-1 EXP-3: d=4 Commutator [H,M] Curvature Detection.

In d=2, [H,M] failed (conformal invariance, all signal mediated by TC+BD).
In d=4, causal structure carries Weyl information. Tests whether the
commutator of the symmetrized/antisymmetrized d=4 BD operator detects
pp-wave curvature after polynomial mediation.

Key difference from d=2: uses d=4 BD coefficients (4, -36, 64, -32)
and rho^{1/2} scaling (not rho as in d=2).

Tests both coscosh (monopole+Weyl) and quadrupole (pure Weyl, TC stable).
Tests both linear (eps) and quadratic (eps^2) predictors.

Run:
  python analysis/scripts/fnd1_exp3_d4_commutator.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_4d_experiment import (
    sprinkle_4d_flat,
    causal_matrix_4d,
    compute_layers_4d,
    bd_action_4d,
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

N_VALUES = [500, 1000, 2000]
N_PRIMARY = 2000
M_ENSEMBLE = 100  # power analysis: d=0.28 needs M>=80 for detection at p<0.01
T_DIAMOND = 1.0
MASTER_SEED = 7733
WORKERS = N_WORKERS

EPS_COSCOSH = [0.0, 0.1, 0.2, 0.3, 0.5]
# TC-stable range: eps<=10 gives <3% TC change. eps=20,40 gives 9-25% TC change
# (mediation controls for this, but cleaner signal at smaller eps).
EPS_QUADRUPOLE = [0.0, 2.0, 5.0, 10.0, 20.0]


# ---------------------------------------------------------------------------
# d=4 BD operator
# ---------------------------------------------------------------------------

def build_bd_L_4d(C, n_matrix, rho):
    """Build d=4 BD off-diagonal operator (lower-triangular).

    Uses d=4 BD coefficients from S_BD = (-4N + 4N0 - 36N1 + 64N2 - 32N3)/sqrt(6).
    Scaling: rho^{2/d} = rho^{1/2} for d=4.
    """
    past = C.T
    n_past = n_matrix.T
    n_int = np.rint(n_past).astype(np.int64)
    causal_mask = past > 0.5

    scale = np.sqrt(rho)
    N = C.shape[0]
    L = np.zeros((N, N), dtype=np.float64)
    L[causal_mask & (n_int == 0)] = 4.0 * scale
    L[causal_mask & (n_int == 1)] = -36.0 * scale
    L[causal_mask & (n_int == 2)] = 64.0 * scale
    L[causal_mask & (n_int == 3)] = -32.0 * scale

    return L


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(args):
    """Compute [H,M] commutator observables for one d=4 sprinkling."""
    seed_int, N, T, eps, profile = args

    rng = np.random.default_rng(seed_int)
    pts = sprinkle_4d_flat(N, T, rng)

    # Build causal matrix with specified profile
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

    total_causal = float(np.sum(C))
    V = np.pi * T ** 4 / 24.0
    rho = N / V

    # Layers + BD
    n_matrix, N0, N1, N2, N3 = compute_layers_4d(C)
    bd = bd_action_4d(N, N0, N1, N2, N3)

    # d=4 BD operator
    L = build_bd_L_4d(C, n_matrix, rho)

    # Commutator: [H,M] = (L^T L - L L^T) / 2  (optimized sparse)
    L_sp = sp.csr_matrix(L)
    comm = ((L_sp.T @ L_sp - L_sp @ L_sp.T) / 2.0).toarray()
    comm = (comm + comm.T) / 2.0  # enforce exact symmetry (sparse rounding)
    comm_eigs = gpu_eigvalsh(comm)

    # Observables
    a = np.abs(comm_eigs)
    s_total = float(np.sum(a))
    if s_total > 0:
        p = a / s_total
        entropy = float(-np.sum(p * np.log(p + 1e-300)))
    else:
        entropy = 0.0

    frobenius = float(np.sqrt(np.sum(comm_eigs ** 2)))
    max_abs = float(np.max(a))

    return {
        "comm_frobenius": frobenius,
        "comm_entropy": entropy,
        "comm_max_abs": max_abs,
        "total_causal": total_causal,
        "bd_action": bd,
        "n_links": N0,
        "eps": eps,
        "profile": profile,
    }


# ---------------------------------------------------------------------------
# Mediation (reused from EXP-1 pattern)
# ---------------------------------------------------------------------------

def partial_corr(x, y, controls):
    """Partial Pearson r controlling for columns of controls."""
    from numpy.linalg import lstsq
    if controls.shape[1] == 0:
        return stats.pearsonr(x, y)
    cx, _, _, _ = lstsq(controls, x, rcond=None)
    cy, _, _, _ = lstsq(controls, y, rcond=None)
    rx = x - controls @ cx
    ry = y - controls @ cy
    if np.std(rx) < 1e-15 or np.std(ry) < 1e-15:
        return 0.0, 1.0
    return stats.pearsonr(rx, ry)


def mediation_one_obs(eps_arr, obs_arr, tc_arr, bd_arr):
    """Partial r for one observable vs eps (linear + quadratic)."""
    controls = np.column_stack([tc_arr, tc_arr ** 2, bd_arr, np.ones(len(tc_arr))])
    out = {}
    for pred_name, pred in [("linear", eps_arr), ("quadratic", eps_arr ** 2)]:
        r_d, p_d = stats.pearsonr(pred, obs_arr)
        r_p, p_p = partial_corr(pred, obs_arr, controls)
        out[f"{pred_name}_r_direct"] = float(r_d)
        out[f"{pred_name}_r_partial"] = float(r_p)
        out[f"{pred_name}_p_partial"] = float(p_p)

    if abs(out["linear_r_partial"]) >= abs(out["quadratic_r_partial"]):
        out["best"] = "linear"
        out["best_r_partial"] = out["linear_r_partial"]
        out["best_p_partial"] = out["linear_p_partial"]
    else:
        out["best"] = "quadratic"
        out["best_r_partial"] = out["quadratic_r_partial"]
        out["best_p_partial"] = out["quadratic_p_partial"]
    return out


def mediation(results):
    """Test ALL three observables vs eps, controlling TC+TC^2+BD.
    Reports best observable (frobenius, entropy, or max_abs)."""
    eps_arr = np.array([r["eps"] for r in results])
    tc_arr = np.array([r["total_causal"] for r in results])
    bd_arr = np.array([r["bd_action"] for r in results])

    out = {"n": len(results)}
    best_abs_r = 0.0
    best_obs = "none"

    for obs_name in ["comm_frobenius", "comm_entropy", "comm_max_abs"]:
        obs_arr = np.array([r[obs_name] for r in results])
        med = mediation_one_obs(eps_arr, obs_arr, tc_arr, bd_arr)
        out[obs_name] = med
        if abs(med["best_r_partial"]) > best_abs_r:
            best_abs_r = abs(med["best_r_partial"])
            best_obs = obs_name

    out["best_observable"] = best_obs
    out["best_r_partial"] = float(out[best_obs]["best_r_partial"])
    out["best_p_partial"] = float(out[best_obs]["best_p_partial"])
    out["best"] = out[best_obs]["best"]

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    meta = ExperimentMeta(
        route=3, name="exp3_d4_commutator",
        description="d=4 commutator [H,M]: Weyl curvature detection via BD operator",
        N=N_PRIMARY, M=M_ENSEMBLE, status="running",
    )

    print("=" * 70, flush=True)
    print("FND-1 EXP-3: d=4 COMMUTATOR [H,M]", flush=True)
    print("=" * 70, flush=True)
    print(f"N values: {N_VALUES}, M={M_ENSEMBLE}", flush=True)
    print(f"Coscosh eps: {EPS_COSCOSH}", flush=True)
    print(f"Quadrupole eps: {EPS_QUADRUPOLE}", flush=True)
    print(flush=True)

    # Benchmark
    print("=== BENCHMARK ===", flush=True)
    for N in N_VALUES:
        t0 = time.perf_counter()
        _worker((42, N, T_DIAMOND, 0.0, "coscosh"))
        dt = time.perf_counter() - t0
        print(f"  N={N:5d}: {dt:.3f}s/task", flush=True)

    ss = np.random.SeedSequence(MASTER_SEED)
    all_results = {}

    for prof_name, eps_list in [("coscosh", EPS_COSCOSH),
                                 ("quadrupole", EPS_QUADRUPOLE)]:
        print(f"\n{'=' * 70}", flush=True)
        print(f"PROFILE: {prof_name}", flush=True)
        print("=" * 70, flush=True)

        for N in N_VALUES:
            results_N = []
            for eps in eps_list:
                eps_ss = ss.spawn(1)[0]
                seeds = eps_ss.spawn(M_ENSEMBLE)
                seed_ints = [int(s.generate_state(1)[0]) for s in seeds]
                args = [(si, N, T_DIAMOND, eps, prof_name) for si in seed_ints]

                t0 = time.perf_counter()
                with Pool(WORKERS, initializer=_init_worker) as pool:
                    raw = pool.map(_worker, args)
                elapsed = time.perf_counter() - t0

                frobs = [r["comm_frobenius"] for r in raw]
                print(f"  {prof_name} N={N} eps={eps:+.1f}: "
                      f"frob={np.mean(frobs):.1f}+-{np.std(frobs):.1f}"
                      f"  [{elapsed:.1f}s]", flush=True)
                results_N.extend(raw)

            # Mediation
            med = mediation(results_N)
            key = f"{prof_name}_N{N}"
            all_results[key] = {
                "mediation": med,
                "n_sprinklings": len(results_N),
            }
            print(f"  Mediation: best={med['best_observable']}"
                  f" ({med['best']})"
                  f" r_partial={med['best_r_partial']:+.4f}"
                  f" p={med['best_p_partial']:.2e}", flush=True)

    # ==================================================================
    # VERDICT
    # ==================================================================

    total_time = time.perf_counter() - t_total

    # Primary: quadrupole at N_PRIMARY (pure Weyl)
    quad_key = f"quadrupole_N{N_PRIMARY}"
    med_q = all_results[quad_key]["mediation"]
    cosc_key = f"coscosh_N{N_PRIMARY}"
    med_c = all_results[cosc_key]["mediation"]

    print(f"\n{'=' * 70}", flush=True)
    print(f"COMPARISON at N={N_PRIMARY}:", flush=True)
    print(f"  Coscosh:    r_partial={med_c['best_r_partial']:+.4f}"
          f" (p={med_c['best_p_partial']:.3f}, {med_c['best']})", flush=True)
    print(f"  Quadrupole: r_partial={med_q['best_r_partial']:+.4f}"
          f" (p={med_q['best_p_partial']:.3f}, {med_q['best']})", flush=True)

    if abs(med_q["best_r_partial"]) > 0.10 and med_q["best_p_partial"] < 0.01:
        verdict = (f"GENUINE (quadrupole): r_partial={med_q['best_r_partial']:+.4f},"
                   f" p={med_q['best_p_partial']:.2e}")
    elif abs(med_c["best_r_partial"]) > 0.10 and med_c["best_p_partial"] < 0.01:
        verdict = (f"COSCOSH ONLY: r_partial={med_c['best_r_partial']:+.4f},"
                   f" quadrupole={med_q['best_r_partial']:+.4f}")
    else:
        verdict = (f"NO SIGNAL: coscosh r={med_c['best_r_partial']:+.4f},"
                   f" quadrupole r={med_q['best_r_partial']:+.4f}")

    print(f"\n{'=' * 70}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print(f"Wall time: {total_time:.0f}s ({total_time / 60:.1f} min)", flush=True)
    print("=" * 70, flush=True)

    meta.status = "completed"
    meta.verdict = verdict
    meta.wall_time_sec = total_time

    output = {
        "parameters": {
            "N_values": N_VALUES, "N_primary": N_PRIMARY,
            "M": M_ENSEMBLE, "T": T_DIAMOND,
            "eps_coscosh": EPS_COSCOSH, "eps_quadrupole": EPS_QUADRUPOLE,
        },
        "results": all_results,
        "verdict": verdict,
        "wall_time_sec": total_time,
    }

    out_path = RESULTS_DIR / "exp3_d4_commutator.json"
    save_experiment(meta, output, out_path)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
