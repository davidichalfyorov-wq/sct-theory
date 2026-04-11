"""
FND-1 EXP-2: d=4 Sorkin-Johnston Vacuum — Signal Verification.

The original fnd1_sj_vacuum.py found 5 EXPLORATORY signals surviving
polynomial TC mediation (primary spectral_gap_ratio failed):
  1. spectral_width: r_poly=+0.55 (strongest)
  2. trace_trunc: r_poly=-0.37
  3. lambda_max: r_poly=+0.38
  4. entropy_trunc: r_poly=-0.36
  5. trace_W: r_poly=-0.25

This experiment verifies these signals with:
  - Different seeds (reproducibility)
  - Quadrupole profile (pure Weyl, TC-stable at eps<=10)
  - Linear + quadratic predictors (Riemann ~ eps)
  - Finite-size scaling (N=1000, 2000, 3000)
  - Per-sprinkling TC+TC^2+BD mediation

Run:
  python analysis/scripts/fnd1_exp2_d4_sj_verification.py
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
    _ppwave_profile,
)
from fnd1_4d_followup import quadrupole_profile
from fnd1_sj_vacuum import build_sj_wightman, sj_observables
from fnd1_experiment_registry import (
    ExperimentMeta, save_experiment, RESULTS_DIR,
)
from fnd1_parallel import N_WORKERS, _init_worker

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_VALUES = [1000, 2000, 3000]
N_PRIMARY = 3000
M_ENSEMBLE = 80
T_DIAMOND = 1.0
MASTER_SEED = 8811
WORKERS = N_WORKERS

EPS_COSCOSH = [0.0, 0.1, 0.2, 0.3, 0.5]
# TC-stable range for quadrupole (eps<=10 gives <3% TC change)
EPS_QUADRUPOLE = [0.0, 2.0, 5.0, 10.0, 20.0]

# The 5 signals to verify (from original experiment) + filtered width diagnostic
VERIFY_OBSERVABLES = [
    "spectral_width", "spectral_width_filtered",
    "trace_trunc", "lambda_max",
    "entropy_trunc", "trace_W",
]


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(args):
    """Build SJ vacuum and extract observables for one d=4 sprinkling."""
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

    n_matrix, N0, N1, N2, N3 = compute_layers_4d(C)
    bd = bd_action_4d(N, N0, N1, N2, N3)

    # Build SJ Wightman function
    pos_evals, all_evals = build_sj_wightman(C, n_matrix, rho)
    obs = sj_observables(pos_evals, all_evals, N)

    # Additional: filtered spectral_width (remove near-zero eigenvalue contamination)
    # Near-zero evals (< 0.01) dominate unfiltered std(log(evals)) via log outliers
    if len(pos_evals) > 0:
        filtered = pos_evals[pos_evals > 0.01]
        if len(filtered) > 1:
            obs["spectral_width_filtered"] = float(np.std(np.log(filtered)))
        else:
            obs["spectral_width_filtered"] = 0.0
        obs["n_near_zero"] = int(np.sum(pos_evals < 0.01))
    else:
        obs["spectral_width_filtered"] = 0.0
        obs["n_near_zero"] = 0

    obs["total_causal"] = total_causal
    obs["bd_action"] = bd
    obs["n_links"] = N0
    obs["eps"] = eps
    obs["profile"] = profile

    return obs


# ---------------------------------------------------------------------------
# Mediation
# ---------------------------------------------------------------------------

def partial_corr(x, y, controls):
    """Partial Pearson r."""
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


def mediation(results):
    """Test all 5 target observables vs eps, controlling TC+TC^2+BD.
    Tests both linear and quadratic predictors."""
    eps_arr = np.array([r["eps"] for r in results])
    tc_arr = np.array([r["total_causal"] for r in results])
    bd_arr = np.array([r["bd_action"] for r in results])
    controls = np.column_stack([tc_arr, tc_arr ** 2, bd_arr, np.ones(len(results))])

    out = {"n": len(results)}
    best_abs_r = 0.0
    best_obs = "none"

    for obs_name in VERIFY_OBSERVABLES:
        obs_arr = np.array([r[obs_name] for r in results])

        med = {}
        for pred_name, pred in [("linear", eps_arr), ("quadratic", eps_arr ** 2)]:
            r_d, p_d = stats.pearsonr(pred, obs_arr)
            r_p, p_p = partial_corr(pred, obs_arr, controls)
            med[f"{pred_name}_r_direct"] = float(r_d)
            med[f"{pred_name}_r_partial"] = float(r_p)
            med[f"{pred_name}_p_partial"] = float(p_p)

        if abs(med["linear_r_partial"]) >= abs(med["quadratic_r_partial"]):
            med["best"] = "linear"
            med["best_r_partial"] = med["linear_r_partial"]
            med["best_p_partial"] = med["linear_p_partial"]
        else:
            med["best"] = "quadratic"
            med["best_r_partial"] = med["quadratic_r_partial"]
            med["best_p_partial"] = med["quadratic_p_partial"]

        out[obs_name] = med

        if abs(med["best_r_partial"]) > best_abs_r:
            best_abs_r = abs(med["best_r_partial"])
            best_obs = obs_name

    out["best_observable"] = best_obs
    out["best_r_partial"] = float(out[best_obs]["best_r_partial"])
    out["best_p_partial"] = float(out[best_obs]["best_p_partial"])
    out["best_predictor"] = out[best_obs]["best"]

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    meta = ExperimentMeta(
        route=2, name="exp2_d4_sj_verification",
        description="d=4 SJ vacuum: verify 5 exploratory signals with quadrupole + mediation",
        N=N_PRIMARY, M=M_ENSEMBLE, status="running",
    )

    print("=" * 70, flush=True)
    print("FND-1 EXP-2: d=4 SJ VACUUM SIGNAL VERIFICATION", flush=True)
    print("=" * 70, flush=True)
    print(f"N values: {N_VALUES}, M={M_ENSEMBLE}", flush=True)
    print(f"Verifying: {VERIFY_OBSERVABLES}", flush=True)
    print(f"Coscosh eps: {EPS_COSCOSH}", flush=True)
    print(f"Quadrupole eps: {EPS_QUADRUPOLE}", flush=True)
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

                sw = [r["spectral_width"] for r in raw]
                print(f"  {prof_name} N={N} eps={eps:+.1f}: "
                      f"spec_width={np.mean(sw):.4f}+-{np.std(sw):.4f}"
                      f"  [{elapsed:.1f}s]", flush=True)
                results_N.extend(raw)

            # Mediation
            med = mediation(results_N)
            key = f"{prof_name}_N{N}"
            all_results[key] = {
                "mediation": med,
                "n_sprinklings": len(results_N),
            }

            # Print top 3
            ranked = sorted(VERIFY_OBSERVABLES,
                            key=lambda o: abs(med[o]["best_r_partial"]),
                            reverse=True)
            print(f"  Top signals ({med['best_predictor']}):", flush=True)
            for obs in ranked[:3]:
                m = med[obs]
                print(f"    {obs:20s}: r_partial={m['best_r_partial']:+.4f}"
                      f" p={m['best_p_partial']:.2e}", flush=True)

    # ==================================================================
    # SCALING + VERDICT
    # ==================================================================

    total_time = time.perf_counter() - t_total

    print(f"\n{'=' * 70}", flush=True)
    print("SCALING + COMPARISON", flush=True)
    print("=" * 70, flush=True)

    # Print scaling table for coscosh spectral_width (strongest original signal)
    print(f"\n  Coscosh spectral_width r_partial:", flush=True)
    for N in N_VALUES:
        med = all_results[f"coscosh_N{N}"]["mediation"]
        m = med["spectral_width"]
        print(f"    N={N}: r_partial={m['best_r_partial']:+.4f}"
              f" p={m['best_p_partial']:.2e}", flush=True)

    # Quadrupole at primary N
    med_q = all_results[f"quadrupole_N{N_PRIMARY}"]["mediation"]
    med_c = all_results[f"coscosh_N{N_PRIMARY}"]["mediation"]

    print(f"\n  Comparison at N={N_PRIMARY}:", flush=True)
    print(f"    Coscosh best:    {med_c['best_observable']}"
          f" r={med_c['best_r_partial']:+.4f}"
          f" p={med_c['best_p_partial']:.2e}", flush=True)
    print(f"    Quadrupole best: {med_q['best_observable']}"
          f" r={med_q['best_r_partial']:+.4f}"
          f" p={med_q['best_p_partial']:.2e}", flush=True)

    # Count reproduced signals with Bonferroni correction for multiple comparisons
    n_obs = len(VERIFY_OBSERVABLES)
    alpha_bonferroni = 0.01 / n_obs  # correct for testing n_obs observables
    n_reproduced_c = sum(1 for obs in VERIFY_OBSERVABLES
                         if abs(med_c[obs]["best_r_partial"]) > 0.10
                         and med_c[obs]["best_p_partial"] < alpha_bonferroni)
    n_reproduced_q = sum(1 for obs in VERIFY_OBSERVABLES
                         if abs(med_q[obs]["best_r_partial"]) > 0.10
                         and med_q[obs]["best_p_partial"] < alpha_bonferroni)

    print(f"\n  Signals reproduced (|r|>0.10, p<{alpha_bonferroni:.4f} Bonferroni):", flush=True)
    print(f"    Coscosh:    {n_reproduced_c}/{n_obs}", flush=True)
    print(f"    Quadrupole: {n_reproduced_q}/{n_obs}", flush=True)

    if n_reproduced_q >= 3:
        verdict = (f"CONFIRMED: {n_reproduced_q}/{n_obs} signals survive quadrupole"
                   f" (best: {med_q['best_observable']}"
                   f" r={med_q['best_r_partial']:+.4f})")
    elif n_reproduced_c >= 3:
        verdict = (f"COSCOSH ONLY: {n_reproduced_c}/{n_obs} survive coscosh,"
                   f" {n_reproduced_q}/{n_obs} quadrupole")
    elif n_reproduced_c >= 1:
        verdict = (f"PARTIAL: {n_reproduced_c}/{n_obs} coscosh,"
                   f" {n_reproduced_q}/{n_obs} quadrupole")
    else:
        verdict = f"NOT REPRODUCED: 0/{n_obs} signals at p<0.01"

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
            "verify_observables": VERIFY_OBSERVABLES,
        },
        "results": all_results,
        "summary": {
            "n_reproduced_coscosh": n_reproduced_c,
            "n_reproduced_quadrupole": n_reproduced_q,
        },
        "verdict": verdict,
        "wall_time_sec": total_time,
    }

    out_path = RESULTS_DIR / "exp2_d4_sj_verification.json"
    save_experiment(meta, output, out_path)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
