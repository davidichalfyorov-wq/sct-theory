"""
FND-1 EXP-20: CRN Curvature Detection Test.

Common Random Numbers: SAME point set, DIFFERENT causal condition.
For each sprinkling, compute the GTA commutator at eps=0 AND eps=target.
Paired t-test on differences. NO mediation needed.

We define the Geometric Temporal Asymmetry (GTA) observable as the
commutator [H,M] = (L^T L - L L^T) / 2, where L is the Benincasa-Dowker
d'Alembertian on a causal set. The non-normality of L reflects the causal
asymmetry of the sprinkling; curvature systematically enhances this
asymmetry beyond the statistical baseline.

Note: L in this code differs from the standard BD operator by an overall
factor of sqrt(6) (the 4/sqrt(6) prefactor is absorbed into the layer
coefficients as {4, -36, 64, -32} rather than {4, -36, 64, -32}/sqrt(6)).
This does NOT affect the CRN paired test, since the same factor applies
to both flat and curved conditions and cancels in the paired difference.

The CRN design eliminates density confounds by construction:
same points -> same density -> any difference IS due to curvature.

Run:
    python analysis/scripts/fnd1_exp20_crn_paired.py
"""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys, time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy import stats
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fnd1_4d_experiment import (
    sprinkle_4d_flat, causal_matrix_4d, compute_layers_4d, bd_action_4d,
)
from fnd1_4d_followup import quadrupole_profile
from fnd1_experiment_registry import ExperimentMeta, save_experiment, RESULTS_DIR
from fnd1_parallel import N_WORKERS, _init_worker

try:
    from fnd1_gpu import gpu_eigvalsh
except ImportError:
    gpu_eigvalsh = np.linalg.eigvalsh

N_VALUES = [500, 1000, 2000]
M_ENSEMBLE = 100
T_DIAMOND = 1.0
MASTER_SEED = 2020
WORKERS = N_WORKERS
# Test multiple curvature strengths
EPS_TARGETS = [2.0, 5.0, 10.0]


def _compute_gta(pts, C, N, T):
    """Compute GTA observables from a causal matrix.

    We define GTA as the commutator [H,M] = (L^T L - L L^T) / 2 of the
    Benincasa-Dowker operator L.  Eigenvalue statistics of [H,M] serve as
    curvature-sensitive observables.

    Note: L here includes an extra factor of sqrt(6) relative to the
    canonical BD normalization.  This cancels in all paired comparisons.
    """
    V = np.pi * T ** 4 / 24.0
    rho = N / V
    n_matrix, N0, N1, N2, N3 = compute_layers_4d(C)

    # BD operator (coefficients {4,-36,64,-32} = standard BD × sqrt(6))
    past = C.T; n_past = n_matrix.T
    n_int = np.rint(n_past).astype(np.int64)
    causal_mask = past > 0.5
    scale = np.sqrt(rho)
    L = np.zeros((N, N), dtype=np.float64)
    L[causal_mask & (n_int == 0)] = 4.0 * scale
    L[causal_mask & (n_int == 1)] = -36.0 * scale
    L[causal_mask & (n_int == 2)] = 64.0 * scale
    L[causal_mask & (n_int == 3)] = -32.0 * scale

    del C, n_matrix  # free memory

    L_sp = sp.csr_matrix(L)
    comm = ((L_sp.T @ L_sp - L_sp @ L_sp.T) / 2.0).toarray()
    comm = (comm + comm.T) / 2.0
    del L, L_sp
    comm_eigs = gpu_eigvalsh(comm)
    del comm

    a = np.abs(comm_eigs)
    s = float(np.sum(a))
    entropy = float(-np.sum((a/s) * np.log(a/s + 1e-300))) if s > 0 else 0.0
    frobenius = float(np.sqrt(np.sum(comm_eigs ** 2)))
    max_abs = float(np.max(a))
    tc = float(np.sum(causal_mask))

    return {
        "comm_entropy": entropy,
        "comm_frobenius": frobenius,
        "comm_max_abs": max_abs,
        "total_causal": tc,
    }


def _worker(args):
    """CRN: same points, flat vs curved. Returns paired difference."""
    seed_int, N, T, eps_target = args

    rng = np.random.default_rng(seed_int)
    pts = sprinkle_4d_flat(N, T, rng)

    # FLAT causal matrix
    C_flat = causal_matrix_4d(pts, 0.0, "flat")
    gta_flat = _compute_gta(pts, C_flat, N, T)

    # CURVED causal matrix (same points!)
    t = pts[:, 0]; x = pts[:, 1]; y = pts[:, 2]; z = pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx ** 2 + dy ** 2 + dz ** 2
    xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
    ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
    f_mid = quadrupole_profile(xm, ym)
    mink = dt ** 2 - dr2
    corr = eps_target * f_mid * (dt + dz) ** 2 / 2.0
    C_curved = ((mink > corr) & (dt > 0)).astype(np.float64)
    del dx, dy, dz, dr2, xm, ym, f_mid, mink, corr

    gta_curved = _compute_gta(pts, C_curved, N, T)

    # Paired differences
    result = {"seed": seed_int, "N": N, "eps": eps_target}
    for key in ["comm_entropy", "comm_frobenius", "comm_max_abs"]:
        result[f"{key}_flat"] = gta_flat[key]
        result[f"{key}_curved"] = gta_curved[key]
        result[f"{key}_delta"] = gta_curved[key] - gta_flat[key]

    result["tc_flat"] = gta_flat["total_causal"]
    result["tc_curved"] = gta_curved["total_causal"]
    result["tc_delta_pct"] = (gta_curved["total_causal"] - gta_flat["total_causal"]) / max(gta_flat["total_causal"], 1) * 100

    return result


def main():
    t_total = time.perf_counter()
    meta = ExperimentMeta(route=3, name="exp20_crn_paired",
        description="CRN paired test: same points, flat vs curved. Curvature detection.",
        N=max(N_VALUES), M=M_ENSEMBLE, status="running")

    print("=" * 70)
    print("FND-1 EXP-20: CRN CURVATURE DETECTION TEST")
    print("=" * 70)
    print(f"N: {N_VALUES}, M={M_ENSEMBLE}, eps_targets: {EPS_TARGETS}")
    print(f"Method: same points, flat vs curved. Paired t-test. No mediation.")
    print(f"Workers: {WORKERS}")
    print()

    ss = np.random.SeedSequence(MASTER_SEED)
    all_children = ss.spawn(len(N_VALUES) * len(EPS_TARGETS) * M_ENSEMBLE)
    child_idx = 0
    results_by_N = {}

    for N in N_VALUES:
        print(f"\nN = {N}")
        n_results = {}

        for eps in EPS_TARGETS:
            seed_ints = [int(all_children[child_idx + i].generate_state(1)[0])
                         for i in range(M_ENSEMBLE)]
            child_idx += M_ENSEMBLE
            args = [(si, N, T_DIAMOND, eps) for si in seed_ints]

            t0 = time.perf_counter()
            with Pool(WORKERS, initializer=_init_worker) as pool:
                raw = pool.map(_worker, args)
            elapsed = time.perf_counter() - t0

            # Paired t-test on deltas
            obs_tests = {}
            for obs in ["comm_entropy", "comm_frobenius", "comm_max_abs"]:
                deltas = [r[f"{obs}_delta"] for r in raw]
                mean_d = float(np.mean(deltas))
                sem_d = float(np.std(deltas, ddof=1) / np.sqrt(len(deltas)))
                t_stat, p_val = stats.ttest_1samp(deltas, 0.0)
                # Effect size: Cohen's d for paired
                cohen_d = mean_d / np.std(deltas, ddof=1) if np.std(deltas, ddof=1) > 1e-20 else 0
                # Wilcoxon signed-rank (nonparametric)
                try:
                    w_stat, w_p = stats.wilcoxon(deltas)
                except ValueError:
                    w_stat, w_p = 0, 1.0

                obs_tests[obs] = {
                    "mean_delta": round(mean_d, 6),
                    "sem": round(sem_d, 6),
                    "t_stat": round(float(t_stat), 4),
                    "p_ttest": float(p_val),
                    "p_wilcoxon": float(w_p),
                    "cohen_d": round(float(cohen_d), 4),
                    "significant_ttest_uncorrected": p_val < 0.01,
                    "significant_wilcoxon_uncorrected": w_p < 0.01,
                }

            # TC change
            tc_deltas = [r["tc_delta_pct"] for r in raw]
            mean_tc = float(np.mean(tc_deltas))

            # Best observable
            best_obs = max(obs_tests.keys(), key=lambda k: abs(obs_tests[k]["cohen_d"]))
            best = obs_tests[best_obs]

            n_results[str(eps)] = {
                "obs_tests": obs_tests,
                "best_observable": best_obs,
                "best_cohen_d": best["cohen_d"],
                "best_p_ttest": best["p_ttest"],
                "best_p_wilcoxon": best["p_wilcoxon"],
                "mean_tc_change_pct": round(mean_tc, 2),
                "n_pairs": len(raw),
            }

            sig = "***" if best["p_ttest"] < 0.001 else "**" if best["p_ttest"] < 0.01 else "*" if best["p_ttest"] < 0.05 else "ns"
            print(f"  eps={eps:+5.1f}: {best_obs} d={best['cohen_d']:+.4f} "
                  f"p_t={best['p_ttest']:.2e} p_w={best['p_wilcoxon']:.2e} "
                  f"TC={mean_tc:+.1f}% {sig}  [{elapsed:.1f}s]")

        results_by_N[str(N)] = n_results

    # Overall verdict
    total_time = time.perf_counter() - t_total

    print(f"\n{'=' * 70}")
    print("CRN PAIRED RESULTS SUMMARY")
    print("=" * 70)

    all_sig = True
    for N in N_VALUES:
        for eps in EPS_TARGETS:
            r = results_by_N[str(N)][str(eps)]
            sig_t = r["best_p_ttest"] < 0.01
            sig_w = r["best_p_wilcoxon"] < 0.01
            both = sig_t and sig_w
            status = "BOTH SIG" if both else "t-test only" if sig_t else "Wilcoxon only" if sig_w else "NOT SIG"
            print(f"  N={N:5d} eps={eps:+5.1f}: d={r['best_cohen_d']:+.4f} "
                  f"p_t={r['best_p_ttest']:.2e} p_w={r['best_p_wilcoxon']:.2e} "
                  f"TC={r['mean_tc_change_pct']:+.1f}% [{status}]")
            if not both:
                all_sig = False

    # Bonferroni: 3 N × 3 eps × 3 obs = 27 tests. But we report best-of-3 per (N,eps).
    # Effectively 9 tests. Bonferroni threshold: 0.01/9 = 0.0011.
    # 3 N × 3 eps × 3 observables = 27 total tests (best-of-3 is still multiple testing)
    bonf_threshold = 0.01 / (len(N_VALUES) * len(EPS_TARGETS) * 3)
    n_bonf = sum(1 for N in N_VALUES for eps in EPS_TARGETS
                 if results_by_N[str(N)][str(eps)]["best_p_ttest"] < bonf_threshold)

    if n_bonf == len(N_VALUES) * len(EPS_TARGETS):
        verdict = (f"DEFINITIVE: ALL {n_bonf}/{len(N_VALUES)*len(EPS_TARGETS)} conditions "
                   f"significant after Bonferroni (p<{bonf_threshold:.4f}). "
                   f"GTA curvature signal detected via CRN.")
    elif n_bonf > 0:
        verdict = (f"PARTIAL: {n_bonf}/{len(N_VALUES)*len(EPS_TARGETS)} conditions "
                   f"survive Bonferroni. GTA detects curvature at sufficient eps/N.")
    else:
        verdict = (f"NEGATIVE: 0/{len(N_VALUES)*len(EPS_TARGETS)} survive Bonferroni. "
                   f"GTA does not detect curvature in CRN design.")

    print(f"\n{'=' * 70}")
    print(f"VERDICT: {verdict}")
    print(f"Wall time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print("=" * 70)

    meta.status = "completed"; meta.verdict = verdict; meta.wall_time_sec = total_time
    output = {
        "parameters": {
            "N_values": N_VALUES, "M": M_ENSEMBLE, "eps_targets": EPS_TARGETS,
            "T": T_DIAMOND, "profile": "quadrupole",
            "method": "CRN paired: same points, flat vs curved, paired t-test + Wilcoxon",
        },
        "results_by_N": results_by_N,
        "bonferroni_threshold": bonf_threshold,
        "n_bonferroni_sig": n_bonf,
        "verdict": verdict,
        "wall_time_sec": total_time,
    }
    save_experiment(meta, output, RESULTS_DIR / "exp20_crn_paired.json")
    print(f"Saved: exp20_crn_paired.json")

if __name__ == "__main__":
    main()
