"""
FND-1: Sorkin-Johnston Vacuum on 4D Causal Sets.

The SJ Wightman function W_SJ = Pos(i*Delta) where Delta = K_R - K_R^T
is the Pauli-Jordan function built from the retarded Green function.

For d=4 massless scalar (Johnston 0909.0944, 1010.5514):
  K_R = a * L  where  a = sqrt(rho) / (2*pi*sqrt(6))
  L = link matrix (Hasse diagram, upper-triangular: L[i,j]=1 if i prec j with 0 between)
  Delta = a*(L - L^T)  (real antisymmetric)
  i*Delta is Hermitian with eigenvalues in +/- pairs
  W_SJ = sum over positive-eigenvalue eigenvectors

This is fundamentally different from all previous FND-1 attempts:
  - Uses the GREEN FUNCTION (inverse of Box), not the operator itself
  - The SJ construction is the unique covariant vacuum (Sorkin 2011)
  - Has non-trivial spectrum even on finite causal sets

Pre-registration:
  PRIMARY = spectral gap of W_SJ (ratio lambda_1/lambda_2 of positive eigenvalues)
  MODE = linear (eps)
  SUCCESS = |r_partial| > 0.10 after TC+N_link+BD polynomial mediation

Run:
  "C:/Users/youre/miniconda3/envs/sct-mkl/python.exe" analysis/scripts/fnd1_sj_vacuum.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from fnd1_gpu import gpu_eigvalsh
except ImportError:
    gpu_eigvalsh = np.linalg.eigvalsh

from fnd1_4d_experiment import (
    sprinkle_4d_flat, sprinkle_4d_ppwave,
    causal_matrix_4d, compute_layers_4d, bd_action_4d,
    build_link_graph, _ppwave_profile,
)
from fnd1_experiment_registry import (
    ExperimentMeta, save_experiment, RESULTS_DIR,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_VALUES = [1000, 2000, 3000]
N_PRIMARY = 3000
M_ENSEMBLE = 50
T_DIAMOND = 1.0
MASTER_SEED = 271
EPSILON_CURVED = [0.0, 0.1, 0.2, 0.3, 0.5]
EPSILON_CONFORMAL = [0.2, 0.5]

PRIMARY_METRIC = "spectral_gap_ratio"
PRIMARY_MODE = "linear"


# ---------------------------------------------------------------------------
# SJ Construction (Johnston d=4 massless)
# ---------------------------------------------------------------------------

def build_sj_wightman(C, n_matrix, rho):
    """Build the Sorkin-Johnston Wightman function for massless scalar in d=4.

    Johnston (0909.0944): K_R = a * L where L is the link matrix,
    a = sqrt(rho) / (2*pi*sqrt(6)).
    Delta = K_R - K_R^T = a*(L - L^T).
    W_SJ = Pos(i*Delta) = positive spectral part of i*Delta.

    Returns: W_SJ (NxN Hermitian PSD), positive eigenvalues (sorted descending),
             all eigenvalues of i*Delta.
    """
    N = C.shape[0]
    a = np.sqrt(rho) / (2.0 * np.pi * np.sqrt(6.0))

    # Link matrix (upper-triangular: L[i,j]=1 if i prec j with 0 intervening)
    past = C.T
    n_past = n_matrix.T
    link_lower = ((past > 0) & (n_past == 0)).astype(np.float64)  # lower-tri
    L_upper = link_lower.T  # upper-tri: L[i,j]=1 if i prec j

    # Pauli-Jordan: Delta = a*(L - L^T)
    Delta = a * (L_upper - L_upper.T)  # antisymmetric

    # i*Delta is Hermitian (since Delta is real antisymmetric, i*Delta is real symmetric...
    # Wait: i * (real antisymmetric) = imaginary antisymmetric. That's anti-Hermitian, not Hermitian.
    # Correction: i*Delta where Delta is real antisymmetric:
    #   (i*Delta)^H = -i * Delta^T = -i * (-Delta) = i*Delta. YES, Hermitian.
    # But i*Delta has purely imaginary entries (i times real), so as a matrix it's
    # imaginary-Hermitian. eigvalsh handles complex Hermitian matrices.

    iDelta = 1j * Delta  # complex Hermitian matrix

    # Eigenvalues only (eigenvectors not needed — saves ~50% time and ~30% memory)
    evals = gpu_eigvalsh(iDelta)
    # eigvalsh returns sorted ascending. Eigenvalues are real.

    # Positive part
    pos_mask = evals > 1e-15
    pos_evals = evals[pos_mask]

    return pos_evals, evals


# ---------------------------------------------------------------------------
# Observables from SJ spectrum
# ---------------------------------------------------------------------------

def sj_observables(pos_evals, all_evals, N):
    """Extract observables from the SJ Wightman function eigenvalues.

    Includes spectral truncation at n_max ~ N^{3/4} (Surya+Yazdi recommendation)
    and SSEE-proxy (spectral entanglement entropy from the truncated spectrum).
    """
    keys = ["trace_W", "n_modes", "spectral_gap_ratio", "entropy_spectral",
            "lambda_max", "lambda_median", "spectral_width",
            "trace_trunc", "entropy_trunc", "ssee_proxy"]
    if len(pos_evals) == 0:
        return {k: 0.0 for k in keys}

    sorted_pos = np.sort(pos_evals)[::-1]  # descending

    # Full spectrum observables
    trace_W = float(np.sum(pos_evals))
    n_modes = len(pos_evals)
    lambda_max = float(sorted_pos[0])
    lambda_median = float(np.median(sorted_pos))

    if len(sorted_pos) >= 2 and sorted_pos[1] > 1e-20:
        spectral_gap_ratio = float(sorted_pos[0] / sorted_pos[1])
    else:
        spectral_gap_ratio = float('inf')

    p = pos_evals / trace_W if trace_W > 0 else np.ones_like(pos_evals) / len(pos_evals)
    entropy = float(-np.sum(p * np.log(p + 1e-300)))

    log_evals = np.log(pos_evals[pos_evals > 1e-20])
    spectral_width = float(np.std(log_evals)) if len(log_evals) > 1 else 0.0

    # TRUNCATED spectrum: keep only top n_max ~ N^{3/4} modes (Surya+Yazdi)
    # These are the "physical" modes; lower ones are lattice artifacts
    n_max = max(1, int(N**0.75))
    trunc = sorted_pos[:n_max]
    trace_trunc = float(np.sum(trunc))
    p_trunc = trunc / trace_trunc if trace_trunc > 0 else np.ones_like(trunc) / len(trunc)
    entropy_trunc = float(-np.sum(p_trunc * np.log(p_trunc + 1e-300)))

    # SSEE proxy: bosonic entanglement entropy from truncated SJ eigenvalues
    # Bosonic formula: S = sum[(n+1)*log(n+1) - n*log(n)] (Sorkin 1205.2953)
    # where n_k = SJ eigenvalues (bosonic occupation numbers, can be > 1)
    # Used as relative observable (flat vs curved), not calibrated absolutely
    ssee_terms = np.where(trunc > 1e-20,
                          (trunc + 1) * np.log(trunc + 1) - trunc * np.log(trunc + 1e-300), 0)
    ssee_proxy = float(np.sum(ssee_terms))

    return {
        "trace_W": trace_W,
        "n_modes": n_modes,
        "spectral_gap_ratio": spectral_gap_ratio,
        "entropy_spectral": entropy,
        "lambda_max": lambda_max,
        "lambda_median": lambda_median,
        "spectral_width": spectral_width,
        "trace_trunc": trace_trunc,
        "entropy_trunc": entropy_trunc,
        "ssee_proxy": ssee_proxy,
    }


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker_sj_wrapper(args):
    """Picklable wrapper for multiprocessing."""
    return worker_sj(*args)


def worker_sj(seed_int, N, T, eps, metric):
    """One sprinkling: build SJ vacuum, extract observables."""
    rng = np.random.default_rng(seed_int)

    if metric == "flat" or (metric == "ppwave" and abs(eps) < 1e-12):
        pts = sprinkle_4d_flat(N, T, rng)
        C = causal_matrix_4d(pts, 0.0, "flat")
    elif metric == "ppwave":
        pts = sprinkle_4d_ppwave(N, T, rng)
        C = causal_matrix_4d(pts, eps, "tidal")
    elif metric == "conformal":
        from fnd1_4d_experiment import sprinkle_4d_conformal
        pts = sprinkle_4d_conformal(N, T, eps, rng)
        C = causal_matrix_4d(pts, 0.0, "flat")

    total_causal = float(np.sum(C))
    n_matrix, N0, N1, N2, N3 = compute_layers_4d(C)
    V = np.pi * T**4 / 24.0
    rho = N / V
    bd = bd_action_4d(N, N0, N1, N2, N3)

    # Build SJ
    pos_evals, all_evals = build_sj_wightman(C, n_matrix, rho)
    obs = sj_observables(pos_evals, all_evals, N)

    return {
        "eps": eps, "metric": metric,
        "total_causal": total_causal,
        "n_links": N0,
        "bd_action": float(bd),
        **obs,
    }


# ---------------------------------------------------------------------------
# Mediation (polynomial, per FND-1 protocol)
# ---------------------------------------------------------------------------

def mediation_poly(eps_arr, obs_arr, tc_arr, nl_arr, bd_arr):
    """Full polynomial mediation: TC + TC^2 + TC^3 + N_link + BD."""
    n = len(eps_arr)
    tc2 = tc_arr**2
    tc3 = tc_arr**3

    def pr_multi(x, y, ctrls):
        X = np.column_stack([*ctrls, np.ones(n)])
        try:
            bx = np.linalg.lstsq(X, x, rcond=None)[0]
            by = np.linalg.lstsq(X, y, rcond=None)[0]
        except Exception:
            return 0.0, 1.0
        xr, yr = x - X @ bx, y - X @ by
        if np.std(xr) > 1e-15 and np.std(yr) > 1e-15:
            return stats.pearsonr(xr, yr)
        return 0.0, 1.0

    r_dir, p_dir = stats.pearsonr(eps_arr, obs_arr) if np.std(obs_arr) > 1e-15 else (0, 1)
    r_tc, p_tc = pr_multi(eps_arr, obs_arr, [tc_arr])
    r_full, p_full = pr_multi(eps_arr, obs_arr, [tc_arr, nl_arr, bd_arr])
    r_poly, p_poly = pr_multi(eps_arr, obs_arr, [tc_arr, tc2, nl_arr, bd_arr])
    r_poly3, p_poly3 = pr_multi(eps_arr, obs_arr, [tc_arr, tc2, tc3, nl_arr, bd_arr])

    return {
        "r_direct": float(r_dir), "p_direct": float(p_dir),
        "r_tc": float(r_tc),
        "r_full": float(r_full), "p_full": float(p_full),
        "r_poly": float(r_poly), "p_poly": float(p_poly),
        "r_poly3": float(r_poly3), "p_poly3": float(p_poly3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    print("=" * 70, flush=True)
    print("FND-1: SORKIN-JOHNSTON VACUUM ON 4D CAUSAL SETS", flush=True)
    print("=" * 70, flush=True)
    print(f"Johnston d=4 massless: K_R = a*L, a = sqrt(rho)/(2*pi*sqrt(6))", flush=True)
    print(f"N values: {N_VALUES} (primary: {N_PRIMARY})", flush=True)
    print(f"M={M_ENSEMBLE}, eps_curved: {EPSILON_CURVED}", flush=True)
    print(f"Pre-registered primary: {PRIMARY_METRIC} ({PRIMARY_MODE})", flush=True)
    print(flush=True)

    # Smoke test
    t0 = time.perf_counter()
    r = worker_sj(42, 500, T_DIAMOND, 0.0, "flat")
    print(f"Smoke test (N=500): {time.perf_counter()-t0:.2f}s, "
          f"n_modes={r['n_modes']}, trace={r['trace_W']:.4f}, "
          f"gap_ratio={r['spectral_gap_ratio']:.4f}", flush=True)

    ss = np.random.SeedSequence(MASTER_SEED)

    # ==================================================================
    # PART 1: PP-WAVE (curvature changes causal structure)
    # ==================================================================

    print(f"\n{'='*60}", flush=True)
    print("PART 1: PP-WAVE (Weyl != 0)", flush=True)
    print("=" * 60, flush=True)

    ppwave_data = []
    for eps in EPSILON_CURVED:
        eps_ss = ss.spawn(1)[0]
        seeds = [int(cs.generate_state(1)[0]) for cs in eps_ss.spawn(M_ENSEMBLE)]
        print(f"\n  eps={eps:+.2f}: {M_ENSEMBLE} sprinklings...", flush=True)
        t0 = time.perf_counter()
        from multiprocessing import Pool
        from fnd1_parallel import _init_worker, N_WORKERS
        args = [(s, N_PRIMARY, T_DIAMOND, eps, "ppwave") for s in seeds]
        with Pool(N_WORKERS, initializer=_init_worker) as pool:
            results = pool.map(_worker_sj_wrapper, args)
        elapsed = time.perf_counter() - t0
        tc = np.mean([r["total_causal"] for r in results])
        tr = np.mean([r["trace_W"] for r in results])
        gap = np.mean([r["spectral_gap_ratio"] for r in results])
        print(f"    Done {elapsed:.1f}s  TC={tc:.0f}  trace_W={tr:.4f}  "
              f"gap_ratio={gap:.4f}", flush=True)
        ppwave_data.extend(results)

    # ==================================================================
    # PART 2: CONFORMAL CONTROL (density only, same causal structure as flat)
    # ==================================================================

    print(f"\n{'='*60}", flush=True)
    print("PART 2: CONFORMAL CONTROL", flush=True)
    print("=" * 60, flush=True)

    conf_data = []
    for eps in EPSILON_CONFORMAL:
        eps_ss = ss.spawn(1)[0]
        seeds = [int(cs.generate_state(1)[0]) for cs in eps_ss.spawn(M_ENSEMBLE)]
        print(f"  eps_conf={eps:+.2f}...", flush=True)
        t0 = time.perf_counter()
        from multiprocessing import Pool
        from fnd1_parallel import _init_worker, N_WORKERS
        args = [(s, N_PRIMARY, T_DIAMOND, eps, "conformal") for s in seeds]
        with Pool(N_WORKERS, initializer=_init_worker) as pool:
            results = pool.map(_worker_sj_wrapper, args)
        print(f"    Done {time.perf_counter()-t0:.1f}s", flush=True)
        conf_data.extend(results)

    # ==================================================================
    # ANALYSIS
    # ==================================================================

    print(f"\n{'='*60}", flush=True)
    print("ANALYSIS: PP-WAVE", flush=True)
    print("=" * 60, flush=True)

    eps_a = np.array([d["eps"] for d in ppwave_data])
    tc_a = np.array([d["total_causal"] for d in ppwave_data])
    nl_a = np.array([d["n_links"] for d in ppwave_data])
    bd_a = np.array([d["bd_action"] for d in ppwave_data])

    metrics = ["trace_W", "n_modes", "spectral_gap_ratio", "entropy_spectral",
               "lambda_max", "lambda_median", "spectral_width",
               "trace_trunc", "entropy_trunc", "ssee_proxy"]

    med_results = {}
    print(f"\n  {'metric':>20} {'r_dir':>7} {'r|TC':>7} {'r|full':>7} {'r|poly':>7} "
          f"{'r|poly3':>7} {'p|poly3':>10} {'surv':>5}", flush=True)
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*10} {'-'*5}", flush=True)

    for metric in metrics:
        obs = np.array([d[metric] for d in ppwave_data])
        if np.std(obs) < 1e-15:
            continue
        med = mediation_poly(eps_a, obs, tc_a, nl_a, bd_a)
        med_results[metric] = med
        surv = abs(med["r_poly3"]) > 0.10 and med["p_poly3"] < 0.10
        mark = "YES" if surv else ""
        print(f"  {metric:>20} {med['r_direct']:+7.3f} {med['r_tc']:+7.3f} "
              f"{med['r_full']:+7.3f} {med['r_poly']:+7.3f} "
              f"{med['r_poly3']:+7.3f} {med['p_poly3']:10.2e} {mark:>5}", flush=True)

    # Group means
    print(f"\n  Group means:", flush=True)
    print(f"  {'eps':>5} {'TC':>8} {'links':>7} {'trace':>8} {'gap':>8} {'entropy':>8} {'n_modes':>7}",
          flush=True)
    for eps in EPSILON_CURVED:
        d = [x for x in ppwave_data if abs(x["eps"] - eps) < 0.01]
        print(f"  {eps:+5.2f} {np.mean([x['total_causal'] for x in d]):8.0f} "
              f"{np.mean([x['n_links'] for x in d]):7.0f} "
              f"{np.mean([x['trace_W'] for x in d]):8.4f} "
              f"{np.mean([x['spectral_gap_ratio'] for x in d]):8.4f} "
              f"{np.mean([x['entropy_spectral'] for x in d]):8.4f} "
              f"{np.mean([x['n_modes'] for x in d]):7.0f}", flush=True)

    # Conformal control
    print(f"\n  Conformal control:", flush=True)
    for eps in EPSILON_CONFORMAL:
        d = [x for x in conf_data if abs(x["eps"] - eps) < 0.01]
        print(f"    eps={eps}: TC={np.mean([x['total_causal'] for x in d]):.0f}, "
              f"gap={np.mean([x['spectral_gap_ratio'] for x in d]):.4f}", flush=True)

    # ==================================================================
    # FINITE-SIZE SCALING
    # ==================================================================

    print(f"\n{'='*60}", flush=True)
    print("FINITE-SIZE SCALING", flush=True)
    print("=" * 60, flush=True)

    for N_test in N_VALUES:
        sc_data = []
        for eps in EPSILON_CURVED:
            n_ss = ss.spawn(1)[0]
            seeds = [int(cs.generate_state(1)[0]) for cs in n_ss.spawn(M_ENSEMBLE // 2)]
            from multiprocessing import Pool
            from fnd1_parallel import _init_worker, N_WORKERS
            sc_args = [(s, N_test, T_DIAMOND, eps, "ppwave") for s in seeds]
            with Pool(N_WORKERS, initializer=_init_worker) as pool:
                sc_data.extend(pool.map(_worker_sj_wrapper, sc_args))

        sc_eps = np.array([d["eps"] for d in sc_data])
        sc_tc = np.array([d["total_causal"] for d in sc_data])
        sc_nl = np.array([d["n_links"] for d in sc_data])
        sc_bd = np.array([d["bd_action"] for d in sc_data])

        parts = []
        for metric in [PRIMARY_METRIC, "trace_W", "entropy_spectral"]:
            sc_obs = np.array([d[metric] for d in sc_data])
            if np.std(sc_obs) > 1e-15:
                med = mediation_poly(sc_eps, sc_obs, sc_tc, sc_nl, sc_bd)
                parts.append(f"{metric}: r_dir={med['r_direct']:+.3f} "
                             f"r|poly3={med['r_poly3']:+.3f}")

        print(f"  N={N_test}: {' | '.join(parts)}", flush=True)

    # ==================================================================
    # VERDICT
    # ==================================================================

    total_time = time.perf_counter() - t_total

    print(f"\n{'='*70}", flush=True)
    print("VERDICT", flush=True)
    print("=" * 70, flush=True)

    primary = med_results.get(PRIMARY_METRIC, {})
    r_poly3 = primary.get("r_poly3", 0)
    p_poly3 = primary.get("p_poly3", 1)
    survives = abs(r_poly3) > 0.10 and p_poly3 < 0.10

    # Check any metric survives
    any_survives = any(
        abs(m.get("r_poly3", 0)) > 0.10 and m.get("p_poly3", 1) < 0.10
        for m in med_results.values()
    )

    if survives:
        verdict = f"SJ SIGNAL: {PRIMARY_METRIC} survives poly3 (r={r_poly3:+.3f}, p={p_poly3:.2e})"
    elif any_survives:
        surv_names = [k for k, m in med_results.items()
                      if abs(m.get("r_poly3", 0)) > 0.10 and m.get("p_poly3", 1) < 0.10]
        verdict = f"EXPLORATORY SIGNAL in {surv_names} (primary {PRIMARY_METRIC} failed)"
    else:
        verdict = "NO SIGNAL: SJ vacuum spectrum does not detect curvature beyond TC+BD"

    print(f"\n  {verdict}", flush=True)
    print(f"  Primary ({PRIMARY_METRIC}): r_poly3={r_poly3:+.4f}, p={p_poly3:.2e}", flush=True)
    print(f"  Wall time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    meta = ExperimentMeta(route=2, name="sj_vacuum",
                          description="Sorkin-Johnston vacuum d=4 pp-wave",
                          N=N_PRIMARY, M=M_ENSEMBLE, status="completed", verdict=verdict)
    meta.wall_time_sec = total_time
    output = {"verdict": verdict, "mediation": med_results, "wall_time_sec": total_time}
    save_experiment(meta, output, RESULTS_DIR / "sj_vacuum.json")
    print(f"  Saved: {RESULTS_DIR / 'sj_vacuum.json'}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
