"""
FND-1 d=4 Follow-up: Three Falsification Tests.

1. TC^2 control: add quadratic TC to mediation. If signal drops -> artifact.
2. Pure quadrupole profile: f=x^2-y^2 (zero mean). If signal vanishes -> monopole artifact.
3. Finite-size scaling of mag_frobenius and link_fiedler at N=500..3000.

Run with MKL:
  "C:/Users/youre/miniconda3/envs/sct-mkl/python.exe" analysis/scripts/fnd1_4d_followup.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_4d_experiment import (
    sprinkle_4d_flat, causal_matrix_4d, compute_layers_4d, bd_action_4d,
    build_link_graph, link_spectral_embedding, build_magnetic_laplacian,
    _ppwave_profile, WORKERS,
)
from fnd1_parallel import _init_worker
from fnd1_experiment_registry import (
    ExperimentMeta, save_experiment, RESULTS_DIR,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_VALUES = [500, 1000, 2000, 3000]
M_ENSEMBLE = 100
T_DIAMOND = 1.0
MASTER_SEED = 77

# Profile A: cos*cosh (original, has monopole)
EPS_COSCOSH = [0.0, 0.1, 0.2, 0.3, 0.5]

# Profile B: x^2-y^2 (zero mean, pure quadrupole) -- needs larger eps
EPS_QUADRUPOLE = [0.0, 5.0, 10.0, 20.0, 40.0]

N_DISTANCE_PAIRS = 8000
K_VALUES = [2, 5, 10, 20]


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

def quadrupole_profile(x, y):
    """Pure quadrupole: f = x^2 - y^2. Zero mean, pure Weyl tidal."""
    return x**2 - y**2


# ---------------------------------------------------------------------------
# Worker (unified for both profiles)
# ---------------------------------------------------------------------------

def _worker_followup(args):
    """Compute observables for one 4D sprinkling with specified profile."""
    seed_int, N, T, eps, profile_name = args

    rng = np.random.default_rng(seed_int)
    pts = sprinkle_4d_flat(N, T, rng)  # PP-wave: det=-1, uniform density

    t = pts[:, 0]
    x = pts[:, 1]
    y = pts[:, 2]
    z = pts[:, 3]

    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx**2 + dy**2 + dz**2
    mink_interval = dt**2 - dr2

    if abs(eps) < 1e-12:
        C = ((mink_interval > 0) & (dt > 0)).astype(np.float64)
    else:
        xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
        ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0

        if profile_name == "coscosh":
            f_mid = _ppwave_profile(xm, ym)
        else:
            f_mid = quadrupole_profile(xm, ym)

        ppwave_corr = eps * f_mid * (dt + dz)**2 / 2.0
        C = ((mink_interval > ppwave_corr) & (dt > 0)).astype(np.float64)

    total_causal = float(np.sum(C))

    # Layers + BD
    n_matrix, N0, N1, N2, N3 = compute_layers_4d(C)
    bd = bd_action_4d(N, N0, N1, N2, N3)

    # Link graph
    A_link = build_link_graph(C, n_matrix)
    mean_link_deg = float(A_link.sum() / N)

    # Link Fiedler
    embedding, link_lams = link_spectral_embedding(A_link, 20, N)
    link_fiedler = float(link_lams[0]) if len(link_lams) > 0 else 0.0

    # Magnetic Laplacian
    L_mag, A_undir, degrees = build_magnetic_laplacian(C)
    eigs_mag = np.linalg.eigvalsh(L_mag)
    mag_frobenius = float(np.sqrt(np.sum(eigs_mag**2)))

    abs_eigs = np.abs(eigs_mag)
    s = np.sum(abs_eigs)
    mag_entropy = float(-np.sum((abs_eigs / s) * np.log(abs_eigs / s + 1e-300))) if s > 0 else 0

    sorted_mag = np.sort(eigs_mag)
    sorted_undir = np.sort(np.linalg.eigvalsh(np.diag(degrees) - A_undir))
    fiedler_mag = float(sorted_mag[1]) if len(sorted_mag) > 1 else 0.0
    fiedler_undir = float(sorted_undir[1]) if len(sorted_undir) > 1 else 0.0
    mag_surplus = fiedler_mag - fiedler_undir

    # Route 2: r_euclidean at k=2
    if embedding.shape[1] >= 2:
        rng2 = np.random.default_rng(seed_int + 999)
        idx_i = rng2.integers(0, N, size=N_DISTANCE_PAIRS)
        idx_j = rng2.integers(0, N, size=N_DISTANCE_PAIRS)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]
        d_eucl = np.sqrt(np.sum((pts[idx_i] - pts[idx_j])**2, axis=1))
        emb = embedding[:, :2]
        d_emb = np.sqrt(np.sum((emb[idx_i] - emb[idx_j])**2, axis=1))
        if np.std(d_emb) > 0:
            r_eucl, _ = stats.pearsonr(d_emb, d_eucl)
        else:
            r_eucl = 0.0
    else:
        r_eucl = 0.0

    return {
        "eps": eps, "profile": profile_name,
        "total_causal": total_causal,
        "n_links": N0,
        "mean_link_deg": mean_link_deg,
        "bd_action": float(bd),
        "link_fiedler": link_fiedler,
        "mag_frobenius": mag_frobenius,
        "mag_entropy": mag_entropy,
        "mag_fiedler": fiedler_mag,
        "mag_surplus": float(mag_surplus),
        "r_eucl_k2": float(r_eucl),
    }


# ---------------------------------------------------------------------------
# Polynomial mediation
# ---------------------------------------------------------------------------

def mediation_polynomial(eps_arr, obs_arr, tc_arr, nlink_arr, bd_arr):
    """Mediation with TC, TC^2, TC^3, N_link, BD (polynomial TC control)."""
    n = len(eps_arr)
    tc2 = tc_arr**2
    tc3 = tc_arr**3

    def partial_r_multi(x, y, ctrls):
        X = np.column_stack([*ctrls, np.ones(n)])
        try:
            bx = np.linalg.lstsq(X, x, rcond=None)[0]
            by = np.linalg.lstsq(X, y, rcond=None)[0]
        except Exception:
            return 0.0, 1.0
        xr, yr = x - X @ bx, y - X @ by
        if np.std(xr) > 0 and np.std(yr) > 0:
            return stats.pearsonr(xr, yr)
        return 0.0, 1.0

    r_lin, p_lin = stats.pearsonr(eps_arr, obs_arr)

    # Linear TC only
    r_tc1, p_tc1 = partial_r_multi(eps_arr, obs_arr, [tc_arr])
    # TC + TC^2
    r_tc2, p_tc2 = partial_r_multi(eps_arr, obs_arr, [tc_arr, tc2])
    # TC + TC^2 + TC^3
    r_tc3, p_tc3 = partial_r_multi(eps_arr, obs_arr, [tc_arr, tc2, tc3])
    # Full original: TC + N_link + BD
    r_full, p_full = partial_r_multi(eps_arr, obs_arr, [tc_arr, nlink_arr, bd_arr])
    # Full + TC^2
    r_poly, p_poly = partial_r_multi(eps_arr, obs_arr, [tc_arr, tc2, nlink_arr, bd_arr])
    # Full + TC^2 + TC^3
    r_poly3, p_poly3 = partial_r_multi(eps_arr, obs_arr, [tc_arr, tc2, tc3, nlink_arr, bd_arr])

    return {
        "r_direct": float(r_lin),
        "r_tc1": float(r_tc1), "p_tc1": float(p_tc1),
        "r_tc2": float(r_tc2), "p_tc2": float(p_tc2),
        "r_tc3": float(r_tc3), "p_tc3": float(p_tc3),
        "r_full_lin": float(r_full), "p_full_lin": float(p_full),
        "r_poly": float(r_poly), "p_poly": float(p_poly),
        "r_poly3": float(r_poly3), "p_poly3": float(p_poly3),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    print("=" * 70, flush=True)
    print("FND-1 d=4 FOLLOW-UP: THREE FALSIFICATION TESTS", flush=True)
    print("=" * 70, flush=True)
    print(f"Test 1: Polynomial TC control (TC + TC^2 + TC^3)", flush=True)
    print(f"Test 2: Pure quadrupole f=x^2-y^2 (zero mean)", flush=True)
    print(f"Test 3: Finite-size scaling of mag_frobenius, link_fiedler", flush=True)
    print(f"N values: {N_VALUES}, M={M_ENSEMBLE}", flush=True)
    print(flush=True)

    # Benchmark
    t0 = time.perf_counter()
    _worker_followup((42, 2000, T_DIAMOND, 0.0, "coscosh"))
    bench = time.perf_counter() - t0
    print(f"Benchmark: {bench:.2f}s/task at N=2000", flush=True)

    ss = np.random.SeedSequence(MASTER_SEED)

    # ==================================================================
    # TEST 1 + 2: Run both profiles at N=2000
    # ==================================================================

    profiles = [
        ("coscosh", EPS_COSCOSH),
        ("quadrupole", EPS_QUADRUPOLE),
    ]

    all_profile_data = {}

    for profile_name, eps_list in profiles:
        print(f"\n{'='*60}", flush=True)
        print(f"PROFILE: {profile_name} (eps: {eps_list})", flush=True)
        print("=" * 60, flush=True)

        profile_data = []
        for eps in eps_list:
            eps_ss = ss.spawn(1)[0]
            seeds = [int(cs.generate_state(1)[0]) for cs in eps_ss.spawn(M_ENSEMBLE)]
            args = [(si, 2000, T_DIAMOND, eps, profile_name) for si in seeds]

            print(f"  eps={eps:+.1f}: {M_ENSEMBLE} sprinklings...", flush=True)
            t0 = time.perf_counter()
            with Pool(WORKERS, initializer=_init_worker) as pool:
                results = pool.map(_worker_followup, args)
            tc = np.mean([r["total_causal"] for r in results])
            mf = np.mean([r["mag_frobenius"] for r in results])
            lf = np.mean([r["link_fiedler"] for r in results])
            print(f"    Done {time.perf_counter()-t0:.1f}s  TC={tc:.0f}  "
                  f"frob={mf:.0f}  fiedler={lf:.4f}", flush=True)
            profile_data.extend(results)

        all_profile_data[profile_name] = profile_data

    # ==================================================================
    # ANALYSIS: Polynomial mediation for both profiles
    # ==================================================================

    metrics_to_test = ["mag_frobenius", "link_fiedler", "mag_entropy", "mag_fiedler", "mag_surplus"]

    for profile_name, eps_list in profiles:
        data = all_profile_data[profile_name]

        print(f"\n{'='*60}", flush=True)
        print(f"POLYNOMIAL MEDIATION: {profile_name}", flush=True)
        print("=" * 60, flush=True)

        eps_a = np.array([d["eps"] for d in data])
        tc_a = np.array([d["total_causal"] for d in data])
        nl_a = np.array([d["n_links"] for d in data])
        bd_a = np.array([d["bd_action"] for d in data])

        print(f"\n  {'metric':>15} {'r_dir':>7} {'r|TC':>7} {'r|TC2':>7} {'r|TC3':>7} "
              f"{'r|full':>7} {'r|poly':>7} {'r|poly3':>7} {'kill?':>6}", flush=True)
        print(f"  {'-'*15} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6}",
              flush=True)

        for metric in metrics_to_test:
            obs = np.array([d[metric] for d in data])
            med = mediation_polynomial(eps_a, obs, tc_a, nl_a, bd_a)

            killed = abs(med["r_poly"]) < 0.10 and abs(med["r_full_lin"]) >= 0.10
            kill_mark = "KILL" if killed else ("surv" if abs(med["r_poly"]) >= 0.10 else "")

            print(f"  {metric:>15} {med['r_direct']:+7.3f} {med['r_tc1']:+7.3f} "
                  f"{med['r_tc2']:+7.3f} {med['r_tc3']:+7.3f} "
                  f"{med['r_full_lin']:+7.3f} {med['r_poly']:+7.3f} "
                  f"{med['r_poly3']:+7.3f} {kill_mark:>6}", flush=True)

    # ==================================================================
    # TEST 3: Finite-size scaling
    # ==================================================================

    print(f"\n{'='*60}", flush=True)
    print("FINITE-SIZE SCALING: mag_frobenius & link_fiedler (coscosh)", flush=True)
    print("=" * 60, flush=True)

    scaling_metrics = ["mag_frobenius", "link_fiedler"]

    for N_test in N_VALUES:
        sc_data = []
        for eps in EPS_COSCOSH:
            n_ss = ss.spawn(1)[0]
            seeds = [int(cs.generate_state(1)[0]) for cs in n_ss.spawn(M_ENSEMBLE // 2)]
            args = [(si, N_test, T_DIAMOND, eps, "coscosh") for si in seeds]
            with Pool(WORKERS, initializer=_init_worker) as pool:
                sc_data.extend(pool.map(_worker_followup, args))

        sc_eps = np.array([d["eps"] for d in sc_data])
        sc_tc = np.array([d["total_causal"] for d in sc_data])
        sc_nl = np.array([d["n_links"] for d in sc_data])
        sc_bd = np.array([d["bd_action"] for d in sc_data])

        parts = []
        for metric in scaling_metrics:
            sc_obs = np.array([d[metric] for d in sc_data])
            med = mediation_polynomial(sc_eps, sc_obs, sc_tc, sc_nl, sc_bd)
            parts.append(f"{metric}: r_dir={med['r_direct']:+.3f} r|full={med['r_full_lin']:+.3f} "
                         f"r|poly={med['r_poly']:+.3f}")

        print(f"  N={N_test}: {' | '.join(parts)}", flush=True)

    # ==================================================================
    # VERDICT
    # ==================================================================

    total_time = time.perf_counter() - t_total

    print(f"\n{'='*70}", flush=True)
    print("VERDICT", flush=True)
    print("=" * 70, flush=True)

    # Check: did coscosh signal survive polynomial control?
    cc_data = all_profile_data["coscosh"]
    cc_eps = np.array([d["eps"] for d in cc_data])
    cc_tc = np.array([d["total_causal"] for d in cc_data])
    cc_nl = np.array([d["n_links"] for d in cc_data])
    cc_bd = np.array([d["bd_action"] for d in cc_data])
    cc_frob = np.array([d["mag_frobenius"] for d in cc_data])
    cc_med = mediation_polynomial(cc_eps, cc_frob, cc_tc, cc_nl, cc_bd)

    poly_survived = abs(cc_med["r_poly"]) >= 0.10

    # Check: did quadrupole profile show signal?
    qp_data = all_profile_data["quadrupole"]
    qp_eps = np.array([d["eps"] for d in qp_data])
    qp_tc = np.array([d["total_causal"] for d in qp_data])
    qp_nl = np.array([d["n_links"] for d in qp_data])
    qp_bd = np.array([d["bd_action"] for d in qp_data])
    qp_frob = np.array([d["mag_frobenius"] for d in qp_data])
    qp_med = mediation_polynomial(qp_eps, qp_frob, qp_tc, qp_nl, qp_bd)

    quad_signal = abs(qp_med["r_full_lin"]) >= 0.10

    if poly_survived and quad_signal:
        verdict = "GENUINE: survives polynomial TC AND appears with pure quadrupole"
    elif poly_survived and not quad_signal:
        verdict = "QUADRUPOLE TEST NEGATIVE: signal in cos*cosh only (monopole component)"
    elif not poly_survived and quad_signal:
        verdict = "POLYNOMIAL KILLS cos*cosh BUT quadrupole shows signal (complex)"
    else:
        verdict = "BOTH TESTS NEGATIVE: signal is nonlinear-TC artifact + monopole"

    print(f"\n  cos*cosh mag_frob: r|full={cc_med['r_full_lin']:+.3f}, "
          f"r|poly={cc_med['r_poly']:+.3f} ({'SURVIVES' if poly_survived else 'KILLED'})",
          flush=True)
    print(f"  quadrupole mag_frob: r|full={qp_med['r_full_lin']:+.3f} "
          f"({'SIGNAL' if quad_signal else 'NO SIGNAL'})", flush=True)
    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"  Wall time: {total_time:.0f}s ({total_time/60:.1f} min)", flush=True)

    # Save
    meta = ExperimentMeta(route=2, name="d4_followup",
                          description="d=4 falsification: polynomial TC + quadrupole + scaling",
                          N=2000, M=M_ENSEMBLE, status="completed", verdict=verdict)
    meta.wall_time_sec = total_time

    output = {
        "verdict": verdict,
        "coscosh_poly": cc_med,
        "quadrupole_poly": qp_med,
        "wall_time_sec": total_time,
    }
    out_path = RESULTS_DIR / "d4_followup.json"
    save_experiment(meta, output, out_path)
    print(f"  Saved: {out_path}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
