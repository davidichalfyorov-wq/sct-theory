"""
FND-1 EXP-4: d=4 Magnetic Laplacian with Geometric Phases.

Route 7 in d=2 used constant phases (+/-1j) which gave frobenius
algebraically equal to undirected Laplacian. Useless for curvature.

This experiment uses DIRECTION-DEPENDENT phases on each link:
  theta_{ij} = arctan2(y_j - y_i, x_j - x_i)  (azimuthal angle of link)
  A_geo[i,j] = exp(i * theta_{ij})
  L_geo = D - A_geo  (Hermitian)

The azimuthal distribution of links changes under pp-wave curvature
(light cone deformation), so the magnetic spectrum should be curvature-sensitive.

Tests:
  1. Does magnetic Fiedler correlate with eps after TC+BD mediation?
  2. Is spectral_diff (mean |eig_geo - eig_undir|) curvature-sensitive?
  3. Quadrupole profile for pure Weyl test.

Run:
  python analysis/scripts/fnd1_exp4_d4_magnetic_phase.py
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

N_VALUES = [500, 1000, 2000]
N_PRIMARY = 2000
M_ENSEMBLE = 100  # increased from 60 for adequate power at small effect sizes
T_DIAMOND = 1.0
MASTER_SEED = 4477
WORKERS = N_WORKERS

EPS_COSCOSH = [0.0, 0.1, 0.2, 0.3, 0.5]
# TC-stable range: eps<=10 gives <3% TC change. eps>=20 gives 9-25%.
EPS_QUADRUPOLE = [0.0, 2.0, 5.0, 10.0, 20.0]


# ---------------------------------------------------------------------------
# Geometric Magnetic Laplacian
# ---------------------------------------------------------------------------

def build_geometric_magnetic_laplacian(pts, A_link_sp):
    """Build Hermitian magnetic Laplacian with direction-dependent phases.

    For each undirected link {i,j}:
      theta = arctan2(y_j - y_i, x_j - x_i)  (azimuthal angle in x-y plane)
      A_geo[i,j] = exp(+i*theta),  A_geo[j,i] = exp(-i*theta)
    L_geo = D - A_geo  (Hermitian, real eigenvalues)

    Also builds the undirected Laplacian for spectral comparison.
    """
    N = pts.shape[0]
    A_link_dense = A_link_sp.toarray() if sp.issparse(A_link_sp) else A_link_sp

    # Undirected Laplacian (real)
    degrees = np.sum(A_link_dense, axis=1)
    L_undir = np.diag(degrees) - A_link_dense

    # Geometric magnetic adjacency (complex Hermitian) — vectorized
    A_geo = np.zeros((N, N), dtype=np.complex128)
    rows, cols = np.where(np.triu(A_link_dense, 1) > 0)

    dx = pts[cols, 1] - pts[rows, 1]
    dy = pts[cols, 2] - pts[rows, 2]
    theta = np.arctan2(dy, dx)
    phases = np.exp(1j * theta)

    A_geo[rows, cols] = phases
    A_geo[cols, rows] = np.conj(phases)

    L_geo = np.diag(degrees.astype(np.complex128)) - A_geo

    return L_geo, L_undir


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(args):
    """Compute geometric magnetic Laplacian observables for one d=4 sprinkling."""
    seed_int, N, T, eps, profile = args

    rng = np.random.default_rng(seed_int)
    pts = sprinkle_4d_flat(N, T, rng)

    # Build causal matrix
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

    n_matrix, N0, N1, N2, N3 = compute_layers_4d(C)
    bd = bd_action_4d(N, N0, N1, N2, N3)

    # Link graph
    A_link = build_link_graph(C, n_matrix)

    # Geometric magnetic Laplacian + undirected
    L_geo, L_undir = build_geometric_magnetic_laplacian(pts, A_link)

    # Eigenvalues
    eigs_geo = gpu_eigvalsh(L_geo)    # real (Hermitian)
    eigs_undir = gpu_eigvalsh(L_undir)

    # Sort ascending
    eigs_geo = np.sort(eigs_geo)
    eigs_undir = np.sort(eigs_undir)

    # Observables
    # Note: Frobenius norm is ALWAYS equal (|exp(i*theta)|^2=1 tautology).
    # Use per-eigenvalue comparison and higher moments instead.

    # Skip near-zero eigenvalues (disconnected components)
    nz_mask_g = eigs_geo > 0.01
    nz_mask_u = eigs_undir > 0.01
    eigs_g_nz = eigs_geo[nz_mask_g]
    eigs_u_nz = eigs_undir[nz_mask_u]

    # Fiedler: first nonzero eigenvalue
    fiedler_geo = float(eigs_g_nz[0]) if len(eigs_g_nz) > 0 else 0.0
    fiedler_undir = float(eigs_u_nz[0]) if len(eigs_u_nz) > 0 else 0.0

    # Spectral diff: mean |eig_geo - eig_undir| over matched nonzero eigenvalues
    n_match = min(len(eigs_g_nz), len(eigs_u_nz))
    if n_match > 0:
        spectral_diff = float(np.mean(np.abs(eigs_g_nz[:n_match] - eigs_u_nz[:n_match])))
    else:
        spectral_diff = 0.0

    # Median nonzero eigenvalue (robust to outliers)
    median_geo = float(np.median(eigs_g_nz)) if len(eigs_g_nz) > 0 else 0.0
    median_undir = float(np.median(eigs_u_nz)) if len(eigs_u_nz) > 0 else 0.0

    # Spectral entropy of magnetic Laplacian
    if len(eigs_g_nz) > 0 and np.sum(eigs_g_nz) > 0:
        p = eigs_g_nz / np.sum(eigs_g_nz)
        entropy_geo = float(-np.sum(p * np.log(p + 1e-300)))
    else:
        entropy_geo = 0.0

    return {
        "fiedler_geo": fiedler_geo,
        "fiedler_undir": fiedler_undir,
        "spectral_diff": spectral_diff,
        "median_geo": median_geo,
        "entropy_geo": entropy_geo,
        "total_causal": total_causal,
        "bd_action": bd,
        "n_links": N0,
        "eps": eps,
        "profile": profile,
    }


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


def mediation(results, observable="spectral_diff"):
    """Test linear+quadratic eps vs observable, controlling TC+TC^2+BD."""
    eps_arr = np.array([r["eps"] for r in results])
    obs_arr = np.array([r[observable] for r in results])
    tc_arr = np.array([r["total_causal"] for r in results])
    bd_arr = np.array([r["bd_action"] for r in results])
    controls = np.column_stack([tc_arr, tc_arr ** 2, bd_arr, np.ones(len(tc_arr))])

    out = {"n": len(results), "observable": observable}
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    meta = ExperimentMeta(
        route=7, name="exp4_d4_magnetic_phase",
        description="d=4 geometric magnetic Laplacian: direction-dependent phases on links",
        N=N_PRIMARY, M=M_ENSEMBLE, status="running",
    )

    print("=" * 70, flush=True)
    print("FND-1 EXP-4: d=4 GEOMETRIC MAGNETIC LAPLACIAN", flush=True)
    print("=" * 70, flush=True)
    print(f"N values: {N_VALUES}, M={M_ENSEMBLE}", flush=True)
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

                sdiffs = [r["spectral_diff"] for r in raw]
                meds = [r["median_geo"] for r in raw]
                print(f"  {prof_name} N={N} eps={eps:+.1f}: "
                      f"spec_diff={np.mean(sdiffs):.4f}+-{np.std(sdiffs):.4f}"
                      f"  median_geo={np.mean(meds):.2f}"
                      f"  [{elapsed:.1f}s]", flush=True)
                results_N.extend(raw)

            # Mediation for multiple observables
            key = f"{prof_name}_N{N}"
            med_sdiff = mediation(results_N, "spectral_diff")
            med_fiedler = mediation(results_N, "fiedler_geo")
            med_median = mediation(results_N, "median_geo")
            med_entropy = mediation(results_N, "entropy_geo")

            all_results[key] = {
                "mediation_spectral_diff": med_sdiff,
                "mediation_fiedler_geo": med_fiedler,
                "mediation_median_geo": med_median,
                "mediation_entropy_geo": med_entropy,
                "n_sprinklings": len(results_N),
            }

            # Print best observable
            best_obs = max([("spectral_diff", med_sdiff),
                            ("fiedler_geo", med_fiedler),
                            ("median_geo", med_median),
                            ("entropy_geo", med_entropy)],
                           key=lambda x: abs(x[1]["best_r_partial"]))
            print(f"  Best observable: {best_obs[0]}"
                  f" r_partial={best_obs[1]['best_r_partial']:+.4f}"
                  f" p={best_obs[1]['best_p_partial']:.2e}", flush=True)

    # ==================================================================
    # VERDICT
    # ==================================================================

    total_time = time.perf_counter() - t_total

    # Primary: quadrupole at N_PRIMARY
    q_key = f"quadrupole_N{N_PRIMARY}"
    c_key = f"coscosh_N{N_PRIMARY}"
    med_q = all_results.get(q_key, {}).get("mediation_spectral_diff", {})
    med_c = all_results.get(c_key, {}).get("mediation_spectral_diff", {})

    print(f"\n{'=' * 70}", flush=True)
    print(f"COMPARISON at N={N_PRIMARY} (spectral_diff):", flush=True)
    if med_c:
        print(f"  Coscosh:    r_partial={med_c.get('best_r_partial', 0):+.4f}"
              f" ({med_c.get('best', '?')})", flush=True)
    if med_q:
        print(f"  Quadrupole: r_partial={med_q.get('best_r_partial', 0):+.4f}"
              f" ({med_q.get('best', '?')})", flush=True)

    # Find best signal across all observables
    best_signal = 0.0
    best_label = "none"
    for prof in ["coscosh", "quadrupole"]:
        for obs in ["spectral_diff", "fiedler_geo", "median_geo", "entropy_geo"]:
            k = f"{prof}_N{N_PRIMARY}"
            m = all_results.get(k, {}).get(f"mediation_{obs}", {})
            rp = abs(m.get("best_r_partial", 0))
            pp = m.get("best_p_partial", 1.0)
            if rp > best_signal and pp < 0.05:
                best_signal = rp
                best_label = f"{prof}/{obs}"

    if best_signal > 0.10:
        verdict = (f"SIGNAL: {best_label} r_partial={best_signal:.4f}")
    else:
        verdict = (f"NO SIGNAL: best |r_partial|={best_signal:.4f}"
                   f" ({best_label})")

    print(f"\nVERDICT: {verdict}", flush=True)
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

    out_path = RESULTS_DIR / "exp4_d4_magnetic_phase.json"
    save_experiment(meta, output, out_path)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
