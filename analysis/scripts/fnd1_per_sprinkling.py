"""
FND-1: Per-sprinkling multi-operator data collection.

Runs ALL operators on the SAME sprinklings to enable:
1. Cross-operator correlation (do different operators see the same thing?)
2. ML curvature regression (can eps be predicted from spectral data?)
3. Nonlinear mediation (random forest residuals)
4. Eigenvalue distribution analysis (clustering, gaps, universality)
5. Discovery of unexpected patterns

Key: same seed → same causal set → all operators see the same geometry.

Run:
    python analysis/scripts/fnd1_per_sprinkling.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from scipy import stats
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_4d_experiment import (
    sprinkle_4d_flat, causal_matrix_4d, compute_layers_4d, bd_action_4d,
    build_link_graph, link_spectral_embedding, _ppwave_profile,
)
from fnd1_4d_followup import quadrupole_profile
from fnd1_parallel import N_WORKERS, _init_worker

try:
    from fnd1_gpu import gpu_eigvalsh, gpu_eigh
except ImportError:
    gpu_eigvalsh, gpu_eigh = np.linalg.eigvalsh, np.linalg.eigh

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_VALUES = [500, 1000, 2000]
M_PER_EPS = 80
EPS_QUADRUPOLE = [0.0, 2.0, 5.0, 10.0]
T_DIAMOND = 1.0
MASTER_SEED = 55555
WORKERS = N_WORKERS
K_EMBED = 20
N_EIGS_SAVE = 30  # save top-30 eigenvalues per operator for distribution analysis

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "speculative" / "numerics" / "ensemble_results"


# ---------------------------------------------------------------------------
# Worker: compute ALL operators on ONE sprinkling
# ---------------------------------------------------------------------------

def _worker(args):
    """Multi-operator analysis on a single d=4 sprinkling."""
    seed_int, N, T, eps = args

    rng = np.random.default_rng(seed_int)
    pts = sprinkle_4d_flat(N, T, rng)

    # Causal matrix (quadrupole only — clean)
    if abs(eps) < 1e-12:
        C = causal_matrix_4d(pts, 0.0, "flat")
    else:
        t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
        dt = t[np.newaxis, :] - t[:, np.newaxis]
        dx = x[np.newaxis, :] - x[:, np.newaxis]
        dy = y[np.newaxis, :] - y[:, np.newaxis]
        dz = z[np.newaxis, :] - z[:, np.newaxis]
        dr2 = dx ** 2 + dy ** 2 + dz ** 2
        xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
        ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
        f_mid = quadrupole_profile(xm, ym)
        mink = dt ** 2 - dr2
        corr = eps * f_mid * (dt + dz) ** 2 / 2.0
        C = ((mink > corr) & (dt > 0)).astype(np.float64)
        del dx, dy, xm, ym, f_mid, mink, corr

    total_causal = float(np.sum(C))
    n_matrix, N0, N1, N2, N3 = compute_layers_4d(C)
    bd = bd_action_4d(N, N0, N1, N2, N3)

    # ---- OPERATOR 1: Link-graph Laplacian ----
    A_link = build_link_graph(C, n_matrix)
    degrees_link = np.array(A_link.sum(axis=1)).ravel()
    L_link = sp.diags(degrees_link) - A_link
    L_dense = L_link.toarray()

    evals_link, evecs_link = gpu_eigh(L_dense)
    # Skip zero eigenvalues (disconnected components)
    skip = max(1, int(np.sum(evals_link < 1e-8)))
    nz_evals = evals_link[skip:]
    fiedler = float(nz_evals[0]) if len(nz_evals) > 0 else 0.0
    # Spectral gap ratio
    gap_ratio = float(nz_evals[0] / nz_evals[1]) if len(nz_evals) > 1 and nz_evals[1] > 1e-15 else 0.0
    mean_link_deg = float(A_link.sum() / N)

    # Geometry recovery
    rng2 = np.random.default_rng(seed_int + 999)
    n_pairs = min(5000, N * 2)
    idx_i = rng2.integers(0, N, size=n_pairs)
    idx_j = rng2.integers(0, N, size=n_pairs)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    d_eucl = np.sqrt(np.sum((pts[idx_i] - pts[idx_j]) ** 2, axis=1))

    # Best-k embedding distance
    skip = max(1, np.sum(evals_link < 1e-8))
    k_eff = min(K_EMBED, evecs_link.shape[1] - skip)
    if k_eff > 0:
        emb = evecs_link[:, skip:skip + k_eff]
        d_emb = np.sqrt(np.sum((emb[idx_i] - emb[idx_j]) ** 2, axis=1))
        rho_geom, _ = stats.spearmanr(d_emb, d_eucl)
    else:
        rho_geom = 0.0

    # ---- OPERATOR 2: BD Commutator [H,M] ----
    V = np.pi * T ** 4 / 24.0
    rho_density = N / V
    scale = np.sqrt(rho_density)
    past = C.T
    n_past = n_matrix.T
    L_bd = np.zeros((N, N))
    for layer, coeff in enumerate([4, -36, 64, -32]):
        mask_layer = (past > 0) & (n_past == layer)
        L_bd[mask_layer] = coeff * scale

    L_sp = sp.csr_matrix(L_bd)
    comm = ((L_sp.T @ L_sp - L_sp @ L_sp.T) / 2.0).toarray()
    comm = (comm + comm.T) / 2.0  # enforce symmetry
    comm_eigs = gpu_eigvalsh(comm)

    comm_abs = np.abs(comm_eigs)
    s_total = float(np.sum(comm_abs))
    if s_total > 0:
        p_comm = comm_abs / s_total
        comm_entropy = float(-np.sum(p_comm * np.log(p_comm + 1e-300)))
    else:
        comm_entropy = 0.0
    comm_frobenius = float(np.sqrt(np.sum(comm_eigs ** 2)))
    comm_max = float(np.max(comm_abs))

    # ---- OPERATOR 3: Link Laplacian Heat Trace ----
    lam_nz = evals_link[evals_link > 1e-8]
    tau_grid = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])
    if len(lam_nz) > 0:
        K_ht = np.sum(np.exp(-tau_grid[:, None] * lam_nz[None, :]), axis=1) / N
        t2K = tau_grid ** 2 * K_ht
    else:
        K_ht = np.zeros(len(tau_grid))
        t2K = np.zeros(len(tau_grid))

    # ---- EIGENVALUE DISTRIBUTION ----
    # Save top eigenvalues for distribution analysis
    link_top = evals_link[-N_EIGS_SAVE:].tolist() if len(evals_link) >= N_EIGS_SAVE else evals_link.tolist()
    comm_top = sorted(comm_abs, reverse=True)[:N_EIGS_SAVE]

    # ---- NOVEL OBSERVABLES ----
    # Eigenvalue entropy of link Laplacian (information content)
    lam_pos = evals_link[evals_link > 1e-10]
    if len(lam_pos) > 0:
        p_link = lam_pos / np.sum(lam_pos)
        link_entropy = float(-np.sum(p_link * np.log(p_link)))
    else:
        link_entropy = 0.0

    # Eigenvalue ratio universality: lambda_1/lambda_2, lambda_2/lambda_3, etc. (nonzero only)
    ratios = []
    for i in range(min(4, len(nz_evals) - 1)):
        if nz_evals[i + 1] > 1e-15:
            ratios.append(float(nz_evals[i] / nz_evals[i + 1]))

    # Algebraic connectivity vs mean degree ratio
    alg_conn_ratio = fiedler / mean_link_deg if mean_link_deg > 0 else 0.0

    # Trace of C^2 (counts 2-paths) normalized
    trace_C2 = float(np.sum(n_matrix)) / (N * N)

    return {
        "seed": seed_int,
        "N": N,
        "eps": eps,
        # Confounders
        "total_causal": total_causal,
        "bd_action": bd,
        "mean_link_deg": mean_link_deg,
        "n_links": int(N0),
        "trace_C2": trace_C2,
        # Operator 1: Link Fiedler
        "fiedler": fiedler,
        "gap_ratio": gap_ratio,
        "rho_geom": float(rho_geom),
        "link_entropy": link_entropy,
        "alg_conn_ratio": alg_conn_ratio,
        # Operator 2: Commutator
        "comm_entropy": comm_entropy,
        "comm_frobenius": comm_frobenius,
        "comm_max": comm_max,
        # Operator 3: Heat trace
        "t2K_tau1": float(t2K[3]) if len(t2K) > 3 else 0.0,  # tau=1.0
        "t2K_tau5": float(t2K[5]) if len(t2K) > 5 else 0.0,  # tau=5.0
        # Distribution data
        "link_top_eigs": link_top,
        "comm_top_eigs": comm_top,
        # Universality ratios
        "eig_ratios": ratios,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_total = time.perf_counter()

    print("=" * 70)
    print("FND-1: PER-SPRINKLING MULTI-OPERATOR DATA COLLECTION")
    print("=" * 70)
    print(f"N values: {N_VALUES}, M={M_PER_EPS}, eps: {EPS_QUADRUPOLE}")
    print(f"Workers: {WORKERS}")
    print(f"Operators: link_laplacian, commutator, heat_trace")
    print(f"Profile: quadrupole only (pure Weyl, TC-stable)")
    print()

    # Benchmark
    print("=== BENCHMARK ===")
    for N in N_VALUES:
        t0 = time.perf_counter()
        _worker((42, N, T_DIAMOND, 0.0))
        dt = time.perf_counter() - t0
        tasks = M_PER_EPS * len(EPS_QUADRUPOLE)
        par = tasks * dt / WORKERS / 60
        print(f"  N={N}: {dt:.2f}s/task, ~{par:.1f} min ({WORKERS}w)")

    ss = np.random.SeedSequence(MASTER_SEED)
    all_data = []

    for N in N_VALUES:
        print(f"\n{'=' * 60}")
        print(f"N = {N}")
        print("=" * 60)

        for eps in EPS_QUADRUPOLE:
            eps_ss = ss.spawn(1)[0]
            seeds = eps_ss.spawn(M_PER_EPS)
            seed_ints = [int(s.generate_state(1)[0]) for s in seeds]
            args = [(si, N, T_DIAMOND, eps) for si in seed_ints]

            t0 = time.perf_counter()
            with Pool(WORKERS, initializer=_init_worker) as pool:
                raw = pool.map(_worker, args)
            elapsed = time.perf_counter() - t0

            # Quick summary
            fiedlers = [r["fiedler"] for r in raw]
            comms = [r["comm_entropy"] for r in raw]
            rhos = [r["rho_geom"] for r in raw]
            print(f"  eps={eps:+5.1f}: fiedler={np.mean(fiedlers):.4f}"
                  f"  comm_S={np.mean(comms):.4f}"
                  f"  rho={np.mean(rhos):+.4f}"
                  f"  [{elapsed:.1f}s]", flush=True)

            all_data.extend(raw)

    total_time = time.perf_counter() - t_total

    # Cross-operator correlations
    print(f"\n{'=' * 70}")
    print("CROSS-OPERATOR CORRELATIONS (all N pooled)")
    print("=" * 70)

    fiedler_arr = np.array([r["fiedler"] for r in all_data])
    comm_arr = np.array([r["comm_entropy"] for r in all_data])
    rho_arr = np.array([r["rho_geom"] for r in all_data])
    t2K_arr = np.array([r["t2K_tau1"] for r in all_data])
    eps_arr = np.array([r["eps"] for r in all_data])
    tc_arr = np.array([r["total_causal"] for r in all_data])

    observables = {
        "fiedler": fiedler_arr,
        "comm_entropy": comm_arr,
        "rho_geom": rho_arr,
        "t2K_tau1": t2K_arr,
    }

    cross_corr = {}
    for name_a, arr_a in observables.items():
        for name_b, arr_b in observables.items():
            if name_a >= name_b:
                continue
            r, p = stats.spearmanr(arr_a, arr_b)
            cross_corr[f"{name_a}_vs_{name_b}"] = {"rho": round(float(r), 4), "p": float(p)}
            print(f"  {name_a} vs {name_b}: rho={r:+.4f} (p={p:.2e})")

    # Curvature regression (simple: can eps be predicted?)
    print(f"\n{'=' * 70}")
    print("CURVATURE PREDICTABILITY (Spearman with eps)")
    print("=" * 70)

    predictability = {}
    for name, arr in observables.items():
        r_raw, p_raw = stats.spearmanr(eps_arr, arr)
        # Partial (control TC + BD)
        from numpy.linalg import lstsq
        bd_arr = np.array([r["bd_action"] for r in all_data])
        ctrl = np.column_stack([tc_arr, tc_arr ** 2, bd_arr, np.ones(len(tc_arr))])
        cx, _, _, _ = lstsq(ctrl, eps_arr, rcond=None)
        cy, _, _, _ = lstsq(ctrl, arr, rcond=None)
        rx = eps_arr - ctrl @ cx
        ry = arr - ctrl @ cy
        if np.std(rx) > 1e-15 and np.std(ry) > 1e-15:
            r_part, p_part = stats.pearsonr(rx, ry)
        else:
            r_part, p_part = 0.0, 1.0

        predictability[name] = {
            "r_raw": round(float(r_raw), 4),
            "r_partial": round(float(r_part), 4),
            "p_partial": float(p_part),
        }
        print(f"  {name}: r_raw={r_raw:+.4f}, r_partial={r_part:+.4f} (p={p_part:.2e})")

    # Eigenvalue ratio universality
    print(f"\n{'=' * 70}")
    print("EIGENVALUE RATIO UNIVERSALITY")
    print("=" * 70)

    # Check if eigenvalue ratios are constant across eps
    for ratio_idx in range(min(4, len(all_data[0].get("eig_ratios", [])))):
        flat_ratios = [r["eig_ratios"][ratio_idx] for r in all_data
                       if abs(r["eps"]) < 0.01 and len(r["eig_ratios"]) > ratio_idx]
        curved_ratios = [r["eig_ratios"][ratio_idx] for r in all_data
                         if r["eps"] > 5 and len(r["eig_ratios"]) > ratio_idx]
        if flat_ratios and curved_ratios:
            mean_f = np.mean(flat_ratios)
            mean_c = np.mean(curved_ratios)
            t_stat, p_val = stats.ttest_ind(flat_ratios, curved_ratios)
            stable = "UNIVERSAL" if p_val > 0.01 else "CURVATURE-DEPENDENT"
            print(f"  lambda_{ratio_idx+1}/lambda_{ratio_idx+2}: "
                  f"flat={mean_f:.4f}, curved={mean_c:.4f}, p={p_val:.2e} [{stable}]")

    # Save everything
    print(f"\n{'=' * 70}")
    print(f"Wall time: {total_time:.0f}s ({total_time / 60:.1f} min)")
    print("=" * 70)

    # Save compact version (without eigenvalue arrays for JSON size)
    compact = []
    for r in all_data:
        c = {k: v for k, v in r.items() if k not in ("link_top_eigs", "comm_top_eigs")}
        compact.append(c)

    output = {
        "_meta": {
            "name": "per_sprinkling_multioperator",
            "N_values": N_VALUES, "M": M_PER_EPS,
            "eps_values": EPS_QUADRUPOLE,
            "n_sprinklings": len(all_data),
            "wall_time_sec": total_time,
        },
        "cross_correlations": cross_corr,
        "curvature_predictability": predictability,
        "per_sprinkling": compact,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "per_sprinkling_multioperator.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=1, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
