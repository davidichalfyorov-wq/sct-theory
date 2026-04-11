"""
FND-1 EXP-11: d=3 Intermediate Dimension.

d=3: Weyl tensor = 0 (like d=2), but Ricci has 6 independent components
(vs 1 in d=2). If d=4 signal is Weyl-specific -> d=3 should be null.
If signal is from non-conformal Ricci -> d=3 might show signal.

Metric: ds^2 = -(1 + eps*r_m^2)*dt^2 + dx^2 + dy^2
This has R != 0 and changes the causal structure (verified: TC changes ~3% at eps=1).

Tests link-graph Fiedler + spectral embedding (same as EXP-1 but in d=3).

Run:
  python analysis/scripts/fnd1_exp11_d3_intermediate.py
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

from fnd1_experiment_registry import (
    ExperimentMeta, save_experiment, RESULTS_DIR,
)
from fnd1_parallel import N_WORKERS, _init_worker
try:
    from fnd1_gpu import gpu_eigh
except ImportError:
    gpu_eigh = np.linalg.eigh

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

N_VALUES = [500, 1000, 2000, 3000]
N_PRIMARY = 2000
M_ENSEMBLE = 80
T_DIAMOND = 1.0
MASTER_SEED = 6611
WORKERS = N_WORKERS

# Curved metric: ds^2 = -(1+eps*r^2)*dt^2 + dx^2 + dy^2
EPS_CURVED = [0.0, 0.5, 1.0, 2.0, 4.0]
K_VALUES_EMBED = [2, 5, 10, 20]
K_MAX = 20
N_DISTANCE_PAIRS = 8000


# ---------------------------------------------------------------------------
# d=3 Sprinkling and Causal Matrix
# ---------------------------------------------------------------------------

def sprinkle_3d_flat(N, T, rng):
    """Sprinkle N points into d=3 Minkowski causal diamond.

    Diamond: {(t,x,y) : |t| + sqrt(x^2+y^2) < T/2}
    """
    pts = np.empty((N, 3))
    count = 0
    half = T / 2.0
    while count < N:
        batch = max(N - count, 500) * 8
        cands = rng.uniform(-half, half, size=(batch, 3))
        r = np.sqrt(cands[:, 1] ** 2 + cands[:, 2] ** 2)
        inside = np.abs(cands[:, 0]) + r < half
        valid = cands[inside]
        n_take = min(len(valid), N - count)
        pts[count:count + n_take] = valid[:n_take]
        count += n_take
    return pts[np.argsort(pts[:, 0])]


def causal_matrix_3d(pts, eps=0.0):
    """Build causal matrix for d=3 metric.

    Flat: dt^2 > dx^2 + dy^2
    Curved: (1+eps*r_m^2)*dt^2 > dx^2 + dy^2
    """
    t = pts[:, 0]; x = pts[:, 1]; y = pts[:, 2]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dr2 = dx ** 2 + dy ** 2

    if abs(eps) > 1e-12:
        rm2 = ((x[np.newaxis, :] + x[:, np.newaxis]) / 2.0) ** 2 + \
              ((y[np.newaxis, :] + y[:, np.newaxis]) / 2.0) ** 2
        C = (((1.0 + eps * rm2) * dt ** 2 > dr2) & (dt > 0)).astype(np.float64)
    else:
        C = ((dt ** 2 > dr2) & (dt > 0)).astype(np.float64)
    return C


# ---------------------------------------------------------------------------
# d=3 Link Graph and Layers
# ---------------------------------------------------------------------------

def build_link_graph_3d(C):
    """Build link adjacency (Hasse diagram) from causal matrix."""
    C_sp = sp.csr_matrix(C)
    n_matrix = (C_sp @ C_sp).toarray()
    past = C.T
    n_past = n_matrix.T
    link_mask = ((past > 0) & (n_past == 0)).astype(np.float64)
    A = link_mask + link_mask.T
    A = (A > 0).astype(np.float64)
    n_links = int(np.sum(link_mask))
    return sp.csr_matrix(A), n_links, n_matrix


def bd_action_3d(N, n_matrix, C):
    """d=3 BD action from Dowker-Glaser arXiv:1305.2588, Table 1.

    n_d = 3 layers (not 4). Coefficients: C = {1, -27/8, 9/4}.
    alpha_3 = -(pi/(3*sqrt(2)))^{2/3} / Gamma(5/3).
    S = -alpha_3 * (N - N0 + (27/8)*N1 - (9/4)*N2).
    """
    import math
    past = C.T
    n_past = n_matrix.T
    N0 = int(np.sum((past > 0) & (n_past == 0)))  # links
    N1 = int(np.sum((past > 0) & (n_past == 1)))  # 2-element intervals
    N2 = int(np.sum((past > 0) & (n_past == 2)))  # 3-element intervals
    alpha_3 = -(math.pi / (3 * math.sqrt(2))) ** (2 / 3) / math.gamma(5 / 3)
    return float(-alpha_3 * (N - N0 + (27 / 8) * N1 - (9 / 4) * N2))


def spectral_embedding_3d(A_link, k, N):
    """Spectral embedding from link-graph Laplacian (dense, robust)."""
    degrees = np.array(A_link.sum(axis=1)).ravel()
    L = sp.diags(degrees) - A_link
    L_dense = L.toarray() if sp.issparse(L) else L
    evals, evecs = gpu_eigh(L_dense)

    start = 0
    while start < len(evals) and evals[start] < 1e-8:
        start += 1
    end = min(start + k, len(evals))
    if end <= start:
        return np.zeros((N, 1)), np.array([0.0])

    lams = evals[start:end]
    vecs = evecs[:, start:end]
    embedding = vecs * np.sqrt(np.maximum(lams, 0))[np.newaxis, :]
    return embedding, lams


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(args):
    """Compute link-graph observables for one d=3 sprinkling."""
    seed_int, N, T, eps = args

    rng = np.random.default_rng(seed_int)
    pts = sprinkle_3d_flat(N, T, rng)
    C = causal_matrix_3d(pts, eps)

    total_causal = float(np.sum(C))
    A_link, n_links, n_matrix = build_link_graph_3d(C)
    bd = bd_action_3d(N, n_matrix, C)
    mean_link_deg = float(A_link.sum() / N)

    # Spectral embedding
    embedding, link_lams = spectral_embedding_3d(A_link, K_MAX, N)
    fiedler = float(link_lams[0]) if len(link_lams) > 0 else 0.0

    # Distance pairs
    rng2 = np.random.default_rng(seed_int + 999)
    idx_i = rng2.integers(0, N, size=N_DISTANCE_PAIRS)
    idx_j = rng2.integers(0, N, size=N_DISTANCE_PAIRS)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    d_eucl = np.sqrt(np.sum((pts[idx_i] - pts[idx_j]) ** 2, axis=1))

    # Null model
    rng_null = np.random.default_rng(seed_int + 55555)
    perm = rng_null.permutation(N)
    d_eucl_shuf = np.sqrt(np.sum((pts[perm][idx_i] - pts[perm][idx_j]) ** 2, axis=1))

    # Multi-k embedding
    embed_by_k = {}
    for k in K_VALUES_EMBED:
        k_eff = min(k, embedding.shape[1])
        if k_eff > 0:
            emb = embedding[:, :k_eff]
            d_emb = np.sqrt(np.sum((emb[idx_i] - emb[idx_j]) ** 2, axis=1))
            rho_sp, _ = stats.spearmanr(d_emb, d_eucl)
            r_null, _ = stats.pearsonr(d_emb, d_eucl_shuf)
        else:
            rho_sp = r_null = 0.0
        embed_by_k[k] = {"rho_spearman": float(rho_sp), "r_null": float(r_null)}

    best_k = max(K_VALUES_EMBED, key=lambda k: embed_by_k[k]["rho_spearman"])

    return {
        "fiedler": fiedler,
        "rho_spearman": float(embed_by_k[best_k]["rho_spearman"]),
        "r_null": float(embed_by_k[best_k]["r_null"]),
        "best_k": best_k,
        "embed_by_k": embed_by_k,
        "total_causal": total_causal,
        "bd_action": bd,
        "mean_link_deg": mean_link_deg,
        "n_links": n_links,
        "eps": eps,
    }


# ---------------------------------------------------------------------------
# Mediation
# ---------------------------------------------------------------------------

def partial_corr(x, y, controls):
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
    """Test fiedler vs eps (linear+quadratic), controlling TC+TC^2+BD."""
    eps_arr = np.array([r["eps"] for r in results])
    fiedler_arr = np.array([r["fiedler"] for r in results])
    tc_arr = np.array([r["total_causal"] for r in results])
    bd_arr = np.array([r["bd_action"] for r in results])
    controls = np.column_stack([tc_arr, tc_arr ** 2, bd_arr, np.ones(len(tc_arr))])

    out = {"n": len(results)}
    for pred_name, pred in [("linear", eps_arr), ("quadratic", eps_arr ** 2)]:
        r_d, _ = stats.pearsonr(pred, fiedler_arr)
        r_p, p_p = partial_corr(pred, fiedler_arr, controls)
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
        route=2, name="exp11_d3_intermediate",
        description="d=3 intermediate: Weyl=0, test if link Fiedler detects Ricci curvature",
        N=N_PRIMARY, M=M_ENSEMBLE, status="running",
    )

    print("=" * 70, flush=True)
    print("FND-1 EXP-11: d=3 INTERMEDIATE DIMENSION", flush=True)
    print("=" * 70, flush=True)
    print(f"N values: {N_VALUES}, M={M_ENSEMBLE}", flush=True)
    print(f"Eps: {EPS_CURVED}", flush=True)
    print(f"d=3: Weyl=0, Ricci nontrivial", flush=True)
    print(flush=True)

    # Benchmark
    print("=== BENCHMARK ===", flush=True)
    for N in N_VALUES:
        t0 = time.perf_counter()
        _worker((42, N, T_DIAMOND, 0.0))
        dt = time.perf_counter() - t0
        tasks = M_ENSEMBLE * len(EPS_CURVED)
        print(f"  N={N:5d}: {dt:.3f}s/task, parallel: {tasks * dt / WORKERS / 60:.1f} min",
              flush=True)

    ss = np.random.SeedSequence(MASTER_SEED)
    results_by_N = {}

    for N in N_VALUES:
        print(f"\n{'=' * 60}", flush=True)
        print(f"N = {N}", flush=True)
        print("=" * 60, flush=True)

        all_results = []
        for eps in EPS_CURVED:
            eps_ss = ss.spawn(1)[0]
            seeds = eps_ss.spawn(M_ENSEMBLE)
            seed_ints = [int(s.generate_state(1)[0]) for s in seeds]
            args = [(si, N, T_DIAMOND, eps) for si in seed_ints]

            t0 = time.perf_counter()
            with Pool(WORKERS, initializer=_init_worker) as pool:
                raw = pool.map(_worker, args)
            elapsed = time.perf_counter() - t0

            fiedlers = [r["fiedler"] for r in raw]
            rhos = [r["rho_spearman"] for r in raw]
            print(f"  eps={eps:+.2f}: fiedler={np.mean(fiedlers):.4f}+-{np.std(fiedlers):.4f}"
                  f"  rho_best={np.mean(rhos):+.4f}"
                  f"  TC={np.mean([r['total_causal'] for r in raw]):.0f}"
                  f"  [{elapsed:.1f}s]", flush=True)
            all_results.extend(raw)

        med = mediation(all_results)
        # Geometry recovery at flat
        flat_r = [r for r in all_results if abs(r["eps"]) < 1e-10]
        rho_flat = float(np.mean([r["rho_spearman"] for r in flat_r])) if flat_r else 0.0
        null_flat = float(np.mean([r["r_null"] for r in flat_r])) if flat_r else 0.0

        results_by_N[str(N)] = {
            "mediation": med,
            "rho_flat": rho_flat,
            "null_flat": null_flat,
            "n_sprinklings": len(all_results),
        }

        print(f"  Mediation ({med['best']}): r_partial={med['best_r_partial']:+.4f}"
              f" p={med['best_p_partial']:.2e}", flush=True)
        print(f"  Geometry (flat): rho={rho_flat:+.4f}, null={null_flat:+.4f}", flush=True)

    # ==================================================================
    # SCALING + VERDICT
    # ==================================================================

    total_time = time.perf_counter() - t_total

    print(f"\n{'=' * 70}", flush=True)
    print("SCALING", flush=True)
    print("=" * 70, flush=True)

    Ns = np.array(N_VALUES, dtype=float)
    partial_rs = []
    rho_flats = []
    for N in N_VALUES:
        r = results_by_N[str(N)]
        med = r["mediation"]
        partial_rs.append(med["best_r_partial"])
        rho_flats.append(r["rho_flat"])
        print(f"  N={N:5d}: r_partial={med['best_r_partial']:+.4f}"
              f" p={med['best_p_partial']:.2e}"
              f" rho_flat={r['rho_flat']:+.4f}", flush=True)

    med_p = results_by_N[str(N_PRIMARY)]["mediation"]

    if abs(med_p["best_r_partial"]) > 0.10 and med_p["best_p_partial"] < 0.01:
        curv_verdict = (f"d=3 DETECTS CURVATURE: r_partial={med_p['best_r_partial']:+.4f}")
    else:
        curv_verdict = (f"d=3 NULL: r_partial={med_p['best_r_partial']:+.4f}"
                        f" (Weyl=0, no additional signal)")

    rho_best = rho_flats[-1]
    geo_verdict = (f"rho={rho_best:.4f}" if rho_best > 0.3
                   else f"rho={rho_best:.4f} (weak)")

    verdict = f"{curv_verdict} | Geometry: {geo_verdict}"

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
            "eps_curved": EPS_CURVED,
            "k_values_embed": K_VALUES_EMBED,
            "n_distance_pairs": N_DISTANCE_PAIRS,
            "metric": "-(1+eps*r_m^2)*dt^2 + dx^2 + dy^2",
        },
        "results_by_N": results_by_N,
        "verdict": verdict,
        "wall_time_sec": total_time,
    }

    out_path = RESULTS_DIR / "exp11_d3_intermediate.json"
    save_experiment(meta, output, out_path)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
