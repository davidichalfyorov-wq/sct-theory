"""
FND-1 Route 2: Link-Graph Laplacian CRN Experiment (d=4)
=========================================================

WHAT THIS DOES
--------------
Previous Route 2 experiments (EXP-12) showed the link-graph Laplacian
spectral embedding recovers spacetime geometry with rho=0.936 at N=5000
in d=2.  This experiment extends that to d=4 using the CRN (Common
Random Numbers) design from the GTA pipeline.

The link graph (Hasse diagram) is the subset of causal relations where
two events are directly connected — no other events lie between them.
Its graph Laplacian L_link = D - A_link encodes the local connectivity
of the discrete spacetime.

The heat trace K(t) = Tr(exp(-t*L_link)) = sum_i exp(-lambda_i * t)
is a standard tool in spectral geometry.  On a smooth d-dimensional
manifold, K(t) has the asymptotic expansion

    K(t) ~ (4*pi*t)^{-d/2} * integral[ 1 + a_1*R*t + a_2*(R^2+...)*t^2 + ... ]

so the heat trace coefficients encode curvature information.  If the
link-graph Laplacian is a good discrete approximation, its heat trace
should also be sensitive to curvature.

CRN DESIGN
----------
Same as the GTA pipeline: sprinkle N points once, compute the link-graph
Laplacian TWICE (flat vs curved), measure the paired difference in
observables.  Any difference must come from the metric change.

OBSERVABLES
-----------
  1. link_count:  number of links (layer-0 pairs)
  2. mean_degree: average link degree
  3. fiedler:     smallest nonzero eigenvalue (spectral gap)
  4. heat_trace:  K(t) at several t values (low eigenvalues dominate)
  5. spectral_dim: d_S(t) = -2 * d(log K)/d(log t) at several t
  6. embedding_rho: Spearman correlation between spectral embedding
                    distances and Euclidean spacetime distances (k=4)

WHY THIS IS FAST
----------------
The link graph is sparse: degree ~15-20 at N=10k.  The Laplacian
eigsh(k=200) on a sparse matrix takes ~5-30 seconds, compared to
~75 seconds for dense eigvalsh of the commutator in the GTA pipeline.
The bottleneck is the causal matrix construction (O(N^2)), not the
eigendecomposition.  This means we can run N=10k in ~40 seconds total,
and the experiments can run in parallel with the GTA pipeline.

USAGE
-----
    python fnd1_route2_crn.py --metric ppwave_quad --N 5000 --M 40
    python fnd1_route2_crn.py --all --N 5000 --M 40
    python fnd1_route2_crn.py --calibrate
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, ArpackError
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MASTER_SEED = 27182  # different from GTA pipeline (31415) to avoid seed overlap
CHECKPOINT_DIR = Path("checkpoints_r2")

METRICS = {
    "ppwave_quad":   {"label": "PP-wave x²-y²",  "eps": [0.0, 2.0, 5.0, 10.0]},
    "ppwave_cross":  {"label": "PP-wave xy",      "eps": [0.0, 2.0, 5.0, 10.0]},
    "schwarzschild": {"label": "Weak Schwarzschild",
                      "eps": [0.0, 0.005, 0.01, 0.02]},
    "flrw":          {"label": "FLRW conformally flat",
                      "eps": [0.0, 0.5, 1.0, 2.0]},
    "conformal":     {"label": "Conformal NULL",  "eps": [0.0, 5.0]},
}

METRIC_SEED_OFFSETS = {
    "ppwave_quad": 100, "ppwave_cross": 200, "schwarzschild": 300,
    "flrw": 400, "conformal": 500,
}

# Heat trace evaluation times (log-spaced)
T_HEAT = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

# Number of bottom eigenvalues to compute
K_EIGSH = 200

# Embedding dimension for geometry recovery test
K_EMBED = 4

# Number of random distance pairs for embedding quality
N_DISTANCE_PAIRS = 10000


# ---------------------------------------------------------------------------
# Sprinkling (same as GTA pipeline)
# ---------------------------------------------------------------------------
def sprinkle(N: int, T: float, rng) -> np.ndarray:
    """Poisson sprinkle into 4D Alexandrov interval (causal diamond).
    Rejection sampling, sorted by time coordinate."""
    pts = np.empty((N, 4))
    count, half = 0, T / 2.0
    while count < N:
        batch = max(N - count, 1000) * 10
        c = rng.uniform(-half, half, size=(batch, 4))
        r = np.sqrt(c[:, 1]**2 + c[:, 2]**2 + c[:, 3]**2)
        v = c[np.abs(c[:, 0]) + r < half]
        n = min(len(v), N - count)
        pts[count:count+n] = v[:n]
        count += n
    return pts[np.argsort(pts[:, 0])]


# ---------------------------------------------------------------------------
# Causal conditions (identical to GTA pipeline — verified by hand)
# ---------------------------------------------------------------------------
def causal_flat(pts) -> np.ndarray:
    t = pts[:, 0]; x = pts[:, 1]; y = pts[:, 2]; z = pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx**2 + dy**2 + dz**2
    del dx, dy, dz
    C = ((dt**2 > dr2) & (dt > 0)).astype(np.float64)
    del dt, dr2
    return C


def causal_ppwave_quad(pts, eps) -> np.ndarray:
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx**2 + dy**2 + dz**2
    del dx, dy
    xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
    ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
    f = xm**2 - ym**2; del xm, ym
    mink = dt**2 - dr2; del dr2
    corr = eps * f * (dt + dz)**2 / 2.0; del f
    C = ((mink > corr) & (dt > 0)).astype(np.float64)
    del dt, dz, mink, corr
    return C


def causal_ppwave_cross(pts, eps) -> np.ndarray:
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx**2 + dy**2 + dz**2
    del dx, dy
    xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
    ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
    f = xm * ym; del xm, ym
    mink = dt**2 - dr2; del dr2
    corr = eps * f * (dt + dz)**2 / 2.0; del f
    C = ((mink > corr) & (dt > 0)).astype(np.float64)
    del dt, dz, mink, corr
    return C


def causal_schwarzschild(pts, eps) -> np.ndarray:
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx**2 + dy**2 + dz**2
    del dx, dy, dz
    xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
    ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
    zm = (z[np.newaxis, :] + z[:, np.newaxis]) / 2.0
    rm = np.sqrt(xm**2 + ym**2 + zm**2) + 0.3
    del xm, ym, zm
    Phi = -eps / rm; del rm
    C = (((1 + 2*Phi) * dt**2 > (1 - 2*Phi) * dr2) & (dt > 0)).astype(np.float64)
    del dt, dr2, Phi
    return C


def causal_flrw(pts, eps) -> np.ndarray:
    # NOTE: FLRW is conformally flat. The exact causal condition (via conformal
    # time integral) gives an identical causal matrix to Minkowski.  This midpoint
    # approximation dt^2 > a^2(t_m)*dr^2 introduces a SYSTEMATIC difference from
    # the flat condition that does NOT cancel in CRN.  Any detected FLRW signal
    # reflects the midpoint approximation error, not curvature.  Standard in causal
    # set literature but results must be interpreted as "coordinate effect control."
    t, x, y, z = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx**2 + dy**2 + dz**2
    del dx, dy, dz
    tm = (t[np.newaxis, :] + t[:, np.newaxis]) / 2.0
    a2 = (1 + eps * tm**2)**2; del tm
    C = ((dt**2 > a2 * dr2) & (dt > 0)).astype(np.float64)
    del dt, dr2, a2
    return C


def causal_conformal(pts, eps) -> np.ndarray:
    return causal_flat(pts)


METRIC_FNS = {
    "ppwave_quad":   causal_ppwave_quad,
    "ppwave_cross":  causal_ppwave_cross,
    "schwarzschild": causal_schwarzschild,
    "flrw":          causal_flrw,
    "conformal":     causal_conformal,
}


# ---------------------------------------------------------------------------
# Link graph construction
# ---------------------------------------------------------------------------
def build_link_graph(C: np.ndarray) -> sp.csr_matrix:
    """Build undirected link (Hasse) adjacency from causal matrix.

    A link i->j exists when j is in the causal future of i with zero
    intervening elements: C[i,j]=1 and (C @ C)[i,j]=0.

    The link graph is SPARSE: mean degree measured by --calibrate.
    This is the key advantage over the commutator (80% dense).

    Memory: C_sp and C^2 are stored as sparse CSR.  C^2 has fill
    comparable to C (~30-40% at large N, not 5%), but CSR avoids
    dense N×N allocation.  Peak memory ~2 GB at N=10k (dense C held
    by caller + C_sp + C^2 coexist; del C only removes local ref).
    """
    C_sp = sp.csr_matrix(C)
    del C; gc.collect()

    # n_between[i,j] = number of elements causally between i and j
    C2_sp = C_sp @ C_sp

    # Link = causal AND zero intervening: keep entries where
    # C[i,j]=1 and C2[i,j]=0.  Subtract to remove non-links:
    # link = C where C2 is zero.  Use boolean NOT via (C2 != 0).
    has_intervening = (C2_sp != 0).astype(np.float64)
    link_sp = C_sp - C_sp.multiply(has_intervening)
    link_sp.eliminate_zeros()
    del C_sp, C2_sp, has_intervening; gc.collect()

    # Symmetrize (undirected graph)
    A = link_sp + link_sp.T
    A = (A > 0.5).astype(np.float64)
    return A.tocsr()


def build_laplacian(A_sp: sp.csr_matrix) -> sp.csr_matrix:
    """Combinatorial graph Laplacian L = D - A."""
    degrees = np.array(A_sp.sum(axis=1)).ravel()
    D = sp.diags(degrees)
    return (D - A_sp).tocsr()


# ---------------------------------------------------------------------------
# Observables from link-graph Laplacian
# ---------------------------------------------------------------------------
def compute_observables(C: np.ndarray, pts: np.ndarray, N: int,
                        k_eigsh: int, rng_dist) -> dict:
    """Compute all Route 2 observables from one causal matrix.

    Returns dict with: link_count, mean_degree, fiedler,
    heat_trace (at T_HEAT), spectral_dim, embedding_rho.
    """
    A_link = build_link_graph(C)
    n_links = int(A_link.sum()) // 2
    mean_degree = float(A_link.sum() / N)

    L = build_laplacian(A_link)
    del A_link; gc.collect()

    # Bottom-k eigenvalues + eigenvectors (sparse eigsh, single call).
    # which='SA' (smallest algebraic) is correct for PSD Laplacian.
    # DO NOT use which='SM' — it internally inverts L, which is singular
    # (graph Laplacian always has eigenvalue 0).
    # v0=ones(N) is a fixed starting vector so that eigsh is deterministic —
    # without this, the conformal null test gives ~1e-7 instead of exact 0.
    k_actual = min(k_eigsh, N - 2)
    v0 = np.ones(N, dtype=np.float64) / np.sqrt(N)
    try:
        evals, evecs = eigsh(L.astype(np.float64), k=k_actual,
                             which='SA', v0=v0)
        order = np.argsort(evals)
        evals = np.real(evals[order])
        evecs = evecs[:, order]
    except ArpackError as e:
        # Fallback: dense (only safe for small N)
        if N > 5000:
            raise RuntimeError(
                f"ARPACK failed at N={N} (too large for dense fallback): {e}"
            ) from e
        print(f"  WARNING: ARPACK failed at N={N}, using dense fallback")
        L_dense = L.toarray()
        all_evals, all_evecs = np.linalg.eigh(L_dense)
        evals = all_evals[:k_actual]
        evecs = all_evecs[:, :k_actual]

    # Clean: clamp tiny negatives to 0 (numerical noise from Lanczos)
    evals = np.maximum(evals, 0.0)

    # Fiedler value (smallest nonzero eigenvalue)
    nonzero = evals[evals > 1e-10]
    fiedler = float(nonzero[0]) if len(nonzero) > 0 else 0.0

    # Check connectivity: multiple zero eigenvalues = disconnected graph
    n_zero = int(np.sum(evals < 1e-10))
    if n_zero > 1:
        # Graph has multiple connected components — flag but continue
        pass  # n_zero recorded below

    # Heat trace: K(t) = sum exp(-lambda_i * t) for bottom-k eigenvalues.
    # The remaining N-k eigenvalues are all >= lambda_k.  We report ONLY
    # the partial sum from the computed eigenvalues.  The tail bound
    # (N-k)*exp(-lambda_k*t) is tracked separately so the CRN delta
    # is computed from the exact partial sums (no truncation bias).
    heat_trace = {}
    tail_bound = {}
    ht_reliable = {}  # True if tail_bound < 10% of partial sum
    for t in T_HEAT:
        kt_exact = float(np.sum(np.exp(-evals * t)))
        tb = 0.0
        if len(evals) > 0 and evals[-1] > 0:
            tb = float((N - k_actual) * np.exp(-evals[-1] * t))
        heat_trace[f"t={t}"] = kt_exact
        tail_bound[f"t={t}"] = tb
        ht_reliable[f"t={t}"] = (tb < 0.1 * abs(kt_exact)) if kt_exact != 0 else False

    # Spectral dimension: d_S(t) = -2 * d(log K)/d(log t)
    # Numerical derivative from adjacent t values.
    # Only reliable where K(t) is in its power-law regime (not plateau).
    spectral_dim = {}
    t_arr = np.array(T_HEAT)
    k_arr = np.array([heat_trace[f"t={t}"] for t in T_HEAT])
    for i in range(1, len(T_HEAT)):
        t_key_lo = f"t={T_HEAT[i-1]}"
        t_key_hi = f"t={T_HEAT[i]}"
        # Only compute d_S where BOTH heat trace values are reliable
        if not (ht_reliable.get(t_key_lo, False) and ht_reliable.get(t_key_hi, False)):
            continue
        if k_arr[i] > 0 and k_arr[i-1] > 0:
            dlogK = np.log(k_arr[i]) - np.log(k_arr[i-1])
            dlogt = np.log(t_arr[i]) - np.log(t_arr[i-1])
            ds = -2.0 * dlogK / dlogt
            t_mid = np.sqrt(t_arr[i] * t_arr[i-1])
            spectral_dim[f"t={t_mid:.4f}"] = float(ds)

    # Embedding quality (Spearman rho at k=K_EMBED).
    # Reuse eigenvectors from the single eigsh call above.
    # Euclidean distance (not Lorentzian) — Spearman is rank-based,
    # so monotonic transforms don't affect rho.
    embedding_rho = 0.0
    try:
        # Skip zero eigenvalue (index 0), take next K_EMBED
        start = 1
        end = min(start + K_EMBED, len(evals))
        lams = np.maximum(evals[start:end], 0.0)
        vecs = evecs[:, start:end]
        embedding = vecs * np.sqrt(lams)[np.newaxis, :]

        # Random distance pairs (same RNG for flat/curved via CRN)
        idx_i = rng_dist.integers(0, N, size=N_DISTANCE_PAIRS)
        idx_j = rng_dist.integers(0, N, size=N_DISTANCE_PAIRS)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]

        # Euclidean spacetime distances
        d_euclid = np.sqrt(np.sum((pts[idx_i] - pts[idx_j])**2, axis=1))

        # Embedding distances
        d_embed = np.sqrt(np.sum((embedding[idx_i] - embedding[idx_j])**2, axis=1))

        if len(d_euclid) > 10:
            rho_sp, _ = stats.spearmanr(d_embed, d_euclid)
            if np.isnan(rho_sp):
                embedding_rho = np.nan
            else:
                embedding_rho = float(rho_sp)
    except Exception:
        pass

    return {
        "link_count": n_links,
        "mean_degree": mean_degree,
        "fiedler": fiedler,
        "n_components": n_zero,  # >1 means disconnected graph
        "heat_trace": heat_trace,
        "tail_bound": tail_bound,
        "ht_reliable": ht_reliable,
        "spectral_dim": spectral_dim,
        "embedding_rho": embedding_rho,
        "n_eigenvalues": len(evals),
        "lambda_max_computed": float(evals[-1]) if len(evals) > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# CRN paired worker
# ---------------------------------------------------------------------------
def crn_one(seed: int, N: int, T: float, eps: float,
            metric_fn, k_eigsh: int = K_EIGSH) -> dict:
    """One CRN paired sprinkling: flat vs curved link-graph observables."""
    if N > 15000:
        raise ValueError(f"N={N} exceeds safe limit (15000). Dense causal "
                         f"matrix would use {N**2 * 8 / 1e9:.1f} GB.")
    rng = np.random.default_rng(seed)
    pts = sprinkle(N, T, rng)

    # Same distance-pair RNG for both flat and curved (CRN)
    rng_dist_flat = np.random.default_rng(seed + 77777)
    rng_dist_curved = np.random.default_rng(seed + 77777)

    # Flat
    C_flat = causal_flat(pts)
    obs_flat = compute_observables(C_flat, pts, N, k_eigsh, rng_dist_flat)
    del C_flat; gc.collect()

    # Curved
    C_curved = metric_fn(pts, eps)
    obs_curved = compute_observables(C_curved, pts, N, k_eigsh, rng_dist_curved)
    del C_curved; gc.collect()

    # Build result with deltas
    result = {"seed": seed, "N": N, "eps": eps}

    # Scalar observables
    for key in ["link_count", "mean_degree", "fiedler", "embedding_rho"]:
        result[f"{key}_flat"] = obs_flat[key]
        result[f"{key}_curved"] = obs_curved[key]
        result[f"{key}_delta"] = obs_curved[key] - obs_flat[key]

    # Heat trace deltas at each t (exact partial sums only).
    # Tail bounds tracked separately; delta_tail_bound flags truncation bias.
    for t_key in obs_flat["heat_trace"]:
        kf = obs_flat["heat_trace"][t_key]
        kc = obs_curved["heat_trace"].get(t_key, kf)
        result[f"ht_{t_key}_flat"] = kf
        result[f"ht_{t_key}_curved"] = kc
        result[f"ht_{t_key}_delta"] = kc - kf
        tb_f = obs_flat["tail_bound"].get(t_key, 0)
        tb_c = obs_curved["tail_bound"].get(t_key, 0)
        result[f"tb_{t_key}_flat"] = tb_f
        result[f"tb_{t_key}_curved"] = tb_c
        result[f"tb_{t_key}_delta"] = tb_c - tb_f
        # Reliability: both sides must have reliable partial sums
        rel_f = obs_flat.get("ht_reliable", {}).get(t_key, False)
        rel_c = obs_curved.get("ht_reliable", {}).get(t_key, False)
        result[f"ht_{t_key}_reliable"] = rel_f and rel_c

    # Spectral dimension deltas (intersect keys to avoid KeyError)
    common_ds_keys = set(obs_flat["spectral_dim"]) & set(obs_curved["spectral_dim"])
    for t_key in sorted(common_ds_keys):
        dsf = obs_flat["spectral_dim"][t_key]
        dsc = obs_curved["spectral_dim"][t_key]
        result[f"ds_{t_key}_flat"] = dsf
        result[f"ds_{t_key}_curved"] = dsc
        result[f"ds_{t_key}_delta"] = dsc - dsf

    # Connectivity
    result["n_components_flat"] = obs_flat["n_components"]
    result["n_components_curved"] = obs_curved["n_components"]

    return result


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def ckpt_path(metric: str, N: int, eps: float) -> Path:
    return CHECKPOINT_DIR / f"r2_{metric}_N{N}_eps{eps}.json"


def save_ckpt(data: list[dict], metric: str, N: int, eps: float):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    p = ckpt_path(metric, N, eps)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    tmp.replace(p)


def load_ckpt(metric: str, N: int, eps: float) -> list[dict]:
    p = ckpt_path(metric, N, eps)
    if p.exists():
        with open(p) as f:
            return [r for r in json.load(f) if isinstance(r, dict)]
    return []


# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------
def run_experiment(metric: str, N: int, M: int, T: float = 1.0):
    cfg = METRICS[metric]
    metric_fn = METRIC_FNS[metric]
    eps_values = cfg["eps"]

    print(f"\n{'='*60}")
    print(f"ROUTE 2 CRN: {cfg['label']}  N={N:,}  M={M}")
    print(f"{'='*60}")

    ss_metric = np.random.SeedSequence(MASTER_SEED + METRIC_SEED_OFFSETS.get(metric, 999))
    # Per-eps SeedSequence: stable when M changes (prefix-stable by design)
    ss_per_eps = ss_metric.spawn(len(eps_values))

    for eps_idx, eps in enumerate(eps_values):
        existing = load_ckpt(metric, N, eps)

        M_this = 5 if abs(eps) < 1e-12 else M
        # Spawn M_this children from this eps's SeedSequence
        child_seeds = ss_per_eps[eps_idx].spawn(M_this)
        seeds = [int(cs.generate_state(1)[0]) for cs in child_seeds]
        seed_set = set(seeds)

        # Filter existing results to current seed set only (handles M decrease)
        results = [r for r in existing if r["seed"] in seed_set]
        done_seeds = {r["seed"] for r in results}

        remaining = [s for s in seeds if s not in done_seeds]
        print(f"  eps={eps}: {len(results)}/{M_this} done, {len(remaining)} remaining")

        t0 = time.perf_counter()
        for j, seed in enumerate(remaining):
            r = crn_one(seed, N, T, eps, metric_fn)
            results.append(r)

            if (j + 1) % 5 == 0 or j == len(remaining) - 1:
                save_ckpt(results, metric, N, eps)
                elapsed = time.perf_counter() - t0
                rate = (j + 1) / elapsed * 60
                eta = (len(remaining) - j - 1) / rate if rate > 0 else 0
                print(f"    {j+1}/{len(remaining)}  "
                      f"{rate:.1f}/min  ETA {eta:.0f} min")

        # Statistics for this eps (NaN-safe: filter NaN values from embedding_rho)
        # NOTE: p-values here are per-test, uncorrected across eps values.
        # All tests are EXPLORATORY. Final Bonferroni across all tests in report.
        if abs(eps) > 1e-12 and len(results) >= 2:
            for key in ["fiedler_delta", "embedding_rho_delta"]:
                all_vals = [r[key] for r in results if key in r]
                n_nan = sum(1 for v in all_vals if not np.isfinite(v))
                vals = [v for v in all_vals if np.isfinite(v)]
                if n_nan > 0:
                    print(f"    {key}: {n_nan}/{len(all_vals)} NaN dropped"
                          f"{'  WARNING: >10%' if n_nan > 0.1 * len(all_vals) else ''}")
                if len(vals) >= 2:
                    arr = np.array(vals)
                    t_stat, p_t = stats.ttest_1samp(arr, 0.0)
                    d = np.mean(arr) / (np.std(arr, ddof=1) + 1e-20)
                    # Wilcoxon signed-rank (nonparametric, no normality assumption)
                    try:
                        w_stat, p_w = stats.wilcoxon(arr)
                    except ValueError:
                        p_w = float('nan')  # all zeros
                    print(f"    {key}: d={d:+.3f} p_t={p_t:.2e} p_w={p_w:.2e}  "
                          f"(n={len(vals)}, raw, uncorrected)")

            # Best heat trace t (only from RELIABLE t-values)
            best_d, best_t, best_n = 0, "", 0
            for t_key in [f"t={t}" for t in T_HEAT]:
                k = f"ht_{t_key}_delta"
                rel_k = f"ht_{t_key}_reliable"
                vals = [r[k] for r in results
                        if k in r and r.get(rel_k, False)]
                if len(vals) >= 2:
                    arr = np.array(vals)
                    d = abs(np.mean(arr) / (np.std(arr, ddof=1) + 1e-20))
                    if d > best_d:
                        best_d, best_t, best_n = d, t_key, len(vals)
            if best_t:
                k = f"ht_{best_t}_delta"
                rel_k = f"ht_{best_t}_reliable"
                vals = [r[k] for r in results
                        if k in r and r.get(rel_k, False)]
                arr = np.array(vals)
                t_stat, p_raw = stats.ttest_1samp(arr, 0.0)
                d = np.mean(arr) / (np.std(arr, ddof=1) + 1e-20)
                # Bonferroni over ALL planned t-values (not data-dependent count)
                p_adj_t = min(1.0, p_raw * len(T_HEAT))
                try:
                    _, p_w_raw = stats.wilcoxon(arr)
                    p_adj_w = min(1.0, p_w_raw * len(T_HEAT))
                except ValueError:
                    p_adj_w = float('nan')
                print(f"    heat_trace({best_t}): d={d:+.3f} "
                      f"p_t_adj={p_adj_t:.2e} p_w_adj={p_adj_w:.2e}  "
                      f"[best of {len(T_HEAT)}, Bonferroni, n={len(vals)}]")

    # Null validation for conformal
    if "conformal" in metric.lower():
        all_results = []
        for eps in eps_values:
            all_results.extend(load_ckpt(metric, N, eps))
        null_keys = ["fiedler_delta", "link_count_delta", "mean_degree_delta"]
        max_delta = 0.0
        for nk in null_keys:
            for r in all_results:
                max_delta = max(max_delta, abs(r.get(nk, 0)))
        status = "PASS" if max_delta < 1e-10 else "FAIL"
        print(f"  NULL TEST: max|delta| over {null_keys} = {max_delta:.2e}  ({status})")


# ---------------------------------------------------------------------------
# Calibrate
# ---------------------------------------------------------------------------
def calibrate(N_values=None):
    if N_values is None:
        N_values = [500, 1000, 2000, 5000, 10000]

    print("=" * 60)
    print("ROUTE 2 CALIBRATION")
    print("=" * 60)

    for N in N_values:
        rng = np.random.default_rng(42)
        pts = sprinkle(N, 1.0, rng)

        t0 = time.perf_counter()
        C = causal_flat(pts)
        t_causal = time.perf_counter() - t0

        t1 = time.perf_counter()
        A = build_link_graph(C)
        t_link = time.perf_counter() - t1

        n_links = int(A.sum()) // 2
        deg = float(A.sum() / N)

        t2 = time.perf_counter()
        L = build_laplacian(A)
        k = min(K_EIGSH, N - 2)
        v0 = np.ones(N, dtype=np.float64) / np.sqrt(N)
        evals = eigsh(L.astype(np.float64), k=k, which='SA',
                      v0=v0, return_eigenvectors=False)
        t_eigsh = time.perf_counter() - t2

        t_total = t_causal + t_link + t_eigsh
        # CRN = 2x (flat + curved)
        t_crn = t_total * 2

        print(f"  N={N:>6,}: causal={t_causal:.1f}s  link={t_link:.1f}s  "
              f"eigsh(k={k})={t_eigsh:.1f}s  total={t_total:.1f}s  "
              f"CRN={t_crn:.1f}s  links={n_links:,}  deg={deg:.1f}")

        del C, A, L; gc.collect()


# ---------------------------------------------------------------------------
# Effect size diagnostic (Rule 13: check before running full pipeline)
# ---------------------------------------------------------------------------
def diagnose(N: int = 5000, T: float = 1.0):
    """One CRN pair per metric at max eps. Reports % causal pairs changed,
    % links changed, and all observable deltas."""
    print("=" * 60)
    print(f"EFFECT SIZE DIAGNOSTIC  N={N:,}")
    print("=" * 60)

    test_cases = [
        ("ppwave_quad",   causal_ppwave_quad,   5.0),
        ("ppwave_cross",  causal_ppwave_cross,  5.0),
        ("schwarzschild", causal_schwarzschild,  0.02),
        ("flrw",          causal_flrw,           2.0),
    ]

    rng = np.random.default_rng(99999)
    pts = sprinkle(N, T, rng)
    C_flat = causal_flat(pts)
    n_causal_flat = int(C_flat.sum())

    A_flat = build_link_graph(C_flat)
    n_links_flat = int(A_flat.sum()) // 2
    del A_flat; gc.collect()

    for name, fn, eps in test_cases:
        C_curved = fn(pts, eps)
        n_causal_curved = int(C_curved.sum())
        n_changed_causal = int(np.sum(C_flat != C_curved))
        pct_causal = 100.0 * n_changed_causal / max(N * N, 1)

        A_curved = build_link_graph(C_curved)
        n_links_curved = int(A_curved.sum()) // 2
        link_delta = n_links_curved - n_links_flat
        pct_links = 100.0 * abs(link_delta) / max(n_links_flat, 1)
        del A_curved; gc.collect()

        # Quick observables (single pair, no stats)
        rng_d = np.random.default_rng(99999 + 77777)
        obs_f = compute_observables(C_flat, pts, N, K_EIGSH, np.random.default_rng(99999 + 77777))
        obs_c = compute_observables(C_curved, pts, N, K_EIGSH, rng_d)

        fiedler_d = obs_c["fiedler"] - obs_f["fiedler"]
        rho_d = obs_c["embedding_rho"] - obs_f["embedding_rho"]

        # Best reliable heat trace delta
        best_ht_d, best_ht_t = 0.0, ""
        for tk in obs_f["heat_trace"]:
            rel_f = obs_f.get("ht_reliable", {}).get(tk, False)
            rel_c = obs_c.get("ht_reliable", {}).get(tk, False)
            if rel_f and rel_c:
                d = obs_c["heat_trace"][tk] - obs_f["heat_trace"][tk]
                if abs(d) > abs(best_ht_d):
                    best_ht_d, best_ht_t = d, tk

        n_reliable = sum(1 for tk in T_HEAT
                         if obs_f.get("ht_reliable", {}).get(f"t={tk}", False)
                         and obs_c.get("ht_reliable", {}).get(f"t={tk}", False))

        print(f"\n  {name} (eps={eps}):")
        print(f"    causal pairs: flat={n_causal_flat:,}  curved={n_causal_curved:,}  "
              f"changed={n_changed_causal:,} ({pct_causal:.2f}%)")
        print(f"    links: flat={n_links_flat:,}  curved={n_links_curved:,}  "
              f"delta={link_delta:+,} ({pct_links:.2f}%)")
        print(f"    fiedler_delta={fiedler_d:+.6f}  embedding_rho_delta={rho_d:+.6f}")
        print(f"    heat_trace reliable: {n_reliable}/9 t-values")
        if best_ht_t:
            print(f"    best reliable ht delta: {best_ht_d:+.4f} at {best_ht_t}")
        else:
            print(f"    NO reliable heat trace t-values!")

        del C_curved; gc.collect()

    del C_flat; gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="FND-1 Route 2 CRN")
    parser.add_argument("--metric", type=str, default=None)
    parser.add_argument("--N", type=int, default=5000)
    parser.add_argument("--M", type=int, default=40)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    args = parser.parse_args()

    if args.calibrate:
        calibrate()
        return

    if args.diagnose:
        diagnose(args.N, args.T)
        return

    if args.all:
        metrics = list(METRICS.keys())
    elif args.metric:
        metrics = [args.metric]
    else:
        metrics = ["ppwave_quad"]

    for metric in metrics:
        run_experiment(metric, args.N, args.M, args.T)

    print("\nDone.")


if __name__ == "__main__":
    main()
