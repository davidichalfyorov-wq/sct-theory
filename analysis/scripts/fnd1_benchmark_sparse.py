"""
FND-1 Sparse Pipeline Benchmark
================================
Measures memory, time, and sparsity scaling for the GTA [H,M] commutator
pipeline at N = 1k..20k. Extrapolates to predict N=50k, 100k, 200k
feasibility on different GPU VRAM sizes.

Key measurements per N:
  - nnz(C), nnz(L), nnz(commutator)
  - Sparse memory (bytes) for each matrix
  - Time for each pipeline step
  - Dense eigvalsh vs sparse eigsh accuracy comparison
  - Fraction of Frobenius norm captured by top-k eigenvalues

Output: JSON with all measurements + extrapolation predictions.

Run (single-threaded, no multiprocessing):
    python analysis/scripts/fnd1_benchmark_sparse.py
"""
from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

import gc
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
T_DIAMOND = 1.0
EPS_BENCHMARK = 5.0       # quadrupole strength (same as EXP-3, EXP-14, EXP-20)
K_EIGSH = 100             # number of eigenvalues for sparse eigsh
SEED = 99999
RESULTS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "speculative" / "numerics" / "ensemble_results"
)

# ---------------------------------------------------------------------------
# Self-contained pipeline functions (no imports from running experiments)
# ---------------------------------------------------------------------------

def sprinkle_4d_flat(N: int, T: float, rng) -> np.ndarray:
    """Sprinkle N points into 4D Minkowski causal diamond via rejection.

    Diamond: {(t,x,y,z) : |t| + sqrt(x^2+y^2+z^2) < T/2}
    Acceptance rate ~13% in 4D.
    """
    pts = np.empty((N, 4))
    count = 0
    half = T / 2.0
    while count < N:
        batch = max(N - count, 1000) * 10
        candidates = rng.uniform(-half, half, size=(batch, 4))
        r = np.sqrt(
            candidates[:, 1] ** 2
            + candidates[:, 2] ** 2
            + candidates[:, 3] ** 2
        )
        inside = np.abs(candidates[:, 0]) + r < half
        valid = candidates[inside]
        n_take = min(len(valid), N - count)
        pts[count : count + n_take] = valid[:n_take]
        count += n_take
    order = np.argsort(pts[:, 0])
    return pts[order]


def quadrupole_profile(x, y):
    """Pure quadrupole: f = x^2 - y^2. Traceless, Weyl-type."""
    return x ** 2 - y ** 2


def build_causal_matrix_dense(pts: np.ndarray, eps: float) -> np.ndarray:
    """Build dense causal matrix C[i,j] = 1 if i prec j.

    Uses pp-wave metric: ds^2 approx Minkowski + eps*f*(dt+dz)^2/2
    with quadrupole profile f = x^2 - y^2.
    """
    t = pts[:, 0]
    x = pts[:, 1]
    y = pts[:, 2]
    z = pts[:, 3]

    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dz = z[np.newaxis, :] - z[:, np.newaxis]
    dr2 = dx ** 2 + dy ** 2 + dz ** 2

    if abs(eps) > 1e-12:
        xm = (x[np.newaxis, :] + x[:, np.newaxis]) / 2.0
        ym = (y[np.newaxis, :] + y[:, np.newaxis]) / 2.0
        f_mid = quadrupole_profile(xm, ym)
        mink = dt ** 2 - dr2
        corr = eps * f_mid * (dt + dz) ** 2 / 2.0
        C = ((mink > corr) & (dt > 0)).astype(np.float64)
        del xm, ym, f_mid, mink, corr
    else:
        C = ((dt ** 2 > dr2) & (dt > 0)).astype(np.float64)

    del dt, dx, dy, dz, dr2
    return C


def build_bd_operator_and_layers(
    C: np.ndarray, N: int, T: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build 4D BD d'Alembertian L (dense) from causal matrix C.

    BD coefficients for d=4: {4, -36, 64, -32} * sqrt(rho).
    Returns (L, n_matrix) where n_matrix = C @ C (interval cardinalities).
    """
    V = np.pi * T ** 4 / 24.0
    rho = N / V
    scale = np.sqrt(rho)

    C_sp = sp.csr_matrix(C)
    n_matrix = (C_sp @ C_sp).toarray()

    past = C.T
    n_past = n_matrix.T
    n_int = np.rint(n_past).astype(np.int64)
    causal_mask = past > 0.5

    L = np.zeros((N, N), dtype=np.float64)
    L[causal_mask & (n_int == 0)] = 4.0 * scale
    L[causal_mask & (n_int == 1)] = -36.0 * scale
    L[causal_mask & (n_int == 2)] = 64.0 * scale
    L[causal_mask & (n_int == 3)] = -32.0 * scale

    return L, n_matrix


def compute_commutator_sparse(L: np.ndarray) -> sp.csr_matrix:
    """Compute GTA commutator [H,M] = (L^T L - L L^T) / 2 as sparse.

    Input L is dense, converted to sparse for the matrix products.
    Result kept sparse. Symmetrized to fix floating-point asymmetry.
    """
    L_sp = sp.csr_matrix(L)
    comm = (L_sp.T @ L_sp - L_sp @ L_sp.T) / 2.0
    comm = (comm + comm.T) / 2.0  # enforce exact symmetry for eigsh
    return comm.tocsr()


def sparse_memory_bytes(M) -> int:
    """Memory usage of a sparse CSR matrix in bytes."""
    if sp.issparse(M):
        csr = M.tocsr()
        return csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes
    return int(M.nbytes)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_single_N(
    N: int,
    eps: float = EPS_BENCHMARK,
    T: float = T_DIAMOND,
    k_eig: int = K_EIGSH,
    seed: int = SEED,
) -> dict:
    """Run full benchmark for a single N. Returns dict with all measurements."""
    result = {"N": N, "eps": eps}
    rng = np.random.default_rng(seed)

    # ── 1. Sprinkle ──────────────────────────────────────────────────────
    t0 = time.perf_counter()
    pts = sprinkle_4d_flat(N, T, rng)
    result["time_sprinkle_s"] = round(time.perf_counter() - t0, 3)

    # ── 2. Dense causal matrix ───────────────────────────────────────────
    t0 = time.perf_counter()
    C = build_causal_matrix_dense(pts, eps)
    result["time_causal_s"] = round(time.perf_counter() - t0, 3)
    result["dense_C_MB"] = round(C.nbytes / 1e6, 1)

    C_sp = sp.csr_matrix(C)
    result["nnz_C"] = int(C_sp.nnz)
    result["density_C"] = round(C_sp.nnz / (N * N), 4)
    result["sparse_C_MB"] = round(sparse_memory_bytes(C_sp) / 1e6, 1)

    # ── 3. BD operator L ─────────────────────────────────────────────────
    t0 = time.perf_counter()
    L, n_matrix = build_bd_operator_and_layers(C, N, T)
    result["time_build_L_s"] = round(time.perf_counter() - t0, 3)

    del C  # free dense causal matrix
    gc.collect()

    L_sp = sp.csr_matrix(L)
    result["nnz_L"] = int(L_sp.nnz)
    result["density_L"] = round(L_sp.nnz / (N * N), 6)
    result["sparse_L_MB"] = round(sparse_memory_bytes(L_sp) / 1e6, 1)
    result["L_nnz_per_row"] = round(L_sp.nnz / N, 1)

    # Layer distribution (how many nonzeros per layer)
    V = np.pi * T ** 4 / 24.0
    rho = N / V
    scale = np.sqrt(rho)
    past = L_sp.copy()
    past.data = np.ones_like(past.data)  # just count nonzeros
    n_layers = {}
    for layer, coeff in enumerate([4.0, -36.0, 64.0, -32.0]):
        threshold = abs(coeff * scale)
        mask = np.abs(np.abs(L_sp.data) - threshold) < 1e-6 * threshold
        n_layers[f"layer_{layer}"] = int(np.sum(mask))
    result["layer_counts"] = n_layers
    del past

    # ── 4. Commutator (sparse) ───────────────────────────────────────────
    t0 = time.perf_counter()
    comm_sp = compute_commutator_sparse(L)
    result["time_commutator_s"] = round(time.perf_counter() - t0, 3)

    result["nnz_comm"] = int(comm_sp.nnz)
    result["density_comm"] = round(comm_sp.nnz / (N * N), 6)
    result["sparse_comm_MB"] = round(sparse_memory_bytes(comm_sp) / 1e6, 1)
    result["comm_nnz_per_row"] = round(comm_sp.nnz / N, 1)

    del L, n_matrix
    gc.collect()

    # ── 5. Sparse eigsh (top-k) ──────────────────────────────────────────
    k_actual = min(k_eig, N - 2)
    t0 = time.perf_counter()
    try:
        evals_sp = eigsh(
            comm_sp, k=k_actual, which="LM", return_eigenvectors=False
        )
        evals_sp = np.sort(evals_sp)[::-1]
        result["time_eigsh_s"] = round(time.perf_counter() - t0, 3)
        result["eigsh_ok"] = True
        result["eigsh_k"] = k_actual
        result["eigsh_top5"] = [round(float(v), 8) for v in evals_sp[:5]]
    except Exception as e:
        result["time_eigsh_s"] = round(time.perf_counter() - t0, 3)
        result["eigsh_ok"] = False
        result["eigsh_error"] = str(e)

    # ── 6. Dense eigvalsh (for validation) ───────────────────────────────
    max_dense_N = 15000
    if N <= max_dense_N:
        comm_dense = comm_sp.toarray()
        t0 = time.perf_counter()
        evals_dense = np.linalg.eigvalsh(comm_dense)
        result["time_eigvalsh_s"] = round(time.perf_counter() - t0, 3)

        # Compare top eigenvalues (by magnitude)
        evals_dense_abs_sorted = np.sort(np.abs(evals_dense))[::-1]
        if result.get("eigsh_ok"):
            evals_sp_abs_sorted = np.sort(np.abs(evals_sp))[::-1]
            result["top5_match"] = bool(
                np.allclose(
                    evals_sp_abs_sorted[:5],
                    evals_dense_abs_sorted[:5],
                    rtol=1e-3,
                )
            )

        # Full spectrum stats
        a_dense = np.abs(evals_dense)
        s_dense = float(np.sum(a_dense))
        result["entropy_full"] = round(
            float(-np.sum((a_dense / s_dense) * np.log(a_dense / s_dense + 1e-300)))
            if s_dense > 0
            else 0.0,
            6,
        )
        result["frobenius_full"] = round(
            float(np.sqrt(np.sum(evals_dense ** 2))), 6
        )

        # Frobenius fraction captured by top-k
        if result.get("eigsh_ok"):
            frob_topk = float(np.sum(evals_sp ** 2))
            frob_full = float(np.sum(evals_dense ** 2))
            result["frob_fraction_topk"] = (
                round(frob_topk / frob_full, 6) if frob_full > 0 else 0.0
            )

        del comm_dense
    else:
        result["time_eigvalsh_s"] = None
        result["top5_match"] = None

    del comm_sp
    gc.collect()

    return result


# ---------------------------------------------------------------------------
# Extrapolation
# ---------------------------------------------------------------------------

def extrapolate(
    results: list[dict],
    target_Ns: list[int] | None = None,
) -> dict:
    """Fit power laws nnz ~ N^alpha and extrapolate to large N."""
    if target_Ns is None:
        target_Ns = [50_000, 100_000, 200_000]

    Ns = np.array([r["N"] for r in results], dtype=float)
    log_Ns = np.log(Ns)

    def fit_and_predict(values, target_N):
        log_vals = np.log(np.array(values, dtype=float))
        coeffs = np.polyfit(log_Ns, log_vals, 1)
        alpha = coeffs[0]
        predicted = float(np.exp(np.polyval(coeffs, np.log(target_N))))
        return alpha, predicted

    predictions = {}

    for target_N in target_Ns:
        pred = {"N": target_N}

        # nnz(L) scaling
        alpha_L, nnz_L = fit_and_predict(
            [r["nnz_L"] for r in results], target_N
        )
        pred["nnz_L"] = int(nnz_L)
        pred["alpha_L"] = round(alpha_L, 3)
        pred["sparse_L_GB"] = round(nnz_L * 12 / 1e9, 2)

        # nnz(comm) scaling
        alpha_comm, nnz_comm = fit_and_predict(
            [r["nnz_comm"] for r in results], target_N
        )
        pred["nnz_comm"] = int(nnz_comm)
        pred["alpha_comm"] = round(alpha_comm, 3)
        pred["sparse_comm_GB"] = round(nnz_comm * 12 / 1e9, 2)

        # eigsh time scaling
        ok_results = [r for r in results if r.get("eigsh_ok")]
        if len(ok_results) >= 2:
            alpha_t, time_pred = fit_and_predict(
                [r["time_eigsh_s"] for r in ok_results], target_N
            )
            pred["time_eigsh_s"] = round(time_pred, 1)
            pred["alpha_time"] = round(alpha_t, 3)

        # Total VRAM: L + comm + eigsh workspace (~1.5x comm for Lanczos vectors)
        pred["vram_GB"] = round(
            pred["sparse_L_GB"] + pred["sparse_comm_GB"] * 2.5, 2
        )
        pred["fits_24GB"] = pred["vram_GB"] < 22
        pred["fits_48GB"] = pred["vram_GB"] < 44
        pred["fits_80GB"] = pred["vram_GB"] < 76

        # Wall time estimate
        if "time_eigsh_s" in pred:
            # Total per sprinkling ~ 2x eigsh (causal matrix + layers + commutator)
            per_sprinkling_min = pred["time_eigsh_s"] * 2 / 60
            pred["min_per_sprinkling"] = round(per_sprinkling_min, 1)
            pred["hours_400_sprinklings"] = round(per_sprinkling_min * 400 / 60, 1)

        predictions[str(target_N)] = pred

    return predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("FND-1 SPARSE PIPELINE BENCHMARK")
    print("=" * 70)
    print(f"Quadrupole eps={EPS_BENCHMARK}, T={T_DIAMOND}, k_eigsh={K_EIGSH}")
    print(f"Purpose: measure nnz scaling to predict N=50k/100k/200k feasibility")
    print()

    # Determine safe N range based on available memory
    N_TEST = [1000, 2000, 5000, 10000]
    try:
        import psutil

        avail_gb = psutil.virtual_memory().available / 1e9
        print(f"Available RAM: {avail_gb:.1f} GB")
        if avail_gb > 35:
            N_TEST.append(15000)
        if avail_gb > 45:
            N_TEST.append(20000)
    except ImportError:
        print("psutil not available, sticking with N <= 10000")

    print(f"Test N: {N_TEST}")
    print()

    results = []
    t_total = time.perf_counter()

    for N in N_TEST:
        print(f"{'─' * 70}")
        print(f"N = {N:,}")
        print(f"{'─' * 70}")
        try:
            r = benchmark_single_N(N)
            results.append(r)

            print(f"  Sprinkle:       {r['time_sprinkle_s']:>7.2f}s")
            print(
                f"  Causal matrix:  {r['time_causal_s']:>7.2f}s   "
                f"density={r['density_C']:.3f}  "
                f"nnz={r['nnz_C']:>12,}  "
                f"dense={r['dense_C_MB']:.0f} MB  "
                f"sparse={r['sparse_C_MB']:.0f} MB"
            )
            print(
                f"  BD operator L:  {r['time_build_L_s']:>7.2f}s   "
                f"density={r['density_L']:.6f}  "
                f"nnz={r['nnz_L']:>12,}  "
                f"nnz/row={r['L_nnz_per_row']:.0f}  "
                f"sparse={r['sparse_L_MB']:.1f} MB"
            )
            lc = r["layer_counts"]
            print(
                f"    Layers: L0={lc['layer_0']:,}  "
                f"L1={lc['layer_1']:,}  "
                f"L2={lc['layer_2']:,}  "
                f"L3={lc['layer_3']:,}"
            )
            print(
                f"  Commutator:     {r['time_commutator_s']:>7.2f}s   "
                f"density={r['density_comm']:.6f}  "
                f"nnz={r['nnz_comm']:>12,}  "
                f"nnz/row={r['comm_nnz_per_row']:.0f}  "
                f"sparse={r['sparse_comm_MB']:.1f} MB"
            )
            if r.get("eigsh_ok"):
                print(
                    f"  eigsh(k={r['eigsh_k']}):  {r['time_eigsh_s']:>7.2f}s   "
                    f"top={r['eigsh_top5'][0]:.4e}"
                )
            else:
                print(f"  eigsh FAILED: {r.get('eigsh_error', '?')}")
            if r.get("time_eigvalsh_s") is not None:
                print(
                    f"  eigvalsh(full): {r['time_eigvalsh_s']:>7.2f}s   "
                    f"match={r.get('top5_match')}"
                )
            if r.get("frob_fraction_topk") is not None:
                print(
                    f"  Frobenius top-{r.get('eigsh_k', K_EIGSH)}: "
                    f"{r['frob_fraction_topk'] * 100:.1f}% of full spectrum"
                )
            print()

        except Exception:
            print(f"  FAILED:")
            traceback.print_exc()
            print()

    wall = time.perf_counter() - t_total

    if len(results) < 2:
        print("Not enough successful benchmarks for extrapolation.")
        return

    # ── Extrapolation ────────────────────────────────────────────────────
    print("=" * 70)
    print("EXTRAPOLATION TO LARGE N")
    print("=" * 70)

    predictions = extrapolate(results)

    for target_str, pred in sorted(predictions.items(), key=lambda x: int(x[0])):
        N_t = int(target_str)
        print(f"\n  N = {N_t:>7,}:")
        print(
            f"    nnz(L):    {pred['nnz_L']:>15,}  "
            f"(alpha={pred['alpha_L']:.2f})  "
            f"sparse={pred['sparse_L_GB']:.2f} GB"
        )
        print(
            f"    nnz(comm): {pred['nnz_comm']:>15,}  "
            f"(alpha={pred['alpha_comm']:.2f})  "
            f"sparse={pred['sparse_comm_GB']:.2f} GB"
        )
        print(f"    VRAM estimate (L + 2.5*comm): {pred['vram_GB']:.1f} GB")
        print(
            f"    Fits 24 GB (3090 Ti):  {'YES' if pred['fits_24GB'] else 'NO'}"
        )
        print(
            f"    Fits 48 GB (PRO 6000): {'YES' if pred['fits_48GB'] else 'NO'}"
        )
        print(
            f"    Fits 80 GB (A100):     {'YES' if pred['fits_80GB'] else 'NO'}"
        )
        if "min_per_sprinkling" in pred:
            print(
                f"    ~{pred['min_per_sprinkling']:.0f} min/sprinkling  "
                f"-> {pred['hours_400_sprinklings']:.0f}h for 400 sprinklings"
            )

    # ── Save ─────────────────────────────────────────────────────────────
    output = {
        "parameters": {
            "eps": EPS_BENCHMARK,
            "T": T_DIAMOND,
            "k_eigsh": K_EIGSH,
            "seed": SEED,
            "N_tested": N_TEST,
        },
        "benchmarks": results,
        "predictions": predictions,
        "wall_time_s": round(wall, 1),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "benchmark_sparse.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path.name}")

    # ── Final verdict ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    for label, N_str in [("N=50k", "50000"), ("N=100k", "100000"), ("N=200k", "200000")]:
        p = predictions.get(N_str, {})
        if p.get("fits_24GB"):
            gpu = "24 GB (3090 Ti, RTX 4090) -> Ray + home GPUs"
        elif p.get("fits_48GB"):
            gpu = "48 GB (PRO 6000, A6000) -> GCP credits or Thunder Compute"
        elif p.get("fits_80GB"):
            gpu = "80 GB (A100) -> Vast.ai or Spheron"
        else:
            gpu = ">80 GB -> needs distributed or reduce experiment size"
        hrs = p.get("hours_400_sprinklings", "?")
        print(f"  {label}: VRAM={p.get('vram_GB', '?'):.1f} GB  -> {gpu}")
        print(f"         ~{hrs}h on single GPU for full experiment")

    print(f"\nBenchmark wall time: {wall:.0f}s ({wall / 60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
