"""
FND-1 Route 1: Ensemble Spectral Observables — Gate 0 & Gate 1 Runner.

Implements the ensemble benchmark for causal-set-derived operators in d=2
flat Minkowski diamond. Computes heat traces, UV exponents, null model
comparisons, and gate verdicts for three operator families (A, B, C).

Reference: speculative/FND1_ENSEMBLE_SPEC.md
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy.stats import linregress

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ZERO_THRESHOLD = 1.0e-10      # eigenvalues below this are treated as zero
N_BOOTSTRAP = 1000             # bootstrap resamples for 95% CI
N_T_GRID = 200                 # log-spaced heat-trace grid points
BASE_SEED = 42                 # reproducible RNG seed

P_TARGETS = {"A": -2.0, "B": -1.0, "C": -1.0}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FamilyResult:
    """Results for one operator family."""
    name: str
    p_target: float
    p_ens: float = 0.0
    p_single: float = 0.0
    p_null: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    relative_error: float = 0.0
    uv_decades: float = 0.0
    mean_R_K_uv: float = 0.0       # mean relative variance in UV window
    gate0_verdict: str = ""
    gate1_verdict: str = ""
    n_eff_mean: float = 0.0        # mean number of non-zero eigenvalues
    lambda_max: float = 0.0
    lambda_min: float = 0.0
    frob_norm_mean: float = 0.0


@dataclass
class EnsembleResults:
    """Full results for Gate 0 + Gate 1."""
    N: int = 0
    M: int = 0
    d: int = 2
    T: float = 1.0
    V: float = 0.5
    rho: float = 0.0
    seed: int = BASE_SEED
    families: dict = field(default_factory=dict)
    wall_time_sec: float = 0.0


# ---------------------------------------------------------------------------
# Causal set generation
# ---------------------------------------------------------------------------

def sprinkle_diamond(N: int, T: float, rng: np.random.Generator):
    """
    Sprinkle N points uniformly into a d=2 Minkowski causal diamond.

    The diamond A(p,q) with p=(-T/2, 0), q=(T/2, 0) in null coordinates
    (u, v) = (t+x, t-x) is the square [-T/2, T/2] x [-T/2, T/2].
    Uniform sampling in null coordinates gives uniform sampling in
    spacetime volume (Jacobian dt dx = (1/2) du dv, constant).

    Returns
    -------
    points : (N, 2) array of (t, x) coordinates, sorted by t (natural labeling)
    C : (N, N) strictly upper-triangular causal matrix
    """
    u = rng.uniform(-T / 2, T / 2, size=N)
    v = rng.uniform(-T / 2, T / 2, size=N)
    t = (u + v) / 2
    x = (u - v) / 2

    # Natural labeling: sort by time coordinate
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    points = np.column_stack([t, x])

    # Causal matrix: x_i ≺ x_j iff t_j - t_i > |x_j - x_i|
    # In natural labeling (sorted by t), C is strictly upper-triangular.
    dt = t[np.newaxis, :] - t[:, np.newaxis]  # dt[i,j] = t_j - t_i
    dx = np.abs(x[np.newaxis, :] - x[:, np.newaxis])  # |x_j - x_i|
    C = ((dt > dx) & (dt > 0)).astype(np.float64)

    return points, C


def compute_interval_cardinalities(C: np.ndarray) -> np.ndarray:
    """
    Compute interval cardinality matrix n[j,i] = number of elements
    strictly between x_j and x_i, for all causal pairs x_j ≺ x_i.

    n = C @ C, where (C @ C)[j,i] = sum_k C[j,k]*C[k,i] counts the
    elements z with x_j ≺ z ≺ x_i.
    Uses sparse matmul (2.5x faster, C is typically 1-25% fill).
    """
    import scipy.sparse as sp
    C_sp = sp.csr_matrix(C)
    return (C_sp @ C_sp).toarray()


# ---------------------------------------------------------------------------
# BD d'Alembertian construction
# ---------------------------------------------------------------------------

def build_bd_L(C: np.ndarray, n_matrix: np.ndarray, rho: float) -> np.ndarray:
    """
    Build the strictly lower-triangular matrix L from the BD d'Alembertian.

    B_rho = -2*rho*I + L, where:
      B[i,j] = 2*rho * {+2 if n=0, -4 if n=1, +2 if n=2, 0 otherwise}
    for causal pairs x_j ≺ x_i (j < i, so B[i,j] is lower-triangular).

    L = B + 2*rho*I is strictly lower-triangular (diagonal of B is -2*rho).

    Parameters
    ----------
    C : (N, N) upper-triangular causal matrix. C[j,i]=1 means x_j ≺ x_i.
    n_matrix : (N, N) interval cardinality matrix from C @ C.
    rho : sprinkling density N/V.

    Returns
    -------
    L : (N, N) strictly lower-triangular matrix.
    """
    N = C.shape[0]

    # C[j,i] = 1 for x_j ≺ x_i (upper triangle). We need the past
    # relation in the lower triangle: past[i,j] = C[j,i] for j < i.
    past = C.T  # past[i,j] = 1 iff x_j ≺ x_i, lower-triangular

    # Interval cardinalities for past pairs: n_past[i,j] = n_matrix[j,i]
    n_past = n_matrix.T

    # Build L: strictly lower-triangular
    # For each (i,j) with j < i and past[i,j]=1:
    #   if n_past[i,j] == 0: L[i,j] = 2*rho * 2 = 4*rho
    #   if n_past[i,j] == 1: L[i,j] = 2*rho * (-4) = -8*rho
    #   if n_past[i,j] == 2: L[i,j] = 2*rho * 2 = 4*rho
    #   else: 0
    L = np.zeros((N, N), dtype=np.float64)

    # Masks for each layer (only in past region)
    causal_mask = past > 0.5  # boolean mask for causal past pairs
    n_int = np.rint(n_past).astype(np.int64)

    mask_n0 = causal_mask & (n_int == 0)
    mask_n1 = causal_mask & (n_int == 1)
    mask_n2 = causal_mask & (n_int == 2)

    L[mask_n0] = 4.0 * rho   # 2*rho * (+2)
    L[mask_n1] = -8.0 * rho  # 2*rho * (-4)
    L[mask_n2] = 4.0 * rho   # 2*rho * (+2)

    return L


# ---------------------------------------------------------------------------
# Operator families — eigenvalue extraction
# ---------------------------------------------------------------------------

def _filter_eigenvalues(eigenvalues: np.ndarray) -> np.ndarray:
    """
    Take absolute values and filter out zero modes (|lambda| < ZERO_THRESHOLD).
    Returns sorted positive eigenvalues.
    """
    abs_eig = np.abs(eigenvalues)
    nonzero = abs_eig[abs_eig > ZERO_THRESHOLD]
    return np.sort(nonzero)


def compute_family_A_eigenvalues(L: np.ndarray) -> np.ndarray:
    """
    Family A: A = i(L - L^T)/2 (anti-Hermitian part of BD).
    A is Hermitian with eigenvalues in +-pairs.
    P_A eigenvalues = |eigenvalues of A|.
    """
    M_antisym = L - L.T  # real antisymmetric
    # A = i*M/2 is complex Hermitian
    A = 1j * M_antisym / 2.0
    eigenvalues = np.linalg.eigvalsh(A)  # eigvalsh: 2.8x faster (no eigvecs)
    return _filter_eigenvalues(eigenvalues)


def compute_family_B_eigenvalues(L: np.ndarray) -> np.ndarray:
    """
    Family B: B_sym = (L + L^T)/2 (symmetrized BD, shifted).
    B_sym is real symmetric, traceless.
    P_B eigenvalues = |eigenvalues of B_sym|.
    """
    B_sym = (L + L.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(B_sym)  # eigvalsh: 2.8x faster
    return _filter_eigenvalues(eigenvalues)


def compute_family_C_eigenvalues(C: np.ndarray) -> np.ndarray:
    """
    Family C: iDelta = (i/2)(C - C^T) (SJ-inverse).
    Pseudo-inverse eigenvalues: 1/mu_k for nonzero mu_k.
    P_C eigenvalues = |1/mu_k|.
    """
    Delta_antisym = C - C.T  # real antisymmetric
    iDelta = 1j * Delta_antisym / 2.0  # complex Hermitian
    eigenvalues = np.linalg.eigvalsh(iDelta)  # eigvalsh: 2.8x faster

    # Pseudo-inverse: 1/mu_k for nonzero mu_k
    abs_eig = np.abs(eigenvalues)
    nonzero_mask = abs_eig > ZERO_THRESHOLD
    if not np.any(nonzero_mask):
        return np.array([])
    inv_eig = 1.0 / abs_eig[nonzero_mask]
    return np.sort(inv_eig)


# ---------------------------------------------------------------------------
# Heat trace computation
# ---------------------------------------------------------------------------

def compute_heat_trace(eigenvalues: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """
    Compute K(t) = sum_k exp(-t * lambda_k) for positive eigenvalues.

    Parameters
    ----------
    eigenvalues : 1D array of positive eigenvalues (zero modes excluded).
    t_grid : 1D array of t values.

    Returns
    -------
    K : 1D array of heat trace values, same length as t_grid.
    """
    if len(eigenvalues) == 0:
        return np.zeros_like(t_grid)
    # Vectorized: K[i] = sum_k exp(-t[i] * lambda_k)
    return np.sum(np.exp(-t_grid[:, np.newaxis] * eigenvalues[np.newaxis, :]),
                  axis=1)


def determine_uv_window(all_eigenvalues: list[np.ndarray],
                        n_t: int = N_T_GRID):
    """
    Determine the UV fitting window and t-grid from the ensemble of eigenvalues.

    Parameters
    ----------
    all_eigenvalues : list of M arrays of positive eigenvalues.
    n_t : number of log-spaced grid points.

    Returns
    -------
    t_grid : 1D array of t values.
    t_min, t_max : window bounds.
    lambda_max, lambda_min : extreme eigenvalues across ensemble.
    """
    # Collect all eigenvalues
    if not all_eigenvalues or all(len(e) == 0 for e in all_eigenvalues):
        return np.logspace(-3, 0, n_t), 1e-3, 1.0, 0.0, 0.0

    all_eig = np.concatenate([e for e in all_eigenvalues if len(e) > 0])
    lambda_max = float(np.max(all_eig))
    lambda_min = float(np.min(all_eig))

    if lambda_max <= 0 or lambda_min <= 0:
        return np.logspace(-3, 0, n_t), 1e-3, 1.0, lambda_max, lambda_min

    t_min = 3.0 / lambda_max
    t_max = min(20.0 / lambda_max, 1.0 / lambda_min)

    if t_max <= t_min:
        # Fallback: use a reasonable window
        t_max = 20.0 / lambda_max

    t_grid = np.logspace(np.log10(t_min), np.log10(t_max), n_t)
    return t_grid, t_min, t_max, lambda_max, lambda_min


# ---------------------------------------------------------------------------
# UV fitting
# ---------------------------------------------------------------------------

def fit_uv_exponent(t_grid: np.ndarray, K: np.ndarray) -> tuple[float, float]:
    """
    Fit ln(K) = p * ln(t) + const via OLS (log-log regression).

    Returns (p, intercept). Returns (np.nan, np.nan) if fit fails.
    """
    valid = (K > 0) & np.isfinite(K) & (t_grid > 0)
    if np.sum(valid) < 5:
        return np.nan, np.nan

    ln_t = np.log(t_grid[valid])
    ln_K = np.log(K[valid])

    result = linregress(ln_t, ln_K)
    return float(result.slope), float(result.intercept)


def bootstrap_ci(K_all: np.ndarray, t_grid: np.ndarray,
                 n_boot: int = N_BOOTSTRAP,
                 rng: np.random.Generator = None) -> tuple[float, float]:
    """
    Bootstrap 95% CI on the UV exponent p.

    Resample M sprinklings (rows of K_all) with replacement,
    compute K_ens_boot, fit p_boot, repeat n_boot times.

    Parameters
    ----------
    K_all : (M, n_t) array of individual heat traces.
    t_grid : (n_t,) array.
    n_boot : number of bootstrap samples.
    rng : random generator.

    Returns
    -------
    (ci_low, ci_high) : 2.5th and 97.5th percentiles of p.
    """
    if rng is None:
        rng = np.random.default_rng(BASE_SEED + 999)

    M = K_all.shape[0]
    p_boots = []
    for _ in range(n_boot):
        indices = rng.integers(0, M, size=M)
        K_boot = np.mean(K_all[indices], axis=0)
        p_val, _ = fit_uv_exponent(t_grid, K_boot)
        if np.isfinite(p_val):
            p_boots.append(p_val)

    if len(p_boots) < 10:
        return np.nan, np.nan

    p_boots = np.array(p_boots)
    return float(np.percentile(p_boots, 2.5)), float(np.percentile(p_boots, 97.5))


# ---------------------------------------------------------------------------
# Null model generation
# ---------------------------------------------------------------------------

def generate_null_A(N: int, target_frob: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Family A null: random real antisymmetric matrix -> i*M/2 -> rescale.
    """
    # Random antisymmetric: upper triangle Gaussian i.i.d.
    M_upper = rng.standard_normal((N, N))
    M_antisym = np.triu(M_upper, k=1) - np.triu(M_upper, k=1).T

    # Form A_null = i*M/2
    A_null = 1j * M_antisym / 2.0
    eigenvalues = np.linalg.eigvalsh(A_null)  # eigvalsh: 2.8x faster
    abs_eig = np.abs(eigenvalues)

    # Frobenius norm: sqrt(sum(eig^2))
    frob_null = np.sqrt(np.sum(abs_eig**2))
    if frob_null > 0:
        abs_eig *= target_frob / frob_null

    return abs_eig[abs_eig > ZERO_THRESHOLD]


def generate_null_B(N: int, target_frob: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Family B null: GOE (real symmetric Gaussian), project to traceless, rescale.
    """
    G = rng.standard_normal((N, N))
    G_sym = (G + G.T) / 2.0
    # Project to traceless
    G_sym -= np.trace(G_sym) / N * np.eye(N)

    eigenvalues = np.linalg.eigvalsh(G_sym)  # eigvalsh: 2.8x faster
    abs_eig = np.abs(eigenvalues)

    frob_null = np.sqrt(np.sum(abs_eig**2))
    if frob_null > 0:
        abs_eig *= target_frob / frob_null

    return abs_eig[abs_eig > ZERO_THRESHOLD]


def generate_null_C(N: int, target_frob: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Family C null: GOE, take |eigenvalues|, rescale Frobenius norm.
    """
    G = rng.standard_normal((N, N))
    G_sym = (G + G.T) / 2.0

    eigenvalues = np.linalg.eigvalsh(G_sym)  # eigvalsh: 2.8x faster
    abs_eig = np.abs(eigenvalues)

    frob_null = np.sqrt(np.sum(abs_eig**2))
    if frob_null > 0:
        abs_eig *= target_frob / frob_null

    return abs_eig[abs_eig > ZERO_THRESHOLD]


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------

def evaluate_gate0(p_ens: float, p_null: float, p_target: float) -> str:
    """
    Gate 0: is the causal result significantly better than null?

    Criterion: |p_ens - p_target| < |p_null - p_target| - 0.15
    """
    if np.isnan(p_ens) or np.isnan(p_null):
        return "INCONCLUSIVE"

    err_ens = abs(p_ens - p_target)
    err_null = abs(p_null - p_target)

    if err_ens < err_null - 0.15:
        return "DISCRIMINATING"
    else:
        return "NON-DISCRIMINATING"


def evaluate_gate1(p_ens: float, p_single: float, p_target: float,
                   gate0_verdict: str, uv_decades: float) -> str:
    """
    Gate 1 verdict per family.

    PASS (strong): |p_ens - p_target|/|p_target| < 0.30 AND gate0 != NON-DISCRIMINATING
    PASS (improvement): |p_ens - p_target| < |p_single - p_target| - 0.10 AND gate0 not ND
    INCONCLUSIVE: window < 0.5 decades, or 0.30 <= rel_error <= 0.50, or gate0 ND
    FAIL: rel_error > 0.50
    """
    if np.isnan(p_ens) or np.isnan(p_single):
        return "INCONCLUSIVE"

    if uv_decades < 0.5:
        return "INCONCLUSIVE (UV window < 0.5 decades)"

    rel_error = abs(p_ens - p_target) / abs(p_target)

    # Check Gate 0
    if gate0_verdict == "NON-DISCRIMINATING":
        return "INCONCLUSIVE (Gate 0 NON-DISCRIMINATING)"

    # Strong pass
    if rel_error < 0.30:
        return "PASS (strong)"

    # Improvement pass
    err_ens = abs(p_ens - p_target)
    err_single = abs(p_single - p_target)
    if err_ens < err_single - 0.10:
        return "PASS (improvement)"

    # Inconclusive range
    if 0.30 <= rel_error <= 0.50:
        return "INCONCLUSIVE"

    # Fail
    return "FAIL"


# ---------------------------------------------------------------------------
# Main ensemble runner
# ---------------------------------------------------------------------------

def run_ensemble(N: int = 200, M: int = 50, T: float = 1.0,
                 seed: int = BASE_SEED) -> EnsembleResults:
    """
    Run Gate 0 + Gate 1 for all three operator families.

    Parameters
    ----------
    N : number of causal set elements per sprinkling.
    M : number of sprinklings (ensemble size).
    T : time extent of the diamond (V = T^2/2).
    seed : base random seed.

    Returns
    -------
    EnsembleResults with all gate verdicts.
    """
    t0_wall = time.perf_counter()

    V = T**2 / 2.0
    rho = N / V  # = 2*N for T=1

    # Seed sequence for reproducible, independent streams
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(M + 1)  # M for sprinklings, 1 for null/bootstrap
    sprinkle_rngs = [np.random.default_rng(s) for s in child_seeds[:M]]
    aux_rng = np.random.default_rng(child_seeds[M])

    print(f"FND-1 Ensemble Runner: N={N}, M={M}, d=2, T={T}, V={V}, rho={rho}")
    print(f"Seed: {seed}")
    print("-" * 70)

    # Storage for eigenvalues per family per sprinkling
    eig_A_all = []
    eig_B_all = []
    eig_C_all = []
    frob_A_all = []
    frob_B_all = []
    frob_C_all = []

    # ---- Phase 1: Sprinkle and compute eigenvalues ----
    print("Phase 1: Generating sprinklings and computing eigenvalues...")
    for i in range(M):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Sprinkling {i+1}/{M}...")

        points, C = sprinkle_diamond(N, T, sprinkle_rngs[i])
        n_matrix = compute_interval_cardinalities(C)
        L = build_bd_L(C, n_matrix, rho)

        # Family A
        eig_A = compute_family_A_eigenvalues(L)
        eig_A_all.append(eig_A)
        frob_A_all.append(np.sqrt(np.sum(eig_A**2)) if len(eig_A) > 0 else 0.0)

        # Family B
        eig_B = compute_family_B_eigenvalues(L)
        eig_B_all.append(eig_B)
        frob_B_all.append(np.sqrt(np.sum(eig_B**2)) if len(eig_B) > 0 else 0.0)

        # Family C
        eig_C = compute_family_C_eigenvalues(C)
        eig_C_all.append(eig_C)
        frob_C_all.append(np.sqrt(np.sum(eig_C**2)) if len(eig_C) > 0 else 0.0)

    print(f"  Done. Mean N_eff: A={np.mean([len(e) for e in eig_A_all]):.1f}, "
          f"B={np.mean([len(e) for e in eig_B_all]):.1f}, "
          f"C={np.mean([len(e) for e in eig_C_all]):.1f}")

    # ---- Phase 2: Compute heat traces ----
    print("\nPhase 2: Computing heat traces and UV fits...")

    family_data = {
        "A": (eig_A_all, frob_A_all, generate_null_A),
        "B": (eig_B_all, frob_B_all, generate_null_B),
        "C": (eig_C_all, frob_C_all, generate_null_C),
    }

    results = EnsembleResults(N=N, M=M, T=T, V=V, rho=rho, seed=seed)

    for fname, (eig_all, frob_all, null_gen) in family_data.items():
        print(f"\n  --- Family {fname} (p_target = {P_TARGETS[fname]}) ---")

        # Determine UV window
        t_grid, t_min, t_max, lam_max, lam_min = determine_uv_window(eig_all)
        uv_decades = np.log10(t_max / t_min) if t_max > t_min else 0.0
        print(f"  UV window: t_min={t_min:.6f}, t_max={t_max:.6f}, "
              f"decades={uv_decades:.3f}")
        print(f"  lambda_max={lam_max:.2f}, lambda_min={lam_min:.6f}")

        # Heat traces for all sprinklings
        K_all = np.zeros((M, len(t_grid)))
        for i in range(M):
            K_all[i] = compute_heat_trace(eig_all[i], t_grid)

        # Ensemble average and variance
        K_ens = np.mean(K_all, axis=0)
        Var_K = np.mean((K_all - K_ens[np.newaxis, :])**2, axis=0)
        R_K = np.where(K_ens > 0, Var_K / K_ens**2, np.inf)

        # Single-matrix baseline (first sprinkling)
        K_single = K_all[0]

        # UV fit: ensemble
        p_ens, _ = fit_uv_exponent(t_grid, K_ens)
        print(f"  p_ens = {p_ens:.4f}")

        # UV fit: single
        p_single, _ = fit_uv_exponent(t_grid, K_single)
        print(f"  p_single = {p_single:.4f}")

        # Bootstrap CI
        ci_low, ci_high = bootstrap_ci(K_all, t_grid, rng=aux_rng)
        print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

        # Mean relative variance in UV window
        mean_R_K = float(np.mean(R_K))
        print(f"  Mean R_K = {mean_R_K:.4f}")

        # ---- Null model ----
        print(f"  Computing null model ({M} samples)...")
        target_frob = float(np.mean(frob_all))
        null_eig_all = []
        for j in range(M):
            null_eig = null_gen(N, target_frob, aux_rng)
            null_eig_all.append(null_eig)

        # Null heat traces on the SAME t-grid
        K_null_all = np.zeros((M, len(t_grid)))
        for j in range(M):
            K_null_all[j] = compute_heat_trace(null_eig_all[j], t_grid)

        K_null_ens = np.mean(K_null_all, axis=0)
        p_null, _ = fit_uv_exponent(t_grid, K_null_ens)
        print(f"  p_null = {p_null:.4f}")

        # ---- Gate 0 ----
        gate0 = evaluate_gate0(p_ens, p_null, P_TARGETS[fname])
        print(f"  Gate 0: {gate0}")

        # ---- Gate 1 ----
        rel_error = abs(p_ens - P_TARGETS[fname]) / abs(P_TARGETS[fname]) \
            if P_TARGETS[fname] != 0 else np.inf
        gate1 = evaluate_gate1(p_ens, p_single, P_TARGETS[fname],
                               gate0, uv_decades)
        print(f"  Gate 1: {gate1}")
        print(f"  Relative error: {rel_error:.4f}")

        # Store results
        fr = FamilyResult(
            name=fname,
            p_target=P_TARGETS[fname],
            p_ens=p_ens,
            p_single=p_single,
            p_null=p_null,
            ci_low=ci_low,
            ci_high=ci_high,
            relative_error=rel_error,
            uv_decades=uv_decades,
            mean_R_K_uv=mean_R_K,
            gate0_verdict=gate0,
            gate1_verdict=gate1,
            n_eff_mean=float(np.mean([len(e) for e in eig_all])),
            lambda_max=lam_max,
            lambda_min=lam_min,
            frob_norm_mean=target_frob,
        )
        results.families[fname] = fr

    results.wall_time_sec = time.perf_counter() - t0_wall

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for fname in ["A", "B", "C"]:
        fr = results.families[fname]
        print(f"\nFamily {fname} (target p = {fr.p_target}):")
        print(f"  p_ens = {fr.p_ens:.4f}  (95% CI: [{fr.ci_low:.4f}, {fr.ci_high:.4f}])")
        print(f"  p_single = {fr.p_single:.4f}")
        print(f"  p_null = {fr.p_null:.4f}")
        print(f"  Relative error = {fr.relative_error:.4f}")
        print(f"  UV decades = {fr.uv_decades:.3f}")
        print(f"  Mean R_K = {fr.mean_R_K_uv:.4f}")
        print(f"  Gate 0: {fr.gate0_verdict}")
        print(f"  Gate 1: {fr.gate1_verdict}")

    print(f"\nWall time: {results.wall_time_sec:.1f} s")

    return results


# ---------------------------------------------------------------------------
# JSON serialization & main
# ---------------------------------------------------------------------------

def save_results(results: EnsembleResults, output_path: Path):
    """Save results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "parameters": {
            "N": results.N,
            "M": results.M,
            "d": results.d,
            "T": results.T,
            "V": results.V,
            "rho": results.rho,
            "seed": results.seed,
        },
        "families": {},
        "wall_time_sec": results.wall_time_sec,
    }

    for fname, fr in results.families.items():
        data["families"][fname] = asdict(fr)

    # Handle NaN for JSON (replace with null)
    def _clean(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    data = _clean(data)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    """Run Gate 0 + Gate 1 with N=200, M=50."""
    results = run_ensemble(N=200, M=50, T=1.0, seed=BASE_SEED)

    # Save to JSON
    project_root = Path(__file__).resolve().parent.parent.parent
    output_path = project_root / "speculative" / "numerics" / "ensemble_results" \
        / "gate0_gate1_N200_M50.json"
    save_results(results, output_path)


if __name__ == "__main__":
    main()
