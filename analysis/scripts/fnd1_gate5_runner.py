"""
FND-1 Route 1: Gate 5 — Curvature Sensitivity.

THE decisive test: does the subleading heat-kernel coefficient from the
Family B (symmetrized BD) operator correlate with spacetime curvature?

A structureless null model (GOE, Wigner semicircle) CANNOT produce
curvature-dependent subleading coefficients. If a_1 proportional to
integral(R dvol), that is a genuine spectral-geometric signal.

Setup:
  - d=2 conformally flat spacetime: ds^2 = Omega^2(t,x)(-dt^2 + dx^2)
  - Omega(t,x) = 1 + epsilon * (t^2 - x^2)
  - R = 4*epsilon / (1 + epsilon*(t^2-x^2))^3  [exact 2D conformal formula]
  - Causal structure is conformally invariant in 2D -> same as Minkowski
  - Curvature enters through non-uniform sprinkling density ~ Omega^2

Family B only (the Gate 3 survivor).
N = 1000, M = 200 per curvature value, 5 curvature values.

Reference: speculative/FND1_ENSEMBLE_SPEC.md, Section 6 (Gate 5).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import linregress, pearsonr
from scipy.integrate import dblquad

# ---------------------------------------------------------------------------
# Import building blocks from the Gate 0/1 runner
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fnd1_ensemble_runner import (
    compute_interval_cardinalities,
    build_bd_L,
    compute_family_B_eigenvalues,
    compute_heat_trace,
    determine_uv_window,
    fit_uv_exponent,
    ZERO_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_POINTS = 1000         # causal set elements per sprinkling
M_ENSEMBLE = 200        # ensemble members per curvature value
T_DIAMOND = 1.0         # diamond half-extent (V_flat = T^2/2 = 0.5)
MASTER_SEED = 42

# 5 curvature values: two negative, flat, two positive
EPSILON_VALUES = [-0.5, -0.25, 0.0, 0.25, 0.5]

# Heat-trace parameter grid
N_T_GRID = 300
T_GRID_MIN = 1e-4
T_GRID_MAX = 5.0

# Subleading fit parameters
N_FIT_POINTS_MIN = 10   # minimum points in fit window


# ---------------------------------------------------------------------------
# Conformally flat sprinkling
# ---------------------------------------------------------------------------

def omega_squared(t: np.ndarray, x: np.ndarray, eps: float) -> np.ndarray:
    """Compute Omega^2(t,x) = (1 + eps*(t^2 - x^2))^2."""
    return (1.0 + eps * (t**2 - x**2))**2


def omega_sq_raw(t: np.ndarray, x: np.ndarray, eps: float) -> np.ndarray:
    """Compute Omega(t,x)^2 = (1 + eps*(t^2 - x^2))^2 as float."""
    val = 1.0 + eps * (t**2 - x**2)
    return val**2


def check_omega_positivity(eps: float, T: float) -> tuple[float, float]:
    """
    Check that Omega > 0 everywhere in the diamond.
    Returns (Omega_min, Omega_max) in the diamond.
    """
    # Omega = 1 + eps*(t^2 - x^2)
    # In the diamond |t| + |x| <= T/2, the extreme of t^2 - x^2 are:
    #   max at (t, x) = (+-T/2, 0): t^2-x^2 = T^2/4
    #   min at (t, x) = (0, +-T/2): t^2-x^2 = -T^2/4
    u_max = (T / 2)**2
    u_min = -(T / 2)**2

    omega_vals = [1.0 + eps * u_max, 1.0 + eps * u_min]
    return min(omega_vals), max(omega_vals)


def sprinkle_curved(N: int, eps: float, T: float,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Sprinkle N points into a d=2 causal diamond with density proportional
    to Omega^2(t,x) via rejection sampling.

    The causal diamond in null coordinates (u, v) = (t+x, t-x) is the
    square [-T/2, T/2]^2. We sample uniformly in (u, v), convert to (t, x),
    then accept with probability Omega^2 / max_Omega^2.

    Parameters
    ----------
    N : number of points to accept.
    eps : curvature parameter (Omega = 1 + eps*(t^2 - x^2)).
    T : diamond time extent.
    rng : numpy random generator.

    Returns
    -------
    points : (N, 2) array of (t, x) coordinates, sorted by t.
    C : (N, N) strictly upper-triangular causal matrix.
    """
    if eps == 0.0:
        # Flat case: uniform sprinkling, no rejection needed
        return _sprinkle_flat(N, T, rng)

    # Determine max Omega^2 in the diamond for rejection bound
    omega_min, omega_max = check_omega_positivity(eps, T)
    if omega_min <= 0:
        raise ValueError(
            f"Omega <= 0 in diamond for eps={eps}, T={T}. "
            f"Omega_min = {omega_min:.4f}"
        )

    max_omega_sq = omega_max**2

    # Rejection sampling
    t_accepted = []
    x_accepted = []
    n_accepted = 0
    n_total = 0
    batch_size = int(N * 1.5)  # overshoot for efficiency

    while n_accepted < N:
        # Sample uniformly in null coordinates
        u = rng.uniform(-T / 2, T / 2, size=batch_size)
        v = rng.uniform(-T / 2, T / 2, size=batch_size)
        t_cand = (u + v) / 2
        x_cand = (u - v) / 2

        # Accept/reject based on Omega^2
        omega_sq = omega_squared(t_cand, x_cand, eps)
        accept_prob = omega_sq / max_omega_sq
        rand_vals = rng.uniform(0, 1, size=batch_size)
        accepted = rand_vals < accept_prob

        t_accepted.extend(t_cand[accepted])
        x_accepted.extend(x_cand[accepted])
        n_accepted = len(t_accepted)
        n_total += batch_size

    # Trim to exactly N
    t_arr = np.array(t_accepted[:N])
    x_arr = np.array(x_accepted[:N])

    # Natural labeling: sort by time
    order = np.argsort(t_arr)
    t_arr = t_arr[order]
    x_arr = x_arr[order]
    points = np.column_stack([t_arr, x_arr])

    # Causal matrix: identical to Minkowski (conformal invariance in 2D)
    dt = t_arr[np.newaxis, :] - t_arr[:, np.newaxis]
    dx = np.abs(x_arr[np.newaxis, :] - x_arr[:, np.newaxis])
    C = ((dt > dx) & (dt > 0)).astype(np.float64)

    acceptance_rate = N / n_total if n_total > 0 else 0
    return points, C


def _sprinkle_flat(N: int, T: float,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Flat-space sprinkling (eps=0) — no rejection needed."""
    u = rng.uniform(-T / 2, T / 2, size=N)
    v = rng.uniform(-T / 2, T / 2, size=N)
    t = (u + v) / 2
    x = (u - v) / 2

    order = np.argsort(t)
    t = t[order]
    x = x[order]
    points = np.column_stack([t, x])

    dt = t[np.newaxis, :] - t[:, np.newaxis]
    dx = np.abs(x[np.newaxis, :] - x[:, np.newaxis])
    C = ((dt > dx) & (dt > 0)).astype(np.float64)

    return points, C


# ---------------------------------------------------------------------------
# Integrated curvature computation
# ---------------------------------------------------------------------------

def compute_integrated_curvature(eps: float, T: float,
                                 n_mc: int = 500000) -> float:
    """
    Compute integral(R dvol) over the causal diamond via Monte Carlo.

    R dvol = R * Omega^2 d^2x
    R = 4*eps / (1 + eps*(t^2-x^2))^3  [exact 2D conformal]
    Omega^2 = (1 + eps*(t^2-x^2))^2
    So R * Omega^2 = 4*eps / (1 + eps*(t^2-x^2))

    integral(R dvol) = integral over diamond of 4*eps / (1+eps*u) d^2x
    where u = t^2 - x^2.

    For eps=0: integral = 0 (flat).
    """
    if abs(eps) < 1e-15:
        return 0.0

    # Monte Carlo integration over the diamond
    rng = np.random.default_rng(12345)  # fixed seed for reproducibility

    # Sample uniformly in null coordinates
    u_null = rng.uniform(-T / 2, T / 2, size=n_mc)
    v_null = rng.uniform(-T / 2, T / 2, size=n_mc)
    t = (u_null + v_null) / 2
    x = (u_null - v_null) / 2

    u = t**2 - x**2
    integrand = 4.0 * eps / (1.0 + eps * u)

    # Diamond coordinate area = T^2 / 2 (in null coords: area = T^2)
    # But Jacobian |d(t,x)/d(u,v)| = 1/2, so area in (t,x) = T^2/2
    # MC: integral = (area) * mean(integrand)
    coord_area = T**2 / 2.0  # area of diamond in (t,x) coords
    integral = coord_area * np.mean(integrand)

    return float(integral)


# ---------------------------------------------------------------------------
# Subleading coefficient extraction
# ---------------------------------------------------------------------------

def extract_subleading(eigenvalues: np.ndarray, t_grid: np.ndarray,
                       d: int = 2) -> dict:
    """
    Extract the subleading heat-kernel coefficient a_1 from the heat trace.

    For d=2: K(t) ~ (4*pi*t)^{-1} * [a_0 + a_1*t + a_2*t^2 + ...]
    Define F(t) = K(t) * (4*pi*t).
    Then F(t) ~ a_0 + a_1*t + ...

    Strategy:
      1. Compute F(t) over the grid.
      2. Find the plateau region (where F is approximately constant = a_0).
      3. Fit F(t) = a_0 + a_1*t in an extended range beyond the plateau.

    Returns dict with a_0, a_1, fit quality metrics.
    """
    K = compute_heat_trace(eigenvalues, t_grid)
    F = K * (4.0 * np.pi * t_grid)

    # Find the plateau: look for the region where F is most stable
    # Use a running variance over log-spaced windows
    valid = np.isfinite(F) & (F > 0)
    if np.sum(valid) < 20:
        return {"a_0": np.nan, "a_1": np.nan, "r_squared": 0.0,
                "F": F, "K": K}

    # Estimate a_0 from the small-t end of F (first 20% of valid range)
    n_valid = np.sum(valid)
    idx_valid = np.where(valid)[0]
    n_plateau = max(5, n_valid // 5)
    a_0_est = np.median(F[idx_valid[:n_plateau]])

    # Fit window: from 10th percentile to 60th percentile of the valid range
    # This captures the transition from plateau to linear deviation
    idx_start = idx_valid[max(0, n_valid // 10)]
    idx_end = idx_valid[min(n_valid - 1, int(0.6 * n_valid))]

    fit_mask = valid.copy()
    fit_mask[:idx_start] = False
    fit_mask[idx_end:] = False

    if np.sum(fit_mask) < N_FIT_POINTS_MIN:
        # Fallback: use wider range
        fit_mask = valid.copy()
        n_use = min(n_valid, max(N_FIT_POINTS_MIN, n_valid // 2))
        fit_mask_idx = idx_valid[:n_use]
        fit_mask = np.zeros_like(valid)
        fit_mask[fit_mask_idx] = True

    t_fit = t_grid[fit_mask]
    F_fit = F[fit_mask]

    if len(t_fit) < 3:
        return {"a_0": a_0_est, "a_1": np.nan, "r_squared": 0.0,
                "F": F, "K": K}

    # Linear regression: F = a_0 + a_1 * t
    result = linregress(t_fit, F_fit)
    a_0 = result.intercept
    a_1 = result.slope
    r_sq = result.rvalue**2

    return {
        "a_0": float(a_0),
        "a_1": float(a_1),
        "a_1_stderr": float(result.stderr),
        "r_squared": float(r_sq),
        "F": F,
        "K": K,
        "t_fit_range": (float(t_fit[0]), float(t_fit[-1])),
    }


def extract_subleading_ensemble(all_eigenvalues: list[np.ndarray],
                                t_grid: np.ndarray) -> dict:
    """
    Extract a_1 from the ensemble-averaged heat trace.

    Also computes individual a_1 values for error estimation.
    """
    M = len(all_eigenvalues)

    # Compute ensemble-averaged F(t)
    K_all = np.zeros((M, len(t_grid)))
    for i in range(M):
        K_all[i] = compute_heat_trace(all_eigenvalues[i], t_grid)

    K_ens = np.mean(K_all, axis=0)
    F_ens = K_ens * (4.0 * np.pi * t_grid)

    # Extract from ensemble average
    ens_result = extract_subleading(
        np.array([]),  # dummy — we'll override
        t_grid,
    )
    # Override: compute from ensemble-averaged K directly
    F = F_ens
    valid = np.isfinite(F) & (F > 0)

    if np.sum(valid) < 20:
        return {
            "a_0": np.nan, "a_1": np.nan, "a_1_stderr": np.nan,
            "r_squared": 0.0, "a_1_individual": [],
            "F_ens": F_ens, "K_ens": K_ens,
        }

    n_valid = np.sum(valid)
    idx_valid = np.where(valid)[0]

    # Plateau estimate
    n_plateau = max(5, n_valid // 5)
    a_0_est = float(np.median(F[idx_valid[:n_plateau]]))

    # Fit window: 10th to 60th percentile
    idx_start = idx_valid[max(0, n_valid // 10)]
    idx_end = idx_valid[min(n_valid - 1, int(0.6 * n_valid))]

    fit_mask = valid.copy()
    fit_mask[:idx_start] = False
    fit_mask[idx_end:] = False

    if np.sum(fit_mask) < N_FIT_POINTS_MIN:
        fit_mask = valid.copy()
        n_use = min(n_valid, max(N_FIT_POINTS_MIN, n_valid // 2))
        fit_mask = np.zeros_like(valid)
        fit_mask[idx_valid[:n_use]] = True

    t_fit = t_grid[fit_mask]
    F_fit = F[fit_mask]

    if len(t_fit) < 3:
        return {
            "a_0": a_0_est, "a_1": np.nan, "a_1_stderr": np.nan,
            "r_squared": 0.0, "a_1_individual": [],
            "F_ens": F_ens, "K_ens": K_ens,
        }

    result = linregress(t_fit, F_fit)
    a_0 = float(result.intercept)
    a_1 = float(result.slope)
    a_1_stderr = float(result.stderr)
    r_sq = float(result.rvalue**2)

    # Individual a_1 values for bootstrap error estimation
    a_1_individual = []
    for i in range(M):
        F_i = K_all[i] * (4.0 * np.pi * t_grid)
        if np.sum(np.isfinite(F_i) & (F_i > 0)) >= N_FIT_POINTS_MIN:
            F_i_fit = F_i[fit_mask]
            if len(F_i_fit) >= 3 and np.all(np.isfinite(F_i_fit)):
                try:
                    res_i = linregress(t_fit, F_i_fit)
                    a_1_individual.append(float(res_i.slope))
                except Exception:
                    pass

    return {
        "a_0": a_0,
        "a_1": a_1,
        "a_1_stderr": a_1_stderr,
        "a_1_ensemble_std": float(np.std(a_1_individual)) if a_1_individual else np.nan,
        "a_1_ensemble_sem": (float(np.std(a_1_individual) / np.sqrt(len(a_1_individual)))
                             if len(a_1_individual) > 1 else np.nan),
        "r_squared": r_sq,
        "a_1_individual": a_1_individual,
        "F_ens": F_ens,
        "K_ens": K_ens,
        "t_fit_range": (float(t_fit[0]), float(t_fit[-1])),
    }


# ---------------------------------------------------------------------------
# Main Gate 5 runner
# ---------------------------------------------------------------------------

@dataclass
class Gate5Result:
    """Results for one curvature value."""
    epsilon: float
    int_R_dvol: float
    a_0: float
    a_1: float
    a_1_stderr: float
    a_1_ensemble_sem: float
    r_squared: float
    n_eff_mean: float
    acceptance_rate: float
    p_ens: float  # UV exponent from ensemble
    wall_time_sec: float


def run_gate5_single_epsilon(eps: float, N: int, M: int, T: float,
                             seed_seq: np.random.SeedSequence,
                             t_grid: np.ndarray) -> tuple[Gate5Result, dict]:
    """
    Run M sprinklings at curvature epsilon, extract subleading coefficient.
    """
    t0 = time.perf_counter()

    V_flat = T**2 / 2.0
    rho = N / V_flat  # base density

    # Spawn M seeds
    child_seeds = seed_seq.spawn(M)

    # Phase 1: sprinkle, build operator, extract eigenvalues
    all_eigenvalues = []
    n_eff_list = []
    n_total_sprinkled = 0
    n_total_attempted = 0

    print(f"  Phase 1: Generating {M} sprinklings at eps={eps:.3f}...")
    for i in range(M):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    Sprinkling {i + 1}/{M}...")

        rng = np.random.default_rng(child_seeds[i])
        points, C = sprinkle_curved(N, eps, T, rng)
        n_matrix = compute_interval_cardinalities(C)
        L = build_bd_L(C, n_matrix, rho)
        eig = compute_family_B_eigenvalues(L)

        all_eigenvalues.append(eig)
        n_eff_list.append(len(eig))

    n_eff_mean = float(np.mean(n_eff_list))
    print(f"    Mean N_eff = {n_eff_mean:.1f}")

    # Phase 2: ensemble heat trace and subleading extraction
    print(f"  Phase 2: Extracting subleading coefficient...")
    sub_result = extract_subleading_ensemble(all_eigenvalues, t_grid)

    # Also compute UV exponent for diagnostics
    K_all = np.zeros((M, len(t_grid)))
    for i in range(M):
        K_all[i] = compute_heat_trace(all_eigenvalues[i], t_grid)
    K_ens = np.mean(K_all, axis=0)

    # UV window
    t_grid_uv, t_min_uv, t_max_uv, lam_max, lam_min = determine_uv_window(
        all_eigenvalues
    )
    K_ens_uv = np.zeros(len(t_grid_uv))
    for i in range(M):
        K_ens_uv += compute_heat_trace(all_eigenvalues[i], t_grid_uv)
    K_ens_uv /= M
    p_ens, _ = fit_uv_exponent(t_grid_uv, K_ens_uv)

    # Integrated curvature
    int_R_dvol = compute_integrated_curvature(eps, T)

    wall_time = time.perf_counter() - t0

    # Estimate acceptance rate
    omega_min, omega_max = check_omega_positivity(eps, T)
    max_omega_sq = omega_max**2
    mean_omega_sq = 1.0  # approximate (symmetric integrand cancels leading correction)
    acc_rate = mean_omega_sq / max_omega_sq if abs(eps) > 1e-15 else 1.0

    result = Gate5Result(
        epsilon=eps,
        int_R_dvol=int_R_dvol,
        a_0=sub_result["a_0"],
        a_1=sub_result["a_1"],
        a_1_stderr=sub_result["a_1_stderr"],
        a_1_ensemble_sem=sub_result.get("a_1_ensemble_sem", np.nan),
        r_squared=sub_result["r_squared"],
        n_eff_mean=n_eff_mean,
        acceptance_rate=acc_rate,
        p_ens=float(p_ens) if np.isfinite(p_ens) else np.nan,
        wall_time_sec=wall_time,
    )

    diagnostics = {
        "F_ens": sub_result["F_ens"],
        "K_ens": sub_result["K_ens"],
        "a_1_individual": sub_result.get("a_1_individual", []),
        "t_fit_range": sub_result.get("t_fit_range", (np.nan, np.nan)),
    }

    return result, diagnostics


def run_gate5() -> dict:
    """
    Run Gate 5: curvature sensitivity test for Family B.

    Returns full results dict.
    """
    t0_total = time.perf_counter()

    print("=" * 70)
    print("FND-1 ROUTE 1: GATE 5 — CURVATURE SENSITIVITY")
    print("=" * 70)
    print(f"Family B (symmetrized BD) ONLY — the Gate 3 survivor")
    print(f"N = {N_POINTS}, M = {M_ENSEMBLE} per curvature value")
    print(f"T = {T_DIAMOND}, V_flat = {T_DIAMOND**2/2:.2f}")
    print(f"Epsilon values: {EPSILON_VALUES}")
    print(f"Master seed: {MASTER_SEED}")
    print()

    # Check Omega positivity for all epsilon values
    print("Omega positivity check:")
    for eps in EPSILON_VALUES:
        o_min, o_max = check_omega_positivity(eps, T_DIAMOND)
        status = "OK" if o_min > 0 else "FAIL"
        print(f"  eps={eps:+.3f}: Omega in [{o_min:.4f}, {o_max:.4f}] — {status}")
    print()

    # Compute integrated curvature for all epsilon values
    print("Integrated curvature (Monte Carlo, 500k samples):")
    int_R_values = {}
    for eps in EPSILON_VALUES:
        int_R = compute_integrated_curvature(eps, T_DIAMOND)
        int_R_values[eps] = int_R
        print(f"  eps={eps:+.3f}: integral(R dvol) = {int_R:+.6f}")
    print()

    # Heat trace t-grid (shared across all epsilon values for comparability)
    t_grid = np.logspace(np.log10(T_GRID_MIN), np.log10(T_GRID_MAX), N_T_GRID)

    # Seed hierarchy
    master_ss = np.random.SeedSequence(MASTER_SEED)
    eps_seeds = master_ss.spawn(len(EPSILON_VALUES))

    # Run for each epsilon value
    results_by_eps = {}
    diagnostics_by_eps = {}

    for idx, eps in enumerate(EPSILON_VALUES):
        print(f"\n{'='*60}")
        print(f"EPSILON = {eps:+.3f} (integral(R dvol) = {int_R_values[eps]:+.6f})")
        print(f"{'='*60}")

        result, diag = run_gate5_single_epsilon(
            eps, N_POINTS, M_ENSEMBLE, T_DIAMOND,
            eps_seeds[idx], t_grid,
        )

        results_by_eps[eps] = result
        diagnostics_by_eps[eps] = diag

        print(f"  a_0 = {result.a_0:.6f}")
        print(f"  a_1 = {result.a_1:.6f} +/- {result.a_1_stderr:.6f}")
        print(f"  a_1 ensemble SEM = {result.a_1_ensemble_sem:.6f}"
              if np.isfinite(result.a_1_ensemble_sem) else
              f"  a_1 ensemble SEM = N/A")
        print(f"  R^2(fit) = {result.r_squared:.4f}")
        print(f"  p_ens (UV exponent) = {result.p_ens:.4f}")
        print(f"  Wall time: {result.wall_time_sec:.1f}s")

    # -----------------------------------------------------------------------
    # Correlation analysis
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("CORRELATION ANALYSIS: a_1 vs integral(R dvol)")
    print(f"{'='*70}")

    eps_arr = np.array(EPSILON_VALUES)
    int_R_arr = np.array([results_by_eps[e].int_R_dvol for e in EPSILON_VALUES])
    a_1_arr = np.array([results_by_eps[e].a_1 for e in EPSILON_VALUES])
    a_1_sem_arr = np.array([results_by_eps[e].a_1_ensemble_sem
                            for e in EPSILON_VALUES])
    a_0_arr = np.array([results_by_eps[e].a_0 for e in EPSILON_VALUES])

    print(f"\n  {'eps':>8} {'int_R_dvol':>12} {'a_0':>12} {'a_1':>12} "
          f"{'a_1_SEM':>12} {'p_ens':>8}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
    for eps in EPSILON_VALUES:
        r = results_by_eps[eps]
        sem_str = f"{r.a_1_ensemble_sem:.6f}" if np.isfinite(r.a_1_ensemble_sem) else "N/A"
        print(f"  {eps:+8.3f} {r.int_R_dvol:+12.6f} {r.a_0:12.6f} "
              f"{r.a_1:+12.6f} {sem_str:>12} {r.p_ens:8.4f}")

    # Pearson correlation: a_1 vs integral(R dvol)
    valid = np.isfinite(a_1_arr) & np.isfinite(int_R_arr)
    if np.sum(valid) >= 3:
        r_pearson, p_value = pearsonr(int_R_arr[valid], a_1_arr[valid])
        print(f"\n  Pearson r = {r_pearson:.6f}")
        print(f"  p-value = {p_value:.2e}")

        # Linear fit: a_1 = k * int_R + b
        lr = linregress(int_R_arr[valid], a_1_arr[valid])
        k_slope = lr.slope
        b_intercept = lr.intercept
        r_sq_corr = lr.rvalue**2
        print(f"  Linear fit: a_1 = {k_slope:.6f} * int(R dvol) + {b_intercept:.6f}")
        print(f"  R^2 = {r_sq_corr:.6f}")

        # Sign consistency: do nonzero-curvature a_1 values have consistent
        # sign relative to flat?
        a_1_flat = results_by_eps[0.0].a_1
        nonzero_eps = [e for e in EPSILON_VALUES if e != 0.0]
        deltas = [(results_by_eps[e].a_1 - a_1_flat) for e in nonzero_eps]
        signs = [np.sign(d) for d in deltas]
        int_R_signs = [np.sign(compute_integrated_curvature(e, T_DIAMOND))
                       for e in nonzero_eps]
        # Sign consistent if sgn(delta_a1) * sgn(int_R) is the same for all
        relative_signs = [s * irs for s, irs in zip(signs, int_R_signs)]
        sign_consistent = (len(set(relative_signs)) == 1 and 0.0 not in relative_signs)
    else:
        r_pearson = np.nan
        p_value = np.nan
        k_slope = np.nan
        b_intercept = np.nan
        r_sq_corr = np.nan
        sign_consistent = False
        print("\n  Insufficient valid data for correlation analysis.")

    # Also test: a_1 vs epsilon directly (should be equivalent since
    # int_R ~ 4*eps to leading order)
    if np.sum(valid) >= 3:
        r_eps, p_eps = pearsonr(eps_arr[valid], a_1_arr[valid])
        lr_eps = linregress(eps_arr[valid], a_1_arr[valid])
        print(f"\n  Direct: a_1 vs eps: Pearson r = {r_eps:.6f}, p = {p_eps:.2e}")
        print(f"  Linear fit: a_1 = {lr_eps.slope:.6f} * eps + {lr_eps.intercept:.6f}")

    # -----------------------------------------------------------------------
    # VERDICT
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("GATE 5 VERDICT")
    print(f"{'='*70}")

    if np.isnan(r_pearson):
        verdict = "INCONCLUSIVE (insufficient data)"
    elif abs(r_pearson) > 0.8 and sign_consistent and p_value < 0.05:
        verdict = "PASS"
    elif abs(r_pearson) > 0.8 and p_value < 0.10:
        verdict = "PASS (marginal)"
    elif abs(r_pearson) > 0.6:
        verdict = "WEAK SIGNAL (r > 0.6 but below threshold)"
    else:
        verdict = "FAIL (no significant correlation)"

    print(f"  |Pearson r| = {abs(r_pearson):.4f} (threshold: 0.8)")
    print(f"  p-value = {p_value:.2e} (threshold: 0.05)")
    print(f"  Sign consistent: {sign_consistent}")
    print(f"  VERDICT: {verdict}")

    total_time = time.perf_counter() - t0_total
    print(f"\n  Total wall time: {total_time:.1f}s "
          f"({total_time/60:.1f} min)")

    # -----------------------------------------------------------------------
    # Assemble output
    # -----------------------------------------------------------------------
    output = {
        "parameters": {
            "N": N_POINTS,
            "M": M_ENSEMBLE,
            "T": T_DIAMOND,
            "V_flat": T_DIAMOND**2 / 2.0,
            "epsilon_values": EPSILON_VALUES,
            "seed": MASTER_SEED,
            "family": "B",
            "n_t_grid": N_T_GRID,
        },
        "curvature_data": {},
        "correlation": {
            "pearson_r": float(r_pearson) if np.isfinite(r_pearson) else None,
            "p_value": float(p_value) if np.isfinite(p_value) else None,
            "slope_k": float(k_slope) if np.isfinite(k_slope) else None,
            "intercept_b": float(b_intercept) if np.isfinite(b_intercept) else None,
            "r_squared": float(r_sq_corr) if np.isfinite(r_sq_corr) else None,
            "sign_consistent": sign_consistent,
        },
        "verdict": verdict,
        "wall_time_sec": total_time,
    }

    for eps in EPSILON_VALUES:
        r = results_by_eps[eps]
        output["curvature_data"][str(eps)] = {
            "epsilon": r.epsilon,
            "int_R_dvol": r.int_R_dvol,
            "a_0": r.a_0,
            "a_1": r.a_1,
            "a_1_stderr": r.a_1_stderr,
            "a_1_ensemble_sem": (float(r.a_1_ensemble_sem)
                                 if np.isfinite(r.a_1_ensemble_sem) else None),
            "r_squared": r.r_squared,
            "n_eff_mean": r.n_eff_mean,
            "p_ens": float(r.p_ens) if np.isfinite(r.p_ens) else None,
            "wall_time_sec": r.wall_time_sec,
        }

    return output, diagnostics_by_eps, t_grid


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(output: dict, diagnostics: dict, t_grid: np.ndarray,
                 save_dir: Path):
    """Generate diagnostic and result plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots.")
        return

    save_dir.mkdir(parents=True, exist_ok=True)

    eps_values = output["parameters"]["epsilon_values"]
    colors = {-0.5: "tab:blue", -0.25: "tab:cyan", 0.0: "black",
              0.25: "tab:orange", 0.5: "tab:red"}

    # ---- Plot 1: F(t) = K(t)*(4*pi*t) for each epsilon ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for eps in eps_values:
        F_ens = diagnostics[eps]["F_ens"]
        label = f"eps={eps:+.2f}"
        ax.semilogx(t_grid, F_ens, color=colors.get(eps, "gray"),
                     label=label, linewidth=1.5)

    ax.set_xlabel("t (heat parameter)")
    ax.set_ylabel("F(t) = K(t) * 4*pi*t")
    ax.set_title("Gate 5: Rescaled Heat Trace F(t) — Family B")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t_grid[0], t_grid[-1])

    fig.tight_layout()
    fig.savefig(save_dir / "gate5_F_vs_t.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_dir / 'gate5_F_vs_t.png'}")

    # ---- Plot 2: a_1 vs integral(R dvol) — the money plot ----
    fig, ax = plt.subplots(figsize=(7, 5))

    int_R_arr = []
    a_1_arr = []
    a_1_err = []

    for eps in eps_values:
        cd = output["curvature_data"][str(eps)]
        int_R_arr.append(cd["int_R_dvol"])
        a_1_arr.append(cd["a_1"])
        sem = cd.get("a_1_ensemble_sem")
        a_1_err.append(sem if sem is not None else 0.0)

    int_R_arr = np.array(int_R_arr)
    a_1_arr = np.array(a_1_arr)
    a_1_err = np.array(a_1_err)

    ax.errorbar(int_R_arr, a_1_arr, yerr=a_1_err, fmt="o", markersize=8,
                capsize=4, color="tab:blue", zorder=5)

    # Add epsilon labels
    for i, eps in enumerate(eps_values):
        ax.annotate(f"  eps={eps:+.2f}", (int_R_arr[i], a_1_arr[i]),
                    fontsize=8, alpha=0.7)

    # Overlay linear fit if available
    corr = output["correlation"]
    if corr["slope_k"] is not None:
        x_fit = np.linspace(int_R_arr.min() - 0.5, int_R_arr.max() + 0.5, 100)
        y_fit = corr["slope_k"] * x_fit + corr["intercept_b"]
        r_str = f"r = {corr['pearson_r']:.3f}" if corr["pearson_r"] is not None else ""
        ax.plot(x_fit, y_fit, "--", color="tab:red", linewidth=1.5,
                label=f"Linear fit ({r_str})")

    ax.set_xlabel(r"$\int R\, \mathrm{dvol}$")
    ax.set_ylabel(r"$a_1$ (subleading coefficient)")
    ax.set_title(f"Gate 5: Subleading vs Curvature — Verdict: {output['verdict']}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(save_dir / "gate5_a1_vs_curvature.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_dir / 'gate5_a1_vs_curvature.png'}")

    # ---- Plot 3: a_1 distribution per epsilon (box plot) ----
    fig, ax = plt.subplots(figsize=(8, 5))

    data_for_box = []
    labels_for_box = []
    for eps in eps_values:
        ind = diagnostics[eps].get("a_1_individual", [])
        if ind:
            data_for_box.append(ind)
            labels_for_box.append(f"eps={eps:+.2f}")

    if data_for_box:
        bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        for i, eps in enumerate(eps_values):
            if i < len(bp["boxes"]):
                bp["boxes"][i].set_facecolor(colors.get(eps, "gray"))
                bp["boxes"][i].set_alpha(0.5)

        ax.set_ylabel(r"$a_1$ (individual sprinklings)")
        ax.set_title("Gate 5: Distribution of Individual a_1 Values")
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(save_dir / "gate5_a1_distributions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_dir / 'gate5_a1_distributions.png'}")

    # ---- Plot 4: K_ens(t) in log-log for UV diagnostics ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for eps in eps_values:
        K_ens = diagnostics[eps]["K_ens"]
        valid = K_ens > 0
        if np.any(valid):
            ax.loglog(t_grid[valid], K_ens[valid],
                      color=colors.get(eps, "gray"),
                      label=f"eps={eps:+.2f}", linewidth=1.5)

    # Reference power law K ~ t^{-1}
    t_ref = np.logspace(-3, 0, 100)
    K_ref = t_ref**(-1.0)
    K_ref *= (diagnostics[0.0]["K_ens"][np.argmin(np.abs(t_grid - 0.01))]
              / t_ref[np.argmin(np.abs(t_ref - 0.01))] if 0.0 in diagnostics else 1)
    ax.loglog(t_ref, K_ref, "k:", linewidth=1, alpha=0.5, label=r"$t^{-1}$ reference")

    ax.set_xlabel("t")
    ax.set_ylabel("K_ens(t)")
    ax.set_title("Gate 5: Ensemble Heat Traces (log-log)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(save_dir / "gate5_heat_traces_loglog.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_dir / 'gate5_heat_traces_loglog.png'}")


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

def save_results_json(output: dict, output_path: Path):
    """Save results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean NaN/inf for JSON
    def _clean(obj):
        if isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
        if isinstance(obj, np.floating):
            v = float(obj)
            if np.isnan(v) or np.isinf(v):
                return None
            return v
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    data = _clean(output)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    print()

    output, diagnostics, t_grid = run_gate5()

    # Save results
    project_root = Path(__file__).resolve().parent.parent.parent
    results_dir = project_root / "speculative" / "numerics" / "ensemble_results"
    save_results_json(output, results_dir / "gate5_curvature_results.json")

    # Generate plots
    fig_dir = results_dir / "gate5_figures"
    print(f"\nGenerating plots...")
    plot_results(output, diagnostics, t_grid, fig_dir)

    # Final summary
    print(f"\n{'='*70}")
    print("GATE 5 COMPLETE")
    print(f"{'='*70}")
    print(f"Verdict: {output['verdict']}")
    corr = output['correlation']
    if corr['pearson_r'] is not None:
        print(f"Pearson r = {corr['pearson_r']:.4f}, p = {corr['p_value']:.2e}")
    print(f"Sign consistent: {corr['sign_consistent']}")
    print(f"Total wall time: {output['wall_time_sec']:.1f}s "
          f"({output['wall_time_sec']/60:.1f} min)")


if __name__ == "__main__":
    main()
