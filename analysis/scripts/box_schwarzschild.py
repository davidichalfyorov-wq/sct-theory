# ruff: noqa: E402, I001
"""Covariant d'Alembertian on static spherically symmetric backgrounds.

Provides finite-difference discretization of the scalar Box operator on
Schwarzschild (and general static SSS) spacetimes, together with matrix
function evaluation for the SCT nonlocal form factors.

Operator:
    For a static scalar Phi(r) decomposed in spherical harmonics Y_l^m:

        Box Phi_l = (1/r^2) d/dr [r^2 A(r) dPhi_l/dr] - l(l+1) A(r) Phi_l / r^2

    where A(r) = 1 - r_s/r for Schwarzschild with r_s = 2M (natural units G=c=1).

Grid strategy:
    Logarithmic spacing near the horizon r_min = r_s(1 + epsilon), transitioning
    to uniform spacing at large r, via tortoise coordinate
        r* = r + r_s ln(r/r_s - 1)
    which maps the horizon r->r_s to r*->-inf and compresses near-horizon physics.

Discretization:
    Second-order centered finite differences on the conservative form:
        (1/r_i^2) [f_{i+1/2}(Phi_{i+1}-Phi_i)/h - f_{i-1/2}(Phi_i-Phi_{i-1})/h] / h
    where f(r) = r^2 A(r), giving a tridiagonal matrix L.

Matrix form factor:
    Given L representing Box, computes F_hat_1(L/Lambda^2) via eigendecomposition:
        L = V diag(lambda_j) V^{-1}
        F_hat_1(L/Lambda^2) = V diag(F_hat_1(lambda_j/Lambda^2)) V^{-1}

Conventions:
    - Natural units: G = c = 1
    - Schwarzschild: A(r) = 1 - 2M/r, r_s = 2M
    - Metric signature: (-,+,+,+)
    - SCT form factors from sct_tools.form_factors
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la
from scipy.sparse import diags as sparse_diags

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from sct_tools.form_factors import (
    F1_total,
    F2_total,
    get_taylor_coefficients,
    phi_fast,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "box_schwarzschild"

# SM multiplicities (from sct_tools.constants)
_N_S = 4     # real scalar d.o.f. (Higgs doublet)
_N_F = 45    # 2-component Weyl fermion d.o.f.
_N_V = 12    # vector d.o.f.
_N_D = _N_F / 2.0  # Dirac fermions


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def schwarzschild_A(r: NDArray | float, M: float = 1.0) -> NDArray | float:
    """Lapse function A(r) = 1 - 2M/r for Schwarzschild."""
    return 1.0 - 2.0 * M / r


def flat_A(r: NDArray | float) -> NDArray | float:
    """Lapse function A(r) = 1 for flat space (M=0 limit)."""
    if isinstance(r, np.ndarray):
        return np.ones_like(r)
    return 1.0


def tortoise_from_r(r: NDArray | float, M: float = 1.0) -> NDArray | float:
    """Tortoise coordinate r* = r + 2M ln(r/(2M) - 1).

    Valid for r > r_s = 2M. Maps:
        r -> r_s+  :  r* -> -inf
        r -> +inf  :  r* -> +inf  (asymptotically r* ~ r)
    """
    r_s = 2.0 * M
    return r + r_s * np.log(r / r_s - 1.0)


def r_from_tortoise(rstar: NDArray | float, M: float = 1.0,
                    tol: float = 1e-14, maxiter: int = 200) -> NDArray | float:
    """Invert r*(r) via Newton iteration.

    The equation is: r + 2M ln(r/(2M) - 1) = r*
    Newton step: r_{n+1} = r_n - [r_n + 2M ln(r_n/(2M)-1) - r*] / [r_n/(r_n-2M)]
    """
    r_s = 2.0 * M
    rstar_arr = np.asarray(rstar, dtype=float)
    scalar = rstar_arr.ndim == 0
    rstar_arr = np.atleast_1d(rstar_arr)

    # Initial guess: r ~ r* for large r*, r ~ r_s(1+exp(r*/r_s)) for small r*
    r = np.where(rstar_arr > 10 * r_s,
                 rstar_arr,
                 r_s * (1.0 + np.exp(np.clip(rstar_arr / r_s, -500, 500))))

    for _ in range(maxiter):
        f = r + r_s * np.log(r / r_s - 1.0) - rstar_arr
        fp = r / (r - r_s)  # dr*/dr = r/(r - r_s)
        dr = f / fp
        r = r - dr
        # Ensure r > r_s at all times
        r = np.maximum(r, r_s * (1.0 + 1e-15))
        if np.all(np.abs(dr) < tol * np.abs(r)):
            break

    if scalar:
        return float(r[0])
    return r


# =============================================================================
# GRID CONSTRUCTION
# =============================================================================

@dataclass
class RadialGrid:
    """Radial grid for finite-difference discretization.

    Attributes:
        r: physical radial coordinates r_i, shape (N,)
        rstar: tortoise coordinates r*_i, shape (N,)
        h: grid spacing in the chosen coordinate (uniform in r*)
        N: number of grid points
        M: black hole mass parameter
        coord: 'tortoise' or 'physical'
        r_min: minimum physical radius
        r_max: maximum physical radius
    """
    r: NDArray
    rstar: NDArray
    h: float
    N: int
    M: float
    coord: str
    r_min: float
    r_max: float

    # Derived
    A: NDArray = field(init=False, repr=False)
    f: NDArray = field(init=False, repr=False)  # f(r) = r^2 * A(r)

    def __post_init__(self):
        self.A = schwarzschild_A(self.r, self.M) if self.M > 0 else flat_A(self.r)
        self.f = self.r**2 * self.A


def make_tortoise_grid(N: int, rstar_min: float, rstar_max: float,
                       M: float = 1.0) -> RadialGrid:
    """Uniform grid in tortoise coordinate r*, mapped back to physical r.

    Parameters:
        N: number of grid points
        rstar_min: minimum tortoise coordinate (should be < 0 for near-horizon)
        rstar_max: maximum tortoise coordinate
        M: black hole mass

    Returns:
        RadialGrid with uniform r* spacing.
    """
    if N < 3:
        raise ValueError(f"Need N >= 3 grid points, got {N}")
    if rstar_min >= rstar_max:
        raise ValueError(f"rstar_min={rstar_min} must be < rstar_max={rstar_max}")

    rstar = np.linspace(rstar_min, rstar_max, N)
    h = float(rstar[1] - rstar[0])
    r = r_from_tortoise(rstar, M=M)

    return RadialGrid(
        r=r, rstar=rstar, h=h, N=N, M=M,
        coord='tortoise', r_min=float(r[0]), r_max=float(r[-1]),
    )


def make_log_grid(N: int, r_min: float, r_max: float,
                  M: float = 1.0) -> RadialGrid:
    """Logarithmic grid in physical radius r.

    Good for capturing near-horizon structure when r_min is close to r_s.
    Grid points: r_i = r_min * (r_max/r_min)^{i/(N-1)}.

    Parameters:
        N: number of grid points
        r_min: minimum radius (must be > r_s = 2M for Schwarzschild)
        r_max: maximum radius
        M: black hole mass

    Returns:
        RadialGrid with logarithmic physical spacing.
    """
    if N < 3:
        raise ValueError(f"Need N >= 3 grid points, got {N}")
    r_s = 2.0 * M
    if M > 0 and r_min <= r_s:
        raise ValueError(f"r_min={r_min} must be > r_s={r_s}")
    if r_min >= r_max:
        raise ValueError(f"r_min={r_min} must be < r_max={r_max}")

    r = np.geomspace(r_min, r_max, N)
    rstar = tortoise_from_r(r, M=M) if M > 0 else r.copy()
    h = float(np.log(r_max / r_min) / (N - 1))  # log-spacing parameter

    return RadialGrid(
        r=r, rstar=rstar, h=h, N=N, M=M,
        coord='log', r_min=float(r[0]), r_max=float(r[-1]),
    )


def make_uniform_grid(N: int, r_min: float, r_max: float,
                      M: float = 0.0) -> RadialGrid:
    """Uniform grid in physical radius r.

    Primarily useful for flat space (M=0) or far from the horizon.

    Parameters:
        N: number of grid points
        r_min: minimum radius
        r_max: maximum radius
        M: black hole mass (default 0 for flat space)

    Returns:
        RadialGrid with uniform physical spacing.
    """
    if N < 3:
        raise ValueError(f"Need N >= 3 grid points, got {N}")
    if r_min >= r_max:
        raise ValueError(f"r_min={r_min} must be < r_max={r_max}")

    r = np.linspace(r_min, r_max, N)
    h = float(r[1] - r[0])
    if M > 0:
        rstar = tortoise_from_r(r, M=M)
    else:
        rstar = r.copy()

    return RadialGrid(
        r=r, rstar=rstar, h=h, N=N, M=M,
        coord='uniform', r_min=float(r[0]), r_max=float(r[-1]),
    )


# =============================================================================
# BOX OPERATOR DISCRETIZATION
# =============================================================================

def box_matrix(grid: RadialGrid, l: int = 0,
               A_func: Callable | None = None) -> NDArray:
    """Construct the matrix L representing the scalar Box operator.

    Discretizes:
        Box Phi_l = (1/r^2) d/dr [r^2 A(r) dPhi/dr] - l(l+1) A(r) Phi / r^2

    using second-order centered finite differences in the physical coordinate.
    The derivative term is treated in conservative (flux) form:

        (1/r_i^2) * [f_{i+1/2}(Phi_{i+1}-Phi_i)/dr - f_{i-1/2}(Phi_i-Phi_{i-1})/dr] / dr

    where f(r) = r^2 * A(r) and dr is the local grid spacing.

    Boundary conditions (Dirichlet):
        Phi(r_min) = 0  (regularity at horizon or inner boundary)
        Phi(r_max) = 0  (asymptotic flatness)

    These are enforced by excluding the boundary points from the matrix
    (interior points only, N-2 x N-2 system).

    Parameters:
        grid: RadialGrid instance
        l: angular momentum quantum number (l >= 0)
        A_func: optional lapse function A(r); if None, uses grid.A

    Returns:
        L: (N-2) x (N-2) matrix representing Box on interior points.
    """
    if l < 0:
        raise ValueError(f"l must be >= 0, got {l}")

    N = grid.N
    r = grid.r
    A = grid.A if A_func is None else np.asarray([A_func(ri) for ri in r])
    f = r**2 * A  # flux function

    # Grid spacings between consecutive points
    dr = np.diff(r)  # dr[i] = r[i+1] - r[i], shape (N-1,)

    # Interior points: indices 1 to N-2 (0-indexed)
    n_int = N - 2
    L = np.zeros((n_int, n_int))

    for i in range(n_int):
        # Map interior index i to global index ig = i + 1
        ig = i + 1

        # Half-point flux function values
        f_right = 0.5 * (f[ig] + f[ig + 1])  # f_{i+1/2}
        f_left = 0.5 * (f[ig - 1] + f[ig])    # f_{i-1/2}

        dr_right = dr[ig]      # r[ig+1] - r[ig]
        dr_left = dr[ig - 1]   # r[ig] - r[ig-1]
        dr_avg = 0.5 * (dr_right + dr_left)

        r2_inv = 1.0 / (r[ig]**2)

        # Radial derivative: (1/r^2) d/dr[f dPhi/dr]
        # = (1/r^2) [f_{i+1/2}(Phi_{i+1}-Phi_i)/dr_right - f_{i-1/2}(Phi_i-Phi_{i-1})/dr_left] / dr_avg
        coeff_right = r2_inv * f_right / (dr_right * dr_avg)
        coeff_left = r2_inv * f_left / (dr_left * dr_avg)
        coeff_center = -(coeff_right + coeff_left)

        # Angular term: -l(l+1) A(r) / r^2
        angular = -l * (l + 1) * A[ig] * r2_inv

        # Diagonal
        L[i, i] = coeff_center + angular

        # Off-diagonals (only if neighbor is an interior point)
        if i + 1 < n_int:
            L[i, i + 1] = coeff_right
        if i - 1 >= 0:
            L[i, i - 1] = coeff_left

    return L


def box_matrix_tortoise(grid: RadialGrid, l: int = 0) -> NDArray:
    """Box matrix using uniform tortoise-coordinate spacing.

    In tortoise coordinates, the radial part of Box for a static scalar is:
        Box Phi = A(r) [d^2 Phi/dr*^2 + (2/r) A(r) dPhi/dr*
                        + A'(r) dPhi/dr*] - l(l+1) A(r) Phi / r^2

    But it is cleaner to keep the conservative form and just use the
    non-uniform physical spacing induced by the tortoise grid.
    This function wraps box_matrix with the tortoise-generated grid.
    """
    return box_matrix(grid, l=l)


def box_sparse(grid: RadialGrid, l: int = 0) -> "sparse matrix":
    """Sparse tridiagonal version of box_matrix for large grids.

    Returns a scipy.sparse CSR matrix.
    """
    L_dense = box_matrix(grid, l=l)
    n = L_dense.shape[0]
    diag_main = np.diag(L_dense)
    diag_upper = np.diag(L_dense, k=1)
    diag_lower = np.diag(L_dense, k=-1)
    return sparse_diags(
        [diag_lower, diag_main, diag_upper],
        offsets=[-1, 0, 1],
        shape=(n, n),
        format='csr',
    )


# =============================================================================
# FLAT-SPACE BOX (ANALYTIC REFERENCE)
# =============================================================================

def box_flat_eigenvalues(N: int, r_min: float, r_max: float,
                         l: int = 0) -> NDArray:
    """Analytic eigenvalues of the flat-space radial Laplacian.

    For the 1D Laplacian d^2/dr^2 on [r_min, r_max] with Dirichlet BC,
    the eigenvalues are:
        lambda_k = -(k*pi/(r_max - r_min))^2,  k = 1, 2, ..., N-2

    This is the leading term for l=0. For l>0 the angular term shifts
    the eigenvalues but is not a simple correction on a finite grid.

    Returns sorted (most negative first) eigenvalues.
    """
    L = r_max - r_min
    k = np.arange(1, N - 1)
    evals = -(k * np.pi / L)**2
    return np.sort(evals)


# =============================================================================
# FORM FACTOR MATRIX FUNCTION
# =============================================================================

def _phi_allz(z: float) -> float:
    """Master function phi(z) for ALL real z (positive, negative, zero).

    phi(z) = int_0^1 exp[-alpha(1-alpha)*z] d_alpha

    For z >= 0: use the fast Dawson-based closed form.
    For z < 0: use the identity phi(-y) = exp(y/4) sqrt(pi/y) erf(sqrt(y)/2)
    for y = -z > 0. For large y, erf -> 1 and this simplifies to
    phi(-y) ~ 2*exp(y/4)/sqrt(y).
    """
    if abs(z) < 1e-14:
        return 1.0
    if z >= 0:
        return phi_fast(z)
    # z < 0: let y = -z > 0
    y = -z
    from scipy.special import erf
    sy = np.sqrt(y)
    # For moderate y (< ~2800), direct computation is fine
    if y < 2800:
        return float(np.exp(y / 4) * np.sqrt(np.pi / y) * erf(sy / 2))
    # For very large y, erf(sqrt(y)/2) -> 1, so phi(-y) ~ 2*exp(y/4)/sqrt(y)
    # But exp(y/4) overflows for y > ~2800. Return the log for composability.
    # Since the form factors involve phi/z^k, we handle overflow in the
    # form factor functions by using logarithmic arithmetic.
    return float('inf')  # Signals overflow; handled in form factor wrappers


def _log_phi_neg(y: float) -> float:
    """Return log(phi(-y)) for y > 0. Stable for all y.

    log(phi(-y)) = y/4 + 0.5*log(pi/y) + log(erf(sqrt(y)/2))
    For large y: log(erf) -> 0, giving y/4 - 0.5*log(y) + 0.5*log(pi).
    """
    from scipy.special import erf, log_ndtr
    sy = np.sqrt(y)
    # erf(x) = 2*Phi(sqrt(2)*x) - 1 where Phi is the normal CDF
    # For small y, compute directly; for large y, erf -> 1
    erf_val = erf(sy / 2)
    if erf_val > 0:
        log_erf = np.log(erf_val)
    else:
        log_erf = 0.0
    return y / 4.0 + 0.5 * np.log(np.pi / y) + log_erf


def _horner_eval(coeffs: NDArray, z: float) -> float:
    """Evaluate polynomial via Horner's method: sum_k coeffs[k] z^k."""
    result = coeffs[-1]
    for k in range(len(coeffs) - 2, -1, -1):
        result = result * z + coeffs[k]
    return float(result)


# Cache Taylor coefficient arrays at module level for speed
_TAYLOR_HC0 = get_taylor_coefficients('hC_scalar')
_TAYLOR_HCD = get_taylor_coefficients('hC_dirac')
_TAYLOR_HCV = get_taylor_coefficients('hC_vector')
_TAYLOR_HRD = get_taylor_coefficients('hR_dirac')
_TAYLOR_HRV = get_taylor_coefficients('hR_vector')
_TAYLOR_HR0_A = get_taylor_coefficients('hR_scalar_A')
_TAYLOR_HR0_B = get_taylor_coefficients('hR_scalar_B')
_TAYLOR_HR0_C = get_taylor_coefficients('hR_scalar_C')

# Threshold: for |z| > _TAYLOR_THRESH, use closed-form phi; below, use Taylor.
_TAYLOR_THRESH = 2.0


def _hC_scalar_allz(z: float) -> float:
    """hC_scalar(z) for all real z, using closed-form phi."""
    if abs(z) < _TAYLOR_THRESH:
        return _horner_eval(_TAYLOR_HC0, z)
    phi_val = _phi_allz(z)
    if not np.isfinite(phi_val):
        # Overflow for very large negative z. Use asymptotic:
        # hC_scalar ~ phi/(2z^2) for |z| >> 1
        # log(|hC_scalar|) ~ log_phi - 2*log|z| - log(2)
        y = -z
        log_abs = _log_phi_neg(y) - 2.0 * np.log(y) - np.log(2.0)
        return float(np.exp(log_abs))  # positive for z < 0
    return 1.0 / (12.0 * z) + (phi_val - 1.0) / (2.0 * z**2)


def _hC_dirac_allz(z: float) -> float:
    """hC_dirac(z) for all real z."""
    if abs(z) < _TAYLOR_THRESH:
        return _horner_eval(_TAYLOR_HCD, z)
    phi_val = _phi_allz(z)
    if not np.isfinite(phi_val):
        y = -z
        # Dominant: 2*phi/z^2 + phi/(2z) -> phi*(2/z^2 + 1/(2z))
        # For very large y: ~ phi * (2/z^2)
        log_abs = _log_phi_neg(y) - 2.0 * np.log(y) + np.log(2.0)
        return float(np.exp(log_abs))
    return (3.0 * phi_val - 1.0) / (6.0 * z) + 2.0 * (phi_val - 1.0) / z**2


def _hC_vector_allz(z: float) -> float:
    """hC_vector(z) for all real z."""
    if abs(z) < _TAYLOR_THRESH:
        return _horner_eval(_TAYLOR_HCV, z)
    phi_val = _phi_allz(z)
    if not np.isfinite(phi_val):
        y = -z
        # Dominant: phi/4 + phi/z + phi/z^2 ~ phi/4
        log_abs = _log_phi_neg(y) - np.log(4.0)
        return float(np.exp(log_abs))
    return phi_val / 4.0 + (6.0 * phi_val - 5.0) / (6.0 * z) + (phi_val - 1.0) / z**2


def _hR_scalar_allz(z: float, xi: float = 0.0) -> float:
    """hR_scalar(z, xi) for all real z."""
    if abs(z) < _TAYLOR_THRESH:
        result = 0.0
        z_pow = 1.0
        for k in range(len(_TAYLOR_HR0_A)):
            result += (_TAYLOR_HR0_A[k] + xi * _TAYLOR_HR0_B[k]
                       + xi**2 * _TAYLOR_HR0_C[k]) * z_pow
            z_pow *= z
        return float(result)
    phi_val = _phi_allz(z)
    if not np.isfinite(phi_val):
        y = -z
        # Dominant at large y: f_u = phi/2 (xi^2 term) + f_ru ~ -phi/4 (xi term)
        # + f_r ~ phi/32 + ... For simplicity, use the xi^2 * phi/2 dominant term
        log_phi = _log_phi_neg(y)
        # phi/32 + xi^2 * phi/2 - xi*phi/4 = phi*(1/32 + xi^2/2 - xi/4)
        c = 1.0 / 32.0 + xi**2 / 2.0 - xi / 4.0
        if c > 0:
            return float(np.exp(log_phi + np.log(c)))
        elif c < 0:
            return -float(np.exp(log_phi + np.log(-c)))
        else:
            return 0.0
    f_ric = 1.0 / (6.0 * z) + (phi_val - 1.0) / z**2
    f_r = phi_val / 32.0 + phi_val / (8.0 * z) - 7.0 / (48.0 * z) - (phi_val - 1.0) / (8.0 * z**2)
    f_ru = -phi_val / 4.0 - (phi_val - 1.0) / (2.0 * z)
    f_u = phi_val / 2.0
    return f_ric / 3.0 + f_r + xi * f_ru + xi**2 * f_u


def _hR_dirac_allz(z: float) -> float:
    """hR_dirac(z) for all real z."""
    if abs(z) < _TAYLOR_THRESH:
        return _horner_eval(_TAYLOR_HRD, z)
    phi_val = _phi_allz(z)
    if not np.isfinite(phi_val):
        y = -z
        # Dominant: 5*phi/(6z^2) + phi/(12z)
        log_abs = _log_phi_neg(y) - 2.0 * np.log(y) + np.log(5.0 / 6.0)
        return float(np.exp(log_abs))
    return (3.0 * phi_val + 2.0) / (36.0 * z) + 5.0 * (phi_val - 1.0) / (6.0 * z**2)


def _hR_vector_allz(z: float) -> float:
    """hR_vector(z) for all real z."""
    if abs(z) < _TAYLOR_THRESH:
        return _horner_eval(_TAYLOR_HRV, z)
    phi_val = _phi_allz(z)
    if not np.isfinite(phi_val):
        y = -z
        # Dominant: -phi/48 + 5*phi/(12z^2) - phi/(12z) ~ -phi/48
        log_abs = _log_phi_neg(y) - np.log(48.0)
        return -float(np.exp(log_abs))  # negative
    return -phi_val / 48.0 + (11.0 - 6.0 * phi_val) / (72.0 * z) + 5.0 * (phi_val - 1.0) / (12.0 * z**2)


def _alpha_C_allz(z: float) -> float:
    """SM-summed Weyl form factor alpha_C(z) for all real z.

    alpha_C(z) = N_s * hC_scalar(z) + N_D * hC_dirac(z) + N_v * hC_vector(z)
    """
    return (_N_S * _hC_scalar_allz(z)
            + _N_D * _hC_dirac_allz(z)
            + _N_V * _hC_vector_allz(z))


_ALPHA_C_0 = 13.0 / 120.0  # alpha_C(0) for SM


def _alpha_R_allz(z: float, xi: float = 0.0) -> float:
    """SM-summed R^2 form factor alpha_R(z, xi) for all real z."""
    return (_N_S * _hR_scalar_allz(z, xi=xi)
            + _N_D * _hR_dirac_allz(z)
            + _N_V * _hR_vector_allz(z))


def _F1_hat_scalar(z: float) -> float:
    """Normalized spin-2 form factor F_hat_1(z) = alpha_C(z) / alpha_C(0).

    Uses closed-form evaluation valid for all real z. For z >= 0 this
    reproduces the standard form_factors.py results. For z < 0 (spatial
    Box eigenvalues) it uses the analytic continuation phi(-y) with erf.

    For the SCT propagator:
        Pi_TT(z) = 1 + (13/60) * z * F_hat_1(z)
    """
    if abs(z) < 1e-30:
        return 1.0
    val = _alpha_C_allz(z) / _ALPHA_C_0
    if not np.isfinite(val):
        return float('inf') if z < 0 else float(F1_total(z) / F1_total(0.0))
    return val


def _F2_hat_scalar(z: float, xi: float = 0.0) -> float:
    """Normalized scalar form factor F_hat_2(z) = alpha_R(z) / alpha_R(0)."""
    alpha_R_0 = _alpha_R_allz(0.0, xi=xi)
    if abs(alpha_R_0) < 1e-40:
        return 1.0
    if abs(z) < 1e-30:
        return 1.0
    val = _alpha_R_allz(z, xi=xi) / alpha_R_0
    if not np.isfinite(val):
        return float('inf') if z < 0 else float(F2_total(z, xi=xi) / F2_total(0.0, xi=xi))
    return val


def _apply_matrix_function(L: NDArray, func: Callable,
                           z_max: float = 0.0) -> NDArray:
    """Apply a scalar function to a matrix via eigendecomposition.

    For eigenvalues z_j with |z_j| > z_max (when z_max > 0), the function
    value is clamped to the value at +/- z_max. This spectral truncation
    prevents overflow from unphysical UV grid modes while preserving the
    IR physics.

    Parameters:
        L: (n, n) real matrix
        func: scalar function f(z) -> float
        z_max: if > 0, clamp eigenvalue magnitudes to this value

    Returns:
        f(L) as an (n, n) real matrix
    """
    n = L.shape[0]
    if n == 0:
        return np.empty((0, 0))

    eigenvalues, V = np.linalg.eig(L)

    # Check condition number
    cond = np.linalg.cond(V)
    if cond > 1e14:
        import warnings
        warnings.warn(
            f"Eigenvector matrix poorly conditioned (cond={cond:.2e}). "
            f"Results may be inaccurate.",
            stacklevel=2,
        )

    # Apply function, with optional spectral truncation
    f_diag = np.empty(n, dtype=float)
    for j in range(n):
        zj = float(np.real(eigenvalues[j]))
        if z_max > 0 and abs(zj) > z_max:
            zj = np.sign(zj) * z_max
        val = func(zj)
        f_diag[j] = val if np.isfinite(val) else func(np.sign(zj) * min(abs(zj), 2000.0))

    V_inv = np.linalg.inv(V)
    return np.real(V @ np.diag(f_diag) @ V_inv)


def form_factor_Fhat1(L: NDArray, Lambda2: float,
                      z_max: float = 0.0) -> NDArray:
    """Apply the SCT form factor F_hat_1(Box/Lambda^2) as a matrix function.

    Given the matrix L representing Box, computes F_hat_1(L/Lambda^2) via
    eigendecomposition:
        L = V diag(lambda_j) V^{-1}
        F_hat_1(L/Lambda^2) = V diag(F_hat_1(lambda_j/Lambda^2)) V^{-1}

    Parameters:
        L: (n, n) matrix representing the Box operator
        Lambda2: Lambda^2 (SCT scale squared)
        z_max: spectral cutoff. If > 0, eigenvalues with |lambda/Lambda^2|
               beyond z_max are clamped. This prevents overflow from
               unphysical UV grid modes. Recommended: z_max ~ 2000.
               If 0 (default), no clamping is applied.

    Returns:
        (n, n) matrix F_hat_1(L/Lambda^2)
    """
    if Lambda2 <= 0:
        raise ValueError(f"Lambda2 must be positive, got {Lambda2}")
    Z = L / Lambda2
    return _apply_matrix_function(Z, _F1_hat_scalar, z_max=z_max)


def form_factor_Fhat1_funm(L: NDArray, Lambda2: float,
                            z_max: float = 2000.0) -> NDArray:
    """F_hat_1(L/Lambda^2) via scipy.linalg.funm (Schur decomposition).

    Alternative to eigendecomposition; uses Schur-Parlett algorithm which
    is more numerically stable for matrices with clustered eigenvalues.
    """
    if Lambda2 <= 0:
        raise ValueError(f"Lambda2 must be positive, got {Lambda2}")

    Z = L / Lambda2

    def _fhat1_safe(z_arr):
        """funm passes 1D arrays; return element-wise evaluation."""
        z_arr = np.atleast_1d(z_arr)
        out = np.empty_like(z_arr, dtype=float)
        for i, z in enumerate(z_arr):
            zr = float(np.real(z))
            if z_max > 0 and abs(zr) > z_max:
                zr = np.sign(zr) * z_max
            val = _F1_hat_scalar(zr)
            out[i] = val if np.isfinite(val) else _F1_hat_scalar(
                np.sign(zr) * min(abs(zr), 2000.0))
        return out

    result = la.funm(Z, _fhat1_safe)
    return np.real(result)


def form_factor_Fhat2(L: NDArray, Lambda2: float, xi: float = 0.0,
                      z_max: float = 0.0) -> NDArray:
    """Apply the SCT form factor F_hat_2(Box/Lambda^2) as a matrix function.

    Same approach as form_factor_Fhat1 but for the scalar (R^2) sector.
    """
    if Lambda2 <= 0:
        raise ValueError(f"Lambda2 must be positive, got {Lambda2}")
    Z = L / Lambda2
    return _apply_matrix_function(Z, lambda z: _F2_hat_scalar(z, xi=xi), z_max=z_max)


# =============================================================================
# PROPAGATOR MATRIX: Pi_TT(Box/Lambda^2)
# =============================================================================

ALPHA_C = 13.0 / 120.0  # SM Weyl coefficient
LOCAL_C2 = 2.0 * ALPHA_C  # = 13/60


def Pi_TT_matrix(L: NDArray, Lambda2: float,
                  z_max: float = 2000.0) -> NDArray:
    """Spin-2 propagator denominator Pi_TT(Box/Lambda^2) as a matrix.

    Pi_TT = I + (13/60) * (Box/Lambda^2) * F_hat_1(Box/Lambda^2)

    Parameters:
        L: Box operator matrix
        Lambda2: Lambda^2
        z_max: spectral cutoff for the form factor (default: 2000)

    Returns:
        Pi_TT matrix (same shape as L).
    """
    n = L.shape[0]
    Z = L / Lambda2
    Fhat1 = form_factor_Fhat1(L, Lambda2, z_max=z_max)
    return np.eye(n) + LOCAL_C2 * Z @ Fhat1


def Pi_scalar_matrix(L: NDArray, Lambda2: float, xi: float = 0.0,
                      z_max: float = 2000.0) -> NDArray:
    """Spin-0 propagator denominator Pi_s(Box/Lambda^2) as a matrix.

    Pi_s = I + 6(xi-1/6)^2 * (Box/Lambda^2) * F_hat_2(Box/Lambda^2, xi)

    At conformal coupling xi=1/6, Pi_s = I (scalar mode decouples).
    """
    n = L.shape[0]
    coeff = 6.0 * (xi - 1.0 / 6.0)**2
    if abs(coeff) < 1e-40:
        return np.eye(n)
    Z = L / Lambda2
    Fhat2 = form_factor_Fhat2(L, Lambda2, xi=xi, z_max=z_max)
    return np.eye(n) + coeff * Z @ Fhat2


# =============================================================================
# EIGENVALUE ANALYSIS
# =============================================================================

def box_eigenvalues(grid: RadialGrid, l: int = 0) -> NDArray:
    """Compute eigenvalues of the Box matrix, sorted by real part.

    Returns real-part-sorted eigenvalues. For a well-posed spatial operator,
    all eigenvalues should be real and negative.
    """
    L = box_matrix(grid, l=l)
    evals = np.linalg.eigvals(L)
    # Sort by real part (most negative first)
    idx = np.argsort(np.real(evals))
    return evals[idx]


def check_eigenvalue_reality(evals: NDArray, tol: float = 1e-10) -> bool:
    """Verify that all eigenvalues are real (imaginary parts < tol)."""
    return bool(np.all(np.abs(np.imag(evals)) < tol))


def check_eigenvalue_negativity(evals: NDArray, tol: float = 1e-10) -> bool:
    """Verify that all eigenvalues have negative real part.

    The spatial Box operator should have only non-positive eigenvalues
    (negative for interior modes, zero for constant mode which is
    excluded by Dirichlet BC).
    """
    return bool(np.all(np.real(evals) < tol))


# =============================================================================
# VERIFICATION UTILITIES
# =============================================================================

def verify_flat_space(N: int = 100, r_min: float = 1.0, r_max: float = 50.0,
                      l: int = 0) -> dict:
    """Verify Box discretization against analytic flat-space Laplacian.

    For M=0, A(r)=1, the Box operator reduces to:
        Box Phi = (1/r^2) d/dr[r^2 dPhi/dr] - l(l+1)Phi/r^2
                = d^2Phi/dr^2 + (2/r) dPhi/dr - l(l+1)Phi/r^2

    For l=0, the substitution u = r*Phi gives Box Phi = (1/r) d^2u/dr^2,
    so the eigenvalues of the full operator converge to -(k*pi/L)^2 for
    large r (where the 2/r term becomes negligible).

    Returns dict with comparison data.
    """
    grid = make_uniform_grid(N, r_min, r_max, M=0.0)
    L = box_matrix(grid, l=l)
    evals_num = np.sort(np.real(np.linalg.eigvals(L)))

    # Analytic eigenvalues for d^2/dr^2 (l=0, large-r approximation)
    if l == 0:
        evals_ref = box_flat_eigenvalues(N, r_min, r_max, l=0)
        # The numerical eigenvalues won't match exactly because of the
        # 2/r correction, but the lowest modes should be close for large r_min/r_max
        n_compare = min(5, len(evals_num))
        relative_errors = np.abs((evals_num[:n_compare] - evals_ref[:n_compare])
                                 / evals_ref[:n_compare])
    else:
        evals_ref = None
        relative_errors = None

    results = {
        'N': N,
        'l': l,
        'r_min': r_min,
        'r_max': r_max,
        'n_eigenvalues': len(evals_num),
        'all_real': check_eigenvalue_reality(np.linalg.eigvals(L)),
        'all_negative': check_eigenvalue_negativity(np.linalg.eigvals(L)),
        'evals_lowest_5': evals_num[:5].tolist(),
        'relative_errors_lowest_5': relative_errors.tolist() if relative_errors is not None else None,
    }
    return results


def verify_local_limit(grid: RadialGrid, Lambda2: float, l: int = 0,
                       tol: float = 0.01) -> dict:
    """Verify that F_hat_1(Box/Lambda^2) -> I in the local limit Lambda -> inf.

    When Lambda^2 >> |eigenvalues of Box|, all z_j = lambda_j/Lambda^2 -> 0,
    so F_hat_1(z_j) -> 1 and the form factor matrix -> identity.
    """
    L = box_matrix(grid, l=l)
    Fhat1 = form_factor_Fhat1(L, Lambda2)
    n = L.shape[0]
    deviation = np.linalg.norm(Fhat1 - np.eye(n)) / n

    return {
        'Lambda2': Lambda2,
        'matrix_size': n,
        'deviation_from_identity': float(deviation),
        'passes': deviation < tol,
    }


def verify_symmetry(grid: RadialGrid, l: int = 0, tol: float = 1e-10) -> dict:
    """Check whether the Box matrix is symmetric.

    For uniform grids in physical coordinate, the Box matrix should be
    symmetric (self-adjoint operator with uniform measure). For non-uniform
    grids, it may not be exactly symmetric but should be close.
    """
    L = box_matrix(grid, l=l)
    asym = np.linalg.norm(L - L.T) / (np.linalg.norm(L) + 1e-300)

    return {
        'coord': grid.coord,
        'asymmetry': float(asym),
        'is_symmetric': asym < tol,
    }


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo_eigenvalue_spectrum(M: float = 1.0, N: int = 200,
                             l_values: tuple[int, ...] = (0, 1, 2)) -> dict:
    """Compute and return eigenvalue spectra for Schwarzschild Box.

    Uses tortoise-coordinate grid from near-horizon to far field.
    """
    r_s = 2.0 * M
    rstar_min = -20.0 * r_s  # deep near-horizon
    rstar_max = 50.0 * r_s   # far field

    grid = make_tortoise_grid(N, rstar_min, rstar_max, M=M)

    results = {'M': M, 'N': N, 'r_s': r_s}
    for l in l_values:
        evals = box_eigenvalues(grid, l=l)
        results[f'l={l}'] = {
            'n_eigenvalues': len(evals),
            'all_real': check_eigenvalue_reality(evals),
            'all_negative': check_eigenvalue_negativity(evals),
            'lambda_min': float(np.min(np.real(evals))),
            'lambda_max': float(np.max(np.real(evals))),
            'lambda_5_lowest': np.real(evals[:5]).tolist(),
        }
    return results


def demo_form_factor(M: float = 1.0, N: int = 100,
                     Lambda_M_values: tuple[float, ...] = (1.0, 10.0, 100.0),
                     l: int = 0) -> dict:
    """Compute F_hat_1(Box/Lambda^2) for Schwarzschild at various Lambda*M.

    Uses a logarithmic grid starting at r_min = 2.5M (safely outside the
    horizon) to avoid the extreme eigenvalue spread from near-horizon
    tortoise grid compression. The log grid keeps the eigenvalue range
    manageable for the matrix function evaluation.

    Parameters:
        M: black hole mass
        N: grid points
        Lambda_M_values: dimensionless Lambda*M values to test
        l: angular momentum

    Returns dict with eigenvalues of Pi_TT and form factor deviation data.
    """
    r_s = 2.0 * M
    # Log grid: r from 2.5M to 50M. Avoids extreme near-horizon compression.
    grid = make_log_grid(N, r_min=2.5 * M, r_max=50.0 * M, M=M)
    L = box_matrix(grid, l=l)
    n = L.shape[0]

    box_evals = np.sort(np.real(np.linalg.eigvals(L)))

    results = {
        'M': M, 'N': N, 'l': l,
        'box_eigenvalue_range': [float(box_evals[0]), float(box_evals[-1])],
    }

    # Spectral cutoff: form factor evaluation is clamped at |z| = z_max
    # to avoid overflow from UV grid modes. Physical modes have
    # |z| = |lambda|/Lambda^2 of order 1; the UV grid modes have
    # |z| >> 1 and their contribution is a discretization artifact.
    z_max = 2000.0

    for lm in Lambda_M_values:
        Lambda = lm / M
        Lambda2 = Lambda**2

        Fhat1 = form_factor_Fhat1(L, Lambda2, z_max=z_max)
        Pi_TT = Pi_TT_matrix(L, Lambda2, z_max=z_max)

        # Eigenvalues of Pi_TT
        pi_evals = np.sort(np.real(np.linalg.eigvals(Pi_TT)))

        # Deviation of Fhat1 from identity
        dev = np.linalg.norm(Fhat1 - np.eye(n)) / n

        # Count physical modes (|z| < z_max) vs UV modes
        z_all = np.abs(box_evals) / Lambda2
        n_physical = int(np.sum(z_all < z_max))

        results[f'Lambda_M={lm}'] = {
            'Lambda': float(Lambda),
            'Lambda2': float(Lambda2),
            'Fhat1_deviation_from_I': float(dev),
            'Pi_TT_eigenvalue_range': [float(pi_evals[0]), float(pi_evals[-1])],
            'Pi_TT_all_positive': bool(np.all(pi_evals > 0)),
            'Pi_TT_min_eigenvalue': float(pi_evals[0]),
            'n_physical_modes': n_physical,
            'n_total_modes': n,
            'z_max_cutoff': z_max,
        }

    return results


# =============================================================================
# SELF-TESTS
# =============================================================================

def run_self_tests() -> list[dict]:
    """Run all built-in verification tests. Returns list of test results."""
    tests = []

    # Test 1: Grid construction
    # r*=40 with M=1 maps to r ~ 34.4, not r > 40. Check only that
    # r > r_s = 2 at the inner boundary and that r is monotonically increasing.
    grid = make_tortoise_grid(50, -20.0, 40.0, M=1.0)
    tests.append({
        'name': 'tortoise_grid_construction',
        'passed': (grid.N == 50 and grid.r[0] > 2.0
                   and np.all(np.diff(grid.r) > 0)),
        'detail': f'N={grid.N}, r_min={grid.r_min:.4f}, r_max={grid.r_max:.4f}',
    })

    # Test 2: Flat-space Box eigenvalues (all real, all negative)
    grid_flat = make_uniform_grid(60, 1.0, 50.0, M=0.0)
    evals_flat = box_eigenvalues(grid_flat, l=0)
    tests.append({
        'name': 'flat_box_eigenvalues_real_negative',
        'passed': (check_eigenvalue_reality(evals_flat)
                   and check_eigenvalue_negativity(evals_flat)),
        'detail': f'range=[{float(np.real(evals_flat[0])):.4f}, {float(np.real(evals_flat[-1])):.6f}]',
    })

    # Test 3: Convergence of flat-space Box eigenvalues with grid refinement.
    # Compare the LEAST negative eigenvalue (longest wavelength / fundamental
    # mode), which should converge as N increases. The most negative eigenvalue
    # grows as ~(N/L)^2 and is resolution-dependent.
    evals_N50 = np.sort(np.real(box_eigenvalues(
        make_uniform_grid(50, 5.0, 100.0, M=0.0), l=0)))
    evals_N100 = np.sort(np.real(box_eigenvalues(
        make_uniform_grid(100, 5.0, 100.0, M=0.0), l=0)))
    # Least negative = last element of sorted (ascending) array
    lam_fund_50 = evals_N50[-1]
    lam_fund_100 = evals_N100[-1]
    convergence_err = abs((lam_fund_50 - lam_fund_100) / lam_fund_100)
    tests.append({
        'name': 'flat_box_eigenvalue_convergence',
        'passed': convergence_err < 0.05,
        'detail': f'lambda_fund(N=50)={lam_fund_50:.6f}, lambda_fund(N=100)={lam_fund_100:.6f}, rel_diff={convergence_err:.4e}',
    })

    # Test 4: Schwarzschild eigenvalues (real and negative)
    grid_sch = make_tortoise_grid(80, -15.0, 40.0, M=1.0)
    evals_sch = box_eigenvalues(grid_sch, l=0)
    tests.append({
        'name': 'schwarzschild_box_eigenvalues_real_negative',
        'passed': (check_eigenvalue_reality(evals_sch, tol=1e-8)
                   and check_eigenvalue_negativity(evals_sch, tol=1e-8)),
        'detail': f'range=[{float(np.real(evals_sch[0])):.4f}, {float(np.real(evals_sch[-1])):.6f}]',
    })

    # Test 5: Local limit (Fhat1 -> I for large Lambda)
    grid_local = make_uniform_grid(30, 3.0, 20.0, M=0.0)
    result_local = verify_local_limit(grid_local, Lambda2=1e6, l=0, tol=0.01)
    tests.append({
        'name': 'local_limit_Fhat1_identity',
        'passed': result_local['passes'],
        'detail': f'deviation={result_local["deviation_from_identity"]:.4e}',
    })

    # Test 6: Schwarzschild eigenvalues shift with l
    evals_l0 = box_eigenvalues(grid_sch, l=0)
    evals_l2 = box_eigenvalues(grid_sch, l=2)
    # l=2 should have more negative eigenvalues (angular term subtracts)
    tests.append({
        'name': 'angular_momentum_eigenvalue_shift',
        'passed': float(np.real(evals_l2[0])) < float(np.real(evals_l0[0])),
        'detail': f'l=0 min={float(np.real(evals_l0[0])):.4f}, l=2 min={float(np.real(evals_l2[0])):.4f}',
    })

    # Test 7: Self-adjointness of flat Box in weighted inner product.
    # The operator (1/r^2) d/dr[r^2 dPhi/dr] is self-adjoint under the
    # inner product <f,g> = integral f*g r^2 dr. On the discrete grid,
    # this means W @ L should be approximately symmetric, where
    # W = diag(r_i^2 * dr_i) is the weight matrix.
    grid_m0 = make_uniform_grid(50, 3.0, 30.0, M=0.0)
    L_m0 = box_matrix(grid_m0, l=0)
    r_int = grid_m0.r[1:-1]
    dr_m0 = grid_m0.r[2] - grid_m0.r[1]  # uniform spacing
    W = np.diag(r_int**2 * dr_m0)
    WL = W @ L_m0
    asym_weighted = np.linalg.norm(WL - WL.T) / (np.linalg.norm(WL) + 1e-300)
    tests.append({
        'name': 'flat_box_weighted_self_adjoint',
        'passed': asym_weighted < 1e-10,
        'detail': f'weighted_asymmetry={asym_weighted:.4e}',
    })

    # Test 8: Tortoise inversion roundtrip
    r_test = np.array([2.5, 3.0, 5.0, 10.0, 50.0, 200.0])
    rstar_test = tortoise_from_r(r_test, M=1.0)
    r_recovered = r_from_tortoise(rstar_test, M=1.0)
    roundtrip_err = np.max(np.abs(r_recovered - r_test) / r_test)
    tests.append({
        'name': 'tortoise_inversion_roundtrip',
        'passed': roundtrip_err < 1e-12,
        'detail': f'max_rel_error={roundtrip_err:.4e}',
    })

    # Test 9: Pi_TT matrix at large Lambda -> I
    grid_pi = make_uniform_grid(20, 3.0, 15.0, M=0.0)
    L_pi = box_matrix(grid_pi, l=0)
    Pi = Pi_TT_matrix(L_pi, Lambda2=1e8)
    n_pi = L_pi.shape[0]
    pi_dev = np.linalg.norm(Pi - np.eye(n_pi)) / n_pi
    tests.append({
        'name': 'Pi_TT_local_limit_identity',
        'passed': pi_dev < 0.01,
        'detail': f'deviation={pi_dev:.4e}',
    })

    # Test 10: Form factor at Lambda*M=1 produces finite non-identity result
    # With spectral cutoff, the form factor matrix is well-behaved
    grid_ff = make_tortoise_grid(40, -8.0, 20.0, M=1.0)
    L_ff = box_matrix(grid_ff, l=0)
    Fhat1 = form_factor_Fhat1(L_ff, Lambda2=1.0, z_max=2000.0)
    ff_dev = np.linalg.norm(Fhat1 - np.eye(L_ff.shape[0])) / L_ff.shape[0]
    tests.append({
        'name': 'form_factor_finite_at_LambdaM_1',
        'passed': (np.all(np.isfinite(Fhat1)) and ff_dev > 1e-10),
        'detail': f'deviation_from_I={ff_dev:.4e}, all_finite={np.all(np.isfinite(Fhat1))}',
    })

    # Test 11: Consistency of eigendecomposition vs funm methods
    grid_comp = make_uniform_grid(20, 3.0, 15.0, M=0.0)
    L_comp = box_matrix(grid_comp, l=0)
    Fhat1_eig = form_factor_Fhat1(L_comp, Lambda2=100.0, z_max=2000.0)
    Fhat1_funm = form_factor_Fhat1_funm(L_comp, Lambda2=100.0, z_max=2000.0)
    method_diff = np.linalg.norm(Fhat1_eig - Fhat1_funm) / (np.linalg.norm(Fhat1_eig) + 1e-300)
    tests.append({
        'name': 'eigendecomp_vs_funm_consistency',
        'passed': method_diff < 1e-8,
        'detail': f'relative_diff={method_diff:.4e}',
    })

    # Test 12: F_hat_1(z) at z=0 is exactly 1 (normalization check)
    f1_at_zero = _F1_hat_scalar(0.0)
    tests.append({
        'name': 'Fhat1_normalization_at_zero',
        'passed': abs(f1_at_zero - 1.0) < 1e-14,
        'detail': f'F_hat_1(0) = {f1_at_zero}',
    })

    # Test 13: F_hat_1 at positive z agrees with sct_tools.form_factors
    z_test = 5.0
    from sct_tools.form_factors import F1_total as _F1
    fhat_here = _F1_hat_scalar(z_test)
    fhat_ref = float(_F1(z_test) / _F1(0.0))
    ref_err = abs(fhat_here - fhat_ref) / abs(fhat_ref)
    tests.append({
        'name': 'Fhat1_positive_z_crosscheck',
        'passed': ref_err < 1e-10,
        'detail': f'z={z_test}, rel_err={ref_err:.4e}',
    })

    return tests


# =============================================================================
# SERIALIZATION
# =============================================================================

def save_results(results: dict, filename: str) -> Path:
    """Save results to JSON in the results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(results, indent=2, default=str), encoding='utf-8')
    return path


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Box operator on Schwarzschild — discretization and form factors')
    parser.add_argument('--test', action='store_true', help='Run self-tests')
    parser.add_argument('--demo', action='store_true', help='Run demonstrations')
    parser.add_argument('--N', type=int, default=100, help='Grid points')
    parser.add_argument('--M', type=float, default=1.0, help='Black hole mass')
    args = parser.parse_args()

    if args.test:
        print("=" * 70)
        print("BOX SCHWARZSCHILD — SELF-TESTS")
        print("=" * 70)
        tests = run_self_tests()
        n_pass = sum(1 for t in tests if t['passed'])
        for t in tests:
            status = "PASS" if t['passed'] else "FAIL"
            print(f"  [{status}] {t['name']}: {t['detail']}")
        print(f"\n  {n_pass}/{len(tests)} tests passed.")
        save_results({'tests': tests}, 'self_test_results.json')
        return 0 if n_pass == len(tests) else 1

    if args.demo:
        print("=" * 70)
        print("BOX SCHWARZSCHILD — DEMONSTRATIONS")
        print("=" * 70)

        print("\n--- Eigenvalue spectrum ---")
        spec = demo_eigenvalue_spectrum(M=args.M, N=args.N)
        for l in (0, 1, 2):
            key = f'l={l}'
            if key in spec:
                d = spec[key]
                print(f"  l={l}: range=[{d['lambda_min']:.4f}, {d['lambda_max']:.6f}], "
                      f"real={d['all_real']}, negative={d['all_negative']}")
        save_results(spec, 'eigenvalue_spectrum.json')

        print("\n--- Form factor F_hat_1(Box/Lambda^2) ---")
        ff = demo_form_factor(M=args.M, N=args.N)
        print(f"  Box eigenvalue range: {ff['box_eigenvalue_range']}")
        for lm in (1.0, 10.0, 100.0):
            key = f'Lambda_M={lm}'
            if key in ff:
                d = ff[key]
                print(f"  Lambda*M={lm}: Fhat1 dev from I = {d['Fhat1_deviation_from_I']:.4e}, "
                      f"Pi_TT range = {d['Pi_TT_eigenvalue_range']}, "
                      f"Pi_TT>0: {d['Pi_TT_all_positive']}, "
                      f"phys modes = {d['n_physical_modes']}/{d['n_total_modes']}")
        save_results(ff, 'form_factor_demo.json')

        return 0

    # Default: run both
    parser.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())
