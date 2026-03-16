"""
SCT Theory — Heat kernel form factors h_C and h_R for spins 0, 1/2, 1.

Provides both fast numpy evaluation and high-precision mpmath evaluation.
All functions implement the BV (Barvinsky-Vilkovisky) formalism results.

Master function:
    phi(x) = int_0^1 exp[-alpha(1-alpha)*x] dalpha

CZ form factors (Codello-Zanusso basis):
    f_Ric(x), f_R(x), f_RU(x), f_U(x), f_Omega(x)

Weyl basis form factors:
    h_C^(s)(x) — coefficient of C_{munurhosigma}^2  (Weyl squared)
    h_R^(s)(x) — coefficient of R^2  (scalar curvature squared)

where s = 0 (scalar), 1/2 (Dirac), 1 (vector).

Key results verified in NT-1, NT-1b:
    h_C^(0)(0) = 1/120,   h_R^(0)(0;xi) = (1/2)(xi-1/6)^2
    h_C^(1/2)(0) = -1/20,  h_R^(1/2)(0) = 0  (conformal invariance)
    h_C^(1)(0) = 1/10,     h_R^(1)(0) = 0    (conformal invariance, Phase 2)
"""

import math

import numpy as np
from scipy.integrate import quad
from scipy.special import dawsn

__all__ = [
    # Master function
    "phi", "phi_closed", "phi_fast", "phi_vec", "phi_mp",
    # CZ basis form factors
    "f_Ric", "f_R", "f_RU", "f_U", "f_Omega",
    # Scalar (spin-0)
    "hC_scalar", "hR_scalar", "hC_scalar_fast", "hR_scalar_fast",
    "scan_hC_scalar", "scan_hR_scalar",
    "hC_scalar_mp", "hR_scalar_mp", "hC_scalar_taylor", "hR_scalar_taylor",
    # Dirac (spin-1/2)
    "hC_dirac", "hR_dirac", "hC_dirac_fast", "hR_dirac_fast",
    "scan_hC_dirac", "scan_hR_dirac",
    "hC_dirac_mp", "hR_dirac_mp",
    # Vector (spin-1)
    "hC_vector", "hR_vector", "hC_vector_fast", "hR_vector_fast",
    "scan_hC_vector", "scan_hR_vector",
    "hC_vector_mp", "hR_vector_mp",
    # Combined SM
    "F1_total", "F2_total", "F1_spectral", "F2_spectral",
    "alpha_C_SM", "alpha_R_SM", "c1_c2_ratio_SM", "scalar_mode_mass_SM",
    "uv_asymptotic_F1_total",
    # Derivatives
    "dphi_dx", "dphi_dx_fast",
    "dhC_scalar_dx", "dhC_dirac_dx", "dhR_dirac_dx", "dhR_scalar_dx",
    "dhC_vector_dx", "dhR_vector_dx",
    # Utilities
    "get_taylor_coefficients", "asymptotic_expansion",
]

# =============================================================================
# MASTER FUNCTION phi(x) — numpy (fast, ~15 digits)
# =============================================================================

def phi(x):
    """Master function phi(x) = int_0^1 exp[-alpha(1-alpha)*x] dalpha.

    Properties:
        phi(0) = 1
        phi(x) ~ sqrt(pi/x) * exp(-x/4) * erfi(sqrt(x)/2) for x > 0
        phi(x) ~ 2/x for x -> infinity
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"phi: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"phi: requires x >= 0, got {x}")
    if abs(x) < 1e-12:
        return 1.0
    result, _ = quad(lambda a: np.exp(-a * (1 - a) * x), 0, 1)
    return result


def phi_closed(x):
    """Closed form: phi(x) = e^{-x/4} sqrt(pi/x) erfi(sqrt(x)/2)."""
    from scipy.special import erfi
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"phi_closed: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"phi_closed: requires x >= 0, got {x}")
    if abs(x) < 1e-12:
        return 1.0
    if x > 2800:
        # erfi overflows before exp(-x/4) underflows at x ~ 2838.
        # Fall back to Dawson-based phi_fast which is numerically stable.
        return phi_fast(x)
    return np.exp(-x / 4) * np.sqrt(np.pi / x) * erfi(np.sqrt(x) / 2)


# =============================================================================
# VECTORIZED FAST VERSIONS (closed-form via Dawson function)
# =============================================================================
# These use the exact identity: phi(x) = 2 * dawsn(sqrt(x)/2) / sqrt(x)
# where dawsn is the Dawson function F(z) = exp(-z^2) * int_0^z exp(t^2) dt.
#
# Why Dawson instead of erfi?
#   phi(x) = e^{-x/4} * sqrt(pi/x) * erfi(sqrt(x)/2)
# For large x, erfi overflows and exp underflows separately → NaN.
# The Dawson form avoids this because dawsn(z) is bounded for all z.
#
# Accuracy: 15 digits (float64 limit). Speed: ~1000x faster than quad loop.


def phi_fast(x):
    """Fast scalar phi(x) via Dawson function. Stable for all x >= 0.

    Identity: phi(x) = 2 * dawsn(sqrt(x)/2) / sqrt(x)
    For arrays, use phi_vec().
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"phi_fast: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"phi_fast: requires x >= 0, got {x}. Use phi() for negative x.")
    if abs(x) < 1e-12:
        return 1.0
    sx = np.sqrt(x)
    return 2.0 * float(dawsn(sx / 2.0)) / sx


def phi_vec(x_array):
    """Vectorized phi(x) over numpy array. Stable for all x >= 0.

    ~1000x faster than [phi(x) for x in x_array].
    """
    x = np.asarray(x_array, dtype=float)
    if np.any(~np.isfinite(x)):
        raise ValueError("phi_vec: received NaN or infinite value(s) in input array")
    if np.any(x < 0):
        raise ValueError("phi_vec: requires all x >= 0. Use phi() for negative x.")
    result = np.ones_like(x)
    mask = np.abs(x) >= 1e-12
    xm = x[mask]
    sx = np.sqrt(xm)
    result[mask] = 2.0 * dawsn(sx / 2.0) / sx
    return result


# =============================================================================
# CANCELLATION-FREE TAYLOR SERIES FOR FORM FACTORS
# =============================================================================
# phi(x) = sum_{n>=0} a_n * x^n, with a_n = (-1)^n * n! / (2n+1)!
#
# The form factors h_C, h_R involve expressions like 1/(12x) + (phi-1)/(2x^2)
# where the 1/x and 1/x^2 terms cancel exactly (the form factors are finite at 0).
# Computing them naively at small x causes catastrophic cancellation.
#
# Solution: pre-compute Taylor coefficients for the COMPLETE form factor,
# so the cancellation is done analytically, not numerically.

_N_TAYLOR = 20  # number of Taylor terms (converges for |x| < ~20)

# phi(x) Taylor coefficients: a_n = (-1)^n * n! / (2n+1)!
_AN = np.array([(-1)**n * math.factorial(n) / math.factorial(2 * n + 1)
                for n in range(_N_TAYLOR + 3)])

# hC_scalar(x) = sum_k c_k * x^k, c_k = a_{k+2} / 2
# Derivation: hC = 1/(12x) + (phi-1)/(2x^2), the 1/x terms cancel since a_1=-1/6.
_HC0_TAYLOR = np.array([_AN[k + 2] / 2 for k in range(_N_TAYLOR)])

# hC_dirac(x) = sum_k c_k * x^k, c_k = a_{k+1}/2 + 2*a_{k+2}
# Derivation: hC = (3phi-1)/(6x) + 2(phi-1)/x^2, 1/x cancels since 1/3+2a_1=0.
_HCD_TAYLOR = np.array([_AN[k + 1] / 2 + 2 * _AN[k + 2] for k in range(_N_TAYLOR)])

# hR_dirac(x) = sum_k c_k * x^k, c_k = a_{k+1}/12 + 5*a_{k+2}/6
# Derivation: hR = (3phi+2)/(36x) + 5(phi-1)/(6x^2), 1/x cancels since 5/36+5a_1/6=0.
_HRD_TAYLOR = np.array([_AN[k + 1] / 12 + 5 * _AN[k + 2] / 6
                         for k in range(_N_TAYLOR)])

# hR_scalar(x, xi) = A(x) + xi*B(x) + xi^2*C(x)
# A_k = a_k/32 + a_{k+1}/8 + 5*a_{k+2}/24  (from fRic/3 + fR)
# B_k = -a_k/4 - a_{k+1}/2                  (from fRU)
# C_k = a_k/2                                (from fU = phi/2)
_HR0_A = np.array([_AN[k] / 32 + _AN[k + 1] / 8 + 5 * _AN[k + 2] / 24
                   for k in range(_N_TAYLOR)])
_HR0_B = np.array([-_AN[k] / 4 - _AN[k + 1] / 2 for k in range(_N_TAYLOR)])
_HR0_C = np.array([_AN[k] / 2 for k in range(_N_TAYLOR)])

# hC_vector(x) = phi/4 + (6phi-5)/(6x) + (phi-1)/x^2
# Taylor: c_k = a_k/4 + a_{k+1} + a_{k+2}
# Derivation: phi/4 = sum a_k/4 x^k; (6phi-5)/(6x) has pole 1/(6x) from
# (a_0-5/6)/x = 1/(6x); (phi-1)/x^2 has pole a_1/x = -1/(6x).
# Poles cancel exactly: 1/(6x) - 1/(6x) = 0.
# Verified: c_0 = 1/4 - 1/6 + 1/60 = 1/10 (= beta_W); c_1 = -11/420 (B7).
_HCV_TAYLOR = np.array([_AN[k] / 4 + _AN[k + 1] + _AN[k + 2]
                         for k in range(_N_TAYLOR)])

# hR_vector(x) = -phi/48 + (11-6phi)/(72x) + 5(phi-1)/(12x^2)
# Taylor: c_k = -a_k/48 - a_{k+1}/12 + 5*a_{k+2}/12
# Derivation: -phi/48 = -sum a_k/48 x^k; (11-6phi)/(72x) has pole
# (11-6a_0)/(72x) = 5/(72x); 5(phi-1)/(12x^2) has pole 5a_1/(12x) = -5/(72x).
# Poles cancel exactly: 5/(72x) - 5/(72x) = 0.
# Verified: c_0 = -1/48 + 1/72 + 1/144 = 0 (= beta_R, conformal); c_1 = 1/630 (B8).
_HRV_TAYLOR = np.array([-_AN[k] / 48 - _AN[k + 1] / 12 + 5 * _AN[k + 2] / 12
                         for k in range(_N_TAYLOR)])

# Threshold for switching from Taylor to Dawson evaluation
_TAYLOR_THRESH = 2.0


def _horner(coeffs, x):
    """Evaluate polynomial via Horner's method: sum_k coeffs[k] * x^k."""
    result = coeffs[-1]
    for k in range(len(coeffs) - 2, -1, -1):
        result = result * x + coeffs[k]
    return float(result)


def hC_scalar_fast(x):
    """Fast scalar h_C^(0)(x). Full float64 accuracy for all x >= 0.

    Uses cancellation-free Taylor series for x < 2, Dawson function for x >= 2.

    Raises:
        ValueError: if x is NaN/inf or x < 0.
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"hC_scalar_fast: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"hC_scalar_fast: requires x >= 0, got {x}")
    if abs(x) < _TAYLOR_THRESH:
        return _horner(_HC0_TAYLOR, x)
    p = phi_fast(x)
    return 1.0 / (12.0 * x) + (p - 1.0) / (2.0 * x * x)


def hR_scalar_fast(x, xi=0.0):
    """Fast scalar h_R^(0)(x; xi). Full float64 accuracy for all x >= 0.

    Uses decomposition h_R = A(x) + xi*B(x) + xi^2*C(x) with Taylor series.

    .. warning::
        For x > ~1e35, the multi-term Dawson-based formula accumulates
        floating-point roundoff (catastrophic cancellation in fRic/3 + fR).
        Use hR_scalar_mp() for arbitrary-precision evaluation at extreme x.
        In practice, x < 1000 covers all physical applications.

    Raises:
        ValueError: if x or xi is NaN/inf, or x < 0.
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"hR_scalar_fast: requires finite x, got {x}")
    if not np.isfinite(float(xi)):
        raise ValueError(f"hR_scalar_fast: requires finite xi, got {xi}")
    if x < 0:
        raise ValueError(f"hR_scalar_fast: requires x >= 0, got {x}")
    if abs(x) < _TAYLOR_THRESH:
        a = _horner(_HR0_A, x)
        b = _horner(_HR0_B, x)
        c = _horner(_HR0_C, x)
        return a + xi * b + xi * xi * c
    p = phi_fast(x)
    fRic = 1.0 / (6.0 * x) + (p - 1.0) / (x * x)
    fR = p / 32.0 + p / (8.0 * x) - 7.0 / (48.0 * x) - (p - 1.0) / (8.0 * x * x)
    fRU = -p / 4.0 - (p - 1.0) / (2.0 * x)
    fU = p / 2.0
    return fRic / 3.0 + fR + xi * fRU + xi * xi * fU


def hC_dirac_fast(x):
    """Fast Dirac h_C^(1/2)(x). Full float64 accuracy for all x >= 0.

    Uses cancellation-free Taylor series for x < 2, Dawson function for x >= 2.

    Raises:
        ValueError: if x is NaN/inf or x < 0.
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"hC_dirac_fast: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"hC_dirac_fast: requires x >= 0, got {x}")
    if abs(x) < _TAYLOR_THRESH:
        return _horner(_HCD_TAYLOR, x)
    p = phi_fast(x)
    return (3.0 * p - 1.0) / (6.0 * x) + 2.0 * (p - 1.0) / (x * x)


def hR_dirac_fast(x):
    """Fast Dirac h_R^(1/2)(x). Full float64 accuracy for all x >= 0.

    Uses cancellation-free Taylor series for x < 2, Dawson function for x >= 2.

    Raises:
        ValueError: if x is NaN/inf or x < 0.
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"hR_dirac_fast: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"hR_dirac_fast: requires x >= 0, got {x}")
    if abs(x) < _TAYLOR_THRESH:
        return _horner(_HRD_TAYLOR, x)
    p = phi_fast(x)
    return (3.0 * p + 2.0) / (36.0 * x) + 5.0 * (p - 1.0) / (6.0 * x * x)


def _validate_scan_array(x_array, func_name):
    """Validate array input for scan_* functions (NaN/inf/negative checks)."""
    x = np.asarray(x_array, dtype=float)
    if x.size > 0:
        if np.any(~np.isfinite(x)):
            raise ValueError(
                f"{func_name}: received NaN or infinite value(s) in input array"
            )
        if np.any(x < 0):
            raise ValueError(
                f"{func_name}: requires all x >= 0, got min={x.min()}"
            )
    return x


def scan_hC_scalar(x_array):
    """Evaluate h_C^(0)(x) over numpy array using the fast (Dawson) branch.

    Uses hC_scalar_fast element-wise, avoiding quad-based cancellation issues.
    """
    x = _validate_scan_array(x_array, "scan_hC_scalar")
    return np.array([hC_scalar_fast(x_val) for x_val in x])


def scan_hR_scalar(x_array, xi=0.0):
    """Vectorized h_R^(0)(x; xi) over numpy array."""
    x = _validate_scan_array(x_array, "scan_hR_scalar")
    return np.array([hR_scalar_fast(x_val, xi) for x_val in x])


def scan_hC_dirac(x_array):
    """Vectorized h_C^(1/2)(x) over numpy array."""
    x = _validate_scan_array(x_array, "scan_hC_dirac")
    return np.array([hC_dirac_fast(x_val) for x_val in x])


def scan_hR_dirac(x_array):
    """Vectorized h_R^(1/2)(x) over numpy array."""
    x = _validate_scan_array(x_array, "scan_hR_dirac")
    return np.array([hR_dirac_fast(x_val) for x_val in x])


# =============================================================================
# CZ FORM FACTORS (Codello-Zanusso basis) — numpy
# =============================================================================
# NOTE: These reference implementations use scipy.integrate.quad (phi(x)).
# Near x = 0, large terms cancel (e.g., 1/(6x) vs (phi-1)/x^2), causing
# catastrophic precision loss.  The Taylor limit at |x| < 1e-5 delegates
# to exact values in that regime.  Quad is reliable for x >= 1e-5.
# For high-accuracy work, prefer the _fast variants (Taylor + Dawson)
# or _mp variants (arbitrary-precision mpmath).
# =============================================================================

def f_Ric(x):
    """CZ form factor f_Ric(x) = 1/(6x) + (phi-1)/x^2."""
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"f_Ric: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"f_Ric: requires x >= 0, got {x}")
    if abs(x) < 1e-5:
        return 1.0 / 60  # Taylor limit
    p = phi(x)
    return 1 / (6 * x) + (p - 1) / x**2


def f_R(x):
    """CZ form factor f_R(x) = phi/32 + phi/(8x) - 7/(48x) - (phi-1)/(8x^2)."""
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"f_R: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"f_R: requires x >= 0, got {x}")
    if abs(x) < 1e-5:
        return 1.0 / 120  # Taylor limit
    p = phi(x)
    return p / 32 + p / (8 * x) - 7 / (48 * x) - (p - 1) / (8 * x**2)


def f_RU(x):
    """CZ form factor f_RU(x) = -phi/4 - (phi-1)/(2x)."""
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"f_RU: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"f_RU: requires x >= 0, got {x}")
    if abs(x) < 1e-5:
        return -1.0 / 6  # Taylor limit
    p = phi(x)
    return -p / 4 - (p - 1) / (2 * x)


def f_U(x):
    """CZ form factor f_U(x) = phi/2."""
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"f_U: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"f_U: requires x >= 0, got {x}")
    return phi(x) / 2


def f_Omega(x):
    """CZ form factor f_Omega(x) = -(phi-1)/(2x)."""
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"f_Omega: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"f_Omega: requires x >= 0, got {x}")
    if abs(x) < 1e-5:
        return 1.0 / 12  # Taylor limit
    p = phi(x)
    return -(p - 1) / (2 * x)


# =============================================================================
# SPIN-0: SCALAR FORM FACTORS
# =============================================================================

def hC_scalar(x):
    """Scalar Weyl form factor h_C^(0)(x) = (1/2) f_Ric(x).

    h_C^(0)(x) = 1/(12x) + (phi-1)/(2x^2)
    h_C^(0)(0) = 1/120
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"hC_scalar: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"hC_scalar: requires x >= 0, got {x}")
    if abs(x) < 1e-5:
        return 1.0 / 120
    p = phi(x)
    return 1 / (12 * x) + (p - 1) / (2 * x**2)


def hR_scalar(x, xi=0.0):
    """Scalar R^2 form factor h_R^(0)(x; xi).

    h_R^(0)(x; xi) = f_{R,bis}(x) + xi * f_RU(x) + xi^2 * f_U(x)

    where f_{R,bis} = (1/3)*f_Ric + f_R.

    Parameters:
        x: dimensionless argument Box/Lambda^2
        xi: non-minimal coupling (xi=0: minimal, xi=1/6: conformal)

    Local limits:
        h_R^(0)(0; xi) = (1/2)(xi - 1/6)^2
        h_R^(0)(0; 0)   = 1/72
        h_R^(0)(0; 1/6) = 0  (conformal invariance)
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"hR_scalar: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"hR_scalar: requires x >= 0, got {x}")
    xi = float(xi)
    if not np.isfinite(xi):
        raise ValueError(f"hR_scalar: requires finite xi, got {xi}")
    return f_Ric(x) / 3 + f_R(x) + xi * f_RU(x) + xi**2 * f_U(x)


# =============================================================================
# SPIN-1/2: DIRAC FORM FACTORS
# =============================================================================

def hC_dirac(x):
    """Dirac Weyl form factor h_C^(1/2)(x).

    h_C^(1/2)(x) = (3*phi - 1)/(6x) + 2*(phi - 1)/x^2
    h_C^(1/2)(0) = -1/20

    Uses E = -R/4 (corrected sign from NT-1).
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"hC_dirac: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"hC_dirac: requires x >= 0, got {x}")
    if abs(x) < 1e-5:
        return -1.0 / 20
    p = phi(x)
    return (3 * p - 1) / (6 * x) + 2 * (p - 1) / x**2


def hR_dirac(x):
    """Dirac R^2 form factor h_R^(1/2)(x).

    h_R^(1/2)(x) = (3*phi + 2)/(36x) + 5*(phi - 1)/(6x^2)
    h_R^(1/2)(0) = 0  (conformal invariance of massless Dirac)

    Uses E = -R/4 (corrected sign).
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"hR_dirac: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"hR_dirac: requires x >= 0, got {x}")
    if abs(x) < 1e-5:
        return 0.0
    p = phi(x)
    return (3 * p + 2) / (36 * x) + 5 * (p - 1) / (6 * x**2)


# =============================================================================
# SPIN-1: VECTOR FORM FACTORS (NT-1b Phase 2 — COMPLETE)
# =============================================================================
# Physical gauge field = unconstrained vector - 2 FP ghosts (xi=0 scalars).
#
# h_C^(1)(x) = phi/4 + (6phi-5)/(6x) + (phi-1)/x^2
# h_R^(1)(x) = -phi/48 + (11-6phi)/(72x) + 5(phi-1)/(12x^2)
#
# Local limits: h_C^(1)(0) = 1/10 (= beta_W), h_R^(1)(0) = 0 (conformal).
# Asymptotics: h_C^(1) ~ -1/(3x) + 2/x^2, h_R^(1) ~ 1/(9x) - 2/(3x^2).
# No xi-dependence (gauge invariance fixes the coupling uniquely).

def hC_vector(x):
    """Vector Weyl form factor h_C^(1)(x) (physical gauge field).

    h_C^(1)(x) = phi(x)/4 + (6*phi(x) - 5)/(6x) + (phi(x) - 1)/x^2

    This is the PHYSICAL form factor after subtracting 2 Faddeev-Popov
    ghost contributions (minimally coupled scalars, xi=0) from the
    unconstrained vector result:
        h_C^(1) = h_C^(1,unconstr) - 2 * h_C^(0)(xi=0)

    Local limit: h_C^(1)(0) = 1/10 = beta_W^(1)
    Large x:     h_C^(1)(x) ~ -1/(3x) + 2/x^2

    Uses scipy.integrate.quad (reference implementation).
    For fast/stable evaluation, prefer hC_vector_fast().
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"hC_vector: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"hC_vector: requires x >= 0, got {x}")
    if abs(x) < 1e-5:
        return 0.1  # 1/10
    p = phi(x)
    return p / 4 + (6 * p - 5) / (6 * x) + (p - 1) / x**2


def hR_vector(x):
    """Vector R^2 form factor h_R^(1)(x) (physical gauge field).

    h_R^(1)(x) = -phi(x)/48 + (11 - 6*phi(x))/(72x) + 5*(phi(x) - 1)/(12x^2)

    This is the PHYSICAL form factor after subtracting 2 FP ghosts.
    Vanishes at x=0 due to conformal invariance of the Maxwell action:
    the Weyl-invariant F^2 cannot generate an R^2 counterterm.

    Local limit: h_R^(1)(0) = 0 = beta_R^(1)  (conformal invariance)
    Large x:     h_R^(1)(x) ~ 1/(9x) - 2/(3x^2)

    Uses scipy.integrate.quad (reference implementation).
    For fast/stable evaluation, prefer hR_vector_fast().
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"hR_vector: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"hR_vector: requires x >= 0, got {x}")
    if abs(x) < 1e-5:
        return 0.0  # beta_R = 0 (conformal)
    p = phi(x)
    return -p / 48 + (11 - 6 * p) / (72 * x) + 5 * (p - 1) / (12 * x**2)


def hC_vector_fast(x):
    """Fast vector h_C^(1)(x). Full float64 accuracy for all x >= 0.

    Uses cancellation-free Taylor series for x < 2, Dawson function for x >= 2.

    Raises:
        ValueError: if x is NaN/inf or x < 0.
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"hC_vector_fast: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"hC_vector_fast: requires x >= 0, got {x}")
    if abs(x) < _TAYLOR_THRESH:
        return _horner(_HCV_TAYLOR, x)
    p = phi_fast(x)
    return p / 4.0 + (6.0 * p - 5.0) / (6.0 * x) + (p - 1.0) / (x * x)


def hR_vector_fast(x):
    """Fast vector h_R^(1)(x). Full float64 accuracy for all x >= 0.

    Uses cancellation-free Taylor series for x < 2, Dawson function for x >= 2.

    Raises:
        ValueError: if x is NaN/inf or x < 0.
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"hR_vector_fast: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"hR_vector_fast: requires x >= 0, got {x}")
    if abs(x) < _TAYLOR_THRESH:
        return _horner(_HRV_TAYLOR, x)
    p = phi_fast(x)
    return -p / 48.0 + (11.0 - 6.0 * p) / (72.0 * x) + 5.0 * (p - 1.0) / (12.0 * x * x)


def scan_hC_vector(x_array):
    """Evaluate h_C^(1)(x) over numpy array using the fast (Dawson) branch.

    Uses hC_vector_fast element-wise, avoiding quad-based cancellation issues.
    """
    x = _validate_scan_array(x_array, "scan_hC_vector")
    return np.array([hC_vector_fast(x_val) for x_val in x])


def scan_hR_vector(x_array):
    """Vectorized h_R^(1)(x) over numpy array."""
    x = _validate_scan_array(x_array, "scan_hR_vector")
    return np.array([hR_vector_fast(x_val) for x_val in x])


# =============================================================================
# COMBINED SM FORM FACTORS
# =============================================================================

def F1_total(x, N_s=4, N_f=45, N_v=12, xi=0.0):
    """Total SM Weyl form factor F_1(x) = alpha_C(x) / (16*pi^2).

    alpha_C(x) = N_s * h_C^(0)(x) + (N_f/2) * h_C^(1/2)(x) + N_v * h_C^(1)(x)

    CONVENTION (NT-1b Phase 3):
        N_f counts 2-component WEYL spinors (SM default: 45).
        h_C^(1/2) is the form factor for one 4-component DIRAC fermion.
        Therefore N_D = N_f / 2 Dirac fermions contribute to the sum.
        Reference: CPR 0805.2909, eq. (3.14).

    Parameters:
        x: dimensionless Box/Lambda^2
        N_s: number of real scalar d.o.f. (default: 4 for Higgs doublet)
        N_f: number of 2-component Weyl fermion d.o.f. (default: 45 for SM)
        N_v: number of vector d.o.f. (default: 12 for SM)
        xi: scalar non-minimal coupling

    Returns:
        F_1(x), the full Weyl-squared form factor including 1/(16*pi^2).

    Local limit (x -> 0, SM defaults):
        F_1(0) = alpha_C / (16*pi^2) = (13/120) / (16*pi^2) = 13 / (1920*pi^2)
    """
    if not np.isfinite(float(x)):
        raise ValueError(f"F1_total: requires finite x, got {x}")
    if float(x) < 0:
        raise ValueError(f"F1_total: requires x >= 0, got {x}")
    for name, val in [("N_s", N_s), ("N_f", N_f), ("N_v", N_v), ("xi", xi)]:
        if not np.isfinite(float(val)):
            raise ValueError(f"F1_total: requires finite {name}, got {val}")
    # N_f is Weyl count; h_C^(1/2) is per-Dirac → divide by 2
    N_D = N_f / 2
    result = N_s * hC_scalar_fast(x) + N_D * hC_dirac_fast(x) + N_v * hC_vector_fast(x)
    return result / (16 * np.pi**2)


def F2_total(x, N_s=4, N_f=45, N_v=12, xi=0.0):
    """Total SM R^2 form factor F_2(x) = alpha_R(x) / (16*pi^2).

    alpha_R(x) = N_s * h_R^(0)(x, xi) + (N_f/2) * h_R^(1/2)(x) + N_v * h_R^(1)(x)

    CONVENTION: Same Weyl-counting as F1_total (N_f/2 Dirac fermions).
    xi only affects the scalar contribution.

    Local limit (x -> 0, SM defaults, xi=0):
        F_2(0) = alpha_R(xi=0) / (16*pi^2) = 2*(0-1/6)^2 / (16*pi^2) = 1/(288*pi^2)
    Local limit (x -> 0, SM defaults, xi=1/6):
        F_2(0) = 0  (conformal coupling — all sectors conformal)
    """
    if not np.isfinite(float(x)):
        raise ValueError(f"F2_total: requires finite x, got {x}")
    if float(x) < 0:
        raise ValueError(f"F2_total: requires x >= 0, got {x}")
    for name, val in [("N_s", N_s), ("N_f", N_f), ("N_v", N_v), ("xi", xi)]:
        if not np.isfinite(float(val)):
            raise ValueError(f"F2_total: requires finite {name}, got {val}")
    # N_f is Weyl count; h_R^(1/2) is per-Dirac → divide by 2
    N_D = N_f / 2
    result = N_s * hR_scalar_fast(x, xi) + N_D * hR_dirac_fast(x) + N_v * hR_vector_fast(x)
    return result / (16 * np.pi**2)


# =============================================================================
# PHASE 3: LOCAL-LIMIT COEFFICIENTS AND c1/c2 RATIO
# =============================================================================
# These functions compute the x -> 0 (local, Seeley-DeWitt) limits of
# the total SM form factors F_1(x) and F_2(x), and the derived
# curvature-squared couplings c_1, c_2 in the {R^2, R_{mn}^2} basis.
#
# Basis conversion (using Gauss-Bonnet E_4 = R^2 - 4R_{mn}^2 + R_{mnrs}^2):
#   In Weyl basis:  S_4 = (f_0 / (16*pi^2)) * int [alpha_C * C^2 + alpha_R * R^2]
#   In Ricci basis: S_4 = (f_0 / (16*pi^2)) * int [c_1 * R^2 + c_2 * R_{mn}^2]
#   where  c_1 = -2*alpha_C/3 + alpha_R,  c_2 = 2*alpha_C
#   (dropping Euler density = total derivative, and factors of f_0 cancel in ratio)
#
# Reference: CPR 0805.2909; BV 1990; Vassilevich hep-th/0306138.

def alpha_C_SM(N_s=4, N_f=45, N_v=12):
    """Total SM Weyl-squared coefficient alpha_C = F_1(0) * 16*pi^2.

    alpha_C = N_s * h_C^(0)(0) + (N_f/2) * h_C^(1/2)(0) + N_v * h_C^(1)(0)
            = N_s/120 + (N_f/2)*(-1/20) + N_v/10

    For SM defaults: alpha_C = 4/120 - 45/40 + 12/10 = 13/120.
    This is xi-INDEPENDENT (Weyl coupling is conformally invariant at one loop).

    Returns:
        alpha_C (float): raw coefficient (not divided by 16*pi^2).
    """
    N_D = N_f / 2
    return N_s / 120 + N_D * (-1 / 20) + N_v / 10


def alpha_R_SM(xi, N_s=4, N_f=45, N_v=12):
    """Total SM R-squared coefficient alpha_R(xi) = F_2(0) * 16*pi^2.

    alpha_R(xi) = N_s * h_R^(0)(0, xi) + (N_f/2) * h_R^(1/2)(0) + N_v * h_R^(1)(0)
               = N_s * (1/2)(xi - 1/6)^2 + 0 + 0
               = 2 * (xi - 1/6)^2       [for SM defaults N_s=4]

    This depends on xi (scalar non-minimal coupling) but NOT on N_f, N_v
    because Dirac and vector beta_R vanish (conformal invariance).

    Parameters:
        xi: scalar non-minimal coupling (xi=1/6 is conformal)

    Returns:
        alpha_R(xi) (float): raw coefficient (not divided by 16*pi^2).
    """
    if not isinstance(xi, (int, float, np.integer, np.floating)):
        raise TypeError(f"alpha_R_SM: xi must be numeric, got {type(xi).__name__}")
    if not np.isfinite(float(xi)):
        raise ValueError(f"alpha_R_SM: xi must be finite, got {xi}")
    N_D = N_f / 2
    # Only scalar contributes; Dirac/vector beta_R = 0
    return N_s * 0.5 * (xi - 1 / 6) ** 2 + N_D * 0 + N_v * 0


def c1_c2_ratio_SM(xi, N_s=4, N_f=45, N_v=12):
    """Ratio c_1/c_2 for full SM field content, as a function of xi.

    In the {R^2, R_{mn}^2} basis (Ricci basis):
        c_1 = -2*alpha_C/3 + alpha_R(xi)
        c_2 = 2*alpha_C

    So: c_1/c_2 = -1/3 + alpha_R(xi) / (2*alpha_C)

    For SM defaults (alpha_C = 13/120):
        c_1/c_2 = -1/3 + 120 * (xi - 1/6)^2 / 13

    Special cases:
        xi = 1/6 (conformal):  c_1/c_2 = -1/3
        xi = 0   (minimal):    c_1/c_2 = -1/3 + 120/(13*36) = -1/3 + 10/39 = -1/13

    Parameters:
        xi: scalar non-minimal coupling

    Returns:
        c_1/c_2 ratio (float).

    Raises:
        ValueError: if alpha_C = 0 (division by zero; requires non-SM content).
    """
    aC = alpha_C_SM(N_s, N_f, N_v)
    aR = alpha_R_SM(xi, N_s, N_f, N_v)
    if abs(aC) < 1e-30:
        raise ValueError(
            "c1_c2_ratio_SM: alpha_C = 0, c_2 = 2*alpha_C = 0, ratio undefined"
        )
    return -1 / 3 + aR / (2 * aC)


def scalar_mode_mass_SM(xi, N_s=4, N_f=45, N_v=12):
    """Scalar-mode coupling 3*c_1 + c_2 for SM field content.

    3*c_1 + c_2 = 3*(-2*alpha_C/3 + alpha_R) + 2*alpha_C
               = -2*alpha_C + 3*alpha_R + 2*alpha_C
               = 3*alpha_R

    So scalar mode coupling = 3*alpha_R(xi) = 3*N_s/2 * (xi - 1/6)^2
    = 6*(xi - 1/6)^2 for SM defaults.

    The spin-0 massive mode of the graviton decouples iff 3*c_1 + c_2 = 0,
    i.e., alpha_R = 0, i.e., xi = 1/6 (conformal coupling).

    Parameters:
        xi: scalar non-minimal coupling

    Returns:
        3*c_1 + c_2 = 3*alpha_R(xi) (float, raw coefficient).
    """
    return 3 * alpha_R_SM(xi, N_s, N_f, N_v)


def uv_asymptotic_F1_total(N_s=4, N_f=45, N_v=12):
    """Leading UV asymptotic coefficient: lim_{x->inf} x * F_1(x) * 16*pi^2.

    In the UV (x -> inf), each h_C^(s)(x) ~ a_s/x + O(1/x^2), so
    x * h_C^(s)(x) -> a_s. The leading UV coefficient of x*F_1 is:

    N_s * a_s^(0) + (N_f/2) * a_s^(1/2) + N_v * a_s^(1)

    where:
        a_s^(0) = 1/12      (from h_C^(0) = 1/(12x) + ...)
        a_s^(1/2) = -1/6    (from h_C^(1/2): x->inf, phi->0, leading term -1/(6x))
        a_s^(1) = -1/3      (from asymptotic: h_C^(1) -> -1/(3x) + ...)

    x * alpha_C_total(x) -> N_s/12 + (N_f/2)*(-1/6) + N_v*(-1/3)
    = 4/12 + 22.5*(-1/6) + 12*(-1/3)
    = 1/3 - 15/4 - 4 = 4/12 - 45/12 - 48/12 = -89/12

    So x*F_1(x) -> -89/(12 * 16*pi^2) = -89/(192*pi^2).

    Returns:
        Leading UV coefficient of x * alpha_C(x), NOT divided by 16*pi^2.
    """
    N_D = N_f / 2
    # Leading 1/x coefficients: x * h_C^(s)(x) -> a_s as x -> inf.
    # Using phi(x) ~ 2/x for large x:
    #   h_C^(0)   ~ 1/(12x)   [from 1/(12x) + O(1/x^2)]
    #   h_C^(1/2) ~ -1/(6x)   [from (3*0-1)/(6x) + O(1/x^2)]
    #   h_C^(1)   ~ -1/(3x)   [phi/4 ~ 1/(2x), (6phi-5)/(6x) ~ -5/(6x);
    #                           sum: 1/(2x) - 5/(6x) = -1/(3x)]
    a_scalar = 1 / 12
    a_dirac = -1 / 6
    a_vector = -1 / 3
    return N_s * a_scalar + N_D * a_dirac + N_v * a_vector


# =============================================================================
# GENERAL SPECTRAL FUNCTION FORMULAS
# =============================================================================

def F1_spectral(z, psi, psi1, psi2, psi1_0, psi2_0):
    """F_1(z) for general spectral function psi.

    F_1(z) = (1/(16*pi^2)) * [Psi_1(0)/(3z)
              + (2/z^2) int_0^1 [Psi_2(a(1-a)z) - Psi_2(0)] da
              - (1/4) int_0^1 (1-2a)^2 psi(a(1-a)z) da]

    where Psi_1(u) = int_u^inf psi(v) dv, Psi_2(u) = int_u^inf (v-u) psi(v) dv.

    Parameters:
        z: dimensionless argument (must be > 0)
        psi: spectral function psi(u)
        psi1: first antiderivative Psi_1(u)
        psi2: second antiderivative Psi_2(u)
        psi1_0: Psi_1(0)
        psi2_0: Psi_2(0)

    Raises:
        ValueError: if z <= 0 (formula has 1/z and 1/z^2 poles;
                    the z->0 limit requires L'Hopital and is not
                    implemented here).
    """
    z = float(z)
    if not np.isfinite(z):
        raise ValueError(f"F1_spectral: requires finite z, got z={z}")
    if z <= 0:
        raise ValueError(f"F1_spectral: requires z > 0, got z={z}")
    t1 = psi1_0 / (3 * z)
    int2, _ = quad(lambda a: psi2(a * (1 - a) * z) - psi2_0, 0, 1)
    t2 = 2 / z**2 * int2
    int3, _ = quad(lambda a: (1 - 2 * a)**2 * psi(a * (1 - a) * z), 0, 1)
    t3 = -0.25 * int3
    return (t1 + t2 + t3) / (16 * np.pi**2)


def F2_spectral(z, psi, psi1, psi2, psi1_0, psi2_0):
    """F_2(z) for general spectral function psi.

    See F1_spectral for parameter descriptions.

    Raises:
        ValueError: if z <= 0.
    """
    z = float(z)
    if not np.isfinite(z):
        raise ValueError(f"F2_spectral: requires finite z, got z={z}")
    if z <= 0:
        raise ValueError(f"F2_spectral: requires z > 0, got z={z}")
    int1, _ = quad(lambda a: (1 - 2 * a)**2 * psi(a * (1 - a) * z), 0, 1)
    t1 = 5 / 24 * int1
    int2, _ = quad(lambda a: psi1(a * (1 - a) * z), 0, 1)
    t2 = 1 / (2 * z) * int2
    t3 = -13 * psi1_0 / (36 * z)
    int4, _ = quad(lambda a: psi2(a * (1 - a) * z) - psi2_0, 0, 1)
    t4 = 5 / (6 * z**2) * int4
    return (t1 + t2 + t3 + t4) / (16 * np.pi**2)


# =============================================================================
# HIGH-PRECISION VERSIONS (mpmath, >= 100 digits)
# =============================================================================

_MP_TAYLOR_NTERMS = 80


def _mp_taylor_eval(form_factor, x, xi=None):
    """Evaluate a form factor via Taylor series in mpmath arithmetic.

    Avoids catastrophic cancellation at small x by analytically cancelling
    the 1/x and 1/x^2 poles in the coefficient formulas.  Called by
    the _mp functions when 0 < x < 2.
    """
    from mpmath import fac, mpf
    N = _MP_TAYLOR_NTERMS
    x = mpf(x)
    # Precompute phi Taylor coefficients: a_n = (-1)^n n! / (2n+1)!
    a = [(-1)**n * fac(n) / fac(2 * n + 1) for n in range(N + 3)]
    s = mpf(0)
    xk = mpf(1)
    for k in range(N):
        if form_factor == 'hC_scalar':
            ck = a[k + 2] / 2
        elif form_factor == 'hR_scalar':
            xi_m = mpf(xi)
            ck = (a[k] / 32 + a[k + 1] / 8 + 5 * a[k + 2] / 24
                  + xi_m * (-a[k] / 4 - a[k + 1] / 2)
                  + xi_m**2 * a[k] / 2)
        elif form_factor == 'hC_dirac':
            ck = a[k + 1] / 2 + 2 * a[k + 2]
        elif form_factor == 'hR_dirac':
            ck = a[k + 1] / 12 + 5 * a[k + 2] / 6
        elif form_factor == 'hC_vector':
            ck = a[k] / 4 + a[k + 1] + a[k + 2]
        elif form_factor == 'hR_vector':
            ck = -a[k] / 48 - a[k + 1] / 12 + 5 * a[k + 2] / 12
        else:
            raise ValueError(f"_mp_taylor_eval: unknown form factor '{form_factor}'")
        s += ck * xk
        xk *= x
    return s


def phi_mp(x, dps=100):
    """High-precision master function using mpmath.

    Parameters:
        x: argument (must be finite and >= 0)
        dps: decimal places of precision (default: 100)
    """
    from mpmath import exp as mpexp
    from mpmath import isfinite as mpisfinite
    from mpmath import mp, mpf
    from mpmath import quad as mpquad
    old_dps = mp.dps
    mp.dps = dps
    try:
        x = mpf(x)
        if not mpisfinite(x):
            raise ValueError(f"phi_mp: requires finite x, got {float(x)}")
        if x < 0:
            raise ValueError(f"phi_mp: requires x >= 0, got {float(x)}")
        if x == 0:
            return mpf(1)
        return mpquad(lambda a: mpexp(-a * (1 - a) * x), [0, 1])
    finally:
        mp.dps = old_dps


def hC_scalar_mp(x, dps=100):
    """High-precision scalar h_C^(0)(x).

    Raises:
        ValueError: if x is non-finite or < 0.
    """
    from mpmath import isfinite as mpisfinite
    from mpmath import mp, mpf
    old_dps = mp.dps
    mp.dps = dps
    try:
        x = mpf(x)
        if not mpisfinite(x):
            raise ValueError(f"hC_scalar_mp: requires finite x, got {float(x)}")
        if x < 0:
            raise ValueError(f"hC_scalar_mp: requires x >= 0, got {float(x)}")
        if x == 0:
            return mpf(1) / 120
        if x < 2:
            return _mp_taylor_eval('hC_scalar', x)
        p = phi_mp(x, dps)
        return mpf(1) / (12 * x) + (p - 1) / (2 * x**2)
    finally:
        mp.dps = old_dps


def hR_scalar_mp(x, xi=0, dps=100):
    """High-precision scalar h_R^(0)(x; xi).

    Raises:
        ValueError: if x or xi is non-finite, or x < 0.
    """
    from mpmath import isfinite as mpisfinite
    from mpmath import mp, mpf
    old_dps = mp.dps
    mp.dps = dps
    try:
        x = mpf(x)
        if not mpisfinite(x):
            raise ValueError(f"hR_scalar_mp: requires finite x, got {float(x)}")
        if x < 0:
            raise ValueError(f"hR_scalar_mp: requires x >= 0, got {float(x)}")
        xi = mpf(xi)
        if not mpisfinite(xi):
            raise ValueError(f"hR_scalar_mp: requires finite xi, got {float(xi)}")
        if x == 0:
            return (xi - mpf(1) / 6)**2 / 2
        if x < 2:
            return _mp_taylor_eval('hR_scalar', x, xi=xi)
        p = phi_mp(x, dps)

        A = p / 32 + (p / 8 - mpf(13) / 144) / x + 5 * (p - 1) / (24 * x**2)
        B = -p / 4 - (p - 1) / (2 * x)
        C = p / 2
        return A + xi * B + xi**2 * C
    finally:
        mp.dps = old_dps


def hC_dirac_mp(x, dps=100):
    """High-precision Dirac h_C^(1/2)(x).

    Raises:
        ValueError: if x is non-finite or < 0.
    """
    from mpmath import isfinite as mpisfinite
    from mpmath import mp, mpf
    old_dps = mp.dps
    mp.dps = dps
    try:
        x = mpf(x)
        if not mpisfinite(x):
            raise ValueError(f"hC_dirac_mp: requires finite x, got {float(x)}")
        if x < 0:
            raise ValueError(f"hC_dirac_mp: requires x >= 0, got {float(x)}")
        if x == 0:
            return mpf(-1) / 20
        if x < 2:
            return _mp_taylor_eval('hC_dirac', x)
        p = phi_mp(x, dps)
        return (3 * p - 1) / (6 * x) + 2 * (p - 1) / x**2
    finally:
        mp.dps = old_dps


def hR_dirac_mp(x, dps=100):
    """High-precision Dirac h_R^(1/2)(x).

    Raises:
        ValueError: if x is non-finite or < 0.
    """
    from mpmath import isfinite as mpisfinite
    from mpmath import mp, mpf
    old_dps = mp.dps
    mp.dps = dps
    try:
        x = mpf(x)
        if not mpisfinite(x):
            raise ValueError(f"hR_dirac_mp: requires finite x, got {float(x)}")
        if x < 0:
            raise ValueError(f"hR_dirac_mp: requires x >= 0, got {float(x)}")
        if x == 0:
            return mpf(0)
        if x < 2:
            return _mp_taylor_eval('hR_dirac', x)
        p = phi_mp(x, dps)
        return (3 * p + 2) / (36 * x) + 5 * (p - 1) / (6 * x**2)
    finally:
        mp.dps = old_dps


def hC_vector_mp(x, dps=100):
    """High-precision vector h_C^(1)(x).

    Raises:
        ValueError: if x is non-finite or < 0.
    """
    from mpmath import isfinite as mpisfinite
    from mpmath import mp, mpf
    old_dps = mp.dps
    mp.dps = dps
    try:
        x = mpf(x)
        if not mpisfinite(x):
            raise ValueError(f"hC_vector_mp: requires finite x, got {float(x)}")
        if x < 0:
            raise ValueError(f"hC_vector_mp: requires x >= 0, got {float(x)}")
        if x == 0:
            return mpf(1) / 10
        if x < 2:
            return _mp_taylor_eval('hC_vector', x)
        p = phi_mp(x, dps)
        return p / 4 + (6 * p - 5) / (6 * x) + (p - 1) / x**2
    finally:
        mp.dps = old_dps


def hR_vector_mp(x, dps=100):
    """High-precision vector h_R^(1)(x).

    Raises:
        ValueError: if x is non-finite or < 0.
    """
    from mpmath import isfinite as mpisfinite
    from mpmath import mp, mpf
    old_dps = mp.dps
    mp.dps = dps
    try:
        x = mpf(x)
        if not mpisfinite(x):
            raise ValueError(f"hR_vector_mp: requires finite x, got {float(x)}")
        if x < 0:
            raise ValueError(f"hR_vector_mp: requires x >= 0, got {float(x)}")
        if x == 0:
            return mpf(0)
        if x < 2:
            return _mp_taylor_eval('hR_vector', x)
        p = phi_mp(x, dps)
        return -p / 48 + (11 - 6 * p) / (72 * x) + 5 * (p - 1) / (12 * x**2)
    finally:
        mp.dps = old_dps


# =============================================================================
# TAYLOR SERIES (for x ~ 0, analytic pole cancellation)
# =============================================================================

def hC_scalar_taylor(x, N=60):
    """h_C^(0)(x) via Taylor series with analytic pole cancellation.

    h_C^(0)(x) = sum_{m=0}^{N-1} (-1)^m * b_{m+2}/2 * x^m

    where b_n = n!/(2n+1)!

    Accurate for |x| < ~2. For larger x, use direct integration.
    """
    from mpmath import fac, mpf, power
    x_float = float(x)
    if not np.isfinite(x_float):
        raise ValueError(f"hC_scalar_taylor: requires finite x, got {x_float}")
    if x_float < 0:
        raise ValueError(f"hC_scalar_taylor: requires x >= 0, got {x_float}")
    x = mpf(x)
    s = mpf(0)
    for m in range(N):
        b_mp2 = fac(m + 2) / fac(2 * (m + 2) + 1)
        s += power(-1, m) * b_mp2 / 2 * power(x, m)
    return s


def hR_scalar_taylor(x, xi=0, N=60):
    """h_R^(0)(x; xi) via Taylor series with analytic pole cancellation.

    h_R(x; xi) = sum_{m=0}^{N-1} (-1)^m * [c_A + xi*c_B + xi^2*c_C] * x^m

    where:
        c_A = b_m/32 - b_{m+1}/8 + 5*b_{m+2}/24
        c_B = -b_m/4 + b_{m+1}/2
        c_C = b_m/2
        b_n = n!/(2n+1)!
    """
    from mpmath import fac, mpf, power
    x_float = float(x)
    if not np.isfinite(x_float):
        raise ValueError(f"hR_scalar_taylor: requires finite x, got {x_float}")
    if x_float < 0:
        raise ValueError(f"hR_scalar_taylor: requires x >= 0, got {x_float}")
    xi_float = float(xi)
    if not np.isfinite(xi_float):
        raise ValueError(f"hR_scalar_taylor: requires finite xi, got {xi_float}")
    x = mpf(x)
    xi = mpf(xi)
    s = mpf(0)
    for m in range(N):
        bm = fac(m) / fac(2 * m + 1)
        bm1 = fac(m + 1) / fac(2 * m + 3)
        bm2 = fac(m + 2) / fac(2 * m + 5)
        c_A = bm / 32 - bm1 / 8 + 5 * bm2 / 24
        c_B = -bm / 4 + bm1 / 2
        c_C = bm / 2
        coeff = c_A + xi * c_B + xi**2 * c_C
        s += power(-1, m) * coeff * power(x, m)
    return s


# =============================================================================
# FORM FACTOR DERIVATIVES d/dx (needed for NT-4 field equations)
# =============================================================================
# Computed analytically from the explicit formulas.
# phi'(x) = -int_0^1 alpha(1-alpha) exp[-alpha(1-alpha)*x] dalpha

def dphi_dx(x):
    """Derivative of master function: phi'(x) = -int_0^1 a(1-a) e^{-a(1-a)x} da."""
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"dphi_dx: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"dphi_dx: requires x >= 0, got {x}")
    if abs(x) < 1e-12:
        return -1.0 / 6  # phi'(0) = -a_1 = -1/6
    result, _ = quad(lambda a: -a * (1 - a) * np.exp(-a * (1 - a) * x), 0, 1)
    return result


def dphi_dx_fast(x):
    """Fast derivative phi'(x) via Dawson function identity.

    phi'(x) = [1 - phi(x)*(1 + x/2)] / (2x)

    Derived by differentiating phi(x) = 2*dawsn(sqrt(x)/2)/sqrt(x).
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"dphi_dx_fast: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"dphi_dx_fast: requires x >= 0, got {x}")
    if abs(x) <= 1e-12:
        return -1.0 / 6
    p = phi_fast(x)
    return (1.0 - p * (1.0 + x / 2.0)) / (2.0 * x)


def dhC_scalar_dx(x):
    """Derivative d/dx of h_C^(0)(x).

    d/dx h_C = -1/(12x^2) + phi'/(2x^2) - (phi-1)/x^3
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"dhC_scalar_dx: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"dhC_scalar_dx: requires x >= 0, got {x}")
    if abs(x) < _TAYLOR_THRESH:
        # Use Taylor series: d/dx sum c_k x^k = sum k*c_k x^{k-1}
        coeffs = np.array([k * _HC0_TAYLOR[k] for k in range(1, _N_TAYLOR)])
        return _horner(coeffs, x)
    p = phi_fast(x)
    dp = dphi_dx_fast(x)
    return -1.0 / (12.0 * x**2) + dp / (2.0 * x**2) - (p - 1.0) / (x**3)


def dhC_dirac_dx(x):
    """Derivative d/dx of h_C^(1/2)(x).

    d/dx h_C = (3phi' - 0)/(6x) - (3phi-1)/(6x^2) + 2phi'/x^2 - 4(phi-1)/x^3
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"dhC_dirac_dx: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"dhC_dirac_dx: requires x >= 0, got {x}")
    if abs(x) < _TAYLOR_THRESH:
        coeffs = np.array([k * _HCD_TAYLOR[k] for k in range(1, _N_TAYLOR)])
        return _horner(coeffs, x)
    p = phi_fast(x)
    dp = dphi_dx_fast(x)
    return (3.0 * dp) / (6.0 * x) - (3.0 * p - 1.0) / (6.0 * x**2) \
        + 2.0 * dp / (x**2) - 4.0 * (p - 1.0) / (x**3)


def dhR_dirac_dx(x):
    """Derivative d/dx of h_R^(1/2)(x).

    d/dx h_R = (3phi')/(36x) - (3phi+2)/(36x^2) + 5phi'/(6x^2) - 10(phi-1)/(6x^3)
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"dhR_dirac_dx: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"dhR_dirac_dx: requires x >= 0, got {x}")
    if abs(x) < _TAYLOR_THRESH:
        coeffs = np.array([k * _HRD_TAYLOR[k] for k in range(1, _N_TAYLOR)])
        return _horner(coeffs, x)
    p = phi_fast(x)
    dp = dphi_dx_fast(x)
    return (3.0 * dp) / (36.0 * x) - (3.0 * p + 2.0) / (36.0 * x**2) \
        + 5.0 * dp / (6.0 * x**2) - 10.0 * (p - 1.0) / (6.0 * x**3)


def dhR_scalar_dx(x, xi=0.0):
    """Derivative d/dx h_R^(0)(x; xi) for scalar field.

    Uses Taylor series derivative for x < 2 (cancellation-safe) and
    analytic derivative via phi_fast/dphi_dx_fast for x >= 2.

    h_R = fRic/3 + fR + xi*fRU + xi^2*fU, so
    d/dx h_R = d(fRic/3)/dx + d(fR)/dx + xi*d(fRU)/dx + xi^2*d(fU)/dx.

    Parameters:
        x: dimensionless argument (= -s*Box/Lambda^2), must be >= 0
        xi: non-minimal coupling parameter (xi=0 minimal, xi=1/6 conformal)

    Returns:
        d/dx h_R^(0)(x; xi) at the given point.
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"dhR_scalar_dx: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"dhR_scalar_dx: requires x >= 0, got {x}")
    xi = float(xi)
    if not np.isfinite(xi):
        raise ValueError(f"dhR_scalar_dx: requires finite xi, got {xi}")
    if abs(x) < _TAYLOR_THRESH:
        deriv_coeffs = np.array([
            k * (_HR0_A[k] + xi * _HR0_B[k] + xi * xi * _HR0_C[k])
            for k in range(1, _N_TAYLOR)
        ])
        return _horner(deriv_coeffs, x)
    p = phi_fast(x)
    dp = dphi_dx_fast(x)
    # d(fRic/3)/dx: fRic = 1/(6x) + (p-1)/x^2
    dA1 = -1.0 / (18.0 * x**2) + dp / (3.0 * x**2) - 2.0 * (p - 1.0) / (3.0 * x**3)
    # d(fR)/dx: fR = p/32 + p/(8x) - 7/(48x) - (p-1)/(8x^2)
    dA2 = (dp / 32.0 + dp / (8.0 * x) - p / (8.0 * x**2)
           + 7.0 / (48.0 * x**2) - dp / (8.0 * x**2) + (p - 1.0) / (4.0 * x**3))
    # d(fRU)/dx: fRU = -p/4 - (p-1)/(2x)
    dB = -dp / 4.0 - dp / (2.0 * x) + (p - 1.0) / (2.0 * x**2)
    # d(fU)/dx: fU = p/2
    dC = dp / 2.0
    return dA1 + dA2 + xi * dB + xi**2 * dC


def dhC_vector_dx(x):
    """Derivative d/dx of h_C^(1)(x).

    d/dx h_C^(1) = phi'/4 + (6phi')/(6x) - (6phi-5)/(6x^2)
                   + phi'/x^2 - 2(phi-1)/x^3
                 = phi'/4 + phi'/x - (6phi-5)/(6x^2) + phi'/x^2 - 2(phi-1)/x^3
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"dhC_vector_dx: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"dhC_vector_dx: requires x >= 0, got {x}")
    if abs(x) < _TAYLOR_THRESH:
        coeffs = np.array([k * _HCV_TAYLOR[k] for k in range(1, _N_TAYLOR)])
        return _horner(coeffs, x)
    p = phi_fast(x)
    dp = dphi_dx_fast(x)
    return dp / 4.0 + (6.0 * dp) / (6.0 * x) - (6.0 * p - 5.0) / (6.0 * x**2) \
        + dp / (x**2) - 2.0 * (p - 1.0) / (x**3)


def dhR_vector_dx(x):
    """Derivative d/dx of h_R^(1)(x).

    d/dx h_R^(1) = -phi'/48 + (-6phi')/(72x) - (11-6phi)/(72x^2)
                   + 5phi'/(12x^2) - 10(phi-1)/(12x^3)
                 = -phi'/48 - phi'/(12x) - (11-6phi)/(72x^2)
                   + 5phi'/(12x^2) - 5(phi-1)/(6x^3)
    """
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"dhR_vector_dx: requires finite x, got {x}")
    if x < 0:
        raise ValueError(f"dhR_vector_dx: requires x >= 0, got {x}")
    if abs(x) < _TAYLOR_THRESH:
        coeffs = np.array([k * _HRV_TAYLOR[k] for k in range(1, _N_TAYLOR)])
        return _horner(coeffs, x)
    p = phi_fast(x)
    dp = dphi_dx_fast(x)
    return -dp / 48.0 - dp / (12.0 * x) - (11.0 - 6.0 * p) / (72.0 * x**2) \
        + 5.0 * dp / (12.0 * x**2) - 5.0 * (p - 1.0) / (6.0 * x**3)


# =============================================================================
# TAYLOR COEFFICIENT ACCESS (for MR-1, MR-6)
# =============================================================================

def get_taylor_coefficients(form_factor, n_terms=None):
    """Export Taylor coefficients for a form factor.

    The form factor is expanded as f(x) = sum_{k=0}^{N-1} c_k x^k.
    Returns array of c_k.

    Parameters:
        form_factor: one of 'hC_scalar', 'hC_dirac', 'hR_dirac',
                     'hR_scalar_A', 'hR_scalar_B', 'hR_scalar_C', 'phi'
        n_terms: number of terms to return (default: all precomputed)

    Returns:
        numpy array of Taylor coefficients.
    """
    coeff_map = {
        'hC_scalar': _HC0_TAYLOR,
        'hC_dirac': _HCD_TAYLOR,
        'hR_dirac': _HRD_TAYLOR,
        'hC_vector': _HCV_TAYLOR,
        'hR_vector': _HRV_TAYLOR,
        'hR_scalar_A': _HR0_A,
        'hR_scalar_B': _HR0_B,
        'hR_scalar_C': _HR0_C,
        'phi': _AN,
    }
    if form_factor not in coeff_map:
        raise ValueError(
            f"get_taylor_coefficients: unknown form factor '{form_factor}'. "
            f"Available: {list(coeff_map.keys())}"
        )
    coeffs = coeff_map[form_factor]
    if n_terms is not None:
        coeffs = coeffs[:n_terms]
    return np.array(coeffs, dtype=float)


def asymptotic_expansion(form_factor, x, n_terms=5):
    """Evaluate large-x asymptotic expansion of a form factor.

    These expansions include terms through O(1/x^3), derived from the
    full phi(x) ~ 2/x + 4/x^2 + 24/x^3 + ... expansion. Accuracy
    improves with increasing x. Use the exact hC_*/hR_* functions for
    production calculations requiring high precision.

    For x >> 1, phi(x) ~ 2/x, so form factors simplify to
    inverse power series in x.

    Parameters:
        form_factor: 'hC_scalar', 'hC_dirac', 'hR_dirac', 'hC_vector', 'hR_vector'
        x: large argument (x >> 1)
        n_terms: deprecated, currently ignored. The expansion uses a fixed
            number of leading terms for each form factor. Kept for backward
            compatibility; will be removed in a future version.

    Returns:
        Leading asymptotic value (float).
    """
    import warnings
    if n_terms != 5:
        warnings.warn(
            "asymptotic_expansion(): n_terms parameter is currently ignored. "
            "The expansion uses a fixed number of leading terms per form factor. "
            "This parameter will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
    x = float(x)
    if not np.isfinite(x):
        raise ValueError(f"asymptotic_expansion: requires finite x, got {x}")
    if x <= 0:
        raise ValueError(f"asymptotic_expansion: requires x > 0, got {x}")
    if form_factor == 'hC_scalar':
        # h_C^(0)(x) = 1/(12x) + (phi-1)/(2x^2)
        # phi ~ 2/x + 4/x^2 + O(1/x^3)
        # 1/(12x) stays
        # (phi-1)/(2x^2) = -1/(2x^2) + 1/x^3 + ...
        # Sum: 1/(12x) - 1/(2x^2) + 1/x^3
        return 1.0 / (12.0 * x) - 1.0 / (2.0 * x**2) + 1.0 / (x**3)
    elif form_factor == 'hC_dirac':
        # h_C^(1/2)(x) = (3phi-1)/(6x) + 2(phi-1)/x^2
        # phi ~ 2/x + 4/x^2 + O(1/x^3)
        # (3phi-1)/(6x) = -1/(6x) + 1/x^2 + 2/x^3 + ...
        # 2(phi-1)/x^2  = -2/x^2 + 4/x^3 + ...
        # Sum: -1/(6x) - 1/x^2 + 6/x^3
        return -1.0 / (6.0 * x) - 1.0 / (x**2) + 6.0 / x**3
    elif form_factor == 'hR_dirac':
        # h_R^(1/2)(x) = (3phi+2)/(36x) + 5(phi-1)/(6x^2)
        # phi ~ 2/x + 4/x^2 + O(1/x^3)
        # (3phi+2)/(36x) = 1/(18x) + 1/(6x^2) + 1/(3x^3) + ...
        # 5(phi-1)/(6x^2) = -5/(6x^2) + 5/(3x^3) + ...
        # Sum: 1/(18x) - 2/(3x^2) + 2/x^3
        return 1.0 / (18.0 * x) - 2.0 / (3.0 * x**2) + 2.0 / x**3
    elif form_factor == 'hC_vector':
        # h_C^(1)(x) = phi/4 + (6phi-5)/(6x) + (phi-1)/x^2
        # phi ~ 2/x + 4/x^2 + 24/x^3 + O(1/x^4)
        # phi/4           = 1/(2x) + 1/x^2 + 6/x^3 + ...
        # (6phi-5)/(6x)   = -5/(6x) + 2/x^2 + 4/x^3 + ...
        # (phi-1)/x^2     = -1/x^2 + 2/x^3 + ...
        # Sum: -1/(3x) + 2/x^2 + 12/x^3
        return -1.0 / (3.0 * x) + 2.0 / (x**2) + 12.0 / (x**3)
    elif form_factor == 'hR_vector':
        # h_R^(1)(x) ~ 1/(9x) - 2/(3x^2)  (Benchmark B6)
        # Note: 1/x^3 coefficient vanishes (interesting UV property).
        return 1.0 / (9.0 * x) - 2.0 / (3.0 * x**2)
    else:
        raise ValueError(f"asymptotic_expansion: form factor '{form_factor}' not available")
