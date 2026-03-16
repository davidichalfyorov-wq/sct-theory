# ruff: noqa: E402, I001
"""
MR-1 sub-task (c): Lorentzian continuation of SCT form factors.

Provides arbitrary-precision evaluations of:
  - phi(-x) for x > 0 (the Lorentzian master function)
  - Per-spin h_C, h_R on the Lorentzian axis
  - Pi_TT, Pi_s on the Lorentzian axis (physical propagator denominators)
  - Spectral function rho(sigma) for the Kallen-Lehmann representation

All functions use mpmath for arbitrary precision.  The Wick rotation
convention is z_E -> z_L = -z_E, so Lorentzian momenta with k^2 > 0
(timelike) correspond to evaluating Euclidean quantities at z = -z_L.

Sign conventions:
  Lorentzian signature (-,+,+,+)
  Box_E -> -Box_L
  z = Box/Lambda^2   (Euclidean: z > 0 for spacelike)
  z_L = k^2/Lambda^2 (Lorentzian: z_L > 0 for timelike)
  Mapping: form_factor(z_E) -> form_factor(-z_L) for Lorentzian

Branch cut: principal branch of sqrt(z) with cut along (-inf, 0).
For z < 0: sqrt(-x + i*eps) -> i*sqrt(x) as eps -> 0+.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mpmath as mp

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.nt2_entire_function import (
    F1_total_complex,
    F2_total_complex,
    hC_dirac_complex,
    hC_scalar_complex,
    hC_vector_complex,
    hR_dirac_complex,
    hR_scalar_complex,
    hR_vector_complex,
    phi_complex_mp,
    phi_series,
    phi_series_coefficient,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "mr1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALPHA_C = mp.mpf(13) / 120
LOCAL_C2 = 2 * ALPHA_C  # 13/60
DEFAULT_DPS = 100


# ===================================================================
# SECTION 1: Lorentzian master function phi(-x)
# ===================================================================

def phi_lorentzian_closed(x: float | mp.mpf, dps: int = DEFAULT_DPS) -> mp.mpf:
    """
    Evaluate phi(-x) for x > 0 using the closed-form expression.

    Derivation (from principal branch of sqrt):
      phi(z) = e^{-z/4} * sqrt(pi/z) * erfi(sqrt(z)/2)

    For z = -x (x > 0):
      sqrt(-x) = i*sqrt(x)  [principal branch, Im > 0]
      erfi(i*sqrt(x)/2) = i * erf(sqrt(x)/2)
      sqrt(pi/(-x)) = sqrt(pi/x) * (1/i) = -i*sqrt(pi/x)

    Result:
      phi(-x) = e^{x/4} * (-i*sqrt(pi/x)) * i * erf(sqrt(x)/2)
              = e^{x/4} * sqrt(pi/x) * erf(sqrt(x)/2)

    This is real and positive for all x > 0.

    Parameters
    ----------
    x : positive real number
    dps : decimal places of precision

    Returns
    -------
    phi(-x) as a real mpf value
    """
    mp.mp.dps = dps
    x_mp = mp.mpf(x)
    if x_mp <= 0:
        raise ValueError(f"x must be positive, got {x_mp}")
    if x_mp < mp.mpf("1e-30"):
        # Use Taylor: phi(-x) = sum_{n>=0} n!/(2n+1)! * x^n
        return phi_lorentzian_taylor(x_mp, dps=dps)
    return mp.exp(x_mp / 4) * mp.sqrt(mp.pi / x_mp) * mp.erf(mp.sqrt(x_mp) / 2)


def phi_lorentzian_taylor(
    x: float | mp.mpf, dps: int = DEFAULT_DPS, n_terms: int = 60
) -> mp.mpf:
    """
    Evaluate phi(-x) for x > 0 using the Taylor series.

    phi(-x) = sum_{n=0}^{inf} n! / (2n+1)! * x^n

    Since a_n(phi) = (-1)^n * n! / (2n+1)!, we have:
      phi(-x) = sum_n a_n * (-x)^n = sum_n (-1)^n * (-1)^n * n!/(2n+1)! * x^n
              = sum_n n!/(2n+1)! * x^n

    All coefficients are positive, so the series converges monotonically
    for x > 0.

    Parameters
    ----------
    x : positive real number
    n_terms : number of Taylor terms
    dps : decimal places of precision

    Returns
    -------
    phi(-x) as a real mpf value
    """
    mp.mp.dps = dps
    x_mp = mp.mpf(x)
    total = mp.mpf(0)
    for n in range(n_terms):
        coeff = mp.factorial(n) / mp.factorial(2 * n + 1)
        total += coeff * x_mp**n
    return total


def phi_lorentzian_integral(
    x: float | mp.mpf, dps: int = DEFAULT_DPS
) -> mp.mpf:
    """
    Evaluate phi(-x) for x > 0 using the proper-time integral.

    phi(-x) = int_0^1 e^{+xi*(1-xi)*x} d(xi)

    Since xi*(1-xi) in [0, 1/4] for xi in [0,1], this integral
    is well-defined and bounded: 1 <= phi(-x) <= e^{x/4}.

    Parameters
    ----------
    x : positive real number
    dps : decimal places of precision

    Returns
    -------
    phi(-x) as a real mpf value
    """
    mp.mp.dps = dps
    x_mp = mp.mpf(x)
    return mp.quad(
        lambda xi: mp.exp(xi * (1 - xi) * x_mp),
        [0, 1],
    )


def phi_lorentzian(
    x: float | mp.mpf, dps: int = DEFAULT_DPS
) -> mp.mpf:
    """
    Evaluate phi(-x) for x > 0 with cross-validation.

    Uses the closed-form as primary and Taylor series as backup
    for small x. Returns a real positive value.

    Parameters
    ----------
    x : positive real number
    dps : decimal places of precision

    Returns
    -------
    phi(-x) as a real mpf value
    """
    mp.mp.dps = dps
    x_mp = mp.mpf(x)
    if x_mp <= 0:
        raise ValueError(f"x must be positive, got {x_mp}")
    if x_mp < mp.mpf("1e-10"):
        return phi_lorentzian_taylor(x_mp, dps=dps, n_terms=80)
    return phi_lorentzian_closed(x_mp, dps=dps)


# ===================================================================
# SECTION 2: Per-spin form factors on the Lorentzian axis
# ===================================================================
# IMPORTANT: We cannot simply call the nt2 complex form factors at z=-x
# because phi_complex_mp(z) has a branch-cut sign error for z < 0
# (see SECTION 5 for details).  The nt2 functions use series expansion
# for |z| < 0.5 (correct) and the closed form for |z| >= 0.5
# (sign-flipped for z < 0).
#
# Instead, we directly implement the Lorentzian form factors using the
# correct phi_lorentzian = phi(-x) > 0 from SECTION 1.

def _hC_scalar_from_phi(x: mp.mpf, phi_val: mp.mpf) -> mp.mpf:
    """h_C^{(0)}(z=-x) using correct phi(-x)."""
    # h_C^{(0)}(z) = 1/(12*z) + (phi(z)-1)/(2*z^2)
    # At z=-x: 1/(12*(-x)) + (phi(-x)-1)/(2*x^2)
    return mp.mpf(-1) / (12 * x) + (phi_val - 1) / (2 * x**2)


def _hC_dirac_from_phi(x: mp.mpf, phi_val: mp.mpf) -> mp.mpf:
    """h_C^{(1/2)}(z=-x) using correct phi(-x)."""
    # h_C^{(1/2)}(z) = (3*phi-1)/(6*z) + 2*(phi-1)/z^2
    # At z=-x: (3*phi(-x)-1)/(6*(-x)) + 2*(phi(-x)-1)/x^2
    return -(3 * phi_val - 1) / (6 * x) + 2 * (phi_val - 1) / x**2


def _hC_vector_from_phi(x: mp.mpf, phi_val: mp.mpf) -> mp.mpf:
    """h_C^{(1)}(z=-x) using correct phi(-x)."""
    # h_C^{(1)}(z) = phi/4 + (6*phi-5)/(6*z) + (phi-1)/z^2
    # At z=-x: phi(-x)/4 + (6*phi(-x)-5)/(6*(-x)) + (phi(-x)-1)/x^2
    return phi_val / 4 - (6 * phi_val - 5) / (6 * x) + (phi_val - 1) / x**2


def _hR_scalar_from_phi(x: mp.mpf, phi_val: mp.mpf, xi: mp.mpf) -> mp.mpf:
    """h_R^{(0)}(z=-x, xi) using correct phi(-x)."""
    # From nt2: h_R^(0) = f_ric/3 + f_r + xi*f_ru + xi^2*f_u
    # where:
    #   f_ric = 1/(6z) + (phi-1)/z^2
    #   f_r = phi/32 + phi/(8z) - 7/(48z) - (phi-1)/(8z^2)
    #   f_ru = -phi/4 - (phi-1)/(2z)
    #   f_u = phi/2
    z = -x
    f_ric = mp.mpf(1) / (6 * z) + (phi_val - 1) / z**2
    f_r = phi_val / 32 + phi_val / (8 * z) - mp.mpf(7) / (48 * z) - (phi_val - 1) / (8 * z**2)
    f_ru = -phi_val / 4 - (phi_val - 1) / (2 * z)
    f_u = phi_val / 2
    return f_ric / 3 + f_r + xi * f_ru + xi**2 * f_u


def _hR_dirac_from_phi(x: mp.mpf, phi_val: mp.mpf) -> mp.mpf:
    """h_R^{(1/2)}(z=-x) using correct phi(-x)."""
    # h_R^{(1/2)}(z) = (3*phi+2)/(36*z) + 5*(phi-1)/(6*z^2)
    return -(3 * phi_val + 2) / (36 * x) + 5 * (phi_val - 1) / (6 * x**2)


def _hR_vector_from_phi(x: mp.mpf, phi_val: mp.mpf) -> mp.mpf:
    """h_R^{(1)}(z=-x) using correct phi(-x)."""
    # h_R^{(1)}(z) = -phi/48 + (11-6*phi)/(72*z) + 5*(phi-1)/(12*z^2)
    return -phi_val / 48 - (11 - 6 * phi_val) / (72 * x) + 5 * (phi_val - 1) / (12 * x**2)


def hC_lorentzian(
    x: float | mp.mpf,
    spin: str,
    dps: int = DEFAULT_DPS,
) -> mp.mpf:
    """
    Evaluate h_C^{(s)}(z=-x) for spin s and x > 0 (Lorentzian axis).

    Uses the correct phi(-x) > 0 (real, positive) to avoid the branch-cut
    sign error in phi_complex_mp.

    Parameters
    ----------
    x : positive real Lorentzian momentum parameter x_L = k^2/Lambda^2
    spin : '0', '1/2', or '1'
    dps : decimal places of precision

    Returns
    -------
    h_C^{(s)}(z=-x) as a real mpf value
    """
    mp.mp.dps = dps
    x_mp = mp.mpf(x)
    if x_mp <= 0:
        raise ValueError(f"x must be positive, got {x_mp}")
    phi_val = phi_lorentzian(x_mp, dps=dps)
    if spin == '0':
        return _hC_scalar_from_phi(x_mp, phi_val)
    elif spin == '1/2':
        return _hC_dirac_from_phi(x_mp, phi_val)
    elif spin == '1':
        return _hC_vector_from_phi(x_mp, phi_val)
    else:
        raise ValueError(f"Unknown spin: {spin}. Use '0', '1/2', or '1'.")


def hR_lorentzian(
    x: float | mp.mpf,
    spin: str,
    xi: float | mp.mpf = 0.0,
    dps: int = DEFAULT_DPS,
) -> mp.mpf:
    """
    Evaluate h_R^{(s)}(z=-x) for spin s and x > 0 (Lorentzian axis).

    Parameters
    ----------
    x : positive real Lorentzian momentum parameter x_L = k^2/Lambda^2
    spin : '0', '1/2', or '1'
    xi : non-minimal coupling (only for spin 0)
    dps : decimal places of precision

    Returns
    -------
    h_R^{(s)}(z=-x) as a real mpf value
    """
    mp.mp.dps = dps
    x_mp = mp.mpf(x)
    if x_mp <= 0:
        raise ValueError(f"x must be positive, got {x_mp}")
    phi_val = phi_lorentzian(x_mp, dps=dps)
    xi_mp = mp.mpf(xi)
    if spin == '0':
        return _hR_scalar_from_phi(x_mp, phi_val, xi_mp)
    elif spin == '1/2':
        return _hR_dirac_from_phi(x_mp, phi_val)
    elif spin == '1':
        return _hR_vector_from_phi(x_mp, phi_val)
    else:
        raise ValueError(f"Unknown spin: {spin}. Use '0', '1/2', or '1'.")


# ===================================================================
# SECTION 3: Propagator denominators on the Lorentzian axis
# ===================================================================
#
# There are two regimes:
#   (a) General complex z (off the negative real axis): the nt2 form
#       factor functions work correctly because phi_complex_mp has no
#       branch-cut issue away from the negative real axis.
#   (b) Real Lorentzian axis z = -x_L (x_L > 0): we must use the
#       corrected Lorentzian form factors from SECTION 2 to avoid
#       the phi_complex_mp sign error.

# SM multiplicities (imported from nt2, but redefined for clarity)
_N_S = 4
_N_D = mp.mpf(45) / 2   # N_f/2 = 22.5
_N_V = 12


def _F1_lorentzian_direct(x: mp.mpf, dps: int = DEFAULT_DPS) -> mp.mpf:
    """
    Total F1 form factor at z=-x (Lorentzian, x > 0), using correct phi(-x).

    F1(z) = [N_s*hC_scalar + N_D*hC_dirac + N_v*hC_vector] / (16*pi^2)
    """
    mp.mp.dps = dps
    phi_val = phi_lorentzian(x, dps=dps)
    hC_s = _hC_scalar_from_phi(x, phi_val)
    hC_d = _hC_dirac_from_phi(x, phi_val)
    hC_v = _hC_vector_from_phi(x, phi_val)
    return (_N_S * hC_s + _N_D * hC_d + _N_V * hC_v) / (16 * mp.pi**2)


def _F2_lorentzian_direct(x: mp.mpf, xi: mp.mpf = mp.mpf(0), dps: int = DEFAULT_DPS) -> mp.mpf:
    """
    Total F2 form factor at z=-x (Lorentzian, x > 0), using correct phi(-x).
    """
    mp.mp.dps = dps
    phi_val = phi_lorentzian(x, dps=dps)
    hR_s = _hR_scalar_from_phi(x, phi_val, xi)
    hR_d = _hR_dirac_from_phi(x, phi_val)
    hR_v = _hR_vector_from_phi(x, phi_val)
    return (_N_S * hR_s + _N_D * hR_d + _N_V * hR_v) / (16 * mp.pi**2)


def _F1_hat_lorentzian(x: mp.mpf, dps: int = DEFAULT_DPS) -> mp.mpf:
    """Normalized F1_hat(z=-x) = F1(-x)/F1(0)."""
    mp.mp.dps = dps
    f1_0 = F1_total_complex(0, dps=dps)  # F1(0) = alpha_C/(16*pi^2)
    f1_neg_x = _F1_lorentzian_direct(x, dps=dps)
    return f1_neg_x / mp.re(f1_0)


def _F2_hat_lorentzian(x: mp.mpf, xi: mp.mpf = mp.mpf(0), dps: int = DEFAULT_DPS) -> mp.mpf:
    """Normalized F2_hat(z=-x, xi) = F2(-x, xi)/F2(0, xi)."""
    mp.mp.dps = dps
    f2_0 = F2_total_complex(0, xi=float(xi), dps=dps)
    if abs(f2_0) < mp.mpf("1e-40"):
        return mp.mpf(1)
    f2_neg_x = _F2_lorentzian_direct(x, xi=xi, dps=dps)
    return f2_neg_x / mp.re(f2_0)


def Pi_TT_complex(z: complex | mp.mpc, xi: float = 0.0, dps: int = DEFAULT_DPS) -> mp.mpc:
    """
    Spin-2 TT propagator denominator for arbitrary complex z.

    Pi_TT(z) = 1 + (13/60) * z * F1_hat(z)

    For z on or near the negative real axis, delegates to the corrected
    Lorentzian-axis evaluator.  For general complex z, uses the nt2 form
    factors (which are correct away from the negative real axis).

    Parameters
    ----------
    z : complex argument (Euclidean convention)
    xi : non-minimal coupling
    dps : decimal places of precision
    """
    mp.mp.dps = dps
    z_mp = mp.mpc(z)

    # Check if z is on the negative real axis (where phi_complex_mp is wrong)
    if abs(mp.im(z_mp)) < mp.mpf("1e-30") and mp.re(z_mp) < -mp.mpf("1e-30"):
        x = -mp.re(z_mp)
        f1_hat = _F1_hat_lorentzian(x, dps=dps)
        return mp.mpc(1 + LOCAL_C2 * (-x) * f1_hat)

    # General complex z: use nt2 functions (correct off negative real axis)
    f0 = F1_total_complex(0, xi=xi, dps=dps)
    if abs(f0) < mp.mpf("1e-40"):
        return mp.mpc(1)
    f1_hat = F1_total_complex(z_mp, xi=xi, dps=dps) / f0
    return 1 + LOCAL_C2 * z_mp * f1_hat


def Pi_scalar_complex(z: complex | mp.mpc, xi: float = 0.0, dps: int = DEFAULT_DPS) -> mp.mpc:
    """
    Spin-0 scalar propagator denominator for arbitrary complex z.

    Pi_s(z, xi) = 1 + 6*(xi - 1/6)^2 * z * F2_hat(z, xi)

    At xi = 1/6: Pi_s = 1 (scalar mode decouples).

    Parameters
    ----------
    z : complex argument (Euclidean convention)
    xi : non-minimal coupling
    dps : decimal places of precision
    """
    mp.mp.dps = dps
    z_mp = mp.mpc(z)
    xi_mp = mp.mpf(xi)
    coeff = 6 * (xi_mp - mp.mpf(1) / 6) ** 2
    if abs(coeff) < mp.mpf("1e-20"):
        return mp.mpc(1)

    # Negative real axis: use corrected Lorentzian evaluator
    if abs(mp.im(z_mp)) < mp.mpf("1e-30") and mp.re(z_mp) < -mp.mpf("1e-30"):
        x = -mp.re(z_mp)
        f2_hat = _F2_hat_lorentzian(x, xi=xi_mp, dps=dps)
        return mp.mpc(1 + coeff * (-x) * f2_hat)

    # General complex z
    f0 = F2_total_complex(0, xi=xi, dps=dps)
    if abs(f0) < mp.mpf("1e-40"):
        return mp.mpc(1)
    f2_hat = F2_total_complex(z_mp, xi=xi, dps=dps) / f0
    return 1 + coeff * z_mp * f2_hat


def Pi_TT_lorentzian(z_L: float | mp.mpf, dps: int = DEFAULT_DPS) -> mp.mpf:
    """
    Spin-2 TT denominator on the Lorentzian axis.

    For Lorentzian momentum z_L > 0 (timelike, k^2 > 0):
      Pi_TT^{Lor}(z_L) = 1 - (13/60)*z_L*F1_hat(-z_L)

    Parameters
    ----------
    z_L : positive real Lorentzian momentum squared (in units of Lambda^2)
    dps : decimal places of precision

    Returns
    -------
    Pi_TT(-z_L) as a real value
    """
    mp.mp.dps = dps
    x = mp.mpf(z_L)
    f1_hat = _F1_hat_lorentzian(x, dps=dps)
    return 1 - LOCAL_C2 * x * f1_hat


def Pi_s_lorentzian(
    z_L: float | mp.mpf, xi: float = 0.0, dps: int = DEFAULT_DPS
) -> mp.mpf:
    """
    Spin-0 scalar denominator on the Lorentzian axis.

    Parameters
    ----------
    z_L : positive real Lorentzian momentum squared (in units of Lambda^2)
    xi : non-minimal coupling
    dps : decimal places of precision
    """
    mp.mp.dps = dps
    xi_mp = mp.mpf(xi)
    coeff = 6 * (xi_mp - mp.mpf(1) / 6) ** 2
    if abs(coeff) < mp.mpf("1e-20"):
        return mp.mpf(1)
    x = mp.mpf(z_L)
    f2_hat = _F2_hat_lorentzian(x, xi=xi_mp, dps=dps)
    return 1 - coeff * x * f2_hat


# ===================================================================
# SECTION 4: Spectral functions (Kallen-Lehmann)
# ===================================================================

def spectral_function_TT(
    sigma: float | mp.mpf,
    Lambda: float | mp.mpf = 1.0,
    dps: int = DEFAULT_DPS,
    eps: float = 1e-20,
) -> mp.mpf:
    """
    Spectral function rho_TT(sigma) for the spin-2 sector.

    From the Kallen-Lehmann representation:
      G_TT(k^2) = int_0^inf rho(sigma) / (k^2 - sigma + i*eps) d(sigma)

    The spectral function is:
      rho_TT(sigma) = -(1/pi) * Im[G_TT(sigma + i*eps)]
                    = -(1/pi) * Im[1 / (sigma * Pi_TT(sigma/Lambda^2 + i*eps))]

    For sigma > 0, Pi_TT is evaluated just above the real axis.
    On the real Euclidean axis, Pi_TT is real, so we need to use
    the Lorentzian prescription: evaluate at z = -sigma/Lambda^2 + i*eps.

    In the Lorentzian theory, k^2 = sigma > 0 maps to the Euclidean
    z = -sigma/Lambda^2, and the Feynman prescription adds +i*eps.

    Parameters
    ----------
    sigma : spectral parameter (mass squared), must be > 0
    Lambda : cutoff scale
    eps : Feynman i*epsilon (small positive number)
    dps : decimal places of precision

    Returns
    -------
    rho_TT(sigma) as a real number.  Positive = normal state, negative = ghost.
    """
    mp.mp.dps = dps
    sigma_mp = mp.mpf(sigma)
    Lambda_mp = mp.mpf(Lambda)
    eps_mp = mp.mpf(eps)

    if sigma_mp <= 0:
        return mp.mpf(0)

    # Euclidean argument with Feynman prescription
    z = -sigma_mp / Lambda_mp**2 + mp.mpc(0, eps_mp)
    Pi_val = Pi_TT_complex(z, dps=dps)
    G_val = 1 / (sigma_mp * Pi_val)
    return -mp.im(G_val) / mp.pi


def spectral_function_scalar(
    sigma: float | mp.mpf,
    xi: float = 0.0,
    Lambda: float | mp.mpf = 1.0,
    dps: int = DEFAULT_DPS,
    eps: float = 1e-20,
) -> mp.mpf:
    """
    Spectral function rho_s(sigma) for the spin-0 sector.

    Same structure as rho_TT but using Pi_scalar.

    Parameters
    ----------
    sigma : spectral parameter (mass squared), must be > 0
    xi : non-minimal coupling
    Lambda : cutoff scale
    eps : Feynman i*epsilon
    dps : decimal places of precision

    Returns
    -------
    rho_s(sigma) as a real number.
    """
    mp.mp.dps = dps
    sigma_mp = mp.mpf(sigma)
    Lambda_mp = mp.mpf(Lambda)
    eps_mp = mp.mpf(eps)

    if sigma_mp <= 0:
        return mp.mpf(0)

    z = -sigma_mp / Lambda_mp**2 + mp.mpc(0, eps_mp)
    Pi_val = Pi_scalar_complex(z, xi=xi, dps=dps)
    G_val = 1 / (sigma_mp * Pi_val)
    return -mp.im(G_val) / mp.pi


# ===================================================================
# SECTION 5: Verification routines
# ===================================================================

def verify_phi_lorentzian(
    test_points: list[float] | None = None,
    dps: int = DEFAULT_DPS,
) -> list[dict]:
    """
    Cross-validate phi(-x) between closed form, Taylor, and integral.

    NOTE: phi_complex_mp(z) from nt2_entire_function uses the principal branch
    sqrt(-x) = i*sqrt(x), which gives phi_complex_mp(-x) = -phi_lorentzian(x)
    for x > 0.  This is a known branch-cut artefact, NOT a bug in the new code.
    The three independent Lorentzian methods (closed, Taylor, integral) agree
    with each other and give the correct POSITIVE result.

    Parameters
    ----------
    test_points : list of positive x values to test
    dps : decimal places of precision

    Returns
    -------
    List of dicts with comparison data for each test point.
    """
    mp.mp.dps = dps
    if test_points is None:
        test_points = [0.01, 0.1, 1.0, 10.0, 100.0]

    results = []
    for x_val in test_points:
        x = mp.mpf(x_val)
        closed = phi_lorentzian_closed(x, dps=dps)
        taylor = phi_lorentzian_taylor(x, dps=dps, n_terms=80)
        integral = phi_lorentzian_integral(x, dps=dps)

        # phi_complex_mp(-x) gives -phi_lorentzian(x) due to branch cut of sqrt.
        # Compare absolute values to confirm the magnitude matches.
        complex_val = mp.re(phi_complex_mp(-x, dps=dps))
        # Expected: complex_val ~ -closed (sign flip from branch cut)
        err_magnitude = float(abs(abs(complex_val) - abs(closed)))

        err_taylor_closed = float(abs(taylor - closed))
        err_integral_closed = float(abs(integral - closed))

        results.append({
            "x": float(x),
            "phi_closed": str(closed),
            "phi_taylor": str(taylor),
            "phi_integral": str(integral),
            "phi_complex_mp_raw": str(complex_val),
            "phi_complex_mp_note": "sign-flipped due to sqrt branch cut; magnitude agrees",
            "err_taylor_vs_closed": err_taylor_closed,
            "err_integral_vs_closed": err_integral_closed,
            "err_magnitude_complex_vs_closed": err_magnitude,
            "phi_positive": float(closed) > 0,
            "phi_geq_1": float(closed) >= 1.0 - 1e-30,
            "phi_leq_exp_x_over_4": float(closed) <= float(mp.exp(x / 4)) + 1e-30,
        })
    return results


def verify_lorentzian_propagator(dps: int = DEFAULT_DPS) -> dict:
    """
    Check key properties of the Lorentzian propagator denominators.

    Returns
    -------
    Dict with Pi_TT and Pi_s values at key points plus consistency checks.
    """
    mp.mp.dps = dps

    # Pi_TT(0) must be 1
    # Note: x = 1e-30 causes overflow in 1/x form-factor terms.
    # Use x = 1e-4 which is small enough to approximate z->0 limit
    # while avoiding numerical instability.
    pi_tt_0 = Pi_TT_lorentzian(mp.mpf("1e-4"), dps=dps)

    # Pi_TT on the Lorentzian axis at several points
    z_L_points = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    pi_tt_values = {}
    for z_L in z_L_points:
        val = Pi_TT_lorentzian(z_L, dps=dps)
        pi_tt_values[str(z_L)] = float(val)

    # Check: does Pi_TT have a zero on the Lorentzian axis?
    # This would be at z_L > 0 such that Pi_TT(-z_L) = 0
    # i.e., 1 - (13/60)*z_L*F1_hat(-z_L) = 0
    lorentzian_tt_zeros = []
    z_prev = mp.mpf("0.01")
    val_prev = Pi_TT_lorentzian(z_prev, dps=dps)
    for z_L_test in [mp.mpf(i) / 10 for i in range(2, 1000)]:
        val = Pi_TT_lorentzian(z_L_test, dps=dps)
        if val_prev * val < 0:
            try:
                root = mp.findroot(
                    lambda t: Pi_TT_lorentzian(t, dps=dps),
                    (z_prev, z_L_test),
                )
                lorentzian_tt_zeros.append(float(mp.re(root)))
            except (ValueError, ZeroDivisionError):
                pass
        z_prev = z_L_test
        val_prev = val

    # Pi_s at xi = 0 (non-conformal)
    pi_s_values = {}
    for z_L in z_L_points:
        val = Pi_s_lorentzian(z_L, xi=0.0, dps=dps)
        pi_s_values[str(z_L)] = float(val)

    # Pi_s at xi = 1/6 (conformal: should be identically 1)
    # Use mpf(1)/6 to avoid Python float precision issues with 1.0/6
    xi_conformal = float(mp.mpf(1) / 6)
    pi_s_conformal = Pi_s_lorentzian(mp.mpf(5), xi=xi_conformal, dps=dps)

    return {
        "Pi_TT_at_0": float(pi_tt_0),
        "Pi_TT_lorentzian": pi_tt_values,
        "Pi_TT_lorentzian_zeros": lorentzian_tt_zeros,
        "Pi_s_lorentzian_xi0": pi_s_values,
        "Pi_s_conformal_check": float(pi_s_conformal),
        "Pi_s_conformal_is_1": abs(float(pi_s_conformal) - 1.0) < 1e-20,
    }


def verify_known_euclidean_zero(dps: int = DEFAULT_DPS) -> dict:
    """
    Verify that the known Euclidean ghost pole at z0 ~ 2.41484 is recovered.

    Returns
    -------
    Dict with z0, Pi_TT(z0), and residue.
    """
    mp.mp.dps = dps

    # Find z0 on the positive real axis
    z_left = mp.mpf("2.0")
    z_right = mp.mpf("3.0")
    z0 = mp.findroot(
        lambda z: mp.re(Pi_TT_complex(z, dps=dps)),
        (z_left, z_right),
    )
    z0 = mp.re(z0)

    # Pi_TT(z0) should be ~0
    Pi_at_z0 = Pi_TT_complex(z0, dps=dps)

    # Derivative
    h = mp.mpf("1e-10")
    Pi_deriv = mp.re(Pi_TT_complex(z0 + h, dps=dps) - Pi_TT_complex(z0 - h, dps=dps)) / (2 * h)

    # Residue
    R2 = 1 / (z0 * Pi_deriv)

    return {
        "z0": str(z0),
        "z0_float": float(z0),
        "Pi_TT_at_z0": float(abs(Pi_at_z0)),
        "Pi_TT_prime_z0": str(Pi_deriv),
        "R2": str(R2),
        "R2_float": float(R2),
        "is_ghost": float(R2) < 0,
        "suppression_vs_stelle": abs(float(R2)) / 1.0,
    }


# ===================================================================
# SECTION 6: Serialization
# ===================================================================

def run_all_lorentzian_checks(dps: int = DEFAULT_DPS) -> dict:
    """Run all verification checks and return combined results."""
    results = {
        "phi_lorentzian_verification": verify_phi_lorentzian(dps=dps),
        "propagator_verification": verify_lorentzian_propagator(dps=dps),
        "euclidean_zero_recovery": verify_known_euclidean_zero(dps=dps),
    }
    return results


def save_results(results: dict, filename: str = "mr1_lorentzian_results.json") -> Path:
    """Save results to JSON."""
    output_path = RESULTS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (mp.mpf, mp.mpc)):
            return str(obj)
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Cannot serialize {type(obj)}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    return output_path


# ===================================================================
# CLI entry point
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MR-1 Lorentzian continuation analysis")
    parser.add_argument("--dps", type=int, default=50, help="Decimal places of precision")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    print(f"MR-1 Lorentzian analysis (dps={args.dps})")
    print("=" * 60)

    # 1. phi(-x) verification
    print("\n--- phi(-x) verification ---")
    phi_results = verify_phi_lorentzian(dps=args.dps)
    for r in phi_results:
        x = r["x"]
        err_ct = r["err_taylor_vs_closed"]
        err_ci = r["err_integral_vs_closed"]
        err_mag = r["err_magnitude_complex_vs_closed"]
        print(f"  x={x:8.2f}: err(Taylor-closed)={err_ct:.2e}, "
              f"err(integral-closed)={err_ci:.2e}, "
              f"err(|complex|-|closed|)={err_mag:.2e}, "
              f"positive={r['phi_positive']}, "
              f"bounds={r['phi_geq_1']} <= phi <= e^{{x/4}}={r['phi_leq_exp_x_over_4']}")

    # 2. Propagator verification
    print("\n--- Lorentzian propagator verification ---")
    prop_results = verify_lorentzian_propagator(dps=args.dps)
    print(f"  Pi_TT(0) = {prop_results['Pi_TT_at_0']:.15f} (should be ~1)")
    print(f"  Pi_s(5, xi=1/6) = {prop_results['Pi_s_conformal_check']:.15f} (should be 1)")
    print(f"  Lorentzian TT zeros: {prop_results['Pi_TT_lorentzian_zeros']}")
    print(f"  Pi_TT on Lorentzian axis:")
    for z_L, val in prop_results["Pi_TT_lorentzian"].items():
        print(f"    z_L={z_L}: Pi_TT={val:.10f}")

    # 3. Euclidean zero recovery
    print("\n--- Euclidean ghost pole recovery ---")
    zero_results = verify_known_euclidean_zero(dps=args.dps)
    print(f"  z0 = {zero_results['z0']}")
    print(f"  |Pi_TT(z0)| = {zero_results['Pi_TT_at_z0']:.2e}")
    print(f"  R2 = {zero_results['R2']}")
    print(f"  ghost = {zero_results['is_ghost']}")
    print(f"  |R2|/|R2_Stelle| = {zero_results['suppression_vs_stelle']:.4f}")

    if args.save:
        all_results = run_all_lorentzian_checks(dps=args.dps)
        out = save_results(all_results)
        print(f"\nResults saved to {out}")
