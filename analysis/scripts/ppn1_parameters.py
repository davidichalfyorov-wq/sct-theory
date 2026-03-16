# ruff: noqa: E402, I001
"""
PPN-1 parameters for SCT: exact nonlocal and local Yukawa approximations.

Three levels of approximation are implemented:

Level 1 (exact nonlocal): The gravitational potentials Phi(r) and Psi(r)
    are computed as Fourier–Bessel integrals over the full SCT propagator
    denominators Pi_TT(z) and Pi_s(z,xi).  The integrand is decomposed
    into a GR piece (giving -GM/r) plus a convergent correction integral.
    The Pi_TT pole at z0 ~ 2.41 is handled by residue decomposition.

Level 2 (local Yukawa): The propagator denominators are replaced by their
    small-z pole approximants, reducing the potentials to the familiar
    Stelle-like form with two Yukawa corrections.

Level 3 (nonlinear PPN): Requires O(h^2) field equations from NT-4b.
    All quantities at this level are marked NOT_DERIVED.

References:
    - Stelle (1977), Phys. Rev. D 16, 953
    - Stelle (1978), Gen. Rel. Grav. 9, 353
    - Edholm, Koshelev, Mazumdar (2016), Phys. Rev. D 94, 104033
    - Bertotti, Iess, Tortora (2003), Nature 425, 374
    - Verma, Fienga, Laskar+ (2014), A&A 561, A115
    - Adelberger, Heckel, Nelson (2003), Ann. Rev. Nucl. Part. Sci. 53, 77
    - Touboul+ (2022), Phys. Rev. Lett. 129, 121102
    - Williams, Turyshev, Boggs (2004), Phys. Rev. Lett. 93, 261101
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from sct_tools.plotting import SCT_COLORS, create_figure, init_style, save_figure  # noqa: E402

# Direct imports to avoid circular import chain through sct_tools.__init__
from scripts.nt2_entire_function import F1_total_complex, F2_total_complex  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "ppn1"
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Physical and mathematical constants
# ---------------------------------------------------------------------------
ALPHA_C = mp.mpf(13) / 120
LOCAL_C2 = 2 * ALPHA_C  # 13/60

# Unit conversions (CODATA 2022 compatible, via scipy.constants)
HBAR_C_EV_M = mp.mpf("1.9732705e-7")  # hbar*c in eV*m
M_TO_EV_INV = 1 / HBAR_C_EV_M  # 1 m in eV^{-1}
AU_M = mp.mpf("1.495978707e11")  # 1 AU in meters
AU_EV_INV = AU_M * M_TO_EV_INV  # 1 AU in eV^{-1}


# ---------------------------------------------------------------------------
# Propagator denominators (self-contained, no circular imports)
# ---------------------------------------------------------------------------
def _F1_at_zero(*, xi: float = 0.0, dps: int = 100) -> mp.mpc:
    """F1(0) — the local limit of the total Weyl form factor."""
    return F1_total_complex(0, xi=xi, dps=dps)


def _F1_shape(z: mp.mpc, *, xi: float = 0.0, dps: int = 100) -> mp.mpc:
    """F1_hat(z) = F1(z)/F1(0) — normalized shape factor."""
    f0 = _F1_at_zero(xi=xi, dps=dps)
    if abs(f0) < mp.mpf("1e-40"):
        return mp.mpc(1)
    return F1_total_complex(z, xi=xi, dps=dps) / f0


def _F2_at_zero(*, xi: float = 0.0, dps: int = 100) -> mp.mpc:
    """F2(0,xi) — the local limit of the total R^2 form factor."""
    return F2_total_complex(0, xi=xi, dps=dps)


def _F2_shape(z: mp.mpc, *, xi: float = 0.0, dps: int = 100) -> mp.mpc:
    """F2_hat(z) = F2(z)/F2(0) — normalized shape factor."""
    f0 = _F2_at_zero(xi=xi, dps=dps)
    if abs(f0) < mp.mpf("1e-40"):
        return mp.mpc(1)
    return F2_total_complex(z, xi=xi, dps=dps) / f0


def _Pi_TT(z: mp.mpc | float, *, xi: float = 0.0, dps: int = 100) -> mp.mpc:
    """Spin-2 TT denominator: Pi_TT(z) = 1 + c2 * z * F1_hat(z)."""
    mp.mp.dps = dps
    z_mp = mp.mpc(z)
    return 1 + LOCAL_C2 * z_mp * _F1_shape(z_mp, xi=xi, dps=dps)


def _scalar_mode_coefficient(xi: float) -> mp.mpf:
    """3c1+c2 = 6(xi-1/6)^2. Returns 0 at conformal coupling."""
    xi_mp = mp.mpf(xi)
    if abs(xi_mp - mp.mpf(1) / 6) < mp.mpf("1e-14"):
        return mp.mpf(0)
    return 6 * (xi_mp - mp.mpf(1) / 6) ** 2


def _Pi_scalar(z: mp.mpc | float, *, xi: float = 0.0, dps: int = 100) -> mp.mpc:
    """Spin-0 denominator: Pi_s(z,xi) = 1 + 6(xi-1/6)^2 * z * F2_hat(z)."""
    mp.mp.dps = dps
    z_mp = mp.mpc(z)
    coeff = _scalar_mode_coefficient(xi)
    if abs(coeff) < mp.mpf("1e-40"):
        return mp.mpc(1)
    return 1 + coeff * z_mp * _F2_shape(z_mp, xi=xi, dps=dps)


def _find_tt_zero(
    *, z_min: float = 0.0, z_max: float = 10.0, step: float = 0.05,
    xi: float = 0.0, dps: int = 100,
) -> mp.mpf:
    """Locate the first positive real zero of Pi_TT."""
    mp.mp.dps = dps
    z_left = mp.mpf(z_min)
    val_left = mp.re(_Pi_TT(z_left, xi=xi, dps=dps))
    z_right = z_left + mp.mpf(step)

    while z_right <= mp.mpf(z_max):
        val_right = mp.re(_Pi_TT(z_right, xi=xi, dps=dps))
        if val_left == 0:
            return z_left
        if val_left * val_right < 0:
            return mp.findroot(lambda t: mp.re(_Pi_TT(t, xi=xi, dps=dps)),
                               (z_left, z_right))
        z_left = z_right
        val_left = val_right
        z_right += mp.mpf(step)

    raise ValueError(f"no TT zero in [{z_min}, {z_max}]")


# ---------------------------------------------------------------------------
# Newton kernels (Level 1)
# ---------------------------------------------------------------------------
def K_Phi(z: float | mp.mpc, xi: float = 0.0, dps: int = 100) -> mp.mpc:
    """Temporal potential kernel: K_Phi(z) = 4/(3*Pi_TT) - 1/(3*Pi_s)."""
    mp.mp.dps = dps
    pi_tt = _Pi_TT(z, xi=xi, dps=dps)
    pi_s = _Pi_scalar(z, xi=xi, dps=dps)
    return mp.mpf(4) / (3 * pi_tt) - mp.mpf(1) / (3 * pi_s)


def K_Psi(z: float | mp.mpc, xi: float = 0.0, dps: int = 100) -> mp.mpc:
    """Spatial potential kernel: K_Psi(z) = 2/(3*Pi_TT) + 1/(3*Pi_s)."""
    mp.mp.dps = dps
    pi_tt = _Pi_TT(z, xi=xi, dps=dps)
    pi_s = _Pi_scalar(z, xi=xi, dps=dps)
    return mp.mpf(2) / (3 * pi_tt) + mp.mpf(1) / (3 * pi_s)


# ---------------------------------------------------------------------------
# Effective masses (local Yukawa approximation)
# ---------------------------------------------------------------------------
def effective_masses(
    Lambda: float | mp.mpf = 1.0, xi: float = 0.0,
) -> tuple[mp.mpf, mp.mpf | None]:
    """Return (m2, m0) with m0=None at conformal coupling xi=1/6."""
    L = mp.mpf(Lambda)
    m2 = L * mp.sqrt(mp.mpf(60) / 13)
    coeff = _scalar_mode_coefficient(xi)
    if abs(coeff) < mp.mpf("1e-40"):
        return m2, None
    m0 = L / mp.sqrt(coeff)
    return m2, m0


# ---------------------------------------------------------------------------
# Level 1: Exact nonlocal potentials via Fourier-Bessel integrals
# ---------------------------------------------------------------------------
#
# FORMULATION (exact, from NT-4a linearized field equations):
#
#   Phi(r)/Phi_N(r) = (2/pi) * int_0^inf sin(kr)/k * K_Phi(k^2/Lambda^2) dk
#   Psi(r)/Psi_N(r) = (2/pi) * int_0^inf sin(kr)/k * K_Psi(k^2/Lambda^2) dk
#
# where K_Phi(0) = K_Psi(0) = 1 ensures the GR limit.
#
# THEORETICAL COMPLICATION:
# Unlike the Stelle (local) case, where K_Phi(z) = (4/3)*m2^2/(k^2+m2^2)
# - (1/3)*m0^2/(k^2+m0^2) is manifestly positive and decays to zero,
# the SCT kernels have two features that make the integral nontrivial:
#
# 1) K_Phi(z -> inf) -> K_inf ~ -0.136 (nonzero UV asymptote), because
#    Pi_TT(z) saturates to a finite negative value (~-13.83) rather than
#    growing linearly. This makes the correction integral K_Phi - 1
#    only conditionally convergent (via the oscillating sin factor).
#
# 2) Pi_TT(z) has a zero at z0 ~ 2.41484, creating a simple pole in
#    K_Phi on the real k-axis at k0 = Lambda*sqrt(z0).  This is
#    qualitatively different from Stelle, where the pole lies on the
#    negative real z-axis (i.e., imaginary k) and never enters the
#    integration domain.
#
# The Euclidean prescription (which defines the physical static potential
# for the spectral action) requires analytic continuation of the pole
# contribution, yielding a Yukawa-type damped exponential rather than an
# oscillatory cosine term.  A naive decomposition (subtract pole on the
# real axis, add back Euclidean Yukawa) leads to an inconsistency because
# the subtracted PV integral and the Euclidean integral differ by
# (4/3)*R_pole/z0 * [2 - exp(-k0 r) - cos(k0 r)].
#
# CURRENT STATUS:
# - The Level 1 formulation is exact and well-defined.
# - Proper numerical implementation requires either:
#   (a) Direct Euclidean-space integration (K_Phi evaluated at z < 0), or
#   (b) Levin-type oscillatory quadrature with analytic continuation, or
#   (c) Contour deformation in the complex k-plane.
# - These methods are deferred to a dedicated numerical study.
# - For all solar-system phenomenology, Level 2 (local Yukawa) applies
#   because r * Lambda >> 1 for any viable Lambda.
#
# The functions phi_exact, psi_exact, gamma_exact below are PLACEHOLDERS
# that document the mathematical structure but do NOT produce reliable
# numerical results.  Use phi_local, psi_local, gamma_local for all
# quantitative work.
#

def _Pi_TT_prime_at_z0(z0: mp.mpf, *, xi: float = 0.0, dps: int = 100) -> mp.mpf:
    """Compute Pi_TT'(z0) via central difference."""
    mp.mp.dps = dps
    h = mp.power(10, -min(10, dps // 4))
    f_plus = _Pi_TT(z0 + h, xi=xi, dps=dps)
    f_minus = _Pi_TT(z0 - h, xi=xi, dps=dps)
    return mp.re(f_plus - f_minus) / (2 * h)


def _pole_residue_data(*, xi: float = 0.0, dps: int = 100) -> dict[str, Any]:
    """Compute z0, Pi_TT'(z0), and the pole residue for 1/Pi_TT."""
    mp.mp.dps = dps
    z0 = _find_tt_zero(xi=xi, dps=dps)
    deriv = _Pi_TT_prime_at_z0(z0, xi=xi, dps=dps)
    # 1/Pi_TT has a simple pole at z = z0 with residue 1/Pi_TT'(z0)
    R_pole = 1 / deriv
    return {"z0": z0, "deriv": deriv, "R_pole": R_pole}


def _correction_integrand_Phi(
    k: mp.mpf, r: mp.mpf, Lambda: mp.mpf, *,
    xi: float = 0.0, dps: int = 100,
    z0: mp.mpf | None = None, R_pole: mp.mpf | None = None,
) -> mp.mpf:
    """Integrand for Delta_Phi: sin(kr)/k * [K_Phi(z) - 1 - pole_subtraction]."""
    mp.mp.dps = dps
    if k <= 0:
        return mp.mpf(0)
    z = (k / Lambda) ** 2

    # Full kernel
    K_val = mp.re(K_Phi(z, xi=xi, dps=dps))

    # Subtract GR piece (K=1) — already accounted for by Dirichlet integral
    delta_K = K_val - 1

    # Subtract pole contribution if present
    # The pole in K_Phi comes from 4/(3*Pi_TT).
    # Near z0: 4/(3*Pi_TT) ~ 4/(3*Pi_TT'(z0)*(z-z0)) = (4/3)*R_pole/(z-z0)
    # This pole contribution will be computed analytically.
    if z0 is not None and R_pole is not None:
        pole_kernel = mp.mpf(4) / 3 * R_pole / (z - z0)
        delta_K -= mp.re(pole_kernel)

    return mp.sin(k * r) / k * delta_K


def _correction_integrand_Psi(
    k: mp.mpf, r: mp.mpf, Lambda: mp.mpf, *,
    xi: float = 0.0, dps: int = 100,
    z0: mp.mpf | None = None, R_pole: mp.mpf | None = None,
) -> mp.mpf:
    """Integrand for Delta_Psi: sin(kr)/k * [K_Psi(z) - 1 - pole_subtraction]."""
    mp.mp.dps = dps
    if k <= 0:
        return mp.mpf(0)
    z = (k / Lambda) ** 2

    K_val = mp.re(K_Psi(z, xi=xi, dps=dps))
    delta_K = K_val - 1

    if z0 is not None and R_pole is not None:
        pole_kernel = mp.mpf(2) / 3 * R_pole / (z - z0)
        delta_K -= mp.re(pole_kernel)

    return mp.sin(k * r) / k * delta_K


def _pole_contribution_potential(
    r: mp.mpf, Lambda: mp.mpf, z0: mp.mpf, R_pole: mp.mpf,
    weight: mp.mpf,
) -> mp.mpf:
    """
    Euclidean pole contribution to the potential from a zero of Pi_TT.

    The pole at z = z0 in 1/Pi_TT contributes to K_Phi a term
    (4/3)*R_pole/(z - z0) = (4/3)*R_pole*Lambda^2/(k^2 - k0^2),
    where k0 = Lambda*sqrt(z0) and R_pole = 1/Pi_TT'(z0).

    The Euclidean prescription (natural for the spectral action) replaces
    k^2 - k0^2 -> k^2 + k0^2, giving:

      (2/pi) * int sin(kr)/(k*(k^2+k0^2)) dk = (1-exp(-k0*r))/k0^2

    so the pole contribution is weight * R_pole * (1-exp(-k0r)) / z0.

    In the local Stelle limit: R_pole = z0 = 1/c2 = 60/13, so
    R_pole/z0 = 1 and we recover (4/3)*(1-exp(-m2*r)).  Verified.

    NOTE: This formula is only correct when the entire integral is performed
    with the Euclidean prescription.  When the pole is subtracted on the
    real k-axis and the residue is added back analytically, the PV integral
    (G-R 3.723.2) gives (1-cos(k0*r))/k0^2 instead, leading to an
    inconsistent decomposition.  See the Level 1 status note above.
    """
    k0 = Lambda * mp.sqrt(z0)
    return weight * R_pole * (1 - mp.exp(-k0 * r)) / z0


def _correction_integral(
    r: mp.mpf, Lambda: mp.mpf, kernel_func: str, *,
    xi: float = 0.0, dps: int = 50, n_points: int = 200,
    z0: mp.mpf | None = None, R_pole: mp.mpf | None = None,
) -> mp.mpf:
    """Compute the regularized correction integral Delta(r).

    WARNING: This function has a known prescription mismatch (see Level 1
    status note) and does NOT produce correct results for the full SCT
    nonlocal case.  The pole subtraction on the real axis is inconsistent
    with the Euclidean analytic continuation used in _pole_contribution_potential.
    Retained for documentation of the mathematical structure.
    """
    mp.mp.dps = dps

    # Choose k_max: the form factors suppress the integrand for k >> Lambda
    k_max = 8 * Lambda  # well beyond the form factor rolloff

    if kernel_func == "Phi":
        integrand = lambda k: _correction_integrand_Phi(  # noqa: E731
            k, r, Lambda, xi=xi, dps=dps, z0=z0, R_pole=R_pole,
        )
    else:
        integrand = lambda k: _correction_integrand_Psi(  # noqa: E731
            k, r, Lambda, xi=xi, dps=dps, z0=z0, R_pole=R_pole,
        )

    # If there is a pole and we have NOT subtracted it, we need to be
    # careful near k0 = Lambda*sqrt(z0). Since we DO subtract the pole,
    # the integrand is regular and we can integrate straightforwardly.

    # Use mpmath quad for oscillatory integral.
    # Split at k0 to help with numerical stability near the subtracted pole.
    if z0 is not None:
        k0 = Lambda * mp.sqrt(z0)
        if k0 < k_max:
            eps = Lambda * mp.mpf("0.01")
            # Three segments: [eps, k0-delta], [k0-delta, k0+delta], [k0+delta, k_max]
            delta = min(k0 * mp.mpf("0.1"), Lambda * mp.mpf("0.2"))
            k_lo = max(eps, k0 - delta)
            k_hi = min(k_max, k0 + delta)
            result = mp.mpf(0)
            if k_lo > eps:
                result += mp.quad(integrand, [eps, k_lo])
            result += mp.quad(integrand, [k_lo, k_hi])
            if k_hi < k_max:
                result += mp.quad(integrand, [k_hi, k_max])
            return result * 2 / mp.pi
    else:
        eps = Lambda * mp.mpf("0.01")

    result = mp.quad(integrand, [eps, k_max])
    return result * 2 / mp.pi


def phi_exact(
    r: float | mp.mpf, Lambda: float | mp.mpf, xi: float = 0.0,
    dps: int = 50, **kwargs: Any,
) -> mp.mpf:
    """Level 1: Exact nonlocal Phi(r)/Phi_N(r) via Fourier-Bessel integral.

    WARNING: This function is a PLACEHOLDER.  The numerical integration
    does not correctly handle the prescription mismatch between the
    real-axis pole subtraction and the Euclidean analytic continuation.
    Results are NOT reliable.  Use phi_local() for all quantitative work.

    Returns the ratio Phi(r) / (GM/r).

    Args:
        r: Radial distance (in natural units, eV^{-1}).
        Lambda: SCT cutoff scale (in eV).
        xi: Non-minimal Higgs coupling.
        dps: Decimal precision for mpmath.
    """
    mp.mp.dps = dps
    r_mp = mp.mpf(r)
    L_mp = mp.mpf(Lambda)

    if r_mp <= 0:
        raise ValueError(f"r must be positive, got {r}")

    # Get pole data
    pole_data = _pole_residue_data(xi=xi, dps=dps)
    z0 = pole_data["z0"]
    R_pole = pole_data["R_pole"]

    # GR part: 1 (from Dirichlet integral)
    # Regularized correction: Delta(r)
    delta = _correction_integral(
        r_mp, L_mp, "Phi", xi=xi, dps=dps, z0=z0, R_pole=R_pole,
        **kwargs,
    )

    # Pole contribution
    pole_phi = _pole_contribution_potential(r_mp, L_mp, z0, R_pole, mp.mpf(4) / 3)

    # Scalar pole contribution (if scalar has a pole — generally it doesn't
    # for the SM parameters, since Pi_s is monotonically increasing for xi=0)
    # For now, assume no scalar pole (verified numerically).

    return 1 + delta + pole_phi


def psi_exact(
    r: float | mp.mpf, Lambda: float | mp.mpf, xi: float = 0.0,
    dps: int = 50, **kwargs: Any,
) -> mp.mpf:
    """Level 1: Exact nonlocal Psi(r)/Psi_N(r) via Fourier-Bessel integral.

    WARNING: PLACEHOLDER — same limitations as phi_exact.  Use psi_local().
    """
    mp.mp.dps = dps
    r_mp = mp.mpf(r)
    L_mp = mp.mpf(Lambda)

    if r_mp <= 0:
        raise ValueError(f"r must be positive, got {r}")

    pole_data = _pole_residue_data(xi=xi, dps=dps)
    z0 = pole_data["z0"]
    R_pole = pole_data["R_pole"]

    delta = _correction_integral(
        r_mp, L_mp, "Psi", xi=xi, dps=dps, z0=z0, R_pole=R_pole,
        **kwargs,
    )
    pole_psi = _pole_contribution_potential(r_mp, L_mp, z0, R_pole, mp.mpf(2) / 3)

    return 1 + delta + pole_psi


def gamma_exact(
    r: float | mp.mpf, Lambda: float | mp.mpf, xi: float = 0.0,
    dps: int = 50, **kwargs: Any,
) -> mp.mpf:
    """Level 1: Exact nonlocal gamma(r) = Psi(r)/Phi(r).

    WARNING: PLACEHOLDER — same limitations as phi_exact.  Use gamma_local().
    """
    phi = phi_exact(r, Lambda, xi=xi, dps=dps, **kwargs)
    psi = psi_exact(r, Lambda, xi=xi, dps=dps, **kwargs)
    if abs(phi) < mp.mpf("1e-40"):
        return mp.nan
    return psi / phi


# ---------------------------------------------------------------------------
# Level 2: Local Yukawa approximation (wraps existing NT-4a code)
# ---------------------------------------------------------------------------
def phi_local(
    r: float | mp.mpf, Lambda: float | mp.mpf, xi: float = 0.0,
    dps: int = 80,
) -> mp.mpf:
    """Level 2: Local Yukawa Phi(r)/Phi_N(r).

    Phi/Phi_N = 1 - (4/3)e^{-m2*r} + (1/3)e^{-m0*r}
    """
    mp.mp.dps = dps
    r_mp = mp.mpf(r)
    if r_mp <= 0:
        raise ValueError(f"r must be positive, got {r}")
    m2, m0 = effective_masses(Lambda=Lambda, xi=xi)
    ratio = 1 - mp.mpf(4) / 3 * mp.exp(-m2 * r_mp)
    if m0 is not None:
        ratio += mp.mpf(1) / 3 * mp.exp(-m0 * r_mp)
    return ratio


def psi_local(
    r: float | mp.mpf, Lambda: float | mp.mpf, xi: float = 0.0,
    dps: int = 80,
) -> mp.mpf:
    """Level 2: Local Yukawa Psi(r)/Psi_N(r).

    Psi/Psi_N = 1 - (2/3)e^{-m2*r} - (1/3)e^{-m0*r}
    """
    mp.mp.dps = dps
    r_mp = mp.mpf(r)
    if r_mp <= 0:
        raise ValueError(f"r must be positive, got {r}")
    m2, m0 = effective_masses(Lambda=Lambda, xi=xi)
    ratio = 1 - mp.mpf(2) / 3 * mp.exp(-m2 * r_mp)
    if m0 is not None:
        ratio -= mp.mpf(1) / 3 * mp.exp(-m0 * r_mp)
    return ratio


def gamma_local(
    r: float | mp.mpf, Lambda: float | mp.mpf = 1.0,
    xi: float = 0.0, dps: int = 80,
) -> mp.mpf:
    """Level 2: Local Yukawa gamma(r) = Psi/Phi."""
    phi = phi_local(r, Lambda, xi=xi, dps=dps)
    psi = psi_local(r, Lambda, xi=xi, dps=dps)
    if abs(phi) < mp.mpf("1e-40"):
        return mp.nan
    return psi / phi


# ---------------------------------------------------------------------------
# Level 3: Nonlinear PPN (stubs — NOT DERIVED)
# ---------------------------------------------------------------------------
NOT_DERIVED = "NOT_DERIVED"


def ppn_table(
    Lambda: float | mp.mpf, xi: float = 0.0,
    r_test: float | mp.mpf | None = None,
) -> dict[str, Any]:
    """Complete PPN parameter table.

    Returns all 10 standard PPN parameters with their status.

    Args:
        Lambda: SCT cutoff in eV.
        xi: Non-minimal Higgs coupling.
        r_test: Radius for gamma evaluation (default: 1 AU in eV^{-1}).
    """
    if r_test is None:
        r_test = AU_EV_INV

    r_mp = mp.mpf(r_test)
    gamma_val = gamma_local(r_mp, Lambda=Lambda, xi=xi, dps=80)

    return {
        "scope": "linear_static_local_yukawa",
        "Lambda_eV": str(Lambda),
        "xi": str(xi),
        "r_test_eV_inv": str(r_test),
        # Level 2 (derived)
        "gamma": str(gamma_val),
        "gamma_minus_1": str(gamma_val - 1),
        # Level 3 (not derived — requires NT-4b)
        "beta": NOT_DERIVED,
        "xi_PPN": NOT_DERIVED,
        "alpha1": "0 (diffeomorphism invariance)",
        "alpha2": "0 (diffeomorphism invariance)",
        "alpha3": "0 (diffeomorphism invariance)",
        "zeta1": "0 (local energy-momentum conservation)",
        "zeta2": "0 (local energy-momentum conservation)",
        "zeta3": "0 (local energy-momentum conservation)",
        "zeta4": "0 (local energy-momentum conservation)",
        # Notes
        "beta_status": "OPEN — requires O(h^2) field equations from NT-4b",
        "alpha_i_note": (
            "alpha_i = 0 from diffeomorphism invariance of the spectral action. "
            "Verified indirectly by NT-4a off-shell gauge invariance check."
        ),
        "zeta_i_note": (
            "zeta_i = 0 from local energy-momentum conservation "
            "(follows from diffeomorphism invariance via Noether)."
        ),
    }


# ---------------------------------------------------------------------------
# Experimental bounds on Lambda
# ---------------------------------------------------------------------------
def lower_bound_Lambda(
    experiment: str, xi: float = 0.0,
) -> dict[str, Any]:
    """Compute lower bound on Lambda from an experimental constraint.

    Returns a dictionary with the bound in eV, the method, and the
    assumptions used.

    Supported experiments:
        "cassini"    — Cassini Shapiro delay, |gamma-1| < 2.3e-5
        "messenger"  — MESSENGER radioscience, |gamma-1| < 2.0e-5
        "eot-wash"   — Eot-Wash torsion balance, |alpha|<1 at lambda=38.6 um
        "microscope" — MICROSCOPE WEP, |eta| < 1.5e-15
        "llr"        — Lunar Laser Ranging, |eta| < 4.4e-4
    """
    mp.mp.dps = 30
    m2_over_Lambda = mp.sqrt(mp.mpf(60) / 13)

    if experiment.lower() == "cassini":
        # Bertotti, Iess, Tortora (2003): |gamma-1| < 2.3e-5 at r ~ 1 AU
        # gamma - 1 ~ (2/3)*exp(-m2*r_AU) for large r (Level 2 expansion)
        # => exp(-m2*r_AU) < 2.3e-5 * 3/2 = 3.45e-5
        # => m2*r_AU > ln(1/3.45e-5)
        threshold = mp.mpf("2.3e-5")
        exp_bound = threshold * 3 / 2
        min_m2_r = -mp.log(exp_bound)
        r_AU = AU_EV_INV
        m2_min = min_m2_r / r_AU
        Lambda_min = m2_min / m2_over_Lambda
        return {
            "experiment": "Cassini (Bertotti+ 2003)",
            "constraint": "|gamma-1| < 2.3e-5",
            "reference": "Nature 425, 374 (2003)",
            "r_test": f"1 AU = {float(r_AU):.4e} eV^-1",
            "Lambda_min_eV": float(Lambda_min),
            "Lambda_min_str": f"{float(Lambda_min):.2e} eV",
            "m2_min_eV": float(m2_min),
            "approximation": "Level 2 (local Yukawa, spin-2 dominant)",
            "assumptions": "m0 >= m2 so spin-2 is the dominant correction",
        }

    if experiment.lower() == "messenger":
        # Verma+ (2014): |gamma-1| < 2.5e-5 (1-sigma, A&A 561 A115)
        threshold = mp.mpf("2.5e-5")
        exp_bound = threshold * 3 / 2
        min_m2_r = -mp.log(exp_bound)
        r_AU = AU_EV_INV
        m2_min = min_m2_r / r_AU
        Lambda_min = m2_min / m2_over_Lambda
        return {
            "experiment": "MESSENGER (Verma+ 2014)",
            "constraint": "|gamma-1| < 2.5e-5",
            "reference": "A&A 561, A115 (2014), arXiv:1306.5569",
            "r_test": f"1 AU = {float(r_AU):.4e} eV^-1",
            "Lambda_min_eV": float(Lambda_min),
            "Lambda_min_str": f"{float(Lambda_min):.2e} eV",
            "m2_min_eV": float(m2_min),
            "approximation": "Level 2 (local Yukawa, spin-2 dominant)",
            "assumptions": "m0 >= m2 so spin-2 is the dominant correction",
        }

    if experiment.lower() in ("eot-wash", "eot_wash", "eotwash"):
        # Lee+ (2020): Yukawa constraint |alpha| < 1 at lambda = 38.6 um
        # arXiv:2002.11761, PRL 124, 101101.
        # In SCT: the spin-2 Yukawa has alpha = -4/3, range lambda = 1/m2.
        # Since |alpha| = 4/3 > 1, the Eot-Wash constraint requires the
        # Yukawa range to be below the |alpha|=1 sensitivity threshold.
        # At |alpha|=4/3 the excluded range may be slightly shorter than
        # 38.6 um, so using 38.6 um is conservative.
        lambda_max_m = mp.mpf("38.6e-6")  # meters (95% CL)
        lambda_max_eV_inv = lambda_max_m * M_TO_EV_INV
        m2_min = 1 / lambda_max_eV_inv  # = hbar*c / lambda_max
        Lambda_min = m2_min / m2_over_Lambda
        return {
            "experiment": "Eot-Wash (Lee+ 2020)",
            "constraint": "|alpha| < 1 at lambda = 38.6 um (95% CL)",
            "reference": "Phys. Rev. Lett. 124, 101101 (2020), arXiv:2002.11761",
            "r_test": f"lambda = 38.6 um = {float(lambda_max_eV_inv):.4e} eV^-1",
            "Lambda_min_eV": float(Lambda_min),
            "Lambda_min_str": f"{float(Lambda_min):.2e} eV",
            "m2_min_eV": float(m2_min),
            "approximation": "Level 2 (local Yukawa, spin-2 range constraint)",
            "assumptions": (
                "SCT spin-2 Yukawa has |alpha|=4/3. "
                "Eot-Wash excludes |alpha|>=1 for lambda > 38.6 um (95% CL), "
                "so the SCT Yukawa range must be lambda < 38.6 um. "
                "Using 38.6 um (not the slightly shorter range at |alpha|=4/3) "
                "gives a conservative lower bound on Lambda."
            ),
        }

    if experiment.lower() == "microscope":
        # MICROSCOPE (Touboul+ 2022): |eta(Ti,Pt)| < 1.5e-15
        # eta = 4*beta - gamma - 3 (Nordtvedt parameter)
        # Since beta is NOT DERIVED, we cannot extract a Lambda bound.
        return {
            "experiment": "MICROSCOPE (Touboul+ 2022)",
            "constraint": "|eta(Ti,Pt)| < 1.5e-15",
            "reference": "Phys. Rev. Lett. 129, 121102 (2022)",
            "Lambda_min_eV": None,
            "Lambda_min_str": NOT_DERIVED,
            "approximation": NOT_DERIVED,
            "assumptions": (
                "The Nordtvedt parameter eta = 4*beta - gamma - 3 "
                "requires knowledge of beta, which is not yet derived. "
                "If beta = 1 (as in GR at leading order), then "
                "eta = -gamma+1 ~ 0 at solar system scales."
            ),
        }

    if experiment.lower() == "llr":
        # Williams, Turyshev, Boggs (2004): |eta| < 4.4e-4
        # Same issue: requires beta.
        return {
            "experiment": "Lunar Laser Ranging (Williams+ 2004)",
            "constraint": "Nordtvedt |eta| < 4.4e-4",
            "reference": "Phys. Rev. Lett. 93, 261101 (2004)",
            "Lambda_min_eV": None,
            "Lambda_min_str": NOT_DERIVED,
            "approximation": NOT_DERIVED,
            "assumptions": (
                "Requires beta from nonlinear field equations (NT-4b). "
                "At present, only gamma is derived."
            ),
        }

    raise ValueError(f"Unknown experiment: {experiment}")


# ---------------------------------------------------------------------------
# Level comparison: |gamma_Level1 - gamma_Level2| vs r*Lambda
# ---------------------------------------------------------------------------
def level_comparison(
    Lambda: float | mp.mpf = 1.0, xi: float = 0.0,
    rL_values: list[float] | None = None,
    dps: int = 30,
) -> list[dict[str, float]]:
    """Compare Level 1 and Level 2 gamma as a function of r*Lambda.

    Returns a list of dicts with r*Lambda, gamma_L1, gamma_L2, and |diff|.
    """
    if rL_values is None:
        rL_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]

    L_mp = mp.mpf(Lambda)
    results = []
    for rL in rL_values:
        r = mp.mpf(rL) / L_mp
        try:
            g_l1 = gamma_exact(r, L_mp, xi=xi, dps=dps)
        except Exception:
            g_l1 = mp.nan
        g_l2 = gamma_local(r, L_mp, xi=xi, dps=dps)
        results.append({
            "rLambda": rL,
            "gamma_L1": float(g_l1) if not mp.isnan(g_l1) else None,
            "gamma_L2": float(g_l2),
            "diff": float(abs(g_l1 - g_l2)) if not mp.isnan(g_l1) else None,
        })
    return results


# ---------------------------------------------------------------------------
# Publication-quality exclusion plot
# ---------------------------------------------------------------------------
def exclusion_plot(
    output_path: str | Path | None = None,
    xi: float = 0.0,
) -> Path:
    """Generate a publication-quality Lambda exclusion plot.

    Shows lower bounds on Lambda from different experiments as vertical
    lines / shaded regions on a logarithmic Lambda axis.
    """
    if output_path is None:
        output_path = FIGURES_DIR / "ppn1_exclusion.pdf"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect bounds
    experiments = ["cassini", "messenger", "eot-wash"]
    bounds = {}
    for exp in experiments:
        result = lower_bound_Lambda(exp, xi=xi)
        if result["Lambda_min_eV"] is not None:
            bounds[result["experiment"]] = result["Lambda_min_eV"]

    # Generate gamma(r*Lambda) curve for the Level 2 approximation
    rL_vals = np.logspace(-1, 3, 200)
    gamma_minus_1 = []
    for rL in rL_vals:
        m2_ov_L = math.sqrt(60 / 13)
        m0_ov_L = math.sqrt(6)
        g_m_1 = abs(
            2 / 3 * math.exp(-m2_ov_L * rL)
            - 2 / 3 * math.exp(-m0_ov_L * rL)
        )
        gamma_minus_1.append(max(g_m_1, 1e-300))

    init_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left panel: |gamma-1| vs r*Lambda
    ax1.semilogy(rL_vals, gamma_minus_1, color=SCT_COLORS["total"], linewidth=1.5)
    ax1.axhline(y=2.3e-5, color=SCT_COLORS["data"], linestyle="--",
                alpha=0.7, label="Cassini bound")
    ax1.set_xlabel(r"$r \Lambda$")
    ax1.set_ylabel(r"$|\gamma - 1|$")
    ax1.set_xscale("log")
    ax1.set_xlim(0.1, 1000)
    ax1.set_ylim(1e-15, 2)
    ax1.set_title(r"PPN $\gamma$ deviation (Level 2)")
    ax1.legend(fontsize=8)

    # Right panel: exclusion plot on Lambda axis
    colors = {
        "Cassini": SCT_COLORS["scalar"],
        "MESSENGER": SCT_COLORS["vector"],
        "Eot-Wash": SCT_COLORS["dirac"],
    }
    y_pos = 0
    for label, lam_min in sorted(bounds.items(), key=lambda x: x[1]):
        short = label.split("(")[0].strip()
        color = colors.get(short, SCT_COLORS["reference"])
        ax2.barh(y_pos, math.log10(lam_min) - (-20), left=-20,
                 height=0.6, color=color, alpha=0.4)
        ax2.axvline(x=math.log10(lam_min), color=color, linestyle="--",
                    linewidth=1.2)
        ax2.text(math.log10(lam_min) + 0.3, y_pos,
                 f"{short}\n" + r"$\Lambda > $" + f"{lam_min:.1e} eV",
                 fontsize=7, va="center")
        y_pos += 1

    ax2.set_xlabel(r"$\log_{10}(\Lambda / \mathrm{eV})$")
    ax2.set_xlim(-20, 2)
    ax2.set_yticks([])
    ax2.set_title(r"Lower bounds on $\Lambda$")
    ax2.axvspan(-20, max(math.log10(v) for v in bounds.values()),
                alpha=0.05, color="red", label="Excluded")

    fig.tight_layout()
    save_figure(fig, output_path.stem, fmt="pdf", directory=output_path.parent)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Snapshot export
# ---------------------------------------------------------------------------
def ppn_snapshot(
    Lambda: float | mp.mpf = 1.0, xi: float = 0.0,
) -> dict[str, Any]:
    """Generate a full PPN-1 snapshot for archival."""
    mp.mp.dps = 50
    m2, m0 = effective_masses(Lambda=Lambda, xi=xi)

    bounds = {}
    for exp in ("cassini", "messenger", "eot-wash", "microscope", "llr"):
        bounds[exp] = lower_bound_Lambda(exp, xi=xi)

    return {
        "phase": "PPN-1",
        "scope": "linear_static",
        "Lambda_eV": str(Lambda),
        "xi": str(xi),
        "m2_eV": str(m2),
        "m0_eV": str(m0) if m0 is not None else "infinity (xi=1/6)",
        "m2_over_Lambda": str(mp.sqrt(mp.mpf(60) / 13)),
        "m0_over_Lambda": str(1 / mp.sqrt(_scalar_mode_coefficient(xi)))
            if abs(_scalar_mode_coefficient(xi)) > 1e-14 else "infinity",
        "bounds": bounds,
        "parameters": ppn_table(Lambda, xi=xi),
        "pole_data": {
            "z0": str(_find_tt_zero(xi=xi, dps=50)),
            "Pi_TT_prime_z0": str(_Pi_TT_prime_at_z0(
                _find_tt_zero(xi=xi, dps=50), xi=xi, dps=50)),
            "ghost_note": (
                "Pi_TT has a zero at z0 ~ 2.41. The massive spin-2 pole "
                "is a ghost (R2 < 0) with |R2| ~ 0.49 (50% suppressed vs Stelle)."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """Run PPN-1 analysis and generate outputs."""
    parser = argparse.ArgumentParser(
        description="PPN-1 parameters for SCT Theory."
    )
    parser.add_argument("--Lambda", type=float, default=1e-3,
                        help="Cutoff scale in eV")
    parser.add_argument("--xi", type=float, default=0.0,
                        help="Non-minimal coupling xi")
    parser.add_argument("--output", type=Path,
                        default=RESULTS_DIR / "ppn1_snapshot.json")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip exclusion plot generation")
    args = parser.parse_args()

    L_mp = mp.mpf(str(args.Lambda))
    snapshot = ppn_snapshot(L_mp, xi=args.xi)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(snapshot, indent=2, default=str), encoding="utf-8",
    )
    print(f"Wrote snapshot to {args.output}")

    if not args.no_plot:
        plot_path = exclusion_plot(xi=args.xi)
        print(f"Wrote exclusion plot to {plot_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PPN-1 Summary")
    print("=" * 60)
    print(f"Lambda = {args.Lambda} eV, xi = {args.xi}")
    m2, m0 = effective_masses(Lambda=args.Lambda, xi=args.xi)
    print(f"m2 = {float(m2):.4e} eV")
    print(f"m0 = {float(m0):.4e} eV" if m0 else "m0 = infinity (conformal)")

    for exp in ("cassini", "messenger", "eot-wash"):
        bound = lower_bound_Lambda(exp, xi=args.xi)
        print(f"{bound['experiment']}: Lambda > {bound['Lambda_min_str']}")

    params = ppn_table(L_mp, xi=args.xi)
    print(f"gamma at 1 AU: {params['gamma']}")
    print(f"beta: {params['beta']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
