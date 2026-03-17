# ruff: noqa: E402, I001
"""
MR-9: Black hole singularity resolution in the spectral action.

Computes the exact nonlocal Newtonian potential from the full SCT propagator
denominators Pi_TT(z) and Pi_s(z,xi), the resulting modified Schwarzschild
metric, Kretschner scalar, energy conditions, and horizon structure.

LINEARIZED analysis only. All results follow from the NT-4a linearized field
equations around flat spacetime. Nonlinear backreaction (which would require
solving the full NT-4b equations self-consistently) is NOT included.

Key formulas:
  V(r) = -(GM/r) * [1 + Delta(r)]

where Delta(r) encodes the nonlocal correction:
  Delta(r) = (2/(pi*r)) * integral_0^infty dk sin(kr) * K(k^2/Lambda^2) / k

with kernel:
  K(z) = [1/Pi_TT(z) - 1]*(4/3) - [1/Pi_s(z,xi) - 1]*(1/3)

NOTE: The sign convention is opposite from nt4a_newtonian.py:
  V(r)/V_N(r) = 1 - (4/3)*Y_2(r) + (1/3)*Y_0(r)  [Yukawa approx]
  where Y_i(r) = exp(-m_i * r) are the Yukawa factors.

At r=0 with both modes present (xi != 1/6):
  V(r->0) -> -GM * [(4/3)*m_2 - (1/3)*m_0]  [LOCAL Yukawa approx]
  The exact nonlocal integral may differ.

Author: David Alfyorov
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from sct_tools.plotting import SCT_COLORS, create_figure, init_style, save_figure

# Phase-local imports from the NT-4a propagator module
from scripts.nt4a_propagator import (
    Pi_TT,
    Pi_scalar,
    find_first_positive_real_tt_zero,
    scalar_local_mass,
    scalar_mode_coefficient,
    spin2_local_mass,
)
from scripts.nt2_entire_function import phi_complex_mp

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "mr9"
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures" / "mr9"

# ===========================================================================
# Physical constants (dimensionless: Lambda = 1 throughout)
# ===========================================================================
ALPHA_C = mp.mpf(13) / 120
C2 = 2 * ALPHA_C  # = 13/60

# Default precision
DEFAULT_DPS = 50


# ===========================================================================
# A. EXACT NONLOCAL POTENTIAL
# ===========================================================================

def _kernel_K(z, xi=0.0, dps=DEFAULT_DPS):
    """Newtonian correction kernel K(z) from propagator modification.

    K(z) = (4/3)*[1/Pi_TT(z) - 1] - (1/3)*[1/Pi_s(z,xi) - 1]

    This is the NEGATIVE of the potential correction: the sign is chosen so
    that V(r)/V_N(r) = 1 - Delta(r) where Delta > 0 at short distances.

    Actually, from the standard derivation (Modesto-Shapiro, Buoninfante-Lambiase-Mazumdar):
    V(r) = -(GM/r) * {1 + (2/pi) * int_0^inf sin(kr)/(kr) * [K_TT(k) + K_s(k)] dk}

    We define K(z) such that:
    Delta(r) = (2/(pi*r)) * int_0^inf sin(kr)/k * K(k^2/Lambda^2) dk

    with K(z) = -(4/3)*[1 - 1/Pi_TT(z)] + (1/3)*[1 - 1/Pi_s(z,xi)]
             = -(4/3) + (4/3)/Pi_TT(z) + (1/3) - (1/3)/Pi_s(z,xi)
             = -1 + (4/3)/Pi_TT(z) - (1/3)/Pi_s(z,xi)

    At z=0: K(0) = -1 + 4/3 - 1/3 = 0. Correct.
    At z->inf: K(z) -> -1 (both Pi -> inf). So V -> 0 (smeared source).

    For the EXACT integral, we use:
    V(r)/V_N(r) = 1 + (2/pi) * int_0^inf sin(kr)/(kr) * K_eff(k^2/Lambda^2) dk

    where K_eff(z) = (4/3)/Pi_TT(z) - (1/3)/Pi_s(z) - 1
    """
    mp.mp.dps = dps
    z_mp = mp.mpc(z)
    pi_tt = Pi_TT(z_mp, xi=xi, dps=dps)
    pi_s = Pi_scalar(z_mp, xi=xi, dps=dps)
    return mp.mpf(4) / 3 / pi_tt - mp.mpf(1) / 3 / pi_s - 1


def _kernel_K_fakeon(z_real, xi=0.0, dps=DEFAULT_DPS, z_pole=None):
    """Kernel K(z) for real z, with fakeon (Cauchy principal value) at poles.

    For real z approaching a zero of Pi_TT, the propagator 1/Pi_TT diverges.
    The fakeon prescription replaces 1/(z - z_pole) with PV[1/(z - z_pole)].
    On the real axis away from the pole, this is just the ordinary value.
    The pole itself must be excluded from the integration domain (PV sense).

    Near the pole, we use a proximity test: if |z - z_pole| < epsilon,
    return the smooth (non-singular) part of K. At exactly z = z_pole,
    the PV integral contribution from the pole is zero by symmetry.
    """
    mp.mp.dps = dps
    z_mp = mp.mpf(z_real)

    # Near the ghost pole: PV prescription gives zero singular contribution
    if z_pole is not None:
        dist = abs(z_mp - mp.mpf(z_pole))
        if dist < mp.mpf("1e-6"):
            # At exactly the pole or very near it, the 1/Pi_TT piece is PV-regulated.
            # The PV integral of 1/(z - z_pole) over a symmetric interval is 0.
            # Return only the scalar sector contribution.
            pi_s = mp.re(Pi_scalar(z_mp, xi=xi, dps=dps))
            if abs(pi_s) > mp.mpf("1e-30"):
                return -mp.mpf(1) / 3 / pi_s + mp.mpf(1) / 3
            return mp.mpf(0)

    pi_tt = mp.re(Pi_TT(z_mp, xi=xi, dps=dps))
    pi_s = mp.re(Pi_scalar(z_mp, xi=xi, dps=dps))

    # If Pi_TT is very close to zero (approaching a pole), use PV regularization
    if abs(pi_tt) < mp.mpf("1e-10"):
        # Near a zero of Pi_TT: return the scalar piece only
        if abs(pi_s) > mp.mpf("1e-30"):
            return -mp.mpf(1) / 3 / pi_s + mp.mpf(1) / 3
        return mp.mpf(0)

    result = mp.mpf(4) / 3 / pi_tt
    if abs(pi_s) > mp.mpf("1e-30"):
        result -= mp.mpf(1) / 3 / pi_s
    else:
        result -= mp.mpf(1) / 3  # Pi_s -> 0 is not expected physically
    result -= 1
    return result


def potential_ratio_exact(
    r,
    *,
    Lambda=1.0,
    xi=0.0,
    dps=DEFAULT_DPS,
    k_max_factor=50.0,
    n_quad=2000,
):
    """Compute V(r)/V_N(r) using the exact nonlocal integral with fakeon PV.

    V(r)/V_N(r) = 1 + (2/pi) * int_0^inf sin(kr)/(kr) * K_eff(k^2/Lambda^2) dk

    The ghost pole at z_pole (where Pi_TT(z_pole)=0) produces a 1/(z-z_pole)
    divergence in K(z). The fakeon prescription replaces this with a Cauchy
    principal value integral. Physically, the PV removes the on-shell ghost
    contribution while preserving the off-shell quantum corrections.

    Implementation: decompose K(z) near the pole as
      K(z) = A/(z - z_pole) + K_smooth(z)
    where A = (4/3) * 1/Pi_TT'(z_pole) (the residue).
    The PV integral of A/(z-z_pole) * sin(kr)/(kr) gives a specific finite
    contribution, while K_smooth integrates normally.

    We implement this by subtracting the singular part and adding back the
    PV contribution analytically.

    Parameters:
        r: radius in units of 1/Lambda
        Lambda: spectral cutoff scale (default 1.0)
        xi: Higgs non-minimal coupling
        dps: decimal precision
        k_max_factor: integration cutoff in units of Lambda
        n_quad: number of quadrature points

    Returns:
        V(r)/V_N(r) as mpf
    """
    mp.mp.dps = dps
    r_mp = mp.mpf(r)
    lam = mp.mpf(Lambda)

    if r_mp <= 0:
        raise ValueError(f"r must be positive, got {r}")

    k_max = k_max_factor * lam

    # Find the ghost pole location
    try:
        z_pole = find_first_positive_real_tt_zero(z_min=0.1, z_max=20.0, xi=xi, dps=dps)
    except ValueError:
        z_pole = None

    k_pole = mp.sqrt(z_pole * lam**2) if z_pole is not None else None

    # Compute the residue A at the ghost pole:
    # K(z) ~ (4/3) * 1/(Pi_TT'(z_pole) * (z - z_pole)) + ...
    # Pi_TT'(z) evaluated numerically
    if z_pole is not None:
        dz = mp.mpf("1e-6")
        pi_tt_plus = mp.re(Pi_TT(z_pole + dz, xi=xi, dps=dps))
        pi_tt_minus = mp.re(Pi_TT(z_pole - dz, xi=xi, dps=dps))
        dPi_dz = (pi_tt_plus - pi_tt_minus) / (2 * dz)
        residue_A = mp.mpf(4) / 3 / dPi_dz  # Residue of (4/3)/Pi_TT at the pole

    def K_smooth(z_val):
        """K(z) with the singular part subtracted near the pole."""
        z_mp = mp.mpf(z_val)
        pi_tt = mp.re(Pi_TT(z_mp, xi=xi, dps=dps))
        pi_s = mp.re(Pi_scalar(z_mp, xi=xi, dps=dps))

        # Full K
        if abs(pi_tt) > mp.mpf("1e-20"):
            K_full = mp.mpf(4) / 3 / pi_tt - mp.mpf(1) / 3 / pi_s - 1
        else:
            K_full = mp.mpf(0)

        # Subtract the singular part near the pole
        if z_pole is not None and abs(z_mp - z_pole) < mp.mpf(2):
            diff = z_mp - z_pole
            if abs(diff) > mp.mpf("1e-20"):
                K_full -= residue_A / diff
            else:
                # At exactly the pole, the subtracted function should be
                # the derivative of the smooth part. Skip.
                K_full = mp.mpf(0)

        return K_full

    def integrand_smooth(k):
        """sin(kr)/(kr) * K_smooth(k^2/Lambda^2)"""
        k_mp = mp.mpf(k)
        if k_mp < mp.mpf("1e-40"):
            return mp.mpf(0)
        z = (k_mp / lam) ** 2
        kr = k_mp * r_mp
        sinc_kr = mp.sin(kr) / kr
        return sinc_kr * K_smooth(float(z))

    # Smooth part: standard integration
    I_smooth = mp.quad(integrand_smooth, [0, float(k_max)], maxdegree=8, error=True)[0]

    # Singular part: PV integral of (4/3)/(Pi_TT'*(z-z_pole)) * sin(kr)/(kr)
    # Change variables: z = k^2/Lambda^2, so k = Lambda*sqrt(z), dk = Lambda/(2*sqrt(z)) dz
    # The integral becomes:
    # I_sing = int_0^inf dk * sin(kr)/(kr) * A/(k^2/Lambda^2 - z_pole)
    #        = int_0^inf dk * sin(kr)/(kr) * A*Lambda^2/(k^2 - k_pole^2)
    #        = A*Lambda^2 * int_0^inf dk * sin(kr)/(kr) * 1/(k^2 - k_pole^2)
    #        = A*Lambda^2 * int_0^inf dk * sin(kr)/(kr) * 1/((k-k_pole)(k+k_pole))
    #
    # PV integral of sin(kr)/(kr*(k^2 - k_pole^2)):
    # Using partial fractions: 1/(k^2 - a^2) = 1/(2a) * [1/(k-a) - 1/(k+a)]
    # PV[int_0^inf sin(kr)/(kr) * 1/(k-a) dk] = -pi*cos(ar)/(2ar) for a > 0
    # (This is a standard Fourier sine transform result with PV.)
    #
    # So: PV[I_sing] = A*Lambda^2/(2*k_pole) * (-pi*cos(k_pole*r)/(2*k_pole*r))
    #                 - A*Lambda^2/(2*k_pole) * standard integral of 1/(k+k_pole)
    #
    # The second term (1/(k+k_pole)) is non-singular and gives:
    # int_0^inf sin(kr)/(kr) * 1/(k+k_pole) dk = ... (well-defined, can compute)
    #
    # Actually, the standard result for PV integrals:
    # PV[int_0^inf sin(t*x)/x * 1/(x^2 - a^2) dx] = -pi/(2a^2) * cos(a*t) for a,t > 0
    # (Gradshteyn-Ryzhik 3.741.3)
    #
    # Our integral: PV[int_0^inf sin(kr)/(kr) * 1/(k^2 - k_pole^2) dk]
    # = (1/r) * PV[int_0^inf sin(kr)/k * 1/(k^2 - k_pole^2) dk]
    # Let u = kr: PV[int_0^inf sin(u)/(u/r) * 1/((u/r)^2 - k_pole^2) (du/r)]
    # = (1/r) * r^2 * PV[int_0^inf sin(u)/u * 1/(u^2 - (k_pole*r)^2) du]
    # = r * PV[int_0^inf sin(u)/u * 1/(u^2 - a^2) du]  where a = k_pole*r
    #
    # Using GR 3.741.3: PV[int_0^inf sin(u)/u * 1/(u^2 - a^2) du] = -pi*cos(a)/(2a^2)
    # for a > 0.
    #
    # So PV contribution = r * (-pi*cos(k_pole*r))/(2*(k_pole*r)^2)
    #                    = -pi*cos(k_pole*r)/(2*k_pole^2*r)

    I_pv_analytic = mp.mpf(0)
    if z_pole is not None and k_pole is not None:
        # Convert residue from z-space to k-space:
        # K(z) = A/(z - z_pole) where z = k^2/Lambda^2
        # In k-space: K dk = A/(k^2/Lambda^2 - z_pole) dk = A*Lambda^2/(k^2 - k_pole^2) dk
        A_k = residue_A * lam**2
        kr_pole = k_pole * r_mp
        if abs(kr_pole) > mp.mpf("1e-40"):
            # GR 3.741.3 gives the PV integral result
            I_pv_analytic = A_k * (-mp.pi * mp.cos(kr_pole)) / (2 * k_pole**2 * r_mp)
        else:
            # Small r limit: cos(0)=1, the integral is finite
            I_pv_analytic = -A_k * mp.pi / (2 * k_pole**2 * r_mp) if r_mp > 0 else mp.mpf(0)

    total = I_smooth + I_pv_analytic
    ratio = 1 + mp.mpf(2) / mp.pi * total
    return ratio


def potential_ratio_yukawa(r, *, Lambda=1.0, xi=0.0, dps=DEFAULT_DPS):
    """Local Yukawa approximation: V(r)/V_N = 1 - (4/3)e^{-m2*r} + (1/3)e^{-m0*r}.

    This is the linearized local approximation from NT-4a.
    At xi=1/6 (conformal), the scalar mode decouples.
    """
    mp.mp.dps = dps
    r_mp = mp.mpf(r)
    if r_mp <= 0:
        raise ValueError(f"r must be positive, got {r}")

    m2 = spin2_local_mass(mp.mpf(Lambda))
    m0 = scalar_local_mass(mp.mpf(Lambda), mp.mpf(xi))

    ratio = 1 - mp.mpf(4) / 3 * mp.exp(-m2 * r_mp)
    if m0 is not None:
        ratio += mp.mpf(1) / 3 * mp.exp(-m0 * r_mp)
    return ratio


def potential_ratio_ghostfree(r, *, Lambda=1.0):
    """Ghost-free IDG (Tomboulis/Modesto) potential ratio: Erf(r*Lambda/2).

    V(r)/V_N(r) = Erf(r*Lambda/2)

    This is the standard result for the exponential form factor
    exp(-Box/Lambda^2) which produces a Gaussian smearing of the source.
    """
    mp.mp.dps = DEFAULT_DPS
    r_mp = mp.mpf(r)
    lam = mp.mpf(Lambda)
    return mp.erf(r_mp * lam / 2)


def small_r_limit_yukawa(*, Lambda=1.0, xi=0.0, dps=DEFAULT_DPS):
    """The r->0 limit of V(r) in the Yukawa approximation.

    V(r->0) = -GM * [(4/3)*m_2 - (1/3)*m_0]  if m_0 exists
    V(r->0) = -inf  (diverges as -GM*(4/3)*m_2/r * ...)  if m_0 = None (xi=1/6)

    Wait, actually: V(r)/V_N(r) -> 0 as r->0 in the Yukawa case IF both
    modes exist, because:
    V(r)/V_N(r) = 1 - (4/3)e^{-m2*r} + (1/3)e^{-m0*r}
    At r=0: = 1 - 4/3 + 1/3 = 0

    So V(r) = -(GM/r) * 0 = 0 at r=0 (L'Hopital: V -> -GM*(4m2/3 - m0/3)).

    Returns the finite r->0 limit of V(r) (not the ratio).
    """
    mp.mp.dps = dps
    m2 = spin2_local_mass(mp.mpf(Lambda))
    m0 = scalar_local_mass(mp.mpf(Lambda), mp.mpf(xi))
    if m0 is None:
        return mp.ninf  # diverges: only spin-2 mode, ratio -> 1 - 4/3 = -1/3
    # V(r) ~ -GM/r * (1 - 4/3 + 1/3 + r*(4m2/3 - m0/3) + ...)
    # = -GM * (4m2/3 - m0/3) + O(r)
    return -(mp.mpf(4) / 3 * m2 - mp.mpf(1) / 3 * m0)


# ===========================================================================
# B. MASS FUNCTION AND MODIFIED METRIC
# ===========================================================================

def mass_function(r, *, Lambda=1.0, xi=0.0, G=1.0, M=1.0,
                  use_exact=False, dps=DEFAULT_DPS):
    """Effective enclosed mass m(r) from the modified potential.

    m(r) = M * [V(r)/V_N(r)]

    so that f(r) = 1 - 2*G*m(r)/r = 1 - 2*G*M/r * [V(r)/V_N(r)].

    Parameters:
        r: radius (natural units, 1/Lambda)
        use_exact: if True, use the full nonlocal integral; otherwise Yukawa
    """
    mp.mp.dps = dps
    r_mp = mp.mpf(r)
    if r_mp <= 0:
        raise ValueError(f"r must be positive, got {r}")

    if use_exact:
        ratio = potential_ratio_exact(r, Lambda=Lambda, xi=xi, dps=dps)
    else:
        ratio = potential_ratio_yukawa(r, Lambda=Lambda, xi=xi, dps=dps)

    return mp.mpf(M) * ratio


def metric_f(r, *, Lambda=1.0, xi=0.0, G=1.0, M=1.0,
             use_exact=False, dps=DEFAULT_DPS):
    """Modified Schwarzschild metric function f(r) = 1 - 2*G*m(r)/r.

    This is the g_tt = -f(r) component in Schwarzschild-like coordinates.
    """
    mp.mp.dps = dps
    r_mp = mp.mpf(r)
    if r_mp <= 0:
        raise ValueError(f"r must be positive, got {r}")

    m_r = mass_function(r, Lambda=Lambda, xi=xi, G=G, M=M,
                        use_exact=use_exact, dps=dps)
    return 1 - 2 * mp.mpf(G) * m_r / r_mp


def mass_function_derivative(r, *, Lambda=1.0, xi=0.0, M=1.0, dr=None,
                             use_exact=False, dps=DEFAULT_DPS):
    """Numerical derivative m'(r) via centered finite difference.

    Uses step size dr = r * 1e-6 by default.
    """
    mp.mp.dps = dps
    r_mp = mp.mpf(r)
    if dr is None:
        dr = r_mp * mp.mpf("1e-6")
        if dr < mp.mpf("1e-15"):
            dr = mp.mpf("1e-10") / mp.mpf(Lambda)
    dr = mp.mpf(dr)

    m_plus = mass_function(float(r_mp + dr), Lambda=Lambda, xi=xi, M=M,
                           use_exact=use_exact, dps=dps)
    m_minus = mass_function(float(r_mp - dr), Lambda=Lambda, xi=xi, M=M,
                            use_exact=use_exact, dps=dps)
    return (m_plus - m_minus) / (2 * dr)


def mass_function_second_derivative(r, *, Lambda=1.0, xi=0.0, M=1.0, dr=None,
                                    use_exact=False, dps=DEFAULT_DPS):
    """Numerical second derivative m''(r) via centered finite difference."""
    mp.mp.dps = dps
    r_mp = mp.mpf(r)
    if dr is None:
        dr = r_mp * mp.mpf("1e-4")
        if dr < mp.mpf("1e-12"):
            dr = mp.mpf("1e-8") / mp.mpf(Lambda)
    dr = mp.mpf(dr)

    m_plus = mass_function(float(r_mp + dr), Lambda=Lambda, xi=xi, M=M,
                           use_exact=use_exact, dps=dps)
    m_center = mass_function(float(r_mp), Lambda=Lambda, xi=xi, M=M,
                             use_exact=use_exact, dps=dps)
    m_minus = mass_function(float(r_mp - dr), Lambda=Lambda, xi=xi, M=M,
                            use_exact=use_exact, dps=dps)
    return (m_plus - 2 * m_center + m_minus) / (dr ** 2)


# ===========================================================================
# C. KRETSCHNER SCALAR
# ===========================================================================

def kretschner_scalar(r, *, Lambda=1.0, xi=0.0, G=1.0, M=1.0,
                      use_exact=False, dps=DEFAULT_DPS):
    """Kretschner scalar K = R_{abcd} R^{abcd} for ds^2 = -f dt^2 + dr^2/f + r^2 dOmega^2.

    For a general spherically symmetric metric with f(r) = 1 - 2 G m(r)/r:

    K = (48 G^2 m^2) / r^6
        - (48 G^2 m m') / r^5
        + (12 G^2 m'^2) / r^4
        + (8 G^2 m' m'') / r^3
        + (4 G^2 m''^2) / r^2

    Note: this is the EXACT expression for the spherically-symmetric static
    metric, not a linearized approximation (though m(r) itself comes from the
    linearized field equations).

    WARNING: The formula above assumes the Schwarzschild-coordinate form.
    The full expression for f(r) = 1 - 2Gm(r)/r is:
    K = (48 G^2/r^6)[m - r m']^2 + (8 G^2/r^2)[m'' - m'/r + m/r^2]^2

    Actually, for the exact Kretschner of ds^2 = -f(r)dt^2 + f(r)^{-1}dr^2 + r^2 dOmega^2:
    K = f''^2 + 4*(f' - (f-1)/r)^2/r^2 + 4*(f-1)^2/r^4

    Let me use this form which is more robust.
    """
    mp.mp.dps = dps
    r_mp = mp.mpf(r)
    g = mp.mpf(G)

    # Compute m(r), m'(r), m''(r)
    m = mass_function(r, Lambda=Lambda, xi=xi, G=1.0, M=M,
                      use_exact=use_exact, dps=dps)
    mp_val = mass_function_derivative(r, Lambda=Lambda, xi=xi, M=M,
                                      use_exact=use_exact, dps=dps)
    mpp = mass_function_second_derivative(r, Lambda=Lambda, xi=xi, M=M,
                                          use_exact=use_exact, dps=dps)

    # f(r) = 1 - 2Gm/r
    # f'(r) = 2Gm/r^2 - 2Gm'/r = 2G(m - r*m')/r^2
    # f''(r) = 2G(-m/r^3 + m'/r^2 - m'/r^2 + r*m''/r^2 - m''/r + ...) -- messy
    # Better to use: f'' = d/dr[2G(m - r*m')/r^2] = 2G[(-2m + 2r*m' - r^2*m'')/r^3]
    # Wait, let me compute carefully:
    # f = 1 - 2Gm/r
    # f' = -2G(m'*r - m)/r^2 = 2G(m - r*m')/r^2
    # f'' = 2G * d/dr[(m - r*m')/r^2]
    #     = 2G * [(m' - m' - r*m'')*r^2 - (m - r*m')*2r] / r^4
    #     = 2G * [-r*m''*r^2 - 2r*(m - r*m')] / r^4
    #     = 2G * [-r^3*m'' - 2r*m + 2r^2*m'] / r^4
    #     = 2G * (-r^2*m'' - 2m + 2r*m') / r^3
    #     = 2G * (2r*m' - 2m - r^2*m'') / r^3

    f = 1 - 2 * g * m / r_mp
    f_prime = 2 * g * (m - r_mp * mp_val) / r_mp**2
    f_double_prime = 2 * g * (2 * r_mp * mp_val - 2 * m - r_mp**2 * mpp) / r_mp**3

    # Kretschner for static spherically symmetric: K = f''^2 + 2(f')^2/r^2 + 2((f-1)/r)^2 * 4/r^2
    # No wait. For the metric ds^2 = -f dt^2 + dr^2/f + r^2 dOmega^2:
    # The nonzero Riemann components (up to symmetries) are:
    # R^{tr}_{tr} = f''/2
    # R^{ttheta}_{ttheta} = f'/(2r)
    # R^{rtheta}_{rtheta} = -f'/(2r)
    # R^{theta phi}_{theta phi} = (1-f)/r^2
    # and cyclic. The Kretschner scalar is:
    # K = 4(R^{tr}_{tr})^2 + 8(R^{ttheta}_{ttheta})^2 + 4(R^{theta phi}_{theta phi})^2
    # Wait, I need to be more careful. Let me use the standard result.

    # The exact formula for a metric g_{tt} = -A(r), g_{rr} = 1/A(r) (i.e. A=f):
    # K = (A'')^2 + (2A'/r - 2(A-1)/r^2)^2 * (2/r^2) ... no.
    #
    # From Stephani (exact solutions, Ch. 15) or direct computation:
    # For ds^2 = -f(r) dt^2 + f(r)^{-1} dr^2 + r^2 d\Omega^2:
    # R_{trtr} = -f''/2
    # R_{t\theta t\theta} = -r f'/2
    # R_{r\theta r\theta} = r f'/(2f)  -- NO, this depends on the coordinate basis
    #
    # In an orthonormal frame:
    # R_{0101} = f''/2 (where 0=t, 1=r)
    # R_{0202} = R_{0303} = f'/(2r)
    # R_{1212} = R_{1313} = -f'/(2r)
    # R_{2323} = (1-f)/r^2
    #
    # K = R_{abcd}R^{abcd} = sum over all components squared (with appropriate multiplicity)
    # = 4*R_{0101}^2 + 8*R_{0202}^2 + 8*R_{1212}^2 + 4*R_{2323}^2
    # Wait, need to count correctly:
    # Independent: (0101), (0202), (0303), (1212), (1313), (2323)
    # With symmetry 0202=0303, 1212=1313:
    # K = R_{0101}^2 * (count: 1 component * 4 from symmetries? No.)
    #
    # The full contraction gives (standard result, e.g., Wald p.50 for vacuum):
    # K = f''^2/4 * 4 + (f'/(2r))^2 * 8 + ((1-f)/r^2)^2 * 4  ? No.
    #
    # Let me just use the well-known formula directly.
    # For f(r) = 1 - 2Gm(r)/r:
    #
    # K = (48 G^2/r^6) * [m(r) - r*m'(r)]^2
    #   + (8 G^2/r^2) * [m''(r)]^2
    #   - (32 G^2/r^3) * [m(r) - r*m'(r)] * m''(r)
    #   + ... (terms from cross products)
    #
    # Actually the cleanest route is to use the direct f-based formula.
    # After careful computation (MTW or Poisson "A Relativist's Toolkit"):
    #
    # K = (f'')^2 + 4/r^2 * (f')^2 + 4/r^4 * (f - 1)^2
    #                              -- NO, this is NOT right either for this metric.
    #
    # I will compute this rigorously from the curvature components.

    # CORRECT formula (see e.g. arXiv:0911.4619 Eq. 2.5, or Visser 1996):
    # For ds^2 = -e^{2Phi(r)} f(r) dt^2 + f(r)^{-1} dr^2 + r^2 dOmega^2
    # with Phi=0 (our case):
    # K = (f'')^2 + (2/r^2)(2r f' + 2f - 2)^2/4 ... hmm, still messy.
    #
    # Let me just use m(r) directly. For f = 1 - 2Gm/r:
    # f - 1 = -2Gm/r
    # f' = 2G(m - rm')/r^2
    # f'' = 2G(2rm' - 2m - r^2 m'')/r^3
    #
    # The Kretschner scalar for the Schwarzschild-like metric is:
    # K = 4/3 * (f'')^2 + 8/3 * [(f')^2 - 2f'(f-1)/r + (f-1)^2/r^2] * (1/r^2 + f''/...)
    # This is getting complicated. Let me use the clean expression in terms of m.
    #
    # From Ansoldi (arXiv:0802.0330, Eq. 2.4):
    # K = R_{abcd}R^{abcd} = (48G^2/r^6)*m^2 for Schwarzschild.
    # For m(r) varying, the full expression is:
    # K = (48G^2/r^6)[m - rm' + r^2 m''/2]^2 + (8G^2/r^4)[m' - rm'']^2 + ...
    #
    # No, I should just compute it properly. The orthonormal-frame Riemann components
    # for ds^2 = -f dt^2 + dr^2/f + r^2 dOmega^2 are:
    #
    # R_{0101} = f''/2
    # R_{0202} = R_{0303} = f'/(2r)
    # R_{1212} = R_{1313} = -f'/(2r)
    # R_{2323} = (1-f)/r^2
    #
    # These satisfy: R_{0101} + R_{0202} + R_{0303} = Ricci component, etc.
    #
    # Wait, but R_{1212} = -f'/(2r) is wrong. Let me recompute.
    # In the orthonormal frame e^0 = sqrt(f) dt, e^1 = dr/sqrt(f), e^2 = r dtheta, e^3 = r sin(theta) dphi:
    # The connection 1-forms give:
    # omega^{01} = f'/(2sqrt(f)) e^0 / sqrt(f) = f'/2 * dt
    # Actually let me use the known result from Poisson (2004, A Relativist's Toolkit, Appendix):
    #
    # For f(r) = 1 - 2m(r)/r (setting G=1 for now):
    # R_{0101} = (-2m + 2rm' - r^2 m'') / r^3  -- no, sign?
    #
    # OK, let me use the direct and well-tested result. I will substitute f directly.
    # With f = 1 - 2Gm/r:
    #
    # Component A = f''/2 = G(2rm' - 2m - r^2 m'')/r^3
    # Component B = f'/(2r) = G(m - rm')/(r^3)
    # Component C = (1-f)/(r^2) = 2Gm/r^3
    #
    # The 6 independent Riemann components (orthonormal frame, Petrov type D):
    # R_{0101} = -A = -f''/2  (note: sign depends on convention)
    # R_{0202} = R_{0303} = -B = -f'/(2r)  ? Or +B?
    # R_{1212} = R_{1313} = B = f'/(2r)  ? Or -B?
    # R_{2323} = C = (1-f)/r^2
    #
    # For Schwarzschild (m=const, m'=m''=0):
    # A = f''/2 = G(-2m)/r^3 = ... f = 1-2Gm/r, f' = 2Gm/r^2, f'' = -4Gm/r^3
    # A = f''/2 = -2Gm/r^3
    # B = f'/(2r) = Gm/r^3
    # C = 2Gm/r^3
    #
    # K_Schw = 48G^2 m^2/r^6
    # = 4*A^2 + 8*B^2 + 4*C^2 ?
    # = 4*(4G^2m^2/r^6) + 8*(G^2m^2/r^6) + 4*(4G^2m^2/r^6)
    # = 16 + 8 + 16 = 40... not 48.
    #
    # Hmm. Let me count the multiplicity correctly.
    # The Kretschner scalar is K = R^{abcd} R_{abcd}.
    # For orthonormal basis, K = sum_{a<b,c<d} R_{abcd}^2 * multiplicity
    # But the sum runs over ALL a,b,c,d (with the symmetries R_{abcd}=R_{cdab}=-R_{bacd}).
    # So K = sum_{a,b,c,d} R_{abcd}^2 = 4 * sum_{a<b, c<d} R_{abcd}^2.
    # Wait no: R_{abcd} with a<b, c<d has (6 choose 1)*(6 choose 1) = 36 but with
    # R_{abcd}=R_{cdab}, we get 6+15=21 independent, but the sum still runs over all 4! arrangements...
    #
    # K = sum_{all a,b,c,d} R_{abcd}^2
    # The symmetries R_{abcd} = -R_{bacd} = -R_{abdc} = R_{cdab} mean:
    # K = sum = 4 * sum_{a<b, c<d} |R_{abcd}|^2 * (1 if (ab)=(cd), else 2)
    # No, actually K = sum_{all} = 4 * sum_{a<b} sum_{c<d} R_{abcd}^2
    # because each (a,b) with a<b appears 2 times in the full sum (ab and ba), etc.
    # So K = 4 * sum_{a<b, c<d} R_{abcd}^2.
    # But then also R_{abcd} = R_{cdab}, so pairs with (ab) != (cd) are counted twice.
    # K = 4 * [sum_{(ab)=(cd)} R_{abab}^2 + 2 * sum_{(ab)<(cd)} R_{abcd}^2]
    # For Petrov type D with the specific structure above:
    # The only nonzero are: (01,01), (02,02), (03,03), (12,12), (13,13), (23,23),
    #                        and (01,23), (02,13), (03,12) [Weyl components].
    #
    # For the simple static spherical metric, there are additional relations. Let me
    # just use the well-known result directly.
    #
    # FINAL CORRECT FORMULA (verified against Schwarzschild):
    # For f(r) = 1 - 2Gm(r)/r:
    # K = (48 G^2 m^2)/r^6                      [Schwarzschild piece]
    #   - (96 G^2 m m' r)/r^6 + (48 G^2 m'^2 r^2)/r^6
    #   + (8 G^2 m''^2 r^4)/r^6
    #   ...no, this still doesn't work cleanly.
    #
    # Let me just compute K = f''^2 + 2(f'/r)^2 + 2((f-1)/r^2)^2
    # and check against Schwarzschild:
    # f = 1-2Gm/r: f'' = -4Gm/r^3, f'/r = 2Gm/r^3, (f-1)/r^2 = -2Gm/r^3
    # K_check = 16G^2m^2/r^6 + 2*4G^2m^2/r^6 + 2*4G^2m^2/r^6 = 16+8+8 = 32. Not 48.
    #
    # Try K = f''^2 + 4(f'/r)^2 + 4((f-1)/r^2)^2:
    # = 16 + 16 + 16 = 48. YES!
    #
    # So: K = (f'')^2 + (2f'/r)^2 + (2(f-1)/r^2)^2
    #       = (f'')^2 + 4(f')^2/r^2 + 4(f-1)^2/r^4

    K = f_double_prime**2 + 4 * f_prime**2 / r_mp**2 + 4 * (f - 1)**2 / r_mp**4
    return K


def kretschner_schwarzschild(r, *, G=1.0, M=1.0):
    """Standard Schwarzschild Kretschner scalar: K = 48 G^2 M^2 / r^6."""
    mp.mp.dps = DEFAULT_DPS
    return 48 * mp.mpf(G)**2 * mp.mpf(M)**2 / mp.mpf(r)**6


# ===========================================================================
# D. ENERGY CONDITIONS
# ===========================================================================

def energy_density(r, *, Lambda=1.0, xi=0.0, G=1.0, M=1.0,
                   use_exact=False, dps=DEFAULT_DPS):
    """Effective energy density rho_eff = m'(r) / (4 pi r^2).

    From the Einstein equation with effective stress-energy:
    G^t_t = -8 pi G rho -> rho = m'(r)/(4 pi r^2)  (in G=1 units).
    """
    mp.mp.dps = dps
    r_mp = mp.mpf(r)
    mp_val = mass_function_derivative(r, Lambda=Lambda, xi=xi, M=M,
                                      use_exact=use_exact, dps=dps)
    return mp_val / (4 * mp.pi * r_mp**2)


def radial_pressure(r, *, Lambda=1.0, xi=0.0, G=1.0, M=1.0,
                    use_exact=False, dps=DEFAULT_DPS):
    """Effective radial pressure p_r = -rho.

    For the metric ansatz f(r) = 1 - 2Gm(r)/r, the diagonal Einstein
    equations give p_r = -rho_eff exactly (from G^r_r = -8piG p_r).
    """
    return -energy_density(r, Lambda=Lambda, xi=xi, G=G, M=M,
                           use_exact=use_exact, dps=dps)


def tangential_pressure(r, *, Lambda=1.0, xi=0.0, G=1.0, M=1.0,
                        use_exact=False, dps=DEFAULT_DPS):
    """Effective tangential pressure p_t.

    From the theta-theta Einstein equation:
    p_t = -rho - r*rho'/2

    where rho' = d(rho)/dr is computed numerically.
    """
    mp.mp.dps = dps
    r_mp = mp.mpf(r)
    rho = energy_density(r, Lambda=Lambda, xi=xi, G=G, M=M,
                         use_exact=use_exact, dps=dps)

    # Numerical derivative of rho
    dr = r_mp * mp.mpf("1e-4")
    if dr < mp.mpf("1e-12") / mp.mpf(Lambda):
        dr = mp.mpf("1e-8") / mp.mpf(Lambda)

    rho_plus = energy_density(float(r_mp + dr), Lambda=Lambda, xi=xi, G=G, M=M,
                              use_exact=use_exact, dps=dps)
    rho_minus = energy_density(float(r_mp - dr), Lambda=Lambda, xi=xi, G=G, M=M,
                               use_exact=use_exact, dps=dps)
    drho_dr = (rho_plus - rho_minus) / (2 * dr)

    return -rho - r_mp * drho_dr / 2


def check_nec_radial(r, **kwargs):
    """Null energy condition (radial): rho + p_r >= 0.

    For our metric: rho + p_r = 0 identically (marginal NEC saturation).
    """
    rho = energy_density(r, **kwargs)
    pr = radial_pressure(r, **kwargs)
    return rho + pr


def check_nec_tangential(r, **kwargs):
    """Null energy condition (tangential): rho + p_t >= 0."""
    rho = energy_density(r, **kwargs)
    pt = tangential_pressure(r, **kwargs)
    return rho + pt


def check_sec(r, **kwargs):
    """Strong energy condition: rho + p_r + 2*p_t >= 0.

    Equals rho + p_r + 2*p_t = 0 + 2*(-rho - r*rho'/2) = -2*rho - r*rho'
    = -r * d(r^2 * rho * 2/r) ... hmm, let me just compute directly:
    = rho + (-rho) + 2*(-rho - r*rho'/2) = -2*rho - r*rho'
    """
    rho = energy_density(r, **kwargs)
    pr = radial_pressure(r, **kwargs)
    pt = tangential_pressure(r, **kwargs)
    return rho + pr + 2 * pt


def raychaudhuri_focusing(r, *, Lambda=1.0, xi=0.0, G=1.0, M=1.0,
                          use_exact=False, dps=DEFAULT_DPS):
    """R_{mu nu} k^mu k^nu for radial null geodesics.

    The focusing condition for the Raychaudhuri equation is R_{ab} k^a k^b >= 0.
    Via Einstein equations: R_{ab} k^a k^b = 8 pi G (rho + p_r) for radial null.
    For tangential null: R_{ab} k^a k^b = 8 pi G (rho + p_t).

    Since rho + p_r = 0 (marginal), the radial focusing is trivially zero.
    The tangential focusing requires checking rho + p_t.
    """
    # Tangential null focusing is the nontrivial one
    return check_nec_tangential(r, Lambda=Lambda, xi=xi, G=G, M=M,
                                use_exact=use_exact, dps=dps)


# ===========================================================================
# E. HORIZON STRUCTURE
# ===========================================================================

def find_horizons(*, Lambda=1.0, xi=0.0, G=1.0, M=1.0,
                  r_min=0.01, r_max=100.0, n_scan=2000,
                  use_exact=False, dps=DEFAULT_DPS):
    """Find zeros of f(r) = 0 (horizons).

    Scans [r_min, r_max] and refines with bisection.

    Returns:
        List of horizon radii in ascending order.
    """
    mp.mp.dps = dps
    radii = [mp.mpf(r_min) + (mp.mpf(r_max) - mp.mpf(r_min)) * i / n_scan
             for i in range(n_scan + 1)]
    values = [metric_f(float(r), Lambda=Lambda, xi=xi, G=G, M=M,
                       use_exact=use_exact, dps=dps) for r in radii]

    horizons = []
    for i in range(n_scan):
        if values[i] * values[i + 1] < 0:
            # Bisection
            lo, hi = float(radii[i]), float(radii[i + 1])
            for _ in range(100):
                mid = (lo + hi) / 2
                f_mid = metric_f(mid, Lambda=Lambda, xi=xi, G=G, M=M,
                                 use_exact=use_exact, dps=dps)
                if f_mid * metric_f(lo, Lambda=Lambda, xi=xi, G=G, M=M,
                                    use_exact=use_exact, dps=dps) < 0:
                    hi = mid
                else:
                    lo = mid
            horizons.append((lo + hi) / 2)

    return sorted(horizons)


# ===========================================================================
# F. COMPARISON: SCT vs GHOST-FREE IDG
# ===========================================================================

def comparison_table(radii_over_lambda, *, Lambda=1.0, xi=0.0, dps=DEFAULT_DPS):
    """Build a comparison table of potential ratios for SCT (Yukawa), SCT (exact), and IDG.

    Parameters:
        radii_over_lambda: list of r*Lambda values

    Returns:
        List of dicts with keys: r_Lambda, yukawa, exact, idg
    """
    results = []
    for r_L in radii_over_lambda:
        r = float(r_L) / float(Lambda)
        row = {
            "r_Lambda": float(r_L),
            "yukawa": float(potential_ratio_yukawa(r, Lambda=Lambda, xi=xi, dps=dps)),
            "idg": float(potential_ratio_ghostfree(r, Lambda=Lambda)),
        }
        try:
            row["exact"] = float(potential_ratio_exact(
                r, Lambda=Lambda, xi=xi, dps=dps, k_max_factor=30.0, n_quad=1000))
        except Exception as exc:
            row["exact"] = None
            row["exact_error"] = str(exc)
        results.append(row)
    return results


# ===========================================================================
# MAIN ANALYSIS: run all computations and generate report
# ===========================================================================

def run_full_analysis(*, Lambda=1.0, xi=0.0, G=1.0, M=1.0, dps=DEFAULT_DPS):
    """Execute the complete MR-9 singularity resolution analysis.

    Returns a dict with all results.
    """
    mp.mp.dps = dps
    report = {
        "phase": "MR-9",
        "parameters": {
            "Lambda": float(Lambda),
            "xi": float(xi),
            "G": float(G),
            "M": float(M),
            "alpha_C": float(ALPHA_C),
            "c2": float(C2),
        },
        "disclaimer": (
            "ALL results are from LINEARIZED field equations (NT-4a). "
            "The mass function m(r) is obtained from the linearized propagator "
            "modification, then inserted into the Schwarzschild ansatz. "
            "Full nonlinear self-consistent solution would require solving "
            "the NT-4b equations, which is beyond current scope."
        ),
    }

    # -- Effective masses --
    m2 = spin2_local_mass(mp.mpf(Lambda))
    m0 = scalar_local_mass(mp.mpf(Lambda), mp.mpf(xi))
    report["masses"] = {
        "m2": float(m2),
        "m2_over_Lambda": float(m2 / mp.mpf(Lambda)),
        "m0": float(m0) if m0 is not None else None,
        "m0_over_Lambda": float(m0 / mp.mpf(Lambda)) if m0 is not None else None,
    }

    # -- Ghost pole --
    try:
        z_pole = find_first_positive_real_tt_zero(xi=xi, dps=dps)
        report["ghost_pole"] = {
            "z_pole": float(z_pole),
            "k_pole_over_Lambda": float(mp.sqrt(z_pole)),
        }
    except ValueError:
        report["ghost_pole"] = {"z_pole": None}

    # -- Potential ratios --
    r_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    yukawa_data = []
    idg_data = []
    for r_L in r_values:
        r = r_L / float(Lambda)
        y = float(potential_ratio_yukawa(r, Lambda=Lambda, xi=xi, dps=dps))
        g_val = float(potential_ratio_ghostfree(r, Lambda=Lambda))
        yukawa_data.append({"r_Lambda": r_L, "ratio": y})
        idg_data.append({"r_Lambda": r_L, "ratio": g_val})

    report["potential_yukawa"] = yukawa_data
    report["potential_idg"] = idg_data

    # -- r->0 limit (Yukawa) --
    v0_limit = small_r_limit_yukawa(Lambda=Lambda, xi=xi, dps=dps)
    report["small_r_limit"] = {
        "V_over_GM": float(v0_limit),
        "is_finite": mp.isfinite(v0_limit),
        "note": "V(r->0) = -GM * (4m2/3 - m0/3) in Yukawa approximation",
    }

    # -- Yukawa V(r)/V_N(r) at r=0 --
    ratio_at_zero = 1 - mp.mpf(4) / 3
    if m0 is not None:
        ratio_at_zero += mp.mpf(1) / 3
    report["ratio_at_r0_yukawa"] = float(ratio_at_zero)
    report["ratio_at_r0_note"] = (
        "V(r)/V_N(r) -> 1 - 4/3 + 1/3 = 0 if both modes present (xi != 1/6). "
        "V(r)/V_N(r) -> 1 - 4/3 = -1/3 if only spin-2 mode (xi = 1/6). "
        f"For xi = {float(xi)}: ratio = {float(ratio_at_zero)}. "
        "V(r) = -(GM/r) * ratio -> 0 if ratio=0 (finite), diverges if ratio != 0."
    )

    # -- Kretschner at selected radii --
    kretschner_data = []
    r_kretschner = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for r_L in r_kretschner:
        r = r_L / float(Lambda)
        try:
            K_sct = float(kretschner_scalar(r, Lambda=Lambda, xi=xi, G=G, M=M, dps=dps))
            K_schw = float(kretschner_schwarzschild(r, G=G, M=M))
            kretschner_data.append({
                "r_Lambda": r_L,
                "K_SCT": K_sct,
                "K_Schwarzschild": K_schw,
                "ratio_K": K_sct / K_schw if K_schw != 0 else None,
            })
        except Exception as exc:
            kretschner_data.append({"r_Lambda": r_L, "error": str(exc)})
    report["kretschner"] = kretschner_data

    # -- Energy conditions at selected radii --
    ec_data = []
    for r_L in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
        r = r_L / float(Lambda)
        try:
            rho = float(energy_density(r, Lambda=Lambda, xi=xi, G=G, M=M, dps=dps))
            pr = float(radial_pressure(r, Lambda=Lambda, xi=xi, G=G, M=M, dps=dps))
            pt = float(tangential_pressure(r, Lambda=Lambda, xi=xi, G=G, M=M, dps=dps))
            nec_r = float(check_nec_radial(r, Lambda=Lambda, xi=xi, G=G, M=M, dps=dps))
            nec_t = float(check_nec_tangential(r, Lambda=Lambda, xi=xi, G=G, M=M, dps=dps))
            sec = float(check_sec(r, Lambda=Lambda, xi=xi, G=G, M=M, dps=dps))
            ec_data.append({
                "r_Lambda": r_L,
                "rho": rho, "p_r": pr, "p_t": pt,
                "NEC_radial": nec_r, "NEC_tangential": nec_t,
                "SEC": sec,
                "NEC_radial_satisfied": nec_r >= -1e-30,
                "NEC_tangential_satisfied": nec_t >= -1e-30,
                "SEC_satisfied": sec >= -1e-30,
            })
        except Exception as exc:
            ec_data.append({"r_Lambda": r_L, "error": str(exc)})
    report["energy_conditions"] = ec_data

    # -- Horizon structure --
    # Use a realistic BH: M in Planck units, G=1
    # For a solar mass BH: r_S = 2GM ~ 2M in natural units
    # We need r_S >> 1/Lambda for the external region to be Schwarzschild-like
    # Set M so that r_S = 2GM = 10/Lambda (a "small" BH at the scale Lambda)
    M_bh = mp.mpf(5) / (mp.mpf(G) * mp.mpf(Lambda))
    try:
        horizons = find_horizons(
            Lambda=Lambda, xi=xi, G=G, M=float(M_bh),
            r_min=0.01 / float(Lambda), r_max=20.0 / float(Lambda),
            n_scan=1000, dps=dps)
        report["horizons"] = {
            "M_test": float(M_bh),
            "r_S_expected": float(2 * mp.mpf(G) * M_bh),
            "horizons_found": [float(h) for h in horizons],
            "n_horizons": len(horizons),
        }
    except Exception as exc:
        report["horizons"] = {"error": str(exc)}

    # -- KRETSCHNER SCALING --
    # Near r=0 in the Yukawa approximation:
    # V/V_N ~ r * C where C = (4m_2/3 - m_0/3)
    # m(r) ~ M * r * C (linear in r)
    # f(r) = 1 - 2GM*C (approaches a CONSTANT, not 1)
    # K ~ 4*(2GMC)^2 / r^4  (STILL DIVERGES, but as 1/r^4 not 1/r^6)
    C_yukawa = mp.mpf(4) / 3 * m2
    if m0 is not None:
        C_yukawa -= mp.mpf(1) / 3 * m0
    f_at_zero = 1 - 2 * mp.mpf(G) * mp.mpf(M) * C_yukawa
    K_coeff = 4 * (2 * mp.mpf(G) * mp.mpf(M) * C_yukawa) ** 2
    report["core_analysis"] = {
        "C_yukawa": float(C_yukawa),
        "f_at_zero": float(f_at_zero),
        "K_leading_coefficient": float(K_coeff),
        "K_scaling": "1/r^4 (softened from Schwarzschild 1/r^6)",
        "de_sitter_core": False,
        "note": (
            "The Yukawa (local pole) approximation gives m(r) ~ C*r near r=0, "
            "which produces K ~ const/r^4. A true de Sitter core requires "
            "m(r) ~ r^3, which would need the full nonlocal form factor "
            "(not just the first poles). The Yukawa approximation softens "
            "the singularity by 2 powers of r but does NOT resolve it."
        ),
    }

    # -- VERDICT --
    both_modes = m0 is not None
    if both_modes:
        verdict = (
            "LINEARIZED RESULT (Yukawa approximation): "
            f"With both spin-2 and spin-0 modes present (xi = {float(xi)}), "
            f"V(r)/V_N(r) -> 0 as r -> 0, hence V(0) = -GM * {float(C_yukawa):.6f} (FINITE). "
            "The 1/r Newtonian divergence is removed and V(0) is finite. "
            f"However, the mass function m(r) ~ {float(C_yukawa):.4f} * M * r near r=0 "
            "(LINEAR, not cubic), which gives the metric function "
            f"f(0) = {float(f_at_zero):.4f} (constant, NOT 1). "
            "CRITICAL: The Kretschner scalar K STILL DIVERGES as "
            f"K ~ {float(K_coeff):.2f} / r^4 (softened from 48/(r^6) to const/r^4, "
            "i.e., two powers of r softer than Schwarzschild, but NOT finite). "
            "The singularity is SOFTENED but NOT RESOLVED in the Yukawa approximation. "
            "A true de Sitter core (K -> const) would require m(r) ~ r^3, which needs "
            "the full nonlocal integral (not just the first poles). "
            "CAVEATS: (1) This is a linearized result; nonlinear backreaction may differ. "
            "(2) The fakeon prescription is an additional assumption. "
            "(3) The full nonlocal potential (not Yukawa) may give different r->0 behavior. "
            "(4) At xi=1/6, the scalar mode decouples and V(r) still diverges as 1/r."
        )
    else:
        verdict = (
            "LINEARIZED RESULT: At conformal coupling (xi = 1/6), the scalar mode "
            "decouples and V(r)/V_N(r) -> -1/3 as r -> 0. "
            "The potential V(r) ~ (GM)/(3r) STILL DIVERGES, just with reduced strength "
            "(1/3 instead of 1). The singularity is NOT resolved in this case. "
            "The Kretschner scalar still diverges as 48/(9r^6) (reduced by factor 1/9). "
            "CONCLUSION: Singularity resolution requires xi != 1/6 (scalar mode active)."
        )
    report["verdict"] = verdict

    return report


# ===========================================================================
# FIGURE GENERATION
# ===========================================================================

def generate_potential_figure(*, Lambda=1.0, xi=0.0, dps=DEFAULT_DPS):
    """Figure 1: V(r)/V_N(r) comparison — SCT Yukawa, ghost-free IDG."""
    init_style()
    fig, ax = create_figure(figsize=(5.0, 3.5))

    r_Lambda_vals = np.linspace(0.01, 10.0, 300)

    # SCT Yukawa
    yukawa_vals = []
    for r_L in r_Lambda_vals:
        r = r_L / float(Lambda)
        yukawa_vals.append(float(potential_ratio_yukawa(r, Lambda=Lambda, xi=xi, dps=30)))
    ax.plot(r_Lambda_vals, yukawa_vals,
            color=SCT_COLORS['prediction'], linewidth=2.0,
            label=r'SCT Yukawa ($\xi=0$)')

    # SCT Yukawa at conformal coupling
    if abs(xi) < 0.01:
        yukawa_conf = []
        for r_L in r_Lambda_vals:
            r = r_L / float(Lambda)
            yukawa_conf.append(float(potential_ratio_yukawa(r, Lambda=Lambda, xi=1/6, dps=30)))
        ax.plot(r_Lambda_vals, yukawa_conf,
                color=SCT_COLORS['scalar'], linewidth=1.5, linestyle='--',
                label=r'SCT Yukawa ($\xi=1/6$)')

    # Ghost-free IDG
    idg_vals = []
    for r_L in r_Lambda_vals:
        r = r_L / float(Lambda)
        idg_vals.append(float(potential_ratio_ghostfree(r, Lambda=Lambda)))
    ax.plot(r_Lambda_vals, idg_vals,
            color=SCT_COLORS['reference'], linewidth=1.5, linestyle='-.',
            label='Ghost-free IDG')

    # GR reference
    ax.axhline(y=1.0, color='black', linewidth=0.5, linestyle=':')

    ax.set_xlabel(r'$r \cdot \Lambda$')
    ax.set_ylabel(r'$V(r) / V_{\mathrm{Newton}}(r)$')
    ax.set_title('Modified Newtonian Potential — MR-9')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.5, 1.3)
    ax.set_xlim(0, 10)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(fig, "mr9_potential_comparison", fmt="pdf", directory=FIGURES_DIR)
    plt.close(fig)
    return fig


def generate_kretschner_figure(*, Lambda=1.0, xi=0.0, G=1.0, M=1.0, dps=DEFAULT_DPS):
    """Figure 2: Kretschner scalar comparison — SCT vs Schwarzschild."""
    init_style()
    fig, ax = create_figure(figsize=(5.0, 3.5))

    r_Lambda_vals = np.logspace(-1, 1.5, 200)

    K_sct = []
    K_schw = []
    for r_L in r_Lambda_vals:
        r = r_L / float(Lambda)
        try:
            k_s = float(kretschner_scalar(r, Lambda=Lambda, xi=xi, G=G, M=M, dps=30))
        except Exception:
            k_s = np.nan
        k_schw = float(kretschner_schwarzschild(r, G=G, M=M))
        K_sct.append(k_s)
        K_schw.append(k_schw)

    ax.loglog(r_Lambda_vals, K_sct,
              color=SCT_COLORS['prediction'], linewidth=2.0,
              label=r'SCT ($\xi=0$)')
    ax.loglog(r_Lambda_vals, K_schw,
              color='black', linewidth=1.0, linestyle=':',
              label='Schwarzschild')

    ax.set_xlabel(r'$r \cdot \Lambda$')
    ax.set_ylabel(r'$K = R_{\mu\nu\rho\sigma} R^{\mu\nu\rho\sigma}$')
    ax.set_title('Kretschner Scalar — MR-9')
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(fig, "mr9_kretschner", fmt="pdf", directory=FIGURES_DIR)
    plt.close(fig)
    return fig


def generate_energy_conditions_figure(*, Lambda=1.0, xi=0.0, G=1.0, M=1.0, dps=DEFAULT_DPS):
    """Figure 3: Energy conditions — rho, p_r, p_t, NEC, SEC."""
    init_style()
    fig, ax = create_figure(figsize=(5.0, 3.5))

    r_Lambda_vals = np.linspace(0.05, 10.0, 200)

    rho_vals, pr_vals, pt_vals, nec_t_vals, sec_vals = [], [], [], [], []
    for r_L in r_Lambda_vals:
        r = r_L / float(Lambda)
        try:
            rho = float(energy_density(r, Lambda=Lambda, xi=xi, G=G, M=M, dps=30))
            pr = float(radial_pressure(r, Lambda=Lambda, xi=xi, G=G, M=M, dps=30))
            pt = float(tangential_pressure(r, Lambda=Lambda, xi=xi, G=G, M=M, dps=30))
            nec_t = float(check_nec_tangential(r, Lambda=Lambda, xi=xi, G=G, M=M, dps=30))
            sec = float(check_sec(r, Lambda=Lambda, xi=xi, G=G, M=M, dps=30))
        except Exception:
            rho = pr = pt = nec_t = sec = np.nan
        rho_vals.append(rho)
        pr_vals.append(pr)
        pt_vals.append(pt)
        nec_t_vals.append(nec_t)
        sec_vals.append(sec)

    ax.plot(r_Lambda_vals, rho_vals,
            color=SCT_COLORS['prediction'], linewidth=1.5, label=r'$\rho$')
    ax.plot(r_Lambda_vals, pr_vals,
            color=SCT_COLORS['scalar'], linewidth=1.5, linestyle='--', label=r'$p_r$')
    ax.plot(r_Lambda_vals, pt_vals,
            color=SCT_COLORS['vector'], linewidth=1.5, linestyle='-.', label=r'$p_t$')
    ax.plot(r_Lambda_vals, nec_t_vals,
            color=SCT_COLORS['dirac'], linewidth=1.5, linestyle=':', label=r'$\rho + p_t$ (NEC)')
    ax.plot(r_Lambda_vals, sec_vals,
            color=SCT_COLORS['data'], linewidth=1.5, linestyle='--', label=r'SEC')

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xlabel(r'$r \cdot \Lambda$')
    ax.set_ylabel('Effective stress-energy components')
    ax.set_title('Energy Conditions — MR-9')
    ax.legend(fontsize=7, loc='upper right')
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_figure(fig, "mr9_energy_conditions", fmt="pdf", directory=FIGURES_DIR)
    plt.close(fig)
    return fig


# ===========================================================================
# CLI
# ===========================================================================

def _build_arg_parser():
    parser = argparse.ArgumentParser(description="MR-9 BH singularity resolution analysis.")
    parser.add_argument("--xi", type=float, default=0.0)
    parser.add_argument("--lambda-scale", type=float, default=1.0)
    parser.add_argument("--dps", type=int, default=DEFAULT_DPS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--figures", action="store_true", default=False,
                        help="Generate publication figures.")
    return parser


def main():
    args = _build_arg_parser().parse_args()
    Lambda = args.lambda_scale
    xi = args.xi
    dps = args.dps

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MR-9: Black Hole Singularity Resolution in the Spectral Action")
    print("=" * 72)
    print(f"Parameters: Lambda = {Lambda}, xi = {xi}, dps = {dps}")
    print()

    report = run_full_analysis(Lambda=Lambda, xi=xi, dps=dps)

    output_path = args.output or RESULTS_DIR / "mr9_analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport written to: {output_path}")

    if args.figures:
        print("\nGenerating figures...")
        generate_potential_figure(Lambda=Lambda, xi=xi, dps=dps)
        generate_kretschner_figure(Lambda=Lambda, xi=xi, dps=dps)
        generate_energy_conditions_figure(Lambda=Lambda, xi=xi, dps=dps)
        print("Figures written to:", FIGURES_DIR)

    # Print key results
    print("\n" + "=" * 72)
    print("KEY RESULTS")
    print("=" * 72)
    print(f"\nMasses: m_2/Lambda = {report['masses']['m2_over_Lambda']:.6f}, "
          f"m_0/Lambda = {report['masses'].get('m0_over_Lambda', 'N/A')}")
    print(f"\nV(r->0)/V_N: {report['ratio_at_r0_yukawa']} (Yukawa)")
    sl = report["small_r_limit"]
    print(f"V(r->0)/(GM): {sl['V_over_GM']:.6f} (finite: {sl['is_finite']})")
    print(f"\nGhost pole: z = {report['ghost_pole'].get('z_pole', 'N/A')}")

    print("\n" + "-" * 72)
    print("VERDICT:")
    print(report["verdict"])
    print("=" * 72)

    return report


if __name__ == "__main__":
    main()
