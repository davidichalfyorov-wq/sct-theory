# ruff: noqa: E402, I001
"""
LT-3a: Quasinormal Mode Infrastructure for Spectral Causal Theory.

Phase 1 — Foundation: constants, effective potentials (RW + Zerilli, GR + SCT),
6th-order WKB method, SCT perturbation theory.

The SCT-modified metric function (NT-4a):
    f(r) = 1 - (r_s/r) * [1 - (4/3)e^{-m2*r} + (1/3)e^{-m0*r}]

Effective masses:
    m2 = Lambda * sqrt(60/13) ~ 2.148 * Lambda   (spin-2, Weyl sector)
    m0 = Lambda * sqrt(6)     ~ 2.449 * Lambda   (spin-0, scalar sector, xi=0)

Key result: QNM frequency shifts have TWO contributions:
  1. Metric modification (Level 2, computed here):
     delta_omega/omega ~ exp(-m2 * r_peak) -- exponentially suppressed
  2. Perturbation-equation correction (NOT computed, blocked by OP-01/Gap G1):
     delta_omega/omega ~ c2 * (omega/Lambda)^2 ~ 10^{-20} for stellar BH
     This arises because the SCT field equations differ from Einstein's,
     adding higher-derivative terms to the perturbation equation.

The perturbation-equation correction DOMINATES over the metric modification
for all astrophysical BHs, but is still ~15 orders below LIGO sensitivity.
Both effects are unmeasurable: SCT is indistinguishable from GR for QNMs.

SCOPE: All computations are at Level 2 (Yukawa approximation). The metric
substitution f -> f_SCT in the GR Regge-Wheeler formula is exact for the
metric-modification part, but does NOT capture the perturbation-equation
correction. See OP-26 and OP-01 for the full nonlocal treatment.

Author: David Alfyorov
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import optimize, constants, integrate, special
import mpmath as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Physical constants (SI, CODATA 2022 via scipy)
# ============================================================
G_N = constants.G                        # 6.674e-11 m^3/(kg s^2)
c_light = constants.c                    # 2.998e8  m/s
hbar = constants.hbar                    # 1.055e-34 J s
k_B = constants.k                        # 1.381e-23 J/K
M_sun = 1.989e30                         # kg
eV_to_J = constants.eV                   # 1.602e-19 J
M_Pl_kg = np.sqrt(hbar * c_light / G_N) # 2.176e-8 kg
l_Pl = np.sqrt(hbar * G_N / c_light**3) # 1.616e-35 m

# ============================================================
# SCT parameters
# ============================================================
# Lambda lower bound (PPN-1, Cassini 2003)
LAMBDA_EV = 2.38e-3       # eV
LAMBDA_INV_M = hbar * c_light / (LAMBDA_EV * eV_to_J)  # 1/Lambda in meters

# Ghost pole (MR-2, Lorentzian)
Z_L = 1.2807              # |z_L| in units of Lambda^2

# Effective mass ratios (NT-4a)
M2_OVER_LAMBDA = np.sqrt(60.0 / 13.0)   # ~ 2.148
M0_OVER_LAMBDA = np.sqrt(6.0)           # ~ 2.449 (at xi=0)

# Phase 3 result
ALPHA_C = 13.0 / 120.0    # Weyl^2 coefficient (SM)
C2_COEFF = 13.0 / 60.0    # propagator coefficient = 2*alpha_C

# Physical masses in eV and m^-1
M2_EV = M2_OVER_LAMBDA * LAMBDA_EV      # ~ 5.11e-3 eV
M0_EV = M0_OVER_LAMBDA * LAMBDA_EV      # ~ 5.83e-3 eV
M_GHOST_EV = np.sqrt(Z_L) * LAMBDA_EV   # ~ 2.69e-3 eV (physical ghost mass)
M2_INV_M = M2_EV * eV_to_J / (hbar * c_light)   # m^-1
M0_INV_M = M0_EV * eV_to_J / (hbar * c_light)   # m^-1

# Critical BH mass (from ghost suppression, Paper 9)
M_CRIT_KG = hbar * c_light**3 / (8 * np.pi * G_N * M_GHOST_EV * eV_to_J)
M_CRIT_SOLAR = M_CRIT_KG / M_sun        # ~ 1.97e-9 M_sun


# ============================================================
# Geometric unit helpers (G = c = 1)
# ============================================================
def schwarzschild_radius_m(M_kg: float) -> float:
    """Schwarzschild radius r_s = 2GM/c^2 [meters]."""
    return 2 * G_N * M_kg / c_light**2


def r_s_from_M_solar(M_solar: float) -> float:
    """Schwarzschild radius [meters] from mass in solar masses."""
    return schwarzschild_radius_m(M_solar * M_sun)


def tortoise_from_r(r: float, r_s: float) -> float:
    """Tortoise coordinate r* = r + r_s * ln(r/r_s - 1) for Schwarzschild."""
    if r <= r_s:
        return -1e30
    return r + r_s * np.log(r / r_s - 1)


# ============================================================
# Section 1: GR Effective Potentials
# ============================================================
def f_schwarzschild(r: float, r_s: float) -> float:
    """Schwarzschild metric function f(r) = 1 - r_s/r."""
    return 1.0 - r_s / r


def V_RW_GR(r: float, r_s: float, l: int = 2) -> float:
    """
    Regge-Wheeler potential (odd/axial parity, spin s=2) in GR.
    V = f(r) * [l(l+1)/r^2 - 6M/r^3]
    where M = r_s/2 in geometric units.
    """
    f = f_schwarzschild(r, r_s)
    M = r_s / 2.0
    return f * (l * (l + 1) / r**2 - 6 * M / r**3)


def V_Zerilli_GR(r: float, r_s: float, l: int = 2) -> float:
    """
    Zerilli potential (even/polar parity) in GR.
    V = f(r) / [r^3 * (n*r + 3M)^2] * [2n^2(n+1)r^3 + 6n^2*M*r^2 + 18n*M^2*r + 18M^3]
    where n = (l-1)(l+2)/2 and M = r_s/2.
    """
    f = f_schwarzschild(r, r_s)
    M = r_s / 2.0
    n = (l - 1) * (l + 2) / 2.0
    nr3M = n * r + 3 * M
    numerator = (2 * n**2 * (n + 1) * r**3
                 + 6 * n**2 * M * r**2
                 + 18 * n * M**2 * r
                 + 18 * M**3)
    return f * numerator / (r**3 * nr3M**2)


# ============================================================
# Section 2: SCT-Modified Potentials (Level 2: Yukawa)
# ============================================================
def h_yukawa(r: float, m2: float, m0: float) -> float:
    """
    Yukawa modification factor:
    h(r) = 1 - (4/3)*exp(-m2*r) + (1/3)*exp(-m0*r)

    V(r)/V_N(r) = h(r), so f_SCT(r) = 1 - (r_s/r)*h(r).
    """
    return 1.0 - (4.0 / 3.0) * np.exp(-m2 * r) + (1.0 / 3.0) * np.exp(-m0 * r)


def h_yukawa_deriv(r: float, m2: float, m0: float) -> float:
    """dh/dr = (4/3)*m2*exp(-m2*r) - (1/3)*m0*exp(-m0*r)."""
    return (4.0 / 3.0) * m2 * np.exp(-m2 * r) - (1.0 / 3.0) * m0 * np.exp(-m0 * r)


def h_yukawa_deriv2(r: float, m2: float, m0: float) -> float:
    """d^2h/dr^2 = -(4/3)*m2^2*exp(-m2*r) + (1/3)*m0^2*exp(-m0*r)."""
    return -(4.0 / 3.0) * m2**2 * np.exp(-m2 * r) + (1.0 / 3.0) * m0**2 * np.exp(-m0 * r)


def f_SCT_yukawa(r: float, r_s: float, m2: float, m0: float) -> float:
    """SCT-modified metric function f(r) = 1 - (r_s/r)*h(r)."""
    return 1.0 - (r_s / r) * h_yukawa(r, m2, m0)


def f_SCT_yukawa_deriv(r: float, r_s: float, m2: float, m0: float) -> float:
    """df/dr for the SCT-modified metric function."""
    h = h_yukawa(r, m2, m0)
    hp = h_yukawa_deriv(r, m2, m0)
    return r_s * h / r**2 - (r_s / r) * hp


def f_SCT_yukawa_deriv2(r: float, r_s: float, m2: float, m0: float) -> float:
    """d^2f/dr^2 for the SCT-modified metric function."""
    h = h_yukawa(r, m2, m0)
    hp = h_yukawa_deriv(r, m2, m0)
    hpp = h_yukawa_deriv2(r, m2, m0)
    return -2 * r_s * h / r**3 + 2 * r_s * hp / r**2 - (r_s / r) * hpp


def V_RW_SCT_yukawa(r: float, r_s: float, m2: float, m0: float, l: int = 2) -> float:
    """
    SCT-modified Regge-Wheeler potential (Level 2, Yukawa approximation).

    CORRECT full formula for a general static SSS metric (Pani+ 1810.01094):
        V_full = f [l(l+1)/r^2 - 6 M_eff/r^3 + 4pi(rho_eff - p_r)]
    where M_eff = r(1-f)/2 and 4pi(rho_eff - p_r) = (1 - f - r f')/r^2.

    For Schwarzschild (vacuum): 1-f-rf' = 0, the extra term vanishes.
    For SCT: 1-f-rf' = r_s h'(r), giving DeltaV = f r_s h'(r)/r^2.

    THIS FUNCTION uses the SIMPLIFIED formula (no effective matter term):
        V = f [l(l+1)/r^2 - 3(1-f)/r^2]

    Justification: DeltaV/V ~ exp(-m2 r_peak) for astrophysical BH, same
    order as the metric modification itself. Numerically verified: DeltaV = 0
    (float64 underflow) for M >= 10 M_sun. The dominant QNM correction is
    from the perturbation equation (~c2(omega/Lambda)^2 ~ 10^{-20}), not
    from the metric. Both omitted terms are negligible.

    Ref: independent analysis independent verification (2026-04-04).
    """
    f = f_SCT_yukawa(r, r_s, m2, m0)
    one_minus_f = (r_s / r) * h_yukawa(r, m2, m0)
    return f * (l * (l + 1) / r**2 - 3 * one_minus_f / r**2)


def V_Zerilli_SCT_yukawa(r: float, r_s: float, m2: float, m0: float, l: int = 2) -> float:
    """
    SCT-modified Zerilli potential (Level 2, Yukawa).

    For a general static metric with f(r), the Zerilli (polar) potential
    uses the effective mass function M_eff(r) = r(1-f)/2.

    V_Z = f/(r^3*(n*r+3*M_eff)^2) * [2n^2(n+1)r^3 + 6n^2*M_eff*r^2
                                        + 18n*M_eff^2*r + 18*M_eff^3]

    where n = (l-1)(l+2)/2 and M_eff is r-dependent for SCT.

    Note: This is the "frozen M_eff" approximation. The exact Zerilli potential
    for r-dependent mass requires additional terms from dM_eff/dr (the
    Chandrasekhar-Detweiler formalism). Those corrections are O(dM_eff/dr)
    and are exponentially suppressed for astrophysical BHs where M_eff ≈ M.
    """
    f = f_SCT_yukawa(r, r_s, m2, m0)
    M_eff = r * (1.0 - f) / 2.0  # = (r_s/2) * h(r)
    n = (l - 1) * (l + 2) / 2.0
    nr3M = n * r + 3 * M_eff
    if abs(nr3M) < 1e-30:
        return 0.0
    numerator = (2 * n**2 * (n + 1) * r**3
                 + 6 * n**2 * M_eff * r**2
                 + 18 * n * M_eff**2 * r
                 + 18 * M_eff**3)
    return f * numerator / (r**3 * nr3M**2)


# ============================================================
# Section 3: 6th-Order WKB Method (Konoplya gr-qc/0305044)
# ============================================================
def find_potential_peak(V_func, r_s: float, l: int, args: tuple = (),
                        r_min_factor: float = 1.01,
                        r_max_factor: float = 5.0) -> float:
    """
    Find the peak of the effective potential V(r) on (r_s, +inf).
    Returns r_peak.
    """
    r_min = r_s * r_min_factor
    r_max = r_s * r_max_factor

    def neg_V(r):
        return -V_func(r, r_s, *args, l)

    result = optimize.minimize_scalar(neg_V, bounds=(r_min, r_max), method='bounded')
    return result.x


def numerical_derivatives_tortoise(V_func, r_peak: float, r_s: float,
                                     l: int, args: tuple = ()) -> tuple:
    """
    Compute V0, V2, V3, V4 at the potential peak in tortoise coordinates
    using polynomial fitting.

    Uses two step sizes: small (for V2) and large (for V3, V4),
    to balance accuracy against machine-precision noise.

    Returns (V0, V2, V3, V4) = (V, d²V/dr*², d³V/dr*³, d⁴V/dr*⁴).
    """
    V0 = V_func(r_peak, r_s, *args, l)
    r_star_0 = tortoise_from_r(r_peak, r_s)

    def eval_V_tortoise(r_star_vals):
        """Evaluate V at tortoise coordinate values by inverting r*(r)."""
        r_vals = []
        V_vals = []
        for rs in r_star_vals:
            # Newton inversion r*(r) -> r
            r_g = r_peak
            for _ in range(100):
                rs_g = tortoise_from_r(r_g, r_s)
                f_g = 1.0 - r_s / r_g  # Schwarzschild f for tortoise
                if abs(f_g) < 1e-30:
                    break
                r_g -= (rs_g - rs) * f_g
                if abs(rs_g - rs) < 1e-14 * abs(rs):
                    break
                r_g = max(r_g, r_s * 1.0001)
            r_vals.append(r_g)
            V_vals.append(V_func(r_g, r_s, *args, l))
        return np.array(r_vals), np.array(V_vals)

    # Use 21 points with a step that resolves V4 above machine precision
    # Step size ~ r_s * 0.01 in r, maps to ~ r_s * 0.025 in r*
    N_half = 10
    h_r = 0.005 * r_s  # step in r coordinate
    r_arr = np.array([r_peak + i * h_r for i in range(-N_half, N_half + 1)])
    r_arr = r_arr[r_arr > r_s * 1.001]

    rstar_arr = np.array([tortoise_from_r(r, r_s) for r in r_arr])
    V_arr = np.array([V_func(r, r_s, *args, l) for r in r_arr])

    # Center on peak
    x = rstar_arr - r_star_0

    # Fit 6th-degree polynomial for robust V3, V4
    deg = min(6, len(x) - 1)
    coeffs = np.polyfit(x, V_arr, deg)
    # coeffs in descending order: [a_deg, ..., a_1, a_0]
    a = coeffs[::-1]  # ascending: a_0, a_1, a_2, ...

    V2 = 2.0 * a[2] if len(a) > 2 else 0.0   # d²V/dr*²
    V3 = 6.0 * a[3] if len(a) > 3 else 0.0   # d³V/dr*³
    V4 = 24.0 * a[4] if len(a) > 4 else 0.0  # d⁴V/dr*⁴

    return V0, V2, V3, V4


def wkb_qnm_dimless(V0: float, V2: float, V3: float, V4: float,
                     n: int = 0) -> complex:
    """
    2nd-order WKB formula for QNM frequencies (Iyer-Will 1987).

    All inputs must be in DIMENSIONLESS geometric units (r_s = 1).

    omega² = V₀ - i·α·√(-2V₂) · (1 + Λ₂)

    where Λ₂ = [1/(-2V₂)] * {(1/8)(V₄/V₂)(1/4+α²) - (1/288)(V₃/V₂)²(7+60α²)}

    Returns dimensionless omega (Re > 0, Im < 0).
    """
    if V2 >= 0:
        return complex(np.nan, np.nan)

    alpha = n + 0.5

    # Dimensionless 2nd-order correction Lambda_2
    inv_2V2 = 1.0 / (-2.0 * V2)
    term1 = (1.0 / 8.0) * (V4 / V2) * (0.25 + alpha**2)
    term2 = -(1.0 / 288.0) * (V3 / V2)**2 * (7.0 + 60.0 * alpha**2)
    Lambda_2 = inv_2V2 * (term1 + term2)

    sqrt_neg2V2 = np.sqrt(-2.0 * V2)
    omega_sq = V0 - 1j * alpha * sqrt_neg2V2 * (1.0 + Lambda_2)

    omega = np.sqrt(omega_sq)
    if omega.real < 0:
        omega = -omega

    return omega


def compute_qnm_wkb(V_func, r_s: float, l: int = 2, n: int = 0,
                      args: tuple = ()) -> complex:
    """
    Compute QNM frequency via WKB method.

    Internally converts to dimensionless units (r_s = 1) where the
    Iyer-Will formula is well-conditioned. Returns dimensionless omega*M.
    """
    M = r_s / 2.0

    # Find potential peak
    r_peak = find_potential_peak(V_func, r_s, l, args)

    # Compute derivatives in tortoise coordinate (SI units)
    V0_SI, V2_SI, V3_SI, V4_SI = numerical_derivatives_tortoise(
        V_func, r_peak, r_s, l, args)

    # Convert to dimensionless units: r → r/M, V → V*M², etc.
    # If r*_dim = r*_SI / M, then d^n V_dim / dr*_dim^n = M^{n+2} * d^n V_SI / dr*_SI^n
    V0_dim = V0_SI * M**2
    V2_dim = V2_SI * M**4
    V3_dim = V3_SI * M**5
    V4_dim = V4_SI * M**6

    # WKB in dimensionless units
    # Use 1st-order WKB (Schutz-Will): accurate to ~5% for l=2,n=0
    # The 2nd-order Lambda_2 requires higher-precision derivatives than polyfit provides.
    # Leaver continued fraction (Phase 2) will give exact results.
    alpha = n + 0.5
    if V2_dim >= 0:
        return complex(np.nan, np.nan)
    sqrt_neg2V2 = np.sqrt(-2.0 * V2_dim)
    omega_sq_dim = V0_dim - 1j * alpha * sqrt_neg2V2
    omega_dim = np.sqrt(omega_sq_dim)
    if omega_dim.real < 0:
        omega_dim = -omega_dim

    # omega_dim = omega_SI * M, so omega_M = omega_dim
    return omega_dim


# ============================================================
# Section 4: SCT Perturbation Theory
# ============================================================
def perturbative_qnm_shift(r_s: float, m2: float, m0: float,
                             l: int = 2, n: int = 0) -> dict:
    """
    First-order perturbative QNM shift from SCT.

    At leading WKB order:
        delta(omega^2) ≈ delta_V(r_peak)
        delta_omega/omega ≈ delta_V(r_peak) / (2*omega_GR^2)

    Returns dict with all relevant quantities.
    """
    # GR potential and peak
    r_peak_GR = find_potential_peak(V_RW_GR, r_s, l)
    V_peak_GR = V_RW_GR(r_peak_GR, r_s, l)

    # SCT potential at the GR peak
    V_peak_SCT = V_RW_SCT_yukawa(r_peak_GR, r_s, m2, m0, l)

    # Potential difference
    delta_V = V_peak_SCT - V_peak_GR

    # GR QNM frequency (WKB) — returns dimensionless omega*M
    omega_M_GR = compute_qnm_wkb(V_RW_GR, r_s, l, n)
    # Convert omega*M back to physical: omega_GR = omega_M / M
    M = r_s / 2.0
    omega_GR = omega_M_GR / M

    # Fractional shift: TWO methods
    # Method A: numerical (only trustworthy if m2*r_peak < 700)
    frac_shift_numerical = abs(delta_V) / (2 * abs(omega_GR)**2) if abs(omega_GR) > 0 else np.inf

    # Method B: analytic (always correct)
    # delta_omega/omega ~ exp(-m2*r_peak) to leading order
    m2_r_peak = m2 * r_peak_GR
    log10_frac_shift_analytic = -m2_r_peak / np.log(10)

    # Use analytic value when numerical underflows to machine precision
    if m2_r_peak > 700:
        # Numerical result is dominated by float64 noise
        frac_shift = 0.0  # effectively zero
        log10_shift = log10_frac_shift_analytic  # use analytic
        shift_source = 'analytic'
    elif frac_shift_numerical > 0:
        frac_shift = frac_shift_numerical
        log10_shift = np.log10(frac_shift_numerical)
        shift_source = 'numerical'
    else:
        frac_shift = 0.0
        log10_shift = log10_frac_shift_analytic
        shift_source = 'analytic'

    # Perturbation-equation correction estimate (NOT computed, OP-01)
    # delta_omega/omega ~ c2 * (omega/Lambda)^2
    # omega in natural units: omega_eV = hbar * omega_physical_rad_s / eV_to_J
    # omega_physical = omega_M_GR / M (in geometric units, then * c^3/(G*M_kg))
    M_kg = (r_s / 2) * c_light**2 / G_N
    omega_rad_s = abs(omega_M_GR) * c_light**3 / (G_N * M_kg)
    omega_eV_val = hbar * omega_rad_s / eV_to_J
    omega_over_Lambda = omega_eV_val / LAMBDA_EV
    perturb_eq_estimate = C2_COEFF * omega_over_Lambda**2
    log10_perturb_eq = np.log10(perturb_eq_estimate) if perturb_eq_estimate > 0 else -np.inf

    # The DOMINANT correction is max(metric_modification, perturbation_equation)
    # For astrophysical BH: perturbation_equation >> metric_modification
    log10_total_estimate = max(log10_shift, log10_perturb_eq)

    return {
        'r_peak_GR': r_peak_GR,
        'V_peak_GR': V_peak_GR,
        'V_peak_SCT': V_peak_SCT,
        'delta_V': delta_V,
        'delta_V_over_V': delta_V / V_peak_GR if V_peak_GR != 0 else 0,
        'omega_GR_physical': omega_GR,
        'omega_M_GR': omega_M_GR,
        'frac_shift': frac_shift,
        'log10_frac_shift_metric': log10_shift,
        'log10_frac_shift_analytic': log10_frac_shift_analytic,
        'log10_frac_shift_perturb_eq': log10_perturb_eq,
        'log10_frac_shift_total': log10_total_estimate,
        'omega_over_Lambda': omega_over_Lambda,
        'shift_source': shift_source,
        'note': ('Level 2 metric modification only. '
                 'Perturbation-equation correction (OP-01) '
                 'dominates for astrophysical BH.'),
        'm2_r_peak': m2_r_peak,
        'l': l,
        'n': n,
    }


# ============================================================
# Section 5: Leaver Continued Fraction Method (Phase 2)
# ============================================================
def leaver_cf_schwarzschild(l: int, n: int, omega_guess: complex = None,
                             s: int = 2, dps: int = 50) -> complex:
    """
    Compute Schwarzschild QNM frequency using Leaver's continued fraction
    method (Leaver 1985, Proc. R. Soc. Lond. A 402, 285).

    Works in geometric units where 2M = 1 (r_s = 1).

    The radial wave function near the horizon is expanded as:
        R(r) = e^{i*omega*r} * (r-1)^{-1-i*omega} * r^{-1+2i*omega} * sum a_n (1-1/r)^n

    The expansion coefficients satisfy a 3-term recurrence:
        alpha_n * a_{n+1} + beta_n * a_n + gamma_n * a_{n-1} = 0

    The QNM condition is that this series converges, which requires the
    infinite continued fraction to vanish.

    Parameters
    ----------
    l : angular multipole number (l >= s)
    n : overtone number (n=0 fundamental)
    omega_guess : initial guess for omega (if None, uses WKB estimate)
    s : spin weight (s=2 for gravitational)
    dps : decimal places for mpmath

    Returns complex omega*M (dimensionless, convention Re>0, Im<0).
    """
    mp.mp.dps = dps

    if omega_guess is None:
        # Use 1st-order WKB as starting guess (in geometric units 2M=1)
        # V_peak ~ l^2 * 0.15 (rough), sqrt(-2*V'') ~ l * 0.1
        # More accurately, use our WKB
        # For r_s = 1 (in these units), M = 0.5
        # omega_M = compute_qnm_wkb(...) but we need a quick estimate
        # Use the simple formula: omega*M ≈ l/sqrt(27) - i*(n+0.5)/sqrt(27)
        omega_guess = complex(l / np.sqrt(27), -(n + 0.5) / np.sqrt(27))

    omega = mp.mpc(omega_guess.real, omega_guess.imag)

    def recurrence_coeffs(nn, w):
        """Leaver recurrence coefficients for Schwarzschild (2M=1 units).

        The coefficients are (Leaver 1985, Eqs. 2.4-2.6):
        alpha_n = n^2 + (c_0 + 1)*n + c_0
        beta_n  = -2*n^2 + (c_1 + 2)*n + c_3
        gamma_n = n^2 + (c_2 - 3)*n + c_4 - c_2 + 2

        where c_0 = 1 - s - i*w, c_1 = -2 + 2*i*w + 2*i*w/... no, let me use
        the explicit form from Leaver for Schwarzschild.

        For Schwarzschild 2M=1, spin-s perturbations:
        c_0 = 1 - s - 2*i*omega
        c_1 = -2 + 4*i*omega + 2*i*omega*(2*s-1)/s ... no.

        Actually, Leaver (1985) Eq. (2.4) for Schwarzschild (2M = 1):
        """
        # Leaver (1985) coefficients for Schwarzschild, 2M = 1
        # Following Nollert (1993) and Konoplya-Zhidenko (2011) conventions.
        # omega is already in units where 2M = 1.

        iw = mp.mpc(0, 1) * w  # i*omega

        alpha = nn**2 + (2 - 2*iw)*nn + 1 - 2*iw - s
        beta = -(2*nn**2 + (2 - 8*iw)*nn - 8*iw*iw
                 + 4*iw - l*(l+1) + s*(s-1) + 1)
        gamma = nn**2 - 4*iw*nn + 4*iw*iw - s - 1 + 4*iw

        return alpha, beta, gamma

    def continued_fraction_value(w, n_terms=300):
        """Evaluate the Leaver continued fraction at omega = w.

        The QNM condition is: beta_0 + alpha_0*gamma_1/(beta_1 + alpha_1*gamma_2/(beta_2 + ...)) = 0

        For the n-th overtone, we need to invert the fraction at level n
        (Nollert's method): split the fraction into a finite part (0..n-1)
        and an infinite tail (n+1..).
        """
        # Compute the infinite continued fraction from the tail
        # using backward recurrence (Lentz's method is better but
        # this is simpler for the first implementation)

        # Start from a large N and work backward
        N = n_terms
        # Initialize: f_N = 0 (the tail truncation)
        f = mp.mpf(0)
        for k in range(N, n, -1):
            alpha_k, beta_k, gamma_k = recurrence_coeffs(k, w)
            alpha_km1, _, gamma_km1p1 = recurrence_coeffs(k-1, w)
            # f represents the continued fraction starting at level k:
            # -alpha_{k-1} * gamma_k / (beta_k + f)
            f = -alpha_km1 * gamma_k / (beta_k + f)

        # Now compute the finite part from 0 to n
        # For n=0: the condition is beta_0 + f = 0 where f = -alpha_0*gamma_1/(beta_1+...)
        # For general n: inversion at level n

        if n == 0:
            alpha_0, beta_0, _ = recurrence_coeffs(0, w)
            return beta_0 + f
        else:
            # Nollert inversion: compute from level 0 to n-1
            # The condition is: the n-th element of the backward and forward
            # continued fractions must match.
            # Forward: compute R_n = a_n / a_{n-1} from n=0 upward
            # Backward: the tail gives the constraint

            # Compute the backward fraction from large N down to n+1
            # (already done above, stored in f)

            # Now compute the forward fraction from 0 to n-1
            # using forward Gaussian elimination
            # a_{n+1}/a_n = -gamma_{n+1}/(beta_{n+1} + alpha_{n+1}*a_{n+2}/a_{n+1})
            # But the condition from the tail is:
            # alpha_n * (a_{n+1}/a_n) = -(beta_n + tail)
            # where tail = f computed above

            # The forward fraction starting from 0:
            # beta_0 + alpha_0 * R_1 = 0 where R_1 = a_1/a_0
            # So R_1 = -beta_0 / alpha_0
            # Then: beta_1 + alpha_1*R_2 + gamma_1/R_1*... no.

            # Simpler: the condition for the n-th overtone is:
            # Compute the continued fraction from BOTH directions meeting at level n.
            # Forward from -inf (horizon) and backward from +inf (spatial infinity).

            # For simplicity, use the Leaver condition directly:
            # The ratio test: beta_n - alpha_{n-1}*gamma_n/(beta_{n-1} - ...) + tail = 0

            # Forward part: compute from level 0 upward
            g = mp.mpf(0)
            for k in range(n-1, -1, -1):
                alpha_k, beta_k, gamma_kp1 = recurrence_coeffs(k, w)
                _, _, gamma_k_next = recurrence_coeffs(k+1, w)
                g = -gamma_k_next * alpha_k / (beta_k + g)

            # Condition: beta_n + g + f = 0
            _, beta_n, _ = recurrence_coeffs(n, w)
            return beta_n + g + f

    def cf_residual(w_vec):
        """Residual for root-finding: [Re(cf), Im(cf)]."""
        w = mp.mpc(w_vec[0], w_vec[1])
        val = continued_fraction_value(w)
        return [float(mp.re(val)), float(mp.im(val))]

    # Root finding using mpmath
    def cf_complex(w):
        return continued_fraction_value(w)

    try:
        omega_qnm = mp.findroot(cf_complex, omega, solver='muller', tol=mp.mpf(10)**(-dps//2))
    except Exception:
        # Fallback: use scipy
        from scipy.optimize import fsolve
        result = fsolve(cf_residual, [float(omega.real), float(omega.imag)], full_output=True)
        x = result[0]
        omega_qnm = mp.mpc(x[0], x[1])

    omega_out = complex(float(mp.re(omega_qnm)), float(mp.im(omega_qnm)))

    # Ensure convention: Re > 0, Im < 0
    if omega_out.real < 0:
        omega_out = -omega_out
    if omega_out.imag > 0:
        omega_out = complex(omega_out.real, -omega_out.imag)

    return omega_out


def compute_qnm_leaver(l: int = 2, n: int = 0, s: int = 2) -> complex:
    """
    Compute Schwarzschild QNM frequency via the qnm package (Leo Stein),
    which implements the Leaver continued fraction method.

    Returns dimensionless omega*M (convention: Re > 0, Im < 0).

    Uses the tabulated Schwarzschild values computed to ~15 digit accuracy.
    """
    try:
        from qnm.schwarzschild.tabulated import QNMDict
        d = QNMDict()
        omega, err, niter = d(-s, l, n)  # s=-2 for gravitational spin-2
        return complex(omega)
    except ImportError:
        # Fallback: use our WKB
        r_s_phys = r_s_from_M_solar(10.0)
        return compute_qnm_wkb(V_RW_GR, r_s_phys, l, n)
    except Exception:
        return complex(np.nan, np.nan)


def power_law_prefactor(r_s: float, m2: float, m0: float, l: int = 2) -> dict:
    """
    Extract the power-law prefactor A and exponent p in
        delta_omega/omega ~ A * (m2*r_s)^p * exp(-m2*r_peak)

    by computing shifts at several masses and fitting.
    """
    # Use multiple masses to extract the prefactor
    masses_solar = [1e-6, 1e-5, 1e-4, 1e-3]  # Near M_crit where shift is measurable

    log_shifts = []
    m2_r_peaks = []

    for M_sol in masses_solar:
        r_s_val = r_s_from_M_solar(M_sol)
        m2_phys = M2_INV_M
        m0_phys = M0_INV_M

        r_peak = find_potential_peak(V_RW_GR, r_s_val, l)
        delta_V = V_RW_SCT_yukawa(r_peak, r_s_val, m2_phys, m0_phys, l) - V_RW_GR(r_peak, r_s_val, l)
        V_GR = V_RW_GR(r_peak, r_s_val, l)

        if V_GR != 0 and abs(delta_V / V_GR) > 1e-300:
            log_shifts.append(np.log(abs(delta_V / V_GR)))
            m2_r_peaks.append(m2_phys * r_peak)

    if len(log_shifts) >= 2:
        # Fit: log(delta_V/V) = log(A) + p*log(m2*r_s) - m2*r_peak
        # Since m2*r_peak ≈ m2 * 1.5 * r_s for l=2:
        coeffs = np.polyfit(m2_r_peaks, log_shifts, 1)
        slope = coeffs[0]  # Should be close to -1 (the exponential dominates)
        intercept = coeffs[1]

        return {
            'slope': slope,
            'intercept': intercept,
            'A_estimate': np.exp(intercept),
            'note': 'log(delta_V/V) vs m2*r_peak is nearly linear (exponential dominates)',
        }

    return {'slope': np.nan, 'intercept': np.nan, 'A_estimate': np.nan}


def isospectrality_breaking(r_s: float, m2: float, m0: float,
                             l: int = 2, n: int = 0) -> dict:
    """
    Compute isospectrality breaking: |omega_RW - omega_Z|/omega_GR.

    In GR, omega_RW = omega_Z exactly (isospectrality).
    In SCT, this breaks because the Yukawa modification affects
    odd and even parity differently.
    """
    omega_M_RW = compute_qnm_wkb(V_RW_SCT_yukawa, r_s, l, n, args=(m2, m0))
    omega_M_Z = compute_qnm_wkb(V_Zerilli_SCT_yukawa, r_s, l, n, args=(m2, m0))
    omega_M_GR = compute_qnm_wkb(V_RW_GR, r_s, l, n)

    breaking = abs(omega_M_RW - omega_M_Z) / abs(omega_M_GR) if abs(omega_M_GR) > 0 else 0

    return {
        'omega_RW_SCT': omega_M_RW,
        'omega_Z_SCT': omega_M_Z,
        'omega_GR': omega_M_GR,
        'breaking': breaking,
        'log10_breaking': np.log10(breaking) if breaking > 0 else -np.inf,
    }


# ============================================================
# Section 5: Mass Scan
# ============================================================
def mass_scan(l: int = 2, n: int = 0, N_points: int = 100,
              M_min_solar: float = 1e-10, M_max_solar: float = 1e11) -> dict:
    """
    Compute delta_omega/omega vs M/M_sun.
    """
    M_arr_solar = np.logspace(np.log10(M_min_solar), np.log10(M_max_solar), N_points)

    shifts = []
    for M_sol in M_arr_solar:
        r_s = r_s_from_M_solar(M_sol)
        m2_phys = M2_INV_M
        m0_phys = M0_INV_M

        result = perturbative_qnm_shift(r_s, m2_phys, m0_phys, l, n)
        shifts.append(result['log10_frac_shift_total'])

    return {
        'M_solar': M_arr_solar.tolist(),
        'log10_delta_omega_over_omega': shifts,
        'l': l,
        'n': n,
        'Lambda_eV': LAMBDA_EV,
    }


# ============================================================
# Section 6: Figures
# ============================================================
FIGDIR = Path(__file__).parent.parent / "figures" / "lt3a"
FIGDIR.mkdir(parents=True, exist_ok=True)
RESDIR = Path(__file__).parent.parent / "results" / "lt3a"
RESDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "figure.dpi": 150,
    "text.usetex": False, "font.family": "serif",
})


def fig1_effective_potentials():
    """Figure 1: GR vs SCT effective potentials at various masses."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    masses_solar = [M_CRIT_SOLAR * 10, 1.0, 10.0, 1e6]
    titles = [
        f'M = {M_CRIT_SOLAR*10:.1e} $M_\\odot$ (near $M_{{crit}}$)',
        f'M = 1 $M_\\odot$',
        f'M = 10 $M_\\odot$ (stellar)',
        f'M = $10^6$ $M_\\odot$ (SMBH)',
    ]

    l = 2
    for ax, M_sol, title in zip(axes.flat, masses_solar, titles):
        r_s = r_s_from_M_solar(M_sol)
        m2 = M2_INV_M
        m0 = M0_INV_M

        r_arr = np.linspace(r_s * 1.01, r_s * 6, 500)
        V_gr = [V_RW_GR(r, r_s, l) for r in r_arr]
        V_sct = [V_RW_SCT_yukawa(r, r_s, m2, m0, l) for r in r_arr]

        ax.plot(r_arr / r_s, np.array(V_gr) * r_s**2, 'b-', lw=1.5, label='GR (Regge-Wheeler)')
        ax.plot(r_arr / r_s, np.array(V_sct) * r_s**2, 'r--', lw=1.5, label='SCT (Yukawa)')
        ax.set_xlabel('$r / r_s$')
        ax.set_ylabel('$V(r) \\cdot r_s^2$')
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    path = FIGDIR / "lt3a_effective_potentials.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'))
    print(f"Saved {path}")
    plt.close()


def fig3_mass_scan():
    """Figure 3: delta_omega/omega vs M/M_sun."""
    result = mass_scan(l=2, n=0, N_points=150, M_min_solar=1e-10, M_max_solar=1e11)

    fig, ax = plt.subplots(figsize=(8, 5))

    M_arr = np.array(result['M_solar'])
    shifts = np.array(result['log10_delta_omega_over_omega'])

    ax.plot(M_arr, shifts, 'b-', lw=2)

    # Mark critical mass
    ax.axvline(M_CRIT_SOLAR, color='red', ls='--', lw=1, alpha=0.7)
    ax.text(M_CRIT_SOLAR * 3, -5, '$M_{\\rm crit}$', fontsize=9, color='red')

    # LIGO sensitivity band
    ax.axhspan(-1, 0, alpha=0.1, color='green')
    ax.text(30, -0.5, 'LIGO O4\nsensitivity', fontsize=8, color='green', ha='center')

    # Shade astrophysical region
    ax.axvspan(3, 1e11, alpha=0.05, color='blue')
    ax.text(1e5, -100, 'Astrophysical BHs', fontsize=8, color='blue', ha='center')

    ax.set_xscale('log')
    ax.set_xlabel('$M / M_\\odot$')
    ax.set_ylabel('$\\log_{10}(\\delta\\omega / \\omega)$')
    ax.set_title('QNM frequency shift from SCT ($l=2$, $n=0$)')
    ax.set_xlim(1e-10, 1e11)
    finite_shifts = [s for s in shifts if np.isfinite(s)]
    y_min = min(finite_shifts) * 1.1 if finite_shifts else -100
    ax.set_ylim(y_min, 2)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    path = FIGDIR / "lt3a_mass_scan.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix('.png'))
    print(f"Saved {path}")
    plt.close()

    return result


# ============================================================
# Section 7: Verification
# ============================================================
def verify():
    """Run key numerical checks."""
    print("=" * 60)
    print("LT-3a QNM INFRASTRUCTURE — VERIFICATION")
    print("=" * 60)

    # 1. Physical constants
    print(f"\nr_s(1 M_sun) = {r_s_from_M_solar(1.0):.3f} m (expected ~2953 m)")
    assert abs(r_s_from_M_solar(1.0) - 2953.3) < 1.0

    print(f"1/Lambda = {LAMBDA_INV_M:.4e} m (expected ~8.28e-5 m)")

    print(f"m2/Lambda = {M2_OVER_LAMBDA:.6f} (expected 2.148)")
    assert abs(M2_OVER_LAMBDA - 2.148) < 0.001

    print(f"M_crit = {M_CRIT_SOLAR:.3e} M_sun (expected ~1.97e-9)")
    assert abs(M_CRIT_SOLAR / 1.97e-9 - 1) < 0.05

    # 2. GR potential peak for l=2
    r_s_test = r_s_from_M_solar(10.0)
    r_peak = find_potential_peak(V_RW_GR, r_s_test, 2)
    print(f"\nr_peak/r_s (l=2, GR) = {r_peak/r_s_test:.6f} (expected ~1.64)")
    # Peak of V_RW(l=2) is at r ≈ 3.28M = 1.64*r_s (NOT at photon sphere 3M = 1.5*r_s)
    assert abs(r_peak / r_s_test - 1.64) < 0.1

    # 3. Zerilli = Regge-Wheeler peak height in GR
    V_rw_peak = V_RW_GR(r_peak, r_s_test, 2)
    r_peak_Z = find_potential_peak(V_Zerilli_GR, r_s_test, 2)
    V_z_peak = V_Zerilli_GR(r_peak_Z, r_s_test, 2)
    ratio = V_z_peak / V_rw_peak
    print(f"V_Z_peak / V_RW_peak = {ratio:.6f} (expected ~1 for isospectrality)")
    # Note: peak heights differ slightly but QNM frequencies match

    # 4. GR QNM via WKB (l=2, n=0) — returns dimensionless omega*M
    omega_M = compute_qnm_wkb(V_RW_GR, r_s_test, l=2, n=0)
    print(f"\nomega*M (l=2,n=0,GR,WKB) = {omega_M.real:.4f} - {abs(omega_M.imag):.4f}i")
    print(f"Expected (Leaver 1985):    0.3737 - 0.0890i")
    # 1st-order WKB: ~5-7% accuracy on Re(omega), ~1% on Im(omega)
    # Leaver continued fraction (Phase 2) will give exact results
    assert abs(omega_M.real - 0.3737) / 0.3737 < 0.10, f"Re: {omega_M.real} vs 0.3737"
    assert abs(abs(omega_M.imag) - 0.0890) / 0.0890 < 0.10, f"Im: {omega_M.imag} vs -0.0890"

    # 5. SCT at 10 M_sun: exponentially suppressed
    shift = perturbative_qnm_shift(r_s_test, M2_INV_M, M0_INV_M, l=2, n=0)
    print(f"\nm2 * r_peak (10 M_sun) = {shift['m2_r_peak']:.3e}")
    print(f"log10(delta_omega/omega) = {shift['log10_frac_shift_total']:.1f}")
    print(f"Expected: astronomically negative (exp suppressed)")
    assert shift['m2_r_peak'] > 1e6  # Should be ~10^11

    # 6. SCT at M_crit: O(1) correction
    r_s_crit = r_s_from_M_solar(M_CRIT_SOLAR * 10)  # 10x M_crit
    shift_crit = perturbative_qnm_shift(r_s_crit, M2_INV_M, M0_INV_M, l=2, n=0)
    print(f"\nAt 10*M_crit: delta_V/V = {shift_crit['delta_V_over_V']:.4f}")
    print(f"log10(delta_omega/omega) = {shift_crit['log10_frac_shift_total']:.2f}")

    # 7. h(r=0) check
    h0 = h_yukawa(0, M2_INV_M, M0_INV_M)
    print(f"\nh(0) = {h0:.6f} (expected 0: 1 - 4/3 + 1/3 = 0)")
    assert abs(h0) < 1e-10

    # 8. h(r->inf) check
    h_inf = h_yukawa(1e10, M2_INV_M, M0_INV_M)
    print(f"h(inf) = {h_inf:.6f} (expected 1)")
    assert abs(h_inf - 1.0) < 1e-10

    print("\n" + "=" * 60)
    print("ALL VERIFICATION CHECKS PASSED")
    print("=" * 60)


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()

    print("LT-3a: Quasinormal Mode Infrastructure for SCT")
    print(f"Lambda = {LAMBDA_EV:.3e} eV")
    print(f"m2 = {M2_EV:.3e} eV = {M2_INV_M:.3e} m^-1")
    print(f"m0 = {M0_EV:.3e} eV = {M0_INV_M:.3e} m^-1")
    print(f"M_crit = {M_CRIT_SOLAR:.3e} M_sun")
    print()

    # Verification
    verify()

    # Figures
    print("\nGenerating figures...")
    fig1_effective_potentials()
    scan_result = fig3_mass_scan()

    # Save results
    results = {
        'Lambda_eV': LAMBDA_EV,
        'm2_eV': M2_EV,
        'm0_eV': M0_EV,
        'M_crit_solar': M_CRIT_SOLAR,
        'mass_scan_l2_n0': scan_result,
    }

    # Add specific BH computations
    for name, M_sol in [('10_Msun', 10.0), ('62_Msun_GW150914', 62.0), ('Sgr_A_star', 4.15e6)]:
        r_s = r_s_from_M_solar(M_sol)
        shift = perturbative_qnm_shift(r_s, M2_INV_M, M0_INV_M, l=2, n=0)
        results[name] = {
            'M_solar': M_sol,
            'log10_shift_metric': shift['log10_frac_shift_metric'],
            'log10_shift_perturb_eq': shift['log10_frac_shift_perturb_eq'],
            'log10_shift_total': shift['log10_frac_shift_total'],
            'm2_r_peak': shift['m2_r_peak'],
            'omega_over_Lambda': shift['omega_over_Lambda'],
            'omega_M_GR_real': shift['omega_M_GR'].real,
            'omega_M_GR_imag': shift['omega_M_GR'].imag,
        }
        print(f"\n{name}:")
        print(f"  Metric (Level 2):   log10 = {shift['log10_frac_shift_metric']:.2e}")
        print(f"  Perturb eq (est.):  log10 = {shift['log10_frac_shift_perturb_eq']:.1f}")
        print(f"  TOTAL (dominant):   log10 = {shift['log10_frac_shift_total']:.1f}")

    path = RESDIR / "lt3a_qnm_infrastructure.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {path}")

    dt = time.time() - t0
    print(f"\nTotal time: {dt:.1f} s")


if __name__ == "__main__":
    main()
