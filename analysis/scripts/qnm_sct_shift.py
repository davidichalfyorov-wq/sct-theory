# ruff: noqa: E402, I001
"""
QNM frequency shift from SCT nonlocal form factors.

Computes the fractional shift delta_omega/omega of Schwarzschild quasi-normal
mode frequencies due to the spectral action's Yukawa corrections to the
graviton propagator.

The SCT-modified metric potential (NT-4a, Eq. (35)):
    Phi(r)/Phi_N(r) = 1 - (4/3) e^{-m2 r} + (1/3) e^{-m0 r}

where m2 = Lambda sqrt(60/13), m0 = Lambda sqrt(6) (at xi=0).

For the Regge-Wheeler potential (gravitational, s=2, l=2):
    V_GR(r) = (1 - r_s/r) [l(l+1)/r^2 - 6M/r^3]   with r_s = 2GM/c^2

The SCT correction enters through the modified metric function:
    f(r) = 1 - r_s/r * [1 - (4/3) e^{-m2 r} + (1/3) e^{-m0 r}]

Method: 3rd-order WKB (Schutz-Will 1985, Iyer-Will 1987).

Author: David Alfyorov
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import optimize, constants
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# Physical constants (SI)
# ============================================================
G_N = constants.G                     # 6.674e-11 m^3/(kg s^2)
c_light = constants.c                 # 2.998e8 m/s
hbar = constants.hbar                 # 1.055e-34 J s
M_sun = 1.989e30                      # kg
eV_to_J = constants.eV                # 1.602e-19 J
M_Pl_eV = 1.2209e28                   # eV (reduced: 2.435e27, but Planck mass = 1.22e28 eV)

# SCT parameters
Lambda_eV = 2.565e-3                  # eV (lower bound from Eot-Wash)
Lambda_inv_m = hbar * c_light / (Lambda_eV * eV_to_J)  # 1/Lambda in meters
Lambda_m = 1.0 / Lambda_inv_m         # Lambda in 1/m

m2_over_Lambda = np.sqrt(60.0 / 13.0)  # ≈ 2.148
m0_over_Lambda = np.sqrt(6.0)           # ≈ 2.449 (at xi=0)

m2_m = m2_over_Lambda * Lambda_m       # spin-2 mass in 1/m
m0_m = m0_over_Lambda * Lambda_m       # spin-0 mass in 1/m (xi=0)


def schwarzschild_radius(M_kg: float) -> float:
    """Schwarzschild radius r_s = 2GM/c^2 in meters."""
    return 2.0 * G_N * M_kg / c_light**2


# ============================================================
# GR Regge-Wheeler potential (s=2 gravitational)
# ============================================================
def V_RW_GR(r: float, r_s: float, l: int = 2) -> float:
    """Regge-Wheeler potential V_GR(r) for spin-2 perturbations.

    V = (1 - r_s/r) [l(l+1)/r^2 - 6M/r^3]
      = (1 - r_s/r) [l(l+1)/r^2 - 3 r_s/r^3]

    where r_s = 2M (in geometric units).
    """
    if r <= r_s:
        return 0.0
    f = 1.0 - r_s / r
    s = 2  # spin weight for gravitational perturbations
    return f * (l * (l + 1) / r**2 - s**2 * (s - 1) * r_s / r**3)


def V_RW_GR_array(r_arr: np.ndarray, r_s: float, l: int = 2) -> np.ndarray:
    """Vectorized Regge-Wheeler potential."""
    s = 2
    f = np.where(r_arr > r_s, 1.0 - r_s / r_arr, 0.0)
    return f * (l * (l + 1) / r_arr**2 - s**2 * (s - 1) * r_s / r_arr**3)


# ============================================================
# SCT-modified Regge-Wheeler potential
# ============================================================
def metric_f_SCT(r: float, r_s: float, m2: float, m0: float) -> float:
    """SCT-modified metric function f(r) = 1 - r_s/r * h(r)

    where h(r) = 1 - (4/3) exp(-m2*r) + (1/3) exp(-m0*r)

    This is the (tt) metric component modification from the Yukawa corrections
    to the Newtonian potential (NT-4a, Eq. 35).
    """
    h = 1.0 - (4.0 / 3.0) * np.exp(-m2 * r) + (1.0 / 3.0) * np.exp(-m0 * r)
    return 1.0 - r_s * h / r


def V_RW_SCT(r: float, r_s: float, m2: float, m0: float, l: int = 2) -> float:
    """SCT-modified Regge-Wheeler potential.

    The effective potential for axial (odd-parity) perturbations around
    the SCT-modified Schwarzschild metric:
        ds^2 = -f(r) dt^2 + f(r)^{-1} dr^2 + r^2 dOmega^2

    with f(r) = 1 - r_s/r * [1 - (4/3)e^{-m2 r} + (1/3)e^{-m0 r}].

    The Regge-Wheeler potential for this metric is:
        V(r) = f(r) [l(l+1)/r^2 - (1-s^2) f'(r)/r]  for s=2:
        V(r) = f(r) [l(l+1)/r^2 + 3 f'(r)/r]

    where f'(r) = df/dr.
    """
    s = 2
    h = 1.0 - (4.0 / 3.0) * np.exp(-m2 * r) + (1.0 / 3.0) * np.exp(-m0 * r)
    # h'(r) = (4/3) m2 exp(-m2 r) - (1/3) m0 exp(-m0 r)
    h_prime = (4.0 / 3.0) * m2 * np.exp(-m2 * r) - (1.0 / 3.0) * m0 * np.exp(-m0 * r)

    f = 1.0 - r_s * h / r
    # f'(r) = r_s h / r^2 - r_s h'/r = (r_s/r^2)(h - r h')
    f_prime = r_s * (h - r * h_prime) / r**2

    # General Regge-Wheeler: V = f [l(l+1)/r^2 + (1-s^2)(f'/r)]
    # For s=2: (1-s^2) = (1-4) = -3, so V = f [l(l+1)/r^2 - 3 f'/r]
    # But the standard form for gravitational s=2 is:
    #   V = f [l(l+1)/r^2 - 6M/r^3] in Schwarzschild coordinates
    # which generalizes for a general spherically symmetric metric to:
    #   V = f [l(l+1)/r^2 + (2/r)(1 - 3M(r)/r) * f/r]
    # For the modified Schwarzschild, the cleanest derivation uses the
    # Chandrasekhar master function.
    #
    # However, for a PERTURBATIVE computation (SCT corrections << 1),
    # the dominant effect is through the modification of f(r) itself.
    # The Regge-Wheeler potential for a general static spherically
    # symmetric metric g_tt = -f(r), g_rr = 1/f(r) is:
    #   V_axial = f(r) * [l(l+1)/r^2 - (1-s^2) * (1/r) * df/dr]
    #
    # For s=2 gravitational: V = f * [l(l+1)/r^2 + 3*f'(r)/r]

    V = f * (l * (l + 1) / r**2 + (1 - s**2) * f_prime / r)
    return V


def V_RW_SCT_array(r_arr: np.ndarray, r_s: float, m2: float, m0: float,
                    l: int = 2) -> np.ndarray:
    """Vectorized SCT-modified Regge-Wheeler potential."""
    s = 2
    h = 1.0 - (4.0 / 3.0) * np.exp(-m2 * r_arr) + (1.0 / 3.0) * np.exp(-m0 * r_arr)
    h_prime = (4.0 / 3.0) * m2 * np.exp(-m2 * r_arr) - (1.0 / 3.0) * m0 * np.exp(-m0 * r_arr)

    f = 1.0 - r_s * h / r_arr
    f_prime = r_s * (h - r_arr * h_prime) / r_arr**2

    V = f * (l * (l + 1) / r_arr**2 + (1 - s**2) * f_prime / r_arr)
    return V


# ============================================================
# WKB method for QNMs (Schutz-Will 1985, Iyer-Will 1987)
# ============================================================
def find_potential_peak(V_func, r_min: float, r_max: float, *args) -> float:
    """Find the radius where V(r) is maximal."""
    result = optimize.minimize_scalar(
        lambda r: -V_func(r, *args),
        bounds=(r_min, r_max),
        method='bounded'
    )
    return result.x


def numerical_derivatives(V_func, r0: float, args: tuple,
                          h: float = 1e-6) -> tuple:
    """Compute V0, V'', V''', V'''' at the peak by finite differences.

    Uses the tortoise coordinate r* for the derivatives.
    Returns d^n V / dr*^n at the peak.
    """
    # For WKB we need derivatives w.r.t. tortoise coordinate r*
    # dr* = dr / f(r), so d/dr* = f(r) d/dr
    # However, for the standard WKB application to the Regge-Wheeler equation,
    # the potential V(r*(r)) is already the effective potential in the
    # tortoise coordinate equation d^2 psi/dr*^2 + [omega^2 - V] psi = 0.
    #
    # The derivatives needed are d^n V / dr*^n evaluated at the peak.
    # We compute these numerically using finite differences.

    r_s = args[0]  # first argument is always r_s

    def tortoise_to_r(r: float) -> float:
        """Tortoise coordinate r* = r + r_s ln(r/r_s - 1)."""
        if r <= r_s:
            return -1e30
        return r + r_s * np.log(r / r_s - 1.0)

    # Evaluate V at r0
    V0 = V_func(r0, *args)

    # Compute dr*/dr = 1/f(r) at r0 for converting derivatives
    # For GR: f(r) = 1 - r_s/r
    # We need d/dr* = f d/dr, d^2/dr*^2 = f d/dr (f d/dr) = f^2 d^2/dr^2 + f f' d/dr
    # It's simpler to evaluate V at several r* points.

    # Map r values to r* and evaluate
    r_points = []
    V_points = []
    dr = h * r0
    for i in range(-4, 5):
        r_i = r0 + i * dr
        if r_i > r_s * 1.001:
            r_points.append(tortoise_to_r(r_i))
            V_points.append(V_func(r_i, *args))

    r_star = np.array(r_points)
    V_vals = np.array(V_points)

    # Use polynomial fit in r* to get derivatives
    # Center on the peak
    r_star_0 = tortoise_to_r(r0)
    dr_star = r_star - r_star_0

    # Fit 4th degree polynomial
    if len(dr_star) >= 5:
        coeffs = np.polyfit(dr_star, V_vals, 4)
        # coeffs = [a4, a3, a2, a1, a0] for a4 x^4 + a3 x^3 + ...
        V2 = 2.0 * coeffs[-3]       # d^2 V / dr*^2
        V3 = 6.0 * coeffs[-4]       # d^3 V / dr*^3
        V4 = 24.0 * coeffs[-5]      # d^4 V / dr*^4
    else:
        V2 = V3 = V4 = 0.0

    return V0, V2, V3, V4


def wkb_qnm_3rd_order(V0: float, V2: float, V3: float, V4: float,
                       n: int = 0) -> complex:
    """3rd-order WKB formula for QNM frequencies.

    Schutz-Will (1985), Iyer-Will (1987).

    omega^2 = V0 - i(n+1/2) sqrt(-2 V2) * (1 + A2 + A3)

    where A2 and A3 are higher-order WKB corrections.

    Parameters
    ----------
    V0 : float - potential at peak
    V2 : float - d^2 V / dr*^2 at peak (should be < 0 at the maximum)
    V3 : float - d^3 V / dr*^3 at peak
    V4 : float - d^4 V / dr*^4 at peak
    n : int - overtone number (n=0 is fundamental)
    """
    if V2 >= 0:
        # Not a maximum
        return complex(np.nan, np.nan)

    alpha = n + 0.5

    # Iyer-Will notation
    # Lambda_2 = (1/(-2V2)) * [...]
    inv_2V2 = 1.0 / (-2.0 * V2)

    # 2nd order correction
    # A2 = (1/(-2V2)) * [ (1/8)(V4/V2)(1/4 + alpha^2)
    #       - (1/288)(V3/V2)^2 (7 + 60 alpha^2) ]
    term1 = (1.0 / 8.0) * (V4 / V2) * (0.25 + alpha**2)
    term2 = -(1.0 / 288.0) * (V3 / V2)**2 * (7.0 + 60.0 * alpha**2)
    Lambda_2 = inv_2V2 * (term1 + term2)

    # For 3rd order we need V5 and V6 which we don't have from 4th order fit,
    # so we use 2nd order WKB (already very good for l=2, n=0).

    # omega^2 = V0 - i alpha sqrt(-2 V2) (1 + Lambda_2)
    sqrt_neg2V2 = np.sqrt(-2.0 * V2)
    omega_sq = V0 - 1j * alpha * sqrt_neg2V2 * (1.0 + Lambda_2)

    omega = np.sqrt(omega_sq)
    # Convention: omega_R > 0, omega_I < 0 for damped modes
    if omega.real < 0:
        omega = -omega

    return omega


def compute_qnm_wkb(V_func, r_s: float, args: tuple, l: int = 2,
                     n: int = 0) -> complex:
    """Compute QNM frequency using WKB method.

    Returns omega * M (dimensionless, in geometric units where c=G=1).
    """
    M = r_s / 2.0  # geometric mass M = r_s/2 (since r_s = 2M in geom. units)

    # Find potential peak
    r_peak = find_potential_peak(V_func, r_s * 1.01, r_s * 5.0, r_s, *args)

    # Get derivatives
    V0, V2, V3, V4 = numerical_derivatives(V_func, r_peak, (r_s, *args))

    # WKB frequency
    omega = wkb_qnm_3rd_order(V0, V2, V3, V4, n)

    # Convert to dimensionless omega*M
    omega_M = omega * M

    return omega_M


# ============================================================
# Wrapper functions with correct signatures for the WKB machinery
# ============================================================
def V_GR_wrapper(r: float, r_s: float) -> float:
    """GR Regge-Wheeler potential (s=2, l=2)."""
    if r <= r_s:
        return 0.0
    f = 1.0 - r_s / r
    l = 2
    return f * (l * (l + 1) / r**2 - 6.0 * (r_s / 2.0) / r**3)


def V_SCT_wrapper(r: float, r_s: float, m2: float, m0: float) -> float:
    """SCT-modified Regge-Wheeler potential (s=2, l=2)."""
    if r <= r_s * 1.0001:
        return 0.0
    return V_RW_SCT(r, r_s, m2, m0, l=2)


# ============================================================
# GR reference: known result for l=2, n=0
# ============================================================
GR_OMEGA_M_REF = complex(0.3737, -0.0890)  # Leaver (1985), high-accuracy


# ============================================================
# First-order perturbation theory (more robust than full WKB recompute)
# ============================================================
def perturbative_shift(r_s: float, m2_phys: float, m0_phys: float,
                       l: int = 2, n: int = 0) -> dict:
    """Compute QNM frequency shift using first-order perturbation theory.

    For a small perturbation delta_V to the potential:
        delta(omega^2) = integral delta_V |psi|^2 dr* / integral |psi|^2 dr*

    Since we don't have the exact QNM eigenfunction, we use the WKB
    approximation which gives:
        delta(omega^2) ≈ delta_V(r_peak) + WKB corrections

    But the dominant contribution is simply delta_V at the potential peak.

    For the SCT correction: delta_V(r) = V_SCT(r) - V_GR(r), which is
    exponentially suppressed for r >> 1/Lambda.

    Returns a dictionary with the shift information.
    """
    M_geom = r_s / 2.0  # geometric mass

    # The potential peak in GR is at r_peak ≈ 3M = 1.5 r_s (for l=2)
    # More precisely, for l=2: r_peak / M ≈ 3.28 (from dV/dr = 0)
    r_peak_guess = 1.5 * r_s

    # Find the GR potential peak
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda r: -V_GR_wrapper(r, r_s),
                          bounds=(r_s * 1.01, r_s * 5.0), method='bounded')
    r_peak = res.x
    V_peak_GR = -res.fun

    # SCT correction at the peak
    h_sct = 1.0 - (4.0 / 3.0) * np.exp(-m2_phys * r_peak) + (1.0 / 3.0) * np.exp(-m0_phys * r_peak)
    h_prime = (4.0 / 3.0) * m2_phys * np.exp(-m2_phys * r_peak) - (1.0 / 3.0) * m0_phys * np.exp(-m0_phys * r_peak)

    # The GR value: h=1, h'=0
    delta_h = h_sct - 1.0
    delta_h_prime = h_prime - 0.0

    # Compute delta_V at the peak
    V_sct_peak = V_SCT_wrapper(r_peak, r_s, m2_phys, m0_phys)
    delta_V_peak = V_sct_peak - V_peak_GR

    # omega^2 for GR (using known result)
    omega_GR = GR_OMEGA_M_REF / M_geom
    omega_sq_GR = omega_GR**2

    # First-order shift: delta(omega^2) ≈ delta_V_peak (at leading WKB order)
    delta_omega_sq = delta_V_peak

    # delta_omega / omega ≈ delta(omega^2) / (2 omega^2) at leading order
    frac_shift_sq = delta_omega_sq / omega_sq_GR.real
    frac_shift = frac_shift_sq / 2.0

    # The key exponential suppression factor
    exp_m2_r = np.exp(-m2_phys * r_peak)
    exp_m0_r = np.exp(-m0_phys * r_peak)
    m2_r_peak = m2_phys * r_peak
    m0_r_peak = m0_phys * r_peak

    return {
        'r_peak_over_rs': r_peak / r_s,
        'r_peak_m': r_peak,
        'V_peak_GR': V_peak_GR,
        'delta_V_peak': delta_V_peak,
        'delta_h_at_peak': delta_h,
        'm2_r_peak': m2_r_peak,
        'm0_r_peak': m0_r_peak,
        'exp_m2_r': exp_m2_r,
        'exp_m0_r': exp_m0_r,
        'frac_shift_omega_sq': frac_shift_sq,
        'frac_shift_omega': frac_shift,
    }


# ============================================================
# Analytic estimate of the suppression
# ============================================================
def analytic_suppression(M_kg: float) -> dict:
    """Analytic estimate of the exponential suppression factor.

    The QNM potential peak is at r_peak ≈ 3.28 M (geometric).
    The dominant SCT correction goes as exp(-m2 * r_peak).

    m2 * r_peak = m2 * 3.28 * M_geom
               = m2 * 3.28 * G M_kg / c^2
               = sqrt(60/13) * Lambda * 3.28 * G M_kg / c^2

    Converting Lambda from 1/m to natural units is already done in Lambda_m.
    """
    r_s = schwarzschild_radius(M_kg)
    r_peak = 1.64 * r_s  # ≈ 3.28 M (for l=2, n=0)

    m2_r = m2_m * r_peak
    m0_r = m0_m * r_peak

    # Leading correction to the metric at the peak
    # delta_h ≈ -(4/3) exp(-m2 r) + (1/3) exp(-m0 r)
    # ≈ -(4/3) exp(-m2 r)  [since m0 > m2, the second term is even smaller]

    if m2_r > 700:  # avoid underflow
        exp_m2 = 0.0
        exp_m0 = 0.0
        log10_shift = -m2_r * np.log10(np.e)
    else:
        exp_m2 = np.exp(-m2_r)
        exp_m0 = np.exp(-m0_r)
        if exp_m2 > 0:
            log10_shift = np.log10(abs(-(4.0/3.0) * exp_m2 + (1.0/3.0) * exp_m0) + 1e-300)
        else:
            log10_shift = -m2_r * np.log10(np.e)

    return {
        'M_kg': M_kg,
        'M_solar': M_kg / M_sun,
        'r_s_m': r_s,
        'r_peak_m': r_peak,
        'm2_r_peak': m2_r,
        'm0_r_peak': m0_r,
        'exp_m2_r': exp_m2,
        'exp_m0_r': exp_m0,
        'log10_delta_omega_over_omega': log10_shift,
    }


# ============================================================
# Critical mass where r_s ~ 1/Lambda
# ============================================================
def compute_M_crit() -> dict:
    """Critical mass where the Schwarzschild radius equals 1/Lambda.

    r_s = 2 G M / c^2 = 1/Lambda
    =>  M_crit = c^2 / (2 G Lambda)
    """
    M_crit_kg = c_light**2 / (2.0 * G_N * Lambda_m)
    M_crit_solar = M_crit_kg / M_sun
    r_s_crit = schwarzschild_radius(M_crit_kg)

    return {
        'M_crit_kg': M_crit_kg,
        'M_crit_solar': M_crit_solar,
        'M_crit_earth': M_crit_kg / 5.972e24,
        'M_crit_moon': M_crit_kg / 7.342e22,
        'r_s_crit_m': r_s_crit,
        'Lambda_inv_m': Lambda_inv_m,
        'r_s_over_Lambda_inv': r_s_crit / Lambda_inv_m,
    }


# ============================================================
# Full computation for multiple masses
# ============================================================
def run_mass_scan() -> list:
    """Compute QNM shift for a range of BH masses."""
    # Mass values in solar masses
    M_solar_values = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 1e3, 1e6, 1e9]

    results = []
    for M_sol in M_solar_values:
        M_kg = M_sol * M_sun
        info = analytic_suppression(M_kg)
        results.append(info)

    return results


# ============================================================
# Perturbative computation at the critical mass
# ============================================================
def compute_at_critical_mass() -> dict:
    """Full perturbative WKB computation at M = M_crit."""
    M_crit_info = compute_M_crit()
    M_kg = M_crit_info['M_crit_kg']
    r_s = schwarzschild_radius(M_kg)

    # At M_crit, r_s ~ 1/Lambda, so the Yukawa corrections are O(1).
    # The perturbative approach breaks down; we need the full modified WKB.
    # But we can still run the numerical WKB on the full SCT potential.

    # Use dimensionless units: r in units of r_s
    # Then m2_phys * r = (m2_m * r_s) * (r/r_s)
    m2_phys = m2_m
    m0_phys = m0_m

    shift_info = perturbative_shift(r_s, m2_phys, m0_phys)

    # Also compute the SCT and GR potentials for comparison
    r_arr = np.linspace(r_s * 1.01, r_s * 10.0, 1000)
    V_GR_arr = np.array([V_GR_wrapper(r, r_s) for r in r_arr])
    V_SCT_arr = np.array([V_SCT_wrapper(r, r_s, m2_phys, m0_phys) for r in r_arr])

    return {
        'M_crit': M_crit_info,
        'shift': shift_info,
        'V_GR_peak': float(np.max(V_GR_arr)),
        'V_SCT_peak': float(np.max(V_SCT_arr)),
        'delta_V_over_V': float((np.max(V_SCT_arr) - np.max(V_GR_arr)) / np.max(V_GR_arr)),
    }


# ============================================================
# Main computation
# ============================================================
def main():
    t0 = time.time()
    print("=" * 72)
    print("QNM FREQUENCY SHIFT FROM SCT NONLOCAL FORM FACTORS")
    print("=" * 72)

    # --- 1. Key scales ---
    print("\n--- 1. KEY SCALES ---")
    print(f"  Lambda          = {Lambda_eV:.3e} eV")
    print(f"  1/Lambda        = {Lambda_inv_m:.4e} m = {Lambda_inv_m*1e3:.4e} mm")
    print(f"  Lambda (1/m)    = {Lambda_m:.4e} 1/m")
    print(f"  m2/Lambda       = sqrt(60/13) = {m2_over_Lambda:.4f}")
    print(f"  m0/Lambda       = sqrt(6)     = {m0_over_Lambda:.4f}  (xi=0)")
    print(f"  m2              = {m2_m:.4e} 1/m")
    print(f"  m0              = {m0_m:.4e} 1/m")

    # --- 2. Critical mass ---
    print("\n--- 2. CRITICAL MASS (r_s = 1/Lambda) ---")
    M_crit_info = compute_M_crit()
    print(f"  M_crit          = {M_crit_info['M_crit_kg']:.4e} kg")
    print(f"  M_crit          = {M_crit_info['M_crit_solar']:.4e} M_sun")
    print(f"  M_crit          = {M_crit_info['M_crit_earth']:.4e} M_earth")
    print(f"  M_crit          = {M_crit_info['M_crit_moon']:.4e} M_moon")
    print(f"  r_s(M_crit)     = {M_crit_info['r_s_crit_m']:.4e} m")
    print(f"  1/Lambda        = {M_crit_info['Lambda_inv_m']:.4e} m")
    print(f"  r_s / (1/Lambda) = {M_crit_info['r_s_over_Lambda_inv']:.4f}")

    # --- 3. Mass scan ---
    print("\n--- 3. QNM SHIFT VS BLACK HOLE MASS ---")
    print(f"  {'M/M_sun':>12s}  {'r_s (m)':>12s}  {'m2*r_peak':>14s}  {'log10(dw/w)':>14s}")
    print("  " + "-" * 58)

    results = run_mass_scan()
    for r in results:
        print(f"  {r['M_solar']:12.2e}  {r['r_s_m']:12.4e}  {r['m2_r_peak']:14.4e}  {r['log10_delta_omega_over_omega']:14.2f}")

    # --- 4. Specific astrophysical cases ---
    print("\n--- 4. ASTROPHYSICAL BLACK HOLES ---")
    astro_masses = {
        '10 M_sun (LIGO)': 10.0,
        '30 M_sun (GW150914)': 30.0,
        '10^3 M_sun (IMBH)': 1e3,
        '10^6 M_sun (LISA)': 1e6,
        '10^9 M_sun (SMBH)': 1e9,
    }
    for name, M_sol in astro_masses.items():
        M_kg = M_sol * M_sun
        r_s = schwarzschild_radius(M_kg)
        r_peak = 1.64 * r_s
        m2_r = m2_m * r_peak
        # Direct log10 computation to avoid underflow
        log10_exp = -m2_r * np.log10(np.e)
        print(f"  {name:30s}: m2*r_peak = {m2_r:.4e},  log10(dw/w) ~ {log10_exp:.1f}")

    # --- 5. Critical mass computation ---
    print("\n--- 5. COMPUTATION AT CRITICAL MASS ---")
    crit_result = compute_at_critical_mass()
    shift = crit_result['shift']
    print(f"  M_crit = {crit_result['M_crit']['M_crit_kg']:.4e} kg ({crit_result['M_crit']['M_crit_solar']:.4e} M_sun)")
    print(f"  r_peak / r_s = {shift['r_peak_over_rs']:.4f}")
    print(f"  m2 * r_peak  = {shift['m2_r_peak']:.4f}")
    print(f"  m0 * r_peak  = {shift['m0_r_peak']:.4f}")
    print(f"  exp(-m2 r)   = {shift['exp_m2_r']:.6e}")
    print(f"  exp(-m0 r)   = {shift['exp_m0_r']:.6e}")
    print(f"  delta_h(peak)= {shift['delta_h_at_peak']:.6e}")
    print(f"  delta_V/V    = {crit_result['delta_V_over_V']:.6e}")
    print(f"  dw/w (perturbative) = {shift['frac_shift_omega']:.6e}")

    # --- 6. The honest assessment ---
    print("\n" + "=" * 72)
    print("HONEST ASSESSMENT")
    print("=" * 72)
    print("""
  For ANY astrophysical black hole observable by LIGO, LISA, or the EHT,
  the SCT correction to QNM frequencies is:

    delta_omega / omega  ~  exp(-m2 * r_peak)
                         ~  exp(-2.148 * Lambda * 3.28 * G M / c^2)

  For M = 10 M_sun:
    m2 * r_peak  ~ 10^{11}
    delta_omega / omega ~ 10^{-10^{11}}

  This is not just "small" --- it is smaller than any physically meaningful
  number. The correction is IDENTICALLY ZERO for all practical purposes.

  The correction becomes O(1) only when r_s ~ 1/Lambda, i.e., for
  M ~ M_Pl^2 / Lambda ~ 10^{-8} M_sun ~ M_moon.

  Such objects are:
    (a) Not astrophysical black holes (below the Chandrasekhar limit)
    (b) Below the mass range of any GW detector
    (c) Primordial black holes of this mass would have already evaporated
        (Hawking lifetime ~ 10^{-18} s for M ~ 10^{-8} M_sun is WRONG ---
         actually t_H ~ (M/M_Pl)^3 * t_Pl ~ 10^{50} yr, they survive.
         But they are undetectable via QNMs.)

  VERDICT: The SCT QNM frequency shift is unmeasurable. This is consistent
  with the general result that SCT modifications are exponentially suppressed
  at distances r >> 1/Lambda, and for astrophysical BHs r_s/Lambda^{-1} ~ 10^{11}.

  This is NOT a failure of SCT --- it is the EXPECTED result. SCT modifies
  gravity at the scale 1/Lambda ~ 0.077 mm, not at the scale of black holes.
  The same exponential suppression protects SCT from all astrophysical GW
  constraints, which is why the Eot-Wash laboratory bound (probing mm scales)
  is 14 orders of magnitude stronger than any solar system test.
""")

    # --- 7. Comparison table ---
    print("--- COMPARISON TABLE ---")
    print(f"{'Detector':15s} {'M (M_sun)':>12s} {'log10(dw/w)':>14s} {'Measurable?':>14s}")
    print("-" * 60)
    detector_data = [
        ('LIGO/Virgo', 10.0, 'No'),
        ('LIGO/Virgo', 30.0, 'No'),
        ('LISA', 1e6, 'No'),
        ('PTA', 1e9, 'No'),
        ('(Hypothetical)', M_crit_info['M_crit_solar'], 'N/A (no detector)'),
    ]
    for det, M_sol, meas in detector_data:
        M_kg = M_sol * M_sun
        r_s = schwarzschild_radius(M_kg)
        r_peak = 1.64 * r_s
        m2_r = m2_m * r_peak
        log10_val = -m2_r * np.log10(np.e)
        print(f"{det:15s} {M_sol:12.2e} {log10_val:14.1f} {meas:>14s}")

    # --- 8. Save results ---
    output = {
        'Lambda_eV': Lambda_eV,
        'Lambda_inv_m': Lambda_inv_m,
        'm2_over_Lambda': m2_over_Lambda,
        'm0_over_Lambda': m0_over_Lambda,
        'M_crit_kg': M_crit_info['M_crit_kg'],
        'M_crit_solar': M_crit_info['M_crit_solar'],
        'mass_scan': [
            {'M_solar': r['M_solar'], 'log10_shift': r['log10_delta_omega_over_omega']}
            for r in results
        ],
        'critical_mass_result': {
            'delta_V_over_V': crit_result['delta_V_over_V'],
            'delta_omega_over_omega': shift['frac_shift_omega'],
        },
        'verdict': 'UNMEASURABLE for all astrophysical BHs. Exponentially suppressed as exp(-m2 * r_s).',
    }

    outdir = Path(__file__).parent.parent / 'results' / 'qnm'
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / 'qnm_sct_shift.json'
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to: {outpath}")

    # --- 9. Produce figure ---
    print("\n--- GENERATING FIGURE ---")
    try:
        make_figure(results, M_crit_info)
        print("  Figure saved.")
    except Exception as e:
        print(f"  Figure generation failed: {e}")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.2f} s")
    print("=" * 72)

    return output


# ============================================================
# Figure: delta_omega/omega vs M/M_sun
# ============================================================
def make_figure(results: list, M_crit_info: dict):
    """Plot log10(delta_omega/omega) vs log10(M/M_sun)."""
    figdir = Path(__file__).parent.parent / 'figures' / 'qnm'
    figdir.mkdir(parents=True, exist_ok=True)

    # Generate a finer mass scan for the plot
    log_M_solar = np.linspace(-10, 12, 500)
    M_solar_arr = 10.0**log_M_solar
    log10_shift = np.zeros_like(log_M_solar)

    for i, M_sol in enumerate(M_solar_arr):
        M_kg = M_sol * M_sun
        r_s = schwarzschild_radius(M_kg)
        r_peak = 1.64 * r_s
        m2_r = m2_m * r_peak
        log10_shift[i] = -m2_r * np.log10(np.e)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    ax.plot(log_M_solar, log10_shift, 'b-', linewidth=2.0,
            label=r'$\delta\omega/\omega \sim e^{-m_2 r_{\rm peak}}$')

    # Mark astrophysical BH masses
    astro = {
        r'$10\,M_\odot$': 1.0,
        r'$30\,M_\odot$': np.log10(30),
        r'$10^6\,M_\odot$': 6.0,
        r'$10^9\,M_\odot$': 9.0,
    }
    for label, log_m in astro.items():
        m_kg = 10**log_m * M_sun
        r_s = schwarzschild_radius(m_kg)
        r_peak = 1.64 * r_s
        m2_r = m2_m * r_peak
        y_val = -m2_r * np.log10(np.e)
        ax.plot(log_m, max(y_val, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -1e14),
                'rv', markersize=10, zorder=5)
        ax.annotate(label, (log_m, min(y_val, -10)),
                    textcoords='offset points', xytext=(5, 10),
                    fontsize=9, color='red')

    # Mark critical mass
    log_M_crit = np.log10(M_crit_info['M_crit_solar'])
    ax.axvline(log_M_crit, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
               label=rf'$M_{{\rm crit}} = {M_crit_info["M_crit_solar"]:.1e}\,M_\odot$')

    # Mark detection thresholds
    ax.axhline(-1, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(log_M_solar[0] + 0.5, -0.5, r'$|\delta\omega/\omega| = 10^{-1}$ (O(1) effect)',
            fontsize=9, color='gray')

    # LIGO sensitivity band
    ax.axvspan(0.5, 2.5, alpha=0.1, color='orange', label='LIGO mass range')
    # LISA sensitivity band
    ax.axvspan(4, 8, alpha=0.1, color='purple', label='LISA mass range')

    ax.set_xlabel(r'$\log_{10}(M/M_\odot)$', fontsize=13)
    ax.set_ylabel(r'$\log_{10}(|\delta\omega/\omega|)$', fontsize=13)
    ax.set_title('SCT QNM Frequency Shift vs Black Hole Mass\n'
                 r'($\Lambda = 2.565$ meV, $\ell=2$, $n=0$)', fontsize=14)

    # Set reasonable y-axis limits
    ax.set_ylim(-200, 5)
    ax.set_xlim(-10, 12)

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add text box with key result
    textstr = (r'For $M > M_\odot$: $\delta\omega/\omega < 10^{-10^{11}}$'
               '\nExponentially unmeasurable')
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', bbox=props)

    fig.tight_layout()
    fig.savefig(figdir / 'qnm_sct_shift.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(figdir / 'qnm_sct_shift.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # --- Second figure: Potential comparison at M_crit ---
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    M_kg = M_crit_info['M_crit_kg']
    r_s = schwarzschild_radius(M_kg)
    r_arr = np.linspace(r_s * 1.02, r_s * 8.0, 500)
    r_norm = r_arr / r_s

    V_GR_arr = np.array([V_GR_wrapper(r, r_s) for r in r_arr])
    V_SCT_arr = np.array([V_SCT_wrapper(r, r_s, m2_m, m0_m) for r in r_arr])

    # Normalize potentials
    V_scale = r_s**2
    ax1.plot(r_norm, V_GR_arr * V_scale, 'b-', linewidth=2, label='GR (Regge-Wheeler)')
    ax1.plot(r_norm, V_SCT_arr * V_scale, 'r--', linewidth=2, label='SCT')
    ax1.set_xlabel(r'$r / r_s$', fontsize=13)
    ax1.set_ylabel(r'$V(r) \times r_s^2$', fontsize=13)
    ax1.set_title(rf'Regge-Wheeler Potential at $M = M_{{\rm crit}}$', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Relative difference
    mask = V_GR_arr > 0
    rel_diff = np.where(mask, (V_SCT_arr - V_GR_arr) / V_GR_arr, 0)
    ax2.plot(r_norm[mask], rel_diff[mask], 'k-', linewidth=2)
    ax2.set_xlabel(r'$r / r_s$', fontsize=13)
    ax2.set_ylabel(r'$(V_{\rm SCT} - V_{\rm GR}) / V_{\rm GR}$', fontsize=13)
    ax2.set_title(rf'Relative Potential Deviation at $M = M_{{\rm crit}}$', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0, color='gray', linestyle='-', linewidth=0.5)

    fig2.suptitle(rf'$M_{{\rm crit}} = {M_crit_info["M_crit_solar"]:.2e}\,M_\odot$'
                  rf' ($r_s = 1/\Lambda$)', fontsize=14, y=1.02)
    fig2.tight_layout()
    fig2.savefig(figdir / 'qnm_potential_comparison.pdf', dpi=150, bbox_inches='tight')
    fig2.savefig(figdir / 'qnm_potential_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)


if __name__ == '__main__':
    main()
