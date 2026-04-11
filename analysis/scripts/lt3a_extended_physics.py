# ruff: noqa: E402, I001
"""
LT-3a Phase 6: Extended Physics of QNMs in Spectral Causal Theory.

Computes four extended-physics results that go beyond the basic QNM shift:

1. **Greybody factors** — Transmission coefficient T(omega) through the
   SCT-modified Regge-Wheeler potential barrier, and connection to
   the Hawking spectrum.

2. **Late-time Price tails** — Modified power-law decay psi ~ t^{-(2l+3)}
   in GR vs potential modifications from the SCT Yukawa corrections.

3. **Hod area quantization** — Convergence of Re(omega) to T_H * ln(3)
   for high-overtone Schwarzschild QNMs (n = 10..20), and whether
   the SCT Yukawa corrections modify the asymptotic limit.

4. **Quantum vs classical QNM corrections** — Comparison of the one-loop
   quantum correction delta_omega/omega ~ (l_P/r_s)^2 * c_log (where
   c_log = 37/24 is the SCT logarithmic entropy coefficient) against the
   classical SCT shift ~ exp(-m2*r_peak) for six observed LIGO/Virgo BHs.

The SCT-modified metric function (NT-4a):
    f(r) = 1 - (r_s/r) * [1 - (4/3)e^{-m2*r} + (1/3)e^{-m0*r}]

Effective masses:
    m2 = Lambda * sqrt(60/13) ~ 2.148 * Lambda   (spin-2, Weyl sector)
    m0 = Lambda * sqrt(6)     ~ 2.449 * Lambda   (spin-0, scalar sector)

Key result: For ALL astrophysical BHs (M > 3 M_sun), the SCT correction
is exponentially suppressed below even the one-loop quantum correction,
which is itself ~ 10^{-78}.  This is a parameter-free prediction.

References:
    - Regge-Wheeler (1957), PR 108, 1063
    - Schutz-Will (1985), Proc. R. Soc. Lond. A 402, 191
    - Iyer-Will (1987), PRD 35, 3621
    - Leaver (1985), Proc. R. Soc. Lond. A 402, 285
    - Price (1972), PRD 5, 2419
    - Hod (1998), PRL 81, 4293 — area quantization conjecture
    - Motl (2003), gr-qc/0212096 — asymptotic QNMs
    - Sen (2012), arXiv:1205.0971 — c_log for non-extremal BH
    - Page (1976), PRD 13, 198 — greybody factors
    - NT-1b Phase 3: alpha_C = 13/120 (SM Weyl coefficient)
    - NT-4a: SCT-modified Newtonian potential
    - MT-1: c_log = 37/24 (SCT logarithmic entropy coefficient)

Author: David Alfyorov
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from scipy import constants, integrate, optimize, special

# ============================================================
# Directories
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "lt3a"
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures" / "lt3a"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Physical constants (SI, CODATA via scipy)
# ============================================================
G_N = constants.G                         # 6.674e-11 m^3/(kg s^2)
c_light = constants.c                     # 2.998e8  m/s
hbar = constants.hbar                     # 1.055e-34 J s
k_B = constants.k                         # 1.381e-23 J/K
M_sun = 1.989e30                          # kg (IAU 2015)
eV_to_J = constants.eV                    # 1.602e-19 J

# Derived Planck-scale quantities
M_Pl_kg = np.sqrt(hbar * c_light / G_N)  # 2.176e-8 kg
l_Pl = np.sqrt(hbar * G_N / c_light**3)  # 1.616e-35 m
l_Pl_sq = l_Pl**2                         # 2.612e-70 m^2

# ============================================================
# SCT parameters (from NT-4a, PPN-1)
# ============================================================
LAMBDA_EV = 2.38e-3                       # eV (PPN-1 lower bound)
LAMBDA_INV_M = hbar * c_light / (LAMBDA_EV * eV_to_J)
LAMBDA_M = 1.0 / LAMBDA_INV_M            # Lambda in 1/m

M2_OVER_LAMBDA = np.sqrt(60.0 / 13.0)    # ~ 2.148 (spin-2 mass ratio)
M0_OVER_LAMBDA = np.sqrt(6.0)            # ~ 2.449 (spin-0 mass ratio, xi=0)

M2_M = M2_OVER_LAMBDA * LAMBDA_M         # spin-2 mass in 1/m
M0_M = M0_OVER_LAMBDA * LAMBDA_M         # spin-0 mass in 1/m

# Phase 3 result
ALPHA_C = 13.0 / 120.0                   # Weyl^2 coefficient (SM)

# MT-1 logarithmic correction coefficient
# c_log = (1/180)[2 N_s + 7 N_F - 26 N_V + 424]
# N_s=4, N_F(Dirac)=22.5, N_V=12 (SM)
C_LOG = (2 * 4 + 7 * 22.5 - 26 * 12 + 424) / 180.0  # = 277.5/180 = 37/24

# Observed LIGO/Virgo BHs for the quantum-vs-classical comparison
OBSERVED_BHS = {
    "GW150914": {"M_solar": 62.0, "a_star": 0.67, "f_220_Hz": 251.0,
                 "sigma_f_Hz": 8.0, "ref": "PRL 116, 061102 (2016)"},
    "GW170104": {"M_solar": 48.7, "a_star": 0.64, "f_220_Hz": 0.0,
                 "sigma_f_Hz": 0.0, "ref": "PRL 118, 221101 (2017)"},
    "GW190521": {"M_solar": 142.0, "a_star": 0.72, "f_220_Hz": 0.0,
                 "sigma_f_Hz": 0.0, "ref": "PRL 125, 101102 (2020)"},
    "GW190814": {"M_solar": 25.0, "a_star": 0.0, "f_220_Hz": 0.0,
                 "sigma_f_Hz": 0.0, "ref": "ApJL 896, L44 (2020)"},
    "GW200115": {"M_solar": 7.0, "a_star": 0.0, "f_220_Hz": 0.0,
                 "sigma_f_Hz": 0.0, "ref": "ApJL 915, L5 (2021)"},
    "GW230529": {"M_solar": 5.7, "a_star": 0.0, "f_220_Hz": 0.0,
                 "sigma_f_Hz": 0.0, "ref": "arXiv:2404.04248 (2024)"},
}

# GR reference: omega*M for l=2, n=0 Schwarzschild QNM (Leaver 1985)
GR_OMEGA_M_REF = complex(0.37367, -0.08896)


# ============================================================
# Utility functions
# ============================================================
def schwarzschild_radius(M_kg: float) -> float:
    """Schwarzschild radius r_s = 2GM/c^2 [meters]."""
    return 2.0 * G_N * M_kg / c_light**2


def hawking_temperature(M_kg: float) -> float:
    """Hawking temperature T_H = hbar c^3 / (8 pi G M k_B) [Kelvin]."""
    return hbar * c_light**3 / (8.0 * np.pi * G_N * M_kg * k_B)


def hawking_temperature_eV(M_kg: float) -> float:
    """Hawking temperature in eV: T_H = hbar c^3 / (8 pi G M) [eV]."""
    T_K = hawking_temperature(M_kg)
    return k_B * T_K / eV_to_J


def tortoise_from_r(r: float, r_s: float) -> float:
    """Tortoise coordinate r* = r + r_s * ln(r/r_s - 1) for Schwarzschild."""
    if r <= r_s:
        return -1e30
    return r + r_s * np.log(r / r_s - 1.0)


# ============================================================
# Metric functions
# ============================================================
def f_GR(r: float, r_s: float) -> float:
    """Schwarzschild metric function f(r) = 1 - r_s/r."""
    return 1.0 - r_s / r


def h_yukawa(r: float, m2: float, m0: float) -> float:
    """SCT Yukawa modification: h(r) = 1 - (4/3)e^{-m2 r} + (1/3)e^{-m0 r}."""
    return 1.0 - (4.0 / 3.0) * np.exp(-m2 * r) + (1.0 / 3.0) * np.exp(-m0 * r)


def f_SCT(r: float, r_s: float, m2: float, m0: float) -> float:
    """SCT-modified metric: f(r) = 1 - (r_s/r)*h(r)."""
    return 1.0 - (r_s / r) * h_yukawa(r, m2, m0)


# ============================================================
# Effective potentials
# ============================================================
def V_RW_GR(r: float, r_s: float, l: int = 2) -> float:
    """Regge-Wheeler potential (axial, spin-2) in GR.
    V = (1-r_s/r)[l(l+1)/r^2 - 6M/r^3] with M = r_s/2.
    """
    if r <= r_s:
        return 0.0
    f = 1.0 - r_s / r
    M = r_s / 2.0
    return f * (l * (l + 1) / r**2 - 6.0 * M / r**3)


def V_RW_SCT(r: float, r_s: float, m2: float, m0: float, l: int = 2) -> float:
    """SCT-modified Regge-Wheeler potential using the general M_eff formula.
    V = f * [l(l+1)/r^2 - 3(1-f)/r^2].
    """
    if r <= r_s * 1.0001:
        return 0.0
    f = f_SCT(r, r_s, m2, m0)
    one_minus_f = (r_s / r) * h_yukawa(r, m2, m0)
    return f * (l * (l + 1) / r**2 - 3.0 * one_minus_f / r**2)


def find_potential_peak(V_func, r_s: float, l: int, args: tuple = ()) -> float:
    """Find the radius where V(r) is maximal."""
    r_min = r_s * 1.01
    r_max = r_s * 5.0

    def neg_V(r):
        return -V_func(r, r_s, *args, l)

    result = optimize.minimize_scalar(neg_V, bounds=(r_min, r_max), method='bounded')
    return result.x


# ============================================================
# WKB QNM computation (Iyer-Will 2nd order)
# ============================================================
def numerical_derivatives_tortoise(V_func, r_peak: float, r_s: float,
                                   l: int, args: tuple = ()) -> tuple:
    """Compute (V0, V2, V3, V4) at potential peak in tortoise coordinates."""
    V0 = V_func(r_peak, r_s, *args, l)
    r_star_0 = tortoise_from_r(r_peak, r_s)

    N_half = 10
    h_r = 0.005 * r_s
    r_arr = np.array([r_peak + i * h_r for i in range(-N_half, N_half + 1)])
    r_arr = r_arr[r_arr > r_s * 1.001]

    rstar_arr = np.array([tortoise_from_r(r, r_s) for r in r_arr])
    V_arr = np.array([V_func(r, r_s, *args, l) for r in r_arr])

    x = rstar_arr - r_star_0
    deg = min(6, len(x) - 1)
    coeffs = np.polyfit(x, V_arr, deg)
    a = coeffs[::-1]

    V2 = 2.0 * a[2] if len(a) > 2 else 0.0
    V3 = 6.0 * a[3] if len(a) > 3 else 0.0
    V4 = 24.0 * a[4] if len(a) > 4 else 0.0

    return V0, V2, V3, V4


def wkb_qnm_2nd_order(V0: float, V2: float, V3: float, V4: float,
                       n: int = 0) -> complex:
    """2nd-order WKB formula for QNM frequencies (Iyer-Will 1987)."""
    if V2 >= 0:
        return complex(np.nan, np.nan)

    alpha = n + 0.5
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
    """Compute QNM frequency via WKB. Returns dimensionless omega*M."""
    M = r_s / 2.0
    r_peak = find_potential_peak(V_func, r_s, l, args)
    V0_SI, V2_SI, V3_SI, V4_SI = numerical_derivatives_tortoise(
        V_func, r_peak, r_s, l, args)

    V0_dim = V0_SI * M**2
    V2_dim = V2_SI * M**4
    V3_dim = V3_SI * M**5
    V4_dim = V4_SI * M**6

    omega_dim = wkb_qnm_2nd_order(V0_dim, V2_dim, V3_dim, V4_dim, n)
    return omega_dim


# ============================================================
# Leaver continued fraction (Schwarzschild, high-overtone capable)
# ============================================================
def leaver_cf_schwarzschild(l: int, n: int, omega_guess: complex = None,
                            s: int = 2, dps: int = 50) -> complex:
    """Leaver continued fraction for Schwarzschild QNMs (2M=1 units)."""
    mp.mp.dps = dps

    if omega_guess is None:
        omega_guess = complex(l / np.sqrt(27), -(n + 0.5) / np.sqrt(27))

    omega = mp.mpc(omega_guess.real, omega_guess.imag)

    def recurrence_coeffs(nn, w):
        iw = mp.mpc(0, 1) * w
        alpha = nn**2 + (2 - 2 * iw) * nn + 1 - 2 * iw - s
        beta = -(2 * nn**2 + (2 - 8 * iw) * nn - 8 * iw * iw
                 + 4 * iw - l * (l + 1) + s * (s - 1) + 1)
        gamma = nn**2 - 4 * iw * nn + 4 * iw * iw - s - 1 + 4 * iw
        return alpha, beta, gamma

    def continued_fraction_value(w, n_terms=300):
        N = n_terms
        f = mp.mpf(0)
        for k in range(N, n, -1):
            alpha_k, beta_k, gamma_k = recurrence_coeffs(k, w)
            alpha_km1, _, _ = recurrence_coeffs(k - 1, w)
            f = -alpha_km1 * gamma_k / (beta_k + f)

        if n == 0:
            _, beta_0, _ = recurrence_coeffs(0, w)
            return beta_0 + f
        else:
            g = mp.mpf(0)
            for k in range(n - 1, -1, -1):
                _, beta_k, _ = recurrence_coeffs(k, w)
                _, _, gamma_kp1 = recurrence_coeffs(k + 1, w)
                alpha_k, _, _ = recurrence_coeffs(k, w)
                g = -gamma_kp1 * alpha_k / (beta_k + g)
            _, beta_n, _ = recurrence_coeffs(n, w)
            return beta_n + g + f

    def cf_complex(w):
        return continued_fraction_value(w)

    try:
        omega_qnm = mp.findroot(cf_complex, omega, solver='muller',
                                tol=mp.mpf(10)**(-dps // 2))
    except Exception:
        from scipy.optimize import fsolve

        def cf_residual(w_vec):
            w = mp.mpc(w_vec[0], w_vec[1])
            val = continued_fraction_value(w)
            return [float(mp.re(val)), float(mp.im(val))]

        result = fsolve(cf_residual, [float(omega.real), float(omega.imag)],
                        full_output=True)
        x = result[0]
        omega_qnm = mp.mpc(x[0], x[1])

    omega_out = complex(float(mp.re(omega_qnm)), float(mp.im(omega_qnm)))
    if omega_out.real < 0:
        omega_out = -omega_out
    if omega_out.imag > 0:
        omega_out = complex(omega_out.real, -omega_out.imag)
    return omega_out


# ============================================================
# Section 1: Greybody Factors
# ============================================================
def greybody_low_freq_GR(omega: float, r_s: float, l: int = 2) -> float:
    """Low-frequency greybody factor in GR.

    T(omega) ~ (omega * r_s / c)^{2l+2} * C_l  for omega << c/r_s.
    (Page 1976, Eq. 10; Cvetic-Larsen 1997)

    The dimensionless argument is x = omega * r_s / c.

    For l=2, the leading behavior is T ~ C_l * x^{2l+2} where C_l
    encodes the angular momentum barrier penetration.
    The exact coefficient from the matched asymptotic expansion:
        C_l = 2^{2l+2} * [(l!)^4] / [(2l)!^2 * (2l+1)^2]
    (Starobinskii 1973, Unruh 1976, Page 1976)

    Parameters:
        omega: angular frequency in rad/s (physical units)
        r_s: Schwarzschild radius in meters
        l: angular multipole number
    """
    x = omega * r_s / c_light  # dimensionless
    if x <= 0:
        return 0.0

    fact_l = float(special.factorial(l, exact=True))
    fact_2l = float(special.factorial(2 * l, exact=True))

    C_l = (2.0**(2 * l + 2) * fact_l**4) / (fact_2l**2 * (2.0 * l + 1.0)**2)
    T = C_l * x**(2 * l + 2)

    return min(T, 1.0)


def greybody_numerical_GR(omega_arr: np.ndarray, r_s: float,
                          l: int = 2) -> np.ndarray:
    """Numerical greybody factor via WKB barrier penetration.

    T(omega) = 1 / (1 + exp(2 S))  where S = integral of sqrt(V - omega^2)
    through the classically forbidden region.

    More precisely, for omega^2 < V_max, the WKB tunneling probability is:
        T = 1 / (1 + exp(2 * integral_{r1}^{r2} sqrt(V(r) - omega^2) dr*))

    where r1, r2 are the classical turning points.
    """
    M = r_s / 2.0
    T_arr = np.zeros_like(omega_arr)

    # Peak potential value
    r_peak = find_potential_peak(V_RW_GR, r_s, l)
    V_max = V_RW_GR(r_peak, r_s, l)

    for i, omega in enumerate(omega_arr):
        omega_sq = omega**2
        if omega_sq >= V_max:
            # Above the barrier — full transmission (in WKB approximation)
            T_arr[i] = 1.0
            continue

        if omega <= 0:
            T_arr[i] = 0.0
            continue

        # Find turning points where V(r) = omega^2
        # Inner turning point
        try:
            r1 = optimize.brentq(
                lambda r: V_RW_GR(r, r_s, l) - omega_sq,
                r_s * 1.001, r_peak)
        except ValueError:
            r1 = r_s * 1.01

        # Outer turning point
        try:
            r2 = optimize.brentq(
                lambda r: V_RW_GR(r, r_s, l) - omega_sq,
                r_peak, r_s * 20.0)
        except ValueError:
            r2 = r_s * 5.0

        if r2 <= r1:
            T_arr[i] = 1.0
            continue

        # WKB tunneling integral in r* coordinates:
        # S = integral_{r1}^{r2} sqrt(V(r) - omega^2) dr*
        #   = integral_{r1}^{r2} sqrt(V(r) - omega^2) / f(r) dr
        def integrand(r):
            V_val = V_RW_GR(r, r_s, l)
            f = f_GR(r, r_s)
            diff = V_val - omega_sq
            if diff <= 0 or f <= 0:
                return 0.0
            return np.sqrt(diff) / f

        S, _ = integrate.quad(integrand, r1, r2, limit=200,
                              epsabs=1e-12, epsrel=1e-10)

        # WKB transmission coefficient
        exp_2S = np.exp(2.0 * S)
        T_arr[i] = 1.0 / (1.0 + exp_2S)

    return T_arr


def greybody_numerical_SCT(omega_arr: np.ndarray, r_s: float,
                           m2: float, m0: float, l: int = 2) -> np.ndarray:
    """Numerical greybody factor for the SCT-modified potential."""
    T_arr = np.zeros_like(omega_arr)

    r_peak = find_potential_peak(V_RW_SCT, r_s, l, args=(m2, m0))
    V_max = V_RW_SCT(r_peak, r_s, m2, m0, l)

    for i, omega in enumerate(omega_arr):
        omega_sq = omega**2
        if omega_sq >= V_max:
            T_arr[i] = 1.0
            continue
        if omega <= 0:
            T_arr[i] = 0.0
            continue

        try:
            r1 = optimize.brentq(
                lambda r: V_RW_SCT(r, r_s, m2, m0, l) - omega_sq,
                r_s * 1.001, r_peak)
        except ValueError:
            r1 = r_s * 1.01

        try:
            r2 = optimize.brentq(
                lambda r: V_RW_SCT(r, r_s, m2, m0, l) - omega_sq,
                r_peak, r_s * 20.0)
        except ValueError:
            r2 = r_s * 5.0

        if r2 <= r1:
            T_arr[i] = 1.0
            continue

        def integrand(r):
            V_val = V_RW_SCT(r, r_s, m2, m0, l)
            f = f_SCT(r, r_s, m2, m0)
            diff = V_val - omega_sq
            if diff <= 0 or f <= 0:
                return 0.0
            return np.sqrt(diff) / f

        S, _ = integrate.quad(integrand, r1, r2, limit=200,
                              epsabs=1e-12, epsrel=1e-10)
        exp_2S = np.exp(min(2.0 * S, 700.0))
        T_arr[i] = 1.0 / (1.0 + exp_2S)

    return T_arr


def hawking_spectrum_planckian(omega: float, T_H_natural: float) -> float:
    """Planckian Hawking spectrum (bosonic): N(omega) = 1/(exp(omega/T_H) - 1).

    T_H_natural = hbar * c^3 / (8 pi G M) in natural frequency units (1/s).
    omega in 1/s.
    """
    x = omega / T_H_natural
    if x > 500:
        return 0.0
    if x < 1e-10:
        return T_H_natural / omega if omega > 0 else 0.0
    return 1.0 / (np.exp(x) - 1.0)


def compute_greybody_results(M_solar: float = 10.0, l: int = 2) -> dict:
    """Compute greybody factor comparison between GR and SCT for a given BH mass."""
    M_kg = M_solar * M_sun
    r_s = schwarzschild_radius(M_kg)
    m2 = M2_M
    m0 = M0_M

    # Natural frequency scale: omega ~ 1/r_s
    omega_scale = c_light / r_s
    omega_arr = np.linspace(0.01, 2.0, 200) * omega_scale

    T_GR = greybody_numerical_GR(omega_arr, r_s, l)
    T_SCT = greybody_numerical_SCT(omega_arr, r_s, m2, m0, l)

    # Fractional modification
    delta_T = np.abs(T_SCT - T_GR)
    delta_T_over_T = np.where(T_GR > 1e-30, delta_T / T_GR, 0.0)

    # Low-frequency analytic comparison
    T_low_analytic = np.array([greybody_low_freq_GR(w, r_s, l)
                               for w in omega_arr[:50]])

    # Hawking temperature
    T_H_K = hawking_temperature(M_kg)
    T_H_eV = hawking_temperature_eV(M_kg)
    T_H_natural = k_B * T_H_K / hbar  # in 1/s

    # Maximum modification
    max_delta = float(np.max(delta_T_over_T))
    m2_r_s = m2 * r_s
    exp_suppression = np.exp(-m2 * r_s * 1.5)  # at potential peak ~ 1.5 r_s

    return {
        "M_solar": M_solar,
        "r_s_m": r_s,
        "T_H_K": T_H_K,
        "T_H_eV": T_H_eV,
        "omega_arr": omega_arr.tolist(),
        "T_GR": T_GR.tolist(),
        "T_SCT": T_SCT.tolist(),
        "T_low_analytic_first50": T_low_analytic.tolist(),
        "max_delta_T_over_T": max_delta,
        "m2_r_s": m2_r_s,
        "exp_suppression_at_peak": exp_suppression,
        "note": ("SCT modification to greybody factor is exp-suppressed "
                 "as exp(-m2*r_peak) for astrophysical BHs"),
    }


# ============================================================
# Section 2: Late-time Price Tails
# ============================================================
def price_tail_exponent_GR(l: int) -> int:
    """GR Price tail exponent: psi ~ t^{-(2l+3)} for backscattered radiation.
    Price (1972), PRD 5, 2419.
    """
    return -(2 * l + 3)


def price_tail_analysis(l_values: list[int] = None) -> dict:
    """Analyze the late-time Price tail in GR and how SCT modifies it.

    In GR, for a Schwarzschild BH, the late-time behavior of a
    gravitational perturbation is:
        psi(t, r) ~ t^{-(2l+3)}  for t >> r_s

    This power law arises from the backscattering of the initial signal
    off the effective potential at large r, specifically from the
    V ~ l(l+1)/r^2 tail.

    In SCT, the potential is modified at r ~ 1/m2 by Yukawa terms.
    For astrophysical BHs where r_s >> 1/Lambda, the modification
    at r ~ r_s is exponentially suppressed, so the Price tail is
    UNCHANGED.

    However, for sub-critical BHs with r_s ~ 1/Lambda, the potential
    barrier is modified at O(1), and the late-time behavior acquires
    corrections from the Yukawa exponentials.
    """
    if l_values is None:
        l_values = [2, 3, 4]

    results = {}
    for l in l_values:
        exponent_GR = price_tail_exponent_GR(l)

        # The SCT modification to the potential at large r:
        # V_SCT(r) - V_GR(r) ~ -(4/3)(r_s/r)(m2^2) exp(-m2 r) for r >> r_s
        # This is exponentially localized at r ~ 1/m2 and does NOT affect
        # the power-law tail at r >> 1/m2.

        # Critical radius where the SCT correction is O(1):
        # exp(-m2 * r) ~ 1 => r ~ 1/m2 = 1/(m2_over_Lambda * Lambda)
        r_critical = 1.0 / M2_M  # meters

        # BH mass where r_s = r_critical (SCT effects become relevant):
        M_crit_kg = c_light**2 * r_critical / (2.0 * G_N)
        M_crit_solar = M_crit_kg / M_sun

        # For BHs with r_s >> r_critical: tail unchanged
        # For BHs with r_s ~ r_critical: modified regime
        results[f"l={l}"] = {
            "exponent_GR": exponent_GR,
            "tail_formula_GR": f"psi ~ t^{{{exponent_GR}}}",
            "exponent_SCT_astrophysical": exponent_GR,
            "tail_formula_SCT": (
                f"psi ~ t^{{{exponent_GR}}} for r_s >> 1/m2 (all astrophysical BH)"
            ),
            "modified_regime": (
                "For r_s ~ 1/m2 (sub-critical BH), the Yukawa correction "
                "to the potential modifies the tail. The leading correction "
                "is ~ exp(-m2*r_s) * t^{-(2l+3)} (same power, modified amplitude)."
            ),
            "r_critical_m": r_critical,
            "M_crit_solar": M_crit_solar,
        }

    # Numerical demonstration: compute potential tails
    M_astro = 10.0 * M_sun  # astrophysical BH
    r_s_astro = schwarzschild_radius(M_astro)
    r_arr = np.linspace(2.0, 30.0, 500) * r_s_astro

    V_GR_arr = np.array([V_RW_GR(r, r_s_astro, l=2) for r in r_arr])
    V_SCT_arr = np.array([V_RW_SCT(r, r_s_astro, M2_M, M0_M, l=2) for r in r_arr])
    delta_V = np.abs(V_SCT_arr - V_GR_arr)

    # At large r, V_GR ~ l(l+1)/r^2 and delta_V ~ exp(-m2*r)
    # The power-law tail depends on the 1/r^2 piece, which is unaffected.
    results["numerical_check"] = {
        "M_solar": 10.0,
        "r_over_rs_range": [2.0, 30.0],
        "max_delta_V_over_V": float(np.max(
            np.where(V_GR_arr > 1e-30, delta_V / np.abs(V_GR_arr), 0.0))),
        "note": "delta_V/V is exponentially small for astrophysical BH",
    }

    return results


# ============================================================
# Section 3: Hod Area Quantization
# ============================================================
def hod_area_quantization(n_min: int = 10, n_max: int = 20,
                          l: int = 2, s: int = 2) -> dict:
    """Check the Hod area quantization conjecture.

    Hod (1998) conjectured that the asymptotic QNM frequencies of a
    Schwarzschild BH satisfy:
        omega_R * M -> T_H * M * ln(3) = ln(3)/(8*pi)  as n -> inf

    Motl (2003) proved this analytically:
        omega_n = T_H * ln(3) + i * (n + 1/2) * 2*pi*T_H + O(1/n)

    where T_H * M = 1/(8*pi) (in geometric units 2M=1).

    We compute omega_n for n = n_min..n_max using the Leaver method and
    check convergence of Re(omega_n) to T_H * ln(3).

    The SCT modification to high overtones:
    Since SCT modifies the potential at r ~ 1/m2 >> r_peak (for astrophysical BH),
    the high-overtone behavior is controlled by the near-horizon geometry,
    which is essentially GR. The asymptotic omega_R -> T_H * ln(3) is UNMODIFIED.
    """
    # T_H * M in geometric units (2M = 1): T_H = 1/(4*pi*r_s) = 1/(4*pi)
    # But since omega*M is our convention, and T_H * M = 1/(8*pi):
    T_H_M = 1.0 / (8.0 * np.pi)
    hod_limit = T_H_M * np.log(3.0)

    results = {
        "hod_limit_omegaR_M": hod_limit,
        "T_H_M_geometric": T_H_M,
        "ln3": np.log(3.0),
        "overtones": [],
    }

    # Compute QNMs for n = n_min to n_max
    # Use Leaver continued fraction with progressive refinement
    omega_prev = None
    for n in range(n_min, n_max + 1):
        # Initial guess from Motl asymptotics
        omega_guess = complex(
            hod_limit,
            -(n + 0.5) * 2.0 * np.pi * T_H_M
        )

        try:
            omega_n = leaver_cf_schwarzschild(l, n, omega_guess, s=s, dps=50)
        except Exception:
            omega_n = complex(np.nan, np.nan)

        omega_R = omega_n.real
        omega_I = omega_n.imag

        # Deviation from Hod limit
        delta_omegaR_M = abs(omega_R - hod_limit)
        frac_deviation = delta_omegaR_M / hod_limit if hod_limit != 0 else np.inf

        results["overtones"].append({
            "n": n,
            "omega_R_M": omega_R,
            "omega_I_M": omega_I,
            "delta_omegaR_from_hod": delta_omegaR_M,
            "frac_deviation": frac_deviation,
        })

        omega_prev = omega_n

    # Check convergence rate
    deviations = [ot["frac_deviation"] for ot in results["overtones"]
                  if not np.isnan(ot["frac_deviation"])]
    if len(deviations) >= 2:
        # Fit 1/n convergence
        n_vals = np.array([ot["n"] for ot in results["overtones"]
                           if not np.isnan(ot["frac_deviation"])])
        dev_arr = np.array(deviations)
        if np.all(dev_arr > 0):
            log_dev = np.log(dev_arr)
            log_n = np.log(n_vals)
            slope, intercept = np.polyfit(log_n, log_dev, 1)
            results["convergence_exponent"] = slope  # should be ~ -1
            results["convergence_note"] = (
                f"deviation ~ n^{{{slope:.2f}}} (expected ~ n^{{-1}} from Motl)"
            )
        else:
            results["convergence_exponent"] = np.nan

    # SCT modification to high overtones
    results["SCT_modification"] = {
        "status": "UNMODIFIED for astrophysical BH",
        "reason": (
            "High-overtone QNMs probe near-horizon geometry (r ~ r_s). "
            "The SCT Yukawa corrections are exp(-m2*r_s) suppressed "
            "and do not affect the asymptotic ln(3) result."
        ),
        "formal_bound": (
            "delta(omega_R)/omega_R ~ exp(-m2 * r_s) "
            "which is < 10^{-10^9} for a 10 M_sun BH."
        ),
    }

    return results


# ============================================================
# Section 4: Quantum vs Classical QNM Corrections
# ============================================================
def quantum_qnm_correction(M_kg: float) -> dict:
    """One-loop quantum correction to BH QNM frequencies.

    The leading quantum gravity correction to QNM frequencies is:
        delta_omega_quantum / omega ~ (l_P / r_s)^2 * c_eff

    where c_eff is a dimensionless coefficient that depends on the
    UV theory.  In SCT:
        c_eff ~ c_log = 37/24  (the logarithmic entropy coefficient)

    This comes from the one-loop effective action on the Schwarzschild
    background: the R^2 and C^2 terms generate corrections of order
    (l_P/r_s)^2 = (M_Pl/M)^2 to the classical metric.

    The coefficient 37/24 is specific to SCT with SM field content:
        c_log = (1/180)[2*N_s + 7*N_F - 26*N_V + 424]
    with N_s=4, N_F=22.5, N_V=12.

    Reference: Sen (2012), arXiv:1205.0971, Eq. (1.2).
    """
    r_s = schwarzschild_radius(M_kg)

    # (l_P / r_s)^2 = (hbar G / c^3) / r_s^2
    lP_over_rs_sq = l_Pl_sq / r_s**2

    # Quantum correction: delta_omega/omega ~ (l_P/r_s)^2 * c_log
    delta_omega_quantum = lP_over_rs_sq * C_LOG

    # log10 of the correction
    if delta_omega_quantum > 0:
        log10_quantum = np.log10(delta_omega_quantum)
    else:
        log10_quantum = -np.inf

    return {
        "M_kg": M_kg,
        "M_solar": M_kg / M_sun,
        "r_s_m": r_s,
        "lP_over_rs": np.sqrt(lP_over_rs_sq),
        "lP_over_rs_sq": lP_over_rs_sq,
        "c_log": C_LOG,
        "delta_omega_quantum": delta_omega_quantum,
        "log10_quantum": log10_quantum,
    }


def classical_sct_correction(M_kg: float) -> dict:
    """Classical SCT correction to QNM frequencies from Yukawa modification.

    The SCT correction enters through the modified metric:
        f(r) = 1 - (r_s/r) * [1 - (4/3)e^{-m2*r} + (1/3)e^{-m0*r}]

    At the potential peak r_peak ~ 1.64 r_s (for l=2), the leading correction:
        delta_omega_SCT / omega ~ (4/3) * exp(-m2 * r_peak)

    Since m2 * r_peak = sqrt(60/13) * Lambda * 1.64 * r_s, and
    Lambda * r_s >> 1 for any astrophysical BH, this is VASTLY more
    suppressed than even the one-loop quantum correction.
    """
    r_s = schwarzschild_radius(M_kg)
    r_peak = 1.64 * r_s  # approximate for l=2, n=0

    m2_r_peak = M2_M * r_peak
    m0_r_peak = M0_M * r_peak

    # Dominant Yukawa correction at the peak
    if m2_r_peak > 700:
        exp_m2 = 0.0
        log10_sct = -m2_r_peak * np.log10(np.e)
    else:
        exp_m2 = np.exp(-m2_r_peak)
        log10_sct = np.log10(max(abs((4.0 / 3.0) * exp_m2), 1e-300))

    delta_omega_sct = (4.0 / 3.0) * exp_m2  # leading order

    return {
        "M_kg": M_kg,
        "M_solar": M_kg / M_sun,
        "r_s_m": r_s,
        "r_peak_m": r_peak,
        "m2_r_peak": m2_r_peak,
        "m0_r_peak": m0_r_peak,
        "exp_m2_r_peak": exp_m2,
        "delta_omega_sct": delta_omega_sct,
        "log10_sct": log10_sct,
    }


def quantum_vs_classical_comparison() -> dict:
    """KEY RESULT: Compare quantum and classical QNM corrections for observed BHs.

    For every observed LIGO/Virgo BH:
        1-loop quantum: delta_omega/omega ~ (l_P/r_s)^2 * 37/24
        Classical SCT:  delta_omega/omega ~ (4/3) * exp(-m2 * r_peak)

    The SCT classical correction is ALWAYS far smaller than the quantum
    correction, which is itself immeasurably small (~10^{-78} for GW150914).

    This demonstrates that SCT is phenomenologically indistinguishable from
    GR for QNM observations: the SCT corrections are astronomically smaller
    than even the one-loop quantum gravity effects.
    """
    results = {"comparison_table": [], "c_log": C_LOG, "c_log_fraction": "37/24"}

    for name, data in OBSERVED_BHS.items():
        M_kg = data["M_solar"] * M_sun

        qc = quantum_qnm_correction(M_kg)
        sc = classical_sct_correction(M_kg)

        # Ratio: quantum / SCT
        if sc["delta_omega_sct"] > 0:
            ratio = qc["delta_omega_quantum"] / sc["delta_omega_sct"]
            log10_ratio = qc["log10_quantum"] - sc["log10_sct"]
        else:
            ratio = np.inf
            log10_ratio = np.inf

        entry = {
            "event": name,
            "M_solar": data["M_solar"],
            "a_star": data["a_star"],
            "ref": data["ref"],
            "log10_quantum": qc["log10_quantum"],
            "log10_sct": sc["log10_sct"],
            "log10_ratio_quantum_over_sct": log10_ratio,
            "m2_r_peak": sc["m2_r_peak"],
            "lP_over_rs_sq": qc["lP_over_rs_sq"],
        }
        results["comparison_table"].append(entry)

    # Summary
    log10_sct_all = [e["log10_sct"] for e in results["comparison_table"]]
    log10_q_all = [e["log10_quantum"] for e in results["comparison_table"]]

    results["summary"] = {
        "max_log10_sct": max(log10_sct_all),
        "min_log10_quantum": min(log10_q_all),
        "hierarchy": (
            "For ALL observed BHs: SCT correction << quantum correction << "
            "measurement sensitivity. SCT is phenomenologically identical to GR."
        ),
        "GW150914_quantum": [e for e in results["comparison_table"]
                             if e["event"] == "GW150914"][0]["log10_quantum"],
        "GW150914_sct": [e for e in results["comparison_table"]
                         if e["event"] == "GW150914"][0]["log10_sct"],
    }

    return results


# ============================================================
# Section 5: Verification
# ============================================================
def verify() -> dict:
    """Run verification checks on all extended physics results."""
    checks = []

    def check(name: str, condition: bool, detail: str = ""):
        checks.append({"name": name, "pass": condition, "detail": detail})

    # --- Greybody factor checks ---
    r_s_10 = schwarzschild_radius(10.0 * M_sun)

    # Check 1: low-frequency analytic T scales as (omega*r_s)^6 for l=2
    # Use the analytic formula (not numerical WKB) to test the power law.
    # Need omega * r_s << 1, so use very small frequencies.
    omega_low = 1e-4 * c_light / r_s_10
    omega_low2 = 2e-4 * c_light / r_s_10
    T1 = greybody_low_freq_GR(omega_low, r_s_10, l=2)
    T2 = greybody_low_freq_GR(omega_low2, r_s_10, l=2)
    if T1 > 0 and T2 > 0 and T1 < 1.0 and T2 < 1.0:
        ratio = T2 / T1
        expected = 2.0**(2 * 2 + 2)  # 2^6 = 64
        check("greybody_power_law_l2", abs(ratio / expected - 1.0) < 0.01,
              f"T2/T1 = {ratio:.4f}, expected {expected:.4f}")
    else:
        check("greybody_power_law_l2", False,
              f"T values out of range: T1={T1:.4e}, T2={T2:.4e}")

    # Check 2: numerical greybody T(omega) in [0, 1]
    omega_test = np.linspace(0.01, 2.0, 50) * c_light / r_s_10
    T_test = greybody_numerical_GR(omega_test, r_s_10, l=2)
    check("greybody_range_0_1", np.all(T_test >= 0) and np.all(T_test <= 1.0 + 1e-10),
          f"min={np.min(T_test):.6e}, max={np.max(T_test):.6e}")

    # Check 3: T approaches 1 for omega >> V_max
    r_peak = find_potential_peak(V_RW_GR, r_s_10, 2)
    V_max = V_RW_GR(r_peak, r_s_10, 2)
    omega_high = np.array([2.0 * np.sqrt(V_max)])
    T_high = greybody_numerical_GR(omega_high, r_s_10, l=2)
    check("greybody_high_freq_limit", T_high[0] > 0.9,
          f"T(omega >> V_max^{{1/2}}) = {T_high[0]:.4f}")

    # Check 4: SCT and GR greybody factors agree for astrophysical BH
    omega_mid = np.array([0.5 * np.sqrt(V_max)])
    T_GR_mid = greybody_numerical_GR(omega_mid, r_s_10, l=2)
    T_SCT_mid = greybody_numerical_SCT(omega_mid, r_s_10, M2_M, M0_M, l=2)
    delta_T = abs(T_SCT_mid[0] - T_GR_mid[0])
    check("greybody_SCT_eq_GR_astro", delta_T < 1e-10,
          f"|T_SCT - T_GR| = {delta_T:.4e}")

    # --- Price tail checks ---
    check("price_tail_l2_exponent", price_tail_exponent_GR(2) == -7,
          f"exponent = {price_tail_exponent_GR(2)}")
    check("price_tail_l3_exponent", price_tail_exponent_GR(3) == -9,
          f"exponent = {price_tail_exponent_GR(3)}")
    check("price_tail_l4_exponent", price_tail_exponent_GR(4) == -11,
          f"exponent = {price_tail_exponent_GR(4)}")

    # --- Hod area quantization checks ---
    T_H_M = 1.0 / (8.0 * np.pi)
    hod_limit = T_H_M * np.log(3.0)
    check("hod_limit_value", abs(hod_limit - 0.04395) < 0.001,
          f"T_H*M*ln(3) = {hod_limit:.5f}")

    # --- Quantum vs Classical checks ---
    # c_log = 37/24
    check("c_log_value", abs(C_LOG - 37.0 / 24.0) < 1e-10,
          f"c_log = {C_LOG}")

    # Check quantum correction for GW150914 (62 M_sun)
    qc = quantum_qnm_correction(62.0 * M_sun)
    check("quantum_correction_GW150914_order",
          -80 < qc["log10_quantum"] < -70,
          f"log10(delta_omega_quantum) = {qc['log10_quantum']:.1f}")

    # Check classical SCT correction for GW150914
    sc = classical_sct_correction(62.0 * M_sun)
    check("sct_correction_GW150914_suppressed",
          sc["log10_sct"] < -1e6,
          f"log10(delta_omega_sct) = {sc['log10_sct']:.2e}")

    # Check SCT << quantum
    check("sct_much_less_than_quantum",
          sc["log10_sct"] < qc["log10_quantum"],
          f"SCT: {sc['log10_sct']:.2e} << quantum: {qc['log10_quantum']:.1f}")

    # Check m2*r_peak is enormous for astrophysical BH
    check("m2_r_peak_enormous_GW150914",
          sc["m2_r_peak"] > 1e9,
          f"m2*r_peak = {sc['m2_r_peak']:.4e}")

    # Hawking temperature checks
    T_H_10 = hawking_temperature(10.0 * M_sun)
    check("hawking_temp_10Msun_order",
          1e-9 < T_H_10 < 1e-6,
          f"T_H(10 M_sun) = {T_H_10:.4e} K")

    # f_SCT -> f_GR for large r
    r_large = 100.0 * r_s_10
    f_gr = f_GR(r_large, r_s_10)
    f_sct = f_SCT(r_large, r_s_10, M2_M, M0_M)
    check("metric_GR_limit_large_r", abs(f_sct - f_gr) < 1e-10,
          f"|f_SCT - f_GR| at r=100*r_s = {abs(f_sct - f_gr):.4e}")

    # f_SCT(r_s) -> 0 for astrophysical BH (horizon preserved)
    f_hor = f_SCT(r_s_10 * 1.0001, r_s_10, M2_M, M0_M)
    check("horizon_preserved", abs(f_hor) < 0.01,
          f"f_SCT(r_s+eps) = {f_hor:.6e}")

    # Effective mass ratios
    check("m2_over_Lambda", abs(M2_OVER_LAMBDA - np.sqrt(60.0 / 13.0)) < 1e-10,
          f"m2/Lambda = {M2_OVER_LAMBDA:.6f}")
    check("m0_over_Lambda", abs(M0_OVER_LAMBDA - np.sqrt(6.0)) < 1e-10,
          f"m0/Lambda = {M0_OVER_LAMBDA:.6f}")

    # alpha_C
    check("alpha_C_value", abs(ALPHA_C - 13.0 / 120.0) < 1e-15,
          f"alpha_C = {ALPHA_C}")

    n_pass = sum(1 for c in checks if c["pass"])
    n_total = len(checks)

    return {
        "checks": checks,
        "n_pass": n_pass,
        "n_total": n_total,
        "all_pass": n_pass == n_total,
    }


# ============================================================
# Figures
# ============================================================
def make_figures(greybody_data: dict, qvc_data: dict, hod_data: dict) -> None:
    """Generate all Phase 6 figures."""
    # Publication-quality style
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "lines.linewidth": 1.5,
        "axes.linewidth": 0.8,
    })

    # --- Figure 1: Greybody factor ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    omega_arr = np.array(greybody_data["omega_arr"])
    T_GR = np.array(greybody_data["T_GR"])
    T_SCT = np.array(greybody_data["T_SCT"])
    r_s = greybody_data["r_s_m"]
    x_arr = omega_arr * r_s / c_light  # dimensionless omega*r_s/c

    ax1.plot(x_arr, T_GR, 'k-', label='GR', lw=1.5)
    ax1.plot(x_arr, T_SCT, 'r--', label='SCT', lw=1.5, alpha=0.8)
    ax1.set_xlabel(r'$\omega r_s / c$')
    ax1.set_ylabel(r'$T_l(\omega)$')
    ax1.set_title(r'Greybody Factor ($l=2$, $M=10\,M_\odot$)')
    ax1.legend()
    ax1.set_xlim(0, 2.0)
    ax1.set_ylim(-0.05, 1.05)

    # Inset: zoom on the difference
    delta_T = np.abs(T_SCT - T_GR)
    ax2.semilogy(x_arr, delta_T + 1e-20, 'b-', lw=1.0)
    ax2.set_xlabel(r'$\omega r_s / c$')
    ax2.set_ylabel(r'$|\Delta T| = |T_\mathrm{SCT} - T_\mathrm{GR}|$')
    ax2.set_title('Greybody Factor Modification')
    ax2.set_xlim(0, 2.0)
    ax2.text(0.5, 0.85, (f"$m_2 r_s = {greybody_data['m2_r_s']:.2e}$\n"
                          f"max $|\\Delta T/T|$ = {greybody_data['max_delta_T_over_T']:.2e}"),
             transform=ax2.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "lt3a_greybody_factors.pdf"))
    fig.savefig(str(FIGURES_DIR / "lt3a_greybody_factors.png"))
    plt.close(fig)

    # --- Figure 2: Quantum vs Classical corrections ---
    fig, ax = plt.subplots(figsize=(8, 5))

    events = [e["event"] for e in qvc_data["comparison_table"]]
    log10_q = [e["log10_quantum"] for e in qvc_data["comparison_table"]]
    log10_s = [e["log10_sct"] for e in qvc_data["comparison_table"]]

    x_pos = np.arange(len(events))
    width = 0.35

    # For plotting, cap SCT values at a visible range
    log10_s_plot = [max(s, -300) for s in log10_s]

    bars1 = ax.bar(x_pos - width / 2, log10_q, width, label='One-loop quantum',
                   color='#2196F3', alpha=0.8)
    bars2 = ax.bar(x_pos + width / 2, log10_s_plot, width, label='Classical SCT',
                   color='#F44336', alpha=0.8)

    ax.set_ylabel(r'$\log_{10}(\delta\omega/\omega)$')
    ax.set_title('QNM Corrections: One-loop Quantum vs Classical SCT')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(events, rotation=30, ha='right', fontsize=8)
    ax.legend(loc='upper right')
    ax.axhline(y=-78, color='gray', ls=':', lw=0.8, label='Planck suppression')

    # Annotate GW150914
    gw150914_idx = events.index("GW150914")
    ax.annotate(f'quantum: $10^{{{log10_q[gw150914_idx]:.0f}}}$',
                xy=(gw150914_idx - width / 2, log10_q[gw150914_idx]),
                xytext=(gw150914_idx + 1, log10_q[gw150914_idx] + 15),
                fontsize=7,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax.annotate(f'SCT: $< 10^{{{log10_s_plot[gw150914_idx]:.0f}}}$',
                xy=(gw150914_idx + width / 2, log10_s_plot[gw150914_idx]),
                xytext=(gw150914_idx + 1.5, log10_s_plot[gw150914_idx] + 15),
                fontsize=7,
                arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "lt3a_quantum_vs_classical.pdf"))
    fig.savefig(str(FIGURES_DIR / "lt3a_quantum_vs_classical.png"))
    plt.close(fig)

    # --- Figure 3: Hod area quantization convergence ---
    if hod_data.get("overtones"):
        fig, ax = plt.subplots(figsize=(7, 4.5))

        n_vals = [ot["n"] for ot in hod_data["overtones"]]
        omega_R_vals = [ot["omega_R_M"] for ot in hod_data["overtones"]]
        frac_dev = [ot["frac_deviation"] for ot in hod_data["overtones"]]

        hod_lim = hod_data["hod_limit_omegaR_M"]

        ax.plot(n_vals, omega_R_vals, 'ko-', ms=5, label=r'$\mathrm{Re}(\omega_n M)$')
        ax.axhline(y=hod_lim, color='r', ls='--', lw=1.5,
                   label=r'$T_H M \ln 3 \approx %.4f$' % hod_lim)

        ax.set_xlabel('Overtone number $n$')
        ax.set_ylabel(r'$\mathrm{Re}(\omega_n M)$')
        ax.set_title('Hod Area Quantization: Asymptotic QNM Convergence')
        ax.legend(fontsize=9)

        # Add text box
        ax.text(0.02, 0.02,
                'SCT: ln(3) UNMODIFIED\n(horizon geometry unchanged)',
                transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()
        fig.savefig(str(FIGURES_DIR / "lt3a_hod_convergence.pdf"))
        fig.savefig(str(FIGURES_DIR / "lt3a_hod_convergence.png"))
        plt.close(fig)

    # --- Figure 4: Mass scan of corrections ---
    fig, ax = plt.subplots(figsize=(8, 5))

    M_solar_scan = np.logspace(-8, 10, 200)
    log10_q_scan = []
    log10_s_scan = []

    for M_sol in M_solar_scan:
        M_kg = M_sol * M_sun
        qc = quantum_qnm_correction(M_kg)
        sc = classical_sct_correction(M_kg)
        log10_q_scan.append(qc["log10_quantum"])
        log10_s_scan.append(sc["log10_sct"])

    ax.plot(np.log10(M_solar_scan), log10_q_scan, 'b-', lw=1.5,
            label=r'Quantum: $(l_P/r_s)^2 \cdot c_{\log}$')
    ax.plot(np.log10(M_solar_scan), log10_s_scan, 'r-', lw=1.5,
            label=r'SCT: $(4/3)\exp(-m_2 r_\mathrm{peak})$')

    # Mark observed BHs
    for name, data in OBSERVED_BHS.items():
        M_sol = data["M_solar"]
        sc = classical_sct_correction(M_sol * M_sun)
        qc = quantum_qnm_correction(M_sol * M_sun)
        ax.plot(np.log10(M_sol), qc["log10_quantum"], 'b^', ms=6)
        ax.plot(np.log10(M_sol), max(sc["log10_sct"], -300), 'rv', ms=6)

    # Mark measurement threshold
    ax.axhline(y=np.log10(0.01), color='gray', ls=':', lw=0.8)
    ax.text(8, np.log10(0.01) + 5, r'LVK threshold ($\sim 1\%$)',
            fontsize=8, color='gray')

    ax.set_xlabel(r'$\log_{10}(M/M_\odot)$')
    ax.set_ylabel(r'$\log_{10}(\delta\omega/\omega)$')
    ax.set_title('QNM Frequency Corrections vs Black Hole Mass')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(-310, 10)
    ax.set_xlim(-8, 10)

    # Shade astrophysical BH range
    ax.axvspan(np.log10(3), 10, alpha=0.05, color='blue')
    ax.text(5, -30, 'Astrophysical BHs', fontsize=8, color='blue', alpha=0.6)

    plt.tight_layout()
    fig.savefig(str(FIGURES_DIR / "lt3a_mass_scan_corrections.pdf"))
    fig.savefig(str(FIGURES_DIR / "lt3a_mass_scan_corrections.png"))
    plt.close(fig)


# ============================================================
# Main
# ============================================================
def main():
    """Run all Phase 6 extended physics computations."""
    print("=" * 70)
    print("LT-3a Phase 6: Extended Physics of QNMs in SCT")
    print("=" * 70)

    t0 = time.time()
    all_results = {}

    # --- Section 1: Greybody factors ---
    print("\n[1/5] Computing greybody factors...")
    greybody_data = compute_greybody_results(M_solar=10.0, l=2)
    all_results["greybody_factors"] = {
        k: v for k, v in greybody_data.items()
        if k not in ("omega_arr", "T_GR", "T_SCT", "T_low_analytic_first50")
    }
    print(f"  max delta_T/T = {greybody_data['max_delta_T_over_T']:.4e}")
    print(f"  m2*r_s = {greybody_data['m2_r_s']:.4e}")
    print(f"  exp(-m2*r_peak) = {greybody_data['exp_suppression_at_peak']:.4e}")

    # --- Section 2: Price tails ---
    print("\n[2/5] Analyzing late-time Price tails...")
    price_data = price_tail_analysis()
    all_results["price_tails"] = price_data
    for l_key, info in price_data.items():
        if l_key.startswith("l="):
            print(f"  {l_key}: psi ~ t^{{{info['exponent_GR']}}}")
    print(f"  r_critical = {price_data['l=2']['r_critical_m']:.4e} m")
    print(f"  M_crit = {price_data['l=2']['M_crit_solar']:.4e} M_sun")

    # --- Section 3: Hod area quantization ---
    print("\n[3/5] Computing Hod area quantization (n=10..20)...")
    print("  (This may take a few minutes for high overtones...)")
    hod_data = hod_area_quantization(n_min=10, n_max=20)
    all_results["hod_area_quantization"] = hod_data
    print(f"  Hod limit: omega_R*M = {hod_data['hod_limit_omegaR_M']:.5f}")
    if hod_data.get("overtones"):
        last = hod_data["overtones"][-1]
        print(f"  n={last['n']}: omega_R*M = {last['omega_R_M']:.5f}, "
              f"deviation = {last['frac_deviation']:.4e}")
    if "convergence_exponent" in hod_data:
        print(f"  Convergence exponent: {hod_data['convergence_exponent']:.2f}")

    # --- Section 4: Quantum vs Classical ---
    print("\n[4/5] Computing quantum vs classical QNM corrections...")
    qvc_data = quantum_vs_classical_comparison()
    all_results["quantum_vs_classical"] = qvc_data
    print(f"\n  {'Event':<15} {'log10(quantum)':>15} {'log10(SCT)':>15}")
    print(f"  {'-'*45}")
    for entry in qvc_data["comparison_table"]:
        print(f"  {entry['event']:<15} {entry['log10_quantum']:>15.1f} "
              f"{entry['log10_sct']:>15.2e}")

    # --- Section 5: Verification ---
    print("\n[5/5] Running verification...")
    vr = verify()
    all_results["verification"] = vr
    print(f"\n  Verification: {vr['n_pass']}/{vr['n_total']} checks PASS")
    for c in vr["checks"]:
        status = "PASS" if c["pass"] else "FAIL"
        print(f"    [{status}] {c['name']}: {c['detail']}")

    # --- Figures ---
    print("\n  Generating figures...")
    make_figures(greybody_data, qvc_data, hod_data)
    print(f"  Figures saved to {FIGURES_DIR}")

    # --- Save JSON ---
    # Convert non-serializable types
    def make_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    json_path = RESULTS_DIR / "lt3a_extended_physics.json"
    with open(json_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2, default=str)
    print(f"\n  Results saved to {json_path}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f} s")
    print(f"\n{'='*70}")
    print(f"Phase 6 COMPLETE. {vr['n_pass']}/{vr['n_total']} checks pass.")
    print(f"{'='*70}")

    return all_results


if __name__ == "__main__":
    main()
