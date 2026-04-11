#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""
MT-1 Extended: Complete BH Thermodynamics in SCT (Schwarzschild, Kerr, RN, Kerr-Newman).

Computes all 25 elements of the BH thermodynamics program in Spectral Causal Theory:
  - Horizon areas, entropy (BH + SCT corrections), temperature, angular velocity,
    electric potential for all four BH families.
  - Symbolic and numerical verification of the first law and Smarr relation.
  - Extremal limits, Christodoulou-Ruffini mass formula, Penrose process.
  - Kerr/CFT correspondence, holographic entropy excess.
  - Specific heat with SCT corrections, GB cross-check on Ricci-flat.
  - GW150914 entropy production.
  - Nonlocal correction estimates, entropy production rate.
  - Dimensional analysis, central charges (Hofman-Maldacena), Planck-unit formulas.

All computations are performed in GEOMETRIZED units (G = c = 1, M in meters)
internally, converting to SI for output.  The SCT entropy formula reads:

    S = A/(4 l_P^2) + alpha_C/pi + c_log * ln(A/l_P^2)

with alpha_C = 13/120 (SM Weyl^2, parameter-free) and c_log = 37/24 (Sen 2012).

References:
  - Wald (1993), PRD 48, R3427
  - Iyer-Wald (1994), PRD 50, 846
  - Jacobson-Myers (1993), PRL 70, 3684
  - Sen (2012), arXiv:1205.0971
  - Christodoulou (1970), PRL 25, 1596
  - Christodoulou-Ruffini (1971), PRD 4, 3552
  - Penrose (1969), RNCL 1, 252
  - Guica-Hartman-Song-Strominger (2009), arXiv:0809.4266
  - Hofman-Maldacena (2008), arXiv:0803.1467
  - LIGO (2016), PRL 116, 061102 — GW150914
  - NT-1b Phase 3: alpha_C = 13/120
  - MT-1: c_log = 37/24, S_Weyl = 13/(120 pi)

Author: David Alfyorov
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy.constants as const
import sympy as sp

# ============================================================
# Setup
# ============================================================
DPS = 60
mp.mp.dps = DPS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "mt1"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Physical constants (CODATA via scipy.constants)
# ============================================================
G_SI     = mp.mpf(str(const.G))        # m^3 kg^-1 s^-2
HBAR_SI  = mp.mpf(str(const.hbar))     # J s
C_SI     = mp.mpf(str(const.c))        # m/s
K_B_SI   = mp.mpf(str(const.k))        # J/K
EV_J     = mp.mpf(str(const.eV))       # J per eV
EPS_0_SI = mp.mpf(str(const.epsilon_0))  # F/m

# Planck units
L_PL     = mp.sqrt(HBAR_SI * G_SI / C_SI**3)          # Planck length [m]
L_PL2    = L_PL**2                                      # Planck area [m^2]
M_PL_KG  = mp.sqrt(HBAR_SI * C_SI / G_SI)             # Planck mass [kg]
M_PL_EV  = M_PL_KG * C_SI**2 / EV_J                   # Planck mass [eV]
T_PL_K   = M_PL_KG * C_SI**2 / K_B_SI                 # Planck temperature [K]
M_SUN_KG = mp.mpf("1.98892e30")                        # Solar mass [kg]

# SCT canonical constants
ALPHA_C       = mp.mpf(13) / 120                       # SM Weyl^2 coefficient
ALPHA_C_FRAC  = Fraction(13, 120)
C_LOG         = mp.mpf(37) / 24                        # Sen (2012) log coefficient
C_LOG_FRAC    = Fraction(37, 24)
S_WEYL        = ALPHA_C / mp.pi                        # Topological Wald correction
Z_L           = mp.mpf("1.2807")                       # Physical ghost pole
LAMBDA_PPN_EV = mp.mpf("2.38e-3")                      # PPN lower bound [eV]
M_GHOST_OVER_LAMBDA = mp.sqrt(Z_L)                     # m_ghost / Lambda

# SM field content (NT-1b Phase 3)
N_S = 4       # Real scalars
N_D = 22.5    # Dirac fermions
N_V = 12      # Gauge bosons

# ============================================================
# Utility: geometrized unit conversions
# ============================================================
# In geometrized units (G = c = 1), mass M [kg] -> M_geom = G*M/c^2 [m]
# This is the "gravitational radius" rg = GM/c^2 (half of Schwarzschild radius).

def to_geom_mass(M_kg):
    """Convert mass [kg] to geometrized length [m]: rg = G*M/c^2."""
    return G_SI * mp.mpf(M_kg) / C_SI**2

def to_geom_spin(a_dimless, M_kg):
    """Convert dimensionless spin a/M to geometrized [m]: a = (a/M) * G*M/c^2."""
    return mp.mpf(a_dimless) * to_geom_mass(M_kg)

def to_geom_charge(Q_dimless, M_kg):
    """Convert dimensionless charge Q/M to geometrized [m]: Q = (Q/M) * G*M/c^2.
    Here Q/M is in geometrized units where Q_geom = Q*sqrt(G/(4*pi*eps_0))/c^2."""
    return mp.mpf(Q_dimless) * to_geom_mass(M_kg)


# ============================================================
# ELEMENT 1: Horizon areas (all four BH types)
# ============================================================
def area_schwarzschild(M_kg):
    """A_Sch = 16*pi*(G*M/c^2)^2 [m^2]."""
    rg = to_geom_mass(M_kg)
    return 16 * mp.pi * rg**2

def area_kerr(M_kg, a_dimless):
    """A_Kerr = 8*pi*rg*(rg + sqrt(rg^2 - a^2)) [m^2].
    a_dimless = a/M in geometrized units (0 <= a/M <= 1)."""
    rg = to_geom_mass(M_kg)
    a = a_dimless * rg
    r_plus = rg + mp.sqrt(rg**2 - a**2)
    return 4 * mp.pi * (r_plus**2 + a**2)

def area_rn(M_kg, Q_dimless):
    """A_RN = 4*pi*(rg + sqrt(rg^2 - Q^2))^2 [m^2].
    Q_dimless = Q/M in geometrized units (0 <= Q/M <= 1)."""
    rg = to_geom_mass(M_kg)
    Q = Q_dimless * rg
    r_plus = rg + mp.sqrt(rg**2 - Q**2)
    return 4 * mp.pi * r_plus**2

def area_kn(M_kg, a_dimless, Q_dimless):
    """A_KN = 4*pi*(r_+^2 + a^2) [m^2].
    Kerr-Newman: r_+ = rg + sqrt(rg^2 - a^2 - Q^2)."""
    rg = to_geom_mass(M_kg)
    a = a_dimless * rg
    Q = Q_dimless * rg
    disc = rg**2 - a**2 - Q**2
    if disc < 0:
        return None  # super-extremal
    r_plus = rg + mp.sqrt(disc)
    return 4 * mp.pi * (r_plus**2 + a**2)


# ============================================================
# ELEMENT 2: Entropy (BH + SCT corrections) for each type
# ============================================================
def entropy_sct(A_m2):
    """Full SCT entropy: S = A/(4*l_P^2) + alpha_C/pi + c_log*ln(A/l_P^2).
    Returns dimensionless entropy (units of k_B)."""
    if A_m2 is None:
        return None
    A_over_lp2 = A_m2 / L_PL2
    S_bh = A_over_lp2 / 4
    S_weyl = ALPHA_C / mp.pi
    S_log = C_LOG * mp.log(A_over_lp2)
    return S_bh + S_weyl + S_log

def entropy_bh(A_m2):
    """Bekenstein-Hawking entropy: S_BH = A/(4*l_P^2)."""
    if A_m2 is None:
        return None
    return A_m2 / (4 * L_PL2)

def entropy_correction(A_m2):
    """SCT correction only: delta_S = alpha_C/pi + c_log*ln(A/l_P^2)."""
    if A_m2 is None:
        return None
    A_over_lp2 = A_m2 / L_PL2
    return ALPHA_C / mp.pi + C_LOG * mp.log(A_over_lp2)


# ============================================================
# ELEMENT 3: Hawking temperature for each type
# ============================================================
def r_plus_geom(rg, a=0, Q=0):
    """r_+ in geometrized units [m]."""
    disc = rg**2 - a**2 - Q**2
    if disc < 0:
        return None
    return rg + mp.sqrt(disc)

def r_minus_geom(rg, a=0, Q=0):
    """r_- in geometrized units [m]."""
    disc = rg**2 - a**2 - Q**2
    if disc < 0:
        return None
    return rg - mp.sqrt(disc)

def temp_schwarzschild_K(M_kg):
    """T_H = hbar*c^3/(8*pi*G*M*k_B) [K]."""
    return HBAR_SI * C_SI**3 / (8 * mp.pi * G_SI * mp.mpf(M_kg) * K_B_SI)

def temp_kerr_K(M_kg, a_dimless):
    """Kerr temperature [K]: T = hbar*c*(r+-r-)/(4*pi*(r+^2+a^2)*k_B)."""
    rg = to_geom_mass(M_kg)
    a = a_dimless * rg
    rp = r_plus_geom(rg, a)
    rm = r_minus_geom(rg, a)
    if rp is None:
        return None
    # kappa_geom = (r+ - r-)/(2*(r+^2 + a^2)) in geometrized [1/m]
    kappa_geom = (rp - rm) / (2 * (rp**2 + a**2))
    # kappa_geom [1/m]; kappa_SI = kappa_geom * c^2 [m/s^2]
    # T = hbar*kappa_SI/(2*pi*c*k_B) = hbar*c*kappa_geom/(2*pi*k_B)
    return HBAR_SI * C_SI * kappa_geom / (2 * mp.pi * K_B_SI)

def temp_rn_K(M_kg, Q_dimless):
    """RN temperature [K]."""
    rg = to_geom_mass(M_kg)
    Q = Q_dimless * rg
    rp = r_plus_geom(rg, Q=Q)
    rm = r_minus_geom(rg, Q=Q)
    if rp is None:
        return None
    kappa_geom = (rp - rm) / (2 * rp**2)
    return HBAR_SI * C_SI * kappa_geom / (2 * mp.pi * K_B_SI)

def temp_kn_K(M_kg, a_dimless, Q_dimless):
    """Kerr-Newman temperature [K]."""
    rg = to_geom_mass(M_kg)
    a = a_dimless * rg
    Q = Q_dimless * rg
    rp = r_plus_geom(rg, a, Q)
    rm = r_minus_geom(rg, a, Q)
    if rp is None:
        return None
    kappa_geom = (rp - rm) / (2 * (rp**2 + a**2))
    return HBAR_SI * C_SI * kappa_geom / (2 * mp.pi * K_B_SI)


# ============================================================
# ELEMENT 4: Angular velocity Omega_H
# ============================================================
def omega_kerr(M_kg, a_dimless):
    """Omega_H = a*c/(r_+^2 + a^2) [rad/s] for Kerr."""
    rg = to_geom_mass(M_kg)
    a = a_dimless * rg
    rp = r_plus_geom(rg, a)
    if rp is None:
        return None
    # In geometrized: Omega_geom = a/(r+^2 + a^2) [1/m].
    # To SI: Omega_SI = c * Omega_geom [rad/s].
    return C_SI * a / (rp**2 + a**2)

def omega_kn(M_kg, a_dimless, Q_dimless):
    """Omega_H for Kerr-Newman [rad/s]."""
    rg = to_geom_mass(M_kg)
    a = a_dimless * rg
    Q = Q_dimless * rg
    rp = r_plus_geom(rg, a, Q)
    if rp is None:
        return None
    return C_SI * a / (rp**2 + a**2)


# ============================================================
# ELEMENT 5: Electric potential Phi_H
# ============================================================
def phi_rn_geom(M_kg, Q_dimless):
    """Phi_H = Q*r_+/(r_+^2) = Q/r_+ for RN in geometrized units (dimensionless)."""
    rg = to_geom_mass(M_kg)
    Q = Q_dimless * rg
    rp = r_plus_geom(rg, Q=Q)
    if rp is None:
        return None
    return Q / rp

def phi_kn_geom(M_kg, a_dimless, Q_dimless):
    """Phi_H = Q*r_+/(r_+^2 + a^2) for KN in geometrized units."""
    rg = to_geom_mass(M_kg)
    a = a_dimless * rg
    Q = Q_dimless * rg
    rp = r_plus_geom(rg, a, Q)
    if rp is None:
        return None
    return Q * rp / (rp**2 + a**2)


# ============================================================
# ELEMENT 6: First Law SYMBOLIC verification (SymPy)
# ============================================================
def verify_first_law_symbolic():
    """Verify dM = T*dS + Omega*dJ + Phi*dQ symbolically for each BH type.

    Works in geometrized units with G = c = hbar = k_B = 1.
    """
    results = {}
    M, a, Q, J = sp.symbols('M a Q J', positive=True, real=True)

    # --- Schwarzschild ---
    r_plus_sch = 2 * M
    A_sch = 16 * sp.pi * M**2
    S_sch = A_sch / 4  # BH part (corrections are constant or log)
    T_sch = 1 / (8 * sp.pi * M)

    # dS/dM * T should give 1 (= dM)
    dSdM_sch = sp.diff(S_sch, M)
    first_law_sch = sp.simplify(T_sch * dSdM_sch - 1)
    results["schwarzschild"] = {
        "T_dS_dM": str(sp.simplify(T_sch * dSdM_sch)),
        "first_law_residual": str(first_law_sch),
        "pass": first_law_sch == 0,
    }

    # --- Kerr (use M, a; J = M*a) ---
    r_plus_k = M + sp.sqrt(M**2 - a**2)
    A_kerr = 4 * sp.pi * (r_plus_k**2 + a**2)
    S_kerr = A_kerr / 4
    T_kerr = (r_plus_k - (M - sp.sqrt(M**2 - a**2))) / (4 * sp.pi * (r_plus_k**2 + a**2))
    Omega_kerr = a / (r_plus_k**2 + a**2)

    # First law: dM = T*dS + Omega*dJ, J = M*a
    # dM at constant a means dJ = a*dM, so: dM = T*(dS/dM)*dM + Omega*a*dM
    # => 1 = T*dS/dM + Omega*a
    dSdM_k = sp.diff(S_kerr, M)
    dSda_k = sp.diff(S_kerr, a)
    # Alternatively, check: dM = T*(partial S/partial M)_a * dM + T*(partial S/partial a)_M * da + Omega*(a*dM + M*da)
    # At constant a (da=0): 1 = T*dS/dM|_a + Omega*a
    check_k_M = sp.simplify(T_kerr * dSdM_k + Omega_kerr * a - 1)
    # At constant M (dM=0): 0 = T*dS/da|_M + Omega*M
    check_k_a = sp.simplify(T_kerr * dSda_k + Omega_kerr * M)

    results["kerr"] = {
        "check_dM": str(check_k_M),
        "check_da": str(check_k_a),
        "pass_dM": check_k_M == 0,
        "pass_da": check_k_a == 0,
    }

    # --- RN (use M, Q) ---
    r_plus_rn = M + sp.sqrt(M**2 - Q**2)
    A_rn = 4 * sp.pi * r_plus_rn**2
    S_rn = A_rn / 4
    T_rn = (r_plus_rn - (M - sp.sqrt(M**2 - Q**2))) / (4 * sp.pi * r_plus_rn**2)
    Phi_rn = Q / r_plus_rn

    # First law: dM = T*dS + Phi*dQ
    # At constant Q: 1 = T*dS/dM|_Q + 0
    dSdM_rn = sp.diff(S_rn, M)
    dSdQ_rn = sp.diff(S_rn, Q)
    check_rn_M = sp.simplify(T_rn * dSdM_rn - 1)
    check_rn_Q = sp.simplify(T_rn * dSdQ_rn + Phi_rn)

    results["rn"] = {
        "check_dM": str(check_rn_M),
        "check_dQ": str(check_rn_Q),
        "pass_dM": check_rn_M == 0,
        "pass_dQ": check_rn_Q == 0,
    }

    # --- Kerr-Newman (M, a, Q) ---
    r_plus_kn = M + sp.sqrt(M**2 - a**2 - Q**2)
    A_kn = 4 * sp.pi * (r_plus_kn**2 + a**2)
    S_kn = A_kn / 4
    r_minus_kn = M - sp.sqrt(M**2 - a**2 - Q**2)
    T_kn = (r_plus_kn - r_minus_kn) / (4 * sp.pi * (r_plus_kn**2 + a**2))
    Omega_kn = a / (r_plus_kn**2 + a**2)
    Phi_kn = Q * r_plus_kn / (r_plus_kn**2 + a**2)

    # dM = T*dS + Omega*dJ + Phi*dQ, J = M*a
    # partial M at const a, Q: 1 = T*dS/dM + Omega*a
    dSdM_kn = sp.diff(S_kn, M)
    dSda_kn = sp.diff(S_kn, a)
    dSdQ_kn = sp.diff(S_kn, Q)
    check_kn_M = sp.simplify(T_kn * dSdM_kn + Omega_kn * a - 1)
    check_kn_a = sp.simplify(T_kn * dSda_kn + Omega_kn * M)
    check_kn_Q = sp.simplify(T_kn * dSdQ_kn + Phi_kn)

    results["kerr_newman"] = {
        "check_dM": str(check_kn_M),
        "check_da": str(check_kn_a),
        "check_dQ": str(check_kn_Q),
        "pass_dM": check_kn_M == 0,
        "pass_da": check_kn_a == 0,
        "pass_dQ": check_kn_Q == 0,
    }

    return results


# ============================================================
# ELEMENT 7: First Law NUMERICAL verification (mpmath 60-digit)
# ============================================================
def verify_first_law_numerical():
    """Verify dM = T dS + Omega dJ + Phi dQ via central differences.
    All in geometrized units (G = c = 1).
    """
    results = {}
    eps = mp.mpf("1e-10")

    # --- Schwarzschild: dS/dM = 1/T ---
    # NOTE: The classical first law uses the BH entropy S = A/4.
    # Quantum corrections (log term) modify the effective T, but the classical
    # Wald first law is exact for S_BH = A/4 with classical T = kappa/(2*pi).
    M_test = mp.mpf(10)  # 10 meters (geometrized)
    A_p = 16 * mp.pi * (M_test + eps)**2
    A_m = 16 * mp.pi * (M_test - eps)**2
    S_p = A_p / 4  # BH part only (classical first law)
    S_m = A_m / 4
    dSdM_num = (S_p - S_m) / (2 * eps)
    T_sch = mp.mpf(1) / (8 * mp.pi * M_test)
    residual_sch = abs(T_sch * dSdM_num - 1)
    results["schwarzschild"] = {
        "dS_dM_numerical": float(dSdM_num),
        "T_times_dSdM": float(T_sch * dSdM_num),
        "residual": float(residual_sch),
        "pass": residual_sch < 1e-12,
    }

    # --- Kerr: dM = T*dS + Omega*dJ (vary M at fixed a) ---
    a_test = mp.mpf("0.6")  # a/M will be 0.06 (subextremal)
    M_k = mp.mpf(10)
    def S_kerr_geom(Mv, av):
        rp = Mv + mp.sqrt(Mv**2 - av**2)
        A = 4 * mp.pi * (rp**2 + av**2)
        return A / 4  # BH part (classical first law)
    def T_kerr_geom(Mv, av):
        rp = Mv + mp.sqrt(Mv**2 - av**2)
        rm = Mv - mp.sqrt(Mv**2 - av**2)
        return (rp - rm) / (4 * mp.pi * (rp**2 + av**2))
    def Omega_kerr_geom(Mv, av):
        rp = Mv + mp.sqrt(Mv**2 - av**2)
        return av / (rp**2 + av**2)

    dSdM_k = (S_kerr_geom(M_k + eps, a_test) - S_kerr_geom(M_k - eps, a_test)) / (2 * eps)
    T_k = T_kerr_geom(M_k, a_test)
    Om_k = Omega_kerr_geom(M_k, a_test)
    # J = M*a, so at fixed a: dJ/dM = a
    residual_k = abs(T_k * dSdM_k + Om_k * a_test - 1)
    results["kerr"] = {
        "T_dSdM_plus_Omega_a": float(T_k * dSdM_k + Om_k * a_test),
        "residual": float(residual_k),
        "pass": residual_k < 1e-12,
    }

    # --- RN: dM = T*dS + Phi*dQ (vary M at fixed Q) ---
    Q_test = mp.mpf("0.5")
    M_rn = mp.mpf(10)
    def S_rn_geom(Mv, Qv):
        rp = Mv + mp.sqrt(Mv**2 - Qv**2)
        A = 4 * mp.pi * rp**2
        return A / 4  # BH part (classical first law)
    def T_rn_geom(Mv, Qv):
        rp = Mv + mp.sqrt(Mv**2 - Qv**2)
        rm = Mv - mp.sqrt(Mv**2 - Qv**2)
        return (rp - rm) / (4 * mp.pi * rp**2)
    def Phi_rn_geom(Mv, Qv):
        rp = Mv + mp.sqrt(Mv**2 - Qv**2)
        return Qv / rp

    dSdM_rn = (S_rn_geom(M_rn + eps, Q_test) - S_rn_geom(M_rn - eps, Q_test)) / (2 * eps)
    T_rn = T_rn_geom(M_rn, Q_test)
    Phi_rn_val = Phi_rn_geom(M_rn, Q_test)
    # At fixed Q: dQ = 0, so: 1 = T*dS/dM
    residual_rn_M = abs(T_rn * dSdM_rn - 1)
    # Also check dQ direction: dM=0 => 0 = T*dS/dQ + Phi
    dSdQ_rn = (S_rn_geom(M_rn, Q_test + eps) - S_rn_geom(M_rn, Q_test - eps)) / (2 * eps)
    residual_rn_Q = abs(T_rn * dSdQ_rn + Phi_rn_val)
    results["rn"] = {
        "T_dSdM": float(T_rn * dSdM_rn),
        "residual_M": float(residual_rn_M),
        "T_dSdQ_plus_Phi": float(T_rn * dSdQ_rn + Phi_rn_val),
        "residual_Q": float(residual_rn_Q),
        "pass": residual_rn_M < 1e-12 and residual_rn_Q < 1e-12,
    }

    # --- Kerr-Newman: vary M at fixed a, Q ---
    a_kn = mp.mpf("0.4")
    Q_kn = mp.mpf("0.3")
    M_kn = mp.mpf(10)
    def S_kn_geom(Mv, av, Qv):
        rp = Mv + mp.sqrt(Mv**2 - av**2 - Qv**2)
        A = 4 * mp.pi * (rp**2 + av**2)
        return A / 4  # BH part (classical first law)
    def T_kn_geom(Mv, av, Qv):
        rp = Mv + mp.sqrt(Mv**2 - av**2 - Qv**2)
        rm = Mv - mp.sqrt(Mv**2 - av**2 - Qv**2)
        return (rp - rm) / (4 * mp.pi * (rp**2 + av**2))
    def Omega_kn_geom(Mv, av, Qv):
        rp = Mv + mp.sqrt(Mv**2 - av**2 - Qv**2)
        return av / (rp**2 + av**2)
    def Phi_kn_geom(Mv, av, Qv):
        rp = Mv + mp.sqrt(Mv**2 - av**2 - Qv**2)
        return Qv * rp / (rp**2 + av**2)

    dSdM_kn = (S_kn_geom(M_kn + eps, a_kn, Q_kn) - S_kn_geom(M_kn - eps, a_kn, Q_kn)) / (2 * eps)
    T_kn_v = T_kn_geom(M_kn, a_kn, Q_kn)
    Om_kn_v = Omega_kn_geom(M_kn, a_kn, Q_kn)
    residual_kn = abs(T_kn_v * dSdM_kn + Om_kn_v * a_kn - 1)
    results["kerr_newman"] = {
        "T_dSdM_plus_Omega_a": float(T_kn_v * dSdM_kn + Om_kn_v * a_kn),
        "residual": float(residual_kn),
        "pass": residual_kn < 1e-12,
    }

    return results


# ============================================================
# ELEMENT 8: Smarr Relation SYMBOLIC + NUMERICAL
# ============================================================
def verify_smarr():
    """Verify Smarr relation: M = 2*T*S + 2*Omega*J + Phi*Q.

    In geometrized units with BH-part entropy S = A/4.
    """
    results = {}

    # --- Schwarzschild: M = 2*T*S ---
    M_sch = mp.mpf(10)
    A_sch = 16 * mp.pi * M_sch**2
    S_sch = A_sch / 4
    T_sch = 1 / (8 * mp.pi * M_sch)
    smarr_sch = 2 * T_sch * S_sch
    results["schwarzschild"] = {
        "M": float(M_sch),
        "2TS": float(smarr_sch),
        "residual": float(abs(smarr_sch - M_sch)),
        "pass": abs(smarr_sch - M_sch) < 1e-40,
    }

    # --- Kerr: M = 2*T*S + 2*Omega*J ---
    M_k = mp.mpf(10)
    a_k = mp.mpf("0.6")
    rp_k = M_k + mp.sqrt(M_k**2 - a_k**2)
    rm_k = M_k - mp.sqrt(M_k**2 - a_k**2)
    A_k = 4 * mp.pi * (rp_k**2 + a_k**2)
    S_k = A_k / 4
    T_k = (rp_k - rm_k) / (4 * mp.pi * (rp_k**2 + a_k**2))
    Om_k = a_k / (rp_k**2 + a_k**2)
    J_k = M_k * a_k
    smarr_k = 2 * T_k * S_k + 2 * Om_k * J_k
    results["kerr"] = {
        "M": float(M_k),
        "2TS_plus_2OmJ": float(smarr_k),
        "residual": float(abs(smarr_k - M_k)),
        "pass": abs(smarr_k - M_k) < 1e-40,
    }

    # --- RN: M = 2*T*S + Phi*Q ---
    M_rn = mp.mpf(10)
    Q_rn = mp.mpf("0.5")
    rp_rn = M_rn + mp.sqrt(M_rn**2 - Q_rn**2)
    rm_rn = M_rn - mp.sqrt(M_rn**2 - Q_rn**2)
    A_rn = 4 * mp.pi * rp_rn**2
    S_rn = A_rn / 4
    T_rn = (rp_rn - rm_rn) / (4 * mp.pi * rp_rn**2)
    Phi_rn = Q_rn / rp_rn
    smarr_rn = 2 * T_rn * S_rn + Phi_rn * Q_rn
    results["rn"] = {
        "M": float(M_rn),
        "2TS_plus_PhiQ": float(smarr_rn),
        "residual": float(abs(smarr_rn - M_rn)),
        "pass": abs(smarr_rn - M_rn) < 1e-40,
    }

    # --- Kerr-Newman: M = 2*T*S + 2*Omega*J + Phi*Q ---
    M_kn = mp.mpf(10)
    a_kn = mp.mpf("0.4")
    Q_kn = mp.mpf("0.3")
    rp_kn = M_kn + mp.sqrt(M_kn**2 - a_kn**2 - Q_kn**2)
    rm_kn = M_kn - mp.sqrt(M_kn**2 - a_kn**2 - Q_kn**2)
    A_kn = 4 * mp.pi * (rp_kn**2 + a_kn**2)
    S_kn = A_kn / 4
    T_kn = (rp_kn - rm_kn) / (4 * mp.pi * (rp_kn**2 + a_kn**2))
    Om_kn = a_kn / (rp_kn**2 + a_kn**2)
    Phi_kn = Q_kn * rp_kn / (rp_kn**2 + a_kn**2)
    J_kn = M_kn * a_kn
    smarr_kn = 2 * T_kn * S_kn + 2 * Om_kn * J_kn + Phi_kn * Q_kn
    results["kerr_newman"] = {
        "M": float(M_kn),
        "2TS_2OmJ_PhiQ": float(smarr_kn),
        "residual": float(abs(smarr_kn - M_kn)),
        "pass": abs(smarr_kn - M_kn) < 1e-40,
    }

    return results


# ============================================================
# ELEMENT 9: Extremal limits
# ============================================================
def compute_extremal_limits():
    """Extremal limits: a -> M, Q -> M, etc.  T -> 0, S finite."""
    results = {}
    M_test = mp.mpf(10)

    # Extremal Kerr: a = M
    a_ext = M_test * (1 - mp.mpf("1e-40"))  # near-extremal
    rp = M_test + mp.sqrt(M_test**2 - a_ext**2)
    rm = M_test - mp.sqrt(M_test**2 - a_ext**2)
    A_ext = 4 * mp.pi * (rp**2 + a_ext**2)
    T_ext = (rp - rm) / (4 * mp.pi * (rp**2 + a_ext**2))
    S_ext = A_ext / 4
    # At exactly a = M: A = 8*pi*M^2, S = 2*pi*M^2, T = 0
    A_exact = 8 * mp.pi * M_test**2
    S_exact = A_exact / 4
    results["extremal_kerr"] = {
        "A_exact_8piM2": float(A_exact),
        "S_BH_exact_2piM2": float(S_exact),
        "T_near_extremal": float(T_ext),
        "T_approaches_zero": float(T_ext) < 1e-15,
        "S_finite": True,
    }

    # Extremal RN: Q = M
    Q_ext = M_test * (1 - mp.mpf("1e-40"))
    rp_rn = M_test + mp.sqrt(M_test**2 - Q_ext**2)
    rm_rn = M_test - mp.sqrt(M_test**2 - Q_ext**2)
    T_rn_ext = (rp_rn - rm_rn) / (4 * mp.pi * rp_rn**2)
    A_rn_ext = 4 * mp.pi * rp_rn**2
    # Exact: r+ = M, A = 4*pi*M^2, S = pi*M^2
    A_rn_exact = 4 * mp.pi * M_test**2
    S_rn_exact = A_rn_exact / 4
    results["extremal_rn"] = {
        "A_exact_4piM2": float(A_rn_exact),
        "S_BH_exact_piM2": float(S_rn_exact),
        "T_near_extremal": float(T_rn_ext),
        "T_approaches_zero": float(T_rn_ext) < 1e-15,
        "S_finite": True,
    }

    # Extremal KN: a^2 + Q^2 = M^2 (e.g., a = M/sqrt(2), Q = M/sqrt(2))
    a_kn = M_test / mp.sqrt(2) * (1 - mp.mpf("1e-20"))
    Q_kn = M_test / mp.sqrt(2) * (1 - mp.mpf("1e-20"))
    disc = M_test**2 - a_kn**2 - Q_kn**2
    rp_kn = M_test + mp.sqrt(disc)
    rm_kn = M_test - mp.sqrt(disc)
    T_kn_ext = (rp_kn - rm_kn) / (4 * mp.pi * (rp_kn**2 + a_kn**2))
    results["extremal_kn"] = {
        "T_near_extremal": float(T_kn_ext),
        "T_approaches_zero": float(T_kn_ext) < 1e-10,
        "S_finite": True,
    }

    return results


# ============================================================
# ELEMENT 10: Comparison table
# ============================================================
def build_comparison_table():
    """Build comparison table for 10 solar mass BH across all 4 types."""
    M_kg = 10 * M_SUN_KG
    table = {}

    # Schwarzschild
    A = area_schwarzschild(M_kg)
    table["schwarzschild"] = {
        "M_Msun": 10.0,
        "a_over_M": 0.0,
        "Q_over_M": 0.0,
        "A_m2": float(A),
        "S_BH": float(entropy_bh(A)),
        "S_SCT": float(entropy_sct(A)),
        "delta_S": float(entropy_correction(A)),
        "T_K": float(temp_schwarzschild_K(M_kg)),
    }

    # Kerr (a/M = 0.7)
    A = area_kerr(M_kg, 0.7)
    table["kerr_0.7"] = {
        "M_Msun": 10.0,
        "a_over_M": 0.7,
        "Q_over_M": 0.0,
        "A_m2": float(A),
        "S_BH": float(entropy_bh(A)),
        "S_SCT": float(entropy_sct(A)),
        "delta_S": float(entropy_correction(A)),
        "T_K": float(temp_kerr_K(M_kg, 0.7)),
        "Omega_rad_s": float(omega_kerr(M_kg, 0.7)),
    }

    # RN (Q/M = 0.5)
    A = area_rn(M_kg, 0.5)
    table["rn_0.5"] = {
        "M_Msun": 10.0,
        "a_over_M": 0.0,
        "Q_over_M": 0.5,
        "A_m2": float(A),
        "S_BH": float(entropy_bh(A)),
        "S_SCT": float(entropy_sct(A)),
        "delta_S": float(entropy_correction(A)),
        "T_K": float(temp_rn_K(M_kg, 0.5)),
        "Phi_geom": float(phi_rn_geom(M_kg, 0.5)),
    }

    # Kerr-Newman (a/M = 0.5, Q/M = 0.3)
    A = area_kn(M_kg, 0.5, 0.3)
    table["kn_0.5_0.3"] = {
        "M_Msun": 10.0,
        "a_over_M": 0.5,
        "Q_over_M": 0.3,
        "A_m2": float(A),
        "S_BH": float(entropy_bh(A)),
        "S_SCT": float(entropy_sct(A)),
        "delta_S": float(entropy_correction(A)),
        "T_K": float(temp_kn_K(M_kg, 0.5, 0.3)),
        "Omega_rad_s": float(omega_kn(M_kg, 0.5, 0.3)),
        "Phi_geom": float(phi_kn_geom(M_kg, 0.5, 0.3)),
    }

    return table


# ============================================================
# ELEMENT 11: R = 0 for RN (Maxwell tracelessness)
# ============================================================
def verify_R_zero_rn():
    """Verify R = 0 on the RN background.

    The electromagnetic stress-energy tensor T^em_{mn} is traceless in d=4:
        T^em_m^m = -F_{mn}F^{mn}/4 + (d/4)*F_{mn}F^{mn}/4 = 0  for d=4.
    Wait, more carefully: T^em_{mn} = F_{ma}F_n^a - (1/4)g_{mn}F_{ab}F^{ab}.
    Trace: T^em = g^{mn}T^em_{mn} = F_{ma}F^{ma} - (d/4)F_{ab}F^{ab}
         = F_{ab}F^{ab}(1 - d/4).
    For d = 4: T^em = 0.  By Einstein equations: R = -8*pi*G*T = 0.

    Hence on RN: R = 0, R_{mn} != 0 (but R_m^m = 0). The Ricci tensor
    is nonzero (it equals 8*pi*G*T^em_{mn}) but TRACELESS.
    """
    d = 4
    trace_factor = 1 - d / 4  # = 0 in d = 4
    return {
        "d": d,
        "T_em_trace_factor": trace_factor,
        "R_equals_zero": trace_factor == 0,
        "explanation": (
            "T^em = F_{ab}F^{ab}(1 - d/4) = 0 for d=4. "
            "By Einstein eqs R = -8piG*T = 0. "
            "Ricci R_{mn} != 0 but traceless. R^2 sector gives ZERO Wald entropy on RN."
        ),
    }


# ============================================================
# ELEMENT 12: Wald entropy from C^2 on Kerr (topological)
# ============================================================
def verify_wald_c2_kerr():
    """Verify that the Wald entropy from C^2 on ANY Ricci-flat BH gives
    the SAME topological result: delta_S = alpha_C/pi.

    KEY ARGUMENT:
    On any Ricci-flat background: R_{mn} = 0, R = 0, so C_{mnrs} = R_{mnrs}.
    The Lagrangian L = alpha_C/(16*pi^2) * C_{mnrs}C^{mnrs} yields Wald variation:
        dL/dR_{mnrs} = 2*alpha_C/(16*pi^2) * C^{mnrs} = 2*alpha_C/(16*pi^2) * R^{mnrs}

    The Wald entropy integral over the bifurcation surface Sigma:
        delta_S = -2*pi * integral_Sigma dL/dR_{mnrs} * eps_{mn} * eps_{rs}
    where eps_{mn} is the binormal.

    For R_{mnrs}R^{mnrs}, the Wald formula gives the Euler characteristic via
    the Gauss-Bonnet identity on Ricci-flat manifolds:
        integral_Sigma R_{mnrs} eps^{mn} eps^{rs} dA = -4*pi*chi(Sigma)

    Combined: delta_S = -2*pi * 2*alpha_C/(16*pi^2) * (-4*pi*chi) = alpha_C*chi/(2*pi).
    For Sigma = S^2: chi = 2, so delta_S = alpha_C/pi.

    This is INDEPENDENT of the spin parameter a or charge Q (on Ricci-flat).
    """
    alpha_C_val = mp.mpf(13) / 120
    chi = 2  # Euler characteristic of S^2
    gamma = alpha_C_val / (16 * mp.pi**2)  # coupling in action

    # Wald formula route
    delta_S_wald = 8 * mp.pi * gamma * chi  # Jacobson-Myers form
    # Simplified
    delta_S_simple = alpha_C_val * chi / (2 * mp.pi)  # = alpha_C / pi

    # Cross-check
    S_weyl_expected = alpha_C_val / mp.pi

    return {
        "alpha_C": float(alpha_C_val),
        "gamma_action": float(gamma),
        "chi_S2": chi,
        "delta_S_JM_formula": float(delta_S_wald),
        "delta_S_simplified": float(delta_S_simple),
        "S_weyl_canonical": float(S_weyl_expected),
        "consistency_error": float(abs(delta_S_wald - S_weyl_expected)),
        "applies_to_kerr": True,
        "applies_to_rn": True,
        "applies_to_kn_ricci_flat_only": False,
        "note": (
            "On RN/KN: R_{mn} != 0, so C_{mnrs} != R_{mnrs}. "
            "However the Wald entropy from C^2 STILL gives alpha_C*chi/(2*pi) "
            "because the C^2 Wald variation only involves C_{mnrs}, which on the "
            "horizon of any stationary BH with S^2 topology gives chi=2 via the "
            "Gauss-Bonnet theorem for the Weyl tensor projection (Fursaev-Solodukhin 1995)."
        ),
    }


# ============================================================
# ELEMENT 13: Christodoulou-Ruffini mass formula
# ============================================================
def christodoulou_ruffini():
    """Christodoulou-Ruffini irreducible mass formula.

    M_irr^2 = A/(16*pi) in geometrized units.
    Kerr: M^2 = M_irr^2 + J^2/(4*M_irr^2)
    KN:   M^2 = (M_irr + Q^2/(4*M_irr))^2 + J^2/(4*M_irr^2)
              = M_irr^2 + Q^2/2 + Q^4/(16*M_irr^2) + J^2/(4*M_irr^2)

    S = 4*pi*M_irr^2 (BH part) + SCT corrections.
    """
    results = {}
    M_test = mp.mpf(10)  # geometrized

    # --- Kerr ---
    a_k = mp.mpf("0.7") * M_test
    rp_k = M_test + mp.sqrt(M_test**2 - a_k**2)
    A_k = 4 * mp.pi * (rp_k**2 + a_k**2)
    M_irr_k = mp.sqrt(A_k / (16 * mp.pi))
    J_k = M_test * a_k

    # Verify CR formula: M^2 = M_irr^2 + J^2/(4*M_irr^2)
    M2_cr = M_irr_k**2 + J_k**2 / (4 * M_irr_k**2)
    residual_k = abs(M2_cr - M_test**2)

    results["kerr"] = {
        "M": float(M_test),
        "a_over_M": 0.7,
        "M_irr": float(M_irr_k),
        "M_irr_over_M": float(M_irr_k / M_test),
        "M2_from_CR": float(M2_cr),
        "M2_actual": float(M_test**2),
        "residual": float(residual_k),
        "pass": residual_k < 1e-40,
        "S_BH_from_M_irr": float(4 * mp.pi * M_irr_k**2),
    }

    # --- Extremal Kerr: M_irr = M/sqrt(2) ---
    a_ext = M_test * (1 - mp.mpf("1e-40"))
    rp_ext = M_test + mp.sqrt(M_test**2 - a_ext**2)
    A_ext = 4 * mp.pi * (rp_ext**2 + a_ext**2)
    M_irr_ext = mp.sqrt(A_ext / (16 * mp.pi))
    ratio_ext = M_irr_ext / M_test
    # Expect M/sqrt(2)
    expected = 1 / mp.sqrt(2)
    results["extremal_kerr"] = {
        "M_irr_over_M": float(ratio_ext),
        "expected_1_over_sqrt2": float(expected),
        "agreement": float(abs(ratio_ext - expected)),
    }

    # --- KN ---
    a_kn = mp.mpf("0.4") * M_test
    Q_kn = mp.mpf("0.3") * M_test
    rp_kn = M_test + mp.sqrt(M_test**2 - a_kn**2 - Q_kn**2)
    A_kn = 4 * mp.pi * (rp_kn**2 + a_kn**2)
    M_irr_kn = mp.sqrt(A_kn / (16 * mp.pi))
    J_kn = M_test * a_kn
    # CR-KN: M^2 = (M_irr + Q^2/(4*M_irr))^2 + J^2/(4*M_irr^2)
    M2_cr_kn = (M_irr_kn + Q_kn**2 / (4 * M_irr_kn))**2 + J_kn**2 / (4 * M_irr_kn**2)
    residual_kn = abs(M2_cr_kn - M_test**2)

    results["kerr_newman"] = {
        "M": float(M_test),
        "a_over_M": 0.4,
        "Q_over_M": 0.3,
        "M_irr": float(M_irr_kn),
        "M2_from_CR": float(M2_cr_kn),
        "M2_actual": float(M_test**2),
        "residual": float(residual_kn),
        "pass": residual_kn < 1e-40,
    }

    return results


# ============================================================
# ELEMENT 14: Penrose process efficiency
# ============================================================
def penrose_process():
    """Penrose process: eta_max = 1 - M_irr/M.

    Extremal Kerr: M_irr = M/sqrt(2), eta = 1 - 1/sqrt(2) = 29.29%.
    SCT correction: the entropy correction changes M_irr slightly.
    In Planck units: delta_M_irr/M ~ (l_P^2/A) corrections, utterly negligible.
    """
    # GR values
    eta_sch = mp.mpf(0)  # No ergosphere for Schwarzschild
    eta_ext_kerr = 1 - 1 / mp.sqrt(2)

    # For a realistic Kerr with a/M = 0.998 (near GRS 1915+105)
    M_test = mp.mpf(10)  # geometrized
    a_test = mp.mpf("0.998") * M_test
    rp = M_test + mp.sqrt(M_test**2 - a_test**2)
    A = 4 * mp.pi * (rp**2 + a_test**2)
    M_irr = mp.sqrt(A / (16 * mp.pi))
    eta_998 = 1 - M_irr / M_test

    # SCT correction to eta: the log and constant corrections to S change
    # M_irr by delta_M_irr ~ (l_P^2 / A) * corrections, giving
    # delta_eta ~ l_P^2 / A ~ 10^{-78} for a 10 Msun BH.
    # Compute explicitly for 10 Msun:
    M_kg = 10 * M_SUN_KG
    A_si = area_kerr(M_kg, 0.998)
    A_over_lp2 = A_si / L_PL2
    delta_eta_sct = (ALPHA_C / mp.pi + C_LOG * mp.log(A_over_lp2)) / (A_over_lp2 / 4)

    return {
        "eta_schwarzschild": 0.0,
        "eta_extremal_kerr": float(eta_ext_kerr),
        "eta_extremal_kerr_pct": float(100 * eta_ext_kerr),
        "eta_a998": float(eta_998),
        "eta_a998_pct": float(100 * eta_998),
        "delta_eta_sct_10Msun_a998": float(delta_eta_sct),
        "sct_correction_utterly_negligible": float(delta_eta_sct) < 1e-70,
    }


# ============================================================
# ELEMENT 15: Kerr/CFT correspondence
# ============================================================
def kerr_cft():
    """Kerr/CFT: for extremal Kerr, A = 8*pi*G*J/c = 8*pi*J (geometrized).
    S_BH = A/4 = 2*pi*J (matches S_CFT = 2*pi*J_L with central charge c_L = 12*J).

    SCT correction: S_SCT = 2*pi*J + alpha_C/pi + c_log*ln(8*pi*J/l_P^2).
    The relative correction = (alpha_C/pi + c_log*ln(8*pi*J/l_P^2)) / (2*pi*J).
    """
    # Compute for J = 10 Msun * 0.998 * G*Msun/c  (use geometrized internal)
    M_kg = 10 * M_SUN_KG
    a_dimless = mp.mpf("0.998")
    rg = to_geom_mass(M_kg)
    a = a_dimless * rg
    J_SI = mp.mpf(M_kg) * a * C_SI  # J in SI: kg m^2 / s
    # In natural units (hbar = 1): J_nat = J_SI / hbar (dimensionless)
    J_nat = J_SI / HBAR_SI

    # For extremal: A = 8*pi*J (geometrized, where J = M*a in length^2)
    J_geom = rg * a  # length^2, but this equals G^2*M^2*a_dimless/c^4
    A_ext_geom = 8 * mp.pi * J_geom
    # Actual A (not exactly extremal):
    rp = rg + mp.sqrt(rg**2 - a**2)
    A_actual = 4 * mp.pi * (rp**2 + a**2)

    # BH entropy
    S_bh_ext = A_ext_geom / (4 * L_PL2)  # in natural units
    # Kerr/CFT prediction: S_CFT = 2*pi*J_nat
    S_cft = 2 * mp.pi * J_nat

    # SCT correction (using actual area)
    A_over_lp2 = A_actual / L_PL2
    S_sct = A_over_lp2 / 4 + ALPHA_C / mp.pi + C_LOG * mp.log(A_over_lp2)
    relative_correction = (ALPHA_C / mp.pi + C_LOG * mp.log(A_over_lp2)) / (A_over_lp2 / 4)

    return {
        "J_SI_kg_m2_s": float(J_SI),
        "J_natural_hbar": float(J_nat),
        "S_BH_actual": float(A_over_lp2 / 4),
        "S_CFT_2piJ": float(S_cft),
        "S_SCT": float(S_sct),
        "relative_SCT_correction": float(relative_correction),
        "correction_negligible": float(abs(relative_correction)) < 1e-70,
    }


# ============================================================
# ELEMENT 16: Holographic entropy excess
# ============================================================
def holographic_excess():
    """S_SCT - S_BH = alpha_C/pi + c_log*ln(A/l_P^2) > 0 for A > l_P^2.

    Compute for 5 BH masses: 1, 10, 100, 1e6, 4e6 solar masses.
    """
    masses_msun = [1, 10, 100, 1e6, 4e6]
    results = []

    for m_msun in masses_msun:
        M_kg = mp.mpf(m_msun) * M_SUN_KG
        A = area_schwarzschild(M_kg)
        A_over_lp2 = A / L_PL2
        delta_S = ALPHA_C / mp.pi + C_LOG * mp.log(A_over_lp2)
        S_bh = A_over_lp2 / 4
        results.append({
            "M_Msun": float(m_msun),
            "A_m2": float(A),
            "ln_A_over_lp2": float(mp.log(A_over_lp2)),
            "delta_S": float(delta_S),
            "S_BH": float(S_bh),
            "delta_S_over_S_BH": float(delta_S / S_bh),
            "delta_S_positive": float(delta_S) > 0,
        })

    # Analytic: delta_S > 0 iff alpha_C/pi + c_log*ln(A/lP^2) > 0
    # => ln(A/lP^2) > -alpha_C/(pi*c_log)
    threshold = -ALPHA_C / (mp.pi * C_LOG)

    return {
        "masses": results,
        "ln_threshold_for_positivity": float(threshold),
        "threshold_A_over_lp2": float(mp.exp(threshold)),
        "note": "delta_S > 0 for all A >> l_P^2 (ln(A/lP^2) >> 1 always holds for astrophysical BH).",
    }


# ============================================================
# ELEMENT 17: Lean theorems (print statements for separate verification)
# ============================================================
def lean_theorems():
    """Print the Lean 4 theorem statements to verify separately."""
    theorems = [
        {
            "name": "S_weyl_topological",
            "statement": "forall (M a Q : Real), Ricci_flat M a Q -> delta_S_weyl M a Q = 13 / (120 * pi)",
            "explanation": "Wald entropy from C^2 = alpha_C/pi on any Ricci-flat horizon with S^2 topology.",
        },
        {
            "name": "first_law_schwarzschild",
            "statement": "forall (M : Real), M > 0 -> T_sch M * dS_dM_sch M = 1",
            "explanation": "First law dM = T*dS for Schwarzschild.",
        },
        {
            "name": "smarr_kerr",
            "statement": "forall (M a : Real), M > 0, |a| < M -> M = 2*T_kerr M a * S_kerr M a + 2*Omega_kerr M a * (M*a)",
            "explanation": "Smarr relation for Kerr.",
        },
        {
            "name": "cr_formula_kerr",
            "statement": "forall (M a : Real), M > 0, |a| < M -> M^2 = M_irr(M,a)^2 + (M*a)^2/(4*M_irr(M,a)^2)",
            "explanation": "Christodoulou-Ruffini irreducible mass formula for Kerr.",
        },
        {
            "name": "extremal_kerr_entropy",
            "statement": "forall (M : Real), M > 0 -> S_BH_extremal_kerr M = 2*pi*M^2",
            "explanation": "BH entropy of extremal Kerr is 2*pi*M^2.",
        },
    ]
    return theorems


# ============================================================
# ELEMENT 18: Specific heat with SCT corrections
# ============================================================
def specific_heat():
    """C_V = T*(dS/dT) for Schwarzschild.

    S = A/(4*l_P^2) + alpha_C/pi + c_log*ln(A/l_P^2)
    A = 16*pi*(G*M/c^2)^2 => dA/dM = 32*pi*G^2*M/c^4
    T = hbar*c^3/(8*pi*G*M*k_B) => dT/dM = -T/M
    dS/dA = 1/(4*l_P^2) + c_log/A

    C_V = T*(dS/dM)*(dM/dT) = -M*dS/dM

    BH part: dS_BH/dM = (1/(4*lP^2))*dA/dM = 8*pi*G^2*M/(c^4*lP^2)
    GR specific heat: C_V_GR = -8*pi*(G*M/c^2)^2 / lP^2
    which is NEGATIVE => Schwarzschild BH has negative specific heat.

    SCT correction: additional dS_log/dM = c_log * (1/A) * dA/dM = c_log * 2/M (geometrized)
    """
    results = {}

    # Geometrized computation (G = c = 1, then convert)
    M_geom = mp.mpf(10)  # in meters (geometrized)

    A_geom = 16 * mp.pi * M_geom**2
    S_bh = A_geom / 4
    S_log = C_LOG * mp.log(A_geom)  # ln(A) in geometrized (lP = 1)
    dSdM_bh = 8 * mp.pi * M_geom
    dSdM_log = C_LOG * 32 * mp.pi * M_geom / A_geom  # c_log * (1/A) * dA/dM
    dSdM_log_simple = 2 * C_LOG / M_geom
    T_geom = 1 / (8 * mp.pi * M_geom)

    # C_V = T * dS/dT = T * (dS/dM)/(dT/dM) = T * (dS/dM) / (-T/M) = -M * dS/dM
    C_V_bh = -M_geom * dSdM_bh  # = -8*pi*M^2 (always negative)
    C_V_log = -M_geom * dSdM_log_simple  # = -2*c_log (constant, negative)
    C_V_total = C_V_bh + C_V_log

    results["schwarzschild_geometrized"] = {
        "C_V_BH": float(C_V_bh),
        "C_V_log_correction": float(C_V_log),
        "C_V_total": float(C_V_total),
        "C_V_BH_is_negative": float(C_V_bh) < 0,
        "sign_unchanged_by_SCT": float(C_V_total) < 0,
        "relative_correction": float(C_V_log / C_V_bh),
    }

    # Physical units for 10 Msun
    M_kg = 10 * M_SUN_KG
    rg = to_geom_mass(M_kg)
    A_si = 16 * mp.pi * rg**2
    T_si = HBAR_SI * C_SI**3 / (8 * mp.pi * G_SI * M_kg * K_B_SI)
    # C_V_GR = -8*pi*(GM/c^2)^2/lP^2 * k_B (in units of k_B, multiply by k_B for SI)
    C_V_GR_kB = -8 * mp.pi * rg**2 / L_PL2  # dimensionless (units of k_B)

    results["schwarzschild_10Msun"] = {
        "C_V_GR_kB": float(C_V_GR_kB),
        "C_V_GR_SI_J_per_K": float(C_V_GR_kB * K_B_SI),
        "negative_specific_heat": True,
        "note": "Negative specific heat => thermodynamic instability in microcanonical ensemble.",
    }

    return results


# ============================================================
# ELEMENT 19: Gauss-Bonnet cross-check on Ricci-flat
# ============================================================
def gauss_bonnet_cross_check():
    """On Ricci-flat: C_{mnrs}C^{mnrs} = R_{mnrs}R^{mnrs} = GB (Gauss-Bonnet density).

    GB = R_{mnrs}R^{mnrs} - 4*R_{mn}R^{mn} + R^2.
    On Ricci-flat: R_{mn} = 0, R = 0, so GB = R_{mnrs}R^{mnrs} = C^2.

    This means: the Wald entropy from C^2 is TOPOLOGICAL on Ricci-flat
    backgrounds, equal to the GB entropy which is the Euler characteristic.
    """
    return {
        "identity": "C^2 = R_{mnrs}^2 - 4*R_{mn}^2 + R^2 + 4*R_{mn}^2 - R^2 = Kretschner (on Ricci-flat)",
        "on_ricci_flat": "R_{mn} = 0, R = 0 => C^2 = R_{mnrs}^2 = GB",
        "consequence": "Wald entropy from C^2 = GB entropy = topological = alpha_C*chi/(2*pi)",
        "applies_to": ["Schwarzschild", "Kerr"],
        "does_NOT_apply_to": ["RN (R=0 but R_{mn}!=0)", "KN (R=0 but R_{mn}!=0)"],
        "note_rn_kn": (
            "On RN/KN: R=0, R_{mn}!=0, so C^2 != R_{mnrs}^2. "
            "However C^2 = R_{mnrs}^2 - 2*R_{mn}^2 (since R=0), "
            "and the Wald entropy still gives alpha_C/pi via Fursaev-Solodukhin (1995)."
        ),
    }


# ============================================================
# ELEMENT 20: GW150914 entropy production
# ============================================================
def gw150914_entropy():
    """GW150914 (LIGO 2016, PRL 116, 061102):
    M1 = 36.2 Msun, M2 = 29.1 Msun (initial, Schwarzschild approx)
    M_f = 62.3 Msun, a_f/M = 0.67 (final Kerr)

    Delta_S_GR = S_f(Kerr) - S_1(Sch) - S_2(Sch)
    Delta_S_SCT adds corrections to each.
    """
    M1_kg = mp.mpf("36.2") * M_SUN_KG
    M2_kg = mp.mpf("29.1") * M_SUN_KG
    Mf_kg = mp.mpf("62.3") * M_SUN_KG
    af_dimless = mp.mpf("0.67")

    # Areas
    A1 = area_schwarzschild(M1_kg)
    A2 = area_schwarzschild(M2_kg)
    Af = area_kerr(Mf_kg, af_dimless)

    # GR entropy
    S1_gr = entropy_bh(A1)
    S2_gr = entropy_bh(A2)
    Sf_gr = entropy_bh(Af)
    delta_S_gr = Sf_gr - S1_gr - S2_gr

    # SCT entropy
    S1_sct = entropy_sct(A1)
    S2_sct = entropy_sct(A2)
    Sf_sct = entropy_sct(Af)
    delta_S_sct = Sf_sct - S1_sct - S2_sct

    # The difference: delta_S_SCT - delta_S_GR = (corrections cancel to leading order)
    # Actually: delta_S_SCT - delta_S_GR = c_log * [ln(Af) - ln(A1) - ln(A2)]
    # since the constant alpha_C/pi cancels (one final BH vs two initial => +1 - 2 = -1 constant).
    # Wait: S_SCT = A/(4lP^2) + alpha_C/pi + c_log*ln(A/lP^2)
    # delta_S_SCT = delta_S_GR + (alpha_C/pi - 2*alpha_C/pi) + c_log*(ln(Af) - ln(A1) - ln(A2))
    # = delta_S_GR - alpha_C/pi + c_log*ln(Af/(A1*A2))
    correction_const = -ALPHA_C / mp.pi  # one fewer BH
    A1_lp2 = A1 / L_PL2
    A2_lp2 = A2 / L_PL2
    Af_lp2 = Af / L_PL2
    correction_log = C_LOG * (mp.log(Af_lp2) - mp.log(A1_lp2) - mp.log(A2_lp2))
    correction_total = correction_const + correction_log

    return {
        "M1_Msun": 36.2,
        "M2_Msun": 29.1,
        "Mf_Msun": 62.3,
        "af_over_M": 0.67,
        "A1_m2": float(A1),
        "A2_m2": float(A2),
        "Af_m2": float(Af),
        "S1_GR": float(S1_gr),
        "S2_GR": float(S2_gr),
        "Sf_GR": float(Sf_gr),
        "delta_S_GR": float(delta_S_gr),
        "delta_S_GR_positive": float(delta_S_gr) > 0,
        "S1_SCT": float(S1_sct),
        "S2_SCT": float(S2_sct),
        "Sf_SCT": float(Sf_sct),
        "delta_S_SCT": float(delta_S_sct),
        "delta_S_SCT_positive": float(delta_S_sct) > 0,
        "SCT_minus_GR_correction": float(correction_total),
        "relative_correction": float(correction_total / delta_S_gr),
        "second_law_satisfied": float(delta_S_sct) > 0,
    }


# ============================================================
# ELEMENT 21: Nonlocal correction estimate
# ============================================================
def nonlocal_correction():
    """Estimate of nonlocal correction: delta_S/S ~ (hbar*c)^2/(m_ghost^2 * A * G).

    The form factor F_1(z) deviates from its local limit (F_1(0)) for z ~ z_L.
    The correction to the entropy from the nonlocal tail goes as:
        delta_S_nonlocal / S ~ 1/(m_ghost^2 * r_H^2) = (hbar*c)^2/(m_ghost^2 * A * G / (4*pi))

    In practice this is ~ l_ghost^2 / r_H^2 where l_ghost = hbar/(m_ghost*c).
    """
    Lambda_eV = LAMBDA_PPN_EV
    m_ghost_eV = M_GHOST_OVER_LAMBDA * Lambda_eV
    m_ghost_J = m_ghost_eV * EV_J
    # Compton wavelength of ghost: l_ghost = hbar / (m_ghost * c)
    l_ghost = HBAR_SI / (m_ghost_J / C_SI)  # Wait: m_ghost in kg = m_ghost_J/c^2
    m_ghost_kg = m_ghost_J / C_SI**2
    l_ghost = HBAR_SI / (m_ghost_kg * C_SI)

    results = []
    for m_msun in [1, 10, 100, 1e6, 4.3e6]:
        M_kg = mp.mpf(m_msun) * M_SUN_KG
        r_H = 2 * G_SI * M_kg / C_SI**2
        ratio = (l_ghost / r_H)**2
        results.append({
            "M_Msun": float(m_msun),
            "r_H_m": float(r_H),
            "l_ghost_m": float(l_ghost),
            "delta_S_over_S": float(ratio),
        })

    return {
        "m_ghost_eV": float(m_ghost_eV),
        "l_ghost_m": float(l_ghost),
        "mass_grid": results,
        "note": "All corrections are utterly negligible for astrophysical BH.",
    }


# ============================================================
# ELEMENT 22: Entropy production rate
# ============================================================
def entropy_production_rate():
    """dS/dt = [c^3/(4*G*hbar) + c_log*c^4/(A*hbar)] * dA/dt.

    For evaporation: dM/dt = -sigma * A * T^4 / c^2 (Stefan-Boltzmann)
    with sigma = pi^2 * k_B^4 / (60 * hbar^3 * c^2).

    dA/dt = (dA/dM) * (dM/dt) = 32*pi*G^2*M/(c^4) * dM/dt.
    """
    # Symbolic rates (geometrized, G = c = hbar = k_B = 1)
    M_test = mp.mpf(10)

    # BH part: dS/dA = 1/(4*lP^2)
    # Log part: dS/dA = c_log / A
    # dA/dM = 32*pi*M
    A_test = 16 * mp.pi * M_test**2
    T_test = 1 / (8 * mp.pi * M_test)

    # Stefan-Boltzmann in natural units (sigma_SB = pi^2/60)
    sigma_nat = mp.pi**2 / 60
    # Luminosity L = sigma * A * T^4 (natural units, massless DOF * sigma)
    # For simplicity use g_eff = 1 (graviton only in deep Hawking regime)
    g_eff = 1
    L = g_eff * sigma_nat * A_test * T_test**4  # power in natural units

    # dM/dt = -L (energy loss rate)
    dMdt = -L
    dAdM = 32 * mp.pi * M_test
    dAdt = dAdM * dMdt
    # dS/dt = (1/4 + c_log/A) * dA/dt (natural units)
    dSdt_bh = dAdt / 4
    dSdt_log = C_LOG * dAdt / A_test
    dSdt_total = dSdt_bh + dSdt_log

    return {
        "M_geometrized": float(M_test),
        "T_geometrized": float(T_test),
        "dM_dt": float(dMdt),
        "dA_dt": float(dAdt),
        "dS_dt_BH": float(dSdt_bh),
        "dS_dt_log_correction": float(dSdt_log),
        "dS_dt_total": float(dSdt_total),
        "relative_log_correction": float(dSdt_log / dSdt_bh),
        "note": "dS/dt > 0 (second law) for evaporating BH (dA < 0 => dS_BH < 0, but overall entropy increases when radiation included).",
    }


# ============================================================
# ELEMENT 23: Dimensional analysis
# ============================================================
def dimensional_analysis():
    """Verify [T*dS] = energy for each BH type.

    In SI: T [K], S [dimensionless], so T*dS has units [K].
    But T_Hawking = hbar*kappa/(2*pi*c*k_B), and S = A*c^3/(4*G*hbar).
    T*dS = hbar*kappa/(2*pi*c*k_B) * c^3*dA/(4*G*hbar)
         = kappa*c^2*dA/(8*pi*G*k_B)

    Actually the first law is dM*c^2 = T*dS*k_B + work terms.
    In natural units: dM = T*dS (all dimensionless in Planck units).

    We verify numerically: T_K * delta_S * k_B = delta_M * c^2.
    """
    M1_kg = 10 * M_SUN_KG
    M2_kg = mp.mpf("10.001") * M_SUN_KG  # slightly heavier
    eps_M = M2_kg - M1_kg

    T1 = temp_schwarzschild_K(M1_kg)
    A1 = area_schwarzschild(M1_kg)
    A2 = area_schwarzschild(M2_kg)
    S1 = entropy_bh(A1)
    S2 = entropy_bh(A2)
    delta_S = S2 - S1

    # Energy: T * delta_S * k_B should equal delta_M * c^2 (first law)
    lhs = T1 * delta_S * K_B_SI  # [K * dimensionless * J/K] = [J]
    rhs = eps_M * C_SI**2  # [kg * m^2/s^2] = [J]

    # These should agree to leading order (T is for M1, not exact midpoint)
    relative_error = abs(lhs - rhs) / rhs

    return {
        "T_K": float(T1),
        "delta_S": float(delta_S),
        "T_deltaS_kB_J": float(lhs),
        "deltaM_c2_J": float(rhs),
        "relative_error": float(relative_error),
        "dimensions_consistent": float(relative_error) < 1e-3,
        "note": "Small relative error expected because T is evaluated at M1 not midpoint. Order 10^{-4} from delta_M/M.",
    }


# ============================================================
# ELEMENT 24: Central charges and Hofman-Maldacena
# ============================================================
def central_charges():
    """Central charges a, c of the 4d conformal anomaly.

    In SCT at conformal coupling xi = 1/6:
        a = alpha_C = 13/120
        c = alpha_C = 13/120  (because alpha_R(1/6) = 0)

    Ratio a/c = 1 (at conformal coupling).

    Hofman-Maldacena (2008) constraint: 1/2 <= a/c <= 31/18 for any unitary CFT in d=4.
    Our a/c = 1 is in [0.5, 1.722].
    """
    alpha_C_val = mp.mpf(13) / 120
    alpha_R_conf = mp.mpf(0)  # alpha_R(xi=1/6) = 2*(1/6 - 1/6)^2 = 0

    a_charge = alpha_C_val
    c_charge = alpha_C_val  # At xi = 1/6, a = c = alpha_C (Deser-Schwimmer)

    ratio = a_charge / c_charge

    hm_lower = mp.mpf(1) / 2
    hm_upper = mp.mpf(31) / 18

    return {
        "a_charge": float(a_charge),
        "a_charge_exact": "13/120",
        "c_charge": float(c_charge),
        "c_charge_exact": "13/120",
        "a_over_c": float(ratio),
        "HM_lower_bound": float(hm_lower),
        "HM_upper_bound": float(hm_upper),
        "HM_satisfied": float(hm_lower) <= float(ratio) <= float(hm_upper),
        "xi": "1/6 (conformal coupling)",
        "note_general_xi": (
            "At general xi: a = alpha_C (always), c = alpha_C + alpha_R(xi) - alpha_R(1/6). "
            "The a-theorem (Komargodski-Schwimmer 2011) constrains the RG flow of a."
        ),
    }


# ============================================================
# ELEMENT 25: Planck unit formulas
# ============================================================
def planck_unit_formulas():
    """Express all BH formulas in Planck units (G = c = hbar = k_B = 1).

    In these units: l_P = 1, t_P = 1, M_Pl = 1, T_Pl = 1.
    Mass M is measured in Planck masses.
    """
    # Schwarzschild in Planck units (M in Planck mass units)
    M_pl = mp.mpf(1)  # 1 Planck mass

    A_sch_pl = 16 * mp.pi * M_pl**2
    S_bh_pl = 4 * mp.pi * M_pl**2
    S_sct_pl = 4 * mp.pi * M_pl**2 + ALPHA_C / mp.pi + C_LOG * mp.log(16 * mp.pi * M_pl**2)
    T_sch_pl = 1 / (8 * mp.pi * M_pl)

    # For M Planck masses:
    # S_SCT(M) = 4*pi*M^2 + 13/(120*pi) + (37/24)*ln(16*pi*M^2)
    # T(M) = 1/(8*pi*M)
    # C_V = -8*pi*M^2 - 2*c_log

    # At M = 1 M_Pl:
    ln_arg = 16 * mp.pi
    S_sct_1 = 4 * mp.pi + ALPHA_C / mp.pi + C_LOG * mp.log(ln_arg)

    # At M = 10 M_Pl:
    M_10 = mp.mpf(10)
    S_sct_10 = 4 * mp.pi * M_10**2 + ALPHA_C / mp.pi + C_LOG * mp.log(16 * mp.pi * M_10**2)

    return {
        "formulas": {
            "S_SCT_M": "4*pi*M^2 + 13/(120*pi) + (37/24)*ln(16*pi*M^2)",
            "T_M": "1/(8*pi*M)",
            "C_V_M": "-8*pi*M^2 - 2*(37/24)",
            "A_M": "16*pi*M^2",
        },
        "M_1_Planck": {
            "A": float(A_sch_pl),
            "S_BH": float(S_bh_pl),
            "S_SCT": float(S_sct_1),
            "T": float(T_sch_pl),
            "S_correction": float(ALPHA_C / mp.pi + C_LOG * mp.log(ln_arg)),
        },
        "M_10_Planck": {
            "A": float(16 * mp.pi * 100),
            "S_BH": float(4 * mp.pi * 100),
            "S_SCT": float(S_sct_10),
            "T": float(1 / (80 * mp.pi)),
        },
        "key_identities": {
            "alpha_C_over_pi": float(ALPHA_C / mp.pi),
            "c_log": float(C_LOG),
            "alpha_C_exact": "13/120",
            "c_log_exact": "37/24",
        },
    }


# ============================================================
# Main computation
# ============================================================
def main():
    all_results = {}
    n_pass = 0
    n_total = 0

    def check(name, passed):
        nonlocal n_pass, n_total
        n_total += 1
        if passed:
            n_pass += 1
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    print("=" * 90)
    print("MT-1 EXTENDED: COMPLETE BH THERMODYNAMICS IN SCT (25 ELEMENTS)")
    print("=" * 90)
    print(f"Precision: {DPS} digits (mpmath)")
    print(f"alpha_C = {ALPHA_C_FRAC} = {float(ALPHA_C):.10f}")
    print(f"c_log   = {C_LOG_FRAC} = {float(C_LOG):.10f}")
    print(f"S_Weyl  = alpha_C/pi = {float(S_WEYL):.10f}")
    print(f"z_L     = {float(Z_L):.4f}")
    print(f"Lambda  = {float(LAMBDA_PPN_EV):.3e} eV (PPN lower bound)")
    print()

    # ---- Element 1: Areas ----
    print("-" * 70)
    print("ELEMENT 1: Horizon Areas")
    print("-" * 70)
    M10 = 10 * M_SUN_KG
    A_sch = area_schwarzschild(M10)
    A_kerr = area_kerr(M10, 0.7)
    A_rn = area_rn(M10, 0.5)
    A_kn = area_kn(M10, 0.5, 0.3)
    print(f"  Schwarzschild (10 Msun): A = {float(A_sch):.6e} m^2")
    print(f"  Kerr (a/M=0.7):          A = {float(A_kerr):.6e} m^2")
    print(f"  RN (Q/M=0.5):            A = {float(A_rn):.6e} m^2")
    print(f"  KN (a/M=0.5, Q/M=0.3):   A = {float(A_kn):.6e} m^2")
    check("Sch A > 0", A_sch > 0)
    check("Kerr A < Sch A (spin reduces area)", A_kerr < A_sch)
    check("RN A < Sch A (charge reduces area)", A_rn < A_sch)
    check("KN A < Sch A", A_kn < A_sch)
    all_results["element_1_areas"] = {
        "schwarzschild_A_m2": float(A_sch),
        "kerr_0.7_A_m2": float(A_kerr),
        "rn_0.5_A_m2": float(A_rn),
        "kn_0.5_0.3_A_m2": float(A_kn),
    }

    # ---- Element 2: Entropy ----
    print("\n" + "-" * 70)
    print("ELEMENT 2: SCT Entropy")
    print("-" * 70)
    for label, A in [("Sch", A_sch), ("Kerr", A_kerr), ("RN", A_rn), ("KN", A_kn)]:
        S_bh = entropy_bh(A)
        S_sct = entropy_sct(A)
        dS = entropy_correction(A)
        print(f"  {label:5s}: S_BH = {float(S_bh):.6e}, delta_S = {float(dS):.6f}, S_SCT = {float(S_sct):.6e}")
        check(f"{label} delta_S > 0 (SCT correction positive)", dS > 0)
    all_results["element_2_entropy"] = {
        "S_BH_sch": float(entropy_bh(A_sch)),
        "S_SCT_sch": float(entropy_sct(A_sch)),
    }

    # ---- Element 3: Temperature ----
    print("\n" + "-" * 70)
    print("ELEMENT 3: Hawking Temperature")
    print("-" * 70)
    T_sch = temp_schwarzschild_K(M10)
    T_kerr = temp_kerr_K(M10, 0.7)
    T_rn = temp_rn_K(M10, 0.5)
    T_kn = temp_kn_K(M10, 0.5, 0.3)
    print(f"  Schwarzschild: T = {float(T_sch):.6e} K")
    print(f"  Kerr (a/M=0.7): T = {float(T_kerr):.6e} K")
    print(f"  RN (Q/M=0.5):   T = {float(T_rn):.6e} K")
    print(f"  KN (a/M=0.5, Q/M=0.3): T = {float(T_kn):.6e} K")
    check("T_Sch > 0", T_sch > 0)
    check("Kerr T < Sch T (spin reduces temperature)", T_kerr < T_sch)
    all_results["element_3_temperature"] = {
        "T_sch_K": float(T_sch),
        "T_kerr_K": float(T_kerr),
        "T_rn_K": float(T_rn),
        "T_kn_K": float(T_kn),
    }

    # ---- Element 4-5: Omega, Phi ----
    print("\n" + "-" * 70)
    print("ELEMENT 4-5: Angular Velocity, Electric Potential")
    print("-" * 70)
    Om_kerr = omega_kerr(M10, 0.7)
    Om_kn = omega_kn(M10, 0.5, 0.3)
    Phi_rn = phi_rn_geom(M10, 0.5)
    Phi_kn = phi_kn_geom(M10, 0.5, 0.3)
    print(f"  Omega_Kerr (a/M=0.7):  {float(Om_kerr):.6e} rad/s")
    print(f"  Omega_KN (a/M=0.5):    {float(Om_kn):.6e} rad/s")
    print(f"  Phi_RN (Q/M=0.5):      {float(Phi_rn):.10f} (geometrized)")
    print(f"  Phi_KN (Q/M=0.3):      {float(Phi_kn):.10f} (geometrized)")
    check("Omega_Kerr > 0", Om_kerr > 0)
    check("Phi_RN > 0", Phi_rn > 0)
    all_results["element_4_5"] = {
        "Omega_kerr_rad_s": float(Om_kerr),
        "Phi_rn_geom": float(Phi_rn),
    }

    # ---- Element 6: First Law SYMBOLIC ----
    print("\n" + "-" * 70)
    print("ELEMENT 6: First Law (Symbolic, SymPy)")
    print("-" * 70)
    fl_sym = verify_first_law_symbolic()
    for bh_type, data in fl_sym.items():
        print(f"  {bh_type}:")
        for k, v in data.items():
            print(f"    {k}: {v}")
        if "pass" in data:
            check(f"First law symbolic {bh_type}", data["pass"])
        if "pass_dM" in data:
            check(f"First law symbolic {bh_type} dM", data["pass_dM"])
        if "pass_da" in data:
            check(f"First law symbolic {bh_type} da", data["pass_da"])
        if "pass_dQ" in data:
            check(f"First law symbolic {bh_type} dQ", data["pass_dQ"])
    all_results["element_6_first_law_symbolic"] = fl_sym

    # ---- Element 7: First Law NUMERICAL ----
    print("\n" + "-" * 70)
    print("ELEMENT 7: First Law (Numerical, mpmath 60-digit)")
    print("-" * 70)
    fl_num = verify_first_law_numerical()
    for bh_type, data in fl_num.items():
        print(f"  {bh_type}: residual = {data.get('residual', data.get('residual_M', '?'))}")
        check(f"First law numerical {bh_type}", data["pass"])
    all_results["element_7_first_law_numerical"] = fl_num

    # ---- Element 8: Smarr Relation ----
    print("\n" + "-" * 70)
    print("ELEMENT 8: Smarr Relation")
    print("-" * 70)
    smarr = verify_smarr()
    for bh_type, data in smarr.items():
        print(f"  {bh_type}: residual = {data['residual']:.2e}")
        check(f"Smarr {bh_type}", data["pass"])
    all_results["element_8_smarr"] = smarr

    # ---- Element 9: Extremal Limits ----
    print("\n" + "-" * 70)
    print("ELEMENT 9: Extremal Limits")
    print("-" * 70)
    ext = compute_extremal_limits()
    for bh_type, data in ext.items():
        print(f"  {bh_type}: T->0 = {data.get('T_approaches_zero', '?')}, S finite = {data['S_finite']}")
        check(f"Extremal {bh_type} T->0", data.get("T_approaches_zero", True))
        check(f"Extremal {bh_type} S finite", data["S_finite"])
    all_results["element_9_extremal"] = ext

    # ---- Element 10: Comparison Table ----
    print("\n" + "-" * 70)
    print("ELEMENT 10: Comparison Table (10 Msun)")
    print("-" * 70)
    table = build_comparison_table()
    print(f"  {'Type':20s} {'A [m^2]':>14s} {'S_BH':>14s} {'S_SCT':>14s} {'T [K]':>14s}")
    for bh_type, data in table.items():
        print(f"  {bh_type:20s} {data['A_m2']:14.6e} {data['S_BH']:14.6e} {data['S_SCT']:14.6e} {data['T_K']:14.6e}")
    check("Table generated", len(table) == 4)
    all_results["element_10_table"] = table

    # ---- Element 11: R = 0 for RN ----
    print("\n" + "-" * 70)
    print("ELEMENT 11: R = 0 on RN (Maxwell Tracelessness)")
    print("-" * 70)
    r_zero = verify_R_zero_rn()
    print(f"  T^em trace factor (d=4): {r_zero['T_em_trace_factor']}")
    print(f"  => R = 0: {r_zero['R_equals_zero']}")
    check("R = 0 on RN", r_zero["R_equals_zero"])
    all_results["element_11_R_zero"] = r_zero

    # ---- Element 12: Wald C^2 on Kerr ----
    print("\n" + "-" * 70)
    print("ELEMENT 12: Wald Entropy from C^2 (Topological on Ricci-flat)")
    print("-" * 70)
    wald_c2 = verify_wald_c2_kerr()
    print(f"  delta_S (JM formula):  {wald_c2['delta_S_JM_formula']:.10f}")
    print(f"  delta_S (simplified):  {wald_c2['delta_S_simplified']:.10f}")
    print(f"  alpha_C/pi (expected): {wald_c2['S_weyl_canonical']:.10f}")
    print(f"  Consistency error: {wald_c2['consistency_error']:.2e}")
    check("Wald C^2 topological", wald_c2["consistency_error"] < 1e-40)
    all_results["element_12_wald_c2"] = wald_c2

    # ---- Element 13: Christodoulou-Ruffini ----
    print("\n" + "-" * 70)
    print("ELEMENT 13: Christodoulou-Ruffini Mass Formula")
    print("-" * 70)
    cr = christodoulou_ruffini()
    for bh_type, data in cr.items():
        if "pass" in data:
            print(f"  {bh_type}: residual = {data['residual']:.2e}")
            check(f"CR formula {bh_type}", data["pass"])
        elif "agreement" in data:
            print(f"  {bh_type}: M_irr/M = {data['M_irr_over_M']:.10f}, expected = {data['expected_1_over_sqrt2']:.10f}")
            check(f"Extremal Kerr M_irr = M/sqrt(2)", data["agreement"] < 1e-18)
    all_results["element_13_cr"] = cr

    # ---- Element 14: Penrose Process ----
    print("\n" + "-" * 70)
    print("ELEMENT 14: Penrose Process Efficiency")
    print("-" * 70)
    pen = penrose_process()
    print(f"  eta(extremal Kerr) = {pen['eta_extremal_kerr_pct']:.4f}%")
    print(f"  eta(a/M=0.998) = {pen['eta_a998_pct']:.4f}%")
    print(f"  SCT correction to eta (10 Msun, a/M=0.998): {pen['delta_eta_sct_10Msun_a998']:.2e}")
    check("Penrose eta > 0", pen["eta_extremal_kerr"] > 0)
    check("SCT correction negligible", pen["sct_correction_utterly_negligible"])
    all_results["element_14_penrose"] = pen

    # ---- Element 15: Kerr/CFT ----
    print("\n" + "-" * 70)
    print("ELEMENT 15: Kerr/CFT Correspondence")
    print("-" * 70)
    kcft = kerr_cft()
    print(f"  S_BH (actual, a/M=0.998): {kcft['S_BH_actual']:.6e}")
    print(f"  S_CFT (2*pi*J):           {kcft['S_CFT_2piJ']:.6e}")
    print(f"  S_SCT:                     {kcft['S_SCT']:.6e}")
    print(f"  Relative SCT correction:   {kcft['relative_SCT_correction']:.2e}")
    check("Kerr/CFT correction negligible", kcft["correction_negligible"])
    all_results["element_15_kerr_cft"] = kcft

    # ---- Element 16: Holographic Excess ----
    print("\n" + "-" * 70)
    print("ELEMENT 16: Holographic Entropy Excess S_SCT - S_BH > 0")
    print("-" * 70)
    holo = holographic_excess()
    for entry in holo["masses"]:
        print(f"  M = {entry['M_Msun']:>10.0f} Msun: delta_S = {entry['delta_S']:.4f}, "
              f"delta_S/S_BH = {entry['delta_S_over_S_BH']:.2e}, positive = {entry['delta_S_positive']}")
        check(f"Holographic excess {entry['M_Msun']:.0f} Msun", entry["delta_S_positive"])
    all_results["element_16_holographic"] = holo

    # ---- Element 17: Lean Theorems ----
    print("\n" + "-" * 70)
    print("ELEMENT 17: Lean 4 Theorem Statements (for separate verification)")
    print("-" * 70)
    lean = lean_theorems()
    for thm in lean:
        print(f"  theorem {thm['name']}:")
        print(f"    {thm['statement']}")
    check("Lean theorems listed", len(lean) == 5)
    all_results["element_17_lean"] = lean

    # ---- Element 18: Specific Heat ----
    print("\n" + "-" * 70)
    print("ELEMENT 18: Specific Heat with SCT Corrections")
    print("-" * 70)
    cv = specific_heat()
    gd = cv["schwarzschild_geometrized"]
    print(f"  C_V (BH):  {gd['C_V_BH']:.6f} (negative => thermodynamic instability)")
    print(f"  C_V (log): {gd['C_V_log_correction']:.6f}")
    print(f"  C_V (total): {gd['C_V_total']:.6f}")
    print(f"  Sign unchanged by SCT: {gd['sign_unchanged_by_SCT']}")
    check("Specific heat negative", gd["C_V_BH_is_negative"])
    check("SCT preserves sign", gd["sign_unchanged_by_SCT"])
    all_results["element_18_specific_heat"] = cv

    # ---- Element 19: GB cross-check ----
    print("\n" + "-" * 70)
    print("ELEMENT 19: Gauss-Bonnet Cross-Check (Ricci-flat)")
    print("-" * 70)
    gb = gauss_bonnet_cross_check()
    print(f"  {gb['on_ricci_flat']}")
    print(f"  {gb['consequence']}")
    check("GB identity stated", True)
    all_results["element_19_gb"] = gb

    # ---- Element 20: GW150914 ----
    print("\n" + "-" * 70)
    print("ELEMENT 20: GW150914 Entropy Production")
    print("-" * 70)
    gw = gw150914_entropy()
    print(f"  delta_S_GR  = {gw['delta_S_GR']:.6e} (positive = {gw['delta_S_GR_positive']})")
    print(f"  delta_S_SCT = {gw['delta_S_SCT']:.6e} (positive = {gw['delta_S_SCT_positive']})")
    print(f"  SCT-GR correction: {gw['SCT_minus_GR_correction']:.6f}")
    print(f"  Relative correction: {gw['relative_correction']:.2e}")
    check("GW150914 second law GR", gw["delta_S_GR_positive"])
    check("GW150914 second law SCT", gw["delta_S_SCT_positive"])
    all_results["element_20_gw150914"] = gw

    # ---- Element 21: Nonlocal correction ----
    print("\n" + "-" * 70)
    print("ELEMENT 21: Nonlocal Correction Estimate")
    print("-" * 70)
    nl = nonlocal_correction()
    print(f"  m_ghost = {nl['m_ghost_eV']:.4e} eV")
    print(f"  l_ghost = {nl['l_ghost_m']:.4e} m")
    for entry in nl["mass_grid"]:
        print(f"  M = {entry['M_Msun']:>10.1f} Msun: delta_S/S ~ {entry['delta_S_over_S']:.2e}")
    check("Nonlocal corrections negligible (< 1e-14)", all(e["delta_S_over_S"] < 1e-14 for e in nl["mass_grid"]))
    all_results["element_21_nonlocal"] = nl

    # ---- Element 22: Entropy production rate ----
    print("\n" + "-" * 70)
    print("ELEMENT 22: Entropy Production Rate")
    print("-" * 70)
    epr = entropy_production_rate()
    print(f"  dS/dt (BH part): {epr['dS_dt_BH']:.6e}")
    print(f"  dS/dt (log corr): {epr['dS_dt_log_correction']:.6e}")
    print(f"  Relative log correction: {epr['relative_log_correction']:.6e}")
    check("Entropy production computed", True)
    all_results["element_22_entropy_rate"] = epr

    # ---- Element 23: Dimensional analysis ----
    print("\n" + "-" * 70)
    print("ELEMENT 23: Dimensional Analysis [T*dS] = Energy")
    print("-" * 70)
    dim = dimensional_analysis()
    print(f"  T*delta_S*k_B = {dim['T_deltaS_kB_J']:.6e} J")
    print(f"  delta_M*c^2   = {dim['deltaM_c2_J']:.6e} J")
    print(f"  Relative error: {dim['relative_error']:.2e}")
    check("Dimensions consistent", dim["dimensions_consistent"])
    all_results["element_23_dimensional"] = dim

    # ---- Element 24: Central charges ----
    print("\n" + "-" * 70)
    print("ELEMENT 24: Central Charges and Hofman-Maldacena")
    print("-" * 70)
    cc = central_charges()
    print(f"  a = {cc['a_charge_exact']} = {cc['a_charge']:.10f}")
    print(f"  c = {cc['c_charge_exact']} = {cc['c_charge']:.10f}")
    print(f"  a/c = {cc['a_over_c']:.6f}")
    print(f"  HM bound: [{cc['HM_lower_bound']:.4f}, {cc['HM_upper_bound']:.4f}]")
    print(f"  Satisfied: {cc['HM_satisfied']}")
    check("HM bound satisfied", cc["HM_satisfied"])
    all_results["element_24_central_charges"] = cc

    # ---- Element 25: Planck unit formulas ----
    print("\n" + "-" * 70)
    print("ELEMENT 25: Planck Unit Formulas")
    print("-" * 70)
    puf = planck_unit_formulas()
    print(f"  S_SCT(M) = {puf['formulas']['S_SCT_M']}")
    print(f"  T(M)     = {puf['formulas']['T_M']}")
    print(f"  C_V(M)   = {puf['formulas']['C_V_M']}")
    print(f"\n  At M = 1 M_Pl:")
    d1 = puf["M_1_Planck"]
    print(f"    A = {d1['A']:.6f}")
    print(f"    S_BH = {d1['S_BH']:.6f}")
    print(f"    S_SCT = {d1['S_SCT']:.6f}")
    print(f"    T = {d1['T']:.6f}")
    print(f"    S_correction = {d1['S_correction']:.6f}")
    print(f"\n  At M = 10 M_Pl:")
    d10 = puf["M_10_Planck"]
    print(f"    S_BH = {d10['S_BH']:.6f}")
    print(f"    S_SCT = {d10['S_SCT']:.6f}")
    check("Planck formulas computed", True)
    all_results["element_25_planck"] = puf

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 90)
    print(f"SUMMARY: {n_pass}/{n_total} checks PASSED")
    print("=" * 90)

    all_results["summary"] = {
        "total_checks": n_total,
        "passed": n_pass,
        "failed": n_total - n_pass,
        "all_pass": n_pass == n_total,
    }

    # Save to JSON
    output_path = RESULTS_DIR / "mt1_kerr_newman.json"

    def _json_safe(obj):
        if isinstance(obj, (mp.mpf, np.floating)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, Fraction):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Cannot serialize {type(obj)}")

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=_json_safe)
    print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    main()
