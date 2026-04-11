#!/usr/bin/env python3
"""
MT-1 Ghost Suppression Theorem: Dual mechanism analysis for the second law
of black hole thermodynamics in higher-derivative gravity.

Model-independent result: for ANY quadratic gravity theory with real ghost mass m > 0,
the ghost contribution to the second law is suppressed by two independent mechanisms:
  I.   Boltzmann (thermal):     exp(-m/T_H)
  II.  Yukawa (spatial):        exp(-m*r_H)

Note: The gravitational Schwinger effect (pair production by surface gravity) gives
exponent m/(2*T_H) = Boltzmann/2, so it is NOT independent — it is a weaker version
of the thermal mechanism (Gibbons-Hawking 1977).

If the ghost is a fakeon (Anselmi 2017), it has zero physical DOF and cannot
thermalize — making this analysis unnecessary but the second law automatically safe.
This analysis applies as a FALLBACK in case the fakeon prescription does not extend
to curved spacetime thermal baths.

Critical mass: M_crit = M_Pl^2 / (8*pi*m)
All observed BH have M/M_crit > 10^9.

Covers: Schwarzschild, Kerr, Reissner-Nordstrom, Kerr-Newman, de Sitter horizons.

Author: David Alfyorov
"""
from __future__ import annotations

import json
import math
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
from scipy import constants as const

DPS = 60
mp.mp.dps = DPS

# ============================================================
# 1. Physical constants (from scipy.constants + CODATA 2022)
# ============================================================
G_SI = mp.mpf(str(const.G))                    # m^3 kg^-1 s^-2
HBAR_SI = mp.mpf(str(const.hbar))              # J s
C_SI = mp.mpf(str(const.c))                    # m/s
K_B_SI = mp.mpf(str(const.k))                  # J/K
EV_J = mp.mpf(str(const.eV))                   # J per eV
M_SUN_KG = mp.mpf("1.98892e30")                # IAU 2015 nominal

# Planck units
L_PL = mp.sqrt(HBAR_SI * G_SI / C_SI**3)      # Planck length [m]
M_PL_KG = mp.sqrt(HBAR_SI * C_SI / G_SI)      # Planck mass [kg]
M_PL_EV = M_PL_KG * C_SI**2 / EV_J            # Planck mass [eV]
T_PL_K = M_PL_KG * C_SI**2 / K_B_SI           # Planck temperature [K]

# SCT ghost parameters
Z_L = mp.mpf("1.2807")           # |z_L| for physical ghost (from MR-2)
Z_0 = mp.mpf("2.41483889")       # z_0 for Euclidean pole (from NT-4a)
LAMBDA_PPN_EV = mp.mpf("2.38e-3") # PPN lower bound [eV]

# Ghost masses
M_GHOST_OVER_LAMBDA = mp.sqrt(Z_L)             # = 1.1318...
M2_LOCAL_OVER_LAMBDA = mp.sqrt(mp.mpf(60)/13)  # = 2.1483... (local Yukawa)


def m_ghost_eV(Lambda_eV):
    """Physical ghost mass [eV]."""
    return M_GHOST_OVER_LAMBDA * Lambda_eV


# ============================================================
# 2. Hawking temperature
# ============================================================
def T_H_eV(M_kg):
    """Hawking temperature [eV] for Schwarzschild BH."""
    return HBAR_SI * C_SI**3 / (8 * mp.pi * G_SI * M_kg) / EV_J


def T_H_K(M_kg):
    """Hawking temperature [K] for Schwarzschild BH."""
    return HBAR_SI * C_SI**3 / (8 * mp.pi * G_SI * M_kg * K_B_SI)


def r_H_m(M_kg):
    """Schwarzschild horizon radius [m]."""
    return 2 * G_SI * M_kg / C_SI**2


# ============================================================
# 3-5. Three suppression mechanisms + critical mass
# ============================================================
def boltzmann_exponent(M_kg, Lambda_eV):
    """Exponent in Boltzmann suppression: m_ghost / T_H."""
    m = m_ghost_eV(Lambda_eV)
    T = T_H_eV(M_kg)
    return m / T


def schwinger_exponent(M_kg, Lambda_eV):
    """Gravitational Schwinger exponent: m/(2*T_H) = Boltzmann/2.

    The gravitational Schwinger effect (pair production by surface gravity kappa)
    gives rate ~ exp(-m/kappa) = exp(-m*4*pi*r_H) = exp(-m/(2*T_H)).
    This equals Boltzmann/2, so it is NOT an independent mechanism.
    Kept for completeness and comparison.
    """
    return boltzmann_exponent(M_kg, Lambda_eV) / 2


def yukawa_exponent(M_kg, Lambda_eV):
    """Exponent in Yukawa spatial decay: m_ghost * r_H."""
    m = m_ghost_eV(Lambda_eV) * EV_J / (HBAR_SI * C_SI)  # [1/m]
    rH = r_H_m(M_kg)
    return m * rH


def M_crit_kg(Lambda_eV):
    """Critical mass [kg] where m_ghost = T_H (Boltzmann exponent = 1)."""
    m = m_ghost_eV(Lambda_eV) * EV_J  # [J]
    # T_H = hbar*c^3/(8*pi*G*M*k_B), in [eV]: T_H_eV = hbar*c^3/(8*pi*G*M) / eV_J
    # m_ghost/T_H = 1 => M = hbar*c^3/(8*pi*G*m_ghost_J) ... wait
    # m/T = m * 8*pi*G*M / (hbar*c^3) * eV_J / eV_J ... in eV units:
    # m_eV / T_eV = m_eV * 8*pi*G*M*eV_J / (hbar*c^3)
    # Set = 1: M = hbar*c^3 / (8*pi*G*m_eV*eV_J)
    return HBAR_SI * C_SI**3 / (8 * mp.pi * G_SI * m * 1)  # m already in J


def M_crit_Msun(Lambda_eV):
    """Critical mass in solar masses."""
    return M_crit_kg(Lambda_eV) / M_SUN_KG


# ============================================================
# 6-7. Ghost partition function and energy density
# ============================================================
def ghost_number_density(M_kg, Lambda_eV):
    """Ghost number density n_ghost [m^-3] in thermal equilibrium (non-relativistic)."""
    m_eV = m_ghost_eV(Lambda_eV)
    T_eV = T_H_eV(M_kg)
    x = float(m_eV / T_eV)
    if x > 500:
        return mp.mpf(0)  # Underflow protection
    n_dof = 5  # spin-2 massive: 5 polarizations
    # n = n_dof * (m*T/(2*pi))^{3/2} * exp(-m/T) in natural units [eV^3]
    # Convert to [m^-3]: multiply by (eV / (hbar*c))^3
    factor = (EV_J / (HBAR_SI * C_SI))**3
    n_nat = n_dof * (m_eV * T_eV / (2 * mp.pi))**mp.mpf("1.5") * mp.exp(-x)
    return n_nat * factor


def ghost_energy_density(M_kg, Lambda_eV):
    """Ghost energy density rho_ghost [J/m^3] = m * n_ghost."""
    n = ghost_number_density(M_kg, Lambda_eV)
    m_J = m_ghost_eV(Lambda_eV) * EV_J
    return m_J * n


def hawking_energy_density(M_kg):
    """Stefan-Boltzmann energy density for Hawking radiation [J/m^3]."""
    T_J = T_H_eV(M_kg) * EV_J
    g_eff = mp.mpf("3.5")  # photons + gravitons (rough)
    return mp.pi**2 * g_eff * T_J**4 / (30 * (HBAR_SI * C_SI)**3)


# ============================================================
# 8. Quantitative violation bound
# ============================================================
def violation_bound(M_kg, Lambda_eV):
    """Upper bound on |dS_ghost/dt| / |dS_matter/dt|.

    Returns log10 of the bound (since the number itself is un-representable).
    Bound: C * (m/T)^{5/2} * exp(-m/T), C = 5*(2*pi)^{-3/2}.
    """
    x = float(boltzmann_exponent(M_kg, Lambda_eV))
    if x > 1e15:
        # Use log directly: log10(C * x^{5/2} * exp(-x)) = log10(C) + 2.5*log10(x) - x*log10(e)
        log10_C = float(mp.log10(5 * (2*mp.pi)**mp.mpf("-1.5")))
        return log10_C + 2.5 * math.log10(x) - x * math.log10(math.e)
    C = 5 * (2 * mp.pi)**mp.mpf("-1.5")
    val = C * mp.power(x, mp.mpf("2.5")) * mp.exp(-x)
    if val > 0:
        return float(mp.log10(val))
    return -1e30  # Effectively -infinity


# ============================================================
# 14-17. Kerr / RN / KN temperatures
# ============================================================
def T_H_kerr_eV(M_kg, a_over_M):
    """Hawking temperature [eV] for Kerr BH with spin parameter a/M."""
    # In geometrized units: a = a_over_M * G*M/c^2
    GM = G_SI * M_kg / C_SI**2  # [m]
    a = a_over_M * GM             # [m]
    r_plus = GM + mp.sqrt(GM**2 - a**2)
    r_minus = GM - mp.sqrt(GM**2 - a**2)
    # T_H = hbar * (r+ - r-) / (4*pi*(r+^2 + a^2)) * c^3 / (k_B) ... in eV:
    # T_H = hbar*c^3*(r+ - r-) / (4*pi*(r+^2 + a^2)) / eV_J
    kappa = (r_plus - r_minus) / (2 * (r_plus**2 + a**2))  # surface gravity * c^2
    return HBAR_SI * C_SI * kappa / (2 * mp.pi) / EV_J


def T_H_rn_eV(M_kg, Q_over_M):
    """Hawking temperature [eV] for RN BH with charge parameter Q/M.

    Q_over_M is dimensionless ratio Q/(M*sqrt(G)) in geometrized units.
    """
    GM = G_SI * M_kg / C_SI**2  # [m]
    Q = Q_over_M * GM             # [m] (Q in geometrized)
    r_plus = GM + mp.sqrt(GM**2 - Q**2)
    r_minus = GM - mp.sqrt(GM**2 - Q**2)
    kappa = (r_plus - r_minus) / (2 * r_plus**2)
    return HBAR_SI * C_SI * kappa / (2 * mp.pi) / EV_J


def T_H_kn_eV(M_kg, a_over_M, Q_over_M):
    """Hawking temperature [eV] for Kerr-Newman BH."""
    GM = G_SI * M_kg / C_SI**2
    a = a_over_M * GM
    Q = Q_over_M * GM
    disc = GM**2 - a**2 - Q**2
    if disc <= 0:
        return mp.mpf(0)  # Extremal or super-extremal
    r_plus = GM + mp.sqrt(disc)
    r_minus = GM - mp.sqrt(disc)
    kappa = (r_plus - r_minus) / (2 * (r_plus**2 + a**2))
    return HBAR_SI * C_SI * kappa / (2 * mp.pi) / EV_J


# ============================================================
# 32. De Sitter horizon
# ============================================================
def T_dS_eV(H_eV):
    """De Sitter temperature [eV] for Hubble parameter H [eV]."""
    return H_eV / (2 * mp.pi)


# ============================================================
# Main computation
# ============================================================
def main():
    print("=" * 90)
    print("MT-1 GHOST SUPPRESSION THEOREM: DUAL MECHANISM ANALYSIS")
    print("=" * 90)

    Lambda = LAMBDA_PPN_EV
    m_g = m_ghost_eV(Lambda)
    print(f"\nGhost parameters (at Lambda_PPN = {float(Lambda):.3e} eV):")
    print(f"  m_ghost/Lambda = sqrt(|z_L|) = {float(M_GHOST_OVER_LAMBDA):.6f}")
    print(f"  m_ghost = {float(m_g):.4e} eV = {float(m_g*1e3):.4f} meV")
    print(f"  z_L = {float(Z_L):.4f} (physical ghost pole)")

    # ---- Mass grid ----
    M_grid_Msun = np.logspace(-8, 10, 30)
    M_grid_kg = [mp.mpf(str(m)) * M_SUN_KG for m in M_grid_Msun]

    # ---- Section 2-4: Two independent mechanisms (Schwinger = Boltzmann/2) ----
    print(f"\n{'='*70}")
    print("TWO INDEPENDENT SUPPRESSION MECHANISMS (Schwarzschild)")
    print("(Gravitational Schwinger = Boltzmann/2, NOT independent)")
    print(f"{'='*70}")
    print(f"{'M/M_sun':>12} {'Boltzmann':>14} {'Schwngr=Bz/2':>14} {'Yukawa':>14} {'Violation':>14}")

    results_grid = []
    for i, (m_msun, m_kg) in enumerate(zip(M_grid_Msun, M_grid_kg)):
        bz = boltzmann_exponent(m_kg, Lambda)
        sw = schwinger_exponent(m_kg, Lambda)
        yk = yukawa_exponent(m_kg, Lambda)
        vb = violation_bound(m_kg, Lambda)

        results_grid.append({
            "M_Msun": float(m_msun),
            "boltzmann_exp": float(bz),
            "schwinger_exp": float(sw),
            "yukawa_exp": float(yk),
            "violation_log10": vb,
        })

        if i % 5 == 0 or i == len(M_grid_Msun) - 1:
            print(f"  {m_msun:12.2e} {float(bz):14.4e} {float(sw):14.4e} "
                  f"{float(yk):14.4e} {vb:14.1f}")

    # ---- Section 5: Critical mass ----
    print(f"\n{'='*70}")
    print("CRITICAL MASS TABLE")
    print(f"{'='*70}")
    Lambda_values = [1e-3, 1, 1e6, 1e12, 1e19]
    print(f"{'Lambda [eV]':>14} {'M_crit [kg]':>14} {'M_crit/M_sun':>14} {'M_crit < 3 M_sun?':>18}")

    crit_table = []
    for lam in Lambda_values:
        lam_mp = mp.mpf(str(lam))
        mc_kg = M_crit_kg(lam_mp)
        mc_msun = mc_kg / M_SUN_KG
        safe = "YES" if mc_msun < 3 else "NO"
        print(f"  {lam:14.1e} {float(mc_kg):14.4e} {float(mc_msun):14.4e} {safe:>18}")
        crit_table.append({
            "Lambda_eV": lam,
            "M_crit_kg": float(mc_kg),
            "M_crit_Msun": float(mc_msun),
            "below_3Msun": safe == "YES",
        })

    # ---- Section 6-7: Partition function ----
    print(f"\n{'='*70}")
    print("GHOST THERMODYNAMICS")
    print(f"{'='*70}")

    thermo_masses = [1, 3, 10, 1e6, 1e10]
    thermo_data = []
    for m_msun in thermo_masses:
        m_kg = mp.mpf(str(m_msun)) * M_SUN_KG
        bz = boltzmann_exponent(m_kg, Lambda)
        n_g = ghost_number_density(m_kg, Lambda)
        rho_g = ghost_energy_density(m_kg, Lambda)
        rho_h = hawking_energy_density(m_kg)
        ratio = float(rho_g / rho_h) if rho_h > 0 and rho_g > 0 else 0.0

        T_eV = T_H_eV(m_kg)
        T_K = T_H_K(m_kg)

        thermo_data.append({
            "M_Msun": m_msun,
            "T_H_eV": float(T_eV),
            "T_H_K": float(T_K),
            "m_over_T": float(bz),
            "n_ghost_m3": float(n_g),
            "rho_ghost_Jm3": float(rho_g),
            "rho_hawking_Jm3": float(rho_h),
            "rho_ratio": ratio,
        })
        print(f"  M = {m_msun:.0e} M_sun: T_H = {float(T_eV):.3e} eV = {float(T_K):.3e} K, "
              f"m/T = {float(bz):.3e}, rho_ghost/rho_Hawking = {ratio:.3e}")

    # ---- Section 10: Observational safety ----
    print(f"\n{'='*70}")
    print("OBSERVATIONAL SAFETY MARGIN (5 real BHs)")
    print(f"{'='*70}")

    observed_bhs = [
        ("Cyg X-1", 21.2, 0.998),      # spinning
        ("GW150914 rem", 62.2, 0.67),   # merger remnant
        ("M87*", 6.5e9, 0.9),           # supermassive
        ("Sgr A*", 4.15e6, 0.5),        # our galactic center
        ("TON 618", 6.6e10, 0.0),       # ultra-massive
    ]

    mc = M_crit_Msun(Lambda)
    obs_data = []
    for name, m_msun, a_over_M in observed_bhs:
        m_kg = mp.mpf(str(m_msun)) * M_SUN_KG
        margin = mp.mpf(str(m_msun)) / mc
        bz = boltzmann_exponent(m_kg, Lambda)
        vb = violation_bound(m_kg, Lambda)

        # Kerr temperature
        T_kerr = T_H_kerr_eV(m_kg, mp.mpf(str(a_over_M)))
        T_sch = T_H_eV(m_kg)
        T_ratio = float(T_kerr / T_sch) if T_sch > 0 else 0

        obs_data.append({
            "name": name,
            "M_Msun": m_msun,
            "a_over_M": a_over_M,
            "margin_over_Mcrit": float(margin),
            "boltzmann_exp": float(bz),
            "violation_log10": vb,
            "T_kerr_over_T_sch": T_ratio,
        })
        print(f"  {name:16s}: M = {m_msun:.1e} M_sun, a/M = {a_over_M}, "
              f"M/M_crit = {float(margin):.2e}, violation < 10^{{{vb:.0f}}}, "
              f"T_Kerr/T_Sch = {T_ratio:.4f}")

    # ---- Section 14-16: Kerr/RN/KN temperatures ----
    print(f"\n{'='*70}")
    print("KERR / RN / KN: SCHWARZSCHILD IS WORST CASE")
    print(f"{'='*70}")

    M_ref = 10 * M_SUN_KG
    T_sch_ref = T_H_eV(M_ref)

    a_grid = np.linspace(0, 0.998, 20)
    kerr_data = []
    for a_over_M in a_grid:
        T_k = T_H_kerr_eV(M_ref, mp.mpf(str(a_over_M)))
        ratio = float(T_k / T_sch_ref)
        kerr_data.append({"a_over_M": float(a_over_M), "T_ratio": ratio})

    print(f"  Kerr (M = 10 M_sun): T_Kerr/T_Sch from {kerr_data[0]['T_ratio']:.4f} "
          f"(a=0) to {kerr_data[-1]['T_ratio']:.4f} (a=0.998)")
    print(f"  Maximum T_ratio = {max(d['T_ratio'] for d in kerr_data):.6f} (should be ≤ 1)")

    Q_grid = np.linspace(0, 0.998, 20)
    rn_data = []
    for Q_over_M in Q_grid:
        T_rn = T_H_rn_eV(M_ref, mp.mpf(str(Q_over_M)))
        ratio = float(T_rn / T_sch_ref)
        rn_data.append({"Q_over_M": float(Q_over_M), "T_ratio": ratio})

    print(f"  RN (M = 10 M_sun): T_RN/T_Sch from {rn_data[0]['T_ratio']:.4f} "
          f"(Q=0) to {rn_data[-1]['T_ratio']:.4f} (Q=0.998)")

    # KN 2D grid
    kn_data = []
    for a in np.linspace(0, 0.95, 10):
        for Q in np.linspace(0, 0.95, 10):
            if a**2 + Q**2 >= 0.99:
                continue
            T_kn = T_H_kn_eV(M_ref, mp.mpf(str(a)), mp.mpf(str(Q)))
            kn_data.append({
                "a_over_M": float(a),
                "Q_over_M": float(Q),
                "T_ratio": float(T_kn / T_sch_ref),
            })

    max_kn = max(d["T_ratio"] for d in kn_data)
    print(f"  KN 2D grid: max T_KN/T_Sch = {max_kn:.6f} (should be ≤ 1)")
    assert max_kn <= 1.0001, f"FAILURE: T_KN > T_Sch at some (a,Q)! max = {max_kn}"
    print("  VERIFIED: Schwarzschild is the worst case (hottest BH for given M)")

    # ---- Section 23: R=0 for RN ----
    print(f"\n{'='*70}")
    print("R^2 VANISHING: Maxwell T^mu_mu = 0 => R = 0 for RN/KN")
    print(f"{'='*70}")
    print("  Electromagnetic stress-energy: T^mu_mu = 0 (traceless)")
    print("  Einstein equation: R = -8*pi*G*T^mu_mu = 0")
    print("  Therefore: delta_S_R2 = 2*alpha_R * R * (...) = 0 on RN/KN")
    print("  Note: R_mn != 0 (Ricci tensor is nonzero), but R^2 in action is the SCALAR")

    # ---- Section 29: Self-consistency of T_H ----
    print(f"\n{'='*70}")
    print("SELF-CONSISTENCY: GHOST METRIC CORRECTION AT HORIZON")
    print(f"{'='*70}")
    for m_msun in [1, 10, 1e6]:
        m_kg = mp.mpf(str(m_msun)) * M_SUN_KG
        yk = yukawa_exponent(m_kg, Lambda)
        print(f"  M = {m_msun:.0e} M_sun: ghost metric correction ~ exp(-{float(yk):.3e}) ~ 0")
    print("  T_H = kappa/(2*pi) is robust to non-perturbatively small accuracy")

    # ---- Section 31: One-loop mass correction ----
    print(f"\n{'='*70}")
    print("ONE-LOOP GHOST MASS CORRECTION")
    print(f"{'='*70}")
    m_over_Mpl = float(m_g / M_PL_EV)
    delta_m_over_m = m_over_Mpl**2
    print(f"  m_ghost/M_Pl = {m_over_Mpl:.4e}")
    print(f"  delta_m/m ~ (m/M_Pl)^2 = {delta_m_over_m:.4e}")
    print(f"  Free-field partition function justified to accuracy {delta_m_over_m:.1e}")

    # ---- Section 32: De Sitter horizon ----
    print(f"\n{'='*70}")
    print("DE SITTER (COSMOLOGICAL) HORIZON")
    print(f"{'='*70}")
    H0_eV = mp.mpf("1.44e-33")  # current Hubble: ~67 km/s/Mpc ~ 1.44e-33 eV
    T_ds = T_dS_eV(H0_eV)
    m_over_T_ds = m_g / T_ds
    print(f"  H_0 ~ {float(H0_eV):.2e} eV")
    print(f"  T_dS = H/(2*pi) = {float(T_ds):.3e} eV")
    print(f"  m_ghost/T_dS = {float(m_over_T_ds):.3e}")
    print(f"  Suppression: exp(-{float(m_over_T_ds):.2e})")
    print(f"  De Sitter horizon is EVEN MORE ghost-safe than BH")

    # ---- Section 33: Page time ----
    print(f"\n{'='*70}")
    print("PAGE TIME TRANSITION")
    print(f"{'='*70}")
    # Page time: t_Page ~ G^2 M^3 / (hbar c^4)
    # At t_Page, M has evaporated to ~M_Pl
    # m_ghost/T_H(M_Pl) ~ m_ghost*8*pi*G*M_Pl = 8*pi*m_ghost/M_Pl
    bz_at_Mpl = float(8 * mp.pi * m_g / M_PL_EV)
    print(f"  At M = M_Pl: m_ghost/T_H = 8*pi * m_ghost/M_Pl = {bz_at_Mpl:.4e}")
    print(f"  Ghost becomes thermally active when m/T < 1, i.e. M < M_crit")
    print(f"  M_crit/M_Pl = M_Pl/(8*pi*m_ghost) = {float(M_PL_EV/(8*mp.pi*m_g)):.4e}")
    print(f"  Transition is smooth (exponential crossover), not abrupt")

    # ---- Summary ----
    print(f"\n{'='*90}")
    print("SUMMARY")
    print(f"{'='*90}")
    mc_msun_val = float(M_crit_Msun(Lambda))
    print(f"  Ghost mass: m = {float(m_g):.4e} eV")
    print(f"  Critical mass: M_crit = {mc_msun_val:.4e} M_sun = {float(M_crit_kg(Lambda)):.4e} kg")
    print(f"  Lightest observed BH: ~3 M_sun")
    print(f"  Safety margin: 3 / {mc_msun_val:.2e} = {3/mc_msun_val:.2e}")
    print(f"  Violation bound (3 M_sun): < 10^{{{violation_bound(3*M_SUN_KG, Lambda):.0f}}}")
    print(f"  Violation bound (10 M_sun): < 10^{{{violation_bound(10*M_SUN_KG, Lambda):.0f}}}")
    print(f"  Schwarzschild is worst case: Kerr/RN/KN temperatures are LOWER")
    print(f"  De Sitter: m/T_dS ~ {float(m_over_T_ds):.1e} (even more suppressed)")
    print(f"  Schwinger = Boltzmann/2 (NOT independent, corrected from v1)")
    print(f"  DUAL mechanisms: Boltzmann (thermal) + Yukawa (spatial)")
    print(f"  Fakeon note: if ghost = fakeon (0 DOF), analysis is unnecessary (2nd law auto-safe)")
    print(f"  This analysis is a FALLBACK in case fakeon doesn't extend to curved thermal baths")
    print(f"  VERDICT: Second law holds for ALL observed BH + cosmological horizon")

    # ---- Save JSON ----
    output = {
        "description": "MT-1 Ghost Suppression Theorem: Dual mechanism analysis (Boltzmann + Yukawa; Schwinger = Boltzmann/2, not independent)",
        "Lambda_PPN_eV": float(Lambda),
        "m_ghost_eV": float(m_g),
        "m_ghost_over_Lambda": float(M_GHOST_OVER_LAMBDA),
        "z_L": float(Z_L),
        "M_crit_kg": float(M_crit_kg(Lambda)),
        "M_crit_Msun": mc_msun_val,
        "grid_results": results_grid,
        "critical_mass_table": crit_table,
        "thermodynamics": thermo_data,
        "observed_bhs": obs_data,
        "kerr_temperature": kerr_data,
        "rn_temperature": rn_data,
        "kn_temperature_2d": kn_data,
        "de_sitter": {
            "H0_eV": float(H0_eV),
            "T_dS_eV": float(T_ds),
            "m_over_T_dS": float(m_over_T_ds),
        },
        "page_time": {
            "m_over_T_at_Mpl": bz_at_Mpl,
            "Mcrit_over_Mpl": float(M_PL_EV / (8 * mp.pi * m_g)),
        },
        "self_consistency": {
            "delta_m_over_m": delta_m_over_m,
            "m_over_Mpl": m_over_Mpl,
        },
    }

    out_dir = Path(__file__).resolve().parent.parent / "fnd1_data"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "mt1_ghost_suppression.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
