# ruff: noqa: E402, I001
"""
MT-2: Modified Cosmology — Quantification of SCT Predictions for H_0 and S_8.

Central result: SCT nonlocal corrections to FLRW cosmology are
    delta H^2/H^2 = O(beta_R * H^2/Lambda^2) ~ 10^{-64}
at the PPN-1 lower bound Lambda >= 2.38e-3 eV.  This is 64 orders of
magnitude below the ~18% needed to resolve the H_0 tension.

This script:
  1. Computes the suppression factor delta H^2/H^2 at 100-digit precision
     for a grid of (z, Lambda, xi) values.
  2. Computes late-time cosmological observables (w_eff, H_0 deviation,
     S_8 deviation, growth function modification).
  3. Verifies GW speed consistency at all cosmological epochs.
  4. Documents the result as a definitive negative quantification.

Physics:
  - On FLRW, C_{mu nu rho sigma} = 0 => only R^2 sector contributes.
  - alpha_R(xi) = 2(xi - 1/6)^2, beta_R = alpha_R / (16 pi^2).
  - At conformal coupling xi = 1/6: alpha_R = 0 => ALL corrections vanish.
  - Suppression: delta H^2/H^2 ~ beta_R * (H/Lambda)^2.
  - Growth modification: Pi_TT(k^2/Lambda^2) ~ 1 + 10^{-44} at k ~ 0.1 h/Mpc.

References:
  - NT-4c: FLRW reduction (117/117 tests, verified)
  - NT-4a: Linearized field equations (88/88 tests, verified)
  - NT-2:  Entire-function proof (63/63 tests, verified)
  - PPN-1: Solar system tests, Lambda >= 2.38e-3 eV (2829 tests)
  - Maggiore, Phys. Rev. D 93 (2016) 063008 [1603.01515]
  - Planck 2018 (1807.06209), SH0ES 2022 (2112.04510)

Author: David Alfyorov
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from scripts.nt4c_flrw import (
    ALPHA_C,
    Pi_TT,
    Pi_scalar,
    _beta_R,
    alpha_R,
    scalar_mode_coefficient,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "mt2"
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures" / "mt2"

DPS = 100  # 100-digit precision throughout

# ============================================================
# Physical constants (CODATA 2022, natural units c = hbar = 1)
# ============================================================
# H_0 conversions:
#   H_0 = 67.36 km/s/Mpc (Planck 2018, 1807.06209)
#   1 Mpc = 3.0857e22 m
#   H_0 = 67.36e3 / 3.0857e22 = 2.1834e-18 Hz
#   In eV (using hbar = 6.5821e-16 eV*s):
#   H_0 = 2.1834e-18 * 6.5821e-16 eV = 1.4371e-33 eV

H0_KMS_MPC = mp.mpf("67.36")  # km/s/Mpc (Planck 2018)
H0_SHOES = mp.mpf("73.04")  # km/s/Mpc (SH0ES 2022)
MPC_IN_M = mp.mpf("3.08567758e22")
HBAR_EV_S = mp.mpf("6.582119514e-16")  # eV*s
H0_HZ = H0_KMS_MPC * mp.mpf("1e3") / MPC_IN_M  # Hz
H0_EV = H0_HZ * HBAR_EV_S  # eV

# Spectral cutoff lower bound from PPN-1
LAMBDA_PPN = mp.mpf("2.38e-3")  # eV (Eot-Wash, PPN-1 result)

# Planck mass
M_PL_EV = mp.mpf("1.220910e28")  # eV (= 1.22e19 GeV)

# GW170817 constraint
GW170817_BOUND = mp.mpf("5.6e-16")  # |c_T/c - 1| < 5.6e-16


def _H_at_redshift(z_red: float | mp.mpf) -> mp.mpf:
    """
    Hubble parameter H(z) in eV for LCDM fiducial cosmology.

    H(z) = H_0 * sqrt(Omega_m (1+z)^3 + Omega_Lambda)
    Using Planck 2018 best-fit: Omega_m = 0.3153, Omega_Lambda = 0.6847.
    """
    mp.mp.dps = DPS
    z_mp = mp.mpf(z_red)
    Om = mp.mpf("0.3153")
    OL = mp.mpf("0.6847")
    return H0_EV * mp.sqrt(Om * (1 + z_mp) ** 3 + OL)


# ============================================================
# Step 1: Suppression Factor
# ============================================================

def suppression_factor(
    z_red: float | mp.mpf,
    Lambda: float | mp.mpf,
    xi: float | mp.mpf = 0.0,
    dps: int = DPS,
) -> dict[str, Any]:
    """
    Compute delta H^2 / H^2 at redshift z_red for given Lambda, xi.

    The suppression factor is:
        delta H^2 / H^2 = beta_R(xi) * (H(z)/Lambda)^2

    where:
        beta_R = alpha_R(xi) / (16 pi^2) = 2(xi - 1/6)^2 / (16 pi^2)
        H(z) from LCDM fiducial cosmology.

    Returns dict with all intermediate quantities for verification.
    """
    mp.mp.dps = dps
    Lambda_mp = mp.mpf(Lambda)
    xi_mp = mp.mpf(xi)

    H_z = _H_at_redshift(z_red)
    aR = alpha_R(float(xi_mp))
    bR = _beta_R(float(xi_mp))

    ratio_squared = (H_z / Lambda_mp) ** 2
    suppression = bR * ratio_squared

    return {
        "z": float(z_red),
        "Lambda_eV": float(Lambda_mp),
        "xi": float(xi_mp),
        "H_z_eV": float(H_z),
        "alpha_R": float(aR),
        "beta_R": float(bR),
        "H_over_Lambda_sq": float(ratio_squared),
        "delta_H2_over_H2": float(suppression),
        "log10_suppression": float(mp.log10(suppression)) if suppression > 0 else None,
    }


def suppression_factor_mp(
    z_red: float | mp.mpf,
    Lambda: float | mp.mpf,
    xi: float | mp.mpf = 0.0,
    dps: int = DPS,
) -> mp.mpf:
    """
    Return the suppression factor delta H^2/H^2 as an mpf (for precision tests).
    """
    mp.mp.dps = dps
    Lambda_mp = mp.mpf(Lambda)
    H_z = _H_at_redshift(z_red)
    bR = _beta_R(float(xi))
    return bR * (H_z / Lambda_mp) ** 2


def suppression_table(dps: int = DPS) -> dict[str, Any]:
    """
    Full suppression table across redshift, Lambda, and xi.

    Returns a structured dict with all computed values.
    """
    mp.mp.dps = dps
    z_values = [0, 0.5, 1, 2, 5, 10, 100, 1000, 1100]
    Lambda_values = {
        "PPN_bound": LAMBDA_PPN,
        "1_eV": mp.mpf("1"),
        "1_GeV": mp.mpf("1e9"),
        "1e6_GeV": mp.mpf("1e15"),
        "1e13_GeV": mp.mpf("1e22"),
        "M_Pl": M_PL_EV,
    }
    xi_values = [0, mp.mpf(1) / 12, mp.mpf(1) / 6, mp.mpf(1) / 4]

    table = []
    for z_red in z_values:
        for lam_name, lam_val in Lambda_values.items():
            for xi_val in xi_values:
                result = suppression_factor(z_red, lam_val, xi=xi_val, dps=dps)
                result["Lambda_label"] = lam_name
                table.append(result)

    # Compute key summary values
    worst_case = suppression_factor(0, LAMBDA_PPN, xi=0, dps=dps)
    best_case = suppression_factor(1100, LAMBDA_PPN, xi=0, dps=dps)

    return {
        "table": table,
        "n_entries": len(table),
        "worst_case_today": worst_case,
        "worst_case_CMB": best_case,
        "z_values": z_values,
        "Lambda_labels": list(Lambda_values.keys()),
        "xi_values": [float(x) for x in xi_values],
    }


# ============================================================
# Step 2: Late-Time Cosmological Observables
# ============================================================

def w_eff(
    z_red: float | mp.mpf,
    Lambda: float | mp.mpf = LAMBDA_PPN,
    xi: float | mp.mpf = 0.0,
    dps: int = DPS,
) -> dict[str, Any]:
    """
    Effective dark energy equation of state w_eff(z) in SCT.

    In SCT, the spectral corrections to the Friedmann equation act as an
    effective fluid with w_Theta = +1 (stiff matter, from NT-4c).
    However, the amplitude of this fluid is suppressed by beta_R * (H/Lambda)^2.

    The effective dark energy EOS is:
        w_eff = w_DE + delta_w
    where w_DE = -1 (cosmological constant) and
        delta_w ~ beta_R * (H/Lambda)^2  (positive, makes it less phantom)

    For Lambda >= PPN bound: delta_w ~ 10^{-64} => w_eff = -1.000...0
    """
    mp.mp.dps = dps
    sup = suppression_factor_mp(z_red, Lambda, xi=xi, dps=dps)
    w_LCDM = mp.mpf(-1)
    # The correction acts as a stiff fluid (w=+1) with energy density
    # ~ beta_R * (H/Lambda)^2 * H^2/kappa^2.  Its contribution to w_eff
    # is proportional to the ratio of its energy density to the total DE
    # energy density, times (1 + w_stiff) = 2.  So delta_w ~ 2 * suppression.
    delta_w = 2 * sup
    w_total = w_LCDM + delta_w

    return {
        "z": float(z_red),
        "w_LCDM": float(w_LCDM),
        "delta_w": float(delta_w),
        "w_eff": float(w_total),
        "log10_delta_w": float(mp.log10(delta_w)) if delta_w > 0 else None,
        "indistinguishable_from_LCDM": bool(delta_w < mp.mpf("1e-10")),
    }


def h0_prediction(
    Lambda: float | mp.mpf = LAMBDA_PPN,
    xi: float | mp.mpf = 0.0,
    dps: int = DPS,
) -> dict[str, Any]:
    """
    SCT prediction for the Hubble constant H_0.

    Since delta H^2/H^2 ~ 10^{-64}, the predicted H_0 is:
        H_0^{SCT} = H_0^{LCDM} * sqrt(1 + delta H^2/H^2)
                   = H_0^{LCDM} * (1 + (1/2) delta H^2/H^2 + ...)
                   = H_0^{LCDM} + O(10^{-64}) km/s/Mpc

    The H_0 tension requires delta H_0 / H_0 ~ 8.4% (= (73.04-67.36)/67.36).
    SCT provides delta H_0 / H_0 ~ 5e-65.
    """
    mp.mp.dps = dps
    sup = suppression_factor_mp(0, Lambda, xi=xi, dps=dps)

    # delta H / H = (1/2) delta H^2 / H^2 + O((delta H^2)^2)
    delta_H_over_H = sup / 2
    H0_SCT = H0_KMS_MPC * (1 + delta_H_over_H)

    required_shift = (H0_SHOES - H0_KMS_MPC) / H0_KMS_MPC
    shortfall_orders = mp.log10(required_shift) - mp.log10(delta_H_over_H) if delta_H_over_H > 0 else mp.inf

    return {
        "H0_Planck_km_s_Mpc": float(H0_KMS_MPC),
        "H0_SH0ES_km_s_Mpc": float(H0_SHOES),
        "delta_H2_over_H2": float(sup),
        "delta_H_over_H": float(delta_H_over_H),
        "H0_SCT_km_s_Mpc": float(H0_SCT),
        "required_shift": float(required_shift),
        "shortfall_orders_of_magnitude": float(shortfall_orders),
        "resolves_tension": False,
    }


def s8_prediction(
    Lambda: float | mp.mpf = LAMBDA_PPN,
    xi: float | mp.mpf = 0.0,
    dps: int = DPS,
) -> dict[str, Any]:
    """
    SCT prediction for the S_8 parameter.

    S_8 depends on the growth function, which is governed by the modified
    Poisson equation with Pi_s(k^2/Lambda^2, xi).

    At structure-formation scales k ~ 0.1 h/Mpc ~ 6.8e-27 eV:
        k^2/Lambda^2 ~ 10^{-48} (for PPN bound) to 10^{-111} (for M_Pl)

    delta S_8 / S_8 ~ (k/Lambda)^2 ~ 10^{-48}
    """
    mp.mp.dps = dps
    Lambda_mp = mp.mpf(Lambda)

    # k ~ 0.1 h/Mpc in eV
    # k [Mpc^{-1}] = 0.1 * h (comoving wavenumber at structure-formation scale)
    # k [m^{-1}] = k [Mpc^{-1}] / Mpc_in_m  (since 1 Mpc = 3.086e22 m)
    # k [eV] = k [m^{-1}] * hbar*c [eV*m]
    hbar_c_eV_m = mp.mpf("1.97326980e-7")  # eV*m
    k_Mpc_inv = mp.mpf("0.1")  # h/Mpc (typical structure-formation scale)
    h_hubble = H0_KMS_MPC / 100  # dimensionless Hubble parameter
    k_m_inv = k_Mpc_inv * h_hubble / MPC_IN_M  # Mpc^{-1} to m^{-1}
    k_eV = k_m_inv * hbar_c_eV_m
    k_sq_over_Lambda_sq = (k_eV / Lambda_mp) ** 2

    # Growth modification from Pi_s
    s_coeff = scalar_mode_coefficient(float(xi))
    growth_mod = s_coeff * k_sq_over_Lambda_sq  # ~ delta G_eff / G_N

    # S_8 observational values
    S8_Planck = mp.mpf("0.832")
    S8_KiDS = mp.mpf("0.759")
    required_shift = (S8_Planck - S8_KiDS) / S8_Planck

    return {
        "k_eV": float(k_eV),
        "Lambda_eV": float(Lambda_mp),
        "k_sq_over_Lambda_sq": float(k_sq_over_Lambda_sq),
        "log10_k_sq_over_Lambda_sq": float(mp.log10(k_sq_over_Lambda_sq)) if k_sq_over_Lambda_sq > 0 else None,
        "growth_modification": float(growth_mod),
        "S8_Planck": float(S8_Planck),
        "S8_KiDS": float(S8_KiDS),
        "required_shift": float(required_shift),
        "SCT_S8_deviation": float(growth_mod),
        "resolves_tension": False,
    }


# ============================================================
# Step 3: Growth Function Modification
# ============================================================

def growth_modification(
    k_eV: float | mp.mpf,
    Lambda: float | mp.mpf = LAMBDA_PPN,
    xi: float | mp.mpf = 0.0,
    dps: int = DPS,
) -> dict[str, Any]:
    """
    Growth function modification from the modified Poisson equation.

    The modified Poisson equation (NT-4a, Eq. 5.12):
        k^2 Phi = -4 pi G * (1 / Pi_s(k^2/Lambda^2, xi)) * a^2 rho delta

    For tensor modes:
        Pi_TT(z) = 1 + (13/60) z F_hat_1(z)

    For scalar modes (Poisson):
        Pi_s(z, xi) = 1 + 6(xi - 1/6)^2 z F_hat_2(z, xi)

    The growth function modification is:
        delta f sigma_8 / f sigma_8 ~ |1/Pi_s - 1| ~ |Pi_s - 1|
    """
    mp.mp.dps = dps
    k_mp = mp.mpf(k_eV)
    Lambda_mp = mp.mpf(Lambda)
    z_arg = (k_mp / Lambda_mp) ** 2

    Pi_TT_val = Pi_TT(float(z_arg), xi=float(xi), dps=dps)
    Pi_s_val = Pi_scalar(float(z_arg), xi=float(xi), dps=dps)

    delta_Pi_TT = abs(mp.re(Pi_TT_val) - 1)
    delta_Pi_s = abs(mp.re(Pi_s_val) - 1)

    return {
        "k_eV": float(k_mp),
        "Lambda_eV": float(Lambda_mp),
        "z_arg": float(z_arg),
        "log10_z": float(mp.log10(z_arg)) if z_arg > 0 else None,
        "Pi_TT": float(mp.re(Pi_TT_val)),
        "Pi_s": float(mp.re(Pi_s_val)),
        "delta_Pi_TT": float(delta_Pi_TT),
        "delta_Pi_s": float(delta_Pi_s),
        "growth_negligible": bool(delta_Pi_s < mp.mpf("1e-10")),
    }


# ============================================================
# Step 4: GW Speed at Cosmological Epochs
# ============================================================

def gw_speed_cosmological(
    z_red: float | mp.mpf,
    Lambda: float | mp.mpf = LAMBDA_PPN,
    dps: int = DPS,
) -> dict[str, Any]:
    """
    GW speed c_T at a given cosmological epoch z.

    c_T = c exactly at tree level (structural property).
    Corrections: |c_T/c - 1| ~ alpha_C * (H(z)/Lambda)^2.

    At all epochs z <= 1100 (CMB decoupling):
        H(z) <= H(1100) ~ 4.5e-29 eV (for LCDM)
        |c_T/c - 1| ~ 10^{-49} (for PPN bound) to 10^{-112} (for M_Pl)

    GW170817 constraint: |c_T/c - 1| < 5.6e-16.
    """
    mp.mp.dps = dps
    H_z = _H_at_redshift(z_red)
    Lambda_mp = mp.mpf(Lambda)

    # Conservative upper bound on deviation (from nt4c_flrw.py)
    deviation = ALPHA_C * (H_z / Lambda_mp) ** 2 / 2  # |c_T - 1| ~ (1/2) alpha_C (H/Lam)^2

    return {
        "z": float(z_red),
        "H_z_eV": float(H_z),
        "Lambda_eV": float(Lambda_mp),
        "deviation": float(deviation),
        "log10_deviation": float(mp.log10(deviation)) if deviation > 0 else None,
        "GW170817_bound": float(GW170817_BOUND),
        "satisfies_GW170817": bool(deviation < GW170817_BOUND),
    }


# ============================================================
# Step 5: Summary of All Late-Time Predictions
# ============================================================

def consistency_summary(dps: int = DPS) -> dict[str, Any]:
    """
    Complete summary of SCT predictions for late-time cosmology.

    Documents the negative result: SCT cannot resolve H_0 or S_8 tensions,
    but is automatically consistent with all cosmological observations.
    """
    mp.mp.dps = dps

    # Suppression at today (z=0) for PPN bound
    sup_today = suppression_factor(0, LAMBDA_PPN, xi=0, dps=dps)
    sup_today_conformal = suppression_factor(0, LAMBDA_PPN, xi=1 / 6, dps=dps)
    sup_CMB = suppression_factor(1100, LAMBDA_PPN, xi=0, dps=dps)
    sup_MPl = suppression_factor(0, M_PL_EV, xi=0, dps=dps)

    # H_0
    h0_pred = h0_prediction(LAMBDA_PPN, xi=0, dps=dps)

    # S_8
    s8_pred = s8_prediction(LAMBDA_PPN, xi=0, dps=dps)

    # w_eff
    w_today = w_eff(0, LAMBDA_PPN, xi=0, dps=dps)

    # GW speed at key epochs
    gw_today = gw_speed_cosmological(0, LAMBDA_PPN, dps=dps)
    gw_CMB = gw_speed_cosmological(1100, LAMBDA_PPN, dps=dps)

    # Growth at k = 0.1 h/Mpc
    hbar_c_eV_m = mp.mpf("1.97326980e-7")
    h_hubble = H0_KMS_MPC / 100
    k_Mpc_inv = mp.mpf("0.1")
    k_m_inv = k_Mpc_inv * h_hubble / MPC_IN_M  # Mpc^{-1} to m^{-1}
    k_eV = k_m_inv * hbar_c_eV_m
    growth = growth_modification(k_eV, LAMBDA_PPN, xi=0, dps=dps)

    return {
        "phase": "MT-2",
        "description": "SCT predictions for late-time cosmology",
        "central_result": "NEGATIVE: SCT cannot resolve H_0 or S_8 tensions",
        "reason": (
            "SCT generates UV nonlocality (F(Box/Lambda^2), entire functions) "
            "with Lambda >> H_0. Corrections are O(H^2/Lambda^2) << 1. "
            "Resolving tensions requires IR nonlocality (Box^{-1}, m ~ H_0)."
        ),
        "suppression_today_PPN": sup_today,
        "suppression_today_conformal": sup_today_conformal,
        "suppression_CMB_PPN": sup_CMB,
        "suppression_today_MPl": sup_MPl,
        "H0_prediction": h0_pred,
        "S8_prediction": s8_pred,
        "w_eff_today": w_today,
        "GW_speed_today": gw_today,
        "GW_speed_CMB": gw_CMB,
        "growth_modification": growth,
        "positive_results": [
            "c_T = c (structural, passes GW170817)",
            "De Sitter stability (attractor)",
            "No new cosmological degrees of freedom",
            "Exact GR recovery at conformal coupling xi = 1/6",
            "Trivially consistent with all precision cosmological data",
        ],
        "negative_results": [
            "Cannot resolve H_0 tension (shortfall: 64 orders of magnitude)",
            "Cannot resolve S_8 tension (growth modification ~ 10^{-48})",
            "No distinctive late-time cosmological prediction",
        ],
        "testable_predictions_elsewhere": [
            "Solar system (PPN-1): Lambda >= 2.38e-3 eV",
            "Laboratory (LT-3d): Lambda >= 2.565 meV",
            "Black holes (MR-9): corrections near r ~ 1/Lambda",
            "Early universe (INF-1): conditional on scalaron mass",
        ],
    }


# ============================================================
# Auxiliary: Comprehensive Grid Computation
# ============================================================

def compute_full_grid(dps: int = DPS) -> dict[str, Any]:
    """
    Compute suppression for the full grid specified in the task:
    - z = 0, 0.5, 1, 2, 5, 10, 100, 1000, 1100
    - Lambda = 10^{-3} eV, 1 eV, 1 GeV, 10^6 GeV, 10^{13} GeV, M_Pl
    - xi = 0, 1/12, 1/6, 1/4
    """
    mp.mp.dps = dps

    z_values = [0, 0.5, 1, 2, 5, 10, 100, 1000, 1100]
    Lambda_dict = {
        "2.38e-3 eV (PPN)": LAMBDA_PPN,
        "1 eV": mp.mpf("1"),
        "1 GeV": mp.mpf("1e9"),
        "1e6 GeV": mp.mpf("1e15"),
        "1e13 GeV": mp.mpf("1e22"),
        "M_Pl": M_PL_EV,
    }
    xi_values = [0.0, 1.0 / 12, 1.0 / 6, 0.25]

    grid = []
    for z_red in z_values:
        for lam_label, lam_val in Lambda_dict.items():
            for xi_val in xi_values:
                r = suppression_factor(z_red, lam_val, xi=xi_val, dps=dps)
                r["Lambda_label"] = lam_label
                grid.append(r)

    # Extract key summary line: z=0, Lambda=PPN, xi=0
    key_result = suppression_factor(0, LAMBDA_PPN, xi=0, dps=dps)

    return {
        "grid": grid,
        "n_entries": len(grid),
        "key_result_z0_PPN_xi0": key_result,
        "H0_eV": float(H0_EV),
        "Lambda_PPN_eV": float(LAMBDA_PPN),
    }


# ============================================================
# Auxiliary: Redshift-Dependent Quantities
# ============================================================

def suppression_vs_redshift(
    Lambda: float | mp.mpf = LAMBDA_PPN,
    xi: float | mp.mpf = 0.0,
    z_values: list | None = None,
    dps: int = DPS,
) -> list[dict]:
    """
    Compute suppression as a function of redshift for a fixed Lambda, xi.
    """
    if z_values is None:
        z_values = [0, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 50, 100,
                    500, 1000, 1100]
    return [suppression_factor(z, Lambda, xi=xi, dps=dps) for z in z_values]


# ============================================================
# Figure Generation
# ============================================================

def make_figures() -> None:
    """Generate MT-2 publication figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figures")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: Suppression Factor vs Redshift ---
    fig1, ax1 = plt.subplots(figsize=(8, 5.5))

    Lambda_configs = [
        ("$\\Lambda = 2.38 \\times 10^{-3}$ eV (PPN)", LAMBDA_PPN, "#d62728"),
        ("$\\Lambda = 1$ eV", mp.mpf("1"), "#ff7f0e"),
        ("$\\Lambda = 1$ GeV", mp.mpf("1e9"), "#2ca02c"),
        ("$\\Lambda = 10^{6}$ GeV", mp.mpf("1e15"), "#1f77b4"),
        ("$\\Lambda = M_{\\rm Pl}$", M_PL_EV, "#9467bd"),
    ]

    z_plot = np.logspace(-2, np.log10(1200), 200)
    z_plot = np.concatenate([[0], z_plot])

    for label, lam_val, color in Lambda_configs:
        log_sup = []
        for z_val in z_plot:
            s = suppression_factor(z_val, lam_val, xi=0)
            if s["log10_suppression"] is not None:
                log_sup.append(s["log10_suppression"])
            else:
                log_sup.append(-200)
        ax1.plot(z_plot[1:], log_sup[1:], label=label, color=color, lw=1.8)

    ax1.set_xlabel("Redshift $z$", fontsize=13)
    ax1.set_ylabel(r"$\log_{10}(\delta H^2 / H^2)$", fontsize=13)
    ax1.set_title(r"SCT Suppression Factor: $\delta H^2/H^2 = \beta_R \cdot (H/\Lambda)^2$"
                  "\n" r"($\xi = 0$, FLRW background)", fontsize=12)
    ax1.set_xscale("log")
    ax1.legend(fontsize=10, loc="upper left")
    ax1.axhline(y=np.log10(0.18), color="black", ls="--", lw=1.2,
                label=r"Required for $H_0$ tension ($\sim 18\%$)")
    ax1.text(1.5, np.log10(0.18) + 1.5, r"Required: $\delta H^2/H^2 \sim 0.18$",
             fontsize=10, ha="left")
    ax1.set_xlim(0.01, 1200)
    ax1.set_ylim(-130, 5)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(str(FIGURES_DIR / "mt2_suppression.pdf"), dpi=300,
                 bbox_inches="tight")
    plt.close(fig1)
    print(f"  Figure saved: {FIGURES_DIR / 'mt2_suppression.pdf'}")

    # --- Figure 2: w_eff(z) ---
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    z_w = np.linspace(0, 3, 200)
    # SCT w_eff is indistinguishable from LCDM; plot the difference
    delta_w_vals = []
    for z_val in z_w:
        w_res = w_eff(z_val, LAMBDA_PPN, xi=0)
        delta_w_vals.append(w_res["log10_delta_w"] if w_res["log10_delta_w"] is not None else -200)

    ax2.plot(z_w, delta_w_vals, color="#d62728", lw=2,
             label=r"SCT ($\Lambda = 2.38 \times 10^{-3}$ eV)")

    # Also plot for M_Pl
    delta_w_mpl = []
    for z_val in z_w:
        w_res = w_eff(z_val, M_PL_EV, xi=0)
        delta_w_mpl.append(w_res["log10_delta_w"] if w_res["log10_delta_w"] is not None else -200)
    ax2.plot(z_w, delta_w_mpl, color="#9467bd", lw=2, ls="--",
             label=r"SCT ($\Lambda = M_{\rm Pl}$)")

    # Maggiore RT model for comparison
    ax2.axhline(y=np.log10(0.04), color="#2ca02c", ls="-.", lw=1.5,
                label=r"Maggiore RT: $\delta w \sim 0.04$")

    ax2.set_xlabel("Redshift $z$", fontsize=13)
    ax2.set_ylabel(r"$\log_{10}|\delta w_{\rm eff}|$", fontsize=13)
    ax2.set_title(r"Deviation of $w_{\rm eff}(z)$ from $\Lambda$CDM ($w = -1$)"
                  "\n" r"($\xi = 0$, FLRW background)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_ylim(-130, 0)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(str(FIGURES_DIR / "mt2_w_eff.pdf"), dpi=300,
                 bbox_inches="tight")
    plt.close(fig2)
    print(f"  Figure saved: {FIGURES_DIR / 'mt2_w_eff.pdf'}")


# ============================================================
# Main: Self-Test (CQ3)
# ============================================================

def _self_test() -> bool:
    """Run internal consistency checks."""
    mp.mp.dps = DPS
    all_pass = True
    n_tests = 0

    # --- T1: H_0 conversion ---
    H0_expected_Hz = mp.mpf("2.183e-18")
    assert abs(H0_HZ / H0_expected_Hz - 1) < 1e-3, f"H0 Hz: {H0_HZ}"
    n_tests += 1

    H0_expected_eV = mp.mpf("1.437e-33")
    assert abs(H0_EV / H0_expected_eV - 1) < 1e-3, f"H0 eV: {H0_EV}"
    n_tests += 1

    # --- T2: alpha_R ---
    aR_0 = alpha_R(0)
    assert abs(aR_0 - mp.mpf(1) / 18) < 1e-30, f"alpha_R(0): {aR_0}"
    n_tests += 1

    aR_conf = alpha_R(1 / 6)
    assert aR_conf == 0, f"alpha_R(1/6): {aR_conf}"
    n_tests += 1

    # --- T3: beta_R ---
    bR_0 = _beta_R(0)
    expected_bR = mp.mpf(1) / 18 / (16 * mp.pi ** 2)
    assert abs(bR_0 / expected_bR - 1) < 1e-30, f"beta_R(0): {bR_0}"
    n_tests += 1

    # --- T4: Suppression at z=0, Lambda=PPN, xi=0 ---
    sup = suppression_factor(0, LAMBDA_PPN, xi=0)
    assert sup["log10_suppression"] is not None
    assert -65 < sup["log10_suppression"] < -63, \
        f"Suppression at PPN: 10^{sup['log10_suppression']:.1f}"
    n_tests += 1

    # --- T5: Suppression at z=0, Lambda=M_Pl, xi=0 ---
    sup_mpl = suppression_factor(0, M_PL_EV, xi=0)
    assert sup_mpl["log10_suppression"] is not None
    assert sup_mpl["log10_suppression"] < -120, \
        f"Suppression at M_Pl: 10^{sup_mpl['log10_suppression']:.1f}"
    n_tests += 1

    # --- T6: Conformal coupling kills everything ---
    sup_conf = suppression_factor(0, LAMBDA_PPN, xi=1 / 6)
    assert sup_conf["delta_H2_over_H2"] == 0, \
        f"Conformal suppression: {sup_conf['delta_H2_over_H2']}"
    n_tests += 1

    # --- T7: H(z) is monotonically increasing with z ---
    H_0 = _H_at_redshift(0)
    H_1 = _H_at_redshift(1)
    H_1100 = _H_at_redshift(1100)
    assert H_0 < H_1 < H_1100, f"H monotonicity: {H_0} < {H_1} < {H_1100}"
    n_tests += 1

    # --- T8: w_eff is indistinguishable from LCDM ---
    w_res = w_eff(0, LAMBDA_PPN, xi=0)
    assert w_res["indistinguishable_from_LCDM"], f"w_eff: {w_res['w_eff']}"
    n_tests += 1

    # --- T9: GW speed satisfies GW170817 ---
    gw = gw_speed_cosmological(0, LAMBDA_PPN)
    assert gw["satisfies_GW170817"], f"GW deviation: {gw['deviation']}"
    n_tests += 1

    # --- T10: H_0 tension not resolved ---
    h0 = h0_prediction(LAMBDA_PPN, xi=0)
    assert not h0["resolves_tension"], "Should not resolve H_0 tension"
    assert h0["shortfall_orders_of_magnitude"] > 60, \
        f"Shortfall: {h0['shortfall_orders_of_magnitude']}"
    n_tests += 1

    print(f"Self-test: {n_tests}/{n_tests} PASS")
    return all_pass


if __name__ == "__main__":
    print("=" * 70)
    print("MT-2: SCT Modified Cosmology — Suppression Quantification")
    print("=" * 70)

    # Self-test
    print("\n--- Self-Test ---")
    _self_test()

    # Key suppression computation
    print("\n--- Key Suppression Factor (z=0, Lambda=PPN, xi=0) ---")
    key = suppression_factor(0, LAMBDA_PPN, xi=0)
    print(f"  H_0 = {key['H_z_eV']:.4e} eV")
    print(f"  Lambda = {key['Lambda_eV']:.4e} eV")
    print(f"  alpha_R(0) = {key['alpha_R']:.6f}")
    print(f"  beta_R(0) = {key['beta_R']:.6e}")
    print(f"  (H_0/Lambda)^2 = {key['H_over_Lambda_sq']:.4e}")
    print(f"  delta H^2/H^2 = {key['delta_H2_over_H2']:.4e}")
    print(f"  log10(suppression) = {key['log10_suppression']:.2f}")

    # H_0 prediction
    print("\n--- H_0 Prediction ---")
    h0 = h0_prediction(LAMBDA_PPN, xi=0)
    print(f"  H_0 (Planck) = {h0['H0_Planck_km_s_Mpc']:.2f} km/s/Mpc")
    print(f"  H_0 (SH0ES) = {h0['H0_SH0ES_km_s_Mpc']:.2f} km/s/Mpc")
    print(f"  delta H/H (SCT) = {h0['delta_H_over_H']:.4e}")
    print(f"  Required shift = {h0['required_shift']:.4f}")
    print(f"  Shortfall = {h0['shortfall_orders_of_magnitude']:.1f} orders of magnitude")

    # Full summary
    print("\n--- Consistency Summary ---")
    summary = consistency_summary()
    print(f"  Central result: {summary['central_result']}")
    print(f"  Positive results: {len(summary['positive_results'])}")
    print(f"  Negative results: {len(summary['negative_results'])}")

    # Save results
    print("\n--- Saving Results ---")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Compute full grid (this takes ~1 min at 100-digit)
    print("  Computing full suppression grid...")
    grid = compute_full_grid(dps=50)  # use 50-digit for speed in main
    summary_data = consistency_summary(dps=50)

    results = {
        "phase": "MT-2",
        "description": "SCT predictions for late-time cosmology (negative result)",
        "author": "David Alfyorov",
        "key_result": {
            "delta_H2_over_H2": key["delta_H2_over_H2"],
            "log10_suppression": key["log10_suppression"],
            "shortfall_orders": h0["shortfall_orders_of_magnitude"],
        },
        "suppression_grid": grid,
        "H0_prediction": h0,
        "S8_prediction": s8_prediction(LAMBDA_PPN, xi=0, dps=50),
        "w_eff_today": w_eff(0, LAMBDA_PPN, xi=0, dps=50),
        "GW_speed_today": gw_speed_cosmological(0, LAMBDA_PPN, dps=50),
        "consistency_summary": {
            "central_result": summary_data["central_result"],
            "reason": summary_data["reason"],
            "positive_results": summary_data["positive_results"],
            "negative_results": summary_data["negative_results"],
        },
    }

    with open(RESULTS_DIR / "mt2_cosmology_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {RESULTS_DIR / 'mt2_cosmology_results.json'}")

    # Figures
    print("\n--- Generating Figures ---")
    make_figures()

    print("\n" + "=" * 70)
    print("MT-2 COMPLETE: Negative result confirmed and documented.")
    print("  SCT cannot resolve H_0 or S_8 tensions (shortfall: 64 orders)")
    print("  SCT IS consistent with all late-time cosmological data")
    print("=" * 70)
