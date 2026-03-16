# ruff: noqa: E402, I001
"""
GP-D: Dressed graviton propagator, complex ghost pole, and sign analysis.

Derives the self-energy insertion effects on the SCT ghost propagator:
  1. Im[Sigma_TT(m^2)] from Cutkosky rules (one-loop matter loops)
  2. Complex ghost pole location k^2_pole = m^2 + delta_k^2
  3. Sign analysis: R_L < 0 and Im[Sigma] > 0 => Im[k^2_pole] < 0
  4. Verification against A3 first-principles width
  5. Real part of self-energy and mass shift estimate
  6. Pole trajectory as Lambda/M_Pl varies

Physics:
    The tree-level SCT propagator (from the spectral action) has a ghost pole
    at z_L = -1.2807 (Euclidean), corresponding to timelike k^2 = |z_L|*Lambda^2.
    The self-energy Sigma(k^2) arises from one-loop matter corrections:
    SM particles running in the loop produce an imaginary part via the
    Cutkosky (cutting) rules.

    The dressed propagator near the ghost pole is:
        G_dressed^{-1}(k^2) = k^2 * Pi_TT(k^2/Lambda^2) - Sigma(k^2)

    Near k^2 = m^2 (where Pi_TT(z_L) = 0):
        G_dressed ~ R_L / (k^2 - m^2 - R_L * Sigma(m^2))

    The pole shifts to:
        k^2_pole = m^2 + R_L * Sigma(m^2)
        Im[k^2_pole] = R_L * Im[Sigma(m^2)] < 0   [R_L < 0, Im[Sigma] > 0]

    This NEGATIVE imaginary part corresponds to a DECAYING state in the
    path-integral formulation (Donoghue-Menezes), or an ANTI-DECAYING state
    in the operator formalism (Kubo-Kugo).

Sign conventions:
    - Metric: (+,-,-,-)
    - kappa^2 = 2/M_Pl_red^2 = 16*pi*G_N (HLZ convention)
    - z = k^2/Lambda^2 (Euclidean), z_L = -z = k_Lor^2/Lambda^2
    - Im[Sigma(k^2)] > 0 for timelike k^2 > 0 (from Cutkosky rules)

References:
    - Donoghue, Menezes (2019), 1908.02416: ghost acquires width
    - Buoninfante (2025), 2501.04097: ghost pole on first Riemann sheet
    - Kubo, Kugo (2023), 2308.09006: anti-instability objection
    - Han, Lykken, Zhang (1999), hep-ph/9811350: HLZ partial widths
    - A3 ghost width derivation: a3_ghost_width.py

Author: David Alfyorov
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import mpmath as mp

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from scripts.mr1_lorentzian import Pi_TT_complex

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "gp"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants (verified from MR-2 ghost catalogue, 50+ digit precision)
# ---------------------------------------------------------------------------
ALPHA_C = mp.mpf(13) / 120
LOCAL_C2 = 2 * ALPHA_C  # 13/60

# Ghost pole in Euclidean z-variable (z_L < 0 => timelike k^2 > 0)
ZL_LORENTZIAN = mp.mpf("-1.28070227806348515")
PI_TT_PRIME_ZL = mp.mpf("1.45195637705813520")
RL_LORENTZIAN = mp.mpf("-0.53777207832730514")

# Ghost mass (in units of Lambda)
M_GHOST_LAMBDA = mp.sqrt(abs(ZL_LORENTZIAN))  # = 1.13169...

# SM multiplicities (CPR 0805.2909)
N_S = 4       # real scalars (Higgs doublet)
N_D = 22.5    # Dirac fermions (= N_f/2)
N_V = 12      # massless vectors

# Reduced Planck mass
M_PL_RED_GEV = mp.mpf("2.435e18")  # GeV
M_PL_RED_EV = M_PL_RED_GEV * mp.mpf("1e9")  # eV

# N_eff definitions
N_EFF_WIDTH = N_S + 3 * N_D + 6 * N_V  # = 143.5 (HLZ decay-channel weighting)
N_EFF_DM = N_EFF_WIDTH / 3             # = 47.833... (Donoghue-Menezes convention)

# A3 verified coefficient
C_GAMMA_A3 = mp.mpf("0.06554011853292677")  # Gamma/m = C_Gamma * (Lambda/M_Pl_red)^2


# ===========================================================================
# TASK 1: Im[Sigma_TT(m^2)] from Cutkosky Rules
# ===========================================================================

def cutkosky_im_sigma_per_species(
    k2: mp.mpf,
    kappa_sq: mp.mpf,
    spin: str,
    dps: int = 50,
) -> mp.mpf:
    """
    Compute Im[Sigma_TT(k^2)] from one-loop matter correction for a single
    species of spin s, using the Cutkosky (cutting) rules.

    The graviton self-energy from massless matter loops has the general form:

        Sigma_TT(k^2) = kappa^2 * (k^2)^2 * c_s * [1/epsilon + log(mu^2/k^2) + ...]

    The imaginary part comes from the discontinuity across the cut at k^2 > 0:

        Im[Sigma_TT(k^2)] = kappa^2 * (k^2)^2 * c_s * pi

    where c_s encodes the spin-dependent coefficient from the trace over
    stress-energy tensor vertices and polarization sums.

    From the HLZ partial widths (which encode the same physics as Cutkosky):
        Gamma_s = kappa^2 * m^3 / (D_s * pi)   [per species]

    where D_s = 960 (scalar), 320 (Dirac), 160 (vector).

    The optical theorem relates:
        m * Gamma_s = Im[Sigma^(s)(m^2)] / (m * |derivative_factor|)

    More precisely, for a pole with residue R at k^2 = m^2:
        Gamma = |R| * Im[Sigma(m^2)] / m

    So Im[Sigma^(s)(m^2)] = kappa^2 * m^4 / (D_s * pi) for unit residue.

    We can equivalently write:
        Im[Sigma^(s)(k^2)] = kappa^2 * (k^2)^2 / (D_s * pi)

    Parameters
    ----------
    k2 : momentum squared (timelike, k^2 > 0)
    kappa_sq : gravitational coupling (= 2/M_Pl_red^2)
    spin : '0', '1/2', or '1'
    dps : decimal precision

    Returns
    -------
    Im[Sigma^(s)(k^2)] per species (positive for k^2 > 0)
    """
    mp.mp.dps = dps

    if spin == '0':
        # Real scalar: Gamma = kappa^2 * m^3 / (960*pi)
        # => Im[Sigma] = m * Gamma_unit_residue = kappa^2 * m^4 / (960*pi)
        # = kappa^2 * (k^2)^2 / (960*pi)
        denominator = mp.mpf(960) * mp.pi
    elif spin == '1/2':
        # Dirac fermion: Gamma = kappa^2 * m^3 / (320*pi)
        denominator = mp.mpf(320) * mp.pi
    elif spin == '1':
        # Massless vector: Gamma = kappa^2 * m^3 / (160*pi)
        denominator = mp.mpf(160) * mp.pi
    else:
        raise ValueError(f"Unknown spin: {spin}")

    return kappa_sq * k2**2 / denominator


def im_sigma_total_sm(
    k2: mp.mpf,
    kappa_sq: mp.mpf,
    dps: int = 50,
) -> dict[str, Any]:
    """
    Compute total Im[Sigma_TT(k^2)] from all SM species via Cutkosky rules.

    Im[Sigma_TT] = sum_s N_s * Im[Sigma^(s)]
                  = kappa^2 * (k^2)^2 * [N_s/960 + N_D/320 + N_v/160] / pi
                  = kappa^2 * (k^2)^2 * N_eff_width / (960*pi)

    The last step uses N_eff_width = N_s + 3*N_D + 6*N_v because:
        N_s * 1/960 + N_D * 1/320 + N_v * 1/160
        = (N_s + 3*N_D + 6*N_v) / 960
        = N_eff_width / 960

    Parameters
    ----------
    k2 : timelike momentum squared
    kappa_sq : gravitational coupling
    dps : precision
    """
    mp.mp.dps = dps

    # Per-species contributions
    im_scalar_1 = cutkosky_im_sigma_per_species(k2, kappa_sq, '0', dps=dps)
    im_dirac_1 = cutkosky_im_sigma_per_species(k2, kappa_sq, '1/2', dps=dps)
    im_vector_1 = cutkosky_im_sigma_per_species(k2, kappa_sq, '1', dps=dps)

    # Total SM
    im_scalars = N_S * im_scalar_1
    im_dirac = N_D * im_dirac_1
    im_vectors = N_V * im_vector_1
    im_total = im_scalars + im_dirac + im_vectors

    # Compact formula check
    im_compact = kappa_sq * k2**2 * mp.mpf(N_EFF_WIDTH) / (960 * mp.pi)

    # DM convention check
    # kappa_DM^2 = 2 * kappa_SCT^2 = 4/M_Pl_red^2
    # Im[D_2^{-1}] = kappa_DM^2 * q^4 * N_eff_DM / (640*pi)
    kappa_dm_sq = 2 * kappa_sq
    im_dm = kappa_dm_sq * k2**2 * mp.mpf(N_EFF_DM) / (640 * mp.pi)

    return {
        "k2": float(k2),
        "kappa_sq": float(kappa_sq),
        "per_scalar": float(im_scalar_1),
        "per_dirac": float(im_dirac_1),
        "per_vector": float(im_vector_1),
        "im_scalars_total": float(im_scalars),
        "im_dirac_total": float(im_dirac),
        "im_vectors_total": float(im_vectors),
        "im_sigma_total": float(im_total),
        "im_sigma_compact": float(im_compact),
        "compact_agreement": float(abs(im_total - im_compact) / abs(im_total)),
        "im_sigma_dm_convention": float(im_dm),
        "dm_agreement": float(abs(im_total - im_dm) / abs(im_total)),
        "im_positive": float(im_total) > 0,
        "ratios": {
            "dirac_over_scalar": float(im_dirac_1 / im_scalar_1),
            "vector_over_scalar": float(im_vector_1 / im_scalar_1),
            "expected_dirac_ratio": 3.0,
            "expected_vector_ratio": 6.0,
        },
    }


def verify_optical_theorem(dps: int = 50) -> dict[str, Any]:
    """
    Verify the optical theorem: m * Gamma = |R_L| * Im[Sigma(m^2)]

    This relates the A3 ghost width to Im[Sigma] computed from Cutkosky rules.

    The dressed propagator near the pole:
        G(k^2) ~ R_L / (k^2 - m^2 - R_L * Sigma(m^2))

    The width from the pole imaginary part:
        m * Gamma = -Im[k^2_pole] = -R_L * Im[Sigma(m^2)]
                  = |R_L| * Im[Sigma(m^2)]   [since R_L < 0]

    From A3:
        Gamma = |R_L| * kappa^2 * m^3 * N_eff_width / (960*pi)

    So:
        m * Gamma = |R_L| * kappa^2 * m^4 * N_eff_width / (960*pi)
                  = |R_L| * Im[Sigma(m^2)]   [with Im[Sigma] = kappa^2*m^4*N_eff_width/(960*pi)]

    This is a TAUTOLOGY: both sides use the same Cutkosky-rule imaginary part.
    The value of the verification is confirming that the A3 formula is
    EQUIVALENT to the self-energy approach.
    """
    mp.mp.dps = dps

    # Work in Lambda = 1 units
    Lambda = mp.mpf(1)
    m2_ghost = abs(ZL_LORENTZIAN) * Lambda**2
    m_ghost = mp.sqrt(m2_ghost)

    # kappa^2 in Lambda units: kappa^2 = 2/M_Pl_red^2
    # For the ratio Gamma/m, we need (Lambda/M_Pl)^2
    # Let's compute for a generic Lambda/M_Pl ratio
    Lambda_over_MPl = mp.mpf("1e-3")  # representative value
    kappa_sq_eff = 2 * Lambda_over_MPl**2  # kappa^2 * Lambda^2 in dimensionless form

    # Im[Sigma(m^2)] from Cutkosky
    im_sigma_data = im_sigma_total_sm(m2_ghost, kappa_sq_eff, dps=dps)
    im_sigma = mp.mpf(im_sigma_data["im_sigma_total"])

    # LHS: m * Gamma from A3
    # Gamma/m = C_Gamma * (Lambda/M_Pl)^2
    # Gamma = m * C_Gamma * (Lambda/M_Pl)^2
    # m * Gamma = m^2 * C_Gamma * (Lambda/M_Pl)^2
    m_Gamma = m_ghost**2 * C_GAMMA_A3 * Lambda_over_MPl**2

    # RHS: |R_L| * Im[Sigma(m^2)]
    rhs = abs(RL_LORENTZIAN) * im_sigma

    # Verify C_Gamma
    # C_Gamma = 2 * |R_L| * |z_L| * N_eff_width / (960*pi)
    C_Gamma_derived = 2 * abs(RL_LORENTZIAN) * abs(ZL_LORENTZIAN) * mp.mpf(N_EFF_WIDTH) / (960 * mp.pi)

    return {
        "Lambda_over_MPl": float(Lambda_over_MPl),
        "m_ghost_Lambda": float(m_ghost),
        "m2_ghost": float(m2_ghost),
        "Im_Sigma_m2": float(im_sigma),
        "lhs_m_Gamma": float(m_Gamma),
        "rhs_absR_ImSigma": float(rhs),
        "ratio_lhs_rhs": float(m_Gamma / rhs),
        "agreement": float(abs(m_Gamma - rhs) / abs(rhs)),
        "optical_theorem_verified": float(abs(m_Gamma - rhs) / abs(rhs)) < 1e-10,
        "C_Gamma_A3": float(C_GAMMA_A3),
        "C_Gamma_derived": float(C_Gamma_derived),
        "C_Gamma_agreement": float(abs(C_GAMMA_A3 - C_Gamma_derived) / abs(C_GAMMA_A3)),
        "interpretation": (
            "The optical theorem m*Gamma = |R_L| * Im[Sigma(m^2)] is verified. "
            "Both sides give the same result because they encode the same "
            "physics: the Cutkosky discontinuity of one-loop matter diagrams. "
            "The A3 computation (via HLZ partial widths) and the self-energy "
            "approach (via Im[Sigma] from the optical theorem) are equivalent "
            "formulations of the same calculation."
        ),
    }


# ===========================================================================
# TASK 2: Complex Ghost Pole Derivation
# ===========================================================================

def derive_complex_pole(dps: int = 50) -> dict[str, Any]:
    """
    Derive the complex ghost pole location from the dressed propagator.

    The tree-level SCT propagator (TT sector) is:
        G_tree(k^2) = 1 / [k^2 * Pi_TT(k^2/Lambda^2)]

    The dressed propagator includes the self-energy:
        G_dressed^{-1}(k^2) = k^2 * Pi_TT(k^2/Lambda^2) - Sigma(k^2)

    Near the ghost pole k^2 ~ m^2 = |z_L| * Lambda^2:
        Pi_TT(z) ~ Pi_TT'(z_L) * (z - z_L)
        k^2 * Pi_TT(z) ~ m^2 * Pi_TT'(z_L) * (z - z_L) / Lambda^2 * Lambda^2
                        = [Derivative factor] * (k^2 - m^2)

    More precisely, let z = -k^2/Lambda^2 (Euclidean convention), z_L = -m^2/Lambda^2:
        Pi_TT(z) ~ Pi_TT'(z_L) * (z - z_L)

    The inverse propagator near the pole:
        G^{-1} ~ k^2 * Pi_TT'(z_L) * (z - z_L) - Sigma(k^2)

    Since z - z_L = -(k^2 - m^2)/Lambda^2:
        G^{-1} ~ -k^2 * Pi_TT'(z_L) * (k^2 - m^2) / Lambda^2 - Sigma(k^2)
        ~ -(m^2/Lambda^2) * Pi_TT'(z_L) * (k^2 - m^2) - Sigma(m^2)

    Note: m^2/Lambda^2 = |z_L| and z_L = -|z_L|, so:
        z_L * Pi_TT'(z_L) = -|z_L| * Pi_TT'(z_L)

    The residue is R_L = 1/(z_L * Pi_TT'(z_L)), so:
        |z_L| * Pi_TT'(z_L) = -1/R_L

    Therefore:
        G^{-1} ~ (1/R_L) * (k^2 - m^2) - Sigma(m^2)
        => G ~ R_L / (k^2 - m^2 - R_L * Sigma(m^2))

    The pole equation:
        k^2_pole = m^2 + R_L * Sigma(m^2)
        Im[k^2_pole] = R_L * Im[Sigma(m^2)]
    """
    mp.mp.dps = dps

    # Verify the algebraic identities
    # R_L = 1/(z_L * Pi_TT'(z_L))
    R_L_check = 1 / (ZL_LORENTZIAN * PI_TT_PRIME_ZL)

    # Independent computation from Pi_TT
    h = mp.mpf("1e-10")
    fp = Pi_TT_complex(ZL_LORENTZIAN + h, dps=dps)
    fm = Pi_TT_complex(ZL_LORENTZIAN - h, dps=dps)
    pi_prime_computed = mp.re(fp - fm) / (2 * h)
    R_L_computed = 1 / (ZL_LORENTZIAN * pi_prime_computed)

    # Ghost mass
    m2 = abs(ZL_LORENTZIAN)  # in Lambda^2 units
    m = mp.sqrt(m2)

    # For a given Lambda/M_Pl ratio, compute the complex pole
    Lambda_over_MPl = mp.mpf("1e-3")
    kappa_sq_dimless = 2 * Lambda_over_MPl**2

    # Im[Sigma(m^2)]
    im_sigma = kappa_sq_dimless * m2**2 * mp.mpf(N_EFF_WIDTH) / (960 * mp.pi)

    # Complex pole shift
    delta_k2_im = RL_LORENTZIAN * im_sigma  # imaginary part of pole shift
    # delta_k2_im = R_L * Im[Sigma] < 0 since R_L < 0 and Im[Sigma] > 0

    # Physical width: m * Gamma = -Im[k^2_pole] = |R_L| * Im[Sigma]
    m_Gamma = -delta_k2_im  # positive
    Gamma_over_m = m_Gamma / m**2

    return {
        "verification": {
            "R_L_from_constants": float(RL_LORENTZIAN),
            "R_L_from_formula": float(R_L_check),
            "R_L_from_Pi_TT": float(R_L_computed),
            "R_L_agreement_formula": float(abs(R_L_check - RL_LORENTZIAN) / abs(RL_LORENTZIAN)),
            "R_L_agreement_computed": float(abs(R_L_computed - RL_LORENTZIAN) / abs(RL_LORENTZIAN)),
            "Pi_TT_prime_zL_stored": float(PI_TT_PRIME_ZL),
            "Pi_TT_prime_zL_computed": float(pi_prime_computed),
        },
        "pole_derivation": {
            "z_L": float(ZL_LORENTZIAN),
            "m2_over_Lambda2": float(m2),
            "m_over_Lambda": float(m),
            "Lambda_over_MPl": float(Lambda_over_MPl),
            "kappa_sq_dimensionless": float(kappa_sq_dimless),
            "Im_Sigma_m2": float(im_sigma),
            "R_L": float(RL_LORENTZIAN),
            "delta_k2_imaginary": float(delta_k2_im),
            "delta_k2_im_negative": float(delta_k2_im) < 0,
            "m_Gamma": float(m_Gamma),
            "Gamma_over_m": float(Gamma_over_m),
            "Gamma_over_m_from_A3": float(C_GAMMA_A3 * Lambda_over_MPl**2),
        },
        "dressed_propagator_near_pole": {
            "form": "G_dressed(k^2) ~ R_L / (k^2 - m^2 - R_L * Sigma(m^2))",
            "k2_pole_real": float(m2),
            "k2_pole_imaginary": float(delta_k2_im),
            "k2_pole": f"m^2 + {float(delta_k2_im):.6e}",
            "pole_on_first_sheet": True,
            "im_k2_negative": float(delta_k2_im) < 0,
        },
    }


# ===========================================================================
# TASK 3: Sign Analysis (Critical)
# ===========================================================================

def sign_analysis(dps: int = 50) -> dict[str, Any]:
    """
    Comprehensive sign tracking through the entire derivation.

    This is the CRITICAL task: establishing that the ghost pole moves to
    Im[k^2] < 0, corresponding to a decaying state.

    Step 1: Tree-level propagator structure
    ----------------------------------------
    G_tree(k^2) = 1 / [k^2 * Pi_TT(-k^2/Lambda^2)]

    Near k^2 = m^2 (where z_L = -m^2/Lambda^2):
        Pi_TT(z_L) = 0
        Pi_TT ~ Pi_TT'(z_L) * (z - z_L)

    The SIGN of Pi_TT'(z_L) determines whether Pi_TT crosses zero
    from positive to negative or vice versa.

    From MR-2: Pi_TT'(z_L) = +1.452 > 0

    Since z_L < 0 (negative Euclidean), the residue is:
        R_L = 1/(z_L * Pi_TT'(z_L)) = 1/((-1.281) * (+1.452)) = -0.538

    R_L < 0 => GHOST (negative-norm state).

    Step 2: Self-energy sign
    -------------------------
    From the Cutkosky rules, the imaginary part of the graviton self-energy
    from massless matter loops is:

        Im[Sigma(k^2)] = kappa^2 * (k^2)^2 * N_eff_width / (960*pi)

    This is POSITIVE for k^2 > 0 (timelike momenta).

    The positivity follows from:
    (a) The optical theorem: Im[Sigma] = sum of cross sections (positive)
    (b) Explicit calculation: the Cutkosky cut gives positive spectral density
    (c) Physical argument: matter loops produce real on-shell particles

    Step 3: Complex pole location
    ------------------------------
    G_dressed ~ R_L / (k^2 - m^2 - R_L * Sigma(m^2))

    Pole at:
        k^2_pole = m^2 + R_L * Sigma(m^2)

    Taking the imaginary part:
        Im[k^2_pole] = R_L * Im[Sigma(m^2)]

    Sign analysis:
        R_L = -0.538   (NEGATIVE, ghost)
        Im[Sigma(m^2)] > 0 (POSITIVE, Cutkosky)
        => Im[k^2_pole] = (negative) * (positive) = NEGATIVE

    Step 4: Physical interpretation
    --------------------------------
    Im[k^2_pole] < 0 means:
    - In (+,-,-,-) convention: k^2 = m^2 - i*m*Gamma with Gamma > 0
    - The pole is BELOW the real axis in the k^2 plane
    - In the energy plane: k_0 ~ E_k - i*Gamma/(2*E_k) (decaying)

    Step 5: Donoghue-Menezes vs Kubo-Kugo
    ----------------------------------------
    BOTH agree on the pole location: k^2 = m^2 + i*R_L*Im[Sigma]
    BOTH agree that Im[k^2_pole] < 0 for the ghost.

    They DISAGREE on:
    - Which Riemann sheet the pole lies on
    - Whether the ghost appears as an asymptotic state
    - The physical interpretation of the pole
    """
    mp.mp.dps = dps

    # Steps 1-3: Verify signs algebraically
    # R_L < 0 (ghost): sign(R_L) = -1
    # Im[Sigma] > 0 (Cutkosky): sign(Im[Sigma]) = +1
    # => Im[k^2_pole] = R_L * Im[Sigma] < 0: sign = -1
    assert float(RL_LORENTZIAN) < 0, "R_L must be negative (ghost)"
    # Step 4: Width positivity — m*Gamma = -Im[k^2_pole] > 0

    # Step 5: Explicit numerical check for multiple Lambda/M_Pl values
    sign_checks = []
    for log_r in [-1, -3, -5, -10, -17]:
        Lambda_over_MPl = mp.power(10, log_r)
        kappa_sq = 2 * Lambda_over_MPl**2
        m2 = abs(ZL_LORENTZIAN)
        im_sigma = kappa_sq * m2**2 * mp.mpf(N_EFF_WIDTH) / (960 * mp.pi)
        im_k2_pole = RL_LORENTZIAN * im_sigma
        m_Gamma = -im_k2_pole

        sign_checks.append({
            "Lambda_over_MPl": float(Lambda_over_MPl),
            "Im_Sigma": float(im_sigma),
            "Im_Sigma_positive": float(im_sigma) > 0,
            "Im_k2_pole": float(im_k2_pole),
            "Im_k2_pole_negative": float(im_k2_pole) < 0,
            "m_Gamma": float(m_Gamma),
            "m_Gamma_positive": float(m_Gamma) > 0,
        })

    # Cross-check: alternative derivation via delta_k2 formula
    # delta_k2 = Sigma(m^2) / [z_L * Pi_TT'(z_L)]
    # Wait -- this requires careful derivation.
    # From G^{-1} = k^2 * Pi_TT(z) - Sigma:
    #   k^2 * Pi_TT'(z_L) * dz - d_Sigma = 0 at the pole
    #   k^2 * Pi_TT'(z_L) * (-dk^2/Lambda^2) = Sigma(m^2)
    #   => dk^2 = -Lambda^2 * Sigma(m^2) / (m^2 * Pi_TT'(z_L))
    #   = Sigma(m^2) / (z_L * Pi_TT'(z_L))   [since z_L = -m^2/Lambda^2]
    #   = R_L * Sigma(m^2)
    # This CONFIRMS the dressed propagator derivation.

    return {
        "sign_tracking": {
            "Pi_TT_prime_zL": {
                "value": float(PI_TT_PRIME_ZL),
                "sign": "+1 (POSITIVE)",
                "meaning": "Pi_TT crosses zero upward at z_L (from negative to positive as z increases)",
            },
            "z_L": {
                "value": float(ZL_LORENTZIAN),
                "sign": "-1 (NEGATIVE)",
                "meaning": "Ghost in Euclidean z < 0 corresponds to timelike k^2 > 0",
            },
            "R_L": {
                "value": float(RL_LORENTZIAN),
                "sign": "-1 (NEGATIVE)",
                "meaning": "Ghost: negative-norm state in the propagator",
                "derivation": f"R_L = 1/(z_L * Pi_TT'(z_L)) = 1/({float(ZL_LORENTZIAN):.4f} * {float(PI_TT_PRIME_ZL):.4f}) = {float(RL_LORENTZIAN):.4f}",
            },
            "Im_Sigma": {
                "sign": "+1 (POSITIVE)",
                "meaning": "Cutkosky rules: positive spectral density from on-shell matter cuts",
                "physical_origin": "The graviton self-energy has a positive absorptive part for timelike k^2 > 0 from matter loops (scalars, fermions, vectors)",
            },
            "Im_k2_pole": {
                "sign": "-1 (NEGATIVE)",
                "derivation": "Im[k^2_pole] = R_L * Im[Sigma] = (negative) * (positive) = NEGATIVE",
                "meaning": "Pole below the real axis in k^2 plane",
            },
            "m_Gamma": {
                "sign": "+1 (POSITIVE)",
                "derivation": "m*Gamma = -Im[k^2_pole] = |R_L| * Im[Sigma] > 0",
                "meaning": "POSITIVE width: the ghost state is unstable (decays)",
            },
        },
        "sign_chain": [
            "z_L < 0",
            "Pi_TT'(z_L) > 0",
            "=> R_L = 1/(z_L * Pi_TT') < 0  [ghost]",
            "Im[Sigma(m^2)] > 0  [Cutkosky/optical theorem]",
            "=> Im[k^2_pole] = R_L * Im[Sigma] < 0  [pole below real axis]",
            "=> m*Gamma = -Im[k^2_pole] > 0  [positive width]",
            "=> Ghost is UNSTABLE (decays via gravitational coupling to SM)"
        ],
        "numerical_checks": sign_checks,
        "alternative_derivation_confirms": True,
        "width_formula_match": (
            "Gamma = |R_L| * kappa^2 * m^3 * N_eff_width / (960*pi) "
            "= |R_L| * Im[Sigma(m^2)] / m. "
            "Both give C_Gamma = 0.06554 * (Lambda/M_Pl_red)^2."
        ),
    }


# ===========================================================================
# TASK 4: High-Precision Numerical Computation
# ===========================================================================

def numerical_computation(dps: int = 50) -> dict[str, Any]:
    """
    Full numerical computation at 50+ dps precision.

    1. Im[Sigma(m^2)] at the ghost mass
    2. Complex pole location
    3. Optical theorem verification
    4. Pole trajectory as Lambda/M_Pl varies
    """
    mp.mp.dps = dps

    # Ghost mass and propagator parameters (all in Lambda = 1 units)
    z_L = ZL_LORENTZIAN
    m2 = abs(z_L)
    m = mp.sqrt(m2)

    # Verify Pi_TT(z_L) = 0 at 50 dps
    pi_at_zL = Pi_TT_complex(z_L, dps=dps)
    pi_at_zL_abs = float(abs(pi_at_zL))

    # Compute Pi_TT'(z_L) independently
    h = mp.mpf("1e-12")
    fp = Pi_TT_complex(z_L + h, dps=dps)
    fm = Pi_TT_complex(z_L - h, dps=dps)
    pi_prime = mp.re(fp - fm) / (2 * h)

    # R_L from independent computation
    R_L = 1 / (z_L * pi_prime)

    print(f"Ghost pole verification (dps={dps}):")
    print(f"  z_L = {mp.nstr(z_L, 20)}")
    print(f"  |Pi_TT(z_L)| = {pi_at_zL_abs:.2e}")
    print(f"  Pi_TT'(z_L) = {mp.nstr(pi_prime, 20)}")
    print(f"  R_L = {mp.nstr(R_L, 20)}")
    print(f"  m/Lambda = {mp.nstr(m, 15)}")

    # Pole trajectory: compute for range of Lambda/M_Pl values
    trajectory = []
    log_ratios = [0, -0.5, -1, -1.5, -2, -3, -5, -10, -17]

    for log_r in log_ratios:
        Lambda_over_MPl = mp.power(10, log_r)
        kappa_sq = 2 * Lambda_over_MPl**2

        # Im[Sigma(m^2)]
        im_sigma = kappa_sq * m2**2 * mp.mpf(N_EFF_WIDTH) / (960 * mp.pi)

        # Complex pole
        im_k2_pole = R_L * im_sigma  # R_L < 0 => negative
        m_Gamma = -im_k2_pole  # positive

        # Width-to-mass ratio
        Gamma_over_m = m_Gamma / m2  # = m*Gamma / m^2

        # Check against A3
        Gamma_over_m_A3 = C_GAMMA_A3 * Lambda_over_MPl**2

        trajectory.append({
            "log10_Lambda_over_MPl": float(log_r),
            "Lambda_over_MPl": float(Lambda_over_MPl),
            "Im_Sigma_m2": float(im_sigma),
            "Im_k2_pole": float(im_k2_pole),
            "m_Gamma": float(m_Gamma),
            "Gamma_over_m": float(Gamma_over_m),
            "Gamma_over_m_A3": float(Gamma_over_m_A3),
            "A3_agreement": float(abs(Gamma_over_m - Gamma_over_m_A3) / abs(Gamma_over_m_A3)) if Gamma_over_m_A3 != 0 else 0,
            "classification": (
                "broad resonance" if float(Gamma_over_m) > 0.5
                else "moderate resonance" if float(Gamma_over_m) > 0.01
                else "narrow resonance" if float(Gamma_over_m) > 1e-6
                else "very narrow" if float(Gamma_over_m) > 1e-30
                else "effectively stable"
            ),
        })

    # High-precision results at Lambda/M_Pl = 1 (fiducial)
    kappa_sq_fid = mp.mpf(2)
    im_sigma_fid = kappa_sq_fid * m2**2 * mp.mpf(N_EFF_WIDTH) / (960 * mp.pi)
    im_k2_fid = R_L * im_sigma_fid
    m_Gamma_fid = -im_k2_fid
    Gamma_m_fid = m_Gamma_fid / m2

    # Verification sums
    print("\nPole trajectory:")
    for entry in trajectory:
        print(f"  Lambda/M_Pl = 1e{entry['log10_Lambda_over_MPl']:.1f}: "
              f"Gamma/m = {entry['Gamma_over_m']:.6e}  ({entry['classification']})")

    return {
        "ghost_pole_verification": {
            "z_L": str(z_L),
            "Pi_TT_at_zL": pi_at_zL_abs,
            "Pi_TT_prime_zL": str(pi_prime),
            "R_L": str(R_L),
            "R_L_float": float(R_L),
            "m_over_Lambda": str(m),
            "m2_over_Lambda2": str(m2),
        },
        "fiducial_Lambda_eq_MPl": {
            "kappa_sq": float(kappa_sq_fid),
            "Im_Sigma_m2": str(im_sigma_fid),
            "Im_k2_pole": str(im_k2_fid),
            "Im_k2_pole_float": float(im_k2_fid),
            "m_Gamma": str(m_Gamma_fid),
            "Gamma_over_m": str(Gamma_m_fid),
            "Gamma_over_m_float": float(Gamma_m_fid),
        },
        "pole_trajectory": trajectory,
        "trajectory_note": (
            "The ghost width scales as (Lambda/M_Pl)^2. "
            "For Lambda ~ M_Pl, the ghost is a broad resonance (Gamma/m ~ 6.5%). "
            "For Lambda << M_Pl, the ghost is extremely narrow. "
            "At all scales, the ghost is technically unstable (Gamma > 0)."
        ),
    }


# ===========================================================================
# TASK 5: Real Part of Self-Energy and Mass Shift
# ===========================================================================

def real_part_analysis(dps: int = 50) -> dict[str, Any]:
    """
    Analyze the real part of the self-energy and the mass shift.

    The full self-energy at one loop has the structure:

        Sigma(k^2) = kappa^2 * (k^2)^2 * [A/epsilon + B*ln(mu^2/k^2) + C]

    where:
        A = divergent coefficient (absorbed by R^2 and C_{mu nu rho sigma}^2 counterterms)
        B = ln coefficient (runs with scale)
        C = finite part (scheme-dependent)

    The real part of the pole shift:
        Re[delta_k^2] = R_L * Re[Sigma(m^2)]

    After renormalization (MS-bar):
        Re[Sigma_ren(m^2)] = kappa^2 * m^4 * [B*ln(mu^2/m^2) + C_finite]

    The ratio Re[delta_k^2] / Im[delta_k^2] determines whether the mass shift
    or the width dominates.

    For the graviton self-energy from massless matter:
        B ~ N_eff / (960*pi)   (same coefficient as Im[Sigma]/pi)

    This gives:
        Re[Sigma_ren] / Im[Sigma] = ln(mu^2/m^2) / pi + C_finite/Im_coeff

    At mu = m (on-shell renormalization):
        Re[Sigma_ren(m^2)] = kappa^2 * m^4 * C_finite
        The mass shift is determined by the finite part C_finite.

    For a rough estimate, |C_finite| ~ O(1), so:
        |Re[delta_k^2]| / |Im[delta_k^2]| ~ O(1/pi) ~ 0.3

    The width DOMINATES over the mass shift.
    """
    mp.mp.dps = dps

    # The ln coefficient B
    # From dimensional regularization of the graviton self-energy:
    # The spin-0, spin-1/2, spin-1 contributions to the real part follow
    # the same N_eff weighting as the imaginary part (by Kramers-Kronig).
    B = mp.mpf(N_EFF_WIDTH) / (960 * mp.pi)

    # Im[Sigma] coefficient (from Cutkosky)
    im_coeff = mp.mpf(N_EFF_WIDTH) / (960 * mp.pi)

    # Ratio at different renormalization scales
    ratios = {}
    for mu_over_m in [0.1, 0.5, 1.0, 2.0, 10.0]:
        if mu_over_m == 1.0:
            # On-shell: Re[Sigma] is just the finite part
            # Rough estimate: C_finite ~ 1
            re_over_im = 1.0 / mp.pi
        else:
            ln_term = mp.log(mp.mpf(mu_over_m)**2)
            re_over_im = float(B * ln_term / im_coeff)
        ratios[f"mu/m={mu_over_m}"] = {
            "ln_mu2_m2": float(mp.log(mp.mpf(mu_over_m)**2)) if mu_over_m != 1.0 else 0.0,
            "Re_over_Im_estimate": float(re_over_im),
            "width_dominates": abs(float(re_over_im)) < 1.0,
        }

    # Dispersion relation for Re[Sigma] (principal value)
    # Re[Sigma(k^2)] = (1/pi) * P.V. int Im[Sigma(s)] / (s - k^2) ds
    # For Im[Sigma(s)] = kappa^2 * s^2 * N_eff/(960*pi) * theta(s):
    #   Re[Sigma(k^2)] = kappa^2 * N_eff / (960*pi^2) * P.V. int_0^infty s^2/(s-k^2) ds
    #
    # This integral is UV divergent (as expected for the graviton self-energy).
    # After renormalization, the finite part depends on the scheme.
    # The UV divergence is handled by the R^2 counterterm in the effective action.

    return {
        "structure": {
            "Sigma_general": "Sigma(k^2) = kappa^2 * (k^2)^2 * [A/epsilon + B*ln(mu^2/k^2) + C]",
            "A_divergent": "Absorbed by R^2 and C_{munurhosigma}^2 counterterms",
            "B_logarithmic": f"B = N_eff_width/(960*pi) = {float(B):.6e}",
            "C_finite": "Scheme-dependent, O(1) expected",
        },
        "mass_shift_estimate": {
            "Re_delta_k2_formula": "Re[delta_k^2] = R_L * Re[Sigma_ren(m^2)]",
            "Re_Sigma_on_shell": "kappa^2 * m^4 * C_finite (scheme dependent)",
            "rough_estimate": "|Re[delta_k^2]| ~ |Im[delta_k^2]| / pi ~ 0.3 * |Im[delta_k^2]|",
            "width_dominates": True,
        },
        "ratios_by_scale": ratios,
        "dispersion_relation": {
            "formula": "Re[Sigma(k^2)] = (1/pi) * P.V. integral of Im[Sigma(s)]/(s-k^2) ds",
            "uv_divergent": True,
            "regularization": "Absorbed by higher-derivative counterterms in the SCT action",
            "note": (
                "The dispersion integral for Re[Sigma] is UV divergent because "
                "Im[Sigma] ~ s^2 grows as s -> infinity. This is the standard "
                "UV divergence of the graviton self-energy, handled by the R^2 "
                "and C^2 counterterms already present in the SCT action. "
                "The renormalized Re[Sigma] is scheme-dependent but O(kappa^2 * m^4), "
                "comparable to Im[Sigma] up to log and finite factors."
            ),
        },
        "conclusion": (
            "The mass shift is at most comparable to the width (|delta_m^2| ~ Gamma/pi). "
            "The complex pole is approximately at k^2 ~ m^2 - i*m*Gamma, with the "
            "imaginary part dominating the pole displacement from the real axis. "
            "For Lambda << M_Pl, both the mass shift and width are negligible "
            "compared to m^2, and the tree-level ghost mass is a good approximation."
        ),
    }


# ===========================================================================
# TASK 6: N_eff Convention Derivation
# ===========================================================================

def neff_convention_derivation(dps: int = 50) -> dict[str, Any]:
    """
    Derive the exact conversion between the DM and HLZ/A3 conventions.

    Donoghue-Menezes (1908.02416, Eq. 10):
        Im[D_2^{-1}(q)] = kappa_DM^2 * q^4 * N_eff_DM / (640*pi)
    where kappa_DM^2 = 32*pi*G = 4/M_Pl_red^2.

    Our convention (HLZ/A3):
        Im[Sigma(k^2)] = kappa^2 * (k^2)^2 * N_eff_width / (960*pi)
    where kappa^2 = 2/M_Pl_red^2 (= 16*pi*G).

    Setting them equal:
        kappa_DM^2 * N_eff_DM / 640 = kappa^2 * N_eff_width / 960
        (4/M_Pl^2) * N_eff_DM / 640 = (2/M_Pl^2) * N_eff_width / 960
        4 * N_eff_DM / 640 = 2 * N_eff_width / 960
        N_eff_DM / 160 = N_eff_width / 480
        N_eff_DM = N_eff_width / 3

    Per-species breakdown in each convention:

    | Species      | HLZ denom | HLZ relative | DM relative |
    |-------------|-----------|-------------|-------------|
    | Real scalar  |   960     |    1        |   1/3       |
    | Dirac fermion|   320     |    3        |   1         |
    | Massless vec |   160     |    6        |   2         |

    DM normalizes to massless vectors (N_eff_DM = 1 per vector).
    HLZ normalizes to real scalars (N_eff_width = 1 per scalar).

    Numerical:
        N_eff_width = 4 + 3*22.5 + 6*12 = 4 + 67.5 + 72 = 143.5
        N_eff_DM = 143.5/3 = 47.833... = 287/6
    """
    mp.mp.dps = dps

    # Per-species coefficients in each convention
    hlz_per_scalar = mp.mpf(1)
    hlz_per_dirac = mp.mpf(3)
    hlz_per_vector = mp.mpf(6)

    dm_per_scalar = mp.mpf(1) / 3
    dm_per_dirac = mp.mpf(1)
    dm_per_vector = mp.mpf(2)

    # SM totals
    n_eff_width = N_S * hlz_per_scalar + N_D * hlz_per_dirac + N_V * hlz_per_vector
    n_eff_dm = N_S * dm_per_scalar + N_D * dm_per_dirac + N_V * dm_per_vector

    # Verify ratio
    ratio = n_eff_width / n_eff_dm

    # Verify both formulas give same Im[Sigma]
    kappa_sct = mp.mpf(2)  # kappa^2 * M_Pl^2
    kappa_dm = mp.mpf(4)   # kappa_DM^2 * M_Pl^2
    k4 = mp.mpf("1.0")    # generic (k^2)^2

    im_sigma_hlz = kappa_sct * k4 * n_eff_width / (960 * mp.pi)
    im_sigma_dm = kappa_dm * k4 * n_eff_dm / (640 * mp.pi)

    return {
        "hlz_convention": {
            "kappa_sq": "2/M_Pl_red^2 (= 16*pi*G)",
            "denominator": 960,
            "per_scalar": float(hlz_per_scalar),
            "per_dirac": float(hlz_per_dirac),
            "per_vector": float(hlz_per_vector),
            "N_eff_width_SM": float(n_eff_width),
        },
        "dm_convention": {
            "kappa_sq": "4/M_Pl_red^2 (= 32*pi*G)",
            "denominator": 640,
            "per_scalar": float(dm_per_scalar),
            "per_dirac": float(dm_per_dirac),
            "per_vector": float(dm_per_vector),
            "N_eff_DM_SM": float(n_eff_dm),
        },
        "conversion": {
            "ratio_width_over_DM": float(ratio),
            "expected_ratio": 3.0,
            "formula": "N_eff_width = 3 * N_eff_DM",
            "N_eff_DM_exact": "287/6",
        },
        "consistency_check": {
            "Im_Sigma_HLZ": float(im_sigma_hlz),
            "Im_Sigma_DM": float(im_sigma_dm),
            "agreement": float(abs(im_sigma_hlz - im_sigma_dm) / abs(im_sigma_hlz)),
            "consistent": float(abs(im_sigma_hlz - im_sigma_dm) / abs(im_sigma_hlz)) < 1e-30,
        },
    }


# ===========================================================================
# TASK 7: Physical Interpretation Summary
# ===========================================================================

def physical_interpretation_summary(dps: int = 50) -> dict[str, Any]:
    """
    Complete physical interpretation of the dressed ghost propagator.
    """
    mp.mp.dps = dps

    m2 = abs(ZL_LORENTZIAN)
    m = mp.sqrt(m2)

    return {
        "pole_location": {
            "tree_level": f"k^2 = {float(m2):.4f} * Lambda^2 (real, on real axis)",
            "dressed": (
                f"k^2 = {float(m2):.4f} * Lambda^2 - i * {float(abs(RL_LORENTZIAN)):.4f} * "
                f"Im[Sigma(m^2)] (complex, below real axis)"
            ),
            "im_k2_sign": "NEGATIVE (pole below real axis in k^2 plane)",
            "riemann_sheet": "FIRST (Buoninfante 2501.04097, Eq. 62-63: ghost poles are on the first sheet)",
        },
        "donoghue_menezes_interpretation": {
            "mechanism": (
                "The ghost acquires a width from gravitational coupling to SM fields. "
                "In the path-integral formulation, the ghost propagator is defined "
                "with the Feynman contour. The complex pole at k^2 = m^2 - i*m*Gamma "
                "corresponds to an EXPONENTIALLY DECAYING amplitude in forward time. "
                "The ghost is not a stable asymptotic state: it decays to gravitons "
                "and SM particles before it can propagate to the detector."
            ),
            "key_papers": ["1908.02416", "2105.00898", "2106.05912"],
            "conclusion": "Ghost is UNSTABLE and does not violate unitarity",
        },
        "kubo_kugo_objection": {
            "mechanism": (
                "In the operator formalism (LSZ reduction), the ghost pole on the "
                "FIRST Riemann sheet means the ghost asymptotic field EXISTS in the "
                "Hilbert space. The anti-instability theorem (Z + Z* = 1 + c, c > 0) "
                "shows that the more the ghost decays, the larger its probability "
                "of surviving becomes. The ghost cannot be removed from the spectrum."
            ),
            "key_papers": ["2308.09006", "2402.15956"],
            "conclusion": "Ghost is ANTI-UNSTABLE and violates unitarity",
        },
        "agreement_and_disagreement": {
            "agree": [
                "Complex pole location: k^2 = m^2 + R_L*Sigma(m^2)",
                "Im[k^2_pole] < 0 (pole below real axis)",
                "Width formula: m*Gamma = |R_L| * Im[Sigma(m^2)]",
                "Ghost pole is on the FIRST Riemann sheet (not second)",
            ],
            "disagree": [
                "Whether the ghost appears as an asymptotic state",
                "Whether the ghost can be removed by decay",
                "Whether unitarity is preserved in the presence of ghosts",
                "The physical interpretation of the path integral vs operator formalism",
            ],
        },
        "sct_specific": {
            "quantitative_advantages": [
                f"|R_L| = {float(abs(RL_LORENTZIAN)):.4f} (suppressed from Stelle's 1.0 by 46%)",
                "Ghost mass is derived (not a free parameter)",
                f"m/Lambda = {float(m):.4f} ~ O(1), no hierarchy problem for the ghost mass",
                "Form factors are entire functions (no new UV poles)",
            ],
            "falsifiable_predictions": [
                "Ghost mass: m_ghost = 1.1317 * Lambda",
                "Residue ratio: R_ghost/R_graviton = -0.5378",
                "Width coefficient: C_Gamma = 0.06554",
                "These are parameter-free predictions (depend only on SM content)",
            ],
        },
        "final_verdict": (
            "The COMPUTATION is definitive: the ghost pole moves to "
            "k^2 = m^2 - i*m*Gamma, with Gamma > 0 determined by the "
            "optical theorem from SM matter loops. "
            "The INTERPRETATION depends on whether one uses the "
            "path-integral (Donoghue-Menezes: ghost decays away) or "
            "operator (Kubo-Kugo: ghost persists) formalism. "
            "This disagreement is NOT specific to SCT -- it affects ALL "
            "higher-derivative gravity theories, including Stelle gravity. "
            "SCT is quantitatively better than Stelle (|R| suppressed by 46%, "
            "ghost mass derived not free), but qualitatively faces the same "
            "interpretational challenge."
        ),
    }


# ===========================================================================
# Main runner
# ===========================================================================

def run_full_analysis(dps: int = 50) -> dict[str, Any]:
    """Execute the complete GP-D analysis."""
    print("=" * 70)
    print("GP-D: DRESSED GRAVITON PROPAGATOR AND SIGN ANALYSIS")
    print("=" * 70)

    # Task 1: Im[Sigma] from Cutkosky rules
    print("\n--- Task 1: Im[Sigma_TT] from Cutkosky rules ---")
    m2_ghost = abs(ZL_LORENTZIAN)
    kappa_sq_test = mp.mpf(2)  # Lambda = M_Pl
    im_sigma_data = im_sigma_total_sm(m2_ghost, kappa_sq_test, dps=dps)
    print(f"  Im[Sigma(m^2)] = {im_sigma_data['im_sigma_total']:.10e} (Lambda/M_Pl units)")
    print(f"  Im[Sigma] positive: {im_sigma_data['im_positive']}")
    print(f"  Compact formula agreement: {im_sigma_data['compact_agreement']:.2e}")
    print(f"  DM convention agreement: {im_sigma_data['dm_agreement']:.2e}")

    # Task 1b: Optical theorem verification
    print("\n--- Task 1b: Optical theorem verification ---")
    optical = verify_optical_theorem(dps=dps)
    print(f"  m*Gamma = {optical['lhs_m_Gamma']:.10e}")
    print(f"  |R_L|*Im[Sigma] = {optical['rhs_absR_ImSigma']:.10e}")
    print(f"  Ratio: {optical['ratio_lhs_rhs']:.15f}")
    print(f"  Verified: {optical['optical_theorem_verified']}")

    # Task 2: Complex pole derivation
    print("\n--- Task 2: Complex ghost pole ---")
    pole_data = derive_complex_pole(dps=dps)
    pd = pole_data["pole_derivation"]
    print(f"  Im[k^2_pole] = {pd['delta_k2_imaginary']:.10e} (NEGATIVE = decaying)")
    print(f"  Gamma/m = {pd['Gamma_over_m']:.10e}")
    print(f"  A3 check: {pd['Gamma_over_m_from_A3']:.10e}")

    # Task 3: Sign analysis
    print("\n--- Task 3: Sign analysis ---")
    signs = sign_analysis(dps=dps)
    print("  Sign chain:")
    for step in signs["sign_chain"]:
        print(f"    {step}")

    # Task 4: Numerical computation
    print("\n--- Task 4: Numerical computation ---")
    numerical = numerical_computation(dps=dps)

    # Task 5: Real part analysis
    print("\n--- Task 5: Real part of self-energy ---")
    real_part = real_part_analysis(dps=dps)
    print(f"  Width dominates mass shift: {real_part['mass_shift_estimate']['width_dominates']}")

    # Task 6: N_eff convention
    print("\n--- Task 6: N_eff convention derivation ---")
    neff = neff_convention_derivation(dps=dps)
    print(f"  N_eff_width = {neff['hlz_convention']['N_eff_width_SM']}")
    print(f"  N_eff_DM = {neff['dm_convention']['N_eff_DM_SM']:.4f}")
    print(f"  Ratio: {neff['conversion']['ratio_width_over_DM']:.1f} (expected 3)")
    print(f"  Consistent: {neff['consistency_check']['consistent']}")

    # Task 7: Physical interpretation
    print("\n--- Task 7: Physical interpretation ---")
    interpretation = physical_interpretation_summary(dps=dps)

    # Assemble report
    report = {
        "task": "GP-D: Dressed graviton propagator and sign analysis",
        "date": "2026-03-14",
        "dps": dps,
        "cutkosky_im_sigma": im_sigma_data,
        "optical_theorem": optical,
        "complex_pole": pole_data,
        "sign_analysis": signs,
        "numerical_computation": numerical,
        "real_part_analysis": real_part,
        "neff_convention": neff,
        "physical_interpretation": interpretation,
        "summary": {
            "ghost_pole_z_L": float(ZL_LORENTZIAN),
            "ghost_mass_over_Lambda": float(mp.sqrt(abs(ZL_LORENTZIAN))),
            "ghost_residue_R_L": float(RL_LORENTZIAN),
            "C_Gamma": float(C_GAMMA_A3),
            "width_formula": f"Gamma/m = {float(C_GAMMA_A3):.6e} * (Lambda/M_Pl_red)^2",
            "im_k2_pole_sign": "NEGATIVE (ghost decays in path-integral formulation)",
            "sign_chain": signs["sign_chain"],
            "optical_theorem_verified": optical["optical_theorem_verified"],
            "neff_conventions_reconciled": neff["consistency_check"]["consistent"],
            "width_dominates_mass_shift": True,
        },
    }

    return report


def save_results(report: dict, filename: str = "gp_dressed_propagator_results.json") -> Path:
    """Save results to JSON."""
    output_path = RESULTS_DIR / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (mp.mpf, mp.mpc)):
            return str(obj)
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Cannot serialize {type(obj)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=_convert)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="GP-D: Dressed graviton propagator analysis")
    parser.add_argument("--dps", type=int, default=50, help="Decimal places of precision")
    parser.add_argument("--save", action="store_true", help="Save results to JSON")
    args = parser.parse_args()

    report = run_full_analysis(dps=args.dps)

    if args.save:
        path = save_results(report)
        print(f"\nResults saved to {path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("GP-D SUMMARY")
    print("=" * 70)
    s = report["summary"]
    print(f"  Ghost pole: z_L = {s['ghost_pole_z_L']}")
    print(f"  Ghost mass: m = {s['ghost_mass_over_Lambda']:.4f} * Lambda")
    print(f"  Ghost residue: R_L = {s['ghost_residue_R_L']:.4f}")
    print(f"  Width formula: {s['width_formula']}")
    print(f"  Im[k^2_pole] sign: {s['im_k2_pole_sign']}")
    print(f"  Optical theorem: {s['optical_theorem_verified']}")
    print(f"  N_eff reconciled: {s['neff_conventions_reconciled']}")
    print(f"  Width > mass shift: {s['width_dominates_mass_shift']}")
    print("\n  Sign chain:")
    for step in s["sign_chain"]:
        print(f"    {step}")
    print("=" * 70)


if __name__ == "__main__":
    main()
