# ruff: noqa: E402, I001
"""
LT-3d: Laboratory and Solar System Bounds on the SCT Spectral Scale Lambda.

Computes all laboratory and solar-system constraints on the SCT spectral
scale Lambda, produces unified exclusion plots, and derives the tightest
lower bounds.

The SCT spectral action predicts a modified Newtonian potential:

    V(r)/V_N(r) = 1 + alpha_1 * exp(-r/lambda_1) + alpha_2 * exp(-r/lambda_2)

where:
    alpha_1 = -4/3  (spin-2 massive mode, repulsive)
    alpha_2 = +1/3  (scalar mode, attractive; decouples at xi=1/6)
    lambda_1 = hbar*c / (Lambda * sqrt(60/13))       [spin-2 Yukawa range]
    lambda_2 = hbar*c / (Lambda * sqrt(1/(2*(xi-1/6)^2)))  [scalar range]

References:
    Lee+ (2020), PRL 124, 101101 [arXiv:2002.11761]
    Kapner+ (2007), PRL 98, 021101 [arXiv:hep-ph/0611184]
    Chen+ (2016), PRL 116, 221102 [arXiv:1410.7267]
    Decca+ (2007), Eur. Phys. J. C 51, 963
    Sabulsky+ (2019), PRL 123, 061102
    Williams+ (2004), PRL 93, 261101
    Everitt+ (2011), PRL 106, 221101 [arXiv:1105.3456]
    Touboul+ (2022), PRL 129, 121102 [arXiv:2209.15487]
    Stelle (1977), Phys. Rev. D 16, 953
    Edholm, Koshelev, Mazumdar (2016), Phys. Rev. D 94, 104033
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import constants as sc
from scipy.interpolate import interp1d
from scipy.optimize import brentq

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from sct_tools.plotting import (
    SCT_COLORS,
    init_style,
    save_figure,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "lt3d"
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures" / "lt3d"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Physical constants (CODATA 2022 via scipy.constants)
# =============================================================================
HBAR = sc.hbar                          # J*s
C_LIGHT = sc.c                          # m/s
G_N = sc.G                              # m^3/(kg*s^2)
EV_J = sc.eV                            # 1 eV in Joules
HBAR_C_EV_M = HBAR * C_LIGHT / EV_J    # hbar*c in eV*m  ~ 1.9733e-7

# =============================================================================
# SCT Parameters (from NT-4a, verified)
# =============================================================================
ALPHA_C = 13 / 120          # total Weyl^2 coefficient (xi-independent)
LOCAL_C2 = 2 * ALPHA_C      # = 13/60

# Yukawa couplings (parameter-free, from Stelle decomposition)
ALPHA_1 = -4 / 3            # spin-2 massive ghost (repulsive)
ALPHA_2 = +1 / 3            # scalar mode (attractive)

# Mass ratios m/Lambda
M2_OVER_LAMBDA = math.sqrt(60 / 13)     # ~ 2.14834
# m_0/Lambda depends on xi: m_0 = Lambda / sqrt(2*(xi-1/6)^2)


# =============================================================================
# (a) SCT Yukawa parametrization
# =============================================================================
def lambda_1(Lambda_eV: float) -> float:
    """Spin-2 Yukawa range in meters.

    lambda_1 = hbar*c / m_2 = hbar*c / (Lambda * sqrt(60/13))
    """
    if Lambda_eV <= 0:
        raise ValueError(f"Lambda must be positive, got {Lambda_eV}")
    return HBAR_C_EV_M / (Lambda_eV * M2_OVER_LAMBDA)


def lambda_2(Lambda_eV: float, xi: float = 0.0) -> float | None:
    """Scalar Yukawa range in meters. Returns None at xi=1/6 (decoupled).

    The scalar mode coefficient in the propagator denominator is:
        3c1 + c2 = 6*(xi - 1/6)^2
    so m_0 = Lambda / sqrt(6*(xi-1/6)^2) = Lambda * sqrt(6) * |xi-1/6|^{-1}/6

    At xi=0: m_0 = Lambda * sqrt(6), lambda_2 = hbar*c / (Lambda*sqrt(6))
    """
    if Lambda_eV <= 0:
        raise ValueError(f"Lambda must be positive, got {Lambda_eV}")
    scalar_coeff = 6 * (xi - 1 / 6) ** 2   # = 3c1 + c2
    if scalar_coeff < 1e-30:
        return None  # scalar decoupled at conformal coupling
    m0_ov_L = 1 / math.sqrt(scalar_coeff)
    return HBAR_C_EV_M / (Lambda_eV * m0_ov_L)


def m0_over_Lambda(xi: float) -> float | None:
    """Return m_0/Lambda for given xi. None if scalar decoupled.

    m_0/Lambda = 1/sqrt(6*(xi-1/6)^2).
    At xi=0: sqrt(6). At xi=1/6: infinity (decoupled).
    """
    scalar_coeff = 6 * (xi - 1 / 6) ** 2
    if scalar_coeff < 1e-30:
        return None
    return 1.0 / math.sqrt(scalar_coeff)


def Lambda_from_lambda1(lam_m: float) -> float:
    """Invert: given spin-2 Yukawa range (meters), return Lambda (eV)."""
    if lam_m <= 0:
        raise ValueError(f"lambda must be positive, got {lam_m}")
    return HBAR_C_EV_M / (lam_m * M2_OVER_LAMBDA)


def Lambda_from_lambda2(lam_m: float, xi: float = 0.0) -> float | None:
    """Invert: given scalar Yukawa range (meters), return Lambda (eV)."""
    if lam_m <= 0:
        raise ValueError(f"lambda must be positive, got {lam_m}")
    scalar_coeff = 6 * (xi - 1 / 6) ** 2
    if scalar_coeff < 1e-30:
        return None
    m0_ov_L = 1.0 / math.sqrt(scalar_coeff)
    return HBAR_C_EV_M / (lam_m * m0_ov_L)


# =============================================================================
# Modified Newtonian potential
# =============================================================================
def V_ratio(r: float | np.ndarray, Lambda_eV: float,
            xi: float = 0.0) -> float | np.ndarray:
    """V(r)/V_N(r) = 1 + alpha_1*exp(-m_2*r) + alpha_2*exp(-m_0*r).

    Parameters:
        r: distance in meters
        Lambda_eV: spectral scale in eV
        xi: Higgs non-minimal coupling

    Returns:
        ratio V/V_N (dimensionless)
    """
    lam1 = lambda_1(Lambda_eV)
    ratio = 1.0 + ALPHA_1 * np.exp(-np.asarray(r) / lam1)
    lam2 = lambda_2(Lambda_eV, xi)
    if lam2 is not None:
        ratio = ratio + ALPHA_2 * np.exp(-np.asarray(r) / lam2)
    return ratio


def V_ratio_limit_r0(xi: float = 0.0) -> float:
    """V(r->0)/V_N(r->0) = 1 + alpha_1 + alpha_2.

    At xi=1/6: 1 - 4/3 = -1/3.  At general xi: 1 - 4/3 + 1/3 = 0.
    """
    if abs(xi - 1 / 6) < 1e-12:
        return 1.0 + ALPHA_1  # = -1/3
    return 1.0 + ALPHA_1 + ALPHA_2  # = 0


# =============================================================================
# (b) Eot-Wash exclusion curve (Lee 2020 + Kapner 2007)
# =============================================================================
# Data points from Lee+ (2020), Fig. 3, and Kapner+ (2007), Fig. 4.
# Format: (lambda in meters, |alpha|_95%)
# Lee 2020 dominates at lambda < ~200 um; Kapner 2007 for longer ranges.
EOT_WASH_DATA_LEE2020 = np.array([
    [20e-6,   150.0],
    [25e-6,   15.0],
    [30e-6,   4.0],
    [33e-6,   2.0],
    [36e-6,   1.3],
    [38.6e-6, 1.0],
    [40e-6,   0.8],
    [45e-6,   0.55],
    [50e-6,   0.4],
    [60e-6,   0.25],
    [70e-6,   0.17],
    [80e-6,   0.10],
    [100e-6,  0.04],
    [120e-6,  0.015],
    [150e-6,  6e-3],
    [200e-6,  3e-3],
])

EOT_WASH_DATA_KAPNER2007 = np.array([
    [200e-6,  3e-3],
    [300e-6,  1e-3],
    [500e-6,  5e-4],
    [700e-6,  3e-4],
    [1000e-6, 2e-4],
    [2000e-6, 1.5e-4],
    [5000e-6, 3e-4],
    [10000e-6, 3e-3],
])


def _build_composite_torsion_balance() -> interp1d:
    """Build a composite log-log interpolation of the torsion balance exclusion.

    Uses the tighter bound at each lambda from Lee 2020 and Kapner 2007.
    """
    all_data = np.vstack([EOT_WASH_DATA_LEE2020, EOT_WASH_DATA_KAPNER2007])
    # Sort by lambda
    idx = np.argsort(all_data[:, 0])
    all_data = all_data[idx]
    # At overlapping lambda, take the tighter (smaller |alpha|) bound
    unique_lam = np.unique(all_data[:, 0])
    best_alpha = []
    for lam in unique_lam:
        mask = all_data[:, 0] == lam
        best_alpha.append(np.min(all_data[mask, 1]))
    log_lam = np.log10(unique_lam)
    log_alpha = np.log10(best_alpha)
    return interp1d(log_lam, log_alpha, kind="linear",
                    fill_value="extrapolate")


_TORSION_INTERP = _build_composite_torsion_balance()


def alpha_95_torsion(lam_m: float) -> float:
    """Look up |alpha|_95% at given Yukawa range from torsion balance data."""
    return 10 ** float(_TORSION_INTERP(math.log10(lam_m)))


def Lambda_min_eotwash(xi: float = 0.0) -> dict[str, Any]:
    """Find the minimum Lambda allowed by the Eot-Wash torsion balance.

    For the spin-2 Yukawa (|alpha_1| = 4/3):
    Find Lambda such that lambda_1(Lambda) is exactly where
    |alpha|_95% = 4/3 on the exclusion curve.
    """
    target_alpha = abs(ALPHA_1)  # = 4/3

    # Search: find lambda_1 where alpha_95(lambda_1) = target_alpha
    # The exclusion curve is monotonically decreasing at small lambda,
    # so we look for the crossing point.
    def objective(log_lam: float) -> float:
        lam = 10 ** log_lam
        return math.log10(alpha_95_torsion(lam)) - math.log10(target_alpha)

    # Bracket: at small lambda, alpha_95 >> target; at large lambda, alpha_95 << target
    # The crossing is near lambda ~ 36 um
    try:
        log_lam_cross = brentq(objective, math.log10(20e-6), math.log10(100e-6))
    except ValueError:
        # Fallback: search wider range
        log_lam_cross = brentq(objective, math.log10(10e-6), math.log10(200e-6))

    lam_cross = 10 ** log_lam_cross  # meters
    Lambda_min = Lambda_from_lambda1(lam_cross)

    # Also get the scalar Yukawa bound if applicable
    scalar_info = None
    if abs(xi - 1 / 6) > 1e-12:
        target_alpha_s = abs(ALPHA_2)  # = 1/3
        try:
            def obj_s(log_lam: float) -> float:
                lam = 10 ** log_lam
                return math.log10(alpha_95_torsion(lam)) - math.log10(target_alpha_s)
            log_lam_s = brentq(obj_s, math.log10(20e-6), math.log10(200e-6))
            lam_cross_s = 10 ** log_lam_s
            Lambda_min_s = Lambda_from_lambda2(lam_cross_s, xi)
            scalar_info = {
                "lambda_cross_m": lam_cross_s,
                "lambda_cross_um": lam_cross_s * 1e6,
                "Lambda_min_eV": Lambda_min_s,
            }
        except (ValueError, TypeError):
            scalar_info = {"note": "scalar bound not constraining"}

    return {
        "experiment": "Eot-Wash (Lee+ 2020, Kapner+ 2007)",
        "xi": xi,
        "spin2_lambda_cross_m": lam_cross,
        "spin2_lambda_cross_um": lam_cross * 1e6,
        "Lambda_min_eV": Lambda_min,
        "Lambda_min_str": f"{Lambda_min:.4e} eV",
        "alpha_1": ALPHA_1,
        "abs_alpha_1": abs(ALPHA_1),
        "scalar_bound": scalar_info,
    }


# =============================================================================
# (c) Casimir bounds
# =============================================================================
# Chen+ (2016) data: approximate 95% CL exclusion in the sub-micron regime
CASIMIR_DATA_CHEN2016 = np.array([
    [40e-9,   1e13],
    [60e-9,   5e12],
    [100e-9,  1e11],
    [200e-9,  1e9],
    [300e-9,  1e8],
    [500e-9,  1e7],
    [1000e-9, 1e6],
    [2000e-9, 1e5],
    [5000e-9, 1e4],
    [8000e-9, 3e3],
])


def _build_casimir_interp() -> interp1d:
    """Build log-log interpolation of the Casimir exclusion curve."""
    log_lam = np.log10(CASIMIR_DATA_CHEN2016[:, 0])
    log_alpha = np.log10(CASIMIR_DATA_CHEN2016[:, 1])
    return interp1d(log_lam, log_alpha, kind="linear",
                    fill_value="extrapolate")


_CASIMIR_INTERP = _build_casimir_interp()


def alpha_95_casimir(lam_m: float) -> float:
    """Look up |alpha|_95% from Casimir experiments (Chen 2016)."""
    return 10 ** float(_CASIMIR_INTERP(math.log10(lam_m)))


def casimir_yukawa_correction(r: float, alpha: float, lam: float) -> float:
    """Fractional Yukawa correction to the Casimir force between parallel plates.

    delta_F/F_Casimir = -(240/pi^3) * alpha * (lam/r)^3
                        * [1 + r/lam + r^2/(2*lam^2)] * exp(-r/lam)

    Valid for r >> lam (exponentially suppressed regime).

    Parameters:
        r: plate separation (meters)
        alpha: Yukawa coupling strength
        lam: Yukawa range (meters)
    """
    x = r / lam
    return -(240 / math.pi ** 3) * alpha * (lam / r) ** 3 * (
        1 + x + x ** 2 / 2
    ) * math.exp(-x)


def Lambda_min_casimir(xi: float = 0.0) -> dict[str, Any]:
    """Casimir bound: find where SCT Yukawa becomes detectable.

    At sub-um separations, the SCT Yukawa is deeply in the
    |alpha|_95% >> 4/3 regime, so the Casimir bound is much weaker
    than the torsion balance.
    """
    # For the spin-2 Yukawa: find lambda_1 where alpha_95_casimir = 4/3
    target = abs(ALPHA_1)

    # Check if the Casimir curve ever reaches 4/3:
    # At lam = 8 um, alpha_95 ~ 3000, so 4/3 is far below.
    # The Casimir bounds only constrain |alpha| > 10^3 in their range,
    # so they cannot constrain |alpha| = 4/3.

    # Still, report the range where Casimir sensitivity crosses 4/3
    # by extrapolation of the curve into the torsion-balance regime.
    min_casimir_lam = 40e-9  # 40 nm
    max_casimir_lam = 8e-6   # 8 um

    alpha_at_max = alpha_95_casimir(max_casimir_lam)
    if alpha_at_max > target:
        # |alpha| = 4/3 is below the exclusion at all Casimir distances
        return {
            "experiment": "Casimir (Chen+ 2016)",
            "xi": xi,
            "Lambda_min_eV": None,
            "note": (
                f"Casimir experiments constrain |alpha| > {alpha_at_max:.0e} "
                f"at lambda = {max_casimir_lam * 1e6:.0f} um. "
                f"The SCT coupling |alpha| = {target:.4f} is far below "
                f"the Casimir exclusion in their entire range."
            ),
            "weakest_alpha_95": alpha_at_max,
            "casimir_correction_at_0p1um": casimir_yukawa_correction(
                0.1e-6, ALPHA_1, lambda_1(1.0)
            ),
        }

    # If somehow it does cross (it shouldn't):
    def objective(log_lam: float) -> float:
        lam = 10 ** log_lam
        return math.log10(alpha_95_casimir(lam)) - math.log10(target)

    log_lam_cross = brentq(
        objective, math.log10(min_casimir_lam), math.log10(max_casimir_lam)
    )
    lam_cross = 10 ** log_lam_cross
    Lambda_min = Lambda_from_lambda1(lam_cross)
    return {
        "experiment": "Casimir (Chen+ 2016)",
        "xi": xi,
        "Lambda_min_eV": Lambda_min,
        "lambda_cross_m": lam_cross,
    }


# =============================================================================
# (d) Atom interferometry bounds
# =============================================================================
def atom_interferometry_bound() -> dict[str, Any]:
    """Atom interferometry: Sabulsky 2019 and MAGIS-100 projections.

    Sabulsky 2019: |alpha| < 10^8 at lambda ~ 1 cm (chameleon/symmetron).
    This is many orders of magnitude above |alpha| = 4/3, so no SCT constraint.

    MAGIS-100: 100 m baseline, sensitive to lambda ~ 100 m.
    For Lambda > 10^{-3} eV, lambda_1 < 92 um, so MAGIS cannot probe SCT.
    """
    sabulsky_alpha_limit = 1e8
    sabulsky_lambda_m = 0.01  # 1 cm

    # MAGIS-100 baseline
    magis_baseline_m = 100.0

    # SCT correction at MAGIS baseline for Lambda = 10^{-3} eV
    Lambda_test = 1e-3  # eV
    lam1 = lambda_1(Lambda_test)
    r_over_lam = magis_baseline_m / lam1

    # exp(-r/lambda) = exp(-100m / 91.9um) ~ exp(-1.09e6) = 0
    magis_correction = abs(ALPHA_1) * math.exp(-min(r_over_lam, 700))

    return {
        "experiment": "Atom interferometry",
        "sabulsky_2019": {
            "alpha_limit": sabulsky_alpha_limit,
            "lambda_m": sabulsky_lambda_m,
            "constrains_sct": False,
            "reason": (
                f"|alpha|_95% = {sabulsky_alpha_limit:.0e} >> "
                f"|alpha_1| = {abs(ALPHA_1):.4f}"
            ),
        },
        "magis_100": {
            "baseline_m": magis_baseline_m,
            "sct_correction_at_1meV": magis_correction,
            "constrains_sct": False,
            "reason": (
                f"At Lambda = {Lambda_test} eV, lambda_1 = {lam1*1e6:.1f} um. "
                f"The MAGIS baseline (100 m) gives r/lambda = {r_over_lam:.0e}, "
                f"so the SCT Yukawa correction is ~ exp(-{r_over_lam:.0e}) = 0."
            ),
        },
    }


# =============================================================================
# (e) LLR bounds
# =============================================================================
def llr_bound(Lambda_eV: float = 2.55e-3) -> dict[str, Any]:
    """Lunar Laser Ranging: SCT correction at Earth-Moon distance.

    r_Moon = 384,400 km = 3.844e8 m.
    m_2 * r in natural units = (m_2 in eV) * (r in eV^{-1})
                              = m_2_eV * r_m / HBAR_C_EV_M.
    """
    r_moon_m = 3.844e8  # meters
    m2_eV = M2_OVER_LAMBDA * Lambda_eV
    m2_r = m2_eV * r_moon_m / HBAR_C_EV_M

    # exp(-m2*r) is negligible
    correction = abs(ALPHA_1) * math.exp(-min(m2_r, 700))

    return {
        "experiment": "Lunar Laser Ranging (Williams+ 2004)",
        "r_moon_m": r_moon_m,
        "Lambda_eV": Lambda_eV,
        "m2_eV": m2_eV,
        "m2_times_r": m2_r,
        "exponent": -m2_r,
        "correction": correction,
        "constrains_sct": False,
        "reason": (
            f"m_2 * r = {m2_r:.2e}. "
            f"exp(-{m2_r:.2e}) = 0 to any precision. "
            f"LLR provides NO constraint on SCT Lambda."
        ),
    }


# =============================================================================
# (f) GP-B bounds
# =============================================================================
def gpb_bound(Lambda_eV: float = 2.55e-3) -> dict[str, Any]:
    """Gravity Probe B: frame-dragging at orbital radius.

    GP-B orbit: ~642 km altitude, r ~ 7000 km from Earth center.
    """
    r_gpb_m = 7.0e6  # 7000 km in meters
    m2_eV = M2_OVER_LAMBDA * Lambda_eV
    m2_r = m2_eV * r_gpb_m / HBAR_C_EV_M

    correction = abs(ALPHA_1) * math.exp(-min(m2_r, 700))

    return {
        "experiment": "Gravity Probe B (Everitt+ 2011)",
        "r_orbit_m": r_gpb_m,
        "Lambda_eV": Lambda_eV,
        "m2_eV": m2_eV,
        "m2_times_r": m2_r,
        "exponent": -m2_r,
        "correction": correction,
        "constrains_sct": False,
        "reason": (
            f"m_2 * r = {m2_r:.2e}. "
            f"exp(-{m2_r:.2e}) = 0 to any precision. "
            f"GP-B provides NO constraint on SCT Lambda."
        ),
    }


# =============================================================================
# (g) MICROSCOPE bounds
# =============================================================================
def microscope_bound() -> dict[str, Any]:
    """MICROSCOPE: WEP test, composition-dependent forces.

    SCT Yukawa couples universally to mass (trace of T_munu).
    Universal coupling does NOT violate the WEP.
    Therefore MICROSCOPE cannot constrain SCT in the universal coupling case.
    """
    return {
        "experiment": "MICROSCOPE (Touboul+ 2022)",
        "eta_limit": 1.5e-15,
        "constrains_sct": False,
        "reason": (
            "SCT Yukawa forces couple universally to mass "
            "(via the trace of the stress-energy tensor). "
            "A composition-independent force does not violate the WEP, "
            "so MICROSCOPE sees zero signal. "
            "At xi != 1/6, the scalar sector couples through the conformal "
            "factor and is approximately universal for non-relativistic matter."
        ),
        "caveat": (
            "If the scalar mode at xi != 1/6 has a composition-dependent "
            "coupling (via differences in nuclear binding energy fractions), "
            "MICROSCOPE could in principle constrain xi. This requires a "
            "detailed analysis of the scalar coupling to nuclear matter."
        ),
    }


# =============================================================================
# (h) xi-dependence analysis
# =============================================================================
def Lambda_min_vs_xi(
    xi_values: list[float] | None = None,
) -> dict[str, Any]:
    """Compute Lambda_min from Eot-Wash for a range of xi values.

    At xi=1/6: only spin-2 Yukawa, |alpha_1|=4/3 gives tightest bound.
    At general xi: both Yukawas present, but spin-2 dominates (|alpha_1| > |alpha_2|).
    """
    if xi_values is None:
        xi_values = [0.0, 0.05, 0.10, 1 / 6, 0.20, 0.25, 0.30, 0.50]

    results = {}
    for xi in xi_values:
        bound = Lambda_min_eotwash(xi)
        xi_key = f"{xi:.6f}"
        results[xi_key] = {
            "xi": xi,
            "Lambda_min_eV": bound["Lambda_min_eV"],
            "lambda_cross_um": bound["spin2_lambda_cross_um"],
        }

        # At general xi, also compute the scalar mass
        m0_ratio = m0_over_Lambda(xi)
        if m0_ratio is not None:
            lam2_at_boundary = lambda_2(bound["Lambda_min_eV"], xi)
            results[xi_key]["m0_over_Lambda"] = m0_ratio
            results[xi_key]["lambda_2_at_boundary_um"] = (
                lam2_at_boundary * 1e6 if lam2_at_boundary else None
            )
            results[xi_key]["m2_over_m0"] = M2_OVER_LAMBDA / m0_ratio
        else:
            results[xi_key]["m0_over_Lambda"] = None
            results[xi_key]["scalar_status"] = "decoupled (xi=1/6)"

    return results


# =============================================================================
# (l) Competing theories
# =============================================================================
def competing_theories() -> dict[str, Any]:
    """Comparison with competing theories on the exclusion plot."""
    return {
        "SCT": {
            "type": "two_yukawa",
            "alpha_values": [ALPHA_1, ALPHA_2],
            "alpha_abs": [abs(ALPHA_1), abs(ALPHA_2)],
            "free_parameters": 1,
            "parameter_name": "Lambda",
            "ghost": "spin-2 (fakeon prescription)",
            "unique_prediction": (
                "alpha_1/alpha_2 = -4 (parameter-free). "
                f"m_2/m_0 = {math.sqrt(10/13):.4f} at xi=0 (parameter-free)."
            ),
        },
        "Stelle_gravity": {
            "type": "two_yukawa",
            "alpha_values": [-4 / 3, +1 / 3],
            "free_parameters": 2,
            "parameter_names": ["m_0", "m_2"],
            "ghost": "spin-2 (Ostrogradsky)",
            "note": "Same functional form as SCT but m_0, m_2 are free.",
        },
        "IDG": {
            "type": "error_function",
            "potential_form": "V/V_N = 1 - Erf(M_s*r/2)",
            "free_parameters": 1,
            "parameter_name": "M_s",
            "ghost": "No (entire propagator)",
            "note": "Gaussian modification, not Yukawa.",
        },
        "ADD_n2": {
            "type": "power_law",
            "potential_form": "V ~ 1/r^{2+n} at r < R",
            "free_parameters": 1,
            "parameter_name": "R (compactification radius)",
            "current_bound": "R < 44 um (Kapner 2007)",
            "ghost": "No",
        },
        "fR_gravity": {
            "type": "one_yukawa",
            "alpha_value": 1 / 3,
            "free_parameters": 2,
            "parameter_names": ["n", "lambda_C"],
            "ghost": "No (chameleon screened)",
        },
    }


# =============================================================================
# Unified exclusion curve (composite of all experiments)
# =============================================================================
def composite_exclusion_data() -> dict[str, np.ndarray]:
    """Return all exclusion data arrays for plotting."""
    # Extend Casimir data into nanometer regime
    # Geraci 2008 (bridge between Casimir and torsion balance)
    geraci_2008 = np.array([
        [5e-6,  2e4],
        [7e-6,  1.5e4],
        [10e-6, 1.4e4],
        [15e-6, 5e3],
        [20e-6, 1e3],
    ])

    # Atom interferometry (weak, at cm-m scales)
    atom_interf = np.array([
        [0.001, 1e9],
        [0.01,  1e8],
        [0.1,   1e7],
        [1.0,   1e6],
    ])

    # LLR (at ~10^5 km = 10^8 m)
    llr_data = np.array([
        [1e5,  1e-8],
        [1e6,  1e-9],
        [1e7,  1e-10],
        [1e8,  1e-11],
    ])

    return {
        "casimir_chen2016": CASIMIR_DATA_CHEN2016,
        "geraci_2008": geraci_2008,
        "lee_2020": EOT_WASH_DATA_LEE2020,
        "kapner_2007": EOT_WASH_DATA_KAPNER2007,
        "atom_interf": atom_interf,
        "llr": llr_data,
    }


# =============================================================================
# (i) Unified exclusion plot
# =============================================================================
def plot_exclusion_unified(
    xi: float = 0.0,
    output_path: Path | None = None,
) -> Path:
    """Generate the unified alpha-lambda exclusion plot with SCT prediction."""
    if output_path is None:
        output_path = FIGURES_DIR / "lt3d_exclusion_unified.pdf"

    init_style()
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    data = composite_exclusion_data()

    # --- Experimental exclusion curves ---
    # Casimir (Chen 2016)
    d = data["casimir_chen2016"]
    ax.plot(d[:, 0], d[:, 1], 'o-', color='#1565C0', ms=3, lw=1.2,
            label='Casimir (Chen+ 2016)')
    ax.fill_between(d[:, 0], d[:, 1], 1e16, alpha=0.08, color='#1565C0')

    # Geraci 2008 (Stanford)
    d = data["geraci_2008"]
    ax.plot(d[:, 0], d[:, 1], 's-', color='#00838F', ms=3, lw=1.2,
            label='Stanford (Geraci+ 2008)')

    # Lee 2020 (Eot-Wash)
    d = data["lee_2020"]
    ax.plot(d[:, 0], d[:, 1], '^-', color='#C62828', ms=3, lw=1.5,
            label='Eot-Wash (Lee+ 2020)')
    ax.fill_between(d[:, 0], d[:, 1], 1e16, alpha=0.08, color='#C62828')

    # Kapner 2007
    d = data["kapner_2007"]
    ax.plot(d[:, 0], d[:, 1], 'v-', color='#E65100', ms=3, lw=1.2,
            label='Eot-Wash (Kapner+ 2007)')

    # Atom interferometry
    d = data["atom_interf"]
    ax.plot(d[:, 0], d[:, 1], 'D-', color='#6A1B9A', ms=3, lw=1.0,
            alpha=0.7, label='Atom interf. (approx.)')

    # --- SCT prediction: horizontal lines ---
    ax.axhline(y=abs(ALPHA_1), color=SCT_COLORS["prediction"], lw=2.5,
               ls='--', label=r'SCT $|\alpha_1| = 4/3$ (spin-2)')
    ax.axhline(y=abs(ALPHA_2), color=SCT_COLORS["scalar"], lw=2.0,
               ls=':', label=r'SCT $|\alpha_2| = 1/3$ (scalar)')

    # Mark the SCT boundary crossing
    bound = Lambda_min_eotwash(xi)
    lam_cross = bound["spin2_lambda_cross_m"]
    Lambda_min_val = bound["Lambda_min_eV"]
    ax.plot(lam_cross, abs(ALPHA_1), '*', color=SCT_COLORS["prediction"],
            ms=15, zorder=10)
    ax.annotate(
        rf'$\Lambda_{{min}} = {Lambda_min_val:.2e}$ eV'
        f'\n$\\lambda_1 = {lam_cross*1e6:.1f}$ $\\mu$m',
        xy=(lam_cross, abs(ALPHA_1)),
        xytext=(lam_cross * 5, abs(ALPHA_1) * 30),
        fontsize=8,
        arrowprops=dict(arrowstyle='->', color=SCT_COLORS["prediction"]),
        color=SCT_COLORS["prediction"],
    )

    # Mark Lambda values along the SCT horizontal line
    for Lambda_val, label_txt in [
        (1e-2, r'$10^{-2}$'),
        (1e-1, r'$10^{-1}$'),
        (1.0,  r'$1$'),
    ]:
        lam = lambda_1(Lambda_val)
        if 1e-9 < lam < 1:
            ax.plot(lam, abs(ALPHA_1), '|', color=SCT_COLORS["prediction"],
                    ms=10, mew=1.5)
            ax.text(lam, abs(ALPHA_1) * 0.3, label_txt + ' eV',
                    fontsize=6, ha='center', color=SCT_COLORS["prediction"])

    # Shade excluded region
    ax.text(3e-7, 2e14, 'EXCLUDED', fontsize=14, color='gray', alpha=0.3,
            ha='center', rotation=0)

    # Axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-9, 1e1)
    ax.set_ylim(1e-5, 1e16)
    ax.set_xlabel(r'Yukawa range $\lambda$ (m)')
    ax.set_ylabel(r'Yukawa coupling $|\alpha|$')
    ax.set_title('SCT Laboratory Constraints: Unified Exclusion Plot')
    ax.legend(loc='upper right', fontsize=7, ncol=1)
    ax.grid(True, which='major', alpha=0.2)
    ax.grid(True, which='minor', alpha=0.05)

    fig.tight_layout()
    save_figure(fig, output_path.stem, fmt="pdf", directory=output_path.parent)
    plt.close(fig)
    return output_path


# =============================================================================
# (i-2) Lambda_min vs xi plot
# =============================================================================
def plot_lambda_min_vs_xi(output_path: Path | None = None) -> Path:
    """Plot Lambda_min as a function of the Higgs non-minimal coupling xi."""
    if output_path is None:
        output_path = FIGURES_DIR / "lt3d_lambda_min_vs_xi.pdf"

    xi_vals = np.linspace(0, 0.5, 50)
    Lambda_mins = []
    for xi in xi_vals:
        bound = Lambda_min_eotwash(xi)
        Lambda_mins.append(bound["Lambda_min_eV"])

    init_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    ax.plot(xi_vals, Lambda_mins, color=SCT_COLORS["prediction"], lw=2)
    ax.axvline(x=1 / 6, color='gray', ls='--', lw=0.8, alpha=0.5,
               label=r'$\xi = 1/6$ (conformal)')

    # Mark specific xi values
    for xi_mark in [0.0, 1 / 6, 0.25]:
        bound = Lambda_min_eotwash(xi_mark)
        ax.plot(xi_mark, bound["Lambda_min_eV"], 'o',
                color=SCT_COLORS["prediction"], ms=5, zorder=5)
        ax.annotate(f'{bound["Lambda_min_eV"]:.3e}',
                    xy=(xi_mark, bound["Lambda_min_eV"]),
                    xytext=(xi_mark + 0.03, bound["Lambda_min_eV"] * 1.1),
                    fontsize=6)

    ax.set_xlabel(r'$\xi$ (Higgs non-minimal coupling)')
    ax.set_ylabel(r'$\Lambda_{\min}$ (eV)')
    ax.set_title(r'Eot-Wash bound on $\Lambda$ vs $\xi$')
    ax.legend(fontsize=7)
    ax.set_xlim(-0.02, 0.52)

    fig.tight_layout()
    save_figure(fig, output_path.stem, fmt="pdf", directory=output_path.parent)
    plt.close(fig)
    return output_path


# =============================================================================
# (j) Potential deviation plot
# =============================================================================
def plot_potential_deviation(output_path: Path | None = None) -> Path:
    """Plot V(r)/V_N(r) vs r for several Lambda values."""
    if output_path is None:
        output_path = FIGURES_DIR / "lt3d_potential_deviation.pdf"

    init_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    Lambda_values = [1e-3, 1e-2, 1e-1, 1.0]
    colors = ['#1565C0', '#C62828', '#2E7D32', '#E65100']

    for Lambda_eV, col in zip(Lambda_values, colors):
        lam1 = lambda_1(Lambda_eV)
        # Plot from 0.01*lambda_1 to 20*lambda_1
        r_min = 0.01 * lam1
        r_max = 20 * lam1
        r = np.logspace(np.log10(r_min), np.log10(r_max), 300)
        vr = V_ratio(r, Lambda_eV, xi=0.0)
        ax.plot(r / lam1, vr, color=col, lw=1.5,
                label=rf'$\Lambda = {Lambda_eV:.0e}$ eV')

    ax.axhline(y=1, color='gray', ls=':', lw=0.5, label=r'$V/V_N = 1$ (Newton)')
    ax.axhline(y=0, color='gray', ls='--', lw=0.5, alpha=0.5)

    ax.set_xlabel(r'$r / \lambda_1$')
    ax.set_ylabel(r'$V(r) / V_N(r)$')
    ax.set_title(r'Modified Newtonian potential ($\xi = 0$)')
    ax.set_xscale('log')
    ax.set_xlim(0.01, 20)
    ax.set_ylim(-0.5, 1.2)
    ax.legend(fontsize=6, loc='lower right')

    fig.tight_layout()
    save_figure(fig, output_path.stem, fmt="pdf", directory=output_path.parent)
    plt.close(fig)
    return output_path


# =============================================================================
# (k) Casimir correction plot
# =============================================================================
def plot_casimir_correction(output_path: Path | None = None) -> Path:
    """Plot delta_F/F_Casimir vs plate separation for several Lambda."""
    if output_path is None:
        output_path = FIGURES_DIR / "lt3d_casimir_correction.pdf"

    init_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    Lambda_values = [0.1, 1.0, 10.0]
    colors = ['#1565C0', '#C62828', '#2E7D32']

    for Lambda_eV, col in zip(Lambda_values, colors):
        lam1 = lambda_1(Lambda_eV)
        r_vals = np.logspace(-8, -4, 200)  # 10 nm to 100 um
        corrections = []
        for r in r_vals:
            corr = casimir_yukawa_correction(r, ALPHA_1, lam1)
            corrections.append(abs(corr))
        ax.plot(r_vals * 1e6, corrections, color=col, lw=1.5,
                label=rf'$\Lambda = {Lambda_eV}$ eV')

    # Chen 2016 precision level (~1% at 0.1 um)
    ax.axhline(y=0.01, color='gray', ls='--', lw=0.8, alpha=0.7,
               label='1% precision (Chen 2016)')

    ax.set_xlabel(r'Plate separation ($\mu$m)')
    ax.set_ylabel(r'$|\delta F / F_{\rm Casimir}|$')
    ax.set_title('SCT correction to Casimir force')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-2, 100)
    ax.set_ylim(1e-30, 10)
    ax.legend(fontsize=6, loc='upper right')

    fig.tight_layout()
    save_figure(fig, output_path.stem, fmt="pdf", directory=output_path.parent)
    plt.close(fig)
    return output_path


# =============================================================================
# Results export
# =============================================================================
def generate_results(xi: float = 0.0) -> dict[str, Any]:
    """Generate comprehensive LT-3d results dictionary."""
    # Primary bound: Eot-Wash
    eotwash = Lambda_min_eotwash(xi)

    # Casimir bound
    casimir = Lambda_min_casimir(xi)

    # xi-dependence
    xi_scan = Lambda_min_vs_xi()

    # Null experiments
    llr = llr_bound(eotwash["Lambda_min_eV"])
    gpb = gpb_bound(eotwash["Lambda_min_eV"])
    microscope = microscope_bound()
    atom = atom_interferometry_bound()

    # Overall Lambda_min (same as Eot-Wash — it is the tightest)
    Lambda_min_overall = eotwash["Lambda_min_eV"]

    # Mass values at the boundary
    m2_at_boundary = M2_OVER_LAMBDA * Lambda_min_overall
    m0_ratio = m0_over_Lambda(xi)
    m0_at_boundary = (
        m0_ratio * Lambda_min_overall if m0_ratio is not None else None
    )

    return {
        "task": "LT-3d (Laboratory and Solar System Tests)",
        "xi": xi,
        "Lambda_min_overall_eV": Lambda_min_overall,
        "Lambda_min_eotwash_eV": eotwash["Lambda_min_eV"],
        "Lambda_min_casimir_eV": casimir.get("Lambda_min_eV"),
        "dominant_experiment": "Eot-Wash (Lee+ 2020)",
        "lambda_1_at_boundary_um": eotwash["spin2_lambda_cross_um"],
        "m2_at_boundary_eV": m2_at_boundary,
        "m0_at_boundary_eV": m0_at_boundary,
        "m2_over_Lambda": M2_OVER_LAMBDA,
        "m0_over_Lambda_xi0": math.sqrt(6),
        "mass_ratio_m2_over_m0_xi0": math.sqrt(10 / 13),
        "alpha_1": ALPHA_1,
        "alpha_2": ALPHA_2,
        "alpha_1_over_alpha_2": ALPHA_1 / ALPHA_2,
        "potential_V0_over_VN": V_ratio_limit_r0(xi),
        "potential_Vinf_over_VN": 1.0,
        "llr_correction": llr["correction"],
        "gpb_correction": gpb["correction"],
        "microscope_constraint": "none (universal coupling)",
        "xi_scan": xi_scan,
        "eotwash_details": eotwash,
        "casimir_details": casimir,
        "llr_details": llr,
        "gpb_details": gpb,
        "microscope_details": microscope,
        "atom_interf_details": atom,
        "comparison_theories": competing_theories(),
    }


# =============================================================================
# Main entry point
# =============================================================================
def main() -> None:
    """Run all LT-3d laboratory computations, generate results and figures."""
    print("=" * 70)
    print("LT-3d: Laboratory and Solar System Bounds on SCT Lambda")
    print("=" * 70)

    # --- Primary bound ---
    xi = 0.0
    eotwash = Lambda_min_eotwash(xi)
    print(f"\nPrimary bound (Eot-Wash, xi={xi}):")
    print(f"  Lambda_min = {eotwash['Lambda_min_eV']:.4e} eV")
    print(f"  lambda_1 crossing at {eotwash['spin2_lambda_cross_um']:.2f} um")

    # --- Conformal coupling ---
    eotwash_conf = Lambda_min_eotwash(1 / 6)
    print("\nAt conformal coupling (xi=1/6):")
    print(f"  Lambda_min = {eotwash_conf['Lambda_min_eV']:.4e} eV")

    # --- Mass values ---
    print(f"\nm_2/Lambda = sqrt(60/13) = {M2_OVER_LAMBDA:.6f}")
    print(f"m_0/Lambda (xi=0) = sqrt(6) = {math.sqrt(6):.6f}")
    print(f"m_2/m_0 (xi=0) = sqrt(10/13) = {math.sqrt(10/13):.6f}")

    # --- V(r) limits ---
    print(f"\nV(r->0)/V_N(r->0) (xi=0) = {V_ratio_limit_r0(0.0):.6f}")
    print(f"V(r->0)/V_N(r->0) (xi=1/6) = {V_ratio_limit_r0(1/6):.6f}")

    # --- Null experiments ---
    llr = llr_bound(eotwash["Lambda_min_eV"])
    print(f"\nLLR: m_2*r_Moon = {llr['m2_times_r']:.2e} -> correction = {llr['correction']:.2e}")

    gpb = gpb_bound(eotwash["Lambda_min_eV"])
    print(f"GP-B: m_2*r_orbit = {gpb['m2_times_r']:.2e} -> correction = {gpb['correction']:.2e}")

    microscope = microscope_bound()
    print(f"MICROSCOPE: constrains SCT = {microscope['constrains_sct']}")

    atom = atom_interferometry_bound()
    print(f"Atom interf: constrains SCT = {atom['sabulsky_2019']['constrains_sct']}")

    # --- Casimir ---
    casimir = Lambda_min_casimir(xi)
    print(f"\nCasimir bound: {casimir.get('note', 'N/A')}")

    # --- xi scan ---
    print("\nLambda_min vs xi:")
    for xi_val in [0.0, 0.05, 0.1, 1 / 6, 0.2, 0.25]:
        b = Lambda_min_eotwash(xi_val)
        print(f"  xi = {xi_val:.4f}: Lambda_min = {b['Lambda_min_eV']:.4e} eV")

    # --- Generate results JSON ---
    results = generate_results(xi=0.0)
    results_path = RESULTS_DIR / "lt3d_laboratory_results.json"

    def _serializer(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    results_path.write_text(
        json.dumps(results, indent=2, default=_serializer),
        encoding="utf-8",
    )
    print(f"\nResults saved to {results_path}")

    # --- Generate figures ---
    print("\nGenerating figures...")
    p1 = plot_exclusion_unified(xi=0.0)
    print(f"  {p1}")
    p2 = plot_lambda_min_vs_xi()
    print(f"  {p2}")
    p3 = plot_potential_deviation()
    print(f"  {p3}")
    p4 = plot_casimir_correction()
    print(f"  {p4}")

    print("\n" + "=" * 70)
    print("LT-3d COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
